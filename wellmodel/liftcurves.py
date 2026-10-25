"""Gradient traverses and lift-curve (VFP) tables.

Lift curves: bottom-node pressure for every combination of rate, top-node pressure, water fraction and gas
fraction, in one vectorised traverse. With surface equipment the top node is the manifold and each point is first
worked back to the wellhead. Written as an Eclipse VFPPROD keyword (FIELD units) and as a flat CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import surface as SF
from . import vlp as V
from .equipment import HeatCapacities

# axis definitions per well kind: (name, unit) for FLO, WFR, GFR as entered in the tool
AXES = {"oil": {"flo": ("Liquid rate", "STB/d"), "wfr": ("Water cut", "%"), "gfr": ("Total GOR", "scf/STB")},
        "gas": {"flo": ("Gas rate", "MMscf/d"), "wfr": ("Water gas ratio", "STB/MMscf"),
                "gfr": ("Condensate gas ratio", "STB/MMscf")}}
MAX_POINTS = 20000
MAX_POINTS_HSP = 800       # each HSP point iterates the power fluid and the pump (1-2 s per point)


def _case(kind, flo, wfr, gfr):
    if kind == "oil":
        return {"q": flo, "wc": wfr / 100.0, "gor": gfr}
    return {"q": flo, "wgr": wfr, "cgr": gfr}


def _split(axes, bhp=None):
    """(thp, wfr, gfr, alq or None, flo) and the BHP array with an ALQ dimension (length 1 when there is none)."""
    if len(axes) == 5:
        thp, wfr, gfr, alq, flo = axes
        return (thp, wfr, gfr, alq, flo), (None if bhp is None else np.asarray(bhp, float))
    thp, wfr, gfr, flo = axes
    return (thp, wfr, gfr, None, flo), (None if bhp is None else np.asarray(bhp, float)[:, :, :, None, :])


def vfp_table(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc=None, surface=None, esp=None,
              alq=None, lift=None, jet=None, jet_alq="p_surf", pcp=None, hsp=None, srp=None, srp_p_max=None):
    """Bottom-node pressure array shaped (n_thp, n_wfr, n_gfr, n_flo). thp is the top-node pressure (the manifold
    when surface equipment is given - a SurfaceSetup from system.py). esp: the installed pump (Equipment tab dict:
    pump_md, pump, stages, freq, wear, sep, motor) - the tables then include the pump (NaN where it pumps the well
    off). alq: ESP operating frequencies (Hz) as the artificial-lift axis; axes are then (thp, wfr, gfr, alq, flo)
    and the array is shaped (n_thp, n_wfr, n_gfr, n_alq, n_flo)."""
    hc = hc or HeatCapacities()
    if srp is not None:
        p_max = srp_p_max or (0.5 * float(path.tvd[-1]) + float(np.max(thp)) + 500.0)
        return _vfp_srp(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc, surface, srp, alq, p_max)
    if hsp is not None:
        return _vfp_hsp(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc, surface, hsp, alq)
    if pcp is not None:
        return _vfp_pcp(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc, surface, pcp, alq)
    if jet is not None:
        return _vfp_jet(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc, surface, jet, alq, jet_alq)
    if lift is not None:
        return _vfp_gas_lift(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc, surface, lift, alq)
    if alq is not None and len(alq):
        if esp is None:
            raise ValueError("A frequency (ALQ) axis needs the ESP.")
        axes = [np.asarray(sorted(set(map(float, a))), float) for a in (thp, wfr, gfr, alq, flo)]
        n = int(np.prod([len(a) for a in axes]))
        if n > MAX_POINTS:
            raise ValueError(f"{n:,} points - reduce the number of values (limit {MAX_POINTS:,}).")
        T_, W_, G_, A_, F_ = (g.ravel() for g in np.meshgrid(*axes, indexing="ij"))
        case = _case(kind, F_, W_, G_)
        top = _top(path, fluid, kind, T_, case, hc, u, surface)
        from .esp import bhp_with_pump
        out = bhp_with_pump(path, fluid, top, case, esp, correlation, p1, p2, u, hc, freq=A_)
        shape = tuple(len(a) for a in axes)
        return axes, out["bhp"].reshape(shape), top.reshape(shape)
    axes = [np.asarray(sorted(set(map(float, a))), float) for a in (thp, wfr, gfr, flo)]
    n = int(np.prod([len(a) for a in axes]))
    if n > MAX_POINTS:
        raise ValueError(f"{n:,} points - reduce the number of values (limit {MAX_POINTS:,}).")
    if any(len(a) == 0 for a in axes):
        raise ValueError("Every axis needs at least one value.")
    T_, W_, G_, F_ = (g.ravel() for g in np.meshgrid(*axes, indexing="ij"))
    case = _case(kind, F_, W_, G_)
    top = _top(path, fluid, kind, T_, case, hc, u, surface)
    shape = tuple(len(a) for a in axes)
    if esp is not None:
        from .esp import bhp_with_pump
        out = bhp_with_pump(path, fluid, top, case, esp, correlation, p1, p2, u, hc)
        return axes, out["bhp"].reshape(shape), top.reshape(shape)
    with np.errstate(all="ignore"):
        P, _ = V.traverse(path, fluid, top, case, correlation, p1, p2, u, hc)
    return axes, P[:, -1].reshape(shape), top.reshape(shape)


def _vfp_gas_lift(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc, surface, lift, alq):
    """Gas-lifted lift curves (gaslift_system.GasLiftSetup). ALQ: lift-gas rate (MMscf/d) - or the casing head
    pressure (psia) for the multipoint method, where the gas rate is a result. No ALQ values: the lift setup's own
    gas rate (or casing pressure) for every point."""
    from . import gaslift_system as GLS
    if kind != "oil":
        raise ValueError("Gas-lifted lift curves are for oil wells.")
    mp = lift.multipoint
    al = [lift.p_casing if mp else lift.qgi] if alq is None or not len(alq) else alq
    axes = [np.asarray(sorted(set(map(float, a))), float) for a in (thp, wfr, gfr, al, flo)]
    n = int(np.prod([len(a) for a in axes]))
    if n > MAX_POINTS:
        raise ValueError(f"{n:,} points - reduce the number of values (limit {MAX_POINTS:,}).")
    if any(len(a) == 0 for a in axes):
        raise ValueError("Every axis needs at least one value.")
    T_, W_, G_, A_, F_ = (g.ravel() for g in np.meshgrid(*axes, indexing="ij"))
    case = _case(kind, F_, W_, G_)
    top = _top(path, fluid, kind, T_, case, hc, u, surface)
    cols = lift.row_inputs({}, F_)
    cols["_pcs" if mp else "_qgi"] = A_
    shape = tuple(len(a) for a in axes)
    with np.errstate(all="ignore"):
        st = GLS.lift_traverse(path, fluid, top, {**case, **cols}, lift, correlation, p1, p2, u, hc)
    bhp, top = st.P[:, -1].reshape(shape), top.reshape(shape)
    if alq is None or not len(alq):
        return [axes[i] for i in (0, 1, 2, 4)], bhp[:, :, :, 0, :], top[:, :, :, 0, :]
    return axes, bhp, top


def _vfp_jet(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc, surface, jet, alq, jet_alq):
    """Jet-pumped lift curves (jetpump.JetPumpInstalled). ALQ: the power fluid injection pressure (psia, jet_alq
    'p_surf') or rate (STB/d, 'q_pf'); none = the installed pump's. NaN where the pump cannot deliver."""
    from .jetpump import bhp_with_jet
    if kind != "oil":
        raise ValueError("Jet-pumped lift curves are for oil wells.")
    al = [getattr(jet, jet_alq)] if alq is None or not len(alq) else alq
    axes = [np.asarray(sorted(set(map(float, a))), float) for a in (thp, wfr, gfr, al, flo)]
    n = int(np.prod([len(a) for a in axes]))
    if n > MAX_POINTS:
        raise ValueError(f"{n:,} points - reduce the number of values (limit {MAX_POINTS:,}).")
    if any(len(a) == 0 for a in axes):
        raise ValueError("Every axis needs at least one value.")
    T_, W_, G_, A_, F_ = (g.ravel() for g in np.meshgrid(*axes, indexing="ij"))
    case = _case(kind, F_, W_, G_)
    top = _top(path, fluid, kind, T_, case, hc, u, surface)
    shape = tuple(len(a) for a in axes)
    out = bhp_with_jet(path, fluid, top, case, jet, correlation, p1, p2, u, hc,
                       **({"p_surf": A_} if jet_alq == "p_surf" else {"q_pf": A_}))
    bhp, top = out["bhp"].reshape(shape), top.reshape(shape)
    if alq is None or not len(alq):
        return [axes[i] for i in (0, 1, 2, 4)], bhp[:, :, :, 0, :], top[:, :, :, 0, :]
    return axes, bhp, top


def _vfp_pcp(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc, surface, pcp, alq):
    """Progressive-cavity-pumped lift curves (pcp.PCPInstalled). ALQ: pump speed (rpm); none = the installed pump's.
    NaN where the pump cannot pass the rate at that speed or would draw the intake below atmospheric."""
    from .pcp import bhp_with_pcp
    if kind != "oil":
        raise ValueError("PCP lift curves are for oil wells.")
    al = [pcp.speed] if alq is None or not len(alq) else alq
    axes = [np.asarray(sorted(set(map(float, a))), float) for a in (thp, wfr, gfr, al, flo)]
    n = int(np.prod([len(a) for a in axes]))
    if n > MAX_POINTS:
        raise ValueError(f"{n:,} points - reduce the number of values (limit {MAX_POINTS:,}).")
    if any(len(a) == 0 for a in axes):
        raise ValueError("Every axis needs at least one value.")
    T_, W_, G_, A_, F_ = (g.ravel() for g in np.meshgrid(*axes, indexing="ij"))
    case = _case(kind, F_, W_, G_)
    top = _top(path, fluid, kind, T_, case, hc, u, surface)
    shape = tuple(len(a) for a in axes)
    out = bhp_with_pcp(path, fluid, top, case, pcp, correlation, p1, p2, u, hc, speed=A_)
    bhp, top = out["bhp"].reshape(shape), top.reshape(shape)
    if alq is None or not len(alq):
        return [axes[i] for i in (0, 1, 2, 4)], bhp[:, :, :, 0, :], top[:, :, :, 0, :]
    return axes, bhp, top


def _vfp_hsp(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc, surface, hsp, alq):
    """Hydraulic-submersible-pumped lift curves (hsp.HSPInstalled): PROSPER's HSP traverse at every point - tubing
    from the top node to the pump with production + the power fluid whose turbine gives the pump power, the intake
    from the pump curve, casing to the bottom node. ALQ: pump speed (rpm); none = the installed pump's. NaN where
    the speed is outside the pump's range, the pump would draw the intake below atmospheric or the turbine cannot
    give the pump power."""
    import dataclasses
    from .hsp import traverse_profile
    if kind != "oil":
        raise ValueError("HSP lift curves are for oil wells.")
    al = [hsp.speed] if alq is None or not len(alq) else alq
    axes = [np.asarray(sorted(set(map(float, a))), float) for a in (thp, wfr, gfr, al, flo)]
    n = int(np.prod([len(a) for a in axes]))
    if n > MAX_POINTS_HSP:
        raise ValueError(f"{n:,} points - reduce the number of values (limit {MAX_POINTS_HSP:,} for an HSP).")
    if any(len(a) == 0 for a in axes):
        raise ValueError("Every axis needs at least one value.")
    T_, W_, G_, A_, F_ = (g.ravel() for g in np.meshgrid(*axes, indexing="ij"))
    case = _case(kind, F_, W_, G_)
    top = _top(path, fluid, kind, T_, case, hc, u, surface)
    pump = hsp.pump_obj
    bhp = np.full(n, np.nan)
    for i in range(n):
        if not pump.min_speed - 1e-6 <= A_[i] <= pump.max_speed + 1e-6:
            continue
        try:
            with np.errstate(all="ignore"):
                s, _ = traverse_profile(path, fluid, float(top[i]), float(F_[i]), float(case["wc"][i]),
                                        float(case["gor"][i]), dataclasses.replace(hsp, speed=float(A_[i])), None,
                                        correlation, p1, p2, u, hc)
            bhp[i] = s["Bottom node pressure (psia)"]
        except (ValueError, ZeroDivisionError, FloatingPointError):
            pass
    shape = tuple(len(a) for a in axes)
    bhp, top = bhp.reshape(shape), np.asarray(top, float).reshape(shape)
    if alq is None or not len(alq):
        return [axes[i] for i in (0, 1, 2, 4)], bhp[:, :, :, 0, :], top[:, :, :, 0, :]
    return axes, bhp, top


def srp_rate_curve(path, fluid, srp_inp, top_p, wc, gor, pwf, correlation="no_slip", p1=1.0, p2=1.0, u=8.0, hc=None):
    """PROSPER's sucker rod lift curve: liquid rate (STB/d) against flowing bottom-hole pressure pwf (psia, array)."""
    from .srp import rate_for_pwf
    return np.array([rate_for_pwf(srp_inp, path, fluid, float(p), wc, gor, top_p, correlation, p1, p2, u, hc)[0]
                     for p in np.atleast_1d(pwf)])


def _vfp_srp(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc, surface, srp_inp, alq, p_max):
    """Rod-pumped lift curves. PROSPER tabulates the rate the pump delivers against the bottom-hole pressure; the
    VFP table (BHP against rate) is its inverse: at each rate the bottom-hole pressure at which the pump delivers it
    (interpolated on a grid of 24 bottom-hole pressures up to p_max). ALQ: pumping speed (strokes/min). NaN for rates
    beyond what the pump delivers at p_max."""
    import dataclasses
    if kind != "oil":
        raise ValueError("Rod pump lift curves are for oil wells.")
    al = [srp_inp.n_spm] if alq is None or not len(alq) else alq
    axes = [np.asarray(sorted(set(map(float, a))), float) for a in (thp, wfr, gfr, al, flo)]
    n_curves = int(np.prod([len(a) for a in axes[:4]]))
    if n_curves * 24 > MAX_POINTS_HSP:
        raise ValueError(f"{n_curves:,} curves - reduce the number of values (limit {MAX_POINTS_HSP // 24} curves "
                         "for a rod pump: 24 bottom-hole pressures each).")
    if any(len(a) == 0 for a in axes):
        raise ValueError("Every axis needs at least one value.")
    pw = np.linspace(14.7, float(p_max), 24)
    shape = tuple(len(a) for a in axes)
    bhp = np.full(shape, np.nan)
    top = np.full(shape, np.nan)
    for i, t in enumerate(axes[0]):
        for j, w in enumerate(axes[1]):
            for k, g in enumerate(axes[2]):
                case = _case(kind, axes[4], np.full(len(axes[4]), w), np.full(len(axes[4]), g))
                tp = _top(path, fluid, kind, np.full(len(axes[4]), t), case, hc, u, surface)
                for a, n in enumerate(axes[3]):
                    tp0 = float(np.nanmean(tp))
                    q = srp_rate_curve(path, fluid, dataclasses.replace(srp_inp, n_spm=float(n)), tp0, w / 100.0, g,
                                       pw, correlation, p1, p2, u, hc)
                    ok = np.isfinite(q)
                    if ok.sum() >= 2:
                        qq, pp = np.maximum.accumulate(q[ok]), pw[ok]
                        bhp[i, j, k, a] = np.where(axes[4] <= qq[-1], np.interp(axes[4], qq, pp), np.nan)
                    top[i, j, k, a] = tp
    if alq is None or not len(alq):
        return [axes[i] for i in (0, 1, 2, 4)], bhp[:, :, :, 0, :], top[:, :, :, 0, :]
    return axes, bhp, top


def _top(path, fluid, kind, T_, case, hc, u, surface):
    """Wellhead pressure for each point: the top-node pressure worked back through the surface equipment."""
    top = np.asarray(T_, float).copy()
    if surface is not None and surface.items:
        w, cp = fluid.mass_cp(case, hc)
        t_wh = V.temperature_profile(path, w, cp, u)[:, 0]
        with np.errstate(all="ignore"):
            top = SF.wellhead_pressure(surface.items, fluid, kind, T_, case, t_wh, surface.correlation, hc,
                                       surface.t_surround, surface.u, surface.choke_method, surface.cd,
                                       p2=surface.friction)
    return top


def vfp_checks(axes, bhp, esp=False):
    """Plain-language warnings about the table. esp: the table includes a pump (BHP below THP is then normal)."""
    (thp, wfr, gfr, alq, flo), bhp = _split(axes, bhp)
    out = []
    if esp and np.isnan(bhp).any():
        out.append(f"{int(np.isnan(bhp).sum())} point(s) blank: at those rates" + (" and ALQ values" if alq is not
                   None else "") + " the pump cannot work (it would draw the intake below atmospheric, the rate is "
                   "outside what the pump or its drive can deliver, or the speed is outside the pump's range). The "
                   "Eclipse file fills them by extrapolating along the rate axis; narrow the rate range if they "
                   "matter.")
        bhp = np.where(np.isnan(bhp), np.nanmax(bhp), bhp)
    if not np.all(np.isfinite(bhp)):
        out.append(f"{int((~np.isfinite(bhp)).sum())} point(s) could not be calculated.")
    huge = int((bhp > 30000).sum())
    if huge:
        out.append(f"{huge} point(s) above 30,000 psia - the rate range is beyond what this well can deliver; "
                   "lower the last rate.")
    if len(thp) > 1:
        d = np.diff(bhp, axis=0)
        bad = int((d < -0.5).sum())
        if bad:
            out.append(f"{bad} place(s) where BHP falls as THP rises (largest drop {-np.nanmin(d):.1f} psi) - "
                       "usually a flow-pattern boundary in the correlation; the simulator interpolates across it.")
    below = int((bhp < thp[:, None, None, None, None] - 0.5).sum())
    if below and not esp:
        out.append(f"{below} point(s) with BHP below THP - not physical for a producer; check the inputs.")
    return out


def vfp_csv(kind, axes, bhp, wh=None, alq_label="ESP frequency (Hz)"):
    (thp, wfr, gfr, alq, flo), bhp = _split(axes, bhp)
    if wh is not None:
        wh = np.asarray(wh, float) if alq is not None else np.asarray(wh, float)[:, :, :, None, :]
    a = AXES[kind]
    rows = []
    for i, t in enumerate(thp):
        for j, w in enumerate(wfr):
            for k, g in enumerate(gfr):
                for l_ in range(1 if alq is None else len(alq)):
                    for m, f in enumerate(flo):
                        r = {"Top node pressure (psia)": t, f"{a['wfr'][0]} ({a['wfr'][1]})": w,
                             f"{a['gfr'][0]} ({a['gfr'][1]})": g}
                        if alq is not None:
                            r[alq_label] = alq[l_]
                        r[f"{a['flo'][0]} ({a['flo'][1]})"] = f
                        r["Bottom node pressure (psia)"] = bhp[i, j, k, l_, m]
                        if wh is not None:
                            r["Wellhead pressure (psia)"] = wh[i, j, k, l_, m]
                        rows.append(r)
    return pd.DataFrame(rows)


def _line(vals, fmt="{:.6g}", per=6):
    s = [fmt.format(v) for v in vals]
    return "\n".join("  " + " ".join(s[i:i + per]) for i in range(0, len(s), per)) + " /"


ALQ_NAMES = {"esp": ("ESP frequency (Hz)", "' '"), "gas_lift": ("lift gas rate (Mscf/d)", "'GRAT'"),
             "multipoint": ("casing head pressure (psia)", "' '"),
             "jet_p_surf": ("jet pump injection pressure (psia)", "' '"),
             "jet_q_pf": ("jet pump power fluid rate (STB/d)", "' '"),
             "pcp_speed": ("PCP speed (rpm)", "' '"), "hsp_speed": ("HSP pump speed (rpm)", "' '"),
             "srp_speed": ("rod pump speed (strokes/min)", "' '")}


def eclipse_vfpprod(kind, axes, bhp, table_no=1, datum_tvd=0.0, title="", alq_kind="esp"):
    """Eclipse VFPPROD keyword, FIELD units. Oil: FLO=LIQ (STB/d), WFR=WCT (fraction), GFR=GOR (Mscf/STB).
    Gas: FLO=GAS (Mscf/d), WFR=WGR (STB/Mscf), GFR=OGR (STB/Mscf). THP and BHP in psia, datum in ft TVD.
    With an ESP frequency axis, ALQ is the frequency in Hz (ALQ type left undefined, ' '). Blank points (an ESP
    pumping off at low rates) are filled along the rate axis from the calculated ones, since Eclipse needs a value
    at every point."""
    (thp, wfr, gfr, alq, flo), bhp = _split(axes, bhp)
    thp, wfr, gfr, flo = (np.asarray(a, float) for a in (thp, wfr, gfr, flo))
    alq_name, alq_type = ALQ_NAMES[alq_kind]
    if alq is not None and alq_kind == "gas_lift":
        alq = np.asarray(alq, float) * 1000.0                    # MMscf/d -> Mscf/d
    if np.isnan(bhp).any():
        bhp = _fill_rates(bhp, flo)
    if kind == "oil":
        f_flo, f_wfr, f_gfr = flo, wfr / 100.0, gfr / 1000.0
        types = ("'LIQ'", "'WCT'", "'GOR'")
    else:
        f_flo, f_wfr, f_gfr = flo * 1000.0, wfr / 1000.0, gfr / 1000.0
        types = ("'GAS'", "'WGR'", "'OGR'")
    out = ["-- Generated by wellmodel" + (f": {title}" if title else ""),
           f"-- {len(flo)} rates x {len(thp)} THP x {len(wfr)} WFR x {len(gfr)} GFR"
           + (", no artificial lift (ALQ = 0)" if alq is None else f" x {len(alq)} ALQ = {alq_name}"),
           "VFPPROD", "-- Table  Datum(ft)  FLO    WFR    GFR    THP    ALQ   Units    Tabulated",
           f"   {int(table_no)}  {datum_tvd:.2f}  {types[0]}  {types[1]}  {types[2]}  'THP'  "
           f"{alq_type if alq is not None else chr(39) + ' ' + chr(39)}  'FIELD'  'BHP' /",
           "-- FLO values", _line(f_flo), "-- THP values (psia)", _line(thp),
           "-- WFR values", _line(f_wfr), "-- GFR values", _line(f_gfr), "-- ALQ values" + ("" if alq is None else f" ({alq_name})"),
           "  0 /" if alq is None else _line(np.asarray(alq, float)),
           "-- NTHP NWFR NGFR NALQ  BHP (psia) for each FLO value"]
    for i in range(len(thp)):
        for j in range(len(wfr)):
            for k in range(len(gfr)):
                for l_ in range(bhp.shape[3]):
                    vals = [f"{v:.2f}" for v in bhp[i, j, k, l_]]
                    lines = [" ".join(vals[x:x + 8]) for x in range(0, len(vals), 8)]   # Eclipse: <= 132 chars
                    out.append(f"  {i + 1} {j + 1} {k + 1} {l_ + 1}  " + "\n    ".join(lines) + " /")
    return "\n".join(out) + "\n"


def _fill_rates(bhp, flo):
    """Fill blank (NaN) points along the rate axis by linear extrapolation from the calculated ones, so the
    simulator gets a complete table; rows with fewer than two points take the row's value or the THP row above."""
    out = bhp.copy()
    it = np.ndindex(*bhp.shape[:-1])
    for idx in it:
        y = bhp[idx]
        ok = np.isfinite(y)
        if ok.all():
            continue
        if ok.sum() >= 2:
            xs, ys = flo[ok], y[ok]
            lo = np.polyfit(xs[:2], ys[:2], 1)
            hi = np.polyfit(xs[-2:], ys[-2:], 1)
            fill = np.where(flo < xs[0], np.polyval(lo, flo), np.where(flo > xs[-1], np.polyval(hi, flo),
                                                                        np.interp(flo, xs, ys)))
            out[idx] = np.where(ok, y, fill)
        elif ok.sum() == 1:
            out[idx] = np.where(ok, y, y[ok][0])
        else:
            out[idx] = np.nanmax(bhp)
    return out


def read_tpd(text: str) -> dict:
    """PROSPER lift-curve export (.Tpd, version 2): rate values, the sensitivity variables' values (as PROSPER lists
    them: variable 3, 2, 1) and the results array shaped (n_var1, n_var2, n_var3, n_rates, n_columns); columns are
    PROSPER's result codes (5000 bottom-hole pressure, 5001 wellhead temperature, ...)."""
    rows = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    i = rows.index("TPDData")
    r = rows[i + 1:]
    n_sens = int(r[4])
    ncol = int(r[6])
    cols = [int(x) for x in r[7].split(",")]
    types = [int(x) for x in r[8].split(",")]
    vals = [np.array([float(x) for x in r[9 + k].split(",")]) for k in range(n_sens + 1)]
    res = np.array([[float(x) for x in ln.split(",")] for ln in r[10 + n_sens:]])
    shape = tuple(len(v) for v in vals[::-1]) + (ncol,)
    return {"rates": vals[0], "variables": vals[1:], "types": types, "columns": cols, "results": res.reshape(shape),
            "first_node": [float(x) for x in r[2].split(",")]}

