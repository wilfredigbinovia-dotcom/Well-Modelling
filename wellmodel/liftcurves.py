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
              alq=None):
    """Bottom-node pressure array shaped (n_thp, n_wfr, n_gfr, n_flo). thp is the top-node pressure (the manifold
    when surface equipment is given - a SurfaceSetup from system.py). esp: the installed pump (Equipment tab dict:
    pump_md, pump, stages, freq, wear, sep, motor) - the tables then include the pump (NaN where it pumps the well
    off). alq: ESP operating frequencies (Hz) as the artificial-lift axis; axes are then (thp, wfr, gfr, alq, flo)
    and the array is shaped (n_thp, n_wfr, n_gfr, n_alq, n_flo)."""
    hc = hc or HeatCapacities()
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
        out.append(f"{int(np.isnan(bhp).sum())} point(s) blank: at those rates" + (" and frequencies" if alq is not
                   None else "") + " the pump would draw the intake below atmospheric (the rate is below what the "
                   "pump delivers against the tubing). The Eclipse file fills them by extrapolating along the rate "
                   "axis; narrow the rate range if they matter.")
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


def vfp_csv(kind, axes, bhp, wh=None):
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
                            r["ESP frequency (Hz)"] = alq[l_]
                        r[f"{a['flo'][0]} ({a['flo'][1]})"] = f
                        r["Bottom node pressure (psia)"] = bhp[i, j, k, l_, m]
                        if wh is not None:
                            r["Wellhead pressure (psia)"] = wh[i, j, k, l_, m]
                        rows.append(r)
    return pd.DataFrame(rows)


def _line(vals, fmt="{:.6g}", per=6):
    s = [fmt.format(v) for v in vals]
    return "\n".join("  " + " ".join(s[i:i + per]) for i in range(0, len(s), per)) + " /"


def eclipse_vfpprod(kind, axes, bhp, table_no=1, datum_tvd=0.0, title=""):
    """Eclipse VFPPROD keyword, FIELD units. Oil: FLO=LIQ (STB/d), WFR=WCT (fraction), GFR=GOR (Mscf/STB).
    Gas: FLO=GAS (Mscf/d), WFR=WGR (STB/Mscf), GFR=OGR (STB/Mscf). THP and BHP in psia, datum in ft TVD.
    With an ESP frequency axis, ALQ is the frequency in Hz (ALQ type left undefined, ' '). Blank points (an ESP
    pumping off at low rates) are filled along the rate axis from the calculated ones, since Eclipse needs a value
    at every point."""
    (thp, wfr, gfr, alq, flo), bhp = _split(axes, bhp)
    thp, wfr, gfr, flo = (np.asarray(a, float) for a in (thp, wfr, gfr, flo))
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
           + (", no artificial lift (ALQ = 0)" if alq is None else f" x {len(alq)} ALQ = ESP frequency (Hz)"),
           "VFPPROD", "-- Table  Datum(ft)  FLO    WFR    GFR    THP    ALQ   Units    Tabulated",
           f"   {int(table_no)}  {datum_tvd:.2f}  {types[0]}  {types[1]}  {types[2]}  'THP'  ' '  'FIELD'  'BHP' /",
           "-- FLO values", _line(f_flo), "-- THP values (psia)", _line(thp),
           "-- WFR values", _line(f_wfr), "-- GFR values", _line(f_gfr), "-- ALQ values" + ("" if alq is None else " (ESP frequency, Hz)"),
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
