"""Surface equipment between the wellhead and the manifold: flowline pipes, fittings (equivalent length) and the
choke. Given the manifold pressure and the rate, the pressure is marched back upstream to the wellhead.

Choke models
------------
sachdeva    Sachdeva, Schmidt, Brill & Blais (1986): two-phase critical pressure ratio from a frozen-quality
            polytropic expansion, critical or subcritical mass flux with a discharge coefficient.
gilbert, ros, baxendell, achong
            Empirical critical-flow correlations  p1 (psig) = A R^B q / S^C  (R scf/STB of liquid, q STB/d liquid,
            S bean size in 64ths). Only valid in critical flow; subcritical points fall back to Sachdeva.

Units: psia, degF, ft, in; rates as in the VLP engine (oil: liquid STB/d; gas: MMscf/d).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import vlp as V
from .equipment import FITTINGS, HeatCapacities, surface_rows, _f

G = 32.174
CHOKE_METHODS = ["Sachdeva", "Gilbert", "Ros", "Baxendell", "Achong"]
GILBERT_TYPE = {"Gilbert": (10.00, 0.546, 1.89), "Ros": (17.40, 0.500, 2.00), "Baxendell": (9.56, 0.546, 1.93),
                "Achong": (3.82, 0.650, 1.88)}
CRIT_RATIO_EMPIRICAL = 0.588     # p2/p1 below which Gilbert-type correlations apply (critical flow)


@dataclass
class Item:
    kind: str            # "pipe" or "choke"
    label: str
    length: float = 0.0  # ft (pipe)
    d_in: float = 0.0    # in (pipe ID or bean size)
    rough_in: float = 0.0006
    rise: float = 0.0    # ft, elevation gain in the flow direction (pipe)
    mult: float = 1.0    # rate multiplier (fraction of the well rate through this item)


def build_items(df: pd.DataFrame) -> tuple[float, list[Item]]:
    """Surface table (manifold first, then back towards the wellhead) -> manifold TVD and items in FLOW order
    (wellhead -> manifold). A pipe's 'TVD of End Point' is its end nearer the wellhead; its other end is the previous
    row's end (the manifold for the first pipe). Fittings become pipe of equivalent length with no elevation change."""
    rows = surface_rows(df)
    if not rows or rows[0]["Type"] != "Manifold":
        raise ValueError("The surface equipment must start with the Manifold.")
    tvd_prev = _f(rows[0]["TVD of End Point (ft)"])
    tvd_prev = 0.0 if np.isnan(tvd_prev) else tvd_prev
    man_tvd = tvd_prev
    out = []
    for r in rows[1:]:
        t = r["Type"]
        d = _f(r["Pipe Inside Diameter (in)"])
        m = _f(r.get("Rate Multiplier"))
        m = 1.0 if np.isnan(m) or m <= 0 else m
        lab = r["Label"] if isinstance(r["Label"], str) and r["Label"].strip() else (r.get("Fitting") or t)
        if t == "Line Pipe":
            tvd_end = _f(r["TVD of End Point (ft)"])
            e = _f(r["Pipe Inside Roughness (in)"])
            out.append(Item("pipe", lab, _f(r["Pipe Length (ft)"]), d, 0.0006 if np.isnan(e) else e,
                            rise=tvd_end - tvd_prev, mult=m))
            tvd_prev = tvd_end
        elif t == "Fitting":
            ld = FITTINGS.get(r.get("Fitting"), 0.0)
            out.append(Item("pipe", lab, ld * d / 12, d, 0.0006, 0.0, mult=m))
        elif t == "Choke":
            out.append(Item("choke", lab, d_in=d, mult=m))
    return man_tvd, out[::-1]       # flow order: wellhead -> manifold


def has_choke(items):
    return any(i.kind == "choke" for i in items)


def with_choke_size(items, size_64ths):
    """Copy of the items with every choke bean set to size_64ths."""
    return [Item(i.kind, i.label, i.length, size_64ths / 64.0 if i.kind == "choke" else i.d_in, i.rough_in, i.rise,
                 i.mult) for i in items]


# ------------------------------------------------------------------ chokes
def _sachdeva_flux(fluid, case, p1, p2, t, hc, cd, k=1.26):
    """Mass flux (lbm/ft2/s), critical ratio and regime for upstream p1, downstream p2."""
    s = fluid.state(p1, t, case)
    ml, mg = s["ql"] * s["rho_l"], s["qg"] * np.maximum(s["rho_g"], 1e-4)
    x1 = np.clip(mg / np.maximum(ml + mg, 1e-15), 0.0, 1.0)
    vl = 1 / np.maximum(s["rho_l"], 1e-3)
    vg1 = 1 / np.maximum(s["rho_g"], 1e-4)
    cpg = hc.gas
    cvg = cpg / k
    cl = hc.oil if fluid.kind == "oil" else hc.water
    n = 1 + x1 * (cpg - cvg) / (x1 * cvg + (1 - x1) * cl)
    kk = k / (k - 1)
    gas = x1 > 1e-6
    xs = np.maximum(x1, 1e-6)
    def resid(y):                                     # y - RHS(y): negative at y->0, positive at y=1
        vg2 = vg1 * y ** (-1 / n)
        a = (1 - xs) * vl / (xs * vg2)
        num = kk + (1 - xs) * vl * (1 - y) / (xs * vg1)
        den = kk + n / 2 + n * a + n / 2 * a * a
        return y - (num / den) ** kk
    shape = np.broadcast(p1, x1).shape
    lo, hi = np.full(shape, 1e-4), np.full(shape, 0.9999)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        pos = resid(mid) > 0
        hi, lo = np.where(pos, mid, hi), np.where(pos, lo, mid)
    y = 0.5 * (lo + hi)
    yc = np.where(gas, y, 0.0)                        # all-liquid flow is never critical
    ya = np.maximum(p2 / p1, yc)
    vg2 = vg1 * ya ** (-1 / n)
    rho_m2 = 1 / (x1 * vg2 + (1 - x1) * vl)
    term = (1 - x1) * (1 - ya) / np.maximum(s["rho_l"], 1e-3) + x1 * kk * (vg1 - ya * vg2)
    flux = cd * np.sqrt(np.maximum(2 * G * 144.0 * p1 * rho_m2 ** 2 * term, 0.0))
    return flux, yc, ya <= yc + 1e-12


def choke_upstream(fluid, case, p2, t, d_in, hc, cd=0.75, method="Sachdeva", kind="oil"):
    """Upstream pressure (psia) for the given downstream pressure, rates (case) and bean ID. Vectorised.
    Returns (p1, regime array of str, note)."""
    p2 = np.asarray(p2, float)
    w, _ = fluid.mass_cp(case, hc)                    # lb/h
    w = w / 3600.0
    area = np.pi / 4 * (d_in / 12) ** 2
    need = w / area                                   # required mass flux
    lo, hi = p2 * (1 + 1e-9), p2 + 30000.0
    lo, hi = np.log(lo), np.log(hi)
    for _ in range(60):                               # flux increases with p1
        mid = 0.5 * (lo + hi)
        f, _, _ = _sachdeva_flux(fluid, case, np.exp(mid), p2, t, hc, cd)
        big = f > need
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    p1 = np.exp(0.5 * (lo + hi))
    _, yc, crit = _sachdeva_flux(fluid, case, p1, p2, t, hc, cd)
    regime = np.where(crit, "Critical", "SubCritical").astype(object)
    note = ""
    if method in GILBERT_TYPE:
        if kind != "oil" or "gor" not in case:
            note = f"{method} needs an oil well with a producing GOR - Sachdeva used."
        else:
            a, b, c = GILBERT_TYPE[method]
            q = np.asarray(case["q"], float)
            glr = np.maximum(case["gor"] * (1 - np.clip(case["wc"], 0, 0.9999)), 1.0)
            p1g = a * glr ** b * q / (d_in * 64) ** c + 14.7
            ok = p2 / p1g <= CRIT_RATIO_EMPIRICAL
            p1 = np.where(ok, p1g, p1)
            regime = np.where(ok, "Critical", "SubCritical (Sachdeva)").astype(object)
            if not ok.all():
                note = f"{method} is a critical-flow correlation; subcritical points use Sachdeva."
    return p1, regime, note


# ---------------------------------------------------------------- network
def _pipe_path(item, n_seg):
    L = item.length / n_seg
    d = item.d_in / 12
    sin_t = np.clip(item.rise / item.length, -1, 1) if item.length > 0 else 0.0
    md = np.linspace(0, item.length, n_seg + 1)
    return V.FlowPath(md=md, tvd=-sin_t * md, t_form=np.zeros(n_seg + 1), seg_len=np.full(n_seg, L),
                      sin_t=np.full(n_seg, sin_t), area=np.full(n_seg, np.pi / 4 * d * d), d_h=np.full(n_seg, d),
                      rough=np.full(n_seg, item.rough_in / 12), d_ht=np.full(n_seg, d))


def item_temperatures(items, fluid, case, t_wh, t_surr, u_surf, hc):
    """Fluid temperature entering each item (flow order) and at the manifold. Pipes relax towards the surrounding
    temperature (U referred to the pipe ID); chokes are taken as isothermal."""
    w, cp = fluid.mass_cp(case, hc)                   # lb/h
    t = np.asarray(t_wh, float)
    temps = []                                        # (inlet, outlet) per item
    for it in items:
        t0 = t
        if it.kind == "pipe" and it.length > 0:
            a = u_surf * np.pi * (it.d_in / 12) / np.maximum(w * cp, 1e-9)
            t = t_surr + (t - t_surr) * np.exp(-a * it.length)
        temps.append((t0, t))
    return temps, t


def wellhead_pressure(items, fluid, kind, p_manifold, case, t_wh, correlation, hc, t_surr=60.0, u_surf=8.0,
                      choke_method="Sachdeva", cd=0.75, p1=1.0, p2=1.0, detail=False):
    """March from the manifold back to the wellhead. All arguments broadcast over rates/cases.
    Returns wellhead pressure (and, with detail=True, a per-item breakdown list)."""
    case = V._broadcast_case({**case, "_p": p_manifold, "_t": t_wh})
    p, t_wh = case.pop("_p"), case.pop("_t")
    temps, t_man = item_temperatures(items, fluid, case, t_wh, t_surr, u_surf, hc)
    corr = V.CORRELATIONS[correlation]
    rows = []
    notes = set()
    for it, (t_in, t_out) in zip(items[::-1], temps[::-1]):      # backwards: manifold -> wellhead
        p_down = p.copy()
        t_avg = 0.5 * (t_in + t_out)
        c_it = case if it.mult == 1.0 else {**case, "q": case["q"] * it.mult}
        if it.kind == "choke":
            p, regime, note = choke_upstream(fluid, c_it, p, t_in, it.d_in, hc, cd, choke_method, kind)
            if note:
                notes.add(note)
            size = f"{it.d_in * 64:.0f}/64 in" if np.ndim(it.d_in) == 0 else "per test"
            rows.append({"Item": it.label, "Type": "Choke", "Size": size,
                         "Upstream pressure (psia)": p, "Downstream pressure (psia)": p_down, "dP (psi)": p - p_down,
                         "Temperature (°F)": t_in, "Regime": regime})
        else:
            n_seg = max(1, int(np.ceil(it.length / 500.0)))
            path = _pipe_path(it, n_seg)
            for i in range(n_seg - 1, -1, -1):           # march upstream within the pipe
                L = path.seg_len[i]
                g0, _ = V._gradient(path, fluid, i, p, t_avg, c_it, corr, p1, p2)
                pu = p + g0 * L
                g1, _ = V._gradient(path, fluid, i, pu, t_avg, c_it, corr, p1, p2)
                p = np.maximum(p + 0.5 * (g0 + g1) * L, 14.7)
            rows.append({"Item": it.label, "Type": "Pipe", "Size": f"{it.d_in:g} in ID, {it.length:,.0f} ft",
                         "Upstream pressure (psia)": p, "Downstream pressure (psia)": p_down, "dP (psi)": p - p_down,
                         "Temperature (°F)": t_avg, "Regime": ""})
    if detail:
        return p, rows[::-1], t_man, sorted(notes)
    return p


# ------------------------------------------------------- forward march (wellhead -> manifold)
def choke_downstream(fluid, case, p1, t, d_in, hc, cd=0.75, method="Sachdeva", kind="oil"):
    """Downstream pressure (psia) for the given upstream pressure and rates. Where the bean cannot pass the rate at
    this upstream pressure even in critical flow, the result is NaN (the inflow curve at the top node drops
    vertically there: in critical flow any downstream pressure gives the same rate). Returns (p2, regime)."""
    p1 = np.asarray(p1, float)
    w, _ = fluid.mass_cp(case, hc)
    need = w / 3600.0 / (np.pi / 4 * (d_in / 12) ** 2)
    _, yc, _ = _sachdeva_flux(fluid, case, p1, np.full_like(p1, 1e-6), t, hc, cd)
    # The Sachdeva critical ratio marks the flux maximum only approximately: locate the peak on a grid of ratios
    ys = np.linspace(0.0, 1.0, 41)[:-1]
    y0 = np.maximum(yc, 14.7 / p1)
    best_f, best_y = np.full_like(p1, -1.0), y0.copy()
    for z in ys:
        y = y0 + z * (1 - y0)
        f, _, _ = _sachdeva_flux(fluid, case, p1, y * p1, t, hc, cd)
        up = f > best_f
        best_f, best_y = np.where(up, f, best_f), np.where(up, y, best_y)
    ok = best_f >= need * (1 - 2e-3)                  # tolerance for the round trip of the backward solve
    lo, hi = best_y * p1, p1.copy()
    for _ in range(50):                               # flux falls as p2 rises above the peak
        mid = 0.5 * (lo + hi)
        f, _, _ = _sachdeva_flux(fluid, case, p1, mid, t, hc, cd)
        big = f > need
        lo, hi = np.where(big, mid, lo), np.where(big, hi, mid)
    p2 = np.where(ok, 0.5 * (lo + hi), np.nan)
    if method in GILBERT_TYPE and kind == "oil" and "gor" in case:
        a, b, c = GILBERT_TYPE[method]
        glr = np.maximum(case["gor"] * (1 - np.clip(case["wc"], 0, 0.9999)), 1.0)
        p1g = a * glr ** b * np.asarray(case["q"], float) / (d_in * 64) ** c + 14.7
        okg = p1 >= p1g * (1 - 1e-9)
        p2 = np.where(okg, np.fmax(p2, CRIT_RATIO_EMPIRICAL * p1g), np.nan)
    regime = np.where(np.isnan(p2), "Critical", "SubCritical").astype(object)
    return p2, regime


def manifold_pressure(items, fluid, kind, p_wellhead, case, t_wh, correlation, hc, t_surr=60.0, u_surf=8.0,
                      choke_method="Sachdeva", cd=0.75, p1=1.0, p2=1.0):
    """March downstream from the wellhead to the manifold. NaN where the surface equipment cannot pass the rate at
    that wellhead pressure (critical choke or pressure exhausted)."""
    case = V._broadcast_case({**case, "_p": p_wellhead, "_t": t_wh})
    p, t_wh = case.pop("_p"), case.pop("_t")
    temps, _ = item_temperatures(items, fluid, case, t_wh, t_surr, u_surf, hc)
    corr = V.CORRELATIONS[correlation]
    for it, (t_in, t_out) in zip(items, temps):
        c_it = case if it.mult == 1.0 else {**case, "q": case["q"] * it.mult}
        bad = ~np.isfinite(p)
        p = np.where(bad, 1000.0, p)                  # placeholder so the property calls stay finite
        if it.kind == "choke":
            p, _ = choke_downstream(fluid, c_it, p, t_in, it.d_in, hc, cd, choke_method, kind)
        else:
            n_seg = max(1, int(np.ceil(it.length / 500.0)))
            path = _pipe_path(it, n_seg)
            t_avg = 0.5 * (t_in + t_out)
            for i in range(n_seg):
                L = path.seg_len[i]
                g0, _ = V._gradient(path, fluid, i, p, t_avg, c_it, corr, p1, p2)
                pd_ = np.maximum(p - g0 * L, 14.7)
                g1, _ = V._gradient(path, fluid, i, pd_, t_avg, c_it, corr, p1, p2)
                pn = p - 0.5 * (g0 + g1) * L
                p = np.where(pn < 14.7, np.nan, pn)
                bad |= ~np.isfinite(p)
                p = np.where(bad, 1000.0, p)
        p = np.where(bad | ~np.isfinite(p), np.nan, p)
    return p


# ------------------------------------------------------------------ match to tests
CD_DEFAULT = 0.75
CD_BOUNDS = (0.3, 1.5)
F_BOUNDS = (0.2, 5.0)


@dataclass
class SurfaceMatch:
    """Discharge coefficient (Sachdeva) and flowline friction multiplier fitted to wellhead pressures measured with
    the manifold pressure."""
    cd: float
    friction: float
    fitted: tuple
    rms_before: float
    rms_after: float
    table: pd.DataFrame
    notes: list = field(default_factory=list)


def _test_items(items, choke):
    return with_choke_size(items, float(choke)) if np.isfinite(choke) and choke > 0 and has_choke(items) else items


def _calc_thp(items, fluid, kind, tests, t_wh, correlation, hc, t_surr, u_surf, cd, friction, detail=False):
    """Wellhead pressure of each test from its manifold pressure, all tests in one vectorised march (the bean
    diameter becomes an array, one value per test)."""
    ch = tests["choke"].to_numpy(float)
    its = []
    for it in items:
        if it.kind == "choke":
            d = np.where(np.isfinite(ch) & (ch > 0), ch / 64.0, it.d_in)
            it = Item(it.kind, it.label, it.length, d, it.rough_in, it.rise, it.mult)
        its.append(it)
    res = wellhead_pressure(its, fluid, kind, tests["pman"].to_numpy(float), V.case_of(tests, kind),
                            np.asarray(t_wh, float), correlation, hc, t_surr, u_surf, "Sachdeva", cd, 1.0, friction,
                            detail=detail)
    if not detail:
        return res
    rows = [[{k: (v[j:j + 1] if isinstance(v, np.ndarray) and v.ndim else v) for k, v in r.items()} for r in res[1]]
            for j in range(len(tests))]
    for j, rr in enumerate(rows):
        for r in rr:
            if r["Type"] == "Choke":
                r["Size"] = f"{float(np.atleast_1d(its[[i.kind for i in its].index('choke')].d_in)[j]) * 64:.0f}/64 in"
    return res[0], rows


def match_surface(items, fluid, kind, tests: pd.DataFrame, t_wh, correlation="beggs_brill", hc=None, t_surr=60.0,
                  u_surf=8.0, cd0=CD_DEFAULT, f0=1.0, fit=("cd", "friction"), sigma_p=5.0) -> SurfaceMatch:
    """Regress the Sachdeva discharge coefficient and/or a flowline friction multiplier so that the wellhead
    pressure calculated back from each test's manifold pressure matches the measured tubing head pressure.
    tests: clean_tests() rows with a manifold pressure (pman); 'choke' (64ths) overrides the bean per test.
    A parameter the data cannot see is held: friction when the choke is critical in every test (the wellhead
    pressure then does not depend on anything downstream of the bean), Cd when there is no choke."""
    hc = hc or HeatCapacities()
    tests = tests[np.isfinite(tests["pman"]) & np.isfinite(tests["thp"])].reset_index(drop=True)
    if not len(tests):
        raise ValueError("No test has both a manifold pressure and a tubing head pressure.")
    t_wh = np.asarray(t_wh, float)[:len(tests)] if len(np.atleast_1d(t_wh)) == len(tests) else \
        np.full(len(tests), float(np.atleast_1d(t_wh)[0]))
    thp = tests["thp"].to_numpy(float)
    notes = []
    fit = list(fit)
    before, rows0 = _calc_thp(items, fluid, kind, tests, t_wh, correlation, hc, t_surr, u_surf, cd0, f0, detail=True)
    if not has_choke(items) and "cd" in fit:
        fit.remove("cd")
        notes.append("No choke in the surface equipment - only the flowline friction is matched.")
    if "friction" in fit:
        crit = [any(r_["Type"] == "Choke" and str(r_["Regime"][0]).startswith("Critical") for r_ in rr)
                for rr in rows0]
        if has_choke(items) and all(crit):
            fit.remove("friction")
            notes.append("The choke is critical in every test, so the wellhead pressure does not depend on the "
                         "flowline: friction multiplier held at 1. Add tests with subcritical flow to match it.")
        elif len(tests) < 2 and "cd" in fit:
            fit.remove("friction")
            notes.append("One test cannot separate choke and flowline losses: only Cd is matched.")
    x0 = [cd0 if k == "cd" else f0 for k in fit]
    lo = [CD_BOUNDS[0] if k == "cd" else F_BOUNDS[0] for k in fit]
    hi = [CD_BOUNDS[1] if k == "cd" else F_BOUNDS[1] for k in fit]

    def unpack(x):
        d = dict(zip(fit, x))
        return d.get("cd", cd0), d.get("friction", f0)

    def resid(x):
        cd, f = unpack(x)
        r = (_calc_thp(items, fluid, kind, tests, t_wh, correlation, hc, t_surr, u_surf, cd, f) - thp) / sigma_p
        prior = [0.1 * (cd - cd0) / 0.25 if k == "cd" else 0.1 * (f - f0) / 0.5 for k in fit]   # weak pull
        return np.concatenate([r, prior])

    if fit:
        sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", diff_step=1e-3)
        cd, f = unpack(sol.x)
    else:
        cd, f = cd0, f0
        notes.append("Nothing to match.")
    after, rows1 = _calc_thp(items, fluid, kind, tests, t_wh, correlation, hc, t_surr, u_surf, cd, f, detail=True)
    for k, v, b in (("cd", cd, CD_BOUNDS), ("friction", f, F_BOUNDS)):
        if k in fit and (v <= b[0] * 1.001 or v >= b[1] * 0.999):
            notes.append(f"{'Cd' if k == 'cd' else 'Friction multiplier'} = {v:.3g} is at its bound - check the "
                         "test data, bean sizes and flowline description.")
    if "cd" in fit and not 0.5 <= cd <= 1.0:
        notes.append(f"Cd = {cd:.3f} is outside the usual 0.5-1.0 range.")
    recs = []
    for j, (_, r) in enumerate(tests.iterrows()):
        ch = next((x for x in rows1[j] if x["Type"] == "Choke"), None)
        pipe_dp = sum(float(x["dP (psi)"][0]) for x in rows1[j] if x["Type"] == "Pipe")
        recs.append({"Test": r["label"], "Choke (64ths)": float(ch["Size"].split("/")[0]) if ch else np.nan,
                     "Manifold pressure (psia)": r["pman"], "Measured THP (psia)": r["thp"],
                     "Calculated THP - before (psia)": before[j], "Calculated THP - matched (psia)": after[j],
                     "Error (psi)": after[j] - r["thp"],
                     "Choke regime": str(ch["Regime"][0]) if ch else "-",
                     "dP choke (psi)": float(ch["dP (psi)"][0]) if ch else 0.0, "dP flowline and fittings (psi)": pipe_dp})
    rms = lambda a: float(np.sqrt(np.mean((a - thp) ** 2)))
    return SurfaceMatch(cd, f, tuple(fit), rms(before), rms(after), pd.DataFrame(recs), notes)
