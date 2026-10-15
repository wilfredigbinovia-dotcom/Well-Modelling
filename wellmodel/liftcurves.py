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


def vfp_table(path, fluid, kind, correlation, p1, p2, u, flo, thp, wfr, gfr, hc=None, surface=None):
    """Bottom-node pressure array shaped (n_thp, n_wfr, n_gfr, n_flo). thp is the top-node pressure (the manifold
    when surface equipment is given - a SurfaceSetup from system.py)."""
    hc = hc or HeatCapacities()
    axes = [np.asarray(sorted(set(map(float, a))), float) for a in (thp, wfr, gfr, flo)]
    n = int(np.prod([len(a) for a in axes]))
    if n > MAX_POINTS:
        raise ValueError(f"{n:,} points - reduce the number of values (limit {MAX_POINTS:,}).")
    if any(len(a) == 0 for a in axes):
        raise ValueError("Every axis needs at least one value.")
    T_, W_, G_, F_ = (g.ravel() for g in np.meshgrid(*axes, indexing="ij"))
    case = _case(kind, F_, W_, G_)
    top = T_.copy()
    if surface is not None and surface.items:
        w, cp = fluid.mass_cp(case, hc)
        t_wh = V.temperature_profile(path, w, cp, u)[:, 0]
        with np.errstate(all="ignore"):
            top = SF.wellhead_pressure(surface.items, fluid, kind, T_, case, t_wh, surface.correlation, hc,
                                       surface.t_surround, surface.u, surface.choke_method, surface.cd,
                                       p2=surface.friction)
    with np.errstate(all="ignore"):
        P, _ = V.traverse(path, fluid, top, case, correlation, p1, p2, u, hc)
    shape = tuple(len(a) for a in axes)
    return axes, P[:, -1].reshape(shape), top.reshape(shape)


def vfp_checks(axes, bhp):
    """Plain-language warnings about the table."""
    thp, wfr, gfr, flo = axes
    out = []
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
    below = int((bhp < thp[:, None, None, None] - 0.5).sum())
    if below:
        out.append(f"{below} point(s) with BHP below THP - not physical for a producer; check the inputs.")
    return out


def vfp_csv(kind, axes, bhp, wh=None):
    thp, wfr, gfr, flo = axes
    a = AXES[kind]
    rows = []
    for i, t in enumerate(thp):
        for j, w in enumerate(wfr):
            for k, g in enumerate(gfr):
                for m, f in enumerate(flo):
                    r = {"Top node pressure (psia)": t, f"{a['wfr'][0]} ({a['wfr'][1]})": w,
                         f"{a['gfr'][0]} ({a['gfr'][1]})": g, f"{a['flo'][0]} ({a['flo'][1]})": f,
                         "Bottom node pressure (psia)": bhp[i, j, k, m]}
                    if wh is not None:
                        r["Wellhead pressure (psia)"] = wh[i, j, k, m]
                    rows.append(r)
    return pd.DataFrame(rows)


def _line(vals, fmt="{:.6g}", per=6):
    s = [fmt.format(v) for v in vals]
    return "\n".join("  " + " ".join(s[i:i + per]) for i in range(0, len(s), per)) + " /"


def eclipse_vfpprod(kind, axes, bhp, table_no=1, datum_tvd=0.0, title=""):
    """Eclipse VFPPROD keyword, FIELD units. Oil: FLO=LIQ (STB/d), WFR=WCT (fraction), GFR=GOR (Mscf/STB).
    Gas: FLO=GAS (Mscf/d), WFR=WGR (STB/Mscf), GFR=OGR (STB/Mscf). THP and BHP in psia, datum in ft TVD."""
    thp, wfr, gfr, flo = (np.asarray(a, float) for a in axes)
    bhp = np.asarray(bhp, float)
    if kind == "oil":
        f_flo, f_wfr, f_gfr = flo, wfr / 100.0, gfr / 1000.0
        types = ("'LIQ'", "'WCT'", "'GOR'")
    else:
        f_flo, f_wfr, f_gfr = flo * 1000.0, wfr / 1000.0, gfr / 1000.0
        types = ("'GAS'", "'WGR'", "'OGR'")
    out = ["-- Generated by wellmodel" + (f": {title}" if title else ""),
           f"-- {len(flo)} rates x {len(thp)} THP x {len(wfr)} WFR x {len(gfr)} GFR, no artificial lift (ALQ = 0)",
           "VFPPROD", "-- Table  Datum(ft)  FLO    WFR    GFR    THP    ALQ   Units    Tabulated",
           f"   {int(table_no)}  {datum_tvd:.2f}  {types[0]}  {types[1]}  {types[2]}  'THP'  ' '  'FIELD'  'BHP' /",
           "-- FLO values", _line(f_flo), "-- THP values (psia)", _line(thp),
           "-- WFR values", _line(f_wfr), "-- GFR values", _line(f_gfr), "-- ALQ values", "  0 /",
           "-- NTHP NWFR NGFR NALQ  BHP (psia) for each FLO value"]
    for i in range(len(thp)):
        for j in range(len(wfr)):
            for k in range(len(gfr)):
                vals = [f"{v:.2f}" for v in bhp[i, j, k]]
                lines = [" ".join(vals[x:x + 8]) for x in range(0, len(vals), 8)]   # Eclipse: <= 132 chars
                out.append(f"  {i + 1} {j + 1} {k + 1} 1  " + "\n    ".join(lines) + " /")
    return "\n".join(out) + "\n"
