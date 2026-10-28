"""Intermittent gas lift design (Brown, The Technology of Artificial Lift Methods Vol. 2a, 1980, section 3.43).

Pressures in this module are psig, as in Brown (the app converts at the boundary); depths are TVD, ft.

1. Valve spacing with unbalanced pressure (casing) valves and a surface-opening-pressure drop per valve
   (Brown 3.4353 graphical, gas column included; 3.4354 analytical, eq. 3.451-3.455):
      top valve:  kill line Pwh + Gkill D meets the casing line of Pso1 = Pko - margin (or the static fluid level
                  when the well is not loaded to the surface),
      next valve: D(n+1) = D(n) + [Pc(n+1) - Pwh - Gu D(n)] / Gs, Gu the unloading gradient,
      operating valve at the deepest depth with an extra drop in Pso (25-100 psi, Brown uses 50).
2. Gas per cycle (eq. 3.439-3.446): the tubing above the operating valve, less the slug remaining at the surface
   after the fallback, filled with gas at the average pressure under the slug (Pvo at the bottom, Pwh plus the
   remaining slug at the top) and average temperature.
3. Casing pressure drop that supplies that gas from the annulus above the valve (Brown Fig. 3H, as an equation);
   under choke control the whole drop is placed in the valve spread, under time-cycle control half of it.
   Port: R = spread / (Pvo - Pt) (eq. 3.447); the largest port with R not above that (Brown picks the lower size).
   Settings: Pvc = Pd = Pvo (1 - R) + Pt R (eq. 3.448), Pd60 = Ct Pd, Ptro = Pd60 / (1 - R) (eq. 3.449).
4. Production by the weighted-average flowing BHP (Brown 3.436, B p.309-315): Time I (slug travel, 1.5 D/1000
   min in 2-in tubing), Time II (after surfacing, 0.5 D/1000 min) and Time III (build-up from the minimum pressure
   back to the tubing load by feed-in), with or without a standing valve:
      Pmin = Ptop + (1 - recovery) (Pload - Ptop) + feed-in during I (+ II),
      feed-in psi/min = q(p) / 1440 / Ct * Gs,  q from the IPR at the BHP the valve pressure gives,
      Pavg = time-weighted pressure at the valve, q = IPR(Pavg).
   The slug check q = cycles/day x recovered slug is reported too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import blackoil as bo
from .gaslift import LiftGas, ValveType, dome_at_60, gas_column, surface_from_depth

PSIA = 14.696
BBL_FT3 = 5.614583

# Brown Table 3.488 (B p.277): R for 1-in (Ab 0.29 in2) and 1.5-in (Ab 0.77 in2) valves, ports in 64ths
IGL_VALVES = {
    "Brown 1.5-in unbalanced (Ab 0.77 in2)": ValveType(
        "Brown 1.5-in unbalanced (Ab 0.77 in2)", 0.77, 40, (12, 16, 18, 20, 24, 28, 32, 36, 40),
        ((12, 0.0359), (16, 0.0638), (18, 0.08068), (20, 0.0996), (24, 0.1434), (28, 0.1952), (32, 0.2562),
         (36, 0.3227), (40, 0.3990))),
    "Brown 1-in unbalanced (Ab 0.29 in2)": ValveType(
        "Brown 1-in unbalanced (Ab 0.29 in2)", 0.29, 28, (12, 16, 18, 20, 22, 24, 28),
        ((12, 0.0863), (16, 0.1534), (18, 0.1942), (20, 0.2397), (22, 0.29), (24, 0.345), (28, 0.4697))),
}

# Brown Table 3.484 (B p.267): unloading gradients (psi/ft), 50 % excess gas, by Pvo - load and port
_GU_DP = (150.0, 200.0, 300.0)
_GU_PORT = (20, 24, 32, 44, 64)
_GU = np.array([[0.0420, 0.0410, 0.0370, 0.0345, 0.0312],
                [0.0390, 0.0380, 0.0354, 0.0320, 0.0270],
                [0.0240, 0.0235, 0.0220, 0.0210, 0.0200]])
# Fig 3F.1/3F.2 (Otis): minimum unloading gradient (psi/ft) vs design rate, 2-in tubing (digitised)
_GU2_Q = (100.0, 200.0, 300.0, 400.0, 500.0, 800.0, 1200.0, 1600.0)
_GU2 = (0.043, 0.062, 0.08, 0.099, 0.11, 0.137, 0.162, 0.175)

# Brown Table 3.489: dynamic loss under the moving slug, % of Pvo, by port and Pvo - Pt
_FSLUG_PORT = (16, 24, 36, 40)            # upper end of each port band, 64ths
_FSLUG = np.array([[20, 25, 30], [15, 20, 25], [10, 15, 20], [8, 10, 15]], float)


def unloading_gradient_table(dp, port_64):
    """Brown Table 3.484 (50 % excess gas), interpolated."""
    dp = float(np.clip(dp, _GU_DP[0], _GU_DP[-1]))
    pp = float(np.clip(port_64, _GU_PORT[0], _GU_PORT[-1]))
    row = [np.interp(pp, _GU_PORT, _GU[k]) for k in range(3)]
    return float(np.interp(dp, _GU_DP, row))


def unloading_gradient_rate(q_bpd, tubing_id_in):
    """Otis minimum unloading gradient (Fig 3F.2) for 2-in tubing scaled to other sizes by flow area."""
    q2 = float(q_bpd) * (1.995 / max(tubing_id_in, 0.5)) ** 2
    return float(np.interp(q2, _GU2_Q, _GU2))


def slug_loss_fraction(port_64, dp_static):
    i = int(np.searchsorted(_FSLUG_PORT, port_64).clip(0, 3))
    return float(np.interp(dp_static, (200.0, 300.0, 400.0), _FSLUG[i])) / 100.0


def tubing_capacity_bbl_ft(d_in):
    return np.pi / 4 * (d_in / 12.0) ** 2 / BBL_FT3


def choose_port(valve: ValveType, r_required, port_fixed=None):
    """Brown: the largest standard port whose R does not exceed the required R (smallest port at least)."""
    ports = valve.ports
    if port_fixed:
        return int(port_fixed), float(valve.r(port_fixed))
    ok = [p for p in ports if valve.r(p) <= r_required + 1e-12]
    p = max(ok) if ok else min(ports)
    return int(p), float(valve.r(p))


def port_label(p64):
    from fractions import Fraction
    f = Fraction(int(p64), 64)
    return f"{f.numerator}/{f.denominator}" if f < 1 else f"{f}"


# ================================================================ input / output
@dataclass
class IGLInput:
    # surface and gas
    p_ko: float = 1050.0              # available injection (kick-off) pressure at the surface, psig
    top_margin: float = 50.0          # Pso of the top valve = p_ko - top_margin
    p_wh: float = 80.0                # wellhead (separator) pressure on top of the slug, psig
    pso_drop: float = 25.0            # surface opening pressure drop per valve
    op_drop: float = 50.0             # extra drop at the operating valve (Brown 25-100)
    # gradients
    g_s: float = 0.40                 # well fluid gradient, psi/ft
    g_kill: float = 0.40              # kill fluid gradient, psi/ft
    g_u: float = 0.04                 # unloading gradient, psi/ft
    well_killed: bool = True          # loaded to the surface; else the top valve sits at the static level
    spacing: str = "Graphical (gas column)"     # or "Analytical (Brown 3.4354)"
    max_depth: float = 7600.0         # deepest valve (TVD), ft
    min_spacing: float = 250.0
    # valves
    valve: str = "Brown 1.5-in unbalanced (Ab 0.77 in2)"
    control: str = "Time cycle"       # or "Choke"
    unloading_port: int = 20          # 64ths
    operating_port: int | None = None  # None: from the gas requirement
    # slug and cycle
    load: str = "Half reservoir pressure"       # or "Pvo - dP"
    dp_load: float = 200.0            # Pvo - tubing load, when load = "Pvo - dP"
    fallback: float = 7.0             # % per 1000 ft
    recovery: float | None = None     # fraction; None: 1 - fallback x D / 1000
    top_allowance: float = 10.0       # gas pressure on top of the slug above Pwh, psi
    t1_factor: float = 1.5            # Time I, min per 1000 ft (2-in tubing)
    t2_factor: float = 0.5            # Time II, min per 1000 ft
    standing_valve: bool = True
    high_pressure_dome: bool = True


@dataclass
class IGLWell:
    tubing_id: float                  # in
    tubing_od: float                  # in
    casing_id: float                  # in
    perf_tvd: float                   # ft, datum of the IPR
    p_res: float                      # psig
    ipr_q: object                     # callable: liquid rate (B/D) at flowing BHP (psig)
    temp: object                      # callable: temperature (degF) at TVD (valve / casing)
    t_surface: float = 80.0


@dataclass
class IGLResult:
    valves: pd.DataFrame
    cycle: dict
    gas: dict
    notes: list = field(default_factory=list)
    unloading: pd.DataFrame | None = None


# ================================================================ gas
def _z(gas: LiftGas, p_psia, t_f):
    return float(bo.gas_props(gas.inp, np.array([p_psia]), np.array([t_f]))[0][0])


def casing_at_depth(gas: LiftGas, pso, tvd, well: IGLWell):
    """Casing pressure (psig) at tvd from the surface pressure (psig): static gas column."""
    return float(gas_column(gas, pso + PSIA, 0.0, tvd, well.t_surface, well.temp(tvd))) - PSIA


def surface_from(gas: LiftGas, p_depth, tvd, well: IGLWell):
    return float(surface_from_depth(gas, p_depth + PSIA, 0.0, tvd, well.t_surface, well.temp(tvd))) - PSIA


def gas_per_cycle(gas: LiftGas, pvo, p_top, depth, slug_ft, g_s, fallback_pct, d_in, temp):
    """Brown eq. 3.439-3.446: scf of gas to fill the tubing behind the slug at the end of the lift.
    pvo, p_top psig; slug_ft initial slug length; temp(tvd) degF."""
    remain = max(1.0 - fallback_pct / 100.0 * depth / 1000.0, 0.0) * slug_ft
    pavg = (pvo + p_top + remain * g_s) / 2.0 + PSIA
    tavg = (temp(remain) + temp(depth)) / 2.0
    hgc = max(depth - remain, 0.0)
    cap = np.pi / 4 * (d_in / 12.0) ** 2
    z = _z(gas, pavg, tavg)
    return hgc * cap * (pavg / PSIA) * (520.0 / (tavg + 460.0)) / z


def casing_drop_for_gas(gas: LiftGas, vg_scf, depth, pc_avg, well: IGLWell):
    """Casing pressure drop (psi) that releases vg_scf from the annulus above depth (Brown Fig. 3H)."""
    v_ann = np.pi / 4 * ((well.casing_id / 12.0) ** 2 - (well.tubing_od / 12.0) ** 2) * depth
    t_avg = (well.t_surface + well.temp(depth)) / 2.0
    z = _z(gas, pc_avg + PSIA, t_avg)
    scf_per_psi = v_ann / PSIA * 520.0 / (t_avg + 460.0) / z
    return vg_scf / scf_per_psi, scf_per_psi


# ================================================================ spacing
def space_valves(inp: IGLInput, gas: LiftGas, well: IGLWell):
    """Valve depths and surface opening pressures (Brown 3.4353 / 3.4354)."""
    pso1 = inp.p_ko - inp.top_margin
    graphical = inp.spacing.startswith("Graphical")
    if inp.well_killed:
        if graphical:
            f = lambda d: inp.p_wh + inp.g_kill * d - casing_at_depth(gas, pso1, d, well)
            d1 = brentq(f, 1.0, 60000.0) if f(60000.0) > 0 else np.inf
        else:
            d1 = (pso1 - inp.p_wh) / inp.g_kill
    else:
        d1 = max(well.perf_tvd - well.p_res / inp.g_s, 0.0)
    depths, psos = [], []
    d, pso = d1, pso1
    while d < inp.max_depth - inp.min_spacing:
        depths.append(d)
        psos.append(pso)
        nxt = pso - inp.pso_drop
        pt = inp.p_wh + inp.g_u * d
        if graphical:
            f = lambda x: pt + inp.g_s * (x - d) - casing_at_depth(gas, nxt, x, well)
            if f(d + 1.0) >= 0:
                break
            dn = brentq(f, d + 1.0, 80000.0) if f(80000.0) > 0 else np.inf
        else:
            dn = d + (pso - pt) / inp.g_s
        if dn - d < inp.min_spacing:
            break
        d, pso = dn, nxt
    last_pso = (psos[-1] if psos else pso1) - inp.op_drop
    depths.append(float(inp.max_depth))
    psos.append(last_pso)
    return np.array(depths), np.array(psos)


# ================================================================ cycle (weighted-average BHP)
def _feed_rate(well: IGLWell, p_valve, depth, g_s):
    """Inflow (B/D) with the valve-depth pressure p_valve (psig): BHP = p_valve + liquid below the valve."""
    pwf = p_valve + g_s * max(well.perf_tvd - depth, 0.0)
    return max(float(well.ipr_q(pwf)), 0.0)


def weighted_bhp(pvo, p_load, p_top, depth, recovery, g_s, ct, feed, t1, t2, standing_valve=True):
    """Brown's weighted-average flowing BHP. feed(p) -> B/D at valve pressure p. Returns a dict."""
    psi_min = lambda p: feed(p) / 1440.0 / ct * g_s           # build-up rate in the tubing, psi/min
    fall = (1.0 - recovery) * (p_load - p_top)
    if standing_valve:
        # pressure under the standing valve builds from the load during Time I
        p_end = p_load
        for _ in range(200):
            new = p_load + psi_min(0.5 * (p_load + p_end)) * t1
            if abs(new - p_end) < 1e-6:
                break
            p_end = new
        dp1 = p_end - p_load
    else:
        p_end = np.nan
        dp1 = psi_min(pvo) * t1
    pmin = p_top + fall + dp1
    for _ in range(200):
        new = p_top + fall + dp1 + psi_min(0.5 * (p_load + pmin)) * t2
        if abs(new - pmin) < 1e-7:
            break
        pmin = new
    out = dict(p_end_I=p_end, dp_I=dp1, fallback_psi=fall, p_min=pmin)
    if pmin >= p_load:
        t3 = 0.0
    else:
        r3 = psi_min(0.5 * (pmin + p_load))
        t3 = (p_load - pmin) / r3 if r3 > 0 else np.inf
    cyc = t1 + t2 + t3
    if not np.isfinite(cyc):
        out.update(t3=np.inf, cycle=np.inf, p_avg=np.nan)
        return out
    if standing_valve:
        pav = (0.5 * (p_load + p_end) * t1 + 0.5 * (p_load + pmin) * t2 + 0.5 * (pmin + p_load) * t3) / cyc
    else:
        pav = (pvo * t1 + 0.5 * (p_load + pmin) * (cyc - t1)) / cyc
    out.update(t3=t3, cycle=cyc, p_avg=pav)
    return out


def cycle_performance(inp: IGLInput, gas: LiftGas, well: IGLWell, depth, pvo, p_load):
    """Cycle, rates and gas for the operating valve at depth with opening pressure pvo and tubing load p_load."""
    ct = tubing_capacity_bbl_ft(well.tubing_id)
    p_top = inp.p_wh + inp.top_allowance
    rec = inp.recovery if inp.recovery is not None else max(1.0 - inp.fallback / 100.0 * depth / 1000.0, 0.05)
    t1, t2 = inp.t1_factor * depth / 1000.0, inp.t2_factor * depth / 1000.0
    feed = lambda p: _feed_rate(well, p, depth, inp.g_s)
    w = weighted_bhp(pvo, p_load, p_top, depth, rec, inp.g_s, ct, feed, t1, t2, inp.standing_valve)
    slug_ft = max(p_load - p_top, 0.0) / inp.g_s
    slug_bbl = slug_ft * ct
    rec_bbl = rec * slug_bbl
    fallback_pct = (1.0 - rec) / max(depth / 1000.0, 1e-9) * 100.0
    vg = gas_per_cycle(gas, pvo, inp.p_wh, depth, slug_ft, inp.g_s, fallback_pct, well.tubing_id, well.temp)
    n = 1440.0 / w["cycle"] if np.isfinite(w["cycle"]) and w["cycle"] > 0 else 0.0
    q_avg = _feed_rate(well, w["p_avg"], depth, inp.g_s) if np.isfinite(w["p_avg"]) else 0.0
    pwf_avg = w["p_avg"] + inp.g_s * max(well.perf_tvd - depth, 0.0) if np.isfinite(w["p_avg"]) else np.nan
    return {
        "Operating valve depth (ft)": depth, "Valve opening pressure at depth (psig)": pvo,
        "Tubing load at valve (psig)": p_load, "Pvo - tubing load (psi)": pvo - p_load,
        "Pressure on top of slug (psig)": p_top,
        "Starting slug (ft)": slug_ft, "Starting slug (bbl)": slug_bbl, "Recovery (fraction)": rec,
        "Recovered per cycle (bbl)": rec_bbl,
        "Time I - slug travel (min)": t1, "Time II - after surfacing (min)": t2,
        "Time III - build-up (min)": w["t3"], "Cycle time (min)": w["cycle"], "Cycles per day": n,
        "Pressure under standing valve, end of Time I (psig)": w["p_end_I"],
        "Minimum pressure at valve (psig)": w["p_min"],
        "Weighted-average pressure at valve (psig)": w["p_avg"],
        "Weighted-average flowing BHP (psig)": pwf_avg,
        "Liquid rate (B/D)": q_avg, "Liquid rate, cycles x recovered slug (B/D)": n * rec_bbl,
        "Gas per cycle (scf)": vg, "Injection gas (Mscf/d)": n * vg / 1000.0,
        "Injection GLR (scf/bbl)": vg / rec_bbl if rec_bbl > 0 else np.nan,
    }


# ================================================================ design
def design(inp: IGLInput, gas: LiftGas, well: IGLWell) -> IGLResult:
    notes = []
    valve = IGL_VALVES.get(inp.valve)
    if valve is None:
        from .gaslift import VALVES
        valve = VALVES[inp.valve]
    depths, psos = space_valves(inp, gas, well)
    n = len(depths)
    pvos = np.array([casing_at_depth(gas, ps, d, well) for ps, d in zip(psos, depths)])
    d_op, pvo_op = depths[-1], pvos[-1]
    p_load = well.p_res / 2.0 if inp.load.startswith("Half") else pvo_op - inp.dp_load
    if inp.load.startswith("Half") and well.perf_tvd > d_op:
        p_load = max(well.p_res / 2.0 - inp.g_s * (well.perf_tvd - d_op), inp.p_wh + inp.top_allowance + 1.0)
    if p_load >= pvo_op:
        notes.append("The tubing load is not below the valve opening pressure - raise the injection pressure "
                     "or lower the load.")
        p_load = pvo_op - 50.0
    perf = cycle_performance(inp, gas, well, d_op, pvo_op, p_load)
    vg = perf["Gas per cycle (scf)"]
    dpc, scf_psi = casing_drop_for_gas(gas, vg, d_op, pvo_op - 0.5 * (pvo_op - psos[-1]), well)
    spread_req = dpc if inp.control.startswith("Choke") else 0.5 * dpc
    r_req = spread_req / max(pvo_op - p_load, 1e-9)
    port_op, r_op = choose_port(valve, r_req, inp.operating_port)
    r_unl = float(valve.r(inp.unloading_port))
    rows = []
    for k in range(n):
        d, pso, pvo = depths[k], psos[k], pvos[k]
        op = k == n - 1
        t = float(well.temp(d))
        pt = p_load if op else inp.p_wh + inp.g_u * d
        port, r = (port_op, r_op) if op else (int(inp.unloading_port), r_unl)
        pvc = pvo * (1 - r) + pt * r
        psc = surface_from(gas, pvc, d, well)
        pd60 = float(dome_at_60(pvc, t, inp.high_pressure_dome))
        rows.append({"Valve": k + 1, "Depth (ft)": d, "Temperature (°F)": t,
                     "Surface opening pressure (psig)": pso, "Opening pressure at depth (psig)": pvo,
                     "Tubing pressure (psig)": pt, "Port (in)": port_label(port), "R": r,
                     "Closing pressure at depth = dome (psig)": pvc, "Surface closing pressure (psig)": psc,
                     "Spread at depth (psi)": pvo - pvc, "Dome at 60 °F (psig)": pd60,
                     "Test-rack opening at 60 °F (psig)": pd60 / (1 - r),
                     "Role": "Operating" if op else "Unloading"})
    vtab = pd.DataFrame(rows)
    spread = pvo_op - (pvo_op * (1 - r_op) + p_load * r_op)
    gas_d = {"Casing pressure drop to supply the gas (psi)": dpc, "Annulus gas per psi (scf/psi)": scf_psi,
             "Spread required in the valve (psi)": spread_req, "R required": r_req,
             "Operating port": port_label(port_op), "Operating valve spread (psi)": spread,
             "Gas from the valve spread (scf/cycle)": spread * scf_psi,
             "Casing build-up above Pvo to supply the gas (psi)": max(dpc - spread, 0.0)}
    f = slug_loss_fraction(port_op, pvo_op - p_load)
    gas_d["Maximum pressure under the moving slug (psig)"] = pvo_op * (1 - f)
    if inp.control.startswith("Choke") and dpc - spread > 1.0:
        notes.append(f"Choke control: the valve spread supplies {spread:.0f} of the {dpc:.0f} psi casing drop; the "
                     "choke supplies the rest during the lift, or build the casing up to open on a lighter load.")
    if perf["Time III - build-up (min)"] <= 0.0:
        notes.append("Feed-in refills the tubing above the tubing load before the slug has cleared - the inflow is "
                     "too high for intermittent lift; this is a continuous-flow well.")
    lim = float(np.interp(well.tubing_id, (1.049, 1.38, 1.61, 1.995, 2.441, 2.992), (50, 75, 125, 200, 250, 300)))
    if perf["Liquid rate (B/D)"] > lim:
        notes.append(f"{perf['Liquid rate (B/D)']:,.0f} B/D is above Brown's intermittent limit for this tubing "
                     f"(about {lim:,.0f} B/D, Table 3.483) - consider continuous lift.")
    if r_req > 1.0:
        notes.append("The gas per cycle needs a casing pressure drop larger than the valve can give "
                     f"(R required {r_req:.2f}) - the annulus is too small for the slug; lower the load or use a "
                     "smaller slug.")
    if well.p_res <= p_load:
        notes.append("Reservoir pressure is below the tubing load - the slug never builds.")
    if d_op / 1000 * 1.5 > perf["Cycle time (min)"]:
        notes.append("Cycle shorter than 1.5 min per 1000 ft - fallback may not settle (Brown B p.274).")
    # interference (Brown B p.281): with the slug passing an upper valve, the tubing pressure there lies on a line from
    # the maximum pressure under the slug at the operating valve to the pressure on top of the slug at the surface;
    # the valve reopens if its opening pressure with that tubing pressure is below the casing pressure there
    pt_max = gas_d["Maximum pressure under the moving slug (psig)"]
    p_top = perf["Pressure on top of slug (psig)"]
    for k in range(n - 1):
        pt_k = p_top + (pt_max - p_top) * depths[k] / d_op
        pvc_k = float(vtab.loc[k, "Closing pressure at depth = dome (psig)"])
        pvo_slug = (pvc_k - pt_k * r_unl) / (1 - r_unl)
        pc_k = casing_at_depth(gas, psos[-1], depths[k], well)
        if pvo_slug <= pc_k:
            notes.append(f"Valve {k + 1} may reopen while the slug passes it (opens at {pvo_slug:.0f} psig against "
                         f"{pc_k:.0f} psig in the casing) - use a larger drop at the operating valve.")
    return IGLResult(vtab, perf, gas_d, notes)


def load_sensitivity(inp: IGLInput, gas: LiftGas, well: IGLWell, res: IGLResult, dps=(100, 150, 200, 250, 300, 350,
                                                                                         400)):
    """Rate, cycle and gas against Pvo - tubing load at the operating valve (Brown/Deschner)."""
    d = float(res.valves["Depth (ft)"].iloc[-1])
    pvo = float(res.valves["Opening pressure at depth (psig)"].iloc[-1])
    rows = []
    for dp in dps:
        pl = pvo - dp
        if pl <= inp.p_wh + inp.top_allowance + 10:
            continue
        r = cycle_performance(inp, gas, well, d, pvo, pl)
        rows.append({"Pvo - load (psi)": dp, "Tubing load (psig)": pl, "Cycle time (min)": r["Cycle time (min)"],
                     "Cycles per day": r["Cycles per day"],
                     "Weighted-average pressure at valve (psig)": r["Weighted-average pressure at valve (psig)"],
                     "Liquid rate (B/D)": r["Liquid rate (B/D)"], "Gas per cycle (scf)": r["Gas per cycle (scf)"],
                     "Injection gas (Mscf/d)": r["Injection gas (Mscf/d)"],
                     "Injection GLR (scf/bbl)": r["Injection GLR (scf/bbl)"]})
    return pd.DataFrame(rows)


# ================================================================ PROSPER constant surface closing pressure spacing
def constant_psc_spacing(p_inj, p_wh, p_sc, g_unload, spacing_factor, gas_grad, max_depth, wh_depth=0.0,
                         max_valves=40):
    """PROSPER Design | Intermittent Gas Lift | Valves | Constant Surface Closing Pressure (psig, ft).

    Reproduces PROSPER's Tutorial 24 table exactly:
      top valve: the unloading fluid from the wellhead pressure at the wellhead (Xmas tree) depth meets the
                 injection pressure line, Pwh + Gs (D1 - Dwh) = Pinj + g (D1 - Dwh);
      next valves: D(n+1) = D(n) + [Pclose(Dn) + g Dn - Psp(Dn)] / (Gs - g);
      displayed pressures: spacing Psp = Pwh + Fs D, closing Psc + g D, injection Pinj + g D (constant annulus
      gas gradient g, depths from the surface). The deepest valve is the last one above the maximum depth."""
    gs, g = float(g_unload), float(gas_grad)
    if gs <= g:
        raise ValueError("The unloading fluid gradient must exceed the annulus gas gradient.")
    rows = [{"Valve": "Wellhead", "Valve Depth (ft)": wh_depth, "Spacing Pressure (psig)": p_wh,
             "Closing Pressure (psig)": p_sc, "Injection Pressure (psig)": p_inj}]
    d = wh_depth + (p_inj - p_wh) / (gs - g)
    n = 0
    while d <= max_depth and n < max_valves:
        n += 1
        psp = p_wh + spacing_factor * d
        rows.append({"Valve": f"Valve {n}", "Valve Depth (ft)": d, "Spacing Pressure (psig)": psp,
                     "Closing Pressure (psig)": p_sc + g * d, "Injection Pressure (psig)": p_inj + g * d})
        step = (p_sc + 2.0 * g * d - psp) / (gs - g)
        if step <= 1.0:
            break
        d = d + step
    return pd.DataFrame(rows)


# ================================================================ PROSPER intermittent calculation (White et al. 1963)
def white_slug_velocity(pc, pt, port_64, tubing_id_in):
    """White et al. (1963) eq. 2: vs = 1430 (Ap/At) sqrt((Pc/Pt)^2 - Pc/Pt), ft/s. PROSPER takes the ratio of the
    gauge pressures."""
    r = pc / pt
    return 1430.0 * (port_64 / 64.0) ** 2 / tubing_id_in ** 2 * np.sqrt(max(r * r - r, 0.0))


def _moody(re, rel_rough):
    if re < 2000:
        return 64.0 / max(re, 1e-9)
    return 0.25 / np.log10(rel_rough / 3.7 + 5.74 / re ** 0.9) ** 2


def white_cycle(p_inj_valve, p_wh, depth, slug_ft, port_64, tubing_id_in, g_liq, v_bubble, rho_liq=None, mu_cp=None,
                rough_in=0.0006, friction=True, gas: LiftGas | None = None, t_valve=None):
    """PROSPER Design | Intermittent Gas Lift results (psig, ft below the wellhead):
       slug pressure  Pt = Pwh + G S [1 + f vs^2 / (2 g d)]           (White eq. 3, solved with eq. 2),
       slug velocity  vs = 1430 (Ap/At) sqrt((Pc/Pt)^2 - Pc/Pt),
       fall back      FB = D / (1 + vs/vb),  produced N = S - FB,
       time to surface (D - N)/vs = cycle time, gas per cycle = in-situ tubing volume above the produced slug."""
    d_ft = tubing_id_in / 12.0
    at = np.pi / 4 * d_ft ** 2
    rho = rho_liq if rho_liq is not None else g_liq * 144.0
    pt = p_wh + g_liq * slug_ft
    vs = white_slug_velocity(p_inj_valve, pt, port_64, tubing_id_in)
    for _ in range(100):
        fr = 0.0
        if friction and mu_cp:
            re = rho * vs * d_ft / (mu_cp * 6.7197e-4)
            fr = _moody(re, rough_in / tubing_id_in) * vs ** 2 / (d_ft * 2 * 32.174)
        pt_new = p_wh + g_liq * slug_ft * (1.0 + fr)
        vs_new = white_slug_velocity(p_inj_valve, pt_new, port_64, tubing_id_in) if pt_new < p_inj_valve else 0.0
        if abs(vs_new - vs) < 1e-10 and abs(pt_new - pt) < 1e-8:
            break
        pt, vs = pt_new, vs_new
    out = {"Downhole Injection Pressure (psig)": p_inj_valve, "Downhole Pressure (psig)": pt,
           "Slug Velocity (ft/s)": vs, "Bubble Velocity (ft/s)": v_bubble}
    if vs <= 0:
        out["note"] = "The injection pressure at the valve does not exceed the slug pressure - the slug is not lifted."
        return out
    if callable(v_bubble):
        v_bubble = float(v_bubble(vs, pt))
        out["Bubble Velocity (ft/s)"] = v_bubble
    fb = depth / (1.0 + vs / v_bubble)
    n = max(slug_ft - fb, 0.0)
    t = (depth - n) / vs
    cap_bbl = at / BBL_FT3
    vg = (depth - n) * at
    out.update({"Fall Back (ft)": fb, "Produced Slug Length (ft)": n, "Produced Slug Volume (bbl)": n * cap_bbl,
                "Pcasing / Pslug": p_inj_valve / pt, "Produced/Accumulated Slug Length": n / slug_ft,
                "Starting Slug Length / Valve Depth": slug_ft / depth, "Time To Surface (s)": t,
                "Cycle Frequency (1/hour)": 3600.0 / t, "Gas per cycle, in situ (ft3)": vg,
                "Gas Injection Rate, in situ (MMcf/day)": vg * 86400.0 / t / 1e6,
                "Daily Production Rate (STB/d)": n * cap_bbl * 86400.0 / t})
    if gas is not None and t_valve is not None:
        # PROSPER's standard gas rate: the in-situ volume at the slug pressure and valve temperature
        bg = float(bo.gas_props(gas.inp, np.array([pt + PSIA]), np.array([t_valve]))[1][0])
        out["Gas Injection Rate (MMscf/day)"] = out["Gas Injection Rate, in situ (MMcf/day)"] / bg
    return out


# PROSPER's bubble velocity is not constant: from seven PROSPER runs of Tutorial 24 (4.052-in tubing, 24/64 and 16/64
# ports, water cut 0-100 %, Pc 1828 psig) it follows vb = 2.2333 + 2.3673 vs^2 / Pt - 0.000579 WC(%) within 0.002 ft/s.
# The tubing-size and injection-pressure dependence is untested; White et al. (1963) measured vb as a constant per
# fluid and tubing size.
VB_FIT = (2.233281, 2.367292, -5.79e-4)


def bubble_velocity_prosper(vs, pt_psig, wc_pct):
    a, b, c = VB_FIT
    return a + b * vs * vs / max(pt_psig, 1.0) + c * wc_pct


def prosper_intermittent(fluid, gas: LiftGas, p_inj, p_wh, wh_tvd, valve_tvd, level_tvd, port_64, tubing_id_in,
                         wc_pct, temp, gor=None, rough_in=0.0006, v_bubble=None, t_fluid=None, p_fluid="average"):
    """PROSPER Design | Intermittent Gas Lift (psig; depths TVD from the surface; temp(tvd) degF).
    The casing gas column starts at the wellhead (Xmas tree) depth at the geothermal temperature; the liquid
    gradient and viscosity come from the PVT at the average slug pressure and temperature; v_bubble None uses
    the PROSPER-calibrated bubble velocity."""
    pc = float(gas_column(gas, p_inj + PSIA, wh_tvd, valve_tvd, temp(wh_tvd), temp(valve_tvd))) - PSIA
    d = valve_tvd - wh_tvd
    s_len = max(valve_tvd - level_tvd, 1.0)
    t_slug = t_fluid if t_fluid is not None else temp(0.5 * (level_tvd + valve_tvd))
    case = {"q": np.array([100.0]), "wc": np.array([wc_pct / 100.0]),
            "gor": np.array([gor if gor is not None else 0.0])}
    g_liq, mu, pt = 0.4, 1.0, p_wh + 0.4 * s_len
    for _ in range(30):
        p_ev = {"average": 0.5 * (p_wh + pt), "bottom": pt, "top": p_wh}[p_fluid]
        st = fluid.state(np.array([p_ev + PSIA]), np.array([t_slug]), case)
        g_new, mu = float(st["rho_l"][0]) / 144.0, float(st["mu_l"][0])
        r = white_cycle(pc, p_wh, d, s_len, port_64, tubing_id_in, g_new, 1.0, rho_liq=g_new * 144.0, mu_cp=mu,
                        rough_in=rough_in)
        pt_new = r["Downhole Pressure (psig)"]
        if abs(pt_new - pt) < 1e-4 and abs(g_new - g_liq) < 1e-8:
            break
        pt, g_liq = pt_new, g_new
    vb = v_bubble if v_bubble is not None else (lambda vs_, pt_: bubble_velocity_prosper(vs_, pt_, wc_pct))
    out = white_cycle(pc, p_wh, d, s_len, port_64, tubing_id_in, g_liq, vb, rho_liq=g_liq * 144.0, mu_cp=mu,
                      rough_in=rough_in, gas=gas, t_valve=temp(valve_tvd))
    out["Liquid gradient (psi/ft)"] = g_liq
    out["Liquid viscosity (cp)"] = mu
    return out


def annulus_gas_gradient(gas: LiftGas, p_inj, wh_tvd, max_tvd, temp):
    """Constant annulus gas gradient (psi/ft) for the constant-Psc spacing: gas density at the average column
    pressure and temperature between the wellhead and the maximum depth (PROSPER's tutorial value is ~1 % higher)."""
    g = 0.045
    for _ in range(30):
        pav = p_inj + g * (max_tvd - wh_tvd) / 2.0 + PSIA
        tav = 0.5 * (temp(wh_tvd) + temp(max_tvd))
        g = float(gas.density(pav, tav)) / 144.0
    return g


# ================================================================ PROSPER system analysis (intermittent)
def prosper_system(fluid, gas: LiftGas, p_inj, p_wh, wh_tvd, valve_tvd, port_64, tubing_id_in, wc_pct, temp, ipr_q,
                   gor=None, rough_in=0.0006, t_fluid=None, v_bubble=None, n=40, below_valve_grad=0.0,
                   perf_tvd=None):
    """PROSPER Calculation | System for an intermittent well: the 'VLP' is the slug pressure at the valve against the
    average intermittent rate as the starting slug length S varies; the solution is the slug length at which the IPR
    rate at that pressure equals the intermittent rate. PROSPER takes the slug pressure at the valve as the flowing
    BHP (below_valve_grad 0); a liquid gradient below the valve can be added (BHP at perf_tvd).
    ipr_q(pwf_psig) -> liquid rate. Returns (solution dict or None, curve DataFrame)."""
    def at(s_len):
        r = prosper_intermittent(fluid, gas, p_inj, p_wh, wh_tvd, valve_tvd, valve_tvd - s_len, port_64,
                                 tubing_id_in, wc_pct, temp, gor=gor, rough_in=rough_in, t_fluid=t_fluid,
                                 v_bubble=v_bubble)
        pwf = r["Downhole Pressure (psig)"] + below_valve_grad * max((perf_tvd or valve_tvd) - valve_tvd, 0.0)
        r["Flowing BHP (psig)"] = pwf
        r["IPR rate (STB/d)"] = float(ipr_q(pwf))
        return r
    d = valve_tvd - wh_tvd
    rows, ss = [], np.linspace(0.02, 0.98, n) * d
    for s_len in ss:
        r = at(s_len)
        if "note" in r:
            break
        rows.append({"Starting slug (ft)": s_len, "Slug pressure at valve (psig)": r["Downhole Pressure (psig)"],
                     "Flowing BHP (psig)": r["Flowing BHP (psig)"],
                     "Intermittent rate (STB/d)": r["Daily Production Rate (STB/d)"],
                     "IPR rate (STB/d)": r["IPR rate (STB/d)"],
                     "Cycle frequency (1/hour)": r["Cycle Frequency (1/hour)"]})
    curve = pd.DataFrame(rows)
    if len(curve) < 2:
        return None, curve
    f = (curve["Intermittent rate (STB/d)"] - curve["IPR rate (STB/d)"]).to_numpy()
    sol = None
    # the curve is a loop (two pressures per rate); PROSPER reports the crossing with the shorter slug (lower BHP)
    for k in range(1, len(f)):
        if f[k - 1] < 0 <= f[k] or f[k - 1] > 0 >= f[k]:
            g = lambda s_: (lambda r_: r_["Daily Production Rate (STB/d)"] - r_["IPR rate (STB/d)"])(at(s_))
            try:
                s_sol = brentq(g, curve["Starting slug (ft)"].iloc[k - 1], curve["Starting slug (ft)"].iloc[k],
                               xtol=1e-3)
                sol = at(s_sol)
                sol["Liquid Level (ft below surface)"] = valve_tvd - s_sol
                sol["Second solution (longer slug)"] = int(np.sum(np.diff(np.sign(f[k:])) != 0) > 0)
            except ValueError:
                sol = None
            break
    return sol, curve


# ================================================================ Opti-Flow spacing (Takacs 6.5.3)
def optiflow_spacing(gas: LiftGas, p_inj, p_wh, g_unload, sbhp, perf_tvd, max_tvd, temp, wh_tvd=0.0,
                     open_drop=100.0, top_margin=50.0, above=(0.30, 0.90), below=(0.55, 0.85), unl_step=20.0,
                     valve: ValveType | None = None, port_64=20, high_pressure_dome=True, min_spacing=50.0):
    """Opti-Flow unloading string (Axelson; Takacs, Gas Lift Manual 6.5.3), graphical method with real gas columns:
       Popen = Pinj - open_drop; spacing lines are gas columns from surface pressures of (30 %, 90 %) of Popen above
       the static liquid level and (55 %, 85 %) below it; static level Lst = Lperf - SBHP/gu (eq. 6.11); top valve
       where Pwh + gu L meets the Pinj - 50 column (eq. 6.12); next valve where a line of gradient gu from the
       minimum spacing line at the valve above meets the maximum spacing line (eq. 6.13-6.14); the bottom valve is
       moved to the maximum depth. Tubing pressure at each valve = minimum spacing line; valves above the static
       level get their surface opening pressure raised by unl_step per valve (counted upward from the deepest
       one) so the slug passing them does not open them; Pd = Pio (1 - R) + Ptmin R, dome at 60 F, TRO."""
    p_open = p_inj - open_drop
    col = lambda ps, d: float(gas_column(gas, ps + PSIA, wh_tvd, d, temp(wh_tvd), temp(d))) - PSIA if d > wh_tvd else ps
    l_static = perf_tvd - sbhp / g_unload
    lines = lambda d: above if d < l_static else below
    f1 = lambda d: p_wh + g_unload * (d - wh_tvd) - col(p_inj - top_margin, d)
    d = brentq(f1, wh_tvd + 1.0, 1e5)
    depths = [d]
    while True:
        lo, hi = lines(d)
        pmin = col(lo * p_open, d)
        f = lambda x: pmin + g_unload * (x - d) - col(hi * p_open, x)
        if f(d + 1.0) >= 0:
            break
        dn = brentq(f, d + 1.0, 1e5)
        if dn >= max_tvd or dn - d < min_spacing:
            break
        depths.append(dn)
        d = dn
    # Takacs step 7: the bottom valve as close as possible to the maximum depth - move the last valve there when
    # it is within half a spacing, else add one
    lo, hi = lines(depths[-1])
    pmin = col(lo * p_open, depths[-1])
    fnext = lambda x: pmin + g_unload * (x - depths[-1]) - col(hi * p_open, x)
    try:
        nxt = brentq(fnext, depths[-1] + 1.0, 1e5) - depths[-1]
    except ValueError:
        nxt = np.inf
    if len(depths) > 1 and max_tvd - depths[-1] < 0.5 * nxt:
        depths[-1] = float(max_tvd)
    elif depths[-1] < max_tvd - min_spacing:
        depths.append(float(max_tvd))
    else:
        depths[-1] = float(max_tvd)
    above_idx = [k for k, x in enumerate(depths) if x < l_static]
    r = float(valve.r(port_64)) if valve is not None else np.nan
    rows = []
    for k, x in enumerate(depths):
        lo, hi = lines(x)
        if k in above_idx:
            pso = p_open + unl_step * (len(above_idx) - above_idx.index(k))
        else:
            pso = p_open
        pio = col(pso, x)
        ptmin, ptmax = col(lo * p_open, x), col(hi * p_open, x)
        t = float(temp(x))
        row = {"Valve": k + 1, "Depth (ft)": x, "Temperature (°F)": t, "Surface opening pressure (psig)": pso,
               "Opening pressure at depth (psig)": pio, "Tubing pressure min (psig)": ptmin,
               "Tubing pressure max (psig)": ptmax, "Role": "Unloading" if k in above_idx else "Below static level"}
        if np.isfinite(r):
            pd_ = pio * (1 - r) + ptmin * r
            pd60 = float(dome_at_60(pd_, t, high_pressure_dome))
            row.update({"Port (in)": port_label(port_64), "R": r, "Dome at depth (psig)": pd_,
                        "Dome at 60 °F (psig)": pd60, "Test-rack opening at 60 °F (psig)": pd60 / (1 - r)})
        rows.append(row)
    return pd.DataFrame(rows), l_static
