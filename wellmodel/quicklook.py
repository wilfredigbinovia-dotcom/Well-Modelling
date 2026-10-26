"""Gas lift QuickLook - PROSPER's Matching | QuickLook (User Guide 2.11.5-2.11.6, Tutorial 33a): a gas-lifted well
test checked two ways.

Top down ("measured"): from the measured tubing head pressure the tubing gradient with the total GOR (formation gas plus
lift gas) down to the injection point, then with the produced gas only to the bottom node - the measured tubing
pressure at the valve and the measured flowing bottom-hole pressure. From the measured casing head pressure the
casing gradient down to the injection point - the measured casing pressure and dP across the valve.

Bottom up ("calculated"): the flowing bottom-hole pressure from the IPR at the test liquid rate (with the static
pressure entered, the IPR at that reservoir pressure), the tubing gradient up to the injection point and on to the
wellhead (calculated tubing head pressure). The dP across the injection port for the injected gas (Thornhill-Craver)
is added to the measured tubing pressure at the valve, and the casing gradient taken back to surface: the calculated
casing head pressure. When the model represents the test, the two tubing and the two casing gradients overlay.

Valves (optional): opening pressure at depth Pvo = (Pd - R Pt) / (1 - R) and closing pressure Pvc = Pd (dome at valve
temperature, PROSPER's gauge Winkler & Eads) at the measured tubing pressure; a valve is reported OPEN when the
measured casing pressure is above its closing pressure (PROSPER's convention: a valve between Pvc and Pvo, once open,
stays open). Injection is NOT POSSIBLE at the injection depth when the measured casing pressure is below the measured
tubing pressure there.

Multipoint QuickLook: top down, each valve passes the gas the valve model gives at the measured casing and tubing
pressures (gaslift_system.valve_rates - or Thornhill-Craver through the full port once open, PROSPER's
'Thornhill-Craver' model type), and the lift gas left below it is reduced; gas the valves cannot take is reported as
unaccounted. Each valve's calculated (capacity) rate is also reported. The results are at the injection depth entered
(the orifice), as PROSPER. Validated on the user's PROSPER Tutorial 33b run with Thornhill-Craver valves.

Units: psia, degF, ft MD, STB/d, MMscf/d.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import gaslift as GL
from . import gaslift_system as GLS
from . import vlp as V
from .correlations import CORRELATIONS
from .equipment import HeatCapacities


@dataclass
class QuickLookInput:
    thp: float                    # psia
    tht: float                    # degF (NaN: not measured)
    q_liq: float                  # STB/d
    wc: float                     # fraction
    q_gas_total: float            # MMscf/d, produced plus injected
    qgi: float                    # MMscf/d injected
    chp: float                    # casing head pressure, psia
    inj_md: float                 # injection depth, ft MD
    port_64: float                # orifice / valve port at the injection depth, 64ths in
    tc_derate: float = 0.0        # Thornhill-Craver de-rating, %
    gauges: tuple = ()            # ((md, p psia), ...)
    static_p: float | None = None # static (reservoir) pressure, psia - replaces the IPR's
    valves: tuple = ()            # GLS.GLValve: depth, port, R, dome at 60 F (valve status; multipoint)
    fit_u: bool = True            # use the U value that reproduces the measured THT

    @property
    def q_oil(self):
        return max(self.q_liq * (1 - self.wc), 1e-6)

    @property
    def gor_produced(self):
        return max(self.q_gas_total - self.qgi, 0.0) * 1e6 / self.q_oil

    @property
    def gor_total(self):
        return self.q_gas_total * 1e6 / self.q_oil


@dataclass
class QuickLookResult:
    summary: dict
    profile: pd.DataFrame         # MD, TVD, tubing (top down / bottom up), casing (top down / bottom up), temperature
    valves: pd.DataFrame | None
    injection_possible: bool
    u_used: float
    notes: list = field(default_factory=list)


def _traverse_up(path, fluid, p_bottom, case, gas_seg, T, correlation, p1, p2):
    """Bottom-up tubing traverse (one case) with lift gas gas_seg (per segment, MMscf/d) and temperatures T."""
    corr = CORRELATIONS[correlation]
    n = len(path.md)
    P = np.full(n, np.nan)
    p = np.array([float(p_bottom)])
    P[-1] = p[0]
    qo = max(float(case["q"][0]) * (1 - float(np.clip(case["wc"][0], 0, 0.9999))), 1e-6)
    for i in range(n - 2, -1, -1):
        c = {**case, "gor": case["gor"] + gas_seg[i] * 1e6 / qo}
        if (i + 1) in path.restrictions:
            dp = V._restriction_dp(fluid.state(p, T[[i + 1]], c), path.restrictions[i + 1])
            p = np.maximum(p - dp, 14.7)
        L = path.seg_len[i]
        g0, _ = V._gradient(path, fluid, i, p, T[[i + 1]], c, corr, p1, p2)
        g1, _ = V._gradient(path, fluid, i, np.maximum(p - g0 * L, 14.7), T[[i]], c, corr, p1, p2)
        p = p - 0.5 * (g0 + g1) * L
        if p[0] < 14.7:
            break
        P[i] = p[0]
    return P


def _chp_from_depth(path, setup, pc_target, k, T, flow):
    """Casing head pressure whose casing gradient gives pc_target at node k."""
    f = lambda ps: float(GLS.casing_profile(path, setup, np.array([ps]), T[None, :], flow[None, :])[0, k]) - pc_target
    lo, hi = max(pc_target * 0.5, 15.0), pc_target * 1.2 + 50
    return brentq(f, lo, hi, xtol=1e-4) if f(lo) < 0 < f(hi) else np.nan


def fitted_u(path, fluid, case, gas_seg, tht, hc, sg=0.7, u_max=200.0):
    """U value that gives the measured tubing head temperature (NaN if it cannot)."""
    top = lambda u: GLS._temperature_gas(path, fluid, case, gas_seg[None, :], u, hc, sg)[0, 0]
    lo, hi = top(1e-4), top(u_max)
    if not (min(lo, hi) <= tht <= max(lo, hi)):
        return np.nan
    return brentq(lambda u: top(u) - tht, 1e-4, u_max, xtol=1e-6)


def quicklook(path: V.FlowPath, fluid, ipr_pwf, setup: GLS.GasLiftSetup, inp: QuickLookInput,
              correlation="pe2_style", p1=1.0, p2=1.0, u=8.0, hc: HeatCapacities | None = None,
              multipoint=False) -> QuickLookResult:
    """Single-point (or multipoint) QuickLook. ipr_pwf(q) -> flowing bottom-hole pressure (psia) from the IPR at the
    static pressure to use; setup supplies the lift gas, the annulus (friction) and the dome correction option."""
    hc = hc or HeatCapacities()
    sg = setup.gas.sg
    notes = []
    case = {"q": np.array([inp.q_liq]), "wc": np.array([inp.wc]), "gor": np.array([inp.gor_produced])}
    mid = 0.5 * (path.md[1:] + path.md[:-1])
    k_inj = path.node(inp.inj_md)
    valves = tuple(sorted(inp.valves, key=lambda v: v.md))
    # ---- lift gas in the tubing (per segment) - single point: everything above the injection depth
    gas_seg = np.where(mid < inp.inj_md, inp.qgi, 0.0)
    u_fit = fitted_u(path, fluid, case, gas_seg, inp.tht, hc, sg) if np.isfinite(inp.tht) else np.nan
    u_use = u_fit if (inp.fit_u and np.isfinite(u_fit)) else u
    if inp.fit_u and not np.isfinite(u_fit) and np.isfinite(inp.tht):
        notes.append("The measured THT cannot be reached with any U - the model U is used.")
    valve_q = None
    if multipoint:
        if not valves:
            raise ValueError("Multipoint QuickLook needs the valves.")
        mds = np.array([v.md for v in valves])
        idx = [path.node(m) for m in mds]
        valve_q = np.zeros(len(valves))
        valve_cap = np.zeros(len(valves))
        remaining = inp.qgi
        for j, v in enumerate(valves):
            seg = GLS._seg_from_valves(path, mds, valve_q[None, :])[0] + np.where(mid < mds[j], remaining, 0.0)
            P, T = GLS.multi_traverse(path, fluid, inp.thp, case, seg[None, :], correlation, p1, p2, u_use, hc, sg)
            flow = seg  # lift gas still going down the annulus = gas in the tubing below each depth (to valve j)
            pc = GLS.casing_profile(path, setup, np.array([inp.chp]), T, flow[None, :])
            v_eff = v
            if v.model == "Thornhill-Craver" and v.kind != "Orifice":
                # full port once open (casing above the closing pressure at the valve temperature)
                pvc = float(GLS.dome_at_t(v.pd60, T[0, idx[j]], setup.dome_high_p)) + 14.696 \
                    if np.isfinite(v.pd60) else -np.inf
                v_eff = replace(v, kind="Orifice") if pc[0, idx[j]] > pvc else replace(v, kind="Orifice", port_64=0.0)
            qv, _, _ = GLS.valve_rates((v_eff,), pc[:, [idx[j]]], P[:, [idx[j]]], T[:, [idx[j]]], sg,
                                       setup.dome_high_p, inp.tc_derate)
            valve_cap[j] = float(qv[0, 0])
            valve_q[j] = min(valve_cap[j], remaining)
            remaining -= valve_q[j]
        gas_seg = GLS._seg_from_valves(path, mds, valve_q[None, :])[0]
        if remaining > 1e-3:
            notes.append(f"{remaining:.3f} MMscf/d of the injected gas cannot pass the valves at the measured "
                         "pressures (unaccounted).")
        # results at the injection depth entered (PROSPER: the orifice), not the deepest valve passing gas
    # ---- top down
    T = GLS._temperature_gas(path, fluid, case, gas_seg[None, :], u_use, hc, sg)[0]
    P_td, _ = GLS.multi_traverse(path, fluid, inp.thp, case, gas_seg[None, :], correlation, p1, p2, u_use, hc, sg,
                                 T=T[None, :])
    P_td = P_td[0]
    flow = gas_seg
    pc_td = GLS.casing_profile(path, setup, np.array([inp.chp]), T[None, :], flow[None, :])[0]
    pt_v, pc_v = P_td[k_inj], pc_td[k_inj]
    possible = pc_v > pt_v
    # ---- bottom up
    static = inp.static_p
    pwf = float(ipr_pwf(inp.q_liq))
    P_bu = _traverse_up(path, fluid, pwf, case, gas_seg, T, correlation, p1, p2)
    # injection port dP for the injected gas at the measured tubing pressure (Thornhill-Craver), casing back to surface
    d_in = inp.port_64 / 64.0
    t_v = float(T[k_inj])
    q_port = (inp.qgi if not multipoint else float(valve_q[np.argmin(np.abs(np.array([v.md for v in valves]) -
                                                                           path.md[k_inj]))]))
    # single point: from the measured (top-down) tubing pressure at the valve; multipoint: from the bottom-up one
    # (PROSPER's T33b run: calculated CHP 1429.12 psig against 1432 this way, 1495 from the top-down pressure)
    pt_base = float(P_bu[k_inj]) if multipoint else pt_v
    pc_calc_v = GL.upstream_for_rate(q_port * 1000.0, d_in, pt_base, t_v, sg, inp.tc_derate)
    dp_valve = pc_calc_v - pt_base
    chp_calc = _chp_from_depth(path, setup, pc_calc_v, k_inj, T, flow)
    pc_bu = GLS.casing_profile(path, setup, np.array([chp_calc]), T[None, :], flow[None, :])[0] \
        if np.isfinite(chp_calc) else np.full(len(path.md), np.nan)
    pc_bu = np.where(path.md <= path.md[k_inj] + 1e-6, pc_bu, np.nan)
    pc_td_show = np.where(path.md <= path.md[k_inj] + 1e-6, pc_td, np.nan)
    pc_crit_base = float(pc_v) if multipoint else pc_calc_v     # multipoint: at the measured casing pressure (T33b)
    q_crit = float(GL.thornhill_craver(d_in, pc_crit_base, pc_crit_base * GL.TC_RCRIT, t_v, sg)) / 1000.0 * \
        (1 - inp.tc_derate / 100.0)
    # critical casing head pressure: the CHP whose casing pressure at the port passes the critical rate against the
    # measured tubing pressure (above it the port runs at or beyond its critical capacity)
    chp_crit = np.nan
    if q_crit > 0:
        try:
            f_c = lambda p: float(GL.thornhill_craver(d_in, p, pt_v, t_v, sg)) / 1000.0 * \
                (1 - inp.tc_derate / 100.0) - q_crit
            pc_crit_v = brentq(f_c, pc_calc_v, max(pt_v / GL.TC_RCRIT, pc_calc_v) + 1.0, xtol=1e-4)
            chp_crit = _chp_from_depth(path, setup, pc_crit_v, k_inj, T, flow)
        except ValueError:
            chp_crit = np.nan
    # produced gas: solution and free GOR at the bubble point
    rs_b = None
    try:                             # solution GOR at the bubble point: the input Rsb with the PVT match on Rs (PROSPER)
        rs_b = float(np.atleast_1d(fluid.m._m("rs").apply(np.array([float(fluid.inp.rsb)])))[0])
    except Exception:              # noqa: BLE001 - fluids without a black-oil solution GOR
        rs_b = None
    gor_sol = min(inp.gor_produced, rs_b) if rs_b is not None else inp.gor_produced
    p_res = static if static is not None else np.nan
    summ = {
        "Flowing bottom-hole pressure, IPR (psia)": pwf,
        "Flowing bottom-hole pressure, top down (psia)": float(P_td[-1]),
        "Static bottom-hole pressure (psia)": p_res,
        "Tubing pressure at valve (psia)": float(pt_v),
        "Casing pressure at valve (psia)": float(pc_v),
        "Temperature at valve (°F)": t_v,
        "Gas oil ratio (scf/STB)": gor_sol,
        "GOR free (scf/STB)": inp.gor_produced - gor_sol,
        "dP across valve, calculated (psi)": float(dp_valve),
        "dP across valve, measured (psi)": float(pc_v - pt_v),
        "Calculated casing head pressure (psia)": float(chp_calc),
        "Calculated tubing head pressure (psia)": float(P_bu[0]),
        "Draw down (psi)": (p_res - pwf) if static is not None else np.nan,
        "Equivalent calculated PI (STB/d/psi)": inp.q_liq / (p_res - pwf) if static is not None and p_res > pwf
        else np.nan,
        "Calculated critical flow rate (MMscf/d)": q_crit,
        "Percentage of critical flow rate (%)": 100 * q_port / q_crit if q_crit > 0 else np.nan,
        "Critical casing head pressure (psia)": float(chp_crit),
        "Fitted thermal coefficient (BTU/h/ft2/F)": u_fit,
        "Injection depth (ft MD)": float(path.md[k_inj]),
        "Injection": "possible" if possible else "NOT POSSIBLE (casing below tubing pressure)",
    }
    for md_g, p_g in inp.gauges:
        summ[f"Gauge at {md_g:,.0f} ft: measured (psia)"] = p_g
        summ[f"Gauge at {md_g:,.0f} ft: top down (psia)"] = float(np.interp(md_g, path.md, P_td))
    if not possible:
        notes.append(f"Injection is not possible at {path.md[k_inj]:,.0f} ft MD: the measured casing pressure "
                     f"({pc_v:,.1f} psia) is below the tubing pressure ({pt_v:,.1f} psia) - try the next valve up.")
    vt = None
    if valves:
        rows = []
        mds = np.array([v.md for v in valves])
        for j, v in enumerate(valves):
            kk = path.node(v.md)
            pt, pc, tv = float(P_td[kk]), float(pc_td[kk]), float(T[kk])
            row = {"Valve": j + 1, "Type": v.kind, "Measured depth (ft)": v.md, "TVD (ft)": float(path.tvd[kk]),
                   "Tubing pressure (psia)": pt, "Casing pressure (psia)": pc, "Temperature (°F)": tv}
            if v.kind == "Tubing" and np.isfinite(v.pd60):
                pbt = float(GLS.dome_at_t(v.pd60, tv, setup.dome_high_p)) + 14.696
                r = min(max(v.r, 0.0), 0.95)
                pvo = (pbt - r * pc) / (1 - r)          # tubing pressure to open (tubing-sensitive)
                row.update({"Opening pressure (psia)": pvo, "Closing pressure (psia)": pbt,
                            "Status": "OPEN" if (pt > pvo and pc > pt) else "CLOSED"})
            elif v.kind == "Valve" and np.isfinite(v.pd60):
                pvc = float(GLS.dome_at_t(v.pd60, tv, setup.dome_high_p)) + 14.696
                r = min(max(v.r, 0.0), 0.95)
                pvo = (pvc - r * pt) / (1 - r)
                row.update({"Opening pressure (psia)": pvo, "Closing pressure (psia)": pvc,
                            "Status": "OPEN" if pc > pvc else "CLOSED"})
            else:
                row.update({"Opening pressure (psia)": np.nan, "Closing pressure (psia)": np.nan,
                            "Status": "OPEN" if pc > pt else "NEG DP"})
            if valve_q is not None:
                row["Gas rate (MMscf/d)"] = float(valve_q[j])
                row["Calculated gas rate (MMscf/d)"] = float(valve_cap[j])
                row["Model"] = v.model or ("Thornhill-Craver" if v.kind == "Orifice" else "Throttling")
            rows.append(row)
        vt = pd.DataFrame(rows)
        if multipoint:
            n_on = int(np.sum(valve_q > 1e-6))
            summ["Multipoint flag"] = "No Injection" if n_on == 0 else ("Single" if n_on == 1 else "Multiple")
    prof = pd.DataFrame({"MD (ft)": path.md, "TVD (ft)": path.tvd, "Tubing pressure, top down (psia)": P_td,
                         "Tubing pressure, bottom up (psia)": P_bu, "Casing pressure, top down (psia)": pc_td_show,
                         "Casing pressure, bottom up (psia)": pc_bu, "Temperature (°F)": T,
                         "Lift gas in tubing (MMscf/d)": np.append(gas_seg, 0.0)})
    return QuickLookResult(summ, prof, vt, bool(possible), float(u_use), notes)
