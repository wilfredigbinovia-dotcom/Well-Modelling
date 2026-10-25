"""Gas-lifted producer in the main model - PROSPER's Options | Artificial lift = Gas lift (continuous) with
System | Gas Lift Data. Used by the system calculation, the gradient traverse and the VFP tables.

Gas lift methods (PROSPER User Guide 2.8.1)
-------------------------------------------
Fixed depth of injection    all the lift gas enters at one depth; the casing pressure is assumed sufficient.
Optimum depth of injection  the deepest point where casing pressure >= tubing pressure + dP across the valve, down to
                            a maximum depth (gaslift.lifted_traverse).
Valve depths specified      the same deepest point, then the injection moves up to the deepest specified valve at
                            or above it - all the gas enters there. No valve above it: no injection (natural flow).
Multipoint                  the casing head pressure is the boundary condition; every valve passes the gas its port
                            allows at the prevailing casing and tubing pressures, so the lift-gas rate is a result.
                            Valve model (valve_rates): orifice = Thornhill-Craver through the port; casing-sensitive
                            IPO valve closed at or below Pvc = Pd (dome at valve temperature), no gas up to its opening
                            pressure Pvo = (Pd - R Pt) / (1 - R), then throttling: the fraction 0.001 (Pc - Pvo) / R of the full-port
                            Thornhill-Craver gas, up to fully open (fitted to PROSPER's multipoint VLP, T23).

Lift-gas rate: an injected gas rate (MMscf/d) or a GLR injected (scf per STB of liquid).

Annulus: the casing pressure is a gas column from the casing head pressure, with the gas 10 F cooler than the flowing
tubing and its Z from Standing's natural-gas pseudo-criticals (DPR) - PROSPER's casing pressures at the valves (Tutorial 23
multipoint, 220 valves) mean 0.1 / rms 1.0 psi, T33a QuickLook within 0.7 psi; with the produced gas's Z and the gas
0.18 (T_tubing - T_formation) cooler (the earlier fit) rms 3.2 psi with a trend with the rate. "Friction loss in annulus" adds the friction of the lift gas flowing down the
tubing-casing annulus (Moody/Chen on the annulus hydraulic diameter Dc - Dto); below the injection point the gas
is static.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import gaslift as GL
from . import vlp as V
from .correlations import CORRELATIONS, friction_factor, CP_TO_LBFTS
from .equipment import HeatCapacities

METHODS = ("Fixed depth of injection", "Optimum depth of injection", "Valve depths specified", "Multipoint")
RATE_MODES = ("Injected gas rate", "GLR injected")
ANNULUS = ("No friction loss in annulus", "Friction loss in annulus")
VALVE_KINDS = ("Valve", "Tubing", "Orifice")   # casing-sensitive IPO, tubing-sensitive, orifice


@dataclass
class GLValve:
    md: float
    kind: str = "Valve"          # "Valve" (casing-sensitive IPO), "Tubing" (tubing-sensitive) or "Orifice"
    port_64: float = 16.0
    r: float = 0.0               # port / bellows area (valves)
    pd60: float = np.nan         # dome pressure at 60 F, psig (valves; multipoint only)
    n: int = 1                   # valves in the mandrel (PROSPER "Number of valves"): n x the gas of one


@dataclass
class Annulus:
    """Tubing-casing annulus per flow-path segment (NaN where the geometry is unknown)."""
    d_h: np.ndarray              # ft
    area: np.ndarray             # ft2
    rough: np.ndarray            # ft

    @property
    def complete(self):
        return bool(np.all(np.isfinite(self.d_h)) and np.all(self.d_h > 0))


def annulus_geometry(path: V.FlowPath, secs: pd.DataFrame) -> Annulus:
    """Annulus between tubing OD and casing ID on each segment of the flow path (from the downhole sections)."""
    mid = 0.5 * (path.md[1:] + path.md[:-1])
    dh, ar, rg = (np.full(len(mid), np.nan) for _ in range(3))
    for j, m in enumerate(mid):
        rows = secs[(secs["MD top (ft)"] < m) & (secs["MD bottom (ft)"] >= m)]
        if not len(rows):
            continue
        r = rows.iloc[0]
        do, dc = r.get("Tubing OD (in)"), r.get("Casing ID (in)")
        if r["Type"] == "Tubing" and pd.notna(do) and pd.notna(dc) and dc > do:
            dh[j] = (dc - do) / 12.0
            ar[j] = np.pi / 4 * ((dc / 12.0) ** 2 - (do / 12.0) ** 2)
            e = r.get("Casing roughness (in)", np.nan)
            rg[j] = (e if pd.notna(e) else (r["Roughness (in)"] if pd.notna(r.get("Roughness (in)")) else 0.0006)) / 12
    return Annulus(dh, ar, rg)


@dataclass
class GasLiftSetup:
    method: str = METHODS[1]
    gas: GL.LiftGas = field(default_factory=GL.LiftGas)
    rate_mode: str = RATE_MODES[0]
    qgi: float = 1.0             # MMscf/d
    glr_inj: float = 500.0       # scf/STB liquid
    p_casing: float = 1214.7     # casing head (injection) pressure, psia
    dp_valve: float = 100.0      # psi
    max_md: float = 1e9          # optimum depth: maximum depth of injection, ft MD
    fixed_md: float = np.nan     # fixed depth of injection, ft MD
    valves: tuple = ()           # GLValve (valve depths specified / multipoint)
    annulus_friction: bool = False
    annulus: Annulus | None = None
    dome_high_p: bool = True     # Winkler-Eads dome correction above 1200 psig
    annulus_t_offset: float = 10.0  # annulus gas temperature = flowing tubing T - this (degF), with the lift gas on
    #                                 Standing's natural-gas pseudo-criticals (gaslift.LIFT_GAS_Z): PROSPER's casing
    #                                 pressures at the valves of its Tutorial 23 multipoint runs (220 valves, 1000-3500
    #                                 STB/d, 1360-1510 psig) mean 0.1, rms 1.0 psi, with no trend with the rate, and its
    #                                 T33a QuickLook casing pressures within 0.7 psi
    annulus_t_frac: float = 0.0  # optional extra cooling: - frac x (tubing T - formation T)
    tc_derate: float = 0.0       # Thornhill-Craver de-rating, %
    coiled: object = None        # coiled.CoiledTubing: coiled tubing gas lift (gas down the coiled tubing, production
    #                              up the coiled tubing - tubing annulus); methods Fixed depth (= PROSPER "Specified
    #                              injection depth") and Optimum depth only

    @property
    def ct_bottom(self):
        """Coiled tubing end (ft MD): the specified depth, or the maximum depth of injection."""
        return float(self.fixed_md if self.method == METHODS[0] else self.max_md)

    def flow_path(self, path: V.FlowPath) -> V.FlowPath:
        """The production flow path: with coiled tubing, the annulus above its end."""
        if self.coiled is None:
            return path
        from . import coiled as CT
        return CT.annular_path(path, self.coiled, min(self.ct_bottom, float(path.md[-1])))

    @property
    def multipoint(self):
        return self.method == METHODS[3]

    def extra_md(self):
        """Depths the flow path needs as nodes."""
        out = [v.md for v in self.valves] if self.method in METHODS[2:] else []
        if self.coiled is not None and self.method in METHODS[2:]:
            raise ValueError("Coiled tubing gas lift: use the specified (fixed) or optimum injection depth.")
        if self.method == METHODS[0] and np.isfinite(self.fixed_md):
            out.append(self.fixed_md)
        if self.method == METHODS[1] and np.isfinite(self.max_md):
            out.append(self.max_md)
        return out

    def row_inputs(self, params: dict, q) -> dict:
        """Per-rate lift inputs for one case: lift gas (MMscf/d), casing head pressure, valve dP, injection depth.
        params may override them: qgi (MMscf/d), glr_inj (scf/STB), pcs (psia), dp_valve (psi), inj_md (ft)."""
        q = np.atleast_1d(np.asarray(q, float))
        if "glr_inj" in params or ("qgi" not in params and self.rate_mode == RATE_MODES[1]):
            glr = float(params.get("glr_inj", self.glr_inj))
            qgi = glr * np.maximum(q, 0.0) / 1e6
        else:
            qgi = np.full(q.shape, float(params.get("qgi", self.qgi)))
        return {"_qgi": qgi, "_pcs": np.full(q.shape, float(params.get("pcs", self.p_casing))),
                "_dpv": np.full(q.shape, float(params.get("dp_valve", self.dp_valve))),
                "_injmd": np.full(q.shape, float(params.get("inj_md", self.fixed_md)))}


GL_VARIABLES = {
    "qgi": "Gas lift gas injection rate (MMscf/d)",
    "glr_inj": "GLR injected (scf/STB)",
    "pcs": "Casing head pressure (psia)",
    "dp_valve": "dP across valve (psi)",
    "inj_md": "Injection depth (ft MD)",
}


def variables_for(setup: GasLiftSetup) -> dict:
    """Sensitivity variables that apply to the gas lift method."""
    m = setup.method
    if m == METHODS[3]:
        return {"pcs": GL_VARIABLES["pcs"]}
    out = {"qgi": GL_VARIABLES["qgi"], "glr_inj": GL_VARIABLES["glr_inj"]}
    if setup.coiled is not None:      # the coiled tubing end fixes the annulus: no injection-depth sensitivity
        out.update({"pcs": "Coiled tubing head (injection) pressure (psia)", "dp_valve": GL_VARIABLES["dp_valve"]})
        return out
    if m == METHODS[0]:
        out["inj_md"] = GL_VARIABLES["inj_md"]
    else:
        out.update({"pcs": GL_VARIABLES["pcs"], "dp_valve": GL_VARIABLES["dp_valve"]})
    return out


# ---------------------------------------------------------------- casing pressure
def casing_profile(path: V.FlowPath, setup: GasLiftSetup, p_surf, T, flow):
    """Casing pressure (cases x nodes) from the casing head pressure p_surf (cases), with the lift gas at the
    tubing temperature T (cases x nodes) and flow (cases x segments, MMscf/d) going down the annulus."""
    gas = setup.gas
    p = np.atleast_1d(np.asarray(p_surf, float)).copy()
    T = np.asarray(T, float)
    T = T - setup.annulus_t_frac * (T - path.t_form[None, :]) - setup.annulus_t_offset
    nc, n = T.shape
    p = np.broadcast_to(p, (nc,)).copy()
    out = np.empty((nc, n))
    out[:, 0] = p
    ann = setup.annulus if setup.annulus_friction else None
    gin = setup.gas.inp
    for i in range(n - 1):
        dz, L = path.tvd[i + 1] - path.tvd[i], path.seg_len[i]

        def grad(pp, tt):
            z, bg, rho, mu = GL.bo.gas_props(gin, pp, tt)
            g = rho * dz / max(L, 1e-9) / 144.0
            if ann is not None and np.isfinite(ann.d_h[i]):
                v = flow[:, i] * 1e6 / 86400.0 * bg / ann.area[i]
                re = rho * np.abs(v) * ann.d_h[i] / (mu * CP_TO_LBFTS)
                f = friction_factor(re, ann.rough[i] / ann.d_h[i])
                g = g - f * rho * v * np.abs(v) / (2 * 32.174 * ann.d_h[i]) / 144.0
            return g

        g0 = grad(p, T[:, i])
        g1 = grad(p + g0 * L, T[:, i + 1])
        p = p + 0.5 * (g0 + g1) * L
        out[:, i + 1] = p
    return out


def _flow_single(path, qgi, inj_md):
    """Annulus gas flow per segment for single-point injection at inj_md (cases)."""
    mid = 0.5 * (path.md[1:] + path.md[:-1])
    return np.where(mid[None, :] < np.asarray(inj_md, float)[:, None], np.asarray(qgi, float)[:, None], 0.0)


# ---------------------------------------------------------------- valve performance
def dome_at_t(pd60_psig, t_f, high_pressure=True):
    """Dome pressure at valve temperature (psig) from the 60 F charge (psig) - inverse of gaslift.dome_at_60."""
    pd60 = np.asarray(pd60_psig, float)
    api = pd60 / GL.ct_api(t_f)
    if not high_pressure:
        return api
    return np.where(api > 1200.0, GL.dome_at_t_winkler(pd60, t_f, atm=GL.PROSPER_ATM), api)


THROTTLE_K = (0.00117, 0.00136)   # a throttling valve's opening a = (K0 - K1 R) (Pc - Pvo) / R - an open flow area
#                       growing with the pressure above the opening pressure in proportion to the bellows area
#                       (Ab = Ap / R); with critical flow it passes the fraction min(a, 1) of its full-port
#                       Thornhill-Craver gas. Fitted to the user's PROSPER multipoint VLP runs on the Tutorial 23 well
#                       (Camco R-20 at 12-24/64, McMurry R-1 / Otis NM16R 16/64, casing 1360-1560 psig): K = 0.00108 at
#                       R 0.066, 0.00103 at 0.103, 0.00097 at 0.147, 0.00095 at 0.165.
THROTTLE_SUB = (1.0, 0.10, 2.5)   # subcritical flow (Pt / Pc above the critical ratio rc, d = Pt/Pc - rc): PROSPER
#                       keeps the valve throttling past a = 1 - at a = 1 it passes x1 = 1 - 1.0 d of the full-port gas,
#                       fully open only from a = 1 + 2.5 d; below a = x1 - 0.10 the fraction is a; straight lines
#                       between (fitted to the user's A2 casing-pressure sweep 1400-1560 psig at 2000 STB/d and the
#                       multipoint runs: valve rates rms 0.038 MMscf/d over 270 valve results, the sweep within 0.026,
#                       against 0.065 / 0.24 with min(a, 1)).


def throttle_fraction(a, ratio):
    """PROSPER-style fraction of the full-port gas a throttling valve passes at opening a and Pt / Pc ratio."""
    a = np.maximum(np.asarray(a, float), 0.0)
    ratio = np.asarray(ratio, float)
    c1, w, c_open = THROTTLE_SUB
    dd = np.maximum(ratio - GL.TC_RCRIT, 0.0)
    x1 = 1.0 - c1 * dd
    a0 = x1 - w
    ao = 1.0 + c_open * dd
    x = np.where(a <= a0, a, np.where(a <= 1.0, a0 + (x1 - a0) * (a - a0) / np.maximum(1.0 - a0, 1e-9),
                 np.where(a < ao, x1 + (1.0 - x1) * (a - 1.0) / np.maximum(ao - 1.0, 1e-9), 1.0)))
    return np.where(dd > 0, x, np.minimum(a, 1.0))


def valve_rates(valves, pc, pt, t, sg, high_pressure=True, derate=0.0):
    """Gas rate (MMscf/d), opening fraction (rate / fully open rate) and status for every valve (cases x valves).
    Orifice: Thornhill-Craver through the port. Casing-sensitive IPO valve (dome Pd at valve temperature, PROSPER's
    gauge-pressure Winkler & Eads): closed while Pc <= Pvc = Pd; no gas until the casing reaches the opening pressure
    Pvo = (Pd - R Pt) / (1 - R); above it the valve throttles, passing the fraction throttle_fraction((K0 - K1 R) (Pc - Pvo) / R) of
    the full-port Thornhill-Craver gas (an open area proportional to the bellows area and the pressure above Pvo)
    until it is fully open. Fitted to PROSPER's multipoint VLP (Tutorial 23), where
    valves between Pvc and Pvo are reported as throttling with no gas."""
    q = np.zeros(pc.shape)
    x = np.zeros(pc.shape)
    status = np.empty(pc.shape, dtype=object)
    for k, v in enumerate(valves):
        pck, ptk, tk = pc[:, k], pt[:, k], t[:, k]
        neg = pck <= ptk
        full = max(int(v.n), 1) * GL.thornhill_craver(v.port_64 / 64.0, pck, np.minimum(ptk, pck), tk, sg) * \
            (1 - derate / 100.0) / 1000.0
        if v.kind == "Orifice" or not np.isfinite(v.pd60):
            qk = full
            closed = np.zeros(pck.shape, bool)
        elif v.kind == "Tubing":
            # tubing-sensitive: the tubing pressure acts on the bellows, the casing on the port - opens when
            # Pt (1 - R) + Pc R reaches the dome pressure at temperature, throttling above it as the IPO valve
            pbt = dome_at_t(v.pd60, tk, high_pressure) + 14.696
            r = min(max(v.r, 0.0), 0.95)
            pvo_t = (pbt - r * pck) / (1 - r)
            qk = full * throttle_fraction((THROTTLE_K[0] - THROTTLE_K[1] * r) * np.maximum(ptk - pvo_t, 0.0)
                                          / max(r, 0.005), np.minimum(ptk, pck) / np.maximum(pck, 1e-9))
            closed = ptk <= pvo_t
        else:
            pvc = dome_at_t(v.pd60, tk, high_pressure) + 14.696
            r = min(max(v.r, 0.0), 0.95)
            pvo = (pvc - r * ptk) / (1 - r)
            qk = full * throttle_fraction((THROTTLE_K[0] - THROTTLE_K[1] * r) * np.maximum(pck - pvo, 0.0)
                                          / max(r, 0.005), np.minimum(ptk, pck) / np.maximum(pck, 1e-9))
            closed = pck <= pvc
        qk = np.where(neg | closed, 0.0, qk)
        xk = np.where(full > 0, qk / np.maximum(full, 1e-12), 0.0)
        q[:, k], x[:, k] = qk, xk
        crit = ptk / np.maximum(pck, 1e-9) <= GL.TC_RCRIT
        status[:, k] = np.where(neg, "Neg dP", np.where(closed, "Closed", np.where(
            xk < 0.999, "Throttle", np.where(crit, "Open Cri", "Open Sub"))))
    return q, x, status


# ---------------------------------------------------------------- traverses
def _temperature_gas(path, fluid, case, gas_seg, u, hc, sg):
    """Flowing temperature with lift gas gas_seg (cases x segments, MMscf/d) added to the heat balance."""
    w, cp = fluid.mass_cp(case, hc)
    n = len(path.md)
    T = np.empty((w.size, n))
    T[:, -1] = path.t_form[-1]
    for i in range(n - 2, -1, -1):
        L = path.seg_len[i]
        wg = gas_seg[:, i] * 1e6 * GL.RHO_AIR_SC * sg / 24.0
        ww = w + wg
        cc = (w * cp + wg * hc.gas) / np.maximum(ww, 1e-12)
        a = u * np.pi * path.d_ht[i] / np.maximum(ww * cc, 1e-9)
        tfb, tft = path.t_form[i + 1], path.t_form[i]
        g = (tft - tfb) / L
        t0 = T[:, i + 1]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            tn = tfb + g * L - g / a + (t0 - tfb + g / a) * np.exp(-a * L)
        T[:, i] = np.where(a * L < 1e-9, t0, tn)
    return T


def multi_traverse(path, fluid, top_p, case, gas_seg, correlation, p1, p2, u, hc, sg, T=None):
    """Top-down traverse with lift gas gas_seg (cases x segments, MMscf/d) in the tubing."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        corr = CORRELATIONS[correlation]
        q, wc = case["q"], np.clip(case["wc"], 0, 0.9999)
        qo = np.maximum(q * (1 - wc), 1e-6)
        if T is None:
            T = _temperature_gas(path, fluid, case, gas_seg, u, hc, sg)
        nc, n = T.shape
        P = np.empty((nc, n))
        p = np.maximum(np.asarray(top_p, float) + np.zeros(nc), 14.7)
        P[:, 0] = p
        for i in range(n - 1):
            c = {**case, "gor": case["gor"] + gas_seg[:, i] * 1e6 / qo}
            L = path.seg_len[i]
            g0, _ = V._gradient(path, fluid, i, p, T[:, i], c, corr, p1, p2)
            g1, _ = V._gradient(path, fluid, i, np.maximum(p + g0 * L, 14.7), T[:, i + 1], c, corr, p1, p2)
            p = np.maximum(p + 0.5 * (g0 + g1) * L, 14.7)
            if (i + 1) in path.restrictions:
                st = fluid.state(p, T[:, i + 1], c)
                p = p + V._restriction_dp(st, path.restrictions[i + 1])
            P[:, i + 1] = p
        return P, T


@dataclass
class LiftState:
    P: np.ndarray                # tubing pressure (cases x nodes)
    T: np.ndarray
    pc: np.ndarray | None        # casing pressure (cases x nodes)
    qgi: np.ndarray              # total lift gas, MMscf/d (cases)
    inj_md: np.ndarray           # deepest injection point (cases), NaN if none
    note: np.ndarray             # per case
    valve_q: np.ndarray | None = None      # multipoint: cases x valves, MMscf/d
    valve_status: np.ndarray | None = None
    valve_pt: np.ndarray | None = None
    valve_pc: np.ndarray | None = None
    valve_t: np.ndarray | None = None
    valve_x: np.ndarray | None = None
    flag: np.ndarray | None = None         # "Single" / "Multiple" / "No Injection"
    inj_p: np.ndarray | None = None        # tubing pressure at the injection point (single point; the node
    #                                        profile kinks there, so interpolating P between nodes is not exact)


def _seg_from_valves(path, mds, g):
    """Lift gas in each tubing segment (cases x segments) from valve rates g (cases x valves)."""
    mid = 0.5 * (path.md[1:] + path.md[:-1])
    below = (np.asarray(mds)[None, :] > mid[:, None]).astype(float)       # segments x valves
    return g @ below.T


def lift_traverse(path: V.FlowPath, fluid, top_p, case: dict, setup: GasLiftSetup, correlation="hagedorn_brown",
                  p1=1.0, p2=1.0, u=8.0, hc: HeatCapacities | None = None, passes=2, n_iter=40) -> LiftState:
    """Gas-lifted traverse for every case. case: q, wc, gor and the lift columns from GasLiftSetup.row_inputs
    (_qgi, _pcs, _dpv, _injmd). The path should contain the valve / injection depths as nodes."""
    hc = hc or HeatCapacities()
    c = dict(case)
    c["_top"] = top_p
    c = V._broadcast_case(c)
    top = c.pop("_top")
    qgi, pcs, dpv, injmd = (c.pop(k) for k in ("_qgi", "_pcs", "_dpv", "_injmd"))
    nc = top.size
    sg = setup.gas.sg
    m = setup.method
    note = np.full(nc, "", dtype=object)
    if setup.coiled is not None:
        return _coiled(setup.flow_path(path), fluid, top, c, setup, qgi, pcs, dpv, injmd, correlation, p1, p2, u, hc,
                       passes)
    if m == METHODS[3]:
        return _multipoint(path, fluid, top, c, setup, pcs, correlation, p1, p2, u, hc, n_iter)

    def pc_fn(T, inj_guess):
        flow = _flow_single(path, qgi, inj_guess)
        return casing_profile(path, setup, pcs, T, flow)

    if m == METHODS[0]:
        if not np.all(np.isfinite(injmd)):
            raise ValueError("Enter the depth of injection.")
        r = GL.lifted_traverse(path, fluid, top, c, qgi, correlation, p1, p2, u, hc, fixed_md=injmd, gas_sg=sg,
                               passes=1)
        inj = r.inj_md
        inj_p = r.inj_p
        pc = None
    else:
        max_md = setup.max_md if m == METHODS[1] else max(v.md for v in setup.valves)
        r = GL.lifted_traverse(path, fluid, top, c, qgi, correlation, p1, p2, u, hc, pc=pc_fn, dp_valve=dpv,
                               max_md=max_md, gas_sg=sg, passes=passes)
        inj = r.inj_md.copy()
        inj_p = r.inj_p.copy()
        none = r.no_injection.copy()
        if m == METHODS[2]:
            mds = np.sort(np.array([v.md for v in setup.valves], float))
            k = np.searchsorted(mds, inj + 1e-6, side="right") - 1
            none |= k < 0
            snapped = np.where(k >= 0, mds[np.clip(k, 0, None)], np.nan)
            move = (~none) & (np.abs(snapped - inj) > 0.5)
            if np.any(move):
                sub = {kk: v[move] for kk, v in c.items()}
                r2 = GL.lifted_traverse(path, fluid, top[move], sub, qgi[move], correlation, p1, p2, u, hc,
                                        fixed_md=snapped[move], gas_sg=sg, passes=1)
                r.P[move], r.T[move] = r2.P, r2.T
                inj_p[move] = r2.inj_p
            inj = np.where(none, np.nan, snapped)
        if np.any(none):
            sub = {kk: v[none] for kk, v in c.items()}
            P0, T0 = V.traverse(path, fluid, top[none], sub, correlation, p1, p2, u, hc)
            r.P[none], r.T[none] = P0, T0
            inj = np.where(none, np.nan, inj)
            note = np.where(none, "no injection - casing pressure below tubing pressure + dP"
                            + (" at every valve" if m == METHODS[2] else " at the top"), note)
        pc = casing_profile(path, setup, pcs, r.T, _flow_single(path, np.where(none, 0.0, qgi),
                                                                np.nan_to_num(inj, nan=0.0)))
        qgi = np.where(none, 0.0, qgi)
        inj_p = np.where(none, np.nan, inj_p)
    return LiftState(r.P, r.T, pc, qgi, inj, note, inj_p=inj_p)


def _coiled(path, fluid, top, c, setup, qgi, pcs, dpv, injmd, correlation, p1, p2, u, hc, passes):
    """Coiled tubing gas lift on the annular path: the lift gas from the coiled tubing head pressure down the coiled
    tubing (coiled.gas_pressure). Specified depth: injection at the coiled tubing end when the gas pressure there is at
    least the tubing pressure + dP, else none (PROSPER). Optimum: the deepest such point down to the end."""
    from . import coiled as CT
    ct = setup.coiled
    nc = top.size
    sg = setup.gas.sg
    note = np.full(nc, "", dtype=object)

    def pc_fn(T, inj_guess):
        return CT.gas_pressure(path, setup.gas, ct, pcs, T, qgi, inj_guess)

    if setup.method == METHODS[0]:
        if not np.all(np.isfinite(injmd)):
            raise ValueError("Enter the coiled tubing specified depth.")
        r = GL.lifted_traverse(path, fluid, top, c, qgi, correlation, p1, p2, u, hc, fixed_md=injmd, gas_sg=sg,
                               passes=passes)
        pc = pc_fn(r.T, injmd)
        pc_inj = np.array([np.interp(injmd[k], path.md, pc[k]) for k in range(nc)])
        none = pc_inj < r.inj_p + dpv
        inj, inj_p = injmd.astype(float).copy(), r.inj_p.copy()
    else:
        r = GL.lifted_traverse(path, fluid, top, c, qgi, correlation, p1, p2, u, hc, pc=pc_fn, dp_valve=dpv,
                               max_md=setup.max_md, gas_sg=sg, passes=passes)
        none = r.no_injection.copy()
        inj, inj_p = r.inj_md.copy(), r.inj_p.copy()
    if np.any(none):
        sub = {kk: v[none] for kk, v in c.items()}
        P0, T0 = V.traverse(path, fluid, top[none], sub, correlation, p1, p2, u, hc)
        r.P[none], r.T[none] = P0, T0
        note = np.where(none, "no injection - coiled tubing gas pressure below tubing pressure + dP"
                        + (" at the injection depth" if setup.method == METHODS[0] else " at the top"), note)
    qgi = np.where(none, 0.0, qgi)
    inj = np.where(none, np.nan, inj)
    inj_p = np.where(none, np.nan, inj_p)
    pc = pc_fn(r.T, np.nan_to_num(inj, nan=0.0))
    return LiftState(r.P, r.T, pc, qgi, inj, note, inj_p=inj_p)


def _multipoint(path, fluid, top, c, setup, pcs, correlation, p1, p2, u, hc, n_iter):
    valves = sorted(setup.valves, key=lambda v: v.md)
    if not valves:
        raise ValueError("Multipoint needs the gas lift valves (transfer them from the design or enter them).")
    mds = np.array([v.md for v in valves])
    idx = np.array([path.node(mm) for mm in mds])
    nc, nv = top.size, len(valves)
    sg = setup.gas.sg
    # start from the most gas the valves can pass (tubing at half the casing pressure) and come down to the
    # largest self-consistent injection: the unloaded well. Starting from no gas would find the trivial
    # "no injection" state whenever the natural-flow tubing pressure exceeds the casing pressure.
    T0 = _temperature_gas(path, fluid, c, np.zeros((nc, len(path.seg_len))), u, hc, sg)
    pc0 = casing_profile(path, setup, pcs, T0, np.zeros((nc, len(path.seg_len))))[:, idx]
    g, _, _ = valve_rates(valves, pc0, 0.5 * pc0, T0[:, idx], sg, setup.dome_high_p, setup.tc_derate)
    seg = _seg_from_valves(path, mds, g)
    P, T = multi_traverse(path, fluid, top, c, seg, correlation, p1, p2, u, hc, sg)
    w = 0.5
    for it in range(n_iter):
        pc = casing_profile(path, setup, pcs, T, seg)
        qn, x, status = valve_rates(valves, pc[:, idx], P[:, idx], T[:, idx], sg, setup.dome_high_p, setup.tc_derate)
        change = np.max(np.abs(qn - g) / np.maximum(np.abs(g).sum(1, keepdims=True), 0.05))
        g = g + w * (qn - g)
        seg = _seg_from_valves(path, mds, g)
        P, T = multi_traverse(path, fluid, top, c, seg, correlation, p1, p2, u, hc, sg)
        if change < 2e-4:
            break
        if it == 15:
            w = 0.3
    pc = casing_profile(path, setup, pcs, T, seg)
    qn, x, status = valve_rates(valves, pc[:, idx], P[:, idx], T[:, idx], sg, setup.dome_high_p, setup.tc_derate)
    dead = qn.sum(1) < 1e-6          # injection collapsed (still decaying when the iterations ran out): natural flow
    if np.any(dead):
        g[dead] = 0.0
        seg = _seg_from_valves(path, mds, g)
        sub = {kk: v[dead] for kk, v in c.items()}
        P0, T0 = multi_traverse(path, fluid, top[dead], sub, seg[dead], correlation, p1, p2, u, hc, sg)
        P[dead], T[dead] = P0, T0
        pc = casing_profile(path, setup, pcs, T, seg)
        qn, x, status = valve_rates(valves, pc[:, idx], P[:, idx], T[:, idx], sg, setup.dome_high_p,
                                    setup.tc_derate)
        qn[dead] = 0.0
    inj_on = qn > 1e-6
    n_on = inj_on.sum(1)
    flag = np.where(n_on == 0, "No Injection", np.where(n_on == 1, "Single", "Multiple"))
    deepest = np.array([mds[np.where(row)[0][-1]] if row.any() else np.nan for row in inj_on])
    note = np.where(n_on == 0, "no injection - every valve closed at this casing pressure", "")
    return LiftState(P, T, pc, g.sum(1), deepest, note, g, status, P[:, idx], pc[:, idx], T[:, idx], x, flag)


def valve_table(setup: GasLiftSetup, st: LiftState, k=0) -> pd.DataFrame:
    """Multipoint valve results for case k."""
    valves = sorted(setup.valves, key=lambda v: v.md)
    rows = []
    for j, v in enumerate(valves):
        pvc = (float(dome_at_t(v.pd60, st.valve_t[k, j], setup.dome_high_p)) + 14.696
               if v.kind in ("Valve", "Tubing") and np.isfinite(v.pd60) else np.nan)
        rows.append({"Valve": j + 1, "Type": v.kind, "Measured depth (ft)": v.md, "Port (64ths in)": v.port_64,
                     "Status": st.valve_status[k, j], "Injection rate (MMscf/d)": st.valve_q[k, j],
                     "Casing pressure (psia)": st.valve_pc[k, j], "Tubing pressure (psia)": st.valve_pt[k, j],
                     "Valve temperature (°F)": st.valve_t[k, j],
                     "Dome pressure at valve T (psia)": pvc,
                     "Opening fraction": st.valve_x[k, j]})
    return pd.DataFrame(rows)


def system_dome(v, valve_type: str) -> float:
    """Dome pressure at 60 F (psig) for the well model's valve from a design valve row. A proportional valve goes in
    as a tubing-sensitive valve with R = k of its trim and the dome that makes it open at its PTO: dome at
    temperature = closing pressure at depth (PTO (1 - k) + Pc k), without the spring."""
    if v.kind == "Orifice":
        return np.nan
    if valve_type == GL.VALVE_TYPES[2] and np.isfinite(v.pvc):
        return float(GL.dome_at_60(v.pvc - 14.696, v.t_valve, True, atm=GL.PROSPER_ATM))
    return float(v.pd60)


def setup_from_design(res, setup: GasLiftSetup) -> GasLiftSetup:
    """Valves of a gaslift.DesignResult (PROSPER Transfer | From design)."""
    vtype = getattr(res.inp, "valve_type", "")
    vk = "Tubing" if vtype in GL.VALVE_TYPES[1:] else "Valve"
    vs = tuple(GLValve(float(v.md), "Orifice" if v.kind == "Orifice" else vk, float(v.port_64),
                       float(v.r) if np.isfinite(v.r) else 0.0, system_dome(v, vtype), int(getattr(v, "n_valves", 1)))
               for v in res.valves)
    from dataclasses import replace
    return replace(setup, valves=vs)


def solution_state(st: LiftState, setup: GasLiftSetup, path: V.FlowPath, k=-1) -> dict:
    """Summary of case k of a lift traverse: lift gas, injection point, pressures there, profile, valves."""
    k = k % st.P.shape[0]
    inj = float(st.inj_md[k]) if np.isfinite(st.inj_md[k]) else np.nan
    pt_inj = float(np.interp(inj, path.md, st.P[k])) if np.isfinite(inj) else np.nan
    if st.inj_p is not None and np.isfinite(inj):
        pt_inj = float(st.inj_p[k])
    pc_inj = float(np.interp(inj, path.md, st.pc[k])) if (st.pc is not None and np.isfinite(inj)) else np.nan
    out = {"Gas lift method": setup.method,
           "Gas lift gas injection rate (MMscf/d)": float(st.qgi[k]),
           "Injection depth (ft MD)": inj,
           "Injection depth (ft TVD)": float(np.interp(inj, path.md, path.tvd)) if np.isfinite(inj) else np.nan,
           "Tubing pressure at injection (psia)": pt_inj,
           "Casing pressure at injection (psia)": pc_inj,
           "dP across injection point (psi)": pc_inj - pt_inj,
           "Temperature at injection (°F)": float(np.interp(inj, path.md, st.T[k])) if np.isfinite(inj) else np.nan,
           "Casing head pressure (psia)": float(st.pc[k, 0]) if st.pc is not None else np.nan,
           "_note": str(st.note[k])}
    if st.flag is not None:
        out["Multipoint flag"] = str(st.flag[k])
        out["_valves"] = valve_table(setup, st, k)
    prof = {"Measured depth (ft)": path.md, "True vertical depth (ft)": path.tvd,
            "Tubing pressure (psia)": st.P[k], "Tubing temperature (°F)": st.T[k]}
    if st.pc is not None:
        prof["Casing pressure (psia)"] = st.pc[k]
    out["_profile"] = pd.DataFrame(prof)
    return out


def _gas_per_segment(path, setup: GasLiftSetup, st: LiftState, k):
    """Lift gas in each tubing segment (MMscf/d) for case k."""
    if st.valve_q is not None:
        mds = np.array(sorted(v.md for v in setup.valves))
        return _seg_from_valves(path, mds, st.valve_q[k:k + 1])[0]
    mid = 0.5 * (path.md[1:] + path.md[:-1])
    inj = st.inj_md[k]
    return np.where(np.isfinite(inj) & (mid < (inj if np.isfinite(inj) else -1.0)), st.qgi[k], 0.0)


def details(path, fluid, params, q, setup: GasLiftSetup, correlation, p1=1.0, p2=1.0, u=8.0, hc=None):
    """Gradient traverse of a gas-lifted oil well at liquid rate q: (summary dict, profile table).
    params: whp, wc (%), gor and any lift overrides (qgi, glr_inj, pcs, dp_valve, inj_md)."""
    from .correlations import NEEDS_GLR, no_slip
    hc = hc or HeatCapacities()
    case = {"q": np.array([float(q)]), "wc": np.array([params["wc"] / 100.0]), "gor": np.array([float(params["gor"])])}
    st = lift_traverse(path, fluid, float(params["whp"]), {**case, **setup.row_inputs(params, [q])}, setup,
                       correlation, p1, p2, u, hc)
    path = setup.flow_path(path)
    P, T = st.P[0], st.T[0]
    gseg = _gas_per_segment(path, setup, st, 0)
    corr = CORRELATIONS[correlation]
    qo = max(float(q) * (1 - params["wc"] / 100.0), 1e-6)
    rows, dpg, dpf = [], 0.0, 0.0
    for i in range(len(path.md) - 1):
        pm, tm, L = 0.5 * (P[i] + P[i + 1]), 0.5 * (T[i] + T[i + 1]), path.seg_len[i]
        c = {**case, "gor": case["gor"] + gseg[i] * 1e6 / qo}
        s = fluid.state(np.array([pm]), np.array([tm]), c)
        a = path.area[i]
        vsl, vsg = max(float(s["ql"][0]) / a, 1e-12), max(float(s["qg"][0]) / a, 0.0)
        args = (np.array([vsl]), np.array([vsg]), s["rho_l"], np.maximum(s["rho_g"], 1e-4), s["mu_l"],
                np.maximum(s["mu_g"], 1e-4), np.maximum(s["sig"], 0.5), path.d_h[i], path.rough[i], path.sin_t[i],
                np.array([pm]))
        lam = vsl / max(vsl + vsg, 1e-30)
        if lam > 0.99999 or lam < 1e-6:
            e, f, hl = no_slip(*args)
        elif corr.__name__ in NEEDS_GLR:
            e, f, hl = corr(*args, glr=V._glr(c, "oil"))
        else:
            e, f, hl = corr(*args)
        if path.fric_mult is not None:
            f = f * path.fric_mult[i]
        dpg += p1 * float(e[0]) * L
        dpf += p2 * float(f[0]) * L
        hl0 = float(hl[0])
        rows.append({"MD (ft)": path.md[i + 1], "TVD (ft)": path.tvd[i + 1], "Pressure (psia)": P[i + 1],
                     "Temperature (°F)": T[i + 1],
                     "Casing pressure (psia)": st.pc[0, i + 1] if st.pc is not None else np.nan,
                     "Lift gas in tubing (MMscf/d)": gseg[i], "Liquid holdup": hl0, "No-slip holdup": lam,
                     "Superficial liquid velocity (ft/s)": vsl, "Superficial gas velocity (ft/s)": vsg,
                     "Mixture velocity (ft/s)": vsl + vsg,
                     "In-situ density (lb/ft3)": hl0 * float(s["rho_l"][0]) + (1 - hl0) * float(s["rho_g"][0]),
                     "Elevation gradient (psi/ft)": p1 * float(e[0]), "Friction gradient (psi/ft)": p2 * float(f[0])})
    top = {k: np.nan for k in rows[0]}
    top.update({"MD (ft)": path.md[0], "TVD (ft)": path.tvd[0], "Pressure (psia)": P[0], "Temperature (°F)": T[0],
                "Casing pressure (psia)": st.pc[0, 0] if st.pc is not None else np.nan})
    summ = solution_state(st, setup, path, 0)
    det = {"Wellhead pressure (psia)": P[0], "Wellhead temperature (°F)": T[0], "Bottom node pressure (psia)": P[-1],
           "Bottom node temperature (°F)": T[-1], "dP gravity - production string (psi)": dpg,
           "dP friction - production string (psi)": dpf,
           **{k: v for k, v in summ.items() if not k.startswith("_")}}
    det["_note"] = summ["_note"]
    if "_valves" in summ:
        det["_valves"] = summ["_valves"]
    return det, pd.DataFrame([top] + rows)
