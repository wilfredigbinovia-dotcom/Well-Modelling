"""Well operating envelope: the range of rates a well can be produced at, case by case (reservoir pressure, water cut
and GOR - or WGR and separator GOR for gas wells - swept), with the limit that binds at each end.

Every operating point lies on the inflow: at rate q the bottom-node pressure is the IPR's Pwf(q) and, traversing the
tubing upwards from it, the wellhead pressure WHP(q) (the top-node inflow curve). One upward traverse per case over a
rate grid therefore gives everything the limits need - the pressure and temperature at every node at every rate:

  rate limits      AOF (the grid's end); stability - the well is stable to the right of the WHP(q) maximum (left of it
                   a fixed back pressure gives the unstable, left-hand VLP/IPR intersection); liquid loading (Turner,
                   gas wells); PE5-style film stability; erosional velocity; hydrates (temperature at every node above
                   the hydrate curve); coning (the critical rate of the coning module, oil wells as liquid rate);
  pressure limits  minimum WHP (separator / flowline back pressure; at least 14.7 psia - the well flows to surface),
                   maximum WHP (facility rating); sanding (Pwf not below the minimum flowing BHP of the sanding module at
                   the case's reservoir pressure); tubing burst, collapse and triaxial (tubulars module) along the
                   traverse from the surface to the packer against the annulus (surface pressure + packer fluid).

Each limit is a margin m(q) >= 0 where satisfied, evaluated on the grid, linearly interpolated between grid rates; the
operating window is where every active margin is non-negative and its ends name the binding limits. Shut-in: the
tubing is also checked with gas to surface (a static gas column from the reservoir pressure, the usual tubing burst
load case), and the highest WHP the tubing tolerates at the wellhead is reported (a line on the WHP view). Segments
shorter than 10 ft (the casing stub below the tubing shoe) are left out of the velocity limits.

Not covered: surface equipment, artificial lift and diluent (the envelope is at the wellhead, natural flow); casing
below the packer; connection ratings, wear, buckling and thermal axial loads.
Rates: oil wells liquid STB/d, gas wells MMscf/d (IPR curves in their own units, gas Mscf/d)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import complementary as CX
from . import coning as CN
from . import sanding as SD
from . import system as S
from . import tubulars as TB
from . import vlp as V
from .equipment import HeatCapacities

PSI_PER_FT_PPG = 0.051948
BIG = 1e9
MIN_SEG_FT = 10.0       # shorter segments (the stub of casing between the tubing shoe and the bottom node) are not
                        # used for the velocity limits - a one-foot stub of wide casing would otherwise set them

LIMIT_LABELS = {
    "whp_min": "Minimum WHP", "whp_max": "Maximum WHP", "stability": "Stability (VLP minimum)",
    "turner": "Liquid loading (Turner)", "pe5": "PE5 film stability", "erosion": "Erosional velocity",
    "hydrate": "Hydrates", "coning": "Coning", "sanding": "Sanding", "burst": "Tubing burst",
    "collapse": "Tubing collapse", "triaxial": "Tubing triaxial", "aof": "AOF", "grid": "None (lowest rate)",
}


@dataclass
class TubingCheck:
    pipe: TB.Tubular
    packer_md: float                    # ft, the string is checked from the surface to here
    annulus_p: float = 14.7             # psia at surface in the A-annulus
    packer_fluid_ppg: float = 8.6
    extra_axial: float = 0.0            # lbf, tension positive
    df: TB.DesignFactors = field(default_factory=TB.DesignFactors)
    method: str = "ISO"
    gas_sg: float | None = None         # gas-to-surface shut-in column; None: the fluid's gas gravity (0.65 if unknown)


@dataclass
class Limits:
    whp_min: float | None = None        # psia
    whp_max: float | None = None
    stability: bool = True
    turner: bool = True
    pe5: bool = True
    erosion: bool = True
    hydrate: bool = True                # needs comp.hydrate
    sanding: SD.SandingInputs | None = None
    coning: Callable | None = None      # f(params, ipr_curve) -> critical rate (VLP units)
    coning_label: str = "Coning"
    tubing: TubingCheck | None = None
    comp: CX.CompOptions = field(default_factory=CX.CompOptions)


@dataclass
class CaseEnvelope:
    params: dict
    rates: np.ndarray                   # VLP units
    pwf: np.ndarray                     # IPR flowing BHP, psia
    whp: np.ndarray                     # wellhead pressure on the inflow, psia (NaN: the well does not reach surface)
    wht: np.ndarray
    margins: dict                       # limit -> margin array (>= 0 satisfied)
    crossings: dict                     # limit -> list of (rate, "min" | "max")
    windows: list                       # [(q_lo, q_hi, limit at lo, limit at hi)]
    aof: float
    cone_rate: float | None = None
    sand_pwf: float | None = None
    vlp_min: np.ndarray | None = None   # bottom-node pressure at the minimum WHP (BHP view)
    vlp_max: np.ndarray | None = None
    shut_in: dict = field(default_factory=dict)
    op: dict = field(default_factory=dict)  # operating point at the case's WHP
    notes: list = field(default_factory=list)
    whp_levels: dict = field(default_factory=dict)

    @property
    def window(self):
        """Main window (widest), or None."""
        return max(self.windows, key=lambda w: w[1] - w[0]) if self.windows else None

    def at(self, q, arr):
        ok = np.isfinite(arr)
        if not np.isfinite(q) or ok.sum() < 2 or q < self.rates[ok][0] - 1e-9 or q > self.rates[ok][-1] + 1e-9:
            return np.nan
        return float(np.interp(q, self.rates[ok], arr[ok]))


# ------------------------------------------------------------------ helpers
def _ipr_for_case(ipr_for, kind, p, c):
    changes = {k: float(p[k]) for k in S.IPR_KEYS if k in c}
    if kind == "oil" and "wc" in p:
        changes["wc"] = float(p["wc"]) / 100.0
    if kind == "oil" and "gor" in p:
        changes["gor"] = float(p["gor"])
    return ipr_for(changes)


def _crossings(rates, m):
    """Zero crossings of margin m: (rate, 'min') where it becomes satisfied going up in rate, (rate, 'max') where it
    stops being satisfied."""
    out = []
    for k in range(len(rates) - 1):
        a, b = m[k], m[k + 1]
        if (a >= 0) != (b >= 0):
            q = rates[k] + (0 - a) / (b - a) * (rates[k + 1] - rates[k]) if b != a else rates[k]
            out.append((float(q), "min" if b >= 0 else "max"))
    return out


def _windows(rates, margins, crossings):
    brk = sorted({float(rates[0]), float(rates[-1]), *[q for v in crossings.values() for q, _ in v]})
    wins = []
    for a, b in zip(brk[:-1], brk[1:]):
        if b - a <= 1e-12 * max(abs(b), 1):
            continue
        mid = 0.5 * (a + b)
        ok = all(np.interp(mid, rates, m) >= 0 for m in margins.values())
        if ok:
            if wins and abs(wins[-1][1] - a) < 1e-9 * max(b, 1):
                wins[-1][1] = b
            else:
                wins.append([a, b])

    def who(q, side):
        best, d = None, np.inf
        for k, v in crossings.items():
            for qc, s in v:
                if s == side and abs(qc - q) < d:
                    best, d = k, abs(qc - q)
        if best is not None and d <= 1e-6 * max(q, 1):
            return best
        return "aof" if side == "max" else "grid"
    return [(a, b, who(a, "min"), who(b, "max")) for a, b in wins]


def _pe5_vec(vsl, vsg, rho_l, rho_g, mu_l, mu_g, d_ft, sin_t):
    out = np.full(vsl.shape, np.nan)
    for j in range(vsl.size):
        out.flat[j] = CX.pe5_stability_ratio(float(vsl.flat[j]), float(vsg.flat[j]), float(rho_l.flat[j]),
                                             float(rho_g.flat[j]), float(mu_l.flat[j]), float(mu_g.flat[j]),
                                             d_ft, sin_t)
    return out


def _node_margins(path, fluid, kind, P, T, case, lim: Limits):
    """Per row (rate) worst margins over the flow path: Turner vm/vt - 1, PE5 ratio - 1, 1 - vm/ve, T - T_hydrate."""
    n_r, n = P.shape
    tur = np.full(n_r, BIG)
    pe5 = np.full(n_r, BIG)
    ero = np.full(n_r, BIG)
    hyd = np.full(n_r, BIG)
    comp = lim.comp
    for i in range(n - 1):
        if path.seg_len[i] < MIN_SEG_FT:
            continue
        pm, tm = 0.5 * (P[:, i] + P[:, i + 1]), 0.5 * (T[:, i] + T[:, i + 1])
        good = np.isfinite(pm)
        if not good.any():
            continue
        s = V.apply_rheology(fluid, fluid.state(np.where(good, pm, 1000.0), tm, case), path.area[i], path.d_h[i])
        a = path.area[i]
        ql = np.asarray(s["ql"], float)
        qg = np.asarray(s["qg"], float)
        vsl, vsg = np.maximum(ql / a, 0.0), np.maximum(qg / a, 0.0)
        vm = vsl + vsg
        rho_l, rho_g = np.asarray(s["rho_l"], float), np.asarray(s["rho_g"], float)
        lam = vsl / np.maximum(vm, 1e-30)
        rho_ns = lam * rho_l + (1 - lam) * np.nan_to_num(rho_g)
        if lim.erosion:
            ve = np.array([CX.erosional_velocity(r, path.d_h[i] * 12, comp) for r in rho_ns])
            ero = np.where(good, np.minimum(ero, 1 - vm / ve), ero)
        if lim.turner and kind == "gas":
            sig = np.asarray(s["sig"], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                vt = comp.turner * np.maximum(sig, 0) ** 0.25 * np.maximum(rho_l - rho_g, 0) ** 0.25 / np.sqrt(rho_g)
            app = good & (vsl > 0) & (vsg > 0) & np.isfinite(vt) & (vt > 0)
            tur = np.where(app, np.minimum(tur, vm / np.where(app, vt, 1) - 1), tur)
        if lim.pe5:
            r = _pe5_vec(np.maximum(vsl, 0), vsg, rho_l, np.maximum(np.nan_to_num(rho_g), 1e-4),
                         np.asarray(s["mu_l"], float), np.maximum(np.asarray(s["mu_g"], float), 1e-4), path.d_h[i],
                         float(path.sin_t[i]))
            r = np.where(np.isfinite(r), r - 1, BIG)
            pe5 = np.where(good, np.minimum(pe5, r), pe5)
    if lim.hydrate and comp.hydrate is not None:
        pc, tc = (np.asarray(x, float) for x in comp.hydrate)
        o = np.argsort(pc)
        th = np.interp(np.nan_to_num(P, nan=0.0), pc[o], tc[o])
        d = np.where(np.isfinite(P), T - th, BIG)
        hyd = d.min(axis=1)
    return {"turner": tur, "pe5": pe5, "erosion": ero, "hydrate": hyd}


def _gas_sg(fluid, tc: TubingCheck):
    if tc.gas_sg:
        return float(tc.gas_sg)
    for obj in (fluid, getattr(fluid, "m", None)):
        inp = getattr(obj, "inp", None)
        if inp is not None and getattr(inp, "gas_sg", None):
            return float(inp.gas_sg)
    return 0.65


def gas_column(path, p_bottom, sg):
    """Static gas column (psia at every node, top first) from p_bottom at the bottom node, at the formation
    temperature: dp/dz = 0.01875 sg p / (z T), z Dranchuk-Abou-Kassem with Sutton's pseudo-criticals."""
    from . import blackoil as bo
    tpc, ppc = bo.gas_pseudocritical(bo.PVTInputs(rsb=1, api=40, gas_sg=sg))
    p = np.empty(len(path.md))
    p[-1] = p_bottom
    for i in range(len(path.md) - 2, -1, -1):
        dz = path.tvd[i + 1] - path.tvd[i]
        tr = 0.5 * (path.t_form[i] + path.t_form[i + 1]) + 459.67
        pi_ = p[i + 1]
        for _ in range(3):
            pm = 0.5 * (pi_ + p[i + 1])
            z = float(bo.z_dak(tr / tpc, pm / ppc))
            pi_ = p[i + 1] - 0.01875 * sg * pm / (z * tr) * dz
        p[i] = pi_
    return p


def _tubing_margins(path, P, tc: TubingCheck):
    """Burst / collapse / triaxial margins (min SF / DF - 1) per row along the tubing, plus the check objects."""
    sel = path.md <= tc.packer_md + 0.5
    tvd = path.tvd[sel]
    bottom = float(np.interp(tc.packer_md, path.md, path.tvd))
    p_out = tc.annulus_p + PSI_PER_FT_PPG * tc.packer_fluid_ppg * tvd
    out = {"burst": [], "collapse": [], "triaxial": []}
    checks = []
    for row in P:
        p_in = row[sel]
        if not np.all(np.isfinite(p_in)):        # column does not reach surface: atmospheric above the fluid level
            p_in = np.where(np.isfinite(p_in), p_in, 14.7)
        ck = TB.check_profile(tc.pipe, tvd, p_in, p_out, bottom, tc.packer_fluid_ppg, tc.extra_axial, tc.df,
                              tc.method)
        s = ck.min_sf()
        out["burst"].append(min(s["burst"], BIG) / tc.df.burst - 1)
        out["collapse"].append(min(s["collapse"], BIG) / tc.df.collapse - 1)
        out["triaxial"].append(min(s["triaxial"], BIG) / tc.df.triaxial - 1)
        checks.append(ck)
    return {k: np.array(v) for k, v in out.items()}, checks


# ------------------------------------------------------------------ envelope
def _vlp_batch(path, fluid, kind, items, correlation, p1, p2, u, hc):
    """Bottom-node pressure for items [(params, curve, rates, whp)] in one downward traverse."""
    qs, wh, flu = [], [], {}
    for p, curve, q, w in items:
        q = np.atleast_1d(np.asarray(q, float))
        qs.append(q)
        wh.append(np.full(q.size, float(w)))
        for k, v in S._flu(kind, p, curve, q).items():
            flu.setdefault(k, []).append(v)
    if not qs:
        return []
    case = {"q": np.concatenate(qs), **{k: np.concatenate(v) for k, v in flu.items()}}
    Pd, _ = V.traverse(path, fluid, np.concatenate(wh), case, correlation, p1, p2, u, hc)
    out, pos = [], 0
    for q in qs:
        out.append(Pd[pos:pos + q.size, -1])
        pos += q.size
    return out


def build(path, fluid, kind, base: dict, cases: list[dict], ipr_for, correlation, lim: Limits, p1=1.0, p2=1.0,
          hc: HeatCapacities | None = None, n=40) -> list[CaseEnvelope]:
    """Envelope of every case. base: whp (the operating WHP), u, wc/gor (oil, wc %) or wgr/sep_gor (gas); cases
    override pr, wc, gor (wgr, sep_gor) and so on (system.combinations of the sweeps)."""
    hc = hc or HeatCapacities()
    scale = 1000.0 if kind == "gas" else 1.0
    u = float(base["u"])
    tr = (correlation, p1, p2, u, hc)
    prep = []
    for c in cases:
        p = {**base, **c}
        curve = _ipr_for_case(ipr_for, kind, p, c)
        aof = float(curve.aof) / scale
        prep.append([p, curve, aof, np.geomspace(aof * 0.003, aof * 0.995, n)])
    pwf_of = lambda curve, q: S._ipr_at(curve, np.atleast_1d(q) * scale)
    # WHP levels: minimum (at least 14.7 psia - the well flows), maximum, and the operating WHP of each case. Their
    # limits are nodal intersections (VLP at the level against the IPR), refined by regula falsi and added to the grid.
    w_min = max(lim.whp_min or 0.0, 14.7)
    levels = [("min", lambda p: w_min)] + ([("max", lambda p: lim.whp_max)] if lim.whp_max is not None else []) \
        + [("op", lambda p: float(p["whp"]))]
    coarse = _vlp_batch(path, fluid, kind, [(p, cv, r, f(p)) for _, f in levels for p, cv, _, r in prep], *tr)
    brackets = []                                  # (case, level, ql, gl, qr, gr) with g = Pwf - VLP
    for li, (name, f) in enumerate(levels):
        for ci, (p, curve, aof, rates) in enumerate(prep):
            g = pwf_of(curve, rates) - coarse[li * len(prep) + ci]
            for k in range(len(rates) - 1):
                if np.isfinite(g[k]) and np.isfinite(g[k + 1]) and (g[k] >= 0) != (g[k + 1] >= 0):
                    brackets.append([ci, li, rates[k], g[k], rates[k + 1], g[k + 1]])
    for _ in range(3):
        if not brackets:
            break
        q0 = [b[2] - b[3] * (b[4] - b[2]) / (b[5] - b[3]) for b in brackets]
        v = _vlp_batch(path, fluid, kind, [(prep[b[0]][0], prep[b[0]][1], q, levels[b[1]][1](prep[b[0]][0]))
                                           for b, q in zip(brackets, q0)], *tr)
        for b, q, vv in zip(brackets, q0, v):
            g0 = float(pwf_of(prep[b[0]][1], q)[0] - vv[0])
            if (g0 >= 0) == (b[3] >= 0):
                b[2], b[3] = q, g0
            else:
                b[4], b[5] = q, g0
    roots = {}
    for b in brackets:
        q = b[2] - b[3] * (b[4] - b[2]) / (b[5] - b[3]) if b[5] != b[3] else b[2]
        roots.setdefault(b[0], []).append(float(q))
    for ci, qs_ in roots.items():
        prep[ci][3] = np.unique(np.concatenate([prep[ci][3], qs_]))
    vlp = _vlp_batch(path, fluid, kind, [(p, cv, r, f(p)) for _, f in levels for p, cv, _, r in prep], *tr)
    vlp_of = {(name, ci): vlp[li * len(prep) + ci] for li, (name, _) in enumerate(levels) for ci in range(len(prep))}
    # one batched upward traverse: all cases and rates -> WHP, temperatures and the tubing pressures
    qs, pb, flu = [], [], {}
    for p, curve, aof, rates in prep:
        qs.append(rates)
        pb.append(pwf_of(curve, rates))
        for k, v in S._flu(kind, p, curve, rates).items():
            flu.setdefault(k, []).append(v)
    case_all = {"q": np.concatenate(qs), **{k: np.concatenate(v) for k, v in flu.items()}}
    P, T = V.traverse_up(path, fluid, np.concatenate(pb), case_all, *tr)
    node_m = _node_margins(path, fluid, kind, P, T, case_all, lim)
    tub_m, checks = _tubing_margins(path, P, lim.tubing) if lim.tubing is not None else (None, None)
    if lim.tubing is not None:                 # shut-in, gas to surface: one static gas column per case
        sg = _gas_sg(fluid, lim.tubing)
        P_si = np.array([gas_column(path, float(p.get("pr", curve.pr)), sg) for p, curve, *_r in prep])
        _, checks_si = _tubing_margins(path, P_si, lim.tubing)
    out = []
    pos = 0
    for ci, (p, curve, aof, rates) in enumerate(prep):
        m = len(rates)
        sl = slice(pos, pos + m)
        pwf = pwf_of(curve, rates)
        whp, wht = P[sl, 0].copy(), T[sl, 0].copy()
        env = CaseEnvelope(p, rates, pwf, whp, wht, {}, {}, [], aof)
        env.vlp_min = vlp_of[("min", ci)]
        env.whp_levels = {"whp_min": w_min, "whp_max": lim.whp_max}
        env.vlp_max = vlp_of.get(("max", ci))
        mg = env.margins
        mg["whp_min"] = pwf - env.vlp_min
        if env.vlp_max is not None:
            mg["whp_max"] = env.vlp_max - pwf
        if lim.stability and np.isfinite(whp).any():
            q_peak = float(rates[int(np.nanargmax(whp))])
            mg["stability"] = rates - q_peak if q_peak > rates[0] else np.full(m, BIG)
        for k in ("turner", "pe5", "erosion", "hydrate"):
            arr = node_m[k][sl]
            if np.any(arr < BIG):
                mg[k] = arr
        if lim.coning is not None:
            qc = float(lim.coning(p, curve))
            env.cone_rate = qc
            mg["coning"] = qc - rates
        if lim.sanding is not None:
            env.sand_pwf = SD.min_pwf(lim.sanding, float(p.get("pr", curve.pr)))
            mg["sanding"] = pwf - env.sand_pwf
        if tub_m is not None:
            for k in ("burst", "collapse", "triaxial"):
                mg[k] = tub_m[k][sl]
            ck = checks_si[ci]
            env.shut_in = {"Shut-in WHP (psia)": float(P_si[ci, 0]),
                           **{f"Shut-in {k} SF": v for k, v in ck.min_sf().items()},
                           "Shut-in check": "Pass" if ck.ok else "Fail",
                           "Max WHP for burst (psia)": TB.max_internal_for_burst(lim.tubing.pipe, lim.tubing.annulus_p,
                                                                                 lim.tubing.df)}
            if not ck.ok:
                env.notes.append("Tubing fails the shut-in check (gas to surface from the reservoir pressure).")
        mg = {k: np.nan_to_num(np.minimum(v, BIG), nan=-BIG if k.startswith("whp") else BIG, neginf=-BIG)
              for k, v in mg.items()}
        env.margins = mg
        env.crossings = {k: _crossings(rates, v) for k, v in mg.items()}
        env.windows = _windows(rates, mg, env.crossings)
        # operating point at the case's WHP: the stable (right-hand) VLP / IPR intersection, as in system analysis
        g = pwf - vlp_of[("op", ci)]
        xs = [q for q, s in _crossings(rates, np.nan_to_num(g, nan=-BIG)) if s == "max"]
        if xs:
            q_op = xs[-1]
            env.op = {"Rate": q_op, "Pwf (psia)": float(pwf_of(curve, q_op)[0]), "WHP (psia)": float(p["whp"]),
                      "Inside window": any(a - 1e-9 <= q_op <= b + 1e-9 for a, b, *_ in env.windows),
                      "Violated": [LIMIT_LABELS.get(k, k) for k, v in mg.items()
                                   if np.interp(q_op, rates, v) < -1e-9 and k != "whp_min"]}
        okw = np.isfinite(whp)
        for lvl, wl in (("whp_min", w_min), ("whp_max", lim.whp_max)):
            up = _crossings(rates[okw], whp[okw] - wl) if wl is not None and okw.sum() > 1 else []
            for q, side in env.crossings.get(lvl, []):
                q_up = [x for x, s_ in up if s_ == side]
                if q_up and min(abs(x - q) for x in q_up) <= 0.02 * q:
                    continue
                env.notes.append(f"At {q:,.4g} the WHP limit of {wl:,.0f} psia is set by the VLP (traversed down from "
                                 "the wellhead, as in system analysis), but the inflow traversed upwards does not "
                                 "reach that WHP within 2 % of this rate: a restriction (SSSV) is near critical flow "
                                 "there, or the pressure is nearly exhausted near surface.")
        if not env.windows:
            env.notes.append("No operating window: some limit is violated at every rate.")
        out.append(env)
        pos += m
    return out


# ------------------------------------------------------------------ coning limit factories
def oil_coning(cone: CN.ConeInputs, update: Callable | None = None, rim: tuple | None = None):
    """Coning limit for oil wells: critical oil rate -> liquid rate at the case's water cut. update(cone, params) may
    return inputs refreshed for the case (PVT at its reservoir pressure); rim=(rho_w, rho_g) for a thin oil rim."""
    def f(params, curve):
        c = update(cone, params) if update else cone
        qo = CN.oil_rim(c, *rim)["Critical rate (STB/d)"] if rim else CN.critical_rate(c)
        wc = float(params.get("wc", 0.0)) / 100.0
        return float(qo) / max(1 - wc, 1e-6)
    return f


def gas_coning(cone: CN.GasConeInputs, update: Callable | None = None, ratio: float | None = None, non_darcy=True):
    """Water coning in gas wells: critical gas rate in MMscf/d. The non-Darcy ratio comes from the case's IPR
    (turbulent / laminar coefficients) unless given; non_darcy=False keeps the Darcy rate."""
    def f(params, curve):
        c = update(cone, params) if update else cone
        if "pr" in params:
            c = CN.GasConeInputs(**{**c.__dict__, "p": float(params["pr"])}) if c.p != float(params["pr"]) else c
        r = (ratio if ratio is not None else CN.ipr_turbulence_ratio(curve)) if non_darcy else None
        return CN.gas_critical_rate(c, r) / 1000.0
    return f


# ------------------------------------------------------------------ tables
def _units(kind):
    return "MMscf/d" if kind == "gas" else "STB/d"


def _end_whp(e: CaseEnvelope, q, limit):
    """WHP at a window end: the level itself where a WHP limit sets it, otherwise the inflow's WHP."""
    if limit in ("whp_min", "whp_max"):
        return float(e.whp_levels[limit])
    return e.at(q, e.whp)


def summary_table(kind, envs: list[CaseEnvelope], names: dict | None = None) -> pd.DataFrame:
    """One row per case: swept parameters, the window and its binding limits, Pwf and WHP at its ends."""
    u = _units(kind)
    rows = []
    for i, e in enumerate(envs, 1):
        row = {"Case": i}
        for k in (names or {}):
            if k in e.params:
                row[names[k]] = e.params[k]
        w = e.window
        if w:
            a, b, la, lb = w
            row.update({f"Min rate ({u})": a, "Lower limit": LIMIT_LABELS.get(la, la),
                        f"Max rate ({u})": b, "Upper limit": LIMIT_LABELS.get(lb, lb),
                        "Pwf at max rate (psia)": e.at(b, e.pwf), "Pwf at min rate (psia)": e.at(a, e.pwf),
                        "WHP at max rate (psia)": _end_whp(e, b, lb), "WHP at min rate (psia)": _end_whp(e, a, la)})
        else:
            row.update({f"Min rate ({u})": np.nan, "Lower limit": "-", f"Max rate ({u})": np.nan,
                        "Upper limit": "No window"})
        row[f"AOF ({u})"] = e.aof
        if e.op:
            row[f"Rate at operating WHP ({u})"] = e.op["Rate"]
            row["Operating point"] = "Inside" if e.op["Inside window"] else ", ".join(e.op["Violated"]) or "Outside"
        if e.cone_rate is not None:
            row[f"Coning critical rate ({u})"] = e.cone_rate
        if e.sand_pwf is not None:
            row["Sanding min Pwf (psia)"] = e.sand_pwf
        if e.shut_in:
            row["Shut-in WHP (psia)"] = e.shut_in["Shut-in WHP (psia)"]
            row["Shut-in tubing check"] = e.shut_in["Shut-in check"]
        rows.append(row)
    return pd.DataFrame(rows)


def limits_table(kind, e: CaseEnvelope) -> pd.DataFrame:
    """Every active limit of one case: the rates where it switches and the kind of bound."""
    u = _units(kind)
    rows = []
    for k, xs in e.crossings.items():
        m = e.margins[k]
        state = "Satisfied everywhere" if np.all(m >= 0) else ("Violated everywhere" if np.all(m < 0) else "")
        if not xs:
            rows.append({"Limit": LIMIT_LABELS.get(k, k), "Bound": "-", f"Rate ({u})": np.nan, "Pwf (psia)": np.nan,
                         "WHP (psia)": np.nan, "Status": state})
        for q, side in xs:
            rows.append({"Limit": LIMIT_LABELS.get(k, k), "Bound": "Minimum rate" if side == "min" else "Maximum rate",
                         f"Rate ({u})": q, "Pwf (psia)": e.at(q, e.pwf), "WHP (psia)": e.at(q, e.whp), "Status": ""})
    return pd.DataFrame(rows)


def curve_table(kind, e: CaseEnvelope) -> pd.DataFrame:
    u = _units(kind)
    df = pd.DataFrame({f"Rate ({u})": e.rates, "Pwf (psia)": e.pwf, "WHP (psia)": e.whp, "WHT (°F)": e.wht})
    if e.vlp_min is not None:
        df["VLP at min WHP (psia)"] = e.vlp_min
    if e.vlp_max is not None:
        df["VLP at max WHP (psia)"] = e.vlp_max
    ok = np.ones(len(e.rates), bool)
    for m in e.margins.values():
        ok &= m >= 0
    df["Inside envelope"] = np.where(ok, "Yes", "No")
    for k, m in e.margins.items():
        df[f"{LIMIT_LABELS.get(k, k)} margin"] = np.where(np.abs(m) >= BIG, np.nan, m)
    return df
