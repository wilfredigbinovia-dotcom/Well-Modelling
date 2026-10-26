"""System analysis (nodal): VLP from the top node (the wellhead, or the manifold when surface equipment is used)
against the IPR at the bottom node, for a base case and for sensitivity combinations of up to three variables.

The VLP curve of every case is a vectorised traverse over its rate grid, so all cases sharing a flow path and U value
run through the tubing in one pass. The solution is the intersection of the VLP and IPR curves at the bottom node,
refined by regula falsi; an intersection is stable when the VLP crosses the IPR from below (it may lie left of the
VLP minimum if the IPR is steeper there); the other, left-hand intersection is unstable.

Rates: oil wells liquid STB/d, gas wells MMscf/d. IPR curves are in their own units (oil STB/d, gas Mscf/d).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import complementary as CX
from . import surface as SF
from . import vlp as V
from . import gaslift_system as GLS
from . import diluent as DL
from .correlations import CORRELATIONS, NEEDS_GLR, no_slip
from .equipment import HeatCapacities

MAX_CASES = 500
EROSION_C = 100.0          # API RP 14E constant for continuous service (lb^0.5 / (ft^0.5 s))

OIL_VARIABLES = {
    "whp": "Top node pressure (psia)",
    "wc": "Water cut (%)",
    "gor": "Total GOR (scf/STB)",
    "pr": "Reservoir pressure (psia)",
    "j": "Productivity index (STB/d/psi)",
    "skin": "Skin",
    "tubing_id": "Tubing ID (in)",
    "u": "Overall heat transfer coefficient (BTU/h/ft2/F)",
    "choke": "Choke size (64ths in)",
    "pump_power": "Multiphase pump power available (hp)",
}
GAS_VARIABLES = {
    "whp": "Top node pressure (psia)",
    "wgr": "Water gas ratio (STB/MMscf)",
    "sep_gor": "Separator GOR (scf/STB)",
    "pr": "Reservoir pressure (psia)",
    "skin": "Skin",
    "tubing_id": "Tubing ID (in)",
    "u": "Overall heat transfer coefficient (BTU/h/ft2/F)",
    "choke": "Choke size (64ths in)",
    "pump_power": "Multiphase pump power available (hp)",
}
SURFACE_ONLY = ("choke", "pump_power")     # variables that need surface equipment with a choke / pump
IPR_KEYS = ("pr", "j", "skin")


def generate_values(first, last, n, spacing="Linear"):
    """n values from first to last, linear or geometric spacing."""
    n = int(n)
    if n < 1:
        raise ValueError("Number of values must be at least 1.")
    if n == 1:
        return np.array([float(first)])
    if spacing.lower().startswith("geo"):
        if first <= 0 or last <= 0:
            raise ValueError("Geometric spacing needs positive first and last values.")
        return np.geomspace(first, last, n)
    return np.linspace(first, last, n)


def combinations(variables: dict) -> list[dict]:
    """All combinations of the sensitivity values: {name: values} -> list of {name: value}."""
    if not variables:
        return [{}]
    names = list(variables)
    combos = [dict(zip(names, vals)) for vals in itertools.product(*[list(variables[k]) for k in names])]
    if len(combos) > MAX_CASES:
        raise ValueError(f"{len(combos)} combinations - reduce the number of values (limit {MAX_CASES}).")
    return combos


def rate_grid(aof, method="Automatic - Geometric", n=25, user_rates=None):
    if method == "User selected":
        r = np.unique(np.asarray([x for x in (user_rates or []) if x is not None and np.isfinite(x) and x > 0], float))
        if len(r) < 2:
            raise ValueError("Enter at least two positive rates.")
        return r
    top = max(float(aof), 1e-6)
    if method == "Automatic - Linear":
        return np.linspace(top * 0.01, top, n)
    return np.geomspace(top * 0.002, top, n)


def _case_arrays(kind, p):
    """Fluid case dict for one parameter set (oil: q filled later)."""
    if kind == "oil":
        return {"wc": p["wc"] / 100.0, "gor": p["gor"]}
    sg = p.get("sep_gor", 0.0)
    return {"wgr": p["wgr"], "cgr": 1e6 / sg if sg and sg > 0 else 0.0}


def _flu(kind, p, curve, q):
    """Per-rate fluid arrays (wc/gor or wgr/cgr) for VLP rates q. A multilayer IPR with .mix(q) sets the producing
    ratios from the layer contributions at each rate (PROSPER: water cut and GOR follow the IPR); otherwise they
    are the case's constants."""
    q = np.atleast_1d(np.asarray(q, float))
    mix = getattr(curve, "mix", None) if curve is not None else None
    if mix is not None:
        m = mix(np.nan_to_num(q) * (1000.0 if kind == "gas" else 1.0))
        if m:
            return {k: np.asarray(v, float) for k, v in m.items()}
    return {k: np.full(len(q), float(v)) for k, v in _case_arrays(kind, p).items()}


def mixed_params(kind, params, curve, q):
    """params with the producing ratios of a multilayer IPR at rate q (wc in %, gor; or wgr, sep_gor)."""
    mix = getattr(curve, "mix", None) if curve is not None else None
    if mix is None or q is None:
        return params
    m = mix(q * (1000.0 if kind == "gas" else 1.0))
    if not m:
        return params
    if kind == "oil":
        return {**params, "wc": float(m["wc"][0]) * 100.0, "gor": float(m["gor"][0])}
    cgr = float(m["cgr"][0])
    return {**params, "wgr": float(m["wgr"][0]), "sep_gor": 1e6 / cgr if cgr > 0 else 0.0}


@dataclass
class CaseResult:
    params: dict
    rates: np.ndarray
    vlp: np.ndarray
    ipr_pwf: np.ndarray
    ipr_curve: object
    q: float | None
    pwf: float | None
    stable: bool
    note: str
    q_vlp_min: float
    aof: float
    wht: np.ndarray | None = None          # wellhead temperature at each rate
    wht_q: float | None = None            # wellhead temperature at the solution
    details: dict = field(default_factory=dict)
    profile: pd.DataFrame | None = None
    pwh: np.ndarray | None = None          # wellhead pressure at each rate (surface equipment in use)
    pwh_q: float | None = None
    node: str = "bottom"                   # solution node: "bottom" or "top"
    top_in: np.ndarray | None = None       # top-node solution: inflow (IPR + tubing [+ surface]) pressure at the top node
    top_in_q: float | None = None          # ... at the solution rate (NaN when a critical choke sets the rate)

    @property
    def node_pressure(self):
        """Solution node pressure: bottom-node flowing pressure, or the top-node pressure."""
        if self.q is None:
            return None
        return self.pwf if self.node == "bottom" else float(self.params["whp"])


@dataclass
class SurfaceSetup:
    """Surface equipment from the manifold back to the wellhead (items in flow order, see surface.build_items)."""
    items: list
    t_surround: float = 60.0
    u: float = 8.0
    correlation: str = "beggs_brill"
    choke_method: str = "Sachdeva"
    cd: float = 0.75
    friction: float = 1.0      # flowline / fittings friction multiplier (from Match Surface)

    def items_for(self, params):
        c = params.get("choke")
        items = SF.with_choke_size(self.items, float(c)) if c is not None and SF.has_choke(self.items) else self.items
        pw = params.get("pump_power")
        if pw is not None and SF.has_pump(items):
            items = SF.with_pump_power(items, float(pw))
        return items


def _surface_key(params):
    """Cases sharing a surface key march through identical surface items."""
    return params.get("choke"), params.get("pump_power")


def _wellhead(kind, path, fluid, plist, qlist, u, surface: SurfaceSetup | None, hc, curves=None):
    """Wellhead pressure arrays for (params, rates) pairs sharing a flow path and U. Without surface equipment the
    top node is the wellhead. With it, params['whp'] is the manifold pressure and the surface items are marched back
    to the wellhead at the wellhead temperature of each rate (batched by choke size)."""
    if surface is None or not surface.items:
        return [np.full(len(q), float(p["whp"])) for p, q in zip(plist, qlist)]
    out = [None] * len(plist)
    by_choke = {}
    for j, p in enumerate(plist):
        by_choke.setdefault(_surface_key(p), []).append(j)
    for _, js in by_choke.items():
        q = np.concatenate([qlist[j] for j in js])
        flu = {}
        for j in js:
            for k, v in _flu(kind, plist[j], curves[j] if curves else None, qlist[j]).items():
                flu.setdefault(k, []).append(v)
        case = {"q": q, **{k: np.concatenate(v) for k, v in flu.items()}}
        w, cp = fluid.mass_cp(case, hc)
        t_wh = V.temperature_profile(path, w, cp, u)[:, 0]
        pm = np.concatenate([np.full(len(qlist[j]), float(plist[j]["whp"])) for j in js])
        pwh = SF.wellhead_pressure(surface.items_for(plist[js[0]]), fluid, kind, pm, case, t_wh, surface.correlation,
                                   hc, surface.t_surround, surface.u, surface.choke_method, surface.cd, p2=surface.friction)
        pos = 0
        for j in js:
            out[j] = pwh[pos:pos + len(qlist[j])]
            pos += len(qlist[j])
    return out


def _ipr_at(curve, rates_ipr_units):
    order = np.argsort(curve.q)
    q, p = curve.q[order], curve.pwf[order]
    out = np.interp(rates_ipr_units, q, p)
    return np.where(rates_ipr_units <= q.max() * (1 + 1e-9), out, np.nan)


def _roots(rates, diff):
    s = np.where(np.isfinite(diff[:-1]) & np.isfinite(diff[1:]) & (np.sign(diff[:-1]) != np.sign(diff[1:])))[0]
    return [(k, rates[k] + diff[k] / (diff[k] - diff[k + 1]) * (rates[k + 1] - rates[k])) for k in s]


def solve_cases(path_for, fluid, kind, base: dict, cases: list[dict], ipr_for, correlation, p1=1.0, p2=1.0,
                hc: HeatCapacities | None = None, method="Automatic - Geometric", n=25, user_rates=None,
                allow_left=False, surface: SurfaceSetup | None = None, node="bottom", lift=None, diluent=None):
    """Solve every case. path_for(tubing_id) -> FlowPath; ipr_for(changes) -> IPR curve (changes: pr, j, skin).
    base holds whp, u, wc/gor (oil, wc in %) or wgr/sep_gor (gas); each case overrides some of them.
    With surface equipment, whp is the manifold pressure and 'choke' (64ths) may be sensitised.
    node="top" also builds the inflow curve at the top node (IPR, tubing traversed upwards, then the surface equipment
    downstream). The solution rate does not depend on the node, so it is solved at the bottom node in both cases."""
    hc = hc or HeatCapacities()
    scale = 1000.0 if kind == "gas" else 1.0          # VLP rate units -> IPR units
    if lift is not None and node == "top":
        raise ValueError("Gas lift: the bottom node as the solution node only.")
    if lift is not None and kind == "gas" and lift.method != GLS.METHODS[0]:
        raise ValueError("Gas lifted gas wells: fixed depth of injection only.")

    if diluent is not None and (node == "top" or kind != "oil" or lift is not None):
        raise ValueError("Diluent injection: oil producers, bottom node, without separate gas lift data.")

    def dil_rates(plist, qlist):
        qd, qg = [], []
        for p_, q_ in zip(plist, qlist):
            a, b = diluent.rates(q_, float({**base, **p_}.get("wc", 0.0)) / 100.0, p_)
            qd.append(a)
            qg.append(b)
        return np.concatenate(qd), np.concatenate(qg)

    def trav(path, whp, case, plist, qlist, u):
        """Tubing traverse; with gas lift, the lift columns of each case are added from its params."""
        if diluent is not None:
            qd, qg = dil_rates(plist, qlist)
            return DL.traverse(path, fluid, whp, case, diluent, correlation, p1, p2, u, hc, qd, qg)
        if lift is None:
            return V.traverse(path, fluid, whp, case, correlation, p1, p2, u, hc)
        cols = {}
        for p_, q_ in zip(plist, qlist):
            for k_, v_ in lift.row_inputs(p_, q_).items():
                cols.setdefault(k_, []).append(v_)
        st_ = GLS.lift_traverse(path, fluid, whp, {**case, **{k_: np.concatenate(v_) for k_, v_ in cols.items()}},
                                lift, correlation, p1, p2, u, hc)
        return st_.P, st_.T
    prep = []
    for c in cases:
        p = {**base, **c}
        changes = {k: float(p[k]) for k in IPR_KEYS if k in c}
        if kind == "oil" and "wc" in p:
            changes["wc"] = float(p["wc"]) / 100.0        # the IPR (composite below Pb) at the case's water cut
        if kind == "oil" and "gor" in p:
            changes["gor"] = float(p["gor"])              # Darcy-type IPRs re-evaluated at the case's GOR (PROSPER)
        curve = ipr_for(changes)
        aof = float(curve.aof) / scale
        rates = rate_grid(aof, method, n, user_rates)
        prep.append((p, curve, aof, rates))
    groups = {}
    for i, (p, *_rest) in enumerate(prep):
        groups.setdefault((p.get("tubing_id"), float(p["u"])), []).append(i)
    vlp = [None] * len(prep)
    pwh_store = [None] * len(prep)
    paths = {}
    for (tid, u), idx in groups.items():
        path = paths.setdefault(tid, path_for(tid))
        qs, flu = [], {}
        whps = _wellhead(kind, path, fluid, [prep[i][0] for i in idx], [prep[i][3] for i in idx], u, surface, hc,
                         [prep[i][1] for i in idx])
        for j, i in enumerate(idx):
            p, curve_i, _, rates = prep[i]
            qs.append(rates)
            pwh_store[i] = whps[j]
            for k, v in _flu(kind, p, curve_i, rates).items():
                flu.setdefault(k, []).append(v)
        case = {"q": np.concatenate(qs), **{k: np.concatenate(v) for k, v in flu.items()}}
        P, _ = trav(path, np.concatenate(whps), case, [prep[i][0] for i in idx], qs, u)
        bottom = P[:, -1]
        pos = 0
        for i in idx:
            m = len(prep[i][3])
            vlp[i] = bottom[pos:pos + m]
            pos += m
    # intersections and regula-falsi refinement (batched per group)
    results = []
    pending = []
    for i, (p, curve, aof, rates) in enumerate(prep):
        ipr_p = _ipr_at(curve, rates * scale)
        diff = vlp[i] - ipr_p
        q_min = float(rates[int(np.nanargmin(vlp[i]))])
        roots = _roots(rates, diff)
        note, pick = "", None
        if roots:
            k, q0 = roots[-1]
            # Stable when the VLP crosses the IPR from below (VLP - IPR rising with rate): a small rate increase
            # then needs more pressure than the reservoir gives, so the flow returns to the intersection. This
            # holds on the falling part of the VLP too, as long as the IPR falls faster (PROSPER reports such
            # points); the unstable left-hand intersection is the one where VLP - IPR falls with rate.
            stable_cross = diff[k + 1] > diff[k]
            if stable_cross or allow_left:
                pick = (k, q0)
                if not stable_cross:
                    note = "left-hand (unstable) intersection"
            else:
                note = "only a left-hand intersection - well unstable (left-hand intersection disallowed)"
        else:
            note = "no intersection - the well does not flow at this top-node pressure"
        results.append(CaseResult(p, rates, vlp[i], ipr_p, curve, None, None, False, note, q_min, aof,
                                  pwh=pwh_store[i] if surface is not None and surface.items else None))
        if pick:
            pending.append((i, pick))
    for _ in range(2):                      # regula falsi refinement of each picked bracket
        if not pending:
            break
        by_group = {}
        for i, (k, q0) in pending:
            p = results[i].params
            by_group.setdefault((p.get("tubing_id"), float(p["u"])), []).append((i, k, q0))
        new_pending = []
        for (tid, u), items in by_group.items():
            path = paths[tid]
            q_arr = np.array([q0 for _, _, q0 in items])
            whp = np.concatenate(_wellhead(kind, path, fluid, [results[i].params for i, _, _ in items],
                                           [np.array([q0]) for _, _, q0 in items], u, surface, hc,
                                           [results[i].ipr_curve for i, _, _ in items]))
            flu = {}
            for i, _, q0 in items:
                for kk, v in _flu(kind, results[i].params, results[i].ipr_curve, [q0]).items():
                    flu.setdefault(kk, []).append(float(v[0]))
            P, _ = trav(path, whp, {"q": q_arr, **{kk: np.array(v) for kk, v in flu.items()}},
                        [results[i].params for i, _, _ in items], [np.array([q0]) for _, _, q0 in items], u)
            wh_of = {i: w for (i, _, _), w in zip(items, whp)}
            for (i, k, q0), pv in zip(items, P[:, -1]):
                r = results[i]
                d0 = pv - _ipr_at(r.ipr_curve, np.array([q0 * scale]))[0]
                dl, dr = r.vlp[k] - r.ipr_pwf[k], r.vlp[k + 1] - r.ipr_pwf[k + 1]
                ql, qr = r.rates[k], r.rates[k + 1]
                # keep the sub-bracket containing the sign change
                if np.sign(d0) == np.sign(dl):
                    ql, dl = q0, d0
                else:
                    qr, dr = q0, d0
                q1 = ql + dl / (dl - dr) * (qr - ql) if dl != dr else q0
                # store refined bracket back into the curve arrays for the next pass
                r.rates = np.insert(r.rates, k + 1, q0)
                r.vlp = np.insert(r.vlp, k + 1, pv)
                r.ipr_pwf = np.insert(r.ipr_pwf, k + 1, pv - d0)
                if r.pwh is not None:
                    r.pwh = np.insert(r.pwh, k + 1, wh_of[i])
                kk = k if q1 <= q0 else k + 1
                new_pending.append((i, (kk, q1)))
        pending = new_pending
    for i, (k, q1) in pending:
        r = results[i]
        r.q = float(q1)
        r.pwf = float(_ipr_at(r.ipr_curve, np.array([q1 * scale]))[0])
        k_ = int(np.searchsorted(r.rates, r.q))
        lo_, hi_ = max(k_ - 1, 0), min(k_, len(r.rates) - 1)
        d_lo, d_hi = r.vlp[lo_] - r.ipr_pwf[lo_], r.vlp[hi_] - r.ipr_pwf[hi_]
        r.stable = bool(d_hi >= d_lo) if hi_ > lo_ else r.q >= r.q_vlp_min * (1 - 1e-9)
    for r in results:                       # wellhead temperatures (independent of pressure)
        path = paths[r.params.get("tubing_id")]
        qq = np.append(r.rates, r.q if r.q is not None else r.rates[-1])
        case = {"q": qq, **_flu(kind, r.params, r.ipr_curve, qq)}
        if lift is not None:
            st_ = GLS.lift_traverse(path, fluid, float(r.params["whp"]), {**case, **lift.row_inputs(r.params, qq)},
                                    lift, correlation, p1, p2, float(r.params["u"]), hc)
            tt = st_.T[:, 0]
            if r.q is not None:
                r.details["gas_lift"] = GLS.solution_state(st_, lift, path, -1)
                gap = float(st_.P[-1, -1]) - r.pwf
                if abs(gap) > max(5.0, 0.005 * r.pwf):
                    r.note = (r.note + " " if r.note else "") + (
                        f"Gas lift injection limit: the VLP jumps at this rate (lifted bottom-node pressure "
                        f"{st_.P[-1, -1]:,.0f} psia, IPR {r.pwf:,.0f} psia) - at higher rates the casing pressure "
                        "cannot keep the injection going, so the well cannot reach a steady lifted rate above it.")
        elif diluent is not None:
            qd, qg = dil_rates([r.params], [qq])
            _, T_ = DL.traverse(path, fluid, float(r.params["whp"]), case, diluent, correlation, p1, p2,
                                float(r.params["u"]), hc, qd, qg)
            tt = T_[:, 0]
            if r.q is not None:
                r.details["diluent"] = {"Diluent rate (STB/d)": float(qd[-1]), "Lift gas rate (MMscf/d)": float(qg[-1]),
                                        "Total liquid (STB/d)": float(qq[-1] + qd[-1]),
                                        "Total water cut (%)": 100 * float((qq[-1] * case["wc"][-1] + qd[-1])
                                                                            / (qq[-1] + qd[-1]))}
        elif path.thermal is not None:                 # temperature coupled to pressure: from the traverse
            whp_ = np.append(r.pwh, r.pwh[-1]) if r.pwh is not None else float(r.params["whp"])
            _, T_ = V.traverse(path, fluid, np.broadcast_to(np.asarray(whp_, float), qq.shape).copy(), case,
                               correlation, p1, p2, float(r.params["u"]), hc)
            tt = T_[:, 0]
        else:
            w, cp = fluid.mass_cp(case, hc)
            tt = V.temperature_profile(path, w, cp, float(r.params["u"]))[:, 0]
        r.wht, r.wht_q = tt[:-1], (float(tt[-1]) if r.q is not None else None)
        if r.pwh is not None and r.q is not None:
            r.pwh_q = float(np.interp(r.q, r.rates, r.pwh))
    if node == "top":
        _top_node_inflow(results, paths, fluid, kind, correlation, p1, p2, hc, surface, scale)
    for r in results:                       # multilayer IPR: the case's ratios become those at the solution rate
        if r.q is not None and getattr(r.ipr_curve, "mix", None) is not None:
            r.params = mixed_params(kind, r.params, r.ipr_curve, r.q)
    return results, paths


def _top_node_inflow(results, paths, fluid, kind, correlation, p1, p2, hc, surface, scale):
    """Inflow curve at the top node for every case (batched per flow path and U)."""
    groups = {}
    for r in results:
        r.node = "top"
        groups.setdefault((r.params.get("tubing_id"), float(r.params["u"])), []).append(r)
    for (tid, u), rs in groups.items():
        path = paths[tid]
        qs, pb, flu, plist = [], [], {}, []
        for r in rs:
            q = np.append(r.rates, r.q if r.q is not None else np.nan)
            pbot = _ipr_at(r.ipr_curve, np.nan_to_num(q, nan=0.0) * scale)
            pbot[-1] = r.pwf if r.q is not None else np.nan
            qs.append(np.nan_to_num(q, nan=r.rates[0]))
            pb.append(pbot)
            for k, v in _flu(kind, r.params, r.ipr_curve, qs[-1]).items():
                flu.setdefault(k, []).append(v)
        case = {"q": np.concatenate(qs), **{k: np.concatenate(v) for k, v in flu.items()}}
        P, T = V.traverse_up(path, fluid, np.concatenate(pb), case, correlation, p1, p2, u, hc)
        top = P[:, 0]
        if surface is not None and surface.items:
            by_choke = {}
            pos = 0
            for r, q in zip(rs, qs):
                by_choke.setdefault(_surface_key(r.params), []).append((pos, len(q), r))
                pos += len(q)
            out = np.full_like(top, np.nan)
            for _, members in by_choke.items():
                idx = np.concatenate([np.arange(a, a + m) for a, m, _ in members])
                sub = {k: v[idx] for k, v in case.items()}
                p_wh = top[idx].copy()
                sol = np.concatenate([np.arange(a, a + m) == a + m - 1 for a, m, _ in members])
                p_wh[sol] *= 1 + 2e-4      # solution rate sits on the critical-choke boundary: step just inside it
                out[idx] = SF.manifold_pressure(surface.items_for(members[0][2].params), fluid, kind, p_wh, sub,
                                                T[idx, 0], surface.correlation, hc, surface.t_surround, surface.u,
                                                surface.choke_method, surface.cd, p2=surface.friction)
            top = out
        pos = 0
        for r, q in zip(rs, qs):
            m = len(q)
            r.top_in = top[pos:pos + m - 1]
            r.top_in_q = float(top[pos + m - 1]) if r.q is not None else None
            pos += m


def rate_table(kind, r: CaseResult) -> pd.DataFrame:
    """PROSPER-style table of the VLP/IPR curves of one case."""
    rows = []
    for i, q in enumerate(r.rates):
        sr = surface_rates(kind, mixed_params(kind, r.params, r.ipr_curve, q), q)
        row = {"Point": i + 1, **sr}
        if r.node == "top" and r.top_in is not None:
            row["Inflow pressure at top node (psia)"] = r.top_in[i]
            row["Top node pressure (psia)"] = float(r.params["whp"])
        row.update({"VLP pressure (psia)": r.vlp[i], "IPR pressure (psia)": r.ipr_pwf[i]})
        if r.node != "top":
            row["Top node pressure (psia)"] = float(r.params["whp"])
        if r.pwh is not None:
            row["Wellhead pressure (psia)"] = r.pwh[i]
        row["Wellhead temperature (°F)"] = r.wht[i]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- details
def _free_gas(fluid, kind, s, case, p, t):
    """Free gas at the node in MMscf/d: in-situ gas mass rate over the gas density at standard conditions."""
    if hasattr(fluid, "free_gas_mmscfd"):
        return fluid.free_gas_mmscfd(float(np.atleast_1d(p)[0]), float(np.atleast_1d(t)[0]), case)
    rho_g = float(s["rho_g"][0])
    qg = float(s["qg"][0])                                   # ft3/s in situ
    sg = getattr(getattr(fluid, "inp", None), "gas_sg", None)
    if not sg or not np.isfinite(rho_g):
        return np.nan
    return qg * rho_g * 86400.0 / (V.RHO_AIR_SC * sg) / 1e6


def solution_details(path, fluid, kind, params, q, correlation, p1=1.0, p2=1.0, hc=None,
                     surface: SurfaceSetup | None = None, comp=None, t_bottom=None):
    """Traverse at the solution rate with a breakdown of pressure losses, velocities and restriction regimes.
    With surface equipment, det also carries the manifold and the surface losses, and the regimes dict the choke."""
    hc = hc or HeatCapacities()
    case = {"q": np.array([q]), **{k: np.array([v]) for k, v in _case_arrays(kind, params).items()}}
    whp, u = float(params["whp"]), float(params["u"])
    surf_rows, surf_det = None, {}
    if surface is not None and surface.items:
        w, cp = fluid.mass_cp(case, hc)
        t_wh = V.temperature_profile(path, w, cp, u, t_bottom)[:, 0]
        pwh, surf_rows, t_man, notes = SF.wellhead_pressure(
            surface.items_for(params), fluid, kind, np.array([whp]), case, t_wh, surface.correlation, hc,
            surface.t_surround, surface.u, surface.choke_method, surface.cd, p2=surface.friction, detail=True)
        dpc = sum(float(r["dP (psi)"][0]) for r in surf_rows if r["Type"] == "Choke")
        dpl = sum(float(r["dP (psi)"][0]) for r in surf_rows if r["Type"] == "Pipe")
        surf_det = {"Manifold pressure (psia)": whp, "Manifold temperature (°F)": float(t_man[0]),
                    "dP choke (psi)": dpc, "dP flowline and fittings (psi)": dpl}
        for r in surf_rows:
            if r["Type"] == "Pump":
                surf_det.update({k: float(np.atleast_1d(v)[0]) for k, v in r["_pump"].items()})
        whp = float(pwh[0])
    P, T = V.traverse(path, fluid, whp, case, correlation, p1, p2, u, hc, t_bottom)
    P, T = P[0], T[0]
    corr = CORRELATIONS[correlation]
    rows, dpg, dpf, transit = [], 0.0, 0.0, 0.0
    vms, ves = [], []
    comp_ = comp if comp is not None else CX.CompOptions()
    comp_rows = []
    for i in range(len(path.md) - 1):
        pm, tm, L = 0.5 * (P[i] + P[i + 1]), 0.5 * (T[i] + T[i + 1]), path.seg_len[i]
        s = fluid.state(np.array([pm]), np.array([tm]), case)
        a = path.area[i]
        vsl, vsg = max(float(s["ql"][0]) / a, 1e-12), max(float(s["qg"][0]) / a, 0.0)
        args = (np.array([vsl]), np.array([vsg]), s["rho_l"], np.maximum(s["rho_g"], 1e-4), s["mu_l"],
                np.maximum(s["mu_g"], 1e-4), np.maximum(s["sig"], 0.5), path.d_h[i], path.rough[i],
                path.sin_t[i], np.array([pm]))
        lam = vsl / max(vsl + vsg, 1e-30)
        if lam > 0.99999 or lam < 1e-6:
            e, f, hl = no_slip(*args)
        elif corr.__name__ in NEEDS_GLR:
            e, f, hl = corr(*args, glr=V._glr(case, kind))
        else:
            e, f, hl = corr(*args)
        dpg += p1 * float(e[0]) * L
        dpf += p2 * float(f[0]) * L
        vm = vsl + vsg
        rho_ns = lam * float(s["rho_l"][0]) + (1 - lam) * float(s["rho_g"][0])
        mu_ns = lam * float(s["mu_l"][0]) + (1 - lam) * float(s["mu_g"][0])
        cv = CX.node_values(kind, vm, vsg, rho_ns, mu_ns, float(s["rho_l"][0]), float(s["rho_g"][0]),
                            float(s["sig"][0]), path.d_h[i] * 12, _free_gas(fluid, kind, s, case, pm, tm), comp_)
        cv.update(CX.path_values(path, i, fluid, kind, case, s, pm, tm, comp_))
        comp_rows.append(cv)
        ve = cv["Erosional Velocity (ft/s)"]
        transit += L / max(vm, 1e-9)
        vms.append(vm)
        ves.append(ve)
        hl0 = float(hl[0])
        rows.append({"MD (ft)": path.md[i + 1], "TVD (ft)": path.tvd[i + 1], "Pressure (psia)": P[i + 1],
                     "Temperature (°F)": T[i + 1], "Liquid holdup": hl0, "No-slip holdup": lam,
                     "Superficial liquid velocity (ft/s)": vsl, "Superficial gas velocity (ft/s)": vsg,
                     "Mixture velocity (ft/s)": vm, "Erosional velocity (ft/s)": ve,
                     "In-situ density (lb/ft3)": hl0 * float(s["rho_l"][0]) + (1 - hl0) * float(s["rho_g"][0]),
                     "Elevation gradient (psi/ft)": p1 * float(e[0]), "Friction gradient (psi/ft)": p2 * float(f[0]),
                     "Max. grain size diameter (in)": cv["Max. Grain Size Diameter (in)"],
                     "Liquid loading velocity (ft/s)": cv["Liquid Loading Velocity (ft/s)"],
                     "Erosion rate (mm/year)": cv["Erosion Rate (mm/year)"],
                     "Corrosion rate (0.001 in/yr)": cv["Corrosion Rate (0.001 in/yr)"]})
    regimes = {}
    for r in surf_rows or []:
        if r["Type"] == "Choke":
            regimes[f"Surface choke - {r['Item']} ({r['Size']})"] = (str(r["Regime"][0]), float(r["dP (psi)"][0]))
    for j, (node, d_in) in enumerate(sorted(path.restrictions.items()), 1):
        s = fluid.state(np.array([P[node]]), np.array([T[node]]), case)
        dp = float(V._restriction_dp(s, d_in)[0])
        p_up, p_dn = P[node], P[node] - dp
        regimes[f"Restriction {j} at {path.md[node]:,.0f} ft (ID {d_in:g} in)"] = (
            ("Critical" if p_dn / max(p_up, 1e-9) < 0.53 else "SubCritical"), dp)
    det = {
        **surf_det,
        "Wellhead pressure (psia)": P[0], "Wellhead temperature (°F)": T[0],
        "Bottom node pressure (psia)": P[-1], "Bottom node temperature (°F)": T[-1],
        "dP friction - production string (psi)": dpf, "dP gravity - production string (psi)": dpg,
        "dP restrictions (psi)": sum(v[1] for k, v in regimes.items() if k.startswith("Restriction")),
        "Cumulative transit time (s)": transit,
        "Max. no-slip velocity (ft/s)": max(vms), "Min. no-slip velocity (ft/s)": min(vms),
        "Max. erosional velocity (ft/s)": max(ves), "Min. erosional velocity (ft/s)": min(ves),
        "Max. velocity / erosional velocity": max(v / e for v, e in zip(vms, ves)),
        "Max. pressure (psia)": float(P.max()), "Min. pressure (psia)": float(P.min()),
        **CX.summary(comp_rows),
    }
    if surf_rows:
        det["_surface"] = surface_table(surf_rows)
    top = {k: np.nan for k in rows[0]} if rows else {}
    top.update({"MD (ft)": path.md[0], "TVD (ft)": path.tvd[0], "Pressure (psia)": P[0], "Temperature (°F)": T[0]})
    return det, regimes, pd.DataFrame([top] + rows)


def surface_table(rows) -> pd.DataFrame:
    """Item-by-item surface breakdown at one rate (wellhead first)."""
    out = []
    for r in rows:
        out.append({"Item": r["Item"], "Type": r["Type"], "Size": r["Size"],
                    "Inlet pressure (psia)": float(r["Upstream pressure (psia)"][0]),
                    "Outlet pressure (psia)": float(r["Downstream pressure (psia)"][0]),
                    "dP (psi)": float(r["dP (psi)"][0]), "Temperature (°F)": float(np.atleast_1d(r["Temperature (°F)"])[0]),
                    "Regime": str(r["Regime"][0]) if r["Type"] == "Choke" else ""})
    return pd.DataFrame(out)


def surface_rates(kind, params, q):
    """Surface rates at a solution rate: oil (liquid, oil, water STB/d, gas MMscf/d) / gas (gas MMscf/d,
    condensate and water STB/d)."""
    if kind == "oil":
        wc = params["wc"] / 100.0
        qo = q * (1 - wc)
        return {"Liquid rate (STB/d)": q, "Oil rate (STB/d)": qo, "Water rate (STB/d)": q * wc,
                "Gas rate (MMscf/d)": qo * params["gor"] / 1e6}
    sg = params.get("sep_gor", 0.0)
    cgr = 1e6 / sg if sg and sg > 0 else 0.0
    return {"Gas rate (MMscf/d)": q, "Condensate rate (STB/d)": q * cgr, "Water rate (STB/d)": q * params["wgr"]}
