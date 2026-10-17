"""Continuous gas lift: lift-gas column, gas-lifted traverses, performance curve, unloading design.

Hydraulics (this file, first part)
----------------------------------
LiftGas                 injection gas (gravity, impurities, surface temperature).
casing_pressure         static gas column in the annulus (no annulus friction - PROSPER's default
                        "No friction loss in annulus"); the casing temperature is a straight line between the
                        surface injection temperature and the formation temperature at the bottom node
                        (PROSPER's casing temperature for gas lift).
lifted_traverse         top-down tubing traverse with the injection point found on the way down: above it the
                        liquid carries the formation GOR plus the lift gas, below it the formation GOR only. The
                        injection point is where the tubing pressure reaches the casing pressure less the
                        desired valve dP, or the maximum injection depth (or a fixed depth). The switch is
                        placed inside a segment by linear interpolation, so the injection depth is not tied to
                        the nodes of the flow path.
performance_curve       liquid/oil rate against lift-gas rate: for every gas rate, the VLP (with the injection
                        depth found as above for every liquid rate) is intersected with the IPR.

Units: psia, degF, ft MD/TVD; liquid STB/d, lift gas MMscf/d.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import blackoil as bo
from . import vlp as V
from .correlations import CORRELATIONS
from .equipment import HeatCapacities

RHO_AIR_SC = 0.0764          # lb/scf


# ================================================================ lift gas and casing column
@dataclass
class LiftGas:
    sg: float = 0.70
    co2: float = 0.0          # mole %
    n2: float = 0.0
    h2s: float = 0.0
    t_surface: float = 80.0   # injection gas temperature at the casing head, degF

    @property
    def inp(self) -> bo.PVTInputs:
        return bo.PVTInputs(rsb=1.0, api=35.0, gas_sg=self.sg, h2s=self.h2s, co2=self.co2, n2=self.n2)

    def density(self, p, t):
        return bo.gas_props(self.inp, p, t)[2]

    def bg(self, p, t):
        return bo.gas_props(self.inp, p, t)[1]

    @property
    def rho_sc(self):
        return RHO_AIR_SC * self.sg


def casing_temperature(path: V.FlowPath, gas: LiftGas):
    """Straight line in TVD from the surface injection temperature to the formation temperature at the bottom."""
    tvd = path.tvd
    span = max(tvd[-1] - tvd[0], 1e-9)
    return gas.t_surface + (path.t_form[-1] - gas.t_surface) * (tvd - tvd[0]) / span


def casing_pressure(path: V.FlowPath, gas: LiftGas, p_surface, t_casing=None):
    """Static gas column: casing pressure at every node (surface pressures x nodes), psia."""
    ps = np.atleast_1d(np.asarray(p_surface, float))
    tc = casing_temperature(path, gas) if t_casing is None else np.asarray(t_casing, float)
    out = np.empty((ps.size, len(path.md)))
    p = ps.copy()
    out[:, 0] = p
    for i in range(len(path.md) - 1):
        dz = path.tvd[i + 1] - path.tvd[i]
        g0 = gas.density(p, tc[i]) / 144.0
        g1 = gas.density(p + g0 * dz, tc[i + 1]) / 144.0
        p = p + 0.5 * (g0 + g1) * dz
        out[:, i + 1] = p
    return out


def gas_column(gas: LiftGas, p_top, tvd_top, tvd_bottom, t_top, t_bottom, n=40):
    """Pressure at tvd_bottom of a static gas column with p_top at tvd_top (linear temperature); vectorised."""
    p = np.asarray(p_top, float) + 0.0
    zs = np.linspace(0.0, 1.0, n + 1)
    dz = (np.asarray(tvd_bottom, float) - np.asarray(tvd_top, float)) / n
    tt, tb = np.asarray(t_top, float), np.asarray(t_bottom, float)
    for a, b in zip(zs[:-1], zs[1:]):
        t0, t1 = tt + (tb - tt) * a, tt + (tb - tt) * b
        g0 = gas.density(p, t0) / 144.0
        g1 = gas.density(p + g0 * dz, t1) / 144.0
        p = p + 0.5 * (g0 + g1) * dz
    return p


def surface_from_depth(gas: LiftGas, p_depth, tvd_top, tvd, t_top, t_depth):
    """Surface pressure of a static gas column that gives p_depth at tvd (inverse of gas_column)."""
    p_depth = np.asarray(p_depth, float)
    ps = p_depth * 0.85
    for _ in range(30):
        f = gas_column(gas, ps, tvd_top, tvd, t_top, t_depth) - p_depth
        if np.all(np.abs(f) < 1e-4):
            break
        dfp = (gas_column(gas, ps + 1.0, tvd_top, tvd, t_top, t_depth) - p_depth - f)
        ps = ps - f / np.where(np.abs(dfp) > 1e-9, dfp, 1.0)
    return ps


# ================================================================ flow-path helpers
def with_nodes(path: V.FlowPath, mds) -> V.FlowPath:
    """Copy of the flow path with extra nodes at the given measured depths (segment properties kept)."""
    mds = sorted({float(m) for m in np.atleast_1d(mds) if path.md[0] < m < path.md[-1]})
    mds = [m for m in mds if np.min(np.abs(path.md - m)) > 0.5]
    if not mds:
        return path
    md = np.sort(np.concatenate([path.md, mds]))
    seg_src = np.clip(np.searchsorted(path.md, 0.5 * (md[1:] + md[:-1])) - 1, 0, len(path.seg_len) - 1)
    tvd = np.interp(md, path.md, path.tvd)
    seg = np.diff(md)
    restr = {int(np.argmin(np.abs(md - path.md[k]))): d for k, d in path.restrictions.items()}
    return V.FlowPath(md, tvd, np.interp(md, path.md, path.t_form), seg, path.sin_t[seg_src], path.area[seg_src],
                      path.d_h[seg_src], path.rough[seg_src], path.d_ht[seg_src], restr)


def tvd_at(path, md):
    return np.interp(md, path.md, path.tvd)


def md_at_tvd(path, tvd):
    return np.interp(tvd, path.tvd, path.md)


# ================================================================ gas-lifted traverse
def lifted_case(case, qgi):
    """Case dict above the injection point: formation GOR plus the lift gas per STB of oil."""
    q, wc = np.asarray(case["q"], float), np.clip(np.asarray(case["wc"], float), 0, 0.9999)
    qo = np.maximum(q * (1 - wc), 1e-6)
    return {**case, "gor": np.asarray(case["gor"], float) + np.asarray(qgi, float) * 1e6 / qo}


def _temperature_split(path, fluid, case, qgi, inj_md, u, hc, gas_sg):
    """Flowing temperature (cases x nodes) with the lift gas added to the heat balance above the injection point."""
    w, cp = fluid.mass_cp(case, hc)
    wg = np.asarray(qgi, float) * 1e6 * RHO_AIR_SC * gas_sg / 24.0
    n = len(path.md)
    T = np.empty((w.size, n))
    T[:, -1] = path.t_form[-1]
    for i in range(n - 2, -1, -1):
        L = path.seg_len[i]
        mid = 0.5 * (path.md[i] + path.md[i + 1])
        above = mid < inj_md
        ww = np.where(above, w + wg, w)
        cc = np.where(above, (w * cp + wg * hc.gas) / np.maximum(w + wg, 1e-12), cp)
        a = u * np.pi * path.d_ht[i] / np.maximum(ww * cc, 1e-9)
        tfb, tft = path.t_form[i + 1], path.t_form[i]
        g = (tft - tfb) / L
        t0 = T[:, i + 1]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            tn = tfb + g * L - g / a + (t0 - tfb + g / a) * np.exp(-a * L)
        T[:, i] = np.where(a * L < 1e-9, t0, tn)
    return T


@dataclass
class Lifted:
    P: np.ndarray            # tubing pressure, cases x nodes
    T: np.ndarray            # tubing temperature
    inj_md: np.ndarray       # injection depth per case, ft MD
    inj_p: np.ndarray        # tubing pressure at the injection point
    inj_t: np.ndarray        # tubing temperature at the injection point
    limited: np.ndarray      # True where the maximum (or fixed) injection depth decided the point
    no_injection: np.ndarray  # True where the casing cannot inject anywhere (casing < tubing + dP at the top)


def lifted_traverse(*args, **kw):
    """Top-down traverse of a gas-lifted oil well (see _lifted_traverse)."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _lifted_traverse(*args, **kw)


def _lifted_traverse(path: V.FlowPath, fluid, top_p, case, qgi, correlation="hagedorn_brown", p1=1.0, p2=1.0, u=8.0,
                     hc: HeatCapacities | None = None, pc=None, dp_valve=0.0, max_md=None, fixed_md=None,
                     gas_sg=0.7, passes=2):
    """Top-down traverse of a gas-lifted oil well. case: q (liquid STB/d), wc (fraction), gor (formation, scf/STB);
    qgi: lift gas MMscf/d (per case). pc: casing pressure at the nodes (nodes or cases x nodes) - the injection
    point is the first depth where tubing + dp_valve reaches it; max_md caps it; fixed_md forces it."""
    hc = hc or HeatCapacities()
    case = dict(case)
    case["_top"], case["_qgi"] = top_p, qgi
    if fixed_md is not None:
        case["_fix"] = fixed_md
    case = V._broadcast_case(case)
    top, qg = case.pop("_top"), case.pop("_qgi")
    fix = case.pop("_fix", None)
    nc, n = top.size, len(path.md)
    corr = CORRELATIONS[correlation]
    lift = lifted_case(case, qg)
    max_md = path.md[-1] if max_md is None else min(float(max_md), path.md[-1])
    if pc is not None:
        pc = np.broadcast_to(np.atleast_2d(np.asarray(pc, float)), (nc, n))
    inj_guess = np.full(nc, max_md) if fix is None else fix
    for it in range(max(passes, 1)):
        T = _temperature_split(path, fluid, case, qg, inj_guess, u, hc, gas_sg)
        P = np.empty((nc, n))
        p = np.maximum(top, 14.7)
        P[:, 0] = p
        done = np.zeros(nc, bool)
        inj_md = np.full(nc, np.nan)
        inj_p = np.full(nc, np.nan)
        limited = np.zeros(nc, bool)
        none = np.zeros(nc, bool)
        if pc is not None and fix is None:
            none = pc[:, 0] - dp_valve <= p
        for i in range(n - 1):
            L = path.seg_len[i]
            m0, m1 = path.md[i], path.md[i + 1]
            Ti, Tj = T[:, i], T[:, i + 1]

            def heun(p_start, c, frac0, frac1):
                ln = (frac1 - frac0) * L
                ta, tb = Ti + (Tj - Ti) * frac0, Ti + (Tj - Ti) * frac1
                g0, _ = V._gradient(path, fluid, i, p_start, ta, c, corr, p1, p2)
                g1, _ = V._gradient(path, fluid, i, np.maximum(p_start + g0 * ln, 14.7), tb, c, corr, p1, p2)
                return np.maximum(p_start + 0.5 * (g0 + g1) * ln, 14.7)

            live = ~done
            p_new = p.copy()
            if np.any(done):
                p_new = np.where(done, heun(p, case, 0.0, 1.0), p_new)
            if np.any(live):
                pl = heun(p, lift, 0.0, 1.0)
                f = np.full(nc, np.inf)
                if fix is not None:
                    hit = (fix > m0 - 1e-9) & (fix <= m1 + 1e-9)
                    f = np.where(hit, np.clip((fix - m0) / L, 0, 1), f)
                    limited |= hit & live
                else:
                    if pc is not None:
                        d0 = pc[:, i] - dp_valve - p
                        d1 = pc[:, i + 1] - dp_valve - pl
                        cross = (d1 <= 0) & live
                        fc = np.where(d0 <= 0, 0.0, d0 / np.where(d0 - d1 > 1e-12, d0 - d1, 1.0))
                        f = np.where(cross, np.clip(fc, 0, 1), f)
                    if m0 - 1e-9 <= max_md <= m1 + 1e-9:
                        fm = np.clip((max_md - m0) / L, 0, 1)
                        limited |= live & (fm < f)
                        f = np.minimum(f, fm)
                    elif i == n - 2:
                        limited |= live & ~np.isfinite(f)
                        f = np.where(np.isfinite(f), f, 1.0)
                sw = live & np.isfinite(f)
                if np.any(sw):
                    fs = np.where(sw, f, 0.0)
                    px = p + fs * (pl - p)                    # linear inside the segment
                    inj_md = np.where(sw, m0 + fs * L, inj_md)
                    inj_p = np.where(sw, px, inj_p)
                    # formation GOR from the injection point to the end of the segment
                    ln_rest = (1 - fs) * L
                    ta = Ti + (Tj - Ti) * fs
                    g0, _ = V._gradient(path, fluid, i, px, ta, case, corr, p1, p2)
                    g1, _ = V._gradient(path, fluid, i, np.maximum(px + g0 * ln_rest, 14.7), Tj, case, corr, p1, p2)
                    pb = np.maximum(px + 0.5 * (g0 + g1) * ln_rest, 14.7)
                    p_new = np.where(sw, pb, np.where(live, pl, p_new))
                    done |= sw
                else:
                    p_new = np.where(live, pl, p_new)
            p = p_new
            if (i + 1) in path.restrictions:
                st = fluid.state(p, T[:, i + 1], {k: np.where(done, case[k], lift[k]) for k in case})
                p = p + V._restriction_dp(st, path.restrictions[i + 1])
            P[:, i + 1] = p
        inj_md = np.where(np.isfinite(inj_md), inj_md, path.md[-1])
        inj_p = np.where(np.isfinite(inj_p), inj_p, P[:, -1])
        if fix is not None or np.allclose(inj_md, inj_guess, atol=25.0):
            break
        inj_guess = inj_md
    inj_t = np.array([np.interp(m, path.md, T[k]) for k, m in enumerate(inj_md)])
    return Lifted(P, T, inj_md, inj_p, inj_t, limited, none)


def pressure_at(path, P, md):
    """Interpolate a (cases x nodes) profile at one measured depth per case (or one depth for all)."""
    md = np.broadcast_to(np.asarray(md, float), (P.shape[0],))
    return np.array([np.interp(m, path.md, row) for m, row in zip(md, P)])


# ================================================================ performance curve
def ipr_pwf(curve, q):
    order = np.argsort(curve.q)
    qq, pp = np.asarray(curve.q)[order], np.asarray(curve.pwf)[order]
    out = np.interp(q, qq, pp)
    return np.where(np.asarray(q) <= qq.max() * (1 + 1e-9), out, np.nan)


def _last_root(rates, diff):
    ok = np.isfinite(diff)
    r, d = rates[ok], diff[ok]
    s = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if not len(s):
        return None
    k = s[-1]
    return k, r, d


@dataclass
class OperatingPoint:
    qgi: float               # MMscf/d
    q: float | None          # liquid STB/d
    pwf: float | None
    inj_md: float | None
    inj_p: float | None      # tubing pressure at the injection point
    pc_inj: float | None     # casing pressure at the injection point
    limited: bool = False
    note: str = ""


def solve_rate(path, fluid, ipr, top_p, wc, gor, qgi, correlation, p1, p2, u, hc, pc, dp_valve, max_md, gas_sg,
               rates=None, fixed_md=None, refine=2):
    """Operating point (VLP with lift gas against the IPR) for each lift-gas rate in qgi (vectorised).
    Returns a list of OperatingPoint."""
    qgi = np.atleast_1d(np.asarray(qgi, float))
    aof = float(np.max(ipr.q))
    if rates is None:
        rates = np.unique(np.concatenate([np.geomspace(max(aof * 0.005, 1.0), aof * 0.999, 22),
                                          np.linspace(aof * 0.05, aof * 0.999, 14)]))
    nq, ng = len(rates), len(qgi)
    qq = np.tile(rates, ng)
    gg = np.repeat(qgi, nq)
    fm = None if fixed_md is None else np.repeat(np.broadcast_to(np.asarray(fixed_md, float), qgi.shape), nq)
    res = lifted_traverse(path, fluid, top_p, {"q": qq, "wc": wc, "gor": gor}, gg, correlation, p1, p2, u, hc,
                          pc=pc, dp_valve=dp_valve, max_md=max_md, fixed_md=fm, gas_sg=gas_sg, passes=1)
    bhp = res.P[:, -1].reshape(ng, nq)
    ipr_p = ipr_pwf(ipr, rates)
    out = []
    brackets = []
    for j, g in enumerate(qgi):
        diff = bhp[j] - ipr_p
        r = _last_root(rates, diff)
        if r is None:
            out.append(OperatingPoint(float(g), None, None, None, None, None, note="no intersection - the well "
                                      "does not flow at this lift-gas rate"))
            brackets.append(None)
            continue
        k, rr, dd = r
        q0 = rr[k] + dd[k] / (dd[k] - dd[k + 1]) * (rr[k + 1] - rr[k])
        out.append(OperatingPoint(float(g), float(q0), None, None, None, None))
        brackets.append([rr[k], dd[k], rr[k + 1], dd[k + 1], q0])
    for _ in range(refine):
        idx = [j for j, b in enumerate(brackets) if b is not None]
        if not idx:
            break
        q_try = np.array([brackets[j][4] for j in idx])
        r2 = lifted_traverse(path, fluid, top_p, {"q": q_try, "wc": wc, "gor": gor}, qgi[idx], correlation, p1, p2,
                             u, hc, pc=pc, dp_valve=dp_valve, max_md=max_md,
                             fixed_md=None if fixed_md is None else np.broadcast_to(fixed_md, qgi.shape)[idx],
                             gas_sg=gas_sg, passes=1)
        d_try = r2.P[:, -1] - ipr_pwf(ipr, q_try)
        for a, j in enumerate(idx):
            ql, dl, qr, dr, q0 = brackets[j]
            if np.sign(d_try[a]) == np.sign(dl):
                ql, dl = q0, d_try[a]
            else:
                qr, dr = q0, d_try[a]
            q1 = ql + dl / (dl - dr) * (qr - ql) if dl != dr else q0
            brackets[j] = [ql, dl, qr, dr, q1]
    # final state at the solution
    idx = [j for j, b in enumerate(brackets) if b is not None]
    if idx:
        q_fin = np.array([brackets[j][4] for j in idx])
        r3 = lifted_traverse(path, fluid, top_p, {"q": q_fin, "wc": wc, "gor": gor}, qgi[idx], correlation, p1, p2,
                             u, hc, pc=pc, dp_valve=dp_valve, max_md=max_md,
                             fixed_md=None if fixed_md is None else np.broadcast_to(fixed_md, qgi.shape)[idx],
                             gas_sg=gas_sg, passes=2)
        pcs = None if pc is None else np.atleast_2d(pc)
        for a, j in enumerate(idx):
            op = out[j]
            op.q = float(q_fin[a])
            op.pwf = float(ipr_pwf(ipr, q_fin[a]))
            op.inj_md = float(r3.inj_md[a])
            op.inj_p = float(r3.inj_p[a])
            op.limited = bool(r3.limited[a])
            if pcs is not None:
                row = pcs[0] if pcs.shape[0] == 1 else pcs[j]
                op.pc_inj = float(np.interp(op.inj_md, path.md, row))
            if r3.no_injection[a]:
                op.note = "casing pressure below tubing pressure + valve dP at the wellhead - no injection possible"
    return out


@dataclass
class Economics:
    oil_price: float = 70.0        # $/STB
    gas_price: float = 3.0         # $/Mscf sales gas (produced formation gas)
    water_cost: float = 1.0        # $/STB produced water handling
    lift_gas_cost: float = 0.5     # $/Mscf lift gas (compression)

    def revenue(self, q, wc, gor, qgi):
        qo, qw = q * (1 - wc), q * wc
        return (qo * self.oil_price + qo * gor / 1000.0 * self.gas_price - qw * self.water_cost
                - qgi * 1000.0 * self.lift_gas_cost)


def performance_curve(path, fluid, ipr, top_p, wc, gor, qgi_max, correlation="hagedorn_brown", p1=1.0, p2=1.0,
                      u=8.0, hc=None, pc=None, dp_valve=100.0, max_md=None, gas_sg=0.7, n=12, fixed_md=None,
                      economics: Economics | None = None) -> pd.DataFrame:
    """Liquid and oil rate against lift-gas rate (0 .. qgi_max MMscf/d), each with its own injection depth."""
    qgi = np.unique(np.concatenate([[0.0], np.linspace(qgi_max / (n - 1), qgi_max, n - 1)]))
    ops = solve_rate(path, fluid, ipr, top_p, wc, gor, qgi, correlation, p1, p2, u, hc, pc, dp_valve, max_md,
                     gas_sg, fixed_md=fixed_md)
    rows = []
    for op in ops:
        q = op.q if op.q is not None else 0.0
        rows.append({"Gas injection rate (MMscf/d)": op.qgi, "Liquid rate (STB/d)": q, "Oil rate (STB/d)": q * (1 - wc),
                     "Water rate (STB/d)": q * wc, "Bottom node pressure (psia)": op.pwf,
                     "Injection depth (ft MD)": op.inj_md, "Tubing pressure at injection (psia)": op.inj_p,
                     "Casing pressure at injection (psia)": op.pc_inj,
                     "Injection GLR (scf/STB)": op.qgi * 1e6 / q if q > 0 else np.nan,
                     "Revenue ($/d)": economics.revenue(q, wc, gor, op.qgi) if economics else np.nan,
                     "Note": op.note})
    return pd.DataFrame(rows)


def optimum(pc_table: pd.DataFrame, by="Oil rate (STB/d)", qgi_available=None, q_liquid_max=None):
    """Design point from a performance curve: the maximum of `by` (oil rate or revenue), smoothly located with a
    parabola through the best three points, then limited by the available gas and the maximum liquid rate."""
    t = pc_table.dropna(subset=[by]).sort_values("Gas injection rate (MMscf/d)")
    g = t["Gas injection rate (MMscf/d)"].to_numpy(float)
    y = t[by].to_numpy(float)
    ql = t["Liquid rate (STB/d)"].to_numpy(float)
    if not len(g):
        return None
    k = int(np.argmax(y))
    g_opt = g[k]
    if 0 < k < len(g) - 1:
        a = np.polyfit(g[k - 1:k + 2], y[k - 1:k + 2], 2)
        if a[0] < 0:
            g_opt = float(np.clip(-a[1] / (2 * a[0]), g[k - 1], g[k + 1]))
    note = "maximum of the performance curve"
    if qgi_available is not None and g_opt > qgi_available:
        g_opt, note = float(qgi_available), "limited by the maximum gas available"
    q_at = float(np.interp(g_opt, g, ql))
    if q_liquid_max is not None and q_at > q_liquid_max:
        inc = np.maximum.accumulate(ql)
        g_opt = float(np.interp(q_liquid_max, inc, g)) if inc[-1] >= q_liquid_max else g_opt
        q_at, note = float(q_liquid_max), "limited by the maximum liquid rate"
    return {"qgi": g_opt, "q": q_at, "note": note}


# ================================================================ valves: flow, mechanics, dome temperature
PORTS_64 = (8, 10, 12, 14, 16, 20, 24, 28, 32)          # standard port / orifice sizes, 64ths in


@dataclass
class ValveType:
    name: str
    bellows_area: float          # Ab, in2
    max_port_64: int
    ports_64: tuple = PORTS_64

    def r(self, port_64):
        ap = np.pi / 4 * (np.asarray(port_64, float) / 64.0) ** 2
        return ap / self.bellows_area

    @property
    def ports(self):
        return tuple(p for p in self.ports_64 if p <= self.max_port_64)


VALVES = {
    "Generic 1-in IPO (Ab = 0.31 in2)": ValveType("Generic 1-in IPO (Ab = 0.31 in2)", 0.31, 16),
    "Generic 1.5-in IPO (Ab = 0.77 in2)": ValveType("Generic 1.5-in IPO (Ab = 0.77 in2)", 0.77, 32),
}
ORIFICE_PORTS = (8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 48)


# ================================================================ design input
DESIGN_RATE = ("Entered by user", "Calculated from max production", "Calculated from max revenue")
VALVE_TYPES = ("Casing sensitive",)
INJECTION_POINT = ("Orifice", "Valve")
SPACING = ("Normal", "Spacing line")
ORIFICE_SIZING = ("Calculated dP at orifice", "Min dP across orifice")


@dataclass
class DesignInput:
    design_rate: str = DESIGN_RATE[1]
    valve_type: str = VALVE_TYPES[0]
    injection_point: str = INJECTION_POINT[0]
    spacing: str = SPACING[0]
    dome_high_p: bool = True              # Winkler-Eads above 1200 psig
    check_ipr: bool = True                # check rate conformance with the IPR
    use_ipr_unloading: bool = True
    orifice_sizing: str = ORIFICE_SIZING[0]
    q_design: float = 1000.0              # liquid STB/d (entered by user)
    q_liquid_max: float = 5000.0
    qgi_max: float = 2.0                  # MMscf/d, normal operation
    qgi_unload_max: float = 2.0           # MMscf/d, during unloading
    top_p: float = 200.0                  # flowing top-node pressure, psia
    top_p_unload: float = 200.0
    p_inj: float = 1200.0                 # operating injection pressure (casing head), psia
    p_ko: float = 1200.0                  # kick-off injection pressure, psia
    dp_valve: float = 100.0               # desired dP across the operating valve/orifice, psi
    max_md: float = 1e9                   # maximum injection depth, ft MD
    gor: float = 300.0                    # formation (total) GOR, scf/STB
    wc: float = 0.0                       # fraction
    load_grad: float = 0.45               # static gradient of the load fluid, psi/ft
    min_transfer_pct: float = 0.0         # minimum transfer dP, % (use_ipr_unloading False)
    safety_last: float = 0.0              # extra dP to close the last unloading valve, psi
    dp_close: float = 20.0                # casing pressure drop per valve (min CHP decrease), psi
    dp_close_spread: bool = True          # drop = max(entered, valve spread R (Pvo - Pt)) - PROSPER "Pvo = gas pressure"
    tc_derate_valve: float = 0.0          # Thornhill-Craver de-rating, %
    tc_derate_orifice: float = 0.0
    min_spacing: float = 250.0            # minimum valve spacing, ft MD
    margin_top: float = 50.0              # casing pressure margin for the first valve, psi
    margin: float = 20.0                  # casing pressure margin (transfer safety) for the deeper valves, psi
    valve_t_profile: str = "Flowing"      # valve temperatures: flowing traverse at the design rate
    valve: str = next(iter(VALVES))
    economics: Economics = field(default_factory=Economics)


# ---------------------------------------------------------------- Thornhill-Craver
TC_CD, TC_K, G_FT = 0.865, 1.27, 32.16
TC_RCRIT = (2 / (TC_K + 1)) ** (TC_K / (TC_K - 1))       # 0.5512


def thornhill_craver(d_in, p1, p2, t_f, sg, cd=TC_CD, k=TC_K):
    """Gas rate (Mscf/d) through a port/orifice of diameter d_in (in): Thornhill-Craver (API Gas Lift), upstream
    p1 and downstream p2 in psia, upstream temperature degF. Critical below r = 0.551 (k = 1.27)."""
    rc = (2 / (k + 1)) ** (k / (k - 1))
    r = np.clip(np.asarray(p2, float) / np.asarray(p1, float), rc, 1.0)
    a = np.pi / 4 * np.asarray(d_in, float) ** 2
    term = 2 * G_FT * k / (k - 1) * np.maximum(r ** (2 / k) - r ** ((k + 1) / k), 0.0)
    return 155.5 * cd * a * np.asarray(p1, float) * np.sqrt(term) / np.sqrt(sg * (np.asarray(t_f, float) + 460.0))


def port_for_rate(q_mscfd, p1, p2, t_f, sg, derate_pct=0.0):
    """Port diameter (in) that passes q at p1 -> p2 (continuous); inf if p2 >= p1."""
    unit = thornhill_craver(1.0, p1, p2, t_f, sg) * (1 - derate_pct / 100.0)
    return np.where(unit > 0, np.sqrt(np.asarray(q_mscfd, float) / np.where(unit > 0, unit, 1.0)), np.inf)


def upstream_for_rate(q_mscfd, d_in, p2, t_f, sg, derate_pct=0.0):
    """Upstream pressure (psia) that passes q through the port with downstream p2 (bisection)."""
    lo, hi = float(p2), float(p2) * 1.0 + 1.0
    f = lambda p1: thornhill_craver(d_in, p1, p2, t_f, sg) * (1 - derate_pct / 100.0) - q_mscfd
    while f(hi) < 0 and hi < 1e5:
        hi = hi * 1.5 + 50
    for _ in range(80):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if f(m) < 0 else (lo, m)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- nitrogen dome temperature correction
def ct_api(t_f):
    """API RP 11V6 temperature correction factor for nitrogen (dome at 60 F / dome at T), Eq. 5.1.15.5."""
    return 1.0 / (1.0 + 0.00215 * (np.asarray(t_f, float) - 60.0))


def dome_at_60(pd_t_psig, t_f, high_pressure=True):
    """Dome pressure at 60 F (psig) from the dome pressure at valve temperature (psig)."""
    if high_pressure:
        return dome_60_winkler(pd_t_psig, t_f)
    return np.asarray(pd_t_psig, float) * ct_api(t_f)


WE_ATM = 14.4        # atmospheric pressure used by Winkler & Eads (1989)


def _we_slope(pb_psia):
    pb = np.asarray(pb_psia, float)
    return np.where(pb < 1238.0, -0.00226 + 0.001934 * pb + 3.054e-7 * pb * pb,
                    -0.267 + 0.002298 * pb + 1.84e-7 * pb * pb)


def dome_at_t_winkler(pb60_psig, t_f):
    """Winkler & Eads (1989) Eq. 1-3: N2 dome pressure at valve temperature (psig) from the 60 F charge (psig)."""
    pb = np.asarray(pb60_psig, float) + WE_ATM
    return pb + _we_slope(pb) * (np.asarray(t_f, float) - 60.0) - WE_ATM


def dome_60_winkler(pd_t_psig, t_f, tol=1e-6):
    """Winkler & Eads (1989): 60 F charge (psig) from the dome pressure at valve temperature (psig) - the PB60
    fixed-point iteration (ideal-gas start)."""
    pbt = np.asarray(pd_t_psig, float) + WE_ATM
    dt = np.asarray(t_f, float) - 60.0
    pb = 520.0 * pbt / (np.asarray(t_f, float) + 460.0)
    for _ in range(100):
        new = pbt - _we_slope(pb) * dt
        if np.all(np.abs(new - pb) < tol):
            pb = new
            break
        pb = new
    return pb - WE_ATM


def dome_at_60(pd_t_psig, t_f, high_pressure=True):
    """Dome pressure at 60 F (psig) from the dome pressure at valve temperature (psig). high_pressure (PROSPER
    'Dome pressure correction > 1200 psig' = Yes): API Ct up to 1200 psig, Winkler & Eads above."""
    api = np.asarray(pd_t_psig, float) * ct_api(t_f)
    if not high_pressure:
        return api
    return np.where(np.asarray(pd_t_psig, float) > 1200.0, dome_60_winkler(pd_t_psig, t_f), api)


# ---------------------------------------------------------------- stability (Asheim 1988)
def asheim(q_liq_stb, qgi_mmscfd, j_stb_psi, port_in, n_ports, pt_inj, t_inj, gas: LiftGas, rho_fi, q_fi_ft3s,
           vt_ft3, vc_ft3, d_tvd, e=0.9):
    """Asheim (1988) criteria. F1 = rho_gsc Bg qgsc^2 J / (qLsc (E Ai)^2) (field units / gc / 144); F2 (Eq. B-10, C = 1)
    = (Vt/Vc) 144 pti / (D (rho_fi - rho_gi)) (qfi + qgi) / (qfi (1 - F1)). Stable if F1 > 1, else if F2 > 1.
    rho_fi (lb/ft3) and q_fi (ft3/s): in-situ reservoir fluid (oil, water and formation gas) at the injection point."""
    bg = float(gas.bg(pt_inj, t_inj))
    qgsc = qgi_mmscfd * 1e6 / 86400.0
    ai = n_ports * np.pi / 4 * (port_in / 12.0) ** 2
    f1 = gas.rho_sc * bg * qgsc ** 2 * j_stb_psi / (max(q_liq_stb, 1e-9) * (e * ai) ** 2 * 32.174 * 144.0)
    rho_gi = gas.rho_sc / bg
    qgi = bg * qgsc
    f2 = np.nan
    if f1 < 1:
        f2 = (vt_ft3 / vc_ft3) * 144.0 * pt_inj / (d_tvd * max(rho_fi - rho_gi, 1e-6)) * (q_fi_ft3s + qgi) / \
             (max(q_fi_ft3s, 1e-12) * (1 - f1))
    stable = f1 > 1 or (np.isfinite(f2) and f2 > 1)
    return {"F1": float(f1), "F2": float(f2), "stable": bool(stable), "Bg": bg, "rho_gi": float(rho_gi),
            "qgi_insitu": float(qgi)}


# ================================================================ valve spacing (pure geometry, any pressure unit)
def _first_crossing(x, f, x_from=-np.inf, x_to=np.inf):
    """First x in (x_from, x_to] where f changes from < 0 to >= 0 (linear interpolation); None if none."""
    k = np.where((x > x_from) & (x <= x_to))[0]
    if len(k) < 2:
        return None
    xs, fs = x[k], f[k]
    if fs[0] >= 0:
        return float(xs[0])
    j = np.where(fs >= 0)[0]
    if not len(j):
        return None
    j = j[0]
    return float(xs[j - 1] + (xs[j] - xs[j - 1]) * (-fs[j - 1]) / (fs[j] - fs[j - 1]))


def space_valves(md, tvd, p_top_unload, transfer, pc_first, pc_next, gs, psf_first, psf, min_spacing, d_op):
    """Unloading valve depths (ft MD) between the load-fluid line and the casing pressure (API RP 11V6 5.1.11-12,
    PROSPER 'Normal' spacing). md/tvd: fine depth grid; transfer(md) tubing pressure at a valve after transfer (design
    gradient or spacing line); pc_first: casing pressure on the grid used for the first valve (kick-off); pc_next(n):
    casing pressure on the grid used to space valve n+1 (surface pressure reduced for the n valves above).
    Returns (depths, reason) - reason why the spacing stopped."""
    f = p_top_unload + gs * (tvd - tvd[0]) - (pc_first - psf_first)
    d1 = _first_crossing(md, f)
    if d1 is None or d1 >= d_op:
        return [], "kick-off pressure reaches the injection depth - no unloading valves needed"
    depths = [d1]
    for _ in range(60):
        dn = depths[-1]
        tv_n = np.interp(dn, md, tvd)
        f = transfer(dn) + gs * (tvd - tv_n) - (pc_next(len(depths)) - psf)
        d = _first_crossing(md, f, dn + 1e-6)
        if d is None or d >= d_op:
            return depths, "injection depth reached"
        if d - dn < min_spacing:
            return depths, "spacing below the minimum"
        depths.append(d)
    return depths, "too many valves"


# ================================================================ full design
@dataclass
class ValveRow:
    n: int
    kind: str                     # "Valve" / "Orifice"
    md: float
    tvd: float
    pt: float                     # flowing tubing pressure at the valve (design gradient), psia
    pc: float                     # casing pressure at the valve (opening, at depth), psia
    p_transfer: float             # tubing pressure at which injection transfers to the next valve, psia
    t_valve: float                # degF
    q_required: float             # unloading/operating gas the valve must pass, Mscf/d
    q_gas: float                  # gas through the chosen port (Thornhill-Craver, de-rated), Mscf/d
    pt_unload: float              # tubing pressure at the valve while unloading through it, psia
    port_64: float
    r: float = np.nan
    pvo: float = np.nan           # opening pressure at depth, psia
    pvc: float = np.nan           # closing pressure at depth, psia
    pd60: float = np.nan          # dome pressure at 60 F, psig
    ptro: float = np.nan          # test rack opening pressure at 60 F, psig
    chp_open: float = np.nan      # surface casing pressure to open, psia
    chp_close: float = np.nan
    unloadable: float = np.nan    # psi/ft
    dp_valve: float = np.nan      # casing - tubing at the valve, psi
    note: str = ""


@dataclass
class DesignResult:
    inp: DesignInput
    q: float                      # design liquid rate, STB/d
    qgi: float                    # design lift gas, MMscf/d
    pwf: float
    valves: list
    inj_md: float
    pso: float                    # operating surface casing pressure, psia
    md: np.ndarray                # profiles on the path nodes
    tvd: np.ndarray
    pt: np.ndarray                # flowing tubing pressure at design
    tt: np.ndarray                # flowing tubing temperature
    pc_op: np.ndarray             # casing pressure (operating)
    pc_ko: np.ndarray             # casing pressure (kick-off)
    transfer: np.ndarray          # transfer-pressure line used for spacing
    perf: pd.DataFrame | None = None
    stop_reason: str = ""
    messages: list = field(default_factory=list)
    iterations: int = 0
    trace: list = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = []
        for v in self.valves:
            rows.append({"Valve": v.n, "Type": v.kind, "Measured depth (ft)": v.md, "True vertical depth (ft)": v.tvd,
                         "Tubing pressure (psia)": v.pt, "Casing pressure (psia)": v.pc,
                         "Transfer pressure (psia)": v.p_transfer, "Temperature at valve (°F)": v.t_valve,
                         "Gas lift rate (Mscf/d)": v.q_gas, "Required gas (Mscf/d)": v.q_required,
                         "Port size (64ths in)": v.port_64, "R value": v.r,
                         "Valve opening pressure (psia)": v.pvo, "Valve closing pressure (psia)": v.pvc,
                         "Dome pressure at 60 °F (psig)": v.pd60, "Test rack opening pressure (psig)": v.ptro,
                         "Opening CHP (psia)": v.chp_open, "Closing CHP (psia)": v.chp_close,
                         "dP across valve (psi)": v.dp_valve, "Unloadable gradient (psi/ft)": v.unloadable,
                         "Note": v.note})
        return pd.DataFrame(rows)


def spacing_factor(pi):
    """PROSPER / spacing-line factor SF by productivity index (STB/d/psi)."""
    return 0.20 if pi < 2 else (0.25 if pi <= 20 else 0.30)


def valve_settings(pvo_d, pt, r, t_valve, chp_open, gas, tvd_top, tvd, t_top, t_valve_gas, high_pressure=True,
                   p_atm=14.696):
    """IPO valve set to open at pvo_d (casing at depth, psia) with tubing pt (psia): dome at temperature,
    closing pressure at depth, dome at 60 F and test-rack opening pressure (psig), closing CHP (psia)."""
    pbt = pvo_d * (1 - r) + pt * r                          # force balance: Pbt Ab = Pc (Ab - Ap) + Pt Ap
    pvc_d = pbt
    pd60 = float(dome_at_60(max(pbt - p_atm, 0.0), t_valve, high_pressure))
    ptro = pd60 / (1 - r)
    chp_close = float(surface_from_depth(gas, pvc_d, tvd_top, tvd, t_top, t_valve_gas))
    return pvc_d, pd60, ptro, chp_close


def _unloading_gas(path, fluid, ipr, inp: DesignInput, mds, q_design, correlation, p1, p2, u, hc, gas_sg, n=10):
    """For each valve depth: the gas rate (MMscf/d) that gives the minimum tubing pressure at the valve while
    unloading through it, and that tubing pressure. Use IPR for unloading: the liquid rate comes from the IPR at each
    gas rate; otherwise the design rate is produced, and the target pressure is raised by the minimum transfer dP %."""
    mds = np.atleast_1d(np.asarray(mds, float))
    nv = len(mds)
    if not nv:
        return np.array([]), np.array([])
    g = np.linspace(inp.qgi_unload_max / n, inp.qgi_unload_max, n)
    gg, fm = np.tile(g, nv), np.repeat(mds, n)
    case = {"q": np.full(gg.size, q_design), "wc": inp.wc, "gor": inp.gor}
    pt = None
    if inp.use_ipr_unloading:
        ops = solve_rate(path, fluid, ipr, inp.top_p_unload, inp.wc, inp.gor, gg, correlation, p1, p2, u, hc, None,
                         0.0, None, gas_sg, fixed_md=fm, refine=1)
        pt = np.array([op.inj_p if op.inj_p is not None else np.nan for op in ops])
    r = None
    if pt is None or not np.all(np.isfinite(pt)):
        r = lifted_traverse(path, fluid, inp.top_p_unload, case, gg, correlation, p1, p2, u, hc, fixed_md=fm,
                            gas_sg=gas_sg, passes=1)
        pt = r.inj_p if pt is None else np.where(np.isfinite(pt), pt, r.inj_p)   # no flow: the design rate
    pt = pt.reshape(nv, n)
    g_out, p_out = np.empty(nv), np.empty(nv)
    for v in range(nv):
        row = pt[v]
        k = int(np.nanargmin(row))
        pmin = float(row[k])
        g_out[v], p_out[v] = g[k], pmin
        if not inp.use_ipr_unloading and inp.min_transfer_pct > 0:
            target = pmin * (1 + inp.min_transfer_pct / 100.0)
            j = np.where(row[:k + 1] <= target)[0]
            j = j[0] if len(j) else k
            if j > 0 and row[j - 1] != row[j]:
                g_out[v] = g[j - 1] + (g[j] - g[j - 1]) * (row[j - 1] - target) / (row[j - 1] - row[j])
            else:
                g_out[v] = g[j]
            p_out[v] = target
        elif 0 < k < n - 1:
            c = np.polyfit(g[k - 1:k + 2], row[k - 1:k + 2], 2)
            if c[0] > 0:
                gk = float(np.clip(-c[1] / (2 * c[0]), g[k - 1], g[k + 1]))
                g_out[v], p_out[v] = gk, float(np.polyval(c, gk))
    return g_out, p_out


def design(path, fluid, ipr, gas: LiftGas, inp: DesignInput, correlation="hagedorn_brown", p1=1.0, p2=1.0, u=8.0,
           hc: HeatCapacities | None = None, step=10.0, progress=None) -> DesignResult:
    """Continuous gas-lift design for a new well (PROSPER Design | Gaslift | New Well, casing-sensitive IPO valves,
    'All valves Pvo = gas pressure'). Steps: design rate (entered, or from the performance curve), operating
    injection depth on the gas-lifted gradient, unloading valve spacing between the load-fluid and design gradients
    with a casing pressure drop per valve, iteration until the valve depths settle, IPR conformance, unloading gas
    and port sizes (Thornhill-Craver), valve settings (dome and test-rack pressures)."""
    hc = hc or HeatCapacities()
    valve_t = VALVES[inp.valve]
    max_md = min(inp.max_md, path.md[-1])
    msgs = []
    tc_gas = casing_temperature(path, gas)

    def say(m):
        if progress:
            progress(m)

    # 1. design rate and gas
    perf = None
    pc_inj = casing_pressure(path, gas, inp.p_inj)[0]
    if inp.design_rate == DESIGN_RATE[0]:
        q_d, qgi_d = float(inp.q_design), float(inp.qgi_max)
    else:
        say("Performance curve...")
        perf = performance_curve(path, fluid, ipr, inp.top_p, inp.wc, inp.gor, inp.qgi_max * 1.5, correlation, p1, p2,
                                 u, hc, pc=pc_inj, dp_valve=inp.dp_valve, max_md=max_md, gas_sg=gas.sg, n=13,
                                 economics=inp.economics)
        by = "Revenue ($/d)" if inp.design_rate == DESIGN_RATE[2] else "Oil rate (STB/d)"
        opt = optimum(perf, by, inp.qgi_max, inp.q_liquid_max)
        if opt is None or opt["q"] <= 0:
            raise ValueError("The well does not produce at any lift-gas rate up to the maximum gas available.")
        q_d, qgi_d = opt["q"], opt["qgi"]
        msgs.append(f"Design point: {q_d:,.1f} STB/d liquid with {qgi_d:.4g} MMscf/d lift gas ({opt['note']}).")
    q_d = min(q_d, inp.q_liquid_max)

    # fine depth grid
    md_f = np.unique(np.concatenate([np.arange(path.md[0], path.md[-1], step), path.md]))
    tvd_f = np.interp(md_f, path.md, path.tvd)

    def on_grid(prof):
        return np.interp(md_f, path.md, prof)

    pc_ko = casing_pressure(path, gas, inp.p_ko)[0]
    r_guess = float(valve_t.r(12 if 12 <= valve_t.max_port_64 else valve_t.max_port_64))
    pds, depths = [], []
    stop = ""
    inj_md = max_md
    trace = []
    r_list = []

    def r_of(k):
        return r_list[k] if k < len(r_list) else (r_list[-1] if r_list else r_guess)

    q_first = q_d
    it = 0
    for sp in range(3):

        def spacing_pass(depths, pds, q_d):
            """One spacing pass with the casing pressures from the current valve count. Returns
            (depths, drops, d_op, d_next, stop, transfer_f)."""
            nv = len(depths)
            pso = inp.p_inj - sum(pds[:nv]) - (inp.safety_last if nv else 0.0)
            pc_op = casing_pressure(path, gas, pso)[0]
            lt = lifted_traverse(path, fluid, inp.top_p, {"q": [q_d], "wc": inp.wc, "gor": inp.gor}, [qgi_d], correlation,
                                 p1, p2, u, hc, pc=pc_op, dp_valve=inp.dp_valve, max_md=max_md, gas_sg=gas.sg)
            if lt.no_injection[0]:
                raise ValueError("The operating casing pressure is below the flowing tubing pressure plus the valve dP "
                                 "at the wellhead - raise the injection pressure or lower the top-node pressure.")
            d_op = float(lt.inj_md[0])
            pt_f = on_grid(lt.P[0])
            if inp.spacing == SPACING[1]:
                pi = q_d / max(float(ipr.pr) - float(ipr_pwf(ipr, q_d)), 1.0)
                sf = spacing_factor(pi)
                p0 = inp.top_p + sf * (inp.p_inj - inp.top_p)
                tv_op = float(np.interp(d_op, md_f, tvd_f))
                line = p0 + (float(np.interp(d_op, md_f, pt_f)) - p0) * (tvd_f - tvd_f[0]) / max(tv_op - tvd_f[0], 1.0)
                transfer_f = np.maximum(line, pt_f)
            else:
                transfer_f = pt_f
            pc_cache = {}

            def pc_surface(n):                   # surface opening pressure of valve n+1 (n valves above it)
                return inp.p_inj - sum(pds[k] if k < len(pds) else (pds[-1] if pds else inp.dp_close) for k in range(n))

            def pc_next(n):
                if n not in pc_cache:
                    pc_cache[n] = on_grid(casing_pressure(path, gas, pc_surface(n))[0])
                return pc_cache[n]

            new, stop = space_valves(md_f, tvd_f, inp.top_p_unload, lambda m: float(np.interp(m, md_f, transfer_f)),
                                     on_grid(pc_ko), pc_next, inp.load_grad, inp.margin_top, inp.margin, inp.min_spacing,
                                     d_op)
            # where the next valve would go: the injection point cannot be deeper than the well can be unloaded to
            d_next = None
            if new:
                dn = new[-1]
                f = float(np.interp(dn, md_f, transfer_f)) + inp.load_grad * (tvd_f - float(np.interp(dn, md_f, tvd_f))) \
                    - (pc_next(len(new)) - inp.margin)
                d_next = _first_crossing(md_f, f, dn + 1e-6)
            # last unloading valve too close to the injection point: move it up (API RP 11V6 Example 1)
            if new and d_op - new[-1] < inp.min_spacing:
                moved = d_op - inp.min_spacing
                if len(new) == 1 or moved - new[-2] >= inp.min_spacing:
                    new[-1] = moved
                else:
                    new = new[:-1]
            pds_new = []
            for k, dk in enumerate(new):
                pd_k = inp.dp_close
                if inp.dp_close_spread:                      # PROSPER "Pvo = gas pressure": at least the valve spread
                    piod = float(np.interp(dk, md_f, pc_next(k) if k else on_grid(casing_pressure(path, gas, inp.p_inj)[0])))
                    pd_k = max(pd_k, r_of(k) * (piod - float(np.interp(dk, md_f, transfer_f))))
                pds_new.append(pd_k)
            return new, pds_new, d_op, d_next, stop, transfer_f

        for outer in range(6):
            hist = []
            for inner in range(12):
                it += 1
                say(f"Iteration {it}: spacing...")
                new, pds_new, d_op, d_next, stop, transfer_f = spacing_pass(depths, pds, q_d)
                trace.append((it, len(depths), len(new), round(d_op, 1), round(q_d, 1), [round(x) for x in new]))
                converged = len(new) == len(depths) and np.allclose(new, depths, atol=2.0) and \
                    np.allclose(pds_new, pds[:len(new)], atol=0.5) and abs(d_op - inj_md) < 2.0
                hist.append((len(depths), new, pds_new, d_op, d_next))
                if len(hist) >= 3 and hist[-1][0] == hist[-3][0] and hist[-1][0] != hist[-2][0]:
                    # the valve count alternates: with n valves the operating point is deep enough for one more valve,
                    # with n + 1 it is not. Keep n valves and inject where the next valve would have gone.
                    lo = min((h for h in hist[-2:]), key=lambda h: h[0])
                    n_lo = lo[0]
                    cand = [h for h in hist if h[0] == n_lo][-1]
                    depths, pds = cand[1][:n_lo], cand[2][:n_lo]
                    if len(cand[1]) > n_lo:
                        inj_md = min(cand[3], cand[1][n_lo])
                    else:
                        inj_md = min(cand[3], cand[4]) if cand[4] is not None else cand[3]
                    stop = "injection point limited by the deepest unloading reach"
                    break
                depths, pds, inj_md = new, pds_new, d_op
                if converged:
                    break
            if not inp.check_ipr:
                break
            say("Checking rate conformance with the IPR...")
            op = solve_rate(path, fluid, ipr, inp.top_p, inp.wc, inp.gor, [qgi_d], correlation, p1, p2, u, hc, None,
                            0.0, None, gas.sg, fixed_md=np.array([inj_md]), refine=2)[0]
            if op.q is None:
                raise ValueError("The design does not flow: no VLP/IPR intersection at the design injection point.")
            q_new = min(op.q, q_first)          # never above the design rate asked for
            if abs(q_new - q_d) <= 0.002 * q_d:
                break
            q_d = q_new
        # 2. final profiles
        nv = len(depths)
        pio = [inp.p_inj - sum(pds[:k]) for k in range(nv)]
        pso = inp.p_inj - sum(pds) - (inp.safety_last if nv else 0.0)
        pc_op = casing_pressure(path, gas, pso)[0]
        lt = lifted_traverse(path, fluid, inp.top_p, {"q": [q_d], "wc": inp.wc, "gor": inp.gor}, [qgi_d], correlation,
                             p1, p2, u, hc, pc=pc_op, dp_valve=inp.dp_valve, max_md=max_md, fixed_md=[inj_md],
                             gas_sg=gas.sg)
        pt_n, tt_n = lt.P[0], lt.T[0]
        pt_f = on_grid(pt_n)
        if inp.spacing == SPACING[1]:
            transfer_f = np.maximum(transfer_f, pt_f)
        else:
            transfer_f = pt_f
        pwf = float(lt.P[0, -1])
        t_top = float(tc_gas[0])
        rows = []
        say("Unloading gas rates...")
        g_unl, pt_unl_all = _unloading_gas(path, fluid, ipr, inp, depths, q_d, correlation, p1, p2, u, hc, gas.sg)
        for k, dk in enumerate(depths):
            tv = float(np.interp(dk, path.md, path.tvd))
            pc_d = float(np.interp(dk, path.md, casing_pressure(path, gas, pio[k])[0]))
            pt_d = float(np.interp(dk, md_f, pt_f))
            ptr = float(np.interp(dk, md_f, transfer_f))
            t_v = float(np.interp(dk, path.md, tt_n))
            tg_v = float(np.interp(dk, path.md, tc_gas))
            q_req, pt_unl = float(g_unl[k]) * 1000.0, float(pt_unl_all[k])
            p_down = max(min(pt_unl, pc_d - 1.0), 14.7)
            port, q_gas, note = None, np.nan, ""
            for pp in valve_t.ports:
                qq = float(thornhill_craver(pp / 64, pc_d, p_down, tg_v, gas.sg)) * (1 - inp.tc_derate_valve / 100)
                if qq >= q_req:
                    port, q_gas = pp, qq
                    break
            if port is None:
                port = valve_t.ports[-1]
                q_gas = float(thornhill_craver(port / 64, pc_d, p_down, tg_v, gas.sg)) * (1 - inp.tc_derate_valve / 100)
                note = "largest port cannot pass the unloading gas"
            r = float(valve_t.r(port))
            pvc, pd60, ptro, chp_c = valve_settings(pc_d, ptr, r, t_v, pio[k], gas, path.tvd[0], tv, t_top, tg_v,
                                                    inp.dome_high_p)
            rows.append(ValveRow(k + 1, "Valve", dk, tv, pt_d, pc_d, ptr, t_v, q_req, q_gas, pt_unl, port, r, pc_d, pvc,
                                 pd60, ptro, pio[k], chp_c, np.nan, pc_d - pt_d, note))

        r_new = [v.r for v in rows]
        if not inp.dp_close_spread or (len(r_new) == len(r_list) and np.allclose(r_new, r_list, atol=1e-4)):
            break
        r_list = r_new
        say("Re-spacing with the sized ports (casing drop = valve spread)...")
    if q_d < q_first * 0.998:
        msgs.append(f"Design rate reduced from {q_first:,.1f} to {q_d:,.1f} STB/d liquid to conform with the IPR "
                    "(rate check against the well inflow at the final injection point).")
    # operating point
    tv = float(np.interp(inj_md, path.md, path.tvd))
    pc_d = float(np.interp(inj_md, path.md, pc_op))
    pt_d = float(np.interp(inj_md, path.md, pt_n))
    t_v = float(np.interp(inj_md, path.md, tt_n))
    tg_v = float(np.interp(inj_md, path.md, tc_gas))
    q_req = qgi_d * 1000.0
    if inp.injection_point == INJECTION_POINT[0]:
        d_cont = float(port_for_rate(q_req, pc_d, pt_d, tg_v, gas.sg, inp.tc_derate_orifice)) * 64
        if inp.orifice_sizing == ORIFICE_SIZING[0]:
            port = d_cont
            dp_act = pc_d - pt_d
        else:
            larger = [p for p in ORIFICE_PORTS if p >= d_cont - 1e-9]
            port = larger[0] if larger else ORIFICE_PORTS[-1]
            p_up = upstream_for_rate(q_req, port / 64, pt_d, tg_v, gas.sg, inp.tc_derate_orifice)
            dp_act = p_up - pt_d
        q_gas = float(thornhill_craver(port / 64, pt_d + dp_act, pt_d, tg_v, gas.sg)) * (1 - inp.tc_derate_orifice / 100)
        rows.append(ValveRow(nv + 1, "Orifice", inj_md, tv, pt_d, pt_d + dp_act, pt_d, t_v, q_req, q_gas, pt_d,
                             port, np.nan, chp_open=pso, dp_valve=dp_act))
    else:
        port, q_gas, note = None, np.nan, ""
        for pp in valve_t.ports:
            qq = float(thornhill_craver(pp / 64, pc_d, pt_d, tg_v, gas.sg)) * (1 - inp.tc_derate_valve / 100)
            if qq >= q_req:
                port, q_gas = pp, qq
                break
        if port is None:
            port = valve_t.ports[-1]
            q_gas = float(thornhill_craver(port / 64, pc_d, pt_d, tg_v, gas.sg)) * (1 - inp.tc_derate_valve / 100)
            note = "largest port cannot pass the design gas"
        r = float(valve_t.r(port))
        pvc, pd60, ptro, chp_c = valve_settings(pc_d, pt_d, r, t_v, pso, gas, path.tvd[0], tv, t_top, tg_v,
                                                inp.dome_high_p)
        rows.append(ValveRow(nv + 1, "Valve", inj_md, tv, pt_d, pc_d, pt_d, t_v, q_req, q_gas, pt_d, port, r, pc_d,
                             pvc, pd60, ptro, pso, chp_c, np.nan, pc_d - pt_d, note))
    # unloadable gradients: from the transfer pressure of the valve above to the casing pressure of this one
    prev_p, prev_tv = inp.top_p_unload, float(path.tvd[0])
    for v in rows:
        v.unloadable = (v.pc - prev_p) / max(v.tvd - prev_tv, 1e-6)
        prev_p, prev_tv = v.p_transfer, v.tvd
    res = DesignResult(inp, q_d, qgi_d, pwf, rows, inj_md, pso, path.md, path.tvd, pt_n, tt_n, pc_op, pc_ko,
                       np.interp(path.md, md_f, transfer_f), perf, stop, msgs, it + 1, trace)
    if any(v.note for v in rows):
        msgs.append("Some ports cannot pass the required gas - see the Note column.")
    return res


# ================================================================ stability of the design (Asheim 1988)
def annulus_volume(secs: pd.DataFrame, top_md, bottom_md):
    """Tubing-casing annulus volume (ft3) between two depths from the downhole sections (needs tubing OD and
    casing ID on the tubing rows); NaN if the geometry is missing."""
    vol = 0.0
    for _, r in secs.iterrows():
        a, b = max(float(r["MD top (ft)"]), top_md), min(float(r["MD bottom (ft)"]), bottom_md)
        if b <= a:
            continue
        if r["Type"] != "Tubing":
            return np.nan
        do, dc = r.get("Tubing OD (in)"), r.get("Casing ID (in)")
        if not (pd.notna(do) and pd.notna(dc) and dc > do):
            return np.nan
        vol += np.pi / 4 * ((dc / 12) ** 2 - (do / 12) ** 2) * (b - a)
    return vol


def tubing_volume(path: V.FlowPath, bottom_md):
    """Flow-path volume (ft3) from the top node down to bottom_md."""
    vol = 0.0
    for i in range(len(path.seg_len)):
        a, b = path.md[i], min(path.md[i + 1], bottom_md)
        if b > a:
            vol += path.area[i] * (b - a)
    return vol


def design_stability(res: DesignResult, path, fluid, gas: LiftGas, ipr, vc_ft3, e=0.9) -> dict:
    """Asheim F1/F2 at the design operating point (PROSPER Results | Stability)."""
    op = res.valves[-1]
    pwf = float(ipr_pwf(ipr, res.q))
    j = res.q / max(float(ipr.pr) - pwf, 1e-6)
    st = fluid.state(np.array([op.pt]), np.array([op.t_valve]), {"q": np.array([res.q]), "wc": np.array([res.inp.wc]),
                                                                  "gor": np.array([res.inp.gor])})
    ql, qg = float(st["ql"][0]), float(st["qg"][0])
    rho_fi = (ql * float(st["rho_l"][0]) + qg * float(st["rho_g"][0])) / max(ql + qg, 1e-12)
    vt = tubing_volume(path, op.md)
    out = asheim(res.q, res.qgi, j, op.port_64 / 64.0, 1, op.pt, op.t_valve, gas, rho_fi, ql + qg, vt, vc_ft3, op.tvd,
                 e)
    out.update({"J": j, "Vt": vt, "Vc": vc_ft3, "rho_fi": rho_fi, "q_fi": ql + qg, "port_64": op.port_64,
                "pt": op.pt, "pc": op.pc, "depth": op.tvd})
    return out
