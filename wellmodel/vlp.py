"""Vertical lift performance (VLP) on the equipment description, with test-data matching.

Pieces
------
FlowPath            nodes from the Xmas tree down to the deepest equipment: geometry per segment
                    (tubing, annulus, tubing + annulus, casing), inclination from the deviation survey,
                    formation temperature from the geothermal gradient, point restrictions (SSSV etc.).
Fluids              BlackOilFluid (matched black oil from the PVT tab, oil wells), WetGasFluid (gas and
                    condensate with gas correlations, condensate carried as wellstream gas),
                    EOSTableFluid (PRP equation of state, flashed on a pressure-temperature grid).
temperature_profile steady heat loss to the formation with an overall heat transfer coefficient U:
                    w Cp dT/dz = -U pi D (T - T_form), solved exactly over each segment from the reservoir up.
traverse            top-down pressure traverse (Heun), gradient = P1 * gravity + P2 * friction.

Matching
--------
estimate_u          U per test so the calculated wellhead temperature equals the measured one.
compare             every correlation against every test (gauge pressure error).
match               regression of Parameter 1 (gravity) and Parameter 2 (friction) over the tests.
vlp_curve/intersect lift curve at test conditions and its operating point with an IPR.

Units: psia, degF, ft, in; oil rates STB/d (liquid), gas MMscf/d; U BTU/h/ft2/F; Cp BTU/lb/F.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq, least_squares

from . import blackoil as bo
from .correlations import CORRELATIONS, LABELS, NEEDS_GLR, no_slip
from .equipment import HeatCapacities

RB = 5.615            # ft3/bbl
RHO_AIR_SC = 0.0764   # lb/scf
VLP_CORRELATIONS = ["duns_ros", "duns_ros_gould", "ansari", "hagedorn_brown", "beggs_brill", "beggs_brill_noslip", "gray",
                    "mukherjee_brill", "orkiszewski", "aziz", "fancher_brown", "no_slip"]
LOWER_BOUNDS = ("fancher_brown", "no_slip")          # quality-check bounds, not suggested for matching
NOT_AVAILABLE = {
    "Petroleum Experts 1-5": "proprietary (Petroleum Experts) - not planned",
}
P1_RANGE, P2_RANGE = (0.9, 1.1), (0.7, 1.3)     # beyond these the match is flagged


# ================================================================ flow path
@dataclass
class FlowPath:
    md: np.ndarray                 # node measured depths, top -> bottom
    tvd: np.ndarray
    t_form: np.ndarray             # formation temperature at nodes, degF
    seg_len: np.ndarray            # per segment (len = nodes - 1), ft
    sin_t: np.ndarray
    area: np.ndarray               # ft2
    d_h: np.ndarray                # hydraulic diameter, ft
    rough: np.ndarray              # ft
    d_ht: np.ndarray               # heat-transfer diameter, ft
    restrictions: dict = field(default_factory=dict)   # node index -> restriction ID (in)

    def node(self, md):
        i = int(np.argmin(np.abs(self.md - md)))
        if abs(self.md[i] - md) > 0.51:
            raise ValueError(f"{md:g} ft MD is not a node of the flow path.")
        return i

    @property
    def top(self):
        return float(self.md[0])

    @property
    def bottom(self):
        return float(self.md[-1])


def build_flow_path(top_md, secs: pd.DataFrame, points: pd.DataFrame, survey_md, survey_tvd, geo_md, geo_t,
                    flow_type="tubing", extra_md=(), max_step=250.0) -> FlowPath:
    if secs is None or not len(secs):
        raise ValueError("No tubing or casing sections in the downhole equipment.")
    bottom = float(secs["MD bottom (ft)"].max())
    marks = {float(top_md), bottom, *map(float, secs["MD bottom (ft)"])}
    marks |= {float(m) for m in survey_md if top_md < m < bottom}
    if points is not None and len(points):
        marks |= {float(m) for m in points["MD (ft)"] if top_md <= m <= bottom}
    for m in extra_md:
        if m is not None and np.isfinite(m):
            if not top_md <= m <= bottom + 1e-6:
                raise ValueError(f"Depth {m:g} ft MD is outside the flow path ({top_md:g}-{bottom:g} ft).")
            marks.add(float(m))
    marks = np.array(sorted(marks))
    md = [marks[0]]
    for a, b in zip(marks[:-1], marks[1:]):
        n = max(1, int(np.ceil((b - a) / max_step)))
        md += list(np.linspace(a, b, n + 1)[1:])
    md = np.array(md)
    md = md[np.concatenate([[True], np.diff(md) > 0.5])]
    tvd = np.interp(md, survey_md, survey_tvd)
    seg = np.diff(md)
    sin_t = np.clip(np.diff(tvd) / seg, -1, 1)
    mid = 0.5 * (md[1:] + md[:-1])
    area, dh, rough, dht = (np.empty_like(mid) for _ in range(4))
    for j, m in enumerate(mid):
        r = secs[(secs["MD top (ft)"] < m) & (secs["MD bottom (ft)"] >= m)].iloc[0]
        e = (r["Roughness (in)"] if pd.notna(r["Roughness (in)"]) else 0.0006) / 12
        if r["Type"] == "Tubing" and flow_type != "tubing":
            di, do, dc = r["Flow ID (in)"] / 12, r["Tubing OD (in)"] / 12, r["Casing ID (in)"] / 12
            if flow_type == "annular":
                a_ = np.pi / 4 * (dc ** 2 - do ** 2)
                dh_ = dc - do
            else:
                a_ = np.pi / 4 * (di ** 2 + dc ** 2 - do ** 2)
                dh_ = 4 * a_ / (np.pi * (di + do + dc))
            area[j], dh[j], rough[j], dht[j] = a_, dh_, e, dc
        else:
            d = r["Flow ID (in)"] / 12
            area[j], dh[j], rough[j], dht[j] = np.pi / 4 * d * d, d, e, d
    restr = {}
    if points is not None and len(points):
        for _, p in points.iterrows():
            if p["Type"] in ("SSSV", "Restriction") and pd.notna(p["ID (in)"]):
                restr[int(np.argmin(np.abs(md - p["MD (ft)"])))] = float(p["ID (in)"])
    t_form = np.interp(md, geo_md, geo_t)
    return FlowPath(md, tvd, t_form, seg, sin_t, area, dh, rough, dht, restr)


# ================================================================== fluids
def _sigma_oil(api, p, t):
    """Baker-Swerdloff dead-oil tension with live-oil pressure correction (dyne/cm)."""
    s68, s100 = 39.0 - 0.2571 * api, 37.5 - 0.2571 * api
    sod = s68 + (s100 - s68) * np.clip((np.asarray(t, float) - 68.0) / 32.0, 0.0, 1.0)
    return np.maximum(sod * (1 - 0.024 * np.maximum(p, 0) ** 0.45), 1.0)


def _sigma_water(p, t):
    return np.clip(70.0 - (np.asarray(t, float) - 74.0) * (17.0 / 206.0), 35.0, 75.0) + 0 * np.asarray(p, float)


def _water(inp, p, t):
    return bo.water_props(inp, np.maximum(p, 14.7), t)


def _water_sg(inp):
    _, rho, _ = bo.water_props(inp, 14.696, 60.0)
    return float(rho / 62.428)


class BlackOilFluid:
    """Oil well, matched black oil. Case keys: q (liquid STB/d), wc (fraction), gor (scf/STB, total)."""
    kind = "oil"

    def __init__(self, model: bo.BlackOilModel):
        self.m = model
        self.inp = model.inp

    def _props(self, p, t, gor):
        m, inp, fam, vf = self.m, self.inp, self.m.family, self.m.visc_family
        rs_sat = m._m("rs").apply(bo._rs_corr(fam, inp, p, t))
        rs = np.clip(rs_sat, 0.0, gor)
        pb_g = np.maximum(m._m("pb").apply(bo.pb_raw(fam, inp, np.maximum(gor, 1.0), t) * bo.impurity_factor(inp, t)),
                          14.7)
        above = rs_sat >= gor
        bo_s = bo.bo_sat(fam, inp, rs, t)
        bo_ = np.where(above, bo.bo_sat(fam, inp, gor, t) * bo.bo_undersat_factor(fam, inp, p, pb_g, np.maximum(gor, 1.0), t),
                       bo_s)
        bo_ = np.maximum(m._m("bo").apply(bo_), 1.0)
        mod = bo.visc_dead(vf, inp, t)
        mu = np.where(above, bo.visc_undersat(vf, bo.visc_live(vf, mod, gor), p, pb_g), bo.visc_live(vf, mod, rs))
        mu = np.maximum(m._m("visc").apply(mu), 1e-3)
        return rs, bo_, mu

    def state(self, p, t, case):
        inp = self.inp
        q, wc, gor = case["q"], np.clip(case["wc"], 0, 0.9999), np.maximum(case["gor"], 0.0)
        qo, qw = q * (1 - wc), q * wc
        rs, bo_, mu_o = self._props(p, t, gor)
        rho_o = (62.428 * inp.gamma_o + 0.0136 * rs * inp.gas_sg) / bo_
        _, bg, rho_g, mu_g = bo.gas_props(inp, p, t)
        bw, rho_w, mu_w = _water(inp, p, t)
        q_o = qo * bo_ * RB / 86400
        q_w = qw * bw * RB / 86400
        q_g = qo * np.maximum(gor - rs, 0.0) * bg / 86400
        ql = q_o + q_w
        fo = np.where(ql > 1e-15, q_o / np.maximum(ql, 1e-15), 1 - wc)
        return dict(ql=ql, qg=q_g, rho_l=fo * rho_o + (1 - fo) * rho_w, rho_g=rho_g,
                    mu_l=fo * mu_o + (1 - fo) * mu_w, mu_g=mu_g,
                    sig=fo * _sigma_oil(inp.api, p, t) + (1 - fo) * _sigma_water(p, t))

    def mass_cp(self, case, hc: HeatCapacities):
        inp = self.inp
        q, wc, gor = case["q"], np.clip(case["wc"], 0, 0.9999), np.maximum(case["gor"], 0.0)
        mo = q * (1 - wc) * RB * 62.428 * inp.gamma_o
        mg = q * (1 - wc) * gor * RHO_AIR_SC * inp.gas_sg
        mw = q * wc * RB * 62.428 * _water_sg(inp)
        w = mo + mg + mw
        return w / 24.0, (mo * hc.oil + mg * hc.gas + mw * hc.water) / np.maximum(w, 1e-12)


def condensate_mw(api):
    """Cragoe (1929) stock-tank liquid molecular weight."""
    return 6084.0 / max(api - 5.9, 1.0)


class WetGasFluid:
    """Gas / condensate well with gas correlations. Condensate is carried as wellstream gas (wet gas);
    water is the liquid phase. Case keys: q (gas MMscf/d), wgr (STB/MMscf), cgr (STB/MMscf)."""
    kind = "gas"

    def __init__(self, inp: bo.PVTInputs, cond_api=50.0):
        self.inp, self.api = inp, cond_api
        self.go = 141.5 / (131.5 + cond_api)
        self.veq = 133000.0 * self.go / condensate_mw(cond_api)     # scf vapour per STB condensate

    def wellstream_sg(self, cgr):
        r = np.where(cgr > 1e-9, 1e6 / np.maximum(cgr, 1e-9), np.inf)       # separator GOR scf/STB
        sg = np.where(np.isfinite(r), (r * self.inp.gas_sg + 4584 * self.go) / (np.where(np.isfinite(r), r, 1)
                                                                               + self.veq), self.inp.gas_sg)
        return sg

    def state(self, p, t, case):
        q, wgr, cgr = case["q"], np.maximum(case["wgr"], 0.0), np.maximum(case["cgr"], 0.0)
        sg = self.wellstream_sg(cgr)
        inp = bo.PVTInputs(rsb=1, api=40, gas_sg=1.0, salinity_ppm=self.inp.salinity_ppm, h2s=self.inp.h2s,
                           co2=self.inp.co2, n2=self.inp.n2)
        shape = np.broadcast(p, t, sg).shape
        p_, t_, sg_ = (np.broadcast_to(a, shape).astype(float) for a in (p, t, sg))
        z, bg, rho_g, mu_g = np.empty(shape), np.empty(shape), np.empty(shape), np.empty(shape)
        for g in np.unique(np.round(sg_, 4)):          # few distinct gravities: loop
            k = np.isclose(np.round(sg_, 4), g)
            inp.gas_sg = float(g)
            z[k], bg[k], rho_g[k], mu_g[k] = bo.gas_props(inp, p_[k], t_[k])
        scf = q * 1e6 + q * cgr * self.veq
        bw, rho_w, mu_w = _water(self.inp, p, t)
        return dict(ql=q * wgr * bw * RB / 86400, qg=scf * bg / 86400, rho_l=rho_w + 0 * rho_g, rho_g=rho_g,
                    mu_l=mu_w + 0 * mu_g, mu_g=mu_g, sig=_sigma_water(p, t) + 0 * mu_g)

    def mass_cp(self, case, hc: HeatCapacities):
        q, wgr, cgr = case["q"], np.maximum(case["wgr"], 0.0), np.maximum(case["cgr"], 0.0)
        mg = q * 1e6 * RHO_AIR_SC * self.inp.gas_sg
        mc = q * cgr * RB * 62.428 * self.go
        mw = q * wgr * RB * 62.428 * _water_sg(self.inp)
        w = mg + mc + mw
        return w / 24.0, (mg * hc.gas + mc * hc.oil + mw * hc.water) / np.maximum(w, 1e-12)


def eos_grid(eos_fluid, p_max, t_min, t_max, n_p=26, n_t=6):
    """Flash the reservoir fluid on a (T, p) grid. Returns dict of 2-D arrays (n_t, n_p)."""
    ps = np.unique(np.concatenate([np.geomspace(14.7, p_max, n_p)]))
    ts = np.linspace(t_min, t_max, n_t)
    keys = ("beta", "vL", "vV", "rhoL", "rhoV", "muL", "muV", "ift")
    out = {k: np.full((len(ts), len(ps)), np.nan) for k in keys}
    for i, t in enumerate(ts):
        for j, p in enumerate(ps):
            r = eos_fluid.flash(p, t)
            if r.phases == "2P":
                L, V = r.liquid, r.vapour
                vals = (r.beta, L.molar_volume, V.molar_volume, L.density, V.density, L.viscosity, V.viscosity,
                        r.ift if np.isfinite(r.ift) else 5.0)
            elif r.phases == "V":
                V = r.vapour
                vals = (1.0, 1.0, V.molar_volume, 40.0, V.density, 0.3, V.viscosity, 5.0)
            else:
                L = r.liquid
                vals = (0.0, L.molar_volume, 1.0, L.density, 0.1, L.viscosity, 0.015, 25.0)
            for k, v in zip(keys, vals):
                out[k][i, j] = v
    out["p"], out["t"] = ps, ts
    return out


class EOSTableFluid:
    """PRP fluid: the wellstream has the reservoir composition; phase split and properties are interpolated
    from a flash grid. Case keys: oil wells q (liquid STB/d), wc; gas wells q (MMscf/d), wgr."""

    def __init__(self, kind, grid, mw, scf_per_lbmol, stb_per_lbmol, oil_density_st, water_inp: bo.PVTInputs):
        self.kind, self.mw = kind, mw
        self.scf, self.stb, self.rho_st = scf_per_lbmol, stb_per_lbmol, oil_density_st
        self.winp = water_inp
        pts = (grid["t"], np.log(grid["p"]))
        self.f = {k: RegularGridInterpolator(pts, grid[k], bounds_error=False, fill_value=None)
                  for k in ("beta", "vL", "vV", "rhoL", "rhoV", "muL", "muV", "ift")}
        self.p_max, self.t_rng = grid["p"].max(), (grid["t"].min(), grid["t"].max())

    def _moles(self, case):     # lbmol/s of hydrocarbon wellstream
        if self.kind == "oil":
            return case["q"] * (1 - np.clip(case["wc"], 0, 0.9999)) / max(self.stb, 1e-12) / 86400
        return case["q"] * 1e6 / max(self.scf, 1e-12) / 86400

    def _water_stb(self, case):
        return case["q"] * np.clip(case["wc"], 0, 0.9999) if self.kind == "oil" else case["q"] * np.maximum(case["wgr"], 0)

    def state(self, p, t, case):
        shape = np.broadcast(p, t, case["q"]).shape
        pp = np.log(np.clip(np.broadcast_to(p, shape), 14.7, self.p_max))
        tt = np.clip(np.broadcast_to(t, shape), *self.t_rng)
        x = np.stack([tt.ravel(), pp.ravel()], axis=1)
        v = {k: f(x).reshape(shape) for k, f in self.f.items()}
        beta = np.clip(v["beta"], 0, 1)
        n = self._moles(case)
        q_hl = n * (1 - beta) * v["vL"]
        q_hv = n * beta * v["vV"]
        bw, rho_w, mu_w = _water(self.winp, np.exp(pp), tt)
        q_w = self._water_stb(case) * bw * RB / 86400
        ql = q_hl + q_w
        fo = np.where(ql > 1e-15, q_hl / np.maximum(ql, 1e-15), 0.0)
        return dict(ql=ql, qg=q_hv, rho_l=fo * v["rhoL"] + (1 - fo) * rho_w, rho_g=v["rhoV"],
                    mu_l=fo * v["muL"] + (1 - fo) * mu_w, mu_g=v["muV"],
                    sig=fo * np.maximum(v["ift"], 0.5) + (1 - fo) * _sigma_water(np.exp(pp), tt))

    def mass_cp(self, case, hc: HeatCapacities):
        n = self._moles(case) * 86400                        # lbmol/d
        m_oil = n * self.stb * RB * self.rho_st if self.stb > 0 else 0 * n
        m_hc = n * self.mw
        m_gas = np.maximum(m_hc - m_oil, 0)
        mw = self._water_stb(case) * RB * 62.428 * _water_sg(self.winp)
        w = m_hc + mw
        return w / 24.0, (m_oil * hc.oil + m_gas * hc.gas + mw * hc.water) / np.maximum(w, 1e-12)


# ============================================================ temperature
def temperature_profile(path: FlowPath, w_lbh, cp, u):
    """Fluid temperature at every node (cases x nodes); reservoir fluid enters at formation temperature."""
    w = np.atleast_1d(np.asarray(w_lbh, float))
    cp = np.broadcast_to(np.asarray(cp, float), w.shape)
    n = len(path.md)
    T = np.empty((w.size, n))
    T[:, -1] = path.t_form[-1]
    for i in range(n - 2, -1, -1):
        L = path.seg_len[i]
        a = u * np.pi * path.d_ht[i] / np.maximum(w * cp, 1e-9)
        tfb, tft = path.t_form[i + 1], path.t_form[i]
        g = (tft - tfb) / L
        t0 = T[:, i + 1]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            tn = tfb + g * L - g / a + (t0 - tfb + g / a) * np.exp(-a * L)
        T[:, i] = np.where(a * L < 1e-9, t0, tn)
    return T


# ================================================================ traverse
def _gradient(path, fluid, i, p, t, case, corr, p1, p2):
    s = fluid.state(p, t, case)
    a = path.area[i]
    vsl, vsg = np.maximum(s["ql"] / a, 1e-12), np.maximum(s["qg"] / a, 0.0)
    args = (vsl, vsg, s["rho_l"], np.maximum(s["rho_g"], 1e-4), s["mu_l"], np.maximum(s["mu_g"], 1e-4),
            np.maximum(s["sig"], 0.5), path.d_h[i], path.rough[i], path.sin_t[i], p)
    ql, qg = np.maximum(s["ql"], 0.0), np.maximum(s["qg"], 0.0)
    lam = ql / np.maximum(ql + qg, 1e-30)
    single = (lam > 0.99999) | (lam < 1e-6)
    if corr.__name__ in NEEDS_GLR:
        e, f, _ = corr(*args, glr=_glr(case, fluid.kind))
    else:
        e, f, _ = corr(*args)
    if np.any(single):
        e0, f0, _ = no_slip(*args)
        e, f = np.where(single, e0, e), np.where(single, f0, f)
    g = p1 * e + p2 * f
    return np.nan_to_num(g, nan=0.433, posinf=1.0, neginf=0.0), s


def _glr(case, kind):
    """Producing gas-liquid ratio, scf per bbl of liquid."""
    if kind == "oil":
        return np.maximum(case.get("gor", 0.0) * (1 - np.clip(case.get("wc", 0.0), 0, 0.9999)), 0.0) \
            if "gor" in case else None
    liq = np.maximum(case.get("cgr", 0.0) + case.get("wgr", 0.0), 1e-9)
    return 1e6 / liq


def _restriction_dp(s, d_in, cd=0.9):
    """Homogeneous orifice pressure drop (psi) through a restriction of ID d_in."""
    ar = np.pi / 4 * (d_in / 12) ** 2
    q = s["ql"] + s["qg"]
    lam = s["ql"] / np.maximum(q, 1e-15)
    rho = lam * s["rho_l"] + (1 - lam) * s["rho_g"]
    v = q / ar
    return rho * v * v / (2 * 32.174 * cd * cd) / 144.0


def _broadcast_case(case):
    keys = list(case)
    arrs = np.broadcast_arrays(*[np.atleast_1d(np.asarray(case[k], float)) for k in keys])
    return {k: a.astype(float).ravel() for k, a in zip(keys, arrs)}


def traverse(path: FlowPath, fluid, whp, case, correlation="beggs_brill", p1=1.0, p2=1.0, u=8.0,
             hc: HeatCapacities | None = None):
    """Pressure (cases x nodes) and temperature (cases x nodes) from the wellhead down."""
    hc = hc or HeatCapacities()
    case = dict(case)
    case["_whp"], case["_p1"], case["_p2"] = whp, p1, p2
    case = _broadcast_case(case)
    whp, p1, p2 = case.pop("_whp"), case.pop("_p1"), case.pop("_p2")
    corr = CORRELATIONS[correlation]
    w, cp = fluid.mass_cp(case, hc)
    T = temperature_profile(path, w, cp, u)
    n = len(path.md)
    P = np.empty((whp.size, n))
    p = np.maximum(whp, 14.7)
    P[:, 0] = p
    for i in range(n - 1):
        L = path.seg_len[i]
        g0, _ = _gradient(path, fluid, i, p, T[:, i], case, corr, p1, p2)
        pp = np.maximum(p + g0 * L, 14.7)
        g1, s1 = _gradient(path, fluid, i, pp, T[:, i + 1], case, corr, p1, p2)
        p = np.maximum(p + 0.5 * (g0 + g1) * L, 14.7)
        if (i + 1) in path.restrictions:
            p = p + _restriction_dp(fluid.state(p, T[:, i + 1], case), path.restrictions[i + 1])
        P[:, i + 1] = p
    return P, T


# ================================================================ test data
OIL_TEST_COLUMNS = ["Test Point Date", "Test Point Comment", "Tubing Head Pressure (psia)",
                    "Tubing Head Temperature (deg F)", "Water Cut (percent)", "Liquid Rate (STB/day)",
                    "Gauge Depth (Measured) (feet)", "Gauge Pressure (psia)", "Reservoir Pressure (psia)",
                    "Gas Oil Ratio (scf/STB)", "GOR Free (scf/STB)", "Manifold Pressure (psia)",
                    "Choke Size (64ths in)"]
GAS_TEST_COLUMNS = ["Test Point Date", "Test Point Comment", "Tubing Head Pressure (psia)",
                    "Tubing Head Temperature (deg F)", "Water Gas Ratio (STB/MMscf)",
                    "Separator Gas Oil Ratio (scf/STB)", "Gas Rate (MMscf/day)", "Gauge Depth (Measured) (feet)",
                    "Gauge Pressure (psia)", "Reservoir Pressure (psia)", "Manifold Pressure (psia)",
                    "Choke Size (64ths in)"]
SURFACE_TEST_COLUMNS = ("Manifold Pressure (psia)", "Choke Size (64ths in)")   # optional: Match Surface


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)


def clean_tests(df: pd.DataFrame, kind: str, path_top=None, path_bottom=None):
    """Standardise the test table. Returns (tests DataFrame, issues). Columns: test, label, thp, tht, q, wc,
    gor, gor_free / wgr, cgr, gauge_md, gauge_p, pr, pman (manifold pressure), choke (64ths) - the last two
    optional, for matching the surface equipment."""
    issues, rows = [], []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        thp = _num(df, "Tubing Head Pressure (psia)").loc[r.name]
        q = _num(df, "Liquid Rate (STB/day)" if kind == "oil" else "Gas Rate (MMscf/day)").loc[r.name]
        if np.isnan(thp) and np.isnan(q):
            continue
        rec = {"test": i, "label": f"{i}" + (f" - {r['Test Point Comment']}" if isinstance(r.get("Test Point Comment"), str)
                                           and r["Test Point Comment"] else ""),
               "date": r.get("Test Point Date"), "thp": thp, "q": q,
               "tht": _num(df, "Tubing Head Temperature (deg F)").loc[r.name],
               "gauge_md": _num(df, "Gauge Depth (Measured) (feet)").loc[r.name],
               "gauge_p": _num(df, "Gauge Pressure (psia)").loc[r.name],
               "pr": _num(df, "Reservoir Pressure (psia)").loc[r.name],
               "pman": _num(df, "Manifold Pressure (psia)").loc[r.name],
               "choke": _num(df, "Choke Size (64ths in)").loc[r.name]}
        if kind == "oil":
            rec["wc"] = _num(df, "Water Cut (percent)").loc[r.name] / 100
            rec["gor"] = _num(df, "Gas Oil Ratio (scf/STB)").loc[r.name]
            rec["gor_free"] = _num(df, "GOR Free (scf/STB)").loc[r.name]
        else:
            rec["wgr"] = _num(df, "Water Gas Ratio (STB/MMscf)").loc[r.name]
            sg = _num(df, "Separator Gas Oil Ratio (scf/STB)").loc[r.name]
            rec["cgr"] = 1e6 / sg if np.isfinite(sg) and sg > 0 else 0.0
        name = f"Test {i}"
        if np.isnan(thp) or thp <= 0:
            issues.append(("error", f"{name}: enter the tubing head pressure."))
        if np.isnan(q) or q <= 0:
            issues.append(("error", f"{name}: enter a positive {'liquid' if kind == 'oil' else 'gas'} rate."))
        if kind == "oil":
            if np.isnan(rec["wc"]) or not 0 <= rec["wc"] < 1:
                issues.append(("error", f"{name}: water cut must be between 0 and 100 %."))
            if np.isnan(rec["gor"]) or rec["gor"] < 0:
                issues.append(("error", f"{name}: enter the producing GOR."))
            if np.isfinite(rec["gor_free"]) and np.isfinite(rec["gor"]) and rec["gor_free"] > rec["gor"]:
                issues.append(("warning", f"{name}: GOR free is larger than the total GOR."))
        else:
            if np.isnan(rec["wgr"]) or rec["wgr"] < 0:
                rec["wgr"] = 0.0 if np.isnan(rec["wgr"]) else rec["wgr"]
                if rec["wgr"] < 0:
                    issues.append(("error", f"{name}: water gas ratio cannot be negative."))
        if np.isfinite(rec["gauge_md"]) or np.isfinite(rec["gauge_p"]):
            if not (np.isfinite(rec["gauge_md"]) and np.isfinite(rec["gauge_p"])):
                issues.append(("error", f"{name}: give both gauge depth and gauge pressure (or neither)."))
            elif path_top is not None and not path_top <= rec["gauge_md"] <= path_bottom + 1e-6:
                issues.append(("error", f"{name}: gauge depth {rec['gauge_md']:g} ft is outside the flow path "
                               f"({path_top:g}-{path_bottom:g} ft MD)."))
            elif np.isfinite(thp) and rec["gauge_p"] <= thp:
                issues.append(("warning", f"{name}: gauge pressure is not above the tubing head pressure."))
        else:
            issues.append(("info", f"{name}: no gauge data - used for temperature and VLP/IPR only."))
        if np.isfinite(rec["pman"]):
            if rec["pman"] <= 0:
                issues.append(("error", f"{name}: manifold pressure must be positive."))
            elif np.isfinite(thp) and rec["pman"] >= thp:
                issues.append(("warning", f"{name}: manifold pressure is not below the tubing head pressure."))
        if np.isfinite(rec["choke"]) and rec["choke"] <= 0:
            issues.append(("error", f"{name}: choke size must be positive (64ths of an inch)."))
        if np.isfinite(rec["pr"]) and np.isfinite(rec["gauge_p"]) and rec["gauge_p"] >= rec["pr"]:
            issues.append(("warning", f"{name}: gauge pressure is not below the reservoir pressure."))
        rows.append(rec)
    return pd.DataFrame(rows), issues


def case_of(tests: pd.DataFrame, kind: str):
    if kind == "oil":
        return {"q": tests["q"].to_numpy(float), "wc": tests["wc"].to_numpy(float),
                "gor": tests["gor"].to_numpy(float)}
    return {"q": tests["q"].to_numpy(float), "wgr": tests["wgr"].to_numpy(float),
            "cgr": tests["cgr"].to_numpy(float)}


def traverse_up(path: FlowPath, fluid, p_bottom, case, correlation="beggs_brill", p1=1.0, p2=1.0, u=8.0,
                hc: HeatCapacities | None = None):
    """Pressure (cases x nodes) from the bottom node up to the wellhead for a given bottom-node pressure (the inflow
    side of a top-node solution). NaN from the node where the pressure is exhausted upwards."""
    hc = hc or HeatCapacities()
    case = dict(case)
    case["_pb"], case["_p1"], case["_p2"] = p_bottom, p1, p2
    case = _broadcast_case(case)
    pb, p1, p2 = case.pop("_pb"), case.pop("_p1"), case.pop("_p2")
    corr = CORRELATIONS[correlation]
    w, cp = fluid.mass_cp(case, hc)
    T = temperature_profile(path, w, cp, u)
    n = len(path.md)
    P = np.full((pb.size, n), np.nan)
    bad = ~np.isfinite(pb) | (pb <= 14.7)
    p = np.where(bad, 1000.0, pb)
    P[:, -1] = np.where(bad, np.nan, p)
    for i in range(n - 2, -1, -1):
        if (i + 1) in path.restrictions:            # restriction at node i+1: pressure above it is lower
            dp = _restriction_dp(fluid.state(p, T[:, i + 1], case), path.restrictions[i + 1])
            p = p - _restriction_dp(fluid.state(np.maximum(p - dp, 14.7), T[:, i + 1], case),
                                    path.restrictions[i + 1])
            bad |= p < 14.7
            p = np.where(bad, 1000.0, p)
        L = path.seg_len[i]
        g0, _ = _gradient(path, fluid, i, p, T[:, i + 1], case, corr, p1, p2)
        pp = np.maximum(p - g0 * L, 14.7)
        g1, _ = _gradient(path, fluid, i, pp, T[:, i], case, corr, p1, p2)
        pn = p - 0.5 * (g0 + g1) * L
        bad |= pn < 14.7
        p = np.where(bad, 1000.0, pn)
        P[:, i] = np.where(bad, np.nan, pn)
    return P, T


# ================================================================ matching
def estimate_u(path, fluid, tests, hc=None, u_max=500.0):
    """U for each test with a measured THT. Returns DataFrame (test, U, note)."""
    hc = hc or HeatCapacities()
    recs = []
    for _, t in tests.iterrows():
        if not np.isfinite(t["tht"]):
            recs.append({"Test": t["label"], "Measured THT (°F)": np.nan, "U (BTU/h/ft2/F)": np.nan,
                         "Note": "no THT"})
            continue
        case = {k: np.array([t[k]]) for k in case_of(tests.iloc[[0]], fluid.kind)}
        w, cp = fluid.mass_cp(case, hc)
        top_t = lambda u: temperature_profile(path, w, cp, u)[0, 0]
        t_hot, t_cold = top_t(1e-6), top_t(u_max)
        if not t_cold <= t["tht"] <= t_hot:
            note = ("measured THT is hotter than the reservoir allows" if t["tht"] > t_hot
                    else f"needs U above {u_max:g}")
            recs.append({"Test": t["label"], "Measured THT (°F)": t["tht"], "U (BTU/h/ft2/F)": np.nan, "Note": note})
            continue
        u = brentq(lambda x: top_t(x) - t["tht"], 1e-6, u_max, xtol=1e-6)
        recs.append({"Test": t["label"], "Measured THT (°F)": t["tht"], "U (BTU/h/ft2/F)": u, "Note": ""})
    return pd.DataFrame(recs)


def gauge_pressures(path, fluid, tests, correlation, p1=1.0, p2=1.0, u=8.0, hc=None):
    P, T = traverse(path, fluid, tests["thp"].to_numpy(float), case_of(tests, fluid.kind), correlation, p1, p2, u, hc)
    idx = [path.node(m) for m in tests["gauge_md"]]
    return P[np.arange(len(tests)), idx], P, T


def compare(path, fluid, tests, correlations=VLP_CORRELATIONS, u=8.0, hc=None):
    """Gauge-pressure error of every correlation (unmatched, P1 = P2 = 1) on tests with gauge data."""
    g = tests[np.isfinite(tests["gauge_p"])].reset_index(drop=True)
    detail, summary, traverses = [], [], {}
    for c in correlations:
        pc, P, T = gauge_pressures(path, fluid, g, c, 1.0, 1.0, u, hc)
        traverses[c] = (P, T)
        err = pc - g["gauge_p"].to_numpy(float)
        for (_, t), e, v in zip(g.iterrows(), err, pc):
            detail.append({"Correlation": LABELS[c], "Test": t["label"], "Measured (psia)": t["gauge_p"],
                           "Calculated (psia)": v, "Error (psi)": e, "Error (%)": 100 * e / t["gauge_p"]})
        summary.append({"key": c, "Correlation": LABELS[c], "Mean abs error (psi)": np.mean(np.abs(err)),
                        "Mean abs error (%)": np.mean(np.abs(err / g["gauge_p"].to_numpy(float))) * 100,
                        "Bias (psi)": np.mean(err), "Tests": len(err)})
    s = pd.DataFrame(summary).sort_values("Mean abs error (psi)").reset_index(drop=True)
    s.insert(0, "Rank", np.arange(1, len(s) + 1))
    return s, pd.DataFrame(detail), traverses, g


@dataclass
class VLPMatch:
    correlation: str
    p1: float
    p2: float
    std: float               # psi, standard deviation of matched residuals
    rmse_before: float
    rmse_after: float
    residuals: np.ndarray
    n: int
    flags: list

    @property
    def label(self):
        return LABELS[self.correlation]


def match(path, fluid, tests, correlation, u=8.0, hc=None, sigma_p=10.0):
    """Regress P1 (gravity) and P2 (friction) on gauge pressures. A light pull towards 1 keeps the problem
    well posed with one or two tests (minimum-change solution)."""
    g = tests[np.isfinite(tests["gauge_p"])].reset_index(drop=True)
    if not len(g):
        raise ValueError("No tests with gauge depth and pressure to match.")
    meas = g["gauge_p"].to_numpy(float)

    def res(x):
        pc, _, _ = gauge_pressures(path, fluid, g, correlation, x[0], x[1], u, hc)
        return np.concatenate([(pc - meas) / sigma_p, [(x[0] - 1) * 20, (x[1] - 1) * 4]])
    before = res([1.0, 1.0])[:-2] * sigma_p
    sol = least_squares(res, [1.0, 1.0], bounds=([0.5, 0.05], [1.5, 5.0]), diff_step=1e-3, xtol=1e-6)
    after = res(sol.x)[:-2] * sigma_p
    flags = []
    if not P1_RANGE[0] <= sol.x[0] <= P1_RANGE[1]:
        flags.append(f"Parameter 1 = {sol.x[0]:.3f} is outside {P1_RANGE[0]}-{P1_RANGE[1]}: the gravity term needs "
                     "more than 10 % correction - check the PVT, water cut/GOR or gauge data, or choose another "
                     "correlation.")
    if not P2_RANGE[0] <= sol.x[1] <= P2_RANGE[1]:
        flags.append(f"Parameter 2 = {sol.x[1]:.3f} is outside {P2_RANGE[0]}-{P2_RANGE[1]}: large friction "
                     "correction - check rates, tubing ID and roughness.")
    std = float(np.std(after, ddof=1)) if len(after) > 1 else float(abs(after[0]))
    return VLPMatch(correlation, float(sol.x[0]), float(sol.x[1]), std,
                    float(np.sqrt(np.mean(before ** 2))), float(np.sqrt(np.mean(after ** 2))), after, len(g), flags)


def match_table(matches: list[VLPMatch]) -> pd.DataFrame:
    rows = [{"key": m.correlation, "Correlation": m.label, "Parameter 1": m.p1, "Parameter 2": m.p2,
             "Std Deviation (psi)": m.std, "RMS error before (psi)": m.rmse_before,
             "RMS error after (psi)": m.rmse_after, "Within limits": not m.flags} for m in matches]
    df = pd.DataFrame(rows)
    df["Score"] = (np.abs(df["Parameter 1"] - 1) * 10 + np.abs(df["Parameter 2"] - 1) * 2
                   + df["RMS error after (psi)"] / 10)
    return df.sort_values(["Within limits", "Score"], ascending=[False, True]).reset_index(drop=True)


# ============================================================ VLP and IPR
def vlp_curve(path, fluid, test_row, rates, correlation, p1=1.0, p2=1.0, u=8.0, hc=None):
    """Bottom-node pressure for each rate at the test's THP, water and gas ratios."""
    base = {k: float(test_row[k]) for k in case_of(pd.DataFrame([test_row]), fluid.kind)}
    case = {k: (np.asarray(rates, float) if k == "q" else base[k]) for k in base}
    P, _ = traverse(path, fluid, float(test_row["thp"]), case, correlation, p1, p2, u, hc)
    return P[:, -1]


def intersect(rates, bhp, ipr_q, ipr_pwf):
    """Stable (highest-rate) intersection of a VLP (bhp vs rate) with an IPR (rate vs pwf).
    Rates in the same units. Returns (q, pwf) or None."""
    order = np.argsort(ipr_q)
    iq, ip = np.asarray(ipr_q)[order], np.asarray(ipr_pwf)[order]
    rates = np.asarray(rates, float)
    ok = rates <= iq.max()
    if ok.sum() < 2:
        return None
    r, b = rates[ok], np.asarray(bhp)[ok]
    diff = b - np.interp(r, iq, ip)
    s = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if not len(s):
        return None
    k = s[-1]
    f = diff[k] / (diff[k] - diff[k + 1])
    q = r[k] + f * (r[k + 1] - r[k])
    return float(q), float(np.interp(q, iq, ip))
