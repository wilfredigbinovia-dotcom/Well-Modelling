"""Water and gas injection wells (PROSPER Well Type = Water Injector / Injector).

Water flows down the tubing from the wellhead: gravity raises the pressure, friction lowers it, and the water warms
towards the formation temperature on the way down (PROSPER's rough-approximation heat balance, entered at the
injection temperature at the wellhead). The injectivity (IPR) gives the rate the reservoir takes at a bottom-hole
pressure above the reservoir pressure. The system solution is the rate at which the bottom-hole pressure from the
tubing equals the one the reservoir needs.

Validated on PROSPER Tutorial 03b (Basic Water Injector) - see tests."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import blackoil as bo
from . import complementary as CX
from .correlations import friction_factor
from .equipment import HeatCapacities

RB = 5.614583          # ft3 per bbl
CP_TO_LBFTS = 6.7197e-4
G = 32.174
STB_LB_FRESH = 350.2
WATER_GRADIENT = 0.433 / 62.43     # psi/ft per lb/ft3


class WaterFluid:
    """Injected water (salinity ppm). Rate q in STB/d."""
    kind = "water"
    rate_unit = "STB/d"
    grav = WATER_GRADIENT
    cp_key = "water"

    def __init__(self, salinity_ppm=0.0):
        self.inp = bo.PVTInputs(rsb=1, api=40, gas_sg=0.7, salinity_ppm=float(salinity_ppm))
        _, rho_sc, _ = bo.water_props(self.inp, 14.696, 60.0)
        self.sg = float(rho_sc) / 62.428

    def props(self, p, t):
        """(Bw RB/STB, density lb/ft3, viscosity cp)."""
        return bo.water_props(self.inp, np.maximum(p, 14.7), t)

    def vol(self, p, t):
        """(in-situ ft3/s per unit rate, density, viscosity)."""
        bw, rho, mu = self.props(p, t)
        return bw * RB / 86400.0, rho, mu

    def mass_lbh(self, q):
        return np.asarray(q, float) * RB * 62.428 * self.sg / 24.0


class GasFluid:
    """Injected dry gas (gaspvt.GasPVTInputs: gravity, impurities, viscosity correlation, PVT match). Rate q in
    MMscf/d. Gravity term rho/144 (PROSPER's gas static gradient)."""
    kind = "gas"
    rate_unit = "MMscf/d"
    grav = 1.0 / 144.0
    cp_key = "gas"

    def __init__(self, gpvt, t_offset=0.0, impurities=True):
        """impurities=False: gas properties from the gravity alone, N2 / CO2 / H2S left out of the pseudo-criticals -
        what PROSPER does in a gas injector (Tutorial 04: its injection traverse densities and viscosities are its PVT
        without the impurities to 0.1-0.3 %, and the injectivity uses the same gas; with that the system rate and
        pressure are PROSPER's to 0.02 % / 0.3 psi). PROSPER's PVT tab still applies the impurities.
        t_offset (F): properties evaluated t_offset below the flowing temperature (an earlier approximation of the
        same effect; 0 = none)."""
        import dataclasses
        from . import gaspvt as gp
        if not impurities:
            gpvt = dataclasses.replace(gpvt, h2s=0.0, co2=0.0, n2=0.0)
        self.gp, self.inp = gp, gpvt
        self.sg = float(gpvt.wellstream_sg())
        self.t_offset = float(t_offset)
        self.impurities = bool(impurities)

    def props(self, p, t):
        """(Bg ft3/scf, density lb/ft3, viscosity cp)."""
        p = np.maximum(np.asarray(p, float), 14.7)
        t = np.asarray(t, float) - self.t_offset
        t = np.broadcast_to(t, p.shape) if p.ndim else t
        z, bg, rho, mu = self.gp.gas_props(self.inp, np.atleast_1d(p), np.atleast_1d(t) + 0 * np.atleast_1d(p))
        if np.ndim(p) == 0:
            return float(bg[0]), float(rho[0]), float(mu[0])
        return bg, rho, mu

    def z(self, p, t):
        return self.gp.gas_props(self.inp, np.atleast_1d(p), np.atleast_1d(t) + 0 * np.atleast_1d(p))[0]

    def vol(self, p, t):
        bg, rho, mu = self.props(p, t)
        return bg * 1e6 / 86400.0, rho, mu

    def mass_lbh(self, q):
        return np.asarray(q, float) * 1e6 * 0.0764 * self.sg / 24.0


@dataclass
class InjectionPath:
    """Flow path for an injector (wellhead at node 0, MD increasing down)."""
    md: np.ndarray
    tvd: np.ndarray
    t_form: np.ndarray
    seg_len: np.ndarray
    sin_t: np.ndarray
    d: np.ndarray          # flow ID, ft (per segment)
    rough: np.ndarray      # ft
    d_ht: np.ndarray       # heat-transfer diameter, ft
    restrictions: dict = field(default_factory=dict)   # node index -> restriction ID (in): SSSV, restrictions


def from_flow_path(path) -> InjectionPath:
    """From a vlp.FlowPath (tubing flow)."""
    return InjectionPath(path.md, path.tvd, path.t_form, path.seg_len, path.sin_t, path.d_h, path.rough, path.d_ht,
                         dict(getattr(path, "restrictions", {}) or {}))


def _restriction_dp(fluid, p, t, q, d_in, cd=0.9):
    """Orifice pressure drop (psi) of the injected fluid through a restriction of ID d_in."""
    vf, rho, _ = fluid.vol(p, t)
    v = q * vf / (np.pi / 4 * (d_in / 12) ** 2)
    return rho * v * v / (2 * G * cd * cd) / 144.0


def temperature_down(path, w_lbh, cp, u, t_inj):
    """Fluid temperature (cases x nodes) of water injected at t_inj (F) at the wellhead, exchanging heat with the
    formation (geothermal gradient) through the overall coefficient u (BTU/h/ft2/F)."""
    w = np.atleast_1d(np.asarray(w_lbh, float))
    n = len(path.md)
    T = np.empty((w.size, n))
    T[:, 0] = t_inj
    for i in range(n - 1):
        L = path.seg_len[i]
        a = u * np.pi * path.d_ht[i] / np.maximum(w * cp, 1e-9)
        tf0, tf1 = path.t_form[i], path.t_form[i + 1]
        g = (tf1 - tf0) / L
        t0 = T[:, i]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            tn = tf0 + g * L - g / a + (t0 - tf0 + g / a) * np.exp(-a * L)
        T[:, i + 1] = np.where(a * L < 1e-9, t0, tn)
    return T


def _grad(fluid, path, i, p, t, q, p1, p2):
    """(gravity, friction) gradients in psi/ft for downward water flow at rates q (STB/d)."""
    vf, rho, mu = fluid.vol(p, t)
    a = np.pi / 4 * path.d[i] ** 2
    v = q * vf / a
    re = rho * np.abs(v) * path.d[i] / (mu * CP_TO_LBFTS)
    f = friction_factor(np.maximum(re, 1e-9), path.rough[i] / path.d[i])
    fric = np.where(v > 0, f * rho * v * v / (2 * G * path.d[i]) / 144.0, 0.0)
    # PROSPER's liquid static gradient: 0.433 psi/ft x specific gravity (rho / 62.43) - i.e. rho / 144.18 (its
    # injection-string static gradient is the density column / 144.18 at every node, Tutorial 03b)
    return p1 * rho * path.sin_t[i] * fluid.grav, p2 * fric, v, rho, mu


def enthalpy_down(path, fluid, P, q, u_seg, t_inj):
    """Improved Approximation for an injector with an enthalpy-capable fluid (co2.CO2InjectionFluid): the energy
    balance h(i+1) = h(i) + L sin(theta)/J + Q marched down from the injection temperature, T from h at the node
    pressure (so Joule-Thomson and heat capacity come from the fluid's enthalpy). Q = U pi d L (Te - T) over the
    segment, integrated exponentially - w cp (Te - T) (1 - exp(-a)), a = U pi d L / (w cp) - so it stays bounded
    at low rates where the fluid reaches the formation temperature within a segment."""
    J = 778.17
    q = np.atleast_1d(np.asarray(q, float))
    w = np.maximum(fluid.mass_lbh(q), 1e-9)
    n = len(path.md)
    T = np.empty((q.size, n))
    T[:, 0] = t_inj
    h = fluid.enthalpy(P[:, 0], T[:, 0])
    for i in range(n - 1):
        L = path.seg_len[i]
        te = 0.5 * (path.t_form[i] + path.t_form[i + 1])
        cp = np.maximum(fluid.cp(P[:, i], T[:, i]), 1e-6)
        a = np.asarray(u_seg)[..., i] * np.pi * path.d_ht[i] * L / (w * cp)
        h = h + L * path.sin_t[i] / J + cp * (te - T[:, i]) * -np.expm1(-a)
        T[:, i + 1] = fluid.temperature(P[:, i + 1], h, T[:, i])
    return T


def traverse(path: InjectionPath, fluid: WaterFluid, whp, q, u=8.0, t_inj=60.0, hc: HeatCapacities | None = None,
             p1=1.0, p2=1.0, thermal=None):
    """Pressure and temperature (cases x nodes) from the wellhead down, for injection rates q (STB/d) at wellhead
    pressure whp (psia). Returns (P, T). thermal: None (Rough Approximation with u) or {"model": "improved",
    "u_seg": per-segment U} - with an enthalpy-capable fluid the temperature then follows its enthalpy balance
    (alternated with the pressure march), otherwise the per-segment U replaces u."""
    hc = hc or HeatCapacities()
    q = np.atleast_1d(np.asarray(q, float))
    whp = np.broadcast_to(np.asarray(whp, float), q.shape).astype(float)
    u_seg = thermal.get("u_seg") if thermal else None
    coupled = u_seg is not None and getattr(fluid, "has_enthalpy", False)
    if u_seg is not None and not coupled:
        T = _temperature_down_seg(path, fluid.mass_lbh(q), getattr(hc, fluid.cp_key), u_seg, t_inj)
    else:
        T = temperature_down(path, fluid.mass_lbh(q), getattr(hc, fluid.cp_key), u, t_inj)

    def down(T):
        n = len(path.md)
        P = np.empty((q.size, n))
        p = np.maximum(whp, 14.7)
        P[:, 0] = p
        for i in range(n - 1):
            L = path.seg_len[i]
            e0, f0, *_ = _grad(fluid, path, i, p, T[:, i], q, p1, p2)
            pp = np.maximum(p + (e0 - f0) * L, 14.7)
            e1, f1, *_ = _grad(fluid, path, i, pp, T[:, i + 1], q, p1, p2)
            p = np.maximum(p + 0.5 * ((e0 - f0) + (e1 - f1)) * L, 14.7)
            if (i + 1) in path.restrictions:                 # injection: pressure below the restriction is lower
                p = np.maximum(p - _restriction_dp(fluid, p, T[:, i + 1], q, path.restrictions[i + 1]), 14.7)
            P[:, i + 1] = p
        return P
    P = down(T)
    if coupled:
        for _ in range(3):
            T = enthalpy_down(path, fluid, P, q, u_seg, t_inj)
            P = down(T)
    return P, T


def _temperature_down_seg(path, w_lbh, cp, u_seg, t_inj):
    """temperature_down with U per segment."""
    w = np.atleast_1d(np.asarray(w_lbh, float))
    n = len(path.md)
    T = np.empty((w.size, n))
    T[:, 0] = t_inj
    for i in range(n - 1):
        L = path.seg_len[i]
        a = np.asarray(u_seg)[..., i] * np.pi * path.d_ht[i] / np.maximum(w * cp, 1e-9)
        tf0, tf1 = path.t_form[i], path.t_form[i + 1]
        g = (tf1 - tf0) / L
        t0 = T[:, i]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            tn = tf0 + g * L - g / a + (t0 - tf0 + g / a) * np.exp(-a * L)
        T[:, i + 1] = np.where(a * L < 1e-9, t0, tn)
    return T


# ======================================================================= injectivity
@dataclass
class InjectionIPR:
    """Linear injectivity: q = J (Pwf - Pr) above the reservoir pressure (STB/d, psia)."""
    model: str
    pr: float
    j: float                              # injectivity index, STB/d/psi
    skin: float = 0.0
    skin_coef: float = 0.0                # psi per STB/d per unit skin
    params: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    fluid: str = "water_injection"

    def rate(self, pwf):
        return np.maximum(self.j * (np.asarray(pwf, float) - self.pr), 0.0)

    def pwf(self, q):
        return self.pr + np.asarray(q, float) / self.j

    def dp_skin(self, q):
        return self.skin_coef * self.skin * np.asarray(q, float)

    @property
    def aof(self):
        """No physical meaning for an injector (no upper limit on the bottom-hole pressure); J x Pr, the rate at
        twice the reservoir pressure, is returned as a scale for plots. (PROSPER's Tutorial 03b reports 410,000
        STB/d - 100 x the reservoir pressure in psig - with J = 7.65.)"""
        return float(self.j * self.pr)

    def table(self, q) -> pd.DataFrame:
        q = np.asarray(q, float)
        return pd.DataFrame({"Injection rate (STB/d)": q, "Pwf (psia)": self.pwf(q),
                             "dP skin (psi)": self.dp_skin(q)})


def darcy_injection(pr, t_res, k, h, area_acres, rw, skin, fluid: WaterFluid, t_visc=None):
    """Darcy (pseudo-steady state) injectivity: J = k h / (141.2 mu Bw (ln(re/rw) - 0.75 + S)), water viscosity and
    Bw at the reservoir pressure and temperature t_visc (default the reservoir temperature)."""
    t_visc = t_res if t_visc is None else t_visc
    bw, _, mu = (float(np.asarray(x)) for x in fluid.props(pr, t_visc))
    re = np.sqrt(area_acres * 43560.0 / np.pi)
    c = 141.2 * mu * bw / (k * h)
    lnr = np.log(re / rw) - 0.75
    j = 1.0 / (c * (lnr + skin))
    return InjectionIPR("Darcy", float(pr), float(j), float(skin), float(c), {
        "Injectivity index (STB/d/psi)": j, "Water viscosity (cp)": mu, "Water FVF (RB/STB)": bw,
        "Drainage radius (ft)": re, "ln(re/rw) - 0.75": lnr, "Viscosity temperature (F)": t_visc})


@dataclass
class GasInjectionIPR:
    """Gas injectivity from the Petroleum Experts gas IPR run in reverse:
        M(Pwf) - M(Pr) = a q + b q^2      (q Mscf/d; a = 1422 T (X + S)/(k_g h), b = 1422 T D/(k_g h))
    with PROSPER's pseudo-pressure M at reservoir temperature. Rates in MMscf/d."""
    model: str
    pr: float
    a: float
    b: float
    m_fn: object
    skin: float = 0.0
    a_skin: float = 0.0                   # part of a due to skin
    p_max: float = 30000.0
    params: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    fluid: str = "gas_injection"

    def __post_init__(self):
        self._p = np.linspace(self.pr, self.p_max, 4000)
        self._m = self.m_fn(self._p)
        self._mr = float(self.m_fn(self.pr))

    def _p_of_m(self, m):
        return np.interp(m, self._m, self._p)

    def pwf(self, q):
        q = np.asarray(q, float) * 1000.0
        return self._p_of_m(self._mr + self.a * q + self.b * q * q)

    def rate(self, pwf):
        dm = np.maximum(self.m_fn(np.asarray(pwf, float)) - self._mr, 0.0)
        return (-self.a + np.sqrt(self.a * self.a + 4 * self.b * dm)) / (2 * self.b) / 1000.0 if self.b > 0 \
            else dm / self.a / 1000.0

    def dp_skin(self, q):
        """Pressure for the skin part of the laminar term."""
        qm = np.asarray(q, float) * 1000.0
        return self.pwf(q) - self._p_of_m(self._mr + (self.a - self.a_skin) * qm + self.b * qm * qm)

    @property
    def aof(self):
        """No physical meaning for an injector; the rate at 2 x Pr is returned as a plot scale."""
        return float(self.rate(2 * self.pr))

    def table(self, q) -> pd.DataFrame:
        q = np.asarray(q, float)
        return pd.DataFrame({"Injection rate (MMscf/d)": q, "Pwf (psia)": self.pwf(q),
                             "dP skin (psi)": self.dp_skin(q)})


def petex_gas_injection(pr, t_res, k, h, area_acres, ca, rw, hp, t_days, phi, swc, skin, fluid: GasFluid,
                        d_entered=None, perm="total"):
    """Petroleum Experts gas IPR (inflow.petex_gas, validated on PROSPER Tutorial 02) for injection."""
    from . import inflow as winf
    p_hi = max(4 * pr, 15000.0)
    if hasattr(fluid, "mu_z"):          # compositional (EOS / CO2) stream: PROSPER's gauge M(P) of its properties
        grid = np.linspace(14.696, p_hi, 3000)
        mu_g, z_g = fluid.mu_z(grid, t_res)
        integ = 2 * (grid - 14.696) / (mu_g * z_g)
        mg = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(grid))])
        m_fn = lambda p: np.interp(p, grid, mg)
        mu3, z3 = fluid.mu_z(np.array([pr - 5, pr, pr + 5]), t_res)
        mu_r, sg = float(mu3[1]), float(fluid.sg)
    else:
        inp = fluid.inp
        grid = np.linspace(14.696, p_hi, 3000)
        mg = fluid.gp.pseudo_pressure(inp, grid, t_res)
        m_fn = lambda p: np.interp(p, grid, mg)
        z3 = fluid.gp.gas_props(inp, np.array([pr - 5, pr, pr + 5]), np.full(3, float(t_res)))[0]
        mu_r = float(fluid.gp.gas_props(inp, np.array([pr]), np.array([float(t_res)]))[3][0])
        sg = inp.gas_sg
    cg = 1 / pr - (z3[2] - z3[0]) / 10 / z3[1]
    cv = winf.petex_gas(pr, t_res, k, h, area_acres, ca, rw, hp, t_days, phi, swc, skin, m_fn, mu_r, cg, sg,
                        d_entered, perm)
    a, b = cv.params["a (laminar)"], cv.params["b (turbulent)"]
    x = cv.params["Drainage term X"]
    out = GasInjectionIPR("Petroleum Experts", float(pr), float(a), float(b), m_fn, float(skin),
                          float(a * skin / (x + skin)) if (x + skin) else 0.0, p_hi, dict(cv.params))
    return out


def pi_injection(pr, j):
    return InjectionIPR("P.I. Entry", float(pr), float(j), 0.0, 0.0, {"Injectivity index (STB/d/psi)": j})


# ======================================================================= system
@dataclass
class InjectionResult:
    q: float
    pwf: float
    whp: float
    P: np.ndarray
    T: np.ndarray
    vlp_q: np.ndarray
    vlp_p: np.ndarray
    ipr_p: np.ndarray


def vlp_curve(path, fluid, whp, rates, u=8.0, t_inj=60.0, hc=None, p1=1.0, p2=1.0, with_t=False, thermal=None):
    P, T = traverse(path, fluid, whp, rates, u, t_inj, hc, p1, p2, thermal)
    return (P[:, -1], T[:, -1]) if with_t else P[:, -1]


def _ipr_p(ipr, q, t):
    """IPR pressure; injectivities that depend on the sandface temperature (TIF) get the tubing's."""
    return ipr.pwf(q, t) if getattr(ipr, "sandface_temperature", False) else ipr.pwf(q)


def solve(path, fluid, whp, ipr: InjectionIPR, u=8.0, t_inj=60.0, hc=None, p1=1.0, p2=1.0, n=20, q_max=None,
          thermal=None):
    """Injection rate where the tubing (VLP) and reservoir (IPR) bottom-hole pressures meet. None if the well
    cannot inject (the static column is below the reservoir pressure)."""
    if q_max is None:                             # rate grid as PROSPER: up to where friction wins clearly
        q_max = 4000.0 if fluid.kind == "water" else 10.0
        for _ in range(12):
            v_, t_ = vlp_curve(path, fluid, whp, [q_max], u, t_inj, hc, p1, p2, with_t=True, thermal=thermal)
            if v_[0] < _ipr_p(ipr, np.array([q_max]), t_)[0]:
                break
            q_max *= 2
    rates = np.linspace(q_max / (n * 50), q_max, n)
    vp, tb = vlp_curve(path, fluid, whp, rates, u, t_inj, hc, p1, p2, with_t=True, thermal=thermal)
    ip = _ipr_p(ipr, rates, tb)
    d = vp - ip
    if d[0] <= 0:
        return None, rates, vp, ip
    k = int(np.argmax(d <= 0)) if (d <= 0).any() else None
    if k is None:
        return None, rates, vp, ip
    def f(x):
        v_, t_ = vlp_curve(path, fluid, whp, [x], u, t_inj, hc, p1, p2, with_t=True, thermal=thermal)
        return v_[0] - _ipr_p(ipr, np.array([x]), t_)[0]
    q = brentq(f, rates[k - 1], rates[k], xtol=1e-3)
    P, T = traverse(path, fluid, whp, [q], u, t_inj, hc, p1, p2, thermal)
    return InjectionResult(q, float(P[0, -1]), float(whp), P[0], T[0], rates, vp, ip), rates, vp, ip


def details(path, fluid, res: InjectionResult, ipr: InjectionIPR, comp: CX.CompOptions | None = None,
            p_unit_gauge=True):
    """PROSPER-style solution summary for an injector."""
    comp = comp or CX.CompOptions()
    P, T, q = res.P, res.T, res.q
    grav = fric = 0.0
    vs, rhos, ves, cfs, grains, transit = [], [], [], [], [], 0.0
    for i in range(len(path.md) - 1):
        L = path.seg_len[i]
        e0, f0, v0, r0, m0 = _grad(fluid, path, i, P[i], T[i], q, 1.0, 1.0)
        e1, f1, v1, r1, m1 = _grad(fluid, path, i, P[i + 1], T[i + 1], q, 1.0, 1.0)
        grav += 0.5 * float(e0 + e1) * L
        fric -= 0.5 * float(f0 + f1) * L
        for v_, r_, m_ in ((v0, r0, m0), (v1, r1, m1)):
            v_, r_, m_ = float(v_), float(r_), float(m_)
            vs.append(v_)
            rhos.append(r_)
            nv = CX.node_values("oil", v_, 0.0, r_, m_, r_, np.nan, 50.0, path.d[i] * 12, np.nan, comp)
            ves.append(nv["Erosional Velocity (ft/s)"])
            cfs.append(nv["C Factor (flow)"])
            grains.append(nv["Max. Grain Size Diameter (in)"])
        transit += L / max(0.5 * float(v0 + v1), 1e-9)
    g = 14.696 if p_unit_gauge else 0.0
    u_ = "psig" if p_unit_gauge else "psia"
    ru = getattr(fluid, "rate_unit", "STB/d")
    dps = ipr.dp_skin(np.array([q]), np.array([T[-1]])) if getattr(ipr, "sandface_temperature", False) \
        else ipr.dp_skin(q)
    rows = [(f"Injection rate ({ru})", q), (("Gas rate" if fluid.kind == "gas" else "Water rate") + f" ({ru})", q),
            (f"Solution node pressure ({u_})", res.pwf - g), ("dP total skin (psi)", float(np.ravel(dps)[0])),
            ("Total skin", ipr.skin), (f"Wellhead pressure ({u_})", res.whp - g),
            ("Wellhead temperature (°F)", float(T[0])), ("Bottom node temperature (°F)", float(T[-1])),
            ("dP friction - injection string (psi)", fric), ("dP gravity - injection string (psi)", grav),
            ("Cumulative transit time (s)", transit),
            ("Max. no-slip velocity (ft/s)", max(vs)), ("Min. no-slip velocity (ft/s)", min(vs)),
            (f"Max. pressure ({u_})", float(P.max()) - g), (f"Min. pressure ({u_})", float(P.min()) - g),
            ("Max. erosional velocity (ft/s)", np.nanmax(ves)), ("Min. erosional velocity (ft/s)", np.nanmin(ves)),
            ("Max. C factor (flow)", np.nanmax(cfs)), ("Min. C factor (flow)", np.nanmin(cfs)),
            ("Max. grain diameter (in)", np.nanmax(grains)), ("Min. grain diameter (in)", np.nanmin(grains)),
            ("Erosional velocity flag", "Yes" if max(v / ve for v, ve in zip(vs, ves) if ve > 0) > 1 else "No")]
    return pd.DataFrame(rows, columns=["Result", "Value"])


def gradient_table(path, fluid, q, P, T, p_unit_gauge=True) -> pd.DataFrame:
    g = 14.696 if p_unit_gauge else 0.0
    recs = []
    for i in range(len(path.md)):
        j = min(i, len(path.md) - 2)
        e, f, v, rho, mu = _grad(fluid, path, j, P[i], T[i], q, 1.0, 1.0)
        fvf, _, _ = fluid.props(P[i], T[i])
        name = "Gas" if fluid.kind == "gas" else "Water"
        fvf_lab = "Gas FVF (ft3/scf)" if fluid.kind == "gas" else "Water FVF (RB/STB)"
        recs.append({"Measured depth (ft)": path.md[i], "TVD (ft)": path.tvd[i],
                     f"Pressure ({'psig' if g else 'psia'})": P[i] - g, "Temperature (°F)": T[i],
                     "Gravity gradient (psi/ft)": float(e), "Friction gradient (psi/ft)": -float(f),
                     "Velocity (ft/s)": float(v), f"{name} density (lb/ft3)": float(rho),
                     f"{name} viscosity (cp)": float(mu), fvf_lab: float(fvf),
                     "Diameter (in)": path.d[j] * 12, "Formation temperature (°F)": path.t_form[i]})
    return pd.DataFrame(recs)


# ======================================================================= well test matching
# PROSPER VLP/IPR Matching for injectors (Tutorials 21 and 22): Estimate U Value, Match VLP, VLP/IPR, Adjust IPR.
def inj_test_columns(gas: bool) -> list[str]:
    return ["Test Point Date", "Test Point Comment", "Injected Fluid Pressure (psia)",
            "Downstream Temperature (deg F)", "Gas Rate (MMscf/d)" if gas else "Water Rate (STB/d)",
            "Gauge Depth (Measured) (ft)", "Gauge Pressure (psia)", "Reservoir Pressure (psia)"]


def clean_inj_tests(df: pd.DataFrame, gas: bool, md_top=None, md_bottom=None):
    """Usable test rows (whp, t_down, q, gauge_md, gauge_p, pr, label) and a list of (level, message)."""
    cols = inj_test_columns(gas)
    num = lambda c: pd.to_numeric(df.get(c, pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(float)
    whp, td, q, gmd, gp_, pr = (num(c) for c in cols[2:])
    rows, issues = [], []
    for i in range(len(df)):
        if not (np.isfinite(whp[i]) or np.isfinite(q[i])):
            continue
        lab = str(df.index[i])
        if not (np.isfinite(whp[i]) and np.isfinite(q[i]) and q[i] > 0):
            issues.append(("error", f"Test {lab}: injected fluid pressure and a positive rate are required."))
            continue
        if np.isfinite(gmd[i]) and md_bottom is not None and not (md_top - 1e-6 <= gmd[i] <= md_bottom + 1e-6):
            issues.append(("error", f"Test {lab}: gauge depth {gmd[i]:g} ft is outside the tubing "
                                    f"({md_top:g}-{md_bottom:g} ft)."))
            continue
        if np.isfinite(gmd[i]) != np.isfinite(gp_[i]):
            issues.append(("warning", f"Test {lab}: gauge depth and gauge pressure go together - the gauge is "
                                      "ignored."))
            gmd[i] = gp_[i] = np.nan
        rows.append({"label": lab, "whp": whp[i], "t_down": td[i], "q": q[i], "gauge_md": gmd[i],
                     "gauge_p": gp_[i], "pr": pr[i]})
    return pd.DataFrame(rows, columns=["label", "whp", "t_down", "q", "gauge_md", "gauge_p", "pr"]), issues


def _node(path, md):
    return int(np.argmin(np.abs(path.md - md)))


def estimate_u(path, fluid, tests, t_inj=60.0, hc=None, u_max=500.0) -> pd.DataFrame:
    """U per test that brings the calculated temperature at the bottom node (PROSPER's 'downstream' temperature
    for an injector) to the measured one."""
    out = []
    for _, t in tests.iterrows():
        u = np.nan
        if np.isfinite(t["t_down"]):
            f = lambda x: traverse(path, fluid, t["whp"], [t["q"]], x, t_inj, hc)[1][0, -1] - t["t_down"]
            try:
                u = brentq(f, 1e-3, u_max, xtol=1e-6)
            except ValueError:
                u = np.nan
        out.append({"Test": t["label"], "Downstream temperature (°F)": t["t_down"], "U (BTU/h/ft2/F)": u})
    return pd.DataFrame(out)


def gauge_pressures(path, fluid, tests, u=8.0, t_inj=60.0, hc=None, p1=1.0, p2=1.0):
    """Calculated pressure at each test's gauge depth (NaN where a test has no gauge)."""
    out = np.full(len(tests), np.nan)
    for i, (_, t) in enumerate(tests.iterrows()):
        if np.isfinite(t["gauge_md"]):
            P, _ = traverse(path, fluid, t["whp"], [t["q"]], u, t_inj, hc, p1, p2)
            out[i] = P[0, _node(path, t["gauge_md"])]
    return out


@dataclass
class InjMatch:
    p1: float
    p2: float
    std: float
    before: np.ndarray       # calculated - measured gauge pressure (psi), unmatched
    after: np.ndarray        # same, matched
    n: int


def match_vlp(path, fluid, tests, u=8.0, t_inj=60.0, hc=None, fit_p2=False) -> InjMatch:
    """Parameter 1 (gravity multiplier) by least squares on the gauge pressures; Parameter 2 (friction) stays 1
    unless fit_p2 (PROSPER's Tutorials 21 and 22 both end with P2 = 1: in an injector friction is a small part of
    the pressure change and gravity carries the match)."""
    g = tests[np.isfinite(tests["gauge_p"])].reset_index(drop=True)
    if not len(g):
        raise ValueError("No tests with gauge depth and pressure to match.")
    meas = g["gauge_p"].to_numpy(float)
    res = lambda x: gauge_pressures(path, fluid, g, u, t_inj, hc, x[0], x[1] if fit_p2 else 1.0) - meas
    before = res([1.0, 1.0])
    from scipy.optimize import least_squares
    if fit_p2:
        rr = lambda x: np.concatenate([res(x), [(x[1] - 1) * 40]])
        x = least_squares(rr, [1.0, 1.0], bounds=([0.5, 0.05], [1.5, 5.0]), diff_step=1e-4, xtol=1e-8).x
    else:
        x = [least_squares(lambda y: res([y[0], 1.0]), [1.0], bounds=([0.5], [1.5]), diff_step=1e-4,
                           xtol=1e-10).x[0], 1.0]
    after = res(x)
    std = float(np.std(after, ddof=1)) if len(after) > 1 else float(abs(after[0]))
    return InjMatch(float(x[0]), float(x[1]), std, before, after, len(g))


def vlp_ipr_test(path, fluid, t, ipr, u=8.0, t_inj=60.0, hc=None, p1=1.0, p2=1.0):
    """One test: system rate and pressure at the test's injected fluid pressure, and the bottom-hole pressure
    on the VLP at the measured rate ('measured' BHP). Returns (result or None, bhp_at_measured_rate)."""
    res, *_ = solve(path, fluid, t["whp"], ipr, u, t_inj, hc, p1, p2)
    bhp = float(vlp_curve(path, fluid, t["whp"], [t["q"]], u, t_inj, hc, p1, p2)[0])
    return res, bhp


def adjust_ipr(path, fluid, t, ipr_for, what="skin", u=8.0, t_inj=60.0, hc=None, p1=1.0, p2=1.0,
               lo=None, hi=None):
    """Skin (what='skin') or reservoir pressure (what='pr') at which the system rate equals the test's rate.
    ipr_for(skin=..., pr=...) builds the injectivity. Returns the value or NaN."""
    if what == "skin":
        lo, hi = (-6.0 if lo is None else lo), (200.0 if hi is None else hi)
    else:
        bhp = float(vlp_curve(path, fluid, t["whp"], [t["q"]], u, t_inj, hc, p1, p2)[0])
        lo, hi = (15.0 if lo is None else lo), (bhp - 0.01 if hi is None else hi)

    def f(x):
        ipr = ipr_for(**{what: x})
        r, *_ = solve(path, fluid, t["whp"], ipr, u, t_inj, hc, p1, p2)
        q = r.q if r is not None else 0.0
        return q - t["q"]
    try:
        return float(brentq(f, lo, hi, xtol=1e-4))
    except ValueError:
        return float("nan")
