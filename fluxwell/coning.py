"""Water and gas coning for the well operating envelope (vertical wells).

Oil wells (Ahmed, Reservoir Engineering Handbook, ch. 9):
  Meyer & Garder (1954)  Qoc = 0.246e-4 [drho(lb/ft3) / ln(re/rw)] [ko / (mu_o Bo)] (h^2 - hp^2)            STB/d
  Schols (1972)          Qoc = [drho(g/cc) ko (h^2 - hp^2) / (2049 mu_o Bo)] [0.432 + pi/ln(re/rw)] (h/re)^0.14
  water coning: drho = rho_w - rho_o, perforations at the TOP of the oil column (hp from the top);
  gas coning:   drho = rho_o - rho_g, perforations at the BOTTOM (hp from the base); design rate = the lower.
  Breakthrough at a rate above critical: Z = 0.00307 drho(g/cc) kh h (h - hp) / (mu_o Bo Qo);
    Sobocinski & Cornelius (1965) tD = (Z/4)(16 + 7Z - 3Z^2)/(7 - 2Z) (Z < 3.5), alpha = 0.5 (M < 1) / 0.6;
    Bournazel & Jeanson (1971) tD = Z/(3 - 0.7Z), alpha = 0.7;
    t_BT (days) = 20,325 mu_o phi h tD / [drho(lb/ft3) kv (1 + M^alpha)], M the end-point mobility ratio.
    Built for water coning with M up to ~10: for gas (M typically 20-100) indicative only.
  Kuo & DesBrisay (1983) water cut after breakthrough: WC_limit = M hw / (M hw + h); WCD = 0 (tD < 0.5),
    0.29 + 0.94 log10(tD) (0.5-5.7), 1 (> 5.7), tD = t / t_BT.
  Thin oil rim (gas cap and aquifer): perforations placed so both cones arrive together (drho_g Dt = drho_w Db);
    Meyer-Garder with drho_eff = drho_g drho_w / (drho_g + drho_w) - their simultaneous-coning result.
  Static gravity screen: drawdown that lifts a static cone to the perforations, 0.433 drho(g/cc) standoff.

Gas wells with bottom water (Hagoort 1988, Fundamentals of Gas Reservoir Engineering; Smulski 2012, Arch. Min. Sci.
57(2) 451-470): the same Meyer-Garder (Dupuit) and Schols forms at reservoir conditions with the gas mobility k krg /
mu_g, drho = rho_w - rho_g and, for Schols, the anisotropy factor (kv/kh)^0.07; converted to standard conditions
with Bg (rb/Mscf = 5.04 z T / p). Non-Darcy flow lowers it (Smulski): the drawdown the cone tolerates is the laminar
one at the Darcy critical rate, so lam q + tur q^2 = lam q_D -> q = (sqrt(1 + 4 r q_D) - 1) / (2 r), r = tur / lam
(1/(Mscf/d)) - from the gas IPR's own coefficients. Against a numerical model Smulski found the Darcy forms 120-200 %
high on average and the non-Darcy forms 44-68 % high: screening values, conservative only after the correction.

Checked against the user's Water Coning and Gas Coning Calculator workbooks exactly."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G_CC_TO_LBFT3 = 62.428


@dataclass
class ConeInputs:
    """Oil-well coning. Densities g/cc at reservoir conditions; k md (kh = effective oil permeability)."""
    h: float                       # ft, oil column (top to WOC for water coning; GOC to base for gas coning)
    hp: float                      # ft, perforated interval (from the top for water coning, from the base for gas)
    re: float                      # ft
    rw: float                      # ft
    kh: float                      # md
    kv: float                      # md
    phi: float
    rho_o: float
    mu_o: float                    # cp
    bo: float                      # rb/STB
    rho_other: float               # water (water coning) or gas (gas coning) density, g/cc
    mu_other: float                # cp
    kr_other: float                # end-point krw or krg
    kro: float = 0.8
    hw: float = 0.0                # ft, water zone thickness (Kuo-DesBrisay)
    kind: str = "water"            # "water" or "gas"

    @property
    def drho(self):
        return abs(self.rho_other - self.rho_o)

    @property
    def standoff(self):
        return self.h - self.hp

    @property
    def mobility_ratio(self):
        return (self.kr_other / self.mu_other) / (self.kro / self.mu_o)


def meyer_garder(c: ConeInputs, hp=None):
    hp = c.hp if hp is None else hp
    return 0.246e-4 * (c.drho * G_CC_TO_LBFT3 / np.log(c.re / c.rw)) * (c.kh / (c.mu_o * c.bo)) * (c.h ** 2 - hp ** 2)


def schols(c: ConeInputs, hp=None):
    hp = c.hp if hp is None else hp
    return (c.drho * c.kh * (c.h ** 2 - hp ** 2) / (2049 * c.mu_o * c.bo)) * (0.432 + np.pi / np.log(c.re / c.rw)) \
        * (c.h / c.re) ** 0.14


def critical_rate(c: ConeInputs):
    """Design critical oil rate (STB/d): water coning the lower of Meyer-Garder and Schols; gas coning Meyer-Garder
    (Schols is a water-coning fit)."""
    return min(meyer_garder(c), schols(c)) if c.kind == "water" else meyer_garder(c)


def gravity_drawdown(c: ConeInputs):
    """Static gravity screen (psi): 0.433 drho(g/cc) x standoff."""
    return 0.433 * c.drho * c.standoff


def cone_height_z(c: ConeInputs, qo):
    return 0.00307 * c.drho * c.kh * c.h * c.standoff / (c.mu_o * c.bo * qo)


def breakthrough_time(c: ConeInputs, qo, method="Sobocinski-Cornelius"):
    """Days to breakthrough at oil rate qo (STB/d); inf where the correlation predicts a stable cone."""
    z = cone_height_z(c, qo)
    m = c.mobility_ratio
    if method == "Bournazel-Jeanson":
        if z >= 3 / 0.7:
            return np.inf
        td, alpha = z / (3 - 0.7 * z), 0.7
    else:
        if z >= 3.5:
            return np.inf
        td = (z / 4) * (16 + 7 * z - 3 * z * z) / (7 - 2 * z)
        alpha = 0.5 if m < 1 else 0.6
    return 20325 * c.mu_o * c.phi * c.h * td / (c.drho * G_CC_TO_LBFT3 * c.kv * (1 + m ** alpha))


def limiting_water_cut(c: ConeInputs):
    m = c.mobility_ratio
    return m * c.hw / (m * c.hw + c.h)


def kuo_desbrisay_wc(c: ConeInputs, t_days, t_bt):
    """Water cut (fraction) at t_days after the start for breakthrough time t_bt."""
    td = np.asarray(t_days, float) / t_bt
    with np.errstate(divide="ignore"):
        wcd = np.where(td < 0.5, 0.0, np.where(td > 5.7, 1.0, 0.29 + 0.94 * np.log10(np.maximum(td, 1e-12))))
    return wcd * limiting_water_cut(c)


def oil_rim(c: ConeInputs, rho_w, rho_g):
    """Simultaneous gas and water coning. Returns dict: Dt (GOC to top perforation), Db (base perforation to WOC),
    drho_eff (lb/ft3), critical rate (STB/d) and gravity drawdown (psi)."""
    dg, dw = c.rho_o - rho_g, rho_w - c.rho_o
    avail = c.h - c.hp
    dt, db = avail * dw / (dg + dw), avail * dg / (dg + dw)
    eff = dg * dw / (dg + dw) * G_CC_TO_LBFT3
    q = 0.246e-4 * (eff / np.log(c.re / c.rw)) * (c.kh / (c.mu_o * c.bo)) * (c.h ** 2 - c.hp ** 2)
    return {"Dt (ft)": dt, "Db (ft)": db, "drho_eff (lb/ft3)": eff, "Critical rate (STB/d)": q,
            "Gravity drawdown (psi)": 0.433 * dg * dt}


# ------------------------------------------------------------------ gas wells with bottom water
@dataclass
class GasConeInputs:
    h: float                       # ft, gas column above the GWC
    hp: float                      # ft, perforated from the top
    re: float
    rw: float
    k: float                       # md, horizontal (absolute)
    kv_kh: float
    krg: float                     # gas relative permeability (end point at connate water)
    rho_w: float                   # g/cc at reservoir conditions
    rho_g: float                   # g/cc at reservoir conditions
    mu_g: float                    # cp at reservoir conditions
    z: float
    t_f: float                     # reservoir temperature, F
    p: float                       # reservoir pressure, psia

    @property
    def drho(self):
        return self.rho_w - self.rho_g

    @property
    def bg_rb_mscf(self):
        return 5.04 * self.z * (self.t_f + 459.67) / self.p


def gas_meyer_garder(c: GasConeInputs):
    """Darcy critical gas rate (Mscf/d), Dupuit / Meyer-Garder form (Hagoort)."""
    q_res = 0.246e-4 * (c.drho * G_CC_TO_LBFT3 / np.log(c.re / c.rw)) * (c.k * c.krg / c.mu_g) * (c.h ** 2 - c.hp ** 2)
    return q_res / c.bg_rb_mscf


def gas_schols(c: GasConeInputs):
    """Darcy critical gas rate (Mscf/d), Schols form with Hagoort's (kv/kh)^0.07."""
    q_res = (c.drho * c.k * c.krg * (c.h ** 2 - c.hp ** 2) / (2049 * c.mu_g)) \
        * (0.432 + np.pi / np.log(c.re / c.rw)) * (c.h / c.re) ** 0.14 * c.kv_kh ** 0.07
    return q_res / c.bg_rb_mscf


def non_darcy_correct(q_darcy, ratio):
    """Critical rate with turbulence (Smulski 2012): ratio = turbulent / laminar IPR coefficient, 1/(Mscf/d)."""
    if not ratio or ratio <= 0:
        return q_darcy
    return (np.sqrt(1 + 4 * ratio * q_darcy) - 1) / (2 * ratio)


def gas_critical_rate(c: GasConeInputs, ratio=None):
    """Design critical gas rate (Mscf/d): the lower of the two Darcy forms, then the non-Darcy correction."""
    return non_darcy_correct(min(gas_meyer_garder(c), gas_schols(c)), ratio)


def ipr_turbulence_ratio(curve):
    """turbulent / laminar coefficient ratio (1/(Mscf/d)) of a gas IPR curve, None if the model has none."""
    p = getattr(curve, "params", {}) or {}
    if "a (laminar)" in p and "b (turbulent)" in p:
        lam, tur = p["a (laminar)"], p["b (turbulent)"]
    elif "a (psi^2/(Mscf/d)^2)" in p and "b (psi^2/(Mscf/d))" in p:
        tur, lam = p["a (psi^2/(Mscf/d)^2)"], p["b (psi^2/(Mscf/d))"]
    else:
        return None
    return tur / lam if lam > 0 else None
