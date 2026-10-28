"""Multiphase pseudo-pressure inflow (oil below the bubble point, gas condensate below the dew point).

Definition (PROSPER help, "Multiphase Pseudo Pressure"):

    m(p) = integral_0^p ( rho_o ko/mu_o + rho_g kg/mu_g + rho_w kw/mu_w ) dp
         = integral_0^p (ko/mu_o) F dp,   F = rho_o + (GOR_free Bg / Bo) rho_g + (WOR Bw / Bo) rho_w

i.e. the total mass mobility, which linearises the radial multiphase diffusivity equation.
The producing ratios fix kg/ko at each pressure but not ko and kg separately; relative
permeability curves (Corey here) close the problem. The steady-state assumption of
Fevang & Whitson (SPE 30714, 1995) is used: the flowing mixture everywhere has the producing
composition, so

    black oil:  krg/kro = (Rp - Rs) Bg mu_g / (Bo mu_o)
    EOS:        krg/kro = (V/L) (mu_g / mu_o) (rho~_o / rho~_g)     (molar flash split of the wellstream)

Mass rate (lb/d) = 5.615 k h / (141.2 (D_dietz + S + D q)) * m-difference, converted to surface
rate with the mass per STB (oil) or per Mscf (gas) of the producing stream. For consistent black-oil
properties the oil form reduces to the classical Evinger-Muskat integral of kro / (mu_o Bo).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .inflow import IPRCurve, _grid, dietz_term

RB = 5.615  # ft3 per bbl


@dataclass
class Corey:
    """Two-phase gas-oil Corey relative permeabilities at connate water."""
    swc: float = 0.2
    sorg: float = 0.2
    sgc: float = 0.05
    kro_end: float = 1.0
    krg_end: float = 1.0
    no: float = 2.0
    ng: float = 2.0

    def __post_init__(self):
        if not (0 <= self.swc < 1 and 0 <= self.sorg < 1 and 0 <= self.sgc < 1):
            raise ValueError("Saturations must be between 0 and 1.")
        if self.swc + self.sorg + self.sgc >= 1:
            raise ValueError("Swc + Sorg + Sgc must be less than 1.")
        if min(self.kro_end, self.krg_end) <= 0 or min(self.no, self.ng) <= 0:
            raise ValueError("End points and Corey exponents must be positive.")

    @property
    def sg_max(self):
        return 1 - self.swc - self.sorg

    def kro(self, sg):
        so_n = np.clip((1 - self.swc - self.sorg - np.asarray(sg, float)) / (1 - self.swc - self.sorg), 0, 1)
        return self.kro_end * so_n ** self.no

    def krg(self, sg):
        sg_n = np.clip((np.asarray(sg, float) - self.sgc) / (1 - self.swc - self.sgc - self.sorg), 0, 1)
        return self.krg_end * sg_n ** self.ng

    def sg_from_ratio(self, ratio):
        """Gas saturation giving krg/kro = ratio (0 -> no free gas, inf -> oil immobile)."""
        if ratio <= 0:
            return 0.0
        if not np.isfinite(ratio):
            return 1 - self.swc
        lo, hi = self.sgc + 1e-9, self.sg_max - 1e-9
        f = lambda s: np.log(self.krg(s) + 1e-300) - np.log(self.kro(s) + 1e-300) - np.log(ratio)
        if f(lo) > 0:
            return lo
        if f(hi) < 0:
            return hi
        return brentq(f, lo, hi, xtol=1e-10)

    def table(self, n=41) -> pd.DataFrame:
        sg = np.linspace(0, 1 - self.swc, n)
        return pd.DataFrame({"Sg": sg, "So": 1 - self.swc - sg, "krg": self.krg(sg), "kro": self.kro(sg)})


def _split(rel: Corey, ratio):
    """Return (Sg, kro, krg) for a krg/kro ratio; single-phase limits use the end points."""
    sg = rel.sg_from_ratio(ratio)
    if ratio <= 0:
        return 0.0, rel.kro_end, 0.0
    if not np.isfinite(ratio):
        return 1 - rel.swc, 0.0, rel.krg_end
    return sg, float(rel.kro(sg)), float(rel.krg(sg))


# ------------------------------------------------------------ integrand tables
def blackoil_table(props, pressures, rp, wor, rel: Corey) -> pd.DataFrame:
    """Oil well, black-oil PVT. props(p) -> DataFrame with the BlackOilModel.table columns.
    rp = producing GOR (scf/STB), wor = water-oil ratio (STB/STB). Integrand in STB/d per (md ft psi)/141.2."""
    t = props(np.asarray(pressures, float))
    p = t["Pressure (psia)"].to_numpy()
    rs, bo, mu_o, rho_o = (t[c].to_numpy() for c in ("Solution GOR (scf/STB)", "Oil FVF (RB/STB)",
                                                       "Oil viscosity (cp)", "Oil density (lb/ft3)"))
    bg = t["Gas FVF (ft3/scf)"].to_numpy() / RB            # RB/scf
    mu_g = t["Gas viscosity (cp)"].to_numpy()
    bw, rho_w = t["Water FVF (RB/STB)"].to_numpy(), t["Water density (lb/ft3)"].to_numpy()
    rho_g = t["Gas density (lb/ft3)"].to_numpy()
    free = np.maximum(rp - rs, 0.0)
    rows = []
    for i in range(len(p)):
        ratio = free[i] * bg[i] * mu_g[i] / (bo[i] * mu_o[i])
        sg, kro, krg = _split(rel, ratio)
        F = rho_o[i] + free[i] * bg[i] / bo[i] * rho_g[i] + wor * bw[i] / bo[i] * rho_w[i]
        lam = kro / mu_o[i] * F                                   # mass mobility / k, lb/ft3/cp
        mass_per_stb = RB * bo[i] * F                             # lb per STB of oil (with its gas and water)
        g = RB * lam / mass_per_stb if mass_per_stb > 0 else 0.0
        g1 = rel.kro_end / (mu_o[i] * bo[i])                      # no free-gas reduction
        rows.append((p[i], sg, 1 - rel.swc - sg, kro, krg, ratio, F, g, g1))
    return pd.DataFrame(rows, columns=["Pressure (psia)", "Sg", "So", "kro", "krg", "krg/kro", "F (lb/ft3)",
                                       "Integrand", "Integrand single-phase"])


def eos_table(fluid, t_f, pressures, rel: Corey, surface_per_lbmol, wgr_bbl_per_lbmol=0.0,
              rho_w=62.4, mu_w=0.5) -> pd.DataFrame:
    """EOS fluid (wellstream = reservoir composition). surface_per_lbmol converts lbmol of wellstream
    to the surface rate unit (Mscf or STB). Integrand in surface units/d per (md ft psi)/141.2."""
    mwz = float(fluid.z @ fluid.mw)
    rows = []
    for p in np.asarray(pressures, float):
        r = fluid.flash(p, t_f)
        if r.phases == "2P":
            L, V = r.liquid, r.vapour
            ratio = (r.beta / (1 - r.beta)) * (V.viscosity / L.viscosity) * (L.molar_volume / V.molar_volume) \
                if r.beta < 1 else np.inf
            sg, kro, krg = _split(rel, ratio)
            lam_hc = kro * L.density / L.viscosity + krg * V.density / V.viscosity
            mu_o, mu_g = L.viscosity, V.viscosity
        else:
            ph = r.vapour if r.phases == "V" else r.liquid
            gas_like = r.phases == "V"
            ratio = np.inf if gas_like else 0.0
            sg, kro, krg = _split(rel, ratio)
            lam_hc = (krg if gas_like else kro) * ph.density / ph.viscosity
            mu_o = np.nan if gas_like else ph.viscosity
            mu_g = ph.viscosity if gas_like else np.nan
        # water flows in the producing ratio: same share of the mass flow everywhere (PDF F term)
        w_mass = wgr_bbl_per_lbmol * RB * rho_w                # lb water per lbmol hydrocarbon
        lam = lam_hc * (1 + w_mass / mwz)
        g = RB * lam / (mwz + w_mass) * surface_per_lbmol
        single = fluid.phase_props(fluid.z, p, t_f)
        end = rel.krg_end if (r.phases == "V" or (r.phases == "2P" and r.beta > 0.5)) else rel.kro_end
        g1 = RB * end * single.density / single.viscosity / mwz * surface_per_lbmol
        rows.append((p, sg, 1 - rel.swc - sg, kro, krg, ratio, mu_o, mu_g, g, g1,
                     {"2P": "Two-phase", "V": "Gas", "L": "Liquid"}[r.phases]))
    return pd.DataFrame(rows, columns=["Pressure (psia)", "Sg", "So", "kro", "krg", "krg/kro",
                                       "Oil viscosity (cp)", "Gas viscosity (cp)", "Integrand",
                                       "Integrand single-phase", "Phases"])


def gas_correlation_table(mu_z, pressures, t_f, rel: Corey) -> pd.DataFrame:
    """Dry/wet gas from correlations (no liquid dropout): single-phase gas at krg end point.
    Integrand = (141.2/1422) 2 p krg / (mu Z T) so that the IPR matches the gas pseudo-pressure form."""
    p = np.asarray(pressures, float)
    mu, z = mu_z(p)
    g = 141.2 / 1422 * 2 * p * rel.krg_end / (mu * z * (t_f + 459.67))
    return pd.DataFrame({"Pressure (psia)": p, "Sg": 1 - rel.swc, "So": 0.0, "kro": 0.0, "krg": rel.krg_end,
                         "krg/kro": np.inf, "Integrand": g, "Integrand single-phase": g})


def pressure_grid(pr, psat=None, n=50):
    p = np.linspace(14.7, pr, n)
    if psat and 14.7 < psat < pr:
        p = np.unique(np.concatenate([p, [psat * (1 - 1e-4), psat * (1 + 1e-4)],
                                      np.linspace(max(14.7, psat * 0.85), psat, 8)]))
    return p


def _cum(p, g):
    return np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(p))])


# ----------------------------------------------------------------- the IPR
def multiphase_pseudo_pressure(fluid, pr, k, h, area_acres, ca, rw, skin, table: pd.DataFrame,
                               d_nd=0.0, psat=None, wor=0.0, params=None, notes=None) -> IPRCurve:
    """IPR from an integrand table (blackoil_table / eos_table). fluid 'oil' (STB/d) or 'gas' (Mscf/d).
    d_nd: non-Darcy coefficient (1/(surface rate unit)), applied as rate-dependent skin.
    Oil curves are returned as liquid rate (oil x (1 + WOR)), like the other oil IPRs."""
    p = table["Pressure (psia)"].to_numpy()
    if p.max() < pr * (1 - 1e-6):
        raise ValueError("Integrand table must reach the reservoir pressure.")
    m = _cum(p, table["Integrand"].to_numpy())
    m1 = _cum(p, table["Integrand single-phase"].to_numpy())
    table = table.copy()
    table["m(p)"] = m
    D = dietz_term(area_acres, ca, rw) + skin
    if D <= 0:
        raise ValueError("Dietz term plus skin must be positive.")
    c = k * h / 141.2
    m_pr, m1_pr = np.interp(pr, p, m), np.interp(pr, p, m1)

    def solve(dm):
        dm = np.maximum(dm, 0.0)
        if d_nd <= 0:
            return c * dm / D
        return (-D + np.sqrt(D * D + 4 * d_nd * c * dm)) / (2 * d_nd)

    liq = (1 + wor) if fluid == "oil" else 1.0

    def rate(pwf):
        return liq * solve(m_pr - np.interp(np.clip(pwf, p[0], pr), p, m))

    aof1 = liq * solve(m1_pr - m1[0])
    curve = IPRCurve("Multiphase Pseudo Pressure", fluid, pr, _grid(pr), None, {}, notes or [], None, rate)
    curve.q = np.maximum(rate(curve.pwf), 0.0)
    aof = curve.aof
    unit = "Mscf/d" if fluid == "gas" else "STB/d"
    out = {"Two-phase productivity loss (%)": 100 * (1 - aof / aof1) if aof1 > 0 else 0.0,
           f"AOF single-phase ({unit})": float(aof1)}
    if psat:
        out["Saturation pressure (psia)"] = float(psat)
    out.update(params or {})
    out.update({"Dietz term": dietz_term(area_acres, ca, rw), "Skin": skin})
    if d_nd > 0:
        out["D (1/(Mscf/d))"] = d_nd
    curve.params = out
    curve.mpp_table = table
    return curve
