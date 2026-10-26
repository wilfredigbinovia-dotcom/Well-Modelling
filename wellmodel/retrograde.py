"""PROSPER's black-oil model for retrograde condensates (Fluid = Retrograde Condensate, Method = Black Oil).

PROSPER does not publish this model. Its PVT tables for Tutorial 38 and five variations (25 temperatures x 25
pressures, 70-300 F, 0-7000 psig) show how it is built:

Dew point      pd(T) = pd(Tres) [1 - 0.002474 dT - 1.3e-5 dT^2], dT = T - Tres  (psia; within 0.01 psi for both
               fluids PROSPER was run with: Tres 300 F / 4596.83 psig and 250 F / 4000 psig)
Dropout        liquid condensed out of the gas, L (STB per MMscf of gas), zero at and above the dew point:
                   L = CGR F(T) (y/ym)(2 - y/ym),   y = 1 - p/pd(T)  (psia)
                   F(T) = 1 - (1 - Fres)(T - 60)/(Tres - 60)   (all the condensate drops at 60 F)
                   Fres = 3.8787 (Tres + 459.67) / pd(Tres)        (exact for the six runs)
                   1 - ym = 0.31888 (TR/759.67)^-1.2625 (CGR/134.536)^0.3146 (gamma_o/0.80829)^-1.283
               CGR = 1e6 / total GOR. Within 0.002 STB/MMscf for each input changed alone, 0.15 with all four.
Gas phase      separator gas carrying the condensate still vaporised, Rv = CGR - L: gravity
                   (gamma_g + 4584 gamma_o Rv/1e6)/(1 + V Rv/1e6),   V = 133,000 gamma_o / M_o (Cragoe M_o)
               density from that gravity and Z (within 0.06 %); Lee et al viscosity (within 0.23 %); Z from
               Dranchuk-Purvis-Robinson with Standing's condensate-gas pseudo-criticals of the separator gas for all
               states (exact without impurities; with CO2/N2 PROSPER's correction was not found - hydrocarbon
               gravity plus Wichert-Aziz, Z within 0.32 %).
Liquid phase   saturated oil by Glaso (solution GOR, FVF, density from mass balance - within 0.01 %) with the
               separator gas gravity and the condensate gravity. PROSPER's condensate viscosity and gas-liquid
               interfacial tension did not match any published correlation tried: Beal/Chew-Connally viscosity
               (median -6 %, up to 22 %) and the oil-well surface tension are used.

Six PROSPER runs: Tutorial 38 (43.56 API, CGR 134.5, 4596.83 psig at 300 F), with the reservoir temperature
(250 F), dew point (4000 psig), separator GOR (10,000) and condensate gravity (50 API) changed one at a time, and all
four together. Fres depends on the reservoir temperature and dew point only; ym on the reservoir temperature, CGR and
condensate gravity only (a power-law fit - its true form is not known). In the tubing the liquid rate is q L Bo and the gas rate follows from mass
conservation (stock-tank mass less the liquid's), as PROSPER's traverse shows (its total mass flow is constant).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import blackoil as bo

G_PSIA = 14.696
DEW_T = (-0.0024740, -1.3e-5)     # pd(T)/pd(Tres) = 1 + a dT + b dT^2,  dT = T - Tres (two fluids: same constants)
DROP_FRES_K = 3.8787              # max dropout at Tres / CGR = K (Tres + 459.67) / pd(Tres)   (psia/R)
DROP_XM = (0.31888, -1.2625, 0.3146, -1.283)   # p/pd at the max dropout = a (TR/759.67)^b (CGR/134.536)^c (go/0.80829)^d


def drop_shape(inp):
    """(ym, F(Tres)) of the dropout curve - from six PROSPER runs (Tutorial 38 and its inputs changed one at a time:
    reservoir temperature 250 F, dew point 4000 psig, separator GOR 10,000, 50 API, and all four together)."""
    a, b, c, d = DROP_XM
    xm = a * ((inp.t_res + 459.67) / 759.67) ** b * (inp.cgr / 134.536) ** c * (inp.gamma_o / 0.80829) ** d
    fres = DROP_FRES_K * (inp.t_res + 459.67) / max(inp.p_dew, 1.0)
    ym = 1 - xm if inp.drop_ym is None else inp.drop_ym
    return float(np.clip(ym, 0.05, 1.0)), float(np.clip(fres if inp.drop_fres is None else inp.drop_fres, 0.0, 1.0))
BG_K = 0.028315                   # Bg = BG_K Z T / p  (ft3/scf, R, psia) - PROSPER's constant for this model
RB = 5.614583


@dataclass
class RetroInputs:
    sep_gor: float                # scf/STB
    sep_sg: float
    tank_gor: float
    tank_sg: float
    api: float
    p_dew: float                  # dew point at reservoir temperature, psia
    t_res: float                  # F
    p_res: float                  # psia
    wgr: float = 0.0              # STB/MMscf
    salinity_ppm: float = 0.0
    h2s: float = 0.0
    co2: float = 0.0
    n2: float = 0.0
    drop_ym: float | None = None    # dropout shape overrides (None: drop_shape)
    drop_fres: float | None = None

    @property
    def gor(self):
        return self.sep_gor + self.tank_gor

    @property
    def gas_sg(self):
        return (self.sep_gor * self.sep_sg + self.tank_gor * self.tank_sg) / max(self.gor, 1e-9)

    @property
    def cgr(self):
        return 1e6 / max(self.gor, 1e-9)

    @property
    def gamma_o(self):
        return 141.5 / (131.5 + self.api)

    @property
    def mo(self):
        return 6084.0 / max(self.api - 5.9, 1.0)

    @property
    def veq(self):
        """scf of vapour per STB of condensate."""
        return 133000.0 * self.gamma_o / self.mo

    def bo_inputs(self, sg=None):
        return bo.PVTInputs(rsb=1.0, api=self.api, gas_sg=self.gas_sg if sg is None else sg,
                            salinity_ppm=self.salinity_ppm, h2s=self.h2s, co2=self.co2, n2=self.n2)

    def check(self):
        e = []
        if self.sep_gor <= 0: e.append("Separator GOR must be positive.")
        if not 0.55 <= self.sep_sg <= 2.0: e.append("Separator gas gravity should be between 0.55 and 2.0.")
        if not 5 <= self.api <= 90: e.append("Condensate gravity should be between 5 and 90 API.")
        if self.p_dew <= G_PSIA: e.append("Enter the dew point at reservoir temperature.")
        if self.t_res <= 60: e.append("Reservoir temperature must be above 60 F.")
        return e


def dew_point(inp: RetroInputs, t):
    a, b = DEW_T
    dt = np.asarray(t, float) - inp.t_res
    return inp.p_dew * (1 + a * dt + b * dt * dt)


def dropout(inp: RetroInputs, p, t):
    """Condensed liquid, STB per MMscf of gas (0 at and above the dew point)."""
    p, t = np.asarray(p, float), np.asarray(t, float)
    y = np.clip(1 - p / dew_point(inp, t), 0.0, 1.0)
    ym, fres = drop_shape(inp)
    f = np.clip(1 - (1 - fres) * (t - 60.0) / max(inp.t_res - 60.0, 1e-6), 0.0, 1.0)
    return np.maximum(inp.cgr * f * (y / ym) * (2 - y / ym), 0.0)


def _pseudocritical(inp: RetroInputs):
    """Standing's condensate-gas pseudo-criticals of the separator gas (R, psia) - PROSPER's exactly without
    impurities (Tutorial 38 run with CO2 = N2 = 0). With impurities: the hydrocarbon gravity and the Wichert-Aziz
    correction (PROSPER's own treatment not found; Z within 0.32 % for 2.33 % CO2, 0.31 % N2)."""
    yn, yc, yh = inp.n2 / 100, inp.co2 / 100, inp.h2s / 100
    y = yn + yc + yh
    g = (inp.sep_sg - 0.9672 * yn - 1.5196 * yc - 1.1767 * yh) / max(1 - y, 1e-6) if y > 0 else inp.sep_sg
    tpc, ppc = 187.0 + 330.0 * g - 71.5 * g * g, 706.0 - 51.7 * g - 11.1 * g * g
    a, b = yc + yh, yh
    eps = 120 * (a ** 0.9 - a ** 1.6) + 15 * (b ** 0.5 - b ** 4)
    t2 = tpc - eps
    return t2, ppc * t2 / (tpc + b * (1 - b) * eps)


def props(inp: RetroInputs, p, t):
    """Phase properties at p (psia), t (F): dict of arrays."""
    p = np.maximum(np.asarray(p, float), G_PSIA)
    t = np.asarray(t, float)
    p, t = np.broadcast_arrays(p, t)
    pd_ = dew_point(inp, t)
    L = dropout(inp, p, t)
    rv = inp.cgr - L
    go = inp.gamma_o
    sg_v = (inp.gas_sg + 4584.0 * go * rv / 1e6) / (1 + inp.veq * rv / 1e6)
    tr = t + 459.67
    tpc, ppc = _pseudocritical(inp)
    z = bo.z_dpr(tr / tpc, p / ppc)
    m = 28.97 * sg_v
    rho_g = m * p / (z * 10.7316 * tr)
    k = (9.4 + 0.02 * m) * tr ** 1.5 / (209 + 19 * m + tr)
    x = 3.5 + 986 / tr + 0.01 * m
    mu_g = 1e-4 * k * np.exp(x * (rho_g / 62.428) ** (2.4 - 0.2 * x))
    oil = bo.PVTInputs(rsb=1.0, api=inp.api, gas_sg=inp.sep_sg)
    rs = bo.rs_raw("glaso", oil, p, t)
    bo_ = bo.bo_sat("glaso", oil, rs, t)
    rho_o = (62.428 * go + 0.0136 * rs * inp.sep_sg) / bo_
    mu_o = bo.visc_live("beal", bo.visc_dead("beal", oil, t), rs)
    from .vlp import sigma_oil_prosper
    ift = sigma_oil_prosper(inp.api, t, rs)
    return dict(p_dew=pd_, dropout=L, rv=rv, gas_sg=sg_v, z=z, bg=BG_K * z * tr / p, rho_g=rho_g, mu_g=mu_g,
                rs=rs, bo=bo_, rho_o=rho_o, mu_o=mu_o, ift=ift)


def pvt_table(inp: RetroInputs, temps, pressures_psig) -> pd.DataFrame:
    """PROSPER-style PVT table (pressures psig)."""
    rows = []
    for t in temps:
        pp = np.asarray(pressures_psig, float) + G_PSIA
        r = props(inp, pp, np.full(pp.shape, float(t)))
        two = r["dropout"] > 0
        bw, rho_w, mu_w = bo.water_props(inp.bo_inputs(), pp, np.full(pp.shape, float(t)))
        for i, pg in enumerate(pressures_psig):
            rows.append({"Temperature (deg F)": t, "Pressure (psig)": pg, "Dew point (psig)": r["p_dew"][i] - G_PSIA,
                         "Reservoir CGR (STB/MMscf)": r["dropout"][i], "Gas density (lb/ft3)": r["rho_g"][i],
                         "Gas viscosity (cp)": r["mu_g"][i], "Gas FVF (ft3/scf)": r["bg"][i], "Z factor": r["z"][i],
                         "Oil density (lb/ft3)": r["rho_o"][i] if two[i] else np.nan,
                         "Oil viscosity (cp)": r["mu_o"][i] if two[i] else np.nan,
                         "Oil FVF (RB/STB)": r["bo"][i] if two[i] else np.nan,
                         "Water density (lb/ft3)": float(rho_w[i]), "Water viscosity (cp)": float(mu_w[i]),
                         "Water FVF (RB/STB)": float(bw[i])})
    return pd.DataFrame(rows)
