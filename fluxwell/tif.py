"""Thermally Induced Fracture (TIF) injectivity for water injectors - PROSPER's IPR model of that name.

PROSPER cites Detienne et al. (SPE 30777, 1998), the Lauwerier reservoir temperature profile (SPE 7964) and Perkins &
Gonzalez's thermo-elastic stress coefficient (SPE 11332), but does not publish its implementation. Eight PROSPER
system runs on Tutorial 03a (base; thermal expansion 1e-5; initial stress 6000 psig; mean historical rate 24,000;
time 100 days; area 500 acres; sweep 0.5; skin 0 - each one input changed) show that its IPR is three regimes, each
with a simple form in the bottom-hole (sandface) temperature T of the injected water. With c = 141.2 q / (k h),
mu_c B_c the water at T, mu_h B_h the water at reservoir temperature, mu_o B_o the reservoir fluid:

1. Radial, unfractured - three concentric zones (Detienne):
       Pwf1 = Pr + c [ mu_c B_c (ln(r_c/rw) + S) + K_H mu_h B_h ln(r_f/r_c) + K_O mu_o B_o PSS(r_f) ]
   cooled radius r_c from the heat balance of the injected volume (mean historical rate x time) against the
   reservoir heat capacity; flooded radius r_f from the volume, porosity and sweep; PSS(r) = 0.5 ln(4A/(1.781 CA r^2)).
   K_H = 2.27 and K_O = 2.48 are calibrated (PROSPER's flooded and outer zones are about 2.3-2.5 times less
   mobile than the plain viscosities give); the cold zone needs no factor.
2. Fracture opening (Pwf1 above the fracturing pressure) - the stress at the well with the thermo-elastic term:
       Pwf2 = sigma_ini + C0 + F_T Fh aE/(1-v) (T - T_res) + c [ A_Q + (A_MU + A_S S) mu_c B_c ]
   Fh = 1/2 + 1/2 / (1 + 1.45 x^0.9 + 0.35 x^2), x = h / (2 r_c) (Perkins & Gonzalez, circular cooled region) - the
   time and historical-rate runs move the opening pressure by exactly this (-118 and -57 psi at 3000 STB/d);
   C0 = -3.9 psi, F_T = 0.951, A_Q = 0.871, A_MU = 1.106, A_S = 0.296.
3. Fracture fully developed - the skin acts with the cold-water viscosity, the reservoir adds little:
       Pwf3 = Pr + c [ (1.165 S - 0.598) mu_c B_c + 1.057 mu_h B_h ]
   (the Tutorial 03a runs reach this regime only at skin 3, which fixes the sum but not the split between the skin
   and the other terms; the split comes from PROSPER's Adjust Skin on Tutorial 21 - skins 4.669 / 6.561 / 8.850 at
   8467 / 5897 / 4456 STB/d and 99 / 112 / 124 F - and the fit covers both tutorials within 7.3 psi)
The IPR is Pwf1 while Pwf1 <= Pwf2, otherwise max(Pwf2, Pwf3). PROSPER's "dP total skin" is Pwf(S) - Pwf(S = 0).

Against the eight runs: regime 2 within 3.5 psi (27 points, all inputs), regime 3 within 7.3 psi (4,500-28,500 STB/d, plus T21),
regime 1 within 12 psi; the crossover rates follow. The coefficients are calibrated on this tutorial's inputs
(k 50 md, h 100 ft, E 3e6 psi, v 0.2, Biot 0.7, the tutorial's heat properties); their dependence on inputs not
varied in those runs is not known. In the IPR tab the temperature is the injected fluid temperature entered; in the
system calculation it is the sandface temperature the tubing calculation gives at each rate, as PROSPER does."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RB = 5.614583

K_H, K_O = 2.27357, 2.47640
C0, F_T, A_Q, A_MU, A_S = -3.949, 0.95051, 0.87116, 1.10637, 0.29641
B3_S, B3_H, B3_C = 1.16516, 1.05733, -0.59775


@dataclass
class TIFInputs:
    pr: float                     # reservoir pressure, psia
    t_res: float                  # reservoir temperature, F
    t_inj: float                  # injected fluid temperature at the sandface (IPR tab), F
    k: float                      # md
    h: float                      # ft
    area: float                   # acres
    ca: float = 31.6              # Dietz shape factor
    rw: float = 0.354             # ft
    phi: float = 0.2
    t_days: float = 20.0
    q_hist: float = 10000.0       # mean historical injection rate, STB/d
    skin: float = 0.0
    sigma_ini: float = 5000.0     # initial reservoir stress, psia
    sweep: float = 0.75
    cp_w: float = 1.0             # BTU/lb/F
    lam_r: float = 2.5            # BTU/hr/ft/F (not used by the calibrated form)
    cp_r: float = 0.25            # BTU/lb/F (overall reservoir)
    rho_r: float = 140.0          # lb/ft3 (overall reservoir)
    lam_s: float = 0.9            # surroundings (not used by the calibrated form)
    cp_s: float = 0.2
    rho_s: float = 200.0
    alpha: float = 5e-6           # thermal expansion 1/F
    biot: float = 0.7
    nu: float = 0.2
    e_mod: float = 3e6            # psi
    mu_res: float = 0.5           # reservoir fluid viscosity, cp (PVT at Pr, T_res)
    b_res: float = 1.2            # reservoir fluid FVF


class TIFModel:
    """water: injection.WaterFluid (water properties)."""

    def __init__(self, inp: TIFInputs, water):
        self.inp, self.water = inp, water
        b_h, _, m_h = water.props(inp.pr, inp.t_res)
        self.mub_h = float(np.asarray(m_h * b_h))
        _, rho_w, _ = water.props(14.696, 60.0)
        rho_w = float(np.asarray(rho_w))
        v = inp.q_hist * inp.t_days * RB
        self.r_c = float(np.sqrt(inp.rw ** 2 + v * rho_w * inp.cp_w / (np.pi * inp.h * inp.rho_r * inp.cp_r)))
        self.r_f = float(max(np.sqrt(inp.rw ** 2 + v / (np.pi * inp.h * inp.phi * inp.sweep)), self.r_c))
        self.r_e = float(np.sqrt(inp.area * 43560.0 / np.pi))
        x = inp.h / (2.0 * self.r_c)                 # Perkins & Gonzalez, circular cooled region of height h
        self.fh = 0.5 + 0.5 / (1.0 + 1.45 * x ** 0.9 + 0.35 * x * x)

    def _mub(self, p, t):
        b, _, m = self.water.props(p, t)
        return np.asarray(m * b, float)

    def regimes(self, q, t, skin=None, p_eval=None):
        """(Pwf1, Pwf2, Pwf3) psia at rates q (STB/d) and sandface temperatures t (F); water FVF at p_eval
        (default the reservoir pressure)."""
        inp = self.inp
        s = inp.skin if skin is None else skin
        q = np.atleast_1d(np.asarray(q, float))
        t = np.broadcast_to(np.asarray(t, float), q.shape)
        c = 141.2 * q / (inp.k * inp.h)
        pe = np.full(q.shape, inp.pr) if p_eval is None else np.broadcast_to(np.asarray(p_eval, float), q.shape)
        mub_c = self._mub(pe, t)
        pss = 0.5 * np.log(4 * inp.area * 43560.0 / (1.781 * inp.ca * self.r_f ** 2))
        p1 = inp.pr + c * (mub_c * (np.log(self.r_c / inp.rw) + s) + K_H * self.mub_h * np.log(self.r_f / self.r_c)
                           + K_O * inp.mu_res * inp.b_res * pss)
        th = self.fh * inp.alpha * inp.e_mod / (1 - inp.nu)
        p2 = inp.sigma_ini + C0 + F_T * th * (t - inp.t_res) + c * (A_Q + (A_MU + A_S * s) * mub_c)
        p3 = inp.pr + c * ((B3_S * s + B3_C) * mub_c + B3_H * self.mub_h)
        return p1, p2, p3

    def pwf(self, q, t, skin=None):
        p1, p2, p3 = self.regimes(q, t, skin)
        out = np.where(p1 <= p2, p1, np.maximum(p2, p3))
        p1, p2, p3 = self.regimes(q, t, skin, p_eval=out)       # water FVF at the flowing pressure (as fitted)
        out = np.where(p1 <= p2, p1, np.maximum(p2, p3))
        return out if np.ndim(q) else float(out[0])

    def regime(self, q, t):
        p1, p2, p3 = self.regimes(q, t)
        return np.where(p1 <= p2, "radial", np.where(p2 >= p3, "fracture opening", "fracture developed"))


@dataclass
class TIFIPR:
    """Injectivity for injection.solve: pwf(q, t) needs the sandface temperature (sandface_temperature = True)."""
    tm: TIFModel
    model: str = "Thermally Induced Fracture"
    fluid: str = "water_injection"
    sandface_temperature: bool = True
    params: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def pr(self):
        return self.tm.inp.pr

    @property
    def skin(self):
        return self.tm.inp.skin

    def pwf(self, q, t=None):
        return self.tm.pwf(q, self.tm.inp.t_inj if t is None else t)

    def dp_skin(self, q, t=None):
        t = self.tm.inp.t_inj if t is None else t
        return self.tm.pwf(q, t) - self.tm.pwf(q, t, skin=0.0)

    def rate(self, pwf, t=None):
        qq = np.linspace(0.0, 4 * max(self.tm.inp.q_hist, 1000.0), 400)
        pp = np.maximum.accumulate(self.pwf(qq, t))
        return np.interp(np.asarray(pwf, float), pp, qq)

    @property
    def aof(self):
        return float(4 * max(self.tm.inp.q_hist, 1000.0))

    def table(self, q, t=None) -> pd.DataFrame:
        q = np.asarray(q, float)
        t = self.tm.inp.t_inj if t is None else t
        return pd.DataFrame({"Injection rate (STB/d)": q, "Pwf (psia)": self.pwf(q, t),
                             "dP skin (psi)": self.dp_skin(q, t), "Regime": self.tm.regime(q, t)})


def tif_ipr(inp: TIFInputs, water) -> TIFIPR:
    tm = TIFModel(inp, water)
    return TIFIPR(tm, params={"Cooled radius (ft)": tm.r_c, "Flooded radius (ft)": tm.r_f,
                              "Thermo-elastic coefficient Fh": tm.fh,
                              "Drainage radius (ft)": tm.r_e, "Water mu*B at reservoir temperature": tm.mub_h})
