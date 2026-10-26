"""Gas hydrate formation curve (PROSPER PVT | Phase envelope | Calculate Hydrate, Tutorial 43) and the water's salts
as inhibitors.

van der Waals - Platteeuw solid solution theory with the Munck, Skjold-Jorgensen & Rasmussen (1988) Langmuir
constants for structures I and II; fugacities of the hydrate formers from the Peng-Robinson / SRK equation of state of
the PRP composition (flashed: the vapour, or the single phase); water activity of the brine from Pitzer's
single-salt osmotic coefficients mixed at the total ionic strength. At equilibrium

    - sum_m nu_m ln(1 - sum_k theta_km) = dmu0/(R T0) - int_T0^T dh/(R T^2) dT + dv p/(R T) - ln a_w

with dh = dh0 - 6011 + dcp (T - T0) (empty lattice minus liquid water). The reference properties were refitted to
PROSPER's hydrate curves of the user's T43 file (Water 1 and Water 2, NaCl / CaCl2 / KBr brines, 35-98 F, 117 -
22,390 psig): dmu0 1380 / 914 J/mol (sI / sII), dv 3.25 / 6.51 cm3/mol, dcp -49.3 J/mol/K, with Munck's dh0
(1151 / 808 J/mol) and Langmuir constants - within 3.4 % of PROSPER's pressures at all 43 points (rms 1.3 %).
PROSPER labels its model "Hydrafact modified cubic"; this is not that model, it is calibrated to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import eos as E

R = 8.314462618
T0 = 273.15
PSI_PER_ATM = 14.69595
PA_PER_PSI = 6894.757

# Munck et al. (1988): C = A / T exp(B / T), A in K/atm (x 1e-3 here), per cavity (sI small, sI large, sII small,
# sII large)
LANGMUIR = {
    "C1": ((0.7228, 3187.0), (23.35, 2653.0), (0.2207, 3453.0), (100.0, 1916.0)),
    "C2": ((0.0, 0.0), (3.039, 3861.0), (0.0, 0.0), (240.0, 2967.0)),
    "C3": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (5.455, 4638.0)),
    "IC4": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (189.3, 3800.0)),
    "NC4": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (30.51, 3699.0)),
    "N2": ((1.617, 2905.0), (6.078, 2431.0), (0.1742, 3082.0), (18.0, 1728.0)),
    "CO2": ((0.2474, 3410.0), (42.46, 2813.0), (0.0845, 3615.0), (851.0, 2025.0)),
    "H2S": ((0.025, 4568.0), (16.34, 3737.0), (0.0034, 5157.0), (87.2, 2633.0)),
}
ALIASES = {"METHANE": "C1", "ETHANE": "C2", "PROPANE": "C3", "ISOBUTANE": "IC4", "I-C4": "IC4", "IC4": "IC4",
           "N-C4": "NC4", "NC4": "NC4", "C4": "NC4", "BUTANE": "NC4", "NITROGEN": "N2", "CO2": "CO2",
           "H2S": "H2S", "N2": "N2", "C1": "C1", "C2": "C2", "C3": "C3"}
# (cavity indices, nu per water molecule, dmu0 J/mol, dh0 (lattice - ice) J/mol, dv (lattice - liquid) m3/mol)
STRUCTURES = {
    "sI": ((0, 1), (1 / 23, 3 / 23), 1380.3609, 1151.0, 3.2517e-6),
    "sII": ((2, 3), (2 / 17, 1 / 17), 914.1843, 808.0, 6.5137e-6),
}
DCP = -49.2568           # J/mol/K, lattice - liquid water (constant)
DH_FUSION = 6011.0       # J/mol, ice -> liquid at T0

# Pitzer parameters at 25 C: MW, nu+, nu-, z+, |z-|, beta0, beta1, C_phi
SALTS = {
    "NaCl": (58.443, 1, 1, 1, 1, 0.0765, 0.2664, 0.00127), "KCl": (74.551, 1, 1, 1, 1, 0.04835, 0.2122, -0.00084),
    "CaCl2": (110.98, 1, 2, 2, 1, 0.3159, 1.614, -0.00034), "Na2SO4": (142.04, 2, 1, 1, 2, 0.01958, 1.113, 0.00497),
    "NaF": (41.988, 1, 1, 1, 1, 0.0215, 0.2107, 0.0), "KBr": (119.0, 1, 1, 1, 1, 0.0569, 0.2212, -0.0018),
    "MgCl2": (95.211, 1, 2, 2, 1, 0.35235, 1.6815, 0.00519), "SrCl2": (158.53, 1, 2, 2, 1, 0.28575, 1.66717, -0.0013),
    "BaCl2": (208.23, 1, 2, 2, 1, 0.2628, 1.49625, -0.01938), "NaBr": (102.89, 1, 1, 1, 1, 0.0973, 0.2791, 0.00116),
    "HCOONa": (68.007, 1, 1, 1, 1, 0.082, 0.2872, -0.0052), "HCOOK": (84.116, 1, 1, 1, 1, 0.1263, 0.3326, -0.0053),
    "K2CO3": (138.21, 2, 1, 1, 2, 0.1288, 1.433, 0.0005), "CaBr2": (199.89, 1, 2, 2, 1, 0.3816, 1.613, -0.00257),
}
A_PHI = 0.38             # Debye-Hueckel osmotic coefficient near 5-15 C
MW_WATER = 0.018015      # kg/mol


def ln_water_activity(salts_wt_pct: dict | None = None, salinity_ppm: float | None = None) -> float:
    """ln a_w of a brine: salts as weight % of the solution ({"NaCl": 2.15, ...}), or a salinity in ppm taken as
    NaCl. Pitzer single-salt osmotic coefficients at the total ionic strength, weighted by each salt's ions."""
    salts = dict(salts_wt_pct or {})
    if not salts and salinity_ppm:
        salts = {"NaCl": float(salinity_ppm) / 1e4}
    salts = {k: float(v) for k, v in salts.items() if v and float(v) > 0}
    if not salts:
        return 0.0
    unknown = [k for k in salts if k not in SALTS]
    if unknown:
        raise ValueError(f"Unknown salt(s): {', '.join(unknown)}. Known: {', '.join(SALTS)}")
    w_w = 1 - sum(salts.values()) / 100
    if w_w <= 0:
        raise ValueError("The salt weight percentages add to 100 % or more.")
    m = {k: v * 10 / SALTS[k][0] / w_w for k, v in salts.items()}       # mol/kg water
    ionic = sum(0.5 * mm * (SALTS[k][1] * SALTS[k][3] ** 2 + SALTS[k][2] * SALTS[k][4] ** 2) for k, mm in m.items())
    sq = np.sqrt(ionic)
    tot = 0.0
    for k, mm in m.items():
        _, npos, nneg, zp, zn, b0, b1, c = SALTS[k]
        nu = npos + nneg
        me = ionic / (0.5 * (npos * zp ** 2 + nneg * zn ** 2))          # molality giving the mixture's I
        phi = (1 - zp * zn * A_PHI * sq / (1 + 1.2 * sq) + me * 2 * npos * nneg / nu * (b0 + b1 * np.exp(-2 * sq))
               + me ** 2 * 2 * (npos * nneg) ** 1.5 / nu * c)
        tot += nu * mm * phi
    return float(-MW_WATER * tot)


def salinity_of(salts_wt_pct: dict) -> float:
    """Total dissolved salts (ppm) of a composition in weight %."""
    return float(sum(float(v) for v in salts_wt_pct.values()) * 1e4)


def former_name(name) -> str | None:
    k = str(name).upper().replace(" ", "")
    k = ALIASES.get(k, k)
    return k if k in LANGMUIR else None


class HydrateModel:
    """Hydrate formation pressure / temperature of an EOS fluid in contact with a brine."""

    def __init__(self, fluid: E.EOSFluid, salts_wt_pct: dict | None = None, salinity_ppm: float | None = None):
        self.f = fluid
        self.formers = [former_name(n) for n in fluid.names]
        if not any(self.formers):
            raise ValueError("No hydrate formers (C1, C2, C3, i/nC4, N2, CO2, H2S) in the composition.")
        self.ln_aw = ln_water_activity(salts_wt_pct, salinity_ppm)
        self._cache = {}

    def fugacities(self, p_psia, t_f):
        """Fugacities (atm) of the hydrate formers at (p, T)."""
        key = (round(float(p_psia), 6), round(float(t_f), 6))
        if key in self._cache:
            return self._cache[key]
        fl = self.f.flash(float(p_psia), float(t_f))
        if fl.phases == "2P":
            y, ph = fl.y, "V"
        else:
            y, ph = fl.x if fl.phases == "L" else fl.y, None
        lnphi, _ = self.f._lnphi(y, float(p_psia), float(t_f) + E.RANKINE, phase=ph)
        fug = y * np.exp(lnphi) * float(p_psia) / PSI_PER_ATM
        out = {n: v for n, v in zip(self.formers, fug) if n is not None}
        self._cache[key] = out
        return out

    def _residual(self, st, p_psia, t_f):
        cav, nus, dmu0, dh0, dv = STRUCTURES[st]
        tk = (t_f - 32) / 1.8 + T0
        fug = self.fugacities(p_psia, t_f)
        h = 0.0
        for c, nu in zip(cav, nus):
            cf = 0.0
            for name, f in fug.items():
                a, b = LANGMUIR[name][c]
                if a:
                    cf += a * 1e-3 / tk * np.exp(b / tk) * f
            h += nu * np.log1p(cf)
        dhl = dh0 - DH_FUSION
        integral = dhl * (1 / T0 - 1 / tk) / R + DCP * (np.log(tk / T0) + T0 / tk - 1) / R
        liq = dmu0 / (R * T0) - integral + dv * p_psia * PA_PER_PSI / (R * tk) - self.ln_aw
        return h - liq

    def pressure(self, t_f, p_min=15.0, p_max=40000.0):
        """Hydrate formation pressure (psia) at t_f (F): the lower of the two structures; nan if none in range."""
        best = np.nan
        for st in STRUCTURES:
            g = lambda lp: self._residual(st, np.exp(lp), t_f)
            try:
                lp = brentq(g, np.log(p_min), np.log(p_max), xtol=1e-5)
            except ValueError:
                continue
            p = float(np.exp(lp))
            best = p if not np.isfinite(best) else min(best, p)
        return best

    def structure(self, t_f):
        out = {}
        for st in STRUCTURES:
            g = lambda lp: self._residual(st, np.exp(lp), t_f)
            try:
                out[st] = float(np.exp(brentq(g, np.log(15.0), np.log(40000.0), xtol=1e-5)))
            except ValueError:
                pass
        return min(out, key=out.get) if out else ""

    def curve(self, temps_f) -> pd.DataFrame:
        recs = []
        for t in temps_f:
            p = self.pressure(float(t))
            if np.isfinite(p):
                recs.append({"Temperature (°F)": float(t), "Pressure (psia)": p, "Pressure (psig)": p - 14.696,
                             "Structure": self.structure(float(t))})
        return pd.DataFrame(recs)


def table_curve(df: pd.DataFrame | None, p_col="Pressure (psig)", t_col="Temperature (°F)", gauge=True):
    """Hydrate curve ((psia...), (F...)) from a table (PVT warnings data); None if fewer than two rows."""
    if df is None:
        return None
    d = df[[p_col, t_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 2:
        return None
    d = d.sort_values(p_col)
    return (tuple(d[p_col].to_numpy(float) + (14.696 if gauge else 0.0)), tuple(d[t_col].to_numpy(float)))
