"""Dry and wet gas black-oil PVT (PROSPER PVT | Input Data for Dry and Wet Gas).

The condensate is assumed to drop out at the separator only: in the well it is carried as an equivalent gas
(wellstream gravity recombined from the separator gas gravity, CGR and condensate gravity), so the hydrocarbon is a
single gas phase and free water the only liquid.

Methods (checked against PROSPER's PVT tables for Tutorial 02, 60-250 F x 15-5300 psig, 400 points):
  wellstream gravity  g_w = (R g_g + 4584 g_o) / (R + 132800 g_o / M_o), R = 1e6 / CGR, M_o Cragoe
                      (0.58 separator gas + 1 STB/MMscf of 50 API condensate: 0.58313; PROSPER's densities imply 0.58322)
  Z factor            Sutton pseudo-criticals, Wichert-Aziz correction for CO2/H2S, Dranchuk-Abou-Kassem
                      (PROSPER: "Standing-Katz charts with Wichert-Aziz"; within 0.8 %, mean -0.09 %)
  gas viscosity       Lee, Gonzalez & Eakin (within 0.01 % of PROSPER at its density) or Carr, Kobayashi & Burrows
                      (Dempsey's fit of the ratio chart, not below the 1-atm value; within 5 %: PROSPER reads the chart)
  pseudo-pressure     m(p) = 2 int p_g / (mu Z) dp at the temperature, gauge pressure from 0 psig (PROSPER's M(P))
  condensed water     saturated water content of the gas, McKetta-Wehe (Bukacek fit) with McCain's salinity
                      correction, lb/MMscf / 350.2 = STB/MMscf. The minimum WGR is the content at reservoir pressure and
                      temperature (PROSPER's "modified" McKetta-Wehe is ~5 % lower: 0.27114 against 0.2853 STB/MMscf).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import blackoil as bo

P_STD = 14.696
RANKINE = 459.67
LB_PER_STB_WATER = 350.2
VISC_CORRELATIONS = {"lee": "Lee et al", "carr": "Carr et al"}


def condensate_mw(api):
    """Cragoe: molecular weight of stock-tank condensate."""
    return 6084.0 / max(api - 5.9, 1.0)


@dataclass
class GasPVTInputs:
    gas_sg: float = 0.70             # separator gas gravity (air = 1)
    sep_p: float = 200.0             # separator pressure, psig (sets the gas-equivalent of the condensate in PROSPER)
    cgr: float = 0.0                 # STB/MMscf
    cond_api: float = 50.0
    wgr: float = 0.0                 # STB/MMscf
    salinity_ppm: float = 0.0
    h2s: float = 0.0                 # mole %
    co2: float = 0.0
    n2: float = 0.0
    visc: str = "lee"                # "lee" | "carr"
    t_res: float = 200.0             # F (condensed water vapour)
    p_res: float = 3000.0            # psig
    match: dict = field(default_factory=dict)   # "z" | "bg" | "lee" | "carr" -> blackoil.LinearMatch (applied)

    @property
    def go(self):
        return 141.5 / (131.5 + self.cond_api)

    def wellstream_sg(self, cgr=None):
        cgr = self.cgr if cgr is None else cgr
        cgr = np.asarray(cgr, float)
        r = 1e6 / np.maximum(cgr, 1e-12)
        sg = (r * self.gas_sg + 4584.0 * self.go) / (r + 132800.0 * self.go / condensate_mw(self.cond_api))
        return np.where(cgr > 1e-12, sg, self.gas_sg)

    def bo_inputs(self, sg=None):
        return bo.PVTInputs(rsb=1, api=40, gas_sg=float(self.wellstream_sg() if sg is None else sg),
                            salinity_ppm=self.salinity_ppm, h2s=self.h2s, co2=self.co2, n2=self.n2,
                            z_sg=float(self.gas_sg))

    def check(self):
        out = []
        if not 0.55 <= self.gas_sg <= 2.0:
            out.append("Gas gravity should be between 0.55 and 2.")
        if self.h2s + self.co2 + self.n2 >= 100:
            out.append("Impurities add up to 100 % or more.")
        if self.cgr < 0 or self.wgr < 0:
            out.append("CGR and WGR cannot be negative.")
        return out


# ------------------------------------------------------------------ viscosity
_DEMPSEY = (-2.46211820, 2.97054714, -2.86264054e-1, 8.05420522e-3, 2.80860949, -3.49803305, 3.60373020e-1,
            -1.04432413e-2, -7.93385684e-1, 1.39643306, -1.49144925e-1, 4.41015512e-3, 8.39387178e-2,
            -1.86408848e-1, 2.03367881e-2, -6.09579263e-4)


def carr_viscosity(sg, t_f, p, tpc, ppc, n2=0.0, co2=0.0, h2s=0.0):
    """Carr, Kobayashi & Burrows (1954): 1-atm viscosity with the non-hydrocarbon corrections, times the
    viscosity ratio (Dempsey's 1965 fit), the ratio kept at or above 1."""
    t_f = np.asarray(t_f, float)
    p = np.asarray(p, float)
    lg = np.log10(sg)
    mu1 = (1.709e-5 - 2.062e-6 * sg) * t_f + 8.188e-3 - 6.15e-3 * lg
    mu1 = mu1 + n2 / 100 * (8.48e-3 * lg + 9.59e-3) + co2 / 100 * (9.08e-3 * lg + 6.24e-3) \
        + h2s / 100 * (8.49e-3 * lg + 3.73e-3)
    tr = (t_f + RANKINE) / tpc
    pr = p / ppc
    a = _DEMPSEY
    ln = (a[0] + a[1] * pr + a[2] * pr ** 2 + a[3] * pr ** 3 + tr * (a[4] + a[5] * pr + a[6] * pr ** 2 + a[7] * pr ** 3)
          + tr ** 2 * (a[8] + a[9] * pr + a[10] * pr ** 2 + a[11] * pr ** 3)
          + tr ** 3 * (a[12] + a[13] * pr + a[14] * pr ** 2 + a[15] * pr ** 3))
    return mu1 * np.maximum(np.exp(ln) / tr, 1.0)


def raw_props(inp: GasPVTInputs, p, t, sg=None):
    """Unmatched correlations: (Z, Bg, density, Lee viscosity, Carr viscosity)."""
    bi = inp.bo_inputs(sg)
    z, bg, rho, mu_lee = bo.gas_props(bi, p, t)
    tpc, ppc = bo.gas_pseudocritical(bi)
    mu_carr = carr_viscosity(bi.gas_sg, np.asarray(t, float) + 0 * np.asarray(p, float), np.maximum(p, P_STD), tpc,
                             ppc, inp.n2, inp.co2, inp.h2s)
    return z, bg, rho, mu_lee, mu_carr


def apply_match(match, z, bg, rho, mu, visc="lee"):
    """PROSPER's match parameters (Parameter 1 x correlation + Parameter 2): Z; gas FVF from the matched Z, then
    its own parameters; density = standard density / matched FVF (so the FVF match sets the in-situ gas volume
    and density in the well); the viscosity correlation in use (Lee from the unmatched density). Checked against
    PROSPER's PVT | Calculate table for Tutorial 06 (FVF to 0.002 %, density x FVF constant)."""
    if not match:
        return z, bg, rho, mu
    zm = match.get("z")
    z2 = zm.apply(z) if zm is not None else z
    bg2 = bg * z2 / np.maximum(z, 1e-9)
    bm = match.get("bg")
    if bm is not None:
        bg2 = np.maximum(bm.apply(bg2), 1e-6)
    rho2 = rho * bg / bg2                  # density = standard-condition density / matched FVF (as PROSPER)
    vm = match.get(visc)
    mu2 = vm.apply(mu) if vm is not None else mu
    return z2, bg2, rho2, np.maximum(mu2, 1e-4)


def gas_props(inp: GasPVTInputs, p, t, sg=None):
    """(Z, Bg ft3/scf, density lb/ft3, viscosity cp) of the wellstream gas at p psia, t F (matched if a PVT match is
    applied)."""
    z, bg, rho, mu_lee, mu_carr = raw_props(inp, p, t, sg)
    mu = mu_carr if inp.visc == "carr" else mu_lee
    return apply_match(inp.match, z, bg, rho, mu, inp.visc)


MATCH_PROPS = {"z": "Z factor", "bg": "Gas FVF", "lee": "Lee et al", "carr": "Carr et al"}


def fvf_consistency(inp: GasPVTInputs, t=None, n=40):
    """Largest relative departure of the matched gas FVF from the one implied by the matched Z (Bg = 0.02827 Z T/p)
    between 1 atm and reservoir pressure. The FVF match sets the in-situ gas volume and density in the well, so a
    large value (an FVF parameter 2 far from 0) changes the friction loss and the deliverability."""
    if not inp.match or "bg" not in inp.match:
        return 0.0
    t = inp.t_res if t is None else t
    p = np.linspace(P_STD + 100.0, inp.p_res + P_STD, n)
    z, bg, rho, mu_l, mu_c = raw_props(inp, p, np.full(n, float(t)))
    z2, bg2, _, _ = apply_match(inp.match, z, bg, rho, mu_l, "lee")
    return float(np.max(np.abs(bg2 / (bg * z2 / z) - 1)))


def match_gas(inp: GasPVTInputs, tables):
    """PROSPER PVT | Matching for dry and wet gas: tables = list of (temperature F, DataFrame with 'Pressure (psia)',
    'Z factor', 'Gas viscosity (cp)', 'Gas FVF (ft3/scf)' - blanks allowed). Each correlation is fitted as
    Parameter 1 x correlation + Parameter 2 by least squares over all tables (Z and both viscosity correlations
    always; gas FVF from the Z-matched FVF). Returns {prop: LinearMatch}."""
    xs = {k: [] for k in MATCH_PROPS}
    ys = {k: [] for k in MATCH_PROPS}
    zcorr_store = []
    for t, df in tables:
        if df is None or not len(df):
            continue
        p = pd.to_numeric(df["Pressure (psia)"], errors="coerce").to_numpy(float)
        ok = np.isfinite(p) & (p > 0)
        if not ok.any():
            continue
        p = p[ok]
        z, bg, rho, mul, muc = raw_props(inp, p, np.full(p.size, float(t)))
        for col, keys in (("Z factor", ("z",)), ("Gas viscosity (cp)", ("lee", "carr"))):
            if col in df:
                y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)[ok]
                for k in keys:
                    xs[k].append((z if k == "z" else (mul if k == "lee" else muc))[np.isfinite(y)])
                    ys[k].append(y[np.isfinite(y)])
        zcorr_store.append((p, t, z, bg, df, ok))
    out = {}
    for k in ("z", "lee", "carr"):
        if xs[k] and sum(a.size for a in xs[k]):
            out[k] = bo.fit_linear(np.concatenate(xs[k]), np.concatenate(ys[k]))
    # gas FVF: against the FVF from the matched Z
    bx, by = [], []
    for p, t, z, bg, df, ok in zcorr_store:
        if "Gas FVF (ft3/scf)" not in df:
            continue
        y = pd.to_numeric(df["Gas FVF (ft3/scf)"], errors="coerce").to_numpy(float)[ok]
        zm = out["z"].apply(z) if "z" in out else z
        x = bg * zm / z
        f = np.isfinite(y)
        bx.append(x[f])
        by.append(y[f])
    if bx and sum(a.size for a in bx):
        out["bg"] = bo.fit_linear(np.concatenate(bx), np.concatenate(by))
    return out


def pseudo_pressure(inp: GasPVTInputs, p, t, n=600):
    """m(p) = 2 int p_g/(mu Z) dp (psi2/cp) at temperature t, for pressures p (psia), with the gauge pressure
    p_g = p - 14.696 in the integrand from 0 psig - as PROSPER does (its M(P) column; within 0.06 % above
    300 psig, against 0.6 % for the absolute-pressure form). Differences of m drive the gas IPRs."""
    p = np.atleast_1d(np.asarray(p, float))
    grid = np.linspace(P_STD, max(float(p.max()), P_STD) * 1.0001 + 1.0, n)
    z, _, _, mu = gas_props(inp, grid, np.full(grid.size, float(t)))
    f = 2 * (grid - P_STD) / (mu * z)
    m = np.concatenate([[0.0], np.cumsum(np.diff(grid) * 0.5 * (f[1:] + f[:-1]))])
    return np.interp(p, grid, m)


# ------------------------------------------------------------------ condensed water vapour
def water_vapour_pressure(t_f):
    """Saturation pressure of pure water (psia), IAPWS-IF97 region 4."""
    tk = (np.asarray(t_f, float) - 32) / 1.8 + 273.15
    n = (0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, 0.12020824702470e5, -0.32325550322333e7,
         0.14915108613530e2, -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849, 0.65017534844798e3)
    th = tk + n[8] / (tk - n[9])
    a = th * th + n[0] * th + n[1]
    b = n[2] * th * th + n[3] * th + n[4]
    c = n[5] * th * th + n[6] * th + n[7]
    return (2 * c / (-b + np.sqrt(b * b - 4 * a * c))) ** 4 * 145.0377


def saturated_wgr(p, t_f, salinity_ppm=0.0):
    """Water vapour content of gas saturated with water, STB/MMscf, at p psia and t F: McKetta-Wehe chart
    (Bukacek's fit W = 47484 pv/p + B, lb/MMscf) with McCain's salinity correction."""
    p = np.maximum(np.asarray(p, float), P_STD)
    t_f = np.asarray(t_f, float)
    w = 47484.0 * water_vapour_pressure(t_f) / p + 10 ** (-3083.87 / (t_f + 459.6) + 6.69449)
    s = salinity_ppm / 1e4
    cs = np.clip(1 - 4.920e-3 * s - 1.7672e-4 * s * s, 0.3, 1.0)
    return w * cs / LB_PER_STB_WATER


def minimum_wgr(inp: GasPVTInputs):
    """Water vaporised in the gas at reservoir conditions (STB/MMscf): produced even with no free water."""
    return float(saturated_wgr(inp.p_res + P_STD, inp.t_res, inp.salinity_ppm))


def water_split(inp: GasPVTInputs, p, t_f, wgr):
    """(vapour, liquid) water, STB/MMscf, at p psia and t F for a produced WGR: the gas leaves the reservoir saturated
    (minimum WGR; a smaller WGR is raised to it) and, in contact with the free water, holds the saturated content at
    local conditions - up to all the water produced (PROSPER: no free water where the gas can hold the whole WGR)."""
    w_res = minimum_wgr(inp)
    total = np.maximum(np.asarray(wgr, float), w_res)
    vap = np.minimum(saturated_wgr(p, t_f, inp.salinity_ppm), total)
    return vap, np.maximum(total - vap, 0.0)


# ------------------------------------------------------------------ tables
def water_compressibility(inp: GasPVTInputs, p, t):
    bi = inp.bo_inputs()
    p = np.asarray(p, float)
    b1, _, _ = bo.water_props(bi, p - 5, t)
    b2, _, _ = bo.water_props(bi, p + 5, t)
    bw, _, _ = bo.water_props(bi, p, t)
    return np.maximum(-(b2 - b1) / 10 / bw, 1e-8)


def pvt_table(inp: GasPVTInputs, temps, pressures_psig):
    """PROSPER-style PVT | Calculate table for each temperature and pressure (psig)."""
    rows = []
    for t in np.atleast_1d(temps):
        p = np.asarray(pressures_psig, float) + P_STD
        z, bg, rho, mu = gas_props(inp, p, np.full(p.size, float(t)))
        m = pseudo_pressure(inp, p, float(t))
        bw, rw, muw = bo.water_props(inp.bo_inputs(), p, np.full(p.size, float(t)))
        cw = water_compressibility(inp, p, np.full(p.size, float(t)))
        _, free = water_split(inp, p, np.full(p.size, float(t)), max(inp.wgr, 0.0))
        for i in range(p.size):
            rows.append({"Temperature (°F)": float(t), "Pressure (psig)": float(p[i] - P_STD),
                         "Gas density (lb/ft3)": rho[i], "Gas viscosity (cp)": mu[i], "Gas FVF (ft3/scf)": bg[i],
                         "Z factor": z[i], "Water density (lb/ft3)": rw[i], "Water viscosity (cp)": muw[i],
                         "Water FVF (RB/STB)": bw[i], "Water compressibility (1/psi)": cw[i],
                         "Pseudo pressure (psi2/cp)": m[i], "Free water (STB/MMscf)": free[i]})
    return pd.DataFrame(rows)
