"""Black-oil PVT correlations, matching to lab data, and correlation suggestion.

Oilfield units: psia, degF, scf/STB, RB/STB, cp, lb/ft3. All functions are
vectorised over pressure.

Correlation families (bubble point Pb, solution GOR Rs, oil FVF Bo)
--------------------------------------------------------------------
Standing (1947), Lasater (1958; Bo from Standing - Lasater published none),
Vasquez-Beggs (1980; gas gravity referred to the 100 psig separator from stock-tank conditions), Glaso (1980), Petrosky-Farshad (1993), Al-Marhoun (1988).
Above Pb, Bo uses each family's undersaturated compressibility: Petrosky-
Farshad and Vasquez-Beggs have their own; the others use Vasquez-Beggs.

Viscosity families (dead oil / gas-saturated / undersaturated)
--------------------------------------------------------------
Beal et al.     Beal (Standing's fit) / Chew-Connally / Beal
Beggs et al.    Beggs-Robinson / Beggs-Robinson / Vasquez-Beggs
Petrosky et al. Petrosky-Farshad (all three)
Egbogah et al.  Egbogah-Jack / Beggs-Robinson / Vasquez-Beggs

De Ghetto et al., Bergman-Sutton and "De Ghetto mod" are listed but not
implemented: their coefficients could not be verified against a published
source (see UNAVAILABLE).

Suggestion
----------
Score per property = average correction P1/P2 apply at the data points
+ standard deviation after matching + a penalty when P1 is outside 0.9-1.1.
The Pb/Rs/Bo family score is the mean over the three properties. Correlations
whose dataset does not cover the oil's API and temperature rank behind those
that do.

Matching
--------
Each property is corrected linearly, corrected = P1 x correlation + P2, and
P1/P2 are fitted to the lab data by least squares. Solution GOR and oil FVF
also pass through a fixed stock-tank point (GOR 0 at 14.7 psia and 60 degF;
FVF 1.0 for gas-free oil at 60 degF), as PROSPER does, so one lab point fixes
both parameters. With fewer points than parameters (a bubble point, or a
viscosity point) the minimum-change solution is taken, which splits the
correction equally between P1 and P2. A correlation
needing P1 close to 1 and P2 close to 0 already describes the fluid well.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RANKINE = 459.67
P_STD = 14.696

PB_FAMILIES = {
    "glaso": "Glaso",
    "standing": "Standing",
    "lasater": "Lasater",
    "vazquez_beggs": "Vazquez-Beggs",
    "petrosky": "Petrosky et al",
    "al_marhoun": "Al-Marhoun",
    "de_ghetto": "De Ghetto et al",
}
VISC_FAMILIES = {
    "beal": "Beal et al",
    "beggs": "Beggs et al",
    "petrosky": "Petrosky et al",
    "egbogah": "Egbogah et al (Heavy oil)",
    "bergman_sutton": "Bergman-Sutton",
    "de_ghetto": "De Ghetto et al (Heavy oil)",
    "de_ghetto_mod": "De Ghetto Mod",
}
UNAVAILABLE = {
    ("visc", "de_ghetto_mod"): "PROSPER's proprietary modification - formulation not published",
}
# Approximate API and temperature (degF) ranges of each correlation's dataset.
RANGES = {
    ("pb", "standing"): ((16.5, 63.8), (100, 258)),
    ("pb", "lasater"): ((17.9, 51.1), (82, 272)),
    ("pb", "vazquez_beggs"): ((15.3, 59.5), (75, 294)),
    ("pb", "glaso"): ((22.3, 48.1), (80, 280)),
    ("pb", "petrosky"): ((16.3, 45.0), (114, 288)),
    ("pb", "al_marhoun"): ((19.4, 44.6), (74, 240)),
    ("pb", "de_ghetto"): ((10.0, 22.3), (131, 250)),
    ("visc", "beal"): ((10.1, 52.5), (98, 250)),
    ("visc", "beggs"): ((16.0, 58.0), (70, 295)),
    ("visc", "petrosky"): ((25.4, 46.1), (114, 288)),
    ("visc", "egbogah"): ((0.0, 22.3), (59, 250)),
    ("visc", "de_ghetto"): ((10.0, 22.3), (131, 250)),
}


def available(kind: str, fam: str) -> bool:
    return (kind, fam) not in UNAVAILABLE


# =========================================================== fluid inputs
@dataclass
class PVTInputs:
    rsb: float                 # solution GOR at bubble point, scf/STB
    api: float                 # oil gravity, degAPI
    gas_sg: float              # gas specific gravity (air = 1)
    salinity_ppm: float = 0.0
    h2s: float = 0.0           # mole percent
    co2: float = 0.0
    n2: float = 0.0
    correct_impurities: bool = False  # Glaso non-hydrocarbon correction on Pb (PROSPER does not apply it)
    watson_k: float = 12.1            # Watson characterisation factor (Bergman-Sutton dead-oil viscosity only)

    @property
    def gamma_o(self) -> float:
        return 141.5 / (131.5 + self.api)

    def check(self) -> list[str]:
        e = []
        if self.rsb <= 0: e.append("Solution GOR must be positive.")
        if not 5 <= self.api <= 70: e.append("Oil gravity should be between 5 and 70 °API.")
        if not 0.55 <= self.gas_sg <= 2.0: e.append("Gas gravity should be between 0.55 and 2.0.")
        if self.salinity_ppm < 0 or self.salinity_ppm > 300_000:
            e.append("Water salinity should be between 0 and 300,000 ppm.")
        if min(self.h2s, self.co2, self.n2) < 0 or self.h2s + self.co2 + self.n2 >= 100:
            e.append("Impurity mole percents must be non-negative and sum below 100.")
        return e


# ================================================= non-hydrocarbon factor
def impurity_factor(inp: PVTInputs, t):
    """Glaso (1980) multiplier on bubble point for N2, CO2, H2S (T in degF)."""
    if not inp.correct_impurities:
        return np.ones_like(np.asarray(t, float))
    t = np.asarray(t, float)
    api = inp.api
    yn, yc, yh = inp.n2 / 100, inp.co2 / 100, inp.h2s / 100
    c_n2 = (1 + ((-2.65e-4 * api + 5.5e-3) * t + (0.0931 * api - 0.8295)) * yn
            + ((1.954e-11 * api ** 4.699) * t + (0.027 * api - 2.366)) * yn ** 2)
    c_co2 = 1 - 693.8 * yc * t ** -1.553
    c_h2s = 1 - (0.9035 + 0.0015 * api) * yh + 0.019 * (45 - api) * yh ** 2
    return c_n2 * c_co2 * c_h2s


# ============================================================ Pb, Rs, Bo
def vb_gas_sg(inp: "PVTInputs", t_sep=60.0, p_sep=14.7):
    """Vazquez-Beggs gas gravity referred to a 100 psig separator: gg (1 + 5.912e-5 API Tsep log10(psep/114.7)).
    The entered gravity is taken as stock-tank gas (separator at 14.7 psia, 60 degF), as PROSPER does."""
    return inp.gas_sg * (1 + 5.912e-5 * inp.api * t_sep * np.log10(p_sep / 114.7))


def _vb_coeffs(api):
    if api <= 30:
        return (0.0362, 1.0937, 25.7240), (4.677e-4, 1.751e-5, -1.811e-8)
    return (0.0178, 1.1870, 23.931), (4.670e-4, 1.100e-5, 1.337e-9)


def _lasater_mo(api):
    return 630 - 10 * api if api <= 40 else 73110 * api ** -1.562


def pb_raw(fam: str, inp: PVTInputs, rs, t):
    """Uncorrected correlation bubble point (psia) of oil holding Rs at T."""
    rs = np.maximum(np.asarray(rs, float), 1e-6)
    t = np.asarray(t, float)
    gg, api, go = inp.gas_sg, inp.api, inp.gamma_o
    tr = t + RANKINE
    if fam == "standing":
        return 18.2 * ((rs / gg) ** 0.83 * 10 ** (0.00091 * t - 0.0125 * api) - 1.4)
    if fam == "lasater":
        mo = _lasater_mo(api)
        yg = (rs / 379.3) / (rs / 379.3 + 350 * go / mo)
        pf = np.where(yg <= 0.6, 0.679 * np.exp(2.786 * yg) - 0.323, 8.26 * yg ** 3.56 + 1.95)
        return pf * tr / gg
    if fam == "vazquez_beggs":
        (c1, c2, c3), _ = _vb_coeffs(api)
        return (rs / (c1 * vb_gas_sg(inp) * np.exp(c3 * api / tr))) ** (1 / c2)
    if fam == "glaso":
        x = np.log10((rs / gg) ** 0.816 * t ** 0.172 / api ** 0.989)
        return 10 ** (1.7669 + 1.7447 * x - 0.30218 * x * x)
    if fam == "petrosky":
        x = 7.916e-4 * api ** 1.5410 - 4.561e-5 * t ** 1.3911
        return 112.727 * (rs ** 0.577421 / (gg ** 0.8439 * 10 ** x) - 12.340)
    if fam == "al_marhoun":
        return 5.38088e-3 * rs ** 0.715082 * gg ** -1.87784 * go ** 3.1437 * tr ** 1.32657
    if fam == "de_ghetto":         # heavy-oil modified Vazquez-Beggs (De Ghetto et al. 1995), as PROSPER
        return (56.434 * rs / (vb_gas_sg(inp) * 10 ** (10.9267 * api / tr))) ** (1 / 1.2057)
    raise KeyError(fam)


def rs_raw(fam: str, inp: PVTInputs, p, t):
    """Uncorrected saturated solution GOR (scf/STB) at pressure p, T."""
    p = np.maximum(np.asarray(p, float), 1.0)
    t = np.asarray(t, float)
    gg, api, go = inp.gas_sg, inp.api, inp.gamma_o
    tr = t + RANKINE
    if fam == "standing":
        rs = gg * ((p / 18.2 + 1.4) * 10 ** (0.0125 * api - 0.00091 * t)) ** (1 / 0.83)
    elif fam == "lasater":
        pf = p * gg / tr
        yg = np.where(pf < 3.29, 0.359 * np.log(1.473 * pf + 0.476),
                      np.maximum(0.121 * pf - 0.236, 0) ** 0.281)
        yg = np.clip(yg, 0, 0.999)
        rs = 132755 * go * yg / (_lasater_mo(api) * (1 - yg))
    elif fam == "vazquez_beggs":
        (c1, c2, c3), _ = _vb_coeffs(api)
        rs = c1 * vb_gas_sg(inp) * p ** c2 * np.exp(c3 * api / tr)
    elif fam == "glaso":
        arg = np.maximum(14.1811 - 3.3093 * np.log10(p), 0)
        pbs = 10 ** (2.8869 - np.sqrt(arg))
        rs = gg * (pbs * api ** 0.989 / t ** 0.172) ** (1 / 0.816)
    elif fam == "petrosky":
        x = 7.916e-4 * api ** 1.5410 - 4.561e-5 * t ** 1.3911
        rs = ((p / 112.727 + 12.340) * gg ** 0.8439 * 10 ** x) ** 1.73184
    elif fam == "al_marhoun":
        rs = (185.843208 * p * gg ** 1.87784 * go ** -3.1437 * tr ** -1.32657) ** 1.398441
    elif fam == "de_ghetto":
        rs = vb_gas_sg(inp) * p ** 1.2057 / 56.434 * 10 ** (10.9267 * api / tr)
    else:
        raise KeyError(fam)
    return np.maximum(rs, 0.0)


def bo_sat(fam: str, inp: PVTInputs, rs, t):
    """Saturated oil FVF (RB/STB) for oil holding Rs at T."""
    rs = np.maximum(np.asarray(rs, float), 0.0)
    t = np.asarray(t, float)
    gg, api, go = inp.gas_sg, inp.api, inp.gamma_o
    if fam in ("standing", "lasater"):      # Standing (1947) original form, as PROSPER uses
        return 0.972 + 0.000147 * (rs * np.sqrt(gg / go) + 1.25 * t) ** 1.175
    if fam in ("vazquez_beggs", "de_ghetto"):         # De Ghetto keeps the Vazquez-Beggs FVF
        _, (a1, a2, a3) = _vb_coeffs(api)
        ggs = vb_gas_sg(inp)
        return 1 + a1 * rs + a2 * (t - 60) * (api / ggs) + a3 * rs * (t - 60) * (api / ggs)
    if fam == "glaso":
        b = np.log10(rs * (gg / go) ** 0.526 + 0.968 * t)
        return 1 + 10 ** (-6.58511 + 2.91329 * b - 0.27683 * b * b)
    if fam == "petrosky":
        return 1.0113 + 7.2046e-5 * (rs ** 0.3738 * gg ** 0.2914 / go ** 0.6265
                                     + 0.24626 * t ** 0.5371) ** 3.0936
    if fam == "al_marhoun":
        f = rs ** 0.742390 * gg ** 0.323294 * go ** -1.202040
        return 0.497069 + 8.62963e-4 * (t + RANKINE) + 1.82594e-3 * f + 3.18099e-6 * f * f
    raise KeyError(fam)


def bo_undersat_factor(fam: str, inp: PVTInputs, p, pb, rsb, t):
    """Bo(p)/Bo(pb) for p >= pb (1 below pb)."""
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    pe = np.maximum(p, pb)
    if fam == "petrosky":
        a = 4.1646e-7 * rsb ** 0.69357 * inp.gas_sg ** 0.1885 * inp.api ** 0.3272 * t ** 0.6729
        return np.exp(a * (pb ** 0.4094 - pe ** 0.4094))
    gg = vb_gas_sg(inp) if fam in ("vazquez_beggs", "de_ghetto") else inp.gas_sg
    a = (-1433 + 5 * rsb + 17.2 * t - 1180 * gg + 12.61 * inp.api) / 1e5
    return (pe / pb) ** (-a)


# ============================================================ viscosity
def _bs_vcf(go, t):
    """ASTM volume correction factor with the Bergman-Sutton coefficients for crude oil (SPE 110194 eq. 11)."""
    rho60 = 0.999012 * go
    a = (2.5042e-4 + 8.302e-5 * rho60) / rho60 ** 2
    dt = t - 60.0
    return np.exp(-a * dt * (1 + 0.8 * a * dt))


def bergman_sutton_dead(api, t, kw):
    """Bergman & Sutton (2007, SPE 110194): modified Twu kinematic viscosity at 100 and 210 degF from Watson K and
    gravity, converted to absolute viscosity, then Bergman's double-log temperature line (eq. 8)."""
    go = 141.5 / (131.5 + api)
    tb = (go * kw) ** 3                                                    # degR
    tc = tb / (0.533272 + 1.91017e-4 * tb + 7.79681e-8 * tb ** 2 - 2.84376e-11 * tb ** 3 + 9.59468e27 / tb ** 13)
    al = 1 - tb / tc
    nu2 = np.exp(2.40219 - 9.59688 * al + 3.45656 * al ** 2 - 143.632 * al ** 4) + 0.152995
    l2 = np.log(nu2)
    nu1 = np.exp(0.701254 + 1.38359 * l2 + 0.103604 * l2 ** 2)
    dg = go - (0.843593 - 0.128624 * al - 3.36159 * al ** 3 - 13749.5 * al ** 12)
    x = abs(2.68316 - 62.0863 / tb ** 0.5)
    f2 = x * dg - 47.6033 * dg ** 2 / tb ** 0.5
    f1 = 0.980633 * x * dg - 47.6033 * dg ** 2 / tb ** 0.5
    k = 232.442 / tb
    nu210 = np.exp(np.log(nu2 + k) * ((1 + 2 * f2) / (1 - 2 * f2)) ** 2) - k
    nu100 = np.exp(np.log(nu1 + k) * ((1 + 2 * f1) / (1 - 2 * f1)) ** 2) - k
    mu100 = nu100 * 0.999012 * go * _bs_vcf(go, 100.0)
    mu210 = nu210 * 0.999012 * go * _bs_vcf(go, 210.0)
    y1, y2 = np.log(np.log(mu100 + 1)), np.log(np.log(mu210 + 1))
    x1, x2 = np.log(410.0), np.log(520.0)
    b = (y2 - y1) / (x2 - x1)
    return np.exp(np.exp(y1 + b * (np.log(np.asarray(t, float) + 310) - x1))) - 1


def visc_dead(fam: str, inp: PVTInputs, t):
    t = np.asarray(t, float)
    api = inp.api
    if fam == "beal":
        a = 10 ** (0.43 + 8.33 / api)
        return (0.32 + 1.8e7 / api ** 4.53) * (360 / (t + 200)) ** a
    if fam in ("beggs",):
        return 10 ** (10 ** (3.0324 - 0.02023 * api) * t ** -1.163) - 1
    if fam == "petrosky":
        x = 4.59388 * np.log10(t) - 22.82792
        return 2.3511e7 * t ** -2.10255 * np.log10(api) ** x
    if fam == "bergman_sutton":
        return bergman_sutton_dead(api, t, inp.watson_k)
    if fam == "egbogah":           # PROSPER "Egbogah (Heavy oil)": De Ghetto's modified Egbogah-Jack (A-6)
        return 10 ** (10 ** (1.90296 - 0.012619 * api - 0.61748 * np.log10(t))) - 1
    if fam == "de_ghetto":         # De Ghetto et al. (1995) heavy oil, modified Egbogah-Jack (A-7)
        return 10 ** (10 ** (2.06492 - 0.0179 * api - 0.70226 * np.log10(t))) - 1
    raise KeyError(fam)


def visc_live(fam: str, mu_od, rs):
    rs = np.maximum(np.asarray(rs, float), 0.0)
    if fam == "beal":            # Chew-Connally (Standing's fit)
        a = 10 ** (rs * (2.2e-7 * rs - 7.4e-4))
        b = 0.68 / 10 ** (8.62e-5 * rs) + 0.25 / 10 ** (1.1e-3 * rs) + 0.062 / 10 ** (3.74e-3 * rs)
        return a * mu_od ** b
    if fam == "beggs":
        return 10.715 * (rs + 100) ** -0.515 * mu_od ** (5.44 * (rs + 150) ** -0.338)
    if fam == "bergman_sutton":    # SPE 110195, eqs 4-6 (as corrected April 2009)
        a = 1 / (1 + (rs / 344.198) ** 0.855344)
        b = 0.382323 + (1 - 0.382323) / (1 + (rs / 567.953) ** 0.819326)
        return a * mu_od ** b
    if fam == "de_ghetto":         # modified Kartoatmodjo, heavy oil (A-9)
        y = 10 ** (-0.00081 * rs)
        f = (0.2478 + 0.6114 * 10 ** (-0.000845 * rs)) * mu_od ** (0.4731 + 0.5158 * y)
        return -0.6311 + 1.078 * f - 0.003653 * f * f
    if fam in ("petrosky", "egbogah"):
        a = 0.1651 + 0.6165 * 10 ** (-6.0866e-4 * rs)
        b = 0.5131 + 0.5109 * 10 ** (-1.1831e-3 * rs)
        return a * mu_od ** b
    raise KeyError(fam)


def visc_undersat(fam: str, mu_ob, p, pb):
    p = np.asarray(p, float)
    dp = np.maximum(p - pb, 0.0)
    if fam == "beal":
        return mu_ob + 0.001 * dp * (0.024 * mu_ob ** 1.6 + 0.038 * mu_ob ** 0.56)
    if fam == "bergman_sutton":    # SPE 103144, eqs 7-9
        lm = np.log(mu_ob)
        a = 6.5698e-7 * lm ** 2 - 1.48211e-5 * lm + 2.27877e-4
        b = 2.24623e-2 * lm + 0.873204
        return mu_ob * np.exp(a * dp ** b)
    if fam == "de_ghetto":         # modified Kartoatmodjo, heavy oil (A-11)
        return 0.9886 * mu_ob + 0.002763 * dp * (-0.01153 * mu_ob ** 1.7933 + 0.0316 * mu_ob ** 1.5939)
    if fam == "beggs":
        pe = np.maximum(p, pb)
        m = 2.6 * pe ** 1.187 * np.exp(-11.513 - 8.98e-5 * pe)
        return mu_ob * (pe / pb) ** m
    if fam in ("petrosky", "egbogah"):
        x = np.log10(mu_ob)
        a = -1.0146 + 1.3322 * x - 0.4876 * x * x - 1.15036 * x ** 3
        return mu_ob + 1.3449e-3 * dp * 10 ** a
    raise KeyError(fam)


# ====================================================== gas and water
def gas_pseudocritical(inp: PVTInputs):
    """Sutton pseudo-criticals with Wichert-Aziz sour-gas correction (degR, psia)."""
    g = inp.gas_sg
    tpc = 169.2 + 349.5 * g - 74.0 * g * g
    ppc = 756.8 - 131.07 * g - 3.6 * g * g
    a = (inp.co2 + inp.h2s) / 100
    b = inp.h2s / 100
    eps = 120 * (a ** 0.9 - a ** 1.6) + 15 * (b ** 0.5 - b ** 4)
    tpc_c = tpc - eps
    ppc_c = ppc * tpc_c / (tpc + b * (1 - b) * eps)
    return tpc_c, ppc_c


def z_dak(tpr, ppr, iters: int = 12):
    tpr = np.asarray(tpr, float)
    ppr = np.asarray(ppr, float)
    A = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)
    c1 = A[0] + A[1] / tpr + A[2] / tpr ** 3 + A[3] / tpr ** 4 + A[4] / tpr ** 5
    c2 = A[5] + A[6] / tpr + A[7] / tpr ** 2
    c3 = A[8] * (A[6] / tpr + A[7] / tpr ** 2)
    c4 = A[9] / tpr ** 3
    rho = 0.27 * ppr / tpr
    for _ in range(iters):
        r2 = rho * rho
        e = np.exp(-A[10] * r2)
        z = 1 + c1 * rho + c2 * r2 - c3 * r2 * r2 * rho + c4 * (r2 + A[10] * r2 * r2) * e
        f = z - 0.27 * ppr / (rho * tpr)
        dz = (c1 + 2 * c2 * rho - 5 * c3 * r2 * r2
              + c4 * e * (2 * rho + 2 * A[10] * r2 * rho - 2 * A[10] ** 2 * r2 * r2 * rho))
        rho = np.clip(rho - f / (dz + 0.27 * ppr / (r2 * tpr)), 1e-8, 3.0)
    return np.clip(0.27 * ppr / (rho * tpr), 0.2, 2.0)


def gas_props(inp: PVTInputs, p, t):
    """Return (Z, Bg ft3/scf, rho_g lb/ft3, mu_g cp)."""
    p = np.maximum(np.asarray(p, float), P_STD)
    tr = np.asarray(t, float) + RANKINE
    tpc, ppc = gas_pseudocritical(inp)
    z = z_dak(tr / tpc, p / ppc)
    bg = 0.02827 * z * tr / p
    m = 28.967 * inp.gas_sg
    rho = m * p / (z * 10.7316 * tr)
    k = (9.4 + 0.02 * m) * tr ** 1.5 / (209 + 19 * m + tr)
    x = 3.5 + 986 / tr + 0.01 * m
    mu = 1e-4 * k * np.exp(x * (rho / 62.428) ** (2.4 - 0.2 * x))
    return z, bg, rho, mu


def water_props(inp: PVTInputs, p, t):
    """McCain (1990): return (Bw RB/STB, rho_w lb/ft3, mu_w cp); salinity in weight %."""
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    s = inp.salinity_ppm / 1e4
    dvt = -1.0001e-2 + 1.33391e-4 * t + 5.50654e-7 * t * t
    dvp = -1.95301e-9 * p * t - 1.72834e-13 * p * p * t - 3.58922e-7 * p - 2.25341e-10 * p * p
    bw = (1 + dvt) * (1 + dvp)
    rho_std = 62.368 + 0.438603 * s + 1.60074e-3 * s * s
    a = 109.574 - 8.40564 * s + 0.313314 * s * s + 8.72213e-4 * s ** 3
    b = 1.12166 - 2.63951e-2 * s + 6.79461e-4 * s * s + 5.47119e-5 * s ** 3 - 1.55586e-6 * s ** 4
    mu1 = a * t ** -b
    mu = mu1 * (0.9994 + 4.0295e-5 * p + 3.1062e-9 * p * p)
    return bw, rho_std / bw, mu


# ============================================================ matching
@dataclass
class LinearMatch:
    p1: float = 1.0
    p2: float = 0.0
    std: float = np.nan        # relative std deviation of matched residuals (n >= 3)
    raw_error_pct: float = np.nan
    n: int = 0
    scale: float = np.nan      # mean measured value
    applied: float = np.nan    # mean |P1*x + P2 - x| / y at the data points

    def apply(self, x):
        return self.p1 * np.asarray(x, float) + self.p2

    @property
    def correction(self) -> float:
        """Average fraction by which P1 and P2 change the correlation at the data."""
        return self.applied

    @property
    def p1_penalty(self) -> float:
        """Extra score when P1 strays outside 0.9-1.1."""
        return 0.25 * max(0.0, abs(self.p1 - 1) - 0.10)

    @property
    def score(self) -> float:
        """Applied correction + std deviation + P1 penalty. Lower is better."""
        if self.n == 0:
            return np.nan
        return self.applied + (0.0 if np.isnan(self.std) else self.std) + self.p1_penalty


def fit_linear(x, y, lam: float = 1e-8, anchor=None) -> LinearMatch:
    """Fit y ~ P1*x + P2. Parameters are solved as deviations from (1, 0) with
    a vanishing ridge, so under-determined cases take the smallest correction.
    anchor = (x0, y0): a fixed point added to the lab points (stock-tank conditions for GOR and FVF, as PROSPER
    does) - it takes part in the fit but is not counted or scored as a lab point."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n_lab = x.size
    if n_lab == 0:
        return LinearMatch()
    xl, yl = x, y
    if anchor is not None and np.all(np.isfinite(anchor)):
        x, y = np.append(x, anchor[0]), np.append(y, anchor[1])
    n = x.size
    s = float(np.mean(np.abs(x))) or 1.0
    a = x / s
    d = (y - x) / s
    M = np.array([[a @ a, a.sum()], [a.sum(), n]]) + lam * n * np.eye(2)
    u, v = np.linalg.solve(M, np.array([a @ d, d.sum()]))
    p1, p2 = 1 + u, v * s
    fit = p1 * xl + p2
    std = float(np.sqrt(np.sum(((fit - yl) / yl) ** 2) / (n - 2))) if n >= 3 else np.nan
    raw = float(np.mean(np.abs(xl - yl) / np.abs(yl)) * 100)
    applied = float(np.mean(np.abs(fit - xl) / np.abs(yl)))
    return LinearMatch(float(p1), float(p2), std, raw, n_lab, float(np.mean(np.abs(yl))), applied)


# ------------------------------------------------------------ lab data
TABLE_COLUMNS = ["Pressure", "Gas Oil Ratio", "Oil FVF", "Oil Viscosity"]


@dataclass
class MatchData:
    temperature: float            # degF
    bubble_point: float | None    # psia (measured)
    table: pd.DataFrame           # TABLE_COLUMNS; blanks allowed

    def clean(self) -> pd.DataFrame:
        t = self.table.copy()
        for c in TABLE_COLUMNS:
            t[c] = pd.to_numeric(t.get(c), errors="coerce")
        t = t[t["Pressure"].notna() & (t["Pressure"] > 0)]
        return t.sort_values("Pressure").reset_index(drop=True)

    def check(self, inp: PVTInputs) -> list[str]:
        """Plain-language consistency warnings on the lab data."""
        w = []
        t = self.clean()
        pb = self.bubble_point
        if not 40 <= self.temperature <= 400:
            w.append("Temperature should be between 40 and 400 °F.")
        if pb is None:
            w.append("No bubble point entered: the bubble point cannot be matched, and points "
                     "cannot be classed as above or below it.")
        if t.empty:
            w.append("No table points entered: only the bubble point can be matched.")
            return w
        hi = t["Gas Oil Ratio"] > inp.rsb * 1.02
        if hi.any():
            w.append(f"GOR above the input solution GOR ({inp.rsb:g}) at "
                     f"{', '.join(f'{p:g}' for p in t.loc[hi, 'Pressure'])} psia.")
        if pb is not None:
            above = t[t["Pressure"] > pb * 1.001]
            bad = above["Gas Oil Ratio"].notna() & (abs(above["Gas Oil Ratio"] - inp.rsb) > 0.02 * inp.rsb)
            if bad.any():
                w.append("Above the bubble point the GOR should equal the solution GOR; it differs at "
                         + ", ".join(f"{p:g}" for p in above.loc[bad, "Pressure"]) + " psia.")
            below = t[t["Pressure"] <= pb * 1.001]
            for col, sign, what in (("Gas Oil Ratio", 1, "GOR"), ("Oil FVF", 1, "Oil FVF"),
                                    ("Oil Viscosity", -1, "Oil viscosity")):
                v = below[col].dropna()
                if v.size >= 2 and np.any(sign * np.diff(v.to_numpy()) < 0):
                    w.append(f"Below the bubble point {what} should "
                             f"{'rise' if sign > 0 else 'fall'} with pressure; check the table.")
            for col, sign, what in (("Oil FVF", -1, "Oil FVF"), ("Oil Viscosity", 1, "Oil viscosity")):
                v = above[col].dropna()
                if v.size >= 2 and np.any(sign * np.diff(v.to_numpy()) < 0):
                    w.append(f"Above the bubble point {what} should "
                             f"{'fall' if sign < 0 else 'rise'} with pressure; check the table.")
        return w


def _pb_corr(fam, inp, t):
    return pb_raw(fam, inp, inp.rsb, t) * impurity_factor(inp, t)


def _rs_corr(fam, inp, p, t):
    return rs_raw(fam, inp, np.asarray(p, float) / impurity_factor(inp, t), t)


def _point_rs(inp, tbl, pb, fam, t):
    """GOR at each table point: measured if given, else Rsb above Pb, else correlation."""
    p = tbl["Pressure"].to_numpy(float)
    rs = tbl["Gas Oil Ratio"].to_numpy(float)
    above = p >= pb if pb is not None else np.zeros_like(p, bool)
    fill = np.where(above, inp.rsb, np.minimum(_rs_corr(fam, inp, p, t), inp.rsb))
    return np.where(np.isfinite(rs), rs, fill), above


def correlation_values(kind, fam, inp: PVTInputs, data: MatchData, vfam_for_rs="standing"):
    """Correlation predictions at the lab points: dict property -> (x_corr, y_meas)."""
    t = data.temperature
    tbl = data.clean()
    pb = data.bubble_point
    out = {}
    if kind == "pb":
        out["pb"] = (np.atleast_1d(_pb_corr(fam, inp, t)),
                     np.array([pb if pb is not None else np.nan]))
        p = tbl["Pressure"].to_numpy(float)
        below = p <= pb * 1.001 if pb is not None else np.ones_like(p, bool)
        m = below & tbl["Gas Oil Ratio"].notna().to_numpy()
        out["rs"] = (_rs_corr(fam, inp, p[m], t), tbl["Gas Oil Ratio"].to_numpy(float)[m])
        rs_pt, above = _point_rs(inp, tbl, pb, fam, t)
        x = bo_sat(fam, inp, rs_pt, t)
        if pb is not None:
            x = np.where(above, bo_sat(fam, inp, inp.rsb, t)
                         * bo_undersat_factor(fam, inp, p, pb, inp.rsb, t), x)
        mb = tbl["Oil FVF"].notna().to_numpy()
        out["bo"] = (x[mb], tbl["Oil FVF"].to_numpy(float)[mb])
    else:
        p = tbl["Pressure"].to_numpy(float)
        rs_pt, above = _point_rs(inp, tbl, pb, vfam_for_rs, t)
        if pb is not None:
            above = p > pb               # at the bubble point itself viscosity is the saturated value (as PROSPER)
        mod = visc_dead(fam, inp, t)
        x = visc_live(fam, mod, rs_pt)
        if pb is not None:
            mob = visc_live(fam, mod, inp.rsb)
            x = np.where(above, visc_undersat(fam, mob, p, pb), x)
        mv = tbl["Oil Viscosity"].notna().to_numpy()
        out["visc"] = (x[mv], tbl["Oil Viscosity"].to_numpy(float)[mv])
    return out


T_STD = 60.0


def stock_tank_anchors(fam, inp):
    """Fixed matching points at stock-tank conditions: no gas in solution at 14.7 psia and 60 degF, and gas-free
    oil with FVF 1.0 at 60 degF. They keep the corrected GOR and FVF curves pinned at the stock tank."""
    return {"rs": (float(_rs_corr(fam, inp, P_STD, T_STD)), 0.0),
            "bo": (float(bo_sat(fam, inp, 0.0, T_STD)), 1.0)}


PROPS = {"pb": "Bubble Point", "rs": "Solution GOR", "bo": "Oil FVF", "visc": "Oil Viscosity"}


@dataclass
class MatchResult:
    matches: dict                  # (prop, fam) -> LinearMatch
    suggestion: dict               # "pb_family" / "visc_family" -> key or None
    rationale: dict                # same keys -> text
    warnings: list = field(default_factory=list)
    inp: PVTInputs | None = None
    temperature: float = float("nan")

    def table(self, prop: str) -> pd.DataFrame:
        """PROSPER-style grid: rows Parameter 1, Parameter 2, Std Deviation; one column per correlation."""
        fams = VISC_FAMILIES if prop == "visc" else PB_FAMILIES
        kind = "visc" if prop == "visc" else "pb"
        rows = {"Parameter 1": [], "Parameter 2": [], "Std Deviation": [],
                "Raw error (%)": [], "Points": []}
        for f in fams:
            m = self.matches.get((prop, f))
            if m is None or not available(kind, f) or m.n == 0:
                for r in rows:
                    rows[r].append(None)
                continue
            rows["Parameter 1"].append(m.p1)
            rows["Parameter 2"].append(m.p2)
            rows["Std Deviation"].append(None if np.isnan(m.std) else m.std)
            rows["Raw error (%)"].append(m.raw_error_pct)
            rows["Points"].append(m.n)
        return pd.DataFrame(rows, index=list(fams.values())).T

    def ranking(self, kind: str) -> pd.DataFrame:
        props = ["visc"] if kind == "visc" else ["pb", "rs", "bo"]
        fams = VISC_FAMILIES if kind == "visc" else PB_FAMILIES
        recs = []
        for f, label in fams.items():
            if not available(kind, f):
                recs.append({"Correlation": label, "Score": None, "Status": UNAVAILABLE[(kind, f)]})
                continue
            ms = [self.matches[(p, f)] for p in props if self.matches[(p, f)].n > 0]
            if not ms:
                continue
            rec = {"Correlation": label, "Score": float(np.mean([m.score for m in ms]))}
            for p in props:
                m = self.matches[(p, f)]
                rec[f"{PROPS[p]} correction"] = None if m.n == 0 else m.correction
            rec["Status"] = _range_note(kind, f, self.inp, self.temperature) or "Within data range"
            recs.append(rec)
        df = pd.DataFrame(recs)
        if df.empty:
            return df
        df["_na"] = df["Score"].isna()
        df = df.sort_values(["_na", "Score"]).drop(columns="_na").reset_index(drop=True)
        df.insert(0, "Rank", [i + 1 if s is not None and not pd.isna(s) else None
                              for i, s in enumerate(df["Score"])])
        return df


def _range_note(kind, fam, inp, t):
    r = RANGES.get((kind, fam))
    if r is None:
        return ""
    (alo, ahi), (tlo, thi) = r
    out = []
    if not alo <= inp.api <= ahi:
        out.append(f"API {inp.api:g} outside {alo:g}-{ahi:g}")
    if not tlo <= t <= thi:
        out.append(f"T {t:g} °F outside {tlo:g}-{thi:g}")
    return "; ".join(out)


def match_all(inp: PVTInputs, data: MatchData) -> MatchResult:
    matches = {}
    for f in PB_FAMILIES:
        if not available("pb", f):
            continue
        vals = correlation_values("pb", f, inp, data)
        anchors = stock_tank_anchors(f, inp)
        for prop in ("pb", "rs", "bo"):
            x, y = vals[prop]
            matches[(prop, f)] = fit_linear(x, y, anchor=anchors.get(prop))
    for f in VISC_FAMILIES:
        if not available("visc", f):
            continue
        x, y = correlation_values("visc", f, inp, data)["visc"]
        matches[("visc", f)] = fit_linear(x, y)

    res = MatchResult(matches, {}, {}, data.check(inp), inp, data.temperature)
    for kind, key, props in (("pb", "pb_family", ["pb", "rs", "bo"]), ("visc", "visc_family", ["visc"])):
        rk = res.ranking(kind)
        ok = rk[rk["Score"].notna()] if not rk.empty else rk
        if ok.empty:
            res.suggestion[key] = None
            res.rationale[key] = "No lab data for these properties - nothing to match."
            continue
        in_range = ok[ok["Status"] == "Within data range"]
        best = (in_range if not in_range.empty else ok).iloc[0]
        fams = VISC_FAMILIES if kind == "visc" else PB_FAMILIES
        fkey = next(k for k, v in fams.items() if v == best["Correlation"])
        res.suggestion[key] = fkey
        res.rationale[key] = _explain(res, kind, fkey, props, ok, best, in_range.empty)
    return res


def _explain(res, kind, fkey, props, ok, best, none_in_range):
    fams = VISC_FAMILIES if kind == "visc" else PB_FAMILIES
    parts = []
    for p in props:
        m = res.matches[(p, fkey)]
        if m.n == 0:
            continue
        sd = "" if np.isnan(m.std) else f", std dev {m.std:.4f}"
        parts.append(f"{PROPS[p]}: P1 {m.p1:.4f}, P2 {m.p2:.4g}{sd}")
    txt = (f"{fams[fkey]} needs the smallest correction to fit the lab data "
           f"(score {best['Score']:.4f}). " + "; ".join(parts) + ".")
    far = [PROPS[p] for p in props if res.matches[(p, fkey)].n and abs(res.matches[(p, fkey)].p1 - 1) > 0.10]
    if far:
        txt += f" Parameter 1 is outside 0.9-1.1 for {', '.join(far)}."
    if best["Correlation"] != ok.iloc[0]["Correlation"]:
        txt += (f" {ok.iloc[0]['Correlation']} scores lower but its dataset does not cover this oil "
                f"({ok.iloc[0]['Status']}).")
    if none_in_range:
        txt += " No correlation's dataset covers this oil's API and temperature; treat with care."
    if best["Score"] > 0.10:
        txt += " Even the best correlation needs a large correction - check the lab data."
    return txt


# ======================================================== matched model
@dataclass
class BlackOilModel:
    inp: PVTInputs
    family: str = "standing"
    visc_family: str = "beal"
    params: dict = field(default_factory=dict)   # prop -> LinearMatch (or absent = unmatched)

    def _m(self, prop):
        return self.params.get(prop, LinearMatch())

    def bubble_point(self, t):
        return float(self._m("pb").apply(_pb_corr(self.family, self.inp, t)))

    def table(self, t: float, pressures) -> pd.DataFrame:
        inp, fam, vf = self.inp, self.family, self.visc_family
        p = np.asarray(pressures, float)
        pb = self.bubble_point(t)
        above = p >= pb * (1 - 1e-9)
        rs = np.minimum(self._m("rs").apply(_rs_corr(fam, inp, p, t)), inp.rsb)
        rs = np.where(above, inp.rsb, np.maximum(rs, 0.0))
        bo = bo_sat(fam, inp, rs, t)
        bob = bo_sat(fam, inp, inp.rsb, t)
        bo = np.where(above, bob * bo_undersat_factor(fam, inp, p, pb, inp.rsb, t), bo)
        bo = self._m("bo").apply(bo)
        mod = visc_dead(vf, inp, t)
        mu = visc_live(vf, mod, rs)
        mu = np.where(p > pb, visc_undersat(vf, visc_live(vf, mod, inp.rsb), p, pb), mu)
        mu = self._m("visc").apply(mu)
        rho_o = (62.428 * inp.gamma_o + 0.0136 * rs * inp.gas_sg) / bo
        z, bg, rho_g, mu_g = gas_props(inp, p, t)
        bw, rho_w, mu_w = water_props(inp, p, t)
        return pd.DataFrame({
            "Pressure (psia)": p, "Solution GOR (scf/STB)": rs, "Oil FVF (RB/STB)": bo,
            "Oil viscosity (cp)": mu, "Oil density (lb/ft3)": rho_o, "Z factor": z,
            "Gas FVF (ft3/scf)": bg, "Gas viscosity (cp)": mu_g, "Gas density (lb/ft3)": rho_g, "Water FVF (RB/STB)": bw,
            "Water viscosity (cp)": mu_w, "Water density (lb/ft3)": rho_w,
        })


def example_data() -> tuple[PVTInputs, MatchData]:
    """Synthetic lab data for trying the tab - not a real fluid. Generated from a
    Standing-like GOR curve, a Vasquez-Beggs-like FVF and a Beal-like viscosity,
    each slightly offset, with measurement noise."""
    inp = PVTInputs(rsb=750, api=34.0, gas_sg=0.80, salinity_ppm=60000, h2s=0.0, co2=1.5, n2=0.5)
    tbl = pd.DataFrame({
        "Pressure": [4500, 3800, 3010, 2600, 2000, 1400, 800, 300],
        "Gas Oil Ratio": [750, 750, 750, 624, 454, 293, 148, 40],
        "Oil FVF": [1.394, 1.399, 1.426, 1.364, 1.285, 1.207, 1.137, 1.089],
        "Oil Viscosity": [0.532, 0.504, 0.478, 0.550, 0.671, 0.842, 1.097, 1.329],
    })
    return inp, MatchData(temperature=205.0, bubble_point=3010.0, table=tbl)
