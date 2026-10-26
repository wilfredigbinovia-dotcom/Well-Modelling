"""Complementary calculations along the flow path (PROSPER System | Solids / calculation Options):

- Maximum grain size diameter: the largest sand grain the flow can carry, from the critical transport velocity
  (Wasp, Kenny & Gandhi 1979, as given by PROSPER). Oil and water wells - a laminar-type equation in CGS units:

      V = [-3 N + (9 N^2 + g r^2 rho_f (rho_s - rho_f)(0.015476 + 0.19841 r))^0.5] / [rho_f (0.011607 + 0.14881 r)]

  (V cm/s = total no-slip velocity, N poise = mixture viscosity, g 980 cm/s2, r particle radius cm, densities
  g/cm3), solved for r. Gas wells - drag on a sphere, viscosity ignored:

      V = sqrt(4 (rho_s - rho_f) g d / (3 rho_f Cd)),  Cd = 0.85,  d ft, V ft/s, densities lb/ft3, g 32.2.

- Erosional velocity: API RP 14E, Ve = C / sqrt(rho_m) (ft/s, rho_m no-slip mixture density lb/ft3). With a sand
  production rate W > 0 also the ConocoPhillips sand-laden form Ve = S D sqrt(rho_m) / sqrt(W) (m/s; D mm,
  rho_m kg/m3, W kg/day) and the smaller of the two is used (PROSPER). The C value of the flow, Vm sqrt(rho_m), is
  reported too.

- Liquid loading (Turner, gas wells): Vt = C sigma^0.25 (rho_l - rho_g)^0.25 / rho_g^0.5 (ft/s; sigma dyne/cm, densities
  lb/ft3). C = 20.4 in Turner's paper (sigma in lbf/ft); PROSPER recommends 2.04 with sigma in dyne/cm. The gas
  velocity is the superficial gas velocity; the critical gas rate scales the free gas rate by Vt / Vsg.

- Erosion rate (DNV-RP-O501, 2015/2021): sand erosion of straight pipe (4.22), of bends from the deviation survey
  (4.28-4.36) and of reducers - tubing ID reductions and restrictions in the flow direction (4.50-4.57). The largest
  of the three is reported (PROSPER). Inputs: sand rate, wall material (Table 3-2) and deformation (ductile/brittle).
  Not PROSPER inputs, set here: particle diameter (DNV default 0.25 mm), geometry factor GF (DNV: 2 when unknown)
  and reducer angle.

- CO2 corrosion rate (NORSOK M-506:2017): CR = Kt fCO2^0.62 (S/19)^(0.146 + 0.0324 log fCO2) f(pH) (mm/y), with the
  15 and 5 degC forms, linear interpolation between the Table 1 temperatures; fCO2 = a y P (a fugacity coefficient,
  P limited to the bubble point for undersaturated oil); in-situ pH from the CO2 / bicarbonate / water equilibria
  (eq. 11, Henry and dissociation constants eq. 14-19) with the alkalinity as the bicarbonate buffer; wall shear
  stress eq. 21-22 with the oil/water emulsion viscosity eq. 23-26. Reported as PROSPER does:
  CR = IRC EFF + URC (1 - EFF), in 0.001 in/yr, and only where water and CO2 are present.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

G_CGS, G_FT = 980.0, 32.2
LBFT3_PER_GCC = 62.428
KGM3_PER_LBFT3 = 16.0185
LB_PER_KG = 2.20462
FT_PER_M = 3.28084
CD_SAND = 0.85
GRAIN_CAP_IN = 1.0         # PROSPER caps the reported grain diameter at 1 in


@dataclass
class CompOptions:
    sand_density: float = 2.65        # g/cm3
    sand_rate: float = 0.0            # lbm/day (0: API 14E only)
    c_factor: float = 400.0           # API 14E C (PROSPER default)
    s_factor: float = 0.05            # ConocoPhillips S (pipe geometry)
    turner: float = 2.04              # Turner constant (PROSPER recommends 2.04, sigma in dyne/cm)
    wall: str = "Carbon steel"        # DNV-RP-O501 Table 3-2
    deformation: str = "Ductile"      # Ductile / Brittle angle dependency
    particle_mm: float = 0.25         # sand particle diameter for the bend model (DNV default)
    geometry_factor: float = 2.0      # DNV GF (2 when unknown)
    reducer_angle: float = 45.0       # deg, reducer contraction angle
    inhibited_rate: float = 0.0       # 0.001 in/yr (PROSPER IRC)
    inhibitor_eff: float = 0.0        # percent (PROSPER EFF)
    alkalinity: float = 0.0           # kg/kg bicarbonate in the water (PROSPER pH buffer)
    hydrate: tuple | None = None      # hydrate formation curve ((psia...), (F...)) for the hydrate flag (PVT warnings)


# ------------------------------------------------------------------ maximum grain size
def _v_oil_cgs(r, n_poise, rho_f, rho_s):
    num = -3.0 * n_poise + np.sqrt(9.0 * n_poise ** 2 + G_CGS * r ** 2 * rho_f * (rho_s - rho_f)
                                   * (0.015476 + 0.19841 * r))
    return num / (rho_f * (0.011607 + 0.14881 * r))


def grain_diameter_oil(v_fts, mu_cp, rho_lbft3, rho_s_gcc=2.65):
    """Largest grain diameter (in) the flow carries - oil and water wells. v total no-slip velocity (ft/s), mu
    mixture viscosity (cp), rho mixture density (lb/ft3)."""
    v = float(v_fts) * 30.48
    n = float(mu_cp) / 100.0
    rf = float(rho_lbft3) / LBFT3_PER_GCC
    if not (np.isfinite(v) and v > 0 and np.isfinite(n) and n > 0 and 0 < rf < rho_s_gcc):
        return np.nan
    f = lambda r: _v_oil_cgs(r, n, rf, rho_s_gcc) - v
    lo, hi = 1e-8, 1.0
    while f(hi) < 0 and hi < 1e4:
        hi *= 10.0
    if f(hi) < 0:
        return np.nan
    return 2.0 * brentq(f, lo, hi, xtol=1e-12, rtol=1e-10) / 2.54


def grain_diameter_gas(v_fts, rho_lbft3, rho_s_gcc=2.65, cd=CD_SAND):
    """Largest grain diameter (in) the flow carries - gas wells."""
    rho_s = rho_s_gcc * LBFT3_PER_GCC
    rho = float(rho_lbft3)
    if not (np.isfinite(v_fts) and v_fts > 0 and 0 < rho < rho_s):
        return np.nan
    return 3.0 * rho * cd * float(v_fts) ** 2 / (4.0 * (rho_s - rho) * G_FT) * 12.0


# ------------------------------------------------------------------ erosional velocity
def erosional_velocity(rho_m_lbft3, d_in, opts: CompOptions):
    """Erosional velocity (ft/s): API 14E, or with sand the smaller of API 14E and ConocoPhillips."""
    rho = max(float(rho_m_lbft3), 1e-9)
    vef = opts.c_factor / np.sqrt(rho)
    if opts.sand_rate > 0:
        w_kgd = opts.sand_rate / LB_PER_KG
        ves = opts.s_factor * float(d_in) * 25.4 * np.sqrt(rho * KGM3_PER_LBFT3) / np.sqrt(w_kgd) * FT_PER_M
        return min(vef, ves)
    return vef


# ------------------------------------------------------------------ liquid loading
def turner_velocity(sig_dyncm, rho_l, rho_g, c=2.04):
    """Turner minimum gas velocity (ft/s) to lift liquid droplets."""
    if not (np.isfinite(sig_dyncm) and np.isfinite(rho_l) and np.isfinite(rho_g) and rho_g > 0 and rho_l > rho_g):
        return np.nan
    return c * sig_dyncm ** 0.25 * (rho_l - rho_g) ** 0.25 / np.sqrt(rho_g)


# ------------------------------------------------------------------ PE5-style stability
# PROSPER's PE5 stability flag (Tutorial 43): "unstable" where the PE5 momentum balance has the liquid flowing back
# down. Not published; reproduced as a liquid-film reversal (flooding) criterion: unstable where the Kutateladze
# number of the gas Ku = vsg rho_g^0.5 / (g sigma (rho_l - rho_g))^0.25 is below 1.637 D*^0.32, D* = D (g (rho_l -
# rho_g) / sigma)^0.5 the Bond diameter. Calibrated on the user's T43 file (black-oil condensate, WGR 50): every VLP
# flag at 1000 and 100 psig (unstable to 10.5-10.7 MMscf/d, stable from 12-15.3, set at the 6.4 in casing at the
# bottom) and the gradient's node flags at 4 MMscf/d (casing unstable; the tubing, stable in PROSPER, sits at the
# threshold here - ratio 0.999). Liquid-continuous flow (no-slip
# liquid fraction above 0.5) is reported stable.
PE5_STAB = (1.637, 0.32)
SIG_POUNDAL = 6.852e-5 * 32.174     # dyne/cm -> poundal/ft


def pe5_stability_ratio(vsl, vsg, rho_l, rho_g, sig_dyncm, d_ft):
    """Ku / Ku_critical (> 1 stable). nan where not applicable (no gas, liquid-continuous, single phase)."""
    vm = vsl + vsg
    if not (vm > 0 and vsg > 0 and vsl > 0 and rho_l > rho_g > 0 and sig_dyncm > 0) or vsl / vm > 0.5:
        return np.nan
    sp = sig_dyncm * SIG_POUNDAL
    ku = vsg * np.sqrt(rho_g) / (G_FT * sp * (rho_l - rho_g)) ** 0.25
    ds = d_ft * np.sqrt(G_FT * (rho_l - rho_g) / sp)
    return float(ku / (PE5_STAB[0] * ds ** PE5_STAB[1]))


def hydrate_flag(p_psia, t_f, curve):
    """True where (p, t) lies on the hydrate side of the curve ((psia...), (F...)): colder than the hydrate
    temperature at that pressure."""
    if curve is None:
        return False
    pc, tc = (np.asarray(x, float) for x in curve)
    o = np.argsort(pc)
    return bool(t_f <= np.interp(p_psia, pc[o], tc[o]))


# ------------------------------------------------------------------ per node
def node_values(kind, vm, vsg, rho_ns, mu_ns, rho_l, rho_g, sig, d_in, free_gas_mmscfd, opts: CompOptions):
    """Complementary results at one node. kind 'oil' or 'gas' selects the grain-size equation."""
    dmax = (grain_diameter_gas(vm, rho_ns, opts.sand_density) if kind == "gas"
            else grain_diameter_oil(vm, mu_ns, rho_ns, opts.sand_density))
    if np.isfinite(dmax):
        dmax = min(dmax, GRAIN_CAP_IN)            # PROSPER reports at most 1 inch
    ve = erosional_velocity(rho_ns, d_in, opts)
    has_gas = vsg > 0 and np.isfinite(rho_g)
    # Turner is a gas-well criterion (droplets carried by the gas core); not reported for oil wells
    vt = turner_velocity(sig, rho_l, rho_g, opts.turner) if kind == "gas" and has_gas and vsg < vm else np.nan
    q_crit = free_gas_mmscfd * vt / vsg if (np.isfinite(vt) and vsg > 0 and np.isfinite(free_gas_mmscfd)) else np.nan
    stab = pe5_stability_ratio(vm - vsg, vsg, rho_l, rho_g, sig, d_in / 12.0)
    return {"Turner Velocity Flag": bool(np.isfinite(vt) and vm < vt),
            "Erosional Velocity Flag": bool(ve > 0 and vm > ve),
            "PE5 Stability Ratio": stab,
            "Max. Grain Size Diameter (in)": dmax,
            "Erosional Velocity (ft/s)": ve,
            "Velocity / Erosional Velocity": vm / ve if ve > 0 else np.nan,
            "C Factor (flow)": vm * np.sqrt(max(rho_ns, 0.0)),
            "Liquid Loading Velocity (ft/s)": vt,
            "Gas Velocity / Loading Velocity": vsg / vt if np.isfinite(vt) and vt > 0 else np.nan,
            "Critical Gas Rate (MMscf/day)": q_crit}


def summary(rows) -> dict:
    """Worst values over the flow path (for system / VLP results), from node_values dicts."""
    def col(k):
        a = np.array([r[k] for r in rows], float)
        return a[np.isfinite(a)]
    out = {}
    d = col("Max. Grain Size Diameter (in)")
    if d.size:
        out["Max. grain diameter (in)"] = float(d.max())
        out["Min. grain diameter (in)"] = float(d.min())
    r = col("Velocity / Erosional Velocity")
    if r.size:
        out["Max. velocity / erosional velocity"] = float(r.max())
        for k_, lab_, fn_ in (("Erosional Velocity (ft/s)", "Min. erosional velocity (ft/s)", np.min),
                              ("C Factor (flow)", "Max. C factor (flow)", np.max),
                              ("C Factor (flow)", "Min. C factor (flow)", np.min)):
            a_ = col(k_)
            if a_.size:
                out[lab_] = float(fn_(a_))
    t = col("Gas Velocity / Loading Velocity")
    if t.size:
        out["Min. gas velocity / Turner velocity"] = float(t.min())
        out["Liquid loading (Turner)"] = "Loading" if t.min() < 1.0 else "No loading"
        qc = col("Critical Gas Rate (MMscf/day)")
        if qc.size:
            out["Max. Turner critical gas rate (MMscf/d)"] = float(qc.max())
    if rows and "Turner Velocity Flag" in rows[0]:
        lv = col("Liquid Loading Velocity (ft/s)")
        if lv.size:
            out["Max. Turner velocity (ft/s)"] = float(lv.max())
            out["Min. Turner velocity (ft/s)"] = float(lv.min())
        out["Erosional velocity flag"] = "Yes" if any(r["Erosional Velocity Flag"] for r in rows) else "No"
        out["Turner velocity flag"] = "Yes" if any(r["Turner Velocity Flag"] for r in rows) else "No"
        st_ = col("PE5 Stability Ratio")
        out["PE5 stability flag"] = "Unstable" if st_.size and st_.min() < 1.0 else "Stable"
        if st_.size:
            out["Min. PE5 stability ratio"] = float(st_.min())
        if any("Hydrate Formation" in r for r in rows):
            out["Hydrates formation flag"] = ("Hydrates" if any(r.get("Hydrate Formation") for r in rows)
                                              else "No hydrates")
    if rows and "Erosion Rate (mm/year)" in rows[0]:
        e = col("Erosion Rate (mm/year)")
        if e.size:
            k = int(np.nanargmax([r["Erosion Rate (mm/year)"] if np.isfinite(r["Erosion Rate (mm/year)"]) else -1
                                  for r in rows]))
            out["Max. erosion rate (mm/year)"] = float(e.max())
            out["Max. erosion rate geometry"] = rows[k]["Erosion Geometry"]
        c = col("Corrosion Rate (0.001 in/yr)")
        out["Max. corrosion rate (0.001 in/yr)"] = float(c.max()) if c.size else 0.0
        out["Min. corrosion rate (0.001 in/yr)"] = float(c.min()) if c.size else 0.0
    return out


# ================================================================== DNV-RP-O501 erosion
# material: (rho_t kg/m3, K (m/s)^-n, n, angle dependency)
WALL_MATERIALS = {
    "Carbon steel": (7800.0, 2.0e-9, 2.6, "Ductile"), "Duplex": (7850.0, 2.0e-9, 2.6, "Ductile"),
    "SS316": (8000.0, 2.0e-9, 2.6, "Ductile"), "Inconel": (8440.0, 2.0e-9, 2.6, "Ductile"),
    "GRP/epoxy": (1800.0, 3.0e-10, 3.6, "Ductile"), "GRP/vinyl ester": (1800.0, 6.0e-10, 3.6, "Ductile"),
    "HDPE": (1150.0, 3.5e-9, 2.9, "Ductile"), "Aluminium": (2700.0, 5.8e-9, 2.3, "Ductile"),
    "Tungsten carbide DC-05": (15250.0, 1.1e-10, 2.3, "Brittle"),
    "Tungsten carbide CS-10": (14800.0, 3.2e-10, 2.2, "Brittle"),
    "Tungsten carbide CR-37": (14600.0, 8.8e-11, 2.5, "Brittle"),
    "Aluminium oxide 95%": (3700.0, 6.8e-8, 2.0, "Brittle"), "Aluminium oxide 99.5%": (3700.0, 9.5e-7, 1.2, "Brittle"),
    "Zirconia PSZ": (5700.0, 4.1e-9, 2.5, "Brittle"), "Zirconia ZrO2-Y3": (6070.0, 4.0e-11, 2.7, "Brittle"),
    "Silicon carbide": (3100.0, 6.5e-9, 1.9, "Brittle"), "Silicon nitride": (3200.0, 2.0e-10, 2.0, "Brittle"),
    "Titanium diboride": (4520.0, 9.3e-9, 1.9, "Brittle"),
}
C_UNIT = 3.15e10          # m/s -> mm/year
RHO_SAND = 2650.0


def angle_function(alpha, deformation="Ductile"):
    """DNV F(alpha), alpha in radians (3.3 ductile, 3.4 brittle)."""
    if deformation == "Brittle":
        return 2.0 * alpha / np.pi
    A, B, C, k = 0.6, 7.2, 20.0, 0.6
    sa = np.sin(alpha)
    return A * (sa + B * (sa - sa * sa)) ** k * (1.0 - np.exp(-C * alpha))


def _material(opts):
    rho_t, K, n, _ = WALL_MATERIALS.get(opts.wall, WALL_MATERIALS["Carbon steel"])
    return rho_t, K, n


def erosion_straight(up, d_m, mp_kgs, opts):
    """Smooth straight pipe (DNV 4.22), mm/year. The correlation is for steel; other materials are scaled by
    K Up^n / rho_t relative to steel."""
    rho_t, K, n = _material(opts)
    scale = (K / rho_t * up ** n) / (2.0e-9 / 7800.0 * up ** 2.6) if up > 0 else 1.0
    return 2.5e-5 * up ** 2.6 * d_m ** -2 * mp_kgs * scale


def erosion_bend(up, d_m, r_diam, rho_m, mu_m, dp_m, mp_kgs, opts):
    """Pipe bend (DNV 4.28-4.36), mm/year. r_diam bend radius in pipe diameters, rho_m kg/m3, mu_m Pa s."""
    rho_t, K, n = _material(opts)
    alpha = np.arctan(1.0 / np.sqrt(2.0 * r_diam))
    a = rho_m ** 2 * np.tan(alpha) * up * d_m / (RHO_SAND * mu_m)
    beta = RHO_SAND / rho_m
    gc = 1.0 / (beta * (1.88 * np.log(a) - 6.04)) if a > np.exp(6.04 / 1.88) else 0.1
    gc = 0.1 if not (0 < gc < 0.1) else gc
    gamma = dp_m / d_m
    g = gamma / gc if gamma < gc else 1.0
    at = np.pi * d_m ** 2 / (4.0 * np.sin(alpha))
    return (K * angle_function(alpha, opts.deformation) * up ** n / (rho_t * at) * g * 2.5 * opts.geometry_factor
            * mp_kgs * C_UNIT)


def erosion_reducer(vm1, d1_m, d2_m, dp_m, mp_kgs, rho_m, opts):
    """Reducer (DNV 4.50-4.57), mm/year: contraction from d1 to d2 in the flow direction."""
    rho_t, K, n = _material(opts)
    alpha = np.radians(opts.reducer_angle)
    up = vm1 * (d1_m / d2_m) ** 2
    at = np.pi / (4.0 * np.sin(alpha)) * (d1_m ** 2 - d2_m ** 2)
    ar = 1.0 - (d2_m / d1_m) ** 2
    c2 = min(1.0, 1e6 * dp_m / (30.0 * rho_m ** 0.5))
    return K * angle_function(alpha, opts.deformation) * up ** n / (rho_t * at) * ar * c2 * opts.geometry_factor \
        * mp_kgs * C_UNIT


# ================================================================== NORSOK M-506 CO2 corrosion
NORSOK_T = np.array([5.0, 15.0, 20.0, 40.0, 60.0, 80.0, 90.0, 120.0, 150.0])
NORSOK_KT = np.array([0.42, 1.59, 4.762, 8.927, 10.695, 9.949, 6.250, 7.770, 5.203])
MPY_PER_MMY = 39.3701


def f_ph(t_tab, ph):
    """NORSOK Table 2 pH function at table temperature t_tab (degC)."""
    if t_tab in (5.0, 15.0, 20.0, 40.0):
        if ph < 4.6:
            return 2.0676 - 0.2309 * ph
        c = {5.0: (4.342, 1.051), 15.0: (4.986, 1.191), 20.0: (5.1885, 1.2353), 40.0: (5.1885, 1.2353)}[t_tab]
        return c[0] - c[1] * ph + 0.0708 * ph ** 2
    if t_tab == 60.0:
        return 1.836 - 0.1818 * ph if ph < 4.6 else 15.444 - 6.1291 * ph + 0.8204 * ph ** 2 - 0.0371 * ph ** 3
    if t_tab == 80.0:
        return 2.6727 - 0.3636 * ph if ph < 4.6 else 331.68 * np.exp(-1.2618 * ph)
    if t_tab == 90.0:
        if ph < 4.57:
            return 3.1355 - 0.4673 * ph
        return 21254.0 * np.exp(-2.1811 * ph) if ph < 5.62 else 0.4014 - 0.0538 * ph
    if t_tab == 120.0:
        if ph < 4.3:
            return 1.5375 - 0.125 * ph
        return 5.9757 - 1.157 * ph if ph < 5.0 else 0.546125 - 0.071225 * ph
    if ph < 3.8:                                     # 150 degC
        return 1.0
    return 17.634 - 7.0945 * ph + 0.715 * ph ** 2 if ph < 5.0 else 0.037


def _cr_at(j, fco2, shear, ph):
    t = NORSOK_T[j]
    kt = NORSOK_KT[j]
    if t == 5.0:
        return kt * fco2 ** 0.36 * f_ph(t, ph)
    e = 0.36 if t == 15.0 else 0.62
    return kt * fco2 ** e * (shear / 19.0) ** (0.146 + 0.0324 * np.log10(fco2)) * f_ph(t, ph)


def norsok_rate(t_c, fco2_bar, shear_pa, ph):
    """Uninhibited CO2 corrosion rate (mm/y). Inputs kept in the model's validity ranges: T 5-150 degC,
    shear 1-150 Pa, pH 3.5-6.5 (CR linear in temperature between the Table 1 temperatures)."""
    if not (np.isfinite(fco2_bar) and fco2_bar > 0):
        return 0.0
    t = float(np.clip(t_c, 5.0, 150.0))
    s = float(np.clip(shear_pa, 1.0, 150.0))
    ph = float(np.clip(ph, 3.5, 6.5))
    j = int(np.clip(np.searchsorted(NORSOK_T, t) - 1, 0, len(NORSOK_T) - 2))
    lo, hi = _cr_at(j, fco2_bar, s, ph), _cr_at(j + 1, fco2_bar, s, ph)
    w = (t - NORSOK_T[j]) / (NORSOK_T[j + 1] - NORSOK_T[j])
    return max(lo + w * (hi - lo), 0.0)


def fugacity_coefficient(p_bar, t_k):
    """NORSOK eq. 7-8 (P limited to 250 bar)."""
    return 10.0 ** (min(p_bar, 250.0) * (0.0031 - 1.4 / t_k))


def _equilibria(t_c, p_psi, ionic):
    """NORSOK eq. 14-19: KH (molar/bar), KA = KH Khyd Kca, KB = KA Kbi, Kw (molar^2)."""
    tk, tf = t_c + 273.15, t_c * 1.8 + 32.0
    pterm = 10.0 ** -(1.79e-5 * p_psi + 0.107 * ionic)
    if t_c <= 80.0:
        kh = 55.5084 * np.exp(-(4.8 + 3934.4 / tk - 941290.2 / tk ** 2)) * pterm
    else:
        kh = 55.5084 * np.exp(-(1713.53 * (1 - tk / 647.0) ** (1 / 3) / tk + 3.875 + 3680.09 / tk
                                - 1198506.1 / tk ** 2)) * pterm
    si = np.sqrt(ionic)
    # pK of the combined CO2(aq)+H2CO3 first dissociation; Kca = K1 / Khyd so KA = KH Khyd Kca = KH K1
    pk1 = (356.3094 + 0.06091964 * tk - 21834.37 / tk - 126.8339 * np.log10(tk) + 1684915.0 / tk ** 2
           - 2.564e-5 * p_psi - 0.491 * si + 0.379 * ionic - 0.06506 * ionic ** 1.5 - 1.458e-3 * ionic * tf)
    pk2 = (107.8871 + 0.03252849 * tk - 5151.79 / tk - 38.92561 * np.log10(tk) + 563713.9 / tk ** 2
           - 2.118e-5 * p_psi - 1.255 * si + 0.867 * ionic - 0.174 * ionic ** 1.5 - 1.588e-3 * ionic * tf)
    pkw = 29.3868 - 0.0737549 * tk + 7.47881e-5 * tk ** 2
    ka = kh * 10.0 ** -pk1
    return kh, ka, ka * 10.0 ** -pk2, 10.0 ** -pkw


def norsok_ph(t_c, p_psi, fco2_bar, alk_molar=0.0, ionic=0.0):
    """In-situ pH of produced water (NORSOK eq. 11, no acetate): H^3 + alk H^2 - (Kw + KA p) H - 2 KB p = 0."""
    _, ka, kb, kw = _equilibria(t_c, p_psi, ionic)
    p = max(fco2_bar, 0.0)
    f = lambda x: (lambda h: h ** 3 + alk_molar * h ** 2 - (kw + ka * p) * h - 2.0 * kb * p)(10.0 ** -x)
    try:
        return brentq(f, 0.0, 14.0, xtol=1e-8)
    except ValueError:
        return np.nan


def water_viscosity_norsok(t_c):
    """NORSOK eq. 25-26 (Pa s)."""
    if t_c <= 20.0:
        return 10.0 ** (1301.0 / (998.333 + 8.1855 * (t_c - 20) + 0.00585 * (t_c - 20) ** 2) - 1.30233) * 1e-3
    return 1.002 * 10.0 ** ((1.3272 * (20 - t_c) - 0.001053 * (t_c - 20) ** 2) / (t_c + 105)) * 1e-3


def emulsion_viscosity(mu_o, mu_w, phi, phi_c=0.5, mu_rel_max=7.06):
    """NORSOK eq. 23-24: oil/water dispersion viscosity (same unit as mu_o, mu_w); phi water cut (fraction)."""
    if phi <= 0:
        return mu_o
    if phi >= 1:
        return mu_w
    if phi <= phi_c:
        b0 = phi_c / (1.187 * (1 - (1 / mu_rel_max) ** 0.4))
        x = phi / b0
        return mu_o * (1 + x / (1.187 - x)) ** 2.5
    r = mu_w / mu_o
    if r >= mu_rel_max:
        return mu_w
    bw = (1 - phi_c) / (1.187 * (1 - (r / mu_rel_max) ** 0.4))
    x = (1 - phi) / bw
    return mu_w * (1 + x / (1.187 - x)) ** 2.5


def wall_shear(rho_m, mu_m, u_m, d_m, k_m):
    """NORSOK eq. 21-22 (Pa); SI inputs."""
    if not (u_m > 0 and rho_m > 0 and mu_m > 0):
        return 0.0
    f = 0.001375 * (1 + (20000 * k_m / d_m + 1e6 * mu_m / (rho_m * u_m * d_m)) ** 0.33)
    return 0.5 * rho_m * f * u_m ** 2


# ================================================================== along the flow path
def _pvt_inputs(fluid):
    return getattr(fluid, "inp", None) or getattr(fluid, "winp", None)


def path_values(path, i, fluid, kind, case, s, p_psia, t_f, opts: CompOptions):
    """Erosion (DNV) and corrosion (NORSOK) for flow-path segment i (0 = top). s is fluid.state at (p, t) for
    the segment; case the 1-element rate case."""
    out = {"Erosion Rate (mm/year)": np.nan, "Erosion Geometry": "", "Corrosion Rate (0.001 in/yr)": np.nan}
    a = path.area[i]
    vsl = max(float(s["ql"][0]) / a, 0.0)
    vsg = max(float(s["qg"][0]) / a, 0.0)
    vm = vsl + vsg
    if vm <= 0:
        return out
    rho_l = float(s["rho_l"][0]) * KGM3_PER_LBFT3
    rho_g = float(s["rho_g"][0]) * KGM3_PER_LBFT3 if vsg > 0 else 0.0
    mu_l, mu_g = float(s["mu_l"][0]) * 1e-3, float(s["mu_g"][0]) * 1e-3 if vsg > 0 else 0.0
    lam = vsl / vm
    rho_m = lam * rho_l + (1 - lam) * rho_g                      # DNV 4.9 / NORSOK mixture density
    up = vm / FT_PER_M
    d_m = path.d_h[i] * 0.3048

    # ---- erosion
    if opts.sand_rate > 0:
        mp = opts.sand_rate / LB_PER_KG / 86400.0
        mu_m = lam * mu_l + (1 - lam) * mu_g
        cand = {"Straight pipe": erosion_straight(up, d_m, mp, opts)}
        if i > 0:                                                # bend at the top node of the segment
            th = np.arccos(np.clip(path.sin_t[[i - 1, i]], -1, 1))
            dth = abs(th[0] - th[1])
            if dth > 1e-4:
                radius = 0.5 * (path.seg_len[i - 1] + path.seg_len[i]) * 0.3048 / dth
                cand["Bend"] = erosion_bend(up, d_m, radius / d_m, rho_m, mu_m, opts.particle_mm * 1e-3, mp, opts)
            if path.d_h[i - 1] < path.d_h[i] * 0.999:             # ID reduction in the flow direction
                cand["Reducer"] = erosion_reducer(up, d_m, path.d_h[i - 1] * 0.3048, opts.particle_mm * 1e-3, mp,
                                                  rho_m, opts)
        if i in path.restrictions and path.restrictions[i] * 0.0254 < d_m:
            cand["Reducer (restriction)"] = erosion_reducer(up, d_m, path.restrictions[i] * 0.0254,
                                                            opts.particle_mm * 1e-3, mp, rho_m, opts)
        g = max(cand, key=cand.get)
        out["Erosion Rate (mm/year)"], out["Erosion Geometry"] = cand[g], g

    # ---- corrosion
    inp = _pvt_inputs(fluid)
    co2 = getattr(inp, "co2", 0.0) if inp is not None else 0.0
    water = float(np.atleast_1d(case.get("wc", case.get("wgr", 0.0)))[0]) > 0
    if co2 > 0 and water:
        t_c = (t_f - 32.0) / 1.8
        p_co2 = p_psia
        gor = np.atleast_1d(case.get("gor", np.nan))
        if kind == "oil" and hasattr(fluid, "pb_at") and np.isfinite(gor[0]):
            p_co2 = min(p_psia, float(fluid.pb_at(np.array([t_f]), gor)[0]))   # undersaturated: CO2 at Pb
        p_bar = p_co2 / 14.5038
        fco2 = fugacity_coefficient(p_bar, t_c + 273.15) * co2 / 100.0 * p_bar
        sal = getattr(inp, "salinity_ppm", 0.0) or 0.0
        ionic = sal / 1000.0 / 58.443                                 # NaCl molarity
        alk = opts.alkalinity * 1e6 / 61.017 / 1000.0                 # kg/kg HCO3- -> mol/l
        ph = norsok_ph(t_c, p_psia, fco2, alk, ionic)
        # liquid viscosity: oil/water dispersion (NORSOK) when oil and water viscosities are known
        mu_liq = mu_l
        wc = float(np.atleast_1d(case.get("wc", 0.0))[0]) if kind == "oil" else 1.0
        if kind == "oil" and hasattr(fluid, "oil_water_viscosity"):
            mo, mw = fluid.oil_water_viscosity(np.array([p_psia]), np.array([t_f]), gor)
            mu_liq = emulsion_viscosity(float(mo[0]), float(mw[0]), wc) * 1e-3
        mu_mix = lam * mu_liq + (1 - lam) * mu_g
        shear = wall_shear(rho_m, mu_mix, up, d_m, path.rough[i] * 0.3048)
        urc = norsok_rate(t_c, fco2, shear, ph) * MPY_PER_MMY
        eff = np.clip(opts.inhibitor_eff / 100.0, 0, 1)
        out.update({"Corrosion Rate (0.001 in/yr)": opts.inhibited_rate * eff + urc * (1 - eff),
                    "CO2 Fugacity (psia)": fco2 * 14.5038, "In-situ pH": ph, "Wall Shear Stress (Pa)": shear})
    return out
