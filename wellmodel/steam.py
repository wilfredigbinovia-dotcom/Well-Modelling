"""Steam injection (PROSPER's "Allow Steam Calculations" with the Improved Approximation; Tutorial 42).

Water and steam properties from the IAPWS-95 formulation (CoolProp HEOS 'Water'): PROSPER's steam calculator gives
the same saturation temperature (650.453 F at 2200 psig), latent heat (422.233 BTU/lb), saturated densities and
enthalpies (IAPWS reference: liquid at the triple point = 0) and the superheated state at 1200 BTU/lb (689.45 F,
4.928 lb/ft3, cp 1.5225 BTU/lb/F) to the digits it shows; viscosities agree to 0.1-0.3 % and thermal conductivity to
3 %.

The injector traverse marches the pressure and the flowing enthalpy down from the wellhead together:
    h(i+1) = h(i) + L sin(theta)/J - U pi d (T - Te) L / w
with T and the steam quality from (p, h); the pressure gradient is gravity minus friction from the multiphase
correlation (steam = gas phase, water = liquid phase), single-phase when the quality is 0 or 1.

Requires the CoolProp package.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PA_PER_PSI = 6894.757
J_PER_KG_PER_BTU_LB = 2326.0
LB_FT3_PER_KG_M3 = 0.0624280
J = 778.17                       # ft lbf / BTU
CONDUCTIVITY = 0.577789          # BTU/h/ft/F per W/m/K


def available():
    try:
        import CoolProp  # noqa: F401
        return True
    except ImportError:
        return False


def _state():
    import CoolProp.CoolProp as CP
    return CP, CP.AbstractState("HEOS", "Water")


def _f(t_k):
    return (t_k - 273.15) * 1.8 + 32.0


def _k(t_f):
    return (t_f - 32.0) / 1.8 + 273.15


def _single(AS, CP):
    out = {"density": AS.rhomass() * LB_FT3_PER_KG_M3, "enthalpy": AS.hmass() / J_PER_KG_PER_BTU_LB,
           "viscosity": AS.viscosity() * 1000.0, "cp": AS.cpmass() / 4186.8, "cv": AS.cvmass() / 4186.8}
    try:
        out["jt"] = AS.first_partial_deriv(CP.iT, CP.iP, CP.iHmass) * 1.8 * PA_PER_PSI
    except ValueError:
        out["jt"] = np.nan
    try:
        out["conductivity"] = AS.conductivity() * CONDUCTIVITY
    except ValueError:
        out["conductivity"] = np.nan
    return out


def saturation(p_psia):
    """Saturated liquid and vapour at pressure p (psia): dict with t_sat (F), latent heat and per-phase properties."""
    CP, AS = _state()
    p = float(p_psia) * PA_PER_PSI
    AS.update(CP.PQ_INPUTS, p, 0.0)
    liq = _single(AS, CP)
    t = _f(AS.T())
    try:
        sig = AS.surface_tension() * 1000.0            # dyn/cm
    except ValueError:
        sig = np.nan
    AS.update(CP.PQ_INPUTS, p, 1.0)
    vap = _single(AS, CP)
    return {"t_sat": t, "latent": vap["enthalpy"] - liq["enthalpy"], "liquid": liq, "vapour": vap, "sigma": sig}


def critical_pressure():
    CP, AS = _state()
    return AS.p_critical() / PA_PER_PSI


def flash_ph(p_psia, h):
    """State at pressure p (psia) and enthalpy h (BTU/lb): dict phase, t (F), quality (0 liquid, 1 vapour), and the
    single-phase properties when single phase."""
    CP, AS = _state()
    p = float(p_psia) * PA_PER_PSI
    AS.update(CP.HmassP_INPUTS, float(h) * J_PER_KG_PER_BTU_LB, p)
    q = AS.Q()
    t = _f(AS.T())
    if 0.0 <= q <= 1.0 and p < AS.p_critical():
        return {"phase": "TWO PHASE", "t": t, "quality": q}
    ph = AS.phase()
    vapour = ph in (CP.iphase_gas, CP.iphase_supercritical_gas) or (
        p < AS.p_critical() and float(h) > saturation(p_psia)["vapour"]["enthalpy"])
    if ph in (CP.iphase_supercritical, CP.iphase_supercritical_liquid, CP.iphase_supercritical_gas):
        name = "SUPERCRITICAL"
    else:
        name = "VAPOUR" if vapour else "LIQUID"
    out = {"phase": name, "t": t, "quality": 1.0 if vapour else 0.0}
    out.update(_single(AS, CP))
    return out


def calculator(p_psia, h=None, t_f=None):
    """PROSPER's steam calculator (single point): from pressure and enthalpy (or temperature). Returns a table."""
    CP, AS = _state()
    if h is None:
        AS.update(CP.PT_INPUTS, float(p_psia) * PA_PER_PSI, _k(float(t_f)))
        h = AS.hmass() / J_PER_KG_PER_BTU_LB
    st = flash_ph(p_psia, h)
    rows = [("Phase", st["phase"]), ("Calculated temperature (°F)", st["t"]), ("Enthalpy (BTU/lb)", float(h))]
    for k, lab in (("density", "Density (lb/ft3)"), ("viscosity", "Viscosity (cp)"), ("cp", "Cp (BTU/lb/F)"),
                   ("cv", "Cv (BTU/lb/F)"), ("jt", "JT coefficient (F/psi)"),
                   ("conductivity", "Thermal conductivity (BTU/h/ft/F)")):
        if k in st:
            rows.append((lab, st[k]))
    if float(p_psia) < critical_pressure():
        s = saturation(p_psia)
        rows += [("Saturation temperature (°F)", s["t_sat"]), ("Latent heat (BTU/lb)", s["latent"]),
                 ("Steam quality", st["quality"])]
        for k, lab in (("density", "Saturated density (lb/ft3)"), ("enthalpy", "Saturated enthalpy (BTU/lb)"),
                       ("viscosity", "Saturated viscosity (cp)"), ("cp", "Saturated Cp (BTU/lb/F)"),
                       ("cv", "Saturated Cv (BTU/lb/F)"), ("jt", "Saturated JT coefficient (F/psi)"),
                       ("conductivity", "Saturated thermal conductivity (BTU/h/ft/F)")):
            rows.append((lab + " - liquid", s["liquid"][k]))
            rows.append((lab + " - vapour", s["vapour"][k]))
    return pd.DataFrame(rows, columns=["Result", "Value"])


def saturation_table(p_list_psia) -> pd.DataFrame:
    """PROSPER's 'Calculate Range': saturation properties over a pressure range."""
    recs = []
    for p in p_list_psia:
        s = saturation(p)
        recs.append({"Pressure (psia)": p, "Saturation temperature (°F)": s["t_sat"], "Latent heat (BTU/lb)": s["latent"],
                     "Liquid density (lb/ft3)": s["liquid"]["density"], "Vapour density (lb/ft3)": s["vapour"]["density"],
                     "Liquid enthalpy (BTU/lb)": s["liquid"]["enthalpy"],
                     "Vapour enthalpy (BTU/lb)": s["vapour"]["enthalpy"],
                     "Liquid viscosity (cp)": s["liquid"]["viscosity"], "Vapour viscosity (cp)": s["vapour"]["viscosity"]})
    return pd.DataFrame(recs)


def enthalpy_of(p_psia, quality=None, t_f=None):
    """Wellhead enthalpy (BTU/lb) from a steam quality (at saturation) or a temperature (single phase)."""
    CP, AS = _state()
    if quality is not None:
        AS.update(CP.PQ_INPUTS, float(p_psia) * PA_PER_PSI, float(quality))
    else:
        AS.update(CP.PT_INPUTS, float(p_psia) * PA_PER_PSI, _k(float(t_f)))
    return AS.hmass() / J_PER_KG_PER_BTU_LB


# PROSPER's liquid phase in a steam injector is its brine, not saturated water from the steam tables (Tutorial 42,
# 200,000 ppm): density = stock-tank brine density / Bw with Bw = (1 + 1.24139 dVwt)(1 + 2.19084 dVwp) (McCain's
# temperature and pressure terms, rescaled: PROSPER's Bw within 0.03 % at 650-713 F, 2200-4260 psia) and viscosity
# 5097.69 T^-1.72967 cp (T in F; within 0.1 %). The steam (vapour) phase, the quality and the temperature come from
# the steam tables.
BW_SCALE = (1.24138987, 2.19083831)
MU_HOT_WATER = (5097.692936740868, 1.7296724909804373)


def brine_fvf(p_psia, t_f):
    p, t = np.asarray(p_psia, float), np.asarray(t_f, float)
    dvt = -1.0001e-2 + 1.33391e-4 * t + 5.50654e-7 * t * t
    dvp = -1.95301e-9 * p * t - 1.72834e-13 * p * p * t - 3.58922e-7 * p - 2.25341e-10 * p * p
    return (1 + BW_SCALE[0] * dvt) * (1 + BW_SCALE[1] * dvp)


def hot_water_viscosity(t_f):
    return MU_HOT_WATER[0] * np.maximum(np.asarray(t_f, float), 60.0) ** -MU_HOT_WATER[1]


def brine_density_std(salinity_ppm):
    """Stock-tank brine density (lb/ft3) from the black-oil water correlation of the PVT tab."""
    from . import blackoil as bo
    inp = bo.PVTInputs(rsb=1.0, api=40.0, gas_sg=0.7, salinity_ppm=float(salinity_ppm))
    _, rho, _ = bo.water_props(inp, 14.696, 60.0)
    return float(np.ravel(rho)[0])


class SteamState:
    """Phase split at (p, h) for the traverse: quality and temperature from the steam tables, in-situ properties of
    the water (PROSPER's brine: rho_std / Bw, hot-water viscosity) and the steam (saturated vapour)."""

    def __init__(self, rho_std=62.37):
        self.CP, self.AS = _state()
        self.rho_std = float(rho_std)
        self._sat = {}

    def _sat_props(self, p_pa):
        key = round(p_pa, -1)
        s = self._sat.get(key)
        if s is None:
            CP, AS = self.CP, self.AS
            AS.update(CP.PQ_INPUTS, key, 0.0)
            hl = AS.hmass() / J_PER_KG_PER_BTU_LB
            AS.update(CP.PQ_INPUTS, key, 1.0)
            v = (AS.rhomass() * LB_FT3_PER_KG_M3, AS.viscosity() * 1000.0, AS.hmass() / J_PER_KG_PER_BTU_LB)
            s = self._sat[key] = (hl, v)
        return s

    def __call__(self, p_psia, h):
        """dict: t (F), x (quality), rho_l, rho_g (lb/ft3), mu_l, mu_g (cp), sig (dyn/cm)."""
        from .vlp import _sigma_water
        CP, AS = self.CP, self.AS
        pp = max(float(p_psia), 1.0)
        p = pp * PA_PER_PSI
        AS.update(CP.HmassP_INPUTS, float(h) * J_PER_KG_PER_BTU_LB, p)
        q = AS.Q()
        t = _f(AS.T())
        sig = float(_sigma_water(pp, t))
        rho_w = self.rho_std / float(brine_fvf(pp, t))
        mu_w = float(hot_water_viscosity(t))
        if 0.0 <= q <= 1.0 and p < AS.p_critical():
            _, (rg, mg, _) = self._sat_props(p)
            return {"t": t, "x": q, "rho_l": rho_w, "rho_g": rg, "mu_l": mu_w, "mu_g": mg, "sig": sig}
        sub = p >= AS.p_critical() or float(h) <= self._sat_props(p)[0]
        if sub:                                       # compressed (sub-cooled) water
            return {"t": t, "x": 0.0, "rho_l": rho_w, "rho_g": rho_w, "mu_l": mu_w, "mu_g": mu_w, "sig": sig}
        rho, mu = AS.rhomass() * LB_FT3_PER_KG_M3, AS.viscosity() * 1000.0       # superheated steam
        return {"t": t, "x": 1.0, "rho_l": rho, "rho_g": rho, "mu_l": mu, "mu_g": mu, "sig": sig}


@dataclass
class SteamTraverse:
    md: np.ndarray
    tvd: np.ndarray
    p: np.ndarray          # psia
    t: np.ndarray          # F
    h: np.ndarray          # BTU/lb
    x: np.ndarray          # steam quality
    holdup: np.ndarray
    grav: np.ndarray       # psi/ft
    fric: np.ndarray       # psi/ft (opposes the flow: subtracted going down)
    rho_l: np.ndarray
    rho_g: np.ndarray
    vsl: np.ndarray
    vsg: np.ndarray
    t_form: np.ndarray
    d_in: np.ndarray

    def table(self, gauge=False) -> pd.DataFrame:
        g = 14.696 if gauge else 0.0
        return pd.DataFrame({
            "Measured depth (ft)": self.md, "TVD (ft)": self.tvd,
            f"Pressure ({'psig' if gauge else 'psia'})": self.p - g, "Temperature (°F)": self.t,
            "Steam quality (%)": 100 * self.x, "Enthalpy (BTU/lb)": self.h, "Liquid holdup": self.holdup,
            "Gravity gradient (psi/ft)": self.grav, "Friction gradient (psi/ft)": -self.fric,
            "Water density (lb/ft3)": self.rho_l, "Steam density (lb/ft3)": self.rho_g,
            "Superficial water velocity (ft/s)": self.vsl, "Superficial steam velocity (ft/s)": self.vsg,
            "Formation temperature (°F)": self.t_form, "Diameter (in)": self.d_in})


def _refine(path, step):
    """Nodes every `step` ft from the top (PROSPER's steam step length), plus the diameter changes and the bottom.
    Returns (md, segment index of the path for each step)."""
    md0, md1 = float(path.md[0]), float(path.md[-1])
    keep = [path.md[i] for i in range(1, len(path.md) - 1)
            if path.d[i] != path.d[i - 1] or path.rough[i] != path.rough[i - 1]]
    md = np.union1d(np.append(np.arange(md0, md1, float(step)), md1), keep)
    mid = 0.5 * (md[1:] + md[:-1])
    seg = np.clip(np.searchsorted(path.md, mid, side="right") - 1, 0, len(path.md) - 2)
    return md, seg


def traverse(path, whp_psia, rate_stb, h_wh, u=2.0, correlation="pe2_style", step=20.0, u_seg=None,
             p1=1.0, p2=1.0, salinity_ppm=0.0) -> SteamTraverse:
    """Steam injected at the wellhead (whp psia, enthalpy h_wh BTU/lb) at rate_stb (STB/d of the injected water,
    salinity_ppm as in the PVT tab: mass = stock-tank brine density x 5.6146 ft3/STB),
    marched down the flow path (an injection.InjectionPath). Heat loss U (BTU/h/ft2/F, on the flow ID) to the
    formation; u_seg: U per path segment instead (Improved Approximation table)."""
    from .correlations import CORRELATIONS, no_slip
    corr = CORRELATIONS[correlation]
    if correlation == "pe2_style":            # PROSPER's PE2 holdup below the Hagedorn & Brown chart's floor (T42)
        import functools
        corr = functools.partial(corr, hb_low=True)
    md, seg = _refine(path, step)
    tvd = np.interp(md, path.md, path.tvd)
    tf = np.interp(md, path.md, path.t_form)
    rho_std = brine_density_std(salinity_ppm)
    w = float(rate_stb) * rho_std * 5.6146 / 24.0                 # lb/h
    ws = w / 3600.0                                               # lb/s
    st = SteamState(rho_std)
    n = len(md)
    out = {k: np.full(n, np.nan) for k in ("p", "t", "h", "x", "hl", "e", "f", "rl", "rg", "vsl", "vsg", "d")}

    def grad(i_seg, p, s):
        a = np.pi / 4 * path.d[i_seg] ** 2
        vsl = ws * (1 - s["x"]) / s["rho_l"] / a
        vsg = ws * s["x"] / s["rho_g"] / a
        args = (np.atleast_1d(max(vsl, 1e-12)), np.atleast_1d(vsg), np.atleast_1d(s["rho_l"]),
                np.atleast_1d(max(s["rho_g"], 1e-4)), np.atleast_1d(s["mu_l"]), np.atleast_1d(max(s["mu_g"], 1e-4)),
                np.atleast_1d(s["sig"]), path.d[i_seg], path.rough[i_seg], path.sin_t[i_seg], np.atleast_1d(p))
        if 1e-6 < s["x"] < 0.99999:
            e, f, hl = corr(*args)
        else:                                                      # single phase (rho_l = rho_g)
            e, f, hl = no_slip(*args)
        return p1 * float(np.ravel(e)[0]), p2 * float(np.ravel(f)[0]), float(np.ravel(hl)[0]), vsl, vsg

    def heat(i_seg, t, tfm, L):
        uu = u if u_seg is None else float(np.asarray(u_seg)[i_seg])
        return uu * np.pi * path.d_ht[i_seg] * (t - tfm) * L / w          # BTU/lb lost

    p, h = max(float(whp_psia), 14.7), float(h_wh)
    s = st(p, h)
    for j in range(n):
        if j > 0:
            i_seg = seg[j - 1]
            L = md[j] - md[j - 1]
            sn = path.sin_t[i_seg]
            e0, f0, *_ = grad(i_seg, p, s)
            tfm = 0.5 * (tf[j - 1] + tf[j])
            hp_ = h + L * sn / J - heat(i_seg, s["t"], tfm, L)
            pp = max(p + (e0 - f0) * L, 14.7)
            s1 = st(pp, hp_)
            e1, f1, *_ = grad(i_seg, pp, s1)
            p_new = max(p + 0.5 * ((e0 - f0) + (e1 - f1)) * L, 14.7)
            h_new = h + L * sn / J - heat(i_seg, 0.5 * (s["t"] + s1["t"]), tfm, L)
            p, h = p_new, h_new
            s = st(p, h)
        i_g = seg[j - 1] if j > 0 else 0
        e, f, hl, vsl, vsg = grad(i_g, p, s)
        for k, v in (("p", p), ("t", s["t"]), ("h", h), ("x", s["x"]), ("hl", hl), ("e", e), ("f", f),
                     ("rl", s["rho_l"]), ("rg", s["rho_g"]), ("vsl", vsl), ("vsg", vsg), ("d", path.d[i_g] * 12)):
            out[k][j] = v
    return SteamTraverse(md, tvd, out["p"], out["t"], out["h"], out["x"], out["hl"], out["e"], out["f"], out["rl"],
                         out["rg"], out["vsl"], out["vsg"], tf, out["d"])
