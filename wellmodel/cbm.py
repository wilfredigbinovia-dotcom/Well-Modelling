"""Coal-bed methane (CBM) producer - PROSPER's Well Type = CBM Producer (Tutorial 45).

During de-watering a CBM well produces water with a little desorbed gas. PROSPER's CBM Producer IPR models the water
inflow with relative permeabilities: the water saturation is the one whose fractional flow gives the gas-liquid ratio,

    GLR = (krg / (mu_g Bg)) / (krw / (mu_w Bw))           (Corey: krw = krw_e Swn^nw, krg = krg_e Sgn^ng)

and the water rate is q = PI krw(Sw) (Pr - Pwf), with the properties at the average of the two pressures. Found on the
user's T45 file: GLR 1 / 10 / 100 / 1000 scf/STB, PI 2, Pr 800 psig, 100 F: rates within 0.3 % (AOF 1222.4 STB/d at
GLR 10). Taking the saturation at the average pressure (pressures in psig, as PROSPER) bends the line slightly.

The well itself (Tutorial 45) is an ESP with a gas separator: the water is pumped up the tubing, the gas goes up the
annulus. The bottom-hole pressure comes from the tubing, the pump and the casing below it; the casing head pressure is
the intake pressure less the annulus columns - stagnant water from the pump up to the liquid level and gas above it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import blackoil as bo
from . import vlp as V

G = 14.696


class CBMFluid:
    """Water with free gas (no oil) for the tubing: case keys q (water STB/d), wc (ignored: all water), gor (the gas-
    liquid ratio, scf/STB - all free gas)."""
    kind = "oil"

    def __init__(self, inp: bo.PVTInputs):
        self.inp = inp

    def _props(self, p, t, gor):
        """(Rs, B, mu) as BlackOilFluid: no gas is dissolved - all of it is free (the separator removes it)."""
        z = np.zeros(np.broadcast(np.asarray(p, float), np.asarray(gor, float)).shape)
        return z, z + 1.0, z + 1.0

    def state(self, p, t, case):
        inp = self.inp
        q, glr = case["q"], np.maximum(case["gor"], 0.0)
        _, bg, rho_g, mu_g = bo.gas_props(inp, p, t)
        bw, rho_w, mu_w = V._water(inp, p, t)
        ql = q * bw * V.RB / 86400
        qg = q * glr * bg / 86400
        return dict(ql=ql, qg=qg, rho_l=rho_w + 0 * ql, rho_g=rho_g, mu_l=mu_w + 0 * ql, mu_g=mu_g,
                    sig=V.sigma_water_gas(rho_w, rho_g) + 0 * ql)

    def oil_water_viscosity(self, p, t, gor):
        mu_w = V._water(self.inp, p, t)[2]
        return mu_w, mu_w

    def mass_cp(self, case, hc: V.HeatCapacities):
        inp = self.inp
        q, glr = case["q"], np.maximum(case["gor"], 0.0)
        mg = q * glr * V.RHO_AIR_SC * inp.gas_sg
        mw = q * V.RB * 62.428 * V._water_sg(inp)
        w = mg + mw
        return w / 24.0, (mg * hc.gas + mw * hc.water) / np.maximum(w, 1e-12)


@dataclass
class CBMInflow:
    """PROSPER's CBM Producer IPR. Pressures psia; pi in STB/d/psi (the single-phase water PI)."""
    inp: bo.PVTInputs
    pr: float
    t: float
    glr: float
    pi: float
    swr: float = 0.2
    sgr: float = 0.0
    krw_e: float = 0.8
    krg_e: float = 0.8
    nw: float = 1.5
    ng: float = 2.0

    def kr(self, sw):
        sw = np.asarray(sw, float)
        den = max(1.0 - self.swr - self.sgr, 1e-9)
        swn = np.clip((sw - self.swr) / den, 0.0, 1.0)
        sgn = np.clip((1.0 - sw - self.sgr) / den, 0.0, 1.0)
        return self.krw_e * swn ** self.nw, self.krg_e * sgn ** self.ng

    def saturation(self, p, glr=None):
        """Water saturation whose fractional flow gives the GLR at pressure p (psia)."""
        glr = self.glr if glr is None else glr
        if glr <= 0:
            return 1.0
        _, bg, _, mug = (float(np.ravel(x)[0]) for x in bo.gas_props(self.inp, np.array([p]), np.array([self.t])))
        bw, _, muw = (float(np.ravel(x)[0]) for x in bo.water_props(self.inp, np.array([p]), self.t))
        bg /= 5.614583

        def f(sw):
            kw, kg = self.kr(sw)
            return (kg / (mug * bg)) - glr * (kw / (muw * bw))
        lo, hi = self.swr + 1e-9, 1.0 - self.sgr - 1e-12
        if f(hi) >= 0:
            return hi
        return brentq(f, lo, hi, xtol=1e-12)

    def krw(self, p, glr=None):
        return float(self.kr(self.saturation(p, glr))[0])

    def rate(self, pwf, glr=None):
        """Water rate (STB/d) at the flowing pressure pwf (psia)."""
        out = []
        for pw in np.atleast_1d(np.asarray(pwf, float)):
            if pw >= self.pr:
                out.append(0.0)
                continue
            p_avg = 0.5 * ((self.pr - G) + (pw - G))          # PROSPER: the gauge average (as a pressure)
            out.append(self.pi * self.krw(max(p_avg, G), glr) * (self.pr - pw))
        r = np.array(out)
        return r if np.ndim(pwf) else float(r[0])

    @property
    def aof(self):
        return self.rate(G)

    def pwf(self, q, glr=None):
        """Flowing pressure (psia) for a water rate q (STB/d)."""
        q = float(q)
        if q <= 0:
            return self.pr
        if q >= self.rate(G, glr):
            return float("nan")
        return brentq(lambda x: self.rate(x, glr) - q, G, self.pr, xtol=1e-6)

    def table(self, n=20):
        pw = np.linspace(self.pr, G, n)
        q = self.rate(pw)
        return {"pwf": pw, "q": q, "qg": q * self.glr / 1e6}


def casing_head_pressure(fluid, pip, t_intake, t_surface, pump_tvd, level_tvd, glr_q=None):
    """Casing head pressure (psia) for the intake pressure pip (psia) with the annulus liquid level at level_tvd
    (ft TVD): stagnant water from the pump up to the level, a static gas column above it (PROSPER's Gas String: the
    gas going up the annulus is too slow for friction). Temperatures linear between surface and intake (PROSPER's gas
    string follows the production string temperatures: t_surface is the wellhead temperature)."""
    inp = fluid.inp
    lvl = float(np.clip(level_tvd, 0.0, pump_tvd))
    n = 40
    # water column, marched up from the pump
    p = float(pip)
    z = np.linspace(pump_tvd, lvl, n + 1)
    for a, b in zip(z[:-1], z[1:]):
        tm = t_surface + (t_intake - t_surface) * 0.5 * (a + b) / max(pump_tvd, 1.0)
        for _ in range(2):
            rho = float(np.ravel(V._water(inp, np.array([p]), np.array([tm]))[1])[0])
        p -= rho / 144.0 * (a - b)
    z = np.linspace(lvl, 0.0, n + 1)
    for a, b in zip(z[:-1], z[1:]):
        tm = t_surface + (t_intake - t_surface) * 0.5 * (a + b) / max(pump_tvd, 1.0)
        pm = p
        for _ in range(3):
            rho = float(np.ravel(bo.gas_props(inp, np.array([max(pm, G)]), np.array([tm]))[2])[0])
            pm = p - 0.5 * rho / 144.0 * (a - b)
        p -= rho / 144.0 * (a - b)
    return p


def ipr_curve(ipr: CBMInflow, n=20):
    """An inflow.IPRCurve (liquid = water, STB/d) for the Calculation tab."""
    from .inflow import IPRCurve
    pw = np.linspace(ipr.pr, G, n)
    q = ipr.rate(pw)
    fn = lambda p: ipr.rate(np.asarray(p, float))
    return IPRCurve("CBM Producer", "oil", ipr.pr, pw, q,
                    params={"PI (STB/d/psi)": ipr.pi, "CBM GLR (scf/STB)": ipr.glr,
                            "krw at Pr": ipr.krw(ipr.pr - G), "Sw at Pr": ipr.saturation(ipr.pr - G)},
                    notes=[f"Water rate = PI x krw(Sw) x (Pr - Pwf); Sw is the saturation whose fractional flow gives "
                           f"the GLR of {ipr.glr:g} scf/STB, at the average of Pr and Pwf."], rate_fn=fn)


def lift_curves(path, fluid, esp: dict, correlation, p1, p2, u, hc, rates, levels, glrs, thps, freqs):
    """PROSPER's CBM lift curves (Tutorial 45): every combination of rate (water STB/d), annulus liquid level (ft TVD),
    CBM GLR (scf/STB), boundary (wellhead) pressure (psia) and ESP frequency (Hz). The liquid level only sets the
    casing head pressure. Returns a dict of arrays shaped (n_freq, n_thp, n_glr, n_level, n_rate) - PROSPER's .tpd
    order - with the axes."""
    from . import esp as E
    rates, levels, glrs, thps, freqs = (np.asarray(sorted(set(map(float, a))), float)
                                        for a in (rates, levels, glrs, thps, freqs))
    shape4 = (len(freqs), len(thps), len(glrs), len(rates))
    F_, T_, Gl_, Q_ = (g.ravel() for g in np.meshgrid(freqs, thps, glrs, rates, indexing="ij"))
    case = {"q": Q_, "wc": np.ones_like(Q_), "gor": Gl_}
    with np.errstate(all="ignore"):
        out = E.bhp_with_pump(path, fluid, T_, case, esp, correlation, p1, p2, u, hc, freq=F_)
    pip, pdp, t_in, t_dis = out["pip"], out["pdp"], out["t_intake"], out["t_discharge"]
    ok = np.isfinite(pip)
    above = E.sub_path(path, path.md[0], esp["pump_md"])
    case_a = {**case, "gor": out["gor_above"]}
    w, cp = fluid.mass_cp(case_a, hc)
    wht = V.temperature_profile(above, w, cp, u, t_bottom=t_dis)[:, 0]
    s = fluid.state(np.where(ok, pip, 1000.0), t_in, case_a)
    q_pump = np.asarray(s["ql"], float) * 86400.0 / V.RB
    rho = np.asarray(s["rho_l"], float)
    head = np.where(ok, (pdp - pip) * 144.0 / rho, np.nan)
    with np.errstate(all="ignore"):
        eff = np.where(out["pump_hp"] > 0, out["hyd_hp"] / out["pump_hp"], 0.0)
    pump_tvd = float(np.interp(esp["pump_md"], path.md, path.tvd))
    chp = np.full(shape4 + (len(levels),), np.nan)
    pip4, tin4, wht4 = pip.reshape(shape4), t_in.reshape(shape4), wht.reshape(shape4)
    for idx in np.ndindex(*shape4):
        if not np.isfinite(pip4[idx]):
            continue
        for k, lv in enumerate(levels):
            chp[idx + (k,)] = casing_head_pressure(fluid, pip4[idx], tin4[idx], wht4[idx], pump_tvd, lv)

    def lv5(a):                     # (f, thp, glr, rate) -> (f, thp, glr, level, rate)
        a = np.asarray(a, float).reshape(shape4)
        return np.repeat(a[:, :, :, None, :], len(levels), axis=3)
    rng = np.array([esp["pump"].range_at(f) for f in freqs])
    res = {"bhp": lv5(out["bhp"]), "wht": lv5(wht), "pump_hp": lv5(out["pump_hp"]), "eff": lv5(eff),
           "head": lv5(head), "q_pump": lv5(np.where(ok, q_pump, np.nan)), "pip": lv5(pip), "pdp": lv5(pdp),
           "t_in": lv5(t_in), "t_dis": lv5(t_dis), "dt": lv5(t_dis - t_in), "chp": np.moveaxis(chp, 4, 3),
           "q_gas": lv5(Q_ * Gl_ / 1e6),
           "q_min": np.broadcast_to(rng[:, 0, None, None, None, None], shape4[:3] + (len(levels), len(rates))),
           "q_max": np.broadcast_to(rng[:, 1, None, None, None, None], shape4[:3] + (len(levels), len(rates)))}
    res["axes"] = {"rates": rates, "levels": levels, "glrs": glrs, "thps": thps, "freqs": freqs}
    return res


TPD_COLUMNS = [(5000, "Flowing Bottom Hole Pressure", "bhp", True), (5001, "Flowing Wellhead Temperature", "wht", False),
               (5002, "ESP - Pump Power", "pump_hp", False), (5003, "ESP - Pump Efficiency", "eff", False),
               (5004, "ESP - Head Generated", "head", False), (5005, "ESP - Rate Through Pump", "q_pump", False),
               (5006, "ESP - Pump Intake Pressure", "pip", True), (5007, "ESP - Pump Outlet Pressure", "pdp", True),
               (5015, "ESP - Pump Intake Temperature", "t_in", False),
               (5019, "ESP - Pump Outlet Temperature", "t_dis", False), (5111, "ESP - Delta T Across Pump", "dt", False),
               (5113, "ESP - Casing Head Pressure", "chp", True), (5114, "ESP - Gas Rate In Casing", "q_gas", False),
               (6053, "Pump Minimum Rate", "q_min", False), (6054, "Pump Maximum Rate", "q_max", False)]


def write_tpd(res, gas_sg, bottom_tvd, title="", first_node=None):
    """PROSPER .tpd lift-curve file (TPD version 2, CBM producer with an ESP: Liquid Level, CBM GLR, Boundary
    Pressure and Operating Frequency as the four sensitivity variables) for GAP / MBAL. Pressures in psig; blank
    points (the pump cannot deliver) are written as PROSPER does, 3.4e+35."""
    ax = res["axes"]
    fn = first_node if first_node is not None else float(ax["thps"][0]) - G
    lines = ["#", f"#  Generated by : wellmodel - {title}".rstrip(" -"), "#  Fluid : Coalbed Methane",
             "#  Well Type : CBM Producer", "#  Artificial Lift : Electrical Submersible Pump", "#",
             "#        Sensitivity Variable  1 :  Operating Frequency",
             "#        Sensitivity Variable  2 :  Boundary Pressure",
             "#        Sensitivity Variable  3 :  CBM GLR", "#        Sensitivity Variable  4 :  Liquid Level", "#",
             "# TPD File - Tubing Performance Data", "# TPD File Signature", "TPDData", "# TPD File Version", "2",
             "# Fluid Type, Well Type, Temperature Model, Lift Type, PVT Method", "0,3,1,4,0",
             "# First Node Pressure, Temperature", f"{fn:g},0",
             "# Oil Gravity, Gas Gravity, (WC/WGR), (GOR/CGR/Total GOR), Top TVD, Bottom TVD",
             f"35,{gas_sg:g},3.4e+35,0,0,{bottom_tvd:g}", "# Number of Sensitivity Variables", "4",
             "# Numbers of :- Rates, Liquid Level values, CBM GLR values, Boundary Pressure values, Operating Frequency "
             "values", ", ".join(str(len(ax[k])) for k in ("rates", "levels", "glrs", "thps", "freqs")),
             "# Number of Calculated Values (columns)", str(len(TPD_COLUMNS))]
    lines += [f"# {c} - {n}" for c, n, _, _ in TPD_COLUMNS]
    lines += [", ".join(str(c) for c, _, _, _ in TPD_COLUMNS), "# Rate and Variable Types",
              "# Rate Variable =  4000 - Liquid Rate", "# Variable 4 =  353 - Liquid Level",
              "# Variable 3 =  354 - CBM GLR", "# Variable 2 =  27 - Boundary Pressure",
              "# Variable 1 =  35 - Operating Frequency", "4000,353,354,27,35", "# Rate Values",
              ", ".join(f"{v:g}" for v in ax["rates"]), "# Variable 4 (Liquid Level) values",
              ", ".join(f"{v:g}" for v in ax["levels"]), "# Variable 3 (CBM GLR) values",
              ", ".join(f"{v:g}" for v in ax["glrs"]), "# Variable 2 (Boundary Pressure) values",
              ", ".join(f"{v - G:g}" for v in ax["thps"]), "# Variable 1 (Operating Frequency) values",
              ", ".join(f"{v:g}" for v in ax["freqs"]), "# 4 Variable TPD Results"]
    shape = res["bhp"].shape
    for idx in np.ndindex(*shape):
        vals = []
        for _, _, key, gauge in TPD_COLUMNS:
            v = float(res[key][idx])
            vals.append("3.4e+35" if not np.isfinite(v) else f"{(v - G) if gauge else v:12.6g}")
        lines.append(", ".join(vals))
    return "\n".join(lines) + "\n"


def cbm_ipr(gas_sg, salinity_ppm, h2s, co2, n2, pr, t, glr, pi, swr=0.2, sgr=0.0, krw_e=0.8, krg_e=0.8, nw=1.5,
            ng=2.0, n=20):
    """The CBM Producer IPR as an inflow.IPRCurve (the app's IPR recipe: pressures psia)."""
    inp = bo.PVTInputs(rsb=1, api=40, gas_sg=gas_sg, salinity_ppm=salinity_ppm, h2s=h2s, co2=co2, n2=n2)
    return ipr_curve(CBMInflow(inp, pr, t, glr, pi, swr, sgr, krw_e, krg_e, nw, ng), n)
