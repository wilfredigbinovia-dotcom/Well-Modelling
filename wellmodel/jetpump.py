"""Jet pump design and system analysis (after PROSPER Design | Jet Pump, Tutorial 32).

Annular injection, tubular production: power fluid down the tubing-casing annulus to the nozzle, production plus
power fluid up the tubing.

- Pump performance (found from PROSPER's Tutorial 32 'actual' pump, R = 0.41): Cunningham's equation with equal
  densities, H = (Pd - Ps)/(Pn - Pd) = N / D,
      N = 2R + 2 M^2 R^2/(1-R) - R^2 (1+M)^2 (1+Kt+Kd) - (1+Ks) M^2 R^2/(1-R)^2
      D = (1+Kn) - 2R - 2 M^2 R^2/(1-R) + R^2 (1+M)^2 (1+Kt+Kd)
  M = average production rate in the pump (RB/d) / power fluid rate (RB/d), R = An/At, efficiency = M H.
- Cavitation limit (reproduces PROSPER's six Mc values): Mc = ((1-R)/R) sqrt((1+Kn) Ps / (Ps + Ic (Pn - Ps))),
  Ps in psig, for cavitation indices Ic = 0.8, 1.35, 1.67.
- Nozzle pressure = surface injection pressure + power fluid static column - annulus friction; the desired nozzle
  area from q = An sqrt(2 g dP / rho) with dP = Pn - Ps (no nozzle loss in it, as PROSPER's design).
- Pump duty: intake pressure from the IPR up the casing to the pump; discharge pressure from the top node down the
  tubing with production + power fluid (the power fluid's water blended into the produced water: its salinity
  enters as a rate-weighted salinity - reproduces PROSPER's discharge pressure to 0.6 psi).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import blackoil as bo
from . import vlp as V
from .esp import sub_path, _rates

G_FT = 32.174
DP_HP = 1.7014e-5           # hp per (bbl/d x psi)
IC = (0.80, 1.35, 1.67)     # cavitation indices PROSPER reports


@dataclass
class Losses:
    nozzle: float = 0.15
    suction: float = 0.0
    throat: float = 0.28
    diffuser: float = 0.10


def head_ratio(R, M, k: Losses = Losses()):
    """H = (Pd - Ps)/(Pn - Pd) of a jet pump with area ratio R at flow ratio M (Cunningham, equal densities)."""
    R, M = np.asarray(R, float), np.asarray(M, float)
    a = 2 * M ** 2 * R ** 2 / (1 - R)
    b = R ** 2 * (1 + M) ** 2 * (1 + k.throat + k.diffuser)
    c = (1 + k.suction) * M ** 2 * R ** 2 / (1 - R) ** 2
    return (2 * R + a - b - c) / ((1 + k.nozzle) - 2 * R - a + b)


def cavitation_m(R, ps_psig, pn_psig, k: Losses = Losses(), ic=IC):
    """Flow ratio at which cavitation starts (PROSPER's Mc) for each cavitation index."""
    return [float((1 - R) / R * np.sqrt((1 + k.nozzle) * ps_psig / (ps_psig + i * (pn_psig - ps_psig)))) for i in ic]


def nozzle_area(q_rbd, pn, ps, rho):
    """Nozzle area (in2) passing q (RB/d) of power fluid of density rho (lb/ft3) at a pressure drop pn - ps (psi)."""
    v = np.sqrt(2 * G_FT * 144.0 * max(pn - ps, 1e-9) / rho)
    return q_rbd * 5.614583 / 86400.0 / v * 144.0


def nozzle_rate(an_in2, pn, ps, rho):
    """Power fluid rate (RB/d) through a nozzle of area an (in2)."""
    v = np.sqrt(2 * G_FT * 144.0 * max(pn - ps, 0.0) / rho)
    return an_in2 / 144.0 * v * 86400.0 / 5.614583


@dataclass
class PowerFluid:
    salinity: float = 75000.0
    density: float | None = None       # lb/ft3; None = water at 60 F and the nozzle pressure from the PVT
    viscosity: float | None = None     # cp; None = water at 60 F

    def props(self, p=3000.0):
        inp = bo.PVTInputs(rsb=1, api=35, gas_sg=0.7, salinity_ppm=self.salinity)
        bw, rho, mu = bo.water_props(inp, np.array([p]), np.array([60.0]))
        bw0, _, _ = bo.water_props(inp, np.array([14.696]), np.array([60.0]))
        rho = float(np.atleast_1d(rho)[0]) if self.density is None else self.density
        mu = float(np.atleast_1d(mu)[0]) if self.viscosity is None else self.viscosity
        return rho, mu, float(np.atleast_1d(bw)[0])


def annulus_friction(q_rbd, rho, mu, length_ft, casing_id, tubing_od, rough=0.0006):
    """Frictional pressure drop (psi) of the power fluid down the annulus: Chen friction factor, Reynolds number on
    Dh = Dc - Dto, relative roughness e / (Dc + Dto) (fitted: PROSPER's 129.4-130.2 psi in its 36 Tutorial 32 system
    runs within 1.1 %; e / Dh gives 12.5 % more)."""
    dh = (casing_id - tubing_od) / 12.0
    area = np.pi / 4 * (casing_id ** 2 - tubing_od ** 2) / 144.0
    v = q_rbd * 5.614583 / 86400.0 / area
    re = 1488.0 * rho * abs(v) * dh / mu
    if re < 2100:
        f = 64.0 / max(re, 1e-9)
    else:
        e = rough / (casing_id + tubing_od)
        a1 = e / 3.7065 - 5.0452 / re * np.log10(e ** 1.1098 / 2.8257 + 5.8506 / re ** 0.8981)
        f = 1.0 / (-2.0 * np.log10(a1)) ** 2
    return float(f * rho * v * v / (2 * G_FT * dh) * length_ft / 144.0)


def _blend_fluid(fluid, q, wc, q_pf, pf_salinity):
    """The produced fluid with the power fluid's water mixed into its water (rate-weighted salinity)."""
    if not hasattr(fluid, "m"):
        return fluid
    m = fluid.m
    qw = q * wc
    sal = (qw * m.inp.salinity_ppm + q_pf * pf_salinity) / max(qw + q_pf, 1e-9)
    import dataclasses
    inp = dataclasses.replace(m.inp, salinity_ppm=sal)
    return V.BlackOilFluid(bo.BlackOilModel(inp, m.family, m.visc_family, m.params))


def power_fluid_column(above, t_tubing, d, pip, pdis, n=60):
    """Power fluid down the annulus from the top node to the pump: (as PROSPER, Tutorial 32's 36 system runs)
    density of the power fluid water at the production tubing's temperature, integrated over measured depth
    (PROSPER's static column; over true vertical depth it is 2.9 % lighter in that deviated well - within 0.25 %
    over measured depth); friction with the mean density and viscosity; rate at the nozzle = surface rate x Bw at
    the intake temperature and the mean of nozzle and discharge pressures (0.003 %)."""
    winp = bo.PVTInputs(rsb=1, api=35, gas_sg=0.7, salinity_ppm=d.power_fluid.salinity)
    md = np.linspace(float(above.md[0]), float(above.md[-1]), n + 1)
    tz = np.interp(md, above.md, t_tubing)
    p = d.p_surf
    static, rhos, mus = 0.0, [], []
    for i in range(n):
        tm = 0.5 * (tz[i] + tz[i + 1])
        _, r_, m_ = bo.water_props(winp, np.array([p]), np.array([tm]))
        r_ = float(np.atleast_1d(r_)[0]) if d.power_fluid.density is None else d.power_fluid.density
        m_ = float(np.atleast_1d(m_)[0]) if d.power_fluid.viscosity is None else d.power_fluid.viscosity
        dp_ = r_ * (md[i + 1] - md[i]) / 144.0
        static += dp_
        p += dp_
        rhos.append(r_)
        mus.append(m_)
    rho, mu = float(np.mean(rhos)), float(np.mean(mus))
    t_in = float(t_tubing[-1])
    pn = d.p_surf + static
    for _ in range(4):
        bw = float(np.atleast_1d(bo.water_props(winp, np.array([0.5 * (pn + pdis)]), np.array([t_in]))[0])[0])
        q_n = d.q_pf * bw
        fric = annulus_friction(q_n, rho, mu, float(md[-1] - md[0]), d.casing_id, d.tubing_od)
        pn = d.p_surf + static - fric
    return {"pn": pn, "q_n": q_n, "static": static, "friction": fric, "rho": rho, "mu": mu, "bw": bw}


@dataclass
class JetDutyInput:
    pump_md: float = 6000.0
    q: float = 4000.0                 # design liquid rate, STB/d
    wc: float = 0.8
    gor: float = 800.0
    top_p: float = 164.696            # psia
    q_pf: float = 10000.0             # surface injection rate, STB/d
    p_surf: float = 2014.696          # surface injection pressure, psia
    losses: Losses = field(default_factory=Losses)
    power_fluid: PowerFluid = field(default_factory=PowerFluid)
    casing_id: float = 6.4
    tubing_od: float = 4.5
    max_od: float = 6.4
    friction_from_md: float | None = None   # top of the annulus (MD); default the top of the flow path


@dataclass
class JetDuty:
    inp: JetDutyInput
    pwf: float
    pip: float
    t_in: float
    q_in: float                 # production at the intake, RB/d (liquid + free gas)
    free_gor: float
    pdis: float
    q_dis_total: float          # production + power fluid at the discharge, RB/d
    mass: float                 # production, lbm/d
    q_avg: float                # production, RB/d (at the mean of intake and discharge pressures)
    sg: float
    head: float
    fluid_hp: float
    glr_in: float
    bo_in: float
    bg_in: float
    pn: float
    static: float
    friction: float
    q_n: float                  # power fluid at the nozzle, RB/d
    rho_pf: float
    M: float
    H: float
    surface_hp: float
    detail: dict = field(default_factory=dict)

    @property
    def efficiency(self):
        return self.M * self.H

    def area_ratios(self, ratios=None):
        """Head ratio and efficiency of each area ratio at the duty's flow ratio; those at or above the required H
        can do the job."""
        ratios = ratios if ratios is not None else np.round(np.arange(0.15, 0.71, 0.01), 2)
        h = head_ratio(ratios, self.M, self.inp.losses)
        mc = [cavitation_m(r, self.pip - 14.696, self.pn - 14.696, self.inp.losses)[0] for r in ratios]
        return pd.DataFrame({"Area ratio R": ratios, "Head ratio H at the duty M": h,
                             "Efficiency (%)": 100 * self.M * h, "Meets required H": h >= self.H,
                             "Cavitation Mc (Ic 0.8)": mc, "Cavitates (M > Mc)": self.M > np.array(mc)})


def duty(path, fluid, pwf, d: JetDutyInput, correlation="pe2_style", p1=1.0, p2=1.0, u=8.0, hc=None):
    """PROSPER 'Calculate Pump Duty' for a jet pump (psia, STB/d, F). pwf: flowing BHP at the bottom node for d.q."""
    hc = hc or V.HeatCapacities()
    below = sub_path(path, d.pump_md, path.md[-1]) if path.md[-1] > d.pump_md + 0.5 else None
    above = sub_path(path, path.md[0], d.pump_md)
    case = {"q": np.array([d.q]), "wc": np.array([d.wc]), "gor": np.array([d.gor])}
    if below is not None:
        Pb, Tb = V.traverse_up(below, fluid, np.array([pwf]), case, correlation, p1, p2, u, hc)
        pip, t_in = float(Pb[0, 0]), float(Tb[0, 0])
    else:
        pip, t_in = pwf, float(path.t_form[-1])
    if not np.isfinite(pip) or pip <= 14.7:
        raise ValueError("The fluid does not reach the pump: the intake pressure falls to atmospheric.")
    ql, qg, rs = _rates(fluid, pip, t_in, case)
    # discharge: top node down the tubing with production + power fluid (power fluid water blended in)
    qt = d.q + d.q_pf
    wct = (d.q * d.wc + d.q_pf) / qt
    gor_t = d.gor                                  # per STB of oil: the power fluid brings no oil or gas
    fl_mix = _blend_fluid(fluid, d.q, d.wc, d.q_pf, d.power_fluid.salinity)
    case_t = {"q": np.array([qt]), "wc": np.array([wct]), "gor": np.array([gor_t])}
    P, T = V.traverse(above, fl_mix, np.array([d.top_p]), case_t, correlation, p1, p2, u, hc, t_in)
    pdis = float(P[0, -1])
    ql_d, qg_d, _ = _rates(fl_mix, pdis, t_in, case_t)
    ql_a, qg_a, _ = _rates(fluid, 0.5 * (pip + pdis), t_in, case)
    q_avg = ql_a + qg_a
    w, _ = fluid.mass_cp(case, hc)
    mass = float(w[0]) * 24.0
    rho = mass / (q_avg * 5.614583)
    pf = power_fluid_column(above, np.asarray(T[0]), d, pip, pdis)
    pn, q_n, static, fric, rho_pf = pf["pn"], pf["q_n"], pf["static"], pf["friction"], pf["rho"]
    M = q_avg / q_n
    H = (pdis - pip) / (pn - pdis)
    bo_in = bg_in = np.nan
    if hasattr(fluid, "m"):
        tab = fluid.m.table(t_in, [pip])
        bo_in, bg_in = float(tab["Oil FVF (RB/STB)"].iloc[0]), float(tab["Gas FVF (ft3/scf)"].iloc[0])
    return JetDuty(d, pwf, pip, t_in, ql + qg, d.gor - rs if np.isfinite(rs) else np.nan, pdis, ql_d + qg_d,
                   mass, q_avg, rho / 62.428, (pdis - pip) * 144.0 / rho, q_avg * (pdis - pip) * DP_HP,
                   qg / max(ql, 1e-9), bo_in, bg_in, pn, static, fric, q_n, rho_pf, M, H,
                   q_n * (d.p_surf - 14.696) * DP_HP,
                   {"t_wellhead": float(T[0, 0]), "free_gas_in": qg, "liquid_in": ql, "rs_in": rs})


def with_pump(du: JetDuty, an, at):
    """The selected pump at the duty's flow ratio (PROSPER's 'Actual' column): H from the pump's area ratio, the
    discharge pressure it reaches, efficiency and cavitation."""
    R = an / at
    k = du.inp.losses
    H = float(head_ratio(R, du.M, k))
    pdis = (du.pip + H * du.pn) / (1 + H)
    mc = cavitation_m(R, du.pip - 14.696, du.pn - 14.696, k)
    return {"Area ratio (R)": R, "Aj (nozzle area, in2)": an, "At (throat area, in2)": at,
            "Dj (in)": float(np.sqrt(4 * an / np.pi)), "Dt (in)": float(np.sqrt(4 * at / np.pi)),
            "Dimensionless flow ratio (M)": du.M, "Dimensionless head ratio (H)": H,
            "Pump efficiency (%)": 100 * du.M * H, "Pump discharge pressure (psia)": pdis,
            "Discharge pressure required (psia)": du.pdis,
            "Meets the duty": bool(pdis >= du.pdis - 0.5),
            **{f"Mc (Ic = {i:.2f})": m for i, m in zip(IC, mc)},
            "Cavitation": bool(du.M > mc[0])}


def duty_table(du: JetDuty, g=14.696):
    d = du.inp
    rows = {"Well head pressure (psig)": d.top_p - g, "Flowing BH pressure (psig)": du.pwf - g,
            "Pump intake pressure (psig)": du.pip - g, "Pump intake rate (RB/d)": du.q_in,
            "Free GOR at pump inlet (scf/STB)": du.free_gor, "Pump discharge pressure (psig)": du.pdis - g,
            "Pump discharge rate (RB/d)": du.q_dis_total, "Total GOR above pump (scf/STB)": d.gor,
            "Mass flow rate (lbm/d)": du.mass, "Total fluid gravity": du.sg,
            "Average downhole rate (RB/d)": du.q_avg, "Head required (ft)": du.head,
            "Fluid power required (hp)": du.fluid_hp, "GLR at pump intake (V/V)": du.glr_in,
            "Bo at pump intake (RB/STB)": du.bo_in, "Inlet temperature (F)": du.t_in,
            "Bg at pump intake (ft3/scf)": du.bg_in, "Power fluid injection rate (STB/d)": d.q_pf,
            "Power fluid surface injection pressure (psig)": d.p_surf - g,
            "Nozzle injection pressure (psig)": du.pn - g, "Dimensionless flow ratio (M)": du.M,
            "Dimensionless head recovery ratio (H)": du.H, "Pump efficiency (%)": 100 * du.efficiency,
            "Surface power required (hp)": du.surface_hp, "Power fluid rate at the nozzle (RB/d)": du.q_n,
            "Power fluid density (lb/ft3)": du.rho_pf, "Power fluid static gradient (psi)": du.static,
            "Power fluid friction (psi)": -du.friction}
    return pd.DataFrame({"Value": rows})


# ----------------------------------------------------------------------------------------------- system analysis
@dataclass
class JetPumpInstalled:
    pump_md: float = 6000.0
    an: float = 0.16476               # nozzle area, in2
    at: float = 0.40185               # throat area, in2
    q_pf: float = 10000.0             # power fluid injection rate, STB/d
    p_surf: float = 2014.696          # surface injection pressure, psia
    losses: Losses = field(default_factory=Losses)
    power_fluid: PowerFluid = field(default_factory=PowerFluid)
    casing_id: float = 6.4
    tubing_od: float = 4.5
    name: str = ""


def _state(path, fluid, pwf, q, wc, gor, top_p, pump: JetPumpInstalled, correlation, p1, p2, u, hc):
    d = JetDutyInput(pump.pump_md, q, wc, gor, top_p, pump.q_pf, pump.p_surf, pump.losses, pump.power_fluid,
                     pump.casing_id, pump.tubing_od)
    du = duty(path, fluid, pwf, d, correlation, p1, p2, u, hc)
    R = pump.an / pump.at
    H = float(head_ratio(R, du.M, pump.losses))
    pd_pump = (du.pip + H * du.pn) / (1 + H) if H > 0 else du.pip   # no head: the pump only passes the fluid
    return du, H, max(pd_pump, du.pip)


def system(path, fluid, ipr_pwf, wc, gor, top_p, pump: JetPumpInstalled, correlation="pe2_style", p1=1.0, p2=1.0,
           u=8.0, hc=None, q_max=None, n=30):
    """PROSPER system analysis for a jet pump well (annular injection, tubing production; power fluid rate and
    surface pressure fixed, as PROSPER - its in-situ rate follows from Bw). For a liquid rate: BHP from the IPR, the
    intake pressure up the casing, the discharge pressure the tubing needs from the top node down (production +
    power fluid) and the one the pump delivers, (Ps + H Pn)/(1 + H) with H from the pump's area ratio at the flow
    ratio M. The solution is the largest rate where the pump's discharge pressure falls through the tubing's.
    ipr_pwf(q) -> psia (NaN past the AOF). Returns (solution dict or None, curve DataFrame)."""
    from scipy.optimize import brentq
    hc = hc or V.HeatCapacities()
    q_hi = q_max or 20000.0
    qs = np.linspace(q_hi * 0.02, q_hi, n)
    rows = []

    def resid(q):
        pwf = float(ipr_pwf(q))
        if not np.isfinite(pwf):
            return None
        try:
            du, H, pdp = _state(path, fluid, pwf, q, wc, gor, top_p, pump, correlation, p1, p2, u, hc)
        except ValueError:
            return None
        return du, H, pdp, pdp - du.pdis

    for q in qs:
        r = resid(q)
        if r is None:
            break
        du, H, pdp, f = r
        rows.append({"Liquid rate (STB/d)": q, "Pump discharge from the pump (psia)": pdp,
                     "Discharge needed by the tubing (psia)": du.pdis, "Pump intake (psia)": du.pip, "M": du.M,
                     "H": H, "_f": f})
    curve = pd.DataFrame(rows)
    if len(curve) < 2:
        return None, curve.drop(columns="_f", errors="ignore")
    f = curve["_f"].to_numpy()
    k = None
    for j in range(len(f) - 1, 0, -1):
        if f[j - 1] > 0 >= f[j]:
            k = j
            break
    curve = curve.drop(columns="_f")
    if k is None:
        return None, curve
    q_sol = brentq(lambda q: resid(q)[3], qs[k - 1], qs[k], xtol=0.5)
    du, H, pdp, _ = resid(q_sol)
    R = pump.an / pump.at
    mc = cavitation_m(R, du.pip - 14.696, du.pn - 14.696, pump.losses)
    sol = {"Liquid rate (STB/d)": q_sol, "Oil rate (STB/d)": q_sol * (1 - wc), "Water rate (STB/d)": q_sol * wc,
           "Gas rate (MMscf/d)": q_sol * (1 - wc) * gor / 1e6, "BHP (psia)": du.pwf,
           "Pump intake pressure (psia)": du.pip, "Pump discharge pressure (psia)": pdp,
           "Nozzle injection pressure (psia)": du.pn, "Power fluid static (psi)": du.static,
           "Power fluid friction (psi)": -du.friction, "Power fluid rate at the nozzle (RB/d)": du.q_n,
           "Pump intake rate (RB/d)": du.q_in, "Average rate through pump (RB/d)": du.q_avg,
           "Pump discharge rate (RB/d)": du.q_dis_total, "Flow ratio M": du.M, "Head ratio H": H,
           "Pump efficiency (%)": 100 * du.M * H, "Pump head generated (ft)": du.head,
           "Pump intake temperature (F)": du.t_in, "Free GOR at pump intake (scf/STB)": du.free_gor,
           "Surface power required (hp)": du.surface_hp, "Cavitation": "Yes" if du.M > mc[0] else "No"}
    return sol, curve


# ----------------------------------------------------------------------------------------------- pump database
def load_pumps(path=None) -> pd.DataFrame:
    """Jet pump database (PROSPER Options | Artificial Lift Database | Jet Pump export): nozzle and throat sizes,
    rate range, pump size. Adds the areas and the area ratio."""
    from pathlib import Path
    path = path or Path(__file__).parent / "data" / "jet_pumps.csv"
    df = pd.read_csv(path)
    df["Nozzle area (in2)"] = np.pi / 4 * df["Nozzle diameter (in)"] ** 2
    df["Throat area (in2)"] = np.pi / 4 * df["Throat diameter (in)"] ** 2
    df["Area ratio"] = df["Nozzle area (in2)"] / df["Throat area (in2)"]
    df["Name"] = df.apply(lambda r: f"{r['Manufacturer']} {r['Model']} - Nozzle ({int(r['Nozzle number'])}) "
                                    f"Throat ({int(r['Throat number'])}) {r['Ratio type']}", axis=1)
    return df


PUMPS = load_pumps()


def candidates(du: JetDuty, pumps: pd.DataFrame | None = None, max_od=None) -> pd.DataFrame:
    """Database pumps for a duty: the nozzle rate range covers the power fluid rate at the nozzle, the pump fits the
    maximum OD; each pump's head ratio at the duty's flow ratio gives the discharge pressure it reaches. Pumps that
    reach the required discharge pressure without cavitating come first, by efficiency."""
    pumps = PUMPS if pumps is None else pumps
    max_od = du.inp.max_od if max_od is None else max_od
    sel = pumps[(pumps["Minimum rate (RB/d)"] <= du.q_n) & (du.q_n <= pumps["Maximum rate (RB/d)"])
                & (pumps["Pump size (in)"] <= max_od + 1e-9)]
    rows = []
    for _, r in sel.iterrows():
        a = with_pump(du, r["Nozzle area (in2)"], r["Throat area (in2)"])
        rows.append({"Pump": r["Name"], "Area ratio (R)": a["Area ratio (R)"],
                     "Aj (in2)": r["Nozzle area (in2)"], "At (in2)": r["Throat area (in2)"],
                     "Head ratio (H)": a["Dimensionless head ratio (H)"], "Efficiency (%)": a["Pump efficiency (%)"],
                     "Discharge reached (psig)": a["Pump discharge pressure (psia)"] - 14.696,
                     "Discharge needed (psig)": du.pdis - 14.696, "Meets the duty": a["Meets the duty"],
                     "Mc (Ic 0.8)": a["Mc (Ic = 0.80)"], "Cavitation": a["Cavitation"],
                     "Rate range (RB/d)": f"{r['Minimum rate (RB/d)']:,.0f} - {r['Maximum rate (RB/d)']:,.0f}",
                     "_centre": abs(du.q_n - 0.5 * (r["Minimum rate (RB/d)"] + r["Maximum rate (RB/d)"])),
                     "_eff": round(a["Pump efficiency (%)"], 3)})
    out = pd.DataFrame(rows)
    if len(out):
        # ties (same area ratio) go to the pump whose rate range is centred nearest the power fluid rate
        out = out.sort_values(["Meets the duty", "Cavitation", "_eff", "_centre"],
                              ascending=[False, True, False, True]).drop(columns=["_centre", "_eff"])
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- traverse
def intake_from_discharge(path, fluid, q, wc, gor, pdis, t_tubing, t_in, pump: JetPumpInstalled, hc=None):
    """Pump intake pressure that lets the jet pump deliver the discharge pressure pdis at liquid rate q (gradient
    traverse from the top node, as PROSPER's: Ps = Pd (1 + H) - H Pn, iterated as M depends on Ps).
    t_tubing: tubing temperature at the nodes of the path above the pump (top -> pump). Returns a dict or raises
    ValueError when the pump cannot (the intake would fall below atmospheric)."""
    hc = hc or V.HeatCapacities()
    above = sub_path(path, path.md[0], pump.pump_md)
    d = JetDutyInput(pump.pump_md, q, wc, gor, 0.0, pump.q_pf, pump.p_surf, pump.losses, pump.power_fluid,
                     pump.casing_id, pump.tubing_od)
    case = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([gor])}
    R = pump.an / pump.at
    ps = 0.6 * pdis
    for _ in range(30):
        pf = power_fluid_column(above, np.asarray(t_tubing), d, ps, pdis)
        ql, qg, _ = _rates(fluid, 0.5 * (ps + pdis), t_in, case)
        M = (ql + qg) / pf["q_n"]
        H = float(head_ratio(R, M, pump.losses))
        ps_new = pdis * (1 + H) - H * pf["pn"] if H > 0 else pdis
        if ps_new <= 14.7:
            raise ValueError(f"at {q:g} STB/d the jet pump cannot deliver the discharge pressure: the intake would "
                             "fall below atmospheric (pumped off).")
        if abs(ps_new - ps) < 0.01:
            ps = ps_new
            break
        ps = 0.5 * (ps + ps_new)
    return {"pip": ps, "M": M, "H": H, "pn": pf["pn"], "q_n": pf["q_n"], "static": pf["static"],
            "friction": pf["friction"]}


# ----------------------------------------------------------------------------------------------- lift curves
def _annulus_friction_v(q_rbd, rho, mu, length_ft, casing_id, tubing_od, rough=0.0006):
    """annulus_friction over arrays."""
    dh = (casing_id - tubing_od) / 12.0
    area = np.pi / 4 * (casing_id ** 2 - tubing_od ** 2) / 144.0
    v = np.asarray(q_rbd, float) * 5.614583 / 86400.0 / area
    re = np.maximum(1488.0 * rho * np.abs(v) * dh / mu, 1e-9)
    e = rough / (casing_id + tubing_od)
    a1 = e / 3.7065 - 5.0452 / re * np.log10(e ** 1.1098 / 2.8257 + 5.8506 / re ** 0.8981)
    f = np.where(re < 2100, 64.0 / re, 1.0 / (-2.0 * np.log10(a1)) ** 2)
    return f * rho * v * v / (2 * G_FT * dh) * length_ft / 144.0


def bhp_with_jet(path, fluid, top_p, case, pump: JetPumpInstalled, correlation="pe2_style", p1=1.0, p2=1.0, u=8.0,
                 hc=None, p_surf=None, q_pf=None, n_pf=40):
    """Bottom-node pressure of a jet pump well from the top-node pressure, vectorised over cases (lift curves): tubing
    from the top node to the pump discharge with production + power fluid (power fluid salinity blended in, cases
    grouped by salinity to 1000 ppm), the intake from the discharge (Ps = Pd (1 + H) - H Pn, iterated), casing to
    the bottom node. p_surf / q_pf: injection pressure (psia) / rate (STB/d) per case (an ALQ axis); default the
    pump's. Returns a dict of arrays: bhp, pip, pdis, pn (NaN where the pump cannot deliver - intake below 14.7)."""
    from .esp import BBL_FT3
    hc = hc or V.HeatCapacities()
    q = np.asarray(case["q"], float)
    n = q.size
    wc = np.broadcast_to(np.asarray(case["wc"], float), q.shape).astype(float)
    gor = np.broadcast_to(np.asarray(case["gor"], float), q.shape).astype(float)
    top_p = np.broadcast_to(np.asarray(top_p, float), q.shape).astype(float)
    ps_s = np.broadcast_to(np.asarray(pump.p_surf if p_surf is None else p_surf, float), q.shape).astype(float)
    qpf = np.broadcast_to(np.asarray(pump.q_pf if q_pf is None else q_pf, float), q.shape).astype(float)
    above = sub_path(path, path.md[0], pump.pump_md)
    below = sub_path(path, pump.pump_md, path.md[-1]) if path.md[-1] > pump.pump_md + 0.5 else None
    w, cp = fluid.mass_cp(case, hc)
    t_in = (V.temperature_profile(below, np.broadcast_to(np.asarray(w, float), q.shape), cp, u)[:, 0]
            if below is not None else np.full(n, float(path.t_form[-1])))
    # tubing: production + power fluid, grouped by blended salinity
    qt = q + qpf
    wct = (q * wc + qpf) / qt
    sal0 = fluid.m.inp.salinity_ppm if hasattr(fluid, "m") else pump.power_fluid.salinity
    sal = np.round((q * wc * sal0 + qpf * pump.power_fluid.salinity) / np.maximum(q * wc + qpf, 1e-9), -3)
    pdis = np.full(n, np.nan)
    T_tub = np.full((n, len(above.md)), np.nan)
    for s_ in np.unique(sal):
        idx = np.where(sal == s_)[0]
        fl_s = fluid
        if hasattr(fluid, "m"):
            import dataclasses as _dc
            m = fluid.m
            fl_s = V.BlackOilFluid(bo.BlackOilModel(_dc.replace(m.inp, salinity_ppm=float(s_)), m.family,
                                                    m.visc_family, m.params))
        c_ = {"q": qt[idx], "wc": wct[idx], "gor": gor[idx]}
        with np.errstate(all="ignore"):
            P, T = V.traverse(above, fl_s, top_p[idx], c_, correlation, p1, p2, u, hc, t_in[idx])
        pdis[idx] = P[:, -1]
        T_tub[idx] = T
    # power fluid column (vectorised power_fluid_column)
    winp = bo.PVTInputs(rsb=1, api=35, gas_sg=0.7, salinity_ppm=pump.power_fluid.salinity)
    md = np.linspace(float(above.md[0]), float(above.md[-1]), n_pf + 1)
    tz = np.array([np.interp(md, above.md, T_tub[i]) for i in range(n)])
    p = ps_s.copy()
    static = np.zeros(n)
    rho_s = np.zeros(n)
    mu_s = np.zeros(n)
    for i in range(n_pf):
        tm = 0.5 * (tz[:, i] + tz[:, i + 1])
        _, r_, m_ = bo.water_props(winp, p, tm)
        r_ = np.asarray(r_, float)
        dp_ = r_ * (md[i + 1] - md[i]) / 144.0
        static += dp_
        p = p + dp_
        rho_s += r_ / n_pf
        mu_s += np.asarray(m_, float) / n_pf
    pn = ps_s + static
    for _ in range(4):
        bw = np.asarray(bo.water_props(winp, 0.5 * (pn + pdis), t_in)[0], float)
        q_n = qpf * bw
        fric = _annulus_friction_v(q_n, rho_s, mu_s, float(md[-1] - md[0]), pump.casing_id, pump.tubing_od)
        pn = ps_s + static - fric
    # intake from the discharge
    R = pump.an / pump.at
    pip = 0.6 * pdis
    ok = np.isfinite(pdis)
    for _ in range(40):
        with np.errstate(all="ignore"):
            st_ = fluid.state(np.where(ok, 0.5 * (pip + pdis), 1000.0), t_in, case)
            qin = (np.asarray(st_["ql"], float) + np.asarray(st_["qg"], float)) * 86400.0 / BBL_FT3
            M = qin / q_n
            H = head_ratio(R, M, pump.losses)
            new = np.where(H > 0, pdis * (1 + H) - H * pn, pdis)
        bad = new <= 14.7
        new = np.where(bad, 14.7, new)
        if np.nanmax(np.abs(np.where(ok, new - pip, 0.0))) < 0.01:
            pip = new
            break
        pip = 0.5 * (pip + new)
    ok &= pip > 14.7 + 1e-6
    if below is not None:
        with np.errstate(all="ignore"):
            Pb, _ = V.traverse(below, fluid, np.where(ok, pip, 1000.0), case, correlation, p1, p2, u, hc)
        bhp = Pb[:, -1]
    else:
        bhp = pip.copy()
    nan = lambda a: np.where(ok, a, np.nan)
    return {"bhp": nan(bhp), "pip": nan(pip), "pdis": pdis, "pn": pn}
