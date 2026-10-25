"""Hydraulic submersible pump (HSP, hydraulic drive downhole pump): a centrifugal pump driven by a turbine on the same
shaft; power fluid (water) goes down the tubing-casing annulus to the turbine and exhausts into the production tubing
above the pump. After PROSPER Design | Hydraulic Drive Downhole Pump and its System button (Tutorial 27).

Found from the user's Tutorial 27 runs (design screen and system analysis at 7500 rpm, CLYDE UNION TP115AH pump,
T60 turbine setting A, 60 % water cut, GOR 800, power fluid 100 % of the design rate):

- Pump: PROSPER's database curves (head and power per stage, fifth order in the rate at the reference speed) with the
  affinity laws for the speed (rate x N/Nref, head x (N/Nref)^2, power x (N/Nref)^3). Power = stages x catalogue
  power per stage x fluid gravity (exact against PROSPER's design power); efficiency (as PROSPER reports it) = rate x
  head / (136,000 x power per stage). Stages = head required / head per stage at the average rate, rounded up.
- Turbine: PROSPER's database curves per blade setting (head loss and power per stage, quadratic, and efficiency,
  sixth order, in the net turbine flow at the reference speed) with the same affinity laws. Net turbine flow = power
  fluid rate x Bw at the top node - the balance supply (3.5 % of the power fluid rate, PROSPER's 'Balance Supply
  Rate'): PROSPER's system run (6516.2 STB/d) reproduces head loss, power and efficiency at one flow, 6463 bbl/d.
  Turbine stages = pump power / turbine power per stage, rounded up. The pressure drop across the turbine is the head
  loss x the power fluid density at the turbine.
- Power fluid: water of the produced water's salinity (PROSPER's T27 traverse with the power fluid blended into the
  produced water reproduces its discharge pressure within 4 psi; 75,000 ppm gives 65 psi more). Down the annulus
  as for the jet pump (density at the production tubing's temperature over measured depth, Chen friction); the
  turbine exhausts at the pump discharge, so injection pressure = discharge pressure + turbine dP - static + friction.
- Pump duty as for the ESP and jet pump: intake from the IPR up the casing; discharge from the top node down the
  tubing with production + power fluid (% of the design rate).
- System: at each liquid rate the pump (stage by stage) sets its discharge pressure and power; the power fluid rate is
  the one whose turbine gives that power at the speed (PROSPER: 'varying the power fluid injected until the turbine
  power equals the pump power'); the solution is where that discharge pressure equals the tubing's with production +
  that power fluid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import blackoil as bo
from . import vlp as V
from . import esp as E
from . import jetpump as J

DATA = Path(__file__).parent / "data"
BALANCE_FRAC = 0.035                 # balance supply rate / power fluid rate (PROSPER: 227.5 of 6500, 228.1 of 6516.2)
HHP_DISPLAY = 136_000.0              # PROSPER's pump efficiency = q H / (136,000 x power per stage)
HHP_EXACT = 144.0 / (E.DP_HP * E.RHO_W)  # efficiency table for the ESP machinery: power = catalogue power x gravity
BTU_HP = 2544.43


# ================================================================ equipment
@dataclass
class HSPPump:
    manufacturer: str
    model: str
    od: float
    ref_speed: float
    min_speed: float
    max_speed: float
    q_min: float                     # at the reference speed, RB/d
    q_max: float
    max_stages: int
    hc: tuple                        # head per stage (ft) polynomial in RB/d at the reference speed, highest first
    pc: tuple                        # power per stage (hp, water)
    curve: E.Pump = field(repr=False, default=None)

    @property
    def name(self):
        return f"{self.manufacturer} {self.model} ({self.min_speed:g}-{self.max_speed:g} rpm)"

    def freq(self, speed):
        """ESP-equivalent frequency: the reference speed plays the 60 Hz curve (affinity laws)."""
        return 60.0 * np.asarray(speed, float) / self.ref_speed

    def head_stage(self, q, speed, wear=0.0):
        return self.curve.head_stage(q, self.freq(speed), wear)

    def power_stage(self, q, speed, sg=1.0):
        r = np.asarray(speed, float) / self.ref_speed
        return np.polyval(self.pc, np.asarray(q, float) / r) * r ** 3 * sg

    def efficiency(self, q, speed):
        """As PROSPER reports it: rate x head / (136,000 x power per stage), fraction."""
        r = np.asarray(speed, float) / self.ref_speed
        qr = np.asarray(q, float) / r
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.clip(qr * np.polyval(self.hc, qr) / (HHP_DISPLAY * np.polyval(self.pc, qr)), 0.0, 1.0)

    def range_at(self, speed):
        r = float(speed) / self.ref_speed
        return self.q_min * r, self.q_max * r


@dataclass
class HSPTurbine:
    manufacturer: str
    model: str
    setting: str
    ref_speed: float
    max_speed: float
    casing_od: float
    max_stages: int
    q_min: float                     # net turbine flow at the reference speed, bbl/d
    q_max: float
    best_eff: float
    hc: tuple                        # head loss per stage (ft) in the net flow at the reference speed
    wc: tuple                        # power per stage (hp)
    ec: tuple                        # efficiency (%)
    max_supply: float = np.nan
    max_work_p: float = np.nan
    max_stage_p: float = np.nan

    @property
    def name(self):
        lo, hi = self.q_min * 0.75, self.q_max * 0.75
        return f"{self.manufacturer} {self.model} - {self.setting}"

    def _r(self, speed):
        return np.asarray(speed, float) / self.ref_speed

    def head_stage(self, qn, speed):
        r = self._r(speed)
        return np.polyval(self.hc, np.asarray(qn, float) / r) * r * r

    def power_stage(self, qn, speed):
        r = self._r(speed)
        return np.polyval(self.wc, np.asarray(qn, float) / r) * r ** 3

    def efficiency(self, qn, speed):
        return np.polyval(self.ec, np.asarray(qn, float) / self._r(speed)) / 100.0

    def range_at(self, speed):
        r = float(speed) / self.ref_speed
        return self.q_min * r, self.q_max * r


def _pump_curve(name, od, q_min, q_max, hc, pc, n_q=80):
    """ESP Pump object on the reference-speed curve (the '60 Hz' table), efficiency table so that the ESP stage-by-
    stage power equals catalogue power x gravity."""
    q_hi = 2.0 * q_max
    q = np.linspace(0.0, q_hi, 800)
    h = np.polyval(hc, q)
    neg = np.nonzero(h <= 0)[0]
    q_run = float(q[neg[0]]) if len(neg) and neg[0] > 0 else q_hi
    q = np.linspace(0.0, q_run, n_q)
    h = np.maximum(np.polyval(hc, q), 0.0)
    p = np.polyval(pc, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(p > 0, q * h / (HHP_EXACT * p), 0.0)
    e = np.clip(np.nan_to_num(e), 0.0, None)
    j = int(np.argmax(e))
    return E.Pump(name, od, q_min, q_max, tuple(q), tuple(h), float(q[j]), float(e[j]), tuple(e),
                  source="PROSPER HSP pump database", power60=tuple(p))


@lru_cache(maxsize=1)
def _load():
    pumps, turbines = {}, {}
    try:
        df = pd.read_csv(DATA / "hsp_pumps.csv")
    except OSError:
        df = pd.DataFrame()
    for _, r in df.iterrows():
        st = max(float(r["stages"]), 1.0)
        hc = tuple(float(r[f"h{k}"]) / st for k in range(1, 7))
        pc = tuple(float(r[f"p{k}"]) / st for k in range(1, 7))
        p = HSPPump(str(r["manufacturer"]), str(r["name"]), float(r["od"]), float(r["ref_speed"]),
                    float(r["min_speed"]), float(r["max_speed"]), float(r["q_min"]), float(r["q_max"]),
                    int(r["max_stages"]), hc, pc)
        p.curve = _pump_curve(p.name, p.od, p.q_min, p.q_max, hc, pc)
        pumps[p.name] = p
    try:
        dt = pd.read_csv(DATA / "hsp_turbines.csv")
    except OSError:
        dt = pd.DataFrame()
    for _, r in dt.iterrows():
        t = HSPTurbine(str(r["manufacturer"]), str(r["name"]), str(r["setting"]), float(r["ref_speed"]),
                       float(r["max_speed"]), float(r["casing_od"]), int(r["max_stages"]), float(r["q_min"]),
                       float(r["q_max"]), float(r["best_eff"]), (float(r["h2"]), float(r["h1"]), float(r["h0"])),
                       (float(r["w2"]), float(r["w1"]), float(r["w0"])),
                       tuple(float(r[f"e{k}"]) for k in range(6, -1, -1)), float(r["max_supply"]),
                       float(r["max_work_p"]), float(r["max_stage_p"]))
        turbines[t.name] = t
    return pumps, turbines


PUMPS, TURBINES = _load()


# ================================================================ power fluid
def water_inp(salinity):
    return bo.PVTInputs(rsb=1, api=35, gas_sg=0.7, salinity_ppm=float(salinity))


def water_bw_rho(salinity, p, t):
    bw, rho, mu = bo.water_props(water_inp(salinity), np.array([float(p)]), np.array([float(t)]))
    return float(np.ravel(bw)[0]), float(np.ravel(rho)[0]), float(np.ravel(mu)[0])


def net_turbine_flow(q_pf, bw_top):
    """Net turbine flow (bbl/d): power fluid rate at the top node conditions less the balance supply."""
    return np.asarray(q_pf, float) * (bw_top - BALANCE_FRAC)


def injection_pressure(above, t_tubing, p_inlet, q_pf, salinity, casing_id, tubing_od):
    """Surface injection pressure (psia) that delivers p_inlet (psia) at the turbine with q_pf (STB/d) down the
    annulus (jet pump power fluid column: static over MD at the tubing temperature, Chen friction)."""
    d = J.JetDutyInput(q_pf=float(q_pf), p_surf=max(float(p_inlet) - 3000.0, 100.0),
                       power_fluid=J.PowerFluid(float(salinity)), casing_id=casing_id, tubing_od=tubing_od)
    pf = None
    for _ in range(8):
        pf = J.power_fluid_column(above, np.asarray(t_tubing, float), d, p_inlet, p_inlet)
        err = float(p_inlet) - pf["pn"]
        d.p_surf += err
        if abs(err) < 0.01:
            break
    return d.p_surf, pf


# ================================================================ design
@dataclass
class HSPDutyInput:
    pump_md: float = 8500.0
    q: float = 6500.0                 # design liquid rate, STB/d
    wc: float = 0.6
    gor: float = 800.0
    top_p: float = 164.696            # psia
    speed: float = 7500.0             # rpm (pump and turbine)
    pf_pct: float = 100.0             # power fluid, % of the design (reservoir) liquid rate
    wear: float = 0.0
    salinity: float | None = None     # power fluid; None = the produced water's
    casing_id: float = 6.4
    tubing_od: float = 4.5
    max_pump_od: float = 6.8
    max_turbine_od: float = 6.8

    @property
    def q_pf(self):
        return self.q * self.pf_pct / 100.0


def _salinity(fluid, s):
    if s is not None:
        return float(s)
    return float(fluid.m.inp.salinity_ppm) if hasattr(fluid, "m") else 75000.0


def duty(path, fluid, pwf, d: HSPDutyInput, correlation="pe2_style", p1=1.0, p2=1.0, u=8.0, hc=None):
    """PROSPER 'Calculate Pump Duty' for an HSP: the jet pump duty (power fluid blended into the tubing) with the
    HSP's power fluid. Returns the JetDuty (its nozzle / M / H fields unused)."""
    sal = _salinity(fluid, d.salinity)
    jd = J.JetDutyInput(pump_md=d.pump_md, q=d.q, wc=d.wc, gor=d.gor, top_p=d.top_p, q_pf=d.q_pf,
                        p_surf=2000.0, power_fluid=J.PowerFluid(sal), casing_id=d.casing_id, tubing_od=d.tubing_od)
    du = J.duty(path, fluid, pwf, jd, correlation, p1, p2, u, hc)
    du.detail["salinity"] = sal
    case = {"q": np.array([d.q]), "wc": np.array([d.wc]), "gor": np.array([d.gor])}
    ql, qg, _ = E._rates(fluid, du.pdis, du.t_in, case)
    du.detail["q_out_prod"] = ql + qg
    return du


def duty_table(du, d: HSPDutyInput, g=14.696):
    gvf = du.detail.get("free_gas_in", np.nan) / max(du.q_in, 1e-9)
    rows = {"Well head pressure (psig)": d.top_p - g, "Flowing BH pressure (psig)": du.pwf - g,
            "Pump intake pressure (psig)": du.pip - g, "Pump intake rate (RB/d)": du.q_in,
            "Free GOR entering pump (scf/STB)": du.free_gor, "Pump discharge pressure (psig)": du.pdis - g,
            "Pump discharge rate (RB/d)": du.detail.get("q_out_prod", np.nan), "Total GOR above pump (scf/STB)": d.gor,
            "Mass flow rate (lbm/d)": du.mass, "Total fluid gravity": du.sg, "Average downhole rate (RB/d)": du.q_avg,
            "Head required (ft)": du.head, "Fluid power required (hp)": du.fluid_hp,
            "GLR at pump intake (V/V)": du.glr_in, "Bo at pump intake (RB/STB)": du.bo_in,
            "Inlet temperature (F)": du.t_in, "Gas void fraction at pump intake": gvf,
            "Bg at pump intake (ft3/scf)": du.bg_in, "Power fluid rate (STB/d)": d.q_pf,
            "Power fluid salinity (ppm)": du.detail.get("salinity", np.nan)}
    return pd.DataFrame({"Value": rows})


def candidate_pumps(du, d: HSPDutyInput, pumps=None):
    """PROSPER's list: the speed within the pump's range, the discharge (and intake) rate within its range at that
    speed, OD below the maximum."""
    pumps = PUMPS if pumps is None else pumps
    rows = []
    for name, p in pumps.items():
        lo, hi = p.range_at(d.speed)
        h = float(p.head_stage(du.q_avg, d.speed, d.wear))
        n = int(np.ceil(du.head / h - 1e-9)) if h > 0 else np.nan
        ok = (p.min_speed <= d.speed <= p.max_speed and p.od <= d.max_pump_od and lo <= du.q_in <= hi
              and lo <= du.detail.get("q_out_prod", du.q_avg) <= hi and h > 0)
        rows.append({"Pump": name, "OD (in)": p.od, "Range at speed (RB/d)": f"{lo:,.0f}-{hi:,.0f}",
                     "Head per stage (ft)": h, "Stages": n,
                     "Efficiency (%)": 100 * float(p.efficiency(du.q_avg, d.speed)),
                     "Within maximum stages": bool(np.isfinite(n) and n <= p.max_stages * 999), "Suitable": ok})
    out = pd.DataFrame(rows)
    return out.sort_values(["Suitable", "OD (in)"], ascending=[False, True]).reset_index(drop=True) if len(out) else out


def candidate_turbines(d: HSPDutyInput, turbines=None, bw_top=1.0):
    turbines = TURBINES if turbines is None else turbines
    qn = float(net_turbine_flow(d.q_pf, bw_top))
    rows = []
    for name, t in turbines.items():
        lo, hi = t.range_at(d.speed)
        rows.append({"Turbine": name, "Casing OD (in)": t.casing_od,
                     "Net flow range at speed (bbl/d)": f"{lo:,.0f}-{hi:,.0f}", "Net turbine flow (bbl/d)": qn,
                     "Power per stage (hp)": float(t.power_stage(qn, d.speed)),
                     "Efficiency (%)": 100 * float(t.efficiency(qn, d.speed)),
                     "Suitable": bool(t.casing_od <= d.max_turbine_od and lo <= qn <= hi and d.speed <= t.max_speed)})
    out = pd.DataFrame(rows)
    return out.sort_values(["Suitable"], ascending=False).reset_index(drop=True) if len(out) else out


def turbine_at(t: HSPTurbine, q_pf, speed, bw_top, n_stages, rho_pf):
    qn = float(net_turbine_flow(q_pf, bw_top))
    head = n_stages * float(t.head_stage(qn, speed))
    return {"q_net": qn, "power": n_stages * float(t.power_stage(qn, speed)), "head": head,
            "dp": head * rho_pf / 144.0, "eff": float(t.efficiency(qn, speed)), "balance": BALANCE_FRAC * q_pf}


def design(path, fluid, du, d: HSPDutyInput, pump: HSPPump, turbine: HSPTurbine, hc=None, u=8.0, stages=None,
           turbine_stages=None):
    """Pump stages, power and efficiency; turbine stages, power, head loss, pressure drop and the surface power fluid
    pressure (PROSPER's HSP design results)."""
    hc = hc or V.HeatCapacities()
    N = d.speed
    h = float(pump.head_stage(du.q_avg, N, d.wear))
    if h <= 0:
        raise ValueError("The pump gives no head at the average rate.")
    n = int(stages) if stages else int(np.ceil(du.head / h - 1e-9))
    power = n * float(pump.power_stage(du.q_avg, N, du.sg))
    eff = float(pump.efficiency(du.q_avg, N))
    hyd = du.fluid_hp * n * h / du.head
    case = {"q": np.array([d.q]), "wc": np.array([d.wc]), "gor": np.array([d.gor])}
    _, cp = fluid.mass_cp(case, hc)
    t_out = du.t_in + max(power - hyd, 0.0) * BTU_HP / max(du.mass / 24.0 * float(np.ravel(cp)[0]), 1e-9)
    sal = du.detail.get("salinity", _salinity(fluid, d.salinity))
    bw_top = water_bw_rho(sal, d.top_p, du.detail.get("t_wellhead", 150.0))[0]
    qn = float(net_turbine_flow(d.q_pf, bw_top))
    p_st = float(turbine.power_stage(qn, N))
    if p_st <= 0:
        raise ValueError("The turbine gives no power at this power fluid rate and speed.")
    nt = int(turbine_stages) if turbine_stages else int(np.ceil(power / p_st - 1e-9))
    rho_t = water_bw_rho(sal, du.pdis, du.t_in)[1]
    tb = turbine_at(turbine, d.q_pf, N, bw_top, nt, rho_t)
    above = E.sub_path(path, path.md[0], d.pump_md)
    t_tub = np.interp(above.md, [above.md[0], above.md[-1]], [du.detail.get("t_wellhead", du.t_in), du.t_in])
    p_surf, pf = injection_pressure(above, t_tub, du.pdis + tb["dp"], d.q_pf, sal, d.casing_id, d.tubing_od)
    lo, hi = pump.range_at(N)
    tlo, thi = turbine.range_at(N)
    notes = []
    if n > pump.max_stages:
        notes.append(f"{n} pump stages: more than the {pump.max_stages} of one pump section in the database "
                     "(PROSPER does not limit it).")
    if nt > turbine.max_stages:
        notes.append(f"{nt} turbine stages: more than the turbine's database maximum of {turbine.max_stages} "
                     "(PROSPER does not limit it).")
    if not lo <= du.q_avg <= hi:
        notes.append(f"Average rate {du.q_avg:,.0f} RB/d is outside the pump's range at {N:g} rpm ({lo:,.0f}-{hi:,.0f}).")
    if not tlo <= qn <= thi:
        notes.append(f"Net turbine flow {qn:,.0f} bbl/d is outside the turbine's range at {N:g} rpm "
                     f"({tlo:,.0f}-{thi:,.0f}).")
    if turbine.casing_od > d.max_turbine_od:
        notes.append("The turbine is larger than the maximum turbine OD.")
    return {"Number of pump stages": n, "Head per stage (ft)": h, "Pump power required (hp)": power,
            "Pump efficiency (%)": 100 * eff, "Pump outlet temperature (F)": t_out,
            "Number of turbine stages": nt, "Turbine power generated (hp)": tb["power"],
            "Delta P across turbine (psi)": tb["dp"], "Head loss across turbine (ft)": tb["head"],
            "Pressure at surface (psia)": p_surf, "% Power fluid of reservoir fluid": d.pf_pct,
            "Balance supply rate (STB/d)": tb["balance"], "Turbine efficiency (%)": 100 * tb["eff"],
            "Turbine flow rate (STB/d)": d.q_pf, "Net turbine flow (bbl/d)": qn,
            "Power fluid static (psi)": pf["static"], "Power fluid friction (psi)": -pf["friction"],
            "Torque on shaft (lb.ft)": 5252.0 * power / N, "notes": notes}


# ================================================================ the HSP in the well
@dataclass
class HSPInstalled:
    pump_md: float = 8500.0
    pump: str = ""
    stages: int = 28
    turbine: str = ""
    turbine_stages: int = 22
    speed: float = 7500.0
    wear: float = 0.0
    salinity: float | None = None
    casing_id: float = 6.4
    tubing_od: float = 4.5

    @property
    def pump_obj(self) -> HSPPump:
        return PUMPS[self.pump]

    @property
    def turbine_obj(self) -> HSPTurbine:
        return TURBINES[self.turbine]


class _NoMotor:
    """The pump's losses only heat the fluid (no electric motor)."""
    hp = 1e12
    load = (0.0, 1e9)
    eff = (100.0, 100.0)


def pump_side(path, fluid, pwf, q, wc, gor, hsp: HSPInstalled, correlation, p1, p2, u, hc):
    """Intake, discharge pressure and power of the pump at liquid rate q (stage by stage)."""
    p = hsp.pump_obj
    f = float(p.freq(hsp.speed))
    pdp, info = E.discharge_pressure(path, fluid, pwf, q, wc, gor, hsp.pump_md, p.curve, hsp.stages, f, hsp.wear,
                                     0.0, correlation, p1, p2, u, hc, "stages", _NoMotor())
    return pdp, info


def power_fluid_for(turbine: HSPTurbine, n_t, speed, power, bw_top, q_hi=None):
    """Power fluid rate (STB/d) at which n_t turbine stages give the pump power at the speed."""
    lo_q = 1.0
    q_hi = q_hi or turbine.q_max * speed / turbine.ref_speed * 3.0 / max(bw_top - BALANCE_FRAC, 0.5)
    f = lambda qpf: n_t * float(turbine.power_stage(net_turbine_flow(qpf, bw_top), speed)) - power
    # the power curve is a parabola with its minimum below the operating range: search the rising branch
    qs = np.linspace(lo_q, q_hi, 200)
    v = np.array([f(x) for x in qs])
    k = None
    for j in range(len(v) - 1, 0, -1):
        if v[j - 1] < 0 <= v[j]:
            k = j
            break
    if k is None:
        return np.nan
    return brentq(f, qs[k - 1], qs[k], xtol=0.01)


def _state(path, fluid, ipr_pwf, q, wc, gor, top_p, hsp: HSPInstalled, correlation, p1, p2, u, hc, t_wh=None):
    pwf = float(ipr_pwf(q))
    if not np.isfinite(pwf):
        return None
    pdp, info = pump_side(path, fluid, pwf, q, wc, gor, hsp, correlation, p1, p2, u, hc)
    if not np.isfinite(pdp):
        return None
    sal = _salinity(fluid, hsp.salinity)
    above = E.sub_path(path, path.md[0], hsp.pump_md)
    t_dis = info["Discharge temperature (F)"]
    t_wh = t_wh if t_wh is not None else float(path.t_form[0]) + 0.8 * (t_dis - float(path.t_form[0]))
    out = None
    for _ in range(3):
        bw_top = water_bw_rho(sal, top_p, t_wh)[0]
        qpf = power_fluid_for(hsp.turbine_obj, hsp.turbine_stages, hsp.speed, info["Pump power (hp)"], bw_top)
        if not np.isfinite(qpf):
            return None
        flm = J._blend_fluid(fluid, q, wc, qpf, sal)
        qt = q + qpf
        case_t = {"q": np.array([qt]), "wc": np.array([(q * wc + qpf) / qt]), "gor": np.array([gor])}
        P, T = V.traverse(above, flm, np.array([top_p]), case_t, correlation, p1, p2, u, hc, t_dis)
        new_wh = float(T[0, 0])
        out = (pwf, pdp, info, qpf, bw_top, float(P[0, -1]), np.asarray(T[0]), above, sal)
        if abs(new_wh - t_wh) < 0.2:
            break
        t_wh = new_wh
    return out


def system(path, fluid, ipr_pwf, wc, gor, top_p, hsp: HSPInstalled, correlation="pe2_style", p1=1.0, p2=1.0, u=8.0,
           hc=None, q_max=None, n=30):
    """PROSPER's HSP system calculation: pump speed fixed, the power fluid injected varied so the turbine power equals
    the pump power; the solution is where the pump's discharge pressure (stage by stage from the IPR's intake) equals
    the tubing's (top node down with production + power fluid). Returns (solution dict or None, curve)."""
    hc = hc or V.HeatCapacities()
    q_hi = q_max or 20000.0
    qs = np.linspace(q_hi * 0.03, q_hi, n)
    rows = []
    for q in qs:
        s = _state(path, fluid, ipr_pwf, q, wc, gor, top_p, hsp, correlation, p1, p2, u, hc)
        if s is None:
            if rows:
                break
            continue
        pwf, pdp, info, qpf, bw, pdt, _, _, _ = s
        rows.append({"Liquid rate (STB/d)": q, "Pump discharge from the pump (psia)": pdp,
                     "Discharge needed by the tubing (psia)": pdt, "Pump intake (psia)": info["PIP"],
                     "Power fluid rate (STB/d)": qpf, "Pump power (hp)": info["Pump power (hp)"]})
    curve = pd.DataFrame(rows)
    if len(curve) < 2:
        return None, curve
    f = (curve["Pump discharge from the pump (psia)"] - curve["Discharge needed by the tubing (psia)"]).to_numpy()
    qv = curve["Liquid rate (STB/d)"].to_numpy()
    k = None
    for j in range(len(f) - 1, 0, -1):
        if f[j - 1] > 0 >= f[j]:
            k = j
            break
    if k is None:
        return None, curve
    g = lambda q: (lambda s: s[1] - s[5])(_state(path, fluid, ipr_pwf, q, wc, gor, top_p, hsp, correlation, p1, p2,
                                                 u, hc))
    q_sol = brentq(g, qv[k - 1], qv[k], xtol=0.5)
    pwf, pdp, info, qpf, bw, pdt, T_tub, above, sal = _state(path, fluid, ipr_pwf, q_sol, wc, gor, top_p, hsp,
                                                             correlation, p1, p2, u, hc)
    p, t = hsp.pump_obj, hsp.turbine_obj
    N = hsp.speed
    mass = info["Average rate (RB/d)"] * E.BBL_FT3 * info["Gravity"] * E.RHO_W
    rho_avg = info["Gravity"] * E.RHO_W
    rho_t = water_bw_rho(sal, pdp, info["Intake temperature (F)"])[1]
    tb = turbine_at(t, qpf, N, bw, hsp.turbine_stages, rho_t)
    p_surf, pf = injection_pressure(above, T_tub, pdp + tb["dp"], qpf, sal, hsp.casing_id, hsp.tubing_od)
    q_avg = info["Average rate (RB/d)"]
    pw = info["Pump power (hp)"]
    case = {"q": np.array([q_sol]), "wc": np.array([wc]), "gor": np.array([gor])}
    ql_d, qg_d, _ = E._rates(fluid, pdp, info["Intake temperature (F)"], case)
    lo, hi = p.range_at(N)
    sol = {"Liquid rate (STB/d)": q_sol, "Oil rate (STB/d)": q_sol * (1 - wc), "Water rate (STB/d)": q_sol * wc,
           "Gas rate (MMscf/d)": q_sol * (1 - wc) * gor / 1e6, "BHP (psia)": pwf,
           "Pump intake pressure (psia)": info["PIP"], "Pump discharge pressure (psia)": pdp,
           "Pump speed (rpm)": N, "Pump intake rate (RB/d)": info["Intake rate (RB/d)"],
           "Average rate through pump (RB/d)": q_avg, "Pump discharge rate (RB/d)": ql_d + qg_d,
           "Pump head generated (ft)": (pdp - info["PIP"]) * 144.0 / rho_avg, "Pump power (hp)": pw,
           "Pump efficiency (%)": 100 * float(p.efficiency(q_avg, N)),
           "Pump torque (lb.ft)": 5252.0 * pw / N, "Free GOR at pump intake (scf/STB)": info["Free GOR (scf/STB)"],
           "Pump intake temperature (F)": info["Intake temperature (F)"],
           "Pump discharge temperature (F)": info["Discharge temperature (F)"],
           "Pump delta T (F)": info["Discharge temperature (F)"] - info["Intake temperature (F)"],
           "Power fluid injection rate (STB/d)": qpf, "Net turbine flow (bbl/d)": tb["q_net"],
           "Turbine power generated (hp)": tb["power"], "Turbine efficiency (%)": 100 * tb["eff"],
           "Delta P across turbine (psi)": tb["dp"], "Head loss across turbine (ft)": tb["head"],
           "Balance supply rate (STB/d)": tb["balance"], "Casing head pressure (psia)": p_surf,
           "Power fluid static (psi)": pf["static"], "Power fluid friction (psi)": -pf["friction"],
           "Wellhead temperature (F)": float(T_tub[0]), "Total fluid gravity": info["Gravity"],
           "In recommended range": bool(lo <= q_avg <= hi)}
    return sol, curve


def traverse_profile(path, fluid, top_p, q, wc, gor, hsp: HSPInstalled, qpf=None, correlation="pe2_style", p1=1.0,
                     p2=1.0, u=8.0, hc=None):
    """PROSPER's HSP gradient traverse at liquid rate q: tubing from the top node to the pump discharge with
    production + power fluid (starting at the pump discharge temperature), the intake from the pump curve (stage by
    stage), casing to the bottom node. qpf: power fluid rate (STB/d); None = the rate whose turbine power equals the
    pump power at the speed (iterated, as the system calculation). Returns (summary dict, profile DataFrame)."""
    hc = hc or V.HeatCapacities()
    p, tb_ = hsp.pump_obj, hsp.turbine_obj
    f = float(p.freq(hsp.speed))
    sal = _salinity(fluid, hsp.salinity)
    above = E.sub_path(path, path.md[0], hsp.pump_md)
    below = E.sub_path(path, hsp.pump_md, path.md[-1])
    case = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([gor])}
    w, cp = fluid.mass_cp(case, hc)
    t_in = float(V.temperature_profile(below, np.asarray(w, float), cp, u)[0, 0])
    fixed = qpf is not None
    qpf = float(qpf) if fixed else float(q)
    t_dis = t_in
    pip = pdp = np.nan
    for _ in range(12):
        flm = J._blend_fluid(fluid, q, wc, qpf, sal)
        qt = q + qpf
        case_t = {"q": np.array([qt]), "wc": np.array([(q * wc + qpf) / qt]), "gor": np.array([gor])}
        P, T = V.traverse(above, flm, np.array([top_p]), case_t, correlation, p1, p2, u, hc, t_dis)
        pdp = float(P[0, -1])
        pip = float(E.intake_from_discharge(fluid, np.array([pdp]), np.array([t_in]), case, p.curve, hsp.stages, f,
                                            hsp.wear, iters=22)[0])       # 22 halvings: 0.001 psi
        if not np.isfinite(pip):
            raise ValueError(f"At {q:g} STB/d the pump would draw the intake below atmospheric.")
        _, hyd, pw, qav = E.pump_gain(fluid, np.array([pip]), np.array([t_in]), case, p.curve, hsp.stages, f, hsp.wear)
        t_new = t_in + max(float(pw[0]) - float(hyd[0]), 0.0) * BTU_HP / max(
            float(np.ravel(w)[0]) * float(np.ravel(cp)[0]), 1e-9)
        q_new = qpf
        if not fixed:
            bw_top = water_bw_rho(sal, top_p, float(T[0, 0]))[0]
            q_new = power_fluid_for(tb_, hsp.turbine_stages, hsp.speed, float(pw[0]), bw_top)
            if not np.isfinite(q_new):
                raise ValueError(f"At {q:g} STB/d the turbine cannot give the pump power ({float(pw[0]):.1f} hp) "
                                 "within its flow range.")
        done = abs(t_new - t_dis) < 0.05 and abs(q_new - qpf) < 0.5
        t_dis, qpf = t_new, q_new
        if done:
            break
    flm = J._blend_fluid(fluid, q, wc, qpf, sal)
    qt = q + qpf
    case_t = {"q": np.array([qt]), "wc": np.array([(q * wc + qpf) / qt]), "gor": np.array([gor])}
    P, T = V.traverse(above, flm, np.array([top_p]), case_t, correlation, p1, p2, u, hc, t_dis)
    Pb, Tb = V.traverse(below, fluid, np.array([pip]), case, correlation, p1, p2, u, hc)
    prof = pd.DataFrame({"MD (ft)": np.r_[above.md, below.md], "TVD (ft)": np.r_[above.tvd, below.tvd],
                         "Pressure (psia)": np.r_[P[0], Pb[0]], "Temperature (F)": np.r_[T[0], Tb[0]],
                         "Section": ["Tubing (production + power fluid)"] * len(above.md) + ["Casing"] * len(below.md)})
    bw_top = water_bw_rho(sal, top_p, float(T[0, 0]))[0]
    rho_t = water_bw_rho(sal, pdp, t_in)[1]
    tb = turbine_at(tb_, qpf, hsp.speed, bw_top, hsp.turbine_stages, rho_t)
    p_surf, pf = injection_pressure(above, T[0], pdp + tb["dp"], qpf, sal, hsp.casing_id, hsp.tubing_od)
    summ = {"Pump discharge pressure (psia)": pdp, "Pump intake pressure (psia)": pip,
            "Bottom node pressure (psia)": float(Pb[0, -1]), "Wellhead temperature (F)": float(T[0, 0]),
            "Pump discharge temperature (F)": t_dis, "Pump intake temperature (F)": t_in,
            "Pump power (hp)": float(pw[0]), "Average rate through pump (RB/d)": float(qav[0]),
            "Power fluid rate (STB/d)": qpf, "Turbine power (hp)": tb["power"], "Delta P across turbine (psi)": tb["dp"],
            "Power fluid injection pressure (psia)": p_surf, "Power fluid static (psi)": pf["static"],
            "Power fluid friction (psi)": -pf["friction"], "salinity": sal}
    return summ, prof
