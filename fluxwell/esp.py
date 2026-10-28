"""Electrical submersible pump (ESP): equipment, pump duty, design and the pump in the well model.

After PROSPER Design | Electrical Submersible Pump (User Guide 2.13.2, Tutorial 26):

Pump duty ("Calculate Pump Duty"):
  - flowing BHP from the IPR at the design rate; gradient up the casing from the bottom node to the pump: pump intake
    pressure (PIP);
  - the gas separator removes a fraction of the free gas at the intake (to the annulus): the GOR above the pump is
    reduced accordingly;
  - gradient down the tubing from the top node pressure to the pump: pump discharge pressure (PDP);
  - intake / discharge rates at PIP / PDP; the average rate through the pump at the average pressure;
  - total fluid gravity = mass rate / average volume rate; head = (PDP - PIP) / gradient of that fluid;
  - fluid power = average rate x (PDP - PIP).
Design: stages = head / head per stage at the average rate (affinity laws for frequency, wear factor); power =
fluid power x (stages x head per stage / head) / pump efficiency; motor load, current, efficiency, power factor and
speed from the motor curves at that load; cable voltage drop (resistance x length x current, corrected to the
average cable temperature); surface voltage, kVA and torque.
In the well model the pump adds N x head per stage (affinity, wear) of fluid gradient at the average rate.

Equipment data are entered as tables. The included Centrilift GC10000 pump and 544-series 300 HP motor curves were
digitised from PROSPER's own plots (Tutorial 26) and are for checking against PROSPER; other pumps should be entered
from the manufacturer's published curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import vlp as V

BBL_FT3 = 5.614583
RHO_W = 62.4                     # lb/ft3, water for specific gravity
HHP_K = 1.0 / 135_773.0          # hydraulic hp = q (bbl/d) x H (ft) x SG x HHP_K
DP_HP = BBL_FT3 * 144.0 / (86400.0 * 550.0)   # hp per (bbl/d x psi)


# ================================================================ equipment
@dataclass
class Pump:
    name: str
    od: float                     # in
    q_min: float                  # recommended range at 60 Hz, RB/d
    q_max: float
    q: tuple                      # 60 Hz rate table, RB/d
    head: tuple                   # 60 Hz head per stage, ft
    q_bep: float                  # best efficiency rate at 60 Hz
    eff_bep: float                # best efficiency, fraction
    eff: tuple = ()               # optional 60 Hz efficiency table (fraction) on the q table
    source: str = ""
    power60: tuple = ()           # optional 60 Hz power per stage on water (hp) on the q table: the pump's power
    #                               is then this x gravity (affinity: x r^3), not hydraulic power / efficiency

    def power_stage(self, q, f=60.0):
        """Power per stage on water (hp) from the power table (affinity laws); None without one."""
        if len(self.power60) != len(self.q):
            return None
        r = f / 60.0
        return np.interp(np.asarray(q, float) / r, self.q, self.power60) * r ** 3

    @property
    def q_runout(self):
        return float(self.q[-1])

    def head60(self, q):
        return np.interp(q, self.q, self.head, right=0.0)

    def eff60(self, q):
        q = np.asarray(q, float)
        if len(self.eff) == len(self.q):
            return np.interp(q, self.q, self.eff, right=0.0)
        x = q / self.q_bep
        lo = self.eff_bep * x * (2.0 - x)
        hi = self.eff_bep * (1.0 - ((q - self.q_bep) / (self.q_runout - self.q_bep)) ** 2)
        return np.clip(np.where(q <= self.q_bep, lo, hi), 0.0, None)

    def head_stage(self, q, f=60.0, wear=0.0):
        """Head per stage (ft) at rate q (RB/d) and frequency f: affinity laws, wear factor."""
        r = f / 60.0
        return self.head60(np.asarray(q, float) / r) * r * r * (1.0 - wear)

    def efficiency(self, q, f=60.0):
        return self.eff60(np.asarray(q, float) * 60.0 / f)

    def range_at(self, f=60.0):
        return self.q_min * f / 60.0, self.q_max * f / 60.0


@dataclass
class Motor:
    name: str
    hp: float                     # nameplate at 60 Hz
    volts: float
    amps: float
    od: float = 5.44
    load: tuple = ()              # % nameplate power
    amps_pct: tuple = ()          # % nameplate amps
    eff: tuple = ()               # %
    pf: tuple = ()                # %
    rpm60: tuple = ()             # speed at 60 Hz, rpm
    source: str = ""

    def at(self, load_pct, f=60.0):
        lp = float(load_pct)
        g = lambda t: float(np.interp(lp, self.load, t))
        return {"amps_pct": g(self.amps_pct), "eff": g(self.eff) / 100.0, "pf": g(self.pf) / 100.0,
                "rpm": g(self.rpm60) * f / 60.0}


@dataclass
class Cable:
    name: str
    v_per_kft_amp: float          # voltage drop, V per 1000 ft per amp (at 77 F)
    max_amps: float
    alpha_f: float = 0.002025     # resistance temperature factor, 1/F (fits PROSPER Tutorial 26)
    t_ref: float = 77.0

    def drop(self, amps, length_ft, t_avg_f):
        return self.v_per_kft_amp * length_ft / 1000.0 * amps * (1.0 + self.alpha_f * (t_avg_f - self.t_ref))


# Centrilift GC10000 (513 series, 5.13 in): 60 Hz head per stage digitised from PROSPER's pump plot (Tutorial 26,
# four frequencies, consistent with the affinity laws within 1 %), anchored to 28.46 ft/stage at 9300.9 RB/d; best
# efficiency 65.93 % at 9429 RB/d (PROSPER's best efficiency line and design efficiency).
_GC_Q = tuple(range(0, 14001, 500)) + (14270,)
_GC_H = (46.64, 46.93, 46.58, 45.74, 44.56, 43.15, 41.62, 40.05, 38.51, 37.05, 35.71, 34.50, 33.43, 32.50, 31.69,
         30.97, 30.31, 29.65, 28.94, 28.12, 27.11, 25.86, 24.27, 22.28, 19.79, 16.72, 12.99, 8.52, 3.23, 0.0)
PUMPS = {
    "Centrilift GC10000 (5.13 in)": Pump("Centrilift GC10000 (5.13 in)", 5.13, 4400.0, 12000.0, _GC_Q, _GC_H,
                                         9428.8, 0.659273,
                                         source="digitised from PROSPER Tutorial 26 pump plot"),
}
# Centrilift 544-series 300 HP 1675 V 105 A, 3480 rpm: curves digitised from PROSPER's motor plot and adjusted to
# PROSPER's design results at 95.6 % and 96.3 % load.
_ML = tuple(range(0, 136, 5))      # above 100 % load: from PROSPER's Tutorial 26 system runs at 70 Hz
MOTORS = {
    "Centrilift 544 300HP 1675V 105A": Motor(
        "Centrilift 544 300HP 1675V 105A", 300.0, 1675.0, 105.0, 5.44, _ML,
        (56.5, 54.65, 53.1, 52.2, 52.7, 53.7, 54.85, 56.8, 59.0, 61.7, 64.9, 67.9, 71.0, 74.4, 77.7, 81.4, 85.0,
         88.7, 92.5, 96.4, 100.4, 105.3, 110.2, 115.1, 120.5, 126.5, 134.1, 142.7),
        (58.1, 60.1, 64.2, 66.1, 68.3, 70.5, 72.4, 74.1, 75.5, 77.0, 78.3, 79.4, 80.4, 81.6, 82.3, 82.8, 83.3,
         83.6, 83.85, 84.0, 84.15, 84.05, 83.95, 83.8, 83.7, 83.55, 83.2, 82.9),
        (45.3, 47.8, 50.5, 53.2, 55.6, 58.0, 60.5, 62.6, 65.0, 66.9, 68.85, 70.8, 72.55, 74.0, 75.4, 76.6, 77.9,
         78.6, 79.9, 81.0, 81.7, 82.2, 82.7, 83.2, 83.7, 84.15, 84.5, 84.9),
        (3363, 3423, 3462, 3494, 3511, 3527, 3536, 3537, 3533, 3529, 3525, 3518, 3511, 3504, 3494, 3486, 3479,
         3471, 3463, 3453, 3447, 3435, 3422, 3409, 3397, 3384, 3369, 3355),
        source="digitised from PROSPER Tutorial 26 motor plot"),
}


def motor_like(name, hp, volts, amps, template="Centrilift 544 300HP 1675V 105A", od=None):
    """A motor from its nameplate with the load curves (amps %, efficiency, power factor, slip) of a template."""
    t = MOTORS[template]
    rpm = tuple(3600.0 - (3600.0 - r) for r in t.rpm60)
    return Motor(name, float(hp), float(volts), float(amps), od or t.od, t.load, t.amps_pct, t.eff, t.pf, rpm,
                 source=f"nameplate; load curves of {template}")


# ================================================================ PROSPER's equipment databases (user's exports)
DATA = Path(__file__).parent / "data"
HHP_DB = 135_773.0               # PROSPER database: efficiency = rate x head / (135,773 x hp) at SG 1


@lru_cache(maxsize=1)
def database_pumps() -> pd.DataFrame:
    """PROSPER's ESP pump database (data/esp_pumps.csv): manufacturer, name, OD, base frequency, the number of stages
    the coefficients are for, recommended range and the fifth-order head (ft) and power (hp) polynomials in the rate
    (RB/d), highest power first. Most vendors' curves are protected in PROSPER and export without coefficients."""
    return pd.read_csv(DATA / "esp_pumps.csv")


def pump_from_coefficients(r, n_q=60):
    """A Pump (60 Hz tables) from a database row with curve coefficients; None if the curve is unusable."""
    hc = [float(r[f"h{k}"]) for k in range(1, 7)]
    pc = [float(r[f"p{k}"]) for k in range(1, 7)]
    if not np.all(np.isfinite(hc + pc)):
        return None
    st, f0 = max(float(r["stages"]), 1.0), float(r["hz"])
    k = 60.0 / f0                                    # affinity laws to 60 Hz
    q_hi = 3.0 * float(r["q_max"])
    q = np.linspace(0.0, q_hi, 600)
    h = np.polyval(hc, q) / st
    neg = np.nonzero(h <= 0)[0]
    q_run = float(q[neg[0]]) if len(neg) and neg[0] > 0 else 1.3 * float(r["q_max"])
    q = np.linspace(0.0, q_run, n_q)
    h = np.maximum(np.polyval(hc, q) / st, 0.0)
    p = np.polyval(pc, q) / st
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(p > 0, q * h / (HHP_DB * p), 0.0)
    e = np.clip(np.nan_to_num(e), 0.0, None)
    rng = (q >= float(r["q_min"])) & (q <= float(r["q_max"]))
    if not rng.any() or e[rng].max() > 0.95 or h[rng].min() <= 0:
        return None                                  # inconsistent database curve (e.g. efficiency above 95 %)
    name = f"{r['manufacturer']} {r['name']} ({float(r['od']):g} in)"
    j = int(np.argmax(np.where(rng, e, 0.0)))
    return Pump(name, float(r["od"]), float(r["q_min"]) * k, float(r["q_max"]) * k, tuple(q * k), tuple(h * k * k),
                float(q[j] * k), float(e[j]), tuple(e),
                source=f"PROSPER pump database ({f0:g} Hz curve, coefficients for {st:g} stage(s))")


def _motor_from_row(r):
    """ESP Motor (60 Hz equivalent nameplate) from a row of PROSPER's motor database: motors with coefficient tables
    use their polynomials in the load fraction, the others the Centrilift 544 load curves (PROSPER's default)."""
    f0 = float(r["hz"])
    k = 60.0 / f0
    name = f"{r['manufacturer']} {r['series']} {float(r['hp']):g}HP {float(r['volts']):g}V {float(r['amps']):g}A"
    if f0 != 60.0:
        name += f" ({f0:g} Hz)"
    src = (f"PROSPER motor database: {float(r['hp']):g} hp, {float(r['volts']):g} V at {f0:g} Hz, "
           f"{float(r['rpm']):g} rpm")
    co = [str(r.get(c, "") or "") for c in ("c_amps", "c_speed", "c_eff", "c_pf")]
    if all(c.strip() and c != "nan" for c in co):
        ca, cs, ce, cp = (np.array([float(v) for v in c.split()]) for c in co)
        load = np.arange(0.0, 136.0, 5.0)
        x = load / 100.0
        amps, eff, pf, rpm = (np.polyval(ca, x), np.polyval(ce, x), np.polyval(cp, x), np.polyval(cs, x) * k)
        sane = (0.7 < np.polyval(ca, 1.0) < 1.3 and 0.4 < np.polyval(ce, 1.0) < 1.0 and 0.3 < np.polyval(cp, 1.0) <= 1.0)
        if sane:
            return Motor(name, float(r["hp"]) * k, float(r["volts"]) * k, float(r["amps"]), float(r["od"]),
                         tuple(load), tuple(100 * amps), tuple(np.clip(100 * eff, 1, 99.9)),
                         tuple(np.clip(100 * pf, 1, 100)), tuple(rpm), source=src + "; its load curves")
        src += "; its coefficient table is inconsistent - Centrilift 544 load curves used"
    m = motor_like(name, float(r["hp"]) * k, float(r["volts"]) * k, float(r["amps"]), od=float(r["od"]))
    m.source = src if "inconsistent" in src else src + "; Centrilift 544 load curves (PROSPER's default)"
    return m


@lru_cache(maxsize=1)
def _database():
    pumps = {}
    for _, r in database_pumps().iterrows():
        p = pump_from_coefficients(r)
        if p is not None:
            pumps.setdefault(p.name, p)
    motors = {}
    for _, r in pd.read_csv(DATA / "motors.csv").iterrows():
        m = _motor_from_row(r)
        motors.setdefault(m.name, m)
    cables = {str(r["cable"]): Cable(str(r["cable"]), float(r["v_per_kft_amp"]), float(r["max_amps"]))
              for _, r in pd.read_csv(DATA / "cables.csv").iterrows()}
    return pumps, motors, cables


DB_PUMPS, DB_MOTORS, CABLES = _database()     # cables: PROSPER's list (V per 1000 ft per amp, maximum current)
for _k, _v in DB_PUMPS.items():
    PUMPS.setdefault(_k, _v)
for _k, _v in DB_MOTORS.items():
    MOTORS.setdefault(_k, _v)


def pump_from_table(name, od, q_min, q_max, q, head, eff=None):
    """User pump from a 60 Hz table of rate (RB/d), head per stage (ft) and optional efficiency (fraction)."""
    q, head = np.asarray(q, float), np.asarray(head, float)
    o = np.argsort(q)
    q, head = q[o], head[o]
    e = np.asarray(eff, float)[o] if eff is not None and len(eff) == len(q) else None
    if e is not None and np.any(np.isfinite(e)):
        k = int(np.nanargmax(e))
        q_bep, e_bep = float(q[k]), float(e[k])
        e_t = tuple(np.nan_to_num(e))
    else:
        q_bep, e_bep, e_t = 0.5 * (q_min + q_max), 0.6, ()
    return Pump(name, float(od), float(q_min), float(q_max), tuple(q), tuple(head), q_bep, e_bep, e_t,
                source="user table")


# ================================================================ flow path helpers
def sub_path(path: V.FlowPath, md_a, md_b) -> V.FlowPath:
    """Part of the flow path between two measured depths (nodes added at both ends)."""
    from .gaslift import with_nodes
    p = with_nodes(path, [md_a, md_b])
    i, j = int(np.argmin(np.abs(p.md - md_a))), int(np.argmin(np.abs(p.md - md_b)))
    restr = {k - i: d for k, d in p.restrictions.items() if i <= k <= j}
    return V.FlowPath(p.md[i:j + 1], p.tvd[i:j + 1], p.t_form[i:j + 1], p.seg_len[i:j], p.sin_t[i:j], p.area[i:j],
                      p.d_h[i:j], p.rough[i:j], p.d_ht[i:j], restr)


def _rates(fluid, p, t, case):
    """(liquid RB/d, free gas RB/d, solution GOR scf/STB) at p (psia), t (F)."""
    s = fluid.state(np.array([p]), np.array([t]), case)
    ql = float(s["ql"][0]) * 86400.0 / BBL_FT3
    qg = float(s["qg"][0]) * 86400.0 / BBL_FT3
    rs = np.nan
    if hasattr(fluid, "_props"):
        rs = float(fluid._props(np.array([p]), np.array([t]), case["gor"])[0][0])
    return ql, qg, rs


# ================================================================ pump duty
@dataclass
class Duty:
    q: float
    pwf: float
    pip: float
    pdp: float
    t_intake: float
    q_in: float
    q_out: float
    q_avg: float
    mass: float                   # lbm/d
    sg: float
    head: float
    fluid_hp: float
    free_gor: float
    gor_above: float
    glr_intake: float
    gas_frac: float
    t_cable: float
    detail: dict = field(default_factory=dict)


def pump_duty(path, fluid, pwf, q, wc, gor, top_p, pump_md, sep_eff=0.0, correlation="hagedorn_brown", p1=1.0,
              p2=1.0, u=8.0, hc=None):
    """PROSPER 'Calculate Pump Duty' (psia, STB/d, F). pwf: flowing BHP at the bottom node for q."""
    hc = hc or V.HeatCapacities()
    below = sub_path(path, pump_md, path.md[-1])
    above = sub_path(path, path.md[0], pump_md)
    case = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([gor])}
    # intake: from the bottom node up the casing to the pump
    Pb, Tb = V.traverse_up(below, fluid, np.array([pwf]), case, correlation, p1, p2, u, hc)
    pip, t_in = float(Pb[0, 0]), float(Tb[0, 0])
    if not np.isfinite(pip) or pip <= 14.7:
        raise ValueError("The fluid does not reach the pump: the intake pressure falls to atmospheric "
                         "(PROSPER reports a PIP of 10 psi).")
    ql_in, qg_in, rs_in = _rates(fluid, pip, t_in, case)
    free = max(gor - rs_in, 0.0) if np.isfinite(rs_in) else np.nan
    gor_above = gor - sep_eff * free if np.isfinite(free) else gor
    case_a = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([gor_above])}
    qg_in_pump = qg_in * (1.0 - sep_eff)
    # discharge: from the top node down the tubing to the pump
    P, T = V.traverse(above, fluid, np.array([top_p]), case_a, correlation, p1, p2, u, hc)
    pdp = float(P[0, -1])
    t_out = float(T[0, -1])
    ql_out, qg_out, _ = _rates(fluid, pdp, t_in, case_a)
    ql_av, qg_av, _ = _rates(fluid, 0.5 * (pip + pdp), t_in, case_a)
    q_in, q_out, q_avg = ql_in + qg_in_pump, ql_out + qg_out, ql_av + qg_av
    w_a, _ = fluid.mass_cp(case_a, hc)
    mass = float(w_a[0]) * 24.0
    rho = mass / (q_avg * BBL_FT3)
    sg = rho / RHO_W
    dp = pdp - pip
    head = dp * 144.0 / rho
    t_cable = 0.5 * (float(T[0, 0]) + t_in)
    return Duty(q, pwf, pip, pdp, t_in, q_in, q_out, q_avg, mass, sg, head, q_avg * dp * DP_HP, free, gor_above,
                qg_in_pump / max(ql_in, 1e-9), qg_in_pump / max(ql_in + qg_in_pump, 1e-9), t_cable,
                {"t_wellhead": float(T[0, 0]), "t_discharge_traverse": t_out, "liquid_in": ql_in})


def duty_table(d: Duty, g=14.696, freq=60.0):
    rows = [("Well Head Pressure", np.nan, "psig"), ("Flowing Bottomhole Pressure", d.pwf - g, "psig"),
            ("Pump Frequency", freq, "Hertz"),
            ("Pump Intake Pressure", d.pip - g, "psig"), ("Pump Intake Temperature", d.t_intake, "deg F"),
            ("Pump Intake Rate", d.q_in, "RB/day"), ("Free GOR Entering Pump", d.free_gor * 1.0, "scf/STB"),
            ("Pump Discharge Pressure", d.pdp - g, "psig"), ("Pump Discharge Rate", d.q_out, "RB/day"),
            ("Total GOR Above Pump", d.gor_above, "scf/STB"), ("Mass Flow Rate", d.mass, "lbm/day"),
            ("Total Fluid Gravity", d.sg, "sp. gravity"), ("Average Downhole Rate", d.q_avg, "RB/day"),
            ("Head Required", d.head, "feet"), ("Fluid Power Required", d.fluid_hp, "hp"),
            ("GLR @ Pump Intake (V/V)", d.glr_intake, "fraction"),
            ("Gas Fraction @ Pump Intake", d.gas_frac, "fraction"),
            ("Average Cable Temperature", d.t_cable, "deg F")]
    return pd.DataFrame(rows, columns=["Result", "Value", "Units"])


# ================================================================ design
def candidate_pumps(d: Duty, pumps: dict, freq=60.0, max_od=99.0, casing_id=99.0, wear=0.0):
    """Pumps whose range covers the intake and discharge rates, OD within limits (PROSPER's list)."""
    rows = []
    for name, p in pumps.items():
        lo, hi = p.range_at(freq)
        h = float(p.head_stage(d.q_avg, freq, wear))
        ok = (p.od <= max_od and p.od < casing_id and lo <= d.q_in <= hi and lo <= d.q_out <= hi and h > 0)
        rows.append({"Pump": name, "OD (in)": p.od, "Range at frequency (RB/d)": f"{lo:,.0f}-{hi:,.0f}",
                     "Head per stage (ft)": h, "Stages": int(np.ceil(d.head / h)) if h > 0 else np.nan,
                     "Efficiency (%)": 100 * float(p.efficiency(d.q_avg, freq)), "Suitable": ok})
    return pd.DataFrame(rows)


def design(d: Duty, pump: Pump, motor: Motor, cable: Cable, freq=60.0, cable_len=9000.0, wear=0.0,
           motor_margin=0.0, stages=None, casing_id=None, motor_od=None):
    """Stages, power, motor and cable (PROSPER ESP design results)."""
    h = float(pump.head_stage(d.q_avg, freq, wear))
    if h <= 0:
        raise ValueError("The pump gives no head at the average rate.")
    n = int(stages) if stages else int(np.ceil(d.head / h - 1e-9))
    eff = float(pump.efficiency(d.q_avg, freq))
    hyd = d.fluid_hp * n * h / d.head
    power = hyd / max(eff * (1.0 - wear), 1e-6)       # wear lowers the head, not the power
    hp_avail = motor.hp * freq / 60.0
    load = 100.0 * power * (1.0 + motor_margin) / hp_avail
    m = motor.at(load, freq)
    amps = m["amps_pct"] / 100.0 * motor.amps
    v_motor = motor.volts * freq / 60.0
    v_drop = cable.drop(amps, cable_len, d.t_cable)
    v_surf = v_motor + v_drop
    kva = np.sqrt(3.0) * v_surf * amps / 1000.0
    torque = 5252.0 * power / m["rpm"]
    # heat: pump and motor losses into the produced fluid
    cp = 0.0
    loss_hp = power * (1.0 - eff) + power * (1.0 / max(m["eff"], 1e-6) - 1.0)
    out = {"Number Of Stages": n, "Head per stage (ft)": h, "Power Required (hp)": power,
           "Pump Efficiency (%)": 100 * eff, "Current Used (amps)": amps, "Motor Efficiency (%)": 100 * m["eff"],
           "Motor Power Generated (hp)": power, "Motor Speed (rpm)": m["rpm"],
           "Voltage Drop Along Cable (Volts)": v_drop, "Voltage Required @ Surface (Volts)": v_surf,
           "Surface KVA": kva, "Torque On Shaft (lb.ft)": torque, "Motor Power Factor (fraction)": m["pf"],
           "Motor Load (%)": load, "Pump and motor losses (hp)": loss_hp}
    if casing_id and motor_od:
        a = np.pi / 4 * ((casing_id / 12) ** 2 - (motor_od / 12) ** 2)
        out["Motor Fluid Velocity (ft/s)"] = d.q_avg * BBL_FT3 / 86400.0 / a
    lo, hi = pump.range_at(freq)
    notes = []
    if not lo <= d.q_avg <= hi:
        notes.append(f"Average rate {d.q_avg:,.0f} RB/d is outside the pump's recommended range at {freq:g} Hz "
                     f"({lo:,.0f}-{hi:,.0f}).")
    if load > 100:
        notes.append(f"Motor overloaded ({load:.0f} % of nameplate power).")
    if amps > cable.max_amps:
        notes.append(f"Current {amps:.0f} A exceeds the cable rating ({cable.max_amps:.0f} A).")
    out["notes"] = notes
    return out


def outlet_temperature(t_intake, loss_hp, mass_lbd, cp_btu_lbf=0.85):
    """Fluid temperature after the pump and motor from their losses (BTU/h = 2544.43 x hp)."""
    return t_intake + loss_hp * 2544.43 / max(mass_lbd / 24.0 * cp_btu_lbf, 1e-9)


MOTOR_EFF_DEFAULT = 0.85      # motor efficiency when no motor is given (heat to the fluid only)
PUMP_T_FRACTION = 0.25        # fluid temperature in the stages: intake + this x the rise to the discharge
PUMP_DT_MAX = 300.0           # PROSPER caps the pump temperature rise (T45 lift curves: 300 F at a few STB/d)


def discharge_loss_hp(hyd_hp, pump_hp, motor=None, freq=60.0):
    """Motor and pump losses (hp) heating the produced fluid: motor input power less the hydraulic power."""
    pump_hp = np.asarray(pump_hp, float)
    if motor is not None:
        load = 100.0 * pump_hp / (motor.hp * np.asarray(freq, float) / 60.0)
        eff = np.interp(np.nan_to_num(load), motor.load, motor.eff) / 100.0
    else:
        eff = MOTOR_EFF_DEFAULT
    return np.maximum(pump_hp / np.maximum(eff, 0.05) - np.asarray(hyd_hp, float), 0.0)


def discharge_temperature(t_intake, hyd_hp, pump_hp, w_lbh, cp, motor=None, freq=60.0):
    """Pump discharge temperature (F, vectorised): all the motor and pump losses - motor input power less the
    hydraulic power - heat the produced fluid (w lb/h, cp BTU/lb/F). Against PROSPER's twelve Tutorial 26 runs the
    rise is 5-20 % below PROSPER's pump delta T (within 1 F)."""
    pump_hp = np.asarray(pump_hp, float)
    if motor is not None:
        load = 100.0 * pump_hp / (motor.hp * np.asarray(freq, float) / 60.0)
        eff = np.interp(np.nan_to_num(load), motor.load, motor.eff) / 100.0
    else:
        eff = MOTOR_EFF_DEFAULT
    loss = np.maximum(pump_hp / np.maximum(eff, 0.05) - np.asarray(hyd_hp, float), 0.0)
    dt = loss * 2544.43 / np.maximum(np.asarray(w_lbh, float) * np.asarray(cp, float), 1e-9)
    return np.asarray(t_intake, float) + np.minimum(dt, PUMP_DT_MAX)


# ================================================================ the pump in the well (system analysis)
def pump_dp(pump: Pump, n_stages, q_avg_rbd, sg, freq=60.0, wear=0.0):
    """Pressure gain (psi) of n stages at the average rate (RB/d) for a fluid of specific gravity sg."""
    h = np.maximum(pump.head_stage(q_avg_rbd, freq, wear), 0.0)
    return n_stages * h * sg * RHO_W / 144.0


def discharge_pressure(path, fluid, pwf, q, wc, gor, pump_md, pump: Pump, n_stages, freq=60.0, wear=0.0,
                       sep_eff=0.0, correlation="hagedorn_brown", p1=1.0, p2=1.0, u=8.0, hc=None, method="average",
                       motor=None):
    """PROSPER's ESP system node: the inflow corrected to the pump discharge, PDP = PIP + pump dP (psia), with the
    average rate and gravity through the pump at the average pressure (iterated). Returns (pdp, info)."""
    hc = hc or V.HeatCapacities()
    below = sub_path(path, pump_md, path.md[-1])
    case = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([gor])}
    Pb, Tb = V.traverse_up(below, fluid, np.array([pwf]), case, correlation, p1, p2, u, hc)
    pip, t_in = float(Pb[0, 0]), float(Tb[0, 0])
    if not np.isfinite(pip) or pip <= 14.7:
        return np.nan, {"PIP": pip}
    ql_in, qg_in, rs_in = _rates(fluid, pip, t_in, case)
    free = max(gor - rs_in, 0.0) if np.isfinite(rs_in) else 0.0
    case_a = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([gor - sep_eff * free])}
    w_a, _ = fluid.mass_cp(case_a, hc)
    mass = float(w_a[0]) * 24.0
    if method == "stages":
        # stage by stage: each stage at its own in-situ rate and density (gas compressed through the pump)
        p, hyd, pw, q_sum = pip, 0.0, 0.0, 0.0
        for _ in range(int(n_stages)):
            ql, qg, _ = _rates(fluid, p, t_in, case_a)
            q_loc = ql + qg
            rho = mass / (q_loc * BBL_FT3)
            h = max(float(pump.head_stage(q_loc, freq, wear)), 0.0)
            dp = h * rho / 144.0
            e = max(float(pump.efficiency(q_loc, freq)), 1e-6)
            hyd += q_loc * dp * DP_HP
            pw += q_loc * dp * DP_HP / (e * max(1.0 - wear, 0.05))   # wear lowers the head, not the power (PROSPER T34)
            q_sum += q_loc
            p += dp
        q_avg = q_sum / max(int(n_stages), 1)
        sg = mass / (q_avg * BBL_FT3) / RHO_W
        _, cp_a = fluid.mass_cp(case_a, hc)
        t_dis = float(discharge_temperature(t_in, hyd, pw, mass / 24.0, float(np.ravel(cp_a)[0]), motor, freq))
        return float(p), {"PIP": pip, "PDP": p, "Average rate (RB/d)": q_avg, "Gravity": sg,
                          "Intake rate (RB/d)": ql_in + qg_in * (1 - sep_eff), "Free GOR (scf/STB)": free,
                          "GOR above pump": gor - sep_eff * free, "Intake temperature (F)": t_in,
                          "Hydraulic power (hp)": hyd, "Pump power (hp)": pw, "Discharge temperature (F)": t_dis}
    pdp = pip + 500.0
    for _ in range(40):
        ql, qg, _ = _rates(fluid, 0.5 * (pip + pdp), t_in, case_a)
        q_avg = ql + qg
        sg = mass / (q_avg * BBL_FT3) / RHO_W
        new = pip + pump_dp(pump, n_stages, q_avg, sg, freq, wear)
        if abs(new - pdp) < 0.01:
            pdp = new
            break
        pdp = new
    hyd = q_avg * (pdp - pip) * DP_HP
    pw = hyd / max(float(pump.efficiency(q_avg, freq)), 1e-6)
    _, cp_a = fluid.mass_cp(case_a, hc)
    t_dis = float(discharge_temperature(t_in, hyd, pw, mass / 24.0, float(np.ravel(cp_a)[0]), motor, freq))
    return float(pdp), {"PIP": pip, "PDP": pdp, "Average rate (RB/d)": q_avg, "Gravity": sg,
                        "Intake rate (RB/d)": ql_in + qg_in * (1 - sep_eff), "Free GOR (scf/STB)": free,
                        "GOR above pump": gor - sep_eff * free, "Intake temperature (F)": t_in,
                        "Discharge temperature (F)": t_dis}


def system(path, fluid, ipr_pwf, wc, gor, top_p, pump_md, pump: Pump, n_stages, freq=60.0, wear=0.0, sep_eff=0.0,
           correlation="hagedorn_brown", p1=1.0, p2=1.0, u=8.0, hc=None, q_max=None, n=30, motor=None,
           method="average"):
    """PROSPER system analysis for an ESP well: the solution node is the pump discharge. VLP = top node down the
    tubing to the pump; 'PDP curve' = IPR corrected to the pump intake plus the pump pressure gain. The solution is
    their intersection (largest rate where the PDP curve falls through the VLP). ipr_pwf(q) -> BHP (psia), NaN past
    the AOF. Returns (solution dict or None, curve DataFrame)."""
    hc = hc or V.HeatCapacities()
    above = sub_path(path, path.md[0], pump_md)
    q_hi = q_max or 20000.0
    qs = np.linspace(q_hi * 0.02, q_hi, n)
    rows = []
    for q in qs:
        pwf = float(ipr_pwf(q))
        if not np.isfinite(pwf):
            break
        pdp, info = discharge_pressure(path, fluid, pwf, q, wc, gor, pump_md, pump, n_stages, freq, wear, sep_eff,
                                       correlation, p1, p2, u, hc, method, motor)
        case_a = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([info.get("GOR above pump", gor)])}
        P, _ = V.traverse(above, fluid, np.array([top_p]), case_a, correlation, p1, p2, u, hc,
                          info.get("Discharge temperature (F)"))
        rows.append({"Liquid rate (STB/d)": q, "VLP at pump discharge (psia)": float(P[0, -1]),
                     "Pump discharge from IPR (psia)": pdp, "Pump intake (psia)": info.get("PIP", np.nan),
                     "Average pump rate (RB/d)": info.get("Average rate (RB/d)", np.nan), "BHP (psia)": pwf})
    curve = pd.DataFrame(rows)
    if len(curve) < 2:
        return None, curve
    f = (curve["Pump discharge from IPR (psia)"] - curve["VLP at pump discharge (psia)"]).to_numpy()
    k = None
    for j in range(len(f) - 1, 0, -1):
        if np.isfinite(f[j - 1]) and np.isfinite(f[j]) and f[j - 1] > 0 >= f[j]:
            k = j
            break
    if k is None:
        return None, curve

    def g(q):
        pwf = float(ipr_pwf(q))
        pdp, info = discharge_pressure(path, fluid, pwf, q, wc, gor, pump_md, pump, n_stages, freq, wear, sep_eff,
                                       correlation, p1, p2, u, hc, method, motor)
        case_a = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([info.get("GOR above pump", gor)])}
        P, _ = V.traverse(above, fluid, np.array([top_p]), case_a, correlation, p1, p2, u, hc,
                          info.get("Discharge temperature (F)"))
        return pdp - float(P[0, -1])
    q_sol = brentq(g, qs[k - 1], qs[k], xtol=0.5)
    pwf = float(ipr_pwf(q_sol))
    pdp, info = discharge_pressure(path, fluid, pwf, q_sol, wc, gor, pump_md, pump, n_stages, freq, wear, sep_eff,
                                   correlation, p1, p2, u, hc, method, motor)
    q_avg, sg = info["Average rate (RB/d)"], info["Gravity"]
    h = float(pump.head_stage(q_avg, freq, wear))
    eff = float(pump.efficiency(q_avg, freq))
    hyd = q_avg * (pdp - info["PIP"]) * DP_HP
    if "Pump power (hp)" in info:
        eff = info["Hydraulic power (hp)"] / max(info["Pump power (hp)"], 1e-9)
    lo, hi = pump.range_at(freq)
    sol = {"Liquid rate (STB/d)": q_sol, "Oil rate (STB/d)": q_sol * (1 - wc), "BHP (psia)": pwf,
           "Pump intake pressure (psia)": info["PIP"], "Pump discharge pressure (psia)": pdp,
           "Pump intake rate (RB/d)": info["Intake rate (RB/d)"], "Average pump rate (RB/d)": q_avg,
           "Head (ft)": n_stages * h, "Pump efficiency (%)": 100 * eff,
           "Pump power (hp)": hyd / max(eff, 1e-6), "Free GOR at intake (scf/STB)": info["Free GOR (scf/STB)"],
           "In recommended range": bool(lo <= q_avg <= hi)}
    case_a = {"q": np.array([q_sol]), "wc": np.array([wc]), "gor": np.array([info.get("GOR above pump", gor)])}
    P, T = V.traverse(above, fluid, np.array([top_p]), case_a, correlation, p1, p2, u, hc,
                      info.get("Discharge temperature (F)"))
    sol["Wellhead temperature (F)"] = float(T[0, 0])
    sol["Pump intake temperature (F)"] = info["Intake temperature (F)"]
    sol["Pump discharge temperature (F)"] = info.get("Discharge temperature (F)", np.nan)
    sol["Average cable temperature (F)"] = 0.5 * (float(T[0, 0]) + info["Intake temperature (F)"])
    sol["Total fluid gravity"] = sg
    if motor is not None:
        sol["Motor load (%)"] = 100.0 * sol["Pump power (hp)"] / (motor.hp * freq / 60.0)
    return sol, curve


def operate(pump_hp, motor: Motor, cable: Cable, freq=60.0, cable_len=9000.0, t_cable=180.0):
    """Motor and cable at a pump power (PROSPER system results): load, current, efficiency, power factor, speed,
    motor input power, cable drop, surface voltage, kVA and torque. Motor power and voltage scale with frequency."""
    hp_avail = motor.hp * freq / 60.0
    load = 100.0 * pump_hp / hp_avail
    m = motor.at(load, freq)
    amps = m["amps_pct"] / 100.0 * motor.amps
    v_drop = cable.drop(amps, cable_len, t_cable)
    v_surf = motor.volts * freq / 60.0 + v_drop
    return {"Motor load (%)": load, "Motor amps (A)": amps, "Motor efficiency (%)": 100 * m["eff"],
            "Motor power factor": m["pf"], "Motor speed (rpm)": m["rpm"],
            "Motor power required (hp)": pump_hp / max(m["eff"], 1e-6), "Voltage drop along cable (V)": v_drop,
            "Voltage at surface (V)": v_surf, "Surface kVA": np.sqrt(3.0) * v_surf * amps / 1000.0,
            "Torque on shaft (lb.ft)": 5252.0 * pump_hp / m["rpm"]}


# ================================================================ vectorised pump (traverse, lift curves)
GROUP_MIDPOINT = True


def pump_gain(fluid, pip, t, case, pump: Pump, n_stages, freq=60.0, wear=0.0, groups=20):
    """Discharge pressure (psia) for intake pressures pip (array), stage by stage in groups of stages, each at its
    in-situ rate and density. case arrays broadcast with pip. Returns (pdp, hydraulic hp, pump hp, average RB/d)."""
    pip = np.atleast_1d(np.asarray(pip, float))
    hc = V.HeatCapacities()
    w, _ = fluid.mass_cp(case, hc)
    mass = np.broadcast_to(np.asarray(w, float) * 24.0, pip.shape)
    ng = int(max(1, min(groups, n_stages)))
    per = n_stages / ng
    p = pip.copy()
    hyd = np.zeros_like(p)
    pw = np.zeros_like(p)
    qs = np.zeros_like(p)
    tt = np.broadcast_to(np.asarray(t, float), p.shape)
    def group(pp):
        s = fluid.state(pp, tt, case)
        q_ = np.maximum((np.asarray(s["ql"], float) + np.asarray(s["qg"], float)) * 86400.0 / BBL_FT3, 1e-6)
        rho_ = mass / (q_ * BBL_FT3)
        return q_, rho_, np.maximum(pump.head_stage(q_, freq, wear), 0.0) * per * rho_ / 144.0

    for _ in range(ng):
        # midpoint (predictor-corrector) over the group: the fluid is compressed as it goes up the stages - the
        # inlet state alone under-predicts the lift of the first gassy stages (PROSPER T26 lift curves, 80 Hz)
        q0, r0, dp0 = group(p)
        q_loc, rho, dp = group(np.maximum(p + 0.5 * dp0, 14.7)) if GROUP_MIDPOINT else (q0, r0, dp0)
        h = dp * 144.0 / np.maximum(rho, 1e-9)
        hyd += q_loc * dp * DP_HP
        ps = pump.power_stage(q_loc, freq)
        if ps is not None:              # catalogue power x gravity (HSP: PROSPER's pump power)
            pw += ps * per * rho / 62.428
        else:
            e = np.maximum(pump.efficiency(q_loc, freq), 1e-6)
            pw += q_loc * dp * DP_HP / (e * max(1.0 - wear, 0.05))   # wear lowers the head, not the power (PROSPER T34)
        qs += q_loc
        p = p + dp
    return p, hyd, pw, qs / ng


def intake_from_discharge(fluid, pdp, t, case, pump: Pump, n_stages, freq=60.0, wear=0.0, groups=20, iters=40):
    """Intake pressure that the pump raises to pdp (vectorised bisection); NaN where even a dead intake (14.7 psia)
    gives more than pdp is impossible (pump too strong is fine) or pdp is not finite."""
    pdp = np.atleast_1d(np.asarray(pdp, float))
    lo = np.full_like(pdp, 14.7)
    hi = np.where(np.isfinite(pdp), pdp, 14.7)
    g_lo = pump_gain(fluid, lo, t, case, pump, n_stages, freq, wear, groups)[0] - pdp
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g = pump_gain(fluid, mid, t, case, pump, n_stages, freq, wear, groups)[0] - pdp
        up = g < 0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    pip = 0.5 * (lo + hi)
    # the pump lifts more than the tubing needs even from atmospheric intake: the well would pump off
    return np.where((g_lo >= 0) | ~np.isfinite(pdp), np.nan, pip)


def bhp_with_pump(path, fluid, top_p, case, esp: dict, correlation="hagedorn_brown", p1=1.0, p2=1.0, u=8.0,
                  hc=None, groups=20, freq=None):
    """Bottom-node pressure of an ESP well from the top-node (wellhead) pressure, vectorised over cases: tubing down
    to the pump discharge, intake pressure from the pump curve (stage groups), casing down to the bottom node. The
    tubing starts at the pump discharge temperature (intake temperature plus the motor and pump losses). The gas
    separator removes sep x free gas at the intake from the tubing stream. Both are iterated with the intake
    pressure. freq: frequency per case (array, e.g. the lift-curve ALQ axis); default esp["freq"].
    Returns a dict of arrays: bhp, pip, pdp, t_intake, t_discharge, gor_above, pump_hp, hyd_hp (NaN where the pump
    would draw the intake below atmospheric)."""
    hc = hc or V.HeatCapacities()
    shape = np.shape(case["q"])
    top_p = np.broadcast_to(np.asarray(top_p, float), shape).copy()
    f = np.broadcast_to(np.asarray(esp["freq"] if freq is None else freq, float), shape).copy()
    above = sub_path(path, path.md[0], esp["pump_md"])
    below = sub_path(path, esp["pump_md"], path.md[-1])
    w, cp = fluid.mass_cp(case, hc)
    t_in = V.temperature_profile(below, np.broadcast_to(np.asarray(w, float), shape), cp, u)[:, 0]
    sep = float(esp.get("sep", 0.0))
    gor = np.asarray(case["gor"], float)
    case_a = dict(case)
    t_dis = t_in.copy()
    t_pump = t_in.copy()
    pump, n_st, wear = esp["pump"], esp["stages"], esp.get("wear", 0.0)
    for _ in range(3):
        with np.errstate(all="ignore"):
            P, _ = V.traverse(above, fluid, top_p, case_a, correlation, p1, p2, u, hc, t_dis)
            pdp = P[:, -1]
            pip = intake_from_discharge(fluid, pdp, t_pump, case_a, pump, n_st, f, wear, groups)
            ok = np.isfinite(pip)
            _, hyd, pw, _ = pump_gain(fluid, np.where(ok, pip, 1000.0), t_pump, case_a, pump, n_st, f, wear, groups)
        if sep > 0 and hasattr(fluid, "_props"):
            rs = np.asarray(fluid._props(np.where(ok, pip, 1000.0), t_in, gor)[0], float)
            case_a = {**case, "gor": gor - sep * np.maximum(gor - rs, 0.0)}
        w_a, cp_a = fluid.mass_cp(case_a, hc)
        t_new = np.where(ok, discharge_temperature(t_in, hyd, pw, w_a, cp_a, esp.get("motor"), f), t_in)
        # the fluid in the stages is warmer than at the intake: PROSPER's T45 lift curves (water, all gas separated) at a
        # few STB/d (temperature rise up to 300 F) put it a quarter of the way to the discharge
        # only for a pump handling liquid alone: with free gas through the stages PROSPER's T26 lift curves (80 Hz,
        # GOR 800) keep the intake temperature
        rs_in = np.asarray(fluid._props(np.where(ok, pip, 1000.0), t_in, np.asarray(case_a["gor"], float))[0], float) \
            if hasattr(fluid, "_props") else np.zeros_like(t_in)
        liquid = np.asarray(case_a["gor"], float) <= rs_in + 1e-6
        t_pump = np.where(ok & liquid, t_in + PUMP_T_FRACTION * (t_new - t_in), t_in)
        if np.nanmax(np.abs(t_new - t_dis)) < 0.2 and not sep:
            t_dis = t_new
            break
        t_dis = t_new
    with np.errstate(all="ignore"):
        Pb, _ = V.traverse(below, fluid, np.where(ok, pip, 1000.0), case, correlation, p1, p2, u, hc)
    nan = lambda a: np.where(ok, a, np.nan)
    return {"bhp": nan(Pb[:, -1]), "pip": pip, "pdp": pdp, "t_intake": t_in, "t_discharge": t_dis, "t_pump": t_pump,
            "gor_above": np.asarray(case_a["gor"], float) * np.ones_like(pip), "pump_hp": nan(pw), "hyd_hp": nan(hyd)}


def liquid_level(fluid, pip, t_intake, case, pump_tvd, chp, top_tvd=0.0, gas_column=False, t_surface=80.0):
    """Annulus liquid level (ft TVD) for a casing head pressure chp (psia), no gas vented: the produced liquid at
    intake conditions stands from the level down to the pump, PIP = CHP (+ gas column) + G_l·(pump - level).
    PROSPER ignores the gas column (its reported levels match this to <1.5% with gas_column=False)."""
    pip = float(pip)
    s = fluid.state(np.array([pip]), np.array([float(t_intake)]), case)
    g_l = float(np.asarray(s["rho_l"]).ravel()[0]) / 144.0
    lvl = pump_tvd - (pip - chp) / g_l
    if gas_column:
        from . import blackoil as bo
        inp = getattr(fluid, "inp", None)
        for _ in range(50):
            p_avg = chp + 0.5 * (pip - chp) * (lvl - top_tvd) / max(pump_tvd - top_tvd, 1.0)
            g_g = float(np.asarray(bo.gas_props(inp, np.array([p_avg]), np.array([0.5 * (t_surface + t_intake)]))[2]).ravel()[0]) / 144.0
            new = (g_l * pump_tvd - (pip - chp) - g_g * top_tvd) / (g_l - g_g)
            if abs(new - lvl) < 0.01:
                lvl = new
                break
            lvl = new
    return float(np.clip(lvl, top_tvd, pump_tvd))
