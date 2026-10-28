"""Progressive cavity pump (PCP): equipment database, pump duty, design, system analysis, traverse and lift curves.

After PROSPER Design | Progressive Cavity Pump and System | Progressive Cavity Pump (User Guide 2.8.4 and 2.13.4,
Tutorial 29). A PCP is a positive-displacement pump: its speed sets the rate, the head it develops sets the slip.

Pump model (PROSPER's database layout: head, flow and shaft power at a reference speed, with a head gradient, a head
intercept and a power gradient per head row). Checked against the user's PROSPER runs (Tutorial 29 file, 20 system
sensitivities on water cut and pump speed plus the base system):
  - rate through the pump (average downhole rate, RB/d) = head gradient x speed + intercept(H), the intercept
    interpolated linearly in the head H the pump develops (ft of the fluid in the pump: dP / average fluid gradient);
    the intercept at zero head is the reference rate less head gradient x reference speed - within 0.05 % of PROSPER
    at 15 of the 21 runs, 0.3-1.3 % at the six whose head lies within 10 ft above a table row;
  - shaft power (hp) = power gradient(H_w) x speed, the power gradient interpolated in the head of fresh water
    H_w = dP / 0.433 psi/ft - within 0.3 %; pump efficiency = average rate x dP / shaft power - within 0.2 %.
Downhole motor drive: the motor turns at the pump speed; its nameplate power and voltage scale with speed as with an
ESP at the frequency speed / 60 (a 60 Hz motor at 3600 rpm synchronous): load = shaft power / (hp x N / (60 f_m)).
Motors with coefficient tables use PROSPER's fifth-order polynomials in the load fraction for amps, speed, efficiency
and power factor; motors without them use PROSPER's default curves - tabulated from its reported values (load 88-173 %,
and the ESP template shape below 80 % load); above 175 % load PROSPER's default curves give unphysical values (motor
efficiency above 100 %) and are held at the 173 % point here. Cable voltage drop = coefficient x length x current x
(1 + 0.0020704 (T - 84.46 F)) - PROSPER's within 0.02 V.
Wear: the pump's head capacity is derated, rate = gradient x speed + intercept(H / (1 - wear)) (PROSPER's system
analysis; its design screen ignores the wear factor).
Sucker rod drive: the produced fluid flows up the rod-tubing annulus (hydraulic diameter tubing ID - rod OD, as
PROSPER's traverse); rod weight, buoyed weight, torque and stresses are our own estimates - PROSPER's rod outputs are not
reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import vlp as V
from .esp import BBL_FT3, DP_HP, Duty, pump_duty, sub_path, _rates

DATA = Path(__file__).parent / "data"
G_W = 0.433                     # psi/ft - fresh water, PROSPER's head for the power gradient
STEEL_DENSITY = 490.0           # lb/ft3


# ================================================================ pumps
@dataclass
class PCPPump:
    manufacturer: str
    series: str
    model: str
    size: float                   # pump OD, in
    max_head: float               # ft
    ref_speed: float              # rpm
    ref_rate: float               # RB/d at the reference speed, zero head
    head_grad: float              # RB/d per rpm
    heads: tuple                  # ft
    intercepts: tuple             # RB/d (rate = head_grad x N + intercept)
    power_grads: tuple            # hp per rpm
    flows: tuple = ()             # table flow at the reference speed, RB/d
    powers: tuple = ()            # table shaft power at the reference speed, hp
    length: float = np.nan        # ft
    rotor_od: float = np.nan      # in

    @property
    def name(self):
        return f"{self.manufacturer} {self.series} {self.model} (Max Head {self.max_head:g} feet)"

    def _h(self):
        h0 = self.ref_rate - self.head_grad * self.ref_speed
        return np.r_[0.0, self.heads], np.r_[h0, self.intercepts]

    def intercept(self, head):
        """Rate intercept at head H (ft), linear in the table; linear beyond the last row."""
        hs, ic = self._h()
        h = np.asarray(head, float)
        out = np.interp(h, hs, ic)
        s_hi = (ic[-1] - ic[-2]) / (hs[-1] - hs[-2])
        return np.where(h > hs[-1], ic[-1] + s_hi * (h - hs[-1]), out)

    def rate(self, speed, head, wear=0.0):
        """Average rate through the pump (RB/d) at speed (rpm) and head (ft); zero when the pump slips it all. Wear
        derates the pump's head capacity as PROSPER: the slip is that of the head H / (1 - wear) (the user's T29
        system run at wear 0.2: 475.75 against PROSPER's 475.7 RB/d)."""
        q = self.head_grad * np.asarray(speed, float) + self.intercept(np.asarray(head, float) / (1.0 - wear))
        return np.maximum(q, 0.0)

    def speed_for(self, rate, head, wear=0.0):
        """Speed (rpm) that delivers rate (RB/d) against head (ft)."""
        return (np.asarray(rate, float) - self.intercept(np.asarray(head, float) / (1.0 - wear))) / self.head_grad

    def head_for(self, speed, rate, wear=0.0):
        """Head (ft) at which the pump at speed delivers rate (inverse of rate(); NaN if even zero head is short)."""
        hs, ic = self._h()
        target = np.asarray(rate, float) - self.head_grad * np.asarray(speed, float)
        # intercept falls with head; extend the last segment
        s_hi = (ic[-1] - ic[-2]) / (hs[-1] - hs[-2])
        h_ext = hs[-1] + 5 * hs[-1]
        hs2, ic2 = np.r_[hs, h_ext], np.r_[ic, ic[-1] + s_hi * (h_ext - hs[-1])]
        h = np.interp(-target, -ic2, hs2) * (1.0 - wear)
        return np.where(target > ic2[0], np.nan, h)

    def power_grad(self, head_water):
        hs = np.asarray(self.heads, float)
        pg = np.asarray(self.power_grads, float)
        h = np.asarray(head_water, float)
        lo = pg[0] + (pg[1] - pg[0]) / (hs[1] - hs[0]) * (h - hs[0])
        hi = pg[-1] + (pg[-1] - pg[-2]) / (hs[-1] - hs[-2]) * (h - hs[-1])
        return np.where(h < hs[0], lo, np.where(h > hs[-1], hi, np.interp(h, hs, pg)))

    def shaft_power(self, speed, dp_psi):
        """Shaft power (hp) at speed (rpm) against a pressure rise dP (psi)."""
        return np.maximum(self.power_grad(np.asarray(dp_psi, float) / G_W), 0.0) * np.asarray(speed, float)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


@lru_cache(maxsize=1)
def _load_pumps():
    df = pd.read_csv(DATA / "pcp_pumps.csv")
    out = {}
    for (man, ser, mod), g in df.groupby(["manufacturer", "series", "model"], sort=False):
        r = g.iloc[0]
        p = PCPPump(man, ser, str(mod), float(r["size"]), float(r["max_head"]), float(r["ref_speed"]),
                    float(r["ref_rate"]), float(r["head_grad"]), tuple(g["head"]), tuple(g["intercept"]),
                    tuple(g["power_grad"]), tuple(g["flow"]), tuple(g["power"]), _f(r["length"]),
                    _f(r["rotor_od"]))
        out[p.name] = p
    return out


PUMPS = _load_pumps()


def pump_from_table(manufacturer, series, model, size, max_head, ref_speed, heads, flows, powers):
    """A user pump from a reference-speed table of head (ft), flow (RB/d) and shaft power (hp): the head gradient is
    taken from the zero-head reference rate (ref rate / ref speed), the intercept = flow - gradient x ref speed and the
    power gradient = power / ref speed (PROSPER's derived columns)."""
    heads, flows, powers = (np.asarray(x, float) for x in (heads, flows, powers))
    o = np.argsort(heads)
    heads, flows, powers = heads[o], flows[o], powers[o]
    # zero-head rate: extrapolate the first two rows when not given
    q0 = flows[0] + (flows[0] - flows[1]) / (heads[1] - heads[0]) * heads[0] if len(heads) > 1 else flows[0]
    hg = q0 / ref_speed
    return PCPPump(manufacturer, series, model, float(size), float(max_head), float(ref_speed), float(q0), hg,
                   tuple(heads), tuple(flows - hg * ref_speed), tuple(powers / ref_speed), tuple(flows),
                   tuple(powers))


# ================================================================ motors and cables
# PROSPER's default motor curves (motors without coefficient tables), from its reported values in the user's
# Tutorial 29 runs (load fraction: amps / nameplate, efficiency, power factor, speed at 60 Hz rpm); below 80 % load
# the shape of the ESP motor template (esp.MOTORS, PROSPER's motor plot).
_ESP_TEMPLATE_SERIES = {"544"}   # PROSPER's default curves differ by motor series (544: the ESP template)
_DEF_LOAD = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.8836, 1.2525, 1.2767, 1.626, 1.7027, 1.7331)
_DEF_AMPS = (0.565, 0.531, 0.522, 0.537, 0.568, 0.617, 0.679, 0.744, 0.814, 0.939056, 1.2160, 1.2380, 1.5434,
             1.5820, 1.5900)
_DEF_EFF = (0.581, 0.642, 0.683, 0.724, 0.755, 0.783, 0.804, 0.823, 0.833, 0.826263, 0.8311, 0.8295, 0.8406,
            0.8789, 0.9030)
_DEF_PF = (0.453, 0.505, 0.556, 0.605, 0.650, 0.6885, 0.723, 0.754, 0.779, 0.805, 0.8300, 0.8346, 0.8356,
           0.7935, 0.7677)
_DEF_RPM = (3483, 3582, 3647, 3656, 3649, 3645, 3631, 3614, 3599, 3475.6, 3408.1, 3402.4, 3305.3, 3299.8, 3294.0)


@dataclass
class PCPMotor:
    manufacturer: str
    series: str
    od: float
    rpm: float                    # nameplate speed at the nameplate frequency
    hz: float
    hp: float
    volts: float
    amps: float
    kind: str = "S"
    c_amps: tuple = ()
    c_speed: tuple = ()
    c_eff: tuple = ()
    c_pf: tuple = ()

    @property
    def name(self):
        return f"{self.manufacturer} {self.series} {self.hp:g}HP {self.volts:g}V {self.amps:g}A"

    def scale(self, speed):
        """Speed / synchronous speed at the nameplate frequency (PROSPER: the motor runs at the pump speed)."""
        return np.asarray(speed, float) / (60.0 * self.hz)

    def at(self, load_pct, speed):
        x = np.asarray(load_pct, float) / 100.0
        if len(self.c_amps) == 6:
            amps, eff = np.polyval(self.c_amps, x), np.polyval(self.c_eff, x)
            pf, rpm = np.polyval(self.c_pf, x), np.polyval(self.c_speed, x)
        else:
            if self.series in _ESP_TEMPLATE_SERIES:
                # PROSPER's 544-series curves: the ESP template (esp.MOTORS, digitised from PROSPER's 544 300 hp motor
                # plot) - reproduces its PCP design screens with 544 300 and 375 hp motors within 0.3 %
                from .esp import MOTORS as _EM
                t = _EM["Centrilift 544 300HP 1675V 105A"]
                lp = np.minimum(100.0 * x, t.load[-1])
                amps, eff = np.interp(lp, t.load, t.amps_pct) / 100.0, np.interp(lp, t.load, t.eff) / 100.0
                pf, rpm = np.interp(lp, t.load, t.pf) / 100.0, np.interp(lp, t.load, t.rpm60) * self.hz / 60.0
            else:
                xc = np.minimum(x, _DEF_LOAD[-1])
                amps, eff = np.interp(xc, _DEF_LOAD, _DEF_AMPS), np.interp(xc, _DEF_LOAD, _DEF_EFF)
                pf, rpm = np.interp(xc, _DEF_LOAD, _DEF_PF), np.interp(xc, _DEF_LOAD, _DEF_RPM) * self.hz / 60.0
        return {"amps_frac": amps, "eff": eff, "pf": pf, "rpm": rpm * self.scale(speed)}


def _coef(s):
    if not isinstance(s, str) or not s.strip():
        return ()
    return tuple(float(v) for v in s.split())


@lru_cache(maxsize=1)
def _load_motors():
    df = pd.read_csv(DATA / "motors.csv")
    out = {}
    for _, r in df.iterrows():
        m = PCPMotor(r["manufacturer"], str(r["series"]), float(r["od"]), float(r["rpm"]), float(r["hz"]),
                     float(r["hp"]), float(r["volts"]), float(r["amps"]), str(r["type"]), _coef(r["c_amps"]),
                     _coef(r["c_speed"]), _coef(r["c_eff"]), _coef(r["c_pf"]))
        out.setdefault(m.name, m)
    return out


MOTORS = _load_motors()


@dataclass
class PCPCable:
    name: str
    v_per_kft_amp: float          # V per 1000 ft per amp (PROSPER @ 20 C column)
    max_amps: float
    alpha_f: float = 0.0020704    # PROSPER's T29 runs: drop x (1 + 0.0020704 (T - 84.46 F)) within 0.02 V
    t_ref: float = 84.46

    def drop(self, amps, length_ft, t_avg_f):
        return self.v_per_kft_amp * length_ft / 1000.0 * amps * (1.0 + self.alpha_f * (t_avg_f - self.t_ref))


@lru_cache(maxsize=1)
def _load_cables():
    df = pd.read_csv(DATA / "cables.csv")
    return {r["cable"]: PCPCable(r["cable"], float(r["v_per_kft_amp"]), float(r["max_amps"])) for _, r in df.iterrows()}


CABLES = _load_cables()


@dataclass
class PCPRod:
    manufacturer: str
    name: str
    id: float                     # in
    od: float                     # in
    weight: float                 # lb/ft
    e: float = 30.4579e6          # psi
    alpha: float = 7.7778e-6      # 1/F

    @property
    def label(self):
        return f"{self.manufacturer} {self.name} {self.od:g} in"

    @property
    def area(self):
        return np.pi / 4.0 * (self.od ** 2 - self.id ** 2)


@lru_cache(maxsize=1)
def _load_rods():
    df = pd.read_csv(DATA / "pcp_rods.csv")
    out = {}
    for _, r in df.iterrows():
        rod = PCPRod(r["manufacturer"], r["name"], float(r["id"]), float(r["od"]), float(r["w"]), float(r["e"]),
                     float(r["alpha"]))
        out.setdefault(rod.label, rod)
    return out


RODS = _load_rods()


# ================================================================ design
DRIVES = ("Downhole motor", "Sucker rod")
DESIGN_HP_K = 133_885.0         # PROSPER's design power = displacement (head gradient x speed, RB/d) x head (ft) / K
HYD_HP_K = 136_000.0            # and its hydraulic power = average rate x head / K: efficiency = the ratio (fitted to
#                                 three PROSPER design screens, 750 and 850 STB/d, to 6 significant figures)


def design_power(pump: PCPPump, speed, head):
    """PROSPER's PCP design-screen power (hp) and efficiency (fraction) at a speed (rpm) and head (ft): power =
    displacement x head / 133,885; efficiency = (average rate x head / 136,000) / power = 0.98445 x volumetric
    efficiency. (Its system analysis uses the power gradients instead - see shaft_power.)"""
    pw = pump.head_grad * np.asarray(speed, float) * np.asarray(head, float) / DESIGN_HP_K
    return pw


def duty(path, fluid, pwf, q, wc, gor, top_p, pump_md, sep_eff=0.0, correlation="hagedorn_brown", p1=1.0, p2=1.0,
         u=8.0, hc=None, rod: PCPRod | None = None) -> Duty:
    """PROSPER 'Calculate Pump Duty' for a PCP (same steps as the ESP: intake from the IPR up the casing, discharge from
    the top node down the tubing - the rod-tubing annulus with a sucker rod drive)."""
    p = with_rod(path, rod, pump_md) if rod is not None else path
    return pump_duty(p, fluid, pwf, q, wc, gor, top_p, pump_md, sep_eff, correlation, p1, p2, u, hc)


def duty_table(d: Duty, g=14.696):
    rows = [("Flowing Bottomhole Pressure", d.pwf - g, "psig"), ("Pump Intake Pressure", d.pip - g, "psig"),
            ("Pump Intake Rate", d.q_in, "RB/day"), ("Free GOR Entering Pump", d.free_gor, "scf/STB"),
            ("Pump Discharge Pressure", d.pdp - g, "psig"), ("Pump Discharge Rate", d.q_out, "RB/day"),
            ("Total GOR Above Pump", d.gor_above, "scf/STB"), ("Mass Flow Rate", d.mass, "lbm/day"),
            ("Total Fluid Gravity", d.sg, "sp. gravity"), ("Average Downhole Rate", d.q_avg, "RB/day"),
            ("Head Required", d.head, "feet"), ("Pump dP Required", d.pdp - d.pip, "psi"),
            ("Fluid Power Required", d.fluid_hp, "hp"), ("GLR At Pump Intake (V/V)", d.glr_intake, "fraction"),
            ("Inlet Temperature", d.t_intake, "deg F"), ("Average Cable Temperature", d.t_cable, "deg F")]
    return pd.DataFrame(rows, columns=["Result", "Value", "Units"])


def candidate_pumps(d: Duty, pumps: dict | None = None, max_od=99.0):
    """Pumps that fit (OD) and can develop the head required, with the speed and efficiency at the design (PROSPER
    lists pumps by OD; the user checks the maximum head)."""
    rows = []
    for name, p in (pumps or PUMPS).items():
        n = float(p.speed_for(d.q_avg, d.head))
        pw = float(design_power(p, n, d.head))
        rows.append({"Pump": name, "OD (in)": p.size, "Maximum head (ft)": p.max_head, "Speed (rpm)": n,
                     "Power (hp)": pw, "Efficiency (%)": 100.0 * d.q_avg * d.head / HYD_HP_K / pw if pw > 0 else np.nan,
                     "Suitable": bool(p.size <= max_od and p.max_head >= d.head and n > 0)})
    df = pd.DataFrame(rows)
    return df.sort_values(["Suitable", "Efficiency (%)"], ascending=[False, False]).reset_index(drop=True)


def motor_results(power_hp, speed, motor: PCPMotor, cable: PCPCable | None, cable_len, t_cable, margin=0.0):
    """Motor and cable at a shaft power (hp) and pump speed (rpm) - PROSPER's PCP motor results."""
    avail = motor.hp * motor.scale(speed)
    load = 100.0 * power_hp * (1.0 + margin) / avail
    m = motor.at(load, speed)
    amps = float(m["amps_frac"]) * motor.amps
    v_motor = motor.volts * float(motor.scale(speed))
    out = {"Motor load (%)": float(load), "Current used (amps)": amps, "Motor efficiency (%)": 100 * float(m["eff"]),
           "Motor power factor": float(m["pf"]), "Motor speed (rpm)": float(m["rpm"]),
           "Motor power generated (hp)": float(power_hp),
           "Motor power required (hp)": float(power_hp) / max(float(m["eff"]), 1e-6),
           "Torque on shaft (lb.ft)": 5252.0 * float(power_hp) / max(float(m["rpm"]), 1e-9),
           "Motor voltage (V)": v_motor}
    if cable is not None:
        vd = cable.drop(amps, cable_len, t_cable)
        out.update({"Cable voltage drop (V)": vd, "Required surface voltage (V)": v_motor + vd,
                    "Surface KVA": np.sqrt(3.0) * (v_motor + vd) * amps / 1000.0,
                    "Cable overloaded": bool(amps > cable.max_amps)})
    notes = []
    if load > 100:
        notes.append(f"Motor overloaded ({load:.0f} % of its power at {speed:,.0f} rpm).")
    if load > 100 * _DEF_LOAD[-1] and not motor.c_amps:
        notes.append("Above 173 % load PROSPER's default motor curves become unphysical; the 173 % values are used.")
    out["notes"] = notes
    return out


def smallest_cable(amps, cables: dict | None = None):
    ok = [c for c in (cables or CABLES).values() if c.max_amps >= amps]
    return min(ok, key=lambda c: (c.max_amps, c.v_per_kft_amp)) if ok else None


def rod_results(rod: PCPRod, length_ft, torque_lbft, rho_fluid, dp_psi, pump: PCPPump | None = None):
    """Sucker rod drive: rod weight, buoyed weight, axial load (buoyed weight + pump thrust on the rotor), stretch,
    axial, torsional and von Mises stresses. Pump thrust = dP x rotor area (rotor OD from the pump database, else 55 %
    of the pump OD). NOT validated against PROSPER."""
    w_air = rod.weight * length_ft
    w_b = w_air * (1.0 - rho_fluid / STEEL_DENSITY)
    d_rot = pump.rotor_od if pump is not None and np.isfinite(pump.rotor_od) and pump.rotor_od < pump.size else (
        0.55 * pump.size if pump is not None else rod.od)
    thrust = dp_psi * np.pi / 4.0 * d_rot ** 2
    load = w_b + thrust
    sa = load / rod.area
    j = np.pi / 32.0 * (rod.od ** 4 - rod.id ** 4)
    tau = torque_lbft * 12.0 * (rod.od / 2.0) / j
    return {"Rod weight in air (lb)": w_air, "Rod weight in fluid (lb)": w_b, "Pump thrust (lb)": thrust,
            "Axial load at the top (lb)": load, "Rod stretch (in)": load * length_ft * 12.0 / (rod.area * rod.e),
            "Axial stress (psi)": sa, "Torsional shear stress (psi)": tau,
            "Combined (von Mises) stress (psi)": float(np.sqrt(sa ** 2 + 3.0 * tau ** 2)),
            "Polished rod torque (lb.ft)": torque_lbft}


def design(d: Duty, pump: PCPPump, drive="Downhole motor", motor: PCPMotor | None = None,
           cable: PCPCable | None = None, cable_len=0.0, wear=0.0, margin=0.0, rod: PCPRod | None = None,
           rod_len=0.0, cp=None, mass_lbd=None):
    """PROSPER PCP design results: speed for the average downhole rate at the head required, power, efficiency, pump
    outlet temperature, then the motor and cable (downhole motor drive) or the rod (sucker rod drive)."""
    # PROSPER's design screen ignores the wear factor (speed and power unchanged at 0.1 and 0.3 - the user's runs);
    # the system analysis, traverse and lift curves apply it
    n = float(pump.speed_for(d.q_avg, d.head))
    dp = d.pdp - d.pip
    pw = float(design_power(pump, n, d.head))
    hyd = d.q_avg * d.head / HYD_HP_K
    eff = hyd / pw if pw > 0 else np.nan
    out = {"Pump speed (rpm)": n, "Power required (hp)": pw, "Pump efficiency (%)": 100.0 * eff,
           "Power at this duty in the system analysis (hp)": float(pump.shaft_power(n, dp))}
    if cp is not None and mass_lbd:
        out["Pump outlet temperature (F)"] = d.t_intake + max(pw - hyd, 0.0) * 2544.43 / (mass_lbd / 24.0 * cp)
    notes = []
    if dp <= 0:
        notes.append("Discharge pressure less than intake pressure: the well flows at this rate without the pump "
                     "(PROSPER reports a negative head and meaningless power) - raise the design rate.")
    if d.head > pump.max_head:
        notes.append(f"Head required {d.head:,.0f} ft exceeds the pump's maximum {pump.max_head:,.0f} ft.")
    if n <= 0:
        notes.append("The pump cannot deliver the rate.")
    if drive == "Downhole motor" and motor is not None:
        mr = motor_results(pw, n, motor, None, cable_len, d.t_cable, margin)
        amps = mr["Current used (amps)"]
        cab = cable or smallest_cable(amps)
        mr = motor_results(pw, n, motor, cab, cable_len, d.t_cable, margin)
        notes += mr.pop("notes")
        out.update(mr)
        out["Cable"] = cab.name if cab else "none carries the current"
    elif drive == "Sucker rod" and rod is not None:
        rho = d.sg * 62.4
        out.update(rod_results(rod, rod_len, 5252.0 * pw / max(n, 1e-9), rho, dp, pump))
    out["notes"] = notes
    return out


# ================================================================ the installed pump
@dataclass
class PCPInstalled:
    pump: PCPPump
    pump_md: float
    speed: float                  # rpm
    wear: float = 0.0
    sep_eff: float = 0.0
    drive: str = "Downhole motor"
    motor: PCPMotor | None = None
    cable: PCPCable | None = None
    cable_len: float = 0.0
    rod: PCPRod | None = None
    rod_len: float = 0.0
    casing_id: float = np.nan     # in, for the motor fluid velocity

    @property
    def label(self):
        return f"{self.pump.name} at {self.speed:,.1f} rpm"


def with_rod(path, rod: PCPRod | None, pump_md):
    """Flow path with the rod in the tubing above the pump (annular flow: area and hydraulic diameter reduced)."""
    if rod is None:
        return path
    from .gaslift import with_nodes
    p = with_nodes(path, [pump_md])
    above = p.md[1:] <= pump_md + 1e-6
    d_rod = rod.od / 12.0
    area = p.area.copy()
    d_h = p.d_h.copy()
    d_in = np.sqrt(4.0 * area / np.pi)
    area = np.where(above, np.pi / 4.0 * (d_in ** 2 - d_rod ** 2), area)
    d_h = np.where(above, d_h - d_rod, d_h)
    return V.FlowPath(p.md, p.tvd, p.t_form, p.seg_len, p.sin_t, area, d_h, p.rough, p.d_ht, dict(p.restrictions))


def _cp_mix(fluid, case, hc):
    w, cp = fluid.mass_cp(case, hc)
    return float(np.ravel(w)[0]) * 24.0, float(np.ravel(cp)[0])


def operating_point(path, fluid, pwf, q, wc, gor, top_p, pcp: PCPInstalled, correlation="hagedorn_brown", p1=1.0,
                    p2=1.0, u=8.0, hc=None):
    """At liquid rate q with the bottom-node pressure pwf (IPR): the duty (PIP, PDP, average rate, head) and the pump's
    own rate at that head - mismatch = pump rate - average rate (RB/d)."""
    hc = hc or V.HeatCapacities()
    d = duty(path, fluid, pwf, q, wc, gor, top_p, pcp.pump_md, pcp.sep_eff, correlation, p1, p2, u, hc, pcp.rod)
    q_pump = float(pcp.pump.rate(pcp.speed, d.head, pcp.wear))
    return d, q_pump - d.q_avg


def results(d: Duty, pcp: PCPInstalled, fluid=None, hc=None, case=None):
    """PROSPER's PCP system results at a solved operating point."""
    dp = d.pdp - d.pip
    pw = float(pcp.pump.shaft_power(pcp.speed, dp))
    out = {"Pump intake pressure (psia)": d.pip, "Pump discharge pressure (psia)": d.pdp,
           "Pump intake rate (RB/d)": d.q_in, "Average rate through pump (RB/d)": d.q_avg,
           "Pump discharge rate (RB/d)": d.q_out, "Pump head generated (ft)": d.head,
           "Pump power requirement (hp)": pw, "Pump efficiency (%)": 100.0 * d.q_avg * dp * DP_HP / pw if pw else np.nan,
           "Free GOR at intake (scf/STB)": d.free_gor, "GOR above pump (scf/STB)": d.gor_above,
           "Pump intake temperature (F)": d.t_intake, "Total fluid gravity": d.sg}
    mass, cp = (d.mass, None)
    if fluid is not None and case is not None:
        mass, cp = _cp_mix(fluid, case, hc or V.HeatCapacities())
    notes = []
    if pcp.drive == "Downhole motor" and pcp.motor is not None:
        mr = motor_results(pw, pcp.speed, pcp.motor, pcp.cable, pcp.cable_len, d.t_cable)
        notes += mr.pop("notes")
        out.update(mr)
        if cp:
            w_h = mass / 24.0 * cp
            out["Motor delta T (F)"] = max(mr["Motor power required (hp)"] - pw, 0.0) * 2544.43 / w_h
        if np.isfinite(pcp.casing_id):
            a = np.pi / 4.0 * ((pcp.casing_id / 12.0) ** 2 - (pcp.motor.od / 12.0) ** 2)
            out["Motor fluid velocity (ft/s)"] = d.q_in * BBL_FT3 / 86400.0 / a
    elif pcp.drive == "Sucker rod" and pcp.rod is not None:
        out.update(rod_results(pcp.rod, pcp.rod_len, 5252.0 * pw / max(pcp.speed, 1e-9), d.sg * 62.4, dp, pcp.pump))
    if cp:
        dt = max(pw - d.q_avg * dp * DP_HP, 0.0) * 2544.43 / (mass / 24.0 * cp)
        out["Pump delta T (F)"] = dt
        out["Pump discharge temperature (F)"] = d.t_intake + dt
    if d.head > pcp.pump.max_head:
        notes.append(f"The pump works against {d.head:,.0f} ft, above its maximum head ({pcp.pump.max_head:,.0f} ft).")
    out["notes"] = notes
    return out


def system(path, fluid, ipr_pwf, wc, gor, top_p, pcp: PCPInstalled, correlation="hagedorn_brown", p1=1.0, p2=1.0,
           u=8.0, hc=None, q_max=None, n=24):
    """PCP system analysis: the rate at which the pump's own rate at the head it develops equals the average rate
    through it, the head following from the IPR (intake) and the tubing (discharge). ipr_pwf(q) -> BHP (psia), NaN
    past the AOF. Returns (solution dict or None, curve DataFrame)."""
    hc = hc or V.HeatCapacities()
    q_hi = q_max or 1.25 * pcp.pump.head_grad * pcp.speed
    qs = np.linspace(q_hi / n, q_hi, n)
    rows, last = [], None
    for q in qs:
        pwf = float(ipr_pwf(q))
        if not np.isfinite(pwf):
            break
        try:
            d, mis = operating_point(path, fluid, pwf, q, wc, gor, top_p, pcp, correlation, p1, p2, u, hc)
        except ValueError:
            break
        rows.append({"Liquid rate (STB/d)": q, "BHP (psia)": pwf, "Pump intake (psia)": d.pip,
                     "Pump discharge (psia)": d.pdp, "Head (ft)": d.head, "Average rate (RB/d)": d.q_avg,
                     "Pump rate at that head (RB/d)": d.q_avg + mis})
        last = q
    curve = pd.DataFrame(rows)
    if len(curve) < 2:
        return None, curve
    f = (curve["Pump rate at that head (RB/d)"] - curve["Average rate (RB/d)"]).to_numpy()
    k = next((j for j in range(1, len(f)) if f[j - 1] > 0 >= f[j]), None)
    sol_q = None
    if k is None:
        if f[-1] > 0:
            return {"Liquid rate (STB/d)": np.nan, "notes": [
                "The pump can move more than the well delivers: it pumps the well off (intake pressure falls to the "
                "atmosphere) - reduce the speed."]}, curve
        return None, curve

    def g(q):
        return operating_point(path, fluid, float(ipr_pwf(q)), q, wc, gor, top_p, pcp, correlation, p1, p2, u, hc)[1]
    sol_q = brentq(g, qs[k - 1], qs[k], xtol=0.05)
    pwf = float(ipr_pwf(sol_q))
    d, _ = operating_point(path, fluid, pwf, sol_q, wc, gor, top_p, pcp, correlation, p1, p2, u, hc)
    case = {"q": np.array([sol_q]), "wc": np.array([wc]), "gor": np.array([d.gor_above])}
    sol = {"Liquid rate (STB/d)": sol_q, "Oil rate (STB/d)": sol_q * (1 - wc), "Water rate (STB/d)": sol_q * wc,
           "BHP (psia)": pwf}
    sol.update(results(d, pcp, fluid, hc, case))
    # the tubing starts at the pump discharge temperature (pump and motor losses heat the fluid)
    w, cp = fluid.mass_cp(case, hc)
    t_dis = d.t_intake + float(heating(pcp, d.pdp - d.pip, d.q_avg, w, cp)[0])
    p_ = with_rod(path, pcp.rod, pcp.pump_md) if pcp.drive == "Sucker rod" else path
    _, T = V.traverse(sub_path(p_, p_.md[0], pcp.pump_md), fluid, np.array([top_p]), case, correlation, p1, p2, u,
                      hc, t_dis)
    sol["Wellhead temperature (F)"] = float(T[0, 0])
    return sol, curve


# ================================================================ traverse and lift curves (vectorised)
def heating(pcp: PCPInstalled, dp, q_avg, w_lbh, cp, speed=None):
    """Temperature rise (F) from the intake to the pump discharge: pump losses (shaft power less the hydraulic
    power) and, with a downhole motor, the motor losses heat the produced fluid (w lb/h, cp BTU/lb/F)."""
    n = pcp.speed if speed is None else speed
    dp, q_avg = np.asarray(dp, float), np.asarray(q_avg, float)
    pw = pcp.pump.shaft_power(n, dp)
    loss = np.maximum(pw - q_avg * dp * DP_HP, 0.0)
    if pcp.drive == "Downhole motor" and pcp.motor is not None:
        load = 100.0 * pw / (pcp.motor.hp * pcp.motor.scale(n))
        eff = np.clip(np.asarray(pcp.motor.at(np.minimum(load, 173.0), n)["eff"], float), 0.3, 1.0)
        loss = loss + pw * (1.0 / eff - 1.0)
    return np.where(np.isfinite(dp), loss * 2544.43 / np.maximum(np.asarray(w_lbh, float) * np.asarray(cp, float),
                                                                    1e-9), 0.0)


PCP_FREEWHEEL = True   # PROSPER lift curves: a rate above what the pump passes at zero head gets no head (BHP still finite)
PCP_RATE_AT = "ends"   # average rate through the pump: mean of the intake and discharge rates (PROSPER T29 lift curves, GOR 800: BHP within 3 %; the rate at the mean pressure gives 7 % low)


def intake_from_discharge(fluid, pdp, t, case, pcp: PCPInstalled, speed=None, iters=4):
    """Intake pressure (psia, arrays) for discharge pressures pdp: the head at which the pump at its speed passes the
    average rate, converted with the average fluid gradient (iterated). NaN where the pump cannot deliver the rate."""
    pdp = np.atleast_1d(np.asarray(pdp, float))
    n = np.broadcast_to(np.asarray(pcp.speed if speed is None else speed, float), pdp.shape)
    hc = V.HeatCapacities()
    w, _ = fluid.mass_cp(case, hc)
    mass = np.broadcast_to(np.asarray(w, float) * 24.0, pdp.shape)
    tt = np.broadcast_to(np.asarray(t, float), pdp.shape)
    pip = pdp - 300.0
    for _ in range(max(iters, 2) * 4):
        if PCP_RATE_AT == "mean":
            s = fluid.state(np.maximum(0.5 * (pip + pdp), 14.7), tt, case)
            q_avg = (np.asarray(s["ql"], float) + np.asarray(s["qg"], float)) * 86400.0 / BBL_FT3
        else:
            s1 = fluid.state(np.maximum(pip, 14.7), tt, case)
            s2 = fluid.state(np.maximum(pdp, 14.7), tt, case)
            q_avg = 0.5 * (np.asarray(s1["ql"], float) + np.asarray(s1["qg"], float) + np.asarray(s2["ql"], float)
                           + np.asarray(s2["qg"], float)) * 86400.0 / BBL_FT3
        rho = mass / (np.maximum(q_avg, 1e-9) * BBL_FT3)
        # at a speed above the one that passes the rate against the pump's maximum head (the pump would over-
        # deliver) PROSPER holds the head at the maximum head (its lift curves at 2000-3000 rpm, Tutorial 29)
        h = np.minimum(pcp.pump.head_for(n, q_avg, pcp.wear), pcp.pump.max_head)
        if PCP_FREEWHEEL:      # rate above the zero-head delivery: no head (PROSPER's lift curves)
            h = np.where(np.isnan(h) & np.isfinite(q_avg), 0.0, h)
        new = pdp - h * rho / 144.0
        if np.nanmax(np.abs(new - pip)) < 0.01:
            pip = new
            break
        pip = new
    return np.where(pip > 14.7, pip, np.nan), q_avg


def bhp_with_pcp(path, fluid, top_p, case, pcp: PCPInstalled, correlation="hagedorn_brown", p1=1.0, p2=1.0, u=8.0,
                 hc=None, speed=None):
    """Bottom-node pressure of a PCP well from the top-node pressure, vectorised over cases (lift curves): tubing (or
    rod-tubing annulus) down to the discharge, intake from the pump at its speed (speed per case, e.g. the ALQ axis),
    casing down to the bottom node. Returns dict of arrays: bhp, pip, pdp, q_avg (NaN where the pump cannot pass the
    rate or would draw the intake below atmospheric)."""
    hc = hc or V.HeatCapacities()
    shape = np.shape(case["q"])
    top_p = np.broadcast_to(np.asarray(top_p, float), shape).copy()
    p_ = with_rod(path, pcp.rod, pcp.pump_md) if pcp.drive == "Sucker rod" else path
    above = sub_path(p_, p_.md[0], pcp.pump_md)
    below = sub_path(p_, pcp.pump_md, p_.md[-1])
    w, cp = fluid.mass_cp(case, hc)
    t_in = V.temperature_profile(below, np.broadcast_to(np.asarray(w, float), shape), cp, u)[:, 0]
    w_b = np.broadcast_to(np.asarray(w, float), shape)
    t_dis = t_in.copy()
    with np.errstate(all="ignore"):
        for _ in range(2):
            P, _ = V.traverse(above, fluid, top_p, case, correlation, p1, p2, u, hc, t_dis)
            pdp = P[:, -1]
            pip, q_avg = intake_from_discharge(fluid, pdp, t_in, case, pcp, speed)
            ok = np.isfinite(pip)
            t_dis = t_in + heating(pcp, np.where(ok, pdp - pip, np.nan), q_avg, w_b, cp, speed)
        Pb, _ = V.traverse(below, fluid, np.where(ok, pip, 1000.0), case, correlation, p1, p2, u, hc)
    nan = lambda a: np.where(ok, a, np.nan)
    return {"bhp": nan(Pb[:, -1]), "pip": pip, "pdp": pdp, "q_avg": q_avg, "t_intake": t_in}


def traverse_profile(path, fluid, top_p, q, wc, gor, pcp: PCPInstalled, correlation="hagedorn_brown", p1=1.0, p2=1.0,
                     u=8.0, hc=None):
    """Gradient traverse at one liquid rate (as PROSPER's): tubing from the top node to the pump discharge, the pump
    intake from the pump at its speed, casing to the bottom node. Returns (summary dict, above path, P above, T above,
    below path, P below, T below)."""
    hc = hc or V.HeatCapacities()
    p_ = with_rod(path, pcp.rod, pcp.pump_md) if pcp.drive == "Sucker rod" else path
    above = sub_path(p_, p_.md[0], pcp.pump_md)
    below = sub_path(p_, pcp.pump_md, p_.md[-1])
    case = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([gor])}
    w, cp = fluid.mass_cp(case, hc)
    Tb0 = V.temperature_profile(below, np.asarray(w, float).reshape(1), cp, u)
    t_in = float(Tb0[0, 0])
    t_dis = t_in
    for _ in range(2):
        P, T = V.traverse(above, fluid, np.array([top_p]), case, correlation, p1, p2, u, hc, t_dis)
        pdp = float(P[0, -1])
        pip, q_avg = intake_from_discharge(fluid, np.array([pdp]), np.array([t_in]), case, pcp)
        pip = float(pip[0])
        if not np.isfinite(pip):
            raise ValueError(f"At {q:,.0f} STB/d the pump at {pcp.speed:,.1f} rpm cannot pass the rate "
                             "(or would draw the intake below atmospheric).")
        t_dis = t_in + float(heating(pcp, pdp - pip, q_avg, w, cp)[0])
    Pb, Tb = V.traverse(below, fluid, np.array([pip]), case, correlation, p1, p2, u, hc)
    s = {"Pump discharge pressure (psia)": pdp, "Pump intake pressure (psia)": pip,
         "Average rate through pump (RB/d)": float(q_avg[0]), "Pump intake temperature (F)": t_in,
         "Pump discharge temperature (F)": t_dis,
         "Pump head (ft)": float(pcp.pump.head_for(pcp.speed, q_avg[0], pcp.wear)),
         "Bottom node pressure (psia)": float(Pb[0, -1])}
    return s, above, P[0], T[0], below, Pb[0], Tb[0]
