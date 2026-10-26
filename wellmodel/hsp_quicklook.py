"""HSP QuickLook (PROSPER Matching | QuickLook | HSP; Tutorial 35): troubleshoot a hydraulic submersible pump well test.

Inputs as PROSPER's 'HSP QuickLook With Pump Speed Entered' screen: tubing head pressure, liquid rate, water cut,
produced GOR, static bottom-hole pressure, pump depth and speed, power fluid rate and surface back (injection)
pressure, pump / turbine stages and wear factor, and downhole pressures (intake, discharge, gauges).

Fixed top node pressure ("top down")   from the measured THP down the tubing with production + the power fluid
                                       (exhausted from the turbine above the pump) to the pump discharge, the intake
                                       from the pump curve (stage by stage, speed, wear), then down the casing with
                                       the produced fluid to the bottom node - hsp.traverse_profile at the test's
                                       power fluid rate.
From static BHP ("bottom up")          from the IPR pressure at the test rate up the casing to the intake, the pump
                                       gain to the discharge, then up the tubing (production + power fluid) to the
                                       wellhead.

Diagnostics as the tutorial uses them: measured / calculated pump dP (PROSPER's first-guess wear factor 1 - ratio;
also the wear factor with which the pump takes the measured intake to the measured discharge); the reservoir pressure
at which the IPR gives the top-down bottom-hole pressure at the test rate. With a turbine from the database: its power
at the test's power fluid rate against the pump's, the power fluid rate it needs, and the surface injection pressure
against the measured back pressure.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import esp as E
from . import hsp as H
from . import jetpump as J
from . import vlp as V
from .esp_quicklook import _interp_md, wear_for_intake

G = 14.696


@dataclass
class HSPQuickLookInput:
    thp: float                      # psia
    q: float                        # liquid STB/d
    wc: float                       # fraction
    gor: float                      # scf/STB
    pres: float                     # static bottom-hole (reservoir) pressure used by the IPR, psia
    speed: float                    # pump (and turbine) speed, rpm
    q_pf: float                     # power fluid rate, STB/d
    p_pf: float | None = None       # measured power fluid surface back (injection) pressure, psia
    gauges: tuple = ()              # ((md, psia), ...)
    pip_meas: float | None = None   # psia
    pdp_meas: float | None = None   # psia


@dataclass
class HSPQuickLookResult:
    summary: dict
    profile: pd.DataFrame
    points: pd.DataFrame
    top_down: dict = field(default_factory=dict)
    bottom_up: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _pres_estimate(pres_for, q, bhp, pres, pwf):
    from scipy.optimize import brentq
    if pres_for is None:
        return pres + (bhp - pwf)

    def f(x):
        try:
            v = float(pres_for(x)(q)) - bhp
        except (ValueError, RuntimeError, ZeroDivisionError):
            return np.nan
        return v if np.isfinite(v) else np.nan
    grid = np.linspace(bhp + 10.0, max(pres, bhp) * 2.0, 25)
    vals = [f(x) for x in grid]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
            return float(brentq(f, a, b, xtol=0.01))
    return np.nan


def hsp_quicklook(path, fluid, ipr_pwf, inp: HSPQuickLookInput, hsp: H.HSPInstalled, correlation, p1=1.0, p2=1.0,
                  u=8.0, hc=None, pres_for=None, groups=20):
    """path must have nodes at the pump and gauge depths. ipr_pwf(q) -> psia at the QuickLook's static pressure;
    pres_for(pr) -> callable q -> psia for the reservoir pressure estimate (None: the IPR shifted by the pressure
    difference - exact for a straight-line PI). hsp.turbine may be '' (not in the database): the pump-side
    calculation needs no turbine."""
    hc = hc or V.HeatCapacities()
    hsp = dataclasses.replace(hsp, speed=float(inp.speed))
    pump = hsp.pump_obj
    tur = H.TURBINES.get(hsp.turbine)
    freq = float(pump.freq(hsp.speed))
    md_p = float(hsp.pump_md)
    above = E.sub_path(path, path.md[0], md_p)
    below = E.sub_path(path, md_p, path.md[-1])
    sal = H._salinity(fluid, hsp.salinity)
    case = {"q": np.array([inp.q]), "wc": np.array([inp.wc]), "gor": np.array([inp.gor])}
    qt = inp.q + inp.q_pf
    case_t = {"q": np.array([qt]), "wc": np.array([(inp.q * inp.wc + inp.q_pf) / qt]), "gor": np.array([inp.gor])}
    flm = J._blend_fluid(fluid, inp.q, inp.wc, inp.q_pf, sal)
    notes = []
    lo, hi = pump.range_at(hsp.speed)
    if not pump.min_speed <= hsp.speed <= pump.max_speed:
        notes.append(f"The speed {hsp.speed:g} rpm is outside the pump's range ({pump.min_speed:g}-{pump.max_speed:g}).")
    # ---------------------------------------------------------------- top down
    nan_a, nan_b = np.full(len(above.md), np.nan), np.full(len(below.md), np.nan)
    try:
        hsp_td = dataclasses.replace(hsp, turbine=hsp.turbine if tur is not None else next(iter(H.TURBINES)))
        td, prof_td = H.traverse_profile(path, fluid, inp.thp, inp.q, inp.wc, inp.gor, hsp_td, inp.q_pf,
                                         correlation, p1, p2, u, hc)
        p_td = prof_td["Pressure (psia)"].to_numpy()
        t_td = prof_td["Temperature (F)"].to_numpy()
        Pa, Pb = p_td[:len(above.md)], p_td[len(above.md):]
        pip_td, pdp_td = td["Pump intake pressure (psia)"], td["Pump discharge pressure (psia)"]
        t_in, t_dis = td["Pump intake temperature (F)"], td["Pump discharge temperature (F)"]
    except ValueError as e:
        notes.append(f"Top down: {e}")
        td, Pa, Pb, t_td = {}, nan_a, nan_b, None
        pip_td = pdp_td = np.nan
        w, cp = fluid.mass_cp(case, hc)
        t_in = float(V.temperature_profile(below, np.asarray(w, float), cp, u)[0, 0])
        t_dis = t_in
    # ---------------------------------------------------------------- bottom up
    pwf = float(ipr_pwf(inp.q))
    Pbu_b, _ = V.traverse_up(below, fluid, np.array([pwf]), case, correlation, p1, p2, u, hc)
    pip_bu = float(Pbu_b[0, 0])
    pdp_bu = pw_bu = qav_bu = np.nan
    Pbu_a = nan_a[None, :]
    if np.isfinite(pip_bu):
        pdp_, hyd_, pw_, qav_ = E.pump_gain(fluid, np.array([pip_bu]), np.array([t_in]), case, pump.curve, hsp.stages,
                                            freq, hsp.wear, groups)
        pdp_bu, pw_bu, qav_bu = float(pdp_[0]), float(pw_[0]), float(qav_[0])
        w, cp = fluid.mass_cp(case, hc)
        t_dis_bu = t_in + max(pw_bu - float(hyd_[0]), 0.0) * H.BTU_HP / max(
            float(np.ravel(w)[0]) * float(np.ravel(cp)[0]), 1e-9)
        Pbu_a, _ = V.traverse_up(above, flm, np.array([pdp_bu]), case_t, correlation, p1, p2, u, hc, t_dis_bu)
    else:
        notes.append("Bottom up: the IPR pressure at the test rate cannot lift the fluid to the pump.")
    thp_bu = float(Pbu_a[0, 0])
    # ---------------------------------------------------------------- profile and measured points
    md = np.concatenate([above.md, below.md])
    prof = pd.DataFrame({"MD (ft)": md, "TVD (ft)": np.concatenate([above.tvd, below.tvd]),
                         "Top down (psia)": np.concatenate([Pa, Pb]),
                         "Bottom up (psia)": np.concatenate([Pbu_a[0], Pbu_b[0]]),
                         "Top down temperature (F)": t_td if t_td is not None else np.full(len(md), np.nan),
                         "Section": ["Tubing (production + power fluid)"] * len(above.md) + ["Casing"] * len(below.md)})
    rows = []
    for m, p in inp.gauges:
        sec_a = m <= md_p
        mds, pt, pb = (above.md, Pa, Pbu_a[0]) if sec_a else (below.md, Pb, Pbu_b[0])
        rows.append({"Measurement": f"Gauge at {m:,.0f} ft", "MD (ft)": m, "Measured (psia)": p,
                     "Top down (psia)": _interp_md(mds, np.asarray(pt), m),
                     "Bottom up (psia)": _interp_md(mds, np.asarray(pb), m)})
    if inp.pip_meas is not None:
        rows.append({"Measurement": "Pump intake", "MD (ft)": md_p, "Measured (psia)": inp.pip_meas,
                     "Top down (psia)": pip_td, "Bottom up (psia)": pip_bu})
    if inp.pdp_meas is not None:
        rows.append({"Measurement": "Pump discharge", "MD (ft)": md_p, "Measured (psia)": inp.pdp_meas,
                     "Top down (psia)": pdp_td, "Bottom up (psia)": pdp_bu})
    pts = pd.DataFrame(rows, columns=["Measurement", "MD (ft)", "Measured (psia)", "Top down (psia)", "Bottom up (psia)"])
    if len(pts):
        pts["Top down - measured (psi)"] = pts["Top down (psia)"] - pts["Measured (psia)"]
        pts["Bottom up - measured (psi)"] = pts["Bottom up (psia)"] - pts["Measured (psia)"]
    # ---------------------------------------------------------------- diagnostics
    bhp_td = float(Pb[-1]) if len(Pb) else np.nan
    s = {"Top down: pump discharge (psia)": pdp_td, "Top down: pump intake (psia)": pip_td,
         "Top down: pump dP (psi)": pdp_td - pip_td, "Top down: bottom-hole pressure (psia)": bhp_td,
         "Top down: pump power (hp)": td.get("Pump power (hp)", np.nan),
         "Bottom up: IPR pressure at the test rate (psia)": pwf, "Bottom up: pump intake (psia)": pip_bu,
         "Bottom up: pump discharge (psia)": pdp_bu, "Bottom up: tubing head pressure (psia)": thp_bu,
         "Bottom up: pump power (hp)": pw_bu, "Measured tubing head pressure (psia)": inp.thp,
         "Pump intake temperature (°F)": t_in, "Pump discharge temperature (°F)": t_dis,
         "Average rate through pump (RB/d)": td.get("Average rate through pump (RB/d)", qav_bu),
         "Pump recommended range at this speed (RB/d)": f"{lo:,.0f} - {hi:,.0f}",
         "Wear factor used": hsp.wear, "Pump speed (rpm)": hsp.speed, "Power fluid rate (STB/d)": inp.q_pf}
    if inp.pip_meas is not None and inp.pdp_meas is not None and np.isfinite(s["Top down: pump dP (psi)"]):
        ratio = (inp.pdp_meas - inp.pip_meas) / max(s["Top down: pump dP (psi)"], 1e-9)
        s["Measured / calculated pump dP"] = ratio
        s["Wear factor - first guess (1 - dP ratio, PROSPER)"] = (1 - ratio) + hsp.wear * ratio
        s["Wear factor for the measured intake"] = wear_for_intake(fluid, inp.pdp_meas, inp.pip_meas, t_in, case,
                                                                   pump.curve, hsp.stages, freq, groups)
    if np.isfinite(bhp_td):
        pr = _pres_estimate(pres_for, inp.q, bhp_td, inp.pres, pwf)
        if np.isfinite(pr):
            s["Reservoir pressure for the top-down BHP (psia)"] = pr
        else:
            notes.append("The reservoir pressure that matches the top-down BHP was not found in range.")
    if tur is not None and td:
        bw_top = H.water_bw_rho(sal, inp.thp, td["Wellhead temperature (F)"])[0]
        s["Turbine power at the power fluid rate (hp)"] = td["Turbine power (hp)"]
        s["Turbine - pump power (hp)"] = td["Turbine power (hp)"] - td["Pump power (hp)"]
        s["Power fluid rate for the pump power (STB/d)"] = H.power_fluid_for(tur, hsp.turbine_stages, hsp.speed,
                                                                             td["Pump power (hp)"], bw_top)
        s["Delta P across turbine (psi)"] = td["Delta P across turbine (psi)"]
        s["Calculated power fluid surface pressure (psia)"] = td["Power fluid injection pressure (psia)"]
        if inp.p_pf is not None:
            s["Measured power fluid surface pressure (psia)"] = inp.p_pf
            s["Calculated - measured power fluid pressure (psi)"] = td["Power fluid injection pressure (psia)"] - inp.p_pf
    elif tur is None:
        notes.append(f"Turbine '{hsp.turbine}' is not in the database: turbine power and power fluid pressure are "
                     "not calculated (the pump-side gradients do not need them).")
    return HSPQuickLookResult(s, prof, pts, td, {"pwf": pwf, "pip": pip_bu, "pdp": pdp_bu, "thp": thp_bu}, notes)
