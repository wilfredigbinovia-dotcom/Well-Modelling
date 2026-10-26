"""ESP QuickLook (PROSPER Matching | QuickLook | ESP; Tutorial 34): troubleshoot an ESP well test.

Two gradients at the test rate, water cut and GOR:

Fixed top node pressure ("top down")   from the measured tubing head pressure down the tubing to the pump
                                       discharge, the intake from the pump curve (stages, frequency, wear factor,
                                       gas separation), then down the casing to the bottom node.
From static BHP ("bottom up")          from the IPR pressure at the test rate (static / reservoir pressure of the
                                       QuickLook) up the casing to the intake, the pump gain to the discharge, then
                                       up the tubing to the wellhead.

If the model represents the test the two overlie each other and pass through the measured gauge, intake and
discharge pressures. Diagnostics, as the tutorial uses them:
  pump        measured dP / top-down calculated dP -> PROSPER's first-guess wear factor 1 - ratio; also the wear
              factor that makes the pump take the measured discharge pressure back to the measured intake;
  reservoir   the reservoir pressure at which the IPR gives the top-down bottom-hole pressure at the test rate;
  electrics   motor current, surface voltage and power at the pump power, against the measured values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import esp as E
from . import vlp as V

G = 14.696


@dataclass
class ESPQuickLookInput:
    thp: float                      # psia
    q: float                        # liquid STB/d
    wc: float                       # fraction
    gor: float                      # scf/STB
    pres: float                     # static bottom-hole (reservoir) pressure used by the IPR, psia
    gauges: tuple = ()              # ((md, psia), ...) - tubing / casing gauges
    pip_meas: float | None = None   # measured pump intake pressure, psia
    pdp_meas: float | None = None   # measured pump discharge pressure, psia
    amps: float | None = None       # measured current, A
    volts: float | None = None      # measured surface voltage, V
    power: float | None = None      # measured power, hp


@dataclass
class ESPQuickLookResult:
    summary: dict
    profile: pd.DataFrame           # MD, TVD, top-down and bottom-up pressures
    points: pd.DataFrame            # measured against calculated at each measurement
    top_down: dict = field(default_factory=dict)
    bottom_up: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _interp_md(md, p, x):
    ok = np.isfinite(p)
    return float(np.interp(x, md[ok], p[ok])) if ok.sum() > 1 else np.nan


def wear_for_intake(fluid, pdp, pip, t, case, pump, n_stages, freq=60.0, groups=20):
    """Wear factor with which the pump raises the measured intake pressure to the measured discharge pressure."""
    f = lambda w: float(E.pump_gain(fluid, np.array([pip]), t, case, pump, n_stages, freq, w, groups)[0][0]) - pdp
    try:
        return float(brentq(f, -0.5, 0.95, xtol=1e-5))
    except ValueError:
        return float("nan")


def esp_quicklook(path, fluid, ipr_pwf, inp: ESPQuickLookInput, esp: dict, correlation, p1=1.0, p2=1.0, u=8.0,
                  hc=None, pres_for=None, groups=20):
    """path must have nodes at the pump and gauge depths. ipr_pwf(q) -> psia at the QuickLook's static pressure;
    pres_for(pr) -> callable q -> psia (the IPR rebuilt at another reservoir pressure) for the reservoir pressure
    estimate (None: the IPR is shifted by the pressure difference - exact for a straight-line PI)."""
    hc = hc or V.HeatCapacities()
    case = {"q": np.array([inp.q]), "wc": np.array([inp.wc]), "gor": np.array([inp.gor])}
    pump, n_st, freq, wear = esp["pump"], esp["stages"], esp["freq"], esp.get("wear", 0.0)
    md_p = float(esp["pump_md"])
    above = E.sub_path(path, path.md[0], md_p)
    below = E.sub_path(path, md_p, path.md[-1])
    notes = []
    # ---------------------------------------------------------------- top down
    td = E.bhp_with_pump(path, fluid, np.array([inp.thp]), case, esp, correlation, p1, p2, u, hc, groups)
    case_a = {k: np.atleast_1d(v) for k, v in case.items()}
    case_a["gor"] = np.atleast_1d(td["gor_above"])
    Pa, Ta = V.traverse(above, fluid, np.array([inp.thp]), case_a, correlation, p1, p2, u, hc, td["t_discharge"])
    pip_td = float(td["pip"][0])
    if np.isfinite(pip_td):
        Pb, Tb = V.traverse(below, fluid, np.array([pip_td]), case, correlation, p1, p2, u, hc)
    else:
        Pb, Tb = np.full((1, len(below.md)), np.nan), np.full((1, len(below.md)), np.nan)
        notes.append("Top down: the pump would draw the intake below atmospheric at this rate (pumped off).")
    # ---------------------------------------------------------------- bottom up
    pwf = float(ipr_pwf(inp.q))
    Pbu_b, _ = V.traverse_up(below, fluid, np.array([pwf]), case, correlation, p1, p2, u, hc)
    pip_bu = float(Pbu_b[0, 0])
    t_in = float(td["t_intake"][0])
    if np.isfinite(pip_bu):
        pdp_bu, hyd_bu, pw_bu, _ = E.pump_gain(fluid, np.array([pip_bu]), t_in, case_a, pump, n_st, freq, wear, groups)
        pdp_bu, pw_bu = float(pdp_bu[0]), float(pw_bu[0])
        Pbu_a, _ = V.traverse_up(above, fluid, np.array([pdp_bu]), case_a, correlation, p1, p2, u, hc)
    else:
        pdp_bu, pw_bu = np.nan, np.nan
        Pbu_a = np.full((1, len(above.md)), np.nan)
        notes.append("Bottom up: the IPR pressure at the test rate cannot lift the fluid to the pump.")
    thp_bu = float(Pbu_a[0, 0])
    # ---------------------------------------------------------------- profile
    md = np.concatenate([above.md, below.md])
    tvd = np.concatenate([above.tvd, below.tvd])
    p_td = np.concatenate([Pa[0], Pb[0]])
    p_bu = np.concatenate([Pbu_a[0], Pbu_b[0]])
    prof = pd.DataFrame({"MD (ft)": md, "TVD (ft)": tvd, "Top down (psia)": p_td, "Bottom up (psia)": p_bu,
                         "Section": ["Tubing"] * len(above.md) + ["Casing"] * len(below.md)})
    # ---------------------------------------------------------------- measured points
    rows = []
    for m, p in inp.gauges:
        sec_a = m <= md_p
        mds, pt, pb = (above.md, Pa[0], Pbu_a[0]) if sec_a else (below.md, Pb[0], Pbu_b[0])
        rows.append({"Measurement": f"Gauge at {m:,.0f} ft", "MD (ft)": m, "Measured (psia)": p,
                     "Top down (psia)": _interp_md(mds, pt, m), "Bottom up (psia)": _interp_md(mds, pb, m)})
    if inp.pip_meas is not None:
        rows.append({"Measurement": "Pump intake", "MD (ft)": md_p, "Measured (psia)": inp.pip_meas,
                     "Top down (psia)": pip_td, "Bottom up (psia)": pip_bu})
    if inp.pdp_meas is not None:
        rows.append({"Measurement": "Pump discharge", "MD (ft)": md_p, "Measured (psia)": inp.pdp_meas,
                     "Top down (psia)": float(td["pdp"][0]), "Bottom up (psia)": pdp_bu})
    pts = pd.DataFrame(rows, columns=["Measurement", "MD (ft)", "Measured (psia)", "Top down (psia)", "Bottom up (psia)"])
    if len(pts):
        pts["Top down - measured (psi)"] = pts["Top down (psia)"] - pts["Measured (psia)"]
        pts["Bottom up - measured (psi)"] = pts["Bottom up (psia)"] - pts["Measured (psia)"]
    # ---------------------------------------------------------------- diagnostics
    s = {"Top down: pump discharge (psia)": float(td["pdp"][0]), "Top down: pump intake (psia)": pip_td,
         "Top down: pump dP (psi)": float(td["pdp"][0]) - pip_td,
         "Top down: bottom-hole pressure (psia)": float(td["bhp"][0]),
         "Bottom up: IPR pressure at the test rate (psia)": pwf, "Bottom up: pump intake (psia)": pip_bu,
         "Bottom up: pump discharge (psia)": pdp_bu, "Bottom up: tubing head pressure (psia)": thp_bu,
         "Measured tubing head pressure (psia)": inp.thp,
         "Pump intake temperature (°F)": t_in, "Pump discharge temperature (°F)": float(td["t_discharge"][0]),
         "Pump power (hp)": float(td["pump_hp"][0]), "Wear factor used": wear}
    s["Top down: pump power (hp)"] = s.pop("Pump power (hp)")
    if inp.pip_meas is not None and inp.pdp_meas is not None:
        ratio = (inp.pdp_meas - inp.pip_meas) / max(s["Top down: pump dP (psi)"], 1e-9)
        s["Measured / calculated pump dP"] = ratio
        s["Wear factor - first guess (1 - dP ratio, PROSPER)"] = (1 - ratio) + wear * ratio if wear else 1 - ratio
        s["Wear factor for the measured intake"] = wear_for_intake(fluid, inp.pdp_meas, inp.pip_meas, t_in, case_a,
                                                                   pump, n_st, freq, groups)
    bhp_td = float(td["bhp"][0])
    if np.isfinite(bhp_td):
        if pres_for is not None:
            def f(x):
                try:
                    v = float(pres_for(x)(inp.q)) - bhp_td
                except (ValueError, RuntimeError, ZeroDivisionError):
                    return np.nan
                return v if np.isfinite(v) else np.nan
            pr = np.nan
            grid = np.linspace(bhp_td + 10.0, max(inp.pres, bhp_td) * 2.0, 25)
            vals = [f(x) for x in grid]
            for a_, b_, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
                if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
                    pr = brentq(f, a_, b_, xtol=0.01)
                    break
            if np.isfinite(pr):
                s["Reservoir pressure for the top-down BHP (psia)"] = float(pr)
            else:
                notes.append("The reservoir pressure that matches the top-down BHP was not found in range.")
        else:
            s["Reservoir pressure for the top-down BHP (psia)"] = inp.pres + (bhp_td - pwf)
    s["Bottom up: pump power (hp)"] = pw_bu
    if esp.get("motor") is not None and esp.get("cable") is not None:
        for tag, hp_ in (("Top down", s["Top down: pump power (hp)"]), ("Bottom up", pw_bu)):
            if not np.isfinite(hp_):
                continue
            op = E.operate(hp_, esp["motor"], esp["cable"], freq, esp.get("cable_len", 9000.0))
            s[f"{tag}: motor current (A)"] = op["Motor amps (A)"]
            s[f"{tag}: surface voltage (V)"] = op["Voltage at surface (V)"]
            s[f"{tag}: motor power required (hp)"] = op["Motor power required (hp)"]
        for k_, m_ in (("Measured current (A)", inp.amps), ("Measured surface voltage (V)", inp.volts),
                       ("Measured power (hp)", inp.power)):
            if m_ is not None:
                s[k_] = m_
    return ESPQuickLookResult(s, prof, pts, {k: np.asarray(v).ravel() for k, v in td.items()},
                              {"pwf": pwf, "pip": pip_bu, "pdp": pdp_bu, "thp": thp_bu}, notes)
