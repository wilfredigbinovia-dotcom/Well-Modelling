"""Surface multiphase pump (PROSPER *Artificial Lift = Multiphase Pump*, Tutorial 31).

The pump is an item of the surface equipment, between the wellhead and the manifold. It is driven by a fixed
power available P (hp) and, as in PROSPER, always draws all of it: the pressure it adds follows from the hydraulic
power balance on the discharge rate

    dP = (1 - wear) * eta * P * 58771 / Qd           (psi; Qd in RB/d, 58771 psi.bbl/d per hp)

where Qd is the in-situ total (liquid + free gas) rate at the discharge pressure and temperature. The efficiency
of a twin-screw / helico-axial multiphase pump rises with the volume it moves and with its compression ratio:

    eta = A (Qd / 10,000)^B (p_out / p_in)^C        (%, clipped to [eta_min, eta_max])

Defaults are the OneSubsea H800/65 (FMPP500 frame, single pump, no wear) fitted to the user's PROSPER Tutorial 31
runs (39 VLP points, 250-600 hp, 7,400-27,800 RB/d at the discharge, intake 80-1000 psig): A 14.60, B 0.7113,
C 0.1013 reproduce PROSPER's pump pressure gain within 1.3 %. PROSPER floors the efficiency at 10 %, and the
intake pressure at 0 psig when the power would pull the intake below atmospheric (very low rates: the discharge
is then simply the downstream pressure). The pump is isothermal (PROSPER: same temperature at its inlet and
outlet). Entering B = C = 0 gives a constant efficiency A.

Units: psia, degF; rates RB/d in situ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HP_PSI_BBLD = 58771.0      # 1 hp = 550 ft.lbf/s = 58,771 psi.bbl/d
RB = 5.615
P_ATM = 14.7


@dataclass
class MultiphasePump:
    power: float = 350.0          # hp available
    a: float = 14.60              # efficiency (%) at Qd = 10,000 RB/d and pressure ratio 1
    b: float = 0.7113             # rate exponent
    c: float = 0.1013             # pressure-ratio exponent
    eta_min: float = 10.0         # %
    eta_max: float = 85.0         # %
    wear: float = 0.0             # fraction of the pressure gain lost to wear
    name: str = "OneSubsea H800/65 (FMPP500)"

    def check(self):
        out = []
        if not self.power > 0:
            out.append(("error", "The power available must be positive."))
        if not self.a > 0:
            out.append(("error", "The efficiency coefficient A must be positive."))
        if not 0 < self.eta_min <= self.eta_max <= 100:
            out.append(("error", "Efficiency limits must satisfy 0 < minimum <= maximum <= 100 %."))
        if not 0 <= self.wear < 1:
            out.append(("error", "The wear factor must be between 0 and 1."))
        return out

    def with_power(self, power):
        return MultiphasePump(float(power), self.a, self.b, self.c, self.eta_min, self.eta_max, self.wear, self.name)

    def efficiency(self, qd, ratio):
        """Efficiency (%) at discharge rate qd (RB/d) and pressure ratio p_out / p_in."""
        eta = self.a * (np.maximum(qd, 1e-9) / 1e4) ** self.b * np.maximum(ratio, 1.0) ** self.c
        return np.clip(eta, self.eta_min, self.eta_max)

    def dp(self, qd, ratio):
        return (1 - self.wear) * self.efficiency(qd, ratio) / 100.0 * self.power * HP_PSI_BBLD / np.maximum(qd, 1e-9)


def insitu_rates(fluid, case, p, t):
    """(total, liquid, gas) in-situ rates in RB/d."""
    s = fluid.state(np.asarray(p, float), np.asarray(t, float), case)
    ql = np.asarray(s["ql"], float) * 86400 / RB
    qg = np.asarray(s["qg"], float) * 86400 / RB
    return ql + qg, ql, qg


def intake_pressure(pump: MultiphasePump, fluid, case, p_out, t):
    """Intake pressure (psia) for the discharge pressure p_out (psia): the backward march through the pump."""
    p_out = np.asarray(p_out, float)
    qd, _, _ = insitu_rates(fluid, case, p_out, t)

    def g(pin):
        return p_out - pin - pump.dp(qd, p_out / pin)

    # g is concave in the intake pressure (the pressure-ratio term grows as the intake falls), so it can have two
    # roots: the physical operating point is the upper one. Locate the maximum of g (golden section), then bisect
    # between it and the discharge pressure; with no root the power would pull the intake below 0 psig.
    a_, b_ = np.full_like(p_out, P_ATM), p_out.copy()
    r = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        x1, x2 = b_ - r * (b_ - a_), a_ + r * (b_ - a_)
        left = g(x1) > g(x2)
        a_, b_ = np.where(left, a_, x1), np.where(left, x2, b_)
    lo, hi = 0.5 * (a_ + b_), p_out.copy()
    clamp = g(lo) <= 0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        pos = g(mid) > 0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    return np.where(clamp, P_ATM, 0.5 * (lo + hi))


def discharge_pressure(pump: MultiphasePump, fluid, case, p_in, t, n_iter=40):
    """Discharge pressure (psia) for the intake pressure p_in (psia): the forward march through the pump."""
    p_in = np.maximum(np.asarray(p_in, float), P_ATM)
    p = p_in + 100.0
    for _ in range(n_iter):
        qd, _, _ = insitu_rates(fluid, case, p, t)
        pn = p_in + pump.dp(qd, p / p_in)
        if np.all(np.abs(pn - p) < 1e-4 * np.maximum(p, 1.0)):
            p = pn
            break
        p = 0.5 * (p + pn)
    return p


def details(pump: MultiphasePump, fluid, case, p_in, p_out, t, hc=None):
    """Pump operating point (first rate of the arrays)."""
    qin, ql_in, qg_in = insitu_rates(fluid, case, p_in, t)
    qd, _, _ = insitu_rates(fluid, case, p_out, t)
    qavg = 0.5 * (qin + qd)
    dp = np.asarray(p_out, float) - np.asarray(p_in, float)
    eta = pump.efficiency(qd, np.asarray(p_out, float) / np.asarray(p_in, float))
    out = {"Surface pump intake pressure (psia)": p_in, "Surface pump discharge pressure (psia)": p_out,
           "Surface pump dP (psi)": dp,
           "Surface pump intake rate (RB/d)": qin, "Average rate through surface pump (RB/d)": qavg,
           "Surface pump discharge rate (RB/d)": qd, "Surface pump power (hp)": np.full_like(dp, pump.power),
           "Surface pump efficiency (%)": eta, "GVF at surface pump inlet": qg_in / np.maximum(qin, 1e-12)}
    if hc is not None:
        w, _ = fluid.mass_cp(case, hc)                      # lb/h
        rho = w / np.maximum(qin * RB / 24.0, 1e-12)         # lb/ft3, mixture at the intake (as PROSPER's head)
        out["Surface pump head (ft)"] = dp * 144.0 / rho
    return out
