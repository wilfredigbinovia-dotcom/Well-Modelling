"""Inflow performance (IPR) models, following the PROSPER model list.

Oilfield units: psia, degF, ft, md, cp. Oil-model rates are liquid STB/d (the
PI refers to liquid rate, as in PROSPER). Gas-model rates are Mscf/d
internally (the app shows MMscf/d).

Oil and water
-------------
PI Entry, Vogel, Composite (Petrobras), Darcy, Fetkovich, Multi-rate
Fetkovich, Jones, Multi-rate Jones, Transient, External entry.
All but the Jones/Fetkovich forms use a straight line above the bubble point
and Vogel below it.

Gas and condensate
------------------
Jones, Forchheimer, Back pressure, C and n, Multi-rate C and n, Multi-rate
Jones, Forchheimer with pseudo-pressure, Multi-rate Forchheimer with
pseudo-pressure, External entry.

Key relations
-------------
Dietz drainage term   0.5 ln(4A / (gamma C_A rw^2)), gamma = 1.781
Darcy (oil)           J = k h / (141.2 mu Bo (dietz + S))
Jones (oil)           Pr - Pwf = a Q^2 + b Q;  b = 141.2 mu Bo (dietz + S)/(k h),
                      a = 2.30e-14 beta Bo^2 rho / (hp^2 rw), beta = 2.33e10 / k^1.201
Jones (gas)           Pr^2 - Pwf^2 = a Q^2 + b Q;  b = 1422 T mu Z (dietz + S)/(k h),
                      a = 1422 T mu Z D/(k h), D = 2.222e-15 beta SG k h/(mu hp^2 rw),
                      beta = 2.73e10 / k^1.1045  (Dake eq. 8.24/8.27, as in the PROSPER guide)
Transient (oil)       Pr - Pwf = 162.6 q mu Bo/(k h) [log(k t/(phi mu Ct rw^2)) - 3.23 + 0.87 S]
Pseudo-pressure       psi(p) = 2 int_0^p p/(mu Z) dp
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

GAMMA_EULER = 1.781
ACRE_FT2 = 43560.0
RANKINE = 459.67


# ================================================================ helpers
def dietz_term(area_acres: float, ca: float, rw: float) -> float:
    """0.5 ln(4A/(gamma C_A rw^2)); equals ln(re/rw) - 0.75 for a centred circle (C_A = 31.62)."""
    return 0.5 * np.log(4 * area_acres * ACRE_FT2 / (GAMMA_EULER * ca * rw ** 2))


DIETZ_SHAPES = {   # common Dietz shape factors (well location in brackets)
    "Circle (centre)": 31.62, "Square (centre)": 30.88, "Hexagon (centre)": 31.6,
    "Square (quarter point)": 4.513, "2:1 rectangle (centre)": 21.84,
    "4:1 rectangle (centre)": 5.379, "Square (corner)": 0.5598,
}


@dataclass
class IPRCurve:
    model: str
    fluid: str                       # "oil" or "gas"
    pr: float                        # reservoir pressure used (may be fitted)
    pwf: np.ndarray
    q: np.ndarray                    # liquid STB/d (oil) or Mscf/d (gas)
    params: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    tests: pd.DataFrame | None = None
    rate_fn: Callable | None = None  # q(pwf), vectorised

    @property
    def aof(self) -> float:
        return float(self.q[np.argmin(self.pwf)])

    def rate(self, pwf):
        if self.rate_fn is not None:
            return np.maximum(self.rate_fn(np.asarray(pwf, float)), 0.0)
        order = np.argsort(self.pwf)
        return np.interp(pwf, self.pwf[order], self.q[order])

    def table(self, wc=0.0, gor=None, cgr=None, wgr=None) -> pd.DataFrame:
        df = pd.DataFrame({"Pwf (psia)": self.pwf})
        if self.fluid == "oil":
            df["Liquid rate (STB/d)"] = self.q
            df["Oil rate (STB/d)"] = self.q * (1 - wc)
            df["Water rate (STB/d)"] = self.q * wc
            if gor:
                df["Gas rate (MMscf/d)"] = self.q * (1 - wc) * gor / 1e6
        else:
            df["Gas rate (MMscf/d)"] = self.q / 1e3
            if cgr is not None:
                df["Condensate rate (STB/d)"] = self.q / 1e3 * cgr
            if wgr is not None:
                df["Water rate (STB/d)"] = self.q / 1e3 * wgr
        return df


def _grid(pr, n=60):
    return np.linspace(pr, 0.0, n)


def _curve_from_rate(model, fluid, pr, rate_fn, params, notes=None, tests=None, n=60):
    pw = _grid(pr, n)
    return IPRCurve(model, fluid, pr, pw, np.maximum(rate_fn(pw), 0.0), params, notes or [], tests, rate_fn)


# ============================================================ oil models
def _pi_vogel_rate(pr, pb, j, wc=0.0, vogel=True):
    """Liquid rate at pwf: straight-line PI above the bubble point, Vogel below. With water (wc > 0) the
    Petrobras composite form is used below the bubble point - oil follows Vogel, water a straight line - as
    PROSPER does for P.I. Entry and Darcy (reproduces its AOF at 80 % water cut to 0.05 %). vogel=False: PROSPER's
    'Vogel below bubble point' switched off - a straight line at every pressure (the user's Tutorial 29 file: 21
    solution-node pressures on one line within 0.2 psi)."""
    if not vogel:
        def fl(pwf):
            return j * (pr - np.clip(pwf, 0, None))
        return fl
    pb_eff = min(pb, pr)
    qb = j * max(pr - pb, 0.0)
    if wc and wc > 0 and pb_eff > 0:
        pwf_of_q, _, _, q_tmax = _composite_pwf(pr, pb, j, float(min(wc, 1.0)))
        qq = np.linspace(0.0, q_tmax, 2001)
        pp = pwf_of_q(qq)                                   # falls with q

        def fc(pwf):
            return np.interp(np.clip(pwf, 0, None), pp[::-1], qq[::-1])
        return fc

    def f(pwf):
        pwf = np.clip(pwf, 0, None)
        x = pwf / max(pb_eff, 1e-9)
        return np.where(pwf >= pb_eff, j * (pr - pwf),
                        qb + j * pb_eff / 1.8 * (1 - 0.2 * x - 0.8 * x * x))
    return f


def pi_entry(pr, pb, j, wc=0.0, vogel=True):
    f = _pi_vogel_rate(pr, pb, j, wc, vogel)
    return _curve_from_rate("P.I. Entry", "oil", pr, f, {"PI (STB/d/psi)": j})


def _vogel_j(pr, pb, q_test, pwf_test):
    pb_eff = min(pb, pr)
    if pwf_test >= pb_eff:
        return q_test / (pr - pwf_test)
    x = pwf_test / pb_eff
    return q_test / (max(pr - pb, 0) + pb_eff / 1.8 * (1 - 0.2 * x - 0.8 * x * x))


def vogel(pr, pb, q_test, pwf_test):
    if not 0 <= pwf_test < pr:
        raise ValueError("Test FBHP must be below reservoir pressure.")
    j = _vogel_j(pr, pb, q_test, pwf_test)
    c = pi_entry(pr, pb, j)
    c.model = "Vogel"
    c.tests = pd.DataFrame({"Rate": [q_test], "Pwf": [pwf_test]})
    return c


def _composite_pwf(pr, pb, j, wc):
    """Petrobras composite IPR: Pwf(q) for liquid rate q (Brown, vol. 4)."""
    pb_eff = min(pb, pr)
    fw, fo = wc, 1 - wc
    qb = j * max(pr - pb, 0.0)
    qomax = qb + j * pb_eff / 1.8
    slope = (fw + 9 * fo) / j
    p_at_qomax = fw * (pr - qomax / j)

    def pwf(q):
        q = np.asarray(q, float)
        u = np.clip((q - qb) / max(qomax - qb, 1e-12), 0, 1)
        mid = fw * (pr - q / j) + fo * 0.125 * pb_eff * (-1 + np.sqrt(81 - 80 * u))
        out = np.where(q <= qb, pr - q / j, np.where(q <= qomax, mid, p_at_qomax - (q - qomax) * slope))
        return out
    q_tmax = qomax + p_at_qomax / slope
    return pwf, qb, qomax, q_tmax


def composite(pr, pb, q_test, pwf_test, wc):
    if not 0 <= pwf_test < pr:
        raise ValueError("Test FBHP must be below reservoir pressure.")
    if not 0 <= wc <= 1:
        raise ValueError("Water cut must be between 0 and 1.")
    f = lambda j: _composite_pwf(pr, pb, j, wc)[0](q_test) - pwf_test
    j = brentq(f, 1e-8, 1e6)
    pwf_of_q, qb, qomax, qtmax = _composite_pwf(pr, pb, j, wc)
    qq = np.linspace(0, qtmax, 400)
    pp = np.maximum(pwf_of_q(qq), 0)
    rate_fn = lambda pw: np.interp(pw, pp[::-1], qq[::-1])
    c = _curve_from_rate("Composite", "oil", pr, rate_fn, {"PI (STB/d/psi)": j, "Water cut": wc,
                                                          "Max oil-type rate (STB/d)": qomax})
    c.tests = pd.DataFrame({"Rate": [q_test], "Pwf": [pwf_test]})
    return c


def darcy_pi(k, h, mu, bo, area_acres, ca, rw, skin):
    return k * h / (141.2 * mu * bo * (dietz_term(area_acres, ca, rw) + skin))


def darcy(pr, pb, k, h, area_acres, ca, rw, skin, mu, bo, wc=0.0, vogel=True):
    j = darcy_pi(k, h, mu, bo, area_acres, ca, rw, skin)
    c = pi_entry(pr, pb, j, wc, vogel)
    c.model = "Darcy"
    c.params.update({"Dietz term": dietz_term(area_acres, ca, rw)})
    return c


def fetkovich(pr, pb, k, h, area_acres, ca, rw, skin, mu, bo, kro):
    """Darcy PI (with oil relative permeability) above Pb; Fetkovich below:
    q = J(Pr - Pb) + J/(2 Pb) (Pb^2 - Pwf^2)."""
    j = darcy_pi(k * kro, h, mu, bo, area_acres, ca, rw, skin)
    pb_eff = min(pb, pr)

    def f(pwf):
        pwf = np.clip(pwf, 0, None)
        return np.where(pwf >= pb_eff, j * (pr - pwf),
                        j * max(pr - pb, 0) + j / (2 * pb_eff) * (pb_eff ** 2 - pwf ** 2))
    return _curve_from_rate("Fetkovich", "oil", pr, f, {"PI (STB/d/psi)": j, "kro": kro})


def _check_tests(tests: pd.DataFrame, min_points: int):
    t = tests.dropna()
    t = t[(t["Rate"] > 0) & (t["Pwf"] >= 0)]
    if len(t) < min_points:
        raise ValueError(f"Need at least {min_points} valid test points (rate > 0).")
    return t.sort_values("Rate").reset_index(drop=True)


def _need_pr_fit(pr, tests, producer=True):
    return producer and pr <= tests["Pwf"].max()


def multirate_fetkovich(pr, tests, fit_pr=None):
    """Q = C ((Pr^2 - Pwf^2)/1000)^n fitted to up to 10 tests."""
    t = _check_tests(tests, 2)
    q, pw = t["Rate"].to_numpy(float), t["Pwf"].to_numpy(float)
    fit_pr = _need_pr_fit(pr, t) if fit_pr is None else fit_pr
    notes = []
    if fit_pr:
        if len(t) < 3:
            raise ValueError("Calculating reservoir pressure needs at least 3 test points.")

        def res(x):
            lc, n, p = x
            return np.log(q) - (lc + n * np.log(np.maximum((p ** 2 - pw ** 2) / 1000, 1e-12)))
        x0 = [0.0, 0.8, pw.max() * 1.1]
        s = least_squares(res, x0, bounds=([-50, 0.3, pw.max() + 1], [50, 1.5, pw.max() * 5]))
        lc, n, pr = s.x
        notes.append(f"Reservoir pressure calculated from the tests: {pr:,.1f} psia.")
    else:
        x = np.log((pr ** 2 - pw ** 2) / 1000)
        n, lc = np.polyfit(x, np.log(q), 1)
    c_ = float(np.exp(lc))
    f = lambda pwf: c_ * np.maximum((pr ** 2 - np.clip(pwf, 0, None) ** 2) / 1000, 0) ** n
    resid = f(pw) - q
    return _curve_from_rate("Multi-rate Fetkovich", "oil", pr, f,
                            {"C": c_, "n": float(n), "Rate RMS error (%)": float(np.sqrt(np.mean((resid / q) ** 2)) * 100)},
                            notes, t)


def _jones_rate(pr, a, b, squared):
    def f(pwf):
        pwf = np.clip(pwf, 0, None)
        dp = (pr ** 2 - pwf ** 2) if squared else (pr - pwf)
        dp = np.maximum(dp, 0)
        if a <= 1e-30:
            return dp / b
        return (-b + np.sqrt(b * b + 4 * a * dp)) / (2 * a)
    return f


def jones_oil_coeffs(k, h, area_acres, ca, rw, skin, mu, bo, rho, hp):
    b = 141.2 * mu * bo * (dietz_term(area_acres, ca, rw) + skin) / (k * h)
    beta = 2.33e10 / k ** 1.201
    a = 2.30e-14 * beta * bo ** 2 * rho / (hp ** 2 * rw)
    return a, b


def jones_oil(pr, k, h, area_acres, ca, rw, skin, mu, bo, rho, hp):
    a, b = jones_oil_coeffs(k, h, area_acres, ca, rw, skin, mu, bo, rho, hp)
    return _curve_from_rate("Jones", "oil", pr, _jones_rate(pr, a, b, False),
                            {"a (psi/(STB/d)^2)": a, "b (psi/(STB/d))": b})


def _fit_jones(pr, tests, squared, fit_pr, producer=True):
    t = _check_tests(tests, 2)
    q, pw = t["Rate"].to_numpy(float), t["Pwf"].to_numpy(float)
    fit_pr = _need_pr_fit(pr, t, producer) if fit_pr is None else fit_pr
    notes = []
    if fit_pr:
        if len(t) < 3:
            raise ValueError("Calculating reservoir pressure needs at least 3 test points.")
        # dp = aQ^2 + bQ, dp = Pr - Pwf (or squared): linear in (a, b, Pr or Pr^2)
        y = pw ** 2 if squared else pw
        A = np.column_stack([-q ** 2, -q, np.ones_like(q)])
        a, b, c0 = np.linalg.lstsq(A, y, rcond=None)[0]
        pr = float(np.sqrt(c0) if squared else c0)
        notes.append(f"Reservoir pressure calculated from the tests: {pr:,.1f} psia.")
    else:
        dp = (pr ** 2 - pw ** 2) if squared else (pr - pw)
        A = np.column_stack([q ** 2, q])
        a, b = np.linalg.lstsq(A, dp, rcond=None)[0]
    if b <= 0:
        raise ValueError("Fit gave a non-positive laminar coefficient b - check the test data.")
    if a < 0:
        notes.append("Fitted turbulence coefficient a was negative (data scatter) - set to zero.")
        a = 0.0
    f = _jones_rate(pr, a, b, squared)
    resid = f(pw) - q
    return pr, float(a), float(b), notes, t, float(np.sqrt(np.mean((resid / q) ** 2)) * 100)


def multirate_jones_oil(pr, tests, fit_pr=None):
    pr, a, b, notes, t, err = _fit_jones(pr, tests, False, fit_pr)
    return _curve_from_rate("Multi-rate Jones", "oil", pr, _jones_rate(pr, a, b, False),
                            {"a (psi/(STB/d)^2)": a, "b (psi/(STB/d))": b, "Rate RMS error (%)": err},
                            notes, t)


def transient_oil(pr, pb, k, h, area_acres, ca, rw, skin, mu, bo, phi, ct, t_days):
    """Transient PI (radial infinite-acting) until it falls to the pseudo-steady-state PI."""
    if t_days < 0.5:
        raise ValueError("Time must be at least 0.5 days.")
    t_hr = t_days * 24
    x = np.log10(k * t_hr / (phi * mu * ct * rw ** 2)) - 3.23 + 0.87 * skin
    j_tr = k * h / (162.6 * mu * bo * x)
    j_pss = darcy_pi(k, h, mu, bo, area_acres, ca, rw, skin)
    j = max(j_tr, j_pss)
    state = "transient" if j_tr > j_pss else "pseudo-steady state"
    c = pi_entry(pr, pb, j)
    c.model = "Transient"
    c.params = {"PI (STB/d/psi)": j, "Transient PI": j_tr, "Pseudo-steady-state PI": j_pss}
    c.notes = [f"At {t_days:g} days the well is in {state} flow."]
    return c


def external(table: pd.DataFrame, fluid: str, rate_scale=1.0):
    t = table.dropna()
    t = t[(t["Rate"] >= 0) & (t["Pwf"] >= 0)].sort_values("Pwf")
    if len(t) < 2:
        raise ValueError("Enter at least 2 (rate, pressure) points.")
    pw, q = t["Pwf"].to_numpy(float), t["Rate"].to_numpy(float) * rate_scale
    if np.any(np.diff(q) > 1e-9):
        raise ValueError("Rate must fall as flowing pressure rises.")
    pr = float(pw.max())
    f = lambda pwf: np.interp(pwf, pw, q)
    return _curve_from_rate("External entry", fluid, pr, f, {"Points": len(t)},
                            ["Reservoir pressure taken as the highest pressure in the table."
                             if q[-1] == 0 else "Highest pressure in the table has a non-zero rate."],
                            None)


# ============================================================ gas models
def non_darcy_d(k, h, hp, rw, mu, sg):
    """Non-Darcy coefficient D (1/(Mscf/d)), Dake eq. 8.27 with beta = 2.73e10 / k^1.1045."""
    beta = 2.73e10 / k ** 1.1045
    return 2.222e-15 * beta * sg * k * h / (mu * hp ** 2 * rw)


def jones_gas_coeffs(k, h, area_acres, ca, rw, skin, t_f, mu, z, sg, hp, d=None):
    tr = t_f + RANKINE
    base = 1422 * tr * mu * z / (k * h)
    d = non_darcy_d(k, h, hp, rw, mu, sg) if d is None else d
    return base * d, base * (dietz_term(area_acres, ca, rw) + skin), d


def jones_gas(pr, k, h, area_acres, ca, rw, skin, t_f, mu, z, sg, hp, d=None):
    a, b, d = jones_gas_coeffs(k, h, area_acres, ca, rw, skin, t_f, mu, z, sg, hp, d)
    return _curve_from_rate("Jones", "gas", pr, _jones_rate(pr, a, b, True),
                            {"a (psi^2/(Mscf/d)^2)": a, "b (psi^2/(Mscf/d))": b, "D (1/(Mscf/d))": d},
                            ["Viscosity and Z held at reservoir pressure (Jones assumption)."])


def forchheimer(pr, a, b):
    """Pr^2 - Pwf^2 = a Q^2 + b Q (a turbulent, b laminar; Q in Mscf/d)."""
    return _curve_from_rate("Forchheimer", "gas", pr, _jones_rate(pr, a, b, True),
                            {"a (psi^2/(Mscf/d)^2)": a, "b (psi^2/(Mscf/d))": b})


def c_and_n(pr, c, n):
    f = lambda pwf: c * np.maximum(pr ** 2 - np.clip(pwf, 0, None) ** 2, 0) ** n
    return _curve_from_rate("C and n", "gas", pr, f, {"C (Mscf/d/psi^2n)": c, "n": n})


def back_pressure(pr, k, h, area_acres, ca, rw, skin, t_f, mu, z, n):
    """C from reservoir properties: Darcy C at n = 1, rescaled so that the AOF is
    unchanged for n < 1 (C = C_darcy Pr^(2 - 2n))."""
    if not 0.5 <= n <= 1:
        raise ValueError("Exponent n must be between 0.5 and 1.")
    c_d = k * h / (1422 * (t_f + RANKINE) * mu * z * (dietz_term(area_acres, ca, rw) + skin))
    c = c_d * pr ** (2 - 2 * n)
    out = c_and_n(pr, c, n)
    out.model = "Back pressure"
    out.notes = ["C derived from the Darcy deliverability, keeping the same absolute open flow for n < 1."]
    return out


def multirate_c_and_n(pr, tests, fit_pr=None):
    t = _check_tests(tests, 2)
    q, pw = t["Rate"].to_numpy(float), t["Pwf"].to_numpy(float)
    fit_pr = _need_pr_fit(pr, t) if fit_pr is None else fit_pr
    notes = []
    if fit_pr:
        if len(t) < 3:
            raise ValueError("Calculating reservoir pressure needs at least 3 test points.")

        def res(x):
            lc, n, p = x
            return np.log(q) - (lc + n * np.log(np.maximum(p ** 2 - pw ** 2, 1e-12)))
        s = least_squares(res, [-10.0, 0.8, pw.max() * 1.1],
                          bounds=([-80, 0.5, pw.max() + 1], [20, 1.0, pw.max() * 5]))
        lc, n, pr = s.x
        notes.append(f"Reservoir pressure calculated from the tests: {pr:,.1f} psia.")
    else:
        x = np.log(pr ** 2 - pw ** 2)
        n, lc = np.polyfit(x, np.log(q), 1)
        if not 0.5 <= n <= 1:
            nc = float(np.clip(n, 0.5, 1.0))
            notes.append(f"Fitted n = {n:.3f} is outside 0.5-1; constrained to {nc:.2f}.")
            n = nc
            lc = float(np.mean(np.log(q) - n * x))
        if len(t) < 3:
            notes.append("Fewer than 3 test points: C and n are poorly constrained.")
    c = float(np.exp(lc))
    out = c_and_n(pr, c, float(n))
    out.model = "Multi-rate C and n"
    resid = out.rate(pw) - q
    out.params["Rate RMS error (%)"] = float(np.sqrt(np.mean((resid / q) ** 2)) * 100)
    out.notes, out.tests = notes, t
    return out


def multirate_jones_gas(pr, tests, fit_pr=None):
    pr, a, b, notes, t, err = _fit_jones(pr, tests, True, fit_pr)
    return _curve_from_rate("Multi-rate Jones", "gas", pr, _jones_rate(pr, a, b, True),
                            {"a (psi^2/(Mscf/d)^2)": a, "b (psi^2/(Mscf/d))": b, "Rate RMS error (%)": err},
                            notes, t)


# ---------------------------------------------------------- pseudo-pressure
def pseudo_pressure_fn(mu_z: Callable, p_max: float, n: int = 300, gauge: bool = False):
    """Return psi(p) = 2 int_0^p p/(mu Z) dp (psi^2/cp) as an interpolating function.
    mu_z(p) returns (mu, z) arrays. gauge=True: PROSPER's M(P), with the gauge pressure p - 14.696 in the integrand."""
    p = np.linspace(0.0, p_max * 1.05, n)
    mu, z = mu_z(np.maximum(p, 14.7))
    integrand = 2 * (np.maximum(p - 14.696, 0.0) if gauge else p) / (mu * z)
    psi = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(p))])
    return lambda x: np.interp(x, p, psi)


def forchheimer_pseudo(pr, a, b, psi):
    """psi(Pr) - psi(Pwf) = a Q + b Q^2 (a laminar, b turbulent, as in the PROSPER guide)."""
    psr = psi(pr)

    def f(pwf):
        d = np.maximum(psr - psi(np.clip(pwf, 0, None)), 0)
        return d / a if b <= 1e-30 else (-a + np.sqrt(a * a + 4 * b * d)) / (2 * b)
    return _curve_from_rate("Forchheimer with pseudo-pressure", "gas", pr, f,
                            {"a (laminar)": a, "b (turbulent)": b, "psi(Pr) (psi^2/cp)": float(psr)})


def multirate_forchheimer_pseudo(pr, tests, psi):
    t = _check_tests(tests, 2)
    q, pw = t["Rate"].to_numpy(float), t["Pwf"].to_numpy(float)
    if pr <= pw.max():
        raise ValueError("Reservoir pressure must exceed every test FBHP for this model.")
    d = psi(pr) - psi(pw)
    a, b = np.linalg.lstsq(np.column_stack([q, q ** 2]), d, rcond=None)[0]
    notes = []
    if b < 0:
        b = 0.0
        a = float(np.sum(d * q) / np.sum(q * q))
        notes.append("Fitted turbulence coefficient was negative (data scatter) - set to zero.")
    out = forchheimer_pseudo(pr, float(a), float(b), psi)
    out.model = "Multi-rate Forchheimer with pseudo-pressure"
    resid = out.rate(pw) - q
    out.params["Rate RMS error (%)"] = float(np.sqrt(np.mean((resid / q) ** 2)) * 100)
    out.notes, out.tests = notes, t
    return out


# ================================================================ catalogue
OIL_MODELS = ["P.I. Entry", "Vogel", "Composite", "Darcy", "Fetkovich", "Multi-rate Fetkovich",
              "Jones", "Multi-rate Jones", "Transient", "External entry"]
GAS_MODELS = ["Petroleum Experts", "Jones", "Forchheimer", "Back pressure", "C and n", "Multi-rate C and n",
              "Multi-rate Jones", "Forchheimer with pseudo-pressure",
              "Multi-rate Forchheimer with pseudo-pressure", "External entry"]
NOT_YET = {
    "oil": ["Hydraulically Fractured Well", "Horizontal Well (no flow / constant pressure boundary)",
            "Multi-layer (incl. dP loss in wellbore)", "Horizontal Well with friction dP",
            "SkinAide", "Dual Porosity", "Horizontal Well with transverse fractures",
            "Thermally Induced Fracture"],
    "gas": ["Petroleum Experts (multiphase pseudo-pressure - proprietary relative-permeability correction)",
            "Hydraulically Fractured Well", "Horizontal Well - no flow boundaries",
            "Multi-layer (incl. dP loss in wellbore)", "Horizontal Well with friction dP",
            "SkinAide", "Dual Porosity", "Horizontal Well with transverse fractures",
            "Modified Isochronal"],
}


# ============================================================ Petroleum Experts gas IPR
PETEX_PSS_OFFSET = 0.4932   # PROSPER's stabilised drainage term = Dietz term + this (fitted: area, CA, rw runs)
# gas relative permeability at connate water (straight-line kr with PROSPER's end-point degradation), from PROSPER
# runs at Swc = 0, 0.1, 0.25, 0.4 (Tutorial 02): cubic through those points, linear beyond 0.4
_KRG_SWC = (0.0, 0.10, 0.25, 0.40)
_KRG_VAL = (1.0, 0.8988, 0.7328, 0.5227)


def petex_krg(swc):
    """Gas relative permeability at connate water saturation (PROSPER Petroleum Experts IPR, fitted)."""
    swc = float(np.clip(swc, 0.0, 0.95))
    if swc <= 0.4:
        c = np.polyfit(_KRG_SWC, _KRG_VAL, 3)
        return float(np.polyval(c, swc))
    slope = (_KRG_VAL[3] - _KRG_VAL[2]) / (_KRG_SWC[3] - _KRG_SWC[2])
    return float(max(_KRG_VAL[3] + slope * (swc - 0.4), 0.01))


# gas relative permeability with produced liquid (PROSPER's 'Calculated Gas Relative Permeability', rel perms off):
# the water saturation rises above connate by KAPPA x the liquid volume fraction at reservoir conditions, liquid =
# WGR + C_CGR x CGR (stock-tank STB/MMscf, B = 1), gas = Bg (rb/MMscf); krg = krg(Swc) x f(Sw) / f(Swc) with
# f(S) = (1 - S)^(1 + a S + b S^2). Fitted on the user's Tutorial 47 file (1750 psig, 300 F, CGR 134.5, WGR 0-2000,
# Swc 0.2 and 0.3: 0.24-0.74, all within 0.5 %); with no liquid it is petex_krg(Swc) (Tutorial 02).
PETEX_KRG_LIQ = (0.68086, 0.59868, 0.76963, 0.82765)     # a, b, KAPPA, C_CGR


def petex_krg_liquid(swc, wgr=0.0, cgr=0.0, bg_rb_mmscf=None):
    """Gas relative permeability at Swc with produced water and condensate (see PETEX_KRG_LIQ)."""
    k0 = petex_krg(swc)
    liq = max(float(wgr), 0.0) + PETEX_KRG_LIQ[3] * max(float(cgr), 0.0)
    if liq <= 0 or not bg_rb_mmscf or bg_rb_mmscf <= 0:
        return k0
    a, b, kap, _ = PETEX_KRG_LIQ
    f = lambda s_: (1 - s_) ** (1 + a * s_ + b * s_ * s_)
    sw = min(swc + kap * liq / (liq + float(bg_rb_mmscf)), 0.99)
    return float(k0 * f(sw) / f(min(swc, 0.99)))


def petex_gas(pr, t_f, k, h, area_acres, ca, rw, hp, t_days, phi, swc, skin, m_fn, mu_r, cg_r, sg_sep,
              d_entered=None, perm="total", cw=3e-6, wgr=0.0, cgr=0.0, z_r=None):
    """PROSPER's Petroleum Experts IPR for dry and wet gas:
        M(Pr) - M(Pwf) = 1422 T / (k_g h) [ (X + S) q + D q^2 ]        (q Mscf/d, M = PROSPER's gauge m(p))
    k_g = k krg(Swc) (gas permeability at connate water; with perm="gas" the entered k is k_g).
    X = min(transient, stabilised): transient 0.5 [ln(k_g t / (phi mu ct rw^2)) - 7.4316] (t hours, mu and ct at
    Pr, ct = Sg cg + Sw cw); stabilised = Dietz term 0.5 ln(4A/(gamma CA rw^2)) + 0.4932 (PROSPER's, fitted).
    D (calculated, PROSPER / Dake 8.24 with 1637): D = 3.161e-12 beta T g / (mu hp^2 rw) * k h / (1637 T),
    beta = 2.73e10 / k_g^1.1045, g the separator gas gravity, mu at Pr, k absolute.
    Validated on PROSPER Tutorial 02 IPR runs (time, area, shape factor, wellbore radius, Swc, D): see tests."""
    tr = t_f + RANKINE
    bg_rb = 0.02827 * z_r * tr / pr / 5.614583 * 1e6 if z_r is not None else None
    liq_ratio = petex_krg_liquid(swc, wgr, cgr, bg_rb) / petex_krg(swc)
    krg = (petex_krg(swc) if perm == "total" else 1.0) * liq_ratio
    k_abs = k if perm == "total" else k / max(petex_krg(swc), 1e-6)
    kg = k_abs * krg if perm == "total" else k * liq_ratio
    dietz = dietz_term(area_acres, ca, rw)
    x_pss = dietz + PETEX_PSS_OFFSET
    ct = (1 - swc) * cg_r + swc * cw
    x_tr = 0.5 * (np.log(kg * t_days * 24.0 / (phi * mu_r * ct * rw * rw)) - 7.4316) if t_days > 0 else np.inf
    x = float(min(x_tr, x_pss))
    beta = 2.73e10 / kg ** 1.1045
    d_calc = 3.161e-12 * beta * tr * sg_sep / (mu_r * hp * hp * rw) * k_abs * h / (1637.0 * tr)
    d = d_calc if d_entered is None else float(d_entered)
    c = 1422.0 * tr / (kg * h)
    a, b = c * (x + skin), c * d
    mr = float(m_fn(pr))

    def f(pwf):
        dm = np.maximum(mr - m_fn(np.clip(pwf, 0, None)), 0.0)
        return dm / a if b <= 1e-30 else (-a + np.sqrt(a * a + 4 * b * dm)) / (2 * b)
    curve = _curve_from_rate("Petroleum Experts", "gas", pr, f, {
        "Gas relative permeability": krg, "Gas permeability (md)": kg, "Drainage term X": x,
        "Flow regime": "transient" if x_tr < x_pss else "stabilised", "Non-Darcy flow factor D (1/(Mscf/d))": d,
        "a (laminar)": a, "b (turbulent)": b, "M(Pr) (psi^2/cp)": mr})
    curve.notes.append("M(P) as PROSPER: 2 int (p - 14.696)/(mu Z) dp at reservoir temperature.")
    return curve
