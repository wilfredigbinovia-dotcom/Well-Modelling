"""Screening-level transient behaviour.

1. Pressure-transient response of a vertical well in a closed circular
   reservoir with wellbore storage and skin (Laplace-space solution,
   Stehfest inversion). Gives drawdown and build-up pressures, the Bourdet
   derivative for diagnostics, and a build-up fit for k, skin and storage
   that yields an independent PI estimate.

2. Production forecast: tank material balance (with optional Schilthuis
   aquifer) coupled to the nodal model at every time step, with
   time-varying water cut, GOR and early-time transient productivity.
   Reports when the reservoir drops below Pb, when flow becomes unstable
   (operating point left of the VLP minimum) and when natural flow stops
   (artificial lift needed).

These are screening models: single-layer homogeneous reservoir, no
wellbore phase redistribution, no relative-permeability-based GOR/WC
(user-controllable trend functions are used instead).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import factorial
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ive, kve

from .ipr import darcy_pi
from .model import WellModel

LN2 = np.log(2.0)


# ------------------------------------------------------------- Stehfest
def stehfest_weights(N: int = 12) -> np.ndarray:
    if N % 2:
        raise ValueError("N must be even")
    V = np.zeros(N)
    h = N // 2
    for i in range(1, N + 1):
        s = 0.0
        for k in range((i + 1) // 2, min(i, h) + 1):
            s += (k ** h * factorial(2 * k)) / (factorial(h - k) * factorial(k) * factorial(k - 1)
                                               * factorial(i - k) * factorial(2 * k - i))
        V[i - 1] = (-1) ** (i + h) * s
    return V


_V12 = stehfest_weights(12)


def invert_laplace(F: Callable, t) -> np.ndarray:
    t = np.atleast_1d(np.asarray(t, float))
    i = np.arange(1, _V12.size + 1)[:, None]
    s = i * LN2 / t[None, :]
    return LN2 / t * np.sum(_V12[:, None] * F(s), axis=0)


def pwd_laplace(s, cd: float, skin: float, red: float | None):
    """Dimensionless wellbore pressure in Laplace space: closed circular
    boundary at reD (None = infinite acting), skin and storage CD."""
    b = np.sqrt(s)
    k0k1 = kve(0, b) / kve(1, b)
    if red is None:
        pd_ = k0k1 / (s * b)
    else:
        a = red * b
        r = kve(1, a) / ive(1, a) * np.exp(np.clip(2 * b - 2 * a, -700, 0))
        num = k0k1 + r * ive(0, b) / kve(1, b)
        den = 1.0 - r * ive(1, b) / kve(1, b)
        pd_ = num / (s * b * den)
    x = s * pd_ + skin
    return x / (s * (1.0 + cd * s * x))


# --------------------------------------------------------- reservoir
@dataclass
class RadialReservoir:
    k: float            # md
    h: float            # ft
    phi: float
    ct: float           # 1/psi
    mu: float           # cp
    bo: float           # rb/STB
    rw: float = 0.354   # ft
    re: float = 1500.0  # ft (None-like large value -> effectively infinite)
    skin: float = 0.0
    C: float = 0.01     # wellbore storage, bbl/psi

    def td(self, t_hr):
        return 0.0002637 * self.k * np.asarray(t_hr, float) / (self.phi * self.mu * self.ct * self.rw ** 2)

    @property
    def cd(self):
        return 0.8936 * self.C / (self.phi * self.ct * self.h * self.rw ** 2)

    @property
    def red(self):
        return self.re / self.rw

    def dp_unit(self, q):
        """psi per unit dimensionless pressure at liquid rate q (STB/d)."""
        return 141.2 * q * self.bo * self.mu / (self.k * self.h)

    def pwd(self, t_hr):
        td = self.td(t_hr)
        return invert_laplace(lambda s: pwd_laplace(s, self.cd, self.skin, self.red), td)

    def drawdown(self, t_hr, q, p_init):
        """Flowing pressure after t hours at constant rate from p_init."""
        return p_init - self.dp_unit(q) * self.pwd(t_hr)

    def buildup(self, dt_hr, tp_hr, q, p_init):
        """Shut-in pressure dt hours after a tp-hour flow period (superposition)."""
        dt = np.asarray(dt_hr, float)
        return p_init - self.dp_unit(q) * (self.pwd(tp_hr + dt) - self.pwd(dt))

    def pss_pi(self):
        return float(darcy_pi(self.k, self.h, self.mu, self.bo, self.re, self.rw, self.skin))

    def t_pss_hr(self):
        """Start of pseudo-steady state for a centred well (tDA = 0.1)."""
        area = np.pi * self.re ** 2
        return 0.1 * self.phi * self.mu * self.ct * area / (0.0002637 * self.k)

    def transient_pi_factor(self, t_hr):
        """PI(t)/PI_pss >= 1 during infinite-acting flow (log approximation)."""
        td = np.maximum(self.td(np.maximum(t_hr, 1e-3)), 1.0)
        pd_ia = 0.5 * (np.log(td) + 0.80907)
        return np.maximum(1.0, (np.log(self.red) - 0.75 + self.skin) / (pd_ia + self.skin))


def bourdet_derivative(t, dp, L: float = 0.15):
    """Bourdet pressure derivative t*d(dp)/dt with log-window smoothing L."""
    t = np.asarray(t, float)
    dp = np.asarray(dp, float)
    x = np.log(t)
    der = np.full(t.size, np.nan)
    for i in range(t.size):
        l = np.where(x[:i] <= x[i] - L)[0]
        r = np.where(x[i + 1:] >= x[i] + L)[0]
        if l.size == 0 or r.size == 0:
            continue
        j, k = l[-1], i + 1 + r[0]
        d1, d2 = x[i] - x[j], x[k] - x[i]
        m1 = (dp[i] - dp[j]) / d1
        m2 = (dp[k] - dp[i]) / d2
        der[i] = (m1 * d2 + m2 * d1) / (d1 + d2)
    return der


@dataclass
class BuildupFit:
    reservoir: RadialReservoir
    p_init: float
    std_errors: dict
    rmse: float
    pi_pss: float


def fit_buildup(dt_hr, pws, tp_hr, q, guess: RadialReservoir, p_init_guess: float | None = None,
                fit=("k", "skin", "C", "p_init")) -> BuildupFit:
    """Fit k, skin, wellbore storage and initial pressure to build-up data.
    Rock/fluid properties (h, phi, ct, mu, Bo, rw, re) are held at `guess`."""
    dt_hr = np.asarray(dt_hr, float)
    pws = np.asarray(pws, float)
    p0 = float(np.max(pws)) + 20 if p_init_guess is None else p_init_guess
    start = {"k": np.log(guess.k), "skin": guess.skin, "C": np.log(guess.C), "p_init": p0}
    names = list(fit)

    def build(x):
        v = dict(zip(names, x))
        res = replace(guess, k=float(np.exp(v.get("k", start["k"]))),
                      skin=float(v.get("skin", start["skin"])),
                      C=float(np.exp(v.get("C", start["C"]))))
        return res, float(v.get("p_init", start["p_init"]))

    def fun(x):
        res, pi0 = build(x)
        return res.buildup(dt_hr, tp_hr, q, pi0) - pws

    x0 = np.array([start[n] for n in names])
    lb = np.array([{"k": np.log(1e-3), "skin": -6, "C": np.log(1e-5), "p_init": pws.max()}[n] for n in names])
    ub = np.array([{"k": np.log(1e5), "skin": 100, "C": np.log(10), "p_init": pws.max() + 5000}[n] for n in names])
    sol = least_squares(fun, np.clip(x0, lb + 1e-9, ub - 1e-9), bounds=(lb, ub), x_scale="jac")
    res, pi0 = build(sol.x)
    r = sol.fun
    dof = max(r.size - sol.x.size, 1)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * (r @ r) / dof
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sd = np.full(len(names), np.nan)
    se = {}
    for i, n in enumerate(names):
        if n == "k":
            se["k"] = res.k * sd[i]
        elif n == "C":
            se["C"] = res.C * sd[i]
        else:
            se[n] = sd[i]
    return BuildupFit(res, pi0, se, float(np.sqrt(np.mean(r ** 2))), res.pss_pi())


# ----------------------------------------------------------- forecast
def logistic_water_cut(wc0: float = 0.05, wc_max: float = 0.95, rf50: float = 0.12,
                       width: float = 0.03) -> Callable:
    """Water cut as a function of recovery factor (Np/N): S-curve from wc0."""
    sig = lambda rf: 1.0 / (1.0 + np.exp(-(rf - rf50) / width))
    s0 = sig(0.0)
    return lambda rf: wc0 + (wc_max - wc0) * (sig(rf) - s0) / (1 - s0)


def solution_gas_gor(beta: float = 4.0) -> Callable:
    """Producing GOR: Rsb above Pb, rising linearly with (Pb - Pr)/Pb below."""
    return lambda pr, pvt: pvt.rsb * (1.0 + beta * np.maximum(pvt.pb - pr, 0.0) / pvt.pb)


@dataclass
class DepletionInputs:
    stoiip: float                       # STB in the well's drainage volume
    p_init: float                       # psia
    swi: float = 0.25
    cw: float = 3e-6
    cf: float = 4e-6
    aquifer_pi: float = 0.0             # rb/d/psi (Schilthuis steady-state)
    water_cut: Callable = field(default_factory=logistic_water_cut)
    gor: Callable = field(default_factory=solution_gas_gor)
    whp: float | Callable = 250.0       # psia, or f(t_days)
    max_liquid_rate: float | None = None


@dataclass
class ForecastResult:
    table: pd.DataFrame
    events: dict


def _time_steps(days, step):
    early = [0.5, 1, 2, 4, 8, 15]
    t = [x for x in early if x < step]
    t += list(np.arange(step, days + 1e-9, step))
    return np.array(sorted(set([0.0] + t)))


def forecast(model: WellModel, inputs: DepletionInputs, days: float = 3650,
             step_days: float = 30, reservoir: RadialReservoir | None = None) -> ForecastResult:
    pvt = model.pvt
    so = 1 - inputs.swi
    _, boi, _, _ = pvt.oil(inputs.p_init, pvt.t_res)
    vp = inputs.stoiip * float(boi) / so                     # rb pore volume
    times = _time_steps(days, step_days)
    pr, np_, wp, gp, we = inputs.p_init, 0.0, 0.0, 0.0, 0.0
    rows = []
    events = {"pss_reached_days": None if reservoir is None else reservoir.t_pss_hr() / 24.0,
              "pr_below_pb_days": None, "unstable_from_days": None,
              "flow_stops_days": None, "np_at_flow_stop": None}
    for i, t in enumerate(times[:-1]):
        dt = times[i + 1] - t
        tm = t + 0.5 * dt
        rf = np_ / inputs.stoiip
        wc = float(np.clip(inputs.water_cut(rf), 0, 0.99))
        gor = float(inputs.gor(pr, pvt))
        whp = inputs.whp(tm) if callable(inputs.whp) else inputs.whp
        fac = 1.0 if reservoir is None else float(reservoir.transient_pi_factor(tm * 24.0))
        pi_eff = model.pi * fac
        op = model.operating_point(pr, whp, wc, gor, pi=pi_eff, n_grid=24)
        q = float(op.rate[0])
        choked = inputs.max_liquid_rate is not None and q > inputs.max_liquid_rate
        if choked:
            q = inputs.max_liquid_rate
        flowing, stable = bool(op.flowing[0]), bool(op.stable[0])
        qo, qw = q * (1 - wc), q * wc
        rs, bo, _, _ = pvt.oil(pr, pvt.t_res)
        _, bg, _, _ = pvt.gas(pr, pvt.t_res)
        bw, _, _ = pvt.water(pr, pvt.t_res)
        free = qo * max(gor - float(rs), 0.0) * float(bg) / 5.615
        q_aq = inputs.aquifer_pi * (inputs.p_init - pr)
        voidage = qo * float(bo) + qw * float(bw) + free - q_aq
        ct = so * float(pvt.total_compressibility_oil(pr)) + inputs.swi * inputs.cw + inputs.cf
        rows.append({"t_days": tm, "reservoir_pressure": pr, "liquid_rate": q, "oil_rate": qo,
                     "water_rate": qw, "gas_rate_mscfd": qo * gor / 1000, "water_cut": wc,
                     "gor": gor, "whp": whp, "pwf": float(op.pwf[0]) if flowing else np.nan,
                     "pi_eff": pi_eff, "flowing": flowing, "stable": stable, "choked": choked,
                     "Np": np_, "Wp": wp, "recovery_factor": rf, "aquifer_influx_rbd": q_aq})
        if events["pr_below_pb_days"] is None and pr < pvt.pb:
            events["pr_below_pb_days"] = t
        if events["unstable_from_days"] is None and flowing and not stable:
            events["unstable_from_days"] = t
        if not flowing:
            events["flow_stops_days"] = t
            events["np_at_flow_stop"] = np_
            break
        np_ += qo * dt
        wp += qw * dt
        gp += qo * gor * dt
        we += q_aq * dt
        pr = max(pr - voidage * dt / (vp * ct), 14.7)
    df = pd.DataFrame(rows)
    return ForecastResult(df, events)
