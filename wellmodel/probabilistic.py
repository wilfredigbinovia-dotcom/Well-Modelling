"""Probabilistic nodal analysis.

Uncertain inputs are sampled by Latin hypercube (or plain Monte Carlo) and
all samples are solved in one vectorised nodal pass. Model-parameter
uncertainty from a multi-test match (gravity/friction multipliers and PI,
with their correlations) can be propagated jointly via the match covariance.

Petroleum convention: P90 is the low case (90 % probability of exceeding),
P10 the high case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

from .matching import MatchResult
from .model import WellModel

VARIABLES = ("reservoir_pressure", "whp", "water_cut", "gor", "pi",
             "gravity_mult", "friction_mult")


# ------------------------------------------------------------ distributions
class Dist:
    def ppf(self, u):
        raise NotImplementedError

    def p50(self):
        return float(self.ppf(np.array([0.5]))[0])


@dataclass
class Fixed(Dist):
    value: float

    def ppf(self, u):
        return np.full(np.shape(u), self.value, float)


@dataclass
class Uniform(Dist):
    low: float
    high: float

    def ppf(self, u):
        return self.low + (self.high - self.low) * np.asarray(u)


@dataclass
class Triangular(Dist):
    low: float
    mode: float
    high: float

    def ppf(self, u):
        c = (self.mode - self.low) / (self.high - self.low)
        return stats.triang(c, loc=self.low, scale=self.high - self.low).ppf(u)


@dataclass
class Normal(Dist):
    mean: float
    sd: float
    low: float = -np.inf
    high: float = np.inf

    def ppf(self, u):
        a, b = (self.low - self.mean) / self.sd, (self.high - self.mean) / self.sd
        return stats.truncnorm(a, b, loc=self.mean, scale=self.sd).ppf(u)


@dataclass
class LogNormal(Dist):
    """Defined by its P90 (low) and P10 (high) values."""
    p90: float
    p10: float

    def ppf(self, u):
        z = stats.norm.ppf(0.9)
        mu = 0.5 * (np.log(self.p90) + np.log(self.p10))
        s = (np.log(self.p10) - np.log(self.p90)) / (2 * z)
        return np.exp(mu + s * stats.norm.ppf(u))


def _as_dist(v):
    return v if isinstance(v, Dist) else Fixed(float(v))


# ------------------------------------------------------------------ result
@dataclass
class MCResult:
    samples: pd.DataFrame        # inputs + rate, pwf, flowing, stable
    summary: pd.DataFrame
    sensitivity: pd.DataFrame    # Spearman rank correlation with rate
    tornado: pd.DataFrame | None

    @property
    def p90(self):
        return float(self.summary.loc["rate", "P90"])

    @property
    def p50(self):
        return float(self.summary.loc["rate", "P50"])

    @property
    def p10(self):
        return float(self.summary.loc["rate", "P10"])


def _summarise(s: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for col in ("rate", "oil_rate", "pwf"):
        v = s[col].dropna().to_numpy()
        if v.size == 0:
            continue
        rows[col] = {"P90": np.percentile(v, 10), "P50": np.percentile(v, 50),
                     "P10": np.percentile(v, 90), "mean": v.mean()}
    out = pd.DataFrame(rows).T
    out.attrs["prob_flowing"] = float(s["flowing"].mean())
    out.attrs["prob_unstable"] = float((s["flowing"] & ~s["stable"]).mean())
    return out


def monte_carlo(model: WellModel, inputs: dict, n: int = 1000, seed: int = 0,
                method: str = "lhs", match: MatchResult | None = None,
                tornado: bool = True) -> MCResult:
    """Run `n` probabilistic nodal cases.

    inputs : {variable: Dist | float}; variables are reservoir_pressure, whp,
        water_cut, gor, pi, gravity_mult, friction_mult. Anything not given
        uses the model's value (PI, multipliers) or must be supplied.
    match  : a MatchResult whose covariance is used to sample the matched
        parameters jointly (overrides any distributions given for them).
    """
    for k in inputs:
        if k not in VARIABLES:
            raise ValueError(f"unknown variable {k}")
    dists = {k: _as_dist(v) for k, v in inputs.items()}
    for k in ("pi", "gravity_mult", "friction_mult"):
        dists.setdefault(k, Fixed(getattr(model, k)))
    missing = [k for k in ("reservoir_pressure", "whp", "water_cut", "gor") if k not in dists]
    if missing:
        raise ValueError(f"inputs missing: {missing}")
    joint = list(match.names) if match is not None else []
    indep = [k for k in VARIABLES if k not in joint]

    d = len(indep) + len(joint)
    if method == "lhs":
        U = qmc.LatinHypercube(d=d, seed=seed).random(n)
    else:
        U = np.random.default_rng(seed).random((n, d))
    U = np.clip(U, 1e-6, 1 - 1e-6)
    data = {k: dists[k].ppf(U[:, i]) for i, k in enumerate(indep)}
    if joint:
        mean = np.array([np.log(match.params[k]) if k == "pi" else match.params[k] for k in joint])
        L = np.linalg.cholesky(match.covariance + 1e-12 * np.eye(len(joint)))
        Z = stats.norm.ppf(U[:, len(indep):]) @ L.T + mean
        for j, k in enumerate(joint):
            data[k] = np.exp(Z[:, j]) if k == "pi" else Z[:, j]
    s = pd.DataFrame(data)
    s["water_cut"] = s["water_cut"].clip(0, 0.999)

    op = model.operating_point(s["reservoir_pressure"], s["whp"], s["water_cut"], s["gor"],
                               pi=s["pi"], gravity_mult=s["gravity_mult"],
                               friction_mult=s["friction_mult"])
    s["rate"] = op.rate
    s["oil_rate"] = op.rate * (1 - s["water_cut"])
    s["pwf"] = op.pwf
    s["flowing"] = op.flowing
    s["stable"] = op.stable

    varying = [k for k in VARIABLES if s[k].std() > 1e-9 * max(1.0, abs(s[k].mean()))]
    sens = pd.DataFrame({
        "variable": varying,
        "spearman_rho": [stats.spearmanr(s[k], s["rate"]).statistic for k in varying],
    })
    sens["abs"] = sens["spearman_rho"].abs()
    sens = sens.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)

    tor = None
    if tornado and varying:
        base = {k: float(np.median(s[k])) for k in VARIABLES}
        cases = []
        for k in varying:
            lo, hi = np.percentile(s[k], [10, 90])
            for tag, v in (("low", lo), ("high", hi)):
                c = dict(base); c[k] = v; c["_var"] = k; c["_tag"] = tag
                cases.append(c)
        c = dict(base); c["_var"] = "base"; c["_tag"] = "base"
        cases.append(c)
        cf = pd.DataFrame(cases)
        o = model.operating_point(cf["reservoir_pressure"], cf["whp"], cf["water_cut"], cf["gor"],
                                  pi=cf["pi"], gravity_mult=cf["gravity_mult"],
                                  friction_mult=cf["friction_mult"])
        cf["rate"] = o.rate
        base_rate = float(cf.loc[cf["_var"] == "base", "rate"].iloc[0])
        rows = []
        for k in varying:
            sub = cf[cf["_var"] == k].set_index("_tag")
            rows.append({"variable": k,
                         "input_low": sub.at["low", k], "input_high": sub.at["high", k],
                         "rate_at_low": sub.at["low", "rate"], "rate_at_high": sub.at["high", "rate"],
                         "swing": abs(sub.at["high", "rate"] - sub.at["low", "rate"])})
        tor = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
        tor.attrs["base_rate"] = base_rate
    return MCResult(s, _summarise(s), sens, tor)
