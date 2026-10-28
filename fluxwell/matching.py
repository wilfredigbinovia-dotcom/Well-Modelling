"""Automated multi-test matching.

All tests are matched simultaneously by robust weighted least squares on two
residuals per test:

* gauge residual  - computed vs measured downhole gauge pressure, from a
  top-down traverse starting at the measured WHP and rate (tests the VLP);
* rate residual   - IPR rate at the traverse Pwf vs measured liquid rate
  (tests the IPR and ties it to the VLP).

Fitted parameters are any of ``gravity_mult``, ``friction_mult`` (PROSPER's
parameters 1 and 2) and ``pi``. PI is fitted in log space. The Jacobian is
built with one vectorised traverse per evaluation, the loss is soft-L1 so a
residual outlier cannot drag the whole match, and parameter standard errors
and correlations come from the Gauss-Newton covariance at the solution.

Test table columns
------------------
liquid_rate (STB/d), water_cut (fraction), gor (scf/STB), whp (psia),
gauge_pressure (psia), reservoir_pressure (psia); optional test_id, date,
weight.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .model import WellModel

REQUIRED = ["liquid_rate", "water_cut", "gor", "whp", "gauge_pressure", "reservoir_pressure"]
FIT_BOUNDS = {"gravity_mult": (0.6, 1.4), "friction_mult": (0.2, 3.0), "pi": (1e-3, 1e3)}
ACCEPTABLE = {"gravity_mult": (0.90, 1.10), "friction_mult": (0.75, 1.25)}


@dataclass
class MatchResult:
    correlation: str
    params: dict
    std_errors: dict
    correlation_matrix: pd.DataFrame
    residuals: pd.DataFrame
    rmse_gauge: float
    mape_rate: float
    flags: list = field(default_factory=list)
    success: bool = True
    n_tests_used: int = 0
    covariance: np.ndarray | None = None
    names: tuple = ()

    def model(self, base: WellModel) -> WellModel:
        """Return a copy of `base` with the matched correlation and parameters."""
        return base.copy(correlation=self.correlation, **self.params)

    def summary(self) -> pd.DataFrame:
        rows = []
        for k, v in self.params.items():
            se = self.std_errors.get(k, np.nan)
            lo, hi = ACCEPTABLE.get(k, (None, None))
            ok = "" if lo is None else ("yes" if lo <= v <= hi else "NO")
            rows.append({"parameter": k, "value": round(v, 4),
                         "std_error": None if np.isnan(se) else round(se, 4),
                         "acceptable_range": "" if lo is None else f"{lo}-{hi}",
                         "within_range": ok})
        return pd.DataFrame(rows)


def _check_tests(tests: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in tests.columns]
    if missing:
        raise ValueError(f"test table missing columns: {missing}")


def _transform(names, values):
    return np.array([np.log(v) if n == "pi" else v for n, v in zip(names, values)], float)


def _untransform(names, x):
    return {n: (float(np.exp(v)) if n == "pi" else float(v)) for n, v in zip(names, x)}


class _Problem:
    def __init__(self, model, tests, names, weights, sigma_p, sigma_q_frac, correlation):
        self.m = model
        self.names = tuple(names)
        self.corr = correlation
        self.q = tests["liquid_rate"].to_numpy(float)
        self.wc = tests["water_cut"].to_numpy(float)
        self.gor = tests["gor"].to_numpy(float)
        self.whp = tests["whp"].to_numpy(float)
        self.pg = tests["gauge_pressure"].to_numpy(float)
        self.pr = tests["reservoir_pressure"].to_numpy(float)
        self.sw = np.sqrt(np.asarray(weights, float))
        self.sp = sigma_p
        self.sq = sigma_q_frac
        self.gi = model.wellbore.gauge_index
        self.n_eval = 0

    def params_of(self, X):
        base = {"gravity_mult": self.m.gravity_mult, "friction_mult": self.m.friction_mult,
                "pi": self.m.pi}
        K = X.shape[0]
        out = {k: np.full(K, v, float) for k, v in base.items()}
        for j, n in enumerate(self.names):
            out[n] = np.exp(X[:, j]) if n == "pi" else X[:, j]
        return out

    def batch(self, X, raw=False):
        """Residuals for K parameter vectors in one traverse -> (K, 2T)."""
        X = np.atleast_2d(X)
        K, T = X.shape[0], self.q.size
        P = self.params_of(X)
        tile = lambda a: np.tile(a, K)
        prof = self.m.traverse(tile(self.q), tile(self.wc), tile(self.gor), tile(self.whp),
                               gravity_mult=np.repeat(P["gravity_mult"], T),
                               friction_mult=np.repeat(P["friction_mult"], T),
                               correlation=self.corr)
        self.n_eval += K
        pg = prof[:, self.gi].reshape(K, T)
        pwf = prof[:, -1].reshape(K, T)
        q_ipr = self.m.ipr_rate(pwf, self.pr[None, :], P["pi"][:, None])
        rg = (pg - self.pg[None, :]) / self.sp
        rq = (q_ipr - self.q[None, :]) / (self.sq * self.q[None, :])
        if raw:
            return pg, pwf, q_ipr
        return np.hstack([rg * self.sw, rq * self.sw])

    def fun(self, x):
        return self.batch(x[None, :])[0]

    def jac(self, x):
        h = 1e-4 * np.maximum(1.0, np.abs(x))
        X = np.vstack([x] + [x + h[j] * np.eye(x.size)[j] for j in range(x.size)])
        R = self.batch(X)
        return ((R[1:] - R[0]) / h[:, None]).T


def match_tests(model: WellModel, tests: pd.DataFrame,
                fit=("gravity_mult", "friction_mult", "pi"),
                weights=None, correlation: str | None = None,
                sigma_p: float = 30.0, sigma_q_frac: float = 0.03,
                loss: str = "soft_l1", x0: dict | None = None,
                compute_nodal: bool = True) -> MatchResult:
    """Match one correlation to many well tests simultaneously.

    weights : per-test weights (e.g. from QC). Zero-weight tests are dropped
        from the regression but still reported.
    sigma_p : expected gauge pressure error (psi) used to normalise residuals.
        This is the *total* error, dominated by input errors (a 3 % GOR
        metering error alone moves a gauge ~40 psi), not the gauge accuracy.
    sigma_q_frac : expected fractional rate error.
    """
    _check_tests(tests)
    tests = tests.reset_index(drop=True)
    corr = correlation or model.correlation
    w = (tests["weight"].to_numpy(float) if weights is None and "weight" in tests
         else np.ones(len(tests)) if weights is None else np.asarray(weights, float))
    use = w > 0
    if use.sum() < max(2, len(fit)):
        raise ValueError("not enough weighted tests to match")
    tu = tests[use]
    prob = _Problem(model, tu, fit, w[use], sigma_p, sigma_q_frac, corr)

    start = {"gravity_mult": model.gravity_mult, "friction_mult": model.friction_mult,
             "pi": model.pi}
    if "pi" in fit and x0 is None:
        # data-driven PI start: q / (Pr - gauge) is within ~10 % of the answer
        dd = np.maximum(tu["reservoir_pressure"] - tu["gauge_pressure"], 50.0)
        start["pi"] = float(np.median(tu["liquid_rate"] / dd))
    if x0:
        start.update(x0)
    xs = _transform(fit, [start[n] for n in fit])
    lb = _transform(fit, [FIT_BOUNDS[n][0] for n in fit])
    ub = _transform(fit, [FIT_BOUNDS[n][1] for n in fit])
    xs = np.clip(xs, lb + 1e-9, ub - 1e-9)
    sol = least_squares(prob.fun, xs, jac=prob.jac, bounds=(lb, ub), loss=loss,
                        f_scale=1.0, x_scale="jac", max_nfev=200)
    params = _untransform(fit, sol.x)

    # covariance (Gauss-Newton, plain residuals)
    J = prob.jac(sol.x)
    r = prob.fun(sol.x)
    dof = max(r.size - sol.x.size, 1)
    s2 = float(r @ r) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov = np.full((sol.x.size,) * 2, np.nan)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    std_err = {n: float(params[n] * sd[i]) if n == "pi" else float(sd[i]) for i, n in enumerate(fit)}
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = cov / np.outer(sd, sd)
    corr_df = pd.DataFrame(cm, index=list(fit), columns=list(fit))

    full = model.copy(correlation=corr, **params)
    res = residual_table(full, tests, w, compute_nodal=compute_nodal)
    used = res[res["weight"] > 0]
    rmse = float(np.sqrt(np.mean(used["gauge_error"] ** 2)))
    mape = float(np.mean(np.abs(used["rate_error_pct"])))

    flags = []
    for n, (lo, hi) in ACCEPTABLE.items():
        if n in params and not lo <= params[n] <= hi:
            flags.append(f"{n} = {params[n]:.3f} outside {lo}-{hi}: the correlation is being "
                         f"forced to fit; consider another correlation or check the data")
    if {"gravity_mult", "friction_mult"} <= set(fit):
        rho = corr_df.loc["gravity_mult", "friction_mult"]
        if np.isfinite(rho) and abs(rho) > 0.95:
            flags.append(f"gravity and friction multipliers are strongly correlated (r = {rho:.2f}); "
                         "tests do not span enough rate range to separate them - fix friction_mult")
    if "pi" in params and std_err["pi"] / params["pi"] > 0.15:
        flags.append("PI uncertainty above 15 %: check reservoir pressures")
    out = used[np.abs(used["gauge_std_resid"]) > 3]
    if len(out):
        flags.append(f"tests with gauge residual > 3 sigma after match: {list(out['test_id'])}")
    if not sol.success:
        flags.append(f"optimiser did not converge: {sol.message}")
    return MatchResult(corr, params, std_err, corr_df, res, rmse, mape, flags,
                       bool(sol.success), int(use.sum()), cov, tuple(fit))


def residual_table(model: WellModel, tests: pd.DataFrame, weights=None,
                   sigma_p: float = 30.0, compute_nodal: bool = True) -> pd.DataFrame:
    """Per-test comparison of a model against measurements."""
    tests = tests.reset_index(drop=True)
    w = np.ones(len(tests)) if weights is None else np.asarray(weights, float)
    prof = model.traverse(tests["liquid_rate"], tests["water_cut"], tests["gor"], tests["whp"])
    pg = prof[:, model.wellbore.gauge_index]
    pwf = prof[:, -1]
    q_ipr = model.ipr_rate(pwf, tests["reservoir_pressure"].to_numpy(float))
    df = pd.DataFrame({
        "test_id": tests["test_id"] if "test_id" in tests else np.arange(len(tests)),
        "weight": w,
        "gauge_meas": tests["gauge_pressure"].to_numpy(float),
        "gauge_calc": pg,
    })
    df["gauge_error"] = df["gauge_calc"] - df["gauge_meas"]
    df["gauge_std_resid"] = df["gauge_error"] / sigma_p
    df["pwf_calc"] = pwf
    df["rate_meas"] = tests["liquid_rate"].to_numpy(float)
    df["rate_ipr"] = q_ipr
    df["rate_error_pct"] = 100 * (q_ipr - df["rate_meas"]) / df["rate_meas"]
    if compute_nodal:
        op = model.operating_point(tests["reservoir_pressure"], tests["whp"],
                                   tests["water_cut"], tests["gor"])
        df["rate_nodal"] = op.rate
        df["nodal_error_pct"] = 100 * (op.rate - df["rate_meas"]) / df["rate_meas"]
    return df
