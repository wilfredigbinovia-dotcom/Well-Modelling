"""Data-driven VLP correlation selection.

For every candidate correlation three numbers are computed against the
QC-weighted tests:

1. raw error     - gauge RMSE with multipliers fixed at 1 (only PI fitted):
                   how good the correlation is "out of the box";
2. matched error - gauge RMSE and the multipliers needed after a full match;
3. LOOCV error   - leave-one-out cross-validation: match on n-1 tests,
                   predict the held-out test's gauge pressure and nodal rate.
                   This is the honest measure of predictive skill, because a
                   correlation with free multipliers can fit almost anything.

Ranking: correlations whose matched multipliers all sit inside the acceptable
ranges are ranked first, by score = LOOCV gauge RMSE x (1 + penalty); those
needing out-of-range multipliers follow (penalty grows with the distance).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .correlations import CORRELATIONS, LABELS
from .matching import ACCEPTABLE, MatchResult, match_tests
from .model import WellModel


@dataclass
class SelectionResult:
    table: pd.DataFrame
    recommended: str
    rationale: str
    matches: dict            # correlation -> MatchResult (full match)
    loocv: dict              # correlation -> DataFrame of held-out predictions


def _penalty(params):
    pen = 0.0
    for n, (lo, hi) in ACCEPTABLE.items():
        v = params.get(n)
        if v is None:
            continue
        width = hi - lo
        if v < lo:
            pen += (lo - v) / width * 2
        elif v > hi:
            pen += (v - hi) / width * 2
    return pen


def compare_correlations(model: WellModel, tests: pd.DataFrame, weights=None,
                         correlations=None, loocv: bool = True,
                         fit=("gravity_mult", "friction_mult", "pi"),
                         verbose: bool = False) -> SelectionResult:
    tests = tests.reset_index(drop=True)
    w = np.ones(len(tests)) if weights is None else np.asarray(weights, float)
    # the unified mechanistic model is ~10x slower; ask for it explicitly in this all-against-all screen
    cands = list(correlations or [c for c in CORRELATIONS if c != "tuffp_unified"])
    rows, matches, cv = [], {}, {}
    for c in cands:
        raw = match_tests(model.copy(gravity_mult=1.0, friction_mult=1.0), tests,
                          fit=("pi",), weights=w, correlation=c, compute_nodal=False)
        used = raw.residuals[raw.residuals["weight"] > 0]
        full = match_tests(model, tests, fit=fit, weights=w, correlation=c)
        matches[c] = full
        row = {"correlation": c, "label": LABELS[c],
               "raw_gauge_rmse": raw.rmse_gauge,
               "raw_gauge_bias": float(used["gauge_error"].mean()),
               "matched_gauge_rmse": full.rmse_gauge,
               "matched_rate_mape": full.mape_rate,
               **{k: full.params.get(k) for k in fit}}
        if loocv:
            idx = np.where(w > 0)[0]
            recs = []
            for k in idx:
                wk = w.copy(); wk[k] = 0.0
                mk = match_tests(model, tests, fit=fit, weights=wk, correlation=c,
                                 x0=full.params, compute_nodal=False)
                fm = mk.model(model)
                r = tests.iloc[[k]]
                pg = fm.gauge_pressure(r["liquid_rate"], r["water_cut"], r["gor"], r["whp"])[0]
                op = fm.operating_point(r["reservoir_pressure"], r["whp"], r["water_cut"], r["gor"])
                recs.append({"test_id": tests.at[k, "test_id"] if "test_id" in tests else k,
                             "gauge_meas": r["gauge_pressure"].iloc[0], "gauge_pred": pg,
                             "rate_meas": r["liquid_rate"].iloc[0], "rate_pred": op.rate[0]})
            d = pd.DataFrame(recs)
            d["gauge_error"] = d["gauge_pred"] - d["gauge_meas"]
            d["rate_error_pct"] = 100 * (d["rate_pred"] - d["rate_meas"]) / d["rate_meas"]
            cv[c] = d
            row["loocv_gauge_rmse"] = float(np.sqrt(np.mean(d["gauge_error"] ** 2)))
            row["loocv_rate_mape"] = float(np.mean(np.abs(d["rate_error_pct"])))
        base = row.get("loocv_gauge_rmse", row["matched_gauge_rmse"])
        row["multiplier_penalty"] = _penalty(full.params)
        row["score"] = base * (1 + row["multiplier_penalty"])
        rows.append(row)
        if verbose:
            print(f"{c}: done")
    table = pd.DataFrame(rows)
    # correlations needing out-of-range multipliers rank behind every in-range one
    table["in_range"] = table["multiplier_penalty"] == 0
    table = table.sort_values(["in_range", "score"], ascending=[False, True]).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    best = table.iloc[0]
    rationale = _rationale(table, loocv)
    return SelectionResult(table, best["correlation"], rationale, matches, cv)


def _rationale(table, loocv):
    b = table.iloc[0]
    metric = "leave-one-out gauge RMSE" if loocv else "matched gauge RMSE"
    val = b["loocv_gauge_rmse"] if loocv else b["matched_gauge_rmse"]
    lines = [f"Recommended: {b['label']} - lowest {metric} ({val:.0f} psi) "
             f"with gravity multiplier {b['gravity_mult']:.3f} and friction multiplier "
             f"{b['friction_mult']:.3f}."]
    if b["multiplier_penalty"] > 0:
        lines.append("Caution: even the best correlation needs multipliers outside the "
                     "acceptable range - check the data or consider a mechanistic model.")
    else:
        lines.append(f"Its out-of-the-box bias is {b['raw_gauge_bias']:+.0f} psi at the gauge, "
                     "and the match stays within the acceptable multiplier ranges.")
    for _, r in table.iloc[1:].iterrows():
        why = []
        if r["multiplier_penalty"] > 0:
            why.append("needs out-of-range multipliers")
        v = r["loocv_gauge_rmse"] if loocv else r["matched_gauge_rmse"]
        why.append(f"{metric} {v:.0f} psi")
        lines.append(f"{r['label']}: rejected - " + ", ".join(why) + ".")
    return "\n".join(lines)
