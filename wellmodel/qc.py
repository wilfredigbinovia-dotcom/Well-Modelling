"""Well test data QC.

Each test is run through a set of physical and statistical checks and
classified ``pass`` (weight 1), ``warn`` (weight 0.5) or ``fail`` (weight 0,
excluded from matching). Every flag carries a plain-language reason.

Checks
------
range        missing values, rate <= 0, water cut outside [0, 1), WHP below
             atmospheric, gauge above reservoir pressure, WHP above gauge
gor          producing GOR below solution GOR while Pr > Pb (gas metering
             error), or far above Rsb while undersaturated (free gas source?)
vlp_envelope measured gauge pressure outside the band spanned by every
             correlation with gravity multiplier 0.85-1.15 (+ tolerance):
             no reasonable tubing model can reproduce it - rate, WHP or gauge
             is wrong
pi_outlier   PI implied by each test (gauge corrected to perfs) compared with
             a robust Theil-Sen time trend; robust z-score from the MAD
stability    short test duration or unsteady rate (if columns supplied)
wc_jump      water-cut jump between consecutive tests (sampling issue)
duplicate    repeated test dates
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .correlations import CORRELATIONS
from .model import WellModel

STATUS_WEIGHT = {"pass": 1.0, "warn": 0.5, "fail": 0.0}


@dataclass
class QCConfig:
    gor_low_frac: float = 0.85         # GOR < 0.85 Rsb while Pr > Pb -> fail
    gor_high_frac: float = 1.6         # GOR > 1.6 Rsb while Pr > Pb -> warn
    envelope_gravity: tuple = (0.85, 1.15)
    envelope_tol_psi: float = 50.0
    pi_z_warn: float = 2.5
    pi_z_fail: float = 3.5
    pi_mad_floor_frac: float = 0.02    # MAD floor as fraction of median PI
    min_duration_hr: float = 6.0
    max_rate_cv: float = 0.10
    wc_jump: float = 0.15


@dataclass
class QCReport:
    table: pd.DataFrame          # per test: status, weight, reasons, diagnostics
    flags: pd.DataFrame          # long format: test_id, check, severity, message

    @property
    def weights(self) -> np.ndarray:
        return self.table["weight"].to_numpy(float)

    def counts(self) -> dict:
        return self.table["status"].value_counts().to_dict()


def _flag(store, tid, check, sev, msg):
    store.append({"test_id": tid, "check": check, "severity": sev, "message": msg})


def qc_tests(model: WellModel, tests: pd.DataFrame, config: QCConfig | None = None) -> QCReport:
    cfg = config or QCConfig()
    t = tests.reset_index(drop=True).copy()
    if "test_id" not in t:
        t["test_id"] = np.arange(len(t))
    ids = t["test_id"].tolist()
    flags: list = []
    pvt = model.pvt

    # ---------------------------------------------------------------- range
    cols = ["liquid_rate", "water_cut", "gor", "whp", "gauge_pressure", "reservoir_pressure"]
    bad_range = np.zeros(len(t), bool)
    for i, r in t.iterrows():
        tid = r["test_id"]
        if r[cols].isna().any():
            _flag(flags, tid, "range", "fail", "missing values"); bad_range[i] = True; continue
        if r["liquid_rate"] <= 0:
            _flag(flags, tid, "range", "fail", "non-positive liquid rate"); bad_range[i] = True
        if not 0 <= r["water_cut"] < 1:
            _flag(flags, tid, "range", "fail", "water cut outside 0-1"); bad_range[i] = True
        if r["whp"] < 14.7:
            _flag(flags, tid, "range", "fail", "WHP below atmospheric"); bad_range[i] = True
        if r["gauge_pressure"] >= r["reservoir_pressure"]:
            _flag(flags, tid, "range", "fail",
                  "gauge pressure at or above reservoir pressure - no drawdown possible"); bad_range[i] = True
        if r["whp"] >= r["gauge_pressure"]:
            _flag(flags, tid, "range", "fail", "WHP above downhole gauge pressure"); bad_range[i] = True

    # ------------------------------------------------------------------ GOR
    for i, r in t.iterrows():
        if bad_range[i]:
            continue
        if r["reservoir_pressure"] > pvt.pb:
            if r["gor"] < cfg.gor_low_frac * pvt.rsb:
                _flag(flags, r["test_id"], "gor", "fail",
                      f"GOR {r['gor']:.0f} below Rsb {pvt.rsb:.0f} while Pr > Pb: gas under-metered")
            elif r["gor"] > cfg.gor_high_frac * pvt.rsb:
                _flag(flags, r["test_id"], "gor", "warn",
                      f"GOR {r['gor']:.0f} well above Rsb while Pr > Pb: free-gas source or metering error")

    ok = ~bad_range
    # --------------------------------------------------------- VLP envelope
    env_lo = np.full(len(t), np.nan)
    env_hi = np.full(len(t), np.nan)
    if ok.any():
        tt = t[ok]
        preds = []
        for c in CORRELATIONS:
            for gm in cfg.envelope_gravity:
                preds.append(model.gauge_pressure(tt["liquid_rate"], tt["water_cut"], tt["gor"],
                                                  tt["whp"], gravity_mult=gm, friction_mult=1.0,
                                                  correlation=c))
        preds = np.vstack(preds)
        env_lo[ok] = preds.min(0) - cfg.envelope_tol_psi
        env_hi[ok] = preds.max(0) + cfg.envelope_tol_psi
    for i, r in t.iterrows():
        if ok[i] and not env_lo[i] <= r["gauge_pressure"] <= env_hi[i]:
            _flag(flags, r["test_id"], "vlp_envelope", "fail",
                  f"gauge {r['gauge_pressure']:.0f} psia outside physically reachable band "
                  f"{env_lo[i]:.0f}-{env_hi[i]:.0f}: rate, WHP or gauge reading is wrong")

    # ---------------------------------------------------------- implied PI
    pi_impl = np.full(len(t), np.nan)
    if ok.any():
        tt = t[ok]
        prof = model.traverse(tt["liquid_rate"], tt["water_cut"], tt["gor"], tt["whp"])
        dp_below_gauge = prof[:, -1] - prof[:, model.wellbore.gauge_index]
        pwf_impl = tt["gauge_pressure"].to_numpy(float) + dp_below_gauge
        unit = model.ipr_rate(pwf_impl, tt["reservoir_pressure"].to_numpy(float), pi=1.0)
        pi_impl[ok] = tt["liquid_rate"].to_numpy(float) / np.maximum(unit, 1e-9)
    t["pi_implied"] = pi_impl
    failed_so_far = {f["test_id"] for f in flags if f["severity"] == "fail"}
    cand = np.array([ok[i] and t.at[i, "test_id"] not in failed_so_far for i in range(len(t))])
    z = np.full(len(t), np.nan)
    if cand.sum() >= 5:
        x = (pd.to_datetime(t["date"]).map(pd.Timestamp.toordinal).to_numpy(float)
             if "date" in t else np.arange(len(t), dtype=float))
        y = pi_impl
        slope, icpt, _, _ = theilslopes(y[cand], x[cand])
        resid = y - (icpt + slope * x)
        mad = 1.4826 * np.median(np.abs(resid[cand] - np.median(resid[cand])))
        mad = max(mad, cfg.pi_mad_floor_frac * np.median(y[cand]))
        z = resid / mad
        for i in range(len(t)):
            if not ok[i]:
                continue
            tid = t.at[i, "test_id"]
            if abs(z[i]) > cfg.pi_z_fail:
                _flag(flags, tid, "pi_outlier", "fail",
                      f"implied PI {pi_impl[i]:.2f} is {z[i]:+.1f} robust sigma from trend: "
                      "inconsistent rate/pressure pair")
            elif abs(z[i]) > cfg.pi_z_warn:
                _flag(flags, tid, "pi_outlier", "warn",
                      f"implied PI {pi_impl[i]:.2f} is {z[i]:+.1f} robust sigma from trend")
    t["pi_z"] = z

    # ------------------------------------------------------------ stability
    if "duration_hr" in t:
        for _, r in t.iterrows():
            if pd.notna(r["duration_hr"]) and r["duration_hr"] < cfg.min_duration_hr:
                _flag(flags, r["test_id"], "stability", "warn",
                      f"test only {r['duration_hr']:.1f} h long (< {cfg.min_duration_hr} h)")
    if "rate_cv" in t:
        for _, r in t.iterrows():
            if pd.notna(r["rate_cv"]) and r["rate_cv"] > cfg.max_rate_cv:
                _flag(flags, r["test_id"], "stability", "warn",
                      f"rate unstable during test (CV {100*r['rate_cv']:.0f} %)")

    # ------------------------------------------------------ WC jump, dups
    order = t.sort_values("date").index if "date" in t else t.index
    prev = None
    for i in order:
        if prev is not None and abs(t.at[i, "water_cut"] - t.at[prev, "water_cut"]) > cfg.wc_jump:
            _flag(flags, t.at[i, "test_id"], "wc_jump", "warn",
                  f"water cut changed {t.at[prev, 'water_cut']:.2f} -> {t.at[i, 'water_cut']:.2f} "
                  "since previous test")
        prev = i
    if "date" in t:
        for tid in t.loc[t["date"].duplicated(keep="first"), "test_id"]:
            _flag(flags, tid, "duplicate", "warn", "duplicate test date")

    fl = pd.DataFrame(flags, columns=["test_id", "check", "severity", "message"])
    status = []
    reasons = []
    for tid in ids:
        f = fl[fl["test_id"] == tid]
        s = "fail" if (f["severity"] == "fail").any() else "warn" if len(f) else "pass"
        status.append(s)
        reasons.append("; ".join(f["message"]))
    t["gauge_envelope_lo"] = env_lo
    t["gauge_envelope_hi"] = env_hi
    t["status"] = status
    t["weight"] = [STATUS_WEIGHT[s] for s in status]
    t["reasons"] = reasons
    return QCReport(t, fl)
