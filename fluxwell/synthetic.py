"""Synthetic data with a known answer, for validating the workflow.

`demo_well()` builds a deviated oil producer whose true behaviour is the
modified Hagedorn & Brown correlation with gravity multiplier 1.04 and
friction multiplier 1.15. `make_tests()` generates noisy well tests from that
truth over three years of depletion and water-cut rise, and plants known
bad tests so the QC can be checked:

* gauge reading 200 psi high (gauge drift / transcription)
* liquid rate 30 % high (misallocation)
* GOR 40 % low while undersaturated (gas meter fault)
* a 3-hour test (stability warning)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ipr import darcy_pi
from .model import WellModel
from .pvt import BlackOilPVT
from .wellbore import Wellbore

TRUTH = {"correlation": "hagedorn_brown", "gravity_mult": 1.04, "friction_mult": 1.15}
RESERVOIR = {"k": 50.0, "h": 60.0, "phi": 0.20, "re": 1500.0, "rw": 0.354,
             "skin": 2.0, "C": 0.03, "swi": 0.25, "cw": 3e-6, "cf": 4e-6}


def demo_pvt() -> BlackOilPVT:
    return BlackOilPVT(api=32.0, gas_sg=0.75, rsb=600.0, t_res=210.0, water_sg=1.07)


def demo_wellbore() -> Wellbore:
    md = [0, 1500, 2000, 2500, 3000, 3500, 4000, 6000, 8000, 10500]
    inc_deg = [0, 0, 5, 12, 19, 25, 30, 30, 30, 30]
    tvd = [0.0]
    for i in range(1, len(md)):
        a = np.radians(0.5 * (inc_deg[i] + inc_deg[i - 1]))
        tvd.append(tvd[-1] + (md[i] - md[i - 1]) * np.cos(a))
    return Wellbore(md, tvd, tubing_id=3.958, perf_md=10400, gauge_md=10000,
                    roughness=0.0006, wht=125.0, bht=210.0, step=300.0)


def truth_pi(pvt: BlackOilPVT | None = None, p_ref: float = 4200.0) -> float:
    pvt = pvt or demo_pvt()
    _, bo, _, mu = pvt.oil(p_ref, pvt.t_res)
    r = RESERVOIR
    return float(darcy_pi(r["k"], r["h"], mu, bo, r["re"], r["rw"], r["skin"]))


def demo_well(truth: bool = True) -> WellModel:
    """Truth model (truth=True) or an unmatched starting model (truth=False)."""
    pvt, wb = demo_pvt(), demo_wellbore()
    if truth:
        return WellModel(pvt, wb, pi=truth_pi(pvt), **TRUTH)
    return WellModel(pvt, wb, correlation="beggs_brill", pi=1.0)


def make_tests(n: int = 14, seed: int = 7, bad: bool = True) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    m = demo_well(truth=True)
    months = np.linspace(0, 36, n)
    dates = pd.Timestamp("2023-01-15") + pd.to_timedelta(months * 30.44, unit="D")
    pr = 4200 - 18.0 * months                              # depletion
    wc = np.clip(0.04 + 0.30 * (months / 36) ** 1.4, 0, 0.95)
    gor = np.full(n, m.pvt.rsb)
    whp = np.tile([200.0, 340.0, 260.0, 400.0, 230.0, 300.0], n)[:n]  # choke changes span rate
    op = m.operating_point(pr, whp, wc, gor)
    if not op.flowing.all():
        raise RuntimeError("synthetic truth well does not flow at every test")
    q = op.rate
    pg = m.gauge_pressure(q, wc, gor, whp)

    df = pd.DataFrame({
        "test_id": [f"T{i+1:02d}" for i in range(n)],
        "date": dates.normalize(),
        "liquid_rate": q * (1 + rng.normal(0, 0.015, n)),
        "water_cut": np.clip(wc + rng.normal(0, 0.008, n), 0, 0.99),
        "gor": gor * (1 + rng.normal(0, 0.02, n)),
        "whp": whp + rng.normal(0, 3, n),
        "gauge_pressure": pg + rng.normal(0, 4, n),
        "reservoir_pressure": pr + rng.normal(0, 15, n),
        "duration_hr": rng.uniform(8, 24, n).round(1),
        "rate_cv": rng.uniform(0.01, 0.05, n).round(3),
    })
    planted = {}
    if bad:
        df.loc[3, "gauge_pressure"] += 200.0;  planted["T04"] = "gauge +200 psi"
        df.loc[7, "liquid_rate"] *= 1.30;      planted["T08"] = "rate +30 %"
        df.loc[10, "gor"] *= 0.60;             planted["T11"] = "GOR -40 % (meter fault)"
        df.loc[5, "duration_hr"] = 3.0;        planted["T06"] = "3-hour test"
    num = ["liquid_rate", "gor", "whp", "gauge_pressure", "reservoir_pressure"]
    df[num] = df[num].round(1)
    df["water_cut"] = df["water_cut"].round(3)
    truth = {**TRUTH, "pi": m.pi, "planted_errors": planted,
             "true_rate": q, "true_gauge": pg}
    return df, truth


def demo_reservoir():
    """RadialReservoir consistent with the truth PI."""
    from .transient import RadialReservoir
    pvt = demo_pvt()
    _, bo, _, mu = pvt.oil(4200.0, pvt.t_res)
    r = RESERVOIR
    ct = (1 - r["swi"]) * float(pvt.total_compressibility_oil(4200.0)) + r["swi"] * r["cw"] + r["cf"]
    return RadialReservoir(k=r["k"], h=r["h"], phi=r["phi"], ct=ct, mu=float(mu), bo=float(bo),
                           rw=r["rw"], re=r["re"], skin=r["skin"], C=r["C"])


def make_buildup(q: float = 3500.0, tp_hr: float = 240.0, p_init: float = 3560.0,
                 seed: int = 3) -> pd.DataFrame:
    """Noisy build-up survey (0.01-120 h) from the truth reservoir."""
    rng = np.random.default_rng(seed)
    res = demo_reservoir()
    dt = np.geomspace(0.01, 120, 70)
    pws = res.buildup(dt, tp_hr, q, p_init) + rng.normal(0, 0.3, dt.size)
    return pd.DataFrame({"dt_hr": dt, "pws": pws.round(2)})


def demo_stoiip() -> float:
    """Oil in place in the well's drainage circle (STB)."""
    r = RESERVOIR
    _, bo, _, _ = demo_pvt().oil(4200.0, 210.0)
    area_acres = np.pi * r["re"] ** 2 / 43560.0
    return float(7758 * area_acres * r["h"] * r["phi"] * (1 - r["swi"]) / bo)
