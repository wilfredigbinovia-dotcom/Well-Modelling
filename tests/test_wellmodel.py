"""Regression and physics tests. Run with:  pytest -q"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wellmodel as wm
from wellmodel import ipr
from wellmodel.synthetic import (TRUTH, demo_reservoir, demo_well, make_buildup, make_tests,
                                 truth_pi)
from wellmodel.transient import invert_laplace, pwd_laplace


@pytest.fixture(scope="module")
def data():
    tests, truth = make_tests()
    base = demo_well(truth=False)
    qc = wm.qc_tests(base, tests)
    return tests, truth, base, qc


# ------------------------------------------------------------------ PVT
def test_bubble_point_and_rs_consistent():
    pvt = demo_well().pvt
    rs, bo, _, _ = pvt.oil(pvt.pb, pvt.t_res)
    assert rs == pytest.approx(pvt.rsb, rel=1e-3)
    assert 1.2 < bo < 1.5


def test_measured_pb_calibration():
    pvt = wm.BlackOilPVT(32, 0.75, 600, 210, pb=3000)
    assert pvt.bubble_point() == pytest.approx(3000, rel=1e-6)
    assert pvt.oil(3000, 210)[0] == pytest.approx(600, rel=1e-3)


def test_z_factor_reasonable():
    z = demo_well().pvt.z_factor(np.array([14.7, 2000, 5000]), 200)
    assert z[0] == pytest.approx(1.0, abs=0.01)
    assert 0.8 < z[1] < 0.95 and 0.85 < z[2] < 1.1


# ------------------------------------------------------------------ IPR
def test_ipr_inverse_roundtrip():
    q = np.linspace(10, 4400, 50)          # AOF = 4511 STB/d
    p = ipr.pwf(q, 3500, 2.0, 2800)
    assert np.allclose(ipr.rate(p, 3500, 2.0, 2800), q, rtol=1e-6)


def test_vogel_continuity_at_pb():
    pb = 2800
    eps = 1e-3
    assert ipr.rate(pb + eps, 3500, 2.0, pb) == pytest.approx(ipr.rate(pb - eps, 3500, 2.0, pb), rel=1e-5)


# ------------------------------------------------------------------ VLP
def test_static_water_column():
    m = demo_well()
    p = m.pwf_vlp(0.1, 0.9999, 0.0, 14.7, gravity_mult=1.0)[0]
    _, rho_w, _ = m.pvt.water(3000, 170)
    assert p == pytest.approx(14.7 + rho_w / 144 * m.wellbore.perf_tvd, rel=0.02)


def test_multipliers_increase_pressure():
    m = demo_well()
    p1 = m.pwf_vlp(3000, 0.2, 600, 300)[0]
    assert m.pwf_vlp(3000, 0.2, 600, 300, gravity_mult=1.1)[0] > p1
    assert m.pwf_vlp(3000, 0.2, 600, 300, friction_mult=1.5)[0] > p1


def test_vectorised_matches_scalar():
    m = demo_well()
    q = np.array([500.0, 2000, 4000])
    vec = m.pwf_vlp(q, 0.2, 600, 300)
    sca = [m.pwf_vlp(x, 0.2, 600, 300)[0] for x in q]
    assert np.allclose(vec, sca)


def test_operating_point_is_intersection():
    m = demo_well()
    op = m.operating_point(4000, 300, 0.1, 600)
    assert op.flowing[0] and op.stable[0]
    pv = m.pwf_vlp(op.rate, 0.1, 600, 300)[0]
    assert pv == pytest.approx(op.pwf[0], abs=3.0)


def test_dead_well_detected():
    m = demo_well()
    op = m.operating_point(2500, 300, 0.8, 600)
    assert not op.flowing[0] and op.rate[0] == 0


# ------------------------------------------------------------------- QC
def test_qc_catches_planted_errors(data):
    _, truth, _, qc = data
    st = qc.table.set_index("test_id")["status"]
    for tid in ("T04", "T08", "T11"):
        assert st[tid] == "fail", tid
    assert st["T06"] == "warn"
    assert (st.drop(["T04", "T08", "T11", "T06"]) == "pass").all()


# ------------------------------------------------------- match/selection
def test_match_recovers_truth(data):
    tests, truth, base, qc = data
    mr = wm.match_tests(base.copy(correlation=TRUTH["correlation"]), tests, weights=qc.weights)
    assert mr.params["gravity_mult"] == pytest.approx(TRUTH["gravity_mult"], abs=0.02)
    assert abs(mr.params["friction_mult"] - TRUTH["friction_mult"]) < 2.5 * mr.std_errors["friction_mult"]
    assert mr.params["pi"] == pytest.approx(truth["pi"], rel=0.03)
    assert mr.flags == []


def test_selection_picks_truth(data):
    tests, _, base, qc = data
    sel = wm.compare_correlations(base, tests, weights=qc.weights)
    assert sel.recommended == TRUTH["correlation"]


# -------------------------------------------------------- probabilistic
def test_monte_carlo_percentiles_ordered():
    m = demo_well()
    mc = wm.monte_carlo(m, {"reservoir_pressure": wm.Uniform(3600, 3900), "whp": 250,
                            "water_cut": wm.Triangular(0.2, 0.25, 0.3), "gor": 600},
                        n=200, seed=0)
    assert mc.p90 < mc.p50 < mc.p10
    assert mc.sensitivity.iloc[0]["variable"] in ("reservoir_pressure", "water_cut")


def test_lognormal_quantiles():
    d = wm.LogNormal(p90=1.0, p10=4.0)
    assert d.ppf(np.array([0.1]))[0] == pytest.approx(1.0, rel=1e-6)
    assert d.ppf(np.array([0.9]))[0] == pytest.approx(4.0, rel=1e-6)


# ------------------------------------------------------------ transient
def test_infinite_acting_matches_log_approximation():
    for td in (1e4, 1e6):
        p = invert_laplace(lambda s: pwd_laplace(s, 0.0, 1.0, None), td)[0]
        assert p == pytest.approx(0.5 * (np.log(td) + 0.80907) + 1.0, abs=0.01)


def test_wellbore_storage_unit_slope():
    p = invert_laplace(lambda s: pwd_laplace(s, 1000.0, 0.0, None), 1.0)[0]
    assert p == pytest.approx(1e-3, rel=0.01)


def test_buildup_fit_recovers_reservoir():
    bu = make_buildup()
    guess = replace(demo_reservoir(), k=10.0, skin=0.0, C=0.01)
    fit = wm.fit_buildup(bu.dt_hr, bu.pws, 240, 3500, guess)
    assert fit.reservoir.k == pytest.approx(50.0, rel=0.02)
    assert fit.reservoir.skin == pytest.approx(2.0, abs=0.1)
    assert fit.pi_pss == pytest.approx(truth_pi(), rel=0.02)


def test_forecast_runs_and_depletes():
    m = demo_well()
    fr = wm.forecast(m, wm.DepletionInputs(stoiip=8e6, p_init=4200, whp=250), days=1500)
    d = fr.table
    assert d["reservoir_pressure"].iloc[-1] < 4200
    assert (np.diff(d["Np"]) >= 0).all()
    assert fr.events["pr_below_pb_days"] is not None


# -------------------------------------------------------------- options
from wellmodel.options import WellOptions, NOT_YET


def test_default_options_computable():
    o = WellOptions()
    assert o.computable and o.validate() == []


def test_gas_injector_with_oil_is_invalid():
    o = WellOptions(well_type="gas_injector", fluid_type="oil")
    assert o.errors and not o.computable


def test_unsupported_choices_block_calculation():
    o = WellOptions(fluid_type="dry_wet_gas", method="eos", flow_type="annular")
    assert not o.errors
    assert not o.computable
    assert o.engine_status()["method"] == NOT_YET
    assert len(o.unsupported) == 3


def test_completion_choices_absorbed_not_blocking():
    o = WellOptions(completion="open_hole", sand_control="gravel_pack")
    assert o.computable
    assert any("pack thickness" in x for x in o.data_requirements()["Equipment"])


def test_condensate_black_oil_warns():
    o = WellOptions(fluid_type="retrograde_condensate")
    assert any(i.level == "warning" and "method" in i.fields for i in o.validate())


def test_cased_hole_slotted_liner_warns():
    o = WellOptions(sand_control="slotted_liner")
    assert any(i.level == "warning" for i in o.validate())


def test_invalid_choice_rejected():
    with pytest.raises(ValueError):
        WellOptions(fluid_type="steam")
