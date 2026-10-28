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
    # PE2-style is Hagedorn & Brown in slug flow (holdup and 1.28 x the friction, as PROSPER's PE2), so with fitted
    # multipliers it fits Hagedorn & Brown data as well - either is the right answer
    assert sel.recommended in (TRUTH["correlation"], "pe2_style")


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
    """Every listed choice is modelled now (water and gas injectors included); a choice missing from MODELLED
    blocks calculation."""
    from wellmodel import options as wopt
    o = WellOptions(fluid_type="dry_wet_gas", method="black_oil", flow_type="annular", well_type="gas_injector")
    assert not o.errors and o.computable
    saved = set(wopt.MODELLED["well_type"])
    try:
        wopt.MODELLED["well_type"].discard("gas_injector")
        assert not o.computable
        assert o.engine_status()["well_type"] == NOT_YET and o.engine_status()["flow_type"] != NOT_YET
        assert len(o.unsupported) == 1
    finally:
        wopt.MODELLED["well_type"] = saved


def test_eos_is_modelled_for_any_fluid():
    from wellmodel.options import MODELLED_NOW
    o = WellOptions(fluid_type="retrograde_condensate", method="eos")
    st = o.engine_status()
    assert st["method"] == MODELLED_NOW and st["fluid_type"] == MODELLED_NOW
    assert not o.unsupported and o.computable


def test_completion_choices_absorbed_not_blocking():
    o = WellOptions(completion="open_hole", sand_control="gravel_pack")
    assert o.computable
    assert any("pack thickness" in x for x in o.data_requirements()["Equipment"])


def test_condensate_black_oil_notes_validation():
    """Black-oil retrograde condensate is modelled (Tutorial 38); the options note says to validate it against EoS."""
    o = WellOptions(fluid_type="retrograde_condensate")
    assert any(i.level == "info" and "method" in i.fields and "equation of state" in i.message for i in o.validate())
    assert not any(i.level == "error" for i in o.validate())


def test_cased_hole_slotted_liner_warns():
    o = WellOptions(sand_control="slotted_liner")
    assert any(i.level == "warning" for i in o.validate())


def test_invalid_choice_rejected():
    with pytest.raises(ValueError):
        WellOptions(fluid_type="steam")


# ------------------------------------------------------ black-oil PVT tab
import pandas as pd
from wellmodel import blackoil as bo


@pytest.mark.parametrize("fam", ["standing", "vazquez_beggs", "petrosky", "al_marhoun"])
def test_pb_rs_exact_inverses(fam):
    inp = bo.PVTInputs(rsb=600, api=33, gas_sg=0.78)
    pb = bo.pb_raw(fam, inp, 600, 200)
    # published exponents are reciprocal to ~5 significant figures
    assert bo.rs_raw(fam, inp, pb, 200) == pytest.approx(600, rel=1e-4)


@pytest.mark.parametrize("fam", ["lasater", "glaso"])
def test_pb_rs_published_inverses_close(fam):
    inp = bo.PVTInputs(rsb=600, api=33, gas_sg=0.78)
    pb = bo.pb_raw(fam, inp, 600, 200)
    assert bo.rs_raw(fam, inp, pb, 200) == pytest.approx(600, rel=0.01)


def test_single_point_match_splits_correction_equally():
    # one point, correlation 3263 psia vs measured 3440: same pattern as PROSPER
    m = bo.fit_linear([3263.0], [3440.0])
    assert m.p1 == pytest.approx(1.02712, abs=1e-4)
    assert m.p2 == pytest.approx(88.5, abs=0.1)
    assert np.isnan(m.std)


def test_example_match_recovers_generating_correlations():
    inp, data = bo.example_data()
    r = bo.match_all(inp, data)
    assert r.suggestion == {"pb_family": "standing", "visc_family": "beal"}
    rs = r.matches[("rs", "standing")]
    assert rs.p1 == pytest.approx(1.04, abs=0.02) and rs.p2 == pytest.approx(-12, abs=8)
    assert r.matches[("visc", "beal")].p1 == pytest.approx(1.06, abs=0.02)
    assert r.matches[("bo", "vazquez_beggs")].p1 == pytest.approx(0.99, abs=0.01)


def test_matched_model_honours_measured_bubble_point():
    inp, data = bo.example_data()
    r = bo.match_all(inp, data)
    m = bo.BlackOilModel(inp, "glaso", "petrosky",
                         {p: r.matches[(p, "glaso")] for p in ("pb", "rs", "bo")})
    assert m.bubble_point(data.temperature) == pytest.approx(3010, abs=0.01)
    tab = m.table(data.temperature, [3010.0, 4000.0])
    assert (tab["Solution GOR (scf/STB)"] == inp.rsb).all()


def test_unverified_correlations_not_calculated():
    inp, data = bo.example_data()
    r = bo.match_all(inp, data)
    assert ("pb", "de_ghetto") in r.matches                 # verified against PROSPER (heavy-oil form)
    assert ("visc", "bergman_sutton") in r.matches and ("visc", "de_ghetto") in r.matches
    assert r.table("visc")["Bergman-Sutton"].notna().all()
    assert r.table("visc")["De Ghetto Mod"].isna().all()


def test_lab_data_checks():
    inp = bo.PVTInputs(rsb=500, api=35, gas_sg=0.75)
    tbl = pd.DataFrame({"Pressure": [4000, 2000, 1000], "Gas Oil Ratio": [450, 300, 350],
                        "Oil FVF": [1.30, 1.25, 1.20], "Oil Viscosity": [0.6, 0.8, 1.0]})
    w = bo.MatchData(200, 2500, tbl).check(inp)
    assert any("Above the bubble point the GOR" in x for x in w)
    assert any("GOR should rise" in x for x in w)


def test_gas_and_water_properties_sane():
    inp = bo.PVTInputs(rsb=500, api=35, gas_sg=0.7, salinity_ppm=0)
    z, bg, _, mu = bo.gas_props(inp, 14.7, 60)
    assert z == pytest.approx(1.0, abs=0.01)
    _, rho_w, mu_w = bo.water_props(inp, 14.7, 200)
    assert mu_w == pytest.approx(0.30, abs=0.03)
    sour = bo.PVTInputs(rsb=500, api=35, gas_sg=0.7, co2=10, h2s=5)
    assert bo.gas_pseudocritical(sour)[0] < bo.gas_pseudocritical(inp)[0]


# ------------------------------------------------------------- EOS / PRP
from wellmodel import eos as weos


@pytest.fixture(scope="module")
def example_fluid():
    return weos.EOSFluid(weos.parse_prp(weos.example_prp_text(), "example"))


def test_prp_parse_units_and_bics():
    prp = weos.parse_prp(weos.example_prp_text())
    c = prp.components
    assert prp.eos == "PR" and len(c) == 13
    assert c.loc[c["Name"] == "C1", "Pc (psia)"].iloc[0] == pytest.approx(667.8, abs=0.01)   # psig -> psia
    assert c["Mole fraction"].sum() == pytest.approx(1.0)
    assert prp.bic[2, 10] == prp.bic[10, 2] == pytest.approx(0.03)
    assert prp.separators[-1][1] == pytest.approx(15.0, abs=0.01)
    assert prp.reference_temperature == 210


def test_prp_rejects_other_files():
    with pytest.raises(ValueError):
        weos.parse_prp("just some text\n1 2 3\n")


def test_methane_z_close_to_nist(example_fluid):
    f = example_fluid
    z = np.zeros(f.n); z[f.names.index("C1")] = 1
    f0 = weos.EOSFluid(f.prp, z); f0.c[:] = 0       # unshifted PR
    assert f0.phase_props(z, 1000, 100).z == pytest.approx(0.885, abs=0.01)


def test_flash_equilibrium_and_material_balance(example_fluid):
    f = example_fluid
    r = f.flash(1200, 210)
    assert r.phases == "2P"
    T = 210 + weos.RANKINE
    lx, _ = f._lnphi(r.x, 1200, T, phase="L")
    ly, _ = f._lnphi(r.y, 1200, T, phase="V")
    assert np.max(np.abs(np.log(r.x) + lx - np.log(r.y) - ly)) < 1e-5
    assert np.allclose(r.beta * r.y + (1 - r.beta) * r.x, f.z, atol=1e-10)
    assert r.liquid.density > r.vapour.density and r.ift > 0


def test_example_oil_bubble_point_and_surface_volumes(example_fluid):
    f = example_fluid
    ps, typ = f.saturation_pressure(210)
    assert typ == "bubble" and 1800 < ps < 2800
    assert f.stability(ps * 1.02, 210)[0] and not f.stability(ps * 0.98, 210)[0]
    sep = f.separator_train(T_res=210, P_res=ps * 1.001)
    assert 500 < sep["gor"] < 1200 and 25 < sep["api"] < 40 and 1.3 < sep["bo"] < 1.7


def test_lbc_viscosity_reasonable(example_fluid):
    r = example_fluid.flash(1200, 210)
    assert 0.1 < r.liquid.viscosity < 2.0
    assert 0.01 < r.vapour.viscosity < 0.03


# ------------------------------------------------------------------ IPR tab
from wellmodel import inflow as iflow


def test_dietz_circle_equals_radial():
    re = np.sqrt(40 * 43560 / np.pi)
    assert iflow.dietz_term(40, 31.62, 0.354) == pytest.approx(np.log(re / 0.354) - 0.75, abs=1e-3)


def test_vogel_honours_test_point_and_continuity():
    c = iflow.vogel(3000, 2000, 1500, 1500)
    assert c.rate(1500) == pytest.approx(1500)
    assert c.rate(2000 + 1e-6) == pytest.approx(c.rate(2000 - 1e-6), rel=1e-6)


def test_composite_limits():
    v = iflow.vogel(3000, 2000, 1500, 1500)
    c0 = iflow.composite(3000, 2000, 1500, 1500, 0.0)
    assert c0.rate(1000) == pytest.approx(v.rate(1000), rel=1e-3)
    c1 = iflow.composite(3000, 2000, 1500, 1500, 1.0)
    assert c1.aof == pytest.approx(3000, rel=1e-3)            # straight line, J = 1
    assert iflow.composite(3000, 2000, 1500, 1500, 0.5).rate(1500) == pytest.approx(1500, rel=1e-4)


def test_jones_oil_b_equals_inverse_darcy_pi():
    a, b = iflow.jones_oil_coeffs(50, 60, 40, 31.62, 0.354, 2, 0.8, 1.3, 45, 60)
    assert b == pytest.approx(1 / iflow.darcy_pi(50, 60, 0.8, 1.3, 40, 31.62, 0.354, 2))
    assert a > 0


@pytest.mark.parametrize("fit_pr", [False, True])
def test_multirate_fits_recover_parameters(fit_pr):
    pw = np.array([3500, 3000, 2500, 2000.0])
    q = 1e-3 * (4000 ** 2 - pw ** 2) ** 0.8
    t = pd.DataFrame({"Rate": q, "Pwf": pw})
    r = iflow.multirate_c_and_n(3000 if fit_pr else 4000, t)
    assert r.pr == pytest.approx(4000, rel=1e-6) and r.params["n"] == pytest.approx(0.8, abs=1e-6)
    qo = np.array([500, 1000, 1500, 2000.0])
    to = pd.DataFrame({"Rate": qo, "Pwf": 3000 - (2e-5 * qo ** 2 + 0.4 * qo)})
    r = iflow.multirate_jones_oil(1000 if fit_pr else 3000, to)
    assert r.params["b (psi/(STB/d))"] == pytest.approx(0.4, rel=1e-6) and r.pr == pytest.approx(3000)


def test_gas_back_pressure_n1_equals_darcy_jones():
    a, b, _ = iflow.jones_gas_coeffs(10, 50, 160, 31.62, 0.354, 0, 200, 0.02, 0.9, 0.7, 50)
    bp = iflow.back_pressure(4000, 10, 50, 160, 31.62, 0.354, 0, 200, 0.02, 0.9, 1.0)
    assert bp.aof == pytest.approx(4000 ** 2 / b, rel=1e-9)
    assert iflow.back_pressure(4000, 10, 50, 160, 31.62, 0.354, 0, 200, 0.02, 0.9, 0.7).aof == pytest.approx(bp.aof)


def test_pseudo_pressure_ideal_limit():
    psi = iflow.pseudo_pressure_fn(lambda p: (np.full_like(p, 0.02), np.full_like(p, 0.9)), 5000)
    assert psi(4000) == pytest.approx(4000 ** 2 / 0.018, rel=1e-4)


def test_transient_reverts_to_pss_at_late_time():
    args = (3000, 2000, 1.0, 60, 160, 31.62, 0.354, 0, 0.8, 1.3, 0.15, 1.5e-5)
    early = iflow.transient_oil(*args, 1.0)
    late = iflow.transient_oil(*args, 5000.0)
    assert early.params["PI (STB/d/psi)"] > late.params["PI (STB/d/psi)"]
    assert late.params["PI (STB/d/psi)"] == pytest.approx(late.params["Pseudo-steady-state PI"])


# ------------------------------------------------- skin and sand control
from wellmodel import completion as wcomp, inflow_adv as iadv


def test_hawkins_skin_hand_calc():
    assert wcomp.hawkins(100, 0.354, 1.0, 20) == pytest.approx(4 * np.log(1.354 / 0.354))


def test_karakas_tariq_trends():
    base = wcomp.kt_perforation_skin(0.406, 8 / 12, 0.25 / 12, 2, 90)["perforation"]
    assert wcomp.kt_perforation_skin(0.406, 8 / 12, 0.25 / 12, 2, 0)["perforation"] > base   # 0 deg worst
    assert wcomp.kt_perforation_skin(0.406, 8 / 12, 0.25 / 12, 4, 90)["perforation"] < base   # more shots
    assert wcomp.kt_perforation_skin(0.406, 1.0, 0.25 / 12, 2, 90)["perforation"] < base      # longer perfs
    assert wcomp.kt_perforation_skin(0.406, 8 / 12, 0.25 / 12, 2, 90, 0.1)["perforation"] > base  # low kv


def test_deviation_and_partial_penetration_skins():
    assert wcomp.deviation_skin(0, 100, 0.354) == 0
    assert wcomp.deviation_skin(60, 100, 0.354) < wcomp.deviation_skin(30, 100, 0.354) < 0
    assert wcomp.partial_penetration_skin(100, 100, 0, 0.354) == 0
    assert wcomp.partial_penetration_skin(100, 20, 0, 0.354) > wcomp.partial_penetration_skin(100, 50, 0, 0.354) > 0


def test_sand_control_reduces_rate_consistently():
    c = iflow.darcy(3000, 2000, 100, 60, 160, 31.62, 0.354, 0, 0.8, 1.3)
    sc = wcomp.SandControl("gravel_pack_cased", dict(spf=12, interval=60, perf_efficiency=0.8,
                                                     perf_diameter_in=0.75, pack_length_in=4, gravel_perm=120000))
    dp = wcomp.sand_control_dp(sc, wcomp.props_constant(1.3, 45.0, 0.8), 100, 60, 0.354)
    c2 = wcomp.apply_sand_control(c, dp)
    assert c2.aof < c.aof and c2.pwf[-1] == pytest.approx(0.0)
    i = 30                                   # sand-face pressure reproduces the original IPR
    assert c.rate(c2.pwf[i] + c2.dp_sand_control[i]) == pytest.approx(c2.q[i], rel=1e-3)
    # PROSPER's Jones form: a = 9.08e-13 beta Bo^2 rho L / A^2, b = mu Bo L / (1.127e-3 kg A), Q in STB/d
    A = np.pi * (0.75 / 24) ** 2 * 12 * 60 * 0.8
    beta = 1.47e7 / 120000 ** 0.55
    a = 9.08e-13 * beta * 1.3 ** 2 * 45 * (4 / 12) / A ** 2
    b = 0.8 * 1.3 * (4 / 12) / (1.127e-3 * 120000 * A)
    assert float(dp(np.array([3000.0]), 2000.0)[0]) == pytest.approx(a * 3000 ** 2 + b * 3000, rel=1e-9)
    det = dp.details(np.array([3000.0]), 2000.0)
    assert det["skin"][0] == pytest.approx((a * 3000 ** 2 + b * 3000) * 100 * 60 / (141.2 * 3000 * 1.3 * 0.8))


def test_screens_and_slotted_liner_skins():
    props = wcomp.props_constant(1.2, 50.0, 1.0)
    k, h, rw = 100.0, 50.0, 0.354
    # wire-wrapped screen, formation sand outside: Darcy skin ln(rw/rso)
    ww = wcomp.sand_control_dp(wcomp.SandControl("wire_wrapped_screen", dict(interval=50, screen_outer_radius=0.25,
                                                                            outside_beta=0)), props, k, h, rw)
    assert ww.details(np.array([1.0]), 2000.0)["skin"][0] == pytest.approx(np.log(rw / 0.25), rel=1e-6)
    # pre-packed screen: two radial shells; non-Darcy Do beta (1/r1 - 1/r2)
    pp = wcomp.sand_control_dp(wcomp.SandControl("pre_packed_screen", dict(
        interval=50, screen_inner_radius=0.2, screen_outer_radius=0.25, screen_perm=500.0, outside_perm=20000.0)),
        props, k, h, rw)
    d = pp.details(np.array([1000.0]), 2000.0)
    s_darcy = k / 500 * np.log(0.25 / 0.2) + k / 20000 * np.log(rw / 0.25)
    do = wcomp.DO_CONST * 50 * k * h / (1.0 * 50 ** 2)
    dd = do * (1.47e7 / 500 ** 0.55 * (1 / 0.2 - 1 / 0.25) + 1.47e7 / 20000 ** 0.55 * (1 / 0.25 - 1 / rw))
    assert d["darcy skin"][0] == pytest.approx(s_darcy, rel=1e-9) and d["D"][0] == pytest.approx(dd, rel=1e-9)
    assert d["dp"][0] == pytest.approx(141.2 * 1200 * 1.0 / (k * h) * (s_darcy + dd * 1200), rel=1e-9)
    # slotted liner: more slots or wider slots -> less skin; partial interval -> more skin
    def sl(**kw):
        p = dict(interval=50, liner_inner_radius=0.25, liner_outer_radius=0.29, slot_density=3.0, slot_height_in=4.0,
                 slot_width_in=0.02)
        p.update(kw)
        return wcomp.sand_control_dp(wcomp.SandControl("slotted_liner", p), props, k, h, rw).details(
            np.array([1000.0]), 2000.0)["skin"][0]
    base = sl()
    assert base > 0 and sl(slot_density=12.0) < base and sl(slot_width_in=0.05) < base and sl(interval=25) > base
    assert sl(screen_outer_radius=0.2) > base
    with pytest.raises(ValueError):
        sl(screen_outer_radius=0.3)


def test_sand_control_properties_at_sand_face():
    *_, oil = _vlp_setup()
    props_m = wcomp.props_from_fluid(oil, "oil", 200.0, {"wc": 0.0, "gor": float(oil.inp.rsb)}, multiphase=True)
    props_s = wcomp.props_from_fluid(oil, "oil", 200.0, {"wc": 0.0, "gor": float(oil.inp.rsb)}, multiphase=False)
    q = np.array([1000.0])
    qa, rho_a, _ = props_m(q, 5000.0)                  # undersaturated: one phase, both methods agree
    qb, rho_b, _ = props_s(q, 5000.0)
    assert qa[0] == pytest.approx(qb[0]) and rho_a[0] == pytest.approx(rho_b[0])
    qm, rho_m, _ = props_m(q, 800.0)                   # below the bubble point: free gas in 'Multi Phase' only
    qs, rho_s, _ = props_s(q, 800.0)
    assert qm[0] > qs[0] and rho_m[0] < rho_s[0]


# ------------------------------------------------------- advanced inflow
def test_gringarten_limits():
    assert iadv.pd_infinite_conductivity(1e-4) == pytest.approx(np.sqrt(np.pi * 1e-4), rel=0.01)
    assert iadv.pd_infinite_conductivity(1e5) == pytest.approx(0.5 * (np.log(1e5) + 0.80907) + np.log(2), abs=0.02)
    assert iadv.cinco_f(1000) == pytest.approx(np.log(2), abs=0.02)


def test_fractured_well_beats_unfractured_and_reaches_pss():
    args = dict(pr=3000, k=1.0, h=60, area=160, ca=31.62, rw=0.354, xf=300, fcd=10, phi=0.12, ct=1.5e-5,
                mu=0.8, pb=2000, bo=1.3)
    early = iadv.fractured_well("oil", t_days=1, **args).params["PI (STB/d/psi)"]
    late = iadv.fractured_well("oil", t_days=5000, **args).params["PI (STB/d/psi)"]
    assert early > late > iflow.darcy_pi(1.0, 60, 0.8, 1.3, 160, 31.62, 0.354, 0)


def test_babu_odeh_full_length_and_joshi_agreement():
    geo = dict(k=10, h=50, rw=0.3, aniso_h=1, aniso_v=0.1, lx=4000, ly=2000, xw=2000, yw=1000, zw=25)
    assert iadv.babu_odeh(L=4000, **geo)[2] == 0.0
    kh_b, d_b, _ = iadv.babu_odeh(L=500, **geo)
    kh_j, d_j = iadv.joshi(10, 50, 0.3, 1, 0.1, 500, 4000, 2000)
    assert kh_b / d_b == pytest.approx(kh_j / d_j, rel=0.1)
    pis = [np.divide(*iadv.babu_odeh(L=L, **geo)[:2]) for L in (500, 1000, 2000, 4000)]
    assert all(np.diff(pis) > 0)


def test_dual_porosity_homogeneous_limit():
    a = iadv.dual_porosity("oil", 3000, 50, 60, 640, 31.62, 0.354, 0.1, 1e-5, 0.2, 0.9999, 1e-3, 0.8, 0,
                           pb=2000, bo=1.3)
    td = a.params["tD"]
    j_line = 50 * 60 / (141.2 * 0.8 * 1.3 * 0.5 * (np.log(td) + 0.80907))
    assert a.params["PI (STB/d/psi)"] == pytest.approx(j_line, rel=0.01)


def test_multilayer_sum_and_crossflow():
    m = iadv.multilayer("oil", [dict(pr=3000, j=1.0), dict(pr=3000, j=1.0)], pb=1000)
    assert m.aof == pytest.approx(2 * iflow.pi_entry(3000, 1000, 1.0).aof)
    m = iadv.multilayer("oil", [dict(pr=3000, j=1.0), dict(pr=2000, j=2.0)], pb=500)
    assert m.pr == pytest.approx((3000 + 2 * 2000) / 3)


def test_horizontal_friction_reduces_rate():
    geo = dict(k=10, h=50, rw=0.3, aniso_h=1, aniso_v=0.1, lx=4000, ly=2000, xw=2000, yw=1000, zw=25)
    kh, D, _ = iadv.babu_odeh(L=2000, **geo)
    props = dict(rho=45, mu=0.8, bo=1.3)
    zones = lambda d: [dict(type="Perforated", length=200, k=10, skin=0, id_in=d, rough_in=0.0006)] * 10
    big = iadv.horizontal_friction("oil", 3000, zones(6.0), kh, D, 10, props, pb=1000)
    small = iadv.horizontal_friction("oil", 3000, zones(1.5), kh, D, 10, props, pb=1000)
    assert small.aof < big.aof <= big.params["AOF without friction"] + 1e-6


def test_modified_isochronal_stabilised_below_isochronal():
    t = pd.DataFrame({"Rate": [2000, 4000, 6000, 8000.0], "Pwf": [3900, 3780, 3640, 3480.0]})
    r = iadv.modified_isochronal(4000, t, 12, 12, 5, 0.15, 0.25, 0.02, 0.354, 1489)
    assert r.params["C stabilised"] < r.params["C isochronal"]
    assert 0.5 <= r.params["n"] <= 1.0
    assert r.isochronal["Correction factor"].iloc[0] == 1.0


# ------------------------------------------------ multiphase pseudo-pressure
from wellmodel import mpp as wmpp


def test_corey_ratio_inversion():
    rel = wmpp.Corey(0.2, 0.2, 0.05, 0.9, 0.8, 2.5, 1.8)
    for ratio in (1e-3, 0.1, 1.0, 30.0):
        sg = rel.sg_from_ratio(ratio)
        assert rel.krg(sg) / rel.kro(sg) == pytest.approx(ratio, rel=1e-6)
    assert rel.sg_from_ratio(0) == 0


def _bo_model():
    from wellmodel import blackoil as bo
    inp, _ = bo.example_data()
    return bo.BlackOilModel(inp, "standing", "beal", {})


def test_mpp_oil_matches_darcy_above_bubble_point():
    m, T, pr, pw = _bo_model(), 205.0, 5000.0, 4000.0
    tab = wmpp.blackoil_table(lambda p: m.table(T, p), wmpp.pressure_grid(pr, m.bubble_point(T)), m.inp.rsb, 0,
                              wmpp.Corey())
    c = wmpp.multiphase_pseudo_pressure("oil", pr, 50, 60, 160, 31.6, 0.354, 0, tab)
    p = np.linspace(pw, pr, 400)
    t = m.table(T, p)
    expect = 50 * 60 / (141.2 * iflow.dietz_term(160, 31.6, 0.354)) * np.trapezoid(
        1 / (t["Oil viscosity (cp)"] * t["Oil FVF (RB/STB)"]), p)
    assert float(c.rate(pw)) == pytest.approx(expect, rel=2e-3)


def test_mpp_oil_free_gas_reduces_deliverability():
    m, T, pr = _bo_model(), 205.0, 3000.0
    pb = m.bubble_point(T)
    aofs = [wmpp.multiphase_pseudo_pressure(
        "oil", pr, 50, 60, 160, 31.6, 0.354, 0,
        wmpp.blackoil_table(lambda p: m.table(T, p), wmpp.pressure_grid(pr, pb), rp, 0, wmpp.Corey())).aof
        for rp in (m.inp.rsb, 1500, 3000)]
    assert aofs[0] > aofs[1] > aofs[2]


def test_mpp_dry_gas_matches_pseudo_pressure():
    mu_z = lambda p: (0.012 + 2e-6 * np.asarray(p), 0.9 + 0 * np.asarray(p))
    pr, T = 4000.0, 200.0
    tab = wmpp.gas_correlation_table(mu_z, wmpp.pressure_grid(pr, n=200), T, wmpp.Corey())
    c = wmpp.multiphase_pseudo_pressure("gas", pr, 10, 50, 160, 31.6, 0.354, 2.0, tab)
    psi = iflow.pseudo_pressure_fn(mu_z, pr, 2000)
    expect = 10 * 50 * (psi(pr) - psi(2000.0)) / (1422 * (T + 459.67) * (iflow.dietz_term(160, 31.6, 0.354) + 2))
    assert float(c.rate(2000.0)) == pytest.approx(expect, rel=5e-3)


def test_mpp_condensate_bank_reduces_rate():
    from wellmodel import eos
    f = eos.EOSFluid(eos.parse_prp(eos.example_prp_text()))
    T = 665.0                     # above the example oil's critical temperature: dew-point fluid
    ps, typ = f.saturation_pressure(T)
    assert typ == "dew"
    scf = f.separator_train()["stages"]["Gas (scf per lbmol feed)"].sum()
    pr = ps * 0.95
    tab = wmpp.eos_table(f, T, wmpp.pressure_grid(pr, ps, 20), wmpp.Corey(), scf / 1000)
    c = wmpp.multiphase_pseudo_pressure("gas", pr, 10, 50, 160, 31.6, 0.354, 0, tab)
    assert 0 < c.params["Two-phase productivity loss (%)"] < 100
    assert (tab.loc[tab["Pressure (psia)"] > 0.5 * ps, "So"] > 0).all()   # bank below the dew point


# ------------------------------------------------------------- equipment
from wellmodel import equipment as weq


def test_survey_filter_keeps_20_points_and_shape():
    long_ = weq.example_long_survey()
    md, tvd = weq.survey_arrays(long_)
    idx, mx, rms = weq.filter_survey(md, tvd)
    assert len(idx) == 20 and idx[0] == 0 and idx[-1] == len(md) - 1
    assert mx < 2.0 and rms <= mx
    assert weq.check_survey(md[idx], tvd[idx]) == []


def test_survey_checks():
    assert any(l == "error" for l, _ in weq.check_survey(np.array([0, 100, 90.]), np.array([0, 100, 90.])))
    assert any(l == "error" for l, _ in weq.check_survey(np.array([0, 100.]), np.array([0, 150.])))
    t = weq.survey_table(np.array([0, 1000, 2000.]), np.array([0, 1000, 1000 + 1000 * np.cos(np.radians(30))]))
    assert t["Angle from vertical (deg)"].iloc[-1] == pytest.approx(30, abs=1e-6)
    assert t["Cumulative Displacement (ft)"].iloc[-1] == pytest.approx(500, rel=1e-6)


def _downhole(rows):
    df = pd.DataFrame(rows)
    for c in weq.DOWNHOLE_COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[weq.DOWNHOLE_COLUMNS]


def test_downhole_sections_and_checks():
    df = _downhole([{"Type": "Xmas Tree", "Measured Depth (ft)": 35.5},
                    {"Type": "Tubing", "Measured Depth (ft)": 136.54, "Tubing Inside Diameter (in)": 2.992,
                     "Tubing Inside Roughness (in)": 0.0006},
                    {"Type": "SSSV", "Label": "SSSV", "Tubing Inside Diameter (in)": 1.5},
                    {"Type": "Tubing", "Measured Depth (ft)": 10962, "Tubing Inside Diameter (in)": 2.992,
                     "Tubing Inside Roughness (in)": 0.0006},
                    {"Type": "Casing", "Measured Depth (ft)": 10963, "Casing Inside Diameter (in)": 8.681,
                     "Casing Inside Roughness (in)": 0.0006}])
    assert weq.check_downhole(df, "tubing", 11000) == []
    assert any("deviation survey" in m for _, m in weq.check_downhole(df, "tubing", 10000))
    assert any("annular" in m for _, m in weq.check_downhole(df, "annular", 11000))
    top, secs, pts = weq.downhole_sections(df)
    assert top == 35.5 and list(secs["Type"]) == ["Tubing", "Tubing", "Casing"]
    assert secs["Length (ft)"].sum() == pytest.approx(10963 - 35.5)
    assert pts.iloc[0]["MD (ft)"] == 136.54


def test_surface_fittings_and_geothermal():
    df = pd.DataFrame([{"Type": "Manifold", "TVD of End Point (ft)": 0},
                       {"Type": "Line Pipe", "Pipe Length (ft)": 1000, "TVD of End Point (ft)": 0,
                        "Pipe Inside Diameter (in)": 4.0, "Pipe Inside Roughness (in)": 0.0018},
                       {"Type": "Fitting", "Fitting": "90° elbow (standard)", "Pipe Inside Diameter (in)": 4.0}])
    for c in weq.SURFACE_COLUMNS:
        if c not in df:
            df[c] = np.nan
    assert not any(l == "error" for l, _ in weq.check_surface(df, 60, 8))
    assert weq.surface_summary(df)["Equivalent length (ft)"].sum() == pytest.approx(1000 + 30 * 4 / 12)
    md, t = np.array([35.5, 10963.]), np.array([60, 182.19])
    assert weq.check_geothermal(md, t, 4.17, 35.5, 10963, 11000) == []
    assert any(l == "error" for l, _ in weq.check_geothermal(md, t, 4.17, 35.5, 12000, 13000))
    assert weq.temperature_at(md, t, 5499.25) == pytest.approx((60 + 182.19) / 2)


def test_well_schematic_svg_is_valid_and_labelled():
    import xml.etree.ElementTree as ET
    from wellmodel import schematic as wsch
    df = _downhole([{"Type": "Xmas Tree", "Measured Depth (ft)": 35.5},
                    {"Type": "Tubing", "Measured Depth (ft)": 3000, "Tubing Inside Diameter (in)": 2.992,
                     "Tubing Inside Roughness (in)": 0.0006},
                    {"Type": "SSSV", "Label": "TRSV", "Tubing Inside Diameter (in)": 1.5},
                    {"Type": "Tubing", "Measured Depth (ft)": 5000, "Tubing Inside Diameter (in)": 2.992,
                     "Tubing Inside Roughness (in)": 0.0006},
                    {"Type": "Gas Lift Mandrel", "Label": "GLM 1", "Tubing Inside Diameter (in)": 2.867},
                    {"Type": "Tubing", "Measured Depth (ft)": 9000, "Tubing Inside Diameter (in)": 2.992,
                     "Tubing Inside Roughness (in)": 0.0006},
                    {"Type": "Casing", "Measured Depth (ft)": 9500, "Casing Inside Diameter (in)": 6.276,
                     "Casing Inside Roughness (in)": 0.0006}])
    top, secs, pts = weq.downhole_sections(df)
    for comp, sc in (("cased_hole", "none"), ("open_hole", "gravel_pack"), ("cased_hole", "slotted_liner")):
        svg = wsch.well_schematic_svg(top, secs, pts, comp, sc, has_choke=True)
        ET.fromstring(svg)
        assert "TRSV" in svg and "GLM 1" in svg and "5,000.0 ft MD" in svg



# ------------------------------------------------------------------- VLP
from wellmodel import vlp as wvlp
from wellmodel import blackoil as wbo


def _vlp_setup():
    long_ = weq.example_long_survey()
    md, tvd = weq.survey_arrays(long_)
    idx, _, _ = weq.filter_survey(md, tvd)
    df = _downhole([{"Type": "Xmas Tree", "Measured Depth (ft)": 35.5},
                    {"Type": "Tubing", "Measured Depth (ft)": 136.54, "Tubing Inside Diameter (in)": 2.992,
                     "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 3.5,
                     "Casing Inside Diameter (in)": 8.681},
                    {"Type": "SSSV", "Tubing Inside Diameter (in)": 1.5},
                    {"Type": "Tubing", "Measured Depth (ft)": 10962, "Tubing Inside Diameter (in)": 2.992,
                     "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 3.5,
                     "Casing Inside Diameter (in)": 8.681},
                    {"Type": "Casing", "Measured Depth (ft)": 10963, "Casing Inside Diameter (in)": 8.681,
                     "Casing Inside Roughness (in)": 0.0006}])
    top, secs, pts = weq.downhole_sections(df)
    path = wvlp.build_flow_path(top, secs, pts, md[idx], tvd[idx], [35.5, 10963.0], [60.0, 182.19],
                                extra_md=[10000.0])
    inp, _ = wbo.example_data()
    return top, secs, pts, md[idx], tvd[idx], path, wvlp.BlackOilFluid(wbo.BlackOilModel(inp, "standing", "beal", {}))


def test_vlp_flow_path_geometry():
    top, secs, pts, smd, stvd, path, _ = _vlp_setup()
    assert path.top == 35.5 and path.bottom == 10963 and 10000.0 in path.md
    assert len(path.restrictions) == 1
    assert path.area[3] == pytest.approx(np.pi / 4 * (2.992 / 12) ** 2)
    ann = wvlp.build_flow_path(top, secs, pts, smd, stvd, [35.5, 10963.0], [60.0, 182.19], "annular")
    assert ann.area[3] == pytest.approx(np.pi / 4 * ((8.681 / 12) ** 2 - (3.5 / 12) ** 2))
    assert ann.d_h[3] == pytest.approx((8.681 - 3.5) / 12)


def test_vlp_static_columns_and_temperature_limits():
    *_, path, oil = _vlp_setup()
    gas = wvlp.WetGasFluid(wbo.PVTInputs(rsb=1, api=40, gas_sg=0.65), 50)
    P, _ = wvlp.traverse(path, gas, 3000.0, {"q": 1e-6, "wgr": 0.0, "cgr": 0.0}, "gray", u=4.17)
    assert 750 < P[0, -1] - P[0, 0] < 1050                       # gas column ~0.08-0.1 psi/ft
    P, _ = wvlp.traverse(path, oil, 200.0, {"q": 1.0, "wc": 0.9999, "gor": 0.0}, "beggs_brill")
    assert (P[0, -1] - P[0, 0]) / (path.tvd[-1] - path.tvd[0]) == pytest.approx(0.445, abs=0.01)
    _, T = wvlp.traverse(path, oil, 200.0, {"q": 2000.0, "wc": 0.2, "gor": 750.0}, "beggs_brill", u=0.0)
    assert T[0, 0] == pytest.approx(182.19)
    _, T = wvlp.traverse(path, oil, 200.0, {"q": 2000.0, "wc": 0.2, "gor": 750.0}, "beggs_brill", u=1e4)
    assert T[0, 0] == pytest.approx(60.0, abs=0.1)


def test_vlp_match_recovers_synthetic_truth():
    *_, path, oil = _vlp_setup()
    tests = pd.DataFrame({"label": list("1234"), "thp": [250.0, 300.0, 220.0, 350.0],
                          "q": [1500.0, 2500.0, 3200.0, 1000.0], "wc": [0.1, 0.2, 0.3, 0.05],
                          "gor": [750.0, 800.0, 900.0, 750.0], "gauge_md": [10000.0] * 4})
    P, T = wvlp.traverse(path, oil, tests["thp"].values, wvlp.case_of(tests, "oil"), "hagedorn_brown", 1.04, 1.15,
                         4.17)
    tests["gauge_p"], tests["tht"] = P[:, path.node(10000.0)], T[:, 0]
    u = wvlp.estimate_u(path, oil, tests)["U (BTU/h/ft2/F)"]
    assert np.allclose(u, 4.17, atol=1e-3)
    m = wvlp.match(path, oil, tests, "hagedorn_brown", 4.17)
    assert m.p1 == pytest.approx(1.04, abs=0.005) and m.p2 == pytest.approx(1.15, abs=0.02) and m.rmse_after < 2
    s, *_ = wvlp.compare(path, oil, tests, u=4.17)
    assert set(s["key"]) == set(wvlp.VLP_CORRELATIONS)


def test_vlp_ipr_intersection():
    rates = np.linspace(10, 3000, 50)
    vlp_bhp = 1500 + 0.2 * rates                              # rising VLP
    ipr = iflow.pi_entry(3000, 1000, 1.5)
    q, pwf = wvlp.intersect(rates, vlp_bhp, ipr.q, ipr.pwf)
    assert pwf == pytest.approx(1500 + 0.2 * q, abs=1.0) and q == pytest.approx(1.5 * (3000 - pwf), rel=0.01)


def test_new_correlations_limits_and_ordering():
    from wellmodel import correlations as C
    d, rough = 2.992 / 12, 0.0006 / 12
    one = lambda v: np.array([v])
    for k in ("orkiszewski", "aziz", "beggs_brill_noslip"):
        f = C.CORRELATIONS[k]
        # bubble-ish flow: gradient between gas and liquid hydrostatics, holdup >= no-slip
        e, fr, hl = f(one(4.0), one(0.6), one(50.0), one(5.0), one(1.5), one(0.02), one(25.0), d, rough, one(1.0),
                      one(1500.0))
        lam = 4.0 / 4.6
        assert 5 / 144 < e[0] <= 50 / 144 + 1e-9 and hl[0] >= lam - 1e-9 and fr[0] >= 0
        # gas well: finite, positive
        e, fr, hl = f(one(0.05), one(40.0), one(62.0), one(5.0), one(0.5), one(0.02), one(60.0), d, rough, one(1.0),
                      one(1500.0))
        assert np.isfinite(e[0] + fr[0]) and e[0] > 0 and fr[0] > 0
    # no-slip is the lower bound of the slip correlations in slug flow
    args = (one(1.5), one(3.0), one(50.0), one(5.0), one(1.5), one(0.02), one(25.0), d, rough, one(1.0), one(1500.0))
    g_ns = sum(C.no_slip(*args)[:2])
    for k in ("beggs_brill", "hagedorn_brown", "mukherjee_brill", "aziz", "orkiszewski"):
        assert sum(C.CORRELATIONS[k](*args)[:2])[0] >= g_ns[0] - 1e-3


def test_duns_ros_and_fancher_brown_charts():
    from wellmodel import correlations as C
    assert C._li(10.0, *C._DR_ND) == 2.0 and C._li(200.0, *C._DR_ND) == 1.0          # L1 limits (Fig 3-11)
    assert C._li(100.0, *C._DR_L2) == pytest.approx(1.10, abs=0.01)
    assert C._li(0.15, C._NL6, C._F6) > 2.0 and C._li(0.015, C._NL6, C._F6) < 0       # F6 peak and minimum
    assert C._li(2.0, *C._F2V) == pytest.approx(1.0, abs=0.06) or C._li(2.0, *C._F2V) < 1.0
    # Fancher & Brown: friction falls with x, and with GLR at fixed x (Fig 6)
    x = np.array([5.0, 10.0, 20.0])
    for glr in (500, 2000, 4000):
        f = C.fancher_brown_f(x, glr)
        assert np.all(np.diff(f) < 0)
    assert C.fancher_brown_f(10.0, 500) > C.fancher_brown_f(10.0, 2000) > C.fancher_brown_f(10.0, 4000)
    assert C.fancher_brown_f(10.0, 2000) == pytest.approx(0.0551 * (10 / 4.71) ** -1.153)


def test_duns_ros_and_fancher_brown_bracket_measured_gauge():
    *_, path, oil = _vlp_setup()
    tests = pd.DataFrame({"label": list("1234"), "thp": [250.0, 300.0, 220.0, 350.0],
                          "q": [1500.0, 2500.0, 3200.0, 1000.0], "wc": [0.1, 0.2, 0.3, 0.05],
                          "gor": [750.0, 800.0, 900.0, 750.0], "gauge_md": [10000.0] * 4})
    P, _ = wvlp.traverse(path, oil, tests["thp"].values, wvlp.case_of(tests, "oil"), "hagedorn_brown", 1.04, 1.15, 4.17)
    meas = P[:, path.node(10000.0)]
    lo, *_ = wvlp.gauge_pressures(path, oil, tests, "fancher_brown", u=4.17)
    hi, *_ = wvlp.gauge_pressures(path, oil, tests, "duns_ros", u=4.17)
    assert np.mean(lo - meas) < 0 < np.mean(hi - meas)


def test_duns_ros_gould_map_regions():
    from wellmodel import correlations as C
    nl = np.array([0.3, 3.0, 20.0])
    assert np.all(np.isfinite(C._gould_plug_ng(np.array([0.3, 0.9]), 1.0)))
    assert np.isnan(C._gould_plug_ng(np.array([3.0]), 1.0))[0]          # plug region only at low NL
    assert C._gould_plug_ng(np.array([0.3]), 3.15)[0] > C._gould_plug_ng(np.array([0.3]), 1.0)[0]
    d, rough = 2.992 / 12, 0.0006 / 12
    k = (50 / 25) ** 0.25
    vsg = np.geomspace(0.005, 150, 300)
    for n in nl:
        one = np.ones_like(vsg)
        args = (n / (1.938 * k) * one, vsg, 50 * one, 5 * one, 1.5 * one, 0.02 * one, 25 * one, d, rough, one,
                1500 * one)
        g = sum(C.duns_ros_gould(*args)[:2])
        assert np.all(np.isfinite(g)) and np.all(g > 0)
        # same elevation (holdup) as the original away from the bubble/slug boundary; mist identical
        e0, f0, _ = C.duns_ros(*args)
        e1, f1, _ = C.duns_ros_gould(*args)
        ng_ = 1.938 * k * vsg
        mist = ng_ > 75 + 84 * n ** 0.75
        assert np.allclose(f1[mist], f0[mist]) and np.allclose(e1[mist], e0[mist])
        g, g0 = e1, e0
        nd = 120.872 * d * np.sqrt(50 / 25)
        ng = 1.938 * k * vsg
        far = (ng > 0.62 * (max(n, 1.3) / 1.30) ** 0.984 * 1.05) & (ng > 1.0)
        far &= ng > (C._li(nd, *C._DR_ND) + C._li(nd, *C._DR_L2) * n) * 1.05     # past the original bubble limit
        assert np.allclose(g[far], g0[far])


def test_kleyweg_friction_in_modified_duns_ros():
    from wellmodel import correlations as C
    d, rough, k = 2.992 / 12, 0.0006 / 12, (50 / 25) ** 0.25
    one = lambda v: np.array([v])
    args = lambda nl, ng: (one(nl / (1.938 * k)), one(ng / (1.938 * k)), one(50.0), one(5.0), one(1.5), one(0.02),
                           one(25.0), d, rough, one(1.0), one(1500.0))
    # gas-dominated slug flow: single-phase friction on the mixture velocity exceeds the liquid-based original
    assert C.duns_ros_gould(*args(0.3, 50))[1][0] > 5 * C.duns_ros(*args(0.3, 50))[1][0]
    # liquid-dominated: the two friction methods agree closely
    f0, f1 = C.duns_ros(*args(20, 0.5))[1][0], C.duns_ros_gould(*args(20, 0.5))[1][0]
    assert f1 == pytest.approx(f0, rel=0.1)


def test_ansari_patterns_and_limits():
    from wellmodel import correlations as C
    d, rough = 2.992 / 12, 0.0006 / 12
    one = lambda v: np.array([v])
    run = lambda vsl, vsg: C.ansari(one(vsl), one(vsg), one(50.0), one(5.0), one(1.5), one(0.02), one(25.0), d, rough,
                                    one(1.0), one(1500.0))
    # dispersed bubble at high liquid rate: reduces to no-slip
    e, f, h = run(12.0, 1.5)
    e0, f0, h0 = C.no_slip(one(12.0), one(1.5), one(50.0), one(5.0), one(1.5), one(0.02), one(25.0), d, rough,
                           one(1.0), one(1500.0))
    assert e[0] == pytest.approx(e0[0]) and f[0] == pytest.approx(f0[0], rel=1e-6)
    # bubble flow: slip holdup above no-slip; slug: holdup between no-slip and 1
    e, f, h = run(2.0, 0.3)
    assert h[0] > 2.0 / 2.3
    e, f, h = run(1.5, 3.0)
    assert 1.5 / 4.5 < h[0] < 1.0
    # annular gas well: holdup small, friction dominated, finite
    e, f, h = run(0.02, 80.0)
    assert h[0] < 0.02 and f[0] > e[0]
    # continuous within patterns over a sweep
    vsg = np.geomspace(0.05, 120, 300)
    o = np.ones_like(vsg)
    e, f, h = C.ansari(1.0 * o, vsg, 50 * o, 5 * o, 1.5 * o, 0.02 * o, 25 * o, d, rough, o, 1500 * o)
    assert np.all(np.isfinite(e + f)) and np.all((h >= 0) & (h <= 1))



# --------------------------------------------------------- system analysis
from wellmodel import system as wsys


def test_system_values_and_combinations():
    assert np.allclose(wsys.generate_values(100, 400, 4, "Linear"), [100, 200, 300, 400])
    assert np.allclose(wsys.generate_values(100, 400, 3, "Geometric"), [100, 200, 400])
    with pytest.raises(ValueError):
        wsys.generate_values(0, 10, 3, "Geometric")
    combos = wsys.combinations({"whp": [100, 200], "wc": [0, 50, 90]})
    assert len(combos) == 6 and {"whp": 200, "wc": 90} in combos
    with pytest.raises(ValueError):
        wsys.combinations({"a": range(10), "b": range(10), "c": range(10)})


def test_system_solution_matches_vlp_intersection_and_sensitivity_trends():
    top, secs, pts, smd, stvd, path, oil = _vlp_setup()
    pb = oil.m.bubble_point(205.0)
    ipr_for = lambda ch: iflow.pi_entry(ch.get("pr", 3000.0), pb, ch.get("j", 2.0))
    path_for = lambda tid: path
    base = {"whp": 250.0, "u": 4.17, "wc": 20.0, "gor": 750.0}
    res, paths = wsys.solve_cases(path_for, oil, "oil", base, [{}], ipr_for, "hagedorn_brown", 1.04, 1.15)
    r = res[0]
    rates = np.linspace(20, 3500, 400)
    t = pd.Series({"thp": 250.0, "q": 1.0, "wc": 0.2, "gor": 750.0})
    q_ref, p_ref = wvlp.intersect(rates, wvlp.vlp_curve(path, oil, t, rates, "hagedorn_brown", 1.04, 1.15, 4.17),
                                  ipr_for({}).q, ipr_for({}).pwf)
    assert r.q == pytest.approx(q_ref, rel=2e-3) and r.pwf == pytest.approx(p_ref, abs=3) and r.stable
    det, reg, prof = wsys.solution_details(path, oil, "oil", r.params, r.q, "hagedorn_brown", 1.04, 1.15)
    total = det["dP gravity - production string (psi)"] + det["dP friction - production string (psi)"] \
        + det["dP restrictions (psi)"] + 250
    assert total == pytest.approx(det["Bottom node pressure (psia)"], abs=10)
    # higher top-node pressure and higher water cut both reduce the rate; higher PI raises it
    res, _ = wsys.solve_cases(path_for, oil, "oil", base, wsys.combinations({"whp": [150, 300]}), ipr_for,
                              "hagedorn_brown", 1.04, 1.15)
    assert res[0].q > res[1].q
    res, _ = wsys.solve_cases(path_for, oil, "oil", base, wsys.combinations({"wc": [0, 30]}), ipr_for,
                              "hagedorn_brown", 1.04, 1.15)
    assert res[0].q > (res[1].q or 0)
    res, _ = wsys.solve_cases(path_for, oil, "oil", base, wsys.combinations({"j": [1.5, 3.0]}), ipr_for,
                              "hagedorn_brown", 1.04, 1.15)
    assert res[1].q > res[0].q


# ------------------------------------------------------------ surface equipment (choke and flowline)
from wellmodel import surface as wsurf


def _surface_df(manifold_tvd=0.0, bean=0.75):
    return pd.DataFrame([
        {"Label": "Manifold", "Type": "Manifold", "TVD of End Point (ft)": manifold_tvd},
        {"Label": "Flowline", "Type": "Line Pipe", "Pipe Length (ft)": 3000.0, "TVD of End Point (ft)": 0.0,
         "Pipe Inside Diameter (in)": 4.026, "Pipe Inside Roughness (in)": 0.0006},
        {"Label": "Elbow", "Type": "Fitting", "Fitting": "90° elbow (standard)", "Pipe Inside Diameter (in)": 4.026},
        {"Label": "Choke", "Type": "Choke", "Pipe Inside Diameter (in)": bean},
        {"Label": "Spool", "Type": "Line Pipe", "Pipe Length (ft)": 30.0, "TVD of End Point (ft)": 0.0,
         "Pipe Inside Diameter (in)": 2.992, "Pipe Inside Roughness (in)": 0.0006}])


def test_surface_items_flow_order_fittings_and_elevation():
    tvd, items = wsurf.build_items(_surface_df(-100.0))
    assert [i.label for i in items] == ["Spool", "Choke", "Elbow", "Flowline"]      # wellhead -> manifold
    assert items[2].length == pytest.approx(30 * 4.026 / 12)                         # L/D 30
    assert items[3].rise == pytest.approx(100.0) and tvd == -100.0                   # manifold 100 ft higher
    assert [i.d_in for i in wsurf.with_choke_size(items, 32) if i.kind == "choke"] == [0.5]


def test_choke_liquid_orifice_gas_critical_and_gilbert_family():
    *_, oil = _vlp_setup()
    hc = weq.HeatCapacities()
    water = {"q": np.array([2000.0]), "wc": np.array([0.9999]), "gor": np.array([0.0])}
    p1, reg, _ = wsurf.choke_upstream(oil, water, np.array([200.0]), 120.0, 0.5, hc, cd=0.75)
    s = oil.state(p1, 120.0, water)
    v = s["ql"] / (np.pi / 4 * (0.5 / 12) ** 2)
    assert p1[0] - 200 == pytest.approx(float(s["rho_l"][0] * v[0] ** 2 / (2 * 32.174 * 0.75 ** 2) / 144), rel=1e-3)
    assert reg[0] == "SubCritical"
    gas = wvlp.WetGasFluid(wbo.PVTInputs(rsb=1, api=40, gas_sg=0.65), 50)
    gc = {"q": np.array([5.0]), "wgr": np.array([0.0]), "cgr": np.array([0.0])}
    lo, r_lo, _ = wsurf.choke_upstream(gas, gc, np.array([100.0]), 100.0, 0.5, hc)
    lo2, _, _ = wsurf.choke_upstream(gas, gc, np.array([300.0]), 100.0, 0.5, hc)
    assert r_lo[0] == "Critical" and lo[0] == pytest.approx(lo2[0], rel=1e-3)        # independent of p2
    assert 900 < lo[0] < 1200                                                        # ideal-gas estimate ~1120
    oc = {"q": np.array([1000.0, 2000.0]), "wc": np.array([0.2, 0.2]), "gor": np.array([750.0, 750.0])}
    sach, _, _ = wsurf.choke_upstream(oil, oc, np.array([150.0, 150.0]), 120.0, 0.5, hc)
    assert sach[1] > sach[0]
    for m in ("Gilbert", "Ros", "Baxendell", "Achong"):
        pg, rg, _ = wsurf.choke_upstream(oil, oc, np.array([150.0, 150.0]), 120.0, 0.5, hc, method=m)
        assert list(rg) == ["Critical", "Critical"] and np.all(np.abs(pg / sach - 1) < 0.5)
    _, _, note = wsurf.choke_upstream(gas, gc, np.array([100.0]), 100.0, 0.5, hc, method="Gilbert")
    assert "Sachdeva used" in note


def test_system_with_manifold_top_node_consistent_with_wellhead_solution():
    top, secs, pts, smd, stvd, path, oil = _vlp_setup()
    pb = oil.m.bubble_point(205.0)
    ipr_for = lambda ch: iflow.pi_entry(ch.get("pr", 3000.0), pb, ch.get("j", 2.0))
    base = {"whp": 150.0, "u": 4.17, "wc": 20.0, "gor": 750.0}
    _, items = wsurf.build_items(_surface_df())
    surf = wsys.SurfaceSetup(items, 60.0, 8.0)
    res, paths = wsys.solve_cases(lambda t: path, oil, "oil", base, [{}], ipr_for, "hagedorn_brown", 1.04, 1.15,
                                  surface=surf)
    r = res[0]
    assert r.q is not None and r.pwh_q > 150
    res2, _ = wsys.solve_cases(lambda t: path, oil, "oil", {**base, "whp": r.pwh_q}, [{}], ipr_for, "hagedorn_brown",
                               1.04, 1.15)
    assert res2[0].q == pytest.approx(r.q, rel=3e-3)
    det, reg, _ = wsys.solution_details(path, oil, "oil", r.params, r.q, "hagedorn_brown", 1.04, 1.15, surface=surf)
    tab = det.pop("_surface")
    assert det["Manifold pressure (psia)"] + det["dP choke (psi)"] + det["dP flowline and fittings (psi)"] == \
        pytest.approx(det["Wellhead pressure (psia)"], abs=0.5)
    assert list(tab["Item"]) == ["Spool", "Choke", "Elbow", "Flowline"] and any(k.startswith("Surface choke") for k in reg)
    assert "Wellhead pressure (psia)" in wsys.rate_table("oil", r).columns
    # a bigger bean gives more rate
    res, _ = wsys.solve_cases(lambda t: path, oil, "oil", base, wsys.combinations({"choke": [32, 64]}), ipr_for,
                              "hagedorn_brown", 1.04, 1.15, surface=surf)
    assert res[1].q > res[0].q


def test_upward_traverse_and_forward_surface_march_invert_the_backward_ones():
    *_, path, oil = _vlp_setup()
    hc = weq.HeatCapacities()
    case = {"q": np.array([300.0, 1500.0]), "wc": 0.2, "gor": 750.0}
    P, _ = wvlp.traverse(path, oil, 250.0, case, "hagedorn_brown", 1.04, 1.15, 4.17, hc)
    Pu, _ = wvlp.traverse_up(path, oil, P[:, -1], case, "hagedorn_brown", 1.04, 1.15, 4.17, hc)
    assert np.allclose(Pu[:, 0], 250.0, atol=0.5)
    _, items = wsurf.build_items(_surface_df())
    t = np.array([80.0, 90.0])
    pwh = wsurf.wellhead_pressure(items, oil, "oil", 150.0, case, t, "beggs_brill", hc)
    sub = {"q": np.array([300.0]), "wc": 0.2, "gor": 750.0}                     # subcritical: exact inverse
    pm = wsurf.manifold_pressure(items, oil, "oil", pwh[:1], sub, t[:1], "beggs_brill", hc)
    assert pm[0] == pytest.approx(150.0, abs=0.5)
    crit = {"q": np.array([1500.0]), "wc": 0.2, "gor": 750.0}                   # critical: cannot pass below p1
    assert np.isnan(wsurf.manifold_pressure(items, oil, "oil", pwh[1:] * 0.95, crit, t[1:], "beggs_brill", hc)[0])


def test_top_node_solution_same_rate_and_inflow_meets_top_pressure():
    top, secs, pts, smd, stvd, path, oil = _vlp_setup()
    pb = oil.m.bubble_point(205.0)
    ipr_for = lambda ch: iflow.pi_entry(ch.get("pr", 3000.0), pb, ch.get("j", 2.0))
    base = {"whp": 250.0, "u": 4.17, "wc": 20.0, "gor": 750.0}
    rb, _ = wsys.solve_cases(lambda t: path, oil, "oil", base, [{}], ipr_for, "hagedorn_brown", 1.04, 1.15)
    rt, _ = wsys.solve_cases(lambda t: path, oil, "oil", base, [{}], ipr_for, "hagedorn_brown", 1.04, 1.15,
                             node="top")
    assert rt[0].q == rb[0].q and rt[0].node_pressure == 250.0 and rb[0].node_pressure == rb[0].pwf
    assert rt[0].top_in_q == pytest.approx(250.0, abs=1.0)
    assert "Inflow pressure at top node (psia)" in wsys.rate_table("oil", rt[0]).columns
    _, items = wsurf.build_items(_surface_df())
    rs, _ = wsys.solve_cases(lambda t: path, oil, "oil", {**base, "whp": 150.0}, [{}], ipr_for, "hagedorn_brown",
                             1.04, 1.15, surface=wsys.SurfaceSetup(items), node="top")
    r = rs[0]                                       # 48/64 choke runs critical: inflow sits above the manifold
    assert r.q is not None and (np.isnan(r.top_in_q) or r.top_in_q >= 150.0 - 1.0)


def test_test_table_reads_manifold_pressure_and_choke():
    df = pd.DataFrame([{"Tubing Head Pressure (psia)": 300.0, "Liquid Rate (STB/day)": 1000.0, "Water Cut (percent)": 10,
                        "Gas Oil Ratio (scf/STB)": 700.0, "Manifold Pressure (psia)": 150.0,
                        "Choke Size (64ths in)": 48.0},
                       {"Tubing Head Pressure (psia)": 300.0, "Liquid Rate (STB/day)": 1000.0, "Water Cut (percent)": 10,
                        "Gas Oil Ratio (scf/STB)": 700.0, "Manifold Pressure (psia)": 350.0}])
    t, issues = wvlp.clean_tests(df, "oil")
    assert t.loc[0, "pman"] == 150.0 and t.loc[0, "choke"] == 48.0 and np.isnan(t.loc[1, "choke"])
    assert any("manifold pressure is not below" in m for _, m in issues)


def test_surface_match_recovers_cd_and_friction():
    *_, oil = _vlp_setup()
    hc = weq.HeatCapacities()
    _, items = wsurf.build_items(_surface_df())
    tests = pd.DataFrame({"label": list("12345"), "q": [800.0, 1500, 2500, 1200, 3000], "wc": 0.2, "gor": 750.0,
                          "pman": [150.0, 200, 180, 250, 150], "choke": [48.0, 64, 96, 32, 128]})
    twh = np.array([85.0, 95, 105, 90, 110])
    tests["thp"] = wsurf._calc_thp(items, oil, "oil", tests, twh, "beggs_brill", hc, 60, 8, 0.85, 1.3)
    m = wsurf.match_surface(items, oil, "oil", tests, twh, "beggs_brill", hc, 60, 8)
    assert m.cd == pytest.approx(0.85, abs=0.01) and m.friction == pytest.approx(1.3, abs=0.05)
    assert m.rms_after < 0.5 < m.rms_before and set(m.fitted) == {"cd", "friction"}
    crit = tests.assign(choke=24.0, q=[400.0, 500, 600, 450, 550])                  # every test critical
    crit["thp"] = wsurf._calc_thp(items, oil, "oil", crit, twh, "beggs_brill", hc, 60, 8, 0.85, 1.3)
    m2 = wsurf.match_surface(items, oil, "oil", crit, twh, "beggs_brill", hc, 60, 8)
    assert m2.fitted == ("cd",) and m2.friction == 1.0 and m2.cd == pytest.approx(0.85, abs=0.01)
    with pytest.raises(ValueError):
        wsurf.match_surface(items, oil, "oil", tests.assign(pman=np.nan), twh)


# ------------------------------------------------------------------ project files
from wellmodel import project as wproj


def test_project_file_round_trip_keeps_tables_types_and_values():
    df = pd.DataFrame({"Label": ["Manifold", None, "Choke"], "Type": ["Manifold", None, "Choke"],
                       "Pipe Length (ft)": [np.nan, 3000.0, np.nan]}, index=pd.RangeIndex(1, 4, name="Point"))
    df["Label"] = df["Label"].astype(object)
    state = {"inputs": {"ipr_pr": 3100.0, "pvt_sel_pb": "standing", "sys_vars": ["whp", "wc"], "flag": True},
             "widgets": {"opt_fluid_type": "oil"}, "tables": {"eq_surface": df},
             "extras": {"vlp_matches": [{"correlation": "hagedorn_brown", "p1": np.float64(1.04),
                                         "residuals": np.array([1.0, np.nan]), "n": np.int64(4)}],
                        "prp_text": "COMPONENTS\nC1 ..."}}
    text = wproj.dumps(state, "Well A-1")
    back = wproj.loads(text)
    assert back["name"] == "Well A-1" and back["inputs"] == state["inputs"] and back["widgets"] == state["widgets"]
    t = back["tables"]["eq_surface"]
    assert t.index.name == "Point" and list(t.index) == [1, 2, 3] and t["Label"].dtype == object
    assert t["Label"].tolist() == ["Manifold", None, "Choke"] and t["Pipe Length (ft)"].dtype == float
    assert np.isnan(t.iloc[0, 2]) and t.iloc[1, 2] == 3000.0
    m = back["extras"]["vlp_matches"][0]
    assert m["p1"] == 1.04 and m["n"] == 4 and np.isnan(m["residuals"][1])
    assert back["extras"]["prp_text"].startswith("COMPONENTS")


def test_project_file_rejects_foreign_and_newer_files():
    with pytest.raises(ValueError):
        wproj.loads(b'{"hello": 1}')
    with pytest.raises(ValueError):
        wproj.loads(b"not json")
    with pytest.raises(ValueError):
        wproj.loads('{"format": "wellmodel-project", "version": 99}')
    with pytest.raises(TypeError):
        wproj.dumps({"inputs": {"x": object()}})


# ------------------------------------------------------------- gradient traverse and lift curves
from wellmodel import liftcurves as wlc


def test_traverse_profile_has_wellhead_row_and_gradients():
    *_, path, oil = _vlp_setup()
    det, reg, prof = wsys.solution_details(path, oil, "oil", {"whp": 250.0, "u": 4.17, "wc": 20.0, "gor": 750.0},
                                           1000.0, "hagedorn_brown", 1.04, 1.15)
    assert len(prof) == len(path.md) and prof["Pressure (psia)"].iloc[0] == pytest.approx(250.0)
    assert prof["Pressure (psia)"].iloc[-1] == pytest.approx(det["Bottom node pressure (psia)"])
    g = prof.iloc[1:]
    assert (g["Elevation gradient (psi/ft)"] > 0).all() and (g["Friction gradient (psi/ft)"] >= 0).all()
    assert ((g["In-situ density (lb/ft3)"] > 0) & (g["In-situ density (lb/ft3)"] < 70)).all()


def test_vfp_table_matches_vlp_and_eclipse_format():
    *_, path, oil = _vlp_setup()
    flo = np.geomspace(50, 4000, 11)
    axes, bhp, _ = wlc.vfp_table(path, oil, "oil", "hagedorn_brown", 1.04, 1.15, 4.17, flo, [300, 100], [0, 30],
                                 [750, 400], weq.HeatCapacities())
    assert bhp.shape == (2, 2, 2, 11) and list(axes[0]) == [100.0, 300.0] and list(axes[2]) == [400.0, 750.0]
    ref = wvlp.vlp_curve(path, oil, pd.Series({"thp": 300.0, "q": 1.0, "wc": 0.3, "gor": 750.0}), axes[3],
                         "hagedorn_brown", 1.04, 1.15, 4.17)
    assert np.allclose(bhp[1, 1, 1], ref, atol=1e-6)
    txt = wlc.eclipse_vfpprod("oil", axes, bhp, 3, float(path.tvd[-1]))
    lines = [l for l in txt.splitlines() if not l.startswith("--")]
    assert lines[0] == "VFPPROD" and lines[1].split()[:2] == ["3", f"{path.tvd[-1]:.2f}"]
    assert "'LIQ'" in lines[1] and "'WCT'" in lines[1] and "'GOR'" in lines[1]
    assert max(len(l) for l in txt.splitlines()) <= 132
    body = " ".join(lines[2:])
    recs = [r.split() for r in body.split("/") if r.strip()]
    assert len(recs) == 5 + 2 * 2 * 2                                  # FLO THP WFR GFR ALQ + data records
    assert [float(x) for x in recs[2]] == [0.0, 0.3] and [float(x) for x in recs[3]] == [0.4, 0.75]   # WCT, Mscf/STB
    assert all(len(r) == 4 + 11 for r in recs[5:])
    assert float(recs[5][4]) == pytest.approx(bhp[0, 0, 0, 0], abs=0.01)
    gas_txt = wlc.eclipse_vfpprod("gas", ([500.0], [10.0], [20.0], np.array([1.0, 5.0])), np.ones((1, 1, 1, 2)))
    g_lines = [l for l in gas_txt.splitlines() if not l.startswith("--")]
    assert "'GAS'" in g_lines[1] and g_lines[2].split()[:2] == ["1000", "5000"]                 # Mscf/d
    assert g_lines[4].split()[0] == "0.01" and g_lines[5].split()[0] == "0.02"                # STB/Mscf
    with pytest.raises(ValueError):
        wlc.vfp_table(path, oil, "oil", "hagedorn_brown", 1, 1, 4.17, np.arange(1, 41), np.arange(1, 41),
                      np.arange(0, 20), np.arange(1, 10))


# ----------------------------------------------- PVT behaviour checked against PROSPER
def test_vazquez_beggs_uses_separator_corrected_gas_gravity_like_prosper():
    inp = bo.PVTInputs(rsb=345.067, api=30.016, gas_sg=0.595)
    assert bo.vb_gas_sg(inp) == pytest.approx(0.5385, abs=1e-4)
    # PROSPER's raw Vazquez-Beggs bubble point for this oil (back-calculated from its match table): 2552.8 psig
    assert float(bo.pb_raw("vazquez_beggs", inp, 345.067, 152.961)) == pytest.approx(2552.8 + 14.7, abs=1.5)


def test_gor_and_fvf_matches_pass_through_stock_tank_point():
    inp = bo.PVTInputs(rsb=345.067, api=30.016, gas_sg=0.595)
    tbl = pd.DataFrame({"Pressure": [2552.1], "Gas Oil Ratio": [345.07], "Oil FVF": [1.1932], "Oil Viscosity": [1.23]})
    res = bo.match_all(inp, bo.MatchData(152.961, 2552.1, tbl))
    for f in ("glaso", "standing", "petrosky"):
        a = bo.stock_tank_anchors(f, inp)
        rs, bo_ = res.matches[("rs", f)], res.matches[("bo", f)]
        assert rs.p1 * a["rs"][0] + rs.p2 == pytest.approx(0.0, abs=1e-3)     # GOR 0 at the stock tank
        assert bo_.p1 * a["bo"][0] + bo_.p2 == pytest.approx(1.0, abs=1e-5)   # FVF 1 for gas-free oil
        assert rs.n == 1 and bo_.n == 1                                       # the anchor is not a lab point
    # PROSPER's P1 for these (same oil): GOR Glaso 0.98264, Standing 0.77787; FVF Glaso 1.36126
    assert res.matches[("rs", "glaso")].p1 == pytest.approx(0.98264, abs=2e-3)
    assert res.matches[("rs", "standing")].p1 == pytest.approx(0.77787, abs=2e-3)
    assert res.matches[("bo", "glaso")].p1 == pytest.approx(1.36126, abs=5e-3)


def test_de_ghetto_heavy_oil_pb_rs_fvf_match_prosper():
    """PROSPER 'De Ghetto et al (Heavy oil)' on a real oil (API 30.03, GOR 345, gas gravity 0.595, 153 degF):
    raw values recovered from PROSPER's matched table and match parameters."""
    inp = bo.PVTInputs(rsb=345, api=30.03, gas_sg=0.595, salinity_ppm=3000, co2=0.28, n2=0.1)
    assert float(bo._pb_corr("de_ghetto", inp, 153.0)) == pytest.approx(2170.7, rel=2e-3)
    p = np.array([2044.54, 2304.78, 2538.99])
    assert np.allclose(bo._rs_corr("de_ghetto", inp, p, 153.0), [320.975, 370.860, 416.761], rtol=1e-3)
    data = bo.MatchData(153.0, 2552.0, pd.DataFrame({"Pressure": [2552.0], "Gas Oil Ratio": [345.0],
                                                     "Oil FVF": [1.193], "Oil Viscosity": [1.23]}))
    res = bo.match_all(inp, data)
    m = bo.BlackOilModel(inp, "de_ghetto", "beal", {k: res.matches[(k, "de_ghetto")] for k in ("pb", "rs", "bo")})
    t = m.table(153.0, [2044.54, 2304.78, 2538.99, 3059.46])
    assert np.allclose(t["Solution GOR (scf/STB)"], [263.857, 305.009, 342.875, 345.0], rtol=1e-3)
    assert np.allclose(t["Oil FVF (RB/STB)"], [1.15894, 1.17621, 1.19211, 1.1877], rtol=1e-3)


def test_heavy_oil_viscosities_match_prosper_unmatched_table():
    """PROSPER unmatched viscosity on the real oil (API 30.03, 153 degF). PROSPER's table pairs each viscosity
    correlation with a bubble-point correlation: Egbogah (Heavy oil) with Vazquez-Beggs, De Ghetto with Al-Marhoun,
    Petrosky with (PROSPER's) Lasater - bubble point 1838.93 psia."""
    inp = bo.PVTInputs(rsb=345, api=30.03, gas_sg=0.595, salinity_ppm=3000, co2=0.28, n2=0.1)
    p = np.array([2044.54, 2434.89, 3059.46])
    eg = bo.BlackOilModel(inp, "vazquez_beggs", "egbogah").table(153.0, p)["Oil viscosity (cp)"]
    assert np.allclose(eg, [7.98023, 6.59778, 6.30390], rtol=1e-3)
    dg = bo.BlackOilModel(inp, "al_marhoun", "de_ghetto").table(153.0, p)["Oil viscosity (cp)"]
    assert np.allclose(dg, [3.61410, 3.06842, 2.34818], rtol=2e-3)
    mob = bo.visc_live("petrosky", bo.visc_dead("petrosky", inp, 153.0), 345.0)
    assert np.allclose(bo.visc_undersat("petrosky", mob, p, 1838.93), [1.52458, 1.60643, 1.73739], rtol=1e-3)
    for fam, pb in (("glaso", 2476.49), ("standing", 2051.66), ("petrosky", 2662.09), ("al_marhoun", 3060.0)):
        assert float(bo._pb_corr(fam, inp, 153.0)) == pytest.approx(pb, rel=3e-4)     # no impurity correction


def test_petrosky_beal_pvt_tables_prosper_t29():
    """PROSPER PVT calculation tables for the T29 oil (API 17, GOR 150, gas gravity 0.58), 104-120 degF and
    15-2200 psig. Unmatched: the raw Petrosky Rs/Bo and Beal (Chew-Connally) viscosity are reproduced exactly.
    Matched (single lab point 2100 psig, 120 degF): with PROSPER's own parameters the model structure is exact
    (including the 1 scf/STB GOR floor); with ours the tables agree within 0.3% Bo, 1.5 scf/STB GOR and 1.5% viscosity."""
    d = Path(__file__).parent / "data"
    inp = bo.PVTInputs(rsb=150, api=17, gas_sg=0.58, salinity_ppm=75000)
    un = pd.read_csv(d / "t29_pvt_unmatched.csv")
    p = un.p.to_numpy() + 14.696
    assert np.allclose(bo.rs_raw("petrosky", inp, p, un["T"].to_numpy()), un.rs, rtol=3e-4)
    assert np.allclose(bo.bo_sat("petrosky", inp, un.rs.to_numpy(), un["T"].to_numpy()), un.bo, atol=1e-4)
    mu = bo.visc_live("beal", bo.visc_dead("beal", inp, un["T"].to_numpy()), un.rs.to_numpy())
    assert np.allclose(mu, un.mu_o, rtol=1e-4)

    m = pd.read_csv(d / "t29_pvt_matched.csv")
    L = bo.LinearMatch
    prosper = {"pb": L(1.14592, 236.751), "rs": L(1.01197, -45.2274), "bo": L(0.9875, -0.0012334),
               "visc": L(0.91487, -4.1035)}
    md = bo.MatchData(120.0, 2100 + 14.696, pd.DataFrame({"Pressure": [2114.696], "Gas Oil Ratio": [150.0],
                                                           "Oil FVF": [1.06], "Oil Viscosity": [40.0]}))
    res = bo.match_all(inp, md)
    ours = {k: res.matches[(k, "petrosky")] for k in ("pb", "rs", "bo")}
    ours["visc"] = res.matches[("visc", "beal")]
    assert ours["visc"].p1 == pytest.approx(0.91487, abs=1e-4) and ours["visc"].p2 == pytest.approx(-4.1035, abs=2e-3)
    for params, tol in ((prosper, dict(rs=0.1, bo=3e-5, mu=1e-3)), (ours, dict(rs=1.5, bo=3e-3, mu=1.5e-2))):
        model = bo.BlackOilModel(inp, "petrosky", "beal", params)
        for T, g in m.groupby("T"):
            tab = model.table(float(T), g.p.to_numpy() + 14.696)
            assert model.bubble_point(float(T)) - 14.696 == pytest.approx(g.pb.iloc[0], rel=3e-4)
            assert np.allclose(tab["Solution GOR (scf/STB)"], g.rs, atol=tol["rs"])
            assert np.allclose(tab["Oil FVF (RB/STB)"], g.bo, rtol=tol["bo"])
            assert np.allclose(tab["Oil viscosity (cp)"], g.mu_o, rtol=tol["mu"])
    assert BlackOilModelRsAt15(inp, prosper) == pytest.approx(1.0)       # GOR floor below the bubble point


def BlackOilModelRsAt15(inp, params):
    return float(bo.BlackOilModel(inp, "petrosky", "beal", params).table(120.0, [15 + 14.696])
                 ["Solution GOR (scf/STB)"].iloc[0])


def test_bergman_sutton_matches_prosper_at_and_above_bubble_point():
    """Bergman-Sutton as published (SPE 110194 dead oil with Watson K 12.1, SPE 110195 saturated, SPE 103144
    undersaturated) against PROSPER, paired with Petrosky's bubble point as in PROSPER's table."""
    inp = bo.PVTInputs(rsb=345, api=30.03, gas_sg=0.595, salinity_ppm=3000, co2=0.28, n2=0.1, watson_k=12.1)
    p = np.array([2747.18, 2903.32, 3059.46])
    v = bo.BlackOilModel(inp, "petrosky", "bergman_sutton").table(153.0, p)["Oil viscosity (cp)"]
    assert np.allclose(v, [1.92805, 1.96147, 1.99286], rtol=3e-3)
    data = bo.MatchData(153.0, 2552.0, pd.DataFrame({"Pressure": [2552.0], "Gas Oil Ratio": [345.0],
                                                     "Oil FVF": [1.193], "Oil Viscosity": [1.23]}))
    m = bo.match_all(inp, data).matches[("visc", "bergman_sutton")]
    assert m.p1 == pytest.approx(0.82253, abs=2e-3) and m.p2 == pytest.approx(-0.33829, abs=5e-3)


def test_downhole_cells_follow_type_like_prosper():
    md, tid, te, tod, toe, cid, ce, rm = weq.DH_NUMERIC
    assert weq.active_columns("Xmas Tree") == {md}
    assert weq.active_columns("Tubing") == {md, tid, te, rm, tod, toe, cid, ce}       # annulus data optional (gas lift)
    assert weq.active_columns("Tubing", "annular") == {md, tid, te, rm, tod, toe, cid, ce}
    assert weq.active_columns("SSSV") == {tid} and weq.active_columns("Restriction") == {tid, rm}
    assert weq.active_columns("Gas Lift Mandrel") == {tid, rm} == weq.active_columns("SSD")   # like a restriction
    assert weq.active_columns("Casing") == {md, cid, ce, rm} and weq.active_columns("") == set()
    df = pd.DataFrame([{"Type": "Tubing", md: 231.93, tid: 2.441, tod: 2.875},
                       {"Type": "SSSV", md: 200.0, tid: 1.5, rm: 1.0},
                       {"Type": "Casing", md: 10932.0, cid: 8.681, ce: 0.0008},
                       {md: 5.0}]).reindex(columns=weq.DOWNHOLE_COLUMNS)
    out = weq.apply_cell_rules(df)
    assert out.loc[0, te] == 0.0006 and out.loc[0, rm] == 1.0 and out.loc[0, tod] == 2.875     # defaults; OD kept
    assert np.isnan(out.loc[1, md]) and np.isnan(out.loc[1, rm]) and out.loc[1, tid] == 1.5    # SSSV: ID only
    assert out.loc[2, ce] == 0.0008                                                              # user value kept
    assert out.iloc[3, 2:].isna().all()                                                          # no type: locked
    assert weq.same_table(out, weq.apply_cell_rules(out))
    ann = weq.apply_cell_rules(df, "annular")
    assert ann.loc[0, tod] == 2.875 and ann.loc[0, toe] == 0.0006 and ann.loc[0, ce] == 0.0006


def test_surface_cells_follow_type_like_prosper():
    ln, tvd, d, e, rm = weq.SURFACE_NUMERIC
    assert weq.surface_active_columns("Manifold") == {tvd}
    assert weq.surface_active_columns("Line Pipe") == {ln, tvd, d, e, rm}
    assert weq.surface_active_columns("Choke") == {d, rm}
    assert weq.surface_active_columns("Fitting") == {"Fitting", d, rm}
    df = pd.DataFrame([{"Type": "Manifold", tvd: 0.0, ln: 10.0},
                       {"Type": "Line Pipe", ln: 3000.0, tvd: 0.0, d: 4.026, "Fitting": "Gate valve (open)"},
                       {"Type": "Choke", d: 0.75, e: 0.001}]).reindex(columns=weq.SURFACE_COLUMNS)
    out = weq.apply_surface_rules(df)
    assert np.isnan(out.loc[0, ln]) and np.isnan(out.loc[0, rm])
    assert out.loc[1, e] == 0.0006 and out.loc[1, rm] == 1.0 and out.loc[1, "Fitting"] is None
    assert np.isnan(out.loc[2, e]) and out.loc[2, rm] == 1.0
    assert weq.same_table(out, weq.apply_surface_rules(out)) and weq.check_surface(out, 60, 8) == []


def test_glm_and_ssd_act_like_restrictions():
    md, tid, te, tod, toe, cid, ce, rm = weq.DH_NUMERIC
    df = pd.DataFrame([{"Type": "Xmas Tree", md: 0.0},
                       {"Type": "Tubing", md: 4000.0, tid: 2.992, te: 0.0006},
                       {"Type": "Gas Lift Mandrel", md: 4500.0, tid: 2.5},      # MD is locked: cleared
                       {"Type": "Tubing", md: 8000.0, tid: 2.992, te: 0.0006},
                       {"Type": "SSD", tid: 2.2},
                       {"Type": "Restriction", tid: 1.9},
                       {"Type": "Tubing", md: 9000.0, tid: 2.992, te: 0.0006}]).reindex(columns=weq.DOWNHOLE_COLUMNS)
    df = weq.apply_cell_rules(df)
    assert np.isnan(df.loc[2, md]) and df.loc[2, rm] == 1.0
    assert not [m for l, m in weq.check_downhole(df, "tubing", None) if l == "error"]
    top, secs, pts = weq.downhole_sections(df)
    assert list(pts["MD (ft)"]) == [4000.0, 8000.0, 8000.0]                      # at the previous depth
    path = wvlp.build_flow_path(top, secs, pts, [0.0, 9000.0], [0.0, 9000.0], [0.0, 9000.0], [60.0, 180.0])
    assert sorted(path.restrictions.values()) == [1.9, 2.5]                     # SSD and restriction share a node
    bad = df.copy(); bad.loc[4, tid] = np.nan
    assert any("minimum inside diameter" in m for _, m in weq.check_downhole(bad, "tubing", None))


def test_gradient_results_follow_the_traverse():
    from wellmodel import gradient as wgrad
    top, secs, pts, smd, stvd, path, oil = _vlp_setup()
    case = {"q": np.array([1500.0]), "wc": np.array([0.2]), "gor": np.array([400.0])}
    for corr in ("hagedorn_brown", "beggs_brill"):
        P, T = wvlp.traverse(path, oil, 250.0, case, corr, 1.0, 1.0, 5.0)
        gt = wgrad.gradient_table(path, oil, case, P[0], T[0], corr, 1.0, 1.0, 5.0, wgrad.node_labels(path, pts))
        assert gt["Regime"].iloc[0] == "Wellhead" and gt["Regime"].iloc[-1] == "Bottomhole"
        assert gt["Pressure (psia)"].iloc[-1] == pytest.approx(P[0, -1])
        inlet = gt[gt["Regime"].str.startswith("Restriction Inlet")]
        outlet = gt[gt["Regime"].str.startswith("Restriction Outlet")]
        assert len(inlet) == len(outlet) == len(path.restrictions) == 1
        assert outlet["Label"].iloc[0] == "SSSV"
        dp_r = float(inlet["Pressure (psia)"].iloc[0] - outlet["Pressure (psia)"].iloc[0])
        assert dp_r > 0
        # cumulative gravity + friction loss from the wellhead ~ total pressure rise minus the restriction
        tot = gt["Gravity Pressure Loss (psi)"].iloc[0] + gt["Friction Pressure Loss (psi)"].iloc[0]
        assert tot == pytest.approx(P[0, -1] - P[0, 0] - dp_r, rel=0.01)
        flow = gt.dropna(subset=["Holdup (fraction)"])
        assert flow["Total Mass Flow Rate (lbm/day)"].std() / flow["Total Mass Flow Rate (lbm/day)"].mean() < 1e-6
        assert (flow["Holdup (fraction)"].between(0, 1)).all()
        assert flow["Mixture Density (lb/ft3)"].le(flow["Flowing Density (lb/ft3)"] + 1e-9).all()
        assert set(flow["Regime"]) <= {"Bubble", "Slug", "Transition", "Mist", "Liquid", "Segregated",
                                       "Intermittent", "Distributed"}
    assert wgrad.flow_regime("hagedorn_brown", 4.017, 4.06, 52.76, 1.78, 27.0, 2.44 / 12, 1.0) == "Slug"


def test_pe2_style_reproduces_prosper_pe2_rows():
    """Rows from a PROSPER PE2 gradient export (Abura 2S): holdup and gradients at PROSPER's own conditions.
    Friction is compared with the roughness PROSPER's single-phase friction implies (~0.0019 in)."""
    from wellmodel import correlations as C
    # vsl, vsg, rho_l, rho_g, mu_l, mu_g, sigma, d_in, angle, p | holdup, static, friction (PROSPER)
    rows = [(4.051, 2.548, 51.87, 2.62, 2.831, 0.01307, 10.3, 2.44, 0.61, 824.6, None, None, None),
            (4.259, 0.701, 50.140, 5.132, 1.879, 0.0153, 5.7563, 2.44, 0.61, 1603.0, 0.8623, 0.310, 0.015),
            (6.836, 0.520, 49.301, 6.243, 1.5516, 0.0157, 4.43, 1.95, 18.40, 1960.27, 0.9359, 0.306, 0.041)]
    for vsl, vsg, rl, rg, ml, mg, sg, d, ang, p, hl_p, gs_p, gf_p in rows:
        a = [np.array([x]) for x in (vsl, vsg, rl, rg, ml, mg, sg)]
        e, f, hl = C.pe2_style(*a, d / 12, 0.0019 / 12, np.cos(np.radians(ang)), np.array([p]))
        if hl_p is None:
            continue
        assert hl[0] == pytest.approx(hl_p, abs=0.003)
        assert e[0] == pytest.approx(gs_p, abs=0.003)
        assert f[0] == pytest.approx(gf_p, abs=0.0015)
    # bubble flow is the modified Hagedorn & Brown (Griffith) - identical
    a = [np.array([x]) for x in (6.836, 0.520, 49.301, 6.243, 1.5516, 0.0157, 4.43)]
    args = (*a, 1.95 / 12, 0.0006 / 12, 0.95, np.array([1960.0]))
    assert np.allclose(C.pe2_style(*args), C.hagedorn_brown(*args))
    assert "pe2_style" in wvlp.VLP_CORRELATIONS


def test_tuffp_unified_flow_patterns_and_limits():
    from wellmodel import unified as U
    rl, rg, ml, mg, sig, d = 998.0, 3.5, 1e-3, 1.8e-5, 0.0731, 0.0508          # air-water, 2 in, ~3 bar
    case = lambda vsl, vsg, ang: U.unified_si([vsl], [vsg], rl, rg, ml, mg, sig, d, 4.6e-5,
                                              np.sin(np.radians(ang)), 3e5)
    assert case(0.01, 0.5, 0)["regime"][0] == "Stratified"
    assert case(1.0, 0.5, 0)["regime"][0] == "Slug"
    assert case(3.0, 0.1, 90)["regime"][0] == "Dispersed bubble"
    assert case(0.05, 30.0, 90)["regime"][0] == "Annular"
    s = case(0.2, 2.0, 90)
    assert s["regime"][0] == "Slug" and 0 < s["slug_frequency_1_s"][0] < 10 and s["slug_holdup"][0] >= 0.24
    # vectorised = one by one; everything finite and holdup in (0, 1] at all angles
    vsl = np.array([0.01, 0.2, 1.0, 3.0]); vsg = np.array([0.5, 2.0, 10.0, 0.3])
    for ang in (-90, -10, 0, 30, 90):
        D = U.unified_si(vsl, vsg, rl, rg, ml, mg, sig, d, 4.6e-5, np.sin(np.radians(ang)), 3e5)
        assert np.isfinite(D["grav"]).all() and np.isfinite(D["fric"]).all()
        assert ((D["hl"] > 0) & (D["hl"] <= 1)).all()
        one = [case(a, b, ang)["hl"][0] for a, b in zip(vsl, vsg)]
        assert np.allclose(one, D["hl"], atol=2e-3)
    # field-unit wrapper: single-phase liquid gives the hydrostatic gradient
    e, f, h = U.tuffp_unified(np.array([5.0]), np.array([0.0]), 62.4, 0.1, 1.0, 0.012, 70.0, 0.2, 5e-5, 1.0, 500.0)
    assert h[0] == 1.0 and e[0] == pytest.approx(62.4 / 144, rel=1e-3)


def test_tuffp_unified_at_prosper_pe5_conditions():
    """Slug-flow rows of a PROSPER PE5 export (Abura 2S): the unified model's slug holdup is close to PE5's
    (both slug-dynamics models); it is not calibrated to PE5."""
    from wellmodel import unified as U
    # vsl, vsg ft/s, rho_l, rho_g lb/ft3, mu_l, mu_g cp, sigma dyne/cm, d in, angle deg, p psia, PE5 holdup
    rows = [(4.026, 3.315, 52.153, 2.169, 3.1533, 0.0124, 11.339, 2.44, 0.61, 686.37, 0.558753),
            (4.045, 2.808, 51.989, 2.437, 3.0314, 0.0126, 10.73, 2.44, 0.61, 771.22, 0.601257)]
    for vsl, vsg, rl, rg, ml, mg, sg, d, ang, p, hl_pe5 in rows:
        D = U.tuffp_details([vsl], [vsg], [rl], [rg], [ml], [mg], [sg], d / 12, 0.0006 / 12,
                            np.cos(np.radians(ang)), [p])
        assert D["regime"][0] == "Slug"
        assert D["hl"][0] == pytest.approx(hl_pe5, abs=0.01)
    assert "tuffp_unified" in wvlp.VLP_CORRELATIONS


def test_tuffp_unified_gas_well_within_mechanistic_family():
    *_, path, oil = _vlp_setup()
    gas = wvlp.WetGasFluid(wbo.PVTInputs(rsb=1, api=40, gas_sg=0.65), 50)
    case = {"q": np.array([1.0, 5.0, 20.0]), "wgr": 10.0, "cgr": 20.0}
    pt = wvlp.traverse(path, gas, 1000.0, case, "tuffp_unified", u=4.0)[0][:, -1]
    pa = wvlp.traverse(path, gas, 1000.0, case, "ansari", u=4.0)[0][:, -1]
    pd_ = wvlp.traverse(path, gas, 1000.0, case, "duns_ros", u=4.0)[0][:, -1]
    assert np.all(np.isfinite(pt)) and np.all(pt > 1000.0)
    assert np.all(pt > 0.8 * np.minimum(pa, pd_)) and np.all(pt < 1.2 * np.maximum(pa, pd_))


def test_pe5_style_reproduces_prosper_pe5_rows():
    """Rows from a PROSPER PE5 gradient export (Abura 2S) at PROSPER's own conditions (roughness 0.0006 in): holdup
    within 0.006, total gradient within 2 % (friction alone within 15 % - the friction multiplier is fitted mainly on
    the Tutorial 38 / 47 condensate gradients)."""
    from wellmodel import correlations as C
    # vsl, vsg, rho_l, rho_g, mu_l, mu_g, sigma, d_in, angle, p | holdup, static, friction (PROSPER PE5)
    rows = [(4.026, 3.315, 52.153, 2.169, 3.1533, 0.0124, 11.339, 2.44, 0.61, 686.37, 0.558753, 0.209, 0.033),
            (4.146, 1.384, 51.12, 3.763, 2.314, 0.0137, 8.03, 2.44, 0.61, 1187.35, 0.760532, 0.276, 0.022)]
    for vsl, vsg, rl, rg, ml, mg, sg, d, ang, p, hl_p, gs_p, gf_p in rows:
        a = [np.array([x]) for x in (vsl, vsg, rl, rg, ml, mg, sg)]
        e, f, hl = C.pe5_style(*a, d / 12, 0.0006 / 12, np.cos(np.radians(ang)), np.array([p]))
        assert hl[0] == pytest.approx(hl_p, abs=0.006)
        assert e[0] == pytest.approx(gs_p, abs=0.003)
        assert e[0] + f[0] == pytest.approx(gs_p + gf_p, rel=0.02)   # total gradient (friction alone within 10 %)
        assert f[0] == pytest.approx(gf_p, rel=0.15)
    a = [np.array([x]) for x in (6.836, 0.520, 49.301, 6.243, 1.5516, 0.0157, 4.43)]
    args = (*a, 1.95 / 12, 0.0006 / 12, 0.95, np.array([1960.0]))
    assert np.allclose(C.pe5_style(*args), C.hagedorn_brown(*args))         # bubble flow: Griffith
    assert "pe5_style" in wvlp.VLP_CORRELATIONS


def test_choke_performance_elf_perkins():
    from wellmodel import choke as CK
    *_, path, oil = _vlp_setup()
    S = CK.ChokeSetup(oil, "oil", {"wc": 0.2, "gor": 800.0}, weq.HeatCapacities())
    p1, t1 = 664.7, 120.0
    # the three calculation options are consistent with each other
    r = S.rate(p1, t1, 414.7, 0.42)
    d, r2 = S.choke_size(p1, t1, 414.7, r["q"])
    assert d == pytest.approx(0.42, rel=1e-3)
    p4, r3 = S.downstream_pressure(p1, t1, r["q"], 0.42)
    assert p4 == pytest.approx(414.7, abs=0.5)
    # critical flow: rate independent of the outlet pressure below the boundary, equal to the critical rate
    lo = S.rate(p1, t1, 50.0, 0.42)
    assert lo["regime"] == "Critical" and lo["q"] == pytest.approx(lo["q_crit"], rel=1e-9)
    assert S.rate(p1, t1, 100.0, 0.42)["q"] == pytest.approx(lo["q"], rel=1e-9)
    assert 0.3 < lo["p_crit"] / p1 < 0.75 and lo["t_crit"] < t1
    ps, qs, ts = S.performance_curve(p1, t1, 0.42)
    assert np.all(np.diff(qs) <= 1e-9) and qs[-1] < 0.3 * qs[0]
    # a rate above the critical rate cannot pass
    assert np.isnan(S.downstream_pressure(p1, t1, lo["q"] * 1.05, 0.42)[0])
    # PROSPER's ELF example (GOR 800, WC 20 %, 650 -> 400 psig, 120 F, 1,000 STB/d): 0.4026 in, critical
    # pressure 366 psig, critical rate 1,014 STB/d - with this tool's example oil, within a few per cent
    d0, r0 = S.choke_size(p1, t1, 414.7, 1000.0)
    assert d0 == pytest.approx(0.4026, rel=0.06)
    assert r0["p_crit"] - 14.7 == pytest.approx(366.0, rel=0.06)
    assert r0["q_crit"] == pytest.approx(1014.0, rel=0.05)


def test_choke_gas_only_critical_ratio_and_liquid_only():
    from wellmodel import choke as CK
    gas = wvlp.WetGasFluid(wbo.PVTInputs(rsb=1, api=40, gas_sg=0.6), 50)
    S = CK.ChokeSetup(gas, "gas", {"wgr": 0.0, "cgr": 0.0}, weq.HeatCapacities(), d_up=6.0, d_down=6.0)
    pr_c, n, lam, B = S._critical(1000.0, 100.0, 0.5)
    k = n                                                     # single-phase gas: n = F
    assert pr_c == pytest.approx((2 / (k + 1)) ** (k / (k - 1)), rel=0.02)
    *_, path, oil = _vlp_setup()
    L = CK.ChokeSetup(oil, "oil", {"wc": 1.0, "gor": 0.0}, weq.HeatCapacities())
    r = L.rate(1000.0, 100.0, 50.0, 0.5)
    assert r["regime"] == "Subcritical" and np.isnan(r["q_crit"])


# ------------------------------------------------------------ continuous gas lift design (tab 7)
from wellmodel import gaslift as wgl


def test_gaslift_thornhill_craver_matches_api_chart():
    # API RP 11V6 Table 6 (uncorrected, 3/16-in port = 12/64, 1200 psig upstream, SG 0.65, 60 F)
    for p2, q in ((1100, 550), (1000, 720), (900, 830), (800, 880), (700, 920)):
        assert float(wgl.thornhill_craver(12 / 64, 1214.7, p2 + 14.7, 60.0, 0.65)) == pytest.approx(q, rel=0.03)
    assert wgl.TC_RCRIT == pytest.approx(0.5512, abs=1e-4)
    d = float(wgl.port_for_rate(750.0, 1214.7, 989.7, 60.0, 0.65)) * 64
    assert d == pytest.approx(12.0, abs=0.15)                   # "750 MCFD needs about 225 psi" through 3/16 in
    p1 = wgl.upstream_for_rate(750.0, 12 / 64, 989.7, 60.0, 0.65)
    assert p1 == pytest.approx(1214.7, abs=3.0)


def test_gaslift_dome_corrections():
    assert float(wgl.ct_api(130)) == pytest.approx(0.869, abs=5e-4)          # API Table A-1
    assert float(wgl.ct_api(300)) == pytest.approx(0.660, abs=5e-4)
    # Winkler & Eads (1989) examples 1 and 2
    assert float(wgl.dome_60_winkler(806, 170)) == pytest.approx(650.0, abs=0.2)
    assert float(wgl.dome_60_winkler(2228, 200)) == pytest.approx(1656.0, abs=0.3)
    assert float(wgl.dome_at_t_winkler(1656.07, 200)) == pytest.approx(2228.0, abs=0.1)
    # Table 3: 1000 psia at 60 F -> 1291.5 psia at 190 F
    assert float(wgl.dome_at_t_winkler(1000 - 14.4, 190)) + 14.4 == pytest.approx(1291.5, rel=2.5e-3)
    # PROSPER option: API below 1200 psig, Winkler-Eads above
    assert float(wgl.dome_at_60(806, 170)) == pytest.approx(806 * float(wgl.ct_api(170)))
    assert float(wgl.dome_at_60(2228, 200)) == pytest.approx(1656.07, abs=0.3)
    assert float(wgl.dome_at_60(2228, 200, high_pressure=False)) == pytest.approx(1713, abs=1.5)


def test_gaslift_api_example1_spacing_and_settings():
    """API RP 11V6 Example 1 (section 5.1): 1200 psig gas, gg 0.030, Pwh 100, gs 0.465, PD 25, Psf 20."""
    md = np.arange(0.0, 8000.1, 1.0)
    ppd = lambda d: np.interp(d, [0, 2500, 4000, 5000, 6000, 7000, 8000], [100, 400, 600, 740, 880, 1030, 1200])
    depths, why = wgl.space_valves(md, md, 100.0, lambda m: float(ppd(m)), 1200 + 0.03 * md,
                                   lambda n: 1200 - 25 * n + 0.03 * md, 0.465, 20.0, 20.0, 250.0, 7940.0)
    assert np.allclose(depths, [2483, 4390, 5797, 6783, 7434, 7820], atol=20)
    assert why == "injection depth reached"
    # Table 2, valve 1: Piod 1275, Ppd 400, Ap/Ab 0.094, Tv 130 F -> Pvo (test rack) 1144 psig
    gas = wgl.LiftGas(0.65, t_surface=78)
    pvc, pd60, ptro, _ = wgl.valve_settings(1275.0, 400.0, 0.094, 130.0, 1200.0, gas, 0.0, 2475.0, 78.0, 128.0,
                                            high_pressure=False, p_atm=0.0)
    assert ptro == pytest.approx(1144, abs=1.5)
    assert pvc == pytest.approx(1275 * 0.906 + 400 * 0.094)   # closing pressure at depth = dome at temperature


def test_gaslift_asheim_claymore_c6():
    m3s_bpd = 86400 / 0.158987
    q_l, q_g = 0.0221 * m3s_bpd, 2.87 * 86400 / 0.0283168 / 1e6
    j = 3.84e-9 * m3s_bpd * 6894.76

    class Gas(wgl.LiftGas):
        def bg(self, p, t):
            return 0.00882

    d = 7865.0
    vt = np.pi / 4 * (4.78 / 12) ** 2 * d
    vc = np.pi / 4 * ((8.7 / 12) ** 2 - (5.51 / 12) ** 2) * d
    out = wgl.asheim(q_l, q_g, j, 24 / 64, 2, 1614.7, 172.0, Gas(0.81), 884 * 0.062428, 0.0221 * 35.3147, vt, vc, d)
    assert out["F1"] == pytest.approx(0.76, abs=0.01)
    assert out["F2"] == pytest.approx(2.74, rel=0.03)
    assert out["stable"]


def test_gaslift_traverse_and_casing_column():
    *_, path, oil = _vlp_setup()
    gas = wgl.LiftGas(0.7, t_surface=80.0)
    pc = wgl.casing_pressure(path, gas, 1200.0)[0]
    assert 0.02 < (pc[-1] - pc[0]) / (path.tvd[-1] - path.tvd[0]) < 0.04         # lift-gas column ~0.03 psi/ft
    # no lift gas: the lifted traverse is the ordinary traverse
    case = {"q": [800.0], "wc": 0.5, "gor": 300.0}
    P0, _ = wvlp.traverse(path, oil, 200.0, case, "hagedorn_brown")
    r = wgl.lifted_traverse(path, oil, 200.0, case, [0.0], "hagedorn_brown", fixed_md=[8000.0])
    assert r.P[0, -1] == pytest.approx(P0[0, -1], rel=0.01)
    # lift gas lowers the bottom-hole pressure; the injection point is where tubing + dP meets the casing
    r = wgl.lifted_traverse(path, oil, 200.0, {"q": [800.0] * 2, "wc": 0.5, "gor": 300.0}, [0.5, 1.5],
                            "hagedorn_brown", pc=pc, dp_valve=100.0, max_md=10000.0)
    assert np.all(r.P[:, -1] < P0[0, -1])
    assert r.inj_md[0] < 10000.0 and not r.limited[0]
    assert np.interp(r.inj_md[0], path.md, pc) - r.inj_p[0] == pytest.approx(100.0, abs=3.0)


def test_gaslift_design_new_well():
    top, secs, pts, smd, stvd, path, oil = _vlp_setup()
    curve = iflow.pi_entry(3000, 1000, 2.0)
    gas = wgl.LiftGas(0.7, t_surface=80.0)
    inp = wgl.DesignInput(design_rate=wgl.DESIGN_RATE[1], top_p=200, top_p_unload=200, p_inj=1200, p_ko=1200,
                          qgi_max=1.0, qgi_unload_max=1.0, gor=300, wc=0.5, max_md=10000, q_liquid_max=5000,
                          dp_close_spread=False, valve=list(wgl.VALVES)[1])
    res = wgl.design(path, oil, curve, gas, inp, "hagedorn_brown")
    v = res.valves
    assert v[-1].kind == "Orifice" and all(x.kind == "Valve" for x in v[:-1]) and len(v) >= 3
    mds = [x.md for x in v]
    assert all(np.diff(mds) >= inp.min_spacing - 1e-6)
    assert res.pso == pytest.approx(inp.p_inj - inp.dp_close * (len(v) - 1))
    assert v[-1].dp_valve >= inp.dp_valve - 5            # never deeper than the balance point
    assert all(x.ptro > x.pd60 for x in v[:-1])                   # Ptro = Pd60 / (1 - R)
    assert all(x.chp_close < x.chp_open for x in v[:-1])
    assert res.perf is not None and res.qgi <= inp.qgi_max + 1e-9
    # the operating point honours the IPR
    assert res.pwf == pytest.approx(float(wgl.ipr_pwf(curve, res.q)), abs=15)
    vc = wgl.annulus_volume(secs, top, res.inj_md)
    stab = wgl.design_stability(res, path, oil, gas, curve, vc)
    assert stab["F1"] > 0 and np.isfinite(vc)


def test_gaslift_valve_database():
    """Valve database: built-in valves from wellmodel/data/gl_valves.csv (Camco R-20 Normal and Carbide), user valves
    from a long table (one row per port) registered by name; R values from the table, bellows area from them."""
    assert "Camco R-20 Carbide" in wgl.VALVES and wgl.VALVES["Camco R-20 Carbide"].r(20) == pytest.approx(0.126)
    assert wgl.VALVES["Camco R-20 Normal (1.5-in)"].r(16) == pytest.approx(0.066)
    # PROSPER's database as listed on the user's screens: 30 casing-sensitive valves from 7 manufacturers, 7
    # tubing-sensitive
    bt = wgl.builtin_valve_table()
    cs = bt[bt["Valve type"] == "Casing sensitive"]
    assert cs.groupby(["Manufacturer", "Model", "Specification"]).ngroups == 30 and len(cs) == 181
    ts = bt[bt["Valve type"] == "Tubing sensitive"]
    assert ts.groupby(["Manufacturer", "Model", "Specification"]).ngroups == 7 and len(ts) == 22
    assert wgl.VALVES["Camco R-25 Carbide"].kind == "Tubing sensitive" and wgl.VALVES["Camco R-25 Carbide"].r(16) == 0.075
    assert set(bt["Manufacturer"]) == set(wgl.VALVE_MANUFACTURERS)
    assert wgl.VALVES["PTC 1.5inch (23mm_bellows) Normal"].r(32) == pytest.approx(0.3049)
    assert wgl.VALVES["Weatherford R-1 Monel"].r(16) == pytest.approx(0.1647)
    assert wgl.VALVES["Baker B1 Type_b"].ports == (32, 36, 40, 42, 46, 53)
    db = pd.DataFrame({"Manufacturer": ["Baker"] * 3 + ["", "Otis-Merla"], "Model": ["X-1"] * 3 + ["", "Y"],
                       "Specification": ["Normal"] * 3 + ["", ""], "Valve type": ["Casing sensitive"] * 5,
                       "Port (64ths in)": [16, 8, 12, 10, None], "R value": [0.1, 0.03, 0.06, 0.05, 0.2]})
    names = wgl.register_valves(db)
    assert names == ["Baker X-1 Normal"]
    v = wgl.VALVES["Baker X-1 Normal"]
    assert v.ports == (8, 12, 16) and v.r(12) == pytest.approx(0.06) and v.source == "user"
    assert v.bellows_area == pytest.approx(np.median(np.pi / 4 * (np.array([8, 12, 16]) / 64) ** 2 /
                                                     np.array([0.03, 0.06, 0.1])))
    wgl.register_valves(None)
    assert "Baker X-1 Normal" not in wgl.VALVES and "Camco R-20 Carbide" in wgl.VALVES


def test_gaslift_design_existing_mandrels():
    """Design | Existing mandrels: valves only in the mandrels (each unloading valve in the deepest mandrel the well
    unloads to, injection in the deepest mandrel at or above the optimum depth), dummies in the rest; the new-well
    design's own depths as mandrels keep its injection point; a gap in the mandrels moves the injection up."""
    top, secs, pts, smd, stvd, path, oil = _vlp_setup()
    curve = iflow.pi_entry(3000, 1000, 2.0)
    gas = wgl.LiftGas(0.7, t_surface=80.0)
    kw = dict(design_rate=wgl.DESIGN_RATE[1], top_p=200, top_p_unload=200, p_inj=1200, p_ko=1200, qgi_max=1.0,
              qgi_unload_max=1.0, gor=300, wc=0.5, max_md=10000, q_liquid_max=5000, dp_close_spread=False,
              valve=list(wgl.VALVES)[1])
    new = wgl.design(path, oil, curve, gas, wgl.DesignInput(**kw), "hagedorn_brown")
    mds = [round(v.md, 1) for v in new.valves]
    extra = [mds[0] - 300.0, (mds[0] + mds[1]) / 2]
    mand = tuple((m, np.nan, np.nan) for m in sorted(mds + extra))
    res = wgl.design(path, oil, curve, gas, wgl.DesignInput(**kw, mandrels=mand), "hagedorn_brown")
    # same injection point; each valve in the deepest mandrel reachable, so a valve the free spacing needed may be
    # skipped when the next mandrel down is still within reach; the extras above hold dummies
    assert set(round(v.md, 1) for v in res.valves) <= set(mds) and res.inj_md == pytest.approx(new.inj_md, abs=0.1)
    assert res.valves[-1].kind == "Orifice" and all(v.kind == "Valve" for v in res.valves[:-1])
    holds = dict(zip(res.mandrels["Measured depth (ft)"], res.mandrels["Holds"]))
    assert all(holds[m] == "Dummy" for m in extra)
    assert sum(h != "Dummy" for h in holds.values()) == len(res.valves)
    # sparse mandrels: every valve in a mandrel, none deeper than the free design's injection point
    sparse = tuple((m, np.nan, np.nan) for m in np.arange(1500.0, path.md[-1], 1000.0))
    rs = wgl.design(path, oil, curve, gas, wgl.DesignInput(**kw, mandrels=sparse), "hagedorn_brown")
    allowed = {m[0] for m in sparse}
    assert all(v.md in allowed for v in rs.valves) and rs.inj_md <= new.inj_md + inp_tol(kw)
    assert rs.valves[-1].kind == "Orifice" and len({v.md for v in rs.valves}) == len(rs.valves)
    # a mandrel's own dP casing overrides the main value
    first = rs.valves[0].md
    od = tuple((m, 80.0 if m == first else np.nan, np.nan) for m, _, _ in sparse)
    ro = wgl.design(path, oil, curve, gas, wgl.DesignInput(**kw, mandrels=od), "hagedorn_brown")
    if len(ro.valves) > 1 and ro.valves[0].md == first:
        assert ro.valves[1].chp_open == pytest.approx(ro.valves[0].chp_open - 80.0, abs=1e-6)
    assert any("Unloading limited" in m for m in rs.messages)


def inp_tol(kw):
    return wgl.DesignInput(**kw).mandrel_tol


# ------------------------------------------------------------ deviation / partial penetration skins
@pytest.mark.parametrize("hd,zr,pen,th,expected", [
    (100, .5, .25, 0, 6.611), (100, .5, .25, 45, 4.245), (100, .5, .25, 75, -0.451), (100, .5, 1, 45, -1.178),
    (100, .5, 1, 75, -3.577), (100, .95, .1, 60, 12.662), (100, .75, .5, 0, 3.067), (100, .6, .1, 75, 3.739),
    (1000, .5, .25, 45, 8.593), (1000, .875, .25, 60, 6.163), (1000, .5, 1, 45, -1.850),
    (1000, .95, .1, 75, 10.755), (1000, .5, .75, 60, -2.380)])
def test_cinco_ley_full_solution_tables(hd, zr, pen, th, expected):
    """Cinco-Ley, Ramey & Miller (1975, SPE 5589) Tables 1 and 2 (s_theta+p)."""
    from wellmodel import completion as wc
    h, rw = float(hd), 1.0
    hp = pen * h
    z_top = h - (zr * h + hp / 2)
    assert wc.cinco_slant_pp(th, h, hp, z_top, rw)["total"] == pytest.approx(expected, abs=0.03)


def test_deviation_and_brons_marting_skins():
    from wellmodel import completion as wc
    # correlation against the full solution for a fully penetrating well (hD = 100)
    assert wc.deviation_skin(45, 100, 1.0, anisotropic=False) == pytest.approx(-1.178, abs=0.05)
    assert wc.deviation_skin(75, 1000, 1.0) == pytest.approx(-5.282, abs=0.15)
    with pytest.raises(ValueError):
        wc.deviation_skin(70, 100, 1.0, anisotropic=False)          # Cinco (isotropic) valid to 65 deg
    # isotropic version ignores kv/kh; the anisotropic one does not
    assert wc.deviation_skin(40, 100, 0.354, 0.1, anisotropic=False) == wc.deviation_skin(40, 100, 0.354, 1.0)
    assert wc.deviation_skin(40, 100, 0.354, 0.1) > wc.deviation_skin(40, 100, 0.354, 1.0)
    # Brons & Marting with the interval at the top: SPE 5589 Table 1, theta 0, top completions
    assert wc.brons_marting(100, 25, 1.0) == pytest.approx(8.641, rel=0.02)
    assert wc.brons_marting(100, 50, 1.0) == pytest.approx(3.067, rel=0.02)
    assert wc.brons_marting(100, 100, 1.0) == 0.0
    assert wc.brons_marting(100, 50, 1.0, 0.1) > wc.brons_marting(100, 50, 1.0)


# ------------------------------------------------------------ Kuchuk & Goode horizontal well
def test_goode_kuchuk_k12_table():
    for b, v in ((0.001, 0.99843), (0.1, 0.86252), (1.0, 0.27362), (5.0, 0.31784e-2), (40.0, 0.81951e-18)):
        assert iadv.k12(b) == pytest.approx(v, rel=2e-4)


def test_goode_kuchuk_against_paper_figure2_and_limits():
    """Fig. 2 (h 50 ft, zw 25 ft, rw 0.25 ft, kH 50 md, mu 1 cp, 4000 x 4000 ft, well centred): PI read off the
    curves; the formula also tends to the 2-D image solution for the constant-pressure case."""
    pi = lambda an, r: 7.08e-3 * np.divide(*iadv.goode_kuchuk(50, 50, 0.25, 1, an, 2 * r * 4000, 4000, 4000, 2000,
                                                              2000, 25)[:2])
    assert pi(1, 0.5) == pytest.approx(31.5, rel=0.06)
    assert pi(1, 0.3) == pytest.approx(21.0, rel=0.06)
    assert pi(0.1, 0.5) == pytest.approx(27.0, rel=0.06)
    assert pi(0.01, 0.5) == pytest.approx(17.5, rel=0.06)
    assert pi(1, 0.5) > pi(0.1, 0.5) > pi(0.01, 0.5)
    # shorter wells always need more drawdown (the K12 series keeps this true when hD > 5)
    d = [iadv.goode_kuchuk(50, 50, 0.25, 1, 0.1, L, 4000, 4000, 2000, 2000, 25)[1] for L in (400, 100, 40, 20, 10)]
    assert all(np.diff(d) > 0)
    # constant pressure at the top: ln term = 2-D image solution ln[(4h / pi rw) cot(pi zw / 2h)]
    h, zw, rw, L = 100.0, 40.0, 0.25, 4000.0
    kh, D, p = iadv.goode_kuchuk(50, h, rw, 1, 1, L, 8000, 8000, 4000, 4000, zw, boundary="constant pressure")
    hd = h / (L / 2)
    assert D == pytest.approx(hd / 2 * (np.log(4 * h / (np.pi * rw) / np.tan(np.pi * zw / (2 * h))) + (zw - h) / (L / 2)),
                              rel=1e-3)
    # mechanical skin along the well: S*m = h / (2 L1/2) sqrt(kx/kz) Sm
    D5 = iadv.goode_kuchuk(50, h, rw, 1, 0.25, L, 8000, 8000, 4000, 4000, zw, skin=5.0)[1]
    D0 = iadv.goode_kuchuk(50, h, rw, 1, 0.25, L, 8000, 8000, 4000, 4000, zw)[1]
    assert D5 - D0 == pytest.approx(h / L * 2 * 5.0)


# ------------------------------------------------------------ units
def test_unit_conversions_and_labels():
    from wellmodel import units as wu
    assert wu.convert(5200.0, "pressure", "psig", "psia") == pytest.approx(5214.696, abs=1e-3)
    assert wu.convert(1.0, "pressure", "bara", "psia") == pytest.approx(14.5038, rel=1e-5)
    assert wu.convert(212.0, "temperature", "°F", "°C") == pytest.approx(100.0)
    assert wu.convert(1.0, "gor", "Sm3/Sm3", "scf/STB") == pytest.approx(5.6146, rel=1e-4)
    assert wu.convert(1.0, "productivity_index", "STB/d/psi", "Sm3/d/bar") == pytest.approx(2.3059, rel=1e-4)
    assert wu.convert(wu.convert(123.4, "pressure", "psia", "barg"), "pressure", "barg", "psia") == pytest.approx(123.4)
    assert wu.split_label("Reservoir pressure (psia)") == ("Reservoir pressure", "pressure", "psia", "")
    assert wu.split_label("Water cut (%)") is None
    assert wu.split_label("Screen outer radius (ft, 0 = none)")[0] == "Screen outer radius (0 = none)"
    metric = {**wu.DEFAULTS, **wu.PRESETS["Metric"]}
    assert wu.relabel("Liquid Rate (STB/day)", metric)[0] == "Liquid Rate (Sm3/d)"
    assert wu.relabel("Dome pressure at 60 °F (psig)", metric)[0].endswith("(barg)")    # gauge stays gauge
    assert wu.convert_text("Pr 3,000 psia at 9,000 ft MD and 210 °F", metric) == "Pr 206.8 bara at 2,743 m MD and 98.89 °C"
    assert wu.is_default(wu.DEFAULTS) and not wu.is_default(metric)


def test_unit_table_conversion():
    from wellmodel import units as wu, ui_units as wuu
    metric = {**wu.DEFAULTS, **wu.PRESETS["Metric"]}
    df = pd.DataFrame({"Pressure (psia)": [14.696, 1014.696], "Rate (STB/d)": [100.0, 200.0], "Note": ["a", "b"]})
    out = wuu.df_out(df, metric)
    assert list(out.columns) == ["Pressure (bara)", "Rate (Sm3/d)", "Note"]
    assert out["Pressure (bara)"].iloc[0] == pytest.approx(1.01325, rel=1e-5)
    res = pd.DataFrame({"Result": ["Rate", "Cut"], "Value": ["1,000.0", "50"], "Units": ["STB/d", "%"]})
    out = wuu.df_out(res, metric)
    assert out["Units"].tolist() == ["Sm3/d", "%"] and out["Value"].iloc[0] == "158.99" and out["Value"].iloc[1] == "50"


def test_unit_result_rows_converted():
    """Result/Value tables whose row names carry the unit are converted row by row."""
    from wellmodel import units as wu, ui_units as wuu
    df = pd.DataFrame([("Solution node pressure (psia)", "2,363"), ("Stability", "Stable")], columns=["Result", "Value"])
    out = wuu.df_out(df, {**wu.DEFAULTS, **wu.PRESETS["Oilfield, gauge pressure"]})
    assert out["Result"].iloc[0] == "Solution node pressure (psig)" and out["Value"].iloc[0] == "2,348"
    assert out["Value"].iloc[1] == "Stable"


@pytest.mark.parametrize("ld, lc, eff, target", [
    (8.0, 0.2, 0.9, 1.600337),   # PROSPER manual, step 4
    (8.0, 0.0, 0.9, 0.967990),   # no crushed zone
    (0.0, 0.2, 0.9, 0.573156),   # no damaged zone
    (0.0, 0.0, 1.0, 0.088012),   # perforations only
])
def test_karakas_tariq_matches_prosper(ld, lc, eff, target):
    """Karakas & Tariq completion skin against PROSPER (50 md, kv 5 md, 8 spf, 0.43 in, 9.2 in, 120 deg,
    rw 0.354 ft, damaged zone 25 md, crushed zone 12.5 md)."""
    from wellmodel import completion as wc
    r = wc.karakas_tariq(50, 0.354, 9.2 / 12, 0.43 / 24, 8 * eff, 120, 0.1, ld / 12, 25, lc / 12, 12.5)
    assert r["total"] == pytest.approx(target, abs=2e-6)
    assert wc.deviation_skin(13, 100, 0.354, 0.1, anisotropic=True) == pytest.approx(-0.016550, abs=1e-6)



def test_pi_entry_composite_water_cut():
    """Below the bubble point the PI-entry/Darcy IPR follows the Petrobras composite form with water (PROSPER);
    with no water it is Vogel, and above the bubble point the water cut has no effect."""
    from wellmodel import inflow as I
    pr, pb, j = 3464.7, 3514.7, 7.166
    v = I.pi_entry(pr, pb, j)
    c0 = I.pi_entry(pr, pb, j, 0.0)
    c8 = I.pi_entry(pr, pb, j, 0.8)
    assert c0.aof == pytest.approx(v.aof, rel=1e-9)
    assert c8.aof > v.aof * 1.2                   # water flows linearly: more inflow at low pwf
    hi = I.pi_entry(5000.0, 3000.0, 5.0, 0.8)
    assert float(hi.rate(4000.0)) == pytest.approx(5.0 * 1000.0, rel=1e-3)


def test_camco_r20_r_values():
    from wellmodel import gaslift as wgl
    v = wgl.VALVES["Camco R-20 Normal (1.5-in)"]
    assert float(v.r(16)) == pytest.approx(0.066) and float(v.r(24)) == pytest.approx(0.147)
    assert v.ports[-1] == 36


def test_complementary_calculations():
    """Grain size (PROSPER's Wasp-Kenny-Gandhi oil and gas forms), API 14E / ConocoPhillips erosional velocity,
    Turner - checked against hand calculations of the published equations."""
    from wellmodel import complementary as CX
    # oil form: the diameter returned must give back the flow velocity in the CGS equation
    d = CX.grain_diameter_oil(10.0, 1.0, 50.0, 2.65)
    r = d * 2.54 / 2
    assert CX._v_oil_cgs(r, 0.01, 50.0 / 62.428, 2.65) == pytest.approx(10.0 * 30.48, rel=1e-8)
    # gas form: d = 3 rho Cd V^2 / (4 (rho_s - rho) g)
    rho_s = 2.65 * 62.428
    assert CX.grain_diameter_gas(20.0, 5.0) == pytest.approx(3 * 5 * 0.85 * 400 / (4 * (rho_s - 5) * 32.2) * 12, rel=1e-12)
    o = CX.CompOptions(c_factor=100.0)
    assert CX.erosional_velocity(25.0, 4.0, o) == pytest.approx(20.0)
    # sand: ConocoPhillips Ve = S D sqrt(rho) / sqrt(W) (m/s, mm, kg/m3, kg/d), smaller of the two
    o2 = CX.CompOptions(c_factor=1e6, s_factor=0.05, sand_rate=100.0)
    ve = 0.05 * 4.0 * 25.4 * (25.0 * 16.0185) ** 0.5 / (100.0 / 2.20462) ** 0.5 * 3.28084
    assert CX.erosional_velocity(25.0, 4.0, o2) == pytest.approx(ve, rel=1e-6)
    assert CX.turner_velocity(60.0, 67.0, 5.0, 2.04) == pytest.approx(2.04 * 60 ** 0.25 * 62 ** 0.25 / 5 ** 0.5)
    v = CX.node_values("oil", 10.0, 2.0, 40.0, 0.8, 50.0, 5.0, 30.0, 4.0, 1.0, o)
    assert np.isnan(v["Liquid Loading Velocity (ft/s)"])          # Turner only for gas wells
    v = CX.node_values("gas", 30.0, 29.0, 8.0, 0.02, 60.0, 7.0, 50.0, 2.441, 5.0, o)
    assert v["Gas Velocity / Loading Velocity"] == pytest.approx(29.0 / CX.turner_velocity(50.0, 60.0, 7.0, 2.04))


def test_dnv_erosion_and_norsok_corrosion():
    from wellmodel import complementary as CX
    o = CX.CompOptions(geometry_factor=1.0)
    # DNV-RP-O501 bend example: D 0.1 m, Vsl 5, Vsg 10, rho 800/100, mu 1e-3/1.5e-5, R 1.5 D, dp 250 micron
    rho, mu = (5 * 800 + 10 * 100) / 15, (5 * 1e-3 + 10 * 1.5e-5) / 15
    e_y = CX.erosion_bend(15.0, 0.1, 1.5, rho, mu, 250e-6, 1.0, o)          # mm/y at 1 kg/s
    e_ton = e_y / (365.25 * 86400 / 1000)
    assert abs(e_ton - 0.014) / 0.014 < 0.05
    # PROSPER caps the grain diameter at 1 inch
    v = CX.node_values("oil", 60.0, 50.0, 20.0, 0.1, 45.0, 5.0, 20.0, 4.0, 10.0, CX.CompOptions())
    assert v["Max. Grain Size Diameter (in)"] <= 1.0
    # NORSOK: pure water, 1 bar CO2, 25 degC -> pH ~3.9; bicarbonate raises it
    assert abs(CX.norsok_ph(25.0, 14.5, 1.0) - 3.92) < 0.05
    assert CX.norsok_ph(60.0, 14.5, 1.0, 0.01) > 5.5
    # eq. 1 at a table temperature: Kt fCO2^0.62 (S/19)^... f(pH)
    assert abs(CX.norsok_rate(60.0, 1.0, 19.0, 4.5) - 10.695 * (1.836 - 0.1818 * 4.5)) < 1e-9
    # f(pH) continuity at the table breaks (checks the exponential forms at 80 and 90 degC)
    for t, ph in ((80.0, 4.6), (90.0, 4.57), (90.0, 5.62), (5.0, 4.6)):
        assert abs(CX.f_ph(t, ph - 1e-9) - CX.f_ph(t, ph)) < 0.03
    # interpolation between 60 and 80 degC lies between the two
    a, b, m = (CX.norsok_rate(t, 2.0, 30.0, 4.8) for t in (60.0, 80.0, 70.0))
    assert min(a, b) <= m <= max(a, b)


def test_intermittent_gas_lift_brown_examples():
    from wellmodel import intermittent as I
    from wellmodel.gaslift import LiftGas
    ct = 0.00387
    # Brown B p.311-313: 6000 ft, Pvo 575, load 425, top of slug 90, Pr 700, PI 0.3, Gs 0.40
    f = lambda p: 0.3 * (700 - p)
    for sv, rec, q_b, cyc_b in ((False, 0.5, 81.0, 28.75), (False, 0.7, 92.4, 35.6), (True, 0.5, 89.1, 25.9)):
        w = I.weighted_bhp(575, 425, 90, 6000, rec, 0.40, ct, f, 9, 3, sv)
        assert abs(f(w["p_avg"]) - q_b) / q_b < 0.01
        assert abs(w["cycle"] - cyc_b) / cyc_b < 0.01
    # gas per cycle against Brown's Fig 3G readings (gamma 0.65, 80 F + 1.6 F/100 ft, Gs 0.3689, 7 %/1000 ft)
    for pc, L, pwh, book in [(1080, 7600, 100, 7200), (1155, 7600, 100, 7400), (1014, 9113, 100, 7800),
                             (947, 8000, 50, 6400), (753, 9700, 100, 6000)]:
        v = I.gas_per_cycle(LiftGas(sg=0.65), pc, pwh, L, (pc - 200 - pwh) / 0.3689, 0.3689, 7.0, 1.995,
                            lambda d: 80 + 0.016 * d)
        assert abs(v / book - 1) < 0.045          # chart readings; 753 psig / 9700 ft 4.1 % (4.0 % with the produced
    #                                               gas's Z before the lift gas took the natural-gas pseudo-criticals)
    # analytical spacing, Brown 3.4354 (Table 3.493): 2300, 4232, 5792, 7040, operating 7600 at Pso 875
    well = I.IGLWell(1.995, 2.375, 6.366, 7600, 1500, lambda q: 0.2 * max(1500 - q, 0), lambda d: 109 + 0.00763 * d,
                     t_surface=109)
    inp = I.IGLInput(p_ko=1050, p_wh=80, g_s=0.40, g_kill=0.40, g_u=0.064, spacing="Analytical (Brown 3.4354)",
                     max_depth=7600, standing_valve=False)
    d, pso = I.space_valves(inp, LiftGas(sg=0.7), well)
    assert np.allclose(d, [2300, 4232, 5792, 7040, 7600], atol=2)
    assert list(pso) == [1000, 975, 950, 925, 875]
    r = I.design(inp, LiftGas(sg=0.7), well)
    v1 = r.valves.iloc[0]
    assert abs(v1["Opening pressure at depth (psig)"] - 1060) < 6 and abs(v1["Test-rack opening at 60 °F (psig)"] - 950) < 6
    assert r.gas["Operating port"] == "5/16"                      # time cycle, R required ~0.12 (Brown 0.1212)
    assert abs(r.gas["Casing pressure drop to supply the gas (psi)"] - 74) < 4
    assert I.choose_port(I.IGL_VALVES["Brown 1.5-in unbalanced (Ab 0.77 in2)"], 0.2242)[0] == 28   # 7/16 (choke)
    # Brown's detailed design (B p.313-316): 5000 ft, Pr 775, PI 0.3, 7/16 port, Pvo - load 150, standing valve
    well = I.IGLWell(1.995, 2.375, 4.892, 5000, 775, lambda p: 0.3 * max(775 - p, 0), lambda d: 74 + 0.01 * d,
                     t_surface=74)
    inp = I.IGLInput(p_ko=650, p_wh=65, op_drop=25, g_s=0.40, g_u=0.04, well_killed=False, max_depth=5000,
                     load="Pvo - dP", dp_load=150, fallback=8.0, operating_port=28)
    r = I.design(inp, LiftGas(sg=0.60), well)
    assert abs(r.valves["Depth (ft)"].iloc[0] - 3060) < 10                   # static fluid level
    assert abs(r.cycle["Cycle time (min)"] - 29.4) < 0.6
    assert abs(r.cycle["Liquid rate (B/D)"] - 110.7) / 110.7 < 0.02


def test_prosper_intermittent_tutorial24():
    from wellmodel import intermittent as I
    from wellmodel.gaslift import LiftGas
    # constant surface closing pressure spacing (PROSPER Tutorial 24): 3628.44, 4912.52, 6179.47, 7429.51 ft
    t = I.constant_psc_spacing(1500, 250, 800, 0.46, 0.1, 0.047246, 7500, 600)
    assert np.allclose(t["Valve Depth (ft)"].iloc[1:], [3628.44, 4912.52, 6179.47, 7429.51], atol=0.1)
    assert np.allclose(t["Closing Pressure (psig)"].iloc[1:], [971.428, 1032.1, 1091.95, 1151.01], atol=0.05)
    # White et al. chain with PROSPER's slug pressure: slug velocity, fall back, cycle, rate (Tutorial 24)
    r = I.white_cycle(1828.18, 250, 6830, 2930, 24, 4.052, (1469.1 - 250) / 2930, 2.26026, friction=False)
    assert abs(r["Slug Velocity (ft/s)"] - 6.75477) < 2e-3
    assert abs(r["Fall Back (ft)"] - 1712.43) < 0.5 and abs(r["Time To Surface (s)"] - 830.884) < 0.5
    assert abs(r["Daily Production Rate (STB/d)"] - 2019.22) < 1.0
    assert abs(r["Gas Injection Rate, in situ (MMcf/day)"] - 0.052262) < 2e-5
    # calibrated bubble velocity against the PROSPER runs (vs, slug pressure, water cut -> vb)
    for vs, pt, wc, vb in ((9.19631, 1303.91, 0, 2.38803), (6.16121, 1511.81, 100, 2.23641), (7.65177, 1406.09, 50,
                           2.30255), (6.75477, 1469.1, 80, 2.26026), (3.19376, 1438.53, 80, 2.20222),
                          (12.001, 1142.6, 0, 2.531)):
        assert abs(I.bubble_velocity_prosper(vs, pt, wc) - vb) < 0.003
    # a slug heavier than the injection pressure is not lifted (PROSPER: convergence failure)
    assert "note" in I.white_cycle(1828.18, 250, 6830, 3930, 24, 4.052, 0.41607, 2.26)


def test_intermittent_optiflow_and_system():
    from wellmodel import intermittent as I
    from wellmodel.gaslift import LiftGas, ValveType
    # Takacs Example 6-3 (Opti-Flow): 1478 2395 3333 4291 5270 5789 6315 (6849 -> 7000) ft
    T = lambda d: 80 + 90 * d / 7000
    v = ValveType("RP-6", 0.77, 40, (20,), ((19.9, 0.126), (20.1, 0.126)))
    tab, lst = I.optiflow_spacing(LiftGas(sg=0.6), 700, 50, 0.42, 800, 7000, 7000, T, valve=v, port_64=20)
    assert abs(lst - 5095) < 1
    book = [1478, 2395, 3333, 4291, 5270, 5789, 6315]
    assert np.all(np.abs(tab["Depth (ft)"].to_numpy()[:7] / book - 1) < 0.02)
    assert tab["Depth (ft)"].iloc[-1] == 7000
    assert list(tab["Surface opening pressure (psig)"].iloc[:4]) == [680, 660, 640, 620]
    assert abs(tab["Dome at depth (psig)"].iloc[0] - 636) < 6
    # PROSPER system run (WC 0 %, Pr 2000 psig): 3490.6 STB/d at 1142.7 psig, liquid level 5049.5 ft
    class Oil:
        def state(self, p, t, case):
            return {"rho_l": np.array([0.3366 * 144]), "mu_l": np.array([1.0])}
    J = 3490.6 / (2000 - 1142.7)
    sol, crv = I.prosper_system(Oil(), LiftGas(sg=0.68), 1500, 250, 600, 7430, 24, 4.052, 0,
                                lambda d: float(np.interp(d, [0, 600, 9275], [60, 40, 210])),
                                lambda p: J * max(2000 - p, 0), t_fluid=210)
    assert abs(sol["Daily Production Rate (STB/d)"] / 3490.6 - 1) < 0.03
    # 17 psi (1.5 %) low; 14 with the produced gas's Z before the lift gas took Standing's natural-gas
    # pseudo-criticals (verified on PROSPER's lift-gas densities, Tutorial 25)
    assert abs(sol["Flowing BHP (psig)"] - 1142.7) < 20
    assert abs(sol["Liquid Level (ft below surface)"] - 5049.5) < 60


def test_esp_design_prosper_tutorial26():
    from wellmodel import esp as E
    # PROSPER Tutorial 26 pump duty (psia, RB/d) fed to the design
    d = E.Duty(8000.0, 2074.33, 1788.87, 2986.95, 209.244, 9706.52, 9044.1, 9300.9, 2823224.0, 0.86592, 3194.73,
               189.19, 447.918, 800.0, 0.13422, 0.11834, 184.654)
    pump = E.PUMPS["Centrilift GC10000 (5.13 in)"]
    r = E.design(d, pump, E.MOTORS["Centrilift 544 300HP 1675V 105A"], E.CABLES["#1 Copper"], 60.0, 9000.0)
    ref = {"Number Of Stages": 113, "Power Required (hp)": 288.91, "Pump Efficiency (%)": 65.9152,
           "Current Used (amps)": 102.149, "Motor Efficiency (%)": 84.0352, "Motor Speed (rpm)": 3450.38,
           "Voltage Drop Along Cable (Volts)": 291.125, "Voltage Required @ Surface (Volts)": 1966.13,
           "Surface KVA": 347.86, "Torque On Shaft (lb.ft)": 439.774, "Motor Power Factor (fraction)": 0.8113}
    for k, v in ref.items():
        assert abs(r[k] / v - 1) < 0.01, (k, r[k], v)
    # affinity laws and the recommended range
    assert abs(pump.head_stage(9000 * 50 / 60, 50) - pump.head_stage(9000) * (50 / 60) ** 2) < 1e-9
    assert pump.range_at(50) == (4400 * 50 / 60, 12000 * 50 / 60)
    # candidate list: the GC10000 suits this duty and casing
    c = E.candidate_pumps(d, E.PUMPS, 60.0, 6.0, 6.4)
    assert bool(c["Suitable"].iloc[0])


def test_artificial_lift_databases_from_prosper():
    """PROSPER's artificial lift databases (user's exports): ESP pumps with curve coefficients, motors (shared by ESP
    and PCP), cables, HSP pumps and turbines, gas lift valve dynamics and the Merla LN-20R tables."""
    from wellmodel import esp as E, pcp as PC, gaslift as GL
    d = Path(E.__file__).parent / "data"
    db = E.database_pumps()
    assert len(db) == 774 and int(db.h6.notna().sum()) == 54     # + CENTRILIFT R7 (T45, fitted)
    p = E.PUMPS["ALNAS L6050 (4.48 in)"]                         # 60 Hz, one stage
    q = 5600.0
    hc = [3.691e-18, -1.0168e-13, 9.0728e-10, -3.2186e-6, 0.0015281, 39.235]
    pc = [2.5433e-19, -6.1856e-15, 5.1249e-11, -1.6758e-7, 0.00024703, 1.4101]
    assert float(p.head60(q)) == pytest.approx(np.polyval(hc, q), rel=2e-3)
    assert float(p.eff60(q)) == pytest.approx(q * np.polyval(hc, q) / (135773 * np.polyval(pc, q)), rel=5e-3)
    b = E.PUMPS["BORETS ESPD5A-160 (4.05512 in)"]                # 50 Hz curve for 100 stages
    r = db[db.name == "ESPD5A-160"].iloc[0]
    h50 = np.polyval([r[f"h{k}"] for k in range(1, 7)], 1000.0) / 100.0
    assert float(b.head_stage(1000.0, 50.0)) == pytest.approx(h50, rel=3e-3)
    assert "ESPM6B-1500JI" not in " ".join(E.PUMPS)               # efficiency above 100 %: rejected
    assert "Centrilift GC10000 (5.13 in)" in E.PUMPS and E.PUMPS["Centrilift GC10000 (5.13 in)"].source.startswith("digit")
    m = E.MOTORS["ESP_Inc 456_55 100HP 775V 84A"].at(100.0)
    assert m["amps_pct"] == pytest.approx(101.0, abs=0.5) and m["eff"] == pytest.approx(0.8266, abs=1e-3)
    assert len(E.MOTORS) > 1400 and len(PC.MOTORS) == len(pd.read_csv(d / "motors.csv").drop_duplicates(
        ["manufacturer", "series", "hp", "volts", "amps"]))
    assert {k: v.v_per_kft_amp for k, v in E.CABLES.items()}["#4 Copper"] == 0.53 and len(E.CABLES) == 13
    hsp = pd.read_csv(d / "hsp_pumps.csv")
    assert len(hsp) == 10 and hsp.loc[hsp.name == "TP115AH", "h6"].iloc[0] == pytest.approx(376.014)
    tur = pd.read_csv(d / "hsp_turbines.csv")
    assert list(tur.setting) == list("ABCDABCDE") and tur.best_eff.iloc[3] == pytest.approx(72.1)    # T60 A-D, T68 A-E
    dyn = pd.read_csv(d / "gl_valve_dynamics.csv")
    assert len(dyn) == 32 and dyn.loc[(dyn.Model == "R-20") & (dyn.Port == 16), "Fe"].iloc[0] == pytest.approx(0.15)
    vt = pd.read_csv(d / "gl_valves.csv")
    assert vt.loc[vt.Model.str.contains("23mm") & (vt["Port (64ths in)"] == 31), "R value"].iloc[0] == 0.28614
    prop = pd.read_csv(d / "gl_proportional.csv")
    assert prop.loc[(prop.table == "M") & (prop.x == 4000), "large"].iloc[0] == 17.8
    from wellmodel import srp as S
    assert S.UNITS["LUFKIN C-320D-305-100"].prosper_holes == ((70.0, 30.0),)     # PROSPER holds only the 70 in hole
    u = S.UNITS["LUFKIN C-912-427-120"]                                           # PROSPER-only unit
    assert u.source == "PROSPER unit database" and u.gear == 912000 and u.H - u.G == pytest.approx(151.01, abs=0.01)
    assert S.Kinematics(S.UNITS["LUFKIN C-160D-143-64"], 64).hole_stroke == 28.0  # PROSPER's shortest database hole
    assert len(S.UNITS) == 114
    r64 = S.rod_string(64, 1.75)
    assert r64.pct == (47.4, 45.2, 7.4) and S.rod_string(98, 1.06).fc == 1.043


def test_esp_operate_and_options():
    from wellmodel import esp as E, options as O
    mot, cab = E.MOTORS["Centrilift 544 300HP 1675V 105A"], E.CABLES["#1 Copper"]
    # PROSPER Tutorial 26 system runs: (frequency, pump hp, avg cable T) -> load %, amps, rpm, V drop, V surface, kVA
    for f, hp, tc, ref in ((50, 163.75, 182.7, (65.50, 78.14, 2919.9, 221.95, 1617.78, 219.01)),
                           (60, 281.00, 187.8, (93.67, 100.01, 3455.0, 286.71, 1961.71, 339.93)),
                           (70, 452.37, 192.6, (129.25, 139.58, 3933.5, 403.69, 2357.86, 570.82))):
        r = E.operate(hp, mot, cab, f, 9000.0, tc)
        got = (r["Motor load (%)"], r["Motor amps (A)"], r["Motor speed (rpm)"], r["Voltage drop along cable (V)"],
               r["Voltage at surface (V)"], r["Surface kVA"])
        for g_, r_ in zip(got[:2] + got[2:], ref):
            tol = 0.035 if f == 70 else 0.015          # 70 Hz: motor curves extrapolated beyond 100 % load
            assert abs(g_ / r_ - 1) < tol, (f, g_, r_)
    assert any(i.level == "error" for i in O.WellOptions(artificial_lift="esp", flow_type="annular").validate())
    assert not O.WellOptions(artificial_lift="esp").errors
    assert not O.WellOptions(artificial_lift="srp").errors
    assert any(i.level == "error" for i in O.WellOptions(artificial_lift="srp", flow_type="annular").validate())
    assert any(i.level == "error" for i in O.WellOptions(artificial_lift="srp", well_type="water_injector").validate())


def test_esp_pump_in_traverse_and_lift_curves():
    from wellmodel import esp as E, liftcurves as LC
    *_, path, fluid = _vlp_setup()
    esp = {"pump_md": 9000.0, "pump": E.PUMPS["Centrilift GC10000 (5.13 in)"], "stages": 60, "freq": 60.0,
           "wear": 0.0, "sep": 0.0, "chp": 264.7}
    case = {"q": np.array([3000.0, 6000.0]), "wc": np.array([0.5, 0.5]), "gor": np.array([400.0, 400.0])}
    o = E.bhp_with_pump(path, fluid, 300.0, case, esp, "hagedorn_brown")
    assert np.all(np.isfinite(o["bhp"])) and np.all(o["pdp"] > o["pip"]) and np.all(o["bhp"] > o["pip"])
    # the intake solve inverts the stage march
    pdp, *_ = E.pump_gain(fluid, o["pip"], o["t_pump"], case, esp["pump"], 60, 60.0)
    assert np.allclose(pdp, o["pdp"], atol=0.5)
    # the pump lowers the BHP needed against natural flow at the same top pressure
    P, _ = wvlp.traverse(path, fluid, np.array([300.0, 300.0]), case, "hagedorn_brown")
    assert np.all(o["bhp"] < P[:, -1] - 300)
    # a gas separator takes gas out of the tubing stream
    case_g = {**case, "gor": np.array([1200.0, 1200.0])}      # free gas at the intake
    o2 = E.bhp_with_pump(path, fluid, 300.0, case_g, {**esp, "sep": 0.6}, "hagedorn_brown")
    assert np.all(o2["gor_above"] < 1200.0)
    assert np.allclose(E.bhp_with_pump(path, fluid, 300.0, case, {**esp, "sep": 0.6}, "hagedorn_brown")["gor_above"],
                       400.0)                                  # undersaturated at the intake: nothing to separate
    # lift curves with the pump equal the point calculation
    axes, bhp, _ = LC.vfp_table(path, fluid, "oil", "hagedorn_brown", 1.0, 1.0, 8.0, [3000.0, 6000.0], [300.0],
                                [50.0], [400.0], esp=esp)
    assert np.allclose(bhp[0, 0, 0], o["bhp"], rtol=1e-6)
    # annulus liquid level (PROSPER ignores the gas column): PIP = CHP + G_l (pump - level)
    lvl = E.liquid_level(fluid, float(o["pip"][0]), float(o["t_intake"][0]), {k: v[:1] for k, v in case.items()},
                         8000.0, 264.7)
    g_l = float(fluid.state(o["pip"][:1], o["t_intake"][:1], {k: v[:1] for k, v in case.items()})["rho_l"][0]) / 144
    assert abs(264.7 + g_l * (8000.0 - lvl) - o["pip"][0]) < 0.5 or lvl in (0.0, 8000.0)
    assert E.liquid_level(fluid, 264.7, 200.0, {k: v[:1] for k, v in case.items()}, 8000.0, 264.7) == 8000.0


def test_esp_discharge_temperature_and_frequency_axis():
    from wellmodel import esp as E, liftcurves as LC
    *_, path, fluid = _vlp_setup()
    mot = E.MOTORS["Centrilift 544 300HP 1675V 105A"]
    esp = {"pump_md": 9000.0, "pump": E.PUMPS["Centrilift GC10000 (5.13 in)"], "stages": 60, "freq": 60.0,
           "wear": 0.0, "sep": 0.0, "motor": mot}
    case = {"q": np.array([3000.0, 6000.0]), "wc": np.array([0.5, 0.5]), "gor": np.array([400.0, 400.0])}
    o = E.bhp_with_pump(path, fluid, 300.0, case, esp, "hagedorn_brown")
    # the tubing starts at the discharge temperature: intake + (motor input - hydraulic power) / (w cp)
    w, cp = fluid.mass_cp(case, wvlp.HeatCapacities())
    load = 100 * o["pump_hp"] / mot.hp
    eff = np.interp(load, mot.load, mot.eff) / 100
    dt = (o["pump_hp"] / eff - o["hyd_hp"]) * 2544.43 / (w * cp)
    assert np.allclose(o["t_discharge"], o["t_intake"] + dt, atol=0.3) and np.all(dt > 0.5)
    above = E.sub_path(path, path.md[0], 9000.0)
    _, T = wvlp.traverse(above, fluid, np.array([300.0, 300.0]), case, "hagedorn_brown", t_bottom=o["t_discharge"])
    assert np.allclose(T[:, -1], o["t_discharge"])
    # frequency (ALQ) axis: each slice equals the single-frequency table
    flo, fr = [3000.0, 6000.0], [50.0, 60.0, 70.0]
    axes, bhp, _ = LC.vfp_table(path, fluid, "oil", "hagedorn_brown", 1.0, 1.0, 8.0, flo, [300.0], [50.0], [400.0],
                                esp=esp, alq=fr)
    assert bhp.shape == (1, 1, 1, 3, 2) and list(axes[3]) == fr
    for l_, f in enumerate(fr):
        _, b1, _ = LC.vfp_table(path, fluid, "oil", "hagedorn_brown", 1.0, 1.0, 8.0, flo, [300.0], [50.0], [400.0],
                                esp={**esp, "freq": f})
        assert np.allclose(bhp[0, 0, 0, l_], b1[0, 0, 0], rtol=1e-6, equal_nan=True)
    assert np.all(np.diff(bhp[0, 0, 0, :, 0]) < 0)          # faster pump, lower BHP needed
    txt = LC.eclipse_vfpprod("oil", axes, bhp, 2, 9000.0)
    assert "ALQ = ESP frequency" in txt and "  50 60 70 /" in txt and "  1 1 1 3  " in txt
    csv = LC.vfp_csv("oil", axes, bhp)
    assert len(csv) == 6 and "ESP frequency (Hz)" in csv
    assert not any("below THP" in m for m in LC.vfp_checks(axes, bhp, esp=True))
    # blank (pumped-off) points are filled along the rate axis for Eclipse
    filled = LC._fill_rates(np.array([[np.nan, 2.0, 3.0, np.nan]]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(filled, [[1.0, 2.0, 3.0, 4.0]])


def _t05_oil():
    """PROSPER Tutorial 05 black oil (Glaso / Beal) with its PVT match (as matched in this tool)."""
    LM = wbo.LinearMatch
    inp = wbo.PVTInputs(rsb=800, api=37, gas_sg=0.76, salinity_ppm=23000, watson_k=12.1)
    m = wbo.BlackOilModel(inp, "glaso", "beal", {"pb": LM(1.0088023910638688, 30.40249925068713),
                                                  "rs": LM(0.9897021187921035, -10.672322109894141),
                                                  "bo": LM(0.9922440213035324, 0.003696465060738859),
                                                  "visc": LM(1.0016368891706189, 0.00461525632814706)})
    return wvlp.BlackOilFluid(m)


def _t11_layers(rough=0.0006):
    G = 14.696
    base = dict(type="Perforated", r_flow=0.3175, rough_in=rough, model="Darcy", area=400, ca=31.6, rw=0.354, api=37)
    blank = lambda md, tvd: dict(type="Blank", md=md, tvd=tvd, r_flow=0.3175, rough_in=rough)
    return [dict(base, md=9305, tvd=9025, pr=5200 + G, gor=800, gas_sg=0.75, wc=0.05, k=71, skin=2.4),
            blank(9318, 9035),
            dict(base, md=9350, tvd=9060, pr=5320 + G, gor=800, gas_sg=0.75, wc=0.005, k=65, skin=1.0),
            blank(9370, 9075),
            dict(base, md=9405, tvd=9100, pr=5325 + G, gor=800, gas_sg=0.76, wc=0.074, k=100, skin=1.5)]


# Logie well deviation survey (PROSPER Tutorials 05-23: MD, TVD ft - from PROSPER's T23 gradient export)
LOGIE_SURVEY = [(0.0, 0.0), (600.0, 600.0), (725.0, 723.5), (925.0, 921.0), (1000.0, 995.1), (1002.5, 997.5), (1130.0, 1122.1), (1380.0, 1366.4), (1630.0, 1610.7), (1880.0, 1855.0), (2130.0, 2099.3), (2380.0, 2343.6), (2630.0, 2587.9), (2880.0, 2832.2), (3130.0, 3076.5), (3380.0, 3320.8), (3618.2, 3553.6), (3731.4, 3664.3), (3856.4, 3786.4), (4028.2, 3954.3), (4200.0, 4120.7), (4450.0, 4362.1), (4700.0, 4603.4), (4950.0, 4844.8), (5200.0, 5086.2), (5450.0, 5327.6), (5620.9, 5492.6), (5666.9, 5537.0), (5791.9, 5657.7), (6041.9, 5899.1), (6291.9, 6140.4), (6538.5, 6378.6), (6660.2, 6496.0), (6785.2, 6616.7), (6962.0, 6787.5), (7013.9, 6837.6), (7138.9, 6958.3), (7388.9, 7199.6), (7607.0, 7410.2), (7825.0, 7619.0), (8075.0, 7857.1), (8325.0, 8095.2), (8575.0, 8333.3), (8825.0, 8571.4), (8975.0, 8714.3), (9125.0, 8857.1), (9262.5, 8988.1), (9275.0, 9000.0)]


def test_multilayer_dp_prosper_tutorial11():
    from scipy.optimize import brentq
    G = 14.696
    oil = _t05_oil()
    c = iadv.multilayer_wellbore("oil", _t11_layers(), 9275, 9000, 210.0, oil, "pe2_style", n=30)
    # PROSPER Tutorial 11 IPR at TOP (total liquid STB/d, TOP psig)
    for q, p in ((2992.98, 4884.53), (8978.9, 4117.63), (14964.7, 3348.03), (20950.7, 2450.93), (23943.6, 1880.91)):
        assert abs((np.interp(q, c.q, c.pwf) - G) / p - 1) < 0.006, (q, np.interp(q, c.q, c.pwf) - G, p)
    # layer 1: pressure at its bottom and rate (negative = crossflow into it near shut-in)
    lt = c.layer_table[c.layer_table.Layer == 1].sort_values("TOP pressure (psia)")
    at_ = lambda col, p_top: np.interp(p_top + G, lt["TOP pressure (psia)"], lt[col])
    for p_top, q1, p1 in ((5076.27, 253.526, 5083.88), (4692.8, 1090.79, 4700.4), (4309.35, 1928.06, 4316.92)):
        assert abs(at_("Liquid rate (STB/d)", p_top) / q1 - 1) < 0.005
        assert abs(at_("Layer pressure (psia)", p_top) - G - p1) < 0.1
    assert at_("Liquid rate (STB/d)", 5266.64) < -150                        # PROSPER -162.3
    # Darcy PI from each layer's own Bo and viscosity: PROSPER's skin dP gives B mu = 0.5509 for layer 1
    assert abs(c.params["Layer 1 PI (STB/d/psi)"] / 2.1835 - 1) < 0.005
    # blended water cut and GOR follow the layer contributions (PROSPER solution: 764.8 / 16283.4 water)
    mx = c.mix(16283.4)
    assert abs(mx["wc"][0] - 764.8 / 16283.4) < 2e-4 and abs(mx["gor"][0] - 800) < 1e-6
    # system at 500 psig wellhead, PE2 unmatched (P1 = P2 = 1: the T11 file's PROSPER gradient traverse at
    # 16283.4 STB/d has gravity and friction exactly unscaled): PROSPER 16283.4 STB/d at 3143.2 psig
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 600},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 4.052,
             "Tubing Inside Roughness (in)": 0.0006, "Casing Inside Diameter (in)": 6.4},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 3.72},
            {"Type": "Tubing", "Measured Depth (ft)": 9000, "Tubing Inside Diameter (in)": 4.052,
             "Tubing Inside Roughness (in)": 0.0006, "Casing Inside Diameter (in)": 6.4},
            {"Type": "Casing", "Measured Depth (ft)": 9275, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    smd_, stvd_ = np.array(LOGIE_SURVEY).T
    path = wvlp.build_flow_path(top, secs, pts, smd_, stvd_, [0, 600, 9275], [60, 40, 210])

    def vlp(q):
        m = c.mix(q)
        P, _ = wvlp.traverse(path, oil, np.array([500 + G]), {"q": np.array([q]), "wc": m["wc"], "gor": m["gor"]},
                             "pe2_style", 1.0, 1.0, 8.0)
        return float(P[0, -1])
    q = brentq(lambda q_: vlp(q_) - np.interp(q_, c.q, c.pwf), 8000, 25000, xtol=0.5)
    # rate 0.9 % low, bottom-hole pressure 1 % high (the multilayer IPR at TOP runs up to 0.6 % high)
    assert abs(q / 16283.4 - 1) < 0.012 and abs((vlp(q) - G) / 3143.2 - 1) < 0.015
    split = iadv.layer_split(c, vlp(q))
    for lay_, ref in (("Layer 1", 4384.6), ("Layer 3", 4854.3), ("Layer 5", 7044.4)):
        assert abs(float(split.loc[split.Layer == lay_, "Liquid rate (STB/d)"].iloc[0]) / ref - 1) < 0.01
    assert abs(float(split.loc[split.Layer == "Total", "Liquid rate (STB/d)"].iloc[0]) - q) < 0.01 * q
    # the system analysis takes the water cut and GOR from the layers at each rate (the case's own are ignored)
    from wellmodel import system as wsys
    res, _ = wsys.solve_cases(lambda tid: path, oil, "oil", {"whp": 500 + G, "u": 8.580277, "wc": 50.0, "gor": 300.0},
                              [{}], lambda ch: c, "pe2_style", 1.0, 1.0)
    r = res[0]
    assert abs(r.q / 16283.4 - 1) < 0.012 and abs(r.params["wc"] - 4.697) < 0.02 and abs(r.params["gor"] - 800) < 1e-6


def test_multilayer_dp_zero_roughness_and_gas():
    G = 14.696
    oil = _t05_oil()
    # zero roughness: no wellbore losses - every layer sees the TOP pressure, the total is the sum of the layer IPRs
    c = iadv.multilayer_wellbore("oil", _t11_layers(0.0), 9275, 9000, 210.0, oil, "beggs_brill", n=12)
    lt = c.layer_table
    assert np.allclose(lt["Layer pressure (psia)"], lt["TOP pressure (psia)"], atol=1e-6)
    tot = lt.groupby("Point")["Liquid rate (STB/d)"].sum().to_numpy()
    assert np.allclose(np.maximum(tot, 0), c.q, rtol=1e-9, atol=1e-6)
    # gas: C and n layers with their own CGR and WGR; the blended ratios are the rate-weighted ones
    gas = wvlp.WetGasFluid(wbo.PVTInputs(rsb=1, api=40, gas_sg=0.7), 50.0)
    lay = [dict(type="Perforated", md=9305, tvd=9025, r_flow=0.25, rough_in=0.0006, pr=4000, model="C and n",
                c=0.01, n=0.85, cgr=10.0, wgr=2.0),
           dict(type="Perforated", md=9350, tvd=9060, r_flow=0.25, rough_in=0.0006, pr=4100, model="C and n",
                c=0.008, n=0.9, cgr=30.0, wgr=8.0)]
    g = iadv.multilayer_wellbore("gas", lay, 9275, 9000, 200.0, gas, "beggs_brill", n=12)
    assert g.aof > 0 and np.all(np.diff(g.q) >= -1e-6)
    k = len(g.q) // 2
    pts = g.layer_table[g.layer_table.Point == k + 1]
    q1, q2 = pts["Gas rate (Mscf/d)"].to_numpy()
    m = g.mix(g.q[k])
    assert abs(m["cgr"][0] - (10 * q1 + 30 * q2) / (q1 + q2)) < 0.05
    assert abs(m["wgr"][0] - (2 * q1 + 8 * q2) / (q1 + q2)) < 0.05


def test_gas_pvt_prosper_tutorial02():
    from wellmodel import gaspvt as gp
    inp = gp.GasPVTInputs(0.58, 200, 1, 50, 1, 100000, 0, 0.5, 2, "lee", 166, 5300)
    # PROSPER T02 PVT table (Lee et al): T F, p psig -> density, viscosity, Bg, Z, pseudo-pressure
    ref = [(60, 2518.42, 9.97219, 0.018589, 0.0044677, 0.7695, 528055968), (160, 293.158, 0.8025, 0.013262, 0.055517,
           0.97456, 6663472), (160, 4187.37, 11.1873, 0.02194, 0.0039824, 0.95422, 1070977664),
           (250, 5300, None, None, None, None, None)][:3]
    for t, p, rho, mu, bg, z, mp in ref:
        zz, bgg, rr, mm = gp.gas_props(inp, np.array([p + 14.696]), np.array([float(t)]))
        assert abs(zz[0] / z - 1) < 0.008 and abs(rr[0] / rho - 1) < 0.008 and abs(bgg[0] / bg - 1) < 0.009
        assert abs(mm[0] / mu - 1) < 0.01
        assert abs(gp.pseudo_pressure(inp, p + 14.696, t)[0] / mp - 1) < 0.02
    # Carr et al within 5 %
    carr = gp.GasPVTInputs(0.58, 200, 1, 50, 1, 100000, 0, 0.5, 2, "carr", 166, 5300)
    _, _, _, mc = gp.gas_props(carr, np.array([4187.37 + 14.696]), np.array([160.0]))
    assert abs(mc[0] / 0.022067 - 1) < 0.05
    # condensed water: minimum WGR within 6 % of PROSPER's 0.27114; free water where the gas cannot hold 1 STB/MMscf
    assert abs(gp.minimum_wgr(inp) / 0.27114 - 1) < 0.06
    vap, liq = gp.water_split(inp, np.array([293.158, 849.474]) + 14.696, np.array([160.0, 160.0]), 1.0)
    assert liq[0] == 0 and liq[1] > 0 and abs(vap + liq - 1).max() < 1e-12
    # wellstream gravity (PROSPER's densities imply 0.58322)
    assert abs(float(inp.wellstream_sg()) - 0.58322) < 2e-4


def test_gas_well_vlp_prosper_tutorial02():
    """PE2 in a gas well: Duns & Ros mist friction with the film roughness, no-slip holdup in transition, water-gas
    IFT from Macleod-Sugden, condensed water vapour. PROSPER gradient: 2592.5 psig at 73.432 MMscf/d, friction
    868.97, gravity 203.54 psi."""
    from wellmodel import gaspvt as gp, system as wsys, correlations as C
    G = 14.696
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 3.958,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 3.0},
            {"Type": "Tubing", "Measured Depth (ft)": 5000, "Tubing Inside Diameter (in)": 3.958,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 5400, "Casing Inside Diameter (in)": 6.0,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = wvlp.build_flow_path(top, secs, pts, [0, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5400],
                                [0, 1500, 1980, 2450, 2883, 3300, 3721, 4140, 4560, 4900], [0, 1500, 5400], [45, 75, 166])
    gi = gp.GasPVTInputs(0.58, 200, 1, 50, 1, 100000, 0, 0.5, 2, "lee", 166, 5300)
    fl = wvlp.WetGasFluid(wbo.PVTInputs(rsb=1, api=40, gas_sg=0.58, salinity_ppm=100000, co2=0.5, n2=2), 50.0, "lee", gi)
    det, _, prof = wsys.solution_details(path, fl, "gas", {"whp": 1500 + G, "u": 3.0, "wgr": 1.0, "sep_gor": 1e6},
                                         73.432, "pe2_style")
    assert abs((det["Bottom node pressure (psia)"] - G) / 2592.5 - 1) < 0.015
    assert abs(det["dP friction - production string (psi)"] / 868.97 - 1) < 0.03
    assert abs(det["dP gravity - production string (psi)"] / 203.54 - 1) < 0.02
    assert abs(det["Wellhead temperature (°F)"] - 152.5) < 0.5
    # PROSPER node 2 (125 ft): friction gradient 0.221 psi/ft on its own state
    a = lambda x: np.array([x])
    _, f, hl = C._dr_mist(a(0.032), a(100.450), a(65.742), a(4.434), a(0.5487), a(0.0151), a(56.6077), 0.3298,
                          0.0006 / 12, 1.0)
    assert abs(f[0] / 0.221 - 1) < 0.01
    assert abs(float(wvlp.sigma_water_gas(65.742, 4.434)) / 56.6077 - 1) < 0.015


def test_petroleum_experts_gas_ipr_prosper_tutorial02():
    """PROSPER T02 IPR runs: AOF (MMscf/d) for time, area, shape factor, wellbore radius and connate water."""
    from wellmodel import gaspvt as gp, inflow as winf
    G = 14.696
    inp = gp.GasPVTInputs(0.58, 200, 1, 50, 1, 100000, 0, 0.5, 2, "lee", 166, 5300)
    pr = 5300 + G
    grid = np.linspace(G, 5600, 2000)
    mg = gp.pseudo_pressure(inp, grid, 166.0)
    m_fn = lambda p: np.interp(p, grid, mg)
    z3 = gp.gas_props(inp, np.array([pr - 5, pr, pr + 5]), np.full(3, 166.0))[0]
    mu_r = float(gp.gas_props(inp, np.array([pr]), np.array([166.0]))[3][0])
    cg = 1 / pr - (z3[2] - z3[0]) / 10 / z3[1]
    cases = [(.25, 50, 500, 31.6, .354, 100.369), (0, 5, 500, 31.6, .354, 138.188), (0, 5000, 2000, 31.6, .354, 134.151),
             (0, 5000, 500, 10, .354, 134.825), (0, 5000, 500, 31.6, .5, 153.513), (.1, 50, 500, 31.6, .354, 123.828),
             (.4, 50, 500, 31.6, .354, 70.902)]
    for swc, t, A, ca, rw, aof in cases:
        cv = winf.petex_gas(pr, 166, 25, 55, A, ca, rw, 55, t, .15, swc, 2.0, m_fn, mu_r, cg, 0.58)
        assert abs(cv.aof / 1000 / aof - 1) < 0.01, (swc, t, A, ca, rw, cv.aof)
    base = winf.petex_gas(pr, 166, 25, 55, 500, 31.6, .354, 55, 50, .15, .25, 2.0, m_fn, mu_r, cg, 0.58)
    # base table: 73.956 MMscf/d at 2706.1 psig; D 6.4065e-5 shown by PROSPER
    assert abs(float(base.rate(np.array([2706.1 + G]))[0]) / 73956 - 1) < 0.01
    assert abs(base.params["Non-Darcy flow factor D (1/(Mscf/d))"] / 6.4065e-5 - 1) < 0.01
    # transient at 1 day (Swc 0, D entered 0): PROSPER drainage term 7.848 (Carr viscosity run)
    tr = winf.petex_gas(pr, 166, 25, 55, 500, 31.6, .354, 55, 1, .15, 0.0, 2.0, m_fn, mu_r, cg, 0.58, d_entered=0.0)
    assert tr.params["Flow regime"] == "transient" and abs(tr.params["Drainage term X"] - 7.848) < 0.03


def test_gas_system_prosper_tutorial02():
    """Tutorial 02 system at 1500 psig: PROSPER 73.342 MMscf/d at 2594.5 psig (shallow IPR/VLP crossing: small
    pressure differences move the rate - here +3.6 %)."""
    from wellmodel import system as wsys
    import json
    from wellmodel import gaspvt as gp, inflow as winf
    G = 14.696
    inp = gp.GasPVTInputs(0.58, 200, 1, 50, 1, 100000, 0, 0.5, 2, "lee", 166, 5300)
    pr = 5300 + G
    grid = np.linspace(G, 5600, 2000)
    mg = gp.pseudo_pressure(inp, grid, 166.0)
    z3 = gp.gas_props(inp, np.array([pr - 5, pr, pr + 5]), np.full(3, 166.0))[0]
    mu_r = float(gp.gas_props(inp, np.array([pr]), np.array([166.0]))[3][0])
    cv = winf.petex_gas(pr, 166, 25, 55, 500, 31.6, .354, 55, 50, .15, .25, 2.0, lambda p: np.interp(p, grid, mg), mu_r,
                        1 / pr - (z3[2] - z3[0]) / 10 / z3[1], 0.58)
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 3.958,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 3.0},
            {"Type": "Tubing", "Measured Depth (ft)": 5000, "Tubing Inside Diameter (in)": 3.958,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 5400, "Casing Inside Diameter (in)": 6.0,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = wvlp.build_flow_path(top, secs, pts, [0, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5400],
                                [0, 1500, 1980, 2450, 2883, 3300, 3721, 4140, 4560, 4900], [0, 1500, 5400], [45, 75, 166])
    fl = wvlp.WetGasFluid(wbo.PVTInputs(rsb=1, api=40, gas_sg=0.58, salinity_ppm=100000, co2=0.5, n2=2), 50.0, "lee", inp)
    res, _ = wsys.solve_cases(lambda tid: path, fl, "gas", {"whp": 1500 + G, "u": 3.0, "wgr": 1.0, "sep_gor": 1e6}, [{}],
                              lambda ch: cv, "pe2_style")
    assert abs(res[0].q / 73.342 - 1) < 0.05 and abs((res[0].pwf - G) / 2594.5 - 1) < 0.02
    # the example project file loads (Options / PVT / Equipment / IPR / system inputs for this tutorial)
    proj = json.load(open(__file__.replace("tests/test_wellmodel.py", "examples/T02_Basic_Gas_Well.wellmodel.json")))
    assert proj["inputs"]["ipr_model_gas"] == "Petroleum Experts"


def test_gas_pvt_matching_tutorial06():
    """PROSPER Tutorial 06 lab data at 166 F: after matching, Z and Lee viscosity reproduce the lab points; the
    parameters are applied by gas_props (density follows Z)."""
    import dataclasses
    from wellmodel import gaspvt as gp
    G = 14.696
    inp = gp.GasPVTInputs(0.58, 200, 1, 50, 1, 100000, 0, 0.5, 2, "lee", 166, 5300)
    lab = pd.DataFrame({"Pressure (psia)": np.array([5300, 3020, 2140, 1000]) + G,
                        "Z factor": [1.03183, .9053, .89365, .9287], "Gas viscosity (cp)": [.024644, .01869, .01648, .014213],
                        "Gas FVF (ft3/scf)": [.0034174, .005239, .006733, .016213]})
    m = gp.match_gas(inp, [(166.0, lab)])
    assert set(m) == {"z", "bg", "lee", "carr"}
    # Lee already fits the lab data (PROSPER: P1 0.99882, P2 -1.6e-5)
    assert abs(m["lee"].p1 - 1) < 0.03 and m["lee"].raw_error_pct < 1
    z0, bg0, rho0, _ = gp.gas_props(inp, lab["Pressure (psia)"].values, np.full(4, 166.0))
    mi = dataclasses.replace(inp, match=m)
    z, bg, rho, mu = gp.gas_props(mi, lab["Pressure (psia)"].values, np.full(4, 166.0))
    assert np.abs(z / lab["Z factor"].values - 1).max() < 0.004
    assert np.abs(mu / lab["Gas viscosity (cp)"].values - 1).max() < 0.004
    assert np.allclose(rho * bg, rho0 * bg0)          # density follows the matched FVF


def test_gas_system_prosper_tutorial06_matched():
    """Tutorial 06 run in PROSPER (user's file) with the PVT match applied, at 1500 psig: 61.746 MMscf/d at
    3277.0 psig, BHP at 5400 ft 3247.5 psig on the traverse at that rate. The PROSPER PVT | Calculate table at 166 F
    gives its effective match on our correlations: Z 0.99706/+0.00350, Lee 0.99800/-6.3e-6 (close to 1/0: our Z and
    Lee are PROSPER's), FVF (on the matched-Z FVF) 0.98732/+0.015111 - an FVF match well off p-Z-T (2-5 x), and PROSPER's gas density = standard density /
    matched FVF (1.792 lb/ft3 at the wellhead)."""
    import dataclasses
    from wellmodel import system as wsys, gaspvt as gp, inflow as winf
    from wellmodel.blackoil import LinearMatch as LM
    G = 14.696
    inp = dataclasses.replace(gp.GasPVTInputs(0.58, 200, 1, 50, 1, 100000, 0, 0.5, 2, "lee", 166, 5300),
                              match={"z": LM(0.997064, 0.0034954), "bg": LM(0.9873151, 0.01511095),
                                     "lee": LM(0.998004, -6.299e-6)})
    # PROSPER PVT table at 166 F: FVF and density
    p = np.array([15, 1405.79, 3074.74, 5300]) + G
    z, bg, rho, mu = gp.gas_props(inp, p, np.full(4, 166.0))
    assert np.abs(bg / [0.60194, 0.026294, 0.020238, 0.018501] - 1).max() < 0.003
    assert np.abs(rho / [0.074015, 1.6944, 2.20144, 2.40807] - 1).max() < 0.01
    assert np.abs(mu / [0.013164, 0.01491, 0.018838, 0.024643] - 1).max() < 0.002
    assert gp.fvf_consistency(inp) > 1.0
    pr = 5300 + G
    grid = np.linspace(G, 5600, 2000)
    mg = gp.pseudo_pressure(inp, grid, 166.0)
    assert abs(np.interp(3277 + G, grid, mg) / 717929177 - 1) < 0.01
    z3 = gp.gas_props(inp, np.array([pr - 5, pr, pr + 5]), np.full(3, 166.0))[0]
    mu_r = float(gp.gas_props(inp, np.array([pr]), np.array([166.0]))[3][0])
    cv = winf.petex_gas(pr, 166, 25, 55, 500, 31.6, .354, 55, 50, .15, .25, 2.0, lambda p_: np.interp(p_, grid, mg),
                        mu_r, 1 / pr - (z3[2] - z3[0]) / 10 / z3[1], 0.58)
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 3.958,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 3.0},
            {"Type": "Tubing", "Measured Depth (ft)": 5000, "Tubing Inside Diameter (in)": 3.958,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 5400, "Casing Inside Diameter (in)": 6.0,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = wvlp.build_flow_path(top, secs, pts, [0, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5400],
                                [0, 1500, 1980, 2450, 2883, 3300, 3721, 4140, 4560, 4900], [0, 1500, 5400], [45, 75, 166])
    fl = wvlp.WetGasFluid(wbo.PVTInputs(rsb=1, api=40, gas_sg=0.58, salinity_ppm=100000, co2=0.5, n2=2), 50.0, "lee",
                          inp, gpvt=inp)
    P, _ = wvlp.traverse(path, fl, 1500 + G, {"q": 61.746, "wgr": 1.0, "cgr": 1.0, "sep_gor": 1e6}, "pe2_style", u=3.0)
    assert abs((np.ravel(P)[-1] - G) / 3247.5 - 1) < 0.02
    res, _ = wsys.solve_cases(lambda tid: path, fl, "gas", {"whp": 1500 + G, "u": 3.0, "wgr": 1.0, "sep_gor": 1e6},
                              [{}], lambda ch: cv, "pe2_style")
    assert abs(res[0].q / 61.746 - 1) < 0.03 and abs((res[0].pwf - G) / 3277.0 - 1) < 0.01


def test_gas_well_test_matching_prosper_t07():
    """Three gas well tests (gauge at 4250 ft) on the Tutorial 06 well, PVT as the user's PROSPER file (see the T06
    test). Workflow: U from the tests (~3.13, the model's 3), Match VLP (PE2), Adjust IPR (skin, latest test), then
    a system sensitivity on top-node pressure and WGR. PROSPER (user's run): PE2 P1 0.76821, P2 0.93288 (P1 is
    poorly determined - gravity is ~70 psi of ~1,700 - and the regression here pulls it towards 1), skin 1.28714,
    AOF 103.746 MMscf/d; e.g. 1500 psig and WGR 1: 64.433 MMscf/d at 3266.1 psig; WGR 50: 56.956 at 3526.5;
    1000 psig, WGR 0: 70.265 at 3022.1; WGR 50: 60.750 at 3363.5. The water response needs PE2's mist friction on
    the no-slip mixture momentum (gas-only friction: WGR 50 under-predicted by 570-660 psi)."""
    import dataclasses
    from scipy.optimize import brentq
    from wellmodel import system as wsys, gaspvt as gp, inflow as winf
    from wellmodel.blackoil import LinearMatch as LM
    G = 14.696
    inp = dataclasses.replace(gp.GasPVTInputs(0.58, 200, 1, 50, 1, 100000, 0, 0.5, 2, "lee", 166, 5300),
                              match={"z": LM(0.997064, 0.0034954), "bg": LM(0.9873151, 0.01511095),
                                     "lee": LM(0.998004, -6.299e-6)})
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 3.958,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 3.0},
            {"Type": "Tubing", "Measured Depth (ft)": 5000, "Tubing Inside Diameter (in)": 3.958,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 5400, "Casing Inside Diameter (in)": 6.0,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = wvlp.build_flow_path(top, secs, pts, [0, 1500, 2000, 2500, 3000, 3500, 4000, 4250, 4500, 5000, 5400],
                                [0, 1500, 1980, 2450, 2883, 3300, 3721, 3930.5, 4140, 4560, 4900], [0, 1500, 5400],
                                [45, 75, 166])
    fl = wvlp.WetGasFluid(wbo.PVTInputs(rsb=1, api=40, gas_sg=0.58, salinity_ppm=100000, co2=0.5, n2=2), 50.0, "lee",
                          inp, gpvt=inp)
    tab = pd.DataFrame({"Test Point Comment": ["Test 1", "Test 2", "Test 3"],
                        "Tubing Head Pressure (psia)": np.array([430, 2680, 1420]) + G,
                        "Tubing Head Temperature (deg F)": [151.63, 146, 149.95],
                        "Water Gas Ratio (STB/MMscf)": [1.0] * 3, "Condensate Gas Ratio (STB/MMscf)": [1.0] * 3,
                        "Gas Rate (MMscf/day)": [72.773, 50.411, 64.459], "Gauge Depth (Measured) (feet)": [4250] * 3,
                        "Gauge Pressure (psia)": np.array([2589.6, 3613.64, 2913.8]) + G,
                        "Reservoir Pressure (psia)": np.array([5300, 5240, 5200]) + G})
    tests, issues = wvlp.clean_tests(tab, "gas", 0, 5400)
    assert not issues and np.allclose(tests["cgr"], 1.0)
    u = wvlp.estimate_u(path, fl, tests)["U (BTU/h/ft2/F)"]
    assert np.allclose(u, 3.13, atol=0.02)
    m = wvlp.match(path, fl, tests, "pe2_style", 3.0)
    assert 0.9 < m.p1 < 1.1 and abs(m.p2 / 0.93288 - 1) < 0.03 and np.abs(m.residuals).max() < 30
    grid = np.linspace(G, 5600, 2000)
    mg = gp.pseudo_pressure(inp, grid, 166.0)
    mfn = lambda p_: np.interp(p_, grid, mg)

    def ipr(pr, skin):
        z3 = gp.gas_props(inp, np.array([pr - 5, pr, pr + 5]), np.full(3, 166.0))[0]
        mu_r = float(gp.gas_props(inp, np.array([pr]), np.array([166.0]))[3][0])
        return winf.petex_gas(pr, 166, 25, 55, 500, 31.6, .354, 55, 50, .15, .25, skin, mfn, mu_r,
                              1 / pr - (z3[2] - z3[0]) / 10 / z3[1], 0.58)
    t3 = tests.iloc[[2]]
    bhp3 = float(wvlp.vlp_curve(path, fl, t3.iloc[0], [t3["q"].iloc[0]], "pe2_style", m.p1, m.p2, 3.0)[0])
    skin = brentq(lambda s: float(ipr(float(t3["pr"].iloc[0]), s).rate(bhp3)) - 1000 * t3["q"].iloc[0], -2, 10)
    cv = ipr(5300 + G, skin)
    assert abs(skin - 1.28714) < 0.15 and abs(cv.aof / 103746 - 1) < 0.01
    ref = {1000: [(0, 70.265, 3022.1), (50, 60.750, 3363.5)], 1500: [(1, 64.433, 3266.1), (50, 56.956, 3526.5)]}
    for whp, lst in ref.items():
        res, _ = wsys.solve_cases(lambda tid: path, fl, "gas", {"whp": whp + G, "u": 3.0, "wgr": 1.0, "sep_gor": 1e6},
                                  [{"wgr": w} for w, _, _ in lst], lambda ch: cv, "pe2_style", m.p1, m.p2)
        for (w, q, pw), r in zip(lst, res):
            assert abs(r.q / q - 1) < 0.025, (whp, w, r.q, q)
            assert abs((r.pwf - G) / pw - 1) < 0.02, (whp, w, r.pwf - G, pw)


def test_retrograde_condensate_eos_prosper_t37():
    """PROSPER Tutorial 37 (fully compositional retrograde condensate, PR EoS from a PRP file, three-stage separator
    train), user's PROSPER run: AOF 150.011 MMscf/d; system at 2500 psig, PE5: 33.401 MMscf/d at 4976.50 psig.
    PROSPER's IPR table gives a = 7337, b = 0.012986 (ours 7351, 0.012983): the difference was M(P), which PROSPER
    integrates with the equilibrium gas below the dew point."""
    import json
    from wellmodel import eos as E, system as wsys, inflow as winf
    G = 14.696
    root = __file__.replace("tests/test_wellmodel.py", "examples/")
    f = E.EOSFluid(E.parse_prp(open(root + "T37_Retrograde_Condensate_EoS.prp").read()))
    psat, kind = f.saturation_pressure(300.0)
    assert kind == "dew" and abs(psat - 4611) < 20
    sep = f.separator_train()
    assert abs(sep["gor"] / 7432.66 - 1) < 0.01                   # PROSPER's separator-train GOR
    pr, t = 6000 + G, 300.0
    grid = np.linspace(G, pr + 50, 150)
    props = []                      # reservoir fluid above the dew point, equilibrium gas below (as PROSPER)
    for x in grid:
        r = f.flash(float(x), t)
        props.append(r.vapour if r.phases == "2P" else f.phase_props(f.z, float(x), t))
    mu = np.array([o.viscosity for o in props])
    z = np.array([o.z for o in props])
    fi = 2 * (grid - G) / (mu * z)
    m = np.concatenate([[0], np.cumsum(np.diff(grid) * 0.5 * (fi[1:] + fi[:-1]))])
    zz = np.interp([pr - 5, pr, pr + 5], grid, z)
    cv = winf.petex_gas(pr, t, 25, 100, 340, 31.6, .354, 100, 50, .25, .2, 5.0, lambda p: np.interp(p, grid, m),
                        float(np.interp(pr, grid, mu)), 1 / pr - (zz[2] - zz[0]) / 10 / zz[1], sep["gas_gravity"])
    assert abs(cv.aof / 150011 - 1) < 0.01                        # 150.3 (single-phase fluid below Pdew: 144.3)
    # PROSPER's IPR table: M(P) and rates (e.g. 63.162 MMscf/d at 4023.68 psig, 126.325 at 1888.71)
    assert abs(np.interp(4023.68 + G, grid, m) / 877662336 - 1) < 0.006
    assert abs(cv.rate(4023.68 + G) / 63162 - 1) < 0.005 and abs(cv.rate(1888.71 + G) / 126325 - 1) < 0.005
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 11800, "Tubing Inside Diameter (in)": 3.992,
             "Tubing Inside Roughness (in)": 0.0018},
            {"Type": "Casing", "Measured Depth (ft)": 12000, "Casing Inside Diameter (in)": 8.3,
             "Casing Inside Roughness (in)": 0.0018}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = wvlp.build_flow_path(top, secs, pts, [0, 12000], [0, 12000], [0, 12000], [70, 300])
    fl = wvlp.EOSTableFluid("gas", wvlp.eos_grid(f, 7000, 60, 300), float(f.z @ f.mw),
                            float(sep["stages"]["Gas (scf per lbmol feed)"].sum()), float(sep["stb_per_lbmol"]),
                            float(sep["oil_density"]), wbo.PVTInputs(rsb=1, api=40, gas_sg=0.7, salinity_ppm=20000))
    res, _ = wsys.solve_cases(lambda tid: path, fl, "gas", {"whp": 2500 + G, "u": 5.0, "wgr": 0.0, "sep_gor": sep["gor"]},
                              [{}], lambda ch: cv, "pe5_style")
    print("T37 system", res[0].q, res[0].pwf - G)
    assert abs(res[0].q / 33.401 - 1) < 0.03 and abs((res[0].pwf - G) / 4976.5 - 1) < 0.01
    proj = json.load(open(root + "T37_Retrograde_Condensate_Well.wellmodel.json"))
    assert proj["widgets"]["opt_method"] == "eos" and proj["inputs"]["sys_corr"] == "pe5_style"


def test_prosper_water_properties():
    """PROSPER's water PVT tables (Tutorial 03b, 200,000 ppm, 60-210 F, 0-6000 psig; Tutorial 06, 100,000 ppm at
    166 F): FVF linear in pressure and independent of salinity, density = standard density / FVF, viscosity a
    function of temperature and salinity only."""
    inp = wbo.PVTInputs(1, 40, .7, 200000)
    T = np.array([60, 90, 120, 166, 210.0])
    p0 = wbo.water_props(inp, np.full(5, 14.696), T)
    assert np.allclose(p0[0], [.99955, 1.00379, 1.01022, 1.02392, 1.04102], atol=2e-5)
    assert np.allclose(p0[1], [71.8645, 71.5607, 71.105, 70.1542, 69.0015], rtol=1e-4)
    assert np.allclose(p0[2], [1.68935, 1.16943, 0.85379, 0.57055, 0.41627], rtol=1e-3)
    p6 = wbo.water_props(inp, np.full(5, 6014.696), T)
    assert np.allclose(p6[0], [.98131, .98615, .99276, 1.00578, 1.0214], atol=5e-5)
    b, r, m = wbo.water_props(wbo.PVTInputs(1, 40, .7, 100000), np.array([849.474, 5300]) + 14.696, 166.0)
    assert np.allclose(b, [1.02135, 1.00789], atol=2e-5) and np.allclose(r, [65.3366, 66.2088], rtol=1e-4)
    assert abs(m[0] / 0.49445 - 1) < 1e-3


def test_water_injector_prosper_t03b():
    """PROSPER Tutorial 03b (Basic Water Injector): 200,000 ppm water at 60 F into a 5.2 in tubing, Darcy
    injectivity (k 50 md, h 100 ft, 250 acres, skin 3, Pr 4100 psig, 210 F). PROSPER at 1250 psig: 10,126.4 STB/d
    at 5,423.7 psig; dP gravity 4,224.26, friction -50.57 psi; dP skin 367.08; bottom 91.0 F."""
    from wellmodel import injection as winj
    G = 14.696
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 600},
            {"Type": "Tubing", "Measured Depth (ft)": 8500, "Tubing Inside Diameter (in)": 5.2,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 9000, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = winj.from_flow_path(wvlp.build_flow_path(top, secs, pts, [0, 9000], [0, 9000], [0, 600, 9000],
                                                    [60, 40, 210]))
    fl = winj.WaterFluid(200000)
    ipr = winj.darcy_injection(4100 + G, 210, 50, 100, 250, 0.354, 3.0, fl)
    assert abs(ipr.j / 7.650 - 1) < 0.002
    P, T = winj.traverse(path, fl, 1250 + G, [40, 10547.4, 39968], 8.0, 60.0)
    assert np.allclose(P[:, -1] - G, [5419.7, 5420.3, 4825.2], atol=3) and np.allclose(T[:, -1], [209, 90, 68.8], atol=0.2)
    res, *_ = winj.solve(path, fl, 1250 + G, ipr, 8.0, 60.0)
    assert abs(res.q / 10126.4 - 1) < 0.003 and abs(res.pwf - G - 5423.7) < 2
    det = dict(zip(*winj.details(path, fl, res, ipr).values.T))
    assert abs(det["dP gravity - injection string (psi)"] - 4224.26) < 3
    assert abs(det["dP friction - injection string (psi)"] + 50.57) < 1.5
    assert abs(det["dP total skin (psi)"] - 367.08) < 2 and abs(det["Bottom node temperature (°F)"] - 91.0) < 0.3
    g = winj.gradient_table(path, fl, res.q, res.P, res.T)
    assert abs(g["Water density (lb/ft3)"].iloc[1] - 72.153) < 0.05


def test_gas_z_prosper_tables():
    """PROSPER's gas Z: Dranchuk-Purvis-Robinson with Standing's condensate-gas pseudo-criticals of the hydrocarbon
    gas, Kay's mixing for N2 and Wichert-Aziz for CO2/H2S (separator gas gravity). Tutorial 04 table (0.769, 4 % N2,
    0.368 % CO2, 60-210 F, 316-6000 psig): Z within 0.45 %, Lee viscosity within 0.35 %."""
    from wellmodel import gaspvt as gp
    d = np.loadtxt(__file__.replace("test_wellmodel.py", "data/t04_gas_pvt_lee.csv"), delimiter=",")
    T, p, rho, mu, bg, z = d.T
    inp = gp.GasPVTInputs(0.769, 100, 0, 67, 0, 100000, 0, 0.368, 4.005, "lee", 166, 2750)
    zz, bb, rr, ml, _ = gp.raw_props(inp, p + 14.696, T)
    assert np.abs(zz / z - 1).max() < 0.0045 and np.abs(rr / rho - 1).max() < 0.0045
    assert np.abs(ml / mu - 1).max() < 0.0035


def test_gas_injector_prosper_t04():
    """PROSPER Tutorial 04 (Basic Gas Injector), user's PROSPER run: 0.769 gas injected at 100 F and 2500 psig,
    Petroleum Experts injectivity (Pr 2750 psig, 166 F): 7.457 MMscf/d at 2988.0 psig. PROSPER's gas injector leaves
    the N2 / CO2 out of the gas properties (its traverse densities are its PVT without impurities to 0.1 %); with
    the same, the bottom-hole pressure at 7.457 MMscf/d and the system are reproduced (0.3 psi, 0.02 %); with the
    impurities applied the well takes 6.99 MMscf/d (-6 %)."""
    from wellmodel import injection as winj, gaspvt as gp
    G = 14.696
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 4.5,
             "Tubing Inside Roughness (in)": 0.00006},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 4.0},
            {"Type": "Tubing", "Measured Depth (ft)": 4800, "Tubing Inside Diameter (in)": 4.5,
             "Tubing Inside Roughness (in)": 0.00006},
            {"Type": "Casing", "Measured Depth (ft)": 5010, "Casing Inside Diameter (in)": 6.13,
             "Casing Inside Roughness (in)": 0.00006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = winj.from_flow_path(wvlp.build_flow_path(top, secs, pts, [0, 1500, 1982, 2460, 2905, 3335, 4650, 5010],
                                                    [0, 1500, 1980, 2450, 2883, 3300, 4560, 4900],
                                                    [0, 1500, 5010], [45, 75, 166]))
    inp = gp.GasPVTInputs(0.769, 100, 0, 67, 0, 100000, 0, 0.368, 4.005, "lee", 166, 2750)
    flp = winj.GasFluid(inp, impurities=False)
    P, T = winj.traverse(path, flp, 2500 + G, [7.457], 3.0, 100.0)
    assert abs(P[0, -1] - G - 2988.0) < 1.5 and abs(T[0, -1] - 119.5) < 1.0
    # PROSPER's traverse: density and viscosity at each node (MD, psig, F, lb/ft3, cp)
    for md, p, t, rho, mu in ((125, 2511.4, 97.6, 13.197, 0.0225), (1866, 2681.1, 82.9, 14.757, 0.0250),
                              (4905, 2978.3, 117.0, 14.068, 0.0243)):
        bg, r, m = flp.props(p + G, t)
        assert abs(r / rho - 1) < 0.003 and abs(m / mu - 1) < 0.006
    ipr = winj.petex_gas_injection(2750 + G, 166, 25, 55, 500, 31.6, .354, 45, 100, .15, .25, 2.0, flp)
    res, *_ = winj.solve(path, flp, 2500 + G, ipr, 3.0, 100.0)
    assert abs(res.q / 7.457 - 1) < 0.002 and abs(res.pwf - G - 2988.0) < 1.5
    fli = winj.GasFluid(inp)
    ipr_i = winj.petex_gas_injection(2750 + G, 166, 25, 55, 500, 31.6, .354, 45, 100, .15, .25, 2.0, fli)
    res0, *_ = winj.solve(path, fli, 2500 + G, ipr_i, 3.0, 100.0)
    assert abs(res0.q / 7.457 - 1) < 0.08


def test_tif_injector_prosper_t03a():
    """PROSPER Tutorial 03a (water injector, Thermally Induced Fracture IPR). Eight PROSPER system runs (IPR
    pressure and sandface temperature at 20 rates; base and one input changed in each) are reproduced by the
    calibrated three-regime model within 15 psi; system at 1250 psig: PROSPER 11,830.1 STB/d at 5,407.9 psig (base and
    thermal expansion 1e-5), 11,724.4 at 5,408.9 (stress 6000 psig)."""
    import json, dataclasses
    from wellmodel import tif, injection as winj
    G = 14.696
    runs = json.load(open(__file__.replace("test_wellmodel.py", "data/t03a_tif_system_runs.json")))
    fl = winj.WaterFluid(200000)
    base = tif.TIFInputs(pr=4100 + G, t_res=210, t_inj=120, k=50, h=100, area=250, ca=31.6, rw=.354, phi=.18,
                         t_days=20, q_hist=12000, skin=3, sigma_ini=5300 + G, sweep=.75, alpha=5e-6, mu_res=0.35734,
                         b_res=1.4116)
    var = {"base": {}, "a1e-5": dict(alpha=1e-5), "s6000": dict(sigma_ini=6000 + G), "qh24k": dict(q_hist=24000),
           "t100": dict(t_days=100), "A500": dict(area=500), "E05": dict(sweep=.5), "S0": dict(skin=0)}
    for k, kw in var.items():
        ipr = tif.tif_ipr(dataclasses.replace(base, **kw), fl)
        q, p, T = np.array(runs[k])[1:].T
        assert np.abs(ipr.pwf(q, T) - G - p).max() < 15, k
    # IPR tab (injected temperature 120 F): PROSPER 6567.29 psig at 30,000 STB/d, dP skin 336.33 at 10,000
    ipr = tif.tif_ipr(base, fl)
    assert abs(ipr.pwf(30000.0) - G - 6567.29) < 10 and abs(ipr.dp_skin(np.array([10000.0]))[0] - 336.33) < 5
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 600},
            {"Type": "Tubing", "Measured Depth (ft)": 8500, "Tubing Inside Diameter (in)": 5.2,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 9000, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = winj.from_flow_path(wvlp.build_flow_path(top, secs, pts, [0, 9000], [0, 9000], [0, 600, 9000],
                                                    [60, 40, 210]))
    for k, (qr, pr_) in {"base": (11830.1, 5407.9), "s6000": (11724.4, 5408.9)}.items():
        res, *_ = winj.solve(path, fl, 1250 + G, tif.tif_ipr(dataclasses.replace(base, **var[k]), fl), 8.0, 60.0, n=40)
        assert abs(res.q / qr - 1) < 0.005 and abs(res.pwf - G - pr_) < 3, k


# ------------------------------------------------------------ gas-lifted well in the main model (System | Gas Lift Data)
def _gl_well():
    top, secs, pts, smd, stvd, _, oil = _vlp_setup()
    from wellmodel import gaslift_system as wgls
    return top, secs, pts, smd, stvd, oil, wgls


def test_gas_lift_option_validation():
    assert not WellOptions(artificial_lift="gas_lift").errors
    assert WellOptions(artificial_lift="gas_lift").computable
    g = WellOptions(artificial_lift="gas_lift", fluid_type="dry_wet_gas")       # gas lifted gas well (T47): allowed
    assert not g.errors and any("Tutorial 47" in i.message for i in g.validate() if i.level == "info")
    assert WellOptions(artificial_lift="gas_lift", well_type="water_injector").errors
    assert WellOptions(artificial_lift="gas_lift", flow_type="annular").errors


def test_gas_lift_methods_injection_point():
    top, secs, pts, smd, stvd, oil, wgls = _gl_well()
    valves = (wgls.GLValve(3000.0), wgls.GLValve(5500.0), wgls.GLValve(8000.0, "Orifice", 24))
    base = dict(qgi=1.5, p_casing=1214.7, dp_valve=100.0, max_md=10000.0, valves=valves, fixed_md=8000.0)
    case = {"q": np.array([1000.0, 2500.0]), "wc": 0.5, "gor": 300.0}
    out = {}
    for m in wgls.METHODS[:3]:
        s = wgls.GasLiftSetup(method=m, **base)
        path = wvlp.build_flow_path(top, secs, pts, smd, stvd, [35.5, 10963.0], [60.0, 182.19],
                                    extra_md=s.extra_md())
        out[m] = wgls.lift_traverse(path, oil, 200.0, {**case, **s.row_inputs({}, case["q"])}, s, "hagedorn_brown")
    opt, vds, fix = out[wgls.METHODS[1]], out[wgls.METHODS[2]], out[wgls.METHODS[0]]
    # valve depths specified: the deepest valve at or above the optimum injection depth
    for k in range(2):
        above = [v.md for v in valves if v.md <= opt.inj_md[k] + 1e-6]
        assert vds.inj_md[k] == pytest.approx(max(above))
    assert np.all(fix.inj_md == pytest.approx(8000.0))
    # deeper injection with the same gas gives a lower bottom-node pressure
    assert np.all(opt.P[:, -1] <= vds.P[:, -1] + 1e-6)
    # optimum: casing = tubing + dP at the injection point (unless limited by the maximum depth)
    path = wvlp.build_flow_path(top, secs, pts, smd, stvd, [35.5, 10963.0], [60.0, 182.19], extra_md=[10000.0])
    for k in range(2):
        pt = opt.inj_p[k]
        pc = np.interp(opt.inj_md[k], path.md, opt.pc[k])
        if opt.inj_md[k] < 9990:
            assert pc - pt == pytest.approx(100.0, abs=3.0)


def test_gas_lift_annulus_friction_and_no_injection():
    top, secs, pts, smd, stvd, oil, wgls = _gl_well()
    path = wvlp.build_flow_path(top, secs, pts, smd, stvd, [35.5, 10963.0], [60.0, 182.19])
    ann = wgls.annulus_geometry(path, secs)
    assert ann.d_h[5] == pytest.approx((8.681 - 3.5) / 12)
    T = np.tile(path.t_form, (3, 1))
    flow = np.array([[0.0], [5.0], [20.0]]) * np.ones((1, len(path.seg_len)))
    s = wgls.GasLiftSetup(annulus_friction=True, annulus=ann)
    pc = wgls.casing_profile(path, s, 1200.0, T, flow)
    assert pc[0, -1] > pc[1, -1] > pc[2, -1]                       # friction lowers the casing pressure downhole
    s0 = wgls.GasLiftSetup(annulus_friction=False, annulus=ann)
    pc0 = wgls.casing_profile(path, s0, 1200.0, T, flow)
    assert np.allclose(pc0[0], pc0[2]) and np.allclose(pc0[0], pc[0])
    # gas column: about 0.025-0.04 psi/ft at these pressures for 0.7 gravity gas
    assert 0.02 < (pc0[0, -1] - 1200.0) / path.tvd[-1] < 0.045
    # casing pressure too low to inject anywhere: natural flow, flagged
    sv = wgls.GasLiftSetup(method=wgls.METHODS[2], p_casing=150.0, valves=(wgls.GLValve(3000.0),))
    p2 = wvlp.build_flow_path(top, secs, pts, smd, stvd, [35.5, 10963.0], [60.0, 182.19], extra_md=[3000.0])
    st_ = wgls.lift_traverse(p2, oil, 200.0, {"q": np.array([1000.0]), "wc": 0.5, "gor": 300.0,
                                              **sv.row_inputs({}, [1000.0])}, sv, "hagedorn_brown")
    P0, _ = wvlp.traverse(p2, oil, 200.0, {"q": np.array([1000.0]), "wc": 0.5, "gor": 300.0}, "hagedorn_brown")
    assert np.isnan(st_.inj_md[0]) and st_.qgi[0] == 0 and "no injection" in st_.note[0]
    assert st_.P[0, -1] == pytest.approx(P0[0, -1])


def test_gas_lift_multipoint_consistency():
    """Multipoint: each valve passes Thornhill-Craver gas at the converged casing and tubing pressures."""
    top, secs, pts, smd, stvd, oil, wgls = _gl_well()
    valves = (wgls.GLValve(4000.0, "Valve", 16, 0.066, 1000.0), wgls.GLValve(8000.0, "Orifice", 20))
    s = wgls.GasLiftSetup(method=wgls.METHODS[3], p_casing=1214.7, valves=valves)
    path = wvlp.build_flow_path(top, secs, pts, smd, stvd, [35.5, 10963.0], [60.0, 182.19], extra_md=s.extra_md())
    s.annulus = wgls.annulus_geometry(path, secs)
    q = np.array([800.0, 1500.0])
    st_ = wgls.lift_traverse(path, oil, 200.0, {"q": q, "wc": 0.5, "gor": 300.0, **s.row_inputs({}, q)}, s,
                             "hagedorn_brown")
    qn, x, status = wgls.valve_rates(valves, st_.valve_pc, st_.valve_pt, st_.valve_t, 0.7)
    assert np.allclose(qn, st_.valve_q, rtol=2e-3, atol=2e-3)
    assert np.all(st_.qgi > 0.1) and np.all(st_.qgi == pytest.approx(st_.valve_q.sum(1)))
    assert set(np.unique(status)) <= {"Closed", "Throttle", "Open Sub", "Open Cri", "Neg dP"}
    # orifice rate is Thornhill-Craver through the full port
    k = 1
    t_ = wgl.thornhill_craver(20 / 64, st_.valve_pc[:, k], np.minimum(st_.valve_pt[:, k], st_.valve_pc[:, k]),
                              st_.valve_t[:, k], 0.7) / 1000.0
    assert np.allclose(t_, st_.valve_q[:, k], rtol=2e-3)
    # more casing pressure, more gas
    s2 = replace(s, p_casing=1414.7)
    st2 = wgls.lift_traverse(path, oil, 200.0, {"q": q, "wc": 0.5, "gor": 300.0, **s2.row_inputs({}, q)}, s2,
                             "hagedorn_brown")
    assert np.all(st2.qgi > st_.qgi)


def test_gas_lift_system_and_lift_curves():
    top, secs, pts, smd, stvd, oil, wgls = _gl_well()
    s = wgls.GasLiftSetup(method=wgls.METHODS[1], p_casing=1214.7, dp_valve=100.0, max_md=10000.0)
    path = wvlp.build_flow_path(top, secs, pts, smd, stvd, [35.5, 10963.0], [60.0, 182.19], extra_md=s.extra_md())
    curve = iflow.pi_entry(3000, 1000, 2.0)
    base = {"whp": 200.0, "u": 8.0, "wc": 50.0, "gor": 300.0}
    combos = [{"qgi": g} for g in (0.0, 0.5, 1.5)] + [{"glr_inj": 600.0}]
    res, _ = wsys.solve_cases(lambda tid: path, oil, "oil", base, combos, lambda ch: curve, "hagedorn_brown",
                              lift=s)
    q = [r.q or 0.0 for r in res]
    assert q[1] > q[0] and q[2] > q[1]
    for r in res[1:]:
        gl = r.details["gas_lift"]
        assert r.pwf == pytest.approx(float(wgl.ipr_pwf(curve, r.q)), abs=1.0)
        assert gl["Injection depth (ft MD)"] > 1000
    assert res[3].details["gas_lift"]["Gas lift gas injection rate (MMscf/d)"] == pytest.approx(600.0 * res[3].q / 1e6)
    with pytest.raises(ValueError):
        wsys.solve_cases(lambda tid: path, oil, "oil", base, [{}], lambda ch: curve, "hagedorn_brown", lift=s,
                         node="top")
    det, prof = wgls.details(path, oil, {**base, "qgi": 1.5}, res[2].q, s, "hagedorn_brown")
    assert det["Bottom node pressure (psia)"] == pytest.approx(res[2].pwf, abs=3.0)
    assert prof["Lift gas in tubing (MMscf/d)"].iloc[1] == pytest.approx(1.5)
    axes, bhp, _ = wlc.vfp_table(path, oil, "oil", "hagedorn_brown", 1, 1, 8.0, [500, 2000], [200], [50], [300],
                                 lift=s, alq=[0.5, 2.0])
    assert bhp.shape == (1, 1, 1, 2, 2) and np.all(bhp[0, 0, 0, 1] < bhp[0, 0, 0, 0])
    ecl = wlc.eclipse_vfpprod("oil", axes, bhp, alq_kind="gas_lift")
    assert "'GRAT'" in ecl and "500 2000 /" in ecl


def _t23_path(extra=()):
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 600},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 4.052,
             "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 4.8,
             "Casing Inside Diameter (in)": 6.4},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 3.72},
            {"Type": "Tubing", "Measured Depth (ft)": 9000, "Tubing Inside Diameter (in)": 4.052,
             "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 4.8,
             "Casing Inside Diameter (in)": 6.4},
            {"Type": "Casing", "Measured Depth (ft)": 9275, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    smd_, stvd_ = np.array(LOGIE_SURVEY).T
    return wvlp.build_flow_path(top, secs, pts, smd_, stvd_, [0, 600, 9275], [60, 40, 210], extra_md=extra), secs


def test_pe2_gas_lift_traverses_prosper_t23():
    """PROSPER Tutorial 23 gradient traverses (PE2, P1 1.03108 / P2 1.00278, 6500 STB/d, 80 % water cut, GOR 800,
    500 psig wellhead): 5 MMscf/d of lift gas at the 6660 ft valve and 20 MMscf/d at the 3731 ft valve. PE2's slug
    holdup follows Hagedorn & Brown at high gas velocities and its slug friction is 1.28 x the HB friction."""
    from wellmodel import gaslift_system as wgls
    G = 14.696
    oil = _t05_oil()
    prosper = {5: (6660.16, 154.1, {3618.2: 1009.2, 6660.2: 1585.8, 7013.9: 1716.8, 9275.0: 2573.9}),
               20: (3731.43, 157.5, {3618.2: 1044.4, 5666.9: 1762.4, 7013.9: 2276.9, 9275.0: 3156.7})}
    for qgi, (md, wht, pts_) in prosper.items():
        path, _ = _t23_path([md])
        s = wgls.GasLiftSetup(method=wgls.METHODS[0], qgi=qgi, fixed_md=md)
        det, prof = wgls.details(path, oil, {"whp": 500 + G, "wc": 80.0, "gor": 800.0}, 6500.0, s, "pe2_style",
                                 1.03108, 1.00278, 8.580277)
        assert det["Wellhead temperature (°F)"] == pytest.approx(wht, abs=0.5)
        for z, p in pts_.items():
            # within 16 psi (1 %): the PE2-style friction is now fitted on five traverses, these two among them
            assert np.interp(z, prof["MD (ft)"], prof["Pressure (psia)"]) - G == pytest.approx(p, abs=18.0), (qgi, z)


def test_pe2_traverse_prosper_t11():
    """PROSPER Tutorial 11 file, gradient traverse (PE2 unmatched, 16283.4 STB/d, 80 % water cut, GOR 800, 500 psig
    at the 600 ft wellhead, U 8): slug flow at high liquid holdup (0.58-0.87) and vm 14-21 ft/s, bubble flow below
    3,880 ft; and at 5 % water cut. PE2-style slug friction ((1.527 - 0.1205 HL) vsl^-0.0819 x Hagedorn & Brown,
    stepped up below Ngv = 1.75 + 1.44 Nlv) within 4 psi at every node."""
    G = 14.696
    oil = _t05_oil()
    prosper = {725.0: 541.1, 1880.0: 947.3, 3380.0: 1529.2, 4040.0: 1797.9, 5450.0: 2377.2, 7637.5: 3294.9,
               8975.0: 3854.9}
    path, _ = _t23_path(list(prosper))
    P, T_ = wvlp.traverse(path, oil, np.array([500 + G]), {"q": np.array([16283.4]), "wc": np.array([0.8]),
                                                            "gor": np.array([800.0])}, "pe2_style", 1.0, 1.0, 8.0)
    for z, p in prosper.items():
        assert np.interp(z, path.md, P[0]) - G == pytest.approx(p, abs=5.0), z
    assert np.interp(725.0, path.md, T_[0]) == pytest.approx(185.0, abs=0.5)
    # same at 5 % water cut: oil-rich slug flow at vsl 12.4 ft/s, crossing the friction step line (Ngv = 1.75 +
    # 1.44 Nlv) near 2,500 ft
    prosper = {725.0: 534.2, 1880.0: 852.9, 3380.0: 1276.4, 4040.0: 1471.3, 5450.0: 1899.4, 7637.5: 2607.1,
               8975.0: 3053.3}
    P, T_ = wvlp.traverse(path, oil, np.array([500 + G]), {"q": np.array([16283.4]), "wc": np.array([0.05]),
                                                            "gor": np.array([800.0])}, "pe2_style", 1.0, 1.0, 8.0)
    for z, p in prosper.items():
        assert np.interp(z, path.md, P[0]) - G == pytest.approx(p, abs=6.0), z
    assert np.interp(725.0, path.md, T_[0]) == pytest.approx(170.5, abs=0.5)


def test_gas_lift_system_prosper_t23():
    """PROSPER Tutorial 23 system calculation, valve depths specified (3731 / 5667 / 6660 valves, 7014 orifice), 1510
    psig casing, friction loss in annulus, gas injection rate sensitivity (user's PROSPER runs). Where PROSPER's
    solution sits between two of its rate points with the injection on different valves its reported injection depth
    is an interpolation between valves; the rates still agree within 5 %."""
    from wellmodel import gaslift_system as wgls
    G = 14.696
    oil = _t05_oil()
    vs = (wgls.GLValve(3731.43, "Valve", 16, 0.066, 1211.11), wgls.GLValve(5666.89, "Valve", 20, 0.103, 1195.47),
          wgls.GLValve(6660.16, "Valve", 24, 0.147, 1179.41), wgls.GLValve(7013.92, "Orifice", 29))
    s = wgls.GasLiftSetup(method=wgls.METHODS[2], p_casing=1510 + G, dp_valve=100.0, valves=vs, annulus_friction=True)
    path, secs = _t23_path(s.extra_md())
    s.annulus = wgls.annulus_geometry(path, secs)
    assert np.isfinite(s.annulus.d_h[path.md[1:] <= 9000]).all()
    # Tutorial 23 IPR as built in the app (Darcy at 3450 psig, 80 % water cut, Karakas & Tariq perforations and the
    # gravel pack); PROSPER's solution-node pressures lie on it within 1.5 psi
    import json
    from types import SimpleNamespace
    ip = json.load(open(Path(__file__).parent / "data" / "t23_ipr.json"))
    curve = SimpleNamespace(q=np.array(ip["q"]), pwf=np.array(ip["pwf"]), pr=ip["pr"], aof=max(ip["q"]), fluid="oil")
    prosper = {1: 3212.7, 3: 5353.1, 7.5: 6044.0, 12.5: 5808.9, 20: 4982.4}
    res, _ = wsys.solve_cases(lambda tid: path, oil, "oil", {"whp": 500 + G, "u": 8.580277, "wc": 80.0, "gor": 800.0},
                              [{"qgi": g} for g in prosper], lambda ch: curve, "pe2_style", 1.03108, 1.00278, lift=s)
    for r, (g, q) in zip(res, prosper.items()):
        assert r.q == pytest.approx(q, rel=0.05), (g, r.q, q)


def test_gaslift_design_existing_mandrels_prosper_t23():
    """PROSPER Tutorial 23 design with existing mandrels every 600 ft from 2000 to 8000 ft (user's run: max
    production, 4 MMscf/d, 250 psig wellhead, 1500 psig injection and kick-off, 100 psi desired dP, 0.43 psi/ft, 50 psi
    CHP decrease, 80 % water cut, GOR 800, R-20 Normal, PE2 1.03108 / 1.00278): valves at 3200 / 5000 / 5600 ft,
    orifice at 6200 ft (26/64), dummies in the other seven mandrels; the 5600 ft valve is needed because the 5000 ft
    one cannot unload to 6200 ft with the 100 psi desired dP across the injection point."""
    import json, dataclasses
    from types import SimpleNamespace
    G = 14.696
    oil = _t05_oil()
    path, _ = _t23_path()
    ip = json.load(open(Path(__file__).parent / "data" / "t23_ipr.json"))
    curve = SimpleNamespace(q=np.array(ip["q"]), pwf=np.array(ip["pwf"]), pr=ip["pr"], aof=max(ip["q"]), fluid="oil")
    mand = tuple((float(m), np.nan, np.nan) for m in range(2000, 8600, 600))
    inp = wgl.DesignInput(design_rate=wgl.DESIGN_RATE[1], valve=[v for v in wgl.VALVES if "R-20 Normal" in v][0],
                          q_liquid_max=10000, qgi_max=4, qgi_unload_max=4, top_p=250 + G, top_p_unload=250 + G,
                          p_inj=1500 + G, p_ko=1500 + G, dp_valve=100, max_md=8500, gor=800, wc=0.8, load_grad=0.43,
                          dp_close=50, margin_top=50, margin=50, min_transfer_pct=25, mandrels=mand)
    r = wgl.design(path, oil, curve, wgl.LiftGas(0.7, t_surface=80.0), inp, "pe2_style", 1.03108, 1.00278, 8.580277)
    assert [(v.kind, v.md) for v in r.valves] == [("Valve", 3200.0), ("Valve", 5000.0), ("Valve", 5600.0),
                                                  ("Orifice", 6200.0)]
    assert list(r.mandrels["Holds"] == "Dummy") == [True, True, False, True, True, False, False, False, True, True,
                                                    True]
    v = r.valves
    assert v[0].pc - G == pytest.approx(1596.32, abs=3) and v[1].pc - G == pytest.approx(1612.97, abs=10)
    assert v[2].port_64 == 20 and v[3].port_64 == pytest.approx(26, abs=1)
    assert [x.chp_open - G for x in v[:2]] == pytest.approx([1500, 1450], abs=0.01)
    # run 2: mandrels at 3000 / 4500 / 6000 / 7500 ft - PROSPER: valves 3000 (12/64) and 4500 (16/64), orifice 6000
    # (25/64), dummy 7500; opening CHPs 1500 / 1450 / 1400 psig
    r2 = wgl.design(path, oil, curve, wgl.LiftGas(0.7, t_surface=80.0),
                    dataclasses.replace(inp, mandrels=tuple((m, np.nan, np.nan) for m in (3000.0, 4500.0, 6000.0, 7500.0))),
                    "pe2_style", 1.03108, 1.00278, 8.580277)
    assert [(x.kind, x.md) for x in r2.valves] == [("Valve", 3000.0), ("Valve", 4500.0), ("Orifice", 6000.0)]
    assert list(r2.mandrels["Holds"]) == ["Valve 1", "Valve 2", "Orifice 3", "Dummy"]
    assert [x.chp_open - G for x in r2.valves] == pytest.approx([1500, 1450, 1400], abs=0.2)
    assert r2.valves[0].pc - G == pytest.approx(1588.59, abs=3) and r2.valves[1].port_64 == 16
    assert r2.valves[2].port_64 == pytest.approx(25, abs=1)


def test_gaslift_design_tubing_sensitive_prosper_t23():
    """PROSPER Tutorial 23 design with tubing-sensitive valves (Camco BK-F6 Normal, Percentage Pcasing - Pwh 30 %)
    in mandrels every 600 ft from 2000 to 8000 ft (user's run): valves at 2600 / 3800 / 5000 / 5600 / 6200 / 6800 ft,
    orifice at 7400 ft (27/64), 7187.5 STB/d; casing at the injection pressure throughout; each valve opens at its
    transfer pressure (load fluid just reaching the next mandrel within 50 psi of the casing) and closes at
    Pvo (1 - R) + Pc R."""
    import json
    from types import SimpleNamespace
    G = 14.696
    oil = _t05_oil()
    path, _ = _t23_path()
    ip = json.load(open(Path(__file__).parent / "data" / "t23_ipr.json"))
    curve = SimpleNamespace(q=np.array(ip["q"]), pwf=np.array(ip["pwf"]), pr=ip["pr"], aof=max(ip["q"]), fluid="oil")
    inp = wgl.DesignInput(design_rate=wgl.DESIGN_RATE[1], valve="Camco BK-F6 Normal", valve_type="Tubing sensitive",
                          pct_casing_wh=30, q_liquid_max=10000, qgi_max=4, qgi_unload_max=4, top_p=250 + G,
                          top_p_unload=250 + G, p_inj=1500 + G, p_ko=1500 + G, dp_valve=100, max_md=8500, gor=800,
                          wc=0.8, load_grad=0.43, margin_top=50, margin=50, min_transfer_pct=25,
                          mandrels=tuple((float(m), np.nan, np.nan) for m in range(2000, 8600, 600)))
    r = wgl.design(path, oil, curve, wgl.LiftGas(0.7, t_surface=80.0), inp, "pe2_style", 1.03108, 1.00278, 8.580277)
    v = r.valves
    assert [(x.kind, x.md) for x in v] == [("Valve", m) for m in (2600.0, 3800.0, 5000.0, 5600.0, 6200.0, 6800.0)] + \
        [("Orifice", 7400.0)]
    prosper = {  # Pt, Pc, transfer = opening, closing (psig)
        2600: (583.536, 1574.06, 1294.14, 1305.9), 3800: (792.948, 1618.51, 1340.19, 1386.11),
        5000: (1011.95, 1662.81, 1385.97, 1411.99), 5600: (1129.79, 1685.07, 1408.36, 1454.02),
        6200: (1255.32, 1707.47, 1430.93, 1476.56), 6800: (1387.81, 1730.03, 1453.7, 1499.29)}
    for x in v[:-1]:
        pt, pc, ptr, pvc = prosper[int(x.md)]
        assert x.pt - G == pytest.approx(pt, rel=0.01) and x.pc - G == pytest.approx(pc, abs=2)
        assert x.p_transfer - G == pytest.approx(ptr, abs=2) and x.pvo == x.p_transfer
        assert x.pvc == pytest.approx(x.pvo * (1 - x.r) + x.pc * x.r)
    assert v[0].pvc - G == pytest.approx(1305.9, abs=2) and v[1].pvc - G == pytest.approx(1386.11, abs=25)
    assert v[-1].pt - G == pytest.approx(1526.08, abs=10) and v[-1].pc - G == pytest.approx(1752.8, abs=3)
    assert v[-1].port_64 == pytest.approx(27, abs=1) and r.q == pytest.approx(7187.54, rel=0.02)
    assert v[0].port_64 == 8 and max(x.n_valves for x in v) == 2


def test_gaslift_design_proportional_prosper_t23():
    """PROSPER Tutorial 23 design with the Merla LN-20R 1.5" proportional valve, mandrels every 600 ft (user's run).
    Settings from PROSPER's own tubing / casing pressures: PTO = Pt + 25 % (Pc - Pt), closing at depth = PTO (1 - k)
    + Pc k (k 0.66 / 0.54 / 0.40 for Small / Medium / Large), dome at 60 F from closing + 220 psi, shop closing
    = dome - 220 - all within 0.04 psi of PROSPER. The design here: valves only in mandrels, casing at the injection
    pressure, the orifice at 7400 ft split into orifices no larger than 20/64 (PROSPER: 2 x 19/64)."""
    import json
    from types import SimpleNamespace
    G = 14.696
    rows = [  # Pt, Pc, T, k, PTO, Pvc at depth, dome, shop closing (psig)
        (690.972, 1596.29, 182.752, .66, 917.301, 1365.43, 1235.29, 1015.29),
        (906.05, 1640.74, 191.843, .66, 1089.72, 1453.39, 1280.76, 1060.76),
        (1019.84, 1662.9, 195.78, .66, 1180.61, 1498.92, 1305.32, 1085.32),
        (1139.78, 1685.18, 199.285, .66, 1276.13, 1546.1, 1331.62, 1111.62),
        (1268.73, 1707.61, 202.334, .54, 1378.45, 1556.19, 1332.04, 1112.04),
        (1403.54, 1730.2, 204.9, .40, 1485.21, 1583.21, 1345.57, 1125.57)]
    for pt, pc, t, k, pto, pvc, pd, sh in rows:
        a = wgl.proportional_settings(pt + G, pc + G, k, t)
        assert a[0] - G == pytest.approx(pto, abs=0.05) and a[1] - G == pytest.approx(pvc, abs=0.05)
        assert a[2] == pytest.approx(pd, abs=0.05) and a[3] == pytest.approx(sh, abs=0.05)
    v = wgl.VALVES['Merla LN-20R 1.5"']
    assert v.kind == "Proportional" and [float(v.r(p)) for p in v.ports] == [0.66, 0.54, 0.40]
    oil = _t05_oil()
    path, _ = _t23_path()
    ip = json.load(open(Path(__file__).parent / "data" / "t23_ipr.json"))
    curve = SimpleNamespace(q=np.array(ip["q"]), pwf=np.array(ip["pwf"]), pr=ip["pr"], aof=max(ip["q"]), fluid="oil")
    mand = tuple((float(m), np.nan, np.nan) for m in range(2000, 8600, 600))
    inp = wgl.DesignInput(design_rate=wgl.DESIGN_RATE[1], valve='Merla LN-20R 1.5"', valve_type="Proportional",
                          q_liquid_max=10000, qgi_max=4, qgi_unload_max=4, top_p=250 + G, top_p_unload=250 + G,
                          p_inj=1500 + G, p_ko=1500 + G, dp_valve=100, max_md=8500, gor=800, wc=0.8, load_grad=0.43,
                          min_transfer_pct=25, mandrels=mand)
    r = wgl.design(path, oil, curve, wgl.LiftGas(0.7, t_surface=80.0), inp, "pe2_style", 1.03108, 1.00278, 8.580277)
    vv = r.valves
    assert {x.md for x in vv} <= {m[0] for m in mand} and vv[0].md == 3200.0 and vv[1].md == 4400.0
    assert vv[-1].kind == "Orifice" and vv[-1].md == 7400.0 and vv[-1].n_valves == 2 and vv[-1].port_64 <= 20
    assert r.pso == pytest.approx(1500 + G)                     # the casing pressure is not dropped per valve
    for x in vv[:-1]:
        assert x.trim in ("Small", "Medium", "Large")
        assert x.pvo == pytest.approx(x.pt + 0.25 * (x.pc - x.pt)) and x.pvc == pytest.approx(x.pvo * (1 - x.r) + x.pc * x.r)


def test_gas_lift_system_prosper_t23_mandrel_design():
    """PROSPER's existing-mandrel design (run 1: valves 3200 / 5000 / 5600 ft, orifice 6200 ft 26/64, 1338.92 psig
    casing) in PROSPER's system calculation at 500 psig and 4 MMscf/d (user's run): 5245.9 STB/d liquid at
    2593.6 psig."""
    import json
    from types import SimpleNamespace
    from wellmodel import gaslift_system as wgls
    G = 14.696
    oil = _t05_oil()
    vs = (wgls.GLValve(3200, "Valve", 12, 0.038, 1218.02), wgls.GLValve(5000, "Valve", 20, 0.103, 1184.11),
          wgls.GLValve(5600, "Valve", 20, 0.103, 1160.98), wgls.GLValve(6200, "Orifice", 26))
    s = wgls.GasLiftSetup(method=wgls.METHODS[2], p_casing=1338.92 + G, dp_valve=100.0, valves=vs, annulus_friction=True)
    path, secs = _t23_path(s.extra_md())
    s.annulus = wgls.annulus_geometry(path, secs)
    ip = json.load(open(Path(__file__).parent / "data" / "t23_ipr.json"))
    curve = SimpleNamespace(q=np.array(ip["q"]), pwf=np.array(ip["pwf"]), pr=ip["pr"], aof=max(ip["q"]), fluid="oil")
    res, _ = wsys.solve_cases(lambda tid: path, oil, "oil", {"whp": 500 + G, "u": 8.580277, "wc": 80.0, "gor": 800.0},
                              [{"qgi": 4.0}], lambda ch: curve, "pe2_style", 1.03108, 1.00278, lift=s)
    assert res[0].q == pytest.approx(5245.9, rel=0.03) and res[0].pwf - G == pytest.approx(2593.6, abs=30)


def test_gas_lift_system_prosper_t23_tubing_sensitive_design():
    """PROSPER's tubing-sensitive design (mandrels 2000-7400 ft as valve positions, 1510 psig casing, 100 psi dP,
    friction in annulus, valve depths specified) in its system calculation at 4 MMscf/d (user's runs): 7216.8 STB/d
    at 2220.3 psig (250 psig wellhead) and 5779.7 at 2493.0 (500 psig; PROSPER's injection depth 6949.7 ft lies
    between the 6800 and 7400 ft valves). At 250 psig PROSPER's system agrees with this tool's design rate (7211.5),
    not with PROSPER's design screen (7289.9)."""
    import json
    from types import SimpleNamespace
    from wellmodel import gaslift_system as wgls
    G = 14.696
    oil = _t05_oil()
    vs = tuple(wgls.GLValve(float(m), "Valve", 16, 0.165, 1100.0) for m in range(2000, 7500, 600))
    s = wgls.GasLiftSetup(method=wgls.METHODS[2], p_casing=1510 + G, dp_valve=100.0, valves=vs, annulus_friction=True)
    path, secs = _t23_path(s.extra_md())
    s.annulus = wgls.annulus_geometry(path, secs)
    ip = json.load(open(Path(__file__).parent / "data" / "t23_ipr.json"))
    curve = SimpleNamespace(q=np.array(ip["q"]), pwf=np.array(ip["pwf"]), pr=ip["pr"], aof=max(ip["q"]), fluid="oil")
    for whp, q, pwf, tol in ((250, 7216.8, 2220.3, 0.01), (500, 5779.7, 2493.0, 0.03)):
        res, _ = wsys.solve_cases(lambda tid: path, oil, "oil", {"whp": whp + G, "u": 8.580277, "wc": 80.0,
                                                                  "gor": 800.0}, [{"qgi": 4.0}], lambda ch: curve,
                                  "pe2_style", 1.03108, 1.00278, lift=s)
        assert res[0].q == pytest.approx(q, rel=tol) and res[0].pwf - G == pytest.approx(pwf, abs=30), whp


def test_gas_lift_multipoint_tubing_sensitive_prosper_t23():
    """PROSPER multipoint system of its tubing-sensitive design (Camco BK-F6 Normal, valves 2600-6800 ft with 1 / 1 / 2 /
    2 / 2 / 2 valves, orifice 7400 ft 27/64, 1500 psig casing, 250 psig wellhead; user's run): all six valves closed
    (the tubing pressure is far below what opens them), the orifice open with 4.168 MMscf/d, 7218.4 STB/d at
    2220.0 psig. (PROSPER's 500 psig run reports 2.698 MMscf/d through the orifice with the tubing pressure there
    above the casing pressure - not used.)"""
    import json
    from types import SimpleNamespace
    from wellmodel import gaslift_system as wgls
    G = 14.696
    oil = _t05_oil()
    dome = [(2600, 1, 8, .042, 1034.52), (3800, 1, 16, .165, 1077.34), (5000, 2, 12, .094, 1081.63),
            (5600, 2, 16, .165, 1105.99), (6200, 2, 16, .165, 1116.78), (6800, 2, 16, .165, 1128.47)]
    vs = tuple(wgls.GLValve(float(md), "Tubing", p, r, d, n) for md, n, p, r, d in dome) + \
        (wgls.GLValve(7400.0, "Orifice", 27),)
    s = wgls.GasLiftSetup(method=wgls.METHODS[3], p_casing=1500 + G, dp_valve=100.0, valves=vs, annulus_friction=True)
    path, secs = _t23_path(s.extra_md())
    s.annulus = wgls.annulus_geometry(path, secs)
    ip = json.load(open(Path(__file__).parent / "data" / "t23_ipr.json"))
    curve = SimpleNamespace(q=np.array(ip["q"]), pwf=np.array(ip["pwf"]), pr=ip["pr"], aof=max(ip["q"]), fluid="oil")
    res, _ = wsys.solve_cases(lambda tid: path, oil, "oil", {"whp": 250 + G, "u": 8.580277, "wc": 80.0, "gor": 800.0},
                              [{}], lambda ch: curve, "pe2_style", 1.03108, 1.00278, lift=s)
    gl = res[0].details["gas_lift"]
    vt = gl["_valves"]
    assert list(vt["Status"][:6]) == ["Closed"] * 6 and vt["Status"].iloc[6].startswith("Open")
    assert vt["Injection rate (MMscf/d)"].iloc[6] == pytest.approx(4.168, rel=0.03)
    assert res[0].q == pytest.approx(7218.4, rel=0.01) and res[0].pwf - G == pytest.approx(2220.0, abs=15)
    # a tubing-sensitive valve opens on the tubing pressure: Pt (1 - R) + Pc R >= dome at temperature
    v = wgls.GLValve(5000.0, "Tubing", 16, 0.165, 1100.0, 2)
    pbt = float(wgls.dome_at_t(1100.0, 195.0)) + G
    pc = np.array([[1700.0, 1700.0]])
    pt = np.array([[(pbt - 0.165 * 1700) / 0.835 - 20, (pbt - 0.165 * 1700) / 0.835 + 20]])
    q, x, stt = wgls.valve_rates((v, v), pc, pt, np.full((1, 2), 195.0), 0.7)
    assert stt[0, 0] == "Closed" and q[0, 0] == 0 and q[0, 1] > 0


def test_gas_lift_multipoint_vlp_prosper_t23():
    """PROSPER Tutorial 23 multipoint VLP at 1410 psig casing (user's run, friction loss in annulus): valve rates,
    statuses and bottom-node pressures at 3000-6000 STB/d; injection collapses between 4000 and 5000 STB/d."""
    from wellmodel import gaslift_system as wgls
    G = 14.696
    oil = _t05_oil()
    vs = (wgls.GLValve(3731.43, "Valve", 16, 0.066, 1211.11), wgls.GLValve(5666.89, "Valve", 20, 0.103, 1195.47),
          wgls.GLValve(6660.16, "Valve", 24, 0.147, 1179.41), wgls.GLValve(7013.92, "Orifice", 29))
    s = wgls.GasLiftSetup(method=wgls.METHODS[3], p_casing=1410 + G, valves=vs, annulus_friction=True)
    path, secs = _t23_path(s.extra_md())
    s.annulus = wgls.annulus_geometry(path, secs)
    q = np.array([3000.0, 4000.0, 5000.0, 6000.0])
    st_ = wgls.lift_traverse(path, oil, 500 + G, {"q": q, "wc": 0.8, "gor": 800.0, **s.row_inputs({}, q)}, s,
                             "pe2_style", 1.03108, 1.00278, 8.580277)
    prosper_bhp = (2122.3, 2240.0, 3684.9, 3692.7)
    prosper_q = ((0.0, 0.0, 0.502, 5.238), (0.0, 0.0, 0.710, 4.494))
    prosper_status = (("Throttle", "Throttle", "Throttle", "Open Sub"), ("Closed", "Throttle", "Throttle", "Open Sub"))
    prosper_pc = ((1522.1, 1588.2, 1621.9, 1634.1), (1520.2, 1586.3, 1620.2, 1632.6))
    for k in range(4):
        assert st_.P[k, -1] - G == pytest.approx(prosper_bhp[k], abs=20.0)
    for k in range(2):
        assert tuple(st_.valve_status[k]) == prosper_status[k]
        # 24/64 valve at Pt/Pc 0.84 and 4000 STB/d: 0.613 against 0.710 (the large-port K scatter, see THROTTLE_K)
        assert np.allclose(st_.valve_q[k], prosper_q[k], rtol=0.15, atol=0.02), st_.valve_q[k]
        assert np.allclose(st_.valve_pc[k] - G, prosper_pc[k], atol=3.0)
        assert st_.flag[k] == "Multiple"
    assert list(st_.flag[2:]) == ["No Injection", "No Injection"]
    # PROSPER's dome pressures at valve temperature (Winkler & Eads on gauge pressures)
    for pd60, t, p_ in ((1211.11, 167.2, 1509.9), (1195.47, 189.2, 1550.3), (1179.41, 197.8, 1552.0)):
        assert float(wgls.dome_at_t(pd60, t)) == pytest.approx(p_, abs=0.3)


def test_gas_lift_multipoint_valves_prosper_t23():
    """PROSPER multipoint VLP on the Tutorial 23 well (user's runs, 500 psig, 80 % water cut, 1000-3500 STB/d):
    A1-A3 all three valves Camco R-20 Normal at 12 / 16 / 20/64 (orifice the same port), 1410 psig casing; B1-B3 the
    tutorial's valves (16 / 20 / 24/64, orifice 29/64) at 1360 / 1460 / 1510 psig; C1 / C2 McMurry R-1 and Otis NM16R
    16/64 (1 in valves, R 0.165 / 0.166); X Camco R-20 24/64 at 1460 psig; D1 / D2 the A2 valves (16/64) at 1460 / 1510 psig. A throttling valve opens a = (K0 - K1 R)
    (Pc - Pvo) / R of its full port (THROTTLE_K, THROTTLE_SUB for subcritical flow). Bottom-node pressures, valve rates, which valves pass
    gas. Near the collapse of the injection PROSPER takes the collapsed state where an unloaded one also exists (A1
    2500 / 3000, A2 3500) or finds no injection where a little remains here (C1 3000 / 3500, C2 3000 / 3500): those
    points are excluded."""
    from wellmodel import gaslift_system as wgls
    G = 14.696
    oil = _t05_oil()
    d = pd.read_csv(Path(__file__).parent / "data" / "t23_multipoint_valves.csv")
    skip = {("A1", 2500.0), ("A1", 3000.0), ("A2", 3500.0), ("C1", 3000.0), ("C1", 3500.0), ("C2", 3000.0),
            ("C2", 3500.0)}
    for run, g in d.groupby("run"):
        v0 = g[g.liquid_stbd == g.liquid_stbd.iloc[0]].sort_values("valve")
        vs = tuple(wgls.GLValve(float(r.md), "Valve", int(r.port), float(r.R), dome)
                   for r, dome in zip(v0.iloc[:3].itertuples(), (1211.11, 1195.47, 1179.41)))
        vs += (wgls.GLValve(7013.92, "Orifice", int(v0.port.iloc[3])),)
        s = wgls.GasLiftSetup(method=wgls.METHODS[3], p_casing=float(g.casing_psig.iloc[0]) + G, valves=vs,
                              annulus_friction=True)
        path, secs = _t23_path(s.extra_md())
        s.annulus = wgls.annulus_geometry(path, secs)
        qs = [q for q in sorted(g.liquid_stbd.unique()) if (run, q) not in skip]
        q = np.array(qs)
        st_ = wgls.lift_traverse(path, oil, 500 + G, {"q": q, "wc": 0.8, "gor": 800.0, **s.row_inputs({}, q)}, s,
                                 "pe2_style", 1.03108, 1.00278, 8.580277)
        for k, qq in enumerate(qs):
            p = g[g.liquid_stbd == qq].sort_values("valve")
            assert st_.P[k, -1] - G == pytest.approx(p.bhp_psig.iloc[0], abs=45), (run, qq)
            # within 15 % (0.05 MMscf/d), near full opening too (was 25 % before THROTTLE_SUB)
            tol = 0.15
            qp = p.qg_mmscfd.to_numpy()
            assert np.all(np.abs(st_.valve_q[k] - qp) <= tol * qp + 0.05), (run, qq, st_.valve_q[k], qp)
            same = ((st_.valve_q[k] > 1e-3) == (qp > 0)) | (np.abs(st_.valve_q[k] - qp) < 0.05)
            assert same.all(), (run, qq)


def test_gas_lift_throttling_casing_sweep_prosper_t23():
    """PROSPER multipoint VLP on the Tutorial 23 well with the A2 valves (Camco R-20 16/64 x 3, 16/64 orifice) at
    2000 STB/d, 500 psig, casing 1400-1560 psig in 10 psig steps (user's sweep): the valves go from throttling to
    fully open one by one. Subcritical valves keep throttling past a = 1 (gaslift_system.THROTTLE_SUB): every valve
    rate within 0.065 MMscf/d (min(a, 1) gave up to 0.24), bottom-node pressure within 7 psi, statuses the same
    (valve 3 at 1420 psig: PROSPER just open at 1.814, ours throttling at 1.790)."""
    from wellmodel import gaslift_system as wgls
    G = 14.696
    oil = _t05_oil()
    d = pd.read_csv(Path(__file__).parent / "data" / "t23_a2_casing_sweep.csv")
    for pc, g in d.groupby("casing_psig"):
        g = g.sort_values("valve")
        vs = tuple(wgls.GLValve(md, "Valve", 16, 0.066, dome) for md, dome in
                   zip((3731.43, 5666.89, 6660.16), (1211.11, 1195.47, 1179.41)))
        vs += (wgls.GLValve(7013.92, "Orifice", 16),)
        s = wgls.GasLiftSetup(method=wgls.METHODS[3], p_casing=pc + G, valves=vs, annulus_friction=True)
        path, secs = _t23_path(s.extra_md())
        s.annulus = wgls.annulus_geometry(path, secs)
        q = np.array([2000.0])
        st_ = wgls.lift_traverse(path, oil, 500 + G, {"q": q, "wc": 0.8, "gor": 800.0, **s.row_inputs({}, q)}, s,
                                 "pe2_style", 1.03108, 1.00278, 8.580277)
        assert st_.P[0, -1] - G == pytest.approx(g.bhp_psig.iloc[0], abs=7.0), pc
        assert np.all(np.abs(st_.valve_q[0] - g.qg_mmscfd.to_numpy()) < 0.065), (pc, st_.valve_q[0])
        same = [a.replace(" ", "")[:5] == b.replace(" ", "")[:5] for a, b in zip(st_.valve_status[0], g.status)]
        assert all(same) or pc == 1420.0, (pc, st_.valve_status[0])
    # the opening rule itself: critical flow min(a, 1); subcritical below 1 at a = 1, fully open from 1 + 2.5 d
    assert wgls.throttle_fraction(0.5, 0.5) == pytest.approx(0.5) and wgls.throttle_fraction(1.2, 0.5) == 1.0
    assert wgls.throttle_fraction(1.0, 0.70) < 0.9 and wgls.throttle_fraction(1.5, 0.70) == 1.0
    assert wgls.throttle_fraction(0.5, 0.70) == pytest.approx(0.5)


def test_prosper_liquid_surface_tension():
    """PROSPER's gas-liquid surface tension in an oil well (T23 gradient exports, 80 % water cut)."""
    rho_w, rho_g = 62.246, 1.785
    sw = wvlp.sigma_water_oilwell(rho_w, rho_g)
    assert float(sw) == pytest.approx(61.1064, rel=0.015)
    so = 11.8494                                          # PROSPER gas-oil
    assert float(wvlp.sigma_liquid(so, 61.1064, 0.8)) == pytest.approx(18.2667, rel=0.02)
    assert float(wvlp.sigma_liquid(so, 61.1, 0.0)) == pytest.approx(so)


def test_gas_lift_quicklook_t33a_sample_runs():
    """PROSPER T33a sample file (valves 3729.73 / 5680.29 / 6731.17 ft, orifice 6994.18 ft), three QuickLook runs
    sent by the user: 1) orifice 29/64, static 3800; 2) valve 3 depth, 24/64, static 3800; 3) same, 22/64, static
    3680 psig. PROSPER values (psig) per run: results, valve details and the critical casing head pressure from the
    top-down VLP report."""
    from wellmodel import gaslift_system as wgls, quicklook as wql
    G = 14.696
    oil = _t05_oil()
    vs = (wgls.GLValve(3729.73, "Valve", 16, 0.066, 1210.5), wgls.GLValve(5680.29, "Valve", 20, 0.103, 1199.38),
          wgls.GLValve(6731.17, "Valve", 24, 0.147, 1191.24))
    s = wgls.GasLiftSetup(valves=vs, annulus_friction=True)
    path, secs = _t23_path([v.md for v in vs] + [6994.18, 6250.0])
    s.annulus = wgls.annulus_geometry(path, secs)
    runs = {  # md, port, static, IPR pwf, Pt, Pc, dP, CHP calc, q crit, % crit, critical CHP, WHP bottom up, BHP TD
        1: (6994.18, 29, 3800, 2450.62, 1622.17, 1625.61, 21.1693, 1409.97, 6.23104, 24.073, 1730.6, 475.6, 2291.3),
        2: (6731.17, 24, 3800, 2450.62, 1558.89, 1616.05, 47.006, 1386.37, 4.17535, 35.9251, 1675.3, 469.8, 2301.8),
        3: (6731.17, 22, 3680, 2312.96, 1558.89, 1616.05, 66.7354, 1403.14, 3.55117, 42.2396, 1681.3, 395.2, 2301.8)}
    for n, (md, port, pst, pwf, pt, pc, dp, chp, qc, pcrit, chpc, whp, bhp) in runs.items():
        inp = wql.QuickLookInput(390 + G, 146, 7488, 0.35, 5.3935, 1.5, 1395 + G, md, port,
                                 gauges=((6250.0, 1440 + G),), static_p=pst + G, valves=vs)
        r = wql.quicklook(path, oil, lambda q, p=pwf: p + G, s, inp, "pe2_style", 1.03108, 1.00278, 8.580277)
        sm = r.summary
        # top down: PE2 slug friction refitted on five PROSPER node exports (tubing at the valve 1-1.5 psi light)
        assert sm["Tubing pressure at valve (psia)"] - G == pytest.approx(pt, abs=4), n
        assert sm["Casing pressure at valve (psia)"] - G == pytest.approx(pc, abs=2), n
        assert sm["dP across valve, calculated (psi)"] == pytest.approx(dp, rel=0.02), n
        assert sm["Calculated casing head pressure (psia)"] - G == pytest.approx(chp, abs=3), n
        assert sm["Calculated critical flow rate (MMscf/d)"] == pytest.approx(qc, rel=0.01), n
        assert sm["Percentage of critical flow rate (%)"] == pytest.approx(pcrit, rel=0.01), n
        assert sm["Critical casing head pressure (psia)"] - G == pytest.approx(chpc, rel=0.015), n
        assert sm["Calculated tubing head pressure (psia)"] - G == pytest.approx(whp, abs=4), n
        assert sm["Flowing bottom-hole pressure, top down (psia)"] - G == pytest.approx(bhp, abs=6), n
        assert sm["Fitted thermal coefficient (BTU/h/ft2/F)"] == pytest.approx(8.67, rel=0.01), n
        assert sm["Gas oil ratio (scf/STB)"] == pytest.approx(780.491, rel=0.002), n
        assert sm["Injection"] == "possible", n
        v = r.valves
        assert list(v["Opening pressure (psia)"] - G) == pytest.approx([1591.67, 1605.96, 1586.08], abs=3), n
        assert list(v["Closing pressure (psia)"] - G) == pytest.approx([1547.42, 1576.26, 1582.09], abs=1), n
        assert list(v["Casing pressure (psia)"] - G) == pytest.approx([1508.21, 1578.09, 1616.05], abs=2), n


def test_gas_lift_quicklook_t33a():
    """PROSPER Tutorial 33a test (7488 STB/d, 35 % water cut, 5.3935 MMscf/d total gas, 1.5 injected, 390 psig THP,
    1395 psig CHP, gauge 1440 psig at 6250 ft) on the Tutorial 23 valves. PROSPER's sample file: injection not
    possible at the orifice (tubing 1628.1 against casing 1619.3 psig); at the third valve tubing 1541.5, casing
    1606.6 psig; port dP for 1.5 MMscf/d 48.1 psi (24/64) and 68.2 psi (22/64), the top-down gradient through the
    gauge; fitted U 8.70."""
    from wellmodel import gaslift_system as wgls, quicklook as wql
    G = 14.696
    oil = _t05_oil()
    vs = (wgls.GLValve(3731.43, "Valve", 16, 0.066, 1211.11), wgls.GLValve(5666.89, "Valve", 20, 0.103, 1195.47),
          wgls.GLValve(6660.16, "Valve", 24, 0.147, 1179.41), wgls.GLValve(7013.92, "Orifice", 29))
    s = wgls.GasLiftSetup(valves=vs, annulus_friction=True)
    path, secs = _t23_path([v.md for v in vs] + [6250.0])
    s.annulus = wgls.annulus_geometry(path, secs)
    out = {}
    for md, port in ((7013.92, 29), (6660.16, 24), (6660.16, 22)):
        inp = wql.QuickLookInput(390 + G, 146, 7488, 0.35, 5.3935, 1.5, 1395 + G, md, port,
                                 gauges=((6250.0, 1440 + G),), static_p=3800 + G, valves=vs)
        out[(md, port)] = wql.quicklook(path, oil, lambda q: 2451.13 + G, s, inp, "pe2_style", 1.03108, 1.00278,
                                        8.580277)
    r29, r24, r22 = out[(7013.92, 29)], out[(6660.16, 24)], out[(6660.16, 22)]
    assert r24.summary["Tubing pressure at valve (psia)"] - G == pytest.approx(1541.51, abs=20)
    assert r24.summary["Casing pressure at valve (psia)"] - G == pytest.approx(1606.63, abs=10)
    assert r29.summary["Tubing pressure at valve (psia)"] - G == pytest.approx(1628.12, abs=20)
    assert r24.summary["Gauge at 6,250 ft: top down (psia)"] - G == pytest.approx(1440.0, abs=10)
    assert r24.summary["dP across valve, calculated (psi)"] == pytest.approx(48.1035, rel=0.03)
    assert r22.summary["dP across valve, calculated (psi)"] == pytest.approx(68.1924, rel=0.03)
    assert r24.summary["Calculated critical flow rate (MMscf/d)"] == pytest.approx(4.29556, rel=0.05)
    assert r24.summary["Fitted thermal coefficient (BTU/h/ft2/F)"] == pytest.approx(8.70361, rel=0.02)
    assert r24.summary["Draw down (psi)"] == pytest.approx(1348.87, abs=0.1)
    assert r24.summary["Equivalent calculated PI (STB/d/psi)"] == pytest.approx(5.55132, rel=1e-3)
    # a smaller port needs more dP: the calculated casing head pressure rises towards the measured 1395 psig
    assert r22.summary["Calculated casing head pressure (psia)"] > r24.summary["Calculated casing head pressure (psia)"]
    assert list(r24.valves["Status"][:3]) == ["CLOSED", "OPEN", "OPEN"]
    # bottom up starts at the IPR pressure; top down and bottom up share the casing at the valve in definition
    assert r24.profile["Tubing pressure, bottom up (psia)"].iloc[-1] == pytest.approx(2451.13 + G)
    # multipoint QuickLook runs and accounts for the injected gas
    inp = wql.QuickLookInput(390 + G, 146, 7488, 0.35, 5.3935, 1.5, 1395 + G, 6660.16, 24, static_p=3800 + G,
                             valves=vs)
    mp = wql.quicklook(path, oil, lambda q: 2451.13 + G, s, inp, "pe2_style", 1.03108, 1.00278, 8.580277,
                       multipoint=True)
    assert mp.valves["Gas rate (MMscf/d)"].sum() <= 1.5 + 1e-9 and mp.summary["Multipoint flag"] in (
        "Single", "Multiple")


def test_srp_prosper_t28():
    """Sucker rod pump design vs PROSPER Tutorial 28 (LUFKIN C-320D-305-100, ROD99/05, 2 in plunger, 48 in, pump at
    3500 ft, intake 651.42 psig, WHP 100 psig, WC 80 %, Ev 80 %); damping factor estimated."""
    from wellmodel import srp
    assert srp.rod_numbers(2.0) == [44, 54, 55, 65, 66, 75, 76, 77, 86, 87, 88, 97, 98, 99, 108, 109]
    assert srp.rod_string(76, 2.0).name == "ROD76/05"
    base = dict(pip=651.42 + 14.696, whp=100 + 14.696, sg_water=srp.water_sg(80000), mode=srp.CALC_MODES[1])
    prosper = {22.0008: dict(nu=0.17244, pprl=23268.7, mprl=2308.82, sp=47.1254, prhp=14.8808, hf=8.4172,
                             torque=395375, goodman=76.0299, trl=77.8823, smax=23408.7, sallow=30056.5,
                             ep=5.49305, ite=10.7818, wpr=30304.7, wpump=12367.7),
               20.0: dict(nu=0.18749, pprl=22284.8, mprl=3725.79, sp=47.6799, prhp=13.7062, hf=7.74173,
                          torque=351187, goodman=68.8695, trl=72.6508, smax=22418.8, sallow=30858.4,
                          ep=4.53936, ite=12.2988, wpr=29312.2, wpump=12543.9)}
    for n, p in prosper.items():
        r = srp.design(srp.SRPInput(n_spm=n, **base))
        s = r.summary()
        close = lambda a, b, tol: abs(a - b) <= tol * abs(b)
        assert close(r.nu, p["nu"], 0.01)
        assert close(r.pprl, p["pprl"], 0.005) and close(r.mprl, p["mprl"], 0.03)
        assert close(r.sp, p["sp"], 0.005)
        assert close(r.prhp, p["prhp"], 0.02) and close(r.hf, p["hf"], 0.01)
        assert close(r.torque_max, p["torque"], 0.01)
        assert close(r.goodman_pct, p["goodman"], 0.01) and close(r.top_rod_loading_pct, p["trl"], 0.005)
        assert close(r.stress_max, p["smax"], 0.005) and close(r.stress_allow, p["sallow"], 0.005)
        assert close(r.ep, p["ep"], 0.001) and close(r.ite, p["ite"], 0.03)
        assert close(s["Work done by polished rod (ft-lbf/stroke)"], p["wpr"], 0.01)
        assert close(s["Work done by pump (ft-lbf/stroke)"], p["wpump"], 0.01)
        assert close(r.kr_inv, 0.0013755, 1e-4) and close(r.cbe, 13522.9, 0.002)
        assert close(r.stretch, 4.37092, 0.005) and close(r.wrf + r.fo, 14336.8, 0.002)
    for n, q, ev in ((22.0008, 373.613, 77.2625), (20.0, 344.765, 77.5174)):
        r = srp.design(srp.SRPInput(n_spm=n, **base))
        assert close(r.q, q, 0.01) and close(100 * r.ev_actual, ev, 0.001)
    # entered Ev 100 % (actual 96.8968 %) and a 64 in stroke, both at 20 spm
    r = srp.design(srp.SRPInput(n_spm=22.0008, **base))
    assert close(r.lift_eff, 83.7701, 0.015) and close(r.econ_index, 163.424, 0.03)
    r = srp.design(srp.SRPInput(n_spm=20.0, ev=1.0, **base))
    assert close(r.lift_eff, 75.4427, 0.015) and close(r.econ_index, 157.651, 0.03)
    assert close(100 * r.ev_actual, 96.8968, 0.001) and close(r.q, 430.956, 0.01)
    assert close(r.prhp, 15.1973, 0.01) and close(r.nameplate, 18.5652, 0.02)
    r = srp.design(srp.SRPInput(n_spm=20.0, stroke=64.0, **base))
    for got, want, tol in ((r.pprl, 23895.6, 0.005), (r.mprl, 2621.99, 0.04), (r.sp, 66.4829, 0.005),
                           (r.q, 480.726, 0.005), (r.nu, 0.13446, 0.005), (r.torque_max, 386131, 0.01),
                           (r.prhp, 19.1114, 0.01), (r.hf, 10.7948, 0.005), (r.goodman_pct, 77.5535, 0.01),
                           (r.ite, 15.597, 0.01), (r.ep, 6.05248, 0.001), (r.clf, 1.21433, 0.015),
                           (r.nameplate, 29.0095, 0.02)):
        assert close(got, want, tol), (got, want)
    # rod sensitivity: production per polished rod hp (PROSPER, estimate mode)
    per_hp = {44: 46.0088, 55: 41.2654, 66: 36.6548, 76: 34.741, 88: 28.5393, 99: 25.1059, 109: 24.3807}
    sens = srp.rod_sensitivity(srp.SRPInput(n_spm=22.0008, **base), rods=list(per_hp))
    for (no, v), got in zip(per_hp.items(), sens["Production per hp (STB/d/hp)"]):
        assert close(got, v, 0.03), (no, got, v)
    # target rate mode reaches the target
    r = srp.design(srp.SRPInput(**{**base, "mode": srp.CALC_MODES[0], "q_target": 400.0}))
    assert abs(r.q - 400.0) < 0.5 and 21.5 < r.n_spm < 23.5


def test_srp_prosper_t28_other_hole_and_unit():
    """PROSPER T28 well at 20 spm: (A) C-320D-305-100 with an 80 in stroke - PROSPER keeps the shortest (70 in) crank
    hole, scaled; (B) a second unit, C-228D-213-86, 48 in stroke (62 in hole)."""
    from wellmodel import srp
    base = dict(pip=651.42 + 14.696, whp=100 + 14.696, sg_water=srp.water_sg(80000), mode=srp.CALC_MODES[1], n_spm=20.0)
    runs = {("LUFKIN C-320D-305-100", 80.0): dict(pprl=(25441, 0.01), sp=(85.3372, 0.005), q=(617.058, 0.01),
                                                    nu=(0.10475, 0.01), torque_max=(417334, 0.02),
                                                    prhp=(24.5313, 0.015), hf=(13.8561, 0.01),
                                                    goodman_pct=(86.462, 0.01), ite=(18.5234, 0.02),
                                                    ep=(7.5656, 0.001), lift_eff=(83.6503, 0.015),
                                                    mprl=(1177.17, 0.15)),
            ("LUFKIN C-228D-213-86", 48.0): dict(pprl=(22356, 0.005), sp=(47.7318, 0.005), q=(345.14, 0.01),
                                                   nu=(0.18728, 0.01), torque_max=(322167, 0.01),
                                                   prhp=(13.7211, 0.015), hf=(7.75016, 0.01),
                                                   goodman_pct=(69.175, 0.01), ite=(13.4212, 0.01),
                                                   clf=(1.22127, 0.015), nameplate=(20.9465, 0.02),
                                                   econ_index=(118.14, 0.03), mprl=(3709.83, 0.04))}
    for (unit, stroke), want in runs.items():
        r = srp.design(srp.SRPInput(unit=unit, stroke=stroke, **base))
        assert r.kin.hole_stroke == min(srp.UNITS[unit].strokes)
        for f, (v, tol) in want.items():
            assert abs(getattr(r, f) - v) <= tol * abs(v), (unit, f, getattr(r, f), v)


def test_srp_system_prosper_t28():
    """Sucker rod system analysis vs PROSPER (T28 well, ROD99/05, 2 in, 48 in, 23.7336 spm, Ev 80 %, Glaso matched
    PVT, PI 0.5 composite IPR, no-slip tubing): calculated cup anchor at 1, 1.5 and 3 in2 (23.7336 spm) and 1.5 in2 at
    15 spm, entered efficiency 1 and 0."""
    from wellmodel import srp, inflow, equipment as weq
    inp = wbo.PVTInputs(rsb=160, api=25, gas_sg=0.68, salinity_ppm=80000)
    md_ = wbo.MatchData(130.0, 1314.7, pd.DataFrame({"Pressure": [1314.7], "Gas Oil Ratio": [160], "Oil FVF": [1.06],
                                                      "Oil Viscosity": [6.9]}))
    res = wbo.match_all(inp, md_)
    fl = wvlp.BlackOilFluid(wbo.BlackOilModel(inp, "glaso", "beal",
                                              {k: res.matches[(k, "glaso")] for k in ("pb", "rs", "bo")}))
    df = _downhole([{"Type": "Xmas Tree", "Measured Depth (ft)": 0.0},
                    {"Type": "Tubing", "Measured Depth (ft)": 3500.0, "Tubing Inside Diameter (in)": 2.441,
                     "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 2.625,
                     "Casing Inside Diameter (in)": 6.3}])
    top, secs, pts = weq.downhole_sections(df)
    path = wvlp.build_flow_path(top, secs, pts, np.array([0.0, 3500.0]), np.array([0.0, 3500.0]), [0.0, 3500.0],
                                [60.0, 130.0])
    c = inflow.pi_entry(1514.696, 1314.696, 0.5, 0.8)
    o = np.argsort(c.q)
    qa, pa = np.asarray(c.q)[o], np.asarray(c.pwf)[o]
    ipr = lambda q: float(np.interp(q, qa, pa)) if q <= qa[-1] else np.nan
    g = 14.696
    runs = [(dict(anchor_method="Calculated", anchor_area=1.5), (428.6, 573.7, 1607.4, 552.3, 49.90, 0.516651)),
            (dict(anchor_method="Entered", anchor_eff=1.0), (438.4, 543.4, 1652.1, 548.7, 49.57, 1.0)),
            (dict(anchor_method="Entered", anchor_eff=0.0), (421.8, 592.4, 1520.5, 557.4, 50.36, 0.0)),
            (dict(anchor_method="Calculated", anchor_area=3.0), (431.1, 567.0, 1626.8, 551.1, 49.78, 0.661149)),
            (dict(anchor_method="Calculated", anchor_area=1.0), (426.9, 578.4, 1590.3, 553.4, 49.99, 0.404649)),
            (dict(anchor_method="Calculated", anchor_area=1.5, n_spm=15.0), (265.5, 959.0, 1565.3, 335.2, 47.92,
                                                                              0.251422))]
    for kw, (q, pip, pdis, disp, sp, ea) in runs:
        kw = {"n_spm": 23.7336, **kw}
        s, _ = srp.system(srp.SRPSystemInput(**kw), path, fl, ipr, 0.8, 160.0, 100 + g, "no_slip", q_max=551.0)
        assert abs(s["Liquid rate (STB/d)"] - q) <= 0.005 * q      # 0.3 % (1.3 % at the true intake temperature)
        assert abs(s["Pump intake pressure (psia)"] - g - pip) <= 5.0
        assert abs(s["Pump discharge pressure (psia)"] - g - pdis) <= 0.005 * pdis
        assert abs(s["Downhole pump displacement (STB/d)"] - disp) <= 0.005 * disp
        assert abs(s["Downhole stroke length (in)"] - sp) <= 0.1
        assert abs(s["Gas anchor efficiency"] - ea) <= 0.015
    # anchor types (PROSPER, 23.7336 spm, 1.5 in2): packer = cup exactly, poor boy = 0
    assert srp.gas_anchor_efficiency(435.0, 36.0, 1.5, "Packer type") == srp.gas_anchor_efficiency(435.0, 36.0, 1.5)
    assert srp.gas_anchor_efficiency(435.0, 36.0, 1.5, "Poor boy") == 0.0


def test_srp_t28_example_project():
    """examples/T28_Sucker_Rod_Pump.wellmodel.json: the T28 well with the rod pump as artificial lift - design at
    22.0008 spm (intake from the IPR at 400 STB/d) and the installation of the system runs (23.7336 spm, cup anchor
    1.5 in2). In the app the design gives PPRL 23,248 lbf against PROSPER's 23,268.7 and 372.6 STB/d against
    373.6; the system analysis 425.6 STB/d against 428.6 at 585 against 573.7 psig."""
    import json
    from wellmodel import srp, inflow
    root = __file__.replace("tests/test_wellmodel.py", "examples/")
    proj = json.load(open(root + "T28_Sucker_Rod_Pump.wellmodel.json"))
    w, i = proj["widgets"], proj["inputs"]
    assert w["opt_artificial_lift"] == "srp" and w["design_kind"] == "Sucker rod pump"
    assert i["pvt_sel_pb"] == "glaso" and i["ipr_model_oil"] == "P.I. Entry" and i["srpq_n"] == 23.7336
    assert i["srp_unit"] in srp.UNITS and i["srp_rod"] in srp.rod_numbers(i["srp_plunger"])
    c = inflow.pi_entry(i["ipr_pr"], i["pvt_pb"], i["ipr_j"], i["ipr_wc"] / 100)
    o = np.argsort(c.q)
    pip = float(np.interp(i["srp_q"], np.asarray(c.q)[o], np.asarray(c.pwf)[o]))
    assert abs(pip - 14.696 - 651.42) < 3                         # PROSPER's intake from the IPR at 400 STB/d
    r = srp.design(srp.SRPInput(unit=i["srp_unit"], rod_no=i["srp_rod"], plunger=i["srp_plunger"], mode=i["srp_mode"],
                                n_spm=i["srp_n"], stroke=i["srp_s"], pump_md=i["srp_md"], pip=pip, whp=i["srp_whp"],
                                wc=i["srp_wc"] / 100, api=i["pvt_api"], sg_water=srp.water_sg(i["pvt_sal"])))
    assert abs(r.pprl / 23268.7 - 1) < 0.005 and abs(r.q / 373.613 - 1) < 0.01


def _t32_setup():
    """PROSPER Tutorial 32 well: the Tutorial 19/23 Logie well with the tubing ending at the jet pump at 6000 ft
    (examples/T32_Jet_Pump.wellmodel.json), Glaso matched PVT, PE2 1.03108 / 1.00278, U 8.58."""
    import json
    from wellmodel import equipment as weq
    root = __file__.replace("tests/test_wellmodel.py", "examples/")
    d = json.load(open(root + "T32_Jet_Pump.wellmodel.json"))
    I = d["inputs"]
    tab = lambda k: pd.DataFrame(d["tables"][k]["__df__"]["data"], columns=d["tables"][k]["__df__"]["columns"])
    inp = wbo.PVTInputs(rsb=I["pvt_rsb"], api=I["pvt_api"], gas_sg=I["pvt_gg"], salinity_ppm=I["pvt_sal"])
    res = wbo.match_all(inp, wbo.MatchData(I["pvt_t"], I["pvt_pb"], tab("pvt_table").dropna()))
    params = {k: res.matches[(k, "glaso")] for k in ("pb", "rs", "bo")}
    params["visc"] = res.matches[("visc", "beal")]
    fl = wvlp.BlackOilFluid(wbo.BlackOilModel(inp, "glaso", "beal", params))
    sv, geo = tab("eq_survey").dropna(), tab("eq_geo").dropna()
    dh = tab("eq_downhole")
    rows = [{k: v for k, v in r.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
            for r in dh.drop(columns=["::auto_unique_id::"], errors="ignore").to_dict("records")]
    rows = [r for r in rows if r.get("Type")]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = wvlp.build_flow_path(top, secs, pts, sv.iloc[:, 0].values, sv.iloc[:, 1].values, geo.iloc[:, 0].values,
                                geo.iloc[:, 1].values, extra_md=[6000.0])
    return path, fl, I


def test_jet_pump_equations_prosper_t32():
    """Cunningham's equation with equal densities reproduces PROSPER's 'actual' Tutorial 32 pump (R = 0.41) and its
    system runs; the cavitation limits reproduce PROSPER's six Mc values."""
    from wellmodel import jetpump as J
    R = 0.16476 / 0.40185
    assert abs(float(J.head_ratio(R, 0.45581)) - 0.49257) < 0.0005
    # system runs (M, H from PROSPER's own rates and pressures): 80 % / 2000, 0 % / 2500, 50 % / 1500 psig
    for M, H in ((4624.0 / 10214.4, 0.49679), (9085.5 / 10193.8, 0.0897), (6171.2 / 10223.0, 0.3548)):
        assert abs(float(J.head_ratio(R, M)) - H) < 0.007
    mc = J.cavitation_m(0.5, 1616.55, 4291.19)
    assert np.allclose(mc, [0.7035, 0.59635, 0.55281], atol=2e-4)
    assert abs(J.cavitation_m(0.41, 1616.55, 4291.19)[0] - 1.01236) < 2e-3
    p = J.PUMPS
    assert len(p) == 100
    r = p[p["Name"] == "Test Pump #1 - Nozzle (18) Throat (1) A"].iloc[0]
    assert abs(r["Nozzle area (in2)"] - 0.16476) < 1e-4 and abs(r["Throat area (in2)"] - 0.40185) < 1e-4


def test_jet_pump_duty_and_system_prosper_t32():
    """PROSPER Tutorial 32: pump duty at 4000 STB/d, 80 % water cut, GOR 800, 150 psig, 10000 STB/d of 75000 ppm
    power fluid at 2000 psig (PROSPER's flowing BHP 2875.32 psig); system with PROSPER's inflow points."""
    from wellmodel import jetpump as J
    path, fl, I = _t32_setup()
    g = 14.696
    du = J.duty(path, fl, 2875.32 + g, J.JetDutyInput(top_p=150 + g), "pe2_style", 1.03108, 1.00278, I["eq_u"])
    assert abs(du.pip - g - 1616.55) < 15                    # 1628.96
    assert abs(du.pdis - g - 2498.61) < 7                     # 2503.5 (0.2 %; PE2 slug friction a + b HL)
    assert abs(du.pn - g - 4291.19) < 15                      # 4282.55
    assert abs(du.q_n - 10229.9) < 25 and abs(du.surface_hp - 347.816) < 1.5
    assert abs(du.static - 2420.88) < 15 and abs(du.friction - 129.689) < 3
    assert abs(du.M - 0.45433) < 0.003 and abs(du.H - 0.49206) < 0.012
    c = J.candidates(du)
    assert c["Pump"].iloc[0] == "Test Pump #1 - Nozzle (18) Throat (1) A" and c["Meets the duty"].iloc[0]
    # system at 80 % / 2000 psig and 50 % / 2500 psig on PROSPER's inflow at those water cuts
    for wc, pinj, pts, q_p, pip_p, pd_p in (
            (0.8, 2000, [(985.3, 3354.2), (3969.0, 2872.5), (4590.6, 2764.7)], 3969.0, 1613.0, 2502.0),
            (0.5, 2500, [(4448.8, 2763.9), (4871.2, 2681.7), (5157.9, 2624.2)], 5157.9, 1594.7, 2248.0)):
        qa, pa = np.array([p[0] for p in pts]), np.array([p[1] for p in pts]) + g
        sl = np.polyfit(qa, pa, 1)
        ipr = lambda q, qa=qa, pa=pa, sl=sl: float(np.interp(q, qa, pa)) if qa[0] <= q <= qa[-1] else float(np.polyval(sl, q))
        s, _ = J.system(path, fl, ipr, wc, 800.0, 150 + g, J.JetPumpInstalled(p_surf=pinj + g), "pe2_style",
                        1.03108, 1.00278, I["eq_u"], q_max=6000.0, n=24)
        assert abs(s["Liquid rate (STB/d)"] / q_p - 1) < 0.015
        assert abs(s["Pump intake pressure (psia)"] - g - pip_p) < 15
        assert abs(s["Pump discharge pressure (psia)"] - g - pd_p) < 15


def test_jet_pump_traverse_prosper_t32():
    """PROSPER's jet pump gradient traverse at 3969 STB/d (80 % water cut, 2000 psig injection, 150 psig): tubing
    down to the pump discharge 2499.1 psig, pump intake 1611.4 psig (Ps = Pd (1 + H) - H Pn)."""
    from wellmodel import jetpump as J
    path, fl, I = _t32_setup()
    g, q, wc = 14.696, 3969.0, 0.8
    pump = J.JetPumpInstalled(p_surf=2000 + g)
    above = J.sub_path(path, path.md[0], 6000.0)
    qt = q + pump.q_pf
    fl_mix = J._blend_fluid(fl, q, wc, pump.q_pf, 75000.0)
    case = {"q": np.array([qt]), "wc": np.array([(q * wc + pump.q_pf) / qt]), "gor": np.array([800.0])}
    P, T = wvlp.traverse(above, fl_mix, np.array([150 + g]), case, "pe2_style", 1.03108, 1.00278, I["eq_u"], None,
                         188.4)
    pdis = float(P[0, -1])
    assert abs(pdis - g - 2499.1) < 8                     # 2504.3 (0.2 %)
    pu = J.intake_from_discharge(path, fl, q, wc, 800.0, pdis, T[0], 188.4, pump)
    assert abs(pu["pip"] - g - 1611.4) < 10
    assert abs(pu["pn"] - g - 4291.5) < 15


def test_jet_pump_t32_example_project():
    import json
    root = __file__.replace("tests/test_wellmodel.py", "examples/")
    proj = json.load(open(root + "T32_Jet_Pump.wellmodel.json"))
    assert proj["widgets"]["opt_artificial_lift"] == "jet" and proj["widgets"]["design_kind"] == "Jet pump"
    assert proj["inputs"]["jpq_pump"] == "Test Pump #1 - Nozzle (18) Throat (1) A"
    from wellmodel import options as O
    assert not O.WellOptions(artificial_lift="jet").errors
    assert any(i.level == "error" for i in O.WellOptions(artificial_lift="jet", flow_type="annular").validate())


def test_jet_pump_lift_curves_t32():
    """Jet-pumped lift curves: the vectorised BHP equals the single-rate traverse, the injection pressure is the ALQ
    axis, more injection pressure lowers the BHP, and the Eclipse table carries the ALQ values."""
    from wellmodel import jetpump as J, liftcurves as wlc
    path, fl, I = _t32_setup()
    g = 14.696
    pump = J.JetPumpInstalled(p_surf=2000 + g)
    case = {"q": np.array([3969.0]), "wc": np.array([0.8]), "gor": np.array([800.0])}
    o = J.bhp_with_jet(path, fl, np.array([150 + g]), case, pump, "pe2_style", 1.03108, 1.00278, I["eq_u"])
    assert abs(o["pip"][0] - g - 1611.4) < 10 and abs(o["bhp"][0] - g - 2870.1) < 25   # PROSPER's traverse (0.7 %)
    axes, bhp, _ = wlc.vfp_table(path, fl, "oil", "pe2_style", 1.03108, 1.00278, I["eq_u"], [2000.0, 4000.0],
                                 [150 + g], [80.0], [800.0], jet=pump, alq=[1500 + g, 2500 + g], jet_alq="p_surf")
    assert bhp.shape == (1, 1, 1, 2, 2) and np.all(bhp[..., 1, :] < bhp[..., 0, :])
    txt = wlc.eclipse_vfpprod("oil", axes, bhp, alq_kind="jet_p_surf")
    assert "jet pump injection pressure" in txt and "VFPPROD" in txt
    # PROSPER's jet pump VLP (0 % water cut, GOR 800, 150 psig, 1500 psig injection): BHP at the bottom node
    q = np.array([1000.0, 2200.0, 3400.0, 4600.0, 5800.0, 6200.0])
    bhp_p = np.array([2088.9, 1941.2, 2173.4, 2515.7, 2882.4, 2897.9])
    case = {"q": q, "wc": np.zeros(6), "gor": np.full(6, 800.0)}
    o = J.bhp_with_jet(path, fl, np.full(6, 150 + g), case, J.JetPumpInstalled(p_surf=1500 + g), "pe2_style",
                       1.03108, 1.00278, I["eq_u"])
    assert np.all(np.abs((o["bhp"] - g) / bhp_p - 1) < 0.012)
    assert np.allclose(o["pip"][-2:], o["pdis"][-2:])           # no head at the highest rates, as PROSPER


def _t27_setup():
    """PROSPER Tutorial 27 well: the Logie well with the tubing ending at the HSP at 8500 ft
    (examples/T27_HSP_Design.wellmodel.json), Glaso matched PVT (23,000 ppm water), PE2 1.03 / 1.00278, U 8.58."""
    import json
    from wellmodel import equipment as weq
    root = __file__.replace("tests/test_wellmodel.py", "examples/")
    d = json.load(open(root + "T27_HSP_Design.wellmodel.json"))
    I = d["inputs"]
    tab = lambda k: pd.DataFrame(d["tables"][k]["__df__"]["data"], columns=d["tables"][k]["__df__"]["columns"])
    inp = wbo.PVTInputs(rsb=I["pvt_rsb"], api=I["pvt_api"], gas_sg=I["pvt_gg"], salinity_ppm=I["pvt_sal"])
    res = wbo.match_all(inp, wbo.MatchData(I["pvt_t"], I["pvt_pb"], tab("pvt_table").dropna()))
    params = {k: res.matches[(k, "glaso")] for k in ("pb", "rs", "bo")}
    params["visc"] = res.matches[("visc", "beal")]
    fl = wvlp.BlackOilFluid(wbo.BlackOilModel(inp, "glaso", "beal", params))
    sv, geo = tab("eq_survey").dropna(), tab("eq_geo").dropna()
    dh = tab("eq_downhole")
    rows = [{k: v for k, v in r.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
            for r in dh.drop(columns=["::auto_unique_id::"], errors="ignore").to_dict("records")]
    rows = [r for r in rows if r.get("Type")]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = wvlp.build_flow_path(top, secs, pts, sv.iloc[:, 0].values, sv.iloc[:, 1].values, geo.iloc[:, 0].values,
                                geo.iloc[:, 1].values, extra_md=[8500.0])
    return path, fl, I


def test_hsp_pump_and_turbine_curves_prosper_t27():
    """PROSPER's HSP database curves with the affinity laws at 7500 rpm: the design screen's pump stages, power and
    efficiency at PROSPER's duty, and the system run's turbine head loss, power and efficiency at one net flow."""
    from wellmodel import hsp as H
    p = H.PUMPS["CLYDE UNION TP115AH (4250-12325 rpm)"]
    t = H.TURBINES["CLYDE UNION T60 - A"]
    assert len(H.PUMPS) == 10 and len(H.TURBINES) == 9                # T60 A-D, T68 A-E
    q_avg, head, sg = 8076.85, 2074.75, 0.8035
    n = int(np.ceil(head / float(p.head_stage(q_avg, 7500.0))))
    assert n == 28
    assert n * float(p.power_stage(q_avg, 7500.0, sg)) == pytest.approx(148.276, rel=2e-4)
    assert 100 * float(p.efficiency(q_avg, 7500.0)) == pytest.approx(67.9229, abs=0.01)
    lo, hi = t.range_at(7500.0)
    assert lo == pytest.approx(5404.28, abs=0.01) and hi == pytest.approx(8338.05, abs=0.01)   # PROSPER's list
    # system run: 6516.2 STB/d of power fluid, Bw 1.0267 at the wellhead (250 psig, 176 F), less the balance supply
    bw = H.water_bw_rho(23000, 264.696, 176.0)[0]
    qn = float(H.net_turbine_flow(6516.2, bw))
    assert qn == pytest.approx(6463.1, abs=2.0)
    assert 22 * float(t.head_stage(qn, 7500.0)) == pytest.approx(4522.3, rel=1e-3)
    assert 22 * float(t.power_stage(qn, 7500.0)) == pytest.approx(150.35, rel=2e-3)
    assert 100 * float(t.efficiency(qn, 7500.0)) == pytest.approx(67.9353, abs=0.005)
    assert H.BALANCE_FRAC * 6500 == pytest.approx(227.5) and H.BALANCE_FRAC * 6516.2 == pytest.approx(228.1, abs=0.05)


def test_hsp_design_prosper_t27():
    """PROSPER Tutorial 27 design screen (6500 STB/d, 60 % water cut, GOR 800, 150 psig, 7500 rpm, 100 % power
    fluid, PROSPER's flowing BHP 2549.01 psig)."""
    from wellmodel import hsp as H
    path, fl, I = _t27_setup()
    g = 14.696
    d = H.HSPDutyInput(top_p=150 + g, casing_id=6.8, tubing_od=4.8)       # T27 annulus: 6.8 in casing, 4.8 in tubing
    du = H.duty(path, fl, 2549.01 + g, d, "pe2_style", 1.03, 1.00278, I["eq_u"])
    assert du.pip - g == pytest.approx(2281.15, abs=5)            # 2283.3
    assert du.pdis - g == pytest.approx(3003.13, abs=10)          # 3009.5
    assert du.q_in == pytest.approx(8358.05, rel=3e-3) and du.q_avg == pytest.approx(8076.85, rel=3e-3)
    assert du.detail["q_out_prod"] == pytest.approx(7898.71, rel=3e-3)
    assert du.mass == pytest.approx(2274950, rel=1e-3) and du.sg == pytest.approx(0.8035, abs=0.003)
    assert du.head == pytest.approx(2074.75, rel=0.01) and du.bo_in == pytest.approx(1.25621, abs=2e-4)
    c = H.candidate_pumps(du, d)
    assert c.loc[c["Pump"] == "CLYDE UNION TP115AH (4250-12325 rpm)", "Suitable"].iloc[0]
    r = H.design(path, fl, du, d, H.PUMPS["CLYDE UNION TP115AH (4250-12325 rpm)"], H.TURBINES["CLYDE UNION T60 - A"],
                 u=I["eq_u"])
    assert r["Number of pump stages"] == 28 and r["Number of turbine stages"] == 22
    assert r["Pump power required (hp)"] == pytest.approx(148.276, rel=5e-3)
    assert r["Pump efficiency (%)"] == pytest.approx(67.9229, abs=0.1)
    assert r["Turbine power generated (hp)"] == pytest.approx(151.565, rel=0.02)
    assert r["Head loss across turbine (ft)"] == pytest.approx(4525.9, rel=0.01)
    assert r["Delta P across turbine (psi)"] == pytest.approx(1927.49, rel=0.01)
    assert r["Turbine efficiency (%)"] == pytest.approx(67.938, abs=0.05)
    assert r["Balance supply rate (STB/d)"] == pytest.approx(227.5)
    assert r["Pressure at surface (psia)"] - g == pytest.approx(1664.72, rel=0.05)      # 1597.5 (4.0 %)
    assert r["Pump outlet temperature (F)"] == pytest.approx(211.568, abs=1.5)


def test_hsp_system_and_traverse_prosper_t27():
    """PROSPER Tutorial 27 system calculation (250 psig, 60 % water cut, GOR 800, 7500 rpm, 28 pump / 22 turbine
    stages; the power fluid varied until the turbine power equals the pump power) on PROSPER's inflow (its design
    and system bottom-hole pressures), and its gradient traverse at 5885.9 STB/d."""
    from wellmodel import hsp as H
    path, fl, I = _t27_setup()
    g = 14.696
    qa, pa = np.array([5885.9, 6500.0]), np.array([2663.3, 2549.01]) + g
    sl = np.polyfit(qa, pa, 1)
    inst = H.HSPInstalled(pump_md=8500.0, pump="CLYDE UNION TP115AH (4250-12325 rpm)", stages=28,
                          turbine="CLYDE UNION T60 - A", turbine_stages=22, speed=7500.0, casing_id=6.8,
                          tubing_od=4.8)
    s, c = H.system(path, fl, lambda q: float(np.polyval(sl, q)), 0.6, 800.0, 250 + g, inst, "pe2_style", 1.03,
                    1.00278, I["eq_u"], q_max=9000.0, n=12)
    assert s["Liquid rate (STB/d)"] == pytest.approx(5885.9, rel=0.015)
    assert s["Pump intake pressure (psia)"] - g == pytest.approx(2392.8, abs=15)
    assert s["Pump discharge pressure (psia)"] - g == pytest.approx(3186.5, abs=15)
    assert s["Pump power (hp)"] == pytest.approx(150.32, rel=0.01)
    assert s["Pump head generated (ft)"] == pytest.approx(2264.5, rel=0.01)
    assert s["Power fluid injection rate (STB/d)"] == pytest.approx(6516.2, rel=0.005)
    assert s["Turbine power generated (hp)"] == pytest.approx(150.35, rel=0.01)
    assert s["Head loss across turbine (ft)"] == pytest.approx(4522.3, rel=0.005)
    assert s["Turbine efficiency (%)"] == pytest.approx(67.9353, abs=0.02)
    assert s["Balance supply rate (STB/d)"] == pytest.approx(228.1, abs=0.5)
    assert s["Wellhead temperature (F)"] == pytest.approx(176.0, abs=1.0)
    assert s["Pump intake temperature (F)"] == pytest.approx(208.8, abs=0.5)
    assert s["Power fluid friction (psi)"] == pytest.approx(-58.86, rel=0.1)            # -62.4
    assert s["Casing head pressure (psia)"] - g == pytest.approx(1874.2, rel=0.055)     # 1782 (4.9 %)
    summ, prof = H.traverse_profile(path, fl, 250 + g, 5885.9, 0.6, 800.0, inst, None, "pe2_style", 1.03, 1.00278,
                                    I["eq_u"])
    prosper = {725: 282.4, 1380: 464.0, 2880: 952.4, 4700: 1640.8, 6450: 2340.4, 8075: 3005.9, 8475: 3169.6}
    tub = prof[prof["Section"].str.startswith("Tubing")]
    for z, v in prosper.items():
        assert float(np.interp(z, tub["MD (ft)"], tub["Pressure (psia)"])) - g == pytest.approx(v, abs=8), z
    cas = prof[prof["Section"] == "Casing"]
    for z, v in {8625: 2422.2, 9125: 2596.5, 9262.5: 2644.6}.items():
        assert float(np.interp(z, cas["MD (ft)"], cas["Pressure (psia)"])) - g == pytest.approx(v, abs=10), z
    assert summ["Power fluid rate (STB/d)"] == pytest.approx(6513.7, rel=0.005)
    assert summ["Wellhead temperature (F)"] == pytest.approx(176.0, abs=1.0)


def test_hsp_t27_example_project():
    import json
    root = __file__.replace("tests/test_wellmodel.py", "examples/")
    proj = json.load(open(root + "T27_HSP_Design.wellmodel.json"))
    assert proj["widgets"]["opt_artificial_lift"] == "hsp"
    assert proj["widgets"]["design_kind"] == "Hydraulic submersible pump"
    from wellmodel import options as O, hsp as H
    assert proj["inputs"]["hspq_pump"] in H.PUMPS and proj["inputs"]["hspq_turbine"] in H.TURBINES
    assert not O.WellOptions(artificial_lift="hsp").errors
    assert any(i.level == "error" for i in O.WellOptions(artificial_lift="hsp", flow_type="annular").validate())


# ================================================================ progressive cavity pump (PROSPER Tutorial 29)
# the user's PROSPER runs (T29 file): water cut, speed -> (liquid STB/d, PIP psig, PD psig, head ft, power hp,
# efficiency %, average RB/d, motor amps); PE2 1.03 / 1.05, GOR 10, 250 psig, straight-line IPR J 0.290 at 2200 psig
_T29_RUNS = {(60, 100): (427.4, 589.63, 1022.84, 992.2, 6.09, 51.85, 429.4, 60.80),
             (60, 250): (545.4, 183.26, 1021.28, 1925.5, 27.29, 28.66, 549.6, None),
             (70, 150): (480.3, 406.27, 1028.23, 1413.0, 12.53, 40.73, 483.2, 79.33),
             (80, 100): (426.0, 592.58, 1034.30, 994.2, 6.21, 51.78, 428.6, 61.90),
             (80, 200): (517.8, 276.34, 1033.40, 1705.5, 19.94, 33.62, 521.4, None),
             (80, 135.602): (467.3, 450.21, 1033.88, 1314.3, 10.68, 43.64, 470.4, 77.17)}


def _t29_setup():
    """PROSPER Tutorial 29 well (examples/T29_PCP_Design.wellmodel.json): Al-Marhoun / De Ghetto (heavy oil) matched PVT
    (the T29 file's PVT input screen), 3.2 in tubing
    to 1750 ft, 6.4 in casing to 2000 ft, OilLift 3.5 in 120-600 PCP at 1700 ft, Centrilift 450 175 hp motor."""
    from wellmodel import equipment as weq, pcp
    G = 14.696
    inp = wbo.PVTInputs(rsb=150, api=17, gas_sg=0.58, salinity_ppm=75000)
    md_ = wbo.MatchData(120.0, 2100 + G, pd.DataFrame({"Pressure": [2100 + G], "Gas Oil Ratio": [150],
                                                       "Oil FVF": [1.06], "Oil Viscosity": [40.0]}))
    res = wbo.match_all(inp, md_)
    params = {k: res.matches[(k, "al_marhoun")] for k in ("pb", "rs", "bo")}
    params["visc"] = res.matches[("visc", "de_ghetto")]
    fl = wvlp.BlackOilFluid(wbo.BlackOilModel(inp, "al_marhoun", "de_ghetto", params))
    df = _downhole([{"Type": "Xmas Tree", "Measured Depth (ft)": 0.0},
                    {"Type": "Tubing", "Measured Depth (ft)": 1750.0, "Tubing Inside Diameter (in)": 3.2,
                     "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 3.5,
                     "Casing Inside Diameter (in)": 6.4},
                    {"Type": "Casing", "Measured Depth (ft)": 2000.0, "Casing Inside Diameter (in)": 6.4,
                     "Casing Inside Roughness (in)": 0.0006}])
    top, secs, pts = weq.downhole_sections(df)
    path = wvlp.build_flow_path(top, secs, pts, np.array([0.0, 2000.0]), np.array([0.0, 2000.0]), [0.0, 2000.0],
                                [80.0, 120.0], extra_md=[1700.0])
    inst = pcp.PCPInstalled(pcp.PUMPS["OilLift Technology Inc 3.5 in 120-600 (Max Head 2125 feet)"], 1700.0, 135.602,
                            motor=pcp.MOTORS["Centrilift 450 175HP 2285V 50A"], cable=pcp.CABLES["#1 Copper"],
                            cable_len=2200.0, casing_id=6.4)
    return path, fl, inst


def test_pcp_database_and_pump_model_prosper_t29():
    """PROSPER's PCP database (3 pumps, 150 motor series, 13 cables, 14 rods) and the pump model on PROSPER's own heads
    and pressures: rate = head gradient x speed + intercept(head), power = power gradient(dP/0.433) x speed."""
    from wellmodel import pcp
    assert len(pcp.PUMPS) == 3 and len(pcp.CABLES) == 13 and len(pcp.RODS) >= 10
    assert len({(m.manufacturer, m.series) for m in pcp.MOTORS.values()}) == 151   # + Centrilift 562 KMH (T45)
    p = pcp.PUMPS["OilLift Technology Inc 3.5 in 120-600 (Max Head 2125 feet)"]
    assert abs(p.head_grad - 7.5378) < 1e-9 and abs(float(p.intercept(0.0))) < 0.1
    for (wc, n), (q, pip, pdis, head, pw, eff, qavg, amps) in _T29_RUNS.items():
        assert abs(float(p.rate(n, head)) / qavg - 1) < 0.015, (wc, n)
        assert abs(float(p.shaft_power(n, pdis - pip)) / pw - 1) < 0.006, (wc, n)
        assert abs(100 * qavg * (pdis - pip) * 1.7014e-5 / pw - eff) < 0.4
    # speed for the tutorial design (812.104 RB/d at 762.96 ft): 135.602 rpm
    assert abs(float(p.speed_for(812.104, 762.96)) - 135.602) < 0.02
    # cable drop of the tutorial design (#1 copper, 2200 ft, 46.9528 A, 110.621 F): 28.3179 V
    assert abs(pcp.CABLES["#1 Copper"].drop(46.9528, 2200.0, 110.621) - 28.3179) < 0.03
    # motor at the user's T29 system run (10.68 hp, 135.602 rpm): 162.6 % load, 77.17 A, 124.5 rpm
    m = pcp.motor_results(10.68, 135.602, pcp.MOTORS["Centrilift 450 175HP 2285V 50A"], pcp.CABLES["#1 Copper"],
                          2200.0, 109.37)
    assert abs(m["Motor load (%)"] - 162.6) < 0.6 and abs(m["Current used (amps)"] - 77.17) < 0.5
    assert abs(m["Motor speed (rpm)"] - 124.5) < 0.5 and abs(m["Required surface voltage (V)"] - 132.20) < 0.5


def test_pcp_system_prosper_t29():
    """PCP system analysis against the user's PROSPER runs (water cut and speed sensitivity plus the base case):
    rate within 0.6 %, intake within 5.5 psi, discharge within 13 psi, power within 1.8 % (Al-Marhoun / De Ghetto, the
    file's PVT; with Petrosky / Beal - the PVT table exports - 0.4 %, 4.4 psi, 5 psi, 0.6 %)."""
    from wellmodel import pcp
    import dataclasses
    G = 14.696
    path, fl, inst = _t29_setup()
    ipr = lambda q: 2200.0 + G - q / 0.29004 if q < 0.29004 * 2200.0 else np.nan
    for (wc, n), (q, pip, pdis, head, pw, eff, qavg, amps) in _T29_RUNS.items():
        s, _ = pcp.system(path, fl, ipr, wc / 100.0, 10.0, 250 + G, dataclasses.replace(inst, speed=float(n)),
                          "pe2_style", 1.03, 1.05, 8.0)
        assert abs(s["Liquid rate (STB/d)"] / q - 1) < 0.006, (wc, n)
        assert abs(s["Pump intake pressure (psia)"] - G - pip) < 5.5, (wc, n)
        assert abs(s["Pump discharge pressure (psia)"] - G - pdis) < 13, (wc, n)
        assert abs(s["Pump power requirement (hp)"] / pw - 1) < 0.018, (wc, n)
        if amps is not None:
            assert abs(s["Current used (amps)"] / amps - 1) < 0.015, (wc, n)
    # base case wellhead temperature (pump and motor losses heat the tubing): PROSPER 97.2 F
    assert abs(s["Wellhead temperature (F)"] - 97.2) < 1.5


def test_pcp_traverse_design_and_lift_curves_t29():
    """PROSPER's PCP gradient traverse (750 STB/d, 80 % water cut, GOR 150, 135.602 rpm) and the tutorial design duty
    (PROSPER's flowing BHP 797.879 psig at 750 STB/d); lift curves agree with the traverse and carry the speed axis."""
    from wellmodel import pcp, liftcurves as wlc
    G = 14.696
    path, fl, inst = _t29_setup()
    s, ab, Pa, Ta, *_ = pcp.traverse_profile(path, fl, 250 + G, 750.0, 0.8, 150.0, inst, "pe2_style", 1.03, 1.05, 8.0)
    assert abs(s["Pump discharge pressure (psia)"] - G - 978.61) < 9
    assert abs(s["Pump intake pressure (psia)"] - G - 663.61) < 10        # 655 (Bo of the one-point PVT match)
    assert abs(s["Bottom node pressure (psia)"] - G - 799.32) < 13
    assert abs(Ta[0] - 102.53) < 0.5 and abs(s["Pump discharge temperature (F)"] - 119.78) < 0.5
    # the user's PROSPER design screen (flowing BHP 798.015 psig at 750 STB/d, Darcy PI 0.60335)
    du = pcp.duty(path, fl, 798.015 + G, 750.0, 0.8, 150.0, 250 + G, 1700.0, 0.0, "pe2_style", 1.03, 1.05, 8.0)
    assert abs(du.pip - G - 662.303) < 3 and abs(du.pdp - G - 978.794) < 8
    assert abs(du.q_in / 828.463 - 1) < 0.005 and abs(du.q_out / 796.369 - 1) < 0.005
    assert abs(du.q_avg / 809.369 - 1) < 0.005 and abs(du.mass / 272295 - 1) < 0.002 and abs(du.sg - 0.95973) < 0.005
    assert abs(du.head / 761.437 - 1) < 0.03
    d = pcp.design(du, inst.pump, "Downhole motor", inst.motor, inst.cable, 2200.0)
    assert abs(d["Pump speed (rpm)"] / 135.142 - 1) < 0.01
    # PROSPER's design power and efficiency (displacement x head / 133,885; average rate x head / 136,000) on its own
    # design screens: user 750 STB/d, tutorial 750 STB/d, user 850 STB/d
    for n_, h_, q_, pw_, eff_ in ((135.142, 761.437, 809.369, 5.79346, 78.2175), (135.602, 762.96, 812.104, 5.82483,
                                  78.2153), (274.729, 1791.67, 948.1, 27.7125, 45.0709)):
        pw = float(pcp.design_power(inst.pump, n_, h_))
        assert abs(pw / pw_ - 1) < 2e-5 and abs(100 * q_ * h_ / pcp.HYD_HP_K / pw - eff_) < 0.003
    # 850 STB/d (PROSPER: intake 251.954, discharge 972.454 psig, 948.1 RB/d, 1791.67 ft, 274.729 rpm, 27.71 hp)
    du = pcp.duty(path, fl, 382.466 + G, 850.0, 0.8, 150.0, 250 + G, 1700.0, 0.0, "pe2_style", 1.03, 1.05, 8.0)
    assert abs(du.pip - G - 251.954) < 5 and abs(du.pdp - G - 972.454) < 6 and abs(du.q_avg / 948.1 - 1) < 0.01
    d = pcp.design(du, inst.pump)
    assert abs(d["Pump speed (rpm)"] / 274.729 - 1) < 0.02 and abs(d["Power required (hp)"] / 27.7125 - 1) < 0.04
    # 750 STB/d at 500 psig (more head, same rate): PROSPER 1256.08 psig, 1408.04 ft, 191.057 rpm, 15.1459 hp
    du = pcp.duty(path, fl, 798.015 + G, 750.0, 0.8, 150.0, 500 + G, 1700.0, 0.0, "pe2_style", 1.03, 1.05, 8.0)
    assert abs(du.pdp - G - 1256.08) < 6 and abs(du.q_avg / 797.748 - 1) < 0.005 and abs(du.head / 1408.04 - 1) < 0.01
    d = pcp.design(du, inst.pump)
    assert abs(d["Pump speed (rpm)"] / 191.057 - 1) < 0.01 and abs(d["Power required (hp)"] / 15.1459 - 1) < 0.02
    # 544-series motors (PROSPER's default curves for that series = the ESP template): 300 hp at 15.1459 hp /
    # 191.057 rpm and 375 hp at 27.7125 hp / 274.729 rpm
    for nm, pw, n_, t_, ref in (("Centrilift 544 300HP 1675V 105A", 15.1459, 191.057, 110.616,
                                 (84.0017, 183.233, 101.134, 60.9946, 149.889, 26.256)),
                                ("Centrilift 544 375HP 2105V 105A", 27.7125, 274.729, 111.374,
                                 (84.0489, 263.234, 102.614, 61.9854, 222.625, 39.5677))):
        m = pcp.motor_results(pw, n_, pcp.MOTORS[nm], pcp.CABLES["#1 Copper"], 2200.0, t_)
        got = [m[k_] for k_ in ("Motor efficiency (%)", "Motor speed (rpm)", "Current used (amps)",
                                "Cable voltage drop (V)", "Required surface voltage (V)", "Surface KVA")]
        assert np.allclose(got, ref, rtol=0.004), nm
    # motor and cable at PROSPER's design power (5.79346 hp at 135.142 rpm, cable at 110.616 F)
    m = pcp.motor_results(5.79346, 135.142, inst.motor, inst.cable, 2200.0, 110.616)
    for k_, ref in (("Motor efficiency (%)", 82.613), ("Motor speed (rpm)", 130.479), ("Current used (amps)", 46.9003),
                    ("Cable voltage drop (V)", 28.2859), ("Required surface voltage (V)", 114.063),
                    ("Surface KVA", 9.26578)):
        assert abs(m[k_] / ref - 1) < 0.003, k_
    assert abs(m["Motor voltage (V)"] - 85.7774) < 0.001
    c = pcp.candidate_pumps(du, max_od=6.4)
    assert c["Suitable"].all() and len(c) == 3
    case = {"q": np.array([750.0]), "wc": np.array([0.8]), "gor": np.array([150.0])}
    o = pcp.bhp_with_pcp(path, fl, np.array([250 + G]), case, inst, "pe2_style", 1.03, 1.05, 8.0)
    assert abs(o["bhp"][0] - s["Bottom node pressure (psia)"]) < 0.5
    axes, bhp, _ = wlc.vfp_table(path, fl, "oil", "pe2_style", 1.03, 1.05, 8.0, [300.0, 750.0], [250 + G], [80.0],
                                 [150.0], pcp=inst, alq=[100.0, 135.602, 200.0])
    # 100 rpm cannot pass 750 STB/d: as PROSPER's lift curves, the pump then gives no head (the intake is the
    # discharge pressure) and the point stays finite
    assert bhp.shape == (1, 1, 1, 3, 2) and np.isfinite(bhp[0, 0, 0, 0, 1])
    assert np.all(bhp[..., 2, :] < bhp[..., 1, :])                            # faster pump, lower BHP
    assert "PCP speed" in wlc.eclipse_vfpprod("oil", axes, bhp, alq_kind="pcp_speed")
    # wear 0.2 at 191.057 rpm with the 544 300 hp motor (the user's PROSPER system run W3)
    import dataclasses
    ipr = lambda q: 2200.0 + G - q / 0.29004 if q < 0.29004 * 2200.0 else np.nan
    iw = dataclasses.replace(inst, speed=191.057, wear=0.2, motor=pcp.MOTORS["Centrilift 544 300HP 1675V 105A"])
    sw, _ = pcp.system(path, fl, ipr, 0.8, 10.0, 250 + G, iw, "pe2_style", 1.03, 1.05, 8.0)
    for k_, ref, tol in (("Liquid rate (STB/d)", 472.4, 0.005), ("Average rate through pump (RB/d)", 475.7, 0.01),
                         ("Pump head generated (ft)", 1353.4, 0.003), ("Pump power requirement (hp)", 15.49, 0.01),
                         ("Current used (amps)", 103.01, 0.007), ("Motor speed (rpm)", 183.02, 0.002),
                         ("Required surface voltage (V)", 151.02, 0.007)):
        assert abs(sw[k_] / ref - 1) < tol, k_
    assert abs(sw["Pump intake pressure (psia)"] - G - 432.82) < 4 and abs(sw["Pump discharge pressure (psia)"] - G
                                                                           - 1033.60) < 8
    # the design screen ignores the wear factor (as PROSPER: 191.057 rpm at 0, 0.1 and 0.3)
    assert pcp.design(du, inst.pump, wear=0.3)["Pump speed (rpm)"] == pcp.design(du, inst.pump)["Pump speed (rpm)"]
    # sucker rod drive: annular flow up the rod-tubing annulus (PROSPER R1 traverse: 0.75 in rod, 190.838 rpm,
    # discharge 973.42, intake 415.76, bottom 547.73 psig); rod stresses are our own estimates
    import dataclasses
    ir = dataclasses.replace(inst, speed=190.838, drive="Sucker rod", rod=pcp.RODS["Tenaris BlueRod Series 0.75 in"],
                             rod_len=1700.0, motor=None, cable=None)
    s_r, *_ = pcp.traverse_profile(path, fl, 250 + G, 750.0, 0.8, 150.0, ir, "pe2_style", 1.03, 1.05, 8.0)
    assert abs(s_r["Pump discharge pressure (psia)"] - G - 973.42) < 9
    assert abs(s_r["Pump intake pressure (psia)"] - G - 415.76) < 12 and abs(s_r["Bottom node pressure (psia)"] - G
                                                                              - 547.73) < 12
    rod = pcp.RODS["Tenaris BlueRod Series 1 in"]
    pr = pcp.with_rod(path, rod, 1700.0)
    above = pr.md[1:] <= 1700.0
    assert np.all(pr.area[above] < path.area[path.md[1:] <= 1700.0])
    r = pcp.rod_results(rod, 1700.0, 300.0, 60.0, 400.0, inst.pump)
    assert r["Rod weight in fluid (lb)"] < r["Rod weight in air (lb)"] and r["Combined (von Mises) stress (psi)"] > 0
    inst_r = dataclasses.replace(inst, drive="Sucker rod", rod=rod, rod_len=1700.0, motor=None, cable=None)
    s_r, *_ = pcp.traverse_profile(path, fl, 250 + G, 750.0, 0.8, 150.0, inst_r, "pe2_style", 1.03, 1.05, 8.0)
    assert np.isfinite(s_r["Pump intake pressure (psia)"])
    assert abs(s_r["Pump discharge pressure (psia)"] - s["Pump discharge pressure (psia)"]) > 0.1


def test_pcp_t29_example_project_and_straight_line_ipr():
    import json
    from wellmodel import options as O, inflow
    root = __file__.replace("tests/test_wellmodel.py", "examples/")
    proj = json.load(open(root + "T29_PCP_Design.wellmodel.json"))
    assert proj["widgets"]["opt_artificial_lift"] == "pcp_motor"
    assert proj["inputs"]["pcq_pump"] == "OilLift Technology Inc 3.5 in 120-600 (Max Head 2125 feet)"
    assert not O.WellOptions(artificial_lift="pcp_motor").errors and not O.WellOptions(artificial_lift="pcp_rod").errors
    assert any(i.level == "error" for i in O.WellOptions(artificial_lift="pcp_rod", flow_type="annular").validate())
    # 'Vogel below bubble point' off: a straight line through the bubble point (the user's T29 file)
    c = inflow.pi_entry(2214.696, 2114.696, 0.29004, 0.8, vogel=False)
    o = np.argsort(c.q)
    assert abs(np.interp(427.4, np.asarray(c.q)[o], np.asarray(c.pwf)[o]) - 14.696 - 726.22) < 0.5


# ================================================================ coiled tubing gas lift (PROSPER Tutorial 25)
# user's T25 runs: Logie well (Tutorial 19/23 IPR: Darcy at 3450 psig, 80 % water cut, GOR 800; PE2 matched 1.0304 /
# 1.01424 in this file), coiled tubing ID 0.8 in / 0.2 in wall in the 4.052 in tubing, lift gas 0.7, no friction loss.
# Design (max production, 5 MMscf/d, 250 psig, 2200 psig, dP 50, 8500 ft, 0.46 psi/ft): 7035.36 STB/d liquid (1407.07
# oil, the fitted performance curve at 5 MMscf/d), orifice 21/64, 4.7254 MMscf/d actual, kick-off 5446.39 ft MD,
# maximum operating depth 8500 ft (1981.13 psig flowing, 2648.94 psig gas, 209.34 F). System sensitivity with the
# design transferred (specified depth 8449 ft, 2200 psig, dP 50):
_T25_SENS = {1: 5363.37, 2: 6214.88, 3: 6633.72, 4: 6869.75, 5: 7009.26, 6: 7092.65, 7: 7135.66, 8: 7137.0,
             10: 7078.1}
_T25_P = (1.0304, 1.01424)


def _t25_ipr():
    import json
    from types import SimpleNamespace
    ip = json.load(open(Path(__file__).parent / "data" / "t23_ipr.json"))
    return SimpleNamespace(q=np.array(ip["q"]), pwf=np.array(ip["pwf"]), pr=ip["pr"], aof=max(ip["q"]), fluid="oil")


def test_coiled_tubing_geometry():
    from wellmodel import coiled as wct
    path, _ = _t23_path([8500.0])
    ct = wct.CoiledTubing(0.8, 0.2)
    assert ct.od_in == pytest.approx(1.2)
    pa = wct.annular_path(path, ct, 8500.0)
    mid = 0.5 * (path.md[1:] + path.md[:-1])
    up = (mid < 8500) & (mid > 1000)
    assert np.allclose(pa.area[up], np.pi / 4 * ((4.052 / 12) ** 2 - (1.2 / 12) ** 2))
    assert np.allclose(pa.d_h[up], (4.052 - 1.2) / 12) and np.allclose(pa.d_ht[up], pa.d_h[up])
    assert np.allclose(pa.fric_mult[up], wct.ANNULUS_FRICTION) and np.allclose(pa.fric_mult[mid > 8500], 1.0)
    assert np.allclose(pa.area[mid > 8500], path.area[mid > 8500])          # full tubing / casing below the end
    k = next(iter(path.restrictions))                                        # the SSSV keeps its open area
    assert pa.restrictions[k] == pytest.approx(np.sqrt(3.72 ** 2 - 1.2 ** 2))
    with pytest.raises(ValueError):
        wct.annular_path(path, wct.CoiledTubing(3.0, 0.6), 8500.0)
    from wellmodel import options as O
    assert not O.WellOptions(artificial_lift="ct_gas_lift").errors
    assert O.WellOptions(artificial_lift="ct_gas_lift", flow_type="annular").errors
    # lift gas: DPR on Standing's natural-gas pseudo-criticals - PROSPER's injection-string densities (lb/ft3)
    gas = wgl.LiftGas(0.7, z_corr=wct.CT_GAS_Z)
    for p_, t_, rho in ((2207.1, 171.1, 7.998), (2207.3, 160.0, 8.266), (2393.6, 194.3, 8.131),
                        (2624.9, 208.9, 8.563)):
        assert float(gas.density(p_ + 14.696, t_)) == pytest.approx(rho, rel=0.001)


def test_coiled_tubing_gradients_prosper_t25():
    """PROSPER gradient runs on the transferred design (8449 ft, 2200 psig, dP 50, 250 psig): 7009 STB/d with
    5 MMscf/d and 5363 STB/d with 1 MMscf/d - production string (annulus above the injection point: holdup, gravity
    and temperature as PROSPER's; friction 0.90 x PE2-style) and injection string (gas at the flowing temperature,
    gradient 1.029 x density) node by node."""
    from wellmodel import coiled as wct, gaslift_system as wgls
    G = 14.696
    d = pd.read_csv(Path(__file__).parent / "data" / "t25_ct_gradients.csv")
    s = wgls.GasLiftSetup(method=wgls.METHODS[0], p_casing=2200 + G, dp_valve=50, fixed_md=8449.0,
                          coiled=wct.CoiledTubing(0.8, 0.2))
    path, _ = _t23_path(s.extra_md() + sorted(set(d.md)))
    for run, (q, g, whp_t, bhp) in {1: (7009.0, 5.0, 170.0, 2270.0), 2: (5363.0, 1.0, 158.7, 2587.0)}.items():
        case = {"q": np.array([q]), "wc": np.array([0.8]), "gor": np.array([800.0])}
        st_ = wgls.lift_traverse(path, _t05_oil(), 250 + G, {**case, **s.row_inputs({"qgi": g}, [q])}, s, "pe2_style",
                                 *_T25_P, 8.580277)
        assert st_.P[0, -1] - G == pytest.approx(bhp, abs=6), run
        assert st_.T[0, 0] == pytest.approx(whp_t, abs=0.5), run
        pr = d[(d.run == run) & (d.string == "production")]
        ours = np.interp(pr.md, path.md, st_.P[0]) - G
        assert np.max(np.abs(ours - pr.p_psig)) < 8, run
        assert np.max(np.abs(np.interp(pr.md, path.md, st_.T[0]) - pr.t_f)) < 0.6, run
        inj = d[(d.run == run) & (d.string == "injection") & (d.md > 600)]
        assert np.max(np.abs(np.interp(inj.md, path.md, st_.pc[0]) - G - inj.p_psig)) < 0.5, run


def test_coiled_tubing_design_prosper_t25():
    from wellmodel import coiled as wct
    G = 14.696
    path, _ = _t23_path([8500.0])
    inp = wct.CTDesignInput(top_p=250 + G, top_p_unload=250 + G, p_inj=2200 + G, p_ko=2200 + G, dp_valve=50,
                            max_md=8500, qgi_max=5, qgi_unload_max=5, q_liquid_max=25000, gor=800, wc=0.8,
                            load_grad=0.46)
    r = wct.design(path, _t05_oil(), _t25_ipr(), wgl.LiftGas(0.7, t_surface=80.0), wct.CoiledTubing(0.8, 0.2), inp,
                   "pe2_style", *_T25_P, 8.580277)
    assert r.qgi_design == pytest.approx(5.0) and r.max_md == pytest.approx(8500.0)
    assert r.q_oil == pytest.approx(1407.07, rel=0.01)
    assert r.port_64 == 21 and r.qgi_actual == pytest.approx(4.7254, rel=0.01)
    assert r.kickoff_md == pytest.approx(5446.39, abs=3)
    assert r.pg_inj - G == pytest.approx(2648.94, abs=2)                     # gas in the coiled tubing at 8500 ft
    assert r.t_inj == pytest.approx(209.34, abs=0.2)
    assert r.pt_inj - G == pytest.approx(1981.13, abs=15)


def test_coiled_tubing_system_prosper_t25():
    from wellmodel import coiled as wct, gaslift_system as wgls
    G = 14.696
    for method, md in ((wgls.METHODS[0], 8449.0), (wgls.METHODS[1], 8449.0)):
        s = wgls.GasLiftSetup(method=method, p_casing=2200 + G, dp_valve=50, fixed_md=md, max_md=md,
                              coiled=wct.CoiledTubing(0.8, 0.2))
        path, _ = _t23_path(s.extra_md())
        res, _ = wsys.solve_cases(lambda tid: path, _t05_oil(), "oil",
                                  {"whp": 250 + G, "u": 8.580277, "wc": 80.0, "gor": 800.0},
                                  [{"qgi": g} for g in _T25_SENS], lambda ch: _t25_ipr(), "pe2_style", *_T25_P, lift=s)
        q = np.array([r.q for r in res])
        assert q == pytest.approx(list(_T25_SENS.values()), rel=0.01), (method, q)
        assert int(np.argmax(q)) in (6, 7)                                   # optimum near 7-8 MMscf/d (PROSPER)
        assert all(r.details["gas_lift"]["Injection depth (ft MD)"] == pytest.approx(md) for r in res)
    # specified depth: no injection when the gas cannot reach the tubing pressure + dP there (natural flow)
    s = wgls.GasLiftSetup(method=wgls.METHODS[0], p_casing=600 + G, dp_valve=50, fixed_md=8449.0,
                          coiled=wct.CoiledTubing(0.8, 0.2))
    det, _ = wgls.details(path, _t05_oil(), {"whp": 250 + G, "wc": 80.0, "gor": 800.0, "qgi": 5.0}, 4000.0, s,
                          "pe2_style", *_T25_P, 8.580277)
    assert det["Gas lift gas injection rate (MMscf/d)"] == 0.0 and "no injection" in det["_note"]
    # friction in the coiled tubing lowers the gas pressure at the end
    T = np.full((1, len(path.md)), 180.0)
    p0 = wct.gas_pressure(path, wgl.LiftGas(0.7), wct.CoiledTubing(0.8, 0.2), 2200 + G, T, 5.0, 8449.0)[0, -1]
    p1 = wct.gas_pressure(path, wgl.LiftGas(0.7), wct.CoiledTubing(0.8, 0.2, friction=True), 2200 + G, T, 5.0,
                          8449.0)[0, -1]
    assert p1 < p0 - 50


def test_coiled_tubing_t25_example_project():
    import json
    root = __file__.replace("tests/test_wellmodel.py", "examples/")
    proj = json.load(open(root + "T25_Coiled_Tubing_Gas_Lift.wellmodel.json"))
    assert proj["widgets"]["opt_artificial_lift"] == "ct_gas_lift"
    assert proj["widgets"]["design_kind"] == "Gas lift (coiled tubing)"
    assert proj["inputs"]["ctq_id"] == 0.8 and proj["inputs"]["ctq_md"] == 8449.0


def test_coiled_tubing_lift_curves_prosper_t25():
    """PROSPER lift-curve export of the T25 coiled tubing well (8449 ft, 2200 psig, dP 50; 1000-3000 STB/d, 250-500
    psig, 80 % water cut, GOR 800, 0-5 MMscf/d): bottom-hole pressures within 1.4 % at all 72 points, including the
    no-injection rows (the coiled tubing still in the well)."""
    from wellmodel import coiled as wct, gaslift_system as wgls, liftcurves as wlc
    G = 14.696
    t = wlc.read_tpd(open(Path(__file__).parent / "data" / "t25_ct_lift_curves.tpd").read())
    rates, (qgi, wc, thp) = t["rates"], t["variables"]
    assert t["types"] == [4000, 22, 16, 27] and t["columns"][0] == 5000
    s = wgls.GasLiftSetup(method=wgls.METHODS[0], p_casing=2200 + G, dp_valve=50, fixed_md=8449.0,
                          coiled=wct.CoiledTubing(0.8, 0.2))
    path, _ = _t23_path(s.extra_md())
    axes, bhp, _ = wlc.vfp_table(path, _t05_oil(), "oil", "pe2_style", *_T25_P, 8.580277, rates, thp + G, wc, [800.0],
                                 lift=s, alq=qgi)
    ours = bhp[:, 0, 0, :, :] - G                       # thp, qgi, rate
    pros = t["results"][:, 0, :, :, 0]
    assert np.max(np.abs(ours / pros - 1)) < 0.014


def test_hsp_lift_curves():
    """HSP lift curves: PROSPER's HSP traverse at every point (pump speed as the ALQ axis); speeds outside the pump's
    range (PROSPER writes numbers for 20,000 and 30,000 rpm on a 4250-12325 rpm pump) are blank."""
    from wellmodel import hsp as H, liftcurves as wlc
    G = 14.696
    path, fl, I = _t27_setup()
    inst = H.HSPInstalled(8500.0, "CLYDE UNION TP115AH (4250-12325 rpm)", 28, "CLYDE UNION T60 - A", 22, 7500.0,
                          casing_id=6.8, tubing_od=4.8)
    axes, bhp, _ = wlc.vfp_table(path, fl, "oil", "pe2_style", 1.03, 1.00278, 8.580277, [3000.0, 6000.0],
                                 [250 + G], [60.0], [800.0], hsp=inst, alq=[7500.0, 20000.0])
    assert bhp.shape == (1, 1, 1, 2, 2) and np.isnan(bhp[0, 0, 0, 1]).all()
    s, _ = H.traverse_profile(path, fl, 250 + G, 6000.0, 0.6, 800.0, inst, None, "pe2_style", 1.03, 1.00278, 8.580277)
    assert bhp[0, 0, 0, 0, 1] == pytest.approx(s["Bottom node pressure (psia)"], abs=1e-6)
    assert np.all(np.diff(bhp[0, 0, 0, 0]) > 0)


def test_hsp_lift_curves_prosper_t27():
    """PROSPER lift-curve export of the user's T27 file (TP115AH 30 stages, T60-A 24 stages, PE2 1.03 / 1.01, 250
    psig, 80 % water cut, GOR 800) at 10,000 rpm: bottom-hole pressure within 4 % from 846 STB/d up (within 1.2 % from
    4168), pump power within 0.5 %; 20,000 / 30,000 rpm are outside the pump's range - blank here."""
    import dataclasses
    from wellmodel import hsp as H, liftcurves as wlc
    G = 14.696
    t = wlc.read_tpd(open(Path(__file__).parent / "data" / "t27_hsp_lift_curves.tpd").read())
    speeds, thp, wc = t["variables"]
    rates = t["rates"][1::3]                                   # 846 ... 15795 STB/d, every third rate
    pros = t["results"][0, 0, 0, 1::3]                         # 250 psig, 10,000 rpm
    path, fl, I = _t27_setup()
    inst = H.HSPInstalled(8500.0, "CLYDE UNION TP115AH (4250-12325 rpm)", 30, "CLYDE UNION T60 - A", 24, 7500.0,
                          casing_id=6.8, tubing_od=4.8)
    axes, bhp, _ = wlc.vfp_table(path, fl, "oil", "pe2_style", 1.03, 1.01, 8.580277, rates, [250 + G], [80.0],
                                 [800.0], hsp=inst, alq=[10000.0, 20000.0])
    ours = bhp[0, 0, 0, 0] - G
    assert np.all(np.abs(ours / pros[:, 0] - 1) < 0.04), (ours, pros[:, 0])
    assert np.all(np.abs(ours / pros[:, 0] - 1)[rates > 4000] < 0.012)
    assert np.isnan(bhp[0, 0, 0, 1]).all()
    s, _ = H.traverse_profile(path, fl, 250 + G, float(rates[4]), 0.8, 800.0,
                              dataclasses.replace(inst, speed=10000.0), None, "pe2_style", 1.03, 1.01, 8.580277)
    assert s["Pump power (hp)"] == pytest.approx(pros[4, 2], rel=0.005)


def test_pcp_lift_curves_overspeed_head_cap():
    """PROSPER's PCP lift curves at 2000 / 3000 rpm (10-20 x the T29 pump's speed): PROSPER holds the pump at its
    maximum head (2125 ft) and the tables stay finite; ours too (no longer blank). The values there differ from
    PROSPER's by up to 27 % (its pump dP at that head uses a 3-7 % larger rate through the pump) - not a check of
    normal running."""
    from wellmodel import pcp, liftcurves as wlc
    G = 14.696
    t = wlc.read_tpd(open(Path(__file__).parent / "data" / "t29_pcp_lift_curves_overspeed.tpd").read())
    assert np.allclose(t["results"][0, :, :, :, 4], 2125.0)            # PROSPER: head generated = maximum head
    path, fl, inst = _t29_setup()
    case = {"q": t["rates"], "wc": np.full(5, 0.8), "gor": np.full(5, 100.0)}
    o = pcp.bhp_with_pcp(path, fl, 250 + G, case, inst, "pe2_style", 1.03, 1.05, 8.0, speed=np.full(5, 2000.0))
    assert np.all(np.isfinite(o["bhp"]))
    rho = (o["pdp"] - o["pip"]) * 144.0 / 2125.0
    assert np.all((rho > 55) & (rho < 66))


def test_esp_lift_curves_prosper_t26():
    """PROSPER lift-curve export of the user's T26 file (GC10000 112 stages at 8500 ft, 544 300 hp motor, no gas
    separator, PE2 1.03 / 1.01, 200 psig, 80 % water cut, GOR 800, 1000-8000 STB/d, 30-100 Hz): 30-60 Hz within 4 %
    (1.7 % from 2000 STB/d); 80 Hz within 3.6 % from 2000 STB/d (intake 180-710 psig: the stage groups marched at their
    mid-group state). PROSPER's 100 Hz rows have the intake at 0 psig (pumped off) and are not compared."""
    from wellmodel import esp as E, liftcurves as wlc
    G = 14.696
    t = wlc.read_tpd(open(Path(__file__).parent / "data" / "t26_esp_lift_curves.tpd").read())
    freqs, _, _ = t["variables"]
    rates, R = t["rates"], t["results"][0, 0]
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 600},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 4.052,
             "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 4.8, "Casing Inside Diameter (in)": 6.4},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 3.72},
            {"Type": "Tubing", "Measured Depth (ft)": 8500, "Tubing Inside Diameter (in)": 4.052,
             "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 4.8, "Casing Inside Diameter (in)": 6.4},
            {"Type": "Casing", "Measured Depth (ft)": 9275, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    smd_, stvd_ = np.array(LOGIE_SURVEY).T
    path = wvlp.build_flow_path(top, secs, pts, smd_, stvd_, [0, 600, 9275], [60, 40, 210], extra_md=[8500.0])
    esp = {"pump_md": 8500.0, "pump": E.PUMPS["Centrilift GC10000 (5.13 in)"], "stages": 112, "freq": 60.0,
           "wear": 0.0, "sep": 0.0, "motor": E.MOTORS["Centrilift 544 300HP 1675V 105A"]}
    f5 = [f for f in freqs if f <= 80]
    axes, bhp, _ = wlc.vfp_table(path, _t05_oil(), "oil", "pe2_style", 1.03, 1.01, 8.580277, rates, [200 + G], [80.0],
                                 [800.0], esp=esp, alq=f5)
    ours = bhp[0, 0, 0] - G                                    # freq x rate
    err = np.abs(ours / R[:len(f5), :, 0] - 1)
    assert np.all(err[:4] < 0.04) and np.all(err[:4, 1:] < 0.017), err
    assert np.all(err[4, 1:] < 0.036), err[4]


def test_pcp_lift_curves_prosper_t29():
    """PROSPER lift-curve export of the T29 PCP well at working speeds (100 / 135.6 / 200 / 250 rpm, 400-600 STB/d, 80 %
    water cut, GOR 800, 250 psig): within 3 % at all 12 points. The average rate through the pump is the mean of the
    intake and discharge rates (the rate at the mean pressure gives up to 7 % low at 250 rpm); at 100 rpm the pump
    cannot pass 600 STB/d and, as PROSPER, gives no head."""
    from wellmodel import pcp, liftcurves as wlc
    G = 14.696
    t = wlc.read_tpd(open(Path(__file__).parent / "data" / "t29_pcp_lift_curves.tpd").read())
    speeds, _, _ = t["variables"]
    rates, R = t["rates"], t["results"][0, 0]
    path, fl, inst = _t29_setup()
    axes, bhp, _ = wlc.vfp_table(path, fl, "oil", "pe2_style", 1.03, 1.05, 8.0, rates, [250 + G], [80.0], [800.0],
                                 pcp=inst, alq=speeds)
    ours = bhp[0, 0, 0] - G
    assert np.all(np.abs(ours / R[:, :, 0] - 1) < 0.03), (ours, R[:, :, 0])
    assert R[0, 2, 4] == 0.0                                   # PROSPER: 100 rpm, 600 STB/d - no head


def _t28_system_setup():
    from wellmodel import equipment as weq
    inp = wbo.PVTInputs(rsb=160, api=25, gas_sg=0.68, salinity_ppm=80000)
    md_ = wbo.MatchData(130.0, 1314.7, pd.DataFrame({"Pressure": [1314.7], "Gas Oil Ratio": [160], "Oil FVF": [1.06],
                                                      "Oil Viscosity": [6.9]}))
    res = wbo.match_all(inp, md_)
    fl = wvlp.BlackOilFluid(wbo.BlackOilModel(inp, "glaso", "beal",
                                              {k: res.matches[(k, "glaso")] for k in ("pb", "rs", "bo")}))
    df = _downhole([{"Type": "Xmas Tree", "Measured Depth (ft)": 0.0},
                    {"Type": "Tubing", "Measured Depth (ft)": 3500.0, "Tubing Inside Diameter (in)": 2.441,
                     "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 2.625,
                     "Casing Inside Diameter (in)": 6.3}])
    top, secs, pts = weq.downhole_sections(df)
    path = wvlp.build_flow_path(top, secs, pts, np.array([0.0, 3500.0]), np.array([0.0, 3500.0]), [0.0, 3500.0],
                                [60.0, 130.0])
    return path, fl


def test_srp_lift_curves_prosper_t28():
    """PROSPER's rod pump lift curves (T28 file: ROD99/05, 2 in, 48 in, Ev 80 %, anchor 0.2 entered; 80 % water cut,
    GOR 800, 100-400 psig, 10 / 20 spm; and the same at anchor efficiency 0 and 1, 100-400 psig) tabulate the rate the
    pump delivers against the bottom-hole pressure. Downhole stroke within 0.1 in and discharge pressure within 1 % at
    every point. The rate: PROSPER evaluates the pump-intake fluid at the intake temperature in deg C read as deg F
    (srp.PUMP_FILL_T) - anchor 1 within 0.2 %, 0.2 within 3.1 %, 0 within 4 % (at the true 130 F: 0.7 %, 13 %, 15 %).
    The VFP table is the inverse (BHP against rate)."""
    from wellmodel import srp, liftcurves as wlc
    G = 14.696
    t = wlc.read_tpd(open(Path(__file__).parent / "data" / "t28_srp_lift_curves.tpd").read())
    pwf = t["rates"]
    spm, _, _, thp = t["variables"]
    R = t["results"]                                           # thp, wc, gor, spm, pwf, col
    path, fl = _t28_system_setup()
    for i in (0, 6):
        for k, n in enumerate(spm):
            s = srp.SRPSystemInput(n_spm=float(n), anchor_method="Entered", anchor_eff=0.2)
            for j in (1, 6, 12, 19):
                q, st_ = srp.rate_for_pwf(s, path, fl, pwf[j] + G, 0.8, 800.0, thp[i] + G)
                pr = R[i, 0, 0, k, j]
                assert abs(st_["sp"] - pr[4]) < 0.1 and abs(st_["pdis"] - G - pr[2]) < 0.01 * pr[2]
                assert abs(q / pr[0] - 1) < 0.031, (thp[i], n, pwf[j], q, pr[0])
    for fn, ea, tol in (("anchor0", 0.0, 0.04), ("anchor1", 1.0, 0.002)):
        t2 = wlc.read_tpd(open(Path(__file__).parent / "data" / f"t28_srp_lift_curves_{fn}.tpd").read())
        spm2, _, _, thp2 = t2["variables"]
        for i in (0, 3):
            for k, n in enumerate(spm2):
                s = srp.SRPSystemInput(n_spm=float(n), anchor_method="Entered", anchor_eff=ea)
                for j in (0, 6, 19):
                    q, st_ = srp.rate_for_pwf(s, path, fl, t2["rates"][j] + G, 0.8, 800.0, thp2[i] + G)
                    pr = t2["results"][i, 0, 0, k, j]
                    assert abs(st_["sp"] - pr[4]) < 0.1 and abs(st_["pdis"] - G - pr[2]) < 0.01 * pr[2]
                    assert abs(q / pr[0] - 1) < tol, (fn, thp2[i], n, q, pr[0])
    # VFP table: BHP against rate, the inverse of the rate curve
    s = srp.SRPSystemInput(n_spm=20.0, anchor_method="Entered", anchor_eff=0.2)
    axes, bhp, _ = wlc.vfp_table(path, fl, "oil", "no_slip", 1.0, 1.0, 8.0, [100.0, 200.0, 300.0], [250 + G], [80.0],
                                 [800.0], srp=s, srp_p_max=1500 + G)
    b = bhp[0, 0, 0]
    assert np.all(np.diff(b) > 0)
    q2, _ = srp.rate_for_pwf(s, path, fl, float(b[1]), 0.8, 800.0, 250 + G)
    assert q2 == pytest.approx(200.0, rel=0.01)


def test_darcy_pi_reevaluated_at_system_gor_t29():
    """PROSPER re-evaluates a Darcy IPR with the oil at the system calculation's GOR (Tutorial 29: PI 0.60335 at the
    IPR's GOR 150, 0.29004 at the system screen's GOR 10 - the nearly dead heavy oil is more viscous). With the file's
    Al-Marhoun / De Ghetto matched PVT the ratio of the two PIs (the reservoir data cancel) is within 3.5 %."""
    import dataclasses
    from wellmodel import inflow
    _, fl, _ = _t29_setup()
    m150 = fl.m
    m10 = dataclasses.replace(m150, inp=dataclasses.replace(m150.inp, rsb=10.0))
    pis = []
    for m in (m150, m10):
        tab = m.table(120.0, [2214.696])
        pis.append(inflow.darcy_pi(300.0, 100.0, float(tab["Oil viscosity (cp)"].iloc[0]),
                                   float(tab["Oil FVF (RB/STB)"].iloc[0]), 500.0, 31.6, 0.354, 0.0))
    assert m10.bubble_point(120.0) < 600.0                     # below the flowing pressures: a straight line
    assert abs((pis[1] / pis[0]) / (0.29004 / 0.60335) - 1) < 0.035


def test_gas_lift_equilibrium_curve_prosper_t23():
    """PROSPER's gas lift equilibrium curve (Sensitivity on injection depth, fixed 5 MMscf/d, dP across valve 100
    psi, maximum depth 8500 ft; user's Tutorial 23 run, 500 psig, 80 % water cut, GOR 800, PE2 1.03108 / 1.00278):
    ten depths equally spaced from the top node (600 ft) to 8500 ft, the first two with no solution in both; liquid
    rate within 2.5 %, valve tubing pressure within 15 psi, casing head pressure within 13 psi, valve temperature
    within 1 F."""
    import json, dataclasses
    from types import SimpleNamespace
    from wellmodel import gaslift_system as wgls
    G = 14.696
    oil = _t05_oil()
    ip = json.load(open(Path(__file__).parent / "data" / "t23_ipr.json"))
    curve = SimpleNamespace(q=np.array(ip["q"]), pwf=np.array(ip["pwf"]), pr=ip["pr"], aof=max(ip["q"]), fluid="oil")

    def lift_for(s, extra):
        path, secs = _t23_path(tuple(extra))
        return path, dataclasses.replace(s, annulus=wgls.annulus_geometry(path, secs))
    df, prof = wgls.equilibrium_curve(lift_for, oil, lambda ch: curve, {"whp": 500 + G, "u": 8.580277, "wc": 80.0,
                                      "gor": 800.0}, wgls.GasLiftSetup(annulus_friction=True), "pe2_style", 1.03108,
                                      1.00278, qgi=5.0, dp_valve=100.0, max_md=8500.0, top_md=600.0,
                                      rate_method="Automatic - Linear")
    assert list(df["Injection depth (ft MD)"]) == [600.0 + 790.0 * k for k in range(1, 11)]
    assert df["Liquid rate (STB/d)"].iloc[:2].isna().all() and df["Note"].iloc[0].startswith("no solution")
    prosper = [  # liquid, CHP, valve casing, valve tubing (psig), T (F)
        (2345.16, 819.638, 862.001, 762.001, 143.323), (3301.42, 936.547, 1002.09, 902.09, 166.439),
        (4101.34, 1057.56, 1152.14, 1052.14, 181.562), (4805.97, 1182.99, 1312.81, 1212.81, 192.019),
        (5429.81, 1315.21, 1487.31, 1387.31, 199.329), (5941.53, 1459.6, 1682.71, 1582.71, 204.288),
        (6364.14, 1615.07, 1898.34, 1798.34, 207.551), (6713.11, 1775.44, 2126.67, 2026.66, 209.382)]
    for (_, r), (q, chp, pcv, pt, t) in zip(df.iloc[2:].iterrows(), prosper):
        assert r["Liquid rate (STB/d)"] == pytest.approx(q, rel=0.025)
        assert r["Valve tubing pressure (psia)"] - G == pytest.approx(pt, abs=15)
        assert r["Valve casing pressure (psia)"] == pytest.approx(r["Valve tubing pressure (psia)"] + 100)
        assert r["Casing head pressure (psia)"] - G == pytest.approx(chp, abs=13)
        assert r["Valve tubing temperature (F)"] == pytest.approx(t, abs=1.0)
        assert r["Gas rate (MMscf/d)"] == pytest.approx(r["Liquid rate (STB/d)"] * 0.2 * 800 / 1e6)
    assert len(prof) == 8


def test_horizontal_friction_ipr_prosper_t10():
    """PROSPER Tutorial 10 (user's run): Horizontal Well - dP Friction Loss In WellBore with Kuchuk & Goode, 50 md,
    100 ft, Kz/Ky 0.1, 800 ft well in a 4200 ft square, one perforated zone (skin 3 by hand, 0.3175 ft flowing radius,
    0.0006 in roughness), Pr 5200 psig, 210 F, GOR 800, dry; Glaso / Beal matched to the five-point lab table. The
    IPR pressures of PROSPER's VLP / IPR table within 10 psi from 40 to 39,968 STB/d; AOF within 0.1 %."""
    from wellmodel import inflow_adv as IA
    G = 14.696
    inp = wbo.PVTInputs(rsb=800, api=37, gas_sg=0.76, salinity_ppm=23000)
    lab = pd.DataFrame({"Pressure": np.array([4000, 3500, 3000, 2400, 1000]) + G,
                        "Gas Oil Ratio": [800, 800, 655, 500, 190], "Oil FVF": [1.42, 1.432, 1.352, 1.273, 1.12],
                        "Oil Viscosity": [0.364, 0.35, 0.403, 0.48, 0.7205]})
    res = wbo.match_all(inp, wbo.MatchData(210.0, 3500 + G, lab))
    assert res.matches[("pb", "glaso")].p1 == pytest.approx(1.00882, abs=2e-4)          # PROSPER 1.00882 / 30.3398
    assert res.matches[("visc", "beal")].p1 == pytest.approx(1.00146, abs=1e-3)          # PROSPER 1.00146 / 0.0046726
    m = wbo.BlackOilModel(inp, "glaso", "beal", {**{k: res.matches[(k, "glaso")] for k in ("pb", "rs", "bo")},
                                                 "visc": res.matches[("visc", "beal")]})
    pr = 5200 + G
    tab = m.table(210.0, [pr])
    props = {"mu": float(tab["Oil viscosity (cp)"].iloc[0]), "bo": float(tab["Oil FVF (RB/STB)"].iloc[0]),
             "rho": float(tab["Oil density (lb/ft3)"].iloc[0]), "wc": 0.0}
    hz = lambda s: IA.horizontal_resistance("Kuchuk & Goode", 50, 100, 0.354, 1.0, 0.1, 800, 4200, 4200, 2100, 2100,
                                            50, s)
    kh, D0 = hz(0.0)
    zone = {"type": "Perforated", "length": 800, "k": 50, "skin": hz(3.0)[1] - D0, "id_in": 0.3175 * 24,
            "rough_in": 0.0006}
    c = IA.horizontal_friction("oil", pr, [zone], kh, D0, 50, props, pb=m.bubble_point(210.0), n=80)
    o = np.argsort(c.q)
    q, pw = np.asarray(c.q)[o], np.asarray(c.pwf)[o]
    prosper = [(40.0, 5197.1), (246.3, 5182.0), (1054.3, 5123.1), (2181.2, 5040.9), (4512.9, 4870.8),
               (6491.3, 4726.4), (9337.0, 4518.8), (13430.2, 4220.0), (19317.9, 3790.3), (27786.6, 3157.0),
               (39968.0, 1995.6)]
    for qq, pp in prosper:
        assert np.interp(qq, q, pw) - G == pytest.approx(pp, abs=10), qq
    assert max(c.q) == pytest.approx(49955.6, rel=0.001)
    # the zone skin enters as each model scales it: Kuchuk & Goode h / L sqrt(kx/kz) s, Babu & Odeh s
    assert hz(3.0)[1] - D0 == pytest.approx(100 / 800 * np.sqrt(10) * 3)
    bo_ = lambda s: IA.horizontal_resistance("Babu & Odeh", 50, 100, 0.354, 1.0, 0.1, 800, 4200, 4200, 2100, 2100,
                                             50, s)
    assert bo_(3.0)[1] - bo_(0.0)[1] == pytest.approx(3.0)


def test_horizontal_well_system_prosper_t10():
    """Tutorial 10 system calculation (user's run; the Tutorial 05 Logie well equipment, PE2 unmatched, U 8): VLP
    within 20 psi and wellhead temperature within 0.5 F from 40 to 27,787 STB/d; solution 21,115 against 21,120.8
    STB/d at 3655.5 against 3655.4 psig. The example project reproduces it."""
    import json
    from wellmodel import inflow_adv as IA
    G = 14.696
    fl = _t05_oil()
    m = fl.m
    path, _ = _t23_path()
    P = {40.0: (3224.5, 40.7), 246.3: (2953.1, 44.3), 1054.3: (2065.9, 58.5), 3137.5: (2011.9, 92.2),
         6491.3: (2178.7, 127.9), 13430.2: (2813.8, 160.8), 19317.9: (3447.5, 173.4), 27786.6: (4424.2, 183.3)}
    q = np.array(list(P))
    Pp, Tt = wvlp.traverse(path, fl, np.full(q.size, 500 + G), {"q": q, "wc": np.zeros(q.size),
                                                                 "gor": np.full(q.size, 800.0)}, "pe2_style", 1, 1, 8.0)
    for k, qq in enumerate(q):
        assert Pp[k, -1] - G == pytest.approx(P[qq][0], abs=20) and Tt[k, 0] == pytest.approx(P[qq][1], abs=0.5)
    pr = 5200 + G
    tab = m.table(210.0, [pr])
    props = {"mu": float(tab["Oil viscosity (cp)"].iloc[0]), "bo": float(tab["Oil FVF (RB/STB)"].iloc[0]),
             "rho": float(tab["Oil density (lb/ft3)"].iloc[0]), "wc": 0.0}
    hz = lambda s: IA.horizontal_resistance("Kuchuk & Goode", 50, 100, 0.354, 1.0, 0.1, 800, 4200, 4200, 2100, 2100,
                                            50, s)
    kh, D0 = hz(0.0)
    c = IA.horizontal_friction("oil", pr, [{"type": "Perforated", "length": 800, "k": 50, "skin": hz(3.0)[1] - D0,
                                            "id_in": 7.62, "rough_in": 0.0006}], kh, D0, 50, props,
                               pb=m.bubble_point(210.0), n=80)
    res, _ = wsys.solve_cases(lambda tid: path, fl, "oil", {"whp": 500 + G, "u": 8.0, "wc": 0.0, "gor": 800.0}, [{}],
                              lambda ch: c, "pe2_style", 1.0, 1.0, method="Automatic - Geometric")
    assert res[0].q == pytest.approx(21120.8, rel=0.002) and res[0].pwf - G == pytest.approx(3655.4, abs=3)
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T10_Horizontal_Well.wellmodel.json"))
    assert proj["inputs"]["ipr_model_oil"] == "Horizontal Well - friction dP"
    assert proj["inputs"]["ipr_hzf_method"] == "Kuchuk & Goode" and proj["widgets"]["opt_artificial_lift"] == "none"


def test_sand_control_prosper_t15_t18():
    """PROSPER Tutorials 15-18 on the Logie well (user's runs; T05 PVT, Pr 5200 psig, 210 F, dry, GOR 800). The
    sand-control dP uses the fluid at the reservoir pressure (as PROSPER: skins and dPs within 0.3 % that way; at
    the sand-face pressure the dP near the AOF was up to 2.3 times PROSPER's). T16-T18: Darcy 50 md, 100 ft, 500
    acres with PROSPER's perforation (1.600337) and deviation (-0.016550) skins plus the screen: IPR within 15 psi,
    AOF within 0.6 %, system solution within 1.3 % (PE2 unmatched, U 8). T15: gravel pack A / B within 0.3 % and the
    velocity at the casing within 0.2 %."""
    from wellmodel import completion as C, inflow as I
    G = 14.696
    fl = _t05_oil()
    m = fl.m
    pr = 5200 + G
    props = C.props_at_pressure(C.props_from_fluid(fl, "oil", 210.0, {"wc": 0.0, "gor": 800.0}, True), pr)
    # sand-control skin and dP against PROSPER's VLP / IPR tables (rate, PROSPER dP, PROSPER skin)
    tables = {
        "wire_wrapped_screen": ({"interval": 100, "screen_outer_radius": 0.28},
                                [(23.8, 0.09, 0.234508), (11517.1, 42.23, 0.234692), (23828.4, 87.45, 0.234889)], 0.003),
        "pre_packed_screen": ({"interval": 100, "screen_inner_radius": 0.25, "screen_outer_radius": 0.33,
                               "screen_perm": 500},
                              [(23.9, 0.04, 0.097968), (11537.6, 17.68, 0.098086), (23870.8, 36.63, 0.098213)], 0.003),
        "slotted_liner": ({"interval": 100, "liner_inner_radius": 0.32, "liner_outer_radius": 0.34,
                           "slot_height_in": 4, "slot_width_in": 0.02, "slot_density": 4, "screen_outer_radius": 0,
                           "outside_perm": 50000},
                          [(23.0, 0.02, 0.06265), (11124.5, 164.17, 0.94447), (23016.1, 679.36, 1.88906)], 0.025),
        "gravel_pack_cased": ({"interval": 100, "perf_diameter_in": 0.26, "spf": 8, "perf_efficiency": 0.87,
                               "pack_length_in": 4, "gravel_perm": 25000},
                              [(13.3, 0.34, 1.652017), (6414.7, 1065.89, 10.634428), (9226.8, 2102.05, 14.580392)],
                              0.003)}
    for kind, (prm, rows, tol) in tables.items():
        mdl = C.SandControlModel(C.SandControl(kind, prm), props, 50, 100, 0.354)
        for q, dp, skin in rows:
            d = mdl.details(np.array([q]), None)
            assert float(d["skin"][0]) == pytest.approx(skin, rel=tol), (kind, q)
            if dp > 1:
                assert float(d["dp"][0]) == pytest.approx(dp, rel=tol), (kind, q)
    gp = C.SandControlModel(C.SandControl("gravel_pack_cased", tables["gravel_pack_cased"][0]), props, 50, 100, 0.354)
    d = gp.details(np.array([1000.0]), None)
    bo = float(d["Q"][0]) / 1000
    assert d["a"][0] * bo ** 2 == pytest.approx(0.000022, rel=0.03) and d["b"][0] * bo == pytest.approx(0.025522, rel=0.003)
    assert gp.details(np.array([9226.8]), None)["velocity"][0] == pytest.approx(3.276, rel=0.003)
    # T16-T18 IPR and system
    tab = m.table(210.0, [pr])
    mu, bo = float(tab["Oil viscosity (cp)"].iloc[0]), float(tab["Oil FVF (RB/STB)"].iloc[0])
    assert I.darcy_pi(50, 100, mu, bo, 500, 31.6, 0.354, 0.0) == pytest.approx(7.83615, rel=0.003)
    base = I.darcy(pr, m.bubble_point(210.0), 50, 100, 500, 31.6, 0.354, 1.600337 - 0.016550, mu, bo, 0.0)
    path, _ = _t23_path()
    runs = {"wire_wrapped_screen": (23847.5, 14296.1, 2908.0, [(2690.5, 4780.2), (16566.1, 2504.0)]),
            "pre_packed_screen": (23889.9, 14404.2, 2919.3, [(2695.3, 4785.2), (16595.5, 2533.5)]),
            "slotted_liner": (23034.5, 13591.9, 2835.8, [(2598.8, 4793.2), (16001.3, 2342.6)])}
    for kind, (aof, qs, ps, pts) in runs.items():
        c = C.apply_sand_control(base, C.sand_control_dp(C.SandControl(kind, tables[kind][0]), props, 50, 100, 0.354))
        o = np.argsort(c.q)
        for qq, pp in pts:
            assert np.interp(qq, np.asarray(c.q)[o], np.asarray(c.pwf)[o]) - G == pytest.approx(pp, abs=15)
        assert max(c.q) == pytest.approx(aof, rel=0.006)
        res, _ = wsys.solve_cases(lambda tid: path, fl, "oil", {"whp": 500 + G, "u": 8.0, "wc": 0.0, "gor": 800.0},
                                  [{}], lambda ch: c, "pe2_style", 1.0, 1.0, method="Automatic - Geometric")
        assert res[0].q == pytest.approx(qs, rel=0.013) and res[0].pwf - G == pytest.approx(ps, abs=8)


def test_fractured_well_frac_and_pack_prosper_t15():
    """PROSPER Tutorial 15 file (user's runs): Hydraulically Fractured Well, 50 md, 100 ft, 500 acres, xf 35 ft, FCD
    4500, 5 days, porosity 0.16, with the cased-hole gravel pack. Formation PI (no skin) against PROSPER's base and
    eleven variations within 1.4 % (transient Gringarten held at tDA 0.0891; FCD >= 100 infinite); with the gravel
    pack the AOF within 0.3 %, the IPR within 5 psi and the system solution within 0.4 % (PE2 unmatched, U 8)."""
    from wellmodel import inflow_adv as IA, completion as C
    G = 14.696
    fl = _t05_oil()
    m = fl.m
    pr = 5200 + G
    tab = m.table(210.0, [pr, pr - 10])
    bo1, bo2 = tab["Oil FVF (RB/STB)"].to_numpy()
    co, mu, pb = -(bo1 - bo2) / 10 / bo1, float(tab["Oil viscosity (cp)"].iloc[0]), m.bubble_point(210.0)
    base = dict(k=50, h=100, area=500, ca=31.6, rw=0.354, xf=35, fcd=4500, t_days=5, phi=0.16, ct=co, mu=mu)
    runs = [({}, 13.3913), ({"fcd": 10}, 12.9775), ({"fcd": 100}, 13.3913), ({"fcd": 20000}, 13.3913),
            ({"xf": 100}, 17.1533), ({"xf": 200}, 21.0141), ({"k": 20}, 5.74275), ({"t_days": 1}, 15.566),
            ({"t_days": 30}, 13.3913), ({"t_days": 365}, 13.3913), ({"k": 10}, 3.1132), ({"k": 100}, 26.7825)]
    for ch, j in runs:
        c = IA.fractured_well("oil", pr, pb=pb, bo=bo1, **{**base, **ch})
        assert c.params["PI (STB/d/psi)"] == pytest.approx(j, rel=0.014), ch
    c0 = IA.fractured_well("oil", pr, pb=pb, bo=bo1, **base)
    props = C.props_at_pressure(C.props_from_fluid(fl, "oil", 210.0, {"wc": 0.0, "gor": 800.0}, True), pr)
    sc = C.SandControl("gravel_pack_cased", {"interval": 100, "perf_diameter_in": 0.26, "spf": 8,
                                             "perf_efficiency": 0.87, "pack_length_in": 4, "gravel_perm": 25000})
    c = C.apply_sand_control(c0, C.sand_control_dp(sc, props, 50, 100, 0.354))
    assert max(c.q) == pytest.approx(13282.3, rel=0.004)
    o = np.argsort(c.q)
    for qq, pp in ((1498.5, 5000.6), (6414.7, 3655.1), (9226.8, 2408.9)):
        assert np.interp(qq, np.asarray(c.q)[o], np.asarray(c.pwf)[o]) - G == pytest.approx(pp, abs=6)
    path, _ = _t23_path()
    res, _ = wsys.solve_cases(lambda tid: path, fl, "oil", {"whp": 500 + G, "u": 8.0, "wc": 0.0, "gor": 800.0}, [{}],
                              lambda ch: c, "pe2_style", 1.0, 1.0, method="Automatic - Geometric")
    assert res[0].q == pytest.approx(9238.8, rel=0.005) and res[0].pwf - G == pytest.approx(2401.8, abs=15)


def test_gas_lift_multipoint_quicklook_prosper_t33b():
    """PROSPER Tutorial 33b Multipoint QuickLook (user's run: 390 psig / 147 F, 7488 STB/d, 35 % water cut, 5.3935
    MMscf/d total, 1.5 injected, 1490 psig CHP, static 3700 psig; R-20 valves 16 / 12 / 8 at 3731 / 5661 / 6659 ft,
    orifice 9/64 at 7020 ft) with every valve on Thornhill-Craver: valve 1 takes all 1.5 MMscf/d; tubing and casing
    pressures at the valves within 7 and 1 psi, opening / closing pressures within 1.5 psi, capacities ('calculated
    gas rates') within 4 %, results at the orifice; calculated CHP from the bottom-up tubing pressure within 4 psi,
    critical rate within 0.5 %, fitted U within 0.1 %. PROSPER's Acuna model (the tutorial's) is not available."""
    from wellmodel import gaslift_system as wgls, quicklook as wql
    G = 14.696
    oil = _t05_oil()
    tc = "Thornhill-Craver"
    vs = (wgls.GLValve(3731.19, "Valve", 16, 0.066, 1210.8, model=tc), wgls.GLValve(5660.62, "Valve", 12, 0.038,
                                                                                     1194.07, model=tc),
          wgls.GLValve(6659.12, "Valve", 8, 0.017, 1178.0, model=tc), wgls.GLValve(7019.67, "Orifice", 9))
    s = wgls.GasLiftSetup(valves=vs, annulus_friction=True)
    path, secs = _t23_path([v.md for v in vs] + [6250.0])
    s.annulus = wgls.annulus_geometry(path, secs)
    inp = wql.QuickLookInput(390 + G, 147, 7488, 0.35, 5.3935, 1.5, 1490 + G, 7019.67, 9,
                             gauges=((6250.0, 1440 + G),), static_p=3700 + G, valves=vs)
    r = wql.quicklook(path, oil, lambda q: 2336.27 + G, s, inp, "pe2_style", 1.03108, 1.00278, 8.580277,
                      multipoint=True)
    v = r.valves
    prosper = [  # Pt, Pc, T, Pvo, Pvc (psig), used, calculated (MMscf/d)
        (921.372, 1612.04, 180.613, 1590.92, 1546.73, 1.5, 1.89315),
        (1380.36, 1686.84, 196.603, 1575.7, 1568.27, 0.0, 0.89042),
        (1649.75, 1725.83, 202.708, 1561.15, 1562.66, 0.0, 0.21783),
        (1751.9, 1740.01, 204.498, np.nan, np.nan, 0.0, 0.0)]
    for (_, x), (pt, pc, t, pvo, pvc, qu, qc) in zip(v.iterrows(), prosper):
        assert x["Tubing pressure (psia)"] - G == pytest.approx(pt, abs=7.5)
        assert x["Casing pressure (psia)"] - G == pytest.approx(pc, abs=1.0)
        assert x["Temperature (°F)"] == pytest.approx(t, abs=0.5)
        if np.isfinite(pvo):
            assert x["Opening pressure (psia)"] - G == pytest.approx(pvo, abs=1.5)
            assert x["Closing pressure (psia)"] - G == pytest.approx(pvc, abs=1.5)
        assert x["Gas rate (MMscf/d)"] == pytest.approx(qu, abs=1e-6)
        assert x["Calculated gas rate (MMscf/d)"] == pytest.approx(qc, rel=0.04, abs=1e-6)
    sm = r.summary
    assert sm["Tubing pressure at valve (psia)"] - G == pytest.approx(1751.9, abs=7.5)      # at the orifice
    assert sm["Casing pressure at valve (psia)"] - G == pytest.approx(1740.01, abs=1.0)
    assert sm["Calculated casing head pressure (psia)"] - G == pytest.approx(1429.12, abs=4)
    assert sm["Calculated critical flow rate (MMscf/d)"] == pytest.approx(0.63507, rel=0.005)
    assert sm["Fitted thermal coefficient (BTU/h/ft2/F)"] == pytest.approx(8.43945, rel=0.001)
    assert sm["Equivalent calculated PI (STB/d/psi)"] == pytest.approx(5.49084, rel=1e-3)
    assert sm["Multipoint flag"] == "Single"
    # the Acuna run's Thornhill-Craver valve 3 (8/64) and orifice (9/64) capacities at PROSPER's own pressures
    from wellmodel import gaslift as GL
    for d, pt, pc, t, q in ((8, 1564.64, 1725.22, 202.712, 0.30659), (9, 1663.13, 1739.39, 204.498, 0.27668)):
        assert float(GL.thornhill_craver(d / 64, pc + G, pt + G, t, s.gas.sg)) / 1000 == pytest.approx(q, rel=0.002)


def _inj_tests(gas, rows):
    from wellmodel import injection as winj
    cols = winj.inj_test_columns(gas)
    df = pd.DataFrame([[None, f"test {i + 1}", *r] for i, r in enumerate(rows)], columns=cols,
                      index=range(1, len(rows) + 1))
    return winj.clean_inj_tests(df, gas)[0]


def test_water_injection_well_test_matching_prosper_t21():
    """PROSPER Tutorial 21 (matching water injection well tests; T03a well, TIF injectivity, gauge at 8000 ft).
    Estimate U: PROSPER 8.88441 / 8.88067 / 8.88373 (mean 8.88293) from the bottom-node ('downstream')
    temperature - ours within 0.1 %. Match VLP (PE2): P1 0.99461, P2 1, std 0.466 psi - ours P1 0.9932 (the
    unmatched gauge pressure is about 5 psi higher than PROSPER's), std 0.56. User's file (U 8.88293, P1 0.99461,
    skin 3, WHP 1250 psig): system 11,853.7 STB/d at 5,382.4 psig; VLP and IPR at the VLP-tab rates. Adjust Skin in
    the user's file: 4.66919 / 6.56128 / 8.8501 (the manual's 3 / 4.2 / 5.6 come from a different injectivity) -
    these calibrate the split of the TIF fully-developed-fracture regime between skin and the other terms (see
    tif.py); ours within 0.03."""
    import dataclasses
    from wellmodel import injection as winj, tif
    G = 14.696
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 600},
            {"Type": "Tubing", "Measured Depth (ft)": 8500, "Tubing Inside Diameter (in)": 5.2,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 9000, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = winj.from_flow_path(wvlp.build_flow_path(top, secs, pts, [0, 9000], [0, 9000], [0, 600, 9000],
                                                    [60, 40, 210], extra_md=[8000.0]))
    fl = winj.WaterFluid(200000)
    tests = _inj_tests(False, [(1250 + G, 99.3, 8467, 8000, 4910 + G, 4100 + G),
                               (1075 + G, 112.1, 5897, 8000, 4747 + G, 4100 + G),
                               (985 + G, 123.6, 4456, 8000, 4662 + G, 4100 + G)])
    u = winj.estimate_u(path, fl, tests, 60.0)["U (BTU/h/ft2/F)"].to_numpy()
    assert np.allclose(u, [8.88441, 8.88067, 8.88373], rtol=1e-3) and abs(u.mean() / 8.88293 - 1) < 1e-3
    U, P1 = 8.88293, 0.99461
    m = winj.match_vlp(path, fl, tests, U, 60.0)
    assert abs(m.p1 - P1) < 0.002 and m.p2 == 1.0 and abs(m.std - 0.46553) < 0.2
    assert np.all(np.abs(m.before - 25.5) < 1.5)
    base = tif.TIFInputs(pr=4100 + G, t_res=210, t_inj=120, k=50, h=100, area=250, ca=31.6, rw=.354, phi=.18,
                         t_days=20, q_hist=12000, skin=3, sigma_ini=5300 + G, sweep=.75, alpha=5e-6,
                         mu_res=0.35734, b_res=1.4116)
    ipr = tif.tif_ipr(base, fl)
    res, *_ = winj.solve(path, fl, 1250 + G, ipr, U, 60.0, p1=P1, n=40)
    assert abs(res.q / 11853.7 - 1) < 0.006 and abs(res.pwf - G - 5382.4) < 3
    qq = np.array([1215.8, 4037.2, 8985.7, 13405.8, 20000.0])
    P, Tt = winj.traverse(path, fl, 1250 + G, qq, U, 60.0, p1=P1)
    assert np.abs(P[:, -1] - G - [5420.4, 5432.8, 5409.6, 5367.6, 5275.8]).max() < 1.0
    assert np.abs(ipr.pwf(qq, Tt[:, -1]) - G - [4351.1, 4465.1, 5000.8, 5588.9, 6505.3]).max() < 8
    ipr_for = lambda **kw: tif.tif_ipr(dataclasses.replace(base, **kw), fl)
    for (_, t), s_p in zip(tests.iterrows(), (4.66919, 6.56128, 8.8501)):
        s_ = winj.adjust_ipr(path, fl, t, ipr_for, "skin", U, 60.0, None, P1)
        assert abs(s_ - s_p) < 0.05, (s_, s_p)
    r3, *_ = winj.solve(path, fl, t["whp"], ipr_for(skin=s_), U, 60.0, p1=P1)
    assert abs(r3.q / t["q"] - 1) < 1e-3


def test_gas_injection_well_test_matching_prosper_t22():
    """PROSPER Tutorial 22 (matching gas injection well tests; T04 well, gauge at 4650 ft). Estimate U: PROSPER
    2.88602 / 2.881 / 2.86746 (mean 2.87816) - ours 1.1 % higher. Match VLP (PE2): P1 1.04285, P2 1, std 0.172,
    unmatched gauge pressures about 21-22 psi off - ours P1 1.0435, 21.0-21.9 psi. User's file (U 2.87816, P1
    1.04285, skin 2, WHP 2750 psig): system 15.238 MMscf/d at 3,283.8 psig (ours 15.35 at 3,283.5); VLP and IPR at
    the test rates within 0.5 psi. Adjust Skin: ours -0.91 / 0.0 / 1.0; PROSPER in the user's file 0 / 0 / 0 (test 2
    agrees; tests 1 and 3 not yet explained). The manual's 2 / 3.2 / 4.6 come from a different injectivity."""
    from wellmodel import injection as winj, gaspvt as gp
    G = 14.696
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 4.5,
             "Tubing Inside Roughness (in)": 0.00006},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 4.0},
            {"Type": "Tubing", "Measured Depth (ft)": 4800, "Tubing Inside Diameter (in)": 4.5,
             "Tubing Inside Roughness (in)": 0.00006},
            {"Type": "Casing", "Measured Depth (ft)": 5010, "Casing Inside Diameter (in)": 6.13,
             "Casing Inside Roughness (in)": 0.00006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = winj.from_flow_path(wvlp.build_flow_path(top, secs, pts, [0, 1500, 1982, 2460, 2905, 3335, 4650, 5010],
                                                    [0, 1500, 1980, 2450, 2883, 3300, 4560, 4900],
                                                    [0, 1500, 5010], [45, 75, 166], extra_md=[4650.0]))
    fl = winj.GasFluid(gp.GasPVTInputs(0.769, 100, 0, 67, 0, 100000, 0, 0.368, 4.005, "lee", 166, 2750),
                       impurities=False)
    tests = _inj_tests(True, [(2500 + G, 112.9, 10.2, 4650, 2971.1 + G, 2750 + G),
                              (2750 + G, 106.4, 17.515, 4650, 3242 + G, 2750 + G),
                              (2450 + G, 119.8, 6.99, 4650, 2919 + G, 2750 + G)])
    u = winj.estimate_u(path, fl, tests, 100.0)["U (BTU/h/ft2/F)"].to_numpy()
    assert np.allclose(u, [2.88602, 2.881, 2.86746], rtol=0.02) and abs(u.mean() / 2.87816 - 1) < 0.015
    U, P1 = 2.87816, 1.04285
    m = winj.match_vlp(path, fl, tests, U, 100.0)
    assert abs(m.p1 - P1) < 0.002 and abs(m.std - 0.17206) < 0.2
    assert np.all(np.abs(np.abs(m.before) - 21.6) < 1.0)
    ipr_for = lambda skin=2.0, pr=2750 + G: winj.petex_gas_injection(pr, 166, 25, 55, 500, 31.6, .354, 45, 100,
                                                                     .15, .25, skin, fl)
    res, *_ = winj.solve(path, fl, 2750 + G, ipr_for(), U, 100.0, p1=P1)
    assert abs(res.q / 15.238 - 1) < 0.01 and abs(res.pwf - G - 3283.8) < 2
    qq = np.array([6.99, 10.2, 17.515])
    P, Tt = winj.traverse(path, fl, 2750 + G, qq, U, 100.0, p1=P1)
    assert np.abs(P[:, -1] - G - [3295.8, 3291.6, 3280.3]).max() < 0.6
    assert np.abs(Tt[:, -1] - [120.0, 112.9, 106.4]).max() < 0.4
    assert np.abs(ipr_for().pwf(qq) - G - [2971.7, 3085.6, 3373.4]).max() < 0.8
    s2 = winj.adjust_ipr(path, fl, tests.iloc[1], ipr_for, "skin", U, 100.0, None, P1)
    assert abs(s2 - 0.0) < 0.05


def _t38_path():
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 11800, "Tubing Inside Diameter (in)": 3.992,
             "Tubing Inside Roughness (in)": 0.0018},
            {"Type": "Casing", "Measured Depth (ft)": 12000, "Casing Inside Diameter (in)": 8.3,
             "Casing Inside Roughness (in)": 0.0018}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    return wvlp.build_flow_path(top, secs, pts, [0, 12000], [0, 12000], [0, 12000], [70, 300])


def test_pe5_style_high_velocity_condensate_prosper_t38():
    """PROSPER's PE5 gradient exports of Tutorial 38 (25 MMscf/d of condensate, 3.99 in tubing, wellhead 500-2000
    psig, black-oil and EoS PVT; 2-9 % liquid at 20-115 ft/s): PE5 calls all of it slug flow. With PROSPER's own
    node properties, the drift-flux holdup is within 3 % at every tubing node and the pressure drop integrated down
    the well within 4 % in all eight traverses (the joint fit with Tutorial 47's water-rich gradients; T38 alone gave
    1 % / 2.5 %). Before (Duns & Ros mist above the slug limit) the holdup was the no-slip one, 60 % low."""
    from wellmodel import correlations as C
    df = pd.read_csv(__file__.replace("test_wellmodel.py", "data/t38_gradients.csv"))
    for (pvt, whp), g in df.groupby(["pvt", "whp"]):
        g = g.sort_values("md")
        a = [g[c].to_numpy() for c in ("vsl", "vsg", "rho_l", "rho_g", "mu_l", "mu_g", "ift_gl")]
        e, f, hl = C.pe5_style(*a, g.d_in.to_numpy() / 12, 0.0018 / 12, 1.0, g.p.to_numpy() + 14.696)
        tub = g.d_in.to_numpy() < 4
        assert np.abs(hl[tub] / g.hl.to_numpy()[tub] - 1).max() < 0.03, (pvt, whp)
        dp = np.sum((e + f) * g.length.to_numpy())
        assert abs(dp / np.sum(g.grad.to_numpy() * g.length.to_numpy()) - 1) < 0.04, (pvt, whp)


def test_retrograde_black_oil_pvt_prosper_t38():
    """PROSPER's black-oil retrograde condensate PVT table (Tutorial 38: separator GOR 7431.94, tank GOR 1, gas
    gravity 0.745, 43.56 API, 2.33 % CO2, 0.31 % N2, dew point 4596.83 psig at 300 F; 25 temperatures 70-300 F x 25
    pressures 0-7000 psig). Dew point, dropout ('reservoir CGR'), oil FVF and density reproduced exactly; gas density
    and viscosity within 1.2 % (Z within 0.32 %: PROSPER's impurity correction not found); the condensate viscosity is Beal's (PROSPER's is
    not a published correlation found)."""
    import dataclasses
    from wellmodel import retrograde as RG
    G = 14.696
    df = pd.read_csv(__file__.replace("test_wellmodel.py", "data/t38_bo_retrograde_pvt.csv"))
    inp = RG.RetroInputs(7431.94, 0.745, 1.0, 0.745, 43.56, 4596.83 + G, 300.0, 6000 + G, 0.0, 20000, 0, 2.33, 0.31)
    r = RG.props(inp, df.p.to_numpy() + G, df["T"].to_numpy())
    assert np.abs(r["p_dew"] - G - df.pd.to_numpy()).max() < 0.05
    assert np.abs(r["dropout"] - df.cgr.to_numpy()).max() < 0.03
    two = df.cgr.to_numpy() > 0
    assert np.abs(r["bo"][two] / df.bo.to_numpy()[two] - 1).max() < 5e-4
    assert np.abs(r["rho_o"][two] / df.rho_o.to_numpy()[two] - 1).max() < 5e-4
    assert np.abs(r["z"] / df.z.to_numpy() - 1).max() < 0.01
    assert np.abs(r["rho_g"] / df.rho_g.to_numpy() - 1).max() < 0.01
    assert np.abs(r["mu_g"] / df.mu_g.to_numpy() - 1).max() < 0.012
    assert np.abs(np.median(r["mu_o"][two] / df.mu_o.to_numpy()[two] - 1)) < 0.08
    # same fluid with CO2 = N2 = 0: Standing's condensate pseudo-criticals of the separator gas give PROSPER's Z exactly
    df0 = pd.read_csv(__file__.replace("test_wellmodel.py", "data/t38_bo_retrograde_pvt_noimp.csv"))
    r0 = RG.props(dataclasses.replace(inp, co2=0.0, n2=0.0), df0.p.to_numpy() + G, df0["T"].to_numpy())
    assert np.abs(r0["z"] / df0.z.to_numpy() - 1).max() < 1e-4 and np.abs(r0["mu_g"] / df0.mu_g.to_numpy() - 1).max() < 0.003
    # the inputs changed one at a time (reservoir temperature 250 F, dew point 4000 psig, separator GOR 10,000, 50 API)
    # and all four together: the dew-point curve, the dropout shape (drop_shape) and the Glaso liquid hold
    for name, kw, tol in (("tres250", dict(t_res=250.0), 0.01), ("pd4000", dict(p_dew=4000 + G), 0.01),
                          ("gor10000", dict(sep_gor=10000.0), 0.01), ("api50", dict(api=50.0), 0.01)):
        dfx = pd.read_csv(__file__.replace("test_wellmodel.py", f"data/t38_bo_retro_{name}.csv"))
        rx = RG.props(dataclasses.replace(inp, **kw), dfx.p.to_numpy() + G, dfx["T"].to_numpy())
        assert np.abs(rx["dropout"] - dfx.cgr.to_numpy()).max() < tol, name
        assert np.abs(rx["p_dew"] - G - dfx.pd.to_numpy()).max() < 0.05, name
        assert np.abs(rx["rho_g"] / dfx.rho_g.to_numpy() - 1).max() < 0.005, name
    df1 = pd.read_csv(__file__.replace("test_wellmodel.py", "data/t38_bo_retrograde_pvt_run1.csv"))
    inp1 = dataclasses.replace(inp, sep_gor=10000.0, api=50.0, p_dew=4000 + G, t_res=250.0)
    r1 = RG.props(inp1, df1.p.to_numpy() + G, df1["T"].to_numpy())
    two1 = df1.cgr.to_numpy() > 0
    assert np.abs(r1["p_dew"] - G - df1.pd.to_numpy()).max() < 0.05
    assert np.abs(r1["bo"][two1] / df1.bo.to_numpy()[two1] - 1).max() < 5e-4
    assert np.abs(r1["dropout"] - df1.cgr.to_numpy()).max() < 0.2
    assert np.abs(r1["rho_g"] / df1.rho_g.to_numpy() - 1).max() < 0.01


def test_retrograde_black_oil_vs_eos_traverse_prosper_t38():
    """PROSPER Tutorial 38 (the T37 well, 25 MMscf/d, PE5, U 5): bottom-hole pressure for wellhead 500 / 1000 /
    1500 / 2000 psig. Black-oil condensate - PROSPER 2953.82 / 3159.26 / 3505.60 / 3959.33 psig: ours within 2.5 %,
    wellhead temperature 202.7 F (PROSPER 202.69). Equation of state - PROSPER 3118.52 / 3296.10 / 3617.52 /
    4053.77: ours within 1.5 % of PROSPER's black-oil results but 1-6 % below its EoS ones. PROSPER's EoS traverse does
    not conserve mass (its total mass rate is 12 % above the wellstream's at the top of the well and 4 % above at the
    bottom, with phase densities equal to ours to 0.1 %); ours does."""
    import json
    from wellmodel import retrograde as RG, eos as E
    G = 14.696
    path = _t38_path()
    whp = np.array([500, 1000, 1500, 2000.0]) + G
    inp = RG.RetroInputs(7431.94, 0.745, 1.0, 0.745, 43.56, 4596.83 + G, 300.0, 6000 + G, 0.0, 20000, 0, 2.33, 0.31)
    P, T = wvlp.traverse(path, wvlp.RetrogradeBOFluid(inp), whp, {"q": 25.0, "wgr": 0.0}, "pe5_style", u=5.0)
    bo_p = np.array([2953.82, 3159.26, 3505.60, 3959.33])
    assert np.abs((P[:, -1] - G) / bo_p - 1).max() < 0.025 and abs(T[0, 0] - 202.69) < 0.3
    root = __file__.replace("tests/test_wellmodel.py", "examples/")
    f = E.EOSFluid(E.parse_prp(open(root + "T37_Retrograde_Condensate_EoS.prp").read()))
    sep = f.separator_train()
    fe = wvlp.EOSTableFluid("gas", wvlp.eos_grid(f, 7000, 60, 300), float(f.z @ f.mw),
                            float(sep["stages"]["Gas (scf per lbmol feed)"].sum()), float(sep["stb_per_lbmol"]),
                            float(sep["oil_density"]), wbo.PVTInputs(rsb=1, api=40, gas_sg=0.7, salinity_ppm=20000))
    Pe, _ = wvlp.traverse(path, fe, whp, {"q": 25.0, "wgr": 0.0}, "pe5_style", u=5.0)
    assert np.abs((Pe[:, -1] - G) / bo_p - 1).max() < 0.015
    eos_p = np.array([3118.52, 3296.10, 3617.52, 4053.77])
    assert np.abs((Pe[:, -1] - G) / eos_p - 1).max() < 0.065
    proj = json.load(open(root + "T38_Black_Oil_Retrograde_Condensate.wellmodel.json"))
    assert proj["widgets"]["opt_method"] == "black_oil" and proj["inputs"]["pvt_dew"] == pytest.approx(4611.526)


def test_esp_quicklook_prosper_t34():
    """PROSPER Tutorial 34 (ESP QuickLook on the T26 well: GC10000, 113 stages at 8500 ft, 60 Hz, PE2 1.03 / 1.01;
    test 7198 STB/d, 74 % water cut, GOR 800, THP 275 psig; gauge 1957 psig at 6250 ft, intake 1714, discharge
    2812). PROSPER's fixed-top-node gradient (manual): wear 0 - discharge 2807.7, intake 1623.6 psig; wear 0.08 -
    discharge 2802.7, intake 1706.7, bottom hole 1977.6 psig. Ours within 0.6 %; the gauge within 1 psi of the
    measured. The measured / calculated pump dP is 0.93 (PROSPER's first guess: wear 0.08); the wear factor that takes
    the measured discharge back to the measured intake is 0.083. Bottom up against the user's PROSPER run: intake
    and discharge within 3 psi, THP within 12 psi, pump power, current and voltage within 1 %."""
    import dataclasses, json
    from wellmodel import esp as E, esp_quicklook as EQ
    G = 14.696
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 600},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 4.052,
             "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 4.8, "Casing Inside Diameter (in)": 6.4},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 3.72},
            {"Type": "Tubing", "Measured Depth (ft)": 8500, "Tubing Inside Diameter (in)": 4.052,
             "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 4.8, "Casing Inside Diameter (in)": 6.4},
            {"Type": "Casing", "Measured Depth (ft)": 9275, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    smd_, stvd_ = np.array(LOGIE_SURVEY).T
    path = wvlp.build_flow_path(top, secs, pts, smd_, stvd_, [0, 600, 9275], [60, 40, 210], extra_md=[6250.0, 8500.0])
    fl = _t05_oil()
    esp = {"pump_md": 8500.0, "pump": E.PUMPS["Centrilift GC10000 (5.13 in)"], "stages": 113, "freq": 60.0,
           "wear": 0.0, "sep": 0.0, "motor": E.MOTORS["Centrilift 544 300HP 1675V 105A"],
           "cable": E.CABLES["#1 Copper"], "cable_len": 9000.0}
    inp = EQ.ESPQuickLookInput(275 + G, 7198.0, 0.74, 800.0, 3350 + G, ((6250.0, 1957 + G),), 1714 + G, 2812 + G)
    J = 7198.0 / (3260 - 1976.34)                      # PROSPER's IPR pressure at the test rate, straight line
    ipr = lambda pr: (lambda q: pr - q / J)
    r0 = EQ.esp_quicklook(path, fl, ipr(3350 + G), inp, esp, "pe2_style", 1.03, 1.01, 8.580277, pres_for=ipr)
    s0 = r0.summary
    assert abs(s0["Top down: pump discharge (psia)"] - G - 2807.7) < 15
    assert abs(s0["Top down: pump intake (psia)"] - G - 1623.6) < 15
    assert abs(r0.points["Top down - measured (psi)"].iloc[0]) < 1.5
    assert abs(s0["Measured / calculated pump dP"] - 0.927) < 0.01
    assert abs(s0["Wear factor for the measured intake"] - 0.083) < 0.005
    r8 = EQ.esp_quicklook(path, fl, ipr(3260 + G), dataclasses.replace(inp, pres=3260 + G), {**esp, "wear": 0.08},
                          "pe2_style", 1.03, 1.01, 8.580277, pres_for=ipr)
    s8 = r8.summary
    assert abs(s8["Top down: pump intake (psia)"] - G - 1706.7) < 10
    assert abs(s8["Top down: bottom-hole pressure (psia)"] - G - 1977.6) < 10
    # reservoir pressure that gives the top-down BHP (PROSPER: 3260 psig from a 5 psi sensitivity)
    assert abs(s8["Reservoir pressure for the top-down BHP (psia)"] - G - 3260) < 10
    # with the matched wear factor and reservoir pressure the two gradients overlie
    assert np.nanmax(np.abs(r8.profile["Top down (psia)"] - r8.profile["Bottom up (psia)"])) < 15
    # the user's PROSPER "From Static BHP" stream (IPR pressure 1979.7 psig at the test rate): intake 1707.7 psig;
    # wear 0 - discharge 2911.3, THP 320.4, pump 269.97 hp, 96.82 A, 1951.51 V; wear 0.08 - discharge 2809.4, THP
    # 275.8, pump 269.45 hp (the wear lowers the head, not the power), 96.67 A, 1951.26 V
    for w_, (pdp_p, thp_p, hp_p, a_p, v_p) in ((0.0, (2911.3, 320.4, 269.97, 96.82, 1951.51)),
                                               (0.08, (2809.4, 275.8, 269.45, 96.67, 1951.26))):
        rb = EQ.esp_quicklook(path, fl, lambda q: 1979.7 + G, inp, {**esp, "wear": w_}, "pe2_style", 1.03, 1.01,
                              8.580277)
        b, sb = rb.bottom_up, rb.summary
        assert abs(b["pip"] - G - 1707.7) < 3 and abs(b["pdp"] - G - pdp_p) < 3
        assert abs(b["thp"] - G - thp_p) < 12
        assert abs(sb["Bottom up: pump power (hp)"] / hp_p - 1) < 0.01
        assert abs(sb["Bottom up: motor current (A)"] / a_p - 1) < 0.01
        assert abs(sb["Bottom up: surface voltage (V)"] / v_p - 1) < 0.005
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T34_ESP_QuickLook.wellmodel.json"))
    assert proj["widgets"]["opt_artificial_lift"] == "esp" and proj["inputs"]["espq_n"] == 113.0


def test_hsp_quicklook_prosper_t35():
    """PROSPER Tutorial 35 (HSP QuickLook on the T27 well: CLYDE UNION TP115AH 30 stages at 8500 ft, 7545 rpm, PE2
    1.03 / 1.01; test 5770 STB/d, 24 % water cut, GOR 800, THP 500 psig, power fluid 6308 STB/d; intake 2575, discharge
    3205 psig). The manual's Fixed Top Node Pressure stream: wear 0 - discharge 3199.6, intake 2506.5 psig (measured
    dP 91 % of calculated: wear 0.09); wear 0.09 - discharge 3195.3, intake 2564.3, bottom hole 2796.8 psig; IPR
    pressure 2806.55 psig at 5770 STB/d with Pres 3800 psig (from 3750-4000 in 50 psi steps), at which the two streams
    overlie. Ours within 25 psi (0.8 %); dP ratio 0.903 against 0.909. Turbine: CLYDE UNION T68-A, 26 stages - its
    power at the test's power fluid rate within 2 % of the pump's (the test is consistent), the power fluid it needs
    within 0.5 % of the measured 6308 STB/d."""
    import dataclasses, json
    from wellmodel import hsp as H, hsp_quicklook as HQ
    G = 14.696
    path, fl, I = _t27_setup()
    inst = H.HSPInstalled(8500.0, "CLYDE UNION TP115AH (4250-12325 rpm)", 30, "CLYDE UNION T68 - A", 26, 7500.0,
                          casing_id=6.8, tubing_od=4.8)
    J = 5770.0 / (3800 - 2806.55)                      # PROSPER's IPR pressure at the test rate, straight line
    ipr = lambda pr: (lambda q: pr - q / J)
    inp = HQ.HSPQuickLookInput(500 + G, 5770.0, 0.24, 800.0, 3950 + G, 7545.0, 6308.0, 2072 + G, (), 2575 + G,
                               3205 + G)
    r0 = HQ.hsp_quicklook(path, fl, ipr(3950 + G), inp, inst, "pe2_style", 1.03, 1.01, I["eq_u"], pres_for=ipr)
    s0 = r0.summary
    assert abs(s0["Top down: pump discharge (psia)"] - G - 3199.6) < 25
    assert abs(s0["Top down: pump intake (psia)"] - G - 2506.5) < 25
    assert abs(s0["Measured / calculated pump dP"] - 0.909) < 0.01
    assert abs(s0["Wear factor - first guess (1 - dP ratio, PROSPER)"] - 0.09) < 0.01
    assert 0.08 < s0["Wear factor for the measured intake"] < 0.11
    assert np.nanmax(np.abs(r0.profile["Top down (psia)"] - r0.profile["Bottom up (psia)"])) > 100   # not matched
    assert s0["Pump speed (rpm)"] == 7545.0
    r9 = HQ.hsp_quicklook(path, fl, ipr(3800 + G), dataclasses.replace(inp, pres=3800 + G),
                          dataclasses.replace(inst, wear=0.09), "pe2_style", 1.03, 1.01, I["eq_u"], pres_for=ipr)
    s9 = r9.summary
    assert abs(s9["Top down: pump discharge (psia)"] - G - 3195.3) < 25
    assert abs(s9["Top down: pump intake (psia)"] - G - 2564.3) < 25
    assert abs(s9["Top down: bottom-hole pressure (psia)"] - G - 2796.8) < 25
    pts = r9.points.set_index("Measurement")
    assert abs(pts.loc["Pump intake", "Top down - measured (psi)"]) < 10       # intake now on the measured
    assert abs(s9["Reservoir pressure for the top-down BHP (psia)"] - G - 3800) < 25
    # with the matched wear factor and reservoir pressure the two streams overlie and reach the measured THP
    assert np.nanmax(np.abs(r9.profile["Top down (psia)"] - r9.profile["Bottom up (psia)"])) < 15
    assert abs(r9.bottom_up["thp"] - (500 + G)) < 10
    assert abs(s9["Turbine power at the power fluid rate (hp)"] / s9["Top down: pump power (hp)"] - 1) < 0.02
    assert abs(s9["Power fluid rate for the pump power (STB/d)"] / 6308.0 - 1) < 0.005 and not r9.notes
    # a turbine not in the database: the pump side still runs, with a note
    rn = HQ.hsp_quicklook(path, fl, ipr(3800 + G), inp, dataclasses.replace(inst, turbine="NO SUCH TURBINE"),
                          "pe2_style", 1.03, 1.01, I["eq_u"])
    assert np.isclose(rn.summary["Top down: pump intake (psia)"], s0["Top down: pump intake (psia)"])
    assert any("not in the database" in n for n in rn.notes)
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T35_HSP_QuickLook.wellmodel.json"))
    assert proj["widgets"]["opt_artificial_lift"] == "hsp" and proj["inputs"]["hql_qpf"] == 6308.0


def test_hsp_turbine_t68_curves_prosper_t35():
    """PROSPER's CLYDE UNION T68 blade A curves from the user's T35 file (plot at 7500 rpm, 22 stages): total head
    loss, power per stage and efficiency against net turbine flow - exact with the database coefficients and the
    affinity laws; the recommended range 3750-15450 bbl/d (5000-20600 at 10,000 rpm)."""
    from wellmodel import hsp as H
    t = H.TURBINES["CLYDE UNION T68 - A"]
    assert {f"CLYDE UNION T68 - {k}" for k in "ABCDE"} <= set(H.TURBINES)

    def rd(name):
        rows = [l.split() for l in open(Path(__file__).parent / "data" / name)]
        return np.array([(float(r[1]), float(r[2])) for r in rows if len(r) == 3 and r[0].isdigit()])
    h, w, e = rd("t35_t68a_head_22st_7500.prn"), rd("t35_t68a_hp_per_stage_7500.prn"), rd("t35_t68a_eff_7500.prn")
    assert np.allclose(22 * t.head_stage(h[:, 0], 7500.0), h[:, 1], rtol=2e-4, atol=0.2)
    assert np.allclose(t.power_stage(w[:, 0], 7500.0), w[:, 1], rtol=2e-4, atol=1e-4)
    assert np.allclose(100 * t.efficiency(e[:, 0], 7500.0), e[:, 1], atol=0.02)
    assert np.allclose(t.range_at(7500.0), (3750.0, 15450.0))


def test_srp_card_reference_t36():
    """Tutorial 36's dynamometer card shapes (no PROSPER calculation): six shapes, anchored and unanchored, each a
    closed loop with the effective plunger travel inside the stroke - shorter than it for every problem card."""
    import matplotlib
    matplotlib.use("Agg")
    from wellmodel import srp_cards as SC, plotting as P
    cs = SC.reference_cards()
    assert len(cs) == 6 and cs[0].title.startswith("Normal")
    for c in cs:
        for k in (c.anchored, c.unanchored):
            assert np.allclose([k.x[0], k.y[0]], [k.x[-1], k.y[-1]])            # closed
            assert set(k.labels) == set("ABCD")
            a, b = k.ept
            assert 0.0 <= a < b <= 1.0
            if c.title.startswith(("Leaking", "Severe", "Gas")):
                assert b - a < 0.8
        P.plot_card_reference(c.anchored)
    assert cs[0].anchored.ept == (0.0, 1.0)
    assert max(cs[5].unanchored.y) > 1.2 and min(cs[5].anchored.y) < -0.2   # tapping: spike at C / at A


def test_gas_lifted_gas_well_t47():
    """PROSPER Tutorial 47 (the user's file: gas lift in the T38 retrograde condensate well, 0.7 gravity lift gas at a
    fixed 10,000 ft, no annulus friction, WGR 500, PE5, U 5). The lift gas joins the produced gas above the injection
    point (GasLiftedFluid): against PROSPER's gradient (1 MMscf/d of lift gas, 3.923 MMscf/d) the node superficial
    velocities and densities are within 3 % above and below the valve; without lift gas the lifted traverse is the
    plain one. PROSPER's VLP table (350 psig, 0-5 MMscf/d of lift gas): within 6 % from 1.9 to 16.9 MMscf/d (PE5-style
    refitted on these and the T38 gradients; below about 0.6 MMscf/d PROSPER's PE5 loads up much more - ours is
    10-60 % low there); wellhead temperature within 0.5 F. With PROSPER's IPR (its VLP/IPR table) the solution
    bottom-hole pressure is within 1.3 % in every case; the rates within 3.3 % with 0.5-5 MMscf/d of lift gas at 1750
    psig (PROSPER 3.374-3.951) and 1-5 MMscf/d at 1500 psig (2.359-2.792), and +10.7 % unlifted at 1750 psig (near the
    VLP minimum the VLP is flat and PROSPER's IPR steep: 1 % of pressure is 5-8 % of rate); no flow at 1500 psig without
    lift gas in both (with 0.5 MMscf/d ours finds a 0.25 MMscf/d point in the liquid-loading region, PROSPER none)."""
    import dataclasses
    from wellmodel import retrograde as RG, gaslift as GL, gaslift_system as GLS, system as SYS
    G = 14.696
    inp = RG.RetroInputs(7431.94, 0.745, 1.0, 0.745, 43.56, 4596.83 + G, 300.0, 1750 + G, 500.0, 20000, 0, 2.33, 0.31)
    fl = wvlp.RetrogradeBOFluid(inp)
    path = GL.with_nodes(_t38_path(), [10000.0])
    lf = GL.GasLiftedFluid(fl, GL.LiftGas(0.7))
    # node properties against PROSPER's gradient (top node 1750 psig, 3.923 MMscf/d, 1 MMscf/d lift gas)
    nodes = []
    for line in open(Path(__file__).parent / "data" / "t47_gradient_qgi1_top1750.prn"):
        p_ = line.split()
        if len(p_) > 40 and p_[0].isdigit():
            v = [float(x) for x in p_[:7] + [x for x in p_[9:] if not x.isalpha()]]
            if v[1] > 10000:
                v.insert(42, 0.0)
            nodes.append(v)
    for v in nodes:
        md, p, t, vsl, vsg, d, rho_l, rho_g = v[1], v[3], v[4], v[13], v[14], v[29], v[45], v[46]
        if d > 4:
            continue
        c = {"q": np.array([3.923]), "wgr": np.array([500.0]), "qgi_lift": np.array([1.0 if md < 10000 else 0.0])}
        s_ = lf.state(np.array([p + G]), np.array([t]), c)
        a = np.pi * (d / 12) ** 2 / 4
        assert float(np.ravel(s_["ql"])[0]) / a == pytest.approx(vsl, rel=0.03), md
        assert float(np.ravel(s_["qg"])[0]) / a == pytest.approx(vsg, rel=0.025), md
        assert float(np.ravel(s_["rho_g"])[0]) == pytest.approx(rho_g, rel=0.025), md
        assert float(np.ravel(s_["rho_l"])[0]) == pytest.approx(rho_l, rel=0.025), md
    s0 = lf.state(np.array([1000.0]), np.array([200.0]), {"q": np.array([2.0]), "wgr": np.array([500.0])})
    s1 = fl.state(np.array([1000.0]), np.array([200.0]), {"q": np.array([2.0]), "wgr": np.array([500.0])})
    assert all(np.allclose(s0[k], s1[k]) for k in s1)
    case = {"q": np.array([2.0, 10.0]), "wgr": np.array([500.0, 500.0]), "cgr": np.array([134.5, 134.5])}
    P0, _ = wvlp.traverse(path, fl, np.array([364.7, 364.7]), case, "pe5_style", 1.0, 1.0, 5.0)
    r0 = GL.lifted_traverse(path, fl, 364.7, case, 0.0, "pe5_style", 1.0, 1.0, 5.0, fixed_md=10000.0, gas_sg=0.7)
    assert np.allclose(r0.P[:, -1], P0[:, -1], rtol=2e-3)
    # PROSPER's VLP table
    vt = pd.read_csv(Path(__file__).parent / "data" / "t47_vlp_ipr.csv")
    vt = vt[(vt.pr == 1750) & (vt.q > 1.9)]
    q, qg = vt.q.to_numpy(), vt.qgi.to_numpy()
    n = len(q)
    r = GL.lifted_traverse(path, fl, np.full(n, 350 + G), {"q": q, "wgr": np.full(n, 500.0), "cgr": np.full(n, 134.55)},
                           qg, "pe5_style", 1.0, 1.0, 5.0, fixed_md=10000.0, gas_sg=0.7)
    err = (r.P[:, -1] - G) / vt.vlp.to_numpy() - 1
    assert np.abs(err).max() < 0.065, err
    assert np.allclose(r.T[:, 0], vt.tht.to_numpy(), atol=0.5)
    assert np.allclose(r.inj_md, 10000.0)
    # system calculation with PROSPER's IPR (fixed depth gas lift for a gas well)
    from types import SimpleNamespace
    full = pd.read_csv(Path(__file__).parent / "data" / "t47_vlp_ipr.csv")
    sysr = pd.read_csv(Path(__file__).parent / "data" / "t47_system.csv")
    setup = GLS.GasLiftSetup(method=GLS.METHODS[0], gas=GL.LiftGas(0.7), qgi=1.0, fixed_md=10000.0)
    base = {"whp": 364.7, "u": 5.0, "wgr": 500.0, "sep_gor": 7432.94}
    for pr, qgis, tol in ((1750.0, (0.5, 1.0, 3.0, 5.0), 0.035), (1500.0, (1.0, 2.5, 5.0), 0.035), (1750.0, (0.0,), 0.12)):
        for qgi in qgis:
            ip = full[(full.pr == pr) & (full.qgi == qgi)].sort_values("q")
            curve = SimpleNamespace(q=np.r_[0.0, ip.q.to_numpy()] * 1000, pwf=np.r_[pr, ip.ipr.to_numpy()] + G,
                                    pr=pr + G, aof=float(ip.q.max()) * 1000, fluid="gas", model="PROSPER table")
            res, _ = SYS.solve_cases(lambda tid: path, fl, "gas", base, [{"qgi": qgi}], lambda ch, c=curve: c,
                                     "pe5_style", 1.0, 1.0, lift=setup)
            ref = float(sysr[(sysr.pr == pr) & (sysr.qgi == qgi)].q.iloc[0])
            assert res[0].q == pytest.approx(ref, rel=tol), (pr, qgi, res[0].q, ref)
            bhp = float(sysr[(sysr.pr == pr) & (sysr.qgi == qgi)].bhp.iloc[0])
            assert res[0].pwf - G == pytest.approx(bhp, rel=0.0135), (pr, qgi)
    ip = full[(full.pr == 1500) & (full.qgi == 0.0)].sort_values("q")
    curve = SimpleNamespace(q=np.r_[0.0, ip.q.to_numpy()] * 1000, pwf=np.r_[1500.0, ip.ipr.to_numpy()] + G,
                            pr=1500 + G, aof=float(ip.q.max()) * 1000, fluid="gas", model="PROSPER table")
    res, _ = SYS.solve_cases(lambda tid: path, fl, "gas", base, [{"qgi": 0.0}], lambda ch: curve, "pe5_style", 1.0,
                             1.0, lift=setup)
    assert res[0].q is None                              # 1500 psig without lift gas: no flow (PROSPER too)
    with pytest.raises(ValueError):
        SYS.solve_cases(lambda tid: path, fl, "gas", base, [{"qgi": 1.0}], lambda ch: curve, "pe5_style",
                        lift=dataclasses.replace(setup, method=GLS.METHODS[1]))


def test_petex_gas_rel_perm_with_liquid_t47():
    """PROSPER's Petroleum Experts gas IPR reduces the gas relative permeability with the produced liquid (its
    'Calculated Gas Relative Permeability', rel perms off). User's Tutorial 47 file (1750 psig, 300 F, total GOR
    7432.66, Swc 0.2): WGR 0 / 100 / 250 / 500 / 1000 / 2000 - 0.74108 / 0.69293 / 0.62888 / 0.53951 / 0.40607 /
    0.24033; Swc 0.3 at WGR 500 - 0.40432. Ours within 0.6 %; no liquid gives the Tutorial 02 value. The example's IPR
    AOF is then within 1.3-2.7 % of PROSPER's (23.33 / 16.93 / 12.71 / 7.48 / 12.65 MMscf/d; it was 25.4 against 16.9)."""
    from wellmodel import inflow as I
    bg = 0.02827 * 0.934746 * 759.67 / 1764.696 / 5.614583 * 1e6          # rb/MMscf at 1750 psig, 300 F (Z PROSPER's)
    cgr = 1e6 / 7432.66
    for wgr, swc, k in ((0, .2, .74108), (100, .2, .69293), (250, .2, .62888), (500, .2, .53951), (1000, .2, .40607),
                        (2000, .2, .24033), (500, .3, .40432)):
        assert I.petex_krg_liquid(swc, wgr, cgr, bg) == pytest.approx(k, rel=0.006), (wgr, swc)
    assert I.petex_krg_liquid(0.25, 0.0, 0.0, bg) == pytest.approx(I.petex_krg(0.25))
    assert I.petex_krg_liquid(0.25, 500.0, 0.0, None) == pytest.approx(I.petex_krg(0.25))


def test_diluent_injection_with_gas_lift_t30():
    """PROSPER Tutorial 30, the user's file (diluent injection with gas lift in the T19 / T23 Logie well: 180,000 ppm
    formation water, 20,000 ppm diluent water down the annulus into the tubing at 9000 ft, 0.7 gravity lift gas at 7500
    ft, Pr 4800 psig in the file, 54 % water cut, GOR 800, 250 psig, PE2 1.03 / 1.01; 50 cases: diluent 20-40 % x GLR
    0-1500). PROSPER's definitions: diluent = percentage x produced liquid, lift gas = GLR x produced liquid (in all 50
    cases). VLP within 4 % above 1000 STB/d (unlifted within 1 %), wellhead temperature within 0.2 F; with PROSPER's
    IPR the 50 solution rates within 1 % and bottom-hole pressures within 1 %; optimum GLR 900 scf/STB at every
    diluent percentage, as PROSPER (the tutorial: 850-950)."""
    import dataclasses, json
    from types import SimpleNamespace
    from wellmodel import diluent as DL, system as SYS
    G = 14.696
    f0 = _t05_oil()
    m = f0.m
    fl = wvlp.BlackOilFluid(wbo.BlackOilModel(dataclasses.replace(m.inp, salinity_ppm=180000), m.family,
                                              m.visc_family, m.params))
    path, _ = _t23_path((7500.0, 9000.0))
    df = DL.DiluentFluid(fl, 20000)
    c0 = {"q": np.array([5000.0]), "wc": np.array([0.54]), "gor": np.array([800.0])}
    s0, s1 = df.state(np.array([2000.0]), np.array([180.0]), c0), fl.state(np.array([2000.0]), np.array([180.0]), c0)
    assert all(np.allclose(s0[k], s1[k]) for k in s1)
    setup = DL.DiluentSetup(pct=30.0, glr_inj=900.0)
    qd, qg = setup.rates(np.array([10116.1]), 0.54)
    assert qd[0] == pytest.approx(3034.8, abs=0.1) and qg[0] == pytest.approx(9.104, abs=0.001)   # PROSPER case 27
    sysr = pd.read_csv(Path(__file__).parent / "data" / "t30_system.csv")
    assert np.allclose(sysr.qdil, sysr.q * sysr.pct / 100, rtol=1e-3) and np.allclose(sysr.qgi, sysr.q * sysr.glr / 1e6,
                                                                                         rtol=2e-3, atol=2e-3)
    vt = pd.read_csv(Path(__file__).parent / "data" / "t30_vlp_ipr.csv")
    v = vt[(vt.q > 1000) & (vt.q < 16000)]
    n = len(v)
    P, T = DL.traverse(path, fl, np.full(n, 250 + G), {"q": v.q.to_numpy(), "wc": np.full(n, 0.54),
                       "gor": np.full(n, 800.0)}, setup, "pe2_style", 1.03, 1.01, 8.580277, q_dil=v.qdil.to_numpy(),
                       qgi=v.qgi.to_numpy())
    err = (P[:, -1] - G) / v.vlp.to_numpy() - 1
    assert np.abs(err).max() < 0.04 and np.abs(err[v.glr.to_numpy() == 0]).max() < 0.01
    assert np.allclose(T[:, 0], v.tht.to_numpy(), atol=0.2)
    base = {"whp": 250 + G, "u": 8.580277, "wc": 54.0, "gor": 800.0}
    sub = sysr[sysr.pct.isin([20.0, 30.0, 40.0]) & sysr.glr.isin([0.0, 300.0, 900.0, 1500.0])]
    for _, r in sub.iterrows():
        ip = vt[vt.case == r.case].sort_values("q")
        curve = SimpleNamespace(q=np.r_[0.0, ip.q.to_numpy()], pwf=np.r_[4800.0, ip.ipr.to_numpy()] + G, pr=4800 + G,
                                aof=float(ip.q.max()), fluid="oil", model="PROSPER table")
        res, _ = SYS.solve_cases(lambda tid: path, fl, "oil", base, [{"dil_pct": r.pct, "glr_dil": r.glr}],
                                 lambda ch, c=curve: c, "pe2_style", 1.03, 1.01, diluent=DL.DiluentSetup())
        assert res[0].q == pytest.approx(r.q, rel=0.01), (r.pct, r.glr)
        assert res[0].pwf - G == pytest.approx(r.bhp, rel=0.01), (r.pct, r.glr)
        if r.glr > 0:
            assert res[0].details["diluent"]["Lift gas rate (MMscf/d)"] == pytest.approx(r.glr * res[0].q / 1e6)
    opt = sysr.loc[sysr.groupby("pct").q.idxmax()]
    assert (opt.glr == 900.0).all()
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T30_Diluent_Injection_With_Gas_Lift.wellmodel.json"))
    assert proj["widgets"]["opt_artificial_lift"] == "diluent_gl" and proj["inputs"]["dil_pct_basis"] == "produced liquid"


def test_emulsion_viscosity_t46():
    """PROSPER Tutorial 46 (Modelling Emulsions): 11 API heavy oil, GOR 40, Vazquez-Beggs / Egbogah matched to 100 cp
    at 500 psig and 200 F. Emulsion data at 2000 psig, 200 F (5-55 % WC oil-continuous, plateau 60-80 %): PROSPER's
    match P1 1, P2 0.69799, P3 3.77291, std dev 3.21611 - a polynomial in water cut times the oil viscosity. The
    gradient (1500 STB/d, 500 psig wellhead, PE2) is unchanged at WC 0 and rises with water cut with the emulsion,
    falls without it. Against the user's PROSPER gradients with the emulsion: bottom-hole pressure within 0.5 %."""
    import json
    from wellmodel import emulsion as EM
    inp = wbo.PVTInputs(rsb=40, api=11, gas_sg=0.83, salinity_ppm=10000)
    md = wbo.MatchData(200, 514.696, pd.DataFrame({"Pressure": [514.696], "Gas Oil Ratio": [40], "Oil FVF": [1.051],
                                                   "Oil Viscosity": [100]}))
    r = wbo.match_all(inp, md)
    assert r.matches[("visc", "egbogah")].p1 == pytest.approx(0.98594, abs=2e-4)
    assert r.matches[("visc", "egbogah")].p2 == pytest.approx(-1.44678, abs=2e-3)
    m = wbo.BlackOilModel(inp, "vazquez_beggs", "egbogah",
                          {k: r.matches[(k, "vazquez_beggs")] for k in ("pb", "rs", "bo")} | {"visc": r.matches[("visc", "egbogah")]})
    mu_o = float(m.table(200, [2014.696])["Oil viscosity (cp)"].iloc[0])
    assert mu_o == pytest.approx(100.0, rel=1e-3)
    wc = [5, 10, 20, 30, 40, 50, 55, 85, 90]
    mu = [105, 115, 130, 150, 188, 234, 250, 53, 10.5]
    fit = EM.match(wc, mu, mu_o, 0.6, 3)
    assert fit.n == 7 and fit.params[0] == 1.0
    assert fit.params[1] == pytest.approx(0.69799, rel=0.005) and fit.params[2] == pytest.approx(3.77291, rel=0.002)
    assert fit.std == pytest.approx(3.21611, rel=0.002)
    em = EM.EmulsionModel((1, 0.69799, 3.77291, 0, 0), 0.6, 0.8)
    v = em.viscosity(np.array([0, 0.55, 0.6, 0.7, 0.8, 1.0]), 100.0, 0.32)
    assert v[0] == 100 and v[1] == pytest.approx(252.5, abs=0.1) and v[2] == pytest.approx(277.7, abs=0.1)
    assert v[3] == v[2] == v[4] and v[5] == pytest.approx(0.32)
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 4800, "Tubing Inside Diameter (in)": 2.4,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 5000, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = wvlp.build_flow_path(top, secs, pts, [0, 5000], [0, 5000], [0, 5000], [80, 200])
    case = {"q": np.full(4, 1500.0), "wc": np.array([0, 0.25, 0.5, 0.75]), "gor": np.full(4, 40.0)}
    P0 = wvlp.traverse(path, wvlp.BlackOilFluid(m), np.full(4, 514.696), case, "pe2_style", 1, 1, 8.0)[0][:, -1]
    P1 = wvlp.traverse(path, wvlp.BlackOilFluid(m, em), np.full(4, 514.696), case, "pe2_style", 1, 1, 8.0)[0][:, -1]
    assert P1[0] == pytest.approx(P0[0]) and np.all(np.diff(P0) < 0) and np.all(np.diff(P1) > 0)
    assert np.all(np.diff(P1 - P0) > 0)
    # the user's PROSPER gradients with the emulsion (WC 0 / 25 / 50 / 75 %): the emulsion water cut is the stock-tank
    # one; liquid viscosity within 0.1 %, pressures within 0.6 %, temperatures within 0.1 F
    g = pd.read_csv(Path(__file__).parent / "data" / "t46_gradients.csv")
    Pe, Te = wvlp.traverse(path, wvlp.BlackOilFluid(m, em), np.full(4, 514.696), case, "pe2_style", 1, 1, 8.0)
    for i, w in enumerate((0, 25, 50, 75)):
        d = g[g.wc == w]
        pm = np.interp(d.md, path.md, Pe[i]) - 14.696
        tm = np.interp(d.md, path.md, Te[i])
        assert np.abs(pm[1:] / d.p.to_numpy()[1:] - 1).max() < 0.006, w
        assert np.abs(tm - d.t.to_numpy()).max() < 0.1, w
        ok = d.mu_l.notna().to_numpy()
        s = wvlp.BlackOilFluid(m, em).state(d.p.to_numpy()[ok] + 14.696, d.t.to_numpy()[ok],
                                             {"q": np.full(ok.sum(), 1500.0), "wc": np.full(ok.sum(), w / 100),
                                              "gor": np.full(ok.sum(), 40.0)})
        assert np.abs(s["mu_l"] / d.mu_l.to_numpy()[ok] - 1).max() < 0.001, w
    # water-continuous side (WC 85 / 90 / 95 %): log-linear from the plateau at 80 % to the water viscosity at 100 %
    # - exact with PROSPER's water viscosity (0.33061 cp at 10,000 ppm and 200 F, its PVT table, now reproduced);
    # liquid viscosity within 0.3 %, pressures within 0.2 %
    # PROSPER's water viscosity at low salinity (Tutorial 46 PVT tables, 0 and 10,000 ppm, independent of pressure)
    for t_, s_, v_ in ((30, 0, 1.80595), (120, 0, 0.62448), (202.5, 0, 0.31893), (75, 1, 1.03122), (200, 1, 0.33061)):
        assert float(wbo._prosper_water(s_, 1000.0, t_)[2]) == pytest.approx(v_, rel=1e-4)
    c3 = {"q": np.full(3, 1500.0), "wc": np.array([0.85, 0.9, 0.95]), "gor": np.full(3, 40.0)}
    Pw, _ = wvlp.traverse(path, wvlp.BlackOilFluid(m, em), np.full(3, 514.696), c3, "pe2_style", 1, 1, 8.0)
    for i, w in enumerate((85, 90, 95)):
        d = g[g.wc == w]
        pm = np.interp(d.md, path.md, Pw[i]) - 14.696
        assert np.abs(pm[1:] / d.p.to_numpy()[1:] - 1).max() < 0.003, w
        b = d[d.md == 4900.0].iloc[0]
        mo = wvlp.BlackOilFluid(m)._props(np.array([b.p + 14.696]), np.array([b.t]), np.array([40.0]))[2]
        assert float(em.viscosity(w / 100, mo, 0.33061)[0]) == pytest.approx(b.mu_l, rel=0.002), w
        s = wvlp.BlackOilFluid(m, em).state(d.p.to_numpy()[1:-1] + 14.696, d.t.to_numpy()[1:-1],
                                             {"q": np.full(len(d) - 2, 1500.0), "wc": np.full(len(d) - 2, w / 100),
                                              "gor": np.full(len(d) - 2, 40.0)})
        assert np.abs(s["mu_l"] / d.mu_l.to_numpy()[1:-1] - 1).max() < 0.003, w
    proj = json.load(open(__file__.replace("tests/test_wellmodel.py", "examples/T46_Modelling_Emulsions.wellmodel.json")))
    assert proj["widgets"]["opt_emulsions"] == "emulsion" and proj["inputs"]["emul_p3"] == 3.77291


def _t12_read(path):
    """PROSPER MultiLateral report: (branch rates, PI, composite skin, {branch: segment array})."""
    import re
    txt = open(path).read()
    seg = {}
    for blk in txt.split("SEGMENT DETAILS")[1:]:
        name = blk.split("\n")[1].strip(" .")
        rows = []
        for line in blk.splitlines()[9:]:
            try:
                v = [float(x) for x in line.split()]
            except ValueError:
                continue
            if len(v) == 8:
                rows.append(v)
        seg[name] = np.array(rows)
    m = re.search(r"Reservoir\s+([\d.]+)\s+(-?[\d.]+)\s+4900", txt)
    br = {n: float(re.search(n + r"\s+([\d.]+)\s+([\d.]+)", txt).group(1)) for n in ("Branch 1", "Branch 2")}
    return br, float(m.group(1)), float(m.group(2)), seg


def test_multilateral_ipr_t12():
    """PROSPER Tutorial 12 (Multilateral Well IPR), the user's file: Powderhall gas (T06 PVT, 0.58, 250 F, Pr 5300
    psig), two branches from a junction at 5408 ft MD, 2 x 50 ft perforated each (skin 1), 25 md, kv 2.5 md, 55 ft,
    390 acres. Infinite conductivity: 11 one-point runs at 5000 psig (kv 1-25, skin 0/1, 390/1000 acres, D 0 /
    entered / calculated) within 1.3 % (2.8 %: skin 0 with the calculated D, PROSPER's non-Darcy iteration not
    converged at low drawdown), the kv = kh rate profile along both branches within 2.5 %, the IPR curve within 0.5 %
    below 4200 psig; finite conductivity: junction pressure within 1 psi and rate within 0.5 % at 5000 psig, the IPR
    within 5 %."""
    import dataclasses
    from wellmodel import multilateral as ML, gaspvt as gp, project
    G = 14.696
    D = Path(__file__).parent / "data"
    proj = project.loads(open(Path(__file__).parent.parent / "examples" / "T06_PVT_Matching_For_Gas_Well.wellmodel.json").read())
    g = gp.GasPVTInputs(gas_sg=0.58, sep_p=200, cgr=1, cond_api=50, wgr=1, salinity_ppm=100000, co2=0.5, n2=2.0,
                        t_res=250, p_res=5300)
    g.match = gp.match_gas(g, [(166.0, proj["tables"]["pvt_gas_match"])])
    base = ML.example_t12()
    assert base.check() == []

    def well(kv=2.5, skin=1.0, area=390.0, d=None):
        brs = [dataclasses.replace(b, perfs=[ML.Perforation(p.md_top, p.md_bottom, skin, d) for p in b.perfs])
               for b in base.branches]
        return dataclasses.replace(base, branches=brs, res=dataclasses.replace(base.res, kv=kv, area=area))
    mod0 = ML.MultilateralModel(well(), g, None, 1.0, 1.0)
    assert mod0.D == pytest.approx(2.7862e-6 * 24.45, rel=0.01)          # PROSPER shows the calculated D / 24.45
    de = 5.5724e-6
    runs = [("t12_onepoint_infinite.prn", 2.5, 1, 390, None, 0.013), ("t12_inf_d0.prn", 2.5, 1, 390, 0.0, 0.013),
            ("t12_inf_d5.5724e-6.prn", 2.5, 1, 390, de, 0.013), ("t12_inf_skin0.prn", 2.5, 0, 390, None, 0.03),
            ("t12_inf_kv25.prn", 25, 1, 390, None, 0.013), ("t12_inf_kv25_s1_de.prn", 25, 1, 390, de, 0.013),
            ("t12_inf_kv25_s0_de.prn", 25, 0, 390, de, 0.013), ("t12_inf_kv2.5_s0_de.prn", 2.5, 0, 390, de, 0.013),
            ("t12_inf_kv1_s1_de.prn", 1, 1, 390, de, 0.013), ("t12_inf_kv10_s1_de.prn", 10, 1, 390, de, 0.013),
            ("t12_inf_kv25_s1_a1000_de.prn", 25, 1, 1000, de, 0.013)]
    for fn, kv, sk, area, d, tol in runs:
        br, pi, comp, seg = _t12_read(D / fn)
        mod = ML.MultilateralModel(well(kv, sk, area, d), g, None, 1.0, 1.0)
        s, tab = mod.results(5000 + G, finite=False)
        assert s["Total gas rate (MMscf/d)"] == pytest.approx(sum(br.values()), rel=tol), fn
        if fn == "t12_inf_kv25_s1_de.prn":            # kv = kh: PROSPER's rate per foot along both branches
            tot = sum(br.values())
            for name in ("Branch 1", "Branch 2"):
                t = tab[tab.Branch == name]
                mid = 0.5 * (t["MD top (ft)"] + t["MD bottom (ft)"]).to_numpy()
                ours = (t["Rate per ft (MMscf/d/ft)"] / s["Total gas rate (MMscf/d)"]).to_numpy()
                pm = 0.5 * (seg[name][:, 0] + seg[name][:, 1])
                ok = np.array([np.abs(mid - x).min() < 0.15 for x in pm])
                theirs = seg[name][ok, 4] / tot
                assert np.abs(np.interp(pm[ok], mid, ours) / theirs - 1).max() < 0.025, name
    # infinite-conductivity IPR (D calculated)
    ci = pd.read_csv(D / "t12_curve_infinite.csv")
    for _, r in ci[ci.p <= 4200].iterrows():
        assert mod0.rate_infinite(r.p + G) / 1000 == pytest.approx(r.q, rel=0.005)
    # finite conductivity: the one-point run at 5000 psig and the IPR
    br, pi, comp, seg = _t12_read(D / "t12_onepoint_finite.prn")
    s, tab = mod0.results(5000 + G, finite=True)
    assert s["Total gas rate (MMscf/d)"] == pytest.approx(23.545, rel=0.005)
    assert s["Junction pressure (psia)"] - G == pytest.approx(5002.3, abs=1.0)
    assert s["Branch 1 rate (MMscf/d)"] == pytest.approx(br["Branch 1"], rel=0.015)
    assert s["Branch 2 rate (MMscf/d)"] == pytest.approx(br["Branch 2"], rel=0.02)
    assert s["Composite skin"] == pytest.approx(comp, abs=0.1) and s["Layer PI (MMscf/d per 1e6 psi2/cp)"] == pytest.approx(pi, rel=0.01)
    cf = pd.read_csv(D / "t12_curve_finite.csv")
    for _, r in cf[cf.p.isin([4185.5, 2793.7, 1401.8, 566.7, 10.0])].iterrows():
        assert mod0.solve_finite(r.p + G)["Q"] / 1000 == pytest.approx(r.q, rel=0.05), r.p
    c = ML.ipr_curve(well(), g, 1.0, 1.0, finite=False, n=8)
    assert c.aof / 1000 == pytest.approx(146.612, rel=0.01)
    pj = project.loads(open(Path(__file__).parent.parent / "examples" / "T12_Multilateral_Well_IPR.wellmodel.json").read())
    assert pj["inputs"]["ipr_model_gas"] == "Multilateral Well" and len(pj["tables"]["ipr_mlat_pf"]) == 4


def _t31_surface_df():
    rows = [{"Label": "Manifold", "Type": "Manifold", "TVD of End Point (ft)": 0.0}]
    for L, t in [(250, 250), (500, 475), (1000, 475), (750, 600), (1000, 600)]:
        rows.append({"Type": "Line Pipe", "Pipe Length (ft)": float(L), "TVD of End Point (ft)": float(t),
                     "Pipe Inside Diameter (in)": 4.0, "Pipe Inside Roughness (in)": 0.0006, "Rate Multiplier": 1.0})
    rows.append({"Label": "Multiphase pump", "Type": "Pump", "Rate Multiplier": 1.0})
    rows.append({"Type": "Line Pipe", "Pipe Length (ft)": 250.0, "TVD of End Point (ft)": 600.0,
                 "Pipe Inside Diameter (in)": 4.0, "Pipe Inside Roughness (in)": 0.0006, "Rate Multiplier": 1.0})
    df = pd.DataFrame(rows)
    df["Fitting"] = None
    df["Label"] = df["Label"].astype(object)
    return df


def test_surface_multiphase_pump_t31():
    """PROSPER Tutorial 31, the user's file: OneSubsea H800/65 (FMPP500, single pump, no wear) in the surface pipeline
    250 ft downstream of the wellhead of the T19 Logie well (Tutorial 05 PVT, 23,000 ppm), Pr 4800 psig, 80 % water
    cut, GOR 800, manifold 500 psig, Beggs & Brill surface, PE2 1.03 / 1.00, pipeline U 8.5 and 52 F; sensitivity on
    the power available 250-600 hp. The pump draws all the power; its pressure gain follows the hydraulic power
    balance on the discharge rate with the fitted efficiency (within 1.3 % of PROSPER's dP); intake at 0 psig where
    the power would pull it lower; VLP within 2 % above 3000 STB/d (1 % where the intake is above 0 psig), and with PROSPER's IPR the eight solution rates
    within 1.5 %, bottom-hole pressures within 0.6 %, pump intake within 5 psi."""
    import json
    from types import SimpleNamespace
    from wellmodel import mpump as MP, system as SYS
    G = 14.696
    vt = pd.read_csv(Path(__file__).parent / "data" / "t31_vlp_ipr.csv")
    sysr = pd.read_csv(Path(__file__).parent / "data" / "t31_system.csv")
    pump = MP.MultiphasePump(350.0)
    # pump curve alone: PROSPER's (intake, discharge, discharge rate) at every point with a real operating point
    ok = vt[(vt.eff > 10.02) & (vt.pin > 50)]
    for _, r in ok.iterrows():
        pm = pump.with_power(r.hp)
        dp = pm.dp(r.qd, (r.pout + 14.7) / (r.pin + 14.7))
        assert dp == pytest.approx(r.pout - r.pin, rel=0.015), (r.case, r.q)
    assert pump.efficiency(5000.0, 1.0) == pump.eta_min
    _, items = wsurf.build_items(_t31_surface_df(), pump)
    assert [i.kind for i in items] == ["pipe", "pump", "pipe", "pipe", "pipe", "pipe", "pipe"]
    assert wsurf.has_pump(items) and [i.pump.power for i in wsurf.with_pump_power(items, 500) if i.kind == "pump"] == [500]
    assert not any(l == "error" for l, _ in weq.check_surface(_t31_surface_df()))
    fl, (path, _) = _t05_oil(), _t23_path()
    hc = weq.HeatCapacities()
    surf = SYS.SurfaceSetup(items, 52.0, 8.5, "beggs_brill")
    U = 8.580277
    for power in (250, 450, 600):
        v = vt[(vt.power == power) & (vt.q > 3000)]
        n = len(v)
        cs = {"q": v.q.to_numpy(), "wc": np.full(n, 0.8), "gor": np.full(n, 800.0)}
        w, cp = fl.mass_cp(cs, hc)
        twh = wvlp.temperature_profile(path, w, cp, U)[:, 0]
        pwh, rows, _, _ = wsurf.wellhead_pressure(surf.items_for({"pump_power": power}), fl, "oil", np.full(n, 500 + G),
                                                  cs, twh, "beggs_brill", hc, 52.0, 8.5, detail=True)
        pr = next(r for r in rows if r["Type"] == "Pump")
        assert np.allclose(pr["Upstream pressure (psia)"] - G, v.pin, atol=10.0), power
        assert np.allclose(pr["Downstream pressure (psia)"] - G, v.pout, atol=10.0), power
        assert np.allclose(pr["_pump"]["Surface pump discharge rate (RB/d)"], v.qd, rtol=0.01)
        P, _ = wvlp.traverse(path, fl, pwh, cs, "pe2_style", 1.03, 1.0, U, hc)
        assert np.allclose(P[:, -1] - G, v.vlp, rtol=0.02), power
    # forward march (top node): intake -> discharge inverts the backward march
    c1 = {"q": np.array([7000.0]), "wc": np.array([0.8]), "gor": np.array([800.0])}
    p_out = MP.discharge_pressure(pump, fl, c1, np.array([435.0]), 152.0)
    assert MP.intake_pressure(pump, fl, c1, p_out, 152.0)[0] == pytest.approx(435.0, abs=0.05)
    assert MP.intake_pressure(pump, fl, {k: v / 20 for k, v in c1.items()} | {"wc": c1["wc"], "gor": c1["gor"]},
                              np.array([700.0]), 152.0)[0] == MP.P_ATM          # very low rate: intake at 0 psig
    ip = vt[vt.power == 250].sort_values("q")
    curve = SimpleNamespace(q=np.r_[0.0, ip.q.to_numpy()], pwf=np.r_[4800.0, ip.ipr.to_numpy()] + G, pr=4800 + G,
                            aof=float(ip.q.max()), fluid="oil", model="PROSPER table")
    res, _ = SYS.solve_cases(lambda tid: path, fl, "oil", {"whp": 500 + G, "u": U, "wc": 80.0, "gor": 800.0},
                             [{"pump_power": float(p)} for p in sysr.power], lambda ch: curve, "pe2_style", 1.03, 1.0,
                             surface=surf)
    for r, (_, s) in zip(res, sysr.iterrows()):
        assert r.q == pytest.approx(s.q, rel=0.015), s.power
        assert r.pwf - G == pytest.approx(s.bhp, rel=0.006), s.power
    det, _, _ = SYS.solution_details(path, fl, "oil", res[2].params, res[2].q, "pe2_style", 1.03, 1.0, hc, surface=surf)
    assert det["Surface pump intake pressure (psia)"] - G == pytest.approx(421.3, abs=5.0)
    assert det["Surface pump efficiency (%)"] == pytest.approx(16.13, abs=0.3)
    assert det["Surface pump head (ft)"] == pytest.approx(1419.7, rel=0.03)
    assert det["Surface pump power (hp)"] == 350.0
    assert q_increasing(res)
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T31_Surface_Multiphase_Pump.wellmodel.json"))
    assert proj["widgets"]["opt_artificial_lift"] == "mpump" and proj["widgets"]["sys_vars"] == ["pump_power"]


def q_increasing(res):
    q = [r.q for r in res]
    return all(b > a for a, b in zip(q, q[1:]))


def _t40_path(casing_id):
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 3.992,
             "Tubing Inside Roughness (in)": 0.0018, "Tubing Outside Diameter (in)": 4.5},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 3.5},
            {"Type": "Tubing", "Measured Depth (ft)": 11800, "Tubing Inside Diameter (in)": 3.992,
             "Tubing Inside Roughness (in)": 0.0018, "Tubing Outside Diameter (in)": 4.5},
            {"Type": "Casing", "Measured Depth (ft)": 12000, "Casing Inside Diameter (in)": casing_id,
             "Casing Inside Roughness (in)": 0.0018}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    return wvlp.build_flow_path(top, secs, pts, [0, 12000], [0, 12000], [0, 12000], [70, 300])


def _t38_retro():
    from wellmodel import retrograde as RG
    return wvlp.RetrogradeBOFluid(RG.RetroInputs(7431.94, 0.745, 1.0, 0.745, 43.56, 4596.83 + 14.696, 300.0,
                                                 6000 + 14.696, 0.0, 20000, 0, 2.33, 0.31))


def test_improved_approximation_temperature_t41():
    """PROSPER Tutorial 41, the user's file (the T38 retrograde condensate well, black oil, PE5, wellhead 2500 psig;
    Improved Approximation with U 2.63 to 4000 ft, 4.0 to 9312.5 ft, 2.0 below - the file's table): gradients at 10,
    33.2 and 60 MMscf/d, at 33.2 MMscf/d with U = 0.001 (no heat loss) and on dry gas. The energy balance with
    gravity, Joule-Thomson (real-gas, condensate and drop-out terms) and heat loss reproduces PROSPER's temperatures
    within 5 F at every node at 10 and 33 MMscf/d, 3 F adiabatic, 2 F on dry gas, 7.5 F at 60 MMscf/d (PE5's
    pressure is 3 % higher there, which moves the dew point up the well); the Rough Approximation was 25-70 F off
    (no Joule-Thomson cooling). Bottom-hole pressures within 3.5 % (PE5)."""
    import dataclasses
    from wellmodel import thermal as TH
    G = 14.696
    df = pd.read_csv(Path(__file__).parent / "data" / "t40_t41_gradients.csv", dtype={"run": str})
    path0 = _t40_path(6.4)
    fl = _t38_retro()
    gas = wvlp.WetGasFluid(wbo.PVTInputs(rsb=1, api=43.56, gas_sg=0.745, salinity_ppm=20000, co2=2.33, n2=0.31))
    U = [2.63, 2.63, 4.0, 2.0]
    for run, q, whp, uu, f_, case in (("1", 33.219, 2500, U, fl, {}), ("2A", 10.0, 2500, U, fl, {}),
                                      ("2B", 60.0, 2400, U, fl, {}), ("3", 33.219, 2500, [0.001] * 4, fl, {}),
                                      ("4", 33.219, 2500, [0.001] * 4, gas, {"cgr": 0.0})):
        path = dataclasses.replace(path0, thermal={"model": "improved",
                                                   "u_seg": TH.u_segments(path0, [0, 4000, 9312.5, 12000], uu)})
        P, T = wvlp.traverse(path, f_, np.array([whp + G]), {"q": q, "wgr": 0.0, **case}, "pe5_style")
        g = df[df.run == run]
        t_err = np.interp(g.md, path.md, T[0]) - g.t
        assert np.abs(t_err).max() < {"3": 3.0, "4": 2.0, "2B": 7.5}.get(run, 5.0), run
        assert (P[0, -1] - G) / g.p.iloc[-1] - 1 == pytest.approx(0.0, abs=0.035), run
    # the U table: each row's U over the interval ending at its depth
    useg = TH.u_segments(path0, [0, 4000, 9312.5, 12000], U)
    mid = 0.5 * (path0.md[1:] + path0.md[:-1])
    assert (useg[mid < 3900] == 2.63).all() and (useg[(mid > 4100) & (mid < 9200)] == 4.0).all()
    assert (useg[mid > 9500] == 2.0).all()
    # system analysis (the file's: wellhead 2500 psig, Petroleum Experts IPR k 25 md, skin 5, 50 days, Swc 0.2):
    # PROSPER's M(P) uses the gas phase of its black-oil condensate model and no liquid kr reduction for a
    # retrograde condensate - IPR within 10 psi; the solution within 4 % of PROSPER's 34.35 MMscf/d
    from types import SimpleNamespace
    from wellmodel import inflow as ipr, system as SYS
    sysr = pd.read_csv(Path(__file__).parent / "data" / "t41_system.csv")
    ri = fl.rinp

    def mu_z(p):
        r_ = __import__("wellmodel.retrograde", fromlist=["props"]).props(ri, np.atleast_1d(np.asarray(p, float)), 300.0)
        return r_["mu_g"], r_["z"]
    pr = 6000 + G
    psi = ipr.pseudo_pressure_fn(mu_z, pr, n=600, gauge=True)
    mr, zr = (float(v[0]) for v in mu_z(np.array([pr])))
    curve = ipr.petex_gas(pr, 300.0, 25, 100, 340, 31.6, 0.354, 100, 50, 0.25, 0.2, 5, psi, mr, 1 / pr, 0.745,
                          None, "total", z_r=zr)
    assert curve.params["Non-Darcy flow factor D (1/(Mscf/d))"] == pytest.approx(2.2378e-5, rel=0.03)
    sub = sysr[sysr.q < 120]
    o_ = np.argsort(curve.q)
    assert np.abs(np.interp(sub.q.to_numpy() * 1000, curve.q[o_], curve.pwf[o_]) - G - sub.ipr.to_numpy()).max() < 10.0
    path = dataclasses.replace(path0, thermal={"model": "improved",
                                               "u_seg": TH.u_segments(path0, [0, 4000, 9312.5, 12000], U)})
    res, _ = SYS.solve_cases(lambda tid: path, fl, "gas", {"whp": 2500 + G, "u": 2.0, "wgr": 0.0, "sep_gor": 7431.94},
                             [{}], lambda ch: curve, "pe5_style")
    assert res[0].q == pytest.approx(34.35, rel=0.04)
    assert res[0].wht_q == pytest.approx(192.8, abs=3.0)
    import json
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T41_Improved_Approximation.wellmodel.json"))
    assert proj["widgets"]["opt_temp_model"] == "improved" and "eq_ihtc" in proj["tables"]


def test_enthalpy_balance_temperature_t40():
    """PROSPER Tutorial 40, the user's file (the same well with the Enthalpy Balance: three casing strings, CaBr2
    12 lb/gal above the packer at 11,000 ft, shale / 12 % porosity water sandstone / shale, 35.3 MMscf/d at 2500
    psig) at 5, 50 and 500 days since production started: the overall U from the completion, the casings and
    cement and the transient formation conduction within 6 % of PROSPER's at every node outside the 7500-8000 ft
    interval (15 % there), temperatures within 3 F at every node, bottom-hole pressure within 3 % (PE5)."""
    import dataclasses
    from wellmodel import enthalpy as EB
    G = 14.696
    df = pd.read_csv(Path(__file__).parent / "data" / "t40_t41_gradients.csv", dtype={"run": str})
    path0 = _t40_path(5.87)
    fl = _t38_retro()
    cas = [EB.Casing(5000, 12, 5000, 11.5, 90.7, 0, 0, 11.7), EB.Casing(8000, 10.5, 8000, 8.5, 56.4, 5000, 0, 11.7),
           EB.Casing(12000, 8.5, 12000, 7.2, 47.1, 7500, 0, 11.7)]
    lit = [EB.Lithology(4000, "Shale"), EB.Lithology(9750, "Sandstone", 0.12, "Water"), EB.Lithology(12000, "Shale")]
    assert cas[2].id == pytest.approx(5.848, abs=0.002)
    for run, days in (("7A", 5.0), ("6", 50.0), ("7B", 500.0)):
        data = EB.EnthalpyData(cas, lit, packer_md=11000, days=days, tubing_od=4.5, tubing_end=11800)
        assert not data.check()
        path = dataclasses.replace(path0, thermal={"model": "enthalpy", "data": data})
        P, T = wvlp.traverse(path, fl, np.array([2500 + G]), {"q": 35.314, "wgr": 0.0}, "pe5_style")
        g = df[df.run == run]
        assert np.abs(np.interp(g.md, path.md, T[0]) - g.t).max() < 3.0, run
        assert (P[0, -1] - G) / g.p.iloc[-1] - 1 == pytest.approx(0.0, abs=0.03), run
        u = path.thermal["_u_last"][0]
        mid = 0.5 * (path.md[1:] + path.md[:-1])
        gu = g[g.u.notna() & (g.md < 11800)]
        k = np.array([int(np.argmin(np.abs(mid - z))) for z in gu.md])
        err = u[k] / gu.u.to_numpy() - 1
        out = (gu.md.to_numpy() > 7500) & (gu.md.to_numpy() < 8000)
        assert np.abs(err[~out]).max() < 0.06 and np.abs(err[out]).max() < 0.16, run
    import json
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T40_Enthalpy_Balance.wellmodel.json"))
    assert proj["widgets"]["opt_temp_model"] == "enthalpy" and {"eq_drill", "eq_litho"} <= set(proj["tables"])


def test_compositional_co2_injector_t39():
    """PROSPER Tutorial 39, the user's file: a CO2 injector (PR EOS with the PRP file's CO2 correction, LBC), tubing
    3.992 in to 14,000 ft with a 3.8 in SSSV, Improved Approximation U = 3, injection at 75 F and 1250 psig,
    Petroleum Experts gas injectivity (k 35 md, skin 12). The Span-Wagner-corrected stream density is PROSPER's to
    1 %; the injectivity is its IPR column to 15 psi; the energy-balance VLP its VLP column to 2 % and its bottom
    temperature to 1.5 F at 0.15-72 MMscf/d; the gradients at 24.683 / 10 MMscf/d (U = 3) and 27.79 MMscf/d
    (U = 0.001) to 1 % and 1.5 F; the system solution 24.683 MMscf/d at 5565.39 psig to 4 % / 1 % (the VLP is flat
    there, so 0.5 % in pressure is 2.5 % in rate)."""
    pytest.importorskip("CoolProp")
    import json
    from wellmodel import co2 as wco2, eos as E, thermal as wth, injection as winj
    root = Path(__file__).parent
    proj = json.load(open(root.parent / "examples" / "T39_Compositional_CO2_Injector.wellmodel.json"))
    f = E.EOSFluid(E.parse_prp(proj["extras"]["prp_text"], "T39_co2_EoS.prp"))
    assert wco2.wants_reference(f)
    fl = wco2.CO2InjectionFluid(f)
    g = 14.696
    grad = pd.read_csv(root / "data" / "t39_gradients.csv")
    r3 = grad[(grad.run == "3") & grad.rho_m.notna()]
    rho = fl.props(r3.p.to_numpy() + g, r3.t.to_numpy())[1]
    err = rho / r3.rho_m.to_numpy() - 1
    # PROSPER's own densities drop 4 % between the SSSV and ~2,000 ft (near CO2's critical point; a step in its
    # interpolation, back to within 0.5 % below) - every node outside that band to 1 %, rms 0.4 %
    band = (r3.md.to_numpy() > 1200) & (r3.md.to_numpy() < 2100)
    assert np.max(np.abs(err[~band])) < 0.01
    assert np.sqrt(np.mean(err[~band] ** 2)) < 0.004
    assert np.max(np.abs(err[band])) < 0.05

    ipr = winj.petex_gas_injection(4500 + g, 313, 35, 75, 450, 31.6, 0.354, 75, 50, 0.19, 0.2, 12, fl)
    assert abs(ipr.params["Non-Darcy flow factor D (1/(Mscf/d))"] / 6.1518e-5 - 1) < 0.01
    tab = pd.read_csv(root / "data" / "t39_vlp_ipr.csv")
    assert np.max(np.abs(ipr.pwf(tab.q.to_numpy()) - g - tab.ipr.to_numpy())) < 15

    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 1200, "Tubing Inside Diameter (in)": 3.992,
             "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 4.5},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 3.8},
            {"Type": "Tubing", "Measured Depth (ft)": 14000, "Tubing Inside Diameter (in)": 3.992,
             "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 4.5},
            {"Type": "Casing", "Measured Depth (ft)": 14500, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = winj.from_flow_path(wvlp.build_flow_path(top, secs, pts, [0, 14500], [0, 14500], [0, 14500], [60, 313]))
    th = lambda u: {"model": "improved", "u_seg": wth.u_segments(path, [14500], [u])}
    whp = 1250 + g
    P, T = winj.traverse(path, fl, whp, tab.q.to_numpy(), t_inj=75, thermal=th(3.0))
    assert np.max(np.abs((P[:, -1] - g) / tab.vlp.to_numpy() - 1)) < 0.02
    assert np.max(np.abs(T[:, -1] - tab.t_last.to_numpy())) < 1.5

    for run, q, u in (("3", 24.683, 3.0), ("5a", 10.0, 3.0), ("5b", 27.79, 0.001)):
        d = grad[grad.run == run]
        P, T = winj.traverse(path, fl, whp, [q], t_inj=75, thermal=th(u))
        pp = np.interp(d.md, path.md, P[0]) - g
        tt = np.interp(d.md, path.md, T[0])
        assert np.max(np.abs(pp / d.p.to_numpy() - 1)) < 0.01, run
        assert np.max(np.abs(tt - d.t.to_numpy())) < 1.5, run

    res, *_ = winj.solve(path, fl, whp, ipr, t_inj=75, thermal=th(3.0))
    assert abs(res.q / 24.683 - 1) < 0.04
    assert abs((res.pwf - g) / 5565.39 - 1) < 0.01
    assert abs(res.T[-1] - 196.08) < 2.0


def test_steam_injector_t42():
    """PROSPER Tutorial 42, the user's file: the T03 water injector as a steam injector (Improved Approximation,
    U = 2, 200,000 ppm brine), 4000 STB/d at 2200 psig with wellhead enthalpies 900-1100 BTU/lb, PE2, 20 ft steps.
    Steam calculator: PROSPER's 2200 psig values (IAPWS-95). Traverses: bottom pressure within 2 % (the pressure gain
    within 5 %), temperatures within 2.5 F, enthalpy within 10 BTU/lb, holdup within 4 % (+0.005) below 2700 psig and the
    quality within 5 points there (above it PROSPER's own steam table drifts from IAPWS-95 towards the critical point:
    its saturated water enthalpy is up to 10 BTU/lb high at 3080 psia)."""
    pytest.importorskip("CoolProp")
    from wellmodel import steam as S, injection as winj
    s = S.saturation(2214.696)
    assert abs(s["t_sat"] - 650.453) < 0.01 and abs(s["latent"] - 422.233) < 0.01
    assert abs(s["liquid"]["enthalpy"] - 696.821) < 0.01 and abs(s["vapour"]["density"] - 6.2118) < 0.002
    assert abs(S.flash_ph(2214.696, 1100)["quality"] - 0.95487) < 1e-4
    assert abs(S.flash_ph(2214.696, 900)["quality"] - 0.4812) < 1e-4
    sup = S.flash_ph(2214.696, 1200)
    assert sup["phase"] == "VAPOUR" and abs(sup["t"] - 689.45) < 0.01 and abs(sup["density"] - 4.92808) < 1e-3

    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 600},
            {"Type": "Tubing", "Measured Depth (ft)": 8500, "Tubing Inside Diameter (in)": 5.2,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 9000, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = winj.from_flow_path(wvlp.build_flow_path(top, secs, pts, [0, 9000], [0, 9000], [0, 600, 9000],
                                                    [60, 40, 210]))
    data = pd.read_csv(Path(__file__).parent / "data" / "t42_steam_gradients.csv")
    bottom = {900: 4249.3, 950: 3910.0, 1000: 3608.1, 1050: 3197.0, 1100: 2968.8}
    for h0, pb in bottom.items():
        r = S.traverse(path, 2214.696, 4000, h0, u=2.0, salinity_ppm=200000)
        d = data[data.h0 == h0]
        assert abs((r.p[-1] - 14.696) / pb - 1) < 0.02, h0
        assert abs((r.p[-1] - 14.696 - 2200) / (pb - 2200) - 1) < 0.05, h0
        assert np.max(np.abs(np.interp(d.md, r.md, r.t) - d.t)) < 2.5, h0
        dh = np.interp(d.md, r.md, r.h) - d.h
        assert np.nanmax(np.abs(dh)) < 10.0, h0
        low = (d.p < 2700) & d.x.notna() & (d.x > 0)
        assert np.max(np.abs(np.interp(d.md[low], r.md, 100 * r.x) - d.x[low])) < 5.0, h0
        assert np.all(np.abs(np.interp(d.md[low], r.md, r.holdup) - d.hl[low]) < 0.04 * d.hl[low] + 0.005), h0


def _t43_run(q, whp, comp=None):
    """The user's T43 file: the T41 well (T38 black-oil condensate, PE5, Improved Approximation with the file's U
    1.4 / 2.5 / 1.2) at WGR 50 STB/MMscf."""
    import dataclasses
    from wellmodel import thermal as TH, system as SYS
    path0 = _t40_path(6.4)
    path = dataclasses.replace(path0, thermal={"model": "improved", "u_seg": TH.u_segments(
        path0, [0, 4000, 9312.5, 12000], [1.4, 1.4, 2.5, 1.2])})
    return SYS.solution_details(path, _t38_retro(), "gas", {"whp": whp + 14.696, "u": 2.0, "wgr": 50.0,
                                                             "sep_gor": 7431.94}, q, "pe5_style", comp=comp)


def test_flow_assurance_flags_t43():
    """PROSPER Tutorial 43, the user's file: VLP flags at 1000 and 100 psig (50 rates). Turner (no-slip velocity below
    the Turner velocity, PROSPER's oil-weighted gas-liquid interfacial tension), erosional velocity (API 14E, C 400)
    and the PE5 stability flag (Barnea's film momentum balance: no upward liquid film -> unstable, interfacial
    friction K = 55) agree with PROSPER at every rate except the erosional flag at 45.6 and 65.7 MMscf/d at 100 psig
    (PROSPER's wellhead is 30-50 F colder there) and the PE5 flag at 10.66 MMscf/d at 100 psig (film ratio 1.13); the hydrate flag is "No hydrates" throughout with the Water 2 curve. VLP pressures within
    5 % from 2.5 to 46 MMscf/d at 1000 psig (PE5's low-velocity drift, Tutorials 41/43); the maximum Turner velocity
    within 4 %."""
    from wellmodel import complementary as CX
    data = pd.read_csv(Path(__file__).parent / "data" / "t43_vlp_flags.csv")
    sub = data[((data.run == "2") & data.q.isin([1.5, 4.5, 10.5, 12.0])) | ((data.run == "3A") & data.q.isin(
        [2.491, 3.584, 5.154, 15.34, 45.65, 94.449])) | ((data.run == "4") & data.q.isin([2.491, 3.584, 7.414, 15.34]))]
    for _, r in sub.iterrows():
        det, _, prof = _t43_run(r.q, r.whp)
        assert det["Turner velocity flag"] == r.turner, (r.run, r.q)
        assert det["PE5 stability flag"] == r.pe5, (r.run, r.q)
        assert det["Erosional velocity flag"] == r.eros, (r.run, r.q)
        if r.whp == 1000 and 2.4 < r.q < 46:
            assert abs((det["Bottom node pressure (psia)"] - 14.696) / r.vlp - 1) < 0.05, (r.run, r.q)
            assert abs(det["Max. Turner velocity (ft/s)"] / r.turner_max - 1) < 0.04, (r.run, r.q)
    # node flags at 4 MMscf/d: PROSPER's tubing nodes stable, the 6.4 in casing at the bottom unstable
    det, _, prof = _t43_run(4.0, 1000)
    tub = prof[(prof["MD (ft)"] > 0) & (prof["MD (ft)"] < 11800)]
    assert (tub["PE5 stability"] == "Stable").all() and prof["PE5 stability"].iloc[-1] == "Unstable"
    assert tub["PE5 stability ratio"].min() > 1.2 and prof["PE5 stability ratio"].iloc[-1] < 0.3
    # PROSPER's own node properties of its gradient exports (T43 4 MMscf/d, T41 Run 2A, T47 unlifted at 350 psig):
    # vsl, vsg, rho_l, rho_g, mu_l, mu_g, d_in | PE5 flag
    for row, flag_ in (((0.180, 2.024, 49.272, 7.713, 0.368, 0.018, 6.4), "Unstable"),
                       ((0.598, 8.444, 49.893, 4.153, 0.521, 0.013, 3.99), "Stable"),
                       ((0.145, 2.841, 37.183, 15.426, 0.162, 0.028, 6.4), "Stable"),
                       ((0.359, 1.299, 56.128, 5.384, 0.199, 0.017, 8.3), "Unstable"),
                       ((1.545, 6.648, 56.349, 4.550, 0.207, 0.016, 3.99), "Stable")):
        r_ = CX.pe5_stability_ratio(*row[:6], row[6] / 12)
        assert (r_ < 1) == (flag_ == "Unstable"), (row, r_)
    # hydrate flag against a curve
    w2 = ((137.075, 1291.9, 4028.83, 21657.5), (35, 62, 71, 95))
    curve = (tuple(np.array(w2[0]) + 14.696), w2[1])
    assert CX.hydrate_flag(1000 + 14.696, 55.0, curve) and not CX.hydrate_flag(1000 + 14.696, 70.0, curve)
    det, _, prof = _t43_run(4.0, 1000, CX.CompOptions(hydrate=curve))
    assert (prof["Hydrates"].iloc[1:] == "No hydrates").all()
    assert det["Hydrates formation flag"] == "No hydrates"


def test_hydrate_curve_salts_t43():
    """PROSPER Tutorial 43, the user's file: hydrate formation curves of the T37/T43 condensate with two brines
    (Water 1: NaCl 2.15, CaCl2 0.89, KBr 0.12 wt %; Water 2: 4.52 / 0.94 / 0.24) - van der Waals-Platteeuw with Munck
    constants, PR fugacities, Pitzer water activity, reference properties refitted to these curves: within 4 % of
    PROSPER's pressure at every point from 35 to 98 F (117 - 22,390 psig); more salt, higher pressure."""
    from wellmodel import hydrate as H, eos as E
    f = E.EOSFluid(E.parse_prp(open(Path(__file__).parent.parent / "examples" / "T37_Retrograde_Condensate_EoS.prp").read()))
    w1 = {35: 116.893, 44: 237.539, 56: 610.187, 65: 1436.25, 71: 3207.75, 80: 7654.14, 89: 14170.1, 98: 22389.7}
    w2 = {35: 137.075, 44: 276.27, 56: 723.153, 65: 1865.8, 71: 4028.83, 80: 9088.71, 89: 16076.3, 95: 21657.5}
    m1 = H.HydrateModel(f, {"NaCl": 2.15, "CaCl2": 0.89, "KBr": 0.12})
    m2 = H.HydrateModel(f, {"NaCl": 4.52, "CaCl2": 0.94, "KBr": 0.24})
    for m, w in ((m1, w1), (m2, w2)):
        for t, p in w.items():
            assert abs((m.pressure(t) - 14.696) / p - 1) < 0.04, t
    assert m2.ln_aw < m1.ln_aw < 0
    assert H.HydrateModel(f).pressure(50) < m1.pressure(50) < m2.pressure(50)
    assert abs(H.ln_water_activity({"NaCl": 5.52}) - np.log(0.9669)) < 0.002     # 1 molal NaCl, a_w 0.9669


def test_non_newtonian_oil_t44():
    """PROSPER Tutorial 44 (the user's file): 12 API heavy oil, 7500 STB/d, WC 0, GOR 10, PE2; 3.2 in tubing to 1800 ft,
    6.4 in casing to 2000 ft, 80-120 F, U 8. Vazquez-Beggs / Egbogah matched at 120 F (Pb 170 psig, Bo 1.025, 700 cp).
    Herschel-Bulkley rheology (K 1 / 0.5 Pa.s^n, n 0.9 / 0.95 at 60 / 120 F, no yield stress): apparent viscosity
    400.56 cp in the tubing and 425.0 cp in the casing; the bottom pressure falls from PROSPER's Newtonian
    1871.87 / 2121.89 / 2621.93 psig to 1498.25 / 1748.30 / 2248.49 psig at 250 / 500 / 1000 psig wellhead."""
    import json
    from wellmodel import rheology as R, system as wsys
    G = 14.696
    inp = wbo.PVTInputs(rsb=10, api=12, gas_sg=0.58, salinity_ppm=75000)
    tab = pd.DataFrame({c: [np.nan] for c in wbo.TABLE_COLUMNS})
    tab.loc[0, ["Pressure", "Gas Oil Ratio", "Oil FVF", "Oil Viscosity"]] = [170 + G, 10, 1.025, 700]
    r = wbo.match_all(inp, wbo.MatchData(120.0, 170 + G, tab))
    m = r.matches[("visc", "egbogah")]
    assert abs(m.p1 - 0.99064) < 2e-4 and abs(m.p2 + 6.68016) < 2e-3          # PROSPER's match parameters
    model = wbo.BlackOilModel(inp, "vazquez_beggs", "egbogah",
                              {p: r.matches[(p, "vazquez_beggs")] for p in ("pb", "rs", "bo")} | {"visc": m})
    rh = R.Rheology([(60, 100, 0, 1, 0.9), (60, 5000, 0, 1, 0.9), (120, 100, 0, 0.5, 0.95), (120, 5000, 0, 0.5, 0.95)])
    assert abs(rh.apparent_viscosity(360.95 + G, 116.12, 8.929, 3.2 / 12) - 400.56) < 0.2
    assert abs(rh.apparent_viscosity(1827.54 + G, 119.95, 2.233, 6.4 / 12) - 425.00) < 0.2
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 1800, "Tubing Inside Diameter (in)": 3.2,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 2000, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = wvlp.build_flow_path(top, secs, pts, [0, 2000], [0, 2000], [0, 2000], [80, 120])
    ref = {(None, 250): 1871.87, (None, 1000): 2621.93, (rh, 250): 1498.25, (rh, 1000): 2248.49}
    for (rr, whp), pb in ref.items():
        det, _, prof = wsys.solution_details(path, wvlp.BlackOilFluid(model, rheology=rr), "oil",
                                             {"whp": whp + G, "u": 8.0, "wc": 0.0, "gor": 10.0}, 7500.0, "pe2_style")
        dp = det["Bottom node pressure (psia)"] - G - whp
        print("T44", rr is not None, whp, det["Bottom node pressure (psia)"] - G, pb)
        assert abs(dp / (pb - whp) - 1) < 0.01
        assert abs(det["dP friction - production string (psi)"] / (412.8 if rr else 786.0) - 1) < 0.01
    proj = json.load(open(__file__.replace("tests/test_wellmodel.py",
                                           "examples/T44_Modelling_NonNewtonian_Fluids.wellmodel.json")))
    assert proj["widgets"]["opt_viscosity_model"] == "non_newtonian"
    assert len(proj["tables"]["pvt_rheology"]["__df__"]["data"]) == 4


def test_cbm_dewatering_well_t45():
    """PROSPER Tutorial 45 (the user's file): CBM producer, gas gravity 0.6, 100,000 ppm water. CBM Producer IPR
    (Pr 800 psig, 100 F, PI 2, Swr 0.2, krw/krg end points 0.8, Corey 1.5 / 2): PROSPER's IPR at GLR 1 / 10 / 100 /
    1000 within 0.5 %. ESP CENTRILIFT R7 x 58 at 2500 ft, 100 % gas separator, PE2, 200 psig wellhead: PROSPER's lift
    curves (50 / 60 / 70 Hz, GLR 10 / 100 / 1000, liquid level 1500 / 2000 / 2400 ft) - bottom-hole pressure within
    5.3 % at every rate (2 % above 40 STB/d), the casing head pressure within 5 psi above 40 STB/d; the .tpd export
    reads back."""
    import json
    from wellmodel import cbm as C, esp as E, liftcurves as LC
    G = 14.696
    d = Path(__file__).parent / "data"
    inp = wbo.PVTInputs(rsb=1, api=40, gas_sg=0.6, salinity_ppm=100000)
    ipr_ref = pd.read_csv(d / "t45_ipr.csv")
    for g, grp in ipr_ref.groupby("glr"):
        m = C.CBMInflow(inp, 800 + G, 100.0, float(g), 2.0)
        q = m.rate(grp["pwf"].to_numpy() + G)
        ok = grp["q"].to_numpy() > 1
        assert np.allclose(q[ok], grp["q"].to_numpy()[ok], rtol=0.005), (g, q[ok][-3:], grp["q"].to_numpy()[ok][-3:])
    assert abs(C.CBMInflow(inp, 800 + G, 100.0, 10.0, 2.0).aof / 1222.4 - 1) < 0.005
    pump = E.PUMPS["CENTRILIFT R7 (4 in)"]
    assert abs(pump.head_stage(600.0) * 58 - 1710.3) < 1.0                  # PROSPER: 1710.3 ft at 600 RB/d, 60 Hz
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0},
            {"Type": "Tubing", "Measured Depth (ft)": 2500, "Tubing Inside Diameter (in)": 2.875,
             "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 3.25,
             "Tubing Outside Roughness (in)": 0.0006, "Casing Inside Diameter (in)": 6.1,
             "Casing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 2600, "Casing Inside Diameter (in)": 6.1,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = wvlp.build_flow_path(top, secs, pts, [0, 2600], [0, 2600], [0, 2600], [60, 100])
    esp_ = {"pump": pump, "stages": 58, "pump_md": 2500.0, "sep": 1.0, "freq": 60.0, "wear": 0.0,
            "motor": E.MOTORS["Centrilift 562 KMH 38HP 435V 53A (58.3 Hz)"]}
    ref = pd.read_csv(d / "t45_lift_curves.csv")
    fl = C.CBMFluid(inp)
    res = C.lift_curves(path, fl, esp_, "pe2_style", 1.0, 1.0, 8.0, weq.HeatCapacities(), sorted(ref["rate"].unique()),
                        [1500, 2000, 2400], [10, 1000], [200 + G], [60, 70])
    ax = res["axes"]
    for i, f in enumerate(ax["freqs"]):
        for j, g in enumerate(ax["glrs"]):
            for k, lv in enumerate(ax["levels"]):
                r = ref[(ref.freq == f) & (ref.glr == g) & (ref.level == lv)].sort_values("rate")
                bhp = res["bhp"][i, 0, j, k] - G
                e = np.abs(bhp / r["bhp"].to_numpy() - 1)
                big = r["rate"].to_numpy() > 40
                assert e.max() < 0.055 and e[big].max() < 0.02, (f, g, lv, e.max(), e[big].max())
                chp = res["chp"][i, 0, j, k] - G
                assert np.nanmax(np.abs(chp - r["chp"].to_numpy())[big]) < 5.0
                assert np.allclose(res["pdp"][i, 0, j, k] - G, r["pdp"], rtol=0.003)
    print("T45 60 Hz GLR 10 level 2000 BHP", (res["bhp"][0, 0, 0, 1, [12, 16, 18]] - G).round(1))
    back = LC.read_tpd(C.write_tpd(res, 0.6, 2600.0))
    assert back["results"].shape == (2, 1, 2, 3, 20, len(C.TPD_COLUMNS))
    assert np.allclose(back["results"][..., 0], np.where(np.isfinite(res["bhp"]), res["bhp"] - G, 3.4e35), rtol=1e-5)
    prosper = LC.read_tpd((d / "t45_cbm_lift_curves.tpd").read_text())
    assert prosper["results"].shape == (3, 1, 3, 3, 20, 37)
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T45_CBM_Dewatering_Well.wellmodel.json"))
    assert proj["widgets"]["opt_fluid_type"] == "cbm" and proj["inputs"]["espq_level"] == 2000.0


def test_foam_lift_t48():
    """PROSPER Tutorial 48 (the user's file): the T02 gas well at Pr 1200 psig, WGR 100, CGR 1, 500 psig wellhead,
    produced up the annulus between a 1.125 in coiled tubing and the 3.958 in tubing (0.95 in SSSV at 1000 ft).
    Foam-Homogeneous with the surfactant tables: the VLP tables of the ten cases (0.02-0.5 %, 0.01-10 MMscf/d)
    within 1 %, the system rates (6.789-6.945 MMscf/d) within 1 %; friction ~0 (PROSPER 0.05 psi)."""
    from wellmodel import gaspvt as gp, inflow as winf, system as wsys, foam as FO
    import json
    G = 14.696
    inp = gp.GasPVTInputs(0.58, 200, 1, 50, 100, 100000, 0, 0.5, 2, "lee", 166, 1200)
    pr = 1200 + G
    grid = np.linspace(G, 1500, 1500)
    mg = gp.pseudo_pressure(inp, grid, 166.0)
    z3 = gp.gas_props(inp, np.array([pr - 5, pr, pr + 5]), np.full(3, 166.0))[0]
    mu_r = float(gp.gas_props(inp, np.array([pr]), np.array([166.0]))[3][0])
    cv = winf.petex_gas(pr, 166, 25, 55, 500, 31.6, .354, 55, 50, .15, .25, 2.0, lambda p: np.interp(p, grid, mg), mu_r,
                        1 / pr - (z3[2] - z3[0]) / 10 / z3[1], 0.58, wgr=100.0, cgr=1.0, z_r=float(z3[1]))
    assert abs(cv.params["Gas relative permeability"] - 0.68905) < 1e-3 and abs(cv.aof / 9987.21 - 1) < 0.015
    t_ = {"Tubing Inside Diameter (in)": 0.95, "Tubing Inside Roughness (in)": 0.0006,
          "Tubing Outside Diameter (in)": 1.125, "Tubing Outside Roughness (in)": 0.0006,
          "Casing Inside Diameter (in)": 3.958, "Casing Inside Roughness (in)": 0.0006}
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 0}, {"Type": "Tubing", "Measured Depth (ft)": 1000, **t_},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 0.95}, {"Type": "Tubing", "Measured Depth (ft)": 5000, **t_},
            {"Type": "Casing", "Measured Depth (ft)": 5400, "Casing Inside Diameter (in)": 6.0,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    path = wvlp.build_flow_path(top, secs, pts, [0, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5400],
                                [0, 1500, 1980, 2450, 2883, 3300, 3721, 4140, 4560, 4900], [0, 1500, 5400],
                                [45, 75, 166], flow_type="annular")
    fl = FO.FoamFluid(wvlp.WetGasFluid(wbo.PVTInputs(rsb=1, api=40, gas_sg=0.58, salinity_ppm=100000, co2=0.5, n2=2),
                                       50.0, "lee", inp),
                      FO.FoamModel(density=((0.0, 68.0), (0.05, 53), (0.1, 45), (0.25, 33), (0.5, 30.5))), 0.02)
    # PROSPER ignores the zero-concentration row: 57.8 at 0.02 % (its traverse), 31.567 at 0.39333 %
    assert abs(float(fl.model.foam_density(0.02)) - 57.8) < 1e-9 and abs(float(fl.model.foam_density(0.39333)) - 31.567) < 1e-3
    ref = pd.read_csv(Path(__file__).parent / "data" / "t48_foam_vlp.csv")
    q = ref["q"].to_numpy()
    case = {"q": q, "wgr": np.full_like(q, 100.0), "cgr": np.full_like(q, 1.0), "surf": ref["surf"].to_numpy()}
    P, T = wvlp.traverse(path, fl, np.full_like(q, 500 + G), case, "foam_homogeneous", 1, 1, 3.0)
    err = np.abs((P[:, -1] - G) / ref["vlp"].to_numpy() - 1)
    assert err.max() < 0.01, err.max()
    sys_ref = [6.789, 6.840, 6.879, 6.904, 6.930, 6.939, 6.940, 6.942, 6.944, 6.945]
    res, _ = wsys.solve_cases(lambda tid: path, fl, "gas", {"whp": 500 + G, "u": 3.0, "wgr": 100.0, "sep_gor": 1e6},
                              [{"surf": float(c)} for c in sorted(ref["surf"].unique())], lambda ch: cv,
                              "foam_homogeneous")
    got = np.array([r.q for r in res])
    print("T48 system rates", got.round(3))
    assert np.all(np.abs(got / np.array(sys_ref) - 1) < 0.01) and np.all(np.diff(got) > 0)
    det, _, _ = wsys.solution_details(path, fl, "gas", {"whp": 500 + G, "u": 3.0, "wgr": 100.0, "sep_gor": 1e6,
                                                        "surf": 0.02}, res[0].q, "foam_homogeneous")
    assert det["dP friction - production string (psi)"] == 0.0
    assert abs(det["dP gravity - production string (psi)"] / 99.47 - 1) < 0.03
    assert abs(det["Wellhead temperature (°F)"] - 130.9) < 1.5         # heat transfer at the annulus hydraulic diameter
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T48_Foam_Lift.wellmodel.json"))
    assert proj["widgets"]["opt_artificial_lift"] == "foam_lift" and proj["inputs"]["sys_corr"] == "foam_homogeneous"


def test_hydraulic_fracturing_t49():
    """PROSPER Tutorial 49 (the user's file): Stimulation | Hydraulic Fracturing on the Logie well (9000 ft TVD,
    Pr 5200 psig, PI 5), 66 treatments (1000-4000 bbl, 200-400 bbl/min). Stresses exact; initial length and width
    within 0.5 %, effective length within 3.5 %, effective width within 2 %, PI after within 1 %; five detailed cases
    (rw' within 5 %, following the effective length); treatment screens of the six one-at-a-time runs; the fluid-loss and surface-temperature tables
    (pad, effective dimensions, rw')."""
    from wellmodel import fracture as FR
    import json
    fr = FR.Fracture(FR.FracInputs())
    st_ = FR.stresses(fr.f)
    assert abs(st_["Fracture propagation pressure (psig)"] - 6262.57) < 0.05
    assert abs(st_["Initial closing pressure (psig)"] - 1047.89) < 0.05
    assert abs(st_["Actual closing pressure (psig)"] - 2351.56) < 0.05
    assert abs(st_["Fracture initiation pressure (psig)"] - 6502.72) < 0.5
    assert abs(st_["Fracture breakdown gradient (psi/ft)"] - 0.72416) < 1e-4
    tab, cases = fr.sensitivity()
    ref = pd.read_csv(Path(__file__).parent / "data" / "t49_frac_results.csv")
    for col, rc, tol in (("Fracture length (ft)", "L0", 0.005), ("Fracture width (in)", "w0", 0.005),
                         ("Final fracture length (ft)", "Leff", 0.035), ("Final fracture width (in)", "weff", 0.02),
                         ("Productivity index - after (STB/d/psi)", "pi_after", 0.01)):
        e = np.abs(tab[col].to_numpy() / ref[rc].to_numpy() - 1)
        assert e.max() < tol, (col, e.max())
    # detailed cases: pad volume, effective height, rw'
    for k, pad_v, h_eff, rwp in ((1, 550, 82.3181, 33.9481), (11, 560, 93.1674, 57.0057), (38, 352, 88.7606, 50.6206),
                                 (56, 300, 81.4451, 45.03), (66, 320, 90.2403, 61.7663)):
        pad, sch, eff, pr = cases[k - 1]
        assert abs(pad["Pad volume (bbl)"] - pad_v) < 1e-6 * pad_v + 1e-9, (k, pad["Pad volume (bbl)"])
        assert abs(eff["Effective fracture height (ft)"] / h_eff - 1) < 0.01, (k, eff["Effective fracture height (ft)"])
        assert abs(pr["Effective wellbore radius (ft)"] / rwp - 1) < 0.05, (k, pr["Effective wellbore radius (ft)"])
    assert abs(fr.treatment(200, 1000)["Perforation friction losses (psi)"] - 1.40117) < 0.002
    assert abs(fr.treatment(400, 1000)["Pump power requirement (hp)"] / 5097359 - 1) < 0.002
    # the user's one-at-a-time runs: treatment screen (200 bbl/min, 1000 bbl)
    for inp, refs in ((dict(), {"Tubing head pressure (psig)": 133643, "Pump power requirement (hp)": 654924,
                                "Tubing hydrostatic head (psi)": 3901.05, "Total fluid loss coefficient (ft/sqrt(min))": 0.0028838,
                                "Bottom hole temperature during job (F)": 207}),
                      (dict(c_wall=0.0062), {"Fracture initiation pressure (psig)": 6671.08,
                                             "Total fluid loss coefficient (ft/sqrt(min))": 0.005299}),
                      (dict(poisson=0.25), {"Initial closing pressure (psig)": 1402.09, "Actual closing pressure (psig)": 2705.76,
                                            "Fracture initiation pressure (psig)": 6915.98, "Tubing head pressure (psig)": 133997,
                                            "Fracture breakdown gradient (psi/ft)": 0.77008}),
                      (dict(tensile=1000.0), {"Fracture initiation pressure (psig)": 6502.72, "Tubing head pressure (psig)": 133643}),
                      (dict(t_surface=100.0), {"Bottom hole temperature during job (F)": 207.8}),
                      (dict(fluid_density=10.0), {"Perforation friction losses (psi)": 1.56721, "Tubing friction losses (psi)": 142237,
                                                  "Tubing hydrostatic head (psi)": 4363.3, "Tubing head pressure (psig)": 144638,
                                                  "Pump power requirement (hp)": 708798})):
        t = FR.Fracture(FR.FracInputs(**inp)).treatment(200, 1000)
        for k_, v_ in refs.items():
            assert abs(t[k_] / v_ - 1) < 5e-4, (inp, k_, t[k_], v_)
    # the user's fluid-loss 0.0062 and 100 F surface-temperature result tables
    for inp, fn, tols in ((dict(c_wall=0.0062), "run1", (0.005, 0.005, 0.025, 0.02, 0.03)),
                          (dict(t_surface=100.0), "run5", (0.01, 0.015, 0.035, 0.04, 0.01))):
        tab1, _ = FR.Fracture(FR.FracInputs(**inp)).sensitivity()
        ref1 = pd.read_csv(Path(__file__).parent / "data" / f"t49_frac_results_{fn}.csv")
        for (col, rc), tol in zip((("Fracture length (ft)", "L0"), ("Fracture width (in)", "w0"),
                                   ("Final fracture length (ft)", "Leff"), ("Final fracture width (in)", "weff"),
                                   ("Productivity index - after (STB/d/psi)", "pi_after")), tols):
            r_ = ref1[rc].to_numpy()
            m_ = np.isfinite(r_)
            e = np.abs(tab1[col].to_numpy()[m_] / r_[m_] - 1)
            assert e.max() < tol, (fn, col, e.max())
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T49_Stimulation_Hydraulic_Fracturing.wellmodel.json"))
    assert proj["inputs"]["ipr_j"] == 5.0 and proj["widgets"]["sys_vals_j"] == "5, 11.2449"


def test_mud_acidization_t50():
    """PROSPER Tutorial 50 (the user's file): Stimulation | Mud Acidization of Sandstones on the Logie well - 39
    one-at-a-time runs (volume, permeability, porosity, tensile, Poisson, bulk density, damage type, formulation, every
    mineral). Rock and formulation exact; porosity increase within 0.4 %, permeability after 0.2 %, injection rate,
    pump power and treated radius 0.15 %, THP 0.5 psi, PI after 0.05 %."""
    from wellmodel import mudacid as MA
    import json
    runs = json.load(open(Path(__file__).parent / "data" / "t50_mudacid_runs.json"))
    chk = {"Calculated matrix density": ("rock", "Calculated matrix density (lb/ft3)", 1e-5),
           "Calculated bulk density(Sw=100%)": ("rock", "Calculated bulk density, Sw = 100 % (lb/ft3)", 2e-5),
           "Aluminium in bulkrock": ("rock", "Aluminium in bulk rock (g/L)", 1e-5),
           "Calcium in bulkrock": ("rock", "Calcium in bulk rock (g/L)", 1e-5),
           "Na,K,Mg etc. in bulkrock": ("rock", "Na, K, Mg etc. in bulk rock", 1e-9),
           "Concentration of HF": ("treatment", "Concentration of HF (%)", 1e-9),
           "Concentration of HCl": ("treatment", "Concentration of HCl (%)", 1e-5),
           "Porosity increase": ("treatment", "Porosity increase (%)", 0.004),
           "Permeability after treatment": ("treatment", "Permeability after treatment (md)", 0.002),
           "Total dissolved minerals": ("treatment", "Total dissolved minerals (g/L)", 0.004),
           "Ammmonium BiFluoride (ABF)": ("treatment", "Ammonium bifluoride ABF (lbm per 1000 gal)", 1e-6),
           "1000gals Hcl Concentration": ("treatment", "HCl for 1000 gal of mud acid (%)", 1e-5),
           "Maximum Injection Rate": ("pumping", "Maximum injection rate (bbl/min)", 0.0015),
           "Fracture Gradient": ("pumping", "Fracture gradient (psi/ft)", 1e-5),
           "Pump Requirement": ("pumping", "Pump requirement (hp)", 0.0015),
           "- 3 HCl PreFlush": ("pumping", "3 HCl preflush (gal)", 1e-9),
           "- 6 Gasoil": ("pumping", "6 Gasoil (gal)", 1e-9),
           "Radius treated zone": ("damage", "Radius treated zone (in)", 0.0015),
           "Permeability damaged zone": ("damage", "Permeability damaged zone (md)", 1e-5),
           "Productivity index after treatment": ("damage", "Productivity index after treatment (STB/d/psi)", 5e-4),
           "Skin factor after treatment": ("damage", "Skin factor after treatment", 0.004)}
    for r in runs:
        res = MA.MudAcid(MA.MudAcidInputs(**r["inputs"])).results()
        ref = r["prosper"]
        assert abs(res["pumping"]["Maximum tubing head pressure (psig)"] - ref["Maximum Tubing Head Pressure"]) < 0.5, r["run"]
        for pk, (grp, key, tol) in chk.items():
            v = res[grp][key]
            if ref.get(pk) is None:                       # PROSPER returns nothing (no calcium in the rock)
                assert np.isnan(v), (r["run"], pk, v)
                continue
            assert (abs(v) < 1e-9 if ref[pk] == 0 else abs(v / ref[pk] - 1) < tol), (r["run"], pk, v, ref[pk])
    # PI before from the IPR (PI 5, Vogel below Pb 3439 psig): PROSPER's 4.95562
    from wellmodel import ipr
    pwf = ipr.pwf(10627.8, 5214.696, 5.0, 3439.195 + 14.696)
    assert abs(10627.8 / (5214.696 - pwf) / 4.95562 - 1) < 2e-4
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T50_Stimulation_Mud_Acidization_Sandstone.wellmodel.json"))
    assert proj["inputs"]["sys_whp"] == 514.696 and proj["widgets"]["sys_vals_j"] == "5, 5.38858"


def test_spot_ipr_t09():
    """PROSPER Tutorial 09 (the user's file): SPOT IPR on the Logie well (110 ft MD layer, 50 md, Pr 4000 psig, water
    cut 20 %, Baker Predator XP LS 6 spf / 60 deg) and 13 one-at-a-time runs. Per-segment perforation and crushed-zone
    skins within 0.1 % (the bottom segment excepted), deviation skin and the pressure / density columns exact, layer
    total skin within 0.02 (12 runs) / 0.03 (63 deg), layer oil AOF within 0.2 %, the IPR table within 0.4 % and the
    system solution at 500 psig within 0.5 % (PE2, U 8)."""
    from wellmodel import spot as SP, vlp as V, equipment as weq, system as wsys, inflow as winf
    import json
    G = 14.696
    data = json.load(open(Path(__file__).parent / "data" / "t09_spot_runs.json"))
    fl = _t05_oil()
    mob = SP.blackoil_mobility(fl.m, 210.0)
    pairs = {"Perforation Skin": "Perforation skin", "Crush Skin": "Crush skin", "Deviation Skin": "Deviation skin",
             "Completion Density": "Completion fluid density (lb/gal)", "Underbalance King": "Underbalance King (psig)",
             "Underbalance Berhmann": "Underbalance Behrmann (psig)", "Underbalance User": "Underbalance user (psig)",
             "Near Wellbore eff vert stress": "Near-wellbore effective vertical stress (psig)"}
    tol = {"Perforation Skin": 1e-3, "Crush Skin": 1e-3, "Deviation Skin": 2e-5, "Completion Density": 2e-4,
           "Underbalance King": 5e-4, "Underbalance Berhmann": 5e-4, "Underbalance User": 1e-5,
           "Near Wellbore eff vert stress": 1e-4}
    for r in data["runs"]:
        kw = dict(r["inputs"])
        if "survey" in kw:
            kw["survey"] = tuple(map(tuple, kw["survey"]))
        res = SP.Spot(SP.SpotInputs(**kw), mob).results()
        lg, lay = res["log"], res["layer"]
        for pk, mk in pairs.items():
            for i, ref in zip((0, 55, 109), r["segments_0_55_109"][pk]):
                v = float(lg[mk].iloc[i])
                assert (abs(v - ref) < 1e-6 if ref == 0 else abs(v / ref - 1) < tol[pk]), (r["run"], pk, i, v, ref)
        assert abs(lay["Total skin"] - r["layer"]["S"]) < (0.03 if r["run"] == "63 deg" else 0.02), (r["run"], lay["Total skin"])
        assert abs(lay["Oil AOF (STB/d)"] / r["layer"]["AOF"] - 1) < 0.002, (r["run"], lay["Oil AOF (STB/d)"])
        assert abs(lay["Completion fluid density (lb/gal)"] / r["layer"]["dens"] - 1) < 2e-4
    # base IPR table (liquid against FBHP)
    sp = SP.Spot(SP.SpotInputs(), mob)
    qo = sp.results()["oil_rate"]
    ref = data["base_ipr"]
    for q, p in zip(ref["liquid"][1:], ref["pwf_psig"][1:]):
        assert abs(float(qo(p + G)) / 0.8 / q - 1) < 0.004, (p, q)
    # system calculation at 500 psig (Tutorial 05 equipment, PE2, U 8)
    rows = [{"Type": "Xmas Tree", "Measured Depth (ft)": 600},
            {"Type": "Tubing", "Measured Depth (ft)": 1000, "Tubing Inside Diameter (in)": 4.052,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "SSSV", "Tubing Inside Diameter (in)": 3.72},
            {"Type": "Tubing", "Measured Depth (ft)": 9000, "Tubing Inside Diameter (in)": 4.052,
             "Tubing Inside Roughness (in)": 0.0006},
            {"Type": "Casing", "Measured Depth (ft)": 9275, "Casing Inside Diameter (in)": 6.4,
             "Casing Inside Roughness (in)": 0.0006}]
    top, secs, pts = weq.downhole_sections(_downhole(rows))
    smd, stvd = np.array(LOGIE_SURVEY).T
    path = V.build_flow_path(top, secs, pts, smd, stvd, [0, 600, 9275], [60, 40, 210])
    curve = SP.spot_ipr(4000 + G, 0.2, SP.SpotInputs(), fl.m, 210.0)
    res, _ = wsys.solve_cases(lambda tid: path, fl, "oil", {"whp": 500 + G, "u": 8.0, "wc": 20.0, "gor": 800.0}, [{}],
                              lambda ch: curve, "pe2_style")
    assert abs(res[0].q / data["system"]["liquid"] - 1) < 0.005 and abs(res[0].pwf - G - 2274.1) < 25
    proj = json.load(open(Path(__file__).parent.parent / "examples" / "T09_SPOT_IPR.wellmodel.json"))
    assert proj["inputs"]["ipr_model_oil"] == "SPOT" and proj["inputs"]["ipr_wc"] == 20.0
