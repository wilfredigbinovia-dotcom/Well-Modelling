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


def test_condensate_black_oil_warns():
    o = WellOptions(fluid_type="retrograde_condensate")
    assert any(i.level == "warning" and "method" in i.fields for i in o.validate())


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
    """Rows from a PROSPER PE5 gradient export (Abura 2S) at PROSPER's own conditions (roughness 0.0006 in)."""
    from wellmodel import correlations as C
    # vsl, vsg, rho_l, rho_g, mu_l, mu_g, sigma, d_in, angle, p | holdup, static, friction (PROSPER PE5)
    rows = [(4.026, 3.315, 52.153, 2.169, 3.1533, 0.0124, 11.339, 2.44, 0.61, 686.37, 0.558753, 0.209, 0.033),
            (4.146, 1.384, 51.12, 3.763, 2.314, 0.0137, 8.03, 2.44, 0.61, 1187.35, 0.760532, 0.276, 0.022)]
    for vsl, vsg, rl, rg, ml, mg, sg, d, ang, p, hl_p, gs_p, gf_p in rows:
        a = [np.array([x]) for x in (vsl, vsg, rl, rg, ml, mg, sg)]
        e, f, hl = C.pe5_style(*a, d / 12, 0.0006 / 12, np.cos(np.radians(ang)), np.array([p]))
        assert hl[0] == pytest.approx(hl_p, abs=0.002)
        assert e[0] == pytest.approx(gs_p, abs=0.002)
        assert f[0] == pytest.approx(gf_p, rel=0.05)
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
        assert abs(v / book - 1) < 0.04
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
    assert abs(sol["Flowing BHP (psig)"] - 1142.7) < 15
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


def test_esp_pump_in_traverse_and_lift_curves():
    from wellmodel import esp as E, liftcurves as LC
    *_, path, fluid = _vlp_setup()
    esp = {"pump_md": 9000.0, "pump": E.PUMPS["Centrilift GC10000 (5.13 in)"], "stages": 60, "freq": 60.0,
           "wear": 0.0, "sep": 0.0, "chp": 264.7}
    case = {"q": np.array([3000.0, 6000.0]), "wc": np.array([0.5, 0.5]), "gor": np.array([400.0, 400.0])}
    o = E.bhp_with_pump(path, fluid, 300.0, case, esp, "hagedorn_brown")
    assert np.all(np.isfinite(o["bhp"])) and np.all(o["pdp"] > o["pip"]) and np.all(o["bhp"] > o["pip"])
    # the intake solve inverts the stage march
    pdp, *_ = E.pump_gain(fluid, o["pip"], o["t_intake"], case, esp["pump"], 60, 60.0)
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
    # system at 500 psig wellhead with PROSPER's PE2 match: 16283.4 STB/d at 3143.2 psig
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
                             "pe2_style", 1.03108, 1.00278, 8.580277)
        return float(P[0, -1])
    q = brentq(lambda q_: vlp(q_) - np.interp(q_, c.q, c.pwf), 8000, 25000, xtol=0.5)
    # within 2 %: the PE2 approximation's slug holdup and friction were recalibrated on PROSPER's T23 gas-lift
    # traverses (80 % water cut); at this well's 5 % water cut and very high gas velocities the rate is 1.6 % low
    # and the bottom-hole pressure 2 % high
    assert abs(q / 16283.4 - 1) < 0.02 and abs((vlp(q) - G) / 3143.2 - 1) < 0.025
    split = iadv.layer_split(c, vlp(q))
    for lay_, ref in (("Layer 1", 4384.6), ("Layer 3", 4854.3), ("Layer 5", 7044.4)):
        assert abs(float(split.loc[split.Layer == lay_, "Liquid rate (STB/d)"].iloc[0]) / ref - 1) < 0.02
    assert abs(float(split.loc[split.Layer == "Total", "Liquid rate (STB/d)"].iloc[0]) - q) < 0.01 * q
    # the system analysis takes the water cut and GOR from the layers at each rate (the case's own are ignored)
    from wellmodel import system as wsys
    res, _ = wsys.solve_cases(lambda tid: path, oil, "oil", {"whp": 500 + G, "u": 8.580277, "wc": 50.0, "gor": 300.0},
                              [{}], lambda ch: c, "pe2_style", 1.03108, 1.00278)
    r = res[0]
    assert abs(r.q / 16283.4 - 1) < 0.02 and abs(r.params["wc"] - 4.697) < 0.02 and abs(r.params["gor"] - 800) < 1e-6


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
    assert WellOptions(artificial_lift="gas_lift", fluid_type="dry_wet_gas").errors
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
            assert np.interp(z, prof["MD (ft)"], prof["Pressure (psia)"]) - G == pytest.approx(p, abs=15.0), (qgi, z)


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
        assert res[0].q == pytest.approx(q, rel=tol) and res[0].pwf - G == pytest.approx(pwf, abs=25), whp


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
        assert np.allclose(st_.valve_q[k], prosper_q[k], atol=0.08)
        assert np.allclose(st_.valve_pc[k] - G, prosper_pc[k], atol=3.0)
        assert st_.flag[k] == "Multiple"
    assert list(st_.flag[2:]) == ["No Injection", "No Injection"]
    # PROSPER's dome pressures at valve temperature (Winkler & Eads on gauge pressures)
    for pd60, t, p_ in ((1211.11, 167.2, 1509.9), (1195.47, 189.2, 1550.3), (1179.41, 197.8, 1552.0)):
        assert float(wgls.dome_at_t(pd60, t)) == pytest.approx(p_, abs=0.3)


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
        assert sm["Tubing pressure at valve (psia)"] - G == pytest.approx(pt, rel=0.01), n
        assert sm["Casing pressure at valve (psia)"] - G == pytest.approx(pc, abs=2), n
        assert sm["dP across valve, calculated (psi)"] == pytest.approx(dp, rel=0.02), n
        assert sm["Calculated casing head pressure (psia)"] - G == pytest.approx(chp, rel=0.01), n
        assert sm["Calculated critical flow rate (MMscf/d)"] == pytest.approx(qc, rel=0.01), n
        assert sm["Percentage of critical flow rate (%)"] == pytest.approx(pcrit, rel=0.01), n
        assert sm["Critical casing head pressure (psia)"] - G == pytest.approx(chpc, rel=0.02), n
        assert sm["Calculated tubing head pressure (psia)"] - G == pytest.approx(whp, rel=0.03), n
        assert sm["Flowing bottom-hole pressure, top down (psia)"] - G == pytest.approx(bhp, rel=0.01), n
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
