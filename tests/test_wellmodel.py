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
    o = WellOptions(fluid_type="dry_wet_gas", method="black_oil", flow_type="annular", well_type="gas_injector")
    assert not o.errors
    assert not o.computable
    assert o.engine_status()["well_type"] == NOT_YET and o.engine_status()["flow_type"] != NOT_YET
    assert len(o.unsupported) == 1


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
    assert weq.active_columns("Tubing") == {md, tid, te, rm}
    assert weq.active_columns("Tubing", "annular") == {md, tid, te, rm, tod, toe, cid, ce}
    assert weq.active_columns("SSSV") == {tid} and weq.active_columns("Restriction") == {tid, rm}
    assert weq.active_columns("Gas Lift Mandrel") == {tid, rm} == weq.active_columns("SSD")   # like a restriction
    assert weq.active_columns("Casing") == {md, cid, ce, rm} and weq.active_columns("") == set()
    df = pd.DataFrame([{"Type": "Tubing", md: 231.93, tid: 2.441, tod: 2.875},
                       {"Type": "SSSV", md: 200.0, tid: 1.5, rm: 1.0},
                       {"Type": "Casing", md: 10932.0, cid: 8.681, ce: 0.0008},
                       {md: 5.0}]).reindex(columns=weq.DOWNHOLE_COLUMNS)
    out = weq.apply_cell_rules(df)
    assert out.loc[0, te] == 0.0006 and out.loc[0, rm] == 1.0 and np.isnan(out.loc[0, tod])   # defaults; OD locked
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

