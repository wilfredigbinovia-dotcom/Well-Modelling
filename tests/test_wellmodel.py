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
    o = WellOptions(fluid_type="dry_wet_gas", method="black_oil", flow_type="annular")
    assert not o.errors
    assert not o.computable
    assert o.engine_status()["flow_type"] == NOT_YET
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
    assert ("pb", "de_ghetto") not in r.matches
    assert ("visc", "bergman_sutton") not in r.matches
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
    dp = wcomp.sand_control_dp(sc, "oil", {"mu": 0.8, "bo": 1.3, "rho": 45})
    c2 = wcomp.apply_sand_control(c, dp)
    assert c2.aof < c.aof
    i = 30                                   # sand-face pressure reproduces the original IPR
    assert c.rate(c2.pwf[i] + c2.dp_sand_control[i]) == pytest.approx(c2.q[i], rel=1e-4)


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
