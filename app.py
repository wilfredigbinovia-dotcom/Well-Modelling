"""Streamlit front end for wellmodel.

Run from the project folder:   streamlit run app.py
"""
from __future__ import annotations

import io
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wellmodel as wm
from wellmodel import plotting as wplt
from wellmodel.correlations import LABELS
from wellmodel.matching import REQUIRED
from wellmodel import options as wopt
from wellmodel.synthetic import demo_wellbore, make_buildup, make_tests

st.set_page_config(page_title="wellmodel", page_icon="🛢️", layout="wide")
ss = st.session_state


# =============================================================== helpers
def show(fig_or_ax):
    fig = fig_or_ax if isinstance(fig_or_ax, plt.Figure) else fig_or_ax.figure
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def clear(*keys):
    for k in keys:
        ss.pop(k, None)


DERIVED = ("qc", "sel", "match", "mc", "bu_fit", "forecast")


def dist_input(label, key, default_kind, a, b=None, c=None, fmt="%.2f", step=None):
    """Distribution picker -> wm.Dist or float."""
    kinds = ["Fixed", "Uniform", "Triangular", "Normal"]
    col0, col1, col2, col3 = st.columns([1.3, 1, 1, 1])
    kind = col0.selectbox(label, kinds, index=kinds.index(default_kind), key=f"{key}_kind")
    kw = dict(format=fmt, step=step)
    if kind == "Fixed":
        return col1.number_input("value", value=float(a), key=f"{key}_v", **kw)
    if kind == "Uniform":
        lo = col1.number_input("low", value=float(a), key=f"{key}_lo", **kw)
        hi = col2.number_input("high", value=float(c if c is not None else b), key=f"{key}_hi", **kw)
        return wm.Uniform(lo, hi) if hi > lo else float(lo)
    if kind == "Triangular":
        lo = col1.number_input("low", value=float(a), key=f"{key}_tlo", **kw)
        mo = col2.number_input("mode", value=float(b), key=f"{key}_tmo", **kw)
        hi = col3.number_input("high", value=float(c), key=f"{key}_thi", **kw)
        return wm.Triangular(lo, mo, hi) if lo <= mo <= hi and hi > lo else float(mo)
    mean = col1.number_input("mean", value=float(b), key=f"{key}_m", **kw)
    sd = col2.number_input("std dev", value=float((c - a) / 4 if c is not None else abs(b) * 0.05),
                           key=f"{key}_sd", **kw)
    return wm.Normal(mean, sd) if sd > 0 else float(mean)



# =============================================================== tabs
TAB_NAMES = ["1 · Options summary", "2 · Well tests", "3 · QC", "4 · Correlation selection",
             "5 · Match", "6 · Nodal", "7 · Probabilistic", "8 · Transient"]
tabs = st.tabs(TAB_NAMES)

CAPTIONS = {
    "fluid_type": {"oil": "Oil with associated gas and water",
                   "dry_wet_gas": "Gas, with or without condensed liquid at surface",
                   "retrograde_condensate": "Gas that drops liquid in the reservoir below its dew point"},
    "method": {"black_oil": "Correlations from gravities and GOR/CGR",
               "eos": "Compositional; tuned to lab data"},
    "flow_type": {"tubing": "Production up the tubing",
                  "annular": "Flow in the casing-tubing annulus",
                  "tubing_annular": "Commingled flow in tubing and annulus"},
    "well_type": {"producer": "Flowing to surface",
                  "gas_injector": "Gas injected down the well",
                  "water_injector": "Water injected down the well"},
    "completion": {"cased_hole": "Cemented casing, perforated",
                   "open_hole": "No casing across the reservoir"},
    "sand_control": {"none": "No sand exclusion", "gravel_pack": "Sized gravel around a screen",
                     "pre_packed_screen": "Screen with resin-bonded gravel layer",
                     "wire_wrapped_screen": "Wire wrapped around a base pipe",
                     "slotted_liner": "Pipe with machined slots"},
}
STATUS_ICON = {wopt.MODELLED_NOW: "✅ Modelled", wopt.LUMPED_IN_PI: "↪️ Absorbed in matched PI",
               wopt.NOT_YET: "⏳ Not modelled yet"}

with tabs[0]:
    st.header("Options summary")
    st.caption("Describe the well first. These choices decide what the setup tabs ask for "
               "and which calculations apply.")
    sel_opts = {}
    groups = [("Fluid description", ["fluid_type", "method"]),
              ("Well", ["well_type", "flow_type"]),
              ("Completion", ["completion", "sand_control"])]
    cols = st.columns(3, gap="large")
    for col, (title, fields) in zip(cols, groups):
        with col:
            st.subheader(title)
            for f in fields:
                name, choices = wopt.FIELDS[f]
                keys = list(choices)
                sel_opts[f] = st.radio(name, keys, format_func=choices.get, key=f"opt_{f}",
                                       captions=[CAPTIONS[f][k] for k in keys])
    opts = wopt.WellOptions(**sel_opts)
    ss["options"] = opts

    st.divider()
    st.markdown(f"**Model:** {opts.describe()}")
    issues = opts.validate()
    for lvl, fn in (("error", st.error), ("warning", st.warning), ("info", st.info)):
        for i in issues:
            if i.level == lvl:
                fn(i.message, icon={"error": "🚫", "warning": "⚠️", "info": "ℹ️"}[lvl])
    if opts.errors:
        st.error("Fix the invalid combination above before continuing.")
    elif opts.computable:
        st.success("The calculation engine supports this configuration: all analysis tabs are available.")
    else:
        st.warning("Analysis tabs are locked for this configuration. The engine does not yet model: "
                   + "; ".join(opts.unsupported) + ". They will unlock as those modules are built.")

    st.subheader("Calculation status")
    stat = opts.engine_status()
    st.dataframe(pd.DataFrame({"Option": [wopt.FIELDS[f][0] for f in wopt.FIELDS],
                               "Selection": [opts.label(f) for f in wopt.FIELDS],
                               "Status": [STATUS_ICON[stat[f]] for f in wopt.FIELDS]}),
                 hide_index=True, width="stretch")
    st.caption("Absorbed in matched PI: the completion's pressure drop is not calculated "
               "separately yet; it is included in the PI fitted to well tests.")

    st.subheader("Data you will need")
    req = opts.data_requirements()
    for colx, (sect, items) in zip(st.columns(len(req), gap="large"), req.items()):
        with colx:
            st.markdown(f"**{sect}**")
            st.markdown("\n".join(f"- {x}" for x in items))


def locked() -> bool:
    """Show why an analysis tab is unavailable; True when it must not run."""
    o = ss["options"]
    if o.computable:
        return False
    st.warning("Not available for the configuration chosen in **Options summary**: "
               + ("; ".join(i.message for i in o.errors) if o.errors
                  else "the engine does not yet model " + "; ".join(o.unsupported) + "."))
    return True


# =============================================================== sidebar
st.sidebar.title("🛢️ wellmodel")
st.sidebar.caption("Oil-well nodal modelling with automated QC, matching, "
                   "correlation selection, probabilistic runs and transient screening.")

if not opts.computable:
    st.sidebar.info("These setup inputs cover oil producers only. They move into dedicated "
                    "setup tabs as the rework continues.")
with st.sidebar.expander("Fluid (black-oil PVT)", expanded=True):
    api = st.number_input("Oil gravity (°API)", 10.0, 60.0, 32.0, 0.5)
    gas_sg = st.number_input("Gas gravity (air = 1)", 0.55, 1.5, 0.75, 0.01)
    rsb = st.number_input("Solution GOR at Pb (scf/STB)", 10.0, 3000.0, 600.0, 10.0)
    t_res = st.number_input("Reservoir temperature (°F)", 80.0, 350.0, 210.0, 1.0)
    water_sg = st.number_input("Water gravity", 1.0, 1.25, 1.07, 0.01)
    use_pb = st.checkbox("I have a measured bubble point")
    pb_meas = st.number_input("Measured Pb (psia)", 100.0, 8000.0, 2863.0, 10.0) if use_pb else None

_wb = demo_wellbore()
with st.sidebar.expander("Wellbore", expanded=False):
    st.caption("Deviation survey (must start at MD 0). Edit, add or paste rows.")
    survey = st.data_editor(pd.DataFrame({"MD_ft": _wb.survey_md, "TVD_ft": np.round(_wb.survey_tvd, 1)}),
                            num_rows="dynamic", key="survey", hide_index=True)
    tubing_id = st.number_input("Tubing ID (in)", 1.0, 8.0, 3.958, 0.001, format="%.3f")
    perf_md = st.number_input("Mid-perf MD (ft)", 500.0, 30000.0, 10400.0, 10.0)
    gauge_md = st.number_input("Gauge MD (ft)", 100.0, 30000.0, 10000.0, 10.0)
    rough = st.number_input("Roughness (in)", 0.00001, 0.01, 0.0006, 0.0001, format="%.5f")
    wht = st.number_input("Flowing wellhead temperature (°F)", 40.0, 300.0, 125.0, 1.0)
    bht = st.number_input("Temperature at perfs (°F)", 60.0, 400.0, float(t_res), 1.0)

setup_key = (tuple(opts.to_dict().values()), api, gas_sg, rsb, t_res, water_sg, pb_meas, tubing_id, perf_md, gauge_md, rough, wht, bht,
             tuple(map(tuple, survey.dropna().to_numpy())))
if ss.get("setup_key") != setup_key:
    ss["setup_key"] = setup_key
    clear(*DERIVED)

try:
    sv = survey.dropna().sort_values("MD_ft")
    pvt = wm.BlackOilPVT(api, gas_sg, rsb, t_res, water_sg, pb=pb_meas)
    wb = wm.Wellbore(sv["MD_ft"].to_numpy(), sv["TVD_ft"].to_numpy(), tubing_id, perf_md,
                     gauge_md=gauge_md, roughness=rough, wht=wht, bht=bht)
    base = wm.WellModel(pvt, wb)
except Exception as e:  # invalid setup
    st.sidebar.error(f"Well setup problem: {e}")
    st.error("Fix the well setup in the sidebar to continue.")
    st.stop()

st.sidebar.markdown(f"**Pb** {pvt.pb:,.0f} psia · **Bob** {pvt.summary()['Bob']} rb/STB · "
                    f"**perf TVD** {wb.perf_tvd:,.0f} ft")
if "match" in ss:
    m = ss["match"]
    st.sidebar.success(f"Matched: {LABELS[m.correlation]}\n\n"
                       f"gravity {m.params.get('gravity_mult', 1):.3f} · friction "
                       f"{m.params.get('friction_mult', 1):.3f} · PI {m.params.get('pi', base.pi):.2f}")


def current_model() -> wm.WellModel:
    return ss["match"].model(base) if "match" in ss else ss.get("manual_model", base.copy(correlation="hagedorn_brown", pi=2.0))



# ---------------------------------------------------------------- tests
with tabs[1]:
    st.header("Well tests")
    st.write("Required columns: " + ", ".join(f"`{c}`" for c in REQUIRED) +
             ". Optional: `test_id`, `date`, `duration_hr`, `rate_cv`. Units: STB/d, fraction, "
             "scf/STB, psia. The gauge depth is set in the sidebar.")
    c1, c2, c3 = st.columns([2, 1, 1])
    up = c1.file_uploader("Upload well tests (CSV)", type="csv")
    if c2.button("Load demo tests", width="stretch"):
        ss["tests"] = make_tests()[0]
        clear(*DERIVED)
    demo_csv = make_tests()[0]
    c3.download_button("Download CSV template", csv_bytes(demo_csv), "well_tests_template.csv",
                       width="stretch")
    if up is not None and ss.get("upload_name") != up.name + str(up.size):
        try:
            df = pd.read_csv(up)
            missing = [c for c in REQUIRED if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if "date" in df:
                    df["date"] = pd.to_datetime(df["date"])
                if "test_id" not in df:
                    df.insert(0, "test_id", [f"T{i+1:02d}" for i in range(len(df))])
                ss["tests"] = df
                ss["upload_name"] = up.name + str(up.size)
                clear(*DERIVED)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    if "tests" in ss:
        edited = st.data_editor(ss["tests"], num_rows="dynamic", key="tests_editor",
                                width="stretch")
        if not edited.equals(ss["tests"]):
            ss["tests"] = edited.reset_index(drop=True)
            clear(*DERIVED)
        st.caption(f"{len(ss['tests'])} tests loaded. Edits here reset downstream results.")
    else:
        st.info("Upload a CSV or load the demo tests (synthetic well with four planted problems).")

tests = ss.get("tests")
need_tests = lambda: st.info("Load well tests in tab 2 first.")

# ------------------------------------------------------------------- QC
with tabs[2]:
    st.header("Test data QC")
    if locked():
        pass
    elif tests is None:
        need_tests()
    else:
        with st.expander("QC thresholds"):
            c1, c2, c3 = st.columns(3)
            cfg = wm.QCConfig(
                gor_low_frac=c1.number_input("Fail if GOR < x·Rsb (Pr>Pb)", 0.3, 1.0, 0.85, 0.05),
                pi_z_fail=c2.number_input("PI outlier fail (robust σ)", 1.5, 10.0, 3.5, 0.5),
                pi_z_warn=c2.number_input("PI outlier warn (robust σ)", 1.0, 10.0, 2.5, 0.5),
                min_duration_hr=c3.number_input("Min test duration (h)", 0.0, 48.0, 6.0, 1.0),
                envelope_tol_psi=c1.number_input("VLP envelope tolerance (psi)", 0.0, 500.0, 50.0, 10.0),
                wc_jump=c3.number_input("Water-cut jump warn", 0.01, 1.0, 0.15, 0.01))
        if st.button("Run QC", type="primary"):
            with st.spinner("Checking tests…"):
                ss["qc"] = wm.qc_tests(base, tests, cfg)
            clear("sel", "match")
        if "qc" in ss:
            qc = ss["qc"]
            n = qc.counts()
            c = st.columns(3)
            c[0].metric("✅ Pass", n.get("pass", 0))
            c[1].metric("⚠️ Warn (half weight)", n.get("warn", 0))
            c[2].metric("❌ Fail (excluded)", n.get("fail", 0))
            show(wplt.plot_qc(qc))
            st.dataframe(qc.table[["test_id", "status", "weight", "pi_implied", "pi_z", "reasons"]],
                         width="stretch", hide_index=True)
            st.caption("You can override a test's weight in the Match tab.")

# ----------------------------------------------------------- selection
with tabs[3]:
    st.header("Correlation selection")
    if locked():
        pass
    elif tests is None:
        need_tests()
    else:
        weights = ss["qc"].weights if "qc" in ss else np.ones(len(tests))
        if "qc" not in ss:
            st.warning("QC has not been run - all tests will be used at full weight.")
        cands = st.multiselect("Correlations to compare", list(LABELS), default=list(LABELS),
                               format_func=LABELS.get)
        loo = st.checkbox("Leave-one-out cross-validation (recommended, slower)", value=True)
        if st.button("Compare correlations", type="primary", disabled=not cands):
            with st.spinner("Matching each correlation and cross-validating - this can take a minute…"):
                ss["sel"] = wm.compare_correlations(base, tests, weights=weights,
                                                    correlations=cands, loocv=loo)
        if "sel" in ss:
            sel = ss["sel"]
            st.success(sel.rationale.splitlines()[0])
            st.text(sel.rationale)
            if loo:
                show(wplt.plot_selection(sel))
            cols = ["rank", "label", "raw_gauge_bias", "matched_gauge_rmse", "loocv_gauge_rmse",
                    "loocv_rate_mape", "gravity_mult", "friction_mult", "pi", "in_range"]
            st.dataframe(sel.table[[c for c in cols if c in sel.table]].round(3),
                         width="stretch", hide_index=True)
            if st.button(f"Use {LABELS[sel.recommended]} match"):
                ss["match"] = sel.matches[sel.recommended]
                st.rerun()

# ---------------------------------------------------------------- match
with tabs[4]:
    st.header("Multi-test match")
    if locked():
        pass
    elif tests is None:
        need_tests()
    else:
        default_corr = ss["sel"].recommended if "sel" in ss else "hagedorn_brown"
        c1, c2 = st.columns(2)
        corr = c1.selectbox("Correlation", list(LABELS), index=list(LABELS).index(default_corr),
                            format_func=LABELS.get)
        fit = c2.multiselect("Parameters to fit", ["gravity_mult", "friction_mult", "pi"],
                             default=["gravity_mult", "friction_mult", "pi"])
        c1, c2 = st.columns(2)
        sig_p = c1.number_input("Expected gauge error σ (psi)", 5.0, 200.0, 30.0, 5.0,
                                help="Total error including input errors - a 3 % GOR error "
                                     "moves the gauge ~40 psi.")
        sig_q = c2.number_input("Expected rate error σ (%)", 0.5, 20.0, 3.0, 0.5) / 100
        w0 = ss["qc"].weights if "qc" in ss else np.ones(len(tests))
        wt = st.data_editor(pd.DataFrame({"test_id": tests.get("test_id", pd.Series(range(len(tests)))),
                                          "weight": w0}),
                            disabled=["test_id"], hide_index=True, key="weights_editor")
        if st.button("Run match", type="primary", disabled=not fit):
            with st.spinner("Matching all tests…"):
                try:
                    ss["match"] = wm.match_tests(base, tests, fit=tuple(fit), weights=wt["weight"].to_numpy(),
                                                 correlation=corr, sigma_p=sig_p, sigma_q_frac=sig_q)
                    clear("mc", "forecast")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        if "match" in ss:
            m = ss["match"]
            st.subheader(f"Result - {LABELS[m.correlation]}")
            for f in m.flags:
                st.warning(f)
            if not m.flags:
                st.success("No flags: multipliers in range, parameters identifiable, no residual outliers.")
            c1, c2 = st.columns([1, 1])
            c1.dataframe(m.summary(), hide_index=True, width="stretch")
            c2.dataframe(m.correlation_matrix.round(2), width="stretch")
            st.caption(f"Gauge RMSE {m.rmse_gauge:.1f} psi · rate MAPE {m.mape_rate:.1f} % · "
                       f"{m.n_tests_used} tests used")
            show(wplt.plot_match(m, sigma_p=sig_p))
            st.dataframe(m.residuals.round(1), width="stretch", hide_index=True)
            st.download_button("Download residuals CSV", csv_bytes(m.residuals), "match_residuals.csv")

# ---------------------------------------------------------------- nodal
with tabs[5]:
    st.header("Nodal analysis")
    if not locked():
        if "match" not in ss:
            st.info("No match yet - set the model manually, or run a match in tab 5.")
            c = st.columns(4)
            mc_corr = c[0].selectbox("Correlation", list(LABELS), index=list(LABELS).index("hagedorn_brown"),
                                     format_func=LABELS.get, key="man_corr")
            pi_guess = 2.0
            if tests is not None:  # rough PI from tests: q / (Pr - gauge)
                dd = (tests["reservoir_pressure"] - tests["gauge_pressure"]).clip(lower=50)
                pi_guess = float(np.median(tests["liquid_rate"] / dd))
            mpi = c[1].number_input("PI (STB/d/psi)", 0.01, 100.0, round(pi_guess, 2), 0.1, key="man_pi",
                                    help="Default estimated from the loaded tests")
            mgm = c[2].number_input("Gravity mult", 0.5, 1.5, 1.0, 0.01, key="man_gm")
            mfm = c[3].number_input("Friction mult", 0.2, 3.0, 1.0, 0.05, key="man_fm")
            ss["manual_model"] = base.copy(correlation=mc_corr, pi=mpi, gravity_mult=mgm, friction_mult=mfm)
        model = current_model()
        last = tests.iloc[-1] if tests is not None else None
        c = st.columns(4)
        pr = c[0].number_input("Reservoir pressure (psia)", 500.0, 15000.0,
                               float(last.reservoir_pressure) if last is not None else 3500.0, 25.0)
        whp = c[1].number_input("WHP (psia)", 15.0, 5000.0, float(last.whp) if last is not None else 250.0, 10.0)
        wc = c[2].slider("Water cut", 0.0, 0.98, float(last.water_cut) if last is not None else 0.2, 0.01)
        gor = c[3].number_input("GOR (scf/STB)", 0.0, 20000.0, float(last.gor) if last is not None else rsb, 10.0)
        compare = st.multiselect("Also draw VLP for (same multipliers)", [k for k in LABELS if k != model.correlation],
                                 format_func=LABELS.get)
        op = model.operating_point(pr, whp, wc, gor)
        k = st.columns(4)
        if op.flowing[0]:
            k[0].metric("Liquid rate", f"{op.rate[0]:,.0f} STB/d")
            k[1].metric("Oil rate", f"{op.rate[0] * (1 - wc):,.0f} STB/d")
            k[2].metric("Pwf", f"{op.pwf[0]:,.0f} psia")
            k[3].metric("Flow", "stable" if op.stable[0] else "⚠️ unstable (left of VLP minimum)")
        else:
            st.error("The well cannot flow naturally at these conditions (IPR and VLP do not intersect) - "
                     "artificial lift needed.")
        models = [model] + [model.copy(correlation=c_) for c_ in compare]
        show(wplt.plot_nodal(models, pr, whp, wc, gor))

        with st.expander("Sensitivity table (WHP × water cut)"):
            whps = np.linspace(max(whp * 0.5, 50), whp * 1.6, 6).round(0)
            wcs = np.clip(np.linspace(wc - 0.15, wc + 0.25, 6), 0, 0.97).round(2)
            W, C = np.meshgrid(whps, wcs, indexing="ij")
            o = model.operating_point(pr, W.ravel(), C.ravel(), gor)
            grid = pd.DataFrame(o.rate.reshape(W.shape).round(0), index=[f"WHP {x:.0f}" for x in whps],
                                columns=[f"WC {x:.0%}" for x in wcs])
            st.dataframe(grid, width="stretch")
            st.caption("Liquid rate (STB/d); 0 = well cannot flow.")

# -------------------------------------------------------- probabilistic
with tabs[6]:
    st.header("Probabilistic nodal run")
    if not locked():
        model = current_model()
        last = tests.iloc[-1] if tests is not None else None
        lp = float(last.reservoir_pressure) if last is not None else 3500.0
        lw = float(last.water_cut) if last is not None else 0.25
        lh = float(last.whp) if last is not None else 250.0
        lg = float(last.gor) if last is not None else rsb
        st.caption("Choose a distribution for each input. P90 = low case, P10 = high case.")
        inputs = {
            "reservoir_pressure": dist_input("Reservoir pressure (psia)", "d_pr", "Triangular",
                                             lp - 150, lp - 70, lp + 10, "%.0f", 10.0),
            "whp": dist_input("WHP (psia)", "d_whp", "Uniform", lh - 20, None, lh + 40, "%.0f", 5.0),
            "water_cut": dist_input("Water cut", "d_wc", "Triangular", max(lw - 0.0, 0), min(lw + 0.03, 0.97),
                                    min(lw + 0.08, 0.98), "%.3f", 0.01),
            "gor": dist_input("GOR (scf/STB)", "d_gor", "Normal", lg - 40, lg, lg + 40, "%.0f", 5.0),
        }
        c1, c2, c3 = st.columns(3)
        n = c1.select_slider("Samples", [200, 500, 1000, 2000, 5000], value=1000)
        use_cov = c2.checkbox("Include model uncertainty from match", value="match" in ss,
                              disabled="match" not in ss)
        seed = c3.number_input("Random seed", 0, 10_000, 1)
        if st.button("Run Monte Carlo", type="primary"):
            with st.spinner(f"Solving {n} nodal cases…"):
                ss["mc"] = wm.monte_carlo(model, inputs, n=n, seed=int(seed),
                                          match=ss.get("match") if use_cov else None)
        if "mc" in ss:
            mc = ss["mc"]
            a = mc.summary.attrs
            oil = mc.summary.loc["oil_rate"]
            k = st.columns(5)
            k[0].metric("Oil P90 (low)", f"{oil['P90']:,.0f}")
            k[1].metric("Oil P50", f"{oil['P50']:,.0f}")
            k[2].metric("Oil P10 (high)", f"{oil['P10']:,.0f}")
            k[3].metric("P(flows naturally)", f"{a['prob_flowing']:.0%}")
            k[4].metric("P(unstable)", f"{a['prob_unstable']:.0%}")
            show(wplt.plot_monte_carlo(mc))
            c1, c2 = st.columns(2)
            c1.dataframe(mc.summary.round(0), width="stretch")
            c2.dataframe(mc.sensitivity.round(3), hide_index=True, width="stretch")
            st.download_button("Download samples CSV", csv_bytes(mc.samples), "monte_carlo_samples.csv")

# ------------------------------------------------------------ transient
with tabs[7]:
    st.header("Transient behaviour")
    if not locked():
        model = current_model()
        sub = st.radio("Section", ["Build-up interpretation", "Depletion forecast"], horizontal=True,
                       label_visibility="collapsed")
        _, bo_i, _, mu_i = pvt.oil(float(tests.reservoir_pressure.iloc[0]) if tests is not None else 4000.0, t_res)
        ct_default = float(0.75 * pvt.total_compressibility_oil(4000.0) + 0.25 * 3e-6 + 4e-6)

        if sub == "Build-up interpretation":
            st.caption("Fits permeability, skin, wellbore storage and initial pressure to a shut-in survey; "
                       "the resulting PSS PI is an independent check on the production-test match.")
            c1, c2 = st.columns([2, 1])
            bu_up = c1.file_uploader("Build-up data CSV with columns `dt_hr`, `pws`", type="csv")
            if c2.button("Load demo build-up", width="stretch"):
                ss["bu_data"] = make_buildup()
                clear("bu_fit")
            if bu_up is not None:
                d = pd.read_csv(bu_up)
                if {"dt_hr", "pws"} <= set(d.columns):
                    ss["bu_data"] = d[d["dt_hr"] > 0].sort_values("dt_hr").reset_index(drop=True)
                else:
                    st.error("CSV needs columns dt_hr and pws.")
            st.subheader("Reservoir and test properties")
            c = st.columns(5)
            h = c[0].number_input("Net pay h (ft)", 1.0, 2000.0, 60.0)
            phi = c[1].number_input("Porosity", 0.01, 0.45, 0.20, 0.01)
            ct = c[2].number_input("Total compressibility (1/psi)", 1e-7, 1e-3, ct_default, format="%.2e")
            mu = c[3].number_input("Oil viscosity (cp)", 0.05, 500.0, float(mu_i), 0.01)
            bo = c[4].number_input("Bo (rb/STB)", 1.0, 3.0, float(bo_i), 0.01)
            c = st.columns(5)
            rw = c[0].number_input("rw (ft)", 0.1, 1.0, 0.354, 0.001, format="%.3f")
            re = c[1].number_input("Drainage radius re (ft)", 100.0, 20000.0, 1500.0, 50.0)
            q = c[2].number_input("Rate before shut-in (STB/d)", 1.0, 1e5, 3500.0, 50.0)
            tp = c[3].number_input("Producing time tp (h)", 1.0, 1e6, 240.0, 10.0)
            k0 = c[4].number_input("Starting guess k (md)", 0.01, 1e4, 10.0)
            if "bu_data" in ss:
                bu = ss["bu_data"]
                if st.button("Fit build-up", type="primary"):
                    guess = wm.RadialReservoir(k=k0, h=h, phi=phi, ct=ct, mu=mu, bo=bo, rw=rw, re=re,
                                               skin=0.0, C=0.01)
                    with st.spinner("Fitting…"):
                        ss["bu_fit"] = wm.fit_buildup(bu["dt_hr"], bu["pws"], tp, q, guess)
                if "bu_fit" in ss:
                    f = ss["bu_fit"]
                    r = f.reservoir
                    k = st.columns(5)
                    k[0].metric("Permeability", f"{r.k:.2f} md", f"± {f.std_errors['k']:.2f}", delta_color="off")
                    k[1].metric("Skin", f"{r.skin:.2f}", f"± {f.std_errors['skin']:.2f}", delta_color="off")
                    k[2].metric("Storage C", f"{r.C:.4f} bbl/psi")
                    k[3].metric("p*", f"{f.p_init:,.1f} psia")
                    k[4].metric("PSS PI", f"{f.pi_pss:.2f}",
                                None if "match" not in ss else f"match PI {ss['match'].params.get('pi', np.nan):.2f}",
                                delta_color="off")
                    st.caption(f"Fit RMSE {f.rmse:.2f} psi · PSS starts after ~{r.t_pss_hr():.0f} h of flow")
                    show(wplt.plot_buildup(bu["dt_hr"].to_numpy(), bu["pws"].to_numpy(), f, tp, q))
            else:
                st.info("Upload build-up data or load the demo survey.")

        else:
            st.caption("Material balance (+ optional aquifer) coupled to the nodal model at every step. "
                       f"Uses the {'matched' if 'match' in ss else 'manual'} well model, PI {model.pi:.2f}.")
            c = st.columns(4)
            stoiip = c[0].number_input("Oil in place in drainage area (MMSTB)", 0.1, 1000.0, 45.0, 1.0) * 1e6
            p_init = c[1].number_input("Initial pressure (psia)", 500.0, 15000.0,
                                       float(tests.reservoir_pressure.iloc[0]) if tests is not None else 4200.0, 25.0)
            aq = c[2].number_input("Aquifer productivity (rb/d/psi)", 0.0, 1000.0, 6.0, 1.0)
            fwhp = c[3].number_input("WHP (psia) ", 15.0, 5000.0, 250.0, 10.0)
            c = st.columns(4)
            qmax = c[0].number_input("Max liquid rate / facility limit (STB/d)", 0.0, 1e5, 5000.0, 100.0)
            wc0 = c[1].number_input("Initial water cut", 0.0, 0.9, 0.04, 0.01)
            rf50 = c[2].number_input("Recovery factor at mid water-cut rise", 0.01, 0.6, 0.10, 0.01)
            width = c[3].number_input("Water-cut rise width (RF)", 0.005, 0.2, 0.035, 0.005, format="%.3f")
            c = st.columns(4)
            wcmax = c[0].number_input("Final water cut", 0.1, 0.99, 0.95, 0.01)
            beta = c[1].number_input("GOR rise below Pb (β)", 0.0, 20.0, 4.0, 0.5)
            years = c[2].slider("Years", 1, 30, 15)
            step = c[3].select_slider("Step (days)", [15, 30, 60, 90], value=30)
            use_res = "bu_fit" in ss and st.checkbox("Use early-time transient PI from build-up fit", value=True)
            if st.button("Run forecast", type="primary"):
                inp = wm.DepletionInputs(stoiip=stoiip, p_init=p_init, aquifer_pi=aq, whp=fwhp,
                                         max_liquid_rate=qmax or None,
                                         water_cut=wm.logistic_water_cut(wc0, wcmax, rf50, width),
                                         gor=wm.solution_gas_gor(beta))
                with st.spinner("Stepping the well through time…"):
                    ss["forecast"] = wm.forecast(model, inp, days=years * 365, step_days=step,
                                                 reservoir=ss["bu_fit"].reservoir if use_res else None)
            if "forecast" in ss:
                fr = ss["forecast"]
                ev = fr.events
                yrs = lambda d: "—" if d is None else f"{float(d) / 365.25:.1f} yr"
                k = st.columns(4)
                k[0].metric("Natural flow stops", yrs(ev["flow_stops_days"]))
                k[1].metric("Oil produced by then" if ev["np_at_flow_stop"] else "Oil produced",
                            f"{(ev['np_at_flow_stop'] or fr.table['Np'].iloc[-1]) / 1e6:.2f} MMSTB")
                k[2].metric("Reservoir below Pb", yrs(ev["pr_below_pb_days"]))
                k[3].metric("Unstable flow from", yrs(ev["unstable_from_days"]))
                if ev["flow_stops_days"]:
                    st.warning("Plan artificial lift (gas lift or ESP) before natural flow stops.")
                show(wplt.plot_forecast(fr, pb=pvt.pb))
                st.download_button("Download forecast CSV", csv_bytes(fr.table), "forecast.csv")
