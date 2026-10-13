"""Streamlit front end for wellmodel.

Run from the project folder:   streamlit run app.py

The app is being rebuilt tab by tab. Current tabs:
  1. Options summary - what kind of well is being modelled
  2. PVT Input Data  - black oil: inputs, lab data, correlation matching and suggestion
                       equation of state: PRP file import, flash, saturation point, envelope
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wellmodel import blackoil as bo
from wellmodel import eos as wm_eos
from wellmodel import options as wopt
from wellmodel import plotting as wplt

st.set_page_config(page_title="wellmodel", page_icon="🛢️", layout="wide")
ss = st.session_state

st.title("🛢️ wellmodel")

TAB_NAMES = ["1 · Options summary", "2 · PVT Input Data"]
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
               wopt.PVT_ONLY: "🧪 PVT only - flow calculations not built yet",
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
        st.success("The calculation engine supports this configuration.")
    elif opts.unsupported:
        st.warning("The calculation engine does not yet model: " + "; ".join(opts.unsupported)
                   + ". Calculations for this configuration become available as those modules are built.")
    else:
        st.info("PVT for this configuration is set up from an equation-of-state (PRP) file in the "
                "PVT Input Data tab. Flow calculations with compositional PVT are not built yet.")

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


# =============================================================== tab 2: PVT
UNITS = {"Pressure": "psia", "Gas Oil Ratio": "scf/STB", "Oil FVF": "RB/STB",
         "Oil Viscosity": "centipoise"}
N_ROWS = 15
INPUT_DEFAULTS = {"pvt_rsb": 500.0, "pvt_api": 35.0, "pvt_gg": 0.75, "pvt_sal": 50000.0,
                  "pvt_h2s": 0.0, "pvt_co2": 0.0, "pvt_n2": 0.0, "pvt_t": 200.0, "pvt_pb": 0.0}
for k, v in INPUT_DEFAULTS.items():
    ss.setdefault(k, v)


def empty_table(n=N_ROWS) -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series([np.nan] * n, dtype=float) for c in bo.TABLE_COLUMNS})
    df.index = pd.RangeIndex(1, n + 1, name="Point")
    return df


ss.setdefault("pvt_table", empty_table())


def _reset_editor():
    ss.pop("pvt_table_editor", None)
    ss.pop("pvt_match", None)


def load_example():
    inp, data = bo.example_data()
    ss.update(pvt_rsb=float(inp.rsb), pvt_api=float(inp.api), pvt_gg=float(inp.gas_sg),
              pvt_sal=float(inp.salinity_ppm), pvt_h2s=float(inp.h2s), pvt_co2=float(inp.co2),
              pvt_n2=float(inp.n2), pvt_t=float(data.temperature), pvt_pb=float(data.bubble_point))
    t = empty_table()
    t.iloc[:len(data.table)] = data.table[bo.TABLE_COLUMNS].to_numpy(float)
    ss["pvt_table"] = t
    _reset_editor()


def clear_table():
    ss["pvt_table"] = empty_table()
    _reset_editor()


def current_inputs() -> tuple[bo.PVTInputs, bo.MatchData]:
    inp = bo.PVTInputs(rsb=ss.pvt_rsb, api=ss.pvt_api, gas_sg=ss.pvt_gg, salinity_ppm=ss.pvt_sal,
                       h2s=ss.pvt_h2s, co2=ss.pvt_co2, n2=ss.pvt_n2,
                       correct_impurities=ss.get("pvt_impurity_corr", True))
    data = bo.MatchData(temperature=ss.pvt_t, bubble_point=ss.pvt_pb if ss.pvt_pb > 0 else None,
                        table=ss.get("pvt_table_current", ss["pvt_table"]))
    return inp, data


def signature(inp, data):
    return (tuple(vars(inp).values()), data.temperature, data.bubble_point,
            tuple(np.nan_to_num(data.table[bo.TABLE_COLUMNS].to_numpy(float), nan=-1).ravel()))


def run_match():
    inp, data = current_inputs()
    res = bo.match_all(inp, data)
    ss["pvt_match"] = res
    ss["pvt_match_sig"] = signature(inp, data)
    if res.suggestion.get("pb_family"):
        ss["pvt_sel_pb"] = res.suggestion["pb_family"]
    if res.suggestion.get("visc_family"):
        ss["pvt_sel_visc"] = res.suggestion["visc_family"]


def grid_view(res: bo.MatchResult, prop: str, suggested: str | None):
    """Match grid as display strings ('—' for not calculated), suggested column highlighted."""
    df = res.table(prop)
    fams = bo.VISC_FAMILIES if prop == "visc" else bo.PB_FAMILIES
    sug_label = fams.get(suggested) if suggested else None
    fmt = {"Parameter 1": "{:.5f}", "Parameter 2": "{:.5g}", "Std Deviation": "{:.5g}",
           "Raw error (%)": "{:.2f}", "Points": "{:.0f}"}
    shown = pd.DataFrame({c: [("—" if v is None or pd.isna(v) else fmt[r].format(float(v)))
                              for r, v in df[c].items()] for c in df.columns}, index=df.index)
    sty = shown.style.set_properties(**{"text-align": "right"})
    if sug_label in shown.columns:
        sty = sty.set_properties(subset=pd.IndexSlice[:, [sug_label]],
                                 **{"background-color": "rgba(42,120,214,0.14)", "font-weight": "600"})
    return sty


def match_plot(model_raw: bo.BlackOilModel, model_m: bo.BlackOilModel, data: bo.MatchData):
    tbl = data.clean()
    pmax = max([data.bubble_point or 0] + tbl["Pressure"].tolist() + [1000.0]) * 1.1
    p = np.linspace(50, pmax, 120)
    a = model_raw.table(data.temperature, p)
    b = model_m.table(data.temperature, p)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [("Gas Oil Ratio", "Solution GOR (scf/STB)"), ("Oil FVF", "Oil FVF (RB/STB)"),
              ("Oil Viscosity", "Oil viscosity (cp)")]
    for ax, (col, out) in zip(axes, panels):
        ax.plot(p, a[out], color=wplt.SERIES[0], lw=1.5, ls="--", label="correlation")
        ax.plot(p, b[out], color=wplt.SERIES[0], lw=2, label="matched")
        d = tbl[tbl[col].notna()]
        ax.scatter(d["Pressure"], d[col], s=36, color=wplt.INK, zorder=4, label="lab data")
        if data.bubble_point:
            ax.axvline(data.bubble_point, color=wplt.INK2, lw=1, ls=":")
        ax.set_xlabel("Pressure (psia)")
        ax.set_title(f"{col} ({UNITS[col]})")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- EOS (PRP import)
EOS_NAMES = {"PR": "Peng-Robinson", "SRK": "Soave-Redlich-Kwong"}
FLUID_FROM_SAT = {"bubble": "oil", "dew": "retrograde_condensate"}


@st.cache_data(show_spinner=False)
def eos_parse(text: str, name: str):
    return wm_eos.parse_prp(text, name)


@st.cache_data(show_spinner=False)
def eos_calculate(text: str, name: str, t_f: float):
    prp = wm_eos.parse_prp(text, name)
    f = wm_eos.EOSFluid(prp)
    ps, typ = f.saturation_pressure(t_f)
    p_res = ps * 1.0005 if ps else 5000.0
    sep = f.separator_train(T_res=t_f, P_res=p_res)
    top = max(ps * 1.3 if ps else 6000.0, 1000.0)
    pres = sorted(set(np.round(np.concatenate([[top, (ps or top) * 1.1, ps or top],
                                               np.linspace((ps or top) * 0.92, 14.7, 11)]), 1)),
                  reverse=True)
    table = f.property_table(t_f, pres)
    return {"psat": ps, "sat_type": typ, "sep": sep, "table": table, "p_res": p_res}


@st.cache_data(show_spinner=False)
def eos_envelope(text: str, name: str):
    f = wm_eos.EOSFluid(wm_eos.parse_prp(text, name))
    env = f.phase_envelope()
    return env, wm_eos.EOSFluid.envelope_summary(env)


def load_example_prp():
    ss["prp_text"] = wm_eos.example_prp_text()
    ss["prp_name"] = "example (synthetic oil).prp"
    ss.pop("prp_t", None)
    ss.pop("prp_show_env", None)


def envelope_plot(env: pd.DataFrame, t_res: float, psat, summ: dict):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for typ, col, lab in (("bubble", wplt.SERIES[0], "Bubble point"), ("dew", wplt.SERIES[1], "Dew point")):
        for br in ("upper", "lower"):
            d = env[(env["type"] == typ) & (env["branch"] == br)].sort_values("T (degF)")
            if not d.empty:
                ax.plot(d["T (degF)"], d["P (psia)"], "-o", ms=3, color=col,
                        label=lab if br == "upper" else None)
    ax.axvline(t_res, color=wplt.INK2, lw=1, ls=":")
    ax.text(t_res, ax.get_ylim()[1] * 0.03, f" {t_res:g} °F", fontsize=8.5, color=wplt.INK2, va="bottom")
    if psat:
        ax.scatter([t_res], [psat], s=70, color=wplt.INK, zorder=5)
        ax.annotate(f"{psat:,.0f} psia", (t_res, psat), textcoords="offset points", xytext=(8, 6),
                    fontsize=8.5, color=wplt.INK)
    if "critical_T_approx" in summ:
        ax.axvline(summ["critical_T_approx"], color=wplt.GRID, lw=6, alpha=0.6, zorder=0)
    ax.set_xlabel("Temperature (°F)")
    ax.set_ylabel("Pressure (psia)")
    ax.set_ylim(bottom=0)
    ax.set_title("Phase envelope")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def properties_plot(table: pd.DataFrame, psat):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    p = table["Pressure (psia)"]
    axes[0].plot(p, table["Liquid volume (%)"], "-o", ms=3, color=wplt.SERIES[0])
    axes[0].set_title("Liquid volume (% of total)")
    for col, c, lab in (("Liquid density (lb/ft3)", wplt.SERIES[0], "liquid"),
                        ("Vapour density (lb/ft3)", wplt.SERIES[1], "vapour")):
        axes[1].plot(p, table[col], "-o", ms=3, color=c, label=lab)
    axes[1].set_title("Density (lb/ft3)")
    axes[1].legend()
    for col, c, lab in (("Liquid viscosity (cp)", wplt.SERIES[0], "liquid"),
                        ("Vapour viscosity (cp)", wplt.SERIES[1], "vapour")):
        axes[2].plot(p, table[col], "-o", ms=3, color=c, label=lab)
    axes[2].set_yscale("log")
    axes[2].set_title("Viscosity (cp, LBC)")
    axes[2].legend()
    for a in axes:
        a.set_xlabel("Pressure (psia)")
        if psat:
            a.axvline(psat, color=wplt.INK2, lw=1, ls=":")
    fig.tight_layout()
    return fig


def render_eos(opts_now):
    st.caption("Equation of state is selected in the Options summary: import the fluid "
               "description as a Petroleum Experts PRP export file.")
    c = st.columns([2, 1])
    up = c[0].file_uploader("PVT file (.prp)", type=["prp", "txt"], key="prp_upload")
    c[1].button("Load example PRP", on_click=load_example_prp, width="stretch",
                help="A small synthetic oil - not a real fluid.")
    if up is not None and ss.get("prp_upload_id") != (up.name, up.size):
        ss["prp_upload_id"] = (up.name, up.size)
        raw = up.getvalue()
        try:
            ss["prp_text"] = raw.decode("utf-8")
        except UnicodeDecodeError:
            ss["prp_text"] = raw.decode("latin-1")
        ss["prp_name"] = up.name
        ss.pop("prp_t", None)
        ss.pop("prp_show_env", None)
    text, name = ss.get("prp_text"), ss.get("prp_name", "")
    if not text:
        st.info("Upload a .prp file exported from PROSPER / PVTP, or load the example.")
        return
    try:
        prp = eos_parse(text, name)
    except Exception as e:
        st.error(f"Could not read {name}: {e}")
        return

    comp = prp.components
    c7 = wm_eos.c7_plus_fraction(prp)
    st.subheader(f"Fluid file: {name}")
    m = st.columns(6)
    m[0].metric("Equation of state", prp.eos, help=EOS_NAMES[prp.eos])
    m[1].metric("Components", len(comp))
    m[2].metric("Non-zero BICs", int((prp.bic != 0).sum() // 2))
    m[3].metric("Volume shift", "On" if prp.volume_shift else "Off")
    m[4].metric("C7+ (mole %)", f"{100 * c7:.2f}")
    m[5].metric("Water salinity (ppm)", f"{prp.salinity_ppm:,.0f}")
    for note in prp.notes:
        st.info(note)
    st.caption(f"Viscosity model in file: oil {prp.oil_viscosity}, gas {prp.gas_viscosity} "
               "(Lohrenz-Bray-Clark is used). Pressures in the file are psig and are shown here in psia.")

    with st.expander("Composition and component properties", expanded=False):
        show = comp.copy()
        show.insert(0, "Mole %", 100 * show.pop("Mole fraction"))
        st.dataframe(show.set_index("Name").round(5), width="stretch")
    with st.expander("Binary interaction coefficients"):
        st.dataframe(pd.DataFrame(prp.bic, index=comp["Name"], columns=comp["Name"])
                     .style.format("{:.4g}").background_gradient(cmap="Blues", axis=None),
                     width="stretch")
    if prp.separators:
        with st.expander("Separator train (from file)"):
            st.dataframe(pd.DataFrame(prp.separators, columns=["T (degF)", "P (psia)"],
                                      index=pd.RangeIndex(1, len(prp.separators) + 1, name="Stage")),
                         width="stretch")

    st.subheader("Calculations")
    ss.setdefault("prp_t", float(prp.reference_temperature or 200.0))
    t_f = st.number_input("Temperature (deg F)", -100.0, 600.0, step=1.0, key="prp_t",
                          help="Defaults to the reference temperature in the file.")
    with st.spinner("Flashing the fluid..."):
        res = eos_calculate(text, name, float(t_f))
    ps, typ, sep = res["psat"], res["sat_type"], res["sep"]

    implied = FLUID_FROM_SAT.get(typ, "dry_wet_gas")
    if opts_now is not None and implied != opts_now.fluid_type:
        st.warning(f"At {t_f:g} °F this fluid has a **{typ or 'no'} point**, which makes it "
                   f"**{wopt.FLUID_TYPES[implied].lower()}**, but the Options summary says "
                   f"**{opts_now.label('fluid_type').lower()}**. Check the fluid type.", icon="⚠️")

    m = st.columns(5)
    if ps:
        m[0].metric(f"{typ.capitalize()} point", f"{ps:,.0f} psia")
    else:
        m[0].metric("Saturation point", "none", help="Single phase at all pressures at this temperature")
    is_gas = typ != "bubble"
    if is_gas:
        m[1].metric("CGR (STB/MMscf)", f"{sep['cgr']:,.1f}" if np.isfinite(sep["cgr"]) else "—")
        m[2].metric("Bg at Psat (ft3/scf)", f"{sep.get('bg', np.nan):.5f}")
    else:
        m[1].metric("GOR (scf/STB)", f"{sep['gor']:,.0f}")
        m[2].metric("Bo at Psat (RB/STB)", f"{sep.get('bo', np.nan):.4f}")
    m[3].metric("Stock-tank liquid (°API)", f"{sep['api']:.1f}" if np.isfinite(sep["api"]) else "—")
    m[4].metric("Separator gas gravity", f"{sep['gas_gravity']:.3f}")
    st.caption(f"Reservoir fluid at {res['p_res']:,.0f} psia: density {sep.get('res_density', np.nan):.2f} lb/ft3, "
               f"viscosity {sep.get('res_viscosity', np.nan):.4f} cp. Surface values from flashing the "
               f"reservoir fluid through the file's {len(prp.separators) or 1}-stage separator train.")
    with st.expander("Separator stages"):
        st.dataframe(sep["stages"].round(4), hide_index=True, width="stretch")

    st.markdown(f"**Properties at {t_f:g} °F**")
    st.pyplot(properties_plot(res["table"], ps), clear_figure=True)
    st.dataframe(res["table"].style.format(precision=4, na_rep="—")
                 .format("{:,.1f}", subset=["Pressure (psia)"]),
                 hide_index=True, width="stretch")
    st.download_button("Download EOS properties (CSV)", res["table"].to_csv(index=False).encode(),
                       "eos_properties.csv")

    st.subheader("Phase envelope")
    if st.button("Calculate phase envelope", help="Scans phase stability over temperature and "
                                                  "pressure - takes about 20 seconds."):
        ss["prp_show_env"] = True
    if ss.get("prp_show_env"):
        with st.spinner("Tracing the phase envelope..."):
            env, summ = eos_envelope(text, name)
        if env.empty:
            st.info("No two-phase region found between -250 and 1300 °F.")
        else:
            m = st.columns(3)
            m[0].metric("Cricondenbar", f"{summ['cricondenbar_psia']:,.0f} psia",
                        f"at {summ['cricondenbar_T']:.0f} °F", delta_color="off")
            m[1].metric("Cricondentherm", f"~{summ['cricondentherm_T']:.0f} °F")
            m[2].metric("Critical temperature", f"~{summ['critical_T_approx']:.0f} °F"
                        if "critical_T_approx" in summ else "outside scan")
            st.pyplot(envelope_plot(env, float(t_f), ps, summ), clear_figure=True)
            st.caption("Envelope traced at 35 °F steps; the shaded band marks where bubble points "
                       "turn into dew points (approximate critical temperature).")


with tabs[1]:
    st.header("PVT Input Data")
    opts_now = ss.get("options")
    if opts_now is not None and opts_now.method == "eos":
        render_eos(opts_now)
    elif opts_now is not None and opts_now.fluid_type != "oil":
        st.info(f"Black-oil PVT input for {opts_now.label('fluid_type').lower()} has not been built yet. "
                "Black-oil correlations here cover oil and water; choose **Equation of state** in the "
                "Options summary to import a compositional PRP file for any fluid type.")
    else:
        # ------------------------------------------------------- fluid inputs
        st.subheader("Input parameters")
        c = st.columns(4)
        c[0].number_input("Solution GOR (scf/STB)", 1.0, 10000.0, step=10.0, key="pvt_rsb")
        c[1].number_input("Oil Gravity (API)", 5.0, 70.0, step=0.5, key="pvt_api")
        c[2].number_input("Gas Gravity (sp. gravity)", 0.55, 2.0, step=0.01, format="%.3f", key="pvt_gg")
        c[3].number_input("Water Salinity (ppm)", 0.0, 300000.0, step=1000.0, format="%.0f", key="pvt_sal")
        c = st.columns(4)
        c[0].number_input("Mole Percent H2S (percent)", 0.0, 100.0, step=0.1, key="pvt_h2s")
        c[1].number_input("Mole Percent CO2 (percent)", 0.0, 100.0, step=0.1, key="pvt_co2")
        c[2].number_input("Mole Percent N2 (percent)", 0.0, 100.0, step=0.1, key="pvt_n2")
        c[3].checkbox("Correct bubble point for N2, CO2, H2S", value=True, key="pvt_impurity_corr",
                      help="Glaso (1980) non-hydrocarbon correction factors, applied to every "
                           "bubble-point correlation. Impurities also correct the gas Z-factor "
                           "(Wichert-Aziz).")

        # ---------------------------------------------------------- lab data
        st.subheader("Match data")
        c = st.columns([1, 1, 1, 1])
        c[0].number_input("Temperature (deg F)", 40.0, 400.0, step=1.0, key="pvt_t")
        c[1].number_input("Bubble Point (psia)", 0.0, 15000.0, step=10.0, key="pvt_pb",
                          help="Measured bubble point at this temperature. Leave 0 if unknown.")
        c[2].button("Load example data", on_click=load_example, width="stretch",
                    help="Synthetic lab data for trying the tab - not a real fluid.")
        c[3].button("Clear table", on_click=clear_table, width="stretch")
        edited = st.data_editor(
            ss["pvt_table"], key="pvt_table_editor", num_rows="dynamic", width="stretch",
            column_config={col: st.column_config.NumberColumn(f"{col} ({UNITS[col]})", format="%g")
                           for col in bo.TABLE_COLUMNS})
        ss["pvt_table_current"] = edited

        inp, data = current_inputs()
        problems = inp.check()
        for e in problems:
            st.error(e)
        for w in data.check(inp):
            st.warning(w, icon="⚠️")

        # ------------------------------------------------------------- match
        c = st.columns([1, 1, 4])
        c[0].button("Match", type="primary", on_click=run_match, disabled=bool(problems),
                    width="stretch")
        c[1].button("Reset match", on_click=lambda: ss.pop("pvt_match", None), width="stretch")
        res: bo.MatchResult | None = ss.get("pvt_match")
        if res is not None and ss.get("pvt_match_sig") != signature(inp, data):
            st.warning("Inputs or lab data changed since the last match - press **Match** again.")

        if res is not None:
            st.subheader("Match parameters")
            st.caption("Each correlation is corrected as Parameter 1 × correlation + Parameter 2. "
                       "Values near 1 and 0 mean the correlation already fits the lab data. "
                       "Std Deviation is the relative scatter after matching (needs 3 or more points). "
                       "The suggested correlation is highlighted. — means not calculated.")
            sug = res.suggestion
            for prop in ("pb", "rs", "bo", "visc"):
                st.markdown(f"**{bo.PROPS[prop]}**")
                key = "visc_family" if prop == "visc" else "pb_family"
                st.dataframe(grid_view(res, prop, sug.get(key)), width="stretch")

            st.subheader("Suggested correlations")
            c = st.columns(2)
            for col, key, title in ((c[0], "pb_family", "Bubble point, solution GOR, oil FVF"),
                                    (c[1], "visc_family", "Oil viscosity")):
                with col:
                    fams = bo.VISC_FAMILIES if key == "visc_family" else bo.PB_FAMILIES
                    f = sug.get(key)
                    if f:
                        st.success(f"**{title}: {fams[f]}**\n\n{res.rationale[key]}")
                    else:
                        st.info(f"**{title}:** {res.rationale[key]}")
            with st.expander("How the suggestion is made"):
                st.markdown(
                    "- **Score** for each property = the average correction Parameter 1 and 2 apply "
                    "at the lab points + the standard deviation after matching + a penalty when "
                    "Parameter 1 is outside 0.9–1.1. Lower is better.\n"
                    "- Bubble point, solution GOR and oil FVF come from one correlation family, so "
                    "the family score is the mean over the three.\n"
                    "- A correlation whose dataset does not cover this oil's API gravity and "
                    "temperature ranks behind those that do (approximate published ranges).\n"
                    "- With a single point (usually the bubble point) the match splits the "
                    "correction equally between Parameter 1 and Parameter 2.")
                st.markdown("**Bubble point / GOR / FVF families**")
                st.dataframe(res.ranking("pb").round(4), hide_index=True, width="stretch")
                st.markdown("**Viscosity families**")
                st.dataframe(res.ranking("visc").round(4), hide_index=True, width="stretch")

            # ------------------------------------------------- selection
            st.subheader("Selected correlations")
            pb_opts = [f for f in bo.PB_FAMILIES if bo.available("pb", f)]
            v_opts = [f for f in bo.VISC_FAMILIES if bo.available("visc", f)]
            ss.setdefault("pvt_sel_pb", pb_opts[0])
            ss.setdefault("pvt_sel_visc", v_opts[0])
            c = st.columns([1, 1, 1])
            fam = c[0].selectbox("Pb, Rs, Bo correlation", pb_opts, format_func=bo.PB_FAMILIES.get,
                                 key="pvt_sel_pb")
            vfam = c[1].selectbox("Oil viscosity correlation", v_opts, format_func=bo.VISC_FAMILIES.get,
                                  key="pvt_sel_visc")
            use_m = c[2].checkbox("Apply match parameters", value=True, key="pvt_use_match")
            params = {p: res.matches[(p, fam)] for p in ("pb", "rs", "bo")}
            params["visc"] = res.matches[("visc", vfam)]
            raw_model = bo.BlackOilModel(inp, fam, vfam, {})
            model = bo.BlackOilModel(inp, fam, vfam, params if use_m else {})
            ss["pvt_model"] = model
            st.pyplot(match_plot(raw_model, model, data), clear_figure=True)
            st.caption("Dashed: correlation as published. Solid: with match parameters. "
                       "Dotted line: measured bubble point.")

            st.subheader("Calculated PVT")
            t_calc = data.temperature
            pb_m = model.bubble_point(t_calc)
            st.markdown(f"Bubble point at {t_calc:g} °F: **{pb_m:,.1f} psia**"
                        + (f" (measured {data.bubble_point:,.1f})" if data.bubble_point else ""))
            pmax = max(pb_m * 1.5, data.clean()["Pressure"].max() if not data.clean().empty else 0)
            pts = np.unique(np.round(np.concatenate([np.linspace(14.7, pmax, 14), [pb_m]]), 1))
            calc = model.table(t_calc, pts)
            st.dataframe(calc.style.format(precision=4).format("{:,.1f}", subset=["Pressure (psia)"]),
                         hide_index=True, width="stretch")
            st.download_button("Download calculated PVT (CSV)", calc.to_csv(index=False).encode(),
                               "calculated_pvt.csv")
        else:
            st.info("Enter the inputs and lab data, then press **Match** to compare the correlations.")
