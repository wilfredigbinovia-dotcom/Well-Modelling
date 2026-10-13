"""Streamlit front end for wellmodel.

Run from the project folder:   streamlit run app.py

The app is being rebuilt tab by tab. Current tabs:
  1. Options summary - what kind of well is being modelled
  2. PVT Input Data  - black oil: inputs, lab data, correlation matching and suggestion
                       equation of state: PRP file import, flash, saturation point, envelope
  3. IPR Data        - reservoir inflow models for oil and gas/condensate producers
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
from wellmodel import inflow as ipr
from wellmodel import inflow_adv as ipr2
from wellmodel import completion as wcomp
from wellmodel import options as wopt
from wellmodel import plotting as wplt

st.set_page_config(page_title="wellmodel", page_icon="🛢️", layout="wide")
ss = st.session_state
# Number inputs are typed in directly: hide the - / + step buttons everywhere.
st.markdown("""<style>
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {display: none !important;}
[data-testid="stNumberInputContainer"] input {padding-right: 0.75rem;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------- input state
# Inputs keep a persistent copy ("_v_" + key) and are always created with an
# explicit value. This survives models/branches that hide an input, and avoids
# an input snapping to its minimum if a click lands during a rerun.
def wkey(key):
    """Widget key for an input. The version suffix changes when code sets the value, so the
    browser starts a fresh widget instead of sending back its old value."""
    v = ss.get("_ver_" + key, 0)
    return key if v == 0 else f"{key}__{v}"


def val(key, default=None):
    """Current value of an input: live widget state, else the stored copy."""
    k = wkey(key)
    if k in ss and ss[k] is not None:
        return ss[k]
    return ss.get("_v_" + key, default)


def put(key, value):
    """Set an input programmatically; it takes effect on its next render."""
    ss.pop(wkey(key), None)
    ss["_v_" + key] = value
    ss["_ver_" + key] = ss.get("_ver_" + key, 0) + 1


def reset(key):
    """Forget an input's value so its default applies again."""
    ss.pop(wkey(key), None)
    ss.pop("_v_" + key, None)
    ss["_ver_" + key] = ss.get("_ver_" + key, 0) + 1


def num(col, label, key, default, mn=None, mx=None, step=None, fmt=None, help=None):
    cur = float(val(key, default))
    if mn is not None:
        cur = max(cur, float(mn))
    if mx is not None:
        cur = min(cur, float(mx))
    kw = {k: v for k, v in dict(min_value=None if mn is None else float(mn),
                                max_value=None if mx is None else float(mx),
                                step=None if step is None else float(step), format=fmt, help=help).items()
          if v is not None}
    v = col.number_input(label, value=cur, key=wkey(key), **kw)
    ss["_v_" + key] = v
    return v


def sel(col, label, options, key, default=None, **kw):
    options = list(options)
    if wkey(key) in ss and ss[wkey(key)] not in options:
        del ss[wkey(key)]
    cur = val(key, default if default is not None else options[0])
    if cur not in options:
        cur = options[0]
    v = col.selectbox(label, options, index=options.index(cur), key=wkey(key), **kw)
    ss["_v_" + key] = v
    return v


def radio(col, label, options, key, default=None, **kw):
    options = list(options)
    if wkey(key) in ss and ss[wkey(key)] not in options:
        del ss[wkey(key)]
    cur = val(key, default if default is not None else options[0])
    if cur not in options:
        cur = options[0]
    v = col.radio(label, options, index=options.index(cur), key=wkey(key), **kw)
    ss["_v_" + key] = v
    return v


st.title("🛢️ wellmodel")

TAB_NAMES = ["1 · Options summary", "2 · PVT Input Data", "3 · IPR Data"]
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
    st.caption("Completion and sand control are modelled in the IPR Data tab (perforation, damage, "
               "deviation and partial-penetration skins; gravel pack, screen and liner pressure drops).")

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
    for k, v in dict(pvt_rsb=inp.rsb, pvt_api=inp.api, pvt_gg=inp.gas_sg, pvt_sal=inp.salinity_ppm,
                     pvt_h2s=inp.h2s, pvt_co2=inp.co2, pvt_n2=inp.n2, pvt_t=data.temperature,
                     pvt_pb=data.bubble_point).items():
        put(k, float(v))
    t = empty_table()
    t.iloc[:len(data.table)] = data.table[bo.TABLE_COLUMNS].to_numpy(float)
    ss["pvt_table"] = t
    _reset_editor()


def clear_table():
    ss["pvt_table"] = empty_table()
    _reset_editor()


def current_inputs() -> tuple[bo.PVTInputs, bo.MatchData]:
    g = lambda k: float(val(k, INPUT_DEFAULTS[k]))
    inp = bo.PVTInputs(rsb=g("pvt_rsb"), api=g("pvt_api"), gas_sg=g("pvt_gg"), salinity_ppm=g("pvt_sal"),
                       h2s=g("pvt_h2s"), co2=g("pvt_co2"), n2=g("pvt_n2"),
                       correct_impurities=ss.get("pvt_impurity_corr", True))
    data = bo.MatchData(temperature=g("pvt_t"), bubble_point=g("pvt_pb") if g("pvt_pb") > 0 else None,
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
        put("pvt_sel_pb", res.suggestion["pb_family"])
    if res.suggestion.get("visc_family"):
        put("pvt_sel_visc", res.suggestion["visc_family"])


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
    reset("prp_t")
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
        reset("prp_t")
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
    t_f = num(st, "Temperature (deg F)", "prp_t", float(prp.reference_temperature or 200.0), -100.0, 600.0, 1.0,
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
        st.subheader("Input parameters")
        st.caption(f"Gas correlations for {opts_now.label('fluid_type').lower()}: Sutton pseudo-criticals with "
                   "Wichert-Aziz impurity correction, Dranchuk-Abou-Kassem Z and Lee-Gonzalez-Eakin viscosity. "
                   "For condensate dropout, choose **Equation of state** in the Options summary and import a PRP file.")
        c = st.columns(4)
        num(c[0], "Gas Gravity (sp. gravity)", "pvt_gas_sg", 0.70, 0.55, 2.0, 0.01, "%.3f")
        num(c[1], "Mole Percent H2S (percent)", "pvt_h2s", 0.0, 0.0, 100.0, 0.1)
        num(c[2], "Mole Percent CO2 (percent)", "pvt_co2", 0.0, 0.0, 100.0, 0.1)
        num(c[3], "Mole Percent N2 (percent)", "pvt_n2", 0.0, 0.0, 100.0, 0.1)
        num(st.columns(4)[0], "Reservoir temperature (deg F)", "pvt_t", 200.0, 40.0, 500.0, 1.0)
    else:
        # ------------------------------------------------------- fluid inputs
        st.subheader("Input parameters")
        c = st.columns(4)
        num(c[0], "Solution GOR (scf/STB)", "pvt_rsb", INPUT_DEFAULTS["pvt_rsb"], 1.0, 10000.0, 10.0)
        num(c[1], "Oil Gravity (API)", "pvt_api", INPUT_DEFAULTS["pvt_api"], 5.0, 70.0, 0.5)
        num(c[2], "Gas Gravity (sp. gravity)", "pvt_gg", INPUT_DEFAULTS["pvt_gg"], 0.55, 2.0, 0.01, "%.3f")
        num(c[3], "Water Salinity (ppm)", "pvt_sal", INPUT_DEFAULTS["pvt_sal"], 0.0, 300000.0, 1000.0, "%.0f")
        c = st.columns(4)
        num(c[0], "Mole Percent H2S (percent)", "pvt_h2s", 0.0, 0.0, 100.0, 0.1)
        num(c[1], "Mole Percent CO2 (percent)", "pvt_co2", 0.0, 0.0, 100.0, 0.1)
        num(c[2], "Mole Percent N2 (percent)", "pvt_n2", 0.0, 0.0, 100.0, 0.1)
        c[3].checkbox("Correct bubble point for N2, CO2, H2S", value=True, key="pvt_impurity_corr",
                      help="Glaso (1980) non-hydrocarbon correction factors, applied to every "
                           "bubble-point correlation. Impurities also correct the gas Z-factor "
                           "(Wichert-Aziz).")

        # ---------------------------------------------------------- lab data
        st.subheader("Match data")
        c = st.columns([1, 1, 1, 1])
        num(c[0], "Temperature (deg F)", "pvt_t", INPUT_DEFAULTS["pvt_t"], 40.0, 400.0, 1.0)
        num(c[1], "Bubble Point (psia)", "pvt_pb", INPUT_DEFAULTS["pvt_pb"], 0.0, 15000.0, 10.0,
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
            c = st.columns([1, 1, 1])
            fam = sel(c[0], "Pb, Rs, Bo correlation", pb_opts, "pvt_sel_pb", format_func=bo.PB_FAMILIES.get)
            vfam = sel(c[1], "Oil viscosity correlation", v_opts, "pvt_sel_visc", format_func=bo.VISC_FAMILIES.get)
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


# =============================================================== tab 3: IPR
def ipr_test_table(key, rate_label, n=10):
    if key not in ss:
        ss[key] = pd.DataFrame({"Rate": pd.Series([np.nan] * n, dtype=float),
                                "Pwf": pd.Series([np.nan] * n, dtype=float)},
                               index=pd.RangeIndex(1, n + 1, name="Point"))
    return st.data_editor(ss[key], key=key + "_ed", num_rows="dynamic", width="stretch",
                          column_config={"Rate": st.column_config.NumberColumn(rate_label, format="%g"),
                                         "Pwf": st.column_config.NumberColumn("FBHP (psia)", format="%g")})


def default_temperature(opts_now) -> float:
    if opts_now.method == "eos" and ss.get("prp_text"):
        return float(val("prp_t", 200.0))
    return float(val("pvt_t", 200.0))


def oil_pvt_at(opts_now, pr, t_f):
    """Oil properties at reservoir pressure from the PVT tab. Returns (dict, source text) or (None, reason)."""
    if opts_now.method == "eos":
        if not ss.get("prp_text"):
            return None, "No PRP file loaded in the PVT tab."
        res = eos_calculate(ss["prp_text"], ss.get("prp_name", ""), float(t_f))
        if res["sat_type"] == "dew":
            return None, "The PRP fluid has a dew point at this temperature - it is not an oil."
        f = wm_eos.EOSFluid(eos_parse(ss["prp_text"], ss.get("prp_name", "")))
        stb = res["sep"]["stb_per_lbmol"]
        p1, p2 = f.phase_props(f.z, pr, t_f), f.phase_props(f.z, pr + 10, t_f)
        co = -(p2.molar_volume - p1.molar_volume) / 10 / p1.molar_volume
        return ({"mu": p1.viscosity, "bo": p1.molar_volume / wm_eos.FT3_PER_BBL / stb, "rho": p1.density,
                 "ct": max(co, 1e-7), "pb": res["psat"] or 14.7, "gor": res["sep"]["gor"]},
                f"Equation of state ({ss.get('prp_name', 'PRP file')}), single-phase at reservoir pressure")
    model = ss.get("pvt_model")
    label = "Black oil, matched correlations from the PVT tab"
    if model is None:
        inp, _ = current_inputs()
        if inp.check():
            return None, "PVT tab inputs are incomplete."
        model = bo.BlackOilModel(inp, "standing", "beal", {})
        label = "Black oil from PVT tab inputs - not matched yet (Standing / Beal)"
    else:
        label += f" ({bo.PB_FAMILIES[model.family]} / {bo.VISC_FAMILIES[model.visc_family]})"
    tab = model.table(t_f, [pr, pr - 10])
    bo1, bo2 = tab["Oil FVF (RB/STB)"].to_numpy()
    rs1, rs2 = tab["Solution GOR (scf/STB)"].to_numpy()
    bg = tab["Gas FVF (ft3/scf)"].iloc[0] / 5.6146
    # co = -(1/Bo) dBo/dp + (Bg/Bo) dRs/dp  (second term only below the bubble point)
    co = -(bo1 - bo2) / 10 / bo1 + bg * (rs1 - rs2) / 10 / bo1
    return ({"mu": float(tab["Oil viscosity (cp)"].iloc[0]), "bo": float(bo1),
             "rho": float(tab["Oil density (lb/ft3)"].iloc[0]), "ct": float(max(abs(co), 1e-7)),
             "pb": model.bubble_point(t_f), "gor": model.inp.rsb}, label)


def gas_pvt_fn(opts_now, t_f):
    """Return (mu_z(p) function, sg, source text)."""
    if opts_now.method == "eos" and ss.get("prp_text"):
        f = wm_eos.EOSFluid(eos_parse(ss["prp_text"], ss.get("prp_name", "")))

        def mu_z(p):
            out = [f.phase_props(f.z, float(x), t_f) for x in np.atleast_1d(p)]
            return np.array([o.viscosity for o in out]), np.array([o.z for o in out])
        return mu_z, float(f.z @ f.mw / 28.9625), f"Equation of state ({ss.get('prp_name', 'PRP file')}), reservoir fluid"
    sg = float(val("pvt_gas_sg", 0.70))
    inp = bo.PVTInputs(rsb=1, api=40, gas_sg=sg, h2s=float(val("pvt_h2s", 0.0)), co2=float(val("pvt_co2", 0.0)),
                       n2=float(val("pvt_n2", 0.0)))

    def mu_z(p):
        z, _, _, mu = bo.gas_props(inp, p, t_f)
        return np.asarray(mu, float), np.asarray(z, float)
    return mu_z, sg, "Gas correlations from gas gravity (Sutton, Dranchuk-Abou-Kassem, Lee-Gonzalez-Eakin)"



# ---------------------------------------------------------------- IPR helpers
VERTICAL_SKIN_MODELS = {"Darcy", "Fetkovich", "Jones", "Transient", "Back pressure", "Dual Porosity"}
HORIZONTAL_MODELS = {"Horizontal Well - No Flow Boundaries", "Horizontal Well - Transverse Vertical Fractures",
                     "Horizontal Well - friction dP"}
OIL_ADV = ["Hydraulically Fractured Well", "Horizontal Well - No Flow Boundaries",
           "Horizontal Well - Transverse Vertical Fractures", "Horizontal Well - friction dP",
           "Dual Porosity", "Multi-layer", "Multi-layer with dP loss in wellbore"]
GAS_ADV = OIL_ADV + ["Modified Isochronal"]
STILL_NOT = {
    "oil": ["Kuchuk & Goode horizontal and constant-pressure upper boundary (equations not public here - "
            "Babu & Odeh and Joshi are provided)", "SkinAide (ELF proprietary method)",
            "Thermally Induced Fracture (water injectors - injection IPRs come with injector support)"],
    "gas": ["Petroleum Experts multiphase pseudo-pressure (proprietary relative-permeability correction)",
            "Kuchuk & Goode horizontal (Babu & Odeh and Joshi are provided)", "SkinAide (ELF proprietary method)"],
}


def ipr_plot(curve: ipr.IPRCurve, pb=None, raw: ipr.IPRCurve | None = None):
    gas = curve.fluid == "gas"
    scale = 1e-3 if gas else 1.0
    c_key = next((k for k in ("C (Mscf/d/psi^2n)", "C stabilised") if k in curve.params), None)
    show_bp = c_key is not None and curve.tests is not None
    fig, axes = plt.subplots(1, 2 if show_bp else 1, figsize=(11 if show_bp else 7.5, 4.4))
    ax = axes[0] if show_bp else axes
    if raw is not None:
        ax.plot(raw.q * scale, raw.pwf, color=wplt.SERIES[0], lw=1.2, ls="--", label="without sand control")
    ax.plot(curve.q * scale, curve.pwf, color=wplt.SERIES[0], label="IPR")
    lr = getattr(curve, "layer_rates", None)
    if lr is not None:
        for i, col in enumerate(lr.columns):
            ax.plot(lr[col].to_numpy() * scale, curve.pwf[:len(lr)], lw=1.2, color=wplt.SERIES[(i + 1) % 4],
                    label=col)
        ax.axvline(0, color=wplt.INK2, lw=0.8)
    if curve.tests is not None and len(curve.tests):
        ax.scatter(curve.tests["Rate"] * scale, curve.tests["Pwf"], s=50, color=wplt.INK, zorder=4,
                   label="test data")
    if pb and curve.fluid == "oil" and pb < curve.pr:
        ax.axhline(pb, color=wplt.INK2, lw=1, ls=":")
        ax.text(curve.aof * 0.02, pb, "bubble point", ha="left", va="bottom", fontsize=8.5, color=wplt.INK2)
    left = min(0, lr.to_numpy().min() * scale * 1.05) if lr is not None else 0
    ax.set_xlim(left, max(curve.aof * scale * 1.05, 1e-6))
    ax.set_ylim(0, curve.pr * 1.05)
    ax.set_xlabel("Gas rate (MMscf/d)" if gas else "Liquid rate (STB/d)")
    ax.set_ylabel("Flowing bottom-hole pressure (psia)")
    ax.set_title(f"{curve.model}  -  AOF {curve.aof * scale:,.{2 if gas else 0}f} {'MMscf/d' if gas else 'STB/d'}",
                 fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    if show_bp:
        b = axes[1]
        t = curve.tests
        dp = curve.pr ** 2 - t["Pwf"] ** 2
        b.loglog(t["Rate"] * scale, dp, "o", color=wplt.INK, label="test data")
        qq = np.geomspace(max(t["Rate"].min() * 0.5, 1e-3), curve.aof, 50)
        b.loglog(qq * scale, (qq / curve.params[c_key]) ** (1 / curve.params["n"]),
                 color=wplt.SERIES[0], label=f"{'stabilised' if c_key == 'C stabilised' else 'fit'}, n = {curve.params['n']:.3f}")
        if c_key == "C stabilised":
            b.loglog(qq * scale, (qq / curve.c_iso) ** (1 / curve.params["n"]), color=wplt.SERIES[1],
                     ls="--", label="isochronal")
        b.set_xlabel("Gas rate (MMscf/d)")
        b.set_ylabel("Pr² − Pwf² (psi²)")
        b.set_title("Back-pressure plot")
        b.grid(True, which="both", color=wplt.GRID, lw=0.5)
        b.legend(fontsize=8)
    fig.tight_layout()
    return fig


def profile_plot(curve):
    pr_ = curve.profile
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4))
    a.plot(pr_["Distance from heel (ft)"], pr_["Pressure (psia)"], "-o", ms=3, color=wplt.SERIES[0])
    a.set_xlabel("Distance from heel (ft)"); a.set_ylabel("Pressure (psia)")
    a.set_title(f"Pressure along the lateral (heel at {curve.profile_heel_pressure:,.0f} psia)", fontsize=10)
    b.bar(pr_["Distance from heel (ft)"], pr_["Rate per ft"], width=pr_["Distance from heel (ft)"].diff().fillna(
        pr_["Distance from heel (ft)"].iloc[0] * 2).to_numpy() * 0.8, color=wplt.SERIES[0])
    b.set_xlabel("Distance from heel (ft)"); b.set_ylabel("Inflow per ft")
    b.set_title("Inflow profile", fontsize=10)
    fig.tight_layout()
    return fig


DIETZ_HELP = ("Dietz shape factor for the drainage area and well position. Typical values: "
              + "; ".join(f"{k} {v:g}" for k, v in ipr.DIETZ_SHAPES.items()) + ".")


def dietz_input(col):
    return num(col, "Dietz shape factor", "ipr_ca", 31.6, 0.001, 100.0, None, "%.4g", help=DIETZ_HELP)


def reservoir_props(nd=False, skin=True):
    """Darcy-type reservoir inputs; returns dict (skin filled later by the skin panel)."""
    c = st.columns(4)
    r = {"k": num(c[0], "Reservoir permeability (md)", "ipr_k", 50.0, 0.001, 1e5, 1.0),
         "h": num(c[1], "Reservoir thickness (ft)", "ipr_h", 60.0, 0.1, 1e4, 1.0),
         "area": num(c[2], "Drainage area (acres)", "ipr_area", 160.0, 0.1, 1e6, 10.0),
         "rw": num(c[3], "Wellbore radius (ft)", "ipr_rw", 0.354, 0.05, 5.0, 0.001, "%.3f")}
    c = st.columns(4)
    r["ca"] = dietz_input(c[0])
    if nd:
        r["hp"] = num(c[1], "Perforated interval (ft)", "ipr_hp", 60.0, 0.1, 1e4, 1.0)
    return r


def horizontal_inputs():
    c = st.columns(4)
    g = {"k": num(c[0], "Reservoir permeability (md)", "ipr_k", 50.0, 0.001, 1e5, 1.0,
                  help="Horizontal permeability perpendicular to the well (Ky)"),
         "h": num(c[1], "Reservoir thickness (ft)", "ipr_h", 60.0, 0.1, 1e4, 1.0),
         "rw": num(c[2], "Wellbore radius (ft)", "ipr_rw", 0.354, 0.05, 5.0, 0.001, "%.3f"),
         "aniso_h": num(c[3], "Horizontal anisotropy (Ky/Kx)", "ipr_anh", 1.0, 0.01, 100.0, 0.1)}
    c = st.columns(4)
    g["aniso_v"] = num(c[0], "Vertical anisotropy (Kz/Ky)", "ipr_anv", 0.1, 0.001, 10.0, 0.01)
    g["L"] = num(c[1], "Length of well (ft)", "ipr_L", 2000.0, 10.0, 3e4, 50.0)
    g["lx"] = num(c[2], "Length of drainage area Lx (ft)", "ipr_lx", 4000.0, 10.0, 1e5, 100.0)
    g["ly"] = num(c[3], "Width of drainage area Ly (ft)", "ipr_ly", 2000.0, 10.0, 1e5, 100.0)
    c = st.columns(4)
    g["xw"] = num(c[0], "Distance along length edge to centre of well Xw (ft)", "ipr_xw", 2000.0, 0.0, 1e5, 50.0)
    g["yw"] = num(c[1], "Distance along width edge to centre of well Yw (ft)", "ipr_yw", 1000.0, 0.0, 1e5, 50.0)
    g["zw"] = num(c[2], "Distance from bottom of reservoir to centre of well Zw (ft)", "ipr_zw", 30.0, 0.0, 1e4, 1.0)
    return g, c[3]


def skin_panel(o, k, rw, h, gas_or_oil):
    """Mechanical/geometrical + deviation/partial-penetration skin. Returns (total, breakdown dict)."""
    st.markdown("**Skin**")
    cased = o.completion == "cased_hole"
    mech_opts = (["Enter skin by hand", "Karakas & Tariq", "MacLeod"] if cased
                 else ["Enter skin by hand", "Hawkins damaged zone"])
    c = st.columns(3)
    mech = sel(c[0], "Mechanical/geometrical skin", mech_opts, "ipr_mech",
                          help="Choices follow the completion in the Options summary. Locke's nomograph method "
                               "is not available.")
    dev_opts = ["None", "Cinco-Ley deviation + Papatzacos partial penetration"]
    dev = sel(c[1], "Deviation/partial penetration skin", dev_opts, "ipr_devpp")
    parts = {}
    if mech == "Enter skin by hand":
        parts["Mechanical (entered)"] = num(c[2], "Skin", "ipr_skin", 0.0, -6.0, 500.0, 0.5)
    else:
        c = st.columns(4)
        kd = ld = kc = lc = 0.0
        if mech in ("Karakas & Tariq", "MacLeod"):
            dp_ = num(c[0], "Perforation diameter (in)", "ipr_pd", 0.4, 0.05, 2.0, 0.05)
            spf = num(c[1], "Shots per foot", "ipr_spf", 4.0, 0.5, 40.0, 1.0)
            lp = num(c[2], "Perforation length (in)", "ipr_lp", 10.0, 0.5, 60.0, 0.5)
            ph = sel(c[3], "Shot phasing (deg)", [0, 45, 60, 90, 120, 180], "ipr_phasing", 90,
                                help="0 = all shots in one direction (360°)")
            c = st.columns(4)
        ld = num(c[0], "Damaged zone thickness (in)", "ipr_ld", 6.0 if mech == "Hawkins damaged zone" else 0.0,
                 0.0, 120.0, 0.5)
        kd = num(c[1], "Damaged zone permeability (md)", "ipr_kd", max(k * 0.2, 0.001), 0.0001, 1e5, 1.0)
        if mech in ("Karakas & Tariq", "MacLeod"):
            lc = num(c[2], "Crushed zone thickness (in)", "ipr_lc", 0.5, 0.0, 5.0, 0.1)
            kc = num(c[3], "Crushed zone permeability (md)", "ipr_kc", max(k * 0.2, 0.001), 0.0001, 1e5, 1.0)
        kv_kh = num(st.columns(4)[0], "Vertical / horizontal permeability", "ipr_kvkh", 1.0, 0.001, 10.0, 0.05) \
            if mech == "Karakas & Tariq" else 1.0
        if mech == "Karakas & Tariq":
            r = wcomp.karakas_tariq(k, rw, lp / 12, dp_ / 24, spf, ph, kv_kh, ld / 12, kd, lc / 12, kc)
            parts.update({"Perforation (K&T)": r["perforation"] if r["damage"] == 0 else k / kd * r["perforation"],
                          "Damaged zone": r["damage"], "Crushed zone": r["crushed zone"]})
        elif mech == "MacLeod":
            r = wcomp.macleod(k, rw, lp / 12, dp_ / 24, spf, ld / 12, kd, lc / 12, kc)
            parts.update({"Damaged zone": r["damage"], "Crushed zone": r["crushed zone"]})
        else:
            parts["Damaged zone (Hawkins)"] = wcomp.hawkins(k, rw, ld / 12, kd)
    if dev != "None":
        c = st.columns(4)
        th = num(c[0], "Well deviation (deg)", "ipr_dev", 0.0, 0.0, 75.0, 1.0)
        hp = num(c[1], "Perforated interval (ft)", "ipr_hpp", h, 0.1, 1e4, 1.0)
        zt = num(c[2], "Top of reservoir to top of perforations (ft)", "ipr_ztop", 0.0, 0.0, 1e4, 1.0)
        kvkh2 = num(c[3], "Vertical / horizontal permeability ", "ipr_kvkh2", 1.0, 0.001, 10.0, 0.05)
        if hp + zt > h + 1e-6:
            st.warning("Perforated interval extends below the reservoir.")
        parts["Deviation (Cinco-Ley)"] = wcomp.deviation_skin(th, h, rw, kvkh2)
        parts["Partial penetration (Papatzacos)"] = wcomp.partial_penetration_skin(h, min(hp, h), zt, rw, kvkh2)
    total = float(sum(parts.values()))
    st.caption("Skin: " + " · ".join(f"{k_} {v:+.2f}" for k_, v in parts.items()) + f"  →  **total {total:+.2f}**")
    return total, parts


def sand_control_panel(o, fluid, h_default):
    kind = o.sand_control
    if kind == "none":
        return wcomp.SandControl("none")
    st.subheader("Sand control")
    if kind == "gravel_pack":
        kind = "gravel_pack_cased" if o.completion == "cased_hole" else "gravel_pack_open"
    sc = wcomp.SandControl(kind)
    st.caption(f"{sc.label()} - from the Options summary.")
    c = st.columns(4)
    p = {"interval": num(c[0], "Completed interval (ft)", "sc_int", h_default, 0.1, 1e4, 1.0)}
    if kind == "gravel_pack_cased":
        p["perf_diameter_in"] = num(c[1], "Perforation diameter (in)", "sc_pd", 0.75, 0.1, 3.0, 0.05)
        p["spf"] = num(c[2], "Shots per foot", "sc_spf", 12.0, 0.5, 40.0, 1.0)
        p["perf_efficiency"] = num(c[3], "Perforation efficiency (fraction open)", "sc_eff", 0.8, 0.01, 1.0, 0.05)
        c = st.columns(4)
        p["gravel_perm"] = num(c[0], "Gravel pack permeability (md)", "sc_kg", 120000.0, 100.0, 1e7, 1000.0)
        p["pack_length_in"] = num(c[1], "Gravel pack length - screen to end of cement (in)", "sc_L", 4.0, 0.1, 40.0, 0.5)
    elif kind in ("gravel_pack_open", "pre_packed_screen"):
        lab = "Gravel pack" if kind == "gravel_pack_open" else "Pre-pack"
        p["gravel_perm"] = num(c[1], f"{lab} permeability (md)", "sc_kg" if kind == "gravel_pack_open" else "sc_kpp",
                               120000.0 if kind == "gravel_pack_open" else 50000.0, 100.0, 1e7, 1000.0)
        p["outer_radius_in"] = num(c[2], "Outer radius - hole or screen (in)", "sc_ro",
                                   4.25 if kind == "gravel_pack_open" else 2.75, 0.5, 20.0, 0.25)
        p["thickness_in"] = num(c[3], f"{lab} thickness (in)", "sc_th", 1.25 if kind == "gravel_pack_open" else 0.5,
                                0.05, 10.0, 0.05)
    elif kind == "wire_wrapped_screen":
        p["screen_od_in"] = num(c[1], "Screen OD (in)", "sc_od", 5.5, 1.0, 12.0, 0.125)
        p["slot_width_in"] = num(c[2], "Slot (gauge) width (in)", "sc_sw", 0.012, 0.001, 0.2, 0.001, "%.3f")
        p["wire_width_in"] = num(c[3], "Wire width (in)", "sc_ww", 0.09, 0.01, 0.5, 0.01)
    else:
        p["slots_per_ft"] = num(c[1], "Slots per foot", "sc_spf2", 30.0, 1.0, 500.0, 1.0)
        p["slot_width_in"] = num(c[2], "Slot width (in)", "sc_sw2", 0.02, 0.001, 0.5, 0.001, "%.3f")
        p["slot_length_in"] = num(c[3], "Slot length (in)", "sc_sl", 2.0, 0.1, 12.0, 0.1)
    sc.params = p
    return sc


def multilayer_table(fluid, with_dp):
    key = f"ipr_layers_{fluid}_{int(with_dp)}"
    if key not in ss:
        base = {"Type": ["Perforated", "Blank", "Perforated"],
                "Layer pressure (psia)": [3000.0, np.nan, 3100.0]}
        if fluid == "oil":
            base.update({"IPR model": ["PI", None, "Darcy"], "PI (STB/d/psi)": [1.5, np.nan, np.nan]})
        else:
            base.update({"IPR model": ["C and n", None, "Jones"], "C (Mscf/d/psi^2n)": [0.01, np.nan, np.nan],
                         "n": [0.85, np.nan, np.nan]})
        base.update({"Permeability (md)": [np.nan, np.nan, 30.0], "Thickness (ft)": [40.0, 20.0, 30.0],
                     "Skin": [0.0, np.nan, 2.0]})
        if with_dp:
            base.update({"MD (ft)": [9300.0, 9318.0, 9350.0], "TVD (ft)": [9300.0, 9318.0, 9350.0],
                         "Flowing ID (in)": [3.0, 3.0, 3.0], "Roughness (in)": [0.0006, 0.0006, 0.0006]})
        ss[key] = pd.DataFrame(base, index=pd.RangeIndex(1, 4, name="Layer"))
    models = ["PI", "Darcy"] if fluid == "oil" else ["C and n", "Jones"]
    return st.data_editor(ss[key], key=key + "_ed", num_rows="dynamic", width="stretch", column_config={
        "Type": st.column_config.SelectboxColumn(options=["Perforated", "Blank"], required=True),
        "IPR model": st.column_config.SelectboxColumn(options=models)})


def zones_table():
    key = "ipr_zones"
    if key not in ss:
        ss[key] = pd.DataFrame({"Type": ["Perforated"] * 4 + ["Blank", "Perforated"],
                                "Length (ft)": [300.0, 300, 300, 300, 200, 600],
                                "Permeability (md)": [50.0, 50, 40, 40, np.nan, 60],
                                "Skin": [0.0, 0, 1, 1, np.nan, 0],
                                "Flowing ID (in)": [4.0] * 6, "Roughness (in)": [0.0006] * 6},
                               index=pd.RangeIndex(1, 7, name="Zone (heel → toe)"))
    return st.data_editor(ss[key], key=key + "_ed", num_rows="dynamic", width="stretch", column_config={
        "Type": st.column_config.SelectboxColumn(options=["Perforated", "Open Hole", "Blank"], required=True)})


def sc_props(fluid, pvt, gasp, pr, t_f):
    if fluid == "oil":
        return {"mu": pvt["mu"], "bo": pvt["bo"], "rho": pvt["rho"]}
    mu_r, z_r, sg = gasp["mu_r"], gasp["z_r"], gasp["sg"]
    tr = t_f + 459.67
    return {"mu": mu_r, "z": z_r, "sg": sg, "t_f": t_f, "pr": pr,
            "rho_g": 28.9625 * sg * pr / (z_r * 10.7316 * tr), "bg": 0.02827 * z_r * tr / pr}


with tabs[2]:
    st.header("IPR Data")
    o = ss.get("options")
    if o is None or o.errors:
        st.info("Fix the configuration in the Options summary first.")
    elif o.is_injector:
        st.info("Injection IPRs are not built yet - this tab currently covers producers.")
    else:
        gas = o.is_gas
        fluid = "gas" if gas else "oil"
        models = (ipr.GAS_MODELS + GAS_ADV) if gas else (ipr.OIL_MODELS + OIL_ADV)
        st.caption(f"{'Gas and condensate' if gas else 'Oil and water'} inflow models - from the fluid type "
                   f"chosen in the Options summary. Completion: {o.label('completion').lower()}, sand control: "
                   f"{o.label('sand_control').lower()}. Pressures in psia.")
        c = st.columns([1.3, 2])
        model = sel(c[0], "Reservoir model", models, f"ipr_model_{fluid}")
        with c[1].expander("Models not available"):
            st.markdown("\n".join(f"- {m}" for m in STILL_NOT[fluid]))

        # ---------------------------------------------------- reservoir data
        st.subheader("Reservoir data")
        c = st.columns(4)
        pr = num(c[0], "Reservoir pressure (psia)", "ipr_pr", 4000.0 if gas else 3000.0, 15.0, 30000.0, 10.0)
        t_pvt = default_temperature(o)
        if c[1].toggle("Temperature from PVT", value=True, key="ipr_t_from_pvt"):
            t_f = t_pvt
            c[1].caption(f"Reservoir temperature {t_f:g} °F (PVT tab)")
        else:
            t_f = num(c[1], "Reservoir temperature (deg F)", "ipr_t", t_pvt, 40.0, 500.0, 1.0)
        wc, cgr, wgr = 0.0, None, None
        if gas:
            wgr = num(c[2], "Water gas ratio (STB/MMscf)", "ipr_wgr", 0.0, 0.0, 1e5, 1.0)
            if o.method == "eos" and ss.get("prp_text"):
                r_ = eos_calculate(ss["prp_text"], ss.get("prp_name", ""), float(t_f))
                cgr_pvt = float(r_["sep"]["cgr"]) if np.isfinite(r_["sep"]["cgr"]) else 0.0
                if c[3].toggle("CGR from PVT", value=True, key="ipr_cgr_from_pvt"):
                    cgr = cgr_pvt
                    c[3].caption(f"{cgr:,.1f} STB/MMscf (separator flash of the PRP fluid)")
                else:
                    cgr = num(c[3], "Condensate gas ratio (STB/MMscf)", "ipr_cgr", round(cgr_pvt, 2), 0.0, 1e5, 1.0)
            else:
                cgr = num(c[3], "Condensate gas ratio (STB/MMscf)", "ipr_cgr", 0.0, 0.0, 1e5, 1.0)
        elif model not in ("Multi-layer", "Multi-layer with dP loss in wellbore"):
            wc = num(c[2], "Water cut (%)", "ipr_wc", 0.0, 0.0, 100.0, 1.0) / 100
        if abs(t_f - t_pvt) > 0.5:
            st.caption(f"Note: the PVT tab uses {t_pvt:g} °F; PVT properties here are evaluated at {t_f:g} °F.")

        # ------------------------------------------- PVT (from the PVT tab)
        pvt, gasp = None, {}
        if gas:
            mu_z, sg, _ = gas_pvt_fn(o, t_f)
            mu_r, z_r = (float(v[0]) for v in mu_z(np.array([pr])))
            gasp = {"mu_z": mu_z, "sg": sg, "mu_r": mu_r, "z_r": z_r, "t_f": t_f}
        else:
            pvt, why = oil_pvt_at(o, pr, t_f)
            if pvt is None:
                st.info(why + " Complete the PVT Input Data tab to calculate the IPR.")
        # ------------------------------------------------------- model data
        if gas or pvt is not None:
            st.subheader(f"Model data - {model}")
            curve, err, skin_parts, kh_for_skin = None, None, {}, None
            oil_kw = {} if gas else {"pb": pvt["pb"], "bo": pvt["bo"]}
            gas_kw = {"t_f": t_f, "z": gasp.get("z_r"), "sg": gasp.get("sg")} if gas else {}
            mu_used = gasp.get("mu_r") if gas else (pvt["mu"] if pvt else None)
            try:
                if model in VERTICAL_SKIN_MODELS:
                    r = reservoir_props(nd=(model == "Jones"))
                    skin, skin_parts = skin_panel(o, r["k"], r["rw"], r["h"], fluid)
                    kh_for_skin = r["k"] * r["h"]
                if not gas and model in ipr.OIL_MODELS:
                    pb = pvt["pb"]
                    if model == "P.I. Entry":
                        j = num(st.columns(4)[0], "Productivity index (STB/d/psi)", "ipr_j", 2.0, 0.001, 1e4, 0.1)
                        curve = ipr.pi_entry(pr, pb, j)
                    elif model in ("Vogel", "Composite"):
                        c = st.columns(4)
                        qt = num(c[0], "Test rate (STB/d, liquid)", "ipr_qt", 1500.0, 0.1, 1e6, 10.0)
                        pt = num(c[1], "Test FBHP (psia)", "ipr_pt", 1500.0, 0.0, 3e4, 10.0)
                        curve = ipr.vogel(pr, pb, qt, pt) if model == "Vogel" else ipr.composite(pr, pb, qt, pt, wc)
                    elif model == "Darcy":
                        curve = ipr.darcy(pr, pb, r["k"], r["h"], r["area"], r["ca"], r["rw"], skin, pvt["mu"], pvt["bo"])
                    elif model == "Fetkovich":
                        kro = num(st.columns(4)[0], "Relative permeability of oil", "ipr_kro", 1.0, 0.001, 1.0, 0.05)
                        curve = ipr.fetkovich(pr, pb, r["k"], r["h"], r["area"], r["ca"], r["rw"], skin,
                                              pvt["mu"], pvt["bo"], kro)
                    elif model == "Jones":
                        curve = ipr.jones_oil(pr, r["k"], r["h"], r["area"], r["ca"], r["rw"], skin, pvt["mu"],
                                              pvt["bo"], pvt["rho"], r["hp"])
                    elif model == "Transient":
                        c = st.columns(4)
                        phi = num(c[0], "Porosity (fraction)", "ipr_phi", 0.2, 0.01, 0.5, 0.01)
                        tdays = num(c[1], "Time (days)", "ipr_tdays", 30.0, 0.5, 1e5, 1.0)
                        curve = ipr.transient_oil(pr, pb, r["k"], r["h"], r["area"], r["ca"], r["rw"], skin,
                                                  pvt["mu"], pvt["bo"], phi, pvt["ct"], tdays)
                    elif model in ("Multi-rate Fetkovich", "Multi-rate Jones"):
                        st.caption("Up to 10 tests. If the reservoir pressure is at or below the highest test FBHP, "
                                   "it is calculated from the tests.")
                        t = ipr_test_table("ipr_tests_oil", "Liquid rate (STB/d)")
                        curve = (ipr.multirate_fetkovich if model == "Multi-rate Fetkovich"
                                 else ipr.multirate_jones_oil)(pr, t)
                    elif model == "External entry":
                        st.caption("Rate against flowing pressure; the highest pressure should have zero rate.")
                        curve = ipr.external(ipr_test_table("ipr_ext_oil", "Liquid rate (STB/d)", 15), "oil")
                elif gas and model in ipr.GAS_MODELS:
                    if model in ("Jones", "Back pressure"):
                        if model == "Jones":
                            c = st.columns(4)
                            dm = sel(c[0], "Non-Darcy flow factor", ["Calculated", "Entered"], "ipr_d_method")
                            d = num(c[1], "D (1/(Mscf/d))", "ipr_d", 1e-4, 0.0, 1.0, None, "%.3e") if dm == "Entered" else None
                            curve = ipr.jones_gas(pr, r["k"], r["h"], r["area"], r["ca"], r["rw"], skin, t_f,
                                                  gasp["mu_r"], gasp["z_r"], gasp["sg"], r["hp"], d)
                        else:
                            n = num(st.columns(4)[0], "Exponent n", "ipr_n_bp", 0.85, 0.5, 1.0, 0.01)
                            curve = ipr.back_pressure(pr, r["k"], r["h"], r["area"], r["ca"], r["rw"], skin, t_f,
                                                      gasp["mu_r"], gasp["z_r"], n)
                    elif model == "Forchheimer":
                        c = st.columns(4)
                        a = num(c[0], "a - turbulent (psi²/(Mscf/d)²)", "ipr_fa", 0.003, 0.0, 1e6, None, "%.4e")
                        b = num(c[1], "b - laminar (psi²/(Mscf/d))", "ipr_fb", 250.0, 1e-9, 1e9, None, "%.4e")
                        curve = ipr.forchheimer(pr, a, b)
                    elif model == "C and n":
                        c = st.columns(4)
                        curve = ipr.c_and_n(pr, num(c[0], "C (Mscf/d/psi^2n)", "ipr_c", 0.01, 1e-12, 1e6, None, "%.4e"),
                                            num(c[1], "n", "ipr_n", 0.85, 0.5, 1.0, 0.01))
                    elif model in ("Multi-rate C and n", "Multi-rate Jones"):
                        st.caption("Up to 10 tests, rate in Mscf/d. If the reservoir pressure is at or below the "
                                   "highest test FBHP, it is calculated from the tests (3+ points needed).")
                        t = ipr_test_table("ipr_tests_gas", "Gas rate (Mscf/d)")
                        curve = (ipr.multirate_c_and_n if model == "Multi-rate C and n" else ipr.multirate_jones_gas)(pr, t)
                    elif model in ("Forchheimer with pseudo-pressure", "Multi-rate Forchheimer with pseudo-pressure"):
                        psi = ipr.pseudo_pressure_fn(gasp["mu_z"], pr, n=120 if o.method == "eos" else 300)
                        if model == "Forchheimer with pseudo-pressure":
                            c = st.columns(4)
                            curve = ipr.forchheimer_pseudo(
                                pr, num(c[0], "a - laminar (psi²/cp/(Mscf/d))", "ipr_pa", 1e5, 1e-9, 1e12, None, "%.4e"),
                                num(c[1], "b - turbulent (psi²/cp/(Mscf/d)²)", "ipr_pb2", 1.0, 0.0, 1e12, None, "%.4e"), psi)
                        else:
                            st.caption("Up to 10 tests, rate in Mscf/d. Reservoir pressure must exceed every test FBHP.")
                            curve = ipr.multirate_forchheimer_pseudo(pr, ipr_test_table("ipr_tests_gas_ps", "Gas rate (Mscf/d)"), psi)
                    elif model == "External entry":
                        st.caption("Gas rate (Mscf/d) against flowing pressure; highest pressure should have zero rate.")
                        curve = ipr.external(ipr_test_table("ipr_ext_gas", "Gas rate (Mscf/d)", 15), "gas")
                    elif model == "Dual Porosity":
                        pass
                # ------------------------------------------------ advanced models
                if model == "Hydraulically Fractured Well":
                    r = reservoir_props()
                    c = st.columns(4)
                    tdays = num(c[0], "Time since production started (days)", "ipr_tdays", 30.0, 0.5, 1e5, 1.0)
                    xf = num(c[1], "Fracture half length (ft)", "ipr_xf", 200.0, 1.0, 5000.0, 10.0)
                    fcd = num(c[2], "Dimensionless fracture conductivity", "ipr_fcd", 10.0, 0.1, 1e4, 1.0,
                              help="FCD = kf bf / (k xf)")
                    face = num(c[3], "Fracture face skin", "ipr_fface", 0.0, 0.0, 50.0, 0.1)
                    c = st.columns(4)
                    phi = num(c[0], "Porosity (fraction)", "ipr_phi", 0.15, 0.01, 0.5, 0.01)
                    if gas:
                        hf = num(c[1], "Fracture height (ft)", "ipr_hf", r["h"], 1.0, 1e4, 1.0)
                        ct = num(c[2], "Total compressibility (1/psi)", "ipr_ctg", 1 / pr, 1e-7, 1e-1, None, "%.2e",
                                 help="Defaults to 1/Pr for gas")
                        curve = ipr2.fractured_well("gas", pr, r["k"], r["h"], r["area"], r["ca"], r["rw"], xf, fcd, tdays,
                                                    phi, ct, gasp["mu_r"], face, hf=hf, **gas_kw)
                    else:
                        curve = ipr2.fractured_well("oil", pr, r["k"], r["h"], r["area"], r["ca"], r["rw"], xf, fcd, tdays,
                                                    phi, pvt["ct"], pvt["mu"], face, **oil_kw)
                elif model in ("Horizontal Well - No Flow Boundaries", "Horizontal Well - Transverse Vertical Fractures"):
                    g, spare = horizontal_inputs()
                    hz_skin = num(spare, "Skin (along the well)", "ipr_hz_skin", 0.0, -5.0, 200.0, 0.5)
                    if model == "Horizontal Well - No Flow Boundaries":
                        meth = radio(st, "Horizontal well model", ["Babu & Odeh", "Joshi"], "ipr_hz_method",
                                     horizontal=True, captions=["Closed box, pseudo-steady state",
                                                                   "Steady state, constant pressure at drainage radius"])
                        curve = ipr2.horizontal_well(fluid, meth, pr, g["k"], g["h"], g["rw"], g["aniso_h"], g["aniso_v"],
                                                     g["L"], g["lx"], g["ly"], g["xw"], g["yw"], g["zw"], mu_used,
                                                     hz_skin, **(gas_kw if gas else oil_kw))
                    else:
                        c = st.columns(4)
                        xf = num(c[0], "Fracture half length (ft)", "ipr_xf", 200.0, 1.0, 5000.0, 10.0)
                        fcd = num(c[1], "Dimensionless fracture conductivity", "ipr_fcd", 10.0, 0.1, 1e4, 1.0)
                        nfr = num(c[2], "Number of fractures", "ipr_nfr", 4.0, 1.0, 100.0, 1.0)
                        ca_f = dietz_input(c[3])
                        curve = ipr2.transverse_fractures(fluid, pr, g["k"], g["h"], g["rw"], g["aniso_h"], g["aniso_v"],
                                                          g["L"], g["lx"], g["ly"], g["xw"], g["yw"], g["zw"], mu_used,
                                                          xf, fcd, nfr, ca_f, hz_skin, **(gas_kw if gas else oil_kw))
                elif model == "Horizontal Well - friction dP":
                    g, spare = horizontal_inputs()
                    st.caption("Overall productivity from Babu & Odeh; zones split it by permeability x length and "
                               "zone skin. Zone lengths should add up to the well length.")
                    z = zones_table().dropna(subset=["Type", "Length (ft)"])
                    zones = [{"type": row["Type"], "length": float(row["Length (ft)"]),
                              "k": float(row["Permeability (md)"]) if pd.notna(row["Permeability (md)"]) else g["k"],
                              "skin": float(row["Skin"]) if pd.notna(row["Skin"]) else 0.0,
                              "id_in": float(row["Flowing ID (in)"]), "rough_in": float(row["Roughness (in)"])}
                             for _, row in z.iterrows()]
                    total_len = sum(zz["length"] for zz in zones)
                    if abs(total_len - g["L"]) > 1:
                        st.warning(f"Zone lengths add up to {total_len:,.0f} ft but the well length is {g['L']:,.0f} ft.")
                    kh_eq, D0, _ = ipr2.babu_odeh(g["k"], g["h"], g["rw"], g["aniso_h"], g["aniso_v"], g["L"], g["lx"],
                                                  g["ly"], g["xw"], g["yw"], g["zw"], 0.0)
                    ph = num(spare, "Heel pressure for profile plot (psia)", "ipr_heelp", round(pr * 0.5), 14.7, 3e4, 10.0)
                    props = (dict(mu=pvt["mu"], bo=pvt["bo"], rho=pvt["rho"], wc=wc) if not gas
                             else dict(mu_z=gasp["mu_z"], sg=gasp["sg"], t_f=t_f, mu_r=gasp["mu_r"], z_r=gasp["z_r"]))
                    curve = ipr2.horizontal_friction(fluid, pr, zones, kh_eq, D0, g["k"], props,
                                                     pvt["pb"] if not gas else None, profile_at=ph)
                elif model == "Dual Porosity":
                    if not gas:
                        pass  # reservoir_props + skin_panel already rendered above
                    c = st.columns(4)
                    phi = num(c[0], "Porosity (fraction)", "ipr_phi", 0.1, 0.001, 0.5, 0.01)
                    tdays = num(c[1], "Time since production started (days)", "ipr_tdays", 30.0, 0.01, 1e5, 1.0)
                    omega = num(c[2], "Storativity ratio (omega)", "ipr_omega", 0.1, 1e-4, 0.9999, 0.01, "%.4f")
                    lam = num(c[3], "Interporosity coefficient (lambda)", "ipr_lambda", 1e-6, 1e-12, 1.0, None, "%.2e")
                    if gas:
                        c = st.columns(4)
                        swc = num(c[0], "Connate water saturation", "ipr_swc", 0.2, 0.0, 0.9, 0.01)
                        ct = (1 - swc) / pr
                        curve = ipr2.dual_porosity("gas", pr, r["k"], r["h"], r["area"], r["ca"], r["rw"], phi, ct, tdays,
                                                   omega, lam, gasp["mu_r"], skin, **gas_kw)
                    else:
                        curve = ipr2.dual_porosity("oil", pr, r["k"], r["h"], r["area"], r["ca"], r["rw"], phi, pvt["ct"],
                                                   tdays, omega, lam, pvt["mu"], skin, **oil_kw)
                elif model in ("Multi-layer", "Multi-layer with dP loss in wellbore"):
                    with_dp = model.endswith("wellbore")
                    c = st.columns(4)
                    area = num(c[0], "Drainage area (acres)", "ipr_area", 160.0, 0.1, 1e6, 10.0)
                    rw = num(c[1], "Wellbore radius (ft)", "ipr_rw", 0.354, 0.05, 5.0, 0.001, "%.3f")
                    ca = dietz_input(c[2])
                    if with_dp:
                        c = st.columns(4)
                        top_md = num(c[0], "TOP node MD (ft)", "ipr_top_md", 9275.0, 0.0, 4e4, 10.0)
                        top_tvd = num(c[1], "TOP node TVD (ft)", "ipr_top_tvd", 9275.0, 0.0, 4e4, 10.0)
                        if not gas:
                            wc = num(c[2], "Water cut (%)", "ipr_wc", 0.0, 0.0, 100.0, 1.0) / 100
                    st.caption("Layer data. Darcy/Jones layers use permeability, thickness and skin; PI / C and n "
                               "layers use their own coefficients. Blank layers carry no inflow."
                               + (" Depths are those of each layer's inflow point; ID and roughness describe the "
                                  "pipe above the layer." if with_dp else ""))
                    tab_ = multilayer_table(fluid, with_dp)
                    layers = []
                    for _, row in tab_.iterrows():
                        if row.get("Type") != "Perforated" or pd.isna(row.get("Layer pressure (psia)")):
                            continue
                        L = {"pr": float(row["Layer pressure (psia)"]), "model": row.get("IPR model")}
                        if not gas:
                            if L["model"] == "Darcy":
                                L["j"] = ipr.darcy_pi(float(row["Permeability (md)"]), float(row["Thickness (ft)"]),
                                                      pvt["mu"], pvt["bo"], area, ca, rw, float(row["Skin"] or 0))
                            else:
                                L["j"] = float(row["PI (STB/d/psi)"])
                        else:
                            if L["model"] == "Jones":
                                a_, b_, _ = ipr.jones_gas_coeffs(float(row["Permeability (md)"]), float(row["Thickness (ft)"]),
                                                                 area, ca, rw, float(row["Skin"] or 0), t_f, gasp["mu_r"],
                                                                 gasp["z_r"], gasp["sg"], float(row["Thickness (ft)"]))
                                L.update(a=a_, b=b_)
                            else:
                                L.update(c=float(row["C (Mscf/d/psi^2n)"]), n=float(row["n"]))
                        if with_dp:
                            L.update(md=float(row["MD (ft)"]), tvd=float(row["TVD (ft)"]),
                                     id_in=float(row["Flowing ID (in)"]), rough_in=float(row["Roughness (in)"]))
                        layers.append(L)
                    if with_dp:
                        props = (dict(mu=pvt["mu"], bo=pvt["bo"], rho=pvt["rho"], wc=wc) if not gas
                                 else dict(mu_z=gasp["mu_z"], sg=gasp["sg"], t_f=t_f))
                        curve = ipr2.multilayer_dp(fluid, layers, top_md, top_tvd, props,
                                                   pvt["pb"] if not gas else None, n=25 if gas else 40)
                    else:
                        curve = ipr2.multilayer(fluid, layers, pvt["pb"] if not gas else None)
                elif model == "Modified Isochronal":
                    c = st.columns(4)
                    k = num(c[0], "Gas permeability (md)", "ipr_k", 50.0, 0.001, 1e5, 1.0)
                    phi = num(c[1], "Porosity (fraction)", "ipr_phi", 0.15, 0.01, 0.5, 0.01)
                    swc = num(c[2], "Connate water saturation", "ipr_swc", 0.2, 0.0, 0.9, 0.01)
                    rw = num(c[3], "Wellbore radius (ft)", "ipr_rw", 0.354, 0.05, 5.0, 0.001, "%.3f")
                    c = st.columns(4)
                    area = num(c[0], "Drainage area (acres)", "ipr_area", 160.0, 0.1, 1e6, 10.0)
                    tfl = num(c[1], "Flow period duration (hours)", "ipr_tfl", 12.0, 0.1, 1e4, 1.0)
                    tsh = num(c[2], "Shut-in period duration (hours)", "ipr_tsh", 12.0, 0.0, 1e4, 1.0)
                    st.caption("Flow periods in test order: rate (Mscf/d) and FBHP at the end of each period. "
                               "The reservoir pressure above is the initial (stabilised) shut-in pressure.")
                    t = ipr_test_table("ipr_tests_iso", "Gas rate (Mscf/d)", 6)
                    curve = ipr2.modified_isochronal(pr, t, tfl, tsh, k, phi, swc, gasp["mu_r"], rw,
                                                     np.sqrt(area * 43560 / np.pi))
            except (ValueError, KeyError, TypeError) as e:
                err = str(e) or e.__class__.__name__

            # ----------------------------------------------------- sand control
            raw_curve = curve
            sc = sand_control_panel(o, fluid, float(val("ipr_h", 60.0)))
            if curve is not None and err is None and sc.kind != "none":
                try:
                    dp_fn = wcomp.sand_control_dp(sc, fluid, sc_props(fluid, pvt, gasp, pr, t_f))
                    curve = wcomp.apply_sand_control(curve, dp_fn)
                    for attr in ("layer_rates", "profile", "profile_heel_pressure", "isochronal", "c_iso"):
                        if hasattr(raw_curve, attr):
                            setattr(curve, attr, getattr(raw_curve, attr))
                except (ValueError, KeyError) as e:
                    err = f"Sand control: {e}"

            # ----------------------------------------------------------- results
            st.subheader("Results")
            if err:
                st.warning(err)
            elif curve is None:
                st.info("Enter the model data above.")
            else:
                ss["ipr_curve"] = curve
                for n_ in curve.notes:
                    st.info(n_)
                items = [(k_, v) for k_, v in curve.params.items() if isinstance(v, (int, float, np.floating))]
                m = st.columns(min(len(items) + 1, 5))
                m[0].metric("AOF", f"{curve.aof / 1e3:,.3f} MMscf/d" if gas else f"{curve.aof:,.0f} STB/d")
                for col, (k_, v) in zip(m[1:], items[:4]):
                    col.metric(k_, f"{v:.4g}" if abs(v) < 1e4 else f"{v:,.0f}")
                if len(items) > 4:
                    st.caption(" · ".join(f"{k_}: {v:.4g}" for k_, v in items[4:]))
                st.pyplot(ipr_plot(curve, None if gas else pvt["pb"],
                                   raw_curve if sc.kind != "none" else None), clear_figure=True)
                if hasattr(curve, "profile"):
                    st.pyplot(profile_plot(curve), clear_figure=True)
                table = curve.table(cgr=cgr, wgr=wgr) if gas else curve.table(wc=wc, gor=pvt.get("gor"))
                if kh_for_skin and skin_parts:
                    s_tot = sum(skin_parts.values())
                    if gas:
                        coef = 1422 * (t_f + 459.67) * gasp["mu_r"] * gasp["z_r"] * s_tot / kh_for_skin
                        table["dP skin (psi)"] = np.sqrt(table["Pwf (psia)"] ** 2 + coef * curve.q) - table["Pwf (psia)"]
                    else:
                        table["dP skin (psi)"] = 141.2 * pvt["mu"] * pvt["bo"] * s_tot / kh_for_skin * curve.q
                if hasattr(curve, "dp_sand_control"):
                    table["dP sand control (psi)"] = curve.dp_sand_control
                lr = getattr(curve, "layer_rates", None)
                if lr is not None:
                    for col in lr.columns:
                        table[f"{col} rate"] = lr[col].to_numpy() / (1e3 if gas else 1)
                with st.expander("IPR table", expanded=False):
                    st.dataframe(table.style.format(precision=2), hide_index=True, width="stretch")
                if hasattr(curve, "isochronal"):
                    with st.expander("Modified isochronal corrections", expanded=True):
                        st.dataframe(curve.isochronal.style.format(precision=4), hide_index=True, width="stretch")
                st.download_button("Download IPR (CSV)", table.to_csv(index=False).encode(), "ipr.csv")
