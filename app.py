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
import inspect

from scipy.optimize import brentq

from wellmodel import inflow as _ipr_mod
from wellmodel import inflow_adv as _ipr2_mod
from wellmodel import completion as wcomp
from wellmodel import mpp as _mpp_mod
from wellmodel import vlp as wvlp
from wellmodel import system as wsys
from wellmodel import surface as wsurf
from wellmodel import project as wproj
from wellmodel import liftcurves as wlc
from wellmodel import gradient as wgrad
from wellmodel import choke as wchoke
from wellmodel import gaslift as wgl
try:                                    # PROSPER-style grids with per-cell locking; optional dependency
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    HAVE_AGGRID = True
except ImportError:                     # falls back to st.data_editor
    HAVE_AGGRID = False
from wellmodel.correlations import LABELS as VLP_LABELS
from wellmodel import equipment as weq
from wellmodel import schematic as wsch
from wellmodel import options as wopt
from wellmodel import plotting as wplt

st.set_page_config(page_title="wellmodel", page_icon="🛢️", layout="wide")
ss = st.session_state


class _RecordIPR:
    """Module proxy: any call that returns an IPR curve is remembered (function and arguments) so the VLP
    tab can rebuild the IPR at another reservoir pressure, PI or skin."""

    def __init__(self, mod):
        self._mod = mod

    def __getattr__(self, name):
        obj = getattr(self._mod, name)
        if not callable(obj) or isinstance(obj, type):
            return obj

        def wrapped(*a, **k):
            out = obj(*a, **k)
            if isinstance(out, _ipr_mod.IPRCurve):
                ss["ipr_recipe"] = (obj, a, k)
            return out
        return wrapped


ipr, ipr2, wmpp = _RecordIPR(_ipr_mod), _RecordIPR(_ipr2_mod), _RecordIPR(_mpp_mod)
# Number inputs are typed in directly: hide the - / + step buttons everywhere.
st.markdown("""<style>
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {display: none !important;}
[data-testid="stNumberInputContainer"] input {padding-right: 0.75rem;}
section[data-testid="stSidebar"][aria-expanded="true"] {min-width: 380px;}
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


def num(col, label, key, default, mn=None, mx=None, step=None, fmt=None, help=None, disabled=False):
    cur = float(val(key, default))
    if mn is not None:
        cur = max(cur, float(mn))
    if mx is not None:
        cur = min(cur, float(mx))
    kw = {k: v for k, v in dict(min_value=None if mn is None else float(mn),
                                max_value=None if mx is None else float(mx),
                                step=None if step is None else float(step), format=fmt, help=help).items()
          if v is not None}
    v = col.number_input(label, value=cur, key=wkey(key), disabled=disabled, **kw)
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


def flag(widget, label, key, default=True, **kw):
    """Checkbox / toggle with a stored value (restorable from a project file)."""
    v = widget(label, value=bool(val(key, default)), key=wkey(key), **kw)
    ss["_v_" + key] = v
    return v


def dflt(key, **kw):
    """Widget default arguments, left out once the key holds a value (set by the user or a project file) -
    Streamlit warns when a widget gets both."""
    return {} if key in ss else kw


def track(store_key, edited):
    """Remember the current contents of a table editor for project files."""
    ss.setdefault("_tables", {})[store_key] = edited
    return edited


# ================================================================ project files
PROJECT_WIDGETS = ("eq_filter_on", "sys_sens_on", "sys_vars", "sys_rates", "vlp_cmp_corrs", "vlp_match_corrs",
                   "vlp_sfit")
PROJECT_WIDGET_PREFIXES = ("opt_", "sys_how_", "sys_vals_", "trav_rates_", "vfp_thp_", "vfp_wfr_", "vfp_gfr_")
PROJECT_EXTRAS = ("prp_text", "prp_name", "surf_matched", "vlp_selected", "vlp_u_accepted", "vlp_adj_accepted")


def project_state():
    """Everything needed to rebuild the model: input values, raw widget values, tables, accepted results."""
    inputs = {k[3:]: v for k, v in ss.items() if k.startswith("_v_") and isinstance(v, (int, float, str, bool, list))}
    widgets = {k: v for k, v in ss.items() if isinstance(k, str) and (k in PROJECT_WIDGETS or
                                                                    k.startswith(PROJECT_WIDGET_PREFIXES))
               and isinstance(v, (int, float, str, bool, list)) and not k.endswith("_ed")}
    tables = dict(ss.get("_tables", {}))
    extras = {k: ss[k] for k in PROJECT_EXTRAS if ss.get(k) is not None}
    ms = ss.get("vlp_matches", {}).get("ms", {})
    if ms:
        extras["vlp_matches"] = [{"correlation": m.correlation, "p1": m.p1, "p2": m.p2, "std": m.std,
                                  "rmse_before": m.rmse_before, "rmse_after": m.rmse_after,
                                  "residuals": np.asarray(m.residuals, float), "n": m.n, "flags": list(m.flags)}
                                 for m in ms.values()]
    extras["pvt_matched"] = ss.get("pvt_match") is not None
    return {"inputs": inputs, "widgets": widgets, "tables": tables, "extras": extras}


def _file_stem(name):
    """Project name -> safe file name stem."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name)).strip("_") or "project"


def _uploader_key():
    return f"proj_upload_{ss.get('_proj_up_ver', 0)}"


def queue_project(key):
    up = ss.get(key)
    if up is not None:
        ss["_proj_pending"] = (up.getvalue(), up.name)


def _clear_session():
    """Forget the model; keep widget version counters (every widget is rebuilt) and give the project uploader a
    fresh, empty box so the same file can be opened again."""
    keep = {k: v + 1 for k, v in ss.items() if isinstance(k, str) and k.startswith("_ver_")}
    up_ver = ss.get("_proj_up_ver", 0) + 1
    for k in list(ss.keys()):
        del ss[k]
    ss.update(keep)
    ss["_proj_up_ver"] = up_ver


def apply_project(data, fname):
    """Replace the model with a project file (called at the top of the script, before any widget exists)."""
    try:
        doc = wproj.loads(data)
    except ValueError as e:
        ss["_proj_msg"] = ("error", f"Could not open {fname}: {e}")
        ss["_proj_up_ver"] = ss.get("_proj_up_ver", 0) + 1
        return
    # start from a clean model; keep version counters (so every widget is rebuilt) and the uploader itself
    _clear_session()
    for k, v in doc["inputs"].items():
        ss["_v_" + k] = v
        ss["_ver_" + k] = ss.get("_ver_" + k, 0) + 1
    for k, v in doc["widgets"].items():
        ss[k] = v
    for k, df in doc["tables"].items():
        ss[k] = df
        ss["_ver_" + k] = ss.get("_ver_" + k, 0) + 1
    if "pvt_table" in doc["tables"]:
        ss["pvt_table_current"] = doc["tables"]["pvt_table"]
    ex = doc["extras"]
    for k in PROJECT_EXTRAS:
        if ex.get(k) is not None:
            ss[k] = ex[k]
    if ex.get("vlp_matches"):
        ss["vlp_matches"] = {"sig": None, "ms": {
            m["correlation"]: wvlp.VLPMatch(m["correlation"], m["p1"], m["p2"], m["std"], m["rmse_before"],
                                            m["rmse_after"], np.asarray(m["residuals"], float), m["n"], m["flags"])
            for m in ex["vlp_matches"]}}
    if ex.get("pvt_matched") and "pvt_table" in ss:
        ss["_proj_rematch"] = True                 # repeated just before the PVT tab draws
    stem = fname.rsplit(".", 1)[0].replace(".wellmodel", "").strip()
    saved = doc.get("name") or ""
    # the file name the user chose wins, unless it is just the saved name made file-safe ("Abura 2S" -> Abura_2S)
    ss["proj_name"] = saved if saved and _file_stem(saved) == stem else (stem or saved or "my_well")
    ss["_proj_msg"] = ("success", f"Opened {fname} (saved {doc.get('saved', '?')}).")


def new_project():
    _clear_session()
    ss["proj_name"] = "my_well"
    ss["_proj_msg"] = ("success", "New project - all inputs cleared.")


if "_proj_pending" in ss:
    apply_project(*ss.pop("_proj_pending"))

hdr = st.columns([5, 2.4, 0.42, 0.42, 0.42], vertical_alignment="bottom", gap="small")
hdr[0].title("🛢️ wellmodel")
pname = hdr[1].text_input("Project name", key="proj_name", **dflt("proj_name", value="my_well"),
                          help="Press Enter after typing - the save button then writes <name>.wellmodel.json.")
save_slot = hdr[2]                                          # filled at the end, once every tab has run
with hdr[3].popover("", icon=":material/folder_open:", help="Open project"):
    st.file_uploader("Open project", type=["json"], key=_uploader_key(), on_change=queue_project,
                     args=(_uploader_key(),),
                     help="A .wellmodel.json file saved from this tool. Replaces the current model.")
    st.caption("Replaces the current model; the PVT match is repeated automatically. Calculation results are "
               "not saved - press Calculate after opening.")
hdr[4].button("", icon=":material/note_add:", on_click=new_project, help="New project - clears every input",
              key="proj_new")
_msg_slot = st.empty()            # always present, so the tabs below keep their identity (and selection)
_msg = ss.pop("_proj_msg", None)
if _msg:
    getattr(_msg_slot, _msg[0])(_msg[1])
schematic_svg = None

TAB_NAMES = ["1 · Options summary", "2 · PVT Input Data", "3 · IPR Data", "4 · Equipment Data", "5 · VLP", "6 · Calculation", "7 · Design"]
tabs = st.tabs(TAB_NAMES, key="main_tabs")

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

    st.subheader("Calculation status")
    stat = opts.engine_status()
    st.dataframe(pd.DataFrame({"Option": [wopt.FIELDS[f][0] for f in wopt.FIELDS],
                               "Selection": [opts.label(f) for f in wopt.FIELDS],
                               "Status": [STATUS_ICON[stat[f]] for f in wopt.FIELDS]}),
                 hide_index=True, width="stretch")
    st.caption("Status covers the tabs built so far (PVT, IPR, Equipment and VLP). Black oil and "
               "equation-of-state PVT both feed the IPR and VLP calculations; tubing, annular and tubing + annular "
               "flow are handled in the VLP. Completion and sand control are modelled in the IPR Data tab (perforation, damage, "
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
                       correct_impurities=bool(val("pvt_impurity_corr", False)))
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
def eos_mpp_table(text: str, name: str, t_f: float, pr: float, rel: tuple, gas: bool, wgr: float):
    """Multiphase pseudo-pressure integrand for the PRP fluid (cached: one flash per pressure)."""
    f = wm_eos.EOSFluid(wm_eos.parse_prp(text, name))
    res = eos_calculate(text, name, t_f)
    scf = float(res["sep"]["stages"]["Gas (scf per lbmol feed)"].sum())
    unit = scf / 1000 if gas else float(res["sep"]["stb_per_lbmol"])
    if unit <= 0:
        raise ValueError("The PRP fluid gives no " + ("separator gas." if gas else "stock-tank oil."))
    w = wgr / 1e6 * scf if gas else 0.0
    return wmpp.eos_table(f, t_f, wmpp.pressure_grid(pr, res["psat"], 40), wmpp.Corey(*rel), unit, w), res["psat"]


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


if ss.pop("_proj_rematch", False):
    try:
        run_match()
    except (ValueError, KeyError, TypeError) as e:
        ss["_proj_msg"] = ("warning", f"PVT match could not be repeated: {e}")

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
        c = st.columns(4)
        num(c[0], "Reservoir temperature (deg F)", "pvt_t", 200.0, 40.0, 500.0, 1.0)
        num(c[1], "Condensate Gravity (API)", "pvt_cond_api", 50.0, 20.0, 90.0, 0.5,
            help="Stock-tank condensate gravity. Used to carry condensate as wellstream gas in the tubing (VLP).")
        num(c[2], "Water Salinity (ppm)", "pvt_sal", INPUT_DEFAULTS["pvt_sal"], 0.0, 300000.0, 1000.0, "%.0f")
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
        flag(c[3].checkbox, "Correct bubble point for N2, CO2, H2S", "pvt_impurity_corr", False,
                      help="Glaso (1980) non-hydrocarbon correction factors on every bubble-point correlation. "
                           "Off by default - PROSPER does not apply them. Impurities always correct the gas "
                           "Z-factor (Wichert-Aziz).")

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
        ss["pvt_table_current"] = track("pvt_table", edited)

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
            use_m = flag(c[2].checkbox, "Apply match parameters", "pvt_use_match")
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
    return track(key, st.data_editor(ss[key], key=key + "_ed", num_rows="dynamic", width="stretch",
                                     column_config={"Rate": st.column_config.NumberColumn(rate_label, format="%g"),
                                                    "Pwf": st.column_config.NumberColumn("FBHP (psia)", format="%g")}))


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
VERTICAL_SKIN_MODELS = {"Darcy", "Fetkovich", "Jones", "Transient", "Back pressure", "Dual Porosity",
                        "Multiphase Pseudo Pressure"}
HORIZONTAL_MODELS = {"Horizontal Well - No Flow Boundaries", "Horizontal Well - Transverse Vertical Fractures",
                     "Horizontal Well - friction dP"}
OIL_ADV = ["Hydraulically Fractured Well", "Horizontal Well - No Flow Boundaries",
           "Horizontal Well - Transverse Vertical Fractures", "Horizontal Well - friction dP",
           "Dual Porosity", "Multi-layer", "Multi-layer with dP loss in wellbore", "Multiphase Pseudo Pressure"]
GAS_ADV = OIL_ADV + ["Modified Isochronal"]
STILL_NOT = {
    "oil": ["Kuchuk & Goode horizontal and constant-pressure upper boundary (equations not public here - "
            "Babu & Odeh and Joshi are provided)", "SkinAide (ELF proprietary method)",
            "Thermally Induced Fracture (water injectors - injection IPRs come with injector support)"],
    "gas": ["Kuchuk & Goode horizontal (Babu & Odeh and Joshi are provided)", "SkinAide (ELF proprietary method)"],
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


def blackoil_model(o):
    """Matched (or default Standing / Beal) black-oil model from the PVT tab, or None."""
    model = ss.get("pvt_model")
    if model is None:
        inp, _ = current_inputs()
        if inp.check():
            return None
        model = bo.BlackOilModel(inp, "standing", "beal", {})
    return model


def mpp_model(o, gas, pr, t_f, r, skin, pvt, gasp, wc, wgr):
    st.markdown("**Relative permeability (Corey, gas-oil at connate water)**")
    c = st.columns(4)
    swc = num(c[0], "Connate water saturation Swc", "mpp_swc", 0.2, 0.0, 0.9, 0.01)
    sorg = num(c[1], "Residual oil saturation to gas Sorg", "mpp_sorg", 0.2, 0.0, 0.9, 0.01)
    sgc = num(c[2], "Critical gas saturation Sgc", "mpp_sgc", 0.05, 0.0, 0.9, 0.01)
    kro_e = num(c[3], "kro end point", "mpp_kro", 1.0, 0.01, 1.0, 0.05)
    c = st.columns(4)
    krg_e = num(c[0], "krg end point", "mpp_krg", 1.0, 0.01, 1.0, 0.05)
    no = num(c[1], "Corey exponent oil", "mpp_no", 2.0, 0.5, 10.0, 0.1)
    ng = num(c[2], "Corey exponent gas", "mpp_ng", 2.0, 0.5, 10.0, 0.1)
    rel_t = (swc, sorg, sgc, kro_e, krg_e, no, ng)
    rel = wmpp.Corey(*rel_t)
    eos = o.method == "eos" and ss.get("prp_text")
    notes = ["Steady-state (Fevang & Whitson) pseudo-pressure: the flowing mixture has the producing "
             "composition at every pressure; relative permeabilities split the mobility between phases."]
    d_nd, psat, params = 0.0, None, {}
    if gas:
        dm = sel(c[3], "Non-Darcy flow factor", ["None", "Calculated", "Entered"], "mpp_d_method")
        if dm == "Calculated":
            d_nd = ipr.non_darcy_d(r["k"], r["h"], r["hp"], r["rw"], gasp["mu_r"], gasp["sg"])
        elif dm == "Entered":
            d_nd = num(st.columns(4)[0], "D (1/(Mscf/d))", "mpp_d", 1e-5, 0.0, 1.0, None, "%.3e")
    with st.spinner("Calculating the pseudo-pressure integral..."):
        if eos:
            table, psat = eos_mpp_table(ss["prp_text"], ss.get("prp_name", ""), float(t_f), float(pr), rel_t,
                                        gas, float(wgr or 0.0))
            notes.append("Phase split, densities and viscosities from flashing the PRP fluid at each pressure.")
        elif gas:
            table = wmpp.gas_correlation_table(gasp["mu_z"], wmpp.pressure_grid(pr), t_f, rel)
            notes.append("Gas correlations cannot drop out liquid, so this is single-phase gas pseudo-pressure. "
                         "Choose Equation of state in the Options summary to model the condensate bank.")
        else:
            model = blackoil_model(o)
            psat = model.bubble_point(t_f)
            rp = num(c[3], "Producing GOR (scf/STB)", "mpp_rp", float(model.inp.rsb), 0.0, 1e6, 10.0,
                     help="Total producing GOR. Above the solution GOR, free gas also flows above the bubble point.")
            params["Producing GOR (scf/STB)"] = rp
            table = wmpp.blackoil_table(lambda p: model.table(t_f, p), wmpp.pressure_grid(pr, psat), rp,
                                        wc / max(1 - wc, 1e-9), rel)
    return wmpp.multiphase_pseudo_pressure("gas" if gas else "oil", pr, r["k"], r["h"], r["area"], r["ca"],
                                           r["rw"], skin, table, d_nd=d_nd, psat=psat,
                                           wor=0.0 if gas else wc / max(1 - wc, 1e-9), params=params, notes=notes)


def mpp_plot(curve):
    t = curve.mpp_table
    p = t["Pressure (psia)"]
    gas = curve.fluid == "gas"
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.plot(p, t["So"], color=wplt.SERIES[1], label="Condensate saturation" if gas else "Oil saturation")
    a.plot(p, t["kro"], color=wplt.SERIES[2], ls="--", label="kro")
    a.plot(p, t["krg"], color=wplt.SERIES[0], ls="--", label="krg")
    a.set_ylim(0, 1.02); a.set_xlabel("Pressure (psia)"); a.set_ylabel("Saturation / relative permeability")
    a.set_title("Flowing saturation and relative permeability", fontsize=10)
    a.legend(fontsize=8)
    b.plot(p, t["Integrand"], color=wplt.SERIES[0], label="two-phase")
    b.plot(p, t["Integrand single-phase"], color=wplt.INK2, ls="--", label="single-phase")
    b.set_xlabel("Pressure (psia)"); b.set_ylabel("dm/dp")
    b.set_title("Pseudo-pressure integrand (area = deliverability)", fontsize=10)
    b.legend(fontsize=8)
    psat = curve.params.get("Saturation pressure (psia)")
    for ax in (a, b):
        if psat and psat < p.max():
            ax.axvline(psat, color=wplt.INK2, lw=0.8, ls=":")
    fig.tight_layout()
    return fig


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


DEV_MODELS = ["None", "Cinco / Martin-Bronz", "Cinco (2) / Martin-Bronz",
              "Cinco-Ley, Ramey & Miller (1975) - full solution", "Cinco-Ley deviation + Papatzacos"]


def skin_panel(o, k, rw, h, gas_or_oil):
    """Mechanical/geometrical + deviation/partial-penetration skin (PROSPER 2.7.7). Returns (total, breakdown)."""
    st.markdown("**Mechanical / geometrical skin**")
    cased = o.completion == "cased_hole"
    mech_opts = (["Enter skin by hand", "Karakas & Tariq", "MacLeod"] if cased
                 else ["Enter skin by hand", "Hawkins damaged zone"])
    c = st.columns(3)
    mech = sel(c[0], "Method", mech_opts, "ipr_mech",
               help="Choices follow the completion in the Options summary. Karakas & Tariq (1991): perforation "
                    "(horizontal, vertical, wellbore), damaged and crushed zones. MacLeod (1983): damaged and crushed "
                    "zones. Locke's nomograph method is not available.")
    parts = {}
    if mech == "Enter skin by hand":
        parts["Mechanical (entered)"] = num(c[1], "Skin", "ipr_skin", 0.0, -6.0, 500.0, 0.5)
    else:
        kd = ld = kc = lc = 0.0
        perf = mech in ("Karakas & Tariq", "MacLeod")
        if perf:
            c = st.columns(4)
            dp_ = num(c[0], "Perforation diameter (in)", "ipr_pd", 0.4, 0.05, 2.0, 0.05, help="Entry hole diameter.")
            spf = num(c[1], "Shots per foot", "ipr_spf", 4.0, 0.5, 40.0, 1.0)
            lp = num(c[2], "Perforation length (in)", "ipr_lp", 10.0, 0.5, 60.0, 0.5,
                     help="Effective perforation length in the formation.")
            eff = num(c[3], "Perforation efficiency", "ipr_peff", 1.0, 0.01, 1.0, 0.05, "%.2f",
                      help="Fraction of the perforations that are open (active).")
            c = st.columns(4)
            if mech == "Karakas & Tariq":
                ph = sel(c[0], "Shot phasing (deg)", [0, 45, 60, 90, 120, 180], "ipr_phasing", 90,
                         help="0 = all shots in one direction (360°)")
                kv = num(c[1], "Vertical permeability (md)", "ipr_kv", float(k), 1e-4, 1e6)
            else:
                h_perf = num(c[0], "Perforation interval (ft)", "ipr_hperf", float(h), 0.1, 1e5)
            c = st.columns(4)
        else:
            c = st.columns(4)
        ld = num(c[0], "Damaged zone thickness (in)", "ipr_ld", 6.0 if mech == "Hawkins damaged zone" else 0.0,
                 0.0, 120.0, 0.5)
        kd = num(c[1], "Damaged zone permeability (md)", "ipr_kd", max(k * 0.2, 0.001), 0.0001, 1e5, 1.0)
        if perf:
            lc = num(c[2], "Crushed zone thickness (in)", "ipr_lc", 0.5, 0.0, 5.0, 0.1)
            kc = num(c[3], "Crushed zone permeability (md)", "ipr_kc", max(k * 0.2, 0.001), 0.0001, 1e5, 1.0)
        if mech == "Karakas & Tariq":
            r = wcomp.karakas_tariq(k, rw, lp / 12, dp_ / 24, spf * eff, ph, kv / k, ld / 12, kd, lc / 12, kc)
            parts.update({"Perforation (K&T)": r["perforation"] if r["damage"] == 0 else k / kd * r["perforation"],
                          "Damaged zone": r["damage"], "Crushed zone": r["crushed zone"]})
        elif mech == "MacLeod":
            r = wcomp.macleod(k, rw, lp / 12, dp_ / 24, spf, ld / 12, kd, lc / 12, kc, eff, h, min(h_perf, h))
            parts.update({"Damaged zone": r["damage"], "Crushed zone": r["crushed zone"]})
        else:
            parts["Damaged zone (Hawkins)"] = wcomp.hawkins(k, rw, ld / 12, kd)
    st.markdown("**Deviation / partial penetration skin**")
    dev = sel(st.columns(3)[0], "Method ", DEV_MODELS, "ipr_devpp",
              help="Cinco / Martin-Bronz: Cinco-Ley deviation correlation, isotropic, to 65°, plus Brons & Marting "
                   "partial penetration. Cinco (2) / Martin-Bronz: the deviation correlation with anisotropy, to 75°. "
                   "Full solution: Cinco-Ley, Ramey & Miller (1975, SPE 5589), infinite-conductivity slanted, "
                   "partially-penetrating well - slant, penetration and the position of the interval together. "
                   "Wong-Clifford is not available.")
    if dev != "None":
        c = st.columns(4)
        prosper = dev.endswith("Martin-Bronz")
        lim = 65.0 if dev.startswith("Cinco /") else 75.0
        th = num(c[0], "Deviation angle (deg)", "ipr_dev", 0.0, 0.0, 89.0 if not prosper else lim, 1.0,
                 help="Angle from the vertical.")
        if prosper:
            frac = num(c[1], "Partial penetration fraction", "ipr_ppf", 1.0, 0.01, 1.0, 0.05, "%.3f",
                       help="Perforated interval / net pay, both in TVD.")
            hp, zt = frac * h, 0.0
        else:
            hp = num(c[1], "Perforated interval, TVD (ft)", "ipr_hpp", h, 0.1, 1e4, 1.0)
            zt = num(c[2], "Top of reservoir to top of perforations, TVD (ft)", "ipr_ztop", 0.0, 0.0, 1e4, 1.0)
        kv2 = num(c[3], "Formation vertical permeability (md)", "ipr_kv2", float(k), 1e-4, 1e6,
                  help="Cinco / Martin-Bronz uses it for the partial penetration skin only (the deviation "
                       "correlation is isotropic); the other models use it for both.")
        kvkh2 = kv2 / k
        if hp + zt > h + 1e-6:
            st.warning("The perforated interval extends below the reservoir.")
        try:
            if prosper:
                parts["Deviation (Cinco" + (")" if lim == 65.0 else " 2)")] = wcomp.deviation_skin(
                    th, h, rw, kvkh2, anisotropic=lim == 75.0)
                parts["Partial penetration (Brons-Marting)"] = wcomp.brons_marting(h, min(hp, h), rw, kvkh2)
            elif dev.startswith("Cinco-Ley, Ramey"):
                r = wcomp.cinco_slant_pp(th, h, min(hp, h), zt, rw, kvkh2)
                parts["Deviation (Cinco-Ley et al.)"] = r["slant"]
                parts["Partial penetration (Cinco-Ley et al.)"] = r["penetration"]
            else:
                parts["Deviation (Cinco-Ley)"] = wcomp.deviation_skin(th, h, rw, kvkh2)
                parts["Partial penetration (Papatzacos)"] = wcomp.partial_penetration_skin(h, min(hp, h), zt, rw,
                                                                                          kvkh2)
        except ValueError as e:
            st.error(str(e))
    total = float(sum(parts.values()))
    st.caption("Skin: " + " · ".join(f"{k_} {v:+.3f}" for k_, v in parts.items()) + f"  →  **total {total:+.3f}**")
    return total, parts


def skin_dp_plot(q, dps, qunit):
    """PROSPER-style skin pressure drop plot: pressure drop of the total skin and of each component against rate."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cols = list(wplt.SERIES) + ["#9467bd", "#8c564b", "#e377c2", "#17becf"]
    i = 0
    for name, dp in dps.items():
        if name != "Total skin" and not np.any(np.abs(dp) > 1e-9):
            continue                                   # components with zero skin are not drawn
        ax.plot(q, dp, lw=2.2 if name == "Total skin" else 1.4,
                color=wplt.INK if name == "Total skin" else cols[i % len(cols)], label=name)
        i += name != "Total skin"
    ax.axhline(0, color=wplt.INK2, lw=0.6)
    ax.set_xlabel(f"Rate ({qunit})"); ax.set_ylabel("Pressure drop (psi)")
    ax.set_xlim(0, float(np.max(q))); ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=8)
    ax.set_title("Skin pressure drop", fontsize=10)
    fig.tight_layout()
    return fig


def _opt_num(col, label, key, help=None):
    """Optional number (blank = not entered) for PROSPER's 'leave blank to calculate' fields."""
    txt = col.text_input(label, key=wkey(key), help=help, **dflt(key, value=""))
    ss["_v_" + key] = txt
    txt = (txt or "").strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        col.caption(":red[not a number - ignored]")
        return None


def sand_control_panel(o, fluid, h_default, res_known=None):
    """Sand control inputs (PROSPER 2.7.8.2). Returns (SandControl, (k, h, rw), multiphase flag)."""
    kind = o.sand_control
    if kind == "none":
        return wcomp.SandControl("none"), None, True
    st.subheader("Sand control")
    if kind == "gravel_pack":
        kind = "gravel_pack_cased" if o.completion == "cased_hole" else "gravel_pack_open"
    sc = wcomp.SandControl(kind)
    st.caption(f"{sc.label()} - from the Options summary. The IPR is calculated to the sand face; the completion "
               "adds dP = 141.2 Q µ (S + D Q) / (k H) with the in-situ rate and the fluid properties recalculated at "
               "the sand-face pressure (PROSPER).")
    if res_known:
        k_, h_, rw_ = res_known["k"], res_known["h"], res_known["rw"]
    else:
        c = st.columns(4)
        k_ = num(c[0], "Reservoir permeability (md)", "sc_k", float(val("ipr_k", 50.0)), 0.001, 1e5)
        h_ = num(c[1], "Reservoir thickness (ft)", "sc_h", float(val("ipr_h", 60.0)), 0.1, 1e4)
        rw_ = num(c[2], "Wellbore radius (ft)", "sc_rw", float(val("ipr_rw", 0.354)), 0.05, 5.0, None, "%.3f")
        st.caption("This IPR model has no permeability/thickness: they are needed here to express the sand-control "
                   "drop as a skin.")
    p = {}
    multiphase = True
    if kind in ("gravel_pack_cased", "gravel_pack_open"):
        left, right = st.columns([2.2, 1])
        with left:
            c = st.columns(3)
            p["gravel_perm"] = num(c[0], "Gravel pack permeability (md)", "sc_kg", 60000.0, 100.0, 1e7, 1000.0,
                                   help="In-situ permeability of the gravel (about half the lab value).")
            p["pack_length_in"] = num(c[1], "Gravel pack length (in)", "sc_L", 4.0 if kind == "gravel_pack_cased" else 1.5,
                                      0.05, 40.0, 0.25, help="Distance from the screen OD to the sand face.")
            p["interval"] = num(c[2], "Perforation interval, TVD (ft)", "sc_int", h_default, 0.1, 1e4, 1.0,
                                help="Used for the flow area / velocities only.")
            c = st.columns(3)
            if kind == "gravel_pack_cased":
                p["perf_diameter_in"] = num(c[0], "Perforation diameter (in)", "sc_pd", 0.75, 0.1, 3.0, 0.05,
                                            help="Diameter of the perforation tunnel.")
                p["spf"] = num(c[1], "Shot density (1/ft)", "sc_spf", 12.0, 0.5, 40.0, 1.0)
                p["perf_efficiency"] = num(c[2], "Perforation efficiency", "sc_eff", 0.8, 0.01, 1.0, 0.05, "%.2f",
                                           help="Proportion of perforations that are open and effective.")
                c = st.columns(3)
                p["deviation"] = num(c[0], "Deviation (deg)", "sc_dev", 0.0, 0.0, 85.0, 1.0,
                                     help="Converts the TVD interval to MD for the flow area.")
            bm = sel(c[1], "Beta (turbulence) method", ["Calculated", "Entered"], "sc_bm",
                     help="Calculated: beta = 1.47e7 / Kg^0.55 (1/ft).")
            if bm == "Entered":
                p["beta"] = num(c[2], "Beta (1/ft)", "sc_beta", float(round(1.47e7 / p["gravel_perm"] ** 0.55)), 0.0, 1e12)
            if fluid == "oil":
                multiphase = sel(st.columns(3)[0], "Method (oil only)", ["Multi Phase", "Single Phase"], "sc_phase",
                                 help="Single Phase: liquid properties only (the classic method). Multi Phase: "
                                      "phase-weighted properties including free gas released at the sand face.") \
                    == "Multi Phase"
        with right:
            st.markdown("Typical gravel permeabilities (lab, md)")
            st.dataframe(pd.DataFrame(wcomp.GRAVEL_PERMEABILITY, columns=["Gravel type", "Mesh", "Lab perm. (md)"]),
                         hide_index=True, width="stretch", height=320)
    else:
        c = st.columns(4)
        p["interval"] = num(c[0], "Production interval (ft)", "sc_int", h_default, 0.1, 1e4, 1.0)
        if kind == "pre_packed_screen":
            p["screen_inner_radius"] = num(c[1], "Screen inner radius (ft)", "sc_rsi", 0.2, 0.01, 5.0, None, "%.4f")
            p["screen_outer_radius"] = num(c[2], "Screen outer radius (ft)", "sc_rso", 0.25, 0.01, 5.0, None, "%.4f")
            p["screen_perm"] = num(c[3], "Screen permeability (md)", "sc_kpp", 500.0, 0.01, 1e7)
            c = st.columns(4)
            p["screen_beta"] = _opt_num(c[0], "Screen beta (1/ft)", "sc_bpp",
                                        help="Blank: calculated (1.47e7 / k^0.55); 0: ignored.")
        elif kind == "wire_wrapped_screen":
            p["screen_outer_radius"] = num(c[1], "Screen outer radius (ft)", "sc_rso_w", 0.25, 0.01, 5.0, None, "%.4f")
            c = st.columns(4)
        else:
            p["liner_inner_radius"] = num(c[1], "Liner inner radius (ft)", "sc_ri", 0.25, 0.01, 5.0, None, "%.4f")
            p["liner_outer_radius"] = num(c[2], "Liner outer radius (ft)", "sc_ro2", 0.29, 0.01, 5.0, None, "%.4f")
            p["slot_density"] = num(c[3], "Slot density (1/ft)", "sc_sden", 100.0, 0.01, 1000.0)
            c = st.columns(4)
            p["slot_height_in"] = num(c[0], "Slot height (in)", "sc_sh", 2.5, 0.01, 48.0)
            p["slot_width_in"] = num(c[1], "Slot width (in)", "sc_sw2", 0.02, 0.001, 1.0, None, "%.3f")
            p["screen_outer_radius"] = num(c[2], "Screen outer radius (ft, 0 = none)", "sc_rss", 0.0, 0.0, 5.0, None,
                                           "%.4f", help="Only for a screen inside the liner; must be below the liner "
                                                        "inner radius.")
            c = st.columns(4)
        p["outside_perm"] = _opt_num(c[1], "Outside permeability (md)", "sc_kout",
                                     help="Material between the screen/liner and the sand face. Blank: formation sand "
                                          "(the reservoir permeability).")
        p["outside_beta"] = _opt_num(c[2], "Outside beta (1/ft)", "sc_bout",
                                     help="Blank: calculated (1.47e7 / k^0.55); 0: ignored.")
        if fluid == "oil":
            multiphase = sel(c[3], "Method (oil only)", ["Multi Phase", "Single Phase"], "sc_phase") == "Multi Phase"
    sc.params = p
    return sc, (k_, h_, rw_), multiphase


def sc_fluid_props(o, fluid, pvt, gasp, t_f, wc, wgr, cgr, multiphase):
    """In-situ property function for the sand-control dP (properties at the sand-face pressure)."""
    if fluid == "oil":
        if o.method != "eos":
            m = blackoil_model(o)
            if m is not None:
                return wcomp.props_from_fluid(wvlp.BlackOilFluid(m), "oil", t_f,
                                              {"wc": wc, "gor": float(m.inp.rsb)}, multiphase), \
                    "black oil at the sand-face pressure"
        return wcomp.props_constant(pvt["bo"], pvt["rho"], pvt["mu"], "oil"), "oil properties at reservoir pressure"
    if o.method != "eos":
        inp = bo.PVTInputs(rsb=1, api=40, gas_sg=float(gasp["sg"]), salinity_ppm=float(val("pvt_sal", 50000.0)),
                           h2s=float(val("pvt_h2s", 0.0)), co2=float(val("pvt_co2", 0.0)), n2=float(val("pvt_n2", 0.0)))
        return wcomp.props_from_fluid(wvlp.WetGasFluid(inp, float(val("pvt_cond_api", 50.0))), "gas", t_f,
                                      {"wgr": wgr or 0.0, "cgr": cgr or 0.0}), "wet gas at the sand-face pressure"
    tr = t_f + 459.67
    pr_ = float(val("ipr_pr", 4000.0))
    bg = 0.02827 * gasp["z_r"] * tr / pr_
    return wcomp.props_constant(bg, 28.9625 * gasp["sg"] * pr_ / (gasp["z_r"] * 10.7316 * tr), gasp["mu_r"], "gas"), \
        "gas properties at reservoir pressure"


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
    return track(key, st.data_editor(ss[key], key=key + "_ed", num_rows="dynamic", width="stretch", column_config={
        "Type": st.column_config.SelectboxColumn(options=["Perforated", "Blank"], required=True),
        "IPR model": st.column_config.SelectboxColumn(options=models)}))


def zones_table():
    key = "ipr_zones"
    if key not in ss:
        ss[key] = pd.DataFrame({"Type": ["Perforated"] * 4 + ["Blank", "Perforated"],
                                "Length (ft)": [300.0, 300, 300, 300, 200, 600],
                                "Permeability (md)": [50.0, 50, 40, 40, np.nan, 60],
                                "Skin": [0.0, 0, 1, 1, np.nan, 0],
                                "Flowing ID (in)": [4.0] * 6, "Roughness (in)": [0.0006] * 6},
                               index=pd.RangeIndex(1, 7, name="Zone (heel → toe)"))
    return track(key, st.data_editor(ss[key], key=key + "_ed", num_rows="dynamic", width="stretch", column_config={
        "Type": st.column_config.SelectboxColumn(options=["Perforated", "Open Hole", "Blank"], required=True)}))


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
        if flag(c[1].toggle, "Temperature from PVT", "ipr_t_from_pvt"):
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
                if flag(c[3].toggle, "CGR from PVT", "ipr_cgr_from_pvt"):
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
            ss["ipr_recipe"], ss["ipr_sc_dp"] = None, None
            oil_kw = {} if gas else {"pb": pvt["pb"], "bo": pvt["bo"]}
            gas_kw = {"t_f": t_f, "z": gasp.get("z_r"), "sg": gasp.get("sg")} if gas else {}
            mu_used = gasp.get("mu_r") if gas else (pvt["mu"] if pvt else None)
            try:
                if model in VERTICAL_SKIN_MODELS:
                    r = reservoir_props(nd=(model == "Jones" or (gas and model == "Multiphase Pseudo Pressure")))
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
                elif model == "Multiphase Pseudo Pressure":
                    curve = mpp_model(o, gas, pr, t_f, r, skin, pvt, gasp, wc, wgr)
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
            sc, sc_res, sc_multi = sand_control_panel(o, fluid, float(val("ipr_h", 60.0)),
                                                      r if model in VERTICAL_SKIN_MODELS else None)
            if curve is not None and err is None and sc.kind != "none":
                try:
                    props_, props_note = sc_fluid_props(o, fluid, pvt, gasp, t_f, wc, wgr, cgr, sc_multi)
                    dp_fn = wcomp.sand_control_dp(sc, props_, *sc_res)
                    st.caption(f"Sand-control dP with {props_note}.")
                    curve = wcomp.apply_sand_control(curve, dp_fn)
                    ss["ipr_sc_dp"] = dp_fn
                    for attr in ("layer_rates", "profile", "profile_heel_pressure", "isochronal", "c_iso"):
                        if attr == "layer_rates" and hasattr(raw_curve, attr) and \
                                len(raw_curve.layer_rates) != len(curve.q):
                            continue                  # per-point layer rates belong to the sand-face curve
                        if hasattr(raw_curve, attr):
                            setattr(curve, attr, getattr(raw_curve, attr))
                except (ValueError, KeyError, ZeroDivisionError) as e:
                    err = f"Sand control: {e}"

            # ----------------------------------------------------------- results
            st.subheader("Results")
            if err or curve is None:
                ss.pop("ipr_curve", None)
            if err:
                st.warning(err)
            elif curve is None:
                st.info("Enter the model data above.")
            else:
                ss["ipr_curve"] = curve
                ss["ipr_meta"] = {"model": model, "fluid": fluid, "skin_parts": dict(skin_parts),
                                  "pb": None if gas else pvt["pb"]}
                for n_ in curve.notes:
                    st.info(n_)
                items = [(k_, v) for k_, v in curve.params.items() if isinstance(v, (int, float, np.floating))]
                m = st.columns(min(len(items) + 1, 4))
                m[0].metric("AOF", f"{curve.aof / 1e3:,.3f} MMscf/d" if gas else f"{curve.aof:,.0f} STB/d")
                for col, (k_, v) in zip(m[1:], items[:3]):
                    col.metric(k_, f"{v:.4g}" if abs(v) < 1e4 else f"{v:,.0f}")
                if len(items) > 3:
                    st.caption(" · ".join(f"{k_}: {v:.4g}" for k_, v in items[3:]))
                st.pyplot(ipr_plot(curve, None if gas else pvt["pb"],
                                   raw_curve if sc.kind != "none" else None), clear_figure=True)
                if hasattr(curve, "mpp_table"):
                    st.pyplot(mpp_plot(curve), clear_figure=True)
                    with st.expander("Pseudo-pressure table"):
                        st.dataframe(curve.mpp_table.style.format(precision=5), hide_index=True, width="stretch")
                if hasattr(curve, "profile"):
                    st.pyplot(profile_plot(curve), clear_figure=True)
                table = curve.table(cgr=cgr, wgr=wgr) if gas else curve.table(wc=wc, gor=pvt.get("gor"))
                skin_dps = {}
                if kh_for_skin and skin_parts:
                    s_tot = sum(skin_parts.values())
                    if gas:
                        coef = 1422 * (t_f + 459.67) * gasp["mu_r"] * gasp["z_r"] * s_tot / kh_for_skin
                        dp_t = np.sqrt(np.maximum(table["Pwf (psia)"].to_numpy() ** 2 + coef * curve.q, 0.0)) \
                            - table["Pwf (psia)"].to_numpy()
                        per_s = dp_t / s_tot if abs(s_tot) > 1e-12 else 0 * dp_t
                    else:
                        per_s = 141.2 * pvt["mu"] * pvt["bo"] / kh_for_skin * np.asarray(curve.q, float)
                        dp_t = per_s * s_tot
                    table["dP skin (psi)"] = dp_t
                    skin_dps["Total skin"] = dp_t
                    if len(skin_parts) > 1:
                        for name_, s_ in skin_parts.items():
                            table[f"dP {name_} (psi)"] = per_s * s_
                            skin_dps[name_] = per_s * s_
                if hasattr(curve, "dp_sand_control"):
                    table["Sand-face pressure (psia)"] = curve.p_sandface
                    table["dP sand control (psi)"] = curve.dp_sand_control
                    if getattr(curve, "sc_skin", None) is not None:
                        table["Sand control skin"] = curve.sc_skin
                        table["Sand control D (1/(1000 RB/d))"] = curve.sc_d * 1000
                    if getattr(curve, "sc_velocity", None) is not None:
                        table["Velocity at casing (ft/s)"] = curve.sc_velocity
                lr = getattr(curve, "layer_rates", None)
                if lr is not None:
                    for col in lr.columns:
                        table[f"{col} rate"] = lr[col].to_numpy() / (1e3 if gas else 1)
                if hasattr(curve, "dp_sand_control") and skin_dps:
                    skin_dps["Sand control"] = np.asarray(curve.dp_sand_control, float)
                if skin_dps:
                    with st.expander("Skin pressure drop", expanded=False):
                        st.pyplot(skin_dp_plot(np.asarray(curve.q, float) / (1e3 if gas else 1), skin_dps,
                                               "MMscf/d" if gas else "STB/d"), clear_figure=True)
                        st.caption("Pressure drop across each skin component at each rate (oil: 141.2 q µ B s / kh; "
                                   "gas: the total skin drop shared in proportion to each component's skin).")
                with st.expander("IPR table", expanded=False):
                    st.dataframe(table.style.format(precision=2), hide_index=True, width="stretch")
                if hasattr(curve, "isochronal"):
                    with st.expander("Modified isochronal corrections", expanded=True):
                        st.dataframe(curve.isochronal.style.format(precision=4), hide_index=True, width="stretch")
                st.download_button("Download IPR (CSV)", table.to_csv(index=False).encode(), "ipr.csv")



# ========================================================= tab 4: equipment
EQ_ROWS = {"survey": weq.MAX_SURVEY_POINTS, "surface": 10, "downhole": 18, "geo": 20}


def blank_grid(cols, n, text_cols=()):
    df = pd.DataFrame({c: pd.Series([None] * n, dtype=object) if c in text_cols
                       else pd.Series([np.nan] * n, dtype=float) for c in cols})
    df.index = pd.RangeIndex(1, n + 1, name="Point")
    return df


def fill_grid(rows: pd.DataFrame, cols, n, text_cols=()):
    df = blank_grid(cols, n, text_cols)
    for i, (_, r) in enumerate(rows.iterrows()):
        if i >= n:
            break
        for c in cols:
            if c in r and pd.notna(r[c]):
                df.iat[i, df.columns.get_loc(c)] = r[c]
    return df


def set_grid(name, df):
    ss["eq_" + name] = df
    ss["_ver_eq_" + name] = ss.get("_ver_eq_" + name, 0) + 1


LOCKED_CELL = "#b9dcef"          # PROSPER's light blue for cells that take no input


def type_grid(name, types, active, numeric_cols, select_cols=None, height=None):
    """Equipment table where each row's Type decides which cells accept input, as in PROSPER: locked cells are
    shaded and cannot be edited. active: {type: [open columns]}; select_cols: {column: options}.
    Returns the edited table (index 'Point'). Uses AG Grid when available, otherwise st.data_editor."""
    import json as _json
    select_cols = select_cols or {}
    base = ss["eq_" + name]
    ver = ss.get("_ver_eq_" + name, 0)
    if not HAVE_AGGRID:
        cfg = {"Type": st.column_config.SelectboxColumn(options=types),
               **{c_: st.column_config.SelectboxColumn(options=o_, width="medium") for c_, o_ in select_cols.items()},
               **{c_: st.column_config.NumberColumn(c_, format="%g") for c_ in numeric_cols}}
        return track("eq_" + name, st.data_editor(base, key=f"eq_{name}_ed_{ver}", num_rows="fixed",
                                                  width="stretch", column_config=cfg))
    df = base.reset_index()
    idx = df.columns[0]
    for c_ in numeric_cols:
        df[c_] = pd.to_numeric(df[c_], errors="coerce").astype(float)
    act = _json.dumps({t_: sorted(v_) for t_, v_ in active.items()})
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, sortable=False, filter=False, suppressMovable=True,
                                wrapHeaderText=True, autoHeaderHeight=True, flex=1, minWidth=88)
    gb.configure_column(idx, editable=False, width=62, flex=0, pinned="left",
                        cellStyle={"color": "#6b7280", "textAlign": "center"})
    gb.configure_column("Label", editable=True, minWidth=110, flex=1.3)
    gb.configure_column("Type", editable=True, minWidth=120, flex=1.3, cellEditor="agSelectCellEditor",
                        cellEditorParams={"values": [""] + list(types)})

    def is_open(col):
        return f"const a={act}; const t=p.data['Type']; return !!(t && a[t] && a[t].includes({_json.dumps(col)}));"
    for c_ in [c_ for c_ in df.columns if c_ not in (idx, "Label", "Type")]:
        kw = dict(editable=JsCode(f"function(p){{{is_open(c_)}}}"),
                  cellStyle=JsCode(f"function(p){{const ok=(function(){{{is_open(c_)}}})(); "
                                   f"return ok ? {{'backgroundColor': ''}} : {{'backgroundColor': '{LOCKED_CELL}'}};}}"))
        if c_ in select_cols:
            gb.configure_column(c_, minWidth=150, flex=1.8, cellEditor="agSelectCellEditor",
                                cellEditorParams={"values": [""] + list(select_cols[c_])}, **kw)
        else:
            gb.configure_column(c_, type=["numericColumn"],
                                valueParser=JsCode("function(p){const v=String(p.newValue).trim(); "
                                                   "if(v==='') return null; const x=Number(v); "
                                                   "return isNaN(x) ? p.oldValue : x;}"), **kw)
    gb.configure_grid_options(singleClickEdit=True, stopEditingWhenCellsLoseFocus=True, rowHeight=32,
                              suppressColumnVirtualisation=True)
    r = AgGrid(df, gridOptions=gb.build(), allow_unsafe_jscode=True, update_on=["cellValueChanged"],
               key=f"eq_{name}_ag_{ver}", height=height or 36 + 32 * min(len(df), 14), theme="streamlit")
    out = r["data"] if r is not None and r.get("data") is not None else df
    out = pd.DataFrame(out).reindex(columns=df.columns).set_index(idx)
    out.index = base.index
    for c_ in numeric_cols:
        out[c_] = pd.to_numeric(out[c_], errors="coerce").astype(float)
    for c_ in [c_ for c_ in out.columns if c_ not in numeric_cols]:
        out[c_] = [v_ if isinstance(v_, str) and v_.strip() else None for v_ in out[c_]]
    return track("eq_" + name, out)


def grid(name, column_config, height="auto"):
    ed_key = f"eq_{name}_ed_{ss.get('_ver_eq_' + name, 0)}"
    return track("eq_" + name, st.data_editor(ss["eq_" + name], key=ed_key, num_rows="fixed", width="stretch",
                                             column_config=column_config, height=height))


SURF_TEXT = ("Label", "Type", "Fitting")
DH_TEXT = ("Label", "Type")
ss.setdefault("eq_survey", blank_grid(weq.SURVEY_COLUMNS, EQ_ROWS["survey"]))
ss.setdefault("eq_surface", blank_grid(weq.SURFACE_COLUMNS, EQ_ROWS["surface"], SURF_TEXT))
ss.setdefault("eq_downhole", blank_grid(weq.DOWNHOLE_COLUMNS, EQ_ROWS["downhole"], DH_TEXT))
ss.setdefault("eq_geo", blank_grid(weq.GEO_COLUMNS, EQ_ROWS["geo"]))
ss.setdefault("eq_long", pd.DataFrame({c: pd.Series(dtype=float) for c in weq.SURVEY_COLUMNS}))


def load_equipment_example():
    long_ = weq.example_long_survey()
    md, tvd = weq.survey_arrays(long_)
    idx, _, _ = weq.filter_survey(md, tvd)
    set_grid("survey", fill_grid(long_.iloc[idx], weq.SURVEY_COLUMNS, EQ_ROWS["survey"]))
    set_grid("long", long_)
    set_grid("surface", fill_grid(pd.DataFrame([
        {"Label": "Manifold", "Type": "Manifold", "TVD of End Point (ft)": 0.0},
        {"Label": "Flowline", "Type": "Line Pipe", "Pipe Length (ft)": 3000.0, "TVD of End Point (ft)": 0.0,
         "Pipe Inside Diameter (in)": 4.026, "Pipe Inside Roughness (in)": 0.0018, "Rate Multiplier": 1.0},
        {"Label": "Elbow", "Type": "Fitting", "Fitting": "90° elbow (standard)", "Pipe Inside Diameter (in)": 4.026},
        {"Label": "Wing valve", "Type": "Fitting", "Fitting": "Gate valve (open)", "Pipe Inside Diameter (in)": 4.026},
        {"Label": "Production choke", "Type": "Choke", "Pipe Inside Diameter (in)": 0.75},
        {"Label": "Spool", "Type": "Line Pipe", "Pipe Length (ft)": 30.0, "TVD of End Point (ft)": 0.0,
         "Pipe Inside Diameter (in)": 2.992, "Pipe Inside Roughness (in)": 0.0006, "Rate Multiplier": 1.0},
    ]), weq.SURFACE_COLUMNS, EQ_ROWS["surface"], SURF_TEXT))
    set_grid("downhole", fill_grid(pd.DataFrame([
        {"Type": "Xmas Tree", "Measured Depth (ft)": 35.5},
        {"Type": "Tubing", "Measured Depth (ft)": 136.54, "Tubing Inside Diameter (in)": 2.992,
         "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 3.5,
         "Tubing Outside Roughness (in)": 0.0006, "Rate Multiplier": 1.0},
        {"Label": "WRSCSSSV", "Type": "SSSV", "Tubing Inside Diameter (in)": 1.5},
        {"Type": "Tubing", "Measured Depth (ft)": 10962.0, "Tubing Inside Diameter (in)": 2.992,
         "Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Diameter (in)": 3.5,
         "Tubing Outside Roughness (in)": 0.0006, "Rate Multiplier": 1.0},
        {"Label": "Casing", "Type": "Casing", "Measured Depth (ft)": 10963.0, "Casing Inside Diameter (in)": 8.681,
         "Casing Inside Roughness (in)": 0.0006, "Rate Multiplier": 1.0},
    ]), weq.DOWNHOLE_COLUMNS, EQ_ROWS["downhole"], DH_TEXT))
    set_grid("geo", fill_grid(pd.DataFrame({weq.GEO_COLUMNS[0]: [35.5, 10963.0], weq.GEO_COLUMNS[1]: [60.0, 182.19]}),
                              weq.GEO_COLUMNS, EQ_ROWS["geo"]))
    put("eq_u", 4.17361)


def clear_equipment():
    set_grid("survey", blank_grid(weq.SURVEY_COLUMNS, EQ_ROWS["survey"]))
    set_grid("surface", blank_grid(weq.SURFACE_COLUMNS, EQ_ROWS["surface"], SURF_TEXT))
    set_grid("downhole", blank_grid(weq.DOWNHOLE_COLUMNS, EQ_ROWS["downhole"], DH_TEXT))
    set_grid("geo", blank_grid(weq.GEO_COLUMNS, EQ_ROWS["geo"]))
    set_grid("long", pd.DataFrame({c: pd.Series(dtype=float) for c in weq.SURVEY_COLUMNS}))


def read_survey_file(f):
    name = f.name.lower()
    if name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(f)
    else:
        raw = pd.read_csv(f, sep=None, engine="python")
    num_ = raw.apply(pd.to_numeric, errors="coerce")
    cols = [c for c in raw.columns if num_[c].notna().mean() > 0.8]
    lower = {c: str(c).lower() for c in cols}
    md_c = next((c for c in cols if "md" in lower[c] or "measured" in lower[c]), None)
    tvd_c = next((c for c in cols if "tvd" in lower[c] or "vertical" in lower[c]), None)
    if md_c is None or tvd_c is None:
        if len(cols) < 2:
            raise ValueError("Could not find two numeric columns (MD and TVD) in the file.")
        md_c, tvd_c = cols[0], cols[1]
    out = pd.DataFrame({weq.SURVEY_COLUMNS[0]: num_[md_c], weq.SURVEY_COLUMNS[1]: num_[tvd_c]}).dropna()
    return out.reset_index(drop=True)


def transfer_filtered(md, tvd, idx):
    set_grid("survey", fill_grid(pd.DataFrame({weq.SURVEY_COLUMNS[0]: md[idx], weq.SURVEY_COLUMNS[1]: tvd[idx]}),
                                 weq.SURVEY_COLUMNS, EQ_ROWS["survey"]))


def show_issues(issues):
    for lvl, msg in issues:
        {"error": st.error, "warning": st.warning, "info": st.info}[lvl](msg)


def status_of(issues):
    lv = {l for l, _ in issues}
    return "❌ Needs fixing" if "error" in lv else ("⚠️ Check warnings" if "warning" in lv else "✅ OK")


def survey_plot(md, tvd, long_md=None, long_tvd=None):
    t = weq.survey_table(md, tvd)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    if long_md is not None:
        lt = weq.survey_table(long_md, long_tvd)
        a.plot(lt["Cumulative Displacement (ft)"], lt["True Vertical Depth (ft)"], color=wplt.INK2, lw=1,
               label=f"full survey ({len(long_md)} points)")
    a.plot(t["Cumulative Displacement (ft)"], t["True Vertical Depth (ft)"], "-o", ms=3.5, color=wplt.SERIES[0],
           label=f"used ({len(md)} points)")
    a.invert_yaxis(); a.set_xlabel("Cumulative displacement (ft)"); a.set_ylabel("TVD (ft)")
    a.set_title("Well path", fontsize=10); a.legend(fontsize=8)
    b.step(t["Measured Depth (ft)"], t["Angle from vertical (deg)"], where="post", color=wplt.SERIES[0])
    b.set_xlabel("Measured depth (ft)"); b.set_ylabel("Angle from vertical (deg)")
    b.set_title("Deviation angle", fontsize=10)
    fig.tight_layout()
    return fig


def geo_plot(gt, secs_bottom=None):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(gt["Formation Temperature (deg F)"], gt["Formation TVD (ft)"], "-o", ms=4, color=wplt.SERIES[1])
    ax.invert_yaxis(); ax.set_xlabel("Formation temperature (°F)"); ax.set_ylabel("TVD (ft)")
    ax.set_title("Geothermal gradient", fontsize=10)
    fig.tight_layout()
    return fig


with tabs[3]:
    st.header("Equipment Data")
    o = ss.get("options")
    st.caption("Well path, surface and downhole equipment, temperature profile and heat capacities for the "
               "tubing (VLP) calculations. Depths are measured from the depth reference used for the "
               "deviation survey (normally RKB).")
    c = st.columns([1, 1, 4])
    c[0].button("Load example", on_click=load_equipment_example, width="stretch",
                help="Synthetic J-shaped well with the downhole and temperature data from the PROSPER screenshots.")
    c[1].button("Clear all", on_click=clear_equipment, width="stretch")
    status_box = st.container()
    sub = st.tabs(["Deviation survey", "Surface equipment", "Downhole equipment", "Geothermal gradient",
                   "Average heat capacities"])
    status = {}

    # ------------------------------------------------------ deviation survey
    with sub[0]:
        st.caption(f"Measured depth and true vertical depth, top down. Up to {weq.MAX_SURVEY_POINTS} points are "
                   "used; turn on the filter to enter a longer survey and pick the "
                   f"{weq.MAX_SURVEY_POINTS} points that best keep its shape.")
        cc = st.columns([1, 1.6])
        with cc[0]:
            ed = grid("survey", {c_: st.column_config.NumberColumn(c_, format="%g") for c_ in weq.SURVEY_COLUMNS},
                      height=36 * (EQ_ROWS["survey"] + 1) + 3)
        md_s, tvd_s = weq.survey_arrays(ed)
        iss = weq.check_survey(md_s, tvd_s)
        status["Deviation survey"] = iss
        long_md = long_tvd = None
        with cc[1]:
            if not any(l == "error" for l, _ in iss):
                st.dataframe(weq.survey_table(md_s, tvd_s).style.format(precision=1), hide_index=True,
                             width="stretch")
            show_issues(iss)

        if st.toggle("Filter a longer survey", key="eq_filter_on",
                     help=f"Enter or upload any number of survey points; the tool keeps the {weq.MAX_SURVEY_POINTS} "
                          "that reproduce the well path best."):
            fc = st.columns([1.2, 1, 1])
            up = fc[0].file_uploader("Upload survey (CSV, TXT or Excel with MD and TVD columns)",
                                     type=["csv", "txt", "xlsx", "xls"], key="eq_survey_file")
            if up is not None and ss.get("_eq_file_seen") != (up.name, up.size):
                try:
                    set_grid("long", read_survey_file(up))
                    ss["_eq_file_seen"] = (up.name, up.size)
                except (ValueError, pd.errors.ParserError) as e:
                    st.error(f"Could not read the file: {e}")
            fc[1].button("Example long survey", on_click=lambda: set_grid("long", weq.example_long_survey()),
                         help="150 noisy survey stations for a J-shaped well.")
            fc[2].button("Clear long survey", on_click=lambda: set_grid(
                "long", pd.DataFrame({c_: pd.Series(dtype=float) for c_ in weq.SURVEY_COLUMNS})))
            st.caption("Or paste rows into this table (any number of rows).")
            long_ed = track("eq_long", st.data_editor(ss["eq_long"], key=f"eq_long_ed_{ss.get('_ver_eq_long', 0)}",
                                     num_rows="dynamic", width="stretch", height=260,
                                     column_config={c_: st.column_config.NumberColumn(c_, format="%g")
                                                    for c_ in weq.SURVEY_COLUMNS}))
            lmd, ltvd = weq.survey_arrays(long_ed)
            if len(lmd) >= 2:
                liss = weq.check_survey(lmd, ltvd, max_points=10 ** 9)
                if any(l == "error" for l, _ in liss):
                    show_issues(liss)
                else:
                    idx, mx, rms = weq.filter_survey(lmd, ltvd)
                    st.info(f"{len(lmd)} points entered → {len(idx)} kept. Largest TVD error from filtering "
                            f"{mx:.2f} ft (RMS {rms:.2f} ft).")
                    st.button(f"Use the {len(idx)} filtered points", type="primary",
                              on_click=transfer_filtered, args=(lmd, ltvd, idx))
                    long_md, long_tvd = lmd, ltvd
        if len(md_s) >= 2 and not any(l == "error" for l, _ in iss):
            st.pyplot(survey_plot(md_s, tvd_s, long_md, long_tvd), clear_figure=True)
    survey_ok = len(md_s) >= 2 and not any(l == "error" for l, _ in status["Deviation survey"])
    survey_max = float(md_s.max()) if survey_ok else None

    # ------------------------------------------------------ surface equipment
    with sub[1]:
        st.caption("Flowline, choke and pipe fittings between the wellhead and the manifold. Enter the "
                   "Manifold first, then the equipment back towards the wellhead. Fittings are converted to an "
                   "equivalent pipe length (L/D method).")
        c = st.columns(4)
        t_sur = num(c[0], "Surrounding temperature (deg F)", "eq_tsur", 60.0, -60.0, 200.0)
        u_sur = num(c[1], "Overall heat transfer coefficient (BTU/h/ft2/F)", "eq_usur", 8.0, 0.0, 1000.0,
                    help="Flowline to surroundings: roughly 1-2 buried, 5-10 bare pipe in air, higher in water.")
        st.caption(("Cells open according to the type in each row, as in PROSPER: shaded cells take no input. "
                    if HAVE_AGGRID else "Cells open according to the type in each row, as in PROSPER: a value typed "
                    "into a locked cell is cleared. ")
                   + "Roughness defaults to 0.0006 in and the rate multiplier to 1 - both can be changed.")
        if not HAVE_AGGRID:
            with st.expander("Which cells each type opens", expanded=True):
                s_cols = ["Fitting"] + weq.SURFACE_NUMERIC
                s_short = {c_: c_.replace(" (in)", "").replace(" (ft)", "").replace("Pipe Inside ", "Pipe ") for c_ in s_cols}
                s_guide = pd.DataFrame([{s_short[c_]: ("open" if c_ in weq.surface_active_columns(t_) else "")
                                         for c_ in s_cols} for t_ in weq.SURFACE_TYPES], index=weq.SURFACE_TYPES)
                st.dataframe(s_guide.style.apply(lambda d: np.where(d == "", "background-color: rgba(120, 170, 215, 0.35)",
                                                                    "background-color: rgba(120, 200, 140, 0.25)"),
                                                 axis=None), width="stretch")
                st.caption("Green: accepts input. Blue: locked for that type (anything typed there is cleared).")
        ed_s = type_grid("surface", weq.SURFACE_TYPES,
                         {t_: weq.surface_active_columns(t_) for t_ in weq.SURFACE_TYPES},
                         weq.SURFACE_NUMERIC, {"Fitting": list(weq.FITTINGS)})
        s_clean = weq.apply_surface_rules(ed_s)
        if not weq.same_table(s_clean, ed_s):            # entry in a locked cell, or a new type needing defaults
            set_grid("surface", s_clean)
            st.rerun()
        iss = weq.check_surface(ed_s, t_sur, u_sur)
        status["Surface equipment"] = iss
        show_issues(iss)
        summ = weq.surface_summary(ed_s)
        if len(summ) and not any(l == "error" for l, _ in iss):
            st.markdown("**Summary** (manifold to wellhead)")
            st.dataframe(summ.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
            st.caption(f"Total equivalent length {summ['Equivalent length (ft)'].sum():,.0f} ft.")

    # ----------------------------------------------------- downhole equipment
    with sub[2]:
        ft = o.flow_type if o else "tubing"
        st.caption("Tubing string and its components from the Xmas tree down, then casing to the deepest "
                   "flowing depth. Point components (SSSV, restriction) sit at the previous depth unless an MD "
                   "is entered." + (" Annular flow is selected: tubing OD and casing ID/roughness are needed on "
                                    "every tubing row." if ft != "tubing" else ""))
        with st.expander("Equipment types"):
            st.markdown("\n".join(f"- **{k}** - {v}" for k, v in weq.TYPE_HELP.items()))
        st.caption(("Cells open according to the type in each row, as in PROSPER: shaded cells take no input. "
                    if HAVE_AGGRID else "Cells open according to the type in each row, as in PROSPER: a value typed "
                    "into a locked cell is cleared. ")
                   + "Roughness defaults to 0.0006 in and the rate multiplier to 1 - both can be changed."
                   + (" Annular flow also opens tubing OD/outside roughness and casing ID/roughness on tubing "
                      "rows." if ft != "tubing" else ""))
        if not HAVE_AGGRID:
            with st.expander("Which cells each type opens", expanded=True):
                short = {c_: c_.replace(" (in)", "").replace(" (ft)", "").replace("Tubing ", "Tbg ").replace(
                    "Casing ", "Csg ").replace("Inside ", "In. ").replace("Outside ", "Out. ") for c_ in weq.DH_NUMERIC}
                guide = pd.DataFrame([{short[c_]: ("open" if c_ in weq.active_columns(t_, ft) else "")
                                       for c_ in weq.DH_NUMERIC} for t_ in weq.DOWNHOLE_TYPES], index=weq.DOWNHOLE_TYPES)
                st.dataframe(guide.style.apply(lambda d: np.where(d == "", "background-color: rgba(120, 170, 215, 0.35)",
                                                                  "background-color: rgba(120, 200, 140, 0.25)"), axis=None),
                             width="stretch")
                st.caption("Green: accepts input. Blue: locked for that type (anything typed there is cleared).")
        ed_d = type_grid("downhole", weq.DOWNHOLE_TYPES,
                         {t_: weq.active_columns(t_, ft) for t_ in weq.DOWNHOLE_TYPES}, weq.DH_NUMERIC)
        dh_clean = weq.apply_cell_rules(ed_d, ft)
        if not weq.same_table(dh_clean, ed_d):          # entry in a locked cell, or a new type needing defaults
            set_grid("downhole", dh_clean)
            st.rerun()
        iss = weq.check_downhole(ed_d, ft, survey_max)
        status["Downhole equipment"] = iss
        show_issues(iss)
        top_md, secs, pts = None, pd.DataFrame(), pd.DataFrame()
        if not any(l == "error" for l, _ in iss):
            top_md, secs, pts = weq.downhole_sections(ed_d, md_s if survey_ok else None,
                                                      tvd_s if survey_ok else None)
            st.markdown("**Flow sections**")
            st.dataframe(secs.style.format(precision=4, na_rep="—"), hide_index=True, width="stretch")
            if len(pts):
                st.markdown("**Components**")
                st.dataframe(pts.style.format(precision=3, na_rep="—"), hide_index=True, width="stretch")
            st.caption("The well schematic is shown in the sidebar once every section is complete.")
    bottom_md = float(secs["MD bottom (ft)"].max()) if len(secs) else None

    # ----------------------------------------------------- geothermal gradient
    with sub[3]:
        st.caption("Formation temperature against measured depth, from the wellhead to the deepest equipment. "
                   "TVD is calculated from the deviation survey. The overall heat transfer coefficient sets the "
                   "heat loss between the flowing fluid and the formation.")
        u_geo = num(st.columns(4)[0], "Overall heat transfer coefficient (BTU/h/ft2/F)", "eq_u", 8.0, 0.0, 1000.0,
                    None, "%.5g")
        cc = st.columns([1, 1.3])
        with cc[0]:
            ed_g = grid("geo", {c_: st.column_config.NumberColumn(c_, format="%g") for c_ in weq.GEO_COLUMNS},
                        height=36 * 12)
        gmd, gtemp = weq.geo_arrays(ed_g)
        iss = weq.check_geothermal(gmd, gtemp, u_geo, top_md, bottom_md, survey_max)
        status["Geothermal gradient"] = iss
        with cc[1]:
            if survey_ok and len(gmd) >= 2 and not np.isnan(gmd).any() and not np.isnan(gtemp).any() \
                    and np.all(np.diff(gmd) > 0):
                gt = weq.geothermal_table(gmd, gtemp, md_s, tvd_s)
                st.dataframe(gt.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
                st.pyplot(geo_plot(gt), clear_figure=True)
            elif not survey_ok:
                st.info("Enter a valid deviation survey to calculate formation TVD.")
        show_issues(iss)

    # --------------------------------------------------- average heat capacities
    with sub[4]:
        st.caption("Average heat capacities used in the temperature calculation.")
        c = st.columns(4)
        hc = weq.HeatCapacities(num(c[0], "Cp Oil (BTU/lb/F)", "eq_cpo", 0.53, 0.01, 5.0, None, "%.3f"),
                                num(c[1], "Cp Gas (BTU/lb/F)", "eq_cpg", 0.51, 0.01, 5.0, None, "%.3f"),
                                num(c[2], "Cp Water (BTU/lb/F)", "eq_cpw", 1.0, 0.01, 5.0, None, "%.3f"))
        if c[3].button("Reset to defaults"):
            for k_ in ("eq_cpo", "eq_cpg", "eq_cpw"):
                reset(k_)
            st.rerun()
        iss = hc.check()
        status["Average heat capacities"] = iss
        show_issues(iss)

    with status_box:
        st.dataframe(pd.DataFrame({"Section": list(status), "Status": [status_of(v) for v in status.values()],
                                   "Messages": [len(v) for v in status.values()]}),
                     hide_index=True, width="stretch")
    all_ok = all(not any(l == "error" for l, _ in v) for v in status.values())
    st.divider()
    st.subheader("String summary")
    if all_ok and len(secs):
        has_choke = "Choke" in set(weq.surface_summary(ed_s).get("Type", pd.Series(dtype=str)))
        svg = wsch.well_schematic_svg(top_md, secs, pts, o.completion if o else "cased_hole",
                                      o.sand_control if o else "none", has_choke)
        schematic_svg = svg
        rows_ = [{"Item": "Xmas tree", "MD (ft)": top_md, "ID (in)": np.nan}]
        rows_ += [{"Item": (f'{r["Type"]} shoe' if r["Type"] == "Casing" else "End of tubing")
                   if r["Label"] == r["Type"] else f'{r["Label"]} ({r["Type"].lower()})',
                   "MD (ft)": r["MD bottom (ft)"], "ID (in)": r["Flow ID (in)"]}
                  for i_, (_, r) in enumerate(secs.iterrows())
                  if r["Type"] == "Casing" or i_ == len(secs) - 1 or secs.iloc[i_ + 1]["Type"] != "Tubing"
                  or secs.iloc[i_ + 1]["Flow ID (in)"] != r["Flow ID (in)"]]
        rows_ += [{"Item": r["Label"] if r["Label"] == r["Type"] else f'{r["Label"]} ({r["Type"]})',
                   "MD (ft)": r["MD (ft)"], "ID (in)": r["ID (in)"]} for _, r in pts.iterrows()]
        summ_ = pd.DataFrame(rows_).sort_values("MD (ft)", kind="stable")
        if survey_ok:
            summ_["TVD (ft)"] = weq.tvd_at(md_s, tvd_s, summ_["MD (ft)"])
        if len(gmd) >= 2:
            summ_["Formation temperature (°F)"] = weq.temperature_at(gmd, gtemp, summ_["MD (ft)"])
        fmt_ = {c_: "{:,.1f}" for c_ in summ_.columns if c_ not in ("Item", "ID (in)")}
        fmt_["ID (in)"] = "{:.3f}"
        st.dataframe(summ_.style.format(fmt_, na_rep="—"),
                     hide_index=True, width="stretch")
        st.caption(f"Completion: {o.label('completion').lower() if o else 'cased hole'}; sand control: "
                   f"{o.label('sand_control').lower() if o else 'none'} (from the Options summary). "
                   "The well schematic is in the sidebar.")
    else:
        st.info("The string summary and well schematic appear once every equipment section above is complete "
                "(no ❌ in the status table).")
    ss["equipment"] = {"survey": (md_s, tvd_s) if survey_ok else None, "surface": ed_s, "t_surround": t_sur,
                       "u_surface": u_sur, "downhole": (top_md, secs, pts), "geothermal": (gmd, gtemp),
                       "u_geo": u_geo, "heat_capacities": hc,
                       "valid": all(not any(l == "error" for l, _ in v) for v in status.values())}



# ================================================================ tab 5: VLP
SURFACE_CORRELATIONS = ["beggs_brill", "mukherjee_brill", "beggs_brill_noslip", "duns_ros", "duns_ros_gould",
                        "no_slip"]
TOP_NODES = ["Manifold", "Wellhead"]


def surface_items(eqd):
    """Surface items (flow order) from the Equipment tab, or None when no surface equipment is entered."""
    df = eqd.get("surface")
    if df is None or not len(weq.surface_rows(df)):
        return None
    try:
        _, items = wsurf.build_items(df)
    except (ValueError, TypeError, KeyError):
        return None
    return items or None


VLP_ROWS = 20
VLP_ACTIONS = ["Estimate U Value", "Correlation Comparison", "Match VLP", "VLP/IPR", "Adjust IPR", "Match Surface"]
EXAMPLE_SURFACE_TRUTH = (0.85, 1.2)        # Cd, flowline friction multiplier used for the example manifold pressures


def vlp_columns(kind):
    return wvlp.OIL_TEST_COLUMNS if kind == "oil" else wvlp.GAS_TEST_COLUMNS


def vlp_blank(kind):
    cols = vlp_columns(kind)
    df = pd.DataFrame({c: pd.Series([None] * VLP_ROWS, dtype=object) if c in ("Test Point Date", "Test Point Comment")
                       else pd.Series([np.nan] * VLP_ROWS, dtype=float) for c in cols})
    df.index = pd.RangeIndex(1, VLP_ROWS + 1, name="Test")
    return df


def vlp_set_tests(kind, df):
    ss["vlp_tests_" + kind] = df
    ss["_ver_vlp_tests_" + kind] = ss.get("_ver_vlp_tests_" + kind, 0) + 1
    for k_ in ("vlp_u", "vlp_cmp", "vlp_matches", "vlp_vi", "vlp_adj", "surf_match"):
        ss.pop(k_, None)


@st.cache_data(show_spinner=False)
def eos_vlp_grid(text, name, p_max, t_min, t_max):
    f = wm_eos.EOSFluid(wm_eos.parse_prp(text, name))
    grid = wvlp.eos_grid(f, p_max, t_min, t_max)
    sep = f.separator_train()
    return grid, float(f.z @ f.mw), float(sep["stages"]["Gas (scf per lbmol feed)"].sum()), \
        float(sep["stb_per_lbmol"]), float(sep["oil_density"]) if np.isfinite(sep["oil_density"]) else 50.0


def vlp_fluid(o, kind, p_max, t_min, t_max):
    """Fluid model for the tubing from the PVT tab. Returns (fluid, description) or (None, reason)."""
    water = bo.PVTInputs(rsb=1, api=40, gas_sg=0.7, salinity_ppm=float(val("pvt_sal", 50000.0)),
                         h2s=float(val("pvt_h2s", 0.0)), co2=float(val("pvt_co2", 0.0)), n2=float(val("pvt_n2", 0.0)))
    if o.method == "eos":
        if not ss.get("prp_text"):
            return None, "Load a PRP file in the PVT Input Data tab."
        p_max = float(np.ceil(p_max / 500) * 500)
        grid, mw, scf, stb, rho = eos_vlp_grid(ss["prp_text"], ss.get("prp_name", ""), p_max,
                                               float(np.floor(t_min / 10) * 10), float(np.ceil(t_max / 10) * 10))
        if kind == "oil" and stb <= 0:
            return None, "The PRP fluid gives no stock-tank oil."
        water.salinity_ppm = float(eos_parse(ss["prp_text"], ss.get("prp_name", "")).salinity_ppm or 0.0)
        return wvlp.EOSTableFluid(kind, grid, mw, scf, stb, rho, water), \
            f"Equation of state ({ss.get('prp_name', 'PRP file')}): flashed on a pressure-temperature grid"
    if kind == "oil":
        model = blackoil_model(o)
        if model is None:
            return None, "Complete the PVT Input Data tab (black oil)."
        fam = f"{bo.PB_FAMILIES[model.family]} / {bo.VISC_FAMILIES[model.visc_family]}"
        return wvlp.BlackOilFluid(model), f"Black oil ({fam}{', matched' if ss.get('pvt_model') else ', not matched'})"
    inp = bo.PVTInputs(rsb=1, api=40, gas_sg=float(val("pvt_gas_sg", 0.70)), salinity_ppm=water.salinity_ppm,
                       h2s=water.h2s, co2=water.co2, n2=water.n2)
    api = float(val("pvt_cond_api", 50.0))
    return wvlp.WetGasFluid(inp, api), f"Gas correlations, gas gravity {inp.gas_sg:.3f}, condensate {api:g} °API (wet gas)"


def vlp_path(eqd, flow_type, extra_md):
    top, secs, pts = eqd["downhole"]
    smd, stvd = eqd["survey"]
    gmd, gt = eqd["geothermal"]
    return wvlp.build_flow_path(top, secs, pts, smd, stvd, gmd, gt, flow_type, extra_md)


def rebuild_ipr(changes):
    """Rebuild the tab-3 IPR with some arguments changed (pr, skin, j). Raises ValueError if not possible."""
    rec = ss.get("ipr_recipe")
    if rec is None:
        raise ValueError("No IPR recipe - recalculate the IPR tab.")
    fn, a, k = rec
    b = inspect.signature(fn).bind(*a, **k)
    for key, v in changes.items():
        if key not in inspect.signature(fn).parameters:
            raise ValueError(f"The {ss.get('ipr_meta', {}).get('model', 'current')} IPR has no '{key}' input.")
        b.arguments[key] = v
    c = fn(*b.args, **b.kwargs)
    if ss.get("ipr_sc_dp") is not None:
        c = wcomp.apply_sand_control(c, ss["ipr_sc_dp"])
    return c


def ipr_arg(name):
    rec = ss.get("ipr_recipe")
    if rec is None:
        return None
    fn, a, k = rec
    b = inspect.signature(fn).bind(*a, **k)
    b.apply_defaults()
    return b.arguments.get(name)


def ipr_for_test(curve, pr_test):
    """IPR at the test reservoir pressure when it differs and the model can be rebuilt."""
    if pr_test is None or not np.isfinite(pr_test) or abs(pr_test - curve.pr) < 0.5:
        return curve, ""
    try:
        return rebuild_ipr({"pr": float(pr_test)}), f"IPR rebuilt at the test reservoir pressure {pr_test:,.0f} psia"
    except (ValueError, TypeError, KeyError) as e:
        return curve, f"IPR kept at {curve.pr:,.0f} psia ({e})"


EXAMPLE_PI_FACTOR = 1.10


def vlp_example_tests(kind, path, fluid, u, hc, pr, ipr_curve=None):
    """Synthetic tests from the current equipment and PVT (truth: Hagedorn & Brown for oil, Gray for gas,
    P1 1.04, P2 1.12). With an IPR, each test is the operating point of the VLP with that IPR made 10 % more
    productive - so VLP/IPR shows a gap that Adjust IPR can close."""
    gauge = float(path.md[np.argmin(np.abs(path.md - (path.top + 0.9 * (path.bottom - path.top))))])
    if kind == "oil":
        t = pd.DataFrame({"thp": [250.0, 300.0, 220.0, 350.0], "q": [1500.0, 2500.0, 3200.0, 1000.0],
                          "wc": [0.10, 0.20, 0.30, 0.05], "gor": [750.0, 780.0, 820.0, 750.0]})
        corr = "hagedorn_brown"
    else:
        t = pd.DataFrame({"thp": [2600.0, 2300.0, 2000.0, 1700.0], "q": [8.0, 12.0, 16.0, 20.0],
                          "wgr": [2.0, 2.5, 3.0, 4.0], "cgr": [25.0, 25.0, 24.0, 24.0]})
        corr = "gray"
    if ipr_curve is not None and ipr_curve.fluid == kind:
        scale = 1000.0 if kind == "gas" else 1.0
        iq, ip = ipr_curve.q * EXAMPLE_PI_FACTOR / scale, ipr_curve.pwf
        rates = np.linspace(iq.max() * 0.005, iq.max(), 60)
        thps = np.array([0.05, 0.065, 0.08, 0.10]) * ipr_curve.pr if kind == "oil" else \
            np.array([0.45, 0.55, 0.62, 0.70]) * ipr_curve.pr
        keep = []
        for i in range(len(t)):
            for thp in (thps[i], thps[i] * 0.7, thps[i] * 0.5):
                row = t.iloc[i].copy()
                row["thp"] = round(float(thp), -1)
                bhp = wvlp.vlp_curve(path, fluid, row, rates, corr, 1.04, 1.12, u, hc)
                hit = wvlp.intersect(rates, bhp, iq, ip)
                if hit:
                    row["q"] = round(hit[0], 0 if kind == "oil" else 3)
                    keep.append(row)
                    break
        if keep:
            t = pd.DataFrame(keep).reset_index(drop=True)
    P, T = wvlp.traverse(path, fluid, t["thp"].to_numpy(), wvlp.case_of(t, kind), corr, 1.04, 1.12, u, hc)
    gp = P[:, path.node(gauge)] + np.array([3.0, -4.0, 2.0, -2.0])[:len(t)]
    pman, beans = example_manifold(kind, fluid, t, T[:, 0], hc)
    rows = []
    for i, r in t.iterrows():
        d = {"Test Point Date": f"2026-0{i + 3}-15", "Test Point Comment": "synthetic",
             "Tubing Head Pressure (psia)": r["thp"], "Tubing Head Temperature (deg F)": round(float(T[i, 0]), 1),
             "Gauge Depth (Measured) (feet)": gauge, "Gauge Pressure (psia)": round(float(gp[i]), 1),
             "Reservoir Pressure (psia)": pr}
        if np.isfinite(pman[i]):
            d["Manifold Pressure (psia)"] = round(float(pman[i]), 1)
            if np.isfinite(beans[i]):
                d["Choke Size (64ths in)"] = beans[i]
        if kind == "oil":
            d.update({"Water Cut (percent)": r["wc"] * 100, "Liquid Rate (STB/day)": r["q"],
                      "Gas Oil Ratio (scf/STB)": r["gor"], "GOR Free (scf/STB)": 0.0})
        else:
            d.update({"Water Gas Ratio (STB/MMscf)": r["wgr"], "Separator Gas Oil Ratio (scf/STB)": round(1e6 / r["cgr"]),
                      "Gas Rate (MMscf/day)": r["q"]})
        rows.append(d)
    df = vlp_blank(kind)
    for i, d in enumerate(rows):
        for c_, v_ in d.items():
            df.iat[i, df.columns.get_loc(c_)] = v_
    return df


def example_manifold(kind, fluid, t, t_wh, hc):
    """Manifold pressures for the example tests: the surface equipment of the Equipment tab marched downstream from
    each test's THP with Cd 0.85 and friction x 1.2, at a mix of bean sizes (some critical, some subcritical)."""
    n = len(t)
    nan = np.full(n, np.nan)
    eqd = ss.get("equipment", {})
    items = surface_items(eqd)
    if items is None:
        return nan, nan
    cd, fr = EXAMPLE_SURFACE_TRUTH
    case = wvlp.case_of(t, kind)
    thp = t["thp"].to_numpy(float)
    choke = wsurf.has_choke(items)
    cands = [24, 32, 40, 48, 64, 80, 96, 128] if choke else [None]
    res = {}
    for b in cands:
        its = wsurf.with_choke_size(items, b) if b else items
        try:
            res[b] = wsurf.manifold_pressure(its, fluid, kind, thp, case, t_wh, "beggs_brill", hc,
                                             float(eqd.get("t_surround", 60.0)), float(eqd.get("u_surface", 8.0)),
                                             "Sachdeva", cd, 1.0, fr)
        except (ValueError, FloatingPointError):
            res[b] = nan
    pman, beans = nan.copy(), nan.copy()
    for i in range(n):
        ok = [b for b in cands if np.isfinite(res[b][i]) and 50.0 < res[b][i] < 0.92 * thp[i]]
        if ok:
            b = ok[(i * 2) % len(ok)]       # spread the tests over the feasible beans
            pman[i] = res[b][i]
            beans[i] = float(b) if b else np.nan
    return pman, beans


def traverse_plot(path, P_by, T_by, test, title, temp_meas=None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, (name, P) in enumerate(P_by.items()):
        a.plot(P, path.tvd, color=wplt.SERIES[i % len(wplt.SERIES)], lw=1.6, label=name)
    if np.isfinite(test.get("gauge_p", np.nan)):
        a.scatter([test["gauge_p"]], [np.interp(test["gauge_md"], path.md, path.tvd)], s=60, color=wplt.INK,
                  zorder=5, label="gauge (measured)")
    a.invert_yaxis(); a.set_xlabel("Pressure (psia)"); a.set_ylabel("TVD (ft)")
    a.set_title(title, fontsize=10); a.legend(fontsize=8, loc="lower left")
    b.plot(path.t_form, path.tvd, color=wplt.INK2, ls="--", lw=1.2, label="formation")
    for i, (name, T) in enumerate(T_by.items()):
        b.plot(T, path.tvd, color=wplt.SERIES[i % len(wplt.SERIES)], lw=1.4, label=name)
    if temp_meas is not None and np.isfinite(temp_meas):
        b.scatter([temp_meas], [path.tvd[0]], s=60, color=wplt.INK, zorder=5, label="THT (measured)")
    b.invert_yaxis(); b.set_xlabel("Temperature (°F)"); b.set_title("Temperature", fontsize=10)
    b.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def show_vlp_issues(issues, infos=True):
    for lvl, msg in issues:
        if lvl == "error":
            st.error(msg)
        elif lvl == "warning":
            st.warning(msg)
        elif infos:
            st.caption("ℹ️ " + msg)


def blank_table(df, cols, fmt="{:.6g}"):
    """Display copy with numbers as text and empty cells blank (st.dataframe shows NaN as 'None')."""
    out = df.copy()
    for c in cols:
        out[c] = [fmt.format(v) if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v) else ""
                  for v in df[c]]
    return out


def gradient_workbook(tables, names):
    """Excel workbook with one sheet of gradient results per correlation."""
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        used = set()
        for key, df in tables.items():
            sheet = "".join(ch for ch in names.get(key, key) if ch not in '[]:*?/\\')[:31] or key[:31]
            while sheet in used:
                sheet = sheet[:28] + f"_{len(used)}"
            used.add(sheet)
            df.to_excel(xw, sheet_name=sheet, index=False)
    return buf.getvalue()


def _test_phase_rates(kind, t, q):
    """Surface phase rates of a test at total rate q (liquid STB/d for oil, MMscf/d gas for gas wells)."""
    if kind == "oil":
        wc, gor = float(t["wc"]), float(t["gor"])
        return {"Liquid": (q, "STB/day"), "Oil": (q * (1 - wc), "STB/day"), "Water": (q * wc, "STB/day"),
                "Gas": (q * (1 - wc) * gor / 1e6, "MMscf/day")}
    return {"Gas": (q, "MMscf/day"), "Condensate": (q * float(t["cgr"]), "STB/day"),
            "Water": (q * float(t["wgr"]), "STB/day")}


def vlp_ipr_results(kind, t, hit, bhp_test):
    """PROSPER-style VLP/IPR result list for one test: calculated and measured rates, bottom hole pressures and
    the percentage differences."""
    calc = _test_phase_rates(kind, t, hit[0]) if hit else {k: (np.nan, u) for k, (_, u) in
                                                           _test_phase_rates(kind, t, 1.0).items()}
    meas = _test_phase_rates(kind, t, float(t["q"]))
    rows = [(f"Calculated {k} Rate", v, f"({u})") for k, (v, u) in calc.items()]
    rows.append(("Calculated Bottom Hole Pressure", hit[1] if hit else np.nan, "(psia)"))
    rows += [(f"Measured {k} Rate", v, f"({u})") for k, (v, u) in meas.items()]
    rows.append(("Measured Bottom Hole Pressure", bhp_test, "(psia)"))
    rows += [(f"% Difference {k} Rate", 100 * (calc[k][0] - v) / v if v else np.nan, "(percent)")
             for k, (v, _) in meas.items()]
    rows = [(l_, 0.0 if u_ == "(percent)" and abs(v_) < 5e-7 else v_, u_) for l_, v_, u_ in rows]
    return pd.DataFrame(rows, columns=["Label", "Value", "Units"])


def vlp_ipr_points(kind, t, rates, bhp, curve_t, scale):
    """Point table: phase rates, IPR pressure and VLP pressure at each calculated rate."""
    order = np.argsort(curve_t.q)
    iq, ip = np.asarray(curve_t.q)[order] / scale, np.asarray(curve_t.pwf)[order]
    rows = []
    for q, b in zip(rates, bhp):
        r = {f"{k} Rate ({u})": v for k, (v, u) in _test_phase_rates(kind, t, float(q)).items()}
        r["IPR Pressure (psia)"] = float(np.interp(q, iq, ip)) if q <= iq.max() else np.nan
        r["VLP Pressure (psia)"] = float(b)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.index = pd.RangeIndex(1, len(df) + 1, name="Point")
    return df


def _set_all(key, options, value):
    for c in options:
        ss[f"{key}__{c}"] = value


def corr_checklist(title, key, options, default, notes=None, height=None):
    """PROSPER-style correlation list: one checkbox per correlation, with an optional note beside each (e.g. the
    matched Parameter 1 and 2). The selection is kept in ss[key] (saved with the project)."""
    chosen0 = set(ss.get(key, default))
    notes = notes or {}
    with st.container(border=True):
        h = st.columns([3, 1, 1], vertical_alignment="center")
        h[0].markdown(f"**{title}**")
        h[1].button("All", key=f"{key}__all", on_click=_set_all, args=(key, options, True), width="stretch")
        h[2].button("None", key=f"{key}__none", on_click=_set_all, args=(key, options, False), width="stretch")
        box = st.container(height=height, border=False) if height else st.container(border=False)
        out, slots = [], {}
        with box:
            for c in options:
                k = f"{key}__{c}"
                row = st.columns([3, 2], vertical_alignment="center")
                if row[0].checkbox(VLP_LABELS.get(c, c), key=k, **({} if k in ss else {"value": c in chosen0})):
                    out.append(c)
                slots[c] = row[1].empty()
                if notes.get(c):
                    slots[c].caption(notes[c])
    ss[key] = out
    return out, slots


def accept_u(u):
    put("eq_u", float(round(u, 5)))
    ss["vlp_u_accepted"] = float(round(u, 5))


def accept_surface(cd, friction, corr):
    ss["surf_matched"] = {"cd": float(cd), "friction": float(friction), "corr": corr}
    put("sys_cd", float(round(cd, 4)))
    put("sys_scorr", corr)
    put("sys_choke", "Sachdeva")
    put("sys_use_smatch", True)


def accept_ipr(key, value, label):
    """Write the adjusted value to the IPR Data tab and go straight to VLP/IPR: the IPR tab runs before this
    one, so the rerun rebuilds the IPR with the new value and VLP/IPR is recalculated against it."""
    put(key, float(value))
    ss["vlp_adj_accepted"] = label
    ss["vlp_adj_just"] = label
    ss["vlp_action"] = "VLP/IPR"
    ss.pop("vlp_vi", None)


with tabs[4]:
    st.header("VLP")
    o = ss.get("options")
    eqd = ss.get("equipment", {})
    if o is None or o.errors:
        st.info("Fix the configuration in the Options summary first.")
    elif o.is_injector:
        st.info("VLP matching for injectors is not built yet - this tab covers producers.")
    elif not eqd.get("valid") or eqd.get("survey") is None or not len(eqd["downhole"][1]):
        st.info("Complete the Equipment Data tab first (every section in its status table must be OK).")
    else:
        kind = "gas" if o.is_gas else "oil"
        st.caption(("Gas" if kind == "gas" else "Oil") + " well test data. Pressures in psia; the tubing is calculated "
                   "from the Xmas tree down to the deepest equipment "
                   f"({eqd['downhole'][1]['MD bottom (ft)'].max():,.0f} ft MD), which is taken as the IPR depth.")
        key = "vlp_tests_" + kind
        ss.setdefault(key, vlp_blank(kind))
        ed_key = f"{key}_ed_{ss.get('_ver_' + key, 0)}"
        cfg = {c_: st.column_config.NumberColumn(c_.split(" (")[0] + "\n(" + c_.split(" (", 1)[1] if " (" in c_ else c_,
                                                 format="%g") for c_ in vlp_columns(kind)[2:]}
        cfg["Test Point Date"] = st.column_config.TextColumn("Test Point Date")
        cfg["Test Point Comment"] = st.column_config.TextColumn("Test Point Comment")
        tbl = track(key, st.data_editor(ss[key], key=ed_key, num_rows="fixed", width="stretch", column_config=cfg,
                                        height=36 * 11))

        u_cur = float(val("eq_u", eqd.get("u_geo", 8.0)))
        hc = eqd["heat_capacities"]
        top_md, secs_, _ = eqd["downhole"]
        tests, t_issues = wvlp.clean_tests(tbl, kind, float(top_md), float(secs_["MD bottom (ft)"].max()))
        ipr_curve = ss.get("ipr_curve")
        prs = [x for x in ([ipr_curve.pr] if ipr_curve is not None else []) + list(tests.get("pr", []))
               + list(tests.get("gauge_p", [])) if np.isfinite(x)]
        p_max = max(prs + [5000.0]) * 1.15
        gmd, gt = eqd["geothermal"]
        t_min = min([float(np.min(gt))] + [x for x in tests.get("tht", []) if np.isfinite(x)]) - 10
        fluid, fluid_note = vlp_fluid(o, kind, p_max, t_min, float(np.max(gt)) + 10)

        c = st.columns([1, 1, 3])
        if fluid is not None:
            def _load_example():
                path0 = vlp_path(eqd, o.flow_type, ())
                pr0 = float(ipr_curve.pr) if ipr_curve is not None else float(val("ipr_pr", 3000.0))
                vlp_set_tests(kind, vlp_example_tests(kind, path0, fluid, u_cur, hc, pr0, ipr_curve))
            c[0].button("Load example tests", on_click=_load_example, width="stretch",
                        help="Synthetic tests generated from the current equipment, PVT and IPR: Hagedorn & Brown "
                             "(oil) or Gray (gas) with P1 1.04 and P2 1.12, at the operating points of the IPR made "
                             "10 % more productive.")
        c[1].button("Clear tests", on_click=vlp_set_tests, args=(kind, vlp_blank(kind)), width="stretch")
        c[2].caption(f"PVT: {fluid_note}. U value {u_cur:.4g} BTU/h/ft2/F (Equipment tab).")
        show_vlp_issues(t_issues)
        bad = any(l == "error" for l, _ in t_issues)

        if fluid is None:
            st.warning(fluid_note)
        elif not len(tests):
            st.info("Enter at least one test, or load the example tests.")
        elif bad:
            st.info("Fix the test table errors above to continue.")
        else:
            labels = tests["label"].tolist()
            use = st.multiselect("Tests to use", labels, default=labels, key=f"vlp_use_{kind}_{len(labels)}")
            tests = tests[tests["label"].isin(use)].reset_index(drop=True)
            try:
                path = vlp_path(eqd, o.flow_type, [m for m in tests["gauge_md"] if np.isfinite(m)])
            except ValueError as e:
                st.error(str(e))
                path = None
            if path is not None and len(tests):
                st.divider()
                bc = st.columns(len(VLP_ACTIONS))
                for i, name in enumerate(VLP_ACTIONS):
                    bc[i].button(name, width="stretch", on_click=lambda n=name: ss.__setitem__("vlp_action", n),
                                 type="primary" if ss.get("vlp_action", VLP_ACTIONS[0]) == name else "secondary")
                action = ss.get("vlp_action", VLP_ACTIONS[0])
                st.subheader(action)
                qunit = "STB/d" if kind == "oil" else "MMscf/d"

                # ------------------------------------------------ estimate U
                if action == "Estimate U Value":
                    st.caption("Finds the overall heat transfer coefficient that makes the calculated wellhead "
                               "temperature equal to the measured tubing head temperature, test by test. The "
                               "temperature model uses the geothermal gradient and heat capacities from the "
                               "Equipment tab.")
                    tab_u = wvlp.estimate_u(path, fluid, tests, hc)
                    good = tab_u["U (BTU/h/ft2/F)"].dropna()
                    st.dataframe(tab_u.style.format({"Measured THT (°F)": "{:.1f}", "U (BTU/h/ft2/F)": "{:.4f}"},
                                                    na_rep="—"), hide_index=True, width="stretch")
                    if len(good):
                        u_new = float(good.median())
                        m = st.columns(4)
                        m[0].metric("Estimated U (median)", f"{u_new:.4g}")
                        m[1].metric("Current U (Equipment tab)", f"{u_cur:.4g}")
                        m[2].metric("Spread across tests", f"{good.min():.3g} – {good.max():.3g}")
                        m[3].button(f"Accept U = {u_new:.4g}", type="primary", on_click=accept_u, args=(u_new,),
                                    help="Transfers the value to the Geothermal gradient section of the Equipment tab.")
                        if ss.get("vlp_u_accepted") is not None and abs(ss["vlp_u_accepted"] - u_cur) < 1e-4:
                            st.success(f"U = {u_cur:.5g} BTU/h/ft2/F transferred to the Geothermal gradient section.")
                        case = wvlp.case_of(tests, kind)
                        w, cp = fluid.mass_cp(case, hc)
                        T_new = wvlp.temperature_profile(path, w, cp, u_new)
                        fig, ax = plt.subplots(figsize=(6.5, 4.4))
                        ax.plot(path.t_form, path.tvd, color=wplt.INK2, ls="--", label="formation")
                        for i, (_, t) in enumerate(tests.iterrows()):
                            col = wplt.SERIES[i % len(wplt.SERIES)]
                            ax.plot(T_new[i], path.tvd, color=col, lw=1.4, label=f"test {t['label']}")
                            if np.isfinite(t["tht"]):
                                ax.scatter([t["tht"]], [path.tvd[0]], color=col, s=45, zorder=5, edgecolor=wplt.INK)
                        ax.invert_yaxis(); ax.set_xlabel("Temperature (°F)"); ax.set_ylabel("TVD (ft)")
                        ax.set_title(f"Flowing temperature at U = {u_new:.4g} (dots: measured THT)", fontsize=10)
                        ax.legend(fontsize=8)
                        fig.tight_layout()
                        st.pyplot(fig, clear_figure=True)
                    else:
                        st.warning("No test has a usable tubing head temperature.")

                # ------------------------------------------- correlation comparison
                elif action == "Correlation Comparison":
                    st.caption("Each correlation, unmatched (Parameter 1 = Parameter 2 = 1), is run from the "
                               "measured tubing head pressure down to the gauge. The one with the smallest error "
                               "is suggested; Match VLP then fine-tunes it.")
                    last_ = ss.get("vlp_cmp_last", {})
                    corrs, slots_ = corr_checklist("Tubing correlation comparison", "vlp_cmp_corrs", wvlp.VLP_CORRELATIONS,
                                           wvlp.VLP_CORRELATIONS,
                                           {c_: f"mean error {v_:+.1f} psi" for c_, v_ in last_.items()})
                    with st.expander("Correlations not available"):
                        st.markdown("\n".join(f"- **{k_}** - {v_}" for k_, v_ in wvlp.NOT_AVAILABLE.items()))
                        st.caption("Fancher & Brown (lowest) and Duns & Ros (highest in slug flow) bound the "
                                   "measured gauge pressure as a quality check; they are flagged automatically below.")
                        st.caption("Suitability: Gray is intended for gas/condensate wells; Orkiszewski and Aziz were "
                                   "developed for oil wells and overstate holdup in high-pressure gas wells, where the "
                                   "Duns & Ros map places the flow in slug flow; Ansari and Mukherjee & Brill are "
                                   "general, with abrupt changes at flow-pattern boundaries.")
                    if not np.isfinite(tests["gauge_p"]).any():
                        st.warning("No test has gauge data - comparison needs a measured downhole pressure.")
                    elif corrs:
                        csig_ = (tuple(corrs), tests.to_json(), u_cur, fluid_note, o.flow_type,
                                 tuple(np.round(path.md, 2)), tuple(np.round(path.area, 8)))
                        if ss.get("vlp_cmp_cache", (None,))[0] != csig_:
                            with st.spinner("Running traverses..."):
                                ss["vlp_cmp_cache"] = (csig_, wvlp.compare(path, fluid, tests, corrs, u_cur, hc))
                        summ, detail, travs, g = ss["vlp_cmp_cache"][1]
                        ss["vlp_cmp_last"] = dict(zip(summ["key"], summ["Bias (psi)"]))
                        for c_, v_ in ss["vlp_cmp_last"].items():
                            if c_ in slots_:
                                slots_[c_].caption(f"mean error {v_:+.1f} psi")
                        cand = summ[~summ["key"].isin(wvlp.LOWER_BOUNDS)]
                        best = cand.iloc[0] if len(cand) else summ.iloc[0]
                        st.success(f"**Suggested: {best['Correlation']}** - mean absolute error "
                                   f"{best['Mean abs error (psi)']:.0f} psi ({best['Mean abs error (%)']:.1f} %) "
                                   f"over {int(best['Tests'])} test(s).")
                        st.dataframe(summ.drop(columns="key").style.format(precision=2), hide_index=True,
                                     width="stretch")
                        tsel = st.selectbox("Plot test", g["label"].tolist(), key="vlp_cmp_test")
                        i = g.index[g["label"] == tsel][0]
                        P_by = {VLP_LABELS[c_]: travs[c_][0][i] for c_ in corrs}
                        T_by = {"flowing": travs[corrs[0]][1][i]}
                        st.pyplot(traverse_plot(path, P_by, T_by, g.iloc[i], f"Test {tsel}: gradient traverses",
                                                g.iloc[i]["tht"]), clear_figure=True)
                        # ---- gradient results: one table per correlation for the plotted test
                        st.markdown(f"**Gradient results - test {tsel}**")
                        case_i = {k_: np.array([v_[i]]) for k_, v_ in wvlp.case_of(g, kind).items()}
                        labels_ = wgrad.node_labels(path, eqd["downhole"][2])
                        grads_ = {}
                        for c_ in corrs:
                            try:
                                grads_[c_] = wgrad.gradient_table(path, fluid, case_i, travs[c_][0][i],
                                                                  travs[c_][1][i], c_, 1.0, 1.0, u_cur, labels_)
                            except (ValueError, FloatingPointError, KeyError) as e:
                                st.warning(f"{VLP_LABELS[c_]}: gradient results not available ({e}).")
                        if grads_:
                            gsel = st.radio("Correlation", list(grads_), format_func=VLP_LABELS.get,
                                            horizontal=True, key="vlp_grad_corr")
                            gt_ = grads_[gsel]
                            num_ = [c_ for c_ in gt_.columns if c_ not in ("Point", "Label", "Regime")]
                            st.dataframe(blank_table(gt_, num_), hide_index=True, width="stretch", height=420,
                                         column_config={"Regime": st.column_config.TextColumn(width="medium"),
                                                        "Point": st.column_config.NumberColumn(width="small"),
                                                        **{c_: st.column_config.TextColumn(alignment="right")
                                                           for c_ in num_}})
                            st.caption("Unmatched (Parameter 1 = Parameter 2 = 1), from the measured THP down to the "
                                       "bottom node. Each row is a node; gradients, holdup, regime, velocities and "
                                       "fluid properties are for the segment above it. Gravity and friction pressure "
                                       "loss are the totals from the node down to the bottom node. Mixture density is "
                                       "the no-slip density; flowing density uses the holdup.")
                            dl_ = st.columns(2)
                            dl_[0].download_button(f"Download {VLP_LABELS[gsel]} (CSV)",
                                                   gt_.to_csv(index=False).encode(),
                                                   f"gradient_{gsel}_{_file_stem(str(tsel))}.csv", width="stretch")
                            dl_[1].download_button("Download all correlations (Excel, one sheet each)",
                                                   gradient_workbook(grads_, VLP_LABELS),
                                                   f"gradient_results_{_file_stem(str(tsel))}.xlsx",
                                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                   width="stretch")
                        with st.expander("Error by test"):
                            st.dataframe(detail.style.format(precision=2), hide_index=True, width="stretch")
                        ss["vlp_suggested"] = best["key"]
                        for lb_ in wvlp.LOWER_BOUNDS:
                            bad_ = detail[(detail["Correlation"] == VLP_LABELS[lb_]) & (detail["Error (psi)"] > 0)]
                            if len(bad_):
                                st.warning(f"{VLP_LABELS[lb_]} (a lower bound) is above the measured gauge pressure "
                                           f"for test(s) {', '.join(bad_['Test'])} - the gauge pressure, THP, rates or "
                                           "PVT for those tests are suspect.")
                                break
                        if "duns_ros" in corrs:
                            hi_ = detail[(detail["Correlation"] == VLP_LABELS["duns_ros"]) & (detail["Error (psi)"] < 0)]
                            if len(hi_):
                                st.info(f"Duns & Ros, usually the highest pressure in slug flow, is below the measured "
                                        f"gauge for test(s) {', '.join(hi_['Test'])} - check those tests.")

                # -------------------------------------------------------- match
                elif action == "Match VLP":
                    st.caption("Regression of Parameter 1 (gravity term multiplier) and Parameter 2 (friction "
                               "term multiplier) so the calculated gauge pressures match the tests. Parameter 1 "
                               "within 0.9-1.1 means the correlation and data are consistent; larger corrections "
                               "point to PVT, rate or gauge problems. TUFFP Unified is left unticked by default: "
                               "it is about 10x slower to regress.")
                    match_list = [c_ for c_ in wvlp.VLP_CORRELATIONS if c_ not in wvlp.LOWER_BOUNDS]
                    ms_prev_ = ss.get("vlp_matches", {}).get("ms", {})
                    corrs, slots_ = corr_checklist(
                        "Correlations to match", "vlp_match_corrs", match_list,
                        [c_ for c_ in match_list if c_ in ("pe5_style", "pe2_style", "duns_ros",
                                                           "duns_ros_gould", "ansari", "beggs_brill", "hagedorn_brown",
                                                           "gray", "mukherjee_brill", "orkiszewski", "aziz")],
                        {c_: f"P1 {m_.p1:.2f}   P2 {m_.p2:.2f}" for c_, m_ in ms_prev_.items()})
                    if not np.isfinite(tests["gauge_p"]).any():
                        st.warning("No test has gauge data to match.")
                    elif corrs:
                        sig = (tuple(corrs), tests.to_json(), u_cur, fluid_note, o.flow_type)
                        if st.button("Run match", type="primary") or (ss.get("vlp_matches", {}).get("sig") == sig):
                            if ss.get("vlp_matches", {}).get("sig") != sig:
                                with st.spinner("Regressing Parameter 1 and 2..."):
                                    ms = [wvlp.match(path, fluid, tests, c_, u_cur, hc) for c_ in corrs]
                                ss["vlp_matches"] = {"sig": sig, "ms": {m_.correlation: m_ for m_ in ms}}
                            ms = ss["vlp_matches"]["ms"]
                            for c_, m_ in ms.items():
                                if c_ in slots_:
                                    slots_[c_].caption(f"P1 {m_.p1:.2f}   P2 {m_.p2:.2f}")
                            mt = wvlp.match_table(list(ms.values()))
                            st.dataframe(mt.drop(columns=["key", "Score"]).style.format(
                                {"Parameter 1": "{:.4f}", "Parameter 2": "{:.4f}", "Std Deviation (psi)": "{:.2f}",
                                 "RMS error before (psi)": "{:.1f}", "RMS error after (psi)": "{:.2f}"}),
                                hide_index=True, width="stretch")
                            best = mt.iloc[0]["key"]
                            opts_ = list(mt["key"])
                            cur = ss.get("vlp_selected", best)
                            chosen = st.selectbox("Correlation to use for VLP/IPR", opts_,
                                                  index=opts_.index(cur) if cur in opts_ else 0,
                                                  format_func=VLP_LABELS.get, key=f"vlp_sel_{len(opts_)}_{best}",
                                                  help="Suggested: the correlation needing the smallest correction "
                                                       "with Parameter 1 within limits.")
                            ss["vlp_selected"] = chosen
                            sel_m = ms[chosen]
                            st.success(f"**{sel_m.label}**: Parameter 1 = {sel_m.p1:.4f}, Parameter 2 = "
                                       f"{sel_m.p2:.4f}, std deviation {sel_m.std:.2f} psi"
                                       + (" (suggested)" if chosen == best else ""))
                            for f_ in sel_m.flags:
                                st.warning(f_)
                            g = tests[np.isfinite(tests["gauge_p"])].reset_index(drop=True)
                            pc0, P0, T0 = wvlp.gauge_pressures(path, fluid, g, chosen, 1, 1, u_cur, hc)
                            pc1, P1_, T1 = wvlp.gauge_pressures(path, fluid, g, chosen, sel_m.p1, sel_m.p2, u_cur, hc)
                            tsel = st.selectbox("Plot test", g["label"].tolist(), key="vlp_match_test")
                            i = g.index[g["label"] == tsel][0]
                            st.pyplot(traverse_plot(path, {"unmatched": P0[i], "matched": P1_[i]}, {"flowing": T1[i]},
                                                    g.iloc[i], f"Test {tsel}: {sel_m.label}", g.iloc[i]["tht"]),
                                      clear_figure=True)
                    else:
                        st.info("Choose at least one correlation.")

                # --------------------------------------------------- match surface
                elif action == "Match Surface":
                    st.caption("Calibrates the surface equipment against tests with a measured manifold pressure: "
                               "from each manifold pressure the calculation works back through the flowline, "
                               "fittings and choke to the wellhead, and the Sachdeva discharge coefficient and a "
                               "flowline friction multiplier are regressed so the calculated wellhead pressure matches "
                               "the measured tubing head pressure. A choke size in the test table overrides the bean "
                               "of the Equipment tab for that test.")
                    s_items = surface_items(eqd)
                    st_ = tests[np.isfinite(tests["pman"])].reset_index(drop=True)
                    if s_items is None:
                        st.info("Enter the surface equipment (manifold, flowline, choke) in the Equipment Data tab.")
                    elif not len(st_):
                        st.info("Enter the manifold pressure (and the choke size, if it varied) for at least one "
                                "test in the table above.")
                    else:
                        cc_ = st.columns([1.2, 2, 1])
                        s_corr = sel(cc_[0], "Surface equipment correlation", SURFACE_CORRELATIONS, "vlp_scorr",
                                     format_func=VLP_LABELS.get)
                        fit_opts = {"cd": "Choke discharge coefficient (Cd)",
                                    "friction": "Flowline friction multiplier"}
                        fit_ = cc_[1].multiselect("Parameters to match", list(fit_opts),
                                                  **dflt("vlp_sfit", default=list(fit_opts)),
                                                  format_func=fit_opts.get, key="vlp_sfit")
                        sig_s = (st_.to_json(), s_corr, tuple(fit_), u_cur, fluid_note, repr(s_items),
                                 float(eqd.get("t_surround", 60.0)), float(eqd.get("u_surface", 8.0)))
                        cc_[2].markdown("&nbsp;")
                        run_s = cc_[2].button("Run surface match", type="primary", width="stretch")
                        if run_s and fit_:
                            case_ = wvlp.case_of(st_, kind)
                            w_, cp_ = fluid.mass_cp(case_, hc)
                            t_calc = wvlp.temperature_profile(path, w_, cp_, u_cur)[:, 0]
                            t_wh_ = np.where(np.isfinite(st_["tht"].to_numpy(float)), st_["tht"].to_numpy(float),
                                             t_calc)
                            try:
                                with st.spinner("Regressing the surface parameters..."):
                                    sm_ = wsurf.match_surface(s_items, fluid, kind, st_, t_wh_, s_corr, hc,
                                                              float(eqd.get("t_surround", 60.0)),
                                                              float(eqd.get("u_surface", 8.0)), fit=tuple(fit_))
                                ss["surf_match"] = {"sig": sig_s, "m": sm_, "corr": s_corr}
                            except (ValueError, FloatingPointError) as e:
                                st.error(f"Surface match failed: {e}")
                        elif run_s:
                            st.info("Choose at least one parameter to match.")
                        sm = ss.get("surf_match")
                        if sm and sm["sig"] == sig_s:
                            m_ = sm["m"]
                            mc = st.columns(4)
                            mc[0].metric("Discharge coefficient (Cd)", f"{m_.cd:.3f}",
                                         None if "cd" in m_.fitted else "held")
                            mc[1].metric("Friction multiplier", f"{m_.friction:.3f}",
                                         None if "friction" in m_.fitted else "held")
                            mc[2].metric("RMS THP error", f"{m_.rms_after:.1f} psi",
                                         f"{m_.rms_after - m_.rms_before:+.1f} psi vs unmatched", delta_color="inverse")
                            mc[3].button("Accept surface match", type="primary", width="stretch",
                                         on_click=accept_surface, args=(m_.cd, m_.friction, sm["corr"]),
                                         help="Uses Cd (Sachdeva), the friction multiplier and this flowline "
                                              "correlation in the Calculation tab.")
                            acc_ = ss.get("surf_matched")
                            if acc_ and abs(acc_["cd"] - m_.cd) < 1e-9 and abs(acc_["friction"] - m_.friction) < 1e-9:
                                st.success(f"Accepted: Cd {m_.cd:.3f}, friction multiplier {m_.friction:.3f}, "
                                           f"{VLP_LABELS[acc_['corr']]} - used by the Calculation tab.")
                            for n_ in m_.notes:
                                st.warning(n_)
                            st.dataframe(m_.table.style.format(precision=1, na_rep="—"), hide_index=True,
                                         width="stretch")
                            fig, ax = plt.subplots(figsize=(6.2, 4.6))
                            tb_ = m_.table
                            lo_ = float(min(tb_["Measured THP (psia)"].min(), tb_["Calculated THP - before (psia)"].min()))
                            hi_ = float(max(tb_["Measured THP (psia)"].max(), tb_["Calculated THP - before (psia)"].max()))
                            pad_ = 0.08 * (hi_ - lo_ + 1)
                            ax.plot([lo_ - pad_, hi_ + pad_], [lo_ - pad_, hi_ + pad_], color=wplt.INK2, lw=1, ls="--")
                            ax.scatter(tb_["Measured THP (psia)"], tb_["Calculated THP - before (psia)"], s=45,
                                       color=wplt.SERIES[1], label="unmatched (Cd 0.75, friction x 1)", zorder=4)
                            ax.scatter(tb_["Measured THP (psia)"], tb_["Calculated THP - matched (psia)"], s=55,
                                       color=wplt.SERIES[0], edgecolor=wplt.INK, label="matched", zorder=5)
                            for _, r_ in tb_.iterrows():
                                ax.annotate(str(r_["Test"]).split(" ")[0], (r_["Measured THP (psia)"],
                                            r_["Calculated THP - matched (psia)"]), xytext=(5, -10),
                                            textcoords="offset points", fontsize=8)
                            ax.set_xlabel("Measured THP (psia)"); ax.set_ylabel("Calculated THP (psia)")
                            ax.set_title("Wellhead pressure from the manifold pressure", fontsize=10)
                            ax.legend(fontsize=8)
                            fig.tight_layout()
                            pc_ = st.columns([1, 1])
                            pc_[0].pyplot(fig, clear_figure=True)
                            pc_[1].caption("Points on the dashed line: calculated wellhead pressure equals the "
                                           "measured THP. The choke regime column shows which tests can see the "
                                           "flowline: with a critical choke the wellhead pressure does not depend "
                                           "on anything downstream of the bean.")
                        elif sm:
                            st.info("Inputs have changed - press Run surface match to update.")

                # --------------------------------------------------------- VLP/IPR
                elif action in ("VLP/IPR", "Adjust IPR"):
                    ms = ss.get("vlp_matches", {}).get("ms", {})
                    chosen = ss.get("vlp_selected")
                    if not ms or chosen not in ms:
                        st.info("Run **Match VLP** first and choose the correlation to use.")
                    elif ipr_curve is None:
                        st.info("Calculate an IPR in the IPR Data tab first.")
                    elif ipr_curve.fluid != kind:
                        st.info("The IPR in the IPR Data tab is for a different fluid type - recalculate it.")
                    else:
                        mm = ms[chosen]
                        scale = 1000.0 if kind == "gas" else 1.0          # IPR gas rates are Mscf/d
                        rows_, curves_ = [], []
                        # Rate range: around the tests, not out to the AOF - a high-PI well has an AOF of
                        # 100,000s STB/d, and a grid spread that wide starts above the operating point.
                        aof_ = float(ipr_curve.aof) / scale
                        q_top = min(aof_, float(tests["q"].max()) * 3.0)
                        with st.spinner("Calculating VLP curves..."):
                            for _, t in tests.iterrows():
                                curve_t, note = ipr_for_test(ipr_curve, t["pr"])
                                q_hi = min(max(q_top, float(t["q"]) * 1.6), aof_)
                                for _ in range(6):                      # widen until the curves cross
                                    rates = np.unique(np.r_[np.geomspace(q_hi * 2e-3, q_hi, 15),
                                                            np.linspace(q_hi * 0.02, q_hi, 30),
                                                            [float(t["q"])] if t["q"] < q_hi else []])
                                    bhp = wvlp.vlp_curve(path, fluid, t, rates, chosen, mm.p1, mm.p2, u_cur, hc)
                                    hit = wvlp.intersect(rates, bhp, curve_t.q / scale, curve_t.pwf)
                                    if hit or q_hi >= aof_ * 0.999 or not np.isfinite(bhp[-1]) \
                                            or bhp[-1] > float(curve_t.pr):
                                        break
                                    q_hi = min(q_hi * 3.0, aof_)
                                q_top = max(q_top, q_hi)
                                bhp_test = float(wvlp.vlp_curve(path, fluid, t, [t["q"]], chosen, mm.p1, mm.p2,
                                                                u_cur, hc)[0])
                                ipr_at_test = float(np.interp(t["q"] * scale, np.sort(curve_t.q),
                                                              curve_t.pwf[np.argsort(curve_t.q)]))
                                rows_.append({"Test": t["label"], f"Test rate ({qunit})": t["q"],
                                              f"Calculated rate ({qunit})": hit[0] if hit else np.nan,
                                              "Rate difference (%)": 100 * (hit[0] - t["q"]) / t["q"] if hit else np.nan,
                                              "BHP from VLP at test rate (psia)": bhp_test,
                                              "BHP from IPR at test rate (psia)": ipr_at_test,
                                              "IPR reservoir pressure (psia)": curve_t.pr, "Note": note
                                              or ("" if hit else "no intersection - well does not flow")})
                                curves_.append((t, curve_t, rates, bhp, bhp_test, hit))
                        vi = pd.DataFrame(rows_)
                        ss["vlp_vi"] = vi
                        if action == "VLP/IPR":
                            just_ = ss.pop("vlp_adj_just", None)
                            if just_:
                                st.success(f"Adjust IPR accepted: {just_} written to the IPR Data tab. The IPR has "
                                           "been rebuilt and VLP/IPR recalculated with it.")
                            st.caption(f"VLP: {mm.label} matched (P1 {mm.p1:.3f}, P2 {mm.p2:.3f}) at the test's "
                                       f"THP and ratios. IPR: {ss.get('ipr_meta', {}).get('model', '')} from the IPR "
                                       "tab, rebuilt at the test's reservoir pressure where possible. Measured bottom "
                                       "hole pressure is the matched VLP at the measured rate (the test point).")
                            labels_ = tests["label"].tolist()
                            tsel_ = (st.selectbox("Test", labels_, key="vlp_vi_test") if len(labels_) > 1
                                     else labels_[0])
                            t, curve_t, rates, bhp, bhp_test, hit = curves_[labels_.index(tsel_)]
                            cc_ = st.columns([1, 1.7])
                            with cc_[0]:
                                res_ = vlp_ipr_results(kind, t, hit, bhp_test)
                                st.dataframe(res_.style.format({"Value": "{:.6g}"}, na_rep="—"), hide_index=True,
                                             width="stretch", height=35 * (len(res_) + 1) + 3)
                            with cc_[1]:
                                pts_ = vlp_ipr_points(kind, t, rates, bhp, curve_t, scale)
                                st.dataframe(blank_table(pts_, list(pts_.columns)), width="stretch",
                                             height=35 * (len(res_) + 1) + 3,
                                             column_config={c_: st.column_config.TextColumn(alignment="right")
                                                            for c_ in pts_.columns})
                            fig, ax = plt.subplots(figsize=(10, 4.6))
                            ax.plot(rates, bhp, color=wplt.SERIES[0], lw=1.8, label="VLP")
                            ax.plot(curve_t.q / scale, curve_t.pwf, color=wplt.SERIES[1], lw=1.8, label="IPR")
                            ax.scatter([t["q"]], [bhp_test], marker="s", s=60, color=wplt.SERIES[2],
                                       edgecolor=wplt.INK, zorder=5, label="test point")
                            if hit:
                                ax.scatter([hit[0]], [hit[1]], marker="o", s=50, facecolor="white",
                                           edgecolor=wplt.INK, lw=1.8, zorder=6, label="solution")
                            ax.set_xlabel(f"Liquid rate ({qunit})" if kind == "oil" else f"Gas rate ({qunit})")
                            ax.set_ylabel("Pressure (psia)")
                            ax.set_ylim(0, max(float(curve_t.pr), float(np.nanmax(bhp[np.isfinite(bhp)]))
                                               if np.isfinite(bhp).any() else 0) * 1.05)
                            ax.set_xlim(0, float(rates.max()))
                            ax.set_title(f"VLP/IPR - test {t['label']}", fontsize=10)
                            ax.grid(alpha=0.3, ls="--")
                            ax.legend(fontsize=8)
                            fig.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                            md_ = vi["Rate difference (%)"].abs().max()
                            if np.isfinite(md_):
                                (st.success if md_ <= 5 else st.warning)(
                                    f"Largest rate difference over {len(vi)} test(s): {md_:.2f} %. " +
                                    ("The IPR is consistent with the tests." if md_ <= 5
                                     else "Use Adjust IPR to bring the IPR in line with the tests."))
                            with st.expander("All tests", expanded=len(vi) > 1):
                                fmt_ = {c_: ("{:,.3f}" if kind == "gas" and "rate" in c_.lower() else "{:,.1f}")
                                        for c_ in vi.columns if c_ not in ("Test", "Note")}
                                fmt_["Rate difference (%)"] = "{:+.2f}"
                                st.dataframe(vi.style.format(fmt_, na_rep="—"), hide_index=True, width="stretch")
                            st.download_button("Download VLP/IPR points (CSV)", pts_.to_csv().encode(),
                                               f"vlp_ipr_{_file_stem(str(t['label']))}.csv")

                        # ----------------------------------------------- adjust IPR
                        else:
                            meta = ss.get("ipr_meta", {})
                            st.caption("Finds the IPR input that makes the IPR pass through a test point: the test "
                                       "rate at the bottomhole pressure the matched VLP gives for that rate. Accepting "
                                       "writes the value back to the IPR Data tab.")
                            tsel = st.selectbox("Test", tests["label"].tolist(), key="vlp_adj_test")
                            t = tests[tests["label"] == tsel].iloc[0]
                            bhp_t = float(vi.loc[vi["Test"] == tsel, "BHP from VLP at test rate (psia)"].iloc[0])
                            q_t = float(t["q"]) * scale
                            avail = []
                            if meta.get("model") == "P.I. Entry":
                                avail.append("Adjust PI")
                            if "Mechanical (entered)" in meta.get("skin_parts", {}) and ipr_arg("skin") is not None:
                                avail.append("Adjust skin")
                            if ipr_arg("pr") is not None:
                                avail.append("Adjust pressure")
                            st.caption(f"Test {tsel}: rate {t['q']:,.4g} {qunit}, BHP from matched VLP "
                                       f"{bhp_t:,.0f} psia. IPR model: {meta.get('model', '?')}.")
                            if not avail:
                                st.info("This IPR model has no adjustable PI, hand-entered skin or reservoir pressure.")
                            else:
                                how = st.radio("Parameter", avail, horizontal=True, key="vlp_adj_how")
                                res_txt, accept = None, None
                                pr_t = float(t["pr"]) if np.isfinite(t["pr"]) else float(ipr_curve.pr)
                                no_drawdown = how != "Adjust pressure" and bhp_t >= pr_t
                                if bhp_t <= 0:
                                    st.warning("The VLP gives no bottomhole pressure at the test rate - check the test.")
                                elif no_drawdown:
                                    dd_ = (f" At the current PI ({float(ipr_arg('j')):.4g} STB/d/psi) the drawdown for "
                                           f"{t['q']:,.0f} STB/d is only {q_t / float(ipr_arg('j')):,.1f} psi, so a VLP "
                                           "difference of a few tens of psi is enough to cause this."
                                           if meta.get("model") == "P.I. Entry" and ipr_arg("j")
                                           and q_t / float(ipr_arg("j")) < 100 else "")
                                    st.warning(
                                        f"The matched VLP gives {bhp_t:,.1f} psia at the test rate, {bhp_t - pr_t:,.1f} psi "
                                        f"above the reservoir pressure ({pr_t:,.1f} psia). No "
                                        f"{'PI' if how == 'Adjust PI' else 'skin'} can make the well flow at this "
                                        f"bottomhole pressure.{dd_} Options: **Adjust pressure** (the reservoir pressure "
                                        "that makes the IPR pass through the test point), or check the test's THP, "
                                        "reservoir pressure and gauge data and the VLP match.")
                                try:
                                    if bhp_t <= 0 or no_drawdown:
                                        raise StopIteration
                                    base = {"pr": pr_t} if ipr_arg("pr") is not None and how != "Adjust pressure" else {}
                                    if how == "Adjust PI":
                                        j0 = float(ipr_arg("j"))
                                        f_ = lambda j: float(rebuild_ipr({**base, "j": j}).rate(bhp_t)) - q_t
                                        j_new = brentq(f_, 1e-6, 1e6, xtol=1e-8)
                                        res_txt = f"PI {j0:.4g} → **{j_new:.4g} STB/d/psi**"
                                        accept = ("ipr_j", j_new, f"PI = {j_new:.4g} STB/d/psi")
                                    elif how == "Adjust skin":
                                        s0 = float(ipr_arg("skin"))
                                        entered = float(meta["skin_parts"]["Mechanical (entered)"])
                                        f_ = lambda s_: float(rebuild_ipr({**base, "skin": s_}).rate(bhp_t)) - q_t
                                        lo = -4.0
                                        while f_(lo) < 0 and lo > -8:
                                            lo -= 1
                                        s_new = brentq(f_, lo, 500.0, xtol=1e-6)
                                        new_entered = entered + (s_new - s0)
                                        res_txt = (f"Total skin {s0:.3g} → **{s_new:.3g}** (hand-entered skin "
                                                   f"{entered:.3g} → **{new_entered:.3g}**)")
                                        accept = ("ipr_skin", new_entered, f"skin = {new_entered:.3g}")
                                    else:
                                        f_ = lambda p_: float(rebuild_ipr({"pr": p_}).rate(bhp_t)) - q_t
                                        p_new = brentq(f_, bhp_t + 1e-3, max(bhp_t * 4, bhp_t + 8000), xtol=0.01)
                                        p0 = float(ipr_curve.pr)
                                        res_txt = f"Reservoir pressure {p0:,.0f} → **{p_new:,.1f} psia**"
                                        accept = ("ipr_pr", p_new, f"reservoir pressure = {p_new:,.1f} psia")
                                except StopIteration:
                                    pass
                                except (ValueError, RuntimeError, TypeError) as e:
                                    st.error("No value of this parameter makes the IPR pass through the test point "
                                             "(rate at the VLP bottomhole pressure)." if "different signs" in str(e)
                                             else f"No solution found: {e}")
                                if res_txt:
                                    st.success(res_txt)
                                    if accept[0] == "ipr_pr" and np.isfinite(t["pr"]) and abs(accept[1] - t["pr"]) > 50:
                                        st.warning(f"The adjusted pressure differs from the test reservoir pressure "
                                                   f"({t['pr']:,.0f} psia) by {accept[1] - t['pr']:+,.0f} psi.")
                                    st.button("Accept", type="primary", on_click=accept_ipr, args=accept)
                                    if ss.get("vlp_adj_accepted") == accept[2]:
                                        st.success(f"Accepted: {accept[2]} written to the IPR Data tab.")



# ======================================================== tab 6: calculation
RATE_METHODS = ["Automatic - Geometric", "Automatic - Linear", "User selected"]


def parse_values(text):
    out = []
    for tok in str(text).replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


def _plot_qmax(shown, full):
    """x-axis limit: the whole rate range, or (default) a window around the solution(s) - a high-PI well's IPR
    runs out to an AOF far beyond anything the tubing can lift."""
    q_all = max(r.rates.max() for r in shown) * 1.02
    sols = [r.q for r in shown if r.q is not None]
    if full or not sols:
        return q_all
    # include the VLP minimum (stability) as well as the solution
    q_min_vlp = max(float(r.rates[int(np.nanargmin(r.vlp))]) if np.isfinite(r.vlp).any() else 0.0 for r in shown)
    return min(q_all, max(3.0 * max(sols), 1.5 * q_min_vlp))


def sys_plot(results, kind, highlight=None, max_curves=12, full=False):
    if results and results[0].node == "top":
        return sys_plot_top(results, kind, max_curves, full)
    qunit = "STB/d" if kind == "oil" else "MMscf/d"
    scale = 1000.0 if kind == "gas" else 1.0
    fig, ax = plt.subplots(figsize=(9, 5.2))
    shown = results if len(results) <= max_curves else results[:max_curves]
    single = len(results) == 1
    cmap = plt.get_cmap("tab20" if len(shown) > 10 else "tab10")
    for i, r in enumerate(shown):
        col = cmap(i % cmap.N)
        lab = "" if single else " " + ", ".join(f"{k}={v:g}" for k, v in r.case_label.items())
        ax.plot(r.rates, r.vlp, color=wplt.SERIES[0] if single else col, lw=1.8, label="VLP" + lab)
        order = np.argsort(r.ipr_curve.q)
        ax.plot(r.ipr_curve.q[order] / scale, r.ipr_curve.pwf[order], color=wplt.SERIES[1] if single else col,
                lw=1.4, ls="-" if single else "--", label="IPR" if single else None)
        if r.q is not None:
            ax.scatter([r.q], [r.pwf], s=60, zorder=5, color=wplt.INK if r.stable else "#c0392b",
                       marker="o" if r.stable else "X")
    ax.set_xlabel(f"Rate ({qunit})"); ax.set_ylabel("Pressure at bottom node (psia)")
    top = max(float(np.nanmax(r.ipr_curve.pr)) for r in shown)
    qmax = _plot_qmax(shown, full)
    vis = [float(np.nanmax(r.vlp[r.rates <= qmax])) for r in shown if np.isfinite(r.vlp[r.rates <= qmax]).any()]
    ax.set_ylim(0, min(max([top] + vis) * 1.08, top * 1.6))
    ax.set_xlim(0, qmax)
    ax.grid(alpha=0.3, ls="--")
    ax.set_title("Inflow (IPR) v Outflow (VLP)" + ("" if single else "  -  dashed: IPR; X = unstable solution"),
                 fontsize=10)
    ax.legend(fontsize=7.5, ncol=2 if len(shown) > 6 else 1)
    fig.tight_layout()
    return fig


def sys_plot_top(results, kind, max_curves=12, full=False):
    """Solution at the top node: inflow (IPR + tubing [+ surface equipment]) against the top-node pressure."""
    qunit = "STB/d" if kind == "oil" else "MMscf/d"
    fig, ax = plt.subplots(figsize=(9, 5.2))
    shown = results if len(results) <= max_curves else results[:max_curves]
    single = len(results) == 1
    cmap = plt.get_cmap("tab20" if len(shown) > 10 else "tab10")
    surf = any(r.pwh is not None for r in shown)
    top_name = "manifold" if surf else "wellhead"
    ymax = 0.0
    for i, r in enumerate(shown):
        col = cmap(i % cmap.N)
        lab = "" if single else " " + ", ".join(f"{k}={v:g}" for k, v in r.case_label.items())
        qx, py = r.rates, r.top_in
        if r.q is not None and r.top_in_q is not None and np.isfinite(r.top_in_q):
            k_ = np.searchsorted(qx, r.q)
            qx, py = np.insert(qx, k_, r.q), np.insert(py, k_, r.top_in_q)
        ok = np.isfinite(py)
        ax.plot(qx[ok], py[ok], color=wplt.SERIES[1] if single else col, lw=1.8,
                label=("Inflow (IPR + tubing" + (" + surface equipment" if surf else "") + ")") if single
                else "Inflow" + lab)
        pt = float(r.params["whp"])
        ax.axhline(pt, color=wplt.SERIES[0] if single else col, lw=1.4, ls="-" if single else "--",
                   label=f"Outflow: {top_name} pressure" if single else None)
        ymax = max(ymax, float(np.nanmax(py)) if ok.any() else 0.0, pt)
        if r.q is not None:
            up = r.top_in_q if r.top_in_q is not None and np.isfinite(r.top_in_q) else \
                (float(np.interp(r.q, qx[ok], py[ok])) if ok.any() else pt)
            if up > pt + 1.0:                      # critical choke: rate set by the bean, vertical drop
                ax.plot([r.q, r.q], [up, pt], color=wplt.SERIES[1] if single else col, lw=1.4, ls=":")
                if single:
                    ax.annotate("critical choke - rate set by the bean", (r.q, 0.5 * (up + pt)),
                                xytext=(8, 0), textcoords="offset points", fontsize=8, va="center")
            ax.scatter([r.q], [pt], s=60, zorder=5, color=wplt.INK if r.stable else "#c0392b",
                       marker="o" if r.stable else "X")
    ax.set_xlabel(f"Rate ({qunit})"); ax.set_ylabel(f"Pressure at top node - {top_name} (psia)")
    ax.set_ylim(0, max(ymax, 1.0) * 1.25)
    ax.set_xlim(0, _plot_qmax(shown, full))
    ax.grid(alpha=0.3, ls="--")
    ax.set_title("Solution at the top node" + ("" if single else "  -  dashed: top-node pressure; X = unstable"),
                 fontsize=10)
    ax.legend(fontsize=7.5, ncol=2 if len(shown) > 6 else 1)
    fig.tight_layout()
    return fig


def sens_plot(tab, var_names, labels, qcol):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = var_names[0]
    others = var_names[1:]
    groups = tab.groupby(others) if others else [((), tab)]
    cmap = plt.get_cmap("tab10")
    for i, (key, g) in enumerate(groups):
        if i >= 12:
            break
        key = key if isinstance(key, tuple) else (key,)
        lab = ", ".join(f"{labels[o].split(' (')[0]} = {v:g}" for o, v in zip(others, key)) or "solution"
        g = g.sort_values(x)
        ax.plot(g[x], g[qcol], "-o", ms=4, color=cmap(i % cmap.N), label=lab)
    ax.set_xlabel(labels[x]); ax.set_ylabel(qcol)
    ax.set_title("Sensitivity: solution rate", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def solution_panel(kind, r, det, regimes):
    sr = wsys.surface_rates(kind, r.params, r.q)
    rows = [(k_, v_) for k_, v_ in sr.items()]
    rows += [("Solution node", "Bottom node" if r.node == "bottom" else
              ("Top node (manifold)" if r.pwh_q is not None else "Top node (wellhead)")),
             ("Solution node pressure (psia)", r.node_pressure),
             ("Top node (manifold) pressure (psia)" if r.pwh_q is not None else "Top node (wellhead) pressure (psia)",
              float(r.params["whp"]))]
    if r.pwh_q is not None:
        rows += [("Wellhead pressure (psia)", r.pwh_q)]
    rows += [("Wellhead temperature (°F)", r.wht_q)]
    rows += [(k_, v_) for k_, v_ in det.items() if k_ not in ("Wellhead pressure (psia)", "Wellhead temperature (°F)",
                                                              "Manifold pressure (psia)") and not k_.startswith("_")]
    rows += [(f"Regime - {k_}", f"{v_[0]} (dP {v_[1]:.1f} psi)") for k_, v_ in regimes.items()]
    rows += [("Stability", "Stable" if r.stable else "UNSTABLE - left of the VLP minimum"),
             ("Rate at VLP minimum", f"{r.q_vlp_min:,.4g} {'STB/d' if kind == 'oil' else 'MMscf/d'}")]
    out = pd.DataFrame(rows, columns=["Result", "Value"])
    out["Value"] = [f"{v_:,.4g}" if isinstance(v_, (int, float, np.floating)) else str(v_) for v_ in out["Value"]]
    return out


CALC_MODES = ["System analysis (IPR + VLP)", "Gradient traverse", "Lift curves (VFP tables)", "Choke performance"]


def _qunit(kind):
    return "STB/d" if kind == "oil" else "MMscf/d"


CHOKE_OPTIONS = ["Predict flow rate", "Predict pressure drop", "Predict choke setting"]


def choke_plot(ps, qs, ts, point, qunit, p1):
    """PROSPER-style choke performance plot: rate against outlet pressure (bottom axis) and against outlet
    temperature (top axis), with the calculated point."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(ps, qs, color=wplt.SERIES[0], lw=1.8, label="Outlet pressure")
    ax.set_xlabel("Outlet pressure (psia)"); ax.set_ylabel(f"Rate ({qunit})")
    ax.set_xlim(0, p1 * 1.02); ax.set_ylim(0, max(np.nanmax(qs), point[1] if point else 0) * 1.12)
    ax.grid(alpha=0.3, ls="--")
    ax2 = ax.twiny()
    ax2.plot(ts, qs, color=wplt.SERIES[1], lw=1.6, label="Outlet temperature")
    ax2.set_xlabel("Outlet temperature (°F)")
    if point:
        ax.scatter([point[0]], [point[1]], s=70, marker="s", color="#2ca02c", edgecolor=wplt.INK, zorder=6,
                   label="Calculated point")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax.set_title("Choke performance plot", fontsize=10, pad=34)
    fig.tight_layout()
    return fig


def choke_mode(o):
    """General-purpose choke calculator - needs only the PVT (tab 2)."""
    kind = "gas" if o.is_gas else "oil"
    qunit = "STB/d" if kind == "oil" else "MMscf/d"
    st.caption("General-purpose choke calculator: flow rate for a given choke size and pressures, the outlet "
               "pressure for a given rate, or the choke setting that gives a rate. Only the PVT data (tab 2) is "
               "needed. The critical values (pressure, temperature and rate) are always calculated. With continuous "
               "gas lift, include the lift gas in the total GOR.")
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        how = radio(st, "Calculation option", CHOKE_OPTIONS, "ck_how")
        model = sel(st, "Choke model", wchoke.MODELS, "ck_model",
                    help="ELF: Perkins (1993), isentropic multiphase flow, critical and subcritical, discharge "
                         "coefficient 0.826 (best fit to 1,432 data sets). Sachdeva et al. (1986), Cd 0.75 typical.")
        c = st.columns(2)
        ratios = {}
        if kind == "oil":
            gor = num(c[0], "Total GOR (scf/STB)", "ck_gor", float(val("sys_gor", 500.0)), 0.0, 1e6)
            wc = num(c[1], "Water cut (%)", "ck_wc", float(val("sys_wc", 0.0)), 0.0, 99.0)
            ratios = {"wc": wc / 100, "gor": gor}
        else:
            wgr = num(c[0], "Water gas ratio (STB/MMscf)", "ck_wgr", float(val("sys_wgr", 0.0)), 0.0, 1e5)
            sgor = num(c[1], "Separator GOR (scf/STB)", "ck_sgor", float(val("sys_sep_gor", 0.0)), 0.0, 1e8,
                       help="0 = dry gas")
            ratios = {"wgr": wgr, "cgr": 1e6 / sgor if sgor > 0 else 0.0}
        c = st.columns(2)
        p1 = num(c[0], "Inlet pressure (psia)", "ck_p1", 650.0, 15.0, 3e4)
        t1 = num(c[1], "Inlet temperature (°F)", "ck_t1", 120.0, 32.0, 400.0)
        c = st.columns(2)
        p4 = q = d64 = None
        if how != "Predict pressure drop":
            p4 = num(c[0], "Outlet pressure (psia)", "ck_p4", 400.0, 14.7, 3e4)
        if how != "Predict flow rate":
            q = num(c[1] if how != "Predict pressure drop" else c[0],
                    f"{'Liquid' if kind == 'oil' else 'Gas'} rate ({qunit})", "ck_q",
                    1000.0 if kind == "oil" else 5.0, 1e-6, 1e7)
        if how != "Predict choke setting":
            d64 = num(c[1] if how == "Predict flow rate" else c[1], "Choke size (64ths in)", "ck_d64", 32.0, 1.0, 256.0)
        c = st.columns(3)
        d_up = num(c[0], "Upstream pipe ID (in)", "ck_dup", 2.992, 0.1, 48.0, None, "%.3f",
                   help="Area upstream of the choke (A1 in Eq. A-28).")
        d_dn = num(c[1], "Downstream pipe ID (in)", "ck_ddn", 2.992, 0.1, 48.0, None, "%.3f",
                   help="Sets the pressure recovery downstream of the choke (Eq. 7).")
        elf = model.startswith("ELF")
        cd = num(c[2], "Discharge coefficient", "ck_cd_elf" if elf else "ck_cd_sach", 0.826 if elf else 0.75,
                 0.1, 1.5, None, "%.3f")
        f_in = num(st, "Gas Cp/Cv (0 = from correlation)", "ck_f", 0.0, 0.0, 3.0, None, "%.3f",
                   help="Ratio of specific heats of the gas. Perkins used Johnson's methane tables; by default the "
                        "ideal-gas value of a natural gas of this gravity is used.")
        go = st.button("Calculate", type="primary", key="ck_go")
    fluid, fluid_note = vlp_fluid(o, kind, max(p1, 1000.0) * 1.2, max(t1 - 80.0, 20.0), t1 + 20.0)
    with right:
        if fluid is None:
            st.warning(fluid_note)
            return
        st.caption(f"PVT: {fluid_note}.")
        setup = wchoke.ChokeSetup(fluid, kind, ratios, ss.get("equipment", {}).get("heat_capacities")
                                  or weq.HeatCapacities(), d_up=d_up, d_down=d_dn, cd=cd, model=model,
                                  f_ratio=f_in if f_in > 1.0 else None)
        sig = (how, model, tuple(sorted(ratios.items())), p1, t1, p4, q, d64, d_up, d_dn, cd, f_in, fluid_note)
        if go or ss.get("ck_res", {}).get("sig") == sig:
            if ss.get("ck_res", {}).get("sig") != sig:
                with st.spinner("Calculating..."):
                    try:
                        if how == "Predict flow rate":
                            d_in = d64 / 64
                            if d_in >= min(d_up, d_dn):
                                raise ValueError("The choke must be smaller than the pipe.")
                            r = setup.rate(p1, t1, p4, d_in)
                            out = {"d": d_in, "p4": p4, "q": r["q"], "r": r}
                        elif how == "Predict pressure drop":
                            d_in = d64 / 64
                            p4_, r = setup.downstream_pressure(p1, t1, q, d_in)
                            out = {"d": d_in, "p4": p4_, "q": q, "r": r}
                        else:
                            if p4 >= p1:
                                raise ValueError("The outlet pressure must be below the inlet pressure.")
                            d_in, r = setup.choke_size(p1, t1, p4, q)
                            out = {"d": d_in, "p4": p4, "q": q, "r": r}
                        if np.isfinite(out["d"]):
                            out["curve"] = setup.performance_curve(p1, t1, out["d"])
                        ss["ck_res"] = {"sig": sig, "out": out, "err": None}
                    except (ValueError, RuntimeError, ZeroDivisionError) as e:
                        ss["ck_res"] = {"sig": sig, "out": None, "err": str(e)}
            res = ss["ck_res"]
            if res["err"]:
                st.error(res["err"])
                return
            out, r = res["out"], res["out"]["r"]
            if how == "Predict pressure drop" and not np.isfinite(out["p4"]):
                st.error(f"The choke cannot pass {q:,.4g} {qunit} at {p1:,.1f} psia: the critical (maximum) rate is "
                         f"{r['q']:,.4g} {qunit}. Open the choke or raise the inlet pressure.")
            elif how == "Predict choke setting" and not np.isfinite(out["d"]):
                st.error("No choke smaller than the pipe passes this rate between these pressures.")
            else:
                rows = []
                if how == "Predict choke setting":
                    rows.append(("Choke setting", out["d"], "in"))
                    rows.append(("Choke setting", out["d"] * 64, "64ths in"))
                if how == "Predict flow rate":
                    rows.append((f"{'Liquid' if kind == 'oil' else 'Gas'} rate", out["q"], qunit))
                if how == "Predict pressure drop":
                    rows.append(("Outlet pressure", out["p4"], "psia"))
                    rows.append(("Pressure drop", p1 - out["p4"], "psi"))
                rows += [("Outlet temperature", r["t_out"], "°F"), ("Flow regime", r["regime"], ""),
                         ("Critical pressure (throat)", r["p_crit"], "psia"),
                         ("Critical rate", r["q_crit"], qunit), ("Critical temperature", r["t_crit"], "°F"),
                         ("Throat pressure", r["p_throat"], "psia"),
                         ("Gas mass fraction at the inlet", r["fg"], ""), ("Polytropic exponent n", r["n"], "")]
                if how == "Predict pressure drop" and r["regime"] == "Critical":
                    st.info("This rate is the critical rate: any outlet pressure up to the one shown gives it.")
                tab = pd.DataFrame(rows, columns=["Result", "Value", "Units"])
                tab["Value"] = [f"{v:,.5g}" if isinstance(v, (float, int, np.floating)) and np.isfinite(v)
                                else ("—" if isinstance(v, (float, np.floating)) else v) for v in tab["Value"]]
                st.dataframe(tab, hide_index=True, width="stretch")
                if "curve" in out:
                    ps, qs, ts = out["curve"]
                    st.pyplot(choke_plot(ps, qs, ts, (out["p4"], out["q"]), qunit, p1), clear_figure=True)
                    cur = pd.DataFrame({"Outlet pressure (psia)": ps, f"Rate ({qunit})": qs,
                                        "Outlet temperature (°F)": ts})
                    st.download_button("Download performance curve (CSV)", cur.to_csv(index=False).encode(),
                                       "choke_performance.csv")
        else:
            st.info("Press Calculate.")


def traverse_mode(col, kind, fluid, eqd, o, base, corr, p1, p2, surface):
    """Gradient traverse: pressure, temperature, holdup and velocities along the well at chosen rates."""
    hc = eqd["heat_capacities"]
    with col:
        st.markdown("**Rates**")
        dflt_r = "500, 1000, 2000, 3000" if kind == "oil" else "2, 5, 10, 20"
        rates = parse_values(st.text_input(f"Rates ({_qunit(kind)}), comma separated - up to 6", key=f"trav_rates_{kind}",
                                           **dflt(f"trav_rates_{kind}", value=dflt_r)))
        rates = [r for r in rates if r > 0][:6]
        st.caption("The traverse starts at the top node " + ("(the manifold) and works back through the surface "
                   "equipment, then " if surface is not None else "(the wellhead) and runs ") +
                   "down the tubing to the bottom node, with the inputs on the left.")
        go = st.button("Calculate traverse", type="primary", disabled=not rates)
    sig = (kind, tuple(sorted(base.items())), corr, p1, p2, tuple(rates), repr(surface), o.flow_type)
    if go:
        try:
            path = vlp_path(eqd, o.flow_type, ())
            out = []
            with st.spinner("Calculating traverses..."):
                for q in rates:
                    det, reg, prof = wsys.solution_details(path, fluid, kind, base, q, corr, p1, p2, hc, surface)
                    out.append((q, det, reg, prof))
            ss["trav_res"] = {"sig": sig, "res": out, "path": path}
        except (ValueError, KeyError, TypeError) as e:
            st.error(f"Traverse failed: {e}")
    tr = ss.get("trav_res")
    if not tr:
        return
    if tr["sig"] != sig:
        st.info("Inputs have changed since the last calculation - press Calculate traverse to update.")
    res, path = tr["res"], tr["path"]
    rows = []
    for q, det, reg, prof in res:
        r = {f"Rate ({_qunit(kind)})": q}
        if "Manifold pressure (psia)" in det:
            r["Manifold pressure (psia)"] = det["Manifold pressure (psia)"]
        r.update({"Wellhead pressure (psia)": det["Wellhead pressure (psia)"],
                  "Bottom node pressure (psia)": det["Bottom node pressure (psia)"],
                  "Wellhead temperature (°F)": det["Wellhead temperature (°F)"],
                  "dP gravity (psi)": det["dP gravity - production string (psi)"],
                  "dP friction (psi)": det["dP friction - production string (psi)"],
                  "Max. velocity / erosional": det["Max. velocity / erosional velocity"]})
        if "dP choke (psi)" in det:
            r["dP choke (psi)"] = det["dP choke (psi)"]
        rows.append(r)
    st.markdown("**Summary**")
    st.dataframe(pd.DataFrame(rows).style.format(precision=2), hide_index=True, width="stretch")
    fig, ax = plt.subplots(1, 4, figsize=(15, 5.2), sharey=True)
    for i, (q, det, reg, prof) in enumerate(res):
        c = wplt.SERIES[i % len(wplt.SERIES)]
        lab = f"{q:g} {_qunit(kind)}"
        ax[0].plot(prof["Pressure (psia)"], prof["TVD (ft)"], color=c, lw=1.6, label=lab)
        ax[1].plot(prof["Temperature (°F)"], prof["TVD (ft)"], color=c, lw=1.6, label=lab)
        ax[2].plot(prof["Liquid holdup"], prof["TVD (ft)"], color=c, lw=1.6, label=lab)
        ax[2].plot(prof["No-slip holdup"], prof["TVD (ft)"], color=c, lw=1.0, ls=":")
        ax[3].plot(prof["Mixture velocity (ft/s)"], prof["TVD (ft)"], color=c, lw=1.6, label=lab)
        ax[3].plot(prof["Erosional velocity (ft/s)"], prof["TVD (ft)"], color=c, lw=1.0, ls="--")
    ax[1].plot(path.t_form, path.tvd, color=wplt.INK2, ls="--", lw=1.1, label="formation")
    for a_, t_ in zip(ax, ("Pressure (psia)", "Temperature (°F)", "Liquid holdup (dotted: no-slip)",
                           "Mixture velocity (dashed: erosional), ft/s")):
        a_.set_xlabel(t_, fontsize=9)
        a_.grid(alpha=0.3)
    ax[0].set_ylabel("TVD (ft)")
    ax[0].invert_yaxis()
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Gradient traverse - {VLP_LABELS[corr]} (P1 {p1:.3f}, P2 {p2:.3f})", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    pick = st.selectbox("Profile for rate", list(range(len(res))), key=f"trav_pick_{len(res)}",
                        format_func=lambda i: f"{res[i][0]:g} {_qunit(kind)}")
    q, det, reg, prof = res[pick]
    surf_tab = det.get("_surface")
    if surf_tab is not None:
        st.markdown("**Surface equipment** (wellhead to manifold)")
        st.dataframe(surf_tab.style.format(precision=2), hide_index=True, width="stretch")
    if reg:
        st.caption("Restrictions: " + "; ".join(f"{k} - {v[0]}, dP {v[1]:.1f} psi" for k, v in reg.items()))
    st.markdown("**Tubing profile** (segment properties are given at the lower node of each segment)")
    st.dataframe(prof.style.format(precision=3, na_rep="—"), hide_index=True, width="stretch", height=380)
    allp = pd.concat([p_.assign(**{f"Rate ({_qunit(kind)})": q_}) for q_, _, _, p_ in res], ignore_index=True)
    st.download_button("Download all profiles (CSV)", allp.to_csv(index=False).encode(), "gradient_traverse.csv")


def _default_list(vals):
    return ", ".join(f"{v:.4g}" for v in vals)


def vfp_mode(col, kind, fluid, eqd, o, base, corr, p1, p2, surface):
    """Lift-curve tables for reservoir simulation: BHP over rate x THP x WFR x GFR."""
    hc = eqd["heat_capacities"]
    ax_ = wlc.AXES[kind]
    ipr_c = ss.get("ipr_curve")
    aof = float(ipr_c.aof) / (1000.0 if kind == "gas" else 1.0) if ipr_c is not None and ipr_c.fluid == kind else \
        (5000.0 if kind == "oil" else 50.0)
    with col:
        st.markdown("**Table axes**")
        c = st.columns(4)
        first = num(c[0], f"First rate ({_qunit(kind)})", f"vfp_first_{kind}", round(aof * 0.01, 3 if kind == "gas" else 0),
                    1e-6, 1e7)
        last = num(c[1], f"Last rate ({_qunit(kind)})", f"vfp_last_{kind}", round(aof, 3 if kind == "gas" else 0),
                   1e-6, 1e7)
        nrate = num(c[2], "Number of rates", f"vfp_n_{kind}", 15.0, 2.0, 40.0, 1.0, "%.0f")
        spacing = sel(c[3], "Spacing", ["Geometric", "Linear"], f"vfp_sp_{kind}")
        try:
            flo = list(wsys.generate_values(first, last, nrate, spacing))
        except ValueError as e:
            st.error(str(e))
            flo = []
        whp0 = float(base["whp"])
        thp = parse_values(st.text_input(
            "Top node pressures (psia)" + (" - at the manifold" if surface is not None else ""), key=f"vfp_thp_{kind}",
            **dflt(f"vfp_thp_{kind}", value=_default_list(np.round(whp0 * np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0]), 0)))))
        if kind == "oil":
            wfr = parse_values(st.text_input("Water cuts (%)", key=f"vfp_wfr_{kind}", **dflt(f"vfp_wfr_{kind}", value="0, 20, 40, 60, 80, 95")))
            g0 = float(base["gor"])
            gfr = parse_values(st.text_input("Total GORs (scf/STB)", key=f"vfp_gfr_{kind}", **dflt(
                f"vfp_gfr_{kind}", value=_default_list(np.round(g0 * np.array([0.75, 1.0, 1.5, 2.5, 4.0]), 0)))))
        else:
            w0 = float(base["wgr"])
            wfr = parse_values(st.text_input("Water gas ratios (STB/MMscf)", key=f"vfp_wfr_{kind}", **dflt(
                f"vfp_wfr_{kind}", value=_default_list(sorted({0.0, w0, max(w0 * 3, 10.0), max(w0 * 10, 50.0)})))))
            sg = float(base.get("sep_gor", 0.0))
            c0 = 1e6 / sg if sg > 0 else 0.0
            gfr = parse_values(st.text_input("Condensate gas ratios (STB/MMscf)", key=f"vfp_gfr_{kind}", **dflt(
                f"vfp_gfr_{kind}", value=_default_list(sorted({0.0, round(c0, 2), round(max(c0 * 2, 20.0), 2)})))))
        c2 = st.columns([1, 2])
        tno = int(num(c2[0], "Table number", "vfp_tno", 1.0, 1.0, 9999.0, 1.0, "%.0f"))
        wfr = [v for v in wfr if v >= 0]
        gfr = [v for v in gfr if v >= 0]
        thp = [v for v in thp if v > 0]
        npts = len(flo) * len(set(thp)) * len(set(wfr)) * len(set(gfr))
        c2[1].caption(f"{npts:,} points ({len(flo)} rates x {len(set(thp))} pressures x {len(set(wfr))} x "
                      f"{len(set(gfr))}).")
        if kind == "oil" and any(v >= 100 for v in wfr):
            st.error("Water cut must be below 100 %.")
            wfr = [v for v in wfr if v < 100]
        go = st.button("Generate lift curves", type="primary", disabled=not (flo and thp and wfr and gfr))
    sig = (kind, corr, p1, p2, tuple(flo), tuple(thp), tuple(wfr), tuple(gfr), repr(surface), o.flow_type, base["u"],
           tno)
    if go:
        try:
            path = vlp_path(eqd, o.flow_type, ())
            with st.spinner(f"Calculating {npts:,} traverses..."):
                axes, bhp, wh = wlc.vfp_table(path, fluid, kind, corr, p1, p2, float(base["u"]), flo, thp, wfr, gfr,
                                              hc, surface)
            ss["vfp_res"] = {"sig": sig, "axes": axes, "bhp": bhp, "wh": wh if surface is not None else None,
                             "datum": float(path.tvd[-1]), "md": float(path.md[-1]), "tno": tno,
                             "label": f"{VLP_LABELS[corr]} P1 {p1:.4f} P2 {p2:.4f}"}
        except (ValueError, KeyError, TypeError) as e:
            st.error(f"Lift curves failed: {e}")
    vr = ss.get("vfp_res")
    if not vr or vr["sig"][0] != kind:
        return
    if vr["sig"] != sig:
        st.info("Inputs have changed since the tables were generated - press Generate lift curves to update.")
    axes, bhp = vr["axes"], vr["bhp"]
    thp_a, wfr_a, gfr_a, flo_a = axes
    st.markdown(f"**Lift curves** - bottom node at {vr['datum']:,.1f} ft TVD ({vr['md']:,.0f} ft MD), "
                f"{vr['label']}")
    for w_ in wlc.vfp_checks(axes, bhp):
        st.warning(w_)
    c = st.columns(2)
    jw = c[0].selectbox(f"{ax_['wfr'][0]} ({ax_['wfr'][1]})", range(len(wfr_a)), format_func=lambda i: f"{wfr_a[i]:g}",
                        key=f"vfp_pw_{len(wfr_a)}")
    kg = c[1].selectbox(f"{ax_['gfr'][0]} ({ax_['gfr'][1]})", range(len(gfr_a)), format_func=lambda i: f"{gfr_a[i]:g}",
                        key=f"vfp_pg_{len(gfr_a)}")
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate(thp_a):
        ax.plot(flo_a, bhp[i, jw, kg], "-o", ms=3, lw=1.5, color=cmap(i / max(len(thp_a) - 1, 1)),
                label=f"THP {t:g} psia")
    ax.set_xlabel(f"{ax_['flo'][0]} ({ax_['flo'][1]})")
    ax.set_ylabel("Bottom node pressure (psia)")
    ax.set_title(f"Lift curves at {ax_['wfr'][0].lower()} {wfr_a[jw]:g} {ax_['wfr'][1]}, "
                 f"{ax_['gfr'][0].lower()} {gfr_a[kg]:g} {ax_['gfr'][1]}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    pc = st.columns([1.4, 1])
    pc[0].pyplot(fig, clear_figure=True)
    ecl = wlc.eclipse_vfpprod(kind, axes, bhp, vr["tno"], vr["datum"], ss.get("proj_name", ""))
    csv = wlc.vfp_csv(kind, axes, bhp, vr["wh"])
    with pc[1]:
        st.markdown("**Export**")
        st.download_button("Download Eclipse VFPPROD (.inc)", ecl.encode(), f"vfpprod_{vr['tno']}.inc",
                           width="stretch")
        st.download_button("Download table (CSV)", csv.to_csv(index=False).encode(), f"lift_curves_{vr['tno']}.csv",
                           width="stretch")
        st.caption("Eclipse FIELD units: " + ("FLO = liquid rate (STB/d), WFR = water cut (fraction), GFR = GOR "
                   "(Mscf/STB)" if kind == "oil" else "FLO = gas rate (Mscf/d), WFR = WGR (STB/Mscf), GFR = OGR "
                   "(STB/Mscf)") + "; THP and BHP in psia; datum = bottom node TVD; no artificial lift (ALQ 0)." +
                   (" THP is the manifold pressure (surface equipment included)." if vr["wh"] is not None else ""))
        with st.expander("Preview VFPPROD"):
            st.code("\n".join(ecl.splitlines()[:40]) + ("\n..." if len(ecl.splitlines()) > 40 else ""),
                    language=None)

with tabs[5]:
    st.header("Calculation")
    mode = radio(st, "Calculation", CALC_MODES, "calc_mode", horizontal=True, label_visibility="collapsed")
    SYS = mode == CALC_MODES[0]
    CHOKE = mode == CALC_MODES[3]
    st.subheader(mode)
    o = ss.get("options")
    eqd = ss.get("equipment", {})
    ipr_curve = ss.get("ipr_curve")
    blocker = None
    if o is None or o.errors:
        blocker = "Fix the configuration in the Options summary first."
    elif CHOKE:
        blocker = None
    elif o.is_injector:
        blocker = "System analysis for injectors is not built yet."
    elif not eqd.get("valid") or eqd.get("survey") is None or not len(eqd["downhole"][1]):
        blocker = "Complete the Equipment Data tab first."
    elif SYS and (ipr_curve is None or ipr_curve.fluid != ("gas" if o.is_gas else "oil")):
        blocker = "Calculate an IPR for this fluid type in the IPR Data tab first."
    if blocker:
        st.info(blocker)
    elif CHOKE:
        choke_mode(o)
    else:
        kind = "gas" if o.is_gas else "oil"
        gmd, gt = eqd["geothermal"]
        tests_ = ss.get("vlp_tests_" + kind)
        t0_ = None
        if tests_ is not None:
            tt_, _ = wvlp.clean_tests(tests_, kind)
            t0_ = tt_.iloc[0] if len(tt_) else None
        pr_ref = float(ipr_curve.pr) if ipr_curve is not None else float(val("ipr_pr", 5000.0))
        fluid, fluid_note = vlp_fluid(o, kind, max(pr_ref, 5000.0) * 1.15,
                                      float(np.min(gt)) - 10, float(np.max(gt)) + 10)
        if fluid is None:
            st.warning(fluid_note)
        else:
            ms = ss.get("vlp_matches", {}).get("ms", {})
            sel_c = ss.get("vlp_selected")
            matched = ms.get(sel_c) if sel_c else None
            left, right = st.columns([1, 1.5], gap="large")
            def _copy_test(row):
                put("sys_whp", float(row["thp"]))
                if kind == "oil":
                    put("sys_wc", float(row["wc"] * 100))
                    put("sys_gor", float(row["gor"]))
                else:
                    put("sys_wgr", float(row["wgr"]))
                    put("sys_sep_gor", float(round(1e6 / row["cgr"])) if row["cgr"] > 0 else 0.0)
            with left:
                st.markdown("**Input data**")
                if tests_ is not None and t0_ is not None:
                    tt_all, _ = wvlp.clean_tests(tests_, kind)
                    cc_ = st.columns([2, 1])
                    tsel_ = cc_[0].selectbox("Copy conditions from test", tt_all["label"].tolist(), key="sys_copy_t")
                    cc_[1].button("Copy", on_click=_copy_test,
                                  args=(tt_all[tt_all["label"] == tsel_].iloc[0],), width="stretch",
                                  help="Wellhead pressure and water/gas ratios from the selected VLP test.")
                s_items = surface_items(eqd)
                if s_items is not None:
                    top_node = sel(st, "Top node", TOP_NODES, "sys_top",
                                   help="Manifold: the calculation starts at the manifold pressure and works back "
                                        "through the flowline, fittings and choke to the wellhead. Wellhead: the "
                                        "surface equipment is ignored.")
                else:
                    top_node = "Wellhead"
                use_surface = top_node == "Manifold"
                if use_surface:
                    whp = num(st, "Top node (manifold) pressure (psia)", "sys_pman",
                              float(round(t0_["thp"] * 0.5)) if t0_ is not None else
                              (150.0 if kind == "oil" else 800.0), 14.7, 2e4)
                else:
                    whp = num(st, "Top node (wellhead) pressure (psia)", "sys_whp",
                              float(t0_["thp"]) if t0_ is not None else (250.0 if kind == "oil" else 1500.0), 14.7,
                              2e4)
                base = {"whp": whp, "u": float(val("eq_u", eqd.get("u_geo", 8.0)))}
                if kind == "oil":
                    base["wc"] = num(st, "Water cut (%)", "sys_wc",
                                     float(t0_["wc"] * 100) if t0_ is not None else float(val("ipr_wc", 0.0)),
                                     0.0, 99.0)
                    m_ = blackoil_model(o) if o.method != "eos" else None
                    base["gor"] = num(st, "Total GOR (scf/STB)", "sys_gor",
                                      float(t0_["gor"]) if t0_ is not None else
                                      (float(m_.inp.rsb) if m_ else 500.0), 0.0, 1e6)
                else:
                    base["wgr"] = num(st, "Water gas ratio (STB/MMscf)", "sys_wgr",
                                      float(t0_["wgr"]) if t0_ is not None else 0.0, 0.0, 1e5)
                    cg_ = float(t0_["cgr"]) if t0_ is not None and t0_["cgr"] > 0 else 0.0
                    base["sep_gor"] = num(st, "Separator GOR (scf/STB)", "sys_sep_gor",
                                          round(1e6 / cg_) if cg_ > 0 else 0.0, 0.0, 1e8,
                                          help="Separator gas per barrel of condensate; 0 = dry gas.")
                surface = None
                if use_surface:
                    s_corr = sel(st, "Surface equipment correlation", SURFACE_CORRELATIONS, "sys_scorr",
                                 format_func=VLP_LABELS.get,
                                 help="Pressure loss in the flowline and fittings (fittings as equivalent pipe length).")
                    choke_method, cd, friction = "Sachdeva", 0.75, 1.0
                    smat = ss.get("surf_matched")
                    use_sm = False
                    if smat is not None:
                        use_sm = st.toggle("Use matched surface parameters", value=bool(val("sys_use_smatch", True)),
                                           key=wkey("sys_use_smatch"),
                                           help="Cd and the flowline friction multiplier from Match Surface (VLP "
                                                "tab); turn off to enter them by hand.")
                        ss["_v_sys_use_smatch"] = use_sm
                    if use_sm:
                        choke_method, cd, friction = "Sachdeva", float(smat["cd"]), float(smat["friction"])
                        c3_ = st.columns(3)
                        c3_[0].text_input("Choke method", "Sachdeva", disabled=True)
                        c3_[1].text_input("Discharge coeff.", f"{cd:.3f}", disabled=True)
                        c3_[2].text_input("Friction multiplier", f"{friction:.3f}", disabled=True)
                        st.caption("Surface parameters from Match Surface.")
                    elif wsurf.has_choke(s_items):
                        cc_ = st.columns([2, 1])
                        choke_method = sel(cc_[0], "Choke method", wsurf.CHOKE_METHODS, "sys_choke",
                                           help="Sachdeva et al. (1986): critical and subcritical two-phase flow. "
                                                "Gilbert, Ros, Baxendell and Achong: oil wells, critical flow only - "
                                                "subcritical points fall back to Sachdeva.")
                        cd = num(cc_[1], "Discharge coeff.", "sys_cd", 0.75, 0.3, 1.2, None, "%.3f",
                                 help="Sachdeva discharge coefficient (0.75 typical).")
                        if choke_method != "Sachdeva" and kind != "oil":
                            st.caption(f"{choke_method} is for oil wells - Sachdeva will be used.")
                    if not use_sm:
                        friction = num(st, "Flowline friction multiplier", "sys_ffric", 1.0, 0.2, 5.0, None, "%.3f",
                                       help="Multiplies the friction loss in the flowline and fittings (1 = as "
                                            "calculated). Match Surface in the VLP tab can regress it from tests.")
                    surface = wsys.SurfaceSetup(s_items, float(eqd.get("t_surround", 60.0)),
                                                float(eqd.get("u_surface", 8.0)), s_corr, choke_method, cd, friction)
                else:
                    st.text_input("Surface equipment correlation",
                                  "Not used - the top node is the wellhead" if s_items is not None else
                                  "Not used - no surface equipment entered", disabled=True)
                corr_opts = [c_ for c_ in wvlp.VLP_CORRELATIONS if c_ not in wvlp.LOWER_BOUNDS]
                corr = sel(st, "Vertical lift correlation", corr_opts, "sys_corr",
                           sel_c if sel_c in corr_opts else "hagedorn_brown", format_func=VLP_LABELS.get)
                m_fit = ms.get(corr)
                use_fit = flag(st.toggle, "Use matched parameters", "sys_use_fit",
                                    disabled=m_fit is None,
                                    help="Parameter 1 and 2 from Match VLP; turn off to enter them by hand.")
                c2 = st.columns(2)
                if m_fit is not None and use_fit:
                    p1, p2 = float(m_fit.p1), float(m_fit.p2)
                    c2[0].text_input("Parameter 1 (gravity)", f"{p1:.4f}", disabled=True)
                    c2[1].text_input("Parameter 2 (friction)", f"{p2:.4f}", disabled=True)
                    st.caption("Parameters from Match VLP.")
                else:
                    p1 = num(c2[0], "Parameter 1 (gravity)", f"sys_p1_{corr}", 1.0, 0.3, 3.0, None, "%.4f")
                    p2 = num(c2[1], "Parameter 2 (friction)", f"sys_p2_{corr}", 1.0, 0.01, 10.0, None, "%.4f")
                    if m_fit is None:
                        st.caption("Not matched - run Match VLP in the VLP tab to calibrate this correlation.")
                if SYS:
                    node_lab = sel(st, "Solution node", ["Bottom node", "Top node"], "sys_node",
                                   help="Bottom node: IPR against VLP at the bottom of the downhole equipment - the "
                                        "solution pressure is the flowing bottom-hole pressure. Top node: inflow (IPR, "
                                        "tubing" + (", surface equipment" if use_surface else "") + ") against the "
                                        "top-node pressure at the " + ("manifold" if use_surface else "wellhead") +
                                        ". The solution rate is the same at either node; only the curves change.")
                    node = "top" if node_lab == "Top node" else "bottom"
                    method = sel(st, "Rate method", RATE_METHODS, "sys_rate_method")
                    user_rates = None
                    if method == "User selected":
                        user_rates = parse_values(st.text_input(
                            f"Rates ({'STB/d' if kind == 'oil' else 'MMscf/d'}), comma separated", key="sys_rates",
                            **dflt("sys_rates", value="100, 500, 1000, 2000, 3000" if kind == "oil" else "1, 5, 10, 20, 30")))
                    lhi = sel(st, "Left-hand intersection", ["DisAllow", "Allow"], "sys_lhi")
                    flags = sel(st, "Stability flags", ["Yes", "No"], "sys_flags")
                    st.caption(f"PVT: {fluid_note}. IPR: {ss.get('ipr_meta', {}).get('model', '')} "
                               f"(Pr {ipr_curve.pr:,.0f} psia). U = {base['u']:.4g} BTU/h/ft2/F.")
                else:
                    st.caption(f"PVT: {fluid_note}. U = {base['u']:.4g} BTU/h/ft2/F.")

            if SYS:
                with right:
                    st.markdown("**Sensitivity**")
                    sens_on = st.toggle("Sensitivity", key="sys_sens_on",
                                        help="Choose up to three variables; every combination of their values is calculated.")
                    var_defs = dict(wsys.OIL_VARIABLES if kind == "oil" else wsys.GAS_VARIABLES)
                    if ipr_arg("pr") is None:
                        var_defs.pop("pr", None)
                    if ipr_arg("j") is None:
                        var_defs.pop("j", None)
                    if ipr_arg("skin") is None:
                        var_defs.pop("skin", None)
                    if o.flow_type != "tubing":
                        var_defs.pop("tubing_id", None)
                    if surface is None or not wsurf.has_choke(surface.items):
                        var_defs.pop("choke", None)
                    if surface is not None:
                        var_defs["whp"] = "Manifold pressure (psia)"
                    sens = {}
                    if sens_on:
                        chosen = st.multiselect("Variables (up to 3)", list(var_defs), format_func=var_defs.get,
                                                max_selections=3, key="sys_vars")
                        tub_id = float(eqd["downhole"][1].query("Type == 'Tubing'")["Flow ID (in)"].iloc[-1])
                        cur = {**base, "pr": ipr_arg("pr"), "j": ipr_arg("j"), "skin": ipr_arg("skin"),
                               "tubing_id": tub_id}
                        if surface is not None and wsurf.has_choke(surface.items):
                            cur["choke"] = round(next(i.d_in for i in surface.items if i.kind == "choke") * 64, 2)
                        for v_ in chosen:
                            with st.container(border=True):
                                st.markdown(f"**{var_defs[v_]}** - current {cur.get(v_) if cur.get(v_) is not None else '—'}")
                                how = st.radio("Values", ["Enter values", "Generate values"], horizontal=True,
                                               key=f"sys_how_{v_}")
                                try:
                                    if how == "Enter values":
                                        vals = parse_values(st.text_input("Values, comma separated (up to 10)",
                                                                          key=f"sys_vals_{v_}"))
                                    else:
                                        cc = st.columns(4)
                                        c0 = float(cur.get(v_) or 1.0)
                                        first = num(cc[0], "First", f"sys_first_{v_}", round(c0 * 0.5, 4))
                                        last = num(cc[1], "Last", f"sys_last_{v_}", round(c0 * 1.5 if c0 else 1.0, 4))
                                        nval = num(cc[2], "Number", f"sys_n_{v_}", 5.0, 1.0, 10.0, 1.0, "%.0f")
                                        spacing = sel(cc[3], "Spacing", ["Linear", "Geometric"], f"sys_sp_{v_}")
                                        vals = list(wsys.generate_values(first, last, nval, spacing))
                                    if len(vals) > 10:
                                        st.warning("Only the first 10 values are used.")
                                        vals = vals[:10]
                                    if vals:
                                        st.caption("Values: " + ", ".join(f"{x:.4g}" for x in vals))
                                        sens[v_] = vals
                                    else:
                                        st.caption("Enter at least one value.")
                                except ValueError as e:
                                    st.error(str(e))
                        n_case = int(np.prod([len(v) for v in sens.values()])) if sens else 1
                        st.caption(f"{n_case} case(s) will be calculated.")

                go = st.button("Calculate", type="primary")
                sig = (kind, tuple(sorted(base.items())), corr, p1, p2, method, tuple(user_rates or []), lhi,
                       tuple((k_, tuple(v_)) for k_, v_ in sens.items()), fluid_note, ipr_curve.pr,
                       float(ipr_curve.aof), repr(surface), node)
                if go:
                    try:
                        combos = wsys.combinations(sens)
                        path_cache = {}

                        def path_for(tid):
                            if tid not in path_cache:
                                top_, secs_, pts_ = eqd["downhole"]
                                s2 = secs_.copy()
                                if tid is not None:
                                    s2.loc[s2["Type"] == "Tubing", "Flow ID (in)"] = float(tid)
                                smd_, stvd_ = eqd["survey"]
                                path_cache[tid] = wvlp.build_flow_path(top_, s2, pts_, smd_, stvd_, gmd, gt, o.flow_type)
                            return path_cache[tid]

                        def ipr_for(ch):
                            return rebuild_ipr(ch) if ch else ipr_curve
                        with st.spinner(f"Calculating {len(combos)} case(s)..."):
                            res, paths_ = wsys.solve_cases(path_for, fluid, kind, base, combos, ipr_for, corr, p1, p2,
                                                           eqd["heat_capacities"], method, 25, user_rates,
                                                           allow_left=(lhi == "Allow"), surface=surface, node=node)
                        for r_, c_ in zip(res, combos):
                            r_.case_label = c_
                        ss["sys_res"] = {"sig": sig, "res": res, "paths": paths_, "vars": list(sens), "kind": kind,
                                         "corr": corr, "p1": p1, "p2": p2, "labels": var_defs, "surface": surface}
                    except (ValueError, TypeError, KeyError) as e:
                        st.error(f"Calculation failed: {e}")
                out = ss.get("sys_res")
                if out and out["kind"] == kind:
                    if out["sig"] != sig:
                        st.info("Inputs have changed since the last calculation - press Calculate to update.")
                    res = out["res"]
                    qcol = "Liquid rate (STB/d)" if kind == "oil" else "Gas rate (MMscf/d)"
                    if len(res) > 1:
                        rows_ = []
                        for r_ in res:
                            d_ = {out["labels"][k_]: v_ for k_, v_ in r_.case_label.items()}
                            if r_.q is not None:
                                d_.update(wsys.surface_rates(kind, r_.params, r_.q))
                                d_.update({"Bottom node pressure (psia)": r_.pwf})
                                if r_.pwh_q is not None:
                                    d_["Wellhead pressure (psia)"] = r_.pwh_q
                                d_.update({"Wellhead temperature (°F)": r_.wht_q, "Stable": "Yes" if r_.stable else "No"})
                            d_["Note"] = r_.note
                            rows_.append(d_)
                        tab_ = pd.DataFrame(rows_)
                        for c_ in ("Stable", "Note"):
                            if c_ in tab_:
                                tab_[c_] = tab_[c_].fillna("—")
                        st.markdown(f"**Sensitivity results** - {len(res)} cases")
                        st.dataframe(tab_.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
                        st.download_button("Download sensitivity results (CSV)", tab_.to_csv(index=False).encode(),
                                           "system_sensitivity.csv")
                        cc = st.columns([1.2, 1])
                        with cc[0]:
                            full_s_ = st.toggle("Show the full rate range (to the AOF)", key="sys_plot_full_sens")
                            st.pyplot(sys_plot(res, kind, full=full_s_), clear_figure=True)
                            if len(res) > 12:
                                st.caption("IPR/VLP plot shows the first 12 cases.")
                        with cc[1]:
                            names_ = [out["labels"][k_] for k_ in out["vars"]]
                            if qcol in tab_:
                                st.pyplot(sens_plot(tab_, names_, {n_: n_ for n_ in names_}, qcol), clear_figure=True)
                        pick = st.selectbox("Case details", range(len(res)), key=f"sys_case_{len(res)}",
                                            format_func=lambda i: f"Case {i + 1}: " + ", ".join(
                                                f"{out['labels'][k_].split(' (')[0]} = {v_:g}"
                                                for k_, v_ in res[i].case_label.items()))
                        r = res[pick]
                    else:
                        r = res[0]
                    st.markdown("**Results**")
                    if r.note:
                        (st.warning if r.q is None or not r.stable else st.info)(r.note)
                    if r.q is not None:
                        if r.node == "top" and r.pwh_q is not None and (r.top_in_q is None or not np.isfinite(r.top_in_q)
                                                                         or r.top_in_q > float(r.params["whp"]) + 1.0):
                            st.caption("The surface choke is critical at the solution: the inflow at the manifold drops "
                                       "vertically at this rate, so the manifold pressure does not affect it.")
                        if flags == "Yes" and not r.stable:
                            st.error("Stability flag: the solution lies left of the VLP minimum - flow is unstable "
                                     "(heading / liquid loading).")
                    cc = st.columns([1.25, 1])
                    with cc[0]:
                        if len(res) == 1:
                            full_ = st.toggle("Show the full rate range (to the AOF)", key="sys_plot_full",
                                              help="Off: the plot zooms to the solution - a high-PI well's IPR "
                                                   "runs far beyond what the tubing can lift. The curve table "
                                                   "always lists every rate.")
                            st.pyplot(sys_plot(res, kind, full=full_), clear_figure=True)
                        st.markdown("**IPR / VLP curve data**")
                        rt_ = wsys.rate_table(kind, r)
                        rt_show_ = rt_.copy()
                        for c_ in rt_show_.columns[1:]:
                            rt_show_[c_] = [f"{v_:,.2f}" if np.isfinite(v_) else "—" for v_ in rt_show_[c_].astype(float)]
                        st.dataframe(rt_show_, hide_index=True, width="stretch", height=330)
                        if r.node == "top":
                            st.caption("Inflow pressure at top node: — where the well cannot lift the rate to the top "
                                       "node or, with a critical choke, where the bean cannot pass it.")
                    with cc[1]:
                        if r.q is not None:
                            det_, reg_, prof_ = wsys.solution_details(out["paths"][r.params.get("tubing_id")], fluid, kind,
                                                                      r.params, r.q, out["corr"], out["p1"], out["p2"],
                                                                      eqd["heat_capacities"], surface=out.get("surface"))
                            surf_tab_ = det_.pop("_surface", None)
                            st.markdown("**Solution**")
                            st.dataframe(solution_panel(kind, r, det_, reg_), hide_index=True, width="stretch",
                                         height=560)
                            if surf_tab_ is not None:
                                st.markdown("**Surface equipment at the solution** (wellhead to manifold)")
                                st.dataframe(surf_tab_.style.format(precision=2), hide_index=True, width="stretch")
                            with st.expander("Pressure and temperature profile at the solution"):
                                st.dataframe(prof_.style.format(precision=3), hide_index=True, width="stretch")
                        st.download_button("Download IPR/VLP table (CSV)", rt_.to_csv(index=False).encode(),
                                           "system_analysis.csv")
            elif mode == CALC_MODES[1]:
                traverse_mode(right, kind, fluid, eqd, o, base, corr, p1, p2, surface)
            else:
                vfp_mode(right, kind, fluid, eqd, o, base, corr, p1, p2, surface)


# ================================================================ tab 7: design (continuous gas lift)
def gl_design_plot(res, gas_label=""):
    """PROSPER-style gas-lift design plot: pressure against true vertical depth."""
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    inp = res.inp
    tvd = res.tvd
    ax.plot(res.pt, tvd, color=wplt.SERIES[0], lw=2.0, label="Flowing tubing pressure (design)")
    ax.plot(res.pc_op, tvd, color=wplt.SERIES[1], lw=2.0, label=f"Casing pressure (operating, {res.pso:,.0f} psia)")
    ax.plot(res.pc_ko, tvd, color=wplt.SERIES[1], lw=1.2, ls="--", label=f"Casing pressure (kick-off, {inp.p_ko:,.0f} psia)")
    if inp.spacing == wgl.SPACING[1]:
        ax.plot(res.transfer, tvd, color=wplt.SERIES[3], lw=1.2, ls=":", label="Spacing (transfer) line")
    # unloading (load-fluid) lines
    p0, t0 = inp.top_p_unload, float(tvd[0])
    first = True
    for v in res.valves:
        if v.kind != "Valve" and v is not res.valves[-1]:
            continue
        ax.plot([p0, p0 + inp.load_grad * (v.tvd - t0)], [t0, v.tvd], color=wplt.SERIES[2], lw=1.0,
                label="Unloading gradient (load fluid)" if first else None)
        first = False
        p0, t0 = v.p_transfer, v.tvd
    xmax = max(float(np.nanmax(res.pc_ko)), float(np.nanmax(res.pt))) * 1.08
    for v in res.valves:
        ax.axhline(v.tvd, color=wplt.INK2, lw=0.6, ls="-", alpha=0.5)
        ax.scatter([v.pc], [v.tvd], marker="s" if v.kind == "Orifice" else ">", s=46, zorder=6,
                   color="#2ca02c" if v.kind == "Orifice" else wplt.INK)
        ax.text(xmax * 0.995, v.tvd, f"{v.kind} {v.n}  {v.md:,.0f} ft MD", ha="right", va="bottom", fontsize=7.5)
    ax.set_xlim(0, xmax)
    ax.set_ylim(float(tvd[-1]) * 1.02, 0)
    ax.set_xlabel("Pressure (psia)")
    ax.set_ylabel("True vertical depth (ft)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title(f"Gas lift design: {res.q:,.1f} STB/d liquid, {res.qgi:.4g} MMscf/d lift gas, "
                 f"injection at {res.inj_md:,.0f} ft MD", fontsize=9.5)
    fig.tight_layout()
    return fig


def gl_perf_plot(perf, res):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    g = perf["Gas injection rate (MMscf/d)"]
    ax.plot(g, perf["Oil rate (STB/d)"], "-o", ms=3.5, color=wplt.SERIES[0], label="Oil rate")
    ax.plot(g, perf["Liquid rate (STB/d)"], "-", lw=1.2, color=wplt.SERIES[2], label="Liquid rate")
    ax.axvline(res.inp.qgi_max, color=wplt.SERIES[1], ls="--", lw=1.0, label="Maximum gas available")
    ax.scatter([res.qgi], [res.q * (1 - res.inp.wc)], s=70, marker="s", color="#2ca02c", edgecolor=wplt.INK, zorder=6,
               label="Design point")
    if perf["Revenue ($/d)"].notna().any() and res.inp.design_rate == wgl.DESIGN_RATE[2]:
        ax2 = ax.twinx()
        ax2.plot(g, perf["Revenue ($/d)"], color=wplt.SERIES[3], lw=1.2, label="Revenue")
        ax2.set_ylabel("Revenue ($/d)")
    ax.set_xlabel("Gas injection rate (MMscf/d)")
    ax.set_ylabel("Rate (STB/d)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Gas lift performance curve", fontsize=10)
    fig.tight_layout()
    return fig


GL_SHOW_COLS = ["Valve", "Type", "Measured depth (ft)", "True vertical depth (ft)", "Tubing pressure (psia)",
                "Casing pressure (psia)", "Transfer pressure (psia)", "Temperature at valve (°F)",
                "Gas lift rate (Mscf/d)", "Port size (64ths in)", "R value", "Valve opening pressure (psia)",
                "Valve closing pressure (psia)", "Dome pressure at 60 °F (psig)", "Test rack opening pressure (psig)",
                "Opening CHP (psia)", "Closing CHP (psia)", "dP across valve (psi)", "Unloadable gradient (psi/ft)",
                "Required gas (Mscf/d)", "Note"]


def gaslift_tab(o, eqd, ipr_curve):
    """Design | Gas lift (continuous), new well - after PROSPER."""
    st.caption("Continuous gas lift design for a new well, after PROSPER Design | Gaslift | New Well and API RP 11V6: "
               "the design rate (entered, or the maximum of the performance curve), the operating injection point, "
               "unloading valve spacing with casing-sensitive (IPO) valves, port sizes (Thornhill-Craver) and valve "
               "settings (dome and test rack pressures), with the Asheim (1988) stability check.")
    gmd, gt = eqd["geothermal"]
    pr_ref = float(ipr_curve.pr)
    fluid, fluid_note = vlp_fluid(o, "oil", max(pr_ref, 5000.0) * 1.15, float(np.min(gt)) - 10, float(np.max(gt)) + 10)
    if fluid is None:
        st.warning(fluid_note)
        return
    top_, secs_, pts_ = eqd["downhole"]
    tub = secs_[secs_["Type"] == "Tubing"]
    bottom_tub = float(tub["MD bottom (ft)"].max()) if len(tub) else float(secs_["MD bottom (ft)"].max())
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        st.markdown("**Design options**")
        c = st.columns(2)
        design_rate = sel(c[0], "Design rate", list(wgl.DESIGN_RATE), "gl_rate_mode", wgl.DESIGN_RATE[1],
                          help="Entered by user: the rate and lift gas are known. Max production: the maximum oil rate "
                               "of the performance curve within the gas available. Max revenue: the maximum of the "
                               "revenue function (oil and gas sales less water handling and lift-gas cost).")
        valve_name = sel(c[1], "Valve type", list(wgl.VALVES), "gl_valve",
                         help="Casing-sensitive, nitrogen-charged IPO bellows valve. R = port area / bellows area; "
                              "ports up to the maximum for the valve size.")
        c = st.columns(2)
        inj_point = sel(c[0], "Injection point", list(wgl.INJECTION_POINT), "gl_injpt",
                        help="The final injection point is an orifice (no setting pressures) or an IPO valve.")
        spacing = sel(c[1], "Valve spacing method", list(wgl.SPACING), "gl_spacing",
                      help="Normal: strict gradient spacing on the design flowing gradient. Spacing line: transfer "
                           "pressures on a straight line from Pwh + SF (Pinj - Pwh) at surface to the flowing "
                           "pressure at the injection depth; SF 0.20 / 0.25 / 0.30 for PI < 2 / 2-20 / > 20.")
        c = st.columns(2)
        orif = sel(c[0], "Orifice sizing", list(wgl.ORIFICE_SIZING), "gl_orif",
                   help="Calculated dP: the orifice that passes the design gas at the desired dP (exact size). Min dP: "
                        "the next standard size up, so the dP is below the desired dP.")
        c2 = c[1].container()
        dome_hp = flag(c2.checkbox, "Dome correction > 1200 psig (Winkler-Eads)", "gl_dome_hp", True,
                       help="Yes: API Ct below 1200 psig and Winkler & Eads (1989) above. No: API Ct at all pressures.")
        c = st.columns(2)
        check_ipr = flag(c[0].checkbox, "Check rate conformance with IPR", "gl_chk_ipr", True)
        use_ipr_unl = flag(c[1].checkbox, "Use IPR for unloading", "gl_ipr_unl", True,
                           help="Yes: the unloading gas at each valve is the rate giving the minimum tubing pressure "
                                "with the liquid rate from the IPR. No: the design rate is assumed; the minimum "
                                "transfer dP % lowers the gas needed.")
        spread = flag(st.checkbox, "Casing drop per valve at least the valve spread R (Pvo - Pt)", "gl_spread", True,
                      help="PROSPER 'All valves Pvo = gas pressure' (recommended for new wells): the casing pressure "
                           "drop to close each valve is the larger of the entered drop and the calculated spread. "
                           "Off: the entered drop only (API RP 11V6 examples).")
        st.markdown("**Input parameters**")
        c = st.columns(2)
        q_design = None
        if design_rate == wgl.DESIGN_RATE[0]:
            q_design = num(c[0], "Design liquid rate (STB/d)", "gl_q", 1000.0, 1.0, 1e6)
        q_max = num(c[1] if q_design is not None else c[0], "Maximum liquid rate (STB/d)", "gl_qmax",
                    float(round(ipr_curve.aof)), 1.0, 1e6)
        c = st.columns(2)
        qgi_max = num(c[0], "Maximum gas available (MMscf/d)", "gl_qgi", 1.0, 0.001, 100.0, None, "%.4f",
                      help="With 'Entered by user' this is the design lift-gas rate.")
        qgi_unl = num(c[1], "Maximum gas during unloading (MMscf/d)", "gl_qgi_unl", float(qgi_max), 0.001, 100.0, None,
                      "%.4f")
        c = st.columns(2)
        top_p = num(c[0], "Flowing top node pressure (psia)", "gl_pwh", float(val("sys_whp", 200.0)), 14.7, 1e4)
        top_pu = num(c[1], "Unloading top node pressure (psia)", "gl_pwh_unl", float(top_p), 14.7, 1e4)
        c = st.columns(2)
        p_inj = num(c[0], "Operating injection pressure (psia)", "gl_pinj", 1200.0, 50.0, 2e4)
        p_ko = num(c[1], "Kick-off injection pressure (psia)", "gl_pko", float(p_inj), 50.0, 2e4)
        c = st.columns(2)
        dp_valve = num(c[0], "Desired dP across valve (psi)", "gl_dpv", 100.0, 1.0, 2000.0,
                       help="Usually 100-200 psi.")
        max_md = num(c[1], "Maximum depth of injection (ft MD)", "gl_maxmd", float(round(bottom_tub - 30.0)), 100.0,
                     float(secs_["MD bottom (ft)"].max()), help="Above the production packer.")
        c = st.columns(2)
        gor = num(c[0], "Total GOR (scf/STB)", "gl_gor", float(val("sys_gor", 300.0)), 0.0, 1e6,
                  help="Formation gas (solution and free); excludes the lift gas.")
        wc = num(c[1], "Water cut (%)", "gl_wc", float(val("sys_wc", 0.0)), 0.0, 99.0)
        c = st.columns(2)
        gs = num(c[0], "Static gradient of load fluid (psi/ft)", "gl_gs", 0.465, 0.1, 1.0, None, "%.3f",
                 help="The heavier of the kill fluid and the packer fluid (0.465 psi/ft = 8.94 ppg).")
        dpc = num(c[1], "Min CHP decrease per valve (psi)", "gl_dpc", 20.0, 0.0, 500.0)
        c = st.columns(2)
        margin_top = num(c[0], "Safety margin, first valve (psi)", "gl_m1", 50.0, 0.0, 500.0,
                         help="The first valve balances the load fluid against the kick-off casing pressure less this.")
        margin = num(c[1], "Transfer safety margin, deeper valves (psi)", "gl_m2", 20.0, 0.0, 500.0,
                     help="API RP 11V6: 20-50 psi.")
        c = st.columns(2)
        safety_last = num(c[0], "Safety for closure of last unloading valve (psi)", "gl_slast", 0.0, 0.0, 500.0)
        min_sp = num(c[1], "Minimum valve spacing (ft)", "gl_minsp", 250.0, 10.0, 5000.0,
                     help="API RP 11V6: 90 ft (high PI, good data) to 500 ft (low PI, poor data).")
        c = st.columns(3)
        min_tr = num(c[0], "Minimum transfer dP (%)", "gl_mintr", 0.0, 0.0, 100.0, disabled=use_ipr_unl,
                     help="Only with 'Use IPR for unloading' off: 5-25 % typical.")
        tc_v = num(c[1], "T-C de-rating, valves (%)", "gl_tcv", 0.0, 0.0, 90.0)
        tc_o = num(c[2], "T-C de-rating, orifice (%)", "gl_tco", 0.0, 0.0, 90.0)
        econ = wgl.Economics()
        if design_rate == wgl.DESIGN_RATE[2]:
            st.markdown("**Economics**")
            c = st.columns(4)
            econ = wgl.Economics(num(c[0], "Oil price ($/STB)", "gl_po", 70.0, 0.0, 1e4),
                                 num(c[1], "Gas price ($/Mscf)", "gl_pg", 3.0, 0.0, 1e4),
                                 num(c[2], "Water cost ($/STB)", "gl_cw", 1.0, 0.0, 1e4),
                                 num(c[3], "Lift gas cost ($/Mscf)", "gl_cg", 0.5, 0.0, 1e4))
        st.markdown("**Lift gas**")
        c = st.columns(5)
        lg = wgl.LiftGas(num(c[0], "Gravity", "gl_sg", 0.70, 0.55, 2.0, None, "%.3f"),
                         num(c[1], "CO2 (%)", "gl_co2", 0.0, 0.0, 100.0), num(c[2], "N2 (%)", "gl_n2", 0.0, 0.0, 100.0),
                         num(c[3], "H2S (%)", "gl_h2s", 0.0, 0.0, 100.0),
                         num(c[4], "Surface T (°F)", "gl_tg", 80.0, 0.0, 300.0))
        st.markdown("**Tubing correlation**")
        ms = ss.get("vlp_matches", {}).get("ms", {})
        sel_c = ss.get("vlp_selected")
        corr_opts = [c_ for c_ in wvlp.VLP_CORRELATIONS if c_ not in wvlp.LOWER_BOUNDS]
        corr = sel(st, "Vertical lift correlation", corr_opts, "gl_corr",
                   sel_c if sel_c in corr_opts else "hagedorn_brown", format_func=VLP_LABELS.get)
        m_fit = ms.get(corr)
        use_fit = flag(st.toggle, "Use matched parameters", "gl_use_fit", disabled=m_fit is None)
        c2 = st.columns(2)
        if m_fit is not None and use_fit:
            p1, p2 = float(m_fit.p1), float(m_fit.p2)
            c2[0].text_input("Parameter 1 (gravity)", f"{p1:.4f}", disabled=True, key="gl_p1_show")
            c2[1].text_input("Parameter 2 (friction)", f"{p2:.4f}", disabled=True, key="gl_p2_show")
        else:
            p1 = num(c2[0], "Parameter 1 (gravity)", f"gl_p1_{corr}", 1.0, 0.3, 3.0, None, "%.4f")
            p2 = num(c2[1], "Parameter 2 (friction)", f"gl_p2_{corr}", 1.0, 0.01, 10.0, None, "%.4f")
        u = float(val("eq_u", eqd.get("u_geo", 8.0)))
        st.caption(f"PVT: {fluid_note}. IPR: {ss.get('ipr_meta', {}).get('model', '')} (Pr {ipr_curve.pr:,.0f} psia). "
                   f"U = {u:.4g} BTU/h/ft2/F.")
        go = st.button("Design", type="primary", key="gl_go")
    inp = wgl.DesignInput(design_rate=design_rate, injection_point=inj_point, spacing=spacing, dome_high_p=dome_hp,
                          check_ipr=check_ipr, use_ipr_unloading=use_ipr_unl, orifice_sizing=orif,
                          q_design=q_design or 1000.0, q_liquid_max=q_max, qgi_max=qgi_max, qgi_unload_max=qgi_unl,
                          top_p=top_p, top_p_unload=top_pu, p_inj=p_inj, p_ko=p_ko, dp_valve=dp_valve,
                          max_md=max_md, gor=gor, wc=wc / 100.0, load_grad=gs, min_transfer_pct=min_tr,
                          safety_last=safety_last, dp_close=dpc, dp_close_spread=spread, tc_derate_valve=tc_v,
                          tc_derate_orifice=tc_o, min_spacing=min_sp, margin_top=margin_top, margin=margin,
                          valve=valve_name, economics=econ)
    sig = (repr(inp), repr(lg), corr, p1, p2, u, fluid_note, float(ipr_curve.pr), float(ipr_curve.aof))
    with right:
        if go:
            path = vlp_path(eqd, "tubing", ())
            msg = st.empty()
            try:
                with st.spinner("Designing..."):
                    res = wgl.design(path, fluid, ipr_curve, lg, inp, corr, p1, p2, u, eqd["heat_capacities"],
                                     progress=lambda m: msg.caption(m))
                vc = wgl.annulus_volume(secs_, float(top_), res.inj_md)
                stab = wgl.design_stability(res, path, fluid, lg, ipr_curve, vc) if np.isfinite(vc) else None
                ss["gl_res"] = {"sig": sig, "res": res, "stab": stab, "err": None}
            except (ValueError, RuntimeError, ZeroDivisionError) as e:
                ss["gl_res"] = {"sig": sig, "res": None, "stab": None, "err": str(e)}
            msg.empty()
        out = ss.get("gl_res")
        if out is None:
            st.info("Set the design options and press Design.")
            return
        if out["err"]:
            st.error(f"Design failed: {out['err']}")
            return
        if out["sig"] != sig:
            st.info("Inputs have changed since the last design - press Design to update.")
        res, stab = out["res"], out["stab"]
        st.markdown("**Design results**")
        n_unl = sum(v.kind == "Valve" for v in res.valves[:-1])
        summ = pd.DataFrame([
            ("Design liquid rate", f"{res.q:,.1f}", "STB/d"),
            ("Design oil rate", f"{res.q * (1 - res.inp.wc):,.1f}", "STB/d"),
            ("Lift gas injected", f"{res.qgi:.4f}", "MMscf/d"),
            ("Injection GLR", f"{res.qgi * 1e6 / max(res.q, 1e-9):,.0f}", "scf/STB"),
            ("Bottom node pressure", f"{res.pwf:,.1f}", "psia"),
            ("Injection point", f"{res.inj_md:,.1f}", "ft MD"),
            ("Operating casing head pressure", f"{res.pso:,.1f}", "psia"),
            ("Unloading valves", f"{n_unl}", ""),
            ("Operating injection", res.valves[-1].kind, f"{res.valves[-1].port_64:.1f}/64 in"),
        ], columns=["Result", "Value", "Units"])
        st.dataframe(summ, hide_index=True, width="stretch")
        for m_ in res.messages:
            st.caption(m_)
        tab_ = res.table()[GL_SHOW_COLS]
        st.dataframe(tab_.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
        st.download_button("Download valve table (CSV)", tab_.to_csv(index=False).encode(), "gaslift_design.csv")
        t1, t2, t3, t4 = st.tabs(["Design plot", "Performance curve", "Stability", "Design gradient"])
        with t1:
            st.pyplot(gl_design_plot(res), clear_figure=True)
        with t2:
            if res.perf is None:
                st.info("No performance curve for a user-entered design rate (as in PROSPER).")
            else:
                st.pyplot(gl_perf_plot(res.perf, res), clear_figure=True)
                st.dataframe(res.perf.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
                st.download_button("Download performance curve (CSV)", res.perf.to_csv(index=False).encode(),
                                   "gaslift_performance.csv")
        with t3:
            if stab is None:
                st.info("Enter the tubing OD and casing ID of the tubing sections (Equipment Data) to calculate the "
                        "annulus volume needed by the stability criteria.")
            else:
                verdict = "stable" if stab["stable"] else "unstable"
                (st.success if stab["stable"] else st.error)(
                    f"Asheim (1988): F1 = {stab['F1']:.3f}" + (f", F2 = {stab['F2']:.3f}" if np.isfinite(stab['F2'])
                                                               else "") + f" - the design is predicted {verdict}.")
                st.caption("F1 > 1: stable (inflow response). Otherwise F2 > 1 is needed (casing pressure-depletion "
                           "response). Orifice efficiency E = 0.9.")
                st.dataframe(pd.DataFrame([
                    ("F1 - inflow response", stab["F1"], ""), ("F2 - pressure-depletion response", stab["F2"], ""),
                    ("Productivity index J", stab["J"], "STB/d/psi"),
                    ("Injection port", stab["port_64"], "64ths in"),
                    ("Lift gas FVF at injection", stab["Bg"], "ft3/scf"),
                    ("Lift gas density at injection", stab["rho_gi"], "lb/ft3"),
                    ("Reservoir fluid density at injection", stab["rho_fi"], "lb/ft3"),
                    ("Reservoir fluid rate at injection", stab["q_fi"], "ft3/s"),
                    ("Lift gas rate at injection", stab["qgi_insitu"], "ft3/s"),
                    ("Tubing volume above injection", stab["Vt"], "ft3"),
                    ("Casing (annulus) volume", stab["Vc"], "ft3"),
                    ("Tubing pressure at injection", stab["pt"], "psia"),
                    ("Vertical depth of injection", stab["depth"], "ft")], columns=["Parameter", "Value", "Units"])
                    .style.format({"Value": "{:,.4g}"}, na_rep="—"), hide_index=True, width="stretch")
        with t4:
            prof = pd.DataFrame({"Measured depth (ft)": res.md, "True vertical depth (ft)": res.tvd,
                                 "Tubing pressure (psia)": res.pt, "Tubing temperature (°F)": res.tt,
                                 "Casing pressure, operating (psia)": res.pc_op,
                                 "Casing pressure, kick-off (psia)": res.pc_ko})
            st.dataframe(prof.style.format(precision=2), hide_index=True, width="stretch", height=380)


with tabs[6]:
    st.header("Design")
    st.radio("Artificial lift", ["Gas lift (continuous)"], horizontal=True, key="design_kind",
             help="Further artificial lift designs (ESP, rod pump, ...) can be added here.")
    o = ss.get("options")
    eqd = ss.get("equipment", {})
    ipr_curve = ss.get("ipr_curve")
    blocker = None
    if o is None or o.errors:
        blocker = "Fix the configuration in the Options summary first."
    elif o.is_gas or o.is_injector:
        blocker = "Continuous gas lift design applies to oil producers."
    elif o.flow_type != "tubing":
        blocker = "Gas lift design needs tubing flow (the lift gas goes down the annulus)."
    elif not eqd.get("valid") or eqd.get("survey") is None or not len(eqd["downhole"][1]):
        blocker = "Complete the Equipment Data tab first."
    elif ipr_curve is None or ipr_curve.fluid != "oil":
        blocker = "Calculate an oil IPR in the IPR Data tab first."
    if blocker:
        st.info(blocker)
    else:
        gaslift_tab(o, eqd, ipr_curve)


# ================================================================ save button (top bar) and schematic (sidebar)
with save_slot:
    try:
        payload = wproj.dumps(project_state(), pname)
        ss["_proj_payload"] = payload
        fname = _file_stem(pname)
        st.download_button("", payload.encode(), f"{fname}.wellmodel.json", "application/json",
                           icon=":material/save:", key="proj_save", on_click="ignore",
                           help=f"Save project as {fname}.wellmodel.json - every input, table and accepted match.")
    except TypeError as e:
        st.button("", icon=":material/error:", disabled=True, help=f"Project cannot be saved: {e}", key="proj_save")

with st.sidebar:
    st.header("Well schematic")
    if schematic_svg is not None:
        st.image(schematic_svg, width="stretch")
        st.caption("Depth spacing is stretched so every component stays readable.")
        st.download_button("Download schematic (SVG)", schematic_svg.encode(), "well_schematic.svg",
                           "image/svg+xml", width="stretch")
    else:
        st.info("The schematic appears once every section of the Equipment Data tab is complete.")
