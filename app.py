"""Streamlit front end for fluxwell.

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

from fluxwell import blackoil as bo
from fluxwell import eos as wm_eos
import inspect
import dataclasses

from scipy.optimize import brentq

from fluxwell import inflow as _ipr_mod
from fluxwell import inflow_adv as _ipr2_mod
from fluxwell import completion as wcomp
from fluxwell import mpp as _mpp_mod
from fluxwell import vlp as wvlp
from fluxwell import system as wsys
from fluxwell import injection as winj
from fluxwell import tif as wtif
from fluxwell import surface as wsurf
from fluxwell import project as wproj
from fluxwell import liftcurves as wlc
from fluxwell import gradient as wgrad
from fluxwell import choke as wchoke
from fluxwell import gaslift as wgl
from fluxwell import gaslift_system as wgls
from fluxwell import coiled as wct
from fluxwell import quicklook as wql
from fluxwell import esp_quicklook as weql
from fluxwell import hsp_quicklook as whql
from fluxwell import srp_cards as wsrpc
from fluxwell import diluent as wdil
from fluxwell import emulsion as wem
from fluxwell import rheology as wrh
from fluxwell import cbm as wcbm
from fluxwell import foam as wfoam
from fluxwell import fracture as wfrac
from fluxwell import mudacid as wacid
from fluxwell import spot as wspot
from fluxwell import multilateral as wmlat
from fluxwell import mpump as wmpump
from fluxwell import thermal as wth
from fluxwell import enthalpy as web
from fluxwell import co2 as wco2
from fluxwell import steam as wsteam
from fluxwell import hydrate as whyd
from fluxwell import intermittent as wigl
from fluxwell import esp as wesp
from fluxwell import srp as wsrp
from fluxwell import jetpump as wjet
from fluxwell import hsp as whsp
from fluxwell import pcp as wpcp
from fluxwell import gaspvt as wgp
from fluxwell import retrograde as wrg
from fluxwell import complementary as wcx
from fluxwell import units as wu
from fluxwell import ui_units as wuu
try:                                    # PROSPER-style grids with per-cell locking; optional dependency
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    HAVE_AGGRID = True
except ImportError:                     # falls back to st.data_editor
    HAVE_AGGRID = False
from fluxwell.correlations import LABELS as VLP_LABELS
from fluxwell import equipment as weq
from fluxwell import schematic as wsch
from fluxwell import options as wopt
from fluxwell import plotting as wplt

_ASSETS = Path(__file__).resolve().parent / "assets"
st.set_page_config(page_title="FluxWell", page_icon=str(_ASSETS / "fluxwell_icon.png"), layout="wide")
st.logo(str(_ASSETS / "fluxwell_logo.png"), icon_image=str(_ASSETS / "fluxwell_icon.png"), size="large")
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
    """Current value of an input (internal units): live widget state, else the stored copy."""
    k = wkey(key)
    qi = ss.get("_qty_" + key)
    if qi is not None:                       # unit-aware input: its widget holds the value in the chosen unit
        q, u_int = qi
        to = wu.display_unit(q, u_int, ss.get("_units"))
        k2 = k if to == u_int else f"{k}@{to}"
        if k2 in ss and ss[k2] is not None:
            return wu.convert(ss[k2], q, to, u_int)
        return ss.get("_v_" + key, default)
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
    for k_ in [k_ for k_ in ss.keys() if k_ == wkey(key) or str(k_).startswith(wkey(key) + "@")]:
        ss.pop(k_, None)
    ss.pop("_v_" + key, None)
    ss["_ver_" + key] = ss.get("_ver_" + key, 0) + 1


UNIT_STYLE = ("<div style='height:2.5rem;display:flex;align-items:center;margin-bottom:1rem;font-size:0.86rem;"
              "color:rgba(49,51,63,0.75);white-space:nowrap'>{}</div>")


def unit_beside(col, unit):
    col.markdown(UNIT_STYLE.format(unit), unsafe_allow_html=True)


def unit_split(unit):
    """Column widths for the input box and the unit text beside it: longer units get more room."""
    return [0.8, 0.2] if len(unit) <= 7 else [0.68, 0.32]


def num(col, label, key, default, mn=None, mx=None, step=None, fmt=None, help=None, disabled=False):
    """Number input. A label ending in '(unit)' with a known unit is shown in the unit chosen on the Units tab,
    with the unit beside the box; the value returned (and stored) is always in the internal unit."""
    cur = float(val(key, default))
    if mn is not None:
        cur = max(cur, float(mn))
    if mx is not None:
        cur = min(cur, float(mx))
    sp = wu.split_label(label)
    if sp is None:
        kw = {k: v for k, v in dict(min_value=None if mn is None else float(mn),
                                    max_value=None if mx is None else float(mx),
                                    step=None if step is None else float(step), format=fmt, help=help).items()
              if v is not None}
        fixed = wu.fixed_unit(label)
        if fixed is not None:      # unit with nothing to convert to: still shown beside the box
            inner = col.columns(unit_split(fixed[1]), vertical_alignment="bottom", gap="small")
            v = inner[0].number_input(fixed[0], value=cur, key=wkey(key), disabled=disabled, **kw)
            unit_beside(inner[1], fixed[1])
            ss["_v_" + key] = v
            return v
        v = col.number_input(label, value=cur, key=wkey(key), disabled=disabled, **kw)
        ss["_v_" + key] = v
        return v
    base, q, u_int, suffix = sp
    to = wu.display_unit(q, u_int, ss.get("_units"))
    ss["_qty_" + key] = (q, u_int)
    conv = (lambda x: x) if to == u_int else (lambda x: float(wu.convert(float(x), q, u_int, to)))
    lo = None if mn is None else conv(mn)
    hi = None if mx is None else conv(mx)
    shown = conv(cur)
    if lo is not None:
        shown = max(shown, lo)
    if hi is not None:
        shown = min(shown, hi)
    if to == u_int:
        fmt2, step2 = fmt, step
    else:
        fmt2 = "%.6g"
        step2 = None if step is None else float(step) * wu.scale(q, u_int, to)
    kw = {k: v for k, v in dict(min_value=lo, max_value=hi, step=step2, format=fmt2, help=help).items()
          if v is not None}
    inner = col.columns(unit_split(wu.unit_label(to, suffix)), vertical_alignment="bottom", gap="small")
    k = wkey(key) if to == u_int else f"{wkey(key)}@{to}"
    v = inner[0].number_input(base, value=float(shown), key=k, disabled=disabled, **kw)
    unit_beside(inner[1], wu.unit_label(to, suffix))
    out = v if to == u_int else float(wu.convert(float(v), q, to, u_int))
    ss["_v_" + key] = out
    return out


def _fmt_list(vals):
    return ", ".join(f"{x:.6g}" for x in vals)


def parse_values(text):
    out = []
    for tok in str(text).replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


def num_list(col, label, key, default_text, note="comma separated", help=None):
    """Comma-separated numbers. A label ending in '(unit)' takes the values in the chosen unit (unit beside the box);
    returns the list in internal units. Invalid entries raise ValueError (as parse_values)."""
    sp = wu.split_label(label)
    if sp is None:
        lab = f"{label}, {note}" if note else label
        return parse_values(col.text_input(lab, key=key, help=help, **dflt(key, value=default_text)))
    base, q, u_int, suffix = sp
    to = wu.display_unit(q, u_int, ss.get("_units"))
    k = key if to == u_int else f"{key}@{to}"
    if k not in ss:
        stored = ss.get("_vl_" + key)
        if stored is None:
            ss[k] = default_text if to == u_int else _fmt_list(wu.convert(parse_values(default_text), q, u_int, to))
        else:
            ss[k] = _fmt_list(stored if to == u_int else wu.convert(list(stored), q, u_int, to))
    inner = col.columns([0.8, 0.2], vertical_alignment="bottom", gap="small")
    txt = inner[0].text_input(f"{base}, {note}" if note else base, key=k, help=help)
    unit_beside(inner[1], wu.unit_label(to, suffix))
    vals = parse_values(txt)
    vals = vals if to == u_int else [float(wu.convert(v, q, to, u_int)) for v in vals]
    ss["_vl_" + key] = vals
    return vals


def unit_of(label):
    """'Wellhead pressure (psia)' -> ' (psia)' (for derived labels such as 'First (psia)'), '' if no unit."""
    sp = wu.split_label(label)
    return f" ({sp[2]})" if sp is not None else ""


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
PROJECT_WIDGET_PREFIXES = ("opt_", "sys_how_", "sys_vals_", "trav_rates_", "vfp_thp_", "vfp_wfr_", "vfp_gfr_", "vfp_alq",
                           "vfp_levels")
PROJECT_EXTRAS = ("prp_text", "prp_name", "surf_matched", "vlp_selected", "vlp_u_accepted", "vlp_adj_accepted")


def project_state():
    """Everything needed to rebuild the model: input values, raw widget values, tables, accepted results."""
    inputs = {k[3:]: v for k, v in ss.items() if k.startswith("_v_") and isinstance(v, (int, float, str, bool, list))}
    widgets = {k: v for k, v in ss.items() if isinstance(k, str) and (k in PROJECT_WIDGETS or
                                                                    k.startswith(PROJECT_WIDGET_PREFIXES)
                                                                    or k.startswith("_vl_"))
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
    stem = fname.rsplit(".", 1)[0].replace(".fluxwell", "").replace(".wellmodel", "").strip()
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
hdr[0].image(str(_ASSETS / "fluxwell_logo.png"), width=300)
pname = hdr[1].text_input("Project name", key="proj_name", **dflt("proj_name", value="my_well"),
                          help="Press Enter after typing - the save button then writes <name>.fluxwell.json.")
save_slot = hdr[2]                                          # filled at the end, once every tab has run
with hdr[3].popover("", icon=":material/folder_open:", help="Open project"):
    st.file_uploader("Open project", type=["json"], key=_uploader_key(), on_change=queue_project,
                     args=(_uploader_key(),),
                     help="A .fluxwell.json file saved from this tool (older .wellmodel.json files also load). Replaces the current model.")
    st.caption("Replaces the current model; the PVT match is repeated automatically. Calculation results are "
               "not saved - press Calculate after opening.")
hdr[4].button("", icon=":material/note_add:", on_click=new_project, help="New project - clears every input",
              key="proj_new")
_msg_slot = st.empty()            # always present, so the tabs below keep their identity (and selection)
_msg = ss.pop("_proj_msg", None)
if _msg:
    getattr(_msg_slot, _msg[0])(_msg[1])
schematic_svg = None

# chosen units (Units tab): every input and output is shown in them; the engine keeps oilfield units
ss["_units"] = {q_: (val("unit_" + q_, wu.default_unit(q_)) if val("unit_" + q_, wu.default_unit(q_)) in wu.QUANTITIES[q_]
                     else wu.default_unit(q_)) for q_ in wu.QUANTITIES}
wuu.install()

TAB_NAMES = ["1 · Options summary", "2 · PVT Input Data", "3 · IPR Data", "4 · Equipment Data", "5 · VLP",
             "6 · Calculation", "7 · Design", "8 · Units"]
tabs = st.tabs(TAB_NAMES, key="main_tabs")

CAPTIONS = {
    "fluid_type": {"cbm": "Coalbed methane: water with desorbed gas (de-watering, PROSPER Tutorial 45)",
                   "oil": "Oil with associated gas and water",
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
    "separator": {"single_stage": "Surface rates at one separator stage"},
    "emulsions": {"none": "Liquid viscosity: oil and water by volume",
                  "emulsion": "Oil-water emulsion viscosity against water cut (PROSPER Tutorial 46): data in PVT"},
    "viscosity_model": {"newtonian": "Oil viscosity from the PVT correlation (matched)",
                        "non_newtonian": "Herschel-Bulkley oil: apparent viscosity at the wall shear rate (PROSPER "
                                         "Tutorial 44): rheological parameters in PVT"},
    "water_vapour": {"none": "Water produced as liquid only",
                     "calculate": "Dry and wet gas: water vaporised in the gas condenses up the well"},
    "temp_model": {"rough": "Temperature from the geothermal gradient and one U value (heat loss only)",
                   "improved": "Energy balance with Joule-Thomson, gravity and U by depth (PROSPER Tutorial 41)",
                   "enthalpy": "Energy balance with U calculated from the completion, casing record, lithology and "
                               "producing time (PROSPER Tutorial 40)"},
    "steam": {"none": "Water injected as liquid",
              "allow": "Steam injector: steam tables, quality from the enthalpy (PROSPER Tutorial 42; Improved "
                       "Approximation)"},
    "artificial_lift": {"none": "Natural flow",
                        "gas_lift": "Continuous gas lift: Gas lift data in Equipment, Calculation (design in Design)",
                        "ct_gas_lift": "Lift gas down a coiled tubing, production up the annulus: Coiled tubing data in "
                                       "Equipment, Calculation (design in Design)",
                        "esp": "Pump in the well: Equipment, VLP at the pump discharge, Calculation",
                        "srp": "Rod pump in the well: Equipment, system analysis at the pump intake, Calculation "
                               "(design in Design)",
                        "jet": "Power fluid down the annulus, production + power fluid up the tubing: Equipment, "
                               "Calculation (design in Design)",
                        "pcp_motor": "Positive-displacement pump driven by a downhole motor: Equipment, Calculation "
                                     "(design in Design)",
                        "pcp_rod": "Positive-displacement pump turned by a rod string from surface (production up "
                                   "the rod-tubing annulus): Equipment, Calculation (design in Design)",
                        "hsp": "Centrifugal pump driven by a turbine on power fluid down the annulus, exhausting into "
                               "the tubing: Equipment, Calculation (design in Design)",
                        "diluent_gl": "Diluent water down the annulus into the tubing, with lift gas at a valve above "
                                      "it (PROSPER Tutorial 30): Equipment, Calculation (system analysis)",
                        "diluent": "Diluent water down the annulus into the tubing: Equipment, Calculation (system "
                                   "analysis)",
                        "foam_lift": "Surfactant injected at the perforations turns the water into foam (PROSPER "
                                     "Tutorial 48): surfactant data in Equipment, Foam-Homogeneous in Calculation",
                        "mpump": "Pump in the surface flowline driven by the power available (PROSPER Tutorial 31): "
                                 "Pump row in the surface equipment and Multiphase pump data in Equipment, Calculation "
                                 "with the manifold as the top node"},
}
STATUS_ICON = {wopt.MODELLED_NOW: "✅ Modelled", wopt.LUMPED_IN_PI: "↪️ Absorbed in matched PI",
               wopt.PVT_ONLY: "🧪 PVT only - flow calculations not built yet",
               wopt.NOT_YET: "⏳ Not modelled yet"}

with tabs[0]:
    st.header("Options summary")
    st.caption("Describe the well first. These choices decide what the setup tabs ask for "
               "and which calculations apply.")
    sel_opts = {}
    groups = [("Fluid description", ["fluid_type", "method", "separator", "water_vapour", "emulsions",
                                     "viscosity_model"]),
              ("Well", ["well_type", "flow_type", "artificial_lift", "temp_model", "steam"]),
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


def run_match(suggest=True):
    """Match all correlations to the lab data; suggest=True selects the best-ranked families (the Match button),
    False keeps the selection (re-matching a saved project, whose saved correlations win)."""
    inp, data = current_inputs()
    res = bo.match_all(inp, data)
    ss["pvt_match"] = res
    ss["pvt_match_sig"] = signature(inp, data)
    if not suggest and val("pvt_sel_pb") and val("pvt_sel_visc"):
        return
    if res.suggestion.get("pb_family"):
        put("pvt_sel_pb", res.suggestion["pb_family"])
    if res.suggestion.get("visc_family"):
        put("pvt_sel_visc", res.suggestion["visc_family"])


def reset_match():
    ss.pop("pvt_match", None)
    ss.pop("pvt_match_sig", None)


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
               "(Lohrenz-Bray-Clark is used). Pressures in the file are psig and are shown here in "
               f"{ss['_units']['pressure']}.")

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
    has_liq = np.isfinite(sep["api"]) and 0 < sep["api"] < 200 and sep.get("stb_per_lbmol", 1.0) > 1e-9
    m[3].metric("Stock-tank liquid (°API)", f"{sep['api']:.1f}" if has_liq else "—")
    m[4].metric("Separator gas gravity", f"{sep['gas_gravity']:.3f}" if np.isfinite(sep["gas_gravity"]) else "—")
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
        run_match(suggest=False)
    except (ValueError, KeyError, TypeError) as e:
        ss["_proj_msg"] = ("warning", f"PVT match could not be repeated: {e}")

GAS_MATCH_COLS = ["Temperature (deg F)", "Pressure (psia)", "Z factor", "Gas viscosity (cp)", "Gas FVF (ft3/scf)"]


def gas_match_tables():
    """Match data from the PVT tab grouped by temperature: [(T, DataFrame)]."""
    df = ss.get("_tables", {}).get("pvt_gas_match")
    if df is None:
        df = ss.get("pvt_gas_match")
    if df is None or not len(df):
        return []
    df = df.copy()
    for c in GAS_MATCH_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df[np.isfinite(df["Temperature (deg F)"]) & np.isfinite(df["Pressure (psia)"])]
    return [(float(t), g) for t, g in df.groupby("Temperature (deg F)")]


@st.cache_resource(show_spinner="Building the CO2 property table (Span-Wagner corrected EOS)...")
def co2_injection_fluid(text: str, name: str):
    """Compositional injected stream from the PRP file (co2.CO2InjectionFluid; the CO2-corrected property table when
    the file asks for PROSPER's CO2 correction)."""
    return wco2.CO2InjectionFluid(wm_eos.EOSFluid(wm_eos.parse_prp(text, name)))


def inj_gas_fluid():
    """Injected gas for the gas-injector calculations: the PRP composition with the equation of state (Tutorial 39),
    otherwise the PVT tab inputs (impurities left out when 'as PROSPER')."""
    o_ = ss.get("options")
    if o_ is not None and o_.method == "eos" and ss.get("prp_text"):
        return co2_injection_fluid(ss["prp_text"], ss.get("prp_name", ""))
    return winj.GasFluid(gas_pvt_inputs(), impurities=not bool(val("inj_gas_prosper", False)))


def gas_pvt_inputs(opts_now=None, matched=None) -> wgp.GasPVTInputs:
    """Dry and wet gas PVT inputs from the PVT tab (pressures psia internally), with the PVT match applied when
    'Apply match parameters' is on (matched=False: the correlations as published)."""
    gi = _gas_pvt_inputs_raw()
    use = bool(val("pvt_gas_use_match", True)) if matched is None else matched
    tabs_ = gas_match_tables() if use else []
    if tabs_:
        try:
            gi.match = wgp.match_gas(gi, tabs_)
        except (ValueError, np.linalg.LinAlgError):
            gi.match = {}
    return gi


def retro_bo_on(o_=None):
    o_ = o_ or ss.get("options")
    return bool(o_ is not None and o_.fluid_type == "retrograde_condensate" and o_.method == "black_oil"
                and o_.well_type == "producer")


def retro_inputs() -> wrg.RetroInputs:
    """PROSPER's black-oil retrograde condensate inputs (PVT tab)."""
    return wrg.RetroInputs(
        sep_gor=float(val("pvt_sep_gor", 7000.0)), sep_sg=float(val("pvt_gas_sg", 0.70)),
        tank_gor=float(val("pvt_tank_gor", 0.0)), tank_sg=float(val("pvt_tank_sg", 0.70)),
        api=float(val("pvt_cond_api", 50.0)), p_dew=float(val("pvt_dew", 4014.7)),
        t_res=float(val("pvt_t", 200.0)), p_res=float(val("pvt_pres", 3014.7)), wgr=float(val("pvt_wgr", 0.0)),
        salinity_ppm=float(val("pvt_sal", INPUT_DEFAULTS["pvt_sal"])), h2s=float(val("pvt_h2s", 0.0)),
        co2=float(val("pvt_co2", 0.0)), n2=float(val("pvt_n2", 0.0)))


def _gas_pvt_inputs_raw() -> wgp.GasPVTInputs:
    if retro_bo_on():               # the wet-gas inputs equivalent to the condensate (IPR pseudo-pressure etc.)
        ri = retro_inputs()
        return wgp.GasPVTInputs(
            gas_sg=ri.gas_sg, sep_p=float(val("pvt_sep_p", 14.696)) - 14.696, cgr=ri.cgr, cond_api=ri.api,
            wgr=ri.wgr, salinity_ppm=ri.salinity_ppm, h2s=ri.h2s, co2=ri.co2, n2=ri.n2,
            visc="carr" if val("pvt_gvisc", "Lee et al") == "Carr et al" else "lee",
            t_res=ri.t_res, p_res=ri.p_res - 14.696)
    return wgp.GasPVTInputs(
        gas_sg=float(val("pvt_gas_sg", 0.70)), sep_p=float(val("pvt_sep_p", 214.7)) - 14.696,
        cgr=float(val("pvt_cgr", 0.0)), cond_api=float(val("pvt_cond_api", 50.0)), wgr=float(val("pvt_wgr", 0.0)),
        salinity_ppm=float(val("pvt_sal", INPUT_DEFAULTS["pvt_sal"])), h2s=float(val("pvt_h2s", 0.0)),
        co2=float(val("pvt_co2", 0.0)), n2=float(val("pvt_n2", 0.0)),
        visc="carr" if val("pvt_gvisc", "Lee et al") == "Carr et al" else "lee",
        t_res=float(val("pvt_t", 200.0)), p_res=float(val("pvt_pres", 3014.7)) - 14.696)


def gas_vapour_on(o_=None):
    o_ = o_ or ss.get("options")
    return bool(o_ is not None and getattr(o_, "water_vapour", "none") == "calculate"
                and o_.fluid_type == "dry_wet_gas" and o_.method == "black_oil")


def steam_injector_calc(o, mode, eqd):
    """PROSPER Tutorial 42: gradient traverse of a steam injector with a wellhead enthalpy (or quality) sensitivity."""
    if not wsteam.available():
        st.error("Steam calculations need the CoolProp package (pip install CoolProp).")
        return
    if mode != CALC_MODES[1]:
        st.info("Steam injector: the gradient traverse (with enthalpy or quality sensitivities) is the calculation - "
                "as PROSPER notes, the reservoir response to steam belongs in a thermal simulator, so there is no "
                "IPR / system analysis here. Choose the gradient (traverse) calculation.")
        return
    try:
        fpath = with_thermal(vlp_path(eqd, "tubing", ()))
        path = winj.from_flow_path(fpath)
    except ValueError as e:
        st.error(str(e))
        return
    thermal = getattr(fpath, "thermal", None)
    u = float(val("eq_u", eqd.get("u_geo", 8.0)))
    u_seg = thermal.get("u_seg") if thermal is not None else None
    sal = float(val("pvt_sal", INPUT_DEFAULTS["pvt_sal"]))
    st.caption("Steam injector: the water (as steam) enters the wellhead with the enthalpy given and flows down; the "
               "enthalpy falls with the heat lost to the formation (U by depth, Improved Approximation table in the "
               "Equipment tab) and rises with the elevation drop; the steam quality and temperature follow from the "
               "steam tables at each node, the water is the PVT tab's brine "
               f"({sal:,.0f} ppm). Multiphase flow: PE2-style, steps of 20 ft (PROSPER's steam step length).")
    c = st.columns(4)
    whp = num(c[0], "First node pressure (psia)", "steam_whp", 2214.7, 14.7, 3200.0, 10.0)
    q = num(c[1], "Liquid rate (STB/d of water)", "steam_q", 4000.0, 1.0, 1e5, 100.0)
    choice = sel(c[2], "Steam choice", ["Enter enthalpy", "Enter quality"], "steam_choice")
    if choice == "Enter enthalpy":
        vals = num_list(c[3], "Steam enthalpy (BTU/lb) - one case each", "steam_h_list", "900, 950, 1000, 1050, 1100")
    else:
        vals = num_list(c[3], "Wellhead steam quality (fraction) - one case each", "steam_x_list", "0.5, 0.7, 0.9")
    step = num(st.columns(4)[0], "Steam step length (ft)", "steam_step", 20.0, 2.0, 500.0, 5.0)
    if not vals:
        return
    cases = []
    with st.spinner("Marching the steam down the well..."):
        for v_ in vals:
            h0 = float(v_) if choice == "Enter enthalpy" else wsteam.enthalpy_of(whp, quality=float(v_))
            cases.append((v_, h0, wsteam.traverse(path, whp, q, h0, u=u, u_seg=u_seg, step=step, salinity_ppm=sal)))
    rows = []
    for v_, h0, r in cases:
        grav = float(np.sum(0.5 * (r.grav[1:] + r.grav[:-1]) * np.diff(r.md)))
        fric = -float(np.sum(0.5 * (r.fric[1:] + r.fric[:-1]) * np.diff(r.md)))
        rows.append({"Wellhead enthalpy (BTU/lb)": h0, "Wellhead quality (%)": 100 * r.x[0],
                     "Wellhead temperature (°F)": r.t[0], "Bottom pressure (psia)": r.p[-1],
                     "Bottom temperature (°F)": r.t[-1], "Bottom enthalpy (BTU/lb)": r.h[-1],
                     "Bottom quality (%)": 100 * r.x[-1], "dP gravity (psi)": grav, "dP friction (psi)": fric})
    st.dataframe(pd.DataFrame(rows).style.format(precision=2), hide_index=True, width="stretch")
    cc = st.columns(3)
    for k_, (lab_, fn_) in enumerate((("Pressure (psia)", lambda r: r.p), ("Steam quality (%)", lambda r: 100 * r.x),
                                      ("Temperature (°F)", lambda r: r.t))):
        fig, ax = plt.subplots(figsize=(5, 5.5))
        for j_, (v_, h0, r) in enumerate(cases):
            ax.plot(fn_(r), r.md, color=wplt.SERIES[j_ % len(wplt.SERIES)], lw=1.6, label=f"h {h0:,.0f}")
        ax.invert_yaxis()
        ax.set_xlabel(lab_)
        ax.set_ylabel("Measured depth (ft)")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=7)
        fig.tight_layout()
        cc[k_].pyplot(fig, clear_figure=True)
    pick = st.selectbox("Gradient table for", list(range(len(cases))),
                        format_func=lambda i: f"Wellhead enthalpy {cases[i][1]:,.1f} BTU/lb", key="steam_pick")
    gt_ = cases[pick][2].table()
    st.dataframe(gt_.style.format(precision=4), hide_index=True, width="stretch", height=360)
    st.download_button("Download gradient (CSV)", gt_.to_csv(index=False).encode(), "steam_gradient.csv")


def injector_calc(o, mode, eqd):
    """Calculation tab for a water injector: system (tubing + injectivity) and gradient traverse."""
    if getattr(o, "steam", "none") == "allow":
        steam_injector_calc(o, mode, eqd)
        return
    try:
        fpath = with_thermal(vlp_path(eqd, "tubing", ()))
        path = winj.from_flow_path(fpath)
    except ValueError as e:
        st.error(str(e))
        return
    thermal = getattr(fpath, "thermal", None)
    if thermal is not None and thermal.get("model") != "improved":
        thermal = None
    gas_inj = o.well_type == "gas_injector"
    fl_ = inj_gas_fluid() if gas_inj else winj.WaterFluid(float(val("pvt_sal", INPUT_DEFAULTS["pvt_sal"])))
    ru = fl_.rate_unit.replace("/d", "/d")
    u = float(val("eq_u", eqd.get("u_geo", 8.0)))
    t_inj = float(val("eq_tinj", 60.0))
    hc = eqd.get("heat_capacities")
    what = "Gas" if gas_inj else "Water"
    p1, p2 = float(val("inj_p1", 1.0)), float(val("inj_p2", 1.0))
    st.caption(f"{what} injector: {what.lower()} at {t_inj:g} °F enters the wellhead and flows down the tubing "
               "(gravity raises the pressure, friction lowers it), exchanging heat with the formation "
               + ((f"(Improved Approximation, U by depth{' - energy balance with the stream enthalpy' if getattr(fl_, 'has_enthalpy', False) else ''}). ")
                  if thermal is not None else f"(U = {u:g} BTU/h/ft2/F). ")
               + "PVT from the PVT tab, injection temperature and U from the Equipment tab."
               + (f" VLP match applied (VLP tab): gravity x {p1:.5g}, friction x {p2:.5g}."
                  if (p1, p2) != (1.0, 1.0) else ""))
    if gas_inj and ss.get("inj_ipr") is not None and getattr(ss["inj_ipr"], "fluid", "") != "gas_injection":
        ss.pop("inj_ipr", None)
    if not gas_inj and ss.get("inj_ipr") is not None and getattr(ss["inj_ipr"], "fluid", "") != "water_injection":
        ss.pop("inj_ipr", None)
    if mode == CALC_MODES[0]:
        ipr_inj = ss.get("inj_ipr")
        if ipr_inj is None:
            st.info("Set up the injectivity in the IPR Data tab first.")
            return
        c = st.columns(4)
        whps = num_list(c[0], "Wellhead pressures (psia)", "sys_inj_whp", "1264.7")
        if not whps:
            return
        rows, sols = [], {}
        with st.spinner("Solving..."):
            for w_ in whps:
                res, rates, vp, ip = winj.solve(path, fl_, float(w_), ipr_inj, u, t_inj, hc, p1, p2, thermal=thermal)
                sols[w_] = (res, rates, vp, ip)
                rows.append({"Wellhead pressure (psia)": w_,
                             f"Injection rate ({ru})": res.q if res else np.nan,
                             "Solution node pressure (psia)": res.pwf if res else np.nan,
                             "dP skin (psi)": (float(np.ravel(ipr_inj.dp_skin(np.array([res.q]), np.array([res.T[-1]])))[0])
                                               if getattr(ipr_inj, "sandface_temperature", False)
                                               else float(ipr_inj.dp_skin(res.q))) if res else np.nan,
                             "Bottom node temperature (°F)": float(res.T[-1]) if res else np.nan,
                             "Note": "" if res else "cannot inject: the static column does not reach the reservoir "
                                                    "pressure"})
        st.dataframe(pd.DataFrame(rows).style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
        pick = (st.selectbox("Case", whps, format_func=lambda x: f"WHP {x:,.1f} psia", key="sys_inj_pick")
                if len(whps) > 1 else whps[0])
        res, rates, vp, ip = sols[pick]
        cc = st.columns([1, 1.6])
        if res is not None:
            with cc[0]:
                det = winj.details(path, fl_, res, ipr_inj, comp_current(), p_unit_gauge=False)
                det["Value"] = [f"{v_:,.4g}" if isinstance(v_, (int, float, np.floating)) else str(v_)
                                for v_ in det["Value"]]
                st.dataframe(det, hide_index=True, width="stretch", height=35 * (len(det) + 1) + 3)
        with cc[1]:
            fig, ax = plt.subplots(figsize=(8, 4.6))
            ax.plot(rates, vp, color=wplt.SERIES[0], lw=1.8, label="VLP (tubing)")
            ax.plot(rates, ip, color=wplt.SERIES[1], lw=1.8, label="IPR (injectivity)")
            if res is not None:
                ax.scatter([res.q], [res.pwf], s=50, facecolor="white", edgecolor=wplt.INK, lw=1.8, zorder=5,
                           label="solution")
            ax.set_xlabel(f"Injection rate ({ru})")
            ax.set_ylabel("Bottom-hole pressure (psia)")
            ax.grid(alpha=0.3, ls="--")
            ax.legend(fontsize=8)
            ax.set_title(f"System - WHP {pick:,.1f} psia", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        if getattr(ipr_inj, "sandface_temperature", False):
            _, tb_ = winj.vlp_curve(path, fl_, float(pick), rates, u, t_inj, hc, p1, p2, with_t=True,
                                    thermal=thermal)
            dps_ = ipr_inj.dp_skin(rates, tb_)
        else:
            tb_, dps_ = np.full(len(rates), np.nan), ipr_inj.dp_skin(rates)
        pts_ = pd.DataFrame({f"Injection rate ({ru})": rates, "VLP pressure (psia)": vp, "IPR pressure (psia)": ip,
                             "dP skin (psi)": dps_, "Sandface temperature (°F)": tb_})
        with st.expander("Rate table (VLP and IPR)"):
            st.dataframe(pts_.style.format(precision=2), hide_index=True, width="stretch")
        if res is not None:
            gt_ = winj.gradient_table(path, fl_, res.q, res.P, res.T, p_unit_gauge=False)
            with st.expander("Gradient at the solution rate"):
                st.dataframe(gt_.style.format(precision=4), hide_index=True, width="stretch")
                st.download_button("Download gradient (CSV)", gt_.to_csv(index=False).encode(), "injector_gradient.csv")
    elif mode == CALC_MODES[1]:
        c = st.columns(4)
        whp = num(c[0], "Wellhead pressure (psia)", "grad_inj_whp", 1264.7, 14.7, 30000.0, 10.0)
        q = (num(c[1], "Gas injection rate (MMscf/d)", "grad_ginj_q", 7.5, 0.0, 1e4, 0.5) if gas_inj
             else num(c[1], "Injection rate (STB/d)", "grad_inj_q", 10000.0, 0.0, 1e6, 100.0))
        P, T = winj.traverse(path, fl_, whp, [q], u, t_inj, hc, p1, p2, thermal)
        gt_ = winj.gradient_table(path, fl_, q, P[0], T[0], p_unit_gauge=False)
        cc = st.columns(2)
        for ax_i, (col_, lab_) in enumerate((("Pressure (psia)", "Pressure (psia)"), ("Temperature (°F)", "Temperature (°F)"))):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(gt_[col_], gt_["Measured depth (ft)"], color=wplt.SERIES[ax_i], lw=1.8)
            if col_.startswith("Temperature"):
                ax.plot(gt_["Formation temperature (°F)"], gt_["Measured depth (ft)"], color=wplt.INK2, lw=1,
                        ls="--", label="formation")
                ax.legend(fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(lab_)
            ax.set_ylabel("Measured depth (ft)")
            ax.grid(alpha=0.3, ls="--")
            fig.tight_layout()
            cc[ax_i].pyplot(fig, clear_figure=True)
        st.dataframe(gt_.style.format(precision=4), hide_index=True, width="stretch")
        st.download_button("Download gradient (CSV)", gt_.to_csv(index=False).encode(), "injector_gradient.csv")
    else:
        st.info("For water injectors this tab covers system analysis and the gradient traverse.")


INJ_MODELS = ["Darcy", "Thermally Induced Fracture", "P.I. Entry"]
INJ_GAS_MODELS = ["Petroleum Experts"]


def injector_ipr_panel(o):
    """Injectivity for a water injector (PROSPER IPR with Well Type = Water Injector)."""
    st.caption("Water injector: the reservoir takes water at a bottom-hole pressure above the reservoir pressure, "
               f"q = J (Pwf - Pr). Pressures in {ss['_units']['pressure']}.")
    gas_inj = o.well_type == "gas_injector"
    model = sel(st, "Reservoir model", INJ_GAS_MODELS if gas_inj else INJ_MODELS,
                "ipr_model_ginj" if gas_inj else "ipr_model_inj")
    c = st.columns(4)
    pr = num(c[0], "Reservoir pressure (psia)", "ipr_pr", 4114.7, 15.0, 30000.0, 10.0)
    t_res = num(c[1], "Reservoir temperature (°F)", "ipr_t_inj_res", 210.0, 32.0, 700.0, 1.0)
    if gas_inj:
        flg = inj_gas_fluid()
        c = st.columns(4)
        k = num(c[0], "Reservoir permeability (md)", "ipr_k", 25.0, 1e-4, 1e5, None, "%.5g")
        h = num(c[1], "Reservoir thickness (ft)", "ipr_h", 55.0, 0.1, 1e4, None, "%.5g")
        area = num(c[2], "Drainage area (acres)", "ipr_area", 500.0, 0.1, 1e5, None, "%.5g")
        ca = num(c[3], "Dietz shape factor", "ipr_ca", 31.6, 0.01, 100.0, None, "%.4g")
        c = st.columns(4)
        rw = num(c[0], "Wellbore radius (ft)", "ipr_rw", 0.354, 0.01, 5.0, None, "%.4g")
        hp = num(c[1], "Perforation interval (ft)", "ipr_hp", 45.0, 0.1, 1e4, None, "%.5g")
        tdays = num(c[2], "Time since production started (days)", "ipr_tdays", 100.0, 0.5, 1e6, 1.0)
        phi = num(c[3], "Reservoir porosity (fraction)", "ipr_phi", 0.15, 0.01, 0.5, 0.01)
        c = st.columns(4)
        swc = num(c[0], "Connate water saturation (fraction)", "ipr_swc", 0.25, 0.0, 0.9, 0.01)
        perm = sel(c[1], "Permeability entered", ["Total permeability", "Gas permeability"], "ipr_perm_kind")
        dm = sel(c[2], "Non-Darcy flow factor (D)", ["Calculated", "Entered"], "ipr_d_method")
        d_ent = num(c[3], "D (1/(Mscf/d))", "ipr_d", 1e-4, 0.0, 1.0, None, "%.4e") if dm == "Entered" else None
        skin = num(st.columns(4)[0], "Skin", "ipr_skin_inj", 2.0, -8.0, 500.0, None, "%.5g")
        perm_ = "gas" if perm == "Gas permeability" else "total"
        cur = winj.petex_gas_injection(pr, t_res, k, h, area, ca, rw, hp, tdays, phi, swc, skin, flg, d_ent, perm_)
        ss["inj_ipr"] = cur
        ss["inj_ipr_for"] = (lambda skin=skin, pr=pr: winj.petex_gas_injection(
            pr, t_res, k, h, area, ca, rw, hp, tdays, phi, swc, skin, flg, d_ent, perm_))
        c = st.columns(4)
        c[0].metric("D", f"{cur.params['Non-Darcy flow factor D (1/(Mscf/d))']:.4e} 1/(Mscf/d)")
        c[1].metric("Drainage term X", f"{cur.params['Drainage term X']:.4g}")
        c[2].metric("Gas permeability", f"{cur.params['Gas permeability (md)']:.4g} md")
        qq = np.linspace(0, max(cur.rate(2 * cur.pr), 1.0), 41)
        tab = cur.table(qq)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(qq, tab["Pwf (psia)"], color=wplt.SERIES[0], lw=1.8, label="Injectivity")
        ax.axhline(cur.pr, color=wplt.INK2, lw=0.8, ls="--", label="Reservoir pressure")
        ax.set_xlabel("Injection rate (MMscf/d)")
        ax.set_ylabel("Bottom-hole pressure (psia)")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption("Petroleum Experts gas IPR run in reverse: M(Pwf) - M(Pr) = a q + b q², M at reservoir "
                   "temperature. The table runs to twice the reservoir pressure; an injector's 'AOF' has no "
                   "physical meaning.")
        with st.expander("IPR table"):
            st.dataframe(tab.style.format(precision=4), hide_index=True, width="stretch")
        return
    fl_ = winj.WaterFluid(float(val("pvt_sal", INPUT_DEFAULTS["pvt_sal"])))
    if model == "Thermally Induced Fracture":
        tif_panel(pr, t_res, fl_)
        return
    if model == "Darcy":
        c = st.columns(4)
        k = num(c[0], "Reservoir permeability (md)", "ipr_k", 50.0, 1e-4, 1e5, None, "%.5g")
        h = num(c[1], "Reservoir thickness (ft)", "ipr_h", 100.0, 0.1, 1e4, None, "%.5g")
        area = num(c[2], "Drainage area (acres)", "ipr_area", 250.0, 0.1, 1e5, None, "%.5g")
        rw = num(c[3], "Wellbore radius (ft)", "ipr_rw", 0.354, 0.01, 5.0, None, "%.4g")
        c = st.columns(4)
        skin = num(c[0], "Skin", "ipr_skin_inj", 0.0, -8.0, 500.0, None, "%.5g")
        cur = ipr_inj = winj.darcy_injection(pr, t_res, k, h, area, rw, skin, fl_)
        ss["inj_ipr_for"] = lambda skin=skin, pr=pr: winj.darcy_injection(pr, t_res, k, h, area, rw, skin, fl_)
    else:
        j = num(st.columns(4)[0], "Injectivity index (STB/d/psi)", "ipr_j_inj", 5.0, 1e-6, 1e6, None, "%.5g")
        cur = ipr_inj = winj.pi_injection(pr, j)
        ss["inj_ipr_for"] = lambda skin=None, pr=pr: winj.pi_injection(pr, j)
    ss["inj_ipr"] = ipr_inj
    c = st.columns(4)
    c[0].metric("Injectivity index", f"{cur.j:.4g} STB/d/psi")
    if "Water viscosity (cp)" in cur.params:
        c[1].metric("Water viscosity at Pr, Tres", f"{cur.params['Water viscosity (cp)']:.4f} cp")
        c[2].metric("Water FVF", f"{cur.params['Water FVF (RB/STB)']:.4f} RB/STB")
    q_hi = cur.j * max(cur.pr, 1000.0)
    qq = np.linspace(0, q_hi, 41)
    tab = cur.table(qq)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(qq, tab["Pwf (psia)"], color=wplt.SERIES[0], lw=1.8, label="Injectivity")
    ax.axhline(cur.pr, color=wplt.INK2, lw=0.8, ls="--", label="Reservoir pressure")
    ax.set_xlabel("Injection rate (STB/d)")
    ax.set_ylabel("Bottom-hole pressure (psia)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("For an injector the 'AOF' has no physical meaning (no upper limit on the bottom-hole pressure); the "
               "table runs to J x Pr.")
    with st.expander("IPR table"):
        st.dataframe(tab.style.format(precision=2), hide_index=True, width="stretch")


def tif_panel(pr, t_res, fl_):
    """PROSPER Thermally Induced Fracture injectivity (water injector)."""
    st.caption("Thermally Induced Fracture: cold water cools the rock near the well and lowers the stress; above the "
               "fracturing pressure the rock fractures. Three regimes - radial, fracture opening, fracture developed "
               "- with the forms and coefficients found from PROSPER's Tutorial 03a (see fluxwell/tif.py). In this "
               "tab the injected temperature below is used; the system calculation uses the sandface temperature "
               "of the tubing at each rate.")
    st.markdown("**Injectivity index parameters**")
    c = st.columns(4)
    t_inj = num(c[2], "Injected fluid temperature (°F)", "ipr_tif_tinj", 120.0, 32.0, 400.0, 1.0)
    skin = num(c[3], "Skin", "ipr_skin_inj", 0.0, -8.0, 500.0, None, "%.5g")
    c = st.columns(4)
    k = num(c[0], "Reservoir permeability (md)", "ipr_k", 50.0, 1e-4, 1e5, None, "%.5g")
    h = num(c[1], "Reservoir thickness (ft)", "ipr_h", 100.0, 0.1, 1e4, None, "%.5g")
    area = num(c[2], "Drainage area (acres)", "ipr_area", 250.0, 0.1, 1e5, None, "%.5g")
    ca = num(c[3], "Dietz shape factor", "ipr_ca", 31.6, 0.01, 100.0, None, "%.4g")
    c = st.columns(4)
    rw = num(c[0], "Wellbore radius (ft)", "ipr_rw", 0.354, 0.01, 5.0, None, "%.4g")
    tdays = num(c[1], "Time (days)", "ipr_tdays", 20.0, 0.5, 1e6, 1.0)
    phi = num(c[2], "Reservoir porosity (fraction)", "ipr_phi", 0.18, 0.01, 0.5, 0.01)
    qh = num(c[3], "Mean historical injection rate (STB/d)", "ipr_tif_qh", 12000.0, 1.0, 1e7, 100.0)
    c = st.columns(4)
    mu_r = num(c[0], "Reservoir fluid viscosity (cp)", "ipr_tif_mur", 0.357, 0.01, 1e4, None, "%.4g",
               help="Fluid in the virgin reservoir at Pr and reservoir temperature (oil from the PVT, PROSPER's "
                    "Tutorial 03a: 0.357 cp).")
    b_r = num(c[1], "Reservoir fluid FVF (RB/STB)", "ipr_tif_bres", 1.41, 0.5, 5.0, None, "%.4g")
    st.markdown("**Thermo-mechanical parameters**")
    c = st.columns(4)
    sig = num(c[0], "Initial reservoir stress (psia)", "ipr_tif_sig", 5314.7, 15.0, 50000.0, 10.0)
    sweep = num(c[1], "Sweep efficiency (fraction)", "ipr_tif_sweep", 0.75, 0.01, 1.0, 0.01)
    cpw = num(c[2], "Injected fluid specific heat capacity (BTU/lb/F)", "ipr_tif_cpw", 1.0, 0.1, 5.0, None, "%.4g")
    lam_r = num(c[3], "Overall reservoir conductivity (BTU/hr/ft/F)", "ipr_tif_lamr", 2.5, 0.01, 50.0, None, "%.4g")
    c = st.columns(4)
    cp_r = num(c[0], "Overall reservoir specific heat capacity (BTU/lb/F)", "ipr_tif_cpr", 0.25, 0.01, 5.0, None, "%.4g")
    rho_r = num(c[1], "Overall reservoir density (lb/ft3)", "ipr_tif_rhor", 140.0, 10.0, 400.0, None, "%.4g")
    lam_s = num(c[2], "Top and bottom surroundings conductivity (BTU/hr/ft/F)", "ipr_tif_lams", 0.9, 0.01, 50.0, None, "%.4g")
    cp_s = num(c[3], "Top and bottom surroundings specific heat capacity (BTU/lb/F)", "ipr_tif_cps", 0.2, 0.01, 5.0,
               None, "%.4g")
    c = st.columns(4)
    rho_s = num(c[0], "Top and bottom surroundings density (lb/ft3)", "ipr_tif_rhos", 200.0, 10.0, 400.0, None, "%.4g")
    alpha = num(c[1], "Reservoir thermal expansion (1/F)", "ipr_tif_alpha", 5e-6, 0.0, 1e-3, None, "%.4e")
    biot = num(c[2], "Biot's constant", "ipr_tif_biot", 0.7, 0.0, 1.0, None, "%.4g")
    nu = num(c[3], "Poisson's ratio", "ipr_tif_nu", 0.2, 0.0, 0.49, None, "%.4g")
    e_mod = num(st.columns(4)[0], "Reservoir Young's modulus (psi)", "ipr_tif_e", 3e6, 1e3, 1e9, None, "%.4g")
    inp = wtif.TIFInputs(pr=pr, t_res=t_res, t_inj=t_inj, k=k, h=h, area=area, ca=ca, rw=rw, phi=phi, t_days=tdays,
                         q_hist=qh, skin=skin, sigma_ini=sig, sweep=sweep, cp_w=cpw, lam_r=lam_r, cp_r=cp_r,
                         rho_r=rho_r, lam_s=lam_s, cp_s=cp_s, rho_s=rho_s, alpha=alpha, biot=biot, nu=nu, e_mod=e_mod,
                         mu_res=mu_r, b_res=b_r)
    cur = wtif.tif_ipr(inp, fl_)
    ss["inj_ipr"] = cur
    ss["inj_ipr_for"] = lambda skin=skin, pr=pr: wtif.tif_ipr(dataclasses.replace(inp, skin=skin, pr=pr), fl_)
    c = st.columns(4)
    c[0].metric("Cooled radius", f"{cur.params['Cooled radius (ft)']:.1f} ft")
    c[1].metric("Flooded radius", f"{cur.params['Flooded radius (ft)']:.1f} ft")
    c[2].metric("Thermo-elastic coefficient Fh", f"{cur.params['Thermo-elastic coefficient Fh']:.3f}")
    qq = np.linspace(0.0, 2.5 * max(qh, 1000.0), 51)
    tab = cur.table(qq)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(qq, tab["Pwf (psia)"], color=wplt.SERIES[0], lw=1.8, label=f"Injectivity at {t_inj:g} °F")
    ax.axhline(cur.pr, color=wplt.INK2, lw=0.8, ls="--", label="Reservoir pressure")
    ax.axhline(sig, color=wplt.SERIES[2], lw=0.8, ls=":", label="Initial stress")
    ax.set_xlabel("Injection rate (STB/d)")
    ax.set_ylabel("Bottom-hole pressure (psia)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    with st.expander("IPR table"):
        st.dataframe(tab.style.format({"Injection rate (STB/d)": "{:,.0f}", "Pwf (psia)": "{:,.1f}",
                                       "dP skin (psi)": "{:,.1f}"}), hide_index=True, width="stretch")


def water_injector_pvt_panel():
    """PROSPER PVT input for a water injector: the injected water's salinity."""
    st.subheader("Injected water")
    st.caption("Water injector: the tubing and the injectivity use water properties only - PROSPER's water "
               "properties (FVF linear in pressure, density from the salinity and FVF, viscosity from temperature "
               "and salinity).")
    c = st.columns(4)
    num(c[0], "Water Salinity (ppm)", "pvt_sal", INPUT_DEFAULTS["pvt_sal"], 0.0, 300000.0, 1000.0, "%.0f")
    fl_ = winj.WaterFluid(float(val("pvt_sal", INPUT_DEFAULTS["pvt_sal"])))
    c[1].metric("Specific gravity (60 °F)", f"{fl_.sg:.4f}")
    st.subheader("Calculate")
    c = st.columns(4)
    temps = num_list(c[0], "Temperatures (°F)", "pvt_w_temps", "60, 120, 210", "comma separated")
    p_lo = num(c[1], "First pressure (psia)", "pvt_w_p0", 14.7, 14.7, 30000.0, 10.0)
    p_hi = num(c[2], "Last pressure (psia)", "pvt_w_p1", 6000.0, 14.7, 30000.0, 10.0)
    npts = int(num(c[3], "Number of pressures", "pvt_w_n", 13.0, 2.0, 100.0, 1.0, "%.0f"))
    if temps and p_hi > p_lo:
        recs = []
        for t_ in temps:
            pp = np.linspace(p_lo, p_hi, npts)
            bw, rho, mu = fl_.props(pp, np.full(npts, float(t_)))
            bw2, _, _ = fl_.props(pp + 1.0, np.full(npts, float(t_)))
            for i_ in range(npts):
                recs.append({"Temperature (°F)": t_, "Pressure (psia)": pp[i_], "Water FVF (RB/STB)": bw[i_],
                             "Water density (lb/ft3)": rho[i_], "Water viscosity (cp)": mu[i_],
                             "Water compressibility (1/psi)": -(bw2[i_] - bw[i_]) / bw[i_]})
        tab = pd.DataFrame(recs)
        st.dataframe(tab.style.format({"Water compressibility (1/psi)": "{:.4e}"}, precision=5), hide_index=True,
                     width="stretch", height=320)
        st.download_button("Download water PVT table (CSV)", tab.to_csv(index=False).encode(), "water_pvt.csv")
    o_ = ss.get("options")
    if o_ is not None and getattr(o_, "steam", "none") == "allow":
        steam_calculator_panel()


FRAC_FLUID_COLS = ["Temperature (°F)", "Flow behaviour index n", "Consistency index K (lbf.s^n/ft2)"]


def _frac_well(o):
    """(TVD ft and MD ft of the bottom node, last tubing ID in) from the Equipment tab, or None."""
    eqd = ss.get("equipment", {})
    if not eqd.get("valid") or eqd.get("survey") is None:
        return None
    md_s, tvd_s = eqd["survey"]
    top_md, secs, _ = eqd["downhole"]
    if secs is None or not len(secs):
        return None
    bottom = float(secs["MD bottom (ft)"].max())
    tub = secs[secs["Type"] == "Tubing"]
    tid = float(tub["Flow ID (in)"].iloc[-1]) if len(tub) else 4.0
    return float(np.interp(bottom, md_s, tvd_s)), bottom, tid


def frac_panel(o):
    """PROSPER Stimulation | Hydraulic Fracturing (Tutorial 49): screening of treatment volume and pump rate."""
    st.subheader("Hydraulic fracturing (stimulation screening)")
    st.caption("PROSPER's Stimulation | Hydraulic Fracturing (Tutorial 49): a linear/vertical (KGD) fracture of the "
               "gross thickness grown with a power-law fluid and Carter leak-off, PROSPER's 100-stage treatment (pad, "
               "proppant ramp) with the proppant bank that settles behind it, and the PI after from the propped "
               "length and conductivity. A screening tool, as PROSPER says - not a fracture simulator. See "
               "fracture.py for what was reconstructed and how closely.")
    well = _frac_well(o)
    if well is None:
        st.info("Complete the Equipment Data tab (deviation survey and downhole equipment) first.")
        return
    tvd, md_b, tid = well
    curve = ss.get("ipr_curve")
    pr = float(val("ipr_pr", 3000.0))
    pi0 = None
    if curve is not None and curve.fluid == "oil":
        pi0 = float(curve.rate(curve.pr - 50.0)) / 50.0
    d = wfrac.FracInputs()
    tabs_ = st.tabs(["Options", "Reservoir", "Well", "Rock", "Proppant", "Fracture fluid"])
    with tabs_[0]:
        c = st.columns(4)
        vmin = num(c[0], "Minimum treatment volume (bbl)", "frac_vmin", d.v_min, 10.0, 1e6, 100.0)
        vmax = num(c[1], "Maximum treatment volume (bbl)", "frac_vmax", d.v_max, 10.0, 1e6, 100.0)
        qmin = num(c[2], "Minimum treatment rate (bbl/min)", "frac_qmin", d.q_min, 1.0, 1e4, 10.0)
        qmax = num(c[3], "Maximum treatment rate (bbl/min)", "frac_qmax", d.q_max, 1.0, 1e4, 10.0)
        c = st.columns(4)
        wall = num(c[0], "Part of fracture wall taking fluid (%)", "frac_wall", 75.0, 0.0, 100.0, 5.0)
        inc = num(c[1], "Wall area increase due to irregularities (%)", "frac_inc", 50.0, 0.0, 500.0, 5.0)
        tsur = num(c[2], "Surface temperature (°F)", "frac_tsur", d.t_surface, -20.0, 200.0, 1.0)
        sloss = num(c[3], "Surface pressure losses (psi)", "frac_sloss", d.surface_losses, 0.0, 1e4, 10.0)
    with tabs_[1]:
        c = st.columns(4)
        hg = num(c[0], "Gross thickness (ft)", "frac_h", d.h_gross, 1.0, 5000.0, 5.0)
        obg = num(c[1], "Overburden gradient (psi/ft)", "frac_obg", d.overburden_grad, 0.5, 2.0, 0.01, "%.5f")
        area = num(c[2], "Drainage area (acres)", "frac_area", d.area_acres, 1.0, 1e5, 10.0)
        c[3].metric("Reservoir pressure", f"{pr:,.1f} psia")
    with tabs_[2]:
        c = st.columns(4)
        bh = num(c[0], "Borehole diameter (in)", "frac_bh", d.borehole_d, 1.0, 40.0, 0.1, "%.4g")
        pint = num(c[1], "Perforated interval (ft)", "frac_pint", d.perf_interval, 1.0, 5000.0, 5.0)
        spf = num(c[2], "Shot density (1/ft)", "frac_spf", d.spf, 0.1, 50.0, 1.0)
        pd_ = num(c[3], "Perforation tunnel diameter (in)", "frac_pd", d.perf_d, 0.05, 3.0, 0.01)
        st.caption(f"From the model: bottom node {md_b:,.0f} ft MD / {tvd:,.1f} ft TVD, tubing ID {tid:g} in.")
    with tabs_[3]:
        c = st.columns(4)
        phi = num(c[0], "Porosity (fraction)", "frac_phi", d.porosity, 0.01, 0.6, 0.01)
        kf = num(c[1], "Formation permeability (md)", "frac_k", d.k, 1e-4, 1e5, 1.0)
        ten = num(c[2], "Rock tensile strength (psi)", "frac_ten", d.tensile, 0.0, 1e4, 50.0)
        nu = num(c[3], "Poisson's ratio", "frac_nu", d.poisson, 0.01, 0.49, 0.01)
        c = st.columns(4)
        ym = num(c[0], "Young's modulus (psi)", "frac_E", d.youngs, 1e4, 1e8, 1e5, "%.4g")
    with tabs_[4]:
        c = st.columns(4)
        cw = num(c[0], "Fluid loss coefficient (ft/sqrt(min))", "frac_cw", d.c_wall, 1e-5, 0.1, 0.0005, "%.5g")
        rho = num(c[1], "Fluid density (lb/gal)", "frac_rho", d.fluid_density, 6.0, 20.0, 0.1)
        cmax = num(c[2], "Maximum proppant concentration (lb/gal)", "frac_cmax", d.c_max, 0.1, 20.0, 0.1)
        c[3].text_input("Proppant", "Graded sand 8-12", disabled=True,
                        help="The proppant PROSPER's tutorial uses; its conductivity against concentration is taken "
                             "from the T49 file (other proppants are not available).")
    with tabs_[5]:
        key = "frac_fluid"
        if key not in ss:
            ss[key] = pd.DataFrame(list(d.fluid_table), columns=FRAC_FLUID_COLS)
        tf = track(key, st.data_editor(ss[key], key=key + "_ed", num_rows="dynamic", width="stretch"))
    c = st.columns(4)
    pi_b = num(c[0], "Productivity index before (STB/d/psi)", "frac_pi0", round(pi0, 4) if pi0 else 5.0, 1e-4, 1e5,
               0.1, "%.5g", help="Defaults to the PI of the IPR tab's curve.")
    go = c[1].button("Calculate fracturing", type="primary")
    try:
        ftab = tuple(map(tuple, pd.DataFrame(tf)[FRAC_FLUID_COLS].apply(pd.to_numeric, errors="coerce")
                         .dropna().to_numpy()))
    except KeyError:
        ftab = d.fluid_table
    fi = wfrac.FracInputs(vmin, vmax, qmin, qmax, wall / 100.0, inc / 100.0, tsur, sloss, hg, obg, area, bh, pint, spf,
                          pd_, phi, kf, ten, nu, ym, cw, 0.0, rho, cmax, ftab or d.fluid_table, tvd, pr,
                          default_temperature(o), pi_b, tid, md_b)
    if go:
        with st.spinner("Calculating the treatments..."):
            fr = wfrac.Fracture(fi)
            tab, cases = fr.sensitivity()
            ss["frac_res"] = {"tab": tab, "cases": cases, "fr": fr}
    res = ss.get("frac_res")
    if not res:
        return
    tab, cases, fr = res["tab"], res["cases"], res["fr"]
    st.markdown(f"**Treatment results** - fluid at {fr.t_fluid:g} °F (n {fr.n:.4g}, K {fr.K:.4g} lbf.s^n/ft2)")
    cc = st.columns(2)
    for col_, (ycol, lab) in zip(cc, (("Final fracture length (ft)", "Final fracture length (ft)"),
                                      ("Final fracture width (in)", "Final fracture width (in)"))):
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for i, (q_, g_) in enumerate(tab.groupby("Pump rate (bbl/min)")):
            ax.plot(g_["Treatment volume (bbl)"], g_[ycol], "-o", ms=3, color=wplt.SERIES[i % len(wplt.SERIES)],
                    label=f"{q_:g} bbl/min")
        ax.set_xlabel("Treatment volume (bbl)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        col_.pyplot(fig, clear_figure=True)
    st.dataframe(tab.style.format(precision=4), width="stretch", height=280)
    st.download_button("Download treatment results (CSV)", tab.to_csv().encode(), "fracture_treatments.csv")
    k = int(st.number_input("Detailed results - case", 1, len(tab), min(38, len(tab)), key="frac_case"))
    pad, sch, eff, pr_ = cases[k - 1]
    q_, v_ = tab.loc[k, "Pump rate (bbl/min)"], tab.loc[k, "Treatment volume (bbl)"]
    t1, t2, t3, t4, t5 = st.tabs(["Treatment", "Fracture dimensions", "Productivity improvement", "Pumping schedule",
                                  "Pad"])
    with t1:
        tr = fr.treatment(q_, v_)
        st.dataframe(pd.DataFrame({"Result": list(tr), "Value": list(tr.values())}), hide_index=True, width="stretch")
        st.caption("Tubing friction: Colebrook with the apparent viscosity - PROSPER's differs by about 15 % (at these "
                   "rates through production tubing the figure is not meaningful). Apparent viscosity and bottom-hole "
                   "temperature during the job: not PROSPER's method.")
    with t2:
        st.dataframe(pd.DataFrame({"Result": list(eff), "Value": list(eff.values())}), hide_index=True, width="stretch")
    with t3:
        st.dataframe(pd.DataFrame({"Result": list(pr_), "Value": list(pr_.values())}), hide_index=True,
                     width="stretch")
        pia = float(pr_["Productivity index - after (STB/d/psi)"])

        def _use_pi(v=pia):
            put("ipr_model_oil", "P.I. Entry")
            put("ipr_j", round(v, 4))
        st.button(f"Use PI {pia:.4f} in the IPR (P.I. Entry)", on_click=_use_pi, key="frac_use_pi",
                  help="Sets the IPR to P.I. Entry with this PI; compare the system before and after (Calculation tab, "
                       "a sensitivity on the PI).")
    with t4:
        st.dataframe(sch.style.format(precision=4), hide_index=True, width="stretch", height=320)
        st.download_button("Download pumping schedule (CSV)", sch.to_csv(index=False).encode(),
                           f"pumping_schedule_case_{k}.csv")
    with t5:
        st.dataframe(pd.DataFrame({"Result": list(pad), "Value": list(pad.values())}), hide_index=True,
                     width="stretch")


ACID_DAMAGE = {"No damaged zone exists (the mud acid removes minerals in the matrix)": "none",
               "Shallow damaged zone caused by mud filtrate invasion exists": "shallow"}
ACID_FORM = {"Calculate optimum formulation based on ionic equilibrium": "optimum",
             "Use standard mud acid formulation (3 % HF and 12 % HCl)": "standard"}


def _pi_at_test(curve, q_test):
    """PROSPER's PI before: the test rate over its drawdown on the IPR curve."""
    if curve is None or getattr(curve, "fluid", "") != "oil" or q_test <= 0:
        return None
    lo, hi = 0.0, float(curve.pr)
    if float(curve.rate(lo)) < q_test:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(curve.rate(mid)) > q_test else (lo, mid)
    return q_test / (float(curve.pr) - 0.5 * (lo + hi))


def mudacid_panel(o):
    """PROSPER Stimulation | Mud Acidization of Sandstones (Tutorial 50)."""
    st.subheader("Mud acidization of sandstones (stimulation screening)")
    st.caption("PROSPER's Stimulation | Mud Acidization of Sandstones (Tutorial 50): the rock's composition sets the "
               "HF / HCl formulation, the porosity and permeability gained, the pumping schedule and limits, and the "
               "PI after treatment. A screening tool, as PROSPER says. See mudacid.py for what was reconstructed and "
               "how closely (fitted to the T50 file's runs).")
    well = _frac_well(o)
    if well is None:
        st.info("Complete the Equipment Data tab (deviation survey and downhole equipment) first.")
        return
    tvd, md_b, _ = well
    pr = float(val("ipr_pr", 3000.0))
    d = wacid.MudAcidInputs()
    tabs_ = st.tabs(["Options", "Reservoir", "Well", "Latest well test", "Rock", "Formation composition"])
    with tabs_[0]:
        c = st.columns(3)
        dmg = sel(c[0], "Damage type", list(ACID_DAMAGE), "acid_damage")
        frm = sel(c[1], "Calculation type", list(ACID_FORM), "acid_form")
        vol = num(c[2], "Volume of acid available (gal)", "acid_vol", d.volume_gal, 100.0, 1e7, 1000.0)
    with tabs_[1]:
        c = st.columns(3)
        obg = num(c[0], "Overburden gradient (psi/ft)", "acid_obg", d.overburden_grad, 0.5, 2.0, 0.01, "%.5f")
        area = num(c[1], "Drainage area (acres)", "acid_area", d.area_acres, 1.0, 1e5, 10.0)
        c[2].metric("Reservoir pressure", f"{pr:,.1f} psia")
    with tabs_[2]:
        c = st.columns(4)
        bh = num(c[0], "Borehole diameter (in)", "acid_bh", d.borehole_d, 1.0, 40.0, 0.1, "%.4g")
        pint = num(c[1], "Perforated interval (ft)", "acid_pint", d.perf_interval, 1.0, 5000.0, 5.0)
        spf = num(c[2], "Shot density (1/ft)", "acid_spf", d.spf, 0.1, 50.0, 1.0)
        pd_ = num(c[3], "Perforation tunnel diameter (in)", "acid_pd", d.perf_d, 0.05, 3.0, 0.01)
        st.caption(f"From the model: bottom node {md_b:,.0f} ft MD / {tvd:,.1f} ft TVD. Shot density and tunnel "
                   "diameter do not change PROSPER's results.")
    with tabs_[3]:
        c = st.columns(3)
        qt = num(c[0], "Liquid production rate (STB/d)", "acid_qt", d.q_test, 0.0, 1e7, 100.0)
        wct = num(c[1], "Test water cut (%)", "acid_wct", d.wc_test, 0.0, 100.0, 1.0)
        kt = num(c[2], "Permeability from test (md)", "acid_k", d.k, 1e-4, 1e5, 1.0)
    with tabs_[4]:
        c = st.columns(4)
        phi = num(c[0], "Porosity (fraction)", "acid_phi", d.porosity, 0.01, 0.6, 0.01)
        rb = num(c[1], "Rock bulk density (lb/ft3)", "acid_rb", d.bulk_density, 50.0, 300.0, 1.0,
                 help="Shown in the results; PROSPER calculates the density from the composition.")
        ten = num(c[2], "Rock tensile strength (psi)", "acid_ten", d.tensile, 0.0, 1e4, 50.0)
        nu = num(c[3], "Poisson's ratio", "acid_nu", d.poisson, 0.01, 0.49, 0.01)
    with tabs_[5]:
        comp = {}
        cols = st.columns(4)
        for i, (m, lab) in enumerate(wacid.MINERAL_LABELS.items()):
            comp[m] = num(cols[i % 4], f"{lab} (weight %)", "acid_m_" + m, d.composition[m], 0.0, 100.0, 1.0)
        tot = sum(comp.values())
        (st.caption if abs(tot - 100) < 1e-6 else st.warning)(f"Total {tot:g} weight %.")
    pi0 = _pi_at_test(ss.get("ipr_curve"), qt)
    c = st.columns(4)
    pi_b = num(c[0], "Productivity index before (STB/d/psi)", "acid_pi0", round(pi0, 5) if pi0 else 5.0, 1e-4, 1e5,
               0.1, "%.5g", help="Defaults to the test rate over its drawdown on the IPR tab's curve (PROSPER's).")
    go = c[1].button("Calculate mud acidization", type="primary")
    if go:
        if tot <= 0:
            st.error("Enter the formation composition.")
            return
        fi = wacid.MudAcidInputs(ACID_DAMAGE[dmg], ACID_FORM[frm], vol, obg, area, bh, pint, spf, pd_, qt, wct, kt,
                                 phi, rb, ten, nu, comp, tvd, pr, pi_b,
                                 wacid.water_density(float(val("pvt_sal", 0.0))))
        ss["acid_res"] = wacid.MudAcid(fi).results()
    res = ss.get("acid_res")
    if not res:
        return
    for w_ in res["warnings"]:
        st.warning(w_)
    t1, t2, t3, t4 = st.tabs(["Calculated rock properties", "Treatment results", "Pumping schedule", "Damage removal"])
    for tab_, key in zip((t1, t2, t3, t4), ("rock", "treatment", "pumping", "damage")):
        with tab_:
            dd = res[key]
            st.dataframe(pd.DataFrame({"Result": list(dd), "Value": [None if (isinstance(v, float) and np.isnan(v))
                                                                      else v for v in dd.values()]}),
                         hide_index=True, width="stretch")
    pia = res["damage"]["Productivity index after treatment (STB/d/psi)"]
    if np.isfinite(pia):
        def _use_pi(v=pia):
            put("ipr_model_oil", "P.I. Entry")
            put("ipr_j", round(float(v), 5))
        st.button(f"Use PI {pia:.5f} in the IPR (P.I. Entry)", on_click=_use_pi, key="acid_use_pi",
                  help="Sets the IPR to P.I. Entry with this PI; compare the system before and after (Calculation "
                       "tab, a sensitivity on the PI).")


def cbm_inputs():
    """Gas and water inputs of the CBM PVT panel (PVTInputs: the oil fields are unused)."""
    g = lambda k, d: float(val(k, d))
    return bo.PVTInputs(rsb=1.0, api=40.0, gas_sg=g("pvt_gg", 0.6), salinity_ppm=g("pvt_sal", 100000.0),
                        h2s=g("pvt_h2s", 0.0), co2=g("pvt_co2", 0.0), n2=g("pvt_n2", 0.0))


def cbm_pvt_panel():
    """PROSPER PVT input for a CBM producer: gas gravity, CBM GLR, water salinity and impurities."""
    st.subheader("Input parameters")
    st.caption("Coalbed methane (PROSPER Tutorial 45): the well produces water with desorbed gas. All the gas is "
               "free gas (no dissolved gas); the water uses PROSPER's water properties and the gas the gas "
               "correlations (Z factor, Lee viscosity).")
    c = st.columns(4)
    num(c[0], "Gas Gravity (sp. gravity)", "pvt_gg", 0.6, 0.55, 2.0, 0.01, "%.3f")
    num(c[1], "CBM GLR (scf/STB)", "pvt_cbm_glr", 10.0, 0.0, 1e6, 1.0)
    num(c[2], "Water Salinity (ppm)", "pvt_sal", 100000.0, 0.0, 300000.0, 1000.0, "%.0f")
    t_ = num(c[3], "Temperature for the table (deg F)", "pvt_t", 100.0, 32.0, 600.0, 1.0)
    c = st.columns(4)
    num(c[0], "Mole Percent H2S (percent)", "pvt_h2s", 0.0, 0.0, 100.0, 0.1)
    num(c[1], "Mole Percent CO2 (percent)", "pvt_co2", 0.0, 0.0, 100.0, 0.1)
    num(c[2], "Mole Percent N2 (percent)", "pvt_n2", 0.0, 0.0, 100.0, 0.1)
    inp = cbm_inputs()
    pp = np.linspace(14.7, 3014.7, 16)
    z, bg, rho_g, mu_g = bo.gas_props(inp, pp, np.full_like(pp, float(t_)))
    bw, rho_w, mu_w = bo.water_props(inp, pp, float(t_))
    tab = pd.DataFrame({"Pressure (psia)": pp, "Z factor": z, "Gas FVF (ft3/scf)": bg, "Gas density (lb/ft3)": rho_g,
                        "Gas viscosity (cp)": mu_g, "Water FVF (RB/STB)": np.broadcast_to(bw, pp.shape),
                        "Water density (lb/ft3)": np.broadcast_to(rho_w, pp.shape),
                        "Water viscosity (cp)": np.broadcast_to(mu_w, pp.shape)})
    st.subheader("Calculated properties")
    st.dataframe(tab.style.format(precision=5).format("{:,.1f}", subset=["Pressure (psia)"]), hide_index=True,
                 width="stretch", height=320)
    st.download_button("Download CBM PVT table (CSV)", tab.to_csv(index=False).encode(), "cbm_pvt.csv")


def cbm_ipr_panel(o):
    """PROSPER's CBM Producer IPR (Tutorial 45): water PI x krw at the saturation that gives the CBM GLR."""
    st.caption("CBM Producer: the water rate is PI x krw(Sw) x (Pr - Pwf), where Sw is the water saturation whose "
               "fractional flow gives the CBM GLR, (krg / mu_g Bg) / (krw / mu_w Bw) = GLR, at the average of Pr and "
               f"Pwf (Corey relative permeabilities). Pressures in {ss['_units']['pressure']}.")
    st.subheader("Reservoir data")
    c = st.columns(4)
    pr = num(c[0], "Reservoir pressure (psia)", "ipr_pr", 814.7, 15.0, 30000.0, 10.0)
    t_f = num(c[1], "Reservoir temperature (deg F)", "ipr_t", 100.0, 32.0, 500.0, 1.0)
    glr = num(c[2], "CBM GLR (scf/STB)", "ipr_cbm_glr", float(val("pvt_cbm_glr", 10.0)), 0.0, 1e6, 1.0)
    pi = num(c[3], "Productivity index (STB/d/psi)", "ipr_j", 2.0, 1e-6, 1e5, 0.1, "%.4g")
    st.subheader("Relative permeability")
    c = st.columns(6)
    swr = num(c[0], "Residual water saturation", "ipr_cbm_swr", 0.2, 0.0, 0.95, 0.01, "%.3g")
    sgr = num(c[1], "Residual gas saturation", "ipr_cbm_sgr", 0.0, 0.0, 0.95, 0.01, "%.3g")
    kw = num(c[2], "Water end-point krw", "ipr_cbm_krw", 0.8, 0.01, 1.0, 0.05, "%.3g")
    kg = num(c[3], "Gas end-point krg", "ipr_cbm_krg", 0.8, 0.01, 1.0, 0.05, "%.3g")
    nw = num(c[4], "Corey exponent (water)", "ipr_cbm_nw", 1.5, 0.1, 10.0, 0.1, "%.3g")
    ng = num(c[5], "Corey exponent (gas)", "ipr_cbm_ng", 2.0, 0.1, 10.0, 0.1, "%.3g")
    inp = cbm_inputs()
    if swr + sgr >= 1.0:
        st.error("The residual saturations must add to less than 1.")
        ss.pop("ipr_curve", None)
        return
    args_ = (inp.gas_sg, inp.salinity_ppm, inp.h2s, inp.co2, inp.n2, pr, t_f, glr, pi, swr, sgr, kw, kg, nw, ng)
    try:
        curve = wcbm.cbm_ipr(*args_)
    except (ValueError, RuntimeError) as e:
        st.warning(f"IPR failed: {e}")
        ss.pop("ipr_curve", None)
        return
    ss["ipr_curve"] = curve
    ss["ipr_recipe"], ss["ipr_sc_dp"], ss["ipr_pvt_ctx"] = (wcbm.cbm_ipr, args_, {}), None, None
    ss["ipr_meta"] = {"model": "CBM Producer", "fluid": "oil", "skin_parts": {}, "pb": None}
    st.subheader("Results")
    m = st.columns(4)
    m[0].metric("AOF", f"{curve.aof:,.1f} STB/d")
    m[1].metric("Sw at Pr", f"{curve.params['Sw at Pr']:.4f}")
    m[2].metric("krw at Pr", f"{curve.params['krw at Pr']:.4f}")
    m[3].metric("Gas at AOF", f"{curve.aof * glr / 1e6:.4f} MMscf/d")
    st.pyplot(ipr_plot(curve, None), clear_figure=True)
    tab = pd.DataFrame({"Water rate (STB/d)": curve.q, "Gas rate (MMscf/d)": curve.q * glr / 1e6,
                        "Pwf (psia)": curve.pwf})
    st.dataframe(tab.style.format(precision=4), hide_index=True, width="stretch")
    st.download_button("Download IPR (CSV)", tab.to_csv(index=False).encode(), "cbm_ipr.csv")


def steam_calculator_panel():
    """PROSPER PVT | Steam Calculator (Tutorial 42): single point from pressure and enthalpy (or temperature), and
    saturation properties over a pressure range. IAPWS-95 steam tables."""
    st.subheader("Steam calculator")
    if not wsteam.available():
        st.error("Steam calculations need the CoolProp package (pip install CoolProp).")
        return
    st.caption("Steam tables (IAPWS-95): saturation temperature at a pressure, and the phase, temperature and steam "
               "quality for an enthalpy - the quality is the steam fraction of the mass, 0 all water and 1 all steam. "
               "Enthalpy reference: saturated liquid at the triple point = 0 (as PROSPER). Pressures in psia.")
    c = st.columns(4)
    mode = sel(c[0], "Calculate", ["Single point", "Range (saturation)"], "steam_calc_mode")
    if mode == "Single point":
        by = sel(c[1], "Input", ["Pressure and enthalpy", "Pressure and temperature"], "steam_calc_by")
        p_ = num(c[2], "Pressure (psia)", "steam_calc_p", 2214.7, 1.0, 3200.0, 10.0)
        if by == "Pressure and enthalpy":
            h_ = num(c[3], "Enthalpy (BTU/lb)", "steam_calc_h", 1100.0, 0.0, 1600.0, 10.0)
            tab = wsteam.calculator(p_, h=h_)
        else:
            t_ = num(c[3], "Temperature (°F)", "steam_calc_t", 650.0, 33.0, 1500.0, 1.0)
            tab = wsteam.calculator(p_, t_f=t_)
        tab["Value"] = [f"{v_:,.6g}" if isinstance(v_, (int, float, np.floating)) else str(v_) for v_ in tab["Value"]]
        st.dataframe(tab, hide_index=True, width="stretch", height=35 * (len(tab) + 1) + 3)
    else:
        c2 = st.columns(4)
        p0 = num(c2[0], "First pressure (psia)", "steam_rng_p0", 100.0, 1.0, 3200.0, 10.0)
        p1 = num(c2[1], "Last pressure (psia)", "steam_rng_p1", 3000.0, 1.0, 3200.0, 10.0)
        n_ = int(num(c2[2], "Number of pressures", "steam_rng_n", 30.0, 2.0, 200.0, 1.0, "%.0f"))
        if p1 > p0:
            tab = wsteam.saturation_table(np.linspace(p0, p1, n_))
            st.dataframe(tab.style.format(precision=4), hide_index=True, width="stretch", height=320)
            st.download_button("Download steam table (CSV)", tab.to_csv(index=False).encode(), "steam_table.csv")


def gas_pvt_panel(opts_now):
    """PROSPER PVT | Input Data for Dry and Wet Gas (black oil)."""
    st.subheader("Input data")
    st.caption("Condensate drops out at the separator only: in the well it is carried as an equivalent gas "
               "(wellstream gravity from the separator gas, CGR and condensate gravity) - a single gas phase plus "
               "free water. Z: Sutton pseudo-criticals with Wichert-Aziz impurity correction and Dranchuk-Abou-Kassem "
               "(PROSPER: Standing-Katz with Wichert-Aziz). For liquid dropout in the well choose Retrograde "
               "condensate.")
    c = st.columns(4)
    num(c[0], "Gas Gravity (sp. gravity)", "pvt_gas_sg", 0.70, 0.55, 2.0, 0.01, "%.3f",
        help="Separator gas gravity (air = 1).")
    num(c[1], "Separator Pressure (psia)", "pvt_sep_p", 214.7, 14.7, 5000.0, 10.0)
    num(c[2], "Condensate Gas Ratio (STB/MMscf)", "pvt_cgr", 0.0, 0.0, 1e4, 1.0)
    num(c[3], "Condensate Gravity (API)", "pvt_cond_api", 50.0, 20.0, 90.0, 0.5)
    c = st.columns(4)
    num(c[0], "Water Gas Ratio (STB/MMscf)", "pvt_wgr", 0.0, 0.0, 1e5, 1.0)
    num(c[1], "Water Salinity (ppm)", "pvt_sal", INPUT_DEFAULTS["pvt_sal"], 0.0, 300000.0, 1000.0, "%.0f")
    sel(c[2], "Gas Viscosity Correlation", ["Lee et al", "Carr et al"], "pvt_gvisc",
        help="Lee, Gonzalez & Eakin (default, wider range) or Carr, Kobayashi & Burrows.")
    c = st.columns(4)
    num(c[0], "Mole Percent H2S (percent)", "pvt_h2s", 0.0, 0.0, 100.0, 0.1)
    num(c[1], "Mole Percent CO2 (percent)", "pvt_co2", 0.0, 0.0, 100.0, 0.1)
    num(c[2], "Mole Percent N2 (percent)", "pvt_n2", 0.0, 0.0, 100.0, 0.1)
    inp = gas_pvt_inputs(opts_now)
    c[3].metric("Wellstream gas gravity", f"{float(inp.wellstream_sg()):.4f}")
    for e in inp.check():
        st.error(e)
    st.subheader("Reservoir data")
    c = st.columns(4)
    num(c[0], "Reservoir Temperature (deg F)", "pvt_t", 200.0, 40.0, 500.0, 1.0)
    num(c[1], "Reservoir Pressure (psia)", "pvt_pres", 3014.7, 14.7, 30000.0, 10.0)
    inp = gas_pvt_inputs(opts_now)
    if gas_vapour_on(opts_now):
        wmin = wgp.minimum_wgr(inp)
        c[2].metric("Minimum WGR", f"{wmin:.5f} STB/MMscf",
                    help="Water vaporised in the gas at reservoir pressure and temperature (McKetta-Wehe with a "
                         "salinity correction). A smaller WGR in a calculation is replaced by it.")
        with st.expander("Water vapour plot"):
            fig, ax = plt.subplots(figsize=(8, 4.2))
            pp = np.linspace(400.0, max(inp.p_res, 1000.0), 60) + 14.696
            for i_, tt in enumerate(np.linspace(60.0, inp.t_res, 7)):
                ax.plot(pp - 14.696, wgp.saturated_wgr(pp, tt, inp.salinity_ppm), color=wplt.SERIES[i_ % 4],
                        lw=1.5, label=f"{tt:.1f} °F")
            ax.set_xlabel("Pressure (psig)")
            ax.set_ylabel("Saturated WGR (STB/MMscf)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
    else:
        c[2].caption("Condensed water vapour is off (Options summary: Water vapour).")
    st.subheader("Matching")
    st.caption("Laboratory (flash) data at one or more temperatures. Each correlation is fitted as Parameter 1 x "
               "correlation + Parameter 2 (least squares): Z factor, gas FVF (from the matched Z) and both viscosity "
               "correlations - the one selected above is used. Leave cells blank where there is no data.")
    if "pvt_gas_match" not in ss:
        ss["pvt_gas_match"] = pd.DataFrame({c_: pd.Series([np.nan] * 8, dtype=float) for c_ in GAS_MATCH_COLS},
                                           index=pd.RangeIndex(1, 9, name="Point"))
    track("pvt_gas_match", st.data_editor(ss["pvt_gas_match"], key=f"pvt_gas_match_ed_{ss.get('_ver_pvt_gas_match', 0)}",
                                          num_rows="dynamic", width="stretch", column_config={
        c_: st.column_config.NumberColumn(c_, format="%g") for c_ in GAS_MATCH_COLS}))
    mt = gas_match_tables()
    flag(st.toggle, "Apply match parameters", "pvt_gas_use_match", True,
         help="On: every calculation (PVT table, IPR, VLP, system) uses the matched correlations.")
    if opts_now is not None and opts_now.well_type == "gas_injector":
        flag(st.toggle, "Injector calculations as PROSPER (impurities left out)", "inj_gas_prosper", False,
             help="PROSPER's gas injector (traverse and injectivity) works out the gas properties from the gravity "
                  "alone, without the N2 / CO2 / H2S correction its PVT tab applies. On: the same, to compare with "
                  "PROSPER. Off: the impurities are applied everywhere (physical; with 4 % N2 the gas is about 3 % "
                  "denser in the tubing and the injectivity changes).")
    if mt:
        raw = gas_pvt_inputs(opts_now, matched=False)
        mres = wgp.match_gas(raw, mt)
        rows_ = [{"Property": wgp.MATCH_PROPS[k], "Parameter 1": m_.p1, "Parameter 2": m_.p2,
                  "Std deviation (relative)": m_.std, "Error before match (%)": m_.raw_error_pct, "Points": m_.n}
                 for k, m_ in mres.items()]
        if val("pvt_gas_use_match", True):
            st.success("PVT is matched - the correlations below are applied with these parameters.")
            dev_ = wgp.fvf_consistency(dataclasses.replace(raw, match=mres))
            if dev_ > 0.10:
                st.warning(f"The matched gas FVF departs from the matched Z (Bg = 0.02827 Z T/p) by up to "
                           f"{dev_:.0%} below reservoir pressure. The FVF sets the gas volume and density in the "
                           f"well, so this changes the friction loss and the rate - check the lab FVF data.")
        st.dataframe(pd.DataFrame(rows_).style.format({"Parameter 1": "{:.5f}", "Parameter 2": "{:.4e}",
                                                       "Std deviation (relative)": "{:.3e}",
                                                       "Error before match (%)": "{:.3f}"}),
                     hide_index=True, width="stretch")
        mi = dataclasses.replace(raw, match=mres)
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
        for i_, (tt, g_) in enumerate(mt):
            pp = np.linspace(max(14.7, g_["Pressure (psia)"].min() * 0.5), g_["Pressure (psia)"].max() * 1.05, 60)
            r0 = wgp.raw_props(raw, pp, np.full(pp.size, tt))
            r1 = wgp.gas_props(mi, pp, np.full(pp.size, tt))
            mu0 = r0[4] if raw.visc == "carr" else r0[3]
            for ax, col, y0, y1 in ((axs[0], "Z factor", r0[0], r1[0]), (axs[1], "Gas viscosity (cp)", mu0, r1[3]),
                                    (axs[2], "Gas FVF (ft3/scf)", r0[1], r1[1])):
                cc = wplt.SERIES[i_ % 4]
                ax.plot(pp, y0, color=cc, lw=1.1, ls="--", label=f"{tt:g} °F correlation")
                ax.plot(pp, y1, color=cc, lw=1.8, label=f"{tt:g} °F matched")
                ax.scatter(g_["Pressure (psia)"], g_[col], color=wplt.INK, s=28, zorder=4, label="lab data" if i_ == 0 else None)
                ax.set_title(col, fontsize=9)
                ax.set_xlabel("Pressure (psia)", fontsize=8)
                ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        axs[2].set_yscale("log")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.caption("No match data - the correlations are used as published.")
    inp = gas_pvt_inputs(opts_now)
    st.subheader("Calculate")
    c = st.columns(4)
    temps = num_list(c[0], "Temperatures (°F)", "pvt_gas_temps", f"60, {inp.t_res:g}", "comma separated")
    p_lo = num(c[1], "First pressure (psia)", "pvt_gas_p0", 29.7, 14.7, 30000.0, 10.0)
    p_hi = num(c[2], "Last pressure (psia)", "pvt_gas_p1", max(inp.p_res + 14.696, 1000.0), 14.7, 30000.0, 10.0)
    npts = int(num(c[3], "Number of pressures", "pvt_gas_n", 20.0, 2.0, 100.0, 1.0, "%.0f"))
    if temps and p_hi > p_lo:
        tab = wgp.pvt_table(inp, temps, np.linspace(p_lo, p_hi, npts) - 14.696)
        tab.insert(1, "Pressure (psia)", tab["Pressure (psig)"] + 14.696)
        tab = tab.drop(columns="Pressure (psig)")
        if not gas_vapour_on(opts_now):
            tab = tab.drop(columns="Free water (STB/MMscf)")
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
        for i_, tt in enumerate(sorted(set(tab["Temperature (°F)"]))):
            d_ = tab[tab["Temperature (°F)"] == tt]
            for ax, col in zip(axs, ("Z factor", "Gas FVF (ft3/scf)", "Gas viscosity (cp)", "Gas density (lb/ft3)")):
                ax.plot(d_["Pressure (psia)"], d_[col], color=wplt.SERIES[i_ % 4], lw=1.6, label=f"{tt:g} °F")
                ax.set_title(col, fontsize=9)
                ax.set_xlabel("Pressure (psia)", fontsize=8)
                ax.grid(alpha=0.3)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.dataframe(tab.style.format(precision=5), hide_index=True, width="stretch", height=320)
        st.download_button("Download PVT table (CSV)", tab.to_csv(index=False).encode(), "gas_pvt.csv")


def retro_pvt_panel(opts_now):
    """PROSPER PVT | Input Data for Retrograde Condensate with the black-oil method."""
    st.subheader("Input data")
    st.caption("PROSPER's black-oil retrograde condensate model (reconstructed from its PVT tables, Tutorial 38 - see "
               "fluxwell/retrograde.py): below the dew point condensate drops out of the gas and flows as liquid in "
               "the well. The separator and tank figures describe the full stream brought to stock-tank conditions "
               "(PROSPER's Tutorial 38: separator at 0 psig and 60 °F, tank GOR 1 scf/STB).")
    c = st.columns(4)
    num(c[0], "Separator Pressure (psia)", "pvt_sep_p", 14.696, 14.696, 5000.0, 10.0)
    num(c[1], "Separator Temperature (deg F)", "pvt_sep_t", 60.0, 0.0, 300.0, 1.0)
    num(c[2], "Separator GOR (scf/STB)", "pvt_sep_gor", 7000.0, 1.0, 1e7, 10.0)
    num(c[3], "Separator Gas Gravity (sp. gravity)", "pvt_gas_sg", 0.70, 0.55, 2.0, 0.001, "%.4f")
    c = st.columns(4)
    num(c[0], "Tank GOR (scf/STB)", "pvt_tank_gor", 0.0, 0.0, 1e6, 1.0)
    num(c[1], "Tank Gas Gravity (sp. gravity)", "pvt_tank_sg", 0.70, 0.55, 3.0, 0.001, "%.4f")
    num(c[2], "Condensate Gravity (API)", "pvt_cond_api", 50.0, 10.0, 90.0, 0.1)
    num(c[3], "Water Gas Ratio (STB/MMscf)", "pvt_wgr", 0.0, 0.0, 1e5, 1.0)
    c = st.columns(4)
    num(c[0], "Water Salinity (ppm)", "pvt_sal", INPUT_DEFAULTS["pvt_sal"], 0.0, 300000.0, 1000.0, "%.0f")
    num(c[1], "Mole Percent H2S (percent)", "pvt_h2s", 0.0, 0.0, 100.0, 0.1)
    num(c[2], "Mole Percent CO2 (percent)", "pvt_co2", 0.0, 0.0, 100.0, 0.1)
    num(c[3], "Mole Percent N2 (percent)", "pvt_n2", 0.0, 0.0, 100.0, 0.1)
    st.subheader("Reservoir data")
    c = st.columns(4)
    num(c[0], "Reservoir Temperature (deg F)", "pvt_t", 200.0, 61.0, 500.0, 1.0)
    num(c[1], "Reservoir Pressure (psia)", "pvt_pres", 3014.7, 14.7, 30000.0, 10.0)
    num(c[2], "Dew Point at Reservoir Temperature (psia)", "pvt_dew", 4014.7, 14.7, 30000.0, 10.0)
    ri = retro_inputs()
    for e in ri.check():
        st.error(e)
    if ri.check():
        return
    c[3].metric("Condensate-gas ratio", f"{ri.cgr:.2f} STB/MMscf")
    if ri.p_res < ri.p_dew:
        st.info("The reservoir pressure is below the dew point: condensate has dropped out in the reservoir too "
                "(the IPR uses the full wellstream).")
    st.caption("Not matched to laboratory data - as PROSPER advises for this model, validate it against an equation "
               "of state instead (Tutorial 38: compare gradient traverses with the EoS model).")
    st.subheader("Calculate")
    c = st.columns(4)
    temps = num_list(c[0], "Temperatures (°F)", "pvt_retro_temps", f"70, {(70 + ri.t_res) / 2:g}, {ri.t_res:g}",
                     "comma separated")
    p_lo = num(c[1], "First pressure (psia)", "pvt_retro_p0", 14.7, 14.7, 30000.0, 10.0)
    p_hi = num(c[2], "Last pressure (psia)", "pvt_retro_p1", max(ri.p_res, ri.p_dew) * 1.1, 14.7, 30000.0, 10.0)
    npts = int(num(c[3], "Number of pressures", "pvt_retro_n", 25.0, 2.0, 100.0, 1.0, "%.0f"))
    if temps and p_hi > p_lo:
        tab = wrg.pvt_table(ri, temps, np.linspace(p_lo, p_hi, npts) - 14.696)
        tab.insert(1, "Pressure (psia)", tab["Pressure (psig)"] + 14.696)
        tab.insert(3, "Dew point (psia)", tab["Dew point (psig)"] + 14.696)
        tab = tab.drop(columns=["Pressure (psig)", "Dew point (psig)"])
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
        for i_, tt in enumerate(sorted(set(tab["Temperature (deg F)"]))):
            d_ = tab[tab["Temperature (deg F)"] == tt]
            for ax, col in zip(axs, ("Reservoir CGR (STB/MMscf)", "Z factor", "Gas density (lb/ft3)",
                                     "Oil FVF (RB/STB)")):
                ax.plot(d_["Pressure (psia)"], d_[col], color=wplt.SERIES[i_ % 4], lw=1.6, label=f"{tt:g} °F")
                ax.set_title("Condensate dropout (STB/MMscf)" if col.startswith("Reservoir CGR") else col, fontsize=9)
                ax.set_xlabel("Pressure (psia)", fontsize=8)
                ax.grid(alpha=0.3)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.dataframe(tab.style.format(precision=5, na_rep="—"), hide_index=True, width="stretch", height=320)
        st.download_button("Download PVT table (CSV)", tab.to_csv(index=False).encode(), "retrograde_pvt.csv")

# ---------------------------------------------------------------- emulsion viscosity (PROSPER Tutorial 46)
EMUL_DEFAULT = pd.DataFrame({"Water cut (%)": [5.0, 10, 20, 30, 40, 50, 55, 85, 90],
                             "Emulsion viscosity (cp)": [105.0, 115, 130, 150, 188, 234, 250, 53, 10.5]})


def emulsion_panel(model):
    """Emulsion data and match (PVT Input Data, Other Data - Emulsion). Stores ss["emulsion_model"]."""
    st.subheader("Emulsion")
    st.caption("PROSPER's emulsion viscosity: the oil viscosity times P1 + P2·WC + P3·WC² + P4·WC³ + P5·WC⁴ (WC as a "
               "fraction) up to the minimum water cut for maximum viscosity, flat to the maximum water cut, then "
               "falling to the water viscosity at 100 % (log-linear). Match P2-P5 to measured viscosities at one "
               "pressure and temperature; the oil-continuous points are used.")
    c = st.columns(4)
    occ = sel(c[0], "Emulsion occurrence", list(wem.OCCURRENCE), "emul_occ", "tubing_pipe",
              format_func=wem.OCCURRENCE.get)
    t_e = num(c[1], "Experimental temperature (°F)", "emul_t", 200.0, 0.0, 600.0, 5.0)
    p_e = num(c[2], "Experimental pressure (psig)", "emul_p", 2000.0, 0.0, 20000.0, 50.0)
    c2 = st.columns(4)
    wmin = num(c2[0], "Minimum water cut for maximum viscosity (%)", "emul_wcmin", 60.0, 0.0, 100.0, 1.0)
    wmax = num(c2[1], "Maximum water cut for maximum viscosity (%)", "emul_wcmax", 80.0, 0.0, 100.0, 1.0)
    npar = int(sel(c2[2], "Parameters to match", [2, 3, 4, 5], "emul_npar", 3,
                   help="P1 is fixed at 1 (the oil viscosity at zero water cut); PROSPER's default matches P2, P3."))
    key = "emul_table"
    if key not in ss:
        ss[key] = EMUL_DEFAULT.copy()
    tbl = track(key, st.data_editor(ss[key], key=key + "_ed", num_rows="dynamic", width="stretch"))
    mu_o = float(model.table(t_e, [p_e + 14.696])["Oil viscosity (cp)"].iloc[0])
    st.markdown(f"Oil viscosity at {p_e:g} psig and {t_e:g} °F: **{mu_o:.4g} cp** (the match reference)")
    wc_d = pd.to_numeric(tbl["Water cut (%)"], errors="coerce").to_numpy(float)
    mu_d = pd.to_numeric(tbl["Emulsion viscosity (cp)"], errors="coerce").to_numpy(float)
    if st.button("Match emulsion parameters", key="emul_match"):
        m = wem.match(wc_d, mu_d, mu_o, wmin / 100.0, npar)
        for i, v in enumerate(m.params):
            put(f"emul_p{i + 1}", float(v))
        ss["_v_emul_std"] = m.std
        st.rerun()
    c3 = st.columns(6)
    pars = tuple(num(c3[i], f"P{i + 1}", f"emul_p{i + 1}", 1.0 if i == 0 else 0.0, None, None, None, "%.5f")
                 for i in range(5))
    std = val("emul_std", None)
    if std is not None and np.isfinite(float(std)):
        c3[5].metric("Std deviation (cp)", f"{float(std):.4f}")
    em = wem.EmulsionModel(pars, wmin / 100.0, max(wmax, wmin) / 100.0, occ)
    ss["emulsion_model"] = em
    x = np.linspace(0, 1, 201)
    mu_w = float(model.table(t_e, [p_e + 14.696])["Water viscosity (cp)"].iloc[0])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(x * 100, em.viscosity(x, mu_o, mu_w), color=wplt.SERIES[0], lw=2, label="model")
    ok = np.isfinite(wc_d) & np.isfinite(mu_d)
    ax.scatter(wc_d[ok], mu_d[ok], s=30, color=wplt.INK, zorder=4, label="measured")
    ax.set_xlabel("Water cut (%)")
    ax.set_ylabel("Emulsion viscosity (cp)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    if occ == "none":
        st.info("Emulsion occurrence is No emulsion corrections: the liquid viscosity stays the volume-weighted "
                "oil-water viscosity.")


# ---------------------------------------------------------------- non-Newtonian oil (PROSPER Tutorial 44)
RHEO_DEFAULT = pd.DataFrame({wrh.COLUMNS[0]: [60.0, 60.0, 120.0, 120.0],
                             wrh.COLUMNS[1]: [100.0, 5000.0, 100.0, 5000.0],
                             wrh.COLUMNS[2]: [0.0, 0.0, 0.0, 0.0],
                             wrh.COLUMNS[3]: [1.0, 1.0, 0.5, 0.5],
                             wrh.COLUMNS[4]: [0.9, 0.9, 0.95, 0.95]})


def rheology_current():
    """The Herschel-Bulkley model of the PVT tab, or None when the table is empty."""
    df = ss.get("_tables", {}).get("pvt_rheology", ss.get("pvt_rheology"))
    if df is None:
        return None
    try:
        return wrh.Rheology.from_table(pd.DataFrame(df), float(val("rheo_mu_max", 10000.0)))
    except ValueError:
        return None


def rheology_panel(model):
    """Rheological parameters (PVT Input Data, Non-Newtonian). Stores the table in ss["pvt_rheology"]."""
    st.subheader("Non-Newtonian rheology")
    st.caption("Herschel-Bulkley oil: shear stress τ = τo + K·γⁿ. The apparent viscosity at the wall shear rate of "
               "laminar pipe flow (Metzner-Reed, γ = 8v/D) is μ = K·((3n+1)/4n)ⁿ·γⁿ⁻¹ + τo/γ, capped at the maximum "
               "viscosity. K in Pa·sⁿ, τo in psi; the parameters are interpolated linearly in pressure, then in "
               "temperature. The apparent viscosity replaces the oil viscosity in the tubing.")
    c = st.columns(4)
    num(c[0], "Maximum viscosity (cp)", "rheo_mu_max", 10000.0, 1.0, 1e7, 100.0)
    key = "pvt_rheology"
    if key not in ss:
        ss[key] = RHEO_DEFAULT.copy()
    tbl = track(key, st.data_editor(ss[key], key=key + "_ed", num_rows="dynamic", width="stretch"))
    try:
        rh = wrh.Rheology.from_table(pd.DataFrame(tbl), float(val("rheo_mu_max", 10000.0)))
    except ValueError as e:
        st.warning(str(e))
        return
    c2 = st.columns(4)
    t_r = num(c2[0], "Temperature for the plot (°F)", "rheo_t", 120.0, None, None, 5.0)
    p_r = num(c2[1], "Pressure for the plot (psig)", "rheo_p", 500.0, 0.0, None, 50.0)
    d_r = num(c2[2], "Pipe inside diameter (in)", "rheo_d", 3.2, 0.1, 60.0, 0.1)
    v = np.geomspace(0.05, 50, 120)
    mu_n = float(model.table(t_r, [p_r + 14.696])["Oil viscosity (cp)"].iloc[0])
    mu = [rh.apparent_viscosity(p_r + 14.696, t_r, x, d_r / 12.0) for x in v]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(v, mu, color=wplt.SERIES[0], lw=2, label="non-Newtonian (apparent)")
    ax.axhline(mu_n, color=wplt.SERIES[1], lw=1.5, ls="--", label="Newtonian (PVT correlation)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mixture velocity (ft/s)")
    ax.set_ylabel("Oil viscosity (cp)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


HYD_COLUMNS = ["Pressure (psig)", "Temperature (°F)"]
SALT_COLUMNS = ["Salt", "Weight %"]


def _blank(cols, n, text=()):
    df = pd.DataFrame({c: pd.Series([None] * n, dtype=object) if c in text else pd.Series([np.nan] * n, dtype=float)
                       for c in cols})
    df.index = pd.RangeIndex(1, n + 1, name="Point")
    return df


def _vtable(key, cols, n, column_config=None, text=()):
    """A table editor kept in the project file (ss[key]); ss['_ver_'+key] bumps to reload it after a programmatic
    change."""
    if key not in ss:
        ss[key] = _blank(cols, n, text)
    return track(key, st.data_editor(ss[key], key=f"{key}_ed_{ss.get('_ver_' + key, 0)}", num_rows="fixed",
                                     width="stretch", column_config=column_config or {}))


def hydrate_curve_current():
    """The hydrate formation curve of the PVT warnings ((psia...), (F...)) when the warning is on, else None."""
    if not bool(val("pvt_hyd_on", False)):
        return None
    df = ss.get("_tables", {}).get("pvt_hydrate", ss.get("pvt_hydrate"))
    return whyd.table_curve(df)


@st.cache_data(show_spinner=False)
def hydrate_curve_eos(text, name, salts, salinity, t0, t1, dt):
    f = wm_eos.EOSFluid(wm_eos.parse_prp(text, name))
    m = whyd.HydrateModel(f, dict(salts) if salts else None, None if salts else salinity)
    return m.curve(np.arange(t0, t1 + 1e-9, dt)), m.ln_aw


def hydrate_panel(opts_now):
    """PROSPER PVT | Warnings - hydrate formation (Tutorial 43): a hydrate curve the calculations flag against
    (gradient nodes, VLP rates, system solution), entered or calculated from the equation of state with the water's
    salts as inhibitors."""
    st.divider()
    st.subheader("PVT warnings - hydrate formation")
    st.caption("With the warning on, every gradient node whose temperature is at or below the hydrate temperature at "
               "its pressure is flagged (gradient and VLP results, system solution). Enter the curve - e.g. from a "
               "laboratory or an equation-of-state calculation - or calculate it below from the PRP composition.")
    flag(st.checkbox, "Enable hydrate formation warning", "pvt_hyd_on", False)
    cc = st.columns([1, 1.4])
    with cc[0]:
        hyd = _vtable("pvt_hydrate", HYD_COLUMNS, 25,
                      {c_: st.column_config.NumberColumn(c_, format="%g") for c_ in HYD_COLUMNS})
    cur = whyd.table_curve(hyd)
    with cc[1]:
        if cur is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(cur[1], np.asarray(cur[0]) - 14.696, color=wplt.SERIES[0], lw=1.8, marker="o", ms=3)
            ax.fill_betweenx(np.asarray(cur[0]) - 14.696, min(cur[1]) - 5, cur[1], color=wplt.SERIES[0], alpha=0.08)
            ax.set_xlabel("Temperature (°F)")
            ax.set_ylabel("Pressure (psig)")
            ax.set_title("Hydrate formation curve (hydrates to the left)", fontsize=10)
            ax.grid(alpha=0.3, ls="--")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
    if opts_now is None or opts_now.method != "eos":
        st.caption("Black-oil PVT: enter the curve (as PROSPER: calculated externally, e.g. with an equation of "
                   "state, and pasted here).")
        return
    if not ss.get("prp_text"):
        st.info("Load a PRP file above to calculate the hydrate curve from the composition.")
        return
    with st.expander("Calculate the hydrate curve from the composition (salts as inhibitors)", expanded=False):
        st.caption("van der Waals - Platteeuw hydrate model (Munck et al. Langmuir constants, structures I and II), "
                   "fugacities from the PRP equation of state, water activity of the brine from Pitzer's model. "
                   "Calibrated to PROSPER's curves of the user's Tutorial 43 file within 3.4 % in pressure.")
        c = st.columns(4)
        mode = sel(c[0], "Salt mode", ["Use water composition", "Use salinity (as NaCl)"], "hyd_salt_mode")
        t0 = num(c[1], "First temperature (°F)", "hyd_t0", 35.0, 33.0, 120.0, 1.0)
        t1 = num(c[2], "Last temperature (°F)", "hyd_t1", 95.0, 34.0, 130.0, 1.0)
        dt = num(c[3], "Step (°F)", "hyd_dt", 3.0, 0.5, 20.0, 0.5)
        salts, sal = (), None
        if mode == "Use water composition":
            ed = _vtable("hyd_salts", SALT_COLUMNS, 8,
                         {"Salt": st.column_config.SelectboxColumn("Salt", options=list(whyd.SALTS)),
                          "Weight %": st.column_config.NumberColumn("Weight %", format="%g")}, text=("Salt",))
            rows = [(r["Salt"], float(r["Weight %"])) for _, r in ed.iterrows()
                    if isinstance(r["Salt"], str) and r["Salt"] and pd.notna(pd.to_numeric(r["Weight %"], errors="coerce"))
                    and float(r["Weight %"]) > 0]
            salts = tuple(rows)
            if salts:
                st.caption(f"Total dissolved salts {whyd.salinity_of(dict(salts)):,.0f} ppm.")
        else:
            sal = num(st.columns(4)[0], "Water salinity (ppm, as NaCl)", "hyd_sal",
                      float(val("pvt_sal", 20000.0)), 0.0, 300000.0, 1000.0, "%.0f")
        if st.button("Calculate hydrate curve", key="hyd_go") and t1 > t0:
            try:
                with st.spinner("Flashing the composition along the hydrate curve..."):
                    tab, lnaw = hydrate_curve_eos(ss["prp_text"], ss.get("prp_name", ""), salts, sal, t0, t1, dt)
                ss["hyd_result"] = (tab, lnaw)
            except ValueError as e:
                st.error(str(e))
        res = ss.get("hyd_result")
        if res is not None:
            tab, lnaw = res
            st.markdown(f"Water activity {np.exp(lnaw):.4f}")
            st.dataframe(tab.style.format(precision=2), hide_index=True, width="stretch")
            if st.button("Use as the hydrate warning curve", key="hyd_use") and len(tab):
                g = _blank(HYD_COLUMNS, max(25, len(tab)))
                g.iloc[:len(tab), 0] = tab["Pressure (psig)"].to_numpy()
                g.iloc[:len(tab), 1] = tab["Temperature (°F)"].to_numpy()
                ss["pvt_hydrate"] = g
                ss["_ver_pvt_hydrate"] = ss.get("_ver_pvt_hydrate", 0) + 1
                put("pvt_hyd_on", True)
                st.rerun()


with tabs[1]:
    st.header("PVT Input Data")
    opts_now = ss.get("options")
    if opts_now is not None and opts_now.well_type == "water_injector":
        water_injector_pvt_panel()
    elif opts_now is not None and opts_now.method == "eos":
        render_eos(opts_now)
    elif retro_bo_on(opts_now):
        retro_pvt_panel(opts_now)
    elif opts_now is not None and opts_now.fluid_type == "cbm":
        cbm_pvt_panel()
    elif opts_now is not None and opts_now.fluid_type != "oil":
        gas_pvt_panel(opts_now)
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
        has_data = data.bubble_point is not None or not data.clean().empty
        c = st.columns([1, 1, 4])
        c[0].button("Match", type="primary", on_click=run_match, disabled=bool(problems) or not has_data,
                    width="stretch", help=None if has_data else "Enter a measured bubble point or lab table "
                                                                 "points to match the correlations.")
        c[1].button("Reset match", on_click=reset_match, width="stretch", disabled=ss.get("pvt_match") is None,
                    help="Remove the match: the selected correlations are used as published.")
        c[2].caption("Matching is optional - without it the selected correlations are used as published.")
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
        elif has_data:
            st.info("Press **Match** to compare the correlations with the lab data (optional).")

        # ------------------------------------------------- selection (always available)
        st.subheader("Selected correlations")
        pb_opts = [f for f in bo.PB_FAMILIES if bo.available("pb", f)]
        v_opts = [f for f in bo.VISC_FAMILIES if bo.available("visc", f)]
        c = st.columns([1, 1, 1])
        fam = sel(c[0], "Pb, Rs, Bo correlation", pb_opts, "pvt_sel_pb", format_func=bo.PB_FAMILIES.get)
        vfam = sel(c[1], "Oil viscosity correlation", v_opts, "pvt_sel_visc", format_func=bo.VISC_FAMILIES.get)
        if res is not None:
            use_m = flag(c[2].checkbox, "Apply match parameters", "pvt_use_match")
        else:
            c[2].checkbox("Apply match parameters", value=False, disabled=True, key="pvt_use_match_off",
                          help="Available after a match.")
            use_m = False
        if problems:
            ss.pop("pvt_model", None)
            st.info("Fix the inputs above to calculate the PVT.")
        else:
            params = {}
            if res is not None and use_m:
                params = {p: res.matches[(p, fam)] for p in ("pb", "rs", "bo")}
                params["visc"] = res.matches[("visc", vfam)]
            raw_model = bo.BlackOilModel(inp, fam, vfam, {})
            model = bo.BlackOilModel(inp, fam, vfam, params)
            ss["pvt_model"] = model
            st.caption(f"Used by the IPR, VLP, Calculation and Design tabs: {bo.PB_FAMILIES[fam]} / "
                       f"{bo.VISC_FAMILIES[vfam]}, " + ("with match parameters." if params else "as published (not matched)."))
            if has_data:
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
            if opts_now is not None and getattr(opts_now, "emulsions", "none") == "emulsion":
                emulsion_panel(model)
            if opts_now is not None and getattr(opts_now, "viscosity_model", "newtonian") == "non_newtonian":
                rheology_panel(model)
    if opts_now is not None and opts_now.well_type == "producer":
        hydrate_panel(opts_now)


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


def oil_pvt_at(opts_now, pr, t_f, gor=None):
    """Oil properties at reservoir pressure from the PVT tab. Returns (dict, source text) or (None, reason).
    gor: evaluate the black oil at this producing GOR instead of the PVT's (PROSPER re-evaluates a Darcy-type IPR at
    the system calculation's GOR: Tutorial 29, PI 0.60335 at GOR 150 becomes 0.29004 at GOR 10)."""
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
    label = "Black oil from the PVT tab"
    if model is None:
        inp, _ = current_inputs()
        if inp.check():
            return None, "PVT tab inputs are incomplete."
        model = bo.BlackOilModel(inp, "standing", "beal", {})
        label = "Black oil from PVT tab inputs - not matched yet (Standing / Beal)"
    else:
        label += (f" ({bo.PB_FAMILIES[model.family]} / {bo.VISC_FAMILIES[model.visc_family]}, "
                  f"{'matched' if model.params else 'not matched'})")
    if gor is not None and abs(float(gor) - model.inp.rsb) > 1e-6 and gor > 0:
        model = dataclasses.replace(model, inp=dataclasses.replace(model.inp, rsb=float(gor)))
        label += f", at GOR {float(gor):g} scf/STB"
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
            # above the dew point the reservoir fluid; below it the equilibrium gas (vapour) phase, as PROSPER's
            # M(P) for a condensate (Tutorial 37: M(P) within 0.4 %, AOF within 0.2 %)
            out = []
            for x in np.atleast_1d(p):
                r = f.flash(float(x), t_f)
                out.append(r.vapour if r.phases == "2P" and r.vapour is not None
                           else f.phase_props(f.z, float(x), t_f))
            return np.array([o.viscosity for o in out]), np.array([o.z for o in out])
        return mu_z, float(f.z @ f.mw / 28.9625), (f"Equation of state ({ss.get('prp_name', 'PRP file')}): reservoir "
                                                   "fluid above the dew point, equilibrium gas below it")
    if retro_bo_on(opts_now):
        ri = retro_inputs()

        def mu_z(p):
            # PROSPER's M(P) for the black-oil retrograde condensate: the gas (vapour) phase of its condensate model
            # (Tutorial 41: viscosity and Z in its system table reproduced, M(P) within 0.4 %)
            r_ = wrg.props(ri, np.atleast_1d(np.asarray(p, float)), float(t_f))
            return np.asarray(r_["mu_g"], float), np.asarray(r_["z"], float)
        return mu_z, float(ri.gas_sg), "Black-oil retrograde condensate: gas phase of the condensate model"
    gi = gas_pvt_inputs(opts_now)
    sg = float(gi.wellstream_sg())

    def mu_z(p):
        z, _, _, mu = wgp.gas_props(gi, p, np.full(np.shape(np.atleast_1d(p)), float(t_f)))
        return np.asarray(mu, float), np.asarray(z, float)
    return mu_z, sg, (f"Gas correlations, wellstream gravity {sg:.4f} (Sutton + Wichert-Aziz, Dranchuk-Abou-Kassem, "
                      f"{wgp.VISC_CORRELATIONS[gi.visc]})")



# ---------------------------------------------------------------- IPR helpers
VERTICAL_SKIN_MODELS = {"Darcy", "Fetkovich", "Jones", "Transient", "Back pressure", "Dual Porosity", "Petroleum Experts",
                        "Multiphase Pseudo Pressure"}
HORIZONTAL_MODELS = {"Horizontal Well - No Flow Boundaries", "Horizontal Well - Constant Pressure Upper Boundary",
                     "Horizontal Well - Transverse Vertical Fractures",
                     "Horizontal Well - friction dP"}
OIL_ADV = ["Hydraulically Fractured Well", "Horizontal Well - No Flow Boundaries",
           "Horizontal Well - Constant Pressure Upper Boundary",
           "Horizontal Well - Transverse Vertical Fractures", "Horizontal Well - friction dP",
           "Dual Porosity", "Multi-layer", "Multi-layer with dP loss in wellbore", "Multiphase Pseudo Pressure"]
GAS_ADV = OIL_ADV + ["Modified Isochronal", "Multilateral Well"]
OIL_ADV = OIL_ADV + ["SPOT"]
STILL_NOT = {
    "oil": ["SkinAide (ELF proprietary method)",
            "Thermally Induced Fracture (water injectors - injection IPRs come with injector support)"],
    "gas": ["SkinAide (ELF proprietary method)"],
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


SPOT_RELPERM_COLS = ["Phase", "Residual saturation", "End point relative permeability", "Corey exponent"]


def spot_ui(o, pr, t_f, wc):
    """PROSPER's SPOT IPR (Shell Perforation Optimisation Tool, Tutorial 09): one layer split into 1 ft segments."""
    model = blackoil_model(o)
    if model is None or o.method == "eos":
        raise ValueError("SPOT needs a black-oil PVT (PVT tab).")
    d = wspot.SpotInputs()
    st.caption("PROSPER's SPOT: the perforation penetration from the gun's API RP 19B data corrected for rock "
               "strength and stress, then Karakas & Tariq perforation, McLeod crushed-zone and Cinco-Ley deviation "
               "skins segment by segment (1 ft), combined into the layer skin; the water cut sets the water "
               "saturation (relative permeabilities). Reservoir pressure, water cut and temperature are the ones "
               "above. See spot.py for what was reconstructed and how closely.")
    t1, t2, t3, t4 = st.tabs(["Options", "Layer", "Log data", "Completion"])
    with t1:
        c = st.columns(4)
        wt = sel(c[0], "Well type", ["Deviated", "Vertical"], "spot_wt")
        inflow = sel(c[1], "Inflow equation", ["Fetkovich", "Vogel"], "spot_inflow")
        kcf = num(c[2], "Crushed zone permeability factor (fraction)", "spot_kcf", d.crush_factor, 0.001, 1.0, 0.05)
        kct = num(c[3], "Crushed zone thickness (in)", "spot_kct", d.crush_thickness, 0.0, 10.0, 0.1)
        st.caption("Lp correlation: Conventional. Non-Darcy skin: no. Activity: new well, single-run perforating, no "
                   "invasion, no sanding model, cased and perforated - the options of PROSPER's Tutorial 09.")
    with t2:
        c = st.columns(4)
        rw = num(c[0], "Well radius (ft)", "spot_rw", d.rw, 0.05, 5.0, 0.001, "%.3f")
        re_ = num(c[1], "Drainage radius (ft)", "spot_re", d.re, 10.0, 1e6, 100.0)
        eff = num(c[2], "Perforation efficiency (fraction)", "spot_eff", d.perf_eff, 0.01, 1.0, 0.05)
        kvkh = num(c[3], "Kv/Kh", "spot_kvkh", d.kv_kh, 0.001, 10.0, 0.05)
        c = st.columns(4)
        top = num(c[0], "Top MD (ft)", "spot_top", d.top_md, 0.0, 4e4, 10.0)
        bot = num(c[1], "Bottom MD (ft)", "spot_bot", d.bottom_md, 0.0, 4e4, 10.0)
        ttop = num(c[2], "Top TVD (ft)", "spot_ttop", 9000.0, 0.0, 4e4, 10.0)
        tbot = num(c[3], "Bottom TVD (ft)", "spot_tbot", 9100.0, 0.0, 4e4, 10.0)
        c = st.columns(4)
        ub = num(c[0], "Underbalance pressure (psi)", "spot_ub", d.underbalance, 0.0, 1e4, 50.0)
        obg = num(c[1], "Overburden pressure gradient (psi/ft)", "spot_obg", d.ob_grad, 0.3, 2.0, 0.01)
        gun = sel(c[2], "Perforating gun", list(wspot.GUNS), "spot_gun",
                  help="Baker 2007 Predator XP LS (6 spf, 60 deg, wireline; API RP 19B Lp 19.2 in, EHD 0.23 in) - "
                       "the gun of PROSPER's Tutorial 09; other guns are not available.")
        key = "spot_relperm"
        if key not in ss:
            ss[key] = pd.DataFrame([["Oil", 0.15, 0.8, 1.5], ["Water", 0.2, 0.6, 1.5]], columns=SPOT_RELPERM_COLS)
        st.caption("Relative permeability (Corey)")
        rp = track(key, st.data_editor(ss[key], key=key + "_ed", num_rows="fixed", width="stretch",
                                       disabled=["Phase"]))
    with t3:
        c = st.columns(4)
        kh = num(c[0], "Horizontal permeability (md)", "spot_k", d.k, 1e-3, 1e5, 1.0)
        phi = num(c[1], "Porosity (fraction)", "spot_phi", d.porosity, 0.01, 0.5, 0.01)
        ucs = num(c[2], "Uniaxial compressive strength (psi)", "spot_ucs", d.ucs, 100.0, 30000.0, 500.0)
        st.caption("A pseudo-log is generated at 1 ft (constant permeability, porosity and strength), as PROSPER's "
                   "Generate with no log data available.")
    with t4:
        c = st.columns(4)
        c[0].text_input("Completion", "Casing 6.7 in, 10 lb/ft, J55 (ID 6.41 in)", disabled=True)
        holdup = num(c[1], "Holdup depth (ft MD)", "spot_holdup", d.bottom_md, 0.0, 4e4, 10.0,
                     help="The deepest point the gun can be run to. Above the layer bottom the rest of the layer is not "
                          "perforated (partial penetration); the segment at the holdup depth has a smaller entry hole.")
        st.caption("The completion and gun enter through the entry-hole size (PROSPER's 0.22427 in for this gun in "
                   "this casing); other casings are not distinguished.")
    try:
        rpv = pd.DataFrame(rp)[SPOT_RELPERM_COLS[1:]].apply(pd.to_numeric, errors="coerce").to_numpy()
        relperm = (float(rpv[0, 0]), float(rpv[1, 0]), float(rpv[0, 1]), float(rpv[1, 1]), float(rpv[0, 2]),
                   float(rpv[1, 2]))
    except (KeyError, IndexError, ValueError):
        relperm = d.relperm
    if bot <= top + 1:
        raise ValueError("SPOT: the bottom MD must be below the top MD.")
    fi = wspot.SpotInputs(top, bot, ((top, ttop), (bot, tbot)), wt == "Deviated", inflow, rw, re_, eff, pr, ub, obg,
                          wc, kvkh, kh, phi, ucs, kcf, kct, wspot.GUNS[gun], relperm, holdup)
    pwf_log = num(st.columns(4)[0], "Log results at FBHP (psia)", "spot_pwf_log", round(min(2764.7, pr * 0.7), 1),
                  14.7, 3e4, 50.0)
    curve = wspot.spot_ipr(pr, wc, fi, model, t_f, pwf_log)
    ss["ipr_recipe"] = (wspot.spot_ipr, (pr, wc, fi, model, t_f), {})
    with st.expander("SPOT layer and log results", expanded=False):
        lay = curve.spot_layer
        st.dataframe(pd.DataFrame({"Result": list(lay), "Value": list(lay.values())}), hide_index=True,
                     width="stretch")
        st.dataframe(curve.spot_log.style.format(precision=5), hide_index=True, width="stretch", height=320)
        st.download_button("Download log results (CSV)", curve.spot_log.to_csv(index=False).encode(),
                           "spot_log_results.csv")
    return curve


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
    """In-situ property function for the sand-control dP: oil at the reservoir pressure (as PROSPER - Tutorials
    15-18), gas at the sand-face pressure."""
    if fluid == "oil":
        if o.method != "eos":
            m = blackoil_model(o)
            if m is not None:
                return wcomp.props_at_pressure(wcomp.props_from_fluid(wvlp.BlackOilFluid(m), "oil", t_f,
                                                                      {"wc": wc, "gor": float(m.inp.rsb)},
                                                                      multiphase), float(val("ipr_pr", 5000.0))), \
                    "black oil at the reservoir pressure (as PROSPER)"
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


ML_DP_OIL = ["Type", "IPR model", "MD bottom (ft)", "TVD bottom (ft)", "Layer pressure (psia)", "Flowing radius (ft)",
             "Roughness (in)", "GOR (scf/STB)", "Oil gravity (API)", "Gas gravity", "Water cut (%)", "PI (STB/d/psi)",
             "Permeability (md)", "Drainage area (acres)", "Dietz shape factor", "Wellbore radius (ft)", "Skin"]
ML_DP_GAS = ["Type", "IPR model", "MD bottom (ft)", "TVD bottom (ft)", "Layer pressure (psia)", "Flowing radius (ft)",
             "Roughness (in)", "CGR (STB/MMscf)", "WGR (STB/MMscf)", "Gas gravity", "C (Mscf/d/psi^2n)", "n",
             "Permeability (md)", "Drainage area (acres)", "Dietz shape factor", "Wellbore radius (ft)", "Skin"]


def ml_dp_example(gas):
    """PROSPER Tutorial 11 layers (oil); a three-layer gas example."""
    nan = np.nan
    if not gas:
        rows = [("Perforated", "Darcy", 9305, 9025, 5214.7, 0.3175, 0.0006, 800, 37, 0.75, 5.0, nan, 71, 400, 31.6, 0.354, 2.4),
                ("Blank", None, 9318, 9035, nan, 0.3175, 0.0006, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan),
                ("Perforated", "Darcy", 9350, 9060, 5334.7, 0.3175, 0.0006, 800, 37, 0.75, 0.5, nan, 65, 400, 31.6, 0.354, 1.0),
                ("Blank", None, 9370, 9075, nan, 0.3175, 0.0006, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan),
                ("Perforated", "Darcy", 9405, 9100, 5339.7, 0.3175, 0.0006, 800, 37, 0.76, 7.4, nan, 100, 400, 31.6, 0.354, 1.5)]
        cols = ML_DP_OIL
    else:
        rows = [("Perforated", "Jones", 9305, 9025, 4014.7, 0.3175, 0.0006, 10, 2, 0.70, nan, nan, 20, 400, 31.6, 0.354, 2.0),
                ("Blank", None, 9318, 9035, nan, 0.3175, 0.0006, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan),
                ("Perforated", "C and n", 9350, 9060, 4114.7, 0.3175, 0.0006, 15, 5, 0.70, 0.01, 0.85, nan, nan, nan, nan, nan)]
        cols = ML_DP_GAS
    return pd.DataFrame(rows, columns=cols, index=pd.RangeIndex(1, len(rows) + 1, name="Layer"))


# ---------------------------------------------------------------- multilateral well (PROSPER Tutorial 12)
def _mlat_example_tables():
    w = wmlat.example_t12()
    t1 = pd.DataFrame({"MD (ft)": [st_[0] for st_ in w.t1_stations], "TVD (ft)": [st_[1] for st_ in w.t1_stations],
                       "Azimuth (deg)": [st_[2] for st_ in w.t1_stations]})
    br = pd.DataFrame({"Branch": [b.name for b in w.branches], "ID (in)": [b.id_in for b in w.branches],
                       "Roughness (in)": [b.rough_in for b in w.branches]})
    sv = pd.DataFrame([{"Branch": b.name, "MD (ft)": float(m), "TVD (ft)": float(t), "Azimuth (deg)": float(a)}
                       for b in w.branches for (m, t, a) in b.stations])
    pf = pd.DataFrame([{"Branch": b.name, "MD start (ft)": p.md_top, "MD end (ft)": p.md_bottom, "Skin": p.skin,
                        "Non-Darcy D (1/(Mscf/d))": np.nan} for b in w.branches for p in b.perfs])
    return {"ipr_mlat_t1": t1, "ipr_mlat_br": br, "ipr_mlat_sv": sv, "ipr_mlat_pf": pf}


def _mlat_load_example():
    for k_, v_ in _mlat_example_tables().items():
        ss[k_] = v_
    ss["_ver_mlat"] = ss.get("_ver_mlat", 0) + 1
    for k_, v_ in {"ipr_mlat_tie_md": 5400.0, "ipr_mlat_tie_tvd": 4900.0, "ipr_mlat_t1_id": 2.8,
                   "ipr_mlat_t1_rough": 0.0006, "ipr_k": 25.0, "ipr_h": 55.0, "ipr_area": 390.0, "ipr_mlat_top": 4900.0,
                   "ipr_mlat_kv": 2.5, "ipr_phi": 0.18, "ipr_swc": 0.2, "ipr_ca": 31.6, "ipr_rw": 0.354}.items():
        put(k_, v_)


def _mlat_table(key, cfg=None):
    if key not in ss:
        ss[key] = _mlat_example_tables()[key]
    return track(key, st.data_editor(ss[key], key=f"{key}_ed_{ss.get('_ver_mlat', 0)}", num_rows="dynamic",
                                     width="stretch", column_config=cfg or {}))


def multilateral_ui(o, pr, t_f, wgr, cgr):
    """PROSPER Inflow Type = MultiLateral Well (Petroleum Experts reservoir model), dry and wet gas."""
    st.caption("A network: the tie point (the bottom of the VLP - set it at the deepest equipment), a tubing T1 down "
               "to the junction, and branches leaving the junction that produce from one reservoir layer through "
               "their perforations. Each perforated survey section is split into 20 segments, each a line source in "
               "the closed drainage box (centred on the tie point); the rate of every segment follows from one "
               "linear solve, with finite conductivity the gravity and friction along T1 and the branches. "
               "Calibrated on the user's PROSPER Tutorial 12 runs (see the README).")
    st.button("Load Tutorial 12 well", key="ipr_mlat_ex", on_click=_mlat_load_example)
    c = st.columns(4)
    tie_md = num(c[0], "Tie point MD (ft)", "ipr_mlat_tie_md", 5400.0, 0.0, 4e4, 10.0)
    tie_tvd = num(c[1], "Tie point TVD (ft)", "ipr_mlat_tie_tvd", 4900.0, 0.0, 4e4, 10.0)
    t1_id = num(c[2], "T1 inside diameter (in)", "ipr_mlat_t1_id", 2.8, 0.5, 20.0, 0.1)
    t1_r = num(c[3], "T1 roughness (in)", "ipr_mlat_t1_rough", 0.0006, 0.0, 0.1, None, "%.5f")
    st.markdown("**T1 deviation survey** (tie point to the junction)")
    t1 = _mlat_table("ipr_mlat_t1")
    st.markdown("**Branches**")
    br = _mlat_table("ipr_mlat_br")
    names = [str(x) for x in br["Branch"].dropna()]
    st.markdown("**Branch deviation surveys** (first point of each branch = the junction)")
    sv = _mlat_table("ipr_mlat_sv", {"Branch": st.column_config.SelectboxColumn(options=names, required=True)})
    st.markdown("**Perforations** (Non-Darcy D blank = calculated from k, the wellbore radius and the interval "
                "length, as PROSPER's Calculate)")
    pf = _mlat_table("ipr_mlat_pf", {"Branch": st.column_config.SelectboxColumn(options=names, required=True)})
    st.markdown("**Reservoir (Petroleum Experts)**")
    c = st.columns(4)
    k = num(c[0], "Reservoir permeability (md)", "ipr_k", 25.0, 0.001, 1e5, 1.0)
    h = num(c[1], "Reservoir thickness (ft)", "ipr_h", 55.0, 0.1, 1e4, 1.0)
    area = num(c[2], "Drainage area (acres)", "ipr_area", 390.0, 0.1, 1e6, 10.0)
    top = num(c[3], "Reservoir top depth (TVD, ft)", "ipr_mlat_top", 4900.0, 0.0, 4e4, 10.0)
    c = st.columns(4)
    kv = num(c[0], "Vertical permeability (md)", "ipr_mlat_kv", 2.5, 0.0001, 1e5, 0.1)
    phi = num(c[1], "Reservoir porosity (fraction)", "ipr_phi", 0.18, 0.01, 0.5, 0.01)
    swc = num(c[2], "Connate water saturation (fraction)", "ipr_swc", 0.2, 0.0, 0.9, 0.01)
    ca = num(c[3], "Dietz shape factor", "ipr_ca", 31.6, 0.01, 100.0, 0.1)
    c = st.columns(4)
    rw = num(c[0], "Wellbore radius (ft)", "ipr_rw", 0.354, 0.05, 5.0, 0.001, "%.3f")
    cond = sel(c[1], "Conductivity", ["Finite conductivity", "Infinite conductivity"], "ipr_mlat_cond")
    f_ = lambda v, d=np.nan: float(v) if pd.notna(v) else d
    branches = []
    for _, b in br.iterrows():
        if pd.isna(b.get("Branch")):
            continue
        nm = str(b["Branch"])
        rows = sv[sv["Branch"] == nm].dropna(subset=["MD (ft)", "TVD (ft)"]).sort_values("MD (ft)")
        stations = [(float(r_["MD (ft)"]), float(r_["TVD (ft)"]), f_(r_.get("Azimuth (deg)"), 0.0)) for _, r_ in rows.iterrows()]
        perfs = [wmlat.Perforation(float(r_["MD start (ft)"]), float(r_["MD end (ft)"]), f_(r_.get("Skin"), 0.0),
                                   (f_(r_.get("Non-Darcy D (1/(Mscf/d))")) if pd.notna(r_.get("Non-Darcy D (1/(Mscf/d))"))
                                    else None))
                 for _, r_ in pf[pf["Branch"] == nm].dropna(subset=["MD start (ft)", "MD end (ft)"]).iterrows()]
        if len(stations) < 2:
            raise ValueError(f"{nm}: enter at least two survey points.")
        branches.append(wmlat.Branch(nm, stations, f_(b.get("ID (in)"), 2.8), f_(b.get("Roughness (in)"), 0.0006), perfs))
    t1r = t1.dropna(subset=["MD (ft)", "TVD (ft)"]).sort_values("MD (ft)")
    t1_st = [(float(r_["MD (ft)"]), float(r_["TVD (ft)"]), f_(r_.get("Azimuth (deg)"), 0.0)) for _, r_ in t1r.iterrows()]
    if len(t1_st) < 2:
        raise ValueError("T1: enter the tie point and the junction.")
    if abs(t1_st[0][0] - tie_md) > 0.5 or abs(t1_st[0][1] - tie_tvd) > 0.5:
        st.warning("The T1 survey should start at the tie point.")
    for b in branches:
        if abs(b.stations[0][0] - t1_st[-1][0]) > 0.5 or abs(b.stations[0][1] - t1_st[-1][1]) > 0.5:
            st.warning(f"{b.name}: the survey should start at the junction ({t1_st[-1][0]:g} ft MD, "
                       f"{t1_st[-1][1]:g} ft TVD).")
    res = wmlat.Reservoir(float(pr), float(t_f), k, h, area, top, kv, phi, swc, ca, rw)
    well = wmlat.MultilateralWell(tie_md, tie_tvd, t1_st, branches, res, t1_id, t1_r)
    for m_ in well.check():
        st.warning(m_)
    gpvt = gas_pvt_inputs(o)
    finite = cond == "Finite conductivity"
    args_ = (well, gpvt, float(wgr or 0.0), float(cgr or 0.0), finite)
    sig_ = repr((repr(well), repr(gpvt), wgr, cgr, finite))
    hit = ss.get("_mlat_cache")
    if hit is not None and hit[0] == sig_:
        curve = hit[1]
    else:
        with st.spinner("Solving the multilateral network..."):
            curve = wmlat.ipr_curve(*args_)
        ss["_mlat_cache"] = (sig_, curve)
    ss["ipr_recipe"] = (wmlat.ipr_curve, args_, {})
    mod = curve.model_obj
    st.markdown("**Results at one tie-point pressure**")
    c = st.columns(4)
    p1 = num(c[0], "Tie-point pressure (psia)", "ipr_mlat_p", min(5014.696, float(pr) - 100.0), 14.7, float(pr) - 0.1, 50.0)
    summ, tab = mod.results(p1, finite=finite)
    c[1].metric("Total gas rate (MMscf/d)", f"{summ['Total gas rate (MMscf/d)']:.3f}")
    c[2].metric("Layer PI (MMscf/d per 10⁶ psi²/cp)", f"{summ['Layer PI (MMscf/d per 1e6 psi2/cp)']:.5f}")
    c[3].metric("Composite skin", f"{summ['Composite skin']:.2f}")
    st.dataframe(pd.DataFrame([{"Branch": b.name, "Rate (MMscf/d)": summ[f"{b.name} rate (MMscf/d)"],
                                "Junction pressure (psia)": summ["Junction pressure (psia)"]} for b in branches]),
                 hide_index=True, width="stretch")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, b in enumerate(branches):
        d_ = tab[tab.Branch == b.name]
        ax.plot(d_["Rate per ft (MMscf/d/ft)"], 0.5 * (d_["MD top (ft)"] + d_["MD bottom (ft)"]), "o", ms=3,
                color=wplt.SERIES[i % len(wplt.SERIES)], label=b.name)
    ax.invert_yaxis()
    ax.set_xlabel("Rate per unit length (MMscf/d/ft)")
    ax.set_ylabel("Measured depth (ft)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    with st.expander("Segment details"):
        st.dataframe(tab.style.format(precision=4), hide_index=True, width="stretch")
        st.caption(f"Non-Darcy D used {mod.D:.4e} 1/(Mscf/d) (PROSPER shows a calculated D divided by about 24.45).")
    return curve


def multilayer_dp_ui(o, gas, t_f, pvt, gasp):
    """PROSPER System | Inflow Performance | MultiLayer - dP Loss in WellBore. Returns the IPR curve at TOP."""
    st.caption("Each perforated layer has its own pressure (at the layer bottom), inflow model and PVT "
               + ("(GOR, oil and gas gravity, water cut)" if not gas else "(CGR, WGR, gas gravity)") +
               ". The well is marched up from the deepest layer: each layer adds its inflow at the pressure at its "
               "bottom and the commingled stream loses pressure between the layers by the multiphase correlation "
               "below, in a pipe of each layer's flowing radius (0.5 x the ID of the pipe from the node above). "
               "Roughness 0 turns the wellbore losses off. Darcy/Jones thickness = TVD from the node above. "
               "Water cut and GOR in the system calculation then follow the layer contributions at each rate.")
    c = st.columns(4)
    top_md = num(c[0], "TOP measured depth (ft)", "ipr_top_md", 9275.0, 0.0, 4e4, 10.0)
    top_tvd = num(c[1], "TOP true vertical depth (ft)", "ipr_top_tvd", 9000.0 if not gas else 9000.0, 0.0, 4e4, 10.0)
    corr_opts = [c_ for c_ in wvlp.VLP_CORRELATIONS if c_ not in wvlp.LOWER_BOUNDS]
    mcorr = sel(c[2], "Wellbore correlation", corr_opts, "ipr_ml_corr", format_func=VLP_LABELS.get,
                help="Multiphase correlation for the pressure loss between the layers.")
    c[3].button("Load Tutorial 11 layers" if not gas else "Load example layers", key="ipr_ml_ex",
                on_click=lambda: (ss.__setitem__(f"ipr_mldp_{'gas' if gas else 'oil'}", ml_dp_example(gas)),
                                  ss.__setitem__("_ver_mldp", ss.get("_ver_mldp", 0) + 1)))
    key = f"ipr_mldp_{'gas' if gas else 'oil'}"
    if key not in ss:
        ss[key] = ml_dp_example(gas)
    cols = ML_DP_GAS if gas else ML_DP_OIL
    tab_ = track(key, st.data_editor(ss[key], key=f"{key}_ed_{ss.get('_ver_mldp', 0)}", num_rows="dynamic",
                                     width="stretch", column_config={
        "Type": st.column_config.SelectboxColumn(options=["Perforated", "Blank"], required=True),
        "IPR model": st.column_config.SelectboxColumn(options=["C and n", "Jones"] if gas else ["PI", "Darcy"])}))
    p_max = float(np.nanmax(pd.to_numeric(tab_["Layer pressure (psia)"], errors="coerce").to_numpy(float),
                            initial=1000.0)) * 1.3
    fluid_, why = vlp_fluid(o, "gas" if gas else "oil", p_max, float(t_f) - 20, float(t_f) + 20)
    if fluid_ is None:
        raise ValueError(why)
    layers, prev_tvd = [], float(top_tvd)
    f_ = lambda row, k_, d=np.nan: float(row[k_]) if k_ in row and pd.notna(row[k_]) else d
    for i, row in tab_.reset_index(drop=True).iterrows():
        if pd.isna(row.get("MD bottom (ft)")) or pd.isna(row.get("TVD bottom (ft)")):
            continue
        L = {"type": row.get("Type") or "Blank", "md": f_(row, "MD bottom (ft)"), "tvd": f_(row, "TVD bottom (ft)"),
             "r_flow": f_(row, "Flowing radius (ft)", 0.25), "rough_in": f_(row, "Roughness (in)", 0.0)}
        h = max(L["tvd"] - prev_tvd, 0.1)
        prev_tvd = L["tvd"]
        if L["type"] == "Perforated":
            if pd.isna(row.get("Layer pressure (psia)")):
                raise ValueError(f"Layer {i + 1}: enter the layer pressure.")
            L.update(pr=f_(row, "Layer pressure (psia)"), model=row.get("IPR model") or ("C and n" if gas else "PI"),
                     h=h, skin=f_(row, "Skin", 0.0))
            darcy_kw = dict(k=f_(row, "Permeability (md)"), area=f_(row, "Drainage area (acres)", 400.0),
                            ca=f_(row, "Dietz shape factor", 31.6), rw=f_(row, "Wellbore radius (ft)", 0.354))
            if not gas:
                L.update(gor=f_(row, "GOR (scf/STB)", float(pvt.get("gor") or 0.0) if pvt else 0.0),
                         wc=f_(row, "Water cut (%)", 0.0) / 100.0)
                if pd.notna(row.get("Oil gravity (API)")):
                    L["api"] = f_(row, "Oil gravity (API)")
                if pd.notna(row.get("Gas gravity")):
                    L["gas_sg"] = f_(row, "Gas gravity")
                if L["model"] == "Darcy":
                    if not np.isfinite(darcy_kw["k"]):
                        raise ValueError(f"Layer {i + 1}: Darcy needs the permeability.")
                    L.update(darcy_kw)
                else:
                    if not np.isfinite(f_(row, "PI (STB/d/psi)")):
                        raise ValueError(f"Layer {i + 1}: enter the PI.")
                    L["j"] = f_(row, "PI (STB/d/psi)")
            else:
                L.update(cgr=f_(row, "CGR (STB/MMscf)", 0.0), wgr=f_(row, "WGR (STB/MMscf)", 0.0))
                sg_ = f_(row, "Gas gravity", gasp["sg"])
                if pd.notna(row.get("Gas gravity")):
                    L["gas_sg"] = sg_
                if L["model"] == "Jones":
                    if not np.isfinite(darcy_kw["k"]):
                        raise ValueError(f"Layer {i + 1}: Jones needs the permeability.")
                    mu_, z_ = (float(v[0]) for v in gasp["mu_z"](np.array([L["pr"]])))
                    a_, b_, _ = ipr.jones_gas_coeffs(darcy_kw["k"], h, darcy_kw["area"], darcy_kw["ca"], darcy_kw["rw"],
                                                     L["skin"], t_f, mu_, z_, sg_, h)
                    L.update(a=a_, b=b_)
                else:
                    if not np.isfinite(f_(row, "C (Mscf/d/psi^2n)")):
                        raise ValueError(f"Layer {i + 1}: enter C and n.")
                    L.update(c=f_(row, "C (Mscf/d/psi^2n)"), n=f_(row, "n", 1.0))
        layers.append(L)
    args_ = ("gas" if gas else "oil", layers, top_md, top_tvd, float(t_f), fluid_, mcorr)
    kw_ = {"n": 25 if gas else 30}
    sig_ = repr((repr(layers), top_md, top_tvd, mcorr, float(t_f), why, repr(getattr(fluid_, "m", None)),
                 repr(getattr(fluid_, "inp", None)), getattr(fluid_, "api", None)))
    hit = ss.get("_mldp_cache")
    if hit is not None and hit[0] == sig_:
        curve = hit[1]
        ss["ipr_recipe"] = (_ipr2_mod.multilayer_wellbore, args_, kw_)
    else:
        with st.spinner("Marching the layers..."):
            curve = ipr2.multilayer_wellbore(*args_, **kw_)
        ss["_mldp_cache"] = (sig_, curve)
    eqd_ = ss.get("equipment") or {}
    try:
        bot_ = float(eqd_["downhole"][1]["MD bottom (ft)"].max())
        if abs(bot_ - top_md) > 1.0:
            st.warning(f"TOP is at {top_md:,.0f} ft MD but the deepest equipment is at {bot_:,.0f} ft MD - the system "
                       "calculation joins the VLP to this IPR at the bottom of the equipment, so set TOP to that depth "
                       "(as PROSPER expects).")
    except (KeyError, TypeError, IndexError):
        pass
    return curve


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


@st.cache_data(show_spinner=False)
def eos_vlp_grid(text, name, p_max, t_min, t_max):
    f = wm_eos.EOSFluid(wm_eos.parse_prp(text, name))
    grid = wvlp.eos_grid(f, p_max, t_min, t_max)
    sep = f.separator_train()
    return grid, float(f.z @ f.mw), float(sep["stages"]["Gas (scf per lbmol feed)"].sum()), \
        float(sep["stb_per_lbmol"]), float(sep["oil_density"]) if np.isfinite(sep["oil_density"]) else 50.0


def vlp_fluid(o, kind, p_max, t_min, t_max):
    """Fluid model for the tubing from the PVT tab (with the surfactant for foam lift). Returns (fluid, description)
    or (None, reason)."""
    fl_, note_ = _vlp_fluid(o, kind, p_max, t_min, t_max)
    fs = ss.get("foam_setup")
    if fl_ is not None and getattr(o, "artificial_lift", "none") == "foam_lift" and fs is not None:
        fl_ = wfoam.FoamFluid(fl_, fs["model"], fs["conc"])
        note_ += f"; foam lift, surfactant {fs['conc']:g} %"
    return fl_, note_


def _vlp_fluid(o, kind, p_max, t_min, t_max):
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
    if getattr(o, "fluid_type", "") == "cbm":
        return wcbm.CBMFluid(cbm_inputs()), ("Coalbed methane: water (salinity "
                                            f"{float(val('pvt_sal', 100000.0)):,.0f} ppm) with free gas of gravity "
                                            f"{float(val('pvt_gg', 0.6)):g} at the CBM GLR")
    if kind == "oil":
        model = blackoil_model(o)
        if model is None:
            return None, "Complete the PVT Input Data tab (black oil)."
        fam = f"{bo.PB_FAMILIES[model.family]} / {bo.VISC_FAMILIES[model.visc_family]}"
        if getattr(o, "viscosity_model", "newtonian") == "non_newtonian":
            rh = rheology_current()
            if rh is None:
                return None, "Enter the rheological parameters in the PVT Input Data tab (non-Newtonian oil)."
            return wvlp.BlackOilFluid(model, rheology=rh), (
                f"Black oil ({fam}{', matched' if model.params else ', not matched'}), non-Newtonian "
                f"(Herschel-Bulkley) oil viscosity in the tubing")
        em = ss.get("emulsion_model") if getattr(o, "emulsions", "none") == "emulsion" else None
        if em is not None and em.in_tubing:
            return wvlp.BlackOilFluid(model, em), (f"Black oil ({fam}{', matched' if model.params else ', not matched'}"
                                                   f"), emulsion viscosity in the tubing")
        return wvlp.BlackOilFluid(model), f"Black oil ({fam}{', matched' if model.params else ', not matched'})"
    if retro_bo_on(o):
        ri = retro_inputs()
        bad = ri.check()
        if bad:
            return None, " ".join(bad)
        return wvlp.RetrogradeBOFluid(ri), (
            f"Black-oil retrograde condensate (dew point {ri.p_dew:,.0f} psia at {ri.t_res:g} °F, CGR {ri.cgr:.1f} "
            f"STB/MMscf, {ri.api:g} °API): condensate drops out in the well below the dew point")
    inp = bo.PVTInputs(rsb=1, api=40, gas_sg=float(val("pvt_gas_sg", 0.70)), salinity_ppm=water.salinity_ppm,
                       h2s=water.h2s, co2=water.co2, n2=water.n2)
    api = float(val("pvt_cond_api", 50.0))
    gi = gas_pvt_inputs(o)
    vap = gi if gas_vapour_on(o) else None
    return wvlp.WetGasFluid(inp, api, gi.visc, vap, gi), (
        f"Gas correlations, gas gravity {inp.gas_sg:.3f}, condensate {api:g} °API (wet gas), "
        f"{wgp.VISC_CORRELATIONS[gi.visc]}" + (", PVT matched" if gi.match else "") + (f", condensed water vapour (minimum WGR {wgp.minimum_wgr(gi):.4f} "
                                               "STB/MMscf)" if vap is not None else ""))


with tabs[2]:
    st.header("IPR Data")
    o = ss.get("options")
    if o is None or o.errors:
        st.info("Fix the configuration in the Options summary first.")
    elif o.is_injector:
        injector_ipr_panel(o)
    elif o.fluid_type == "cbm":
        cbm_ipr_panel(o)
    else:
        gas = o.is_gas
        fluid = "gas" if gas else "oil"
        models = (ipr.GAS_MODELS + GAS_ADV) if gas else (ipr.OIL_MODELS + OIL_ADV)
        st.caption(f"{'Gas and condensate' if gas else 'Oil and water'} inflow models - from the fluid type "
                   f"chosen in the Options summary. Completion: {o.label('completion').lower()}, sand control: "
                   f"{o.label('sand_control').lower()}. Pressures in {ss['_units']['pressure']}.")
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
            ss["ipr_pvt_ctx"] = (o, t_f)          # lets rebuild_ipr re-evaluate mu and Bo at another pressure
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
                    r = reservoir_props(nd=(model in ("Jones", "Petroleum Experts")
                                            or (gas and model == "Multiphase Pseudo Pressure")))
                    skin, skin_parts = skin_panel(o, r["k"], r["rw"], r["h"], fluid)
                    kh_for_skin = r["k"] * r["h"]
                if not gas and model in ipr.OIL_MODELS:
                    pb = pvt["pb"]
                    vogel_ = True
                    if model in ("P.I. Entry", "Darcy"):
                        vogel_ = flag(st.toggle, "Vogel below bubble point", "ipr_vogel", True,
                                      help="PROSPER's switch for P.I. Entry and Darcy: on, the inflow follows Vogel "
                                           "(composite with water) below the bubble point; off, a straight line at "
                                           "every pressure.")
                    if model == "P.I. Entry":
                        j = num(st.columns(4)[0], "Productivity index (STB/d/psi)", "ipr_j", 2.0, 0.001, 1e4, 0.1)
                        curve = ipr.pi_entry(pr, pb, j, wc, vogel_)
                    elif model in ("Vogel", "Composite"):
                        c = st.columns(4)
                        qt = num(c[0], "Test rate (STB/d, liquid)", "ipr_qt", 1500.0, 0.1, 1e6, 10.0)
                        pt = num(c[1], "Test FBHP (psia)", "ipr_pt", 1500.0, 0.0, 3e4, 10.0)
                        curve = ipr.vogel(pr, pb, qt, pt) if model == "Vogel" else ipr.composite(pr, pb, qt, pt, wc)
                    elif model == "Darcy":
                        curve = ipr.darcy(pr, pb, r["k"], r["h"], r["area"], r["ca"], r["rw"], skin, pvt["mu"], pvt["bo"], wc,
                                          vogel_)
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
                    elif model == "Petroleum Experts":
                        c = st.columns(4)
                        tdays = num(c[0], "Time since production started (days)", "ipr_tdays", 50.0, 0.5, 1e6, 1.0)
                        phi = num(c[1], "Reservoir porosity (fraction)", "ipr_phi", 0.15, 0.01, 0.5, 0.01)
                        swc = num(c[2], "Connate water saturation (fraction)", "ipr_swc", 0.25, 0.0, 0.9, 0.01)
                        perm = sel(c[3], "Permeability entered", ["Total permeability", "Gas permeability"],
                                   "ipr_perm_kind")
                        c = st.columns(4)
                        dm = sel(c[0], "Non-Darcy flow factor (D)", ["Calculated", "Entered"], "ipr_d_method")
                        d_ent = num(c[1], "D (1/(Mscf/d))", "ipr_d", 1e-4, 0.0, 1.0, None, "%.4e") \
                            if dm == "Entered" else None
                        psi_g = ipr.pseudo_pressure_fn(gasp["mu_z"], pr, n=200 if o.method == "eos" else 600,
                                                       gauge=True)
                        zz_ = gasp["mu_z"](np.array([pr - 5.0, pr, pr + 5.0]))[1]
                        cg_ = 1.0 / pr - (zz_[2] - zz_[0]) / 10.0 / zz_[1]
                        if o.method == "eos":         # D uses the separator gas gravity (separator train)
                            sg_sep = float(wm_eos.EOSFluid(eos_parse(ss["prp_text"], ss.get("prp_name", "")))
                                           .separator_train()["gas_gravity"])
                        else:
                            sg_sep = gas_pvt_inputs(o).gas_sg
                        # retrograde condensate above the dew point: the condensate is in the reservoir gas - PROSPER
                        # applies no liquid relative-permeability reduction (T41, Pr 6000 psig: D 2.2378e-5, krg at
                        # Swc only); below it the dropped-out liquid reduces krg (T47, Pr 1750 psig)
                        cgr_k = cgr if cgr else (float(retro_inputs().cgr) if retro_bo_on(o) else 0.0)
                        if retro_bo_on(o) and pr >= float(retro_inputs().p_dew):
                            cgr_k = 0.0
                        curve = ipr.petex_gas(pr, t_f, r["k"], r["h"], r["area"], r["ca"], r["rw"], r["hp"], tdays,
                                              phi, swc, skin, psi_g, gasp["mu_r"], float(cg_), float(sg_sep), d_ent,
                                              "gas" if perm == "Gas permeability" else "total", wgr=float(wgr or 0.0),
                                              cgr=float(cgr_k or 0.0), z_r=float(gasp["z_r"]))
                        kh_for_skin = curve.params["Gas permeability (md)"] * r["h"]
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
                elif model in ("Horizontal Well - No Flow Boundaries", "Horizontal Well - Transverse Vertical Fractures",
                               "Horizontal Well - Constant Pressure Upper Boundary"):
                    g, spare = horizontal_inputs()
                    hz_skin = num(spare, "Skin (along the well)", "ipr_hz_skin", 0.0, -5.0, 200.0, 0.5,
                                  help="Mechanical (van Everdingen) skin along the well length.")
                    if model in ("Horizontal Well - No Flow Boundaries", "Horizontal Well - Constant Pressure Upper Boundary"):
                        if model == "Horizontal Well - No Flow Boundaries":
                            meth = radio(st, "Horizontal well model", ["Kuchuk & Goode", "Babu & Odeh", "Joshi"],
                                         "ipr_hz_method", "Babu & Odeh", horizontal=True,
                                         captions=["Closed box, pseudo-steady state, any well length and position "
                                                   "(Goode & Kuchuk 1991)",
                                                   "Closed box, pseudo-steady state",
                                                   "Steady state, constant pressure at drainage radius"])
                        else:
                            meth = "Kuchuk & Goode - constant pressure upper boundary"
                            st.caption("Goode & Kuchuk (1991) steady state: the top of the reservoir is at constant "
                                       "pressure (gas cap, or an active aquifer when the model is turned upside down). "
                                       "The reservoir pressure entered above is the pressure at that boundary; Zw is "
                                       "measured from the no-flow bottom. The drainage area does not enter.")
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
                    meth_f = radio(st, "Horizontal well model", ["Kuchuk & Goode", "Babu & Odeh", "Joshi"],
                                   "ipr_hzf_method", "Kuchuk & Goode", horizontal=True,
                                   help="The overall productivity of the well (PROSPER's Horizontal Well Model); each "
                                        "zone's skin (mechanical, along the well) enters it as that model scales it.")
                    st.caption("Zones (heel to toe) split the overall productivity by permeability x length and zone "
                               "skin; the wellbore friction is marched from the toe to the heel. Zone lengths should "
                               "add up to the well length. Flowing ID = 2 x PROSPER's flowing radius.")
                    z = zones_table().dropna(subset=["Type", "Length (ft)"])
                    zones = [{"type": row["Type"], "length": float(row["Length (ft)"]),
                              "k": float(row["Permeability (md)"]) if pd.notna(row["Permeability (md)"]) else g["k"],
                              "skin": float(row["Skin"]) if pd.notna(row["Skin"]) else 0.0,
                              "id_in": float(row["Flowing ID (in)"]), "rough_in": float(row["Roughness (in)"])}
                             for _, row in z.iterrows()]
                    total_len = sum(zz["length"] for zz in zones)
                    if abs(total_len - g["L"]) > 1:
                        st.warning(f"Zone lengths add up to {total_len:,.0f} ft but the well length is {g['L']:,.0f} ft.")
                    _hz = lambda s_: ipr2.horizontal_resistance(meth_f, g["k"], g["h"], g["rw"], g["aniso_h"],
                                                                g["aniso_v"], g["L"], g["lx"], g["ly"], g["xw"],
                                                                g["yw"], g["zw"], s_)
                    kh_eq, D0 = _hz(0.0)
                    for zz in zones:
                        zz["skin"] = _hz(zz["skin"])[1] - D0 if zz["skin"] else 0.0
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
                elif model == "SPOT":
                    curve = spot_ui(o, pr, t_f, wc)
                elif model == "Multilateral Well":
                    curve = multilateral_ui(o, pr, t_f, wgr, cgr)
                elif model == "Multi-layer with dP loss in wellbore":
                    curve = multilayer_dp_ui(o, gas, t_f, pvt, gasp)
                elif model == "Multi-layer":
                    with_dp = False
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
            if curve is not None and err is None and sc.kind != "none" and hasattr(curve, "layer_table"):
                st.info("Sand control is not applied to the multilayer dP model (PROSPER enters a gravel pack layer "
                        "by layer for it; not available here yet) - the IPR is at the sand face of each layer.")
            elif curve is not None and err is None and sc.kind != "none":
                try:
                    props_, props_note = sc_fluid_props(o, fluid, pvt, gasp, t_f, wc, wgr, cgr, sc_multi)
                    dp_fn = wcomp.sand_control_dp(sc, props_, *sc_res)
                    st.caption(f"Sand-control dP with {props_note}.")
                    curve = wcomp.apply_sand_control(curve, dp_fn)
                    ss["ipr_sc_dp"] = dp_fn
                    for attr in ("layer_rates", "profile", "profile_heel_pressure", "isochronal", "c_iso", "layer_table",
                                 "mix", "top_md", "top_tvd"):
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
                if hasattr(curve, "layer_table"):
                    with st.expander("Layer results (PROSPER: Results, layer by layer)", expanded=False):
                        lt_ = curve.layer_table
                        pick_ = st.selectbox("Layer", sorted(lt_["Layer"].unique()), key="ipr_ml_pick",
                                             format_func=lambda v: f"Layer {v}")
                        st.dataframe(lt_[lt_["Layer"] == pick_].drop(columns="Layer").style.format(precision=3),
                                     hide_index=True, width="stretch")
                        st.caption("Negative rates: crossflow into the layer.")
                if hasattr(curve, "isochronal"):
                    with st.expander("Modified isochronal corrections", expanded=True):
                        st.dataframe(curve.isochronal.style.format(precision=4), hide_index=True, width="stretch")
                st.download_button("Download IPR (CSV)", table.to_csv(index=False).encode(), "ipr.csv")
    if o is not None and not o.errors and not o.is_injector and o.fluid_type == "oil":
        st.divider()
        with st.expander("Stimulation - hydraulic fracturing (PROSPER Tutorial 49)",
                         expanded=bool(ss.get("frac_res"))):
            frac_panel(o)
        with st.expander("Stimulation - mud acidization of sandstones (PROSPER Tutorial 50)",
                         expanded=bool(ss.get("acid_res"))):
            mudacid_panel(o)



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
    units_ = wuu.active()
    conv_, heads_ = {}, {}
    if units_ is not None:                      # show the numeric columns in the chosen units
        for c_ in numeric_cols:
            name_, info_ = wu.relabel(c_, units_)
            if info_ is not None and info_[2] != info_[1]:
                conv_[c_], heads_[c_] = info_, name_
                df[c_] = wu.convert(df[c_].to_numpy(float), info_[0], info_[1], info_[2])
    usig = "" if not conv_ else "@" + "-".join(sorted({i_[2] for i_ in conv_.values()}))
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
            if c_ in heads_:
                kw["headerName"] = heads_[c_]
                kw["valueFormatter"] = JsCode("function(p){return (p.value===null||p.value===undefined||p.value==='')"
                                              " ? '' : String(Number(Number(p.value).toPrecision(6)));}")
            gb.configure_column(c_, type=["numericColumn"],
                                valueParser=JsCode("function(p){const v=String(p.newValue).trim(); "
                                                   "if(v==='') return null; const x=Number(v); "
                                                   "return isNaN(x) ? p.oldValue : x;}"), **kw)
    gb.configure_grid_options(singleClickEdit=True, stopEditingWhenCellsLoseFocus=True, rowHeight=32,
                              suppressColumnVirtualisation=True)
    r = AgGrid(df, gridOptions=gb.build(), allow_unsafe_jscode=True, update_on=["cellValueChanged"],
               key=f"eq_{name}_ag_{ver}{usig}", height=height or 36 + 32 * min(len(df), 14), theme="streamlit")
    out = r["data"] if r is not None and r.get("data") is not None else df
    out = pd.DataFrame(out).reindex(columns=df.columns).set_index(idx)
    out.index = base.index
    for c_ in numeric_cols:
        out[c_] = pd.to_numeric(out[c_], errors="coerce").astype(float)
        if c_ in conv_:
            q_, u_, to_ = conv_[c_]
            out[c_] = wu.convert(out[c_].to_numpy(float), q_, to_, u_)
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
IHTC_COLUMNS = ["Measured Depth (ft)", "Heat Transfer Coefficient (BTU/h/ft2/F)"]
DRILL_COLUMNS = ["Drilling Depth (ft)", "Hole Diameter (in)", "Casing Shoe Depth (ft)", "Casing Outside Diameter (in)",
                 "Casing Weight (lb/ft)", "Top Cement Depth (ft)", "Casing Top Depth (ft)", "Mud Density (lb/gal)"]
LITHO_COLUMNS = ["Formation Type", "Bottom Depth (ft)", "Porosity (fraction)", "In Situ Fluid"]
ss.setdefault("eq_ihtc", blank_grid(IHTC_COLUMNS, 12))
ss.setdefault("eq_drill", blank_grid(DRILL_COLUMNS, 8))
ss.setdefault("eq_litho", blank_grid(LITHO_COLUMNS, 10, ("Formation Type", "In Situ Fluid")))


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


def esp_library_all():
    pumps = dict(wesp.PUMPS)
    pumps.update(ss.get("esp_user_pumps", {}))
    motors = dict(wesp.MOTORS)
    motors.update(ss.get("esp_user_motors", {}))
    return pumps, motors, dict(wesp.CABLES)


def _esp_copy_design():
    d = ss.get("esp_design")
    if not d:
        return
    put("espq_md", float(d["pump_md"]))
    put("espq_f", float(d["freq"]))
    put("espq_n", float(d["stages"]))
    put("espq_pump", d["pump"])
    put("espq_motor", d["motor"])
    put("espq_cable", d["cable"])
    for k_ in ("cable", "sep", "wear"):
        if f"esp_{k_}" in ss.get("_esp_design_inputs", {}):
            put({"cable": "espq_len", "sep": "espq_sep", "wear": "espq_wear"}[k_], ss["_esp_design_inputs"][f"esp_{k_}"])


FOAM_SIG_COLS = ["Surfactant concentration (%)", "Surface tension (dyne/cm)"]
FOAM_RHO_COLS = ["Surfactant concentration (%)", "Foam density (lb/ft3)"]


def foam_data_panel():
    """PROSPER Foam Input Data (foam lift, Tutorial 48): surfactant tables and the surfactant mass percent."""
    st.subheader("Foam lift - surfactant")
    st.caption("PROSPER's default surfactant model (Petroleum Society paper 2007-118, Yang and Zhang): surface tension "
               "and foam density against the surfactant mass percent, interpolated linearly. With the Foam-Homogeneous "
               "correlation the gas and the foam flow as a no-slip mixture (the liquid at the foam density) with no "
               "wall friction; the surface tension is also used by the other correlations and the Turner check.")
    c = st.columns(3)
    conc = num(c[0], "Surfactant mass percent (%)", "foam_conc", 0.2, 0.0, 100.0, 0.01, "%.5g")
    for key, cols, dflt_ in (("foam_sigma", FOAM_SIG_COLS, wfoam.SIGMA_DEFAULT),
                             ("foam_density", FOAM_RHO_COLS, wfoam.DENSITY_DEFAULT)):
        if key not in ss:
            ss[key] = pd.DataFrame(list(dflt_), columns=cols)
    cc = st.columns(2)
    with cc[0]:
        st.markdown("**Surface tension**")
        t_s = track("foam_sigma", st.data_editor(ss["foam_sigma"], key="foam_sigma_ed", num_rows="dynamic",
                                                  width="stretch"))
    with cc[1]:
        st.markdown("**Foam density**")
        t_d = track("foam_density", st.data_editor(ss["foam_density"], key="foam_density_ed", num_rows="dynamic",
                                                    width="stretch"))
    try:
        m = wfoam.FoamModel(tuple(map(tuple, pd.DataFrame(t_s)[FOAM_SIG_COLS].apply(pd.to_numeric, errors="coerce")
                                      .dropna().to_numpy())),
                            tuple(map(tuple, pd.DataFrame(t_d)[FOAM_RHO_COLS].apply(pd.to_numeric, errors="coerce")
                                      .dropna().to_numpy())))
    except ValueError as e:
        st.warning(str(e))
        ss.pop("foam_setup", None)
        return
    ss["foam_setup"] = {"model": m, "conc": float(conc)}
    c[1].metric("Surface tension", f"{float(m.surface_tension(conc)):.3f} dyne/cm")
    c[2].metric("Foam density", f"{float(m.foam_density(conc)):.3f} lb/ft3")
    x = np.linspace(0.0, max(0.5, max(r[0] for r in m.sigma + m.density)), 200)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x, m.surface_tension(x), color=wplt.SERIES[0], lw=2, label="Surface tension (dyne/cm)")
    ax.plot(x, m.foam_density(x), color=wplt.SERIES[1], lw=2, label="Foam density (lb/ft3)")
    ax.axvline(conc, color=wplt.INK, lw=1, ls="--")
    ax.set_xlabel("Surfactant concentration (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def esp_equipment_panel(top_md, secs):
    """System | Electrical Submersible Pump: the installed pump used by the Calculation tab."""
    st.subheader("Electrical submersible pump")
    st.caption("The pump used in the Calculation tab (PROSPER System | Electrical Submersible Pump). The tubing "
               "should end at the pump; below it the flow is in the casing. Run the ESP design (Design tab) and copy "
               "it here, or enter the equipment directly.")
    pumps, motors, cables = esp_library_all()
    tub = secs[secs["Type"] == "Tubing"] if secs is not None and len(secs) else None
    bottom = float(tub["MD bottom (ft)"].max()) if tub is not None and len(tub) else 5000.0
    st.button("Copy from the ESP design", on_click=_esp_copy_design, disabled=not ss.get("esp_design"),
              key="espq_copy")
    c = st.columns(4)
    md = num(c[0], "Pump depth (ft MD)", "espq_md", float(round(bottom)), float(top_md) + 50, 60000.0)
    f = num(c[1], "Operating frequency (Hz)", "espq_f", 60.0, 20.0, 90.0, None, "%.4g")
    n = num(c[2], "Number of stages", "espq_n", 100.0, 1.0, 1000.0, 1.0, "%.0f")
    ln = num(c[3], "Length of cable (ft)", "espq_len", float(round(bottom)), 100.0, 60000.0)
    c = st.columns(3)
    pn = sel(c[0], "Pump", list(pumps), "espq_pump")
    mn = sel(c[1], "Motor", list(motors), "espq_motor")
    cn = sel(c[2], "Cable", list(cables), "espq_cable")
    c = st.columns(4)
    sep = num(c[0], "Gas separator efficiency (%)", "espq_sep", 0.0, 0.0, 100.0)
    wear = num(c[1], "Pump wear factor (fraction)", "espq_wear", 0.0, -0.5, 0.9, None, "%.3g")
    o_ = ss.get("options")
    cbm_ = o_ is not None and getattr(o_, "fluid_type", "") == "cbm"
    if cbm_:
        level = num(c[2], "Liquid level (ft TVD)", "espq_level", 2000.0, 0.0, 60000.0,
                    help="PROSPER Liquid Level Method = Entered (CBM producer): the annulus liquid level; the casing "
                         "head pressure is the result (intake pressure less the water column up to the level and "
                         "the gas column above it).")
        chp = 264.7
    else:
        chp = num(c[2], "Casing head pressure (psia)", "espq_chp", 264.7, 14.7, 20000.0,
                  help="For the annulus liquid level (PROSPER: Casing Head Pressure). No gas is vented up the annulus.")
    meth = sel(c[3], "Pump calculation", ["Stage by stage", "Average rate"], "espq_meth",
               help="Stage by stage: each stage at its own in-situ rate and density as the gas is compressed "
                    "(closest to PROSPER's system results). Average rate: all stages at the average rate.")
    if abs(bottom - md) > 1.0:
        st.warning(f"The tubing ends at {bottom:,.0f} ft MD but the pump is at {md:,.0f} ft MD. PROSPER expects the "
                   "tubing to end at the pump.")
    out = {"pump_md": md, "freq": f, "stages": int(n), "cable_len": ln, "pump": pumps[pn], "motor": motors[mn],
           "cable": cables[cn], "sep": sep / 100.0, "wear": wear, "chp": chp,
           "method": "stages" if meth == "Stage by stage" else "average"}
    if cbm_:
        out["level"] = level
    ss["esp_equipment"] = out
    return out


SRP_VARS = {"n_spm": "Pumping speed (strokes/min)", "stroke": "Surface stroke length (in)",
            "ev": "Pump volumetric efficiency (%)", "plunger": "Pump diameter (in)"}


def _srp_copy_design():
    out = ss.get("srp_result")
    if not out:
        return
    r = out[0]
    put("srpq_md", float(r.inp.pump_md))
    put("srpq_plunger", float(r.inp.plunger))
    put("srpq_rod", int(r.inp.rod_no))
    put("srpq_s", float(r.inp.stroke))
    put("srpq_n", float(round(r.n_spm, 4)))
    put("srpq_ev", float(100 * r.inp.ev))


def srp_equipment_panel(top_md, secs):
    """System | Sucker Rod Pumps: the installed rod pump used by the Calculation tab."""
    st.subheader("Sucker rod pump")
    st.caption("The rod pump used in the Calculation tab (PROSPER System | Sucker Rod Pumps; anchored tubing). Run "
               "the sucker rod pump design (Design tab) and copy it here, or enter the installation directly.")
    tub = secs[secs["Type"] == "Tubing"] if secs is not None and len(secs) else None
    bottom = float(tub["MD bottom (ft)"].max()) if tub is not None and len(tub) else 5000.0
    st.button("Copy from the sucker rod design", on_click=_srp_copy_design, disabled=not ss.get("srp_result"),
              key="srpq_copy")
    c = st.columns(4)
    md = num(c[0], "Pump depth (ft MD)", "srpq_md", float(round(bottom)), float(top_md) + 50, 60000.0)
    ev = num(c[1], "Pump volumetric efficiency (%)", "srpq_ev", 80.0, 1.0, 100.0)
    s_ = num(c[2], "Surface stroke length (in)", "srpq_s", 48.0, 10.0, 400.0, None, "%.5g")
    n_ = num(c[3], "Pumping speed (strokes/min)", "srpq_n", 20.0, 1.0, 40.0, None, "%.6g")
    c = st.columns(4)
    plunger = sel(c[0], "Pump diameter (in)", list(wsrp.PLUNGERS), "srpq_plunger", 2.0,
                  format_func=lambda x: f"{x:.2f}")
    rods = wsrp.rod_numbers(plunger)
    rod_no = sel(c[1], "Rod number", rods, "srpq_rod", 99 if 99 in rods else rods[-1],
                 format_func=lambda n: wsrp.rod_string(n, plunger).name)
    tub_model = sel(c[2], "Tubing model above the pump", ["No slip (as PROSPER)", "Calculation tab correlation"],
                    "srpq_tub", help="No slip reproduces PROSPER's rod pump discharge pressures (six runs, 0.3 %).")
    c = st.columns(4)
    a_meth = sel(c[0], "Gas anchor method", ["Calculated", "Entered"], "srpq_am", "Calculated")
    a_type = sel(c[1], "Gas anchor type", list(wsrp.GAS_ANCHOR_TYPES), "srpq_at", "Cup type")
    a_area = num(c[2], "Annulus area (in2)", "srpq_aa", 1.5, 1.0, 1e6, None, "%.4g", disabled=a_meth != "Calculated")
    a_eff = num(c[3], "Efficiency (fraction)", "srpq_ae", 0.5, 0.0, 1.0, None, "%.4g", disabled=a_meth != "Entered")
    if a_meth == "Calculated" and a_type == "Poor boy":
        st.caption("PROSPER's calculated efficiency for a poor-boy anchor is zero (no gas separated).")
    if abs(bottom - md) > 1.0:
        st.warning(f"The tubing ends at {bottom:,.0f} ft MD but the pump is at {md:,.0f} ft MD.")
    out = {"pump_md": md, "ev": ev, "stroke": s_, "n_spm": n_, "plunger": float(plunger), "rod_no": int(rod_no),
           "no_slip": tub_model.startswith("No slip"), "anchor_method": a_meth, "anchor_type": a_type,
           "anchor_area": a_area, "anchor_eff": a_eff}
    ss["srp_equipment"] = out
    return out


JET_VARS = {"p_surf": "Power fluid injection pressure (psia)", "q_pf": "Power fluid injection rate (STB/d)"}


def _jet_section_at(secs, md):
    """Tubing OD and casing ID at a depth (the tubing section reaching it)."""
    tub = secs[(secs["Type"] == "Tubing")]
    hit = tub[(tub["MD top (ft)"] <= md + 0.5) & (tub["MD bottom (ft)"] >= md - 0.5)]
    row = hit.iloc[-1] if len(hit) else (tub.iloc[-1] if len(tub) else None)
    if row is None or not np.isfinite(row.get("Tubing OD (in)", np.nan)) or not np.isfinite(row.get("Casing ID (in)",
                                                                                                    np.nan)):
        return None, None
    return float(row["Tubing OD (in)"]), float(row["Casing ID (in)"])


def _jet_pump_inputs(prefix, col_md, bottom, top_md, defaults=None):
    """Pump, power fluid and loss inputs shared by the design and the Equipment section."""
    dfl = defaults or {}
    pumps = wjet.PUMPS
    names = ["Custom (enter nozzle and throat areas)"] + list(pumps["Name"])
    c = st.columns(3)
    md = num(c[0], "Pump depth (ft MD)", f"{prefix}_md", float(round(bottom)), float(top_md) + 50, 60000.0)
    q_pf = num(c[1], "Power fluid injection rate (STB/d)", f"{prefix}_qpf", 10000.0, 1.0, 1e6)
    p_surf = num(c[2], "Surface injection pressure (psia)", f"{prefix}_pinj", 2014.7, 14.7, 2e4)
    c = st.columns(4)
    kn = num(c[0], "Nozzle loss coefficient", f"{prefix}_kn", 0.15, 0.0, 2.0, None, "%.3g")
    ks = num(c[1], "Suction loss coefficient", f"{prefix}_ks", 0.0, 0.0, 2.0, None, "%.3g")
    kt = num(c[2], "Throat loss coefficient", f"{prefix}_kt", 0.28, 0.0, 2.0, None, "%.3g")
    kd = num(c[3], "Diffuser loss coefficient", f"{prefix}_kd", 0.10, 0.0, 2.0, None, "%.3g")
    sal = num(st, "Power fluid (water) salinity (ppm)", f"{prefix}_sal", 75000.0, 0.0, 400000.0,
              help="PVT | Power Fluid in PROSPER: water. Its density follows the production tubing's temperature down "
                   "the annulus.")
    return md, q_pf, p_surf, wjet.Losses(kn, ks, kt, kd), sal, names


def _jet_copy_design():
    jd = ss.get("jet_design")
    if not jd:
        return
    for k_ in ("md", "qpf", "pinj", "kn", "ks", "kt", "kd", "sal"):
        if f"jpd_{k_}" in ss or val(f"jpd_{k_}") is not None:
            put(f"jpq_{k_}", val(f"jpd_{k_}"))
    if jd.get("pump"):
        put("jpq_pump", jd["pump"])


def jet_equipment_panel(top_md, secs):
    """System | Jet Pumps: the installed jet pump used by the Calculation tab."""
    st.subheader("Jet pump")
    st.caption("The jet pump used in the Calculation tab (PROSPER System | Jet Pumps; annular injection, tubing "
               "production). The tubing should end at the pump. Run the jet pump design (Design tab) and copy it "
               "here, or enter the pump directly.")
    tub = secs[secs["Type"] == "Tubing"] if secs is not None and len(secs) else None
    bottom = float(tub["MD bottom (ft)"].max()) if tub is not None and len(tub) else 5000.0
    st.button("Copy from the jet pump design", on_click=_jet_copy_design, disabled=not ss.get("jet_design"),
              key="jpq_copy")
    md, q_pf, p_surf, losses, sal, names = _jet_pump_inputs("jpq", None, bottom, top_md)
    pick = sel(st, "Jet pump", names, "jpq_pump", names[1] if len(names) > 1 else names[0])
    if pick.startswith("Custom"):
        c = st.columns(2)
        an = num(c[0], "Nozzle area (in2)", "jpq_an", 0.16476, 0.001, 10.0, None, "%.5g")
        at = num(c[1], "Throat area (in2)", "jpq_at", 0.40185, 0.001, 20.0, None, "%.5g")
    else:
        r_ = wjet.PUMPS[wjet.PUMPS["Name"] == pick].iloc[0]
        an, at = float(r_["Nozzle area (in2)"]), float(r_["Throat area (in2)"])
        st.caption(f"Nozzle area {an:.5f} in2, throat area {at:.5f} in2, area ratio {an / at:.3f}; rate range "
                   f"{r_['Minimum rate (RB/d)']:,.0f} - {r_['Maximum rate (RB/d)']:,.0f} RB/d.")
    tod, cid = _jet_section_at(secs, md) if secs is not None else (None, None)
    if tod is None:
        st.warning("Enter the tubing outside diameter and casing inside diameter of the tubing at the pump (the "
                   "power fluid annulus).")
        ss.pop("jet_equipment", None)
        return None
    if abs(bottom - md) > 1.0:
        st.warning(f"The tubing ends at {bottom:,.0f} ft MD but the pump is at {md:,.0f} ft MD.")
    out = {"pump_md": md, "q_pf": q_pf, "p_surf": p_surf, "losses": losses, "sal": sal, "an": an, "at": at,
           "name": pick, "tubing_od": tod, "casing_id": cid}
    ss["jet_equipment"] = out
    return out


# ================================================================ hydraulic submersible pump
HSP_VARS = {"hsp_speed": "HSP pump speed (rpm)", "hsp_wear": "HSP pump wear factor (fraction)"}


def _hsp_copy_design():
    d = ss.get("hsp_design")
    if not d:
        return
    for k_, v_ in d.items():
        put(f"hspq_{k_}", v_)


def _hsp_salinity_input(prefix):
    own = flag(st.toggle, "Power fluid salinity: the produced water's", f"{prefix}_sal_own", True,
               help="PROSPER's Tutorial 27 power fluid is the produced water (its traverse reproduces PROSPER's "
                    "within 4 psi only with that salinity).")
    if own:
        return None
    return num(st, "Power fluid (water) salinity (ppm)", f"{prefix}_sal", 23000.0, 0.0, 400000.0)


def hsp_equipment_panel(top_md, secs):
    """System | Hydraulic Submersible Pumps: the installed HSP used by the Calculation tab."""
    st.subheader("Hydraulic submersible pump")
    st.caption("The HSP used in the Calculation tab: a centrifugal pump driven by a turbine on the same shaft; the "
               "power fluid (water) goes down the tubing-casing annulus and exhausts into the tubing above the pump. "
               "The tubing should end at the pump. Run the HSP design (Design tab) and copy it here, or enter the "
               "equipment directly.")
    tub = secs[secs["Type"] == "Tubing"] if secs is not None and len(secs) else None
    bottom = float(tub["MD bottom (ft)"].max()) if tub is not None and len(tub) else 5000.0
    st.button("Copy from the HSP design", on_click=_hsp_copy_design, disabled=not ss.get("hsp_design"),
              key="hspq_copy")
    if not whsp.PUMPS or not whsp.TURBINES:
        st.error("The HSP pump and turbine databases are missing (fluxwell/data/hsp_pumps.csv, hsp_turbines.csv).")
        return None
    c = st.columns(3)
    md = num(c[0], "Pump depth (ft MD)", "hspq_md", float(round(bottom)), float(top_md) + 50, 60000.0)
    speed = num(c[1], "Pump speed (rpm)", "hspq_speed", 7500.0, 100.0, 30000.0)
    wear = num(c[2], "Pump wear factor (fraction)", "hspq_wear", 0.0, -0.5, 0.9, None, "%.3g")
    c = st.columns(2)
    pump = sel(c[0], "Pump", list(whsp.PUMPS), "hspq_pump")
    n = int(num(c[1], "Number of pump stages", "hspq_n", 28.0, 1.0, 2000.0, 1.0, "%.0f"))
    c = st.columns(2)
    tur = sel(c[0], "Turbine (blade setting)", list(whsp.TURBINES), "hspq_turbine")
    nt = int(num(c[1], "Number of turbine stages", "hspq_nt", 22.0, 1.0, 2000.0, 1.0, "%.0f"))
    sal = _hsp_salinity_input("hspq")
    p_, t_ = whsp.PUMPS[pump], whsp.TURBINES[tur]
    lo, hi = p_.range_at(speed)
    tlo, thi = t_.range_at(speed)
    st.caption(f"Pump {p_.od:g} in OD, {p_.min_speed:g}-{p_.max_speed:g} rpm, range {lo:,.0f}-{hi:,.0f} RB/d at "
               f"{speed:g} rpm. Turbine {t_.casing_od:g} in casing OD, net flow {tlo:,.0f}-{thi:,.0f} bbl/d at "
               f"{speed:g} rpm, best efficiency {t_.best_eff:g} %.")
    if not p_.min_speed <= speed <= p_.max_speed:
        st.warning("The speed is outside the pump's speed range.")
    tod, cid = _jet_section_at(secs, md) if secs is not None else (None, None)
    if tod is None:
        st.warning("Enter the tubing outside diameter and casing inside diameter of the tubing at the pump (the "
                   "power fluid annulus).")
        ss.pop("hsp_equipment", None)
        return None
    if abs(bottom - md) > 1.0:
        st.warning(f"The tubing ends at {bottom:,.0f} ft MD but the pump is at {md:,.0f} ft MD.")
    out = {"pump_md": md, "pump": pump, "stages": n, "turbine": tur, "turbine_stages": nt, "speed": speed,
           "wear": wear, "sal": sal, "casing_id": cid, "tubing_od": tod}
    ss["hsp_equipment"] = out
    return out


def _hsp_installed(eq, **over):
    d = {**eq, **over}
    return whsp.HSPInstalled(d["pump_md"], d["pump"], int(d["stages"]), d["turbine"], int(d["turbine_stages"]),
                             float(d["speed"]), float(d["wear"]), d["sal"], d["casing_id"], d["tubing_od"])


def hsp_pump_plot(pump, n, speed, q_op=None, h_op=None, wear=0.0):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sp = sorted({pump.min_speed, 0.5 * (pump.min_speed + pump.max_speed), pump.max_speed, float(speed)})
    for i, s_ in enumerate(sp):
        lo, hi = pump.range_at(s_)
        q = np.linspace(0, max(pump.curve.q) * s_ / pump.ref_speed, 120)
        h = n * pump.head_stage(q, s_, wear)
        cur = abs(s_ - speed) < 1e-9
        ax.plot(q, h, color="#d62728" if cur else wplt.SERIES[i % len(wplt.SERIES)], lw=2.0 if cur else 1.1,
                label=f"{s_:,.0f} rpm")
        qq = np.array([lo, hi])
        ax.plot(qq, n * pump.head_stage(qq, s_, wear), "o", color=wplt.INK2, ms=3)
    if q_op is not None and h_op is not None:
        ax.plot([q_op], [h_op], "s", color="#9467bd", ms=8, label="Operating point", zorder=5)
    ax.set_xlabel("Rate through the pump (RB/d)")
    ax.set_ylabel(f"Head, {n} stages (ft)")
    ax.set_title(f"{pump.name} - {n} stages", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def hsp_turbine_plot(turbine, nt, speed, qn_op=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    lo, hi = turbine.range_at(speed)
    q = np.linspace(0.3 * lo, 1.3 * hi, 150)
    ax.plot(q, nt * turbine.head_stage(q, speed), color="#d62728", lw=1.6, label=f"Head loss, {nt} stages (ft)")
    ax2 = ax.twinx()
    ax2.plot(q, turbine.power_stage(q, speed), color="#1f77b4", lw=1.4, label="Power per stage (hp)")
    ax2.plot(q, 100 * turbine.efficiency(q, speed), color="#2ca02c", lw=1.2, ls="--", label="Efficiency (%)")
    for x_ in (lo, hi):
        ax.axvline(x_, color=wplt.INK2, lw=0.8, ls=":")
    if qn_op is not None:
        ax.axvline(qn_op, color="#9467bd", lw=1.2)
        ax.plot([qn_op], [nt * float(turbine.head_stage(qn_op, speed))], "s", color="#9467bd", ms=7)
    ax.set_xlabel("Net turbine flow rate (bbl/d)")
    ax.set_ylabel("Head loss (ft)")
    ax2.set_ylabel("Power per stage (hp) / efficiency (%)")
    ax.set_title(f"{turbine.name} at {speed:,.0f} rpm", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, ls="--")
    fig.tight_layout()
    return fig


def hsp_design_tab(o, eqd, ipr_curve):
    """Design | Hydraulic Drive Downhole Pump - after PROSPER (Tutorial 27)."""
    st.caption("HSP design after PROSPER Design | Hydraulic Drive Downhole Pump: the pump duty at the design rate "
               "(intake from the IPR up the casing, discharge from the top node down the tubing with production + "
               "power fluid), the pumps whose range covers it at the speed, the stages and power of the chosen pump, "
               "then the turbine stages that give that power with the power fluid rate, the pressure drop across the "
               "turbine and the surface injection pressure.")
    if not whsp.PUMPS or not whsp.TURBINES:
        st.error("The HSP pump and turbine databases are missing.")
        return
    top_, secs_, pts_ = eqd["downhole"]
    tub = secs_[secs_["Type"] == "Tubing"]
    bottom_tub = float(tub["MD bottom (ft)"].max()) if len(tub) else float(secs_["MD bottom (ft)"].max())
    gmd, gt = eqd["geothermal"]
    fluid, fluid_note = vlp_fluid(o, "oil", max(float(ipr_curve.pr), 5000.0) * 1.15, float(np.min(gt)) - 10,
                                  float(np.max(gt)) + 10)
    if fluid is None:
        st.warning(fluid_note)
        return
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        st.markdown("**Input data**")
        c = st.columns(3)
        md = num(c[0], "Pump depth (ft MD)", "hspd_md", float(round(bottom_tub)), float(top_) + 50, 60000.0)
        max_od = num(c[1], "Pump maximum OD (in)", "hspd_od", 6.8, 1.0, 20.0, None, "%.3g")
        max_tod = num(c[2], "Turbine maximum OD (in)", "hspd_tod", 6.8, 1.0, 20.0, None, "%.3g")
        c = st.columns(3)
        q_des = num(c[0], "Design (liquid) rate (STB/d)", "hspd_q", 6500.0, 1.0, 1e6)
        wc = num(c[1], "Water cut (%)", "hspd_wc", float(val("sys_wc", 0.0)), 0.0, 100.0)
        gor = num(c[2], "Total GOR (scf/STB)", "hspd_gor", float(val("sys_gor", 300.0)), 0.0, 1e6)
        c = st.columns(3)
        top_p = num(c[0], "Top node pressure (psia)", "hspd_top", float(val("sys_whp", 164.7)), 14.7, 1e4)
        speed = num(c[1], "Pump speed (rpm)", "hspd_speed", 7500.0, 100.0, 30000.0)
        pct = num(c[2], "% Power fluid of reservoir fluid", "hspd_pct", 100.0, 1.0, 1000.0,
                  help="Power fluid rate = this % of the design liquid rate (100 %: as much power fluid as produced "
                       "liquid).")
        wear = num(st, "Pump wear factor (fraction)", "hspd_wear", 0.0, -0.5, 0.9, None, "%.3g")
        sal = _hsp_salinity_input("hspd")
        ms = ss.get("vlp_matches", {}).get("ms", {})
        sel_c = ss.get("vlp_selected")
        corr_opts = [c_ for c_ in wvlp.VLP_CORRELATIONS if c_ not in wvlp.LOWER_BOUNDS]
        corr = sel(st, "Tubing correlation", corr_opts, "hspd_corr", sel_c if sel_c in corr_opts else "pe2_style",
                   format_func=VLP_LABELS.get)
        m_fit = ms.get(corr)
        use_fit = flag(st.toggle, "Use matched parameters", "hspd_use_fit", m_fit is not None, disabled=m_fit is None)
        if m_fit is not None and use_fit:
            p1, p2 = float(m_fit.p1), float(m_fit.p2)
        else:
            c2 = st.columns(2)
            p1 = num(c2[0], "Parameter 1 (gravity)", f"hspd_p1_{corr}", 1.0, 0.3, 3.0, None, "%.4f")
            p2 = num(c2[1], "Parameter 2 (friction)", f"hspd_p2_{corr}", 1.0, 0.01, 10.0, None, "%.4f")
        go = st.button("Calculate pump duty", type="primary", key="hspd_go")
    tod, cid = _jet_section_at(secs_, md)
    d = whsp.HSPDutyInput(md, q_des, wc / 100.0, gor, top_p, speed, pct, wear, sal, cid or 6.4, tod or 4.5, max_od,
                          max_tod)
    u_ = float(val("eq_u", eqd.get("u_geo", 8.0)))
    if go:
        if tod is None:
            st.error("Enter the tubing outside diameter and the casing inside diameter at the pump (Equipment Data) - "
                     "the power fluid goes down that annulus.")
        else:
            cv = ipr_curve
            try:
                cv = rebuild_ipr({"wc": wc / 100.0})
            except ValueError:
                pass
            path = vlp_path(eqd, "tubing", (md,))
            try:
                pwf = float(wgl.ipr_pwf(cv, q_des))
                if not np.isfinite(pwf):
                    raise ValueError("The design rate is above the well's AOF.")
                with st.spinner("Calculating the pump duty ..."):
                    du = whsp.duty(path, fluid, pwf, d, corr, p1, p2, u_, eqd["heat_capacities"])
                ss["hsp_duty"] = {"du": du, "d": d, "path": path, "corr": (corr, p1, p2)}
            except ValueError as e:
                st.error(str(e))
    with right:
        out = ss.get("hsp_duty")
        if out is None:
            st.info("Set the design conditions and press Calculate pump duty.")
            return
        du, d0, path = out["du"], out["d"], out["path"]
        if d0 != d:
            st.info("Inputs have changed - press Calculate pump duty to update.")
        g = 14.696
        m = st.columns(4)
        m[0].metric("Intake (psig)", f"{du.pip - g:,.1f}")
        m[1].metric("Discharge (psig)", f"{du.pdis - g:,.1f}")
        m[2].metric("Head required (ft)", f"{du.head:,.1f}")
        m[3].metric("Average rate (RB/d)", f"{du.q_avg:,.0f}")
        t1, t2, t3, t4 = st.tabs(["Pump duty", "Equipment and results", "Pump plot", "Turbine plot"])
        with t1:
            st.dataframe(whsp.duty_table(du, d0).style.format("{:,.6g}"), use_container_width=True)
        res = None
        with t2:
            cands = whsp.candidate_pumps(du, d0)
            st.markdown("**Pumps** (speed within the pump's range, intake and discharge rates within its range at "
                        "the speed, OD within the maximum)")
            st.dataframe(cands.style.format(precision=3), hide_index=True, use_container_width=True)
            ok = list(cands.loc[cands["Suitable"], "Pump"]) or list(cands["Pump"])
            c = st.columns(2)
            pname = sel(c[0], "Pump", ok, "hspd_pump")
            pump = whsp.PUMPS[pname]
            sal_ = du.detail.get("salinity")
            bw_top = whsp.water_bw_rho(sal_, d0.top_p, du.detail.get("t_wellhead", 150.0))[0]
            tc = whsp.candidate_turbines(d0, bw_top=bw_top)
            tok = list(tc.loc[tc["Suitable"], "Turbine"]) or list(tc["Turbine"])
            tname = sel(c[1], "Turbine", tok, "hspd_turbine",
                        help="It is the user's responsibility to check that the turbine is compatible with the pump "
                             "(PROSPER).")
            st.markdown("**Turbines** (net turbine flow = power fluid at the top node less the balance supply)")
            st.dataframe(tc.style.format(precision=3), hide_index=True, use_container_width=True)
            try:
                res = whsp.design(path, fluid, du, d0, pump, whsp.TURBINES[tname], eqd["heat_capacities"], u_)
            except ValueError as e:
                st.error(str(e))
            if res is not None:
                rows_ = []
                for k_, v_ in res.items():
                    if k_ == "notes":
                        continue
                    if "(psia)" in k_:
                        k_, v_ = k_.replace("(psia)", "(psig)"), v_ - g
                    rows_.append((k_, v_))
                st.markdown("**Results**")
                st.dataframe(pd.DataFrame(rows_, columns=["Result", "Value"]).style.format({"Value": "{:,.6g}"}),
                             hide_index=True, use_container_width=True)
                for n_ in res["notes"]:
                    st.warning(n_)
                ss["hsp_design"] = {"md": md, "speed": speed, "wear": wear, "pump": pname,
                                    "n": float(res["Number of pump stages"]), "turbine": tname,
                                    "nt": float(res["Number of turbine stages"])}
        with t3:
            if res is not None:
                st.pyplot(hsp_pump_plot(pump, res["Number of pump stages"], speed, du.q_avg,
                                        res["Number of pump stages"] * res["Head per stage (ft)"], wear),
                          clear_figure=True)
        with t4:
            if res is not None:
                st.pyplot(hsp_turbine_plot(whsp.TURBINES[tname], res["Number of turbine stages"], speed,
                                           res["Net turbine flow (bbl/d)"]), clear_figure=True)


def hsp_system_calc(go, fluid, eqd, base, corr, p1, p2, sens, ipr_for, labels):
    """Calculation | System for an HSP well (power fluid varied so the turbine power equals the pump power)."""
    eq = ss.get("hsp_equipment")
    if not eq:
        st.info("Enter the HSP in the Equipment Data tab (Hydraulic submersible pump section).")
        return
    sig = ("hsp", tuple(sorted(base.items())), corr, p1, p2, tuple((k_, tuple(v_)) for k_, v_ in sens.items()),
           tuple(sorted((k_, repr(v_)) for k_, v_ in eq.items())))
    if go:
        try:
            combos = wsys.combinations(sens)
            path = vlp_path(eqd, "tubing", (eq["pump_md"],))
            results = []
            with st.spinner(f"Calculating {len(combos)} case(s)..."):
                for ch in combos:
                    prm = {**base, **{k_: v_ for k_, v_ in ch.items() if k_ in ("whp", "wc", "gor", "u")}}
                    inst = _hsp_installed(eq, speed=float(ch.get("hsp_speed", eq["speed"])),
                                          wear=float(ch.get("hsp_wear", eq["wear"])))
                    changes = {k_: float(ch[k_]) for k_ in wsys.IPR_KEYS if k_ in ch}
                    changes["wc"] = float(prm["wc"]) / 100.0
                    if "gor" in prm:
                        changes["gor"] = float(prm["gor"])
                    cv = ipr_for(changes)
                    sol, crv = whsp.system(path, fluid, lambda q, cv=cv: float(wgl.ipr_pwf(cv, q)), prm["wc"] / 100.0,
                                           prm["gor"], prm["whp"], inst, corr, p1, p2, prm["u"],
                                           eqd["heat_capacities"], q_max=float(np.max(cv.q)) * 0.995, n=20)
                    results.append((ch, sol, crv))
            ss["hsp_sys_res"] = {"sig": sig, "results": results}
        except (ValueError, RuntimeError, ZeroDivisionError, KeyError) as e:
            st.error(f"Calculation failed: {e}")
    out = ss.get("hsp_sys_res")
    if not out:
        st.info("Press Calculate.")
        return
    if out["sig"] != sig:
        st.info("Inputs have changed since the last calculation - press Calculate to update.")
    rows = []
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for i, (ch, sol, crv) in enumerate(out["results"]):
        lab = ", ".join(f"{labels.get(k_, k_).split(' (')[0]} {v_:g}" for k_, v_ in ch.items()) or "base"
        d_ = {labels.get(k_, k_): v_ for k_, v_ in ch.items()}
        if sol is None:
            d_["Note"] = "No solution - the HSP cannot lift this case to the top node pressure."
        else:
            d_.update(sol)
        rows.append(d_)
        if len(crv) and i < 8:
            col = wplt.SERIES[i % len(wplt.SERIES)]
            ax.plot(crv["Liquid rate (STB/d)"], crv["Pump discharge from the pump (psia)"], color=col, lw=1.5,
                    label=f"Pump - {lab}")
            ax.plot(crv["Liquid rate (STB/d)"], crv["Discharge needed by the tubing (psia)"], color=col, lw=1.2,
                    ls="--", label=f"Tubing - {lab}")
            if sol is not None:
                ax.scatter([sol["Liquid rate (STB/d)"]], [sol["Pump discharge pressure (psia)"]], color=col, zorder=5)
    tab_ = pd.DataFrame(rows)
    st.markdown(f"**Results** - {len(rows)} case(s); pump {eq['pump']} ({eq['stages']} stages), turbine "
                f"{eq['turbine']} ({eq['turbine_stages']} stages)")
    st.dataframe(tab_.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
    st.download_button("Download HSP system results (CSV)", tab_.to_csv(index=False).encode(), "hsp_system.csv")
    ax.set_xlabel("Liquid rate (STB/d)")
    ax.set_ylabel("Pump discharge pressure (psia)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("Solid: the discharge pressure the pump reaches from the IPR's intake (stage by stage); dashed: the "
               "discharge pressure the tubing needs with production + the power fluid that drives it.")


def hsp_traverse(path, fluid, kind, base, q, corr, p1, p2, hc, hsp):
    """Gradient traverse of an HSP well at one liquid rate (as PROSPER's): the power fluid rate that balances the
    turbine and pump powers, tubing from the top node to the pump discharge with production + power fluid, the pump
    intake from the pump curve, casing to the bottom node."""
    inst = _hsp_installed(hsp)
    wc, gor = base["wc"] / 100.0, float(base["gor"])
    summ, _ = whsp.traverse_profile(path, fluid, float(base["whp"]), q, wc, gor, inst, None, corr, p1, p2,
                                    float(base["u"]), hc)
    md_p = hsp["pump_md"]
    above, below = wesp.sub_path(path, path.md[0], md_p), wesp.sub_path(path, md_p, path.md[-1])
    qpf = summ["Power fluid rate (STB/d)"]
    qt = q + qpf
    wct = (q * wc + qpf) / qt
    fl_mix = wjet._blend_fluid(fluid, q, wc, qpf, summ["salinity"])
    det, reg, prof = wsys.solution_details(above, fl_mix, kind, {**base, "wc": 100 * wct}, qt, corr, p1, p2, hc,
                                           None, comp=comp_current(), t_bottom=summ["Pump discharge temperature (F)"])
    pdis = float(prof["Pressure (psia)"].iloc[-1])
    pip = summ["Pump intake pressure (psia)"]
    det_b, _, prof_b = wsys.solution_details(below, fluid, kind, {**base, "whp": pip}, q, corr, p1, p2, hc, None,
                                             comp=comp_current())
    z_p = float(np.interp(md_p, path.md, path.tvd))
    t_in = summ["Pump intake temperature (F)"]
    pump_row = pd.DataFrame([{"MD (ft)": md_p, "TVD (ft)": z_p, "Pressure (psia)": pdis,
                              "Temperature (°F)": summ["Pump discharge temperature (F)"], "Section": "Pump discharge"},
                             {"MD (ft)": md_p, "TVD (ft)": z_p, "Pressure (psia)": pip, "Temperature (°F)": t_in,
                              "Section": "Pump intake"}])
    prof = pd.concat([prof.assign(Section="Tubing"), pump_row, prof_b.assign(Section="Casing")], ignore_index=True)
    det = dict(det)
    det["Bottom node pressure (psia)"] = float(prof_b["Pressure (psia)"].iloc[-1])
    det["dP gravity - production string (psi)"] += det_b["dP gravity - production string (psi)"]
    det["dP friction - production string (psi)"] += det_b["dP friction - production string (psi)"]
    det["Pump discharge pressure (psia)"], det["Pump intake pressure (psia)"] = pdis, pip
    det["Pump dP (psi)"] = pdis - pip
    det["Pump intake temperature (°F)"] = t_in
    det["Pump discharge temperature (°F)"] = summ["Pump discharge temperature (F)"]
    det["HSP power fluid rate (STB/d)"] = qpf
    det["HSP pump power (hp)"] = summ["Pump power (hp)"]
    det["Power fluid injection pressure (psia)"] = summ["Power fluid injection pressure (psia)"]
    det["Power fluid static (psi)"], det["Power fluid friction (psi)"] = (summ["Power fluid static (psi)"],
                                                                           summ["Power fluid friction (psi)"])
    det["Delta P across turbine (psi)"] = summ["Delta P across turbine (psi)"]
    return det, reg, prof


# ================================================================ progressive cavity pump
PCP_VARS = {"pcp_speed": "PCP speed (rpm)", "pcp_wear": "PCP wear factor (fraction)"}


def pcp_drive(o):
    return "Sucker rod" if getattr(o, "artificial_lift", "") == "pcp_rod" else "Downhole motor"


def _pcp_pick_lists():
    pumps = list(wpcp.PUMPS)
    motors = sorted(wpcp.MOTORS, key=lambda n: (wpcp.MOTORS[n].manufacturer, wpcp.MOTORS[n].series,
                                                wpcp.MOTORS[n].hp, wpcp.MOTORS[n].volts))
    return pumps, motors, list(wpcp.CABLES), list(wpcp.RODS)


def _pcp_installed(eq, speed=None, wear=None):
    """PCPInstalled from the Equipment tab dict (names resolved in the databases)."""
    return wpcp.PCPInstalled(
        wpcp.PUMPS[eq["pump"]], float(eq["pump_md"]), float(eq["speed"] if speed is None else speed),
        float(eq["wear"] if wear is None else wear), float(eq["sep"]), eq["drive"],
        wpcp.MOTORS.get(eq.get("motor")) if eq["drive"] == "Downhole motor" else None,
        wpcp.CABLES.get(eq.get("cable")) if eq["drive"] == "Downhole motor" else None, float(eq.get("cable_len", 0.0)),
        wpcp.RODS.get(eq.get("rod")) if eq["drive"] == "Sucker rod" else None, float(eq.get("rod_len", 0.0)),
        float(eq.get("casing_id") or np.nan))


def _pcp_copy_design():
    pd_ = ss.get("pcp_design")
    if not pd_:
        return
    for k_ in ("md", "wear", "sep", "cable_len", "rod_len"):
        if val(f"pcd_{k_}") is not None:
            put(f"pcq_{k_}", val(f"pcd_{k_}"))
    for k_ in ("pump", "motor", "cable", "rod"):
        if pd_.get(k_):
            put(f"pcq_{k_}", pd_[k_])
    if pd_.get("speed"):
        put("pcq_speed", round(float(pd_["speed"]), 3))


def pcp_equipment_panel(o, top_md, secs):
    """System | Progressive Cavity Pump: the installed PCP used by the Calculation tab."""
    drive = pcp_drive(o)
    st.subheader(f"Progressive cavity pump ({drive.lower()} drive)")
    st.caption("The PCP used in the Calculation tab (PROSPER System | Progressive Cavity Pump). The pump's speed sets "
               "the rate; the head it develops sets the slip. Run the PCP design (Design tab) and copy it here, or "
               "enter the pump directly.")
    tub = secs[secs["Type"] == "Tubing"] if secs is not None and len(secs) else None
    bottom = float(tub["MD bottom (ft)"].max()) if tub is not None and len(tub) else 3000.0
    st.button("Copy from the PCP design", on_click=_pcp_copy_design, disabled=not ss.get("pcp_design"),
              key="pcq_copy")
    pumps, motors, cables, rods = _pcp_pick_lists()
    c = st.columns(4)
    md = num(c[0], "Pump depth (ft MD)", "pcq_md", float(round(min(bottom - 50.0, bottom))), float(top_md) + 10,
             60000.0)
    speed = num(c[1], "Pump speed (rpm)", "pcq_speed", 150.0, 1.0, 2000.0)
    wear = num(c[2], "Pump wear factor (fraction)", "pcq_wear", 0.0, 0.0, 0.9, None, "%.3g",
               help="Derates the pump's head capacity as PROSPER: the slip at head H is that of H / (1 - wear). "
                    "PROSPER's design screen ignores it; the Calculation tab applies it.")
    sep = num(c[3], "Gas separator efficiency (fraction)", "pcq_sep", 0.0, 0.0, 1.0, None, "%.3g")
    pump = sel(st, "Pump", pumps, "pcq_pump", pumps[-1])
    p_ = wpcp.PUMPS[pump]
    st.caption(f"{p_.size:g} in OD, maximum head {p_.max_head:,.0f} ft, {p_.head_grad:.4f} RB/d per rpm "
               f"({p_.ref_rate:,.0f} RB/d at {p_.ref_speed:g} rpm and zero head).")
    tod, cid = _jet_section_at(secs, md) if secs is not None else (None, None)
    out = {"pump_md": md, "speed": speed, "wear": wear, "sep": sep, "pump": pump, "drive": drive,
           "casing_id": cid, "name": pump}
    if drive == "Downhole motor":
        c = st.columns([2, 1, 1])
        dflt_m = "Centrilift 450 175HP 2285V 50A" if "Centrilift 450 175HP 2285V 50A" in motors else motors[0]
        out["motor"] = sel(c[0], "Motor", motors, "pcq_motor", dflt_m)
        out["cable"] = sel(c[1], "Cable", cables, "pcq_cable", "#1 Copper" if "#1 Copper" in cables else cables[0])
        out["cable_len"] = num(c[2], "Cable length (ft)", "pcq_cable_len", float(round(md + 500.0)), 0.0, 60000.0)
        m_ = wpcp.MOTORS[out["motor"]]
        st.caption(f"Motor {m_.od:g} in OD, {m_.hp:g} hp / {m_.volts:g} V / {m_.amps:g} A at {m_.hz:g} Hz "
                   f"({60 * m_.hz:g} rpm synchronous); at {speed:g} rpm it gives "
                   f"{m_.hp * float(m_.scale(speed)):,.2f} hp at {m_.volts * float(m_.scale(speed)):,.1f} V. "
                   + ("Load curves: the manufacturer's coefficients." if m_.c_amps else
                      "Load curves: PROSPER's default motor curves."))
        if cid is not None and m_.od >= cid:
            st.warning(f"The motor ({m_.od:g} in) does not fit the casing ({cid:g} in ID).")
    else:
        c = st.columns([2, 1])
        out["rod"] = sel(c[0], "Sucker rod", rods, "pcq_rod", rods[0])
        out["rod_len"] = num(c[1], "Total rod length (ft)", "pcq_rod_len", float(round(md)), 1.0, 60000.0)
        st.caption("With a sucker rod drive the production flows up the rod-tubing annulus (PROSPER). Rod loads and "
                   "stresses are our own estimates (PROSPER's rod outputs are not reproduced).")
    if abs(bottom - md) > 1.0 and md > bottom:
        st.warning(f"The tubing ends at {bottom:,.0f} ft MD, above the pump at {md:,.0f} ft MD.")
    ss["pcp_equipment"] = out
    return out


def pcp_pump_plot(pump: "wpcp.PCPPump", d=None, n_des=None, sol=None):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    n = np.linspace(0, 1.25 * max(pump.ref_speed, n_des or 0), 80)
    heads = sorted(set([0.0] + list(pump.heads)))
    cm = plt.get_cmap("viridis")
    for i, h in enumerate(heads):
        q = pump.rate(n, h)
        ax.plot(n, q, color=cm(i / max(len(heads) - 1, 1)), lw=1.5, label=f"{h:,.0f} ft")
    if d is not None and n_des is not None:
        ax.scatter([n_des], [d.q_avg], color="#d62728", zorder=6,
                   label=f"Design: {d.q_avg:,.0f} RB/d at {d.head:,.0f} ft")
    if sol is not None:
        ax.scatter([sol[0]], [sol[1]], marker="s", color="k", zorder=6, label="System solution")
    ax.set_xlabel("Pump speed (rpm)")
    ax.set_ylabel("Rate through the pump (RB/d)")
    ax.set_title(pump.name, fontsize=10)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=7, title="Head", title_fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def pcp_design_tab(o, eqd, ipr_curve):
    """Design | Progressive Cavity Pump - after PROSPER (Tutorial 29): pump duty, pump, motor and cable or rod."""
    drive = pcp_drive(o) if getattr(o, "artificial_lift", "") in ("pcp_motor", "pcp_rod") else None
    st.caption("PCP design after PROSPER Design | Progressive Cavity Pump: the pump duty at the design rate (intake "
               "pressure from the IPR up the casing, discharge pressure from the top node down the tubing), the head "
               "and the average downhole rate the pump must deliver, then the pump (its speed and power at that "
               "head), and the motor and cable (downhole motor drive) or the rod (sucker rod drive).")
    if drive is None:
        drive = radio(st, "Drive", list(wpcp.DRIVES), "pcd_drive", horizontal=True,
                      help="Set by the artificial lift in the Options summary when a PCP is chosen there.")
    top_, secs_, pts_ = eqd["downhole"]
    tub = secs_[secs_["Type"] == "Tubing"]
    bottom_tub = float(tub["MD bottom (ft)"].max()) if len(tub) else float(secs_["MD bottom (ft)"].max())
    gmd, gt = eqd["geothermal"]
    fluid, fluid_note = vlp_fluid(o, "oil", max(float(ipr_curve.pr), 5000.0) * 1.15, float(np.min(gt)) - 10,
                                  float(np.max(gt)) + 10)
    if fluid is None:
        st.warning(fluid_note)
        return
    pumps, motors, cables, rods = _pcp_pick_lists()
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        st.markdown("**Input data**")
        c = st.columns(2)
        md = num(c[0], "Pump depth (ft MD)", "pcd_md", float(round(bottom_tub - 50.0)), float(top_) + 10, 60000.0)
        max_od = num(c[1], "Maximum OD (in)", "pcd_od", 6.4, 1.0, 20.0, None, "%.3g")
        c = st.columns(2)
        sep = num(c[0], "Gas separator efficiency (fraction)", "pcd_sep", 0.0, 0.0, 1.0, None, "%.3g")
        wear = num(c[1], "Pump wear factor (fraction)", "pcd_wear", 0.0, 0.0, 0.9, None, "%.3g")
        c = st.columns(2)
        q_des = num(c[0], "Design rate (STB/d)", "pcd_q", 750.0, 1.0, 1e6)
        wc = num(c[1], "Water cut (%)", "pcd_wc", float(val("sys_wc", 0.0)), 0.0, 100.0)
        c = st.columns(2)
        gor = num(c[0], "Total GOR (scf/STB)", "pcd_gor", float(val("sys_gor", 300.0)), 0.0, 1e6)
        top_p = num(c[1], "Top node pressure (psia)", "pcd_top", float(val("sys_whp", 264.7)), 14.7, 1e4)
        if drive == "Downhole motor":
            c = st.columns(3)
            cable_len = num(c[0], "Cable length (ft)", "pcd_cable_len", float(round(md + 500.0)), 0.0, 60000.0)
            v_surf = num(c[1], "Voltage at surface (V)", "pcd_vsurf", 9000.0, 1.0, 1e5,
                         help="Available surface voltage - compared with the voltage the motor and cable need.")
            margin = num(c[2], "Motor power safety margin (fraction)", "pcd_margin", 0.0, 0.0, 2.0, None, "%.3g")
            rod_len = 0.0
        else:
            rod_len = num(st, "Total rod length (ft)", "pcd_rod_len", float(round(md)), 1.0, 60000.0)
            cable_len, v_surf, margin = 0.0, np.nan, 0.0
        ms = ss.get("vlp_matches", {}).get("ms", {})
        sel_c = ss.get("vlp_selected")
        corr_opts = [c_ for c_ in wvlp.VLP_CORRELATIONS if c_ not in wvlp.LOWER_BOUNDS]
        corr = sel(st, "Tubing correlation", corr_opts, "pcd_corr", sel_c if sel_c in corr_opts else "pe2_style",
                   format_func=VLP_LABELS.get)
        m_fit = ms.get(corr)
        use_fit = flag(st.toggle, "Use matched parameters", "pcd_use_fit", m_fit is not None,
                       disabled=m_fit is None)
        if m_fit is not None and use_fit:
            p1, p2 = float(m_fit.p1), float(m_fit.p2)
        else:
            c2 = st.columns(2)
            p1 = num(c2[0], "Parameter 1 (gravity)", f"pcd_p1_{corr}", 1.0, 0.3, 3.0, None, "%.4f")
            p2 = num(c2[1], "Parameter 2 (friction)", f"pcd_p2_{corr}", 1.0, 0.01, 10.0, None, "%.4f")
        rod_d = None
        if drive == "Sucker rod":
            rod_d = wpcp.RODS[sel(st, "Sucker rod", rods, "pcd_rod", rods[0])]
        go = st.button("Calculate pump duty", type="primary", key="pcd_go")
    hc = eqd["heat_capacities"]
    u_ = float(val("eq_u", eqd.get("u_geo", 8.0)))
    if go:
        cv = ipr_curve
        try:
            cv = rebuild_ipr({"wc": wc / 100.0})
        except ValueError:
            pass
        path = vlp_path(eqd, "tubing", (md,))
        try:
            with st.spinner("Calculating the pump duty ..."):
                du = wpcp.duty(path, fluid, float(wgl.ipr_pwf(cv, q_des)), q_des, wc / 100.0, gor, top_p, md, sep,
                               corr, p1, p2, u_, hc, rod_d if drive == "Sucker rod" else None)
                case = {"q": np.array([q_des]), "wc": np.array([wc / 100.0]), "gor": np.array([du.gor_above])}
                mass, cp = wpcp._cp_mix(fluid, case, hc)
            ss["pcp_duty"] = {"duty": du, "cp": cp, "mass": mass, "drive": drive}
        except ValueError as e:
            st.error(str(e))
    with right:
        res = ss.get("pcp_duty")
        if res is None:
            st.info("Set the design conditions and press Calculate pump duty.")
            return
        du = res["duty"]
        m = st.columns(4)
        m[0].metric("Intake (psig)", f"{du.pip - 14.696:,.1f}")
        m[1].metric("Discharge (psig)", f"{du.pdp - 14.696:,.1f}")
        m[2].metric("Head required (ft)", f"{du.head:,.1f}")
        m[3].metric("Average downhole rate (RB/d)", f"{du.q_avg:,.1f}")
        t1, t2, t3 = st.tabs(["Pump duty", "Select equipment", "Pump plot"])
        with t1:
            st.dataframe(wpcp.duty_table(du).style.format({"Value": "{:,.6g}"}), hide_index=True,
                         width="stretch")
        with t2:
            cands = wpcp.candidate_pumps(du, max_od=max_od)
            st.dataframe(cands.style.format(precision=3), hide_index=True, width="stretch")
            ok = list(cands.loc[cands["Suitable"], "Pump"]) or list(cands["Pump"])
            pick = sel(st, "Pump", ok, "pcd_pump", ok[0])
            pump = wpcp.PUMPS[pick]
            motor = cable = rod = None
            if res["drive"] == "Downhole motor":
                n_ = float(pump.speed_for(du.q_avg, du.head, wear))
                pw_ = float(wpcp.design_power(pump, n_, du.head))
                fit = [n for n in motors if wpcp.MOTORS[n].od <= max_od and
                       wpcp.MOTORS[n].hp * float(wpcp.MOTORS[n].scale(n_)) >= pw_ * (1 + margin)]
                st.caption(f"{len(fit)} motors within the maximum OD give the {pw_:,.2f} hp needed at {n_:,.1f} rpm"
                           + (f" (+{100 * margin:g} % margin)" if margin else "") + ".")
                mlist = fit or motors
                dflt_m = ("Centrilift 450 175HP 2285V 50A" if "Centrilift 450 175HP 2285V 50A" in mlist
                          else mlist[0])
                motor = wpcp.MOTORS[sel(st, "Motor", mlist, "pcd_motor", dflt_m)]
                auto = "Smallest that carries the current"
                cab_pick = sel(st, "Cable", [auto] + cables, "pcd_cable", auto)
                cable = None if cab_pick == auto else wpcp.CABLES[cab_pick]
            else:
                rod = rod_d
            r = wpcp.design(du, pump, res["drive"], motor, cable, cable_len, wear, margin, rod, rod_len,
                            res["cp"], res["mass"])
            notes = r.pop("notes")
            if res["drive"] == "Downhole motor" and np.isfinite(v_surf) and "Required surface voltage (V)" in r:
                if r["Required surface voltage (V)"] > v_surf:
                    notes.append(f"The motor and cable need {r['Required surface voltage (V)']:,.0f} V at surface, "
                                 f"above the {v_surf:,.0f} V available.")
            st.dataframe(pd.DataFrame({"Design": {k_: (f"{v_:,.4f}" if isinstance(v_, (float, np.floating)) else
                                                        str(v_)) for k_, v_ in r.items()}}),
                         width="stretch")
            for n_ in notes:
                st.warning(n_)
            ss["pcp_design"] = {"pump": pick, "speed": r["Pump speed (rpm)"],
                                "motor": motor.name if motor else None,
                                "cable": (cable.name if cable else r.get("Cable")) if motor else None,
                                "rod": rod.label if rod else None}
            st.caption("Copy the design to the Equipment tab (PCP section) to use it in the Calculation tab.")
        with t3:
            p_ = wpcp.PUMPS[ss.get("pcp_design", {}).get("pump") or cands["Pump"].iloc[0]]
            st.pyplot(pcp_pump_plot(p_, du, float(p_.speed_for(du.q_avg, du.head, wear))), clear_figure=True)
            st.caption("Rate through the pump against speed at the database heads: rate = head gradient x speed + "
                       "intercept(head). The dot is the design (average downhole rate at the head required).")


def pcp_system_calc(go, fluid, eqd, base, corr, p1, p2, sens, ipr_for, labels):
    """Calculation | System for a PCP well: the rate at which the pump's own rate at the head it develops equals
    the average rate through it."""
    eq = ss.get("pcp_equipment")
    if not eq:
        st.info("Enter the pump in the Equipment Data tab (Progressive cavity pump section).")
        return
    sig = ("pcp", tuple(sorted(base.items())), corr, p1, p2, tuple((k_, tuple(v_)) for k_, v_ in sens.items()),
           tuple(sorted((k_, repr(v_)) for k_, v_ in eq.items())))
    if go:
        try:
            combos = wsys.combinations(sens)
            path = vlp_path(eqd, "tubing", (eq["pump_md"],))
            results = []
            with st.spinner(f"Calculating {len(combos)} case(s)..."):
                for ch in combos:
                    prm = {**base, **{k_: v_ for k_, v_ in ch.items() if k_ in ("whp", "wc", "gor", "u")}}
                    inst = _pcp_installed(eq, ch.get("pcp_speed"), ch.get("pcp_wear"))
                    changes = {k_: float(ch[k_]) for k_ in wsys.IPR_KEYS if k_ in ch}
                    changes["wc"] = float(prm["wc"]) / 100.0
                    if "gor" in prm:
                        changes["gor"] = float(prm["gor"])
                    cv = ipr_for(changes)
                    sol, crv = wpcp.system(path, fluid, lambda q, cv=cv: float(wgl.ipr_pwf(cv, q)),
                                           prm["wc"] / 100.0, prm["gor"], prm["whp"], inst, corr, p1, p2, prm["u"],
                                           eqd["heat_capacities"], q_max=float(np.max(cv.q)) * 0.995, n=24)
                    results.append((ch, sol, crv))
            ss["pcp_sys_res"] = {"sig": sig, "results": results}
        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            st.error(f"Calculation failed: {e}")
    out = ss.get("pcp_sys_res")
    if not out:
        st.info("Press Calculate.")
        return
    if out["sig"] != sig:
        st.info("Inputs have changed since the last calculation - press Calculate to update.")
    rows, notes = [], []
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for i, (ch, sol, crv) in enumerate(out["results"]):
        lab = ", ".join(f"{labels.get(k_, k_).split(' (')[0]} {v_:g}" for k_, v_ in ch.items()) or "base"
        d_ = {labels.get(k_, k_): v_ for k_, v_ in ch.items()}
        if sol is None:
            d_["Note"] = "No solution - the pump cannot lift this case to the top node pressure."
        else:
            sol = dict(sol)
            for n_ in sol.pop("notes", []):
                notes.append(f"{lab}: {n_}")
            d_.update(sol)
        rows.append(d_)
        if len(crv) and i < 8:
            col = wplt.SERIES[i % len(wplt.SERIES)]
            ax.plot(crv["Liquid rate (STB/d)"], crv["Pump rate at that head (RB/d)"], color=col, lw=1.5,
                    label=f"Pump - {lab}")
            ax.plot(crv["Liquid rate (STB/d)"], crv["Average rate (RB/d)"], color=col, lw=1.2, ls="--",
                    label=f"Well - {lab}")
            if sol is not None and np.isfinite(sol.get("Liquid rate (STB/d)", np.nan)):
                ax.scatter([sol["Liquid rate (STB/d)"]], [sol["Average rate through pump (RB/d)"]], color=col,
                           zorder=5)
    tab_ = pd.DataFrame(rows)
    st.markdown(f"**Results** - {len(rows)} case(s); pump {eq['pump']}")
    st.dataframe(tab_.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
    st.download_button("Download PCP system results (CSV)", tab_.to_csv(index=False).encode(), "pcp_system.csv")
    for n_ in dict.fromkeys(notes):
        st.warning(n_)
    ax.set_xlabel("Liquid rate (STB/d)")
    ax.set_ylabel("Rate through the pump (RB/d)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    if len(tab_) > 1 and "Oil rate (STB/d)" in tab_:
        var_cols = [labels.get(k_, k_) for k_ in sens]
        if var_cols:
            fig2, ax2 = plt.subplots(figsize=(7.8, 4.0))
            x_ = var_cols[0]
            groups = tab_.groupby(var_cols[1:]) if len(var_cols) > 1 else [("", tab_)]
            for gk, gdf in groups:
                ax2.plot(gdf[x_], gdf["Oil rate (STB/d)"], marker="o", ms=4,
                         label=(f"{var_cols[1].split(' (')[0]} {gk if not isinstance(gk, tuple) else gk[0]:g}"
                                if len(var_cols) > 1 else None))
            ax2.set_xlabel(x_)
            ax2.set_ylabel("Oil rate (STB/d)")
            ax2.grid(alpha=0.3, ls="--")
            if len(var_cols) > 1:
                ax2.legend(fontsize=7)
            fig2.tight_layout()
            st.pyplot(fig2, clear_figure=True)
    st.caption("As PROSPER: at each rate the intake pressure comes from the IPR (up the casing) and the discharge "
               "pressure from the top node (down the tubing); the head between them sets the pump's own rate at its "
               "speed (head gradient x speed + intercept(head)). The solution is where that equals the average rate "
               "through the pump. Power = power gradient(dP / 0.433 psi/ft) x speed; motor and cable from the "
               "Equipment tab.")


def pcp_traverse(path, fluid, kind, base, q, corr, p1, p2, hc, pcp_eq):
    """Gradient traverse of a PCP well at one rate (as PROSPER's): tubing (or rod-tubing annulus) from the top node
    to the pump discharge, the pump at its speed (intake from the head at which it passes the rate), casing to the
    bottom node."""
    inst = _pcp_installed(pcp_eq)
    md_p = inst.pump_md
    p_ = wpcp.with_rod(path, inst.rod, md_p) if inst.drive == "Sucker rod" else path
    s, *_ = wpcp.traverse_profile(p_, fluid, base["whp"], q, base["wc"] / 100.0, float(base["gor"]), inst, corr, p1,
                                  p2, float(base["u"]), hc)
    above, below = wesp.sub_path(p_, p_.md[0], md_p), wesp.sub_path(p_, md_p, p_.md[-1])
    det, reg, prof = wsys.solution_details(above, fluid, kind, base, q, corr, p1, p2, hc, None, comp=comp_current(),
                                           t_bottom=s["Pump discharge temperature (F)"])
    pip, pdp = s["Pump intake pressure (psia)"], s["Pump discharge pressure (psia)"]
    det_b, _, prof_b = wsys.solution_details(below, fluid, kind, {**base, "whp": pip}, q, corr, p1, p2, hc, None,
                                             comp=comp_current())
    z_p = float(np.interp(md_p, path.md, path.tvd))
    pump_row = pd.DataFrame([{"MD (ft)": md_p, "TVD (ft)": z_p, "Pressure (psia)": pdp,
                              "Temperature (°F)": s["Pump discharge temperature (F)"], "Section": "Pump discharge"},
                             {"MD (ft)": md_p, "TVD (ft)": z_p, "Pressure (psia)": pip,
                              "Temperature (°F)": s["Pump intake temperature (F)"], "Section": "Pump intake"}])
    prof = pd.concat([prof.assign(Section="Tubing"), pump_row, prof_b.assign(Section="Casing")], ignore_index=True)
    det = dict(det)
    det["Bottom node pressure (psia)"] = float(prof_b["Pressure (psia)"].iloc[-1])
    det["dP gravity - production string (psi)"] += det_b["dP gravity - production string (psi)"]
    det["dP friction - production string (psi)"] += det_b["dP friction - production string (psi)"]
    det["Pump discharge pressure (psia)"], det["Pump intake pressure (psia)"] = pdp, pip
    det["Pump dP (psi)"] = pdp - pip
    det["Pump intake temperature (°F)"] = s["Pump intake temperature (F)"]
    det["Pump discharge temperature (°F)"] = s["Pump discharge temperature (F)"]
    det["PCP head (ft)"] = s["Pump head (ft)"]
    det["PCP average rate (RB/d)"] = s["Average rate through pump (RB/d)"]
    return det, reg, prof


GL_VALVE_COLS = ["Measured depth (ft)", "Type", "Number of valves", "Port size (64ths in)", "R value",
                 "Dome pressure at 60 °F (psig)"]


def _gl_valves_blank(n=10):
    df = pd.DataFrame({c: pd.Series([np.nan] * n, dtype=float) for c in GL_VALVE_COLS})
    df["Type"] = pd.Series([None] * n, dtype=object)
    df.index = pd.RangeIndex(1, n + 1, name="Point")
    return df


def gl_valve_db_panel():
    """Gas lift valve database: built-in valves plus the user's (saved with the project, CSV import / export)."""
    if "gl_valve_db" not in ss:
        ss["gl_valve_db"] = pd.DataFrame({c_: pd.Series(dtype=float if c_ in ("Port (64ths in)", "R value") else object)
                                          for c_ in wgl.VALVE_DB_COLS})
    ver = ss.get("_ver_gl_valve_db", 0)
    cur = ss.get("_tables", {}).get("gl_valve_db", ss["gl_valve_db"]) if ss.get("_gl_vdb_ver") == ver \
        else ss["gl_valve_db"]               # a new table (project, import) replaces the editor's last state
    ss["_gl_vdb_ver"] = ver
    wgl.register_valves(cur)
    n_user = len(wgl.valves_from_table(cur))
    with st.expander(f"Gas lift valve database ({len(wgl.VALVES)} valves, {n_user} of yours)"):
        st.caption("Valves for the design: manufacturer, model, specification, valve type and, for every port size, "
                   "the R value (port area / bellows area) from the manufacturer's data - one row per port. Your "
                   "valves are saved with the project; a valve with the same manufacturer, model and specification "
                   "as a built-in one replaces it. PROSPER's database lists Altec, Baker, Camco, Mcmurry-Macco, "
                   "Otis-Merla, PTC and Weatherford valves under Design | Gaslift | Current Valve Type.")
        bt = wgl.builtin_valve_table()
        st.markdown("**Built-in**")
        st.dataframe(bt, hide_index=True, width="stretch", height=min(35 * (len(bt) + 1) + 3, 260))
        st.markdown("**Your valves**")
        up = st.file_uploader("Import valves (CSV with the columns below)", type=["csv"], key="gl_vdb_up")
        if up is not None and ss.get("_gl_vdb_last") != (up.name, up.size):
            try:
                imp = pd.read_csv(up)
                miss = [c_ for c_ in ("Manufacturer", "Model", "Port (64ths in)", "R value") if c_ not in imp.columns]
                if miss:
                    st.error("Missing columns: " + ", ".join(miss))
                else:
                    for c_ in wgl.VALVE_DB_COLS:
                        if c_ not in imp.columns:
                            imp[c_] = "" if c_ not in ("Port (64ths in)", "R value") else np.nan
                    ss["gl_valve_db"] = pd.concat([cur, imp[wgl.VALVE_DB_COLS]], ignore_index=True)
                    ss["_ver_gl_valve_db"] = ss.get("_ver_gl_valve_db", 0) + 1
                    ss["_gl_vdb_last"] = (up.name, up.size)
                    st.rerun()
            except (ValueError, UnicodeDecodeError) as e:
                st.error(f"Could not read the file: {e}")
        ed = track("gl_valve_db", st.data_editor(
            ss["gl_valve_db"], key=f"gl_valve_db_ed_{ss.get('_ver_gl_valve_db', 0)}", num_rows="dynamic",
            width="stretch", column_config={
                "Manufacturer": st.column_config.SelectboxColumn(options=list(wgl.VALVE_MANUFACTURERS) + ["Other"]),
                "Valve type": st.column_config.SelectboxColumn(options=list(wgl.VALVE_KINDS)),
                "Port (64ths in)": st.column_config.NumberColumn(format="%g", min_value=1, max_value=128),
                "R value": st.column_config.NumberColumn(format="%.4f", min_value=0.0, max_value=0.99)}))
        wgl.register_valves(ed)
        st.download_button("Export your valves (CSV)", ed.to_csv(index=False).encode(), "gaslift_valves.csv",
                           disabled=not len(ed.dropna(how="all")))


GL_MANDREL_COLS = ["Measured depth (ft)", "dP casing (psi)", "Maximum gas (MMscf/d)", "Enabled"]


def _gl_mandrels_blank(n=12):
    df = pd.DataFrame({c: pd.Series([np.nan] * n, dtype=float) for c in GL_MANDREL_COLS[:3]})
    df["Enabled"] = pd.Series([True] * n, dtype=bool)
    df.index = pd.RangeIndex(1, n + 1, name="Mandrel")
    return df


def _gl_mandrels_from_valves():
    """Design | Existing mandrels - Transfer: the depths of the Equipment tab's gas lift valves."""
    src = ss.get("_tables", {}).get("gl_valves", ss.get("gl_valves"))
    mds = []
    if src is not None:
        mds = sorted(float(x) for x in pd.to_numeric(src["Measured depth (ft)"], errors="coerce").dropna() if x > 0)
    if not mds:
        ss["_gl_mand_msg"] = "No gas lift valves in the Equipment tab (Gas lift data) to transfer."
        return
    ss.pop("_gl_mand_msg", None)
    df = _gl_mandrels_blank(max(12, len(mds)))
    for i, m in enumerate(mds, 1):
        df.loc[i, "Measured depth (ft)"] = m
    ss["gl_mandrels"] = df
    ss["_ver_gl_mandrels"] = ss.get("_ver_gl_mandrels", 0) + 1


def _gl_transfer_design():
    out = ss.get("gl_res")
    if not out or out.get("res") is None:
        ss["_gl_tr_msg"] = "No gas lift design yet - run Design | Gas lift (continuous) in the Design tab first."
        return
    ss.pop("_gl_tr_msg", None)
    vtype_ = getattr(out["res"].inp, "valve_type", "")
    tub = vtype_ in wgl.VALVE_TYPES[1:]
    rows = [{"Measured depth (ft)": float(v.md),
             "Type": "Orifice" if v.kind == "Orifice" else ("Tubing" if tub else "Valve"),
             "Number of valves": int(getattr(v, "n_valves", 1)),
             "Port size (64ths in)": round(float(v.port_64), 2), "R value": float(v.r) if np.isfinite(v.r) else np.nan,
             "Dome pressure at 60 °F (psig)": round(wgls.system_dome(v, vtype_), 2) if v.kind != "Orifice" else np.nan}
            for v in out["res"].valves]
    df = _gl_valves_blank(max(10, len(rows)))
    for i, r in enumerate(rows, 1):
        for k, v in r.items():
            df.loc[i, k] = v
    ss["gl_valves"] = df
    ss["_ver_gl_valves"] = ss.get("_ver_gl_valves", 0) + 1
    put("glq_pc", float(out["res"].inp.p_inj))
    put("glq_dpv", float(out["res"].inp.dp_valve))
    put("glq_sg", float(val("gl_sg", 0.7)))


DIL_VARS = {"dil_pct": "Salt diluent rate percent (% of the produced liquid)", "dil_rate": "Diluent injection rate (STB/d)",
            "glr_dil": "GLR injected (scf/STB)"}


def diluent_data_panel(o, top_md, secs):
    """System | Diluent Injection (PROSPER; Tutorial 30): diluent water down the annulus, optionally with gas lift."""
    gl = getattr(o, "artificial_lift", "") == "diluent_gl"
    st.subheader("Diluent injection" + (" with gas lift" if gl else ""))
    st.caption("Water of its own salinity is injected down the tubing-casing annulus and enters the tubing at the "
               "diluent injection depth" + ("; lift gas enters at the gas lift valve depth above it" if gl else "")
               + ". Used by the system analysis (the rate and GLR can be sensitivity variables).")
    tub = secs[secs["Type"] == "Tubing"] if secs is not None and len(secs) else None
    bottom = float(tub["MD bottom (ft)"].max()) if tub is not None and len(tub) else 5000.0
    c = st.columns(4)
    dmd = num(c[0], "Diluent injection depth (ft MD)", "dil_md", float(round(bottom)), float(top_md) + 10, 60000.0)
    how = sel(c[1], "Diluent rate input", ["Rate (STB/d)", "Percent of the liquid"], "dil_how",
              help="PROSPER's 'Salt Diluent Rate Percent' sensitivity variable gives the diluent as a percentage of "
                   "the produced liquid rate.")
    if how == "Rate (STB/d)":
        rate = num(c[2], "Diluent injection rate (STB/d)", "dil_rate", 5000.0, 0.0, 1e6)
        pct = None
    else:
        pct = num(c[2], "Diluent rate (% of the produced liquid)", "dil_pct", 30.0, 0.0, 500.0)
        rate = float(val("dil_rate", 5000.0))
    sal = num(c[3], "Diluent water salinity (ppm)", "dil_sal", 20000.0, 0.0, 400000.0,
              help="PVT | Diluent Properties: diluent type water.")
    gas = wgl.LiftGas()
    glr, vmd = 0.0, dmd
    if gl:
        c = st.columns(4)
        gas = wgl.LiftGas(num(c[0], "Injected gas gravity", "dil_gsg", 0.7, 0.55, 2.0, None, "%.3f"),
                          num(c[2], "CO2 (%)", "dil_co2", 0.0, 0.0, 100.0), num(c[3], "N2 (%)", "dil_n2", 0.0, 0.0, 100.0),
                          num(c[1], "H2S (%)", "dil_h2s", 0.0, 0.0, 100.0))
        c = st.columns(4)
        glr = num(c[0], "GLR injected (scf/STB)", "dil_glr", 50.0, 0.0, 1e5)
        vmd = num(c[1], "Gas lift valve depth (ft MD)", "dil_vmd", min(7500.0, dmd), float(top_md) + 10, 60000.0)
        if vmd > dmd:
            st.warning("The gas lift valve is below the diluent injection depth.")
    c = st.columns(2)
    pb = sel(c[0], "Percent basis", ["produced liquid", "produced water", "total water"], "dil_pct_basis",
             help="PROSPER: the diluent is the percentage of the produced liquid rate. Alternatives: of the produced "
                  "water, or of the total water leaving the well (produced + diluent).")
    gb = sel(c[1], "GLR basis", ["produced liquid", "total liquid"], "dil_glr_basis", disabled=not gl,
             help="PROSPER: lift gas = GLR injected x the produced liquid rate. Alternative: x the total liquid in the "
                  "tubing (produced + diluent).")
    if dmd > bottom + 1.0:
        st.warning(f"The diluent injection depth is below the end of the tubing ({bottom:,.0f} ft MD).")
    setup = wdil.DiluentSetup(dil_md=dmd, rate=rate, pct=pct, salinity=sal, gas_lift=gl, gas=gas, glr_inj=glr,
                              valve_md=vmd, pct_basis=pb, glr_basis=gb)
    ss["diluent_setup"] = setup
    return setup


def temperature_model_panel(o, secs, top_md, bottom_md):
    """Data of the Improved Approximation (U by depth) or the Enthalpy Balance (completion, casing record,
    lithology). Stores ss['thermal_setup'] and returns the issues for the status table."""
    model = getattr(o, "temp_model", "rough") if o is not None else "rough"
    if model == "rough":
        ss.pop("thermal_setup", None)
        return []
    st.divider()
    iss = []
    if model == "improved":
        st.markdown("**Improved approximation - heat transfer coefficient by depth**")
        st.caption("PROSPER Tutorial 41: the energy balance with gravity, Joule-Thomson and heat loss to the geothermal "
                   "gradient above. Each row's U applies over the interval ending at its measured depth (U referred to "
                   "the flow ID, as the single U value of the Rough Approximation, which it replaces).")
        cc = st.columns([1, 1.3])
        with cc[0]:
            ed = grid("ihtc", {c_: st.column_config.NumberColumn(c_, format="%g") for c_ in IHTC_COLUMNS},
                      height=36 * 8)
        d = ed.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        md_, u_ = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
        if not len(d):
            iss.append(("error", "Enter at least one depth and heat transfer coefficient."))
        elif np.isnan(md_).any() or np.isnan(u_).any() or (u_ <= 0).any():
            iss.append(("error", "Every row needs a depth and a positive heat transfer coefficient."))
        elif not np.all(np.diff(md_) > 0):
            iss.append(("error", "Depths must increase down the table."))
        elif bottom_md is not None and md_.max() < bottom_md - 1:
            iss.append(("warning", f"The table stops at {md_.max():,.0f} ft; its last U is used down to "
                                   f"{bottom_md:,.0f} ft."))
        show_issues(iss)
        if not any(l == "error" for l, _ in iss):
            ss["thermal_setup"] = {"model": "improved", "md": md_, "u": u_}
        else:
            ss.pop("thermal_setup", None)
        return iss
    st.markdown("**Enthalpy balance - drilling, completion and lithology**")
    st.caption("PROSPER Tutorial 40: U at every depth from forced convection, the annulus (free convection and "
               "radiation in the completion fluid above the packer), casings, cement and mud, and transient "
               "conduction into the formation layer for the producing time; the same energy balance as the Improved "
               "Approximation. Calibrated on the user's T40 file - see the README.")
    c = st.columns(4)
    fluid_c = sel(c[0], "Completion fluid", list(web.COMPLETION_FLUIDS), "eb_cfluid")
    dens = num(c[1], "Completion fluid density (lb/US gal)", "eb_cdens", 12.0, 6.0, 25.0)
    packer = num(c[2], "Packer depth (ft MD)", "eb_packer", float(round(bottom_md or 10000.0)), 0.0, 60000.0)
    days = num(c[3], "Days since production started", "eb_days", float(val("ipr_tdays", 50.0)), 0.001, 1e5)
    c = st.columns(4)
    kcem = num(c[0], "Cement conductivity (BTU/h/ft/F)", "eb_kcem", 0.5, 0.01, 10.0, None, "%.3g")
    kcas = num(c[1], "Casing conductivity (BTU/h/ft/F)", "eb_kcas", 26.0, 0.1, 100.0, None, "%.3g")
    st.markdown("Drilling details (one row per casing string)")
    ed_d = grid("drill", {c_: st.column_config.NumberColumn(c_, format="%g") for c_ in DRILL_COLUMNS}, height=36 * 6)
    st.markdown("Lithology (one row per formation layer, top down)")
    ed_l = grid("litho", {"Formation Type": st.column_config.SelectboxColumn("Formation Type", options=list(web.ROCKS)),
                          "Bottom Depth (ft)": st.column_config.NumberColumn("Bottom Depth (ft)", format="%g"),
                          "Porosity (fraction)": st.column_config.NumberColumn("Porosity (fraction)", format="%g"),
                          "In Situ Fluid": st.column_config.SelectboxColumn("In Situ Fluid", options=list(web.IN_SITU))},
                height=36 * 6)
    casings = []
    for _, r in ed_d.iterrows():
        v = [pd.to_numeric(r[c_], errors="coerce") for c_ in DRILL_COLUMNS]
        if all(pd.isna(x) for x in v):
            continue
        if any(pd.isna(x) for x in v[:6]):
            iss.append(("error", "Each drilling row needs drilling depth, hole diameter, shoe, OD, weight and top "
                                 "of cement."))
            continue
        casings.append(web.Casing(*[float(x) for x in v[:6]], 0.0 if pd.isna(v[6]) else float(v[6]),
                                  10.0 if pd.isna(v[7]) else float(v[7])))
    litho = []
    for _, r in ed_l.iterrows():
        t_, b_ = r["Formation Type"], pd.to_numeric(r["Bottom Depth (ft)"], errors="coerce")
        if not isinstance(t_, str) or not t_:
            continue
        if pd.isna(b_):
            iss.append(("error", f"Lithology {t_}: enter the bottom depth."))
            continue
        ph = pd.to_numeric(r["Porosity (fraction)"], errors="coerce")
        litho.append(web.Lithology(float(b_), t_, 0.0 if pd.isna(ph) else float(ph),
                                   r["In Situ Fluid"] if isinstance(r["In Situ Fluid"], str) and r["In Situ Fluid"]
                                   else "Water"))
    tub = secs[secs["Type"] == "Tubing"] if secs is not None and len(secs) else None
    tod = None
    if tub is not None and len(tub) and "Tubing OD (in)" in tub and tub["Tubing OD (in)"].notna().any():
        tod = float(tub["Tubing OD (in)"].dropna().iloc[-1])
    tend = float(tub["MD bottom (ft)"].max()) if tub is not None and len(tub) else None
    if tod is None:
        iss.append(("info", "No tubing OD in the downhole equipment: tubing ID + 0.5 in is used for the annulus."))
    data = web.EnthalpyData(casings, litho, packer, fluid_c, dens, days, kcem, kcas, tubing_od=tod, tubing_end=tend)
    iss += data.check()
    if litho and bottom_md is not None and max(l.bottom for l in litho) < bottom_md - 1:
        iss.append(("warning", "The lithology stops above the bottom of the well; its last layer is used below."))
    show_issues(iss)
    if not any(l == "error" for l, _ in iss):
        ss["thermal_setup"] = {"model": "enthalpy", "data": data}
    else:
        ss.pop("thermal_setup", None)
    return iss


def with_thermal(path):
    """Attach the Improved Approximation / Enthalpy Balance model of the Equipment tab to a flow path."""
    th = ss.get("thermal_setup")
    o_ = ss.get("options")
    if th is None or path is None or getattr(o_, "temp_model", "rough") == "rough":
        return path
    if th["model"] == "improved":
        return dataclasses.replace(path, thermal={"model": "improved",
                                                  "u_seg": wth.u_segments(path, th["md"], th["u"])})
    return dataclasses.replace(path, thermal={"model": "enthalpy", "data": th["data"]})


MPUMP_MODELS = ["OneSubsea H800/65 (FMPP500) - fitted", "Constant efficiency", "User coefficients"]


def mpump_data_panel(o, surface_df):
    """System | Multiphase Pump (PROSPER; Tutorial 31): a surface pump in the flowline, driven by the power available."""
    st.subheader("Multiphase pump (surface)")
    st.caption("The pump sits in the surface equipment where its **Pump** row is (manifold first, then back towards "
               "the wellhead). It draws all the power available and adds dP = (1 - wear) x efficiency x power x 58,771 "
               "/ Qd (psi, Qd = in-situ liquid + free gas at the discharge, RB/d); efficiency = A (Qd / 10,000)^B "
               "(p_out / p_in)^C, clipped to its limits; the intake cannot fall below 0 psig. The power available can "
               "be a system sensitivity variable. Use the manifold as the top node in the Calculation tab.")
    has = any(r["Type"] == "Pump" for r in weq.surface_rows(surface_df)) if surface_df is not None else False
    if not has:
        st.warning("Add a **Pump** row to the surface equipment table at the pump's place in the flowline.")
    c = st.columns(4)
    power = num(c[0], "Power available (hp)", "mpump_power", 350.0, 1.0, 1e5)
    wear = num(c[1], "Pump wear factor", "mpump_wear", 0.0, 0.0, 0.99, 0.01, "%.3f",
               help="Fraction of the pressure gain lost to wear (0 = new pump).")
    model = sel(c[2], "Efficiency", MPUMP_MODELS, "mpump_model",
                help="OneSubsea H800/65 on the FMPP500 frame, single pump: coefficients fitted to PROSPER's "
                     "Tutorial 31 pump (pump dP within 1.3 % from 7,400 to 27,800 RB/d at 250-600 hp).")
    sel(c[3], "Pump configuration", ["Single pump"], "mpump_config", disabled=True)
    d = wmpump.MultiphasePump()
    c = st.columns(4)
    if model == MPUMP_MODELS[0]:
        a, b, cc = d.a, d.b, d.c
        c[0].text_input("A (%)", f"{a:.2f}", disabled=True)
        c[1].text_input("B (rate exponent)", f"{b:.4f}", disabled=True)
        c[2].text_input("C (pressure-ratio exponent)", f"{cc:.4f}", disabled=True)
    elif model == "Constant efficiency":
        a, b, cc = num(c[0], "Efficiency (%)", "mpump_eta", 30.0, 1.0, 100.0), 0.0, 0.0
    else:
        a = num(c[0], "A (%)", "mpump_a", d.a, 0.01, 100.0, None, "%.4g")
        b = num(c[1], "B (rate exponent)", "mpump_b", d.b, -3.0, 3.0, None, "%.4f")
        cc = num(c[2], "C (pressure-ratio exponent)", "mpump_c", d.c, -3.0, 3.0, None, "%.4f")
    c = st.columns(4)
    lo = num(c[0], "Minimum efficiency (%)", "mpump_eta_min", d.eta_min if model != "Constant efficiency" else 1.0,
             0.1, 100.0)
    hi = num(c[1], "Maximum efficiency (%)", "mpump_eta_max", d.eta_max if model != "Constant efficiency" else 100.0,
             0.1, 100.0)
    pump = wmpump.MultiphasePump(power, a, b, cc, lo, hi, wear,
                                 d.name if model == MPUMP_MODELS[0] else f"Multiphase pump ({model.lower()})")
    iss = pump.check()
    show_issues(iss)
    ss["mpump_setup"] = pump
    if not iss:
        qd = np.array([5000.0, 7500.0, 10000.0, 15000.0, 20000.0, 30000.0])
        st.dataframe(pd.DataFrame({"Discharge rate (RB/d)": qd,
                                   **{f"Efficiency (%) at p_out/p_in {r_:g}": pump.efficiency(qd, r_) for r_ in (1.5, 3.0)},
                                   **{f"dP (psi) at p_out/p_in {r_:g}": pump.dp(qd, r_) for r_ in (1.5, 3.0)}}
                                  ).style.format(precision=1), hide_index=True, width="stretch")
    return pump


def gaslift_data_panel(o, top_md, secs):
    """System | Gas Lift Data: the installed continuous gas lift used by the Calculation tab."""
    st.subheader("Gas lift data (continuous)")
    st.caption("The gas lift used by the Calculation tab (PROSPER System | Gas Lift Data). Design a new installation "
               "in the Design tab and transfer its valves here, or enter the valve depths of an existing well.")
    c = st.columns(3)
    ann = sel(c[0], "Gas lift type", list(wgls.ANNULUS), "glq_ann",
              help="No friction: static gas column in the annulus (most installations). Friction loss: the lift gas "
                   "flowing down the tubing-casing annulus loses pressure by friction (needs tubing OD and casing "
                   "ID). In both the annulus gas is at the flowing tubing temperature (PROSPER).")
    method = sel(c[1], "Gas lift method", list(wgls.METHODS), "glq_method", wgls.METHODS[2],
                 help="Fixed depth: all the gas enters at one depth (casing pressure assumed sufficient). Optimum "
                      "depth: the deepest point where casing pressure >= tubing pressure + valve dP. Valve depths "
                      "specified: that point moved up to the deepest valve above it. Multipoint: from the casing "
                      "head pressure every valve passes the gas its port allows - the lift gas rate is a result.")
    mp = method == wgls.METHODS[3]
    rate_mode = sel(c[2], "Input method", list(wgls.RATE_MODES), "glq_rate_mode", disabled=mp,
                    help="Injected gas rate (MMscf/d) or GLR injected (scf per STB of liquid). Either can also be a "
                         "sensitivity variable in the system calculation.")
    c = st.columns(5)
    gas = wgl.LiftGas(num(c[0], "Gas lift gas gravity", "glq_sg", float(val("gl_sg", 0.7)), 0.55, 2.0, None, "%.3f"),
                      num(c[2], "CO2 (%)", "glq_co2", 0.0, 0.0, 100.0), num(c[3], "N2 (%)", "glq_n2", 0.0, 0.0, 100.0),
                      num(c[1], "H2S (%)", "glq_h2s", 0.0, 0.0, 100.0))
    if rate_mode == wgls.RATE_MODES[0] or mp:
        qgi = num(c[4], "Injected gas rate (MMscf/d)", "glq_qgi", 1.0, 0.0, 200.0, None, "%.4g", disabled=mp)
        glr = float(val("glq_glr", 500.0))
    else:
        glr = num(c[4], "GLR injected (scf/STB)", "glq_glr", 500.0, 0.0, 1e6)
        qgi = float(val("glq_qgi", 1.0))
    tub = secs[secs["Type"] == "Tubing"] if secs is not None and len(secs) else None
    bottom = float(tub["MD bottom (ft)"].max()) if tub is not None and len(tub) else 5000.0
    c = st.columns(4)
    pc = dp = max_md = fixed = np.nan
    if method != wgls.METHODS[0]:
        pc = num(c[0], "Casing head pressure (psia)", "glq_pc", float(val("gl_pinj", 1214.7)), 14.7, 2e4,
                 help="Injection pressure at the casing head (PROSPER: Casing Pressure). For a casing-operated "
                      "design, subtract the closing drop of the unloading valves above the operating valve.")
    if method in wgls.METHODS[1:3]:
        dp = num(c[1], "dP across valve (psi)", "glq_dpv", float(val("gl_dpv", 100.0)), 0.0, 2000.0)
    if method == wgls.METHODS[1]:
        max_md = num(c[2], "Maximum depth of injection (ft MD)", "glq_maxmd", float(round(bottom - 30.0)),
                     float(top_md) + 10, 60000.0)
    if method == wgls.METHODS[0]:
        fixed = num(c[0], "Depth of injection (ft MD)", "glq_fixmd", float(round(bottom - 30.0)), float(top_md) + 10,
                    60000.0)
    if mp:
        c2 = st.columns(4)
        hp = flag(c2[0].checkbox, "Dome correction > 1200 psig (Winkler-Eads)", "glq_dome_hp", True)
        derate = num(c2[1], "Thornhill-Craver de-rating (%)", "glq_derate", 0.0, 0.0, 90.0)
    else:
        hp, derate = True, 0.0
    valves = ()
    if method in wgls.METHODS[2:]:
        st.button("Transfer from the gas lift design", on_click=_gl_transfer_design, key="glq_transfer",
                  help="Valve depths, ports, R values and dome pressures of the last design (Design tab), with its "
                       "operating injection pressure and valve dP.")
        if ss.get("_gl_tr_msg"):
            st.warning(ss["_gl_tr_msg"])
        if "gl_valves" not in ss:
            ss["gl_valves"] = _gl_valves_blank()
        if "Number of valves" not in ss["gl_valves"].columns:      # projects saved before the column existed
            ss["gl_valves"] = ss["gl_valves"].copy()
            ss["gl_valves"].insert(2, "Number of valves", 1.0)
        cfg = {"Type": st.column_config.SelectboxColumn(
            "Type", options=list(wgls.VALVE_KINDS),
            help="Valve: casing-sensitive (IPO). Tubing: tubing-sensitive (opened by the tubing pressure). Orifice.")}
        for c_ in GL_VALVE_COLS:
            if c_ != "Type":
                cfg[c_] = st.column_config.NumberColumn(c_, format="%g")
        ed = track("gl_valves", st.data_editor(ss["gl_valves"], key=f"gl_valves_ed_{ss.get('_ver_gl_valves', 0)}",
                                               num_rows="dynamic", width="stretch", column_config=cfg))
        rows = ed.dropna(subset=["Measured depth (ft)"])
        out = []
        for _, r in rows.iterrows():
            kind = r["Type"] if r["Type"] in wgls.VALVE_KINDS else "Valve"
            port = float(r["Port size (64ths in)"]) if pd.notna(r["Port size (64ths in)"]) else np.nan
            if mp and not np.isfinite(port):
                st.error(f"Valve at {r['Measured depth (ft)']:g} ft: enter the port size (multipoint).")
            if mp and kind in ("Valve", "Tubing") and not pd.notna(r["Dome pressure at 60 °F (psig)"]):
                st.error(f"Valve at {r['Measured depth (ft)']:g} ft: enter the dome pressure (multipoint).")
            out.append(wgls.GLValve(float(r["Measured depth (ft)"]), kind, port if np.isfinite(port) else 16.0,
                                    float(r["R value"]) if pd.notna(r["R value"]) else 0.0,
                                    float(r["Dome pressure at 60 °F (psig)"]) if pd.notna(
                                        r["Dome pressure at 60 °F (psig)"]) else np.nan,
                                    int(r["Number of valves"]) if pd.notna(r.get("Number of valves", np.nan))
                                    and r.get("Number of valves", 0) >= 1 else 1))
        valves = tuple(sorted(out, key=lambda v: v.md))
        if not valves:
            st.warning("Enter the gas lift valve depths (or transfer them from the design).")
        if mp:
            st.caption("Multipoint valve model: orifice - Thornhill-Craver through the port. Casing-sensitive valve "
                       "- closed at or below Pd (dome at valve temperature), no gas up to Pvo = (Pd - R Pt)/(1 - R), "
                       "then throttling with gas in proportion to Pc - Pvo (0.247 MMscf/d per psi per in2 of port) up "
                       "to the full-port Thornhill-Craver rate. Fitted to PROSPER's multipoint VLP (Tutorial 23, "
                       "Camco R-20); PROSPER's Petex throttling model itself is not published.")
    fric = ann == wgls.ANNULUS[1]
    if fric and secs is not None and len(secs):
        t_ = secs[secs["Type"] == "Tubing"]
        if not len(t_) or t_["Tubing OD (in)"].isna().any() or t_["Casing ID (in)"].isna().any():
            st.error("Friction loss in annulus needs the tubing OD and casing ID of every tubing section "
                     "(Downhole equipment).")
    setup = wgls.GasLiftSetup(method=method, gas=gas, rate_mode=rate_mode, qgi=qgi, glr_inj=glr,
                              p_casing=pc if np.isfinite(pc) else 1214.7, dp_valve=dp if np.isfinite(dp) else 100.0,
                              max_md=max_md if np.isfinite(max_md) else 1e9, fixed_md=fixed, valves=valves,
                              annulus_friction=fric, dome_high_p=hp, tc_derate=derate)
    ss["gl_setup"] = setup
    return setup


def _ct_transfer_design():
    out = ss.get("ct_res")
    if not out or out.get("res") is None:
        ss["_ct_tr_msg"] = "No coiled tubing design yet - run Design | Gas lift (coiled tubing) first."
        return
    ss.pop("_ct_tr_msg", None)
    r = out["res"]
    put("ctq_method", wct.METHODS[0])
    put("ctq_md", float(round(r.max_md, 2)))
    put("ctq_pc", float(r.p_inj_actual))
    put("ctq_dpv", float(r.inp.dp_valve))
    put("ctq_qgi", float(round(r.qgi_actual, 4)))


def coiled_data_panel(o, top_md, secs):
    """System | Coiled Tubing Data: the installed coiled tubing gas lift used by the Calculation tab."""
    st.subheader("Coiled tubing gas lift data")
    st.caption("PROSPER System | Coiled Tubing Data: the lift gas goes down a coiled tubing inside the tubing and the "
               "well produces up the coiled tubing - tubing annulus above the injection point (the full tubing below "
               "it). Design a new installation in the Design tab and transfer the injection depth here.")
    c = st.columns(3)
    fric = sel(c[0], "Coiled tubing type", list(wct.FRICTION), "ctq_fric",
               help="No friction: static gas column in the coiled tubing. Friction loss: the lift gas flowing down "
                    "the coiled tubing ID loses pressure by friction.")
    method = sel(c[1], "Lift method", list(wct.METHODS), "ctq_method", wct.METHODS[0],
                 help="Specified injection depth: the gas enters at the coiled tubing end if its pressure there is at "
                      "least the tubing pressure + dP across the valve, else not at all. Optimum injection depth: the "
                      "deepest point down to the coiled tubing end where that holds.")
    rate_mode = sel(c[2], "Input method", list(wgls.RATE_MODES), "ctq_rate_mode")
    c = st.columns(5)
    gas = wgl.LiftGas(num(c[0], "Gas lift gas gravity", "ctq_sg", 0.7, 0.55, 2.0, None, "%.3f"),
                      num(c[2], "CO2 (%)", "ctq_co2", 0.0, 0.0, 100.0), num(c[3], "N2 (%)", "ctq_n2", 0.0, 0.0, 100.0),
                      num(c[1], "H2S (%)", "ctq_h2s", 0.0, 0.0, 100.0))
    if rate_mode == wgls.RATE_MODES[0]:
        qgi = num(c[4], "Injected gas rate (MMscf/d)", "ctq_qgi", 1.0, 0.0, 200.0, None, "%.4g")
        glr = float(val("ctq_glr", 500.0))
    else:
        glr = num(c[4], "GLR injected (scf/STB)", "ctq_glr", 500.0, 0.0, 1e6)
        qgi = float(val("ctq_qgi", 1.0))
    c = st.columns(4)
    ct = wct.CoiledTubing(num(c[0], "Coiled tubing inside diameter (in)", "ctq_id", 0.8, 0.1, 5.0, None, "%.4g"),
                          num(c[1], "Coiled tubing thickness (in)", "ctq_wall", 0.2, 0.01, 1.0, None, "%.4g"),
                          num(c[2], "Inside roughness (in)", "ctq_ein", 0.0006, 0.0, 0.1, None, "%.5f"),
                          num(c[3], "Outside roughness (in)", "ctq_eout", 0.0006, 0.0, 0.1, None, "%.5f"),
                          fric == wct.FRICTION[1])
    tub = secs[secs["Type"] == "Tubing"] if secs is not None and len(secs) else None
    bottom = float(tub["MD bottom (ft)"].max()) if tub is not None and len(tub) else 5000.0
    c = st.columns(3)
    depth = num(c[0], "Coiled tubing specified depth (ft MD)" if method == wct.METHODS[0] else
                "Maximum depth of injection (ft MD)", "ctq_md", float(round(bottom - 30.0)), float(top_md) + 10,
                60000.0, help="The coiled tubing end: the injection depth (specified) or the deepest injection "
                              "point (optimum).")
    pc = num(c[1], "Gas injection pressure (psia)", "ctq_pc", 1214.7, 14.7, 2e4,
             help="Pressure of the lift gas at the coiled tubing head.")
    dp = num(c[2], "dP across valve (psi)", "ctq_dpv", 50.0, 0.0, 2000.0,
             help="The gas pressure at the injection point must exceed the tubing pressure by at least this.")
    st.button("Transfer from the coiled tubing design", on_click=_ct_transfer_design, key="ctq_transfer",
              help="Maximum operating depth (as the specified depth), injection pressure, dP and the gas through the "
                   "selected orifice of the last coiled tubing design.")
    if ss.get("_ct_tr_msg"):
        st.warning(ss["_ct_tr_msg"])
    if tub is not None and len(tub) and (tub["Flow ID (in)"] <= ct.od_in * 1.03).any():
        st.error(f"The coiled tubing OD ({ct.od_in:g} in) does not fit inside the tubing.")
    st.caption(f"Coiled tubing OD {ct.od_in:g} in. Lift gas in the coiled tubing at the flowing temperature less "
               f"{wct.CT_T_FRAC:g} x (flowing - formation temperature) - PROSPER Tutorial 25.")
    m = wgls.METHODS[0] if method == wct.METHODS[0] else wgls.METHODS[1]
    setup = wgls.GasLiftSetup(method=m, gas=gas, rate_mode=rate_mode, qgi=qgi, glr_inj=glr, p_casing=pc, dp_valve=dp,
                              max_md=depth, fixed_md=depth, coiled=ct)
    ss["gl_setup"] = setup
    return setup


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
        c_u = st.columns(4)
        u_geo = num(c_u[0], "Overall heat transfer coefficient (BTU/h/ft2/F)", "eq_u", 8.0, 0.0, 1000.0,
                    None, "%.5g")
        o_eq = ss.get("options")
        if o_eq is not None and o_eq.is_injector:
            num(c_u[1], "Injected fluid temperature at the wellhead (°F)", "eq_tinj", 60.0, -40.0, 400.0, None,
                "%.4g", help="Temperature of the injected fluid entering the wellhead; it exchanges heat with the "
                             "formation on the way down.")
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
        status["Temperature model"] = temperature_model_panel(o, secs, top_md, bottom_md)

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
    if o is not None and getattr(o, "artificial_lift", "none") == "esp":
        st.divider()
        esp_equipment_panel(top_md, secs)
    else:
        ss.pop("esp_equipment", None)
    if o is not None and getattr(o, "artificial_lift", "none") == "srp":
        st.divider()
        srp_equipment_panel(top_md, secs)
    else:
        ss.pop("srp_equipment", None)
    if o is not None and getattr(o, "artificial_lift", "none") == "jet":
        st.divider()
        jet_equipment_panel(top_md, secs)
    else:
        ss.pop("jet_equipment", None)
    if o is not None and getattr(o, "artificial_lift", "none") in ("pcp_motor", "pcp_rod"):
        st.divider()
        pcp_equipment_panel(o, top_md, secs)
    else:
        ss.pop("pcp_equipment", None)
    if o is not None and getattr(o, "artificial_lift", "none") == "hsp":
        st.divider()
        hsp_equipment_panel(top_md, secs)
    else:
        ss.pop("hsp_equipment", None)
    if o is not None and getattr(o, "artificial_lift", "none") in ("diluent", "diluent_gl"):
        st.divider()
        diluent_data_panel(o, top_md, secs)
    else:
        ss.pop("diluent_setup", None)
    if o is not None and getattr(o, "artificial_lift", "none") == "foam_lift":
        st.divider()
        foam_data_panel()
    else:
        ss.pop("foam_setup", None)
    if o is not None and getattr(o, "artificial_lift", "none") == "mpump":
        st.divider()
        mpump_data_panel(o, ed_s)
    else:
        ss.pop("mpump_setup", None)
    if o is not None and getattr(o, "artificial_lift", "none") == "gas_lift":
        st.divider()
        gaslift_data_panel(o, top_md, secs)
    elif o is not None and getattr(o, "artificial_lift", "none") == "ct_gas_lift":
        st.divider()
        coiled_data_panel(o, top_md, secs)
    else:
        ss.pop("gl_setup", None)
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
        _, items = wsurf.build_items(df, ss.get("mpump_setup"))
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


def vlp_path(eqd, flow_type, extra_md):
    top, secs, pts = eqd["downhole"]
    smd, stvd = eqd["survey"]
    gmd, gt = eqd["geothermal"]
    return with_thermal(wvlp.build_flow_path(top, secs, pts, smd, stvd, gmd, gt, flow_type, extra_md))


def rebuild_ipr(changes):
    """Rebuild the tab-3 IPR with some arguments changed (pr, skin, j). Raises ValueError if not possible."""
    rec = ss.get("ipr_recipe")
    if rec is None:
        raise ValueError("No IPR recipe - recalculate the IPR tab.")
    fn, a, k = rec
    b = inspect.signature(fn).bind(*a, **k)
    changes = dict(changes)
    if "wc" in changes and "wc" not in inspect.signature(fn).parameters:
        changes.pop("wc")                          # models without a water-cut input ignore it
    gor_new = changes.pop("gor", None)
    params0 = inspect.signature(fn).parameters
    if gor_new is not None and "glr" in params0:          # CBM Producer: the GLR sets the water saturation
        changes["glr"] = float(gor_new)
        gor_new = None
    if gor_new is not None and "mu" in params0 and "bo" in params0 and ss.get("ipr_pvt_ctx") is not None:
        # PROSPER re-evaluates a Darcy-type PI with the oil at the calculation's GOR (and its bubble point)
        o_, t_ = ss["ipr_pvt_ctx"]
        pv0, _ = oil_pvt_at(o_, 1000.0, t_)
        if pv0 is not None and abs(float(gor_new) - float(pv0.get("gor", gor_new))) > 1e-6:
            changes["_gor"] = float(gor_new)
    if not changes:
        if ss.get("ipr_curve") is not None:
            return ss["ipr_curve"]
        c = fn(*b.args, **b.kwargs)
        return wcomp.apply_sand_control(c, ss["ipr_sc_dp"]) if ss.get("ipr_sc_dp") is not None else c
    gor_eval = changes.pop("_gor", None)
    if not changes and gor_eval is None:
        if ss.get("ipr_curve") is not None:
            return ss["ipr_curve"]
    for key, v in changes.items():
        if key not in inspect.signature(fn).parameters:
            raise ValueError(f"The {ss.get('ipr_meta', {}).get('model', 'current')} IPR has no '{key}' input.")
        b.arguments[key] = v
    # A new reservoir pressure changes the oil viscosity and FVF the Darcy-type PI is evaluated with (PROSPER
    # recalculates them at the reservoir pressure).
    params = inspect.signature(fn).parameters
    if ("pr" in changes or gor_eval is not None) and "mu" in params and "bo" in params \
            and ss.get("ipr_pvt_ctx") is not None:
        o_, t_ = ss["ipr_pvt_ctx"]
        b.apply_defaults()
        pv_, _ = oil_pvt_at(o_, float(b.arguments["pr"]), t_, gor_eval)
        if pv_ is not None:
            b.arguments["mu"], b.arguments["bo"] = pv_["mu"], pv_["bo"]
            if gor_eval is not None and "pb" in params:
                b.arguments["pb"] = float(pv_["pb"])
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


def comp_current():
    """Complementary-calculation options (Calculation tab, Options) from the stored inputs."""
    return wcx.CompOptions(sand_density=float(val("comp_rho_s", 2.65)), sand_rate=float(val("comp_w", 0.0)),
                           c_factor=float(val("comp_c", 400.0)), s_factor=float(val("comp_s", 0.05)),
                           turner=float(val("comp_turner", 2.04)), wall=str(val("comp_wall", "Carbon steel")),
                           deformation=str(val("comp_def", "Ductile")), particle_mm=float(val("comp_dp", 250.0)) / 1000,
                           geometry_factor=float(val("comp_gf", 2.0)), reducer_angle=float(val("comp_ang", 45.0)),
                           inhibited_rate=float(val("comp_irc", 0.0)), inhibitor_eff=float(val("comp_eff", 0.0)),
                           alkalinity=float(val("comp_alk", 0.0)), hydrate=hydrate_curve_current())


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
            d.update({"Water Gas Ratio (STB/MMscf)": r["wgr"], "Condensate Gas Ratio (STB/MMscf)": r["cgr"],
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


INJ_VLP_ACTIONS = ["Estimate U Value", "Match VLP", "VLP/IPR", "Adjust IPR"]


def inj_tests_key(gas):
    return "vlp_tests_inj_" + ("gas" if gas else "water")


def inj_blank(gas):
    cols = winj.inj_test_columns(gas)
    df = pd.DataFrame({c: pd.Series([None] * VLP_ROWS, dtype=object) if c in ("Test Point Date", "Test Point Comment")
                       else pd.Series([np.nan] * VLP_ROWS, dtype=float) for c in cols})
    df.index = pd.RangeIndex(1, VLP_ROWS + 1, name="Test")
    return df


def inj_set_tests(gas, df):
    k = inj_tests_key(gas)
    ss[k] = df
    ss["_ver_" + k] = ss.get("_ver_" + k, 0) + 1
    for k_ in ("inj_match", "inj_vi", "inj_adj"):
        ss.pop(k_, None)


def accept_inj_match(p1, p2):
    put("inj_p1", float(round(p1, 6)))
    put("inj_p2", float(round(p2, 6)))


def accept_inj_ipr(key, value):
    put(key, float(value))
    ss["inj_adj_accepted"] = (key, float(value))
    ss["inj_action"] = "VLP/IPR"


def injector_vlp_match(o, eqd):
    """PROSPER VLP/IPR Matching for injectors (Tutorials 21 and 22): Estimate U Value (bottom-node temperature),
    Match VLP (gravity multiplier on the gauge pressures), VLP/IPR at each test, Adjust IPR (skin or reservoir
    pressure per test)."""
    gas = o.well_type == "gas_injector"
    fl_ = inj_gas_fluid() if gas else winj.WaterFluid(float(val("pvt_sal", INPUT_DEFAULTS["pvt_sal"])))
    qunit = "MMscf/d" if gas else "STB/d"
    u_cur = float(val("eq_u", eqd.get("u_geo", 8.0)))
    t_inj = float(val("eq_tinj", 60.0))
    hc = eqd["heat_capacities"]
    p1c, p2c = float(val("inj_p1", 1.0)), float(val("inj_p2", 1.0))
    top_md, secs_, _ = eqd["downhole"]
    md_bot = float(secs_["MD bottom (ft)"].max())
    st.caption(f"{'Gas' if gas else 'Water'} injection well tests. The injected fluid pressure is at the wellhead; "
               "the downstream temperature is the temperature at the bottom of the tubing (PROSPER's estimate of U "
               f"for an injector). Current U {u_cur:.5g} BTU/h/ft2/F, injection temperature {t_inj:g} °F (Equipment "
               f"tab); VLP multipliers gravity x {p1c:.5g}, friction x {p2c:.5g}.")
    key = inj_tests_key(gas)
    ss.setdefault(key, inj_blank(gas))
    cfg = {c_: st.column_config.NumberColumn(c_.split(" (")[0] + "\n(" + c_.split(" (", 1)[1] if " (" in c_ else c_,
                                             format="%g") for c_ in winj.inj_test_columns(gas)[2:]}
    tbl = track(key, st.data_editor(ss[key], key=f"{key}_ed_{ss.get('_ver_' + key, 0)}", num_rows="fixed",
                                    width="stretch", column_config=cfg, height=36 * 11))
    st.button("Clear tests", on_click=inj_set_tests, args=(gas, inj_blank(gas)))
    tests, issues = winj.clean_inj_tests(tbl, gas, float(top_md), md_bot)
    show_vlp_issues(issues)
    if any(l_ == "error" for l_, _ in issues):
        st.info("Fix the test table errors above to continue.")
        return
    if not len(tests):
        st.info("Enter at least one test.")
        return
    try:
        path = winj.from_flow_path(vlp_path(eqd, "tubing", [m for m in tests["gauge_md"] if np.isfinite(m)]))
    except ValueError as e:
        st.error(str(e))
        return
    st.divider()
    bc = st.columns(len(INJ_VLP_ACTIONS))
    for i, name in enumerate(INJ_VLP_ACTIONS):
        bc[i].button(name, width="stretch", key=f"inj_act_{i}",
                     on_click=lambda n=name: ss.__setitem__("inj_action", n),
                     type="primary" if ss.get("inj_action", INJ_VLP_ACTIONS[0]) == name else "secondary")
    action = ss.get("inj_action", INJ_VLP_ACTIONS[0])
    st.subheader(action)
    tsig = (tests.to_json(), fl_.rate_unit, str(getattr(fl_, "inp", getattr(fl_, "salinity", ""))), t_inj,
            str(hc), tuple(np.round(path.md, 2)), tuple(np.round(path.d, 6)))

    if action == "Estimate U Value":
        tab_u = winj.estimate_u(path, fl_, tests, t_inj, hc)
        good = tab_u["U (BTU/h/ft2/F)"].dropna()
        st.dataframe(tab_u.style.format({"Downstream temperature (°F)": "{:.1f}", "U (BTU/h/ft2/F)": "{:.5f}"},
                                        na_rep="—"), hide_index=True, width="stretch")
        if not len(good):
            st.warning("No test has a usable downstream temperature.")
            return
        u_new = float(good.mean())
        m = st.columns(4)
        m[0].metric("Mean U", f"{u_new:.5g}")
        m[1].metric("Standard deviation", f"{float(good.std(ddof=1)) if len(good) > 1 else 0.0:.3g}")
        m[2].metric("Current U (Equipment tab)", f"{u_cur:.5g}")
        m[3].button(f"Transfer mean U = {u_new:.5g}", type="primary", on_click=accept_u, args=(u_new,),
                    help="Writes the mean to the Geothermal gradient section of the Equipment tab.")
        if abs(u_new - u_cur) < 1e-4:
            st.success(f"U = {u_cur:.5g} BTU/h/ft2/F is in use.")
        return

    if action == "Match VLP":
        st.caption("Parameter 1 multiplies the gravity pressure change and Parameter 2 the friction one. The "
                   "traverse runs from the injected fluid pressure down to each gauge at the current U; Parameter 1 "
                   "is regressed on the gauge pressures (PROSPER's Tutorials 21 and 22 end with Parameter 2 = 1 - "
                   "friction is a small part of an injector's pressure change).")
        if not np.isfinite(tests["gauge_p"]).any():
            st.warning("No test has gauge data - matching needs a measured downhole pressure.")
            return
        fit2 = st.checkbox("Also regress Parameter 2 (friction)", key="inj_fit_p2", value=False)
        msig = tsig + (u_cur, fit2)
        if ss.get("inj_match", (None,))[0] != msig:
            with st.spinner("Matching..."):
                ss["inj_match"] = (msig, winj.match_vlp(path, fl_, tests, u_cur, t_inj, hc, fit2))
        mt = ss["inj_match"][1]
        g = tests[np.isfinite(tests["gauge_p"])].reset_index(drop=True)
        m = st.columns(4)
        m[0].metric("Parameter 1 (gravity)", f"{mt.p1:.5f}")
        m[1].metric("Parameter 2 (friction)", f"{mt.p2:.5f}")
        m[2].metric("Standard deviation (psi)", f"{mt.std:.4g}")
        m[3].button("Accept match", type="primary", on_click=accept_inj_match, args=(mt.p1, mt.p2),
                    help="Applies the multipliers to the injector's system and gradient calculations.")
        if abs(mt.p1 - p1c) < 1e-6 and abs(mt.p2 - p2c) < 1e-6:
            st.success("This match is applied to the calculations.")
        st.dataframe(pd.DataFrame({"Test": g["label"], "Gauge depth (ft)": g["gauge_md"],
                                   "Measured (psia)": g["gauge_p"], "Unmatched - measured (psi)": mt.before,
                                   "Matched - measured (psi)": mt.after}).style.format(precision=2),
                     hide_index=True, width="stretch")
        return

    ipr_for = ss.get("inj_ipr_for")
    if ipr_for is None or ss.get("inj_ipr") is None:
        st.info("Set up the injectivity in the IPR Data tab first.")
        return
    cur_ipr = ss["inj_ipr"]
    isig = (str(cur_ipr.model), float(cur_ipr.pr), float(getattr(cur_ipr, "skin", 0.0)), str(cur_ipr.params))

    def ipr_t(t, **kw):
        kw.setdefault("pr", float(t["pr"]) if np.isfinite(t["pr"]) else float(cur_ipr.pr))
        if "skin" not in kw and getattr(cur_ipr, "skin", None) is not None and cur_ipr.model != "P.I. Entry":
            kw["skin"] = float(cur_ipr.skin)
        return ipr_for(**kw)

    if action == "VLP/IPR":
        st.caption("Each test: the system (tubing with the VLP multipliers, injectivity at the test's reservoir "
                   "pressure) at the test's injected fluid pressure. The measured bottom-hole pressure is the VLP's "
                   "at the measured rate.")
        if ss.get("inj_adj_accepted"):
            k_, v_ = ss.pop("inj_adj_accepted")
            st.success(f"{'Skin' if k_ == 'ipr_skin_inj' else 'Reservoir pressure'} = {v_:.5g} written to the "
                       "IPR Data tab.")
        vsig = tsig + (u_cur, p1c, p2c, isig)
        if ss.get("inj_vi", (None,))[0] != vsig:
            out = []
            with st.spinner("Solving each test..."):
                for _, t in tests.iterrows():
                    ipr = ipr_t(t)
                    res, rates, vp, ip = winj.solve(path, fl_, t["whp"], ipr, u_cur, t_inj, hc, p1c, p2c)
                    bhp = float(winj.vlp_curve(path, fl_, t["whp"], [t["q"]], u_cur, t_inj, hc, p1c, p2c)[0])
                    out.append((t, res, rates, vp, ip, bhp))
            ss["inj_vi"] = (vsig, out)
        out = ss["inj_vi"][1]
        rows = []
        for t, res, rates, vp, ip, bhp in out:
            qc = res.q if res is not None else np.nan
            rows.append({"Test": t["label"], f"Measured rate ({qunit})": t["q"], f"Calculated rate ({qunit})": qc,
                         "Rate difference (%)": 100 * (qc - t["q"]) / t["q"],
                         "Measured BHP (psia)": bhp, "Calculated BHP (psia)": res.pwf if res is not None else np.nan,
                         "BHP difference (%)": 100 * ((res.pwf if res is not None else np.nan) - bhp) / bhp})
        st.dataframe(pd.DataFrame(rows).style.format(precision=3, na_rep="—"), hide_index=True, width="stretch")
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for i, (t, res, rates, vp, ip, bhp) in enumerate(out):
            col = wplt.SERIES[i % len(wplt.SERIES)]
            ax.plot(rates, vp, color=col, lw=1.6, label=f"test {t['label']} VLP")
            ax.plot(rates, ip, color=col, lw=1.0, ls="--", label=f"test {t['label']} IPR")
            ax.scatter([t["q"]], [bhp], color=col, marker="s", s=40, zorder=5, edgecolor=wplt.INK)
            if res is not None:
                ax.scatter([res.q], [res.pwf], facecolor="white", edgecolor=col, s=50, lw=1.8, zorder=5)
        ax.set_xlabel(f"Injection rate ({qunit})")
        ax.set_ylabel("Bottom-hole pressure (psia)")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=7, ncol=2)
        ax.set_title("VLP/IPR by test (squares: measured, circles: calculated)", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        return

    # Adjust IPR
    pi_entry = cur_ipr.model == "P.I. Entry"
    opts = (["Reservoir pressure"] if pi_entry else ["Skin", "Reservoir pressure"])
    what = st.radio("Adjust", opts, horizontal=True, key="inj_adj_what")
    st.caption("For each test, the " + ("skin" if what == "Skin" else "reservoir pressure") + " at which the system "
               "(tubing with the VLP multipliers at the test's injected fluid pressure) takes the measured rate. "
               "Accept writes the mean of the selected tests to the IPR Data tab."
               + (" P.I. Entry has no skin." if pi_entry else ""))
    w_ = "skin" if what == "Skin" else "pr"
    asig = tsig + (u_cur, p1c, p2c, isig, w_)
    if ss.get("inj_adj", (None,))[0] != asig:
        vals = []
        with st.spinner("Adjusting..."):
            for _, t in tests.iterrows():
                f_ = (lambda t_: (lambda **kw: ipr_t(t_, **kw)))(t)
                vals.append(winj.adjust_ipr(path, fl_, t, f_, w_, u_cur, t_inj, hc, p1c, p2c))
        ss["inj_adj"] = (asig, vals)
    vals = ss["inj_adj"][1]
    lab = "Skin" if w_ == "skin" else "Reservoir pressure (psia)"
    st.dataframe(pd.DataFrame({"Test": tests["label"], f"Measured rate ({qunit})": tests["q"], lab: vals})
                 .style.format(precision=4, na_rep="not found"), hide_index=True, width="stretch")
    pick = st.multiselect("Tests to average", tests["label"].tolist(), default=tests["label"].tolist(),
                          key="inj_adj_pick")
    sel_ = [v for l_, v in zip(tests["label"], vals) if l_ in pick and np.isfinite(v)]
    if sel_:
        mean_ = float(np.mean(sel_))
        st.button(f"Accept {lab.split(' (')[0].lower()} = {mean_:.5g}", type="primary", on_click=accept_inj_ipr,
                  args=("ipr_skin_inj" if w_ == "skin" else "ipr_pr", mean_))
        if len(sel_) > 1 and w_ == "skin" and np.ptp(sel_) > 1:
            st.caption("The skins differ from test to test: the near-well conditions are changing with time "
                       "(cooling, scaling or plugging around an injector), as PROSPER's Tutorials 21 and 22 discuss.")


with tabs[4]:
    st.header("VLP")
    o = ss.get("options")
    eqd = ss.get("equipment", {})
    if o is None or o.errors:
        st.info("Fix the configuration in the Options summary first.")
    elif o.is_injector:
        if not eqd.get("valid") or eqd.get("survey") is None or not len(eqd["downhole"][1]):
            st.info("Complete the Equipment Data tab first (every section in its status table must be OK).")
        else:
            injector_vlp_match(o, eqd)
    elif not eqd.get("valid") or eqd.get("survey") is None or not len(eqd["downhole"][1]):
        st.info("Complete the Equipment Data tab first (every section in its status table must be OK).")
    else:
        kind = "gas" if o.is_gas else "oil"
        st.caption(("Gas" if kind == "gas" else "Oil") + f" well test data. Pressures in {ss['_units']['pressure']}; the tubing is calculated "
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
                                                                  travs[c_][1][i], c_, 1.0, 1.0, u_cur, labels_,
                                                                  comp=comp_current())
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


def _fmt_result(v):
    """Result value: thousands separators without exponent for large numbers, 4 significant figures otherwise."""
    v = float(v)
    if not np.isfinite(v):
        return str(v)
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"


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
    rows += [("Stability", "Stable" if r.stable else "UNSTABLE - left-hand intersection (VLP crosses the IPR from above)"),
             ("Rate at VLP minimum", f"{r.q_vlp_min:,.4g} {'STB/d' if kind == 'oil' else 'MMscf/d'}")]
    out = pd.DataFrame(rows, columns=["Result", "Value"])
    out["Value"] = [_fmt_result(v_) if isinstance(v_, (int, float, np.floating)) else str(v_) for v_ in out["Value"]]
    return out


def comp_panel():
    """PROSPER System | Solids / calculation Options: inputs of the complementary calculations."""
    with st.expander("Options - grain size, erosion, liquid loading, corrosion"):
        st.caption("Reported along the well in the gradient results and, as the worst value along the well, in the "
                   "solution details. Grain size: largest sand grain the flow carries (Wasp, Kenny & Gandhi; oil and "
                   "gas forms). Erosional velocity: API RP 14E, Ve = C / sqrt(rho_m); with a sand rate also the "
                   "ConocoPhillips form Ve = S D sqrt(rho_m) / sqrt(W), the smaller one used. Liquid loading: Turner, "
                   "Vt = C sigma^0.25 (rho_l - rho_g)^0.25 / rho_g^0.5 against the superficial gas velocity.")
        c = st.columns(3)
        num(c[0], "Sand density (g/cm3)", "comp_rho_s", 2.65, 1.0, 5.0, None, "%.3f")
        num(c[1], "Turner constant", "comp_turner", 2.04, 0.1, 50.0, None, "%.3f",
            help="2.04 with interfacial tension in dyne/cm (PROSPER); Turner's paper prints 20.4.")
        c = st.columns(3)
        num(c[0], "Sand production rate (lbm/day)", "comp_w", 0.0, 0.0, 1e6, None, "%.4g",
            help="0: API 14E only. Above 0 the ConocoPhillips sand-laden velocity is also computed.")
        num(c[1], "C-factor", "comp_c", 400.0, 1.0, 5000.0, None, "%.4g",
            help="API RP 14E: 100 continuous, 125 intermittent, 150-200 inhibited; PROSPER default 400.")
        num(c[2], "S-factor", "comp_s", 0.05, 1e-4, 10.0, None, "%.4g",
            help="ConocoPhillips geometry factor (pipe: 0.05).")
        st.markdown("**Erosion rate - DNV-RP-O501** (needs a sand production rate)")
        c = st.columns(3)
        mats = list(wcx.WALL_MATERIALS)
        wall = sel(c[0], "Wall material", mats, "comp_wall", "Carbon steel",
                   help="DNV-RP-O501 Table 3-2 (density, K, n).")
        sel(c[1], "Deformation", ["Ductile", "Brittle"], "comp_def", wcx.WALL_MATERIALS[wall][3],
            help="Angle dependency F(alpha): ductile for steels and polymers, brittle for carbides and ceramics.")
        num(c[2], "Particle diameter (micron)", "comp_dp", 250.0, 1.0, 5000.0, None, "%.4g",
            help="Not a PROSPER input; DNV default 250 micron. Used by the bend and reducer models.")
        c = st.columns(3)
        num(c[0], "Geometry factor GF", "comp_gf", 2.0, 1.0, 4.0, None, "%.3g",
            help="Not a PROSPER input; DNV recommends 2 when unknown (1 reproduces the DNV worked examples).")
        num(c[1], "Reducer angle (deg)", "comp_ang", 45.0, 1.0, 90.0, None, "%.3g",
            help="Contraction angle of tubing ID reductions and restrictions (not a PROSPER input).")
        st.markdown("**Corrosion rate - NORSOK M-506** (needs water and CO2 in the PVT)")
        c = st.columns(3)
        num(c[0], "Inhibited rate (0.001 in/yr)", "comp_irc", 0.0, 0.0, 1e5, None, "%.4g")
        num(c[1], "Inhibitor efficiency (percent)", "comp_eff", 0.0, 0.0, 100.0, None, "%.4g",
            help="CR = IRC x EFF + URC x (1 - EFF), URC the NORSOK uninhibited rate (PROSPER).")
        num(c[2], "Alkalinity, pH buffer (kg/kg)", "comp_alk", 0.0, 0.0, 0.05, None, "%.4g",
            help="Bicarbonate in the produced water (mass fraction). 0: unbuffered water, lowest pH.")
    return comp_current()


CALC_MODES = ["System analysis (IPR + VLP)", "Gradient traverse", "Lift curves (VFP tables)", "Choke performance"]
QL_MODE = "Gas lift QuickLook"
ESPQL_MODE = "ESP QuickLook"
HSPQL_MODE = "HSP QuickLook"
EQC_MODE = "Gas lift equilibrium curve"


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


def esp_traverse(path, fluid, kind, base, q, corr, p1, p2, hc, surface, esp):
    """Gradient traverse of an ESP well at one rate: tubing (top node to the pump discharge), the pump, casing (pump
    intake to the bottom node). Same outputs as wsys.solution_details plus pump rows in det."""
    md_p = esp["pump_md"]
    above, below = wesp.sub_path(path, path.md[0], md_p), wesp.sub_path(path, md_p, path.md[-1])
    det, reg, prof = wsys.solution_details(above, fluid, kind, base, q, corr, p1, p2, hc, surface, comp=comp_current())
    case = {"q": np.array([q]), "wc": np.array([base["wc"] / 100.0]), "gor": np.array([float(base["gor"])])}
    out = wesp.bhp_with_pump(path, fluid, np.array([det["Wellhead pressure (psia)"]]), case, esp, corr, p1, p2,
                             base["u"], hc)
    pip, pdp = float(out["pip"][0]), float(out["pdp"][0])
    if not np.isfinite(pip):
        raise ValueError(f"at {q:g} STB/d the pump would draw the intake below atmospheric (pumped off) - the rate "
                         "is below what this pump delivers against the tubing.")
    t_dis = float(out["t_discharge"][0])
    det, reg, prof = wsys.solution_details(above, fluid, kind, {**base, "gor": float(out["gor_above"][0])}, q,
                                           corr, p1, p2, hc, surface, comp=comp_current(), t_bottom=t_dis)
    det_b, _, prof_b = wsys.solution_details(below, fluid, kind, {**base, "whp": pip}, q, corr, p1, p2, hc, None,
                                             comp=comp_current())
    z_p = float(np.interp(md_p, path.md, path.tvd))
    pump_row = pd.DataFrame([{"MD (ft)": md_p, "TVD (ft)": z_p, "Pressure (psia)": pdp, "Temperature (°F)": t_dis,
                              "Section": "Pump discharge"},
                             {"MD (ft)": md_p, "TVD (ft)": z_p, "Pressure (psia)": pip,
                              "Temperature (°F)": float(out["t_intake"][0]), "Section": "Pump intake"}])
    prof = pd.concat([prof.assign(Section="Tubing"), pump_row, prof_b.assign(Section="Casing")], ignore_index=True)
    det = dict(det)
    det["Bottom node pressure (psia)"] = float(out["bhp"][0])
    det["dP gravity - production string (psi)"] += det_b["dP gravity - production string (psi)"]
    det["dP friction - production string (psi)"] += det_b["dP friction - production string (psi)"]
    det["Pump discharge pressure (psia)"], det["Pump intake pressure (psia)"] = pdp, pip
    det["Pump dP (psi)"] = pdp - pip
    det["Pump intake temperature (°F)"], det["Pump discharge temperature (°F)"] = float(out["t_intake"][0]), t_dis
    if "level" in esp:                        # CBM: the liquid level is entered, the casing head pressure follows
        det["Liquid level in annulus (ft TVD)"] = float(esp["level"])
        det["Casing head pressure (psia)"] = wcbm.casing_head_pressure(
            fluid, pip, float(out["t_intake"][0]), float(det.get("Wellhead temperature (°F)", path.t_form[0])),
            float(np.interp(md_p, path.md, path.tvd)), float(esp["level"]))
        det["Gas rate up the annulus (MMscf/d)"] = q * float(base["gor"]) / 1e6
        return det, reg, prof
    det["Liquid level in annulus (ft TVD)"] = wesp.liquid_level(
        fluid, pip, float(out["t_intake"][0]), case, float(np.interp(md_p, path.md, path.tvd)), esp.get("chp", 264.7),
        float(path.tvd[0]))
    return det, reg, prof


def jet_traverse(path, fluid, kind, base, q, corr, p1, p2, hc, jet):
    """Gradient traverse of a jet pump well at one liquid rate (as PROSPER's): tubing from the top node to the pump
    discharge with production + power fluid, the pump (intake from the discharge, Ps = Pd (1 + H) - H Pn), casing
    from the intake down to the bottom node."""
    md_p = jet["pump_md"]
    above, below = wesp.sub_path(path, path.md[0], md_p), wesp.sub_path(path, md_p, path.md[-1])
    wc, gor = base["wc"] / 100.0, float(base["gor"])
    case = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([gor])}
    w, cp = fluid.mass_cp(case, hc)
    t_in = float(wvlp.temperature_profile(below, np.asarray(w, float), cp, float(base["u"]))[0, 0])
    qt = q + jet["q_pf"]
    wct = (q * wc + jet["q_pf"]) / qt
    fl_mix = wjet._blend_fluid(fluid, q, wc, jet["q_pf"], jet["sal"])
    det, reg, prof = wsys.solution_details(above, fl_mix, kind, {**base, "wc": 100 * wct}, qt, corr, p1, p2, hc,
                                           None, comp=comp_current(), t_bottom=t_in)
    pdis = float(prof["Pressure (psia)"].iloc[-1])
    t_tub = np.interp(above.md, prof["MD (ft)"].to_numpy(), prof["Temperature (°F)"].to_numpy())
    pump = wjet.JetPumpInstalled(md_p, jet["an"], jet["at"], jet["q_pf"], jet["p_surf"], jet["losses"],
                                 wjet.PowerFluid(jet["sal"]), jet["casing_id"], jet["tubing_od"], jet["name"])
    pu = wjet.intake_from_discharge(path, fluid, q, wc, gor, pdis, t_tub, t_in, pump, hc)
    pip = pu["pip"]
    det_b, _, prof_b = wsys.solution_details(below, fluid, kind, {**base, "whp": pip}, q, corr, p1, p2, hc, None,
                                             comp=comp_current())
    z_p = float(np.interp(md_p, path.md, path.tvd))
    pump_row = pd.DataFrame([{"MD (ft)": md_p, "TVD (ft)": z_p, "Pressure (psia)": pdis, "Temperature (°F)": t_in,
                              "Section": "Pump discharge"},
                             {"MD (ft)": md_p, "TVD (ft)": z_p, "Pressure (psia)": pip, "Temperature (°F)": t_in,
                              "Section": "Pump intake"}])
    prof = pd.concat([prof.assign(Section="Tubing"), pump_row, prof_b.assign(Section="Casing")], ignore_index=True)
    det = dict(det)
    det["Bottom node pressure (psia)"] = float(prof_b["Pressure (psia)"].iloc[-1])
    det["dP gravity - production string (psi)"] += det_b["dP gravity - production string (psi)"]
    det["dP friction - production string (psi)"] += det_b["dP friction - production string (psi)"]
    det["Pump discharge pressure (psia)"], det["Pump intake pressure (psia)"] = pdis, pip
    det["Pump dP (psi)"] = pdis - pip
    det["Pump intake temperature (°F)"] = t_in
    det["Nozzle injection pressure (psia)"] = pu["pn"]
    det["Power fluid static (psi)"], det["Power fluid friction (psi)"] = pu["static"], -pu["friction"]
    det["Flow ratio M"], det["Head ratio H"] = pu["M"], pu["H"]
    return det, reg, prof


def gl_lift_for(eqd, o, gl, extra=()):
    """Flow path with the gas lift depths as nodes, and the lift setup with its annulus."""
    path = vlp_path(eqd, o.flow_type, tuple(gl.extra_md()) + tuple(extra))
    return path, dataclasses.replace(gl, annulus=wgls.annulus_geometry(path, eqd["downhole"][1]))


VLPC_MODE = "VLP (tubing curves)"


def vlp_curve_mode(col, kind, fluid, eqd, o, base, corr, p1, p2, surface):
    """PROSPER Calculation | VLP (tubing curves) with its flow-assurance flags (Tutorial 43): bottom-hole pressure
    against rate, the minimum point of stability, and per rate the Turner, erosional-velocity, PE5 stability and
    hydrate flags."""
    hc = eqd["heat_capacities"]
    with col:
        st.markdown("**Rates**")
        how = sel(st, "Rate method", ["Automatic - geometric", "User selected"], "vlpc_how")
        if how == "User selected":
            rates = num_list(st, f"Rates ({_qunit(kind)})", f"vlpc_rates_{kind}",
                             "1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5" if kind == "gas" else "250, 500, 1000, 2000, 4000")
        else:
            c = st.columns(3)
            lo = num(c[0], f"First rate ({_qunit(kind)})", f"vlpc_lo_{kind}", 0.136 if kind == "gas" else 50.0,
                     1e-4, 1e6, None, "%.4g")
            hi = num(c[1], f"Last rate ({_qunit(kind)})", f"vlpc_hi_{kind}", 135.854 if kind == "gas" else 20000.0,
                     1e-3, 1e7, None, "%.5g")
            n_ = int(num(c[2], "Number of rates", "vlpc_n", 20.0, 3.0, 40.0, 1.0, "%.0f"))
            rates = list(np.geomspace(lo, hi, n_)) if hi > lo else []
        rates = [float(r) for r in rates if r > 0][:40]
        st.caption("Each rate is a full gradient traverse from the top node with the inputs on the left; the flags "
                   "are raised if they apply at any node (Calculation options: Turner constant and erosional C "
                   "factor; PVT tab: hydrate curve).")
        go = st.button("Calculate VLP", type="primary", disabled=not rates, key="vlpc_go")
    sig = (kind, tuple(sorted(base.items())), corr, p1, p2, tuple(np.round(rates, 6)), repr(surface),
           repr(comp_current()))
    if go:
        try:
            path = vlp_path(eqd, o.flow_type, ())
            out = []
            with st.spinner(f"Calculating {len(rates)} traverses..."):
                for q in rates:
                    det, reg, prof = wsys.solution_details(path, fluid, kind, base, q, corr, p1, p2, hc, surface,
                                                          comp=comp_current())
                    out.append((q, det))
            ss["vlpc_res"] = {"sig": sig, "res": out}
        except (ValueError, KeyError, TypeError) as e:
            st.error(f"VLP calculation failed: {e}")
    vr = ss.get("vlpc_res")
    if not vr:
        return
    if vr["sig"] != sig:
        st.info("Inputs have changed since the last calculation - press Calculate VLP to update.")
    res = vr["res"]
    q_ = np.array([q for q, _ in res])
    p_ = np.array([d["Bottom node pressure (psia)"] for _, d in res])
    k_min = int(np.argmin(p_))
    rows = []
    for (q, d), pv in zip(res, p_):
        r = {f"Rate ({_qunit(kind)})": q}
        r.update({k_: v_ for k_, v_ in wsys.surface_rates(kind, base, q).items() if not k_.startswith("Gas rate")
                  or kind != "gas"})
        r.update({"VLP pressure (psia)": pv, "Wellhead temperature (°F)": d["Wellhead temperature (°F)"],
                  "dP friction (psi)": d["dP friction - production string (psi)"],
                  "dP gravity (psi)": d["dP gravity - production string (psi)"],
                  "Max. no-slip velocity (ft/s)": d["Max. no-slip velocity (ft/s)"],
                  "Min. erosional velocity (ft/s)": d["Min. erosional velocity (ft/s)"],
                  "Max. Turner velocity (ft/s)": d.get("Max. Turner velocity (ft/s)", np.nan),
                  "Min. Turner velocity (ft/s)": d.get("Min. Turner velocity (ft/s)", np.nan),
                  "Erosional velocity flag": d.get("Erosional velocity flag", ""),
                  "Turner velocity flag": d.get("Turner velocity flag", ""),
                  "PE5 stability flag": d.get("PE5 stability flag", ""),
                  "Hydrates formation flag": d.get("Hydrates formation flag", "(no curve)"),
                  "VLP shape": "Minimum" if q == q_[k_min] else ("Unstable (left of minimum)" if q < q_[k_min]
                                                                 else "Stable")})
        rows.append(r)
    tab = pd.DataFrame(rows)
    m = st.columns(4)
    m[0].metric("Minimum point of stability", f"{q_[k_min]:,.4g} {_qunit(kind)}",
                help="The VLP minimum: rates to the left are in the gravity-dominated (unstable) region.")
    m[1].metric("VLP pressure at the minimum", f"{p_[k_min]:,.1f} psia")
    tf = [q for q, d in res if d.get("Turner velocity flag") == "Yes"]
    m[2].metric("Turner flag up to", f"{max(tf):,.4g} {_qunit(kind)}" if tf else "none")
    us = [q for q, d in res if d.get("PE5 stability flag") == "Unstable"]
    m[3].metric("PE5 unstable up to", f"{max(us):,.4g} {_qunit(kind)}" if us else "none")
    if k_min in (0, len(q_) - 1):
        st.caption("The minimum is at the end of the rate range - extend the range to locate it.")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axvspan(q_.min(), q_[k_min], color=wplt.SERIES[1], alpha=0.08, label="unstable (left of minimum)")
    ax.plot(q_, p_, color=wplt.SERIES[0], lw=1.8, marker="o", ms=3, label="VLP")
    ax.scatter([q_[k_min]], [p_[k_min]], s=60, facecolor="white", edgecolor=wplt.INK, lw=1.8, zorder=5,
               label="minimum point of stability")
    for flag_, mk_, lab_ in (("Turner velocity flag", "v", "Turner flag"), ("Erosional velocity flag", "^", "erosional flag")):
        sel_ = [i for i, (q, d) in enumerate(res) if d.get(flag_) == "Yes"]
        if sel_:
            ax.scatter(q_[sel_], p_[sel_], marker=mk_, s=36, color=wplt.SERIES[2 if mk_ == "v" else 3], zorder=4,
                       label=lab_)
    ax.set_xlabel(f"Rate ({_qunit(kind)})")
    ax.set_ylabel("VLP (bottom node) pressure (psia)")
    if q_.max() / max(q_.min(), 1e-9) > 30:
        ax.set_xscale("log")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.dataframe(tab.style.format(precision=3), hide_index=True, width="stretch")
    st.download_button("Download VLP table (CSV)", tab.to_csv(index=False).encode(), "vlp_tubing_curves.csv")


def traverse_mode(col, kind, fluid, eqd, o, base, corr, p1, p2, surface, esp=None, gl=None, jet=None, pcp=None,
                  hsp=None):
    """Gradient traverse: pressure, temperature, holdup and velocities along the well at chosen rates."""
    hc = eqd["heat_capacities"]
    with col:
        st.markdown("**Rates**")
        dflt_r = "500, 1000, 2000, 3000" if kind == "oil" else "2, 5, 10, 20"
        rates = num_list(st, f"Rates ({_qunit(kind)})", f"trav_rates_{kind}", dflt_r, "comma separated - up to 6")
        rates = [r for r in rates if r > 0][:6]
        st.caption("The traverse starts at the top node " + ("(the manifold) and works back through the surface "
                   "equipment, then " if surface is not None else "(the wellhead) and runs ") +
                   "down the tubing to the bottom node, with the inputs on the left.")
        if gl is not None:
            st.caption(f"With gas lift from the Equipment tab ({gl.method.lower()}" + (
                f", casing head {gl.p_casing:,.1f} psia" if gl.method != wgls.METHODS[0] else "") + (
                f", {gl.qgi:g} MMscf/d" if gl.rate_mode == wgls.RATE_MODES[0] and not gl.multipoint else
                (f", GLR injected {gl.glr_inj:g} scf/STB" if not gl.multipoint else "")) + ").")
            if surface is not None:
                st.caption("Surface equipment is not used with gas lift: the traverse starts at the wellhead.")
        if esp:
            st.caption(f"With the ESP from the Equipment tab: {esp['pump'].name}, {esp['stages']} stages at "
                       f"{esp['freq']:g} Hz, set at {esp['pump_md']:,.0f} ft MD. Above the pump the flow is in the "
                       "tubing, below it in the casing; the pressure steps up across the pump.")
        if jet:
            st.caption(f"With the jet pump from the Equipment tab: {jet['name']} at {jet['pump_md']:,.0f} ft MD, "
                       f"{jet['q_pf']:,.0f} STB/d of power fluid at {jet['p_surf']:,.1f} psia. Above the pump the "
                       "tubing carries production + power fluid, below it the casing carries production; the pressure "
                       "steps up across the pump.")
            if surface is not None:
                st.caption("Surface equipment is not used with the jet pump: the traverse starts at the wellhead.")
        if pcp:
            st.caption(f"With the PCP from the Equipment tab: {pcp['pump']} at {pcp['speed']:,.1f} rpm, set at "
                       f"{pcp['pump_md']:,.0f} ft MD. Above the pump the flow is in the "
                       + ("rod-tubing annulus" if pcp["drive"] == "Sucker rod" else "tubing") + ", below it in the "
                       "casing; the pressure steps up across the pump by the head at which it passes the rate.")
            if surface is not None:
                st.caption("Surface equipment is not used with the PCP: the traverse starts at the wellhead.")
        if hsp:
            st.caption(f"With the HSP from the Equipment tab: {hsp['pump']} ({hsp['stages']} stages) and "
                       f"{hsp['turbine']} ({hsp['turbine_stages']} stages) at {hsp['speed']:,.0f} rpm, set at "
                       f"{hsp['pump_md']:,.0f} ft MD. At each rate the power fluid is the rate whose turbine power "
                       "equals the pump power; above the pump the tubing carries production + power fluid.")
            if surface is not None:
                st.caption("Surface equipment is not used with the HSP: the traverse starts at the wellhead.")
        go = st.button("Calculate traverse", type="primary", disabled=not rates)
    sig = (kind, tuple(sorted(base.items())), corr, p1, p2, tuple(rates), repr(surface), o.flow_type,
           None if not jet else tuple(sorted((k_, repr(v_)) for k_, v_ in jet.items())),
           None if not esp else (esp["pump_md"], esp["freq"], esp["stages"], esp["pump"].name, esp["sep"],
                                 esp["wear"], esp.get("chp")), repr(gl),
           None if not pcp else tuple(sorted((k_, repr(v_)) for k_, v_ in pcp.items())),
           None if not hsp else tuple(sorted((k_, repr(v_)) for k_, v_ in hsp.items())))
    if go:
        try:
            path = vlp_path(eqd, o.flow_type, (esp["pump_md"],) if esp else ((jet["pump_md"],) if jet else (
                (pcp["pump_md"],) if pcp else ((hsp["pump_md"],) if hsp else ()))))
            if gl is not None:
                path, gl_ = gl_lift_for(eqd, o, gl)
            out = []
            with st.spinner("Calculating traverses..."):
                for q in rates:
                    if gl is not None:
                        det, prof = wgls.details(path, fluid, base, q, gl_, corr, p1, p2, float(base["u"]), hc)
                        det["Max. velocity / erosional velocity"] = np.nan
                        prof["Erosional velocity (ft/s)"] = np.nan
                        reg = {}
                    elif esp:
                        det, reg, prof = esp_traverse(path, fluid, kind, base, q, corr, p1, p2, hc, surface, esp)
                    elif jet:
                        det, reg, prof = jet_traverse(path, fluid, kind, base, q, corr, p1, p2, hc, jet)
                    elif pcp:
                        det, reg, prof = pcp_traverse(path, fluid, kind, base, q, corr, p1, p2, hc, pcp)
                    elif hsp:
                        det, reg, prof = hsp_traverse(path, fluid, kind, base, q, corr, p1, p2, hc, hsp)
                    else:
                        det, reg, prof = wsys.solution_details(path, fluid, kind, base, q, corr, p1, p2, hc, surface,
                                                              comp=comp_current())
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
        for k_ in ("Surface pump intake pressure (psia)", "Surface pump discharge pressure (psia)",
                   "Surface pump efficiency (%)"):
            if k_ in det:
                r[k_] = det[k_]
        for k_ in ("Gas lift gas injection rate (MMscf/d)", "Injection depth (ft MD)",
                   "Tubing pressure at injection (psia)", "Casing pressure at injection (psia)", "Multipoint flag",
                   "Pump discharge pressure (psia)", "Pump intake pressure (psia)", "Pump dP (psi)",
                   "Pump intake temperature (°F)", "Pump discharge temperature (°F)",
                   "Liquid level in annulus (ft TVD)", "Casing head pressure (psia)", "Gas rate up the annulus (MMscf/d)",
                   "Nozzle injection pressure (psia)", "Power fluid static (psi)",
                   "Power fluid friction (psi)", "Flow ratio M", "Head ratio H", "PCP head (ft)",
                   "PCP average rate (RB/d)", "HSP power fluid rate (STB/d)", "HSP pump power (hp)",
                   "Power fluid injection pressure (psia)", "Delta P across turbine (psi)"):
            if k_ in det:
                r[k_] = det[k_]
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
    for i, (q, det, reg, prof) in enumerate(res):
        if "Casing pressure (psia)" in prof and prof["Casing pressure (psia)"].notna().any():
            ax[0].plot(prof["Casing pressure (psia)"], prof["TVD (ft)"], color=wplt.SERIES[i % len(wplt.SERIES)],
                       lw=1.0, ls="--")
        z_i = det.get("Injection depth (ft TVD)")
        if z_i is not None and np.isfinite(z_i):
            ax[0].plot([det["Tubing pressure at injection (psia)"]], [z_i], "o", ms=5,
                       color=wplt.SERIES[i % len(wplt.SERIES)])
    if res and "Pump intake pressure (psia)" in res[0][1]:
        z_p = float(res[0][3].loc[res[0][3]["Section"] == "Pump intake", "TVD (ft)"].iloc[0])
        for a_ in ax:
            a_.axhline(z_p, color=wplt.INK2, lw=0.8, ls="-.")
        ax[0].text(ax[0].get_xlim()[0], z_p, " Jet pump" if jet else (" PCP" if pcp else (" HSP" if hsp else " ESP")), va="bottom", fontsize=8,
                   color=wplt.INK2)
    for a_, t_ in zip(ax, ("Pressure (psia)" + (" (dashed: casing)" if gl is not None else ""), "Temperature (°F)",
                           "Liquid holdup (dotted: no-slip)",
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
    if det.get("_note"):
        st.warning(det["_note"])
    if det.get("_valves") is not None:
        st.markdown("**Gas lift valves** (multipoint)")
        st.dataframe(det["_valves"].style.format(precision=3), hide_index=True, width="stretch")
    surf_tab = det.get("_surface")
    if surf_tab is not None:
        st.markdown("**Surface equipment** (wellhead to manifold)")
        st.dataframe(surf_tab.style.format(precision=2), hide_index=True, width="stretch")
    if reg:
        st.caption("Restrictions: " + "; ".join(f"{k} - {v[0]}, dP {v[1]:.1f} psi" for k, v in reg.items()))
    st.markdown("**Tubing profile** (segment properties are given at the lower node of each segment)" if
                "Section" not in prof else "**Profile** - tubing to the pump, then casing below it (segment "
                "properties at the lower node of each segment)")
    st.dataframe(prof.style.format(precision=3, na_rep="—"), hide_index=True, width="stretch", height=380)
    allp = pd.concat([p_.assign(**{f"Rate ({_qunit(kind)})": q_}) for q_, _, _, p_ in res], ignore_index=True)
    st.download_button("Download all profiles (CSV)", allp.to_csv(index=False).encode(), "gradient_traverse.csv")


def ql_plot(res, path, valves_tab, gauges, inj_md):
    """PROSPER-style QuickLook plot: tubing and casing gradients top down and bottom up against TVD."""
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    pr = res.profile
    z = pr["TVD (ft)"]
    ax.plot(pr["Casing pressure, top down (psia)"], z, color="#c62828", lw=1.8, label="Casing pressure (top down)")
    ax.plot(pr["Casing pressure, bottom up (psia)"], z, color="#1565c0", lw=1.4, ls="--",
            label="Casing pressure (bottom up)")
    ax.plot(pr["Tubing pressure, top down (psia)"], z, color="#00838f", lw=1.8, label="Tubing pressure (top down)")
    ax.plot(pr["Tubing pressure, bottom up (psia)"], z, color="#2e7d32", lw=1.4, ls="--",
            label="Tubing pressure (bottom up)")
    z_inj = float(np.interp(inj_md, path.md, path.tvd))
    ok = res.injection_possible
    ax.axhline(z_inj, color="#ef6c00" if ok else "#b71c1c", lw=1.0)
    ax.text(ax.get_xlim()[1], z_inj, ("Injection depth " if ok else "INJECTION NOT POSSIBLE - injection depth ")
            + f"{inj_md:,.0f} ft MD", ha="right", va="bottom", fontsize=8, color="#ef6c00" if ok else "#b71c1c")
    if valves_tab is not None:
        for _, v in valves_tab.iterrows():
            ax.axhline(v["TVD (ft)"], color=wplt.INK2, lw=0.6, ls=":")
            ax.text(ax.get_xlim()[0], v["TVD (ft)"], f" {v['Status']} {v['Type']} {int(v['Valve'])}", va="bottom",
                    fontsize=7.5, color=wplt.INK2)
            if np.isfinite(v.get("Opening pressure (psia)", np.nan)):
                ax.plot(v["Opening pressure (psia)"], v["TVD (ft)"], "o", color="#43a047", ms=5)
                ax.plot(v["Closing pressure (psia)"], v["TVD (ft)"], "x", color="#e53935", ms=6)
    if gauges:
        ax.plot([p for _, p in gauges], [float(np.interp(m, path.md, path.tvd)) for m, _ in gauges], "s",
                color="#6a1b9a", ms=6, label="Flowing measurements")
    ax.set_xlabel("Pressure (psia)")
    ax.set_ylabel("True vertical depth (ft)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Gas lift QuickLook (valves: o opening, x closing pressure)", fontsize=10)
    fig.tight_layout()
    return fig


def eqc_plot(df, top_md, whp):
    """PROSPER-style equilibrium curve: valve tubing pressures against depth (the curve), the casing pressure line
    from each casing head pressure down to its valve, and the depth of each point."""
    G = 14.696
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ok = df[df["Liquid rate (STB/d)"].notna()]
    if len(ok):
        d = np.r_[top_md, ok["Injection depth (ft MD)"].to_numpy()]
        p = np.r_[whp, ok["Valve tubing pressure (psia)"].to_numpy()] - G
        ax.plot(p, d, "s-", color="#1f3fbf", lw=2, label="Equilibrium curve")
        for _, r in ok.iterrows():
            ax.plot([r["Casing head pressure (psia)"] - G, r["Valve casing pressure (psia)"] - G],
                    [top_md, r["Injection depth (ft MD)"]], "s-", color="#d62728", lw=1.6, ms=5)
            ax.axhline(r["Injection depth (ft MD)"], color="#9b59b6", ls="--", lw=0.8)
            ax.annotate(f"{r['Injection depth (ft MD)']:,.0f} ft, Qinj {r['Gas injection rate (MMscf/d)']:.3g} "
                        f"MMscf/d, QL {r['Liquid rate (STB/d)']:,.0f} STB/d",
                        (1.0, r["Injection depth (ft MD)"]), xycoords=("axes fraction", "data"), ha="right",
                        va="bottom", fontsize=7, color="#7d3c98")
        ax.plot([], [], "s-", color="#d62728", label="Casing pressure (CHP to the valve)")
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Pressure (psig)")
    ax.set_ylabel("Measured depth (ft)")
    ax.grid(ls=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def equilibrium_mode(col, fluid, eqd, o, base, corr, p1, p2, gl, ipr_curve):
    """Gas lift equilibrium curve (PROSPER: Sensitivity on injection depth): the system solved with the lift gas
    injected at depths from the top node to the maximum depth of injection."""
    G = 14.696
    with col:
        st.caption("At each injection depth the system (IPR + VLP, bottom node) is solved with the lift gas injected "
                   "there; the curve is the tubing pressure at the injection point against depth, and the casing "
                   "head pressure is the one that gives tubing + dP across the valve at that depth (lift gas column, "
                   "annulus friction and temperature from the Gas lift data). Validated on the user's PROSPER "
                   "Tutorial 23 run: rates within 2.5 %, valve tubing pressures within 15 psi, casing head within "
                   "13 psi.")
        c = st.columns(3)
        sel(c[0], "Equilibrium curve method", ["Sensitivity on injection depth"], "eqc_method")
        rm = sel(c[1], "Injection rate method", list(wgls.EQ_RATE_METHODS), "eqc_rate_mode",
                 help="Fixed injection rate (validated against PROSPER), or a fixed GLR injected (scf/STB of liquid).")
        if rm == wgls.EQ_RATE_METHODS[0]:
            qgi = num(c[2], "Gas injection rate (MMscf/d)", "eqc_qgi", float(gl.qgi), 0.001, 500.0, None, "%.5g")
            glr = None
        else:
            glr = num(c[2], "GLR injected (scf/STB)", "eqc_glr", float(gl.glr_inj), 1.0, 1e5, None, "%.5g")
            qgi = 0.0
        c = st.columns(3)
        sel(c[0], "Valve settings", ["dP across valve"], "eqc_vset")
        dpv = num(c[1], "dP across valve (psi)", "eqc_dp", float(gl.dp_valve), 0.0, 5000.0)
        top_, secs_, _ = eqd["downhole"]
        tub = secs_[secs_["Type"] == "Tubing"]
        deep = float(tub["MD bottom (ft)"].max()) if len(tub) else float(secs_["MD bottom (ft)"].max())
        mx0 = float(gl.max_md) if np.isfinite(gl.max_md) and gl.max_md < 1e8 else deep
        max_md = num(c[2], "Maximum depth of injection (ft MD)", "eqc_max", min(mx0, deep), 1.0, 60000.0)
        c = st.columns(3)
        npts = int(num(c[0], "Number of injection points", "eqc_n", 10.0, 2.0, 50.0, 1.0, "%.0f"))
        method = sel(c[1], "Rate method", RATE_METHODS, "sys_rate_method")
        user_rates = None
        if method == "User selected":
            user_rates = num_list(c[2], "Rates (STB/d)", "sys_rates", "100, 500, 1000, 2000, 3000")
        go = st.button("Calculate equilibrium curve", type="primary")
        sig = (repr(base), corr, p1, p2, rm, qgi, glr, dpv, max_md, npts, method, repr(user_rates), repr(gl),
               id(ipr_curve))
        if go:
            if max_md > deep + 1e-6:
                st.error(f"The maximum depth of injection is below the deepest tubing ({deep:,.0f} ft MD).")
                return
            def lift_for(s_, extra):
                return gl_lift_for(eqd, o, s_, extra)
            def ipr_for(ch):
                if not ch:
                    return ipr_curve
                try:
                    return rebuild_ipr(ch)
                except ValueError:
                    if set(ch) <= {"wc", "gor"}:
                        return ipr_curve
                    raise
            top_md = float(vlp_path(eqd, o.flow_type, ()).md[0])
            try:
                with st.spinner(f"Solving {npts} injection depths..."):
                    df, _ = wgls.equilibrium_curve(lift_for, fluid, ipr_for, base, gl, corr, p1, p2,
                                                   eqd["heat_capacities"], qgi=qgi, glr_inj=glr, dp_valve=dpv,
                                                   max_md=max_md, top_md=top_md, n=npts, rate_method=method,
                                                   user_rates=user_rates)
                ss["eqc_res"] = {"sig": sig, "df": df, "top_md": top_md, "whp": float(base["whp"])}
            except (ValueError, TypeError, KeyError) as e:
                st.error(f"Equilibrium curve failed: {e}")
        out = ss.get("eqc_res")
        if out is None:
            st.info("Press Calculate equilibrium curve.")
            return
        if out["sig"] != sig:
            st.info("Inputs have changed since the last calculation - press Calculate equilibrium curve to update.")
        df = out["df"]
        if df["Liquid rate (STB/d)"].isna().all():
            st.warning("No solution at any injection depth: the well does not flow with this lift gas.")
        st.pyplot(eqc_plot(df, out["top_md"], out["whp"]), clear_figure=True)
        show = df.copy()
        for c_ in ("Casing head pressure (psia)", "Valve casing pressure (psia)", "Valve tubing pressure (psia)"):
            show[c_.replace("psia", "psig")] = show.pop(c_) - G
        st.dataframe(show.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
        st.download_button("Download equilibrium curve (CSV)", df.to_csv(index=False).encode(),
                           "equilibrium_curve.csv")


def _espql_transfer_wear(w):
    put("espq_wear", float(round(w, 4)))
    ss["espql_wear_done"] = float(round(w, 4))


def esp_quicklook_mode(col, fluid, eqd, o, corr, p1, p2, esp_eq, ipr_curve):
    """Matching | QuickLook | ESP (PROSPER Tutorial 34): top-down and bottom-up gradients of an ESP well test."""
    hc = eqd["heat_capacities"]
    if not esp_eq:
        col.info("Enter the ESP in the Equipment Data tab first.")
        return
    with col:
        st.markdown("**Well test**")
        c = st.columns(3)
        thp = num(c[0], "Tubing head pressure (psia)", "eql_thp", 289.7, 14.7, 2e4)
        q = num(c[1], "Liquid rate (STB/d)", "eql_q", 7198.0, 1.0, 1e6)
        wc = num(c[2], "Water cut (%)", "eql_wc", 74.0, 0.0, 99.9)
        c = st.columns(3)
        gor = num(c[0], "Produced GOR (scf/STB)", "eql_gor", 800.0, 0.0, 1e6)
        pres = num(c[1], "Static bottom hole pressure (psia)", "eql_pres", float(ipr_curve.pr), 14.7, 2e4,
                   help="The reservoir pressure the IPR is rebuilt at for the bottom-up gradient.")
        wear = num(c[2], "Pump wear factor (fraction)", "eql_wear", float(esp_eq.get("wear", 0.0)), -0.5, 0.9, None,
                   "%.4g", help="Defaults to the Equipment tab's; Transfer wear factor writes a new one back.")
        st.markdown("**Downhole measurements**")
        c = st.columns(3)
        gtxt = c[0].text_input("Gauges: MD ft = pressure psia, separated by ';'", key="eql_gauges",
                               **dflt("eql_gauges", value="6250 = 1971.696"))
        pip_m = num(c[1], "Pump intake pressure (psia, 0 = none)", "eql_pip", 1728.7, 0.0, 2e4)
        pdp_m = num(c[2], "Pump discharge pressure (psia, 0 = none)", "eql_pdp", 2826.7, 0.0, 2e4)
        gauges = []
        for part in [x for x in gtxt.split(";") if x.strip()]:
            try:
                a_, b_ = part.split("=")
                gauges.append((float(a_), float(b_)))
            except ValueError:
                st.error(f"Could not read gauge entry '{part.strip()}'.")
        st.markdown("**Surface electrical data**")
        c = st.columns(3)
        amps = num(c[0], "Current (A, 0 = none)", "eql_amps", 110.0, 0.0, 1e4)
        volts = num(c[1], "Surface voltage (V, 0 = none)", "eql_volts", 1970.0, 0.0, 1e5)
        power = num(c[2], "Power (hp, 0 = none)", "eql_power", 325.0, 0.0, 1e5)
        st.caption(f"Pump from the Equipment tab: {esp_eq.get('stages')} stages at {esp_eq['pump_md']:,.0f} ft MD, "
                   f"{esp_eq['freq']:g} Hz, gas separator {100 * esp_eq.get('sep', 0.0):g} %. Correlation and "
                   "parameters from the left.")
        go = st.button("Calculate QuickLook", type="primary", key="eql_go")
    sig = (thp, q, wc, gor, pres, wear, tuple(gauges), pip_m, pdp_m, amps, volts, power, corr, p1, p2,
           repr(sorted((k, str(v)) for k, v in esp_eq.items())), float(ipr_curve.pr))
    if go:
        try:
            path = vlp_path(eqd, o.flow_type, tuple([esp_eq["pump_md"]] + [m for m, _ in gauges]))
            ch = {"wc": wc / 100.0, "gor": gor}
            try:
                curve = rebuild_ipr({**ch, "pr": pres})
                pres_for = lambda pr_: (lambda qq, c_=rebuild_ipr({**ch, "pr": pr_}): float(wgl.ipr_pwf(c_, qq)))
            except ValueError:
                curve, pres_for = ipr_curve, None
                st.warning("The IPR model cannot be rebuilt at the static pressure - the IPR tab curve is used and "
                           "the reservoir pressure estimate shifts it by the pressure difference.")
            esp_ = {**esp_eq, "wear": wear}
            inp = weql.ESPQuickLookInput(thp, q, wc / 100.0, gor, pres, tuple(gauges), pip_m or None, pdp_m or None,
                                         amps or None, volts or None, power or None)
            with st.spinner("Calculating..."):
                res = weql.esp_quicklook(path, fluid, lambda qq: float(wgl.ipr_pwf(curve, qq)), inp, esp_, corr, p1,
                                         p2, float(val("eq_u", eqd.get("u_geo", 8.0))), hc, pres_for)
            ss["eql_res"] = {"sig": sig, "res": res, "path": path, "gauges": gauges, "pip": pip_m, "pdp": pdp_m,
                             "pump_md": esp_eq["pump_md"]}
        except (ValueError, RuntimeError) as e:
            st.error(f"ESP QuickLook failed: {e}")
    out = ss.get("eql_res")
    if not out:
        return
    if out["sig"] != sig:
        st.info("Inputs have changed since the last calculation - press Calculate QuickLook to update.")
    res = out["res"]
    for n_ in res.notes:
        st.warning(n_)
    cc = st.columns([1.3, 1])
    with cc[0]:
        pr_ = res.profile
        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        for sec_, ls_ in (("Tubing", "-"), ("Casing", "-")):
            d_ = pr_[pr_["Section"] == sec_]
            ax.plot(d_["Top down (psia)"], d_["MD (ft)"], color="#ef6c00", lw=1.8, ls=ls_,
                    label="Fixed top node pressure (top down)" if sec_ == "Tubing" else None)
            ax.plot(d_["Bottom up (psia)"], d_["MD (ft)"], color="#ad1457", lw=1.5, ls="--",
                    label="From static BHP (bottom up)" if sec_ == "Tubing" else None)
        pm = out["pump_md"]
        td_ = res.summary
        ax.plot([td_["Top down: pump intake (psia)"], td_["Top down: pump discharge (psia)"]], [pm, pm], color="#ef6c00", lw=1.2)
        ax.plot([td_["Bottom up: pump intake (psia)"], td_["Bottom up: pump discharge (psia)"]], [pm, pm], color="#ad1457",
                lw=1.0, ls="--")
        meas = [(p_, m_) for m_, p_ in out["gauges"]] + [(p_, pm) for p_ in (out["pip"], out["pdp"]) if p_]
        if meas:
            ax.plot([a for a, _ in meas], [b for _, b in meas], "s", color=wplt.INK, ms=7, label="Downhole data")
        ax.invert_yaxis(); ax.set_xlabel("Pressure (psia)"); ax.set_ylabel("Measured depth (ft)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title("ESP QuickLook", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with cc[1]:
        st.markdown("**QuickLook results**")
        st.dataframe(pd.DataFrame([(k_, _fmt_result(v_) if isinstance(v_, (int, float, np.floating)) else str(v_))
                                   for k_, v_ in res.summary.items()], columns=["Result", "Value"]),
                     hide_index=True, width="stretch", height=560)
        w_ = res.summary.get("Wear factor for the measured intake")
        if w_ is not None and np.isfinite(w_):
            st.button(f"Transfer wear factor {w_:.4f} to the Equipment tab", on_click=_espql_transfer_wear, args=(w_,),
                      help="PROSPER's tutorial takes 1 - measured/calculated dP as a first guess; this is the wear "
                           "factor with which the pump takes the measured discharge back to the measured intake.")
        if ss.get("espql_wear_done") is not None:
            st.success(f"Wear factor {ss['espql_wear_done']:g} written to the ESP in the Equipment tab.")
    if len(res.points):
        st.markdown("**Measured against calculated**")
        st.dataframe(res.points.style.format(precision=1, na_rep="—"), hide_index=True, width="stretch")
    with st.expander("Gradients (top down and bottom up)"):
        st.dataframe(res.profile.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
    st.caption("Fixed top node pressure: tubing from the measured THP down to the pump discharge, intake from the "
               "pump curve, casing to the bottom node. From static BHP: IPR pressure at the test rate (IPR rebuilt at "
               "the static pressure), casing up to the intake, pump gain, tubing up to the wellhead. If the top-down "
               "gradient passes through the gauge and discharge but not the intake, the pump is worn (wear factor); "
               "if the two gradients differ below the pump, the reservoir pressure has changed - use the reservoir "
               "pressure estimate as the new static pressure.")


def _hspql_transfer_wear(w):
    put("hspq_wear", float(round(w, 4)))
    ss["hspql_wear_done"] = float(round(w, 4))


def hsp_quicklook_mode(col, fluid, eqd, o, corr, p1, p2, hsp_eq, ipr_curve):
    """Matching | QuickLook | HSP (PROSPER Tutorial 35): top-down and bottom-up gradients of an HSP well test."""
    hc = eqd["heat_capacities"]
    if not hsp_eq:
        col.info("Enter the hydraulic submersible pump in the Equipment Data tab first.")
        return
    with col:
        st.markdown("**Well test**")
        c = st.columns(3)
        thp = num(c[0], "Tubing head pressure (psia)", "hql_thp", 514.7, 14.7, 2e4)
        q = num(c[1], "Liquid rate (STB/d)", "hql_q", 5770.0, 1.0, 1e6)
        wc = num(c[2], "Water cut (%)", "hql_wc", 24.0, 0.0, 99.9)
        c = st.columns(3)
        gor = num(c[0], "Produced GOR (scf/STB)", "hql_gor", 800.0, 0.0, 1e6)
        pres = num(c[1], "Static bottom hole pressure (psia)", "hql_pres", float(ipr_curve.pr), 14.7, 2e4,
                   help="The reservoir pressure the IPR is rebuilt at for the bottom-up gradient.")
        wear = num(c[2], "Pump wear factor (fraction)", "hql_wear", float(hsp_eq.get("wear", 0.0)), -0.5, 0.9, None,
                   "%.4g", help="Defaults to the Equipment tab's; Transfer wear factor writes a new one back.")
        c = st.columns(3)
        speed = num(c[0], "Pump speed (rpm)", "hql_speed", float(hsp_eq.get("speed", 7500.0)), 100.0, 30000.0)
        qpf = num(c[1], "Power fluid rate (STB/d)", "hql_qpf", 6308.0, 1.0, 1e6)
        ppf = num(c[2], "Power fluid surface back pressure (psia, 0 = none)", "hql_ppf", 2086.7, 0.0, 2e4)
        st.markdown("**Downhole measurements**")
        c = st.columns(3)
        gtxt = c[0].text_input("Gauges: MD ft = pressure psia, separated by ';'", key="hql_gauges",
                               **dflt("hql_gauges", value=""))
        pip_m = num(c[1], "Pump intake pressure (psia, 0 = none)", "hql_pip", 2589.7, 0.0, 2e4)
        pdp_m = num(c[2], "Pump discharge pressure (psia, 0 = none)", "hql_pdp", 3219.7, 0.0, 2e4)
        gauges = []
        for part in [x for x in gtxt.split(";") if x.strip()]:
            try:
                a_, b_ = part.split("=")
                gauges.append((float(a_), float(b_)))
            except ValueError:
                st.error(f"Could not read gauge entry '{part.strip()}'.")
        st.caption(f"Pump from the Equipment tab: {hsp_eq['pump']}, {hsp_eq['stages']} stages at "
                   f"{hsp_eq['pump_md']:,.0f} ft MD; turbine {hsp_eq['turbine']}, {hsp_eq['turbine_stages']} stages. "
                   "Pump speed entered (PROSPER's 'With Pump Speed Entered'). Correlation and parameters from the left.")
        go = st.button("Calculate QuickLook", type="primary", key="hql_go")
    sig = (thp, q, wc, gor, pres, wear, speed, qpf, ppf, tuple(gauges), pip_m, pdp_m, corr, p1, p2,
           repr(sorted((k, str(v)) for k, v in hsp_eq.items())), float(ipr_curve.pr))
    if go:
        try:
            path = vlp_path(eqd, o.flow_type, tuple([hsp_eq["pump_md"]] + [m for m, _ in gauges]))
            ch = {"wc": wc / 100.0, "gor": gor}
            try:
                curve = rebuild_ipr({**ch, "pr": pres})
                pres_for = lambda pr_: (lambda qq, c_=rebuild_ipr({**ch, "pr": pr_}): float(wgl.ipr_pwf(c_, qq)))
            except ValueError:
                curve, pres_for = ipr_curve, None
                st.warning("The IPR model cannot be rebuilt at the static pressure - the IPR tab curve is used and "
                           "the reservoir pressure estimate shifts it by the pressure difference.")
            inst = _hsp_installed(hsp_eq, wear=wear, speed=speed)
            inp = whql.HSPQuickLookInput(thp, q, wc / 100.0, gor, pres, speed, qpf, ppf or None, tuple(gauges),
                                         pip_m or None, pdp_m or None)
            with st.spinner("Calculating..."):
                res = whql.hsp_quicklook(path, fluid, lambda qq: float(wgl.ipr_pwf(curve, qq)), inp, inst, corr, p1,
                                         p2, float(val("eq_u", eqd.get("u_geo", 8.0))), hc, pres_for)
            ss["hql_res"] = {"sig": sig, "res": res, "gauges": gauges, "pip": pip_m, "pdp": pdp_m,
                             "pump_md": hsp_eq["pump_md"]}
        except (ValueError, RuntimeError, KeyError) as e:
            st.error(f"HSP QuickLook failed: {e}")
    out = ss.get("hql_res")
    if not out:
        return
    if out["sig"] != sig:
        st.info("Inputs have changed since the last calculation - press Calculate QuickLook to update.")
    res = out["res"]
    for n_ in res.notes:
        st.warning(n_)
    cc = st.columns([1.3, 1])
    with cc[0]:
        pr_ = res.profile
        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        for j_, sec_ in enumerate(pr_["Section"].unique()):
            d_ = pr_[pr_["Section"] == sec_]
            ax.plot(d_["Top down (psia)"], d_["MD (ft)"], color="#ef6c00", lw=1.8,
                    label="Fixed top node pressure (top down)" if j_ == 0 else None)
            ax.plot(d_["Bottom up (psia)"], d_["MD (ft)"], color="#ad1457", lw=1.5, ls="--",
                    label="From static BHP (bottom up)" if j_ == 0 else None)
        pm = out["pump_md"]
        td_ = res.summary
        ax.plot([td_["Top down: pump intake (psia)"], td_["Top down: pump discharge (psia)"]], [pm, pm],
                color="#ef6c00", lw=1.2)
        ax.plot([td_["Bottom up: pump intake (psia)"], td_["Bottom up: pump discharge (psia)"]], [pm, pm],
                color="#ad1457", lw=1.0, ls="--")
        meas = [(p_, m_) for m_, p_ in out["gauges"]] + [(p_, pm) for p_ in (out["pip"], out["pdp"]) if p_]
        if meas:
            ax.plot([a for a, _ in meas], [b for _, b in meas], "s", color=wplt.INK, ms=7, label="Downhole data")
        ax.invert_yaxis(); ax.set_xlabel("Pressure (psia)"); ax.set_ylabel("Measured depth (ft)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title("HSP QuickLook", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with cc[1]:
        st.markdown("**QuickLook results**")
        st.dataframe(pd.DataFrame([(k_, _fmt_result(v_) if isinstance(v_, (int, float, np.floating)) else str(v_))
                                   for k_, v_ in res.summary.items()], columns=["Result", "Value"]),
                     hide_index=True, width="stretch", height=560)
        w_ = res.summary.get("Wear factor for the measured intake")
        if w_ is not None and np.isfinite(w_):
            st.button(f"Transfer wear factor {w_:.4f} to the Equipment tab", on_click=_hspql_transfer_wear,
                      args=(w_,), key="hql_transfer",
                      help="PROSPER's tutorial takes 1 - measured/calculated dP as a first guess (0.09 in Tutorial "
                           "35); this is the wear factor with which the pump takes the measured intake to the "
                           "measured discharge.")
        if ss.get("hspql_wear_done") is not None:
            st.success(f"Wear factor {ss['hspql_wear_done']:g} written to the HSP in the Equipment tab.")
    if len(res.points):
        st.markdown("**Measured against calculated**")
        st.dataframe(res.points.style.format(precision=1, na_rep="—"), hide_index=True, width="stretch")
    with st.expander("Gradients (top down and bottom up)"):
        st.dataframe(res.profile.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
    st.caption("Fixed top node pressure: tubing (production + power fluid) from the measured THP down to the pump "
               "discharge, intake from the pump curve, casing to the bottom node. From static BHP: IPR pressure at "
               "the test rate (IPR rebuilt at the static pressure), casing up to the intake, pump gain, tubing up to "
               "the wellhead. If the top-down gradient passes through the discharge but not the intake, the pump is "
               "worn (wear factor); if the two differ below the pump, the reservoir pressure has changed - use the "
               "reservoir pressure estimate as the new static pressure.")


def quicklook_mode(col, fluid, eqd, o, corr, p1, p2, gl, ipr_curve):
    """Matching | QuickLook for a gas-lifted well (PROSPER User Guide 2.11.5; Tutorial 33a)."""
    G = 14.696
    hc = eqd["heat_capacities"]
    valves = tuple(sorted(gl.valves, key=lambda v: v.md))
    with col:
        st.markdown("**Surface measurements (well test)**")
        c = st.columns(3)
        thp = num(c[0], "Tubing head pressure (psia)", "ql_thp", 404.7, 14.7, 2e4)
        tht = num(c[1], "Tubing head temperature (°F)", "ql_tht", 146.0, 0.0, 400.0)
        qliq = num(c[2], "Liquid rate (STB/d)", "ql_q", 7488.0, 1.0, 1e6)
        c = st.columns(4)
        wc = num(c[0], "Water cut (%)", "ql_wc", 35.0, 0.0, 99.0)
        qgt = num(c[1], "Total gas rate (MMscf/d)", "ql_qgt", 5.3935, 0.0, 500.0, None, "%.5g",
                  help="Produced plus injected gas.")
        qgi = num(c[2], "Gas injection rate (MMscf/d)", "ql_qgi", 1.5, 0.0, 500.0, None, "%.5g")
        chp = num(c[3], "Casing head pressure (psia)", "ql_chp", 1409.7, 14.7, 2e4)
        st.markdown("**Injection point**")
        c = st.columns(4)
        labels = [f"Valve {i + 1} - {v.kind.lower()} at {v.md:,.1f} ft ({v.port_64:g}/64)" for i, v in enumerate(valves)]
        pick = c[0].selectbox("Copy from valve", ["—"] + labels, key="ql_pick",
                              help="Fills the injection depth and port size from a valve of the Gas lift data.")
        if pick != "—" and ss.get("_ql_last_pick") != pick:
            v_ = valves[labels.index(pick)]
            put("ql_md", float(v_.md))
            put("ql_port", float(v_.port_64))
            ss["_ql_last_pick"] = pick
            st.rerun()
        ss["_ql_last_pick"] = pick
        md0 = float(valves[-1].md) if valves else float(eqd["downhole"][1]["MD bottom (ft)"].max()) - 100
        inj = num(c[1], "Injection depth (ft MD)", "ql_md", md0, 10.0, 60000.0,
                  help="QuickLook uses this depth even if the casing pressure cannot reach it - check the plot.")
        port = num(c[2], "Orifice / port diameter (64ths in)", "ql_port",
                   float(valves[-1].port_64) if valves else 16.0, 1.0, 128.0, None, "%.4g")
        derate = num(c[3], "Thornhill-Craver de-rating (%)", "ql_derate", 0.0, 0.0, 90.0)
        st.markdown("**Static and flowing measurements** (optional)")
        c = st.columns(3)
        use_static = c[0].checkbox("Static pressure at the bottom node", key="ql_use_static",
                                   help="Taken as the reservoir pressure of the test: the IPR is rebuilt at it.")
        static = num(c[1], "Static pressure (psia)", "ql_static", float(ipr_curve.pr), 14.7, 2e4,
                     disabled=not use_static)
        fit_u = flag(c[2].checkbox, "Use the U that matches the measured THT", "ql_fitu", True)
        gauge_txt = st.text_input("Gauges: MD ft = pressure psia, separated by ';' (e.g. 6250 = 1454.7)", key="ql_gauges",
                                  **dflt("ql_gauges", value=""))
        gauges = []
        for part in [x for x in gauge_txt.split(";") if x.strip()]:
            try:
                a, b = part.split("=")
                gauges.append((float(a), float(b)))
            except ValueError:
                st.error(f"Could not read gauge entry '{part.strip()}'.")
        multi = st.toggle("Multipoint QuickLook", key="ql_multi", disabled=not valves,
                          help="Top down, each valve passes the gas the valve model gives at the measured pressures; "
                               "the lift gas left below it is reduced.")
        vmodel = sel(st, "Valve model (multipoint)", ["Throttling", "Thornhill-Craver"], "ql_vmodel",
                     disabled=not multi,
                     help="Throttling: this tool's rule fitted to PROSPER's multipoint VLP. Thornhill-Craver: the full "
                          "port once the casing is above the closing pressure (PROSPER's model type of that name). "
                          "PROSPER's Acuna and Nieberding models are not available (unpublished).")
        if multi and vmodel == "Thornhill-Craver":
            valves = tuple(dataclasses.replace(v_, model="Thornhill-Craver") if v_.kind != "Orifice" else v_
                           for v_ in valves)
        st.caption("Uses the gas lift data of the Equipment tab (lift gas, annulus friction, valves) and the "
                   "correlation and parameters on the left; the other inputs on the left are not used.")
        go = st.button("Calculate QuickLook", type="primary")
    sig = (thp, tht, qliq, wc, qgt, qgi, chp, inj, port, derate, use_static, static, fit_u, tuple(gauges), multi,
           corr, p1, p2, repr(gl), float(ipr_curve.pr), repr(valves))
    if go:
        try:
            if qgi >= qgt:
                raise ValueError("The gas injection rate must be below the total gas rate.")
            mds = [v.md for v in valves] + [inj] + [m for m, _ in gauges]
            path, gl_ = gl_lift_for(eqd, o, gl, tuple(mds))
            curve = ipr_curve
            ch = {"wc": wc / 100.0}
            if use_static:
                ch["pr"] = static
            try:
                curve = rebuild_ipr(ch)
            except ValueError:
                if use_static:
                    st.warning("The IPR model cannot be rebuilt at the static pressure - the IPR tab curve is used.")
            inp = wql.QuickLookInput(thp, tht, qliq, wc / 100.0, qgt, qgi, chp, inj, port, derate, tuple(gauges),
                                     static if use_static else float(curve.pr), valves, fit_u)
            with st.spinner("Calculating..."):
                res = wql.quicklook(path, fluid, lambda q: float(wgl.ipr_pwf(curve, q)), gl_, inp, corr, p1, p2,
                                    float(val("eq_u", eqd.get("u_geo", 8.0))), hc, multipoint=multi)
            ss["ql_res"] = {"sig": sig, "res": res, "path": path, "inj": inj, "gauges": gauges}
        except (ValueError, RuntimeError) as e:
            st.error(f"QuickLook failed: {e}")
    out = ss.get("ql_res")
    if not out:
        return
    if out["sig"] != sig:
        st.info("Inputs have changed since the last calculation - press Calculate QuickLook to update.")
    res = out["res"]
    for n_ in res.notes:
        st.warning(n_)
    cc = st.columns([1.3, 1])
    with cc[0]:
        st.pyplot(ql_plot(res, out["path"], res.valves, out["gauges"], out["inj"]), clear_figure=True)
    with cc[1]:
        st.markdown("**QuickLook results**")
        st.dataframe(pd.DataFrame([(k_, _fmt_result(v_) if isinstance(v_, (int, float, np.floating)) else str(v_))
                                   for k_, v_ in res.summary.items()], columns=["Result", "Value"]),
                     hide_index=True, width="stretch", height=560)
    if res.valves is not None:
        st.markdown("**Valves** (pressures from the top-down gradients; OPEN: casing above the closing pressure)")
        st.dataframe(res.valves.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
    with st.expander("Gradients (top down and bottom up)"):
        st.dataframe(res.profile.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
    st.caption("Top down: tubing from the measured THP with the total GOR to the injection point and the produced "
               "GOR below; casing from the measured CHP. Bottom up: from the IPR pressure at the test rate, tubing up "
               "to the wellhead; casing from the measured tubing pressure at the valve plus the port dP "
               "(Thornhill-Craver) back to surface. Overlying gradients mean the model represents the test.")


def _default_list(vals):
    return ", ".join(f"{v:.4g}" for v in vals)


def vfp_mode(col, kind, fluid, eqd, o, base, corr, p1, p2, surface, esp=None, gl=None, jet=None, pcp=None, hsp=None,
             srp=None):
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
        thp = num_list(st, "Top node pressures (psia)", f"vfp_thp_{kind}",
                       _default_list(np.round(whp0 * np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0]), 0)),
                       "comma separated" + (" - at the manifold" if surface is not None else ""))
        cbm_ = getattr(o, "fluid_type", "") == "cbm"
        levels = []
        if cbm_:
            wfr = [100.0]
            g0 = float(base["gor"])
            gfr = num_list(st, "CBM GLRs (scf/STB)", "vfp_gfr_cbm",
                           _default_list(sorted({0.1, round(g0, 4), round(g0 * 10, 4), round(g0 * 100, 4)})))
            if esp:
                l0 = float(esp.get("level", 2000.0))
                levels = num_list(st, "Annulus liquid levels (ft)", "vfp_levels",
                                  _default_list(sorted({max(l0 - 500, 0.0), l0, l0 + 400})),
                                  "comma separated; they set the casing head pressure only (PROSPER's fourth "
                                  "variable)")
                levels = sorted({v for v in levels if v >= 0})
        elif kind == "oil":
            wfr = parse_values(st.text_input("Water cuts (%)", key=f"vfp_wfr_{kind}", **dflt(f"vfp_wfr_{kind}", value="0, 20, 40, 60, 80, 95")))
            g0 = float(base["gor"])
            gfr = num_list(st, "Total GORs (scf/STB)", f"vfp_gfr_{kind}",
                           _default_list(np.round(g0 * np.array([0.75, 1.0, 1.5, 2.5, 4.0]), 0)))
        else:
            w0 = float(base["wgr"])
            wfr = num_list(st, "Water gas ratios (STB/MMscf)", f"vfp_wfr_{kind}",
                           _default_list(sorted({0.0, w0, max(w0 * 3, 10.0), max(w0 * 10, 50.0)})))
            sg = float(base.get("sep_gor", 0.0))
            c0 = 1e6 / sg if sg > 0 else 0.0
            gfr = num_list(st, "Condensate gas ratios (STB/MMscf)", f"vfp_gfr_{kind}",
                           _default_list(sorted({0.0, round(c0, 2), round(max(c0 * 2, 20.0), 2)})))
        alq = []
        if esp:
            f0 = float(esp["freq"])
            alq = num_list(st, "ESP frequencies (Hz) - the ALQ axis", "vfp_alq",
                           _default_list(sorted({max(f0 - 10, 30.0), f0 - 5, f0, f0 + 5, min(f0 + 10, 90.0)})),
                           "comma separated; one value = a single frequency (no ALQ axis)")
            alq = sorted({v for v in alq if 10.0 <= v <= 120.0})
        if gl is not None:
            if gl.multipoint:
                p0 = float(gl.p_casing)
                alq = num_list(st, "Casing head pressures (psia) - the ALQ axis", "vfp_alq_gl",
                               _default_list([p0 - 200, p0 - 100, p0]),
                               "comma separated; multipoint: the lift gas rate is a result of the casing pressure")
            else:
                g0 = float(gl.qgi)
                alq = num_list(st, "Lift gas rates (MMscf/d) - the ALQ axis", "vfp_alq_gl",
                               _default_list(sorted({0.0, g0 * 0.5, g0, g0 * 2})),
                               "comma separated; exported as GRAT (Mscf/d)")
            alq = sorted({v for v in alq if v >= 0})
        jet_alq = "p_surf"
        if jet:
            ja = sel(st, "Jet pump ALQ axis", ["Power fluid injection pressure", "Power fluid injection rate"],
                     "vfp_jet_axis")
            jet_alq = "p_surf" if ja.endswith("pressure") else "q_pf"
            v0 = float(jet[jet_alq])
            alq = num_list(st, ("Injection pressures (psia)" if jet_alq == "p_surf" else
                                "Power fluid rates (STB/d)") + " - the ALQ axis", f"vfp_alq_jet_{jet_alq}",
                           _default_list(sorted({round(v0 * 0.75), round(v0), round(v0 * 1.25)})),
                           "comma separated; one value = no ALQ axis")
            alq = sorted({v for v in alq if v > 0})
        if pcp:
            n0 = float(pcp["speed"])
            alq = num_list(st, "Pump speeds (rpm) - the ALQ axis", "vfp_alq_pcp",
                           _default_list(sorted({round(n0 * 0.75, 1), round(n0, 3), round(n0 * 1.25, 1)})),
                           "comma separated; one value = no ALQ axis")
            alq = sorted({v for v in alq if v > 0})
        if srp:
            n0 = float(srp["n_spm"])
            alq = num_list(st, "Pumping speeds (strokes/min) - the ALQ axis", "vfp_alq_srp",
                           _default_list(sorted({round(n0 * 0.75, 2), round(n0, 4), round(n0 * 1.25, 2)})),
                           "comma separated; one value = no ALQ axis")
            alq = sorted({v for v in alq if v > 0})
        if hsp:
            n0 = float(hsp["speed"])
            alq = num_list(st, "Pump speeds (rpm) - the ALQ axis", "vfp_alq_hsp",
                           _default_list(sorted({round(n0 * 0.9), round(n0), round(n0 * 1.1)})),
                           "comma separated; one value = no ALQ axis")
            alq = sorted({v for v in alq if v > 0})
        c2 = st.columns([1, 2])
        tno = int(num(c2[0], "Table number", "vfp_tno", 1.0, 1.0, 9999.0, 1.0, "%.0f"))
        wfr = [v for v in wfr if v >= 0]
        gfr = [v for v in gfr if v >= 0]
        thp = [v for v in thp if v > 0]
        n_alq = max(len(alq), 1)
        npts = len(flo) * len(set(thp)) * len(set(wfr)) * len(set(gfr)) * n_alq
        c2[1].caption(f"{npts:,} points ({len(flo)} rates x {len(set(thp))} pressures x {len(set(wfr))} x "
                      f"{len(set(gfr))}" + (f" x {n_alq} " + ("ALQ values" if (gl is not None or jet or pcp or hsp or srp) else "frequencies")
                                            if len(alq) > 1 else "") + ").")
        if kind == "oil" and not cbm_ and any(v >= 100 for v in wfr):
            st.error("Water cut must be below 100 %.")
            wfr = [v for v in wfr if v < 100]
        if esp:
            st.caption(f"Tables include the ESP from the Equipment tab ({esp['pump'].name}, {esp['stages']} stages, "
                       f"{esp['pump_md']:,.0f} ft MD). With more than one frequency, frequency is the ALQ axis of the "
                       "VFPPROD table (set it in the simulator through the well's ALQ). Points where the pump would "
                       "draw the intake below atmospheric (rate too low for the pump) are left blank.")
        if jet:
            st.caption(f"Tables include the jet pump from the Equipment tab ({jet['name']}, {jet['pump_md']:,.0f} ft "
                       "MD): tubing to the pump discharge with production + power fluid, the intake from the pump, "
                       "casing to the bottom node. With more than one value, the injection pressure or rate is the "
                       "ALQ axis of the VFPPROD table. Points the pump cannot deliver are left blank.")
        if pcp:
            st.caption(f"Tables include the PCP from the Equipment tab ({pcp['pump']}, {pcp['pump_md']:,.0f} ft MD): "
                       "tubing to the pump discharge, the intake from the head at which the pump passes the rate at its "
                       "speed, casing to the bottom node. With more than one speed, the speed (rpm) is the ALQ axis of "
                       "the VFPPROD table. Points the pump cannot pass at that speed are left blank.")
        if srp:
            st.caption(f"Tables include the rod pump from the Equipment tab ({srp['plunger']:g} in pump at "
                       f"{srp['pump_md']:,.0f} ft MD, {srp['stroke']:g} in stroke): as PROSPER's rod pump lift "
                       "curves, the rate the pump delivers against the bottom-hole pressure (displacement x volumetric "
                       "efficiency x liquid fraction at the intake, the downhole stroke from the fluid load), inverted "
                       "to bottom-hole pressure against rate for the VFP table. With more than one speed, the pumping "
                       "speed is the ALQ axis. Rates beyond what the pump delivers are left blank.")
        if hsp:
            st.caption(f"Tables include the HSP from the Equipment tab ({hsp['pump']}, {int(hsp['stages'])} stages; "
                       f"{hsp['turbine']}, {int(hsp['turbine_stages'])} stages; {hsp['pump_md']:,.0f} ft MD): at "
                       "every point the power fluid is the rate whose turbine gives the pump power (PROSPER's HSP "
                       "traverse). With more than one speed, the pump speed (rpm) is the ALQ axis of the VFPPROD "
                       "table. Points outside the pump's speed range, or that the pump or turbine cannot deliver, are "
                       "left blank. About 1-2 s per point.")
        go = st.button("Generate lift curves", type="primary", disabled=not (flo and thp and wfr and gfr))
    sig = (kind, corr, p1, p2, tuple(flo), tuple(thp), tuple(wfr), tuple(gfr), repr(surface), o.flow_type, base["u"],
           tno, None if not esp else (esp["pump_md"], esp["freq"], esp["stages"], esp["pump"].name, esp["sep"],
                                      esp["wear"], esp["motor"].name, tuple(alq)), repr(gl), tuple(alq),
           None if not jet else (tuple(sorted((k_, repr(v_)) for k_, v_ in jet.items())), jet_alq),
           None if not pcp else tuple(sorted((k_, repr(v_)) for k_, v_ in pcp.items())),
           None if not hsp else tuple(sorted((k_, repr(v_)) for k_, v_ in hsp.items())),
           None if not srp else tuple(sorted((k_, repr(v_)) for k_, v_ in srp.items())), tuple(levels))
    if go and cbm_ and esp:
        try:
            path = vlp_path(eqd, o.flow_type, (esp["pump_md"],))
            with st.spinner(f"Calculating {npts:,} traverses..."):
                res_ = wcbm.lift_curves(path, fluid, esp, corr, p1, p2, float(base["u"]), hc, flo,
                                        levels or [float(esp.get("level", 2000.0))], gfr, thp,
                                        alq or [float(esp["freq"])])
            b_ = res_["bhp"][:, :, :, 0, :]                         # (freq, thp, glr, rate)
            a_ = res_["axes"]
            if len(a_["freqs"]) > 1:
                axes = (a_["thps"], np.array([100.0]), a_["glrs"], a_["freqs"], a_["rates"])
                bhp = np.transpose(b_, (1, 2, 0, 3))[:, None]
            else:
                axes = (a_["thps"], np.array([100.0]), a_["glrs"], a_["rates"])
                bhp = b_[0][:, None]
            ss["vfp_res"] = {"sig": sig, "axes": axes, "bhp": bhp, "wh": None, "alq_kind": "esp",
                             "datum": float(path.tvd[-1]), "md": float(path.md[-1]), "tno": tno,
                             "label": f"{VLP_LABELS[corr]} P1 {p1:.4f} P2 {p2:.4f}", "cbm": res_}
        except (ValueError, KeyError, TypeError) as e:
            st.error(f"Lift curves failed: {e}")
        go = False
    if go:
        try:
            path = vlp_path(eqd, o.flow_type, (esp["pump_md"],) if esp else ((jet["pump_md"],) if jet else (
                (pcp["pump_md"],) if pcp else ((hsp["pump_md"],) if hsp else (
                    (srp["pump_md"],) if srp else ())))))
            hsp_ = _hsp_installed(hsp) if hsp else None
            srp_ = None
            ipr_s = ss.get("ipr_curve")
            corr_v, p1_v, p2_v = corr, p1, p2
            if srp:
                pl_ = min(wsrp.PLUNGERS, key=lambda x: abs(x - srp["plunger"]))
                srp_ = wsrp.SRPSystemInput(rod_no=int(srp["rod_no"]), plunger=pl_, stroke=srp["stroke"],
                                           n_spm=alq[0] if len(alq) == 1 else srp["n_spm"], ev=srp["ev"] / 100.0,
                                           pump_md=srp["pump_md"], anchor_method=srp["anchor_method"],
                                           anchor_eff=srp["anchor_eff"], anchor_type=srp["anchor_type"],
                                           anchor_area=srp["anchor_area"])
                if srp["no_slip"]:
                    corr_v, p1_v, p2_v = "no_slip", 1.0, 1.0
            gl_ = None
            jet_ = None
            pcp_ = None
            if pcp:
                pcp_ = _pcp_installed(pcp, alq[0] if len(alq) == 1 else None)
            if jet:
                jet_ = wjet.JetPumpInstalled(jet["pump_md"], jet["an"], jet["at"], jet["q_pf"], jet["p_surf"],
                                             jet["losses"], wjet.PowerFluid(jet["sal"]), jet["casing_id"],
                                             jet["tubing_od"], jet["name"])
                if len(alq) == 1:
                    jet_ = dataclasses.replace(jet_, **{jet_alq: alq[0]})
            if gl is not None:
                path, gl_ = gl_lift_for(eqd, o, gl)
                if len(alq) == 1:
                    gl_ = dataclasses.replace(gl_, **({"p_casing": alq[0]} if gl_.multipoint else
                                                      {"qgi": alq[0], "rate_mode": wgls.RATE_MODES[0]}))
            with st.spinner(f"Calculating {npts:,} traverses..."):
                esp_ = {**esp, "freq": alq[0]} if esp and len(alq) == 1 else esp
                axes, bhp, wh = wlc.vfp_table(path, fluid, kind, corr_v, p1_v, p2_v, float(base["u"]), flo, thp, wfr,
                                              gfr, hc, surface, esp_, alq if (esp or gl or jet or pcp or hsp or srp)
                                              and len(alq) > 1 else None, lift=gl_, jet=jet_, jet_alq=jet_alq, pcp=pcp_,
                                              srp=srp_, srp_p_max=(float(ipr_s.pr) * 1.1 if srp and ipr_s is not None
                                                                   else None),
                                              hsp=None if not hsp else (dataclasses.replace(hsp_, speed=alq[0])
                                                                        if len(alq) == 1 else hsp_))
            ss["vfp_res"] = {"sig": sig, "axes": axes, "bhp": bhp, "wh": wh if surface is not None else None,
                             "alq_kind": ("srp_speed" if srp else "hsp_speed" if hsp else "pcp_speed" if pcp else f"jet_{jet_alq}" if jet else
                                          "esp" if gl is None else ("multipoint" if gl.multipoint else "gas_lift")),
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
    (thp_a, wfr_a, gfr_a, alq_a, flo_a), bhp5 = wlc._split(axes, bhp)
    st.markdown(f"**Lift curves** - bottom node at {vr['datum']:,.1f} ft TVD ({vr['md']:,.0f} ft MD), "
                f"{vr['label']}")
    alq_kind = vr.get("alq_kind", "esp")
    alq_lab = {"esp": "ESP frequency (Hz)", "gas_lift": "Lift gas rate (MMscf/d)",
               "multipoint": "Casing head pressure (psia)", "jet_p_surf": "Jet pump injection pressure (psia)",
               "jet_q_pf": "Jet pump power fluid rate (STB/d)", "pcp_speed": "PCP speed (rpm)",
               "hsp_speed": "HSP pump speed (rpm)", "srp_speed": "Rod pump speed (strokes/min)"}[alq_kind]
    for w_ in wlc.vfp_checks(axes, bhp, esp=bool(esp) or bool(jet) or bool(pcp) or bool(hsp) or bool(srp)):
        st.warning(w_)
    c = st.columns(3 if alq_a is not None else 2)
    jw = c[0].selectbox(f"{ax_['wfr'][0]} ({ax_['wfr'][1]})", range(len(wfr_a)), format_func=lambda i: f"{wfr_a[i]:g}",
                        key=f"vfp_pw_{len(wfr_a)}")
    kg = c[1].selectbox(f"{ax_['gfr'][0]} ({ax_['gfr'][1]})", range(len(gfr_a)), format_func=lambda i: f"{gfr_a[i]:g}",
                        key=f"vfp_pg_{len(gfr_a)}")
    la = 0
    if alq_a is not None:
        la = c[2].selectbox(alq_lab, range(len(alq_a)), format_func=lambda i: f"{alq_a[i]:g}",
                            key=f"vfp_pa_{len(alq_a)}")
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate(thp_a):
        ax.plot(flo_a, bhp5[i, jw, kg, la], "-o", ms=3, lw=1.5, color=cmap(i / max(len(thp_a) - 1, 1)),
                label=f"THP {t:g} psia")
    ax.set_xlabel(f"{ax_['flo'][0]} ({ax_['flo'][1]})")
    ax.set_ylabel("Bottom node pressure (psia)")
    ax.set_title(f"Lift curves at {ax_['wfr'][0].lower()} {wfr_a[jw]:g} {ax_['wfr'][1]}, "
                 f"{ax_['gfr'][0].lower()} {gfr_a[kg]:g} {ax_['gfr'][1]}"
                 + (f", {alq_lab.split(' (')[0].lower()} {alq_a[la]:g}" if alq_a is not None else ""), fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    pc = st.columns([1.4, 1])
    pc[0].pyplot(fig, clear_figure=True)
    ecl = wlc.eclipse_vfpprod(kind, axes, bhp, vr["tno"], vr["datum"], ss.get("proj_name", ""), alq_kind=alq_kind)
    csv = wlc.vfp_csv(kind, axes, bhp, vr["wh"], alq_label=alq_lab)
    with pc[1]:
        st.markdown("**Export**")
        st.download_button("Download Eclipse VFPPROD (.inc)", ecl.encode(), f"vfpprod_{vr['tno']}.inc",
                           width="stretch")
        st.download_button("Download table (CSV)", csv.to_csv(index=False).encode(), f"lift_curves_{vr['tno']}.csv",
                           width="stretch")
        st.caption("Eclipse FIELD units: " + ("FLO = liquid rate (STB/d), WFR = water cut (fraction), GFR = GOR "
                   "(Mscf/STB)" if kind == "oil" else "FLO = gas rate (Mscf/d), WFR = WGR (STB/Mscf), GFR = OGR "
                   "(STB/Mscf)") + "; THP and BHP in psia; datum = bottom node TVD; " +
                   ({"esp": "ALQ = ESP frequency (Hz).", "gas_lift": "ALQ = GRAT, lift gas rate (Mscf/d).",
                     "multipoint": "ALQ = casing head pressure (psia); set the ALQ type in the simulator.",
      "jet_p_surf": "ALQ = jet pump power fluid injection pressure (psia); set the ALQ type in the simulator.",
      "jet_q_pf": "ALQ = jet pump power fluid rate (STB/d); set the ALQ type in the simulator.",
      "pcp_speed": "ALQ = PCP speed (rpm); set the ALQ type in the simulator.",
      "hsp_speed": "ALQ = HSP pump speed (rpm); set the ALQ type in the simulator.",
      "srp_speed": "ALQ = rod pump speed (strokes/min); set the ALQ type in the simulator."}[alq_kind]
                    if alq_a is not None else "no artificial lift ALQ axis (ALQ 0).") +
                   (" THP is the manifold pressure (surface equipment included)." if vr["wh"] is not None else ""))
        if vr.get("cbm") is not None:
            tpd = wcbm.write_tpd(vr["cbm"], float(val("pvt_gg", 0.6)), vr["datum"], ss.get("proj_name", ""))
            st.download_button("Download PROSPER lift curves for GAP / MBAL (.tpd)", tpd.encode(),
                               f"lift_curves_{vr['tno']}.tpd", width="stretch")
            st.caption("The .tpd file carries PROSPER's four CBM variables (liquid level, CBM GLR, boundary "
                       "pressure, frequency) with the bottom-hole pressure, the pump results and the casing head "
                       "pressure; pressures in psig.")
        with st.expander("Preview VFPPROD"):
            st.code("\n".join(ecl.splitlines()[:40]) + ("\n..." if len(ecl.splitlines()) > 40 else ""),
                    language=None)

# ================================================================ sucker rod pump design
def srp_cards_plot(res):
    cd = res.card
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(np.r_[cd["Surface position (in)"], cd["Surface position (in)"].iloc[0]],
            np.r_[cd["Surface load (lbf)"], cd["Surface load (lbf)"].iloc[0]], color=wplt.SERIES[0], lw=1.6,
            label="Surface (polished rod) card")
    ax.plot(np.r_[cd["Pump position (in)"], cd["Pump position (in)"].iloc[0]],
            np.r_[cd["Pump load (lbf)"], cd["Pump load (lbf)"].iloc[0]], color=wplt.SERIES[1], lw=1.6,
            label="Downhole (pump) card")
    for y, lab in ((res.wrf, "Wrf"), (res.wrf + res.fo, "Wrf + Fo")):
        ax.axhline(y, color=wplt.INK2, ls=":", lw=0.9)
        ax.text(0.5, y, f" {lab}", fontsize=7, va="bottom", color=wplt.INK2)
    ax.set_xlabel("Position (in)")
    ax.set_ylabel("Load (lbf)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=7.5)
    ax.set_title(f"{res.rods.name}, {res.n_spm:.4g} strokes/min, {res.inp.stroke:.4g} in stroke", fontsize=9.5)
    fig.tight_layout()
    return fig


def srp_torque_plot(res):
    cd = res.card.sort_values("Crank angle (deg)")
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(cd["Crank angle (deg)"], cd["Torque (lb-in)"] / 1000, color=wplt.SERIES[0], lw=1.6, label="Net torque")
    ax.axhline(res.kin.unit.gear / 1000, color="#d62728", ls="--", lw=1.0, label="Gear reducer rating")
    ax.axhline(0, color=wplt.INK2, lw=0.6)
    ax.set_xlim(0, 360)
    ax.set_xlabel("Crank angle from 12 o'clock (deg)")
    ax.set_ylabel("Torque (1000 lb-in)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def srp_card_reference():
    """Dynamometer card troubleshooting reference (PROSPER Tutorial 36)."""
    with st.expander("Dynamometer card troubleshooting - typical pump card shapes (Tutorial 36)"):
        st.caption("Idealised downhole (pump) cards: load against plunger position. A - bottom of the stroke, "
                   "B - fluid load picked up, C - top of the stroke, D - fluid load released. Fo is the differential "
                   "(fluid) load on the plunger, MPT the maximum plunger travel, EPT the effective plunger travel "
                   "(the part that moves fluid). Unanchored tubing stretches as the fluid load moves between "
                   "tubing and rods, which slopes the sides of the card. Real cards often combine several effects: "
                   "use them with what is known about the well.")
        for i, c in enumerate(wsrpc.reference_cards()):
            if i:
                st.divider()
            cc = st.columns([1, 1.15, 1], vertical_alignment="center")
            with cc[0]:
                st.pyplot(wplt.plot_card_reference(c.anchored, title="Anchored tubing"), clear_figure=True)
            with cc[1]:
                st.markdown(f"**{c.title}**")
                st.write(c.description)
                st.caption("Look for: " + c.look_for)
            with cc[2]:
                st.pyplot(wplt.plot_card_reference(c.unanchored, title="Unanchored tubing"), clear_figure=True)


def srp_tab(o, eqd, ipr_curve):
    """Design | Sucker Rod Pump - after PROSPER (Tutorial 28)."""
    st.caption("Sucker rod pump design after PROSPER Design | Sucker Rod Pump: the rod string (API RP 11L Table 4.1) "
               "and the pumping unit (Lufkin conventional geometry) drive a damped wave equation for the rods with "
               "an ideal full pump at the bottom (anchored tubing). Surface and pump dynamometer cards, loads, "
               "stresses (modified Goodman), counterbalance, torque and powers follow. The damping factor is "
               "estimated as PROSPER does (rod friction power one eighth of the rod weight times the mean rod speed, Gibbs' damping) "
               "or entered.")
    srp_card_reference()
    top_, secs_, pts_ = eqd["downhole"]
    path = vlp_path(eqd, "tubing", ())
    tvd_at = lambda md: float(np.interp(md, path.md, path.tvd))
    bottom = float(secs_["MD bottom (ft)"].max())
    pinp, _ = current_inputs()
    g = 14.696
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        st.markdown("**Input data**")
        units = list(wsrp.UNITS)
        unit = sel(st, "Pumping unit", units, "srp_unit", "LUFKIN C-320D-305-100" if "LUFKIN C-320D-305-100" in units
                   else units[0])
        c = st.columns(2)
        plunger = sel(c[0], "Pump (plunger) diameter (in)", list(wsrp.PLUNGERS), "srp_plunger", 2.0,
                      format_func=lambda x: f"{x:.2f}")
        rods = wsrp.rod_numbers(plunger)
        rod_no = sel(c[1], "Rod string (API RP 11L)", rods, "srp_rod", 99 if 99 in rods else rods[-1],
                     format_func=lambda n: wsrp.rod_string(n, plunger).name)
        c = st.columns(2)
        grade = sel(c[0], "Rod grade", list(wsrp.TENSILE), "srp_grade", "D")
        service = sel(c[1], "Service", list(wsrp.SERVICE), "srp_service", "Non-corrosive")
        mode = sel(st, "Calculation", list(wsrp.CALC_MODES), "srp_mode", wsrp.CALC_MODES[0])
        c = st.columns(2)
        q_t = num(c[0], "Target production rate (STB/d)", "srp_q", 400.0, 1.0, 1e5,
                  help="The rate the stroke rate is estimated for, and (both modes, as PROSPER) the rate at which the "
                       "intake pressure is taken from the IPR.")
        if mode == wsrp.CALC_MODES[0]:
            n_spm = 20.0
        else:
            n_spm = num(c[1], "Stroke rate (strokes/min)", "srp_n", 20.0, 1.0, 40.0, None, "%.5g")
        c = st.columns(2)
        stroke = num(c[0], "Surface stroke length (in)", "srp_s", 48.0, 10.0, 400.0, None, "%.5g")
        c = st.columns(2)
        pump_md = num(c[0], "Pump depth (ft MD)", "srp_md", float(round(bottom)), 100.0, bottom)
        whp = num(c[1], "Wellhead pressure (psia)", "srp_whp", float(val("sys_whp", 114.7)), 14.7, 5000.0)
        c = st.columns(2)
        wc = num(c[0], "Water cut (%)", "srp_wc", float(val("sys_wc", 0.0)), 0.0, 100.0)
        ev = num(c[1], "Pump volumetric efficiency (%)", "srp_ev", 80.0, 1.0, 100.0)
        pip_m = sel(st, "Pump intake pressure", list(wsrp.INTAKE_METHODS[:2]), "srp_pipm", wsrp.INTAKE_METHODS[1])
        c = st.columns(2)
        pip = num(c[0], "Pump intake pressure (psia)", "srp_pip", 666.1, 14.7, 2e4,
                  disabled=pip_m != wsrp.INTAKE_METHODS[0])
        nu_est = flag(c[1].checkbox, "Estimate damping factor", "srp_nu_est", True)
        nu = num(c[1], "Damping factor", "srp_nu", 0.18, 0.01, 2.0, None, "%.4g", disabled=nu_est,
                 help="Dimensionless rod damping, c = nu pi a / (2 L). Typical 0.05 - 0.3.")
        if nu_est:
            nu = None
        c = st.columns(2)
        go = c[0].button("Calculate", type="primary", key="srp_go")
        go_s = c[1].button("Rod sensitivity", key="srp_go_s", help="Repeat the design for every rod string "
                           "of the plunger size (takes a while for deep wells).")
    inp = wsrp.SRPInput(unit=unit, rod_no=int(rod_no), plunger=float(plunger), grade=grade, service=service,
                        mode=mode, q_target=q_t, n_spm=n_spm, stroke=stroke, pump_md=pump_md,
                        pump_tvd=tvd_at(pump_md), pip=pip, whp=whp, wc=wc / 100.0, api=pinp.api,
                        sg_water=wsrp.water_sg(pinp.salinity_ppm), ev=ev / 100.0, nu=nu)
    pwf_fn = None
    if pip_m == wsrp.INTAKE_METHODS[1]:
        order = np.argsort(ipr_curve.q)
        qa, pa = np.asarray(ipr_curve.q)[order], np.asarray(ipr_curve.pwf)[order]
        pwf_fn = lambda q: float(np.interp(q, qa, pa))
    if go or go_s:
        try:
            with st.spinner("Solving the rod string wave equation ..."):
                res = wsrp.design(inp, pwf_fn, tvd_at(bottom))
                sens = wsrp.rod_sensitivity(dataclasses.replace(inp, pip=res.inp.pip)) if go_s else None
            ss["srp_result"] = (res, sens)
        except (ValueError, KeyError) as e:
            st.error(str(e))
    with right:
        out = ss.get("srp_result")
        if not out:
            st.info("Set the inputs and press Calculate.")
            return
        res, sens = out
        u = res.kin.unit
        m = st.columns(4)
        m[0].metric("Production (STB/d)", f"{res.q:,.1f}")
        m[1].metric("Stroke rate (spm)", f"{res.n_spm:.4g}")
        m[2].metric("PPRL / MPRL (lbf)", f"{res.pprl:,.0f} / {res.mprl:,.0f}")
        m[3].metric("Max torque (1000 lb-in)", f"{res.torque_max / 1000:,.0f}")
        warn = []
        if res.torque_max > u.gear:
            warn.append(f"peak torque {res.torque_max:,.0f} lb-in exceeds the gear reducer rating {u.gear:,.0f}")
        if res.pprl > u.prl:
            warn.append(f"peak load {res.pprl:,.0f} lbf exceeds the structure rating {u.prl:,.0f}")
        if res.goodman_pct > 100:
            warn.append(f"top rod at {res.goodman_pct:.0f} % of the Goodman diagram")
        if res.inp.stroke > u.max_stroke:
            warn.append(f"stroke {res.inp.stroke:g} in exceeds the unit's maximum {u.max_stroke:g} in")
        for w_ in warn:
            st.warning("Overload: " + w_ + ".")
        t1, t2, t3, t4 = st.tabs(["Dynamometer cards", "Torque", "Results", "Rod sensitivity"])
        with t1:
            st.pyplot(srp_cards_plot(res))
        with t2:
            st.pyplot(srp_torque_plot(res))
            st.caption("Net torque = TF (PRL - SU) - CBE TF(90) sin(theta), with the structural unbalance taken as "
                       "zero (fits PROSPER's torque curve better than the catalogue value) and the counterbalance "
                       "set to CBE.")
        with t3:
            summ = res.summary()
            summ = {"Pump intake pressure (psia)": res.inp.pip, **summ}
            st.dataframe(pd.DataFrame({"Value": summ}).style.format("{:,.6g}"), use_container_width=True)
            st.download_button("Download cards (CSV)", res.card.to_csv(index=False), "srp_cards.csv", key="srp_dl")
        with t4:
            if sens is None:
                st.info("Press Rod sensitivity to compare the rod strings.")
            else:
                st.dataframe(sens, hide_index=True, use_container_width=True)
                st.caption("Production per polished rod horsepower is the comparison PROSPER makes; in the target "
                           "rate mode each rod string gets its own stroke rate.")


def srp_system_tab(o, eqd, ipr_curve):
    """System | Sucker Rod Pumps + system analysis - after PROSPER (solution node at the pump intake)."""
    st.caption("System analysis of an existing rod pump installation after PROSPER System | Sucker Rod Pumps: for a "
               "liquid rate, the IPR gives the flowing BHP and the intake pressure; the gas anchor takes its share of "
               "the gas; the discharge pressure comes from the top node down the tubing; the downhole stroke is "
               "S + Ep - Fo L / (E A) with Fo = Ap (Pdis - PIP); the pump delivers displacement x Ev x the liquid "
               "fraction of its intake volume. The solution is where the pump rate equals the inflow rate. (PROSPER "
               "has no gradient traverse for rod pumps.)")
    srp_card_reference()
    gmd, gt = eqd["geothermal"]
    top_, secs_, pts_ = eqd["downhole"]
    tub = secs_[secs_["Type"] == "Tubing"]
    bottom_tub = float(tub["MD bottom (ft)"].max()) if len(tub) else float(secs_["MD bottom (ft)"].max())
    fluid, fluid_note = vlp_fluid(o, "oil", max(float(ipr_curve.pr), 5000.0) * 1.15, float(np.min(gt)) - 10,
                                  float(np.max(gt)) + 10)
    if fluid is None:
        st.warning(fluid_note)
        return
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        st.markdown("**Sucker rod pump input data**")
        c = st.columns(2)
        pump_md = num(c[0], "Pump depth (ft MD)", "srps_md", float(round(bottom_tub)), float(top_) + 50,
                      float(secs_["MD bottom (ft)"].max()))
        ev = num(c[1], "Pump volumetric efficiency (%)", "srps_ev", 80.0, 1.0, 100.0)
        c = st.columns(2)
        plunger = sel(c[0], "Pump diameter (in)", list(wsrp.PLUNGERS), "srps_plunger", 2.0,
                      format_func=lambda x: f"{x:.2f}")
        rods = wsrp.rod_numbers(plunger)
        rod_no = sel(c[1], "Rod number", rods, "srps_rod", 99 if 99 in rods else rods[-1],
                     format_func=lambda n: wsrp.rod_string(n, plunger).name)
        c = st.columns(2)
        stroke = num(c[0], "Surface stroke length (in)", "srps_s", 48.0, 10.0, 400.0, None, "%.5g")
        n_spm = num(c[1], "Pumping speed (strokes/min)", "srps_n", 20.0, 1.0, 40.0, None, "%.6g")
        st.markdown("**Gas anchor**")
        c = st.columns(2)
        a_meth = sel(c[0], "Method", ["Calculated", "Entered"], "srps_am", "Calculated")
        a_type = sel(c[1], "Type", list(wsrp.GAS_ANCHOR_TYPES), "srps_at", "Cup type")
        c = st.columns(2)
        a_area = num(c[0], "Annulus area (in2)", "srps_aa", 1.5, 1.0, 1e6, None, "%.4g",
                     disabled=a_meth != "Calculated")
        a_eff = num(c[1], "Efficiency (fraction)", "srps_ae", 0.5, 0.0, 1.0, None, "%.4g",
                    disabled=a_meth != "Entered")
        if a_meth == "Calculated" and a_type == "Poor boy":
            st.caption("PROSPER's calculated efficiency for a poor-boy anchor is zero (no gas separated).")
        st.markdown("**Well conditions**")
        c = st.columns(2)
        whp = num(c[0], "Top node pressure (psia)", "srps_whp", float(val("sys_whp", 114.7)), 14.7, 1e4)
        wc = num(c[1], "Water cut (%)", "srps_wc", float(val("sys_wc", 0.0)), 0.0, 100.0)
        c = st.columns(2)
        gor = num(c[0], "Total GOR (scf/STB)", "srps_gor", float(val("sys_gor", 300.0)), 0.0, 1e6)
        corr_opts = [c_ for c_ in wvlp.VLP_CORRELATIONS if c_ not in wvlp.LOWER_BOUNDS]
        corr = sel(c[1], "Tubing correlation", corr_opts, "srps_corr",
                   "no_slip" if "no_slip" in corr_opts else corr_opts[0], format_func=VLP_LABELS.get,
                   help="No slip reproduces PROSPER's rod pump discharge pressures (six runs, 0.3 %).")
        ms = ss.get("vlp_matches", {}).get("ms", {})
        m_fit = ms.get(corr)
        use_fit = flag(st.toggle, "Use matched parameters", "srps_use_fit", disabled=m_fit is None)
        p1, p2 = (float(m_fit.p1), float(m_fit.p2)) if (m_fit is not None and use_fit) else (1.0, 1.0)
        go = st.button("Calculate", type="primary", key="srps_go")
    if go:
        cv = ipr_curve
        if cv.fluid == "oil":
            try:
                cv = rebuild_ipr({"wc": wc / 100.0})
            except ValueError:
                pass
        path = vlp_path(eqd, "tubing", (pump_md,))
        inp = wsrp.SRPSystemInput(rod_no=int(rod_no), plunger=float(plunger), stroke=stroke, n_spm=n_spm,
                                  ev=ev / 100.0, pump_md=pump_md, anchor_method=a_meth, anchor_eff=a_eff,
                                  anchor_type=a_type, anchor_area=a_area)
        try:
            with st.spinner("Solving pump against inflow ..."):
                sol, crv = wsrp.system(inp, path, fluid, lambda q: float(wgl.ipr_pwf(cv, q)), wc / 100.0, gor, whp,
                                       corr, p1, p2, float(val("eq_u", eqd.get("u_geo", 8.0))),
                                       eqd["heat_capacities"], q_max=float(np.max(cv.q)) * 0.995)
            ss["srps_result"] = (sol, crv)
        except ValueError as e:
            st.error(str(e))
    with right:
        out = ss.get("srps_result")
        if not out:
            st.info("Enter the installation and press Calculate.")
            return
        sol, crv = out
        if sol is None:
            st.warning("No solution: the pump rate does not meet the inflow in the rate range (pump too large for "
                       "the well - it pumps off - or the IPR too weak).")
            if len(crv):
                st.dataframe(crv, hide_index=True)
            return
        m = st.columns(3)
        m[0].metric("Liquid rate (STB/d)", f"{sol['Liquid rate (STB/d)']:,.1f}")
        m[1].metric("Pump intake (psig)", f"{sol['Pump intake pressure (psia)'] - 14.696:,.1f}")
        m[2].metric("Pump discharge (psig)", f"{sol['Pump discharge pressure (psia)'] - 14.696:,.1f}")
        st.dataframe(pd.DataFrame({"Value": sol}).style.format("{:,.6g}"), use_container_width=True)
        if len(crv):
            fig, ax = plt.subplots(figsize=(7.8, 4.2))
            ax.plot(crv["q"], crv["q_pump"], color=wplt.SERIES[0], lw=1.6, label="Pump rate at the inflow's intake "
                                                                                 "pressure")
            ax.plot(crv["q"], crv["q"], color=wplt.INK2, ls="--", lw=1.0, label="Inflow rate")
            ax.scatter([sol["Liquid rate (STB/d)"]], [sol["Liquid rate (STB/d)"]], color="#b000b0", zorder=5,
                       label="Solution")
            ax.set_xlabel("Inflow liquid rate (STB/d)")
            ax.set_ylabel("Liquid rate (STB/d)")
            ax.grid(alpha=0.3, ls="--")
            ax.legend(fontsize=7.5)
            fig.tight_layout()
            st.pyplot(fig)


def jet_design_tab(o, eqd, ipr_curve):
    """Design | Jet Pump - after PROSPER (Tutorial 32): pump duty, area ratios, database pumps, selected pump."""
    st.caption("Jet pump design after PROSPER Design | Jet Pump (annular injection, tubing production): the pump "
               "duty at the design rate (intake pressure from the IPR up the casing, discharge pressure from the top "
               "node down the tubing with production + power fluid, nozzle pressure = surface pressure + power fluid "
               "column - annulus friction), the flow and head ratios it needs, the area ratios that deliver them "
               "(Cunningham's equation), and the database pumps that do.")
    top_, secs_, pts_ = eqd["downhole"]
    tub = secs_[secs_["Type"] == "Tubing"]
    bottom_tub = float(tub["MD bottom (ft)"].max()) if len(tub) else float(secs_["MD bottom (ft)"].max())
    gmd, gt = eqd["geothermal"]
    fluid, fluid_note = vlp_fluid(o, "oil", max(float(ipr_curve.pr), 5000.0) * 1.15, float(np.min(gt)) - 10,
                                  float(np.max(gt)) + 10)
    if fluid is None:
        st.warning(fluid_note)
        return
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        st.markdown("**Input data**")
        md, q_pf, p_surf, losses, sal, names = _jet_pump_inputs("jpd", None, bottom_tub, top_)
        c = st.columns(2)
        q_des = num(c[0], "Design rate (STB/d)", "jpd_q", 4000.0, 1.0, 1e6)
        max_od = num(c[1], "Maximum OD (in)", "jpd_od", 6.4, 1.0, 20.0, None, "%.3g")
        c = st.columns(3)
        wc = num(c[0], "Water cut (%)", "jpd_wc", float(val("sys_wc", 0.0)), 0.0, 100.0)
        gor = num(c[1], "Total GOR (scf/STB)", "jpd_gor", float(val("sys_gor", 300.0)), 0.0, 1e6)
        top_p = num(c[2], "Top node pressure (psia)", "jpd_top", float(val("sys_whp", 164.7)), 14.7, 1e4)
        ms = ss.get("vlp_matches", {}).get("ms", {})
        sel_c = ss.get("vlp_selected")
        corr_opts = [c_ for c_ in wvlp.VLP_CORRELATIONS if c_ not in wvlp.LOWER_BOUNDS]
        corr = sel(st, "Tubing correlation", corr_opts, "jpd_corr", sel_c if sel_c in corr_opts else "pe2_style",
                   format_func=VLP_LABELS.get)
        m_fit = ms.get(corr)
        use_fit = flag(st.toggle, "Use matched parameters", "jpd_use_fit", m_fit is not None,
                       disabled=m_fit is None)
        if m_fit is not None and use_fit:
            p1, p2 = float(m_fit.p1), float(m_fit.p2)
        else:
            c2 = st.columns(2)
            p1 = num(c2[0], "Parameter 1 (gravity)", f"jpd_p1_{corr}", 1.0, 0.3, 3.0, None, "%.4f")
            p2 = num(c2[1], "Parameter 2 (friction)", f"jpd_p2_{corr}", 1.0, 0.01, 10.0, None, "%.4f")
        go = st.button("Calculate pump duty", type="primary", key="jpd_go")
    tod, cid = _jet_section_at(secs_, md)
    if go:
        if tod is None:
            st.error("Enter the tubing outside diameter and the casing inside diameter at the pump (Equipment Data) - "
                     "the power fluid goes down that annulus.")
        else:
            cv = ipr_curve
            try:
                cv = rebuild_ipr({"wc": wc / 100.0})
            except ValueError:
                pass
            path = vlp_path(eqd, "tubing", (md,))
            d = wjet.JetDutyInput(md, q_des, wc / 100.0, gor, top_p, q_pf, p_surf, losses, wjet.PowerFluid(sal),
                                  cid, tod, max_od)
            try:
                with st.spinner("Calculating the pump duty ..."):
                    du = wjet.duty(path, fluid, float(wgl.ipr_pwf(cv, q_des)), d, corr, p1, p2,
                                   float(val("eq_u", eqd.get("u_geo", 8.0))), eqd["heat_capacities"])
                ss["jet_duty"] = du
            except ValueError as e:
                st.error(str(e))
    with right:
        du = ss.get("jet_duty")
        if du is None:
            st.info("Set the design conditions and press Calculate pump duty.")
            return
        m = st.columns(4)
        m[0].metric("Intake (psig)", f"{du.pip - 14.696:,.1f}")
        m[1].metric("Discharge needed (psig)", f"{du.pdis - 14.696:,.1f}")
        m[2].metric("Nozzle (psig)", f"{du.pn - 14.696:,.1f}")
        m[3].metric("M / H needed", f"{du.M:.3f} / {du.H:.3f}")
        t1, t2, t3 = st.tabs(["Pump duty", "Area ratios", "Select pump"])
        with t1:
            st.dataframe(wjet.duty_table(du).style.format("{:,.6g}"), use_container_width=True)
        with t2:
            ar = du.area_ratios()
            st.dataframe(ar.style.format(precision=4), hide_index=True, use_container_width=True)
            ok = ar[ar["Meets required H"] & ~ar["Cavitates (M > Mc)"]]
            if len(ok):
                best = ok.loc[ok["Efficiency (%)"].idxmax()]
                st.caption(f"Area ratios {ok['Area ratio R'].min():.2f} - {ok['Area ratio R'].max():.2f} deliver the "
                           f"head ratio needed; the most efficient is R = {best['Area ratio R']:.2f} "
                           f"({best['Efficiency (%)']:.1f} %).")
            else:
                st.warning("No area ratio delivers the head ratio needed at this flow ratio: raise the injection "
                           "pressure or rate, or lower the design rate.")
        with t3:
            cands = wjet.candidates(du)
            if not len(cands):
                st.warning("No database pump covers the power fluid rate at the nozzle "
                           f"({du.q_n:,.0f} RB/d) within the maximum OD.")
                return
            st.dataframe(cands.style.format(precision=4), hide_index=True, use_container_width=True)
            pick = sel(st, "Selected pump", list(cands["Pump"]), "jpd_pump", cands["Pump"].iloc[0])
            r_ = wjet.PUMPS[wjet.PUMPS["Name"] == pick].iloc[0]
            act = wjet.with_pump(du, float(r_["Nozzle area (in2)"]), float(r_["Throat area (in2)"]))
            show = {k_: (v_ - 14.696 if "(psia)" in k_ else v_) for k_, v_ in act.items()}
            show = {k_.replace("(psia)", "(psig)"): v_ for k_, v_ in show.items()}
            st.dataframe(pd.DataFrame({"Actual": show}).astype(str), use_container_width=True)
            ss["jet_design"] = {"pump": pick}
            if not act["Meets the duty"]:
                st.warning("This pump does not reach the discharge pressure needed at the design rate.")


# ================================================================ ESP design
def esp_library():
    pumps = dict(wesp.PUMPS)
    pumps.update(ss.get("esp_user_pumps", {}))
    motors = dict(wesp.MOTORS)
    motors.update(ss.get("esp_user_motors", {}))
    return pumps, motors, dict(wesp.CABLES)


def esp_pump_plot(pump, n, res=None, duty=None, points=None, wear=0.0):
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    marks = {40: "s", 50: "o", 60: "+", 70: "^"}
    lo_l, hi_l, be_l = [], [], []
    for f, mk in marks.items():
        r = f / 60.0
        q = np.linspace(0, pump.q_runout * r, 60)
        h = n * pump.head_stage(q, f, wear)
        ax.plot(q, h, color="#2ca02c", lw=1.4, marker=mk, markevery=6, ms=5, label=f"{f} Hz")
        lo, hi = pump.range_at(f)
        lo_l.append((lo, n * float(pump.head_stage(lo, f, wear))))
        hi_l.append((hi, n * float(pump.head_stage(hi, f, wear))))
        be_l.append((pump.q_bep * r, n * float(pump.head_stage(pump.q_bep * r, f, wear))))
    for pts_, lab, col in ((lo_l, "Min. operating range", "#d62728"), (hi_l, "Max. operating range", "#d62728"),
                           (be_l, f"Best efficiency line ({100 * pump.eff_bep:.2f} %)", "#1f3fd6")):
        a = np.array(pts_)
        ax.plot(a[:, 0], a[:, 1], color=col, lw=1.6, marker="v" if "Min" in lab else (">" if "Max" in lab else "D"),
                ms=5, label=lab)
    if res is not None and duty is not None:
        ax.scatter([duty.q_avg], [res["Number Of Stages"] * res["Head per stage (ft)"]], s=70, color="#b000b0",
                   zorder=7, marker="D", label="Pump operating point")
    if points is not None and len(points):
        ax.scatter(points["Average pump rate (RB/d)"], points["Head (ft)"], s=40, color="#b000b0", zorder=7,
                   marker="D", label="System operating points")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel("Operating rate (RB/d)")
    ax.set_ylabel("Head (ft)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"{pump.name} - {n} stages", fontsize=9.5)
    fig.tight_layout()
    return fig


def esp_motor_plot(motor, load=None):
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    L = np.array(motor.load)
    ax.plot(L, motor.amps_pct, color="#2ca02c", marker="s", ms=4, label="Nameplate amps (%)")
    ax.plot(L, motor.eff, color="#1f3fd6", marker="o", ms=4, label="Motor efficiency (%)")
    ax.plot(L, motor.pf, color="#b000b0", marker="+", ms=6, label="Power factor (%)")
    ax.set_xlabel("Percentage nameplate power (%)")
    ax.set_ylabel("Amps, efficiency, power factor (%)")
    ax2 = ax.twinx()
    for f, mk in ((40, "^"), (50, "v"), (60, "<"), (70, ">")):
        ax2.plot(L, np.array(motor.rpm60) * f / 60.0, color="#d62728", marker=mk, ms=4, lw=1.0)
    ax2.set_ylabel("Motor speed (rpm), 40-70 Hz")
    if load is not None:
        ax.axvline(load, color=wplt.INK2, ls="--", lw=1.0)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"{motor.name}", fontsize=9.5)
    fig.tight_layout()
    return fig


def esp_tab(o, eqd, ipr_curve):
    """Design | Electrical Submersible Pump - after PROSPER (User Guide 2.13.2, Tutorial 26)."""
    st.caption("ESP design after PROSPER Design | Electrical Submersible Pump: Calculate pump duty (flowing BHP from "
               "the IPR at the design rate, gradient up the casing to the pump intake, gradient down the tubing from "
               "the top node to the pump discharge; head from the pressure rise and the average fluid gravity in "
               "the pump), then select the pump, motor and cable (stages, power, current, cable voltage drop, "
               "surface voltage and kVA), and check the pump over a range of water cuts and frequencies. The pump "
               "should sit at the bottom of the tubing; below it the flow is in the casing.")
    gmd, gt = eqd["geothermal"]
    top_, secs_, pts_ = eqd["downhole"]
    tub = secs_[secs_["Type"] == "Tubing"]
    bottom_tub = float(tub["MD bottom (ft)"].max()) if len(tub) else float(secs_["MD bottom (ft)"].max())
    fluid, fluid_note = vlp_fluid(o, "oil", max(float(ipr_curve.pr), 5000.0) * 1.15, float(np.min(gt)) - 10,
                                  float(np.max(gt)) + 10)
    if fluid is None:
        st.warning(fluid_note)
        return
    pumps, motors, cables = esp_library()
    g = 14.696
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        st.markdown("**Input data**")
        c = st.columns(2)
        pump_md = num(c[0], "Pump depth (ft MD)", "esp_md", float(round(bottom_tub)), float(top_) + 50,
                      float(secs_["MD bottom (ft)"].max()) - 1.0)
        freq = num(c[1], "Operating frequency (Hz)", "esp_f", 60.0, 20.0, 90.0, None, "%.4g")
        c = st.columns(2)
        max_od = num(c[0], "Maximum pump OD (in)", "esp_od", 6.0, 2.0, 12.0, None, "%.3g")
        cable_len = num(c[1], "Length of cable (ft)", "esp_cable", float(round(bottom_tub)), 100.0, 40000.0)
        c = st.columns(2)
        sep = num(c[0], "Gas separator efficiency (%)", "esp_sep", 0.0, 0.0, 100.0, help="Share of the free gas at "
                  "the intake vented to the annulus; the rest goes through the pump.")
        q_des = num(c[1], "Design rate (STB/d)", "esp_q", 1000.0, 1.0, 1e6)
        c = st.columns(2)
        wc = num(c[0], "Water cut (%)", "esp_wc", float(val("sys_wc", 0.0)), 0.0, 100.0)
        gor = num(c[1], "Total GOR (scf/STB)", "esp_gor", float(val("sys_gor", 300.0)), 0.0, 1e6)
        c = st.columns(2)
        top_p = num(c[0], "Top node pressure (psia)", "esp_top", float(val("sys_whp", 264.7)), 14.7, 1e4)
        margin = num(c[1], "Motor power safety margin (%)", "esp_margin", 0.0, 0.0, 100.0)
        wear = num(st, "Pump wear factor (fraction)", "esp_wear", 0.0, -0.5, 0.9, None, "%.3g",
                   help="Head = (1 - wear) x catalogue head; 0.05 -> 95 %.")
        ms = ss.get("vlp_matches", {}).get("ms", {})
        sel_c = ss.get("vlp_selected")
        corr_opts = [c_ for c_ in wvlp.VLP_CORRELATIONS if c_ not in wvlp.LOWER_BOUNDS]
        corr = sel(st, "Tubing correlation", corr_opts, "esp_corr",
                   sel_c if sel_c in corr_opts else "hagedorn_brown", format_func=VLP_LABELS.get)
        m_fit = ms.get(corr)
        use_fit = flag(st.toggle, "Use matched parameters", "esp_use_fit", disabled=m_fit is None)
        c2 = st.columns(2)
        if m_fit is not None and use_fit:
            p1, p2 = float(m_fit.p1), float(m_fit.p2)
            c2[0].text_input("Parameter 1 (gravity)", f"{p1:.4f}", disabled=True, key="esp_p1_show")
            c2[1].text_input("Parameter 2 (friction)", f"{p2:.4f}", disabled=True, key="esp_p2_show")
        else:
            p1 = num(c2[0], "Parameter 1 (gravity)", f"esp_p1_{corr}", 1.0, 0.3, 3.0, None, "%.4f")
            p2 = num(c2[1], "Parameter 2 (friction)", f"esp_p2_{corr}", 1.0, 0.01, 10.0, None, "%.4f")
        u = float(val("eq_u", eqd.get("u_geo", 8.0)))
        if abs(bottom_tub - pump_md) > 1.0:
            st.caption(f"Note: the tubing ends at {bottom_tub:,.0f} ft MD, not at the pump; below the pump the "
                       "flow then follows the tubing geometry.")
        go = st.button("Calculate pump duty", type="primary", key="esp_go")
        with st.expander("Equipment library - add a pump or motor"):
            st.caption("Enter the 60 Hz catalogue curve of a pump (rate, head per stage, efficiency) from the "
                       "manufacturer's published curve sheet, or a motor's nameplate (its load curves are taken "
                       "from the Centrilift 544 motor).")
            c = st.columns(4)
            pn = c[0].text_input("Pump name", "", key="esp_new_pn")
            pod = num(c[1], "OD (in)", "esp_new_od", 4.0, 1.0, 12.0, None, "%.3g")
            pmin = num(c[2], "Min rate (RB/d)", "esp_new_min", 1000.0, 0.0, 1e5)
            pmax = num(c[3], "Max rate (RB/d)", "esp_new_max", 3000.0, 0.0, 1e5)
            tab0 = pd.DataFrame({"Rate (RB/d)": [0.0, 1000, 2000, 3000, 4000],
                                 "Head per stage (ft)": [30.0, 28, 24, 18, 0],
                                 "Efficiency (%)": [0.0, 45, 62, 55, 0]})
            ptab = st.data_editor(tab0, num_rows="dynamic", key="esp_new_tab", hide_index=True)
            if st.button("Add pump", key="esp_add_pump") and pn.strip():
                pt = ptab.dropna()
                up = dict(ss.get("esp_user_pumps", {}))
                up[pn.strip()] = wesp.pump_from_table(pn.strip(), pod, pmin, pmax, pt["Rate (RB/d)"],
                                                      pt["Head per stage (ft)"], pt["Efficiency (%)"] / 100.0)
                ss["esp_user_pumps"] = up
                st.rerun()
            c = st.columns(4)
            mn = c[0].text_input("Motor name", "", key="esp_new_mn")
            mhp = num(c[1], "Power (hp)", "esp_new_hp", 150.0, 1.0, 2000.0)
            mv = num(c[2], "Voltage (V)", "esp_new_v", 1500.0, 100.0, 6000.0)
            ma = num(c[3], "Current (A)", "esp_new_a", 60.0, 1.0, 500.0)
            if st.button("Add motor", key="esp_add_motor") and mn.strip():
                um = dict(ss.get("esp_user_motors", {}))
                um[mn.strip()] = wesp.motor_like(mn.strip(), mhp, mv, ma)
                ss["esp_user_motors"] = um
                st.rerun()
    curve_ = ipr_curve
    if curve_.fluid == "oil":
        try:
            curve_ = rebuild_ipr({"wc": wc / 100.0})
        except ValueError:
            pass
    path = vlp_path(eqd, "tubing", (pump_md,))
    sig = (pump_md, freq, sep, q_des, wc, gor, top_p, corr, p1, p2, u, float(curve_.pr), float(curve_.aof))
    with right:
        if go:
            try:
                pwf = float(wgl.ipr_pwf(curve_, q_des))
                if not np.isfinite(pwf):
                    raise ValueError("The design rate is above the well's AOF.")
                with st.spinner("Calculating pump duty..."):
                    d = wesp.pump_duty(path, fluid, pwf, q_des, wc / 100.0, gor, top_p, pump_md, sep / 100.0, corr,
                                       p1, p2, u, eqd["heat_capacities"])
                ss["esp_duty"] = {"sig": sig, "d": d, "err": None}
            except (ValueError, RuntimeError, ZeroDivisionError) as e:
                ss["esp_duty"] = {"sig": sig, "d": None, "err": str(e)}
        out = ss.get("esp_duty")
        if out is None:
            st.info("Enter the design data and press Calculate pump duty.")
            return
        if out["err"]:
            st.error(out["err"])
            return
        if out["sig"] != sig:
            st.info("Inputs have changed - press Calculate pump duty to update.")
        d = out["d"]
        st.markdown("**Pump duty**")
        dt = wesp.duty_table(d, g, freq)
        dt.loc[0, "Value"] = top_p - g
        st.dataframe(dt.style.format({"Value": "{:,.6g}"}, na_rep="—"), hide_index=True, width="stretch")
        st.markdown("**Equipment**")
        cid = float(tub["Casing ID (in)"].iloc[-1]) if len(tub) and pd.notna(tub["Casing ID (in)"].iloc[-1]) else 99.0
        cand = wesp.candidate_pumps(d, pumps, freq, max_od, cid, wear)
        st.dataframe(cand.style.format(precision=2), hide_index=True, width="stretch")
        ok_names = list(cand.loc[cand["Suitable"], "Pump"]) or list(pumps)
        c = st.columns(3)
        pname = sel(c[0], "Pump", ok_names, "esp_pump")
        pump = pumps[pname]
        n_auto = int(np.ceil(d.head / float(pump.head_stage(d.q_avg, freq, wear))))
        p_need = d.fluid_hp * n_auto * float(pump.head_stage(d.q_avg, freq, wear)) / d.head / max(
            float(pump.efficiency(d.q_avg, freq)), 1e-6) * (1 + margin / 100.0)
        mot_ok = [k for k, m in motors.items() if m.hp * freq / 60.0 >= p_need and m.od < cid] or list(motors)
        mot_ok = sorted(mot_ok, key=lambda k: motors[k].hp)
        mname = sel(c[1], "Motor", mot_ok, "esp_motor",
                    help=f"Motors able to give {p_need:,.0f} hp at {freq:g} Hz that fit the casing, smallest first "
                    "(PROSPER's motor database; 50 Hz motors shown at their 60 Hz equivalent rating).")
        motor = motors[mname]
        # smallest cable carrying the current (PROSPER)
        r0 = wesp.design(d, pump, motor, cables["#1 Copper"], freq, cable_len, wear, margin / 100.0)
        cab_ok = [k for k, cb in sorted(cables.items(), key=lambda kv: -kv[1].v_per_kft_amp)
                  if cb.max_amps >= r0["Current Used (amps)"]] or list(cables)
        cname = sel(c[2], "Cable", cab_ok, "esp_cable_sel",
                    format_func=lambda k: f"{k}  {cables[k].v_per_kft_amp:g} V/1000 ft/A, {cables[k].max_amps:g} A max")
        res = wesp.design(d, pump, motor, cables[cname], freq, cable_len, wear, margin / 100.0, casing_id=cid,
                          motor_od=motor.od)
        cp_ = float(fluid.mass_cp({"q": np.array([q_des]), "wc": np.array([wc / 100]), "gor": np.array([gor])},
                                  eqd["heat_capacities"])[1][0])
        res["Pump Outlet Temperature (deg F)"] = wesp.outlet_temperature(d.t_intake, res["Pump and motor losses (hp)"],
                                                                         d.mass, cp_)
        st.markdown("**Results**")
        rows_ = [(k_.rsplit(" (", 1)[0], v_, k_.rsplit(" (", 1)[1][:-1] if k_.endswith(")") else "")
                 for k_, v_ in res.items() if k_ != "notes"]
        st.dataframe(pd.DataFrame(rows_, columns=["Result", "Value", "Units"]).style.format({"Value": "{:,.6g}"}),
                     hide_index=True, width="stretch")
        for n_ in res["notes"]:
            st.warning(n_)
        ss["esp_design"] = {"pump": pname, "motor": mname, "cable": cname, "stages": res["Number Of Stages"],
                            "freq": freq, "pump_md": pump_md}
        ss["_esp_design_inputs"] = {"esp_cable": cable_len, "esp_sep": sep, "esp_wear": wear}
        t1, t2, t3 = st.tabs(["Pump plot", "Motor plot", "System sensitivity"])
        with t1:
            st.pyplot(esp_pump_plot(pump, res["Number Of Stages"], res, d, wear=wear), clear_figure=True)
            st.caption(f"Source: {pump.source}.")
        with t2:
            st.pyplot(esp_motor_plot(motor, res["Motor Load (%)"]), clear_figure=True)
            st.caption(f"Source: {motor.source}.")
        with t3:
            st.caption("System analysis with this pump (PROSPER's convention: solution at the pump discharge, the "
                       "VLP from the top node against the IPR corrected to the intake plus the pump pressure gain).")
            c = st.columns(2)
            wcs = c[0].text_input("Water cuts (%)", "60, 70, 80, 90", key="esp_s_wc")
            frs = c[1].text_input("Frequencies (Hz)", f"{freq:g}", key="esp_s_f")
            if st.button("Calculate", key="esp_s_go"):
                try:
                    wl = [float(x) for x in wcs.replace(";", ",").split(",") if x.strip()]
                    fl = [float(x) for x in frs.replace(";", ",").split(",") if x.strip()]
                    rows_s = []
                    with st.spinner("System analysis..."):
                        for w_ in wl:
                            cv = curve_
                            try:
                                cv = rebuild_ipr({"wc": w_ / 100.0})
                            except ValueError:
                                pass
                            for f_ in fl:
                                sol, _ = wesp.system(path, fluid, lambda q, cv=cv: float(wgl.ipr_pwf(cv, q)),
                                                     w_ / 100.0, gor, top_p, pump_md, pump, res["Number Of Stages"],
                                                     f_, wear, sep / 100.0, corr, p1, p2, u, eqd["heat_capacities"],
                                                     q_max=float(np.max(cv.q)) * 0.995, n=25, motor=motor,
                                                     method="stages")
                                row = {"Water cut (%)": w_, "Frequency (Hz)": f_}
                                if sol is None:
                                    row["Liquid rate (STB/d)"] = np.nan
                                else:
                                    row.update({k: v for k, v in sol.items()})
                                rows_s.append(row)
                    ss["esp_sys"] = pd.DataFrame(rows_s)
                except (ValueError, RuntimeError) as e:
                    ss["esp_sys"] = str(e)
            sres = ss.get("esp_sys")
            if isinstance(sres, str):
                st.error(sres)
            elif sres is not None and len(sres):
                st.dataframe(sres.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
                okp = sres.dropna(subset=["Liquid rate (STB/d)"])
                if len(okp):
                    st.pyplot(esp_pump_plot(pump, res["Number Of Stages"], points=okp, wear=wear), clear_figure=True)
                    fig, ax = plt.subplots(figsize=(7.0, 3.8))
                    for f_, gdf in okp.groupby("Frequency (Hz)"):
                        ax.plot(gdf["Water cut (%)"], gdf["Oil rate (STB/d)"], marker="o", label=f"{f_:g} Hz")
                    ax.set_xlabel("Water cut (%)")
                    ax.set_ylabel("Oil rate (STB/d)")
                    ax.grid(alpha=0.3, ls="--")
                    ax.legend(fontsize=8)
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True)


def esp_sys_plot(curves):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for k, (lab, crv, sol) in enumerate(curves[:8]):
        col = wplt.SERIES[k % len(wplt.SERIES)]
        ax.plot(crv["Liquid rate (STB/d)"], crv["VLP at pump discharge (psia)"], color=col, lw=1.6, label=f"VLP {lab}")
        ax.plot(crv["Liquid rate (STB/d)"], crv["Pump discharge from IPR (psia)"], color=col, lw=1.4, ls="--",
                label=f"Pump discharge {lab}")
        if sol is not None:
            ax.scatter([sol["Liquid rate (STB/d)"]], [sol["Pump discharge pressure (psia)"]], color=col, s=40, zorder=6)
    ax.set_xlabel("Liquid rate (STB/d)")
    ax.set_ylabel("Pressure at the pump discharge (psia)")
    ax.set_xlim(0, None)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_title("ESP system: VLP against the inflow at the pump discharge", fontsize=9.5)
    fig.tight_layout()
    return fig


def jet_system_calc(go, fluid, eqd, base, corr, p1, p2, sens, ipr_for, labels):
    """Calculation | System for a jet pump well (the jet pump against the tubing at the pump)."""
    eq = ss.get("jet_equipment")
    if not eq:
        st.info("Enter the pump in the Equipment Data tab (Jet pump section).")
        return
    sig = ("jet", tuple(sorted(base.items())), corr, p1, p2, tuple((k_, tuple(v_)) for k_, v_ in sens.items()),
           tuple(sorted((k_, repr(v_)) for k_, v_ in eq.items())))
    if go:
        try:
            combos = wsys.combinations(sens)
            path = vlp_path(eqd, "tubing", (eq["pump_md"],))
            results = []
            with st.spinner(f"Calculating {len(combos)} case(s)..."):
                for ch in combos:
                    prm = {**base, **{k_: v_ for k_, v_ in ch.items() if k_ in ("whp", "wc", "gor", "u")}}
                    pump = wjet.JetPumpInstalled(eq["pump_md"], eq["an"], eq["at"], float(ch.get("q_pf", eq["q_pf"])),
                                                 float(ch.get("p_surf", eq["p_surf"])), eq["losses"],
                                                 wjet.PowerFluid(eq["sal"]), eq["casing_id"], eq["tubing_od"],
                                                 eq["name"])
                    changes = {k_: float(ch[k_]) for k_ in wsys.IPR_KEYS if k_ in ch}
                    changes["wc"] = float(prm["wc"]) / 100.0
                    if "gor" in prm:
                        changes["gor"] = float(prm["gor"])
                    cv = ipr_for(changes)
                    sol, crv = wjet.system(path, fluid, lambda q, cv=cv: float(wgl.ipr_pwf(cv, q)), prm["wc"] / 100.0,
                                           prm["gor"], prm["whp"], pump, corr, p1, p2, prm["u"],
                                           eqd["heat_capacities"], q_max=float(np.max(cv.q)) * 0.995, n=25)
                    results.append((ch, sol, crv))
            ss["jet_sys_res"] = {"sig": sig, "results": results}
        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            st.error(f"Calculation failed: {e}")
    out = ss.get("jet_sys_res")
    if not out:
        st.info("Press Calculate.")
        return
    if out["sig"] != sig:
        st.info("Inputs have changed since the last calculation - press Calculate to update.")
    rows = []
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for i, (ch, sol, crv) in enumerate(out["results"]):
        lab = ", ".join(f"{labels.get(k_, k_).split(' (')[0]} {v_:g}" for k_, v_ in ch.items()) or "base"
        d_ = {labels.get(k_, k_): v_ for k_, v_ in ch.items()}
        if sol is None:
            d_["Note"] = "No solution - the jet pump cannot lift this case to the top node pressure."
        else:
            d_.update(sol)
        rows.append(d_)
        if len(crv) and i < 8:
            col = wplt.SERIES[i % len(wplt.SERIES)]
            ax.plot(crv["Liquid rate (STB/d)"], crv["Pump discharge from the pump (psia)"], color=col, lw=1.5,
                    label=f"Pump - {lab}")
            ax.plot(crv["Liquid rate (STB/d)"], crv["Discharge needed by the tubing (psia)"], color=col, lw=1.2,
                    ls="--", label=f"Tubing - {lab}")
            if sol is not None:
                ax.scatter([sol["Liquid rate (STB/d)"]], [sol["Pump discharge pressure (psia)"]], color=col, zorder=5)
    tab_ = pd.DataFrame(rows)
    st.markdown(f"**Results** - {len(rows)} case(s); pump {ss['jet_equipment']['name']}")
    st.dataframe(tab_.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
    st.download_button("Download jet pump system results (CSV)", tab_.to_csv(index=False).encode(), "jet_system.csv")
    if "Cavitation" in tab_ and (tab_["Cavitation"] == "Yes").any():
        st.warning("Cavitation predicted in some cases (flow ratio above the cavitation limit at Ic = 0.8).")
    ax.set_xlabel("Liquid rate (STB/d)")
    ax.set_ylabel("Pump discharge pressure (psia)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    if len(tab_) > 1 and "Oil rate (STB/d)" in tab_:
        var_cols = [labels.get(k_, k_) for k_ in sens]
        if var_cols:
            fig2, ax2 = plt.subplots(figsize=(7.8, 4.0))
            x_ = var_cols[0]
            groups = tab_.groupby(var_cols[1:]) if len(var_cols) > 1 else [("", tab_)]
            for gk, gdf in groups:
                ax2.plot(gdf[x_], gdf["Oil rate (STB/d)"], marker="o", ms=4,
                         label=(f"{var_cols[1].split(' (')[0]} {gk if not isinstance(gk, tuple) else gk[0]:g}"
                                if len(var_cols) > 1 else None))
            ax2.set_xlabel(x_)
            ax2.set_ylabel("Oil rate (STB/d)")
            ax2.grid(alpha=0.3, ls="--")
            if len(var_cols) > 1:
                ax2.legend(fontsize=7)
            fig2.tight_layout()
            st.pyplot(fig2, clear_figure=True)
    st.caption("As PROSPER: the power fluid rate and surface pressure are fixed; the pump's discharge pressure is "
               "(Ps + H Pn)/(1 + H) with H from its area ratio at the flow ratio M (Cunningham); the solution is "
               "where it meets the discharge pressure the tubing needs for production + power fluid.")


def srp_system_calc(go, fluid, eqd, base, corr, p1, p2, sens, ipr_for, labels):
    """Calculation | System for a sucker rod pump well (PROSPER: solution node at the pump intake)."""
    eq = ss.get("srp_equipment")
    if not eq:
        st.info("Enter the pump in the Equipment Data tab (Sucker rod pump section).")
        return
    corr_ = "no_slip" if eq["no_slip"] else corr
    p1_, p2_ = (1.0, 1.0) if eq["no_slip"] else (p1, p2)
    sig = ("srp", tuple(sorted(base.items())), corr_, p1_, p2_, tuple((k_, tuple(v_)) for k_, v_ in sens.items()),
           tuple(sorted(eq.items())))
    if go:
        try:
            combos = wsys.combinations(sens)
            path = vlp_path(eqd, "tubing", (eq["pump_md"],))
            results = []
            with st.spinner(f"Calculating {len(combos)} case(s)..."):
                for ch in combos:
                    prm = {**base, **{k_: v_ for k_, v_ in ch.items() if k_ in ("whp", "wc", "gor", "u")}}
                    e_ = {**eq, **{k_: float(ch[k_]) for k_ in SRP_VARS if k_ in ch}}
                    pl = min(wsrp.PLUNGERS, key=lambda x: abs(x - e_["plunger"]))
                    rod = e_["rod_no"] if e_["rod_no"] in wsrp.rod_numbers(pl) else wsrp.rod_numbers(pl)[-1]
                    inp = wsrp.SRPSystemInput(rod_no=int(rod), plunger=pl, stroke=e_["stroke"], n_spm=e_["n_spm"],
                                              ev=e_["ev"] / 100.0, pump_md=e_["pump_md"],
                                              anchor_method=e_["anchor_method"], anchor_eff=e_["anchor_eff"],
                                              anchor_type=e_["anchor_type"], anchor_area=e_["anchor_area"])
                    changes = {k_: float(ch[k_]) for k_ in wsys.IPR_KEYS if k_ in ch}
                    changes["wc"] = float(prm["wc"]) / 100.0
                    if "gor" in prm:
                        changes["gor"] = float(prm["gor"])
                    cv = ipr_for(changes)
                    sol, crv = wsrp.system(inp, path, fluid, lambda q, cv=cv: float(wgl.ipr_pwf(cv, q)),
                                           prm["wc"] / 100.0, prm["gor"], prm["whp"], corr_, p1_, p2_, prm["u"],
                                           eqd["heat_capacities"], q_max=float(np.max(cv.q)) * 0.995)
                    results.append((ch, sol, crv))
            ss["srp_sys_res"] = {"sig": sig, "results": results}
        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            st.error(f"Calculation failed: {e}")
    out = ss.get("srp_sys_res")
    if not out:
        st.info("Press Calculate.")
        return
    if out["sig"] != sig:
        st.info("Inputs have changed since the last calculation - press Calculate to update.")
    rows = []
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    qmax = 0.0
    for i, (ch, sol, crv) in enumerate(out["results"]):
        lab = ", ".join(f"{labels.get(k_, k_).split(' (')[0]} {v_:g}" for k_, v_ in ch.items()) or "base"
        d_ = {labels.get(k_, k_): v_ for k_, v_ in ch.items()}
        if sol is None:
            d_["Note"] = "No solution - the pump rate does not meet the inflow (the pump would pump the well off)."
        else:
            d_.update(sol)
        rows.append(d_)
        if len(crv):
            col = wplt.SERIES[i % len(wplt.SERIES)]
            ax.plot(crv["q"], crv["q_pump"], color=col, lw=1.5, label=lab)
            qmax = max(qmax, float(crv["q"].max()))
            if sol is not None:
                ax.scatter([sol["Liquid rate (STB/d)"]], [sol["Liquid rate (STB/d)"]], color=col, zorder=5)
    tab_ = pd.DataFrame(rows)
    st.markdown(f"**Results** - {len(rows)} case(s)")
    st.dataframe(tab_.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
    st.download_button("Download sucker rod system results (CSV)", tab_.to_csv(index=False).encode(),
                       "srp_system.csv")
    if qmax > 0:
        ax.plot([0, qmax], [0, qmax], color=wplt.INK2, ls="--", lw=1.0, label="Inflow rate")
        ax.set_xlabel("Inflow liquid rate (STB/d)")
        ax.set_ylabel("Pump liquid rate (STB/d)")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=7)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    st.caption("PROSPER's convention for rod pump wells: the solution node is the pump intake. For each inflow rate "
               "the pump rate follows from the intake and discharge pressures (downhole stroke S + Ep - stretch, "
               "volumetric efficiency, gas through the pump after the anchor); the solution is where the two rates "
               "agree. The tubing above the pump carries (1 - E) of the gas.")


def esp_system_calc(go, fluid, eqd, base, corr, p1, p2, sens, ipr_for, labels):
    """Calculation | System for an ESP well (PROSPER: solution node at the pump discharge)."""
    esp = ss.get("esp_equipment")
    if not esp:
        st.info("Enter the pump in the Equipment Data tab (Electrical submersible pump section).")
        return
    pump, motor, cable = esp["pump"], esp["motor"], esp["cable"]
    sig = ("esp", tuple(sorted(base.items())), corr, p1, p2, tuple((k_, tuple(v_)) for k_, v_ in sens.items()),
           esp["pump_md"], esp["freq"], esp["stages"], pump.name, motor.name, cable.name, esp["sep"], esp["wear"],
           esp["method"], esp.get("chp"))
    if go:
        try:
            combos = wsys.combinations(sens)
            path = vlp_path(eqd, "tubing", (esp["pump_md"],))
            results = []
            with st.spinner(f"Calculating {len(combos)} case(s)..."):
                for ch in combos:
                    prm = {**base, **{k_: v_ for k_, v_ in ch.items() if k_ in ("whp", "wc", "gor", "u")}}
                    f_ = float(ch.get("esp_freq", esp["freq"]))
                    n_ = int(round(ch.get("esp_stages", esp["stages"])))
                    changes = {k_: float(ch[k_]) for k_ in wsys.IPR_KEYS if k_ in ch}
                    changes["wc"] = float(prm["wc"]) / 100.0
                    if "gor" in prm:
                        changes["gor"] = float(prm["gor"])
                    cv = ipr_for(changes)
                    sol, crv = wesp.system(path, fluid, lambda q, cv=cv: float(wgl.ipr_pwf(cv, q)),
                                           prm["wc"] / 100.0, prm["gor"], prm["whp"], esp["pump_md"], pump, n_, f_,
                                           esp["wear"], esp["sep"], corr, p1, p2, prm["u"], eqd["heat_capacities"],
                                           q_max=float(np.max(cv.q)) * 0.995, n=25, motor=motor,
                                           method=esp["method"])
                    if sol is not None:
                        sol.update(wesp.operate(sol["Pump power (hp)"], motor, cable, f_, esp["cable_len"],
                                                sol["Average cable temperature (F)"]))
                        sol["Pump min. rate (RB/d)"], sol["Pump max. rate (RB/d)"] = pump.range_at(f_)
                        if "level" in esp:
                            sol["Liquid level in annulus (ft TVD)"] = float(esp["level"])
                            sol["Casing head pressure (psia)"] = wcbm.casing_head_pressure(
                                fluid, sol["Pump intake pressure (psia)"], sol["Pump intake temperature (F)"],
                                float(path.t_form[0]), float(np.interp(esp["pump_md"], path.md, path.tvd)),
                                float(esp["level"]))
                            results.append((ch, f_, n_, sol, crv))
                            continue
                        sol["Casing head pressure (psia)"] = esp.get("chp", 264.7)
                        sol["Liquid level in annulus (ft TVD)"] = wesp.liquid_level(
                            fluid, sol["Pump intake pressure (psia)"], sol["Pump intake temperature (F)"],
                            {"q": np.array([sol["Liquid rate (STB/d)"]]), "wc": np.array([prm["wc"] / 100.0]),
                             "gor": np.array([float(prm["gor"])])},
                            float(np.interp(esp["pump_md"], path.md, path.tvd)), esp.get("chp", 264.7),
                            float(path.tvd[0]))
                    results.append((ch, f_, n_, sol, crv))
            ss["esp_sys_res"] = {"sig": sig, "results": results}
        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            st.error(f"Calculation failed: {e}")
    out = ss.get("esp_sys_res")
    if not out:
        st.info("Press Calculate.")
        return
    if out["sig"] != sig:
        st.info("Inputs have changed since the last calculation - press Calculate to update.")
    rows, curves, pts = [], [], []
    for ch, f_, n_, sol, crv in out["results"]:
        lab = ", ".join(f"{labels.get(k_, k_).split(' (')[0]} {v_:g}" for k_, v_ in ch.items()) or "base"
        d_ = {labels.get(k_, k_): v_ for k_, v_ in ch.items()}
        if sol is None:
            d_["Note"] = "No solution - the pump cannot lift this case to the top node pressure."
        else:
            d_.update({k_: v_ for k_, v_ in sol.items() if not isinstance(v_, bool)})
            d_["In recommended range"] = "Yes" if sol["In recommended range"] else "No"
            pts.append({"Average pump rate (RB/d)": sol["Average pump rate (RB/d)"], "Head (ft)": sol["Head (ft)"]})
        rows.append(d_)
        curves.append((lab, crv, sol))
    tab_ = pd.DataFrame(rows)
    st.markdown(f"**Results** - {len(rows)} case(s); pump {pump.name}, {out['results'][0][2]} stages, motor "
                f"{motor.name}, cable {cable.name}")
    st.dataframe(tab_.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
    st.download_button("Download ESP system results (CSV)", tab_.to_csv(index=False).encode(), "esp_system.csv")
    over = tab_[tab_.get("Motor load (%)", pd.Series(dtype=float)) > 100] if "Motor load (%)" in tab_ else []
    if len(over):
        st.warning(f"{len(over)} case(s) overload the motor (load above 100 % of nameplate at that frequency).")
    cc = st.columns([1.1, 1])
    with cc[0]:
        st.pyplot(esp_sys_plot(curves), clear_figure=True)
    with cc[1]:
        st.pyplot(esp_pump_plot(pump, out["results"][0][2], points=pd.DataFrame(pts), wear=esp["wear"]),
                  clear_figure=True)
    st.caption("PROSPER's convention for ESP wells: the solution node is the pump discharge. VLP = top node down "
               "the tubing to the pump; pump discharge = IPR corrected up the casing to the intake plus the pump "
               "pressure gain. Motor load, current and voltages from the Equipment tab's motor and cable.")


with tabs[5]:
    st.header("Calculation")
    _o = ss.get("options")
    ESP_ON = bool(_o is not None and getattr(_o, "artificial_lift", "none") == "esp")
    GL_ON = bool(_o is not None and getattr(_o, "artificial_lift", "none") in ("gas_lift", "ct_gas_lift"))
    SRP_ON = bool(_o is not None and getattr(_o, "artificial_lift", "none") == "srp")
    JET_ON = bool(_o is not None and getattr(_o, "artificial_lift", "none") == "jet")
    PCP_ON = bool(_o is not None and getattr(_o, "artificial_lift", "none") in ("pcp_motor", "pcp_rod"))
    HSP_ON = bool(_o is not None and getattr(_o, "artificial_lift", "none") == "hsp")
    DIL_ON = bool(_o is not None and getattr(_o, "artificial_lift", "none") in ("diluent", "diluent_gl"))
    FOAM_ON = bool(_o is not None and getattr(_o, "artificial_lift", "none") == "foam_lift")
    LIFT_ON = bool(_o is not None and getattr(_o, "artificial_lift", "none") not in ("none", "foam_lift"))
    _modes = CALC_MODES[:2] + ([VLPC_MODE] if not LIFT_ON and _o is not None and not _o.is_injector else []) + \
        CALC_MODES[2:] + ([QL_MODE] if GL_ON else []) + ([ESPQL_MODE] if ESP_ON else []) + ([HSPQL_MODE] if HSP_ON else []) + \
        ([EQC_MODE] if GL_ON and getattr(_o, "artificial_lift", "") == "gas_lift" else [])
    if val("calc_mode") not in _modes:
        put("calc_mode", CALC_MODES[0])
    mode = radio(st, "Calculation", _modes, "calc_mode", horizontal=True, label_visibility="collapsed")
    SYS = mode == CALC_MODES[0]
    CHOKE = mode == CALC_MODES[3]
    QLK = mode == QL_MODE
    EQL = mode == ESPQL_MODE
    HQL = mode == HSPQL_MODE
    EQC = mode == EQC_MODE
    gl_setup = ss.get("gl_setup") if GL_ON else None
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
        blocker = (None if eqd.get("valid") and eqd.get("survey") is not None and len(eqd["downhole"][1])
                   else "Complete the Equipment Data tab first.")
    elif not eqd.get("valid") or eqd.get("survey") is None or not len(eqd["downhole"][1]):
        blocker = "Complete the Equipment Data tab first."
    elif (SYS or QLK or EQC or EQL or HQL) and (ipr_curve is None or ipr_curve.fluid != ("gas" if o.is_gas else "oil")):
        blocker = "Calculate an IPR for this fluid type in the IPR Data tab first."
    elif SRP_ON and mode == CALC_MODES[1]:
        blocker = ("PROSPER has no gradient traverse for a sucker rod pump well: use System analysis (solution at the "
                   "pump intake) or Lift curves.")
    elif JET_ON and not ss.get("jet_equipment"):
        blocker = "Enter the jet pump in the Equipment Data tab first."
    elif HSP_ON and not ss.get("hsp_equipment"):
        blocker = "Enter the hydraulic submersible pump in the Equipment Data tab first."
    elif PCP_ON and not ss.get("pcp_equipment"):
        blocker = "Enter the progressive cavity pump in the Equipment Data tab first."
    elif SRP_ON and not ss.get("srp_equipment"):
        blocker = "Enter the sucker rod pump in the Equipment Data tab first."
    elif GL_ON and gl_setup is None:
        blocker = "Enter the gas lift data in the Equipment Data tab first."
    elif DIL_ON and not ss.get("diluent_setup"):
        blocker = "Enter the diluent injection data in the Equipment Data tab first."
    elif DIL_ON and mode != CALC_MODES[0] and not CHOKE:
        blocker = ("Diluent injection is modelled in the system analysis (with the diluent rate, its percentage and "
                   "the GLR injected as sensitivity variables).")
    elif GL_ON and not (QLK or EQC) and gl_setup.method in wgls.METHODS[2:] and not gl_setup.valves:
        blocker = "Enter the gas lift valve depths in the Equipment Data tab (Gas lift data) first."
    if blocker:
        st.info(blocker)
    elif CHOKE:
        choke_mode(o)
    elif o.is_injector:
        comp_panel()
        injector_calc(o, mode, eqd)
    else:
        comp_panel()
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
                if s_items is not None and GL_ON:
                    top_node = "Wellhead"
                    st.caption("Gas lift: the top node is the wellhead (surface equipment not used).")
                elif s_items is not None:
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
                if o.fluid_type == "cbm":
                    base["wc"] = 100.0
                    base["gor"] = num(st, "CBM GLR (scf/STB)", "sys_gor", float(val("pvt_cbm_glr", 10.0)), 0.0, 1e6,
                                      help="Gas-liquid ratio of the produced water (all free gas).")
                    st.caption("Coalbed methane: water only (water cut 100 %); the gas leaves through the ESP gas "
                               "separator up the annulus.")
                elif kind == "oil":
                    base["wc"] = num(st, "Water cut (%)", "sys_wc",
                                     float(t0_["wc"] * 100) if t0_ is not None else float(val("ipr_wc", 0.0)),
                                     0.0, 99.0)
                    m_ = blackoil_model(o) if o.method != "eos" else None
                    base["gor"] = num(st, "Total GOR (scf/STB)", "sys_gor",
                                      float(t0_["gor"]) if t0_ is not None else
                                      (float(m_.inp.rsb) if m_ else 500.0), 0.0, 1e6)
                    if getattr(ipr_curve, "mix", None) is not None:
                        st.caption("Multilayer IPR: the water cut and GOR above are not used - at each rate they "
                                   "follow the layer contributions (PROSPER: dependent on the IPR pressure).")
                else:
                    base["wgr"] = num(st, "Water gas ratio (STB/MMscf)", "sys_wgr",
                                      float(t0_["wgr"]) if t0_ is not None else 0.0, 0.0, 1e5)
                    cg_ = float(t0_["cgr"]) if t0_ is not None and t0_["cgr"] > 0 else 0.0
                    base["sep_gor"] = num(st, "Separator GOR (scf/STB)", "sys_sep_gor",
                                          round(1e6 / cg_) if cg_ > 0 else 0.0, 0.0, 1e8,
                                          help="Separator gas per barrel of condensate; 0 = dry gas.")
                    if FOAM_ON and ss.get("foam_setup"):
                        base["surf"] = float(ss["foam_setup"]["conc"])
                        st.caption(f"Foam lift: surfactant {base['surf']:g} % (Equipment tab; a sensitivity "
                                   "variable in the system analysis).")
                    if getattr(ipr_curve, "mix", None) is not None:
                        st.caption("Multilayer IPR: the WGR and separator GOR above are not used - at each rate they "
                                   "follow the layer contributions.")
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
                if FOAM_ON:
                    corr_opts = ["foam_homogeneous"] + corr_opts
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
                    node_lab = sel(st, "Solution node", ["Bottom node", "Top node"], "sys_node", disabled=GL_ON or DIL_ON,
                                   help="Bottom node: IPR against VLP at the bottom of the downhole equipment - the "
                                        "solution pressure is the flowing bottom-hole pressure. Top node: inflow (IPR, "
                                        "tubing" + (", surface equipment" if use_surface else "") + ") against the "
                                        "top-node pressure at the " + ("manifold" if use_surface else "wellhead") +
                                        ". The solution rate is the same at either node; only the curves change.")
                    node = "top" if node_lab == "Top node" and not (GL_ON or DIL_ON) else "bottom"
                    if GL_ON:
                        st.caption(f"Gas lift ({gl_setup.method.lower()}): solved at the bottom node.")
                    method = sel(st, "Rate method", RATE_METHODS, "sys_rate_method")
                    user_rates = None
                    if method == "User selected":
                        user_rates = num_list(st, f"Rates ({'STB/d' if kind == 'oil' else 'MMscf/d'})", "sys_rates",
                                              "100, 500, 1000, 2000, 3000" if kind == "oil" else "1, 5, 10, 20, 30")
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
                    if surface is None or not wsurf.has_pump(surface.items):
                        var_defs.pop("pump_power", None)
                    if surface is not None:
                        var_defs["whp"] = "Manifold pressure (psia)"
                    if ESP_ON:
                        var_defs.pop("tubing_id", None)
                        var_defs.pop("choke", None)
                        var_defs["esp_freq"] = "Operating frequency (Hz)"
                        var_defs["esp_stages"] = "Number of stages"
                    if JET_ON:
                        var_defs.pop("tubing_id", None)
                        var_defs.pop("choke", None)
                        var_defs.update(JET_VARS)
                    if PCP_ON:
                        var_defs.pop("tubing_id", None)
                        var_defs.pop("choke", None)
                        var_defs.update(PCP_VARS)
                    if HSP_ON:
                        var_defs.pop("tubing_id", None)
                        var_defs.pop("choke", None)
                        var_defs.update(HSP_VARS)
                    if SRP_ON:
                        var_defs.pop("tubing_id", None)
                        var_defs.pop("choke", None)
                        var_defs.update(SRP_VARS)
                    if GL_ON:
                        var_defs.update(wgls.variables_for(gl_setup))
                    if FOAM_ON:
                        var_defs["surf"] = "Surfactant mass percent (%)"
                    if DIL_ON:
                        var_defs.update(DIL_VARS if ss["diluent_setup"].gas_lift else
                                        {k_: v_ for k_, v_ in DIL_VARS.items() if k_ != "glr_dil"})
                    sens = {}
                    if sens_on:
                        chosen = st.multiselect("Variables (up to 3)", list(var_defs), format_func=var_defs.get,
                                                max_selections=3, key="sys_vars")
                        tub_id = float(eqd["downhole"][1].query("Type == 'Tubing'")["Flow ID (in)"].iloc[-1])
                        cur = {**base, "pr": ipr_arg("pr"), "j": ipr_arg("j"), "skin": ipr_arg("skin"),
                               "tubing_id": tub_id}
                        if GL_ON:
                            cur.update({"qgi": gl_setup.qgi, "glr_inj": gl_setup.glr_inj, "pcs": gl_setup.p_casing,
                                        "dp_valve": gl_setup.dp_valve, "inj_md": gl_setup.fixed_md})
                        if DIL_ON:
                            ds_ = ss["diluent_setup"]
                            cur.update({"dil_pct": ds_.pct, "dil_rate": ds_.rate, "glr_dil": ds_.glr_inj})
                            if "dil_pct" in chosen and "dil_rate" in chosen:
                                st.error("Choose either the diluent rate or its percentage, not both.")
                        if GL_ON and "qgi" in chosen and "glr_inj" in chosen:
                            st.error("Choose either the gas lift gas injection rate or the GLR injected, not both.")
                        if JET_ON and ss.get("jet_equipment"):
                            cur.update({k_: ss["jet_equipment"][k_] for k_ in JET_VARS})
                        if HSP_ON and ss.get("hsp_equipment"):
                            cur.update({"hsp_speed": ss["hsp_equipment"]["speed"],
                                        "hsp_wear": ss["hsp_equipment"]["wear"]})
                        if PCP_ON and ss.get("pcp_equipment"):
                            cur.update({"pcp_speed": ss["pcp_equipment"]["speed"],
                                        "pcp_wear": ss["pcp_equipment"]["wear"]})
                        if SRP_ON and ss.get("srp_equipment"):
                            cur.update({k_: ss["srp_equipment"][k_] for k_ in SRP_VARS})
                        if FOAM_ON and ss.get("foam_setup"):
                            cur["surf"] = ss["foam_setup"]["conc"]
                        if ESP_ON and ss.get("esp_equipment"):
                            cur["esp_freq"] = ss["esp_equipment"]["freq"]
                            cur["esp_stages"] = ss["esp_equipment"]["stages"]
                        if surface is not None and wsurf.has_choke(surface.items):
                            cur["choke"] = round(next(i.d_in for i in surface.items if i.kind == "choke") * 64, 2)
                        if surface is not None and wsurf.has_pump(surface.items):
                            cur["pump_power"] = next(i.pump.power for i in surface.items if i.kind == "pump")
                        for v_ in chosen:
                            with st.container(border=True):
                                u_lab_ = unit_of(var_defs[v_]).strip(" ()")
                                st.markdown(f"**{var_defs[v_]}** - current "
                                            + (f"{cur.get(v_):.6g} {u_lab_}".strip() if cur.get(v_) is not None else "—"))
                                how = st.radio("Values", ["Enter values", "Generate values"], horizontal=True,
                                               key=f"sys_how_{v_}")
                                try:
                                    if how == "Enter values":
                                        vals = num_list(st, "Values" + unit_of(var_defs[v_]), f"sys_vals_{v_}", "",
                                                        "comma separated (up to 10)")
                                    else:
                                        cc = st.columns(4)
                                        c0 = float(cur.get(v_) or 1.0)
                                        first = num(cc[0], "First" + unit_of(var_defs[v_]), f"sys_first_{v_}",
                                                    round(c0 * 0.5, 4))
                                        last = num(cc[1], "Last" + unit_of(var_defs[v_]), f"sys_last_{v_}",
                                                   round(c0 * 1.5 if c0 else 1.0, 4))
                                        nval = num(cc[2], "Number", f"sys_n_{v_}", 5.0, 1.0, 10.0, 1.0, "%.0f")
                                        spacing = sel(cc[3], "Spacing", ["Linear", "Geometric"], f"sys_sp_{v_}")
                                        vals = list(wsys.generate_values(first, last, nval, spacing))
                                    if len(vals) > 10:
                                        st.warning("Only the first 10 values are used.")
                                        vals = vals[:10]
                                    if vals:
                                        sp_ = wu.split_label(var_defs[v_])
                                        shown_ = vals if sp_ is None else wu.convert(
                                            list(vals), sp_[1], sp_[2], wu.display_unit(sp_[1], sp_[2], ss["_units"]))
                                        st.caption("Values: " + ", ".join(f"{x:.4g}" for x in shown_))
                                        sens[v_] = vals
                                    else:
                                        st.caption("Enter at least one value.")
                                except ValueError as e:
                                    st.error(str(e))
                        n_case = int(np.prod([len(v) for v in sens.values()])) if sens else 1
                        st.caption(f"{n_case} case(s) will be calculated.")

                go = st.button("Calculate", type="primary")
                if ESP_ON:
                    def ipr_for_esp(ch):
                        try:
                            return rebuild_ipr(ch)
                        except ValueError:
                            return ipr_curve
                    esp_system_calc(go, fluid, eqd, base, corr, p1, p2, sens, ipr_for_esp, var_defs)
                elif JET_ON:
                    def ipr_for_jet(ch):
                        try:
                            return rebuild_ipr(ch)
                        except ValueError:
                            return ipr_curve
                    jet_system_calc(go, fluid, eqd, base, corr, p1, p2, sens, ipr_for_jet, var_defs)
                elif HSP_ON:
                    def ipr_for_hsp(ch):
                        try:
                            return rebuild_ipr(ch)
                        except ValueError:
                            return ipr_curve
                    hsp_system_calc(go, fluid, eqd, base, corr, p1, p2, sens, ipr_for_hsp, var_defs)
                elif PCP_ON:
                    def ipr_for_pcp(ch):
                        try:
                            return rebuild_ipr(ch)
                        except ValueError:
                            return ipr_curve
                    pcp_system_calc(go, fluid, eqd, base, corr, p1, p2, sens, ipr_for_pcp, var_defs)
                elif SRP_ON:
                    def ipr_for_srp(ch):
                        try:
                            return rebuild_ipr(ch)
                        except ValueError:
                            return ipr_curve
                    srp_system_calc(go, fluid, eqd, base, corr, p1, p2, sens, ipr_for_srp, var_defs)
                else:
                    sig = (kind, tuple(sorted(base.items())), corr, p1, p2, method, tuple(user_rates or []), lhi,
                           tuple((k_, tuple(v_)) for k_, v_ in sens.items()), fluid_note, ipr_curve.pr,
                           float(ipr_curve.aof), repr(surface), node, repr(gl_setup))
                    if go:
                        try:
                            combos = wsys.combinations(sens)
                            path_cache = {}

                            gl_extra = gl_setup.extra_md() if GL_ON else []
                            if DIL_ON:
                                gl_extra = list(gl_extra) + ss["diluent_setup"].extra_md()
                            if GL_ON and "inj_md" in sens:
                                gl_extra = list(gl_extra) + list(sens["inj_md"])

                            def path_for(tid):
                                if tid not in path_cache:
                                    top_, secs_, pts_ = eqd["downhole"]
                                    s2 = secs_.copy()
                                    if tid is not None:
                                        s2.loc[s2["Type"] == "Tubing", "Flow ID (in)"] = float(tid)
                                    smd_, stvd_ = eqd["survey"]
                                    path_cache[tid] = with_thermal(wvlp.build_flow_path(
                                        top_, s2, pts_, smd_, stvd_, gmd, gt, o.flow_type, gl_extra))
                                return path_cache[tid]
                            lift_ = None
                            if GL_ON:
                                lift_ = dataclasses.replace(gl_setup, annulus=wgls.annulus_geometry(
                                    path_for(None), eqd["downhole"][1]))

                            def ipr_for(ch):
                                if not ch:
                                    return ipr_curve
                                try:
                                    return rebuild_ipr(ch)
                                except ValueError:
                                    if set(ch) <= {"wc", "gor"}:   # no recipe: keep the IPR tab curve
                                        return ipr_curve
                                    raise
                            with st.spinner(f"Calculating {len(combos)} case(s)..."):
                                res, paths_ = wsys.solve_cases(path_for, fluid, kind, base, combos, ipr_for, corr, p1, p2,
                                                               eqd["heat_capacities"], method, 25, user_rates,
                                                               allow_left=(lhi == "Allow"), surface=surface, node=node,
                                                               lift=lift_,
                                                               diluent=ss["diluent_setup"] if DIL_ON else None)
                            for r_, c_ in zip(res, combos):
                                r_.case_label = c_
                            ss["sys_res"] = {"sig": sig, "res": res, "paths": paths_, "vars": list(sens), "kind": kind,
                                             "corr": corr, "p1": p1, "p2": p2, "labels": var_defs, "surface": surface,
                                             "lift": lift_}
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
                                    if r_.details.get("diluent"):
                                        d_.update(r_.details["diluent"])
                                    gl_ = r_.details.get("gas_lift")
                                    if gl_:
                                        d_.update({k_: v_ for k_, v_ in gl_.items() if not k_.startswith("_")
                                                   and k_ not in ("Gas lift method", "Injection depth (ft TVD)",
                                                                  "Temperature at injection (°F)")})
                                d_["Note"] = r_.note + ((" " + r_.details["gas_lift"]["_note"]) if r_.details.get(
                                    "gas_lift", {}).get("_note") else "")
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
                                    if out.get("lift") is not None and "Oil rate (STB/d)" in tab_ and len(names_) == 1:
                                        st.pyplot(sens_plot(tab_, names_, {n_: n_ for n_ in names_}, "Oil rate (STB/d)"),
                                                  clear_figure=True)
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
                                st.error("Stability flag: the solution is the left-hand intersection - the VLP crosses the IPR "
                                         "from above, so flow is unstable (heading / liquid loading).")
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
                            if r.q is not None and out.get("lift") is not None:
                                pth_ = out["paths"][r.params.get("tubing_id")]
                                det_, prof_ = wgls.details(pth_, fluid, r.params, r.q, out["lift"], out["corr"],
                                                           out["p1"], out["p2"], float(r.params["u"]),
                                                           eqd["heat_capacities"])
                                vt_ = det_.pop("_valves", None)
                                note_ = det_.pop("_note", "")
                                if note_:
                                    st.warning(note_)
                                st.markdown("**Solution**")
                                st.dataframe(solution_panel(kind, r, det_, {}), hide_index=True, width="stretch",
                                             height=560)
                                if vt_ is not None:
                                    st.markdown("**Gas lift valves at the solution** (multipoint)")
                                    st.dataframe(vt_.style.format(precision=3), hide_index=True, width="stretch")
                                with st.expander("Pressure and temperature profile at the solution (tubing and casing)"):
                                    st.dataframe(prof_.style.format(precision=3, na_rep="—"), hide_index=True,
                                                 width="stretch")
                            elif r.q is not None:
                                det_, reg_, prof_ = wsys.solution_details(out["paths"][r.params.get("tubing_id")], fluid, kind,
                                                                          r.params, r.q, out["corr"], out["p1"], out["p2"],
                                                                          eqd["heat_capacities"], surface=out.get("surface"),
                                                                          comp=comp_current())
                                surf_tab_ = det_.pop("_surface", None)
                                st.markdown("**Solution**")
                                st.dataframe(solution_panel(kind, r, det_, reg_), hide_index=True, width="stretch",
                                             height=560)
                                if surf_tab_ is not None:
                                    st.markdown("**Surface equipment at the solution** (wellhead to manifold)")
                                    st.dataframe(surf_tab_.style.format(precision=2), hide_index=True, width="stretch")
                                if hasattr(r.ipr_curve, "layer_table"):
                                    st.markdown("**Layer contributions at the solution**")
                                    st.dataframe(ipr2.layer_split(r.ipr_curve, r.pwf).style.format(precision=2),
                                                 hide_index=True, width="stretch")
                                with st.expander("Pressure and temperature profile at the solution"):
                                    st.dataframe(prof_.style.format(precision=3), hide_index=True, width="stretch")
                            st.download_button("Download IPR/VLP table (CSV)", rt_.to_csv(index=False).encode(),
                                               "system_analysis.csv")
            elif QLK:
                quicklook_mode(right, fluid, eqd, o, corr, p1, p2, gl_setup, ipr_curve)
            elif EQL:
                esp_quicklook_mode(right, fluid, eqd, o, corr, p1, p2, ss.get("esp_equipment"), ipr_curve)
            elif HQL:
                hsp_quicklook_mode(right, fluid, eqd, o, corr, p1, p2, ss.get("hsp_equipment"), ipr_curve)
            elif EQC:
                equilibrium_mode(right, fluid, eqd, o, base, corr, p1, p2, gl_setup, ipr_curve)
            elif mode == VLPC_MODE:
                vlp_curve_mode(right, kind, fluid, eqd, o, base, corr, p1, p2, surface)
            elif mode == CALC_MODES[1]:
                traverse_mode(right, kind, fluid, eqd, o, base, corr, p1, p2, surface,
                              ss.get("esp_equipment") if ESP_ON else None, gl_setup,
                              ss.get("jet_equipment") if JET_ON else None,
                              ss.get("pcp_equipment") if PCP_ON else None,
                              ss.get("hsp_equipment") if HSP_ON else None)
            else:
                vfp_mode(right, kind, fluid, eqd, o, base, corr, p1, p2, surface,
                         ss.get("esp_equipment") if ESP_ON else None, gl_setup,
                         ss.get("jet_equipment") if JET_ON else None,
                         ss.get("pcp_equipment") if PCP_ON else None,
                         ss.get("hsp_equipment") if HSP_ON else None,
                         ss.get("srp_equipment") if SRP_ON else None)


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


GL_SHOW_COLS = ["Valve", "Number of valves", "Type", "Trim", "Measured depth (ft)", "True vertical depth (ft)", "Tubing pressure (psia)",
                "Casing pressure (psia)", "Transfer pressure (psia)", "Temperature at valve (°F)",
                "Gas lift rate (Mscf/d)", "Port size (64ths in)", "R value", "Valve opening pressure (psia)",
                "Valve closing pressure (psia)", "Dome pressure at 60 °F (psig)", "Test rack opening pressure (psig)",
                "Opening CHP (psia)", "Closing CHP (psia)", "dP across valve (psi)", "Unloadable gradient (psi/ft)",
                "Required gas (Mscf/d)", "Note"]


def gaslift_tab(o, eqd, ipr_curve):
    """Design | Gas lift (continuous), new well or existing mandrels - after PROSPER."""
    st.caption("Continuous gas lift design for a new well or existing mandrels, after PROSPER Design | Gaslift and "
               "API RP 11V6: "
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
    gl_valve_db_panel()
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        st.markdown("**Design options**")
        c = st.columns(2)
        design_rate = sel(c[0], "Design rate", list(wgl.DESIGN_RATE), "gl_rate_mode", wgl.DESIGN_RATE[1],
                          help="Entered by user: the rate and lift gas are known. Max production: the maximum oil rate "
                               "of the performance curve within the gas available. Max revenue: the maximum of the "
                               "revenue function (oil and gas sales less water handling and lift-gas cost).")
        vkind = sel(c[1], "Valve type", list(wgl.VALVE_TYPES), "gl_vkind",
                    help="Casing sensitive (IPO): opened by the casing pressure; the surface casing pressure drops "
                         "for each valve. Tubing sensitive (fluid operated): opened by the tubing pressure; the "
                         "casing pressure stays at the injection pressure and each valve is spaced from the tubing "
                         "pressure it transfers at, Pt + % (Pc - Pt) (PROSPER 'Percentage Pcasing - Pwh').")
        kind_vs = {k: v for k, v in wgl.VALVES.items() if v.kind == vkind or (vkind == wgl.VALVE_TYPES[0] and
                                                                            not v.manufacturer)}
        mans = sorted({v.manufacturer for v in kind_vs.values() if v.manufacturer})
        c_v = st.columns([1, 1.6])
        man = sel(c_v[0], "Manufacturer", ["All"] + mans, "gl_valve_man", "All")
        names = [k for k, v in kind_vs.items() if man == "All" or v.manufacturer == man] or list(kind_vs)
        valve_name = sel(c_v[1], "Valve", names, "gl_valve",
                         help="Valves of the selected type from the valve database (above): R = port area / bellows "
                              "area for each port size.")
        vt_ = wgl.VALVES[valve_name]
        st.caption("Ports / R: " + ", ".join(f"{p_:g}: {float(vt_.r(p_)):.3f}" for p_ in vt_.ports))
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
        c = st.columns(2)
        unl_method = sel(c[0], "Unloading gas per valve", ["Transfer pressure", "Minimum tubing pressure"], "gl_unl_m",
                         help="Transfer pressure (PROSPER): the gas that lowers the tubing pressure at the valve to "
                              "Pt + transfer % (Pc - Pt) while the design liquid rate is lifted - this sets the port "
                              "sizes. Minimum tubing pressure: the gas rate giving the lowest tubing pressure at the valve.")
        ann_t = flag(c[1].checkbox, "Annulus gas at the flowing tubing temperature", "gl_ann_t", True,
                     help="PROSPER: the lift gas reaches the flowing tubing temperature, which sets the gas-column "
                          "weight in the annulus. Off: a straight line from the surface gas temperature to the "
                          "formation temperature.")
        spread = flag(st.checkbox, "Casing drop per valve at least the valve spread R (Pvo - Pt)", "gl_spread", True,
                      help="PROSPER 'All valves Pvo = gas pressure' (recommended for new wells): the casing pressure "
                           "drop to close each valve is the larger of the entered drop and the calculated spread. "
                           "Off: the entered drop only (API RP 11V6 examples).")
        well_kind = sel(st, "Well", ["New well", "Existing mandrels"], "gl_well",
                        help="Existing mandrels (PROSPER Design | Existing Mandrels): the valves go in the mandrels "
                             "listed - each unloading valve in the deepest mandrel the well can be unloaded to, the "
                             "injection point in the deepest mandrel at or above the optimum depth, dummies in the "
                             "rest. Enter every mandrel, including those with dummy valves.")
        mand_tuple = ()
        if well_kind == "Existing mandrels":
            if "gl_mandrels" not in ss:
                ss["gl_mandrels"] = _gl_mandrels_blank()
            st.button("Transfer from the Equipment tab's gas lift valves", key="gl_mand_tr",
                      on_click=_gl_mandrels_from_valves)
            if ss.get("_gl_mand_msg"):
                st.warning(ss["_gl_mand_msg"])
            med = track("gl_mandrels", st.data_editor(
                ss["gl_mandrels"], key=f"gl_mandrels_ed_{ss.get('_ver_gl_mandrels', 0)}", num_rows="dynamic",
                width="stretch", column_config={
                    "Measured depth (ft)": st.column_config.NumberColumn(format="%.2f"),
                    "dP casing (psi)": st.column_config.NumberColumn(
                        format="%g", help="Overrides the min CHP decrease per valve for a valve in this mandrel."),
                    "Maximum gas (MMscf/d)": st.column_config.NumberColumn(
                        format="%g", help="Overrides the maximum gas during unloading for a valve in this mandrel."),
                    "Enabled": st.column_config.CheckboxColumn()}))
            rows_m = []
            for _, r_ in med.iterrows():
                md_ = pd.to_numeric(r_.get("Measured depth (ft)"), errors="coerce")
                if pd.notna(md_) and md_ > 0 and (r_.get("Enabled") is not False):
                    rows_m.append((float(md_), float(pd.to_numeric(r_.get("dP casing (psi)"), errors="coerce")),
                                   float(pd.to_numeric(r_.get("Maximum gas (MMscf/d)"), errors="coerce"))))
            mand_tuple = tuple(sorted(rows_m))
            if not mand_tuple:
                st.info("Enter the mandrel depths (all mandrels, including those with dummy valves).")
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
        if vkind == wgl.VALVE_TYPES[1]:
            pct_cw = num(c[1], "Percentage Pcasing - Pwh (%)", "gl_pct_cw", 30.0, 0.0, 100.0,
                         help="Tubing-sensitive valves: each valve transfers when the tubing pressure at it is this "
                              "% of the way from the flowing tubing pressure to the casing pressure (at surface for "
                              "the first valve, from the unloading wellhead pressure to the kick-off pressure).")
            dpc = float(val("gl_dpc", 20.0))
        else:
            dpc = num(c[1], "Min CHP decrease per valve (psi)", "gl_dpc", 20.0, 0.0, 500.0)
            pct_cw = float(val("gl_pct_cw", 30.0))
        c = st.columns(2)
        margin_top = num(c[0], "Safety margin, first valve (psi)", "gl_m1", 50.0, 0.0, 500.0,
                         help="The first valve balances the load fluid against the kick-off casing pressure less this.")
        margin = num(c[1], "Transfer safety margin, deeper valves (psi)", "gl_m2", 50.0, 0.0, 500.0,
                     help="API RP 11V6: 20-50 psi. PROSPER uses 50 psi for every valve.")
        c = st.columns(2)
        safety_last = num(c[0], "Safety for closure of last unloading valve (psi)", "gl_slast", 0.0, 0.0, 500.0)
        min_sp = num(c[1], "Minimum valve spacing (ft)", "gl_minsp", 250.0, 10.0, 5000.0,
                     help="API RP 11V6: 90 ft (high PI, good data) to 500 ft (low PI, poor data).")
        c = st.columns(3)
        min_tr = num(c[0], "Minimum transfer dP (%)", "gl_mintr", 25.0, 0.0, 100.0,
                     disabled=use_ipr_unl and unl_method != "Transfer pressure",
                     help="Transfer pressure = Pt + this % of (Pc - Pt) at each valve; 5-25 % typical.")
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
                          valve=valve_name, economics=econ, unloading_gas=unl_method, annulus_t_flowing=ann_t,
                          mandrels=mand_tuple, valve_type=vkind, pct_casing_wh=pct_cw)
    if ipr_curve.fluid == "oil":            # the IPR at the design water cut (composite below the bubble point)
        try:
            ipr_curve = rebuild_ipr({"wc": wc / 100.0})
        except ValueError:
            pass
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
        if res.mandrels is not None:
            st.markdown("**Mandrels**")
            st.dataframe(res.mandrels.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
        tab_ = res.table()[GL_SHOW_COLS]
        st.dataframe(tab_.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
        if getattr(res.inp, "valve_type", "") == wgl.VALVE_TYPES[2]:
            st.caption("Proportional valves (PROSPER's columns): valve opening pressure = PTO, the tubing pressure "
                       "to open (Pt + transfer % of Pc - Pt); closing pressure at depth = PTO (1 - k) + Pc k with the "
                       "trim's k (R value column); dome at 60 °F from the closing pressure + the 220 psi spring; the "
                       "test rack column is the shop closing pressure (dome - spring). Port size = the trim's "
                       "equivalent port (Small 12, Medium 16, Large 20/64 - assumed).")
        elif getattr(res.inp, "valve_type", "") == wgl.VALVE_TYPES[1]:
            st.caption("Tubing-sensitive valves: the opening pressure is a tubing pressure (the transfer pressure); "
                       "the closing pressure is Pvo (1 - R) + Pc R.")
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

def igl_design_plot(res, inp, well, gas, pso_list):
    """Intermittent gas lift spacing plot: pressure against TVD with the casing lines, kill and unloading lines."""
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    v = res.valves
    dmax = float(v["Depth (ft)"].max())
    z = np.linspace(0.0, dmax, 60)
    for k, pso in enumerate(pso_list):
        pc = [wigl.casing_at_depth(gas, pso, d, well) for d in z]
        ax.plot(pc, z, color=wplt.SERIES[1], lw=1.6 if k == 0 else 0.9, ls="-" if k == 0 else "--",
                label="Casing opening pressure lines (Pso per valve)" if k == 0 else None)
    ax.plot(inp.p_wh + inp.g_u * z, z, color=wplt.SERIES[0], lw=1.6, label=f"Unloading gradient {inp.g_u:.3f} psi/ft")
    if inp.well_killed:
        d1 = float(v["Depth (ft)"].iloc[0])
        ax.plot([inp.p_wh, inp.p_wh + inp.g_kill * d1], [0, d1], color=wplt.SERIES[2], lw=1.2,
                label=f"Kill fluid {inp.g_kill:.3f} psi/ft")
    pt0, d0 = inp.p_wh + inp.g_u * float(v["Depth (ft)"].iloc[0]), float(v["Depth (ft)"].iloc[0])
    for k in range(1, len(v)):
        d1 = float(v["Depth (ft)"].iloc[k])
        ax.plot([pt0, pt0 + inp.g_s * (d1 - d0)], [d0, d1], color=wplt.SERIES[2], lw=0.8, ls=":")
        pt0, d0 = inp.p_wh + inp.g_u * d1, d1
    pl = res.cycle["Tubing load at valve (psig)"]
    ax.plot([res.cycle["Pressure on top of slug (psig)"], pl], [dmax - (pl - res.cycle["Pressure on top of slug (psig)"])
                                                                / inp.g_s, dmax],
            color=wplt.SERIES[3], lw=2.0, label="Starting slug (tubing load)")
    xmax = float(v["Opening pressure at depth (psig)"].max()) * 1.12
    for _, r in v.iterrows():
        ax.axhline(r["Depth (ft)"], color=wplt.INK2, lw=0.6, alpha=0.5)
        ax.scatter([r["Opening pressure at depth (psig)"]], [r["Depth (ft)"]], marker=">", s=46, zorder=6,
                   color="#2ca02c" if r["Role"] == "Operating" else wplt.INK)
        ax.text(xmax * 0.995, r["Depth (ft)"], f"Valve {int(r['Valve'])}  {r['Port (in)']} in", ha="right",
                va="bottom", fontsize=7.5)
    ax.set_xlim(0, xmax)
    ax.set_ylim(dmax * 1.03, 0)
    ax.set_xlabel("Pressure (psig)")
    ax.set_ylabel("True vertical depth (ft)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=7.5, loc="lower left")
    c = res.cycle
    ax.set_title(f"Intermittent gas lift: {c['Liquid rate (B/D)']:,.1f} STB/d, {c['Cycles per day']:.1f} cycles/day, "
                 f"{c['Gas per cycle (scf)']:,.0f} scf/cycle", fontsize=9.5)
    fig.tight_layout()
    return fig


def igl_sens_plot(sens):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(sens["Pvo - load (psi)"], sens["Liquid rate (B/D)"], marker="o", color=wplt.SERIES[0], lw=1.8,
            label="Liquid rate")
    ax.set_xlabel("Valve opening pressure - tubing load (psi)")
    ax.set_ylabel("Liquid rate (STB/d)")
    ax2 = ax.twinx()
    ax2.plot(sens["Pvo - load (psi)"], sens["Injection GLR (scf/bbl)"], marker="s", color=wplt.SERIES[1], lw=1.4,
             ls="--", label="Injection GLR")
    ax2.set_ylabel("Injection GLR (scf/STB)")
    ax.grid(alpha=0.3, ls="--")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def igl_spacing_plot(tab):
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    d = tab["Valve Depth (ft)"].to_numpy()
    ax.plot(tab["Spacing Pressure (psig)"], d, color="#2ca02c", marker="s", lw=1.8, label="Spacing pressure")
    ax.plot(tab["Closing Pressure (psig)"], d, color=wplt.SERIES[1], marker="+", lw=1.6, label="Closing pressure")
    ax.plot(tab["Injection Pressure (psig)"], d, color=wplt.SERIES[0], marker="o", lw=1.6, label="Injection pressure")
    for k in range(1, len(tab)):
        ax.annotate("", xy=(tab["Closing Pressure (psig)"].iloc[k], d[k]),
                    xytext=(tab["Spacing Pressure (psig)"].iloc[k], d[k]), arrowprops=dict(arrowstyle="->", lw=1.0))
        ax.plot([tab["Spacing Pressure (psig)"].iloc[k - 1], tab["Injection Pressure (psig)"].iloc[k] if k == 1 else
                 tab["Closing Pressure (psig)"].iloc[k]], [d[k - 1], d[k]], color=wplt.INK2, lw=0.7)
    ax.set_ylim(max(d) * 1.08, 0)
    ax.set_xlabel("Pressure (psig)")
    ax.set_ylabel("Depth (ft)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Valve spacing - constant surface closing pressure", fontsize=9.5)
    fig.tight_layout()
    return fig


def igl_optiflow_plot(tab, p_wh, g_unl):
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    d = tab["Depth (ft TVD)"].to_numpy()
    ax.plot(tab["Opening pressure at depth (psig)"], d, color=wplt.SERIES[0], marker="o", lw=1.6,
            label="Valve opening pressure at depth")
    ax.plot(tab["Tubing pressure max (psig)"], d, color=wplt.SERIES[1], marker="^", lw=1.2, ls="--",
            label="Maximum spacing line")
    ax.plot(tab["Tubing pressure min (psig)"], d, color="#2ca02c", marker="s", lw=1.2, ls="--",
            label="Minimum spacing line (tubing pressure)")
    ax.plot([p_wh, p_wh + g_unl * d[0]], [0, d[0]], color=wplt.INK2, lw=1.0, label="Unloading gradient")
    for k in range(1, len(d)):
        p0 = tab["Tubing pressure min (psig)"].iloc[k - 1]
        ax.plot([p0, p0 + g_unl * (d[k] - d[k - 1])], [d[k - 1], d[k]], color=wplt.INK2, lw=0.8)
    for x in d:
        ax.axhline(x, color=wplt.INK2, lw=0.5, alpha=0.5)
    ax.set_ylim(max(d) * 1.05, 0)
    ax.set_xlim(0, None)
    ax.set_xlabel("Pressure (psig)")
    ax.set_ylabel("True vertical depth (ft)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Valve spacing - Opti-Flow", fontsize=9.5)
    fig.tight_layout()
    return fig


def igl_system_plot(curve, sol):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(curve["Intermittent rate (STB/d)"], curve["Flowing BHP (psig)"], color=wplt.SERIES[0], lw=1.8,
            label="Intermittent 'VLP' (by starting slug)")
    ax.plot(curve["IPR rate (STB/d)"], curve["Flowing BHP (psig)"], color=wplt.SERIES[1], lw=1.8, label="IPR")
    if sol is not None:
        ax.scatter([sol["Daily Production Rate (STB/d)"]], [sol["Flowing BHP (psig)"]], s=60, zorder=6, color=wplt.INK)
    ax.set_xlabel("Average liquid rate (STB/d)")
    ax.set_ylabel("Pressure at the operating valve (psig)")
    ax.set_xlim(0, None)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def igl_prosper_tab(o, eqd, ipr_curve):
    """Design | Intermittent Gas Lift, PROSPER's model: White et al. (1963) slug velocity and fall back."""
    st.caption("PROSPER's intermittent model (Design | Intermittent Gas Lift): the slug above the operating valve "
               "(tubing liquid level to valve) is lifted at the slug velocity of White et al. (1963), "
               "vs = 1430 (Ap/At) sqrt((Pc/Pt)^2 - Pc/Pt), solved with the slug pressure Pt = Pwh + G S (1 + f vs^2/2gd); "
               "fall back FB = D / (1 + vs/vb); one cycle is the time for the produced slug to surface; gas per cycle "
               "is the tubing volume above the produced slug. No IPR is involved: the liquid level is an input.")
    g = 14.696
    gmd, gt = eqd["geothermal"]
    top_, secs_, pts_ = eqd["downhole"]
    path = vlp_path(eqd, "tubing", ())
    fluid, fluid_note = vlp_fluid(o, "oil", 6000.0, float(np.min(gt)) - 10, float(np.max(gt)) + 10)
    if fluid is None:
        st.warning(fluid_note)
        return
    seg_at = lambda md: int(np.clip(np.searchsorted(path.md, md) - 1, 0, len(path.d_h) - 1))
    t_id = float(path.d_h[seg_at(float(val("iglp_md", path.md[-1] * 0.8)) - 1.0)] * 12)
    tvd_at = lambda md: float(np.interp(md, path.md, path.tvd))
    temp = lambda d: float(np.interp(d, path.tvd, path.t_form))
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        st.markdown("**Input data**")
        c = st.columns(2)
        p_inj = num(c[0], "Surface injection pressure (psia)", "iglp_pinj", 1514.7, 50.0, 2e4)
        inj_md = num(c[1], "Injection depth (ft MD)", "iglp_md", float(round(path.md[-1] * 0.8)), float(path.md[0]) + 10,
                     float(path.md[-1]))
        c = st.columns(2)
        sg = num(c[0], "Gas lift gas gravity", "iglp_sg", float(val("gl_sg", 0.70)), 0.55, 2.0, None, "%.3f")
        pwh = num(c[1], "Wellhead pressure (psia)", "iglp_pwh", 264.7, 14.7, 5000.0)
        c = st.columns(2)
        port = num(c[0], "Valve port size (1/64 in)", "iglp_port", 24.0, 4.0, 128.0, None, "%.4g")
        wc = num(c[1], "Water cut (%)", "iglp_wc", float(val("gl_wc", 0.0)), 0.0, 100.0)
        c = st.columns(2)
        level = num(c[0], "Tubing liquid level (ft MD)", "iglp_level", float(round(path.md[-1] * 0.5)),
                    float(path.md[0]), float(path.md[-1]))
        gor = num(c[1], "Total GOR (scf/STB)", "iglp_gor", float(val("sys_gor", 300.0)), 0.0, 1e5,
                  help="Gas in solution in the slug (liquid density and viscosity).")
        vb_mode = sel(st, "Bubble velocity", ["PROSPER-calibrated", "Entered"], "iglp_vbm",
                      help="White et al. measured vb as a constant for a fluid and tubing size. PROSPER's values vary "
                           "with slug velocity, slug pressure and water cut; the calibrated form reproduces seven "
                           "PROSPER runs (4.052-in tubing) within 0.1 %; other tubing sizes are untested.")
        vb_in = num(st, "Bubble velocity (ft/s)", "iglp_vb", 2.3, 0.1, 20.0, None, "%.4g", disabled=vb_mode != "Entered")
        st.caption(f"Tubing ID at the valve {t_id:.3f} in. Liquid properties from the PVT at the bottom-hole temperature "
                   f"({float(path.t_form[-1]):.0f} °F) and the average slug pressure; Moody friction. {fluid_note}.")
        go = st.button("Calculate", type="primary", key="iglp_go")
        st.markdown("**Valves**")
        vmeth = sel(st, "Valve spacing", ["Constant surface closing pressure", "Opti-Flow"], "iglp_vmeth",
                    help="Constant surface closing pressure: PROSPER's procedure (Tutorial 24). Opti-Flow (Axelson; "
                         "Takacs, Gas Lift Manual 6.5.3): for wells with little information on potential - injection "
                         "always at the deepest possible valve, tubing pressure taken as a fraction of the gas "
                         "pressure at each valve.")
        c = st.columns(2)
        g_unl = num(c[0], "Unloading fluid gradient (psi/ft)", "iglp_gu", 0.46, 0.2, 0.7, None, "%.3f")
        max_md = num(c[1], "Maximum depth of injection (ft MD)", "iglp_maxmd", float(round(path.md[-1] * 0.8)),
                     float(path.md[0]) + 10, float(path.md[-1]))
        lg_ = wgl.LiftGas(sg, t_surface=temp(float(path.tvd[0])))
        if vmeth == "Constant surface closing pressure":
            c = st.columns(2)
            sf = num(c[0], "Spacing factor (psi/ft)", "iglp_sf", 0.1, 0.0, 0.5, None, "%.3f",
                     help="Slope of the spacing pressure line from the wellhead pressure; set by tubing size and rate.")
            psc = num(c[1], "Surface closing pressure (psia)", "iglp_psc", 814.7, 14.7, 2e4)
            gg_def = wigl.annulus_gas_gradient(lg_, p_inj - g, float(path.tvd[0]), tvd_at(max_md), temp)
            c = st.columns(2)
            gg_calc = flag(c[0].checkbox, "Calculate annulus gas gradient", "iglp_ggc", True,
                           help="Gas density at the average column pressure and temperature between the wellhead and "
                                "the maximum depth (PROSPER's tutorial value is about 1 % higher).")
            gg = gg_def if gg_calc else num(c[1], "Annulus gas gradient (psi/ft)", "iglp_gg", round(gg_def, 5), 0.001,
                                            0.2, None, "%.5f")
            if gg_calc:
                c[1].text_input("Annulus gas gradient (psi/ft)", f"{gg_def:.5f}", disabled=True, key="iglp_gg_show")
        else:
            c = st.columns(2)
            sbhp = num(c[0], "Static BHP (psia)", "iglp_sbhp", float(ipr_curve.pr) if ipr_curve is not None else 2000.0,
                       14.7, 2e4, help="Sets the static liquid level, Lperf - SBHP / unloading gradient (eq. 6.11). "
                                       "Default: the IPR reservoir pressure.")
            perf_md = num(c[1], "Perforation depth (ft MD)", "iglp_perf", float(path.md[-1]), float(path.md[0]) + 10,
                          float(path.md[-1]))
            c = st.columns(3)
            odrop = num(c[0], "Opening pressure below injection (psi)", "iglp_odrop", 100.0, 0.0, 1000.0,
                        help="Popen = Pinj - this (Takacs: 100).")
            tmarg = num(c[1], "Top valve margin (psi)", "iglp_tmarg", 50.0, 0.0, 500.0)
            ustep = num(c[2], "Unloading valve Pso step (psi)", "iglp_ustep", 20.0, 0.0, 200.0,
                        help="Valves above the static level open this much higher per valve (counted up from the "
                             "deepest) so the passing slug does not open them (Takacs Table 6-2: 20).")
            c = st.columns(4)
            a_lo = num(c[0], "Above static: min line (%)", "iglp_alo", 30.0, 1.0, 100.0)
            a_hi = num(c[1], "Above static: max line (%)", "iglp_ahi", 90.0, 1.0, 100.0)
            b_lo = num(c[2], "Below static: min line (%)", "iglp_blo", 55.0, 1.0, 100.0)
            b_hi = num(c[3], "Below static: max line (%)", "iglp_bhi", 85.0, 1.0, 100.0)
            wgl.register_valves(ss.get("_tables", {}).get("gl_valve_db", ss.get("gl_valve_db")))
            ovalve = sel(st, "Valve", list(wigl.IGL_VALVES) + list(wgl.VALVES), "iglp_ovalve",
                         help="R = port / bellows area for the settings; the port is the valve port size above.")
        go_v = st.button("Valves", key="iglp_valves")
        st.markdown("**System analysis**")
        st.caption("PROSPER Calculation | System for an intermittent well: the starting slug (and so the liquid "
                   "level) is found where the IPR rate at the slug pressure equals the intermittent rate. PROSPER "
                   "takes the slug pressure at the valve as the flowing BHP.")
        add_below = flag(st.checkbox, "Add the liquid column below the valve to the BHP", "iglp_below", False,
                         help="Off as in PROSPER. On: BHP at the IPR datum = slug pressure + liquid gradient x "
                              "(datum - valve depth).")
        go_s = st.button("System", key="iglp_sys", disabled=ipr_curve is None)
    lg = wgl.LiftGas(sg, t_surface=temp(float(path.tvd[0])))
    with right:
        if go_s:
            try:
                curve_ = ipr_curve
                if curve_.fluid == "oil":
                    try:
                        curve_ = rebuild_ipr({"wc": wc / 100.0})
                    except ValueError:
                        pass
                sol, crv = wigl.prosper_system(
                    fluid, lg, p_inj - g, pwh - g, float(path.tvd[0]), tvd_at(inj_md), port, t_id, wc, temp, lambda pwf: float(wgl.ipr_rate(curve_, pwf + g)), gor=gor,
                    rough_in=float(path.rough[seg_at(inj_md - 1.0)] * 12), t_fluid=float(path.t_form[-1]),
                    v_bubble=vb_in if vb_mode == "Entered" else None,
                    below_valve_grad=0.0, perf_tvd=float(path.tvd[-1]))
                if add_below and sol is not None:
                    gl_ = sol["Liquid gradient (psi/ft)"]
                    sol, crv = wigl.prosper_system(
                        fluid, lg, p_inj - g, pwh - g, float(path.tvd[0]), tvd_at(inj_md), port, t_id, wc, temp,
                        lambda pwf: float(wgl.ipr_rate(curve_, pwf + g)), gor=gor,
                        rough_in=float(path.rough[seg_at(inj_md - 1.0)] * 12), t_fluid=float(path.t_form[-1]),
                        v_bubble=vb_in if vb_mode == "Entered" else None, below_valve_grad=gl_,
                        perf_tvd=float(path.tvd[-1]))
                ss["iglp_sys_res"] = (sol, crv)
            except (ValueError, RuntimeError, ZeroDivisionError) as e:
                ss["iglp_sys_res"] = str(e)
        sr = ss.get("iglp_sys_res")
        if isinstance(sr, str):
            st.error(sr)
        elif sr is not None:
            sol, crv = sr
            st.markdown("**System analysis**")
            if sol is None:
                st.warning("No solution: the intermittent rate does not meet the IPR for any slug length (the "
                           "injection pressure may not lift a slug heavy enough, or the well flows more than the "
                           "intermittent cycle can remove).")
            else:
                keys_ = ["Daily Production Rate (STB/d)", "Flowing BHP (psig)", "Liquid Level (ft below surface)",
                         "Downhole Injection Pressure (psig)", "Gas Injection Rate (MMscf/day)",
                         "Gas Injection Rate, in situ (MMcf/day)", "Slug Velocity (ft/s)", "Bubble Velocity (ft/s)",
                         "Fall Back (ft)", "Produced Slug Length (ft)", "Produced Slug Volume (bbl)",
                         "Pcasing / Pslug", "Produced/Accumulated Slug Length", "Starting Slug Length / Valve Depth",
                         "Time To Surface (s)", "Cycle Frequency (1/hour)"]
                rows_ = [(k_.rsplit(" (", 1)[0], sol[k_], k_.rsplit(" (", 1)[1][:-1] if k_.endswith(")") else "")
                         for k_ in keys_ if k_ in sol]
                st.dataframe(pd.DataFrame(rows_, columns=["Result", "Value", "Units"]).style.format(
                    {"Value": "{:,.6g}"}), hide_index=True, width="stretch")
            if sol is not None and sol.get("Second solution (longer slug)"):
                st.caption("The intermittent curve crosses the IPR a second time at a longer slug (higher BHP, lower "
                           "rate); as PROSPER, the shorter-slug solution is reported.")
            if len(crv):
                st.pyplot(igl_system_plot(crv, sol), clear_figure=True)
        if go:
            try:
                r = wigl.prosper_intermittent(
                    fluid, lg, p_inj - g, pwh - g, float(path.tvd[0]), tvd_at(inj_md), tvd_at(level), port, t_id, wc,
                    temp, gor=gor, rough_in=float(path.rough[seg_at(inj_md - 1.0)] * 12), t_fluid=float(path.t_form[-1]),
                    v_bubble=vb_in if vb_mode == "Entered" else None)
                ss["iglp_res"] = r
            except (ValueError, RuntimeError, ZeroDivisionError) as e:
                ss["iglp_res"] = {"note": str(e)}
        r = ss.get("iglp_res")
        if r is not None:
            st.markdown("**Results**")
            if "note" in r:
                st.error(r["note"])
            rows_ = [(k_.rsplit(" (", 1)[0], v_, k_.rsplit(" (", 1)[1][:-1] if k_.endswith(")") else "")
                     for k_, v_ in r.items() if k_ != "note"]
            st.dataframe(pd.DataFrame(rows_, columns=["Result", "Value", "Units"]).style.format(
                {"Value": "{:,.6g}"}), hide_index=True, width="stretch")
        if go_v:
            try:
                if vmeth == "Constant surface closing pressure":
                    ss["iglp_vtab"] = wigl.constant_psc_spacing(p_inj - g, pwh - g, psc - g, g_unl, sf, gg, max_md,
                                                                  float(path.md[0]))
                else:
                    vt = wigl.IGL_VALVES.get(ovalve) or wgl.VALVES[ovalve]
                    tab_, lst_ = wigl.optiflow_spacing(
                        lg, p_inj - g, pwh - g, g_unl, sbhp - g, tvd_at(perf_md), tvd_at(max_md), temp,
                        wh_tvd=float(path.tvd[0]), open_drop=odrop, top_margin=tmarg,
                        above=(a_lo / 100, a_hi / 100), below=(b_lo / 100, b_hi / 100), unl_step=ustep, valve=vt,
                        port_64=int(round(port)))
                    tab_.insert(2, "Depth (ft MD)", [float(np.interp(x, path.tvd, path.md)) for x in tab_["Depth (ft)"]])
                    ss["iglp_vtab"] = ("optiflow", tab_.rename(columns={"Depth (ft)": "Depth (ft TVD)"}), lst_)
            except ValueError as e:
                ss["iglp_vtab"] = str(e)
        vt_ = ss.get("iglp_vtab")
        if isinstance(vt_, tuple):
            st.markdown("**Valve spacing - Opti-Flow**")
            st.dataframe(vt_[1].style.format(precision=2), hide_index=True, width="stretch")
            st.caption(f"Static liquid level {vt_[2]:,.0f} ft TVD. Spacing lines: gas columns from "
                       f"{a_lo:.0f} / {a_hi:.0f} % of the surface opening pressure above the static level and "
                       f"{b_lo:.0f} / {b_hi:.0f} % below; tubing pressure at each valve from the minimum line.")
            st.pyplot(igl_optiflow_plot(vt_[1], pwh - g, g_unl), clear_figure=True)
        elif isinstance(vt_, str):
            st.error(vt_)
        elif vt_ is not None:
            st.markdown("**Valve spacing - constant surface closing pressure**")
            st.dataframe(vt_.style.format(precision=2), hide_index=True, width="stretch")
            st.caption(f"Deepest valve {vt_['Valve Depth (ft)'].iloc[-1]:,.1f} ft: use it as the injection depth.")
            st.pyplot(igl_spacing_plot(vt_), clear_figure=True)


def igl_tab(o, eqd, ipr_curve):
    """Design | Intermittent gas lift - Brown, The Technology of Artificial Lift Methods Vol. 2a, section 3.43."""
    st.caption("Intermittent gas lift after Brown (Vol. 2a, 3.43): valve spacing with unbalanced pressure valves and a "
               "surface opening pressure drop per valve, the operating valve port from the gas needed per cycle and "
               "the casing pressure drop that supplies it (choke or time-cycle control), valve settings, and the "
               "production from Brown's weighted-average flowing BHP (slug travel, after-flow and build-up periods, "
               "with or without a standing valve, fallback per 1000 ft) with the IPR for the feed-in. Brown: consider "
               "continuous lift above about 200 B/D in 2-in tubing (300 in 3-in, 400-600 with chambers).")
    gmd, gt = eqd["geothermal"]
    top_, secs_, pts_ = eqd["downhole"]
    tub = secs_[secs_["Type"] == "Tubing"]
    if not len(tub):
        st.info("Intermittent gas lift needs tubing in the Equipment Data.")
        return
    t_od, c_id = tub["Tubing OD (in)"].iloc[-1], tub["Casing ID (in)"].iloc[-1]
    if not (pd.notna(t_od) and pd.notna(c_id) and c_id > t_od):
        st.info("Enter the tubing OD and casing ID of the tubing sections (Equipment Data): the annulus stores the "
                "gas for each cycle.")
        return
    path = vlp_path(eqd, "tubing", ())
    t_id = float(path.d_h[-1] * 12)
    perf_tvd = float(path.tvd[-1])
    bottom_tub = float(tub["MD bottom (ft)"].max())
    fluid, fluid_note = vlp_fluid(o, "oil", max(float(ipr_curve.pr), 5000.0) * 1.15, float(np.min(gt)) - 10,
                                  float(np.max(gt)) + 10)
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        st.markdown("**Design options**")
        c = st.columns(2)
        control = sel(c[0], "Surface control", ["Time cycle", "Choke"], "igl_ctrl",
                      help="Time cycle (intermitter): half of the casing drop that supplies the gas is put in the valve "
                           "spread. Choke: the whole drop - a larger port.")
        spacing = sel(c[1], "Spacing", ["Graphical (gas column)", "Analytical (Brown 3.4354)"], "igl_spacing",
                      help="Graphical: the load line from each valve meets the next valve's casing line with the gas "
                           "column. Analytical: Brown eq. 3.451-3.455, gas column neglected, the pressure of the valve "
                           "above used for the next one.")
        c = st.columns(2)
        wgl.register_valves(ss.get("_tables", {}).get("gl_valve_db", ss.get("gl_valve_db")))
        valve = sel(c[0], "Valve", list(wigl.IGL_VALVES) + list(wgl.VALVES), "igl_valve",
                    help="Unbalanced pressure (casing) valves; Brown Table 3.488 R values.")
        vt = wigl.IGL_VALVES.get(valve) or wgl.VALVES[valve]
        ports = list(vt.ports)
        unl_port = sel(c[1], "Unloading valve port", ports, "igl_uport", 20 if 20 in ports else ports[0],
                       format_func=lambda p_: f"{wigl.port_label(p_)} in")
        c = st.columns(2)
        op_auto = sel(c[0], "Operating valve port", ["From gas requirement"] + ports, "igl_oport",
                      format_func=lambda p_: p_ if isinstance(p_, str) else f"{wigl.port_label(p_)} in")
        load = sel(c[1], "Tubing load", ["Half reservoir pressure", "Pvo - dP"], "igl_load",
                   help="Brown: a design load of half the static BHP, or the valve opening pressure less 150-300 psi.")
        c = st.columns(2)
        killed = flag(c[0].checkbox, "Well loaded to the surface (killed)", "igl_killed", True,
                      help="Off: the top valve is placed at the static fluid level.")
        sv = flag(c[1].checkbox, "Standing valve", "igl_sv", True,
                  help="Recommended for intermittent installations (Brown): the formation does not see the gas "
                       "pressure while the slug is lifted.")
        dome_hp = flag(st.checkbox, "Dome correction > 1200 psig (Winkler-Eads)", "igl_dome_hp", True)
        st.markdown("**Pressures and gradients**")
        c = st.columns(2)
        p_ko = num(c[0], "Available injection pressure (psia)", "igl_pko", float(val("gl_pko", 1064.7)), 50.0, 2e4)
        pwh = num(c[1], "Wellhead pressure on top of slug (psia)", "igl_pwh", 94.7, 14.7, 5000.0)
        c = st.columns(2)
        m_top = num(c[0], "Top valve margin below available (psi)", "igl_mtop", 50.0, 0.0, 500.0)
        dpso = num(c[1], "Surface opening drop per valve (psi)", "igl_dpso", 25.0, 0.0, 200.0,
                   help="Brown recommends 10-25 psi.")
        c = st.columns(2)
        op_drop = num(c[0], "Extra drop at operating valve (psi)", "igl_opdrop", 50.0, 0.0, 300.0,
                      help="25-100 psi flags the operating valve and keeps the upper valves closed (Brown).")
        dp_load = num(c[1], "Pvo - tubing load (psi)", "igl_dpl", 200.0, 50.0, 600.0, disabled=load != "Pvo - dP",
                      help="Brown: start with 200-300; 150-170 gives the lowest GLR, 300-400 more liquid.")
        gs_pvt = np.nan
        if fluid is not None:
            try:
                st_ = fluid.state(np.array([max(float(ipr_curve.pr) / 2, 100.0)]), np.array([float(np.mean(gt))]),
                                  {"q": np.array([100.0]), "wc": np.array([float(val("igl_wc", 0.0)) / 100]),
                                   "gor": np.array([0.0])})
                gs_pvt = float(st_["rho_l"][0]) / 144.0
            except Exception:
                gs_pvt = np.nan
        c = st.columns(2)
        g_s = num(c[0], "Well fluid gradient (psi/ft)", "igl_gs", round(gs_pvt, 3) if np.isfinite(gs_pvt) else 0.40,
                  0.2, 0.6, None, "%.3f", help="Dead liquid at the design water cut from the PVT by default.")
        g_kill = num(c[1], "Kill fluid gradient (psi/ft)", "igl_gkill", 0.40, 0.2, 0.7, None, "%.3f",
                     disabled=not killed)
        c = st.columns(2)
        gu_default = wigl.unloading_gradient_rate(200.0, t_id)
        g_u = num(c[0], "Unloading gradient (psi/ft)", "igl_gu", round(gu_default, 3), 0.005, 0.3, None, "%.3f",
                  help="Brown: minimum 0.04 psi/ft in 2-in and 2.5-in tubing; Otis Fig 3F gives 0.064 at 200 B/D "
                       "in 2-in. Default: Fig 3F at 200 B/D scaled to this tubing.")
        wc = num(c[1], "Water cut (%)", "igl_wc", float(val("gl_wc", 0.0)), 0.0, 99.0)
        st.markdown("**Slug and cycle**")
        c = st.columns(2)
        fb = num(c[0], "Fallback (% per 1000 ft)", "igl_fb", 7.0, 0.0, 15.0, None, "%.3g",
                 help="Brown: 5-7 % per 1000 ft in 2-in tubing (3-5 % for chambers); field data about 10 %.")
        top_allow = num(c[1], "Gas on top of slug above Pwh (psi)", "igl_topa", 10.0, 0.0, 200.0)
        c = st.columns(2)
        t1f = num(c[0], "Slug travel, Time I (min/1000 ft)", "igl_t1", 1.5, 0.3, 5.0, None, "%.3g",
                  help="Brown, 2-in tubing: 1.5 min per 1000 ft (about 1000 ft/min slug velocity, valve open).")
        t2f = num(c[1], "After surfacing, Time II (min/1000 ft)", "igl_t2", 0.5, 0.0, 5.0, None, "%.3g")
        max_md = num(st, "Operating valve depth (ft MD)", "igl_maxmd", float(round(bottom_tub - 30.0)), 100.0,
                     float(path.md[-1]), help="The deepest valve, above the packer.")
        st.markdown("**Lift gas**")
        c = st.columns(5)
        lg = wgl.LiftGas(num(c[0], "Gravity", "igl_sg", float(val("gl_sg", 0.70)), 0.55, 2.0, None, "%.3f"),
                         num(c[1], "CO2 (%)", "igl_co2", 0.0, 0.0, 100.0), num(c[2], "N2 (%)", "igl_n2", 0.0, 0.0, 100.0),
                         num(c[3], "H2S (%)", "igl_h2s", 0.0, 0.0, 100.0),
                         num(c[4], "Surface T (°F)", "igl_tg", float(val("gl_tg", 80.0)), 0.0, 300.0))
        st.caption(f"Tubing ID {t_id:.3f} in, OD {t_od:.3f} in, casing ID {c_id:.3f} in. IPR: "
                   f"{ss.get('ipr_meta', {}).get('model', '')} (Pr {ipr_curve.pr:,.0f} psia, datum {path.md[-1]:,.0f} "
                   f"ft MD). Gradient from PVT: {gs_pvt:.3f} psi/ft.")
        go = st.button("Design", type="primary", key="igl_go")
    if ipr_curve.fluid == "oil":
        try:
            ipr_curve = rebuild_ipr({"wc": wc / 100.0})
        except ValueError:
            pass
    g = 14.696
    max_tvd = float(np.interp(max_md, path.md, path.tvd))
    inp = wigl.IGLInput(p_ko=p_ko - g, top_margin=m_top, p_wh=pwh - g, pso_drop=dpso, op_drop=op_drop, g_s=g_s,
                        g_kill=g_kill, g_u=g_u, well_killed=killed, spacing=spacing, max_depth=max_tvd, valve=valve,
                        control=control, unloading_port=int(unl_port),
                        operating_port=None if isinstance(op_auto, str) else int(op_auto), load=load,
                        dp_load=dp_load, fallback=fb, top_allowance=top_allow, t1_factor=t1f, t2_factor=t2f,
                        standing_valve=sv, high_pressure_dome=dome_hp)
    curve = ipr_curve
    temp = lambda d: float(np.interp(d, path.tvd, path.t_form))
    well = wigl.IGLWell(t_id, float(t_od), float(c_id), perf_tvd, float(curve.pr) - g,
                        lambda pwf: float(wgl.ipr_rate(curve, pwf + g)), temp, t_surface=lg.t_surface)
    sig = (repr(inp), repr(lg), t_id, float(t_od), float(c_id), perf_tvd, float(curve.pr), float(curve.aof))
    with right:
        if go:
            try:
                with st.spinner("Designing..."):
                    res = wigl.design(inp, lg, well)
                    sens = wigl.load_sensitivity(inp, lg, well, res)
                ss["igl_res"] = {"sig": sig, "res": res, "sens": sens, "err": None,
                                 "inp": inp, "well": well, "gas": lg}
            except (ValueError, RuntimeError, ZeroDivisionError) as e:
                ss["igl_res"] = {"sig": sig, "res": None, "err": str(e)}
        out = ss.get("igl_res")
        if out is None:
            st.info("Set the design options and press Design.")
            return
        if out["err"]:
            st.error(f"Design failed: {out['err']}")
            return
        if out["sig"] != sig:
            st.info("Inputs have changed since the last design - press Design to update.")
        res = out["res"]
        c_ = res.cycle
        st.markdown("**Design results**")
        summ = pd.DataFrame([
            ("Liquid rate", c_["Liquid rate (B/D)"], "STB/d"),
            ("Oil rate", c_["Liquid rate (B/D)"] * (1 - wc / 100.0), "STB/d"),
            ("Cycle time", c_["Cycle time (min)"], "min"),
            ("Cycles per day", c_["Cycles per day"], ""),
            ("Recovered per cycle", c_["Recovered per cycle (bbl)"], "bbl"),
            ("Gas per cycle", c_["Gas per cycle (scf)"], "scf"),
            ("Injection gas", c_["Injection gas (Mscf/d)"] / 1000.0, "MMscf/d"),
            ("Injection GLR", c_["Injection GLR (scf/bbl)"], "scf/STB"),
            ("Weighted-average flowing BHP", c_["Weighted-average flowing BHP (psig)"] + g, "psia"),
            ("Operating valve", f"{res.valves['Depth (ft)'].iloc[-1]:,.0f} ft TVD, port "
                                f"{res.gas['Operating port']} in", ""),
        ], columns=["Result", "Value", "Units"])
        summ["Value"] = [f"{v_:,.4g}" if isinstance(v_, (float, np.floating)) else v_ for v_ in summ["Value"]]
        st.dataframe(summ, hide_index=True, width="stretch")
        for m_ in res.notes:
            st.warning(m_)
        vt_ = res.valves.copy()
        vt_.insert(2, "Depth (ft MD)", [float(np.interp(d_, path.tvd, path.md)) for d_ in vt_["Depth (ft)"]])
        vt_ = vt_.rename(columns={"Depth (ft)": "Depth (ft TVD)"})
        st.dataframe(vt_.style.format(precision=2), hide_index=True, width="stretch")
        st.download_button("Download valve table (CSV)", vt_.to_csv(index=False).encode(), "intermittent_valves.csv")
        t1, t2, t3, t4 = st.tabs(["Design plot", "Cycle", "Gas and port", "Load sensitivity"])
        psos = list(res.valves["Surface opening pressure (psig)"])
        with t1:
            st.pyplot(igl_design_plot(res, out["inp"], out["well"], out["gas"], psos), clear_figure=True)
        with t2:
            st.caption("Brown's weighted-average flowing BHP: Time I slug travel (valve open), Time II after the slug "
                       "surfaces (minimum pressure), Time III build-up from the minimum back to the tubing load. "
                       "Pressures at the operating valve; the BHP adds the liquid below the valve.")
            rows_ = [(k_, v_) for k_, v_ in c_.items()]
            st.dataframe(pd.DataFrame(rows_, columns=["Result", "Value"]).style.format(precision=4, na_rep="—"),
                         hide_index=True, width="stretch")
        with t3:
            st.dataframe(pd.DataFrame([(k_, v_ if isinstance(v_, str) else f"{v_:,.4g}") for k_, v_ in
                                       res.gas.items()], columns=["Result", "Value"]),
                         hide_index=True, width="stretch")
            st.caption("Gas per cycle: Brown eq. 3.439-3.446 (the tubing above the valve less the slug left at "
                       "the surface, at the average pressure under the slug). The annulus above the valve supplies it "
                       "by a casing pressure drop; under choke control the whole drop is put in the valve spread, "
                       "under time-cycle control half. Port: R = spread / (Pvo - Pt), the largest port not above.")
        with t4:
            sens = out["sens"]
            if len(sens):
                st.pyplot(igl_sens_plot(sens), clear_figure=True)
                st.dataframe(sens.style.format(precision=2), hide_index=True, width="stretch")
                st.caption("Same valve and injection pressure, tubing load varied. Brown: more Pvo - load gives more "
                           "liquid at a higher GLR; 150-170 psi gives the lowest GLR.")


def ct_design_plot(res):
    """Design | Coiled tubing gas lift - Plot: load fluid, kick-off and operating gas pressures, flowing gradient."""
    pr = res.profile
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    tv = pr["True vertical depth (ft)"]
    ax.plot(pr["Flowing tubing pressure (psia)"], tv, color=wplt.SERIES[0], lw=1.6, label="Flowing gradient (design)")
    ax.plot(pr["Lift-gas pressure, operating (psia)"], tv, color=wplt.SERIES[1], lw=1.4, label="Lift gas, operating")
    ax.plot(pr["Lift-gas pressure, kick-off (psia)"], tv, color=wplt.SERIES[3], lw=1.0, ls="--",
            label="Lift gas, kick-off")
    ax.plot(pr["Load fluid (psia)"], tv, color=wplt.SERIES[2], lw=1.2, ls=":", label="Load fluid")
    ax.axhline(res.kickoff_tvd, color=wplt.SERIES[3], lw=0.8, alpha=0.6)
    ax.axhline(res.max_tvd, color=wplt.SERIES[1], lw=0.8, alpha=0.6)
    ax.annotate(f"Kick-off {res.kickoff_md:,.0f} ft MD", (ax.get_xlim()[0], res.kickoff_tvd), fontsize=8,
                xytext=(4, -10), textcoords="offset points")
    ax.annotate(f"Maximum operating depth {res.max_md:,.0f} ft MD", (ax.get_xlim()[0], res.max_tvd), fontsize=8,
                xytext=(4, -10), textcoords="offset points")
    ax.set_xlim(left=0, right=max(float(pr["Lift-gas pressure, kick-off (psia)"].max()),
                                  float(pr["Lift-gas pressure, operating (psia)"].max())) * 1.1)
    ax.invert_yaxis()
    ax.set_xlabel("Pressure (psia)")
    ax.set_ylabel("True vertical depth (ft)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Coiled tubing gas lift design", fontsize=10)
    fig.tight_layout()
    return fig


def ct_design_tab(o, eqd, ipr_curve):
    """Design | Coiled Tubing Gas Lift - after PROSPER (User Guide 2.13.5, Tutorial 25)."""
    st.caption("Coiled tubing gas lift design after PROSPER Design | Coiled Tubing Gas Lift: the lift gas goes down a "
               "coiled tubing and the well produces up the coiled tubing - tubing annulus. No unloading valves: the "
               "design rate (entered, or from the performance curve), the deepest operating injection depth, the "
               "orifice at the coiled tubing end (Thornhill-Craver) and the depth the kick-off pressure unloads the "
               "load fluid to.")
    gmd, gt = eqd["geothermal"]
    fluid, fluid_note = vlp_fluid(o, "oil", max(float(ipr_curve.pr), 5000.0) * 1.15, float(np.min(gt)) - 10,
                                  float(np.max(gt)) + 10)
    if fluid is None:
        st.warning(fluid_note)
        return
    top_, secs_, pts_ = eqd["downhole"]
    tub = secs_[secs_["Type"] == "Tubing"]
    bottom_tub = float(tub["MD bottom (ft)"].max()) if len(tub) else float(secs_["MD bottom (ft)"].max())
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        st.markdown("**Options**")
        c = st.columns(2)
        design_rate = sel(c[0], "Design rate method", list(wgl.DESIGN_RATE), "ct_rate_mode", wgl.DESIGN_RATE[1])
        orif = sel(c[1], "Orifice sizing on", list(wgl.ORIFICE_SIZING), "ct_orif",
                   help="Calculated dP at orifice: the whole 64th below the size that passes the design gas at the "
                        "calculated dP, and the gas it passes (PROSPER). Min dP: the size above, passing the design "
                        "gas at a smaller dP.")
        c = st.columns(2)
        check_ipr = flag(c[0].checkbox, "Check rate conformance with IPR", "ct_chk_ipr", True)
        use_ipr_unl = flag(c[1].checkbox, "Use IPR for unloading", "ct_ipr_unl", True)
        st.markdown("**Input parameters**")
        c = st.columns(2)
        q_design = None
        if design_rate == wgl.DESIGN_RATE[0]:
            q_design = num(c[0], "Design liquid rate (STB/d)", "ct_q", 1000.0, 1.0, 1e6)
        q_max = num(c[1] if q_design is not None else c[0], "Maximum liquid rate (STB/d)", "ct_qmax",
                    float(round(ipr_curve.aof)), 1.0, 1e6)
        c = st.columns(2)
        qgi_max = num(c[0], "Maximum gas available (MMscf/d)", "ct_qgi", 5.0, 0.001, 100.0, None, "%.4f")
        qgi_unl = num(c[1], "Maximum gas during unloading (MMscf/d)", "ct_qgi_unl", float(qgi_max), 0.001, 100.0,
                      None, "%.4f")
        c = st.columns(2)
        top_p = num(c[0], "Flowing top node pressure (psia)", "ct_pwh", float(val("sys_whp", 200.0)), 14.7, 1e4)
        top_pu = num(c[1], "Unloading top node pressure (psia)", "ct_pwh_unl", float(top_p), 14.7, 1e4)
        c = st.columns(2)
        p_inj = num(c[0], "Operating injection pressure (psia)", "ct_pinj", 1214.7, 50.0, 2e4,
                    help="Gas pressure available at the coiled tubing head.")
        p_ko = num(c[1], "Kick-off injection pressure (psia)", "ct_pko", float(p_inj), 50.0, 2e4)
        c = st.columns(2)
        dp_valve = num(c[0], "Desired dP across valve (psi)", "ct_dpv", 50.0, 1.0, 2000.0)
        max_md = num(c[1], "Maximum depth of injection (ft MD)", "ct_maxmd", float(round(bottom_tub - 30.0)), 100.0,
                     float(secs_["MD bottom (ft)"].max()), help="Above the production packer.")
        c = st.columns(3)
        wc = num(c[0], "Water cut (%)", "ct_wc", float(val("sys_wc", 0.0)), 0.0, 99.0)
        gs = num(c[1], "Static gradient of load fluid (psi/ft)", "ct_gs", 0.46, 0.1, 1.0, None, "%.3f")
        gor = num(c[2], "Total GOR (scf/STB)", "ct_gor", float(val("sys_gor", 300.0)), 0.0, 1e6)
        econ = wgl.Economics()
        if design_rate == wgl.DESIGN_RATE[2]:
            st.markdown("**Economics**")
            c = st.columns(4)
            econ = wgl.Economics(num(c[0], "Oil price ($/STB)", "ct_po", 70.0, 0.0, 1e4),
                                 num(c[1], "Gas price ($/Mscf)", "ct_pg", 3.0, 0.0, 1e4),
                                 num(c[2], "Water cost ($/STB)", "ct_cw", 1.0, 0.0, 1e4),
                                 num(c[3], "Lift gas cost ($/Mscf)", "ct_cg", 0.5, 0.0, 1e4))
        lg = wgl.LiftGas(float(val("ctq_sg", 0.7)), float(val("ctq_co2", 0.0)), float(val("ctq_n2", 0.0)),
                         float(val("ctq_h2s", 0.0)))
        ct = wct.CoiledTubing(float(val("ctq_id", 0.8)), float(val("ctq_wall", 0.2)), float(val("ctq_ein", 0.0006)),
                              float(val("ctq_eout", 0.0006)), val("ctq_fric", wct.FRICTION[0]) == wct.FRICTION[1])
        st.caption(f"Lift gas and coiled tubing from the Equipment tab (Coiled tubing data): gravity {lg.sg:g}, "
                   f"ID {ct.id_in:g} in, OD {ct.od_in:g} in, "
                   + ("friction loss in the coiled tubing." if ct.friction else "no friction loss in the coiled tubing."))
        st.markdown("**Tubing correlation**")
        ms = ss.get("vlp_matches", {}).get("ms", {})
        sel_c = ss.get("vlp_selected")
        corr_opts = [c_ for c_ in wvlp.VLP_CORRELATIONS if c_ not in wvlp.LOWER_BOUNDS]
        corr = sel(st, "Vertical lift correlation", corr_opts, "ct_corr",
                   sel_c if sel_c in corr_opts else "hagedorn_brown", format_func=VLP_LABELS.get)
        m_fit = ms.get(corr)
        use_fit = flag(st.toggle, "Use matched parameters", "ct_use_fit", disabled=m_fit is None)
        c2 = st.columns(2)
        if m_fit is not None and use_fit:
            p1, p2 = float(m_fit.p1), float(m_fit.p2)
            c2[0].text_input("Parameter 1 (gravity)", f"{p1:.4f}", disabled=True, key="ct_p1_show")
            c2[1].text_input("Parameter 2 (friction)", f"{p2:.4f}", disabled=True, key="ct_p2_show")
        else:
            p1 = num(c2[0], "Parameter 1 (gravity)", f"ct_p1_{corr}", 1.0, 0.3, 3.0, None, "%.4f")
            p2 = num(c2[1], "Parameter 2 (friction)", f"ct_p2_{corr}", 1.0, 0.01, 10.0, None, "%.4f")
        u = float(val("eq_u", eqd.get("u_geo", 8.0)))
        st.caption(f"PVT: {fluid_note}. IPR: {ss.get('ipr_meta', {}).get('model', '')} (Pr {ipr_curve.pr:,.0f} psia). "
                   f"U = {u:.4g} BTU/h/ft2/F.")
        go = st.button("Design", type="primary", key="ct_go")
    inp = wct.CTDesignInput(design_rate=design_rate, q_design=q_design or 1000.0, q_liquid_max=q_max, qgi_max=qgi_max,
                            qgi_unload_max=qgi_unl, top_p=top_p, top_p_unload=top_pu, p_inj=p_inj, p_ko=p_ko,
                            dp_valve=dp_valve, max_md=max_md, gor=gor, wc=wc / 100.0, load_grad=gs,
                            check_ipr=check_ipr, use_ipr_unloading=use_ipr_unl, orifice_sizing=orif, economics=econ)
    try:
        ipr_curve = rebuild_ipr({"wc": wc / 100.0})
    except ValueError:
        pass
    sig = (repr(inp), repr(lg), repr(ct), corr, p1, p2, u, fluid_note, float(ipr_curve.pr), float(ipr_curve.aof))
    with right:
        if go:
            path = vlp_path(eqd, "tubing", (max_md,))
            msg = st.empty()
            try:
                with st.spinner("Designing..."):
                    res = wct.design(path, fluid, ipr_curve, lg, ct, inp, corr, p1, p2, u, eqd["heat_capacities"],
                                     progress=lambda m: msg.caption(m))
                ss["ct_res"] = {"sig": sig, "res": res, "err": None}
            except (ValueError, RuntimeError, ZeroDivisionError) as e:
                ss["ct_res"] = {"sig": sig, "res": None, "err": str(e)}
            msg.empty()
        out = ss.get("ct_res")
        if out is None:
            st.info("Set the design inputs and press Design.")
            return
        if out["err"]:
            st.error(f"Design failed: {out['err']}")
            return
        if out["sig"] != sig:
            st.info("Inputs have changed since the last design - press Design to update.")
        res = out["res"]
        st.markdown("**Design results**")
        summ = pd.DataFrame([
            ("Design liquid rate", f"{res.q:,.2f}", "STB/d"),
            ("Design oil rate", f"{res.q_oil:,.2f}", "STB/d"),
            ("Design gas rate", f"{res.qgi_design:.4f}", "MMscf/d"),
            ("Actual injected gas rate", f"{res.qgi_actual:.4f}", "MMscf/d"),
            ("Actual injection pressure", f"{res.p_inj_actual:,.1f}", "psia"),
            ("Orifice size", f"{res.port_64:g}", "64ths in"),
            ("Kick-off injection depth", f"{res.kickoff_md:,.2f}", f"ft MD ({res.kickoff_tvd:,.1f} ft TVD)"),
            ("Maximum operating depth", f"{res.max_md:,.2f}", f"ft MD ({res.max_tvd:,.1f} ft TVD)"),
            ("Tubing pressure at the operating depth", f"{res.pt_inj:,.2f}", "psia"),
            ("Lift-gas pressure at the operating depth", f"{res.pg_inj:,.2f}", "psia"),
            ("Temperature at the operating depth", f"{res.t_inj:,.2f}", "°F"),
        ], columns=["Result", "Value", "Units"])
        st.dataframe(summ, hide_index=True, width="stretch")
        for m_ in res.messages:
            st.caption(m_)
        st.caption("Transfer the design to the Equipment tab (Coiled tubing data: Transfer from the coiled tubing "
                   "design) to use it in the Calculation tab.")
        t1, t2, t3 = st.tabs(["Design plot", "Performance curve", "Design gradient"])
        with t1:
            st.pyplot(ct_design_plot(res), clear_figure=True)
        with t2:
            if res.perf is None:
                st.info("No performance curve for a user-entered design rate (as in PROSPER).")
            else:
                from types import SimpleNamespace
                res_ = SimpleNamespace(inp=res.inp, qgi=res.qgi_design, q=res.q)
                st.pyplot(gl_perf_plot(res.perf, res_), clear_figure=True)
                if np.isfinite(res.std_dev):
                    st.caption(f"Standard deviation of the points about the fitted curve: {res.std_dev:.2f} STB/d.")
                st.dataframe(res.perf.style.format(precision=2, na_rep="—"), hide_index=True, width="stretch")
        with t3:
            st.dataframe(res.profile.style.format(precision=2), hide_index=True, width="stretch")
            st.download_button("Download (CSV)", res.profile.to_csv(index=False).encode(), "ct_design_gradient.csv")


with tabs[6]:
    st.header("Design")
    design_kind = st.radio("Artificial lift", ["Gas lift (continuous)", "Gas lift (intermittent)",
                                               "Gas lift (coiled tubing)",
                                               "Electrical submersible pump", "Sucker rod pump", "Jet pump",
                                               "Progressive cavity pump", "Hydraulic submersible pump"],
                           horizontal=True,
                           key="design_kind", help="Further artificial lift designs (ESP, rod pump, ...) can be added "
                                                   "here.")
    o = ss.get("options")
    eqd = ss.get("equipment", {})
    ipr_curve = ss.get("ipr_curve")
    blocker = None
    if o is None or o.errors:
        blocker = "Fix the configuration in the Options summary first."
    elif o.is_gas or o.is_injector:
        blocker = "Artificial lift design applies to oil producers."
    elif o.flow_type != "tubing":
        blocker = ("Artificial lift design needs tubing flow (lift gas or produced gas goes up the annulus; the "
                   "pump sits on the tubing).")
    elif not eqd.get("valid") or eqd.get("survey") is None or not len(eqd["downhole"][1]):
        blocker = "Complete the Equipment Data tab first."
    elif ipr_curve is None or ipr_curve.fluid != "oil":
        blocker = "Calculate an oil IPR in the IPR Data tab first."
    if blocker:
        st.info(blocker)
    elif design_kind == "Electrical submersible pump":
        esp_tab(o, eqd, ipr_curve)
    elif design_kind == "Jet pump":
        jet_design_tab(o, eqd, ipr_curve)
    elif design_kind == "Progressive cavity pump":
        pcp_design_tab(o, eqd, ipr_curve)
    elif design_kind == "Hydraulic submersible pump":
        hsp_design_tab(o, eqd, ipr_curve)
    elif design_kind == "Sucker rod pump":
        srp_mode = st.radio("Sucker rod pump", ["Design (new installation)", "System analysis (existing installation)"],
                            horizontal=True, key="srp_page")
        if srp_mode.startswith("Design"):
            srp_tab(o, eqd, ipr_curve)
        else:
            srp_system_tab(o, eqd, ipr_curve)
    elif design_kind == "Gas lift (coiled tubing)":
        ct_design_tab(o, eqd, ipr_curve)
    elif design_kind == "Gas lift (intermittent)":
        igl_method = st.radio("Method", ["PROSPER (White et al. slug model)", "Brown (weighted-average BHP)"],
                              horizontal=True, key="igl_method")
        if igl_method.startswith("PROSPER"):
            igl_prosper_tab(o, eqd, ipr_curve)
        else:
            igl_tab(o, eqd, ipr_curve)
    else:
        gaslift_tab(o, eqd, ipr_curve)


# ================================================================ tab 8: units
def _commit_tables():
    """Before the units change, make every table's current edits its base (the editors restart in new units)."""
    for k_, df_ in (ss.get("_tables") or {}).items():
        if isinstance(df_, pd.DataFrame):
            ss[k_] = df_.copy()


def _apply_preset(box_key):
    name = ss.get(box_key)
    preset = wu.PRESETS.get(name)
    if not preset:
        return
    _commit_tables()
    for q_, u_ in preset.items():
        put("unit_" + q_, u_)


with tabs[7]:
    st.header("Units")
    st.caption("Choose the units for inputs and results. Every value is stored in oilfield units, so changing units "
               "never changes the model - inputs, tables, plots and downloads are converted on screen. Pressure "
               "can be absolute or gauge; dome and test-rack pressures stay gauge.")
    cur_ = ss["_units"]
    match_ = next((n_ for n_, p_ in wu.PRESETS.items() if all(cur_.get(q_) == u_ for q_, u_ in p_.items())), "Custom")
    c = st.columns([1.2, 2])
    box_ = f"unit_preset_box_{match_}"
    c[0].selectbox("Unit system", list(wu.PRESETS) + ["Custom"],
                   index=(list(wu.PRESETS) + ["Custom"]).index(match_), key=box_,
                   on_change=_apply_preset, args=(box_,),
                   help="Presets set every quantity; change any of them below for a custom set.")
    qs_ = list(wu.QUANTITIES)
    cols_ = st.columns(3)
    for i_, q_ in enumerate(qs_):
        sel(cols_[i_ % 3], wu.QUANTITY_LABELS[q_], list(wu.QUANTITIES[q_]), "unit_" + q_, wu.default_unit(q_),
            on_change=_commit_tables)
    st.caption("Not converted: water cut and other percentages and fractions, gravities, FVFs and ratios without "
               "units, choke and port sizes in 64ths of an inch, the well schematic labels (ft and inches), and the Calculation tab's lift-curve export (Eclipse "
               "VFP tables are always written in FIELD units).")


# ================================================================ save button (top bar) and schematic (sidebar)
with save_slot:
    try:
        payload = wproj.dumps(project_state(), pname)
        ss["_proj_payload"] = payload
        fname = _file_stem(pname)
        st.download_button("", payload.encode(), f"{fname}.fluxwell.json", "application/json",
                           icon=":material/save:", key="proj_save", on_click="ignore",
                           help=f"Save project as {fname}.fluxwell.json - every input, table and accepted match.")
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
