"""Unit conversion at the Streamlit screen edge.

The app keeps every value in the engine's oilfield units. These wrappers convert what is shown:
  * tables (st.dataframe / Styler): columns whose name ends in '(unit)', and 'Value' / 'Units' result tables;
  * input tables (st.data_editor): shown in the chosen units, returned in internal units;
  * metrics and short texts (caption, info, markdown, ...): 'number unit' pairs;
  * matplotlib figures (st.pyplot): axes labelled 'name (unit)' have their data, limits and labels converted;
  * CSV downloads: the columns as in tables.
The chosen units live in st.session_state['_units'] (quantity -> unit); with the defaults nothing is touched.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from . import units as U

_TEXT_METHODS = ("caption", "info", "warning", "success", "error", "markdown")


def current():
    return st.session_state.get("_units") or {}


def active():
    u = current()
    return None if U.is_default(u) else u


# ------------------------------------------------------------------ tables
def _parse_num(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v), None
    if isinstance(v, str):
        t = v.strip().replace(",", "")
        try:
            return float(t), v
        except ValueError:
            return None, v
    return None, v


def _conv_value(v, q, u, to):
    x, orig = _parse_num(v)
    if x is None or not np.isfinite(x):
        return v
    y = U.convert(x, q, u, to)
    return U._fmt_like(orig, y) if orig is not None else y


def _conv_series(s, q, u, to):
    if pd.api.types.is_numeric_dtype(s):
        return pd.Series(U.convert(s.to_numpy(float), q, u, to), index=s.index, name=s.name)
    return s.map(lambda v: _conv_value(v, q, u, to))


def df_out(df, units, mapping=None):
    """DataFrame in the chosen units (new frame; column names relabelled). mapping receives old -> new names."""
    if not isinstance(df, pd.DataFrame) or isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    new_cols = []
    for c in out.columns:
        if isinstance(c, str):
            name, info = U.relabel(c, units)
            if info is not None and info[2] != info[1]:
                out[c] = _conv_series(out[c], *info)
            new_cols.append(name)
        else:
            new_cols.append(c)
    unit_col = next((c for c in ("Units", "Unit", "units") if c in out.columns), None)
    val_cols = [c for c in out.columns if isinstance(c, str) and c.split(" (")[0] in ("Value", "Value 1", "Value 2")]
    if unit_col is not None and val_cols:
        for i in range(len(out)):
            uu = out[unit_col].iloc[i]
            p = U.parse_unit(uu) if isinstance(uu, str) and uu else None
            if p is None:
                continue
            q, u, suffix = p
            to = U.display_unit(q, u, units)
            if to == u:
                continue
            for vc in val_cols:
                out.iat[i, out.columns.get_loc(vc)] = _conv_value(out[vc].iloc[i], q, u, to)
            lab = U.unit_label(to, suffix if not suffix.startswith(",") else "")
            out.iat[i, out.columns.get_loc(unit_col)] = f"({lab})" if uu.strip().startswith("(") else lab
    name_col = next((c for c in ("Result", "Parameter", "Quantity", "Property", "Item", "Variable")
                     if c in out.columns), None)
    if unit_col is None and name_col is not None:
        vcols = val_cols or [c for c in out.columns if c != name_col]
        for i in range(len(out)):
            nm = out[name_col].iloc[i]
            if not isinstance(nm, str):
                continue
            new_nm, info = U.relabel(nm, units)
            if info is None or info[2] == info[1]:
                continue
            for vc in vcols:
                out.iat[i, out.columns.get_loc(vc)] = _conv_value(out[vc].iloc[i], *info)
            out.iat[i, out.columns.get_loc(name_col)] = new_nm
    if mapping is not None:
        mapping.update({o: n for o, n in zip(df.columns, new_cols) if o != n})
    out.columns = new_cols
    return out


def _styler_out(sty, units, mapping):
    data = sty.data
    new = df_out(data, units, mapping)
    if new is data:
        return sty
    sty.data = new
    sty.columns = new.columns
    return sty


def _cfg_rename(cfg, mapping):
    if not cfg or not mapping:
        return cfg
    return {mapping.get(k, k): v for k, v in cfg.items()}


# ------------------------------------------------------------------ figures
def _axis_info(ax, which, units):
    sib = (ax.get_shared_x_axes() if which == "x" else ax.get_shared_y_axes()).get_siblings(ax)
    for a in [ax] + [s for s in sib if s is not ax]:
        lab = a.get_xlabel() if which == "x" else a.get_ylabel()
        if lab:
            name, info = U.relabel(lab, units)
            return (info if info is not None and info[2] != info[1] else None), name, a
    return None, None, None


def _fn(info):
    if info is None:
        return None
    q, u, to = info
    return lambda v: U.convert(np.asarray(v, float), q, u, to)


def convert_figure(fig, units):
    done_x, done_y = set(), set()
    for ax in fig.axes:
        xi, xname, xowner = _axis_info(ax, "x", units)
        yi, yname, yowner = _axis_info(ax, "y", units)
        fx, fy = _fn(xi), _fn(yi)
        if xi is not None and xowner is ax:
            ax.set_xlabel(xname)
        if yi is not None and yowner is ax:
            ax.set_ylabel(yname)
        if fx or fy:
            for ln in ax.lines:
                cx, cy = ln.get_transform().contains_branch_seperately(ax.transData)
                x, y = ln.get_xdata(orig=False), ln.get_ydata(orig=False)
                if fx and cx:
                    x = fx(x)
                if fy and cy:
                    y = fy(y)
                ln.set_data(x, y)
            for coll in ax.collections:
                offs = coll.get_offsets()
                if offs is not None and len(offs) and coll.get_offset_transform().contains_branch(ax.transData):
                    o = np.array(offs, float)
                    if fx:
                        o[:, 0] = fx(o[:, 0])
                    if fy:
                        o[:, 1] = fy(o[:, 1])
                    coll.set_offsets(o)
                elif coll.get_transform().contains_branch(ax.transData):
                    for path in coll.get_paths():
                        v = path.vertices
                        if fx:
                            v[:, 0] = fx(v[:, 0])
                        if fy:
                            v[:, 1] = fy(v[:, 1])
            for p in ax.patches:
                try:
                    cx, cy = p.get_transform().contains_branch_seperately(ax.transData)
                    if hasattr(p, "get_x") and hasattr(p, "get_width"):
                        x0, w = p.get_x(), p.get_width()
                        y0, h = p.get_y(), p.get_height()
                        if fx and cx:
                            a, b = fx([x0, x0 + w]); p.set_x(a); p.set_width(b - a)
                        if fy and cy:
                            a, b = fy([y0, y0 + h]); p.set_y(a); p.set_height(b - a)
                except (AttributeError, TypeError, ValueError):
                    pass
            for t in ax.texts:
                try:
                    cx, cy = t.get_transform().contains_branch_seperately(ax.transData)
                    x, y = t.get_position()
                    t.set_position((float(fx(x)) if fx and cx else x, float(fy(y)) if fy and cy else y))
                except (AttributeError, TypeError, ValueError):
                    pass
            gx = frozenset(id(a) for a in ax.get_shared_x_axes().get_siblings(ax))
            gy = frozenset(id(a) for a in ax.get_shared_y_axes().get_siblings(ax))
            if fx and gx not in done_x:
                lo, hi = ax.get_xlim()
                ax.set_xlim(*fx([lo, hi]))
                done_x.add(gx)
            if fy and gy not in done_y:
                lo, hi = ax.get_ylim()
                ax.set_ylim(*fy([lo, hi]))
                done_y.add(gy)
        ax.set_title(U.convert_text(ax.get_title(), units))
        for t in ax.texts:
            t.set_text(U.convert_text(t.get_text(), units))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_text(U.convert_text(t.get_text(), units))
    if fig._suptitle is not None:
        fig._suptitle.set_text(U.convert_text(fig._suptitle.get_text(), units))
    return fig


# ------------------------------------------------------------------ patches
def _wrap_dataframe(orig):
    def dataframe(self, data=None, *a, **kw):
        units = active()
        if units is not None:
            mapping = {}
            if isinstance(data, pd.DataFrame):
                data = df_out(data, units, mapping)
            elif data is not None and data.__class__.__name__ == "Styler":
                data = _styler_out(data, units, mapping)
            if "column_config" in kw:
                kw["column_config"] = _cfg_rename(kw["column_config"], mapping)
            if "column_order" in kw and kw["column_order"]:
                kw["column_order"] = [mapping.get(c, c) for c in kw["column_order"]]
        return orig(self, data, *a, **kw)
    dataframe._wm = True
    return dataframe


def _label_of(cfg, col):
    c = (cfg or {}).get(col)
    if isinstance(c, dict) and c.get("label"):
        return c["label"]
    if isinstance(c, str):
        return c
    return col


def _wrap_data_editor(orig):
    def data_editor(self, data, *a, key=None, column_config=None, **kw):
        units = active()
        if units is None or not isinstance(data, pd.DataFrame):
            return orig(self, data, *a, key=key, column_config=column_config, **kw)
        conv, cfg = {}, dict(column_config or {})
        for c in data.columns:
            if not isinstance(c, str):
                continue
            lab = _label_of(cfg, c)
            name, info = U.relabel(lab.replace("\n", " "), units)
            if info is None or info[2] == info[1]:
                continue
            conv[c] = info
            base = cfg.get(c)
            if isinstance(base, dict):
                base = dict(base)
                base["label"] = name
            else:
                base = {"label": name}
            tc = base.get("type_config") or {}
            if isinstance(tc, dict) and not tc.get("format"):
                base["type_config"] = {**tc, "format": "%g"}
            cfg[c] = base
        if not conv:
            return orig(self, data, *a, key=key, column_config=column_config, **kw)
        disp = data.copy()
        for c, (q, u, to) in conv.items():
            disp[c] = U.convert(pd.to_numeric(disp[c], errors="coerce").to_numpy(float), q, u, to)
        sig = "|".join(f"{c}:{to}" for c, (_, _, to) in conv.items())
        key2 = f"{key}@{abs(hash(sig)) % 10 ** 8}" if key is not None else None
        out = orig(self, disp, *a, key=key2, column_config=cfg, **kw)
        back = out.copy()
        for c, (q, u, to) in conv.items():
            if c in back:
                back[c] = U.convert(pd.to_numeric(back[c], errors="coerce").to_numpy(float), q, to, u)
        return back
    data_editor._wm = True
    return data_editor


def _wrap_metric(orig):
    def metric(self, label, value, *a, **kw):
        units = active()
        if units is not None:
            name, info = U.relabel(label, units)
            if info is not None and info[2] != info[1]:
                value = _conv_value(value, *info)
                label = name
            elif isinstance(value, str):
                value = U.convert_text(value, units)
            label = U.convert_text(label, units)
        return orig(self, label, value, *a, **kw)
    metric._wm = True
    return metric


def _wrap_text(orig):
    def text(self, body, *a, **kw):
        units = active()
        if units is not None and isinstance(body, str):
            body = U.convert_text(body, units)
        return orig(self, body, *a, **kw)
    text._wm = True
    return text


def _wrap_pyplot(orig):
    def pyplot(self, fig=None, *a, **kw):
        units = active()
        if units is not None and fig is not None:
            try:
                convert_figure(fig, units)
            except Exception:                    # never let a display conversion break the page
                pass
        return orig(self, fig, *a, **kw)
    pyplot._wm = True
    return pyplot


def _wrap_download(orig):
    def download_button(self, label, data, file_name=None, *a, **kw):
        units = active()
        if units is not None and isinstance(file_name, str) and file_name.lower().endswith(".csv"):
            try:
                raw = data.encode() if isinstance(data, str) else data
                df = pd.read_csv(io.BytesIO(raw))
                data = df_out(df, units).to_csv(index=False).encode()
            except Exception:
                pass
        return orig(self, label, data, file_name, *a, **kw)
    download_button._wm = True
    return download_button


def install():
    """Wrap the Streamlit display functions once per process (DeltaGenerator methods and st.* shortcuts)."""
    wrappers = {"dataframe": _wrap_dataframe, "data_editor": _wrap_data_editor, "metric": _wrap_metric,
                "pyplot": _wrap_pyplot, "download_button": _wrap_download,
                **{m: _wrap_text for m in _TEXT_METHODS}}
    for name, wrap in wrappers.items():
        meth = getattr(DeltaGenerator, name)
        if not getattr(meth, "_wm", False):
            setattr(DeltaGenerator, name, wrap(meth))
    main = st._main if hasattr(st, "_main") else None
    for name in wrappers:
        fn = getattr(st, name)
        if not getattr(getattr(fn, "__func__", fn), "_wm", False):
            if main is not None:
                setattr(st, name, getattr(main, name))
