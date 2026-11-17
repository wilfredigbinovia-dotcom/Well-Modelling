"""Title Case at the Streamlit screen edge.

Every label the app shows - widget labels and options, tabs, expanders, headers, metric labels, table column
headers and chart axis labels, titles and legends - is shown with the first letter of every word capitalised,
except 'of' and 'and' (and unit / symbol tokens such as psi, ft, md, kh or kro, which keep their case). Text in
brackets (units, abbreviations) is left as it is. Only what is displayed changes: the values the code compares, the
column names it reads and the session state keep their original spelling.
"""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

LOWER = {"of", "and"}
# unit / symbol tokens kept as written (outside brackets)
KEEP = {"psi", "psia", "psig", "ft", "in", "md", "cp", "kh", "kro", "krw", "krg", "ko", "kw", "kg", "kv", "scf",
        "stb", "bbl", "mmscf", "mscf", "lb", "lbm", "lbf", "gal", "hp", "rpm", "bpd", "cc", "mm", "cm", "m", "km",
        "g", "s", "h", "hr", "min", "d", "x", "ppm", "mol", "lbmol", "btu", "mv", "kva", "v", "hz", "a", "e", "n",
        "pH", "vs", "via", "per", "sec", "deg", "e.g.", "i.e.", "etc.", "dP", "dT", "dp", "dt", "rb", "ln",
        "kd", "ks", "kl", "rw", "rp", "rd", "lp", "pb", "pr", "pwf", "qo", "qw", "qg", "mu", "bo", "bw",
        "bg", "rs", "rv", "z", "k", "t", "p", "q", "api", "sg"}
_WORD = re.compile(r"[A-Za-z][A-Za-z'’.\-]*")


def _cap_word(w, first):
    core = w.rstrip(".,:;!?")
    if not first and core.lower() in LOWER:
        return w.lower() if core == core.lower() else w
    if core in KEEP or core.lower() in KEEP and core == core.lower():
        return w
    if w[0].islower():
        return w[0].upper() + w[1:]
    return w


def _case_plain(text, first_word):
    out, pos = [], 0
    for m in _WORD.finditer(text):
        out.append(text[pos:m.start()])
        prev = text[:m.start()].rstrip()
        # a word straight after a digit or '=' / '/' is a value or a unit (4,000 psia; x/y)
        nxt = text[m.end():m.end() + 1]
        if (prev and (prev[-1].isdigit() or prev[-1] in "=/^*×") and not first_word[0]) or nxt == "_" \
                or text[m.start() - 1:m.start()] == "_":
            out.append(m.group())
        else:
            out.append(_cap_word(m.group(), first_word[0]))
        first_word[0] = False
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def _is_unit(t):
    try:
        from . import units as U
        return U.parse_unit(t) is not None
    except Exception:
        return False


def title_case(text):
    """'Liquid rate at the end of the tubing (STB/d)' -> 'Liquid Rate At The End of The Tubing (STB/d)'."""
    if not isinstance(text, str) or not text or text.startswith(("http", "/", ":")):
        return text
    parts = re.split(r"(\([^()]*\)|\[[^\[\]]*\]|`[^`]*`|\$[^$]*\$)", text)
    first = [True]
    for i, p in enumerate(parts):
        if i % 2 == 0:
            parts[i] = _case_plain(p, first)
        elif p:
            inner = p[1:-1]
            if p[0] in "([" and inner and not _is_unit(inner) and re.search(r"[a-z]{3,}", inner) \
                    and len(inner.split()) >= 1 and not re.fullmatch(r"[^ ]*[/.^*][^ ]*", inner):
                parts[i] = p[0] + _case_plain(inner, [False]) + p[-1]
            first[0] = False
    return "".join(parts)


def _md(body):
    """Markdown: header lines and a leading bold label ('**Results** - 3 cases') only; prose is left alone."""
    if not isinstance(body, str):
        return body
    lines = body.split("\n")
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*#{1,6}\s+)(.*)$", ln)
        if m:
            lines[i] = m.group(1) + title_case(m.group(2))
            continue
        m = re.match(r"^(\s*\*\*)([^*]{1,80})(\*\*)(.*)$", ln)
        if m and len(m.group(2).split()) <= 10:
            lines[i] = m.group(1) + title_case(m.group(2)) + m.group(3) + m.group(4)
    return "\n".join(lines)


def _fmt(ff):
    def f(v):
        r = ff(v) if ff is not None else v
        if isinstance(r, str):
            return title_case(r)
        return str(r)
    return f


def _wrap_label(orig, opts=False):
    def w(self, label, *a, **kw):
        if opts:
            kw["format_func"] = _fmt(kw.get("format_func"))
        return orig(self, title_case(label), *a, **kw)
    w._wc = True
    return w


def _wrap_tabs(orig):
    def tabs(self, labels, *a, **kw):
        return orig(self, [title_case(x) for x in labels], *a, **kw)
    tabs._wc = True
    return tabs


def _wrap_md(orig):
    def md(self, body, *a, **kw):
        return orig(self, _md(body), *a, **kw)
    md._wc = True
    return md


def _labels(cols, cfg):
    """Column headers in Title Case through column_config labels (the data and its column names are untouched)."""
    cfg = dict(cfg or {})
    for c in cols:
        if not isinstance(c, str):
            continue
        base = cfg.get(c)
        if base is None:
            new = title_case(c)
            if new != c:
                cfg[c] = {"label": new}
        elif isinstance(base, str):
            cfg[c] = title_case(base)
        elif isinstance(base, dict):
            b = dict(base)
            b["label"] = title_case(b.get("label") or c)
            cfg[c] = b
    return cfg


def _wrap_dataframe(orig):
    def dataframe(self, data=None, *a, **kw):
        df = data if isinstance(data, pd.DataFrame) else (
            data.data if data is not None and data.__class__.__name__ == "Styler" else None)
        if isinstance(df, pd.DataFrame) and not isinstance(df.columns, pd.MultiIndex) and df.columns.is_unique:
            kw["column_config"] = _labels(df.columns, kw.get("column_config"))
        return orig(self, data, *a, **kw)
    dataframe._wc = True
    return dataframe


def _wrap_editor(orig):
    def data_editor(self, data, *a, column_config=None, **kw):
        if isinstance(data, pd.DataFrame) and not isinstance(data.columns, pd.MultiIndex) and data.columns.is_unique:
            column_config = _labels(data.columns, column_config)
        return orig(self, data, *a, column_config=column_config, **kw)
    data_editor._wc = True
    return data_editor


def case_figure(fig):
    for ax in fig.axes:
        ax.set_xlabel(title_case(ax.get_xlabel()))
        ax.set_ylabel(title_case(ax.get_ylabel()))
        ax.set_title(title_case(ax.get_title()))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_text(title_case(t.get_text()))
            if leg.get_title() is not None:
                leg.get_title().set_text(title_case(leg.get_title().get_text()))
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text(title_case(fig._suptitle.get_text()))
    return fig


def _wrap_pyplot(orig):
    def pyplot(self, fig=None, *a, **kw):
        if fig is not None:
            try:
                case_figure(fig)
            except Exception:                    # never let a display change break the page
                pass
        return orig(self, fig, *a, **kw)
    pyplot._wc = True
    return pyplot


def _wrap_metric(orig):
    def metric(self, label, value, *a, **kw):
        return orig(self, title_case(label), value, *a, **kw)
    metric._wc = True
    return metric


def install():
    """Wrap the Streamlit display functions once per process (after ui_units.install, so this runs first)."""
    plain = ["number_input", "text_input", "text_area", "checkbox", "toggle", "button", "download_button",
             "slider", "date_input", "file_uploader", "expander", "header", "subheader", "title", "popover",
             "form_submit_button", "color_picker", "time_input"]
    with_opts = ["selectbox", "multiselect", "radio", "select_slider", "pills", "segmented_control"]
    wrappers = {**{m: (lambda o: _wrap_label(o)) for m in plain},
                **{m: (lambda o: _wrap_label(o, True)) for m in with_opts},
                "tabs": _wrap_tabs, "markdown": _wrap_md, "dataframe": _wrap_dataframe,
                "data_editor": _wrap_editor, "pyplot": _wrap_pyplot, "metric": _wrap_metric}
    for name, wrap in wrappers.items():
        meth = getattr(DeltaGenerator, name, None)
        if meth is not None and not getattr(meth, "_wc", False):
            setattr(DeltaGenerator, name, wrap(meth))
    main = st._main if hasattr(st, "_main") else None
    for name in wrappers:
        fn = getattr(st, name, None)
        if fn is not None and not getattr(getattr(fn, "__func__", fn), "_wc", False) and main is not None:
            setattr(st, name, getattr(main, name))
