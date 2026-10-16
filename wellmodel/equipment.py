"""Equipment data: deviation survey (with a 20-point filter), surface equipment, downhole equipment,
geothermal gradient and average heat capacities. Oilfield units: ft, in, degF, BTU/h/ft2/F, BTU/lb/F.

Every check returns Issue-style tuples (level, message) with level "error" (cannot be used),
"warning" (questionable) or "info".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_SURVEY_POINTS = 20

# ---------------------------------------------------------- deviation survey
SURVEY_COLUMNS = ["Measured Depth (ft)", "True Vertical Depth (ft)"]


def survey_arrays(df: pd.DataFrame):
    """Non-blank (MD, TVD) rows as float arrays, in entry order."""
    d = df[SURVEY_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)


def check_survey(md, tvd, max_points=MAX_SURVEY_POINTS):
    out = []
    if np.isnan(md).any() or np.isnan(tvd).any():
        out.append(("error", "Every survey point needs both a measured depth and a true vertical depth."))
        return out
    if len(md) < 2:
        out.append(("error", "Enter at least two survey points (the top point is normally 0, 0)."))
        return out
    if len(md) > max_points:
        out.append(("error", f"{len(md)} points entered; at most {max_points} are used. Use the filter."))
    if md[0] != 0 or tvd[0] != 0:
        out.append(("warning", "The first survey point is normally MD 0 ft, TVD 0 ft at the depth reference."))
    dmd, dtvd = np.diff(md), np.diff(tvd)
    bad = np.where(dmd <= 0)[0]
    if len(bad):
        out.append(("error", "Measured depth must increase down the survey (check point "
                    + ", ".join(str(i + 2) for i in bad[:5]) + ")."))
        return out
    bad = np.where(np.abs(dtvd) > dmd * (1 + 1e-6) + 0.15)[0]   # 0.15 ft allows for survey rounding
    if len(bad):
        out.append(("error", "TVD changes by more than the MD between points "
                    + ", ".join(str(i + 2) for i in bad[:5]) + " - not physically possible."))
    up = np.where(dtvd < 0)[0]
    if len(up):
        out.append(("warning", "TVD decreases (well builds above horizontal) at point "
                    + ", ".join(str(i + 2) for i in up[:5]) + "."))
    return out


def survey_table(md, tvd) -> pd.DataFrame:
    """Survey with calculated angle from vertical and cumulative horizontal displacement."""
    dmd, dtvd = np.diff(md), np.diff(tvd)
    ratio = np.clip(dtvd / np.where(dmd > 0, dmd, 1), -1, 1)
    angle = np.degrees(np.arccos(ratio))
    disp = np.concatenate([[0.0], np.cumsum(np.sqrt(np.maximum(dmd ** 2 - dtvd ** 2, 0)))])
    return pd.DataFrame({"Measured Depth (ft)": md, "True Vertical Depth (ft)": tvd,
                         "Cumulative Displacement (ft)": disp,
                         "Angle from vertical (deg)": np.concatenate([[0.0], angle])})


def filter_survey(md, tvd, n=MAX_SURVEY_POINTS):
    """Choose n points from a long survey that best keep its shape.

    Greedy insertion: start with the first and last points, then repeatedly add the point whose
    TVD is worst reproduced by linear interpolation (in MD) between the points already kept.
    Returns (indices kept, max TVD error ft, RMS TVD error ft)."""
    md, tvd = np.asarray(md, float), np.asarray(tvd, float)
    order = np.argsort(md)
    md, tvd = md[order], tvd[order]
    if len(md) <= n:
        return order, 0.0, 0.0
    keep = {0, len(md) - 1}
    while len(keep) < n:
        k = np.array(sorted(keep))
        err = np.abs(tvd - np.interp(md, md[k], tvd[k]))
        err[k] = -1
        keep.add(int(np.argmax(err)))
    k = np.array(sorted(keep))
    err = tvd - np.interp(md, md[k], tvd[k])
    return order[k], float(np.max(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def tvd_at(md_survey, tvd_survey, md):
    return np.interp(md, md_survey, tvd_survey)


# --------------------------------------------------------- surface equipment
SURFACE_TYPES = ["Manifold", "Line Pipe", "Choke", "Fitting"]
FITTINGS = {   # equivalent length in pipe diameters, L/D (Crane TP-410 typical values)
    "90° elbow (standard)": 30, "90° elbow (long radius)": 20, "45° elbow": 16,
    "Tee - flow through run": 20, "Tee - flow through branch": 60,
    "Gate valve (open)": 8, "Ball valve (open)": 3, "Globe valve (open)": 340,
    "Swing check valve": 100, "Plug valve (open)": 18,
}
SURFACE_COLUMNS = ["Label", "Type", "Fitting", "Pipe Length (ft)", "TVD of End Point (ft)",
                   "Pipe Inside Diameter (in)", "Pipe Inside Roughness (in)", "Rate Multiplier"]


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v


# Cells each surface type unlocks, as in PROSPER's surface equipment screen.
SURFACE_NUMERIC = SURFACE_COLUMNS[3:]
SURFACE_DEFAULTS = {"Pipe Inside Roughness (in)": 0.0006, "Rate Multiplier": 1.0}


def surface_active_columns(row_type):
    """Input columns (numeric and the Fitting name) open for a surface row of this type."""
    ln, tvd, d, e, rm = SURFACE_NUMERIC
    if row_type == "Manifold":
        return {tvd}
    if row_type == "Line Pipe":
        return {ln, tvd, d, e, rm}
    if row_type == "Choke":
        return {d, rm}
    if row_type == "Fitting":
        return {"Fitting", d, rm}
    return set()


def apply_surface_rules(df):
    """Clear locked surface cells and fill empty roughness / rate multiplier cells with 0.0006 in and 1."""
    out = df.copy()
    for c in SURFACE_NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out["Fitting"] = out["Fitting"].astype(object)
    for i, r in out.iterrows():
        t = r.get("Type")
        act = surface_active_columns(t if isinstance(t, str) else "")
        if "Fitting" not in act:
            out.at[i, "Fitting"] = None
        for c in SURFACE_NUMERIC:
            if c not in act:
                out.at[i, c] = np.nan
            elif c in SURFACE_DEFAULTS and np.isnan(out.at[i, c]):
                out.at[i, c] = SURFACE_DEFAULTS[c]
    return out


def surface_rows(df):
    return [r for _, r in df.iterrows() if isinstance(r.get("Type"), str) and r["Type"]]


def check_surface(df, surrounding_t=None, u=None):
    """Rows in PROSPER order: manifold first, then equipment back towards the wellhead."""
    out, rows = [], surface_rows(df)
    if not rows:
        out.append(("info", "No surface equipment entered: calculations will start at the wellhead."))
        return out
    if rows[0]["Type"] != "Manifold":
        out.append(("error", "The first surface entry must be the Manifold (with its TVD)."))
    if sum(r["Type"] == "Manifold" for r in rows) > 1:
        out.append(("error", "Only one Manifold is allowed."))
    last_id = np.nan
    for i, r in enumerate(rows, 1):
        t, L, d, e = r["Type"], _f(r["Pipe Length (ft)"]), _f(r["Pipe Inside Diameter (in)"]), \
            _f(r["Pipe Inside Roughness (in)"])
        name = f"Row {i} ({t})"
        if t == "Manifold":
            if np.isnan(_f(r["TVD of End Point (ft)"])):
                out.append(("error", f"{name}: enter the manifold TVD."))
            continue
        if np.isnan(d) or d <= 0:
            out.append(("error", f"{name}: enter a positive inside diameter."))
            continue
        if t == "Line Pipe":
            if np.isnan(L) or L <= 0:
                out.append(("error", f"{name}: enter a positive pipe length."))
            if np.isnan(_f(r["TVD of End Point (ft)"])):
                out.append(("error", f"{name}: enter the TVD of the pipe end point."))
            elif not np.isnan(L) and i > 1:
                prev_tvd = _f(rows[i - 2]["TVD of End Point (ft)"])
                if not np.isnan(prev_tvd) and abs(_f(r["TVD of End Point (ft)"]) - prev_tvd) > L:
                    out.append(("error", f"{name}: the TVD change is larger than the pipe length."))
            if np.isnan(e) or e < 0 or e >= d / 2:
                out.append(("error", f"{name}: roughness must be between 0 and half the diameter."))
            last_id = d
        elif t == "Choke":
            if not np.isnan(last_id) and d >= last_id:
                out.append(("warning", f"{name}: choke bean ({d:g} in) is not smaller than the adjacent pipe."))
        elif t == "Fitting":
            if r.get("Fitting") not in FITTINGS:
                out.append(("error", f"{name}: choose the fitting type."))
        if not np.isnan(_f(r.get("Rate Multiplier"))) and _f(r["Rate Multiplier"]) <= 0:
            out.append(("error", f"{name}: rate multiplier must be positive."))
    if surrounding_t is not None and not -60 <= surrounding_t <= 200:
        out.append(("warning", "Surrounding temperature looks unusual."))
    if u is not None and u <= 0:
        out.append(("error", "The overall heat transfer coefficient must be positive."))
    return out


def surface_summary(df) -> pd.DataFrame:
    recs = []
    for r in surface_rows(df):
        t, d = r["Type"], _f(r["Pipe Inside Diameter (in)"])
        if t == "Manifold":
            recs.append({"Item": r["Label"] or t, "Type": t, "Length (ft)": 0.0, "ID (in)": np.nan,
                         "Equivalent length (ft)": 0.0, "Note": f"TVD {_f(r['TVD of End Point (ft)']):g} ft"})
        elif t == "Line Pipe":
            L = _f(r["Pipe Length (ft)"])
            recs.append({"Item": r["Label"] or t, "Type": t, "Length (ft)": L, "ID (in)": d,
                         "Equivalent length (ft)": L, "Note": ""})
        elif t == "Choke":
            recs.append({"Item": r["Label"] or t, "Type": t, "Length (ft)": 0.0, "ID (in)": d,
                         "Equivalent length (ft)": 0.0, "Note": f"bean {d * 64:.0f}/64 in"})
        elif t == "Fitting":
            ld = FITTINGS.get(r.get("Fitting"), np.nan)
            recs.append({"Item": r["Label"] or r.get("Fitting") or t, "Type": t, "Length (ft)": 0.0, "ID (in)": d,
                         "Equivalent length (ft)": ld * d / 12, "Note": f"L/D = {ld:g}"})
    return pd.DataFrame(recs)


# -------------------------------------------------------- downhole equipment
DOWNHOLE_TYPES = ["Xmas Tree", "Tubing", "SSSV", "Restriction", "Gas Lift Mandrel", "SSD", "Casing"]
POINT_TYPES = {"SSSV", "Restriction", "Gas Lift Mandrel", "SSD"}
DOWNHOLE_COLUMNS = ["Label", "Type", "Measured Depth (ft)", "Tubing Inside Diameter (in)",
                    "Tubing Inside Roughness (in)", "Tubing Outside Diameter (in)",
                    "Tubing Outside Roughness (in)", "Casing Inside Diameter (in)",
                    "Casing Inside Roughness (in)", "Rate Multiplier"]
TYPE_HELP = {
    "Xmas Tree": "Depth of the Xmas tree (top of the downhole string). Only the MD is used.",
    "Tubing": "Tubing from the previous depth down to this MD.",
    "SSSV": "Sub-surface safety valve: minimum ID, at the previous depth unless an MD is entered.",
    "Restriction": "Nipple or other restriction: minimum ID, at the previous depth unless an MD is entered.",
    "Gas Lift Mandrel": "Gas lift mandrel: minimum ID, at the previous depth (treated like a restriction).",
    "SSD": "Sliding side door: minimum ID, at the previous depth (treated like a restriction).",
    "Casing": "Casing or liner from the previous depth down to this MD (flow below the tubing shoe).",
}


# Cells each type unlocks, as in PROSPER's downhole equipment screen (natural flow up the tubing). Annular flow
# also opens the tubing outside diameter/roughness and the casing ID/roughness on tubing rows.
DH_NUMERIC = DOWNHOLE_COLUMNS[2:]
DH_DEFAULTS = {"Tubing Inside Roughness (in)": 0.0006, "Tubing Outside Roughness (in)": 0.0006,
               "Casing Inside Roughness (in)": 0.0006, "Rate Multiplier": 1.0}


def active_columns(row_type, flow_type="tubing"):
    """Numeric columns that accept input for a row of this type."""
    md, tid, te, tod, toe, cid, ce, rm = DH_NUMERIC
    annulus = flow_type in ("annular", "tubing_annular")
    if row_type == "Xmas Tree":
        return {md}
    if row_type == "Tubing":
        return {md, tid, te, rm} | ({tod, toe, cid, ce} if annulus else set())
    if row_type == "SSSV":
        return {tid}
    if row_type in ("Restriction", "Gas Lift Mandrel", "SSD"):
        return {tid, rm}
    if row_type == "Casing":
        return {md, cid, ce, rm}
    return set()


def blocked_mask(df, flow_type="tubing"):
    """Boolean frame: True where a cell is locked for its row's type (Label and Type are always open)."""
    m = pd.DataFrame(False, index=df.index, columns=df.columns)
    for i, r in df.iterrows():
        t = r.get("Type")
        act = active_columns(t if isinstance(t, str) else "", flow_type)
        for c in DH_NUMERIC:
            if c in m.columns:
                m.at[i, c] = c not in act
    return m


def apply_cell_rules(df, flow_type="tubing"):
    """Clear locked cells and fill empty unlocked roughness / rate-multiplier cells with their defaults
    (0.0006 in and 1), which stay editable."""
    out = df.copy()
    for c in DH_NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for i, r in out.iterrows():
        t = r.get("Type")
        act = active_columns(t if isinstance(t, str) else "", flow_type)
        for c in DH_NUMERIC:
            if c not in act:
                out.at[i, c] = np.nan
            elif c in DH_DEFAULTS and np.isnan(out.at[i, c]):
                out.at[i, c] = DH_DEFAULTS[c]
    return out


def same_table(a, b):
    """True when two equipment tables hold the same values (NaN = NaN, empty text = None)."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if c in DH_NUMERIC or c in SURFACE_NUMERIC or c in SURVEY_COLUMNS or c in GEO_COLUMNS:
            x, y = pd.to_numeric(a[c], errors="coerce").to_numpy(float), pd.to_numeric(b[c], errors="coerce").to_numpy(float)
            if not np.allclose(x, y, equal_nan=True, rtol=0, atol=0):
                return False
        elif [("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)) for v in x] != \
                [("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)) for v in y]:
            return False
    return True


def downhole_rows(df):
    return [r for _, r in df.iterrows() if isinstance(r.get("Type"), str) and r["Type"]]


def check_downhole(df, flow_type="tubing", survey_max_md=None):
    """flow_type: 'tubing', 'annular' or 'tubing_annular' (annular flow needs tubing OD and casing ID)."""
    out, rows = [], downhole_rows(df)
    if not rows:
        out.append(("error", "Enter the downhole equipment, starting with the Xmas tree."))
        return out
    if rows[0]["Type"] != "Xmas Tree":
        out.append(("error", "The first entry must be the Xmas Tree."))
    if sum(r["Type"] == "Xmas Tree" for r in rows) > 1:
        out.append(("error", "Only one Xmas Tree is allowed."))
    annulus = flow_type in ("annular", "tubing_annular")
    last_md, last_tub_id, seen_casing, n_tubing = -np.inf, np.nan, False, 0
    for i, r in enumerate(rows, 1):
        t, md = r["Type"], _f(r["Measured Depth (ft)"])
        tid, te = _f(r["Tubing Inside Diameter (in)"]), _f(r["Tubing Inside Roughness (in)"])
        tod, ce = _f(r["Tubing Outside Diameter (in)"]), _f(r["Casing Inside Roughness (in)"])
        cid = _f(r["Casing Inside Diameter (in)"])
        name = f"Row {i} ({t})"
        if t == "Xmas Tree":
            if np.isnan(md) or md < 0:
                out.append(("error", f"{name}: enter the Xmas tree measured depth."))
            else:
                last_md = md
            continue
        needs_md = t in ("Tubing", "Casing")
        if np.isnan(md):
            if needs_md:
                out.append(("error", f"{name}: enter the measured depth."))
                continue
        elif md < last_md - 1e-9 or (needs_md and t in ("Tubing", "Casing") and md <= last_md):
            out.append(("error", f"{name}: measured depth must be deeper than the previous entry "
                        f"({last_md:g} ft)."))
        if t == "Tubing":
            if seen_casing:
                out.append(("error", f"{name}: tubing cannot be below casing in the flow path."))
            n_tubing += 1
            if np.isnan(tid) or tid <= 0:
                out.append(("error", f"{name}: enter the tubing inside diameter."))
            if np.isnan(te) or te < 0 or (not np.isnan(tid) and te >= tid / 2):
                out.append(("error", f"{name}: tubing roughness must be between 0 and half the ID."))
            if annulus:
                if np.isnan(tod) or (not np.isnan(tid) and tod <= tid):
                    out.append(("error", f"{name}: annular flow needs a tubing OD larger than the ID."))
                if np.isnan(cid) or (not np.isnan(tod) and cid <= tod):
                    out.append(("error", f"{name}: annular flow needs a casing ID larger than the tubing OD."))
                if np.isnan(_f(r["Tubing Outside Roughness (in)"])) or np.isnan(ce):
                    out.append(("error", f"{name}: annular flow needs tubing outside and casing roughness."))
            elif not np.isnan(tod) and not np.isnan(tid) and tod <= tid:
                out.append(("error", f"{name}: tubing OD must be larger than its ID."))
            last_tub_id = tid
        elif t == "Casing":
            seen_casing = True
            if np.isnan(cid) or cid <= 0:
                out.append(("error", f"{name}: enter the casing inside diameter."))
            if np.isnan(ce) or ce < 0 or (not np.isnan(cid) and ce >= cid / 2):
                out.append(("error", f"{name}: casing roughness must be between 0 and half the ID."))
            if not np.isnan(cid) and not np.isnan(last_tub_id) and cid <= last_tub_id:
                out.append(("warning", f"{name}: casing ID is not larger than the tubing ID above it."))
        else:  # point component
            if n_tubing == 0:
                out.append(("error", f"{name}: place {t} below a tubing entry."))
            if np.isnan(tid) or tid <= 0:
                out.append(("error", f"{name}: enter the minimum inside diameter."))
            if not np.isnan(tid) and not np.isnan(last_tub_id) and tid > last_tub_id:
                out.append(("warning", f"{name}: ID {tid:g} in is larger than the tubing ID {last_tub_id:g} in."))
        if not np.isnan(md):
            last_md = max(last_md, md)
        rm = _f(r.get("Rate Multiplier"))
        if not np.isnan(rm) and rm <= 0:
            out.append(("error", f"{name}: rate multiplier must be positive."))
    if n_tubing == 0:
        out.append(("error", "Enter at least one tubing section."))
    if survey_max_md is not None and np.isfinite(last_md) and last_md > survey_max_md + 1e-6:
        out.append(("error", f"The deepest equipment ({last_md:,.0f} ft MD) is below the end of the deviation "
                    f"survey ({survey_max_md:,.0f} ft MD). Extend the survey."))
    return out


def downhole_sections(df, md_survey=None, tvd_survey=None):
    """Flow sections (tubing/casing) and point components, top down. Point components without an MD sit
    at the previous depth."""
    secs, points, top, last_tub = [], [], None, None
    for r in downhole_rows(df):
        t, md = r["Type"], _f(r["Measured Depth (ft)"])
        if t == "Xmas Tree":
            top = md
            continue
        if t in ("Tubing", "Casing"):
            start = secs[-1]["MD bottom (ft)"] if secs else top
            rec = {"Type": t, "Label": r["Label"] if isinstance(r["Label"], str) and r["Label"].strip() else t, "MD top (ft)": start, "MD bottom (ft)": md,
                   "Length (ft)": md - start,
                   "Flow ID (in)": _f(r["Tubing Inside Diameter (in)"]) if t == "Tubing"
                   else _f(r["Casing Inside Diameter (in)"]),
                   "Roughness (in)": _f(r["Tubing Inside Roughness (in)"]) if t == "Tubing"
                   else _f(r["Casing Inside Roughness (in)"]),
                   "Tubing OD (in)": _f(r["Tubing Outside Diameter (in)"]) if t == "Tubing" else np.nan,
                   "Casing ID (in)": _f(r["Casing Inside Diameter (in)"]),
                   "Rate multiplier": _f(r.get("Rate Multiplier")) if not np.isnan(_f(r.get("Rate Multiplier"))) else 1.0}
            secs.append(rec)
            if t == "Tubing":
                last_tub = rec
        else:
            at = md if not np.isnan(md) else (secs[-1]["MD bottom (ft)"] if secs else top)
            d = _f(r["Tubing Inside Diameter (in)"])
            if np.isnan(d) and last_tub is not None:
                d = last_tub["Flow ID (in)"]
            points.append({"Type": t, "Label": r["Label"] if isinstance(r["Label"], str) and r["Label"].strip() else t, "MD (ft)": at, "ID (in)": d})
    secs, points = pd.DataFrame(secs), pd.DataFrame(points)
    if md_survey is not None and len(secs):
        secs["TVD bottom (ft)"] = tvd_at(md_survey, tvd_survey, secs["MD bottom (ft)"])
    if md_survey is not None and len(points):
        points["TVD (ft)"] = tvd_at(md_survey, tvd_survey, points["MD (ft)"])
    return top, secs, points


# --------------------------------------------------------- geothermal gradient
GEO_COLUMNS = ["Formation Measured Depth (ft)", "Formation Temperature (deg F)"]


def geo_arrays(df):
    d = df[GEO_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)


def check_geothermal(md, temp, u, top_md=None, bottom_md=None, survey_max_md=None):
    out = []
    if np.isnan(md).any() or np.isnan(temp).any():
        out.append(("error", "Every point needs a measured depth and a temperature."))
        return out
    if len(md) < 2:
        out.append(("error", "Enter at least two points (wellhead and reservoir)."))
        return out
    if np.any(np.diff(md) <= 0):
        out.append(("error", "Measured depth must increase down the table."))
        return out
    if u is None or u <= 0:
        out.append(("error", "The overall heat transfer coefficient must be positive."))
    if np.any(temp < -60) or np.any(temp > 700):
        out.append(("warning", "Some temperatures are outside -60 to 700 °F."))
    if top_md is not None and md[0] > top_md + 1e-6:
        out.append(("error", f"The profile starts at {md[0]:,.1f} ft MD, below the Xmas tree ({top_md:,.1f} ft)."))
    if bottom_md is not None and md[-1] < bottom_md - 1e-6:
        out.append(("error", f"The profile ends at {md[-1]:,.0f} ft MD, above the deepest equipment "
                    f"({bottom_md:,.0f} ft)."))
    if survey_max_md is not None and md[-1] > survey_max_md + 1e-6:
        out.append(("error", f"The profile goes below the end of the deviation survey ({survey_max_md:,.0f} ft MD)."))
    if np.any(np.diff(temp) < 0):
        out.append(("warning", "Temperature decreases with depth somewhere - check the profile."))
    return out


def geothermal_table(md, temp, md_survey, tvd_survey):
    tvd = tvd_at(md_survey, tvd_survey, md)
    grad = np.concatenate([[np.nan], np.diff(temp) / np.maximum(np.diff(tvd), 1e-9) * 100])
    return pd.DataFrame({"Formation TVD (ft)": tvd, "Formation Measured Depth (ft)": md,
                         "Formation Temperature (deg F)": temp, "Gradient (°F/100 ft TVD)": grad})


def temperature_at(md_geo, temp_geo, md):
    return np.interp(md, md_geo, temp_geo)


# ----------------------------------------------------------- heat capacities
@dataclass
class HeatCapacities:
    oil: float = 0.53     # BTU/lb/F
    gas: float = 0.51
    water: float = 1.0

    def check(self):
        out = []
        for name, v, lo, hi in (("Oil", self.oil, 0.3, 0.8), ("Gas", self.gas, 0.4, 0.9), ("Water", self.water, 0.9, 1.1)):
            if v <= 0:
                out.append(("error", f"{name} heat capacity must be positive."))
            elif not lo <= v <= hi:
                out.append(("warning", f"{name} heat capacity {v:g} BTU/lb/F is outside the usual {lo}-{hi} range."))
        return out


# ------------------------------------------------------------ example data
def example_long_survey(n=150, md_total=10963.0, kop=2500.0, build_end=5000.0, hold_angle=22.5, seed=3):
    """Synthetic J-shaped survey with gauge noise: vertical to KOP, build to hold_angle, hold.
    TVD at total depth is about 10,443 ft for the defaults."""
    rng = np.random.default_rng(seed)
    md = np.linspace(0, md_total, n)
    inc = np.where(md < kop, 0.3, np.where(md < build_end, 0.3 + (hold_angle - 0.3) * (md - kop) / (build_end - kop),
                                           hold_angle))
    inc = np.clip(inc + rng.normal(0, 0.15, n), 0, 89)
    tvd = np.concatenate([[0.0], np.cumsum(np.diff(md) * np.cos(np.radians(0.5 * (inc[1:] + inc[:-1]))))])
    return pd.DataFrame({SURVEY_COLUMNS[0]: np.round(md, 1), SURVEY_COLUMNS[1]: np.round(tvd, 1)})
