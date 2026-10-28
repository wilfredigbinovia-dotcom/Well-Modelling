"""Illustrated well schematic (SVG) built from the downhole equipment.

Not to scale: depths are spread so every component stays readable, but the order and relative spacing
follow measured depth. Labels give the measured depth of each item.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

W, H = 420, 940
CX = 165                       # well centre line
GROUND = 186                   # ground level (y)
CASING_HW, TUBING_HW = 40, 12  # half widths (px)
LABEL_X = 232


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _depth_map(mds, y0, y1, min_gap=44):
    """Monotone MD -> y map: half linear, half evenly spaced, with a minimum gap between features."""
    mds = np.unique(np.asarray([m for m in mds if m is not None and np.isfinite(m)], float))
    if len(mds) == 1:
        return lambda m: np.full_like(np.asarray(m, float), (y0 + y1) / 2)
    lin = (mds - mds[0]) / (mds[-1] - mds[0])
    rank = np.linspace(0, 1, len(mds))
    y = y0 + (y1 - y0) * (0.55 * lin + 0.45 * rank)
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + min_gap)
    if y[-1] > y1:                                    # squeeze back into the frame
        y = y0 + (y - y0) * (y1 - y0) / (y[-1] - y0)
    return lambda m: np.interp(m, mds, y)


def _defs():
    rng = np.random.default_rng(7)
    dots = "".join(f'<circle cx="{rng.uniform(0, 40):.1f}" cy="{rng.uniform(0, 40):.1f}" r="{rng.uniform(0.4, 1.1):.1f}" '
                   f'fill="#000" opacity="{rng.uniform(0.05, 0.16):.2f}"/>' for _ in range(28))
    sand = "".join(f'<circle cx="{rng.uniform(0, 24):.1f}" cy="{rng.uniform(0, 24):.1f}" r="{rng.uniform(0.6, 1.4):.1f}" '
                   f'fill="#7a6326" opacity="0.45"/>' for _ in range(14))
    return f"""
<defs>
  <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#5f7d95"/><stop offset="0.6" stop-color="#a9bccb"/><stop offset="1" stop-color="#dde5ea"/>
  </linearGradient>
  <linearGradient id="metal" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#5d6469"/><stop offset="0.35" stop-color="#f3f5f6"/>
    <stop offset="0.6" stop-color="#b9bfc4"/><stop offset="1" stop-color="#4f555a"/>
  </linearGradient>
  <linearGradient id="bore" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#9da3a8"/><stop offset="0.3" stop-color="#e9ecee"/>
    <stop offset="0.7" stop-color="#d3d7da"/><stop offset="1" stop-color="#8d9398"/>
  </linearGradient>
  <linearGradient id="tool" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#8a7a63"/><stop offset="0.4" stop-color="#efe6d6"/>
    <stop offset="0.7" stop-color="#cbbda5"/><stop offset="1" stop-color="#7a6b55"/>
  </linearGradient>
  <linearGradient id="bronze" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#4a2a17"/><stop offset="0.45" stop-color="#9a6a45"/><stop offset="1" stop-color="#3d2212"/>
  </linearGradient>
  <linearGradient id="earth" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#c9a87c"/><stop offset="0.45" stop-color="#ad8e69"/>
    <stop offset="0.8" stop-color="#8a7258"/><stop offset="1" stop-color="#5e4e3e"/>
  </linearGradient>
  <pattern id="grain" width="40" height="40" patternUnits="userSpaceOnUse">{dots}</pattern>
  <pattern id="sand" width="24" height="24" patternUnits="userSpaceOnUse">
    <rect width="24" height="24" fill="#d9c48d"/>{sand}</pattern>
  <pattern id="gravel" width="8" height="8" patternUnits="userSpaceOnUse">
    <rect width="8" height="8" fill="#b9ab8c"/><circle cx="2" cy="2" r="1.6" fill="#8d7f62"/>
    <circle cx="6" cy="6" r="1.6" fill="#a39373"/></pattern>
  <pattern id="wire" width="6" height="3" patternUnits="userSpaceOnUse">
    <rect width="6" height="3" fill="#c3c8cc"/><line x1="0" y1="1.5" x2="6" y2="1.5" stroke="#6b7177" stroke-width="1"/></pattern>
  <filter id="soft"><feGaussianBlur stdDeviation="6"/></filter>
</defs>"""


def _landscape():
    out = [f'<rect x="20" y="40" width="{W - 40}" height="{GROUND - 40}" fill="url(#sky)"/>']
    for cx, cy, rx, ry, op in ((90, 80, 60, 14, .55), (250, 66, 80, 16, .5), (330, 98, 50, 10, .45), (170, 108, 70, 9, .4)):
        out.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#fff" opacity="{op}" filter="url(#soft)"/>')
    out.append(f'<path d="M20 150 Q60 128 100 146 T180 140 T260 132 T340 146 T{W - 20} 140 L{W - 20} 166 L20 166 Z" '
               f'fill="#55604a" opacity="0.85"/>')
    out.append(f'<rect x="20" y="160" width="{W - 40}" height="14" fill="#6d7a45"/>')
    out.append(f'<rect x="20" y="172" width="{W - 40}" height="{GROUND - 172}" fill="#cfc8a2"/>')
    return out


def _earth(y_res_top, y_res_bot):
    out = [f'<rect x="20" y="{GROUND}" width="{W - 40}" height="{H - 20 - GROUND}" fill="url(#earth)"/>']
    rng = np.random.default_rng(11)
    ys = np.linspace(GROUND + 70, y_res_top - 20, 5)
    for i, y in enumerate(ys):
        a, ph = rng.uniform(4, 9), rng.uniform(0, 6)
        xs = np.linspace(20, W - 20, 30)
        pts = " ".join(f"{x:.1f},{y + a * np.sin(x / 45 + ph):.1f}" for x in xs)
        out.append(f'<polyline points="{pts}" fill="none" stroke="#5a4835" stroke-width="{1.2 + 0.3 * i:.1f}" '
                   f'opacity="0.25"/>')
    out.append(f'<rect x="20" y="{GROUND}" width="{W - 40}" height="{H - 20 - GROUND}" fill="url(#grain)"/>')
    # reservoir band
    out.append(f'<rect x="20" y="{y_res_top}" width="{W - 40}" height="{y_res_bot - y_res_top}" fill="url(#sand)" '
               f'opacity="0.95"/>')
    out.append(f'<line x1="20" y1="{y_res_top}" x2="{W - 20}" y2="{y_res_top}" stroke="#6e5a2a" stroke-width="1.2"/>')
    out.append(f'<line x1="20" y1="{y_res_bot}" x2="{W - 20}" y2="{y_res_bot}" stroke="#6e5a2a" stroke-width="1.2"/>')
    out.append(f'<text x="28" y="{y_res_top + 15}" font-size="10.5" fill="#4a3c1a" font-weight="600" '
               f'font-style="italic">Reservoir</text>')
    return out


def _xmas_tree(y_top, has_choke):
    """Wellhead and tree sitting on the ground, flowline to the right."""
    x, y = CX, GROUND
    out = [f'<rect x="{x - 46}" y="{y - 6}" width="92" height="8" rx="2" fill="url(#bronze)"/>',   # flange
           f'<rect x="{x - 13}" y="{y - 52}" width="26" height="48" fill="url(#bronze)"/>',         # master valves
           f'<rect x="{x - 9}" y="{y - 64}" width="18" height="13" rx="2" fill="url(#bronze)"/>',    # swab
           f'<rect x="{x - 4}" y="{y - 74}" width="8" height="11" fill="#3d2212"/>']                 # cap
    for yy in (y - 42, y - 20):
        for sgn in (-1, 1):
            hx = x + sgn * 26
            out.append(f'<line x1="{x + sgn * 13}" y1="{yy}" x2="{hx}" y2="{yy}" stroke="#4a2a17" stroke-width="3"/>')
            out.append(f'<circle cx="{hx}" cy="{yy}" r="8" fill="none" stroke="#5a3520" stroke-width="2.4"/>')
            out.append(f'<line x1="{hx - 8}" y1="{yy}" x2="{hx + 8}" y2="{yy}" stroke="#5a3520" stroke-width="1.2"/>')
            out.append(f'<line x1="{hx}" y1="{yy - 8}" x2="{hx}" y2="{yy + 8}" stroke="#5a3520" stroke-width="1.2"/>')
    fy = y - 31
    out.append(f'<rect x="{x + 13}" y="{fy - 4}" width="{W - 20 - x - 13}" height="8" fill="url(#metal)"/>')
    if has_choke:
        cx_ = x + 74
        out.append(f'<path d="M{cx_ - 9} {fy - 9} L{cx_ + 9} {fy + 9} L{cx_ + 9} {fy - 9} L{cx_ - 9} {fy + 9} Z" '
                   f'fill="#7b2d26" stroke="#3d1410" stroke-width="1"/>')
        out.append(f'<text x="{cx_}" y="{fy - 13}" text-anchor="middle" font-size="9.5" fill="#20262b">choke</text>')
    out.append(f'<text x="{W - 26}" y="{fy - 8}" text-anchor="end" font-size="9.5" fill="#20262b">to manifold →</text>')
    return out


def _label(y, text, sub, x_from):
    return [f'<line x1="{x_from}" y1="{y:.1f}" x2="{LABEL_X - 4}" y2="{y:.1f}" stroke="#20262b" stroke-width="0.8" '
            f'stroke-dasharray="2 2"/>',
            f'<rect x="{LABEL_X - 2}" y="{y - 13:.1f}" width="{W - 24 - LABEL_X}" height="26" rx="3" fill="#ffffff" '
            f'opacity="0.82"/>',
            f'<text x="{LABEL_X + 3}" y="{y - 1:.1f}" font-size="10.5" font-weight="600" fill="#1d2327">{_esc(text)}</text>',
            f'<text x="{LABEL_X + 3}" y="{y + 10:.1f}" font-size="9.5" fill="#3d454c">{_esc(sub)}</text>']


def _component(kind, y):
    x = CX
    if kind == "SSSV":
        return [f'<rect x="{x - 19}" y="{y - 17:.1f}" width="38" height="34" rx="4" fill="url(#tool)" stroke="#5a4c3a" '
                f'stroke-width="0.8"/>',
                f'<rect x="{x - 19}" y="{y - 3:.1f}" width="38" height="6" fill="#b02a22"/>',
                f'<line x1="{x - 15}" y1="{y - 12:.1f}" x2="{x + 15}" y2="{y - 12:.1f}" stroke="#5a4c3a" stroke-width="0.6"/>',
                f'<line x1="{x - 15}" y1="{y + 12:.1f}" x2="{x + 15}" y2="{y + 12:.1f}" stroke="#5a4c3a" stroke-width="0.6"/>']
    if kind == "Gas Lift Mandrel":
        return [f'<rect x="{x - 17}" y="{y - 24:.1f}" width="34" height="48" rx="5" fill="url(#tool)" stroke="#5a4c3a" '
                f'stroke-width="0.8"/>',
                f'<rect x="{x - 29}" y="{y - 16:.1f}" width="14" height="32" rx="4" fill="url(#tool)" stroke="#5a4c3a" '
                f'stroke-width="0.8"/>',
                f'<line x1="{x - 22}" y1="{y - 10:.1f}" x2="{x - 22}" y2="{y + 10:.1f}" stroke="#5a4c3a" stroke-width="1.4"/>',
                f'<line x1="{x + 4}" y1="{y - 18:.1f}" x2="{x + 4}" y2="{y + 18:.1f}" stroke="#7d6e58" stroke-width="0.8"/>']
    if kind == "SSD":
        out = [f'<rect x="{x - 17}" y="{y - 16:.1f}" width="34" height="32" rx="3" fill="url(#tool)" stroke="#5a4c3a" '
               f'stroke-width="0.8"/>']
        for dx in (-9, -3, 3, 9):
            out.append(f'<rect x="{x + dx - 1.2}" y="{y - 8:.1f}" width="2.4" height="16" fill="#3a3128"/>')
        return out
    # restriction / nipple
    return [f'<rect x="{x - 16}" y="{y - 7:.1f}" width="32" height="14" rx="2" fill="url(#tool)" stroke="#5a4c3a" '
            f'stroke-width="0.8"/>',
            f'<path d="M{x - 8} {y - 7:.1f} L{x - 4} {y:.1f} L{x - 8} {y + 7:.1f} M{x + 8} {y - 7:.1f} L{x + 4} {y:.1f} '
            f'L{x + 8} {y + 7:.1f}" fill="none" stroke="#5a4c3a" stroke-width="1"/>']


def well_schematic_svg(top_md, secs: pd.DataFrame, points: pd.DataFrame, completion="cased_hole",
                       sand_control="none", has_choke=False, title="WELL SCHEMATIC") -> str:
    secs = secs if secs is not None else pd.DataFrame()
    points = points if points is not None else pd.DataFrame()
    tub = secs[secs["Type"] == "Tubing"] if len(secs) else secs
    cas = secs[secs["Type"] == "Casing"] if len(secs) else secs
    top = float(top_md) if top_md is not None and np.isfinite(top_md) else 0.0
    tub_end = float(tub["MD bottom (ft)"].max()) if len(tub) else top
    bottom = float(secs["MD bottom (ft)"].max()) if len(secs) else tub_end
    features = [top, tub_end, bottom] + list(cas["MD bottom (ft)"] if len(cas) else []) \
        + list(points["MD (ft)"] if len(points) else [])
    y_res_bot = H - 70
    y_res_top = y_res_bot - 95
    ymap = _depth_map(features, GROUND + 50, y_res_top + 55)
    y_tub = float(ymap(tub_end))
    y_bot = max(float(ymap(bottom)), y_res_top + 55)
    y_well_end = y_res_bot - 8

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
           f'style="max-width:100%;height:auto;font-family:Segoe UI,Helvetica,Arial,sans-serif">',
           _defs(),
           f'<rect width="{W}" height="{H}" rx="6" fill="#dfe1e3"/>',
           f'<rect x="12" y="32" width="{W - 24}" height="{H - 44}" fill="#f4f5f6" stroke="#b8bcc0"/>',
           f'<rect x="{W / 2 - 62}" y="8" width="124" height="18" fill="#111"/>',
           f'<text x="{W / 2}" y="21" text-anchor="middle" font-size="11" font-weight="700" fill="#e33" '
           f'letter-spacing="0.6">{_esc(title)}</text>']
    out += _landscape()
    out += _earth(y_res_top, y_res_bot)

    open_hole = completion == "open_hole"
    # casing: from surface to the last casing shoe (or above the reservoir for open hole)
    shoe_md = float(cas["MD bottom (ft)"].max()) if len(cas) else bottom
    y_shoe = float(ymap(shoe_md))
    if open_hole:
        y_shoe = min(y_shoe, y_res_top - 6)
    else:
        y_shoe = max(y_shoe, y_well_end)
    out.append(f'<rect x="{CX - CASING_HW - 4}" y="{GROUND}" width="{2 * CASING_HW + 8}" height="{y_shoe - GROUND:.1f}" '
               f'fill="#6f757a"/>')
    out.append(f'<rect x="{CX - CASING_HW}" y="{GROUND}" width="{2 * CASING_HW}" height="{y_shoe - GROUND:.1f}" '
               f'fill="url(#bore)"/>')
    for sgn in (-1, 1):                                                       # casing shoe
        xs = CX + sgn * (CASING_HW + 4)
        out.append(f'<path d="M{xs} {y_shoe - 9:.1f} L{xs} {y_shoe:.1f} L{xs - sgn * 9} {y_shoe:.1f} Z" fill="#2f3438"/>')
    if open_hole:                                                             # open-hole section below the shoe
        ys = np.linspace(y_shoe, y_well_end, 18)
        rng = np.random.default_rng(5)
        wl = [CX - CASING_HW + 4 - rng.uniform(0, 5) for _ in ys]
        wr = [CX + CASING_HW - 4 + rng.uniform(0, 5) for _ in ys]
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(wl, ys)) + " " + \
            " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(reversed(wr), reversed(ys)))
        out.append(f'<polygon points="{pts}" fill="#5b4a37" opacity="0.55"/>')
    else:                                                                     # perforations
        for yy in np.linspace(y_res_top + 14, y_res_bot - 16, 6):
            for sgn in (-1, 1):
                x0 = CX + sgn * (CASING_HW + 4)
                out.append(f'<path d="M{x0} {yy - 3:.1f} L{x0 + sgn * 26} {yy:.1f} L{x0} {yy + 3:.1f} Z" fill="#3a2c1d" '
                           f'opacity="0.85"/>')

    # sand control across the reservoir
    if sand_control in ("gravel_pack", "pre_packed_screen"):
        out.append(f'<rect x="{CX - CASING_HW + 3}" y="{y_res_top + 4}" width="{2 * CASING_HW - 6}" '
                   f'height="{y_res_bot - y_res_top - 12}" fill="url(#gravel)" opacity="0.95"/>')
    if sand_control in ("gravel_pack", "pre_packed_screen", "wire_wrapped_screen"):
        out.append(f'<rect x="{CX - TUBING_HW - 5}" y="{y_res_top + 4}" width="{2 * TUBING_HW + 10}" '
                   f'height="{y_res_bot - y_res_top - 12}" fill="url(#wire)" stroke="#555b60" stroke-width="0.8"/>')
    elif sand_control == "slotted_liner":
        out.append(f'<rect x="{CX - TUBING_HW - 5}" y="{y_res_top + 4}" width="{2 * TUBING_HW + 10}" '
                   f'height="{y_res_bot - y_res_top - 12}" fill="url(#metal)" stroke="#555b60" stroke-width="0.8"/>')
        for yy in np.arange(y_res_top + 10, y_res_bot - 12, 9):
            for dx in (-10, 0, 10):
                out.append(f'<rect x="{CX + dx - 1}" y="{yy:.1f}" width="2" height="6" fill="#2e3236"/>')

    # tubing
    if len(tub):
        out.append(f'<rect x="{CX - TUBING_HW}" y="{GROUND - 6}" width="{2 * TUBING_HW}" height="{y_tub - GROUND + 6:.1f}" '
                   f'fill="url(#metal)" stroke="#4d5358" stroke-width="0.6"/>')
        for yy in np.arange(GROUND + 40, y_tub - 10, 70):                     # couplings
            out.append(f'<rect x="{CX - TUBING_HW - 2}" y="{yy:.1f}" width="{2 * TUBING_HW + 4}" height="5" '
                       f'fill="url(#metal)" stroke="#4d5358" stroke-width="0.4"/>')
        out.append(f'<path d="M{CX - TUBING_HW} {y_tub:.1f} L{CX + TUBING_HW} {y_tub + 7:.1f} L{CX + TUBING_HW} {y_tub:.1f} Z" '
                   f'fill="#3b4045"/>')                                       # mule shoe

    out += _xmas_tree(GROUND, has_choke)

    # control line to the SSSV
    sssv = points[points["Type"] == "SSSV"] if len(points) else points
    if len(sssv):
        y_s = float(ymap(sssv["MD (ft)"].iloc[0]))
        out.append(f'<path d="M{CX - 13} {GROUND - 40} L{CX - TUBING_HW - 3} {GROUND + 4} L{CX - TUBING_HW - 3} {y_s - 14:.1f}" '
                   f'fill="none" stroke="#b02a22" stroke-width="1.1"/>')

    labels = [(GROUND - 3, "Xmas tree", f"{top:,.1f} ft MD", CX + 46)]
    for _, p in (points.iterrows() if len(points) else []):
        y = float(ymap(p["MD (ft)"]))
        out += _component(p["Type"], y)
        idtxt = f' · ID {p["ID (in)"]:g}"' if pd.notna(p.get("ID (in)")) else ""
        labels.append((y, p["Label"] if p["Label"] != p["Type"] else p["Type"], f'{p["MD (ft)"]:,.1f} ft MD{idtxt}',
                       CX + 30))
    if len(tub):
        t_id = tub["Flow ID (in)"].iloc[-1]
        labels.append((y_tub, "End of tubing", f'{tub_end:,.1f} ft MD · ID {t_id:g}"', CX + TUBING_HW + 2))
    for _, c in (cas.iterrows() if len(cas) else []):
        lbl = c["Label"] if c["Label"] != "Casing" else "Casing shoe"
        labels.append((float(ymap(c["MD bottom (ft)"])) if not open_hole else y_shoe, lbl,
                       f'{c["MD bottom (ft)"]:,.1f} ft MD · ID {c["Flow ID (in)"]:g}"', CX + CASING_HW + 5))
    # spread labels so they never overlap
    labels.sort(key=lambda t: t[0])
    placed = []
    for y, a, b, xf in labels:
        yl = max(y, placed[-1][0] + 30) if placed else y
        placed.append((yl, a, b, xf, y))
    for yl, a, b, xf, y in placed:
        if abs(yl - y) > 0.5:
            out.append(f'<line x1="{xf}" y1="{y:.1f}" x2="{LABEL_X - 10}" y2="{yl:.1f}" stroke="#20262b" '
                       f'stroke-width="0.8" stroke-dasharray="2 2"/>')
            xf = LABEL_X - 10
        out += _label(yl, a, b, xf)

    comp = {"cased_hole": "Cased hole", "open_hole": "Open hole"}.get(completion, completion)
    scl = {"none": "no sand control", "gravel_pack": "gravel pack", "pre_packed_screen": "pre-packed screen",
           "wire_wrapped_screen": "wire-wrapped screen", "slotted_liner": "slotted liner"}.get(sand_control, sand_control)
    out.append(f'<text x="{W / 2}" y="{H - 28}" text-anchor="middle" font-size="10" fill="#f3eee4">'
               f'{comp}, {scl} · depths not to scale</text>')
    out.append("</svg>")
    return "\n".join(out)
