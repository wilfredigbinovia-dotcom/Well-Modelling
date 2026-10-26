"""Matplotlib figures for each step of the workflow."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .correlations import LABELS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STATUS = {"pass": ("#0ca30c", "o", "pass"), "warn": ("#fab219", "^", "warn"),
          "fail": ("#d03b3b", "X", "fail")}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2,
        "font.size": 9.5, "figure.dpi": 110,
    })


style()


def _legend(ax, **kw):
    ax.legend(labelcolor=INK2, **kw)


# ------------------------------------------------------------------- QC
def plot_qc(report, ax=None):
    t = report.table
    ax = ax or plt.subplots(figsize=(8, 3.6))[1]
    x = t["date"] if "date" in t else np.arange(len(t))
    ok = t["status"] != "fail"
    if ok.sum() >= 2:
        xn = np.arange(len(t), dtype=float)
        if "date" in t:
            xn = x.map(lambda d: d.toordinal()).to_numpy(float)
        c = np.polyfit(xn[ok], t.loc[ok, "pi_implied"], 1)
        ax.plot(x, np.polyval(c, xn), color=INK2, lw=1, ls="--", label="trend (accepted tests)")
    for s, (col, mk, lab) in STATUS.items():
        sel = t["status"] == s
        if sel.any():
            ax.scatter(x[sel], t.loc[sel, "pi_implied"], color=col, marker=mk, s=60,
                       edgecolor=SURFACE, linewidth=1.5, label=lab, zorder=3)
    for i, r in t[t["status"] != "pass"].iterrows():
        ax.annotate(r["test_id"], (x.iloc[i] if hasattr(x, "iloc") else x[i], r["pi_implied"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
    ax.set_title("Test QC: PI implied by each test")
    ax.set_ylabel("Implied PI (STB/d/psi)")
    _legend(ax, loc="upper left", ncol=4)
    return ax


# ---------------------------------------------------------------- match
def plot_match(match, sigma_p: float = 30.0):
    r = match.residuals
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (m, c, unit, title) in zip(axes, [
            ("gauge_meas", "gauge_calc", "psia", "Gauge pressure"),
            ("rate_meas", "rate_ipr", "STB/d", "Liquid rate (IPR at VLP Pwf)")]):
        lo = min(r[m].min(), r[c].min()) * 0.95
        hi = max(r[m].max(), r[c].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1)
        if m == "gauge_meas":
            ax.fill_between([lo, hi], [lo - sigma_p, hi - sigma_p], [lo + sigma_p, hi + sigma_p],
                            color=GRID, alpha=0.7, lw=0, label=f"±{sigma_p:.0f} psi")
        used = r["weight"] > 0
        ax.scatter(r.loc[used, m], r.loc[used, c], color=SERIES[0], s=40,
                   edgecolor=SURFACE, lw=1.2, label="used in match", zorder=3)
        if (~used).any():
            ax.scatter(r.loc[~used, m], r.loc[~used, c], color=STATUS["fail"][0], marker="X",
                       s=55, edgecolor=SURFACE, lw=1.2, label="excluded by QC", zorder=3)
            for _, row in r[~used].iterrows():
                ax.annotate(row["test_id"], (row[m], row[c]), textcoords="offset points",
                            xytext=(6, -10), fontsize=8, color=INK)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Measured ({unit})"); ax.set_ylabel(f"Model ({unit})")
        ax.set_title(title)
        _legend(ax, loc="upper left")
    fig.suptitle(f"Multi-test match - {LABELS[match.correlation]}", fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ selection
def plot_selection(sel):
    t = sel.table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    y = np.arange(len(t))
    col = [SERIES[0] if ok else "#b9b8b3" for ok in t["in_range"]]
    ax.barh(y, t["loocv_gauge_rmse"], color=col, height=0.55)
    for yi, (_, r) in zip(y, t.iterrows()):
        note = (f"gravity {r['gravity_mult']:.2f}, friction {r['friction_mult']:.2f}"
                + ("" if r["in_range"] else "  - out of range"))
        ax.text(r["loocv_gauge_rmse"] + 0.5, yi, f"{r['loocv_gauge_rmse']:.0f} psi   ({note})",
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, t["label"])
    ax.set_xlim(0, t["loocv_gauge_rmse"].max() * 2.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Leave-one-out gauge RMSE (psi) - lower is better")
    ax.set_title("Correlation selection (blue = multipliers in acceptable range)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- nodal
def plot_nodal(models, pr, whp, wc, gor, measured=None, ax=None, title=None):
    """IPR with one VLP curve per model. `models` is a WellModel or a list of
    them (e.g. each correlation with its *own* matched parameters).
    `measured` = (rate, pwf) plots a test point for comparison."""
    models = models if isinstance(models, (list, tuple)) else [models]
    ax = ax or plt.subplots(figsize=(7.5, 4.6))[1]
    ref = models[0]
    q_ipr, p_ipr = ref.ipr_curve(pr)
    ax.plot(q_ipr, p_ipr, color=INK, label="IPR")
    qmax = float(ref.aof(pr))
    for i, mm in enumerate(models):
        col = SERIES[i % 4]
        q, p = mm.vlp_curve(whp, wc, gor, qmax, n=30)
        ax.plot(q, p, color=col, label=f"VLP - {LABELS[mm.correlation]} "
                f"(g {mm.gravity_mult:.2f}, f {mm.friction_mult:.2f})")
        op = mm.operating_point(pr, whp, wc, gor)
        if op.flowing[0]:
            ax.scatter(op.rate, op.pwf, s=60, color=col, edgecolor=SURFACE, lw=2, zorder=4)
    if measured is not None:
        ax.scatter([measured[0]], [measured[1]], s=90, marker="D", facecolor="none",
                   edgecolor=INK, lw=1.5, zorder=5, label="measured test")
    ax.set_xlim(0, qmax); ax.set_ylim(0, pr * 1.05)
    ax.set_xlabel("Liquid rate (STB/d)"); ax.set_ylabel("Pwf (psia)")
    ax.set_title(title or f"Nodal analysis: Pr {pr:.0f} psia, WHP {whp:.0f} psia, WC {wc:.0%}")
    _legend(ax, loc="lower left")
    return ax


# -------------------------------------------------------- probabilistic
def plot_monte_carlo(mc):
    s = mc.samples
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    q = np.sort(s["oil_rate"].to_numpy())
    exceed = 1 - np.arange(q.size) / q.size
    ax.plot(q, exceed * 100, color=SERIES[0])
    oil = mc.summary.loc["oil_rate"]
    for lab in ("P90", "P50", "P10"):
        v = oil[lab]
        ax.axvline(v, color=INK2, lw=1, ls=":")
        ax.text(v, 102, f"{lab}\n{v:,.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Oil rate (STB/d)"); ax.set_ylabel("Probability of exceeding (%)")
    ax.set_title("Oil-rate exceedance curve", pad=26)
    ax = axes[1]
    tor = mc.tornado
    if tor is not None:
        base = tor.attrs["base_rate"]
        t = tor.iloc[::-1]
        y = np.arange(len(t))
        lo = np.minimum(t["rate_at_low"], t["rate_at_high"]) - base
        hi = np.maximum(t["rate_at_low"], t["rate_at_high"]) - base
        ax.barh(y, hi.clip(lower=0), left=0, color=SERIES[0], height=0.55, label="increase")
        ax.barh(y, lo.clip(upper=0), left=0, color=SERIES[1], height=0.55, label="decrease")
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(y, t["variable"].str.replace("_", " "))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"Change in liquid rate from base {base:,.0f} STB/d")
        ax.set_title("Tornado (each input at its P90 / P10)")
        _legend(ax, loc="lower right")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ transient
def plot_buildup(dt, pws, fit, tp, q):
    from .transient import bourdet_derivative
    dp = pws - pws[0]
    res = fit.reservoir
    pm = res.buildup(dt, tp, q, fit.p_init)
    dpm = pm - pm[0]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.loglog(dt, dp, "o", ms=4, color=SERIES[0], label="Δp measured")
    ax.loglog(dt, bourdet_derivative(dt, dp), "o", ms=4, color=SERIES[1], label="derivative measured")
    ax.loglog(dt, dpm, color=SERIES[0], lw=1.5, label="Δp model")
    ax.loglog(dt, bourdet_derivative(dt, dpm), color=SERIES[1], lw=1.5, label="derivative model")
    ax.set_xlabel("Shut-in time Δt (hr)"); ax.set_ylabel("Δp, derivative (psi)")
    ax.set_title(f"Build-up diagnostic: k = {res.k:.1f} md, skin = {res.skin:.2f}, "
                 f"C = {res.C:.3f} bbl/psi")
    ax.grid(True, which="both", color=GRID, lw=0.5)
    _legend(ax, loc="lower right")
    return fig


def plot_forecast(fr, pb=None):
    d = fr.table
    ev = fr.events
    t = d["t_days"] / 365.25
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    a = axes[0, 0]
    a.plot(t, d["liquid_rate"], color=SERIES[0], label="liquid")
    a.plot(t, d["oil_rate"], color=SERIES[1], label="oil")
    a.set_ylabel("Rate (STB/d)"); a.set_title("Rates"); _legend(a)
    a = axes[0, 1]
    a.plot(t, d["reservoir_pressure"], color=SERIES[0], label="reservoir pressure")
    a.plot(t, d["pwf"], color=SERIES[1], label="flowing BHP")
    if pb:
        a.axhline(pb, color=INK2, lw=1, ls=":")
        a.text(t.iloc[-1], pb, " Pb", va="bottom", ha="right", fontsize=8.5, color=INK2)
    a.set_ylabel("Pressure (psia)"); a.set_title("Pressures"); _legend(a)
    a = axes[1, 0]
    a.plot(t, d["water_cut"] * 100, color=SERIES[0])
    a.set_ylabel("Water cut (%)"); a.set_title("Water cut"); a.set_xlabel("Years")
    a = axes[1, 1]
    a.plot(t, d["gor"], color=SERIES[0])
    a.set_ylabel("GOR (scf/STB)"); a.set_title("Producing GOR"); a.set_xlabel("Years")
    if ev.get("flow_stops_days"):
        ts = ev["flow_stops_days"] / 365.25
        for a in axes.ravel():
            a.axvline(ts, color=STATUS["fail"][0], lw=1.2, ls="--")
        axes[0, 0].text(ts, axes[0, 0].get_ylim()[1] * 0.92, " natural flow stops",
                        color=INK, fontsize=8.5, ha="right")
    fig.tight_layout()
    return fig


def plot_card_reference(card, ax=None, title=None):
    """One schematic dynamometer card of wellmodel.srp_cards (load against position, normalised)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(3.4, 2.5))
    fig = ax.figure
    ax.plot(card.x, card.y, color=SERIES[0], lw=2.0, solid_joinstyle="round")
    off = {"A": (-0.05, -0.08), "B": (-0.05, 0.05), "C": (0.03, 0.05), "D": (0.03, -0.08)}
    for k, (x, y) in card.labels.items():
        dx, dy = off[k]
        ax.text(x + dx, y + dy, k, fontsize=9, fontweight="bold", color=INK, ha="center", va="center")
    ymin = min(-0.2, float(np.min(card.y)) - 0.12)
    y_mpt, y_ept = ymin - 0.12, 0.5
    ax.annotate("", (0.0, y_mpt), (1.0, y_mpt), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
    ax.text(0.5, y_mpt - 0.04, "MPT", fontsize=8, color=INK2, ha="center", va="top")
    a, b = card.ept
    ax.annotate("", (a, y_ept), (b, y_ept), arrowprops=dict(arrowstyle="<->", color=SERIES[1], lw=1.1))
    ax.text(0.5 * (a + b), y_ept + 0.04, "EPT", fontsize=8, color=SERIES[1], ha="center", va="bottom")
    if card.fo:
        ax.annotate("", (1.12, 0.0), (1.12, 1.0), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
        ax.text(1.15, 0.5, "Fo", fontsize=8, color=INK2, va="center")
    ax.set_xlim(-0.12, 1.25)
    ax.set_ylim(y_mpt - 0.2, max(1.2, float(np.max(card.y)) + 0.15))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8.5, color=INK2)
    return fig
