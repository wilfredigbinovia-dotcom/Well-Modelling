"""PROSPER-style gradient results: a node-by-node table of one traverse (pressure, temperature, holdup, flow
regime, gradients, cumulative losses, velocities, mass and volume rates, phase densities, viscosities, interfacial
tensions and formation volume factors).

The traverse itself (P, T at every node) comes from vlp.traverse, so the table is consistent with the pressures
used everywhere else; the extra quantities are evaluated at each node's pressure and temperature with the
geometry of the segment above the node.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import blackoil as bo
from . import correlations as C
from .correlations import CORRELATIONS, NEEDS_GLR, no_slip
from . import vlp as V

LB_PER_SCF_AIR = V.RHO_AIR_SC
CRITICAL_RATIO = 0.53


# ------------------------------------------------------------------ flow regime
def flow_regime(correlation, vsl, vsg, rho_l, rho_g, sig, d_ft, sin_t):
    """Flow pattern at one node. Beggs & Brill family: Beggs & Brill (1973) map. The others: Griffith & Wallis
    bubble-flow limit (gas fraction below 1.071 - 0.2218 vm^2 / d, at least 0.13), then the Duns & Ros (1963)
    dimensionless map for slug / transition / mist."""
    vm = vsl + vsg
    lam = vsl / max(vm, 1e-30)
    if lam >= 0.99999:
        return "Liquid"
    if lam <= 1e-6:
        return "Gas"
    if correlation in ("beggs_brill", "beggs_brill_noslip"):
        fr = vm * vm / (C.G * d_ft)
        l1, l2 = 316 * lam ** 0.302, 0.0009252 * lam ** -2.4684
        l3, l4 = 0.1 * lam ** -1.4516, 0.5 * lam ** -6.738
        if (lam < 0.01 and fr < l1) or (lam >= 0.01 and fr < l2):
            return "Segregated"
        if lam >= 0.01 and l2 <= fr <= l3:
            return "Transition"
        if (0.01 <= lam < 0.4 and l3 < fr <= l1) or (lam >= 0.4 and l3 < fr <= l4):
            return "Intermittent"
        return "Distributed"
    k = (rho_l / max(sig, 1e-6)) ** 0.25
    nlv, ngv = 1.938 * vsl * k, 1.938 * vsg * k
    nd = 120.872 * d_ft * np.sqrt(rho_l / max(sig, 1e-6))
    l1, l2 = float(C._li(np.array([nd]), *C._DR_ND)[0]), float(C._li(np.array([nd]), *C._DR_L2)[0])
    ls, lm = 50 + 36 * nlv, 75 + 84 * nlv ** 0.75
    lb = max(1.071 - 0.2218 * vm * vm / max(d_ft, 1e-6), 0.13)
    if 1 - lam < lb and ngv < l1 + l2 * nlv:
        return "Dispersed bubble" if correlation == "pe5_style" else "Bubble"
    if ngv < ls:
        return "Slug"
    if ngv < lm:
        return "Transition"
    return "Mist"


# ------------------------------------------------------------------ phase detail
def _phases(fluid, p, t, case):
    """Per-phase properties at (p, t) for one case (arrays of length 1). NaN where the fluid model has none."""
    p, t = np.atleast_1d(np.asarray(p, float)), np.atleast_1d(np.asarray(t, float))
    s = fluid.state(p, t, case)
    nan = np.full_like(p, np.nan)
    out = {"ql": s["ql"], "qg": s["qg"], "rho_l": s["rho_l"], "rho_g": s["rho_g"], "mu_l": s["mu_l"],
           "mu_g": s["mu_g"], "sig_l": s["sig"]}
    if isinstance(fluid, V.BlackOilFluid):
        inp = fluid.inp
        q, wc, gor = case["q"], np.clip(case["wc"], 0, 0.9999), np.maximum(case["gor"], 0.0)
        qo, qw = q * (1 - wc), q * wc
        rs, bo_, mu_o = fluid._props(p, t, gor)
        rho_o = (62.428 * inp.gamma_o + 0.0136 * rs * inp.gas_sg) / bo_
        _, bg, rho_g, mu_g = bo.gas_props(inp, p, t)
        bw, rho_w, mu_w = V._water(inp, p, t)
        free = qo * np.maximum(gor - rs, 0.0)                       # scf/d
        out.update(rho_o=rho_o, rho_w=rho_w, mu_o=mu_o, mu_w=mu_w, bo=bo_, bw=bw, bg=bg,
                   sig_go=V._sigma_oil(inp.api, p, t), sig_gw=V._sigma_water(p, t),
                   m_oil=qo * (RB_LB(inp.gamma_o) + LB_PER_SCF_AIR * inp.gas_sg * rs),
                   m_water=qw * RB_LB(V._water_sg(inp)), m_gas=free * LB_PER_SCF_AIR * inp.gas_sg,
                   free_gas=free / 1e6)
    else:
        out.update(rho_o=nan, rho_w=nan, mu_o=nan, mu_w=nan, bo=nan, bw=nan, bg=nan, sig_go=nan, sig_gw=nan,
                   m_oil=nan, m_water=nan, m_gas=s["qg"] * s["rho_g"] * 86400,
                   free_gas=nan)
        if isinstance(fluid, V.WetGasFluid):
            bw, rho_w, mu_w = V._water(fluid.inp, p, t)
            out.update(rho_w=rho_w, mu_w=mu_w, bw=bw, sig_gw=V._sigma_water(p, t),
                       m_water=s["ql"] * s["rho_l"] * 86400)
    return out


def RB_LB(sg):
    """lbm per stock-tank barrel of a liquid of specific gravity sg."""
    return V.RB * 62.428 * sg


def _surface_rates(kind, case):
    q = float(case["q"][0])
    if kind == "oil":
        wc, gor = float(np.clip(case["wc"][0], 0, 1)), float(case["gor"][0])
        return {"Oil Rate (STB/day)": q * (1 - wc), "Water Rate (STB/day)": q * wc, "Liquid Rate (STB/day)": q,
                "Gas Rate (MMscf/day)": q * (1 - wc) * gor / 1e6}
    wgr, cgr = float(case["wgr"][0]), float(case["cgr"][0])
    return {"Gas Rate (MMscf/day)": q, "Condensate Rate (STB/day)": q * cgr, "Water Rate (STB/day)": q * wgr}


def _restriction_split(fluid, case, p_in, t, d_in):
    """Outlet (upstream-of-flow is the inlet, below) pressure and dP of a restriction whose inlet pressure is
    p_in: vlp.traverse adds dP(p_out) to p_out, so solve p_out + dP(p_out) = p_in."""
    p_out = p_in
    for _ in range(30):
        dp = float(V._restriction_dp(fluid.state(np.array([p_out]), np.array([t]), case), d_in)[0])
        new = p_in - dp
        if abs(new - p_out) < 1e-6:
            break
        p_out = new
    return p_out, p_in - p_out


# ------------------------------------------------------------------ table
def gradient_table(path, fluid, case, P, T, correlation, p1=1.0, p2=1.0, u=np.nan, labels=None, comp=None):
    """Gradient results for one traverse.

    case: one-case fluid dict (arrays of length 1); P, T: node pressures/temperatures of that case from
    vlp.traverse (wellhead first). labels: optional {node index: label} for equipment at nodes.
    """
    case = {k: np.atleast_1d(np.asarray(v, float))[:1] for k, v in case.items()}
    kind = fluid.kind
    corr = CORRELATIONS[correlation]
    labels = labels or {}
    n = len(path.md)
    P, T = np.asarray(P, float), np.asarray(T, float)
    rates = _surface_rates(kind, case)

    seg = []                                      # per segment (top node i -> node i + 1), at the lower node
    for i in range(n - 1):
        node = i + 1
        pn = (_restriction_split(fluid, case, P[node], T[node], path.restrictions[node])[0]
              if node in path.restrictions else P[node])
        pm, tm = 0.5 * (P[i] + pn), 0.5 * (T[i] + T[node])
        L = path.seg_len[i]
        ph = _phases(fluid, np.array([pm]), np.array([tm]), case)
        a = path.area[i]
        vsl, vsg = max(float(ph["ql"][0]) / a, 1e-12), max(float(ph["qg"][0]) / a, 0.0)
        rho_l, rho_g = float(ph["rho_l"][0]), max(float(ph["rho_g"][0]), 1e-4)
        sig = max(float(ph["sig_l"][0]), 0.5)
        args = (np.array([vsl]), np.array([vsg]), np.array([rho_l]), np.array([rho_g]), ph["mu_l"],
                np.maximum(ph["mu_g"], 1e-4), np.array([sig]), path.d_h[i], path.rough[i], path.sin_t[i],
                np.array([pm]))
        lam = vsl / max(vsl + vsg, 1e-30)
        if lam > 0.99999 or lam < 1e-6:
            e, f, hl = no_slip(*args)
        elif corr.__name__ in NEEDS_GLR:
            e, f, hl = corr(*args, glr=V._glr(case, kind))
        else:
            e, f, hl = corr(*args)
        seg.append(dict(ph=ph, vsl=vsl, vsg=vsg, rho_l=rho_l, rho_g=rho_g, sig=sig, lam=lam, hl=float(hl[0]),
                        e=p1 * float(e[0]), f=p2 * float(f[0]), L=L, i=i, pm=pm, tm=tm))
    dpg = np.array([s_["e"] * s_["L"] for s_ in seg])
    dpf = np.array([s_["f"] * s_["L"] for s_ in seg])
    g_below = np.r_[np.cumsum(dpg[::-1])[::-1], 0.0]          # loss from each node down to the bottom node
    f_below = np.r_[np.cumsum(dpf[::-1])[::-1], 0.0]

    rows = []
    blank = {}

    def add(md, tvd, p, t, label, regime, **kw):
        rows.append({"Label": label, "Measured Depth (ft)": md, "True Vertical Depth (ft)": tvd,
                     "Pressure (psia)": p, "Temperature (°F)": t, "Regime": regime, **kw})

    add(path.md[0], path.tvd[0], P[0], T[0], labels.get(0, ""), "Wellhead",
        **{"Gravity Pressure Loss (psi)": g_below[0], "Friction Pressure Loss (psi)": f_below[0]})
    for s_ in seg:
        i, node = s_["i"], s_["i"] + 1
        ph = {k: float(np.atleast_1d(v)[0]) for k, v in s_["ph"].items()}
        hl, lam = s_["hl"], s_["lam"]
        if correlation == "tuffp_unified" and 1e-6 < lam < 0.99999:
            from .unified import tuffp_details
            regime = str(tuffp_details([s_["vsl"]], [s_["vsg"]], [s_["rho_l"]], [s_["rho_g"]], [ph["mu_l"]],
                                       [ph["mu_g"]], [s_["sig"]], path.d_h[i], path.rough[i], path.sin_t[i],
                                       [0.5 * (P[i] + P[node])])["regime"][0])
        else:
            regime = flow_regime(correlation, s_["vsl"], s_["vsg"], s_["rho_l"], s_["rho_g"], s_["sig"],
                                 path.d_h[i], path.sin_t[i])
        rho_mix_ns = lam * s_["rho_l"] + (1 - lam) * s_["rho_g"]
        rho_flow = hl * s_["rho_l"] + (1 - hl) * s_["rho_g"]
        m_l = ph["m_oil"] + ph["m_water"] if np.isfinite(ph["m_oil"]) else s_["vsl"] * path.area[i] * s_["rho_l"] * 86400
        m_g = ph["m_gas"]
        sig_ow = np.nan
        p_in = P[node]
        last = node == n - 1
        rec = {
            "Gradient (psi/ft)": s_["e"] + s_["f"], "Holdup (fraction)": hl,
            "Static Gradient (psi/ft)": s_["e"], "Friction Gradient (psi/ft)": s_["f"],
            "Gravity Pressure Loss (psi)": g_below[node], "Friction Pressure Loss (psi)": f_below[node],
            "Heat Transfer Coefficient (BTU/h/ft2/F)": u, "Surrounding Temperature (°F)": path.t_form[node],
            "Superficial Liquid Velocity (ft/s)": s_["vsl"], "Superficial Gas Velocity (ft/s)": s_["vsg"],
            "Slip Liquid Velocity (ft/s)": s_["vsl"] / max(hl, 1e-9),
            "Slip Gas Velocity (ft/s)": s_["vsg"] / max(1 - hl, 1e-9) if s_["vsg"] > 0 else 0.0,
            "Total Velocity (ft/s)": s_["vsl"] + s_["vsg"],
            "Diameter (in)": path.d_h[i] * 12, "Angle From Vertical (deg)": float(np.degrees(np.arccos(
                np.clip(path.sin_t[i], -1, 1)))), "Node Length (ft)": s_["L"],
            "Total Mass Flow Rate (lbm/day)": m_l + m_g, "Oil Mass Flow Rate (lbm/day)": ph["m_oil"],
            "Water Mass Flow Rate (lbm/day)": ph["m_water"], "Liquid Mass Flow Rate (lbm/day)": m_l,
            "Gas Mass Flow Rate (lbm/day)": m_g,
            **rates, "Free Gas Rate (MMscf/day)": ph["free_gas"],
            "Oil Density (lb/ft3)": ph["rho_o"], "Water Density (lb/ft3)": ph["rho_w"],
            "Liquid Density (lb/ft3)": s_["rho_l"], "Gas Density (lb/ft3)": s_["rho_g"] if s_["vsg"] > 0 else np.nan,
            "Mixture Density (lb/ft3)": rho_mix_ns, "Flowing Density (lb/ft3)": rho_flow,
            "Oil Viscosity (cp)": ph["mu_o"], "Water Viscosity (cp)": ph["mu_w"], "Liquid Viscosity (cp)": ph["mu_l"],
            "Gas Viscosity (cp)": ph["mu_g"] if s_["vsg"] > 0 else np.nan,
            "Gas-Liquid Interfacial Tension (dyne/cm)": s_["sig"] if s_["vsg"] > 0 else np.nan,
            "Gas-Water Interfacial Tension (dyne/cm)": ph["sig_gw"] if s_["vsg"] > 0 else np.nan,
            "Gas-Oil Interfacial Tension (dyne/cm)": ph["sig_go"] if s_["vsg"] > 0 else np.nan,
            "Oil Formation Volume Factor (RB/STB)": ph["bo"], "Water Formation Volume Factor (RB/STB)": ph["bw"],
            "Gas Formation Volume Factor (ft3/scf)": ph["bg"] if s_["vsg"] > 0 else np.nan,
        }
        if comp is not None:
            from .complementary import node_values
            mu_ns = lam * ph["mu_l"] + (1 - lam) * (ph["mu_g"] if np.isfinite(ph["mu_g"]) else 0.0)
            fg = ph["free_gas"] if np.isfinite(ph["free_gas"]) else rates.get("Gas Rate (MMscf/day)", np.nan)
            rec.update(node_values(kind, s_["vsl"] + s_["vsg"], s_["vsg"], rho_mix_ns, mu_ns, s_["rho_l"],
                                   s_["rho_g"], s_["sig"], path.d_h[i] * 12, fg, comp, ph["mu_l"],
                                   ph["mu_g"] if np.isfinite(ph["mu_g"]) else None, float(path.sin_t[i])))
            from .complementary import path_values
            st_ = fluid.state(np.array([s_["pm"]]), np.array([s_["tm"]]), case)
            rec.update(path_values(path, i, fluid, kind, case, st_, s_["pm"], s_["tm"], comp))
        if node in path.restrictions:
            d_in = path.restrictions[node]
            p_out, dp = _restriction_split(fluid, case, p_in, T[node], d_in)
            reg = "Critical" if p_out / max(p_in, 1e-9) < CRITICAL_RATIO else "Subcritical"
            add(path.md[node], path.tvd[node], p_out, T[node], "", regime, **rec)
            add(path.md[node], path.tvd[node], p_out, T[node], labels.get(node, ""),
                f"Restriction Outlet - {reg}", **{"Diameter (in)": d_in})
            add(path.md[node], path.tvd[node], p_in, T[node], "", f"Restriction Inlet - {reg}",
                **{"Diameter (in)": d_in})
        else:
            add(path.md[node], path.tvd[node], P[node], T[node], labels.get(node, ""), regime, **rec)
    add(path.md[-1], path.tvd[-1], P[-1], T[-1], "", "Bottomhole")
    df = pd.DataFrame(rows)
    df.insert(0, "Point", np.arange(1, len(df) + 1))
    order = ["Point", "Label", "Measured Depth (ft)", "True Vertical Depth (ft)", "Pressure (psia)",
             "Temperature (°F)", "Gradient (psi/ft)", "Holdup (fraction)", "Regime"]
    df = df[order + [c for c in df.columns if c not in order]]
    return df.dropna(axis=1, how="all")


def node_labels(path, points):
    """{node: label} for the downhole point components (SSSV, restrictions, mandrels, SSDs)."""
    out = {}
    if points is None or not len(points):
        return out
    for _, r in points.iterrows():
        k = int(np.argmin(np.abs(path.md - r["MD (ft)"])))
        if abs(path.md[k] - r["MD (ft)"]) < 1.0:
            lab = r["Label"] if r["Label"] != r["Type"] else r["Type"]
            out[k] = f"{out[k]} / {lab}" if k in out else str(lab)
    return out
