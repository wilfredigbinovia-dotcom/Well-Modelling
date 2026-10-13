"""Completion effects on inflow: skin models and sand-control pressure drops.

Skin (dimensionless, added to the reservoir skin of an IPR model)
-----------------------------------------------------------------
Karakas & Tariq (1991, SPE 18247)  perforation skin: horizontal (phasing),
    vertical and wellbore terms, with damaged-zone and crushed-zone effects.
MacLeod (1983)                     damaged zone (Hawkins) + crushed zone around
    perforation tunnels: (1/(Lp n')) (k/kc - k/kd) ln(rc/rp).
Hawkins (1956)                     damaged zone: (k/kd - 1) ln(rd/rw).
Cinco-Ley, Miller & Ramey (1975)   deviation skin
    s = -(t'/41)^2.06 - (t'/56)^1.865 log10(hD/100),
    t' = atan(sqrt(kv/kh) tan t) in degrees, hD = (h/rw) sqrt(kh/kv); valid to 75 deg.
Papatzacos (1987)                  partial penetration skin (closed form).

Sand control (extra drawdown between sand face and wellbore)
------------------------------------------------------------
Cased-hole gravel pack   linear flow through gravel-filled perforation tunnels
    oil  dp  = 9.08e-13 beta Bo^2 rho L q^2 / A^2 + mu Bo L q / (1.127e-3 kg A)
    gas  dp2 = 1.247e-10 beta SG T Z L q^2 / A^2 + 8.93e3 mu T Z L q / (kg A)
    beta = 1.47e7 / kg^0.55 (1/ft), A = open perforation area (ft2), L (ft),
    q in STB/d (oil) or Mscf/d (gas)  (Jones, Blount & Glaze; Golan & Whitson).
Open-hole gravel pack and pre-packed screen: radial Darcy + Forchheimer flow
    across the pack annulus (same coefficients as the Jones radial equations).
Wire-wrapped screen and slotted liner: orifice flow through the open slot area,
    dp = rho v^2 / (2 g Cd^2). Plugging is not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

RANKINE = 459.67

# Karakas & Tariq constants by phasing (deg): a_theta, a1, a2, b1, b2, c1, c2
KT_TABLE = {
    0:   (0.250, -2.091, 0.0453, 5.1313, 1.8672, 1.6e-1, 2.675),
    180: (0.500, -2.025, 0.0943, 3.0373, 1.8115, 2.6e-2, 4.532),
    120: (0.648, -2.018, 0.0634, 1.6136, 1.7770, 6.6e-3, 5.320),
    90:  (0.726, -1.905, 0.1038, 1.5674, 1.6935, 1.9e-3, 6.155),
    60:  (0.813, -1.898, 0.1023, 1.3654, 1.6490, 3.0e-4, 7.509),
    45:  (0.860, -1.788, 0.2398, 1.1915, 1.6392, 4.6e-5, 8.791),
}


# ================================================================== skins
def kt_perforation_skin(rw, lp, rp, spf, phasing, kv_kh=1.0):
    """Karakas & Tariq perforation skin for an undamaged formation.
    rw, lp, rp in ft; spf shots/ft; phasing in degrees (0 = 360)."""
    if phasing not in KT_TABLE:
        raise ValueError(f"Phasing must be one of {sorted(KT_TABLE)} degrees.")
    at, a1, a2, b1, b2, c1, c2 = KT_TABLE[phasing]
    rw_eff = lp / 4 if phasing == 0 else at * (rw + lp)
    s_h = np.log(rw / rw_eff)
    hp = 1.0 / spf                                   # perforation spacing (ft)
    h_d = hp / lp * np.sqrt(1 / kv_kh)
    r_d = rp / (2 * hp) * (1 + np.sqrt(kv_kh))
    a = a1 * np.log10(r_d) + a2
    b = b1 * r_d + b2
    s_v = 10 ** a * h_d ** (b - 1) * r_d ** b
    r_wd = rw / (lp + rw)
    s_wb = c1 * np.exp(c2 * r_wd)
    return {"horizontal": float(s_h), "vertical": float(s_v), "wellbore": float(s_wb),
            "perforation": float(s_h + s_v + s_wb)}


def karakas_tariq(k, rw, lp, rp, spf, phasing, kv_kh=1.0, ld=0.0, kd=None, lc=0.0, kc=None):
    """Total completion skin: perforations, damaged zone (ld, kd) and crushed zone (lc, kc).
    Lengths in ft, permeabilities in md."""
    kd = k if (kd is None or ld <= 0) else kd
    out = {}
    if ld > 0 and lp > ld:                          # perforations reach past the damage
        lp_e = lp - (1 - kd / k) * ld
        rw_e = rw + (1 - kd / k) * ld
        p = kt_perforation_skin(rw_e, lp_e, rp, spf, phasing, kv_kh)
        out.update(p)
        out["damage"] = 0.0
        total = p["perforation"]
    else:                                            # perforations end inside the damage
        p = kt_perforation_skin(rw, lp, rp, spf, phasing, kv_kh)
        out.update(p)
        sd = (k / kd - 1) * np.log((rw + ld) / rw) if ld > 0 else 0.0
        out["damage"] = float(sd)
        total = sd + k / kd * p["perforation"]
    sc = 0.0
    if lc > 0 and kc:
        k_around = k if (ld <= 0 or lp > ld) else kd      # rock surrounding the crushed zone
        sc = (1.0 / spf) / lp * (k / kc - k / k_around) * np.log((rp + lc) / rp)
    out["crushed zone"] = float(sc)
    out["total"] = float(total + sc)
    return out


def hawkins(k, rw, ld, kd):
    return float((k / kd - 1) * np.log((rw + ld) / rw)) if ld > 0 else 0.0


def macleod(k, rw, lp, rp, spf, ld=0.0, kd=None, lc=0.0, kc=None):
    kd = k if (kd is None or ld <= 0) else kd
    sd = hawkins(k, rw, ld, kd)
    sc = (1.0 / spf) / lp * (k / kc - k / kd) * np.log((rp + lc) / rp) if (lc > 0 and kc) else 0.0
    return {"damage": sd, "crushed zone": float(sc), "total": float(sd + sc)}


def deviation_skin(theta_deg, h, rw, kv_kh=1.0):
    """Cinco-Ley, Miller & Ramey (1975)."""
    if theta_deg <= 0:
        return 0.0
    if theta_deg > 75:
        raise ValueError("Deviation skin correlation is valid up to 75 degrees.")
    tp = np.degrees(np.arctan(np.sqrt(kv_kh) * np.tan(np.radians(theta_deg))))
    hd = h / rw * np.sqrt(1 / kv_kh)
    return float(-(tp / 41) ** 2.06 - (tp / 56) ** 1.865 * np.log10(hd / 100))


def partial_penetration_skin(h, hp, z_top, rw, kv_kh=1.0):
    """Papatzacos (1987). h reservoir thickness, hp perforated length, z_top distance
    from top of reservoir to top of perforations (ft)."""
    if hp >= h - 1e-9:
        return 0.0
    hpd = hp / h
    rd = rw / h * np.sqrt(kv_kh)
    h1d = z_top / h
    A = 1 / (h1d + hpd / 4)
    B = 1 / (h1d + 3 * hpd / 4)
    return float((1 / hpd - 1) * np.log(np.pi / (2 * rd))
                 + 1 / hpd * np.log(hpd / (2 + hpd) * np.sqrt((A - 1) / (B - 1))))


# ============================================================ sand control
@dataclass
class SandControl:
    kind: str                 # none | gravel_pack_cased | gravel_pack_open | pre_packed_screen | wire_wrapped_screen | slotted_liner
    params: dict = field(default_factory=dict)

    def label(self):
        return {"none": "None", "gravel_pack_cased": "Gravel pack (cased hole)",
                "gravel_pack_open": "Gravel pack (open hole)", "pre_packed_screen": "Pre-packed screen",
                "wire_wrapped_screen": "Wire-wrapped screen", "slotted_liner": "Slotted liner"}[self.kind]


def _beta_gravel(kg):
    return 1.47e7 / kg ** 0.55


def sand_control_dp(sc: SandControl, fluid: str, props: dict):
    """Return a function giving the sand-control 'drawdown' at rate q:
    oil -> dp (psi) at q STB/d (liquid); gas -> d(p^2) (psi^2) at q Mscf/d.
    props: oil {mu, bo, rho}; gas {mu, z, sg, t_f, rho_g (lb/ft3 at Pr), bg (ft3/scf)}."""
    p = sc.params
    if sc.kind == "none":
        return lambda q: 0.0 * np.asarray(q, float)
    if sc.kind == "gravel_pack_cased":
        n_open = p["spf"] * p["interval"] * p["perf_efficiency"]
        A = n_open * np.pi * (p["perf_diameter_in"] / 24) ** 2       # ft2
        L = p["pack_length_in"] / 12
        kg, beta = p["gravel_perm"], _beta_gravel(p["gravel_perm"])
        if fluid == "oil":
            a = 9.08e-13 * beta * props["bo"] ** 2 * props["rho"] * L / A ** 2
            b = props["mu"] * props["bo"] * L / (1.127e-3 * kg * A)
        else:
            tr = props["t_f"] + RANKINE
            a = 1.247e-10 * beta * props["sg"] * tr * props["z"] * L / A ** 2
            b = 8.93e3 * props["mu"] * tr * props["z"] * L / (kg * A)
        return lambda q: a * np.asarray(q, float) ** 2 + b * np.asarray(q, float)
    if sc.kind in ("gravel_pack_open", "pre_packed_screen"):
        ro = p["outer_radius_in"] / 12
        ri = ro - p["thickness_in"] / 12
        if ri <= 0:
            raise ValueError("Pack thickness must be smaller than the outer radius.")
        h, kg = p["interval"], p["gravel_perm"]
        beta = _beta_gravel(kg)
        if fluid == "oil":
            b = 141.2 * props["mu"] * props["bo"] * np.log(ro / ri) / (kg * h)
            a = 2.30e-14 * beta * props["bo"] ** 2 * props["rho"] * (1 / ri - 1 / ro) / h ** 2
        else:
            tr = props["t_f"] + RANKINE
            b = 1422 * tr * props["mu"] * props["z"] * np.log(ro / ri) / (kg * h)
            a = 3.161e-12 * beta * props["sg"] * tr * props["z"] * (1 / ri - 1 / ro) / h ** 2
        return lambda q: a * np.asarray(q, float) ** 2 + b * np.asarray(q, float)
    if sc.kind in ("wire_wrapped_screen", "slotted_liner"):
        if sc.kind == "wire_wrapped_screen":
            open_frac = p["slot_width_in"] / (p["slot_width_in"] + p["wire_width_in"])
            A = open_frac * np.pi * p["screen_od_in"] / 12 * p["interval"]
        else:
            A = p["slots_per_ft"] * p["interval"] * (p["slot_width_in"] / 12) * (p["slot_length_in"] / 12)
        cd = p.get("cd", 0.6)
        if fluid == "oil":
            vol = lambda q: np.asarray(q, float) * props["bo"] * 5.6146 / 86400
            rho = props["rho"]
            return lambda q: rho * (vol(q) / A) ** 2 / (2 * 32.174 * cd ** 2) / 144
        vol = lambda q: np.asarray(q, float) * 1000 * props["bg"] / 86400
        rho, pr = props["rho_g"], props["pr"]
        # convert dp to d(p^2) about the sand-face pressure ~ Pr: d(p^2) ~ 2 p dp
        return lambda q: 2 * pr * rho * (vol(q) / A) ** 2 / (2 * 32.174 * cd ** 2) / 144
    raise KeyError(sc.kind)


def apply_sand_control(curve, dp_fn):
    """Re-solve an IPRCurve with an extra sand-control drawdown in series.
    For each wellbore pressure pwf find q with q = IPR(p_sf), p_sf = pwf + dp(q)
    (oil) or p_sf^2 = pwf^2 + dp2(q) (gas)."""
    from .inflow import IPRCurve
    gas = curve.fluid == "gas"

    def psf(pwf, q):
        d = float(dp_fn(q))
        return np.sqrt(pwf ** 2 + d) if gas else pwf + d

    qs, dps = [], []
    for pw in curve.pwf:
        qmax = float(curve.rate(pw))
        if qmax <= 0:
            qs.append(0.0); dps.append(0.0)
            continue
        f = lambda q: float(curve.rate(min(psf(pw, q), curve.pr))) - q
        q = brentq(f, 0.0, qmax, xtol=1e-6) if f(qmax) < 0 else qmax
        qs.append(q)
        dps.append(psf(pw, q) - pw)
    qs = np.array(qs)
    out = IPRCurve(curve.model, curve.fluid, curve.pr, curve.pwf.copy(), qs, dict(curve.params),
                   list(curve.notes), curve.tests, None)
    out.dp_sand_control = np.array(dps)
    out.params["AOF without sand control"] = curve.aof
    return out
