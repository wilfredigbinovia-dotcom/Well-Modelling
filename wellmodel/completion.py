"""Completion effects on inflow: skin models and sand-control pressure drops.

Skin (dimensionless, added to the reservoir skin of an IPR model)
-----------------------------------------------------------------
Karakas & Tariq (1991, SPE 18247)  perforation skin: horizontal (phasing),
    vertical and wellbore terms, with damaged-zone and crushed-zone effects.
MacLeod (1983)                     damaged zone (Hawkins) + crushed zone around
    perforation tunnels: (1/(Lp n')) (k/kc - k/kd) ln(rc/rp).
Hawkins (1956)                     damaged zone: (k/kd - 1) ln(rd/rw).
Cinco-Ley, Miller & Ramey (1975)   deviation skin correlation
    s = -(t'/41)^2.06 - (t'/56)^1.865 log10(hD/100); "Cinco (2)" (anisotropic, to 75 deg):
    t' = atan(sqrt(kv/kh) tan t), hD = (h/rw) sqrt(kh/kv); "Cinco" (isotropic, to 65 deg): t' = t, hD = h/rw.
Brons & Marting (1961)             partial penetration (PROSPER "Martin-Bronz").
Cinco-Ley, Ramey & Miller (1975)   SPE 5589 full solution: infinite-conductivity slanted, partially
    penetrating line source in a slab (segment fluxes, Eqs. 4-6) - slant + penetration + interval location.
Papatzacos (1987)                  partial penetration skin (closed form).

Sand control (PROSPER 2.7.8.2 / 2.19.2.6): dP between the sand face and the wellbore,
dP = 141.2 Q mu / (k H) (S + D Q), in-situ rate and properties at the sand-face pressure
------------------------------------------------------------------------------------
Cased-hole gravel pack   Jones linear flow through gravel-filled perforation tunnels:
    dP = a Q^2 + b Q, a = 9.08e-13 beta rho L / A^2, b = mu L / (1.127e-3 kg A),
    A = pi (d/24)^2 spf (interval / cos dev) efficiency, beta = 1.47e7 / kg^0.55.
Open-hole gravel pack    radial flow through the pack between screen and sand face.
Pre-packed screen        Darcy (k/k_pp) ln(Rso/Rsi) + (k/k_out) ln(Rw/Rso); non-Darcy Do beta (1/r1 - 1/r2).
Wire-wrapped screen      (k/k_out) ln(Rw/Rso) and its non-Darcy term.
Slotted liner            radial flow centred on the well (Roe, Rse), linear flow through the slots and radial
                         flow converging to them (PROSPER's Rosl / Risl / Roe / Rse construction).
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
        # PROSPER: horizontal (plane-flow) skin on the original geometry; vertical and wellbore skins on the
        # damage-modified length and radius (reproduces PROSPER's perforation skin to 1e-6)
        p = kt_perforation_skin(rw_e, lp_e, rp, spf, phasing, kv_kh)
        p["horizontal"] = kt_perforation_skin(rw, lp, rp, spf, phasing, kv_kh)["horizontal"]
        p["perforation"] = p["horizontal"] + p["vertical"] + p["wellbore"]
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
        beyond = ld <= 0 or lp > ld
        k_around = k if beyond else kd                    # rock surrounding the crushed zone
        lp_c = lp - (1 - kd / k) * ld if (ld > 0 and lp > ld) else lp   # damage-modified length (as PROSPER)
        sc = (1.0 / spf) / lp_c * (k / kc - k / k_around) * np.log((rp + lc) / rp)
    out["crushed zone"] = float(sc)
    out["total"] = float(total + sc)
    return out


def hawkins(k, rw, ld, kd):
    return float((k / kd - 1) * np.log((rw + ld) / rw)) if ld > 0 else 0.0


def macleod(k, rw, lp, rp, spf, ld=0.0, kd=None, lc=0.0, kc=None, efficiency=1.0, h=None, h_perf=None):
    """McLeod (1983): damaged zone (Hawkins) plus the crushed zone around the perforation tunnels,
        s_c = h / (Lp n) (k/kc - k/kd) ln(rc/rp),   n = spf x efficiency x perforated interval
    (h / n = 1 / spf when the whole formation is perforated with every shot open)."""
    kd = k if (kd is None or ld <= 0) else kd
    sd = hawkins(k, rw, ld, kd)
    ratio = 1.0 / (spf * efficiency)
    if h is not None and h_perf:
        ratio *= h / h_perf
    sc = ratio / lp * (k / kc - k / kd) * np.log((rp + lc) / rp) if (lc > 0 and kc) else 0.0
    return {"damage": sd, "crushed zone": float(sc), "total": float(sd + sc)}


DEV_LIMIT_ISO, DEV_LIMIT_ANISO = 65.0, 75.0      # PROSPER: Cinco valid to 65 deg, Cinco (2) to 75 deg


def deviation_skin(theta_deg, h, rw, kv_kh=1.0, anisotropic=True):
    """Cinco-Ley, Miller & Ramey (1975) deviation (slant) pseudo-skin correlation
        s_theta = -(t'/41)^2.06 - (t'/56)^1.865 log10(hD/100).
    anisotropic=True (PROSPER "Cinco (2)"): t' = atan(sqrt(kv/kh) tan t), hD = (h/rw) sqrt(kh/kv), valid to 75 deg.
    anisotropic=False (PROSPER "Cinco"): isotropic, t' = t, hD = h/rw, valid to 65 deg."""
    if theta_deg <= 0:
        return 0.0
    lim = DEV_LIMIT_ANISO if anisotropic else DEV_LIMIT_ISO
    if theta_deg > lim + 1e-9:
        raise ValueError(f"The {'Cinco (2)' if anisotropic else 'Cinco'} deviation skin is valid up to {lim:g} degrees.")
    if anisotropic:
        tp = np.degrees(np.arctan(np.sqrt(kv_kh) * np.tan(np.radians(theta_deg))))
        hd = h / rw * np.sqrt(1 / kv_kh)
    else:
        tp, hd = theta_deg, h / rw
    return float(-(tp / 41) ** 2.06 - (tp / 56) ** 1.865 * np.log10(hd / 100))


def brons_marting(h, hp, rw, kv_kh=1.0):
    """Brons & Marting (1961) partial-penetration pseudo-skin (PROSPER "Martin-Bronz"), interval open at the top
    or bottom of the formation:  s_p = (1/b - 1) [ln hD - G(b)],  b = hp/h,  hD = (h/rw) sqrt(kh/kv),
    G(b) = 2.948 - 7.363 b + 11.45 b^2 - 4.675 b^3."""
    b = min(max(hp / h, 1e-6), 1.0)
    if b >= 1.0 - 1e-9:
        return 0.0
    hd = h / rw * np.sqrt(1.0 / kv_kh)
    g = 2.948 - 7.363 * b + 11.45 * b * b - 4.675 * b ** 3
    return float((1 / b - 1) * (np.log(hd) - g))


def cinco_slant_pp(theta_deg, h, hp, z_top, rw, kv_kh=1.0, n_seg=None, n_images=5):
    """Cinco-Ley, Ramey & Miller (1975, SPE 5589) pseudo-skin of a partially-penetrating slanted well of infinite
    conductivity in an infinite slab: the line source is split into M segments whose fluxes make the pseudo-skin
    (Eq. 4) equal at every segment midpoint (Eqs. 5-6). h formation thickness, hp the vertical (TVD) height of
    the completed interval, z_top the vertical distance from the top of the formation to the top of the interval,
    rw wellbore radius (ft). Anisotropy by the usual scaling (z * sqrt(kh/kv), tan t * sqrt(kv/kh)).
    Returns dict(total, slant, penetration): s_theta+p, s_theta = s_theta+p - s_p, s_p (the same interval vertical)."""
    def solve(theta):
        th = np.radians(theta)
        sc = np.sqrt(1.0 / kv_kh)
        hD = h / rw * sc
        hvD = min(hp, h) / rw * sc                       # vertical extent of the interval
        tan_e = np.tan(th) / sc
        th_e = np.arctan(tan_e)
        c, s = np.cos(th_e), np.sin(th_e)
        hwD = hvD / c                                     # length along the (scaled) axis
        zwD = hD - (z_top / rw * sc + hvD / 2)            # middle of the interval, from the bottom
        M = n_seg or int(np.clip(np.floor(hwD), 4, 60))
        seg = hwD / M
        t = (np.arange(M) + 0.5) * seg - hwD / 2          # along-axis position of segment midpoints
        zwi = zwD + t * c
        xi = t * s
        zj, xj = zwi, xi                                  # observation points: segment midpoints on the axis
        G = np.zeros((M, M))
        V = np.zeros(M)
        for n in range(-n_images, n_images + 1):
            A = (zj[:, None] - zwD) * tan_e - xi[None, :]
            B = zj[:, None] + 2 * n * hD - zwi[None, :]
            C = zj[:, None] + 2 * n * hD + zwi[None, :]
            hh = seg / 2

            def lnr(u_hi, u_lo, w):
                return np.log((u_hi + np.sqrt(u_hi ** 2 + 1 + w ** 2)) / (u_lo + np.sqrt(u_lo ** 2 + 1 + w ** 2)))
            t1 = lnr(hh - A * s - B * c, -hh - A * s - B * c, -B * s + A * c)
            t2 = lnr(hh + A * s - C * c, -hh + A * s - C * c, C * s + A * c)
            G += 0.5 * hD / seg * (t1 + t2)                 # segment i carries the share x_i of the rate
            u = zj + 2 * n * hD
            V += 0.5 * np.log((u + hD + np.sqrt((u + hD) ** 2 + 1)) / (u - hD + np.sqrt((u - hD) ** 2 + 1)))
        # unknowns: x_i = share of the well rate from segment i (sum 1) and s:  sum_i G_ji x_i - V_j = s
        K = np.zeros((M + 1, M + 1))
        K[:M, :M] = G
        K[:M, M] = -1.0
        K[M, :M] = 1.0
        rhs = np.concatenate([V, [1.0]])
        sol = np.linalg.solve(K, rhs)
        return float(sol[M])
    if hp >= h - 1e-9 and theta_deg <= 0:
        return {"total": 0.0, "slant": 0.0, "penetration": 0.0}
    s_tot = solve(theta_deg)
    s_p = solve(0.0) if theta_deg > 0 else s_tot
    return {"total": s_tot, "slant": s_tot - s_p, "penetration": s_p}


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
# After PROSPER (User Guide 2.7.8.2 and Appendix 2.19.2.6). The IPR is calculated to the sand face (r_w); the
# completion adds dP = (141.2 Q mu / k H) (S + D Q) between the sand face and the wellbore, where Q is the in-situ
# rate (RB/d) and the fluid properties (density, viscosity, FVF) are recalculated at the sand-face pressure.
# Non-Darcy terms use Do = 9.08e-13 / (4 pi^2 141.2) rho k H / (mu hp^2) = 1.629e-16 rho k H / (mu hp^2)
# (rho in lb/ft3 - PROSPER prints 1.0205e-14 with rho as specific gravity), beta = 1.47e7 / k^0.55 (1/ft).
SC_KINDS = {"none": "None", "gravel_pack_cased": "Gravel pack (cased hole)",
            "gravel_pack_open": "Gravel pack (open hole)", "pre_packed_screen": "Pre-packed screen",
            "wire_wrapped_screen": "Wire-wrapped screen", "slotted_liner": "Slotted liner"}
GRAVEL_PERMEABILITY = [("Ottawa sand", "12/20", 500000), ("Ottawa sand", "20/40", 150000),
                       ("Ottawa sand", "30/50", 90000), ("Ottawa sand", "40/60", 60000),
                       ("Ottawa sand", "50/70", 30000), ("Carbolite", "20/40", 350000),
                       ("Carbolite", "16/20", 500000), ("Isopac", "20/40", 110000)]   # lab values, md
DO_CONST = 9.08e-13 / (4 * np.pi ** 2 * 141.2)        # 1.629e-16
RB_FT3 = 5.6146


@dataclass
class SandControl:
    kind: str                 # none | gravel_pack_cased | gravel_pack_open | pre_packed_screen | wire_wrapped_screen | slotted_liner
    params: dict = field(default_factory=dict)

    def label(self):
        return SC_KINDS[self.kind]


def _beta_gravel(kg):
    return 1.47e7 / kg ** 0.55


def _beta(value, k):
    """PROSPER convention: blank (None) = calculated from the permeability, 0 = ignored, else as entered."""
    return _beta_gravel(k) if value is None else float(value)


def props_from_fluid(fluid, kind, t_f, ratios, multiphase=True):
    """In-situ properties at the sand face from a VLP fluid (vlp.BlackOilFluid / WetGasFluid / EOSTableFluid).
    Returns props(q, p) -> (Q RB/d, rho lb/ft3, mu cp); q is STB/d of liquid (oil) or Mscf/d (gas).
    Oil, single phase (PROSPER 'Single Phase'): the liquid only - the free gas released at the sand face is
    ignored; 'Multi Phase': oil, water and free gas, volume-weighted viscosity and mass-weighted density."""
    def props(q, p):
        q0 = np.atleast_1d(np.asarray(q, float))
        q = np.maximum(q0, 1e-6)                       # properties at a vanishing rate stay defined
        p = np.broadcast_to(np.maximum(np.asarray(p, float), 14.7), q.shape)
        case = {"q": q / 1000.0 if kind == "gas" else q, **{k: np.full(q.shape, v) for k, v in ratios.items()}}
        st = fluid.state(p, np.full(q.shape, t_f), case)
        ql, qg = np.maximum(st["ql"], 0.0), np.maximum(st["qg"], 0.0)
        if kind == "oil" and not multiphase:
            qg = 0.0 * qg
        qt = np.maximum(ql + qg, 1e-30)
        rho = (ql * st["rho_l"] + qg * st["rho_g"]) / qt
        mu = (ql * st["mu_l"] + qg * st["mu_g"]) / qt
        return qt * 86400.0 / RB_FT3 * (q0 / q), rho, mu
    return props


def props_at_pressure(props, p_fixed):
    """The property function evaluated at one pressure (PROSPER evaluates an oil well's sand-control dP with the
    fluid at the reservoir pressure: its Tutorial 15-18 skins and dPs within 0.3 % that way, against up to 2.3 times
    the dP near the AOF with the sand-face pressure)."""
    def f(q, p=None):
        q = np.atleast_1d(np.asarray(q, float))
        return props(q, np.full(q.shape, float(p_fixed)))
    return f


def props_constant(fvf, rho, mu, kind="oil"):
    """Constant properties (fvf RB/STB for oil, ft3/scf for gas)."""
    def props(q, p):
        q = np.atleast_1d(np.asarray(q, float))
        qr = q * fvf if kind == "oil" else q * 1000.0 * fvf / RB_FT3
        return qr, np.full(q.shape, float(rho)), np.full(q.shape, float(mu))
    return props


class SandControlModel:
    """dP across a sand-control completion as a function of rate and sand-face pressure."""

    def __init__(self, sc: SandControl, props, k, h, rw):
        self.sc, self.props, self.k, self.h, self.rw = sc, props, float(k), float(h), float(rw)
        p, kind = sc.params, sc.kind
        self.notes = []
        self.area_open = None
        if kind == "gravel_pack_cased":
            hp_md = p["interval"] / max(np.cos(np.radians(p.get("deviation", 0.0))), 1e-6)
            self.area_open = np.pi * (p["perf_diameter_in"] / 24) ** 2 * p["spf"] * hp_md * p["perf_efficiency"]
            self.L = p["pack_length_in"] / 12
            self.kg = p["gravel_perm"]
            self.beta = _beta(p.get("beta"), self.kg)
            self.hp = p["interval"]
            return
        self.hp = p["interval"]
        k_out = p.get("outside_perm") or self.k                # blank: formation sand outside the screen
        b_out = _beta(p.get("outside_beta"), k_out)
        f = self.k / self.hp * self.h / 1.0                   # k H / hp (Darcy skin of a radial shell over hp)
        rw = self.rw
        if kind == "gravel_pack_open":
            rs = rw - p["pack_length_in"] / 12
            if rs <= 0:
                raise ValueError("The gravel pack length must be smaller than the wellbore radius.")
            kg = p["gravel_perm"]
            bg_ = _beta(p.get("beta"), kg)
            self.parts = {"gravel pack": (f / kg * np.log(rw / rs), bg_ * (1 / rs - 1 / rw))}
        elif kind == "pre_packed_screen":
            rsi, rso = p["screen_inner_radius"], p["screen_outer_radius"]
            if not 0 < rsi < rso <= rw:
                raise ValueError("Need 0 < screen inner radius < screen outer radius <= wellbore radius.")
            kpp = p["screen_perm"]
            bpp = _beta(p.get("screen_beta"), kpp)
            s1, g1 = self.k * self.h / self.hp / kpp * np.log(rso / rsi), bpp * (1 / rsi - 1 / rso)
            s2, g2 = self.k * self.h / self.hp / k_out * np.log(rw / rso), b_out * (1 / rso - 1 / rw)
            self.parts = {"screen (pre-pack)": (s1, g1), "screen to sand face": (s2, g2)}
        elif kind == "wire_wrapped_screen":
            rso = p["screen_outer_radius"]
            if not 0 < rso <= rw:
                raise ValueError("Need 0 < screen outer radius <= wellbore radius.")
            self.parts = {"screen to sand face": (self.k * self.h / self.hp / k_out * np.log(rw / rso),
                                                  b_out * (1 / rso - 1 / rw))}
        elif kind == "slotted_liner":
            ri, ro, rs = p["liner_inner_radius"], p["liner_outer_radius"], p.get("screen_outer_radius") or 0.0
            hs, ws = p["slot_height_in"] / 12, p["slot_width_in"] / 12
            if not 0 < ri < ro <= rw:
                raise ValueError("Need 0 < liner inner radius < liner outer radius <= wellbore radius.")
            if rs and rs >= ri:
                raise ValueError("The screen outer radius must be smaller than the liner inner radius (0 = no screen).")
            spc = p["slot_density"] * hs                       # slots around the circumference
            if spc <= 0:
                raise ValueError("Slot density and slot height must be positive.")
            slo, sli = 2 * np.pi * ro / spc, 2 * np.pi * ri / spc
            rosl = rw - ro if slo / 2 > rw - ro else slo / 2
            risl = (ri - rs if sli / 2 > ri - rs else sli / 2) if rs else 0.0
            roe = np.sqrt(ro ** 2 + 0.5 * spc * rosl ** 2)
            rse = np.sqrt(ri ** 2 + 0.5 * spc * risl ** 2) if rs else 0.0
            kH = self.k * self.h / self.hp / k_out
            s_rad = kH * (np.log(rw / roe) if rw > roe else 0.0) + (kH * np.log(rse / rs) if rs else 0.0)
            s_l1 = np.pi * (ro - ri) / ws
            s_l2 = np.log(2 * risl / ws) if rs and risl > ws / 2 else 0.0
            s_l3 = np.log(2 * rosl / ws) if rosl > ws / 2 else 0.0
            s_slot = 2 * kH / spc * (s_l1 + s_l2 + s_l3)       # flow shared among spc slots, through k_out
            g_sl = 39.5 * b_out * (ro - ri) / (spc * ws) ** 2
            g_osl = 4 * b_out / spc ** 2 * (2 / ws - 1 / rosl) if rosl > 0 else 0.0
            g_isl = 4 * b_out / spc ** 2 * (2 / ws - 1 / risl) if rs and risl > 0 else 0.0
            g_oe = b_out * (1 / roe - 1 / rw) if rw > roe else 0.0
            g_se = b_out * (1 / rs - 1 / rse) if rs else 0.0
            self.parts = {"radial flow (well)": (s_rad, g_oe + g_se), "slots": (s_slot, g_sl + g_osl + g_isl)}
            self.geom = {"slots around circumference": spc, "Rosl (ft)": rosl, "Risl (ft)": risl, "Roe (ft)": roe,
                         "Rse (ft)": rse}
        else:
            raise KeyError(kind)
        self.s_d = sum(v[0] for v in self.parts.values())
        self.g_nd = sum(v[1] for v in self.parts.values())

    # ------------------------------------------------------------------
    def details(self, q, p_sf):
        """dict of arrays: dp (psi), skin (rate-dependent), darcy skin, D (1/(RB/d)), in-situ rate, velocity.
        Fluid properties at the sand-face pressure (checked against PROSPER's Step 5 manual example: dP within
        about 3.5 % at the solution point and 1 % at the AOF)."""
        Q, rho, mu = self.props(q, p_sf)
        if self.sc.kind == "gravel_pack_cased":
            A = self.area_open
            a = 9.08e-13 * self.beta * rho * self.L / A ** 2
            b = mu * self.L / (1.127e-3 * self.kg * A)
            dp = a * Q ** 2 + b * Q
            c = 141.2 * mu / (self.k * self.h)
            s_d = b / c
            d = a / c
            v = Q * RB_FT3 / 86400 / A
            return {"dp": dp, "skin": s_d + d * Q, "darcy skin": s_d, "D": d, "Q": Q, "velocity": v, "a": a, "b": b}
        c = 141.2 * mu / (self.k * self.h)
        do = DO_CONST * rho * self.k * self.h / (mu * self.hp ** 2)
        d = do * self.g_nd
        dp = c * Q * (self.s_d + d * Q)
        return {"dp": dp, "skin": self.s_d + d * Q, "darcy skin": self.s_d + 0 * Q, "D": d, "Q": Q,
                "velocity": None}

    def __call__(self, q, p_sf):
        return self.details(q, p_sf)["dp"]


def sand_control_dp(sc: SandControl, props, k=None, h=None, rw=None):
    """Sand-control dP model: callable dp(q, p_sf) (psi). props(q, p) -> (Q RB/d, rho, mu); see props_from_fluid.
    k (md), h (ft) and rw (ft) of the reservoir (the dP is reported as a skin on k h)."""
    if sc.kind == "none":
        return lambda q, p=None: 0.0 * np.asarray(q, float)
    return SandControlModel(sc, props, k if k else 100.0, h if h else 50.0, rw if rw else 0.354)


def apply_sand_control(curve, dp_fn):
    """IPR at the wellbore with the sand-control dP in series. Each point of the sand-face IPR (p_sf, q) moves to
    pwf = p_sf - dP(q, p_sf), with the fluid properties at the sand-face pressure (PROSPER); the curve ends where
    pwf reaches zero."""
    from .inflow import IPRCurve
    order = np.argsort(-np.asarray(curve.pwf, float))
    psf = np.asarray(curve.pwf, float)[order]
    q = np.asarray(curve.q, float)[order]
    if len(psf) < 80:                                  # refine so the end of the curve is resolved
        pf = np.linspace(psf[0], psf[-1], 160)
        q = np.interp(pf, psf[::-1], q[::-1]) if curve.rate_fn is None else np.asarray(curve.rate(pf), float)
        psf = pf
    dp = np.nan_to_num(np.asarray(dp_fn(q, psf), float), nan=0.0)
    pw = psf - dp
    keep = pw >= 0.0
    k = int(np.argmax(~keep)) if not keep.all() else len(pw)
    pw_o, q_o, dp_o, psf_o = list(pw[:k]), list(q[:k]), list(dp[:k]), list(psf[:k])
    if 0 < k < len(pw):                                # interpolate the zero-pwf end (AOF)
        f = pw[k - 1] / (pw[k - 1] - pw[k])
        q_o.append(q[k - 1] + f * (q[k] - q[k - 1])); pw_o.append(0.0)
        dp_o.append(dp[k - 1] + f * (dp[k] - dp[k - 1])); psf_o.append(psf[k - 1] + f * (psf[k] - psf[k - 1]))
    out = IPRCurve(curve.model, curve.fluid, curve.pr, np.array(pw_o), np.array(q_o), dict(curve.params),
                   list(curve.notes), curve.tests, None)
    out.dp_sand_control = np.array(dp_o)
    out.p_sandface = np.array(psf_o)
    if hasattr(dp_fn, "details"):
        det = dp_fn.details(out.q, out.p_sandface)
        out.sc_skin = np.asarray(det["skin"], float)
        out.sc_velocity = None if det.get("velocity") is None else np.asarray(det["velocity"], float)
        out.sc_d = np.asarray(det["D"], float)
    out.params["AOF without sand control"] = curve.aof
    return out
