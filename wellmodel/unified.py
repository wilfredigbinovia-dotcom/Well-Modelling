"""TUFFP unified model for gas-liquid pipe flow via slug dynamics.

Zhang, H.-Q., Wang, Q., Sarica, C. and Brill, J. P. (2003). Unified Model for Gas-Liquid Pipe Flow via Slug
Dynamics - Part 1: Model Development; Part 2: Model Validation. ASME J. Energy Resour. Technol. 125, 266-283.

All inclinations from -90 to +90 deg. Flow patterns: dispersed bubble, bubbly, slug (intermittent),
stratified / annular. Everything is vectorised over arrays of flow conditions; internally SI units.

Implementation notes (where the paper leaves a choice open):
- Slug flow is solved as in the paper: continuity (Eqs. 1, 3, 4, 7) with the combined momentum equation (11)
  for the film holdup, iterated with the Zhang et al. slug-holdup model (55-57); Gregory et al. (60) starts it.
- Slug -> stratified/annular: the paper's boundary is where the film length becomes infinite. Here slug flow is
  taken to exist where the slug equations have a solution with a finite positive film length (at most
  LF_MAX slug lengths); otherwise the stratified/annular equations (12-15) are solved. This is the same
  boundary reached directly instead of through the vSL iteration of Eqs. 27-31.
- Slug -> dispersed bubble: dispersed bubble when the no-slip liquid fraction is at least the slug-body holdup
  of Eq. 55 (Zhang, Wang & Brill 2003 hypothesis) evaluated with the wall-shear term only.
- Slug -> bubbly (near-vertical upward): Taitel et al. (1980): bubbly flow when the Taylor-bubble drift
  velocity exceeds the Harmathy rise velocity (large pipes) and the void fraction is below 0.25.
- Friction factors: Fanning; laminar 16/Re below Re 2000, turbulent from the Chen (1979) rough-pipe equation
  above 3000, interpolated between (paper: "an alternative correlation for rough pipe").
- Entrainment (Oliemans et al., Eq. 50) only above the Turner/Wallis drop-carrying gas velocity (Kutateladze
  number 3.2, blended to 4.0): the correlation is an annular-flow fit and otherwise predicts large entrainment
  for viscous oils in slug flow.
- Interfacial friction: stratified (Andritsos & Hanratty as Eq. 37) for a turbulent film, annular
  (Ambrosini et al., Eq. 38) for a laminar film (blended between film Reynolds numbers 1500 and 2000) or a film
  wetting the whole wall (wetted fraction above 0.9, as in vertical annular flow) - and only
  for a thin film (film holdup below 0.05, blended to 0.15): a thick laminar film of viscous oil is outside
  Eq. 38's range; and Eq. 38 (low-pressure data) is blended into Whalley & Hewitt (1978) at gas/liquid density
  ratios from 0.005 to 0.025, where its h_F+ term would otherwise grow without bound. Film wall shear: laminar Eq. 20 blended into the turbulent form between Re 1500 and 2500.
"""
from __future__ import annotations

import numpy as np

from .correlations import friction_factor

G = 9.80665
FT, LBFT3, CP, DYNECM = 0.3048, 16.018463, 1e-3, 1e-3
PA_PER_M_TO_PSI_PER_FT = FT / 6894.757
SIGMA_WATER = 0.0731        # N/m, reference in the Grolman correlation
HLS_MIN = 0.24
LF_MAX = 2000.0             # film lengths beyond this many slug lengths count as stratified/annular
N_GRID = 16


# ------------------------------------------------------------------ closures
def _fanning(re, rel_rough):
    re = np.maximum(np.asarray(re, float), 1e-9)
    lam = 16.0 / re
    turb = friction_factor(np.maximum(re, 3000.0), rel_rough) / 4.0
    w = np.clip((re - 2000.0) / 1000.0, 0.0, 1.0)
    return np.where(re < 2000.0, lam, np.where(re > 3000.0, turb, (1 - w) * (16.0 / 2000.0) + w * turb))


def _entrainment(vsl, vsg, rl, rg, ml, mg, sig, d):
    """Oliemans et al. (1986), Eq. 50 (beta4 = 0.27)."""
    vsl, vsg = np.maximum(vsl, 1e-9), np.maximum(vsg, 1e-9)
    we = rg * vsg ** 2 * d / sig
    fr = vsg / np.sqrt(G * d)
    re_l, re_g = rl * vsl * d / ml, rg * vsg * d / mg
    x = 0.003 * we ** 1.8 * fr ** -0.92 * re_l ** 0.7 * (rl / rg) ** 0.38 * (ml / mg) ** 0.97 / re_g ** 1.24
    x = np.clip(x, 0.0, 1e6)
    fe = np.clip(x / (1.0 + x), 0.0, 0.99)
    # The correlation was fitted on annular-flow data. Below the Turner / Wallis gas velocity needed to carry
    # drops (Kutateladze number 3.2) there is no droplet-laden core: no entrainment, blended in up to 4.0.
    ku = vsg * np.sqrt(rg) / np.maximum(sig * G * np.maximum(rl - rg, 1e-6), 1e-12) ** 0.25
    return fe * np.clip((ku - 3.2) / 0.8, 0.0, 1.0)


def _wetted_fraction(a, vsl, vsg, rl, rg, sig, d, cos_t):
    """Grolman (1994), Eq. 44, with the flat-interface fraction from Biberg's (1999) explicit geometry."""
    a = np.clip(a, 1e-9, 1 - 1e-9)
    gamma = np.pi * a + (1.5 * np.pi) ** (1 / 3) * (1 - 2 * a + a ** (1 / 3) - (1 - a) ** (1 / 3))
    th0 = np.clip(gamma / np.pi, 0.0, 1.0)
    c = np.abs(cos_t)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        extra = (rg / ((rl - rg) * np.maximum(c, 1e-12)) * (rl * vsl ** 2 * d / sig) ** 0.25
                 * (vsg ** 2 / ((1 - a) ** 2 * G * d)) ** 0.8)
    th = th0 * (SIGMA_WATER / sig) ** 0.15 + np.nan_to_num(extra, nan=1.0, posinf=1.0)
    return np.clip(th, th0, 1.0)


def _perimeters(a, th, d):
    area = np.pi * d * d / 4
    sf = np.pi * d * th
    sc = np.pi * d * (1 - th)
    scd = d * np.sin(np.pi * th)
    acd = d * d / 4 * (np.pi * th - np.sin(2 * np.pi * th) / 2)
    af = a * area
    si = np.where(acd > 1e-12, (sf * np.maximum(acd - af, 0) + scd * af) / np.maximum(acd, 1e-12), np.pi * d)
    return sf, sc, np.maximum(si, 1e-9)


def _interfacial(a, vf, vc, rho_c, rl, rg, mg, sig, d, sf, si, re_f, rho_g0, vsg, f_c, area, th):
    """Interfacial friction factor: stratified (Eq. 37) for a turbulent film, annular (Eq. 38) for a laminar
    film, blended between film Reynolds numbers 1500 and 2000."""
    vsgt = 5.0 * np.sqrt(np.maximum(rho_g0, 1e-6) / rg)
    f_strat = f_c * (1 + 14.3 * np.sqrt(np.clip(a, 0, 1)) * np.maximum(vsg / vsgt - 1, 0.0))
    vcm = np.maximum(np.abs(vc), 1e-6)
    we = rg * vcm ** 2 * d / sig
    re_g = rg * vcm * d / mg
    f_g = 0.046 * np.maximum(re_g, 1.0) ** -0.2
    h_f = 2 * area * a / (sf + si)
    f_ann = f_g
    need = a < 0.15
    for _ in range(4 if np.any(need) else 0):                     # h_F+ depends on tau_I (Eqs. 38, 41, 42)
        tau_i = f_ann * rho_c * (vc - vf) ** 2 / 2
        h_plus = rg * h_f * np.sqrt(np.maximum(tau_i, 0) / rg) / mg
        f_ann = f_g * (1 + 13.8 * we ** 0.2 * re_g ** -0.6 * np.maximum(h_plus - 200 * np.sqrt(rg / rl), 0))
    # annular form for a laminar AND thin film; a thick film (holdup > 0.15) takes the stratified form even
    # when laminar (viscous oil), where Eq. 38's h_F+ term is far outside its range.
    # Eq. 38 rests on low-pressure (air-water) data; at gas/liquid density ratios above ~0.005 its h_F+ term
    # grows without bound (gas kinematic viscosity falls with pressure). Blend to Whalley & Hewitt (1978),
    # fI = fC (1 + 24 (rhoL/rhoG)^(1/3) delta/d), between density ratios 0.005 and 0.025.
    delta = d / 2 * (1 - np.sqrt(np.clip(1 - a, 0, 1)))
    f_wh = f_c * (1 + 24 * (rl / rg) ** (1 / 3) * delta / d)
    w_p = np.clip((rg / rl - 0.005) / 0.02, 0.0, 1.0)
    f_ann = (1 - w_p) * f_ann + w_p * f_wh
    w_lam = 1 - np.clip((re_f - 1500.0) / 500.0, 0.0, 1.0)
    w_thin = 1 - np.clip((a - 0.05) / 0.10, 0.0, 1.0)
    w_wall = np.clip((th - 0.8) / 0.1, 0.0, 1.0)        # thin film around the whole wall: annular
    w = np.clip(w_lam + w_wall, 0.0, 1.0) * w_thin
    return w * f_ann + (1 - w) * np.maximum(f_strat, f_c)


def _film_stresses(a, vf, vc, vsl, vsg, rl, rg, rho_c, ml, mg, sig, d, eps, cos_t, rho_g0):
    area = np.pi * d * d / 4
    th = _wetted_fraction(a, vsl, vsg, rl, rg, sig, d, cos_t)
    sf, sc, si = _perimeters(a, th, d)
    af, ac = a * area, (1 - a) * area
    re_f = 4 * af * np.abs(vf) * rl / (np.maximum(sf, 1e-12) * ml)
    re_c = 4 * ac * np.abs(vc) * rg / ((sc + si) * mg)
    f_f = _fanning(re_f, eps / np.maximum(4 * af / np.maximum(sf, 1e-12), 1e-9))
    f_c = _fanning(re_c, eps / d)
    f_i = _interfacial(a, vf, vc, rho_c, rl, rg, mg, sig, d, sf, si, re_f, rho_g0, vsg, f_c, area, th)
    tau_i = f_i * rho_c * (vc - vf) * np.abs(vc - vf) / 2
    tau_f_turb = f_f * rl * vf * np.abs(vf) / 2
    h_f = 2 * area * a / (sf + si)
    tau_f_lam = 3 * ml * vf / np.maximum(h_f, 1e-9) - tau_i / 2          # Eq. 20
    w = np.clip((re_f - 1500.0) / 1000.0, 0.0, 1.0)           # laminar -> turbulent, blended
    tau_f = (1 - w) * tau_f_lam + w * tau_f_turb
    tau_c = f_c * rg * vc * np.abs(vc) / 2
    return tau_f, tau_c, tau_i, sf, sc, si, th


# ------------------------------------------------------------------ slug flow
def _slug_continuity(a, hls, vt, vs, vsl, fe):
    """Film velocity, core holdup and core velocity from Eqs. 1, 3, 7 for film holdup a."""
    k = fe / np.maximum(1 - fe, 1e-9)
    vf = vt - hls * (vt - vs) / a
    hlc = np.zeros_like(vf)
    vc = (vs - a * vf) / (1 - a)
    if not np.any(fe > 0):
        return vf, hlc, vc
    for _ in range(6):
        vc = (vs - a * vf) / (1 - a)
        hlc = np.minimum(np.maximum(k * a * vf / np.where(np.abs(vc) > 1e-9, vc, 1e-9), 0.0), (1 - a) * 0.999)
        hlc = np.where(vf > 0, hlc, 0.0)          # no entrainment from a falling film
        vf = vt - (hls * (vt - vs) - hlc * (vt - vc)) / a
    return vf, hlc, vc


def _slug_residual(a, P):
    vf, hlc, vc = _slug_continuity(a, P["hls"], P["vt"], P["vs"], P["vsl"], P["fe"])
    rl, rg = P["rl"], P["rg"]
    rho_c = (rg * (1 - a - hlc) + rl * hlc) / (1 - a)
    denom = P["vsl"] - a * vf - hlc * vc
    lf = P["ls"] * (P["hls"] * P["vs"] - P["vsl"]) / np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    tau_f, tau_c, tau_i, sf, sc, si, _ = _film_stresses(a, vf, vc, P["vsl"], P["vsg"], rl, rg, rho_c, P["ml"],
                                                         P["mg"], P["sig"], P["d"], P["eps"], P["cos"], P["rg0"])
    area = np.pi * P["d"] ** 2 / 4
    exch = (rl * (P["vt"] - vf) * (P["vs"] - vf) - rho_c * (P["vt"] - vc) * (P["vs"] - vc)) / np.where(
        lf > 0, lf, np.inf)
    r = (exch - tau_f * sf / (a * area) + tau_c * sc / ((1 - a) * area)
         + tau_i * si * (1 / (a * area) + 1 / ((1 - a) * area)) - (rl - rho_c) * G * P["sin"])
    valid = (lf > 0) & (lf < LF_MAX * P["ls"]) & np.isfinite(r) & (vc > 0)
    return r, valid, dict(vf=vf, hlc=hlc, vc=vc, lf=lf, rho_c=rho_c, tau_f=tau_f, tau_c=tau_c, sf=sf, sc=sc)


def _expand(P, m):
    return {k: (np.repeat(v, m) if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 1 else v)
            for k, v in P.items()}


def _root_scan(fun, lo, hi, P, n=N_GRID, iters=22):
    """Vectorised root finding in (lo, hi) per row: the whole grid in one call, the first valid sign change,
    then bisection on that bracket. fun(x, P) -> (residual, valid). Returns x (nan where there is no bracket)."""
    nr = lo.size
    t = np.linspace(0.0, 1.0, n) ** 1.6
    grid = lo[:, None] + (hi - lo)[:, None] * t[None, :]
    Rg, Vg = fun(grid.ravel(), _expand(P, n) if nr > 1 else P)
    R, V = Rg.reshape(nr, n), Vg.reshape(nr, n)
    ok = V[:, :-1] & V[:, 1:] & (np.sign(R[:, :-1]) != np.sign(R[:, 1:]))
    has = ok.any(axis=1)
    k = np.argmax(ok, axis=1)
    r_ = np.arange(nr)
    a, b = grid[r_, k], grid[r_, k + 1]
    fa = R[r_, k]
    for _ in range(iters):
        c = 0.5 * (a + b)
        fc, _v = fun(c, P)
        left = np.sign(fc) == np.sign(fa)
        a = np.where(left, c, a); fa = np.where(left, fc, fa)
        b = np.where(left, b, c)
    return np.where(has, 0.5 * (a + b), np.nan)


def _slug_holdup_zhang(rl, rg, sig, vs, rho_s_fn, ml, d, eps, ce, exch_l, exch_c):
    """Eqs. 55-57 (exch_* are the film momentum-exchange terms already divided by l_S, Pa)."""
    hls = np.full_like(vs, 0.8)
    for _ in range(6):
        rho_s = rho_s_fn(hls)
        f_s = _fanning(rho_s * vs * d / ml, eps / d)
        tsm = (f_s / 2 * rho_s * vs ** 2 + d / 4 * (exch_l + exch_c)) / ce
        hls = 1.0 / (1.0 + tsm / (3.16 * np.sqrt(np.maximum((rl - rg) * G * sig, 1e-12))))
    return np.maximum(hls, HLS_MIN)


# ------------------------------------------------------------------ main
def unified_si(vsl, vsg, rl, rg, ml, mg, sig, d, eps, sin_t, p_pa):
    """Arrays in SI. Returns dict: hl, dpdz_grav, dpdz_fric (Pa/m, positive = pressure loss upwards),
    regime (array of str), plus slug details."""
    vsl, vsg = np.maximum(np.asarray(vsl, float), 1e-9), np.maximum(np.asarray(vsg, float), 0.0)
    shape = np.broadcast(vsl, vsg, rl, rg, ml, mg, sig, p_pa).shape
    b = lambda x: np.broadcast_to(np.asarray(x, float), shape).astype(float).ravel()
    vsl, vsg, rl, rg, ml, mg, sig, p_pa = map(b, (vsl, vsg, rl, rg, ml, mg, sig, p_pa))
    n = vsl.size
    sin_t = float(np.clip(sin_t, -1, 1)); cos_t = float(np.sqrt(max(1 - sin_t ** 2, 0.0)))
    d, eps = float(d), float(eps)
    vs = vsl + vsg
    lam = vsl / np.maximum(vs, 1e-12)
    rg0 = rg * 101325.0 / np.maximum(p_pa, 101325.0)
    area = np.pi * d * d / 4

    hl = np.full(n, np.nan); grav = np.full(n, np.nan); fric = np.full(n, np.nan)
    regime = np.full(n, "", dtype=object)
    extra = {"slug_length_m": np.full(n, np.nan), "film_length_m": np.full(n, np.nan),
             "slug_frequency_1_s": np.full(n, np.nan), "slug_holdup": np.full(n, np.nan),
             "film_holdup": np.full(n, np.nan)}

    def homogeneous(idx, h, label):
        rho = rl[idx] * h + rg[idx] * (1 - h)
        rho_ns = rl[idx] * lam[idx] + rg[idx] * (1 - lam[idx])
        mu = ml[idx] * lam[idx] + mg[idx] * (1 - lam[idx])
        f = _fanning(rho_ns * vs[idx] * d / mu, eps / d)
        hl[idx] = h
        grav[idx] = rho * G * sin_t
        fric[idx] = 2 * f * rho_ns * vs[idx] ** 2 / d
        regime[idx] = label

    single = lam >= 0.99999
    if single.any():
        homogeneous(np.where(single)[0], np.ones(single.sum()), "Liquid")
    gas_only = (lam <= 1e-7) & ~single
    if gas_only.any():
        homogeneous(np.where(gas_only)[0], np.zeros(gas_only.sum()), "Gas")
    todo = ~(single | gas_only)

    # --- dispersed bubble: no-slip liquid fraction at least the slug-body holdup (wall-shear term only)
    ce = (2.5 - abs(sin_t)) / 2
    hls_db = _slug_holdup_zhang(rl, rg, sig, vs, lambda h: rl * h + rg * (1 - h), ml, d, eps, ce, 0.0, 0.0)
    disp = todo & (lam >= hls_db)
    if disp.any():
        homogeneous(np.where(disp)[0], lam[disp], "Dispersed bubble")
    todo &= ~disp

    # --- bubbly (near-vertical upward, large pipe): Taitel et al. (1980)
    vo = 1.53 * (G * np.maximum(rl - rg, 1e-6) * sig / rl ** 2) ** 0.25          # Harmathy, Eq. 16
    vd_taylor = 0.35 * np.sqrt(G * d) * max(sin_t, 0.0)
    if sin_t > 0.5:
        slip = vo * np.sqrt(sin_t)
        h_b = np.clip((-(vs - slip) + np.sqrt(np.maximum((vs - slip) ** 2 + 4 * slip * vsl, 0))) / (2 * slip), lam, 1)
        bub = todo & (vd_taylor > vo) & (1 - h_b < 0.25)
        if bub.any():
            homogeneous(np.where(bub)[0], h_b[bub], "Bubbly")
        todo &= ~bub

    idx = np.where(todo)[0]
    if len(idx):
        P = {k: v[idx] for k, v in dict(vsl=vsl, vsg=vsg, rl=rl, rg=rg, ml=ml, mg=mg, sig=sig, vs=vs,
                                           rg0=rg0).items()}
        P.update(d=d, eps=eps, sin=sin_t, cos=cos_t)
        re_m = P["rl"] * P["vs"] * d / P["ml"]
        cs = np.where(re_m < 2000, 2.0, np.where(re_m > 3000, 1.2, 2.0 - 0.8 * (re_m - 2000) / 1000))
        vd = 0.54 * np.sqrt(G * d) * cos_t + 0.35 * np.sqrt(G * d) * sin_t        # Bendiksen, Eq. 62
        P["vt"] = cs * P["vs"] + vd
        P["ls"] = np.full(len(idx), (32.0 * cos_t ** 2 + 16.0 * sin_t ** 2) * d)  # Eq. 63
        P["fe"] = _entrainment(P["vsl"], P["vsg"], P["rl"], P["rg"], P["ml"], P["mg"], P["sig"], d)
        hls = np.maximum(1.0 / (1.0 + (P["vs"] / 8.66) ** 1.39), HLS_MIN)          # Gregory et al., Eq. 60
        a_sol = np.full(len(idx), np.nan)
        for _ in range(6):
            P["hls"] = hls
            lo = np.full(len(idx), 1e-5)
            # the film length becomes infinite where the film carries the whole liquid flux (Eq. 4):
            # a vF = vSL, i.e. a = (vSL + HLS (vT - vS)) / vT without entrainment - the slug/stratified-annular
            # boundary of the paper. Search below it.
            a_crit = (P["vsl"] + hls * (P["vt"] - P["vs"])) / P["vt"]
            hi = np.clip(np.minimum(np.minimum(hls, 0.9999) * 0.9999, a_crit * (1 - 1e-4)), 2e-5, 0.9999)
            fun = lambda a, PP: _slug_residual(a, PP)[:2]
            if np.isfinite(a_sol).all():          # warm start: bracket around the previous film holdup
                lo2, hi2 = np.maximum(a_sol * 0.8, lo), np.minimum(a_sol * 1.25 + 1e-3, hi)
                a_try = _root_scan(fun, lo2, hi2, P, n=6)
                a_sol = a_try if np.isfinite(a_try).all() else np.where(np.isfinite(a_try), a_try,
                                                                         _root_scan(fun, lo, hi, P))
            else:
                a_sol = _root_scan(fun, lo, hi, P)
            ok = np.isfinite(a_sol)
            a_eval = np.where(ok, a_sol, 0.5 * hls)
            _, _, S = _slug_residual(a_eval, P)
            ex_l = P["rl"] * a_eval * (P["vt"] - S["vf"]) * (P["vs"] - S["vf"]) / P["ls"]
            ex_c = S["rho_c"] * (1 - a_eval) * (P["vt"] - S["vc"]) * (P["vs"] - S["vc"]) / P["ls"]
            new = _slug_holdup_zhang(P["rl"], P["rg"], P["sig"], P["vs"],
                                     lambda h: P["rl"] * h + P["rg"] * (1 - h), P["ml"], d, eps, ce, ex_l, ex_c)
            new = np.where(ok, new, hls)
            if np.nanmax(np.abs(new - hls)) < 1e-3:
                hls = new
                break
            hls = 0.5 * (hls + new)
        P["hls"] = hls
        # slug-body holdup at or below the no-slip liquid fraction: dispersed bubble (same hypothesis as above,
        # now with the film momentum-exchange terms)
        db2 = ~np.isfinite(a_sol) & (hls <= lam[idx] + 1e-9)
        if db2.any():
            homogeneous(idx[db2], lam[idx][db2], "Dispersed bubble")
        slug = np.isfinite(a_sol)
        if slug.any():
            s_idx = idx[slug]
            Ps = {k: (v[slug] if isinstance(v, np.ndarray) else v) for k, v in P.items()}
            a = a_sol[slug]
            _, _, S = _slug_residual(a, Ps)
            ls, lf = Ps["ls"], S["lf"]
            lu = ls + lf
            rho_s = Ps["rl"] * Ps["hls"] + Ps["rg"] * (1 - Ps["hls"])
            f_s = _fanning(rho_s * Ps["vs"] * d / Ps["ml"], eps / d)
            tau_s = f_s * rho_s * Ps["vs"] ** 2 / 2
            rho_film_zone = Ps["rl"] * a + S["rho_c"] * (1 - a)
            hl[s_idx] = (ls * Ps["hls"] + lf * (a + S["hlc"])) / lu
            grav[s_idx] = (ls * rho_s + lf * rho_film_zone) / lu * G * sin_t
            fric[s_idx] = (ls * 4 * tau_s / d + lf * (S["tau_f"] * S["sf"] + S["tau_c"] * S["sc"]) / area) / lu
            regime[s_idx] = "Slug"
            extra["slug_length_m"][s_idx] = ls
            extra["film_length_m"][s_idx] = lf
            extra["slug_frequency_1_s"][s_idx] = Ps["vt"] / lu
            extra["slug_holdup"][s_idx] = Ps["hls"]
            extra["film_holdup"][s_idx] = a
        sa = ~slug & ~db2
        if sa.any():
            _stratified_annular(idx[sa], {k: (v[sa] if isinstance(v, np.ndarray) else v) for k, v in P.items()},
                                hl, grav, fric, regime, sin_t)
    # anything that failed: no-slip fallback
    bad = ~np.isfinite(hl)
    if bad.any():
        homogeneous(np.where(bad)[0], lam[bad], "No-slip (fallback)")
    return dict(hl=hl.reshape(shape), grav=grav.reshape(shape), fric=fric.reshape(shape),
                regime=regime.reshape(shape), **{k: v.reshape(shape) for k, v in extra.items()})


def _stratified_annular(idx, P, hl, grav, fric, regime, sin_t):
    """Eqs. 12-15 with Eq. 7: film holdup from the combined momentum equation without momentum exchange."""
    d = P["d"]; area = np.pi * d * d / 4
    vsl, vsg, rl, rg, fe = P["vsl"], P["vsg"], P["rl"], P["rg"], P["fe"]

    def state(a, PP=P):
        vsl, vsg, rl, rg, fe = PP["vsl"], PP["vsg"], PP["rl"], PP["rg"], PP["fe"]
        vf = vsl * (1 - fe) / a
        vc = (vsg + fe * vsl) / (1 - a)
        hlc = np.clip(fe * vsl / np.maximum(vc, 1e-9), 0, (1 - a) * 0.999)
        rho_c = (rg * (1 - a - hlc) + rl * hlc) / (1 - a)
        tau_f, tau_c, tau_i, sf, sc, si, th = _film_stresses(a, vf, vc, vsl, vsg, rl, rg, rho_c, PP["ml"], PP["mg"],
                                                             PP["sig"], d, PP["eps"], PP["cos"], PP["rg0"])
        r = (-tau_f * sf / (a * area) + tau_c * sc / ((1 - a) * area)
             + tau_i * si * (1 / (a * area) + 1 / ((1 - a) * area)) - (rl - rho_c) * G * PP["sin"])
        return r, dict(vf=vf, vc=vc, hlc=hlc, rho_c=rho_c, tau_f=tau_f, tau_c=tau_c, sf=sf, sc=sc, th=th)

    n = len(idx)
    def fun(x, PP):
        r = state(x, PP)[0]
        return r, np.isfinite(r)
    a = _root_scan(fun, np.full(n, 1e-6), np.full(n, 0.9999), P)
    ok = np.isfinite(a)
    a = np.where(ok, a, np.nan)
    if ok.any():
        k = np.where(ok)[0]
        _, S = state(a[k])
        hl[idx[k]] = a[k] + S["hlc"]
        grav[idx[k]] = (rl[k] * a[k] + S["rho_c"] * (1 - a[k])) * G * sin_t
        fric[idx[k]] = (S["tau_f"] * S["sf"] + S["tau_c"] * S["sc"]) / area
        regime[idx[k]] = np.where(S["th"] >= 0.9, "Annular", "Stratified")


# ------------------------------------------------------------------ field-unit wrapper
def tuffp_unified(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    """Correlation interface (field units: ft/s, lb/ft3, cp, dyne/cm, ft, psia) -> (elevation psi/ft,
    friction psi/ft, holdup)."""
    out = unified_si(np.asarray(vsl, float) * FT, np.asarray(vsg, float) * FT, np.asarray(rho_l, float) * LBFT3,
                     np.maximum(np.asarray(rho_g, float), 1e-4) * LBFT3, np.asarray(mu_l, float) * CP,
                     np.maximum(np.asarray(mu_g, float), 1e-4) * CP, np.maximum(np.asarray(sig_l, float), 0.5) * DYNECM,
                     float(d) * FT, float(rough) * FT, float(sin_t), np.asarray(p, float) * 6894.757)
    return (out["grav"] * PA_PER_M_TO_PSI_PER_FT, out["fric"] * PA_PER_M_TO_PSI_PER_FT, out["hl"])


def tuffp_details(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    """Same inputs as tuffp_unified; returns the full result dict (regime, slug length and frequency...)."""
    return unified_si(np.asarray(vsl, float) * FT, np.asarray(vsg, float) * FT, np.asarray(rho_l, float) * LBFT3,
                      np.maximum(np.asarray(rho_g, float), 1e-4) * LBFT3, np.asarray(mu_l, float) * CP,
                      np.maximum(np.asarray(mu_g, float), 1e-4) * CP, np.maximum(np.asarray(sig_l, float), 0.5) * DYNECM,
                      float(d) * FT, float(rough) * FT, float(sin_t), np.asarray(p, float) * 6894.757)
