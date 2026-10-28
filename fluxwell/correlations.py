"""Multiphase-flow pressure-gradient correlations (vectorised).

Every function takes superficial velocities and fluid properties as numpy
arrays and returns (elevation_gradient, friction_gradient, liquid_holdup)
with gradients in psi/ft along measured depth, positive for upward
production flow. Multipliers ("parameter 1/2" in PROSPER terms) are applied
by the caller: dp/dL = gravity_mult * elevation + friction_mult * friction.

Implemented
-----------
beggs_brill     Beggs & Brill (1973) with Payne et al. (1979) holdup correction
hagedorn_brown  Modified Hagedorn & Brown (1965) with Griffith bubble flow
                and the no-slip holdup floor (Brill & Mukherjee, 1999)
no_slip         Homogeneous no-slip model (lower bound on holdup)
gray            Gray (1974) for gas/condensate wells
mukherjee_brill Mukherjee & Brill (1985), uphill coefficients
orkiszewski     Orkiszewski (1967): Griffith bubble, Griffith-Wallis slug, Duns & Ros mist
aziz            Aziz, Govier & Fogarasi (1972) with Duns & Ros mist
beggs_brill_noslip  Beggs & Brill friction with no-slip holdup
"""
from __future__ import annotations

import numpy as np

G = 32.174          # ft/s2
CP_TO_LBFTS = 6.7197e-4


def friction_factor(re, rel_rough):
    """Darcy-Moody friction factor: laminar 64/Re, Chen (1979) explicit
    turbulent, linear blend through the 2000-4000 transition."""
    re = np.maximum(np.asarray(re, float), 1.0)
    rr = np.maximum(np.asarray(rel_rough, float), 1e-9)
    a = rr / 3.7065 - 5.0452 / re * np.log10(rr ** 1.1098 / 2.8257 + 5.8506 / re ** 0.8981)
    f_turb = (-2.0 * np.log10(np.maximum(a, 1e-12))) ** -2
    f_lam = 64.0 / re
    w = np.clip((re - 2000.0) / 2000.0, 0.0, 1.0)
    return (1 - w) * f_lam + w * f_turb


def _no_slip_lambda(vsl, vsg):
    vm = np.maximum(vsl + vsg, 1e-9)
    return vm, np.clip(vsl / vm, 1e-9, 1.0)


# ---------------------------------------------------------------------------
def no_slip(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    vm, lam = _no_slip_lambda(vsl, vsg)
    rho_n = rho_l * lam + rho_g * (1 - lam)
    mu_n = mu_l * lam + mu_g * (1 - lam)
    re = rho_n * vm * d / (mu_n * CP_TO_LBFTS)
    f = friction_factor(re, rough / d)
    fric = f * rho_n * vm * vm / (2 * G * d) / 144.0
    elev = rho_n * sin_t / 144.0
    return elev, fric, lam


def foam_homogeneous(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    """PROSPER's Foam-Homogeneous (foam lift, Tutorial 48): the no-slip mixture of gas and foam (rho_l is the foam
    density, passed by vlp._gradient) for gravity, and no wall friction (see foam.py)."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    rho_n = rho_l * lam + rho_g * (1 - lam)
    return rho_n * sin_t / 144.0, 0.0 * rho_n, lam


# ---------------------------------------------------------------------------
_BB = {  # a, b, c, d', e, f, g (uphill)
    "seg": (0.98, 0.4846, 0.0868, 0.011, -3.768, 3.539, -1.614),
    "int": (0.845, 0.5351, 0.0173, 2.96, 0.305, -0.4473, 0.0978),
    "dis": (1.065, 0.5824, 0.0609, None, None, None, None),
}


def beggs_brill(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    vm, lam = _no_slip_lambda(vsl, vsg)
    nfr = np.maximum(vm * vm / (G * d), 1e-12)
    l1 = 316.0 * lam ** 0.302
    l2 = 0.0009252 * lam ** -2.4684
    l3 = 0.10 * lam ** -1.4516
    l4 = 0.5 * lam ** -6.738
    seg = ((lam < 0.01) & (nfr < l1)) | ((lam >= 0.01) & (nfr < l2))
    trans = (lam >= 0.01) & (nfr >= l2) & (nfr <= l3)
    dist = ((lam < 0.4) & (nfr >= l1)) | ((lam >= 0.4) & (nfr > l4))
    nlv = np.maximum(1.938 * vsl * (rho_l / sig_l) ** 0.25, 1e-12)
    theta = np.arcsin(np.clip(sin_t, -1, 1))
    s18 = np.sin(1.8 * theta)

    def holdup(key):
        a, b, c, d_, e, f, g = _BB[key]
        hl0 = np.maximum(a * lam ** b / nfr ** c, lam)
        if d_ is None:
            C = 0.0
        else:
            arg = np.maximum(d_ * lam ** e * nlv ** f * nfr ** g, 1e-300)
            C = np.maximum((1 - lam) * np.log(arg), 0.0)
        return hl0 * (1 + C * (s18 - 0.333 * s18 ** 3))

    h_seg, h_int, h_dis = holdup("seg"), holdup("int"), holdup("dis")
    A = np.clip((l3 - nfr) / np.maximum(l3 - l2, 1e-12), 0, 1)
    hl = np.select([seg, dist, trans], [h_seg, h_dis, A * h_seg + (1 - A) * h_int], default=h_int)
    hl = np.where(sin_t >= 0, 0.924 * hl, 0.685 * hl)          # Payne correction
    hl = np.clip(hl, lam, 1.0)
    hl = np.where(lam > 0.99999, 1.0, hl)

    rho_n = rho_l * lam + rho_g * (1 - lam)
    mu_n = mu_l * lam + mu_g * (1 - lam)
    re = rho_n * vm * d / (mu_n * CP_TO_LBFTS)
    fn = friction_factor(re, rough / d)
    y = np.maximum(lam / (hl * hl), 1e-12)
    ly = np.log(y)
    den = -0.0523 + 3.182 * ly - 0.8725 * ly ** 2 + 0.01853 * ly ** 4
    S = np.where(np.abs(den) > 1e-9, ly / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
    S = np.where((y > 1.0) & (y < 1.2), np.log(np.maximum(2.2 * y - 1.2, 1e-12)), S)
    ftp = fn * np.exp(np.clip(S, -5, 5))
    fric = ftp * rho_n * vm * vm / (2 * G * d) / 144.0
    rho_s = rho_l * hl + rho_g * (1 - hl)
    elev = rho_s * sin_t / 144.0
    return elev, fric, hl


# ---------------------------------------------------------------------------
def hagedorn_brown(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    vm, lam = _no_slip_lambda(vsl, vsg)
    k = (rho_l / sig_l) ** 0.25
    nlv = 1.938 * vsl * k
    ngv = np.maximum(1.938 * vsg * k, 1e-9)
    nd = 120.872 * d * np.sqrt(rho_l / sig_l)
    nl = 0.15726 * mu_l * (1.0 / (rho_l * sig_l ** 3)) ** 0.25
    cnl = ((0.0019 + 0.0322 * nl - 0.6642 * nl ** 2 + 4.9951 * nl ** 3)
           / (1 - 10.0147 * nl + 33.8696 * nl ** 2 + 277.2817 * nl ** 3))
    cnl = np.clip(cnl, 0.0019, 0.0115)
    x1 = nlv / ngv ** 0.575 * (np.maximum(p, 14.7) / 14.7) ** 0.1 * cnl / nd
    hl_psi = np.sqrt(np.clip((0.0047 + 1123.32 * x1 + 729489.64 * x1 ** 2)
                             / (1 + 1097.1566 * x1 + 722153.97 * x1 ** 2), 0, 1))
    x2 = ngv * nl ** 0.380 / nd ** 2.14
    psi = ((1.0886 - 69.9473 * x2 + 2334.3497 * x2 ** 2 - 12896.683 * x2 ** 3)
           / (1 - 53.4401 * x2 + 1517.9369 * x2 ** 2 - 8419.8115 * x2 ** 3))
    psi = np.where(x2 < 0.01, 1.0, np.clip(psi, 1.0, 1.83))
    hl_hb = np.clip(hl_psi * psi, lam, 1.0)

    # Griffith bubble-flow region
    lb = np.maximum(1.071 - 0.2218 * vm * vm / d, 0.13)
    bubble = (vsg / vm) < lb
    vs = 0.8
    disc = np.maximum((1 + vm / vs) ** 2 - 4 * vsg / vs, 0.0)
    hl_gb = np.clip(1 - 0.5 * (1 + vm / vs - np.sqrt(disc)), lam, 1.0)
    hl = np.where(bubble, hl_gb, hl_hb)
    hl = np.where(lam > 0.99999, 1.0, hl)

    rho_s = rho_l * hl + rho_g * (1 - hl)
    rho_n = rho_l * lam + rho_g * (1 - lam)
    # Hagedorn-Brown friction
    mu_s = mu_l ** hl * mu_g ** (1 - hl)
    re = rho_n * vm * d / (mu_s * CP_TO_LBFTS)
    f = friction_factor(re, rough / d)
    fric_hb = f * rho_n ** 2 * vm * vm / (2 * G * d * rho_s) / 144.0
    # Griffith friction on in-situ liquid velocity
    vl = vsl / hl
    re_l = rho_l * vl * d / (mu_l * CP_TO_LBFTS)
    f_l = friction_factor(re_l, rough / d)
    fric_gb = f_l * rho_l * vl * vl / (2 * G * d) / 144.0
    fric = np.where(bubble, fric_gb, fric_hb)
    elev = rho_s * sin_t / 144.0
    return elev, fric, hl


# ---------------------------------------------------------------------------
DYNE_TO_LBM_S2 = 0.00220462      # 1 dyne/cm = 1 g/s2 = 0.0022046 lbm/s2


def gray(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    """Gray (1974, API 14B) for gas wells producing condensate and/or water.
    Holdup from the velocity and diameter numbers; friction on the no-slip mixture with an
    effective roughness set by the liquid film (Brill & Mukherjee, 1999)."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    sig = np.maximum(sig_l, 1.0) * DYNE_TO_LBM_S2
    drho = np.maximum(rho_l - rho_g, 1e-6)
    rho_n = rho_l * lam + rho_g * (1 - lam)
    nv = rho_n ** 2 * vm ** 4 / (G * sig * drho)
    nd = G * drho * d * d / sig
    r = vsl / np.maximum(vsg, 1e-12)
    b = 0.0814 * (1 - 0.0554 * np.log(1 + 730 * r / (r + 1)))
    f1 = -2.314 * (nv * (1 + 205.0 / nd)) ** b
    hl = np.clip(1 - (1 - lam) * (1 - np.exp(f1)), lam, 1.0)
    ke0 = 28.5 * sig / (rho_n * np.maximum(vm, 1e-6) ** 2)
    ke = np.where(r >= 0.007, ke0, rough + r * (ke0 - rough) / 0.007)
    ke = np.clip(np.maximum(ke, 2.77e-5), rough, 0.5 * d)
    mu_n = mu_l * lam + mu_g * (1 - lam)
    re = rho_n * vm * d / (mu_n * CP_TO_LBFTS)
    f = friction_factor(re, ke / d)
    fric = f * rho_n * vm * vm / (2 * G * d) / 144.0
    rho_s = rho_l * hl + rho_g * (1 - hl)
    return rho_s * sin_t / 144.0, fric, hl


_MB_UP = (-0.380113, 0.129875, -0.119788, 2.343227, 0.475686, 0.288657)
_MB_HR = (np.array([0.01, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 10.0]),
          np.array([1.00, 0.98, 1.20, 1.25, 1.30, 1.25, 1.00, 1.00]))


def mukherjee_brill(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    """Mukherjee & Brill (1985), uphill flow: empirical holdup, bubble/slug/annular pattern
    transitions; annular friction corrected by the holdup ratio."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    k = (rho_l / sig_l) ** 0.25
    nlv = np.maximum(1.938 * vsl * k, 1e-12)
    ngv = np.maximum(1.938 * vsg * k, 1e-12)
    nl = 0.15726 * mu_l * (1.0 / (rho_l * sig_l ** 3)) ** 0.25
    st = np.clip(sin_t, 0.0, 1.0)
    c1, c2, c3, c4, c5, c6 = _MB_UP
    hl = np.exp((c1 + c2 * st + c3 * st * st + c4 * nl * nl) * ngv ** c5 / nlv ** c6)
    hl = np.clip(hl, lam, 1.0)
    ngv_sm = 10 ** (1.401 - 2.694 * nl + 0.521 * nlv ** 0.329)
    annular = ngv > ngv_sm
    rho_n = rho_l * lam + rho_g * (1 - lam)
    mu_n = mu_l * lam + mu_g * (1 - lam)
    re = rho_n * vm * d / (mu_n * CP_TO_LBFTS)
    fn = friction_factor(re, rough / d)
    rho_s = rho_l * hl + rho_g * (1 - hl)
    fric_bs = fn * rho_s * vm * vm / (2 * G * d) / 144.0
    fr = np.interp(lam / np.maximum(hl, 1e-12), *_MB_HR)
    fric_an = fn * fr * rho_n * vm * vm / (2 * G * d) / 144.0
    fric = np.where(annular, fric_an, fric_bs)
    return rho_s * sin_t / 144.0, fric, hl


# ---------------------------------------------------------------------------
# Duns & Ros mist flow (used by Orkiszewski and Aziz et al. in the mist region)
def _dr_mist(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, mixture=False):
    """No-slip holdup; friction on the gas phase with a film roughness from the Weber number. mixture=True (PE2):
    the friction term carries the no-slip mixture density and velocity instead of the gas alone - the liquid's
    momentum, which makes the pressure drop rise with water as PROSPER's PE2 does (Tutorial 06 well, WGR 0-50)."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    sig = np.maximum(sig_l, 0.5) * DYNE_TO_LBM_S2
    vg2 = np.maximum(vsg, 1e-6) ** 2
    nwnm = rho_g * vg2 * (mu_l * CP_TO_LBFTS) ** 2 / (rho_l * sig * sig)
    # film roughness in the field-unit form (sigma in dyne/cm, rho lb/ft3, v ft/s -> ft; i.e. 34 sigma/(rho v^2) in
    # consistent units x 1/453.6) - the form PROSPER uses: it reproduces PE2's mist friction in PROSPER Tutorial 02
    sig_d = np.maximum(sig_l, 0.5)
    eps = np.where(nwnm <= 0.005, 0.0749 * sig_d / (rho_g * vg2), 0.3713 * sig_d * nwnm ** 0.302 / (rho_g * vg2))
    eps = np.clip(np.maximum(eps, rough), 1e-9, 0.25 * d)
    db = d - eps                                       # gas core diameter d - eps (Duns & Ros refinement)
    vsg_b = np.maximum(vsg, 1e-6) * d * d / (db * db)
    rr = np.clip(eps / db, 1e-6, 0.5)
    re_g = rho_g * vsg_b * db / (mu_g * CP_TO_LBFTS)
    f_moody = friction_factor(re_g, rr)
    f_rough = 4 * (1 / (4 * np.log10(0.27 * rr)) ** 2 + 0.067 * rr ** 1.73)
    f = np.where(rr > 0.05, f_rough, f_moody)
    rho_n = rho_l * lam + rho_g * (1 - lam)
    if mixture:
        fric = f * rho_n * (vm * d * d / (db * db)) ** 2 / (2 * G * db) / 144.0
    else:
        fric = f * rho_g * vsg_b ** 2 / (2 * G * db) / 144.0
    return rho_n * sin_t / 144.0, fric, lam


# ---------------------------------------------------------------------------
def _griffith_bubble(vsl, vsg, rho_l, rho_g, mu_l, d, rough, sin_t):
    vm = vsl + vsg
    vs = 0.8
    disc = np.maximum((1 + vm / vs) ** 2 - 4 * vsg / vs, 0.0)
    hl = np.clip(1 - 0.5 * (1 + vm / vs - np.sqrt(disc)), 0.0, 1.0)
    vl = vsl / np.maximum(hl, 1e-9)
    f = friction_factor(rho_l * vl * d / (mu_l * CP_TO_LBFTS), rough / d)
    rho_s = rho_l * hl + rho_g * (1 - hl)
    return rho_s * sin_t / 144.0, f * rho_l * vl * vl / (2 * G * d) / 144.0, hl


def _orkiszewski_slug(vsl, vsg, rho_l, rho_g, mu_l, d, rough, sin_t, water_cont):
    vm = np.maximum(vsl + vsg, 1e-9)
    nre_l = 1488.0 * rho_l * vm * d / mu_l
    sgd = np.sqrt(G * d)
    vb = (0.546 + 8.74e-6 * nre_l) * sgd                          # first guess (NReb <= 3000)
    for _ in range(3):
        nre_b = 1488.0 * rho_l * vb * d / mu_l
        vbs = (0.251 + 8.74e-6 * nre_l) * sgd
        vb = np.where(nre_b <= 3000, (0.546 + 8.74e-6 * nre_l) * sgd,
                      np.where(nre_b >= 8000, (0.35 + 8.74e-6 * nre_l) * sgd,
                               0.5 * (vbs + np.sqrt(vbs ** 2 + 13.59 * mu_l / (rho_l * np.sqrt(d))))))
    lg_mu, lg_mu1, lg_d, lg_vm = np.log10(np.maximum(mu_l, 1e-6)), np.log10(mu_l + 1), np.log10(d), np.log10(vm)
    g_w_lo = 0.013 * lg_mu / d ** 1.38 - 0.681 + 0.232 * lg_vm - 0.428 * lg_d
    g_w_hi = 0.045 * lg_mu / d ** 0.799 - 0.709 - 0.162 * lg_vm - 0.888 * lg_d
    g_o_lo = 0.0127 * lg_mu1 / d ** 1.415 - 0.284 + 0.167 * lg_vm + 0.113 * lg_d
    x = -lg_vm * (0.01 * lg_mu1 / d ** 1.571 + 0.397 + 0.63 * lg_d)
    g_o_hi = 0.0274 * lg_mu1 / d ** 1.371 + 0.161 + 0.569 * lg_d + x
    lo = vm < 10
    gam = np.where(water_cont, np.where(lo, g_w_lo, g_w_hi), np.where(lo, g_o_lo, g_o_hi))
    rho_s0 = (rho_l * (vsl + vb) + rho_g * vsg) / (vm + vb)
    gam = np.where(lo, np.maximum(gam, -0.065 * vm),
                   np.maximum(gam, -vb / (vm + vb) * (1 - rho_s0 / rho_l)))
    rho_s = np.clip(rho_s0 + rho_l * gam, rho_g, rho_l)
    f = friction_factor(nre_l, rough / d)
    fric = f * rho_l * vm * vm / (2 * G * d) * ((vsl + vb) / (vm + vb) + gam) / 144.0
    hl = (rho_s - rho_g) / np.maximum(rho_l - rho_g, 1e-6)
    return rho_s * sin_t / 144.0, np.maximum(fric, 0.0), hl


def orkiszewski(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p, water_cut=None):
    """Orkiszewski (1967): Griffith bubble flow, Griffith-Wallis slug flow with the liquid distribution
    coefficient, Duns & Ros transition and mist flow. Pattern boundaries: Griffith L_B, Duns & Ros L_S, L_M."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    k = (rho_l / sig_l) ** 0.25
    nlv, ngv = 1.938 * vsl * k, 1.938 * vsg * k
    lb = np.maximum(1.071 - 0.2218 * vm * vm / d, 0.13)
    ls = 50 + 36 * nlv
    lm = 75 + 84 * nlv ** 0.75
    wc = rho_l > 60.0 if water_cut is None else water_cut >= 0.75
    eb, fb, hb = _griffith_bubble(vsl, vsg, rho_l, rho_g, mu_l, d, rough, sin_t)
    es, fs, hs = _orkiszewski_slug(vsl, vsg, rho_l, rho_g, mu_l, d, rough, sin_t, wc)
    em, fm, hm = _dr_mist(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t)
    a = np.clip((lm - ngv) / np.maximum(lm - ls, 1e-9), 0, 1)
    bubble = (vsg / vm) < lb
    slug, mist = ~bubble & (ngv < ls), ngv > lm
    pick = lambda xb, xs, xm: np.where(bubble, xb, np.where(slug, xs, np.where(mist, xm, a * xs + (1 - a) * xm)))
    hl = np.clip(pick(hb, hs, hm), lam, 1.0)
    rho_s = rho_l * hl + rho_g * (1 - hl)        # no-slip floor: the slug-flow liquid distribution coefficient
    return rho_s * sin_t / 144.0, pick(fb, fs, fm), hl   # can otherwise give a mixture lighter than no-slip


def aziz(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    """Aziz, Govier & Fogarasi (1972): mechanistic bubble and slug flow on the Aziz flow map, Duns & Ros mist,
    interpolated transition."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    sig = np.maximum(sig_l, 0.5) * DYNE_TO_LBM_S2
    drho = np.maximum(rho_l - rho_g, 1e-6)
    fac = ((rho_l / 62.4) * (72.0 / np.maximum(sig_l, 0.5))) ** 0.25
    nx = vsg * (rho_g / 0.0764) ** (1 / 3) * fac
    ny = vsl * fac
    n1 = 0.51 * (100 * ny) ** 0.172
    n2 = 8.6 + 3.8 * ny
    n3 = 70 * (100 * np.maximum(ny, 1e-9)) ** -0.152
    mu_lb = mu_l * CP_TO_LBFTS
    re_l = rho_l * vm * d / mu_lb
    f = friction_factor(re_l, rough / d)
    # bubble flow
    vbf = 1.41 * (sig * G * drho / rho_l ** 2) ** 0.25
    hg_b = np.clip(vsg / (1.2 * vm + vbf), 0, 1 - lam * 0)
    hl_b = np.clip(1 - hg_b, lam, 1.0)
    rho_b = rho_l * hl_b + rho_g * (1 - hl_b)
    eb, fb = rho_b * sin_t / 144.0, f * rho_b * vm * vm / (2 * G * d) / 144.0
    # slug flow (Taylor bubble rise velocity, Wallis coefficient)
    ne = G * d * d * drho / sig
    nv = np.sqrt(d ** 3 * G * rho_l * drho) / mu_lb
    m = np.where(nv >= 250, 10.0, np.where(nv > 18, 69.0 * nv ** -0.35, 25.0))
    c = 0.345 * (1 - np.exp(-0.029 * nv)) * (1 - np.exp((3.37 - ne) / m))
    vb = c * np.sqrt(G * d * drho / rho_l)
    hl_s = np.clip(1 - vsg / (1.2 * vm + vb), lam, 1.0)
    rho_s = rho_l * hl_s + rho_g * (1 - hl_s)
    es, fs = rho_s * sin_t / 144.0, f * rho_l * hl_s * vm * vm / (2 * G * d) / 144.0
    em, fm, hm = _dr_mist(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t)
    hi = ny >= 4
    bubble = nx < n1
    slug = ~bubble & np.where(hi, nx < 26.5, nx < n2)
    mist = np.where(hi, nx >= 26.5, nx > n3)
    a = np.clip((n3 - nx) / np.maximum(n3 - n2, 1e-9), 0, 1)
    pick = lambda xb, xs, xm: np.where(bubble, xb, np.where(slug, xs, np.where(mist, xm, a * xs + (1 - a) * xm)))
    return pick(eb, es, em), pick(fb, fs, fm), np.clip(pick(hl_b, hl_s, hm), lam, 1.0)


def beggs_brill_noslip(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    """Beggs & Brill with no-slip holdup (no inclination correction) and the Beggs & Brill two-phase friction
    factor evaluated at H_L = lambda."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    rho_n = rho_l * lam + rho_g * (1 - lam)
    mu_n = mu_l * lam + mu_g * (1 - lam)
    fn = friction_factor(rho_n * vm * d / (mu_n * CP_TO_LBFTS), rough / d)
    y = np.maximum(1 / lam, 1.0 + 1e-12)
    ly = np.log(y)
    den = -0.0523 + 3.182 * ly - 0.8725 * ly ** 2 + 0.01853 * ly ** 4
    S = np.where(np.abs(den) > 1e-9, ly / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
    S = np.where((y > 1.0) & (y < 1.2), np.log(np.maximum(2.2 * y - 1.2, 1e-12)), S)
    fric = fn * np.exp(np.clip(S, -5, 5)) * rho_n * vm * vm / (2 * G * d) / 144.0
    return rho_n * sin_t / 144.0, fric, lam


# ---------------------------------------------------------------------------
# Duns & Ros (1963) charts, digitised from Brill & Beggs, Two-Phase Flow in Pipes (1991), Figs 3-11 to 3-14.
_LOG = np.log10
_DR_ND = (np.array([10, 26.4, 30, 32.7, 39.4, 47.6, 52.3, 57.5, 63.0, 70, 300]),            # L1 (Fig 3-11)
          np.array([2.0, 2.0, 1.946, 1.839, 1.570, 1.283, 1.151, 1.050, 1.014, 1.0, 1.0]))
_DR_L2 = (np.array([14.8, 20.4, 29.7, 39.4, 47.6, 57.5, 69.3, 83.6, 300]),
          np.array([0.470, 0.566, 0.756, 0.925, 0.996, 1.057, 1.093, 1.100, 1.10]))
_NL1 = np.array([2.0e-3, 4.2e-3, 1.13e-2, 2.53e-2, 3.3e-2, 4.32e-2, 6.76e-2, 0.106, 0.166, 0.237, 0.34, 0.486,
                 0.70, 1.0, 1.43, 1.88])
_F1 = np.array([1.238, 1.238, 1.266, 1.306, 1.365, 1.492, 1.743, 1.948, 2.055, 2.036, 1.888, 1.652, 1.383,
                1.143, 0.978, 0.863])
_NL2 = np.array([2.24e-3, 1.47e-2, 1.94e-2, 2.76e-2, 4.32e-2, 6.76e-2, 9.7e-2, 0.138, 0.198, 0.286, 0.486, 1.0, 1.88])
_F2 = np.array([0.239, 0.239, 0.258, 0.352, 0.549, 0.749, 0.915, 0.987, 1.013, 0.987, 0.915, 0.783, 0.701])
_NL3 = np.array([2.01e-3, 4.6e-3, 6.6e-3, 8.6e-3, 1.13e-2, 1.77e-2, 2.76e-2, 4.32e-2, 6.76e-2, 9.7e-2, 0.166,
                 0.406, 1.0, 2.05])
_F3 = np.array([0.826, 0.856, 0.915, 1.093, 1.335, 1.744, 2.177, 2.623, 2.997, 3.276, 3.52, 3.74, 3.95, 4.06])
_NL4 = np.array([2.0e-3, 2.93e-3, 4.6e-3, 7.2e-3, 1.13e-2, 1.77e-2, 2.76e-2, 4.32e-2, 6.76e-2, 0.166, 0.406, 2.05])
_F4 = np.array([-18.9, -6.4, 5.2, 15.8, 26.4, 36.0, 44.1, 50.5, 54.3, 56.6, 56.3, 55.3])
_NL5 = np.array([1.85e-3, 4.76e-3, 9.8e-3, 1.85e-2, 2.9e-2, 4.15e-2, 5.4e-2, 7.1e-2, 8.5e-2, 0.102, 0.122, 0.146,
                 0.176, 0.275, 0.433, 0.68, 1.06, 1.9])
_F5 = np.array([0.2245, 0.207, 0.190, 0.168, 0.154, 0.134, 0.112, 0.0859, 0.0687, 0.0575, 0.049, 0.0459, 0.047,
                0.0575, 0.0703, 0.0840, 0.0982, 0.1175])
_NL6 = np.array([1.98e-3, 3.03e-3, 4.76e-3, 7.5e-3, 1.17e-2, 1.47e-2, 1.85e-2, 2.4e-2, 3.47e-2, 4.96e-2, 7.1e-2,
                 8.5e-2, 0.107, 0.146, 0.21, 0.30, 0.52, 1.06, 1.83])
_F6 = np.array([0.829, 0.459, 0.167, -0.008, -0.144, -0.183, -0.163, -0.008, 0.537, 1.062, 1.626, 1.996, 2.101,
                2.035, 1.918, 1.821, 1.743, 1.724, 1.735])
_NL7 = np.array([2.07e-3, 4.76e-3, 9.8e-3, 1.85e-2, 3.47e-2, 5.9e-2, 0.102, 0.176, 0.36, 0.74, 1.4, 3.2])
_F7 = np.array([0.1331, 0.110, 0.0878, 0.0703, 0.0549, 0.0470, 0.0407, 0.0351, 0.0300, 0.0263, 0.0248, 0.0244])
_F2V = (np.array([1e-3, 7.7e-3, 8.5e-2, 0.364, 0.58, 1.0, 1.82, 3.16, 5.75, 10.5, 19, 35, 63.3, 101, 380, 1000]),
        np.array([1.0, 1.037, 1.050, 1.056, 1.006, 0.839, 0.658, 0.517, 0.405, 0.334, 0.282, 0.242, 0.217, 0.211,
                  0.214, 0.217]))


def _li(x, xs, ys):
    """Interpolation in log10(x), clamped to the chart range."""
    return np.interp(_LOG(np.clip(x, xs[0], xs[-1])), _LOG(xs), ys)


def _dr_holdup(s_num, vsl, vsg, k):
    vs = np.maximum(s_num / (1.938 * k), 1e-9)          # slip velocity from S = vs (rho_L / (g sigma))^0.25
    vm = vsl + vsg
    return np.clip((vs - vm + np.sqrt((vm - vs) ** 2 + 4 * vs * vsl)) / (2 * vs), 0.0, 1.0)


# Gould, Tek & Katz (1974) vertical flow-regime map (JPT Aug 1974, Fig. 2), digitised: Wallis bubble boundary,
# the dashed bubble/slug transition line, and the plug (slug) region for 1-in. and 3.15-in. pipe.
_G_PLUG_1 = (np.array([0.1, 0.184, 0.384, 0.706, 1.0]), np.array([0.475, 0.475, 0.406, 0.268, 0.148]))
_G_PLUG_315 = (np.array([0.1, 0.184, 0.384, 0.798, 1.30, 1.88]), np.array([0.80, 0.858, 0.900, 0.829, 0.616, 0.285]))


def _gould_plug_ng(nl, d_in):
    """Gas velocity number at the plug/slug boundary (nan above the top of the curve)."""
    w = np.clip(np.log(np.maximum(d_in, 1e-3)) / np.log(3.15), 0.0, 1.0)
    g1 = np.where(nl <= 1.0, _li(nl, *_G_PLUG_1), np.nan)
    g3 = np.where(nl <= 1.88, _li(nl, *_G_PLUG_315), np.nan)
    top = 1.0 + w * 0.88
    out = 10 ** ((1 - w) * _LOG(np.where(np.isnan(g1), g3, g1)) + w * _LOG(np.where(np.isnan(g3), g1, g3)))
    return np.where(nl <= top, out, np.nan)


def duns_ros_gould(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    """Duns & Ros modified: Duns & Ros holdup on the Gould, Tek & Katz (1974) flow-regime map (Wallis bubble
    boundary, holdup interpolated across the bubble/slug transition band, plug region treated as slug), with
    Kleyweg friction in bubble and slug flow - a single-phase friction factor on the average (mixture) velocity
    with holdup-weighted density and viscosity. The froth region has no drawn boundary in the paper and is
    treated as slug flow; mist flow keeps the Duns & Ros gas-film friction."""
    return duns_ros(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p, gould=True)


def duns_ros(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p, gould=False):
    """Duns & Ros (1963), original: bubble/slug slip from the F1-F7 charts, bubble/slug friction
    f1 f2 / f3 on the liquid, Duns & Ros mist flow, and linear interpolation through the transition zone
    (with the modified gas density rho_g * Ngv / Lm in the mist part)."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    k = (rho_l / sig_l) ** 0.25
    nlv, ngv = 1.938 * vsl * k, 1.938 * vsg * k
    nd = 120.872 * d * np.sqrt(rho_l / sig_l)
    nl = 0.15726 * mu_l * (1.0 / (rho_l * sig_l ** 3)) ** 0.25
    l1, l2 = _li(nd, *_DR_ND), _li(nd, *_DR_L2)
    ls, lm = 50 + 36 * nlv, 75 + 84 * nlv ** 0.75
    f1, f2, f3, f4 = _li(nl, _NL1, _F1), _li(nl, _NL2, _F2), _li(nl, _NL3, _F3), _li(nl, _NL4, _F4)
    f5, f6, f7 = _li(nl, _NL5, _F5), _li(nl, _NL6, _F6), _li(nl, _NL7, _F7)
    s_b = f1 + f2 * nlv + (f3 - f4 / nd) * (ngv / (1 + nlv)) ** 2
    s_s = (1 + f5) * (ngv ** 0.982 + 0.029 * nd + f6) / (1 + f7 * nlv) ** 2
    hl_b = np.maximum(_dr_holdup(s_b, vsl, vsg, k), lam)
    hl_s = np.maximum(_dr_holdup(s_s, vsl, vsg, k), lam)
    # bubble / slug friction on the liquid phase
    fm1 = friction_factor(rho_l * vsl * d / (mu_l * CP_TO_LBFTS), rough / d)
    x2 = fm1 * vsg * nd ** (2 / 3) / np.maximum(vsl, 1e-9)
    ff2 = _li(x2, *_F2V)
    ff3 = 1 + fm1 * np.sqrt(vsg / (50 * np.maximum(vsl, 1e-9)))
    fric_bs = fm1 * ff2 / ff3 * rho_l * vsl * vm / (2 * G * d) / 144.0
    rho_sb = rho_l * hl_b + rho_g * (1 - hl_b)
    rho_ss = rho_l * hl_s + rho_g * (1 - hl_s)
    em, fm, hm = _dr_mist(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t)
    rho_gt = rho_g * ngv / np.maximum(lm, 1e-9)          # transition: modified gas density in the mist part
    emt, fmt, _ = _dr_mist(vsl, vsg, rho_l, np.maximum(rho_gt, rho_g), mu_l, mu_g, sig_l, d, rough, sin_t)
    mist = ngv >= lm
    a = np.clip((lm - ngv) / np.maximum(lm - ls, 1e-9), 0, 1)
    if gould:
        nlc = np.maximum(nlv, 1e-6)
        ng_b = 0.148 * nlc ** 0.98                        # Wallis bubble boundary
        ng_d = 0.62 * (nlc / 1.30) ** 0.984               # dashed bubble/slug transition line
        ng_p = _gould_plug_ng(nlc, d * 12)
        plug = np.isfinite(ng_p) & (ngv < ng_p)
        bubble = ~plug & (ngv <= ng_b)
        band = ~plug & ~bubble & (ngv < ng_d) & (nlc >= 1.0)
        wb = np.clip(_LOG(np.maximum(ngv, 1e-12) / ng_b) / np.maximum(_LOG(ng_d / ng_b), 1e-9), 0, 1)
        hl_s = np.where(band, (1 - wb) * hl_b + wb * hl_s, hl_s)        # interpolated holdup in the band
        rho_ss = rho_l * hl_s + rho_g * (1 - hl_s)
        slug = ~bubble & (ngv <= ls)
        # Kleyweg friction: single-phase (Moody) friction factor on the average (mixture) velocity, with
        # holdup-weighted mixture density and viscosity, instead of the two-phase f1 f2 / f3 correction
        hl_f = np.where(bubble, hl_b, hl_s)
        rho_m = rho_l * hl_f + rho_g * (1 - hl_f)
        mu_m = mu_l * hl_f + mu_g * (1 - hl_f)
        f_k = friction_factor(rho_m * vm * d / (mu_m * CP_TO_LBFTS), rough / d)
        fric_bs = f_k * rho_m * vm * vm / (2 * G * d) / 144.0
    else:
        bubble = ngv <= l1 + l2 * nlv
        slug = ~bubble & (ngv <= ls)
    es, eb = rho_ss * sin_t / 144.0, rho_sb * sin_t / 144.0
    elev = np.where(bubble, eb, np.where(slug, es, np.where(mist, em, a * es + (1 - a) * emt)))
    fric = np.where(bubble | slug, fric_bs, np.where(mist, fm, a * fric_bs + (1 - a) * fmt))
    hl = np.where(bubble, hl_b, np.where(slug, hl_s, np.where(mist, hm, a * hl_s + (1 - a) * hm)))
    return elev, np.maximum(fric, 0.0), np.clip(hl, lam, 1.0)


# ---------------------------------------------------------------------------
# Fancher & Brown (1963), SPEJ March 1963 pp. 59-69: no-slip density, energy-loss factor correlated against
# 1.4737e-5 M q / d for three producing gas-liquid ratio ranges (Figs 7-9, digitised).
_FB_LOW = (np.array([3.55, 5.21, 7.66, 11.2, 16.5, 24.3, 33.2, 72.2]),
           np.array([0.245, 0.0931, 0.0551, 0.035, 0.0228, 0.0142, 0.0097, 0.00427]))


def fancher_brown_f(x, glr):
    """Fanning-type friction factor. x = 1.4737e-5 M q / d (lb/d per ft), glr in scf/bbl of liquid."""
    x = np.maximum(x, 0.5)
    lo = 10 ** np.interp(_LOG(x), _LOG(_FB_LOW[0]), _LOG(_FB_LOW[1]),
                         left=None, right=None)
    # log-linear extrapolation beyond the digitised range
    lo = np.where(x < _FB_LOW[0][0], _FB_LOW[1][0] * (x / _FB_LOW[0][0]) ** -2.6, lo)
    lo = np.where(x > _FB_LOW[0][-1], _FB_LOW[1][-1] * (x / _FB_LOW[0][-1]) ** -1.15, lo)
    mid = 0.0551 * (x / 4.71) ** -1.153
    hi = 0.0357 * (x / 3.74) ** -1.26
    f = np.where(glr < 1500, lo, np.where(glr < 3000, mid, hi))
    return np.clip(f, 1e-4, 2.0)


def fancher_brown(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p, glr=None):
    """dp/dH = (1/144) [rho_ns + f (qM)^2 / (7.413e10 rho_ns d^5)]; no-slip holdup, roughness-independent
    friction. glr: producing gas-liquid ratio (scf/bbl); without it the in-situ ratio is used."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    rho_n = rho_l * lam + rho_g * (1 - lam)
    w = (rho_l * vsl + rho_g * vsg) * np.pi * d * d / 4 * 86400.0          # lb/d  (= q M)
    if glr is None:
        glr = 5.615 * vsg / np.maximum(vsl, 1e-12)
    f = fancher_brown_f(1.4737e-5 * w / d, glr)
    fric = f * w * w / (7.413e10 * rho_n * d ** 5) / 144.0
    return rho_n * sin_t / 144.0, fric, lam


NEEDS_GLR = {"fancher_brown"}


# ---------------------------------------------------------------------------
# Ansari, Sylvester, Sarica, Shoham & Brill (1994), SPE Production & Facilities: mechanistic model.
def _bisect(fun, lo, hi, n=50):
    """Vectorised bisection for a function that is positive at lo and negative at hi."""
    lo, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
    lo, hi = lo.copy(), hi.copy()
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        pos = fun(mid) > 0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def _ansari_annular(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig, d, rough, sin_t):
    """Alves et al. film/core model. Returns (dp/dz psi/ft, elevation psi/ft, holdup, film fraction, XM2, YM)."""
    mul, mug = mu_l * CP_TO_LBFTS, mu_g * CP_TO_LBFTS
    vcrit = 1e4 * vsg * mug / sig * np.sqrt(rho_g / rho_l)
    fe = np.clip(1 - np.exp(-0.125 * (vcrit - 1.5)), 0.0, 0.999)          # Wallis entrainment
    vsc = vsg + fe * vsl
    lam_c = fe * vsl / np.maximum(vsc, 1e-12)
    rho_c = rho_l * lam_c + rho_g * (1 - lam_c)
    mu_c = mul * lam_c + mug * (1 - lam_c)
    f_sc = friction_factor(rho_c * vsc * d / mu_c, rough / d)             # Darcy
    dp_sc = f_sc * rho_c * vsc ** 2 / (2 * d)                             # lbm/(ft2 s2) per ft
    vsl_f = np.maximum(vsl * (1 - fe), 1e-9)
    f_sl = friction_factor(rho_l * vsl_f * d / mul, rough / d)
    dp_sl = f_sl * rho_l * vsl_f ** 2 / (2 * d)
    xm2 = dp_sl / np.maximum(dp_sc, 1e-12)
    ym = G * sin_t * (rho_l - rho_c) / np.maximum(dp_sc, 1e-12)
    zf = lambda dl: np.where(fe > 0.9, 1 + 300 * dl, 1 + 24 * (rho_l / rho_g) ** (1 / 3) * dl)

    def resid(dl):                     # combined film/core momentum balance (positive for too-thin film)
        af = 4 * dl * (1 - dl)
        ac = (1 - 2 * dl) ** 2
        vf = vsl_f / af
        f_f = friction_factor(rho_l * vf * af * d / mul, rough / d)
        tau_f = f_f / 4 * rho_l * vf ** 2 / 2
        vc = vsc / ac
        tau_i = zf(dl) * f_sc / 4 * rho_c * vc ** 2 / 2
        si_ac = 4 * (1 - 2 * dl) / (d * ac)
        si_af = 4 * (1 - 2 * dl) / (d * af)
        sf_af = 4 / (d * af)
        return tau_f * sf_af - tau_i * (si_af + si_ac) - (rho_l - rho_c) * G * sin_t
    dl = _bisect(resid, 1e-7, 0.4999)
    ac = (1 - 2 * dl) ** 2
    vc = vsc / ac
    tau_i = zf(dl) * f_sc / 4 * rho_c * vc ** 2 / 2
    fric = tau_i * 4 * (1 - 2 * dl) / (d * ac) / (G * 144.0)
    elev = rho_c * sin_t / 144.0
    alpha_f = 4 * dl * (1 - dl)
    hl = alpha_f + lam_c * ac
    return elev, fric, hl, alpha_f, xm2, ym


def ansari(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    """Ansari et al. (1994) mechanistic model for upward flow in wells.
    Patterns: Taitel-Dukler bubble/slug, Barnea dispersed bubble (void < 0.76), Taitel/Barnea annular stability.
    Bubble: drift flux v_g = 1.2 vm + v_inf H_L^0.5; dispersed bubble: no slip; slug: Fernandes et al. / Sylvester
    slug unit with Taylor bubble rise, Brotz falling film and slug-body mass balance; annular: Alves et al."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    vsl = np.maximum(vsl, 1e-9)
    sig = np.maximum(sig_l, 0.5) * DYNE_TO_LBM_S2
    drho = np.maximum(rho_l - rho_g, 1e-6)
    mul, mug = mu_l * CP_TO_LBFTS, mu_g * CP_TO_LBFTS
    rho_n = rho_l * lam + rho_g * (1 - lam)
    mu_n = mul * lam + mug * (1 - lam)
    v_inf = 1.53 * (G * sig * drho / rho_l ** 2) ** 0.25                 # Harmathy bubble rise
    # ---- annular (computed first: its film fraction decides the annular/slug transition)
    ea, fa, ha, alpha_f, xm2, ym = _ansari_annular(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig, d, rough, sin_t)
    taitel = vsg > 3.1 * (G * sig * drho / rho_g ** 2) ** 0.25
    barnea_ok = (alpha_f <= 0.12) & (ym <= (2 - 1.5 * alpha_f) / (alpha_f ** 3 * (1 - 1.5 * alpha_f)) * xm2)
    annular = taitel & barnea_ok
    # ---- dispersed bubble (Barnea)
    f_m = friction_factor(rho_n * vm * d / mu_n, rough / d) / 4         # Fanning
    lhs = 2 * (0.4 * sig / (drho * G)) ** 0.5 * (rho_l / sig) ** 0.6 * (2 * f_m / d) ** 0.4 * vm ** 1.2
    dispersed = ~annular & (lhs > 0.725 + 4.15 * np.sqrt(vsg / vm)) & ((vsg / vm) < 0.76)
    # ---- bubble (Taitel): only in pipes large enough for bubbles to exist
    big = d > 19.01 * np.sqrt(drho * sig / (rho_l ** 2 * G))
    bubble = ~annular & ~dispersed & big & (vsg < 0.25 * v_inf + 0.333 * vsl)
    slug = ~annular & ~dispersed & ~bubble
    # bubble flow: H_g from v_sg / H_g = 1.2 vm + v_inf (1 - H_g)^0.5
    hg_b = _bisect(lambda hg: vsg - hg * (1.2 * vm + v_inf * np.sqrt(1 - hg)), 0.0, 0.9999)
    hl_b = np.clip(1 - hg_b, lam, 1.0)
    rho_b = rho_l * hl_b + rho_g * (1 - hl_b)
    f_b = friction_factor(rho_l * vm * d / mul, rough / d)
    eb, fb = rho_b * sin_t / 144.0, f_b * rho_b * vm ** 2 / (2 * G * d) / 144.0
    # dispersed bubble: no slip
    f_d = friction_factor(rho_n * vm * d / mu_n, rough / d)
    ed, fd = rho_n * sin_t / 144.0, f_d * rho_n * vm ** 2 / (2 * G * d) / 144.0
    # slug flow
    v_tb = 1.2 * vm + 0.35 * np.sqrt(G * d * drho / rho_l)
    hg_ls = np.clip(vsg / (0.425 + 2.65 * vm), 0.0, 0.9)                  # Sylvester
    hl_ls = 1 - hg_ls
    v_gls = 1.2 * vm + v_inf * np.sqrt(hl_ls)
    v_lls = (vm - v_gls * hg_ls) / hl_ls
    lhs_l = np.maximum((v_tb - v_lls) * hl_ls, 1e-9)
    # liquid balance in the bubble frame: (v_TB - v_LLS) H_LLS = (v_TB + v_LTB) H_LTB, Brotz film velocity
    hl_tb = _bisect(lambda h: lhs_l - (v_tb + 9.916 * np.sqrt(G * d * (1 - np.sqrt(np.clip(1 - h, 0, 1))))) * h,
                    1e-6, 1.0)
    v_ltb = 9.916 * np.sqrt(G * d * (1 - np.sqrt(np.clip(1 - hl_tb, 0, 1))))
    beta = np.clip((v_lls * hl_ls - vsl) / np.maximum(v_lls * hl_ls + v_ltb * hl_tb, 1e-9), 0.0, 1.0)
    rho_ls = rho_l * hl_ls + rho_g * hg_ls
    mu_ls = mul * hl_ls + mug * hg_ls
    f_ls = friction_factor(rho_ls * vm * d / mu_ls, rough / d)
    es = ((1 - beta) * rho_ls + beta * rho_g) * sin_t / 144.0
    fs = f_ls * rho_ls * vm ** 2 * (1 - beta) / (2 * G * d) / 144.0
    hl_s = (1 - beta) * hl_ls + beta * hl_tb
    elev = np.select([annular, dispersed, bubble], [ea, ed, eb], default=es)
    fric = np.select([annular, dispersed, bubble], [fa, fd, fb], default=fs)
    hl = np.select([annular, dispersed, bubble], [ha, lam, hl_b], default=hl_s)
    return elev, np.maximum(fric, 0.0), np.clip(hl, 0.0, 1.0)



# ---------------------------------------------------------------------------
PE2_SLUG_SLIP = 0.155          # ft/s - slug-flow slip velocity calibrated on PROSPER PE2 gradients (Abura 2S)
PE2_SLUG_GRAVITY = 1.0145      # slug-flow static gradient / in-situ density, same calibration (T23: 1.046 / P1 1.03108)
PE2_SLUG_FRICTION = (1.5270, -0.1205, -0.0819)   # slug friction / Hagedorn & Brown friction
#                                = (a + b HL) vsl^c (vsl ft/s, held to 1-20): PROSPER PE2 node exports T23 (5 and 20 MMscf/d),
#                                T33a QuickLook, Tutorial 11 at 80 % and 5 % water cut (HL 0.21-0.87, vsl 4.8-12.5 ft/s,
#                                vm 6-85 ft/s) - segment friction within 4 %, rms 1.7 %, then 2 % up to balance the full
#                                traverses (T33a and Tutorial 11 within 6 psi, T23 within 16 psi = 1 %)
PE2_BAND = (1.75, 1.44, 0.241, 0.4797)   # slug friction step: where Ngv < 1.75 + 1.44 Nlv (a line in the Duns & Ros
#                                velocity numbers) and the no-slip gas fraction is above 0.241, x 1 + 0.4797 (lambda_g - 0.241)
#                                - PROSPER PE2 friction steps up 10-20 % across that line (T33a, T23 5 MMscf/d, Tutorial 11)
PE2_HB_R = (-0.08631087, -1.84429638, -8.66210136)   # PE2 / our HB chart fit, quadratic in ln X (T23 traverses)
PE2_HB_U = (-10.684034, -9.199713)                    # ln X: vertex (held below) and where the ratio returns to 1
PE2_HB_LOW = (0.00717988, 0.2452605, 2.0209254, -11.0)  # below ln X = -11 (X < 1.7e-5): HL = a u^2 + b u + c (u = ln X)
#                                where smaller - PROSPER PE2 at very low liquid velocity numbers (Tutorial 42 steam, Nlv 0.2-1,
#                                Ngv 25-42: HL 0.016-0.16 against the chart's 0.07 floor; fitted within 0.001). Steam only
#                                (hb_low=True): on gas-lifted oil wells (T23 equilibrium curve) PROSPER keeps the chart value


def _pe2_hb_holdup(vsl, vsg, rho_l, mu_l, sig_l, d, p, low=False):
    """Hagedorn & Brown holdup as PROSPER's PE2 gives it at high gas velocities: our HB chart fit (HL/psi vs X) is
    up to 19 % low at the left end of the chart (X = 2e-5 - 1e-4) against PROSPER PE2 gradients (Tutorial 23, 5 and
    20 MMscf/d of lift gas, vsg 11-82 ft/s, with PROSPER's liquid surface tension - vlp.sigma_liquid) - corrected
    there by a ratio fitted on those traverses (within 0.2 %)."""
    k = (rho_l / sig_l) ** 0.25
    nlv = 1.938 * vsl * k
    ngv = np.maximum(1.938 * vsg * k, 1e-9)
    nd = 120.872 * d * np.sqrt(rho_l / sig_l)
    nl = 0.15726 * mu_l * (1.0 / (rho_l * sig_l ** 3)) ** 0.25
    cnl = ((0.0019 + 0.0322 * nl - 0.6642 * nl ** 2 + 4.9951 * nl ** 3)
           / (1 - 10.0147 * nl + 33.8696 * nl ** 2 + 277.2817 * nl ** 3))
    cnl = np.clip(cnl, 0.0019, 0.0115)
    x1 = nlv / ngv ** 0.575 * (np.maximum(p, 14.7) / 14.7) ** 0.1 * cnl / nd
    hl_psi = np.sqrt(np.clip((0.0047 + 1123.32 * x1 + 729489.64 * x1 ** 2)
                             / (1 + 1097.1566 * x1 + 722153.97 * x1 ** 2), 0, 1))
    x2 = ngv * nl ** 0.380 / nd ** 2.14
    psi = ((1.0886 - 69.9473 * x2 + 2334.3497 * x2 ** 2 - 12896.683 * x2 ** 3)
           / (1 - 53.4401 * x2 + 1517.9369 * x2 ** 2 - 8419.8115 * x2 ** 3))
    psi = np.where(x2 < 0.01, 1.0, np.clip(psi, 1.0, 1.83))
    ux = np.log(np.maximum(x1, 1e-30))
    u = np.clip(ux, PE2_HB_U[0], PE2_HB_U[1])
    r = np.maximum(np.polyval(PE2_HB_R, u), 1.0)
    hl = hl_psi * r * psi
    a_, b_, c_, u_lo = PE2_HB_LOW
    low_hl = np.maximum(a_ * ux * ux + b_ * ux + c_, 0.0) * psi
    hl = np.where(ux < u_lo, np.minimum(hl, low_hl), hl) if low else hl
    return np.clip(hl, 0.0, 1.0)


def pe2_style(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p, hb_low=False):
    """PE2-style: a public-domain stand-in for Petroleum Experts' proprietary PE2, built from the pieces PROSPER's
    gradient output shows it uses and calibrated against PROSPER PE2 gradients.

    Bubble flow (Griffith & Wallis limit, gas fraction below max(1.071 - 0.2218 vm^2/d, 0.13)): Griffith holdup
    (slip 0.8 ft/s) and Griffith friction - as in the modified Hagedorn & Brown. Slug flow: holdup from a constant
    slip velocity (0.155 ft/s, at least the no-slip holdup) or, where larger (high gas velocities), the Hagedorn &
    Brown holdup as PROSPER's PE2 gives it (_pe2_hb_holdup); a static gradient 1.45 % above the in-situ density (as
    PROSPER PE2 reports); friction the Hagedorn & Brown friction times (1.527 - 0.1205 HL) vsl^-0.0819, stepped up by
    1 + 0.4797 (lambda_g - 0.241) where the gas velocity number is below 1.75 + 1.44 Nlv (PROSPER T23, T33a and
    Tutorial 11 node exports, segment friction within 4 %). Transition and
    mist (Duns & Ros map): no-slip holdup with the Duns & Ros mist friction (film roughness). NOT Petroleum
    Experts' PE2 - an approximation of it."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    e_hb, f_hb, hl_hb = hagedorn_brown(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p)
    e_dr, f_dr, hl_dr = duns_ros(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p)
    lb = np.maximum(1.071 - 0.2218 * vm * vm / d, 0.13)
    bubble = (vsg / np.maximum(vm, 1e-12)) < lb
    k = (rho_l / sig_l) ** 0.25
    nlv, ngv = 1.938 * vsl * k, 1.938 * vsg * k
    ls = 50 + 36 * nlv
    beyond_slug = ngv >= ls
    # slug: drift with a small constant slip
    vs = PE2_SLUG_SLIP
    disc = np.maximum((1 + vm / vs) ** 2 - 4 * vsg / vs, 0.0)
    hl_s = np.clip(1 - 0.5 * (1 + vm / vs - np.sqrt(disc)), lam, 1.0)
    # at high gas velocities PROSPER's PE2 holdup follows Hagedorn & Brown (its chart), above the small-slip value
    hl_s = np.maximum(hl_s, _pe2_hb_holdup(vsl, vsg, rho_l, mu_l, sig_l, d, p, hb_low))
    # slug friction: Hagedorn & Brown's form (no-slip density squared over the slip density, two-phase viscosity
    # mu_L^HL mu_G^(1-HL)) times (1.527 - 0.1205 HL) vsl^-0.0819 - PROSPER PE2 gradients (T23, T33a, Tutorial 11)
    rho_ss = rho_l * hl_s + rho_g * (1 - hl_s)
    rho_nn = rho_l * lam + rho_g * (1 - lam)
    mu_ss = mu_l ** hl_s * mu_g ** (1 - hl_s)
    ff_s = friction_factor(rho_nn * vm * d / (mu_ss * CP_TO_LBFTS), rough / d)
    a_, b_, c_ = PE2_SLUG_FRICTION
    f_s = ((a_ + b_ * hl_s) * np.clip(vsl, 1.0, 20.0) ** c_ * ff_s * rho_nn ** 2 * vm * vm / (2 * G * d * rho_ss)
           / 144.0)
    # PROSPER's PE2 slug friction steps up 10-20 % where the gas velocity number falls below 1.75 + 1.44 Nlv, the
    # excess fading linearly with the no-slip gas fraction (T33a, T23 5 MMscf/d and Tutorial 11 at 5 % water cut)
    lam_g = 1.0 - lam
    a_b, b_b, lo, slope = PE2_BAND
    step = (ngv < a_b + b_b * nlv) & (lam_g > lo)
    f_s = f_s * np.where(step, 1.0 + slope * (lam_g - lo), 1.0)
    e_s = PE2_SLUG_GRAVITY * rho_ss * sin_t / 144.0
    # beyond the Duns & Ros slug boundary PROSPER's PE2 reports no-slip holdup and the Duns & Ros mist (film
    # roughness) friction, in the transition region too (Tutorial 02 gas well: holdup = no-slip, friction within 1 %)
    e_m, f_m, hl_m = _dr_mist(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, mixture=True)
    e = np.where(bubble, e_hb, np.where(beyond_slug, e_m, e_s))
    f = np.where(bubble, f_hb, np.where(beyond_slug, f_m, f_s))
    hl = np.where(bubble, hl_hb, np.where(beyond_slug, hl_m, hl_s))
    single = lam > 0.99999
    return np.where(single, e_hb, e), np.where(single, f_hb, f), np.where(single, 1.0, hl)


# ---------------------------------------------------------------------------
PE5_SLUG_SLIP = 0.32           # ft/s - slug-flow slip velocity, calibrated on PROSPER PE5 gradients (Abura 2S)
PE5_DRIFT = (1.04213, -0.30582, 1.70985, 0.63258)  # vsg/(1-HL) = (a + b lambda) vm + c lambda^n (ft/s)
PE5_LOW_VM_DRIFT = (0.65, 10.0, 20.0)  # drift at least 0.65 ft/s below vm = 10 ft/s, fading out by 20 ft/s - PROSPER PE5 at
#                                low mixture velocity (Tutorials 41 and 43, 6-9 ft/s, 7-12 % liquid: drift 0.56-0.74 ft/s where
#                                c lambda^n gives 0.3-0.4; holdup was 14-21 % low, now within 3 % on average)
PE5_LOW_VM_RHO = 6.0          # lb/ft3 - the low-velocity drift scales down with the gas density below this (T47 lifted
#                                low rates at 350 psig and T43 at 100 psig need much less)
PE5_DRIFT_CAP = 3.0            # drift-flux holdup at most 3 x the no-slip holdup (dry gas tends to no slip)
PE5_CAP_RHO = (2.218, 0.626)   # ... x max(1, (2.218 / rho_g)^0.626) at low gas density: PROSPER's T43 gradients at 100 psig
#                                (rho_g 0.5-1 lb/ft3, 50-200 ft/s) carry 5-8 x the no-slip holdup (the 3 x cap was 30-60 % low)
PE5_INCL_FRICTION = 0.447      # slug friction x (1 - 0.447 sin^2(deviation)): PROSPER's PE5 friction gradient steps down where
#                                the T43 well turns off vertical - x 0.89 at 30 deg, 0.665 at 60 deg (0.60 at 85 deg; this gives 0.555)
PE5_FRICTION = (-2.28432, 0.29878, 0.10340, -0.72371, 0.32939)  # slug friction / Moody =
#   exp(a + b HL + c ln vm + d ln mu_g + e ln(rho_g / rho_l))
#   both fitted on PROSPER PE5 gradients: Tutorial 38 (384 nodes, condensate, 2-9 % liquid, 19-115 ft/s, 560-4020 psig) and
#   Tutorial 47 (four gradients, 197 nodes, water-rich, 3-38 % liquid, 5-53 ft/s, 360-4015 psig), with Abura 2S


def pe5_style(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    """PE5-style: a calibrated approximation of Petroleum Experts' proprietary PE5, from what PROSPER's PE5
    gradient output shows (Abura 2S oil well; Tutorial 38 condensate well, 3.99 in tubing, 20-115 ft/s). NOT PE5
    itself.

    Bubble flow (Griffith & Wallis limit, gas fraction below max(1.071 - 0.2218 vm^2/d, 0.13)): Griffith holdup
    (0.8 ft/s slip) and Griffith friction - PE5 reports this region as dispersed bubble with exactly that slip.
    Everything else is reported by PE5 as slug flow, even at 110 ft/s gas velocity with 2 % liquid where Duns & Ros
    would give mist (Tutorial 38): holdup the larger of a constant 0.32 ft/s slip (low velocity, Abura 2S) and a
    drift flux vsg/(1-HL) = (1.0421 - 0.3058 lambda) vm + 1.710 lambda^0.6326 ft/s (the drift grows with the liquid
    fraction; at least 0.65 ft/s below 10 ft/s mixture velocity, fading out by 20 ft/s - Tutorials 41 / 43; capped at
    3 x the no-slip holdup, more below 2.2 lb/ft3 of gas - Tutorial 43 at 100 psig); static gradient from the in-situ density; friction =
    exp(-2.284 + 0.299 HL + 0.1034 ln vm - 0.724 ln mu_g + 0.329 ln(rho_g/rho_l)) x the Moody friction of the slug
    mixture (in-situ density, mixture velocity, liquid viscosity), x (1 - 0.447 sin^2 deviation) off vertical (Tutorial 43
    at 30 / 60 / 85 deg). Fitted on PROSPER's PE5 gradients of Tutorial 38
    (condensate, 2-9 % liquid, 19-115 ft/s, BO and EoS, 8 traverses) and Tutorial 47 (water-rich, 3-38 % liquid,
    5-53 ft/s, 5 traverses), 624 nodes: holdup within 2.1 % (T38) / 5.2 % (T47) at every node, pressure drop per
    traverse within 3.8 % (T47 within 2.2 %); Abura 2S holdup within 0.6 %, total gradient within 2 %."""
    vm, lam = _no_slip_lambda(vsl, vsg)
    e_hb, f_hb, hl_hb = hagedorn_brown(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p)
    lb = np.maximum(1.071 - 0.2218 * vm * vm / d, 0.13)
    bubble = (vsg / np.maximum(vm, 1e-12)) < lb
    vs = PE5_SLUG_SLIP
    disc = np.maximum((1 + vm / vs) ** 2 - 4 * vsg / vs, 0.0)
    hl_slip = 1 - 0.5 * (1 + vm / vs - np.sqrt(disc))
    a0, a1, c_d, n_d = PE5_DRIFT
    v0, vlo, vhi = PE5_LOW_VM_DRIFT
    vd = np.maximum(c_d * lam ** n_d, v0 * np.clip((vhi - vm) / (vhi - vlo), 0.0, 1.0)
                    * np.clip(rho_g / PE5_LOW_VM_RHO, 0.0, 1.0))
    fade = np.clip((vhi - vm) / (vhi - vlo), 0.0, 1.0)
    k_cap = PE5_DRIFT_CAP * np.maximum(1.0, (PE5_CAP_RHO[0] / np.maximum(rho_g, 1e-3)) ** PE5_CAP_RHO[1])
    cap = np.where(fade > 0, 1.0, k_cap * lam)                  # the no-slip cap only at high mixture velocity
    hl_free = 1 - vsg / np.maximum((a0 + a1 * lam) * vm + vd, 1e-9)
    hl_df = np.minimum(hl_free, cap)
    hl_s = np.clip(np.maximum(hl_slip, hl_df), lam, 1.0)
    rho_s = rho_l * hl_s + rho_g * (1 - hl_s)
    # friction on the 3 x no-slip capped holdup, as fitted (T38 / T47): the extra low-density holdup is static only -
    # with it in the friction density the T43 100 psig friction came out 30-70 % high
    hl_f = np.clip(np.maximum(hl_slip, np.minimum(hl_free, np.where(fade > 0, 1.0, PE5_DRIFT_CAP * lam))), lam, 1.0)
    rho_f = rho_l * hl_f + rho_g * (1 - hl_f)
    f_m = friction_factor(rho_f * vm * d / (mu_l * CP_TO_LBFTS), rough / d)
    a_, b_, c_, d_, e_ = PE5_FRICTION
    mult = np.exp(a_ + b_ * hl_f + c_ * np.log(np.maximum(vm, 1e-6)) + d_ * np.log(np.maximum(mu_g, 1e-4))
                  + e_ * np.log(np.maximum(rho_g, 1e-3) / np.maximum(rho_l, 1e-3)))
    s2 = 1.0 - np.minimum(np.asarray(sin_t, float) ** 2, 1.0)    # sin^2 of the deviation from vertical
    f_s = (1.0 - PE5_INCL_FRICTION * s2) * mult * f_m * rho_f * vm * vm / (2 * G * d) / 144.0
    e_s = rho_s * sin_t / 144.0
    e = np.where(bubble, e_hb, e_s)
    f = np.where(bubble, f_hb, f_s)
    hl = np.where(bubble, hl_hb, hl_s)
    single = lam > 0.99999
    return np.where(single, e_hb, e), np.where(single, f_hb, f), np.where(single, 1.0, hl)


from .unified import tuffp_unified  # noqa: E402  (unified uses friction_factor above)

class _Correlations(dict):
    """The selectable correlations; special-purpose ones (EXTRA: foam lift) are found by name but not listed."""

    def __missing__(self, key):
        return EXTRA[key]


EXTRA = {"foam_homogeneous": foam_homogeneous}

CORRELATIONS = _Correlations({
    "tuffp_unified": tuffp_unified,
    "pe5_style": pe5_style,
    "pe2_style": pe2_style,
    "beggs_brill": beggs_brill,
    "hagedorn_brown": hagedorn_brown,
    "no_slip": no_slip,
    "gray": gray,
    "mukherjee_brill": mukherjee_brill,
    "orkiszewski": orkiszewski,
    "aziz": aziz,
    "beggs_brill_noslip": beggs_brill_noslip,
    "duns_ros": duns_ros,
    "duns_ros_gould": duns_ros_gould,
    "fancher_brown": fancher_brown,
    "ansari": ansari,
})

LABELS = {
    "foam_homogeneous": "Foam-Homogeneous",
    "tuffp_unified": "TUFFP Unified (Zhang et al. 2003)",
    "pe5_style": "PE5-style (calibrated approximation)",
    "pe2_style": "PE2-style (calibrated approximation)",
    "beggs_brill": "Beggs & Brill",
    "hagedorn_brown": "Hagedorn & Brown (mod.)",
    "no_slip": "No-slip (lower bound)",
    "gray": "Gray",
    "mukherjee_brill": "Mukherjee & Brill",
    "orkiszewski": "Orkiszewski",
    "aziz": "Aziz, Govier & Fogarasi",
    "beggs_brill_noslip": "Beggs & Brill (no-slip)",
    "duns_ros": "Duns & Ros (original)",
    "duns_ros_gould": "Duns & Ros (modified)",
    "fancher_brown": "Fancher & Brown",
    "ansari": "Ansari",
}
