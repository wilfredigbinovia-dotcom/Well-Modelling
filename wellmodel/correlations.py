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


CORRELATIONS = {
    "beggs_brill": beggs_brill,
    "hagedorn_brown": hagedorn_brown,
    "no_slip": no_slip,
}

LABELS = {
    "beggs_brill": "Beggs & Brill",
    "hagedorn_brown": "Hagedorn & Brown (mod.)",
    "no_slip": "No-slip homogeneous",
}
