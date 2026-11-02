"""PE5 Extended-style: a calibrated approximation of Petroleum Experts' proprietary PE5 Extended (PROSPER), from what
its gradient output shows. NOT PE5 Extended itself - its equations are not published.

PROSPER describes PE5 Extended as PE5's momentum balances with the flow regime found as part of the solution, an
improved slug model, droplet entrainment / deposition and a liquid-film stability calculation, more stable at low rates.
Fitted on the user's T43 runs (gas condensate, 3.99 in tubing, 6.4 in casing below 11,800 ft): 30 PE5 Extended gradient
traverses - 500 psig at 1.5-5 MMscf/d, 100 and 1000 psig at 1.5 / 3 / 5 MMscf/d, and 60 deg below 2000 ft at
1.5-5 MMscf/d - with PE5 alongside at 500 psig. Closures from PROSPER's own node properties (static and friction
gradients from its cumulative losses), the constants then checked on the traverses' bottom-hole pressures.

  Against PE5: no low-rate film reversal (PE5's negative friction and high holdup at 1.5-2.5 MMscf/d / 500 psig: BHP
  1557 against Extended's 1411 psig), and a slug -> annular transition PE5 does not make (BHP 1161 against 980 psig at
  4 MMscf/d).

  regime    annular when the Barnea liquid-film ratio (PE5 stability ratio, K = 55) reaches PE5X_FILM_RATIO = 2.74:
            both vertical boundaries in the first set sit at 2.73-2.77 (500 psig / 3.5 MMscf/d, 3 % liquid; 1000 psig /
            5 MMscf/d, 6 % liquid); the liquid Froude number vsl (rho_l / (drho g d))^0.5 is at least PE5X_FL_MIN
            (near-dry flow stays slug); and the no-slip liquid fraction is at most PE5X_LIQ_MAX = 0.65 (PROSPER's oil
            annular flow reaches 62 % liquid - the 30 % cut of the PE5 stability check does not apply). Off vertical
            the threshold is x (1 + 1.0 sin^2 deviation). Every node of the second and third sets (T43 / T38 at 30 and
            60 deg, T13 oil, T38 at 100 psig) as PROSPER. Not reproduced: PROSPER's "Slug (Induced)" at 35-82 % liquid
            in the GOR 800 oil run (14,482 STB/d), which depends on the flow below (bubble flow in the casing), not on
            local conditions.
  slug      drift flux vsg / (1 - HL) = C0 vm + vd, C0 = 1.0233 - 0.08 sin^2 dev, vd = exp(-1.2868) (drho / rho_g)^0.1512
            vm^-0.0283 (1 + 4.2985 sin^2 dev) ft/s - fitted on the holdup PROSPER's gravity term implies (its reported
            holdup is 1-6 % lower), all three sets: every run within 6.5 % (the earlier vd, (drho/rho_g)^0.398, was
            19 % high at 1 lb/ft3 of gas). Friction = Moody (no-slip mixture) x exp(1.9621) vm^-0.7953
            lambda^0.1026 (rho_g / rho_l)^-0.1057 exp(-0.2142 sin^2 dev) (18 % rms; condensate and oil: PROSPER's
            oil-well slug friction is a steady 0.64 x no-slip Moody).
  annular   holdup between the slug holdup at the transition and its high-velocity limit (1.055 + 0.12 lambda)
            lambda (PROSPER's gravity-implied annular holdup settles at 1.05-1.11 x no slip): HL = lambda_a +
            w (HL_slug - lambda_a), w = min(1, exp(-3.714) (Fr_g - 0.94)^-1.071), Fr_g = vsg (rho_g / (drho g d))^0.5.
            PROSPER's annular holdup also develops along the pipe (entrainment / deposition), which a local closure
            approximates. Friction = Moody (no-slip) x exp(-0.351) Fr_g^-0.3747 lambda^-0.0383
            max(HL / lambda_a, 1)^-1.1328 (rho_g / rho_l)^-0.1343 (15.5 % rms over all annular nodes; Fr_g 1-5.7).
  bubble / single phase as PE5-style.

Data (the user's PROSPER PE5 Extended exports): 22 T43 traverses (first set); T43 at 30 / 60 deg, 8-20 MMscf/d, T38 at
60 deg / 25 MMscf/d and the T13 oil well with VLP tables at water cut 0 / 50 % and GOR 2000 (second); T13 at 2000-16,000
STB/d and GOR 2000 / 50 % water, T38's black-oil condensate at 100 psig vertical (1.5-5 MMscf/d) and 60 deg (10, 15)
(third). Bottom-hole pressures: 47 points, mean 3.4 %, max 11.6 % (T43 100 psig / 3 MMscf/d on PROSPER's EoS fluid -
7 % on the black-oil fluid); T13 oil within 5 %; 500 psig vertical within 3.3 %, 1000 psig 1.9 %.

A screening approximation: check it against PROSPER before relying on it, oil wells rest on one gradient and three VLP tables."""
from __future__ import annotations

import numpy as np

from . import complementary as CX
from .correlations import _no_slip_lambda, friction_factor, pe5_style, CP_TO_LBFTS

G_FT = 32.174
PE5X_FILM_RATIO = 2.74
PE5X_FILM_RATIO_DEV = 1.0          # threshold x (1 + k sin^2 deviation)
PE5X_FL_MIN = 0.08
PE5X_LIQ_MAX = 0.65                 # annular possible up to this no-slip liquid fraction (oil wells: 0.62 seen)
PE5X_C0 = (1.0233, 0.08)           # C0 = a - b sin^2 dev
PE5X_VD = (-1.2868, 0.1512, -0.0283, 4.2985)     # vd = exp(a) (drho/rho_g)^b vm^c (1 + k sin^2 dev), ft/s
PE5X_SLUG_FRICTION = (1.9621, -0.7953, 0.1026, -0.1057, -0.2142)  # x Moody_ns: exp(a) vm^b lam^c (rg/rl)^d exp(e s2)
PE5X_ANN_W = (-3.714, -1.071, 0.94)        # w = exp(a) (Fr - c)^b
PE5X_ANN_FLOOR = (1.055, 0.12)             # high-velocity annular holdup = (a + b lambda) lambda
PE5X_ANN_FRICTION = (-0.351, -0.3747, -0.0383, -1.1328, -0.1343) # x Moody_ns: exp(a) Fr^b lam^c max(HL/lam,1)^d (rg/rl)^e
_FILM_H = np.geomspace(1e-5, 0.6, 600)


def film_ratio(vsl, vsg, rho_l, rho_g, mu_l, mu_g, d_ft, sin_t, liq_cut=None):
    """Vectorised Barnea film ratio (complementary.pe5_stability_ratio): > 1 an upward film exists; NaN above
    PE5_LIQ_CUT liquid fraction or without gas, inf horizontal / downward."""
    vsl, vsg, rho_l, rho_g, mu_l, mu_g, sin_t, d_ft = np.broadcast_arrays(*(np.asarray(x, float) for x in
                                                                          (vsl, vsg, rho_l, rho_g, mu_l, mu_g, sin_t,
                                                                           d_ft)))
    vm = vsl + vsg
    ok = (vm > 0) & (vsg > 0) & (vsl > 0) & (rho_l > rho_g) & (rho_g > 0) & (mu_l > 0) & (mu_g > 0)
    ok &= vsl / np.where(vm > 0, vm, 1) <= (CX.PE5_LIQ_CUT if liq_cut is None else liq_cut)
    out = np.full(vm.shape, np.nan)
    if not ok.any():
        return out

    dk = d_ft[ok]

    def dpdz(rho, v, mu):
        re = rho * v * dk / (mu * 6.7197e-4)
        f = np.where(re < 2100, 16.0 / np.maximum(re, 1e-9), 0.046 * np.maximum(re, 1e-9) ** -0.2)
        return 2 * f * rho * v * v / (G_FT * dk)
    s = np.clip(sin_t[ok], 0.0, 1.0)
    dl, dg = dpdz(rho_l[ok], vsl[ok], mu_l[ok]), dpdz(rho_g[ok], vsg[ok], mu_g[ok])
    x2 = dl / dg
    y = (rho_l[ok] - rho_g[ok]) * s * (1.0 + CX.PE5_DRAIN * (1.0 - s * s)) / dg
    h = _FILM_H[None, :]
    rhs = (1 + CX.PE5_FILM_K * h) / ((1 - h) ** 2.5 * h) - x2[:, None] / h ** 3
    r = rhs.max(axis=1) / np.maximum(y, 1e-30)
    out[ok] = np.where(s <= 1e-6, np.inf, r)
    return out


def _parts(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, p = (np.asarray(x, float) for x in
                                                     (vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, p))
    sin_arr = np.broadcast_to(np.asarray(sin_t, float), vsl.shape)
    vm, lam = _no_slip_lambda(vsl, vsg)
    s2 = 1.0 - np.minimum(sin_arr ** 2, 1.0)
    rg = np.maximum(rho_g, 1e-3)
    drho = np.maximum(rho_l - rg, 1e-3)
    # slug: drift flux
    c0 = PE5X_C0[0] - PE5X_C0[1] * s2
    a, b, c, k = PE5X_VD
    vd = np.exp(a) * (drho / rg) ** b * np.maximum(vm, 1e-6) ** c * (1 + k * s2) * np.clip(sin_arr, 0.0, 1.0) ** 0.0
    hl_s = np.clip(1 - vsg / np.maximum(c0 * vm + vd, 1e-9), lam, 1.0)
    # regime
    ratio = film_ratio(vsl, vsg, rho_l, rg, mu_l, mu_g, d, sin_arr, PE5X_LIQ_MAX)
    fl = vsl * np.sqrt(rho_l / (drho * G_FT * d))
    annular = np.isfinite(ratio) & (ratio >= PE5X_FILM_RATIO * (1 + PE5X_FILM_RATIO_DEV * s2)) & (fl >= PE5X_FL_MIN)
    fr = vsg * np.sqrt(rg / (drho * G_FT * d))
    wa, wb, wc = PE5X_ANN_W
    w = np.where(fr > wc, np.minimum(1.0, np.exp(wa) * np.maximum(fr - wc, 1e-9) ** wb), 1.0)
    lam_a = np.minimum((PE5X_ANN_FLOOR[0] + PE5X_ANN_FLOOR[1] * lam) * lam, np.maximum(hl_s, lam))
    hl_a = lam_a + w * (hl_s - lam_a)
    hl = np.where(annular, hl_a, hl_s)
    # friction: Moody on the no-slip mixture x the regime's factor
    rho_ns = lam * rho_l + (1 - lam) * rg
    mu_ns = (lam * mu_l + (1 - lam) * mu_g) * CP_TO_LBFTS
    f_m = friction_factor(rho_ns * vm * d / np.maximum(mu_ns, 1e-12), rough / d)
    base = f_m * rho_ns * vm * vm / (2 * G_FT * d) / 144.0
    fa, fb, fc, fd, fe = PE5X_SLUG_FRICTION
    mult_s = np.exp(fa) * np.maximum(vm, 1e-6) ** fb * np.maximum(lam, 1e-6) ** fc * (rg / np.maximum(rho_l, 1e-3)) ** fd \
        * np.exp(fe * s2)
    ga, gb, gc, gd, ge = PE5X_ANN_FRICTION
    mult_a = np.exp(ga) * np.maximum(fr, 0.05) ** gb * np.maximum(lam, 1e-6) ** gc \
        * np.maximum(hl / np.maximum(lam_a, 1e-9), 1.0) ** gd * (rg / np.maximum(rho_l, 1e-3)) ** ge
    f = base * np.where(annular, mult_a, mult_s)
    e = (rho_l * hl + rg * (1 - hl)) * sin_arr / 144.0
    return e, f, hl, annular, vm, lam


def pe5_extended_style(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p):
    """PE5 Extended-style (see the module docstring). Returns (elevation, friction psi/ft, holdup)."""
    e, f, hl, annular, vm, lam = _parts(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p)
    # bubble flow and single phase as PE5-style
    e_p, f_p, hl_p = pe5_style(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p)
    lb = np.maximum(1.071 - 0.2218 * vm * vm / d, 0.13)
    other = ((np.asarray(vsg, float) / np.maximum(vm, 1e-12)) < lb) | (lam > 0.99999)
    return np.where(other, e_p, e), np.where(other, f_p, f), np.where(other, hl_p, hl)


def regime(vsl, vsg, rho_l, rho_g, mu_l, mu_g, d, sin_t):
    """'Annular' or 'Slug' per element, as pe5_extended_style decides (for reports)."""
    vsl = np.asarray(vsl, float)
    _, _, _, annular, _, _ = _parts(vsl, vsg, rho_l, rho_g, mu_l, mu_g, np.full(vsl.shape, 20.0), d, 0.0, sin_t,
                                    np.full(vsl.shape, 1000.0))
    return np.where(annular, "Annular", "Slug")
