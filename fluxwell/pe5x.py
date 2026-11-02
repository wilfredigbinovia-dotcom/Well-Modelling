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
            both vertical boundaries in the data sit at 2.73-2.77 (500 psig / 3.5 MMscf/d, 3 % liquid; 1000 psig /
            5 MMscf/d, 6 % liquid), where a gas Froude or Kutateladze number does not separate them; and the liquid
            Froude number vsl (rho_l / (drho g d))^0.5 is at least PE5X_FL_MIN (near-dry flow stays slug: 100 psig at
            1.5 MMscf/d, 0.065, against 0.095 for the least annular node). Off vertical the threshold is x (1 + 1.0
            sin^2 deviation): the 60 deg runs stay slug up to a ratio of 4.5 at 5 MMscf/d, and the second set (T43 at
            30 and 60 deg, 8-20 MMscf/d, ratios from 6.9; T38 at 60 deg) is annular at every node, as PROSPER.
  slug      drift flux vsg / (1 - HL) = C0 vm + vd, C0 = 1.0233 - 0.08 sin^2 dev, vd = 0.1666 (drho / rho_g)^0.398
            vm^-0.078 (1 + 3.7 sin^2 dev) ft/s (vertical nodes within 3 %, 60 deg within 5 %, 10th-90th percentile);
            friction = Moody (no-slip mixture) x exp(1.9549) vm^-0.8403 lambda^0.1575 (rho_g / rho_l)^-0.2069
            exp(-0.2077 sin^2 dev) - fitted on the condensate and the T13 oil well together (19 % rms on the
            condensate nodes, no bias on the oil: PROSPER's slug friction there is a steady 0.64 x no-slip Moody at
            35-82 % liquid; the condensate-only fit gave 3 x that).
  annular   holdup between the slug holdup at the transition and no slip: HL = lambda + w (HL_slug - lambda),
            w = min(1, exp(-3.714) (Fr_g - 0.94)^-1.071) with the gas Froude number vsg (rho_g / (drho g d))^0.5. In
            PROSPER the annular holdup also carries the flow's history (it starts at the slug value where the flow turns
            annular and develops upward - entrainment and deposition along the pipe), which a local closure cannot:
            about +-40 % on the film part, small in absolute terms above Fr_g ~ 1.1. Friction = Moody (no-slip) x
            exp(-0.4737) Fr_g^-0.3635 lambda^-0.0774 max(HL / lambda, 1)^-1.0481 (rho_g / rho_l)^-0.127 (12 % rms on
            the vertical, deviated and oil annular nodes; Fr_g 1-5.7, no separate deviation term needed).
  bubble / single phase as PE5-style.

Data: the user's 22 T43 PE5 Extended traverses (first set) and a second set - T43 at 30 / 60 deg below 2000 ft (8, 12,
20 MMscf/d at 500 psig), T38's black-oil condensate at 60 deg (25 MMscf/d, 100 psig) and the T13 Logie oil well
(14,482 STB/d; VLP tables at water cut 0 / 50 % and GOR 2000, 2000-16,000 STB/d). Bottom-hole pressures: first set mean
4.4 %, T43 deviated -3 to -6 % (PROSPER's EoS fluid; T38 on the same black-oil fluid -5.8 %), T13 within 9.4 % (mean
3.6 %). PROSPER's reported holdup is not exactly what its gravity term uses (the static gradient implies 1-6 % more
liquid); the closures follow the static and friction gradients.

A screening approximation: check it against PROSPER before relying on it, oil wells rest on one gradient and three VLP tables."""
from __future__ import annotations

import numpy as np

from . import complementary as CX
from .correlations import _no_slip_lambda, friction_factor, pe5_style, CP_TO_LBFTS

G_FT = 32.174
PE5X_FILM_RATIO = 2.74
PE5X_FILM_RATIO_DEV = 1.0          # threshold x (1 + k sin^2 deviation)
PE5X_FL_MIN = 0.08
PE5X_C0 = (1.0233, 0.08)           # C0 = a - b sin^2 dev
PE5X_VD = (-1.792, 0.398, -0.078, 3.7)     # vd = exp(a) (drho/rho_g)^b vm^c (1 + k sin^2 dev), ft/s
PE5X_SLUG_FRICTION = (1.9549, -0.8403, 0.1575, -0.2069, -0.2077)  # x Moody_ns: exp(a) vm^b lam^c (rg/rl)^d exp(e s2)
PE5X_ANN_W = (-3.714, -1.071, 0.94)        # w = exp(a) (Fr - c)^b
PE5X_ANN_FRICTION = (-0.4737, -0.3635, -0.0774, -1.0481, -0.127)  # x Moody_ns: exp(a) Fr^b lam^c max(HL/lam,1)^d (rg/rl)^e
_FILM_H = np.geomspace(1e-5, 0.6, 600)


def film_ratio(vsl, vsg, rho_l, rho_g, mu_l, mu_g, d_ft, sin_t):
    """Vectorised Barnea film ratio (complementary.pe5_stability_ratio): > 1 an upward film exists; NaN above
    PE5_LIQ_CUT liquid fraction or without gas, inf horizontal / downward."""
    vsl, vsg, rho_l, rho_g, mu_l, mu_g, sin_t, d_ft = np.broadcast_arrays(*(np.asarray(x, float) for x in
                                                                          (vsl, vsg, rho_l, rho_g, mu_l, mu_g, sin_t,
                                                                           d_ft)))
    vm = vsl + vsg
    ok = (vm > 0) & (vsg > 0) & (vsl > 0) & (rho_l > rho_g) & (rho_g > 0) & (mu_l > 0) & (mu_g > 0)
    ok &= vsl / np.where(vm > 0, vm, 1) <= CX.PE5_LIQ_CUT
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
    ratio = film_ratio(vsl, vsg, rho_l, rg, mu_l, mu_g, d, sin_arr)
    fl = vsl * np.sqrt(rho_l / (drho * G_FT * d))
    annular = np.isfinite(ratio) & (ratio >= PE5X_FILM_RATIO * (1 + PE5X_FILM_RATIO_DEV * s2)) & (fl >= PE5X_FL_MIN)
    fr = vsg * np.sqrt(rg / (drho * G_FT * d))
    wa, wb, wc = PE5X_ANN_W
    w = np.where(fr > wc, np.minimum(1.0, np.exp(wa) * np.maximum(fr - wc, 1e-9) ** wb), 1.0)
    hl_a = lam + w * (hl_s - lam)
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
        * np.maximum(hl / np.maximum(lam, 1e-9), 1.0) ** gd * (rg / np.maximum(rho_l, 1e-3)) ** ge
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
