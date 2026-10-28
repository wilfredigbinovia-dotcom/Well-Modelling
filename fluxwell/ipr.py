"""Inflow performance: composite straight-line PI / Vogel (liquid basis).

Above the bubble point inflow is linear, q = J (Pr - Pwf). Below Pb the
Vogel curve is attached with matching slope, so J is always the PI at
pressures above Pb. All functions broadcast over numpy arrays.
"""
from __future__ import annotations

import numpy as np


def _prep(pr, pi, pb):
    pr = np.asarray(pr, float)
    pi = np.asarray(pi, float)
    pb_eff = np.minimum(pb, pr)
    qb = pi * np.maximum(pr - pb, 0.0)
    return pr, pi, pb_eff, qb


def aof(pr, pi, pb):
    """Absolute open flow (liquid rate at Pwf = 0)."""
    pr, pi, pb_eff, qb = _prep(pr, pi, pb)
    return qb + pi * pb_eff / 1.8


def rate(pwf, pr, pi, pb):
    """Liquid rate (STB/d) delivered at flowing bottomhole pressure `pwf`."""
    pr, pi, pb_eff, qb = _prep(pr, pi, pb)
    pwf = np.clip(np.asarray(pwf, float), 0.0, None)
    x = pwf / np.maximum(pb_eff, 1e-9)
    q_lin = pi * (pr - pwf)
    q_vog = qb + pi * pb_eff / 1.8 * (1 - 0.2 * x - 0.8 * x * x)
    q = np.where(pwf >= pb_eff, q_lin, q_vog)
    return np.maximum(q, 0.0)


def pwf(q, pr, pi, pb):
    """Inverse IPR: Pwf (psia) needed to deliver liquid rate `q`.
    Returns 0 for rates at or beyond AOF."""
    pr, pi, pb_eff, qb = _prep(pr, pi, pb)
    q = np.asarray(q, float)
    p_lin = pr - q / np.maximum(pi, 1e-12)
    c = 1 - (q - qb) * 1.8 / np.maximum(pi * pb_eff, 1e-12)
    x = (-0.2 + np.sqrt(np.maximum(0.04 + 3.2 * c, 0.0))) / 1.6
    p_vog = np.where(c > 0, x * pb_eff, 0.0)
    return np.maximum(np.where(q <= qb, p_lin, p_vog), 0.0)


def darcy_pi(k, h, mu, bo, re, rw, skin):
    """Pseudo-steady-state PI (STB/d/psi) for a well centred in a circular
    drainage area: J = 0.00708 k h / (mu Bo (ln(re/rw) - 0.75 + S))."""
    return 0.00708 * k * h / (mu * bo * (np.log(re / rw) - 0.75 + skin))
