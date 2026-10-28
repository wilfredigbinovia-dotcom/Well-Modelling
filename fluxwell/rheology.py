"""Non-Newtonian (shear-thinning / yield-stress) oil viscosity - PROSPER's Viscosity Model = Non-Newtonian Fluid
(Tutorial 44).

Herschel-Bulkley rheology tau = tau0 + K gamma^n, with tau0 (psi), K (Pa.s^n) and n tabulated against temperature
and pressure (PROSPER's Rheological Parameters table) and interpolated linearly. PROSPER's "equivalent apparent
viscosity" is the Metzner-Reed generalised viscosity of the pipe flow at the wall shear rate 8 v / D:

    mu_app = K ((3n + 1) / (4n))^n (8 v / D)^(n - 1)  [+ tau0 / (8 v / D) with a yield stress]

capped at the maximum viscosity entered; it replaces the oil viscosity in the gradient (Reynolds number, friction).
Found on the user's T44 file: 400.56 cp in the 3.2 in tubing (8.93 ft/s, 116.1 F) and 425.0 cp in the 6.4 in casing
(2.23 ft/s, 120 F) reproduced to 0.1 cp. The yield-stress term is the Bingham-plastic analogue (not checked against
PROSPER: the tutorial's tau0 is zero).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PA_PER_PSI = 6894.757
COLUMNS = ["Temperature (°F)", "Pressure (psig)", "Yield stress TAUo (psi)", "Consistency index K", "Shear thinning index n"]


class Rheology:
    def __init__(self, rows, mu_max_cp=10000.0):
        """rows: iterable of (T F, P psig, tau0 psi, K Pa.s^n, n)."""
        a = np.array([[float(x) for x in r] for r in rows], float)
        if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] != 5:
            raise ValueError("Enter at least one row of temperature, pressure, yield stress, K and n.")
        if (a[:, 3] <= 0).any() or (a[:, 4] <= 0).any():
            raise ValueError("K and n must be positive.")
        self.rows = a[np.lexsort((a[:, 1], a[:, 0]))]
        self.temps = np.unique(self.rows[:, 0])
        self.mu_max = float(mu_max_cp)

    @classmethod
    def from_table(cls, df: pd.DataFrame, mu_max_cp=10000.0):
        d = df[COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
        return cls(d.to_numpy(float), mu_max_cp)

    def _at_t(self, t_row, p_psig):
        g = self.rows[self.rows[:, 0] == t_row]
        return [np.interp(p_psig, g[:, 1], g[:, k]) for k in (2, 3, 4)]

    def params(self, p_psia, t_f):
        """(tau0 psi, K, n) at pressure p (psia) and temperature t (F): linear in pressure within each temperature,
        then linear in temperature (held at the table ends)."""
        p = np.atleast_1d(np.asarray(p_psia, float)) - 14.696
        t = np.broadcast_to(np.atleast_1d(np.asarray(t_f, float)), p.shape)
        if len(self.temps) == 1:
            out = self._at_t(self.temps[0], p)
            return tuple(np.asarray(x, float) for x in out)
        vals = np.array([self._at_t(tr, p) for tr in self.temps])        # (nT, 3, np)
        res = []
        for k in range(3):
            res.append(np.array([np.interp(t[i], self.temps, vals[:, k, i]) for i in range(p.size)]))
        return tuple(res)

    def apparent_viscosity(self, p_psia, t_f, v_fts, d_ft):
        """Equivalent apparent (Metzner-Reed) viscosity, cp."""
        tau0, k, n = self.params(p_psia, t_f)
        v = np.maximum(np.abs(np.broadcast_to(np.asarray(v_fts, float), tau0.shape)), 1e-6)
        gamma = 8 * v / float(d_ft)
        mu = k * ((3 * n + 1) / (4 * n)) ** n * gamma ** (n - 1) + tau0 * PA_PER_PSI / gamma     # Pa.s
        return np.minimum(mu * 1000.0, self.mu_max)
