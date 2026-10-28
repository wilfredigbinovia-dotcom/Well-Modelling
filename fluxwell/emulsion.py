"""Emulsion viscosity (PROSPER Options: Emulsions = "Emulsion + Pump Viscosity Correction"; Tutorial 46).

PROSPER's emulsion model scales the oil viscosity by a polynomial in water cut (fraction):

    oil-continuous   WC <= WC_min      mu = mu_o (P1 + P2 WC + P3 WC^2 + P4 WC^3 + P5 WC^4)
    plateau          WC_min..WC_max    mu = the value at WC_min ("minimum water cut for maximum viscosity")
    water-continuous WC > WC_max       log-linear from the plateau to the water viscosity at WC = 1 (PROSPER's
                                       gradients at WC 85 / 90 / 95 %: exact with its water viscosity)

The parameters are matched to measured emulsion viscosities (at one pressure and temperature) by least squares on
the oil-continuous points with P1 fixed at 1, relative to the oil viscosity at the measurement conditions. Tutorial
46 (11 API oil, mu_o 100 cp at 2000 psig and 200 degF; points 5-55 % WC) gives P2 0.69799, P3 3.77291 and a standard
deviation of 3.21611 cp (RMS over the points) - reproduced here to 0.2 % (0.69918, 3.77190, 3.21789 with mu_o 100;
PROSPER's oil viscosity at 2000 psig is 100.015 cp).

The emulsion viscosity replaces the volume-weighted oil-water liquid viscosity in the tubing (and pipe) when the
emulsion occurs there, at the stock-tank water cut (PROSPER's gradient viscosities at WC 25 / 50 / 75 % within 0.05 %);
at WC 0 it is the oil viscosity, so the gradients are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OCCURRENCE = {
    "none": "No emulsion corrections",
    "tubing_pipe": "Tubing and pipe",
    "tubing": "Tubing only",
    "pipe": "Pipe only",
}


@dataclass
class EmulsionModel:
    params: tuple = (1.0, 0.0, 0.0, 0.0, 0.0)      # P1..P5
    wc_min: float = 0.6                            # minimum water cut for maximum viscosity (fraction)
    wc_max: float = 0.8                            # maximum water cut for maximum viscosity (fraction)
    occurrence: str = "tubing_pipe"

    @property
    def in_tubing(self) -> bool:
        return self.occurrence in ("tubing_pipe", "tubing")

    @property
    def in_pipe(self) -> bool:
        return self.occurrence in ("tubing_pipe", "pipe")

    def poly(self, wc):
        x = np.asarray(wc, float)
        p = list(self.params) + [0.0] * (5 - len(self.params))
        return p[0] + x * (p[1] + x * (p[2] + x * (p[3] + x * p[4])))

    def ratio(self, wc):
        """Emulsion / oil viscosity up to the end of the plateau."""
        x = np.clip(np.asarray(wc, float), 0.0, 1.0)
        return np.maximum(self.poly(np.minimum(x, self.wc_min)), 1e-6)

    def viscosity(self, wc, mu_o, mu_w):
        """Emulsion viscosity (cp) at water cut wc (fraction) for oil and water viscosities mu_o, mu_w."""
        x = np.clip(np.asarray(wc, float), 0.0, 1.0)
        mu_o = np.asarray(mu_o, float)
        mu_w = np.maximum(np.asarray(mu_w, float), 1e-6)
        top = np.maximum(mu_o * self.ratio(x), 1e-6)
        span = max(1.0 - self.wc_max, 1e-9)
        f = np.clip((x - self.wc_max) / span, 0.0, 1.0)
        water_side = np.exp((1 - f) * np.log(top) + f * np.log(mu_w))
        return np.where(x > self.wc_max, water_side, top)


@dataclass
class EmulsionMatch:
    params: tuple
    std: float          # RMS deviation of the oil-continuous points (cp), as PROSPER's "Standard Deviation"
    n: int


def match(wc_pct, mu, mu_o, wc_min=0.6, n_params=3, fix_p1=True) -> EmulsionMatch:
    """Fit P2..Pn (P1 = 1) to measured emulsion viscosities mu (cp) at water cuts wc_pct (%) and oil viscosity mu_o
    (cp) at the measurement conditions. Only the points at or below wc_min (oil-continuous) are used."""
    wc = np.asarray(wc_pct, float) / 100.0
    mu = np.asarray(mu, float)
    ok = np.isfinite(wc) & np.isfinite(mu) & (wc <= wc_min + 1e-9) & (wc > 0)
    x, y = wc[ok], mu[ok] / float(mu_o)
    k = max(1, min(int(n_params), 5)) - (1 if fix_p1 else 0)
    p0 = 1.0 if fix_p1 else None
    if x.size == 0 or k <= 0:
        return EmulsionMatch((1.0, 0.0, 0.0, 0.0, 0.0), float("nan"), int(x.size))
    pw = np.arange(1, k + 1) if fix_p1 else np.arange(0, k + 1)
    A = x[:, None] ** pw[None, :]
    coef, *_ = np.linalg.lstsq(A, y - (p0 or 0.0), rcond=None)
    full = ([p0] if fix_p1 else []) + list(coef)
    full = tuple(float(v) for v in (full + [0.0] * 5)[:5])
    fit = EmulsionModel(full, wc_min).poly(x) * mu_o
    std = float(np.sqrt(np.mean((fit - mu[ok]) ** 2)))
    return EmulsionMatch(full, std, int(x.size))
