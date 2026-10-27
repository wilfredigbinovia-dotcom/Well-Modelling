"""Foam lift - PROSPER's Artificial Lift = Foam Lift (Tutorial 48).

A surfactant injected at the bottom of a liquid-loading gas well (here down a coiled tubing to the perforations, the
gas produced up the annulus) turns the water into foam. PROSPER's default surfactant model (Petroleum Society paper
2007-118, Yang and Zhang) tabulates the water-gas surface tension and the foam density against the surfactant mass
percent; both are interpolated linearly in the concentration (see FoamModel._lookup for PROSPER's table ends).

The Foam-Homogeneous vertical lift correlation (correlations.foam_homogeneous): no-slip mixture of the gas and the
foam - the liquid taken at the foam density - for the gravity gradient, and no wall friction (PROSPER reports
0.05 psi of friction in the production string). Restrictions (the SSSV) still take the real fluid. Found on the
user's T48 file: the VLP tables of the ten surfactant cases (0.02-0.5 %, 0-10 MMscf/d at 500 psig) within 0.73 %,
the system rates within 0.8 %. The surfactant's surface tension is also the water-gas surface tension seen by the
other correlations and the Turner liquid-loading check (PROSPER's gradient columns).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIGMA_DEFAULT = ((0.02, 66.0), (0.05, 59.0), (0.1, 52.0), (0.15, 46.0), (0.2, 44.0), (0.5, 36.0))
DENSITY_DEFAULT = ((0.0, 68.0), (0.05, 53.0), (0.1, 45.0), (0.25, 33.0), (0.5, 30.5))     # the user's T48 file


@dataclass
class FoamModel:
    """Surface tension (dyne/cm) and foam density (lb/ft3) against surfactant concentration (mass percent)."""
    sigma: tuple = SIGMA_DEFAULT
    density: tuple = DENSITY_DEFAULT

    def __post_init__(self):
        for name in ("sigma", "density"):
            t = np.array(sorted((float(c), float(v)) for c, v in getattr(self, name)
                                if np.isfinite(c) and np.isfinite(v)), float).reshape(-1, 2)
            if len(t) < 1:
                raise ValueError(f"The foam {name} table needs at least one point.")
            setattr(self, name, tuple(map(tuple, t)))

    @staticmethod
    def _lookup(table, c):
        """PROSPER's table lookup: rows at zero concentration are ignored, linear interpolation between the others,
        linear extrapolation below the first (through the first two points), held beyond the last. Found on the user's
        T48 file: foam density (0, 68), (0.05, 53), (0.1, 45) ... gives 57.8 lb/ft3 at 0.02 % (not 62.0)."""
        t = np.array(table, float)
        t = t[t[:, 0] > 0] if (t[:, 0] > 0).any() else t
        c = np.asarray(c, float)
        out = np.interp(c, t[:, 0], t[:, 1])
        if len(t) >= 2:
            slope = (t[1, 1] - t[0, 1]) / max(t[1, 0] - t[0, 0], 1e-12)
            out = np.where(c < t[0, 0], t[0, 1] + slope * (c - t[0, 0]), out)
        return np.maximum(out, 0.0)

    def surface_tension(self, c):
        return self._lookup(self.sigma, c)

    def foam_density(self, c):
        return self._lookup(self.density, c)


class FoamFluid:
    """A gas-well fluid with surfactant: the state gains 'rho_foam' (lb/ft3) and the surfactant surface tension.
    Case key 'surf' (mass percent) overrides the concentration per case (a system sensitivity variable)."""

    def __init__(self, base, model: FoamModel, conc: float):
        self.base, self.model, self.conc = base, model, float(conc)
        self.kind = base.kind

    def __getattr__(self, k):
        return getattr(self.base, k)

    def state(self, p, t, case):
        s = dict(self.base.state(p, t, case))
        c = case.get("surf", self.conc) if isinstance(case, dict) else self.conc
        shape = np.shape(np.asarray(s["ql"], float))
        c = np.broadcast_to(np.asarray(c, float), shape) if np.ndim(c) else np.full(shape, float(c))
        s["rho_foam"] = self.model.foam_density(c)
        s["sig"] = self.model.surface_tension(c)
        return s
