"""WellModel: PVT + wellbore + correlation + IPR, with a vectorised pressure
traverse and nodal (IPR/VLP) solver.

All "case" inputs (rates, water cut, GOR, WHP, PI, multipliers, reservoir
pressure) broadcast against each other, so thousands of cases - Monte Carlo
samples, test points, rate grids - run through the tubing in one pass.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import ipr as _ipr
from .correlations import CORRELATIONS
from .pvt import BlackOilPVT
from .wellbore import Wellbore


@dataclass
class OperatingPoint:
    rate: np.ndarray          # liquid STB/d (0 when the well cannot flow)
    pwf: np.ndarray           # psia (nan when not flowing)
    flowing: np.ndarray       # bool: IPR and VLP intersect
    stable: np.ndarray        # bool: intersection on the right of the VLP minimum
    q_vlp_min: np.ndarray     # rate at minimum of the VLP curve (STB/d)
    aof: np.ndarray

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class WellModel:
    pvt: BlackOilPVT
    wellbore: Wellbore
    correlation: str = "beggs_brill"
    gravity_mult: float = 1.0
    friction_mult: float = 1.0
    pi: float = 1.0                 # liquid PI above Pb, STB/d/psi

    def copy(self, **changes) -> "WellModel":
        return replace(self, **changes)

    # ------------------------------------------------------------ gradient
    def _gradient(self, p, t, ql, wc, gor, d, rough, sin_t, corr, gm, fm):
        pvt = self.pvt
        wc = np.clip(wc, 0.0, 0.9999)
        qo = ql * (1 - wc)
        qw = ql * wc
        rs, bo, rho_o, mu_o = pvt.oil(p, t, gor)
        _, bg, rho_g, mu_g = pvt.gas(p, t)
        bw, rho_w, mu_w = pvt.water(p, t)
        so, sw = pvt.surface_tension(p, t)
        area = np.pi * d * d / 4.0
        vso = qo * bo * 5.615 / 86400.0 / area
        vsw = qw * bw * 5.615 / 86400.0 / area
        vsg = qo * np.maximum(gor - rs, 0.0) * bg / 86400.0 / area
        vsl = vso + vsw
        fo = np.where(vsl > 1e-12, vso / np.maximum(vsl, 1e-12), 1 - wc)
        rho_l = fo * rho_o + (1 - fo) * rho_w
        mu_l = fo * mu_o + (1 - fo) * mu_w
        sig_l = fo * so + (1 - fo) * sw
        elev, fric, _ = corr(vsl, vsg, rho_l, rho_g, mu_l, mu_g, sig_l, d, rough, sin_t, p)
        g = gm * elev + fm * fric
        return np.nan_to_num(g, nan=0.433, posinf=1.0, neginf=0.0)

    # ------------------------------------------------------------ traverse
    def traverse(self, rate, wc, gor, whp, gravity_mult=None, friction_mult=None,
                 correlation=None):
        """Top-down pressure traverse. Returns array (N, n_nodes) of pressure
        at every grid node (wellbore.md) for each of the N broadcast cases."""
        gm = self.gravity_mult if gravity_mult is None else gravity_mult
        fm = self.friction_mult if friction_mult is None else friction_mult
        corr = CORRELATIONS[correlation or self.correlation]
        ql, wc, gor, whp, gm, fm = [np.atleast_1d(np.asarray(a, float)) for a in
                                    np.broadcast_arrays(rate, wc, gor, whp, gm, fm)]
        wb = self.wellbore
        rough = wb.roughness / 12.0
        out = np.empty((ql.size, wb.md.size))
        p = np.maximum(whp.copy(), 14.7)
        out[:, 0] = p
        for i in range(wb.dmd.size):
            dz, d, s = wb.dmd[i], wb.seg_d_ft[i], wb.sin_theta[i]
            g0 = self._gradient(p, wb.temp[i], ql, wc, gor, d, rough, s, corr, gm, fm)
            pp = np.maximum(p + g0 * dz, 14.7)
            g1 = self._gradient(pp, wb.temp[i + 1], ql, wc, gor, d, rough, s, corr, gm, fm)
            p = np.maximum(p + 0.5 * (g0 + g1) * dz, 14.7)
            out[:, i + 1] = p
        return out

    def pwf_vlp(self, rate, wc, gor, whp, **kw):
        """Flowing bottomhole pressure at mid-perfs from the tubing (VLP)."""
        return self.traverse(rate, wc, gor, whp, **kw)[:, -1]

    def gauge_pressure(self, rate, wc, gor, whp, **kw):
        if self.wellbore.gauge_index is None:
            raise ValueError("wellbore has no gauge depth")
        return self.traverse(rate, wc, gor, whp, **kw)[:, self.wellbore.gauge_index]

    # ----------------------------------------------------------------- IPR
    def ipr_rate(self, pwf, pr, pi=None):
        return _ipr.rate(pwf, pr, self.pi if pi is None else pi, self.pvt.pb)

    def ipr_pwf(self, q, pr, pi=None):
        return _ipr.pwf(q, pr, self.pi if pi is None else pi, self.pvt.pb)

    def aof(self, pr, pi=None):
        return _ipr.aof(pr, self.pi if pi is None else pi, self.pvt.pb)

    # --------------------------------------------------------------- nodal
    def operating_point(self, pr, whp, wc, gor, pi=None, gravity_mult=None,
                        friction_mult=None, correlation=None,
                        n_grid: int = 32, n_refine: int = 2, n_fine: int = 10) -> OperatingPoint:
        """Vectorised nodal solution for N broadcast cases.

        Coarse rate grid (fractions of AOF) -> last sign change of
        f(q) = Pwf_VLP(q) - Pwf_IPR(q) from negative to positive (the stable,
        right-hand intersection) -> `n_refine` passes of a finer grid inside
        the bracket -> linear interpolation of the root.
        """
        pi = self.pi if pi is None else pi
        gm = self.gravity_mult if gravity_mult is None else gravity_mult
        fm = self.friction_mult if friction_mult is None else friction_mult
        pr, whp, wc, gor, pi, gm, fm = [np.atleast_1d(np.asarray(a, float)) for a in
                                        np.broadcast_arrays(pr, whp, wc, gor, pi, gm, fm)]
        n = pr.size
        aof = self.aof(pr, pi)
        kw = dict(correlation=correlation)

        def f_eval(q):  # q shape (n, m)
            m = q.shape[1]
            rep = lambda a: np.repeat(a, m)
            pv = self.pwf_vlp(q.ravel(), rep(wc), rep(gor), rep(whp),
                              gravity_mult=rep(gm), friction_mult=rep(fm), **kw).reshape(n, m)
            pi_ = self.ipr_pwf(q, pr[:, None], pi[:, None])
            return pv - pi_, pv

        frac = np.concatenate([np.geomspace(0.002, 0.05, 6), np.linspace(0.08, 1.0, n_grid - 6)])
        q = aof[:, None] * frac[None, :]
        f, pv = f_eval(q)
        jmin = np.argmin(pv, axis=1)
        q_vlp_min = q[np.arange(n), jmin]
        cross = (f[:, :-1] < 0) & (f[:, 1:] >= 0)
        flowing = cross.any(axis=1)
        idx = (q.shape[1] - 2) - np.argmax(cross[:, ::-1], axis=1)
        rows = np.arange(n)
        lo = q[rows, idx]
        hi = q[rows, idx + 1]
        stable = flowing & (idx + 1 >= jmin)
        for _ in range(n_refine):
            qq = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, n_fine)[None, :]
            ff, _ = f_eval(qq)
            c = (ff[:, :-1] < 0) & (ff[:, 1:] >= 0)
            j = (n_fine - 2) - np.argmax(c[:, ::-1], axis=1)
            j = np.where(c.any(axis=1), j, 0)
            lo, hi = qq[rows, j], qq[rows, j + 1]
            flo, fhi = ff[rows, j], ff[rows, j + 1]
        w = np.clip(-flo / np.where(fhi - flo == 0, 1, fhi - flo), 0, 1)
        q_op = lo + w * (hi - lo)
        q_op = np.where(flowing, q_op, 0.0)
        pwf = np.where(flowing, self.ipr_pwf(q_op, pr, pi), np.nan)
        return OperatingPoint(q_op, pwf, flowing, stable, q_vlp_min, aof)

    # -------------------------------------------------------------- curves
    def ipr_curve(self, pr, n: int = 50, pi=None):
        pw = np.linspace(pr, 0, n)
        return self.ipr_rate(pw, pr, pi), pw

    def vlp_curve(self, whp, wc, gor, q_max, n: int = 30, **kw):
        q = np.linspace(q_max / n, q_max, n)
        return q, self.pwf_vlp(q, wc, gor, whp, **kw)

    def describe(self) -> dict:
        return {"correlation": self.correlation, "gravity_mult": self.gravity_mult,
                "friction_mult": self.friction_mult, "PI": self.pi,
                **{f"pvt_{k}": v for k, v in self.pvt.summary().items()}}
