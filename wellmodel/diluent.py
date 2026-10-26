"""Diluent injection (PROSPER Artificial Lift = Diluent Injection; Tutorial 30, "Annular Injection with Gaslift").

A diluent (water of its own salinity) is injected down the tubing-casing annulus and enters the tubing at the
diluent injection depth; with gas lift, lift gas enters at the gas lift valve depth (above it). Three zones,
top down:

    top node .. valve depth        produced fluid + diluent + lift gas
    valve depth .. diluent depth   produced fluid + diluent
    diluent depth .. bottom node   produced fluid

The diluent joins the liquid as its own water stream (DiluentFluid: water properties at its salinity; liquid
density and viscosity by volume with the produced liquid, water-gas tension for its share) - the same result as
blending the salinities by rate (the jet pump power fluid, validated on PROSPER Tutorials 27 / 32), but per case.
The lift gas adds GOR on the produced oil above the valve (gaslift.lifted_case). The temperature balance carries
the mass of each stream from where it enters (entering at the flowing temperature).

Rates (Tutorial 30 sensitivity variables):
  diluent rate    STB/d, or "Salt Diluent Rate Percent" - PROSPER: a percentage of the produced liquid rate (the user's
                  Tutorial 30 runs: diluent = pct x liquid in all 50 cases); other bases: produced water, total water;
  lift gas        GLR injected (scf/STB) x the produced liquid rate (PROSPER: 50 scf/STB x 10,116 STB/d = 0.506
                  MMscf/d in its gradient); or x produced + diluent liquid.
The water salinity PROSPER reports is the rate-weighted blend (136,757 ppm for 20 % diluent at 20,000 ppm in
180,000 ppm water) - the same water as the separate stream here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import blackoil as bo
from . import gaslift as GL
from . import vlp as V
from .correlations import CORRELATIONS
from .equipment import HeatCapacities

RB = 5.614583
DILUENT_PCT_BASIS = "produced liquid"    # PROSPER; or "produced water", "total water" (produced + diluent)
GLR_BASIS = "produced liquid"            # PROSPER; or "total liquid" (produced + diluent)


def _water_inp(salinity):
    return bo.PVTInputs(rsb=1.0, api=35.0, gas_sg=0.7, salinity_ppm=float(salinity))


class DiluentFluid:
    """The produced fluid with a diluent water stream (case key q_dil, STB/d); without it (or 0) the base fluid."""

    def __init__(self, base, salinity):
        self.base, self.salinity = base, float(salinity)
        self.kind = base.kind
        self._winp = _water_inp(salinity)
        for a in ("inp", "m"):
            if hasattr(base, a):
                setattr(self, a, getattr(base, a))
        for a in ("pb_at", "oil_water_viscosity", "_props"):
            if hasattr(base, a):
                setattr(self, a, getattr(base, a))

    def state(self, p, t, case):
        st = self.base.state(p, t, case)
        qd = case.get("q_dil")
        if qd is None or not np.any(np.asarray(qd) > 0):
            return st
        bw, rho_d, mu_d = bo.water_props(self._winp, np.maximum(p, 14.7), t)
        q_d = np.asarray(qd, float) * bw * RB / 86400.0
        ql = st["ql"] + q_d
        f = np.where(ql > 1e-15, q_d / np.maximum(ql, 1e-15), 0.0)
        out = dict(st)
        out["ql"] = ql
        out["rho_l"] = (1 - f) * st["rho_l"] + f * rho_d
        out["mu_l"] = (1 - f) * st["mu_l"] + f * mu_d
        out["sig"] = (1 - f) * st["sig"] + f * V.sigma_water_oilwell(rho_d, st["rho_g"])
        return out

    def mass_cp(self, case, hc: HeatCapacities):
        w, cp = self.base.mass_cp(case, hc)
        qd = case.get("q_dil")
        if qd is None:
            return w, cp
        sg = float(np.ravel(bo.water_props(self._winp, 14.696, 60.0)[1])[0]) / 62.428
        wd = np.asarray(qd, float) * RB * 62.428 * sg / 24.0
        wt = w + wd
        return wt, (w * cp + wd * hc.water) / np.maximum(wt, 1e-12)


@dataclass
class DiluentSetup:
    dil_md: float = 9000.0             # diluent injection depth, ft MD
    rate: float = 5000.0               # diluent rate, STB/d (used when pct is None)
    pct: float | None = None           # diluent rate, % (DILUENT_PCT_BASIS: PROSPER's - of the produced liquid)
    salinity: float = 20000.0          # ppm
    gas_lift: bool = True
    gas: GL.LiftGas = field(default_factory=GL.LiftGas)
    glr_inj: float = 50.0              # scf/STB (GLR_BASIS)
    valve_md: float = 7500.0           # gas lift valve depth, ft MD
    pct_basis: str = DILUENT_PCT_BASIS
    glr_basis: str = GLR_BASIS

    def extra_md(self):
        return [self.dil_md] + ([self.valve_md] if self.gas_lift else [])

    def rates(self, q, wc, params=None):
        """(diluent STB/d, lift gas MMscf/d) for produced liquid rates q (STB/d) and water cut wc (fraction).
        params may override: dil_pct (%), dil_rate (STB/d), glr_dil (scf/STB)."""
        params = params or {}
        q = np.atleast_1d(np.asarray(q, float))
        qw = q * np.clip(wc, 0, 1)
        pct = params.get("dil_pct", self.pct if "dil_rate" not in params else None)
        if pct is not None:
            x = float(pct) / 100.0
            qd = (q * x if self.pct_basis == "produced liquid" else
                  qw * x / max(1 - x, 1e-6) if self.pct_basis == "total water" else qw * x)
        else:
            qd = np.full(q.shape, float(params.get("dil_rate", self.rate)))
        glr = float(params.get("glr_dil", self.glr_inj)) if self.gas_lift else 0.0
        liq = q + qd if self.glr_basis == "total liquid" else q
        return qd, glr * liq / 1e6


def _temperature(path, fluid, case, q_dil, qgi, dil_md, valve_md, u, hc, gas_sg):
    """Flowing temperature: the produced fluid from the bottom node; the diluent's mass from its depth up, the lift
    gas's from the valve up (each entering at the flowing temperature)."""
    w0, cp0 = fluid.mass_cp(case, hc)
    w0 = np.broadcast_to(np.asarray(w0, float), np.shape(q_dil))
    cp0 = np.broadcast_to(np.asarray(cp0, float), np.shape(q_dil))
    sg_d = float(np.ravel(bo.water_props(fluid._winp, 14.696, 60.0)[1])[0]) / 62.428 if hasattr(fluid, "_winp") else 1.0
    wd = q_dil * RB * 62.428 * sg_d / 24.0
    wg = qgi * 1e6 * GL.RHO_AIR_SC * gas_sg / 24.0
    n = len(path.md)
    T = np.empty((w0.size, n))
    T[:, -1] = path.t_form[-1]
    for i in range(n - 2, -1, -1):
        L = path.seg_len[i]
        mid = 0.5 * (path.md[i] + path.md[i + 1])
        ww, cc = w0, w0 * cp0
        if mid < dil_md:
            ww, cc = ww + wd, cc + wd * hc.water
        if valve_md is not None and mid < valve_md:
            ww, cc = ww + wg, cc + wg * hc.gas
        cc = cc / np.maximum(ww, 1e-12)
        a = u * np.pi * path.d_ht[i] / np.maximum(ww * cc, 1e-9)
        tfb, tft = path.t_form[i + 1], path.t_form[i]
        g = (tft - tfb) / L
        t0 = T[:, i + 1]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            tn = tfb + g * L - g / a + (t0 - tfb + g / a) * np.exp(-a * L)
        T[:, i] = np.where(a * L < 1e-9, t0, tn)
    return T


def traverse(path, fluid, top_p, case, setup: DiluentSetup, correlation="pe2_style", p1=1.0, p2=1.0, u=8.0,
             hc: HeatCapacities | None = None, q_dil=None, qgi=None):
    """Top-down traverse with diluent (and gas lift). case: q (produced liquid STB/d), wc, gor (oil fluid).
    q_dil / qgi (per case) default to setup.rates(q, wc). The path must have nodes at the injection depths.
    Returns (P, T) cases x nodes."""
    hc = hc or HeatCapacities()
    c = dict(case)
    c["_top"] = top_p
    c = V._broadcast_case(c)
    top = c.pop("_top")
    nc = top.size
    if q_dil is None or qgi is None:
        qd_, qg_ = setup.rates(c["q"], c["wc"])
        q_dil = qd_ if q_dil is None else q_dil
        qgi = qg_ if qgi is None else qgi
    q_dil = np.broadcast_to(np.asarray(q_dil, float), (nc,)).copy()
    qgi = np.broadcast_to(np.asarray(qgi, float), (nc,)).copy() if setup.gas_lift else np.zeros(nc)
    fl = fluid if isinstance(fluid, DiluentFluid) else DiluentFluid(fluid, setup.salinity)
    base_case = {**c, "q_dil": np.zeros(nc)}
    dil_case = {**c, "q_dil": q_dil}
    lift_case = {**GL.lifted_case(c, qgi), "q_dil": q_dil} if setup.gas_lift else dil_case
    valve = setup.valve_md if setup.gas_lift else None
    T = _temperature(path, fl, base_case, q_dil, qgi, setup.dil_md, valve, u, hc, setup.gas.sg)
    corr = CORRELATIONS[correlation]
    n = len(path.md)
    P = np.empty((nc, n))
    p = np.maximum(top, 14.7)
    P[:, 0] = p
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(n - 1):
            mid = 0.5 * (path.md[i] + path.md[i + 1])
            cs = lift_case if (valve is not None and mid < valve) else (dil_case if mid < setup.dil_md else base_case)
            L = path.seg_len[i]
            g0, _ = V._gradient(path, fl, i, p, T[:, i], cs, corr, p1, p2)
            g1, _ = V._gradient(path, fl, i, np.maximum(p + g0 * L, 14.7), T[:, i + 1], cs, corr, p1, p2)
            p = np.maximum(p + 0.5 * (g0 + g1) * L, 14.7)
            if (i + 1) in path.restrictions:
                p = p + V._restriction_dp(fl.state(p, T[:, i + 1], cs), path.restrictions[i + 1])
            P[:, i + 1] = p
    return P, T
