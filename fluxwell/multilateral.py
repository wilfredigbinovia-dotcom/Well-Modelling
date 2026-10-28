"""Multilateral well IPR (PROSPER Inflow Type = MultiLateral Well, reservoir model "Petroleum Experts"; Tutorial 12).

The well is a network: a tie point (top of the IPR, bottom of the VLP), a tubing (T1) down to a junction, and
completions (branches) that leave the junction and produce from one reservoir layer through perforation intervals.

Reservoir model (dry and wet gas)
---------------------------------
Each survey section of a branch that is perforated is split into 20 segments (as PROSPER; its segment report has
20 rows per section). Every segment is a uniform-flux line source in a closed box (drainage area, thickness h, top
TVD), centred on the tie point: the pressure at segment i is

    m(Pr) - m(p_i) = c q sum_j G_ij f_j + c q f_i w_i         c = 1422 T / (k h)   (q Mscf/d, k absolute)

G is the pseudo-steady-state Green's function (line integral of 1/r over the segment, images in the top and bottom
boundaries, regularised so that far from the wells it becomes the Dietz areal term 0.5 ln(4A/(gamma C_A r^2)) plus
PROSPER's stabilised offset; a vertical well across the layer gives the Petex X term exactly), w_i the local skin
term s h / L_i. With an infinite-conductivity wellbore (uniform pressure) the rate fractions f and the dimensionless
resistance P follow from one linear solve.

Against the user's PROSPER runs (T12 file; 6 branch geometries x skin x kv x area, infinite conductivity):
  * with kv = kh the rate per foot along both branches is reproduced to 3 significant figures in every segment;
  * the drainage-area term is exact (390 -> 1000 acres: PROSPER +0.471 = 0.5 ln(1000/390));
  * PROSPER is more productive than the Green's function by a constant ML_OFFSET;
  * vertical permeability does not act as an anisotropy transform would (PROSPER's rate profile barely changes),
    but adds ML_ANISO ln(kh/kv) to P, weighted by the horizontal share of the flow (sin^2 of the inclination), and
    multiplies the local skin by (kh/k_perp)^ML_SKIN_EXP;
  * non-Darcy flow does not redistribute the flow: PROSPER adds it as a rate-dependent skin D q on every segment
    after the Darcy split, i.e. a q^2 term ML_ND x c x dP/ds x D (dP/ds the sensitivity of P to a uniform skin).
    D 'Calculate' is the Petex D of the perforation interval (k, rw, interval length; beta from k_g = k krg(Swc)).
    PROSPER displays the calculated value divided by ~24.45; an entered D is used as typed.
The constants are fitted to those runs.

PROSPER's layer PI is q / [(m(Pr) - m(Pwf)) / 1e6] (MMscf/d per 1e6 psi^2/cp) and its composite skin is
(m(Pr) - m(Pwf)) k h / (1422 T q) - X with X = the Petex stabilised drainage term (Dietz + 0.4932, C_A and rw of the
completion).

Finite conductivity: the pressure in T1 and along each branch follows from single-phase gas gravity + Moody
friction with the flow accumulating towards the junction, plus a junction loss of ML_JUNCTION_K velocity heads in
T1 (PROSPER's junction is 2.3 psi above the tie point at 23.5 MMscf/d in the user's run - the 8 ft of T1 account for
0.8; 3 velocity heads reproduce its finite-conductivity IPR); the segment drawdowns then differ, and the total rate
is the root of the implied segment rates (brentq) with the rate shape updated between passes.

Validation (tests/test_fluxwell.py::test_multilateral_ipr_t12): 11 infinite-conductivity one-point runs (kv 1-25 md,
skin 0/1, 390/1000 acres, D 0 / entered / calculated) within 1.3 % (2.8 % for skin 0 with the calculated D, where
PROSPER's non-Darcy iteration is not converged at low drawdown - its effective q^2 coefficient rises 11 % from 5000 to
4400 psig and then stays), the infinite-conductivity IPR (20 points) within 0.2 % below 4200 psig, the finite-
conductivity IPR within 4.6 % (junction pressure at 5000 psig within 1 psi, rate within 0.2 %).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import inflow as ipr

GAMMA = 1.781072418
ACRE = 43560.0
NSUB = 20                       # segments per perforated survey section (PROSPER)
ML_OFFSET = -0.2628             # fitted to the user's T12 runs (see module docstring)
ML_ANISO = 0.8029
ML_SKIN_EXP = 1.0
ML_ND = 1.2664
ML_JUNCTION_K = 3.0            # junction loss (velocity heads in T1)
ML_ACCEL = False
ML_EK_MAX = 0.9
ML_FRICTION = 1.0              # multiplier on the Moody friction in T1 and the branches (finite conductivity)
PSS_OFFSET = ipr.PETEX_PSS_OFFSET


# ------------------------------------------------------------------ geometry
def trajectory(stations, start_xyz):
    """stations [(md, tvd, azimuth deg)], first at start_xyz (x east, y north, z TVD). Straight lines between stations,
    the horizontal step sqrt(dMD^2 - dTVD^2) along the azimuth of the lower station. Returns (md, xyz)."""
    pts = [np.asarray(start_xyz, float)]
    for (m0, t0, _), (m1, t1, a1) in zip(stations[:-1], stations[1:]):
        dh = np.sqrt(max((m1 - m0) ** 2 - (t1 - t0) ** 2, 0.0))
        az = np.radians(a1)
        pts.append(pts[-1] + np.array([dh * np.sin(az), dh * np.cos(az), t1 - t0]))
    return np.array([s[0] for s in stations], float), np.array(pts)


def point_at(md, mds, pts):
    i = int(np.clip(np.searchsorted(mds, md, side="right") - 1, 0, len(mds) - 2))
    f = (md - mds[i]) / (mds[i + 1] - mds[i])
    return pts[i] + f * (pts[i + 1] - pts[i])


@dataclass
class Perforation:
    md_top: float
    md_bottom: float
    skin: float = 0.0
    d: float | None = None          # non-Darcy D 1/(Mscf/d); None = calculate (Petex, interval length)


@dataclass
class Branch:
    name: str
    stations: list                  # [(md, tvd, azimuth)], first = junction
    id_in: float = 2.8
    rough_in: float = 0.0006
    perfs: list = field(default_factory=list)


@dataclass
class Reservoir:
    pr: float                       # psia
    t_f: float
    k: float                        # md (horizontal)
    h: float                        # ft
    area: float                     # acres
    top: float                      # TVD ft
    kv: float | None = None         # md; None = k
    phi: float = 0.2
    swc: float = 0.2
    ca: float = 31.6
    rw: float = 0.354


@dataclass
class MultilateralWell:
    tie_md: float
    tie_tvd: float
    t1_stations: list               # [(md, tvd, azimuth)] tie point -> junction
    branches: list
    res: Reservoir
    t1_id: float = 2.8
    t1_rough: float = 0.0006

    def junction_xyz(self):
        _, p = trajectory(self.t1_stations, (0.0, 0.0, self.tie_tvd))
        return p[-1]

    def segments(self):
        j = self.junction_xyz()
        out = []
        for b in self.branches:
            mds, pts = trajectory(b.stations, j)
            for k, pf in enumerate(b.perfs):
                for i in range(len(mds) - 1):
                    lo, hi = max(pf.md_top, mds[i]), min(pf.md_bottom, mds[i + 1])
                    if hi - lo <= 1e-9:
                        continue
                    e = np.linspace(lo, hi, NSUB + 1)
                    for a, c in zip(e[:-1], e[1:]):
                        out.append({"branch": b.name, "perf": k, "md0": a, "md1": c, "p0": point_at(a, mds, pts),
                                    "p1": point_at(c, mds, pts), "skin": pf.skin})
        if not out:
            raise ValueError("No perforated interval lies on a branch.")
        return out

    def check(self):
        bad = []
        r = self.res
        for s in self.segments():
            for p in (s["p0"], s["p1"]):
                if not (r.top - 1e-6 <= p[2] <= r.top + r.h + 1e-6):
                    bad.append(f"{s['branch']}: perforations at {p[2]:.1f} ft TVD lie outside the reservoir "
                               f"({r.top:g}-{r.top + r.h:g} ft TVD).")
                    break
        half = np.sqrt(r.area * ACRE) / 2
        for s in self.segments():
            if max(abs(s["p1"][0]), abs(s["p1"][1])) > half:
                bad.append(f"{s['branch']}: the well leaves the drainage area.")
                break
        return sorted(set(bad))


# ------------------------------------------------------------------ Green's function
def _avg_inv_r(P, A, B):
    L = np.linalg.norm(B - A, axis=-1)
    R1 = np.linalg.norm(P - A, axis=-1)
    R2 = np.linalg.norm(P - B, axis=-1)
    return np.log((R1 + R2 + L) / np.maximum(R1 + R2 - L, 1e-30)) / np.maximum(L, 1e-12)


def green(segs, res: Reservoir, nimg=12):
    """Dimensionless PSS resistance matrix (isotropic geometry) at the wellbore wall of each segment."""
    h = res.h
    A = np.array([np.r_[s["p0"][:2], s["p0"][2] - res.top] for s in segs])
    B = np.array([np.r_[s["p1"][:2], s["p1"][2] - res.top] for s in segs])
    n = len(segs)
    M = 0.5 * (A + B)
    D = B - A
    perp = np.stack([-D[:, 1], D[:, 0], np.zeros(n)], 1)
    nn = np.linalg.norm(perp, axis=1)
    perp = np.where(nn[:, None] > 1e-9, perp / np.maximum(nn, 1e-12)[:, None], np.array([1.0, 0.0, 0.0]))
    T = (M + perp * res.rw)[:, None, :]
    Lp = 2 * h
    G = np.zeros((n, n))
    for fam in (1, -1):
        for k in range(-nimg, nimg + 1):
            Aj, Bj = A.copy(), B.copy()
            Aj[:, 2] = fam * A[:, 2] + k * Lp
            Bj[:, 2] = fam * B[:, 2] + k * Lp
            G += _avg_inv_r(T, Aj[None], Bj[None])
    G *= h / 2
    Z = (nimg + 0.5) * Lp
    rho = np.linalg.norm(M[:, None, :2] - M[None, :, :2], axis=-1)
    for fam in (1, -1):
        d = M[:, None, 2] - fam * M[None, :, 2]
        G -= h / 2 / Lp * (np.log((Z + d) + np.sqrt((Z + d) ** 2 + rho ** 2))
                           + np.log((Z - d) + np.sqrt((Z - d) ** 2 + rho ** 2)))
    G += 0.5 * np.log(4 * res.area * ACRE / (GAMMA * res.ca)) + PSS_OFFSET
    return G


def _incl(segs):
    L = np.array([np.linalg.norm(s["p1"] - s["p0"]) for s in segs])
    dz = np.array([s["p1"][2] - s["p0"][2] for s in segs])
    sin2 = np.clip(1 - (dz / np.maximum(L, 1e-12)) ** 2, 0.0, 1.0)
    return L, sin2


@dataclass
class DarcySolution:
    segs: list
    f: np.ndarray                   # rate fractions
    P: float                        # dimensionless resistance (incl. skin and the calibration terms)
    dPds: float                     # sensitivity of P to a uniform skin
    G: np.ndarray
    w_unit: np.ndarray              # per-segment local-skin weight per unit skin


def darcy(well: MultilateralWell, pressures_dm=None, c=None, q_guess=None):
    """Infinite conductivity (pressures_dm None): rate fractions and P. With pressures_dm (per-segment
    m(Pr) - m(p_i), psi2/cp) and c, the per-segment rates (Mscf/d) of the linear system (finite conductivity)."""
    r = well.res
    segs = well.segments()
    G = green(segs, r)
    L, sin2 = _incl(segs)
    kv = r.k if r.kv is None else r.kv
    ratio = max(r.k / max(kv, 1e-9), 1.0)
    kperp = np.sqrt(1 - sin2 + sin2 / ratio)               # k_perp / kh (1 vertical, sqrt(kv/kh) horizontal)
    w_unit = r.h / L * (1 / kperp) ** ML_SKIN_EXP
    sk = np.array([s["skin"] for s in segs])
    K = G + np.diag(sk * w_unit)
    n = len(segs)
    Mx = np.zeros((n + 1, n + 1))
    Mx[:n, :n] = K
    Mx[:n, n] = -1
    Mx[n, :n] = 1
    rhs = np.zeros(n + 1)
    rhs[n] = 1
    x = np.linalg.solve(Mx, rhs)
    f, P0 = x[:n], x[n]
    # sensitivity to a uniform skin at fixed fractions
    dPds = float(np.sum(f * f * w_unit))
    P = P0 + ML_OFFSET + ML_ANISO * np.log(ratio) * float(np.sum(f * sin2))
    sol = DarcySolution(segs, f, float(P), dPds, G, w_unit)
    if pressures_dm is None:
        return sol
    # finite conductivity: K q_i... per segment: c (K q)_i + calib = dm_i (calibration terms as a uniform shift)
    extra = (ML_OFFSET + ML_ANISO * np.log(ratio) * float(np.sum(f * sin2)))
    qtot = q_guess if q_guess is not None else float(np.mean(pressures_dm)) / (c * P)
    rhs = np.asarray(pressures_dm, float) / c - extra * qtot
    q = np.linalg.solve(K, rhs)
    return sol, q


# ------------------------------------------------------------------ hydraulics and the IPR
def _flow_path(md, tvd, d_in, rough_in, t_f):
    from .vlp import FlowPath
    md = np.asarray(md, float)
    tvd = np.asarray(tvd, float)
    seg = np.diff(md)
    n = len(seg)
    return FlowPath(md, tvd, np.full(len(md), float(t_f)), seg, np.clip(np.diff(tvd) / seg, -1, 1),
                    np.full(n, np.pi / 4 * (d_in / 12) ** 2), np.full(n, d_in / 12), np.full(n, rough_in / 12),
                    np.full(n, d_in / 12))


@dataclass
class MultilateralModel:
    """Solver for a MultilateralWell with a gas PVT (gaspvt.GasPVTInputs) and the wellbore fluid (vlp.WetGasFluid)."""
    well: MultilateralWell
    gpvt: object
    fluid: object = None             # unused (kept for compatibility); wellbore gradients are single-phase gas
    wgr: float = 0.0
    cgr: float = 0.0

    def __post_init__(self):
        from . import gaspvt as gp
        r = self.well.res
        self.sol = darcy(self.well)
        self.c = 1422.0 * (r.t_f + 459.67) / (r.k * r.h)
        grid = np.linspace(14.696, r.pr * 1.002 + 5, 800)
        self._mgrid = grid
        self._m = gp.pseudo_pressure(self.gpvt, grid, r.t_f)
        self.mu_r = float(gp.gas_props(self.gpvt, np.array([r.pr]), np.array([r.t_f]))[3][0])
        self.d_seg = self._d_per_segment()
        self.D = float(np.sum(self.sol.f * self.d_seg))
        self.b = ML_ND * self.c * self.sol.dPds * self.D
        kv = r.k if r.kv is None else r.kv
        L, sin2 = _incl(self.sol.segs)
        self.extra = ML_OFFSET + ML_ANISO * np.log(max(r.k / max(kv, 1e-9), 1.0)) * float(np.sum(self.sol.f * sin2))

    def m(self, p):
        return np.interp(p, self._mgrid, self._m)

    def calc_d(self, length):
        """Petex non-Darcy D (1/(Mscf/d)) of a perforation interval: Dake with beta = 2.73e10 / k_g^1.1045
        (k_g = k krg(Swc)), mu at Pr, the interval length as both thickness and production interval."""
        r = self.well.res
        kg = r.k * ipr.petex_krg(r.swc)
        beta = 2.73e10 / kg ** 1.1045
        return 3.161e-12 * beta * self.gpvt.gas_sg * r.k / (1637.0 * self.mu_r * max(length, 1e-6) * r.rw)

    def _d_per_segment(self):
        out = []
        for s in self.sol.segs:
            b = next(b for b in self.well.branches if b.name == s["branch"])
            pf = b.perfs[s["perf"]]
            out.append(self.calc_d(pf.md_bottom - pf.md_top) if pf.d is None else float(pf.d))
        return np.array(out)

    @property
    def layer_x(self):
        r = self.well.res
        return ipr.dietz_term(r.area, r.ca, r.rw) + PSS_OFFSET

    # ---- infinite conductivity
    def rate_infinite(self, pwf):
        """Total gas rate (Mscf/d) with the whole wellbore at pwf (psia)."""
        dm = max(float(self.m(self.well.res.pr) - self.m(pwf)), 0.0)
        a = self.c * self.sol.P
        return dm / a if self.b <= 0 else (-a + np.sqrt(a * a + 4 * self.b * dm)) / (2 * self.b)

    # ---- finite conductivity
    def _gas_tables(self):
        if getattr(self, "_tab", None) is None:
            from . import gaspvt as gp
            r = self.well.res
            p = np.linspace(14.7, r.pr * 1.05 + 50, 600)
            z, bg, rho, mu = gp.gas_props(self.gpvt, p, np.full(p.size, r.t_f))
            self._tab = (p, np.asarray(rho, float), np.asarray(mu, float), np.asarray(bg, float))
        return self._tab

    def _march(self, path, p0, q_seg_flow):
        """Pressure at path nodes from p0 at node 0 (flow towards node 0) with the flow (Mscf/d) in each path
        segment: single-phase gas (the liquid of a wet gas is a few bbl/MMscf), gravity + Moody friction x
        ML_FRICTION, Heun steps."""
        from .correlations import friction_factor
        pt, rt, mt, bt = self._gas_tables()
        P = [p0]
        p = p0
        for i in range(len(path.seg_len)):
            q = max(float(q_seg_flow[i]), 0.0) * 1000.0           # scf/d
            d = path.d_h[i]
            a = path.area[i]
            eps = path.rough[i] / d

            def grad(pp):
                rho, mu, bg = np.interp(pp, pt, rt), np.interp(pp, pt, mt), np.interp(pp, pt, bt)
                v = q * bg / 86400.0 / a
                re = 1488.0 * rho * v * d / max(mu, 1e-6)
                f = float(friction_factor(max(re, 1.0), eps)) if q > 0 else 0.0
                g = rho * path.sin_t[i] / 144.0 + ML_FRICTION * f * rho * v * v / (2 * 32.174 * d) / 144.0
                ek = min(rho * v * v / (32.174 * 144.0 * max(pp, 14.7)), ML_EK_MAX) if ML_ACCEL else 0.0
                return g / (1.0 - ek)

            L = path.seg_len[i]
            g0 = grad(p)
            g1 = grad(max(p + g0 * L, 14.7))
            p = max(p + 0.5 * (g0 + g1) * L, 14.7)
            P.append(p)
        return np.array(P)

    def _segment_pressures(self, p_tie, q):
        """Wellbore pressure at each segment midpoint for segment rates q (Mscf/d); also the junction pressure."""
        w = self.well
        st = w.t1_stations
        t1 = _flow_path([s[0] for s in st], [s[1] for s in st], w.t1_id, w.t1_rough, w.res.t_f)
        Q = float(np.sum(q))
        pj = float(self._march(t1, p_tie, np.full(len(t1.seg_len), Q))[-1])
        if ML_JUNCTION_K > 0 and Q > 0:              # junction loss, ML_JUNCTION_K velocity heads in T1
            pt, rt, mt, bt = self._gas_tables()
            for _ in range(3):
                pm = 0.5 * (p_tie + pj)
                rho, bg = np.interp(pm, pt, rt), np.interp(pm, pt, bt)
                v = Q * 1000.0 * bg / 86400.0 / t1.area[0]
                dpk = ML_JUNCTION_K * rho * v * v / (2 * 32.174) / 144.0
                pj = float(self._march(t1, p_tie, np.full(len(t1.seg_len), Q))[-1]) + dpk
        segs = self.sol.segs
        p_seg = np.empty(len(segs))
        for b in w.branches:
            idx = [i for i, s in enumerate(segs) if s["branch"] == b.name]
            mds = [s_[0] for s_ in b.stations]
            tvds = [s_[1] for s_ in b.stations]
            mids = [0.5 * (segs[i]["md0"] + segs[i]["md1"]) for i in idx]
            nodes = np.unique(np.r_[mds[0], mids, mds[1:-1]])
            nodes = nodes[nodes <= max(mids) + 1e-9]
            tv = np.interp(nodes, mds, tvds)
            path = _flow_path(nodes, tv, b.id_in, b.rough_in, w.res.t_f)
            md0 = np.array([segs[i]["md0"] for i in idx])
            qb = q[idx]
            flow = [float(qb[md0 >= nodes[k + 1] - 1e-9].sum() + 0.5 * qb[(md0 < nodes[k + 1] - 1e-9)
                    & (np.array([segs[i]["md1"] for i in idx]) > nodes[k + 1] + 1e-9)].sum()) for k in range(len(nodes) - 1)]
            # flow through a path segment = rate entering beyond its lower node
            flow = [float(qb[np.array(mids) >= nodes[k + 1] - 1e-9].sum()) for k in range(len(nodes) - 1)]
            P = self._march(path, pj, flow) if len(nodes) > 1 else np.array([pj])
            p_seg[idx] = np.interp(mids, nodes, P)
        return p_seg, pj

    def solve_finite(self, p_tie, passes=3):
        """Finite conductivity at tie-point pressure p_tie (psia): wellbore pressures from the T1 and branch gradients
        with the flow accumulating towards the junction. The total rate is the root of Q -> sum of the segment
        rates implied by the segment drawdowns (brentq), with the rate shape updated between passes.
        Returns dict with segment rates (Mscf/d), pressures, junction pressure and total."""
        from scipy.optimize import brentq
        G = self.sol.G
        sk = np.array([s["skin"] for s in self.sol.segs])
        Kinv = np.linalg.inv(G + np.diag(sk * self.sol.w_unit))
        mr = float(self.m(self.well.res.pr))
        q_inf = self.rate_infinite(p_tie)
        n = len(sk)
        if q_inf <= 0:
            return {"q_seg": np.zeros(n), "p_seg": np.full(n, p_tie), "pj": p_tie, "Q": 0.0}
        shape = self.sol.f.copy()

        def implied(Q, shape):
            p_seg, pj = self._segment_pressures(p_tie, shape * Q)
            dm = mr - self.m(p_seg) - self.c * (self.extra * Q + ML_ND * self.sol.dPds * self.D * Q * Q)
            return Kinv @ (dm / self.c), p_seg, pj

        Q = q_inf
        for _ in range(passes):
            g = lambda Q_: float(np.sum(implied(Q_, shape)[0])) - Q_
            hi = q_inf * 1.0001
            Q = brentq(g, 1e-9 * hi, hi, xtol=1e-6 * hi) if g(hi) < 0 else hi
            q, p_seg, pj = implied(Q, shape)
            if np.all(q > 0) and q.sum() > 0:
                shape = q / q.sum()
        q = shape * Q
        p_seg, pj = self._segment_pressures(p_tie, q)
        return {"q_seg": q, "p_seg": p_seg, "pj": pj, "Q": Q}

    def results(self, p_tie, finite=True):
        """(summary dict, segment table) at tie-point pressure p_tie (psia), PROSPER's MultiLateral report."""
        r = self.well.res
        if finite:
            s = self.solve_finite(p_tie)
            q, p_seg, pj, Q = s["q_seg"], s["p_seg"], s["pj"], s["Q"]
        else:
            Q = self.rate_infinite(p_tie)
            q, p_seg, pj = self.sol.f * Q, np.full(len(self.sol.f), p_tie), p_tie
        # PROSPER's layer PI and composite skin use the wellbore pressure at the perforations (flow-weighted)
        p_w = float(np.sum(q * p_seg) / np.sum(q)) if np.sum(q) > 0 else p_tie
        dm = float(self.m(r.pr) - self.m(p_w))
        pi = (Q / 1000.0) / (dm / 1e6) if dm > 0 else np.nan
        comp = dm * r.k * r.h / (1422.0 * (r.t_f + 459.67) * Q) - self.layer_x if Q > 0 else np.nan
        segs = self.sol.segs
        tab = pd.DataFrame({"Branch": [x["branch"] for x in segs], "MD top (ft)": [x["md0"] for x in segs],
                            "MD bottom (ft)": [x["md1"] for x in segs],
                            "TVD (ft)": [0.5 * (x["p0"][2] + x["p1"][2]) for x in segs],
                            "Rate (MMscf/d)": q / 1000.0,
                            "Rate per ft (MMscf/d/ft)": q / 1000.0 / np.array([x["md1"] - x["md0"] for x in segs]),
                            "Pressure (psia)": p_seg})
        branches = {b.name: float(tab.loc[tab.Branch == b.name, "Rate (MMscf/d)"].sum()) for b in self.well.branches}
        summ = {"Total gas rate (MMscf/d)": Q / 1000.0, "Junction pressure (psia)": pj,
                "Layer PI (MMscf/d per 1e6 psi2/cp)": pi, "Composite skin": comp,
                **{f"{k} rate (MMscf/d)": v for k, v in branches.items()},
                "Non-Darcy D used (1/(Mscf/d))": self.D}
        return summ, tab

    def curve(self, p_min=10.0 + 14.696, n=20, finite=True):
        """IPR at the tie point: pressures from Pr down to p_min (psia); rates MMscf/d."""
        pr = self.well.res.pr
        p = np.linspace(pr - 1.0, p_min, n)
        q = np.array([(self.solve_finite(x)["Q"] if finite else self.rate_infinite(x)) / 1000.0 for x in p])
        return p, q


def ipr_curve(well: MultilateralWell, gpvt, wgr=0.0, cgr=0.0, finite=True, n=20, pr=None, p_min=24.696):
    """IPR at the tie point (Mscf/d against psia) as an inflow.IPRCurve; pr overrides the reservoir pressure."""
    import dataclasses
    if pr is not None:
        well = dataclasses.replace(well, res=dataclasses.replace(well.res, pr=float(pr)))
    mod = MultilateralModel(well, gpvt, None, wgr, cgr)
    prr = well.res.pr
    pw = np.r_[prr, np.linspace(prr - max(prr * 0.002, 0.5), p_min, n - 1)]
    if finite:
        q = np.r_[0.0, [mod.solve_finite(x)["Q"] for x in pw[1:]]]
    else:
        q = np.array([mod.rate_infinite(x) for x in pw])
    order = np.argsort(pw)
    pws, qs = pw[order], q[order]
    s0, _ = mod.results(prr - 300.0 if prr > 600 else prr * 0.5, finite=finite)
    params = {"Conductivity": "finite" if finite else "infinite", "Segments": len(mod.sol.segs),
              "Dimensionless resistance P": mod.sol.P, "Skin sensitivity dP/ds": mod.sol.dPds,
              "Non-Darcy D (1/(Mscf/d))": mod.D, "Composite skin (300 psi drawdown)": s0["Composite skin"],
              "Layer PI (300 psi drawdown, MMscf/d per 1e6 psi2/cp)": s0["Layer PI (MMscf/d per 1e6 psi2/cp)"]}
    curve = ipr.IPRCurve("Multilateral Well", "gas", prr, pw, q, params, [], None,
                         lambda p: np.interp(np.asarray(p, float), pws, qs))
    curve.notes.append("Multilateral network (PROSPER Tutorial 12): line-source segments in a closed box, "
                       + ("finite-conductivity wellbore (gravity + friction in T1 and the branches)."
                          if finite else "infinite-conductivity wellbore."))
    curve.model_obj = mod
    return curve


def example_t12():
    """The user's PROSPER Tutorial 12 file (Powderhall gas: T06 PVT, two branches from a junction at 5408 ft MD)."""
    perfs = lambda: [Perforation(5420, 5470, 1.0), Perforation(5510, 5560, 1.0)]
    b1 = Branch("Branch 1", [(5408, 4905, 0), (5430, 4920, 54), (5465, 4940, 58), (5510, 4945, 62), (5532, 4940, 68),
                             (5562, 4930, 74)], 2.8, 0.0006, perfs())
    b2 = Branch("Branch 2", [(5408, 4905, 0), (5435, 4920, 240), (5454, 4925, 235), (5480, 4915, 227),
                             (5500, 4910, 221), (5550, 4920, 214), (5570, 4935, 210)], 2.8, 0.0006, perfs())
    res = Reservoir(5300 + 14.696, 250.0, 25.0, 55.0, 390.0, 4900.0, 2.5, 0.18, 0.2, 31.6, 0.354)
    return MultilateralWell(5400.0, 4900.0, [(5400, 4900, 0), (5408, 4905, 0)], [b1, b2], res)
