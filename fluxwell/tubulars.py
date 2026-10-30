"""Tubing and casing strength for the well operating envelope: API TR 5C3 / ISO TR 10400 ratings and a depth-by-depth
burst / collapse / axial / triaxial check against the pressure profiles inside and outside the string.

Ratings (pipe body, nominal wall, no connection, wear, corrosion or buckling - those are not covered):
  burst     Barlow with 87.5 % minimum wall:  Pb = 0.875 * 2 Y t / D
  collapse  API four regimes chosen by D/t (yield, plastic, transition, elastic) with the A, B, C, F, G coefficients
            from the yield strength; reproduces the published API tables (3-1/2" 9.30 lb/ft L-80: 10,535 psi plastic).
  axial     tension reduces the collapse yield: Ypa = [sqrt(1 - 0.75 s^2) - 0.5 s] Y with s = sigma_a / Y (API 5C3) or
            s = (sigma_a + p_i) / Y (ISO TR 10400, internal pressure included - the default here); compression is ignored
            (Ypa = Y), as in the API method.
  internal pressure on collapse: equivalent external pressure  pe = po - (1 - 2 / (D/t)) pi.
  triaxial  von Mises at the inner wall: sigma_r = -pi, sigma_theta from Lame, sigma_a = axial force / As.
Axial force = buoyed weight of the string below the depth (buoyancy 1 - rho/65.5 with the annulus fluid) + a single
additional load (tension positive) from a tubing movement analysis.

Checked against the user's Tubing Burst & Collapse Calculator workbook (3-1/2" 9.30 lb/ft L-80, packer 9,000 ft): every
rating in its catalog, and the load profile's burst, collapse, axial and triaxial safety factors, to 1e-9 (API reduced
yield)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRADES = {"H-40": 40000.0, "J-55": 55000.0, "K-55": 55000.0, "L-80": 80000.0, "N-80": 80000.0, "C-90": 90000.0,
          "T-95": 95000.0, "P-110": 110000.0, "Q-125": 125000.0}

# nominal API tubing / casing: label -> (OD in, nominal weight lb/ft, wall in)
TUBING = {'2-3/8" 4.70 lb/ft': (2.375, 4.70, 0.190), '2-7/8" 6.50 lb/ft': (2.875, 6.50, 0.217),
          '2-7/8" 8.70 lb/ft': (2.875, 8.70, 0.308), '3-1/2" 9.30 lb/ft': (3.5, 9.30, 0.254),
          '3-1/2" 12.95 lb/ft': (3.5, 12.95, 0.375), '4-1/2" 12.75 lb/ft': (4.5, 12.75, 0.271),
          '5-1/2" 17.00 lb/ft': (5.5, 17.00, 0.304)}
CASING = {'5-1/2" 17.00 lb/ft': (5.5, 17.00, 0.304), '5-1/2" 20.00 lb/ft': (5.5, 20.00, 0.361),
          '5-1/2" 23.00 lb/ft': (5.5, 23.00, 0.415), '7" 23.00 lb/ft': (7.0, 23.00, 0.317),
          '7" 26.00 lb/ft': (7.0, 26.00, 0.362), '7" 29.00 lb/ft': (7.0, 29.00, 0.408),
          '7" 32.00 lb/ft': (7.0, 32.00, 0.453), '7-5/8" 29.70 lb/ft': (7.625, 29.70, 0.375),
          '7-5/8" 33.70 lb/ft': (7.625, 33.70, 0.430), '9-5/8" 47.00 lb/ft': (9.625, 47.00, 0.472),
          '9-5/8" 53.50 lb/ft': (9.625, 53.50, 0.545)}


def collapse_coefficients(y):
    """API 5C3 / ISO TR 10400 collapse coefficients A, B, C, F, G and the D/t regime boundaries for yield y (psi)."""
    a = 2.8762 + 1.0679e-6 * y + 2.1301e-11 * y ** 2 - 5.3132e-17 * y ** 3
    b = 0.026233 + 5.0609e-7 * y
    c = -465.93 + 0.030867 * y - 1.0483e-8 * y ** 2 + 3.6989e-14 * y ** 3
    ba = b / a
    r = 3 * ba / (2 + ba)
    f = 46.95e6 * r ** 3 / (y * (r - ba) * (1 - r) ** 2)
    g = f * ba
    dt_yp = (np.sqrt((a - 2) ** 2 + 8 * (b + c / y)) + (a - 2)) / (2 * (b + c / y))
    dt_pt = y * (a - f) / (c + y * (b - g))
    dt_te = (2 + ba) / (3 * ba)
    return dict(A=a, B=b, C=c, F=f, G=g, dt_yp=dt_yp, dt_pt=dt_pt, dt_te=dt_te)


def collapse_regime(dt, y):
    k = collapse_coefficients(y)
    return "Yield" if dt <= k["dt_yp"] else "Plastic" if dt <= k["dt_pt"] else (
        "Transition" if dt <= k["dt_te"] else "Elastic")


def collapse_rating(od, t, y):
    """API collapse pressure (psi), no axial load."""
    dt = od / t
    k = collapse_coefficients(y)
    if dt <= k["dt_yp"]:
        return 2 * y * (dt - 1) / dt ** 2
    if dt <= k["dt_pt"]:
        return y * (k["A"] / dt - k["B"]) - k["C"]
    if dt <= k["dt_te"]:
        return y * (k["F"] / dt - k["G"])
    return 46.95e6 / (dt * (dt - 1) ** 2)


def burst_rating(od, t, y):
    """API burst (internal yield) pressure, Barlow with 87.5 % wall (psi)."""
    return 0.875 * 2 * y * t / od


def reduced_yield(y, sigma_a, p_i=0.0, method="ISO"):
    """Yield strength reduced by axial tension for collapse; compression ignored. method "API" (sigma_a only) or
    "ISO" (sigma_a + p_i, ISO TR 10400)."""
    s = (sigma_a + (p_i if method == "ISO" else 0.0)) / y
    if s <= 0:
        return y
    return max((np.sqrt(max(1 - 0.75 * s * s, 0.0)) - 0.5 * s) * y, 0.0)


def von_mises_inner(od, t, p_i, p_o, sigma_a):
    """Von Mises stress at the inner wall (psi): radial -p_i, hoop from Lame, axial sigma_a."""
    ro, ri = od / 2, od / 2 - t
    s_r = -p_i
    s_t = (p_i * (ro ** 2 + ri ** 2) - 2 * p_o * ro ** 2) / (ro ** 2 - ri ** 2)
    return np.sqrt(((s_r - s_t) ** 2 + (s_t - sigma_a) ** 2 + (sigma_a - s_r) ** 2) / 2)


@dataclass
class Tubular:
    """One string. od, t in inches; weight lb/ft; grade from GRADES or yield given; derate multiplies the yield."""
    od: float
    t: float
    weight: float
    grade: str = "L-80"
    derate: float = 1.0
    yield_psi: float | None = None

    @property
    def y(self):
        return (self.yield_psi if self.yield_psi else GRADES[self.grade]) * self.derate

    @property
    def dt(self):
        return self.od / self.t

    @property
    def area(self):
        return np.pi / 4 * (self.od ** 2 - (self.od - 2 * self.t) ** 2)

    @property
    def burst(self):
        return burst_rating(self.od, self.t, self.y)

    @property
    def collapse(self):
        return collapse_rating(self.od, self.t, self.y)

    def collapse_at(self, sigma_a, p_i=0.0, method="ISO"):
        return collapse_rating(self.od, self.t, reduced_yield(self.y, sigma_a, p_i, method))

    @property
    def body_yield(self):
        return self.y * self.area


@dataclass
class DesignFactors:
    burst: float = 1.1
    collapse: float = 1.0
    axial: float = 1.3
    triaxial: float = 1.25


@dataclass
class LoadCheck:
    """Depth-by-depth results of one load case."""
    tvd: np.ndarray
    burst_load: np.ndarray
    collapse_load: np.ndarray
    collapse_rating: np.ndarray
    axial_force: np.ndarray
    sf_burst: np.ndarray
    sf_collapse: np.ndarray
    sf_axial: np.ndarray
    sf_vme: np.ndarray
    df: DesignFactors = field(default_factory=DesignFactors)

    def min_sf(self):
        def m(a):
            a = a[np.isfinite(a)]
            return float(a.min()) if a.size else np.inf
        return {"burst": m(self.sf_burst), "collapse": m(self.sf_collapse), "axial": m(self.sf_axial),
                "triaxial": m(self.sf_vme)}

    def passes(self):
        s = self.min_sf()
        return {k: s[k] >= getattr(self.df, k) for k in s}

    @property
    def ok(self):
        return all(self.passes().values())


def axial_force(pipe: Tubular, tvd, bottom_tvd, fluid_ppg, extra=0.0):
    """Buoyed weight of the string below each depth plus the additional load (lbf, tension positive)."""
    return pipe.weight * (bottom_tvd - np.asarray(tvd, float)) * (1 - fluid_ppg / 65.5) + extra


def check_profile(pipe: Tubular, tvd, p_in, p_out, bottom_tvd, fluid_ppg=8.6, extra_axial=0.0,
                  df: DesignFactors | None = None, method="ISO") -> LoadCheck:
    """Burst, collapse (axial-reduced, equivalent external pressure), axial and von Mises safety factors at each depth
    for inside / outside pressure profiles (psi, gauge or absolute alike - differences only) over tvd (ft)."""
    df = df or DesignFactors()
    tvd, p_in, p_out = (np.asarray(x, float) for x in (tvd, p_in, p_out))
    fa = axial_force(pipe, tvd, bottom_tvd, fluid_ppg, extra_axial)
    sa = fa / pipe.area
    burst = p_in - p_out
    coll = p_out - (1 - 2 / pipe.dt) * p_in
    rating = np.array([pipe.collapse_at(s, pi_, method) for s, pi_ in zip(sa, p_in)])
    with np.errstate(divide="ignore", invalid="ignore"):
        sf_b = np.where(burst > 0, pipe.burst / burst, np.inf)
        sf_c = np.where(coll > 0, rating / coll, np.inf)
        sf_a = np.where(fa > 0, pipe.body_yield / fa, np.inf)
    vme = np.array([von_mises_inner(pipe.od, pipe.t, pi_, po_, s) for pi_, po_, s in zip(p_in, p_out, sa)])
    sf_v = pipe.y / np.maximum(vme, 1e-9)
    return LoadCheck(tvd, burst, coll, rating, fa, sf_b, sf_c, sf_a, sf_v, df)


def min_internal_for_collapse(pipe: Tubular, p_out, sigma_a=0.0, df: DesignFactors | None = None, method="ISO"):
    """Lowest internal pressure (psi) that keeps the collapse safety factor at the design factor for outside pressure
    p_out: p_out - (1 - 2/(D/t)) p_i = rating(p_i) / DF, solved for p_i (rating depends on p_i through the ISO
    reduced yield). 0 when the string can be evacuated."""
    df = df or DesignFactors()
    k = 1 - 2 / pipe.dt

    def excess(pi_):
        return p_out - k * pi_ - pipe.collapse_at(sigma_a, pi_, method) / df.collapse

    if excess(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, max(p_out / k, 1.0) * 2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi


def max_internal_for_burst(pipe: Tubular, p_out, df: DesignFactors | None = None):
    """Highest internal pressure (psi) at the burst design factor for outside (back-up) pressure p_out."""
    df = df or DesignFactors()
    return p_out + pipe.burst / df.burst
