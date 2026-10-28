"""PROSPER's Enthalpy Balance temperature model (User Guide 2.19.2.3.2; Tutorial 40).

The temperature march is the Improved Approximation's (thermal.improved_profile: gravity, Joule-Thomson and heat
loss with the same fluid thermodynamics), but the heat-transfer coefficient is calculated at every segment from the
completion, the drilling record and the lithology, and depends on the producing time:

    dQ = 2 pi (Tf - Te) dL / ( f(t) / ke  +  1 / (r_to U_to) )
    f(t) = 0.982 ln(1 + 1.81 sqrt(alpha t) / r_h)          (Kwan-Chu & Thakur; alpha = ke / (rho c))
    1 / (r_to U_to) = 1/(r_ti h_f) + ln(r_to/r_ti)/k_tubing + 1/(r_to (h_c + h_r))  + sum ln(r2/r1)/k  (casings,
                       cement below each string's top of cement, mud above it)

- annulus (tubing to the innermost casing, completion fluid above the packer): free convection
  h_c = C 0.049 (Gr Pr)^(1/3) Pr^0.074 k / (r_to ln(r_ci/r_to)) plus radiation between the walls;
- formation: the lithology layer at the depth, the hole of the deepest string there as r_h; dry-rock properties from
  PROSPER's default database, wetted by the in-situ fluid in the pores.
Calibrated on the user's PROSPER Tutorial 40 file (gradients at 5, 50 and 500 days since production started, three
casing strings, CaBr2 completion fluid, shale / sandstone / shale): FORMATION_RHOC_MULT 1.77 on the rock + pore-fluid
volumetric heat capacity, WET_ROCK 4.76 (k_wet = k_dry (1 + WET_ROCK phi k_fluid / k_water): 1.57 x for the 12 %
porosity water sandstone), ANNULUS_CONVECTION 1.21, no annulus resistance below the packer (PROSPER's U jumps up
there). PROSPER's U at every node (3 times x 46 nodes) within 3 % rms, 13 % at worst (the 500 ft between the
innermost string's top of cement and the next shoe).
U is reported referred to the flow ID (as the Rough and Improved Approximations' U).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import thermal as TH

SIGMA = 1.713e-9            # Stefan-Boltzmann, BTU/h/ft2/R4
RHO_W = 62.428
G_FT_H2 = 32.174 * 3600.0 ** 2
FORMATION_RHOC_MULT = 1.77
WET_ROCK = 4.76
ANNULUS_CONVECTION = 1.21

ROCKS = {   # dry rock: Cp BTU/lb/F, conductivity BTU/h/ft/F, specific gravity (PROSPER default database)
    "Sandstone": (0.183, 1.06, 2.64), "Shale": (0.224, 0.7, 2.4), "Limestone": (0.202, 0.54, 2.71),
    "Dolomite": (0.219, 1.0, 2.87), "Halite": (0.219, 2.8, 2.17), "Anhydrite": (0.265, 0.75, 2.96),
    "Gypsum": (0.259, 0.75, 2.32), "Lignite": (0.3, 2.0, 1.5), "Volcanics": (0.2, 1.6, 2.65),
}
POROUS = {"Sandstone", "Limestone", "Dolomite"}
IN_SITU = {   # Cp, conductivity, specific gravity
    "Water": (1.0, 0.35, 1.0), "Heavy Oil": (0.49, 0.083, 0.96), "Medium Oil": (0.5, 0.0825, 0.9),
    "Light Oil": (0.5, 0.0815, 0.84), "Gas": (0.26, 0.0215, 0.01),
}
COMPLETION_FLUIDS = {   # liquid: (Cp, conductivity, viscosity cp, thermal expansion 1/F)
    "Calcium Bromide": (0.65, 0.35, 0.8, 2.5e-4), "Calcium Chloride": (0.75, 0.36, 0.8, 2.5e-4),
    "Sodium Chloride": (0.85, 0.36, 0.6, 2.5e-4), "Water": (1.0, 0.36, 0.4, 3.0e-4), "Diesel": (0.5, 0.08, 1.5, 4.5e-4),
}
LBGAL_TO_LBFT3 = 7.4805


@dataclass
class Casing:
    drilling_depth: float      # ft MD, bottom of the hole section
    hole_d: float              # in
    shoe: float                # ft MD
    od: float                  # in
    weight: float              # lb/ft
    top_cement: float          # ft MD
    top: float = 0.0           # ft MD
    mud_density: float = 10.0  # lb/gal

    @property
    def id(self):
        """Inside diameter from OD and weight (steel 490 lb/ft3)."""
        return float(np.sqrt(max(self.od ** 2 - 4 * self.weight / 490.0 * 144.0 / np.pi, 0.01)))


@dataclass
class Lithology:
    bottom: float              # ft MD
    rock: str = "Shale"
    porosity: float = 0.0
    fluid: str = "Water"


@dataclass
class EnthalpyData:
    casings: list
    lithology: list
    packer_md: float = 1e9
    completion_fluid: str = "Calcium Bromide"
    completion_density: float = 12.0     # lb/gal
    days: float = 50.0
    cement_k: float = 0.5
    casing_k: float = 26.0
    tubing_k: float = 26.0
    tubing_emissivity: float = 0.65
    casing_emissivity: float = 0.65
    tubing_od: float | None = None       # in (None: tubing ID + 0.5)
    tubing_end: float | None = None      # ft MD (None: the flow path bottom)

    def check(self):
        out = []
        if not self.casings:
            out.append(("error", "Enter the drilling and casing record."))
        for c in self.casings:
            if not (c.od > 0 and c.weight > 0 and c.hole_d > c.od):
                out.append(("error", f"Casing to {c.shoe:g} ft: hole diameter must exceed the OD, OD and weight positive."))
        if not self.lithology:
            out.append(("error", "Enter the lithology (formation layers to the bottom of the well)."))
        for l in self.lithology:
            if l.rock not in ROCKS:
                out.append(("error", f"Unknown formation type {l.rock!r}."))
        if self.days <= 0:
            out.append(("error", "Days since production started must be positive."))
        return out


def _layer_at(eb: EnthalpyData, md):
    for l in sorted(eb.lithology, key=lambda l: l.bottom):
        if md <= l.bottom:
            return l
    return sorted(eb.lithology, key=lambda l: l.bottom)[-1]


def _formation(eb, md):
    l = _layer_at(eb, md)
    cp_r, k_r, sg_r = ROCKS[l.rock]
    phi = l.porosity if l.rock in POROUS else 0.0
    cp_f, k_f, sg_f = IN_SITU.get(l.fluid, IN_SITU["Water"])
    ke = k_r * (1.0 + WET_ROCK * phi * k_f / 0.35)
    rhoc = FORMATION_RHOC_MULT * ((1 - phi) * sg_r * RHO_W * cp_r + phi * sg_f * RHO_W * cp_f)
    return ke, ke / rhoc


def segment_geometry(path, eb: EnthalpyData):
    """Per segment: strings present (innermost first), hole radius, innermost casing ID (ft)."""
    mids = 0.5 * (path.md[1:] + path.md[:-1])
    geo = []
    for z in mids:
        strings = sorted([c for c in eb.casings if c.top <= z <= c.shoe], key=lambda c: c.od)
        if not strings:                                 # below the casing record: open hole of the deepest section
            deep = max(eb.casings, key=lambda c: c.drilling_depth)
            geo.append(([], deep.hole_d / 24.0, deep.hole_d / 24.0))
            continue
        geo.append((strings, strings[-1].hole_d / 24.0, strings[0].id / 24.0))
    return mids, geo


def _wall_resistance(eb, z, strings):
    """Casings, cement and mud outside the innermost casing ID (X units: ln(r2/r1)/k)."""
    R = 0.0
    for j, c in enumerate(strings):
        R += np.log(c.od / c.id) / eb.casing_k
        outer = strings[j + 1].id if j + 1 < len(strings) else c.hole_d
        if outer > c.od:
            R += np.log(outer / c.od) / (eb.cement_k if z >= c.top_cement else 0.5)
    return R


def u_segments(path, eb: EnthalpyData, Tf, Te, n_iter=3):
    """Overall U (BTU/h/ft2/F, referred to the flow ID) per segment for fluid temperature Tf and formation Te
    (per segment arrays, cases x segments). Also returns casing-inside and annulus temperatures."""
    Tf = np.atleast_2d(Tf)
    nseg = Tf.shape[1]
    mids, geo = segment_geometry(path, eb)
    t_hours = eb.days * 24.0
    cp_c, k_c, mu_c, beta_c = COMPLETION_FLUIDS.get(eb.completion_fluid, COMPLETION_FLUIDS["Water"])
    rho_c = eb.completion_density * LBGAL_TO_LBFT3
    mu_c = mu_c * 2.4191                        # lb/ft/h
    end = eb.tubing_end if eb.tubing_end is not None else path.md[-1]
    U = np.empty_like(Tf)
    Tci = np.full_like(Tf, np.nan)
    Tann = np.full_like(Tf, np.nan)
    for i in range(nseg):
        z = mids[i]
        strings, r_h, r_ci = geo[i]
        ke, alpha = _formation(eb, z)
        f = 0.982 * np.log(1 + 1.81 * np.sqrt(alpha * t_hours) / r_h)
        Rf = f / ke
        r_flow = path.d_ht[i] / 2.0
        Rw = _wall_resistance(eb, z, strings)
        tubing = z <= end + 1e-6
        if tubing:
            r_ti = r_flow
            r_to = (eb.tubing_od / 24.0) if eb.tubing_od else r_ti + 0.5 / 24.0
            Rt = np.log(r_to / r_ti) / eb.tubing_k
            dT = np.maximum(np.abs(Tf[:, i] - Te[:, i]) * 0.03, 0.05)        # first guess across the annulus
            Rann = np.zeros(Tf.shape[0])
            for _ in range(n_iter):
                if z <= eb.packer_md and r_ci > r_to:
                    gap = r_ci - r_to
                    gr = gap ** 3 * rho_c ** 2 * beta_c * G_FT_H2 * dT / mu_c ** 2
                    pr = cp_c * mu_c / k_c
                    hc = ANNULUS_CONVECTION * 0.049 * (gr * pr) ** (1 / 3) * pr ** 0.074 * k_c / (r_to * np.log(r_ci / r_to))
                    ta = Tf[:, i] + 459.67
                    tb = ta - dT
                    hr = SIGMA * (ta ** 2 + tb ** 2) * (ta + tb) / (1 / eb.tubing_emissivity
                                                                     + r_to / r_ci * (1 / eb.casing_emissivity - 1))
                    Rann = 1.0 / (r_to * (hc + hr))
                else:
                    Rann = np.zeros(Tf.shape[0])
                X = Rt + Rann + Rw + Rf
                dT = np.maximum((Tf[:, i] - Te[:, i]) * Rann / X, 0.05)
            X = Rt + Rann + Rw + Rf
            tci = Tf[:, i] - (Tf[:, i] - Te[:, i]) * (Rt + Rann) / X
            Tci[:, i] = tci
            Tann[:, i] = Tf[:, i] - (Tf[:, i] - Te[:, i]) * (Rt + 0.5 * Rann) / X
        else:
            X = np.full(Tf.shape[0], Rw + Rf)
        U[:, i] = 1.0 / (r_flow * X)
    return U, Tci, Tann


def profile(path, fluid, P, case, w, hc, th, t_bottom=None, n_outer=3):
    """Temperature at every node for the pressure profile P: U from the completion at the current temperatures."""
    eb = th["data"]
    te_seg = 0.5 * (path.t_form[1:] + path.t_form[:-1])
    T = th.get("_t_guess")
    if T is None or T.shape != P.shape:
        from .vlp import temperature_profile
        T = temperature_profile(path, np.broadcast_to(np.asarray(w, float), P[:, 0].shape), 0.65, 2.0, t_bottom)
    for _ in range(n_outer):
        tf_seg = 0.5 * (T[:, 1:] + T[:, :-1])
        U, _, _ = u_segments(path, eb, tf_seg, np.broadcast_to(te_seg, tf_seg.shape))
        T = TH.improved_profile(path, fluid, P, case, w, hc, U, t_bottom)
    th["_u_last"] = U
    return T
