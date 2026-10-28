"""CO2-rich streams (PROSPER's EOS option "Correct CO2 Density" / "CO2 correction"; Tutorial 39).

PROSPER describes the option as a correction of the cubic EOS so that the density, heat capacity and enthalpy of
CO2 streams reproduce the Span-Wagner reference equation, with impurities. Found on the user's Tutorial 39 file:
the pure-CO2 Span-Wagner minus cubic-EOS difference (volume, weighted by the CO2 mole fraction; enthalpy, per
lbmol) added to the cubic EOS of the mixture - see CO2Table. Tabulated once on a pressure-temperature grid and
interpolated (as PROSPER does for its EOS); LBC viscosity on the corrected molar volume.

Requires the CoolProp package (Span-Wagner CO2); without it, or with components it does not know, the plain cubic
EOS is used.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import eos as E

COOLPROP_NAMES = {
    "CO2": "CO2", "C1": "Methane", "METHANE": "Methane", "CH4": "Methane", "C2": "Ethane", "ETHANE": "Ethane",
    "C3": "Propane", "PROPANE": "Propane", "IC4": "IsoButane", "I-C4": "IsoButane", "NC4": "n-Butane",
    "N-C4": "n-Butane", "IC5": "Isopentane", "I-C5": "Isopentane", "NC5": "n-Pentane", "N-C5": "n-Pentane",
    "C6": "n-Hexane", "NC6": "n-Hexane", "N2": "Nitrogen", "H2S": "HydrogenSulfide", "H2O": "Water", "O2": "Oxygen",
    "AR": "Argon", "H2": "Hydrogen", "CO": "CarbonMonoxide",
}
PA_PER_PSI = 6894.757
J_PER_BTU_LB = 2326.0                 # J/kg per BTU/lb
LB_FT3 = 0.0624280


def available():
    try:
        import CoolProp  # noqa: F401
        return True
    except ImportError:
        return False


def reference_names(fluid: E.EOSFluid):
    """CoolProp names of the components, or None if one is unknown (pseudo-components)."""
    out = []
    for n in fluid.names:
        k = str(n).upper().replace(" ", "")
        if k not in COOLPROP_NAMES:
            return None
        out.append(COOLPROP_NAMES[k])
    return out


def wants_reference(fluid: E.EOSFluid):
    """PROSPER's CO2 correction is on in the PRP file and the stream is CO2-rich and reference-model compatible."""
    flag = str(fluid.prp.other.get("CO2 correction", "0")).strip()
    try:
        on = float(flag) != 0
    except ValueError:
        on = flag.lower() in ("yes", "true")
    if not on or not available() or reference_names(fluid) is None:
        return False
    i = [str(n).upper() for n in fluid.names].index("CO2") if "CO2" in [str(n).upper() for n in fluid.names] else None
    return i is not None and fluid.z[i] >= 0.5


# ideal-gas heat capacity, J/(mol K), T in K: a + b T + c T^2 + d T^3 (Reid, Prausnitz & Poling)
CP_IDEAL = {
    "CO2": (19.80, 7.344e-2, -5.602e-5, 1.715e-8), "Methane": (19.25, 5.213e-2, 1.197e-5, -1.132e-8),
    "Ethane": (5.409, 1.781e-1, -6.938e-5, 8.713e-9), "Propane": (-4.224, 3.063e-1, -1.586e-4, 3.215e-8),
    "IsoButane": (-1.390, 3.847e-1, -1.846e-4, 2.895e-8), "n-Butane": (9.487, 3.313e-1, -1.108e-4, -2.822e-9),
    "Isopentane": (-9.525, 5.066e-1, -2.729e-4, 5.723e-8), "n-Pentane": (-3.626, 4.873e-1, -2.580e-4, 5.305e-8),
    "n-Hexane": (-4.413, 5.820e-1, -3.119e-4, 6.494e-8), "Nitrogen": (31.15, -1.357e-2, 2.680e-5, -1.168e-8),
    "HydrogenSulfide": (31.94, 1.436e-3, 2.432e-5, -1.176e-8), "Water": (32.24, 1.924e-3, 1.055e-5, -3.596e-9),
    "Oxygen": (28.11, -3.680e-6, 1.746e-5, -1.065e-8), "Argon": (20.786, 0.0, 0.0, 0.0),
    "Hydrogen": (27.14, 9.274e-3, -1.381e-5, 7.645e-9), "CarbonMonoxide": (30.87, -1.285e-2, 2.789e-5, -1.272e-8),
}
J_MOL_TO_BTU_LBMOL = 0.429923
R_BTU = 1.98588                     # BTU/(lbmol R)


def h_ideal(coef, T_R):
    """Ideal-gas enthalpy (BTU/lbmol) of each component from 0 K."""
    TK = np.asarray(T_R, float) / 1.8
    a, b, c, d = np.asarray(coef, float).T
    return (a * TK + b * TK ** 2 / 2 + c * TK ** 3 / 3 + d * TK ** 4 / 4) * J_MOL_TO_BTU_LBMOL


def pr_enthalpy(fluid: E.EOSFluid, coef, x, P, T_F, phase=None):
    """Cubic-EOS enthalpy (BTU/lbmol): ideal gas + departure, for composition x at P psia, T F."""
    T = T_F + E.RANKINE
    _, Z = fluid._lnphi(x, P, T, phase=phase)
    am = lambda t: x @ fluid._a(t) @ x
    a = am(T)
    dadt = (am(T + 1e-3) - am(T - 1e-3)) / 2e-3
    bm = x @ fluid.b
    B = bm * P / (E.R * T)
    d1, d2 = fluid.d1, fluid.d2
    dep = E.R * T * (Z - 1) + (T * dadt - a) / (bm * (d1 - d2)) * np.log((Z + d1 * B) / (Z + d2 * B))
    return float(x @ h_ideal(coef, T) + dep * 144.0 / 778.17), float(Z)


class CO2Table:
    """PROSPER's CO2-corrected cubic EOS on a (p psia, T F) grid: density (lb/ft3), enthalpy (BTU/lb, arbitrary
    reference), cp (BTU/lb/F), Z and LBC viscosity (cp).

    Reverse-engineered from the user's PROSPER Tutorial 39 traverses (Peng-Robinson, 89.1 % CO2 + C1-C3, no PRP
    volume shift, CO2 correction on):
      volume    v = v_EOS(mixture) - z_CO2 [v_EOS(pure CO2) - v_SW(pure CO2)]     (density within 0.35 % rms)
      enthalpy  H = H_EOS(mixture) + [H_SW(pure CO2) - H_EOS(pure CO2)]          (per lbmol: adiabatic temperatures
                                                                                within 0.6 F over 14,000 ft)
    i.e. the pure-CO2 Span-Wagner correction of the cubic EOS carried over to the mixture (SW = Span & Wagner 1996,
    CoolProp). Single phase (the min-Gibbs root of the cubic) everywhere on the grid."""

    P_GRID = np.unique(np.concatenate([np.geomspace(100.0, 1000.0, 12), np.linspace(1000.0, 3000.0, 41),
                                       np.linspace(3000.0, 16000.0, 27)]))
    T_GRID = np.linspace(20.0, 420.0, 51)

    def __init__(self, fluid: E.EOSFluid):
        import CoolProp.CoolProp as CP
        from CoolProp.CoolProp import AbstractState
        self.fluid = fluid
        names = reference_names(fluid)
        coef = np.array([CP_IDEAL.get(n, (0.0, 0.0, 0.0, 0.0)) for n in names])
        i_co2 = names.index("CO2")
        zc = float(fluid.z[i_co2])
        pure = E.EOSFluid(fluid.prp, z=np.where(np.arange(fluid.n) == i_co2, 1.0, 1e-12))
        sw = AbstractState("HEOS", "CO2")
        self.mw = float(fluid.z @ fluid.mw)
        mw_co2 = float(fluid.mw[i_co2])
        shape = (len(self.P_GRID), len(self.T_GRID))
        v, h = np.empty(shape), np.empty(shape)
        for i, p in enumerate(self.P_GRID):
            for j, t in enumerate(self.T_GRID):
                hm, zm = pr_enthalpy(fluid, coef, fluid.z, p, t)
                hp, zp = pr_enthalpy(pure, coef, pure.z, p, t)
                tr = t + E.RANKINE
                sw.update(CP.PT_INPUTS, p * PA_PER_PSI, (t - 32.0) / 1.8 + 273.15)
                v_sw = mw_co2 / (sw.rhomass() * LB_FT3)
                h_sw = sw.hmass() / J_PER_BTU_LB * mw_co2
                v[i, j] = zm * E.R * tr / p - zc * (zp * E.R * tr / p - v_sw)
                h[i, j] = (hm + (h_sw - hp)) / self.mw
        rho = self.mw / v
        cp = np.gradient(h, self.T_GRID, axis=1)
        tr = self.T_GRID[None, :] + E.RANKINE
        z = self.P_GRID[:, None] * v / (E.R * tr)
        mu = np.empty(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                mu[i, j] = fluid.lbc(fluid.z, v[i, j], tr[0, j])
        kw = dict(bounds_error=False, fill_value=None)
        g = (self.P_GRID, self.T_GRID)
        self._rho = RegularGridInterpolator(g, rho, **kw)
        self._h = RegularGridInterpolator(g, h, **kw)
        self._cp = RegularGridInterpolator(g, cp, **kw)
        self._z = RegularGridInterpolator(g, z, **kw)
        self._mu = RegularGridInterpolator(g, mu, **kw)

    def _pt(self, p, t):
        p = np.clip(np.asarray(p, float), self.P_GRID[0], self.P_GRID[-1])
        t = np.clip(np.asarray(t, float), self.T_GRID[0], self.T_GRID[-1])
        p, t = np.broadcast_arrays(p, t)
        return np.stack([p.ravel(), t.ravel()], axis=-1), p.shape

    def __call__(self, name, p, t):
        pts, shape = self._pt(p, t)
        return getattr(self, "_" + name)(pts).reshape(shape)

    def temperature(self, p, h, t_guess):
        """Temperature (F) with enthalpy h (BTU/lb) at pressure p (Newton on cp)."""
        t = np.asarray(t_guess, float).copy()
        for _ in range(15):
            dt = (h - self("h", p, t)) / np.maximum(self("cp", p, t), 0.05)
            t = t + np.clip(dt, -40.0, 40.0)
            if np.all(np.abs(dt) < 1e-4):
                break
        return t


class CO2InjectionFluid:
    """Injected CO2-rich stream (EOS composition) for injection.traverse: rate MMscf/d (standard gas of the whole
    stream, 379.49 scf/lbmol). Single phase from the reference table; the cubic EOS where it finds two phases."""
    kind = "gas"
    rate_unit = "MMscf/d"
    grav = 1.0 / 144.0
    cp_key = "gas"

    def __init__(self, fluid: E.EOSFluid, table: CO2Table | None = None):
        self.eos = fluid
        self.table = table if table is not None else (CO2Table(fluid) if wants_reference(fluid) else None)
        self.mw = float(fluid.z @ fluid.mw)
        self.sg = self.mw / 28.9625
        self.lb_per_scf = self.mw / E.SCF_PER_LBMOL
        self.inp = None
        self.has_enthalpy = self.table is not None     # enthalpy (energy-balance temperatures) needs the table

    def props(self, p, t):
        """(Bg ft3/scf, density lb/ft3, viscosity cp)."""
        p = np.maximum(np.asarray(p, float), 14.7)
        t = np.broadcast_to(np.asarray(t, float), p.shape)
        if self.table is not None:
            rho, mu = self.table("rho", p, t), self.table("mu", p, t)
        else:
            rho, mu = np.empty(p.shape), np.empty(p.shape)
            for k, (pp, tt) in enumerate(zip(p.ravel(), t.ravel())):
                ph = self.eos.phase_props(self.eos.z, float(pp), float(tt))
                rho.flat[k], mu.flat[k] = ph.density, ph.viscosity
        bg = self.lb_per_scf / rho
        if np.ndim(p) == 0:
            return float(bg), float(rho), float(mu)
        return bg, rho, mu

    def z(self, p, t):
        if self.table is not None:
            return self.table("z", p, t)
        return np.array([self.eos.phase_props(self.eos.z, float(pp), float(tt)).z
                         for pp, tt in zip(np.atleast_1d(p), np.broadcast_to(np.atleast_1d(t), np.shape(np.atleast_1d(p))))])

    def vol(self, p, t):
        bg, rho, mu = self.props(p, t)
        return bg * 1e6 / 86400.0, rho, mu

    def mass_lbh(self, q):
        return np.asarray(q, float) * 1e6 * self.lb_per_scf / 24.0

    def enthalpy(self, p, t):
        return self.table("h", p, t)

    def cp(self, p, t):
        return self.table("cp", p, t)

    def temperature(self, p, h, t_guess):
        return self.table.temperature(p, h, t_guess)

    def mu_z(self, p, t_f):
        """(viscosity, Z) arrays at temperature t_f - for the pseudo-pressure of the injectivity."""
        p = np.atleast_1d(np.asarray(p, float))
        _, _, mu = self.props(p, np.full(p.shape, float(t_f)))
        return np.asarray(mu, float), np.asarray(self.z(p, np.full(p.shape, float(t_f))), float)
