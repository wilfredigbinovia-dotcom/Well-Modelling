"""Black-oil PVT correlations (oilfield units, fully vectorised).

Units: pressure psia, temperature degF, GOR scf/STB, densities lb/ft3,
viscosity cp, surface tension dyne/cm, Bo/Bw rb/STB, Bg ft3/scf.

Correlations
------------
Bubble point / Rs .... Standing (1947), optionally calibrated to a measured Pb
Bo (saturated) ....... Standing
Bo (undersaturated) .. Vasquez-Beggs compressibility (integrated form)
Oil viscosity ........ Beggs-Robinson (dead/live), Vasquez-Beggs (undersaturated)
Gas Z-factor ......... Dranchuk-Abou-Kassem with Sutton pseudo-criticals
Gas viscosity ........ Lee-Gonzalez-Eakin
Water ................ McCain Bw, Brill-Beggs viscosity
Surface tension ...... Baker-Swerdloff (oil), linear temperature fit (water)
"""
from __future__ import annotations

import numpy as np

RANKINE = 459.67
P_STD = 14.696


class BlackOilPVT:
    """Black-oil fluid description.

    Parameters
    ----------
    api : stock-tank oil gravity, degAPI
    gas_sg : separator gas specific gravity (air = 1)
    rsb : solution GOR at the bubble point, scf/STB
    t_res : reservoir temperature, degF
    water_sg : formation water specific gravity
    pb : optional measured bubble point at t_res, psia. When given, the
        Standing correlation is shifted (pressure scaling) so Rs(pb) = rsb.
    """

    def __init__(self, api: float, gas_sg: float, rsb: float, t_res: float,
                 water_sg: float = 1.07, pb: float | None = None):
        self.api = float(api)
        self.gamma_o = 141.5 / (131.5 + self.api)
        self.gas_sg = float(gas_sg)
        self.rsb = float(rsb)
        self.t_res = float(t_res)
        self.water_sg = float(water_sg)
        pb_std = self._standing_pb(self.rsb, self.t_res)
        self.pb = float(pb) if pb is not None else float(pb_std)
        self._p_scale = pb_std / self.pb

    # ------------------------------------------------------------------ oil
    def _standing_pb(self, rs, t):
        rs = np.maximum(np.asarray(rs, float), 1e-6)
        a = 10 ** (0.00091 * t - 0.0125 * self.api)
        return 18.2 * ((rs / self.gas_sg) ** 0.83 * a - 1.4)

    def bubble_point(self, rs=None, t=None):
        """Saturation pressure (psia) of oil holding `rs` scf/STB at `t` degF."""
        rs = self.rsb if rs is None else rs
        t = self.t_res if t is None else t
        return np.maximum(self._standing_pb(rs, t) / self._p_scale, P_STD)

    def _rs_saturated(self, p, t):
        pe = np.asarray(p, float) * self._p_scale
        a = 10 ** (0.0125 * self.api - 0.00091 * np.asarray(t, float))
        rs = self.gas_sg * ((pe / 18.2 + 1.4) * a) ** (1 / 0.83)
        return np.maximum(rs, 0.0)

    def solution_gor(self, p, t, gor=None):
        """Rs (scf/STB): limited by Rsb and by the total gas available (GOR)."""
        cap = self.rsb if gor is None else np.minimum(gor, self.rsb)
        return np.minimum(self._rs_saturated(p, t), cap)

    def _bo_sat(self, rs, t):
        x = rs * np.sqrt(self.gas_sg / self.gamma_o) + 1.25 * t
        return 0.9759 + 0.00012 * x ** 1.2

    def _vb_exponent(self, rs, t):
        # Vasquez-Beggs: co = A / p  ->  Bo = Bob * (p/pb)^-A
        return (-1433 + 5 * rs + 17.2 * t - 1180 * self.gas_sg + 12.61 * self.api) / 1e5

    def _dead_oil_visc(self, t):
        x = 10 ** (3.0324 - 0.02023 * self.api) * np.maximum(t, 1.0) ** -1.163
        return 10 ** x - 1

    def oil(self, p, t, gor=None):
        """Return (rs, bo, rho_o, mu_o) arrays."""
        p = np.maximum(np.asarray(p, float), P_STD)
        t = np.asarray(t, float)
        cap = self.rsb if gor is None else np.minimum(gor, self.rsb)
        cap = np.maximum(cap, 0.0)
        rs_sat = self._rs_saturated(p, t)
        under = rs_sat >= cap
        rs = np.where(under, cap, rs_sat)
        pb_loc = np.maximum(self._standing_pb(np.maximum(cap, 1e-6), t) / self._p_scale, P_STD)
        bob = self._bo_sat(rs, t)
        A = self._vb_exponent(rs, t)
        ratio = np.maximum(p / pb_loc, 1.0)
        bo = np.where(under, bob * ratio ** (-A), bob)
        rho_o = (62.428 * self.gamma_o + 0.0136 * rs * self.gas_sg) / bo
        mu_od = self._dead_oil_visc(t)
        a_ = 10.715 * (rs + 100) ** -0.515
        b_ = 5.44 * (rs + 150) ** -0.338
        mu_ob = a_ * mu_od ** b_
        m = 2.6 * p ** 1.187 * np.exp(-11.513 - 8.98e-5 * p)
        mu_o = np.where(under, mu_ob * ratio ** m, mu_ob)
        return rs, bo, rho_o, mu_o

    # ------------------------------------------------------------------ gas
    def _pseudo_critical(self):
        g = self.gas_sg
        return 169.2 + 349.5 * g - 74.0 * g * g, 756.8 - 131.07 * g - 3.6 * g * g

    def z_factor(self, p, t, iters: int = 10):
        """Dranchuk-Abou-Kassem Z-factor (Newton on reduced density)."""
        tpc, ppc = self._pseudo_critical()
        tr = (np.asarray(t, float) + RANKINE) / tpc
        pr = np.maximum(np.asarray(p, float), P_STD) / ppc
        A = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475,
             -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)
        c1 = A[0] + A[1] / tr + A[2] / tr ** 3 + A[3] / tr ** 4 + A[4] / tr ** 5
        c2 = A[5] + A[6] / tr + A[7] / tr ** 2
        c3 = A[8] * (A[6] / tr + A[7] / tr ** 2)
        c4 = A[9] / tr ** 3
        a11 = A[10]
        rho = 0.27 * pr / tr
        for _ in range(iters):
            r2 = rho * rho
            e = np.exp(-a11 * r2)
            z = 1 + c1 * rho + c2 * r2 - c3 * r2 * r2 * rho + c4 * (r2 + a11 * r2 * r2) * e
            f = z - 0.27 * pr / (rho * tr)
            dz = (c1 + 2 * c2 * rho - 5 * c3 * r2 * r2
                  + c4 * e * (2 * rho + 2 * a11 * r2 * rho - 2 * a11 * a11 * r2 * r2 * rho))
            df = dz + 0.27 * pr / (r2 * tr)
            rho = np.clip(rho - f / df, 1e-8, 3.0)
        return np.clip(0.27 * pr / (rho * tr), 0.2, 2.0)

    def gas(self, p, t):
        """Return (z, bg [ft3/scf], rho_g [lb/ft3], mu_g [cp])."""
        p = np.maximum(np.asarray(p, float), P_STD)
        tR = np.asarray(t, float) + RANKINE
        z = self.z_factor(p, t)
        bg = 0.02827 * z * tR / p
        rho_g = 28.967 * self.gas_sg * p / (z * 10.7316 * tR)
        M = 28.967 * self.gas_sg
        K = (9.4 + 0.02 * M) * tR ** 1.5 / (209 + 19 * M + tR)
        X = 3.5 + 986 / tR + 0.01 * M
        Y = 2.4 - 0.2 * X
        mu_g = 1e-4 * K * np.exp(X * (rho_g / 62.428) ** Y)
        return z, bg, rho_g, mu_g

    # ---------------------------------------------------------------- water
    def water(self, p, t):
        """Return (bw, rho_w, mu_w)."""
        p = np.asarray(p, float)
        t = np.asarray(t, float)
        dvt = -1.0001e-2 + 1.33391e-4 * t + 5.50654e-7 * t * t
        dvp = (-1.95301e-9 * p * t - 1.72834e-13 * p * p * t
               - 3.58922e-7 * p - 2.25341e-10 * p * p)
        bw = (1 + dvt) * (1 + dvp)
        rho_w = 62.428 * self.water_sg / bw
        mu_w = np.exp(1.003 - 1.479e-2 * t + 1.982e-5 * t * t)
        return bw, rho_w, mu_w

    # ------------------------------------------------------- surface tension
    def surface_tension(self, p, t):
        """Return (sigma_oil_gas, sigma_water_gas) in dyne/cm."""
        p = np.asarray(p, float)
        t = np.asarray(t, float)
        s68 = 39.0 - 0.2571 * self.api
        s100 = 37.5 - 0.2571 * self.api
        frac = np.clip((t - 68.0) / 32.0, 0.0, 1.0)
        sod = s68 + (s100 - s68) * frac
        so = np.maximum(sod * (1 - 0.024 * np.maximum(p, 0) ** 0.45), 1.0)
        sw = np.clip(70.0 - (t - 74.0) * (17.0 / 206.0), 35.0, 75.0)
        return so, sw

    # --------------------------------------------------------- convenience
    def total_compressibility_oil(self, p, dp: float = 5.0):
        """Effective oil compressibility (1/psi) at reservoir temperature,
        including gas liberation below Pb: co = -1/Bo dBo/dp + Bg/Bo dRs/dp."""
        p = np.asarray(p, float)
        t = self.t_res
        rs1, bo1, _, _ = self.oil(p - dp, t)
        rs2, bo2, _, _ = self.oil(p + dp, t)
        _, bo, _, _ = self.oil(p, t)
        _, bg, _, _ = self.gas(p, t)
        dbo = (bo2 - bo1) / (2 * dp)
        drs = (rs2 - rs1) / (2 * dp)
        return -dbo / bo + (bg / 5.615) * drs / bo

    def summary(self) -> dict:
        rs, bo, rho, mu = self.oil(self.pb, self.t_res)
        return {"API": self.api, "gas_sg": self.gas_sg, "Rsb": self.rsb,
                "T_res": self.t_res, "Pb": round(self.pb, 1),
                "Bob": round(float(bo), 4), "mu_ob": round(float(mu), 3),
                "rho_ob": round(float(rho), 2)}
