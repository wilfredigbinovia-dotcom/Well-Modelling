"""Choke performance: rate, pressure drop or choke size for multiphase flow through a choke.

Models
- ELF: Perkins, T.K. (1993). Critical and Subcritical Flow of Multiphase Mixtures Through Chokes. SPE Drilling &
  Completion 8(4), 271-276 (SPE 20633). Isentropic (adiabatic, frictionless) frozen flow from the general energy
  equation; critical throat pressure ratio from the maximum of Eq. A-28 (equivalent to Eq. A-30); subcritical
  throat pressure from the recovered downstream pressure with Eq. 7; actual rate = K x isentropic rate (Eq. C-4),
  K = 0.826 by default (best value over 1,432 data sets).
- Sachdeva et al. (1986), as used for surface chokes elsewhere in the tool (wellmodel.surface).

Fluid properties come from the tool's PVT (tab 2) rather than Perkins' own correlations (Lasater, Vennix &
Kobayashi), so the choke uses the same fluid as the rest of the model. Phase mass fractions are frozen at the
upstream conditions, as in the paper.

Implementation notes
- Oil heat capacity: Eq. 1 (Kern) as printed has 0.0051 for the temperature coefficient, which gives about
  0.95 Btu/lb/F at 100 F for a 30 API oil; Kern's correlation gives ~0.49. The coefficient is taken as 0.00051.
- Ratio of specific heats F = Cp/Cv of the gas: Perkins interpolated Johnson's methane tables (Fig. 1); here the
  ideal-gas value from a natural-gas heat capacity correlation is used unless F is entered.
- Outlet temperature: Eq. C-1 (polytropic path of the frozen mixture) evaluated at the downstream pressure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from . import blackoil as bo

R_GAS = 1545.35        # ft-lbf / (lbmol R)
GC = 32.174
J = 778.17             # ft-lbf / Btu
K_PERKINS = 0.826
MODELS = ["ELF (Perkins 1993)", "Sachdeva (1986)"]


def gas_heat_capacity_ratio(mw, t_f):
    """Ideal-gas Cp/Cv of a natural gas: Cp ~ 8.5 Btu/lbmol/R for methane at 60 F, rising with molecular weight
    (ethane ~12.6) and temperature."""
    cp = 8.5 + 0.29 * (mw - 16.04) + 0.006 * (t_f - 60.0)
    return cp / (cp - 1.986)


def oil_cv(api, t_f):
    """Eq. 1 (Kern), ft-lbf/lbm/F - with 0.00051 for the printed 0.0051 (see module notes)."""
    return J * ((0.355 + 0.00176 * api) + (0.00051 + 1.167e-5 * api) * t_f)


@dataclass
class ChokeSetup:
    fluid: object
    kind: str                 # "oil" | "gas"
    ratios: dict              # oil: wc (fraction), gor; gas: wgr, cgr
    hc: object                # HeatCapacities (Sachdeva)
    d_up: float = 2.992       # in, pipe upstream of the choke
    d_down: float = 2.992     # in, pipe downstream
    cd: float = K_PERKINS
    model: str = MODELS[0]
    f_ratio: float | None = None

    def case(self, q):
        return {"q": np.atleast_1d(np.asarray(q, float)), **{k: np.atleast_1d(float(v)) for k, v in self.ratios.items()}}

    def mass_per_rate(self):
        """lbm/s per unit rate (STB/d liquid for oil, MMscf/d for gas)."""
        w, _ = self.fluid.mass_cp(self.case(1.0), self.hc)
        return float(np.atleast_1d(w)[0]) / 3600.0

    # ------------------------------------------------------------- Perkins basis
    def _basis(self, p1, t1):
        s = self.fluid.state(np.array([p1]), np.array([t1]), self.case(1.0))
        ql, qg = float(s["ql"][0]), float(s["qg"][0])
        rl, rg = float(s["rho_l"][0]), max(float(s["rho_g"][0]), 1e-6)
        ml, mg = ql * rl, qg * rg
        fg = mg / max(ml + mg, 1e-30)
        fl = 1 - fg
        inp = getattr(self.fluid, "inp", None)
        if self.kind == "oil":
            wc = float(np.clip(self.ratios.get("wc", 0.0), 0, 1))
            api = float(getattr(inp, "api", 35.0)) if inp is not None else 35.0
            go = 141.5 / (131.5 + api)
            m_w = wc * 62.4 * 1.02
            m_o = (1 - wc) * 62.4 * go
            fw_l = m_w / max(m_w + m_o, 1e-30)
        else:
            api = float(getattr(self.fluid, "api", 50.0))
            fw_l = 1.0 if self.ratios.get("cgr", 0.0) <= 0 else 0.5
        sg = float(getattr(inp, "gas_sg", 0.65)) if inp is not None else 0.65
        if self.kind == "gas" and hasattr(self.fluid, "wellstream_sg"):
            sg = float(np.atleast_1d(self.fluid.wellstream_sg(np.array([self.ratios.get("cgr", 0.0)])))[0])
        mw = 28.97 * sg
        v1 = 1.0 / rg                                            # gas specific volume, ft3/lbm
        z = 144.0 * p1 * mw * v1 / (R_GAS * (t1 + 460.0))
        return dict(fg=fg, fo=fl * (1 - fw_l), fw=fl * fw_l, rho_l=rl, v1=v1, z=z, mw=mw, api=api, sg=sg)

    def _n_lambda(self, B, t_f):
        F = self.f_ratio or gas_heat_capacity_ratio(B["mw"], t_f)
        cvg = B["z"] * R_GAS / (B["mw"] * (F - 1))               # Eq. A-17
        cvo, cvw = oil_cv(B["api"], t_f), J
        heat = B["fg"] * cvg + B["fo"] * cvo + B["fw"] * cvw
        n = (B["fg"] * F * cvg + B["fo"] * cvo + B["fw"] * cvw) / max(heat, 1e-12)      # Eq. A-19
        lam = B["fg"] + heat * B["mw"] / (B["z"] * R_GAS)                                # Eq. A-9
        return n, lam, F, cvg, cvo

    def _wi(self, pr, p1, B, n, lam, d_choke):
        """Isentropic mass rate (lbm/s), Eq. A-28, for throat/upstream pressure ratio pr (array)."""
        pr = np.clip(np.atleast_1d(np.asarray(pr, float)), 1e-6, 1.0)
        a2 = np.pi / 4 * (d_choke / 12) ** 2
        ratio = (d_choke / self.d_up) ** 2
        a1 = (B["fo"] + B["fw"]) / (B["rho_l"] * B["v1"])         # alpha_1 = (fo/rho_o + fw/rho_w) / v1
        fg = B["fg"]
        num = lam * (1 - pr ** ((n - 1) / n)) + a1 * (1 - pr)
        term = fg * pr ** (-1 / n) + a1
        den = (1 - ratio ** 2 * ((fg + a1) / term) ** 2) * term ** 2
        return a2 * np.sqrt(np.maximum(288.0 * GC * p1 / B["v1"] * num / np.maximum(den, 1e-30), 0.0))

    def _critical(self, p1, t1, d_choke):
        """Critical pressure ratio (maximum of Eq. A-28, iterating n at the mean throat conditions per Appendix
        C), the n and lambda used, and the basis."""
        B = self._basis(p1, t1)
        t_avg = t1
        pr_c = 0.0
        for _ in range(4):
            n, lam, *_ = self._n_lambda(B, t_avg)
            if B["fg"] < 1e-7:                               # liquid only: no critical flow
                pr_c = 0.0
                break
            res = minimize_scalar(lambda x: -self._wi(x, p1, B, n, lam, d_choke)[0], bounds=(1e-3, 0.999),
                                  method="bounded", options={"xatol": 1e-6})
            pr_c = float(res.x)
            t2 = (t1 + 460) * pr_c ** ((n - 1) / n) - 460                   # Eq. C-1
            t_new = 0.5 * (t1 + t2)                                          # Eq. C-3
            if abs(t_new - t_avg) < 0.05:
                break
            t_avg = t_new
        return pr_c, n, lam, B

    # ------------------------------------------------------------- public
    def mass_rate(self, p1, t1, p4, d_choke):
        """Actual mass rate (lbm/s) and details for upstream p1 (psia), T1 (F), recovered downstream p4 (psia)
        and choke bean d_choke (in)."""
        if self.model.startswith("Sachdeva"):
            return self._sachdeva(p1, t1, p4, d_choke)
        pr_c, n, lam, B = self._critical(p1, t1, d_choke)
        p2c = pr_c * p1
        beta = min(d_choke / self.d_down, 0.999)
        p3 = p1 - (p1 - p4) / (1 - beta ** 1.85)                             # Eq. 7
        critical = p3 <= p2c
        pr = pr_c if critical else max(p3 / p1, 1e-6)
        wi = float(self._wi(pr, p1, B, n, lam, d_choke)[0])
        wc_ = float(self._wi(pr_c, p1, B, n, lam, d_choke)[0]) if pr_c > 0 else np.nan
        t2 = (t1 + 460) * pr ** ((n - 1) / n) - 460
        t2c = (t1 + 460) * pr_c ** ((n - 1) / n) - 460 if pr_c > 0 else np.nan
        # downstream pressure that just gives critical flow: p3 = p2c
        p4_boundary = p1 - (p1 - p2c) * (1 - beta ** 1.85) if pr_c > 0 else np.nan
        return dict(w=self.cd * wi, regime="Critical" if critical else "Subcritical", p_throat=pr * p1,
                    p_crit=p2c, t_throat=t2, t_crit=t2c, w_crit=self.cd * wc_ if pr_c > 0 else np.nan,
                    p4_boundary=p4_boundary, n=n, fg=B["fg"], t_out=self.outlet_temperature(p1, t1, p4, B))

    def _sachdeva(self, p1, t1, p4, d_choke):
        from .surface import _sachdeva_flux
        case = self.case(1.0)
        area = np.pi / 4 * (d_choke / 12) ** 2
        flux, yc, crit = _sachdeva_flux(self.fluid, case, np.array([p1]), np.array([p4]), np.array([t1]), self.hc,
                                        self.cd)
        fcrit, *_ = _sachdeva_flux(self.fluid, case, np.array([p1]), np.array([1e-3 * p1]), np.array([t1]), self.hc,
                                   self.cd)
        yc = float(yc[0])
        B = self._basis(p1, t1)
        n, *_ = self._n_lambda(B, t1)
        t2c = (t1 + 460) * yc ** ((n - 1) / n) - 460 if yc > 0 else np.nan
        return dict(w=float(flux[0]) * area, regime="Critical" if bool(crit[0]) else "Subcritical",
                    p_throat=max(p4, yc * p1), p_crit=yc * p1, t_throat=np.nan, t_crit=t2c,
                    w_crit=float(fcrit[0]) * area if yc > 0 else np.nan, p4_boundary=yc * p1 if yc > 0 else np.nan,
                    n=n, fg=B["fg"], t_out=self.outlet_temperature(p1, t1, p4, B))

    def outlet_temperature(self, p1, t1, p4, B=None):
        """Outlet temperature on the polytropic path of the frozen mixture, Eq. C-1 applied at the downstream
        pressure: (T + 460) = (T1 + 460) (p4/p1)^((n-1)/n) - the throat temperature relation of the paper,
        which is also how PROSPER reports the choke outlet temperature."""
        B = B or self._basis(p1, t1)
        if p4 >= p1:
            return t1
        n, *_ = self._n_lambda(B, t1)
        t = t1
        for _ in range(3):                              # n at the mean temperature
            t = (t1 + 460) * (p4 / p1) ** ((n - 1) / n) - 460
            n, *_ = self._n_lambda(B, 0.5 * (t1 + t))
        return (t1 + 460) * (p4 / p1) ** ((n - 1) / n) - 460

    # rate-based wrappers --------------------------------------------------------
    def rate(self, p1, t1, p4, d_choke):
        r = self.mass_rate(p1, t1, p4, d_choke)
        m = self.mass_per_rate()
        r["q"] = r["w"] / m
        r["q_crit"] = r["w_crit"] / m if np.isfinite(r["w_crit"]) else np.nan
        return r

    def downstream_pressure(self, p1, t1, q, d_choke):
        """Recovered downstream pressure for rate q. Returns (p4, result); p4 = nan if q exceeds the critical rate
        (the choke cannot pass it at this upstream pressure). At exactly critical flow p4 is the highest downstream
        pressure that still gives critical flow."""
        hi = self.rate(p1, t1, 1e-3, d_choke)
        q_max = hi["q"]
        if q > q_max * (1 + 1e-6):
            return np.nan, hi
        f = lambda p4: self.rate(p1, t1, p4, d_choke)["q"] - q
        lo_p = hi["p4_boundary"] if np.isfinite(hi["p4_boundary"]) else 1e-3
        if f(lo_p) <= 0:
            return lo_p, self.rate(p1, t1, lo_p, d_choke)
        p4 = brentq(f, lo_p, p1 * (1 - 1e-9), xtol=1e-4)
        return p4, self.rate(p1, t1, p4, d_choke)

    def choke_size(self, p1, t1, p4, q, d_max=None):
        """Bean ID (in) that passes rate q from p1 to p4."""
        d_max = d_max or 0.98 * min(self.d_up, self.d_down)
        f = lambda d: self.rate(p1, t1, p4, d)["q"] - q
        lo = 1.0 / 64
        if f(lo) > 0:
            return lo, self.rate(p1, t1, p4, lo)
        if f(d_max) < 0:
            return np.nan, self.rate(p1, t1, p4, d_max)
        d = brentq(f, lo, d_max, xtol=1e-5)
        return d, self.rate(p1, t1, p4, d)

    def performance_curve(self, p1, t1, d_choke, n=40):
        """Rate and outlet temperature against downstream pressure (0 to p1)."""
        ps = np.linspace(14.7, p1 * 0.999, n)
        rows = [self.rate(p1, t1, p, d_choke) for p in ps]
        return ps, np.array([r["q"] for r in rows]), np.array([r["t_out"] for r in rows])
