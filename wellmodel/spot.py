"""SPOT inflow model - PROSPER's Shell Perforation Optimisation Tool IPR (Tutorial 09).

Reconstructed from the user's T09 file (the Logie well: one 110 ft MD layer, 9275-9385 ft MD / 9000-9100 ft TVD,
50 md, Pr 4000 psig, Baker 2007 Predator XP LS 6 spf / 60 deg wireline gun, crushed zone 0.1 x k over 1.2 in) and
14 one-at-a-time runs (UCS 4000 / 12000, crushed-zone factor and thickness, kv/kh 1, k 200 md, underbalance 500 psi,
overburden gradient 0.8 psi/ft, layer pressure 3000 psig, perforation efficiency 1, vertical well, 63 deg well,
Vogel). The layer is split into 1 ft MD segments (top to bottom inclusive). What was found:

Per segment (exact on all runs unless stated):
  penetration Lp (in)  = 1.85055 (Lp_norm / 20.1883) ln(32357 / UCS) ln(26990 / (sigma_v - 1.0256 Pr))
                         sigma_v = overburden gradient x TVD, Pr the layer pressure (psig) - fitted: within 0.04 %
                         of the Lp behind PROSPER's skins on 12 depth / strength / stress points; the underbalance does
                         not enter. Scaling with the gun's normalised Lp is assumed (one gun was available).
  entry hole           = 0.97509 x the gun's EHD (PROSPER: 0.22427 in for 0.23; casing and standoff effects not
                         separated - one gun / casing).
  perforation skin     = Karakas & Tariq with spf x perforation efficiency / cos(deviation) shots per ft of
                         TVD, the phasing, kv/kh, rw and rp = EHD / 2.
  crushed-zone skin    = McLeod: (1/f - 1) ln((rp + t) / rp) / (n Lp), f the crushed-zone permeability factor,
                         t its thickness, n the same shots per ft.
  deviation skin       = Cinco-Ley (anisotropic) with h = number of segments x cos(deviation) (100.9 ft here).
  completion fluid density = (Pr - underbalance) / (0.051948 TVD) lb/gal; near-wellbore effective vertical stress
                         = sigma_v - (Pr - underbalance) - 14.7; underbalance pressures: King 3099.6 k^-0.370 and
                         Behrmann 46235 phi k^-0.3636 psi below Pr (from the 50 / 200 md and porosity 0.18 / 0.25 runs), user
                         Pr - underbalance. Porosity changes nothing else in PROSPER's SPOT results.
Layer:
  total skin = cos^0.965 (cos^2 + sin^2 kv/kh)^0.4946 (1.002 Sperf + 0.9866 Scrush) + Sdev  (layer means;
               fitted - within 0.015 on 11 runs, totals 0.7 to 15),
  WIQI = A / (A + S), A = ln(re / rw) - 0.75.
  Water cut entered: the water saturation that gives it (Corey relative permeabilities, mobilities at Pr) sets
  kro and krw (exact), ko = k kro.
  IPR: oil q = 0.00708 ko h / (A + S) [ int(Pwf..Pr) dp / (mu_o B_o) above Pb; below it Fetkovich
  (Pb^2 - Pwf^2) / (2 Pb) / (mu_o B_o)_Pb or Vogel ], liquid = oil / (1 - water cut); h the layer's TVD thickness.
  PROSPER's layer table: oil AOF on n / (n - 1) of that thickness (the segment count) - within 0.02 % (Vogel 0.12 %).

Not identified: PROSPER's layer "PI" column (2 decimals; not the Darcy oil PI), the Min Lp column, the thick-wall-
cylinder columns (from the log; not used), the IPR temperature column.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from . import completion as wcomp

LP_C, LP_U0, LP_S0, LP_ALPHA, LP_NORM_REF = 1.85055313, 32357.0, 26989.6, 1.0256, 20.1883
EHD_FACTOR = 0.22427 / 0.23
AGG = (1.002, 0.9866, 0.965, 0.9891)         # layer total skin: perf, crush, cos exponent, anisotropy exponent
KING = (3099.6, 0.36997)
BEHRMANN = (46234.67, 0.36362)      # dP = C phi k^-b
G = 14.696


@dataclass
class SpotGun:
    name: str = "Baker - 2007 Predator XP LS (6 spf / 60 deg / wireline)"
    gun_od: float = 2.0             # in
    spf: float = 6.0
    phasing: float = 60.0
    lp_api: float = 19.2            # section 1 penetration, in
    lp_norm: float = 20.1883        # normalised penetration, in
    ehd: float = 0.23               # in
    cement_strength: float = 5658.3


GUNS = {SpotGun().name: SpotGun()}


@dataclass
class SpotInputs:
    top_md: float = 9275.0
    bottom_md: float = 9385.0
    survey: tuple = ((9275.0, 9000.0), (9385.0, 9100.0))     # (MD, TVD) ft
    deviated: bool = True
    inflow: str = "Fetkovich"        # "Fetkovich" | "Vogel"
    rw: float = 0.354                # ft
    re: float = 4200.0               # drainage radius, ft
    perf_eff: float = 0.9
    pr: float = 4000.0 + G           # psia
    underbalance: float = 100.0      # psi
    ob_grad: float = 1.0             # psi/ft
    wc: float = 0.2                  # fraction
    kv_kh: float = 0.1
    k: float = 50.0                  # md
    porosity: float = 0.18
    ucs: float = 8000.0              # psi
    crush_factor: float = 0.1
    crush_thickness: float = 1.2     # in
    gun: SpotGun = field(default_factory=SpotGun)
    relperm: tuple = (0.15, 0.2, 0.8, 0.6, 1.5, 1.5)   # Sor, Swr, kro end, krw end, n oil, n water


def corey(sw, rp):
    sor, swr, ko_e, kw_e, no, nw = rp
    den = 1 - sor - swr
    so_n = np.clip((1 - sw - sor) / den, 0, 1)
    sw_n = np.clip((sw - swr) / den, 0, 1)
    return ko_e * so_n ** no, kw_e * sw_n ** nw


def penetration(gun: SpotGun, ucs, sigma_v, p_psig):
    s = sigma_v - LP_ALPHA * p_psig
    lp = LP_C * gun.lp_norm / LP_NORM_REF * np.log(LP_U0 / ucs) * np.log(LP_S0 / np.maximum(s, 1.0))
    return np.maximum(lp, 0.01)


class Spot:
    def __init__(self, f: SpotInputs, mob_fn):
        """mob_fn(p psia array) -> (1/(mu_o Bo), 1/(mu_w Bw), Pb psia)."""
        self.f = f
        self.mob_fn = mob_fn
        md_s, tvd_s = np.array(f.survey, float).T
        n = int(round(f.bottom_md - f.top_md)) + 1
        self.md = f.top_md + np.arange(n)
        self.tvd = np.interp(self.md, md_s, tvd_s, left=None, right=None)
        if len(md_s) >= 2:
            dt = np.interp(f.bottom_md, md_s, tvd_s) - np.interp(f.top_md, md_s, tvd_s)
            self.cos = float(np.clip(dt / max(f.bottom_md - f.top_md, 1e-9), 1e-6, 1.0)) if f.deviated else 1.0
        else:
            self.cos = 1.0
        self.h_tvd = (f.bottom_md - f.top_md) * self.cos
        self.A = np.log(f.re / f.rw) - 0.75

    def log(self):
        f, gun = self.f, self.f.gun
        p_g = f.pr - G
        sv = f.ob_grad * self.tvd
        lp = penetration(gun, f.ucs, sv, p_g)
        ehd = np.full_like(lp, gun.ehd * EHD_FACTOR)
        rp_in = ehd / 2
        n_eff = gun.spf * f.perf_eff / self.cos
        sp = np.array([wcomp.kt_perforation_skin(f.rw, L / 12, r / 12, n_eff, int(gun.phasing), f.kv_kh)["perforation"]
                       for L, r in zip(lp, rp_in)])
        sc = (1 / f.crush_factor - 1) * np.log((rp_in + f.crush_thickness) / rp_in) / (n_eff * lp / 12)
        sdev = self.deviation_skin()
        dk = KING[0] * f.k ** -KING[1]
        db = BEHRMANN[0] * f.porosity * f.k ** -BEHRMANN[1]
        import pandas as pd
        return pd.DataFrame({
            "Measured depth (ft)": self.md, "True vertical depth (ft)": self.tvd,
            "Penetration Lp (in)": lp, "Entry hole diameter (in)": ehd,
            "Reservoir pressure (psig)": p_g, "Underbalance Behrmann (psig)": p_g - db,
            "Underbalance King (psig)": p_g - dk, "Underbalance user (psig)": p_g - f.underbalance - G,
            "Completion fluid density (lb/gal)": (p_g - f.underbalance) / (0.0519481 * self.tvd),
            "Perforation skin": sp, "Damage skin": 0.0, "Damage, perf, crush skin": sp + sc, "Crush skin": sc,
            "Deviation skin": sdev,
            "Near-wellbore effective vertical stress (psig)": sv - (p_g - f.underbalance) - G})

    def deviation_skin(self):
        f = self.f
        if not f.deviated or self.cos >= 1 - 1e-9:
            return 0.0
        th = np.degrees(np.arccos(self.cos))
        h = len(self.md) * self.cos
        return wcomp.deviation_skin(min(th, 75.0), h, f.rw, f.kv_kh, True)

    def total_skin(self, lg=None):
        f = self.f
        lg = self.log() if lg is None else lg
        c = self.cos
        s2 = 1 - c * c
        F = c ** AGG[2] * np.sqrt(c * c + s2 * f.kv_kh) ** AGG[3]
        return float(F * (AGG[0] * lg["Perforation skin"].mean() + AGG[1] * lg["Crush skin"].mean())
                     + self.deviation_skin())

    def saturation(self):
        f = self.f
        mo, mw, _ = self.mob_fn(np.array([f.pr]))
        if f.wc <= 0:
            sw = f.relperm[1]
        else:
            target = f.wc / max(1 - f.wc, 1e-12)
            g = lambda s: (corey(s, f.relperm)[1] * mw[0]) / max(corey(s, f.relperm)[0] * mo[0], 1e-30) - target
            lo, hi = f.relperm[1] + 1e-9, 1 - f.relperm[0] - 1e-9
            sw = brentq(g, lo, hi) if g(lo) < 0 < g(hi) else hi
        kro, krw = corey(sw, f.relperm)
        return float(sw), float(kro), float(krw)

    def oil_rate_fn(self, s_tot, h=None):
        f = self.f
        _, kro, _ = self.saturation()
        h = self.h_tvd if h is None else h
        C = 0.00708 * f.k * kro * h / (self.A + s_tot)
        pr = f.pr
        grid = np.linspace(14.7, pr, 400)
        mo, _, pb = self.mob_fn(grid)
        pb = float(pb)
        # cumulative integral of the oil mobility from each pressure up to Pr
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (mo[1:] + mo[:-1]) * np.diff(grid))])
        I_to_pr = lambda p: cum[-1] - np.interp(p, grid, cum)
        if pr <= pb:
            m_r = float(np.interp(pr, grid, mo))
            if f.inflow == "Vogel":
                return lambda p: C * m_r * pr / 1.8 * (1 - 0.2 * np.clip(p, 0, pr) / pr - 0.8 * (np.clip(p, 0, pr) / pr) ** 2)
            return lambda p: C * m_r * (pr ** 2 - np.clip(p, 0, pr) ** 2) / (2 * pr)
        qb = C * float(I_to_pr(pb))
        m_b = float(np.interp(pb, grid, mo))
        j_avg = qb / (pr - pb)

        def q(p):
            p = np.clip(np.asarray(p, float), 0, pr)
            above = C * I_to_pr(np.maximum(p, pb))
            if f.inflow == "Vogel":
                below = qb + j_avg * pb / 1.8 * (1 - 0.2 * p / pb - 0.8 * (p / pb) ** 2)
            else:
                below = qb + C * m_b * (pb ** 2 - p ** 2) / (2 * pb)
            return np.where(p >= pb, above, below)
        return q

    def results(self, pwf_log=None):
        f = self.f
        lg = self.log()
        s = self.total_skin(lg)
        sw, kro, krw = self.saturation()
        qo = self.oil_rate_fn(s)
        n = len(self.md)
        aof_layer = float(qo(0.0)) * n / max(n - 1, 1)
        layer = {"Top MD (ft)": f.top_md, "Bottom MD (ft)": f.bottom_md,
                 "Completion fluid density (lb/gal)": (f.pr - G - f.underbalance) /
                 (0.0519481 * float(np.mean([self.tvd[0], self.tvd[-1]]))),
                 "Total skin": s, "Deviation skin": self.deviation_skin(), "Partial penetration skin": 0.0,
                 "WIQI": self.A / (self.A + s), "Oil AOF (STB/d)": aof_layer,
                 "Water saturation": sw, "kro": kro, "krw": krw, "ko (md)": f.k * kro, "kw (md)": f.k * krw,
                 "Mean penetration Lp (in)": float(lg["Penetration Lp (in)"].mean()),
                 "TVD thickness (ft)": self.h_tvd}
        if pwf_log is not None:
            # each segment's share of the layer rate at this FBHP
            c = self.cos
            F = c ** AGG[2] * np.sqrt(c * c + (1 - c * c) * f.kv_kh) ** AGG[3]
            w = 1 / (self.A + F * (AGG[0] * lg["Perforation skin"] + AGG[1] * lg["Crush skin"])
                     + self.deviation_skin())
            q_tot = float(qo(pwf_log)) * n / max(n - 1, 1)
            lg["Qo at FBHP (STB/d)"] = q_tot * w / w.sum()
            lg["Qw at FBHP (STB/d)"] = lg["Qo at FBHP (STB/d)"] * f.wc / max(1 - f.wc, 1e-12)
            lg["Qo AOF (STB/d)"] = aof_layer * w / w.sum()
        return {"layer": layer, "log": lg, "oil_rate": qo, "skin": s}


def blackoil_mobility(model, t_f):
    """mob_fn for Spot from a BlackOilModel at temperature t_f (F)."""
    pb = float(model.bubble_point(t_f))

    def fn(p):
        p = np.atleast_1d(np.asarray(p, float))
        tab = model.table(t_f, list(np.maximum(p, 14.7)))
        mo = 1 / (tab["Oil viscosity (cp)"].to_numpy() * tab["Oil FVF (RB/STB)"].to_numpy())
        mw = 1 / (tab["Water viscosity (cp)"].to_numpy() * tab["Water FVF (RB/STB)"].to_numpy())
        return mo, mw, pb
    return fn


def spot_ipr(pr, wc, f: SpotInputs, model, t_f, pwf_log=None):
    """IPRCurve (liquid, STB/d against psia) of the SPOT layer; pr psia and wc fraction override f's."""
    import dataclasses
    from .inflow import IPRCurve
    f = dataclasses.replace(f, pr=float(pr), wc=float(wc))
    sp = Spot(f, blackoil_mobility(model, t_f))
    res = sp.results(pwf_log)
    qo = res["oil_rate"]
    fl = lambda p: qo(p) / max(1 - f.wc, 1e-9)
    pw = np.linspace(f.pr, 0.0, 41)
    lay = res["layer"]
    c = IPRCurve("SPOT", "oil", f.pr, pw, np.asarray(fl(pw), float),
                 params={"Total skin": lay["Total skin"], "WIQI": lay["WIQI"], "kro": lay["kro"],
                         "Water saturation": lay["Water saturation"]},
                 notes=[f"SPOT: total skin {lay['Total skin']:.4f} (deviation {lay['Deviation skin']:.5f}), "
                        f"water saturation {lay['Water saturation']:.4f} for the water cut, ko {lay['ko (md)']:.2f} md."],
                 rate_fn=fl)
    c.spot_layer, c.spot_log = lay, res["log"]
    return c
