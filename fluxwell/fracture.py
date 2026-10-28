"""Hydraulic fracturing screening - PROSPER's Stimulation | Hydraulic Fracturing (Tutorial 49).

Reconstructed from the user's T49 file (Logie well, 9000 ft TVD, 100 ft pay; 66 treatments of 1000-4000 bbl at
200-400 bbl/min; five of them in detail). What was found, and how well it reproduces PROSPER:

Stresses (exact on the base and the Poisson 0.25 run): overburden sigma_v = gradient x TVD; minimum horizontal stress
sigma_h = nu/(1-nu) (sigma_v - p) + p (the fracture propagation pressure, 6262.57 psig); initial closing pressure
sigma_h - p; actual closing pressure sigma_h - 0.75 p (2351.56 psig - the closure on the proppant); breakdown gradient
= initiation pressure / TVD. Initiation pressure above sigma_h: 240.15 (Ct/0.0028838)^0.8807 ((sigma_h-p)/1062.58)^0.7647
psi, fitted to three runs (base 240.15, fluid loss 0.0062: 408.51, Poisson 0.25: 299.22); the entered tensile strength
does not change any PROSPER output (a run at 1000 psi is identical to the base).

Total fluid loss coefficient: Ct = Cw sqrt(1 - 43.45 Cw) (0.0031 -> 0.0028838, 0.0062 -> 0.005299; fitted).

Geometry (initial length and width within 0.3 % of PROSPER on the base and fluid-loss runs, 1.2 % with a 100 F surface
temperature): linear/vertical (KGD) fracture of the gross thickness, power-law fluid (n, K interpolated at the mean of
surface and reservoir temperature). Wellbore width
    w = 1.37644 [ K (q/H)^n L^2 (1 - nu) / G ]^(1/(2n+2))         (consistent units, G = E / (2 (1 + nu)))
and Carter's leak-off material balance with the average width 0.79827 w and the leak-off coefficient
Ct x (part of wall taking fluid) x (1 + wall area increase). The 1.37644 and 0.79827 are fitted.

Pumping schedule (PROSPER's 100 equal stages): pad until the wellbore width is twice the largest proppant diameter
(0.0942 in), then slurry with the concentration ramped 0.1 lb/gal per stage to the maximum. The proppant settles at the
power-law Stokes velocity of an 0.0843 in grain (273 ft/min at the base temperature). Bank: (A) transport - its length
0.33754 V_slurry / (w (1 - h/H)), its height growing with v_s c dt, until the length reaches q / (2 w v_s); (B) held
there, each stage filling gamma c of the open height above it; (C) when the velocity above the bank exceeds
9.70 (q/200)^0.124 L^0.29 (v_s/273)^-0.3 stages extend the bank instead. Effective (propped) dimensions: the bank
length and height; the width from the proppant mass at the pack density 137.54 lb/ft3. Effective length and width
within 2 % (base, fluid-loss run) and 3.7 % (100 F surface).

Productivity: conductivity c0 + c1 C + c2 C^2 md.ft (C proppant concentration, lb/ft2 - graded sand 8-12 at this
closure, fitted on the five detailed cases); rw' = xf / (2 + pi / FcD) (Meyer & Jacot - exact on all five); a fully
penetrating frac gives J = PI0 (ln(re/rw) - 0.2623) / (ln(re/rw') - 0.1987) and PI after = PI0 + (J - PI0)
(h_eff/H)^0.3777. Within 0.9 % on the base and 100 F runs, 2.5 % with the doubled fluid loss. PROSPER's PI after does
not depend on the formation permeability entered on the frac screen (a run at 100 md is identical).

Treatment (exact on runs 1-6 except where noted): PROSPER's treating-fluid density is 62.416 lb/ft3 at 8.83 lb/gal and
69.81 at 10 (taken linear); hydrostatic head and perforation friction scale with it; THP = propagation pressure + 500
+ tubing friction + perforation friction - hydrostatic (the 500 is a constant, not the tensile strength); pump power
THP q / 40.81; bottom-hole temperature during the job T_res - 0.02 (T_res - T_surface). Tubing friction is PROSPER's
T49 value scaled as q^1.98 rho^0.75 (L/D^4.75 for other tubing) - at 200-400 bbl/min in 4 in tubing it is not a
meaningful number anyway (130 780 psi).

Not identified: the apparent viscosity PROSPER reports (0.50092 cp, unchanged by the surface temperature) and the
exact fluid temperature it uses for n and K (the mean fits the base exactly; at a 100 F surface ~1 F more fits better).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import erfcx

BBL = 5.614583
G_ACC = 32.174
CK = 1.37644                 # KGD power-law width coefficient (fitted)
AVG = 0.79827                # average / wellbore width for the leak-off balance (fitted)
CT_K = 43.45                 # PROSPER's total fluid loss coefficient Cw sqrt(1 - 43.45 Cw) (T49: 0.0031 -> 0.0028838,
#                              0.0062 -> 0.005299, both within 0.03 %; its origin was not identified)
PAD_D = 0.0942               # 8-12 sand: pad until w >= 2 PAD_D (PROSPER ~0.094 in; our widths run 0.2 % high)
D_PROP = 0.08433             # effective grain diameter of 8-12 sand for settling (in, fitted - see Fracture.settling)
VS_REF, THR_VS = 273.0, -0.3   # phase C threshold scales as (v_s / 273 ft/min)^-0.3 (T49 at 60 and 100 F surface)
K_LEN_A = 0.33754            # transport length = K_LEN_A V_slurry / (w (1 - h/H)) (bbl, in)
SG_PROP = 2.65               # sand; PROSPER's proppant density = SG x the fracture fluid density
PACK = 137.54                # proppant pack density, lb/ft3
THR = (9.70, 0.124, 0.29)    # phase C velocity threshold: a (q/200)^b L^c
CLOSE_P = 0.75   # actual closing = sigma_h - 0.75 p_res: exact for Poisson 0.2 and 0.25 (T49 runs base and 3)
INIT = (240.15, 0.0028838, 0.8807, 1062.58, 0.7647)   # initiation - propagation (psi): a (Ct/Ct0)^b (sh'/sh0')^c
RHO_EFF = (62.416, 6.3197, 8.83)   # PROSPER's treating-fluid density (lb/ft3) for 8.83 and 10 lb/gal: linear
TUB = (130780.0, 1.9827, 0.75)     # PROSPER's tubing friction at 200 bbl/min (Logie, 4.052 in to 9275 ft): q^b rho^c
PI_X0, PI_X1, PI_HEXP = -0.2623, -0.1987, 0.3777
# graded sand 8-12 at 2366 psi closure: conductivity (md.ft) = c0 + c1 C + c2 C^2 (C lb/ft2), fitted
COND = (10780.33, 1008.822, 352.1725)


@dataclass
class FracInputs:
    v_min: float = 1000.0            # bbl
    v_max: float = 4000.0
    q_min: float = 200.0             # bbl/min
    q_max: float = 400.0
    wall_fluid: float = 0.75         # part of the fracture wall taking fluid
    area_increase: float = 0.50      # wall area increase due to irregularities
    t_surface: float = 60.0          # F
    surface_losses: float = 200.0    # psi
    h_gross: float = 100.0           # ft
    overburden_grad: float = 1.05167  # psi/ft
    area_acres: float = 500.0
    borehole_d: float = 8.496        # in
    perf_interval: float = 100.0
    spf: float = 8.0
    perf_d: float = 0.62             # in
    porosity: float = 0.2
    k: float = 50.0                  # md
    tensile: float = 500.0           # psi
    poisson: float = 0.2
    youngs: float = 3.0e6            # psi
    c_wall: float = 0.0031           # ft/sqrt(min)
    spurt: float = 0.0
    fluid_density: float = 8.83      # lb/gal
    c_max: float = 1.0               # lb/gal
    fluid_table: tuple = ((60, 0.73, 0.0009), (100, 0.78, 0.0005), (120, 0.83, 0.00028), (140, 0.90, 0.00012),
                          (160, 0.99, 0.00004))   # (T F, n, K lbf.s^n/ft2)
    # well (from the PROSPER model)
    tvd: float = 9000.0
    p_res: float = 5214.696          # psia
    t_res: float = 210.0
    pi_before: float = 5.0
    tubing_id: float = 4.052         # in
    tubing_md: float = 9275.0
    roughness: float = 0.0006
    n_volumes: int = 11
    n_rates: int = 6

    def fluid_nk(self, t):
        a = np.array(self.fluid_table, float)
        a = a[np.argsort(a[:, 0])]
        return float(np.interp(t, a[:, 0], a[:, 1])), float(np.interp(t, a[:, 0], a[:, 2]))


def total_loss_coefficient(c_wall):
    """PROSPER's total fluid loss coefficient from the entered one (see CT_K)."""
    return c_wall * np.sqrt(max(1.0 - CT_K * c_wall, 1e-6))


def stresses(f: FracInputs):
    g = 14.696
    sv = f.overburden_grad * f.tvd
    a = f.poisson / (1 - f.poisson)
    sh = a * (sv - f.p_res) + f.p_res
    # initiation above propagation: fitted on the user's T49 runs (base 240.15 psi; fluid loss 0.0062: 408.51;
    # Poisson 0.25: 299.22 - within 0.03 %); the entered tensile strength does not change it in PROSPER
    ct = total_loss_coefficient(f.c_wall)
    dpi = INIT[0] * (ct / INIT[1]) ** INIT[2] * ((sh - f.p_res) / INIT[3]) ** INIT[4]
    p_init = sh + dpi
    return {"Overburden stress (psia)": sv, "Initial closing pressure (psig)": sh - f.p_res - g,
            "Actual closing pressure (psig)": sh - CLOSE_P * f.p_res - g, "Fracture propagation pressure (psig)": sh - g,
            "Fracture initiation pressure (psig)": p_init - g, "Fracture breakdown gradient (psi/ft)": p_init / f.tvd,
            "_sh": sh, "_closure": sh - CLOSE_P * f.p_res}


class Fracture:
    def __init__(self, f: FracInputs):
        self.f = f
        self.t_fluid = 0.5 * (f.t_surface + f.t_res)
        self.n, self.K = f.fluid_nk(self.t_fluid)
        self.G = f.youngs / (2 * (1 + f.poisson))
        self.c_total = f.c_wall * np.sqrt(max(1.0 - CT_K * f.c_wall, 1e-6))
        self.ce = self.c_total * f.wall_fluid * (1 + f.area_increase)

    def width(self, q, L):
        """Wellbore width (in) of a KGD fracture of half-length L (ft) at q (bbl/min)."""
        n, H = self.n, self.f.h_gross
        qs = q * BBL / 60.0
        x = self.K / 144.0 * (qs / H) ** n * np.asarray(L, float) ** 2 * (1 - self.f.poisson) / self.G
        return 12.0 * CK * x ** (1.0 / (2 * n + 2))

    def geometry(self, q, t):
        """(half-length ft, wellbore width in) after t minutes at q bbl/min: Carter with the KGD width."""
        H, ce = self.f.h_gross, self.ce

        def area(L):
            wb = AVG * self.width(q, L) / 12.0
            x = 2 * ce * np.sqrt(np.pi * t) / wb
            return q * BBL * wb / (4 * np.pi * ce * ce) * (erfcx(x) + 2 * x / np.sqrt(np.pi) - 1) - 2 * H * L
        L = brentq(area, 1e-3, 1e6)
        return L, float(self.width(q, L))

    def schedule(self, q, v):
        """PROSPER's 100-stage treatment: pad, slurry ramp, proppant bank. Returns (pad dict, schedule DataFrame,
        effective dimensions dict)."""
        f, H = self.f, self.f.h_gross
        dv = v / 100.0
        dt = dv / q
        geo = [self.geometry(q, k * dt) for k in range(1, 101)]
        k_pad = next((k for k, (_, w) in enumerate(geo, 1) if w >= 2 * PAD_D), 100)
        lp, wp = geo[k_pad - 1]
        h = lb = m_tot = 0.0
        phase = "A"
        vs = self.settling()
        gam = 1.0 / (SG_PROP * f.fluid_density)        # proppant volume fraction per lb/gal (PROSPER's density)
        rows = []
        for k in range(k_pad + 1, 101):
            ns = k - k_pad
            L, w = geo[k - 1]
            c = min(0.1 * ns, f.c_max)
            m = c * dv * 42.0
            m_tot += m
            if phase == "A":
                l_cap = q * BBL * 12.0 / (2.0 * w * vs)
                l_tr = K_LEN_A * ns * dv / (w * (1 - h / H))
                if l_tr >= l_cap:
                    lb, phase = l_cap, "B"
                    h += gam * c * (H - h)
                else:
                    lb = l_tr
                    h += vs * gam * c * ns * dt + (wp if ns == 1 else 0.0)
            else:
                u = q / (w * (H - h))
                if u < THR[0] * (q / 200.0) ** THR[1] * L ** THR[2] * (vs / VS_REF) ** THR_VS:
                    h += gam * c * (H - h)
                else:
                    phase = "C"
                    lb += np.pi * m / (4 * PACK * h * w / 12.0)
            rows.append({"Pump time (min)": k * dt, "Volume pumped (bbl)": k * dv, "Fracture length (ft)": L,
                         "Fracture width (in)": w, "Proppant concentration (lb/gal)": c, "Proppant amount (lb)": m,
                         "Proppant total (lb)": m_tot, "Bank length (ft)": lb, "Bank height (ft)": h})
        sch = pd.DataFrame(rows)
        w_eff = m_tot / (PACK * 2 * lb * h) * 12.0 if lb > 0 and h > 0 else 0.0
        L0, w0 = geo[-1]
        eff = {"Initial fracture height (ft)": H, "Initial fracture length (ft)": L0, "Initial fracture width (in)": w0,
               "Effective fracture height (ft)": h, "Effective fracture length (ft)": lb,
               "Effective fracture width (in)": w_eff, "Proppant (lb)": m_tot}
        pad = {"Pad time (min)": k_pad * dt, "Pad volume (bbl)": k_pad * dv, "Fracture length (ft)": lp,
               "Fracture width (in)": wp}
        return pad, sch, eff

    def settling(self):
        """Proppant settling velocity (ft/min): Stokes in the power-law fluid (n, K at the fluid temperature),
        v = [g d^(n+1) (rho_p - rho_f) / (18 K 3^(n-1))]^(1/n), PROSPER's proppant density 2.65 x the fluid density.
        d = 0.08433 in reproduces the bank of the T49 file at 60 F surface temperature (273 ft/min) and predicts its
        100 F run (387 ft/min against ~390 fitted)."""
        n, K = self.n, self.K * G_ACC
        rho_f = self.f.fluid_density * 7.48052
        d = D_PROP / 12.0
        return (G_ACC * d ** (n + 1) * (SG_PROP - 1.0) * rho_f / (18.0 * K * 3.0 ** (n - 1))) ** (1.0 / n) * 60.0

    def productivity(self, eff):
        f = self.f
        conc = eff["Effective fracture width (in)"] / 12.0 * PACK
        cond = max(COND[0] + COND[1] * conc + COND[2] * conc * conc, 1.0)
        xf = eff["Effective fracture length (ft)"]
        if xf <= 0 or eff["Effective fracture height (ft)"] <= 0:
            return {"Proppant concentration (lb/ft2)": 0.0, "Fracture conductivity (md.ft)": 0.0,
                    "Dimensionless conductivity": 0.0, "Effective wellbore radius (ft)": f.borehole_d / 24.0,
                    "Productivity index - before (STB/d/psi)": f.pi_before,
                    "Productivity index - after (STB/d/psi)": float("nan")}
        fcd = cond / (f.k * xf)
        rwp = xf / (2 + np.pi / fcd)
        re = np.sqrt(f.area_acres * 43560.0 / np.pi)
        rw = f.borehole_d / 24.0
        j_frac = f.pi_before * (np.log(re / rw) + PI_X0) / (np.log(re / rwp) + PI_X1)
        hf = min(eff["Effective fracture height (ft)"] / f.h_gross, 1.0) ** PI_HEXP
        pi_after = f.pi_before + (j_frac - f.pi_before) * hf
        return {"Proppant concentration (lb/ft2)": conc, "Fracture conductivity (md.ft)": cond,
                "Dimensionless conductivity": fcd, "Effective wellbore radius (ft)": rwp,
                "Productivity index - before (STB/d/psi)": f.pi_before,
                "Productivity index - after (STB/d/psi)": pi_after}

    def treatment(self, q, v):
        """Treatment pressures as PROSPER reports them (T49 runs: base, fluid density 10 lb/gal, Poisson 0.25,
        surface temperature 100 F). Its tubing friction (at these rates through production tubing, not a meaningful
        number) is reproduced by scaling its T49 value: it is not a general pipe-flow calculation."""
        f = self.f
        st = stresses(f)
        rho = RHO_EFF[0] + RHO_EFF[1] * (f.fluid_density - RHO_EFF[2])        # lb/ft3, PROSPER's
        n_perf = f.spf * f.perf_interval
        perf = 0.37521 * q * q * (rho / 7.0686) / (n_perf ** 2 * f.perf_d ** 4)
        tub = (TUB[0] * (q / 200.0) ** TUB[1] * (rho / RHO_EFF[0]) ** TUB[2] * (f.tubing_md / 9275.0)
               * (4.052 / f.tubing_id) ** 4.75)
        hyd = rho / 144.0 * f.tvd
        thp = st["Fracture propagation pressure (psig)"] + 500.0 + tub + perf - hyd
        bht = f.t_res - 0.02 * (f.t_res - f.t_surface)
        return {"Treatment volume (bbl)": v, "Pump rate (bbl/min)": q,
                "Fracture fluid apparent viscosity (cp)": self.apparent_viscosity(),
                "Total fluid loss coefficient (ft/sqrt(min))": self.c_total,
                **{k: v_ for k, v_ in st.items() if not k.startswith("_")},
                "Perforation friction losses (psi)": perf, "Tubing friction losses (psi)": tub,
                "Tubing hydrostatic head (psi)": hyd, "Tubing head pressure (psig)": thp,
                "Surface pressure losses (psi)": f.surface_losses, "Pump power requirement (hp)": thp * q / 40.8095,
                "Bottom hole temperature during job (F)": bht, "Surface temperature during job (F)": f.t_surface}

    def apparent_viscosity(self):
        """Not PROSPER's (0.50092 cp in T49, origin not identified): the power-law viscosity at 1000 1/s at the
        bottom-hole temperature."""
        n, K = self.f.fluid_nk(self.f.t_res - 3.0)
        return K * 1000.0 ** (n - 1) * 47880.26

    def sensitivity(self):
        f = self.f
        vols = np.linspace(f.v_min, f.v_max, f.n_volumes)
        rates = np.linspace(f.q_min, f.q_max, f.n_rates)
        out, cases = [], []
        for q in rates:
            for v in vols:
                pad, sch, eff = self.schedule(q, v)
                pr = self.productivity(eff)
                out.append({"Pump rate (bbl/min)": q, "Treatment volume (bbl)": v,
                            "Fracture length (ft)": eff["Initial fracture length (ft)"],
                            "Fracture width (in)": eff["Initial fracture width (in)"],
                            "Final fracture length (ft)": eff["Effective fracture length (ft)"],
                            "Final fracture width (in)": eff["Effective fracture width (in)"],
                            "Effective fracture height (ft)": eff["Effective fracture height (ft)"],
                            "Productivity index - after (STB/d/psi)": pr["Productivity index - after (STB/d/psi)"]})
                cases.append((pad, sch, eff, pr))
        tab = pd.DataFrame(out)
        tab.index = pd.RangeIndex(1, len(tab) + 1, name="Case")
        return tab, cases
