"""Hydraulic fracturing screening - PROSPER's Stimulation | Hydraulic Fracturing (Tutorial 49).

Reconstructed from the user's T49 file (Logie well, 9000 ft TVD, 100 ft pay; 66 treatments of 1000-4000 bbl at
200-400 bbl/min; five of them in detail). What was found, and how well it reproduces PROSPER:

Stresses (exact): overburden sigma_v = gradient x TVD; minimum horizontal stress sigma_h = nu/(1-nu) (sigma_v - p) + p
(the fracture propagation pressure, 6262.57 psig); initial closing pressure sigma_h - p; actual closing pressure
nu/(1-nu) sigma_v (the closure on the proppant once the well is depleted, 2351.56 psig); breakdown gradient = initiation
pressure / TVD. Initiation pressure ~ sigma_h + T (1-2nu)(1-nu) (240.0 psi above sigma_h against PROSPER's 240.15).

Geometry (within 0.35 % of PROSPER's length and width at every time step of five schedules and the 66 final values):
linear/vertical (KGD) fracture of the gross thickness, power-law fluid (n, K interpolated at the mean of surface and
reservoir temperature). Wellbore width
    w = 1.37644 [ K (q/H)^n L^2 (1 - nu) / G ]^(1/(2n+2))         (consistent units, G = E / (2 (1 + nu)))
and Carter's leak-off material balance with the average width 0.79827 w and the leak-off coefficient
C_total x (part of wall taking fluid) x (1 + wall area increase). The 1.37644 and 0.79827 are fitted.

Pumping schedule (PROSPER's 100 equal stages): pad until the wellbore width is twice the largest proppant diameter
(PROSPER ~0.094 in for 8-12 sand), then slurry with the concentration ramped 0.1 lb/gal per stage to the maximum. The proppant
bank: (A) its length is the slurry volume over 2 w (H - h) (transport), its height grows by 116.67 dt c^2 (the first
stage adds the pad width - PROSPER's); (B) once the length reaches the settling length 0.12336 q / w (q bbl/min, w in)
it is held, and each stage fills gamma c (gamma 0.04272) of the open height above the bank; (C) when the velocity
above the bank q / (w (H - h)) exceeds a threshold that grows with the fracture length, stages extend the bank instead
(by the stage's proppant volume over the bank's height, pi/2 x the w h section). The phase-C threshold
9.70 (q/200)^0.124 L^0.29 is fitted to the five schedules. Effective (propped) dimensions: the bank length and height;
the width from the proppant mass at the pack density 137.54 lb/ft3. Against the 66 PROSPER treatments: effective
length within 1.6 %, width within 1.2 %.

Productivity: conductivity from the proppant concentration (lb/ft2) - PROSPER's graded sand 8-12 at this closure,
interpolated through its five detailed cases; effective wellbore radius rw' = xf / (2 + pi / FcD) (Meyer & Jacot -
exact on all five); PI after = PI before x (ln(re/rw) - 0.9687) / (ln(re/rw') - 0.4084) (fitted: within 0.3 % on
the five, 0.9 % on the 66).

Not identified: PROSPER's total fluid-loss coefficient (0.0028838 from the entered 0.0031 - taken as that ratio), its
apparent viscosity (0.50092 cp), bottom-hole temperature during the job (reservoir - 3 F) and its tubing friction
(Colebrook here, ~15 % different; at 200-400 bbl/min in 4 in tubing the number is not meaningful anyway).
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
CT_RATIO = 0.93026           # PROSPER's total / entered fluid loss coefficient (T49)
PAD_D = 0.0942               # 8-12 sand: pad until w >= 2 PAD_D (PROSPER ~0.094 in; our widths run 0.2 % high)
K_HEIGHT_A = 116.6667        # phase A bank height growth (ft per min per (lb/gal)^2)
K_CAP = 0.12336              # settling length = K_CAP q / w (q bbl/min, w in)
K_LEN_A = 0.33754            # transport length = K_LEN_A V_slurry / (w (1 - h/H)) (bbl, in)
GAMMA = 0.04272              # phase B: open height filled per stage per lb/gal
PACK = 137.54                # proppant pack density, lb/ft3
THR = (9.70, 0.124, 0.29)    # phase C velocity threshold: a (q/200)^b L^c
PI_X0, PI_X1 = -0.9687, -0.4084
# PROSPER graded sand 8-12 at 2366 psi closure: conductivity (md.ft) against concentration (lb/ft2)
COND_TABLE = ((0.640, 10738.9), (0.8209, 11578.1), (1.619, 13168.5), (2.456, 15462.0), (3.697, 21002.9))


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


def stresses(f: FracInputs):
    g = 14.696
    sv = f.overburden_grad * f.tvd
    a = f.poisson / (1 - f.poisson)
    sh = a * (sv - f.p_res) + f.p_res
    p_init = sh + f.tensile * (1 - 2 * f.poisson) * (1 - f.poisson)
    return {"Overburden stress (psia)": sv, "Initial closing pressure (psig)": sh - f.p_res - g,
            "Actual closing pressure (psig)": a * sv - g, "Fracture propagation pressure (psig)": sh - g,
            "Fracture initiation pressure (psig)": p_init - g, "Fracture breakdown gradient (psi/ft)": p_init / f.tvd,
            "_sh": sh, "_closure": a * sv}


class Fracture:
    def __init__(self, f: FracInputs):
        self.f = f
        self.t_fluid = 0.5 * (f.t_surface + f.t_res)
        self.n, self.K = f.fluid_nk(self.t_fluid)
        self.G = f.youngs / (2 * (1 + f.poisson))
        self.ce = f.c_wall * CT_RATIO * f.wall_fluid * (1 + f.area_increase)

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
        rows = []
        for k in range(k_pad + 1, 101):
            ns = k - k_pad
            L, w = geo[k - 1]
            c = min(0.1 * ns, f.c_max)
            m = c * dv * 42.0
            m_tot += m
            if phase == "A":
                l_cap = K_CAP * q / w
                l_tr = K_LEN_A * ns * dv / (w * (1 - h / H))
                if l_tr >= l_cap:
                    lb, phase = l_cap, "B"
                    h += GAMMA * c * (H - h)
                else:
                    lb = l_tr
                    h += K_HEIGHT_A * dt * c * c + (wp if ns == 1 else 0.0)
            else:
                u = q / (w * (H - h))
                if u < THR[0] * (q / 200.0) ** THR[1] * L ** THR[2]:
                    h += GAMMA * c * (H - h)
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

    def productivity(self, eff):
        f = self.f
        conc = eff["Effective fracture width (in)"] / 12.0 * PACK
        t = np.array(COND_TABLE)
        if conc <= t[0, 0]:
            cond = t[0, 1] + (conc - t[0, 0]) * (t[1, 1] - t[0, 1]) / (t[1, 0] - t[0, 0])
        elif conc >= t[-1, 0]:
            cond = t[-1, 1] + (conc - t[-1, 0]) * (t[-1, 1] - t[-2, 1]) / (t[-1, 0] - t[-2, 0])
        else:
            cond = float(np.interp(conc, t[:, 0], t[:, 1]))
        cond = max(cond, 1.0)
        xf = eff["Effective fracture length (ft)"]
        fcd = cond / (f.k * xf)
        rwp = xf / (2 + np.pi / fcd)
        re = np.sqrt(f.area_acres * 43560.0 / np.pi)
        rw = f.borehole_d / 24.0
        pi_after = f.pi_before * (np.log(re / rw) + PI_X0) / (np.log(re / rwp) + PI_X1)
        return {"Proppant concentration (lb/ft2)": conc, "Fracture conductivity (md.ft)": cond,
                "Dimensionless conductivity": fcd, "Effective wellbore radius (ft)": rwp,
                "Productivity index - before (STB/d/psi)": f.pi_before,
                "Productivity index - after (STB/d/psi)": pi_after}

    def treatment(self, q, v):
        f = self.f
        st = stresses(f)
        rho = f.fluid_density * 7.48052                       # lb/ft3
        n_perf = f.spf * f.perf_interval
        perf = 0.37521 * q * q * f.fluid_density / (n_perf ** 2 * f.perf_d ** 4)   # fitted constant (Cd ~0.79)
        d = f.tubing_id / 12.0
        vel = q * BBL / 60.0 / (np.pi / 4 * d * d)
        mu = self.apparent_viscosity()
        re = rho * vel * d / (mu * 6.7197e-4)
        fr = brentq(lambda x: 1 / np.sqrt(x) + 2 * np.log10(f.roughness / f.tubing_id / 3.7 + 2.51 / (re * np.sqrt(x))),
                    1e-5, 0.5)
        tub = fr * rho * vel * vel * f.tubing_md / (2 * G_ACC * d) / 144.0
        hyd = 0.433445 * f.tvd                                 # PROSPER: water gradient
        thp = st["Fracture propagation pressure (psig)"] + f.tensile + tub + perf - hyd
        return {"Treatment volume (bbl)": v, "Pump rate (bbl/min)": q, "Fracture fluid apparent viscosity (cp)": mu,
                "Total fluid loss coefficient (ft/sqrt(min))": f.c_wall * CT_RATIO,
                **{k: v_ for k, v_ in st.items() if not k.startswith("_")},
                "Perforation friction losses (psi)": perf, "Tubing friction losses (psi)": tub,
                "Tubing hydrostatic head (psi)": hyd, "Tubing head pressure (psig)": thp,
                "Surface pressure losses (psi)": f.surface_losses, "Pump power requirement (hp)": thp * q / 40.8095,
                "Bottom hole temperature during job (F)": f.t_res - 3.0, "Surface temperature during job (F)": f.t_surface}

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
