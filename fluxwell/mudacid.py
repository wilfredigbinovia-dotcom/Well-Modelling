"""Mud acid stimulation of sandstones - PROSPER's Stimulation | Mud Acidization of Sandstones (Tutorial 50).

Reconstructed from the user's T50 file (the Logie well of Tutorial 01: 9000 ft TVD, Pr 5200 psig, 100 ft perforated,
k 50 md, porosity 0.2) and 39 one-at-a-time runs (acid volume, permeability, porosity, tensile strength, Poisson's
ratio, bulk density, damage type, formulation, and every mineral). What was found, and how well it reproduces PROSPER:

Rock (exact): matrix density = weight-fraction average of the mineral densities (MINERALS); bulk density at
Sw = 100 % = (1 - phi) rho_matrix + phi rho_water (the formation water's density); Al and Ca in the bulk rock
(PROSPER labels them gmol/L, the numbers are g/L) = rho_matrix (1 - phi) x sum(w_i x element mass fraction_i).
The Na, K, Mg line is a constant 6.15 in PROSPER whatever the minerals. The entered rock bulk density is not used.

Formulation (exact): optimum HCl = max(7, 7.3 Ca/Al) %; optimum HF 5 % with 5 % clays or more, 2.5 % with
less, 0.5 % with none (clays = kaolinite + illite + montmorillonite + mixed clays; PROSPER: 2.5 % at 4 %, 5 % at 5 %). The standard formulation is 3 % HF / 12 % HCl. ABF 118.9 lbm per % HF (per 1000 gal);
the HCl for 1000 gal of mud acid = HCl + 0.9125 HF (what the ABF consumes).

Dissolution (fitted, within 0.35 % on the 33 runs that return it): porosity increase (%) =
carbonates (calcite, dolomite; x 0.9865 for the standard formulation's weaker HCl) + h(HF) x silicates / (1 - phi)
- each a weight-% sum with per-mineral coefficients (DISSOLUTION) - and a step for the presence of clays. Total
dissolved minerals (g/L of acid) = porosity increase x rho_matrix; porosity after = phi (1 + increase). PROSPER returns
nothing when the rock has no calcium (no carbonates); so does this. Permeability after (exact):
k1/k0 = (phi1/phi0)^2 exp(2 (phi1 - phi0)).

Pumping (fitted on the runs): fracture gradient (2 sigma_h - Pr + tensile strength) / TVD (exact); maximum injection
rate = the Darcy rate at the fracture pressure with the acid's viscosity (relative to water 1 + 0.03887 w +
0.000463 w^2, w = HCl + HF %; within 0.11 %), capped at 8 bbl/min; maximum THP = overburden pressure - acid
hydrostatic head (gradient 0.43189 (1 + 0.005782 w) psi/ft) + tubing friction (within 0.21 psi); pump power
(THP + 200) q / 40.8165 (exact). Flush volumes (exact): gasoil, IPA and HCl preflushes 0.2 x the mud acid volume,
IPA and gasoil overflushes 0.08 x.

Damage removal: treated radius r = rw + 16.485 (V/10000 gal)^0.333 q^0.0652 exp(0.00359 q) in (q the injection
rate; within 0.025 in; it does not depend on porosity, minerals or acid strength); skin after (Hawkins, exact)
(k/k1 - 1) ln(r/rw); PI after = PI before (k/kd) ln(re/rw) / (ln(re/rw) + s) (exact; kd = 0.999888 k - PROSPER's
"permeability damaged zone" with no damage; 0.998754 k with shallow mud-filtrate damage, which leaves the PI after
unchanged). PI before = the test rate over its drawdown on the IPR (PROSPER's 4.95562 against 4.940 here: its
bubble point for the IPR is a little lower).

Not identified: the dissolution chemistry ("ionic equilibrium") itself - the coefficients are fitted to this file's
rock; the absolute acid viscosity and tubing friction (only their dependence on the acid and rate was measured, for
this tubing). The perforation shot density and tunnel diameter do not change any result in PROSPER's T50 runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G_CC = 62.428                          # lb/ft3 per g/cc
# mineral: (density g/cc, Al mass fraction, Ca mass fraction)
MINERALS = {
    "quartz": (2.65, 0.0, 0.0), "kaolinite": (2.59, 0.206, 0.0), "illite": (2.50, 0.181, 0.0),
    "montmorillonite": (2.50, 0.123, 0.0), "muscovite": (2.76, 0.203, 0.0), "mixed_clays": (2.59, 0.190, 0.0),
    "albite": (2.61, 0.122, 0.0), "microcline": (2.56, 0.104, 0.0), "mixed_feldspars": (2.59, 0.113, 0.0),
    "calcite": (2.71, 0.0, 0.400), "dolomite": (2.87, 0.0, 0.217)}
MINERAL_LABELS = {"quartz": "Quartz", "kaolinite": "Kaolinite", "illite": "Illite",
                  "montmorillonite": "Montmorillonite", "muscovite": "Muscovite (mica)", "mixed_clays": "Mixed clays",
                  "albite": "Albite (soda feldspar)", "microcline": "Microcline (potash feldspar)",
                  "mixed_feldspars": "Mixed feldspars", "calcite": "Calcite", "dolomite": "Dolomite"}
CLAYS = ("kaolinite", "illite", "montmorillonite", "mixed_clays")
NA_K_MG = 6.15
# dissolution (porosity increase, % of phi): carbonates per wt%, silicates per wt% (x h(HF) / (1 - phi)), clay step
CARB = {"calcite": 1.070869, "dolomite": 1.102077}
STD_CARB = 0.986487
SIL = {"quartz": 0.043659, "kaolinite": 0.0314, "illite": 0.056526, "montmorillonite": 0.117886,
       "muscovite": 0.030264, "albite": 0.103614, "microcline": 0.126582, "mixed_feldspars": 0.114755,
       "mixed_clays": 0.04598}
CLAY_STEP = 0.614805
H_HF = {5.0: 1.0, 3.0: 0.645066, 2.5: 0.466457, 0.5: -0.028725}
HF_CLAY_SWITCH = 5.0                   # wt% clays for 5 % HF (PROSPER: 4 % -> 2.5 % HF, 5 % -> 5 %)
KD_RATIO = {"none": 0.999888, "shallow": 0.998754}
Q_CAP = 8.0                            # bbl/min
INJ = (21.19946, 0.0388652, 0.000462890)   # uncapped rate at k 50 md, h 100 ft, T49 well; viscosity (1 + a w + b w^2)
THP_C = (3886.98356, 0.00578184852, 7.28397368, 1.80529684, 0.503542006, 0.10039134)   # H0, b, c, q exp, rho exp, mu exp
RS = (16.4850645, 0.0651985, 0.00358968, 0.333)
HP = 33000.0 / (144.0 * 5.614583)      # psi.bbl/min per hp = 40.8165


@dataclass
class MudAcidInputs:
    damage: str = "none"               # "none" | "shallow" (mud-filtrate invasion)
    formulation: str = "optimum"       # "optimum" | "standard" (3 % HF, 12 % HCl)
    volume_gal: float = 10000.0
    overburden_grad: float = 1.05167
    area_acres: float = 500.0
    borehole_d: float = 8.496
    perf_interval: float = 100.0
    spf: float = 8.0
    perf_d: float = 0.62
    q_test: float = 10627.8
    wc_test: float = 0.0
    k: float = 50.0
    porosity: float = 0.2
    bulk_density: float = 166.0        # lb/ft3 (shown, not used - as PROSPER)
    tensile: float = 500.0
    poisson: float = 0.2
    composition: dict = field(default_factory=lambda: dict(
        quartz=67.0, kaolinite=0.0, illite=0.0, montmorillonite=0.0, muscovite=2.0, mixed_clays=12.0, albite=0.0,
        microcline=0.0, mixed_feldspars=1.0, calcite=18.0, dolomite=0.0))
    # from the well model
    tvd: float = 9000.0
    p_res: float = 5214.696            # psia
    pi_before: float = 4.95562
    water_density: float = 63.418      # lb/ft3


def water_density(salinity_ppm):
    """Formation water density at standard conditions (lb/ft3) - PROSPER's T50 value for 23000 ppm."""
    return G_CC * (1.0 + 0.6895e-6 * salinity_ppm)


def hf_optimum(comp):
    clay = sum(comp.get(m, 0.0) for m in CLAYS)
    return 0.5 if clay <= 0 else (2.5 if clay < HF_CLAY_SWITCH else 5.0)


class MudAcid:
    def __init__(self, f: MudAcidInputs):
        self.f = f
        tot = sum(f.composition.values())
        if tot <= 0:
            raise ValueError("Formation composition is empty")
        self.w = {m: f.composition.get(m, 0.0) / tot for m in MINERALS}     # fractions summing to 1
        self.composition_total = tot

    def rock(self):
        f, w = self.f, self.w
        rho_m = sum(w[m] * MINERALS[m][0] for m in MINERALS)              # g/cc
        solid = rho_m * 1000.0 * (1 - f.porosity)                         # g/L of bulk rock
        return {"Given bulk density (lb/ft3)": f.bulk_density, "Calculated matrix density (lb/ft3)": rho_m * G_CC,
                "Calculated bulk density, Sw = 100 % (lb/ft3)": (1 - f.porosity) * rho_m * G_CC
                + f.porosity * f.water_density,
                "Aluminium in bulk rock (g/L)": solid * sum(w[m] * MINERALS[m][1] for m in MINERALS),
                "Na, K, Mg etc. in bulk rock": NA_K_MG,
                "Calcium in bulk rock (g/L)": solid * sum(w[m] * MINERALS[m][2] for m in MINERALS)}

    def formulation(self):
        rk = self.rock()
        if self.f.formulation == "standard":
            return 3.0, 12.0
        al, ca = rk["Aluminium in bulk rock (g/L)"], rk["Calcium in bulk rock (g/L)"]
        hcl = max(7.0, 7.3 * ca / al) if al > 0 else 7.0
        return hf_optimum({m: 100 * v for m, v in self.w.items()}), hcl

    def treatment(self):
        f = self.f
        rk = self.rock()
        hf, hcl = self.formulation()
        out = {"Concentration of HF (%)": hf, "Concentration of HCl (%)": hcl}
        pc = {m: 100 * v for m, v in self.w.items()}
        nan = float("nan")
        if rk["Calcium in bulk rock (g/L)"] <= 0:
            inc = nan
        else:
            carb = sum(pc[m] * c for m, c in CARB.items()) * (STD_CARB if f.formulation == "standard" else 1.0)
            sil = sum(pc[m] * c for m, c in SIL.items()) + CLAY_STEP * (sum(pc[m] for m in CLAYS) > 0)
            h = H_HF.get(hf, float(np.interp(hf, sorted(H_HF), [H_HF[k] for k in sorted(H_HF)])))
            inc = carb + h * sil / (1 - f.porosity)
        phi1 = f.porosity * (1 + inc / 100)
        kr = (phi1 / f.porosity) ** 2 * np.exp(2 * (phi1 - f.porosity))
        out.update({"Porosity increase (%)": inc, "Porosity before treatment": f.porosity,
                    "Porosity after treatment": phi1, "Permeability increase (%)": 100 * (kr - 1),
                    "Permeability before treatment (md)": f.k, "Permeability after treatment (md)": f.k * kr,
                    "Total dissolved minerals (g/L)": inc / 100 * rk["Calculated matrix density (lb/ft3)"] / G_CC * 1000,
                    "Ammonium bifluoride ABF (lbm per 1000 gal)": 118.9 * hf,
                    "HCl for 1000 gal of mud acid (%)": hcl + 0.9125 * hf})
        return out

    def stresses(self):
        f = self.f
        sv = f.overburden_grad * f.tvd
        sh = f.poisson / (1 - f.poisson) * (sv - f.p_res) + f.p_res
        p_frac = 2 * sh - f.p_res + f.tensile
        return sv, sh, p_frac

    def pumping(self):
        f = self.f
        hf, hcl = self.formulation()
        w = hf + hcl
        sv, sh, p_frac = self.stresses()
        mu_r = 1 + INJ[1] * w + INJ[2] * w * w
        re = np.sqrt(f.area_acres * 43560.0 / np.pi)
        rw = f.borehole_d / 24.0
        # the T49/T50 well (h 100 ft, re/rw of 500 acres / 8.496 in, Pfrac - Pr 2625.14 psi) gives INJ[0] at 50 md
        q_free = (INJ[0] * (f.k / 50.0) * (f.perf_interval / 100.0) * ((p_frac - f.p_res) / 2625.144)
                  * (8.914345 / np.log(re / rw)) / mu_r)
        q = min(Q_CAP, q_free)
        rr = 1 + THP_C[1] * w
        thp = ((sv - 14.696) - THP_C[0] * rr * (f.tvd / 9000.0)
               + THP_C[2] * q ** THP_C[3] * rr ** THP_C[4] * mu_r ** THP_C[5])
        v = f.volume_gal
        return {"Number of stages": 1, "Maximum injection rate (bbl/min)": q,
                "Maximum tubing head pressure (psig)": thp, "Fracture gradient (psi/ft)": p_frac / f.tvd,
                "Pump requirement (hp)": (thp + 200.0) * q / HP,
                "1 Gasoil (gal)": 0.2 * v, "2 Alcohol, IPA (gal)": 0.2 * v, "3 HCl preflush (gal)": 0.2 * v,
                "4 Mud acid (gal)": v, "5 Alcohol, IPA (gal)": 0.08 * v, "6 Gasoil (gal)": 0.08 * v}

    def damage_removal(self):
        f = self.f
        tr = self.treatment()
        q = self.pumping()["Maximum injection rate (bbl/min)"]
        rw_in = f.borehole_d / 2.0
        rs = rw_in + RS[0] * (f.volume_gal / 10000.0 * 100.0 / f.perf_interval) ** RS[3] * q ** RS[1] * np.exp(RS[2] * q)
        k1 = tr["Permeability after treatment (md)"]
        s = (f.k / k1 - 1) * np.log(rs / rw_in)
        re = np.sqrt(f.area_acres * 43560.0 / np.pi)
        a = np.log(re / (rw_in / 12.0))
        pi1 = f.pi_before / KD_RATIO["none"] * a / (a + s)
        return {"Radius treated zone (in)": rs, "Permeability damaged zone (md)": f.k * KD_RATIO.get(f.damage, 1.0),
                "Productivity index before treatment (STB/d/psi)": f.pi_before,
                "Productivity index after treatment (STB/d/psi)": pi1, "Skin factor after treatment": s}

    def warnings(self):
        w = []
        if self.rock()["Calcium in bulk rock (g/L)"] > 0:
            w.append("Use of HCl spearhead is vital to prevent precipitation of calcium fluoride.")
        else:
            w.append("No calcium in the rock: PROSPER returns no dissolution results for this composition.")
        if abs(self.composition_total - 100.0) > 1e-6:
            w.append(f"Formation composition sums to {self.composition_total:g} % - normalised to 100 %.")
        return w

    def results(self):
        return {"rock": self.rock(), "treatment": self.treatment(), "pumping": self.pumping(),
                "damage": self.damage_removal(), "warnings": self.warnings()}
