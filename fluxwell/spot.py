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
                         not enter. Scaling with the gun's normalised Lp (stored in PROSPER's gun database) is
                         assumed (one gun was available).
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
  total skin = cos (cos^2 + sin^2 kv/kh)^0.478 (0.996 Sperf + 0.989 Scrush) + 0.997 Sdev + Spart
               (layer means over the perforated segments; fitted - within 0.025 on the 14 fully perforated runs),
  WIQI = (A' + Sdev + Spart) / (A' + S), A' = ln(re / rw) - 0.75 + 0.1405 (fitted, within 0.9 %; 1.8 % held up).
  Gun held up above the layer bottom (holdup depth; 6 runs, 25-81 of 111 ft perforated, 0-63 deg, Kv/Kh 0.1 and 1):
  segments below it unperforated; the slanted, partially-penetrating well of Cinco-Ley, Ramey & Miller on the
  perforated segments gives the penetration skin (the same interval vertical) and the deviation skin (the rest) -
  their sum within 0.04 of PROSPER's on the half-perforated runs, though PROSPER splits it differently (its
  penetration skin 0.03 higher, 0.13 on the 95 % perforated run); the mechanical skin x n / n_perf x
  (0.987 - 0.040 (1 - cos)) (fitted). Total skin within 0.08, oil AOF within 0.4 %. The segment at the holdup depth
  has a 0.9434 x smaller entry hole (every run).
  Water cut entered: the water saturation that gives it (Corey relative permeabilities, mobilities at Pr) sets
  kro and krw (exact), ko = k kro.
  IPR: oil q = 0.00708 ko h / (A + S) [ int(Pwf..Pr) dp / (mu_o B_o) above Pb; below it Fetkovich
  (Pb^2 - Pwf^2) / (2 Pb) / (mu_o B_o)_Pb or Vogel ], liquid = oil / (1 - water cut); h the layer's TVD thickness.
  PROSPER's layer table: oil AOF on n / (n - 1) of that thickness (the segment count; on the thickness itself with
  the gun held up) - within 0.2 % (1 % held up); PI = the oil rate at half the layer pressure (gauge) over that
  drawdown (within 0.007 of PROSPER's 2-decimal values on 20 runs);
  Min Lp = max(0, 1.30082 Lp - 4.41478) (within PROSPER's 2 decimals).

Not identified: PROSPER's exact split of the slant and penetration skins with the gun held up (above), the
thick-wall-cylinder columns (from the log; not used), the IPR temperature column (a function of the flowing pressure
and slightly of the water cut only - not of the layer temperature, pressure or rate - not used by any calculation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from . import completion as wcomp

LP_C, LP_U0, LP_S0, LP_ALPHA, LP_NORM_REF = 1.85055313, 32357.0, 26989.6, 1.0256, 20.1883
MINLP = (1.30082, -4.41478)                 # PROSPER's Min Lp = max(0, 1.30082 Lp - 4.41478)
EHD_FACTOR = 0.22427 / 0.23
AGG = (0.9958, 0.98885, 0.99737, 0.99676, 0.4783)   # total skin: perf, crush, Sdev factors; cos and (cos2+sin2 kv/kh) exps
PP_MECH = (0.987, 0.040)            # held up: mechanical skin x n / n_perf x (0.987 - 0.040 (1 - cos))
HOLDUP_EHD = 0.21157 / 0.22427      # the segment at the holdup depth: smaller entry hole
WIQI_DA = 0.1405                    # WIQI = (A' + Sdev + Spart) / (A' + S), A' = A + 0.1405
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
    test_standoff: float = 0.5      # in, the API test's


GUNS = {SpotGun().name: SpotGun()}          # the Tutorial 09 gun (older projects store this name)
KT_PHASINGS = (0, 45, 60, 90, 120, 180)       # Karakas & Tariq's tabulated phasings
TUTORIAL_GUN = 43                             # its row in PROSPER's gun database (wireline)


def gun_database():
    """PROSPER's perforating gun database (PXWell.pxdb, 824 guns in PROSPER's order): wireline then TCP, each by
    gun OD. Read from the binary file, columns named as PROSPER's gun table (checked against the user's export of
    it: 823 of 824 rows, the same values bar the export's transcription slips) - section 1 (API RP 43 / 19B
    concrete target) penetration and entry hole, the normalised penetration PROSPER's SPOT uses, the test's cement
    strength, casing and standoff, explosive weight, ratings, minimum restriction, gun swell, debris and remarks."""
    import pandas as pd
    from pathlib import Path
    return pd.read_csv(Path(__file__).parent / "data" / "perforating_guns.csv", index_col="Gun",
                       keep_default_na=False, na_values=[""])


def kt_phasing(phasing):
    """The phasing used in Karakas & Tariq: the gun's own - PROSPER interpolates the constants between the tabulated
    phasings (Run 14, 72 deg: exact). Kept for callers; returns the phasing modulo 360."""
    return float(phasing) % 360


def kt_phasing_nearest(phasing):
    """Nearest of Karakas & Tariq's tabulated phasings (0 for in-line guns; 360 counts as 0)."""
    p = float(phasing) % 360
    return min(KT_PHASINGS, key=lambda a: min(abs(p - a), 360 - abs(p - a)) if a == 0 else abs(p - a))


def gun_from_row(row) -> SpotGun:
    name = (f"{row['Vendor']} - {row['Charge Name']} ({row['Shot Density (spf)']:g} spf / "
            f"{row['Gun Phasing (deg)']:g} deg / {str(row['Gun Type']).lower()}, {row['Gun OD (in)']:g} in)")
    return SpotGun(name=name, gun_od=float(row["Gun OD (in)"]), spf=float(row["Shot Density (spf)"]),
                   phasing=float(row["Gun Phasing (deg)"]), lp_api=float(row["Lp sec 1 (in)"]),
                   lp_norm=float(row["Norm. Penetration (in)"]), ehd=float(row["EHD sec 1 (in)"]),
                   cement_strength=float(row["Cement sec 1 (psi)"]),
                   test_standoff=float(row.get("Test Standoff (in)", 0.0) or 0.0))


# ---------------------------------------------------------------- PROSPER's SPOT screens beyond Tutorial 09
# Everything below the calibrated core follows the PROSPER manual (SPOT, pp. 445-493). Where the manual names a
# Shell correlation without giving it, a placeholder is used and listed in PROVISIONAL - each is to be calibrated
# against PROSPER runs; until then the factor is neutral (the Tutorial 09 result is unchanged).
PROVISIONAL = {
    "casing": "casing grades L80 / P105 (entry hole from hardness; J55 and P110 calibrated)",
    "multi_casing": "a third casing string (the second-string correction applied again)",
    "synthetic": "Synthetic Rock Lp at other UCS (calibrated at 8000 psi)",
    "gun": "guns other than Tutorial 09's and Run 14's UltraJet 4505 (Lp_API / ln(29665 / cement) scaling)",
    "gravel_ext": "external (underreamed) gravel pack (PROSPER's -24 perforation skin not reproduced)",
    "invasion_calc": "Calculated invasion depths (exposure time and radial sqrt(t) as PROSPER; magnitude from filtration theory)",
    "ucs_twc": "UCS / TWC equations 1-7: published correlations (McNally, Moos, Vernik, Rzhevsky, TWC = 2.5 / 3 UCS)",
    "sanding": "QinetiQ sanding: Shields traction (horizontal perforations) / settling velocity (vertical)",
    "non_darcy": "Perforation non-Darcy skin: Forchheimer flow through the crushed zone (McLeod)",
    "gravel": "gravel-pack non-Darcy (Golan / Geertsma)",
    "extensions": "SPOT IPR Extensions = No: as Yes",
}
# --- calibrated against the user's PROSPER runs on the Tutorial 09 file (November 2026)
QINETIQ_CRUSH = (0.33842, 0.5)  # QinetiQ Test Results: crush skin 1.51659 against 10.1298 entered (Run 5)
SO_LP = (0.076972, 0.028489)    # Lp ~ exp(-a s - b s^2), s = standoff (in): runs 2 (0.5, 1.5 in), 1a (2.10), base
                                # (2.205) and 10b (5.0): 1.30024, 1.13770, 1.02061, 1, 0.45442 within 0.06 %
SO_EHD = (0.072928, -0.062427, 0.0077126)   # entry hole ~ 1 + c1 s + c2 s^2 + c3 s^3 (same runs, within 1e-4)
SO_REF = (2.205, 0.5)           # Tutorial 09: downhole (centred 2 in gun in 6.41 in ID) and API test standoffs
GRADE_EHD = {"J55": 1.0, "L80": 0.9055, "P105": 0.8558, "P110": 0.84907}  # entry hole vs J55 (P110 Run 1b;
                                # L80 / P105 from Brinell hardness^0.376 - provisional); Lp unchanged by grade
TWO_STRINGS = (0.5, 2.93363, 1.05997)   # a second casing string: Lp -> 0.5 Lp - 2.93363 x its wall (in),
                                        # entry hole x 1.05997 (Run 1c: 7-5/8 in 29.7 lb/ft over the 6.7 in)
SYNTHETIC = (0.40962, 31906.0)  # Synthetic Rock Lp: 0.40962 x the conventional with S0 31906 (Run 3, 0.45 x)
AGG_D = 0.98693                 # total skin weight of the damaged rows' M Sp and damage skin (runs 7, 8)
GP_C = 0.24834                  # internal gravel pack (no gravel non-Darcy unless Calculate Non-Darcy Skin): S = k (Lp + rw - casing ID / 2) / (C k_gp n A_p)
TWC_UCS = 1.9049                # PROSPER's generated TWC / UCS (15239.53 psi at 8000 psi)
LP_UC = 29665.0                 # other guns: Lp scales with Lp_API / ln(LP_UC / test cement strength) - Run 14
                                # (Schlumberger UltraJet 4505 HMX, 42.6 in at 7293.3 psi: 14.2513 in downhole)
LOG_COLS = ["MD (ft)", "Porosity", "Permeability (md)", "UCS (psi)", "TWC (psi)", "FDC (g/cc)",
            "Compressional Sonic (us/ft)", "Shear Sonic (us/ft)", "Caliper (in)", "Gamma Ray (API)",
            "Resistivity (ohm.m)", "CBL (mV)", "Net", "Invasion Depth (in)", "Perforated"]


@dataclass
class SpotInputs:
    """One SPOT layer and the options it is calculated with (PROSPER's Options / Layers / Completion tabs)."""
    top_md: float = 9275.0
    bottom_md: float = 9385.0
    survey: tuple = ((9275.0, 9000.0), (9385.0, 9100.0))     # (MD, TVD) ft
    deviated: bool = True
    inflow: str = "Fetkovich"        # "Fetkovich" | "Vogel" | horizontal: "Economides-Joshi" | "Borisov" | ...
    rw: float = 0.354                # ft
    re: float = 4200.0               # drainage radius, ft
    perf_eff: float = 0.9
    pr: float = 4000.0 + G           # psia
    underbalance: float = 100.0      # psi
    ob_grad: float = 1.0             # psi/ft
    wc: float = 0.2                  # fraction
    kv_kh: float = 0.1
    k: float = 50.0                  # md (the log's when there is no log)
    porosity: float = 0.18
    ucs: float = 8000.0              # psi
    crush_factor: float = 0.1
    crush_thickness: float = 1.2     # in
    gun: SpotGun = field(default_factory=SpotGun)
    relperm: tuple = (0.15, 0.2, 0.8, 0.6, 1.5, 1.5)   # Sor, Swr, kro end, krw end, n oil, n water
    holdup_md: float | None = None   # deepest point the gun reaches (None: the layer bottom)
    # --- options beyond Tutorial 09 (defaults reproduce it) ---
    log: object = None               # DataFrame with LOG_COLS (None: a constant 1 ft log of k, porosity, ucs)
    t_f: float | None = None         # layer temperature (F) for the PVT (None: the IPR's)
    lp_corr: str = "Conventional"    # | "Synthetic Rock"
    rock: str = "Sandstone"          # | "Carbonate"
    lower_completion: str = "Cased and Perforated"   # | "Open Hole" | "Open Hole Perforated"
    perf_method: str = "Single Run"  # | "Double Run" | "Re-perforating"
    reperf: str = "New and Old holes"    # | "New holes only" | "Old holes only"
    old_gun: SpotGun | None = None
    old_perf_eff: float = 0.5
    invasion: str = "No invasion"    # | "Enter discrete invasion depth" | "Enter mud invasion log" | "Calculate invasion"
    inv_depth: float = 0.0           # in (discrete), filtrate zone
    solids_depth: float = 0.0        # in, solids-invaded zone (calculated with spurt, or entered)
    kl: float = 0.89                 # filtrate-zone permeability factor
    ks: float = 0.05                 # solids-zone permeability factor
    crush_model: str = "Entered"     # | "QinetiQ Test Results"
    non_darcy: bool = False
    standoff: float | None = None    # in (None: centralised)
    casing: tuple = ((6.7, 10.0, "J55", 6.41),)      # (OD in, lb/ft, grade, ID in) strings over the layer
    add_height: float = 0.0          # ft below the open hole (open-hole completions)
    gravel: dict | None = None       # {"type": "Internal"|"External", "k": md, "top", "bottom", "od", "id",
                                     #  "method": "Golan 1 phase"|"Geertsma 1 phase"|"Geertsma 2 phase"}
    sanding: dict | None = None      # {"d": in, "rho_s": lb/gal, "angle": deg, "rough": in}
    horizontal: dict | None = None   # {"h", "xw", "yw", "zw", "length", "width", "reh"} (horizontal wells)
    well_type: str = "Deviated"      # | "Vertical" | "Horizontal"
    sw: float | None = None          # water saturation entered instead of the water cut (Input Phase Ratios = No)


def generate_log(top, bottom, step=1.0, k=50.0, phi=0.18, ucs=8000.0, twc=None):
    """PROSPER's Generate (no log data available): constant rows every step ft from top to bottom inclusive."""
    import pandas as pd
    md = np.arange(top, bottom + 1e-6, step)
    df = pd.DataFrame({c: np.nan for c in LOG_COLS}, index=range(len(md)))
    df["MD (ft)"], df["Porosity"], df["Permeability (md)"], df["UCS (psi)"] = md, float(phi), float(k), float(ucs)
    df["TWC (psi)"] = TWC_UCS * float(ucs) if twc is None else float(twc)
    df["Net"], df["Perforated"] = 1.0, True
    return df


# ---- log Fill-in correlations (provisional: PROSPER's equations 1-7 are Shell's; published forms are used)
UCS_EQS = {"Sandstone": ["Equation 1 (density, compressional sonic)", "Equation 2 (density, shear sonic)",
                         "Equation 3 (porosity)", "Equation 4 (TWC)"],
           "Carbonate": ["Equation 5 (porosity)", "Equation 6 (TWC)"]}
TWC_EQS = {"Sandstone": ["Equation 1 (density, compressional sonic)", "Equation 2 (density, compressional sonic)",
                         "Equation 3 (density, shear sonic)", "Equation 4 (porosity)", "Equation 5 (UCS)"],
           "Carbonate": ["Equation 6 (porosity)", "Equation 7 (UCS)"]}
MPA = 145.0377


def ucs_relation(eq, phi=None, rho=None, dtc=None, dts=None, twc=None):
    """UCS (psi). Eq 1: Moos et al. (2003) from density and compressional sonic; Eq 2: Bradford et al. (1998)
    from the dynamic Young's modulus (density, shear sonic, Poisson 0.25); Eq 3: Vernik et al. (1993) clean
    sandstone; Eq 4: TWC / 2.5; Eq 5: Rzhevsky & Novick (1971) carbonate; Eq 6: TWC / 3."""
    n = eq.split()[1]
    if n == "1":
        vp = 304800.0 / dtc                               # m/s
        return MPA * 3.87 * np.exp(1.14e-10 * rho * 1000 * vp ** 2)
    if n == "2":
        vs = 304800.0 / dts
        e_gpa = 2 * rho * 1000 * vs ** 2 * 1.25 / 1e9     # E = 2 G (1 + nu), nu 0.25
        return MPA * np.maximum(2.28 + 4.1089 * e_gpa, 0.0)
    if n == "3":
        return MPA * 254 * np.maximum(1 - 2.7 * phi, 0.0) ** 2
    if n == "4":
        return twc / 2.5
    if n == "5":
        return MPA * 276 * np.maximum(1 - 3 * phi, 0.0) ** 2
    return twc / 3.0


def twc_relation(eq, phi=None, rho=None, dtc=None, dts=None, ucs=None):
    """TWC (psi): 2.5 x the sandstone UCS relations (Eqs 1-4 from the same data, 5 from UCS), 3 x for carbonate."""
    n = eq.split()[1]
    if n in ("1", "2"):
        return 2.5 * ucs_relation("Equation 1", rho=rho, dtc=dtc)
    if n == "3":
        return 2.5 * ucs_relation("Equation 2", rho=rho, dts=dts)
    if n == "4":
        return 2.5 * ucs_relation("Equation 3", phi=phi)
    if n == "5":
        return 2.5 * ucs
    if n == "6":
        return 3.0 * ucs_relation("Equation 5", phi=phi)
    return 3.0 * ucs


def fill_log(log, rock="Sandstone", ucs_eq=None, twc_eq=None, rho_matrix=2.65, rho_fluid=1.0, m=None, c=None):
    """PROSPER's Fill-in: compute the empty columns - porosity from FDC (oil / water wells), permeability from
    log10 k = m phi + c, UCS and TWC from the chosen relations. Returns (filled log, list of columns filled)."""
    df = log.copy()
    done = []
    num = lambda col: df[col].astype(float) if col in df else None
    if df["Porosity"].isna().any() and "FDC (g/cc)" in df and df["FDC (g/cc)"].notna().any():
        phi = (rho_matrix - df["FDC (g/cc)"].astype(float)) / (rho_matrix - rho_fluid)
        df["Porosity"] = df["Porosity"].fillna(phi.clip(0.0, 0.5))
        done.append("Porosity")
    if df["Permeability (md)"].isna().any() and m is not None and c is not None:
        df["Permeability (md)"] = df["Permeability (md)"].fillna(10 ** (m * df["Porosity"].astype(float) + c))
        done.append("Permeability (md)")
    kw = dict(phi=num("Porosity"), rho=num("FDC (g/cc)"), dtc=num("Compressional Sonic (us/ft)"),
              dts=num("Shear Sonic (us/ft)"))
    if df["UCS (psi)"].isna().any() and ucs_eq:
        df["UCS (psi)"] = df["UCS (psi)"].fillna(ucs_relation(ucs_eq, twc=num("TWC (psi)"), **kw))
        done.append("UCS (psi)")
    if df["TWC (psi)"].isna().any() and twc_eq:
        df["TWC (psi)"] = df["TWC (psi)"].fillna(twc_relation(twc_eq, ucs=num("UCS (psi)"), **kw))
        done.append("TWC (psi)")
    return df, done


# ---- mud invasion (PROSPER's "Calculate invasion" - provisional: static filtration through a Darcy filter cake)
def invasion_calc(method, md, tvd, phi, swi, rw_ft, pr_psig, mud_ppg, drill_days, down_days, **d):
    """(filtrate depth, solids depth) in inches for every log row. Overbalance = mud hydrostatic - layer pressure;
    each depth is exposed for the downtime plus the drilling time below it (equivalent constant rate of
    penetration). Filtrate volume per area V = spurt + m sqrt(dP t): m from the lab (Fluid loss vs time, Total
    fluid loss) scaled with sqrt(dP / dP_lab), or from the cake (k_cake 80e-21 m2 OBM / 160e-21 WBM, porosity
    0.05) for the no-data options. Radial piston-like invasion: pi ((rw+L)^2 - rw^2) phi (1 - Swi) = 2 pi rw V."""
    md, tvd, phi = (np.asarray(x, float) for x in (md, tvd, phi))
    dp = np.maximum(0.052 * mud_ppg * tvd - pr_psig, 0.0)                       # psi
    span = max(md.max() - md.min(), 1e-9)
    t = (down_days + drill_days * (md.max() - md) / span) * 86400.0            # s
    spurt = 0.0
    if method == "Fluid loss vs time":
        tt, vv = np.asarray(d["times"], float), np.asarray(d["losses"], float)     # s, cm3
        A = np.vstack([np.ones_like(tt), np.sqrt(tt)]).T
        (sp_, mm), *_ = np.linalg.lstsq(A, vv / d["area"], rcond=None)            # cm, cm/s^0.5
        spurt, m = max(sp_, 0.0), mm / np.sqrt(d["dp_lab"])
    elif method == "Total fluid loss time and volume":
        m = d["loss"] / d["area"] / np.sqrt(d["time"]) / np.sqrt(d["dp_lab"])
    else:
        obm = d.get("obm", False)
        kc = (80e-21 if obm else 160e-21)                                           # m2
        if method == "No fluid loss data - option 2":
            rho_s = 4.2 if d.get("weight", "Barite") == "Barite" else 2.7
            brine = d.get("brine_sg", 1.1)
            liq = (0.8 * 0.8 + 0.2 * brine) if (obm and d.get("owr", "80/20") == "80/20") else \
                  ((0.75 * 0.8 + 0.25 * brine) if obm else brine)
            fsm = np.clip((mud_ppg / 8.345 - liq) / (rho_s - liq), 0.01, 0.6)
            mu = d.get("mu", 2.0 if obm else 0.5)
        else:
            rho_s, w = d["rho_solids"], d["solids_frac"]                          # sg, mass fraction
            fsm = np.clip(w * (mud_ppg / 8.345) / rho_s, 0.01, 0.6)
            mu = d["mu"]
        fsc = 0.95
        # V/A = sqrt(2 kc dP t fsm / (mu (fsc - fsm)))   (SI) -> cm, per sqrt(psi s)
        m = np.sqrt(2 * kc * 6894.76 * fsm / (mu * 1e-3 * (fsc - fsm))) * 100.0
    v = spurt + m * np.sqrt(dp * t)                                              # cm3 / cm2 = cm
    rw = rw_ft * 30.48
    fil = np.sqrt(rw ** 2 + 2 * rw * v / np.maximum(phi * (1 - swi), 1e-6)) - rw
    sol = np.sqrt(rw ** 2 + 2 * rw * spurt / np.maximum(phi, 1e-6)) - rw if spurt > 0 else np.zeros_like(fil)
    return fil / 2.54, np.minimum(sol, fil) / 2.54


def return_perm_factor(loss_cm3, area_cm2, phi, length_cm, swi, k_return):
    """Filtrate-zone permeability factor from a return-permeability core test: the filtrate fills a length
    x = V / (A phi (1 - Swi)) of the plug; the measured return permeability is the series average."""
    x = min(loss_cm3 / (area_cm2 * phi * (1 - swi)), length_cm)
    kr = k_return
    if x <= 1e-9:
        return 1.0
    return float(np.clip(x / (length_cm / kr - (length_cm - x)), 1e-4, 1.0))


# ---- QinetiQ sanding (provisional) -----------------------------------------------------------------------
def sand_produced(v_ft_s, rho_f, mu_cp, d_in, rho_s_ppg, angle, rough_in, rp_in, vertical_well):
    """True when the tunnel-mouth velocity moves failed sand: horizontal perforations (angle < 45) by traction
    (Shields 0.047, log-law velocity over the roughness), vertical ones by suspension (settling velocity, Schiller-
    Naumann drag). Vertical wells use the horizontal model; deviated / horizontal wells the model of the angle."""
    g, d = 9.81, d_in * 0.0254
    rf, rs = rho_f * 16.0185, rho_s_ppg * 119.826
    mu = mu_cp * 1e-3
    u_star = np.sqrt(0.047 * (rs - rf) * g * d / rf)
    rh = max(rp_in * 0.0254 / 2, 1e-4)
    ks = max(rough_in * 0.0254, d)
    u_trac = u_star * 5.75 * np.log10(max(12 * rh / ks, 1.01))
    w = (rs - rf) * g * d ** 2 / (18 * mu)
    for _ in range(30):
        re = rf * w * d / mu
        cd = 24 / max(re, 1e-9) * (1 + 0.15 * re ** 0.687)
        w = np.sqrt(4 * (rs - rf) * g * d / (3 * cd * rf))
    v = v_ft_s * 0.3048
    horiz = vertical_well or angle < 45
    return bool(v > u_trac) if horiz else bool(v > w)


def _so_lp(s):
    return np.exp(-SO_LP[0] * s - SO_LP[1] * s * s)


def _so_ehd(s):
    return 1 + SO_EHD[0] * s + SO_EHD[1] * s * s + SO_EHD[2] * s ** 3


def corey(sw, rp):
    sor, swr, ko_e, kw_e, no, nw = rp
    den = 1 - sor - swr
    so_n = np.clip((1 - sw - sor) / den, 0, 1)
    sw_n = np.clip((sw - swr) / den, 0, 1)
    return ko_e * so_n ** no, kw_e * sw_n ** nw


def gun_scale(gun: SpotGun):
    """The gun's penetration relative to Tutorial 09's: API section 1 Lp over ln(LP_UC / its test cement strength)
    (Run 14: the database's normalised penetration was 5.5 % low, this exact)."""
    ref = SpotGun()
    return (gun.lp_api / np.log(LP_UC / gun.cement_strength)) / (ref.lp_api / np.log(LP_UC / ref.cement_strength))


def penetration(gun: SpotGun, ucs, sigma_v, p_psig, standoff=None, outer_walls=(), synthetic=False):
    """Downhole Lp (in): the gun's normalised penetration corrected for rock strength and effective stress (Tutorial
    09), for the downhole standoff against the gun's test standoff (relative to Tutorial 09's 2.21 / 0.5 in, which
    the strength-stress fit carries), for extra casing strings, and the Synthetic Rock correlation (runs 1-3)."""
    s = np.maximum(sigma_v - LP_ALPHA * p_psig, 1.0)
    if synthetic:
        lp = SYNTHETIC[0] * LP_C * gun_scale(gun) * np.log(LP_U0 / ucs) * np.log(SYNTHETIC[1] / s)
    else:
        lp = LP_C * gun_scale(gun) * np.log(LP_U0 / ucs) * np.log(LP_S0 / s)
    so = SO_REF[0] if standoff is None else standoff
    lp = lp * (_so_lp(so) / _so_lp(gun.test_standoff)) / (_so_lp(SO_REF[0]) / _so_lp(SO_REF[1]))
    for w in outer_walls:
        lp = TWO_STRINGS[0] * lp - TWO_STRINGS[1] * w
    return np.maximum(lp, 0.01)


def entry_hole(gun: SpotGun, standoff=None, grade="J55", outer_walls=(), cased=True):
    """Downhole entry-hole diameter (in): API section 1 x 0.97509 (Tutorial 09) corrected for standoff against the
    test standoff, casing grade (P110 0.849 of J55) and extra casing strings."""
    so = SO_REF[0] if standoff is None else standoff
    e = gun.ehd * EHD_FACTOR * (_so_ehd(so) / _so_ehd(gun.test_standoff)) / (_so_ehd(SO_REF[0]) / _so_ehd(SO_REF[1]))
    if cased:
        e *= GRADE_EHD.get(str(grade).upper(), 1.0)
    return e * TWO_STRINGS[2] ** len(outer_walls)


class Spot:
    def __init__(self, f: SpotInputs, mob_fn):
        """mob_fn(p psia array) -> (1/(mu_o Bo), 1/(mu_w Bw), Pb psia)."""
        self.f = f
        self.mob_fn = mob_fn
        md_s, tvd_s = np.array(f.survey, float).T
        if f.log is not None and len(f.log):
            lg = f.log[(f.log["MD (ft)"] >= f.top_md - 1e-6) & (f.log["MD (ft)"] <= f.bottom_md + 1e-6)]
            if len(lg) == 0:
                raise ValueError(f"SPOT: no log rows between {f.top_md:g} and {f.bottom_md:g} ft MD.")
            lg = lg.sort_values("MD (ft)")
            self.md = lg["MD (ft)"].to_numpy(float)
            col = lambda c, dflt: lg[c].astype(float).fillna(dflt).to_numpy() if c in lg else np.full(len(lg), dflt)
            self.k_row = col("Permeability (md)", f.k)
            self.phi_row = col("Porosity", f.porosity)
            self.ucs_row = col("UCS (psi)", f.ucs)
            self.twc_row = col("TWC (psi)", 2.5 * f.ucs)
            cal = col("Caliper (in)", np.nan)
            self.rw_row = np.where(np.isfinite(cal) & (cal > 0), cal / 24.0, f.rw)
            self.inv_row = col("Invasion Depth (in)", 0.0)
            self.tick = lg["Perforated"].fillna(True).astype(bool).to_numpy() if "Perforated" in lg else \
                np.ones(len(lg), bool)
        else:
            n = int(round(f.bottom_md - f.top_md)) + 1
            self.md = f.top_md + np.arange(n)
            self.k_row, self.phi_row, self.ucs_row = (np.full(n, v) for v in (f.k, f.porosity, f.ucs))
            self.twc_row, self.rw_row = np.full(n, 2.5 * f.ucs), np.full(n, f.rw)
            self.inv_row, self.tick = np.zeros(n), np.ones(n, bool)
        self.k_mean = float(np.mean(self.k_row))
        self.tvd = np.interp(self.md, md_s, tvd_s, left=None, right=None)
        if len(md_s) >= 2:
            dt = np.interp(f.bottom_md, md_s, tvd_s) - np.interp(f.top_md, md_s, tvd_s)
            self.cos = float(np.clip(dt / max(f.bottom_md - f.top_md, 1e-9), 1e-6, 1.0)) if f.deviated else 1.0
        else:
            self.cos = 1.0
        self.h_tvd = (f.bottom_md - f.top_md) * self.cos
        self.A = np.log(f.re / f.rw) - 0.75
        self.holdup = f.bottom_md if f.holdup_md is None else float(f.holdup_md)
        self.open_hole = f.lower_completion == "Open Hole"
        self.perf = (self.md <= self.holdup + 1e-9) & (self.tick | self.open_hole)
        if self.open_hole:
            self.perf = np.ones(len(self.md), bool)
        self.n_perf = int(self.perf.sum())
        if self.n_perf == 0:
            raise ValueError(f"SPOT: no perforated rows in the layer {f.top_md:g}-{f.bottom_md:g} ft MD.")
        self.first = int(np.argmax(self.perf))
        self._geo = None

    # shot sets: (gun, efficiency, multiplier). PROSPER's Double Run and Re-perforating (old holes ticked as
    # Existing Perforation) give the Single Run result (runs 6a-6c identical to the base; 6d does not calculate).
    def shot_sets(self):
        f = self.f
        return [(f.gun, f.perf_eff, 1.0)]

    def standoff_rows(self):
        """Downhole standoff (in) by row: entered, or the gun centred in the innermost casing ID (open hole
        perforated: the hole; external gravel pack rows: the underreamed diameter)."""
        f, gun = self.f, self.f.gun
        if f.lower_completion == "Cased and Perforated" and f.casing:
            inner = min(f.casing, key=lambda c: c[0])
            so = np.full(len(self.md), max((inner[3] - gun.gun_od) / 2, 0.0))
        else:
            so = np.maximum(self.rw_row * 24 - gun.gun_od, 0.0) / 2
        gp = f.gravel
        if gp and gp.get("type") == "External":
            in_gp = (self.md >= gp["top"] - 1e-6) & (self.md < gp["bottom"] - 1e-6)
            so = np.where(in_gp, max((gp["od"] - gun.gun_od) / 2, 0.0), so)
        if f.standoff is not None:
            so = np.full(len(self.md), float(f.standoff))
        return so

    def invasion_depths(self):
        f = self.f
        if f.invasion == "No invasion":
            z = np.zeros(len(self.md))
            return z, z
        if f.invasion == "Enter discrete invasion depth":
            return np.full(len(self.md), f.inv_depth), np.full(len(self.md), min(f.solids_depth, f.inv_depth))
        # mud invasion log, or calculated (written to the log's invasion column by the caller)
        return self.inv_row, np.minimum(np.full(len(self.md), f.solids_depth), self.inv_row)

    def log(self):
        f = self.f
        p_g = f.pr - G
        sv = f.ob_grad * self.tvd
        sets = self.shot_sets()
        n_tot = sum(g.spf * e * m for g, e, m in sets)
        so = self.standoff_rows()
        cased = f.lower_completion == "Cased and Perforated"
        strings = sorted(f.casing, key=lambda c: c[0]) if cased and f.casing else []
        walls = tuple((c[0] - c[3]) / 2 for c in strings[1:])
        grade = strings[0][2] if strings else "J55"
        gp = f.gravel
        ext = np.zeros(len(self.md), bool)
        if gp and gp.get("type") == "External":
            ext = (self.md >= gp["top"] - 1e-6) & (self.md < gp["bottom"] - 1e-6)
        lp = sum(g.spf * e * m * np.where(ext, penetration(g, self.ucs_row, sv, p_g, so, (), f.lp_corr != "Conventional"),
                                          penetration(g, self.ucs_row, sv, p_g, so, walls, f.lp_corr != "Conventional"))
                 for g, e, m in sets) / n_tot
        ehd = sum(g.spf * e * m * np.where(ext, entry_hole(g, so, grade, (), False), entry_hole(g, so, grade, walls, cased))
                  for g, e, m in sets) / n_tot
        ehd = np.asarray(ehd, float) * np.ones(len(lp))
        if self.holdup <= f.bottom_md + 1e-9:
            ehd[np.abs(self.md - self.holdup) < 0.5] *= HOLDUP_EHD
        phasing = kt_phasing(sets[0][0].phasing)
        rp_in = ehd / 2
        n_eff = n_tot / self.cos
        kcf, kct = (f.crush_factor, f.crush_thickness) if f.crush_model == "Entered" else QINETIQ_CRUSH
        ld_in, ls_in = self.invasion_depths()
        kd_fac = np.where(ld_in > 0, f.kl, 1.0)
        rw = self.rw_row
        rw_gp = np.where(ext, np.maximum(gp["od"] / 24.0, rw), rw) if ext.any() else rw.copy()
        sp, sc, sd = (np.zeros(len(lp)) for _ in range(3))
        mult = np.ones(len(lp))
        beyond = np.ones(len(lp), bool)
        for i in range(len(lp)):
            ld, kd = ld_in[i] / 12, kd_fac[i] * self.k_row[i]
            if self.open_hole:
                sd[i] = _two_zone_hawkins(self.k_row[i], rw_gp[i], ls_in[i] / 12, f.ks * self.k_row[i], ld, kd)
                lp[i], ehd[i], beyond[i] = 0.0, 0.0, ld <= 0
                continue
            base = wcomp.kt_perforation_skin(rw_gp[i], lp[i] / 12, rp_in[i] / 12, n_eff, phasing, f.kv_kh)["perforation"]
            sp[i] = base
            beyond[i] = ld <= 0 or lp[i] / 12 > ld
            if ld > 0 and not beyond[i]:
                # PROSPER (runs 7, 8): two-zone Hawkins damage skin; the perforation skin x M, M = 1 + Sd / ln(rd/rw)
                sd[i] = _two_zone_hawkins(self.k_row[i], rw_gp[i], ls_in[i] / 12, f.ks * self.k_row[i], ld, kd)
                mult[i] = 1 + sd[i] / np.log((rw_gp[i] + ld) / rw_gp[i])
            elif ld > 0:              # perforations past the invaded zone: Karakas & Tariq (not calibrated)
                tot = wcomp.karakas_tariq(self.k_row[i], rw_gp[i], lp[i] / 12, rp_in[i] / 12, n_eff, phasing,
                                          f.kv_kh, ld, kd)["total"]
                sd[i] = tot - base + (_two_zone_hawkins(self.k_row[i], rw_gp[i], ls_in[i] / 12, f.ks * self.k_row[i],
                                                        ls_in[i] / 12, f.ks * self.k_row[i]) if ls_in[i] > 0 else 0.0)
            sc[i] = (1 / kcf - 1) * np.log((rp_in[i] + kct) / rp_in[i]) / (n_eff * lp[i] / 12)
        sgp = self.gravel_skin(rp_in, n_eff, lp)
        sdev = self.deviation_skin()
        off = ~self.perf
        lp, ehd, sp, sc, sd, sgp = (np.where(off, 0.0, x) for x in (lp, ehd, sp, sc, sd, sgp))
        dk = KING[0] * self.k_row ** -KING[1]
        db = BEHRMANN[0] * self.phi_row * self.k_row ** -BEHRMANN[1]
        import pandas as pd
        return pd.DataFrame({
            "Measured depth (ft)": self.md, "True vertical depth (ft)": self.tvd,
            "Porosity": self.phi_row, "Permeability (md)": self.k_row, "UCS (psi)": self.ucs_row,
            "TWC (psi)": self.twc_row, "Perforated": self.perf,
            "Penetration Lp (in)": lp, "Min Lp (in)": np.maximum(MINLP[0] * lp + MINLP[1], 0.0),
            "Entry hole diameter (in)": ehd,
            "Reservoir pressure (psig)": p_g, "Underbalance Behrmann (psig)": p_g - db,
            "Underbalance King (psig)": p_g - dk, "Underbalance user (psig)": p_g - f.underbalance - G,
            "Completion fluid density (lb/gal)": (p_g - f.underbalance) / (0.0519481 * self.tvd),
            "Perforation skin": sp, "Damage skin": sd, "Damage, perf, crush skin": mult * sp + sc + sd,
            "Crush skin": sc, "Perforation skin multiplier": mult, "Mean standoff (in)": so,
            "Gravel pack skin": sgp,
            "Deviation skin": np.where(off, 0.0, sdev), "Partial penetration skin": np.where(off, 0.0, self.partial_skin()),
            "Invasion depth (in)": ld_in, "Beyond Zone": beyond & self.perf,
            "Near-wellbore effective vertical stress (psig)": sv - (p_g - f.underbalance) - G})

    def gravel_skin(self, rp_in, n_eff, lp_in):
        """Gravel-pack skin by row. Internal (PROSPER Run 10a): Darcy flow along every packed tunnel of length
        Lp + (rw - casing ID / 2), S = k L / (C k_gp n A_p), C 0.24834 (k_gp 1.25e5 md), top <= MD < bottom.
        External (underreamed; not calibrated): radial flow through the pack, (k / k_gp) ln(r_gp / r_screen)."""
        f, gp = self.f, self.f.gravel
        out = np.zeros(len(self.md))
        if not gp:
            return out
        in_gp = (self.md >= gp["top"] - 1e-6) & (self.md < gp["bottom"] - 1e-6)
        if gp.get("type") == "External":
            r_o, r_i = gp["od"] / 24.0, max(gp.get("id", gp["od"] * 0.6), 0.1) / 24.0
            out[in_gp] = self.k_row[in_gp] / gp["k"] * np.log(max(r_o / r_i, 1.0))
            return out
        cas_id = min(f.casing, key=lambda c: c[0])[3] if f.casing else gp["od"]
        L = (np.asarray(lp_in) + np.maximum(self.rw_row * 12 - cas_id / 2, 0.0)) / 12
        ap = np.pi * (rp_in / 12) ** 2
        out[in_gp] = (self.k_row * L / (GP_C * gp["k"] * n_eff * ap))[in_gp]
        return out

    def _geometry(self):
        """(deviation skin, partial-penetration skin). Fully perforated: Cinco-Ley's deviation correlation on the
        segment-count thickness. Gun held up / rows not perforated: the slanted, partially-penetrating well of
        Cinco-Ley, Ramey & Miller on the perforated span (penetration = the same interval vertical, deviation =
        the rest). Open hole: the additional reservoir height below the well is added to the thickness."""
        if self._geo is not None:
            return self._geo
        f = self.f
        if f.well_type == "Horizontal":             # along-hole flow: the horizontal well model carries the geometry
            self._geo = (0.0, 0.0)
            return self._geo
        dev = f.deviated and self.cos < 1 - 1e-9
        th = np.degrees(np.arccos(self.cos)) if dev else 0.0
        h = len(self.md) * self.cos
        extra = f.add_height if f.lower_completion != "Cased and Perforated" else 0.0
        if self.n_perf >= len(self.md) and extra <= 0:
            sd = wcomp.deviation_skin(min(th, 75.0), h, f.rw, f.kv_kh, True) if dev else 0.0
            self._geo = (sd, 0.0)
        else:
            span = int(np.nonzero(self.perf)[0][-1]) - self.first + 1
            r = wcomp.cinco_slant_pp(min(th, 75.0), h + extra, max(span, 1) * self.cos, self.first * self.cos,
                                     f.rw, f.kv_kh)
            self._geo = (r["slant"], r["penetration"])
        return self._geo

    def deviation_skin(self):
        return self._geometry()[0]

    def partial_skin(self):
        return self._geometry()[1]

    def _factor(self):
        c, kv = self.cos, self.f.kv_kh
        fac = c ** AGG[3] * (c * c + (1 - c * c) * kv) ** AGG[4]
        if self.n_perf < len(self.md):            # part of the layer not perforated: mechanical skin on that part
            fac *= len(self.md) / max(self.n_perf, 1) * (PP_MECH[0] - PP_MECH[1] * (1 - c))
        return fac

    def _row_mech(self, lg):
        """Row skin entering the layer total: PROSPER's weights - Tutorial 09's for the perforation and crush skins,
        0.98693 for the invaded rows' M x perforation and damage skins (runs 7, 8), cos(deviation) for the gravel-pack
        skin (Run 10a)."""
        m = lg["Perforation skin multiplier"]
        dam = np.where(m > 1, AGG_D * (m * lg["Perforation skin"] + lg["Damage skin"]),
                       AGG[0] * lg["Perforation skin"] + AGG_D * lg["Damage skin"])
        return self._factor() * (dam + AGG[1] * lg["Crush skin"]) + self.cos * lg["Gravel pack skin"]

    def total_skin(self, lg=None):
        lg = self.log() if lg is None else lg
        m = self.perf
        w = self.k_row[m] / self.k_row[m].sum()                  # permeability-weighted over the perforated rows
        return float((self._row_mech(lg)[m] * w).sum() + AGG[2] * self.deviation_skin() + self.partial_skin())

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

    def denominator(self, s_tot):
        """A + S of the radial inflow, or the horizontal well's equivalent (Economides-Joshi, Borisov, Babu-Odeh,
        Goode-Wilkinson) expressed on k h of the layer."""
        f = self.f
        if f.well_type != "Horizontal" or not f.horizontal:
            return self.A + s_tot
        hz = f.horizontal
        step = float(np.median(np.diff(self.md))) if len(self.md) > 1 else 1.0
        L = max((self.n_perf if f.lower_completion != "Open Hole" else len(self.md)) * step, 1.0)
        h, k = hz["h"], self.k_mean
        if f.inflow == "Borisov":
            reh = hz.get("reh", f.re)
            beta = 1 / np.sqrt(f.kv_kh)
            D = np.log(4 * reh / L) + beta * h / L * np.log(h / (2 * np.pi * f.rw)) + beta * h / L * s_tot
            return D * (k * self.h_tvd) / (k * h)
        from . import inflow_adv as ia
        meth = {"Economides-Joshi": "Joshi", "Babu-Odeh": "Babu & Odeh", "Goode-Wilkinson": "Kuchuk & Goode"}[f.inflow]
        lx, ly = hz.get("length", 2 * f.re), hz.get("width", 2 * f.re)
        kh_eq, D = ia.horizontal_resistance(meth, k, h, f.rw, 1.0, f.kv_kh, L, lx, ly, hz.get("xw", lx / 2),
                                            hz.get("yw", ly / 2), hz.get("zw", h / 2), s_tot)
        return D * (k * self.h_tvd) / kh_eq

    def oil_rate_fn(self, s_tot, h=None, D_nd=0.0):
        f = self.f
        _, kro, _ = self.saturation()
        h = self.h_tvd if h is None else h
        den = self.denominator(s_tot)
        C = 0.00708 * self.k_mean * kro * h / den
        pr = f.pr
        grid = np.linspace(14.7, pr, 400)
        mo, _, pb = self.mob_fn(grid)
        pb = float(pb)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (mo[1:] + mo[:-1]) * np.diff(grid))])
        I_to_pr = lambda p: cum[-1] - np.interp(p, grid, cum)
        if pr <= pb:
            m_r = float(np.interp(pr, grid, mo))
            if f.inflow == "Vogel":
                q0 = lambda p: C * m_r * pr / 1.8 * (1 - 0.2 * np.clip(p, 0, pr) / pr - 0.8 * (np.clip(p, 0, pr) / pr) ** 2)
            else:
                q0 = lambda p: C * m_r * (pr ** 2 - np.clip(p, 0, pr) ** 2) / (2 * pr)
        else:
            qb = C * float(I_to_pr(pb))
            m_b = float(np.interp(pb, grid, mo))
            j_avg = qb / (pr - pb)

            def q0(p):
                p = np.clip(np.asarray(p, float), 0, pr)
                above = C * I_to_pr(np.maximum(p, pb))
                if f.inflow == "Vogel":
                    below = qb + j_avg * pb / 1.8 * (1 - 0.2 * p / pb - 0.8 * (p / pb) ** 2)
                else:
                    below = qb + C * m_b * (pb ** 2 - p ** 2) / (2 * pb)
                return np.where(p >= pb, above, below)
        if D_nd <= 0:
            return q0
        # rate-dependent skin D q:  q (den + D q) = den q0  ->  q = (-den + sqrt(den^2 + 4 D den q0)) / (2 D)
        Dn = D_nd / (self.k_mean * kro * h) * (self.k_mean * self.h_tvd)          # on the same k h basis
        return lambda p: (-den + np.sqrt(den ** 2 + 4 * Dn * den * np.maximum(q0(p), 0.0))) / (2 * Dn)

    def non_darcy_D(self, lg):
        """Rate-dependent skin coefficient (1 / (STB/d)) - provisional: Forchheimer flow through the crushed zone of
        every tunnel (McLeod) when Calculate Non-Darcy Skin is Yes, and through the packed tunnels of an internal
        gravel pack (Golan: beta = 1.47e7 k^-0.55; Geertsma: 48511 / (phi^5.5 k^0.5), / (1 - Sw)^5.5 two-phase)."""
        f = self.f
        m = self.perf & (lg["Penetration Lp (in)"] > 0)
        if not m.any():
            return 0.0
        mo, _, _ = self.mob_fn(np.array([f.pr]))
        mu_b = 1 / float(mo[0])                                    # mu B (cp rb/STB) at Pr
        bo, rho = 1.2, 50.0                                        # rb/STB, lb/ft3 (screening values)
        kh = self.k_mean * self.h_tvd
        n_eff = sum(g.spf * e * mm for g, e, mm in self.shot_sets()) / self.cos
        N = n_eff * self.h_tvd * self.n_perf / len(self.md)
        lp = float(lg["Penetration Lp (in)"][m].mean()) / 12
        rp = float(lg["Entry hole diameter (in)"][m].mean()) / 24
        D = 0.0
        if f.non_darcy and lp > 0:
            kcf, kct = (f.crush_factor, f.crush_thickness) if f.crush_model == "Entered" else QINETIQ_CRUSH
            beta = 2.33e10 / max(kcf * self.k_mean, 1e-6) ** 1.201
            rc = rp + kct / 12
            a = 9.08e-13 * beta * bo ** 2 * rho * (1 / rp - 1 / rc) / (2 * np.pi * lp) ** 2 / N ** 2
            D += a * kh / (141.2 * mu_b)
        gp = f.gravel
        if gp and gp.get("type") != "External" and f.non_darcy:    # PROSPER Run 10a: Darcy only otherwise
            kg = gp["k"]
            meth = gp.get("method", "Golan 1 phase")
            beta = 1.47e7 / kg ** 0.55 if meth.startswith("Golan") else 48511 / (0.35 ** 5.5 * kg ** 0.5)
            if meth == "Geertsma 2 phase":
                beta /= max(1 - self.saturation()[0], 0.05) ** 5.5
            cas_id = f.casing[0][3] if f.casing else gp["od"]
            L = max(f.rw - cas_id / 24.0, 1.0 / 12)
            ap = np.pi * rp ** 2
            a = 9.08e-13 * beta * bo ** 2 * rho * L / (N * ap) ** 2
            D += a * kh / (141.2 * mu_b)
        return float(D)

    def results(self, pwf_log=None, gor=None):
        f = self.f
        lg = self.log()
        s = self.total_skin(lg)
        sw, kro, krw = self.saturation()
        D = self.non_darcy_D(lg)
        qo = self.oil_rate_fn(s, D_nd=D)
        n = len(self.md)
        fac = n / max(n - 1, 1) if self.n_perf >= n else 1.0     # PROSPER's layer table (see the docstring)
        aof_layer = float(qo(0.0)) * fac
        p_half = (f.pr - G) / 2                     # PROSPER's layer PI: oil rate at half the layer pressure (gauge)
        pi_half = float(qo(p_half + G)) / p_half
        A_w = self.A + WIQI_DA
        layer = {"Top MD (ft)": f.top_md, "Bottom MD (ft)": f.bottom_md,
                 "Layer pressure (psig)": f.pr - G, "Water cut (%)": 100 * f.wc,
                 "Completion fluid density (lb/gal)": (f.pr - G - f.underbalance) /
                 (0.0519481 * float(np.mean([self.tvd[0], self.tvd[self.perf][-1]]))),
                 "PI (STB/d/psi)": pi_half, "Net PI (STB/d/psi)": pi_half,
                 "Total skin": s, "Deviation skin": self.deviation_skin(),
                 "Partial penetration skin": self.partial_skin(),
                 "Non-Darcy coefficient (1/(STB/d))": D,
                 "WIQI": (A_w + self.deviation_skin() + self.partial_skin()) / (A_w + s),
                 "Oil AOF (STB/d)": aof_layer,
                 "Water saturation": sw, "kro": kro, "krw": krw, "ko (md)": self.k_mean * kro,
                 "kw (md)": self.k_mean * krw,
                 "Mean penetration Lp (in)": float(lg["Penetration Lp (in)"][self.perf].mean()),
                 "Perforated rows": self.n_perf, "Log rows": n,
                 "Rows beyond invasion zone": int(lg["Beyond Zone"].sum()),
                 "TVD thickness (ft)": self.h_tvd}
        # each row's share of the layer rate: k / (A + its skin)
        w = np.where(self.perf, self.k_row / (self.A + self._row_mech(lg) + AGG[2] * self.deviation_skin()
                                              + self.partial_skin()), 0.0)
        w = w / w.sum()
        wor = f.wc / max(1 - f.wc, 1e-12)
        lg["Qo AOF (STB/d)"] = aof_layer * w
        lg["Qw AOF (STB/d)"] = lg["Qo AOF (STB/d)"] * wor
        if pwf_log is not None:
            q_tot = float(qo(pwf_log)) * fac
            lg["Qo (STB/d)"] = q_tot * w
            lg["Qw (STB/d)"] = lg["Qo (STB/d)"] * wor
            if gor:
                lg["Qg (MMscf/d)"] = lg["Qo (STB/d)"] * gor / 1e6
            lg["Qo Cum (STB/d)"] = lg["Qo (STB/d)"][::-1].cumsum()[::-1]
            lg["Qw Cum (STB/d)"] = lg["Qw (STB/d)"][::-1].cumsum()[::-1]
            if f.sanding:
                lg["Sand Production"] = self.sanding_flags(lg, pwf_log)
        lg["Qo Cum AOF (STB/d)"] = lg["Qo AOF (STB/d)"][::-1].cumsum()[::-1]
        return {"layer": layer, "log": lg, "oil_rate": qo, "skin": s}

    def sanding_flags(self, lg, pwf):
        f, sd = self.f, self.f.sanding
        mo, mw, _ = self.mob_fn(np.array([f.pr]))
        mu = 1 / float(mo[0]) / 1.2
        rho = 50.0 * (1 - f.wc) + 62.4 * f.wc
        n_eff = sum(g.spf * e * mm for g, e, mm in self.shot_sets()) / self.cos
        out = []
        for _, r in lg.iterrows():
            if not r["Perforated"] or r["Entry hole diameter (in)"] <= 0:
                out.append(False)
                continue
            q_rb = (r["Qo (STB/d)"] * 1.2 + r["Qw (STB/d)"]) * 5.615 / 86400        # ft3/s
            a = n_eff * np.pi * (r["Entry hole diameter (in)"] / 24) ** 2
            out.append(sand_produced(q_rb / max(a, 1e-9), rho, mu, sd["d"], sd["rho_s"], sd["angle"], sd["rough"],
                                     r["Entry hole diameter (in)"] / 2, f.well_type == "Vertical"))
        return out


def _two_zone_hawkins(k, rw, ls, ks, ld, kd):
    """Hawkins skin of a solids zone (rw..rw+ls, ks) inside a filtrate zone (..rw+ld, kd)."""
    s = 0.0
    if ls > 0 and ks > 0:
        s += (k / ks - 1) * np.log((rw + ls) / rw)
    if ld > ls and kd > 0:
        s += (k / kd - 1) * np.log((rw + ld) / (rw + max(ls, 0.0)))
    return float(s)


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


def spot_ipr(pr, wc, f, model, t_f, pwf_log=None, gor=None):
    """IPRCurve (liquid, STB/d against psia) of the SPOT well: f is one layer (SpotInputs) or a list of layers.
    pr (psia) and wc (fraction) are the first layer's; a change of either (sensitivities) shifts every layer's
    pressure by the same amount and sets every layer's water cut. Layer rates add (no crossflow)."""
    import dataclasses
    import pandas as pd
    from .inflow import IPRCurve
    layers = list(f) if isinstance(f, (list, tuple)) else [f]
    dp = float(pr) - layers[0].pr
    wc_new = None if abs(float(wc) - layers[0].wc) < 1e-12 else float(wc)
    lays = [dataclasses.replace(L, pr=L.pr + dp, wc=L.wc if wc_new is None else wc_new) for L in layers]
    for i, L in enumerate(lays):                 # water saturation entered: the water cut it gives at Pr
        if L.sw is not None and wc_new is None:
            mo, mw, _ = blackoil_mobility(model, t_f if L.t_f is None else L.t_f)(np.array([L.pr]))
            kro, krw = corey(L.sw, L.relperm)
            qw, qo_ = krw * mw[0], kro * mo[0]
            lays[i] = dataclasses.replace(L, wc=float(qw / max(qw + qo_, 1e-30)), sw=None)
    res, rates = [], []
    for L in lays:
        sp = Spot(L, blackoil_mobility(model, t_f if L.t_f is None else L.t_f))
        r = sp.results(pwf_log, gor)
        res.append(r)
        wcL = L.wc
        rates.append((r["oil_rate"], wcL, L.pr))
    pmax = max(L.pr for L in lays)

    def fl(p):
        p = np.asarray(p, float)
        return sum(np.where(p < prL, np.maximum(qo(np.minimum(p, prL)), 0.0), 0.0) / max(1 - w, 1e-9)
                   for qo, w, prL in rates)
    pw = np.linspace(pmax, 0.0, 41)
    q = np.asarray(fl(pw), float)
    lay0 = res[0]["layer"]
    s_txt = ", ".join(f"{r['layer']['Total skin']:.4f}" for r in res)
    qo_aof = sum(float(qo(0.0)) for qo, _, _ in rates)
    ql_aof = float(fl(0.0))
    wc_tot = 1 - qo_aof / ql_aof if ql_aof > 0 else lays[0].wc
    c = IPRCurve("SPOT", "oil", pmax, pw, q,
                 params={"Total skin": lay0["Total skin"], "WIQI": lay0["WIQI"], "kro": lay0["kro"],
                         "Water saturation": lay0["Water saturation"], "Layers": len(lays),
                         "Water cut at AOF (%)": 100 * wc_tot},
                 notes=[f"SPOT: {len(lays)} layer(s), total skin {s_txt}; water saturation "
                        f"{lay0['Water saturation']:.4f} for the water cut, ko {lay0['ko (md)']:.2f} md (layer 1)."],
                 rate_fn=fl)
    c.spot_layer = lay0
    c.spot_layers = pd.DataFrame([r["layer"] for r in res], index=pd.RangeIndex(1, len(res) + 1, name="Layer"))
    c.spot_log = pd.concat([r["log"].assign(Layer=i + 1) for i, r in enumerate(res)], ignore_index=True) \
        if len(res) > 1 else res[0]["log"]
    return c
