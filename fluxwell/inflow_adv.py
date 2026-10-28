"""Advanced inflow models: fractured, horizontal, dual porosity, multilayer,
horizontal with friction, and modified isochronal gas tests.

All models reduce to an "equivalent radial" resistance where possible:
    oil  J = kh_eq / (141.2 mu Bo D)                         (Vogel below Pb)
    gas  Pr^2 - Pwf^2 = 1422 T mu Z / kh_eq * (D q + D_nd q^2)
with D the dimensionless resistance (drainage + skin terms).

Methods and sources
-------------------
Hydraulically fractured well
    early/transient: Gringarten, Ramey & Raghavan (1974) infinite-conductivity
    solution  pD = 0.5 sqrt(pi tD)[erf(0.134/sqrt tD) + erf(0.866/sqrt tD)]
                   - 0.067 Ei(-0.018/tD) - 0.433 Ei(-0.75/tD)   (tD on xf)
    finite conductivity: Cinco-Ley & Samaniego (1981) equivalent skin,
    s_f = ln(rw/xf) + f(FcD), f(u = ln FcD) = (1.65 - 0.328u + 0.116u^2)/
          (1 + 0.180u + 0.064u^2 + 0.005u^3)
    The transient term adds f(FcD) - ln 2; bilinear flow is not modelled.
Horizontal well (no-flow boundaries): Babu & Odeh (1989) pseudo-steady state,
    with their partial-penetration skin; Joshi (1988) steady state also offered.
Horizontal well with transverse fractures: Mukherjee & Economides (1991):
    n fractures each draining A/n, skin ln(rw/xf) + f(FcD) + choke skin
    s_c = (h / (FcD xf)) [ln(h / 2rw) - pi/2].
Dual porosity: Warren & Root (1963) line source with skin.
Multilayer: commingled layers with crossflow (linear injectivity above the
    layer pressure). Multilayer with wellbore dP (PROSPER's recommended model):
    each layer with its own pressure, inflow model and PVT; the commingled
    stream is marched up between the layers with a multiphase correlation;
    validated against PROSPER Tutorial 11.
Horizontal well with friction dP: lateral split into zones, inflow allocated
    by k x length and zone skin, friction marched from toe to heel.
Modified isochronal: correction factors, n regression, stabilisation as in
    the PROSPER guide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import erf, exp1

from .correlations import friction_factor
from .inflow import IPRCurve, _curve_from_rate, _jones_rate, _pi_vogel_rate, dietz_term, non_darcy_d

RANKINE = 459.67
LN2 = np.log(2.0)


# ================================================================ helpers
def _oil_curve(model, pr, pb, kh_eq, mu, bo, D, params, notes=None):
    j = kh_eq / (141.2 * mu * bo * D)
    c = _curve_from_rate(model, "oil", pr, _pi_vogel_rate(pr, pb, j),
                         {"PI (STB/d/psi)": j, **params}, notes)
    return c


def _gas_curve(model, pr, t_f, mu, z, kh_eq, D, d_nd, params, notes=None):
    base = 1422 * (t_f + RANKINE) * mu * z / kh_eq
    a, b = base * d_nd, base * D
    return _curve_from_rate(model, "gas", pr, _jones_rate(pr, a, b, True),
                            {"a (psi^2/(Mscf/d)^2)": a, "b (psi^2/(Mscf/d))": b, **params}, notes)


def cinco_f(fcd):
    u = np.log(fcd)
    return (1.65 - 0.328 * u + 0.116 * u * u) / (1 + 0.180 * u + 0.064 * u * u + 0.005 * u ** 3)


def pd_infinite_conductivity(td):
    """Gringarten et al. infinite-conductivity vertical fracture, wellbore pD (xf-based)."""
    td = np.asarray(td, float)
    st = np.sqrt(td)
    return (0.5 * np.sqrt(np.pi * td) * (erf(0.134 / st) + erf(0.866 / st))
            + 0.067 * exp1(0.018 / td) + 0.433 * exp1(0.75 / td))


# ======================================================= fractured well
FRAC_TDA_CAP = 0.0891     # PROSPER's fractured well: the transient solution at most to tDA = 0.0891 (tD on the radius
#                           of a circle of the drainage area = 0.28); later times give the same PI - its Tutorial 15
#                           file: identical PI at 5, 30 and 365 days, the transient one at 1 day (k t = 50 md.d) and
#                           with 10 / 20 md, and the cap's D falling as ln xf (xf 35 / 100 / 200 ft)
FRAC_FCD_INFINITE = 100.0  # PROSPER: FCD >= 100 behaves as infinite conductivity (same PI at 100 to 20000)


def fractured_well(fluid, pr, k, h, area, ca, rw, xf, fcd, t_days, phi, ct, mu,
                   face_skin=0.0, pb=None, bo=None, t_f=None, z=None, sg=None, hf=None):
    """Hydraulically fractured well, as PROSPER: Gringarten infinite-conductivity transient solution at the time since
    production started, held at its value at tDA = FRAC_TDA_CAP for later times, plus Cinco-Ley's finite-conductivity
    correction f(FCD) - ln 2 (zero from FCD = 100) and the fracture face skin. Against PROSPER's Tutorial 15 file and
    twelve variations (k 10-100 md, xf 35-200 ft, FCD 10-20000, 1-365 days): PI within 1.5 %."""
    if xf <= rw:
        raise ValueError("Fracture half-length must exceed the wellbore radius.")
    a_ft2 = area * 43560.0
    f = LN2 if fcd >= FRAC_FCD_INFINITE else float(cinco_f(np.clip(fcd, 0.1, FRAC_FCD_INFINITE)))
    td = 0.0002637 * k * t_days * 24 / (phi * mu * ct * xf ** 2)
    td_cap = FRAC_TDA_CAP * a_ft2 / xf ** 2
    D = float(pd_infinite_conductivity(min(td, td_cap))) + (f - LN2) + face_skin
    state = "transient" if td < td_cap else f"at its tDA = {FRAC_TDA_CAP:g} limit (PROSPER)"
    params = {"Equivalent skin": float(np.log(rw / xf) + f + face_skin), "tD (xf)": td,
              "Dimensionless resistance D": D}
    notes = [f"At {t_days:g} days the fractured well's solution is {state}.",
             "Early bilinear flow (low FcD) is not modelled."]
    if 2 * xf > np.sqrt(a_ft2):
        notes.append("Fracture length is comparable to the drainage dimension - the radial "
                     "equivalent-skin approach is approximate.")
    if fluid == "oil":
        return _oil_curve("Hydraulically Fractured Well", pr, pb, k * h, mu, bo, D, params, notes)
    d_nd = non_darcy_d(k, h, hf or h, rw, mu, sg)
    params["D (1/(Mscf/d))"] = d_nd
    return _gas_curve("Hydraulically Fractured Well", pr, t_f, mu, z, k * h, D, d_nd, params, notes)


# ======================================================= horizontal well
def _F(x):
    x = np.asarray(x, float)
    x2 = 2 - x
    small = -x * (0.145 + np.log(np.maximum(x, 1e-12)) - 0.137 * x * x)
    large = x2 * (0.145 + np.log(np.maximum(x2, 1e-12)) - 0.137 * x2 * x2)
    return np.where(x <= 1, small, large)


def babu_odeh(k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw, skin=0.0):
    """Babu & Odeh PSS resistance. PROSPER geometry: k = permeability perpendicular to
    the well (Ky); aniso_h = Ky/Kx (Kx along the well); aniso_v = Kz/Ky; Lx along the
    well, Ly across it; Xw, Yw, Zw position of the well centre.
    Returns (kh_eq, D, s_R)."""
    kx = k                 # Babu x: horizontal, perpendicular to the well
    ky = k / aniso_h       # Babu y: along the well
    kz = k * aniso_v
    a, b = ly, lx
    x0, ymid, z0 = yw, xw, zw
    if not (0 < z0 < h and 0 < x0 < a and L <= b + 1e-9):
        raise ValueError("Well must lie inside the drainage box and be no longer than its length.")
    if ymid - L / 2 < -1e-6 or ymid + L / 2 > b + 1e-6:
        raise ValueError("Well extends beyond the drainage box along its length.")
    sin_term = np.sin(np.pi * z0 / h)
    ln_ch = (6.28 * a / h * np.sqrt(kz / kx) * (1 / 3 - x0 / a + (x0 / a) ** 2)
             - np.log(sin_term) - 0.5 * np.log(a / h * np.sqrt(kz / kx)) - 1.088)
    s_r = 0.0
    if L < b - 1e-6:
        pxyz = (b / L - 1) * (np.log(h / rw) + 0.25 * np.log(kx / kz) - np.log(sin_term) - 1.84)
        if a / np.sqrt(kx) >= 0.75 * b / np.sqrt(ky):            # case 1
            pxy_p = (2 * b * b / (L * h)) * np.sqrt(kz / ky) * (
                _F(L / (2 * b)) + 0.5 * (_F((4 * ymid + L) / (2 * b)) - _F((4 * ymid - L) / (2 * b))))
            s_r = pxyz + float(pxy_p)
        else:                                                      # case 2
            py = (6.28 * b * b / (a * h)) * np.sqrt(kx * kz) / ky * (
                (1 / 3 - ymid / b + (ymid / b) ** 2) + L / (24 * b) * (L / b - 3))
            pxy = (b / L - 1) * (6.28 * a / h) * np.sqrt(kz / kx) * (1 / 3 - x0 / a + (x0 / a) ** 2)
            s_r = pxyz + py + pxy
    kh_eq = b * np.sqrt(kx * kz)
    D = np.log(np.sqrt(a * h) / rw) + ln_ch - 0.75 + s_r + skin
    return kh_eq, float(D), float(s_r)


def joshi(k, h, rw, aniso_h, aniso_v, L, lx, ly, skin=0.0):
    """Joshi steady state (Economides anisotropy form). Returns (kh_eq, D)."""
    kh = k / np.sqrt(aniso_h)                     # geometric mean of horizontal perms
    kv = k * aniso_v
    beta = np.sqrt(kh / kv)
    reh = np.sqrt(lx * ly / np.pi)
    if L / 2 >= 0.9 * reh:
        raise ValueError("Joshi requires the well to be shorter than the drainage radius.")
    aj = L / 2 * np.sqrt(0.5 + np.sqrt(0.25 + (2 * reh / L) ** 4))
    D = (np.log((aj + np.sqrt(aj ** 2 - (L / 2) ** 2)) / (L / 2))
         + beta * h / L * (np.log(beta * h / ((beta + 1) * rw)) + skin))
    return kh * h, float(D)


# Goode & Kuchuk (1991, SPE 21460 - "Kuchuk & Goode" in PROSPER): horizontal well of length 2 L1/2, parallel to x,
# anywhere in a rectangular box Lx x Ly x h; upper boundary no-flow (pseudo-steady state, Eq. 6/A-20 with the
# geometric skin of Eq. A-21) or constant pressure (gas cap / aquifer: steady state, Eq. B-3).
def k12(beta):
    """K_12(beta) = integral_0^inf exp(-beta cosh u) / cosh^2 u du (Goode & Kuchuk Eq. A-22, Table 1)."""
    from scipy.integrate import quad
    if beta > 700:
        return 0.0
    with np.errstate(over="ignore", under="ignore"):
        return float(quad(lambda u: np.exp(-beta * np.cosh(u)) / np.cosh(u) ** 2, 0.0, np.inf, limit=200)[0])


def _gk_geometry(k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw):
    kx, ky, kz = k / aniso_h, k, k * aniso_v          # x along the well (PROSPER: k = Ky, Ky/Kx, Kz/Ky)
    half = L / 2
    if not (0 < zw < h):
        raise ValueError("The well must lie between the top and bottom of the reservoir (0 < Zw < h).")
    if not (0 < yw < ly) or xw - half < -1e-6 or xw + half > lx + 1e-6:
        raise ValueError("The well must lie inside the drainage area (Xw +/- L/2 within Lx, 0 < Yw < Ly).")
    hd = h / half * np.sqrt(kx / kz)
    rwp = rw / 2 * (1 + np.sqrt(kz / ky))
    return kx, ky, kz, half, hd, rwp


def goode_kuchuk_szd(h, zw, hd, rwp):
    """Geometric (vertical convergence) skin S_zD, Eq. A-21 - the K12 series makes it exact for any hD."""
    x = zw / h
    s = -hd / 2 * (np.log(2 * np.pi * rwp / h * np.sin(np.pi * x)) + hd * (1 / 3 - x + x * x))
    ser, j = 0.0, 1
    while True:
        b = 2 * j * np.pi / hd
        if b > 60:                                         # K12(60) ~ 1e-28
            break
        ser += k12(b) * np.cos(j * np.pi * x) ** 2 / j ** 2
        j += 1
    return float(s + hd ** 2 / np.pi ** 2 * ser)


def goode_kuchuk(k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw, skin=0.0, boundary="no flow", n_terms=2000):
    """Goode & Kuchuk (1991). Returns (kh_eq, D, params) for q = kh_eq (p - pwf) / (141.2 mu B D):
    kh_eq = sqrt(kx ky) h, D = pwD + S*m with S*m = h / (2 L1/2) sqrt(kx/kz) Sm (Eqs. 3-5).
    boundary='no flow': p = average reservoir pressure (pseudo-steady state, Eq. 6);
    'constant pressure': p = pressure at the upper constant-pressure boundary (steady state, Eq. B-3)."""
    kx, ky, kz, half, hd, rwp = _gk_geometry(k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw)
    if boundary == "no flow":
        t1 = 2 * np.pi * ly / lx * np.sqrt(kx / ky) * (1 / 3 - yw / ly + (yw / ly) ** 2)
        n = np.arange(1, n_terms + 1)
        a = -2 * n * np.pi / lx * np.sqrt(kx / ky)
        with np.errstate(over="ignore", under="ignore"):
            xi = (2 * np.exp(a * ly) + np.exp(a * (ly - yw)) + np.exp(a * yw)) / (1 - np.exp(a * ly))
        t2 = 2 * lx ** 2 / (np.pi ** 2 * half ** 2) * float(np.sum(
            np.sin(n * np.pi * half / lx) ** 2 * np.cos(n * np.pi * xw / lx) ** 2 * (1 + xi) / n ** 3))
        szd = goode_kuchuk_szd(h, zw, hd, rwp)
        pwd = t1 + t2 + szd
        params = {"Areal term": float(t1 + t2), "Geometric skin S_zD": szd}
    else:
        x = zw / h
        pwd = hd / 2 * (np.log(4 * h / (np.pi * rwp) / np.tan(np.pi * x / 2)) + (zw - h) / half * np.sqrt(kx / kz))
        ser, j = 0.0, 1
        while True:
            al = 2 * j - 1
            b = al * np.pi / hd
            if b > 60:
                break
            ser += k12(b) * np.cos(al * np.pi * x / 2) ** 2 / al ** 2
            j += 1
        pwd = float(pwd + 4 * hd ** 2 / np.pi ** 2 * ser)
        params = {}
    s_star = h / (2 * half) * np.sqrt(kx / kz) * skin
    params.update({"pwD": float(pwd), "Dimensionless thickness hD": float(hd), "Skin S*m": float(s_star)})
    return float(np.sqrt(kx * ky) * h), float(pwd + s_star), params


def horizontal_well(fluid, method, pr, k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw, mu,
                    skin=0.0, pb=None, bo=None, t_f=None, z=None, sg=None):
    if method == "Babu & Odeh":
        kh_eq, D, s_r = babu_odeh(k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw, skin)
        params = {"Partial penetration skin s_R": s_r}
    elif method in ("Kuchuk & Goode", "Kuchuk & Goode - constant pressure upper boundary"):
        kh_eq, D, params = goode_kuchuk(k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw, skin,
                                        "no flow" if method == "Kuchuk & Goode" else "constant pressure")
    else:
        kh_eq, D = joshi(k, h, rw, aniso_h, aniso_v, L, lx, ly, skin)
        params = {}
    name = f"Horizontal Well ({method})"
    if fluid == "oil":
        return _oil_curve(name, pr, pb, kh_eq, mu, bo, D, params)
    d_nd = 0.0                                   # non-Darcy negligible along a long lateral
    return _gas_curve(name, pr, t_f, mu, z, kh_eq, D, d_nd, params,
                      ["Non-Darcy flow neglected along the lateral."])


def horizontal_resistance(method, k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw, skin=0.0):
    """(kh_eq, D) of a horizontal well for PROSPER's horizontal well models (q = kh_eq dp / (141.2 mu B D)); the
    skin is the mechanical skin along the well, entering D as each model scales it (Kuchuk & Goode: h / L
    sqrt(kx/kz) s; Babu & Odeh: s; Joshi: beta h / L s)."""
    if method == "Babu & Odeh":
        kh_eq, D, _ = babu_odeh(k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw, skin)
    elif method in ("Kuchuk & Goode", "Kuchuk & Goode - constant pressure upper boundary"):
        kh_eq, D, _ = goode_kuchuk(k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw, skin,
                                   "no flow" if method == "Kuchuk & Goode" else "constant pressure")
    else:
        kh_eq, D = joshi(k, h, rw, aniso_h, aniso_v, L, lx, ly, skin)
    return float(kh_eq), float(D)


def transverse_fractures(fluid, pr, k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw, mu,
                         xf, fcd, n_frac, ca=31.62, skin=0.0, pb=None, bo=None, t_f=None, z=None, sg=None):
    n_frac = int(n_frac)
    if n_frac < 1:
        raise ValueError("At least one fracture is needed.")
    area_each = lx * ly / n_frac / 43560
    s_choke = h / (fcd * xf) * (np.log(h / (2 * rw)) - np.pi / 2)
    s_frac = np.log(rw / xf) + float(cinco_f(np.clip(fcd, 0.1, 1000))) + s_choke + skin
    D_f = dietz_term(area_each, ca, rw) + s_frac
    kh_f = n_frac * k * h
    kh_h, D_h, _ = babu_odeh(k, h, rw, aniso_h, aniso_v, L, lx, ly, xw, yw, zw, skin)
    frac_better = kh_f / D_f >= kh_h / D_h
    kh_eq, D = (kh_f, D_f) if frac_better else (kh_h, D_h)
    notes = [f"{n_frac} fractures: {'fractures dominate' if frac_better else 'unfractured lateral dominates'}."]
    if L / n_frac < 2 * xf * 0.25:
        notes.append("Fractures are closely spaced relative to their length - interference makes this optimistic.")
    if 2 * xf > ly:
        notes.append("Fracture tip-to-tip length exceeds the reservoir width.")
    params = {"Choke skin": float(s_choke), "Skin per fracture": float(s_frac),
              "Fracture / lateral PI ratio": float((kh_f / D_f) / (kh_h / D_h))}
    name = "Horizontal Well - Transverse Vertical Fractures"
    if fluid == "oil":
        return _oil_curve(name, pr, pb, kh_eq, mu, bo, D, params, notes)
    return _gas_curve(name, pr, t_f, mu, z, kh_eq, D, 0.0, params, notes)


# ========================================================= dual porosity
def dual_porosity(fluid, pr, kf, h, area, ca, rw, phi, ct, t_days, omega, lam, mu, skin=0.0,
                  pb=None, bo=None, t_f=None, z=None, sg=None, d_nd=None):
    if not 0 < omega < 1:
        raise ValueError("Storativity ratio must be between 0 and 1.")
    td = 0.0002637 * kf * t_days * 24 / (phi * ct * mu * rw ** 2)
    pd_tr = 0.5 * (np.log(td) + 0.80907 - exp1(lam * td / (omega * (1 - omega))) + exp1(lam * td / (1 - omega)))
    d_tr = float(pd_tr) + skin
    d_pss = dietz_term(area, ca, rw) + skin
    D = min(d_tr, d_pss)
    state = "transient" if d_tr < d_pss else "pseudo-steady state"
    params = {"tD": td}
    notes = [f"At {t_days:g} days the well is in {state} flow (Warren & Root)."]
    if fluid == "oil":
        return _oil_curve("Dual Porosity", pr, pb, kf * h, mu, bo, D, params, notes)
    d_nd = non_darcy_d(kf, h, h, rw, mu, sg) if d_nd is None else d_nd
    params["D (1/(Mscf/d))"] = d_nd
    return _gas_curve("Dual Porosity", pr, t_f, mu, z, kf * h, D, d_nd, params, notes)


# ============================================================ multilayer
def layer_rate_fn(fluid, layer, pb=None):
    """q(p) for one layer, negative (crossflow into the layer) above its pressure.
    layer: {"pr", "model": "PI"|"Darcy"|"C and n"|"Jones", params...}"""
    pr = layer["pr"]
    if fluid == "oil":
        j = layer["j"]
        prod = _pi_vogel_rate(pr, pb if pb is not None else 0.0, j)
        return lambda p: np.where(np.asarray(p, float) <= pr, prod(np.minimum(p, pr)),
                                  -j * (np.asarray(p, float) - pr))
    if layer["model"] == "C and n":
        c, n = layer["c"], layer["n"]
        return lambda p: np.sign(pr ** 2 - np.asarray(p, float) ** 2) * c * np.abs(pr ** 2 - np.asarray(p, float) ** 2) ** n
    a, b = layer["a"], layer["b"]

    def f(p):
        dp2 = pr ** 2 - np.asarray(p, float) ** 2
        mag = (-b + np.sqrt(b * b + 4 * a * np.abs(dp2))) / (2 * a) if a > 0 else np.abs(dp2) / b
        return np.sign(dp2) * mag
    return f


def multilayer(fluid, layers: list, pb=None, n=60):
    if not layers:
        raise ValueError("Enter at least one producing layer.")
    fns = [layer_rate_fn(fluid, L, pb) for L in layers]
    pmax = max(L["pr"] for L in layers)
    total = lambda p: sum(f(p) for f in fns)
    pw = np.linspace(pmax, 0.0, n)
    q = np.asarray(total(pw), float)
    per = pd.DataFrame({f"Layer {i + 1}": np.asarray(f(pw), float) for i, f in enumerate(fns)})
    # IPR starts where the commingled well stops flowing (total rate = 0)
    p0 = brentq(lambda p: float(total(np.array([p]))[0]), 0.0, pmax) if q[0] < 0 else pmax
    curve = _curve_from_rate("Multi-layer", fluid, p0, lambda p: np.maximum(total(p), 0.0), {
        "Layers": len(layers), "Commingled shut-in pressure (psia)": p0}, n=n)
    curve.layer_rates = pd.DataFrame({f"Layer {i + 1}": np.asarray(f(curve.pwf), float)
                                      for i, f in enumerate(fns)})
    if any(np.asarray(f(np.array([p0])), float)[0] < 0 for f in fns):
        curve.notes.append("At low drawdown some layers take crossflow (negative layer rate).")
    return curve


def _wellbore_props(fluid, props):
    """Return functions rho(p) lb/ft3, mu(p) cp, vol(q, p) ft3/s for the wellbore fluid."""
    if fluid == "oil":
        wc = props.get("wc", 0.0)
        rho = (1 - wc) * props["rho"] + wc * 62.4 * props.get("water_sg", 1.05)
        mu = (1 - wc) * props["mu"] + wc * 0.5
        bo = props["bo"]
        return (lambda p: rho, lambda p: mu,
                lambda q, p: q * ((1 - wc) * bo + wc) * 5.6146 / 86400)
    mu_z, sg, t_f = props["mu_z"], props["sg"], props["t_f"]
    tr = t_f + RANKINE

    def rho(p):
        _, z = mu_z(np.array([max(p, 14.7)]))
        return 28.9625 * sg * max(p, 14.7) / (float(z[0]) * 10.7316 * tr)

    def mu(p):
        m, _ = mu_z(np.array([max(p, 14.7)]))
        return float(m[0])

    def vol(q, p):
        _, z = mu_z(np.array([max(p, 14.7)]))
        return q * 1000 * 0.02827 * float(z[0]) * tr / max(p, 14.7) / 86400
    return rho, mu, vol


def _segment_dp(q, p, length, dtvd, d_in, rough_in, rho_fn, mu_fn, vol_fn):
    """Pressure lost moving up/along a segment carrying rate q (signed)."""
    d = d_in / 12
    area = np.pi * d * d / 4
    rho, mu = rho_fn(p), mu_fn(p)
    v = vol_fn(q, p) / area
    re = rho * abs(v) * d / (mu * 6.7197e-4)
    f = float(friction_factor(max(re, 1.0), rough_in / 12 / d)) if rough_in > 0 else 0.0
    fric = np.sign(v) * f * rho * v * v * length / (2 * 32.174 * d) / 144
    return rho * dtvd / 144 + fric


# ================================================ horizontal with friction
def horizontal_friction(fluid, pr, zones: list, overall_kh_eq, overall_D, k_ref, props, pb=None, n=40,
                        profile_at=None):
    """zones (heel -> toe): {"type": "Perforated"|"Open Hole"|"Blank", "length", "k", "skin",
    "id_in", "rough_in"}. overall_kh_eq / overall_D: horizontal-well resistance without skin; a zone's skin is its
    increment to D (horizontal_resistance with the zone's mechanical skin less without - PROSPER's 'By Hand' zone
    skin). Against PROSPER Tutorial 10 (Kuchuk & Goode, one 800 ft perforated zone, skin 3, 0.3175 ft flowing
    radius): heel pressure within 9 psi from 40 to 39,968 STB/d, AOF 49,971 against 49,955.6 STB/d."""
    if not zones:
        raise ValueError("Enter at least one zone.")
    open_z = [z for z in zones if z["type"] != "Blank"]
    if not open_z:
        raise ValueError("At least one zone must be open to flow.")
    sum_kl = sum(z["k"] * z["length"] for z in open_z)
    rho_fn, mu_fn, vol_fn = _wellbore_props(fluid, props)
    tr = (props.get("t_f", 200) + RANKINE)
    fns = []
    for z in zones:
        if z["type"] == "Blank":
            fns.append(lambda p: 0.0)
            continue
        w = z["k"] * z["length"] / sum_kl * overall_D / (overall_D + z["skin"])
        if fluid == "oil":
            j = overall_kh_eq / (141.2 * props["mu"] * props["bo"] * overall_D) * w
            fns.append(lambda p, j=j: float(_pi_vogel_rate(pr, pb or 0.0, j)(np.array([p]))[0]))
        else:
            mu_r, z_r = props["mu_r"], props["z_r"]
            b = 1422 * tr * mu_r * z_r * overall_D / overall_kh_eq / w
            fns.append(lambda p, b=b: max(pr ** 2 - p ** 2, 0.0) / b)
    centres = np.cumsum([z["length"] for z in zones]) - np.array([z["length"] for z in zones]) / 2

    def march(p_toe):
        p = p_toe
        qcum = 0.0
        prof = []
        for i in range(len(zones) - 1, -1, -1):
            q_i = fns[i](p)
            qcum += q_i
            prof.append((centres[i], p, q_i / max(zones[i]["length"], 1e-9)))
            nxt = centres[i - 1] if i > 0 else 0.0
            p = p - _segment_dp(qcum, p, centres[i] - nxt, 0.0, zones[i]["id_in"], zones[i]["rough_in"],
                                rho_fn, mu_fn, vol_fn)
        return p, qcum, prof[::-1]

    def solve(p_heel):
        return brentq(lambda pt: march(pt)[0] - p_heel, p_heel, pr, xtol=0.01) if p_heel < pr else pr

    pw = np.linspace(pr, 0.0, n)
    qs = np.array([march(solve(p))[1] for p in pw])
    order = np.argsort(pw)
    f = lambda x: np.interp(x, pw[order], qs[order])
    # no-friction reference
    q_nf = sum(fn(0.0) for fn in fns)
    curve = IPRCurve("Horizontal Well - friction dP", fluid, pr, pw, qs,
                     {"AOF without friction": q_nf, "Open zones": len(open_z)}, [], None, f)
    ph = profile_at if profile_at is not None else 0.5 * pr
    _, qh, prof = march(solve(ph))
    curve.profile = pd.DataFrame(prof, columns=["Distance from heel (ft)", "Pressure (psia)", "Rate per ft"])
    curve.profile_heel_pressure = ph
    curve.notes.append("Wellbore friction uses single-phase properties; non-Darcy flow neglected.")
    return curve


# ===================================================== modified isochronal
def modified_isochronal(pres, tests: pd.DataFrame, t_flow_hr, t_shut_hr, k, phi, swc, mu, rw, re):
    """PROSPER-guide modified isochronal analysis. tests: Rate (Mscf/d), Pwf (psia) in test order."""
    t = tests.dropna()
    t = t[(t["Rate"] > 0)].reset_index(drop=True)
    if len(t) < 2:
        raise ValueError("Need at least 2 flow periods.")
    q, pw = t["Rate"].to_numpy(float), t["Pwf"].to_numpy(float)
    if np.any(pw >= pres):
        raise ValueError("All flowing pressures must be below the initial reservoir pressure.")
    nper = len(t)
    starts = np.arange(nper) * (t_flow_hr + t_shut_hr)
    ends = starts + t_flow_hr

    def p_t(dt_hr):
        td = 0.00633 * (dt_hr / 24) * pres * k / (phi * (1 - swc) * mu * rw ** 2)
        return 1.1513 * np.log10(td) + 0.40453, td

    cf = np.ones(nper)
    tds = np.zeros(nper)
    for j in range(nper):
        pt1, tds[j] = p_t(ends[j] - starts[0])
        if j == 0:
            continue
        denom = q[0] * pt1
        for i in range(1, j + 1):
            denom += (q[i] - q[i - 1]) * p_t(ends[j] - starts[i])[0]
        cf[j] = q[j] * pt1 / denom
    dp2 = pres ** 2 - pw ** 2
    dp2_iso = cf * dp2
    n, lc = np.polyfit(np.log(dp2_iso), np.log(q), 1) if nper >= 2 else (1.0, 0.0)
    notes = []
    if not 0.5 <= n <= 1:
        nc = float(np.clip(n, 0.5, 1))
        notes.append(f"Isochronal n = {n:.3f} constrained to {nc:.2f}.")
        n = nc
        lc = float(np.mean(np.log(q) - n * np.log(dp2_iso)))
    c_iso = float(np.exp(lc))
    ratio = (2 * np.log10(0.60 * re / rw) / np.log10(2.25 * tds)) ** (1 / n)
    dp2_stab = ratio * dp2_iso
    c_stab = float(np.exp(np.mean(np.log(q) - n * np.log(dp2_stab))))
    t_stab_days = 0.1 * np.pi * re ** 2 * phi * (1 - swc) * mu / (0.00633 * k * pres)
    f = lambda pwf: c_stab * np.maximum(pres ** 2 - np.clip(pwf, 0, None) ** 2, 0) ** n
    curve = _curve_from_rate("Modified Isochronal", "gas", pres, f, {
        "C stabilised": c_stab, "C isochronal": c_iso, "n": float(n),
        "Time to stabilise (days)": float(t_stab_days)}, notes)
    curve.tests = pd.DataFrame({"Rate": q, "Pwf": pw})
    curve.isochronal = pd.DataFrame({"Rate (Mscf/d)": q, "FBHP (psia)": pw, "Correction factor": cf,
                                     "dP2 measured": dp2, "dP2 isochronal": dp2_iso, "dP2 stabilised": dp2_stab})
    curve.c_iso = c_iso
    return curve


# ================================================ multilayer with wellbore dP (PROSPER "MultiLayer - dP Loss in WellBore")
def _with_pvt(fluid, api=None, gas_sg=None):
    """The wellbore fluid with another oil gravity / gas gravity (black oil and wet gas; matches kept). Other fluids
    (EOS tables) are returned unchanged."""
    import dataclasses
    from . import vlp as V
    if fluid is None or (api is None and gas_sg is None):
        return fluid
    if isinstance(fluid, V.BlackOilFluid):
        inp = dataclasses.replace(fluid.inp, **({"api": float(api)} if api is not None else {}),
                                  **({"gas_sg": float(gas_sg)} if gas_sg is not None else {}))
        return V.BlackOilFluid(dataclasses.replace(fluid.m, inp=inp))
    if isinstance(fluid, V.WetGasFluid):
        inp = dataclasses.replace(fluid.inp, **({"gas_sg": float(gas_sg)} if gas_sg is not None else {}))
        return V.WetGasFluid(inp, float(api) if api is not None else fluid.api)
    return fluid


def _ml_rate_fn(kind, L, lf, t_res):
    """q(p) for one layer of the dP model: liquid STB/d (oil) or Mscf/d (gas); negative above the layer pressure
    (crossflow into the layer). Oil: PI or Darcy (J from the layer's own Bo and viscosity at its pressure), straight
    line above the layer bubble point, Vogel / Petrobras composite below it."""
    from .inflow import _pi_vogel_rate, darcy_pi
    pr = L["pr"]
    if kind == "oil":
        gor = L.get("gor", 0.0)
        pb = float(np.ravel(lf.pb_at(t_res, gor))[0]) if lf is not None and gor > 0 else L.get("pb", 0.0)
        if L.get("model", "PI") == "Darcy":
            _, bo_, mu = lf._props(np.array([pr]), np.array([t_res]), np.array([max(gor, 1.0)]))
            j = darcy_pi(L["k"], L["h"], float(mu[0]), float(bo_[0]), L["area"], L["ca"], L["rw"], L.get("skin", 0.0))
            L["bo"], L["mu"] = float(bo_[0]), float(mu[0])
        else:
            j = L["j"]
        L["j_calc"], L["pb_calc"] = j, pb
        prod = _pi_vogel_rate(pr, pb, j, L.get("wc", 0.0))
        return lambda p: np.where(np.asarray(p, float) <= pr, prod(np.minimum(p, pr)), -j * (np.asarray(p, float) - pr))
    return layer_rate_fn("gas", L)


def multilayer_wellbore(kind, layers: list, top_md, top_tvd, t_res, fluid, correlation="beggs_brill", n=30):
    """PROSPER's MultiLayer - dP Loss in WellBore IPR at the TOP node.

    layers (any order): {"type": "Perforated"|"Blank", "md", "tvd" (bottom of the layer), "r_flow" (ft, radius of the
    pipe from the node above down to this layer), "rough_in"} and for perforated layers "pr" (psia, at the layer
    bottom), "model" (oil: "PI" with "j" | "Darcy" with "k", "area", "ca", "rw", "skin"; gas: "C and n" with "c", "n" |
    "Jones" with "a", "b"), and the layer PVT - oil: "gor", "wc" (fraction), optional "api", "gas_sg"; gas: "cgr",
    "wgr", optional "gas_sg". Darcy/Jones thickness "h" defaults to the TVD from the node above.
    The well is marched up from the deepest layer: each layer adds (or, above its pressure, takes) fluid at the
    pressure at its bottom; between nodes the commingled stream (rates, GOR and water cut blended from the layers)
    loses pressure by the multiphase correlation in a pipe of the layer's flowing radius. Zero roughness turns the
    wellbore losses off (layer pressures referred to the TOP depth), as in PROSPER.
    Returns the IPR curve at TOP with .layer_table (per point and layer: pressure, liquid/oil/water/gas rates) and
    .mix(q) -> dict of the blended producing ratios at a total rate (wc fraction and gor, or cgr and wgr)."""
    from . import vlp as V
    from .correlations import CORRELATIONS
    from .inflow import IPRCurve
    if fluid is None:
        raise ValueError("The multilayer dP model needs the PVT (complete the PVT tab).")
    lay = sorted([dict(L) for L in layers], key=lambda L: L["md"])
    if not lay or not any(L["type"] == "Perforated" for L in lay):
        raise ValueError("Enter at least one perforated layer.")
    if lay[0]["md"] < top_md - 0.01:
        raise ValueError("The TOP depth must be at or above the shallowest layer.")
    nodes_md = np.array([top_md] + [L["md"] for L in lay], float)
    nodes_tvd = np.array([top_tvd] + [L["tvd"] for L in lay], float)
    if np.any(np.diff(nodes_md) < -1e-6) or np.any(np.diff(nodes_tvd) < -1e-6):
        raise ValueError("Layer depths must increase downwards (MD and TVD).")
    prev_tvd = nodes_tvd[:-1]
    perf = []
    for i, L in enumerate(lay):
        if L["type"] != "Perforated":
            continue
        L.setdefault("h", max(L["tvd"] - prev_tvd[i], 0.1))
        if kind == "oil":
            L["fluid"] = _with_pvt(fluid, L.get("api"), L.get("gas_sg"))
        else:
            L["fluid"] = _with_pvt(fluid, None, L.get("gas_sg"))
        L["fn"] = _ml_rate_fn(kind, L, L["fluid"], t_res)
        perf.append(i)
    # wellbore column between the nodes (segment i: node i -> node i+1, pipe of layer i)
    seg = np.diff(nodes_md)
    dz = np.diff(nodes_tvd)
    r = np.array([L.get("r_flow", 0.25) for L in lay], float)
    rough = np.array([L.get("rough_in", 0.0) for L in lay], float)
    path = V.FlowPath(nodes_md, nodes_tvd, np.full(len(nodes_md), float(t_res)), np.maximum(seg, 1e-6),
                      np.clip(dz / np.maximum(seg, 1e-6), -1.0, 1.0), np.pi * r * r, 2 * r, rough / 12.0, 2 * r)
    corr = CORRELATIONS[correlation]
    on = rough > 0

    def phases(i, q):
        """(oil or condensate STB/d, water STB/d, gas scf/d) produced by layer i at rate(s) q."""
        L = lay[i]
        if kind == "oil":
            qo, qw = q * (1 - L.get("wc", 0.0)), q * L.get("wc", 0.0)
            return qo, qw, qo * L.get("gor", 0.0)
        g = q * 1000.0
        return g * L.get("cgr", 0.0) / 1e6, g * L.get("wgr", 0.0) / 1e6, g

    def case_of(qo, qw, qg):
        if kind == "oil":
            ql = qo + qw
            safe = np.where(np.abs(ql) > 1e-12, ql, 1.0)
            return {"q": np.maximum(np.abs(ql), 1e-6), "wc": np.clip(np.where(np.abs(ql) > 1e-12, qw / safe, 0.0), 0, 1),
                    "gor": np.where(qo > 1e-9, qg / np.where(qo > 1e-9, qo, 1.0), 0.0)}
        sg_ = np.where(qg > 1e-9, qg, 1.0)
        return {"q": np.maximum(np.abs(qg) / 1e6, 1e-9), "cgr": np.where(qg > 1e-9, qo / sg_ * 1e6, 0.0),
                "wgr": np.where(qg > 1e-9, qw / sg_ * 1e6, 0.0)}

    # wellbore fluid of each segment: oil and gas gravities blended over the layers below it, weighted by their
    # productivity (PROSPER blends by rate; the gravities rarely differ enough for the weighting to matter)
    def seg_fluid(i):
        below = [k for k in perf if k >= i]
        if not below:
            return fluid
        w = []
        for k in below:
            L = lay[k]
            wk = L.get("j_calc") or (L.get("c", 1.0) if kind == "gas" else 1.0)
            w.append((k, max(float(wk), 1e-12)))
        tot = sum(x for _, x in w)
        if kind == "oil":
            apis = [(lay[k].get("api"), x) for k, x in w if lay[k].get("api") is not None]
            sgs = [(lay[k].get("gas_sg"), x) for k, x in w if lay[k].get("gas_sg") is not None]
            api = sum(a * x for a, x in apis) / sum(x for _, x in apis) if apis else None
            sg = sum(a * x for a, x in sgs) / sum(x for _, x in sgs) if sgs else None
            return _with_pvt(fluid, api, sg)
        sgs = [(lay[k].get("gas_sg"), x) for k, x in w if lay[k].get("gas_sg") is not None]
        return _with_pvt(fluid, None, sum(a * x for a, x in sgs) / sum(x for _, x in sgs) if sgs else None)
    fl_seg = [seg_fluid(i) for i in range(len(lay))]

    def march(p_bottom):
        """Up from the deepest node for an array of bottom pressures: layer rates (layers x N), node pressures
        (nodes x N)."""
        p = np.array(p_bottom, float)
        N = p.size
        rates = np.zeros((len(lay), N))
        pn = np.empty((len(nodes_md), N))
        pn[-1] = p
        qo = np.zeros(N); qw = np.zeros(N); qg = np.zeros(N)
        tt = np.full(N, float(t_res))
        for i in range(len(lay) - 1, -1, -1):
            if i in perf:
                q = np.asarray(lay[i]["fn"](p), float)
                rates[i] = q
                a, b, c = phases(i, q)
                tot = qo + qw if kind == "oil" else qg
                lt = (a + b) if kind == "oil" else c
                take = (q < 0) & (tot > 0)              # crossflow: the layer takes the stream it sees
                f = np.where(take, np.minimum(-lt / np.maximum(tot, 1e-12), 1.0), 0.0)
                a, b, c = np.where(take, -qo * f, a), np.where(take, -qw * f, b), np.where(take, -qg * f, c)
                qo, qw, qg = qo + a, qw + b, qg + c
            if on[i] and seg[i] > 1e-6:
                cs = case_of(qo, qw, qg)
                sign = np.where((qo + qw if kind == "oil" else qg) >= 0, 1.0, -1.0)
                g0, _ = V._gradient(path, fl_seg[i], i, p, tt, cs, corr, 1.0, sign)
                pp = np.maximum(p - g0 * seg[i], 14.7)
                g1, _ = V._gradient(path, fl_seg[i], i, pp, tt, cs, corr, 1.0, sign)
                p = np.maximum(p - 0.5 * (g0 + g1) * seg[i], 14.7)
            pn[i] = p
        return rates, pn

    def solve_top(p_top):
        """Layer rates and node pressures for TOP pressures (array): fixed-point on the bottom pressure."""
        p_top = np.atleast_1d(np.asarray(p_top, float))
        pb_ = p_top + 0.3 * max(nodes_tvd[-1] - top_tvd, 0.0)
        prev = None
        for _ in range(20):
            rates, pn = march(pb_)
            err = p_top - pn[0]
            if np.nanmax(np.abs(err)) < 0.01:
                return rates, pn
            step = err.copy()
            if prev is not None:                       # secant on d(TOP)/d(bottom), guarded
                dp_, de_ = pb_ - prev[0], err - prev[1]
                slope = np.where(np.abs(dp_) > 1e-9, -de_ / np.where(np.abs(dp_) > 1e-9, dp_, 1.0), 1.0)
                slope = np.clip(slope, 0.2, 5.0)
                step = err / slope
            prev = (pb_.copy(), err)
            pb_ = np.maximum(pb_ + step, 14.7)
        return march(pb_)

    total = lambda rates: rates[perf].sum(axis=0)
    p_hi = max(lay[i]["pr"] for i in perf)
    grid = np.linspace(15.7, p_hi + 0.5 * max(nodes_tvd[-1] - top_tvd, 0.0) + 50.0, 40)
    tq = total(solve_top(grid)[0])
    k = np.where((tq[:-1] > 0) & (tq[1:] <= 0))[0]
    if len(k):
        lo, hi = grid[k[0]], grid[k[0] + 1]
        flo, fhi = tq[k[0]], tq[k[0] + 1]
        p0 = hi
        for _ in range(12):                        # shut-in TOP pressure: total rate zero (Illinois regula falsi)
            p0 = hi - fhi * (hi - lo) / (fhi - flo) if fhi != flo else 0.5 * (lo + hi)
            f0 = total(solve_top(np.array([p0]))[0])[0]
            if abs(f0) < 1e-3:
                break
            if f0 > 0:
                lo, flo = p0, f0
                fhi *= 0.5
            else:
                hi, fhi = p0, f0
                flo *= 0.5
    else:
        p0 = p_hi
    pw = np.linspace(p0, 15.7, n)
    rates, pn = solve_top(pw)
    rows = []
    qs = np.maximum(total(rates), 0.0)
    mixes = np.zeros((n, 3))
    for i in perf:
        a, b, c = phases(i, rates[i])
        mixes += np.c_[a, b, c]
        for kk in range(n):
            rows.append({"Point": kk + 1, "TOP pressure (psia)": pw[kk], "Layer": i + 1,
                         "Layer pressure (psia)": pn[i + 1, kk],
                         ("Liquid rate (STB/d)" if kind == "oil" else "Gas rate (Mscf/d)"): rates[i, kk],
                         ("Oil rate (STB/d)" if kind == "oil" else "Condensate rate (STB/d)"): a[kk],
                         "Water rate (STB/d)": b[kk], "Gas rate (MMscf/d)": c[kk] / 1e6})
    order = np.argsort(pw)
    f = lambda x: np.interp(x, pw[order], qs[order])
    params = {"Layers": len(perf), "Shut-in TOP pressure (psia)": p0}
    for i in perf:
        L = lay[i]
        if "j_calc" in L:
            params[f"Layer {i + 1} PI (STB/d/psi)"] = L["j_calc"]
        if L.get("pb_calc"):
            params[f"Layer {i + 1} bubble point (psia)"] = L["pb_calc"]
    curve = IPRCurve("Multi-layer with dP loss in wellbore", kind, p0, pw, qs, params, [], None, f)
    curve.layer_table = pd.DataFrame(rows).sort_values(["Layer", "Point"]).reset_index(drop=True)
    curve.layer_rates = curve.layer_table.pivot(index="Point", columns="Layer",
                                                values="Liquid rate (STB/d)" if kind == "oil" else "Gas rate (Mscf/d)")
    curve.layer_rates.columns = [f"Layer {c}" for c in curve.layer_rates.columns]
    curve.top_md, curve.top_tvd = float(top_md), float(top_tvd)
    good = qs > 1e-6

    def mix(q):
        """Blended producing ratios at total rate(s) q (IPR units)."""
        q = np.atleast_1d(np.asarray(q, float))
        if good.sum() < 1:
            return {}
        qq, mm = qs[good], mixes[good]
        o_, w_, g_ = (np.interp(q, qq, mm[:, j] / qq) for j in range(3))
        if kind == "oil":
            liq = o_ + w_
            return {"wc": w_ / np.maximum(liq, 1e-12), "gor": g_ / np.maximum(o_, 1e-12)}
        return {"cgr": o_ / np.maximum(g_, 1e-12) * 1e6, "wgr": w_ / np.maximum(g_, 1e-12) * 1e6}
    curve.mix = mix
    if not on.any():
        curve.notes.append("Zero roughness everywhere: no wellbore losses between layers (layer pressures are "
                           "referred to the TOP depth).")
    return curve


def layer_split(curve, p_top):
    """Per-layer results of a multilayer dP IPR at a TOP pressure (psia), interpolated between the IPR points:
    layer pressure and rates (negative = crossflow into the layer), plus the total."""
    lt = curve.layer_table
    rows = []
    for lay_, g in lt.groupby("Layer"):
        g = g.sort_values("TOP pressure (psia)")
        x = g["TOP pressure (psia)"].to_numpy(float)
        r = {"Layer": f"Layer {int(lay_)}"}
        for c in g.columns:
            if c in ("Point", "Layer", "TOP pressure (psia)"):
                continue
            r[c] = float(np.interp(p_top, x, g[c].to_numpy(float)))
        rows.append(r)
    out = pd.DataFrame(rows)
    tot = {"Layer": "Total"}
    for c in out.columns[1:]:
        tot[c] = np.nan if c == "Layer pressure (psia)" else out[c].sum()
    out = pd.concat([out, pd.DataFrame([tot])], ignore_index=True)
    return out
