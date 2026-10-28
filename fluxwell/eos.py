"""Compositional PVT: Petroleum Experts PRP import and a cubic equation of state.

PRP files (PROSPER/PVTP "PRP Export File") carry the EOS description:
equation type, component table (composition, critical properties, acentric
factor, MW, SG, Vc, volume shift, Tb, parachor, Omega A/B), non-zero binary
interaction coefficients, viscosity model, salinity, separator train and
reference temperature. Pressures in the file are psig, temperatures degF.

Engine
------
* Peng-Robinson (PR78 m(w) for w > 0.49) or Soave-Redlich-Kwong, with the
  file's per-component Omega A / Omega B and BICs.
* Peneloux volume shift (c_i = s_i * b_i) applied to phase volumes; it does
  not change phase equilibrium.
* Michelsen stability test, two-phase flash (successive substitution with a
  bounded Rachford-Rice solve), saturation pressure by stability bisection,
  phase envelope by scanning stability over a pressure grid.
* Lohrenz-Bray-Clark viscosity, Macleod-Sugden (parachor) interfacial tension.
* Separator train flash to stock tank: GOR / CGR, API, gas gravity, Bo / Bg.

Internal units: psia, degR, ft3/lbmol, R = 10.7316 psia ft3/(lbmol degR).
"""
from __future__ import annotations

import io
import shlex
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

R = 10.7316
RANKINE = 459.67
P_ATM = 14.696
SCF_PER_LBMOL = 379.49     # 60 degF, 14.696 psia
FT3_PER_BBL = 5.614583
LKMOL_PER_FT3LBMOL = 62.428

COMPONENT_COLUMNS = ["Mole fraction", "Tc (degF)", "Pc (psia)", "Acentric factor", "MW",
                     "Specific gravity", "Vc (l/kg.mole)", "Volume shift", "Tb (degF)", "Parachor",
                     "Omega A", "Omega B", "Costald volume", "Costald acentric", "Name"]


# ================================================================ parsing
@dataclass
class PRPFile:
    eos: str                          # "PR" or "SRK"
    components: pd.DataFrame          # COMPONENT_COLUMNS, pressures already psia
    bic: np.ndarray                   # full symmetric matrix
    volume_shift: bool = True
    oil_viscosity: str = "LBC"
    gas_viscosity: str = "LBC"
    salinity_ppm: float = 0.0
    separators: list = field(default_factory=list)   # [(T degF, P psia), ...]
    reference_temperature: float | None = None       # degF
    version: str = ""
    source_name: str = ""
    notes: list = field(default_factory=list)
    other: dict = field(default_factory=dict)


def _blocks(text: str):
    """Split a PRP file into (comment text, [data lines]) blocks."""
    blocks, comments, data = [], [], []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("*"):
            if data:
                blocks.append((" ".join(comments), data))
                comments, data = [], []
            comments.append(line.lstrip("* ").strip())
        elif line.strip():
            data.append(line.strip())
    if data:
        blocks.append((" ".join(comments), data))
    return blocks


def parse_prp(text: str, source_name: str = "") -> PRPFile:
    if "PetexPRP" not in text and "PRP Export" not in text:
        raise ValueError("Not a Petroleum Experts PRP export file (signature #PetexPRP not found).")
    blocks = _blocks(text)
    version = ""
    for raw in text.splitlines():
        if raw.startswith("*#") and raw[2:].strip().isdigit():
            version = raw[2:].strip()
    notes = []

    def find(*keys, start=0):
        for i in range(start, len(blocks)):
            c = blocks[i][0].lower()
            if all(k.lower() in c for k in keys):
                return i
        return None

    i_eos = find("equation type")
    if i_eos is None:
        raise ValueError("Equation type section missing.")
    eos_code = int(float(blocks[i_eos][1][0].split()[0]))
    eos = {0: "PR", 1: "SRK"}.get(eos_code)
    if eos is None:
        raise ValueError(f"Unsupported equation type code {eos_code}.")

    i_nc = find("number of components")
    n_comp, n_lump = (int(x) for x in blocks[i_nc][1][0].split()[:2])
    if n_lump:
        notes.append(f"{n_lump} lumped components present - lumping rules ignored, "
                     "full component list used.")
    i_tab = i_nc + 1
    comment = blocks[i_tab][0].lower()
    p_psig = "psig" in comment
    rows = []
    for line in blocks[i_tab][1][:n_comp]:
        parts = shlex.split(line)
        nums = [float(x) for x in parts[:14]]
        rows.append(nums + [parts[14] if len(parts) > 14 else f"C{len(rows) + 1}"])
    comp = pd.DataFrame(rows, columns=COMPONENT_COLUMNS)
    if p_psig:
        comp["Pc (psia)"] = comp["Pc (psia)"] + P_ATM
    total = comp["Mole fraction"].sum()
    if abs(total - 100) < 1.0:
        comp["Mole fraction"] /= 100.0
        notes.append("Composition given in percent - converted to fractions.")
    elif abs(total - 1.0) > 1e-3:
        notes.append(f"Composition summed to {total:.5f} - normalised to 1.")
    comp["Mole fraction"] /= comp["Mole fraction"].sum()

    bic = np.zeros((n_comp, n_comp))
    i_nb = find("number of non-zero bic")
    if i_nb is not None:
        nb = int(float(blocks[i_nb][1][0].split()[0]))
        lines = blocks[i_nb][1][1:] + (blocks[i_nb + 1][1] if len(blocks[i_nb][1]) == 1 else [])
        for line in lines[:nb]:
            a, b, v = line.split()[:3]
            a, b = int(a), int(b)
            bic[a, b] = bic[b, a] = float(v)

    def scalar(*keys, cast=float, default=None):
        i = find(*keys)
        if i is None:
            return default
        tok = blocks[i][1][0].split()[0]
        try:
            return cast(tok)
        except ValueError:
            return default

    vs = scalar("volume shift flag", cast=lambda s: int(float(s)), default=1)
    oil_v = scalar("oil viscosity options", cast=str, default="LBC")
    gas_v = scalar("gas viscosity options", cast=str, default="LBC")
    sal = scalar("salinity", default=0.0)
    tref = scalar("reference temperature", default=None)
    seps = []
    i_sep = find("separators")
    if i_sep is not None:
        hdr = blocks[i_sep][1][0].split()
        ns = int(float(hdr[0]))
        for line in blocks[i_sep][1][1:1 + ns]:
            t, p = (float(x) for x in line.split()[:2])
            seps.append((t, p + P_ATM))          # separator pressures are psig in PRP files
    other = {}
    for key, label in (("co2 correction", "CO2 correction"), ("hydrate model", "Hydrate model"),
                       ("enthalpy model", "Enthalpy model")):
        v = scalar(key, cast=str)
        if v is not None:
            other[label] = v
    if oil_v.upper() not in ("LBC",):
        notes.append(f"Oil viscosity option '{oil_v}' in file - LBC is used here.")
    if gas_v.upper() not in ("LBC",):
        notes.append(f"Gas viscosity option '{gas_v}' in file - LBC is used here.")
    return PRPFile(eos, comp, bic, bool(vs), oil_v, gas_v, float(sal or 0), seps, tref,
                   version, source_name, notes, other)


# ===================================================================== EOS
@dataclass
class PhaseProps:
    z: float                  # compressibility (volume-shifted)
    mw: float
    molar_volume: float       # ft3/lbmol (shifted)
    density: float            # lb/ft3
    viscosity: float          # cp (LBC)


@dataclass
class FlashResult:
    p: float
    t: float                  # degF
    beta: float               # vapour mole fraction (0 = all liquid, 1 = all vapour)
    x: np.ndarray             # liquid composition
    y: np.ndarray             # vapour composition
    phases: str               # "2P", "L", "V"
    liquid: PhaseProps | None
    vapour: PhaseProps | None
    ift: float = np.nan       # dyne/cm
    converged: bool = True


class EOSFluid:
    def __init__(self, prp: PRPFile, z: np.ndarray | None = None):
        self.prp = prp
        c = prp.components
        self.names = c["Name"].tolist()
        self.z = (np.asarray(z, float) if z is not None else c["Mole fraction"].to_numpy(float))
        self.z = self.z / self.z.sum()
        self.tc = c["Tc (degF)"].to_numpy(float) + RANKINE
        self.pc = c["Pc (psia)"].to_numpy(float)
        self.w = c["Acentric factor"].to_numpy(float)
        self.mw = c["MW"].to_numpy(float)
        self.vc = c["Vc (l/kg.mole)"].to_numpy(float) / LKMOL_PER_FT3LBMOL
        self.oa = c["Omega A"].to_numpy(float)
        self.ob = c["Omega B"].to_numpy(float)
        self.parachor = c["Parachor"].to_numpy(float)
        self.kij = prp.bic
        self.b = self.ob * R * self.tc / self.pc
        s = c["Volume shift"].to_numpy(float) if prp.volume_shift else np.zeros(len(c))
        self.c = s * self.b
        if prp.eos == "PR":
            self.d1, self.d2 = 1 + np.sqrt(2), 1 - np.sqrt(2)
            w = self.w
            self.m = np.where(w <= 0.49, 0.37464 + 1.54226 * w - 0.26992 * w ** 2,
                              0.379642 + 1.48503 * w - 0.164423 * w ** 2 + 0.016666 * w ** 3)
        else:
            self.d1, self.d2 = 1.0, 0.0
            self.m = 0.480 + 1.574 * self.w - 0.176 * self.w ** 2
        self.n = len(self.z)

    # ------------------------------------------------------------ core
    def _a(self, T):
        alpha = (1 + self.m * (1 - np.sqrt(T / self.tc))) ** 2
        ai = self.oa * R ** 2 * self.tc ** 2 / self.pc * alpha
        return np.sqrt(np.outer(ai, ai)) * (1 - self.kij)

    def _zroots(self, A, B):
        d1, d2 = self.d1, self.d2
        u, w = d1 + d2, d1 * d2
        coeffs = [1.0, (u - 1) * B - 1, A + w * B * B - u * B - u * B * B, -(A * B + w * B * B + w * B ** 3)]
        r = np.roots(coeffs)
        r = np.real(r[np.abs(np.imag(r)) < 1e-9])
        return np.sort(r[r > B])

    def _lnphi(self, x, P, T, aij=None, phase=None):
        """ln fugacity coefficients and Z for composition x. phase: None (min Gibbs),
        'L' (smallest root) or 'V' (largest root)."""
        aij = self._a(T) if aij is None else aij
        RT = R * T
        am = x @ aij @ x
        bm = x @ self.b
        A = am * P / RT ** 2
        B = bm * P / RT
        roots = self._zroots(A, B)
        d1, d2 = self.d1, self.d2

        def g(Z):
            return Z - 1 - np.log(Z - B) - A / (B * (d1 - d2)) * np.log((Z + d1 * B) / (Z + d2 * B))
        if phase == "L":
            Z = roots[0]
        elif phase == "V":
            Z = roots[-1]
        else:
            Z = roots[0] if len(roots) == 1 else min((roots[0], roots[-1]), key=g)
        sa = 2 * (aij @ x) / am
        bb = self.b / bm
        lnphi = (bb * (Z - 1) - np.log(Z - B)
                 - A / (B * (d1 - d2)) * (sa - bb) * np.log((Z + d1 * B) / (Z + d2 * B)))
        return lnphi, Z

    def wilson_k(self, P, T):
        return self.pc / P * np.exp(5.373 * (1 + self.w) * (1 - self.tc / T))

    # ------------------------------------------------------- stability
    def stability(self, P, T_f, z=None, max_iter=400):
        """Michelsen tangent-plane test. Returns (stable, trial composition, K estimate)."""
        z = self.z if z is None else z
        T = T_f + RANKINE
        aij = self._a(T)
        lnphi_z, _ = self._lnphi(z, P, T, aij)
        d = np.log(np.maximum(z, 1e-300)) + lnphi_z
        K = self.wilson_k(P, T)
        best = (True, None, None, 0.0)
        for W0 in (z * K, z / K, z * K ** (1 / 3), z / K ** (1 / 3)):
            W = W0.copy()
            for _ in range(max_iter):
                w = W / W.sum()
                lnphi_w, _ = self._lnphi(w, P, T, aij)
                lnW = d - lnphi_w
                Wn = np.exp(lnW)
                if np.sum((np.log(np.maximum(W, 1e-300)) - lnW) ** 2) < 1e-14:
                    W = Wn
                    break
                W = Wn
            w = W / W.sum()
            trivial = np.sum((w - z) ** 2) < 1e-10
            tm = 1 - W.sum()
            if not trivial and tm < -1e-9 and tm < best[3]:
                best = (False, w, w / z if (w @ self.mw) < (z @ self.mw) else z / w, tm)
        return best[0], best[1], best[2]

    # ----------------------------------------------------------- flash
    @staticmethod
    def _rachford_rice(z, K):
        kmin, kmax = K.min(), K.max()
        if kmax <= 1:
            return 0.0
        if kmin >= 1:
            return 1.0
        lo, hi = 1 / (1 - kmax) + 1e-12, 1 / (1 - kmin) - 1e-12
        f = lambda b: np.sum(z * (K - 1) / (1 + b * (K - 1)))
        return brentq(f, lo, hi, xtol=1e-14, maxiter=200)

    def flash(self, P, T_f, z=None, max_iter=3000) -> FlashResult:
        z = self.z if z is None else np.asarray(z, float)
        T = T_f + RANKINE
        stable, w, K = self.stability(P, T_f, z)
        if stable:
            return self._single(P, T_f, z)
        aij = self._a(T)
        lnK = np.log(K)
        converged = False
        for it in range(max_iter):
            K = np.exp(lnK)
            beta = self._rachford_rice(z, K)
            x = z / (1 + beta * (K - 1))
            y = K * x
            x, y = x / x.sum(), y / y.sum()
            lpl, _ = self._lnphi(x, P, T, aij, "L")
            lpv, _ = self._lnphi(y, P, T, aij, "V")
            lnK_new = lpl - lpv
            err = np.sum((lnK_new - lnK) ** 2)
            lnK = lnK_new
            if err < 1e-14:
                converged = True
                break
        K = np.exp(lnK)
        beta = self._rachford_rice(z, K)
        if beta <= 1e-10 or beta >= 1 - 1e-10 or np.allclose(K, 1, atol=1e-4):
            return self._single(P, T_f, z)
        x = z / (1 + beta * (K - 1)); x /= x.sum()
        y = K * x; y /= y.sum()
        liq = self.phase_props(x, P, T_f, "L")
        vap = self.phase_props(y, P, T_f, "V")
        if liq.density < vap.density:     # label by density
            liq, vap, x, y, beta = vap, liq, y, x, 1 - beta
        ift = self.ift(x, y, liq, vap)
        return FlashResult(P, T_f, beta, x, y, "2P", liq, vap, ift, converged)

    def _single(self, P, T_f, z):
        pp = self.phase_props(z, P, T_f)
        # single phase labelled vapour-like or liquid-like by pseudo-reduced temperature
        tpc = z @ self.tc
        is_vap = (T_f + RANKINE) > tpc
        if is_vap:
            return FlashResult(P, T_f, 1.0, z.copy(), z.copy(), "V", None, pp)
        return FlashResult(P, T_f, 0.0, z.copy(), z.copy(), "L", pp, None)

    # -------------------------------------------------------- properties
    def phase_props(self, x, P, T_f, phase=None) -> PhaseProps:
        T = T_f + RANKINE
        _, Z = self._lnphi(x, P, T, phase=phase)
        v = Z * R * T / P - x @ self.c
        mw = x @ self.mw
        rho = mw / v
        return PhaseProps(P * v / (R * T), mw, v, rho, self.lbc(x, v, T))

    def lbc(self, x, v, T):
        """Lohrenz-Bray-Clark viscosity (cp); v in ft3/lbmol, T in degR."""
        tc_k = self.tc / 1.8
        pc_atm = self.pc / P_ATM
        xi_i = tc_k ** (1 / 6) / (np.sqrt(self.mw) * pc_atm ** (2 / 3))
        tr = T / self.tc
        hi = 17.78e-5 * np.maximum(4.58 * tr - 1.67, 1e-12) ** 0.625
        mu_i = np.where(tr <= 1.5, 34e-5 * tr ** 0.94, hi) / xi_i
        sq = np.sqrt(self.mw)
        mu0 = (x * mu_i * sq).sum() / (x * sq).sum()
        xi = (x @ tc_k) ** (1 / 6) / (np.sqrt(x @ self.mw) * (x @ pc_atm) ** (2 / 3))
        rr = (x @ self.vc) / v
        poly = 0.1023 + 0.023364 * rr + 0.058533 * rr ** 2 - 0.040758 * rr ** 3 + 0.0093324 * rr ** 4
        return mu0 + (poly ** 4 - 1e-4) / xi

    def ift(self, x, y, liq: PhaseProps, vap: PhaseProps):
        """Macleod-Sugden IFT (dyne/cm); molar densities in g-mol/cm3."""
        to_gcc = 1 / LKMOL_PER_FT3LBMOL          # lbmol/ft3 -> g-mol/cm3
        rl = 1 / liq.molar_volume * to_gcc
        rv = 1 / vap.molar_volume * to_gcc
        s = np.sum(self.parachor * (x * rl - y * rv))
        return float(max(s, 0.0) ** 4)

    # ------------------------------------------------ saturation pressure
    def saturation_pressure(self, T_f, p_max=15000.0, p_min=P_ATM, n_scan=60, tol=0.5):
        """Upper saturation pressure at T (psia) and its type ('bubble'/'dew'),
        or (None, None) if the fluid is single phase at every pressure."""
        grid = np.geomspace(p_max, p_min, n_scan)
        prev = None
        for P in grid:
            stable, w, _ = self.stability(P, T_f)
            if not stable:
                if prev is None:
                    return p_max, self._sat_type(P, T_f, w)
                lo, hi = P, prev
                while hi - lo > tol:
                    mid = 0.5 * (lo + hi)
                    s, wm, _ = self.stability(mid, T_f)
                    if s:
                        hi = mid
                    else:
                        lo, w = mid, wm
                return 0.5 * (lo + hi), self._sat_type(lo, T_f, w)
            prev = P
        return None, None

    def _sat_type(self, P, T_f, w):
        if w is None:
            return "unknown"
        feed = self.phase_props(self.z, P, T_f)
        inc = self.phase_props(w, P, T_f)
        return "bubble" if inc.density < feed.density else "dew"

    def phase_envelope(self, t_start=-250.0, t_step=35.0, t_max=1300.0, p_max=15000.0, n_scan=45):
        """Trace the two-phase boundary by scanning stability over a pressure grid at
        increasing temperatures until past the cricondentherm. Returns a DataFrame
        of boundary points (T, P, branch 'upper'/'lower', type 'bubble'/'dew')."""
        rows = []
        grid = np.geomspace(P_ATM, p_max, n_scan)
        T, found_any = t_start, False
        while T <= t_max:
            st = [self.stability(P, T)[0] for P in grid]
            hits = 0
            for i in range(1, len(grid)):
                if st[i] == st[i - 1]:
                    continue
                lo, hi = grid[i - 1], grid[i]
                s_lo = st[i - 1]
                for _ in range(16):
                    mid = 0.5 * (lo + hi)
                    if self.stability(mid, T)[0] == s_lo:
                        lo = mid
                    else:
                        hi = mid
                unstable_p = lo if not s_lo else hi
                _, w, _ = self.stability(unstable_p, T)
                rows.append({"T (degF)": T, "P (psia)": 0.5 * (lo + hi),
                             "branch": "upper" if st[i] else "lower",
                             "type": self._sat_type(unstable_p, T, w)})
                hits += 1
            if hits:
                found_any = True
            elif found_any:
                break                      # past the cricondentherm
            T += t_step
        return pd.DataFrame(rows)

    @staticmethod
    def envelope_summary(env: pd.DataFrame) -> dict:
        if env.empty:
            return {}
        up = env[env["branch"] == "upper"]
        out = {"cricondenbar_psia": float(up["P (psia)"].max()),
               "cricondenbar_T": float(up.loc[up["P (psia)"].idxmax(), "T (degF)"]),
               "cricondentherm_T": float(env["T (degF)"].max())}
        types = up.sort_values("T (degF)")["type"].tolist()
        temps = up.sort_values("T (degF)")["T (degF)"].tolist()
        for i in range(1, len(types)):
            if types[i - 1] == "bubble" and types[i] == "dew":
                out["critical_T_approx"] = 0.5 * (temps[i - 1] + temps[i])
        return out

    # --------------------------------------------------- separator train
    def separator_train(self, stages=None, T_res=None, P_res=None):
        """Flash the feed through the separator stages (last = stock tank).
        Returns dict of surface results and per-stage table."""
        stages = stages or self.prp.separators or [(60.0, P_ATM)]
        feed = self.z.copy()
        n_feed = 1.0
        gas_moles, gas_mw_moles = 0.0, 0.0
        recs = []
        for i, (T, P) in enumerate(stages):
            r = self.flash(P, T, feed)
            if r.phases == "2P":
                ng, nl = n_feed * r.beta, n_feed * (1 - r.beta)
                yv, xl = r.y, r.x
            elif r.phases == "V":
                ng, nl, yv, xl = n_feed, 0.0, feed, feed
            else:
                ng, nl, yv, xl = 0.0, n_feed, feed, feed
            gas_moles += ng
            gas_mw_moles += ng * (yv @ self.mw)
            recs.append({"Stage": i + 1, "T (degF)": T, "P (psia)": P,
                         "Gas (scf per lbmol feed)": ng * SCF_PER_LBMOL,
                         "Gas gravity": (yv @ self.mw) / 28.9625 if ng > 0 else np.nan,
                         "Liquid (lbmol per lbmol feed)": nl})
            feed, n_feed = xl, nl
            if nl <= 1e-12:
                break
        stock_oil_moles = n_feed
        out = {"stages": pd.DataFrame(recs)}
        gas_scf = gas_moles * SCF_PER_LBMOL
        out["gas_gravity"] = gas_mw_moles / gas_moles / 28.9625 if gas_moles > 0 else np.nan
        if stock_oil_moles > 1e-12:
            st_T, st_P = stages[-1]
            po = self.phase_props(feed, max(st_P, P_ATM), st_T, "L")
            oil_ft3 = stock_oil_moles * po.molar_volume
            stb = oil_ft3 / FT3_PER_BBL
            sg = po.density / 62.366
            out.update(stb_per_lbmol=stb, gor=gas_scf / stb, cgr=stb / gas_scf * 1e6 if gas_scf else np.inf,
                       api=141.5 / sg - 131.5, oil_density=po.density)
        else:
            out.update(stb_per_lbmol=0.0, gor=np.inf, cgr=0.0, api=np.nan, oil_density=np.nan)
        if T_res is not None and P_res is not None:
            pr = self.phase_props(self.z, P_res, T_res)
            res_ft3 = pr.molar_volume
            if out["stb_per_lbmol"] > 0:
                out["bo"] = res_ft3 / FT3_PER_BBL / out["stb_per_lbmol"]
            out["bg"] = res_ft3 / gas_scf if gas_scf > 0 else np.nan
            out["res_density"] = pr.density
            out["res_viscosity"] = pr.viscosity
        return out

    # ------------------------------------------------------ property table
    def property_table(self, T_f, pressures) -> pd.DataFrame:
        rows = []
        for P in pressures:
            r = self.flash(P, T_f)
            L, V = r.liquid, r.vapour
            if r.phases == "2P":
                vl, vv = (1 - r.beta) * L.molar_volume, r.beta * V.molar_volume
                liq_vol = 100 * vl / (vl + vv)
            else:
                liq_vol = 100.0 if r.phases == "L" else 0.0
            rows.append({
                "Pressure (psia)": P, "Phases": {"2P": "Two-phase", "L": "Liquid", "V": "Vapour"}[r.phases],
                "Vapour mole fraction": r.beta, "Liquid volume (%)": liq_vol,
                "Liquid density (lb/ft3)": L.density if L else np.nan,
                "Vapour density (lb/ft3)": V.density if V else np.nan,
                "Liquid viscosity (cp)": L.viscosity if L else np.nan,
                "Vapour viscosity (cp)": V.viscosity if V else np.nan,
                "Liquid Z": L.z if L else np.nan, "Vapour Z": V.z if V else np.nan,
                "IFT (dyne/cm)": r.ift,
            })
        return pd.DataFrame(rows)


def classify(sat_type: str | None, c7plus: float, cgr: float | None = None) -> str:
    """Fluid type implied by the EOS at reservoir temperature."""
    if sat_type == "bubble":
        return "oil"
    if sat_type == "dew":
        return "retrograde_condensate"
    return "dry_wet_gas" if c7plus < 0.02 else "oil"


def c7_plus_fraction(prp: PRPFile) -> float:
    c = prp.components
    heavy = c["MW"] >= 90
    return float(c.loc[heavy, "Mole fraction"].sum())


def example_prp_text() -> str:
    """A small synthetic black-oil-like PRP file for trying the tab (not a real fluid)."""
    rows = [
        ("N2", 0.0050, -232.51, 493.1, 0.040, 28.01, 0.809, 89.8, -0.154, -320.4, 60.4),
        ("CO2", 0.0150, 87.89, 1071.0, 0.225, 44.01, 0.818, 94.0, -0.100, -109.2, 78.0),
        ("C1", 0.3600, -116.67, 667.8, 0.011, 16.04, 0.300, 99.2, -0.154, -258.7, 70.0),
        ("C2", 0.0750, 89.92, 707.8, 0.099, 30.07, 0.356, 148.3, -0.100, -127.5, 115.0),
        ("C3", 0.0600, 206.06, 616.3, 0.152, 44.10, 0.507, 203.0, -0.085, -43.7, 155.0),
        ("iC4", 0.0120, 274.46, 529.1, 0.185, 58.12, 0.563, 263.0, -0.079, 10.9, 181.5),
        ("nC4", 0.0300, 305.62, 550.7, 0.200, 58.12, 0.584, 255.0, -0.064, 31.1, 189.9),
        ("iC5", 0.0130, 369.10, 490.4, 0.228, 72.15, 0.625, 306.0, -0.044, 82.1, 225.0),
        ("nC5", 0.0150, 385.80, 488.6, 0.252, 72.15, 0.631, 304.0, -0.040, 96.9, 231.5),
        ("C6", 0.0350, 453.70, 436.9, 0.300, 86.18, 0.664, 370.0, -0.020, 155.7, 271.0),
        ("C7-C12", 0.2300, 610.00, 410.0, 0.330, 125.0, 0.770, 450.0, 0.040, 310.0, 380.0),
        ("C13-C19", 0.0900, 800.00, 290.0, 0.580, 210.0, 0.840, 780.0, 0.090, 510.0, 600.0),
        ("C20+", 0.0600, 980.00, 200.0, 0.900, 330.0, 0.900, 1200.0, 0.130, 720.0, 850.0),
    ]
    lines = ["*", "* Petroleum Experts - PRP Export File", "*", "*Export File Signature", "*#PetexPRP",
             "*Export File Version", "*#18", "*", "* Synthetic example generated by fluxwell - not a real fluid",
             "*", "* Equation Type 0 - Peng Robinson 1 - SRK", "0", "*",
             "* Number of Components   Number of Lumped Components", f"{len(rows)} 0", "*",
             "*  Mole Critical Critical Accentric Molecular Specific Critical Volume Boiling Parachor OmegaA OmegaB Costald Costald Name",
             "*  percent degrees F psig - lb/lb.mole Gravity l/kg.mole - degrees F - - - l/kg.mole -", "*"]
    for n, zf, tc, pc, w, mw, sg, vc, s, tb, par in rows:
        lines.append(f"{zf} {tc} {pc - P_ATM:.4f} {w} {mw} {sg} {vc} {s} {tb} {par} "
                     f"0.457235529 0.0777960739 {vc} {w} \"{n}\"")
    bics = [(10, 2, 0.03), (11, 2, 0.04), (12, 2, 0.05), (1, 0, -0.02)]
    lines += ["*", "* Binary Interaction Coefficients", "* Number of non-zero BICs to be written or read", "*",
              str(len(bics)), "* Only first half of array is exported", "*"]
    lines += [f"{a} {b} {v}" for a, b, v in bics]
    lines += ["*", "* Volume Shift Flag", "*", "1", "*", "* OIL Viscosity Options", "*", "LBC",
              "*", "* GAS Viscosity Options", "*", "LBC", "*", "*Salinity", "50000", "*",
              "* Separators T, P", "*", "2 1", "100 85.304", "60 0.304", "*",
              "* Reference Temperature", "*", "210", "*", "*# End of file"]
    return "\n".join(lines) + "\n"
