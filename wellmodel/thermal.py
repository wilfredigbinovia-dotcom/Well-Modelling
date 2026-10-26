"""Temperature models that couple temperature to pressure (PROSPER's Improved Approximation; the Enthalpy Balance is
in enthalpy.py and uses the same fluid thermodynamics).

The flowing temperature follows the energy equation along the pipe (x measured upwards, PROSPER User Guide 2.19.2.3)

    dT/dx = [ -U pi d (T - Te) / w  -  sin(theta) / J ] / cp  +  mu_JT dP/dx

with U per depth interval (referred to the flow ID d, as the Rough Approximation's U), w the mass rate, J = 778.17
ft.lbf/BTU, cp the mixture heat capacity and mu_JT the mixture Joule-Thomson coefficient. Reverse-engineered from the
user's PROSPER Tutorial 41 runs (black-oil retrograde condensate at 10, 33.2 and 60 MMscf/d, with U and with U = 0.001,
and the same well on dry gas): PROSPER's mixture cp is the mass-weighted phase cp, and

    cp mu_JT = x_g (T dv_g/dT - v_g)  +  x_o C_OIL  -  K_COND (dx_o/dP)_T  +  x_w (T dv_w/dT - v_w)

- gas: the real-gas term R T^2 / (p M) (dZ/dT)_p of the phase at fixed composition (its Z correlation): PROSPER's
  single-phase gas column (0.0035-0.0040 F/psi near the bottom, 0.0125-0.0147 for the dry gas) reproduced;
- condensate liquid: C_OIL = -0.0133 BTU/lb/psi (fitted; PROSPER's oil enthalpy slope gives -0.016 to -0.02);
- condensation: K_COND = 152 BTU/lb on the pressure derivative of the liquid mass fraction (retrograde drop-out as
  the pressure falls cools the stream: PROSPER's two-phase JT rises from 0.014 to 0.043 F/psi towards the dew point);
- heat capacities: gas cp = CP0_A + CP0_B T + residual (from Z by the thermodynamic integral -T int (d2V/dT2) dp),
  condensate cp = 1.03 (0.388 + 0.00045 T) / sqrt(SG) (Gambill, scaled to PROSPER's 0.542-0.598), water from the
  Average Heat Capacities.
Effective JT within 6 % rms of PROSPER's at every two-phase segment; temperatures within 3 F along all four T41
traverses.
"""
from __future__ import annotations

import numpy as np

from . import blackoil as bo

J = 778.17                    # ft.lbf / BTU
R = 10.7316                   # psia ft3 / (lbmol R)
PSI_FT3 = 144.0 / J           # BTU per (psi ft3)
C_OIL = -0.0133               # BTU/lb/psi
K_COND = 152.0                # BTU/lb
CP0_A, CP0_B = 0.4917, 4.374e-4
WATER_TERM = -0.0024           # (T dv/dT - v) of water ~ -0.8 v, BTU/lb/psi
MODELS = {"rough": "Rough approximation", "improved": "Improved approximation", "enthalpy": "Enthalpy balance"}


def cp_oil(t, gamma):
    return 1.03 * (0.388 + 0.00045 * np.asarray(t, float)) / np.sqrt(gamma)


def cp_gas(p, t, M, tpc, ppc, n=32):
    """Real-gas cp (BTU/lb/F): ideal part linear in T plus the residual from the Z correlation."""
    p, t, M = np.broadcast_arrays(np.asarray(p, float), np.asarray(t, float), np.asarray(M, float))
    tr = t + 459.67
    s = np.linspace(0.0, 1.0, n)
    ps = 1.0 + s[None, :] * (np.maximum(p, 1.5)[..., None].reshape(-1, 1) - 1.0)
    trr = tr.reshape(-1, 1)
    h = 0.5
    z = lambda T: bo.z_dpr(T / tpc, ps / ppc)
    z0, zp, zm = z(trr), z(trr + h), z(trr - h)
    zt, ztt = (zp - zm) / (2 * h), (zp - 2 * z0 + zm) / (h * h)
    f = R / (ps * M.reshape(-1, 1)) * (2 * zt + trr * ztt)
    res = -trr[:, 0] * np.trapezoid(f, ps, axis=1) * PSI_FT3
    return CP0_A + CP0_B * t + res.reshape(t.shape)


def gas_jt_term(p, t, M, tpc, ppc):
    """T dv/dT - v for a real gas at fixed composition, BTU/lb/psi."""
    tr = np.asarray(t, float) + 459.67
    p = np.maximum(np.asarray(p, float), 1.5)
    dz = bo.z_dpr((tr + 0.5) / tpc, p / ppc) - bo.z_dpr((tr - 0.5) / tpc, p / ppc)
    return R * tr * tr / (p * M) * dz * PSI_FT3


def _retro(fluid, p, t, case, hc):
    from . import retrograde as RG
    r = fluid.rinp
    q, wgr = np.asarray(case["q"], float), np.maximum(np.asarray(case.get("wgr", 0.0), float), 0.0)
    m_hc = 1e6 * 0.0764 * r.gas_sg + r.cgr * 5.615 * 62.428 * r.gamma_o            # lb per MMscf
    m_w = wgr * 5.615 * 62.428 * 1.02
    tpc, ppc = RG._pseudocritical(r)

    def x_liq(pp):
        pr = RG.props(r, pp, t)
        return pr["dropout"] * pr["bo"] * 5.615 * pr["rho_o"] / m_hc, pr
    xo, pr = x_liq(p)
    dxp = (x_liq(p + 5.0)[0] - x_liq(np.maximum(p - 5.0, 15.0))[0]) / 10.0
    M = 28.97 * pr["gas_sg"]
    fw = m_w / (m_hc + m_w)
    x_o, x_g = (1 - fw) * xo, (1 - fw) * (1 - xo)
    cpo, cpg = cp_oil(t, r.gamma_o), cp_gas(p, t, M, tpc, ppc)
    cp = x_o * cpo + x_g * cpg + fw * hc.water
    num = x_g * gas_jt_term(p, t, M, tpc, ppc) + x_o * C_OIL - (1 - fw) * K_COND * dxp + fw * WATER_TERM
    return cp, num / cp


def _wet_gas(fluid, p, t, case, hc):
    inp = fluid.inp
    q = np.asarray(case["q"], float)
    wgr, cgr = np.maximum(np.asarray(case.get("wgr", 0.0), float), 0.0), np.maximum(np.asarray(case.get("cgr", 0.0), float), 0.0)
    sg = fluid.wellstream_sg(cgr)
    g = bo.PVTInputs(rsb=1, api=40, gas_sg=float(np.mean(sg)), h2s=inp.h2s, co2=inp.co2, n2=inp.n2)
    tpc, ppc = bo.gas_pseudocritical(g)
    m_g = 1e6 * 0.0764 * inp.gas_sg + cgr * 5.615 * 62.428 * fluid.go
    m_w = wgr * 5.615 * 62.428 * 1.02
    fw = m_w / (m_g + m_w)
    M = 28.97 * sg
    cp = (1 - fw) * cp_gas(p, t, M, tpc, ppc) + fw * hc.water
    return cp, ((1 - fw) * gas_jt_term(p, t, M, tpc, ppc) + fw * WATER_TERM) / cp


def _black_oil(fluid, p, t, case, hc):
    """Oil wells: free gas (real-gas term), oil (C_OIL) and water by mass; no phase-change term (not validated)."""
    inp = fluid.inp
    s = fluid.state(p, t, case)
    q, wc = np.asarray(case["q"], float), np.clip(np.asarray(case["wc"], float), 0, 0.9999)
    m_w = q * wc * 5.615 * 62.428 * 1.02
    m_g = s["qg"] * s["rho_g"] * 86400
    m_tot = np.maximum(fluid.mass_cp(case, hc)[0] * 24.0, 1e-9)
    xg, xw = np.clip(m_g / m_tot, 0, 1), np.clip(m_w / m_tot, 0, 1)
    xo = np.clip(1 - xg - xw, 0, 1)
    tpc, ppc = bo.gas_pseudocritical(inp)
    M = 28.97 * inp.gas_sg
    cp = xo * cp_oil(t, inp.gamma_o) + xg * cp_gas(p, t, M, tpc, ppc) + xw * hc.water
    num = xg * gas_jt_term(p, t, M, tpc, ppc) + xo * C_OIL + xw * WATER_TERM
    return cp, num / cp


def _generic(fluid, p, t, case, hc):
    """Any other fluid: mixture cp from the Average Heat Capacities and JT from the mixture specific volume."""
    w, cp = fluid.mass_cp(case, hc)
    w = np.maximum(np.asarray(w, float), 1e-9)

    def v(T):
        s = fluid.state(p, T, case)
        return (s["ql"] + s["qg"]) * 3600.0 / w
    tr = np.asarray(t, float) + 459.67
    term = tr * (v(t + 1.0) - v(t - 1.0)) / 2.0 - v(t)
    cp = np.broadcast_to(np.asarray(cp, float), np.shape(term))
    return cp, term * PSI_FT3 / cp


def fluid_thermo(fluid, p, t, case, hc):
    """(mixture cp BTU/lb/F, Joule-Thomson coefficient F/psi) at p (psia), t (F)."""
    from . import vlp as V
    p = np.asarray(p, float)
    t = np.broadcast_to(np.asarray(t, float), p.shape).astype(float)
    if isinstance(fluid, V.RetrogradeBOFluid):
        cp, mu = _retro(fluid, p, t, case, hc)
    elif isinstance(fluid, V.WetGasFluid):
        cp, mu = _wet_gas(fluid, p, t, case, hc)
    elif isinstance(fluid, V.BlackOilFluid):
        cp, mu = _black_oil(fluid, p, t, case, hc)
    else:
        cp, mu = _generic(fluid, p, t, case, hc)
    return np.broadcast_to(cp, p.shape), np.nan_to_num(np.broadcast_to(mu, p.shape))


def u_segments(path, table_md, table_u):
    """U per segment from a (MD, U) table: each row's U applies over the interval ending at its depth (PROSPER's
    Improved Approximation table; the first row's U also above it). Segments straddling a row depth take the U at their
    lower node."""
    md = np.asarray(table_md, float)
    u = np.asarray(table_u, float)
    o = np.argsort(md)
    md, u = md[o], u[o]
    lower = path.md[1:] - 1e-6
    idx = np.clip(np.searchsorted(md, lower, side="left"), 0, len(u) - 1)
    return u[idx]


def improved_profile(path, fluid, P, case, w, hc, u_seg, t_bottom=None, n_inner=3):
    """Temperature at every node (cases x nodes) for the pressure profile P (psia), marching up from the bottom node."""
    n = len(path.md)
    T = np.empty_like(P)
    T[:, -1] = path.t_form[-1] if t_bottom is None else np.broadcast_to(np.asarray(t_bottom, float), P[:, -1].shape)
    w = np.broadcast_to(np.maximum(np.asarray(w, float), 1e-9), P[:, 0].shape)
    for i in range(n - 2, -1, -1):
        L = path.seg_len[i]
        tfb, tft = path.t_form[i + 1], path.t_form[i]
        g = (tft - tfb) / L
        t0 = T[:, i + 1]
        pm = 0.5 * (P[:, i] + P[:, i + 1])
        dpdx = (P[:, i] - P[:, i + 1]) / L
        tn = t0.copy()
        for _ in range(n_inner):
            cp, mu = fluid_thermo(fluid, pm, 0.5 * (t0 + tn), case, hc)
            a = np.asarray(u_seg)[..., i] * np.pi * path.d_ht[i] / (w * cp)
            s = -path.sin_t[i] / (J * cp) + mu * dpdx
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                tn = tfb + g * L - (g - s) / a + (t0 - tfb + (g - s) / a) * np.exp(-a * L)
            tn = np.where(a * L < 1e-9, t0 + s * L, tn)
        T[:, i] = tn
    return T
