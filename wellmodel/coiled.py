"""Coiled tubing gas lift - PROSPER Options | Artificial lift = Coiled Tubing Gas Lift, System | Coiled Tubing Data
and Design | Coiled Tubing Gas Lift (User Guide 2.8.5 and 2.13.5, Tutorial 25).

The lift gas goes down a coiled tubing string hung inside the production tubing and leaves it at its end; the well
produces up the annulus between the coiled tubing and the tubing down to the injection point, and up the full tubing
below it. There are no unloading valves: the design finds the depth the kick-off pressure can unload the well to
and the deepest operating injection depth, and sizes the orifice at the end of the coiled tubing.

Geometry   above the coiled tubing end every tubing segment loses the coiled tubing's outside area; its hydraulic
           diameter becomes D_tubing - OD_ct (restrictions such as an SSSV keep their open area: sqrt(ID^2 - OD_ct^2)).
Lift gas   gas column inside the coiled tubing from the injection pressure at the coiled tubing head (the top node),
           the gas a little cooler than the flowing production around it: T_flow - 0.1 (T_flow - T_formation)
           (PROSPER Tutorial 25: 2648.9 psig at 8500 ft from 2200 psig and the 5446 ft kick-off depth - the gas at
           the flowing temperature gives 2637 psig and 5432 ft, the gas annulus rule of conventional gas lift
           (0.18) 2655 psig and 5468 ft). "Friction loss in coiled tubing" adds the friction of the gas flowing down
           the coiled tubing ID (Moody / Chen, inside roughness).
Kick-off   the deepest depth where the kick-off gas pressure less the desired dP reaches the static load fluid
           (unloading top-node pressure + load gradient x TVD below the top node).
Injection  specified depth: the gas enters at the coiled tubing end if its pressure there is at least the tubing
           pressure plus the dP across the valve, else not at all (natural flow). Optimum depth: the deepest point
           down to the coiled tubing end where that holds.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import gaslift as GL
from . import vlp as V
from .correlations import CP_TO_LBFTS, friction_factor
from .equipment import HeatCapacities

FRICTION = ("No friction loss in coiled tubing", "Friction loss in coiled tubing")
METHODS = ("Specified injection depth", "Optimum injection depth")


@dataclass
class CoiledTubing:
    id_in: float = 0.8              # inside diameter, in
    wall_in: float = 0.2            # wall thickness, in
    rough_in: float = 0.0006        # inside roughness, in (gas friction)
    rough_out: float = 0.0006       # outside roughness, in
    friction: bool = False          # friction loss in coiled tubing

    @property
    def od_in(self):
        return self.id_in + 2.0 * self.wall_in


ANNULUS_FRICTION = 0.90  # friction multiplier in the coiled tubing - tubing annulus (PROSPER T25 gradients, see above)


def annular_path(path: V.FlowPath, ct: CoiledTubing, depth_md: float) -> V.FlowPath:
    """Flow path of the production: the coiled tubing - tubing annulus above depth_md, the path unchanged below."""
    p = copy.deepcopy(path)
    do = ct.od_in / 12.0
    a_ct = np.pi / 4 * do * do
    mid = 0.5 * (p.md[1:] + p.md[:-1])
    fm = np.ones(len(mid)) if p.fric_mult is None else np.array(p.fric_mult, float)
    for j, m in enumerate(mid):
        if m < depth_md:
            if p.area[j] <= a_ct * 1.05 or p.d_h[j] <= do:
                raise ValueError(f"The coiled tubing (OD {ct.od_in:g} in) does not fit the flow path at {m:,.0f} ft MD.")
            # wetted perimeter of the annulus: the pipe plus the coiled tubing; area less the coiled tubing
            area = p.area[j] - a_ct
            perim = 4 * p.area[j] / p.d_h[j] + np.pi * do
            p.area[j] = area
            p.d_h[j] = 4 * area / perim
            p.d_ht[j] = p.d_h[j]          # heat transfer on the hydraulic diameter (PROSPER T25 wellhead temperatures)
            fm[j] = ANNULUS_FRICTION
    p.fric_mult = fm
    for k, d in list(p.restrictions.items()):
        if p.md[k] < depth_md:
            if d <= ct.od_in:
                raise ValueError(f"The coiled tubing (OD {ct.od_in:g} in) does not pass the {d:g} in restriction at "
                                 f"{p.md[k]:,.0f} ft MD.")
            p.restrictions[k] = float(np.sqrt(d * d - ct.od_in ** 2))
    return p


CT_GAS_Z = "prosper_lift"   # lift-gas Z in the coiled tubing: DPR on Standing's natural-gas pseudo-criticals (PROSPER's
#                             injection-string densities within 0.03 %, gaslift.LIFT_GAS_Z)
CT_GRAVITY = 1.029  # gas column gradient / (density x vertical): PROSPER's injection string (T25: 1.0291 +- 0.0005 on
#                     every segment of both gradient runs, with its reported densities)
CT_T_FRAC = 0.0    # lift gas temperature = T_flow - CT_T_FRAC (T_flow - T_formation): PROSPER, the flowing temperature


def gas_temperature(path: V.FlowPath, t_flow):
    """Lift-gas temperature in the coiled tubing (cases x nodes) from the flowing temperature (cases x nodes)."""
    t = np.atleast_2d(np.asarray(t_flow, float))
    return t - CT_T_FRAC * (t - path.t_form[None, :])


def gas_pressure(path: V.FlowPath, gas: GL.LiftGas, ct: CoiledTubing, p_surf, t_flow, qgi=0.0, depth_md=None):
    """Lift-gas pressure inside the coiled tubing at every node (cases x nodes), psia: a gas column from p_surf
    (cases) with the gas temperature from the flowing temperature t_flow (cases x nodes, or nodes), with friction
    down to depth_md when ct.friction."""
    gas = dataclasses.replace(gas, z_corr=CT_GAS_Z)
    ps = np.atleast_1d(np.asarray(p_surf, float))
    tgn = gas_temperature(path, t_flow)
    q = np.atleast_1d(np.asarray(qgi, float))
    nc = max(ps.size, tgn.shape[0], q.size)
    ps, q = (np.broadcast_to(x, (nc,)).copy() for x in (ps, q))
    tgn = np.broadcast_to(tgn, (nc, len(path.md)))
    n = len(path.md)
    out = np.empty((nc, n))
    p = ps.copy()
    out[:, 0] = p
    d = ct.id_in / 12.0
    area = np.pi / 4 * d * d
    e = ct.rough_in / 12.0
    bottom = np.broadcast_to(np.asarray(path.md[-1] if depth_md is None else depth_md, float), (nc,))
    for i in range(n - 1):
        dz, L = path.tvd[i + 1] - path.tvd[i], path.seg_len[i]
        flowing = ct.friction & (0.5 * (path.md[i] + path.md[i + 1]) < bottom)

        def grad(pp, tg):
            z, bg, rho, mu = GL.bo.gas_props(gas.inp, pp, tg)
            g = CT_GRAVITY * rho * dz / max(L, 1e-9) / 144.0
            if np.any(flowing):
                v = np.where(flowing, q, 0.0) * 1e6 / 86400.0 * bg / area
                re = rho * np.abs(v) * d / (mu * CP_TO_LBFTS)
                f = friction_factor(np.maximum(re, 1e-9), e / d)
                g = g - np.where(flowing, f * rho * v * np.abs(v) / (2 * 32.174 * d) / 144.0, 0.0)
            return g

        g0 = grad(p, tgn[:, i])
        g1 = grad(p + g0 * L, tgn[:, i + 1])
        p = p + 0.5 * (g0 + g1) * L
        out[:, i + 1] = p
    return out


# ================================================================ design
@dataclass
class CTDesignInput:
    design_rate: str = GL.DESIGN_RATE[1]
    q_design: float = 1000.0          # liquid STB/d (entered by user)
    q_liquid_max: float = 25000.0
    qgi_max: float = 5.0              # MMscf/d
    qgi_unload_max: float = 5.0
    top_p: float = 264.7              # flowing top-node pressure, psia
    top_p_unload: float = 264.7
    p_inj: float = 2214.7             # operating injection pressure at the coiled tubing head, psia
    p_ko: float = 2214.7              # kick-off injection pressure, psia
    dp_valve: float = 50.0            # desired dP across the orifice, psi
    max_md: float = 8500.0            # maximum depth of injection, ft MD
    gor: float = 800.0
    wc: float = 0.8                   # fraction
    load_grad: float = 0.46           # static gradient of the load fluid, psi/ft
    check_ipr: bool = True
    use_ipr_unloading: bool = True
    orifice_sizing: str = GL.ORIFICE_SIZING[0]
    economics: GL.Economics = field(default_factory=GL.Economics)


@dataclass
class CTDesignResult:
    inp: CTDesignInput
    q: float                          # design liquid rate, STB/d
    q_oil: float
    qgi_design: float                 # design gas (performance curve), MMscf/d
    qgi_actual: float                 # gas through the selected orifice, MMscf/d
    p_inj_actual: float               # injection pressure at the coiled tubing head, psia
    port_64: float                    # orifice size, 64ths in
    port_exact_64: float
    kickoff_md: float
    kickoff_tvd: float
    max_md: float                     # maximum operating depth, ft MD
    max_tvd: float
    pt_inj: float                     # flowing tubing pressure at the operating depth, psia
    pg_inj: float                     # lift-gas pressure there, psia
    t_inj: float
    perf: pd.DataFrame | None
    profile: pd.DataFrame
    messages: list
    std_dev: float = np.nan           # performance curve: points about the fitted curve


def _fit_curve(perf: pd.DataFrame):
    """Smooth performance curve: quadratic in the square root of the gas rate through the calculated points
    (PROSPER fits a curve to its points and reports their standard deviation about it)."""
    t = perf.dropna(subset=["Oil rate (STB/d)"])
    g = t["Gas injection rate (MMscf/d)"].to_numpy(float)
    y = t["Oil rate (STB/d)"].to_numpy(float)
    if len(g) < 4:
        return None, np.nan
    x = np.sqrt(g)
    c = np.polyfit(x, y, 3)
    res = y - np.polyval(c, x)
    return (lambda gg: np.polyval(c, np.sqrt(np.maximum(gg, 0.0)))), float(np.std(res, ddof=1))


def design(path: V.FlowPath, fluid, ipr, gas: GL.LiftGas, ct: CoiledTubing, inp: CTDesignInput,
           correlation="hagedorn_brown", p1=1.0, p2=1.0, u=8.0, hc: HeatCapacities | None = None, n_curve=11,
           progress=None) -> CTDesignResult:
    """Coiled tubing gas lift design (PROSPER Design | Coiled Tubing Gas Lift): design rate from the performance
    curve (or entered), maximum operating injection depth on the gas-lifted gradient at the design rate, orifice
    (Thornhill-Craver) and the gas it passes, and the kick-off depth the kick-off pressure unloads the load fluid to."""
    hc = hc or HeatCapacities()
    msgs = []
    max_md = min(float(inp.max_md), float(path.md[-1]))
    if max_md not in set(np.round(path.md, 6)):
        path = GL.with_nodes(path, [max_md])
    pa = annular_path(path, ct, max_md)

    def say(m):
        if progress:
            progress(m)

    # gas temperature from the flowing temperature of a first lifted traverse (half the AOF, the gas available)
    q0 = 0.5 * float(np.max(ipr.q))
    lt0 = GL.lifted_traverse(pa, fluid, inp.top_p, {"q": [q0], "wc": inp.wc, "gor": inp.gor}, [inp.qgi_max],
                             correlation, p1, p2, u, hc, fixed_md=[max_md], gas_sg=gas.sg)
    pg = gas_pressure(pa, gas, ct, inp.p_inj, lt0.T[0], inp.qgi_max, max_md)[0]

    perf = None
    sd = np.nan
    if inp.design_rate == GL.DESIGN_RATE[0]:
        q_d, qgi_d = float(inp.q_design), float(inp.qgi_max)
    else:
        say("Performance curve...")
        perf = GL.performance_curve(pa, fluid, ipr, inp.top_p, inp.wc, inp.gor, inp.qgi_max * 2.0, correlation, p1,
                                    p2, u, hc, pc=pg, dp_valve=inp.dp_valve, max_md=max_md, gas_sg=gas.sg, n=n_curve,
                                    economics=inp.economics)
        by = "Revenue ($/d)" if inp.design_rate == GL.DESIGN_RATE[2] else "Oil rate (STB/d)"
        opt = GL.optimum(perf, by, inp.qgi_max, inp.q_liquid_max)
        if opt is None or opt["q"] <= 0:
            raise ValueError("The well does not produce at any lift-gas rate up to the maximum gas available.")
        q_d, qgi_d = opt["q"], opt["qgi"]
        fit, sd = _fit_curve(perf)
        msgs.append(f"Design rate: {qgi_d:.4g} MMscf/d lift gas, {q_d:,.1f} STB/d liquid ({opt['note']}).")
    q_d = min(q_d, inp.q_liquid_max)
    if inp.check_ipr:
        say("Rate conformance with the IPR...")
        op = GL.solve_rate(pa, fluid, ipr, inp.top_p, inp.wc, inp.gor, [qgi_d], correlation, p1, p2, u, hc, pg,
                           inp.dp_valve, max_md, gas.sg)[0]
        if op.q is None:
            raise ValueError("The well does not flow at the design gas rate.")
        if op.q < q_d - 1e-6:
            msgs.append(f"Design rate lowered from {q_d:,.1f} to {op.q:,.1f} STB/d to conform with the IPR.")
        if inp.design_rate == GL.DESIGN_RATE[0] or op.q < q_d:
            q_d = float(op.q)

    # operating point: deepest injection at the design rate (gas pressure - desired dP reaches the tubing pressure)
    say("Operating injection depth...")
    lt = GL.lifted_traverse(pa, fluid, inp.top_p, {"q": [q_d], "wc": inp.wc, "gor": inp.gor}, [qgi_d], correlation,
                            p1, p2, u, hc, pc=pg, dp_valve=inp.dp_valve, max_md=max_md, gas_sg=gas.sg)
    if lt.no_injection[0]:
        raise ValueError("The injection pressure is below the flowing tubing pressure plus the desired dP at the "
                         "top - raise the injection pressure or lower the top-node pressure.")
    d_op = float(lt.inj_md[0])
    t_inj = float(np.interp(d_op, pa.md, lt.T[0]))
    pg = gas_pressure(pa, gas, ct, inp.p_inj, lt.T[0], qgi_d, d_op)[0]
    pt_d = float(lt.inj_p[0])
    pg_d = float(np.interp(d_op, pa.md, pg))

    # orifice at the coiled tubing end
    q_req = qgi_d * 1000.0
    t_gi = float(np.interp(d_op, pa.md, gas_temperature(pa, lt.T[0])[0]))
    exact = float(GL.port_for_rate(q_req, pg_d, pt_d, t_gi, gas.sg)) * 64
    if inp.orifice_sizing == GL.ORIFICE_SIZING[0]:
        port = float(max(np.floor(exact + 1e-9), 1.0))       # PROSPER: the whole 64th below (T25: 21.6 -> 21)
        q_act = float(GL.thornhill_craver(port / 64, pg_d, pt_d, t_gi, gas.sg)) / 1000.0
    else:
        port = float(np.ceil(exact - 1e-9))
        q_act = qgi_d
    msgs.append(f"Orifice {port:g}/64 in ({exact:.2f}/64 passes the design gas at {pg_d - pt_d:,.1f} psi): "
                f"{q_act:.4f} MMscf/d.")

    # kick-off: the depth the kick-off pressure unloads the static load fluid to
    pko = gas_pressure(pa, gas, ct, inp.p_ko, lt.T[0], 0.0)[0]
    load = inp.top_p_unload + inp.load_grad * (pa.tvd - pa.tvd[0])
    f = pko - inp.dp_valve - load
    ko = max_md
    if f[np.searchsorted(pa.md, max_md)] < 0:
        k = np.where(f < 0)[0][0]
        ko = float(pa.md[k - 1] + f[k - 1] / (f[k - 1] - f[k]) * (pa.md[k] - pa.md[k - 1])) if k > 0 else pa.md[0]
        msgs.append(f"Kick-off: the kick-off pressure unloads the load fluid to {ko:,.1f} ft MD; deeper injection "
                    "follows as the well unloads (no unloading valves).")
    prof = pd.DataFrame({"Measured depth (ft)": pa.md, "True vertical depth (ft)": pa.tvd,
                         "Flowing tubing pressure (psia)": lt.P[0], "Flowing temperature (°F)": lt.T[0],
                         "Lift-gas pressure, operating (psia)": pg, "Lift-gas pressure, kick-off (psia)": pko,
                         "Load fluid (psia)": load})
    return CTDesignResult(inp, q_d, q_d * (1 - inp.wc), qgi_d, q_act, inp.p_inj, port, exact, ko,
                          float(np.interp(ko, pa.md, pa.tvd)), d_op, float(np.interp(d_op, pa.md, pa.tvd)), pt_d, pg_d,
                          t_inj, perf, prof, msgs, sd)
