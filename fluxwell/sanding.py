"""Critical drawdown for sanding - Willson et al. (2002, SPE/ISRM 78168), as used by Palmer et al. (2003, SPE 84499):

    CBHFP = (3 sigma_1 - sigma_3 - U - A Pr) / (2 - A),   critical drawdown = Pr - CBHFP,
    A = (1 - 2 nu) alpha / (1 - nu),   U = 3.1 x TWC  (lab to field scale factor; TWC = thick-walled cylinder strength,
    measured or k x UCS),

sigma_1 / sigma_3 the largest / smallest total stress normal to the wellbore: a vertical well sees sigma_H and sigma_h; a
horizontal well sees the overburden Sv and the horizontal stress normal to its azimuth (sigma_h for a well drilled along
sigma_H, sigma_H for one drilled along sigma_h). Depletion lowers the horizontal stresses by the stress path
gamma = d(sigma_h)/d(Pp) times (Pi - Pr); Sv and U are unchanged. The envelope uses the critical drawdown less a
safety margin: minimum flowing BHP = Pr - (1 - margin) x max(0, critical drawdown).

Checked against the user's Critical Drawdown Calculator workbook (Calculator, Sensitivity and Depletion sheets) exactly."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCALE_U = 3.1


@dataclass
class SandingInputs:
    tvd: float                    # ft, mid-perforation
    p_initial: float              # psia, the reservoir pressure at which the stress gradients apply
    sv_grad: float = 1.0          # psi/ft overburden
    sh_grad: float = 0.75         # psi/ft minimum horizontal stress
    sH_ratio: float = 1.0         # sigma_H / sigma_h
    nu: float = 0.25              # static Poisson's ratio
    biot: float = 1.0
    twc: float | None = 2000.0    # psi, measured TWC; None -> ucs * twc_ucs
    ucs: float = 800.0
    twc_ucs: float = 2.5
    margin: float = 0.2           # fraction taken off the critical drawdown
    gamma: float = 0.6            # stress path
    well: str = "vertical"        # "vertical" or "horizontal"
    azimuth: str = "along sigma_H"  # horizontal well: "along sigma_H" (sees sigma_h) or "along sigma_h" (sees sigma_H)

    @property
    def twc_used(self):
        return self.twc if self.twc is not None else self.ucs * self.twc_ucs

    @property
    def u(self):
        return SCALE_U * self.twc_used

    @property
    def a(self):
        return (1 - 2 * self.nu) * self.biot / (1 - self.nu)


def stresses(inp: SandingInputs, pr=None):
    """(Sv, sigma_h, sigma_H) in psi at reservoir pressure pr (depleted from p_initial along the stress path)."""
    pr = inp.p_initial if pr is None else pr
    sv = inp.sv_grad * inp.tvd
    sh = inp.sh_grad * inp.tvd - inp.gamma * (inp.p_initial - pr)
    return sv, sh, sh * inp.sH_ratio


def principal_normal(inp: SandingInputs, pr=None):
    """(sigma_1, sigma_3) normal to the wellbore."""
    sv, sh, sH = stresses(inp, pr)
    if inp.well == "horizontal":
        other = sh if inp.azimuth == "along sigma_H" else sH
        pair = (sv, other)
    else:
        pair = (sH, sh)
    return max(pair), min(pair)


def cbhfp(inp: SandingInputs, pr=None):
    """Critical bottom-hole flowing pressure (psia) at reservoir pressure pr."""
    pr = inp.p_initial if pr is None else pr
    s1, s3 = principal_normal(inp, pr)
    return (3 * s1 - s3 - inp.u - inp.a * pr) / (2 - inp.a)


def critical_drawdown(inp: SandingInputs, pr=None):
    pr = inp.p_initial if pr is None else pr
    return pr - cbhfp(inp, pr)


def drawdown_limit(inp: SandingInputs, pr=None):
    """Drawdown allowed on the envelope: (1 - margin) x critical drawdown, not below 0."""
    return max(0.0, critical_drawdown(inp, pr) * (1 - inp.margin))


def min_pwf(inp: SandingInputs, pr=None):
    """Sanding line: the lowest flowing BHP (psia) allowed at reservoir pressure pr."""
    pr = inp.p_initial if pr is None else pr
    return pr - drawdown_limit(inp, pr)


def verdict(inp: SandingInputs, pr=None):
    cd = critical_drawdown(inp, pr)
    if cd <= 0:
        return "Sand control required: the rock fails at any drawdown"
    if cd < 300:
        return "Marginal: very small drawdown window"
    return "Drawdown window available"


def eaton_sh(inp: SandingInputs, pr=None):
    """Eaton check on sigma_h (no tectonic term): nu/(1-nu) (Sv - alpha Pp) + alpha Pp."""
    pr = inp.p_initial if pr is None else pr
    sv = inp.sv_grad * inp.tvd
    return inp.nu / (1 - inp.nu) * (sv - inp.biot * pr) + inp.biot * pr


def depletion_table(inp: SandingInputs, pressures):
    rows = []
    for pr in np.atleast_1d(pressures):
        sv, sh, sH = stresses(inp, pr)
        rows.append({"Pr (psia)": pr, "sigma_h (psi)": sh, "sigma_H (psi)": sH, "CBHFP (psia)": cbhfp(inp, pr),
                     "Critical drawdown (psi)": critical_drawdown(inp, pr),
                     "Envelope limit (psi)": drawdown_limit(inp, pr), "Min Pwf (psia)": min_pwf(inp, pr)})
    return rows
