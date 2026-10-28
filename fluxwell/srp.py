"""Sucker rod pump design (after PROSPER Design | Sucker Rod Pump, Tutorial 28).

Pieces:
- API RP 11L Table 4.1 rod strings (rod number, plunger size -> taper %, average weight, elastic constant, frequency
  factor); PROSPER names them ROD<no>/<plunger index> (ROD99/05 = rod 99 with the 2.00 in plunger).
- Lufkin conventional pumping units (API RP 11E geometry from the Lufkin catalogue, fluxwell/data/pumping_units.csv).
- Polished rod motion from the unit geometry. As PROSPER does (fitted to its Tutorial 28 cards): the shortest crank
  hole whose stroke is at least the entered surface stroke, its motion scaled to the entered stroke; the torque
  factor is the real geometry's (unscaled).
- Damped wave equation for the rod string (finite differences, lumped masses, tapered sections), ideal full pump at
  the bottom (anchored tubing): plunger held while the load transfers, Fo on the upstroke, none on the downstroke.
  Damping c = nu pi a / (2 L) with the dimensionless damping factor nu (PROSPER's "Damping Factor").
- API RP 11L quantities: fluid load Fo = 0.340 G D^2 H, rod weight in fluid Wr L (1 - 0.128 G), 1/Kr = Er L,
  Coberly overtravel, counterbalance effect CBE = 1.06 (Wrf + Fo/2); rod stresses and the modified Goodman diagram
  (API RP 11BR): Sa = (T/4 + 0.5625 Smin) SF; torque = TF (PRL - SU) - CBE TF(90) sin(theta).
"""
from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

G_FT = 32.174
E_STEEL = 30.5e6                     # psi (wave equation, stretch)
ROD_D = {4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875, 8: 1.0, 9: 1.125, 10: 1.25}            # rod size code -> in
ROD_W = {4: 0.726, 5: 1.135, 6: 1.634, 7: 2.224, 8: 2.904, 9: 3.676, 10: 4.538}       # lb/ft (API RP 11L)
PLUNGERS = (1.06, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.25, 3.75, 4.75)        # PROSPER's /01 .. /11
TENSILE = {"C": 90000.0, "K": 90000.0, "D": 115000.0}                                 # minimum tensile, psi
SERVICE = {"Non-corrosive": 1.0, "Salt water": 0.65, "Hydrogen sulphide": 0.5}        # Goodman service factor
CALC_MODES = ("Enter production rate, estimate stroke rate", "Enter stroke rate, calculate production rate")
GIBBS_K = 1.11433
STROKE_LAG_S = 0.09332     # s of each stroke lost: PROSPER's actual Ev = Ev (1 - N lag / 60), Tutorial 28 at 20 and 22 spm
DAMP_K = GIBBS_K * 550 * G_FT * 3600 * 0.125 * 576 / (6 * 33000 * 2 * np.pi ** 2)    # c = DAMP_K / (N Sp), 1/s
INTAKE_METHODS = ("Entered value", "Calculated from IPR", "Calculated from fluid level")

# API RP 11L (1988) Table 4.1 - rod no. -> {plunger in: (Wr lb/ft, Er 1e-6 in/lb ft, Fc, % of each size, largest
# first)}; "All" for straight strings. Checked against PROSPER's rod database (user's export, 180 rows): the size
# split of 64/04 and 107/04-08 and Fc of 98/01-02 taken from it; PROSPER's own typing errors not followed (Er of
# 65/08-09 1.99 and 1.93, of rod 66 0.833 against API 0.990, 0.930, 0.883; Fc of 107/05 1.95 against 1.195).
_T41 = {
    44: {"All": (0.726, 1.990, 1.000, (100.0,))},
    54: {1.06: (0.908, 1.668, 1.138, (44.6, 55.4)), 1.25: (0.929, 1.633, 1.140, (49.5, 50.5)),
         1.50: (0.957, 1.584, 1.137, (56.4, 43.6)), 1.75: (0.990, 1.525, 1.122, (64.6, 35.4)),
         2.00: (1.027, 1.460, 1.095, (73.7, 26.3)), 2.25: (1.067, 1.391, 1.061, (83.4, 16.6)),
         2.50: (1.108, 1.318, 1.023, (93.5, 6.5))},
    55: {"All": (1.135, 1.270, 1.000, (100.0,))},
    64: {1.06: (1.164, 1.382, 1.229, (33.3, 33.1, 33.5)), 1.25: (1.211, 1.319, 1.215, (37.2, 35.9, 26.9)),
         1.50: (1.275, 1.232, 1.184, (42.3, 40.4, 17.3)), 1.75: (1.341, 1.141, 1.145, (47.4, 45.2, 7.4))},
    65: {1.06: (1.307, 1.138, 1.098, (34.4, 65.6)), 1.25: (1.321, 1.127, 1.104, (37.3, 62.7)),
         1.50: (1.343, 1.110, 1.110, (41.8, 58.2)), 1.75: (1.369, 1.090, 1.114, (46.9, 53.1)),
         2.00: (1.394, 1.070, 1.114, (52.0, 48.0)), 2.25: (1.426, 1.045, 1.110, (58.4, 41.6)),
         2.50: (1.460, 1.018, 1.099, (65.2, 34.8)), 2.75: (1.497, 0.990, 1.082, (72.5, 27.5)),
         3.25: (1.574, 0.930, 1.037, (88.1, 11.9))},
    66: {"All": (1.634, 0.883, 1.000, (100.0,))},
    75: {1.06: (1.566, 0.997, 1.191, (27.0, 27.4, 45.6)), 1.25: (1.604, 0.973, 1.193, (29.4, 29.8, 40.8)),
         1.50: (1.664, 0.935, 1.189, (33.3, 33.3, 33.3)), 1.75: (1.732, 0.892, 1.174, (37.8, 37.0, 25.1)),
         2.00: (1.803, 0.847, 1.151, (42.4, 41.3, 16.3)), 2.25: (1.875, 0.801, 1.121, (46.9, 45.8, 7.2))},
    76: {1.06: (1.802, 0.816, 1.072, (28.5, 71.5)), 1.25: (1.814, 0.812, 1.077, (30.6, 69.4)),
         1.50: (1.833, 0.804, 1.082, (33.8, 66.2)), 1.75: (1.855, 0.795, 1.088, (37.5, 62.5)),
         2.00: (1.880, 0.785, 1.093, (41.7, 58.3)), 2.25: (1.908, 0.774, 1.096, (46.5, 53.5)),
         2.50: (1.934, 0.764, 1.097, (50.8, 49.2)), 2.75: (1.967, 0.751, 1.094, (56.5, 43.5)),
         3.25: (2.039, 0.722, 1.078, (68.7, 31.3)), 3.75: (2.119, 0.690, 1.047, (82.3, 17.7))},
    77: {"All": (2.224, 0.649, 1.000, (100.0,))},
    85: {1.06: (1.883, 0.873, 1.261, (22.2, 22.4, 22.4, 33.0)), 1.25: (1.943, 0.841, 1.253, (23.9, 24.2, 24.3, 27.6)),
         1.50: (2.039, 0.791, 1.232, (26.7, 27.4, 26.8, 19.2)), 1.75: (2.138, 0.738, 1.201, (29.6, 30.4, 29.6, 10.6))},
    86: {1.06: (2.058, 0.742, 1.151, (22.6, 23.0, 54.4)), 1.25: (2.087, 0.732, 1.156, (24.3, 24.5, 51.2)),
         1.50: (2.133, 0.717, 1.162, (26.8, 27.0, 46.2)), 1.75: (2.185, 0.699, 1.164, (29.4, 30.0, 40.6)),
         2.00: (2.247, 0.679, 1.161, (32.8, 33.2, 34.0)), 2.25: (2.315, 0.656, 1.153, (36.9, 36.0, 27.1)),
         2.50: (2.385, 0.633, 1.138, (40.6, 39.7, 19.7)), 2.75: (2.455, 0.610, 1.119, (44.5, 43.3, 12.2))},
    87: {1.06: (2.390, 0.612, 1.055, (24.3, 75.7)), 1.25: (2.399, 0.610, 1.058, (25.7, 74.3)),
         1.50: (2.413, 0.607, 1.062, (27.7, 72.3)), 1.75: (2.430, 0.603, 1.066, (30.3, 69.7)),
         2.00: (2.450, 0.598, 1.071, (33.2, 66.8)), 2.25: (2.472, 0.594, 1.075, (36.4, 63.6)),
         2.50: (2.496, 0.588, 1.079, (39.9, 60.1)), 2.75: (2.523, 0.582, 1.082, (43.9, 56.1)),
         3.25: (2.575, 0.570, 1.084, (51.6, 48.4)), 3.75: (2.641, 0.556, 1.078, (61.2, 38.8)),
         4.75: (2.793, 0.522, 1.038, (83.6, 16.4))},
    88: {"All": (2.904, 0.497, 1.000, (100.0,))},
    96: {1.06: (2.382, 0.670, 1.222, (19.1, 19.2, 19.4, 42.3)), 1.25: (2.435, 0.655, 1.224, (20.5, 20.6, 20.6, 38.3)),
         1.50: (2.511, 0.633, 1.223, (22.4, 22.5, 22.8, 32.3)), 1.75: (2.607, 0.606, 1.213, (24.8, 25.0, 25.1, 25.1)),
         2.00: (2.703, 0.578, 1.196, (27.1, 27.9, 27.4, 17.6)), 2.25: (2.806, 0.549, 1.172, (29.6, 30.8, 29.8, 9.8))},
    97: {1.06: (2.645, 0.568, 1.120, (19.7, 20.0, 60.3)), 1.25: (2.670, 0.563, 1.124, (20.8, 21.2, 58.0)),
         1.50: (2.707, 0.556, 1.131, (22.5, 23.0, 54.5)), 1.75: (2.751, 0.548, 1.137, (24.5, 25.0, 50.5)),
         2.00: (2.801, 0.538, 1.141, (26.8, 27.4, 45.8)), 2.25: (2.856, 0.528, 1.143, (29.4, 30.2, 40.4)),
         2.50: (2.921, 0.515, 1.141, (32.5, 33.1, 34.4)), 2.75: (2.989, 0.503, 1.135, (36.1, 35.3, 28.6)),
         3.25: (3.132, 0.475, 1.111, (42.9, 41.9, 15.2))},
    98: {1.06: (3.068, 0.475, 1.043, (21.2, 78.8)), 1.25: (3.076, 0.474, 1.045, (22.2, 77.8)),
         1.50: (3.089, 0.472, 1.048, (23.8, 76.2)), 1.75: (3.103, 0.470, 1.051, (25.7, 74.3)),
         2.00: (3.118, 0.468, 1.055, (27.7, 72.3)), 2.25: (3.137, 0.465, 1.058, (30.1, 69.9)),
         2.50: (3.157, 0.463, 1.062, (32.7, 67.3)), 2.75: (3.180, 0.460, 1.066, (35.6, 64.4)),
         3.25: (3.231, 0.453, 1.071, (42.2, 57.8)), 3.75: (3.289, 0.445, 1.074, (49.7, 50.3)),
         4.75: (3.412, 0.428, 1.064, (65.7, 34.3))},
    99: {"All": (3.676, 0.393, 1.000, (100.0,))},
    107: {1.06: (2.977, 0.524, 1.184, (16.9, 16.8, 17.1, 49.2)), 1.25: (3.019, 0.517, 1.189, (17.9, 17.8, 18.0, 46.3)),
          1.50: (3.085, 0.506, 1.195, (19.5, 19.2, 19.4, 41.9)), 1.75: (3.158, 0.494, 1.197, (21.0, 21.0, 21.2, 36.9)),
          2.00: (3.238, 0.480, 1.195, (22.7, 22.8, 23.1, 31.4)), 2.25: (3.336, 0.464, 1.187, (25.0, 25.0, 25.0, 25.0)),
          2.50: (3.435, 0.447, 1.174, (26.9, 27.7, 27.1, 18.2)), 2.75: (3.537, 0.430, 1.156, (29.1, 30.2, 29.3, 11.3))},
    108: {1.06: (3.325, 0.447, 1.097, (17.3, 17.8, 64.9)), 1.25: (3.345, 0.445, 1.101, (18.1, 18.6, 63.3)),
          1.50: (3.376, 0.441, 1.106, (19.4, 19.9, 60.7)), 1.75: (3.411, 0.437, 1.111, (20.9, 21.4, 57.7)),
          2.00: (3.452, 0.432, 1.117, (22.6, 23.0, 54.4)), 2.25: (3.498, 0.427, 1.121, (24.5, 25.0, 50.5)),
          2.50: (3.548, 0.421, 1.124, (26.5, 27.2, 46.3)), 2.75: (3.603, 0.416, 1.126, (28.7, 29.6, 41.7)),
          3.25: (3.731, 0.400, 1.123, (34.6, 33.9, 31.5)), 3.75: (3.873, 0.383, 1.108, (40.6, 39.5, 19.9))},
    109: {1.06: (3.839, 0.378, 1.035, (18.9, 81.1)), 1.25: (3.845, 0.378, 1.036, (19.6, 80.4)),
          1.50: (3.855, 0.377, 1.038, (20.7, 79.3)), 1.75: (3.867, 0.376, 1.040, (22.1, 77.9)),
          2.00: (3.880, 0.375, 1.043, (23.7, 76.3)), 2.25: (3.896, 0.374, 1.046, (25.4, 74.6)),
          2.50: (3.911, 0.372, 1.048, (27.2, 72.8)), 2.75: (3.930, 0.371, 1.051, (29.4, 70.6)),
          3.25: (3.971, 0.367, 1.057, (34.2, 65.8)), 3.75: (4.020, 0.363, 1.063, (39.9, 60.1)),
          4.75: (4.120, 0.354, 1.066, (51.5, 48.5))},
    1010: {"All": (4.538, 0.318, 1.000, (100.0,))},
}


def _codes(no: int):
    """Largest and smallest rod size codes (eighths of an inch) of an API rod number: 76 -> (7, 6), 108 -> (10, 8),
    1010 -> (10, 10)."""
    s = str(no)
    if s.startswith("10"):
        hi, rest = 10, s[2:]
    else:
        hi, rest = int(s[0]), s[1:]
    return hi, int(rest)


@dataclass
class RodString:
    no: int
    plunger: float
    wr: float                    # average weight, lb/ft (in air)
    er: float                    # elastic constant, in per lb ft
    fc: float                    # frequency factor
    sizes: tuple                 # rod diameters, in (top first)
    pct: tuple                   # % of each size (top first)
    grade: str = "D"

    @property
    def name(self):
        return f"ROD{self.no}/{PLUNGERS.index(self.plunger) + 1:02d}" if self.plunger in PLUNGERS else f"ROD{self.no}"

    def sections(self, length):
        """(length ft, area in2, weight lb/ft) per section, top first."""
        out = []
        for d, p in zip(self.sizes, self.pct):
            code = {v: k for k, v in ROD_D.items()}[d]
            out.append((length * p / 100.0, np.pi / 4 * d * d, ROD_W[code]))
        return out


def rod_numbers(plunger: float):
    """API rod numbers PROSPER offers for a plunger size: the straight strings 44 - 99 and the two- and three-size
    tapers that Table 4.1 lists for the plunger (no four-size tapers, no 1010)."""
    return [n for n, t in _T41.items() if n != 1010 and ("All" in t or (plunger in t and len(t[plunger][3]) < 4))]


def rod_string(no: int, plunger: float, grade="D") -> RodString:
    t = _T41[no]
    row = t.get("All") or t.get(plunger)
    if row is None:
        raise ValueError(f"API RP 11L Table 4.1 has no rod {no} for a {plunger} in plunger.")
    hi, lo = _codes(no)
    codes = list(range(hi, lo - 1, -1))[:len(row[3])]
    return RodString(no, plunger, row[0], row[1] * 1e-6, row[2], tuple(ROD_D[c] for c in codes), tuple(row[3]), grade)


# ----------------------------------------------------------------------------------------------- pumping units
@dataclass
class PumpingUnit:
    name: str
    A: float
    C: float
    I: float
    P: float
    H: float
    G: float
    radii: tuple                 # crank radii, in (largest stroke first)
    strokes: tuple               # nominal strokes, in
    su: float                    # structural unbalance, lb
    gear: float                  # gear reducer rating, in-lb
    prl: float                   # structure rating, lb
    max_stroke: float
    kind: str = "Conventional"
    prosper_holes: tuple = ()    # (stroke in, crank radius in) of the crank holes in PROSPER's unit database
    source: str = "Lufkin catalogue"


def load_units(path=None) -> dict:
    path = path or Path(__file__).parent / "data" / "pumping_units.csv"
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            u = PumpingUnit(r["Unit"], float(r["A (in)"]), float(r["C (in)"]), float(r["I (in)"]), float(r["P (in)"]),
                            float(r["H (in)"]), float(r["G (in)"]),
                            tuple(float(x) for x in r["Crank radii R (in)"].split()),
                            tuple(float(x) for x in r["Strokes (in)"].split()),
                            float(r["Structural unbalance (lb)"]), float(r["Gear reducer rating (in-lb)"]),
                            float(r["Structure rating (lb)"]), float(r["Maximum stroke (in)"]), r["Geometry"])
            out[u.name] = u
    _add_prosper_units(out)
    return out


def _add_prosper_units(units: dict, path=None):
    """PROSPER's pumping unit database (user's export, data/prosper_pumping_units.csv): one row per crank hole it
    holds - often only one, and not always a catalogue hole (88 in on the 144 in units). Its geometry agrees with the
    Lufkin catalogue (H - G = sqrt(K^2 - I^2)). The holes are attached to the catalogue unit of the same designation
    (C-320-305-100 -> C-320D-305-100), whose motion PROSPER computes on its (shortest) database hole; units the
    catalogue does not have are added under PROSPER's designation (gear reducer and structure ratings and maximum
    stroke from the designation)."""
    path = path or Path(__file__).parent / "data" / "prosper_pumping_units.csv"
    try:
        df = pd.read_csv(path)
    except OSError:
        return
    for (des, uid), d in df.groupby(["designation", "unit_id"], sort=False):
        r = d.iloc[0]
        holes = tuple(sorted(zip(d["stroke_in"].astype(float), d["R_in"].astype(float))))
        man = str(r["manufacturer"]).upper()
        p = str(des).split("-")
        cat = f"{man} C-{p[1]}D-{p[2]}-{p[3]}" if len(p) == 4 else None
        if cat in units and not units[cat].prosper_holes:
            units[cat].prosper_holes = holes
            continue
        name = f"{man} {des}"
        if name in units:
            name = f"{name} ({uid})"
        hg = float(np.sqrt(float(r["K_in"]) ** 2 - float(r["I_in"]) ** 2))
        units[name] = PumpingUnit(name, float(r["A_in"]), float(r["C_in"]), float(r["I_in"]), float(r["P_in"]), hg, 0.0,
                                  tuple(h[1] for h in holes[::-1]), tuple(h[0] for h in holes[::-1]),
                                  float(r["structural_unbalance_lb"]), float(p[1]) * 1000.0, float(p[2]) * 100.0,
                                  float(p[3]), str(r["geometry"]), holes, "PROSPER unit database")


UNITS = load_units()


class Kinematics:
    """Polished rod motion of a conventional unit (crank angle theta from 12 o'clock, clockwise with the well to the
    right). As PROSPER: the shortest-stroke crank hole, its motion scaled
    to the surface stroke; position 0 at the bottom of the stroke."""

    def __init__(self, unit: PumpingUnit, stroke: float, n=7201):
        self.unit, self.stroke = unit, float(stroke)
        # PROSPER always takes the shortest-stroke crank hole and scales its motion to the entered stroke, also for a
        # stroke longer than that hole's (Tutorial 28 runs: 48 in and 80 in on the C-320D-305-100 both on the 70 in
        # hole; the 85 in hole gives a third more peak torque than PROSPER at 80 in)
        # PROSPER's unit database often holds a single hole per unit (the C-320D-305-100: only the 70 in one), which
        # explains this; where it lists holes, its shortest is used
        self.hole_stroke, self.R = min(unit.prosper_holes) if unit.prosper_holes else min(zip(unit.strokes, unit.radii))
        th = np.linspace(0.0, 2 * np.pi, n)
        b = self._beam(th)
        self._b0, self._span = b.min(), np.ptp(b)
        self.scale = self.stroke / (unit.A * self._span)
        self.theta_bottom = float(th[np.argmin(b)])

    def _beam(self, th):
        u = self.unit
        th = np.atleast_1d(np.asarray(th, float))
        x, y = self.R * np.sin(th), self.R * np.cos(th)
        dx, dy = x - u.I, y - (u.H - u.G)
        j = np.hypot(dx, dy)
        gam = np.arccos(np.clip((u.C ** 2 + j ** 2 - u.P ** 2) / (2 * u.C * j), -1.0, 1.0))
        a = np.arctan2(dy, dx)
        c1, c2 = a + gam, a - gam
        b = np.where(np.sin(c1) > np.sin(c2), c1, c2)
        return np.unwrap(b) if b.size > 1 else b

    def position(self, th):
        """Polished rod position above the bottom of the stroke, in."""
        return (self._beam(th) - self._b0) * self.unit.A * self.scale

    def torque_factor(self, th, h=1e-5):
        """Torque factor of the unit (in, real geometry: d(position) / d(theta) without the stroke scaling)."""
        th = np.atleast_1d(np.asarray(th, float))
        return (self._beam(th + h) - self._beam(th - h)) / (2 * h) * self.unit.A


# ----------------------------------------------------------------------------------------------- wave equation
def simulate(kin: Kinematics, n_spm: float, sections, fo: float, nu: float, nseg=120, cycles=20, tol=0.5):
    """Damped wave equation for the rod string driven by the polished rod motion, ideal full pump (anchored tubing).
    sections: (length ft, area in2, weight lb/ft) top first. nu: damping factor, c = nu pi a / (2 L).
    Returns dict of arrays over one cycle (theta rad, surface position in, surface dynamic load lb, plunger position
    in, pump load lb above the static) and the damping coefficient."""
    L = sum(s[0] for s in sections)
    dxs, ea, m = [], [], []
    for ls, a_, w in sections:
        k = max(int(round(nseg * ls / L)), 2)
        dxs += [ls / k] * k
        ea += [E_STEEL * a_] * k
        m += [w / G_FT] * k
    dxs, ea, m = np.array(dxs), np.array(ea), np.array(m)
    a_s = np.sqrt(ea / m)                                    # ft/s
    a_avg = L / np.sum(dxs / a_s)                            # travel-time average
    c = nu * np.pi * a_avg / (2 * L)
    T = 60.0 / n_spm
    dt = 0.9 * np.min(dxs / a_s)
    nst = int(np.ceil(T / dt))
    dt = T / nst
    nn = len(dxs) + 1
    mn = np.zeros(nn)
    mn[:-1] += m * dxs / 2
    mn[1:] += m * dxs / 2
    kel = ea / dxs
    cd2 = c * dt / 2
    th = 2 * np.pi * (np.arange(1, nst + 1) / nst) + kin.theta_bottom
    xtop = kin.position(th) / 12.0
    u = np.zeros(nn)
    uold = np.zeros(nn)
    u[:] = uold[:] = xtop[-1]
    state = 0
    prev = None
    for _ in range(cycles):
        rec = np.empty((nst, 4))
        for i in range(nst):
            ten = kel * (u[:-1] - u[1:])
            f = np.zeros(nn)
            f[:-1] -= ten
            f[1:] += ten
            unew = (2 * u - uold * (1 - cd2) + f / mn * dt * dt) / (1 + cd2)       # damping centred: stable
            unew[0] = xtop[i]
            ten_l = kel[-1] * (unew[-2] - u[-1])
            if state in (0, 2):
                if ten_l >= fo:
                    state = 1
                elif ten_l <= 0.0:
                    state = 3
            if state in (1, 3):
                load = fo if state == 1 else 0.0
                unew[-1] = (2 * u[-1] - uold[-1] * (1 - cd2) + (ten[-1] - load) / mn[-1] * dt * dt) / (1 + cd2)
                if state == 1 and unew[-1] < u[-1]:
                    state, unew[-1] = 2, u[-1]
                elif state == 3 and unew[-1] > u[-1]:
                    state, unew[-1] = 0, u[-1]
            else:
                unew[-1] = u[-1]
            uold, u = u, unew
            rec[i] = (u[0] * 12, kel[0] * (u[0] - u[1]), u[-1] * 12, kel[-1] * (u[-2] - u[-1]))
        if prev is not None and np.max(np.abs(rec[:, 1] - prev[:, 1])) < tol:
            break
        prev = rec
    xp = rec[:, 2] - rec[:, 2].min()
    return {"theta": th - kin.theta_bottom, "x": rec[:, 0],
            "f_dyn": rec[:, 1], "x_pump": xp, "f_pump": rec[:, 3], "c": c, "a": a_avg}


# ----------------------------------------------------------------------------------------------- design
@dataclass
class SRPInput:
    unit: str = "LUFKIN C-320D-305-100"
    rod_no: int = 99
    plunger: float = 2.0                  # in (pump diameter)
    grade: str = "D"
    service: str = "Non-corrosive"
    mode: str = CALC_MODES[0]
    q_target: float = 400.0               # STB/d (mode 0)
    n_spm: float = 20.0                   # strokes/min (mode 1)
    stroke: float = 48.0                  # surface stroke length, in
    pump_md: float = 3500.0
    pump_tvd: float | None = None
    pip: float = 666.1                    # pump intake pressure, psia
    whp: float = 114.7                    # wellhead pressure, psia
    wc: float = 0.8
    api: float = 25.0
    sg_water: float = 1.0556              # at standard conditions (water_sg(salinity))
    ev: float = 0.80                      # pump volumetric efficiency
    unit_eff: float = 0.75
    anchored: bool = True
    nu: float | None = None               # damping factor; None = PROSPER's estimate (rod friction 1/8 of the
    #                                       rod weight, Gibbs damping)
    bl: float = 1.0                       # liquid formation volume factor at the intake (production = displacement
    #                                       x Ev / Bl)


@dataclass
class SRPResult:
    inp: SRPInput
    rods: RodString
    kin: Kinematics
    n_spm: float
    q: float
    ev_actual: float
    sp: float
    pprl: float
    mprl: float
    fo: float
    wrf: float
    w_air: float
    kr_inv: float
    stretch: float
    ep: float
    cbe: float
    torque_max: float
    stress_max: float
    stress_min: float
    stress_allow: float
    goodman_pct: float
    top_rod_loading_pct: float
    card_hp: float               # surface card area x N (PROSPER's "work done by polished rod" per stroke)
    pump_hp: float
    hhp: float
    hf: float                    # frictional power from the damping factor (Gibbs)
    prhp: float                  # polished rod power = hydraulic + frictional (as PROSPER reports it)
    ite: float
    nu: float
    sg: float
    card: pd.DataFrame = field(default_factory=pd.DataFrame)
    clf: float = float("nan")

    @property
    def nameplate(self):
        """PROSPER: polished rod power x CLF / entered volumetric efficiency (Tutorial 28, Ev 80 and 100 %)."""
        return self.prhp * self.clf / self.inp.ev

    @property
    def lift_eff(self):
        """Lift efficiency, % (PROSPER): hydraulic power of the theoretical pump displacement (0.1166 Sp N D^2) against a
        fresh-water column at pump depth, Q dP / 58800 with dP = 0.433 TVD, over the polished rod power."""
        tvd = self.inp.pump_tvd if self.inp.pump_tvd is not None else self.inp.pump_md
        pd_ = 0.1166 * self.sp * self.n_spm * self.inp.plunger ** 2
        return 100 * pd_ * 0.433 * tvd / 58800 / self.prhp

    @property
    def econ_index(self):
        """Economic index (PROSPER): 1e-7 PPRL x peak torque x PRHP / LE(%); lower is better."""
        return 1e-7 * self.pprl * self.torque_max * self.prhp / self.lift_eff

    def summary(self) -> dict:
        return {
            "Actual liquid production rate (STB/d)": self.q,
            "Actual pump speed (strokes/min)": self.n_spm,
            "Volumetric efficiency (%)": 100 * self.ev_actual,
            "Pump stroke length (in)": self.sp,
            "Peak polished rod load (lbf)": self.pprl,
            "Minimum polished rod load (lbf)": self.mprl,
            "Fluid load Fo (lbf)": self.fo,
            "Weight of rods in fluid Wrf (lbf)": self.wrf,
            "Total load Wrf + Fo (lbf)": self.wrf + self.fo,
            "Static stretch (in)": self.stretch,
            "Plunger overtravel Ep (in)": self.ep,
            "1 / Kr (in/lb)": self.kr_inv,
            "Fo / SKr": self.fo * self.kr_inv / self.inp.stroke,
            "Counterweight required CBE (lbf)": self.cbe,
            "Maximum torque (lb-in)": self.torque_max,
            "Gear reducer loading (%)": 100 * self.torque_max / self.kin.unit.gear,
            "Structure loading (%)": 100 * self.pprl / self.kin.unit.prl,
            "Stress max (psi)": self.stress_max,
            "Stress min (psi)": self.stress_min,
            "Stress max allowable (psi)": self.stress_allow,
            "Top rod % of Goodman diagram": self.goodman_pct,
            "Top rod loading (%)": self.top_rod_loading_pct,
            "Polished rod power (hp)": self.prhp,
            "Frictional power (hp)": self.hf,
            "Hydraulic power (hp)": self.hhp,
            "Work done by polished rod (ft-lbf/stroke)": self.card_hp * 33000 / self.n_spm,
            "Work done by pump (ft-lbf/stroke)": self.pump_hp * 33000 / self.n_spm,
            "Torsional effectiveness ITE (%)": self.ite,
            "Damping factor": self.nu,
            "Cyclic load factor (approx.)": self.clf,
            "Nameplate power (hp)": self.nameplate,
            "Lift efficiency (%)": self.lift_eff,
            "Economic index": self.econ_index,
            "Crank hole (nominal stroke, in)": self.kin.hole_stroke,
        }


def water_sg(salinity_ppm):
    """Brine gravity at standard conditions (fits PROSPER's Tutorial 28 loads: 80000 ppm -> 1.0556)."""
    return 1.0 + 0.695e-6 * salinity_ppm


def pip_from_pwf(pwf, sg, pump_tvd, perf_tvd):
    """Pump intake pressure from the flowing BHP with a liquid column between the pump and the perforations."""
    return pwf - 0.433 * sg * max(perf_tvd - pump_tvd, 0.0)


def fluid_sg(api, sg_water, wc):
    return wc * sg_water + (1 - wc) * 141.5 / (131.5 + api)


def _run(inp: SRPInput, n_spm, rods: RodString, kin: Kinematics, sg):
    L = inp.pump_md
    tvd = inp.pump_tvd if inp.pump_tvd is not None else L
    ap = np.pi / 4 * inp.plunger ** 2
    h_net = tvd - (inp.pip - inp.whp) / (0.433 * sg)
    fo = 0.340 * sg * inp.plunger ** 2 * h_net
    w_air = rods.wr * L
    wrf = w_air * (1 - 0.128 * sg)
    secs = rods.sections(L)
    T = 60.0 / n_spm
    if inp.nu is not None:
        sim = simulate(kin, n_spm, secs, fo, inp.nu)
        nu = inp.nu
    else:
        # PROSPER (fitted to Tutorial 28 at 20 and 22 spm and its rod sensitivity): frictional power
        # Hf = W_air (2 Sp N / 12) / 8 / 33000 and Gibbs' damping c = K 550 g Hf T^2 / (2 pi^2 W_air (Sp/24)^2),
        # K = 1.11433, so c = DAMP_K / (N Sp); fixed point on the pump stroke.
        sp_ = inp.stroke
        a_bar = L / sum(ls / np.sqrt(E_STEEL * ar / (w / G_FT)) for ls, ar, w in secs)
        w1 = np.pi * a_bar / (2 * L)
        for _ in range(8):
            nu = DAMP_K / (n_spm * sp_) / w1
            sim = simulate(kin, n_spm, secs, fo, nu)
            sp_new = max(float(np.ptp(sim["x_pump"])), 0.02 * inp.stroke)
            if abs(sp_new - sp_) < 1e-3:
                break
            sp_ = sp_new
    f_surf = wrf + sim["f_dyn"]
    x = kin.position(sim["theta"] + kin.theta_bottom)
    buoy = w_air * 0.128 * sg
    f_pump = sim["f_pump"] - buoy
    sp = float(np.ptp(sim["x_pump"]))
    ev_act = inp.ev * max(1.0 - n_spm * STROKE_LAG_S / 60.0, 0.0) / max(inp.bl, 1e-6)
    q = 0.1166 * sp * n_spm * inp.plunger ** 2 * ev_act
    a_top = np.pi / 4 * rods.sizes[0] ** 2
    pprl, mprl = float(f_surf.max()), float(f_surf.min())
    smax, smin = pprl / a_top, mprl / a_top
    sa = (TENSILE.get(inp.grade, 115000.0) / 4 + 0.5625 * smin) * SERVICE.get(inp.service, 1.0)
    kr_inv = rods.er * L
    stretch = fo * sum(ls * 12 / (E_STEEL * ar) for ls, ar, _ in secs)
    ep = 40.8 * L ** 2 * (inp.stroke * n_spm ** 2 / 70471.2) / 30e6
    cbe = 1.06 * (wrf + fo / 2)
    th = sim["theta"] + kin.theta_bottom
    tf = kin.torque_factor(th)
    tf90 = float(kin.torque_factor(np.array([np.pi / 2]))[0])
    torque = tf * f_surf - cbe * tf90 * np.sin(th)
    xc, fc = np.r_[x, x[0]], np.r_[f_surf, f_surf[0]]
    area = abs(np.sum((xc[1:] - xc[:-1]) * (fc[1:] + fc[:-1]) / 2))
    card_hp = area * n_spm / (12 * 33000)
    xp_, fp_ = np.r_[sim["x_pump"], sim["x_pump"][0]], np.r_[f_pump, f_pump[0]]
    pump_area = abs(np.sum((xp_[1:] - xp_[:-1]) * (fp_[1:] + fp_[:-1]) / 2))
    pump_hp = pump_area * n_spm / (12 * 33000)
    hhp = 7.36e-6 * q * sg * h_net
    # Gibbs: c = K 550 g Hf T^2 / (2 pi^2 W (Sp/24)^2), K = 1.11433 (PROSPER, Tutorial 28 at both speeds)
    hf = sim["c"] * 2 * np.pi ** 2 * w_air * (sp / 24) ** 2 / (GIBBS_K * 550 * T * T * G_FT)
    prhp = hhp + hf
    t_avg = prhp * 63025 / n_spm
    # cyclic load factor: rms / mean of the absolute polished rod power (within 1.1 % of PROSPER's in four runs; its
    # own definition is not known)
    pw = f_surf * np.gradient(x)
    clf = float(np.sqrt(np.mean(pw ** 2)) / np.mean(np.abs(pw)))
    card = pd.DataFrame({"Crank angle (deg)": np.degrees(sim["theta"]) % 360, "Surface position (in)": x,
                         "Surface load (lbf)": f_surf, "Pump position (in)": sim["x_pump"],
                         "Pump load (lbf)": f_pump, "Torque (lb-in)": torque})
    return SRPResult(inp, rods, kin, n_spm, q, ev_act, sp, pprl, mprl, fo, wrf, w_air, kr_inv,
                     stretch, ep, cbe, float(torque.max()), smax, smin, sa, 100 * (smax - smin) / (sa - smin),
                     100 * smax / sa, card_hp, pump_hp, hhp, hf, prhp, 100 * t_avg / max(torque.max(), 1e-9), nu, sg, card, clf)


def design(inp: SRPInput, pwf_fn=None, perf_tvd=None) -> SRPResult:
    """Sucker rod pump design at the entered stroke rate, or the stroke rate that gives the target production.
    pwf_fn (q STB/d -> flowing BHP psia, from the IPR) sets the pump intake pressure at the target rate, in both
    modes (as PROSPER)."""
    if pwf_fn is not None:
        # as PROSPER: the intake pressure from the IPR at the target rate, also when the stroke rate is entered (its
        # 20 and 22 spm Tutorial 28 runs have the same fluid load)
        tvd = inp.pump_tvd if inp.pump_tvd is not None else inp.pump_md
        perf = perf_tvd if perf_tvd is not None else tvd
        sg = fluid_sg(inp.api, inp.sg_water, inp.wc)
        pip = max(pip_from_pwf(float(pwf_fn(inp.q_target)), sg, tvd, perf), 14.7)
        return design(dataclasses.replace(inp, pip=pip))
    unit = UNITS[inp.unit]
    rods = rod_string(inp.rod_no, inp.plunger, inp.grade)
    kin = Kinematics(unit, inp.stroke)
    sg = fluid_sg(inp.api, inp.sg_water, inp.wc)
    if inp.mode == CALC_MODES[1]:
        return _run(inp, inp.n_spm, rods, kin, sg)
    # target rate: secant on N from the static-stroke estimate; if that does not settle (rod resonance makes the
    # rate non-monotonic in N), march N up from a low value to the first N reaching the target, then bisect.
    n0 = inp.q_target / (0.1166 * inp.stroke * inp.plunger ** 2 * inp.ev / inp.bl)
    tol = 1e-3 * inp.q_target
    n, res = n0, _run(inp, n0, rods, kin, sg)
    for _ in range(8):
        if abs(res.q - inp.q_target) <= tol:
            return res
        n = min(max(n * inp.q_target / max(res.q, 1e-6), 0.5 * n), 2 * n)
        res = _run(inp, n, rods, kin, sg)
    if abs(res.q - inp.q_target) <= tol:
        return res
    lo, r_lo = 0.25 * n0, None
    n, r_hi = lo, None
    while n < 40.0:
        r = _run(inp, n, rods, kin, sg)
        if r.q >= inp.q_target:
            r_hi = r
            break
        lo, r_lo = n, r
        n *= 1.1
    if r_hi is None:
        raise ValueError(f"{rods.name}: the target rate is not reached below 40 strokes/min.")
    hi = n
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        r = _run(inp, mid, rods, kin, sg)
        if abs(r.q - inp.q_target) <= tol:
            return r
        if r.q < inp.q_target:
            lo = mid
        else:
            hi, r_hi = mid, r
    return r_hi


def rod_sensitivity(inp: SRPInput, rods=None) -> pd.DataFrame:
    rows = []
    for no in rods or rod_numbers(inp.plunger):
        try:
            r = design(SRPInput(**{**inp.__dict__, "rod_no": no}))
        except (ValueError, FloatingPointError, ZeroDivisionError) as e:   # noqa: PERF203
            rows.append({"Rod": f"ROD{no}", "Note": str(e)})
            continue
        rows.append({"Rod": r.rods.name, "Production (STB/d)": r.q, "Stroke rate (strokes/min)": r.n_spm,
                     "Polished rod power (hp)": r.prhp,
                     "Production per hp (STB/d/hp)": r.q / r.prhp if r.prhp > 0 else np.nan,
                     "Peak polished rod load (lbf)": r.pprl, "Top rod loading (%)": r.top_rod_loading_pct,
                     "Maximum torque (lb-in)": r.torque_max, "Lift efficiency (%)": r.lift_eff,
                     "Economic index": r.econ_index})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- system analysis
# PROSPER evaluates the pump-intake fluid (free gas, liquid volume) for the pump fillage at the intake temperature in
# deg C taken as deg F (130 F -> 54.4 F): the user's T28 lift curves with anchor efficiency 0, 0.2 and 1 and the six
# system runs reproduce only with it (at the true temperature the rate is 5-15 % low at GOR 800 and the liquid volume
# factor 1-2 % high). "prosper" mimics that; "actual" uses the true intake temperature.
PUMP_FILL_T = "prosper"
ANCHOR_FILL_T = "actual"
PUMP_FILL_BL = True          # pump rate in STB/d: the liquid volume through the pump / its volume factor     # the gas anchor correlation's in-situ rates (fitted at the true temperature)


def _fill_t(t_f, mode):
    return (t_f - 32.0) / 1.8 if mode == "prosper" else t_f


def _intake(fluid, pip, t_in, case, ea_fn):
    """Pump intake: (anchor efficiency, liquid fraction of the pump volume, liquid volume factor, rs)."""
    from .esp import _rates
    qa = _rates(fluid, pip, _fill_t(t_in, ANCHOR_FILL_T), case)
    ea = ea_fn(qa[0], qa[1])
    ql, qg, rs = qa if PUMP_FILL_T == ANCHOR_FILL_T else _rates(fluid, pip, _fill_t(t_in, PUMP_FILL_T), case)
    ql, qg = float(np.ravel(ql)[0]), float(np.ravel(qg)[0])
    lf = ql / (ql + (1.0 - ea) * qg) if ql > 0 else 1.0
    return ea, lf, ql / float(np.ravel(case["q"])[0]), rs


GAS_ANCHOR_TYPES = ("Cup type", "Packer type", "Poor boy")
GAS_ANCHOR_FIT = (0.070708, 0.50082, -0.55155)   # cup type: 1 - E = C vl^a GVF^b (four PROSPER runs, to 0.0001)


def gas_anchor_efficiency(ql_rbd, qg_rbd, annulus_in2, kind="Cup type"):
    """Gas anchor efficiency, 'Calculated' (PROSPER System | Sucker Rod Pumps), as the share of the gas it takes out
    of the pump feed. Fitted to four PROSPER runs of a cup-type anchor (1, 1.5 and 3 in2 at 23.7 spm, intake ~570
    psig: 0.4046, 0.5167, 0.6611; 1.5 in2 at 15 spm, intake 959 psig: 0.2514): 1 - E = 0.070708 vl^0.50082
    GVF^-0.55155, vl the in-situ liquid velocity in the anchor annulus (ft/s), GVF the no-slip free-gas fraction at
    the intake. No free gas -> E = 0. Packer type: the same as the cup type (PROSPER, identical to six decimals);
    poor boy: PROSPER's calculated efficiency is 0 (identical to an entered 0)."""
    if kind == "Poor boy":
        return 0.0
    if qg_rbd <= 0 or ql_rbd <= 0:
        return 0.0
    c, a, b = GAS_ANCHOR_FIT
    vl = ql_rbd * 5.614583 / 86400.0 / (annulus_in2 / 144.0)
    gvf = qg_rbd / (ql_rbd + qg_rbd)
    return float(np.clip(1.0 - c * vl ** a * gvf ** b, 0.0, 1.0))


def pump_stroke_api(stroke, n_spm, pump_tvd, fo, sections):
    """Downhole stroke as PROSPER's system calculation: Sp = S + Ep - Fo L / (E A) (Coberly overtravel, rod stretch
    under the fluid load; anchored tubing). Reproduces PROSPER's four T28 system runs to 0.01 in."""
    ep = 40.8 * pump_tvd ** 2 * (stroke * n_spm ** 2 / 70471.2) / 30e6
    stretch = fo * sum(ls * 12.0 / (E_STEEL * ar) for ls, ar, _ in sections)
    return stroke + ep - stretch


@dataclass
class SRPSystemInput:
    rod_no: int = 99
    plunger: float = 2.0
    stroke: float = 48.0
    n_spm: float = 20.0
    ev: float = 0.80
    pump_md: float = 3500.0
    anchor_method: str = "Calculated"          # or "Entered"
    anchor_eff: float = 0.0                    # when entered
    anchor_type: str = "Cup type"
    anchor_area: float = 1.5                   # in2


def system(inp: SRPSystemInput, path, fluid, ipr_pwf, wc, gor, top_p, correlation="no_slip", p1=1.0, p2=1.0,
           u=8.0, hc=None, q_max=None):
    """PROSPER sucker rod system analysis (solution node at the pump intake). For a liquid rate q: flowing BHP from
    the IPR, intake pressure up the casing to the pump, gas anchor efficiency E, discharge pressure from the top node
    down the tubing with a GOR of (1 - E) x total GOR (PROSPER takes the anchor's share of all the gas), fluid load
    Fo = Ap (Pdis - PIP), downhole stroke Sp = S + Ep - stretch, displacement 0.1166 Sp N D^2 and pump rate
    = displacement x Ev x liquid fraction of the pump intake volume (free gas x (1 - E) through the pump). The
    solution is the rate where the pump rate equals q. ipr_pwf(q) -> psia (NaN past the AOF). Default tubing model:
    no slip (homogeneous) - it reproduces PROSPER's discharge pressure in all six Tutorial 28 system runs (two rates)
    to 0.3 %, where the slip correlations miss by up to 4 % at the lower rate."""
    from scipy.optimize import brentq
    from . import vlp as V
    from .esp import sub_path, _rates
    rods = rod_string(inp.rod_no, inp.plunger)
    above = sub_path(path, path.md[0], inp.pump_md)
    below = sub_path(path, inp.pump_md, path.md[-1]) if path.md[-1] > inp.pump_md + 0.5 else None
    tvd = float(np.interp(inp.pump_md, path.md, path.tvd))
    secs = rods.sections(tvd)
    ap = np.pi / 4 * inp.plunger ** 2

    def at(q):
        pwf = float(ipr_pwf(q))
        if not np.isfinite(pwf):
            return None
        case = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([gor])}
        if below is not None:
            Pb, Tb = V.traverse_up(below, fluid, np.array([pwf]), case, correlation, p1, p2, u, hc)
            pip, t_in = float(Pb[0, 0]), float(Tb[0, 0])
        else:
            pip, t_in = pwf, float(path.t_form[-1])
        if not np.isfinite(pip) or pip <= 14.7:
            return None
        ea_fn = (lambda ql_, qg_: gas_anchor_efficiency(ql_, qg_, inp.anchor_area, inp.anchor_type)) \
            if inp.anchor_method == "Calculated" else (lambda ql_, qg_: float(np.clip(inp.anchor_eff, 0.0, 1.0)))
        ea, liq_frac, bl, rs = _intake(fluid, pip, t_in, case, ea_fn)
        case_a = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([(1.0 - ea) * gor])}
        P, T = V.traverse(above, fluid, np.array([top_p]), case_a, correlation, p1, p2, u, hc, t_in)
        pdis = float(P[0, -1])
        fo = ap * (pdis - pip)
        sp = pump_stroke_api(inp.stroke, inp.n_spm, tvd, fo, secs)
        disp = 0.1166 * sp * inp.n_spm * inp.plunger ** 2
        return {"q": q, "pwf": pwf, "pip": pip, "pdis": pdis, "t_in": t_in, "ea": ea, "fo": fo, "sp": sp,
                "disp": disp, "q_pump": disp * inp.ev * liq_frac / (bl if PUMP_FILL_BL else 1.0), "liq_frac": liq_frac, "rs": rs,
                "t_wh": float(T[0, 0])}

    q_hi = q_max or 20000.0
    qs = np.linspace(q_hi * 0.01, q_hi, 60)
    rows, k = [], None
    for q in qs:
        r = at(q)
        if r is None:
            break
        rows.append(r)
        if len(rows) > 1 and rows[-2]["q_pump"] - rows[-2]["q"] > 0 >= r["q_pump"] - r["q"]:
            k = len(rows) - 1
            break
    curve = pd.DataFrame(rows)
    if k is None:
        return None, curve
    q_sol = brentq(lambda q: at(q)["q_pump"] - q, rows[k - 1]["q"], rows[k]["q"], xtol=0.05)
    r = at(q_sol)
    sol = {"Liquid rate (STB/d)": q_sol, "Oil rate (STB/d)": q_sol * (1 - wc), "Water rate (STB/d)": q_sol * wc,
           "Gas rate (MMscf/d)": q_sol * (1 - wc) * gor / 1e6, "BHP (psia)": r["pwf"],
           "Pump intake pressure (psia)": r["pip"], "Pump discharge pressure (psia)": r["pdis"],
           "Downhole pump displacement (STB/d)": r["disp"], "Downhole stroke length (in)": r["sp"],
           "Gas anchor efficiency": r["ea"], "Fluid load Fo (lbf)": r["fo"],
           "Liquid fraction at pump intake": r["liq_frac"], "Wellhead temperature (F)": r["t_wh"]}
    return sol, curve


def rate_for_pwf(inp: SRPSystemInput, path, fluid, pwf, wc, gor, top_p, correlation="no_slip", p1=1.0, p2=1.0, u=8.0,
                 hc=None, q_hi=None):
    """PROSPER's sucker rod lift curve: the liquid rate (STB/d) the pump delivers at a flowing bottom-hole pressure
    pwf (psia) - the rate q at which pump rate (displacement x Ev x liquid fraction at the intake) equals q, with the
    intake from pwf up the casing at q and the discharge from the top node at q. Returns (q, state dict) or
    (nan, None)."""
    from scipy.optimize import brentq
    from . import vlp as V
    from .esp import sub_path, _rates
    rods = rod_string(inp.rod_no, inp.plunger)
    above = sub_path(path, path.md[0], inp.pump_md)
    below = sub_path(path, inp.pump_md, path.md[-1]) if path.md[-1] > inp.pump_md + 0.5 else None
    tvd = float(np.interp(inp.pump_md, path.md, path.tvd))
    secs = rods.sections(tvd)
    ap = np.pi / 4 * inp.plunger ** 2

    def at(q):
        case = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([gor])}
        if below is not None:
            Pb, Tb = V.traverse_up(below, fluid, np.array([pwf]), case, correlation, p1, p2, u, hc)
            pip, t_in = float(Pb[0, 0]), float(Tb[0, 0])
        else:
            pip, t_in = float(pwf), float(path.t_form[-1])
        if not np.isfinite(pip) or pip <= 0.0:
            return None
        ea_fn = (lambda ql_, qg_: gas_anchor_efficiency(ql_, qg_, inp.anchor_area, inp.anchor_type)) \
            if inp.anchor_method == "Calculated" else (lambda ql_, qg_: float(np.clip(inp.anchor_eff, 0.0, 1.0)))
        ea, liq_frac, bl, rs = _intake(fluid, max(pip, 14.7), t_in, case, ea_fn)
        case_a = {"q": np.array([q]), "wc": np.array([wc]), "gor": np.array([(1.0 - ea) * gor])}
        P, T = V.traverse(above, fluid, np.array([top_p]), case_a, correlation, p1, p2, u, hc, t_in)
        pdis = float(P[0, -1])
        fo = ap * (pdis - pip)
        sp = pump_stroke_api(inp.stroke, inp.n_spm, tvd, fo, secs)
        disp = 0.1166 * sp * inp.n_spm * inp.plunger ** 2
        return {"q": q, "pip": pip, "pdis": pdis, "sp": sp, "disp": disp, "ea": ea, "q_pump": disp * inp.ev * liq_frac / (bl if PUMP_FILL_BL else 1.0),
                "t_wh": float(T[0, 0])}

    q_hi = q_hi or 0.1166 * inp.stroke * 1.5 * inp.n_spm * inp.plunger ** 2
    f = lambda q: (lambda r: np.nan if r is None else r["q_pump"] - q)(at(q))
    lo, hi = 1e-3, q_hi
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo <= 0 or fhi >= 0:
        return np.nan, None
    q = brentq(f, lo, hi, xtol=1e-3)
    return q, at(q)
