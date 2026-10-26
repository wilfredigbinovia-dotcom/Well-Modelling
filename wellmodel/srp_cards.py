"""Dynamometer card troubleshooting reference (PROSPER Examples Guide, Tutorial 36).

Idealised downhole (pump) card shapes for anchored and unanchored tubing, drawn as schematics: load against plunger
position, both normalised (0-1). The loop runs A (bottom of stroke, start of upstroke) -> B (fluid load picked up)
-> C (top of stroke) -> D (fluid load released) -> A. Fo is the differential (fluid) load on the plunger, MPT the
maximum plunger travel and EPT the effective plunger travel - the part of the stroke that moves fluid. The shapes are
qualitative: in practice several effects overlap and the diagnosis needs knowledge of the well.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STRETCH = 0.14        # unanchored tubing: travel lost while the fluid load moves between tubing and rods


def _smooth(a, b, n=24, shape="cos"):
    """Points from a to b (x, y) with an eased transition in y."""
    t = np.linspace(0.0, 1.0, n)
    if shape == "cos":
        s = 0.5 - 0.5 * np.cos(np.pi * t)
    elif shape == "out":                        # fast first, then levelling off (rounded corner at the end)
        s = np.sin(0.5 * np.pi * t)
    elif shape == "in":                         # slow first, then fast
        s = 1 - np.cos(0.5 * np.pi * t)
    else:
        s = t
    x = a[0] + (b[0] - a[0]) * t
    y = a[1] + (b[1] - a[1]) * s
    return np.c_[x, y]


def _poly(*segs):
    pts = np.vstack(segs)
    return np.vstack([pts, pts[:1]])


@dataclass
class Card:
    x: np.ndarray            # closed loop, normalised position
    y: np.ndarray            # normalised load (Fo = 1)
    labels: dict             # letter -> (x, y)
    ept: tuple               # (x start, x end) of the effective plunger travel
    fo: bool = True          # mark Fo


@dataclass
class CardShape:
    title: str
    description: str
    look_for: str
    anchored: Card
    unanchored: Card


def _card(pts, labels, ept, fo=True):
    return Card(pts[:, 0], pts[:, 1], labels, ept, fo)


def _normal(e):
    A, B, C, D = (0.0, 0.0), (e, 1.0), (1.0, 1.0), (1.0 - e, 0.0)
    return _card(_poly(_smooth(A, B, 2, "lin"), _smooth(B, C, 2, "lin"), _smooth(C, D, 2, "lin"),
                       _smooth(D, A, 2, "lin")), {"A": A, "B": B, "C": C, "D": D}, (e, 1.0 - e) if e else (0.0, 1.0))


def _travelling_valve(e):
    # load picked up slowly (fluid slips past the plunger) and lost before the top of the stroke
    A, B, C, D = (0.0, 0.0), (0.22 + e, 1.0), (0.78, 1.0), (1.0, 0.0)
    up = _smooth(A, B, 30, "out")
    top = _smooth(B, C, 2, "lin")
    down = _smooth(C, D, 30, "in")
    bot = _smooth(D, A, 2, "lin")
    pts = _poly(up, top, down, bot)
    return _card(pts, {"A": A, "B": B, "C": C, "D": D}, (B[0], C[0]))


def _standing_valve(e):
    # fluid load released slowly on the downstroke and picked up early, before the bottom of the stroke
    A, B, C, D = (0.24, 0.0), (0.0 + e, 1.0), (1.0, 1.0), (0.78 - e, 0.0)
    down = _smooth(C, D, 30, "out")
    bot = _smooth(D, A, 2, "lin")
    end = _smooth(A, B, 30, "in")
    top = _smooth(B, C, 2, "lin")
    pts = np.vstack([top, down, bot, end])
    pts = np.vstack([pts, pts[:1]])
    return _card(pts, {"A": A, "B": B, "C": C, "D": D}, (A[0], D[0]))


def _fluid_pound(e):
    # barrel partly filled: full load carried down until the plunger hits the liquid, then a sharp drop
    A, B, C, D = (0.0, 0.0), (e, 1.0), (1.0, 1.0), (0.48, 0.0)
    knee = (0.55, 0.97)
    pts = _poly(_smooth(A, B, 2, "lin"), _smooth(B, C, 2, "lin"), _smooth(C, knee, 6, "lin"),
                _smooth(knee, D, 8, "cos"), _smooth(D, A, 2, "lin"))
    return _card(pts, {"A": A, "B": B, "C": C, "D": D}, (e, D[0]))


def _gas_interference(e):
    # gas in the barrel compresses before the travelling valve opens: a gradual, curved unloading
    A, B, C, D = (0.0, 0.0), (e, 1.0), (1.0, 1.0), (0.32, 0.0)
    pts = _poly(_smooth(A, B, 2, "lin"), _smooth(B, C, 2, "lin"), _smooth(C, D, 40, "out"),
                _smooth(D, A, 2, "lin"))
    return _card(pts, {"A": A, "B": B, "C": C, "D": D}, (e, D[0]))


def _tapping(anchored):
    if anchored:
        # plunger hits the bottom of the pump: a load spike at the bottom of the stroke (A)
        A, B, C, D = (0.0, -0.45), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)
        pts = np.vstack([(0.03, 0.0), (0.0, -0.45), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.03, 0.0)])
        return _card(pts, {"A": A, "B": B, "C": C, "D": D}, (0.0, 1.0))
    # unanchored: tapping at the top of the stroke - a spike at C
    e = STRETCH
    A, B, C, D = (0.0, 0.0), (e, 1.0), (1.0, 1.45), (1.0 - e, 0.0)
    pts = np.vstack([A, B, (0.94, 1.0), C, (0.97, 1.0), D, A])
    return _card(pts, {"A": A, "B": B, "C": C, "D": D}, (e, 1.0 - e))


def reference_cards():
    """The six shapes of PROSPER's Tutorial 36 table, anchored and unanchored tubing."""
    e = STRETCH
    return [
        CardShape("Normal (full pump)",
                  "The idealised card of a pump working properly, its barrel full of liquid with no free gas.",
                  "Square corners (anchored) or a parallelogram whose sloping sides are the tubing stretch "
                  "(unanchored); EPT close to MPT.", _normal(0.0), _normal(e)),
        CardShape("Leaking travelling valve or excessive plunger slippage",
                  "Fluid slips back past the plunger or the travelling valve, so the load is picked up late on the "
                  "upstroke and starts to fall off before the top.",
                  "Rounded upper corners (B and C); the top of the card is short.", _travelling_valve(0.0),
                  _travelling_valve(0.06)),
        CardShape("Leaking standing valve",
                  "Fluid leaks back through the standing valve on the downstroke, so the fluid load is released "
                  "slowly and picked up again before the bottom of the stroke.",
                  "Rounded lower corners (A and D); the bottom of the card is short.", _standing_valve(0.0),
                  _standing_valve(0.06)),
        CardShape("Severe fluid pound (well pumped off)",
                  "The barrel is only partly filled, so the plunger carries the full load down until it strikes the "
                  "liquid, then unloads sharply. The pump displacement is larger than the inflow.",
                  "Full load held into the downstroke, then a steep drop at D; EPT well short of MPT.",
                  _fluid_pound(0.0), _fluid_pound(e)),
        CardShape("Gas interference",
                  "Free gas in the barrel has to be compressed before the travelling valve opens, so the load comes "
                  "off gradually and effective travel is lost.",
                  "A curved, gradual unloading from C to D (sharp in fluid pound).",
                  _gas_interference(0.0), _gas_interference(e)),
        CardShape("Pump tapping",
                  "The plunger strikes the pump: at the bottom of the stroke with anchored tubing, at the top of "
                  "the stroke with unanchored tubing (re-space the pump).",
                  "A load spike at A (anchored) or at C (unanchored).", _tapping(True), _tapping(False)),
    ]
