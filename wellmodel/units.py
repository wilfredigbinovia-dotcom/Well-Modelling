"""Units of measure: quantities, the units each can be shown in, and conversions.

The calculation engine always works in oilfield units (psia, degF, ft, in, STB/d, MMscf/d, ...). The app converts
inputs and outputs at the screen edge only: every value is stored in the internal unit, shown in the unit the user
chose for its quantity on the Units tab.

A unit is (factor, offset) to a base unit: base = value * factor + offset.
Labels carry the internal unit in brackets, e.g. "Reservoir pressure (psia)"; `split_label` recognises it.
"""
from __future__ import annotations

import re

import numpy as np

PSI = 6894.757293168
ATM = 101325.0

# quantity -> {unit: (factor, offset)} ; first unit = default (the tool's internal / oilfield unit)
QUANTITIES = {
    "pressure": {"psia": (PSI, 0.0), "psig": (PSI, ATM), "bara": (1e5, 0.0), "barg": (1e5, ATM),
                 "kPa(a)": (1e3, 0.0), "kPa(g)": (1e3, ATM), "MPa(a)": (1e6, 0.0), "MPa(g)": (1e6, ATM),
                 "kg/cm2(a)": (98066.5, 0.0), "kg/cm2(g)": (98066.5, ATM), "atm": (ATM, 0.0)},
    "pressure_diff": {"psi": (PSI, 0.0), "bar": (1e5, 0.0), "kPa": (1e3, 0.0), "MPa": (1e6, 0.0),
                      "kg/cm2": (98066.5, 0.0), "atm": (ATM, 0.0)},
    "temperature": {"°F": (5 / 9, 459.67 * 5 / 9), "°C": (1.0, 273.15), "K": (1.0, 0.0), "°R": (5 / 9, 0.0)},
    "length": {"ft": (0.3048, 0.0), "m": (1.0, 0.0)},
    "diameter": {"in": (0.0254, 0.0), "mm": (1e-3, 0.0), "cm": (1e-2, 0.0), "ft": (0.3048, 0.0), "m": (1.0, 0.0)},
    "liquid_rate": {"STB/d": (0.158987295, 0.0), "Sm3/d": (1.0, 0.0), "MSTB/d": (158.987295, 0.0),
                    "m3/h": (24.0, 0.0)},
    "gas_rate": {"MMscf/d": (28316.846592, 0.0), "Mscf/d": (28.316846592, 0.0), "scf/d": (0.028316846592, 0.0),
                 "Sm3/d": (1.0, 0.0), "1000 Sm3/d": (1e3, 0.0), "MMSm3/d": (1e6, 0.0)},
    "gor": {"scf/STB": (0.028316846592 / 0.158987295, 0.0), "Sm3/Sm3": (1.0, 0.0), "Mscf/STB": (28.316846592 / 0.158987295, 0.0)},
    "cgr": {"STB/MMscf": (0.158987295 / 28316.846592, 0.0), "Sm3/MMSm3": (1e-6, 0.0), "Sm3/Sm3": (1.0, 0.0),
            "STB/Mscf": (0.158987295 / 28.316846592, 0.0)},
    "permeability": {"md": (1.0, 0.0), "D": (1000.0, 0.0), "μm2": (1 / 9.869233e-4, 0.0)},
    "viscosity": {"cp": (1.0, 0.0), "mPa·s": (1.0, 0.0), "Pa·s": (1000.0, 0.0)},
    "density": {"lb/ft3": (16.01846337, 0.0), "kg/m3": (1.0, 0.0), "g/cm3": (1000.0, 0.0)},
    "heat_transfer": {"BTU/h/ft2/F": (5.678263341, 0.0), "W/m2/K": (1.0, 0.0)},
    "heat_capacity": {"BTU/lb/F": (4186.8, 0.0), "kJ/kg/K": (1000.0, 0.0), "J/kg/K": (1.0, 0.0)},
    "pressure_gradient": {"psi/ft": (PSI / 0.3048, 0.0), "bar/m": (1e5, 0.0), "kPa/m": (1e3, 0.0),
                          "psi/100ft": (PSI / 30.48, 0.0)},
    "productivity_index": {"STB/d/psi": (0.158987295 / PSI, 0.0), "Sm3/d/bar": (1e-5, 0.0),
                           "m3/d/kPa": (1e-3, 0.0)},
    "area": {"acres": (4046.8564224, 0.0), "ha": (1e4, 0.0), "km2": (1e6, 0.0), "ft2": (0.09290304, 0.0),
             "m2": (1.0, 0.0)},
    "velocity": {"ft/s": (0.3048, 0.0), "m/s": (1.0, 0.0)},
    "compressibility": {"1/psi": (1 / PSI, 0.0), "1/bar": (1e-5, 0.0), "1/kPa": (1e-3, 0.0)},
    "volume": {"ft3": (0.028316846592, 0.0), "bbl": (0.158987295, 0.0), "m3": (1.0, 0.0)},
    "gas_volume": {"Mscf": (28.316846592, 0.0), "MMscf": (28316.846592, 0.0), "Sm3": (1.0, 0.0)},
}

QUANTITY_LABELS = {
    "pressure": "Pressure", "pressure_diff": "Pressure difference", "temperature": "Temperature",
    "length": "Length / depth", "diameter": "Diameter / small length", "liquid_rate": "Liquid rate",
    "gas_rate": "Gas rate", "gor": "Gas-oil ratio", "cgr": "Condensate / water-gas ratio",
    "permeability": "Permeability", "viscosity": "Viscosity", "density": "Density",
    "heat_transfer": "Heat transfer coefficient", "heat_capacity": "Heat capacity",
    "pressure_gradient": "Pressure gradient", "productivity_index": "Productivity index", "area": "Area",
    "velocity": "Velocity", "compressibility": "Compressibility", "volume": "Volume", "gas_volume": "Gas volume",
}

# internal unit spellings used in labels -> (quantity, canonical internal unit)
ALIASES = {
    "psia": ("pressure", "psia"), "psig": ("pressure", "psig"),
    "psi": ("pressure_diff", "psi"),
    "deg F": ("temperature", "°F"), "degF": ("temperature", "°F"), "°F": ("temperature", "°F"),
    "deg f": ("temperature", "°F"), "F": ("temperature", "°F"),
    "ft": ("length", "ft"), "feet": ("length", "ft"), "ft MD": ("length", "ft"), "ft TVD": ("length", "ft"),
    "in": ("diameter", "in"), "inches": ("diameter", "in"),
    "STB/d": ("liquid_rate", "STB/d"), "STB/day": ("liquid_rate", "STB/d"), "BFPD": ("liquid_rate", "STB/d"),
    "BPD": ("liquid_rate", "STB/d"), "B/D": ("liquid_rate", "STB/d"), "BLPD": ("liquid_rate", "STB/d"),
    "STB/d, liquid": ("liquid_rate", "STB/d"),
    "MMscf/d": ("gas_rate", "MMscf/d"), "MMscf/day": ("gas_rate", "MMscf/d"), "Mscf/d": ("gas_rate", "Mscf/d"),
    "Mscf/day": ("gas_rate", "Mscf/d"), "MSCF/D": ("gas_rate", "Mscf/d"), "MCFD": ("gas_rate", "Mscf/d"),
    "scf/d": ("gas_rate", "scf/d"),
    "scf/STB": ("gor", "scf/STB"),
    "STB/MMscf": ("cgr", "STB/MMscf"),
    "md": ("permeability", "md"), "mD": ("permeability", "md"),
    "cp": ("viscosity", "cp"),
    "lb/ft3": ("density", "lb/ft3"),
    "BTU/h/ft2/F": ("heat_transfer", "BTU/h/ft2/F"), "BTU/hr/ft2/F": ("heat_transfer", "BTU/h/ft2/F"),
    "BTU/lb/F": ("heat_capacity", "BTU/lb/F"),
    "psi/ft": ("pressure_gradient", "psi/ft"),
    "STB/d/psi": ("productivity_index", "STB/d/psi"), "STB/day/psi": ("productivity_index", "STB/d/psi"),
    "acres": ("area", "acres"),
    "ft/s": ("velocity", "ft/s"), "ft/sec": ("velocity", "ft/s"),
    "1/psi": ("compressibility", "1/psi"),
    "ft3": ("volume", "ft3"),
}
PRESETS = {
    "Oilfield (default)": {q: next(iter(u)) for q, u in QUANTITIES.items()},
    "Oilfield, gauge pressure": {**{q: next(iter(u)) for q, u in QUANTITIES.items()}, "pressure": "psig"},
    "Metric": {"pressure": "bara", "pressure_diff": "bar", "temperature": "°C", "length": "m", "diameter": "mm",
               "liquid_rate": "Sm3/d", "gas_rate": "1000 Sm3/d", "gor": "Sm3/Sm3", "cgr": "Sm3/MMSm3",
               "permeability": "md", "viscosity": "cp", "density": "kg/m3", "heat_transfer": "W/m2/K",
               "heat_capacity": "kJ/kg/K", "pressure_gradient": "bar/m", "productivity_index": "Sm3/d/bar",
               "area": "km2", "velocity": "m/s", "compressibility": "1/bar", "volume": "m3", "gas_volume": "Sm3"},
    "Metric, gauge pressure": None,     # filled below
    "SI (kPa)": {"pressure": "kPa(a)", "pressure_diff": "kPa", "temperature": "°C", "length": "m", "diameter": "mm",
                 "liquid_rate": "Sm3/d", "gas_rate": "Sm3/d", "gor": "Sm3/Sm3", "cgr": "Sm3/MMSm3",
                 "permeability": "μm2", "viscosity": "Pa·s", "density": "kg/m3", "heat_transfer": "W/m2/K",
                 "heat_capacity": "J/kg/K", "pressure_gradient": "kPa/m", "productivity_index": "m3/d/kPa",
                 "area": "m2", "velocity": "m/s", "compressibility": "1/kPa", "volume": "m3", "gas_volume": "Sm3"},
}
PRESETS["Metric, gauge pressure"] = {**PRESETS["Metric"], "pressure": "barg"}
DEFAULTS = PRESETS["Oilfield (default)"]


def default_unit(q):
    return next(iter(QUANTITIES[q]))


def convert(x, q, from_u, to_u):
    """Convert a value / array between units of one quantity (None and NaN pass through)."""
    if from_u == to_u or x is None:
        return x
    f1, o1 = QUANTITIES[q][from_u]
    f2, o2 = QUANTITIES[q][to_u]
    if isinstance(x, (list, tuple)):
        return type(x)(convert(v, q, from_u, to_u) for v in x)
    return (np.asarray(x, float) * f1 + o1 - o2) / f2 if isinstance(x, np.ndarray) else (float(x) * f1 + o1 - o2) / f2


def scale(q, from_u, to_u):
    """Multiplicative factor for differences (steps, spans) between two units."""
    return QUANTITIES[q][from_u][0] / QUANTITIES[q][to_u][0]


_BRACKET = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


def parse_unit(text):
    """'psia' / 'ft MD' / '(STB/day)' / 'STB/d, liquid' -> (quantity, internal unit, suffix) or None."""
    t = text.strip()
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()
    if t in ALIASES:
        q, u = ALIASES[t]
        suffix = t.split()[1] if t in ("ft MD", "ft TVD") else ""
        return q, u, suffix
    head, _, rest = t.partition(",")
    if rest and head.strip() in ALIASES:
        q, u = ALIASES[head.strip()]
        return q, u, "," + rest
    return None


def split_label(label):
    """'Reservoir pressure (psia)' -> ('Reservoir pressure', quantity, internal unit, extra); None if no unit.
    'Screen radius (ft, 0 = none)' -> ('Screen radius (0 = none)', 'length', 'ft', '')."""
    m = _BRACKET.match(label)
    if not m:
        return None
    base, inside = m.group(1), m.group(2)
    p = parse_unit(inside)
    if p is None:
        return None
    q, u, suffix = p
    if suffix.startswith(","):
        base = f"{base} ({suffix[1:].strip()})"
        suffix = ""
    return base, q, u, suffix


FIXED_UNITS = {"%", "percent", "1/ft", "fraction", "days", "sp. gravity", "API", "ppm", "deg", "hours",
               "$/STB", "$/Mscf", "Mscf/d/psi^2n"}


def fixed_unit(label):
    """'Water salinity (ppm)' -> ('Water salinity', 'ppm') for units that have no alternatives; else None."""
    m = _BRACKET.match(label)
    if not m or m.group(2).strip() not in FIXED_UNITS:
        return None
    return m.group(1), m.group(2).strip()


def unit_label(unit, suffix=""):
    return f"{unit} {suffix}".strip()


def relabel(name, units):
    """Column / axis name with the internal unit in brackets -> the same name in the chosen unit."""
    s = split_label(name)
    if s is None:
        return name, None
    base, q, u, suffix = s
    to = display_unit(q, u, units)
    return f"{base} ({unit_label(to, suffix)})", (q, u, to)


GAUGE_OF = {"psia": "psig", "bara": "barg", "kPa(a)": "kPa(g)", "MPa(a)": "MPa(g)", "kg/cm2(a)": "kg/cm2(g)"}


def display_unit(q, internal_unit, units):
    """The unit a value is shown in. Pressures kept in gauge by convention (psig: dome and test-rack pressures)
    stay gauge: they follow the gauge version of the chosen pressure unit."""
    to = (units or DEFAULTS).get(q, internal_unit)
    if q == "pressure" and internal_unit == "psig":
        to = GAUGE_OF.get(to, to)
    return to


# ------------------------------------------------------------------ free text
def _num_re():
    toks = sorted(((a, v) for a, v in ALIASES.items() if a not in ("in", "inches", "F", "deg f", "ft", "feet", "md",
                                                                    "mD", "cp")), key=lambda kv: -len(kv[0]))
    alt = "|".join(re.escape(a) for a, _ in toks)
    return re.compile(r"(?<![\w.,])(-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)(\s*)(" + alt + r")(?![\w/²³])"), dict(toks)


_TEXT_RE, _TEXT_MAP = _num_re()
_FT_RE = re.compile(r"(?<![\w.,])(-?\d[\d,]*\.?\d*)(\s*)(ft)(?![\w/²³])")


def _fmt_like(orig, x):
    s = orig.replace(",", "")
    if abs(x) >= 1e6 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.4g}"
    digits = s.lstrip("+-").replace(".", "").lstrip("0")
    sig = max(4, len(digits.rstrip("0")) if "." not in s else len(digits))   # keep the original precision
    mag = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
    dec = int(min(6, max(0, sig - 1 - mag)))
    return f"{x:,.{dec}f}" if "," in orig or abs(x) >= 1e4 else f"{x:.{dec}f}"


def convert_text(text, units):
    """Convert 'number unit' pairs in a sentence (e.g. '3,000 psia', '205 °F') to the chosen units."""
    if not units or not isinstance(text, str):
        return text

    def rep(m):
        q, u = _TEXT_MAP[m.group(3)]
        to = display_unit(q, u, units)
        if to == u:
            return m.group(0)
        try:
            x = float(m.group(1).replace(",", ""))
        except ValueError:
            return m.group(0)
        if not np.isfinite(x):
            return m.group(0)
        tok = m.group(3)
        suffix = " " + tok.split()[1] if tok in ("ft MD", "ft TVD") else ""
        return _fmt_like(m.group(1), convert(x, q, u, to)) + m.group(2) + to + suffix
    out = _TEXT_RE.sub(rep, text)

    def rep_ft(m):
        to = display_unit("length", "ft", units)
        if to == "ft":
            return m.group(0)
        x = float(m.group(1).replace(",", ""))
        return _fmt_like(m.group(1), convert(x, "length", "ft", to)) + m.group(2) + to
    return _FT_RE.sub(rep_ft, out)


def is_default(units):
    return not units or all(units.get(q, default_unit(q)) == default_unit(q) for q in QUANTITIES)
