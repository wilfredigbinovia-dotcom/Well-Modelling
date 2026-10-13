"""Model options (the "options summary"): what kind of well is being modelled.

The options decide which data the later setup steps ask for and which
calculations apply. `WellOptions.validate()` checks the combination,
`engine_status()` reports what the calculation engine can model today, and
`data_requirements()` lists the data each later step will need.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

FLUID_TYPES = {
    "oil": "Oil and water",
    "dry_wet_gas": "Dry and wet gas",
    "retrograde_condensate": "Retrograde condensate",
}
METHODS = {
    "black_oil": "Black oil",
    "eos": "Equation of state",
}
FLOW_TYPES = {
    "tubing": "Tubing flow",
    "annular": "Annular flow",
    "tubing_annular": "Tubing + annular flow",
}
WELL_TYPES = {
    "producer": "Producer",
    "gas_injector": "Gas injector",
    "water_injector": "Water injector",
}
COMPLETIONS = {
    "cased_hole": "Cased hole",
    "open_hole": "Open hole",
}
SAND_CONTROL = {
    "none": "None",
    "gravel_pack": "Gravel pack",
    "pre_packed_screen": "Pre-packed screen",
    "wire_wrapped_screen": "Wire-wrapped screen",
    "slotted_liner": "Slotted liner",
}

FIELDS = {  # field -> (display name, choices)
    "fluid_type": ("Fluid", FLUID_TYPES),
    "method": ("Method", METHODS),
    "flow_type": ("Flow type", FLOW_TYPES),
    "well_type": ("Well type", WELL_TYPES),
    "completion": ("Completion", COMPLETIONS),
    "sand_control": ("Sand control", SAND_CONTROL),
}

# What the calculation engine models today. Anything else is accepted as a
# setup choice but blocks calculations until the matching module exists.
MODELLED = {   # what the setup tabs built so far (PVT, IPR) can calculate
    "fluid_type": {"oil", "dry_wet_gas", "retrograde_condensate"},
    "method": {"black_oil", "eos"},
    "flow_type": {"tubing"},
    "well_type": {"producer"},
}
# Completion and sand control change near-wellbore pressure drop. Until they
# are modelled explicitly, that drop is absorbed in the PI fitted to tests.
LUMPED = {"completion", "sand_control"}

MODELLED_NOW, LUMPED_IN_PI, NOT_YET = "Modelled", "Absorbed in matched PI", "Not modelled yet"
PVT_ONLY = "PVT only"   # kept for compatibility; equation of state now feeds the IPR tab too


@dataclass(frozen=True)
class Issue:
    level: str        # "error" (invalid), "warning" (questionable), "info" (consequence)
    fields: tuple
    message: str


@dataclass
class WellOptions:
    fluid_type: str = "oil"
    method: str = "black_oil"
    flow_type: str = "tubing"
    well_type: str = "producer"
    completion: str = "cased_hole"
    sand_control: str = "none"

    def __post_init__(self):
        for f, (name, choices) in FIELDS.items():
            if getattr(self, f) not in choices:
                raise ValueError(f"{name}: '{getattr(self, f)}' is not one of {list(choices)}")

    # ------------------------------------------------------------ helpers
    def label(self, field: str) -> str:
        return FIELDS[field][1][getattr(self, field)]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_gas(self) -> bool:
        return self.fluid_type in ("dry_wet_gas", "retrograde_condensate")

    @property
    def is_injector(self) -> bool:
        return self.well_type != "producer"

    @property
    def uses_annulus(self) -> bool:
        return self.flow_type in ("annular", "tubing_annular")

    def describe(self) -> str:
        sc = "" if self.sand_control == "none" else f" with {self.label('sand_control').lower()}"
        return (f"{self.label('well_type')} - {self.label('fluid_type').lower()} "
                f"({self.label('method').lower()}), {self.label('flow_type').lower()}, "
                f"{self.label('completion').lower()} completion{sc}")

    # --------------------------------------------------------- validation
    def validate(self) -> list[Issue]:
        out = []
        add = lambda lvl, fields, msg: out.append(Issue(lvl, tuple(fields), msg))

        if self.well_type == "gas_injector" and self.fluid_type == "oil":
            add("error", ["well_type", "fluid_type"],
                "A gas injector needs a gas fluid description. Choose Dry and wet gas "
                "(or Retrograde condensate if rich gas is being re-injected).")
        if self.well_type == "gas_injector" and self.fluid_type == "retrograde_condensate":
            add("warning", ["well_type", "fluid_type"],
                "Injected gas is normally lean. Use Dry and wet gas unless rich gas is "
                "being cycled back into the reservoir.")
        if self.fluid_type == "retrograde_condensate" and self.method == "black_oil":
            add("warning", ["fluid_type", "method"],
                "Black oil can only approximate condensate dropout below the dew point. "
                "Equation of state is recommended for retrograde condensates.")
        if self.completion == "cased_hole" and self.sand_control == "slotted_liner":
            add("warning", ["completion", "sand_control"],
                "Slotted liners are normally run in open hole. Check the completion, or "
                "choose a screen or gravel pack for a cased-hole well.")
        if self.well_type == "water_injector":
            add("info", ["well_type", "fluid_type"],
                "Tubing flow and injectivity use water properties; the fluid type describes "
                "the reservoir fluid the water displaces.")
        if self.well_type == "water_injector" and self.method == "eos":
            add("info", ["well_type", "method"],
                "The equation of state applies to the reservoir fluid only; injected water "
                "uses water correlations.")
        if self.completion == "open_hole" and self.sand_control == "gravel_pack":
            add("info", ["completion", "sand_control"],
                "Open-hole gravel pack: the pressure drop is across the pack and screen; "
                "there are no perforation tunnels.")
        if self.completion == "cased_hole" and self.sand_control == "gravel_pack":
            add("info", ["completion", "sand_control"],
                "Cased-hole gravel pack: most of the extra pressure drop is in the "
                "gravel-filled perforation tunnels, so perforation details matter.")
        if self.uses_annulus and self.well_type == "producer" and self.fluid_type == "oil":
            add("info", ["flow_type"],
                "Oil flowing up the annulus has a large flow area: expect higher liquid "
                "holdup and an earlier loading point than tubing flow.")
        return out

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.validate() if i.level == "error"]

    # ----------------------------------------------------- engine support
    def engine_status(self) -> dict[str, str]:
        st = {}
        for f in FIELDS:
            if f in LUMPED:
                st[f] = MODELLED_NOW        # skin and sand-control models in the IPR tab
            else:
                st[f] = MODELLED_NOW if getattr(self, f) in MODELLED[f] else NOT_YET
        return st

    @property
    def pvt_only(self) -> list[str]:
        return [FIELDS[f][0] + ": " + self.label(f)
                for f, s in self.engine_status().items() if s == PVT_ONLY]

    @property
    def unsupported(self) -> list[str]:
        return [FIELDS[f][0] + ": " + self.label(f)
                for f, s in self.engine_status().items() if s == NOT_YET]

    @property
    def computable(self) -> bool:
        """True when the engine can calculate this configuration today."""
        return not self.errors and not self.unsupported and not self.pvt_only

    # ------------------------------------------------- data requirements
    def data_requirements(self) -> dict[str, list[str]]:
        pvt, eq, ipr = [], [], []

        # ---- PVT
        if self.well_type == "water_injector":
            pvt.append("Injected water salinity or specific gravity")
        elif self.well_type == "gas_injector":
            pvt.append("Injected gas gravity and impurities (H2S, CO2, N2)")
        if self.method == "eos" and self.well_type != "water_injector":
            pvt += ["PRP file exported from PROSPER / PVTP: composition, component properties, "
                    "binary interaction coefficients, separator train and reference temperature",
                    "The EOS should already be tuned to lab data (saturation pressure, CCE/CVD or "
                    "differential liberation) before export"]
        elif self.well_type == "producer":
            if self.fluid_type == "oil":
                pvt += ["Oil gravity (°API), solution GOR and separator gas gravity",
                        "Water salinity, reservoir temperature",
                        "Optional lab data to tune correlations: bubble point, Bo, oil viscosity"]
            elif self.fluid_type == "dry_wet_gas":
                pvt += ["Separator gas gravity and impurities (H2S, CO2, N2)",
                        "Condensate-gas ratio and condensate gravity (wet gas)",
                        "Water-gas ratio and water salinity"]
            else:
                pvt += ["Separator gas gravity and impurities (H2S, CO2, N2)",
                        "Condensate-gas ratio, condensate gravity and dew-point pressure",
                        "Water-gas ratio and water salinity"]

        # ---- Equipment
        eq += ["Deviation survey (MD vs TVD)",
               "Temperature profile (wellhead and reservoir, or geothermal gradient)"]
        if self.flow_type == "tubing":
            eq.append("Tubing string: inside diameter and roughness by depth")
        elif self.flow_type == "annular":
            eq.append("Annulus: casing inside diameter, tubing outside diameter, roughness")
        else:
            eq.append("Tubing inside and outside diameters, casing inside diameter, roughness "
                      "of both flow paths")
        if self.completion == "cased_hole":
            eq.append("Perforations: interval, shot density, perforation diameter, "
                      "penetration and phasing")
        else:
            eq.append("Open-hole diameter (bit size) over the completed interval")
        eq += {
            "none": [],
            "gravel_pack": ["Gravel pack: gravel permeability, "
                            + ("perforation tunnel length" if self.completion == "cased_hole"
                               else "pack thickness (hole to screen)") + ", screen OD"],
            "pre_packed_screen": ["Pre-packed screen: screen OD/ID and pre-pack permeability "
                                  "and thickness"],
            "wire_wrapped_screen": ["Wire-wrapped screen: screen OD/ID and slot (gauge) width"],
            "slotted_liner": ["Slotted liner: OD/ID, slot width and length, slots per foot"],
        }[self.sand_control]

        # ---- Inflow
        if self.well_type == "producer":
            ipr.append("Reservoir pressure and temperature")
            if self.fluid_type == "oil":
                ipr += ["Productivity index, or k, h, skin and drainage area",
                        "Water cut and producing GOR"]
            else:
                ipr += ["Back-pressure C and n, or k, h, skin and drainage area "
                        "(with non-Darcy coefficient)",
                        "Condensate-gas and water-gas ratios"]
        else:
            ipr += ["Reservoir pressure and temperature",
                    "Injectivity index, or k, h, skin and drainage area",
                    "Injection constraints: maximum surface pressure or rate, injection "
                    "temperature at surface"]
        return {"PVT": pvt, "Equipment": eq, "Inflow (IPR)": ipr}
