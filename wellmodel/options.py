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
    "cbm": "Coalbed methane (CBM)",
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

SEPARATORS = {
    "single_stage": "Single-stage separator",
}
WATER_VAPOUR = {
    "none": "No calculations",
    "calculate": "Calculate condensed water vapour",
}

EMULSIONS = {
    "none": "No",
    "emulsion": "Emulsion + pump viscosity correction",
}

VISCOSITY_MODELS = {
    "newtonian": "Newtonian fluid",
    "non_newtonian": "Non-Newtonian fluid",
}

ARTIFICIAL_LIFT = {
    "none": "None",
    "gas_lift": "Gas lift (continuous)",
    "ct_gas_lift": "Coiled tubing gas lift (gas down the coiled tubing, production up the annulus)",
    "esp": "Electrical submersible pump",
    "srp": "Sucker rod pump",
    "jet": "Jet pump (annular injection, tubing production)",
    "pcp_motor": "Progressive cavity pump (downhole motor drive)",
    "pcp_rod": "Progressive cavity pump (sucker rod drive)",
    "hsp": "Hydraulic submersible pump (turbine driven, power fluid down the annulus)",
    "diluent_gl": "Diluent injection - annular injection with gas lift",
    "diluent": "Diluent injection - annular injection",
    "mpump": "Multiphase pump (surface, OneSubsea H series)",
}

TEMP_MODELS = {
    "rough": "Rough approximation",
    "improved": "Improved approximation",
    "enthalpy": "Enthalpy balance",
}

STEAM = {
    "none": "No steam calculations",
    "allow": "Allow steam calculations",
}

FIELDS = {  # field -> (display name, choices)
    "fluid_type": ("Fluid", FLUID_TYPES),
    "method": ("Method", METHODS),
    "separator": ("Separator", SEPARATORS),
    "water_vapour": ("Water vapour", WATER_VAPOUR),
    "emulsions": ("Emulsions", EMULSIONS),
    "viscosity_model": ("Viscosity model", VISCOSITY_MODELS),
    "flow_type": ("Flow type", FLOW_TYPES),
    "well_type": ("Well type", WELL_TYPES),
    "completion": ("Completion", COMPLETIONS),
    "sand_control": ("Sand control", SAND_CONTROL),
    "artificial_lift": ("Artificial lift", ARTIFICIAL_LIFT),
    "temp_model": ("Temperature model", TEMP_MODELS),
    "steam": ("Steam calculation", STEAM),
}

# What the calculation engine models today. Anything else is accepted as a
# setup choice but blocks calculations until the matching module exists.
MODELLED = {   # what the tabs built so far (PVT, IPR, equipment, VLP) can calculate
    "fluid_type": {"oil", "dry_wet_gas", "retrograde_condensate", "cbm"},
    "method": {"black_oil", "eos"},
    "flow_type": {"tubing", "annular", "tubing_annular"},
    "well_type": {"producer", "water_injector", "gas_injector"},
    "artificial_lift": {"none", "gas_lift", "ct_gas_lift", "esp", "srp", "jet", "pcp_motor", "pcp_rod", "hsp",
                        "diluent_gl", "diluent", "mpump"},
    "separator": {"single_stage"},
    "water_vapour": {"none", "calculate"},
    "emulsions": {"none", "emulsion"},
    "viscosity_model": {"newtonian", "non_newtonian"},
    "temp_model": {"rough", "improved", "enthalpy"},
    "steam": {"none", "allow"},
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
    separator: str = "single_stage"
    water_vapour: str = "none"
    emulsions: str = "none"
    viscosity_model: str = "newtonian"
    flow_type: str = "tubing"
    well_type: str = "producer"
    completion: str = "cased_hole"
    sand_control: str = "none"
    artificial_lift: str = "none"
    temp_model: str = "rough"
    steam: str = "none"

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
        al = "" if self.artificial_lift == "none" else f", {self.label('artificial_lift').lower()}"
        return (f"{self.label('well_type')} - {self.label('fluid_type').lower()} "
                f"({self.label('method').lower()}), {self.label('flow_type').lower()}, "
                f"{self.label('completion').lower()} completion{sc}{al}")

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
            add("info", ["fluid_type", "method"],
                "Black-oil retrograde condensate: PROSPER's model as reconstructed from its PVT tables (Tutorial 38). "
                "As PROSPER advises, validate it against an equation of state rather than matching it to lab data.")
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
        if self.fluid_type == "cbm":
            if self.well_type != "producer":
                add("error", ["fluid_type", "well_type"], "Coalbed methane is modelled for producers (PROSPER's CBM "
                    "Producer well type).")
            if self.method != "black_oil":
                add("error", ["fluid_type", "method"], "Coalbed methane uses black-oil (water and gas) properties.")
            if self.flow_type != "tubing":
                add("error", ["fluid_type", "flow_type"], "A CBM producer produces water up the tubing: choose Tubing "
                    "flow (the gas goes up the annulus).")
            add("info", ["fluid_type"],
                "CBM producer (PROSPER Tutorial 45, de-watering): water with desorbed gas. PVT: gas gravity, CBM GLR "
                "and water salinity; IPR: CBM Producer (water PI scaled by the water relative permeability at the GLR). "
                "With an ESP and a gas separator, the annulus liquid level gives the casing head pressure, and the "
                "lift curves export to GAP / MBAL (.tpd).")
        if self.artificial_lift == "esp":
            if self.well_type != "producer" or self.fluid_type not in ("oil", "cbm"):
                add("error", ["artificial_lift", "fluid_type", "well_type"],
                    "An electrical submersible pump applies to oil (and water) producers.")
            if self.flow_type != "tubing":
                add("error", ["artificial_lift", "flow_type"],
                    "An ESP produces up the tubing: choose Tubing flow (gas from the separator goes up the "
                    "annulus).")
        if self.artificial_lift == "srp":
            if self.well_type != "producer" or self.fluid_type != "oil":
                add("error", ["artificial_lift", "fluid_type", "well_type"],
                    "A sucker rod pump applies to oil (and water) producers.")
            if self.flow_type != "tubing":
                add("error", ["artificial_lift", "flow_type"],
                    "A rod pump produces up the tubing: choose Tubing flow (separated gas goes up the annulus).")
        if self.artificial_lift == "jet":
            if self.well_type != "producer" or self.fluid_type != "oil":
                add("error", ["artificial_lift", "fluid_type", "well_type"],
                    "A jet pump applies to oil (and water) producers.")
            if self.flow_type != "tubing":
                add("error", ["artificial_lift", "flow_type"],
                    "The jet pump takes its power fluid down the annulus and produces up the tubing: choose Tubing "
                    "flow.")
        if self.artificial_lift == "hsp":
            if self.well_type != "producer" or self.fluid_type != "oil":
                add("error", ["artificial_lift", "fluid_type", "well_type"],
                    "A hydraulic submersible pump applies to oil (and water) producers.")
            if self.flow_type != "tubing":
                add("error", ["artificial_lift", "flow_type"],
                    "The HSP's turbine takes its power fluid down the annulus and exhausts into the tubing: choose "
                    "Tubing flow.")
        if self.artificial_lift in ("pcp_motor", "pcp_rod"):
            if self.well_type != "producer" or self.fluid_type != "oil":
                add("error", ["artificial_lift", "fluid_type", "well_type"],
                    "A progressive cavity pump applies to oil (and water) producers.")
            if self.flow_type != "tubing":
                add("error", ["artificial_lift", "flow_type"],
                    "A PCP produces up the tubing (with a sucker rod drive, the rod-tubing annulus): choose Tubing "
                    "flow (separated gas goes up the casing annulus).")
        if self.artificial_lift == "gas_lift":
            if self.well_type != "producer":
                add("error", ["artificial_lift", "well_type"], "Continuous gas lift is modelled for producers.")
            elif self.fluid_type != "oil":
                add("info", ["artificial_lift", "fluid_type"],
                    "Gas lifted gas well (PROSPER Tutorial 47): fixed depth of injection, injected gas rate, no "
                    "friction in the annulus; the lift gas joins the produced gas above the injection point.")
            if self.flow_type != "tubing":
                add("error", ["artificial_lift", "flow_type"],
                    "Gas lift injects down the annulus and produces up the tubing: choose Tubing flow.")
        if self.artificial_lift in ("diluent", "diluent_gl"):
            if self.well_type != "producer" or self.fluid_type != "oil":
                add("error", ["artificial_lift", "fluid_type", "well_type"],
                    "Diluent injection is modelled for oil (and water) producers.")
            if self.flow_type != "tubing":
                add("error", ["artificial_lift", "flow_type"],
                    "The diluent goes down the annulus and the well produces up the tubing: choose Tubing flow.")
        if self.artificial_lift == "ct_gas_lift":
            if self.well_type != "producer" or self.fluid_type != "oil":
                add("error", ["artificial_lift", "fluid_type", "well_type"],
                    "Coiled tubing gas lift is modelled for oil (and water) producers.")
            if self.flow_type != "tubing":
                add("error", ["artificial_lift", "flow_type"],
                    "Coiled tubing gas lift injects down a coiled tubing inside the tubing and produces up the coiled "
                    "tubing - tubing annulus: choose Tubing flow (the annulus is applied above the coiled tubing end).")
        if self.artificial_lift == "mpump" and self.well_type != "producer":
            add("error", ["artificial_lift", "well_type"],
                "The surface multiphase pump boosts a producing well's flowline: choose a producer.")
        if self.steam == "allow" and self.temp_model == "rough":
            add("error", ["steam", "temp_model"],
                "Steam calculations need a coupled temperature model: choose the Improved Approximation.")
        if self.steam == "allow" and self.well_type != "water_injector":
            add("error", ["steam", "well_type"],
                "Steam calculations are for a steam injector: choose Water injector (the water is injected as steam).")
        if self.steam == "allow" and self.well_type == "water_injector" and self.temp_model == "improved":
            add("info", ["steam", "well_type"],
                "Steam injector (PROSPER Tutorial 42): the injected water's enthalpy (or steam quality) at the wellhead "
                "is marched down with the heat loss; steam tables (IAPWS-95) for the quality and the steam, the PVT "
                "tab's brine for the water. Steam calculator in the PVT tab; gradients in the Calculation tab.")
        if self.temp_model == "enthalpy" and self.well_type != "producer":
            add("error", ["temp_model", "well_type"],
                "The Enthalpy Balance temperature model is built for producers; injectors use the Rough or the "
                "Improved Approximation.")
        if self.temp_model == "improved" and self.well_type != "producer":
            add("info", ["temp_model", "well_type"],
                "Injector with the Improved Approximation: an equation-of-state gas with the PRP file's CO2 correction "
                "gets the full energy balance (gravity, Joule-Thomson, heat loss; PROSPER Tutorial 39); other "
                "injected fluids use the U table with the average heat capacities.")
        if self.temp_model != "rough" and self.artificial_lift != "none":
            add("warning", ["temp_model", "artificial_lift"],
                "With artificial lift the coupled temperature model applies to the plain tubing traverses only; the "
                "lift calculations keep their own (Rough Approximation) temperatures.")
        if self.water_vapour == "calculate" and not (self.fluid_type == "dry_wet_gas" and self.method == "black_oil"):
            add("warning", ["water_vapour", "fluid_type", "method"],
                "Condensed water vapour is calculated for dry and wet gas with the black-oil method only; it is "
                "ignored for this fluid.")
        if self.emulsions == "emulsion":
            if self.fluid_type != "oil" or self.method != "black_oil" or self.well_type != "producer":
                add("warning", ["emulsions", "fluid_type", "method"],
                    "The emulsion viscosity applies to oil (and water) producers with black-oil PVT; it is ignored "
                    "for this fluid.")
            else:
                add("info", ["emulsions"],
                    "Emulsion viscosity (PROSPER Tutorial 46): match it to measured emulsion viscosities in the PVT "
                    "Input Data tab; it replaces the oil-water liquid viscosity where the emulsion occurs.")
        if self.viscosity_model == "non_newtonian":
            if self.fluid_type != "oil" or self.method != "black_oil":
                add("warning", ["viscosity_model", "fluid_type", "method"],
                    "The non-Newtonian (Herschel-Bulkley) oil viscosity applies to oil with black-oil PVT; it is "
                    "ignored for this fluid.")
            else:
                add("info", ["viscosity_model"],
                    "Non-Newtonian oil (PROSPER Tutorial 44): enter the rheological parameters (yield stress, "
                    "consistency index K, shear thinning index n against temperature and pressure) in the PVT Input "
                    "Data tab; the oil viscosity in the tubing is the apparent viscosity at the wall shear rate.")
            if self.emulsions == "emulsion":
                add("warning", ["viscosity_model", "emulsions"],
                    "Choose either emulsion or non-Newtonian viscosity: the non-Newtonian oil viscosity is used.")
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
                if self.emulsions == "emulsion" and self.method == "black_oil":
                    pvt.append("Emulsion viscosity against water cut (one pressure and temperature), the water "
                               "cuts bounding the maximum viscosity, and where the emulsion occurs")
                if self.viscosity_model == "non_newtonian" and self.method == "black_oil":
                    pvt.append("Rheological parameters (yield stress, consistency index K, shear thinning index n) "
                               "against temperature and pressure, and the maximum viscosity")
            elif self.fluid_type == "cbm":
                pvt += ["Gas gravity and impurities (H2S, CO2, N2)", "CBM gas-liquid ratio", "Water salinity"]
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
        if self.artificial_lift == "gas_lift":
            eq += ["Gas lift data: lift gas gravity and impurities, method (fixed depth, optimum depth, valve depths "
                   "or multipoint), casing head pressure, dP across the valve, valve depths (or transfer them from "
                   "the gas lift design); tubing OD and casing ID for annulus friction"]
        if self.artificial_lift == "ct_gas_lift":
            eq += ["Coiled tubing data: lift gas gravity and impurities, coiled tubing ID, wall thickness and "
                   "roughness, method (specified or optimum injection depth), coiled tubing depth, injection pressure "
                   "and dP across the valve (or transfer them from the coiled tubing gas lift design)"]
        if self.artificial_lift in ("diluent", "diluent_gl"):
            eq += ["Diluent injection data: injection depth, diluent rate (or % of the water) and salinity"
                   + ("; lift gas gravity and impurities, GLR injected and gas lift valve depth"
                      if self.artificial_lift == "diluent_gl" else "")]
        if self.temp_model == "improved":
            eq += ["Improved approximation: heat transfer coefficient by depth interval (MD, U), with the "
                   "geothermal gradient"]
        if self.temp_model == "enthalpy":
            eq += ["Enthalpy balance: drilling and casing record (hole sizes, casing OD and weight, shoes, tops of "
                   "cement, mud density), completion fluid and its density, packer depth, tubing OD, lithology "
                   "(formation type, porosity, in-situ fluid by depth), days since production started"]
        if self.artificial_lift == "mpump":
            eq += ["Surface equipment with a Pump row at the pump's place in the flowline (manifold first), the "
                   "power available to the pump, its wear factor and efficiency curve (OneSubsea H800/65 by default)"]
        if self.artificial_lift == "esp":
            eq += ["ESP: pump depth (the tubing ends at the pump; casing ID below it), pump, motor and cable "
                   "(or run the ESP design), number of stages, operating frequency, cable length, gas separator "
                   "efficiency; tubing OD and casing ID for the motor annulus"]

        # ---- Inflow
        if self.well_type == "producer":
            ipr.append("Reservoir pressure and temperature")
            if self.fluid_type == "oil":
                ipr += ["Productivity index, or k, h, skin and drainage area",
                        "Water cut and producing GOR"]
            elif self.fluid_type == "cbm":
                ipr += ["Water productivity index and CBM GLR",
                        "Water-gas relative permeabilities: residual saturations, end points, Corey exponents"]
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
