# wellmodel

An open, scriptable oil-well modelling toolkit in Python, built around five things commercial nodal tools do poorly or not at all:

| Feature | What it does |
|---|---|
| **Test data QC** | Flags each well test pass / warn / fail with a plain-language reason: range checks, GOR vs Rsb, a physical VLP envelope, implied-PI outliers against a robust time trend, test stability, water-cut jumps. Weights feed straight into the match. |
| **Automated multi-test matching** | Fits gravity multiplier, friction multiplier and PI to *all* good tests at once (robust least squares on gauge-pressure and rate residuals), with standard errors, a parameter correlation matrix and warnings for out-of-range or unidentifiable multipliers. |
| **Correlation selection** | Ranks VLP correlations by leave-one-out cross-validation - each test predicted by a model that never saw it - and ranks any correlation that needs out-of-range multipliers behind those that don't. Writes a rationale. |
| **Probabilistic runs** | Latin-hypercube Monte Carlo over any inputs, with match covariance propagating model-parameter uncertainty jointly. P90/P50/P10, probability of natural flow, Spearman sensitivities and a tornado. |
| **Transient behaviour** | Pressure-transient solution with wellbore storage, skin and a closed boundary (Stehfest); Bourdet derivative; build-up fitting for k, skin, C and an independent PI. Depletion forecast with material balance + aquifer, coupled to nodal at every step, reporting when the well drops below Pb, goes unstable and stops flowing. |

Everything is vectorised: thousands of nodal solutions (Monte Carlo samples, test points, rate grids) run through the tubing in one pass.

## Physics

- **PVT (black oil):** Standing Pb/Rs/Bo (optionally calibrated to a measured Pb), Vasquez-Beggs undersaturated Bo and viscosity, Beggs-Robinson viscosity, Dranchuk-Abou-Kassem Z, Lee-Gonzalez-Eakin gas viscosity, McCain water, Baker-Swerdloff surface tension.
- **IPR:** composite straight-line PI / Vogel (liquid basis); Darcy PSS PI helper.
- **VLP:** Beggs & Brill (Payne-corrected), modified Hagedorn & Brown (Griffith bubble flow, no-slip floor), Gray (gas wells), Mukherjee & Brill (uphill), Orkiszewski (Griffith bubble, Griffith-Wallis slug, Duns & Ros mist/transition; no-slip density floor), Aziz-Govier-Fogarasi (with Duns & Ros mist), Beggs & Brill no-slip, no-slip homogeneous (lower bound). Duns & Ros (original: F1-F7, L1/L2 and f2 charts digitised from Brill & Beggs 1991, Figs 3-11 to 3-14; Duns & Ros mist with the d - eps correction and the modified gas density in transition) and Fancher & Brown (1963: no-slip density with the energy-loss factor digitised from Figs 7-9 for the three GLR ranges). Fancher & Brown and no-slip act as lower bounds and Duns & Ros as the slug-flow upper bound in the comparison's data checks. Duns & Ros (modified): Duns & Ros holdup on the Gould, Tek & Katz (1974) flow-regime map - Wallis bubble boundary, holdup interpolated across the bubble/slug transition band, plug region (1-in. and 3.15-in. curves, interpolated by tubing size) treated as slug - with Kleyweg friction in bubble and slug flow: a single-phase Moody friction factor on the average (mixture) velocity with holdup-weighted density and viscosity. The froth region has no drawn boundary in the paper and is treated as slug flow; mist flow keeps the Duns & Ros gas-film friction. The exact mixture properties in Kleyweg's method are an interpretation, so results may differ somewhat from commercial implementations. Ansari et al. (1994) mechanistic model: Taitel-Dukler bubble/slug and Barnea dispersed-bubble (void < 0.76) and annular-stability transitions; drift-flux bubble flow, no-slip dispersed bubble, Fernandes/Sylvester slug unit (Taylor bubble, Brotz falling film, slug-body balance) and Alves et al. annular film/core model with Wallis entrainment, its film thickness solved from the combined momentum balance. Not available: Petroleum Experts 1-5 (proprietary, not planned). Digitised chart values carry a reading uncertainty of a few per cent. Orkiszewski and Aziz switch regimes abruptly, as published; Aziz in particular can misbehave at high liquid rates where its map gives mist flow. Deviation survey, tapered strings, linear temperature profile. Heun-integrated top-down traverse.
- **Nodal:** stable (right-hand) IPR/VLP intersection with stability flag at the VLP minimum.

Oilfield units throughout: psia, °F, STB/d, scf/STB, ft, in, cp.

## Quick start

```bash
pip install -e ".[app]"
streamlit run app.py            # point-and-click app in your browser
pytest -q                       # 105 tests
jupyter notebook notebooks/demo.ipynb
```

### The Streamlit app (`app.py`)

The app is being rebuilt tab by tab. It currently has six tabs:

1. **Options summary** - fluid type, method, flow type, well type, completion and sand control. Checks the combination, shows what the calculation engine can model today, and lists the data the later setup tabs will need.
2. **PVT Input Data** - shows black-oil or compositional input depending on the Method chosen in tab 1. For dry/wet gas with black oil it takes gas gravity, impurities and temperature.

   **Equation of state:** import a Petroleum Experts PRP export (.prp). The tool reads the EOS (Peng-Robinson or SRK), components, binary interaction coefficients, volume shifts, separator train and reference temperature, then calculates the saturation pressure (bubble or dew), separator-train GOR/CGR, stock-tank API and gas gravity, Bo or Bg, phase properties versus pressure (LBC viscosity, parachor IFT, liquid dropout) and the phase envelope. It warns when the fluid's behaviour contradicts the fluid type chosen in tab 1.

   **Black oil:** solution GOR, oil and gas gravity, water salinity and H2S/CO2/N2, plus a lab-data table (pressure, GOR, oil FVF, oil viscosity) with temperature and bubble point. **Match** fits Parameter 1 (multiplier) and Parameter 2 (shift) for every correlation, reports the standard deviation, and suggests the Pb/Rs/Bo family and viscosity correlation needing the smallest correction. Shows match plots and a calculated PVT table.

   - Bubble point / GOR / FVF: Glaso, Standing, Lasater, Vasquez-Beggs, Petrosky et al, Al-Marhoun.
   - Viscosity: Beal et al, Beggs et al, Petrosky et al, Egbogah et al.
   - De Ghetto et al, Bergman-Sutton and De Ghetto Mod are listed but not calculated: their coefficients could not be verified against a published source (De Ghetto Mod is PROSPER's own unpublished variant).
3. **IPR Data** - producer inflow models chosen by the fluid type in tab 1, using viscosity, FVF, Z and bubble point from tab 2 (black oil or equation of state) or values entered by hand.

   - Oil: P.I. Entry, Vogel, Composite, Darcy, Fetkovich, Multi-rate Fetkovich, Jones, Multi-rate Jones, Transient, External entry.
   - Gas and condensate: Jones, Forchheimer, Back pressure, C and n, Multi-rate C and n, Multi-rate Jones, Forchheimer with pseudo-pressure, Multi-rate Forchheimer with pseudo-pressure, External entry.
   - Advanced (oil and gas): Hydraulically fractured well (Gringarten infinite-conductivity with Cinco-Ley finite-conductivity correction, transient to PSS), Horizontal well - no-flow boundaries (Babu & Odeh, Joshi), Horizontal well - transverse vertical fractures (Mukherjee-Economides), Horizontal well - friction dP (segmented toe-to-heel march), Dual porosity (Warren-Root), Multi-layer (with crossflow), Multi-layer with wellbore dP. Gas also has Modified Isochronal (correction factors and stabilised C).
   - **Multiphase Pseudo Pressure** (oil and gas): total mass-mobility pseudo-pressure from the PROSPER definition, closed with Corey relative permeabilities and the Fevang & Whitson steady-state assumption. Black oil uses the producing GOR; equation of state flashes the PRP fluid at each pressure, so the condensate bank below the dew point (or free gas below the bubble point) shows as flowing saturation, kro/krg and a productivity loss against single-phase flow.
   - Properties (viscosity, FVF, Z, bubble point, compressibility) come from tab 2 and are not repeated here. Models that use a drainage area take the Dietz shape factor as an input (typical values in the field's help).
   - Multi-rate models calculate the reservoir pressure from the tests when the entered pressure is at or below the highest test FBHP.
   - **Skin:** entered by hand, Karakas & Tariq (perforation, damaged and crushed zones) or MacLeod for cased hole; Hawkins damaged zone for open hole. Optional Cinco-Ley deviation and Papatzacos partial penetration. Shows the skin breakdown and the dP across the skin.
   - **Sand control** (chosen in tab 1): cased gravel pack (linear flow through gravel-filled perforation tunnels, Darcy + non-Darcy), open-hole gravel pack and pre-packed screen (radial Darcy + Forchheimer), wire-wrapped screen and slotted liner (orifice flow through slots). Applied in series with the reservoir IPR; the plot shows the curve with and without sand control.
   - Not available: Kuchuk & Goode (Babu & Odeh / Joshi are used instead), SkinAide and thermally induced fractures (injectors).

4. **Equipment Data** - five sub-tabs, with a status table showing which sections are complete:

   - **Deviation survey:** up to 20 MD/TVD points, with calculated angle and cumulative displacement and a well-path plot. **Filter a longer survey** accepts any number of points (paste, or upload CSV/TXT/Excel) and keeps the 20 that best reproduce the TVD profile, reporting the largest and RMS TVD error.
   - **Surface equipment:** manifold, line pipe, choke and fittings (equivalent length from L/D values), surrounding temperature and overall heat transfer coefficient. As in PROSPER, each type opens only its own cells (manifold: TVD; line pipe: length, TVD, ID, roughness, rate multiplier; choke: ID, rate multiplier; fitting: fitting type, ID, rate multiplier), with roughness 0.0006 in and rate multiplier 1 as editable defaults.
   - **Downhole equipment:** Xmas tree, tubing, SSSV, restriction, gas lift mandrel, SSD and casing, PROSPER-style. As in PROSPER, each type opens only its own cells (Xmas tree: MD; tubing: MD, ID, roughness, rate multiplier; SSSV: ID; restriction: ID, rate multiplier; casing: MD, ID, roughness, rate multiplier; annular flow also opens tubing OD/outside roughness and casing ID/roughness on tubing rows) - anything typed in a locked cell is cleared, and roughness (0.0006 in) and rate multiplier (1) fill in as editable defaults. A shaded guide shows the rule per type. Checks depths, diameters and roughness, requires tubing OD and casing ID when annular flow is selected, and checks the string against the survey. Shows flow sections, components and a schematic.
   - **Geothermal gradient:** formation temperature against MD (TVD from the survey), gradient per interval and the overall heat transfer coefficient.
   - **Average heat capacities:** oil 0.53, gas 0.51, water 1.0 BTU/lb/F by default.
   - **Well schematic:** appears once every section is complete - an illustrated, not-to-scale drawing of the Xmas tree, flowline and choke, casing, tubing and its components (SSSV with control line, gas lift mandrels, SSDs, restrictions), tubing end and reservoir, showing perforations or open hole and the sand control from tab 1. Comes with a string summary (MD, ID, TVD, formation temperature) and an SVG download.

5. **VLP** - well test table laid out like PROSPER's (oil: THP, THT, water cut, liquid rate, gauge depth and pressure, reservoir pressure, GOR, GOR free; gas: THP, THT, WGR, separator GOR, gas rate, gauge, reservoir pressure), then five actions:

   - **Estimate U Value** - the overall heat transfer coefficient that reproduces each measured tubing head temperature; **Accept** writes the median to the Geothermal gradient section.
   - **Correlation Comparison** - Duns & Ros (original and modified), Ansari, Hagedorn & Brown (mod.), Beggs & Brill, Beggs & Brill (no-slip), Gray, Mukherjee & Brill, Orkiszewski, Aziz-Govier-Fogarasi, Fancher & Brown and the No-slip lower bound, unmatched, from THP to the gauge; ranks them by error and plots pressure and temperature traverses against the measured gauge point.
   - **Match VLP** - regression of Parameter 1 (gravity) and Parameter 2 (friction) for each correlation over all tests, with standard deviation, before/after error and flags when Parameter 1 is outside 0.9-1.1. Suggests the correlation needing the least correction; you choose the one to carry forward.
   - **VLP/IPR** - matched lift curve at each test's THP and ratios against the IPR from tab 3 (rebuilt at the test reservoir pressure where the model allows), the calculated operating rate against the measured rate, and the plot.
   - **Adjust IPR** - for a chosen test, solves for the PI (P.I. Entry), the hand-entered skin, or the reservoir pressure that puts the IPR through the test point; **Accept** writes it back to the IPR tab.
   - **Match Surface** - for tests with the optional **Manifold Pressure** (and **Choke Size**, in 64ths, when the bean changed between tests) columns filled: from each manifold pressure the calculation works back through the flowline, fittings and choke to the wellhead, starting at the measured tubing head temperature, and regresses the Sachdeva discharge coefficient and a flowline friction multiplier so the calculated wellhead pressure matches the measured THP. Choose the parameters and the flowline correlation; a parameter the data cannot see is held (friction when the choke is critical in every test, Cd without a choke) and values at their bounds or Cd outside 0.5-1.0 are flagged. Shows the fitted values, RMS error before and after, a test-by-test table with choke regime and losses, and a measured-v-calculated plot. **Accept** sends Cd, the friction multiplier and the correlation to the Calculation tab ("Use matched surface parameters"). The example tests carry manifold pressures generated with Cd 0.85 and friction x 1.2.

   The tubing model runs from the Xmas tree to the deepest equipment (taken as the IPR depth) on the tab 4 equipment: survey inclination, tubing / annulus / tubing + annulus geometry, SSSV and restriction pressure drops, and a heat-loss temperature profile from the geothermal gradient, U value and heat capacities. Fluids come from tab 2: matched black oil, gas correlations with condensate carried as wellstream gas, or the PRP equation of state flashed on a pressure-temperature grid. **Load example tests** generates synthetic tests from the current equipment, PVT and IPR (known answer: P1 1.04, P2 1.12).

6. **Calculation - System analysis (IPR + VLP)** - PROSPER-style nodal analysis at the bottom node. Inputs: top node - the **manifold** (through the surface equipment) or the wellhead - and its pressure, water cut and total GOR (oil) or WGR and separator GOR (gas) - copied from any VLP test with one click - the vertical lift correlation with Parameter 1/2 from Match VLP (or entered by hand), rate method (automatic geometric, automatic linear or user-selected rates), solution node - **bottom node** (default: IPR against VLP at the bottom of the downhole equipment, the solution pressure is the flowing bottom-hole pressure) or **top node** (the inflow - IPR, tubing traversed upwards and, with a manifold top node, the surface equipment downstream - against the top-node pressure; a critical choke shows as a vertical drop at the rate it sets; the solution rate is the same at either node), left-hand intersection (allow/disallow) and stability flags. With the manifold as top node the calculation works back from the manifold pressure through the flowline, fittings (equivalent length) and choke to the wellhead, then down the tubing: choose the surface (flowline) correlation (Beggs & Brill default, Mukherjee & Brill, Beggs & Brill no-slip, Duns & Ros, no-slip), the choke method - Sachdeva et al. (1986, critical and subcritical two-phase flow, discharge coefficient 0.75 by default) or the Gilbert, Ros, Baxendell and Achong critical-flow correlations for oil wells (subcritical points fall back to Sachdeva) - and the fluid cools along each pipe towards the surrounding temperature with the surface U value. Elevation changes come from the TVD column of the surface table and the rate multiplier scales the rate through each item. Results: solution rates (liquid, oil, water, gas or gas, condensate, water), bottom node pressure, wellhead temperature, the IPR/VLP plot, the curve table, and a solution panel with dP friction and gravity in the production string, restriction pressure drops and regime, transit time, maximum/minimum no-slip and erosional velocities (API RP 14E, C = 100), pressure range and stability; with surface equipment also the wellhead pressure, manifold temperature, dP choke, dP flowline and fittings, the choke regime (critical/subcritical) and an item-by-item surface table.

   **Sensitivity** - pick up to three variables (top node pressure, water cut / WGR, GOR / separator GOR, reservoir pressure, PI, skin, tubing ID, U value, choke size in 64ths when the manifold is the top node - only those the current IPR supports), enter up to 10 values each or generate them with linear or geometric spacing; every combination is calculated (limit 500). Shows a results table (CSV download), IPR/VLP curves for the cases, a solution-rate sensitivity plot and full details for any chosen case. Cases that do not flow or only intersect left of the VLP minimum are flagged.

   **Gradient traverse** - the same inputs (top node, pressures, ratios, correlation with its matched parameters, surface equipment) at up to six rates: a summary per rate (manifold, wellhead and bottom-node pressure, wellhead temperature, gravity, friction and choke losses, velocity/erosional ratio), plots of pressure, temperature, liquid holdup (with no-slip holdup) and mixture v erosional velocity against TVD, the surface equipment item by item, and the full profile at every node (superficial velocities, in-situ density, elevation and friction gradients) with a CSV of all rates. No IPR is needed.

   **Lift curves (VFP tables)** - bottom-node pressure for every combination of rate (generated, linear or geometric), top-node pressure, water cut / WGR and GOR / CGR, in one vectorised calculation (up to 20,000 points; ~3,000 in a few seconds). With the manifold as top node each point is worked back through the surface equipment first. Plots the curves for any water/gas fraction, flags points where BHP falls as THP rises or exceeds 30,000 psia, and exports an **Eclipse VFPPROD** include file (FIELD units: FLO = LIQ in STB/d with WCT and GOR in Mscf/STB for oil; FLO = GAS in Mscf/d with WGR and OGR in STB/Mscf for gas; datum = bottom-node TVD; ALQ 0) and a flat CSV. No IPR is needed.

**Project files (sidebar).** *Save project* downloads one `.wellmodel.json` file with every input on every tab, all tables (lab PVT, IPR test and layer tables, survey, surface and downhole equipment, geothermal gradient, well tests), the PRP file, the accepted U value and IPR adjustments, and the VLP and surface matches. *Open project* replaces the current model with a saved one (the PVT match is repeated automatically) and clears itself, so the same file can be opened again; *New project* clears everything. The save button shows the file name it will write - press Enter after typing the project name. On opening, the project takes the name of the file chosen. The file is plain JSON - opening it never runs code - and calculation results are not stored: press Calculate after opening.

The full workflow (QC, correlation selection, matching, nodal, probabilistic and transient) remains available from Python and in `notebooks/demo.ipynb` while the app's tabs are rebuilt.

```python
import wellmodel as wm

pvt = wm.BlackOilPVT(api=32, gas_sg=0.75, rsb=600, t_res=210)
wb  = wm.Wellbore(survey_md=[0, 2000, 10500], survey_tvd=[0, 2000, 9300],
                  tubing_id=3.958, perf_md=10400, gauge_md=10000, wht=125, bht=210)
base = wm.WellModel(pvt, wb)

qc    = wm.qc_tests(base, tests)                       # tests: pandas DataFrame
sel   = wm.compare_correlations(base, tests, weights=qc.weights)
match = sel.matches[sel.recommended]
model = match.model(base)

op = model.operating_point(pr=3500, whp=250, wc=0.3, gor=600)
mc = wm.monte_carlo(model, {"reservoir_pressure": wm.Triangular(3400, 3500, 3600),
                            "whp": 250, "water_cut": wm.Uniform(0.25, 0.35), "gor": 600},
                    match=match)
```

### Test table columns
`liquid_rate` (STB/d), `water_cut` (fraction), `gor` (scf/STB), `whp` (psia), `gauge_pressure` (psia), `reservoir_pressure` (psia). Optional: `test_id`, `date`, `duration_hr`, `rate_cv`, `weight`.

## Package layout

```
wellmodel/
  pvt.py            black-oil correlations
  wellbore.py       survey, tubing, temperature, grid
  correlations.py   multiphase gradient correlations
  ipr.py            PI / Vogel inflow
  model.py          WellModel: traverse + nodal
  qc.py             test QC
  matching.py       multi-test regression
  selection.py      LOOCV correlation ranking
  probabilistic.py  Monte Carlo / LHS, tornado
  transient.py      pressure transient, build-up fit, forecast
  options.py        options summary: choices, validation, engine status
  blackoil.py       black-oil correlation families, lab-data matching, suggestion
  eos.py            PRP import, PR/SRK equation of state, flash, envelope, separators
  inflow.py         IPR models (oil, gas/condensate), multi-rate fitting, pseudo-pressure
  inflow_adv.py     fractured, horizontal, dual-porosity, multilayer, friction, modified isochronal
  liftcurves.py     lift-curve (VFP) tables and the Eclipse VFPPROD writer
  project.py        project files: JSON save/open of inputs, tables and accepted matches
  surface.py        surface equipment: flowline/fittings march, Sachdeva and Gilbert-type chokes, line cooling
  system.py         system analysis: multi-case VLP/IPR solve, sensitivity combinations, solution details
  vlp.py            VLP engine: flow path, fluids (black oil, wet gas, EOS grid), heat loss, U estimate, comparison, P1/P2 match, VLP/IPR
  equipment.py      deviation survey and filter, surface/downhole equipment, geothermal gradient
  schematic.py      illustrated well schematic (SVG)
  mpp.py            multiphase pseudo-pressure (Corey, black oil / EOS integrands)
  completion.py     skin models (Karakas & Tariq, MacLeod, Hawkins, deviation, partial penetration), sand control
  synthetic.py      known-answer test data
  plotting.py       figures
app.py                Streamlit app
notebooks/demo.ipynb  end-to-end walkthrough
tests/                pytest suite
```

## Validation so far

**Against PROSPER - PVT (real oil: API 30.03, GOR 345 scf/STB, gas gravity 0.595, CO2 0.28 %, N2 0.1 %; one lab point at the 2552 psia bubble point, 153 °F).** Compared on PROSPER's match parameters and on its calculated PVT tables, matched and unmatched, 2045-3060 psia:

- Unmatched bubble point: Glaso, Standing, Petrosky and Al-Marhoun within 0.02 % once the Glaso N2/CO2/H2S correction is off (PROSPER does not apply it - now the default); Vazquez-Beggs and De Ghetto within 0.06 %.
- Solution GOR and oil FVF, unmatched and matched: within 0.25 % for Glaso, Standing, Vazquez-Beggs, Petrosky, Al-Marhoun and De Ghetto (Standing FVF uses its original 1947 form, as PROSPER). Exception: Al-Marhoun matched FVF 0.7 % - PROSPER's stock-tank anchor value differs.
- Oil viscosity: Beal, Beggs and Petrosky within 0.07 %; De Ghetto (heavy oil: modified Egbogah-Jack A-7, modified Kartoatmodjo A-9/A-11, SPE 30316) within 0.1 %; PROSPER's "Egbogah et al (Heavy oil)" - De Ghetto's modified Egbogah-Jack dead oil (A-6) with Petrosky live and undersaturated viscosity - within 0.06 %. Match parameters agree to 4-5 digits. PROSPER's PVT table pairs each viscosity column with the bubble-point correlation in the same position (Beal-Glaso, Beggs-Standing, Petrosky-Lasater, Egbogah-Vazquez-Beggs, Bergman-Sutton-Petrosky, De Ghetto-Al-Marhoun), which must be allowed for when comparing.
- De Ghetto bubble point and GOR are De Ghetto's heavy-oil modified Vazquez-Beggs (gas gravity referred to 100 psig) and its FVF is Vazquez-Beggs, as in PROSPER.
- Bergman-Sutton is implemented as published: dead oil SPE 110194 (modified Twu with a Watson K of 12.1 - not shown as an input), saturated SPE 110195 (eqs 4-6 as corrected in 2009), undersaturated SPE 103144. With K = 12.1 it matches PROSPER above the bubble point within 0.2 % and the match parameters within 0.06 %; below the bubble point PROSPER's curve is steeper and the tool is up to 7 % lower at 2045 psia (PROSPER's saturated form evidently differs from the published equations; PROSPER does not show the Watson K it uses).
- IPR (same oil: reservoir pressure 3791.3 psia, 153 °F, water cut 14.3 %, bubble point 2552 psia): P.I. Entry with PI 100 STB/d/psi gives an AOF of 265,708 STB/d against PROSPER's 267,193 (-0.56 %); Vogel from the test point (1773.63 STB/d at 3773.6 psia) gives 266,253 against 267,551 (-0.49 %).
- Not reproduced: Lasater (PROSPER's chart fit differs in shape - its raw bubble point is 1839 psia against 2440 from the published fits) and De Ghetto Mod (PROSPER's own, unpublished; unavailable in the tool).


On synthetic data with a known answer (truth: Hagedorn & Brown, gravity 1.04, friction 1.15, PI 2.61):

- QC caught all four planted problems (gauge +200 psi, rate +30 %, GOR −40 %, 3-hour test) and passed every clean test.
- Matching recovered gravity 1.045, friction 1.10 ± 0.09, PI 2.60.
- Selection picked Hagedorn & Brown. Beggs & Brill fit almost as well statistically but only with friction ≈ 0.70, outside the acceptable range.
- Build-up fit recovered k = 50.0 md, skin 2.00, C 0.030 bbl/psi; build-up PI agreed with production-test PI.
- Pressure-transient solution matches the infinite-acting log approximation, the PSS slope and the wellbore-storage unit slope.

## Limitations (screening level)

- Multilayer-with-dP and horizontal friction models treat wellbore flow as single phase. Fractured wells assume infinite-conductivity geometry with a Cinco-Ley correction; bilinear flow is not modelled.
- Steady-state VLP at each forecast step - no slugging or start-up dynamics.
- VLP: acceleration pressure loss is neglected; SSSV/restrictions use a homogeneous orifice equation (Cd 0.9); with the equation of state the wellstream keeps the PRP composition (test GOR/CGR are not used to re-mix it); downhole rate multipliers are not applied yet; the surface equipment is calibrated by Match Surface only from tests with a manifold pressure (the Gilbert-type chokes have no discharge coefficient, so the match uses Sachdeva); the choke is taken as isothermal and Sachdeva uses a fixed gas heat-capacity ratio k = 1.26; restriction regime uses a 0.53 pressure-ratio rule.
- Single-layer homogeneous reservoir; forecast water cut and GOR follow trend functions, not relative permeability.
- No artificial lift design yet (gas-lift and ESP are natural next modules).
- **Validated on synthetic data only.** Check against real field tests before relying on results.
