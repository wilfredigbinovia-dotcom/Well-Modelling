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
- **VLP:** Beggs & Brill (Payne-corrected), modified Hagedorn & Brown (Griffith bubble flow, no-slip floor), no-slip homogeneous. Deviation survey, tapered strings, linear temperature profile. Heun-integrated top-down traverse.
- **Nodal:** stable (right-hand) IPR/VLP intersection with stability flag at the VLP minimum.

Oilfield units throughout: psia, °F, STB/d, scf/STB, ft, in, cp.

## Quick start

```bash
pip install -e ".[app]"
streamlit run app.py            # point-and-click app in your browser
pytest -q                       # 74 tests
jupyter notebook notebooks/demo.ipynb
```

### The Streamlit app (`app.py`)

The app is being rebuilt tab by tab. It currently has four tabs:

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
   - **Surface equipment:** manifold, line pipe, choke and fittings (equivalent length from L/D values), surrounding temperature and overall heat transfer coefficient.
   - **Downhole equipment:** Xmas tree, tubing, SSSV, restriction, gas lift mandrel, SSD and casing, PROSPER-style. Checks depths, diameters and roughness, requires tubing OD and casing ID when annular flow is selected, and checks the string against the survey. Shows flow sections, components and a schematic.
   - **Geothermal gradient:** formation temperature against MD (TVD from the survey), gradient per interval and the overall heat transfer coefficient.
   - **Average heat capacities:** oil 0.53, gas 0.51, water 1.0 BTU/lb/F by default.

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
  equipment.py      deviation survey and filter, surface/downhole equipment, geothermal gradient
  mpp.py            multiphase pseudo-pressure (Corey, black oil / EOS integrands)
  completion.py     skin models (Karakas & Tariq, MacLeod, Hawkins, deviation, partial penetration), sand control
  synthetic.py      known-answer test data
  plotting.py       figures
app.py                Streamlit app
notebooks/demo.ipynb  end-to-end walkthrough
tests/                pytest suite
```

## Validation so far

On synthetic data with a known answer (truth: Hagedorn & Brown, gravity 1.04, friction 1.15, PI 2.61):

- QC caught all four planted problems (gauge +200 psi, rate +30 %, GOR −40 %, 3-hour test) and passed every clean test.
- Matching recovered gravity 1.045, friction 1.10 ± 0.09, PI 2.60.
- Selection picked Hagedorn & Brown. Beggs & Brill fit almost as well statistically but only with friction ≈ 0.70, outside the acceptable range.
- Build-up fit recovered k = 50.0 md, skin 2.00, C 0.030 bbl/psi; build-up PI agreed with production-test PI.
- Pressure-transient solution matches the infinite-acting log approximation, the PSS slope and the wellbore-storage unit slope.

## Limitations (screening level)

- Multilayer-with-dP and horizontal friction models treat wellbore flow as single phase. Fractured wells assume infinite-conductivity geometry with a Cinco-Ley correction; bilinear flow is not modelled.
- Steady-state VLP at each forecast step - no slugging or start-up dynamics.
- Single-layer homogeneous reservoir; forecast water cut and GOR follow trend functions, not relative permeability.
- No artificial lift design yet (gas-lift and ESP are natural next modules).
- **Validated on synthetic data only.** Check against real field tests before relying on results.
