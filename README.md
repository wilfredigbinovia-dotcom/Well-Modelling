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
pytest -q                       # 26 tests
jupyter notebook notebooks/demo.ipynb
```

### The Streamlit app (`app.py`)

The sidebar holds the well setup (PVT, deviation survey, tubing, temperatures); it is being moved into dedicated setup tabs. Eight tabs follow the workflow in order:

1. **Options summary** - fluid type, method, flow type, well type, completion and sand control. Checks the combination, shows what the engine can calculate today, and lists the data the later steps will need. Analysis tabs lock when a choice is not modelled yet.
2. **Well tests** - upload a CSV (a template is downloadable) or load the demo tests; edit in place.
3. **QC** - adjustable thresholds; pass/warn/fail counts, chart and reasons.
4. **Correlation selection** - leave-one-out ranking with rationale; one click adopts the winner.
5. **Match** - choose correlation, parameters and error levels; override any test weight.
6. **Nodal** - live operating point, stability flag, VLP comparison and a WHP × water-cut sensitivity grid.
7. **Probabilistic** - pick a distribution per input; P90/P50/P10, exceedance, tornado; include match uncertainty.
8. **Transient** - build-up fit (k, skin, storage, PI) and the depletion forecast with flow-stop warning.

Results reset automatically when the well setup or tests change. Tables download as CSV.

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

- Black-oil correlations only; no compositional PVT, no gas/condensate wells yet.
- Steady-state VLP at each forecast step - no slugging or start-up dynamics.
- Single-layer homogeneous reservoir; forecast water cut and GOR follow trend functions, not relative permeability.
- No artificial lift design yet (gas-lift and ESP are natural next modules).
- **Validated on synthetic data only.** Check against real field tests before relying on results.
