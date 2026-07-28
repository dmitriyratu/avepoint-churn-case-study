# Running this project

Environment, notebook order, runtimes and the test
suite. Split out of the README.

```bash
cd AvePoint
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Versions are pinned with `~=`, not `>=`. The scores here are separated by less
than 0.05 AUC, which is inside the drift between estimator versions — an unpinned
environment reproduces the narrative but not the numbers.

Notebooks are in [jupytext percent format](https://jupytext.readthedocs.io/) and
also execute as plain scripts:

```bash
cd notebooks && for nb in 0*.py 1*.py; do MPLBACKEND=Agg python "$nb" || echo "FAILED: $nb"; done
```

**`MPLBACKEND=Agg` is required for headless runs.** Without it matplotlib picks
an interactive backend and `plt.show()` blocks on a window nobody closes, so the
run hangs rather than failing. In Jupyter the backend is already non-interactive
and the variable is unnecessary.

All sixteen run clean end-to-end. `03` and `04` assert the leakage suite passes
and stop the pipeline if it does not. Notebooks must run **in order** — `05`,
`07`, `09`, `13` and `15` read `outputs/models/config.json` or
`churn_model.joblib`, which `04` writes.

Measured on one serial pass, nothing else running:

| Notebook | Runtime |
|---|---:|
| 01 EDA, 02 cleaning, 05 results, 07 audit | 4–9 s each |
| 15 retention economics + power | 11 s |
| 12 survival analysis | 16 s |
| 06 leakage quantification | 19 s |
| 16 generator audit (800 hazard simulations) | 33 s |
| 14 causal + uplift | 37 s |
| 11 churn reasons (500-permutation segment null) | 48 s |
| 10 sanity checks | 105 s |
| 09 classifier sweep | 129 s |
| 03 features + horizon/buffer sweep | 150 s |
| 13 drivers (SHAP nulls + permutation importance) | 160 s |
| 08 diagnostics + rolling-origin pooling | 220 s |
| 04 modelling (repeated nested CV dominates) | 645 s |
| **total** | **~27 min** |

`04` is 41% of the total on its own and the six product-question notebooks add
~5 min between them. Run them alone against an existing `outputs/models/` if you only
want the product-question work.

Everything runs serially by default, because `n_jobs=-1` deadlocks under some
container runtimes. Set `CHURN_N_JOBS=4` to opt into outer-loop parallelism.

`04` is slow for a reason worth keeping: the nested CV is repeated over 5 outer
splits because a single one moves by ~0.09 AUC on the seed alone, wider than the
effect being measured. `09`'s expensive part (20 sweeps over shuffled labels) is
cached to `outputs/reports/selection_null.csv`; set `REGENERATE = True` to
recompute.

Or drive the src modules directly:

```bash
python -c "
from src import pipeline
from src.model import evaluate_ladder, nested_ladder_cv

data = pipeline.build(verify=True)      # asserts the leakage suite before returning
print(data.summary.to_string())
print(evaluate_ladder(data.X, data.y).to_string(index=False))
print(nested_ladder_cv(data.X, data.y)[1].to_string())
"
```

### Tests

```bash
pytest tests/ -q          # 61 tests
```

Most of the bugs in this project's history were leaks, point-in-time errors, and
stale reporting that passed review and produced plausible numbers. The suite
asserts the properties that would have caught them: no observation row reaching
the cutoff (at several cutoffs, not just the configured one), ticket outcomes
censored when unresolved, positives falling strictly inside the prediction
window, every cohort account actually at risk, forbidden columns absent, the
cohort summary describing the arguments it was given rather than the configured
default, and pooled cutoffs sharing a column set — plus a negative control that
injects a leak and requires the gate to fire.

`test_analyses.py` does the same for notebooks 11–16, and three of its cases
encode bugs that were live in earlier drafts of that code and produced entirely
plausible output:

- **trimming that trimmed nothing.** The propensity score was clipped to
  `[0.05, 0.95]` and then tested against those same bounds, so the overlap filter
  was a silent no-op reporting "0 trimmed" while `overlap_report` flagged 16% of
  rows as outside support.
- **an AIPW churn rate of −0.17.** Handing all 71 features to a propensity model
  on 177 rows separated the arms almost perfectly, weights reached 100, and the
  potential-outcome mean left the unit interval. Fixed with a pre-specified
  confounder set and heavier regularisation; the test asserts the range.
- **a retention triangle filled past its horizon**, which made three-month-old
  cohorts appear to have twelve-month retention.

Plus the invariants specific to this kind of analysis: censored accounts never
counted as events, follow-up never extending past the extract date, survival
covariates being baseline rather than as-of-extraction state, calendar hazard
bounded by the risk set, out-of-fold uplift, E-value symmetry under `RR ↔ 1/RR`,
and `sample_size` and `minimum_detectable_effect` inverting each other.

Seven more pin the right-truncation null, and they guard it in both directions.
The uniformity test is required to *reject* a front-loaded churn process, so
"KS p = 0.92" is a result rather than a test that cannot fail. The simulation is
required to preserve every column except `churn_date`, so reproducing the
finding cannot be an artefact of the simulation itself. And the observed rate
ratio is required to land *inside* the null band rather than merely near it —
if a future change made the data genuinely distinguishable from the null, that
test fails and the withdrawn recommendation would need revisiting.

## Structure

```
AvePoint/
├── data/
│   ├── raw/                  # original CSVs (not committed)
│   └── processed/            # cleaned tables + feature matrix
├── notebooks/
│   ├── 01_eda.py             # EDA — quality pass on all rows, target pass on exploration split
│   ├── 02_cleaning.py        # cleaning: what is stateless, and what belongs in the fold
│   ├── 03_feature_engineering.py  # feature families + horizon/buffer sweep
│   ├── 04_modeling.py        # model ladder, nested CV, permutation test, operating point
│   ├── 05_results_validation.py  # recommendations, deployment, monitoring, mentoring
│   ├── 06_leakage_quantification.py   # what each form of leakage is worth
│   ├── 07_leakage_audit.py   # automated leakage + cleaning gates
│   ├── 08_diagnostics.py     # error analysis, learning curves, rolling-origin pooling
│   ├── 09_classifier_sweep.py # 15-model sweep + selection-bias null
│   ├── 10_sanity_checks.py   # positive controls, cohort variants, label coherence
│   │
│   │                         # --- the three product questions ---
│   ├── 11_churn_reasons.py   # reason codes, segment scan, cohort retention triangle
│   ├── 12_survival_analysis.py # KM, Cox PH, hazard shape, tenure/cohort/period split
│   ├── 13_drivers.py         # SHAP, permutation importance, ALE — each vs a null
│   ├── 14_causal_uplift.py   # propensity, AIPW, E-values, placebo, uplift/Qini
│   ├── 15_retention_actions.py # CLV, break-even precision, decision curve, power
│   └── 16_generator_audit.py # is the timing finding a fact, or a fact about the file?
├── src/
│   ├── config.py             # cutoff/buffer/horizon, extract date, exclusion lists
│   ├── load_data.py          # load the 5 raw tables
│   ├── clean.py              # parsing, dedup, integrity_report
│   ├── labeling.py           # cohort construction, observation-window truncation
│   ├── features/             # one module per feature family
│   ├── model.py              # model ladder, nested CV, permutation test, oof threshold
│   ├── robustness.py         # horizon/buffer sweep, feature-set comparison, pooling
│   ├── audit.py              # leakage + quality gates
│   ├── reasons.py            # retrospective churn description (never feeds a model)
│   ├── survival.py           # time-to-event: KM, Cox, hazard shape, calendar hazard
│   ├── generator.py          # right-truncation null: does the file produce this on its own?
│   ├── drivers.py            # SHAP/permutation/ALE with shuffled-label nulls
│   ├── causal.py             # propensity, IPW/AIPW, E-value, placebo, uplift
│   ├── economics.py          # CLV, break-even precision, decision curve, power
│   └── pipeline.py           # build() — one call for the whole chain
├── tests/
│   ├── test_pipeline.py      # 24 invariants: leakage, labels, point-in-time, pooling
│   └── test_analyses.py      # 37 invariants: censoring, AIPW range, trimming, power, the null
├── build/                    # builders (source); everything they write lands in outputs/
│   ├── build_main_deck.py    # the 16-slide case-study deck
│   ├── build_exec_summary.py # the 6-slide executive summary
│   ├── deck_style.py         # shared layout, palette, plain-English rule
│   └── build_explorer.py     # the raw-data orientation page
├── outputs/                  # all generated; safe to delete and rebuild
│   ├── decks/                # the two .pptx deliverables
│   ├── explorer/             # data_explorer.html — schema, grain, cutoff, traps
│   ├── figures/              # 44 PNGs, prefixed by the notebook that writes them
│   ├── models/               # persisted estimators + config.json
│   └── reports/              # 17 CSVs, one per analysis
└── docs/
    ├── PRODUCT_QUESTIONS.md  # the three questions answered, with evidence strength
    ├── ASSUMPTIONS.md        # key decisions and their rationale
    ├── CLEANING_CHECKLIST.md # cleaning/preprocessing practices, with findings
    ├── DATA_DICTIONARY.md    # field-by-field availability-at-prediction-time audit
    ├── EDA_CHECKLIST.md      # EDA practices followed, with what each one found
    └── FEATURE_ENGINEERING.md # churn FE taxonomy, sweep results, what helped
```

`reasons.py` reads `churn_events` on purpose — reason codes, refunds and
feedback text — which the feature layer bans by name. Reading them *after the
fact* to describe churn is a different question from using them to predict it,
and keeping the two in separate modules is what stops the distinction eroding.
`tests/test_analyses.py` asserts the separation holds.
