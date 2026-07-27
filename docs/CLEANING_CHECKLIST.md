# Cleaning & Preprocessing Checklist

Researched practice, with what each item found in this project.

Sources:
- [scikit-learn — Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [MachineLearningMastery — Data preparation without data leakage](https://machinelearningmastery.com/data-preparation-without-data-leakage/)
- [Towards Data Science — Data leakage in preprocessing, a visual guide](https://towardsdatascience.com/data-leakage-in-preprocessing-explained-a-visual-guide-with-code-examples-33cbf07507b7/)
- [Jaeger & Tierney, *When to Impute? Imputation before and during cross-validation*, arXiv:2010.00718](https://arxiv.org/abs/2010.00718)
- [Bookdown — Imputation (Missing Data), ch.17](https://bookdown.org/mike/data_analysis/imputation-missing-data.html)

---

## The organising principle

scikit-learn states it directly: transformations must be applied to train and
test alike, but **learnt from the training data only**. This risk applies to
"almost all transformations… including `StandardScaler`, `SimpleImputer`, and
`PCA`."

That splits every preprocessing step into two categories:

| Category | Examples | Where it belongs |
|---|---|---|
| **Stateless** — same answer row by row | date parsing, dropping a redundant column, deduplication, type coercion, unit fixes | `clean.py`, applied once |
| **Stateful** — learns a parameter from data | median for imputation, mean/sd for scaling, category levels for encoding, feature selection | **inside the CV pipeline** |

Putting a stateful step in the cleaning layer is a leak even when it looks
harmless. The practical test: *if I re-ran this on a single new row, would it
give the same answer?* A median fill would not.

---

## 1. Split before any stateful step

**Finding.** The first version imputed `satisfaction_score` with a median
computed over all 2,000 tickets inside `clean.py`. Every training row was then
filled with a statistic that had seen the validation rows.

Moved into `model._pipe` as a `SimpleImputer`, refit per fold.

**Nuance worth knowing.** Jaeger & Tierney find that imputing *before* CV carries
only a slight optimistic bias and has *lower variance*, giving better RMSE
overall for unsupervised imputation on identically-distributed data. So the
error here was small. In-fold imputation remains the conservative default, and
at this dataset size the compute cost is nil — but it is worth knowing the
literature does not treat it as a cardinal sin.

## 2. Encode categoricals inside the pipeline, not with `pd.get_dummies`

Two distinct problems with encoding up front:

1. **Leakage** — category levels are learned from every row, validation included.
2. **It breaks in production** — an unseen `industry` value changes the column
   set. There is no `handle_unknown` for `get_dummies`.

**Finding.** The feature layer called `pd.get_dummies(df, drop_first=True)` on the
full frame. Replaced with a `ColumnTransformer` inside every pipeline rung:

```python
OneHotEncoder(handle_unknown="ignore", drop="first")
```

Categoricals now leave the feature layer as raw strings. An unseen category
encodes to all-zeros instead of altering the schema.

The change was made on correctness grounds rather than for a score. At this
sample size the measured difference would sit well inside fold-to-fold noise
(sd ≈ 0.09), so quoting a before/after AUC for it would be reading signal into
a coin flip — the argument is that global encoding is *invalid*, not that it is
slower or lower-scoring.

## 3. Scale inside the pipeline

**Finding.** Already correct — `StandardScaler` sits in `_pipe`, so it is refit
per fold. Applied only to the linear rungs; tree models are scale-invariant and
scaling them is wasted work.

## 4. Understand *why* values are missing before choosing a treatment

The MCAR / MAR / MNAR distinction decides what is defensible:

- **MCAR** — missingness unrelated to anything. Complete-case analysis and
  imputation are both valid.
- **MAR** — related to *observed* variables. Imputation is valid; complete-case
  is biased.
- **MNAR** — related to the unobserved value itself. Imputation needs a model of
  the missingness mechanism; a plain median fill is not defensible.

**Finding.** `satisfaction_score` is 41.2% missing. Tested rather than assumed:
per-account missing rate is unrelated to churn (t-test p = 0.63) and flat across
ticket priority (0.405–0.422). Consistent with MCAR, so median imputation is
defensible — and a `satisfaction_missing` indicator is retained anyway, since
"did not respond" costs nothing to encode.

This also corrected a fabricated rationale: the original justification
("response rates differ by ticket severity") is contradicted by the flat rates.

**Separately worth knowing before using the column at all**: the observed values
are only 3, 4 and 5, in near-equal proportions, against a documented 1–5 scale.
Whatever we do about the missingness, the column cannot express dissatisfaction.
See `02_cleaning.py`.

## 5. Disposition by cause, not by percentage

A threshold rule like ">60% missing → drop" is a trap.

**Finding.** `subscriptions.end_date` is 90.3% null — and that null *is the
information*: the subscription is still open. A percentage rule would have
discarded one of the most informative columns in the dataset. Encoded as
`n_open_subs` / `pct_subs_ended` instead.

| Column | Missing | Cause | Treatment |
|---|---|---|---|
| `end_date` | 90.3% | structural (still active) | encode as a flag, never fill |
| `satisfaction_score` | 41.2% | no response (MCAR) | in-fold median + indicator |
| `feedback_text` | 24.7% | optional field | not used as a feature |

## 6. `NaN` does not mean zero

**Finding.** The first version applied a blanket `fillna(0)`, conflating three
meanings:

| Feature type | Missing means | Fill |
|---|---|---|
| counts (`n_tickets`, `total_usage_events`) | genuinely zero activity | `0` |
| recency (`days_since_last_usage`) | never happened — maximally stale | observation-window length |
| rates/means (`avg_satisfaction`, `error_rate`) | unknown | `NaN`, imputed in-fold |

An account with no tickets has an *undefined* average satisfaction. Filling it
with 0 invents a maximally unhappy customer out of an absence of data.

## 7. Deduplicate at the key level

**Finding.** 21 duplicate `usage_id` values, which inflated per-account event
counts. The original cleaning asserted uniqueness on `account_id` and
`subscription_id` but never checked `usage_id`.

## 8. Drop redundant columns, and say why

**Finding.** `arr_amount == mrr_amount * 12` for all 5,000 rows — perfectly
collinear. Dropped in `clean.py` with the reason recorded.

Near-duplicates pruned at |r| > 0.98: `feature_breadth` was
`unique_features_used / 40`, correlated at exactly 1.000.

## 9. Outliers: quantify, then justify keeping or removing

R4DS guidance: run the analysis with and without; if the effect is negligible,
replacing with `NA` is defensible; if substantial, do not drop without saying why.

**Finding.** MRR has a heavy right tail — 9.4% of subscriptions sit above the
Tukey upper fence of \$6,538, up to \$33,830. These are large enterprise
contracts, not data errors, and dropping them would remove exactly the accounts
the business most cares about retaining. Kept, with scaling handled in-fold.

## 10. Validate ranges and referential integrity; surface what you cannot fix

**Finding.** No negative durations, no out-of-range satisfaction scores, no
orphan foreign keys. But 1,077 of 2,000 tickets predate their account's signup,
and 19,128 of 24,979 usage rows predate their subscription's start.

These are generator artefacts that cannot be repaired without inventing data.
Reported by `clean.integrity_report()` rather than silently corrected — in a real
engagement they go back to data engineering first.

## 11. Control randomness explicitly

scikit-learn: passing integers to CV splitters "is usually the safest option and
is preferable" for reproducibility.

**Finding.** Already correct — integer `random_state` on every estimator, splitter
and permutation test.

## 12. Make cleaning reproducible and idempotent

**Finding.** `clean_all()` is a pure function of the raw tables. It was *not*
idempotent in one respect worth noting: a CSV round-trip turns booleans into the
strings `"True"`/`"False"`, and a `fillna(0)` on a boolean column left a
three-valued `{True, False, 0}` mix. `prep_xy` now coerces these back, so the
pipeline behaves identically from memory or from disk.

---

## Quick reference

| # | Check | Where |
|---|---|---|
| 1 | Stateful steps inside the CV pipeline | `model._pipe` |
| 2 | Encode with `OneHotEncoder(handle_unknown="ignore")` | `model._pipe` |
| 3 | Scale in-fold, linear models only | `model._pipe` |
| 4 | Test the missingness mechanism | `02_cleaning.py` |
| 5 | Disposition by cause, not percentage | `audit.missingness_report` |
| 6 | Count / recency / rate get different fills | `features/assemble.py` |
| 7 | Deduplicate on every key | `clean.py` |
| 8 | Drop redundant + collinear, with reasons | `clean.py`, `features/assemble.py` |
| 9 | Quantify outliers, justify the decision | `01_eda.py` |
| 10 | Range and integrity checks, surfaced | `clean.integrity_report` |
| 11 | Integer `random_state` everywhere | `model.py` |
| 12 | Cleaning is pure and round-trip safe | `clean.py`, `model.prep_xy` |
