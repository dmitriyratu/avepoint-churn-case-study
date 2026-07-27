# EDA Checklist

Synthesised from Roger Peng's *Exploratory Data Analysis with R* checklist,
the variation/covariation framing in *R for Data Science* ch. 7, and the
leakage-prevention guidance below. Each item records what it found in this
project.

Sources:
- [Peng, *EDA Checklist*](https://bookdown.org/rdpeng/exdata/exploratory-data-analysis-checklist.html)
- [Wickham & Grolemund, *R4DS* ch.7 — Exploratory Data Analysis](https://r4ds.had.co.nz/exploratory-data-analysis.html)
- [Hex — Feature leakage: detect, prevent, fix](https://hex.tech/blog/feature-leakage/)
- [MachineLearningMastery — Data preparation without data leakage](https://machinelearningmastery.com/data-preparation-without-data-leakage/)
- [Analytics Vidhya — Data leakage and its effect on model performance](https://www.analyticsvidhya.com/blog/2021/07/data-leakage-and-its-effect-on-the-performance-of-an-ml-model/)

---

## 0. Split before you explore  ← the one that governs all the others

> "EDA should operate exclusively on `X_train` and `y_train`. If you're computing
> correlations or running baseline models on the full dataset, you've already
> introduced the conditions for contamination." — Hex

Any *target-related* statistic computed on all rows leaks into every modelling
decision you make afterwards, because you chose those decisions knowing the
answer. Distribution checks and data-quality work are fine on everything; the
moment the target enters, you must be on the training split.

**Finding**: the first version of `01_eda.py` computed churn rate by segment,
feature-vs-churn boxplots, and target correlations on all 500 accounts. Every
feature idea that came out of it was informed by the full label set. Now split
into two passes — see §1.

**How far this actually goes, stated honestly.** The split keeps any individual
EDA number off the full label set. It does *not* make the headline CV score
unbiased, because modelling cross-validates over the whole cohort — rows
inspected in Part B reappear in the folds that produce the reported figure. With
177 accounts, permanently sacrificing 30% of them to a sealed confirmation set
would cost more precision than the residual bias is worth. That is a trade-off,
not a guarantee, and the nested CV in `04_modeling.py` is what carries the
weight instead.

## 1. Formulate the question first

Peng's first item, and the one that prevents aimless plotting. The question
determines the unit of analysis and the label.

**Finding**: "why do users churn" is not a modelling question. Sharpened to:
*given what we know about an account on 2024-06-30, will it churn in the next
90 days?* That immediately implies an observation window and a cohort, which is
what the first pass was missing. The horizon itself is swept rather than
assumed — see `FEATURE_ENGINEERING.md`.

## 2. Check the packaging

Rows, columns, file sizes — do they match what you were told?

**Finding**: 5 tables, row counts match the dataset README exactly
(500 / 5,000 / 25,000 / 2,000 / 600).

## 3. Structure, dtypes, and the top *and* bottom of the data

Peng stresses looking at both ends — sorted data often hides its problems at the
tail.

**Finding**: all date columns arrived as strings. `arr_amount` is an exact
`mrr_amount * 12`. After a CSV round-trip, booleans return as the strings
`"True"`/`"False"` — which crashed the modelling notebooks until fixed.

## 4. Check your "n"s

Count things you can independently verify. Peng's point is that a count you can
reason about is the cheapest bug detector available.

**Finding**: 500 accounts vs 352 appearing in `churn_events` vs 110 with
`churn_flag = True`. Those three numbers cannot all be describing the same
thing — and they aren't. `churn_flag` agrees with the event log for only 37.6%
of accounts. This was the single most important EDA finding in the project and
the first version missed it entirely.

Counting the same concept a *fourth* way (accounts with an ended subscription)
makes it worse, and it turns out to matter for the cohort rule as well as the
label — see `10_sanity_checks.py`.

## 5. Validate against an external source

**Finding**: the dataset README documents the intended generation process
(referential integrity, "signup ≤ subscription ≤ churn"). Checking against it
showed the generator did *not* honour its own temporal claims: 1,077 of 2,000
tickets predate their account's signup, and 19,128 of 24,979 usage rows predate
their subscription's start.

## 6. Variation — examine each variable alone

Distribution, spread, skew, and specifically:

- [x] **Missing values** — rate per column, and *why* each is missing
- [x] **Constant / near-constant columns** — carry no information
- [x] **Cardinality** — high-cardinality categoricals need care in encoding
- [x] **Duplicates** — at row level and at key level
- [x] **Outliers and impossible values** — negatives where impossible, out-of-range scores
- [x] **Structural zeros** — a zero that means "none" vs one that means "unknown"

**Finding**: 21 duplicate `usage_id`s. `end_date` is 90.3% null, but that is
*structural* — the subscription is still open — so a naive ">60% missing, drop
it" rule would have discarded one of the most informative fields in the table.
Missingness disposition has to key on cause, not percentage.

> R4DS: repeat the analysis with and without outliers. If the effect is
> negligible, replacing them with `NA` is defensible; if it is substantial, do
> not drop them without saying why.

## 7. Covariation — relationships between variables

- [x] Feature-vs-feature correlation (collinearity, redundancy)
- [x] Feature-vs-target association — **training split only**
- [x] Categorical-vs-target rates with base-rate reference lines
- [x] Interactions worth engineering

**Finding**: `feature_breadth` correlates with `unique_features_used` at exactly
1.000 (it is that column divided by 40). Four such near-duplicate pairs found.

## 8. Understand the target

For classification specifically:

- [x] Class balance — drives metric choice and whether weighting is needed
- [x] Is the label well-defined, dated, and internally consistent?
- [x] Base rate, so AP can be read against something

**Finding**: 30.5% positive in the temporal cohort (mild imbalance; class
weights suffice, no SMOTE). Accuracy is unusable as a metric — a majority-class
predictor scores 69.5%. Reporting ROC-AUC plus average precision against the
base rate instead.

Also found here, and more consequential: `satisfaction_score` is documented as
1–5 but only ever takes the values 3, 4 and 5, near-uniformly. A field that
cannot record dissatisfaction will not predict churn.

## 9. Leakage screen — do this in EDA, not after modelling

> "The cheapest place to catch leakage is during exploratory data analysis,
> before you've trained anything."

- [x] For each column, ask: **would I have this value at prediction time?**
- [x] Any single feature with suspiciously high target association
- [x] Identifier or row-order correlation with the target
- [x] Fields that resolve *after* the row is created (censoring)

**Finding**: `churn_events` columns reconstruct the label — restoring them takes
CV AUC from ~0.54 to ~0.91 (measured in `06_leakage_quantification.py`). Also
caught 5 support tickets opened before the cutoff but closed after it, whose
resolution fields were unknowable at prediction time. Automated in
`src/audit.py`; per-field verdicts in `DATA_DICTIONARY.md`.

Worth noting *which* leak the statistical gate would have caught.
`n_churn_events` scores 0.92 on its own and trips it immediately.
`total_refund_usd` scores 0.64 and sails through — despite being a column that
exists only because the customer left. Leakage is a property of provenance, not
of effect size, which is why `audit.forbidden_columns` checks by name.

## 10. Try the easy solution first, then challenge it

Peng's last three items. Establish a floor, then make the result work to beat it,
then try to break your own conclusion.

**Finding**: the floor (`DummyClassifier`) was never established in the first
pass, so ~0.55 AUC looked like a result rather than noise. The model ladder now
starts at a prior-only classifier, a 300-shuffle permutation test asks whether
anything above it is real, and — because the reported model was *chosen* as the
ladder maximum — nested CV asks the same question about the selection itself.
The second check is the one that changes the answer; see `04_modeling.py`.

## 11. Document decisions as you go

Every drop, fill, and exclusion needs a stated reason. If the reason turns out
to be unsupported, record the correction rather than quietly changing it.

**Finding**: the original satisfaction-score imputation was justified by
"response rates differ by ticket severity." They don't — missing rates are
0.405–0.422 across all four priorities. Correction recorded in `ASSUMPTIONS.md`.

---

## Quick reference

| # | Check | Where |
|---|---|---|
| 0 | Split before exploring the target | `01_eda.py` §B |
| 1 | Question defines unit of analysis + label | `labeling.py` |
| 2 | Packaging: shapes match expectations | `01_eda.py` §A1 |
| 3 | dtypes, head **and** tail | `01_eda.py` §A2 |
| 4 | Check your n's against each other | `01_eda.py` §A3 |
| 5 | Validate against external documentation | `clean.integrity_report` |
| 6 | Variation: missing, constant, duplicate, outlier | `01_eda.py` §A4-A6 |
| 7 | Covariation: collinearity, target association | `01_eda.py` §B2 |
| 8 | Target: balance, definition, base rate | `01_eda.py` §B1 |
| 9 | Leakage screen per column | `audit.py`, `DATA_DICTIONARY.md` |
| 10 | Baseline floor, then challenge it | `model.model_ladder` |
| 11 | Document every decision and correction | `ASSUMPTIONS.md` |
