# Robustness — where this breaks

Every negative result in this project, and the test that establishes it.
Split out of the README so the front page stays readable; nothing here is
abridged.

### Horizon and lead time

Prediction window opening 2024-06-30 throughout. **Buffer** = lead time the model
must give; zero is the standard default (score today, act today), non-zero forces
a warning *before* the customer is visibly on the way out.

| Horizon | Buffer 0 | Buffer 30 | Buffer 60 |
|---:|---:|---:|---:|
| 30 d | 0.561 | 0.522 | 0.454 |
| 60 d | 0.566 | 0.480 | 0.490 |
| **90 d** | **0.583** | 0.424 | 0.455 |
| 180 d | 0.587 | 0.490 | 0.427 |

**None of the twelve cells clears chance at the lower bound of its interval.**
The whole grid spans 0.42–0.59 while a typical interval is ±0.20 wide.

There are deliberately **no per-cell p-values**: twelve tests with the smallest
highlighted is the selection error described below. One permutation test is run,
on the pre-specified primary cell (90 d / no buffer): **p = 0.076**.

Two readings survive:

1. **Lead time is the dial that matters.** Every zero-buffer cell scores
   0.56–0.59; every cell demanding 30 or 60 days of warning drops to 0.42–0.52,
   consistently across four horizons. Whatever weak association exists sits in
   the period immediately before the customer leaves — accurate too late to act
   on.
2. **Horizon barely matters** — 30 → 180 days moves the zero-buffer score by
   0.026, a third of the fold-to-fold noise.

Short horizons are **underpowered rather than proven null**: 23 positives at 30
days cannot detect a modest effect.

### Leakage is worth more than the model

| Design | Features may see | CV ROC-AUC |
|---|---|---:|
| A. correct | observation window only | 0.583 |
| B. leaky | + rows dated after the cutoff | 0.422 |
| C. leaky | + columns derived from `churn_events` | **0.787** |

Neither of the two lessons here is the usual one.

**Leakage is not always helpful.** Design B scores *below* A: post-cutoff rows
add mostly noise. That kind of leak never announces itself with a suspicious
score, so the defence has to be structural rather than "watch for a number that
looks too good".

**A leak landing at 0.79 is more dangerous than one landing at 0.99.** Read C
against B, since C is built on B's frame: the outcome columns alone are worth
**+0.366**, taking the model from below chance to comfortably above it. 0.79 is
a perfectly plausible AUC for a churn model, which is the problem.
`config.POST_OUTCOME_COLS` excludes these by name, because the statistical
signature is not reliable enough to depend on.

### Feature engineering did not help

| | Features | CV ROC-AUC |
|---|---:|---:|
| Baseline (counts and levels) | 52 | **0.599** |
| + window ladder, acceleration, trend slope, gaps, MRR volatility | 73 | 0.583 |

Twenty-one engineered columns moved the score down by 0.016, well inside noise,
so the honest reading is "no effect". The binding constraint is data, not
features. The enriched set stays as the default on purpose: switching to the
smaller one *because* it scored 0.016 higher would be the same selection error
this project documents elsewhere.

### The selection trap — why "run every classifier" misleads here

The `caret` / **PyCaret** / **LazyPredict** approach — fit ~30 classifiers,
report the winner — is a fast screen, but at 177 rows it manufactures results.
`notebooks/09_classifier_sweep.py` measures how much: fifteen classifiers under
identical repeated CV, then the whole sweep repeated 20 times **on shuffled
labels**.

| | CV ROC-AUC |
|---|---:|
| Individual model, shuffled labels | 0.498 |
| **Best-of-15, shuffled labels** | **0.566** (max reached **0.624**) |
| Observed best (AdaBoost) | 0.594 |
| **Selection-corrected p** | **0.300** |

The winner of the sweep is beaten by **30% of pure-noise runs**.

The substantive conclusion survives and is better supported: the spread across
fifteen model families (0.520–0.594, a range of 0.074) is **smaller than the
fold-to-fold noise within any one of them** (sd ≈ 0.090).

### Is the result real, or a mistake? — `notebooks/10_sanity_checks.py`

A near-chance AUC should not be accepted on trust. Four checks.

**The pipeline is not broken.** Planting targets of known strength and re-running
everything:

| Planted target | CV ROC-AUC |
|---|---:|
| Strong (label = MRR > median) | 0.965 |
| Strong, 25% of labels flipped | 0.669 |
| **Weak but genuinely real** (2 features + noise) | **0.584** |
| Null (shuffled labels) | 0.494 |
| **Actual churn label** | **0.583** |

Read row 3 against row 5: our result is also what a real-but-weak relationship
looks like at this sample size, so "there is definitely nothing here" overstates
the evidence.

**No single feature carries signal.** The cheapest version of the whole argument:

| | Max single-feature AUC |
|---|---:|
| Observed, across 86 encoded features | 0.622 |
| **Shuffled labels, max of the same 86** | **0.612** (sd 0.020) |
| P(noise beats the observed max) | **0.275** |

The best feature in the matrix is what screening 86 coin flips produces. Two
lines of code, and it makes the same point as several notebooks of increasingly
elaborate negative results.

**The cohort definition is not the problem.**

| Definition | n | Positives | CV AUC |
|---|---:|---:|---:|
| **A. current** (live sub, exclude prior churners) | 177 | 54 | **0.583** |
| B. reactivations allowed back in | 335 | 83 | 0.545 |
| C. no live-subscription requirement | 347 | 88 | 0.552 |
| D. event = subscription ends | 335 | 57 | 0.462 |

Nearly doubling the cohort makes it *worse*. No alternative rescues the result.

**The label is not coherent — this is the ceiling.** Three separate ways to say
an account churned, each compared against its own chance baseline rather than
against 50% (`src.audit.label_source_agreement`):

| | Agreement | Expected if unrelated | κ | p |
|---|---:|---:|---:|---:|
| `churn_flag` vs `churn_events` | 37.6% | 38.6% | −0.016 | 0.56 |
| `churn_flag` vs ended subscription | 44.4% | 43.1% | +0.024 | 0.45 |
| `churn_events` vs ended subscription | 58.0% | 55.1% | +0.065 | 0.14 |
| **All three agree** | **20.0%** | | | |

Every pair lands on its own chance baseline. These are not three noisy views of
one fact — they are three unrelated columns wearing the same name.

The date check is the strongest form of the argument because it never touches
`churn_flag`, so it survives anyone who simply declares the event log
authoritative (`src.audit.churn_date_coherence`). Of the 386 churn events that
can be compared to a subscription ending, **6 land on the same day (1.6%)**, 12%
within a week, **median gap 62 days** — and the remaining 214 events belong to
accounts with no ended subscription at all. Two systems recording the same
departure should agree on the date. These do not agree even approximately.

The accurate statement commits to neither reading: **if a weak relationship
exists, this dataset cannot resolve it.** 54 positives, three mutually unrelated
recordings of the outcome, timestamps that do not order events correctly.

### Why 0.58 and not higher — diagnosed, not guessed

`notebooks/08_diagnostics.py` separates the fixable causes from the rest.

| Candidate cause | Evidence | Verdict |
|---|---|---|
| Data leakage | full audit suite passes, incl. a by-name forbidden-column gate | ruled out |
| Feature engineering | 21 added features moved AUC *down*; cutting to the top 3 costs 0.06 | not the constraint |
| Overfitting | train 0.97–1.00 vs validation 0.54–0.58 | real, but a symptom |
| **Unlinked inputs** | usage and ticket timestamps are uniform over the whole extract and correlate with their own account's signup date at **r = 0.002** / **r = 0.014**; inside those tables priority does not predict resolution time (p = 0.33) and plan does not predict usage (p = 0.92) | **primary constraint** |
| **A random target** | churn dates are a uniform draw between signup and the extract boundary (KS p = 0.92) | **primary constraint** |
| **Data quality** | the three recorded churn signals are mutually unrelated (all κ ≈ 0), so the choice of ground truth is unverifiable | major contributor |
| Sample size | learning curve still rising: +0.09 AUC per 100 rows | real, but secondary — see below |
| **Irreducible** | nearest neighbours disagree at 0.410 vs 0.424 for random pairs | **large floor** |

The sample-size row was previously listed as the primary constraint, on the
strength of a learning curve that had not flattened. `notebooks/16` supersedes
that reading: only `subscriptions` has internal structure (price tracks plan and
seats, ARR is exactly 12 × MRR), and nothing in it relates to churn. More rows
of unlinked timestamps do not help, and the rising learning curve was noise.
**AUC 0.534 is the correct answer to the question this data can be asked.**

Three results worth singling out:

**Every model memorises.** The boosters reach train AUC **1.000** on 177 rows and
validate between 0.54 and 0.57. Not a tuning problem — far more capacity than 54
positives can constrain. Even the selected L2 logistic memorises to 0.97.

**Fewer features makes it worse**, monotonically (3 features → 0.525, all 73 →
0.588). The signal is not concentrated in a few predictors drowned out by noise;
it is smeared thinly across many weak ones. That rules out the obvious remedy.

**Neighbours are barely more alike than strangers.** Accounts adjacent in feature
space disagree on outcome 41.0% of the time, against 42.4% for two customers
picked at random — a ratio of 0.97. That is the Bayes floor, and it belongs to
the data, not the model.

**Error analysis**: of 15 false negatives, only 1 sits near the threshold. The
rest are scored confidently safe, so no operating point recovers them. The
churners we catch are newer, higher-MRR, fewer-seat accounts; the ones we miss
are older, larger-seat, lower-MRR accounts contracting slowly. That part is
reducible, with telemetry that captures large-account contraction.

### More rows: rolling origin

The diagnosis above says sample size is the binding constraint, and the pipeline
is parameterised on the cutoff — so the remedy is available. Rebuilding the
cohort at four quarterly cutoffs and pooling, with folds **grouped by account**
so no customer straddles a split:

| | n | Positives | CV ROC-AUC | Fold sd |
|---|---:|---:|---:|---:|
| Single cutoff (headline) | 177 | 54 | 0.583 | 0.098 |
| **Pooled, account-grouped** | **648** | **159** | **0.560** | **0.034** |
| Pooled, ungrouped *(wrong)* | 648 | 159 | 0.576 | — |

Three times the positives from 281 distinct accounts, and **fold-to-fold noise
falls by 65%**: a much better-pinned-down measurement that lands in the same
place. Forgetting to group by account adds **+0.016** of optimism, which is why
the wrong version is reported next to the right one.

Two caveats. Quarterly cutoffs 91 days apart mean an account's features overlap
heavily even though its label windows barely do, so the effective sample is
below the row count. And the earlier cutoffs have materially lower positive
rates, so part of the result averages over different base rates.

### Cross-check against published work

The one substantive public analysis of this dataset
([saas-growth-retention-strategy](https://github.com/saishyam43-oss/saas-growth-retention-strategy))
**did not build a predictive model** — "retention analysis is based on observed
churn events rather than predictive labels," no AUC reported. The only published
work on RavenStack chose diagnostics over prediction.

| Their claim | Our data | |
|---|---|---|
| ~66% eventually churn | 70.4% | confirmed |
| Churn is front-loaded | <3mo 44.7% → 12mo+ 16.7%, but non-monotone | partly |
| Broad exploration → more churn | corr = −0.13 | contradicted |
| Time-to-first-value ~76 days | ours: **−251 days** | not reproducible |

Their time-to-value lever was a feature we lacked, and it scored a single-feature
AUC of 0.610. But **98% of accounts have a negative TTFV** (first usage precedes
signup), it correlates with `days_since_signup` at **+0.90**, and 0.610 is below
what shuffled labels produce as a max-of-86 anyway. It is the documented
timestamp corruption wearing a new name. Not added, and the decision is recorded
in `docs/FEATURE_ENGINEERING.md` rather than left silent.

---

Leakage controls and the audit gates live below.

Every number above is produced under an automated audit suite (`src/audit.py`)
that must pass before any result is reported:

| Gate | Threshold | Result |
|---|---|---|
| Temporal provenance — no datetime >= cutoff, every column | 0 violations | PASS |
| Forbidden columns — outcome + point-in-time-unsafe, by name | none present | PASS |
| Single-feature AUC | fail >= 0.80 | PASS (max 0.622) |
| Perfect separation | none | PASS |
| Identifier / row-order leakage | AUC < 0.60 | PASS |
| Duplicate rows | none | PASS |
| Constant columns | none | PASS |

The provenance gate caught a leak code review missed: 5 support tickets opened
before the cutoff but closed after it, whose resolution fields were not knowable
at prediction time. See `docs/DATA_DICTIONARY.md`.

The by-name gate matters more than it looks. `n_churn_events` scores 0.92 alone
and any threshold catches it; `total_refund_usd` scores 0.64 and no reasonable
threshold would — yet it is just as invalid. Leakage is a property of provenance,
not of effect size.
