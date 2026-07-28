# The three product questions

> The product team wants to understand:
> 1. Why users are leaving
> 2. How to predict churn before it happens
> 3. What actions can improve retention

This document answers those three directly, with the evidence behind each answer
and how strong that evidence is. It is the summary layer over notebooks 11–15;
every number is produced by code and named to its notebook.

Two of the three answers are negative. Where that is the case the document says
what would change it, because "no" without a route forward is not an answer a
product team can use.

---

## Short version

| Question | Answer | Strength |
|---|---|---|
| **1. Why are users leaving?** | **This extract cannot say.** No attribute, segment, tenure band or stated reason explains who leaves — and the ×2.8/yr calendar rise that looked like the answer is reproduced in full by drawing each churn date at random between signup and the last day of the file. | Strong negatives throughout; the artefact is demonstrated, not suspected |
| **2. Can we predict churn before it happens?** | **No.** Three independent formulations land at chance (nested CV 0.534, Cox concordance 0.509, log-rank/Cox global p = 0.57). Expected given (1): the target is a random date and two of the five tables are not joined to their accounts in time. | Strong |
| **3. What actions improve retention?** | **Not identifiable from this data** — no intervention is recorded anywhere in the schema. Separately, the economics say targeting is unnecessary: break-even churn probability is **10.3%** against a base rate of **30.5%**. | Structural; the economics are robust to the model being useless |

---

## Question 1 — Why are users leaving?

Five standard routes to "why", then a sixth section that tests the only one of
them that returned anything. All six end in a negative, and the negatives matter
because each closes off an analysis that would otherwise be shipped with
confidence.

### 1a. The stated reasons — unusable (notebook 11)

`churn_events` records a `reason_code` and a `feedback_text` for every
departure. It is the field that exists to answer this question, and it fails
three independent tests:

| Test | Result |
|---|---|
| Distribution over 6 codes vs uniform | χ² p = **0.70** — no dominant cause |
| Reason vs 9 behavioural measures (Kruskal-Wallis, BH-corrected) | min p = **0.91** |
| Reason code vs the customer's own free text | Cramér's V = **0.09** — independent |

Accounts coded `support` have the same ticket counts and satisfaction scores as
accounts coded `pricing`. Accounts coded `features` used the same number of
features as everyone else. Accounts coded `budget` write "missing features" as
often as "too expensive".

**A "top churn reasons" chart built from the counts in this field would be
confident and entirely fictional.** That chart is the most likely wrong output
of this dataset, which is why the test is run rather than the field simply used.

### 1b. Segments — nothing (notebooks 11, 12)

| Method | Result |
|---|---|
| Churn rate by 5 segments, χ², BH-corrected | min p = **0.71** |
| Widest segment gap vs a shuffled-label null | observed 21.6pp vs null mean 22.4pp, **p = 0.47** |
| Log-rank over 7 baseline segments (uses event timing) | min BH p = **0.39** |
| Cox PH, 21 baseline covariates, global LR test | p = **0.57**, C = 0.571 |

The proportional-hazards assumption holds (0 of 21 covariates violate it), so
the null is a real null rather than a misspecification artefact.

`referral_source` is closest in both frameworks (χ² p = 0.14, log-rank p = 0.056)
and survives neither correction. It is tracked rather than dismissed.

### 1c. Model-derived drivers — an artefact (notebook 13)

SHAP produces a clean ranking led by `tickets_per_seat`. Every check says not to
believe it:

| Check | Observed | Null | Verdict |
|---|---|---|---|
| Top feature mean \|SHAP\| | 0.318 | 0.278 (p = **0.24**) | inside the noise distribution |
| Attribution concentration (Gini) | 0.678 | 0.635 (p = 0.08) | marginal, does not clear 0.05 |
| Top-10 stability under bootstrap | 0.30 Jaccard | 0.20 shuffled | **12 different winners in 25 fits** |
| Permutation importance (held-out) | 0.012 | 0.050 (p = **1.0**) | *beaten* by noise |
| TreeSHAP vs logistic top-10 agreement | 2 of 10 | — | models disagree on the drivers |

The gap between SHAP and permutation importance is the diagnosis: SHAP scores
contribution to the **fitted** predictions, permutation scores contribution to
**held-out** AUC. A feature that helps in-sample and hurts out-of-sample is
memorisation, and every model here memorises (train AUC 1.000, validation 0.54).

The transferable point: **explainability tooling has no failure mode.** SHAP
does not decline to rank when the model is worthless, and the output is more
persuasive than a p-value because it is visual and specific.

### 1d. Tenure — a composition artefact (notebook 12)

This one overturns a recommendation the project previously made.

The pooled hazard falls sharply with tenure — Weibull **ρ = 0.737, 95% CI
[0.673, 0.801], p = 1.7e-13**, the most emphatic result anywhere in this work.
Read alone it says churn is front-loaded and onboarding is where to spend.

It does not survive decomposition:

- **Within each signup cohort, ρ returns to 1** (0.87, 1.25, 1.17, 1.06, 0.88 —
  two point the other way, none significant against exponential).
- **At fixed tenure, cohorts differ enormously**: 30-day survival runs from 0.98
  for the 2023Q2 cohort to 0.59 for 2024Q4, measured at identical age, so
  censoring cannot explain it.

Recent cohorts churn faster at *every* age, and recent cohorts contribute most of
the short-tenure observations. That produces a falling pooled hazard without any
account ever becoming safer.

> **Tenure is not a risk factor. A structured onboarding programme is the wrong
> intervention** — within any cohort, a day-300 account is as likely to leave as
> a day-10 account.

### 1e. Calendar time — where the answer appeared to be (notebook 12)

The person-period table separates cohort from period, and it reads as a period
effect:

| | Result |
|---|---|
| Monthly churn hazard, 2023 mean | 0.050 |
| Monthly churn hazard, 2024 mean | 0.119 |
| December 2024 | **0.225** |
| Poisson trend, log(at-risk) offset | RR **1.089/month**, 95% CI [1.067, 1.112] |
| Implied annual | **×2.8** |
| p | **2e-16** |

The at-risk base is flat at ~200 accounts through 2024, so this is not a growth
artefact, and cohort retention curves agree: month-3 retention fell from 0.82
(2023 cohorts) to 0.54 (2024 cohorts).

### 1f. That answer does not survive the right null (notebook 16)

One enormous, exquisitely significant result surrounded by nothing but nulls is
a pattern worth explaining before acting on it. The tests above cannot explain
it, because their null is "the rate is flat" — which is not the null that
matters on an extract with a hard end date.

> No churn date can land after 2024-12-31. A generator that draws each one
> uniformly between signup and that boundary produces a hazard of
> `1 / (END − t)`, rising without limit toward the end of the file, on data where
> nothing whatever happened.

The churn dates are exactly that draw, and rebuilding the file from that one
rule reproduces the finding in full:

| | Observed | Uniform-draw null |
|---|---|---|
| Churn date position in `[signup, END]` | mean 0.503, KS p = **0.92** | uniform by construction |
| Monthly rate ratio | 1.0893 | **1.0885**, 95% band [1.072, 1.106] |
| Implied annual | ×2.79 | **×2.78** |
| Poisson trend p | 2.2e-16 | median **2.8e-16** |
| Months inside the null band | — | **24 of 24** |
| Where the observed effect sits in the null | — | **52nd percentile** |

Uniformity holds inside every signup quarter, so this is not a mixture effect.
The null's median p-value is marginally *more* significant than the real data's:
**a small p-value is evidence against one null, and this was the wrong one.**

The same draw manufactures the tenure gradient in 1d. Simulated data with no
tenure effect in it clears p < 0.05 in **93%** of runs (median p = 0.002 against
the observed 0.006), because a recent signup has a shorter window for a random
date to land in.

> **Answer to Question 1: this extract cannot say why customers leave.** Not
> "we need a price-change log to attribute the rise" — there is no rise to
> attribute. Asking the business to hunt for a 2024 pricing event or outage
> would send a team after something that did not happen. The business timeline
> is still the right request, but only against an export whose timestamps are
> recorded against the accounts they belong to.

---

## Question 2 — How to predict churn before it happens

**Answer: not from this data.** Three independent formulations of the prediction
problem land at chance.

| Formulation | Metric | Result |
|---|---|---|
| Binary classification, 90-day window, model ladder | CV ROC-AUC | 0.583, CI [0.37, 0.75] |
| ...with model selection cross-validated | nested CV AUC | **0.534 ± 0.016** |
| Cox PH, baseline covariates, 500 accounts / 352 events | Harrell's C | **0.571**, global p = 0.57 |
| Cox PH, cohort features, time-to-event outcome | CV concordance | **0.509 ± 0.085** |

The last row matters most. Same 177 accounts and same point-in-time features, but
the outcome becomes "how long until they left" censored at extraction rather than
"did they leave within 90 days". That recovers **81 events instead of 54** from
the identical accounts — and still lands at chance. The negative is not an
artefact of the binary framing, the 90-day window, or the single-cutoff design.

### The "before it happens" part is answered separately, and worse

The horizon/buffer sweep (notebook 03) demands lead time explicitly. Every
zero-buffer cell scores 0.56–0.59; every cell requiring 30 or 60 days of warning
drops to 0.42–0.52, consistently across four horizons.

> Whatever weak association exists sits in the period immediately before the
> customer leaves — **accurate too late to act on**.

### Why this follows from Q1

Section 1f leaves nothing for a model to find. The target is a random date, and
the tables the features are built from are not joined to their accounts in time:

| Column | Uniform over the whole extract? | Correlation with its own account's signup date |
|---|---|---|
| `feature_usage.usage_date` | yes, KS p = 0.48 | **r = 0.002** |
| `support_tickets.submitted_at` | yes, KS p = 0.53 | **r = 0.014** |
| `subscriptions.start_date` | no, KS p < 1e-4 | r = 0.68 |

That is the origin of the quality numbers reported since notebook 01 — 77% of
usage rows before their own subscription, 53% of tickets before the customer
existed. They are not columns to repair; they are the visible edge of "the date
was drawn at random".

`generator.table_linkage` goes one level more basic and asks whether the columns
*inside* each table relate to each other as the schema says. Only subscriptions
passes: price tracks plan tier and seat count, ARR is exactly 12 × MRR. Usage
volume does not differ by plan (p = 0.92) or for beta features (p = 0.45).
Ticket priority does not predict resolution time (p = 0.33) or first response
(p = 0.13), and escalated tickets do not score lower on satisfaction (p = 0.33).

### What is diagnosed as the constraint

| Candidate | Evidence | Verdict |
|---|---|---|
| Data leakage | full audit suite passes | ruled out |
| Feature engineering | 21 added features moved AUC *down* | not the constraint |
| **Unlinked inputs** | usage/ticket timestamps at r = 0.002 / 0.014 to their own account | **primary** |
| **Random target** | churn dates uniform on `[signup, END]`, KS p = 0.92 | **primary** |
| Label quality | 3 definitions agree on **20%** of accounts | major |
| Sample size | learning curve rising at +0.09 AUC/100 rows | secondary — the curve was reading noise |
| Irreducible | neighbours disagree 41.0% vs 42.4% for random pairs | large floor |

Sample size was previously listed as the primary constraint. More rows of
unlinked timestamps do not help; **AUC 0.534 is the correct answer to the
question this data can be asked.**

---

## Question 3 — What actions can improve retention?

### The blocking fact

**The dataset records no intervention.** No account was offered a discount,
called by a CSM, or enrolled in a programme — or if any were, the extract does
not say so. There is no treatment variable, so the causal question is *not
identified* here under any estimator. Doubly-robust estimation is robust to
misspecifying one of two nuisance models; it is not robust to the treatment never
having happened.

### What the observational proxies say (notebook 14)

Four things accounts did that a product team might want to encourage or prevent,
estimated with cross-fitted AIPW under an explicit unconfoundedness assumption:

| Proxy action | Naive diff | Adjusted ATE | 95% CI | p | E-value |
|---|---|---|---|---|---|
| Upgraded | −0.126 | **−0.110** | [−0.276, +0.032] | 0.14 | 2.22 |
| Downgraded | +0.016 | +0.090 | [−0.111, +0.323] | 0.50 | 1.95 |
| Ticket escalated | −0.028 | −0.015 | [−0.239, +0.216] | 0.89 | 1.29 |
| Auto-renew on | −0.055 | −0.035 | [−0.357, +0.201] | 0.87 | 1.45 |

Every interval crosses zero. Three findings sit behind the table:

**Confounding is severe and adjustment only partly removes it.** Unadjusted,
upgraders differ from non-upgraders by SMD 0.87 on subscription count, 0.81 on
usage, 0.73 on tenure. Bigger, older, more-engaged accounts both upgrade *and*
stay; neither causes the other. Weighting cuts imbalanced covariates from 13 to 5.

**An E-value of 2.2 is not a defensible basis for a decision.** A confounder
associated with both upgrading and retention by a risk ratio of 2.2 erases the
whole estimate. *"The account was doing well"* — a champion user, a successful
rollout, next year's budget approved — clears that easily and is unmeasurable in
this schema.

**The design's noise floor is ±15pp.** Running the entire AIPW pipeline on
*randomly assigned* placebo treatments, where the true effect is exactly zero,
produces estimates up to ±0.15. Every observed estimate sits inside that band.
An independent closed-form power calculation puts the minimum detectable effect
at 17.2% for 88 accounts per arm — two methods agreeing to within two points.

**Uplift modelling does not rescue it.** The Qini coefficient is 0.46 against a
null mean of −1.16 with sd 1.97 (p = 0.30), and more fundamentally: uplift
assumes treatment was randomly assigned within covariate levels. Here accounts
chose to upgrade, so the ranked "uplift" mixes treatment effect with selection.
A Qini curve on observational data is not evidence that targeting works.

### The economics answer a different question, and answer it clearly (notebook 15)

An intervention costing `C` that saves a customer worth `V` with effectiveness
`e` pays off on an account only when its churn probability exceeds `C / (e·V)`.

| | |
|---|---|
| CLV (median MRR, 3yr, discounted, integrated from the KM curve) | **$7,310** |
| Campaign cost per account | $150 |
| Assumed effectiveness | 20% |
| **Break-even churn probability** | **10.3%** |
| **Cohort base rate** | **30.5%** |

> **The base rate is three times the break-even threshold.** Contacting every
> account in the at-risk cohort is already profitable, and no ranking improves on
> a policy that is correct for everyone.

Confirmed two ways: net campaign value at the best threshold is $53,000 against
$52,400 for treat-everyone (a 1% gain, itself optimistic from in-sample threshold
selection), and decision curve analysis shows treat-all dominating the model at
every threshold probability below 21% — the relevant one here being 10.3%.

**This is a process finding, not just a result.** One division, run first, would
have shown that a working model was not the binding constraint on the decision —
before the modelling effort that established the model does not work.

### What would answer the question

A randomised pilot. Sizing it is the useful output:

| Absolute reduction | n per arm | Months to readout |
|---|---|---|
| 15 pp (halving churn) | 121 | **15** |
| 10 pp | 296 | 31 |
| 7 pp | 629 | 63 |
| 5 pp | 1,264 | **124** |

At ~21 signups per month, **only a churn-halving effect is testable at this
company's scale.** A 5pp reduction — an excellent result for a real retention
programme — needs over a decade of signups. That is a fact about running an
experiment on 500 customers, not about the analysis.

---

## Recommended actions, in order

| # | Action | Evidence | Strength |
|---|---|---|---|
| 1 | **Get an export whose timestamps belong to their accounts** | usage/ticket dates at r = 0.002 / 0.014 to their own account; 19,128/24,979 usage rows predate their subscription; 3 churn definitions agree on 20% | strong |
| 2 | **Do _not_ investigate the calendar-time churn increase** | the ×2.8/yr rise sits at the 52nd percentile of a random-date null | strong |
| 3 | **Contact the whole at-risk cohort; do not rank it** | base rate 30.5% vs break-even 10.3% | strong |
| 4 | **Instrument interventions** (CSM touch, discount, campaign) | no treatment variable exists | structural |
| 5 | **Run a randomised pilot**, only for an intervention expected to halve churn | 15pp detectable in ~15 months | moderate |
| 6 | **Do _not_ build a structured onboarding programme** | within-cohort hazard flat, ρ ≈ 1 | strong |

Items 2 and 6 both reverse earlier recommendations in this project, and both
for the same reason. Notebook 05 proposed onboarding on the strength of
`days_since_signup` being the top single feature; notebook 12 showed that was
composition; notebook 16 shows the composition is itself produced by the random
churn date, which also produced the calendar trend that item 2 now withdraws.
One cause, two reversals, both left recorded where they were found.

---

## What each question would need

| Question | Blocker | What unblocks it |
|---|---|---|
| 1. Why | timestamps are not recorded against the accounts they belong to, so there is no real timeline to explain | an export where every event carries its account and its true date — *then* pricing changes, release log, competitor events joined on date |
| 2. Predict | target is a random date; usage and support tables unlinked; label incoherent; 54–81 events | coherent dated label, and event-level telemetry whose timestamps belong to the right accounts |
| 3. Actions | no intervention recorded | log every touch with a timestamp; then a randomised pilot sized as above |

All three are data-collection problems rather than modelling problems, and they
are ordered by leverage: item 1 in the recommendation table unblocks
re-measurement for all three at once. Item 4 is independent of it and can start
immediately — it costs one field in the CRM.
