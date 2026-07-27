# Churn Feature Engineering — Practice and Findings

Researched practice, the taxonomy applied here, and — most importantly — what
happened when the design was made operationally honest.

Sources:
- [A Rolling-Window Framework for Churn Prediction and Behavioral Driver Identification (arXiv:2606.06776)](https://arxiv.org/pdf/2606.06776)
- [Growth-onomics — Behavioral feature engineering for churn](https://growth-onomics.com/ultimate-guide-to-behavioral-feature-engineering-for-churn/)
- [Growth-onomics — Build time-series churn models](https://growth-onomics.com/build-time-series-churn-models/)
- [The Dao of Data — Churn modelling part 2: window selection](https://tollyvellis.com/churn-modelling-part-2-window-selection/)
- [Nature Sci. Reports — Hybrid deep learning churn prediction using RFM](https://www.nature.com/articles/s41598-026-53220-0)

---

## The headline finding

Standard practice inserts a **buffer** (also called a latency or implementation
window) between the last observable day and the first day a churn can count
against the model. The reason is blunt:

> Using data from months 1–3 to predict churn in month 4 may look reasonable but
> is often misleading. The model learns to detect signals that occur immediately
> before a customer leaves — a sudden drop in usage or payment activity. This
> produces high apparent accuracy but no business value: by the time the model
> flags the customer, they are already gone.

This project had **no buffer**. Adding one and sweeping its length:

| Buffer | Eligible | Positives | CV ROC-AUC | 95% CI | Permutation p |
|---:|---:|---:|---:|---|---:|
| 0 days | 187 | 88 | **0.616** | [0.50, 0.74] | **0.025** |
| 15 days | 176 | 80 | 0.574 | [0.40, 0.75] | 0.254 |
| 30 days | 168 | 74 | 0.548 | [0.38, 0.70] | 0.249 |
| 60 days | 154 | 67 | 0.545 | [0.35, 0.71] | 0.209 |
| 90 days | 139 | 57 | 0.467 | [0.27, 0.59] | 0.960 |

**The signal is almost entirely reactive.** It survives only when the model is
allowed to see behaviour right up to the moment churn becomes possible. Give a
CSM even two weeks of lead time and the model is no longer distinguishable from
chance.

This is the single most important result in the project, and it inverts the
earlier conclusion. The honest statement is not "we built a weak churn model."
It is: **on this dataset there is no actionable churn signal at a realistic
intervention horizon.** A model shipped without a buffer would have looked
defensible in cross-validation and been useless in production.

The project therefore defaults to `BUFFER_DAYS = 30` (`src/config.py`) — the
configuration that answers the question the business actually has.

---

## Feature taxonomy applied

### RFM, the standard spine

| Dimension | Implemented as |
|---|---|
| **Recency** | `days_since_last_usage`, `days_since_last_ticket`, `days_since_last_sub_start`, `usage_span_days` |
| **Frequency** | `usage_last_{30,60,90,180}d`, `active_days_last_{...}d`, `n_tickets`, `tickets_last_{30,90,180}d`, `n_subscriptions` |
| **Monetary** | `total_mrr`, `avg_mrr`, `max_mrr`, `latest_mrr`, `mrr_per_seat`, `mrr_growth_pct`, `mrr_cv` |

### Window ladder rather than lifetime totals

The literature is consistent that lifetime aggregates hide the thing you care
about: an account can have huge total usage and have stopped last month. Every
engagement metric is therefore computed over 30/60/90/180-day windows.

### Acceleration — the differences between windows

Direction matters more than level. Rates are length-normalised so short and long
windows compare fairly:

- `accel_30d_vs_90d`, `accel_30d_vs_180d`, `accel_90d_vs_180d` — above 1 means
  the account is more active lately than its own baseline
- `usage_momentum` — last 90 days against the *preceding* 90, non-overlapping
- `usage_delta_90d` — the raw difference
- `ticket_accel_30d_vs_90d` — rising support load is a classic precursor

### Trend slope

Ratios are coarse and miss a steady grind downward. `usage_trend_slope` fits a
line to weekly event counts over the observation window.

### Regularity

Two accounts with identical volume but different rhythms are different risks.
`mean_gap_days` and `max_gap_days` measure spacing between active days.

### Expansion / contraction

`seat_growth`, `mrr_growth`, `mrr_growth_pct`, `upgrade_net`, `pct_subs_ended`.

### Behavioural before demographic

The literature reports behavioural features outperforming demographics by 5–10
AUROC points. Both are included here, and the L1 penalty decides.

---

## What the enriched features actually bought

Nothing measurable.

| Configuration | CV ROC-AUC |
|---|---:|
| 30-day buffer, original feature set | 0.551 |
| 30-day buffer, enriched feature set (window ladder, acceleration, slope, gaps, volatility) | **0.548** |

Adding roughly twenty engineered features moved the score slightly *down*. With
74 positives, extra columns cost more in variance than they return in signal, and
L1 shrinks the model to three terms regardless.

Reporting this rather than quietly keeping the richer set is the point. The
constraint here is **data, not feature engineering**:

1. Only 168 eligible accounts and 74 positives.
2. Synthetic usage logs — 19,128 of 24,979 rows predate their own subscription's
   start, so trend and recency features are built on incoherent timestamps.
3. `churn_flag` disagrees with the event log for 62% of accounts.

No amount of feature work fixes any of those.

---

## What I would build with real telemetry

The engineering above is the right shape; it needs inputs that carry signal.

1. **Event-level product telemetry** with trustworthy timestamps — sessions,
   depth of use, seat-level activation rather than account-level totals.
2. **Seat utilisation**, the strongest B2B SaaS churn predictor in practice:
   licences paid for versus licences actually active.
3. **Sequence features** — order and timing of actions, not just counts.
4. **Billing events** — failed payments, invoice disputes, renewal proximity.
5. **Relationship signals** — champion departure, exec sponsor engagement,
   QBR attendance.
6. **Survival framing** — a discrete-time hazard model uses *when* rather than
   only *whether*, and handles the censoring this data is full of.

---

## Checklist

| # | Practice | Status |
|---|---|---|
| 1 | Buffer between observation and prediction windows | Implemented, `BUFFER_DAYS = 30` |
| 2 | Buffer sensitivity reported, not assumed | `outputs/reports/buffer_sensitivity.csv` |
| 3 | RFM spine | Implemented |
| 4 | Multi-window ladder rather than lifetime totals | 30/60/90/180 |
| 5 | Acceleration ratios between windows | Implemented |
| 6 | Fitted trend slope | Implemented |
| 7 | Regularity / inter-event gaps | Implemented |
| 8 | Expansion & contraction | Implemented |
| 9 | Support-load trend | Implemented |
| 10 | Every feature cutoff-parameterised, audit-gated | `build_model_dataset(as_of=...)` |
| 11 | Zero-variance and collinear columns pruned | Automatic |
| 12 | Feature value measured, not assumed | Measured — it was zero |
