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

Two design dials decide everything, and they have to be swept separately:

- **Horizon** — how far forward the label looks. "Churn in the next N days."
- **Buffer** — how much lead time the model must give. Standard practice is no
  buffer (score today, act today); a buffer is a refinement for when you cannot
  act immediately, or want to stop the model learning end-of-life signals.

Sweeping both, with the prediction window opening on 2024-06-30 throughout:

| Horizon | Buffer | n | Positives | CV ROC-AUC | Permutation p |
|---:|---:|---:|---:|---:|---:|
| 30 d | 0 | 187 | 25 | 0.402 | 0.72 |
| 30 d | 30 | 168 | 22 | 0.472 | 0.38 |
| 60 d | 0 | 187 | 45 | 0.527 | 0.49 |
| 60 d | 30 | 168 | 39 | 0.489 | 0.45 |
| 90 d | 0 | 187 | 59 | 0.569 | 0.23 |
| 90 d | 30 | 168 | 50 | 0.469 | 0.78 |
| **180 d** | **0** | **187** | **88** | **0.615** | **0.020** |
| 180 d | 30 | 168 | 74 | 0.545 | 0.25 |

**Exactly one cell beats chance**: a 180-day horizon with no lead time at all.
Every operationally normal configuration — 30, 60 or 90 days, which is the usual
range for SaaS churn — is indistinguishable from a coin flip.

Two things follow.

1. **The horizon dial matters as much as the buffer.** An earlier version of
   this analysis swept only the buffer and concluded the signal was purely
   reactive. That was half the picture: nothing works below 180 days regardless
   of lead time.

2. **The one working cell is not a churn model in any useful sense.** At 180
   days the positive rate is 47% and the dominant feature is
   `days_since_signup`. It is predicting "this fairly new account will probably
   be gone sometime in the next six months" — a survivorship base rate, not a
   behavioural early warning. Add 30 days of lead time and even that goes
   (p = 0.25).

**Caveat, stated plainly**: the short-horizon rows are underpowered. At 30 days
there are only 25 positives, so a modest real effect could not be detected. The
fair reading is "no signal at 90 days, where 59 positives give reasonable power"
rather than a confident negative at 30.

The project defaults to a 90-day horizon with a 30-day buffer as the
operationally honest configuration, and reports that it does not work.

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
