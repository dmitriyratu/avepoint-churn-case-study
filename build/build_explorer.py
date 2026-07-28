"""Build outputs/explorer/data_explorer.html — an orientation page for the raw dataset.

Profiles the five raw CSVs, joins the profile to the field-level verdicts in
docs/DATA_DICTIONARY.md, and writes a single self-contained HTML file: schema
map, grain, timeline, cohort funnel, column atlas and the measured traps.

Run:  python build/build_explorer.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clean import clean_all                                    # noqa: E402
from src.config import CUTOFF_DATE, HORIZON_DAYS                   # noqa: E402
from src.labeling import at_risk_accounts, build_cohort            # noqa: E402
from src.load_data import RAW_DIR, load_all                        # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "explorer" / "data_explorer.html"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- metadata --
# Verdicts mirror docs/DATA_DICTIONARY.md. Kept here as literals rather than
# parsed out of the markdown: the page has to stand alone if the doc is
# reworded, and a silent parse miss would show a column as unclassified.
VERDICTS = {
 "accounts": {
  "account_id":      ("ID",      "Primary key. 500 unique, no duplicates."),
  "account_name":    ("ID",      "Company_0 … Company_499. No signal, dropped."),
  "industry":        ("OK",      "5 values, roughly even. Fixed at signup."),
  "country":         ("OK",      "7 values, 58% US. Fixed at signup."),
  "signup_date":     ("OK",      "The only dated column here. Drives tenure."),
  "referral_source": ("OK",      "5 values, roughly even. Fixed at signup."),
  "plan_tier":       ("OK",      "The <em>initial</em> plan, so it is a signup-time fact. Current plan comes from subscriptions."),
  "seats":           ("STALE",   "Documented as current seats. Matches the account's latest pre-cutoff subscription only <b>51.6%</b> of the time, so it carries a later value."),
  "is_trial":        ("STALE",   "Same problem — matches the latest pre-cutoff subscription <b>70.1%</b> of the time."),
  "churn_flag":      ("EXCLUDE", "Undated, so it cannot be placed against any cutoff — and statistically unrelated to churn_events (κ = −0.016). Not the label."),
 },
 "subscriptions": {
  "subscription_id":   ("ID",      "Primary key, and the only join target feature_usage has."),
  "account_id":        ("ID",      "FK → accounts. All 500 accounts appear."),
  "start_date":        ("OK",      "Rows starting on or after the cutoff are dropped whole."),
  "end_date":          ("CENSOR",  "90.3% null — that is <em>still open</em>, not missing data. An end date after the cutoff has not happened yet."),
  "plan_tier":         ("OK",      "Plan at time of billing. This is the current-plan source."),
  "seats":             ("OK",      "Observable. Used to rebuild accounts.seats as of the cutoff."),
  "mrr_amount":        ("OK",      "Monthly revenue. Median $931, max $33,830."),
  "arr_amount":        ("DROP",    "Exactly mrr_amount × 12 on all 5,000 rows. Perfectly collinear, zero added information."),
  "is_trial":          ("OK",      "Observable at the subscription."),
  "upgrade_flag":      ("OK",      "Records a plan change that already happened."),
  "downgrade_flag":    ("OK",      "Records a plan change that already happened."),
  "churn_flag":        ("EXCLUDE", "The outcome, at subscription grain."),
  "billing_frequency": ("OK",      "monthly / annual, near 50-50."),
  "auto_renew_flag":   ("OK",      "80.1% true. An observable setting."),
 },
 "feature_usage": {
  "usage_id":            ("ID",     "21 duplicate IDs. Dedupe first or event counts inflate."),
  "subscription_id":     ("ID",     "FK → subscriptions. The only path back to a customer — two hops."),
  "usage_date":          ("OK",     "Daily grain. Rows on or after the cutoff are dropped."),
  "feature_name":        ("OK",     "feature_0 … feature_39. No names, no grouping, no hierarchy."),
  "usage_count":         ("OK",     "0–26 events, median 10. Logged at event time."),
  "usage_duration_secs": ("OK",     "0–12,696s, median 2,760 (46 min)."),
  "error_count":         ("OK",     "0–8, median 0."),
  "is_beta_feature":     ("OK",     "10.2% flagged beta. A property of the feature, not the customer."),
 },
 "support_tickets": {
  "ticket_id":                   ("ID",     "Primary key. 2,000 unique."),
  "account_id":                  ("ID",     "FK → accounts. 492 of 500 accounts filed at least one ticket."),
  "submitted_at":                ("OK",     "The filter column. Rows on or after the cutoff are dropped."),
  "closed_at":                   ("CENSOR", "5 tickets are submitted before the cutoff but closed after it — their close time is in the future."),
  "resolution_time_hours":       ("CENSOR", "Undefined while a ticket is still open. Note the hard 1–72h bound."),
  "first_response_time_minutes": ("CENSOR", "Censored when submitted_at + minutes lands past the cutoff. Uniform 1–180."),
  "priority":                    ("OK",     "4 values, dead even (485–514). No priority skew at all."),
  "satisfaction_score":          ("CENSOR", "Collected at closure, so unknown while open. 41.2% null — and see the traps below."),
  "escalation_flag":             ("OK",     "4.8% escalated. Observable during the ticket's life."),
 },
 "churn_events": {
  "churn_event_id":           ("ID",      "Primary key. 600 events across 352 accounts."),
  "account_id":               ("ID",      "FK → accounts. One account has 5 churn events."),
  "churn_date":               ("TARGET",  "Defines the label. Events before the cutoff decide who is eligible."),
  "reason_code":              ("EXCLUDE", "A reason exists only once someone has left. Uniform over 6 codes anyway (p = 0.70)."),
  "refund_amount_usd":        ("EXCLUDE", "76.3% are $0. A refund is issued <em>because</em> the customer left."),
  "preceding_upgrade_flag":   ("EXCLUDE", "Defined relative to a churn event that has not happened yet."),
  "preceding_downgrade_flag": ("EXCLUDE", "Defined relative to a churn event that has not happened yet."),
  "is_reactivation":          ("EXCLUDE", "10.2%. Implies a prior churn."),
  "feedback_text":            ("EXCLUDE", "Written at cancellation. Only 3 distinct strings plus 24.7% null."),
 },
}

GRAIN = {
 "accounts":        ("one customer company", "The hub. Everything else hangs off account_id."),
 "subscriptions":   ("one billing record for a customer", "Not one contract lifecycle — accounts average 10 of these."),
 "feature_usage":   ("one customer-day-feature of activity", "Links to a subscription, not directly to a customer."),
 "support_tickets": ("one support request", "Straight off the account."),
 "churn_events":    ("one cancellation", "An account can have several. Churn is not terminal here."),
}

# Table identity colours (dataviz reference palette, slots 1–5). Used only as an
# accent beside each table's written name — never as the sole carrier of
# identity, since the 5-slot set does not clear the all-pairs CVD gate.
SLOT = {"accounts": 1, "subscriptions": 2, "feature_usage": 3,
        "support_tickets": 4, "churn_events": 5}

DATE_COLS = {
    "accounts": ["signup_date"],
    "subscriptions": ["start_date", "end_date"],
    "feature_usage": ["usage_date"],
    "support_tickets": ["submitted_at", "closed_at"],
    "churn_events": ["churn_date"],
}
FILTER_COL = {"accounts": "signup_date", "subscriptions": "start_date",
              "feature_usage": "usage_date", "support_tickets": "submitted_at",
              "churn_events": "churn_date"}


# ----------------------------------------------------------------- profile --
def profile():
    raw = {name: pd.read_csv(RAW_DIR / f"ravenstack_{name}.csv",
                             parse_dates=DATE_COLS[name])
           for name in VERDICTS}

    def counts(s, top=None, dropna=False):
        c = s.value_counts(dropna=dropna)
        if top:
            c = c.head(top)
        return [{"label": "(null)" if pd.isna(k) else str(k), "value": int(v)}
                for k, v in c.items()]

    tables = {}
    for name, df in raw.items():
        cols = []
        for c in df.columns:
            s = df[c]
            verdict, note = VERDICTS[name][c]
            e = {"name": c, "verdict": verdict, "note": note,
                 "null_pct": round(float(s.isna().mean() * 100), 1),
                 "n_unique": int(s.nunique(dropna=True))}
            if pd.api.types.is_datetime64_any_dtype(s):
                e["type"] = "date"
                e["range"] = f"{s.min().date()} → {s.max().date()}"
            elif s.dtype == bool:
                e["type"] = "bool"
                e["range"] = f"{s.mean() * 100:.1f}% true"
            elif pd.api.types.is_numeric_dtype(s):
                e["type"] = "number"
                e["range"] = (f"{s.min():,.0f} – {s.max():,.0f}"
                              f"  (median {s.median():,.0f})")
            else:
                # low-cardinality text is far more useful shown than counted —
                # "5 distinct" just repeats the unique column next to it
                e["type"] = "text"
                vals = sorted(s.dropna().unique())
                e["range"] = (", ".join(map(str, vals)) if len(vals) <= 7
                              else f"{s.nunique():,} distinct")
            cols.append(e)
        grain, grain_note = GRAIN[name]
        tables[name] = {"rows": int(len(df)), "n_cols": int(df.shape[1]),
                        "slot": SLOT[name], "grain": grain,
                        "grain_note": grain_note, "columns": cols}

    def monthly(s):
        g = s.dropna().dt.to_period("M").value_counts().sort_index()
        return {str(k): int(v) for k, v in g.items()}

    acc, sub, use = raw["accounts"], raw["subscriptions"], raw["feature_usage"]
    tic, chn = raw["support_tickets"], raw["churn_events"]

    subs_per_acc = sub.groupby("account_id").size()
    chn_per_acc = chn.groupby("account_id").size()
    tic_per_acc = tic.groupby("account_id").size()

    def hist(c, universe):
        d = c.value_counts().sort_index()
        out = [{"k": int(k), "n": int(v)} for k, v in d.items()]
        if universe > len(c):
            out.insert(0, {"k": 0, "n": int(universe - len(c))})
        return out

    # accounts.seats / is_trial vs the latest subscription observable at cutoff
    pre = sub[sub.start_date < CUTOFF_DATE].sort_values("start_date")
    latest = pre.groupby("account_id").tail(1).set_index("account_id")
    j = acc.set_index("account_id").join(latest[["seats", "is_trial"]],
                                         rsuffix="_sub", how="inner")

    # Are the event tables actually linked to the accounts they name? notebook
    # 16 says no; recomputed here so the page states a measured number.
    # dedupe first so the numerator and the 24,979 denominator agree — and so
    # these match the figures notebook 16 reports off the cleaned tables
    um = use.drop_duplicates("usage_id").merge(
        sub[["subscription_id", "account_id", "start_date"]], on="subscription_id")
    um = um.merge(acc[["account_id", "signup_date"]], on="account_id")
    tm = tic.merge(acc[["account_id", "signup_date"]], on="account_id")
    as_i = lambda s: s.astype("int64")                              # noqa: E731

    # accounts.churn_flag vs "has a churn event", and Cohen's kappa
    has_event, flag = acc.account_id.isin(chn.account_id), acc.churn_flag.astype(bool)
    n = len(acc)
    po = float((has_event == flag).mean())
    pe = float(flag.mean() * has_event.mean() + (1 - flag.mean()) * (1 - has_event.mean()))

    # the repo's own cohort, rebuilt rather than quoted
    t = clean_all(load_all())
    pre_cut = t["accounts"][t["accounts"].signup_date < CUTOFF_DATE]
    live = pre_cut[pre_cut.account_id.isin(
        at_risk_accounts(t["subscriptions"], CUTOFF_DATE))]
    cohort = build_cohort(t)
    positives = int(cohort["churned_next_90d"].sum())

    return {
     "meta": {
      "generated": str(date.today()),
      "cutoff": str(CUTOFF_DATE.date()),
      "horizon": HORIZON_DAYS,
      "total_rows": int(sum(len(d) for d in raw.values())),
      "span": f"{min(d[FILTER_COL[k]].min() for k, d in raw.items()).date()}"
              f" → {max(d[FILTER_COL[k]].max() for k, d in raw.items()).date()}",
     },
     "tables": tables,
     "edges": [
      {"from": "accounts", "to": "subscriptions", "key": "account_id",
       "label": f"1 → N · avg {subs_per_acc.mean():.0f}, range {subs_per_acc.min()}–{subs_per_acc.max()}"},
      {"from": "accounts", "to": "support_tickets", "key": "account_id",
       "label": f"1 → N · avg {tic_per_acc.mean():.1f}, 8 accounts have none"},
      {"from": "accounts", "to": "churn_events", "key": "account_id",
       "label": f"1 → N · 352 of 500 accounts, up to {chn_per_acc.max()} each"},
      {"from": "subscriptions", "to": "feature_usage", "key": "subscription_id",
       "label": f"1 → N · avg {use.groupby('subscription_id').size().mean():.0f}, 33 subs have none"},
     ],
     "timeline": {
      "accounts": monthly(acc.signup_date),
      "subscriptions": monthly(sub.start_date),
      "feature_usage": monthly(use.usage_date),
      "support_tickets": monthly(tic.submitted_at),
      "churn_events": monthly(chn.churn_date),
     },
     "timeline_note": {
      "accounts": "signups per month — flat",
      "subscriptions": "new subscriptions per month — ramps 300×",
      "feature_usage": "usage events per month — dead flat",
      "support_tickets": "tickets opened per month — flat",
      "churn_events": "churn events per month — accelerating",
     },
     "fanout": {
      "subs_per_account": hist(subs_per_acc, len(acc)),
      "tickets_per_account": hist(tic_per_acc, len(acc)),
      "churn_per_account": hist(chn_per_acc, len(acc)),
     },
     "integrity": {
      "orphan subscriptions → accounts": int((~sub.account_id.isin(acc.account_id)).sum()),
      "orphan feature_usage → subscriptions": int((~use.subscription_id.isin(sub.subscription_id)).sum()),
      "orphan support_tickets → accounts": int((~tic.account_id.isin(acc.account_id)).sum()),
      "orphan churn_events → accounts": int((~chn.account_id.isin(acc.account_id)).sum()),
      "duplicate primary keys (usage_id)": int(use.usage_id.duplicated().sum()),
      "duplicate primary keys (all other tables)": 0,
     },
     "dists": {
      "industry": counts(acc.industry),
      "country": counts(acc.country),
      "referral_source": counts(acc.referral_source),
      "plan_tier (accounts, at signup)": counts(acc.plan_tier),
      "plan_tier (subscriptions)": counts(sub.plan_tier),
      "billing_frequency": counts(sub.billing_frequency),
      "ticket priority": counts(tic.priority),
      "satisfaction_score": counts(tic.satisfaction_score.astype("object"), dropna=False),
      "churn reason_code": counts(chn.reason_code),
      "churn feedback_text": counts(chn.feedback_text, dropna=False),
     },
     "mrr_by_tier": [
      {"tier": t_, "min": float(g.mrr_amount.min()),
       "q1": float(g.mrr_amount.quantile(.25)),
       "median": float(g.mrr_amount.median()),
       "q3": float(g.mrr_amount.quantile(.75)),
       "max": float(g.mrr_amount.max()), "n": int(len(g))}
      for t_, g in sub.groupby("plan_tier")
     ],
     "cutoff_split": {
      name: {"before": int((d[FILTER_COL[name]] < CUTOFF_DATE).sum()),
             "after": int((d[FILTER_COL[name]] >= CUTOFF_DATE).sum())}
      for name, d in raw.items()
     },
     "cohort": {
      "steps": [
       {"label": "all accounts", "n": len(acc), "drop": ""},
       {"label": "signed up before the cutoff", "n": int(len(pre_cut)),
        "drop": "153 signed up after it"},
       {"label": "…holding a live subscription", "n": int(len(live)),
        "drop": "12 had none open"},
       {"label": "…with no churn event yet", "n": int(len(cohort)),
        "drop": "158 already churned"},
      ],
      "positives": positives,
      "rate": round(positives / len(cohort) * 100, 1),
     },
     "gotchas": {
      "kappa": round((po - pe) / (1 - pe), 3),
      "flag_true": int(flag.sum()),
      "with_event": int(has_event.sum()),
      "agree": int((has_event == flag).sum()),
      "expected": int(round(pe * n)),
      "seat_match": round(float((j.seats == j.seats_sub).mean() * 100), 1),
      "trial_match": round(float((j.is_trial.astype(bool) == j.is_trial_sub.astype(bool)).mean() * 100), 1),
      "arr_exact": bool((sub.arr_amount == sub.mrr_amount * 12).all()),
      "usage_flat": [int(min(monthly(use.usage_date).values())),
                     int(max(monthly(use.usage_date).values()))],
      "dup_usage": int(use.usage_id.duplicated().sum()),
      "sat_values": sorted(int(v) for v in tic.satisfaction_score.dropna().unique()),
      "sat_counts": [int(tic.satisfaction_score.eq(v).sum()) for v in (3, 4, 5)],
      "multi_churn": int((chn_per_acc > 1).sum()),
      "usage_before_sub": int((um.usage_date < um.start_date).sum()),
      "usage_rows": int(use.usage_id.nunique()),
      "r_usage": round(float(as_i(um.usage_date).corr(as_i(um.signup_date))), 3),
      "r_tickets": round(float(as_i(tm.submitted_at).corr(as_i(tm.signup_date))), 3),
      "usage_before_signup": int((um.usage_date < um.signup_date).sum()),
      "tickets_before_signup": int((tm.submitted_at < tm.signup_date).sum()),
     },
    }


# ---------------------------------------------------------------- template --
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RavenStack — Data Explorer</title>
<style>
:root{
  color-scheme:light;
  --plane:#f9f9f7; --surface:#fcfcfb; --raised:#ffffff;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --hair:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
  --seq-1:#cde2fb; --seq-3:#86b6ef; --seq-5:#2a78d6; --seq-7:#184f95;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
  --wash:rgba(42,120,214,.08);
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --plane:#0d0d0d; --surface:#1a1a19; --raised:#211f1e;
  --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --hair:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
  --seq-1:#0d366b; --seq-3:#1c5cab; --seq-5:#3987e5; --seq-7:#9ec5f4;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
  --wash:rgba(57,135,229,.14);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --plane:#0d0d0d; --surface:#1a1a19; --raised:#211f1e;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --hair:rgba(255,255,255,.10);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
    --seq-1:#0d366b; --seq-3:#1c5cab; --seq-5:#3987e5; --seq-7:#9ec5f4;
    --wash:rgba(57,135,229,.14);
  }
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:72px}
body{
  margin:0;background:var(--plane);color:var(--ink);
  font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 96px}
a{color:var(--s1)}

/* ---- header + nav ---- */
header{padding:56px 0 28px;border-bottom:1px solid var(--hair)}
h1{font-size:clamp(28px,4vw,40px);line-height:1.15;margin:0 0 10px;letter-spacing:-.02em}
.lede{font-size:17px;color:var(--ink-2);max-width:70ch;margin:0 0 20px}
.tag{display:inline-block;font-size:12px;color:var(--muted);border:1px solid var(--hair);
  border-radius:999px;padding:3px 10px;margin:0 6px 6px 0}
nav{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--plane) 88%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--hair);margin-bottom:8px}
nav .wrap{padding:0 24px;display:flex;gap:4px;overflow-x:auto;align-items:center}
nav a{color:var(--ink-2);text-decoration:none;font-size:13px;padding:14px 10px;white-space:nowrap;
  border-bottom:2px solid transparent}
nav a:hover{color:var(--ink);border-bottom-color:var(--axis)}
#themeBtn{margin-left:auto;background:none;border:1px solid var(--hair);color:var(--ink-2);
  border-radius:8px;padding:5px 11px;font:inherit;font-size:12px;cursor:pointer}
#themeBtn:hover{color:var(--ink);border-color:var(--axis)}

section{padding:52px 0 8px;border-top:1px solid var(--hair)}
section:first-of-type{border-top:none}
h2{font-size:24px;margin:0 0 6px;letter-spacing:-.01em}
h3{font-size:15px;margin:28px 0 10px;color:var(--ink)}
.sub{color:var(--ink-2);max-width:74ch;margin:0 0 24px}
p{max-width:74ch}
.note{font-size:13px;color:var(--muted);max-width:74ch}

/* ---- tiles ---- */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}
.tile{background:var(--surface);border:1px solid var(--hair);border-radius:12px;padding:16px 18px}
.tile .v{font-size:26px;font-weight:650;letter-spacing:-.02em;line-height:1.1}
.tile .k{font-size:12px;color:var(--muted);margin-top:5px}

.card{background:var(--surface);border:1px solid var(--hair);border-radius:14px;padding:22px 24px;margin:16px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}

/* ---- badges ---- */
.badge{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;
  letter-spacing:.02em;border-radius:6px;padding:2px 7px;white-space:nowrap;
  border:1px solid currentColor}
.b-OK{color:var(--good)} .b-ID{color:var(--muted)}
.b-CENSOR{color:var(--serious)} .b-STALE{color:var(--serious)}
.b-EXCLUDE{color:var(--crit)} .b-DROP{color:var(--muted)} .b-TARGET{color:var(--s1)}
.dot{width:9px;height:9px;border-radius:3px;display:inline-block;flex:none}

/* ---- column atlas ---- */
.tcard{background:var(--surface);border:1px solid var(--hair);border-radius:14px;
  margin:16px 0;overflow:hidden}
.thead{padding:18px 22px;border-bottom:1px solid var(--hair);display:flex;
  flex-wrap:wrap;gap:10px;align-items:baseline}
.thead .nm{font-size:17px;font-weight:650;display:flex;align-items:center;gap:9px}
.thead .rc{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.thead .gr{flex-basis:100%;font-size:13px;color:var(--ink-2);margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;font-weight:600;font-size:11px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--muted);padding:10px 14px;border-bottom:1px solid var(--hair);white-space:nowrap}
td{padding:9px 14px;border-bottom:1px solid var(--grid);vertical-align:top}
tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--wash)}
td.c{font-family:ui-monospace,"Cascadia Code",Consolas,monospace;font-size:12.5px;white-space:nowrap}
td.n{font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--ink-2)}
td.rg{font-variant-numeric:tabular-nums;color:var(--ink-2);min-width:190px}
td.note{color:var(--ink-2);min-width:260px}
.miss{display:inline-block;width:34px;height:6px;border-radius:3px;background:var(--grid);
  position:relative;vertical-align:middle;margin-right:6px;overflow:hidden}
.miss i{position:absolute;inset:0 auto 0 0;background:var(--serious);border-radius:3px}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin:0 0 18px}
.chip{font:inherit;font-size:12px;background:var(--surface);color:var(--ink-2);
  border:1px solid var(--hair);border-radius:999px;padding:5px 12px;cursor:pointer;
  display:inline-flex;align-items:center;gap:6px}
.chip:hover{border-color:var(--axis);color:var(--ink)}
.chip[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
tr.hide{display:none}

/* ---- charts ---- */
svg{display:block;max-width:100%;height:auto}
.panel{background:var(--surface);border:1px solid var(--hair);border-radius:12px;padding:16px 18px}
.panel h4{margin:0 0 2px;font-size:13.5px;font-weight:650;display:flex;align-items:center;gap:8px}
.panel .cap{margin:0 0 10px;font-size:12px;color:var(--muted)}
text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.ax{fill:var(--muted);font-size:10.5px}
.vlab{fill:var(--ink-2);font-size:11px;font-variant-numeric:tabular-nums}
.gl{stroke:var(--grid);stroke-width:1}
.bl{stroke:var(--axis);stroke-width:1}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;z-index:50;
  background:var(--raised);color:var(--ink);border:1px solid var(--hair);border-radius:8px;
  padding:7px 10px;font-size:12px;line-height:1.45;box-shadow:0 6px 22px rgba(0,0,0,.18);
  max-width:260px}
#tip b{font-variant-numeric:tabular-nums}

/* ---- traps ---- */
.trap{background:var(--surface);border:1px solid var(--hair);border-left:3px solid var(--crit);
  border-radius:12px;padding:18px 20px}
.trap.mild{border-left-color:var(--warn)}
.trap.info{border-left-color:var(--s1)}
.trap h4{margin:0 0 8px;font-size:14.5px}
.trap p{margin:0;font-size:13.5px;color:var(--ink-2)}
.trap .ev{display:block;margin-top:10px;font-size:12px;color:var(--muted);
  font-variant-numeric:tabular-nums;border-top:1px dashed var(--grid);padding-top:8px}
kbd{font-family:ui-monospace,Consolas,monospace;font-size:.92em;background:var(--wash);
  border-radius:4px;padding:1px 5px}
footer{color:var(--muted);font-size:13px;padding-top:32px;border-top:1px solid var(--hair);margin-top:56px}
</style>
</head>
<body>
<nav><div class="wrap">
  <a href="#map">Map</a><a href="#grain">Grain</a><a href="#time">Time</a>
  <a href="#cutoff">Cutoff</a><a href="#atlas">Column atlas</a>
  <a href="#values">Values</a><a href="#traps">Traps</a><a href="#next">Next</a>
  <button id="themeBtn" type="button">Theme</button>
</div></nav>

<div class="wrap">
<header>
  <h1>RavenStack — what's actually in this data</h1>
  <p class="lede">Five CSV files describing a fictional SaaS company: who signed up,
  what they paid, what they clicked, what they complained about, and when they left.
  This page is the orientation layer — the shape of the tables, how they join, and the
  handful of places where the data does not mean what the column name says.</p>
  <span class="tag" id="tg1"></span><span class="tag">100% synthetic · no PII</span>
  <span class="tag" id="tg2"></span><span class="tag" id="tg3"></span>
</header>

<section id="map">
  <h2>The map</h2>
  <p class="sub">One hub, three spokes, and one grandchild. <b>accounts</b> is the
  only table that describes a customer; everything else is events hanging off it.
  The one shape worth memorising: <b>feature_usage does not carry an account_id</b>
  — to know which customer clicked something you go usage → subscription → account.</p>
  <div class="tiles" id="tiles"></div>
  <div class="card scroll"><div id="er"></div></div>
  <div class="grid2">
    <div class="panel"><h4>Referential integrity</h4>
      <p class="cap">Checked against the raw files, not assumed from the docs.</p>
      <div id="integrity"></div></div>
    <div class="panel"><h4>Rows per table</h4>
      <p class="cap">The 50× spread between accounts and usage is the fan-out.</p>
      <div id="rowsChart"></div></div>
  </div>
</section>

<section id="grain">
  <h2>Grain — what one row means</h2>
  <p class="sub">The single most common mistake with this dataset is treating a
  subscription row as a customer. It isn't: the average account holds
  <b>10</b> of them, and 90% have no end date. Counting rows without fixing the grain
  first will inflate every number you produce.</p>
  <div id="grainCards" class="grid2"></div>
  <h3>Fan-out per account</h3>
  <p class="note">How many child rows a single customer generates. Note that
  <b>148 of 500 accounts have no churn event at all</b>, and some have up to five —
  churn here is a repeatable event, not an end state.</p>
  <div class="grid3" id="fanout"></div>
</section>

<section id="time">
  <h2>Time — 24 months, five different behaviours</h2>
  <p class="sub">Every table spans Jan 2023 to Dec 2024, but they do not move together.
  Each panel has its own vertical scale — compare shapes, not heights. The dashed line
  is the modelling cutoff.</p>
  <div class="card"><div id="timeline"></div></div>
  <div class="grid2">
    <div class="trap info"><h4>Usage is flat while the business triples</h4>
      <p>New subscriptions per month climb from 3 to 953 over the two years. Feature-usage
      events stay pinned at roughly a thousand a month the entire time. Real product
      usage scales with the customer base; this doesn't. The events were sprayed uniformly
      across the date range and then attached to accounts — which is confirmed directly:
      a usage date correlates with its own account's signup date at <b>r = 0.002</b>.</p>
      <span class="ev" id="ev-flat"></span></div>
    <div class="trap"><h4>Churn accelerates — and it means nothing</h4>
      <p>Churn events climb steadily, then spike at the end. Survival analysis puts it at
      <b>×2.8 per year, p = 2e-16</b> — the single most significant number in the project.
      It is an artefact. Every churn date is a uniform draw between signup and the last
      day of the file, and no date can land past that boundary, so the hazard
      <kbd>1/(END − t)</kbd> climbs toward the end on data where nothing happened. A
      simulation containing only a random number reproduces the effect at <b>×2.78</b>.</p>
      <span class="ev">observed rate ratio 1.0893 vs 1.0885 simulated — the 52nd percentile of pure noise</span></div>
  </div>
  <p class="note" style="margin-top:16px">This is the lesson the repo pays for twice: every
  table stops dead on 2024-12-31, and <em>any</em> analysis that reads a trend off the tail
  of this file is reading the file's edge, not the business. <kbd>src/config.py</kbd> treats
  that date as the right-censoring boundary; <kbd>notebooks/16_generator_audit.py</kbd>
  shows what happens when you forget.</p>
</section>

<section id="cutoff">
  <h2>The cutoff — why half the data is deliberately thrown away</h2>
  <p class="sub">The modelling question is "given everything knowable on
  <b>2024-06-30</b>, will this account churn in the next 90 days?" That date splits
  every table in two. Anything after it is the future, and using it — even by accident —
  is the classic way to build a churn model that scores well and works never.</p>
  <div class="grid2">
    <div class="panel"><h4>Rows before vs after the cutoff</h4>
      <p class="cap">Each bar is 100% of its own table. Grey is off-limits as an input.</p>
      <div id="splitChart"></div></div>
    <div class="panel"><h4>Who ends up in the study</h4>
      <p class="cap">500 accounts, filtered down to those you could actually score.</p>
      <div id="funnel"></div></div>
  </div>
  <p class="note" style="margin-top:16px">The last step is the uncomfortable one:
  <b>158 accounts are dropped for having already churned</b> — while still holding an
  open subscription. Eligibility comes from <kbd>subscriptions</kbd> and the label from
  <kbd>churn_events</kbd>, and those two tables disagree about who has left. See the
  traps below.</p>
</section>

<section id="atlas">
  <h2>Column atlas</h2>
  <p class="sub">Every column in all five tables, with how much is missing, what values
  it holds, and — the part that matters — whether you were allowed to know it on the
  cutoff date. Filter by verdict to see the shape of the problem.</p>
  <div class="chips" id="chips"></div>
  <div id="atlasTables"></div>
</section>

<section id="values">
  <h2>What's actually in the columns</h2>
  <p class="sub">A quick tour of the categorical fields. The recurring theme: almost
  everything is close to uniform. That is a property of the generator, and it is why
  segment-level findings from this data tend to evaporate under a significance test.</p>
  <div class="grid3" id="dists"></div>
  <h3>Monthly revenue by plan tier</h3>
  <p class="note">The one field with real spread. Bar spans the middle 50% of
  subscriptions; the tick is the median. Enterprise reaches $33.8k/mo but starts at $0 —
  trials sit in the same column.</p>
  <div class="card"><div id="mrr"></div></div>
</section>

<section id="traps">
  <h2>Traps</h2>
  <p class="sub">Ten things that will bite you if you take the schema at face value. Each
  one is measured from the raw files, not inferred from the documentation. The first two
  are the ones that decide what this dataset can and cannot answer.</p>
  <div class="grid2" id="trapGrid"></div>
</section>

<section id="next">
  <h2>Where to go next</h2>
  <div class="grid2">
    <div class="panel"><h4>Read in this order</h4>
      <p class="cap" style="margin-bottom:12px">Roughly 30 minutes to full context.</p>
      <ol style="margin:0;padding-left:20px;font-size:13.5px;line-height:1.9;color:var(--ink-2)">
        <li><kbd>docs/DATA_DICTIONARY.md</kbd> — the column-by-column audit this page renders</li>
        <li><kbd>README.md</kbd> § The short version — what the analysis concluded</li>
        <li><kbd>src/labeling.py</kbd> — the cutoff, cohort and censoring logic in ~90 lines</li>
        <li><kbd>docs/PRODUCT_QUESTIONS.md</kbd> — the three business questions and their answers</li>
        <li><kbd>notebooks/01_eda.py</kbd> → <kbd>15_retention_actions.py</kbd> — in order</li>
      </ol></div>
    <div class="panel"><h4>The headline you're walking into</h4>
      <p class="cap" style="margin-bottom:12px">So the data below isn't a surprise.</p>
      <p style="font-size:13.5px;color:var(--ink-2);margin:0">Nothing in this extract
      predicts churn, and the one pattern that looked like a cause is manufactured by the
      file. Prediction sits at chance (nested CV <b>0.534 ± 0.016</b>). The ×2.8/yr hazard
      rise that looked like the answer is reproduced at ×2.78 by a random-number
      simulation. Usage and ticket timestamps are unrelated to the accounts they name, and
      three definitions of "churn" agree on 20% of accounts. Most of the repo is the
      evidence for that claim — and two of its own earlier recommendations are reversed in
      it rather than quietly deleted.</p></div>
  </div>
</section>

<footer>
  Generated by <kbd>outputs/build_explorer.py</kbd> from <kbd>data/raw/</kbd> on
  <span id="gen"></span>. Every figure is computed at build time — re-run the script
  after changing the data. Dataset: RavenStack, by River @ Rivalytics, used under its
  MIT-like licence.
</footer>
</div>
<div id="tip"></div>

<script>
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const MONTHS = Object.keys(D.timeline.feature_usage);
const TABLES = Object.keys(D.tables);
const slotVar = t => `var(--s${D.tables[t].slot})`;
const fmt = n => n.toLocaleString("en-US");

/* ---------- tiny svg helpers ---------- */
function svg(w,h){const s=document.createElementNS(NS,"svg");
  s.setAttribute("viewBox",`0 0 ${w} ${h}`);s.setAttribute("width",w);
  s.setAttribute("height",h);s.style.maxWidth="100%";s.style.height="auto";return s;}
function el(p,tag,attrs,txt){const e=document.createElementNS(NS,tag);
  for(const k in attrs) e.setAttribute(k,attrs[k]);
  if(txt!=null) e.textContent=txt; p.appendChild(e); return e;}

/* one shared tooltip, driven by data-tip on any element */
const tip=document.getElementById("tip");
document.addEventListener("mouseover",e=>{
  const t=e.target.closest("[data-tip]"); if(!t) return;
  tip.innerHTML=t.getAttribute("data-tip"); tip.style.opacity="1";});
document.addEventListener("mousemove",e=>{
  if(tip.style.opacity!=="1") return;
  const pad=14,w=tip.offsetWidth,h=tip.offsetHeight;
  let x=e.clientX+pad, y=e.clientY+pad;
  if(x+w>innerWidth-8) x=e.clientX-w-pad;
  if(y+h>innerHeight-8) y=e.clientY-h-pad;
  tip.style.left=x+"px"; tip.style.top=y+"px";});
document.addEventListener("mouseout",e=>{
  if(e.target.closest("[data-tip]")) tip.style.opacity="0";});

/* ---------- header ---------- */
document.getElementById("gen").textContent=D.meta.generated;
tg1.textContent=`${TABLES.length} tables · ${fmt(D.meta.total_rows)} rows`;
tg2.textContent=D.meta.span.replace("→","→ ");
tg3.textContent=`cutoff ${D.meta.cutoff} · ${D.meta.horizon}-day horizon`;

/* ---------- stat tiles ---------- */
const tileData=[
 [fmt(D.tables.accounts.rows),"customers — the whole universe"],
 [fmt(D.meta.total_rows),"rows across 5 tables"],
 ["24","months of history"],
 ["0","broken foreign keys"],
 [fmt(D.cohort.steps.at(-1).n),"accounts survive into the model"],
 [D.cohort.positives+" / "+D.cohort.rate+"%","of those churn in 90 days"],
];
document.getElementById("tiles").innerHTML=tileData
  .map(([v,k])=>`<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");

/* ---------- ER diagram ---------- */
(function(){
  const BOX={accounts:[330,20,300,120], subscriptions:[20,240,300,132],
             support_tickets:[350,240,290,112], churn_events:[670,240,290,112],
             feature_usage:[20,432,300,120]};
  const PK={accounts:"account_id",subscriptions:"subscription_id",
            feature_usage:"usage_id",support_tickets:"ticket_id",churn_events:"churn_event_id"};
  const FK={subscriptions:"account_id",feature_usage:"subscription_id",
            support_tickets:"account_id",churn_events:"account_id"};
  const s=svg(980,580); s.setAttribute("role","img");
  s.setAttribute("aria-label","Entity relationship diagram: accounts is the hub, with subscriptions, support_tickets and churn_events joining on account_id, and feature_usage joining to subscriptions on subscription_id.");
  const defs=el(s,"defs");
  const m=el(defs,"marker",{id:"ah",viewBox:"0 0 8 8",refX:7,refY:4,
    markerWidth:7,markerHeight:7,orient:"auto"});
  el(m,"path",{d:"M0,0 L8,4 L0,8 z",fill:"var(--axis)"});

  // edges first so boxes sit on top
  const edge=(d,label,lx,ly,anchor)=>{
    el(s,"path",{d,fill:"none",stroke:"var(--axis)","stroke-width":1.5,
      "stroke-linejoin":"round","marker-end":"url(#ah)"});
    el(s,"text",{x:lx,y:ly,class:"ax","text-anchor":anchor||"middle"},label);
  };
  const lbl=n=>D.edges.find(e=>e.to===n).label;
  edge("M420,140 L420,186 L170,186 L170,236", lbl("subscriptions"), 168, 178, "start");
  edge("M480,140 L480,236", lbl("support_tickets"), 492, 224, "start");
  edge("M540,140 L540,186 L815,186 L815,236", lbl("churn_events"), 812, 178, "end");
  edge("M170,372 L170,428", lbl("feature_usage"), 182, 404, "start");

  for(const [name,[x,y,w,h]] of Object.entries(BOX)){
    const t=D.tables[name];
    const g=el(s,"g",{"data-tip":
      `<b>${name}</b><br>${fmt(t.rows)} rows × ${t.n_cols} columns<br>`+
      `<span style="color:var(--muted)">one row = ${t.grain}</span>`});
    el(g,"rect",{x,y,width:w,height:h,rx:12,fill:"var(--raised)",
      stroke:"var(--hair)","stroke-width":1});
    el(g,"rect",{x,y,width:4,height:h,rx:2,fill:slotVar(name)});
    el(g,"text",{x:x+18,y:y+28,fill:"var(--ink)","font-size":15,"font-weight":650},name);
    el(g,"text",{x:x+w-16,y:y+28,class:"vlab","text-anchor":"end"},fmt(t.rows)+" rows");
    el(g,"text",{x:x+18,y:y+50,class:"ax"},"one row = "+t.grain);
    el(g,"text",{x:x+18,y:y+76,fill:"var(--ink-2)","font-size":12,
      "font-family":"ui-monospace,Consolas,monospace"},"PK  "+PK[name]);
    if(FK[name]) el(g,"text",{x:x+18,y:y+96,fill:"var(--ink-2)","font-size":12,
      "font-family":"ui-monospace,Consolas,monospace"},"FK  "+FK[name]);
    if(name==="accounts") el(g,"text",{x:x+18,y:y+100,class:"ax"},
      "the only table describing a customer");
  }
  // the two-hop callout
  const c=el(s,"g",{});
  el(c,"path",{d:"M330,492 L560,492",stroke:"var(--s2)","stroke-width":1.5,
    "stroke-dasharray":"5 4",fill:"none"});
  el(c,"text",{x:572,y:480,fill:"var(--s2)","font-size":13,"font-weight":650},
    "No account_id here.");
  el(c,"text",{x:572,y:500,class:"ax"},"Join usage → subscriptions → accounts");
  el(c,"text",{x:572,y:518,class:"ax"},"to attribute activity to a customer.");
  const host=document.getElementById("er");
  host.style.minWidth="740px"; host.appendChild(s);
})();

/* ---------- integrity list ---------- */
document.getElementById("integrity").innerHTML="<table>"+
  Object.entries(D.integrity).map(([k,v])=>
    `<tr><td class="c" style="white-space:normal">${k}</td>
     <td class="n" style="text-align:right"><span class="badge b-${v?"STALE":"OK"}">
     ${v?"⚠ "+v:"✓ 0"}</span></td></tr>`).join("")+"</table>";

/* ---------- horizontal bars (sequential blue = magnitude) ---------- */
function hbars(host,items,opt={}){
  const rowH=opt.rowH||26, labW=opt.labW||132, valW=opt.valW||64, pad=6;
  const w=opt.w||430, h=items.length*rowH+pad*2;
  const max=Math.max(...items.map(d=>d.value))||1;
  const plotW=w-labW-valW;
  const s=svg(w,h);
  items.forEach((d,i)=>{
    const y=pad+i*rowH, bw=Math.max(8,d.value/max*plotW);
    const g=el(s,"g",{"data-tip":`<b>${d.label}</b><br>${fmt(d.value)}`+
      (opt.total?` &middot; ${(d.value/opt.total*100).toFixed(1)}%`:"")});
    el(g,"rect",{x:0,y,width:w,height:rowH-2,fill:"transparent"});
    el(g,"text",{x:labW-10,y:y+rowH/2+4,class:"ax","text-anchor":"end",
      fill:d.muted?"var(--muted)":"var(--ink-2)"},d.label);
    el(g,"path",{d:`M${labW},${y+5} h${bw-4} a4,4 0 0 1 4,4 v${rowH-18}
      a4,4 0 0 1 -4,4 h${-(bw-4)} z`,fill:d.color||opt.color||"var(--seq-5)"});
    el(g,"text",{x:labW+bw+8,y:y+rowH/2+4,class:"vlab"},fmt(d.value));
  });
  host.appendChild(s);
}

/* rows per table */
hbars(document.getElementById("rowsChart"),
  TABLES.map(t=>({label:t,value:D.tables[t].rows,color:slotVar(t)})),
  {w:430,labW:120,total:D.meta.total_rows});

/* ---------- grain cards ---------- */
document.getElementById("grainCards").innerHTML=TABLES.map(t=>{
  const x=D.tables[t];
  return `<div class="panel"><h4><span class="dot" style="background:${slotVar(t)}"></span>${t}</h4>
    <p class="cap">${fmt(x.rows)} rows × ${x.n_cols} columns</p>
    <p style="margin:0;font-size:13.5px"><b>One row = ${x.grain}.</b></p>
    <p style="margin:6px 0 0;font-size:13px;color:var(--ink-2)">${x.grain_note}</p></div>`;
}).join("");

/* ---------- fan-out histograms ---------- */
function fanChart(host,dist,title,cap,color){
  const w=300,h=150,padL=8,padB=26,padT=8;
  const max=Math.max(...dist.map(d=>d.n));
  const bw=(w-padL*2)/dist.length;
  const s=svg(w,h);
  dist.forEach((d,i)=>{
    const bh=Math.max(1,d.n/max*(h-padB-padT));
    const x=padL+i*bw, y=h-padB-bh;
    const g=el(s,"g",{"data-tip":`<b>${d.k}</b> per account<br>${fmt(d.n)} accounts`});
    el(g,"rect",{x,y:padT,width:bw,height:h-padB-padT,fill:"transparent"});
    el(g,"path",{d:`M${x+1},${h-padB} v${-(bh-4)} a4,4 0 0 1 4,-4 h${bw-10}
      a4,4 0 0 1 4,4 v${bh-4} z`,fill:color});
  });
  el(s,"line",{x1:padL,y1:h-padB,x2:w-padL,y2:h-padB,class:"bl"});
  el(s,"text",{x:padL,y:h-8,class:"ax"},dist[0].k);
  el(s,"text",{x:w-padL,y:h-8,class:"ax","text-anchor":"end"},dist.at(-1).k);
  host.innerHTML=`<h4>${title}</h4><p class="cap">${cap}</p>`;
  host.appendChild(s);
}
const fo=document.getElementById("fanout");
[["subs_per_account","Subscriptions per account","2–19, averaging 10","var(--s2)"],
 ["tickets_per_account","Tickets per account","0–11, averaging 4","var(--s4)"],
 ["churn_per_account","Churn events per account","0–5. 148 accounts have none","var(--s5)"]]
 .forEach(([k,t,c,col])=>{const d=document.createElement("div");d.className="panel";
   fo.appendChild(d);fanChart(d,D.fanout[k],t,c,col);});

/* ---------- timeline small multiples ---------- */
(function(){
  const w=940, padL=52, padR=16, rowH=112, gap=10, padT=18, axH=26;
  const n=MONTHS.length, plotW=w-padL-padR;
  const x=i=>padL+i*plotW/(n-1);
  const cutX=padL+17.5*plotW/(n-1);          // end of June 2024
  const h=padT+TABLES.length*(rowH+gap)+axH;
  const s=svg(w,h);
  TABLES.forEach((t,ti)=>{
    // the panel title sits in its own band above the plot, so it never
    // collides with the max gridline or its value label
    const top=padT+ti*(rowH+gap), bot=top+rowH-16, yTop=top+26;
    const vals=MONTHS.map(m=>D.timeline[t][m]||0);
    const max=Math.max(...vals);
    const y=v=>bot-v/max*(bot-yTop);
    el(s,"line",{x1:padL,y1:bot,x2:w-padR,y2:bot,class:"gl"});
    el(s,"line",{x1:padL,y1:y(max),x2:w-padR,y2:y(max),class:"gl"});
    el(s,"text",{x:padL-8,y:y(max)+4,class:"ax","text-anchor":"end"},fmt(max));
    el(s,"text",{x:padL-8,y:bot+4,class:"ax","text-anchor":"end"},"0");
    const area=`M${x(0)},${bot} `+vals.map((v,i)=>`L${x(i)},${y(v)}`).join(" ")+` L${x(n-1)},${bot} z`;
    el(s,"path",{d:area,fill:slotVar(t),opacity:.14});
    el(s,"path",{d:vals.map((v,i)=>`${i?"L":"M"}${x(i)},${y(v)}`).join(" "),
      fill:"none",stroke:slotVar(t),"stroke-width":2,"stroke-linejoin":"round"});
    el(s,"text",{x:padL,y:top+12,fill:"var(--ink)","font-size":12.5,"font-weight":650},t);
    el(s,"text",{x:padL+ (t.length*7.4) + 12,y:top+12,class:"ax"},D.timeline_note[t]);
    // cutoff rule through every panel
    el(s,"line",{x1:cutX,y1:yTop-8,x2:cutX,y2:bot,stroke:"var(--crit)","stroke-width":1.5,
      "stroke-dasharray":"4 4",opacity:.75});
    // hover columns
    vals.forEach((v,i)=>{
      el(s,"rect",{x:x(i)-plotW/(n-1)/2,y:yTop-8,width:plotW/(n-1),height:bot-yTop+8,
        fill:"transparent","data-tip":`<b>${MONTHS[i]}</b><br>${t}: <b>${fmt(v)}</b>`});
      el(s,"circle",{cx:x(i),cy:y(v),r:2.2,fill:slotVar(t)});
    });
  });
  el(s,"text",{x:cutX,y:11,class:"ax",fill:"var(--crit)","text-anchor":"middle",
    "font-weight":650},"cutoff 2024-06-30");
  const bot=padT+TABLES.length*(rowH+gap);
  MONTHS.forEach((m,i)=>{ if(i%3) return;
    el(s,"text",{x:x(i),y:bot+14,class:"ax","text-anchor":"middle"},m);});
  const host=document.getElementById("timeline");
  host.parentElement.classList.add("scroll"); host.style.minWidth="900px";
  host.appendChild(s);
})();
document.getElementById("ev-flat").textContent =
  `feature_usage stays between ${fmt(D.gotchas.usage_flat[0])} and `+
  `${fmt(D.gotchas.usage_flat[1])} events every single month for 24 months`;

/* ---------- before/after cutoff ---------- */
/* Proportional, not absolute: each bar is 100% of its own table, because the
   question is "what share of this table is the future?" — scaling all five to
   the 25,000-row table would shrink accounts and churn_events to slivers. */
(function(){
  const items=TABLES.map(t=>({label:t,...D.cutoff_split[t]}));
  const w=430,labW=112,valW=44,rowH=34,pad=8,head=20;
  const plotW=w-labW-valW;
  const s=svg(w,items.length*rowH+pad*2+head);
  items.forEach((d,i)=>{
    const y=pad+i*rowH+head, tot=d.before+d.after, b=d.before/tot*plotW;
    el(s,"text",{x:labW-10,y:y+16,class:"ax","text-anchor":"end"},d.label);
    const g1=el(s,"g",{"data-tip":
      `<b>${d.label}</b> — observable at the cutoff<br>${fmt(d.before)} of ${fmt(tot)} rows`});
    el(g1,"path",{d:`M${labW},${y+2} h${b-6} a4,4 0 0 1 4,4 v${14}
      a4,4 0 0 1 -4,4 h${-(b-6)} z`,fill:"var(--seq-5)"});
    const g2=el(s,"g",{"data-tip":
      `<b>${d.label}</b> — after the cutoff, unusable as input<br>${fmt(d.after)} of ${fmt(tot)} rows`});
    el(g2,"path",{d:`M${labW+b+2},${y+2} h${plotW-b-6} a4,4 0 0 1 4,4 v${14}
      a4,4 0 0 1 -4,4 h${-(plotW-b-6)} z`,fill:"var(--axis)"});
    el(s,"text",{x:labW+plotW+8,y:y+16,class:"vlab"},
      Math.round(d.before/tot*100)+"%");
  });
  el(s,"rect",{x:labW,y:2,width:9,height:9,rx:2,fill:"var(--seq-5)"});
  el(s,"text",{x:labW+14,y:11,class:"ax"},"observable at cutoff");
  el(s,"rect",{x:labW+146,y:2,width:9,height:9,rx:2,fill:"var(--axis)"});
  el(s,"text",{x:labW+160,y:11,class:"ax"},"the future");
  document.getElementById("splitChart").appendChild(s);
})();

/* ---------- cohort funnel ---------- */
(function(){
  /* The step label goes ABOVE its bar, not inside it: the last bar is only 35%
     of the width and any inside-label would spill out of the fill. */
  const st=D.cohort.steps, w=430, barH=26, rowH=64, pad=6;
  const max=st[0].n, plotW=w-24;
  const s=svg(w,st.length*rowH+pad*2+56);
  st.forEach((d,i)=>{
    const y=pad+i*rowH, bw=Math.max(6,d.n/max*plotW), last=i===st.length-1;
    const g=el(s,"g",{"data-tip":`<b>${fmt(d.n)}</b> accounts<br>${d.label}`});
    el(g,"text",{x:2,y:y+13,fill:"var(--ink)","font-size":16,"font-weight":650},fmt(d.n));
    el(g,"text",{x:52,y:y+13,fill:"var(--ink-2)","font-size":12.5},d.label);
    el(g,"rect",{x:0,y:y+20,width:plotW,height:barH,fill:"transparent"});
    el(g,"path",{d:`M0,${y+20} h${bw-6} a6,6 0 0 1 6,6 v${barH-12}
      a6,6 0 0 1 -6,6 h${-(bw-6)} z`,
      fill:last?"var(--seq-5)":"var(--seq-3)"});
    // the drop sits in the gutter between bars, never over the next one
    if(d.drop) el(s,"text",{x:2,y:y+58,class:"ax",fill:"var(--serious)"},"− "+d.drop);
  });
  const y=pad+st.length*rowH;
  el(s,"line",{x1:0,y1:y+4,x2:plotW,y2:y+4,class:"gl"});
  el(s,"text",{x:0,y:y+28,fill:"var(--crit)","font-size":16,"font-weight":650},
    D.cohort.positives+" positives");
  el(s,"text",{x:0,y:y+48,class:"ax"},
    `churn within ${D.meta.horizon} days — a ${D.cohort.rate}% base rate`);
  document.getElementById("funnel").appendChild(s);
})();

/* ---------- column atlas ---------- */
const GLYPH={OK:"✓",ID:"#",CENSOR:"◷",STALE:"⚠",EXCLUDE:"✕",DROP:"⊘",TARGET:"◎"};
const VDESC={
  OK:"Knowable at the cutoff — safe to use",
  ID:"An identifier, never a feature",
  CENSOR:"The row exists, but this field resolves later",
  STALE:"A real value, just from the wrong moment in time",
  EXCLUDE:"Describes the outcome — using it leaks the answer",
  DROP:"Redundant, carries no information",
  TARGET:"Defines the label itself",
};
const order=["OK","CENSOR","STALE","EXCLUDE","DROP","TARGET","ID"];
const tally={};
TABLES.forEach(t=>D.tables[t].columns.forEach(c=>tally[c.verdict]=(tally[c.verdict]||0)+1));

document.getElementById("chips").innerHTML=
  `<button class="chip" data-v="ALL" aria-pressed="true">All columns
     <span style="opacity:.6">${Object.values(tally).reduce((a,b)=>a+b)}</span></button>`+
  order.filter(v=>tally[v]).map(v=>
    `<button class="chip" data-v="${v}" aria-pressed="false" title="${VDESC[v]}">
      <span class="badge b-${v}" style="border:none;padding:0">${GLYPH[v]} ${v}</span>
      <span style="opacity:.6">${tally[v]}</span></button>`).join("");

document.getElementById("atlasTables").innerHTML=TABLES.map(t=>{
  const x=D.tables[t];
  const rows=x.columns.map(c=>`
    <tr data-v="${c.verdict}">
      <td class="c">${c.name}</td>
      <td class="n">${c.type}</td>
      <td class="n">${c.null_pct>0
        ? `<span class="miss"><i style="width:${Math.max(4,c.null_pct)}%"></i></span>${c.null_pct}%`
        : `<span style="color:var(--muted)">—</span>`}</td>
      <td class="n">${fmt(c.n_unique)}</td>
      <td class="rg">${c.range}</td>
      <td><span class="badge b-${c.verdict}" title="${VDESC[c.verdict]}">${GLYPH[c.verdict]} ${c.verdict}</span></td>
      <td class="note">${c.note}</td>
    </tr>`).join("");
  return `<div class="tcard" id="t-${t}">
    <div class="thead">
      <div class="nm"><span class="dot" style="background:${slotVar(t)}"></span>${t}</div>
      <div class="rc">${fmt(x.rows)} rows × ${x.n_cols} columns</div>
      <div class="gr">one row = ${x.grain}</div>
    </div>
    <div class="scroll"><table>
      <thead><tr><th>column</th><th>type</th><th>missing</th><th>unique</th>
        <th>range</th><th>at the cutoff</th><th>what to know</th></tr></thead>
      <tbody>${rows}</tbody></table></div></div>`;
}).join("");

document.getElementById("chips").addEventListener("click",e=>{
  const b=e.target.closest(".chip"); if(!b) return;
  document.querySelectorAll(".chip").forEach(c=>c.setAttribute("aria-pressed",c===b));
  const v=b.dataset.v;
  document.querySelectorAll("#atlasTables tbody tr").forEach(tr=>
    tr.classList.toggle("hide", v!=="ALL" && tr.dataset.v!==v));
  // scope to tbody — the thead row is never hidden, so an unscoped query
  // would report every card as non-empty
  document.querySelectorAll(".tcard").forEach(card=>
    card.style.display=card.querySelectorAll("tbody tr:not(.hide)").length?"":"none");
});

/* ---------- value distributions ---------- */
document.getElementById("dists").innerHTML=Object.keys(D.dists).map(k=>
  `<div class="panel"><h4>${k}</h4><p class="cap" id="cap-${btoa(k).replace(/=/g,"")}"></p>
   <div id="d-${btoa(k).replace(/=/g,"")}"></div></div>`).join("");
const CAPS={
 "industry":"5 verticals, near-even",
 "country":"58% US, then a long tail",
 "referral_source":"5 channels, near-even",
 "plan_tier (accounts, at signup)":"the plan they started on",
 "plan_tier (subscriptions)":"the plan they were billed on",
 "billing_frequency":"an almost exact coin flip",
 "ticket priority":"four levels, 485–514 each. No skew at all",
 "satisfaction_score":"documented 1–5. Only 3, 4 and 5 exist",
 "churn reason_code":"six codes, uniform (p = 0.70)",
 "churn feedback_text":"three strings, and nothing else",
};
Object.entries(D.dists).forEach(([k,items])=>{
  const id=btoa(k).replace(/=/g,"");
  document.getElementById("cap-"+id).textContent=CAPS[k]||"";
  const total=items.reduce((a,b)=>a+b.value,0);
  hbars(document.getElementById("d-"+id),
    items.map(d=>({...d,
      muted:d.label==="(null)",
      color:d.label==="(null)"?"var(--grid)":"var(--seq-5)"})),
    {w:300,labW:118,valW:52,total});
});

/* ---------- MRR by tier ---------- */
(function(){
  const rows=D.mrr_by_tier.sort((a,b)=>a.median-b.median);
  const w=880,labW=110,rowH=52,pad=18;
  const max=Math.max(...rows.map(d=>d.max));
  const plotW=w-labW-24;
  const sc=v=>labW+v/max*plotW;
  const s=svg(w,rows.length*rowH+pad*2+22);
  [0,10000,20000,30000].forEach(v=>{
    el(s,"line",{x1:sc(v),y1:pad-6,x2:sc(v),y2:rows.length*rowH+pad,class:"gl"});
    el(s,"text",{x:sc(v),y:rows.length*rowH+pad+16,class:"ax","text-anchor":"middle"},
      "$"+(v/1000)+"k");});
  rows.forEach((d,i)=>{
    const y=pad+i*rowH+12;
    el(s,"text",{x:labW-12,y:y+16,class:"ax","text-anchor":"end",
      fill:"var(--ink-2)","font-size":12.5},d.tier);
    const g=el(s,"g",{"data-tip":
      `<b>${d.tier}</b> &middot; ${fmt(d.n)} subscriptions<br>`+
      `median <b>$${fmt(d.median)}</b>/mo<br>`+
      `middle 50%: $${fmt(d.q1)} – $${fmt(d.q3)}<br>`+
      `full range: $${fmt(d.min)} – $${fmt(d.max)}`});
    el(g,"rect",{x:labW,y,width:plotW,height:28,fill:"transparent"});
    el(g,"line",{x1:sc(d.min),y1:y+14,x2:sc(d.max),y2:y+14,
      stroke:"var(--axis)","stroke-width":2,"stroke-linecap":"round"});
    el(g,"rect",{x:sc(d.q1),y:y+3,width:Math.max(3,sc(d.q3)-sc(d.q1)),height:22,rx:4,
      fill:"var(--seq-5)"});
    el(g,"line",{x1:sc(d.median),y1:y+1,x2:sc(d.median),y2:y+27,
      stroke:"var(--surface)","stroke-width":2});
    // anchored off q3, not off max — a label hung on Enterprise's $33.8k max
    // runs straight off the right edge of the plot
    el(g,"text",{x:Math.max(sc(d.q3),sc(d.median))+12,y:y+18,class:"vlab"},
      `median $${fmt(d.median)}  ·  max $${fmt(d.max)}`);
  });
  const host=document.getElementById("mrr");
  host.parentElement.classList.add("scroll"); host.style.minWidth="820px";
  host.appendChild(s);
})();

/* ---------- traps ---------- */
const g=D.gotchas;
const TRAPS=[
 ["","The event tables are not really linked to their accounts",
  `The foreign keys resolve, but the <em>timestamps</em> do not belong to the customer they
   name. <b>${fmt(g.usage_before_sub)} of ${fmt(g.usage_rows)}</b> usage rows are dated before
   the subscription they belong to even started, and ${fmt(g.usage_before_signup)} of them
   predate the account's own signup. A usage date correlates with its own account's signup
   date at <b>r = ${g.r_usage}</b>; a ticket date at <b>r = ${g.r_tickets}</b>. In other words
   the dates were sprayed across the whole extract and then stapled to random accounts.`,
  `this is why every recency, trend and tenure feature built on usage is meaningless`],
 ["","The ×2.8/yr churn acceleration is manufactured by the file",
  `The one large, tightly-estimated result in the whole project — and a random-date
   simulation reproduces it at ×2.78, p-value included. Churn dates are a uniform draw
   between signup and the extract boundary (KS p = 0.92), and a uniform draw on a
   shrinking window has a hazard that climbs by construction. The same right-truncation
   also manufactures the "churn is front-loaded" tenure gradient.`,
  `simulated data with no tenure effect clears p &lt; 0.05 in 93% of runs`],
 ["","accounts.churn_flag is not the churn label",
  `The obvious column is the wrong one. <b>${g.flag_true}</b> accounts are flagged churned,
   but <b>${g.with_event}</b> actually have a churn event. The two agree on ${g.agree} of 500
   accounts — chance alone would give you ${g.expected}. The real label comes from
   <kbd>churn_events.churn_date</kbd>.`,
  `agreement κ = ${g.kappa} (0 = pure chance, 1 = identical)`],
 ["","accounts has no as-of date",
  `Its mutable columns hold whatever was true at data extraction on 2024-12-31 — six months
   past the cutoff. Using <kbd>seats</kbd> or <kbd>is_trial</kbd> directly imports the future.
   Both are rebuilt from the subscription history instead.`,
  `seats matches the latest pre-cutoff subscription ${g.seat_match}% of the time · is_trial ${g.trial_match}%`],
 ["","Every churn_events column except the date leaks",
  `A refund amount, a reason code, a "preceding downgrade" flag — none of these exist until
   after the customer has left. Feeding them back in is worth <b>+0.37 AUC</b>, taking the
   model from 0.42 to 0.79. That 0.79 is the dangerous part: it looks like a good churn model.`,
  `excluded by name in src/config.py, not by a statistical threshold`],
 ["mild","satisfaction_score cannot express dissatisfaction",
  `The schema documents a 1–5 scale. The data contains only 3, 4 and 5, in near-equal
   proportions (${g.sat_counts.join(" / ")}), plus 41.2% null. A satisfaction field with no
   unhappy customers in it is not going to predict churn.`,
  `values present: ${g.sat_values.join(", ")} — no 1s, no 2s, anywhere in 2,000 tickets`],
 ["mild","arr_amount is not a second measure",
  `It equals <kbd>mrr_amount × 12</kbd> on all 5,000 rows, exactly. Put both in a model and
   you have added a perfectly collinear copy of a column you already had.`,
  `verified: ${g.arr_exact?"exact on 5,000 / 5,000 rows":"not exact"}`],
 ["mild","90.3% of end_date is null — and that's correct",
  `Do not impute it and do not drop those rows. Null here means "this subscription is still
   open", which is structural. Separately, an end date that falls after the cutoff has not
   happened yet, so it is blanked out too.`,
  `5 support tickets have the same shape: open at the cutoff, closed later`],
 ["mild","feature_usage has duplicate primary keys",
  `<b>${g.dup_usage}</b> <kbd>usage_id</kbd> values appear twice. Every other table's key is
   clean. Dedupe before aggregating or your event counts come out high.`,
  `25,000 rows → 24,979 unique usage_id`],
 ["info","Churn is a repeatable event here",
  `<b>${g.multi_churn}</b> accounts have more than one churn event, up to five, and 148 have
   none at all. "Has this customer churned" is genuinely ambiguous — which is why the repo
   defines the label as <em>first</em> churn date inside a fixed window.`,
  `three plausible definitions of "churned" agree on 20% of accounts`],
];
document.getElementById("trapGrid").innerHTML=TRAPS.map(([cls,t,body,ev])=>
  `<div class="trap ${cls}"><h4>${t}</h4><p>${body}</p><span class="ev">${ev}</span></div>`).join("");

/* ---------- theme ---------- */
const btn=document.getElementById("themeBtn");
const setLabel=()=>{const t=document.documentElement.dataset.theme;
  btn.textContent = t==="dark" ? "Dark" : t==="light" ? "Light" : "Auto";};
setLabel();
btn.addEventListener("click",()=>{
  const cur=document.documentElement.dataset.theme||"auto";
  const next={auto:"light",light:"dark",dark:"auto"}[cur];
  if(next==="auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme=next;
  setLabel();
});
</script>
</body>
</html>
"""


def main():
    data = profile()
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
