"""The 73 features grouped by the transformation that made them.

Grouping by source table answers "where did this come from". Grouping by
technique answers "what did you actually do", which is the question that gets
asked. The classification is checked against the real feature list at build
time, so it cannot drift from the pipeline.

Run from the repo root:  python build/fig_feature_catalogue.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline  # noqa: E402

OUT = ROOT / "outputs" / "figures" / "22_feature_catalogue.png"
PAGE = ROOT / "outputs" / "figures" / "22_feature_catalogue.html"

INK, MUTED, RULE = "#1A1A1A", "#5A6270", "#D8DCE0"
SOURCE = "#98A0AA"
BLUE, RED, GREEN = "#2E5F8A", "#B02E2E", "#1E7A4B"
W, H = 1800, 900

# (technique, what it is for, the exact columns)
FAMILIES = [
    ("State at the cutoff",
     "what the customer looked like on 30 June, and where they started",
     ["latest_mrr", "latest_seats", "latest_plan_tier", "latest_is_trial",
      "billing_freq", "first_mrr", "first_seats", "tenure_days",
      "days_since_signup"]),
    ("Aggregates per customer",
     "count, sum, mean, max, min and spread over every row they own",
     ["n_subscriptions", "n_upgrades", "n_downgrades", "n_trial_subs",
      "n_ended_subs", "total_mrr", "max_mrr", "avg_mrr", "mrr_std", "n_tickets",
      "n_escalations", "n_urgent_high", "n_open_tickets", "avg_resolution_hours",
      "max_resolution_hours", "avg_first_response_mins", "avg_satisfaction",
      "min_satisfaction", "total_usage_events", "total_usage_duration_mins",
      "total_errors", "avg_usage_count", "unique_features_used"]),
    ("Trailing windows",
     "the same count over the last 30, 60, 90 and 180 days",
     ["usage_last_30d", "usage_last_60d", "usage_last_90d", "usage_last_180d",
      "usage_prior_90d", "tickets_last_30d", "tickets_last_90d",
      "tickets_last_180d"]),
    ("Trend and acceleration",
     "a short window against a long one, length-normalised, plus a fitted slope",
     ["accel_30d_vs_90d", "accel_30d_vs_180d", "accel_90d_vs_180d",
      "ticket_accel_30d_vs_90d", "usage_momentum", "usage_delta_90d",
      "usage_trend_slope", "recency_ratio_90d"]),
    ("Recency and rhythm",
     "how long since the last event, and how evenly events are spaced",
     ["days_since_last_usage", "days_since_last_ticket",
      "days_since_last_sub_start", "usage_span_days", "mean_gap_days",
      "max_gap_days"]),
    ("Direction of travel",
     "latest minus first, so growth is separated from size",
     ["seat_growth", "mrr_growth", "mrr_growth_pct", "upgrade_net"]),
    ("Rates and shares",
     "a count divided by its own denominator, so it is comparable across customers",
     ["auto_renew_pct", "pct_subs_ended", "urgent_pct", "escalation_rate",
      "error_rate", "beta_feature_pct", "sat_missing_rate", "mrr_cv"]),
    ("Per-seat normalisation",
     "so a 10-seat and a 500-seat customer sit on the same scale",
     ["usage_per_seat", "tickets_per_seat", "mrr_per_seat"]),
    ("Account attributes",
     "encoded inside each fold, never on the full dataset",
     ["industry", "country", "referral_source", "plan_tier"]),
]

DECISIONS = [
    ("Missing means something different per family", GREEN,
     "A count of zero is a real zero. A recency of never is maximally stale, "
     "not today, so it is filled with the length of the window. A rate that is "
     "unknown stays blank and is imputed inside the fold, never before it."),
    ("Per-seat uses the latest pre-cutoff contract", RED,
     "accounts.seats is current as of extraction, so dividing by it would leak "
     "a later state into a June feature. The denominator is the seat count on "
     "the newest contract that had already started."),
    ("Near-duplicates are dropped, not kept", GREEN,
     "Any pair correlated above 0.98 loses its second member — six went, "
     "including feature_breadth against unique_features_used. Keeping both "
     "splits one effect across two coefficients."),
    ("Outcomes not yet known are blanked", RED,
     "A ticket still open on 30 June has no resolution time and no satisfaction "
     "score, because nobody knew them that day. An automatic check found the "
     "five this applied to; reading the code did not."),
    ("One function, two callers", GREEN,
     "Training passes 30 June 2024 and production passes today. The same code "
     "builds both, so training and serving cannot drift apart."),
]


def verify():
    """The catalogue must account for every shipped feature, exactly once."""
    actual = set(pipeline.build(prune=True).X.columns)
    listed = [c for _, _, cols in FAMILIES for c in cols]
    assert len(listed) == len(set(listed)), "a feature is listed twice"
    missing, extra = actual - set(listed), set(listed) - actual
    assert not missing, f"not in the catalogue: {sorted(missing)}"
    assert not extra, f"catalogue lists features that do not exist: {sorted(extra)}"
    return len(actual)


TOTAL = verify()

rows = "".join(
    f'<tr><td class="n">{len(cols)}</td>'
    f'<td><b>{name}</b><span class="why">{why}</span></td>'
    f'<td class="ex">{", ".join(cols[:3])}'
    f'{f" +{len(cols) - 3} more" if len(cols) > 3 else ""}</td></tr>'
    for name, why, cols in FAMILIES)

cards = "".join(
    f'<div class="rule" style="--c:{colour}">'
    f'<b>{title}</b><p>{body}</p></div>'
    for title, colour, body in DECISIONS)

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; }}
  body {{ background: #fff;
         font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
  .wrap {{ width: {W}px; height: {H}px; display: grid;
           grid-template-columns: 1000px 1fr; gap: 0 56px; padding: 34px 44px; }}

  h3 {{ font-size: 19px; font-weight: 650; color: {INK}; margin-bottom: 3px; }}
  .sub {{ font-size: 13.5px; color: {MUTED}; margin-bottom: 16px; }}

  table {{ width: 100%; border-collapse: collapse; }}
  th {{ font-size: 11.5px; font-weight: 600; color: {MUTED}; text-align: left;
        text-transform: uppercase; letter-spacing: .55px;
        padding: 0 10px 7px 0; border-bottom: 1.5px solid {RULE}; }}
  td {{ padding: 9px 10px 9px 0; border-bottom: 1px solid #F0F2F4;
        vertical-align: top; }}
  td.n {{ font-size: 20px; font-weight: 650; color: {BLUE}; width: 46px;
          text-align: right; padding-right: 18px;
          font-variant-numeric: tabular-nums; }}
  td b {{ font-size: 14.5px; color: {INK}; font-weight: 600; display: block; }}
  .why {{ font-size: 12.5px; color: {MUTED}; display: block; margin-top: 1px; }}
  td.ex {{ font-size: 12px; color: {SOURCE}; font-family: "Consolas",
           monospace; width: 330px; line-height: 1.5; }}

  .rule {{ border-left: 3px solid var(--c); padding: 0 0 0 14px;
           margin-bottom: 15px; }}
  .rule b {{ font-size: 14px; color: var(--c); font-weight: 650; }}
  .rule p {{ font-size: 12.5px; color: {MUTED}; line-height: 1.55;
             margin-top: 3px; }}
</style>
<div class="wrap">
  <div>
    <h3>{TOTAL} features, grouped by what was done to make them</h3>
    <div class="sub">Not by which table they came from — by the transformation.
      Every shipped feature appears exactly once.</div>
    <table>
      <tr><th style="text-align:right;padding-right:18px">n</th>
          <th>Technique</th><th>Examples</th></tr>
      {rows}
    </table>
  </div>
  <div>
    <h3>The decisions that are easy to get wrong</h3>
    <div class="sub">Each of these changes the score, and none is visible in a
      feature list.</div>
    {cards}
  </div>
</div>
"""

PAGE.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": W, "height": H},
                            device_scale_factor=2)
    page.goto(PAGE.as_uri())
    page.locator(".wrap").screenshot(path=str(OUT))
    browser.close()

print("wrote", OUT)
print(f"  {TOTAL} features across {len(FAMILIES)} techniques, all accounted for")
for name, _, cols in FAMILIES:
    print(f"    {len(cols):3d}  {name}")
