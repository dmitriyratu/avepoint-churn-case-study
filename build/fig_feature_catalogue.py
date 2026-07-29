"""The 73 features grouped by the transformation that made them.

Grouping by source table says where a feature came from. Grouping by technique
says what was done to it. The classification is checked against the real
feature list at build time, so it cannot drift from the pipeline.

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
BLUE = "#2E5F8A"
W, H = 1660, 750

# (technique, what it is, the exact columns)
FAMILIES = [
    ("State at the cutoff", "latest value, first value, tenure",
     ["latest_mrr", "latest_seats", "latest_plan_tier", "latest_is_trial",
      "billing_freq", "first_mrr", "first_seats", "tenure_days",
      "days_since_signup"]),
    ("Aggregates per customer", "count, sum, mean, max, min, spread",
     ["n_subscriptions", "n_upgrades", "n_downgrades", "n_trial_subs",
      "n_ended_subs", "total_mrr", "max_mrr", "avg_mrr", "mrr_std", "n_tickets",
      "n_escalations", "n_urgent_high", "n_open_tickets", "avg_resolution_hours",
      "max_resolution_hours", "avg_first_response_mins", "avg_satisfaction",
      "min_satisfaction", "total_usage_events", "total_usage_duration_mins",
      "total_errors", "avg_usage_count", "unique_features_used"]),
    ("Trailing windows", "the same count over 30, 60, 90 and 180 days",
     ["usage_last_30d", "usage_last_60d", "usage_last_90d", "usage_last_180d",
      "usage_prior_90d", "tickets_last_30d", "tickets_last_90d",
      "tickets_last_180d"]),
    ("Trend and acceleration", "recent rate over older rate, and a fitted slope",
     ["accel_30d_vs_90d", "accel_30d_vs_180d", "accel_90d_vs_180d",
      "ticket_accel_30d_vs_90d", "usage_momentum", "usage_delta_90d",
      "usage_trend_slope", "recency_ratio_90d"]),
    ("Recency and rhythm", "days since the last event, and the gaps between them",
     ["days_since_last_usage", "days_since_last_ticket",
      "days_since_last_sub_start", "usage_span_days", "mean_gap_days",
      "max_gap_days"]),
    ("Direction of travel", "latest minus first",
     ["seat_growth", "mrr_growth", "mrr_growth_pct", "upgrade_net"]),
    ("Rates and shares", "a count over its own denominator",
     ["auto_renew_pct", "pct_subs_ended", "urgent_pct", "escalation_rate",
      "error_rate", "beta_feature_pct", "sat_missing_rate", "mrr_cv"]),
    ("Per-seat normalisation", "divided by the seat count",
     ["usage_per_seat", "tickets_per_seat", "mrr_per_seat"]),
    ("Account attributes", "encoded inside each fold",
     ["industry", "country", "referral_source", "plan_tier"]),
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
    f'<td class="tech"><b>{name}</b><span class="why">{why}</span></td>'
    f'<td class="ex">{", ".join(cols)}</td></tr>'
    for name, why, cols in FAMILIES)

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; }}
  body {{ background: #fff;
         font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
  .wrap {{ width: {W}px; height: {H}px; padding: 32px 40px; }}

  table {{ width: 100%; border-collapse: collapse; }}
  th {{ font-size: 11px; font-weight: 600; color: {MUTED}; text-align: left;
        text-transform: uppercase; letter-spacing: .6px;
        padding: 0 0 9px 0; border-bottom: 1.5px solid {RULE}; }}
  td {{ padding: 13px 0 13px 0; border-bottom: 1px solid #F0F2F4;
        vertical-align: baseline; }}
  td.n {{ font-size: 21px; font-weight: 650; color: {BLUE}; width: 52px;
          text-align: right; padding-right: 20px;
          font-variant-numeric: tabular-nums; }}
  td.tech {{ width: 360px; padding-right: 40px; }}
  td b {{ font-size: 15px; color: {INK}; font-weight: 600; }}
  .why {{ font-size: 13px; color: {MUTED}; display: block; margin-top: 3px; }}
  td.ex {{ font-size: 11.5px; color: {SOURCE}; font-family: "Consolas",
           monospace; line-height: 1.75; }}
</style>
<div class="wrap">
  <table>
    <tr><th style="text-align:right;padding-right:22px">n</th>
        <th>Technique</th><th>Columns</th></tr>
    {rows}
  </table>
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
