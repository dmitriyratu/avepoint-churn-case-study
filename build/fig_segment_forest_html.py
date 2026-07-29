"""No segment predicts churn — the exploration figure, rendered through HTML/CSS.

Bars of churn rate invited the comparison the slide argues against: a viewer saw
Cybersecurity at 9% against DevTools at 38% and read a threefold effect, when it
is two leavers out of twenty-three. Showing the count beside every interval
makes that impossible to miss, and the interval shows how little a group that
size can settle.

Text here is kept to labels. Anything that explains or argues belongs in the
slide beside it.

Run from the repo root:  python build/fig_segment_forest_html.py
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline  # noqa: E402
from src.config import CUTOFF_DATE, TARGET  # noqa: E402

VARIABLES = [
    ("industry", "Industry"),
    ("referral_source", "How they found us"),
    ("plan_tier", "Plan"),
    ("country", "Country"),
    ("is_trial", "Started on a trial"),
]
XMAX = 0.70

cohort = pipeline.build().cohort
base = cohort[TARGET].mean()


def pct(value):
    return min(value, XMAX) / XMAX * 100


n_total = len(cohort)
n_left = int(cohort[TARGET].sum())
total_lo, total_hi = stats.binomtest(n_left, n_total).proportion_ci(
    0.95, method="wilson")

# The base rate gets a row of its own. Quoting it only as "31%" left readers
# guessing at the denominator, and it has an interval like everything else.
blocks = f"""
<div class="block">
  <div class="head"><span>Everyone in the cohort</span><em>at {CUTOFF_DATE.day} {CUTOFF_DATE:%B %Y}, one cutoff</em></div>
  <div class="row total">
    <span class="name">all customers</span>
    <span class="track">
      <i class="base" style="left:{pct(base):.2f}%"></i>
      <i class="ci" style="left:{pct(total_lo):.2f}%;width:{pct(total_hi) - pct(total_lo):.2f}%"></i>
      <i class="dot" style="left:{pct(base):.2f}%"></i>
    </span>
    <span class="count"><b>{n_left}</b> of {n_total}</span>
  </div>
</div>"""

for column, label in VARIABLES:
    table = pd.crosstab(cohort[column], cohort[TARGET])
    p = stats.chi2_contingency(table)[1]
    grouped = cohort.groupby(column)[TARGET].agg(["mean", "size"])
    grouped = grouped.sort_values("mean", ascending=False)

    rows = ""
    for name, (rate, n) in grouped.iterrows():
        n, left = int(n), int(round(rate * n))
        # Wilson: these groups are small and some hold zero leavers, which is
        # where the textbook interval returns impossible values.
        lo, hi = stats.binomtest(left, n).proportion_ci(0.95, method="wilson")
        rows += f"""
        <div class="row">
          <span class="name">{name}</span>
          <span class="track">
            <i class="base" style="left:{pct(base):.2f}%"></i>
            <i class="ci" style="left:{pct(lo):.2f}%;width:{pct(hi) - pct(lo):.2f}%"></i>
            <i class="dot" style="left:{pct(rate):.2f}%"></i>
          </span>
          <span class="count"><b>{left}</b> of {n}</span>
        </div>"""

    blocks += f"""
    <div class="block">
      <div class="head"><span>{label}</span><em>p = {p:.2f}</em></div>
      {rows}
    </div>"""

ticks = "".join(f'<span style="left:{pct(t):.2f}%">{t:.0%}</span>'
                for t in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 940px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ padding: 20px 24px 22px; }}

  .block {{ margin-bottom: 15px; }}
  .head {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 5px;
           padding-bottom: 4px; border-bottom: 1px solid #E4E7EB; }}
  .head span {{ font-size: 14.5px; font-weight: 700; }}
  .head em {{ font-size: 12.5px; font-style: normal; color: #5A6270; }}

  .row {{ display: flex; align-items: center; height: 25px; }}
  .name {{ flex: 0 0 132px; text-align: right; padding-right: 15px;
           font-size: 13.5px; }}
  .track {{ flex: 1 1 auto; min-width: 0; position: relative; height: 25px; }}
  .track i {{ position: absolute; display: block; }}
  .base {{ top: 0; bottom: 0; width: 2px; background: #B02E2E;
           transform: translateX(-1px); }}
  .ci {{ top: 50%; height: 3px; margin-top: -1.5px; background: #2a78d6;
         opacity: .45; border-radius: 2px; }}
  .dot {{ top: 50%; width: 10px; height: 10px; margin: -5px 0 0 -5px;
          border-radius: 50%; background: #2a78d6; }}
  .count {{ flex: 0 0 84px; padding-left: 18px; font-size: 13px; color: #5A6270; }}
  .count b {{ color: #1A1A1A; font-weight: 700; }}
  .total .name, .total .count {{ color: #B02E2E; font-weight: 700; }}
  .total .count b {{ color: #B02E2E; }}
  .total .ci {{ background: #B02E2E; opacity: .38; }}
  .total .dot {{ background: #B02E2E; }}

  .axis {{ position: relative; height: 19px; margin: 4px 84px 0 132px; }}
  .axis span {{ position: absolute; transform: translateX(-50%); font-size: 12px;
                color: #5A6270; }}
  .cap {{ text-align: center; margin: 5px 84px 0 132px; font-size: 12.5px;
          color: #5A6270; }}
</style>
<div class="wrap">
  {blocks}
  <div class="axis">{ticks}</div>
  <div class="cap">share of the group that left within 90 days &nbsp;·&nbsp;
    <span style="color:#B02E2E">red line = the {base:.0%} everyone rate</span></div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "05_segment_forest.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "05_segment_forest_html.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 940, "height": 900},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".wrap").screenshot(path=str(out))
    browser.close()

print("wrote", out)
