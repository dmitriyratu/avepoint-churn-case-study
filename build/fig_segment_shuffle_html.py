"""What chance produces — the shuffle test for the exploration slide.

Confidence intervals show precision but do not perform the test, and reading
significance off whether they overlap a line is a heuristic that disagrees with
the real test. This does the test directly and with no formula: scramble who
left, recompute the widest gap between groups, repeat twenty thousand times.
Where the real gap falls in that pile is the answer.

The point of the slide is the last row. Industry on its own lands at p = 0.03
and looks like a finding. Once the null accounts for having asked five
questions rather than one, it is p = 0.14 and it is not.

Run from the repo root:  python build/fig_segment_shuffle_html.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline  # noqa: E402
from src.config import CUTOFF_DATE, TARGET  # noqa: E402

VARIABLES = [
    ("industry", "Industry"),
    ("referral_source", "How they found us"),
    ("country", "Country"),
    ("plan_tier", "Plan"),
    ("is_trial", "Started on a trial"),
]
SHUFFLES = 20_000
XMAX, BINS = 0.80, 56
SEED = 0

cohort = pipeline.build().cohort
left = cohort[TARGET].to_numpy()

codes, sizes = [], []
for column, _ in VARIABLES:
    c = cohort[column].astype("category").cat.codes.to_numpy()
    codes.append(c)
    sizes.append(np.bincount(c, minlength=c.max() + 1))


def widest_gap(labels):
    """Biggest difference in churn rate between any two groups, per variable."""
    out = []
    for c, n in zip(codes, sizes):
        rate = np.bincount(c, weights=labels, minlength=len(n)) / n
        out.append(rate.max() - rate.min())
    return np.array(out)


rng = np.random.default_rng(SEED)
observed = widest_gap(left)
null = np.array([widest_gap(rng.permutation(left)) for _ in range(SHUFFLES)])
p_each = np.array([(null[:, i] >= observed[i]).mean() for i in range(len(VARIABLES))])

# Calibrate "the smallest of five" against the same shuffles: for every shuffle,
# score all five variables and keep its best. That is the null for the question
# actually asked, which was not "does industry matter" but "does anything".
p_null = np.empty_like(null)
for i in range(len(VARIABLES)):
    ordered = np.sort(null[:, i])
    p_null[:, i] = 1 - np.searchsorted(ordered, null[:, i], side="left") / SHUFFLES
family_p = (p_null.min(axis=1) <= p_each.min()).mean()

edges = np.linspace(0, XMAX, BINS + 1)
rows = ""
for i, (_, label) in enumerate(VARIABLES):
    counts, _ = np.histogram(np.clip(null[:, i], 0, XMAX), bins=edges)
    tallest = counts.max()
    bars = "".join(
        f'<i style="height:{c / tallest * 100:.1f}%"></i>' for c in counts)
    verdict = "looks real" if p_each[i] < 0.05 else "nothing"
    rows += f"""
    <div class="row">
      <span class="name">{label}</span>
      <span class="plot">
        <span class="bars">{bars}</span>
        <i class="obs" style="left:{observed[i] / XMAX * 100:.2f}%"></i>
        <em class="lab" style="left:{observed[i] / XMAX * 100:.2f}%">{observed[i]:.0%}</em>
      </span>
      <span class="p {'hot' if p_each[i] < 0.05 else ''}">p = {p_each[i]:.2f}
        <b>{verdict}</b></span>
    </div>"""

ticks = "".join(f'<span style="left:{t / XMAX * 100:.2f}%">{t:.0%}</span>'
                for t in (0, 0.2, 0.4, 0.6, 0.8))

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1000px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ padding: 20px 24px 22px; }}
  .muted {{ color: #5A6270; }}

  .row {{ display: flex; align-items: flex-end; height: 74px; margin-bottom: 9px; }}
  .name {{ flex: 0 0 150px; text-align: right; padding: 0 16px 21px 0;
           font-size: 14px; font-weight: 700; }}
  .plot {{ flex: 1 1 auto; min-width: 0; position: relative; height: 66px;
           border-bottom: 1px solid #D8DCE0; }}
  .bars {{ position: absolute; inset: 0; display: flex; align-items: flex-end;
           gap: 1px; }}
  .bars i {{ flex: 1 1 0; background: #C3D4EA; display: block; }}
  .obs {{ position: absolute; top: -4px; bottom: 0; width: 2px;
          background: #B02E2E; transform: translateX(-1px); }}
  .lab {{ position: absolute; top: -20px; transform: translateX(-50%);
          font-size: 12.5px; font-style: normal; font-weight: 700;
          color: #B02E2E; white-space: nowrap; }}
  .p {{ flex: 0 0 132px; padding-left: 20px; font-size: 13.5px; color: #5A6270;
        padding-bottom: 2px; }}
  .p b {{ display: block; font-size: 12.5px; font-weight: 400; }}
  .p.hot {{ color: #B02E2E; font-weight: 700; }}

  .axis {{ position: relative; height: 18px; margin: 2px 132px 0 166px; }}
  .axis span {{ position: absolute; transform: translateX(-50%); font-size: 12px;
                color: #5A6270; }}
  .cap {{ margin: 4px 132px 0 166px; font-size: 12.5px; color: #5A6270;
          text-align: center; }}

  .verdict {{ display: flex; align-items: center; gap: 14px; margin: 16px 0 0 166px;
              padding: 11px 16px; border-left: 3px solid #B02E2E;
              background: #FAF3F3; }}
  .verdict b {{ font-size: 15px; flex: 0 0 auto; white-space: nowrap; }}
  .verdict span {{ font-size: 13.5px; color: #5A6270; }}
</style>
<div class="wrap">
  {rows}
  <div class="axis">{ticks}</div>
  <div class="cap">widest gap in churn rate between two groups &nbsp;·&nbsp;
    pale bars = {SHUFFLES:,} shuffles &nbsp;·&nbsp;
    <span style="color:#B02E2E">red = what we actually saw</span></div>
  <div class="verdict">
    <b>Ask all five at once: p = {family_p:.2f}</b>
    <span>Industry alone is p = {p_each.min():.2f}. Score every shuffle on all five
      variables and keep its best, and {family_p:.0%} of them beat what we saw.</span>
  </div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "05_segment_shuffle.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "05_segment_shuffle.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1000, "height": 640},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".wrap").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"cutoff {CUTOFF_DATE.date()}  observed gaps "
      + " ".join(f"{v[1]}={o:.3f}(p={p:.3f})"
                 for v, o, p in zip(VARIABLES, observed, p_each)))
print(f"family-wise p = {family_p:.3f}")
