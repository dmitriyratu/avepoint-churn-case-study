"""Does the pooled model beat its own null — the figure for the performance slide.

On one cutoff the answer is no: the best of ten models scores 0.583 and the same
search on shuffled labels reaches 0.594. Pooling four cutoffs triples the
labelled rows and the answer changes, but only just.

The null has to be built carefully. Shuffling labels freely across the pooled
rows would destroy the 17/17/31/31 churn rates along with any customer signal,
so the null would sit near chance and the pooled result would look decisive for
the wrong reason. Shuffling inside each cutoff keeps the calendar and asks the
question that matters: beyond knowing which quarter a row came from, is anything
there?

Run from the repo root:  python build/fig_pooled_null_html.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OBSERVED = 0.6081       # best of ladder on pooled rows, averaged over 10 CV seeds
OBSERVED_LO = 0.5941
OBSERVED_HI = 0.6202
CALENDAR = 0.5898       # AUC from knowing only which of the four dates a row is from
SINGLE_OBSERVED = 0.583
SINGLE_NULL_BEST = 0.594
XLO, XHI, BINS = 0.46, 0.68, 44

null = pd.read_csv(ROOT / "outputs" / "reports" /
                   "pooled_selection_null.csv")["best_auc"].to_numpy()
p_value = (null >= OBSERVED).mean()
counts, edges = np.histogram(np.clip(null, XLO, XHI),
                             bins=np.linspace(XLO, XHI, BINS + 1))
tallest = counts.max()


def pos(value):
    return (min(max(value, XLO), XHI) - XLO) / (XHI - XLO) * 100


bars = "".join(f'<i style="height:{c / tallest * 100:.1f}%"></i>' for c in counts)
ticks = "".join(f'<span style="left:{pos(t):.2f}%">{t:.2f}</span>'
                for t in (0.50, 0.55, 0.60, 0.65))

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1000px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ padding: 22px 26px 22px; }}

  .plot {{ position: relative; height: 190px; margin: 34px 20px 0 20px;
           border-bottom: 1px solid #D8DCE0; }}
  .bars {{ position: absolute; inset: 0; display: flex; align-items: flex-end;
           gap: 1px; }}
  .bars i {{ flex: 1 1 0; background: #C3D4EA; display: block; }}
  .mark {{ position: absolute; top: -6px; bottom: 0; width: 2px;
           transform: translateX(-1px); }}
  .mark.obs {{ background: #B02E2E; }}
  .mark.cal {{ background: #eb6834; }}
  .mark.band {{ top: -6px; bottom: 0; width: auto; background: #B02E2E;
                opacity: .10; transform: none; }}
  .tag {{ position: absolute; transform: translateX(-50%); white-space: nowrap;
          font-size: 12.5px; font-weight: 700; }}
  .tag em {{ display: block; font-style: normal; font-weight: 400;
             font-size: 11.5px; }}
  .tag.obs {{ top: -46px; color: #B02E2E; }}
  .tag.cal {{ top: -46px; color: #eb6834; }}

  .axis {{ position: relative; height: 20px; margin: 3px 20px 0; }}
  .axis span {{ position: absolute; transform: translateX(-50%); font-size: 12px;
                color: #5A6270; }}
  .cap {{ text-align: center; margin: 2px 20px 0; font-size: 12.5px;
          color: #5A6270; }}

  .facts {{ display: flex; gap: 0; margin-top: 20px; }}
  .fact {{ flex: 1 1 0; padding: 0 18px; border-left: 1px solid #E4E7EB; }}
  .fact:first-child {{ border-left: 0; padding-left: 0; }}
  .fact b {{ display: block; font-size: 20px; }}
  .fact span {{ display: block; font-size: 12.5px; color: #5A6270; margin-top: 3px;
                line-height: 1.35; }}
  .fact.hot b {{ color: #B02E2E; }}
</style>
<div class="wrap">
  <div class="plot">
    <span class="bars">{bars}</span>
    <i class="mark band" style="left:{pos(OBSERVED_LO):.2f}%;
       width:{pos(OBSERVED_HI) - pos(OBSERVED_LO):.2f}%"></i>
    <i class="mark cal" style="left:{pos(CALENDAR):.2f}%"></i>
    <i class="mark obs" style="left:{pos(OBSERVED):.2f}%"></i>
    <span class="tag cal" style="left:{pos(CALENDAR):.2f}%">{CALENDAR:.3f}
      <em>calendar only</em></span>
    <span class="tag obs" style="left:{pos(OBSERVED):.2f}%">{OBSERVED:.3f}
      <em>our best model</em></span>
  </div>
  <div class="axis">{ticks}</div>
  <div class="cap">pale bars = the best of ten models on {len(null)} shuffles,
    labels scrambled inside each cutoff so the calendar survives</div>

  <div class="facts">
    <div class="fact hot"><b>p = {p_value:.3f}</b>
      <span>{int((null >= OBSERVED).sum())} of {len(null)} shuffles reached what
        we saw. The 95th percentile of the null is {np.percentile(null, 95):.3f},
        against our {OBSERVED:.3f}.</span></div>
    <div class="fact"><b>+{OBSERVED - CALENDAR:.3f} AUC</b>
      <span>what every customer feature adds over knowing only which of the four
        quarters a row came from.</span></div>
    <div class="fact"><b>{SINGLE_OBSERVED:.3f} vs {SINGLE_NULL_BEST:.3f}</b>
      <span>on a single cutoff the same test fails outright: shuffled labels beat
        the real ones.</span></div>
  </div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "09_pooled_selection_null.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "09_pooled_selection_null.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1000, "height": 480},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".wrap").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"observed {OBSERVED:.4f}  null mean {null.mean():.4f}  "
      f"p95 {np.percentile(null, 95):.4f}  max {null.max():.4f}  p = {p_value:.3f}")
