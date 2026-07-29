"""The model ladder, one cutoff against four pooled — the figure for slide 8.

Ten rungs from a no-feature prior to gradient boosting, every one scored on
identical splits. On a single cutoff no rung's interval clears chance, which is
the finding. Pooling four cutoffs appears to fix that — six of nine clear it and
the error bars halve — and the right-hand panel exists to show why that is not
an improvement. Knowing only which of the four dates a row came from scores
0.590 on its own, because the churn rate runs 17, 17, 31, 31 percent across
them. That is the generator artefact, not the customers.

Reads the cached ladder tables. Regenerate the pooled one with
scratchpad/pooled_ladder.py if the pipeline changes.

Run from the repo root:  python build/fig_model_ladder_html.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, robustness  # noqa: E402
from src.config import TARGET  # noqa: E402
from src.labeling import build_cohort  # noqa: E402

XLO, XHI = 0.30, 0.80
CHANCE = 0.50

SHORT = {
    "0. Prior (no features)": "Prior, no features",
    "1. Single decision stump": "One decision stump",
    "2. Logistic (L2, C=1)": "Logistic L2, C = 1",
    "3. Logistic (L2, C=0.05)": "Logistic L2, C = 0.05",
    "4. Logistic (L1, C=0.1)": "Logistic L1, C = 0.1",
    "5. Random forest (depth 4)": "Random forest, depth 4",
    "6. LightGBM (pipelined)": "LightGBM",
    "7. LightGBM (native NaN + categoricals)": "LightGBM, native",
    "8. XGBoost (native NaN + categoricals)": "XGBoost, native",
    "9. HistGradientBoosting (native NaN)": "HistGradientBoosting",
}

single = pd.read_csv(ROOT / "outputs" / "reports" / "model_ladder.csv").rename(
    columns={"roc_auc_mean": "auc"})
pooled = pd.read_csv(ROOT / "outputs" / "reports" / "pooled_model_ladder.csv")

# What a row's cutoff alone is worth, which is the reference the pooled panel
# has to be read against.
cutoffs = robustness.rolling_origin_cutoffs(n=4)
tables = pipeline.clean_all(pipeline.load_all())
labels, origin = [], []
for i, c in enumerate(cutoffs):
    cohort = build_cohort(tables, cutoff=c, prediction_start=c)
    labels += list(cohort[TARGET])
    origin += [i] * len(cohort)
labels, origin = np.array(labels), np.array(origin)
rates = [labels[origin == i].mean() for i in range(len(cutoffs))]
calendar_auc = roc_auc_score(labels, np.array([rates[i] for i in origin]))


def pos(value):
    return (min(max(value, XLO), XHI) - XLO) / (XHI - XLO) * 100


def lanes(frame, extra=""):
    out = ""
    for _, r in frame.iterrows():
        clears = r["ci_lo"] > CHANCE
        flat = r["ci_hi"] - r["ci_lo"] < 1e-9
        out += f"""
        <div class="lane">
          <span class="track">
            <i class="chance" style="left:{pos(CHANCE):.2f}%"></i>{extra}
            {'' if flat else f'<i class="ci{" up" if clears else ""}" style="left:{pos(r["ci_lo"]):.2f}%;width:{pos(r["ci_hi"]) - pos(r["ci_lo"]):.2f}%"></i>'}
            <i class="dot{" up" if clears else ""}" style="left:{pos(r["auc"]):.2f}%"></i>
          </span>
          <span class="val{" up" if clears else ""}">{r["auc"]:.3f}</span>
        </div>"""
    return out


names = "".join(f'<div class="lane"><span class="name">{SHORT[m]}</span></div>'
                for m in single["model"])
ticks = "".join(f'<span style="left:{pos(t):.2f}%">{t:.1f}</span>'
                for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
calendar_mark = (f'<i class="calendar" style="left:{pos(calendar_auc):.2f}%"></i>')

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1060px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ display: flex; padding: 18px 22px 20px; gap: 0; }}
  .col {{ display: flex; flex-direction: column; }}
  .col.names {{ flex: 0 0 176px; }}
  .col.panel {{ flex: 1 1 0; padding: 0 18px; }}

  .title {{ height: 42px; font-size: 14px; font-weight: 700; }}
  .title em {{ display: block; font-style: normal; font-size: 12.5px;
               font-weight: 400; color: #5A6270; margin-top: 2px; }}

  .lane {{ height: 27px; display: flex; align-items: center; }}
  .name {{ width: 100%; text-align: right; padding-right: 14px; font-size: 13px; }}
  .track {{ flex: 1 1 auto; min-width: 0; position: relative; height: 27px; }}
  .track i {{ position: absolute; display: block; }}
  .chance {{ top: 0; bottom: 0; width: 0; border-left: 1px dashed #B02E2E; }}
  .calendar {{ top: 0; bottom: 0; width: 2px; background: #eb6834;
               transform: translateX(-1px); }}
  .ci {{ top: 50%; height: 3px; margin-top: -1.5px; background: #94A6BC;
         border-radius: 2px; }}
  .ci.up {{ background: #2a78d6; }}
  .dot {{ top: 50%; width: 9px; height: 9px; margin: -4.5px 0 0 -4.5px;
          border-radius: 50%; background: #5A6270; }}
  .dot.up {{ background: #2a78d6; }}
  .val {{ flex: 0 0 46px; padding-left: 14px; font-size: 12.5px; color: #5A6270; }}
  .val.up {{ color: #2a78d6; font-weight: 700; }}

  .axis {{ position: relative; height: 18px; margin: 5px 46px 0 0; }}
  .axis span {{ position: absolute; transform: translateX(-50%); font-size: 11.5px;
                color: #5A6270; }}
  .foot {{ margin: 4px 46px 0 0; font-size: 12px; color: #5A6270; }}
  .foot b {{ color: #B02E2E; font-weight: 400; }}
  .foot i {{ color: #eb6834; font-style: normal; font-weight: 700; }}
  .pad {{ height: 42px; }}
</style>
<div class="wrap">
  <div class="col names">
    <div class="pad"></div>
    {names}
  </div>
  <div class="col panel">
    <div class="title">One cutoff<em>177 customers, 54 left &middot; 50 folds</em></div>
    {lanes(single)}
    <div class="axis">{ticks}</div>
    <div class="foot"><b>- - - chance, 0.50</b> &middot; no rung clears it</div>
  </div>
  <div class="col panel">
    <div class="title">Four cutoffs pooled<em>648 rows, 159 left &middot; 25 grouped folds</em></div>
    {lanes(pooled, calendar_mark)}
    <div class="axis">{ticks}</div>
    <div class="foot"><i>&#9474; {calendar_auc:.3f}</i> = knowing only which of the
      four dates a row came from</div>
  </div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "04_model_ladder.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "04_model_ladder_html.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1060, "height": 520},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".wrap").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"calendar-only AUC {calendar_auc:.4f}  rates "
      + ", ".join(f"{r:.1%}" for r in rates))
