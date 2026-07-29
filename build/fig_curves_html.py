"""ROC, precision-recall and calibration for the chosen model.

Three curves rather than one, because they fail differently. ROC is invariant to
class balance and so keeps its 0.50 floor across the four cutoffs. PR is not,
which is why its baseline is drawn at the base rate rather than at zero, and why
average precision barely clears it. Calibration is the one that decides whether
a score can be read as a probability, which is what the break-even argument on
the slide depends on.

The operating point is marked on all three: the threshold the model actually
ships with, not an abstract sweep.

Drawn as inline SVG so the type matches the rest of the deck.

Run from the repo root:  python build/fig_curves_html.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import model, pipeline, robustness  # noqa: E402

W, H = 250, 250          # drawing area of each panel, in SVG units
PAD = 1

built = pipeline.build()
y = built.y.to_numpy()
threshold, _, proba = model.oof_threshold(robustness._selected(), built.X, built.y)
base_rate = y.mean()

fpr, tpr, roc_thr = roc_curve(y, proba)
auc = roc_auc_score(y, proba)
prec, rec, pr_thr = precision_recall_curve(y, proba)
ap = average_precision_score(y, proba)
prob_true, prob_pred = calibration_curve(y, proba, n_bins=5, strategy="quantile")

# Where the shipped threshold sits on each curve.
roc_at = int(np.argmin(np.abs(roc_thr - threshold)))
pr_at = int(np.argmin(np.abs(pr_thr - threshold)))


def path(xs, ys):
    """Screen-space polyline: x rightwards, y upwards from the bottom-left."""
    return " ".join(f"{PAD + x * W:.2f},{PAD + (1 - yy) * H:.2f}"
                    for x, yy in zip(xs, ys))


def dot(x, yy):
    return f'cx="{PAD + x * W:.2f}" cy="{PAD + (1 - yy) * H:.2f}"'


def panel(title, subtitle, body, xlabel, ylabel):
    return f"""
    <div class="panel">
      <h2>{title}<em>{subtitle}</em></h2>
      <div class="chart">
        <span class="ytop">1</span>
        <span class="ybot">0</span>
        <span class="ylab">{ylabel}</span>
        <svg viewBox="0 0 {W + 2 * PAD} {H + 2 * PAD}">
          {body}
        </svg>
      </div>
      <div class="axis"><span>0</span><span>{xlabel}</span><span>1</span></div>
    </div>"""


roc_svg = f"""
  <polyline class="ref" points="{path([0, 1], [0, 1])}"/>
  <polyline class="curve" points="{path(fpr, tpr)}"/>
  <circle class="op" r="5" {dot(fpr[roc_at], tpr[roc_at])}/>"""

pr_svg = f"""
  <polyline class="ref" points="{path([0, 1], [base_rate, base_rate])}"/>
  <polyline class="curve" points="{path(rec, prec)}"/>
  <circle class="op" r="5" {dot(rec[pr_at], prec[pr_at])}/>"""

cal_svg = f"""
  <polyline class="ref" points="{path([0, 1], [0, 1])}"/>
  <polyline class="curve" points="{path(prob_pred, prob_true)}"/>
  """ + "".join(f'<circle class="pt" r="4" {dot(x, t)}/>'
                for x, t in zip(prob_pred, prob_true))

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1080px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ display: flex; gap: 40px; padding: 18px 24px 20px; }}
  .panel {{ flex: 1 1 0; }}
  h2 {{ font-size: 14px; font-weight: 700; margin-bottom: 10px; }}
  h2 em {{ display: block; font-style: normal; font-weight: 400; font-size: 12px;
           color: #5A6270; margin-top: 2px; }}

  .chart {{ position: relative; padding-left: 34px; }}
  .chart svg {{ width: 100%; aspect-ratio: 1 / 1; height: auto; display: block;
                border-left: 1px solid #D8DCE0; border-bottom: 1px solid #D8DCE0; }}
  .ylab {{ position: absolute; left: -2px; top: 50%; white-space: nowrap;
           font-size: 11.5px; color: #5A6270;
           transform: rotate(-90deg) translate(-50%, -50%);
           transform-origin: left top; }}
  .ytop, .ybot {{ position: absolute; right: calc(100% - 30px); font-size: 11.5px;
                  color: #5A6270; }}
  .ytop {{ top: -6px; }}
  .ybot {{ bottom: -6px; }}
  .curve {{ fill: none; stroke: #2a78d6; stroke-width: 2.2;
            vector-effect: non-scaling-stroke; }}
  .ref {{ fill: none; stroke: #AEB6C2; stroke-width: 1.4; stroke-dasharray: 5 4;
          vector-effect: non-scaling-stroke; }}
  .op {{ fill: #B02E2E; stroke: #fff; stroke-width: 2.5; }}
  .pt {{ fill: #2a78d6; }}

  .axis {{ display: flex; justify-content: space-between; margin: 5px 0 0 20px;
           font-size: 11.5px; color: #5A6270; }}
  .key {{ display: flex; gap: 26px; align-items: center; margin: 14px 24px 0;
          font-size: 12.5px; color: #5A6270; }}
  .key i {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%;
            background: #B02E2E; margin-right: 7px; vertical-align: -1px; }}
  .key u {{ display: inline-block; width: 22px; border-top: 1.4px dashed #AEB6C2;
            margin-right: 7px; vertical-align: 4px; }}
</style>
<div class="page">
<div class="wrap">
  {panel("ROC", f"AUC = {auc:.3f}, out of fold", roc_svg,
         "false positive rate", "true positive rate")}
  {panel("Precision-recall", f"AP = {ap:.3f} against a {base_rate:.2f} base rate",
         pr_svg, "recall", "precision")}
  {panel("Calibration", "5 quantile bins", cal_svg,
         "mean predicted probability", "observed churn rate")}
</div>
<div class="key">
  <span><i></i>operating point &mdash; threshold {threshold}</span>
  <span><u></u>chance, or the base rate on the PR panel</span>
  <span>ROC keeps a 0.50 floor at any class balance; PR does not, which is why
    its floor is drawn at {base_rate:.2f}</span>
</div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "04_curves.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "04_curves_html.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1080, "height": 560},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".page").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"AUC {auc:.4f}  AP {ap:.4f}  base rate {base_rate:.4f}  threshold {threshold}")
print(f"operating point: FPR {fpr[roc_at]:.3f} TPR {tpr[roc_at]:.3f}  "
      f"recall {rec[pr_at]:.3f} precision {prec[pr_at]:.3f}")
