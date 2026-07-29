"""What the model does to a call list — the figure for the performance slide.

AUC says how well the model sorts. It does not say what the success team lives
with, which is a list of names, a share of leavers caught, and a share of calls
wasted. This is that.

Deliberately free of currency. The repo carries an illustrative call cost and
success rate, and neither appears anywhere in the RavenStack tables, so quoting
dollars here would be inventing the load-bearing numbers. Everything below is a
count or a rate taken from the cohort, and the break-even is stated as the ratio
it actually is: a call is worth making whenever it costs less than the churn
risk of the person being called, measured against the value of a save.

Run from the repo root:  python build/fig_performance_html.py
"""
import sys
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import model, pipeline, robustness  # noqa: E402
from src.config import CUTOFF_DATE, HORIZON_DAYS  # noqa: E402
from src.model import model_ladder  # noqa: E402

MODEL_NAME = model_ladder()[2][0].split(". ", 1)[1]

built = pipeline.build()
y = built.y.to_numpy()
threshold, _, proba = model.oof_threshold(robustness._selected(), built.X, built.y)
tn, fp, fn, tp = confusion_matrix(y, (proba >= threshold).astype(int),
                                  labels=[0, 1]).ravel()

recall, precision = tp / (tp + fn), tp / (tp + fp)
f1 = f1_score(y, (proba >= threshold).astype(int))
auc_cv = roc_auc_score(y, proba)
AUC_LADDER, AUC_NESTED, AUC_NESTED_SE, CHANCE = 0.583, 0.534, 0.016, 0.500
base_rate = y.mean()
skipped_rate = fn / (tn + fn)
lift = precision / base_rate

BAR = 250  # px for the widest rate bar


def bar(rate):
    return BAR * rate / max(precision, base_rate, skipped_rate)


HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1180px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .page {{ padding: 18px 24px 22px; }}
  .head {{ display: flex; gap: 26px; align-items: baseline; padding-bottom: 10px;
           border-bottom: 1px solid #D8DCE0; }}
  .head b {{ font-size: 14.5px; }}
  .head span {{ font-size: 12.5px; color: #5A6270; }}
  .head span i {{ font-style: normal; color: #1A1A1A; font-weight: 700; }}
  .stats {{ display: flex; gap: 0; padding: 12px 0 12px; border-bottom: 1px solid #D8DCE0; }}
  .stat {{ flex: 1 1 0; padding: 0 16px; border-left: 1px solid #E4E7EB; }}
  .stat:first-child {{ border-left: 0; padding-left: 0; }}
  .stat b {{ display: block; font-size: 17px; }}
  .stat span {{ display: block; font-size: 11.5px; color: #5A6270; margin-top: 3px;
                line-height: 1.35; }}
  .wrap {{ display: flex; gap: 34px; padding: 14px 0 0; align-items: stretch; }}
  .panel {{ flex: 1 1 0; }}
  h2 {{ font-size: 14px; font-weight: 700; }}
  h2 em {{ display: block; font-style: normal; font-weight: 400; font-size: 12px;
           color: #5A6270; margin-top: 2px; }}
  .muted {{ color: #5A6270; }}

  .cm {{ display: grid; grid-template-columns: 62px 1fr 1fr; gap: 4px;
         margin-top: 14px; }}
  .cm .hd {{ font-size: 11.5px; color: #5A6270; display: flex; align-items: center;
             justify-content: center; text-align: center; height: 24px; }}
  .cm .rh {{ font-size: 11.5px; color: #5A6270; display: flex; align-items: center;
             justify-content: flex-end; text-align: right; padding-right: 8px; }}
  .cell {{ height: 62px; border-radius: 2px; padding: 7px 9px; color: #fff; }}
  .cell em {{ float: right; font-style: normal; font-size: 11.5px; font-weight: 700;
              opacity: .75; letter-spacing: .04em; }}
  .cell b {{ font-size: 22px; display: block; line-height: 1; }}
  .cell span {{ font-size: 11.5px; display: block; margin-top: 4px; opacity: .9; }}
  .good {{ background: #2a78d6; }}
  .bad {{ background: #B02E2E; }}
  .idle {{ background: #AEB6C2; }}

  .lines {{ margin-top: 11px; font-size: 12.5px; color: #5A6270; line-height: 1.55; }}
  .lines b {{ color: #1A1A1A; }}

  .rate {{ display: flex; align-items: center; height: 40px; gap: 10px; }}
  .rlab {{ flex: 0 0 108px; font-size: 12.5px; text-align: right; }}
  .rbar {{ height: 20px; border-radius: 2px; background: #AEB6C2; }}
  .rbar.hi {{ background: #2a78d6; }}
  .rbar.base {{ background: #B02E2E; }}
  .rval {{ font-size: 13.5px; font-weight: 700; }}

  .box {{ margin-top: 14px; padding: 10px 13px; background: #F4F6F9;
          border-left: 3px solid #2a78d6; font-size: 12.5px; line-height: 1.5; }}
  .box.warn {{ background: #FAF3F3; border-left-color: #B02E2E; }}
  .box b {{ font-size: 13.5px; }}
</style>
<div class="page">
<div class="head">
  <b>{MODEL_NAME}</b>
  <span>one cutoff, <i>{CUTOFF_DATE.day} {CUTOFF_DATE:%B %Y}</i> &mdash; not pooled</span>
  <span><i>{len(y)}</i> customers, <i>{int(y.sum())}</i> left within
    {HORIZON_DAYS} days</span>
  <span>scores and threshold both out of fold</span>
</div>
<div class="stats">
  <div class="stat"><b>{AUC_LADDER:.3f}</b><span>ROC-AUC, cross-validated</span></div>
  <div class="stat"><b>{AUC_NESTED:.3f} &plusmn; {AUC_NESTED_SE:.3f}</b>
    <span>nested CV &mdash; selection inside the outer folds</span></div>
  <div class="stat"><b>{CHANCE:.3f}</b><span>chance</span></div>
  <div class="stat"><b>{threshold}</b><span>threshold, maximising out-of-fold F1</span></div>
  <div class="stat"><b>{precision:.3f} / {recall:.3f} / {f1:.3f}</b>
    <span>precision / recall / F1 at that threshold</span></div>
</div>
<div class="wrap">

  <div class="panel">
    <h2>Confusion matrix<em>at threshold {threshold}</em></h2>
    <div class="cm">
      <div class="hd"></div><div class="hd">model says call</div>
      <div class="hd">model says skip</div>
      <div class="rh">actually<br>left</div>
      <div class="cell good"><em>TP</em><b>{tp}</b><span>caught</span></div>
      <div class="cell bad"><em>FN</em><b>{fn}</b><span>missed</span></div>
      <div class="rh">actually<br>stayed</div>
      <div class="cell idle"><em>FP</em><b>{fp}</b><span>called anyway</span></div>
      <div class="cell idle" style="background:#DCE1E8;color:#5A6270">
        <em>TN</em><b>{tn}</b><span>correctly skipped</span></div>
    </div>
    <div class="lines">
      recall = TP/(TP+FN) = {tp}/{tp + fn} = <b>{recall:.3f}</b><br>
      precision = TP/(TP+FP) = {tp}/{tp + fp} = <b>{precision:.3f}</b><br>
      <b>{tp + fp}</b> names on the list, out of {len(y)}
    </div>
  </div>

  <div class="panel">
    <h2>Who it separates<em>churn rate in each group it produces</em></h2>
    <div style="margin-top:16px">
      <div class="rate">
        <span class="rlab">told to call</span>
        <span class="rbar hi" style="width:{bar(precision):.0f}px"></span>
        <span class="rval">{precision:.1%}</span>
      </div>
      <div class="rate">
        <span class="rlab">everyone</span>
        <span class="rbar base" style="width:{bar(base_rate):.0f}px"></span>
        <span class="rval" style="color:#B02E2E">{base_rate:.1%}</span>
      </div>
      <div class="rate">
        <span class="rlab">told to skip</span>
        <span class="rbar" style="width:{bar(skipped_rate):.0f}px"></span>
        <span class="rval muted">{skipped_rate:.1%}</span>
      </div>
    </div>
    <div class="box warn">
      <b>There is no safe group.</b><br>
      The {tn + fn} customers it tells us to skip still churn at
      {skipped_rate:.0%}. A flagged customer is only {lift:.2f}&times; more
      likely to leave than one picked at random.
    </div>
  </div>

  <div class="panel">
    <h2>What this means for the decision<em>no cost figures &mdash; none exist in
      the data</em></h2>
    <div class="box" style="margin-top:16px">
      A call is worth making when its cost is below the customer's churn risk,
      measured as a share of what a save is worth.
    </div>
    <div class="lines" style="margin-top:14px;font-size:13px">
      Calling <b>everyone</b> pays whenever a call costs less than
      <b>{base_rate:.1%}</b> of a save.<br><br>
      Calling only the <b>flagged</b> pays whenever a call costs less than
      <b>{precision:.1%}</b> of a save.
    </div>
    <div class="box warn" style="margin-top:14px">
      The model moves that bar by <b>{precision - base_rate:.1%}</b>, and gives
      up <b>{fn}</b> of {int(y.sum())} leavers to do it. Unless a call costs
      between {base_rate:.0%} and {precision:.0%} of a save, the list does not
      change the decision.
    </div>
  </div>

</div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "04_performance.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "04_performance.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1180, "height": 560},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".page").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"model {MODEL_NAME}  threshold {threshold}  TP {tp} FP {fp} FN {fn} TN {tn}")
print(f"recall {recall:.3f}  precision {precision:.3f}  base {base_rate:.3f}  "
      f"skipped {skipped_rate:.3f}  lift {lift:.2f}")
