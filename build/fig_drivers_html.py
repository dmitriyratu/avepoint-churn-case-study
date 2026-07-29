"""What the model says matters, and how far that is from meaning anything.

The shipped model is a logistic regression, so its coefficients can be read
directly and there is no need to reach for SHAP to answer "what drives churn".
The chart shows the twelve largest, in the direction the model applies them,
against the size chance alone reaches: refit on shuffled labels 500 times and
record the largest coefficient each time. The dashed lines are the 95th
percentile of that. No real coefficient crosses them.

Run from the repo root:  python build/fig_drivers_html.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, robustness  # noqa: E402
from src.model import feature_names  # noqa: E402

SHUFFLES = 500
TOP = 12
SEED = 0

# Categorical coefficients are read against the category one-hot encoding drops,
# so the label has to name it. Numeric ones are standardised, so theirs is a
# move of one standard deviation. The two are not the same unit and the chart
# keeps them in separate blocks for that reason.
PLAIN = {
    "cat__latest_plan_tier_Pro": "On the Pro plan, against Basic",
    "cat__referral_source_partner": "Came through a partner, against ads",
    "cat__industry_DevTools": "A DevTools company, against Cybersecurity",
    "cat__referral_source_event": "Came through an event, against ads",
    "cat__latest_plan_tier_Enterprise": "On the Enterprise plan, against Basic",
    "cat__referral_source_organic": "Found us organically, against ads",
    "cat__billing_freq_monthly": "Billed monthly, against annually",
    "cat__country_US": "Based in the US, against Australia",
    "num__n_subscriptions": "Number of subscriptions held",
    "num__mrr_std": "Spend varies month to month",
    "num__pct_subs_ended": "Share of subscriptions already ended",
    "num__accel_30d_vs_90d": "Usage speeding up, 30 days against 90",
    "num__usage_last_30d": "Usage in the last 30 days",
    "num__days_since_last_sub_start": "Days since the last subscription began",
    "num__unique_features_used": "Number of product features used",
    "num__avg_mrr": "Average monthly spend",
    "num__n_trial_subs": "Number of trial subscriptions",
    "num__escalation_rate": "Share of tickets escalated",
    "num__usage_trend_slope": "Direction of the usage trend",
    "num__n_ended_subs": "Number of subscriptions ended",
}

BLOCKS = [
    ("cat__", "Being in a category, against the reference category"),
    ("num__", "A one standard deviation move in a number"),
]

data = pipeline.build()
X, y = data.X, data.y
estimator = robustness._selected()
fitted = estimator.fit(X, y)
names = feature_names(estimator, X)
real = fitted[-1].coef_[0]

# Permutation p rather than Wald p: 86 coefficients fitted on 177 rows under an
# L2 penalty, where the shrinkage invalidates the standard errors and the
# asymptotics do not hold. Shuffling asks the same question without assuming any
# of that.
rng = np.random.default_rng(SEED)
null_largest = np.empty(SHUFFLES)
null_each = np.empty((SHUFFLES, len(real)))
for b in range(SHUFFLES):
    shuffled = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    drawn = estimator.fit(X, shuffled)[-1].coef_[0]
    null_each[b] = np.abs(drawn)
    null_largest[b] = np.abs(drawn).max()
band = float(np.percentile(null_largest, 95))
family_p = float((null_largest >= np.abs(real).max()).mean())
p_each = (null_each >= np.abs(real)).mean(axis=0)
n_small = int((p_each < 0.05).sum())

everything = (pd.DataFrame({"feature": names, "coef": real, "p": p_each})
              .assign(size=lambda t: t["coef"].abs())
              .sort_values("size", ascending=False))
blocks = {prefix: everything[everything["feature"].str.startswith(prefix)].head(TOP // 2)
          for prefix, _ in BLOCKS}

LIMIT = max(band, everything["size"].head(TOP).max()) * 1.12


def pos(value):
    """Coefficient to a percentage across a track centred on zero."""
    return (value + LIMIT) / (2 * LIMIT) * 100


rows = ""
for prefix, heading in BLOCKS:
    rows += f'<div class="block">{heading}</div>'
    for _, r in blocks[prefix].iterrows():
        label = PLAIN.get(r["feature"], r["feature"])
        left, width = sorted([pos(0), pos(r["coef"])])
        rows += f"""
    <div class="row">
      <span class="name">{label}</span>
      <span class="track">
        <i class="zero" style="left:{pos(0):.2f}%"></i>
        <i class="band" style="left:{pos(-band):.2f}%"></i>
        <i class="band" style="left:{pos(band):.2f}%"></i>
        <i class="bar {'up' if r['coef'] > 0 else 'down'}"
           style="left:{left:.2f}%;width:{width - left:.2f}%"></i>
      </span>
      <span class="odds">&times;{np.exp(r['coef']):.2f}</span>
      <span class="p{' small' if r['p'] < 0.05 else ''}">p = {r['p']:.3f}</span>
    </div>"""

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1020px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ padding: 20px 24px 22px; }}

  .head {{ display: flex; align-items: flex-end; margin-bottom: 10px; }}
  .head .name {{ flex: 0 0 268px; }}
  .head .odds {{ flex: 0 0 62px; padding-left: 18px; font-size: 12px;
                 color: #5A6270; font-weight: 400; }}
  .head .p {{ flex: 0 0 82px; padding-left: 10px; font-size: 12px;
              color: #5A6270; }}
  .head .track {{ flex: 1 1 auto; position: relative; height: 20px;
                  font-size: 12.5px; color: #5A6270; }}
  .head .track b {{ position: absolute; font-weight: 400; }}

  .row {{ display: flex; align-items: center; height: 30px; }}
  .name {{ flex: 0 0 268px; text-align: right; padding-right: 18px;
           font-size: 13.5px; }}
  .track {{ flex: 1 1 auto; min-width: 0; position: relative; height: 30px; }}
  .track i {{ position: absolute; display: block; }}
  .zero {{ top: 0; bottom: 0; width: 1px; background: #C9CFD6; }}
  .band {{ top: 0; bottom: 0; width: 0; border-left: 2px dashed #B02E2E; }}
  .bar {{ top: 50%; height: 13px; margin-top: -6.5px; border-radius: 2px; }}
  .bar.up {{ background: #eb6834; }}
  .bar.down {{ background: #2a78d6; }}

  .block {{ margin: 14px 144px 5px 268px; padding: 0 0 4px 18px; font-size: 12.5px;
            font-weight: 700; color: #5A6270;
            border-bottom: 1px solid #E4E7EB; }}
  .block:first-child {{ margin-top: 0; }}
  .odds {{ flex: 0 0 62px; padding-left: 18px; font-size: 13.5px;
           font-weight: 700; }}
  .p {{ flex: 0 0 82px; padding-left: 10px; font-size: 12.5px; color: #9AA3AF; }}
  .p.small {{ color: #1A1A1A; }}

  .foot {{ margin: 12px 144px 0 268px; padding-left: 18px; font-size: 12.5px;
           color: #5A6270; }}
  .foot b {{ color: #B02E2E; font-weight: 400; }}
  .foot span {{ display: block; margin-top: 3px; }}
</style>
<div class="wrap">
  <div class="head">
    <span class="name"></span>
    <span class="track">
      <b style="right:52%">&#9664; makes leaving less likely</b>
      <b style="left:52%">makes leaving more likely &#9654;</b>
    </span>
    <span class="odds">odds</span>
    <span class="p">chance of<br>this by luck</span>
  </div>
  {rows}
  <div class="foot"><b>- - - {band:.2f}</b> = the biggest coefficient chance
    reaches, from {SHUFFLES} runs on shuffled labels. Nothing reaches it.
    <span>p is per feature, from the same {SHUFFLES} runs. {n_small} of
    {len(real)} features fall under 0.05, where chance alone delivers about
    {0.05 * len(real):.0f}. Asked once for the whole model, p = {family_p:.2f}.</span></div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "13_drivers.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "13_drivers.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1020, "height": 520},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".wrap").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"largest real |coef| {np.abs(real).max():.3f}   null 95th pct {band:.3f}   "
      f"family-wise p {family_p:.3f}")
