"""Which model wins each nested fold — the figure for slide 8.

The nested score says the search is worth 0.534. This says why. Twenty-five
times over, the ten-model contest is re-run on customers the eventual test fold
never touched, and the winner keeps changing. A genuinely better model wins
nearly every round; here eight of the ten win at least once, which is what a
tie between ten equally uninformative models looks like.

Reads the cached fold table written by `model.nested_ladder_cv` (notebook 04).

Run from the repo root:  python build/fig_winners_html.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import model_ladder  # noqa: E402

folds = pd.read_csv(ROOT / "outputs" / "reports" / "nested_cv_folds.csv")
names = [n for n, _ in model_ladder()]
wins = folds["selected"].value_counts()

# "3. Logistic (L2, C=0.05)" -> ("3", "Logistic (L2, C=0.05)")
def split_name(name):
    num, label = name.split(". ", 1)
    return num, label


# One colour per family, so the grid reads as "the winner keeps changing kind",
# not just "the winner keeps changing number".
def family(name):
    label = name.lower()
    if "prior" in label or "stump" in label:
        return "#8a94a6"
    if "logistic" in label:
        return "#2a78d6"
    if "forest" in label:
        return "#1e7a4b"
    return "#eb6834"


cells = ""
for _, row in folds.iterrows():
    num, label = split_name(row["selected"])
    cells += (f'<div class="cell" style="background:{family(row["selected"])}" '
              f'title="{label}">{num}</div>')

tally = ""
for name in names:
    num, label = split_name(name)
    won = int(wins.get(name, 0))
    tally += f"""
    <div class="mrow{' zero' if not won else ''}">
      <i style="background:{family(name) if won else '#D8DCE0'}">{num}</i>
      <span class="mname">{label}</span>
      <span class="bar" style="width:{won / wins.max() * 168:.0f}px;
            background:{family(name) if won else 'transparent'}"></span>
      <span class="won">{won or '&mdash;'}</span>
    </div>"""

n_winners = folds["selected"].nunique()
top = wins.max()

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1180px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ display: flex; gap: 64px; padding: 24px 28px 26px; }}
  h2 {{ font-size: 15.5px; font-weight: 700; margin-bottom: 6px; }}
  h2 em {{ color: #5A6270; font-style: normal; margin-right: 11px;
           letter-spacing: .09em; font-size: 13px; }}
  .note {{ color: #5A6270; font-style: italic; font-size: 13.5px; }}

  .grid {{ display: grid; grid-template-columns: repeat(5, 46px); gap: 6px;
           margin-top: 16px; }}
  .cell {{ height: 40px; border-radius: 2px; color: #fff; font-size: 16px;
           font-weight: 700; display: flex; align-items: center;
           justify-content: center; }}

  .tally {{ flex: 1 1 auto; min-width: 0; }}
  .mrow {{ display: flex; align-items: center; gap: 10px; height: 30px; }}
  .mrow i {{ flex: none; width: 22px; height: 22px; border-radius: 2px;
             color: #fff; font-size: 12.5px; font-weight: 700; font-style: normal;
             display: flex; align-items: center; justify-content: center; }}
  .mname {{ flex: 0 0 258px; font-size: 13.5px; }}
  .bar {{ flex: none; height: 13px; border-radius: 1px; }}
  .won {{ font-size: 13.5px; color: #5A6270; margin-left: 4px; }}
  .zero .mname, .zero .won {{ color: #9AA3AF; }}
</style>
<div class="wrap">
  <div>
    <h2><em>25 FOLDS</em>Who won each round</h2>
    <div class="note">the contest re-run 25 times, on different customers</div>
    <div class="grid">{cells}</div>
  </div>
  <div class="tally">
    <h2><em>THE TALLY</em>{n_winners} of {len(names)} models won at least once</h2>
    <div class="note">a genuinely better model wins nearly every round</div>
    <div style="margin-top:14px">{tally}</div>
  </div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "08_nested_winners.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "08_nested_winners.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1180, "height": 420},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".wrap").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"{len(folds)} folds, {n_winners} distinct winners, most wins {top}, "
      f"nested AUC {folds['outer_auc'].mean():.4f}")
