"""Two decision trees side by side: real labels, and labels I shuffled.

The SHAP figure makes the same point, but a bar chart asks the reader to
trust a method. A tree can just be read. Both panels tell a clean story;
only one of them was fitted on labels that mean anything.

Rendered with sklearn's `export_graphviz` through Graphviz, then relabelled
into plain English and composed side by side.
"""
import re
import sys
import tempfile
import warnings
from pathlib import Path

import graphviz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline  # noqa: E402
from src.config import CUTOFF_DATE, HORIZON_DAYS  # noqa: E402

OUT = ROOT / "outputs" / "figures" / "17_tree_real_vs_shuffled.png"

INK = "#1A1A1A"
MUTED = "#5A6270"
BLUE = "#2E5F8A"
RED = "#B02E2E"
RULE = "#C9CFD6"

DEPTH, LEAF, SEED = 3, 15, 0
DPI = 200


def numeric_frame():
    """Trees need numbers. Categories become codes, and are remembered as such
    so a split on one is never printed as if the code meant anything."""
    data = pipeline.build(cutoff=CUTOFF_DATE, prediction_start=CUTOFF_DATE,
                          horizon_days=HORIZON_DAYS, prune=False)
    X, categorical = data.X.copy(), set()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = X[c].astype("category").cat.codes
            categorical.add(c)
    return X.fillna(-999), data.y.values, categorical


def fit(X, y):
    """Tree, its cross-validated AUC, and its training accuracy."""
    tree = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=LEAF,
                                  random_state=SEED).fit(X, y)
    cv = cross_val_score(tree, X, y, scoring="roc_auc",
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    return tree, cv.mean(), cv.std(), tree.score(X, y)


WORDS = {
    "latest_mrr": "monthly spend",
    "avg_mrr": "average spend",
    "error_rate": "error rate",
    "tickets_per_seat": "tickets per seat",
    "unique_features_used": "features used",
    "recency_ratio_90d": "recent activity",
    "avg_first_response_mins": "first reply time",
    "days_since_last_ticket": "days since last ticket",
    "days_since_signup": "days since signup",
    "tenure_days": "tenure",
    "n_trial_subs": "trial signups",
    "industry": "industry",
}


def plain(name):
    """Feature names as a reader would say them out loud."""
    return WORDS.get(name, name.replace("_", " "))


def relabel(dot, categorical):
    """Rewrite Graphviz labels into plain English, and colour by verdict.

    export_graphviz writes 'feature <= 1234.5\\nsamples = 79\\nvalue = ...\\n
    class = stays'. None of that is language. Splits become a question, leaves
    become a verdict and a count, and the fill says which way the leaf went.
    """
    def node(match):
        head, label = match.group(1), match.group(2)
        lines = label.split("\\n")
        counts = next(l for l in lines if l.startswith("value = "))
        stay, go = (int(round(float(v))) for v in
                    re.findall(r"[\d.]+", counts))
        churn = go > stay
        split = lines[0] if "<=" in lines[0] else None

        if split is None:
            colour = RED if churn else BLUE
            text = ("LEAVES" if churn else "stays") + \
                   f"\\n{go} of {stay + go} left"
            return (f'{head} [label="{text}", shape=box, style="rounded,filled",'
                    f' color="{colour}", fillcolor="{colour}22",'
                    f' fontcolor="{colour}", penwidth=1.4]')

        feature, thr = split.rsplit(" <= ", 1)
        thr = float(thr)
        if feature in categorical:
            test = "in group A?"
        else:
            test = ("≤ " + (f"{thr:,.0f}" if abs(thr) >= 10
                                 else f"{thr:.2f}") + " ?")
        return (f'{head} [label="{plain(feature)}\\n{test}", shape=box,'
                f' style="rounded", color="{RULE}", fontcolor="{INK}",'
                f' penwidth=1.3]')

    dot = re.sub(r'(\d+) \[label="([^"]*)"[^\]]*\]', node, dot)
    # The root keeps its yes/no edge labels; the rest inherit the same rule.
    dot = dot.replace('headlabel="True"', 'label="yes"')
    dot = dot.replace('headlabel="False"', 'label="no"')
    dot = re.sub(r'labeldistance=[\d.]+, labelangle=-?[\d.]+, ', '', dot)
    return dot.replace(
        "digraph Tree {",
        'digraph Tree {\n'
        '  graph [ranksep=0.42, nodesep=0.28, bgcolor="white"];\n'
        f'  node [fontname="DejaVu Sans", fontsize=13, margin="0.20,0.11"];\n'
        f'  edge [fontname="DejaVu Sans", fontsize=11, color="{RULE}",'
        f' fontcolor="{MUTED}", penwidth=1.4, arrowsize=0.0];')


def render(tree, names, categorical, tmp, tag):
    dot = export_graphviz(tree, feature_names=names, filled=False,
                          impurity=False, rounded=True, special_characters=False)
    src = graphviz.Source(relabel(dot, categorical))
    path = src.render(tmp / tag, format="png", cleanup=True)
    return Image.open(path).convert("RGB")


def font(size, bold=False):
    for name in (("seguisb.ttf", "segoeui.ttf") if bold
                 else ("segoeui.ttf", "seguisb.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(panels):
    """Two rendered trees, each under its own heading, on one white canvas."""
    pad, gap, head = int(0.22 * DPI), int(0.30 * DPI), int(0.62 * DPI)
    col = max(im.width for im, *_ in panels)
    rows = max(im.height for im, *_ in panels)
    canvas = Image.new("RGB", (pad * 2 + col * 2 + gap, pad * 2 + head + rows),
                       "white")
    draw = ImageDraw.Draw(canvas)
    title_f, sub_f = font(30, bold=True), font(21)

    for i, (im, title, auc, sd, acc, accent) in enumerate(panels):
        x = pad + i * (col + gap)
        draw.text((x, pad), title, font=title_f, fill=accent)
        draw.text((x, pad + int(0.20 * DPI)),
                  f"cross-validated AUC {auc:.3f} ± {sd:.3f}"
                  f"   ·   fits the training rows {acc:.0%} right",
                  font=sub_f, fill=MUTED)
        canvas.paste(im, (x + (col - im.width) // 2, pad + head))
    return canvas


def main():
    X, y, categorical = numeric_frame()
    names = list(X.columns)
    real = fit(X, y)
    shuffled = fit(X, np.random.default_rng(1).permutation(y))

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        panels = [
            (render(real[0], names, categorical, tmp, "real"),
             "Fitted on the real labels", *real[1:], BLUE),
            (render(shuffled[0], names, categorical, tmp, "shuffled"),
             "Fitted on labels I shuffled at random", *shuffled[1:], RED),
        ]
        compose(panels).save(OUT, dpi=(DPI, DPI))

    print("wrote", OUT)


if __name__ == "__main__":
    main()
