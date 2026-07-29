"""Slides drafted after the main deck was finished, kept in their own file.

These are candidates, not commitments — they land in Downloads so they can be
reviewed, reordered or dropped without touching AvePoint_Case_Study.pptx. Move
one into build_main_deck.py once it has earned its place.

Same styling module as the main deck, so anything moved across needs no rework.

Run from the repo root:  python build/build_new_slides.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deck_style import INK, RED, blank, bullets, footnote, header, new_deck, note, picture  # noqa: E402
from pptx.util import Inches  # noqa: E402

OUT = Path.home() / "Downloads" / "AvePoint_New_Slides.pptx"

prs = new_deck()


def add(kicker, title, colour=INK):
    s = blank(prs)
    header(s, kicker, title, colour)
    return s


# ================================================== A · ONE QUESTION, FOUR DATES
s = add("Part 1 · Problem framing", "The same question, asked from four dates")
picture(s, "02_rolling_origin_html.png", top=Inches(2.05), height=Inches(3.20))
footnote(s, "One date states the question. Four measure it: 648 rows and 159 "
            "leavers across 281 customers, with no record invented.")
note(s, "Sits after the target slide. The picture carries three facts at once. "
        "Features never cross the red line. The answer window always closes "
        "inside the extract, which is why the last cutoff is September and not "
        "December. And this is one question repeated, not four different "
        "questions.\n\n"
        "Rows are split by customer, never at random — the same customer appears "
        "at up to four dates and those rows are not independent. Forgetting that "
        "flatters the score by 0.016.\n\n"
        "Pooling cut measurement noise by 65% and did not move the answer: "
        "grouped AUC 0.560, sd 0.034. Built by robustness.rolling_origin_cutoffs "
        "and robustness.pooled_cv, shown in notebook 08. The figure is built by "
        "build/fig_timeline_html.py.")

# ================================================== B · WHERE THE RANGE COMES FROM
s = add("Part 1 · Problem framing", "Where the range around a score comes from")
picture(s, "03_how_the_range_is_made_html.png", top=Inches(1.95), height=Inches(2.55))
bullets(s, [
    ("Two steps that are easy to confuse. Asking the question at four dates "
     "decides how much data there is. Splitting those rows and re-scoring "
     "decides how sure we are.", INK, True),
    "A score measured once is a single draw. Measured 25 times on different "
    "customers, it moves — and how far it moves is the only honest statement "
    "of what we know.",
    "With 54 leavers at a single date, one number would look far more precise "
    "than it is, and would hide whether guessing is still on the table.",
], top=Inches(4.85), size=14, space=7)
footnote(s, "The same model every time. Nothing changes between the dots except "
            "which customers were hidden.")
note(s, "Sits after the metrics slide, against the line 'I always report a "
        "range around it, never a single number' — which means nothing until "
        "you can see where the range comes from.\n\n"
        "The mean line is deliberately unlabelled. This slide teaches the "
        "method; the headline number belongs on the performance slide, and two "
        "different figures quoting two different scores would read as "
        "carelessness.\n\n"
        "The figure is built by build/fig_range_html.py.")

# ================================================== C · NO BEST MODEL TO FIND
s = add("Part 3 · Modelling", "There was no best model to find", RED)
picture(s, "08_nested_winners.png", top=Inches(1.80), height=Inches(2.90))
bullets(s, [
    "Ten models, judged 25 times over, each time on customers who had no say in "
    "the judging.",
    ("The winner changed almost every round. Eight of the ten won at least once, "
     "and the most successful won 7 times in 25. A genuinely better model wins "
     "nearly every round.", INK, True),
    "Taking the best score at face value gives 0.583. Measuring the choice "
    "properly gives 0.534. That 0.049 gap was the choosing, not the model.",
    "Reshuffling the folds alone moves the answer by 0.090 — twice the entire "
    "signal we thought we had.",
], top=Inches(5.00), size=14, space=7)
footnote(s, "This is what turns “my best model scores 0.58” into "
            "“there is no best model”.")
note(s, "Sits after the performance slide. That one says the number falls from "
        "0.583 to 0.534; this one says there was never a winner to find.\n\n"
        "The inner score averages 0.591 against an outer 0.534, so selecting is "
        "worth 0.057 of optimism on its own. Colour is model family — grey "
        "rule-based, blue linear, green forest, orange boosting — so the eye "
        "sees the winner changing kind, not just number.\n\n"
        "If challenged that 25 folds is too few: the point is not the count. A "
        "genuine winner concentrates the wins, and this does not.\n\n"
        "From model.nested_ladder_cv, cached in outputs/reports/"
        "nested_cv_folds.csv and shown in notebook 04. The figure is built by "
        "build/fig_winners_html.py.")

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"{OUT}  ({OUT.stat().st_size / 1e6:.2f} MB, "
      f"{len(prs.slides._sldIdLst)} slides)")
