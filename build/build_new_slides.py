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

# ==================================================== D · NO SEGMENT STANDS OUT
s = add("Part 2 · Data exploration", "No group of customers stands out")
picture(s, "05_segment_forest_html.png", top=Inches(1.72), height=Inches(4.95),
        left=Inches(0.50))
bullets(s, [
    ("I checked industry, country, plan, how they found us, and whether they "
     "started on a trial. Not one of them separates the customers who leave "
     "from the ones who stay.", INK, True),
    "",
    ("The red line is 31%: 54 of the 177 customers left.", INK, True),
    "  This is one cutoff, 30 June 2024 — a single line drawn in the data, not "
    "the four dates pooled. Every group below is a slice of those same 177. If "
    "a slice also sits at 31%, knowing a customer is in it tells us nothing we "
    "did not already know.",
    "",
    ("The lines say how little a group this size can tell you.", INK, True),
    "  Cybersecurity is 2 leavers out of 23. That is 9%, but 23 customers cannot "
    "pin down a rate: anything from 2% to 27% would produce what we saw. The "
    "width comes from the group size, not from the churn.",
    "",
    ("One group does miss the line, and it is the one to volunteer.", INK, True),
    "  Cybersecurity. But testing 22 groups at 95% confidence, you expect 1.1 to "
    "miss by chance. We got one.",
    "",
    "  Strongest association across all five is p = 0.08, before accounting for "
    "having looked at five. I explored this on a held-out slice, so the data I "
    "modelled on did not influence what I chose to look at.",
], top=Inches(1.80), left=Inches(6.55), width=Inches(6.10), size=13, space=3)
footnote(s, "A group of 23 customers cannot tell you much. The width of these "
            "intervals is the finding, not the position of the dots.")
note(s, "Replaces the three bar charts. Bars of churn rate invited exactly the "
        "comparison the slide argues against — a viewer saw Cybersecurity at "
        "0.09 against DevTools at 0.41 and read a threefold effect, when it is "
        "two churners out of twenty-three.\n\n"
        "Volunteer Cybersecurity rather than waiting to be asked. Producing the "
        "one result that looks significant, and showing it is what 22 "
        "comparisons produce on their own, is the same move as the generator "
        "null later in the deck.\n\n"
        "IF ASKED HOW THE LINES ARE CALCULATED. They are Wilson intervals. The "
        "question they answer is 'which true churn rates could have produced "
        "what I saw'. For Cybersecurity we saw 2 of 23. A true rate of 5% "
        "produces that easily, 25% produces it occasionally, 45% almost never — "
        "so the line runs 2% to 27%.\n\n"
        "The textbook version is p +/- 2 x sqrt(p(1-p)/n), and it breaks here: "
        "for Cybersecurity it returns -3% to 21%, a negative churn rate, and "
        "for Canada with 0 of 6 it returns 0% to 0%, claiming certainty from "
        "six customers. Wilson is the same idea done properly and cannot leave "
        "the 0 to 1 range. Standard choice at these group sizes.\n\n"
        "Width is driven by n under a square root, so quadrupling the group "
        "halves the line: CA at n=6 spans +/-0.20, Cybersecurity at n=23 "
        "+/-0.12, the US at n=100 +/-0.09.\n\n"
        "Chi-square p-values: industry 0.079, referral 0.086, plan 0.837, "
        "country 0.428, trial 0.548. The figure is built by "
        "build/fig_segment_forest.py.")

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"{OUT}  ({OUT.stat().st_size / 1e6:.2f} MB, "
      f"{len(prs.slides._sldIdLst)} slides)")
