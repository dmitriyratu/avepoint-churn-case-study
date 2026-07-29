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

# ================================================= D · WHAT CHANCE ALONE PRODUCES
s = add("Part 2 · Data exploration", "No group stands out, tested against chance")
picture(s, "05_segment_shuffle.png", top=Inches(1.66), height=Inches(4.05),
        left=Inches(0.45))
bullets(s, [
    ("Ten ways a success team could filter a call list. None of them works.",
     INK, True),
    "",
    ("Method", INK, True),
    "  Hold the groups fixed and shuffle who left, so the only thing destroyed "
    "is the link between the two. Measure the widest gap in churn rate between "
    "any two groups. Repeat 20,000 times.",
    "  The pale bars are the gaps chance produced. The red line is the gap we "
    "observed. If the line sits inside the bars, the grouping explains nothing.",
    "",
    ("Industry clears the bar on its own", RED, True),
    "  A 32-point gap, beaten by 3% of shuffles. Reported alone, that is a "
    "finding.",
    "  It was not asked alone. Ten variables were tested. Scoring each shuffle "
    "on all ten and keeping its best, 26% beat what we observed. The result "
    "does not survive the question that was actually asked.",
    "",
    ("What the widths show", INK, True),
    "  Country reaches 35 points on chance alone, because some countries hold "
    "six customers. Billing never exceeds 8, because its two groups are large. "
    "The shuffle prices that in without any assumption about distributions.",
], top=Inches(1.78), left=Inches(7.35), width=Inches(5.30), size=12.5, space=3)
footnote(s, "No group here is worth targeting, including the one that looked "
            "like it was.")
note(s, "Replaces both the bar charts and an earlier version built on "
        "confidence intervals. Intervals show precision but do not perform the "
        "test, and reading significance off whether they cross a line "
        "disagreed with the actual test — Cybersecurity's interval excluded the "
        "base rate while industry sat at p = 0.08.\n\n"
        "This does the test directly, with no formula and no distributional "
        "assumption. Shuffle the labels, keep the group sizes, recompute the "
        "statistic, repeat.\n\n"
        "The last row is the slide. Industry at p = 0.03 is exactly the finding "
        "I would have reported if I had stopped there, and it is the same "
        "mistake the deck later catches on the calendar-hazard result: testing "
        "against the wrong null. Volunteer it rather than waiting to be asked.\n\n"
        "The last five variables come from subscriptions, the one table with "
        "internal rules, so a null there carries more weight than a null in the "
        "tables already known to be scrambled.\n\n"
        "Per-variable p: industry 0.033, contract size 0.058, company size "
        "0.098, tenure 0.118, referral 0.136, country 0.501, trial 0.543, "
        "billing 0.751, plan 0.863, downgrade 1.000. Family-wise 0.262 from "
        "20,000 shuffles, seed 0, at the 30 June 2024 cutoff. Built by "
        "build/fig_segment_shuffle_html.py.")

# ============================================ E · NO SEGMENT STANDS OUT (INTERVALS)
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

# ============================================ F · WHAT MORE ROWS ACTUALLY BOUGHT
s = add("Part 3 · Modelling", "What tripling the rows actually bought", RED)
picture(s, "09_pooled_selection_null.png", top=Inches(1.70), height=Inches(2.90))
bullets(s, [
    ("The single-cutoff result was measured on 177 customers. Asking the same "
     "question at four quarter-ends gives 648 rows and 159 leavers, so the "
     "search is worth re-running.", INK, True),
    "",
    ("The null has to keep the calendar.", RED, True),
    "  Churn runs 17, 17, 31 and 31 percent across the four dates, and slide 11 "
    "shows that rise is produced by the file rather than by customers. Scramble "
    "labels freely and that structure is destroyed along with everything else, "
    "which would make any pooled model look decisive. Scrambling inside each "
    "cutoff keeps it, and asks whether anything is left once the quarter is "
    "already known.",
    "",
    ("A weak signal appears, and it is too small to act on.", INK, True),
    "  The best model reaches 0.608 against a null 95th percentile of 0.604, "
    "which is p = 0.045. Knowing only the quarter is already worth 0.590, so "
    "every customer feature in the build adds 0.018 of AUC.",
    "  Two framings were tried and this is the one that worked, so the honest "
    "reading is weaker still. Nothing here changes the recommendation on "
    "slide 13: at a 31% base rate, calling everyone already pays.",
], top=Inches(4.78), size=12.5, space=3)
footnote(s, "The single-cutoff finding stands: shuffled labels beat the real "
            "ones, 0.594 against 0.583. More rows move that to a 0.018 edge.")
note(s, "Corrects the performance slide, which reported the single-cutoff "
        "selection null only. The obvious challenge to that slide is 'you had "
        "177 rows, of course nothing showed' — this answers it with the data "
        "already built rather than with an argument.\n\n"
        "Say the calendar point before the p-value, not after. Shuffling freely "
        "across pooled rows destroys the 17/17/31/31 rates, the null collapses "
        "toward chance, and 0.608 looks overwhelming. That version of the test "
        "would have been wrong in the same way the calendar-hazard result on "
        "slide 11 was wrong.\n\n"
        "If pushed on p = 0.045: it is 9 of 200 shuffles, and it is the second "
        "framing tried. A Bonferroni correction over the two takes it to 0.09. "
        "The effect size is the better argument — 0.018 of AUC over a calendar "
        "lookup is not a model anybody should ship.\n\n"
        "Observed 0.6081 is the best of the ten-rung ladder averaged over 10 CV "
        "seeds, range 0.594 to 0.620. Null mean 0.560, p95 0.604, max 0.627, "
        "from 200 within-cutoff shuffles. Grouped by account throughout. Built "
        "by build/fig_pooled_null_html.py from outputs/reports/"
        "pooled_selection_null.csv.")

# ================================================= G · PERFORMANCE, PROPERLY
s = add("Part 3 · Modelling", "How it performs")
bullets(s, [
    ("Logistic regression, L2 penalty, C = 1. It won the ten-rung ladder on the "
     "30 June 2024 cohort: 177 customers, 54 of whom left within 90 days.",
     INK, True),
    "  ROC-AUC 0.583 in cross-validation. Nested CV, which puts model selection "
    "inside the outer folds, gives 0.534 ± 0.016. Chance is 0.500. The 0.049 "
    "gap is selection optimism.",
], top=Inches(1.72), size=13.5, space=4)
picture(s, "04_performance.png", top=Inches(2.54), height=Inches(2.72))
bullets(s, [
    ("Threshold 0.245, chosen to maximise F1 on out-of-fold predictions rather "
     "than at the 0.5 default.", INK, True),
    "  At that operating point: precision 0.368, recall 0.722, F1 0.487. The "
    "model recovers 39 of 54 leavers and puts 106 of 177 customers on the list.",
    "",
    ("The list it produces does not separate anybody.", RED, True),
    "  Flagged customers churn at 36.8%, the cohort at 30.5%, and the 71 it tells "
    "us to skip still churn at 21.1%. There is no group it identifies as safe.",
    "",
    ("No currency figures appear here, because none exist in the data.", INK, True),
    "  Nothing records what an outreach costs or how often one works. Stated as "
    "a ratio instead: calling everyone pays whenever a call costs under 30.5% of "
    "a save, and using the list moves that to 36.8%. Unless the true cost falls "
    "inside that six-point band, the model changes no decision.",
], top=Inches(5.52), size=12.5, space=3)
footnote(s, "Scores and threshold are both out of fold. Every number above is a "
            "count or a rate from this cohort.")
note(s, "The brief asks for performance, so lead with the confusion matrix and "
        "the threshold reasoning rather than with methodology.\n\n"
        "Threshold 0.245, picked out of fold on F1. TP 39, FP 67, FN 15, TN 56. "
        "Recall 0.722, precision 0.368, 106 names on a 177-customer list. "
        "Lift over random is 1.21.\n\n"
        "Deliberately no currency. src/economics.py carries an illustrative call "
        "cost and success rate; neither appears anywhere in the RavenStack "
        "tables, and every money figure downstream scales with them. If finance "
        "supplies real numbers the same slide takes them without changing "
        "shape. If asked for a dollar answer, say that and give the ratio.\n\n"
        "The strongest line is the skip list. A model that cannot find a safe "
        "group cannot save anyone a phone call, which is the whole reason to "
        "rank in the first place.\n\n"
        "One inconsistency worth owning if raised: the threshold maximises F1, "
        "which weights the two errors equally, while the argument above says "
        "they are not equal. A cost-minimising threshold needs the cost figures "
        "we do not have.\n\n"
        "Built by build/fig_performance_html.py.")

# ------------------------------------------------ G2 · CURVES
s = add("Part 3 · Modelling", "Ranking quality, three ways")
picture(s, "04_curves_html.png", top=Inches(1.72), height=Inches(3.05))
bullets(s, [
    ("ROC-AUC 0.589 out of fold. The curve sits just above the diagonal for its "
     "whole length, with no region where the model separates cleanly.", INK, True),
    "",
    ("Average precision 0.378 against a 0.31 base rate.", INK, True),
    "  PR is the honest picture for an imbalanced problem, because its floor "
    "moves with the class balance rather than staying at 0.50. Precision hovers "
    "just above the base-rate line at every recall, which is the same finding as "
    "the confusion matrix: flagging a customer barely raises their risk.",
    "",
    ("Calibration is the reason the break-even argument is usable.", INK, True),
    "  Predicted probabilities track observed churn well enough in the middle "
    "bins to compare a score against a decision threshold. The curve is flat "
    "rather than diagonal, which is what a weak model looks like — it "
    "over-predicts at the bottom and under-predicts at the top.",
], top=Inches(4.98), size=12.5, space=3)
footnote(s, "The red dot on each panel is the shipped threshold of 0.245, not an "
            "abstract sweep. Scores are out of fold.")
note(s, "Companion to the performance slide, for the reviewer who wants the "
        "curves rather than a single operating point.\n\n"
        "AUC here is 0.589 from pooled out-of-fold predictions, against 0.583 as "
        "the mean of per-fold AUCs on the ladder slide. Same quantity, two "
        "estimators; say so if the difference is noticed.\n\n"
        "The PR floor is the point worth making. A reviewer who only sees "
        "AP = 0.378 may read it as weak-but-real; drawn against a 0.31 base rate "
        "it is plainly near-worthless. This is also why ROC-AUC is the headline "
        "metric across the four cutoffs, where churn rates run 17% to 31% and a "
        "PR floor would move underneath the comparison.\n\n"
        "Calibration uses 5 quantile bins because 177 rows will not support "
        "more. The flat shape is characteristic of a model with little signal: "
        "predictions are pulled toward the base rate at both ends.\n\n"
        "Built by build/fig_curves_html.py.")


# =================================================== F · WHAT THE MODEL SAYS
s = add("Part 3 · Modelling", "What the model says drives churn", RED)
picture(s, "13_drivers.png", top=Inches(1.72), height=Inches(3.35))
bullets(s, [
    ("The shipped model is a logistic regression, so it can simply be read. "
     "Each bar is one input and the direction the model applies it.", INK, True),
    "",
    ("Taken at face value it is a briefing.", INK, True),
    "  Pro-plan customers stay. Partner-sourced customers leave. Customers "
    "holding several subscriptions leave. Every one of those is plausible, and "
    "a product team could act on all three tomorrow.",
    "",
    ("The dashed lines are why I am not reporting any of it.", RED, True),
    "  I shuffled the churn labels 500 times, refitting each time, and recorded "
    "the largest coefficient. Chance reaches 1.68. The largest real coefficient "
    "is 1.45. Not one input in the model is as strong as what chance produces "
    "from labels that mean nothing.",
    "",
    "  Seven of the 86 inputs clear p = 0.05 individually. Testing 86 inputs, "
    "chance delivers about 4. Asked properly — is any coefficient larger than "
    "chance reaches — the answer is p = 0.22.",
], top=Inches(5.18), size=13, space=4)
footnote(s, "There is no driver list to hand the product team. Not weak "
            "drivers: none.")
note(s, "Replaces the SHAP slide, which answered a question nobody asked. The "
        "brief asks what the model says is important, so answer that first, "
        "then say why it cannot be reported.\n\n"
        "Reading coefficients directly is also the honest choice here. The "
        "shipped rung is Logistic L2 C=1 on standardised inputs, so the "
        "coefficients are the explanation. SHAP would add a layer of machinery "
        "over a linear model and change nothing.\n\n"
        "The comparison that matters is against shuffled labels, not against "
        "zero. Any fitted model produces a clean ordered ranking; the question "
        "is whether it is larger than the ranking you get from nothing. Here it "
        "is not: largest real coefficient 1.45, chance reaches 1.68 at the 95th "
        "percentile, family-wise p = 0.222 over 500 refits.\n\n"
        "If asked why the coefficient signs look stable across refits: five-fold "
        "folds share 80% of their rows, so sign agreement across them is close "
        "to guaranteed and is not evidence. The shuffle test is the one that "
        "carries weight.\n\n"
        "Built by build/fig_drivers_html.py, tables in "
        "outputs/reports/coef_null.csv and coef_stability.csv.")

# ========================================== H · CHURN IS NOT ACCELERATING
s = add("Part 4 · Recommendation 1", "Do not go looking for what changed in 2024",
        RED)
picture(s, "12_calendar_null.png", top=Inches(1.66), height=Inches(4.30))
bullets(s, [
    ("Churn looks like it is accelerating. About 5 customers in every 100 left "
     "each month in 2023, rising to 22 in 100 by December 2024. A 2.8x rise per "
     "year, p = 2e-16, and by far the strongest number in this study.",
     INK, True),
    "",
    ("I rebuilt the file with the churn dates replaced by random draws and "
     "nothing else changed. It produced the same rise.", RED, True),
    "  x2.76 a year against our x2.79. Every one of the 24 months sits inside the "
    "range those rebuilds produce, and our result lands at the 58th percentile "
    "of pure noise. So there is nothing here to investigate, and that is the "
    "recommendation.",
], top=Inches(6.10), size=13, space=4)
footnote(s, "A small p-value only rejects one null. Here it rejected “the rate "
            "is flat”, which was never the question.")
note(s, "Say it in three sentences and stop. Churn looks like it is "
        "accelerating, 2.8x a year, p = 2e-16. I rebuilt the file with random "
        "churn dates and nothing else changed, and it produced 2.76x. The trend "
        "belongs to the file, so there is nothing to investigate.\n\n"
        "Everything below is backup, only if pushed.\n\n"
        "WHY RANDOM DATES WERE THE NULL. Take each churn date and ask where it "
        "falls between that customer's signup and the last day of the extract. "
        "Real churn clusters somewhere - early if onboarding fails, late if it "
        "is renewal-driven. This is flat across all ten slices, KS p = 0.92.\n\n"
        "WHY A RANDOM DATE MAKES THE RATE CLIMB. A customer's chance of churning "
        "in a given month is roughly one over the months left before the file "
        "ends. That gets bigger every month, so the rate rises with nothing "
        "happening in the business.\n\n"
        "IF ASKED WHY IT DOES NOT REACH 100%. The null keeps each account's real "
        "number of churn events, so the roughly 30% who never churn stay that "
        "way. They sit in the denominator throughout, which caps the level "
        "without removing the trend.\n\n"
        "From generator.simulate_churn_dates and generator.calendar_hazard_null, "
        "200 rebuilds, seed 0, shown in notebook 16. Figure built by "
        "build/fig_calendar_null_html.py.")

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"{OUT}  ({OUT.stat().st_size / 1e6:.2f} MB, "
      f"{len(prs.slides._sldIdLst)} slides)")
