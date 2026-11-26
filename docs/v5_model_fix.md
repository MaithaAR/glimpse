# v5: fixing the model, with a design part and a confirmation part (registered 26 Nov 2026, before any run)

Branch `v4-candidate`, after the submission (tag `submission-final`). Nothing submitted, served or frozen
changes, and the locked eight are not scored. The result is a candidate for the 30 Nov talk.

## The problem with "keep going until it improves"

The author asked for the model to be improved and for the work not to stop until it is. Trying variants on
the same people until one wins proves nothing. So the 23 development people are split by their existing
outer folds, before anything is tried:

- Design people: outer folds 1, 2 and 3 (15 people, 315 scored meals). Any number of variants may be tried
  here. Every variant tried is logged in outputs/v5_model_fix/design_log.json, including the ones that lose.
- Confirmation people: outer folds 4 and 5 (8 people, 166 scored meals). No v5 variant is scored on them
  until the design is frozen in a commit. Then the frozen design is scored on them once.

Every prediction stays out of fold as before: a person is never in the training set of the model that
scores them. The 14 type-2 people stay training only.

Caveat stated now: the confirmation people are not as clean as the locked eight were. v1 was chosen with
them in the development set, and the v4 study (docs/v4_within_person.md) printed results pooled over all 23.
The confirmation is clean for the new v5 choices only.

## What may be tried in the design part

From the v4 audit's three causes, in any combination:

1. The per-person intercept: shrinkage, what it is fitted on, when it starts, whether it is frozen.
2. Meal content: carbs the user confirms (the logged carbs stand in for that), with the photo estimate as
   the value when nothing is confirmed. Both cases are scored: "confirmed carbs" and "photo only".
3. Label noise at the cut: training on the size of the rise, soft labels, or dropping near-cut training meals.
4. Person-centred inputs (v4's B1).
5. The fusion model class and its inputs, within the 23-person capacity (linear models, small boosted trees).

Not tried: anything that uses a held-out person's own labels except through the causal history the served
system already uses (earlier meals whose 2-hour window has closed).

## Selection rule in the design part

Score on design people, out of fold, with the per-person history replayed in time order:
pooled AUROC plus mean within-person AUROC (people with both outcomes). A variant is kept only if it raises
the sum over the current best by at least 0.01 and lowers neither part by more than 0.01. Ties go to the
simpler variant.

## Confirmation, run once after the design is frozen

Baseline: v1's shipped risk on the same meals (outputs/evaluation_v2/dev_answers.csv). v1 does not answer
meals without a photo; the comparison uses the meals v1 answers, and v5's numbers on all meals are given too.

- Primary: pooled AUROC of v5 minus v1 on the confirmation people, paired bootstrap over people, 2,000 draws.
- Co-primary: mean within-person AUROC, v5 minus v1, same people.
- "Confirmed": both differences are above zero and at least one 95% interval excludes zero.
- "Direction only": both above zero, neither interval excludes zero.
- "Not confirmed": anything else.

With 8 people the intervals will be wide; that is stated with the result. Brier score, ECE (15 equal-width
bins) and spikes among cleared meals at v1's cut-off rule are reported as secondary.

Prediction: the design part finds a gain of 0.02 to 0.05 in pooled AUROC and more within a person, mostly
from the intercept and from confirmed carbs. The confirmation shows the same direction, with intervals
that include zero.

## After the confirmation

No further variant is scored on the confirmation people. v1 stays served. Whatever the outcome, the design
log and the confirmation go into the talk as they are.

## Design frozen (26 Nov, before the confirmation people are scored)

56 variants were scored on the design people; all are in outputs/v5_model_fix/design_log.json. v1's shipped
risk on the same people: pooled AUROC 0.737, within-person 0.586 (299 answered meals).

What the design part showed:
- The post-hoc intercept trades one number for the other. Every version of it (stronger shrinkage, a running
  rate, frozen after 5 or 10 meals, a late start) moved pooled and within-person AUROC in opposite directions.
- Putting the person's history into the model as an input works better than adding it afterwards: the mean
  rise of the person's earlier labelled meals (`prev_rise`), with no offset at all.
- Trend numbers relative to the person's own earlier meals help (v4's B1 change).
- Dropping training meals within 5 mg/dL of the cut helps; 3 and 8 help less. Training on the rise itself and
  soft labels did not help. Boosted trees were worse.
- Confirmed carbs help most within a person, and with them the 4 photo inputs are better left out.

The frozen design, v5, is two logistic models with the same recipe (C tuned on the inner folds, per-fold
Platt map, no post-hoc intercept, training meals within 5 mg/dL of the cut dropped):

| | inputs | design people: pooled / within-person |
|---|---|---|
| v5, carbs confirmed by the user | 15 trend and profile + 7 person-centred trend + carbs + carbs relative to the person's usual + `prev_rise` (25) | 0.766 / 0.729 |
| v5, photo only | 15 trend and profile + 4 photo + 7 person-centred trend + `prev_rise` (27) | 0.761 / 0.680 |

When the user confirms nothing, the first model is not used; the second answers. A meal with no photo and no
confirmed carbs is answered by the second model with its photo inputs imputed (v1 refuses those meals).

These numbers are selected on the design people and are optimistic. The confirmation below is the estimate.
