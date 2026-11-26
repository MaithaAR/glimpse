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

## Confirmation (26 Nov, after the freeze commit c74d69c; scored once)

outputs/v5_model_fix/confirmation.json. 8 people, 161 meals v1 answers (166 scored). The first attempt was
killed by a time limit inside the bootstrap, before any number was printed or saved; the bootstrap code was
made faster and the run repeated. The code refuses a second scoring.

| on the 161 meals v1 answers | pooled AUROC | within-person AUROC (7 people) | Brier | ECE | cleared at v1's fold cut-offs: missed |
|---|---|---|---|---|---|
| v1 as shipped | 0.804 | 0.636 | 0.186 | 0.084 | 32: 3 |
| v5, photo only | 0.806 | 0.701 | 0.192 | 0.117 | 48: 6 |
| v5, carbs confirmed | 0.793 | 0.693 | 0.194 | 0.144 | 43: 7 |

| v5 minus v1 (95% over people) | pooled | within-person | verdict by the registered rule |
|---|---|---|---|
| photo only | +0.003 (−0.059, +0.056) | +0.065 (−0.020, +0.142); 4 of 7 people better | direction only |
| carbs confirmed | −0.010 (−0.061, +0.040) | +0.056 (−0.032, +0.160); 5 of 7 better | not confirmed |

What this says, plainly:
- Ranking meals within a person improved in direction on people the design never saw: 0.636 to about 0.70.
  The interval includes zero, as predicted for 8 people.
- Pooled AUROC did not improve. The design part's pooled gain (0.737 to 0.76) did not carry over.
- Calibration got worse. Without the per-person offset the risk is less well centred on each person: ECE
  rises from 0.084 to 0.12 or more, and at v1's cut-offs v5 clears more meals and misses more of them
  (6 of 48 against 3 of 32).
- Confirmed carbs did not beat the photo-only model on these people, against the design part and against
  the prediction.

So v5 is not a better served model than v1. It ranks a person's own meals better and calibrates worse. The
piece worth keeping is the mechanism: history as a model input (`prev_rise`) and person-centred trend inputs
recover the within-person ranking that the post-hoc intercept destroys. A served version would need the
offset's calibration and v5's ranking together, and a new set of people to test it on. No further variant is
scored on the confirmation people.

## Part 2, registered 26 Nov before any run: v6 and a transfer check on people never scored

Aim: keep v5's ranking and get v1's calibration back.

Design, on the design people only (folds 1 to 3), logged as before. Allowed: the v5 photo-only model with a
per-person offset on top (shrinkage, frozen after N labelled meals, late start), and nothing else new. A
variant is kept only if, against v5 photo only on the design people, its Brier score is lower and the sum
of pooled and within-person AUROC is not more than 0.01 lower. The confirmation people (folds 4 and 5) are
not scored again.

Transfer check, run once after v6 is frozen in a commit. The 14 type-2 people have never been scored by any
model in this project; they were training data only. They are outside the product's scope (rule R15), so
this is a check of the recipe on unseen people, not a claim for people with diabetes. They are split into
two halves by sorted person id (odd and even positions). Each half is scored by models trained on the 23
development people plus the other half, with the development outer folds as the inner folds. Photo inputs
are the committed estimates from models that never saw the person (outputs/photo_cv/).

Three recipes are refitted the same way and compared on the same meals: the v1 recipe (label logistic with
v1's per-fold median m = 1.0 intercept), v5 photo only, and v6.

- Endpoints: pooled AUROC, mean within-person AUROC, Brier score; v6 minus the v1 recipe, bootstrap over the
  14 people, 2,000 draws, seed 0.
- "Transfers": within-person AUROC higher with an interval above zero, and neither pooled AUROC nor Brier
  worse beyond its interval (pooled interval not wholly below zero, Brier interval not wholly above zero).
- "Direction only": within-person higher, interval includes zero, and the other two not worse as above.
- "Does not transfer": anything else.

Prediction: direction only. The type-2 people spike more often and their meals differ, so the size of any
gain is not expected to match.

## Part 2 result: no v6 (26 Nov)

18 offset variants on top of v5 photo only were scored on the design people (design_log.json). All lowered
the Brier score (0.1707 to between 0.1645 and 0.1694), and all lowered the sum of pooled and within-person
AUROC by more than the 0.01 allowed (best: frozen after 10 meals with m = 8, 1.4301 against 1.4411). By the
registered rule none is kept. The transfer check on the type-2 people was not run and stays unused.

## Part 3, registered 26 Nov before any run: new inputs (v7), then the transfer check

The author asked for every reasonable method and a literature search. Two sources read on 26 Nov:
Kondratyeva et al. (arXiv 2609.24453) on CGMacros lunches report that meal composition is the dominant
predictor (correlation with the 2-hour response 0.24 from clinical variables alone, 0.59 with image-estimated
macronutrients, 0.74 with the gut microbiome added, 0.80 with true macronutrients); a public XGBoost write-up
on the Hall data uses 24-hour and 4-hour CGM statistics with macronutrients. GLIMPSE uses carbs only, four
profile fields and a 2-hour window. The data it already has holds much more.

Design on the design people only (folds 1 to 3), every variant logged. Input families that may be added to
v5, alone or together:

1. Meal composition from the photo: protein, fat, fibre and calories estimated by ridge heads on the frozen
   photo features, fitted without the person (as the committed carb ridge baseline is).
2. Meal composition confirmed by the user: the logged protein, fat, fibre and calories next to the carbs.
3. More of the profile: sex, fasting insulin, triglycerides, HDL, insulin resistance (HOMA-IR).
4. Context before the meal: minutes since the previous meal, activity in the last hour, heart-rate slope,
   meal type, and CGM statistics over the last 4 and 24 hours (mean, spread, time above 140 mg/dL).
5. Gut microbiome summaries (a few principal components fitted on training people only).
6. Model class: logistic, small boosted trees, and their average.

Selection as in the first design part (sum of pooled and within-person AUROC up by at least 0.01 per kept
step, neither part down by more than 0.01), with one addition: the Brier score must not be worse than v5's.
Two frozen models again: photo only (families 1, 3, 4, 5) and meal confirmed (family 2 as well).

Then the transfer check registered in Part 2 runs once, with v7 in the place of v6, on the 14 type-2 people.
The confirmation people (folds 4 and 5) are also reported for v7, labelled as a second look at people already
used once for v5, so not clean.

Prediction: meal composition and the longer CGM history give the largest gains; the microbiome does not help
at 23 people; the design-part gain is at least 0.03 in pooled AUROC, and less than half of it transfers.

## Part 3 design frozen (26 Nov, before the transfer check)

104 variants are now in the design log (48 in Part 3). On the design people, against v5:

| added to v5 photo only (0.761 / 0.680, Brier 0.171) | pooled / within-person | Brier | kept? |
|---|---|---|---|
| gut-health principal components (3) | 0.779 / 0.682 | 0.166 | yes, +0.021 |
| profile: sex, insulin, triglycerides, HDL, HOMA-IR | 0.771 / 0.684 | 0.167 | +0.014 alone; +0.002 on top of the microbiome, so no |
| 4-hour and 24-hour CGM statistics | 0.768 / 0.679 | 0.168 | no, +0.005 |
| protein, fat, fibre, calories estimated from the photo | 0.757 / 0.674 | 0.172 | no, worse |
| context (time since the last meal, activity, meal type) | 0.756 / 0.669 | 0.172 | no, worse |
| a CNN trained to forecast the next 2 hours on all CGM data (no meal labels), its forecast as an input | 0.763 / 0.684 | 0.171 | no, +0.005 |
| boosted trees; logistic and trees averaged | 0.735 / 0.651; 0.768 / 0.686 | 0.173; 0.166 | no on top of the microbiome |

| added to v5 carbs confirmed (0.766 / 0.729, Brier 0.168) | pooled / within-person | Brier | kept? |
|---|---|---|---|
| logged protein, fat, fibre, calories, carb share | 0.774 / 0.744 | 0.168 | yes, +0.023 |
| the five profile inputs | 0.787 / 0.745 | 0.162 | yes, +0.014 |
| logistic and trees averaged | 0.792 / 0.763 | 0.160 | yes, +0.023 |
| gut-health components | 0.792 / 0.740 | 0.160 | no |

Frozen v7 (no post-hoc intercept, per-fold Platt, near-cut training meals dropped, as v5):
- photo only: v5's 27 inputs plus 3 gut-health components, logistic. Design people 0.779 / 0.682.
- meal confirmed: carbs, protein, fat, fibre, calories, carb share, carbs relative to the person's usual, the
  15 trend and profile inputs, 7 person-centred trend inputs, `prev_rise` and 5 more profile inputs; the
  average of a logistic model and small boosted trees. Design people 0.792 / 0.763.

v1 as shipped on the same people: 0.737 / 0.586. The microbiome was predicted not to help and did on the
design people; with 23 people that is the input most likely to be chance, and the transfer check is where
that shows. The gains are steady and small. Nothing in the data gives a large jump: the published ceiling
with true macronutrients and the microbiome is a correlation of 0.80 on lunches.

## Transfer check (26 Nov, after the freeze commit b2c080b; scored once)

outputs/v5_model_fix/transfer_check.json. The 14 type-2 people, 261 scored meals, spike rate 58.6%. Every
recipe is refitted the same way on the 23 development people plus the other half of the type-2 people.

| recipe | pooled AUROC | within-person AUROC | Brier | ECE |
|---|---|---|---|---|
| v1 recipe (post-hoc intercept, m = 1) | 0.791 | 0.787 | 0.186 | 0.085 |
| v5, photo only | 0.791 | 0.834 | 0.182 | 0.062 |
| v7, photo only | 0.797 | 0.837 | 0.179 | 0.052 |
| v7, meal confirmed | 0.825 | 0.857 | 0.165 | 0.097 |

| minus the v1 recipe (95% over the 14 people) | pooled | within-person | Brier | verdict |
|---|---|---|---|---|
| v5, photo only | −0.001 (−0.025, +0.039) | +0.048 (+0.021, +0.082) | −0.005 (−0.018, +0.006) | transfers |
| v7, photo only | +0.006 (−0.023, +0.046) | +0.050 (+0.026, +0.079) | −0.008 (−0.021, +0.004) | transfers |
| v7, meal confirmed | +0.034 (−0.011, +0.086) | +0.071 (+0.026, +0.121) | −0.022 (−0.042, +0.001) | transfers |

By the registered rule all three transfer: within-person ranking is higher with an interval above zero, and
neither pooled AUROC nor the Brier score is worse. The prediction was "direction only"; the result is
stronger than predicted.

Second look at the confirmation people (folds 4 and 5, second_look.json; already used once for v5, so not
clean): v7 photo only 0.814 pooled and 0.719 within-person against v1's 0.804 and 0.636, within-person
+0.083 (−0.009, +0.170), Brier 0.193 against 0.186. v7 meal confirmed 0.843 and 0.779, within-person +0.143
(+0.012, +0.300), Brier 0.178.

What can be said:
- On three separate groups of people (15 design, 8 confirmation, 14 type-2) the new recipe ranks a person's
  own meals better than v1's recipe. On the one group never used for anything, the interval excludes zero.
- The gain comes from moving the person's history into the model (`prev_rise`), person-centred inputs and
  dropping near-cut training meals (v5). The microbiome adds little on top (v7 photo only against v5: +0.003
  within-person on the type-2 people), as first predicted.
- Pooled AUROC does not move for the photo-only model. With the meal confirmed it rises by about 0.03 to 0.04
  in all three groups, never with an interval above zero.
- Calibration: better than the v1 recipe on the type-2 people, worse than shipped v1 on the confirmation
  people. Mixed, so not claimed.

What cannot be said:
- That v7 beats the shipped v1 on the product's own population on clean data. The type-2 people are outside
  the product's scope and spike far more often; the comparison there is recipe against recipe, not the
  shipped files. The locked eight stay spent.
- That the "meal confirmed" model is a product. It needs the user to confirm macronutrients, which the
  product was built to avoid; it is the measured value of that information.

v1 stays served. v7 is the candidate for the talk.

## Part 4, registered 27 Nov before any run: does the photo help within a person under the v5 recipe?

The report's photo gate (docs/gate_s8.md) tested pooled AUROC under v1's recipe and was not met. It never
asked the within-person question, and v1's post-hoc intercept scrambles within-person ranking, which would
hide a meal-level signal. The photo is a meal-level input, so the fair test of it is within a person, under
a recipe without that intercept.

Contrast: v5 photo only, exactly as frozen, against the same model without its 4 photo inputs. Control: the
4 photo inputs permuted among each held-out person's own meals (seed 0), which should remove any gain.

People, all reported: the 14 type-2 people (primary; they were scored once for the recipe comparison, and
this contrast was not computed then), the 8 confirmation people, the 15 design people. The design part once
printed a related pair (a variant with and without the photo inputs, also with a history-rate input), so the
design people are not blind to this; the other two groups are.

Endpoint: mean within-person AUROC, with photo minus without, bootstrap over people (2,000 draws, seed 0).
Pooled AUROC is reported beside it.
- "The photo adds within a person": the primary interval is above zero and the permuted control's is not.
- "Direction only": the point is above zero, the interval includes zero.
- "No": anything else.

Prediction: +0.02 to +0.05 within a person on the type-2 people, direction only; pooled unchanged.
Whatever it shows goes into the report next to the original gate, which stands as it was.
