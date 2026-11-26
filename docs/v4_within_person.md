# v4 candidate: ranking meals within a person (registered 26 Nov 2026, before any run)

Written after the final submission (tag `submission-final`, commit a466e19) on the branch `v4-candidate`.
Nothing here changes the submitted report, the served v1, the frozen v2 or any locked number. The locked
eight are spent: no model in this study is scored on them. Every result is development only and is labelled
a candidate.

## Why

The report's third weakest point: within a person, v1 ranks meals at a mean AUROC of 0.60 with the intercept
(about 0.68 on the raw fusion; the intercept costs 0.078). Most of the system's edge is each person's base
rate. A user needs the other thing: is this meal worse than my usual one?

The council (26 Nov) asked for a label audit first, because the label is a rise above the pre-meal baseline
and the weak within-person ranking may come from the label and not from the model.

## Part A: label audit (descriptive, no model fitted)

Meals: the 460 development meals v1 answers with a raw risk (outputs/evaluation_v2/dev_answers.csv). People:
those with both outcomes among them (22 in the report).

- A1. Mean within-person AUROC of single signals, direction fixed here: logged carbs (higher means spike),
  last pre-meal glucose (lower means spike), 30-minute slope (lower means spike), pre-meal baseline (lower
  means spike). Prediction: logged carbs reach at least 0.65; last glucose at least 0.55.
- A2. Share of meals whose rise is within 5 mg/dL of the 50 mg/dL cut, per person and overall; mean
  within-person AUROC of v1's raw risk on all meals and with those meals removed. Prediction: removing them
  raises it by less than 0.03 (label noise at the cut is not the main cause).
- A3. Mean within-person Spearman correlation of v1's raw risk with the rise and with the absolute 2-hour
  peak (baseline plus rise). Prediction: the correlation with the peak is the higher of the two, because the
  model reads the pre-meal level.

No decision depends on Part A. It is reported whatever it shows.

## Part B: two candidates, one run each

Same outer folds, inner folds, C grid and seed as the fusion study (src/glimpse/training/fusion.py). The
baseline arm is refitted in the same loop and must reproduce the committed `photo` column of
outputs/fusion/fusion_oof.csv to 1e-9; if it does not, the study stops and reports that.

- B1, person-centred fusion. v1's 19 inputs plus 7 new ones: each of the 7 trend numbers minus the mean of
  that number over the same person's earlier meals in time order (0 for a person's first meal). Computed the
  same way for training and held-out people. It uses no labels, so it needs no history file.
- B2, SensorCNN with a within-person ranking term. The same 5,217-parameter network, the same epochs per fold
  as the committed cross-validation (outputs/cv/cv_summary.json), seed 0. Loss: binary cross-entropy plus
  1.0 times a pairwise logistic loss, softplus(score of the non-spike meal minus score of the spike meal),
  over up to 512 random pairs of same-person training meals with different labels, drawn at every step.
  Baseline: the committed seed-0 SensorCNN predictions (outputs/cv/oof_predictions.csv).

Primary endpoint, one per candidate: mean within-person AUROC over people with both outcomes, candidate
minus its baseline, on raw out-of-fold scores (B1 on the 460 meals with a photo, B2 on all 481).
Interval: bootstrap over people of the per-person differences, 2,000 draws, seed 0.

Success, fixed here: a gain of at least +0.03 with the lower end of the 95% interval above zero, and a pooled
AUROC no more than 0.02 below the baseline's. Anything else is reported as "not shown".

Secondary, descriptive: pooled AUROC of B1 with the shipped intercept replayed (no Platt) against v1 under the
same replay; the number of people who rank better and worse.

Prediction: B1 gains between 0 and +0.03 and is not shown; B2 gains less than B1.

## What happens with the result

- Either way v1 stays served and the submitted report is not edited.
- A success is named "v4 candidate" and shown in the 30 Nov talk with this registration and the words
  "development only".
- No second variant is tried after the results are seen. Any follow-up is labelled post hoc.
- Two more items the council placed after the upload are registered separately before they run: a cold-start
  guard and per-person monitoring.

## Results (26 Nov, after the registration commit 3248006; a rerun after a formatting-only edit gave identical numbers)

outputs/v4_within_person/v4_within_person.json; `just v4-within-person`. Development only.

Part A, label audit (460 meals, 22 people with both outcomes):

| check | prediction | result |
|---|---|---|
| A1 logged carbs alone, within person | at least 0.65 | 0.708: met |
| A1 last pre-meal glucose alone (lower means spike) | at least 0.55 | 0.607: met. Baseline 0.609; the 30-minute slope 0.484, no signal |
| A2 meals within 5 mg/dL of the cut | removing them adds less than 0.03 | 9.3% of meals (0 to 27% per person); v1 raw goes from 0.680 to 0.720, +0.040: not met |
| A3 v1 raw risk against rise and against peak, within person | peak higher | rise 0.38, peak 0.20: not met |

What it says. Label noise at the cut costs more than predicted, about 0.04 of within-person AUROC. The model
follows the rise, not the absolute level, so the weak ranking is not the model reading the wrong target. The
largest loss is the intercept: v1 ranks at 0.680 raw and 0.602 as shipped. Logged carbs alone rank at 0.708,
above the whole raw fusion, so the missing within-person signal is meal content, which the photo does not
deliver.

Part B, candidates:

| candidate | within-person AUROC (baseline) | gain (95% over people) | pooled AUROC (baseline) | verdict |
|---|---|---|---|---|
| B1 person-centred fusion | 0.693 (0.680) | +0.013 (−0.002, +0.030); 11 people better, 9 worse | 0.749 (0.726) | not shown |
| B2 SensorCNN with the ranking term | 0.612 (0.607) | +0.005 (−0.025, +0.034); 12 better, 9 worse | 0.647 (0.678) | not shown |

The refitted baseline reproduced fusion_oof.csv to 7e-16. Both predictions were met: B1 gained between 0 and
+0.03 and B2 gained less. Secondary, descriptive: with the shipped intercept replayed without Platt, B1 pools
at 0.774 against 0.763 for v1.

Verdict by the registered rule: neither candidate is shown to rank meals better within a person. B1's pooled
AUROC is 0.023 higher on the raw score, but that was not the endpoint, so it is reported as an observation and
not as a result. The ranking term bought nothing within a person and cost 0.031 pooled.

Not done, as registered: no second variant. One note for any later work: a cold-start guard was already
counted in report_metrics.json (cold_start_what_if), including on the locked eight, so it cannot be registered
fresh; it stays a what-if.
