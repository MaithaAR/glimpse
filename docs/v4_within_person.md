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
