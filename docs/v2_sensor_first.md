# v2, the sensor-first product: pre-registration (26 Nov 2026, committed before any v2 run)

Written by the assistant on 26 Nov after the author chose, in the session, to ship the product the gate's
Paragraph B describes. Nothing below has been run when this file is committed; the first v2 run is a
later commit.

## Why
The week-4 gate (docs/gate_s8.md, 1 Nov) failed: the photo's AUROC gain over profile + trend was +0.025
(-0.010, +0.061). Its pre-written Paragraph B shipped verbatim and says the pre-meal answer rests on the
glucose trend and profile, with the photo head kept as a carb-entry assist. The 5 Nov amendment instead
regenerated the photo-arm fusion, so the frozen v1 uses the photo estimate as 4 of its 19 inputs and
refuses a meal without a usable photo. v2 builds what Paragraph B said. v1 stays in the repository,
unchanged, as the product the locked eight measured.

## What v2 is (fixed now)
1. Risk model: the gate's own comparator arm, unchanged. Logistic regression (median imputation,
   standard scaling, L2) over the 15 trend + profile inputs of `trend_features` (last glucose, 30- and
   60-min slopes, window mean, spread and rise, mean heart rate, the 8 profile inputs). No photo input,
   no typed carbs. C from the committed grid by the committed inner-fold recipe. Heart rate stays: the
   feature set is the gate's, and dropping it would be a choice made after seeing the ablation.
2. Calibration and decisions: exactly v1's procedure. Per-fold Platt on inner out-of-fold risk; the
   per-person intercept on earlier labelled meals under the two-hour gate; m and the cut-offs chosen
   together by minimum expected cost at the declared prices (`choose_m_and_cutoffs`). Rules R1-R13 as
   committed.
3. The photo becomes the carb assist. When a usable photo exists, the answer carries the photo model's
   grams and carb range with its confidence, for the user to confirm. A missing, unreadable, blank or
   unfamiliar photo no longer refuses the meal: the risk is still answered, the carb assist is withheld,
   and a flag names why (`carbs_unavailable_<check>`). Typed carbs are echoed, never used in the risk.
4. Serving switch: `prototype.product` in the config selects `sensor_first` (v2) or `photo_fusion` (v1).
   v2's frozen files are new (`checkpoints/sensor_first/`); every v1 file stays byte-identical.

## Evaluation (development only)
- Same partition v2, same five outer folds and inner folds, same seed, same bootstrap (2,000 person
  draws). Every development meal answered by its fold's v2 Pipeline in deployment order, as in v1.
  Output: outputs/evaluation_sensor_first/.
- Endpoints, all reported as measured:
  (a) AUROC with the person-clustered interval, log-loss, Brier, ECE (15 equal-width bins);
  (b) the four usefulness targets with the protocol's labels;
  (c) paired v2 minus v1 AUROC on the meals both answer, with a non-inferiority margin of 0.03 (the
      RQ1 margin), descriptive;
  (d) answered share: v2 is expected to answer the 21 development meals v1 refused for lack of a photo.
- Expected before the run: AUROC close to the gate's no-meal arm with the intercept (0.741 in the
  fair-baseline replay); usefulness close to v1's.
- The locked eight are never scored with v2. v1's single locked result stays the only clean measurement
  and is reported as v1's.

## Decision rule
v2 ships whatever the development result: Paragraph B decided that on 1 Nov. The result is reported as
measured with no change to v2 after it is seen. If v2 fails non-inferiority to v1 (paired lower bound
below -0.03), the report says so in the opening and names v1 as the more accurate model and v2 as the
product the pre-registration required.

## Freeze
After the evaluation, the deployed v2 values come from the all-development replay exactly as the 5 Nov
amendment did for v1: C by the committed recipe with the five dev folds as holds; Platt on the pooled
out-of-fold replay; m and cut-offs by `choose_m_and_cutoffs`. Written once to
checkpoints/sensor_first/ with a sha-guarded thresholds file; a guard refuses to overwrite it.
