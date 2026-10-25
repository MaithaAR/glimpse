# GLIMPSE: intermediate report
Before eating: a photo + the last 2 h of CGM (glucose sensor) + a short profile → **low risk**, **caution** or **can't judge** (with a reason).
Maitha Alhosani · MAAI7103 · October 2026 · `just install` → `just demo`

## 1. User and workflow
The user is an adult with prediabetes in a 10–14 day CGM programme, run by a prevention clinic or a CGM app. They want fewer spikes (a rise of 50 mg/dL or more within 2 h), but food logging stops after a few days, so the CGM only shows a spike once it is too late.

**At the meal:**

- They take one photo before eating.
- About 0.3 s later they get *caution* with a nudge (a smaller carb portion, or a walk), *low risk*, or *can't judge* with the reason.
- There is no logging.

**The buyer:** the programme or app, which keeps users engaged by replacing manual logging. Section 3 shows the photo performs on par with logged carbs. It is a wellness nudge, not a medical device.

**Feedback (mocked):**

- The CGM labels each meal 2 h later.
- The user can correct it in `just app` ("did not eat it", "ate part of it").
- Validated `FeedbackRecord`s are saved for review and retraining.

## 2. Work completed
**Data:**

- **Source:** CGMacros (PhysioNet, CC BY-NC-SA), 45 adults, downloaded by `just install`.
- **All four inputs reach the model:** photo, CGM, heart rate, profile.
- **Validation:** Pydantic checks each meal at load time.
- **Preprocessing:** 24 five-minute CGM slots ending at the meal.
- **Augmentation:** sensor noise; photo flips and crops.
- **Leakage controls:** scaling from training people only; no photo duplicated across groups.
- **Fine-tuning ledger:** 1,221 photos (2,021 with planned Nutrition5k) out of 10,000.
- **Splits:** by person, seed 20260927.

| Group | People | Scored meals | Spike rate |
|---|---|---|---|
| Train (14 healthy/prediabetes + 14 type 2 diabetes; the latter train only, never evaluated) | 28 | 547 | 48% |
| Validation / Test (test scored once) | 4 / 5 | 99 / 96 | 19% / 29% |
| Locked for the final (hash-sealed) | 8 | – | – |

**Main estimate:** 5-fold person cross-validation (**CV**) over all 23 non-locked healthy/prediabetes people (481 meals). It includes the validation and test people, so results *without* the test people are also reported.

**Models** (`just params`). Pretrained + fine-tuned: 27.8M built (49.8M with the planned model) of 1B.

| Model | Role | Parameters (limit) |
|---|---|---|
| SensorCNN: two 1-D convolutions (kernel = 25 min) over glucose, heart rate and a missing-data flag, + profile | from scratch; the required baseline, not deployed | 5,217 (50M) |
| ConvNeXt-Tiny (ImageNet-22k) + carbs head + 4-bin carb-range head | fine-tuned | 27.8M (200M) |
| Fusion: logistic regression on profile + 7 glucose/heart-rate trend numbers + photo outputs | classical | 20 |
| DINOv2-S photo check | *planned (estimate)* | 22M (200M) |

**Why these models:**

- **1-D CNN:** learns short rises and dips with few weights.
- **ConvNeXt-Tiny + carb-range head** (as suggested in the feedback): a CPU-friendly pretrained backbone, needed with about 1,000 photos.
- **Rules run first:** CGM complete and in range; photo readable, not blank, and similar to training photos (5 nearest neighbours). Frozen cut-offs give the answer.

**Training.** Seeds are fixed; settings are in `configs/default.yaml`, validated at start-up; per-epoch logs go to Loguru and a CSV.

- **CNN:** BCE loss, AdamW, warm-up + cosine schedule; 13 epochs (median of 4 inner folds); validation AUROC 0.662.
- **Photo model:** Huber loss on log(carbs) + cross-entropy on 4 carb ranges.
  - 15 epochs for the heads on the frozen backbone.
  - Then 2 epochs for the last stage, because the inner-fold error fell from 30.7 to 25.4 g. The checkpoint `e17` = 15 + 2.

<img src="../outputs/training/sensor_cnn_curves.png" width="49%"> <img src="../outputs/training/photo_convnext_curves.png" width="49%">

**Notes:**

- The CNN's validation loss is below its training loss because the validation people spike less.
- The photo model overfits from epoch 3.
- Carb error in CV (`outputs/photo_cv/photo_cv_summary.json`): 22.5 g, against 27.7 g for guessing the median. A frozen ConvNeXt + ridge scored 20.4 g, but its carb ranges were worse (F1 0.38 vs 0.61), so fine-tuning is kept.

**Pipeline.** `just demo`, `just predict` and `just app` share one path:

- **Input:** raw readings + photo + profile → `MealInput`.
- **Steps:** the training window code → rules → photo model → fusion → cut-offs.
- **Output:** a Pydantic `Answer` in JSON, e.g. `{"decision": "low risk", "risk": 0.108, "reason": "risk 11% is below the low-risk cut-off 14%", "carb_bin": "under 30 g", "carb_bin_prob": 0.58}`.
- **Bad input:** gives *can't judge* with the reason.
- **Packaging:** a Linux container runs offline; CI tests it.

## 3. Results
**Baselines.** AUROC (ranking quality; 0.5 = chance) is the primary metric. Ranges are 95%, from resampling people. On the test, every model is trained on the same 28 people.

| Model | CV AUROC (range) | CV without test people | Test AUROC |
|---|---|---|---|
| Profile only | 0.672 (0.567–0.764) | 0.659 | 0.765 |
| Profile + trend (logistic regression) | 0.686 (0.604–0.772) | 0.686 | 0.695 |
| Profile + SensorCNN | 0.689 (0.603–0.773) | 0.691 | 0.674 |
| **+ photo (main model)** | **0.724 (0.659–0.796)** | **0.719** | 0.742 |
| + logged carbs (reference) | 0.717 (0.648–0.796) | 0.715 | – |
| + shuffled photo (control) | 0.679 (0.590–0.766) | – | – |

Gradient boosting (0.689) and the Brier scores are in `results.md`.

**What this shows:**

- **The photo adds a small gain:** +0.037 (+0.003 to +0.073), and shuffling removes it. Without the test people the gain is similar, +0.033, but its range (−0.007 to +0.077) now crosses zero. Treat it as small and not yet certain.
- **The photo matches logged carbs:** +0.007 (−0.020 to +0.032).
- **The carb-range head carries most of the gain:** +0.020 (+0.005 to +0.035; a check added after the results).
- **The CNN ties the trend numbers.**
- **On the 5 test people:** main model − profile-only = −0.023 (−0.114 to +0.159). The two are indistinguishable.
- **Caveat:** the post-hoc checks were judged on a CV that includes the test people. No setting was tuned on the test alone.

**Calibration** (main model, test; 15 equal-width bins; meals per bin in the figure and in `results.md`):

- **ECE** (the gap between predicted and observed rate) is 0.129 (CV: 0.052).
- **It over-predicts:** in the 0.3–0.6 bins it says 45% where 30% spiked (46 meals). The likely cause is the type 2 diabetes training mix.
- **Effect:** meals move from *low risk* to *caution*.

<img src="../outputs/evaluation/fig3_reliability_test.png" width="38%">

**Targets** (*met* = the whole range meets the target):

| Target | CV / test | Status |
|---|---|---|
| Photo adds signal | +0.037 (+0.003 to +0.073) | met (not without the test people) |
| RQ1: photo within 0.03 of logged carbs | +0.007 (−0.020 to +0.032) | met |
| RQ2: CNN beats trend numbers | +0.002 (−0.016 to +0.022) | on track by the rule; in practice a tie |
| ECE ≤ 0.05 | 0.052 / 0.129 | at risk |
| Cost saving ≥ 10% | 9% / 7% | at risk |
| Meals cleared ≥ 20% | 14% / 3% | at risk |
| Spikes among cleared ≤ 10% | 12% / 0% | at risk |
| *Can't judge* ≤ 20% | 11% / 16% (9% if a meal without a photo got an average estimate) | met |
| Response time p95 ≤ 300 ms (Mac CPU, `just demo`) | 283–286 ms | met (close to the limit) |

**On the test:** the app answered *caution* for 78 meals, *low risk* for 3, and *can't judge* for 15 (6 without a photo, 7 borderline, 2 unfamiliar photos). Safe, but not yet useful.

**Cut-offs:**

- The planned rule chose 0.31 on the 4 validation people. It failed its stability check, because those people spike rarely. So, **before the test was opened**, the cut-offs were derived from the costs alone, with no data.
- **The costs** (if it spikes / if not) are: low risk 5/0, caution 0/1, can't judge 2/0.5. Low risk is the cheapest answer when p < 0.143, and caution when p ≥ 0.20. In between, *can't judge* is cheapest.
- The cut-offs are committed with a hash, which the app checks.

**Test cases** (`outputs/demo/cases.md`; first matching meal):

| Case | Input | Expected | Actual |
|---|---|---|---|
| Normal | test meal at 15:32 (glucose then fell) | low risk | **low risk** (11%) |
| Abstain | test meal at 19:03, last 40 min of CGM removed | can't judge | **can't judge**: "0 readings in the last 30 min" |
| Abstain | a plain white picture | can't judge | **can't judge**: "photo is almost blank" |
| Borderline | test meal at 12:51 (no spike) | can't judge | **can't judge**: "risk 17% is borderline" |
| Wrong | test meal at 12:47 (rose 35 mg/dL, no spike) | low risk | **caution** (48%) |

**The wrong case:** the photo shows a large carb portion ("60–89 g, 92% sure"), but this person reacts mildly. Personal response is what the final stage targets.

## 4. Changes, risks and plan
**Proposal feedback → response:**

| Feedback | Response |
|---|---|
| Too dense | Short report, plain terms |
| Weak framing | User moment, buyer, feedback loop |
| Photo model weak | ConvNeXt-Tiny + carb-range head (+0.020) |
| Scratch CNN not justified | Kept as the required baseline; it ties the trend numbers, so it is not deployed |
| Fusion overkill | Logistic regression |
| Small dataset | Person-level CV with ranges |

**Changes from the proposal:**

- A nearest-neighbour check + blank-photo rule instead of DINOv2. It flags 3.1% of real meal photos and every synthetic non-meal image.
- Conformal intervals, W&B, Nutrition5k and the user test move to the final.

**Added after the results** (labelled in `docs/evaluation_protocol.md`): the regularisation grid, the ridge baseline, the carb-range check, the cut-offs, and the CV without the test people.

**Risks:**

- **The type 2 diabetes training mix** causes over-prediction.
- **One photo seed.**
- **The two photo outputs disagree** on 51 of the 90 meals with a photo, so users see only the carb range.

**Final submission:**

1. Recalibrate and down-weight the type 2 diabetes people. Choose the cut-offs on validation with a stability check. Run the locked 8 people once as the clean held-out result.
2. Use 3 seeds, and give the CNN a fair try (longer window). Make the two photo outputs agree. Add Nutrition5k.
3. Compare DINOv2 against the nearest-neighbour check, and add tiered escalation.
4. Robustness checks, latency breakdown, explainability, tracker exports, and a user test with 3–5 people.
