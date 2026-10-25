# GLIMPSE: intermediate report
Before eating: a photo + the last 2 h of CGM glucose and heart rate + a short profile → **low risk**, **caution** or **can't judge** (with a reason).
Maitha Alhosani · MAAI7103 · October 2026 · `just install` → `just demo`

**Bottom line:**

- **All four inputs** run end to end (the rubric requires one).
- **The photo matches logged carbs** for predicting spikes, so no food logging is needed.
- **The system is safe but not yet useful:** on the test it clears only 3% of meals, because risk is over-predicted.

## 1. User and workflow
**User:** an adult with prediabetes in a 10–14 day CGM programme (clinic or app). They want fewer spikes (a rise of 50 mg/dL or more within 2 h), but stop logging food after a few days, so the CGM only shows a spike once it is too late.

**Workflow:**

- They take one photo before eating.
- About 0.3 s later they get *caution* with a nudge (a smaller carb portion, or a walk), *low risk*, or *can't judge* with the reason.

**Buyer:** the programme or app, which replaces manual logging. It is a wellness nudge, not a medical device.

**Feedback (mocked):** the CGM labels each meal 2 h later. The user corrects it in `just app` ("did not eat it"). Validated `FeedbackRecord`s are saved for retraining.

## 2. Work completed
**Tooling:**

- `uv` with Python 3.14 and a lock file; `just` recipes; strict Pydantic for config, data and outputs; Loguru logs; ruff + 82 tests.
- CI on Linux, plus an offline Docker self-test.
- No dependency blockers on Python 3.14. PyTorch on Linux uses uv's CPU index.

**Data:**

- **Source:** CGMacros (PhysioNet, CC BY-NC-SA), 45 adults, downloaded by `just install`.
- **Validation:** Pydantic checks each meal at load time.
- **Window:** 24 five-minute CGM slots ending at the meal.
- **Augmentation:** sensor noise; photo flips and crops.
- **Leakage controls:** scaling from training people only; no photo duplicated across groups (`outputs/data/`).
- **Splits:** by person (seed 20260927):

| Group | People | Scored meals | Spike rate |
|---|---|---|---|
| Train: 14 healthy/prediabetes + 14 type 2 diabetes (the latter train only) | 28 | 547 | 48% |
| Validation / Test (test scored once) | 4 / 5 | 99 / 96 | 19% / 29% |
| Locked for the final (hash-sealed) | 8 | – | – |

**Main estimate:** 5-fold person cross-validation (**CV**) over the 23 non-locked healthy/prediabetes people (481 meals). It includes the validation and test people, so the cleaner **CV without the test people** is also reported.

**Models and budget** (`just params`):

| Model | Role | Used / limit |
|---|---|---|
| SensorCNN: two 1-D convolutions (25-min kernel) over glucose, heart rate and a missing-data flag, + profile | from scratch | 5,217 / 50M |
| ConvNeXt-Tiny (ImageNet-22k) + carbs head + 4-bin carb-range head | fine-tuned | 27.8M / 200M |
| Fusion: logistic regression on 15 profile/trend + 4 photo inputs | classical | 20 parameters |
| DINOv2-S photo check | *planned (estimate)* | 22M / 200M |
| **Fine-tuning photos** | | 1,221 used (2,021 planned) / 10,000 |

**Why these models:**

- **The scratch CNN** tests whether the raw 2 h trace holds more than 7 hand-made trend numbers. Small kernels learn rises and dips with few weights, which suits about 600 training meals.
- **ConvNeXt-Tiny with a carb-range head** follows the proposal feedback. A pretrained backbone is needed with about 1,000 photos.
- **Plain rules run first:** CGM complete and in range; photo readable, not blank, and similar to the training photos (5 nearest). Frozen cut-offs then give the answer.

**Training.** Seeds are fixed; settings are validated at start-up; per-epoch logs go to Loguru and a CSV.

- **CNN:** BCE loss, AdamW, warm-up + cosine; 13 epochs (median of 4 inner folds); validation AUROC 0.662.
- **Photo model:** Huber loss on log(carbs) + cross-entropy on 4 carb ranges.
  - The backbone is frozen first, because about 1,000 photos would overfit it: 15 epochs train the heads.
  - The last stage then trains for 2 epochs, because the inner-fold error fell from 30.7 to 25.4 g. Validation error is 26.2 g.

<img src="../outputs/training/sensor_cnn_curves.png" width="49%"> <img src="../outputs/training/photo_convnext_curves.png" width="49%">

**Notes:**

- The CNN's validation loss is below its training loss because the validation people spike less.
- The photo model overfits from epoch 3.
- In CV, carb error is 22.5 g against 27.7 g for guessing the median. Frozen ConvNeXt + ridge scored 20.4 g but had worse carb ranges (F1 0.38 vs 0.61), so fine-tuning is kept.

**Pipeline.** `just demo`, `just predict` and `just app` share one path: `MealInput` → the training window code → rules → photo model → fusion → cut-offs → a Pydantic `Answer` in JSON, for example:

`{"decision": "low risk", "risk": 0.108, "reason": "risk 11% is below the low-risk cut-off 14%", "carb_bin": "under 30 g", "carb_bin_prob": 0.58}`

Bad input gives *can't judge* with the reason.

## 3. Results
**Baselines.** AUROC (ranking quality; 0.5 = chance) is primary. Ranges are 95%, from resampling people. On the test, every model is trained on the same 28 people. Gradient boosting (0.689), Brier scores and all ranges are in `outputs/evaluation/results.md`.

| Model | CV AUROC (range) | CV without test people | Test AUROC |
|---|---|---|---|
| Profile only | 0.672 (0.567–0.764) | 0.659 | 0.765 |
| Profile + trend (logistic regression) | 0.686 (0.604–0.772) | 0.686 | 0.695 |
| Profile + SensorCNN | 0.689 (0.603–0.773) | 0.691 | 0.674 |
| **+ photo (main model)** | **0.724 (0.659–0.796)** | **0.719** | 0.742 |
| + logged carbs (reference) | 0.717 (0.648–0.796) | 0.715 | – |
| + shuffled photo (control) | 0.679 (0.590–0.766) | – | – |

**What this shows:**

- **The photo adds a small gain:** +0.037 (+0.003 to +0.073), and shuffling the photos removes it. Without the test people the gain is +0.033 (−0.007 to +0.077), so it is small and not yet certain.
- **The photo matches logged carbs:** +0.007 (−0.020 to +0.032).
- **The carb-range head carries most of the gain:** +0.020 (+0.005 to +0.035).
- **The scratch CNN ties the trend numbers,** confirming the proposal concern. It is kept as the required baseline and not deployed.
- **On the 5 test people:** main model − profile-only = −0.023 (−0.114 to +0.159). The two cannot be told apart.
- **Caveat:** several checks were added after the results, on a CV that includes the test people (labelled in `docs/evaluation_protocol.md`). Nothing was tuned on the test.

**Calibration** (main model, test; 15 equal-width bins; meals per bin in the figure and in `results.md`):

- **ECE** (the gap between predicted and observed rate) is 0.129 (CV 0.052).
- **It over-predicts:** in the 0.3–0.6 bins it says 45% where 30% spiked (46 meals).
- **Likely cause:** type 2 diabetes people (59% spikes) in training. To be tested in the final.
- **Effect:** meals move from *low risk* to *caution*.

<img src="../outputs/evaluation/fig3_reliability_test.png" width="38%">

**Targets.** *met* = the whole range meets the target.

| Target | CV / test | Status |
|---|---|---|
| Photo adds signal | +0.037 (+0.003 to +0.073) | met (not without the test people) |
| RQ1: photo within 0.03 of logged carbs | +0.007 (−0.020 to +0.032) | met |
| RQ2: CNN beats trend numbers | +0.002 (−0.016 to +0.022) | on track by the rule; in practice a tie |
| ECE ≤ 0.05 | 0.052 / 0.129 | at risk |
| Cost saving ≥ 10%; meals cleared ≥ 20%; spikes among cleared ≤ 10% | 9%, 14%, 12% / 7%, 3%, 0% | at risk |
| *Can't judge* ≤ 20% | 11% / 16% | met |
| Response time p95 ≤ 300 ms (Mac CPU, `just demo`) | 286 ms (`outputs/demo/demo_summary.json`) | met (close to the limit) |

**How the 96 test meals were answered:**

- *caution* 78, *low risk* 3, *can't judge* 15;
- of the 15: 6 had no photo, 7 were borderline, 2 photos were unfamiliar;
- `results.md` reports 9% *can't judge* because it gives meals without a photo an average estimate instead of refusing them.

**Cut-offs (a deviation from rubric 5b):**

- The rule chose 0.31 on the 4 validation people. It failed its stability check, because they spike rarely.
- Before the test was opened, the cut-offs were instead derived from the costs alone, with no data, so nothing was tuned on the test.
- With costs (if spike / if not) of low risk 5/0, caution 0/1 and *can't judge* 2/0.5: *low risk* when p < 0.143, *caution* when p ≥ 0.20, *can't judge* in between.
- They are hash-sealed and checked by the app. In the final, the cut-offs will be chosen on validation, after recalibration.

**Test cases** (`outputs/demo/cases.md`; first matching meal):

| Case | Input | Expected | Actual |
|---|---|---|---|
| Normal | test meal at 15:32 (glucose then fell 33 mg/dL) | low risk | **low risk** (11%) |
| Abstain | test meal at 19:03, last 40 min of CGM removed | can't judge | **can't judge**: "0 readings in the last 30 min" |
| Abstain | a plain white picture | can't judge | **can't judge**: "photo is almost blank" |
| Borderline | test meal at 12:51 (no spike) | can't judge | **can't judge**: "risk 17% is borderline" |
| Wrong | test meal at 12:47 (rose 35 mg/dL; no spike, so low risk was expected) | low risk | **caution** (48%) |

**The wrong case:** the photo shows a large carb portion ("60–89 g, 92% sure"), but this person responds mildly.

## 4. Feedback, risks and plan
**How the proposal feedback was addressed:**

- **Too dense:** shorter, with a bottom line first.
- **Weak framing:** a user, a buyer and a feedback loop.
- **Photo model:** ConvNeXt-Tiny + carb-range head.
- **Scratch CNN:** tested; it ties, so it is not deployed.
- **Fusion overkill:** a logistic regression.
- **Small data:** person-level CV with ranges, plus CV without the test people.

**Changes from the proposal:**

- A nearest-neighbour check + blank-photo rule instead of DINOv2. It flags 3.1% of real meal photos and all synthetic non-meal images.
- Conformal intervals, W&B, Nutrition5k and the user test move to the final.

**Risks:**

- The type 2 diabetes training mix.
- One photo seed.
- The two photo outputs disagree on 51 of the 90 meals with a photo, so users see only the carb range.

**Final submission:**

1. Recalibrate and test down-weighting the type 2 diabetes people. Choose the cut-offs on validation. Run the locked 8 people once.
2. Run 3 seeds. Give the CNN a fair try (a longer window). Make the two photo outputs agree. Add Nutrition5k.
3. Compare DINOv2 with the nearest-neighbour check; add tiered escalation.
4. Robustness checks, latency breakdown, explainability, tracker exports, and a user test with 3–5 people.
