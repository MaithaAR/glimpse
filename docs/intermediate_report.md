# GLIMPSE: intermediate report
Before eating, a photo + the last 2 h of CGM (glucose sensor) + a short profile → **low risk**, **caution** or **can't judge** (with a reason).
Maitha Alhosani · MAAI7103 · October 2026 · `just install` → `just demo`

## 1. User and workflow
The user is an adult with prediabetes in a 10–14 day CGM programme, run by a prevention clinic or a CGM app. They want fewer spikes (a rise of 50 mg/dL or more within 2 h) but stop logging food after a few days.

- **Before eating:** they take one photo. About 0.3 s later they get *caution* with a nudge (a smaller carb portion, or a walk), *low risk*, or *can't judge* with the reason. There is no logging.
- **Who pays:** the programme or app, as an add-on. It is a wellness nudge, not a medical device.
- **Feedback (mocked):** 2 h later the CGM labels the meal; the user can correct it in `just app` ("did not eat it"). Validated `FeedbackRecord`s are saved for review and retraining.

## 2. Work completed
**Data:**

- **Source:** CGMacros (PhysioNet, CC BY-NC-SA), 45 adults, downloaded by `just install`.
- **All four inputs reach the model:** photo, CGM, heart rate and profile.
- **Validation:** Pydantic checks every meal at load time.
- **Preprocessing:** 24 five-minute CGM slots ending at the meal; sensor noise and photo flips/crops as augmentation; scaling from training people only; no photo duplicated across groups.
- **Fine-tuning ledger:** 1,221 photos (2,021 with the planned Nutrition5k) out of 10,000.
- **Splits:** by person (seed 20260927). Each person is in exactly one group:

| Group | People | Scored meals | Spike rate |
|---|---|---|---|
| Train (14 healthy/prediabetes + 14 type 2 diabetes, never scored) | 28 | 547 | 48% |
| Validation / Test (scored once) | 4 / 5 | 99 / 96 | 19% / 29% |
| Locked for the final (hash-sealed) | 8 | – | – |

- **Main estimate:** 5-fold person cross-validation (**CV**) over all 23 non-locked healthy/prediabetes people (481 meals, 33% spikes). It therefore includes the validation and test people. No setting was chosen on the test.

**Models** (`just params`). Pretrained + fine-tuned: 27.8M built (49.8M with the planned model) of 1B.

| Model | Role | Parameters (limit) |
|---|---|---|
| SensorCNN: two 1-D convolutions (kernel = 25 min) over glucose, heart rate and a missing-data flag, + profile | from scratch | 5,217 (50M) |
| ConvNeXt-Tiny (ImageNet-22k) + carbs head + 4-bin carb-range head | fine-tuned | 27.8M (200M) |
| Fusion: logistic regression on profile + 7 glucose/heart-rate trend numbers + photo outputs | classical | 20 |
| DINOv2-S photo check | *planned (estimate)* | 22M (200M) |

**Why these models:**

- **A 1-D CNN** learns short rises and dips with few weights.
- **ConvNeXt-Tiny with a carb-range head** (as suggested in the feedback) is a strong, CPU-friendly pretrained backbone, needed to read food from about 1,000 photos.
- **Plain rules run first:** CGM complete and in range; photo readable, not blank, and similar to training photos (5 nearest neighbours). Frozen cut-offs then turn the risk into the answer.

**Training.** Seeds are fixed; all settings are in `configs/default.yaml`, validated at start-up; per-epoch logs go to Loguru and a CSV.

- **CNN:** BCE loss, AdamW, warm-up + cosine schedule; 13 epochs (median of 4 inner folds); validation AUROC 0.662.
- **Photo model:** Huber loss on log(carbs) + cross-entropy on 4 carb ranges (<30, 30–59, 60–89, ≥90 g).
  - Heads trained on the frozen backbone (15 epochs).
  - Then the last stage for 2 epochs, because it cut the inner-fold error from 30.7 to 25.4 g.

<img src="../outputs/training/sensor_cnn_curves.png" width="49%"> <img src="../outputs/training/photo_convnext_curves.png" width="49%">

**Notes:**

- The CNN's validation loss is below its training loss because the validation people spike less.
- The photo model overfits from epoch 3, so 2 epochs are kept.
- Carb error in CV: 22.5 g, against 27.7 g for guessing the median. A frozen ConvNeXt + ridge scored 20.4 g, but its carb ranges were worse (F1 0.38 vs 0.61), which is why fine-tuning is kept.
- **Plan:** 3 seeds and more photos.

**Pipeline.** `just demo`, `just predict` and `just app` share one path:

- **Input:** raw readings + photo + profile, validated as `MealInput`.
- **Steps:** the training window code → rules → photo model → fusion → cut-offs.
- **Output:** a Pydantic `Answer` in JSON, e.g. `{"decision": "low risk", "risk": 0.108, "reason": "risk 11% is below the low-risk cut-off 14%", "carb_bin": "under 30 g", "carb_bin_prob": 0.58}`.
- **Bad input:** malformed or empty input gives *can't judge* with the reason.
- **Packaging:** a Linux container runs offline; CI tests it.
- **SensorCNN** is not used in the app, because it ties the simpler trend numbers.

## 3. Results
**Baselines.** AUROC ranks spiking above non-spiking meals (0.5 = chance) and is the primary metric. CV results carry 95% ranges from resampling people. The test column uses the same models, trained on the same 28 people.

| Model | CV AUROC (range) | CV Brier | Test AUROC |
|---|---|---|---|
| Profile only | 0.672 (0.567–0.764) | 0.204 | 0.765 |
| Profile + trend (logistic regression) | 0.686 (0.604–0.772) | 0.199 | 0.695 |
| Profile + trend (gradient boosting) | 0.689 (0.610–0.778) | 0.204 | 0.632 |
| Profile + SensorCNN | 0.689 (0.603–0.773) | 0.201 | 0.674 |
| **+ photo (main model)** | **0.724 (0.659–0.796)** | **0.191** | 0.742 |
| + logged carbs (reference) | 0.717 (0.648–0.796) | 0.192 | – |
| + shuffled photo (control) | 0.679 (0.590–0.766) | 0.204 | – |

**What this shows:**

- **The photo adds a small gain:** +0.037 (+0.003 to +0.073), and shuffling removes it. It matches logged carbs: +0.007 (−0.020 to +0.032).
- **The carb-range head carries most of the gain:** +0.020 (+0.005 to +0.035), a check added after the results. The CNN ties the trend numbers.
- **On the test, profile-only is higher by 0.023,** but the paired range (−0.159 to +0.114) cannot separate the models with 5 people. CV is the main estimate.

**Calibration** (main model on the test, 15 equal-width bins; meals per bin on the figure and in `results.md`):

- **ECE** (the gap between predicted and observed spike rate) is 0.129 (CV: 0.052).
- **The model over-predicts:** in the 0.3–0.6 bins it says 45% where 30% spiked (46 meals). The likely cause is the type 2 diabetes people in training.
- **Effect:** meals shift from *low risk* to *caution*.

<img src="../outputs/evaluation/fig3_reliability_test.png" width="38%">

**Targets** (*met* = the whole range meets the target):

| Target | CV / test | Status |
|---|---|---|
| Photo adds signal | +0.037 (+0.003 to +0.073) | met |
| RQ1: photo within 0.03 of logged carbs | +0.007 (−0.020 to +0.032) | met |
| RQ2: CNN beats trend numbers | about 0 | not met (tie) |
| ECE ≤ 0.05 | 0.052 / 0.129 | at risk |
| Cost saving ≥ 10%; meals cleared ≥ 20% | 9%, 14% / 7%, 3% | at risk |
| *Can't judge* ≤ 20% | 11% / 16% | met |
| Response time p95 ≤ 300 ms (Mac CPU) | 283–303 ms across runs | borderline |

On the test, the app said *caution* to 78 meals, *low risk* to 3, and *can't judge* to 15 (6 had no photo, 7 were borderline, 2 photos were unfamiliar). Safe, but not yet useful.

**Cut-offs:**

- The planned rule chose 0.31 on the 4 validation people. It failed its stability check (they spike rarely).
- **Before the test was opened**, it was replaced by cut-offs that follow from the costs alone (no data). The costs, if it spikes / if not, are: *low risk* 5 / 0, *caution* 0 / 1, *can't judge* 2 / 0.5.
- So *low risk* when p < 0.143, *caution* when p ≥ 0.20, and *can't judge* in between. They were committed with a hash, which the app checks.

**Test cases** (`outputs/demo/cases.md`; first matching meal):

| Case | Input | Expected | Actual |
|---|---|---|---|
| Normal | test meal at 15:32 (glucose then fell) | low risk | **low risk** (11%) |
| Abstain | test meal at 19:03, last 40 min of CGM removed | can't judge | **can't judge**: "0 readings in the last 30 min" |
| Abstain | a plain white picture | can't judge | **can't judge**: "photo is almost blank" |
| Borderline | test meal at 12:51 (no spike) | can't judge | **can't judge**: "risk 17% is borderline" |

**Discrepancy:** the 19:03 meal with full data got *caution* at 25% risk. That is by design, because a missed spike costs 5× a needless warning, and it did spike.

## 4. Response to the proposal feedback
| Feedback | Response |
|---|---|
| Too dense; weak startup framing | Shorter report, terms explained, user moment, payer, feedback loop |
| Photo model weak | ConvNeXt-Tiny + carb-range head (+0.020 AUROC from the ranges) |
| Scratch CNN may not be justified | Kept tiny as required; it ties the trend numbers and is not deployed |
| Fusion overkill; small dataset | Logistic regression; person-level CV with ranges |

## 5. Risks and plan
**Changes from the proposal:**

- ConvNeXt-Tiny instead of EfficientNet-B0, and a logistic-regression fusion.
- A nearest-neighbour photo check plus a blank-photo rule instead of DINOv2. It flags 3.1% of real meal photos and every synthetic non-meal image.
- Conformal intervals, W&B, Nutrition5k and the user test move to the final.

**Added after seeing results** (labelled in `docs/evaluation_protocol.md`): the regularisation grid, the ridge baseline, the carb-range check, and the cut-offs.

**Risks:**

- The type 2 diabetes training mix causes over-prediction.
- One photo seed.
- The two photo outputs disagree on 51 of the 90 meals with a photo, so users see only the carb range.

**Final submission:**

1. Recalibrate and down-weight the type 2 diabetes people; re-check the cut-offs; run the locked 8 people once.
2. Ablations with 3 seeds; make the two photo outputs agree.
3. DINOv2 vs the nearest-neighbour check, and tiered escalation.
4. Robustness, latency breakdown, explainability, tracker exports, and a user test with 3–5 people.
