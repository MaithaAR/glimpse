# GLIMPSE: intermediate report
Before eating: a meal photo + the last 2 h of CGM glucose and heart rate + a short profile → **low risk**, **caution** or **can't judge** (with a reason).
Maitha Alhosani · MAAI7103 · October 2026 · `just install` → `just demo`

**Bottom line.** All four inputs run end to end. In cross-validation, carbs read from the photo predict spikes as well as logged carbs, so a photo can replace food logging; but the gain over profile + glucose trend is small (+0.037 AUROC). Not useful yet: on the 5 test people the model does not beat profile-only, over-predicts risk (ECE 0.129) and clears only 3% of meals.

## 1. User and workflow
An adult with prediabetes in a 10–14 day CGM programme wants fewer spikes (a rise ≥ 50 mg/dL within 2 h) but stops logging food within days. Before eating they take one photo and get *caution* with a nudge (smaller carb portion, a walk), *low risk*, or *can't judge* with the reason. The programme pays, as it drops manual logging; a wellness nudge, not a medical device. **Feedback (mocked):** the CGM labels each meal 2 h later, the user can correct it in `just app`, and validated `FeedbackRecord`s are stored for retraining.

## 2. Work completed
**Tooling:** uv + Python 3.14 + lock file; `just` recipes; strict Pydantic for config, data and outputs; Loguru to stderr and a rotating file; ruff + 82 tests; Linux CI with an offline Docker self-test. No dependency blockers.

**Data:** CGMacros (PhysioNet, CC BY-NC-SA, 45 adults), downloaded by `just install`, each meal validated with Pydantic; 24 five-minute CGM and heart-rate slots before the meal; augmentation: sensor noise, photo flips and crops. **Person splits** (seed 20260927): train 28 people (14 healthy/prediabetes + 14 type 2 diabetes, 547 meals, 48% spikes), validation 4 (99 meals, 19%), test 5 (96 meals, 29%, scored once), 8 locked for the final. Scaling is fitted on train only; no photo crosses splits; quality checks and the fine-tuning ledger are in `outputs/data/`. **Main estimate:** 5-fold person cross-validation (CV) over the 23 non-locked healthy/prediabetes people (481 meals), since 5 test people cannot rank models.

**Models** (`just params` → `outputs/param_budget.csv`):

| Model | Role | Parameters / limit |
|---|---|---|
| SensorCNN: two 1-D convolutions over glucose, heart rate, missing flag + profile | from scratch | 5,217 / 50M |
| ConvNeXt-Tiny (ImageNet-22k) + carbs head + 4-bin carb-range head | fine-tuned | 27.8M / 200M |
| Fusion: logistic regression on 15 profile/trend + 4 photo inputs | classical | 20 |
| DINOv2-S photo check | *planned* | 22M / 200M |
| **Pretrained total · fine-tuning photos** | | 27.8M (+22M) / 1B · 1,221 (2,021 planned) / 10,000 |

The CNN tests whether the raw trace beats 7 hand-made trend numbers (small kernels suit 547 meals); ConvNeXt-Tiny + carb-range head follows the proposal feedback (865 photos need a pretrained backbone). Deterministic rules run first (CGM complete and in range; photo readable, not blank, familiar), then the models, then fixed cut-offs.

**Training** (seeded; settings validated at start-up; per-epoch logs to Loguru and CSV; checkpoints `sensor_cnn_e13_valauc0.662.pt`, `photo_convnext_e17_valmae26.2.pt`). CNN: BCE, AdamW, warm-up + cosine, 13 epochs chosen on inner folds; validation AUROC 0.662. Photo: Huber loss on log carbs + cross-entropy on carb ranges; 15 epochs train the heads on a frozen backbone (865 photos would overfit it), then the last stage trains 2 epochs because inner-fold error fell from 30.7 to 25.4 g; validation error 26.2 g.

<img src="../outputs/training/sensor_cnn_curves.png" width="49%"> <img src="../outputs/training/photo_convnext_curves.png" width="49%">

CNN validation loss is below training loss because validation people spike less. Photo validation error flattens near 30 g from epoch 5 while training loss falls (mild overfitting; the epoch-3 jump is in grams only, from back-converting a few large log predictions); unfreezing cuts it to 26 g. In CV the carb error is 22.5 g vs 27.7 g for the median guess; frozen ConvNeXt + ridge gets 20.4 g but far worse carb ranges (F1 0.38 vs 0.61), which carry the spike gain, so fine-tuning is kept.

**Pipeline:** `just demo`, `predict` and `app` share one path: `MealInput` → window code → rules → photo model → fusion → cut-offs → Pydantic `Answer` JSON, e.g. `{"decision": "low risk", "risk": 0.108, "reason": "risk 11% is below the low-risk cut-off 14%"}`. Bad input returns *can't judge* with its reason.

## 3. Results
AUROC is primary; 95% ranges come from resampling people; on the test every model is trained on the same 28 people. Full tables (AUPRC, Brier, all ranges): `outputs/evaluation/results.md`, `.json`, `.csv`.

| Model | CV AUROC (range) | CV without test people | Test AUROC |
|---|---|---|---|
| Profile only (logistic regression) | 0.672 (0.567–0.764) | 0.659 | 0.765 |
| Profile + trend (logistic regression) | 0.686 (0.604–0.772) | 0.686 | 0.695 |
| Profile + trend (gradient boosting) | 0.689 (0.610–0.778) | – | 0.632 |
| Profile + SensorCNN (scratch) | 0.689 (0.603–0.773) | 0.691 | 0.674 |
| **+ photo (main model)** | **0.724 (0.659–0.796)** | **0.719** | 0.742 |
| + logged carbs (reference) · + shuffled photo (control) | 0.717 · 0.679 | 0.715 · – | – |

The photo adds +0.037 (+0.003 to +0.073) and shuffling removes it; without the test people it is +0.033 (−0.007 to +0.077), so not yet certain. Photo vs logged carbs: +0.007 (−0.020 to +0.032); the carb-range head gives +0.020 of the gain. The scratch CNN ties the trend numbers, confirming the proposal concern, so it stays a baseline. On the test, main − profile-only = −0.023 (−0.114 to +0.159): indistinguishable with 5 people. Some checks were added after the results (labelled in `docs/evaluation_protocol.md`); nothing was tuned on the test.

**Calibration** (main model, test, 15 equal-width bins, meals per bin in the figure): ECE 0.129 (CV 0.052). It over-predicts in the 0.3–0.6 bins (45% predicted, 30% spiked, 46 meals) and under-predicts above 0.6 (10 meals). CV models share the training mix and are calibrated, so the likely (untested) cause is that these test people spike less. Over-prediction pushes meals to *caution*, so only 3% are cleared; recalibration comes first in the final.

<img src="../outputs/evaluation/fig3_reliability_test.png" width="30%">

| Target (*met* = whole range meets it) | CV / test | Status |
|---|---|---|
| Photo adds signal | +0.037 | met in CV; not without the test people |
| RQ1: photo not worse than logged carbs by > 0.03 | +0.007 (−0.020 to +0.032) | met |
| RQ2: scratch CNN beats trend numbers | +0.002 (−0.016 to +0.022) | at risk (tie) |
| ECE ≤ 0.05 | 0.052 / 0.129 | at risk |
| Cost saving ≥ 10%; cleared ≥ 20%; spikes among cleared ≤ 10% | 9%, 14%, 12% / 7%, 3%, 0% | at risk |
| *Can't judge* ≤ 20%; response time p95 ≤ 300 ms | 11% / 16%; 282 ms | met |

**Abstention cut-offs (deviation from 5b).** The cut-off chosen on validation (0.31) failed its pre-stated stability check (no cost saving in CV). Before the test was opened, the cut-offs were derived from the stated costs alone, using no data: *low risk* below 0.143, *caution* from 0.20, *can't judge* between. They live in the config and are checked by checksum. On the test, 0.31 would clear 42% of meals with 15% spikes among them. Details: `docs/evaluation_protocol.md`.

**Test cases** (`outputs/demo/cases.md`):

| Case | Input | Expected | Actual |
|---|---|---|---|
| Normal | meal at 15:32 (glucose then fell 33 mg/dL) | low risk | **low risk** (11%) |
| Warning | meal at 19:03 (then rose 58 mg/dL) | caution | **caution** (25%) |
| Abstain | same meal, last 40 min of CGM removed | can't judge | **can't judge**: "0 readings in the last 30 min" |
| Abstain | a plain white picture | can't judge | **can't judge**: "photo is almost blank" |
| Wrong | meal at 12:47 (rose 35 mg/dL, no spike) | low risk | **caution** (48%): large carb portion, but this person responds mildly |

## 4. Risks and plan
**Changes from the proposal:** nearest-neighbour + blank-photo checks replace DINOv2 (they flag all synthetic non-meal images and 3.1% of real meal photos); conformal intervals, W&B, Nutrition5k and the user test move to the final. **Risks:** type 2 diabetes training mix, one photo seed, the two photo outputs often disagree (users see only the carb range), 5 test people. **Final submission:** (1) recalibrate, choose cut-offs on validation, add tiered escalation, run the locked 8 people once; (2) 3 photo seeds, Nutrition5k, a longer CNN window, a meal-type/time-only arm, DINOv2 vs nearest-neighbour; (3) robustness, latency breakdown, explainability, tracker exports, a user test with 3–5 people.
