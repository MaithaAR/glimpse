# GLIMPSE: intermediate report
Before eating: a meal photo + the last 2 h of CGM glucose and heart rate + a short profile → **low risk**, **caution** or **can't judge** (with a reason). Offline, local weights, Docker self-test.
Maitha Alhosani · MAAI7103 · October 2026 · `just install` → `just demo`

**Bottom line.** The one claim, that the photo adds spike signal (+0.037 AUROC over profile + trend logistic regression), is backed by a shuffled-photo control, a logged-carbs reference it matches (+0.007), and a hold-out scored once under a protocol frozen before training. The product runs end to end on all four inputs (nothing is stubbed) but does not yet meet its clearance target: the model cannot tell rare from frequent spikers, so it over-predicts for the people who matter most (test ECE 0.129) and clears 3% of test meals. Section 4 states each limit with its fix and pass target.

## 1. User and workflow
An adult with prediabetes in a 10–14 day continuous glucose monitor (CGM) programme wants fewer spikes (a rise ≥ 50 mg/dL within 2 h) but stops logging food within days. Before eating they take one photo and get *caution* with a nudge (smaller carb portion, a walk), *low risk*, or *can't judge* with the reason. The programme pays, as it drops manual logging; a wellness nudge, not a medical device. **Feedback (mocked):** the CGM labels each meal 2 h later, the user can correct it in `just app`, and validated `FeedbackRecord`s are stored for retraining.

## 2. Work completed
**Tooling:** uv + Python 3.14 + lock; `just`; strict Pydantic for config, data and outputs; Loguru (stderr + rotating file); ruff + 84 tests; Linux CI with an offline Docker self-test. **Blockers:** none for dependencies. Photo cross-validation costs 90 min per seed on the Mac: one seed now, three in the final.

**Data:** CGMacros (PhysioNet, CC BY-NC-SA, 45 adults), downloaded by `just install`, each meal validated with Pydantic; 24 five-minute CGM and heart-rate slots before the meal; augmentation: sensor noise, photo flips and crops (the same window code serves training and the demo). **Person splits** (seed 20260927): train 28 people (14 healthy/prediabetes + 14 type 2 diabetes, 547 meals, 48% spikes), validation 4 (99 meals, 19%), test 5 (96 meals, 29%, scored once), 8 locked for the final. `tests/test_splits.py` asserts no shared people, scaling fitted on train only, and zero cross-split photo duplicates; quality counts and the fine-tuning ledger are in `outputs/data/`. **Main estimate:** 5-fold person cross-validation (CV) over the 23 non-locked healthy/prediabetes people (481 meals), since 5 test people cannot rank models.

**Models** (`just params` → `outputs/param_budget.csv`; forward-pass shape tests in `tests/test_models.py`):

| Model | Role | Parameters / limit |
|---|---|---|
| SensorCNN: two 1-D convolutions (25-min kernels) over glucose, heart rate, missing flag + profile | from scratch | 5,217 / 50M |
| ConvNeXt-Tiny (ImageNet-22k) + carbs head + 4-bin carb-range head | fine-tuned | 27.8M / 200M |
| Fusion: logistic regression on 15 profile/trend + 4 photo inputs | classical | 20 |
| DINOv2-S photo check (final; compared with the current nearest-neighbour check) | *planned* | 22M / 200M |
| **Pretrained total** | | 27.8M (+22M) / 1B |
| **Fine-tuning photos** (every photo any fit touched, so the cap is never under-counted; 865 unique in the final model) | | 1,221 (2,021 with Nutrition5k) / 10,000 |

The CNN tests whether the raw trace beats 7 hand-made trend numbers; two layers with 25-min kernels cover the 2 h window with few weights, suiting 547 meals. ConvNeXt-Tiny + carb-range head follows the proposal feedback; with 865 photos the backbone is kept frozen except its last stage. Learned parts (CNN, photo model, fusion) and deterministic parts (input rules, cut-offs) are separated in code: rules run first (CGM complete and in range; photo readable, not blank, familiar), then the models, then fixed cut-offs.

**Training** (seeded; settings validated at start-up; per-epoch logs to Loguru and CSV; checkpoints `sensor_cnn_e13_valauc0.662.pt`, `photo_convnext_e17_valmae26.2.pt`). CNN: BCE, AdamW, warm-up + cosine, 13 epochs chosen on inner folds; validation AUROC 0.662; CV AUROC over 3 seeds 0.688 ± 0.009. Photo: Huber loss on log carbs + cross-entropy on carb ranges; 15 epochs train the heads on a frozen backbone, then the last backbone stage is unfrozen for 2 epochs because inner-fold error fell from 30.7 to 25.4 g; validation error 26.2 g.

<img src="../outputs/training/sensor_cnn_curves.png" width="49%"> <img src="../outputs/training/photo_convnext_curves.png" width="49%">

CNN validation loss is below training loss because validation people spike less (19% vs 48%). Photo validation error flattens near 30 g from epoch 5 while training loss keeps falling (mild overfitting; the epoch-3 spike is in the gram error only, the loss is smooth); unfreezing cuts it to 26 g. Photo ablation in CV (`outputs/photo_cv/photo_cv_summary.json`):

| Carb estimate | Error (g) | Carb-range F1 | Spike AUROC when fused |
|---|---|---|---|
| Median guess | 27.7 | 0.08 | – |
| Frozen ConvNeXt + ridge | 20.4 | 0.38 | 0.703 |
| Fine-tuned, grams only | 22.5 | – | 0.704 |
| Fine-tuned, grams + carb ranges (used) | 22.5 | 0.61 | **0.724** |

Fine-tuning worsens the gram error (22.5 vs 20.4 g) but its carb-range head raises the fused AUROC by +0.020 (+0.005 to +0.035); it is kept for that. In the curves, the dashed line marks the last backbone stage being unfrozen (epochs 16–17).

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

<img src="../outputs/evaluation/fig1_auroc_ladder.png" width="46%">

CV AUROC is pooled across folds, so the constant baseline scores 0.40 rather than 0.50 (fold spike rates differ); within-fold means are in `results.md`.

The photo adds +0.037 (+0.003 to +0.073) and shuffling removes it; without the test people the lower bound is −0.007. Photo vs logged carbs: +0.007 (−0.020 to +0.032). The scratch CNN ties the trend numbers (+0.002, −0.016 to +0.022), confirming the proposal concern, so it stays a baseline. Test AUROCs are reported for completeness: with 5 people they do not separate models (main − profile-only = −0.023, −0.114 to +0.159). Post-hoc checks are labelled in `docs/evaluation_protocol.md`; nothing was tuned on the test.

**Calibration** (main model; 15 equal-width bins; meals per bin in the figures): CV ECE 0.052, test ECE 0.129. On the test it over-predicts in the 0.3–0.6 bins (45% predicted, 30% spiked, 46 meals) and under-predicts above 0.6 (10 meals). **Cause, checked without test labels:** in CV without the test people, rare spikers get 30% predicted vs 16% observed and frequent spikers 42% vs 54% (correlation −0.77); the errors cancel in the CV ECE. The validation people (31% vs 19%) and most test people are rare spikers.

<img src="../outputs/evaluation/fig2_reliability.png" width="30%"> <img src="../outputs/evaluation/fig3_reliability_test.png" width="30%">

| Target (*met* = whole range meets it) | CV / test | Status |
|---|---|---|
| Photo adds signal | +0.037 (+0.003 to +0.073); without test people −0.007 to +0.077 | met (CV rule); at risk once the test people are left out |
| RQ1: photo not worse than logged carbs by > 0.03 | +0.007 (−0.020 to +0.032) | met |
| RQ2: scratch CNN beats trend numbers | +0.002 (−0.016 to +0.022) | at risk (tie) |
| ECE ≤ 0.05 | 0.052 / 0.129 | at risk |
| Cost saving ≥ 10%; cleared ≥ 20%; spikes among cleared ≤ 10% (vs the best always-same answer) | 9%, 14%, 12% / 7%, 3%, 0% | at risk |
| *Can't judge* ≤ 20%; response time p95 ≤ 300 ms | 11% / 16% in the demo path (6 no photo, 7 borderline, 2 unfamiliar; `results.md` gives 9% because its evaluation path has no photo checks and fills missing photos with an average estimate); p95 282–299 ms across runs | met |

**Abstention cut-offs and rubric 5b.** A validation-chosen rule is implemented and frozen with a checksum (`just recalibrate`: Platt recalibration fitted on the 4 validation people, cut-offs chosen on that risk) but not deployed, because it failed its stability check on 14 CV people whose labels it never saw: it cleared 39% of meals with 22% spikes (cost −22%), since fitting to 4 rare spikers pushed frequent spikers to 36% predicted vs 59% observed. The pre-registered validation cut-off 0.31 had failed the same kind of check before the test was opened. Four validation people are too few and too alike to choose a rule, so the deployed cut-offs are the pre-registered cost-based ones, frozen before the test: *low risk* below 0.143, *caution* from 0.20, *can't judge (borderline)* between. This deviates from 5b and is the first fix in §4. Over-prediction pushes meals to *caution*, so 3% are cleared; 0.31 would clear 42% of test meals with 15% spikes (target ≤ 10%). Details: `outputs/evaluation/recalibration.md`.

**Test cases** (`outputs/demo/cases.md`):

| Case | Input | Expected | Actual |
|---|---|---|---|
| Normal | meal at 15:32 (glucose then fell 33 mg/dL) | low risk | **low risk** (11%) |
| Warning | meal at 19:03 (then rose 58 mg/dL) | caution | **caution** (25%) |
| Abstain | same meal, last 40 min of CGM removed | can't judge | **can't judge**: "0 readings in the last 30 min" |
| Abstain | a plain white picture | can't judge | **can't judge**: "photo is almost blank" |
| Wrong | meal at 12:47 (rose 35 mg/dL, no spike) | low risk | **caution** (48%): large carb portion, but this person responds mildly |

## 4. Limits, deviations and plan for the final
**Proposal feedback:** ConvNeXt-Tiny + carb-range head; fusion cut to a 20-parameter logistic regression; scratch CNN tested (it ties); person CV and a locked hold-out for the small data. **Changes from the proposal:** nearest-neighbour + blank-photo checks replace DINOv2 for now (together they flag all 6 synthetic non-meal images and 3.1% of real meal photos); conformal intervals, W&B, Nutrition5k (overhead plate photos with weighed carbs, joined as pre-training of the carb heads, 800 items counted in the ledger) and the user test move to the final.

| Limit | Evidence now | Fix in the final | Test (split · metric · pass) |
|---|---|---|---|
| 1. Model cannot tell rare from frequent spikers | CV ex-test: 30% vs 16%, 42% vs 54%; r = −0.77 | Per-person prior from the first 5 labelled meals (cold start: *caution* only); recalibration on CV folds, not 4 people | locked 8 · ECE overall and within each spike-rate half · ≤ 0.05 |
| 2. Cut-offs not validation-chosen (5b); 3% cleared | two validation rules failed stability | after fix 1: choose cut-offs on recalibrated CV-fold risk, committed with a hash before the locked set is scored; tiered escalation (borderline → nudge, unfamiliar photo → human) | locked 8 · cleared ≥ 20%, spikes among cleared ≤ 10%, cost saving ≥ 10% |
| 3. Photo gain uncertain; one photo seed | lower bound −0.007 ex-test | 3 photo seeds (4.5 h); Nutrition5k pre-training of the heads; a meal-type/time-only arm to show the gain is carbs | CV + locked 8 (31 people) · AUROC gain lower bound · > 0; seed sd of carb error < 2 g |
| 4. Backbone freezing untested | frozen vs last stage only | ablation frozen / last stage / full fine-tuning | CV · carb-range F1 · best kept |
| 5. Scratch CNN ties trend numbers | +0.002 (−0.016 to +0.022) | 4 h window, 3 seeds; else kept as baseline only | CV · RQ2 lower bound · > 0 |
| 6. Two photo outputs disagree | agreement not yet measured; users see the range only | single head with ordinal bins, or a consistency loss | CV · agreement rate · ≥ 90% |
| 7. Unfamiliar photos | nearest-neighbour check; 3.1% false flags | DINOv2-S vs nearest-neighbour | synthetic + held-out photos · recall 100%, false flags ≤ 2% |

Fixes run in this order (1 gates 2). Also in the final: robustness (blur, lighting, CGM gaps), latency breakdown, explainability of the fusion weights, experiment-tracker exports, and a user test with 3–5 people.
