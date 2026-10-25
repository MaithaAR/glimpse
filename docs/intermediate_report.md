# GLIMPSE: intermediate report
Before eating: a meal photo + the last 2 h of CGM glucose and heart rate + a short profile → **low risk**, **caution** or **can't judge** (with a reason).
Maitha Alhosani · MAAI7103 · October 2026 · `just install` → `just demo`

**Bottom line:**

- **End to end:** all four inputs, one validated pipeline (CLI, app, offline Docker).
- **A photo can replace food logging:** in cross-validation, carbs read from the photo predict spikes as well as logged carbs (difference +0.007 AUROC), and shuffled photos lose the gain. The gain itself is small: +0.037 over profile + glucose trend.
- **Not useful yet:** on the 5 test people the full model does not beat profile-only (0.742 vs 0.765), risk is over-predicted (ECE 0.129), and only 3% of test meals are *cleared* (answered *low risk*).

## 1. User and workflow
**User:** an adult with prediabetes in a 10–14 day CGM programme (clinic or app). They want fewer spikes (a rise of 50 mg/dL or more within 2 h) but stop logging food within days, so the CGM shows a spike only when it is too late.

**Workflow:** one photo before eating → in about 0.3 s, *caution* with a nudge (smaller carb portion, a walk), *low risk*, or *can't judge* with the reason. **Buyer:** the programme, which drops manual logging. A wellness nudge, not a medical device.

**Feedback (mocked):** the CGM labels each meal 2 h later; the user corrects it in `just app` ("did not eat it"); validated `FeedbackRecord`s are saved for retraining.

## 2. Work completed
**Tooling:** uv + Python 3.14 + lock file; `just`; strict Pydantic for config, data and outputs; Loguru (stderr + rotating file); ruff + 82 tests; Linux CI with an offline Docker self-test. No dependency blockers.

**Data:** CGMacros (PhysioNet, CC BY-NC-SA), 45 adults, downloaded by `just install`, each meal checked with Pydantic. Window: 24 five-minute CGM slots before the meal. Augmentation: sensor noise; photo flips and crops. Scaling fitted on training people only; no photo duplicated across groups (`outputs/data/`). Splits by person (seed 20260927; *scored meals* have usable CGM before and after):

| Group | People | Scored meals | Spike rate |
|---|---|---|---|
| Train: 14 healthy/prediabetes + 14 type 2 diabetes (T2D only here, which raises the rate) | 28 | 547 | 48% |
| Validation / Test (test scored once) | 4 / 5 | 99 / 96 | 19% / 29% |
| Locked for the final (checksum sealed in the proposal) | 8 | – | – |

**Main estimate:** 5-fold person cross-validation (**CV**) over the 23 non-locked healthy/prediabetes people (481 meals), since 5 test people are too few to rank models. It includes the test people, so **CV without them** is also reported.

**Models and budget** (`just params` → `outputs/param_budget.csv`):

| Model | Role | Used / limit |
|---|---|---|
| SensorCNN: two 1-D convolutions (25-min kernel) over glucose, heart rate and a missing-data flag, + profile | from scratch | 5,217 / 50M |
| ConvNeXt-Tiny (ImageNet-22k) + carbs head + 4-bin carb-range head | fine-tuned | 27.8M / 200M |
| Fusion: logistic regression on 15 profile/trend + 4 photo inputs (log grams + 3 carb-range probabilities) | classical | 20 parameters |
| DINOv2-S photo check | *planned (estimate)* | 22M / 200M |
| **Pretrained / fine-tuned total** | | 27.8M (+22M planned) / 1B |
| **Fine-tuning photos** (augmented copies not counted) | | 865 final model; 1,221 worst case over CV folds; 2,021 with planned Nutrition5k / 10,000 |

**Why these models:** the scratch CNN tests whether the raw trace holds more than 7 hand-made trend numbers; small kernels learn rises and dips with few weights, suiting 547 meals. ConvNeXt-Tiny + carb-range head follows the proposal feedback; 865 photos need a pretrained backbone. **Plain rules run first** (CGM complete and in range; photo readable, not blank, similar to training photos), then the models, then frozen cut-offs.

**Training** (seeds fixed; settings validated at start-up; per-epoch logs to Loguru and CSV; checkpoints `sensor_cnn_e13_valauc0.662.pt`, `photo_convnext_e17_valmae26.2.pt`):

- **CNN:** BCE loss, AdamW, warm-up + cosine; 13 epochs (median of the 4 inner folds' lowest-validation-loss epochs); 3 seeds in CV. AUROC 0.662 on the 4 validation people (0.689 in CV).
- **Photo model:** Huber loss on log(carbs) + cross-entropy on 4 carb ranges; one seed. The backbone is frozen first because 865 photos would overfit it: 15 epochs train the heads. The last stage then trains for 2 epochs, kept because the inner-fold error fell from 30.7 to 25.4 g (rule: keep if it drops ≥ 0.5 g). Validation error 26.2 g.

<img src="../outputs/training/sensor_cnn_curves.png" width="49%"> <img src="../outputs/training/photo_convnext_curves.png" width="49%">

- **CNN curve:** in the final run validation loss is below training loss: the validation people spike less.
- **Photo curve:** validation error levels off near 30 g from epoch 5 while training loss keeps falling (mild overfitting); the epoch-3 jump appears in grams only (the loss is smooth): a few large log-carb predictions blow up when converted back to grams. Unfreezing the last stage cuts it to 26 g.
- **Carbs in CV** (`outputs/photo_cv/photo_cv_summary.json`): 22.5 g vs 27.7 g for the median guess. Frozen ConvNeXt + ridge is better on grams (20.4 g) but worse on carb ranges (F1 0.38 vs 0.61), which carry the spike gain, so fine-tuning is kept.

**Pipeline.** `just demo`, `just predict` and `just app` share one path: `MealInput` → training window code → rules → photo model → fusion → cut-offs → a Pydantic `Answer` in JSON, e.g.
`{"decision": "low risk", "risk": 0.108, "reason": "risk 11% is below the low-risk cut-off 14%", "carb_bin": "under 30 g", "carb_bin_prob": 0.58}`. Bad input gives *can't judge* with the reason, never a crash.

## 3. Results
**Baselines** (AUROC primary, 0.5 = chance; 95% ranges from resampling people; on the test every model is trained on the same 28 people; AUPRC, Brier and all ranges in `outputs/evaluation/results.md`, `results.json` and `results_table.csv`):

| Model | CV AUROC (range) | CV without test people | Test AUROC |
|---|---|---|---|
| Profile only | 0.672 (0.567–0.764) | 0.659 | 0.765 |
| Profile + trend (logistic regression) | 0.686 (0.604–0.772) | 0.686 | 0.695 |
| Profile + trend (gradient boosting) | 0.689 (0.610–0.778) | – | 0.632 |
| Profile + SensorCNN | 0.689 (0.603–0.773) | 0.691 | 0.674 |
| **+ photo (main model)** | **0.724 (0.659–0.796)** | **0.719** | 0.742 |
| + logged carbs (reference) | 0.717 (0.648–0.796) | 0.715 | – |
| + shuffled photo (control) | 0.679 (0.590–0.766) | – | – |

Averaged within folds instead of pooled, the order is the same (main 0.734 vs profile + trend 0.699).

**What this shows:**

- **Photo gain** (vs profile + trend): +0.037 (+0.003 to +0.073), and shuffling the photos removes it. Without the test people: +0.033 (−0.007 to +0.077), so small and not yet certain.
- **Photo vs logged carbs:** +0.007 (−0.020 to +0.032). Being slightly ahead may mean the photo also picks up meal type or time (tested in the final).
- **The carb-range head** gives +0.020 (+0.005 to +0.035) of that gain.
- **The scratch CNN ties the trend numbers** (+0.002), confirming the proposal concern; it stays as a baseline, not deployed.
- **Test:** main − profile-only = −0.023 (−0.114 to +0.159); with 5 people they cannot be told apart.
- **Caveat:** some checks were added after the results (labelled in `docs/evaluation_protocol.md`); nothing was tuned on the test.

**Calibration** (main model, test; 15 equal-width bins; meals per bin in the figure and `results.md`):

- **ECE** 0.129 on the test (CV 0.052). It over-predicts in the middle: in the 0.3–0.6 bins it says 45% where 30% spiked (46 meals). Above 0.6 it under-predicts, on only 10 meals.
- **Cause (untested):** not the training mix alone, since the CV models use it and are calibrated. The test people spike less than the CV average (29% vs 33%); with 5 people this may be chance.
- **Effect on abstention:** the cost-based cut-offs assume honest risks, so over-prediction pushes meals from *low risk* to *caution*: only 3% are cleared. Recalibration comes first in the final.

<img src="../outputs/evaluation/fig3_reliability_test.png" width="38%">

**Targets** (*met* = the whole range meets the target; RQ1 and RQ2 are judged on the lower end):

| Target | CV / test | Status |
|---|---|---|
| Photo adds signal | +0.037 (+0.003 to +0.073) | met in CV; not without the test people (−0.007 to +0.077) |
| RQ1: photo not worse than logged carbs by more than 0.03 | +0.007 (−0.020 to +0.032) | met (lower end above −0.03) |
| RQ2: scratch CNN beats trend numbers | +0.002 (−0.016 to +0.022) | at risk (a tie: lower end below 0) |
| ECE ≤ 0.05 | 0.052 / 0.129 | at risk |
| Cost saving ≥ 10%; cleared ≥ 20%; spikes among cleared ≤ 10% | 9%, 14%, 12% / 7%, 3%, 0% | at risk |
| *Can't judge* ≤ 20% | 11% / 16% | met |
| Response time p95 ≤ 300 ms (Mac CPU) | 282 ms (`outputs/demo/demo_summary.json`) | met, close to the limit |

Test *can't judge* counts meals without a photo, as the app does (6 no photo, 7 borderline, 2 unfamiliar of 96); `results.md` (9%) gives them an average photo estimate instead.

**Cut-offs (a deviation from rubric 5b):** the cut-off chosen on validation (0.31) failed its pre-stated stability check: on the CV it saved no cost, because the 4 validation people rarely spike. Before the test was opened, the cut-offs were therefore derived from the stated costs alone, using no data. Costs (if spike / if not): *low risk* 5/0, *caution* 0/1, *can't judge* 2/0.5. This gives *low risk* below 0.143, *caution* from 0.20, *can't judge* between (`docs/evaluation_protocol.md`). They are sealed by checksum and checked by the app.

| Cut-offs on the test | Cleared | Spikes among cleared | Cost saving |
|---|---|---|---|
| Cost-only (used) | 3% | 0% | 7% |
| Validation-chosen 0.31 (not used) | 42% | 15% | 6% |

The final chooses them on validation after recalibration.

**Test cases** (`outputs/demo/cases.md`; test people, frozen pipeline):

| Case | Input | Expected | Actual |
|---|---|---|---|
| Normal | meal at 15:32 (glucose then fell 33 mg/dL) | low risk | **low risk** (11%) |
| Warning | meal at 19:03 (then rose 58 mg/dL) | caution | **caution** (25%) |
| Abstain | same 19:03 meal, last 40 min of CGM removed | can't judge | **can't judge**: "0 readings in the last 30 min" |
| Abstain | a plain white picture | can't judge | **can't judge**: "photo is almost blank" |
| Wrong | meal at 12:47 (rose 35 mg/dL, no spike) | low risk | **caution** (48%) |

**The wrong case:** the photo shows a large carb portion ("60–89 g, 92% sure"), but this person responds mildly.

## 4. Deviations, risks and plan
**Changes from the proposal:** a nearest-neighbour check + blank-photo rule replace DINOv2; together they flag all synthetic non-meal images and 3.1% of real meal photos (the first alone misses a plain white picture). Conformal intervals, W&B, Nutrition5k and the user test move to the final.

**Risks:** T2D training mix and spike-rate shift; one photo seed; the two photo outputs often disagree (users see only the carb range); 5 test people.

**Proposal feedback acted on:** ConvNeXt-Tiny + carb-range head; logistic-regression fusion; scratch CNN tested as a baseline; person-level CV for the small data.

**Final submission:**

1. **Abstention:** recalibrate, test down-weighting T2D, choose cut-offs on validation, add tiered escalation; run the locked 8 people once.
2. **Models:** 3 photo seeds, Nutrition5k, agreeing photo outputs, a longer CNN window, a meal-type/time-only arm; DINOv2 vs nearest-neighbour check.
3. **Studies:** robustness, latency breakdown, explainability, tracker exports, a user test with 3–5 people.
