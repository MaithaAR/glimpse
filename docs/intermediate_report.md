# GLIMPSE: intermediate report
**Will this meal spike my glucose?** Photo + last 2 h of CGM + profile → *low risk*, *caution* or *can't judge* (with a reason).
Maitha Alhosani · MAAI7103 · October 2026 · `just install` → `just demo`

*Terms used below:*

- **CGM:** glucose sensor.
- **Spike:** a rise of 50 mg/dL or more within 2 h.
- **AUROC:** ranking quality (0.5 = chance).
- **ECE:** the gap between predicted and observed spike rate.
- **CV:** 5-fold cross-validation by person.
- **Range:** 95% interval from resampling people.

## 1. User and workflow
The user is an adult with prediabetes in a CGM programme (for example, 10–14 days via a prevention clinic) who stops logging food after a few days.

Before eating, they take one photo in the CGM app. Within 0.3 s they get one of:

- *caution*, with a nudge (a smaller carb portion, or a walk afterwards);
- *low risk*;
- *can't judge*, with the reason.

**Who pays:** prevention programmes and CGM apps, buying a "no logging" add-on. It is a wellness nudge, not a medical device.

**Feedback (mocked):**

- 2 h later, the CGM labels the meal automatically.
- The user can correct it ("did not eat it", "ate part of it") in `just app`.
- Records are validated (`FeedbackRecord`) and saved to `outputs/feedback/feedback.jsonl`, for a dietitian's review and retraining.

## 2. Work completed
**Data.** All four inputs (photo, CGM, heart rate, profile) are in the pipeline, so no modality remains.

- **Source:** CGMacros (PhysioNet, CC BY-NC-SA), 45 adults. Downloaded by `just install`, never committed.
- **Validation:** Pydantic checks every meal at load time.
- **Augmentation:** sensor noise; photo flips, brightness and crops.
- **Leakage controls:** scaling statistics from training people only; no duplicate photos across splits.
- **Splits:** by person, seed 20260927.

| Group | People | Scored meals | Spike rate |
|---|---|---|---|
| Train | 28 (14 healthy/prediabetes + 14 type 2 diabetes) | 547 | 48% |
| Validation / Test (scored once) | 4 / 5 | 99 / 96 | 19% / 29% |
| CV, the main estimate (all 23 target people: includes validation and test) | 23 | 481 | 33% |
| Locked for the final | 8 | – | – |

- **Fine-tuning ledger:** 1,221 photos used so far; 2,021 including the planned Nutrition5k estimate (800), against the 10,000 cap.
- Details: `docs/data_card.md`.

**Models** (`just params`, counted in code):

| Model | Role | Parameters (limit) |
|---|---|---|
| SensorCNN: two 1-D convolutions (kernel = 25 min) over glucose, heart rate and a missing-data flag, + profile | from scratch | 5,217 (50M) |
| ConvNeXt-Tiny (ImageNet-22k) + carbs head + 4-bin carb-range head | fine-tuned | 27.8M (200M) |
| Fusion: logistic regression on profile + 7 trend numbers + photo outputs | classical | 20 |
| Unfamiliar-photo check (5 nearest training photos) | no weights | 0 |
| DINOv2-S photo check | *planned (estimate)* | 22M (200M) |

Pretrained and fine-tuned models: 27.8M built, 49.8M with the planned model, against a 1B limit.

**Why these models:**

- **A 1-D CNN** learns short rises and dips in a regular sensor trace with very few weights.
- **ConvNeXt-Tiny with a carb-bin head**, as suggested in the feedback. It is a strong, CPU-friendly pretrained backbone, which is needed to read food from about 1,000 photos.
- **Plain rules run before any model** (CGM complete, photo readable and familiar). Frozen cut-offs turn the risk into the answer.

**Training.** Seeds are fixed; every setting is in `configs/default.yaml` and validated at start-up; per-epoch logs go to Loguru and a CSV.

- **CNN:** BCE loss, AdamW, warm-up + cosine schedule. 13 epochs, the median of 4 inner folds' best epochs. Validation AUROC 0.662.
- **Photo model:** Huber loss on log(carbs) + cross-entropy on the 4 bins.
  - Frozen backbone first: 15 epochs for the heads.
  - Then the last stage was unfrozen for 2 epochs, because the inner folds showed a gain (30.7 → 25.4 g).

<img src="../outputs/training/sensor_cnn_curves.png" width="49%"> <img src="../outputs/training/photo_convnext_curves.png" width="49%">

**Honest notes:**

- The CNN's validation loss sits below its training loss because the validation people spike less.
- The photo model learns only modestly: 22.5 g carb error in CV, against 27.7 g for always guessing the median.
- Its last stage overfits from epoch 3, so 2 epochs are kept.
- **Plan:** more photos (Nutrition5k) and more seeds.

**Pipeline** (`just demo`, `just predict`, `just app`, one code path):

- **Input:** raw readings + photo + profile, validated as `MealInput`.
- **Steps:** the same window code as training → rules → photo model → fusion → cut-offs.
- **Output:** a validated `Answer` in JSON.
- **Bad input:** gives *can't judge* with the reason.
- **Packaging:** a Linux container runs it offline; CI builds and self-tests it.
- **SensorCNN** is not used in the app, because it ties the simpler trend numbers.

## 3. Results
**Baselines.** Primary metric: CV AUROC (23 people). The last column shows the fixed test (5 people), with every model trained on the same 28 people.

| Model | CV AUROC (range) | CV Brier | Test AUROC |
|---|---|---|---|
| Profile only | 0.672 (0.567–0.764) | 0.204 | 0.765 |
| Profile + CGM trend (logistic regression) | 0.686 (0.604–0.772) | 0.199 | 0.695 |
| Profile + CGM trend (gradient boosting) | 0.689 (0.610–0.778) | 0.204 | 0.632 |
| Profile + SensorCNN | 0.689 (0.603–0.773) | 0.201 | 0.674 |
| **+ photo (main model)** | **0.724 (0.659–0.796)** | **0.191** | 0.742 |
| + logged carbs (reference) | 0.717 (0.648–0.796) | 0.192 | – |
| + shuffled photo (control) | 0.679 (0.590–0.766) | 0.204 | – |

**What this shows:**

- **The photo adds a small gain:** +0.037 (+0.003 to +0.073). Shuffling the photos removes it.
- **The photo matched logged carbs on this data:** +0.007 (−0.020 to +0.032).
- **The carb-range head carries most of it:** +0.020 (+0.005 to +0.035). This check was added after the results.
- **The CNN ties the trend numbers,** as the feedback expected.
- **Fine-tuning lost to frozen ConvNeXt + ridge on grams** (22.5 vs 20.4 g) but gave better carb ranges.
- **On the 5 test people, profile-only scores highest.** Five people cannot separate these models, so the 23-person CV is the main estimate.

**Calibration** (main model, test split, 15 equal-width bins, meals per bin on the figure):

- **ECE:** 0.129 on the test (0.052 in CV).
- **The model is over-confident about spikes:** in the 0.3–0.6 bins it predicts 45% where 30% spiked (46 meals).
- **Likely cause:** half the training people have type 2 diabetes.
- **Effect:** the cut-offs assume honest risks, so meals shift from *low risk* to *caution*. Recalibration is planned.

<img src="../outputs/evaluation/fig3_reliability_test.png" width="40%">

**Targets** (*met* = the whole range meets the target):

| Target (proposal) | Result | Status |
|---|---|---|
| Photo adds signal | +0.037 (+0.003 to +0.073) | met |
| RQ1: photo within 0.03 of logged carbs | +0.007 (−0.020 to +0.032) | met |
| RQ2: CNN beats trend features | +0.002 (−0.016 to +0.022) | on track by the rule, really a tie |
| ECE ≤ 0.05 | 0.052 CV / 0.129 test | at risk |
| Cost saving ≥ 10% | 9% CV / 7% test | at risk |
| Meals cleared ≥ 20%, spikes among them ≤ 10% | 14%, 12% CV / **3%, 0% test** | at risk |
| *Can't judge* ≤ 20% | 11% CV / 16% test | met |
| Response time p95 ≤ 300 ms (Mac CPU) | 285–303 ms across runs | borderline (met in the committed run) |

**In practice:** on the test, the app said *caution* to 81% of meals. That is safe, but not yet useful.

**Cut-offs:**

- The planned rule chose 0.31 on the 4 validation people. It failed its stability check, because those people spike rarely (19%).
- **Before the test was opened**, the cut-offs were replaced by values derived from the costs alone (low risk < 0.143, caution ≥ 0.20). They use no data, were committed with a hash, and the app checks its config against that file.

**Test cases** (`outputs/demo/cases.md`, first matching meal):

| Case | Input | Expected | Actual |
|---|---|---|---|
| Normal | test meal at 15:32 (real rise −33 mg/dL) | low risk | **low risk** (11%), "under 30 g", 282 ms |
| Abstain | test meal at 19:03 with its last 40 min of CGM removed | can't judge | **can't judge**: "0 readings in the last 30 min … gap 40 min" |
| Abstain | a grey picture as the photo | can't judge | **can't judge**: "photo looks unlike the meal photos" |

**Discrepancies:**

- With full data, the 19:03 meal gets *caution* at only 25% risk. That is by design: a missed spike costs 5× a needless warning. It did spike.
- A 35 mg/dL riser got *caution* (48%): a large plate, but a mild responder.

## 4. Response to the proposal feedback
| Feedback | Response |
|---|---|
| Too dense / weak startup framing | Terms up front, tables, about 3 pages; user, payer and feedback loop in Section 1 |
| Photo model: ConvNeXt-Tiny/EfficientNet-B3 + carb-bin head | ConvNeXt-Tiny + bin head; the bins add +0.020 AUROC |
| Scratch CNN not justified | Kept tiny as the required scratch model; it ties trend features |
| Fusion overkill | Logistic regression |
| Small dataset | Person-level CV and ranges; the test as a sanity check |

## 5. Blockers, risks and plan
**Changes from the proposal:**

- ConvNeXt-Tiny instead of EfficientNet-B0.
- Logistic-regression fusion.
- A nearest-neighbour photo check instead of DINOv2. It flags 3.1% of real meal photos and 5 of 6 non-meal images; plain white passes.
- Nutrition5k, conformal intervals, W&B, Emirati photos and the user test move to the final.

**Added after seeing results** (labelled in `docs/evaluation_protocol.md`): the regularisation grid, the ridge arm, the carb-range check, and the cut-offs.

**Risks:**

- One photo seed.
- The two photo outputs disagree on 51 of 90 meals, so users see only the range.
- The test people are inside the CV.
- One PhysioNet download dropped during a fresh install; re-running `just install` resumed it.

**Final submission:**

1. Recalibrate on CV predictions and down-weight the type 2 diabetes people; re-check the cut-offs; run the locked 8 people once.
2. Ablations with 3 seeds (carb range vs grams, frozen vs fine-tuned, Nutrition5k).
3. A stronger abstention path: DINOv2 vs the nearest-neighbour check on real non-meal photos, plus tiered escalation.
4. Robustness, latency breakdown, explainability, tracker exports, and a 3–5 person user test.
