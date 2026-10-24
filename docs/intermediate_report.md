# GLIMPSE: intermediate report

**Will this meal spike my glucose?** A before-meal photo, the last 2 h of CGM and a short profile give *low risk*, *caution* or *can't judge*.
Maitha Alhosani · MAAI7103 · October 2026 · `just install` → `just demo` (see README)

## 1. User and workflow
The user is an adult with prediabetes who already wears a CGM (for example, a 10–14 day sensor programme from a diabetes-prevention clinic). They want fewer spikes, but they stop logging food after a few days.

Before eating, they take one photo in the CGM app. In about 0.3 s GLIMPSE answers with one of:

- *caution*, with a nudge (smaller carb portion, or a short walk afterwards);
- *low risk*;
- *can't judge*, with the reason (sensor gap, no photo, not a meal, borderline risk).

There is no logging. It is a wellness nudge, not a medical device.

**Feedback (mocked):**

- 2 h later, the CGM labels the meal automatically using the training rule (`feedback.auto_rise`).
- The user can correct it: "did not eat it" or "ate part of it" (`just app`).
- Records are validated (`FeedbackRecord`) and appended to `outputs/feedback/feedback.jsonl`, for a weekly dietitian summary and later retraining.

## 2. Work completed
**Data.** All four inputs are in the pipeline: photo, CGM, heart rate and profile.

- **Source:** CGMacros (PhysioNet, CC BY-NC-SA 4.0), 45 adults, downloaded by `just install` and never committed.
- **Validation:** Pydantic checks every meal at load time.
- **Windows:** 24 five-minute slots ending at the meal.
- **Splits:** by person, seed 20260927.
  - Main estimate: 5-fold cross-validation (CV) over 23 people (481 scored meals, 33% spikes).
  - Fixed test: 5 people (96 meals). Validation: 4 people. Training: 28 people.
  - 8 locked people, sealed with a hash.
- **Leakage controls:** scaling statistics come from training people only, and no photo is duplicated across splits.
- **Fine-tuning ledger:** at most 1,221 photos, against the 10,000 cap.
- Details: `docs/data_card.md`.

**Models** (`just params`, counted in code):

| Model | Role | Parameters (limit) |
|---|---|---|
| SensorCNN: 1-D CNN over the 2 h CGM/heart-rate window + profile | from scratch | 5,217 (50M) |
| PhotoCarbModel: ConvNeXt-Tiny (ImageNet-22k, pinned) + carbs head + 4-bin head | pretrained, fine-tuned | 27,823,973 (200M) |
| Fusion: logistic regression on profile + trend + photo inputs | classical | 20 |
| Unfamiliar-photo check: 5 nearest training photos in the photo features | no weights | 0 |
| DINOv2-S photo check | planned, final stage (estimate) | 22,000,000 (200M) |

Pretrained plus fine-tuned total: 49.8M of 1B.

**Why these models:**

- **ConvNeXt-Tiny with a carb-bin head:** a pretrained network is needed to read food and portions from about 1,000 photos.
- **The CNN is the required scratch model,** kept tiny for a small dataset.
- **The fusion stays a transparent logistic regression.**
- **Rules run before any model:** CGM completeness and range, photo readable and familiar, frozen cut-offs.

**Training.** Seeds are fixed; settings live in `configs/default.yaml` and are validated at start-up; every epoch is logged to Loguru and a CSV.

- **CNN:**
  - BCE loss, AdamW, warm-up + cosine schedule, noise augmentation;
  - 13 epochs, the median of the 4 inner folds' best epochs;
  - validation AUROC 0.662.
- **Photo model:**
  - Huber loss on log(carbs) + 0.5 × cross-entropy on 4 bins (<30, 30–59, 60–89, ≥90 g);
  - frozen backbone with cached features for the heads (15 epochs);
  - last stage unfrozen only if it cut the inner-fold error by ≥ 0.5 g. It did (30.7 → 25.4 g), so it was kept for 2 epochs;
  - validation error 26.2 g.
- **Checkpoints:** `sensor_cnn_e13_valauc0.662.pt`, `photo_convnext_e17_valmae26.2.pt`.

<img src="../outputs/training/sensor_cnn_curves.png" width="49%"> <img src="../outputs/training/photo_convnext_curves.png" width="49%">

**Training behaviour:**

- The CNN's validation loss is *below* its training loss because the validation people spike less (19% vs 48%), not because of a bug. A per-group loss check confirmed this.
- The photo model's last stage overfits from its 3rd epoch, so only 2 epochs are kept.

**Pipeline** (`just demo`, `just predict`, `just app`, with one code path for all three):

- **Input:** raw readings + photo + profile, validated as `MealInput`.
- **Steps:** the same window code as training (the rebuilt windows match the dataset exactly) → rules → photo model → fusion → cut-offs.
- **Output:** a validated `Answer` in JSON (decision, sentence, reason, risk, carb range).
- **Bad input:** malformed or empty input returns *can't judge* with the reason.
- **Packaging:** a Linux container runs offline on local weights; GitHub CI builds it and self-tests it.

## 3. Results
**Main estimate: 5-fold person CV (23 people, 481 meals), with 95% ranges from resampling people.**

| Model | AUROC (95% range) | AUPRC | Brier |
|---|---|---|---|
| Base rate (same guess for all) | 0.403* | 0.285 | 0.231 |
| Profile only (logistic regression) | 0.672 (0.567–0.764) | 0.487 | 0.204 |
| Profile + CGM trend (logistic regression) | 0.686 (0.604–0.772) | 0.547 | 0.199 |
| Profile + CGM trend (gradient boosting) | 0.689 (0.610–0.778) | 0.531 | 0.204 |
| Profile + SensorCNN (scratch) | 0.689 (0.603–0.773) | 0.551 | 0.201 |
| **+ photo (main model)** | **0.724 (0.659–0.796)** | 0.585 | 0.191 |
| + logged carbs (reference; not available in the product) | 0.717 (0.648–0.796) | 0.587 | 0.192 |
| + shuffled photo (control) | 0.679 (0.590–0.766) | 0.502 | 0.204 |

\*0.50 within each fold; pooling 5 folds' constant guesses pushes it below 0.5.

**Fixed test (5 people, 96 meals, scored once; every model trained on the same 28 people), AUROC:**

| Base rate | Profile only | Profile + trend | Gradient boosting | SensorCNN | **+ photo** |
|---|---|---|---|---|---|
| 0.500 | 0.765 | 0.695 | 0.632 | 0.674 | **0.742** |

**Interpretation:**

- **The photo is the only input that clearly adds signal:** +0.037 (+0.003 to +0.073), a small gain whose range barely clears zero.
- **Shuffling the photos removes the gain** (−0.008).
- **The photo matched logged carbs:** +0.007 (−0.020 to +0.032), non-inferior at the proposal's 0.03 margin. This answers RQ1: a photo can stand in for manual logging.
- **The carb-bin head carries most of this** (+0.020, +0.005 to +0.035, versus the grams-only version; a check added after seeing results).
- **The scratch CNN ties hand-made trend features** (+0.002), as the feedback expected.
- **Fine-tuning helped only partly.** It lost to frozen ConvNeXt + ridge on grams (22.5 vs 20.4 g; the median guess scores 27.7 g) but won on bins (F1 0.61 vs 0.38) and in the fusion.
- **On the 5 test people, profile-only scores highest.** Five people cannot separate these models, which is why the CV across 23 people is the main estimate.

**Calibration** (15 equal-width bins; meals per bin printed on the figure):

- **CV:** ECE 0.052, against a target of 0.05.
- **Fixed test:** ECE 0.129. The deployed 28-person model **over-predicts spikes between 0.3 and 0.6**: on average it says 31–38% where 19–29% spiked.
- **Effect on the cut-offs:** the cut-offs come from the costs alone and assume honest risks, so over-prediction pushes meals toward *caution*. That is safe, but less useful. Recalibration is planned for the final submission.

<img src="../outputs/evaluation/fig3_reliability_test.png" width="45%">

**Targets** (fixed before training; *met* = the whole 95% range meets it):

| Target | Result | Status |
|---|---|---|
| Photo adds to spike prediction | +0.037 (+0.003 to +0.073) | met |
| RQ1: photo vs logged carbs, within 0.03 | +0.007 (−0.020 to +0.032) | met |
| RQ2: CNN beats hand-made trends | +0.002 (−0.016 to +0.022) | tie, at risk |
| ECE ≤ 0.05 | 0.052 CV / 0.129 test | at risk |
| Cost saving ≥ 10% vs best simple policy | 9% (−4% to +23%) | at risk |
| Meals cleared ≥ 20% / spikes among cleared ≤ 10% | 14% / 12% | at risk |
| *Can't judge* ≤ 20% | 11% CV / 16% test | met |
| Response time p95 ≤ 300 ms (Mac CPU) | 285 ms | met |

**Test cases** (`outputs/demo/cases.md`; chosen by a fixed rule, not hand-picked):

| Case | Input | Expected | Actual |
|---|---|---|---|
| Normal | test meal, photo + full 2 h CGM (real rise −33 mg/dL) | low risk | **low risk** 11%, "under 30 g (58% sure)", 282 ms |
| Abstain | same kind of meal, last 40 min of CGM removed | can't judge | **can't judge**: "0 readings in the last 30 min … gap 40 min" |
| Abstain | grey picture instead of a meal | can't judge | **can't judge**: "photo looks unlike the meal photos…" |
| Error | real rise 35 mg/dL | low risk | **caution** 48%: a large carb portion (60–89 g), but this person reacted less than typical |

**Cut-off change before the test:**

- The pre-registered rule chose its cut-off (0.31) on the 4 validation people, who spike unusually rarely (19% vs 33%). It failed the planned stability check.
- The cut-offs were replaced, before the test was opened, by values derived from the costs alone: low risk below 0.143, caution from 0.20. They use no data, were committed with a hash, and the config is checked against that file.

## 4. Response to the proposal feedback
| Feedback | Response |
|---|---|
| Writing too dense | Tables first, short paragraphs, about 3 pages |
| Startup framing weak | Concrete user moment, no logging, reasons for *can't judge*, feedback loop (Section 1) |
| Photo model weak: ConvNeXt-Tiny/EfficientNet-B3 + carb-bin head | ConvNeXt-Tiny + bin head; the bins add +0.020 AUROC |
| Scratch CNN may not be justified | Kept tiny (5k parameters) as the required scratch model; it ties trend features |
| Fusion with quantiles and monotone constraint is overkill | Replaced by a logistic regression |
| Small dataset | Person-level CV main estimate, 95% ranges by person, the 5-person test as a sanity check |

## 5. Blockers, risks and plan
**Changes from the proposal:**

- ConvNeXt-Tiny instead of EfficientNet-B0.
- No Nutrition5k pre-training yet.
- Logistic-regression fusion instead of an MLP with quantiles.
- A nearest-neighbour photo check instead of DINOv2. It flags 3.1% of held-out meal photos and 5 of 6 synthetic non-meal pictures; plain white passes.
- Conformal intervals, W&B, the Emirati photos and the user test move to the final stage.

**Added after seeing results** (labelled in `docs/evaluation_protocol.md`): the C grid widened, the ridge photo arm, the carb-bin check, and the cut-off amendment.

**Risks:**

- One photo seed (the photo CV takes about 1.5 h).
- The two photo heads disagree on 51 of 90 test meals, so users see only the bin range.
- The test people also sit inside the CV.
- The model over-predicts for low-spike people.

**Tooling:** PyTorch on Python 3.14 needs uv's CPU index on Linux; Gradio pins Pillow below 13.

**Final submission:**

1. Recalibrate (Platt or temperature, fitted on CV predictions) and re-check the cut-offs.
2. Run the 8 locked people once.
3. Ablations with 3 seeds: bins vs grams, frozen vs fine-tuned. Make the two photo heads agree.
4. A stronger abstention path: DINOv2 vs the nearest-neighbour check on real non-meal photos, plus tiered escalation.
5. Robustness to CGM gaps and blurred photos, a latency breakdown, and explainability probes.
6. Experiment-tracker exports, and a think-aloud test with 3–5 users.
