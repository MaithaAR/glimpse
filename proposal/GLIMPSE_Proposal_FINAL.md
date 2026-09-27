# GLIMPSE: Does a Meal Photo Preserve the Glucose-Spike Signal? A Pre-Meal Risk Card from Photo, Sensor and Profile Data

**Team:** Maitha Alhosani (solo) · **Course:** MAAI7103 Deep Learning · **Proposal date:** 27 September 2026

Before eating, a person with prediabetes photographs her plate. GLIMPSE (*Glycemic-response Inference from Meal Photos and Sensor Evidence*) combines the photo with the last two hours of her continuous glucose monitor (CGM) and heart-rate data, plus her profile. It returns a calibrated risk of a large glucose spike, or abstains.

**Research questions.**
- **RQ1:** Does a photo recover the predictive value that logged meal information adds over sensor and profile data alone?
- **RQ2:** Does a learned pre-meal sensor representation beat engineered trend features?

## 1. The venture and its user

Some prediabetes and wellness programmes give members a CGM for a few weeks. It only shows a spike after eating, which is too late to change the decision.

- **User:** an adult with prediabetes, or a wellness member.
- **Buyer:** the programme provider, an employer or insurer wellness scheme.
- **Differentiation (untested hypothesis):** January AI already predicts glucose responses from food photos without a sensor. GLIMPSE bets that live pre-meal glucose, calibration and explicit abstention make that guidance more trustworthy.

**Scenario.** At lunch, Mariam photographs rice with chicken. Her glucose has been rising since a mid-morning coffee with dates. GLIMPSE shows *caution*: a high spike probability, a predicted-rise interval, and the relevant inputs it observed (a large carbohydrate estimate and a rising trend). These are shown as context, not causes. A slider shows how the *model's* prediction changes with less carbohydrate. That is model sensitivity, not dietary advice.

**Journey.**
- **Onboarding:** a profile, with labs optional. Missing labs are handled by missingness indicators and lab-dropout training, and evaluated separately. Meals whose windows finish within the first 72 hours set a personal prior: spike rate and mean rise. With fewer than 5 such meals, GLIMPSE uses a population prior from the training target cohort. The 72-hour wait is acknowledged friction.
- **At each meal:** one of three actions:
  - *low risk*;
  - *caution*, with the relevant inputs;
  - *abstain*: "cannot judge this meal", plus generic guidance. This is triggered by an unreadable or unfamiliar photo, a CGM gap or low confidence.
- **Afterwards:** abstained meals and a 5% audit go to a dietitian queue for learning. The target is ≤0.5 reviews per user per day, about 1 minute. Users can correct carbohydrate estimates. Completed CGM windows label predictions.

**Boundaries.** GLIMPSE makes no dosing, hypoglycaemia or causal claims. CGMacros has no medication data, so the intended exclusion of insulin and sulfonylurea users cannot be verified. The licence (CC BY-NC-SA) limits use to coursework.

## 2. The system concept

**Capabilities:**
1. A **risk card**.
2. A **carbohydrate-sensitivity slider**.
3. **Abstention** with a review queue.
4. **Feedback and monitoring** of confidence, abstention, calibration drift, out-of-distribution (OOD) rate and latency.

**Modalities, each with its own encoder:**
- the **before-meal photo**;
- a **CGM and heart-rate series** (24 five-minute steps);
- **tabular context** (profile, labs, time of day).

| Component | Model | Params | Notes |
|---|---|---|---|
| Photo → served carbs, protein, fat (**fine-tuned**) | EfficientNet-B0 | ~5M | Nutrition5k overhead subset (<5,006), then CGMacros development before-photos (<1,640). Fibre is added at the CGMacros stage. |
| Sensor encoder (**from scratch**) | 1-D CNN | 0.1–1M | The smallest is a serious candidate for 24 steps. |
| Fusion (**from scratch**) | MLP | <1M | Spike logit, rise quantiles, meal-only and sensor-only auxiliary heads. |
| OOD embedding (**frozen**) | DINOv2-S | 22M | Photo distance to the training data. |

**Training.**
- **Image model:** Huber loss on log-macros. The backbone is frozen first, then its last blocks are unfrozen.
- **Fusion:** cross-entropy for spikes, pinball loss for the 10/50/90% rise quantiles, and auxiliary losses weighted 0.3, with weight decay and early stopping. Inner folds select sizes, loss weights and freezing depth.
- **Carbohydrate path:** carbohydrate enters the spike logit only through a non-negative-weighted term, so predicted risk cannot rise when the carbohydrate estimate alone falls.
- **Abstention signals:** head disagreement and DINOv2 distance are kept only if they beat a simpler rule based on the fused probability on the risk–coverage curve.

**Budget.** About 27M pretrained or fine-tuned parameters (limit 1B), plus under 2M trained from scratch. The latency target is p95 ≤ 300 ms, offline, in PyTorch and arm64 Docker, on the author's MacBook Pro CPU (4 threads, chip logged).

## 3. The data and learning plan

**CGMacros** (PhysioNet) follows 45 adults (15 healthy, 16 with prediabetes, 14 with type 2 diabetes) for about 10 days. It records Dexcom and Libre CGM, Fitbit heart rate, before- and after-meal photos, labs, and macros estimated for the *consumed* meal with a percentage consumed.

**Nutrition5k** has 5,006 cafeteria plates, with overhead images for a subset. Per its official repository (to be confirmed on download), it has carbohydrate, protein and fat labels, but no fibre. Transfer from these cafeteria photos to phone photos is evaluated.

**Reference macros are noisy supervision.** The before-photo shows the served plate, so the reference is consumed macros ÷ share eaten, after normalising the inconsistently coded share field.
- This assumes proportional consumption, which *may fail* when particular components are left uneaten.
- Labels are unusable if the share is below 10%, the before-photo is missing, or a photo-pair audit finds a clear mismatch.
- Only 8.8% of meals are partly eaten, so a fully-consumed-only analysis tests the assumption.
- Consumed macros serve as a *retrospective comparator*.

**Inputs and label.**
- **Inputs** are only the before-photo, CGM and heart rate up to the meal, and profile, labs and time.
- **Label:** a Dexcom peak rise of ≥50 mg/dL over the 30-minute pre-meal mean within 2 hours. This is an operational definition; 30 and 70 mg/dL are also reported.
- **Usable windows** need ≥3 baseline readings, ≥70% of post-meal readings, and no gap over 20 minutes, because a missed peak can hide a spike. A stricter rule (≥90% of readings, no gap over 10 minutes) is also reported.

**Pilot accounting.** 1,526 meals pass all rules. Removing 453 onboarding meals, 19 that cross the 72-hour boundary, and 158 followed by another meal within 2 hours leaves **896**. Of these, 634 come from the 31-person **target cohort** (healthy plus prediabetes), with a 35.2% spike rate.

| Set | People | Use |
|---|---|---|
| Development | 23 target + 14 type 2 diabetes (training only) | **Primary analyses** by nested person-grouped cross-validation (CV). Each outer fold uses only its own training people. Inner 4-fold cross-fitting gives out-of-fold predictions, which fit the temperature, the conformal quantile and the cost thresholds. These are applied to the fold's model refitted on all its training people, then judged on the untouched outer fold. |
| Locked final set | 8 target (4 healthy, 4 prediabetes) | A secondary check. IDs were frozen on 27 September (seeded draw, SHA-256 hash). These people were in the pilot, so results here are internal evidence only. |
| Emirati stress set | 30 dishes (to be collected) + 30 in-distribution photos | An exploratory visual-OOD check. The threshold accepts 95% of development photos. |

**Evaluation populations.** RQ1 uses the common set of meals with a usable before-photo. Operational metrics (actions, abstention, workload, cost) use *all* eligible meals. Meals with a missing or unusable photo count as abstentions.

**Leakage controls.**
- Every learned stage shares the same person exclusions.
- The fusion model trains on **out-of-fold photo predictions**.
- Near-duplicate photos will be removed.
- A *macro-profile group* test holds meals with identical logged macros out together. It tests macro profiles, not recipes, because CGMacros has no food identifiers. About 30 recurring groups will be checked by eye.
- A 9-photo spot check found 3–4 packaging or drink photos. These will form a hard-example set.

**Feasibility milestone (week 4).** Carbohydrate mean absolute error (MAE) and bias on ≥100 audited CGMacros photos, from person-excluded predictions only, plus a first RQ1 estimate on the development folds.

## 4. Scope, evidence, and success

**Deliverables.** An offline containerised Gradio app (risk card, slider, abstention, review queue, monitoring), plus W&B logs, git history and a reproducibility package. The pilot scripts, logs and locked-set file are already committed.

**Out of scope:** dosing, hypoglycaemia, causal claims, substitution advice and live CGM integration.

**Exploratory pilot.** Gradient-boosted trees (GBM) were trained within each fold on that fold's training people, from all groups. Scores are AUROC on held-out target-cohort meals (5-fold person CV × 3 seeds). The locked set is included, so these results are optimistic.

| Inputs besides sensor, profile and prior | AUROC |
|---|---|
| None (no meal information) | 0.660 |
| Reconstructed served macros | 0.741 |
| Consumed macros (retrospective) | 0.740 |
| Served macros, fully consumed meals only | 0.741 (no-meal: 0.665) |

- **Meal information** adds **+0.081**, with a paired person-bootstrap 95% CI of +0.050 to +0.122.
- **Neural network vs GBM:** the difference between a small from-scratch network and the GBM was inconclusive (−0.013, CI −0.051 to +0.028).
- **Learned vs engineered sequence features:** learned features trailed engineered trends (−0.021, CI −0.042 to +0.005), so RQ2 is genuinely open.

**RQ1 (primary: development CV; secondary: locked set).** Three otherwise identical models are compared. They share the same sensor, profile and prior inputs and use no image embedding. They differ only in meal information: *none*, *photo-predicted* served macros, or *reconstructed* served macros.

Let Δ = AUROC(photo) − AUROC(reconstructed), with a paired person-bootstrap 95% CI.
- **Non-inferior** if the lower bound exceeds −0.03.
- **Inferior** if the upper bound is below −0.03.
- **Inconclusive** otherwise.

**Incremental value.** The photo model shows incremental value only if the 95% CI of G_photo = AUROC(photo) − AUROC(none) excludes zero. A positive point estimate alone is exploratory.
- **Reporting:** G_photo and G_reference are reported separately. A retained-gain percentage is descriptive only, because it is unstable when G_reference is small.
- **Margin:** 0.03 is about one-third of the *pilot* meal-information gain.
- **Simulation:** an illustrative run with multiplicative log-normal macro noise (CV 25%) gave CI widths of about 0.04 for the target cohort and 0.06 for 8 people. Real photo errors may be systematic, so this is not a power guarantee.
- **Also reported:** carbohydrate MAE and bias. The image embedding is tested separately.

**RQ2.** A paired ΔAUROC compares the learned sensor encoder with engineered trends, with all other inputs identical.

**Exploratory targets** (target cohort, reported with counts and denominators):
- **Discrimination:** fusion AUROC ≥ a *fair GBM* given the same photo-predicted macros, engineered sensor features, profile and prior.
- **Rise:** MAE below that of a GBM rise regressor.
- **Calibration:** expected calibration error (ECE) ≤ 0.05 (15 equal-mass bins), with reliability diagrams.
- **Intervals:** cross-fitted conformal quantile regression. The conformal quantile comes from the inner out-of-fold residuals, weighting each person by 1/(their meal count). No coverage guarantee is claimed. Target pooled coverage is 75–85% at 80% nominal, with person-averaged coverage and width reported.
- **Usefulness:** requires all three of the following, with action proportions reported and zero cleared meals counting as failure:
  - expected cost ≥ 10% below the better of *always caution* and *always low risk*;
  - ≥ 20% of meals cleared as low risk, with ≤ 10% of those spiking;
  - abstention ≤ 20%.

**Costs** (spike / no spike) are assumptions: low risk 5/0, caution 0/1, abstain 2/0.5. Sweeps vary the missed-spike cost from 3 to 10 and scale the abstain costs by 0.5–4×.

**Formative checks.** A 5-person think-aloud test checks that users can tell *abstain* from *low risk*. Latency is measured on the stated hardware.

**Explainability and robustness.**
- **Probes:** photo swaps with the sensor state held fixed; flat vs rising traces at equal glucose; branch masking. These complement the retrained no-meal baseline.
- **Failure analysis:** failures are grouped by photo type, health group and trend.
- **Stress tests:** blur and camera angle, CGM gaps, Dexcom-to-Libre transfer, missing heart rate or labs, and a sensor-free mode.

No overlap with MAAI7102 (1–5-year data-centre power forecasting).
