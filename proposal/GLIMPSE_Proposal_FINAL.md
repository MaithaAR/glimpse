# GLIMPSE: Does a Meal Photo Preserve the Glucose-Spike Signal? A Pre-Meal Risk Card from Photo, Sensor and Profile Data

**Team:** Maitha Alhosani (solo) · **Course:** MAAI7103 Deep Learning · **Proposal date:** 27 September 2026

Before eating, a person with prediabetes photographs her plate. GLIMPSE (*Glycemic-response Inference from Meal Photos and Sensor Evidence*) combines the photo with her last two hours of continuous glucose monitor (CGM) and heart-rate data and her profile. It returns a calibrated risk of a large glucose spike, or it abstains.

The project asks two questions:
- **RQ1:** Does a photo recover most of the predictive value that logged meal information adds over sensor and profile data alone?
- **RQ2:** Does a learned pre-meal sensor representation beat engineered trend features?

## 1. The venture and its user

Some prediabetes and wellness programmes give members a CGM for a few weeks. The CGM shows a spike only after eating, which is too late to change the decision.

- **User:** an adult with prediabetes or a wellness member.
- **Buyer:** the programme provider, such as an employer or insurer wellness scheme.
- **Differentiation (an untested hypothesis):** January AI already predicts glucose responses from food photos without a sensor. GLIMPSE's bet is that adding live pre-meal glucose, calibration and explicit abstention makes this guidance more trustworthy.

**Scenario.** At lunch, Mariam photographs rice with chicken. Her glucose has been rising since a mid-morning coffee with dates. GLIMPSE shows *caution*. The card gives a high spike probability, a predicted-rise interval and the relevant observed inputs (a large carbohydrate estimate and a rising trend). These inputs are shown as context, not claimed as causes. A slider shows how the *model's* prediction changes with a lower carbohydrate estimate. This is model sensitivity, not dietary advice.

**Journey.**
- **Onboarding:** the user enters a profile. Labs are optional; missing labs are handled with missingness indicators, lab-dropout training, and a separate no-labs evaluation.
- **Personal prior:** meals whose 2-hour windows finish within the first 72 hours set the user's spike rate and mean rise. With fewer than 5 such meals, a population prior from the training target cohort is used. The 72-hour sensor period is acknowledged adoption friction.
- **At each meal**, GLIMPSE takes one action:
  - *low risk*;
  - *caution*, with the relevant inputs;
  - *abstain* ("cannot judge this meal", plus generic guidance). This covers an unreadable or unfamiliar photo, a CGM gap, or low confidence.
- **Afterwards:** abstained meals and a 5% random audit go to a dietitian queue, for learning only. The target is ≤0.5 reviews per user per day, which is about 1 minute of dietitian time. Users can correct the carbohydrate estimate, and completed CGM windows label each prediction.

**Boundaries.** GLIMPSE makes no dosing, hypoglycaemia or causal claims. CGMacros has no medication data, so the intended exclusion of insulin and sulfonylurea users cannot be verified. The data licence (CC BY-NC-SA) limits use to coursework.

## 2. The system concept

GLIMPSE has four capabilities:
1. A **risk card**.
2. A **carbohydrate-sensitivity slider**.
3. **Abstention** with a review queue.
4. **Feedback and monitoring**, covering confidence, abstention rate, calibration drift, out-of-distribution (OOD) rate and latency.

Three modalities each have their own encoder: the **before-meal photo**, a **CGM and heart-rate series** (24 five-minute steps), and **static tabular context** (profile, labs, time of day).

| Component | Model | Params | Notes |
|---|---|---|---|
| Photo → served carbs, protein, fat (**fine-tuned**) | EfficientNet-B0 | ~5M | Trained on the Nutrition5k overhead subset (<5,006), then CGMacros development before-photos (<1,640). Fibre is added at the CGMacros stage. |
| Sensor encoder (**from scratch**) | 1-D CNN | 0.1–1M | The smallest size is a serious candidate for 24 steps. |
| Fusion (**from scratch**) | MLP | <1M | Outputs a spike logit, rise quantiles, and meal-only and sensor-only auxiliary heads. |
| OOD embedding (**frozen**) | DINOv2-S | 22M | Measures a photo's distance to the training data. |

**Training.**
- **Image model:** Huber loss on log-macros. The backbone is frozen first, then its last blocks are unfrozen.
- **Fusion model:** binary cross-entropy for the spike, pinball loss for the 10/50/90% rise quantiles, and auxiliary-head losses weighted 0.3, with weight decay and early stopping. Inner folds select sizes, loss weights and freezing depth.
- **Carbohydrate path:** carbohydrate enters the spike logit only through a non-negative-weighted term, so predicted risk cannot rise when the carbohydrate estimate alone falls.
- **Abstention signals:** head disagreement and DINOv2 distance are each kept only if they improve the risk–coverage curve over a simpler rule based on the fused probability.

**Budget.** About 27M pretrained or fine-tuned parameters (limit 1B) and under 2M trained from scratch. The latency target is p95 ≤300 ms, offline, in PyTorch inside arm64 Docker on the author's MacBook Pro CPU (4 threads, chip logged).

## 3. The data and learning plan

**Data sources.**
- **CGMacros** (PhysioNet): 45 adults (15 healthy, 16 prediabetes, 14 type 2 diabetes), each followed for about 10 days. It includes Dexcom and Libre CGM, Fitbit heart rate, before- and after-meal photos, labs, and macros estimated for the *consumed* meal with a percentage consumed.
- **Nutrition5k:** 5,006 cafeteria plates, with overhead images for a subset. Per its official repository (to be confirmed on download), it has carbohydrate, protein and fat labels but no fibre. How well it transfers to phone photos is evaluated.

**Reference macros are noisy supervision.** The before-photo shows the served plate, so the reference is consumed macros divided by share eaten, after normalising the inconsistently coded share field.
- This assumes proportional consumption, which *may fail* when someone leaves particular components uneaten.
- A meal's labels are **unusable** if its share is below 10%, it has no before-photo, or an audit of the photo pair finds a clear photo–log mismatch.
- Only 8.8% of meals are partly eaten, so a fully-consumed-only analysis tests the assumption.
- Consumed macros are reported as a *retrospective comparator*.

**Inputs and label.**
- **Inputs** are only the before-photo, CGM and heart rate up to the meal, and profile, labs and time.
- **Label:** a Dexcom peak rise of at least 50 mg/dL over the 30-minute pre-meal mean within 2 hours. This threshold is an operational definition; 30 and 70 mg/dL are also reported.
- **Usable windows:** the baseline has at least 3 readings, at least 70% of post-meal readings are present, and no gap exceeds 20 minutes, since a missed peak can hide a spike. A stricter version (≥90% present, no gap over 10 minutes) is also reported.

**Pilot accounting.** 1,526 meals pass all rules. Removing 453 onboarding meals, 19 that cross the 72-hour boundary and 158 followed by another meal within 2 hours leaves **896**. Of these, 634 come from the 31-person **target cohort** (healthy plus prediabetes), where the spike base rate is 35.2%. Overlapping meals are also reported separately.

| Set | People | Use |
|---|---|---|
| Development | 23 target + 14 type 2 diabetes (training only) | **Primary analyses** by nested person-grouped cross-validation (CV). Within each outer fold, only its training people are used. Inner 4-fold cross-fitting produces out-of-fold predictions that fit temperature scaling (probabilities), conformal corrections and cost thresholds. No separate, repeatedly reused calibration set exists. |
| Locked final set | 8 target (4 healthy, 4 prediabetes) | A secondary check. IDs were frozen on 27 September (seeded draw, SHA-256 hash). These people were in the exploratory pilot, so the set is disclosed as internal evidence only. It evaluates new users after onboarding. |
| Emirati stress set | 30 dishes (to be collected) + 30 in-distribution photos | An exploratory visual-OOD check. The threshold accepts 95% of development photos. |

**Leakage controls.**
- Every learned stage uses the same person exclusions.
- Fusion trains on **out-of-fold photo predictions**.
- Near-duplicate photos will be removed.
- A *macro-profile group* test holds out meals with identical logged macros together. CGMacros has no food identifiers, so this tests generalisation across macro profiles rather than recipes. About 30 recurring groups will be checked by eye.
- A 9-photo spot check found 3–4 packaging or drink photos. Photos like these will form a hard-example set.

**Feasibility milestone (week 4).** Carbohydrate mean absolute error (MAE) and bias on at least 100 audited CGMacros photos, and a first RQ1 estimate on the development folds.

## 4. Scope, evidence, and success

**Deliverables.** An offline containerised Gradio app (risk card, slider, abstention, review queue, monitoring), plus W&B logs, git history and a reproducibility package. The pilot scripts, logs and locked-set file are already committed.

**Out of scope:** dosing, hypoglycaemia, causal claims, substitution advice and live CGM integration.

**Stretch goals:** a text-note branch, a pretrained time-series encoder, self-supervised pretraining and int8 quantisation.

**Exploratory pilot.** A gradient-boosted tree (GBM) was trained on all 45 people and scored as AUROC on held-out target-cohort meals (5-fold person CV × 3 seeds). The locked set is included, so these numbers are optimistic.

| Inputs besides sensor, profile and personal prior | AUROC |
|---|---|
| None (no meal information) | 0.660 |
| Reconstructed served macros | 0.741 |
| Consumed macros (retrospective) | 0.740 |
| Served macros, fully consumed meals only | 0.741 (no-meal: 0.665) |

Meal information adds **+0.083** (95% CI +0.050 to +0.122). A small from-scratch network *tied* the GBM (−0.013, CI −0.051 to +0.028). Learned sequence features trailed engineered trends (−0.021, CI −0.042 to +0.005), so RQ2 is genuinely open.

**RQ1 (primary: development CV; secondary: locked set).** Three otherwise identical models (same sensor, profile and prior inputs; no image embedding) are compared: *no meal information*, *photo-predicted served macros* and *reconstructed served macros*. Let Δ = AUROC(photo) − AUROC(reconstructed), with a paired person-bootstrap 95% CI.
- **Non-inferior** if the lower bound of Δ exceeds −0.03.
- **Inferior** if the upper bound is below −0.03.
- **Inconclusive** otherwise.

The photo model must *also* beat the no-meal model. The 0.03 margin is about 36% of the pilot meal-information gain, so passing means a photo retains roughly two-thirds of that gain. An illustrative simulation (multiplicative log-normal macro noise, CV 25%) suggested a CI width of about 0.04 over the target cohort and about 0.06 for 8 people. Real photo errors may be systematic, so this is not a power guarantee. Carbohydrate MAE and bias are reported, and the image embedding is tested separately.

**RQ2.** A paired ΔAUROC compares the learned sensor encoder with engineered trends, with all other inputs identical.

**Exploratory targets** (target cohort, with counts and denominators):
- **Discrimination:** fusion AUROC ≥ a *fair GBM* given the same photo-predicted macros, engineered sensor features, profile and prior.
- **Rise:** MAE below a GBM rise regressor.
- **Calibration:** expected calibration error (ECE) ≤0.05 (15 equal-mass bins), with reliability diagrams.
- **Intervals:** participant-weighted split-conformal quantile regression, weighting each person's scores by 1/their meal count. No guarantee is claimed under within-person dependence. Target pooled coverage is 75–85% at 80% nominal, with person-averaged coverage and width reported.
- **Usefulness:** action proportions are reported separately. The system counts as useful only if all of the following hold:
  - expected cost is ≥10% below the better of *always caution* and *always low risk*;
  - ≥20% of meals are cleared as low risk, with ≤10% of cleared meals spiking;
  - abstention is ≤20%.

  Zero cleared meals counts as failure.

**Cost assumptions** (spike / no spike): low risk 5 / 0, caution 0 / 1, abstain 2 / 0.5. The sweeps vary the missed-spike cost from 3 to 10 and scale both abstain costs by 0.5–4×.

**Formative evidence.** A 5-person think-aloud test checks that users can tell *abstain* from *low risk*. Latency is measured on the stated hardware.

**Explainability and robustness.**
- **Probes:** photo swaps with the sensor state fixed; flat vs rising traces at equal current glucose; branch masking. These complement, not replace, the retrained no-meal baseline.
- **Failure taxonomy:** by photo type, health group and trend.
- **Stress tests:** blur and angle, CGM gaps, a switch from Dexcom to Libre, missing heart rate or labs, and a sensor-free mode.

There is no overlap with MAAI7102 (1–5-year data-centre power forecasting).
