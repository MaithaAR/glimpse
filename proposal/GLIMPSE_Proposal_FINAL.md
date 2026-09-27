# GLIMPSE: Pre-Meal Glucose-Spike Risk from a Meal Photo, CGM Trend and Profile

**Maitha Alhosani (solo)** · MAAI7103 Deep Learning · 27 September 2026 · Code and logs: `~/projects/glimpse`, commit `117746b`

## 1. User, problem and system

**Problem.** Prediabetes and wellness programmes lend members a continuous glucose monitor (CGM) for a few weeks, but a CGM shows a spike only *after* eating. GLIMPSE (*Glycemic-response Inference from Meal Photos and Sensor Evidence*) gives a calibrated spike risk *before* the first bite. The user is an adult with prediabetes; the buyer is the programme provider. January AI already predicts responses from food photos. GLIMPSE's untested bet is that live pre-meal glucose, calibration and explicit abstention make that guidance more trustworthy.

**Scenario.** Mariam photographs rice with chicken at lunch. Her glucose has been rising since mid-morning. The risk card shows **caution**, a predicted-rise interval and the inputs behind it (large carbohydrate estimate, rising trend). A carbohydrate slider shows how the *model's* risk changes, not dietary advice. When the photo is unreadable, the CGM has a gap or confidence is low, the card **abstains**.

**Capabilities.**
- Three actions (low risk, caution, abstain) chosen by an explicit cost model.
- A 72-hour onboarding sets a personal prior; with fewer than 5 meals, a population prior is used.
- A dietitian review queue receives abstentions and a 5% audit. Monitoring tracks calibration drift, abstention, out-of-distribution (OOD) rate and latency.
- No dosing, hypoglycaemia or causal claims. CGMacros has no medication data.

| Component | Model | Params | Training |
|---|---|---|---|
| Photo → served carbs, protein, fat, fibre, kcal | EfficientNet-B0 | 5.3M | **Fine-tuned** (Nutrition5k, then CGMacros) |
| Pre-meal CGM and heart-rate encoder (24 × 5 min) | 1-D CNN | 0.1–1M | **From scratch** |
| Fusion: spike logit, 10/50/90% rise quantiles | MLP, monotone carb path | <1M | From scratch |
| Photo OOD distance | DINOv2-S | 22M | Frozen |

The total is about 29M parameters. The app is offline Gradio in PyTorch/Docker, with a p95 latency target of ≤300 ms on a laptop CPU.

## 2. Data, splits and experiments

**Data.** CGMacros (PhysioNet, CC BY-NC-SA) follows 45 adults (15 healthy, 16 prediabetes, 14 type 2 diabetes) for about 10 days. It records Dexcom CGM, Fitbit heart rate, before-meal photos, labs and consumed macros with the share eaten. Nutrition5k (5,006 plates) provides photo pre-training.

- **Reference macros:** consumed ÷ share eaten gives "served" macros, which are noisy (share below 10% is excluded).
- **Inputs available before eating:** photo, CGM and heart rate up to the meal, profile, labs and time.
- **Label:** Dexcom peak rise ≥50 mg/dL above the 30-min pre-meal mean within 2 h (30 and 70 are sensitivity thresholds).
- **Usable window:** ≥3 baseline readings, ≥70% post-meal coverage, no gap over 20 min.

**Counts.** 1,526 meals pass the window rules. After removing 453 onboarding meals and 19 that cross the 72-h boundary, 1,054 operational meals remain. Of these, 896 are isolated (no further meal within 2 h), and 634 of those belong to the 31-person **target cohort** (healthy plus prediabetes, 35.2% spikes). The downloaded photo set is 1,486 before-photos. Type 2 diabetes participants are used for training only.

**Splits.**
- **Primary:** nested person-grouped cross-validation (CV) on the development people. Every learned stage (photo head, fusion, temperature, conformal quantile, thresholds) is fit only on each outer fold's training people. Fusion trains on inner out-of-fold photo predictions.
- **Secondary:** a locked set of 8 target people (IDs frozen with a seed and SHA-256 hash in `splits/locked_final_set.json`). They appeared in the pilot, so it is internal evidence only.

**RQ1: does a photo preserve the meal signal?** Three models have identical sensor, profile and prior inputs and differ only in meal information: *none*, *photo-predicted* or *reference* served macros. All intervals are paired person-bootstrap 95% CIs (2,000 resamples of people). Decisions are fixed in advance:
- **Non-inferior** if the lower bound of Δ = AUROC(photo) − AUROC(reference) exceeds −0.03, **inferior** if the upper bound is below −0.03, otherwise **inconclusive**.
- **Photo adds value** if the lower bound of G_photo = AUROC(photo) − AUROC(none) exceeds 0.

**RQ2: learned vs engineered sensor features.** Paired ΔAUROC of the CNN encoder vs engineered trend features, with everything else identical.

**One executable evaluation.** Running `python3 scripts/03_photo_experiment.py` builds out-of-fold photo predictions for 5 person folds × 3 seeds, fits the three gradient-boosted (GBM) classifiers per fold, and writes carbohydrate MAE, bias, AUROCs, Δ and G_photo with their CI bounds and **interval width** to `results/photo_experiment.json`. The final evaluation reruns the same script with the fine-tuned image model and neural fusion.

## 3. Evidence, success criteria and plan

**Photo pilot (person-excluded, completed 27 September).**
- **Setup:** frozen ImageNet EfficientNet-B0 features with a ridge head predicting log served macros (α = 300, fixed before running). The head is nested inside every outer fold.
- **Scoring:** 611 matched isolated target-cohort meals (31 people, 35.0% spikes) that have a photo.
- **Source:** `results/photo_experiment.log`.

| Meal information (sensor, profile and prior identical) | AUROC | Gain over none (95% CI) |
|---|---|---|
| None | 0.675 | n/a |
| Reference served macros | 0.752 | +0.077 (+0.044, +0.117) |
| **Photo-predicted macros (person-excluded)** | **0.722** | **+0.047 (+0.019, +0.077)** |
| Random-noise images (negative control) | 0.678 | +0.004 (−0.013, +0.021) |

- **Carbohydrates:** MAE 26.2 g (vs 27.2 g for a constant predictor), bias −4.5 g, Spearman 0.37, against a mean reference of 52.6 g. Absolute accuracy is weak, but the ranking signal is real.
- **RQ1 now:**
  - Photo adds value: the G_photo lower bound is above 0.
  - Non-inferiority is **inconclusive**: Δ = −0.030 (−0.064, +0.002), a width of 0.066 at 31 people.
  - At this width, non-inferiority requires the fine-tuned model to roughly match the reference (Δ ≳ 0).
- **Robustness:** the gain stays positive at α = 100 (+0.030) and α = 1,000 (+0.050) (`results/photo_experiment_alpha_sensitivity.log`). The random-image control (`results/negative_control_random_images.log`) shows the pipeline does not leak the label.
- **Earlier pilots (`pilot/`):**
  - Neural network vs GBM: −0.013 (−0.051, +0.028).
  - Learned vs engineered sequence features: −0.021 (−0.042, +0.005). RQ2 remains open.
  - Expected calibration error (ECE): 0.037–0.040.

All pilot figures include the locked people and are optimistic.

**Success criteria (target cohort).**
- **Discrimination:** RQ1 and RQ2 as defined above; fusion AUROC ≥ a GBM given the same inputs.
- **Calibration:** ECE ≤0.05. Conformal rise intervals with 75–85% pooled coverage at 80% nominal (no guarantee claimed).
- **Usefulness,** on the operational cohort with costs as assumptions (low risk 5/0, caution 0/1, abstain 2/0.5 for spike / no spike), swept in sensitivity analysis:
  - cost ≥10% below the best trivial policy;
  - ≥20% of meals cleared, with ≤10% of those spiking;
  - abstention ≤20%.

**Milestones.**
- **Week 4 gate:** fine-tuned carbohydrate MAE below 26.2 g and Δ re-estimated.
- **Week 6:** RQ2.
- **Week 7:** calibration and abstention.
- **Week 10:** app.
- **Week 11:** locked set run once.

**Minimum deliverable and fallback.** The minimum deliverable is the fused CNN + MLP model with calibration, abstention and the app. If fine-tuning fails the week-4 gate, or G_photo loses its positive lower bound, the app switches to **confirm-carbs mode**: the photo proposes an estimate and the user adjusts it. RQ1 is then reported as a negative result. The course requirements (three modalities, a fine-tuned and a from-scratch model) hold in either case.

**Product assumption, not yet validated.** The assumption is that programme dietitians would accept "cannot judge this meal" rather than a forced answer. It will be tested in week 2 with two short dietitian conversations and a 5-person think-aloud (can users tell *abstain* from *low risk*?), and the result reported as found.

**Limitations.**
- There are 45 participants, with a single dataset and cuisine.
- Served macros assume proportional eating.
- A 9-photo spot check flagged packaging and drinks as a hard category.
- The Emirati-dish OOD set (30 photos) is exploratory.
- The licence restricts use to coursework.
- There is no overlap with MAAI7102.
