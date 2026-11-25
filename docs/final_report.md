# GLIMPSE: final report
Before eating: a meal photo + the last 2 h of CGM glucose and heart rate + a short profile → **low risk**,
**caution** or **can't judge** (with a reason). Offline, local weights, one Mac.
Maitha Alhosani · MAAI7103 · November 2026 · `just install` → `just predict-folder <in> <out>`


**Bottom line.** A per-person intercept, fitted only on each person's earlier meals and replayed in
deployment order, is the one change whose direction held: AUROC rises in all five development folds (0.726
to 0.762) and log-loss falls, though the AUROC gain is not established and most of the system's edge is the
intercept's own. The photo failed its pre-registered gate and is not claimed as a risk signal, yet the
frozen fusion still uses it (§2). On the 8 locked people, scored exactly once, AUROC is 0.645 and the
usefulness targets are not met. The grader's folder command, without history or onboarding priors as the
samples are, clears 11% of development meals and 16% of those spike (§4). The numbers stand unadjusted.

| Status at a glance | development (481 meals) | locked eight (153) | served, no history: dev / locked |
|---|---|---|---|
| AUROC, shipped (95% person-clustered) | 0.762 (0.682–0.829) | 0.645 (0.540–0.736) | – |
| intercept's gain over raw, paired | +0.037 (−0.004, +0.071), 5 of 5 folds: not established | +0.033 (−0.054, +0.100) | – |
| same intercept on gradient boosting (sensors + profile) | 0.750; shipped minus it +0.013 (−0.022, +0.043) | – | – |
| photo gate, pre-registered | +0.025 (−0.010, +0.061): not established | not computed (§6) | – |
| ECE, registered bins (target ≤ 0.05) | 0.052: not assessable, a perfectly calibrated model's median here | 0.061, below that median (0.094); the one-shot record gave 0.098 equal-mass | 0.023 / 0.162 |
| usefulness targets (protocol labels) | on track: all 4 estimates meet, intervals cross | not met (at risk): 3 of 4 miss | not met: 2 of 4 miss / 2 of 4 miss |
| spikes among cleared meals (cap 10%) | 7.3% (2.9–14.2%) | 2 of 10 (exact 2.5–55.6%; person bootstrap 0–100%) | 9.6% / 21.9% |
| no history and no onboarding priors (the samples' path; dev, approximate) | – | – | AUROC 0.637; 11.4% cleared, 9 of 55 spike; cost −1.5% / – |
| robustness | flips between low risk and caution ≤1.8% at every registered level; 3.7% without heart rate | – | – |
| monitoring | photo drift fires only on a synthetic week; the outcome rule is too noisy | no rule cleanly catches the cohort shift | – |

*Every cell is traced in the sections below and tested against outputs/report_metrics/report_metrics.json and the other evidence files.*

## 1. System and tooling
The objective: develop and internally validate, on person-grouped splits, a pre-meal spike-risk model,
and test under pre-registered gates whether the meal photo adds signal beyond sensors and profile. An
adult with prediabetes in a CGM programme wants fewer spikes but stops logging food within days; they
photograph the meal and get an answer with its reason in one plain sentence. The CGM labels every meal
two hours later (`cgm_auto`), so the system can learn each
person without logging; abstentions name a next step (confirm carbs, a dietitian). A wellness nudge, not a medical device.

One scoring path (Figure 1): every answer here, the demo, the app and the grader's
`just predict-folder <in> <out>` go through the same `Pipeline.answer`, with plain-code input rules first
(§4); served-point, baseline and what-if rows are arithmetic on those logged answers. Without history the
folder command and the app answer at the population calibration, since the intercept needs labelled meals,
which the evaluation replays in deployment order. An optional `history.csv` per meal folder (earlier
answers' fusion logits and CGM labels) now feeds it through the same two-hour gate; the samples carry
none. A profile without onboarding priors (every instructor case) gets training medians drawn from all 37
non-locked people, type 2 diabetes included (spike rate 0.455, rise 48 mg/dL). Tooling: uv
+ Python 3.14 (locked), strict Pydantic throughout, a `just` recipe for every number, 168 tests in CI (151 run without the dataset or weights; 16 skip, 1 expected failure),
W&B offline runs replayed from the committed training histories, and a git history (from 2 Nov; earlier
ordering in the dated design docs) in which every pre-registration precedes its run.

![Figure 1: the one scoring path, its refusal exits and the labelling loop](../outputs/figures/f1_system.png)

| Parameters and cost | value | cap |
|---|---|---|
| largest model, deployed (ConvNeXt-Tiny + heads) | 27.8M | 200M |
| pretrained total | 27.8M | 1B |
| fine-tuned, deployed (last ConvNeXt stage + heads) | 15.5M | – |
| trained from scratch: heads 3,845 (also in the row above) + fusion 20 (deployed), SensorCNN 5,217 (evaluated only) | 9,082 | 50M |
| fine-tuning items (CGMacros photos; the planned 800 Nutrition5k items were cut) | 1,221 | 10,000 |
| cold first answer / warm median / p95 (model loaded once; measured 7 Nov on the frozen artifacts) | 1.57 s / 265 ms / 271 ms | ≤300 ms p95 |
| peak memory / checkpoint | 796 MB / 31 MB | – |

| Quality vs cost (development, all 481 meals, pooled over the same folds) | trained parameters | result |
|---|---|---|
| trend + profile features, logistic regression | 16 | AUROC 0.686 |
| SensorCNN, from scratch (mean ± sd of 3 seeds' AUROCs) | 5,217 | AUROC 0.688 ± 0.007 |
| photo: frozen ConvNeXt features + ridge (post hoc arm) | 769 | carb MAE 20.4 g, bin F1 0.38; fusion gain +0.017 (−0.008, +0.041) |
| photo: last stage + heads fine-tuned (shipped) | 15.5M | carb MAE 22.6 g, bin F1 0.61; fusion gain +0.025 (−0.007, +0.059) |

| Where serving time goes (one answer, 2-CPU cloud machine, median of 50) | ms |
|---|---|
| photo decode + transform | 11.3 |
| ConvNeXt-Tiny pass, fp32 (int8 dynamic: 29.7, decisions not compared) | 47.4 |
| kNN check, heads, fusion and cut-offs | 0.4 |
| whole photo path / no-photo path (sensors + profile only) | 59.1 / 0.35 |
| backbone's share of the photo path | 80% |

*Sensor rows: outputs/cv/cv_summary.json (§3 gives the same models on the 460 answered meals). Timing: outputs/efficiency/stage_timing.json, random-initialised backbone (same arithmetic as the pinned weights); the 265 ms warm median above is the author's Mac, end to end. Photo rows: outputs/photo_cv/, outputs/fusion/ (gain over no meal on all 481 meals; the gate's photo-meals interval is −0.010 to +0.061). Both photo rows cost one backbone pass at serving. Budget: outputs/param_budget.csv, outputs/efficiency/, outputs/report_metrics/.*

## 2. Multimodal design and the photo gate
Three parts. A pinned, SHA-verified ConvNeXt-Tiny backbone (ImageNet-22k pre-trained, 1k fine-tuned)
feeds two small heads, grams regression and a 4-bin carb classifier, with only the last backbone stage
fine-tuned. The profile comprises eight inputs: age, BMI, HbA1c, fasting glucose, meal hour (as sine and
cosine) and two onboarding priors (spike rate and mean rise). Hand-made trend features summarise the 2-h
CGM/HR window. SensorCNN, the required from-scratch model, is kept as the RQ2 baseline only: its paired
gain over the 16-parameter regression is +0.002 (−0.016, +0.022), failing the pre-registered bar, so the
hand-made features stayed.

Fusion is a 20-parameter logistic regression. With 23 development people and four inner folds behind
every choice, that is the capacity the data supports, and it is the course's prescription for this data
size (frozen pretrained features plus a simple classifier). Gradient boosting on the same 19 fusion
inputs is no better: the shipped regression is non-inferior under the pre-registered bar (paired +0.016,
−0.024 to +0.062). A sequence model with attention would face the bar the SensorCNN tie already settles.

| Photo recipe | value | how it was set |
|---|---|---|
| optimiser and rates | AdamW; heads 1e-3 (warm-up + cosine in the heads phase, constant after); last stage constant 1e-4 | fixed in the config (taught differential rates) |
| weight decay, batch | 0.05 (SensorCNN 0.01); 64 heads, 32 last stage | config; last-stage batch fixed in code |
| heads epochs per outer fold → shipped | 12–16 → 12 (median) | early stopping on 4 inner folds |
| last stage: keep, epochs → shipped | kept in 5 of 5 folds; 2–3 → 2 | kept only if inner MAE drops ≥0.5 g (it fell from 27.5–29.8 g to 24.2–26.1 g) |
| loss | Huber on log(1+grams) + 0.5 × bin cross-entropy | fixed in code |
| augmentation | flips, crops keeping 80% of the image area, brightness and contrast ±10%, no hue change; 2 saved views per photo, last-stage phase only | fixed in code, not ablated |

*Per-fold values in outputs/report_metrics/report_metrics.json (photo_recipe), from outputs/photo_cv/; settings in configs/default.yaml, src/glimpse/data/photos.py and src/glimpse/training/photo.py.*

| What the photo branch does (every committed result, gate first) | result |
|---|---|
| risk gate, pre-registered: AUROC gain over no meal information | +0.025 (−0.010, +0.061): not established |
| secondary endpoint, pre-registered: log-loss gain | +0.016, one-sided p = 0.094 |
| the same gate on the locked eight (D7's primary test) | not computed: the one locked run scored only the shipped pipeline (§6) |
| shuffled-photo control | −0.010 |
| participants' true logged carbs instead (RQ1 ceiling) | +0.024 (−0.003, +0.056) in the fusion arms; +0.031 (+0.003, +0.063) in the earlier trend + profile check |
| photo estimate vs logged carbs (margin 0.03) | +0.002 (−0.027, +0.027): non-inferior |
| D8: exact-macro training meals removed / size-matched random drop / paired difference | −0.005 / +0.037 / −0.041 (−0.093, +0.008) |
| carb estimate on held-out people: MAE, Spearman, bin F1 | 22.6 g, 0.53, 0.61 (ridge on frozen features 20.4 g, 0.56, 0.38; training median 27.7 g) |
| same recipe, 3 fresh seeds (intermediate split: 28 training, 4 validation people) | 24.9 ± 1.9 g, bin F1 0.55 ± 0.04 |
| grams head vs bin head, same bin | 51% (old aim 90%: missed) |
| refusals it adds | dev: 21 of 481 without a photo (the unfamiliar check does not apply to precomputed out-of-fold estimates); locked: 2 without a photo, 1 unfamiliar, of 153 |

*Sources: outputs/fusion/ (incl. carbs_check.json), outputs/ablations/, outputs/photo_cv/, outputs/training/photo_convnext_seeds.json, outputs/promised/, outputs/report_metrics/.*

Per the pre-commitment (docs/gate_s8.md held both outcome paragraphs before the run), the photo is not an established risk signal. That paragraph went further:
the answer would rest on trend and profile, with the photo as a carb-entry assist. That product was not
built. The 5 Nov amendment regenerated the existing photo-arm fusion, so the frozen model takes the photo
estimate as 4 of its 19 inputs and refuses a meal without a usable photo; the typed-carbs tier exists in
code, but no input sets it, and risk would not be monotone in typed carbs (T8). The verdict changed the claim, not the model; nothing is reshipped.
RQ1 says why the photo adds little: true carbs add at most about 0.03 here, a ceiling the photo
matches; D8 is consistent with the gain riding on repeated standardised meals. This leaves all but 20 of the
27.82M deployed parameters in a branch that failed its gate: the price of testing the photo properly. It buys better carb bins than the ridge (F1 0.61 against 0.38) though a worse
MAE and Spearman, where patients' own counts err by about 15–20% (Brazeau et al. 2013, Vasiloglou et al.
2018), plus the unfamiliar-photo refusal. Heart rate earns nothing either: retrained without it, AUROC changes by +0.007
(imputed at serving, +0.003); it stays only because the feature set was fixed before the ablation. The
backbone is 80% of the photo path's time; the gate's no-photo arm, given the same intercept, scores 0.741
(shipped minus it +0.021, −0.003 to +0.053) in 0.35 ms of model time, the product a next version should ship. The profile carries the system:
dropping it costs 0.122 (Figure 4).

## 3. Evaluation and ablations
The 45 CGMacros participants are healthy, prediabetic and T2D adults (ages 18 to 69, 64% female), each
wearing a CGM for about ten days. 158 of the 481 development meals (32.8%) spiked, with per-person rates
from 0% to 68%: per-person calibration is the central problem. In partition v2 the 23 development target
people form five outer folds, with four inner folds choosing everything (fusion C, Platt, m, the
cut-offs); 14 T2D people only train; 8 people have been locked since the proposal. splits/reconciliation.md asserts
each of the 481 development meals is scored exactly once. One declared
caveat: fold 1 contains the five people the intermediate evaluation already scored, and the calibration
fix was chosen after a diagnosis made on those people, so the development estimate carries selection
pressure the locked eight do not. The locked eight (§6) are the only clean measurement.

Postprandial response is person-specific (Zeevi et al. 2015; Berry et al. 2020). The calibration fix: a
per-fold Platt map on inner out-of-fold risks plus a penalised per-person intercept b_p(t), fitted only
on the person's earlier meals and replayed in deployment order under a 2-hour label-availability gate.
Platt scaling is temperature scaling plus a bias; the intercept adds one penalised parameter per
person. Within every fold Platt AUROC equals raw; the pooled 0.714 mixes five fold maps. Figure 2 shows raw and shipped per fold.

![Figure 2: per-fold AUROC](../outputs/figures/f2_per_fold_auroc.png)

| risk (answered meals) | AUROC (95% person-clustered) | AUPRC (shipped: 95% CI; else base rate) | Brier (shipped: 95% CI; else constant forecast) | log-loss | ECE, 15 bins: equal-width / equal-mass |
|---|---|---|---|---|---|
| dev baselines (460): prevalence / profile only / trend LR / gradient boosting on trend + profile / SensorCNN, 3 seeds' predictions averaged | 0.414 / 0.676 / 0.695 / 0.700 / 0.696 | 0.289 / 0.490 / 0.555 / 0.538 / 0.558 | 0.231 / 0.203 / 0.198 / 0.201 / 0.199 | – | – |
| dev raw fusion (460) | 0.726 (0.659–0.797) | 0.554 (0.328) | 0.192 (0.221) | 0.567 | 0.060 / 0.059 |
| dev + per-fold Platt | 0.714 pooled (= raw within every fold) | 0.541 | 0.194 | 0.572 | 0.023 / 0.056 |
| dev + personal intercept (shipped) | **0.762 (0.682–0.829)** | **0.591 (0.451–0.694)** | **0.180 (0.148–0.210)** | **0.537** | 0.052 / 0.049 |
| dev shipped, excluding fold 1 (370) | 0.757 (0.651–0.833) | 0.586 (0.335) | 0.185 (0.223) | 0.547 | 0.070 / 0.057 |
| locked raw fusion (150 of 153) | 0.611 (0.510–0.743) | 0.531 (0.420) | 0.251 (0.244) | 0.707 | 0.150 / 0.143 |
| locked shipped (150 of 153) | **0.645 (0.540–0.736)** | **0.547 (0.409–0.658)** | **0.230 (0.199–0.257)** | **0.650** | 0.061 / 0.098 |

*All AUROCs pooled over out-of-fold predictions (within-fold means are in the evidence files); the prevalence baseline pools five fold constants, hence below 0.5. Dev: 460 answered of 481 (21 had no photo). Locked: the 4 can't-judge answers are 2 without a photo, 1 unfamiliar photo (no risk) and 1 borderline (risk 0.158, scored). outputs/report_metrics/, outputs/evaluation_v2/, outputs/locked_eval/.*

| shipped minus raw, paired by meal (95% person-clustered) | development | locked |
|---|---|---|
| AUROC | +0.037 (−0.004, +0.071); higher in 5 of 5 folds | +0.033 (−0.054, +0.100) |
| Brier | −0.012 (−0.023, +0.000) | −0.021 (−0.059, +0.005) |
| log-loss | −0.030 (−0.060, −0.001) | −0.057 (−0.156, +0.009) |
| Brier reliability (calibration part), raw → shipped | 0.006 → 0.004 | 0.037 → 0.007 |
| Brier resolution (ranking part), raw → shipped | 0.035 → 0.043 | 0.029 → 0.021 |

*Paired person bootstrap, 2,000 draws; Brier parts over the 15 registered bins (Platt only, development: reliability 0.001, resolution 0.028). outputs/report_metrics/report_metrics.json.*

| the same per-person intercept on each model (dev, 460) | AUROC | shipped minus it (95% person-clustered) |
|---|---|---|
| profile only | 0.708 | +0.054 (+0.029, +0.090) |
| trend LR | 0.742 | +0.021 (−0.003, +0.051) |
| gradient boosting on trend + profile | 0.750 | +0.013 (−0.022, +0.043) |
| SensorCNN, 3 seeds averaged | 0.736 | +0.026 (+0.003, +0.059) |
| shipped fusion without its Platt map | 0.763 | −0.001 (−0.008, +0.005) |
| fusion without the photo (the gate's no-meal arm) | 0.741 | +0.021 (−0.003, +0.053) |

*The shipped replay applied to each model's out-of-fold logits: same folds, m and gated history; it reproduces the shipped risk exactly (largest difference 1e-16). No Platt map was fitted for the baselines. Profile only against the no-photo arm (0.708 against 0.741) is the CGM-removal contrast. report_metrics.json (fair_baselines).*

![Figure 3: reliability in the registered binning, development and locked (15 equal-width bins, meals per bin printed, hollow markers for bins under 5 meals; equal-mass bins hold 30 or 31 development meals, 10 locked)](../outputs/figures/f3_reliability.png)

| calibration | development, shipped | locked, shipped | served, Platt only: development / locked |
|---|---|---|---|
| ECE, registered 15 equal-width bins (target ≤ 0.05) | 0.052 | 0.061 | 0.023 / 0.162 |
| ECE a perfectly calibrated model shows at this size: median, 95th percentile | 0.052, 0.072 | 0.094, 0.136 | – |
| share of perfectly calibrated draws scoring at or above the observed ECE | 48% | 94% | – |
| ECE, 15 equal-mass bins (used here until 26 Nov, switch unrecorded) | 0.049 | 0.098 | 0.056 / 0.157 |
| calibration-in-the-large (mean risk − spike rate) | −0.015 | −0.010 | +0.002 / −0.108 |
| per-person offset b, logit: median (10th, 90th percentile) | −0.17 (−0.94, +0.77) | +0.45 (−0.43, +1.62) | 0 |
| median abs. b by the person's labelled meals: 0 / 1–4 / 5–9 / 10+ | 0 / 0.36 / 0.50 / 0.66 | 0 / 0.36 / 0.44 / 0.78 | – |
| at the cut-offs: risk 0.10–0.15 / 0.15–0.20 (predicted → observed, meals) | 0.13 → 0.08 (51) / 0.18 → 0.26 (57) | – | – |
| under-confident / over-confident bins (observed, meals) | 0.50–0.63 (0.61–0.75, 77) / 0.77 (0.62, 13) | 0.30 (0.45, 20) / 0.70 (0.50, 10) | – |

*Floor: labels drawn as Bernoulli(risk), 2,000 draws. Offset b = logit(shipped) − logit(Platt only). Served = what the folder command returns. outputs/report_metrics/report_metrics.json.*

The intercept's evidence is log-loss and every fold, not pooled AUROC, and most of the gain is its own:
with the same intercept, the fused system's margin over gradient boosting is not established. On
development it trades calibration (Platt alone is better) for resolution; on the locked eight it repairs
calibration, its offsets moving up toward the harder cohort, where the served Platt-only risk
under-predicts by 11 points. The registered ECE target was mis-specified for this size: a perfectly
calibrated model scores 0.052 on development. Discrimination is even across glycaemic status, but usefulness is not: the
intended users, people with prediabetes, get 10% of their meals cleared and are under-predicted (table). Per-person gaps shrank: rare spikers predicted 20.9% vs observed 14.2% (was
29.3%); frequent 42.6% vs 52.9% (was 37.0%). The deployed values were frozen from the all-development
replay by a pre-registered procedure (docs/final_design.md amendment): Platt (0.96, −0.20), m = 1.0,
cut-offs (0.15, 0.16). Like folds 1 and 5, whose bands have zero width, the replay chose a near-degenerate
abstention band; §4 reports what that costs.

| development by glycaemic status | people, meals, spike rate | AUROC (95% person-clustered) | mean risk − spike rate | cleared, spikes among them, people with a clear |
|---|---|---|---|---|
| healthy | 11, 224, 21.0% | 0.738 (0.566–0.859) | +0.036 | 37.5% (84), 7, 11 of 11 |
| prediabetes | 12, 257, 43.2% | 0.736 (0.648–0.796) | −0.061 | 9.7% (25), 1, 6 of 12 |

*AUROC and calibration over answered meals; shares over all meals. report_metrics.json (subgroups).*

| fold | m | cut-offs (lo, hi) |
|---|---|---|
| 1 | 1.0 | .21, .21 |
| 2 | 2.0 | .13, .23 |
| 3 | 0.5 | .13, .20 |
| 4 | 1.0 | .15, .17 |
| 5 | 1.0 | .14, .14 |

## 4. Trust and safety
Decisions are priced: a missed spike costs 5× an unnecessary caution, and abstaining costs 2 if the meal
spikes and 0.5 if not. Both prices are declared assumptions (5:1 is a ~17% action threshold; Vickers &
Elkin 2006), swept in a pre-registered grid that re-chooses the cut-offs per setting. The development
verdict holds only at the declared prices: at half the abstain price abstention takes 62–76% of meals,
far over its cap; at twice it, savings fall to 3.1% and missed-among-cleared rises to 11.7%.

| missed-spike cost ↓ / abstain price → (cost saved · cleared · missed among cleared) | ×0.5 | ×1 (declared) | ×2 | ×4 |
|---|---|---|---|---|
| 3× | 40.0% · 14.6% · 7.1% | 22.9% · 43.5% · 15.3% | 16.4% · 43.5% · 15.3% | 3.4% · 43.5% · 15.3% |
| 5× (declared) | 37.2% · 8.3% · 7.5% | **14.4% · 22.7% · 7.3%** | 3.1% · 26.6% · 11.7% | −9.9% · 26.6% · 11.7% |
| 7× | 36.8% · 1.9% · 11.1% | 6.5% · 11.9% · 8.8% | 2.2% · 20.6% · 8.1% | −10.8% · 20.6% · 8.1% |
| 10× | 38.2% · 0.4% · 0.0% | 4.5% · 8.3% · 7.5% | −8.0% · 11.9% · 8.8% | −21.1% · 11.9% · 8.8% |

*All 16 pre-registered settings, cut-offs re-chosen per fold for each; cost saved is against the better of always low risk and always caution, the protocol's simple policies (always abstaining would cost less at half the abstain price). outputs/evaluation_v2/dev_summary.json (cost_sweep).*

| Usefulness target (fixed before any result) | target | dev (481) | locked (153) | served, no history: dev / locked |
|---|---|---|---|---|
| cost reduction vs the best constant policy (always caution, both cohorts) | ≥10% | 14.4% (−4.6, +31.5) | −5.7% (−19.4, +5.9) | 8.7% (−4.6, +22.0) / −17.0% (−54.4, +17.8) |
| cleared as low risk | ≥20% | 22.7% (13.2–34.0%) | 6.5% (0.6–15.9%) | 15.2% / 20.9% |
| can't judge | ≤20% | 13.7% (8.7–19.2%) | 2.6% (0.6–5.0%) | 6.0% / 3.3% |
| missed among cleared | ≤10% | 7.3% (2.9–14.2%) | 20% (2 of 10; exact 2.5–55.6%; person bootstrap 0–100%) | 9.6% (7 of 73) / 21.9% (7 of 32) |
| the same without onboarding priors (dev, approximate) | – | – | – | 16.4% (9 of 55); 11.4% cleared; cost −1.5% / – |
| people with a cleared meal | – | 17 of 23 | 3 of 8 (8 of the 10 clears from one person) | – |
| mean cost per meal: shipped / always caution / always low risk | – | 0.575 / 0.672 / 1.642 | 0.608 / 0.575 / 2.124 | – |
| verdict (protocol labels: met / on track / at risk) | – | on track: estimates meet, intervals cross | not met (at risk): 3 of 4 miss | not met: 2 of 4 / 2 of 4 |

*Served = the deployed cut-offs on the Platt-only risk, recomputed from the committed answers (dev: per-fold models; locked: frozen Platt on the frozen fusion's raw risk); nothing refitted or rescored. outputs/report_metrics/report_metrics.json (decisions, errors, served_operating_point, served_imputed_priors, share_cis), outputs/figures/figures.json (the dev miss interval), outputs/locked_eval/locked_summary.json.*

In plain terms: of every 100 development meals the app clears 22.7, flags 63.6 and declines 13.7; of the
cleared, about 7 still spike, visible to a dietitian since the CGM reports every outcome. Only 17 people
had cleared meals, so the miss interval's upper end exceeds the cap; the 8 misses spread across 6 people. The frozen band (0.15, 0.16) is one point wide. On the locked cohort, where
risks ran higher, 139 of 153 meals landed above it: a dietitian would see nearly every meal flagged, and
always flagging is then the cheaper policy.
Without history the served point (last column) fails two targets on development and misses 7 of 32 clears
on the locked eight. The samples also lack onboarding priors: filled with the serving medians (development,
frozen coefficients standing in for the fold models'), AUROC drops to 0.637 and 9 of 55 cleared meals
spike, costlier than always flagging. Without history and priors no meal should be cleared. Abstention
does not pay here. Deciding the 45 borderline development meals would have cost less (0.539 against 0.555
per answered meal): they spiked at 27%, above the 20% at which abstaining beats caution at the declared
prices, because risks just above the low cut are under-predicted. Abstaining on the meals closest to the
cut (Figure 7) keeps the miss rate among cleared meals at 5.6–8.4% from 0 to 30% abstention while clears
fall from 24% to 10% of answered meals. Abstention is established practice (Kompa et al. 2021), but here
it buys little.

![Figure 4: effect sizes with 95% person-clustered intervals; removals are measured against the fusion study's photo arm (pooled 0.719), not the evaluation's 0.726](../outputs/figures/f4_effects.png)
![Figure 5: every registered perturbation, one run per condition (the table's levels, the meal-time shifts, missing heart rate), per-fold pipelines without history](../outputs/figures/f5_robustness.png)

| stress at three magnitudes | level 1 | level 2 | level 3 |
|---|---|---|---|
| CGM dropout 10/30/50% (same decision / low risk↔caution) | 100% / 0.0% | 99.8% / 0.0% | 99.6% / 0.2% |
| pre-meal gap 10/25/40 min (same decision / new refusals) | 100% / 0% | 11% / 89% | 11% / 89% |
| photo blur r=2/6/10 (same decision / low risk↔caution) | 98% / 0.7% | 94% / 0.9% | 89% / 1.7% |
| photo brightness ×0.5/×0.25/×0.1 (same decision / new refusals) | 98% / 1% | 96% / 2% | 10% / 90% |

*"Same decision" includes meals already refused before the perturbation. Pre-registered 22 Nov; outputs/robustness/levels.json, the run Figure 5 draws with timing.json and promised.json (outputs/robustness/robustness.json is the earlier 2 Nov single-level run under pre-freeze pipelines, kept for the record).*

![Figure 7: coverage vs risk, abstaining on the meals closest to the low-risk cut (development at each fold's cut, locked at 0.15; n of cleared meals printed), descriptive](../outputs/figures/f7_coverage.png)

Degradation is graded while inputs stay usable (blur) and turns into refusal at the designed thresholds
(the 20-minute gap rule, the contrast rule in near-darkness); past a rule every meal refuses, so the gap
and a misstated meal time are steps by design. Flips between low risk and caution, either direction, stay
below 1.8% at every level; with heart rate missing, 3.7%, the worst case. All stress runs use the
no-history per-fold pipelines. The 23 hostile inputs (timezone offsets, malformed JSON, binary CSVs,
broken photos, empty folders) give zero crashes; five refuse on a placeholder photo after parsing their CSV
stressor. R1-R5 and R8-R12 were each added against an observed failure; tests enforce them.

| monitoring signal (docs/monitoring.md) | demonstration | result |
|---|---|---|
| 1 photo drift: weekly median kNN distance above the training q90 (0.445) | simulated 4 weeks: 3 resampled training weeks, then 6 synthetic non-meal pictures | fires on week 4 only (0.709; weeks 1–3 0.25–0.37) |
| 1, on real photos | the 14 real phone photos | median 0.433: R7-style cuisine drift would not fire it |
| 2a missed-among-cleared above 10%, 80-answer windows | replay 1: dev answers at each fold's cut-offs, then locked | 7 of 21 ordinary dev windows (2–5 misses among 10–25 clears; overlapping windows over two runs of misses); locked windows too few clears; 1 of 3 mixed windows fires |
| 2a, deployed cut-offs on the evaluated (with-history) risk | replay 2 | 3 of 21 dev windows |
| 3 behaviour: caution share above the highest dev week | both replays | cannot fire: one dev week is all caution |
| calibration-in-the-large | both replays | pooled −0.015 dev, −0.010 locked; locked window median +0.038, against a registered expectation of negative |
| 4 calibration drift (median abs. b_p over months) | not computable from ten days per person | – |
| CUSUM on the served risk's calibration (replay 3, registered first) | dev then locked, blocks of 20, k = 0.5, h = 4 | no alarm (statistic 2.4 of 4), against a registered expectation of a locked alarm |
| CUSUM on the caution share of logged decisions (registered) | same | one alarm, 7th locked week; dev blocks used each fold's cut-offs, locked the deployed ones |
| the same at the deployed cut-offs (added after that result) | same | no alarm; the statistic climbs to 2.6 of 4 by the last locked week |

*Replays: logged answers as users arriving one after another, rules registered before each run; outputs/monitoring/.*

The photo-drift rule detects only pictures the per-answer gate already refuses, and the outcome rule
fails as an alarm at this volume. The registered caution-share rule fired once, but part of that step is
the cut-off change at the dev/locked seam: with both streams at the deployed cut-offs it climbs without
alarming. The expected calibration alarm did not come. No rule caught the real shift cleanly.

Feedback (src/glimpse/feedback.py): after a meal, a validated record comes from the user (ate it all,
part of it, none; optionally the rise they saw) or from the CGM itself at +2 h (`cgm_auto`: the training
rule on raw readings, so near 50 mg/dL it can disagree with the training label); "did not eat it" yields no label. Records are stored; the folder command accepts them as an optional
history.csv, the app not yet. The same records
would drive the outcome monitor and queue for human review before any retraining. The planned dietitian consultation did not complete: its three questions (cap
acceptability, decision wording, the rising-baseline case) are committed, and the cost matrix stays a
declared assumption hedged by the sweep.

## 5. Explainability and failure analysis
Every class in the taxonomy was observed; none is hypothetical.

| failure class | count | corpus | root cause | action |
|---|---|---|---|---|
| T1 rule refusal | 8 of 13; 23 of 23 | instructor cases; stress suite | input (format, units, gaps) | named refusal; rules R1-R12, before/after in the rule ledger |
| T2 missing-photo abstention | 21 of 481 | development meals | input | refuse; typed carbs not wired |
| T3 borderline abstention | 45 of 481 | development meals (folds 2–4 only) | threshold band | dietitian review |
| T4 near-cut false clear | 4 of 109 | cleared development meals | threshold: the closer half, 0.005–0.030 below the cut | none; the price of a hard cut (Figure 7) |
| T5 confident false clear | 4 of 109 | cleared development meals | model: the farther half, 0.047–0.138 below; two cold starts after steep falls, two with 10 and 20 labels | "falling now" check proposed, not shipped |
| locked false clear | 2 of 10 | cleared locked meals | threshold and cohort shift; 8 of 10 clears from one person | outcome monitor (too few clears to fire) |
| false caution (caution, no spike) | 175 of 306; 78 of 139 | development; locked cautions | one-point band, 33–42% base rate | none; cost-optimal at the declared prices |
| spike among can't judge | 19 of 66; 2 of 4 | development; locked | abstention near the base rate | dietitian sees the outcome |
| T6 confident wrong carb bin (R7) | 1 dish | instructor case 11; threshold sweep on 14 real phone photos | the embedding measures look, not nutrition | confirm-carbs prompt; nutrition-aware embedding later |
| T7 label-definition surprise | 1 probe | development counterfactual (probe B) | baseline-relative label | dietitian question; wording |
| T8 typed carbs not monotone | 1 tier | frozen fusion, typed 5–120 g | one-hot bins out of distribution | tier unwired; strict known-failure test |

*Counts from outputs/folder_run, outputs/stress_run, outputs/report_metrics/report_metrics.json (errors, false_clears_exploratory), outputs/ood/ and outputs/probes/.*

| T5 cold start (one earlier labelled meal) | pre-meal fall among the 109 cleared | four largest SHAP terms, logit |
|---|---|---|
| row 1024 | 3rd steepest | heart rate −0.39, window spread −0.35, prior rise +0.31, window mean −0.30 |
| row 847 | 2nd steepest | window spread −0.59, age −0.33, HbA1c −0.26, window rise +0.23 |

*Exploratory, chosen after the outcome; frozen fusion, fitted on these meals, not the fold models that answered. report_metrics.json (shap_terms_t5).*

T5 is the genuine blind spot, and no single mechanism explains it. For row 847 every sizeable profile and trend
term except window rise pulled risk down; for row 1024 the priors pushed it up, outweighed by heart rate,
spread and mean. Two rules chosen after the outcome trade clears for misses: leads, not results.

| exploratory rule, not shipped | development: clears, misses, cost saved | locked: clears, misses, cost saved |
|---|---|---|
| as shipped | 109, 8, 14.4% | 10, 2, −5.7% |
| "falling now": never clear a meal in the steepest-falling tenth of all meals | 99, 5, 16.9% | – |
| cold-start hold: no clear before the person's third labelled meal | 97, 6, 14.4% | 7, 1, −2.3% |

*Arithmetic on the committed answers; flipped clears become caution. report_metrics.json (falling_now_what_if, cold_start_what_if).*

Probe B was registered as an expected pass (a rising pre-meal trace raises risk on at least 90% of
meals) and failed: risk rose on only 8%. The label is baseline-relative, so a rising trace lowers
predicted risk; the registration encoded our naive physiology (docs/probes_s13.md). Exact SHAP for the frozen fusion (Figure 6; Platt and the intercept act
after it and are not attributed) ranks prior rise first, then
age, heart rate and prior spike rate. Attribution and ablation disagree for heart rate and for the four
photo terms (together 0.33 logit per meal against an unestablished gain): attribution measures in-sample
use, ablation out-of-sample loss. Heart rate is not redundant (R² 0.19 from the other inputs), so its
−0.21 coefficient is best read as in-sample signal that does not generalise.

![Figure 6: exact linear SHAP for the frozen fusion, in-sample (the sum to the logit is an arithmetic check); the local meal is the one whose out-of-fold raw risk is closest to 0.5](../outputs/figures/f6_shap.png)

Probe C, registered last, moved each profile input from its 10th to its 90th percentile: risk rises with
HbA1c, fasting glucose, prior spike rate and prior rise on every meal (median +0.06 to +0.12), changing
11–17% of decisions. Filling the priors with the serving medians changes 11% of decisions and cuts
in-sample clears from 68 to 46 (outputs/probes/probe_c.json).

Probe A: risk moves in the photo's direction on 100% of meals, which a linear fusion guarantees (a
wiring check, not learning), yet the extreme bottom↔top-decile plate swap leaves 68.5% of decisions unchanged (median swing 0.102), as pre-committed: the gate's verdict seen
mechanistically. R7 (rule ledger): the kNN check measures visual, not nutritional, familiarity. A local
lamb-and-rice dish (instructor case 11) scores 0.318 against the 0.614 cut (more familiar than the clean tuna bowl at 0.406)
and gets a confident wrong carb bin; the sweep under a ≤2% false-flag budget shows no threshold fixes it,
and the finding survived the final model's regenerated reference. Mitigations: the confirm-carbs tier
and per-answer distance monitoring; a nutrition-aware embedding is future work.

## 6. Reproducibility
The 8 locked subjects were scored once, on 25 Nov, from the frozen artifacts (locked rows of the §3
tables), through the same Pipeline.answer path the instructor command invokes. Decisions: 10 low risk,
139 caution, 4 can't judge; missed-among-cleared 2 of 10; cost reduction −5.7% against always caution.
The development estimate (0.762) lies above the locked interval: §3's selection pressure and the harder
cohort (42.0% of answered meals spiked vs 32.8%) both act here, inseparable at n=8. The intercept still beats raw on
the locked people at the point estimate on AUROC, Brier and log-loss, with paired intervals that include
zero. As pre-committed, the result is reported as measured, with no re-tuning after. D7 also
pre-registered the photo gate on the locked eight as the primary test; that run scored only the shipped
pipeline, so the endpoint was never computed and cannot be now.

After the locked run, a fresh clone on the Mac under the final serving code (install, data, pinned
weights, prepare, selftest, evaluate, figures, both input suites) reproduced every number in the report
of that day exactly (the record is outputs/finalize/fresh_clone_2026-11-25.md; the terminal log was not committed). `just evaluate` reruns the development evaluation, then `just report-numbers`
recomputes the descriptive metrics, replays and figures from committed outputs (no dataset or weights);
`just reproduce-all` chains the compute studies, and the README maps every table to its recipe. Tests check the main tables'
cells against those files on every commit; `just verify-outputs` fails if the data-free outputs drift.
The overnight round was committed in batches, so commit order, not clock time, shows each registration
preceding its run.

**AI-assisted development, verified.** Code and documents were built with an LLM assistant, as the
course expects, and the commits say so; the author set the goals and approved each pre-registration,
freeze and verdict before its run up to the tag; the overnight round's registrations were made by the
assistant under her standing instruction. The defence is the verification trail: a write-up stated n=397 where the
output says 460, caught by the number-to-file check. After the tag `submission-v1` (25 Nov), three cold grades by AI agents given
the rubric led to a dated round, run overnight under the author's standing instruction and listed in
docs/changelog.md; no frozen artifact or locked number changed.

## 7. Limitations and conclusion
CGMacros is de-identified public PhysioNet data collected under the source study's ethics approval; no
new human data. Forty-five people from one study, repeated standardised meals, no local cuisine (R7). T2D participants are training-only, so no claim is made for them. Heart rate adds nothing here
and would be dropped in a next version; the app should take a person's history and priors,
and the product should become the sensor-first one the gate described. The photo is not established as
a risk signal, nor is the intercept's AUROC gain, ECE sits at its noise floor, and the label's baseline-relative definition can diverge from a
user's intuition (probe B). The errors fall on different people. A false clear tells someone with prediabetes a spiking meal is safe,
the costliest error and the one the cap guards. On a harder cohort the opposite fails: with 91% of locked
meals flagged, the nudge becomes noise users learn to ignore and a dietitian cannot triage. This is an
advisory prototype with no clinical validation, end-user test or
regulatory review, built so that a clinical study could evaluate it.

| Taught technique (course week) | where |
|---|---|
| group split, pristine locked set (4–5) | §3, §6 |
| nested CV, thresholds over class weights (2) | §3, outputs/promised |
| wall-clock as a first-class number (3) | §1 |
| frozen features + simple classifier, differential rates, hue-preserving augmentation (5–6) | §1, §2 |
| shuffled-photo control, ceiling analysis, error taxonomy, pre-registration, subgroup caution, graceful degradation (11) | §2, §4, §5 |
| cost-chosen thresholds, refusal to human review, validated outputs, worst-slice reporting (9) | §4 |
| sequence models measured against the bar (8–10) | §2 (RQ2) |
| not used, reasons in the text: attention, full fine-tuning, ensembles (beyond averaging three SensorCNN seeds for its baseline), self-supervised pretraining | §2 |

| Glossary | meaning |
|---|---|
| RQ1, RQ2 | the proposal's questions: does the photo match true logged carbs (RQ1); does a from-scratch CNN beat hand-made trend features (RQ2) |
| D1-D8 | dated design decisions in docs/final_design.md (D7: the gate's endpoints; D8: the exact-macro exclusion test) |
| R1-R12 | input rules in docs/rule_ledger.md: R1-R5 and R8-R12 added after observed failures, R6 a preventive check, R7 a recorded limitation |
| T1-T8 | failure classes (§5) |
| partition v2 | the final split: 23 development people in 5 outer folds, 14 T2D training-only, 8 locked |
| cgm_auto | the CGM's own label at +2 h, the training spike rule |
| Platt, b_p, m | population recalibration of the fusion logit; the per-person offset; its shrinkage |
| served point | the deployed cut-offs on the no-history risk, what the folder command returns |

**References.** Zeevi et al., Cell 2015, doi:10.1016/j.cell.2015.11.001 · Berry et al., Nature
Medicine 2020, doi:10.1038/s41591-020-0934-0 · Vickers & Elkin, Med Decis Making 2006,
doi:10.1177/0272989X06295361 · Brazeau et al., Diabetes Res Clin Pract 2013 (carb counting, T1D),
doi:10.1016/j.diabres.2012.10.024 · Vasiloglou et al., Nutrients 2018 (GoCARB),
doi:10.3390/nu10060741 · Kompa, Snoek & Beam, npj Digital Medicine 2021,
doi:10.1038/s41746-020-00367-3 · CGMacros: Scientific Data 2025, doi:10.1038/s41597-025-05851-7,
and PhysioNet v1.0.0.
