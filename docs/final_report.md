# GLIMPSE: final report
Before eating: a meal photo + the last 2 h of CGM glucose and heart rate + a short profile → **low risk**,
**caution** or **can't judge** (with a reason). Offline, local weights, one Mac.
Maitha Alhosani · MAAI7103 · November 2026 · `just install` → `just predict-folder <in> <out>`


**Bottom line.** The central fix this phase was calibration. A per-person intercept, fitted only on each
person's earlier meals and replayed in deployment order, improves AUROC in all five development folds
(0.726 to 0.762 pooled) and lowers log-loss (paired −0.030, interval −0.060 to −0.001). By the standard
the photo gate is held to, the pooled AUROC gain, +0.037 (−0.004 to +0.071), is not established, and the
registered ECE target (0.05) is missed narrowly (0.052). The photo went through a pre-registered gate and
did not pass (+0.025 AUROC, −0.010 to +0.061), so, per the paragraph committed before the run, the
shipped product is sensor + confirmed carbs, with the photo head as a carb-entry assist. 8 of 109
cleared development meals spiked (7.3%, interval 2.9–14.2% against a pre-registered 10% cap). Severe input
degradation triggers refusal; flips between low risk and caution stay below 1.8% at every registered
stress level (3.7% with heart rate missing). On the 8 locked people (153 meals, 42.5% spike
rate), scored exactly once: AUROC 0.645 (0.540–0.736), raw 0.611; 10 cleared, 139 caution, 4 can't judge;
2 of the 10 cleared spiked and the cost reduction was −5.7%, failing the pre-set usefulness targets. Per
the pre-registration these numbers stand unadjusted and the development estimate does not replace them.

## 1. System and tooling
The objective: develop and internally validate, on person-grouped splits, a pre-meal spike-risk model,
and test under pre-registered gates whether the meal photo adds signal beyond sensors and profile. An
adult with prediabetes in a CGM programme wants fewer spikes but stops logging food within days; they
photograph the meal and get an answer with its reason in one plain sentence. The CGM labels every meal
two hours later (`cgm_auto`, the training spike rule), so the system can learn each person without
logging; abstentions escalate to confirm-carbs or a dietitian. A wellness nudge, not a medical device.

One scoring path: every decision and risk here, the demo, the app and the grader's
`just predict-folder <in> <out>` go through the same `Pipeline.answer`; plain-code input rules run first
(§4). The folder command and the app answer each meal without history, at the population calibration:
the personal intercept needs a person's labelled meals, which the evaluation replays in deployment order
and a deployed app would collect from `cgm_auto`; the folder format has no history field yet. Tooling: uv
+ Python 3.14 (locked), strict Pydantic throughout, a `just` recipe for every number, 157 tests in CI,
W&B offline runs replayed from the committed training histories, and a git history (from 2 Nov; earlier
ordering in the dated design docs) in which every pre-registration precedes its run (Figure 1).

![Figure 1: the one scoring path, its refusal exits and the labelling loop](../outputs/figures/f1_system.png)

| Parameters and cost | value | cap |
|---|---|---|
| largest model, deployed (ConvNeXt-Tiny + heads) | 27.8M | 200M |
| pretrained total | 27.8M | 1B |
| fine-tuned, deployed (last ConvNeXt stage + heads) | 15.5M | – |
| trained from scratch: heads 3,845 + fusion 20 (deployed), SensorCNN 5,217 (evaluated only) | 9,082 | 50M |
| fine-tuning items (CGMacros photos; the planned 800 Nutrition5k items were cut) | 1,221 | 10,000 |
| cold first answer / warm median / p95 (model loaded once; measured 7 Nov on the frozen artifacts) | 1.57 s / 265 ms / 271 ms | ≤300 ms p95 |
| peak memory / checkpoint | 796 MB / 31 MB | – |

| Quality vs cost (development, pooled over the same folds) | trained parameters | result |
|---|---|---|
| trend + profile features, logistic regression | 16 | AUROC 0.686 |
| SensorCNN, from scratch (3 seeds) | 5,217 | AUROC 0.688 ± 0.007 |
| photo: frozen ConvNeXt features + ridge | 769 | carb MAE 20.4 g, bin F1 0.38; fusion gain +0.017 (−0.008, +0.041) |
| photo: last stage + heads fine-tuned (shipped) | 15.5M | carb MAE 22.6 g, bin F1 0.61; fusion gain +0.025 (−0.007, +0.059) |

*Sensor rows: outputs/cv/cv_summary.json. Photo rows: outputs/photo_cv/, outputs/fusion/ (gain over no meal on all 481 meals; the gate's photo-meals interval is −0.010 to +0.061). Both photo rows cost one backbone pass at serving. Budget: outputs/param_budget.csv, outputs/efficiency/, outputs/report_metrics/.*

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
| optimiser and rates | AdamW; heads 1e-3 with warm-up + cosine; last stage constant 1e-4 | fixed in the config (taught differential rates) |
| weight decay, batch | 0.05 (SensorCNN 0.01); 64 heads, 32 last stage | fixed in the config |
| heads epochs per outer fold → shipped | 12–16 → 12 (median) | early stopping on 4 inner folds |
| last stage: keep, epochs → shipped | kept in 5 of 5 folds; 2–3 → 2 | kept only if inner MAE drops ≥0.5 g (it fell from 27.5–29.8 g to 24.2–26.1 g) |
| loss | Huber on log(1+grams) + 0.5 × bin cross-entropy | fixed in the config |
| augmentation | flips, crops keeping 80% of the plate, brightness and contrast ±10%, no hue change; 2 saved views per photo, last-stage phase only | fixed in the config |

*Per-fold values in outputs/report_metrics/report_metrics.json (photo_recipe), from outputs/photo_cv/; settings in configs/default.yaml.*

| What the photo branch does (every committed result, gate first) | result |
|---|---|
| risk gate, pre-registered: AUROC gain over no meal information | +0.025 (−0.010, +0.061): not established |
| shuffled-photo control | −0.010 |
| participants' true logged carbs instead (RQ1 ceiling) | +0.024 (−0.003, +0.056) in the fusion arms; +0.031 (+0.003, +0.063) in the earlier trend + profile check |
| photo estimate vs logged carbs (margin 0.03) | +0.002 (−0.027, +0.027): non-inferior |
| D8: exact-macro training meals removed / size-matched random drop / paired difference | −0.005 / +0.037 / −0.041 (−0.093, +0.008) |
| carb estimate on held-out people: MAE, Spearman, bin F1 | 22.6 g, 0.53, 0.61 (ridge on frozen features 20.4 g, 0.56, 0.38; training median 27.7 g) |
| same recipe, 3 fresh seeds, validation people | 24.9 ± 1.9 g, bin F1 0.55 ± 0.04 |
| grams head vs bin head, same bin | 51% (old aim 90%: missed) |
| refusals it adds | dev: 21 of 481 without a photo (the unfamiliar check does not apply to precomputed out-of-fold estimates); locked: 2 without a photo, 1 unfamiliar, of 153 |

*Sources: outputs/fusion/ (incl. carbs_check.json), outputs/ablations/, outputs/photo_cv/, outputs/training/photo_convnext_seeds.json, outputs/promised/, outputs/report_metrics/.*

Per the pre-commitment (docs/gate_s8.md committed both outcome paragraphs before the run; the losing one
was deleted on the verdict), the photo is a carb-entry assist, not an established risk signal. RQ1 says
why: true carbs add at most about 0.03 over profile + trend on this cohort, a carbohydrate-information
ceiling the photo already matches, and D8 is consistent with the small photo gain riding on CGMacros'
repeated standardised meals. This leaves 27.8M of the 27.83M parameters, 15.5M of them fine-tuned, in the
branch that failed its gate. That is the price of testing the photo properly. What the budget buys is
the carb assist: better carb bins than the ridge (F1 0.61 against 0.38) though a worse MAE and Spearman,
where patients' own counts err by about 15–20% (Brazeau et al. 2013, Vasiloglou et al. 2018), plus the
unfamiliar-photo refusal. Heart rate earns nothing either: dropping it changes AUROC by +0.007, and it
stays only because the feature set was fixed before the ablation. The profile carries the system:
dropping it costs 0.122 (Figure 4).

## 3. Evaluation and ablations
The 45 CGMacros participants are healthy, prediabetic and T2D adults (ages 18 to 69, 64% female), each
wearing a CGM for about ten days; meals without a usable 2-hour pre-meal window are excluded by the app's
own checks. 158 of the 481 development meals (32.8%) met the spike definition, with per-person rates from
0% to 68%: per-person calibration is the central problem. In partition v2 the 23 development target
people form five outer folds; four inner folds inside each fold's training people are the validation
split for every choice (fusion C, Platt, the shrinkage m, the cut-offs); 14 T2D people are training-only
and never scored; 8 people have been locked since the proposal. splits/reconciliation.md asserts
each of the 481 development meals is scored exactly once. One declared
caveat: fold 1 contains the five people the intermediate evaluation already scored, and the calibration
fix was chosen after a diagnosis made on those people, so the development estimate carries selection
pressure the locked eight do not. The locked eight (§6) are the only clean measurement.

Postprandial response is person-specific (Zeevi et al. 2015; Berry et al. 2020). The calibration fix: a
per-fold Platt map on inner out-of-fold risks plus a penalised per-person intercept b_p(t), fitted only
on the person's earlier meals and replayed in deployment order under a 2-hour label-availability gate.
Platt scaling is temperature scaling plus a bias; the intercept adds one penalised parameter per
person. Within every fold Platt AUROC equals raw; the pooled 0.714 mixes five fold maps (Figure 2).

![Figure 2: per-fold AUROC](../outputs/figures/f2_per_fold_auroc.png)

| risk (answered meals) | AUROC (95% person-clustered) | AUPRC (base rate) | Brier (constant forecast) | log-loss | ECE, 15 bins: equal-width / equal-mass |
|---|---|---|---|---|---|
| dev baselines: prevalence / profile only / trend LR / gradient boosting / SensorCNN | 0.414 / 0.676 / 0.695 / 0.700 / 0.696 | 0.289 / 0.490 / 0.555 / 0.538 / 0.558 | 0.231 / 0.203 / 0.198 / 0.201 / 0.199 | – | – |
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

*Paired person bootstrap, 2,000 draws; Brier parts over the 15 registered bins. outputs/report_metrics/report_metrics.json.*

![Figure 3: reliability in the registered binning, development and locked (15 equal-width bins, meals per bin printed; equal-mass bins hold 30 or 31 development meals, 10 locked)](../outputs/figures/f3_reliability.png)

The intercept's evidence is log-loss and every fold, not pooled AUROC. On development it mostly adds
resolution; on the locked eight it mostly repairs calibration. Against the registered calibration target
(ECE ≤ 0.05; the protocol's equal-width bins, 15 in the config) the shipped risk scores 0.052 on
development, a narrow miss, and 0.061 locked; Platt alone scores 0.023, so the intercept trades some
calibration for resolution. This report used equal-mass bins until 26 Nov (0.049, a pass) without
recording the switch; both are now shown. Where it errs: under-confident at 0.50–0.63 on development
(observed 0.61–0.75, 77 meals) and at 0.30 locked (0.45, 20 meals); over-confident near 0.70 locked
(0.50, 10 meals). Discrimination is even across glycaemic status: AUROC 0.738 for the 11 healthy people
(217 meals, 21% spike rate), 0.736 for the 12 prediabetic (243 meals, 43%), though the cohort limits
firmer fairness claims. Per-person gaps shrank: rare spikers predicted 20.9% vs observed 14.2% (was
29.3%); frequent 42.6% vs 52.9% (was 37.0%). The deployed values were frozen from the all-development
replay by a pre-registered procedure (docs/final_design.md amendment): Platt (0.96, −0.20), m = 1.0,
cut-offs (0.15, 0.16). Like folds 1 and 5, whose bands have zero width, the replay chose a near-degenerate
abstention band; §4 reports what that costs.

| fold | m | cut-offs (lo, hi) |
|---|---|---|
| 1 | 1.0 | .21, .21 |
| 2 | 2.0 | .13, .23 |
| 3 | 0.5 | .13, .20 |
| 4 | 1.0 | .15, .17 |
| 5 | 1.0 | .14, .14 |

## 4. Trust and safety
Decisions are priced, not guessed: a missed spike costs 5× an unnecessary caution, and abstaining costs 2
if the meal spikes and 0.5 if not (a dietitian's look is cheaper than a miss and dearer than nothing).
These are declared assumptions; the 5:1 ratio is a ~17% action threshold in decision-curve terms (Vickers
& Elkin 2006). A pre-registered sweep over both prices re-chooses the cut-offs per setting. Pricing a
miss higher clears fewer meals; the miss rate among them does not fall monotonically. The development
verdict holds only at abstain prices up to the declared one: at twice that price, cost savings fall to
3.1% and missed-among-cleared rises to 11.7%, failing two targets.

| missed-spike cost ↓ / abstain price → (cost saved · cleared · missed among cleared) | ×0.5 | ×1 (declared) | ×2 | ×4 |
|---|---|---|---|---|
| 3× | 40.0% · 14.6% · 7.1% | 22.9% · 43.5% · 15.3% | 16.4% · 43.5% · 15.3% | 3.4% · 43.5% · 15.3% |
| 5× (declared) | 37.2% · 8.3% · 7.5% | **14.4% · 22.7% · 7.3%** | 3.1% · 26.6% · 11.7% | −9.9% · 26.6% · 11.7% |
| 7× | 36.8% · 1.9% · 11.1% | 6.5% · 11.9% · 8.8% | 2.2% · 20.6% · 8.1% | −10.8% · 20.6% · 8.1% |
| 10× | 38.2% · 0.4% · 0.0% | 4.5% · 8.3% · 7.5% | −8.0% · 11.9% · 8.8% | −21.1% · 11.9% · 8.8% |

*All 16 pre-registered settings, cut-offs re-chosen per fold for each; outputs/evaluation_v2/dev_summary.json (cost_sweep).*

| Usefulness target (fixed before any result) | target | dev (481) | locked (153) |
|---|---|---|---|
| cost reduction vs the best constant policy (always caution, both cohorts) | ≥10% | 14.4% | −5.7% |
| cleared as low risk | ≥20% | 22.7% | 6.5% |
| can't judge | ≤20% | 13.7% | 2.6% |
| missed among cleared | ≤10% | 7.3% (2.9–14.2%) | 20% (2 of 10; 0–100%) |
| people with a cleared meal | – | 17 of 23 | 3 of 8 (8 of the 10 clears from one person) |
| mean cost per meal: shipped / always caution / always low risk | – | 0.575 / 0.672 / 1.642 | 0.608 / 0.575 / 2.124 |
| verdict (protocol labels) | – | on track at the point estimate | fails |

*outputs/report_metrics/report_metrics.json (decisions, errors), outputs/locked_eval/locked_summary.json.*

In plain terms: of every 100 development meals the app clears about 23, flags 64 and declines 14; of the
cleared, about 7 still spike, visible to a dietitian since the CGM reports every outcome. Only 17 people
had cleared meals, so the miss interval's upper end exceeds the cap; the 8 misses spread across 6 people. The frozen band (0.15, 0.16) is one point wide. On the locked cohort, where
risks ran higher, 139 of 153 meals landed above it: a dietitian would see nearly every meal flagged, and
always flagging is then the cheaper policy. That is the practical face of the usefulness failure (§6).
Abstaining instead of guessing is established practice for medical ML (Kompa et al. 2021).

![Figure 4: effect sizes with 95% person-clustered intervals](../outputs/figures/f4_effects.png)
![Figure 5: every registered perturbation, one run per condition (the table's levels, the meal-time shifts, missing heart rate), per-fold pipelines without history](../outputs/figures/f5_robustness.png)

| stress at three magnitudes | level 1 | level 2 | level 3 |
|---|---|---|---|
| CGM dropout 10/30/50% (unchanged / low risk↔caution) | 100% / 0.0% | 99.8% / 0.0% | 99.6% / 0.2% |
| pre-meal gap 10/25/40 min (unchanged / new refusals) | 100% / 0% | 11% / 89% | 11% / 89% |
| photo blur r=2/6/10 (unchanged / low risk↔caution) | 98% / 0.7% | 94% / 0.9% | 89% / 1.7% |
| photo brightness ×0.5/×0.25/×0.1 (unchanged / new refusals) | 98% / 1% | 96% / 2% | 10% / 90% |

*Pre-registered 22 Nov; outputs/robustness/levels.json, the same run Figure 5 draws (outputs/robustness/robustness.json is the earlier 2 Nov single-level run, kept for the record).*

Degradation is graded while inputs stay usable (blur) and turns into refusal at the designed thresholds
(the 20-minute gap rule, the contrast rule in near-darkness). The gap and an early-stated meal time are
steps by design: past the rule every meal refuses (a meal time stated 15 minutes early refuses 89%).
Flips between low risk and caution, counted in either direction, stay below 1.8% at every registered
level; with heart rate missing and imputed, 3.7%, the worst case measured. Every stress run uses the
per-fold pipelines without history, the operating point the folder command serves. Stress suite: 23
hostile inputs (timezone offsets, malformed JSON, binary CSVs, broken photos, empty folders) give zero
crashes and zero raw error rows; rules R8-R10 were added only against observed failures of a recorded
naive run, like R1-R5, and R11 against ten escapes a red-team round found; tests enforce all of them.

Monitoring has four signals with alert rules (docs/monitoring.md). On a simulated four-week stream the
photo-drift rule fires on the drifted week only (weekly median 0.709 against the 0.445 rule; weeks one to
three 0.25–0.37), but that week is six synthetic non-meal pictures, and the 14 real phone photos sit just
below the rule (median 0.433): cuisine drift like R7's would not trip it. Replaying the logged answers,
development then the locked eight (users arriving one after another, not calendar time), under rules
registered before each run, tests the outcome and behaviour rules, and they fail as alarms. Missed-among-cleared crosses the 10% cap in 7 of 21 ordinary development windows of 80 answers at
each fold's cut-offs (2–5 misses among 10–25 cleared meals), and in 3 of 21 re-decided at the deployed
cut-offs. On the locked cohort every window has too few cleared meals to judge, and the behaviour rule
(caution share above the highest development week) cannot fire, because one development week is all
caution. At this volume only the photo rule is a usable alarm; the outcome rule needs pooling over
months or a sequential test (future work, not a post-hoc change). Calibration-in-
the-large was expected negative on the locked stream and measured +0.038 (outputs/monitoring/).

Feedback (src/glimpse/feedback.py): after a meal, a validated record comes from the user (ate it all,
part of it, none; optionally the rise they saw) or from the CGM itself at +2 h (`cgm_auto`, the training
label rule); "did not eat it" yields no label. Records are stored; feeding them back as a person's
history exists in the evaluation replay, not yet in the app or the folder command. The same records
would drive the outcome monitor and queue for human review before any retraining. The planned dietitian consultation did not complete: its three questions (cap
acceptability, decision wording, the rising-baseline case) are committed, and the cost matrix stays a
declared assumption hedged by the sweep.

## 5. Explainability and failure analysis
Every class in the taxonomy was observed; none is hypothetical.

| failure class | count | corpus | what happens |
|---|---|---|---|
| T1 rule refusal | 8 of 13; 23 of 23 | instructor cases; stress suite | structured refusal naming the problem; the other 5 instructor cases answer, two with flags |
| T2 missing-photo abstention | 21 of 481 | development meals | no photo, no guess |
| T3 borderline abstention | 45 of 481 | development meals (folds 2–4 only) | the band exists only where cost optimisation chose one |
| T4 near-cut false clear | 4 of 109 | cleared development meals | the closer half: 0.005–0.030 below the fold's cut |
| T5 confident false clear | 4 of 109 | cleared development meals | the farther half: 0.047–0.138 below, 4 people |
| locked false clear | 2 of 10 | cleared locked meals | 8 of the 10 clears came from one person |
| false caution (caution, no spike) | 175 of 306; 78 of 139 | development; locked cautions | the price of a one-point band and a 33–42% base rate |
| spike among can't judge | 19 of 66; 2 of 4 | development; locked | abstention catches spikes near the base rate |
| T6 confident wrong carb bin (R7) | 1 dish | 14 real phone photos | visually familiar, nutritionally unfamiliar |
| T7 label-definition surprise | 1 probe | development counterfactual (probe B) | a rising trace lowers predicted risk |

*Counts from outputs/folder_run, outputs/stress_run, outputs/report_metrics/report_metrics.json (errors, false_clears_exploratory), outputs/ood/ and outputs/probes/.*

T4 is the price of any hard threshold. T5 is the genuine blind spot: two of the four were cold starts
with one earlier labelled meal, two had 10 and 20. Exploratory, chosen after the outcome, and computed on
the frozen fusion rather than the fold models that answered: the two cold starts had the 2nd and 3rd
steepest pre-meal falls of the 109 cleared meals. A steep fall leaves a high two-hour mean and spread,
which the fusion reads as lower risk (window mean −0.30 and −0.21 logit, spread −0.35 and −0.59),
outweighing the fall's own upward term (+0.10, +0.23). The label's baseline is the last 30 minutes,
already lower after the fall, so the meal had room to spike: probe B's mechanism in reverse. A "glucose
falling now" check would be the action (not shipped).

Probe B was registered as an expected pass (a rising pre-meal trace raises risk on at least 90% of
meals) and failed: risk rose on only 8%. The label is baseline-relative, so a rising trace lowers the
predicted spike probability; the registration encoded our naive physiology, and the model learned the
label (docs/probes_s13.md). Exact SHAP for the frozen fusion (Figure 6; terms verified to sum to each
meal's logit; Platt and the intercept act after it and are not attributed) ranks prior rise first, then
age, heart rate and prior spike rate. Where attribution and ablation disagree, for heart rate and for
the four photo terms (together 0.33 logit per meal against an unestablished gain), attribution measures
use and ablation measures loss: a refit recovers most of an input's signal from correlated ones (heart
rate correlates with five inputs at |r| up to 0.22). A higher pre-meal heart rate lowers predicted risk
(−0.21 per standard deviation), plausibly recent activity; untested.

![Figure 6: SHAP for the frozen fusion](../outputs/figures/f6_shap.png)

Probe A: risk moves in the photo's direction on 100% of meals, yet the extreme bottom↔top-decile plate
swap leaves 68.5% of decisions unchanged (median swing 0.102), as pre-committed: the gate's verdict seen
mechanistically. R7 (rule ledger): the kNN check measures visual, not nutritional, familiarity. A local
lamb-and-rice dish scores 0.318 against the 0.614 cut (more familiar than the clean tuna bowl at 0.406)
and gets a confident wrong carb bin; the sweep under a ≤2% false-flag budget shows no threshold fixes it,
and the finding survived the final model's regenerated reference. Mitigations: the confirm-carbs tier
and per-answer distance monitoring; a nutrition-aware embedding is future work.

## 6. Reproducibility
The 8 locked subjects were scored once, on 25 Nov, from the frozen artifacts (locked rows of the §3
tables), through the same Pipeline.answer path the instructor command invokes. Decisions: 10 low risk,
139 caution, 4 can't judge; missed-among-cleared 2 of 10; cost reduction −5.7% against always caution.
The development estimate (0.762) lies above the locked interval: §3's selection pressure and the harder
cohort (42.5% vs 32.8% spike rate) both act here, inseparable at n=8. The intercept still beats raw on
the locked people at the point estimate on AUROC, Brier and log-loss, with paired intervals that include
zero. As pre-committed, the result is reported as measured, with no re-tuning after.

After the locked run, a fresh clone on the Mac under the final serving code (install, data, pinned
weights, prepare, selftest, evaluate, figures, both input suites) reproduced every number in the report
of that day exactly. `just evaluate` reruns the development evaluation, then `just report-numbers`
recomputes the descriptive metrics and the replay and redraws the figures from committed outputs (no
dataset, no weights, about a minute). The compute studies have their own recipes; `just reproduce-all`
chains them in order, and the README maps every table to its recipe and file. A test checks the report's
table cells against those files on every commit.

**AI-assisted development, verified.** Code and documents were built with an LLM assistant, as the
course expects, and the commits say so; the author set the goals and approved each pre-registration,
freeze and verdict before its run. The defence is the verification trail: a probe write-up stated n=397
where the output says 460, caught by the number-to-file check; the freeze procedure stopped its own first
run over a 0.001 near-miss. After the tag `submission-v1` (25 Nov), three cold grades by AI agents given
the rubric led to a dated round, run overnight under the author's standing instruction and listed in
docs/changelog.md; no frozen artifact or locked number changed.

## 7. Limitations and conclusion
CGMacros is a de-identified public PhysioNet dataset collected under the source study's ethics approval;
no new human data. Forty-five people from one study, with repeated standardised meals that D8 suggests
inflate any photo-memorisation effect; no local-cuisine data, and R7 names the dish class that would
suffer. T2D participants are training-only, so no claim is made for them. Heart rate adds nothing here
and would be dropped in a next version, and the folder command should accept a person's history. The
photo is not established as a risk signal, the intercept's AUROC gain is not established either, the
registered ECE target is narrowly missed, and the label's baseline-relative definition can diverge from a
user's intuition (probe B). This is an advisory prototype with no clinical validation, end-user test or
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
| not used, reasons in the text: attention, full fine-tuning, ensembles, self-supervised pretraining | §2 |

**References.** Zeevi et al., Cell 2015, doi:10.1016/j.cell.2015.11.001 · Berry et al., Nature
Medicine 2020, doi:10.1038/s41591-020-0934-0 · Vickers & Elkin, Med Decis Making 2006,
doi:10.1177/0272989X06295361 · Brazeau et al., Diabetes Res Clin Pract 2013 (carb counting, T1D),
doi:10.1016/j.diabres.2012.10.024 · Vasiloglou et al., Nutrients 2018 (GoCARB),
doi:10.3390/nu10060741 · Kompa, Snoek & Beam, npj Digital Medicine 2021,
doi:10.1038/s41746-020-00367-3 · CGMacros: Scientific Data 2025, doi:10.1038/s41597-025-05851-7,
and PhysioNet v1.0.0.
