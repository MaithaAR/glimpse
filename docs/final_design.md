# GLIMPSE final submission — design document

Written 1 November 2026, **before** any new result, as the gate for the final stage. Deadline 27 November 2026.
Nothing is built that does not fill a row in the evidence table below.

## 0. Standard of work

Every claim is judged by its interval, not its point estimate. A step is finished only when an independent cold
reading would accept it. Freeze dates are absolute. Where the evidence does not support the product story, the
report says so and the design changes, not the wording.

## 1. Pre-registered decisions (dated; no re-anchoring)

**D1 — Partition v2 (post-intermediate redesign, declared before new results).** Development = 23 target people
(5 outer folds) + 14 type 2 diabetes people (training only). The 4 inner folds inside each outer training set are
**the validation split** for every selection: regularisation C, epochs, Platt parameters, shrinkage m, cut-offs.
Test = the 8 locked people, scored once at the end; **this is the primary results table**, because it is the only
split no selection has seen. The 23-person outer cross-validation is reported alongside it and labelled
*development*: it is better powered but its people were already seen at the intermediate stage, so it is
supporting evidence, not the headline. The intermediate's fixed 4-person validation and 5-person test groups
dissolve into development; the intermediate scored those 5 once and that result stands as reported.

Stated plainly in the report, because it is the honest cost of this redesign: those 5 people's labels informed the
intermediate's calibration diagnosis, so development estimates are optimistic, and the locked 8 are the only clean
measurement. The 8 are not pooled with the 23 for the headline — a clean test loses its meaning the moment it is
averaged with a seen one.

**D2 — One scoring path.** `Pipeline.answer(meal, history) -> Answer` is used by `evaluate`, `demo`, `predict`,
`predict-folder` and the app. Evaluation builds one Pipeline per outer fold and calls it on every held-out meal;
usefulness, abstention and cost come from `Answer.decision`. A missing photo is *can't judge* everywhere; the
intermediate's average-photo fill is deleted. A test asserts demo and evaluate agree meal by meal.

**D3 — Fix 1, per-person intercept, expanding window.** The population risk is Platt-recalibrated on inner-fold
out-of-fold predictions. Each person then carries an offset on the recalibrated logit, fitted only on their own
*earlier* meals:

    b_p(t) = argmin_b  sum over meals i < t of  NLL(y_i, sigmoid(f_i + b))  +  (m/2) b^2,
    scored risk for meal t = sigmoid(f(x_t) + b_p(t)),  b_p(1) = 0.

Every meal is scored, in the order the user would experience it, which is exactly what deployment does. The
shrinkage m is mandatory, not cosmetic: with five identical labels the unpenalised estimate diverges. m is chosen
by replaying this same protocol inside the 4 inner folds and minimising expected cost; it never sees a held-out
label, and the shrinkage uses the population variance, not the person's own.

Rejected alternative (recorded): fitting the offset on each person's first 5 meals and excluding those meals from
scoring. It drops 115 of 481 development meals (23.9%), biases the scored set by removing the earliest and
noisiest meals, and reintroduces the gap between what is evaluated and what is deployed. The expanding window has
none of these faults.

**D3 amendments (S5 audit, 1 Nov, recorded before any run).** (i) *Label availability gate:* at
deployment the CGM labels a meal two hours after it (feedback.py, `cgm_auto`), so the history at meal t
contains exactly the earlier meals whose two-hour window closed before t — no label at or after t is ever
visible, and the replay reproduces this information set. (ii) *History needs a logit:* the NLL term requires
the meal's fusion logit, so a meal the system refused without producing one (no usable photo, failed input
checks) cannot enter the sum; a borderline refusal has a logit and does. (iii) *Wording:* the penalty
(m/2)b^2 is a Gaussian prior with precision m chosen by the inner-fold replay; the earlier sentence "the
shrinkage uses the population variance" overstated it and is corrected here. (iv) *Selection and ties:* for
each m in {0.5, 1, 2, 4, 8, 16, 32, inf} the cut-offs (lo, hi) are re-chosen by minimum expected cost on the
replayed inner-fold risks (fixing them would score each m at the wrong operating point); exact cost ties
break toward the LARGEST m — the most-shrunk model, closest to population, is the conservative default —
and the full cost-versus-m curve is reported so a flat selection is visible. m = inf is exactly the
no-intercept system, so it doubles as one D3b ablation arm. (v) *Two realisations of D4:* development
decision metrics use cut-offs selected inside each fold's inner folds only, so no reported number uses
thresholds that saw its own held-out labels; the config values are a deployment artifact frozen at the final
stage by the identical procedure on all development data.

**D3b — The intercept is a claim, so it is ablated.** Hypothesis: the population model cannot separate rare from
frequent spikers (intermediate: 30% predicted against 16% observed for rare spikers, 42% against 54% for frequent,
correlation −0.77), so a per-person offset should cut ECE within each spike-rate half without reducing AUROC.
The ablation reports the system with the offset, without it, and with a population-only recalibration.

**D4 — Fix 2, operating thresholds.** Cut-offs chosen on the recalibrated inner-fold risk by minimum expected cost,
stored in config, with a sensitivity sweep over the pre-registered cost settings.

**D5 — Tiered escalation.** Tier 0 deterministic input checks → tier 1 model answer → tier 2 **confirm-carbs**
(unfamiliar or unreadable photo: the user supplies a carb estimate and the risk is still produced) → tier 3 human
(dietitian queue). Tier 2 is the proposal's fallback, promoted because the kNN check flags unfamiliar food.

**D6 — Scratch model.** SensorCNN stays as the required from-scratch model and the RQ2 answer only. One cheap
4-hour-window / 3-seed check, one row, and it is not in the deployed Pipeline.

**D7 — The primary test, and both outcomes pre-written.** The primary endpoint stays the one the proposal
pre-registered: the photo arm's AUROC gain over profile + trend, with a person-clustered bootstrap interval, now
reported on the locked 8 as the test and on development as supporting evidence.

A secondary endpoint is **added at this stage and labelled as such**, with its reason recorded before any result:
the paired difference in mean log-loss between the photo and no-meal arms, clustered by person, one-sided at 5%.
It is added because the AUROC difference is underpowered — the person-level standard error of about 0.021 on 18
people does not shrink enough at 31 to clear zero — not because it gives a friendlier answer. Both are reported
whatever they say, and the primary remains primary.

Before the seeds run, both result paragraphs are written. If the gain's lower bound stays at or below zero, the
product story becomes sensor plus confirm-carbs, and the report states that the photo's contribution is not
established rather than softening the sentence.

**D8 — Recurring meal profiles.** 62% of isolated target meals share an exact macro profile with another person's
meal, which the proposal flagged and the intermediate never tested. The profile-excluded split (meals sharing a
macro profile with a held-out person are removed from that fold's training set) is run as the second half of the
week-4 gate, and its result is reported next to the main gain whether or not it survives.

**Cut now, not later:** Nutrition5k pre-training, conformal intervals, the carb-sensitivity slider, the CNN+GBM
ensemble, DINOv2 (the kNN check is tuned instead), a full fine-tuning arm. Each gets one row in the proposal
status table with the reason.

## 2. Target code map

Current: 5,172 lines over 29 modules; `run_evaluation.py` 583, `training/photo.py` 448, `demo_run.py` 237,
`data/legacy.py` 136. The professor could not navigate it, and named those files.

A full re-architecture would cost 10–15 hours and earn part of one rubric bullet. The change is therefore
confined to what was actually criticised: split the three named files, delete the two dead ones, add the two
packages the new work needs, and give the reader a map. Existing `data/`, `models/` and `training/` stay.

```
src/glimpse/
  data/        unchanged (legacy.py deleted)
  models/      unchanged
  training/    train_scratch, train_photo, train_final, common   # photo.py split, cv part moved out
  cv/          cv_sensor, cv_photo, cv_fusion, folds
  evaluation/  metrics, tables, figures, locked_set              # run_evaluation.py split
  studies/     ablations, robustness, ood, explain, efficiency, failures   # new
  serve/       pipeline, predict, demo, app, feedback, monitoring # demo_run.py merged in
  calibration.py  cli.py  config.py  schemas.py
```

Correction (S2 audit, 1 Nov): `legacy.py` and `demo_run.py` turned out to be live, only misnamed — the first is
the meal-table builder behind `just prepare`, the second builds the demo cases and selftest. They were renamed
(`data/meal_table.py`, `demo_cases.py`), not deleted; nothing in the tree currently has zero call sites. The
`run_evaluation.py` split moves to right after S3, which decides what survives in it. Archived to the git
tag `proposal-era`: `pilot/`, `scripts/`, `results/`. The README carries a one-line map of every module and the
recipe that runs it; each module's docstring names the report table or figure it produces. Budget: 4 hours.

## 3. Evidence table

Every rubric bullet, the recipe that produces its evidence, the file it lands in, and the report section that
cites it. A step is closed only when its row points at a real file.

| Rubric | Evidence | Recipe | Output | Report |
|---|---|---|---|---|
| 1a uv, caps, lock | install from clean checkout | `just install` | `uv.lock`, CI log | §2 |
| 1b recipes incl. folder command | all recipes run unattended | `just --list` | `README.md` map | §2 |
| 1c Pydantic + Loguru | Answer, FeedbackRecord, monitor alerts | `just test` | `tests/`, `logs/glimpse.log` | §2 |
| 2a product, workflow, scope | user journey incl. escalation | – | report | §1 |
| 2b budget, scratch + fine-tuned | counted in code | `just params` | `outputs/param_budget.csv` | §2 |
| 2c three modalities, 3–5 capabilities | modality-removal ablation | `just ablate` | `outputs/ablate/modality.csv` | §2, §4 |
| 2d rules traceable to observed failures | rules ledger | `just predict-folder` | `docs/rule_ledger.md` | §2, §7 |
| 2e demo + feedback | CLI and Gradio, abstention case | `just demo`, `just app` | `outputs/demo/cases.md` | §2 |
| 3a data, splits, leakage, ledger | reconciliation, duplicate check | `just prepare` | `outputs/data/*`, `docs/data_card.md` | §3 |
| 3b metrics, baselines, CIs, targets | baseline ladder, person bootstrap | `just evaluate` | `outputs/evaluation/results.md` | §4 |
| 3c ablations with hypotheses | 7 ablations | `just ablate` | `outputs/ablate/` | §4 |
| 3d training choices | LR, augmentation, freezing, epochs | `just train-*` + logs | `experiments/`, W&B export | §4 |
| 3e efficiency + generalisation gap | per-stage latency, Pareto, hard set | `just efficiency` | `outputs/efficiency/` | §6 |
| 4a calibration before/after | reliability, ECE, per-bin counts | `just evaluate` | `outputs/evaluation/fig2,3` | §5 |
| 4b cost model, thresholds, sensitivity | min-expected-cost on validation | `just recalibrate` | `checkpoints/thresholds_v2.json` | §5 |
| 4c OOD, abstention, tiers, coverage-risk | ID/OOD sets, tier rules | `just ood` | `outputs/ood/` | §5 |
| 4d robustness across levels | blur, lighting, HR dropout, gaps | `just robust` | `outputs/robust/curves.csv` | §5 |
| 4e monitoring detects a shift | stream with injected shift | `just monitor` | `outputs/monitor/` | §5 |
| 5a probes with hypotheses | swaps, flat vs rising, branch mask | `just probe` | `outputs/probe/` | §7 |
| 5b failure taxonomy with counts | categories, root causes, actions | `just failures` | `outputs/failures/taxonomy.csv` | §7 |
| 5c insights, limits, harm | – | – | report | §8 |
| 6a reproducibility | every number regenerated | `just evaluate` | all of `outputs/` | all |
| 6b development evidence | incremental commits, tracker exports | git, W&B | `experiments/` | appendix |
| 6c instructor readiness | folder of unseen inputs | `just predict-folder` | `results.jsonl`, `summary.md` | §2 |
| 6d README, lint, docstrings, tests | – | `just check` | CI log | – |
| 2b fine-tuning items vs the 10,000 cap | ledger incl. anything planned | `just prepare` | `outputs/data/finetune_ledger.csv` | §2 |
| 2e feedback → retraining | how records would retrain, simulated | `just app` | `outputs/feedback/feedback.jsonl` | §2 |
| 4b test cost vs naive policies | always-caution, always-clear, baseline | `just evaluate` | `outputs/evaluation/results.md` | §5 |
| 2c heart rate earns its place | modality-removal arm for HR alone | `just ablate` | `outputs/ablate/modality.csv` | §2, §4 |
| 3e generalisation gap | hard set: profile-excluded, stressed, unfamiliar | `just efficiency` | `outputs/efficiency/gap.csv` | §6 |
| 5c user and dietitian sessions | 3–5 users, 2 dietitians, structured notes | – | `docs/user_test.md` | §8 |

New recipes, in build order (each depends on the one before it):

`wordcount` and `manifest` (0.5 h each, they enforce the freezes) → `serve/pipeline` with the demo-equals-evaluate
test → `predict-folder` (5 h; it is what produces the observed failures the rules ledger needs) → `failures` (2 h)
→ `ablate` (4 h) → `ood` (3 h) → `probe` (3.5 h) → `robust` (5 h) → `monitor` (2.5 h) → `efficiency` (2.5 h).
`just evaluate` becomes the umbrella last, once the output paths have stopped moving. About 29 hours in total,
most of it reusing the cross-validated predictions and photo feature caches that already exist; only the
modality-removal arm and the blur and lighting arms need fresh photo passes, and those run overnight.

Three tasks that will quietly consume an evening each, and the guard against it: HEIC and EXIF handling
(pin the decoder, convert to JPEG at ingest, one fixture per format); the demo-equals-evaluate assertion
(write it on day one against three meals, not on day twenty against seven hundred); and the twelve-row behaviour
table (one parametrised test over `samples/instructor/`, written as each guard is added, never afterwards).

## 4. Report budget (≤ 3,000 words, 5–8 pages; tables and figures excluded)

| § | Section | Words | Figure |
|---|---|---|---|
| 1 | Venture and user | 150 | user journey incl. escalation |
| 2 | System and parameter budget | 350 | system diagram, learned vs rule boundary |
| 3 | Data, splits, reconciliation | 300 | count reconciliation |
| 4 | Results: baselines, ablations, training choices | 550 | paired-difference forest plot |
| 5 | Trust: calibration, cost, OOD, robustness, monitoring | 650 | 2×2 panel |
| 6 | Efficiency and generalisation | 250 | quality-vs-cost Pareto |
| 7 | Probes and failure analysis | 450 | taxonomy with counts |
| 8 | Insights, limitations, risks | 300 | targets scorecard |

`just wordcount` enforces the limit in CI.

## 5. Rules ledger (template; filled as failures are observed)

| Rule | Observed failure (meal id / log line / date) | Commit | Meals it fires on | Δcost without it |
|---|---|---|---|---|

New rules are written only after running `just predict-folder` on our own instructor-style set, so the tracing
is honest rather than retrospective. Rules that predate an observed failure are marked *pre-emptive*.

## 6. Instructor input specification

```
inputs/<meal_id>/
  photo.jpg | png | heic          optional; absent → can't judge
  cgm.csv                         timestamp, glucose_mgdl[, heart_rate]
  profile.json                    age, bmi, a1c, fasting_glucose, meal_time
```

`just predict-folder IN OUT` → `OUT/results.jsonl` (one validated `Answer` per meal, failures included:
decision, risk, carb_bin, reason, checks_passed, checks_failed, flags, latency_ms, model_versions) and
`OUT/summary.md` (counts per decision, abstention reasons). Exit code 0 even when individual meals fail.

Behaviours fixed now and tested in `samples/instructor/`:

| Input | Required behaviour |
|---|---|
| glucose in mmol/L (values 4–9) | can't judge: "values look like mmol/L; multiply by 18" |
| unsorted timestamps | sort, answer, log the correction |
| timestamps after the meal time | can't judge, naming the inconsistency |
| gap > 20 min in the window | can't judge, naming the gap length |
| heart rate column missing | answer, reason notes degraded input |
| profile implausible (age 7, BMI 85) | can't judge, naming the field |
| profile fields blank | can't judge, no defaults filled |
| dark or heavily blurred photo | can't judge or caution; never low risk |
| screenshot or non-meal image | flagged unfamiliar → confirm-carbs tier |
| EXIF-rotated duplicate | identical decision to the original |
| unfamiliar local dish | confirm-carbs tier, not a confident carb bin |
| missing or zero-byte file | per-item error entry; the run continues |

## 7. Freezes

App 17 November · repository 21 November · locked set scored once on 22 November · documentation only after that.
No retuning after the 12 November gate verdict. One overnight job per night, logged the next morning.

## Amendment (5 Nov 2026, pre-registered BEFORE any run): serving-artifact regeneration, pulled forward

The deployed demo/app/folder command still serve intermediate-era artifacts (fusion and thresholds fit
on the old 28-person split). They are regenerated ONCE, now, from partition v2, by the following fixed
procedure; S16 (22 Nov) then only scores the locked 8 against these frozen artifacts and re-runs the
fresh-clone check. Nothing in this amendment re-opens the S8 gate, the evaluation protocol, or any
shipped verdict; the dev numbers in the report remain the per-fold evaluation's.

1. **Photo serving model.** CORRECTED 5 Nov, before execution (the first wording said "heads only,
   frozen backbone" — that is NOT the CV recipe and would ship a different model class than the one
   evaluated): the final model follows the photo-CV v2 recipe exactly, at the all-development level —
   heads trained on ALL non-locked people's photos (23 dev targets + 14 T2D) for the MEDIAN of the five
   per-fold epoch choices (12), then the last backbone stage fine-tuned for the median choice (2
   epochs). No new selection of any kind.
2. **Unfamiliar-photo reference.** Rebuilt from all non-locked training photos' embeddings UNDER THE
   FINAL MODEL (the fine-tuned last stage changes embeddings, so the reference must match the model
   that serves);
   the cut is re-derived by the deployed rule unchanged (leave-one-out distance quantile 0.99) and
   re-checked against the S10 criteria: <=2% in-distribution flags, 0/14 real phone photos flagged,
   screenshot and dark instructor cases still flagged. No sweep, no new rule.
3. **Fusion.** One logistic regression; C chosen by the existing recipe with the five dev folds as the
   holds; fit on all non-locked meals. Photo features for training rows come from the committed CV
   tables: dev targets use their out-of-fold estimates; T2D rows use fold 1's nested estimates (an
   arbitrary fixed choice, recorded here; T2D people are never scored).
4. **Calibration.** Platt (a, b) fit on the pooled out-of-fold fusion risks of the five-fold
   evaluation (the standard out-of-fold proxy for a final model); then (m, lo, hi) chosen by the
   committed choose_m_and_cutoffs on those replayed risks in deployment order — D4's "frozen from an
   all-development replay", executed.
5. **Freeze.** thresholds.json v2 carries lo, hi, m and the Platt pair under the existing sha-256
   guard; config prototype values updated to match; the final bundle carries the fusion and the
   calibration so the served Pipeline uses them by default.
6. **Verification before the app freeze.** just selftest, the 13-case instructor suite, the 23-case
   stress suite and the app run against the new artifacts; outputs committed as the final-model
   evidence the professor's stress testing will hit.

**Execution note (5 Nov, recorded when the first finalize run stopped itself).** The reference
criteria as first written required BOTH the screenshot and the dark instructor photos to be flagged by
the unfamiliar-photo check. The run stopped because the dark photo scored 0.613 against the new cut
0.614 (screenshot 0.771 flagged; in-dist 1.1% within budget; real photos 0/14). The committed serving
evidence shows the dark photo never reaches that check at all: the contrast rule (grey-level spread
0.0, need 8 -- model-independent, runs first) refuses it end-to-end (outputs/folder_run). The
criterion mis-encoded the layered design -- S10 had scored the dark photo by calling the kNN check in
isolation. Decision, in the open: the criterion is corrected to the end-to-end property (screenshot
flagged by the kNN check; dark photo refused by any designed check), the near-miss value is recorded
here, and nothing else changes: same rule, same quantile, no sweep.
