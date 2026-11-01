# GLIMPSE final submission — design document

Written 1 November 2026, **before** any new result, as the gate for the final stage. Deadline 27 November 2026.
Nothing is built that does not fill a row in the evidence table below.

## 0. Hard constraint

The course total must not fall below 91. Points already decided: proposal 9.1/10, mini-project 21.0/25,
intermediate 18.0/20 = 48.1 of 55. The remaining 45 points (presentation 10, final deliverable 20, viva 15)
must therefore average **95.3%**, against a ceiling of 93.1 total. One day late costs 5 points and ends it.

Consequences adopted here: a step is not done until it would survive a cold grading at 95%; the deliverable,
the presentation and the viva are planned together, not sequentially; every freeze date is absolute.

## 1. Pre-registered decisions (dated; no re-anchoring)

**D1 — Partition v2 (post-intermediate redesign, declared before new results).** Development = 23 target people
(5 outer folds) + 14 type 2 diabetes people (training only). The 4 inner folds inside each outer training set are
**the validation split** for every selection: regularisation C, epochs, Platt parameters, shrinkage m, cut-offs.
Test = the 8 locked people, scored once at the end. The intermediate's fixed 4-person validation and 5-person test
groups dissolve into development; the intermediate scored those 5 once and that result stands as reported.
Reason: the intermediate's three overlapping roles produced the double count the professor flagged.

**D2 — One scoring path.** `Pipeline.answer(meal, history) -> Answer` is used by `evaluate`, `demo`, `predict`,
`predict-folder` and the app. Evaluation builds one Pipeline per outer fold and calls it on every held-out meal;
usefulness, abstention and cost come from `Answer.decision`. A missing photo is *can't judge* everywhere; the
intermediate's average-photo fill is deleted. A test asserts demo and evaluate agree meal by meal.

**D3 — Fix 1, per-person intercept.** Within each outer fold: fit on fold-training people → inner-fold out-of-fold
risk → Platt recalibration → choose shrinkage m and cut-offs by minimum expected cost on that recalibrated risk.
For each held-out person, their first 5 chronological meals set a shrunk intercept, are answered *caution*
(cold start) and are **excluded from scoring**; the rest are scored. Held-out labels never touch a choice.
Cold-start share is reported and must stay under 20%.

**D4 — Fix 2, operating thresholds.** Cut-offs chosen on the recalibrated inner-fold risk by minimum expected cost,
stored in config, with a sensitivity sweep over the pre-registered cost settings.

**D5 — Tiered escalation.** Tier 0 deterministic input checks → tier 1 model answer → tier 2 **confirm-carbs**
(unfamiliar or unreadable photo: the user supplies a carb estimate and the risk is still produced) → tier 3 human
(dietitian queue). Tier 2 is the proposal's fallback, promoted because the kNN check flags unfamiliar food.

**D6 — Scratch model.** SensorCNN stays as the required from-scratch model and the RQ2 answer only. One cheap
4-hour-window / 3-seed check, one row, and it is not in the deployed Pipeline.

**D7 — Both outcomes pre-written.** The main claim is the photo's AUROC gain. Before the seeds run, both the
"shown" and the "not shown" paragraphs are written. If the lower bound stays at or below zero on 31 people, the
product story becomes sensor + confirm-carbs, stated plainly, and the report says so rather than softening.

**Cut now, not later:** Nutrition5k pre-training, conformal intervals, the carb-sensitivity slider, the CNN+GBM
ensemble, DINOv2 (the kNN check is tuned instead), a full fine-tuning arm. Each gets one row in the proposal
status table with the reason.

## 2. Target code map

Current: 5,172 lines over 29 modules; `run_evaluation.py` 583, `training/photo.py` 448, `demo_run.py` 237,
`data/legacy.py` 136. The professor could not navigate it. Target: one job per module, ≤ 250 lines, named by
pipeline stage, each docstring naming the report table or figure it produces.

```
src/glimpse/
  data/        download meals windows splits photos quality normalise dataset preprocess
  models/      sensor_cnn photo_model fusion budget weights
  training/    train_scratch train_photo train_final common
  cv/          folds cv_sensor cv_photo cv_fusion
  calibration/ recalibrate thresholds          # fix 1 and fix 2, nothing else
  evaluation/  metrics tables figures locked_set
  studies/     ablations robustness ood explain efficiency failures
  serve/       pipeline predict demo app feedback monitoring
  cli.py config.py schemas.py
```

Deleted: `data/legacy.py`, `demo_run.py`, any module with no call site. Archived to the git tag `proposal-era`:
`pilot/`, `scripts/`, `results/`. No module imports from a later stage. CI fails on dead code.

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

New recipes to build: `predict-folder`, `ablate`, `robust`, `ood`, `probe`, `monitor`, `efficiency`, `failures`,
`manifest`, `wordcount`. `just evaluate` becomes the umbrella that regenerates every table and figure.

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
