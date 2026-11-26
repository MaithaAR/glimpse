# GLIMPSE: will this meal spike my glucose?

A before-meal photo, the last 2 hours of CGM (continuous glucose monitor) and heart rate, and a short profile give
one of three answers: **low risk**, **caution**, or **can't judge** (always with the reason). The answer comes in
under 0.3 s (265 ms median on the author's Mac, 7 Nov; 116 ms on a 2-CPU cloud machine) and runs offline on local weights. It is a research prototype for adults with
prediabetes: a second look before eating, not a medical device. MAAI7103 course project (Maitha Alhosani).

**Grader path (three commands, about 10 minutes):**

```bash
just install                                                    # Python, locked libraries, data, pinned weights
just predict-folder samples/instructor out/instructor           # 13 meals: answers and named refusals
just predict-folder samples/returning_user out/returning_user   # the three answers: caution, can't judge, low risk
```

With neither a `history.csv` of the person's earlier labelled meals nor onboarding priors in the profile, no
meal is answered "low risk": rule R13 withholds it, by design, because there is no personal basis to clear a
meal. The instructor and stress samples are in that state. `samples/returning_user/02_with_priors_and_history`
carries one.

**Final submission (report: [`docs/final_report.md`](docs/final_report.md)):**

- the full data pipeline for all four inputs (photo, CGM, heart rate, profile);
- both required models trained (a from-scratch 1-D CNN, and a fine-tuned ConvNeXt-Tiny);
- a person-level evaluation with baselines, calibration and targets;
- a working command-line demo, a local screen and a Linux container.

The final report is [`docs/final_report.md`](docs/final_report.md); the intermediate milestone report is kept at [`docs/intermediate_report.md`](docs/intermediate_report.md).

**Two products, one code path.** v1 (served, frozen 6 Nov) fuses the photo model's carb estimate with the
CGM/heart-rate trend and the profile; it is the only model scored on the 8 locked people (once; usefulness
targets not met). v2 (frozen 26 Nov) is the sensor-first product the photo gate's pre-written verdict called
for; it was built after the locked result, tested on development only, not shown non-inferior and let more
spikes through its clears, so it stays as the tested alternative. `prototype.product` in
`configs/default.yaml` switches between them; the decision is recorded in
[`docs/final_switch.md`](docs/final_switch.md).

| development, 481 meals | v1 (served) | v1 on the locked eight (scored once) | v2 (alternative) |
|---|---|---|---|
| AUROC (95% person-clustered) | 0.762 (0.682–0.829) | 0.645 (0.540–0.736) | 0.754 (0.673–0.815) |
| ECE (target ≤ 0.05) | 0.052 | 0.061 | 0.029 |
| spikes among cleared meals (cap 10%) | 7.3% | 2 of 10 | 12.7% |

## Architecture (v1, served)
```
raw CGM/HR readings ─┐                                  (rules run before any model)
profile ─────────────┼─> checks: CGM complete & in range? not low? profile valid & in scope? ──no──> CAN'T JUDGE + reason
                     │          │ yes
before-meal photo ───┼─> readable, not blank, familiar? ──no──> CAN'T JUDGE + reason (v2: risk still given)
                     │          │ yes: ConvNeXt-Tiny → grams + carb range (shown to confirm)
                     └─ 2-h window (24 × 5 min, same code as training) + profile + 4 carb inputs ─> logistic regression
                         R16: a sensor input beyond the training range (risk-lowering side) → CAN'T JUDGE
                         + per-person intercept from earlier labelled meals (history.csv), when there are any
                         risk < 0.15 → LOW RISK · ≥ 0.16 → CAUTION · in between → CAN'T JUDGE (borderline)
                         R13: no LOW RISK without history or both onboarding priors → CAN'T JUDGE + reason
```
The from-scratch SensorCNN (5k parameters) is trained and evaluated as the required scratch model. It ties
hand-made trend features, so the deployed fusion uses those features (see the report).

## Quick start (macOS or Linux; needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/), no pip)
```bash
just install     # Python 3.14 + locked libraries, CGMacros download (PhysioNet), pinned photo weights, dataset + splits
just check       # ruff + the test suite (some tests skip until the dataset and weights are downloaded)
just demo        # worked cases (in-sample), two can't-judge cases, timing -> outputs/demo/, then the UI at :7860
just evaluate    # every report table and figure (needs data and weights)        -> outputs/
just report-numbers   # the same tables and figures from committed outputs only (no data, no weights, about 5 min on a 2-CPU machine)
just predict-folder samples/instructor outputs/folder_run   # the grader's entry point: a folder of meals in, structured answers out
just app         # local screen at http://127.0.0.1:7860 (optional extra: Gradio)
just predict --cgm samples/cgm_example.csv --photo my_lunch.jpg --age 45 --bmi 28 --a1c 5.9 --fasting 100
just selftest    # offline check of every frozen part, no dataset needed
just clean       # remove generated files
```

**Linux container** (offline after the build; the data is mounted, not copied, because of its licence):

```bash
just docker-build && just docker-selftest     # GitHub CI also builds the image and runs this with --network none
just docker-demo                              # needs `just install` first, for the data
```

## Retraining from scratch (all seeded; times on an Apple-silicon CPU)
| Recipe | What | Time |
|---|---|---|
| `just prepare` | meal table, windows, person splits, training-only scaling, quality report | 1 min |
| `just train-scratch` · `just train-photo` · `just train-final` | SensorCNN · photo model · the intermediate 28-person fusion (`just train` runs all three; the DEPLOYED artifacts come from `just finalize`, the pre-registered regeneration, so train-photo and train-final refuse to overwrite them unless `GLIMPSE_OVERWRITE_FROZEN=1`) | 1 · 15 · 8 min |
| `just cv` · `just photo-cv` · `just fusion` | 5-fold person cross-validation of all models (resumable) | 2 · 90 · 2 min |
| `just evaluate` | parameter budget, both dev evaluations (v1 and v2), v2's metrics, probes and explanation, the unfamiliar-photo check, monitoring, probes A and B, then `just report-numbers` (every other table and figure re-derived from committed outputs); `just evaluate-all` also recomputes the stress, int8 and timing studies; the locked-8 run is committed once by design (`just evaluate-locked` refuses a rerun) | about 15 min on a 2-CPU machine (`evaluate-all` about 45) |
| `just photo-seeds` | seed spread of the final photo model on the validation people (deployed checkpoint untouched) -> `outputs/training/photo_convnext_seeds.json` | 10-15 min |

Cross-validated predictions are committed in `outputs/`, so `just evaluate` works straight after `just install`,
and `just report-numbers` works on a bare clone. `just reproduce-all` reruns the compute studies in dependency
order (prepare → splits-v2 → params → cv → carbs-check → photo-cv → fusion → ablate → promised → robustness →
robustness-levels → timing → probes → ood → monitoring → efficiency → stage-timing → stress-quality →
v2-input-stress → photo-probes → end-to-end-timing → photo-seeds → evaluate → selftest → instructor and
returning-user folders → stress → demo; about 3 h). The registered last-day checks (`just final-checks`,
`just robustness-v3`) run under `just evaluate-all`. It never reruns `just finalize` (the frozen serving artifacts,
regenerated once under a pre-registered procedure) or `just evaluate-locked` (the locked eight, scored once).

**What the grader's folder command does and does not do.** `just predict-folder` answers each meal folder on
its own, at the population calibration, unless the folder holds an optional `history.csv` (the person's earlier
answers: `meal_time,fusion_logit,spiked`; see samples/README.md), which feeds the personal intercept through the
evaluation's two-hour gate. Only `samples/returning_user/02_with_priors_and_history` carries one (illustrative). A profile.json without `prior_spike_rate` /
`prior_rise` gets training medians from all 37 non-locked people, type 2 diabetes included (0.455, 48 mg/dL).
With priors but no history, v1 clears 15.2% of development meals with 9.6% spiking, and 21.9% spiking on the
locked eight (report §4); with no history and no priors rule R13 never answers low risk. The three
`samples/returning_user` folders show the three answers: caution, can't judge (borderline), low risk. `history.csv` holds the logit of the product that wrote
it; rows tagged with the other product are skipped and counted in the answer's flags.

## Results (AUROC, 95% ranges clustered by person)
The locked test measured v1 (served) once and is the only clean measurement; v2 (alternative) has development numbers
only, and they carry disclosed selection pressure (fold 1 was scored at the intermediate stage; v2 was chosen
after the locked result). "v2" in `outputs/evaluation_v2/` and "partition v2" name the split, not the product:
that folder is v1; product v2 lives in `outputs/evaluation_sensor_first/`.

| Risk model | **locked test (8 people, 153 meals; 150 with a risk)** | development (23 people; v1 460, v2 481 meals) |
|---|---|---|
| v1 raw fusion | 0.611 (0.510–0.743) | 0.726 (0.659–0.797) |
| **v1 + per-person calibration (served, locked-tested)** | **0.645 (0.540–0.736)** | 0.762 (0.682–0.829) |
| v2 sensor first + per-person calibration (alternative) | never scored on the locked eight | 0.754 (0.673–0.815) |
| v1 AUPRC (base rate) | 0.547 (0.420) | 0.591 (0.328) |
| v1 ECE, registered 15 equal-width bins / equal-mass (target ≤ 0.05) | 0.061 / 0.098 | 0.052 / 0.049 (v2: 0.029 / 0.076) |

**What the results show (details and evidence in [`docs/final_report.md`](docs/final_report.md)):**

- Per-person calibration improves every development fold (5/5) and log-loss (paired interval below zero); the pooled
  AUROC gain, +0.037 (−0.004, +0.071), is not established; given the same intercept, gradient boosting on sensors
  and profile reaches 0.750. Development ECE (0.052) equals what a perfectly calibrated model shows at this size, so
  the registered 0.05 target cannot be judged at n = 460. On the locked test it improves AUROC,
  Brier and log-loss at the point estimate, with intervals that include zero.
- The photo is NOT established as a risk signal: pre-registered gate +0.025 (−0.010, +0.061). v1 still takes the photo
  estimate as 4 of its 19 inputs; v2, built on 26 Nov to the gate's pre-written verdict, uses it only as a carb estimate.
- v2 is not shown non-inferior to v1 (−0.015, −0.045 to +0.010) and lets 12.7% of its clears spike (6 of 30 for people
  with prediabetes, against 1 of 25 in v1).
- v1: usefulness targets hold at the point on development (23% cleared, 14% cost saving, 7.3% missed vs a 10% cap) and FAIL on
  the locked test (10 of 153 cleared, 2 spiked, cost reduction −5.7%), reported unadjusted, per the pre-registration.
- The 8 locked people were scored exactly once, on 25 Nov, from the frozen artifacts (`just evaluate-locked` refuses a rerun).
- Monitoring: the photo-drift rule fires on the drifted (synthetic) week of a simulated stream, but real phone
  photos sit just below its line; replayed on logged answers, the outcome rule is too noisy at 80-answer windows
  to serve as an alarm (`outputs/monitoring/replay.json`).
- Full tables: `outputs/evaluation_v2/`, `outputs/locked_eval/`, `outputs/report_metrics/` and `just report-numbers`.
- Submitted state: tag `submission-v1` (25 Nov), then a dated improvement round listed in `docs/changelog.md`.

## Repository
```
configs/default.yaml   every setting (validated by Pydantic at start-up)
src/glimpse/           data/ · models/ · training/ (incl. finalize.py, the pre-registered artifact regeneration) ·
                       evaluation/ (run.py, locked.py, metrics.py, report_metrics.py) · serve/ (pipeline, rules, io,
                       folder: the one scoring path) · studies/ (robustness, promised, ood, probes, monitoring,
                       monitoring_replay, monitoring_power, efficiency, stage_timing, stress_quality,
                       photo_probes, v2_input_stress) · evaluation/v2_metrics.py, v2_explain.py ·
                       figures.py · demo_cases.py · app.py · feedback.py · cli.py (`glimpse` command)
checkpoints/           trained models, frozen cut-offs (hashed), unfamiliar-photo reference
outputs/               evaluation tables and figures, learning curves, cross-validated predictions, demo cases
docs/                  final report, model card, data card, evaluation protocol, design doc, changelog, monitoring
samples/               instructor suite (13 meals), stress suite (25 hostile meals), returning_user/ (one meal at 3 stages:
                       new user, with priors, with priors and history), local_photos/ (14 unseen phone photos), example CSV
```

## Promised vs delivered
Every commitment from the proposal and the intermediate is reconciled, with its outcome, in
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) (with dated errata); the intermediate-era
version of this table is kept at `docs/archive/readme_promised_intermediate.md`.

## Checkpoint files: which one serves

| file | status |
|---|---|
| `photo_convnext_final_v2_e14.pt` (named in `photo_convnext_latest.txt`), `photo_ood_reference.pt` | served: the photo model and the unfamiliar-photo reference (both products) |
| `fusion_final.joblib`, `thresholds.json` | served: v1's fusion, Platt map, m and cut-offs 0.15 / 0.16 (frozen 6 Nov) |
| `sensor_first/fusion.joblib`, `sensor_first/thresholds.json` | v2, the tested alternative (frozen 26 Nov) |
| `sensor_cnn_e13_valauc0.662.pt` (named in `sensor_cnn_latest.txt`), `normaliser.json` | SensorCNN, the from-scratch baseline: evaluated, not served |
| `photo_convnext_e17_valmae26.2.pt`, `thresholds_amended.json`, `thresholds_validation_recal.json` | superseded intermediate-stage files, kept as evidence for the dated design documents; serving never loads them |

`just train` reproduces the intermediate-stage runs on the old 28-person split and refuses to overwrite the
served files (unless `GLIMPSE_OVERWRITE_FROZEN=1`); the served files came from `just finalize` on 6 Nov.

## Repository map

| Module | Job | Recipe | Output |
|---|---|---|---|
| `data/` (`meal_table.py` builds the labelled meals) | download, build, validate, split | `just install`, `just prepare` | `data/`, `outputs/data/` |
| `models/` | SensorCNN, ConvNeXt-Tiny heads, budget | `just params` | `outputs/param_budget.csv` |
| `training/photo.py` | shared photo-model library (caches, losses, loops) | – | used by the three below |
| `training/train_photo.py`, `scratch.py`, `final.py` | the training runs | `just train-*`, `just photo-seeds` | `checkpoints/`, `outputs/training/` |
| `training/cv.py`, `photo_cv.py`, `fusion.py` | person-level cross-validation | `just cv`, `just photo-cv`, `just fusion` | `outputs/cv,photo_cv,fusion/` |
| `evaluation/` (run.py, locked.py, metrics.py) | dev evaluation, the one-shot locked run, statistics | `just evaluate`, `just evaluate-locked` | `outputs/evaluation_v2/`, `outputs/locked_eval/` |
| `evaluation/report_metrics.py`, `studies/monitoring_replay.py` | AUPRC, Brier, both ECE binnings, baselines, policy costs; monitoring replay | `just report-numbers` | `outputs/report_metrics/`, `outputs/monitoring/replay.json` |
| `serve/` + `demo_cases.py`, `app.py`, `feedback.py` | the one scoring path, folder runner, worked cases, UI, feedback | `just demo`, `just predict-folder`, `just app` | `outputs/demo,folder_run,feedback/` |
| `studies/` + `figures.py` | robustness (+levels), promised comparisons, OOD, probes, monitoring, efficiency, report figures | `just robustness`, `just promised`, `just ood`, `just probes`, `just monitoring`, `just efficiency`, `just figures` | `outputs/…`, `outputs/figures/` |

## Report table → recipe → evidence file

| Report table or figure | Recipe | File |
|---|---|---|
| §1 budget, latency | `just params`, `just efficiency`, `just stage-timing` | `outputs/param_budget.csv`, `outputs/efficiency/` |
| §1 quality vs cost, §3 baselines | `just cv`; `just report-numbers` | `outputs/cv/cv_summary.json`, `outputs/report_metrics/` |
| §2 photo recipe, what the photo does | `just photo-cv`, `just fusion`, `just ablate`, `just photo-seeds`, `just promised` | `outputs/photo_cv/`, `outputs/fusion/`, `outputs/ablations/`, `outputs/training/`, `outputs/promised/` |
| §3 results table, fold cut-offs, Figures 2 and 4 | `just evaluate` (`evaluate-locked` once) | `outputs/evaluation_v2/`, `outputs/locked_eval/`, `outputs/report_metrics/` |
| §4 cost sweep, usefulness targets | `just evaluate` | `outputs/evaluation_v2/dev_summary.json`, `outputs/report_metrics/` |
| §1 end-to-end timing, int8 | `just end-to-end-timing`, `just photo-probes` | `outputs/efficiency/end_to_end.json`, `int8.json` |
| §3-§5 v2 calibration, taxonomy, SHAP, Figures 4 and 5 | `just v2-explain` (inside `just evaluate`) | `outputs/v2_explain/` |
| §3-§5 course checks, OOD AUROC, monitoring power, calibration null | `just final-checks` | `outputs/final_checks/final_checks.json` |
| §5 debugging table R1-R4, v3 candidate (R5-R7), §6 re-weighted test (R8) | `just robustness-v3` | `outputs/robustness_v3/robustness_v3.json` |
| §4 stress table | `just stress-quality`, `just v2-input-stress` | `outputs/robustness/stress_quality.json`, `v2_input_stress.json` |
| appendix stress levels, Figure A1 | `just robustness-levels`, `just timing`, `just promised` | `outputs/robustness/`, `outputs/promised/` |
| §4 monitoring | `just monitoring`, `just monitoring-replay` | `outputs/monitoring/` |
| §5 taxonomy, probes, Figure 7 | `just predict-folder`, `just stress`, `just probes`, `just ood`, `just figures` | `outputs/folder_run/`, `outputs/stress_run/`, `outputs/probes/`, `outputs/ood/`, `outputs/figures/` |

`tests/test_report_numbers.py` checks the report's headline numbers against these files on every commit.

History note: the GitHub history runs from the proposal (27 Sep) to the final commit. Proposal-era pilot
analyses (`pilot/`, `scripts/`, `results/`) were removed from the tree on 1 Nov and live at the git tag
`proposal-era`. From 2 Nov every registration commit precedes the commit with its results.

## Data and licence
CGMacros v1.0.0 (PhysioNet, CC BY-NC-SA 4.0) is downloaded by `just install` and never committed.
Pretrained weights: `timm/convnext_tiny.fb_in22k_ft_in1k`, pinned to a Hugging Face commit and checked by SHA-256.
Seeds: `seed: 20260927` in the config fixes Python, NumPy and PyTorch.
Note: Gradio requires Pillow < 13, so the lock uses Pillow 12.3; all results were checked unchanged.
