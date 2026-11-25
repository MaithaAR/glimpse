# GLIMPSE: will this meal spike my glucose?

A before-meal photo, the last 2 hours of CGM (continuous glucose monitor) and heart rate, and a short profile give
one of three answers: **low risk**, **caution**, or **can't judge** (always with the reason). The answer comes in
about 0.3 s on a laptop CPU and runs offline on local weights. It is a research prototype for adults with
prediabetes: a second look before eating, not a medical device. MAAI7103 course project (Maitha Alhosani).

**Final submission (report: [`docs/final_report.md`](docs/final_report.md)):**

- the full data pipeline for all four inputs (photo, CGM, heart rate, profile);
- both required models trained (a from-scratch 1-D CNN, and a fine-tuned ConvNeXt-Tiny);
- a person-level evaluation with baselines, calibration and targets;
- a working command-line demo, a local screen and a Linux container.

The final report is [`docs/final_report.md`](docs/final_report.md); the intermediate milestone report is kept at [`docs/intermediate_report.md`](docs/intermediate_report.md).

## Architecture
```
raw CGM/HR readings ─┐                                  (rules run before any model)
before-meal photo ───┼─> checks: CGM complete & in range? photo readable, not blank, familiar? ──no──> CAN'T JUDGE + reason
profile ─────────────┘          │ yes
                                ├─ 2-h window (24 × 5 min, same code as training) + profile ─┐
                                └─ ConvNeXt-Tiny photo model → carb range (4 bins) + grams ──┴─> logistic-regression
                                                                                                fusion → spike risk
                    risk < 0.15 → LOW RISK · ≥ 0.16 → CAUTION · in between → CAN'T JUDGE (borderline)
```
The from-scratch SensorCNN (5k parameters) is trained and evaluated as the required scratch model. It ties
hand-made trend features, so the deployed fusion uses those features (see the report).

## Quick start (macOS or Linux; needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/), no pip)
```bash
just install     # Python 3.14 + locked libraries, CGMacros download (PhysioNet), pinned photo weights, dataset + splits
just check       # ruff + 143 tests
just demo        # worked cases, two can't-judge cases, response time  -> outputs/demo/cases.md
just evaluate    # all dev tables + the six report figures             -> outputs/evaluation_v2/, outputs/figures/
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
| `just train-scratch` · `just train-photo` · `just train-final` | SensorCNN · photo model · the intermediate 28-person fusion (`just train` runs all three; the DEPLOYED artifacts come from `just finalize`, the pre-registered regeneration) | 1 · 15 · 8 min |
| `just cv` · `just photo-cv` · `just fusion` | 5-fold person cross-validation of all models (resumable) | 2 · 90 · 2 min |
| `just evaluate` | every dev table and figure in the report; the locked-8 table is committed once by design (`just evaluate-locked` refuses a rerun); budget/levels/latency have `just params` / `robustness-levels` / `efficiency` | 1 min |
| `just photo-seeds` | seed spread of the final photo model on the validation people (deployed checkpoint untouched) -> `outputs/training/photo_convnext_seeds.json` | 10 min |
| `just recalibrate` | superseded by partition v2: the deployed calibration comes from `just finalize`; the recipe says so and exits | – |

Cross-validated predictions are committed in `outputs/`, so `just evaluate` works straight after `just install`.

## Results (95% ranges clustered by person)
| Risk model | development (23 people, 481 meals) | locked test (8 people, 153 meals) |
|---|---|---|
| raw fusion | 0.726 (0.659–0.797) | 0.611 (0.510–0.743) |
| **+ per-person calibration (shipped)** | **0.762 (0.682–0.829)** | **0.645 (0.540–0.736)** |

**What the results show (details and evidence in [`docs/final_report.md`](docs/final_report.md)):**

- Per-person calibration improves every development fold (5/5) and the locked test; dev ECE 0.059 raw → 0.049 shipped.
- The photo is NOT established as a risk signal: pre-registered gate +0.025 (−0.010, +0.061). It ships as a carb-entry assist.
- Usefulness targets hold at the point on development (23% cleared, 14% cost saving, 7.3% missed vs a 10% cap) and FAIL on
  the locked test (10 of 153 cleared, 2 spiked, cost reduction −5.7%) — reported unadjusted, per the pre-registration.
- The 8 locked people were scored exactly once, on 25 Nov, from the frozen artifacts (`just evaluate-locked` refuses a rerun).
- Full tables: `outputs/evaluation_v2/`, `outputs/locked_eval/` and `just evaluate`.

## Repository
```
configs/default.yaml   every setting (validated by Pydantic at start-up)
src/glimpse/           data/ · models/ · training/ (incl. finalize.py, the pre-registered artifact regeneration) ·
                       evaluation/ (run.py, locked.py, metrics.py) · serve/ (pipeline, rules, io, folder — the one
                       scoring path) · studies/ (robustness, promised, ood, probes, monitoring, efficiency) ·
                       figures.py · demo_cases.py · app.py · feedback.py · cli.py (`glimpse` command)
checkpoints/           trained models, frozen cut-offs (hashed), unfamiliar-photo reference
outputs/               evaluation tables and figures, learning curves, cross-validated predictions, demo cases
docs/                  intermediate report, data card, model card, evaluation protocol (decisions fixed in advance)
samples/               instructor suite (13 meals), stress suite (23 hostile meals), example CSV
```

## Promised in the proposal vs delivered (as of the intermediate; kept for the record — the final reconciliation of every commitment is in docs/evaluation_protocol.md)
| Proposal | Now | Why |
|---|---|---|
| EfficientNet-B0 photo model | ConvNeXt-Tiny + carb-bin head | instructor feedback |
| Fusion MLP with quantiles + monotone carbs | logistic regression | feedback ("overkill"), small data |
| DINOv2 unfamiliar-photo check | nearest-neighbour check in the photo model's features | no extra model; DINOv2 comparison planned for the final |
| Calibration + cost-based cut-offs | cut-offs from the costs (frozen before the test); recalibration planned | test calibration shows over-prediction |
| Conformal rise intervals, W&B, Emirati photo set, user test | final submission | not required at this stage |
| Gradio app in Docker | `just app` (Gradio) + Dockerfile (CLI demo and self-test) | done |

Changes made after seeing results (C grid, ridge arm, carb-bin check, cut-off amendment) are listed and
labelled in [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md).

## Repository map

| Module | Job | Recipe | Output |
|---|---|---|---|
| `data/` (`meal_table.py` builds the labelled meals) | download, build, validate, split | `just install`, `just prepare` | `data/`, `outputs/data/` |
| `models/` | SensorCNN, ConvNeXt-Tiny heads, budget | `just params` | `outputs/param_budget.csv` |
| `training/photo.py` | shared photo-model library (caches, losses, loops) | – | used by the three below |
| `training/train_photo.py`, `scratch.py`, `final.py` | the training runs | `just train-*`, `just photo-seeds` | `checkpoints/`, `outputs/training/` |
| `training/cv.py`, `photo_cv.py`, `fusion.py` | person-level cross-validation | `just cv`, `just photo-cv`, `just fusion` | `outputs/cv,photo_cv,fusion/` |
| `evaluation/` (run.py, locked.py, metrics.py) | dev evaluation, the one-shot locked run, statistics | `just evaluate`, `just evaluate-locked` | `outputs/evaluation_v2/`, `outputs/locked_eval/` |
| `serve/` + `demo_cases.py`, `app.py`, `feedback.py` | the one scoring path, folder runner, worked cases, UI, feedback | `just demo`, `just predict-folder`, `just app` | `outputs/demo,folder_run,feedback/` |
| `studies/` + `figures.py` | robustness (+levels), promised comparisons, OOD, probes, monitoring, efficiency, report figures | `just robustness`, `just promised`, `just ood`, `just probes`, `just monitoring`, `just efficiency`, `just figures` | `outputs/…`, `outputs/figures/` |

Proposal-era pilot analyses (`pilot/`, `scripts/`, `results/`) were removed from the tree and live at the git tag
`proposal-era`; copies sit in the course proposal folder.

## Data and licence
CGMacros v1.0.0 (PhysioNet, CC BY-NC-SA 4.0) is downloaded by `just install` and never committed.
Pretrained weights: `timm/convnext_tiny.fb_in22k_ft_in1k`, pinned to a Hugging Face commit and checked by SHA-256.
Seeds: `seed: 20260927` in the config fixes Python, NumPy and PyTorch.
Note: Gradio requires Pillow < 13, so the lock uses Pillow 12.3; all results were checked unchanged.
