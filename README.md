# GLIMPSE: will this meal spike my glucose?

A before-meal photo, the last 2 hours of CGM (continuous glucose monitor) and heart rate, and a short profile give
one of three answers: **low risk**, **caution**, or **can't judge** (always with the reason). The answer comes in
about 0.3 s on a laptop CPU and runs offline on local weights. It is a research prototype for adults with
prediabetes: a second look before eating, not a medical device. MAAI7103 course project (Maitha Alhosani).

**Intermediate milestone:**

- the full data pipeline for all four inputs (photo, CGM, heart rate, profile);
- both required models trained (a from-scratch 1-D CNN, and a fine-tuned ConvNeXt-Tiny);
- a person-level evaluation with baselines, calibration and targets;
- a working command-line demo, a local screen and a Linux container.

The report is [`docs/intermediate_report.md`](docs/intermediate_report.md).

## Architecture
```
raw CGM/HR readings ─┐                                  (rules run before any model)
before-meal photo ───┼─> checks: CGM complete & in range? photo readable, not blank, familiar? ──no──> CAN'T JUDGE + reason
profile ─────────────┘          │ yes
                                ├─ 2-h window (24 × 5 min, same code as training) + profile ─┐
                                └─ ConvNeXt-Tiny photo model → carb range (4 bins) + grams ──┴─> logistic-regression
                                                                                                fusion → spike risk
                    risk < 0.143 → LOW RISK · ≥ 0.20 → CAUTION · in between → CAN'T JUDGE (borderline)
```
The from-scratch SensorCNN (5k parameters) is trained and evaluated as the required scratch model. It ties
hand-made trend features, so the deployed fusion uses those features (see the report).

## Quick start (macOS or Linux; needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/), no pip)
```bash
just install     # Python 3.14 + locked libraries, CGMacros download (PhysioNet), pinned photo weights, dataset + splits
just check       # ruff + 80+ tests
just demo        # worked cases, two can't-judge cases, response time  -> outputs/demo/cases.md
just evaluate    # all metrics, baselines, calibration, fixed test     -> outputs/evaluation/ (about 1 min)
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
| `just train-scratch` · `just train-photo` · `just train-final` | SensorCNN · photo model · 28-person fusion + frozen cut-offs (`just train` runs all three) | 1 · 15 · 8 min |
| `just cv` · `just photo-cv` · `just fusion` | 5-fold person cross-validation of all models (resumable) | 2 · 90 · 2 min |
| `just evaluate` | tables, figures, targets; scores the fixed test (refuses if the frozen cut-offs were changed) | 1 min |
| `just recalibrate` | validation-chosen cut-offs on recalibrated risk, checked on CV without the test people (no test labels) -> `outputs/evaluation/recalibration.md` | 1 min |

Cross-validated predictions are committed in `outputs/`, so `just evaluate` works straight after `just install`.

## Results (5-fold person cross-validation, 23 people, 481 meals; 95% ranges by person)
| Model | AUROC |
|---|---|
| Profile only | 0.672 (0.567–0.764) |
| Profile + CGM trend | 0.686 (0.604–0.772) |
| Profile + SensorCNN (from scratch) | 0.689 (0.603–0.773) |
| **+ photo (main)** | **0.724 (0.659–0.796)** |
| + logged carbs (reference) | 0.717 (0.648–0.796) |

**What the results show:**

- The photo adds +0.037 (+0.003 to +0.073), and it matches logged carbs (non-inferior at 0.03).
- Calibration: ECE 0.052 on the cross-validation, 0.129 on the fixed test (over-predicts).
- Usefulness targets are not met yet: 9% cost saving, 14% of meals cleared.
- Full tables are in `outputs/evaluation/results.md`.

## Repository
```
configs/default.yaml   every setting (validated by Pydantic at start-up)
src/glimpse/           data/ (download, meals, windows, splits, photos, quality) · models/ · training/ (scratch, photo,
                       cv, photo_cv, fusion, final) · evaluation.py + run_evaluation.py · demo.py (app logic) · app.py
                       (screen) · feedback.py (mocked feedback) · cli.py (`glimpse` command)
checkpoints/           trained models, frozen cut-offs (hashed), unfamiliar-photo reference
outputs/               evaluation tables and figures, learning curves, cross-validated predictions, demo cases
docs/                  intermediate report, data card, model card, evaluation protocol (decisions fixed in advance)
samples/               synthetic CGM CSV for `just predict`
pilot/ scripts/ results/  proposal-pilot code and logs, frozen (GitHub only; left out of the submitted zip)
```

## Promised in the proposal vs delivered (intermediate)
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

## Data and licence
CGMacros v1.0.0 (PhysioNet, CC BY-NC-SA 4.0) is downloaded by `just install` and never committed.
Pretrained weights: `timm/convnext_tiny.fb_in22k_ft_in1k`, pinned to a Hugging Face commit and checked by SHA-256.
Seeds: `seed: 20260927` in the config fixes Python, NumPy and PyTorch.
Note: Gradio requires Pillow < 13, so the lock uses Pillow 12.3; all results were checked unchanged.
