# GLIMPSE — Glycemic-response Inference from Meal Photos and Sensor Evidence

MAAI7103 Deep Learning course project (solo: Maitha Alhosani). Before a meal, GLIMPSE estimates the risk of a large glucose spike from a meal photo, the pre-meal CGM/heart-rate trace and a short profile, and abstains with a reason when it cannot judge.

## Quick start (Python 3.14, uv, just)
```bash
just install    # uv sync + pretrained weights + CGMacros CSVs and photos (resumable, ~350 MB)
just demo       # two synthetic sample requests: one answered, one abstained (runs offline)
just test       # unit and end-to-end tests
just lint       # ruff
```
Other recipes: `just train` (`train-scratch`, `train-finetune`), `just evaluate`, `just clean`. Run `just` to list them.
The CPU is the reference device; on Linux `uv` installs CPU-only torch.

## Layout
```
configs/default.yaml       every rule and hyperparameter (validated at startup by glimpse.config)
src/glimpse/
  config.py                Pydantic config model (unknown keys / invalid values fail fast)
  schemas.py               Pydantic contracts: PreMealInput, MealSample, RiskCard, SpikeEval, PhotoEval, DecisionEval, TargetStatus
  cli.py                   `glimpse` command used by every Justfile recipe
  data/                    download (CGMacros, weights), meal table builder, shared preprocessing
  pipeline/                deterministic checks -> model -> thresholded decision / abstention
  models/                  sensor CNN (from scratch) and fine-tuned photo model
  evaluation/              metrics, person bootstrap, calibration, cost model
  utils/                   Loguru setup, seeding
samples/                   synthetic demo inputs (no participant data)
splits/                    locked final-test people (SHA-256 checked) and the photo list
tests/                     pytest suite
pilot/, results/           proposal-stage pilot scripts and logs (archived evidence, Python 3.11)
proposal/                  submitted proposal
```

## Data
CGMacros (PhysioNet, open access, CC BY-NC-SA 4.0): 45 adults, ~10 days, Dexcom/Libre CGM, Fitbit heart rate, meal photos, macros. `just install` downloads only the CSVs and the 1,486 before-meal photos listed in `splits/photo_list.txt`. No participant data is committed.

## Rules that must never be broken
1. The 8 locked people (`splits/`) are never used for training, tuning, calibration or thresholds.
2. Every learned stage is fit on training people only; preprocessing statistics come from training data only.
3. Model inputs are only information available before eating: before-photo, CGM and heart rate up to the meal, profile.
4. Seeds are set from `configs/default.yaml`; runs are logged to `logs/` and results written to `outputs/`.
