# GLIMPSE — Glycemic-response Inference from Meal Photos and Sensor Evidence

MAAI7103 Deep Learning course project (solo: Maitha Alhosani). Pre-meal glucose-spike risk from a before-meal photo, CGM/heart-rate time series and profile, with calibration and abstention.

## Research questions
- **RQ1:** Do photo-predicted served macros preserve the signal of reference macros reconstructed from the meal log? (non-inferiority margin 0.03 AUROC; paired person-bootstrap)
- **RQ2:** Does a learned pre-meal sensor encoder beat engineered trend features?

## Layout
```
configs/default.yaml        all prespecified rules (label, window, onboarding, costs, targets)
splits/locked_final_set.json 8 locked people, frozen 27 Sep 2026 (seed + SHA-256; verified on load)
src/glimpse/data.py         meal table under the prespecified rules (pre-meal inputs only)
src/glimpse/evaluation.py   person folds, paired bootstrap, non-inferiority, ECE, costs, usefulness, conformal
scripts/01_build_meals.py   builds the meal table + prints the proposal's accounting
scripts/02_baselines.py     GBM baselines on development people only
pilot/                      exploratory pilot scripts and logs (27 Sep 2026)
tests/                      unit tests (pytest)
```

## Quick start
```bash
python3 -m pip install -r requirements.txt
mkdir -p data && ln -s /path/to/unzipped/cgmacros data/cgmacros   # folder containing CGMacros/
python3 scripts/01_build_meals.py      # expect: usable 1526 | onboarding 453 | post-onboarding 1054 | evaluated 896 (634 target, 31 people)
python3 -m pytest -q tests
```

## Rules that must never be broken
1. The 8 locked people (splits/) are never used for training, tuning, calibration or thresholds.
2. Every learned stage (image model, fusion, calibrator, conformal, thresholds, OOD threshold) is fit on the outer fold's training people only.
3. Model inputs are only: before-photo, CGM + HR up to meal time, profile/labs/time. Never share eaten, after-photos or consumed macros.
4. Every experiment is logged to W&B and committed.
