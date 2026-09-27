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
scripts/00_fetch_photos.py  downloads only the before-meal photos used (resumable)
scripts/03_photo_experiment.py  person-excluded photo pilot (RQ1): none | reference | photo macros
scripts/04_usefulness_pilot.py  cost-model usefulness + ECE on the pilot's out-of-fold predictions
results/                    committed logs/JSON behind the proposal's pilot table
pilot/                      exploratory pilot scripts and logs (27 Sep 2026)
tests/                      unit tests (pytest)
```

## Quick start (reproduces every number in the proposal)
```bash
python3 -m pip install -r requirements.txt
mkdir -p data && ln -s /path/to/unzipped/cgmacros data/cgmacros   # folder containing CGMacros/ (PhysioNet, open access)
python3 -m pytest -q tests
python3 scripts/01_build_meals.py      # expect: usable 1526 | onboarding 453 | post-onboarding 1054 | evaluated 896 (634 target, 31 people)
python3 scripts/00_fetch_photos.py     # downloads the 1,486 before-meal photos listed in splits/photo_list.txt (~290 MB)
curl -L -o data/efficientnet-b0-355c32eb.pth https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
python3 scripts/03_photo_experiment.py                      # pilot table, row "Photo, person-excluded"  -> results/photo_experiment.{log,json}
python3 scripts/03_photo_experiment.py --profile-excluded   # row "Photo, also profile-excluded"
python3 scripts/03_photo_experiment.py --noise-control      # row "Noise images (control)"
python3 scripts/03_photo_experiment.py --alpha 100          # ridge-strength sensitivity (also --alpha 1000)
python3 scripts/04_usefulness_pilot.py                      # usefulness/ECE pilot baseline -> results/usefulness_pilot.log
```
Runtime: about 4 minutes for the first 03 run on a 2-core CPU (image features are then cached in results/), about a minute per later run.
No dataset files are stored in this repository (CGMacros is CC BY-NC-SA 4.0; download it from PhysioNet).

## Rules that must never be broken
1. The 8 locked people (splits/) are never used for training, tuning, calibration or thresholds.
2. Every learned stage (image model, fusion, calibrator, conformal, thresholds, OOD threshold) is fit on the outer fold's training people only.
3. Model inputs are only: before-photo, CGM + HR up to meal time, profile/labs/time. Never share eaten, after-photos or consumed macros.
4. Every experiment is logged to W&B and committed.
