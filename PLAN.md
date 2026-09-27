# GLIMPSE — 12-week delivery plan (solo)

Principle: build every evaluation script **early**, so the busy second half of term is "run scripts and write up", not "build".

| Week | Deliverable (done = committed + logged) | Risk it retires |
|---|---|---|
| 1 | Git repo live; W&B project; `01_build_meals` reproduces 896/634; unit tests pass; Nutrition5k overhead subset downloaded and counted | Data access, reproducibility |
| 2 | Macro-label audit on photo pairs (share-eaten semantics); fully-consumed subset flag; meal-type grouping for unseen-type test | Served-macro assumption |
| 3 | Image model v1: EfficientNet-B0 on Nutrition5k (frozen → partial unfreeze), carb MAE/bias logged | Photo pipeline works at all |
| 4 | Image model v2 on CGMacros before-photos; **out-of-fold photo predictions** for all development people | Leakage-safe RQ1 inputs |
| 5 | **RQ1 on development folds** (photo vs reconstructed served macros, identical other inputs) — first real answer | Core research question |
| 6 | Sensor encoder (0.1–1M) + fusion with monotone carb path; **RQ2 on development folds** | "Why deep learning" |
| 7 | Calibration (temperature), participant-weighted conformal intervals, thresholds from cost model; usefulness report | Calibration/abstention targets |
| 8 | OOD (DINOv2 distance) + 30 Emirati photos collected; branch-disagreement test | OOD claim |
| 9 | Probes (photo swap, flat vs rising trace, branch masking), failure taxonomy, stress tests | Explainability, robustness |
| 10 | Gradio app in Docker: risk card, slider, abstention, review queue, monitoring; latency benchmark | Product + latency |
| 11 | 5-person think-aloud test; **run locked final set once**; sensitivity sweeps (thresholds 30/70, costs, strict windows) | Final evidence |
| 12 | Report, reproducibility package, demo video | Submission |

**Cut order if behind schedule:** stretch goals → Emirati set reduced to exploratory 15 photos → usability test to 3 people. Never cut: RQ1, RQ2, calibration, leakage rules, locked-set protocol.
