# GLIMPSE pilot (exploratory, 27 Sep 2026)

Exploratory feasibility analysis on CGMacros v1.0.0 (PhysioNet). All 45 participants were used, so these numbers are optimistic and are NOT hold-out results.

| Script | What it does | Log |
|---|---|---|
| analysis2.py | Meal extraction (Dexcom reference, 2-h window), spike >= 50 mg/dL label, GBM baselines B0-B4 with person-level 5-fold x 3 seeds, person-bootstrap CIs, within-group AUROC, simulated photo-macro error, rise MAE | analysis2.log |
| test3.py | Untuned from-scratch 1-D CNN vs GBM; ECE; tiered-policy cost pilot (old cost convention: abstain = 0.5 regardless of outcome); unconstrained vs monotone GBM swap test | test3.log |
| test4.py | Improved CNN (separate encoders, multitask, 5-seed ensemble), self-supervised pretraining, monotone-carbohydrate head, ensembles, swap test on held-out meals | test4.log |

Data path in the scripts: R = "<folder>/cgmacros/CGMacros" (edit to where CGMacros was unzipped). Meal accounting: 1,640 meals with complete 2-h window -> 509 onboarding (first ~30% per person) -> 1,131 -> 170 overlapping (next meal < 2 h) -> 961 evaluation meals (928 with photo).
