# v2 explained and stressed on its own inputs (registered 26 Nov 2026, before any run)

Written by the assistant at the author's request, after round-11 grading pointed out that every
explanation, reliability figure and failure table in the report described v1, not the served v2. The
author's sign-off is in docs/changelog.md.

Both studies are descriptive. Nothing here changes a frozen file: the v2 fusion, Platt map, m and
cut-offs stay as frozen on 26 Nov, the v1 files stay as tagged, and the locked eight are not touched.
Everything except V2-S reads committed development answers (outputs/evaluation_sensor_first/) and the
frozen v2 fusion. Commands: `glimpse v2-explain` and `glimpse v2-input-stress`.

## V2-E: explanation, calibration and failures of the served model

| | what | rule fixed now |
|---|---|---|
| E1 | v2 calibration: ECE noise floor (2,000 Bernoulli draws at v2's own risks), person-bootstrap interval, equal-width reliability with counts and Wilson intervals; a v2 panel in Figure 3 | verdict "met" only if the interval's top is ≤ 0.05 (the protocol's rule); point ≤ 0.05 with a wider interval is "on track" |
| E2 | the frozen band (0.15, 0.22) applied to the development risks | labelled in-sample: these cut-offs were chosen on all development people |
| E3 | missed among cleared against share cleared, low-risk cut from 0.05 to 0.30 | descriptive; the served cut does not move |
| E4 | v2 failure taxonomy: each false clear as near-cut (within 0.031 of its fold's cut, T4) or confident (T5), cold start (at most one labelled meal), fold, glycaemic group, and whether the frozen band would have referred it | same T4/T5 boundary as v1 |
| E5 | exact linear SHAP of the frozen v2 fusion on the 481 meals (in-sample), global and for one meal: the v2 false clear with the lowest risk; plus how much of each served logit is the per-person intercept and how much the fusion | the local meal is chosen by that rule, not by look |
| E6 | heart rate's sign: frozen coefficient, between-person correlation of mean heart rate with spike rate, within-person logistic on person-centred heart rate | none expected; reported either way |
| E7 | what-if, not shipped: withhold low risk for a prediabetic profile (HbA1c 5.7 to 6.4%) until N labelled meals, N in 5, 10, 20 and never | exploratory and post hoc; it would need its own registration and test before serving |
| E8 | each person's last third of meals (by time) against the first two thirds: AUROC and missed among cleared | descriptive |
| E9 | the carb estimate's own value: out-of-fold carb MAE and range accuracy of the photo model against typing the training median | descriptive |

Expected before the run: v2's ECE interval reaches above 0.05, so E1 reads "on track", not "met".
The frozen band refers more meals than the per-fold bands did. Most v2 false clears are confident,
not near-cut. In E7, withholding clears for prediabetic profiles removes most of their misses and
costs most of their clears.

## V2-S: v2 under stress on the inputs it uses

Every development meal, answered by its fold's v2 pipeline with gated history taken from the clean
answers, as in docs/stress_quality.md. Conditions: heart rate removed; half the heart-rate readings
dropped; both onboarding priors removed; history cut to the last 3 and last 10 labelled meals;
10% and 25% of history labels flipped (seeded). Reported per condition: AUROC on co-answered meals
beside its own clean value, caution to low risk and low risk to caution flips, cleared, missed among
cleared.

Expected before the run: removing heart rate lowers risk on average (the folder sample moved from
48.4% to 37.8%) and adds a few caution to low risk flips; flipped history labels cost more AUROC than
a shorter history.
