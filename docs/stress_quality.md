# Stress quality at the evaluated operating point: pre-registration (26 Nov 2026, before its run)

Written by the assistant after the graders asked for quality, not only decision flips, under stress. The
22 Nov levels study (outputs/robustness/levels.json) stays as it is.

What runs: every development meal, answered by its fold's pipeline exactly as the evaluation answers it
(per-fold fusion, Platt, m and cut-offs; the person's earlier meals as history under the two-hour gate),
once clean and once per perturbation of the current meal only; history always comes from the clean
answers. Both products: v2 (served, sensor first) and v1 (photo fusion; its photo perturbations use the
frozen photo model, in-sample for these photos, as the 22 Nov study did).

Levels (the 22 Nov ones plus points around the rules' thresholds, fixed now):
CGM dropout 10/30/50/70%; pre-meal gap 10/15/18/20/22/25/40 min; photo blur radius 2/6/10;
photo brightness x0.5/0.25/0.2/0.15/0.1.

Reported per level: meals answered clean and perturbed; AUROC on the meals answered both ways, clean and
perturbed, and the change; spikes among cleared meals; cost reduction against the best constant policy;
flips split by direction (caution to low risk, the dangerous one, and low risk to caution); newly refused;
recovered (refused clean, answered perturbed).

Expected before the run: v2's risk does not move under photo perturbations (the photo is not a risk
input); v1's does. CGM dropout changes AUROC by less than 0.01 up to 50%. Gaps above 20 minutes refuse
almost every meal, by design. Reported as measured in outputs/robustness/stress_quality.json.

## Deviation note (26 Nov, after the run)
The first run compared live perturbed photo answers with precomputed clean estimates, so blur appeared to
raise AUROC. It was discarded before commit; the code was fixed to compare the live photo model with
itself (374c178) and rerun. The reported numbers are from the second run only. Heart rate, priors and
history were not in this study; they were registered separately in docs/v2_explain.md (V2-S).