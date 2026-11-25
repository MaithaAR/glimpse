# Changelog: improvement round after the 25 Nov cold grades

The submission was tagged `submission-v1` (48693f5) on 25 Nov after two audit rounds. Three further
independent cold grades that night (scores 92.5, 86.5 and 88.5 of 100) produced the findings below. Each
change is listed with its reason. Nothing in this round changed a serving artifact, a threshold, a
pre-registered verdict or a locked-set number: `checkpoints/` is byte-identical to the tag, and the locked
eight were not rescored (new locked metrics are arithmetic on the committed `locked_answers.csv`).

## 26 Nov 2026

| Change | Why (grader finding) | Evidence |
|---|---|---|
| Monitoring replay registered in docs/monitoring.md, then run; result reported as measured (the outcome rule fires on 7 of 21 ordinary dev windows and cannot judge the locked cohort; the new behaviour rule cannot fire) | monitoring demonstrated one signal on a six-photo week (3 of 3) | `just monitoring-replay` -> outputs/monitoring/replay.json |
| AUPRC and Brier reported for dev and locked, with person-clustered intervals | registered in the protocol, never reported (3 of 3) | `just report-metrics` -> outputs/report_metrics/report_metrics.json |
| ECE reported in the protocol's registered binning (15 equal-width) next to equal-mass; the narrow dev miss (0.052) stated; Figure 3 redrawn in the registered bins with meals per bin | binning switch unrecorded, no per-bin counts, target status never given (3 of 3) | same file; docs/evaluation_protocol.md errata |
| Figure 5 redrawn from the same run as the stress table (levels.json), plus the meal-time shifts and missing heart rate | the figure showed the earlier single-point run beside the table (3 of 3) | outputs/figures/f5_robustness.png |
| Dangerous-flip claim scoped to the registered levels; the 3.7% missing-heart-rate case reported | the claim omitted a committed worse case (1 of 3) | outputs/promised/promised.json |
| Feedback mechanism described in the report | rubric 2e, absent (2 of 3) | src/glimpse/feedback.py |
| Probe B described as a failed expected pass | the report said "committed expected-fail" (1 of 3) | docs/probes_s13.md |
| Baselines, quality-vs-cost table, photo recipe table, "what the photo does" table, usefulness table with constant-policy costs, taxonomy table with a corpus column | rubric 2b, 2c, 3b, 3d, 3e, 4b, 5b (1-3 of 3) | outputs/report_metrics/, outputs/cv/, outputs/photo_cv/ |
| RQ2 quoted on partition v2 (SensorCNN 0.688 ± 0.007 vs 0.686, paired +0.002) | an intermediate-partition number sat among v2 numbers (2 of 3) | outputs/cv/cv_summary.json |
| T5 confident false clears: an exploratory look at their inputs, labelled as chosen after the outcome | the blind spot was counted, not examined (1 of 3) | report_metrics.json `false_clears_exploratory` |
| `just report-numbers` (data-free) chained into `just evaluate`; README maps every report table to its recipe and file; tests/test_report_numbers.py checks report numbers against the files | `just evaluate` did not regenerate every report number (2 of 3) | Justfile, tests/ |
| Figure 6 renders without the dataset from the committed standardised features; its local label now shows the frozen fusion's own risk (0.51, was the per-fold raw risk 0.50) | label mismatch (1 of 3) | outputs/figures/ |
| Figure 1 says R1-R11; Figure 2 says answered meals | stale labels (2 of 3) | outputs/figures/ |
| Final model card written; the intermediate card moved to docs/archive/ | the front-door card was the superseded one (1 of 3) | docs/model_card.md |
| Errata blocks (nothing above them edited): protocol (+0.016 and +0.006 points, the unsourced +0.020, dietitian status, binning, AUPRC/Brier); design doc (actual freeze dates, evidence names as shipped); rule ledger (R11 commit order vs the locked run) | stale companion documents (2 of 3) | docs/ |
| Fine-tuning ledger: the cut Nutrition5k row now counts 0 items (total 1,221); generator label corrected to match | the CSV still said "final, planned" (1 of 3) | outputs/data/finetune_ledger.csv, src/glimpse/data/quality.py |
| Data card leads with partition v2 | the v1 split read as current (1 of 3) | docs/data_card.md |
| README leads with the locked column | the front door bolded the contaminated estimate first (1 of 3) | README.md |

Declined, with reasons: rescoring or retuning anything (forbidden by the pre-registration); a full
three-seed photo CV (about 4.5 h of retraining for a variance estimate the frozen verdict cannot use; the
seed study of the final recipe stands); an occlusion probe of the ConvNeXt (needs the dataset and weights,
not reachable from the machine doing this round); committing the W&B runtime directory (the runs are
replays of the committed CSVs, which stay the primary evidence).
