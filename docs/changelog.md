# Changelog: improvement round after the 25 Nov cold grades

The submission was tagged `submission-v1` (48693f5) on 25 Nov after two audit rounds. Three further
cold grades that night by AI agents given the rubric (scores 92.5, 86.5 and 88.5 of 100) produced the findings below. Each
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

## 26 Nov 2026, second pass (after a second set of three cold grades: 83.0, 79.5, 82.25)

| Change | Why | Evidence |
|---|---|---|
| The intercept's AUROC gain is reported paired with its interval, +0.037 (−0.004, +0.071), and called not established, the photo gate's standard; log-loss (−0.030, −0.060 to −0.001) and 5 of 5 folds carry the claim; Brier split into reliability and resolution | the central claim was held to a weaker standard than the photo (2 of 3) | report_metrics.json `paired_personal_minus_raw`, `brier_decomposition` |
| Disclosed that the folder command and app answer without history (population calibration), in the report, README, model card and Figure 1 | the report implied the learning loop runs in the product (1 of 3) | src/glimpse/serve/io.py, folder.py |
| T5 explanation corrected: the falling trace's own term raised risk; the high window mean and spread lowered it (2nd and 3rd steepest falls, not 1st and 2nd) | wrong mechanism and rank (2 of 3) | report_metrics.json `shap_terms_t5` |
| Removed "embeddings cached" (no cache exists); latency dated 7 Nov | unsupported claim (2 of 3) | src/glimpse/serve/pipeline.py |
| "Dangerous flips" renamed to flips between low risk and caution (the metric counts both directions); stress runs stated as no-history | mislabelled metric (3 of 3) | src/glimpse/studies/robustness.py `_flips` |
| Full 4 × 4 cost grid; the dev verdict fails two targets at twice the abstain price, now stated; the monotonic-miss sentence corrected; the abstain price explained | hidden sweep axis, contradicted sentence (3 of 3) | dev_summary.json `cost_sweep` |
| Monitoring: the drift week is six synthetic pictures and real photos sit at 0.433 against 0.445; the replay stream is person-ordered; replay 2 at the deployed cut-offs registered, then run (3 of 21 dev windows alert); the missed calibration-in-the-large expectation reported | undisclosed weaknesses and a missed expectation (3 of 3) | outputs/monitoring/, docs/monitoring.md |
| Locked reliability panel in Figure 3; where the model is over- and under-confident; Platt-only ECE (0.023) beside the shipped 0.052 | no test-split reliability diagram (3 of 3) | outputs/figures/f3_reliability.png |
| Quality-vs-cost table now includes the photo branch (frozen + ridge vs fine-tuned); parameter table separates deployed from evaluated-only and adds the 15.5M fine-tuned count | efficiency study omitted the real cost question (2 of 3) | report_metrics.json `fine_tuned_parameters` |
| Photo recipe table says which settings were fixed and which were searched; RQ1's two committed estimates reconciled; D8 "shows" softened to "consistent with", with the paired difference −0.041 (−0.093, +0.008); ridge's better MAE and Spearman stated | overclaims (2 of 3) | outputs/fusion/carbs_check.json, report_metrics.json `d8_paired` |
| Taxonomy adds false cautions, locked errors and spikes among can't judge; T4/T5 split defined as the median distance | only false clears were counted (1 of 3) | report_metrics.json `errors` |
| SHAP sentence: "prior rise first" (not "priors first"); correlation behind the redundancy claim; heart-rate coefficient read; Figure 6 shows the base value | overstatement (3 of 3) | report_metrics.json `shap_terms_t5` |
| Figure 1 redrawn (kNN as a refusal gate, the labelling loop, escalation, no overflow); Figure 4 labels corrected and the RQ1 ceiling bar added | wrong flow and labels (2 of 3) | outputs/figures/ |
| Serving text: an answered meal's reason prints one decimal so it never equals its cut ("risk 15% is below the low-risk cut-off 15%"); decisions unchanged; committed answer files keep their old wording; R11 ledger entry and a test | self-contradicting reason (1 of 3) | src/glimpse/serve/rules.py `decision_reason` |
| `just train-photo` and `just train-final` refuse to overwrite the frozen serving artifacts unless asked; test | `just train` silently replaced them (1 of 3) | src/glimpse/training/final.py |
| `just reproduce-all` (every compute study in order); README rebuild order; CI also checks partition v2; `just clean` no longer deletes tracked demo files | rebuild path and hygiene (1 of 3) | Justfile, .github/workflows/ci.yml |
| Report tests now anchor table cells (results, paired, cost grid, usefulness, stress, parameters) | substring checks could pass on the wrong cell (2 of 3) | tests/test_report_numbers.py |
| Front door: README's intermediate-era table archived; data card's split lines; samples README lists the suites; figures.json binning note | stale pointers (1 of 3) | docs/, samples/ |
| Who graded: the cold grades were run by AI agents given the rubric (stated in the report) | undisclosed grader identity (1 of 3) | report §6 |
| finetune_ledger.csv was edited by hand on 26 Nov to match the corrected generator (rows 1-2 unchanged); regenerating it needs the dataset | possible hand-edited evidence (1 of 3) | outputs/data/finetune_ledger.csv |

Declined in this pass: a history input for the folder command (new serving code that could not be run
end-to-end on the machine doing this round; the grader's entry point is not changed blind); a stress run
through the personalised path, a direction split of the flips and a 15-minute gap level (all need the
dataset); an occlusion probe (needs the weights).
