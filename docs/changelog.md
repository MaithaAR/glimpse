# Changelog: improvement round after the 25 Nov cold grades

The submission was tagged `submission-v1` (48693f5) on 25 Nov after two audit rounds. Three further
cold grades that night by AI agents given the rubric (scores 92.5, 86.5 and 88.5 of 100) produced the findings below. Each
change is listed with its reason. Nothing in this round changed a serving artifact, a threshold, a
pre-registered verdict or a locked-set number; serving behaviour changed in wording (R11 additions), one
refusal reason (R12) and one output rule (R13, eighth pass), none of which alters an evaluated or locked decision (R13 changes the no-history, no-prior served path, reported as such): `checkpoints/` is byte-identical to the tag, and the locked
eight were not rescored (new locked metrics are arithmetic on the committed `locked_answers.csv`).

**Superseded on 26 Nov by the tenth pass (below):** that pass added a second product, v2, with its own frozen
files in `checkpoints/sensor_first/`, and switched the served product to it in `configs/default.yaml`. The
sentence above stays true of every v1 file, which is still byte-identical to the tag, and of the locked eight,
which were not rescored.

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

## 26 Nov 2026, third pass (after a third set of cold grades: 83.5, 77.5, 85.1)

| Change | Why | Evidence |
|---|---|---|
| The product claim corrected everywhere: the gate's Paragraph B described a sensor-first product that was not built; the frozen fusion takes the photo estimate as 4 of 19 inputs; the typed-carbs tier has no input surface | the report described a product the code does not ship (3 of 3) | src/glimpse/serve/pipeline.py; docs/final_design.md note |
| Typed-carbs risk shown not monotone; recorded as a strict known-failure test (T8) | unevaluated tier with a defect (1 of 3) | tests/test_known_failures.py |
| The served operating point (no history, deployed cut-offs) recomputed from the committed answers: fails two targets on dev, misses 7 of 32 clears locked; prior imputation for profiles without priors disclosed and tested | the point the grader runs was never evaluated (3 of 3) | report_metrics.json `served_operating_point` |
| ECE against its noise floor (a perfectly calibrated model scores 0.052 at n = 460) and a person-bootstrap interval; the "narrow miss" reading withdrawn; per-person offsets reported | ECE read as signal (1 of 3) | `ece_noise_floor`, `person_offsets` |
| Coverage-vs-risk figure (Figure 7) | rubric 4c asks for it by name (3 of 3) | `coverage_risk`, outputs/figures/f7_coverage.png |
| D7's locked primary endpoint disclosed as never computed; secondary endpoint reported | unreported pre-registered endpoint (1 of 3) | docs/final_design.md, docs/evaluation_protocol.md |
| Cost grid: abstention at half the abstain price disclosed (62–76%); constant-policy set stated | hidden failing target (1 of 3) | dev_summary.json `cost_sweep` |
| Monitoring results as a table; pooled calibration-in-the-large (−0.010 locked); overlapping windows and the mixed window stated; signal 4 marked not computable; monitoring.md 30 ms and latency lines corrected | overstated or missing (3 of 3) | outputs/monitoring/ |
| Probe A's 100% stated as guaranteed by linearity; the heart-rate redundancy claim withdrawn (R² 0.19); T5 top-four terms tabled; taxonomy gains root cause, action and T8; T6 corpus fixed (instructor case 11) | overclaims (2 of 3) | report_metrics.json |
| Baselines labelled by meal set and estimator; recipe table says which settings live in code; seed study labelled as the intermediate split; glossary added | ambiguous numbers (3 of 3) | report §1-§3 |
| R12: zero or negative glucose refuses as a broken export, not as mmol/L; incomplete-window reasons name only the failing criteria | wrong advice to the user (1 of 3) | src/glimpse/serve/pipeline.py, rules.py; tests |
| Justfile: dead recipes removed, stale comments fixed, `uv sync --locked`, `just verify-outputs` | front-door hygiene (2 of 3) | Justfile |
| Docs: design-doc note's R11 date corrected (8 Nov); data card's spike-rate ranges labelled by cohort; pipeline docstring (37 people) | stale or wrong (2 of 3) | docs/, src/ |

Declined in this pass: wiring a carbs or history input into the folder command (new serving code on the
grader's entry point that could not be run end-to-end here); a stress run through the personalised path;
real photos in the five stress cases and real-photo format cases; an int8 or no-photo latency study; an
occlusion probe (all need the dataset or the weights).

## 26 Nov 2026, fourth pass (after a fourth set of cold grades: 81.0, 76.75, 77.5)

| Change | Why | Evidence |
|---|---|---|
| Fair baselines: the shipped intercept replayed exactly on every comparison model; gradient boosting reaches 0.750 and the fused system's margin over it is not established; the bottom line now says the intercept carries most of the edge | the results table compared a personalised model with non-personalised baselines (1 of 3) | report_metrics.json `fair_baselines` (reproduces the shipped risk to 1e-16) |
| The abstention band shown to cost money on development (0.555 vs 0.539 per answered meal; the borderline meals spiked at 27%, above the 20% break-even); calibration at the cut-offs added | the band was presented as harmless (1 of 3) | `band_and_served_calibration` |
| Served (Platt-only) calibration added: fine on development (0.023), under-predicts the locked cohort by 11 points (ECE 0.162) | the confidence the grader receives was never calibrated (1 of 3) | same |
| Person-bootstrap intervals on cost reduction for every column | the main business metric had no interval (1 of 3) | same |
| ECE: the biased bootstrap interval dropped; the target called mis-specified for n = 460; README and protocol wording aligned | inconsistent readings across documents (3 of 3) | docs/evaluation_protocol.md errata |
| `cgm_auto` described as the training rule on raw readings, not the training label itself | the label differs by up to 9.5 mg/dL in the demo cases (1 of 3) | src/glimpse/feedback.py |
| Figure 1: R1–R12, the labelling loop marked as evaluation-replay only, the blank-photo exit, the kNN gate labelled; Figure 7 prints n per point | regressions from the third pass and missing counts (3 of 3) | outputs/figures/ |
| T5 text no longer claims one mechanism; taxonomy root cause split; recipe table's warm-up scoped to the heads phase; stress-suite sentence corrected (the CSV stressor is parsed, the answer never reached); monitoring table's cluster and mixed-window counts corrected; replay 2's description corrected | wording errors (2 of 3) | report §4-§5, docs/monitoring.md |
| Test count stated with skips and the expected failure; batch commits disclosed; data card's "no participant data" reworded; instructor MANIFEST notes; Justfile comments; schema note that confirm-carbs is advice only | front-door accuracy (1-2 of 3) | README, docs/, samples/, Justfile |

Still declined, with reasons unchanged: a history or carbs input for the folder command, regenerating the
folder, stress and demo runs under the 26 Nov reason wording, an occlusion probe, an int8 or no-photo latency
study, real photos in the five stress cases (all need the weights or dataset, which this machine did not have;
the regeneration is two commands on the author's Mac, listed in the hand-off).

## 26 Nov 2026, fifth pass (after strict grades 84.25, 80.0, 84.0 and a holistic instructor grade of 85.5)

| Change | Why | Evidence |
|---|---|---|
| Opening rewritten as a short bottom line plus a status table (development, locked, served) | readability; the opening was hard to read (holistic grade) | report top |
| Optional `history.csv` per meal folder feeds the personal intercept through the evaluation's two-hour gate; without it the call is unchanged; tested with a recording stub | the main contribution was unreachable from the grader's entry point (3 of 3) | src/glimpse/serve/folder.py, io.py; tests/test_folder_history.py |
| Stage timing on a 2-CPU cloud machine: the backbone is 80% of the photo path; the no-photo path takes 0.35 ms; int8 timed as cost only | no real quality-vs-cost study (3 of 3) | `just stage-timing` -> outputs/efficiency/stage_timing.json |
| Replay 3 registered, then run: CUSUM on the served risk's calibration does not fire; on the caution share it fires once, in the 7th locked week, with no dev false alarm | no rule caught the real cohort shift (3 of 3) | outputs/monitoring/replay.json `cusum_served` |
| Exploratory "falling now" what-if: 10 of 109 clears flip, 3 of 8 misses avoided, 7 clean clears lost | taxonomy actions had no before/after (3 of 3) | report_metrics.json `falling_now_what_if` |
| T5 prose corrected per meal (row 1024's priors pushed risk up); R6/R7 described as preventive and limitation; heart rate's two numbers reconciled; exact binomial interval for 2 of 10; the verdict cell says the intervals cross the targets; shares sum to 100 | contradictions and overstatements (2-3 of 3) | report §2, §4, §5 |
| Fine-tuned count includes the final norm trained with the last stage (15,475,973) | off by 1,536 (1 of 3) | report_metrics.json |
| Stale code comments fixed (app disclaimer 37 adults, Pipeline.answer, tier-2 docstring, heads comment, section references); Dockerfile copies samples/; figures log prints the registered ECE; Figure 3 hollow markers for bins under 5 meals | front-door accuracy (2 of 3) | src/, Dockerfile, outputs/figures/ |
| Model card R11 dates and ECE floor row; README history.csv, skip note and squash note; rule ledger provenance note; design-doc map of leftover planned names; outputs/evaluation/ labelled as intermediate | stale or dangling references (2 of 3) | docs/, README.md, outputs/evaluation/README.md |

## 26 Nov 2026, sixth pass (after strict grades 82.75, 75.5, 83.0 and a holistic grade of 84)

| Change | Why | Evidence |
|---|---|---|
| Caution-share CUSUM rerun with both streams at the deployed cut-offs (shipped and served risk): no alarm; the fifth pass's locked-week alarm came from the cut-off change at the dev/locked seam; status table and §4 corrected | the alarm was read as catching the cohort shift (2 of 3) | outputs/monitoring/replay.json `cusum_served`; docs/monitoring.md |
| "80% of serving time" corrected to 80% of the photo path; no-photo fusion given the same intercept (0.741, shipped minus it +0.021, −0.003 to +0.053) added to the fair-baseline table | the timing claim misread its own file; the cost side had no quality row (3 of 3) | report §1, §2, §3; report_metrics.json `fair_baselines` |
| Person-bootstrap intervals for the cleared and can't-judge shares; ECE reported with the share of perfectly calibrated draws at or above it; status labels use the protocol's met / on track / at risk | the protocol asks a range for every criterion and names its labels (2 of 3) | report_metrics.json `share_cis`, `ece_noise_floor` |
| Exploratory cold-start what-if (no clear before three labelled meals) and the offset's size by labelled meals | T5 cold starts had no before/after; the intercept's growth with history was unshown | report_metrics.json `cold_start_what_if`, `intercept_by_history` |
| History statements made consistent (the folder command accepts history.csv; the app does not); heads counted once; 27.82M | contradictions (2 of 3) | report §1, §2, §4, §7 |
| `just stage-timing` recipe added and chained in `reproduce-all` | the file named a recipe that did not exist (1 of 3) | Justfile, README |

## 26 Nov 2026, seventh pass (after strict grades 79.75, 78.4, 78.25 and a holistic grade of 82.5)

| Change | Why | Evidence |
|---|---|---|
| The samples' real path evaluated: no history and no onboarding priors (serving medians), approximated on development with the frozen fusion's prior coefficients; AUROC 0.637, 11.4% cleared, 9 of 55 spike, cost −1.5%; status table, §4, model card | the served column used each person's real priors, which no sample has (2 of 3) | report_metrics.json `served_imputed_priors` |
| Probe C registered (commit 32aec59) then run (91f737a): all four profile inputs raise risk on every meal; imputed priors change 11% of decisions | the profile carries the system but had no counterfactual (1 of 3) | docs/probes_s13.md; outputs/probes/probe_c.json; `just probe-c` |
| CUSUM verdict reworded: the registered caution-share rule fired; part of the step is the cut-off change; at deployed cut-offs the statistic reaches 2.6 of 4; the calibration expectation failed | the sixth pass called the alarm an artifact on post-hoc rows (1 of 3) | report §4; docs/monitoring.md |
| Exploratory what-if table (falling now, cold-start hold) with cost; post-hoc labelled | before/after for the taxonomy actions (3 of 3) | report §5; report_metrics.json |
| Figures: F1 shows the history.csv path; F2 title neutral with the pooled interval; F3 counts unhidden, lines only through bins with 5+ meals; F6 says "window spread" | stale or misleading figure text (3 of 3) | outputs/figures/ |
| Text: "the change that worked" softened; "not met" next to the protocol labels; dev ECE "not assessable"; locked one-shot 0.098 shown; person-bootstrap 0–100% beside the exact interval; scoring-path sentence scoped; harm paragraph; ownership sentence separates the overnight registrations; stale monitoring lines corrected | overstatement and inconsistencies (2-3 of 3) | report, docs/monitoring.md, docs/model_card.md |
| Subgroup table by glycaemic status (AUROC interval, calibration gap, clears and misses): the intended users, people with prediabetes, get 9.7% of meals cleared and are under-predicted by 6 points | the fairness line had no intervals or usefulness split (2 of 3) | report §3; report_metrics.json `subgroups` |
| Fresh-clone record committed from the dated notes (not a terminal log); ridge arm labelled post hoc; dev miss-interval source named; test count 168 | unverifiable claim; provenance (2 of 3) | outputs/finalize/fresh_clone_2026-11-25.md |

The overnight registrations (replay 3, probe C, the 26 Nov what-ifs) were written by the assistant under the
author's standing instruction, not approved one by one.

## 26 Nov 2026, eighth pass (after strict grades 76.25, 76.5, 73.0 and a holistic grade of 81)

| Change | Why | Evidence |
|---|---|---|
| T5: the photo block is negative on all four confident false clears but on 82% of clean clears too; 4 of 8 false clears are in fold 1; a fold-1-excluded usefulness row (19.0%, 4 of 73, 13.3%) | the photo's combined term and the contaminated fold were not followed into the errors (1-2 of 3) | report_metrics.json `t5_photo_and_heart_rate`, `usefulness_excluding_fold_1` |
| Probe C's direction result called guaranteed by linearity, like probe A; decision changes reported as the finding | a tautology presented as a pass (3 of 3) | report §5 |
| Harm paragraph corrected: 7 of 8 development false clears fell on healthy people | the paragraph named the wrong group (1 of 3) | report §7; `subgroups` |
| Promised comparisons table (class weighting, label threshold, masked photo); served-decision CUSUM row; "trend + profile LR" naming; dev verdict "1 met, 3 on track"; T1 count 7 of 13; "evaluated system (with history)"; 150 scored locked meals; locked path "at the tag" | omissions and inconsistencies (1-3 of 3) | report §3-§6 |
| Test: today's decision function reproduces every committed locked decision at the frozen cut-offs (nothing rescored) | the post-tag serving changes needed a check against the locked answers (1 of 3) | tests/test_report_numbers.py |
| R7 mitigation sentence no longer cites the unwired confirm-carbs tier; abstention next step says "a new photo" | the escalation named a surface that does not exist (2 of 3) | report §1, §5 |
| Rule R13 (output rule): no low risk without history or both onboarding priors; the meal is referred with a named reason. On the samples' path (dev, approximate) clears 55 -> 0, misses 9 -> 0, can't judge 17.3%, cost -0.3%. No risk, weight, Platt map or cut-off changed; evaluated and locked answers all carry priors, and a test re-decides the locked answers unchanged. Unit and bare-pipeline tests; not run end to end with weights | the report said no meal should be cleared on that path while the code cleared 11% (3 of 3) | src/glimpse/serve/rules.py, pipeline.py; docs/rule_ledger.md; tests/test_serve.py |
| Figure 1: R13 in the cut-off box, "new photo or dietitian", profile with optional priors | stale labels (2 of 3) | outputs/figures/f1_system.png |

## 26 Nov 2026, ninth pass (after strict grades 76.0, 81.5, 74.5 and a holistic grade of 88)

| Change | Why | Evidence |
|---|---|---|
| Within-person AUROC: averaged per person, raw 0.680 against shipped 0.602; a no-model running spike rate under the same gate scores 0.699, shipped +0.063 (+0.030, +0.113). The intercept's gain is between people (base rates), not within them | the pooled AUROC could come entirely from base-rate learning; no within-person number existed (1 of 3) | report_metrics.json `within_person`; report §3 |
| Bottom line corrected for R13 (the samples' path now clears nothing); Figure 1 subtitle scoped; gate interval labelled "not the gate" in §1; recovered answers under degraded photos noted; probe C counts reconciled with §4; R13 described as answering a measured failure | the opening described pre-R13 behaviour; inconsistencies (2-3 of 3) | report; outputs/figures/f1_system.png |
| README diagram, samples/README (prior fields, R13), instructor MANIFEST row 09, this changelog's header | front-door staleness (2 of 3) | README.md, samples/ |
| CGM-removal contrast with a paired interval: no-photo arm minus profile only, both with the intercept, +0.033 (+0.007, +0.065), the one established sensor contribution; unfamiliar-photo table (training leave-one-out 1.1% refused, 0 of 14 real local photos, synthetic week median 0.709, instructor screenshot refused, dark photo 0.613 caught by the contrast rule) | no direct CGM ablation; OOD evidence scattered (3 of 3) | report_metrics.json `fair_baselines.cgm_removal`; report §3, §4 |
| Figure 7 gains a cost panel from `coverage_risk`: on development any abstention raises cost; on the locked eight abstaining the nearest 10% would have lowered it (0.543 against 0.593), descriptive | the coverage curve was not tied to the cost model (2 of 3) | outputs/figures/f7_coverage.png |
| Figure 6's local panel shows row 847, a confident false clear, instead of the meal nearest risk 0.5 | the old example said nothing about the 0.15 cut (1 of 3) | outputs/figures/f6_shap.png |
| Figure 4 adds the intercept's paired gain and the CGM contrast (green) and states the removal sign convention | the central effect and the CGM contribution were missing from the effects figure (2 of 3) | outputs/figures/f4_effects.png |

## 26 Nov 2026, close of the round (after strict grades 75.0, 79.0, 77.0 and a holistic grade of 85)

Last fixes: the CGM contrast labelled exploratory (the no-photo arm keeps heart rate); R13 described the same
way everywhere (no low risk with neither history nor both onboarding priors); R13's cost stated as -0.3%;
Figure 4's D8 row drawn as a control; Figure 6's local risk labelled raw fusion; this header no longer
claims R13 changes no decision. Still open and needing the author's Mac (weights and data): rerun the
folder, stress and demo suites under the current code and commit them with a fresh-clone terminal log;
int8 decision agreement; an image-level probe of the ConvNeXt; author sign-off on the overnight
registrations and on R13.
Added at the close: the within-person drop has a person-bootstrap interval, −0.078 (−0.110, −0.046), with
19 of 22 people worse: the intercept is established to hurt the ranking of a person's own meals while it
helps between people (report §3; report_metrics.json `within_person`).
Also at the close: signal 2b (rolling AUROC below the locked value) added to the monitoring table, labelled
circular because its threshold is the locked AUROC itself (outputs/monitoring/replay.json `summary`).
Figure 3 shows a 95% Wilson interval for each shipped bin, so small bins are not over-read.

## 26 Nov 2026, tenth pass: v2, the sensor-first product (the author chose it after the ninth pass)

The author chose to ship the product the gate's Paragraph B called for. Everything below was pre-registered in
a commit before it ran; the assistant wrote the registrations under her instruction. The data and pinned
weights were copied from her Mac into the cloud workspace for this round (read only on her side).

| Change | Why | Evidence |
|---|---|---|
| Folder, stress and demo runs regenerated under the current serving code with the weights and data; every v1 decision equals the committed one (risks within 4e-7) | committed runs predated R11-R13 (3 of 3) | outputs/v1_photo_fusion/; commit d54ded7 |
| v2 registered (6874698), built, evaluated on development (AUROC 0.754, 0.673-0.815, all 481 meals; ECE 0.029; spikes among cleared 12.7%), frozen (Platt 0.98, -0.17; m 1; cut-offs 0.15, 0.22) and served by a config switch; v1's files byte-identical | the gate's Paragraph B was never built; the budget sat in a branch that failed its gate (3 of 3) | docs/v2_sensor_first.md; outputs/evaluation_sensor_first/; checkpoints/sensor_first/ |
| v2 minus v1 on the 460 meals both answer: -0.017 (-0.045, +0.010), not non-inferior at 0.03; reported in the opening as the registration requires | registered endpoint | outputs/report_metrics/v2_metrics.json |
| Stress quality at the evaluated operating point, both products, 19 levels each including 15, 18, 20 and 22-minute gaps (registered 61b0c2e). The first run compared live perturbed photo answers with precomputed clean ones; it was discarded before commit, the code fixed (374c178) and rerun | quality, not flips, under stress; gap levels around the rule (3 of 3) | outputs/robustness/stress_quality.json; docs/stress_quality.md |
| Photo probes D (centre vs border occlusion: hypothesis holds, 84%) and E (zoom: grams rise on only 40%), and int8 against fp32 (97.0% of v1 decisions kept, 55 against 77 ms) (registered c783ce3) | the photo model was never probed; int8 had no quality side (3 of 3) | outputs/probes/photo_probes.json; outputs/efficiency/int8.json |
| Monitoring replay 4: detection power under a simulated cohort shift (registered ec630a8): calibration CUSUM 23% within 120 answers, 5% false alarms; expectation not met | no evidence any rule can catch a realistic shift (3 of 3) | outputs/monitoring/power.json |
| End-to-end timing with the pinned weights: a whole folder-command meal takes 115 ms (v2) and 116 ms (v1), 6.4 ms for v2 without a photo; CSV parsing 4 ms | 265 ms against a 59 ms stage table never reconciled (3 of 3) | outputs/efficiency/end_to_end.json |
| samples/returning_user: the same meal at three stages (new user caution, with priors caution, with priors and history low risk) | the grader could never see a low-risk answer (3 of 3) | outputs/returning_user_run/ |
| Report rewritten around v1 and v2 (2,193 prose words); secondary tables and the previous prose moved to docs/report_appendix.md, tested the same way; model card, README, rule ledger and Figure 1 updated | readability and the product change (3 of 3) | docs/ |
| `just evaluate` runs both evaluations and v2's metrics; `evaluate-all`, `stress-quality`, `photo-probes`, `monitoring-power`, `finalize-sensor-first` recipes; outputs/ tracked; CI re-derives the data-free outputs | `evaluate` did not rebuild every table (3 of 3) | Justfile; .github/workflows/ci.yml |

## 26 Nov 2026, eleventh pass (round-11 grades: 76.25, 77.0, 78.5; holistic 84)

All three strict graders and the holistic one found the report explaining and charting v1 while serving v2.

| Change | Why | Evidence |
|---|---|---|
| v2 explained, calibrated and its failures sorted (registered 10679f9 before the run): ECE noise floor and interval (0.029, 0.032-0.091: on track, not met), reliability panel in Figure 3, frozen band in-sample (refers 18%), miss curve, taxonomy of v2's 15 false clears (6 near-cut, 9 confident, 6 on people with prediabetes), exact SHAP and the intercept split (Figure 8), heart rate's within-person sign, a prediabetes what-if (not shipped), last-third check, the carb estimate's own value | the served model had no reliability diagram, attribution or taxonomy (4 of 4) | outputs/v2_explain/; docs/v2_explain.md |
| v2 under stress on its own inputs (same registration): heart rate removed gives 17 caution to low risk flips, the largest of any condition; 25% wrong history labels cost 0.055 AUROC | v2's robustness skipped its own dependencies (3 of 3) | outputs/robustness/v2_input_stress.json |
| Report: "calibrated" and "met" replaced by the measured verdict; the band sentence corrected (the 40 referred meals came from per-fold bands 0-10 points wide); a paragraph on v2 as the riskier product and the decision timeline; v2 taxonomy in the main table; like-for-like 0.748; parameter table without double counting; probe D and E caveats; int8 carb MAE and the 1,221 to 460 photo deviation; stress denominators; caution-share alarm stated as measured; Figure 5 to the appendix; the old v1 prose to docs/archive/ | overstatements and internal inconsistencies (3 of 3) | docs/final_report.md |
| `just evaluate` adds the v2 explanation; `evaluate-all` adds v2 input stress, end-to-end timing and the 22 Nov levels; `just end-to-end-timing`; `verify-outputs` compares numbers to 1e-9 instead of bytes; `evaluation_v2/dev_answers.csv` regenerated under the final reason wording (decisions unchanged) | 1e-16 noise failed the byte check; recipes missing (2 of 3) | Justfile; src/glimpse/verify.py |
| README, samples README, instructor manifest, rule ledger and model card brought up to v2 | stale statements (3 of 3) | – |

## 26 Nov 2026, twelfth pass (round-12 grades: 78.0, 77.75, 79.2; holistic 86)

| Change | Why | Evidence |
|---|---|---|
| The last-third check now reports misses among clears (9.6% to 17.8%), and the §7 recommendation no longer claims history makes clears safer | selective reporting (3 of 3) | v2_explain.json temporal |
| ECE shown against the perfect-model floor (0.029 is lower than 99% of draws, median 0.051) instead of an interval that excluded its point; the Wilson sentence corrected | the point lay outside its interval (4 of 4) | v2_explain.json calibration |
| Heart-rate removal's lower misses and higher saving reported; "recent activity" reworded as an untested candidate; probe C's scope named; GB wording; registration-order claim narrowed to commit order; locked verdict "not met"; v1 named the more accurate model in the opening | overstatements (2 of 3) | docs/final_report.md |
| Figures renumbered in reading order (the 22 Nov levels figure is A1 in the appendix); Figure 4 marks the evaluated point and uses one feature name | figure order and labels (3 of 3) | outputs/figures/ |
| The sentence on the author's sign-off now points to the entry below, which she writes | the report claimed a sign-off that was not yet there (3 of 3) | – |

## 26 Nov 2026, thirteenth pass (registered 131bd8e before any change)

| Change | Why | Evidence |
|---|---|---|
| R14 low glucose and R15 out of scope; refusals name who acts next (`fix_input` or `human`); carb heads' disagreement flagged and shown as unsure; history rows tagged by product; device-export columns, sampling interval, profile field names and inferred meal time handled | an unseen hypoglycaemic meal got walk advice; T2D profiles were served; Figure 1 contradicted the code; 36.5 g shown as under 30 g; a v1 history could set v2's intercept (3 of 3) | docs/rule_ledger.md; tests/test_hardening_r14.py; samples/stress/24, 25 |
| v2 probes B and C, v2 input monitors, false-clear decomposition, history-length table, carb-head disagreement rate; v2 in Figures 2 and 5 | v2 had no controlled probes or monitors of its own inputs (3 of 3) | outputs/probes/v2_probes.json; outputs/v2_explain/ |
| v1 studies (probes, promised, robustness) pin `sensor_first=False`; under the v2 default config they crashed | found while making `just evaluate` cover every table | src/glimpse/studies/ |
| `just evaluate` adds the budget, v2 probes, OOD, monitoring and probes A and B; the budget CSV has a v2 row and no never-built model; recipe times corrected | `evaluate` did not rebuild every table (3 of 3) | Justfile; outputs/param_budget.csv |

## 26 Nov 2026, fourteenth pass (round-13 grades: 81.9, 81.0, 80.75; holistic 86)

| Change | Why | Evidence |
|---|---|---|
| `pytest` and `uv run pytest` (so `just test`, `just check` and CI) failed at collection since ebb2878: tests imported `tests.test_serve` with no path set; the logs had used `python -m pytest`. `pythonpath` added; logs regenerated with `just test` | broken documented command (1 of 3, the most serious finding) | pyproject.toml; outputs/finalize/ |
| v2 minus v1 now the observed difference (−0.015), as v1's paired rows are; the bootstrap mean (−0.017) is kept in v2_metrics.json | mixed conventions in one table (1 of 3) | outputs/report_metrics/v2_metrics.json |
| Bottom line: calibration "on track", v1 "scores higher by a difference the data cannot establish"; within-person AUROC levels (0.60, 0.54); baselines on 481 meals; photo gain like for like (+0.021); v2 false cautions and the missing-heart-rate failure in the taxonomy; registered misses labelled (probe C half met, B4 expectation not met); Figure 4 left reframed, 847 explained; R1-R15 everywhere, R15 preventive; dates corrected to 26 Nov (all commits are 26 Nov, +0400) | inconsistencies and unlabelled misses (3 of 3) | docs/final_report.md |
| LibreView exports with a title line and mmol/L columns now reach rule R1's x18 guidance; R14's message prints one decimal (53.7, not 54) | unseen device exports (1 of 3) | tests/test_hardening_r14.py |
| The photo recipe and unfamiliar-photo tables moved to the appendix with one sentence each in the report, to keep the PDF at 8 pages | page limit | docs/report_appendix.md |

Registration-to-results times since the tag (commit order is the only evidence of order): v2 6874698 to
3a7e72f, 4 min; replay 4 ec630a8 to 7fd3e8f, 1 min; stress quality 61b0c2e to 374c178, 23 min; probes and int8
c783ce3 to d7bfc6e, 26 min; v2 explanation 10679f9 to a0e11ef, 20 min; hardening 131bd8e to 24d601e, 11 min.
Code was sometimes drafted before its registration was committed.

## 26 Nov 2026, fifteenth pass (after a council review of the last day's options)

| Change | Why | Evidence |
|---|---|---|
| Rule R16: v2 refuses a CGM or heart-rate input beyond the training range in the risk-lowering direction; registered b9c2654 and approved by the author before the run, then narrowed by her twice after the first counts: to the 7 sensor inputs (70 of 153 locked meals lay outside the training ages or BMI) and to the risk-lowering side (9 smooth synthetic stress traces were refused) | a heart rate of 160 bpm alone turned a prediabetic caution into low risk (round 14) | docs/v2_hardening.md Part C; outputs/probes/v2_hr_sweep.json |
| The prediabetes withholding rule stays undeployed by the author's decision; the report states that the served v2 breaks the cap and why | the report recommended a rule the served product does not apply | docs/final_report.md §4 |
| Figure files renamed to their report numbers (fig1 to fig7, figA1 in the appendix) | file names did not match (3 of 3) | outputs/figures/ |

## 26 Nov 2026, sixteenth pass (the author's decision after a council review; registered 7c8f7f6 and 3f580fc)

| Change | Why | Evidence |
|---|---|---|
| v1 served again; v2 kept as the tested alternative; R16 extended to v1 (no evaluated decision changes) | v2 broke the 10% cap and had no clean test; v1 is the only model scored on the locked eight | docs/final_switch.md; configs/default.yaml |
| Served runs are v1's (outputs/folder_run, stress_run, returning_user_run, demo); v2's moved to outputs/v2_sensor_first/ | the served product changed | outputs/ |
| Studies O1 (photo OOD AUROC 0.988), O2 (out-of-scope AUROC 0.89), M1 (monitoring power 23-69%, expectation not met), C1 (v1's ECE typical under an adaptive-intercept null; v2's is not), S1-S5 course checks (shuffled labels 0.475, learning curve still rising, worst slices, no photo identity shortcut, photo-model overfitting) | the course's own tests, found by reading every lecture and lab | outputs/final_checks/final_checks.json |
| D8 moved into the report: the photo's gain vanishes without repeated meal profiles | the professor's intermediate feedback named this risk | docs/final_report.md §2 |
| Report rewritten around v1 served; a debugging-checklist table names the gaps (LR range test, label audit, zero-shot baseline) | the report contradicted the served product | docs/final_report.md |

## 26 Nov 2026, seventeenth pass (after three cold grades against the rubric; presentation only, no new study)

| Change | Why | Evidence |
|---|---|---|
| Registered predictions marked met or not met (S1-S5, C1, O1, O2); S1 shown as a range; S3 meal counts corrected | a consistency read found failed predictions reported without saying so | docs/final_report.md §3, §4 |
| Training curves in the report (Figure 3, from the committed histories); photo recipe, freezing ablation and seed rows in §2; SensorCNN architecture; ConvNeXt cited | rubric 3d: training choices read as fixed, not shown (3 of 3 graders) | src/glimpse/figures.py `f9_training`, outputs/figures/fig_training.png |
| The coverage figure, v2's history-length result and duplicated "why v1" text moved to the appendix | page limit; graders asked for less v2 bulk | docs/report_appendix.md |
| Failure table states before/after for the rules (naive run 3ee062b: 2 raw errors, 1 whole-file rejection; now 0 of 25) | rubric 5b | docs/final_report.md §5 |
| README grader path with the three answers and the no-history rule; history note corrected (GitHub history runs from 27 Sep); W&B offline runs committed; memory and parameter-budget sources in §1 | rubric 6b, 6c | README.md, outputs/wandb/ |

## 26 Nov 2026, eighteenth pass (registered b99375c, docs/robustness_v3.md, before any run)

| Change | Why | Evidence |
|---|---|---|
| R1-R4 run: LR range test (used rates within 10x of the suggestion), one-batch overfit (both models fit), label audit (worst 20 meals: 20% boundary rises against 9%), classical photo baseline (27.5 g, worse than frozen summaries + ridge); zero-shot not run (no network for CLIP weights) | the report listed these as gaps; the professor's week 3, 7 and 11 tests | outputs/robustness_v3/robustness_v3.json |
| v3 photo candidate (R5-R7): regularised heads chosen on inner folds (the strongest setting won in every fold); gap 0.35 to 0.13, carb MAE 22.6 to 26.0 g (prediction not met), AUROC −0.007; without repeated meals the photo gain is +0.009 (−0.010, +0.031); seed spread 0.005 against a 0.030 gain (prediction not met). Not recommended by the registered rule; v1 stays served | the 0.35 overfit gap and D8 | docs/final_report.md §5 |
| R8: development re-weighted to the locked mix: AUROC 0.762 to 0.749 | is the locked drop just cohort mix? | docs/final_report.md §6 |
| Probes A and C detail, the glossary and two v2 paragraphs moved to the appendix | page limit | docs/report_appendix.md |

## 26 Nov 2026, nineteenth pass (a line-by-line audit of about 770 claims against the evidence files)

| Change | Why | Evidence |
|---|---|---|
| "8 people, 6 with prediabetes" corrected to 6 of the 15 false clears (3 of the 8 people have prediabetes) | a count of meals written as a count of people | outputs/v2_explain/v2_explain.json (taxonomy) |
| R7 reworded: the seed spread is below v3's gain (prediction not met); "real, and it is meal recognition" removed | an overclaim, and a failed prediction not labelled | outputs/robustness_v3/robustness_v3.json |
| README: no low-risk answer needs both no history and no priors (it said no history alone); figure numbers; the returning-user manifest now shows v1's answers | stale or wrong after the switch to v1 | README.md, samples/returning_user/MANIFEST.md |
| Smaller precision fixes: heart rate 140 bpm (v1) and 160 bpm (v2); memory measured for v1 only; the intercept's gain stated with its interval; noise-floor 95th centiles added; the locked no-history point given as 20.9% cleared, 21.9% spiking; probe B 8.5% in both products; sources added to three captions; the parameter-budget label | imprecise wording or a missing source (about 30 of 770 claims) | docs/final_report.md |
| Figures 2, 6 and 7 narrower; the glycaemic-status table moved to the appendix | page limit | docs/report_appendix.md |

## 26 Nov 2026, twentieth pass (three fresh grades, 78 to 83; items 1 to 7 of the plan that followed)

| Change | Why | Evidence |
|---|---|---|
| Rules R17 to R19 (registered 1ba839d): semicolon CSVs read, a folder's only image used, "Low"/"High" glucose text mapped to the sensor limits | 4 of a grader's 19 unseen inputs were mishandled | docs/rule_ledger.md, tests/test_input_rules_r17_r19.py |
| Bottom line rewritten in plain words; two status rows moved to the appendix | all three graders found page one hard to read | docs/final_report.md |
| A "what was trained" table in §2 (start weights, trained parameters, loss and optimiser, epochs, validation result) | the deep learning read as thin | §2 |
| Codes replaced with names (repeated-meal check, robustness checks 1 to 8, the final split); the "trained from scratch in the product" row removed and SensorCNN named as the from-scratch network | 22 undefined codes; the row read as dodging the rule | §1 to §6 |
| Figure 5 shows v1's and v2's misses as the cut moves (v2's SHAP panel dropped); Figure 3's loss axis made readable; Figure 1 wider | v1's curve was missing; Figure 5 repeated Figure 7 | src/glimpse/figures.py |
| §7 ends with the three weakest points and what the evidence supports | the week 13 habit; rubric 5c | §7 |
| Rules R20, R21 (registered 2376328): day/month dates resolved by the meal time, refused when both or neither fit (a Libre export had been misread silently); units in profile values | a fresh grader's unseen inputs (grade 85 after the pass) | docs/rule_ledger.md |
| v2-only stress rows, decisions per 100 meals, probe E and one quality row moved to the appendix | page limit | docs/report_appendix.md |

## 26 Nov 2026, twenty-first pass (two more fresh grades, 90.5 and 84; changes approved by the author)

| Change | Why | Evidence |
|---|---|---|
| Rule R22 (no clear on a pre-meal glucose of 140 mg/dL or more, or rising faster than 1.5 mg/dL/min) and the R14 amendment (a falling 54-69 mg/dL is referred); registered 69734f7, approved before the count. On development R22 turns 22 of v1's 109 clears into caution (2 spiked): clears 18.1%, misses among clears 6.9% | a grader got "low risk" at 185 mg/dL, while rising 92 to 164, and at 58 falling | outputs/rules_r22/rules_r22.json |
| Holm over the 7 paired development contrasts (2 survive); the intercept headline gives its interval; person-clustered locked miss interval; subgroup intervals; the served no-history point and the rules' effect on page one | statistics flagged by the adversarial grader | outputs/report_metrics/holm.json |
| How the served cut-offs were chosen; abstention does not buy accuracy; which ablations were registered | rubric 4b, 3c | §3, §4 |
| A meal time with a UTC offset next to local CGM times refused with the right reason; extra images flagged; SI profile units get the conversion | the grader's unseen inputs | src/glimpse/serve/ |
| README checkpoint table (served, alternative, baseline, superseded) | rubric 6d | README.md |
| v3 paragraph and v2's monitoring rows moved to the appendix | page limit | docs/report_appendix.md |

Author sign-off on the tenth to twenty-first passes (v2, the return to v1, R13 to R16 and every registration since the tag):
to be added here by the author, dated, in her words.

