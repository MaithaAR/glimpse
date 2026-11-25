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
