# Archived: the report's v1 prose before the v2 rewrite (26 Nov 2026)

Kept for the record; superseded by docs/final_report.md.


These paragraphs describe v1 and were checked number by number at the time; the report now summarises them.

**Bottom line.** A per-person intercept, fitted only on each person's earlier meals and replayed in

### 1. System and tooling

The objective: develop and internally validate, on person-grouped splits, a pre-meal spike-risk model,
and test under pre-registered gates whether the meal photo adds signal beyond sensors and profile. An
adult with prediabetes in a CGM programme wants fewer spikes but stops logging food within days; they
photograph the meal and get an answer with its reason in one plain sentence. The CGM labels every meal
two hours later (`cgm_auto`), so the system can learn each
person without logging; abstentions name a next step (a new photo, a dietitian). A wellness nudge, not a medical device.

One scoring path (Figure 1): every answer here, the demo, the app and the grader's
`just predict-folder <in> <out>` go through the same `Pipeline.answer`, with plain-code input rules first
(§4); served-point, baseline and what-if rows are arithmetic on those logged answers. Without history the
folder command and the app answer at the population calibration, since the intercept needs labelled meals,
which the evaluation replays in deployment order. An optional `history.csv` per meal folder (earlier
answers' fusion logits and CGM labels) now feeds it through the same two-hour gate; the samples carry
none. A profile without onboarding priors (every instructor case) gets training medians drawn from all 37
non-locked people, type 2 diabetes included (spike rate 0.455, rise 48 mg/dL). Tooling: uv
+ Python 3.14 (locked), strict Pydantic throughout, a `just` recipe for every number, 176 tests in CI (159 run without the dataset or weights; 16 skip, 1 expected failure),
W&B offline runs replayed from the committed training histories, and a git history (from 2 Nov; earlier
ordering in the dated design docs) in which every pre-registration precedes its run.

### 2. Multimodal design and the photo gate

Three parts. A pinned, SHA-verified ConvNeXt-Tiny backbone (ImageNet-22k pre-trained, 1k fine-tuned)
feeds two small heads, grams regression and a 4-bin carb classifier, with only the last backbone stage
fine-tuned. The profile comprises eight inputs: age, BMI, HbA1c, fasting glucose, meal hour (as sine and
cosine) and two onboarding priors (spike rate and mean rise). Hand-made trend features summarise the 2-h
CGM/HR window. SensorCNN, the required from-scratch model, is kept as the RQ2 baseline only: its paired
gain over the 16-parameter regression is +0.002 (−0.016, +0.022), failing the pre-registered bar, so the
hand-made features stayed.

Fusion is a 20-parameter logistic regression, the capacity 23 development people support and the course's
prescription for this data size (frozen pretrained features plus a simple classifier). Gradient boosting
on the same 19 fusion inputs is no better: the shipped regression is non-inferior under the pre-registered
bar (paired +0.016, −0.024 to +0.062).

Per the pre-commitment (docs/gate_s8.md held both outcome paragraphs before the run), the photo is not an established risk signal. That paragraph went further:
the answer would rest on trend and profile, with the photo as a carb-entry assist. That product was not
built. The 5 Nov amendment regenerated the existing photo-arm fusion, so the frozen model takes the photo
estimate as 4 of its 19 inputs and refuses a meal without a usable photo; the typed-carbs tier exists in
code, but no input sets it, and risk would not be monotone in typed carbs (T8). The verdict changed the claim, not the model; nothing is reshipped.
RQ1 says why the photo adds little: true carbs add at most about 0.03 here, a ceiling the photo
matches; D8 is consistent with the gain riding on repeated standardised meals. This leaves all but 20 of the
27.82M deployed parameters in a branch that failed its gate: the price of testing the photo properly. It buys better carb bins than the ridge (F1 0.61 against 0.38) though a worse
MAE and Spearman, where patients' own counts err by about 15–20% (Brazeau et al. 2013, Vasiloglou et al.
2018), plus the unfamiliar-photo refusal. Heart rate earns nothing either: retrained without it, AUROC changes by +0.007
(imputed at serving, +0.003); it stays only because the feature set was fixed before the ablation. The
backbone is 80% of the photo path's time; the gate's no-photo arm, given the same intercept, scores 0.741
(shipped minus it +0.021, −0.003 to +0.053) in 0.35 ms of model time, the product a next version should ship. The profile carries the system
(dropping it costs 0.122, Figure 4); the CGM block, with heart rate, adds +0.033 (exploratory, §3).

### 3. Evaluation and ablations

The 45 CGMacros participants are healthy, prediabetic and T2D adults (ages 18 to 69, 64% female), each
wearing a CGM for about ten days. 158 of the 481 development meals (32.8%) spiked, with per-person rates
from 0% to 68%: per-person calibration is the central problem. In partition v2 the 23 development target
people form five outer folds, with four inner folds choosing everything (fusion C, Platt, m, the
cut-offs); 14 T2D people only train; 8 people have been locked since the proposal. splits/reconciliation.md asserts
each of the 481 development meals is scored exactly once. One declared
caveat: fold 1 contains the five people the intermediate evaluation already scored, and the calibration
fix was chosen after a diagnosis made on those people, so the development estimate carries selection
pressure the locked eight do not. The locked eight (§6) are the only clean measurement.

Postprandial response is person-specific (Zeevi et al. 2015; Berry et al. 2020). The calibration fix: a
per-fold Platt map on inner out-of-fold risks plus a penalised per-person intercept b_p(t), fitted only
on the person's earlier meals and replayed in deployment order under a 2-hour label-availability gate.
Within every fold Platt AUROC equals raw; the pooled 0.714 mixes five fold maps. Figure 2 shows raw and shipped per fold.

The intercept's evidence is log-loss and every fold, not pooled AUROC, and most of the gain is its own:
with the same intercept, the fused system's margin over gradient boosting is not established. It works
between people, not within them: averaged per person, AUROC falls from 0.680 to 0.602 (19 of 22 people worse) as
the offset drifts with history. The gain is learning each person's base rate, which the system does 0.063 better than a
running spike rate alone. On
development it trades calibration (Platt alone is better) for resolution; on the locked eight it repairs
calibration, its offsets moving up toward the harder cohort, where the served Platt-only risk
under-predicts by 11 points. The registered ECE target was mis-specified for this size: a perfectly
calibrated model scores 0.052 on development. Discrimination is even across glycaemic status, but usefulness is not: the
intended users, people with prediabetes, get 10% of their meals cleared and are under-predicted (table). Per-person gaps shrank (rare spikers predicted 20.9% vs
observed 14.2%, was 29.3%). The deployed values were frozen from the all-development
replay by a pre-registered procedure (docs/final_design.md amendment): Platt (0.96, −0.20), m = 1.0,
cut-offs (0.15, 0.16). Like folds 1 and 5, whose bands have zero width, the replay chose a near-degenerate
abstention band; §4 reports what that costs.

### 4. Trust and safety

Decisions are priced: a missed spike costs 5× an unnecessary caution, and abstaining costs 2 if the meal
spikes and 0.5 if not. Both prices are declared assumptions (5:1 is a ~17% action threshold; Vickers &
Elkin 2006), swept in a pre-registered grid that re-chooses the cut-offs per setting. The development
verdict holds only at the declared prices: at half the abstain price abstention takes 62–76% of meals,
far over its cap; at twice it, savings fall to 3.1% and missed-among-cleared rises to 11.7%.

Of every 100 development meals the evaluated system (with history) clears 22.7, flags 63.6 and declines 13.7; of the
cleared, about 7 still spike. Only 17 people
had cleared meals, so the miss interval's upper end exceeds the cap; the 8 misses spread across 6 people. The frozen band (0.15, 0.16) is one point wide. On the locked cohort, where
risks ran higher, 139 of 153 meals landed above it: a dietitian would see nearly every meal flagged, and
always flagging is then the cheaper policy.
Without history the served point (last column) fails two targets on development and misses 7 of 32 clears
on the locked eight. The samples also lack onboarding priors: filled with the serving medians (development,
frozen coefficients standing in for the fold models'), AUROC drops to 0.637 and 9 of 55 cleared meals
spike, costlier than always flagging. Rule R13 (26 Nov, after the locked run) therefore refers such meals
instead of clearing them: no misses, 17.3% can't judge, about the cost of always flagging (−0.3%). Abstention
does not pay here. Deciding the 45 borderline development meals would have cost less (0.539 against 0.555
per answered meal): they spiked at 27%, above the 20% at which abstaining beats caution at the declared
prices, because risks just above the low cut are under-predicted. Abstaining on the meals closest to the
cut (Figure 7) keeps the miss rate among cleared meals at 5.6–8.4% from 0 to 30% abstention while clears
fall from 24% to 10% of answered meals. Abstention is established practice (Kompa et al. 2021), but here
it buys little.

Degradation is graded while inputs stay usable (blur) and turns into refusal at the designed thresholds
(the 20-minute gap rule, the contrast rule in near-darkness); past a rule every meal refuses, so the gap
and a misstated meal time are steps by design. Flips between low risk and caution, either direction, stay
below 1.8% at every level; with heart rate missing, 3.7%, the worst case. All stress runs use the
no-history per-fold pipelines. The 23 hostile inputs (timezone offsets, malformed JSON, binary CSVs,
broken photos, empty folders) give zero crashes; five refuse on a placeholder photo after parsing their CSV
stressor. R1-R5 and R8-R12 answer observed failures, R13 a measured one; tests enforce them.

The photo-drift rule detects only pictures the per-answer gate already refuses, and the outcome rule
fails as an alarm at this volume. The registered caution-share rule fired once, but part of that step is
the cut-off change at the dev/locked seam: with both streams at the deployed cut-offs it climbs without
alarming. The expected calibration alarm did not come. No rule caught the real shift cleanly.

Feedback (src/glimpse/feedback.py): after a meal, a validated record comes from the user (ate it all,
part of it, none; optionally the rise they saw) or from the CGM itself at +2 h (`cgm_auto`: the training
rule on raw readings, so near 50 mg/dL it can disagree with the training label); "did not eat it" yields no label. Records are stored; the folder command accepts them as an optional
history.csv, the app not yet. The same records
would drive the outcome monitor and queue for human review before any retraining. The planned dietitian consultation did not complete: its three questions (cap
acceptability, decision wording, the rising-baseline case) are committed, and the cost matrix stays a
declared assumption hedged by the sweep.

### 5. Explainability and failure analysis

Every class in the taxonomy was observed; none is hypothetical.

T5 is the genuine blind spot, and no single mechanism explains it. For row 847 every sizeable profile and trend
term except window rise pulled risk down; for row 1024 the priors pushed it up, outweighed by heart rate,
spread and mean. The four photo terms together pulled risk down on all four T5 meals (−0.28 to −0.51
logit), but equally on 82% of the cleared meals that did not spike (median −0.31): no worse than usual.
Four of the 8 false clears sit in fold 1, the fold under selection pressure. Two rules chosen after the outcome trade clears for misses: leads, not results.

Probe B was registered as an expected pass (a rising pre-meal trace raises risk on at least 90% of
meals) and failed: risk rose on only 8%. The label is baseline-relative, so a rising trace lowers
predicted risk; the registration encoded our naive physiology (docs/probes_s13.md). Exact SHAP for the frozen fusion (Figure 6; Platt and the intercept act
after it and are not attributed) ranks prior rise first, then
age, heart rate and prior spike rate. Attribution and ablation disagree for heart rate and for the four
photo terms (together 0.33 logit per meal against an unestablished gain): attribution measures in-sample
use, ablation out-of-sample loss. Heart rate is not redundant (R² 0.19 from the other inputs), so its
−0.21 coefficient is best read as in-sample signal that does not generalise.

Probe C, registered last, moved each profile input from its 10th to its 90th percentile: risk rises with
HbA1c, fasting glucose, prior spike rate and prior rise on every meal (median +0.06 to +0.12), which the
positive coefficients of a linear fusion guarantee; the finding is that this changes 11–17% of decisions. Filling the priors with the serving medians changes 11% of decisions and cuts
in-sample clears from 68 to 46 (outputs/probes/probe_c.json; §4's per-fold figures are 73 and 55).

Probe A: risk moves in the photo's direction on 100% of meals, which a linear fusion guarantees (a
wiring check, not learning), yet the extreme bottom↔top-decile plate swap leaves 68.5% of decisions unchanged (median swing 0.102), as pre-committed: the gate's verdict seen
mechanistically. R7 (rule ledger): the kNN check measures visual, not nutritional, familiarity. A local
lamb-and-rice dish (instructor case 11) scores 0.318 against the 0.614 cut
and gets a confident wrong carb bin; the sweep under a ≤2% false-flag budget shows no threshold fixes it.
The confirm-carbs tier is unwired, so the fix is a nutrition-aware embedding (future work).

### 6. Reproducibility

The 8 locked subjects were scored once, on 25 Nov, from the frozen artifacts (locked rows of the §3
tables), through the Pipeline.answer path the instructor command invoked at the tag. Decisions: 10 low risk,
139 caution, 4 can't judge; missed-among-cleared 2 of 10; cost reduction −5.7% against always caution.
The development estimate (0.762) lies above the locked interval: §3's selection pressure and the harder
cohort (42.0% of the 150 scored meals spiked vs 32.8%) both act here, inseparable at n=8. The intercept still beats raw on
the locked people at the point estimate on AUROC, Brier and log-loss, with paired intervals that include
zero. D7 also
pre-registered the photo gate on the locked eight as the primary test; that run scored only the shipped
pipeline, so the endpoint was never computed and cannot be now.

After the locked run, a fresh clone on the Mac under the final serving code (install, data, pinned
weights, prepare, selftest, evaluate, figures, both input suites) reproduced every number in the report
of that day exactly (record, plus a data-free rerun of the final state: outputs/finalize/). `just evaluate` reruns the development evaluation, then `just report-numbers`
recomputes the descriptive metrics, replays and figures from committed outputs (no dataset or weights);
`just reproduce-all` chains the compute studies (the README maps tables to recipes). Tests check the main
tables' cells against those files on every commit.

**AI-assisted development, verified.** Code and documents were built with an LLM assistant, as the
course expects, and the commits say so; the author set the goals and approved each pre-registration,
freeze and verdict before its run up to the tag; the overnight round's registrations were made by the
assistant under her standing instruction. The defence is the verification trail: a write-up stated n=397
where the output says 460, caught by the number-to-file check. After the tag `submission-v1` (25 Nov),
cold grades by AI agents given the rubric led to the dated round in docs/changelog.md; no frozen artifact
or locked number changed.

### 7. Limitations and conclusion

CGMacros is de-identified public PhysioNet data collected under the source study's ethics approval; no
new human data. Forty-five people from one study, repeated standardised meals, no local cuisine (R7). T2D participants are training-only, so no claim is made for them. Heart rate adds nothing here
and would be dropped in a next version; the app should take a person's history and priors,
and the product should become the sensor-first one the gate described. The label's baseline-relative
definition can diverge from a user's intuition (probe B). The errors fall on different people. A false clear, the costliest error, tells someone a spiking meal is
safe; 7 of the 8 development false clears fell on healthy people, while people with prediabetes, the
intended users, were rarely cleared at all. On a harder cohort the opposite fails: with 91% of locked
meals flagged, the nudge becomes noise users learn to ignore and a dietitian cannot triage. This is an
advisory prototype with no clinical validation, end-user test or
regulatory review, built so that a clinical study could evaluate it.

Medicine 2020, doi:10.1038/s41591-020-0934-0 · Vickers & Elkin, Med Decis Making 2006,
doi:10.1177/0272989X06295361 · Brazeau et al., Diabetes Res Clin Pract 2013 (carb counting, T1D),
doi:10.1016/j.diabres.2012.10.024 · Vasiloglou et al., Nutrients 2018 (GoCARB),
doi:10.3390/nu10060741 · Kompa, Snoek & Beam, npj Digital Medicine 2021,
doi:10.1038/s41746-020-00367-3 · CGMacros: Scientific Data 2025, doi:10.1038/s41597-025-05851-7,
and PhysioNet v1.0.0.

