# S13 — explainability probes (pre-registered 1 Nov 2026, before any run)

Two counterfactual probes, every development meal answered by the SAME per-fold Pipelines the
evaluation uses (fold fusion, Platt, per-fold cut-offs; no history, so differences come only from the
manipulated input). Settings and expected outcomes below are fixed before the first run; the results
section is appended afterwards and nothing above it changes.

## Probe A — photo-swap counterfactual

For each fold: among the fold's held meals with a photo estimate, take the meals at the 10th and 90th
percentile of estimated carbs (the fold's low-carb and high-carb reference plates). Re-answer every
answered-with-photo held meal twice, with its photo estimate replaced by each reference (sensors and
profile untouched). Record risk_low, risk_high, base decision and swapped decisions.

Expected outcomes (committed):
- A1 (direction): risk_high >= risk_low for at least 90% of meals — the fusion uses the photo arm in
  the causally right direction.
- A2 (PRE-COMMITTED EXPECTED FAIL): the decision stays unchanged for MOST meals (>60%) even under this
  extreme bottom-decile <-> top-decile swap. We expect this "failure" of the photo to move decisions:
  it is the S8 gate finding (+0.025 AUROC, CI crossing 0) seen mechanistically — the photo nudges the
  risk, the trend and profile arms decide.

## Probe B — flat-vs-rising CGM

For each answered held meal, two CGM counterfactuals on the same photo and profile:
- flat: every glucose value in the 2-hour pre-meal window replaced by the window's own mean;
- rising: values in the final 45 minutes before the meal get a linear ramp added, 0 at minute -45 up
  to +40 mg/dL at the meal (times untouched; all readings remain plausible mg/dL).

Expected outcomes (committed):
- B1 (direction): risk_rising > risk_flat for at least 90% of meals — the trend arm responds to an
  incoming rise.
- B2 (magnitude, descriptive): the median risk movement under Probe B is reported next to Probe A's
  median |risk_high - risk_low|; we expect the CGM manipulation to move risk at least as much as the
  extreme photo swap, consistent with the gate. Reported, not pass/fail.

Report use: section 5 (explainability & failure), one small table + two sentences; A2 is quoted as the
designed expected-fail.

## Results (run 1 Nov 2026, 23:56; outputs/probes/probes.json; n=460 answered meals; n corrected 7 Nov — the first write-up said 397 without reading it from the output)

Probe A: direction ok 100% of meals (A1 PASS); decision unchanged 68.5% (A2 expected-fail CONFIRMED);
median |risk_high - risk_low| = 0.102. The fusion uses the photo in the right direction on every single
meal, and even the bottom-decile <-> top-decile plate swap leaves two thirds of decisions where they
were: the photo nudges, the trend and profile arms decide -- the S8 gate, mechanistically.

Probe B: risk_rising > risk_flat on only 8% of meals (B1 FAIL as registered); median move -0.047;
median |move| 0.049 vs probe A's 0.102.

B1 is a genuine failed pre-registration, and the failure is OURS, not the model's. The label is
rise = max(post-meal glucose) - mean(pre-meal baseline window): a spike is measured RELATIVE to the
pre-meal baseline. The +40 mg/dL ramp raises that baseline, so the identical post-meal curve clears
+50 less often, and a model that has learned the label's definition should -- and does -- lower the
risk for a trace rising into the meal. Our registered direction encoded the naive physiological
intuition ("already rising = more danger") instead of the label's arithmetic. The marginal correlation
of pre-meal slope with the spike label is weakly positive (+0.06 slope60, +0.10 slope20), but the
counterfactual holds everything else fixed and moves last reading, window rise and slope together --
the conditional direction, which is negative. Report use: section 5 quotes B1 verbatim as the probe
that caught the authors' wrong intuition while confirming the model respects the baseline-relative
label; no retuning follows from it.

## Probe C, profile and onboarding priors (registered 26 Nov 2026, before its run)

Why: the profile carries the system (dropping it costs 0.122 AUROC) and prior rise ranks first in the
in-sample SHAP, yet no counterfactual has tested the profile. Written after every other result was known,
so it is descriptive. Method: the frozen fusion on the 460 answered development meals' standardised inputs
(outputs/figures/shap_features_z.csv), Platt only, deployed cut-offs (0.15, 0.16). For each of HbA1c,
fasting glucose, prior spike rate and prior rise, set that one input to its 10th and then its 90th
percentile over the 460 meals, everything else unchanged. Report per input: the share of meals whose risk
rises from 10th to 90th, the median risk change, and the share of meals whose decision differs between the
two settings. A fifth case sets both priors to the serving imputer's medians (the instructor's path).
Expected, from physiology: risk rises with each of the four inputs on at least 90% of meals. Because the
fusion is linear the share is 0% or 100% per input; the direction is the test. Reported as measured in
outputs/probes/probe_c.json (`glimpse probe-c`).
