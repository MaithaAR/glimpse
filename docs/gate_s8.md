# Week-4 gate (S8): pre-registered decision rule and both verdict paragraphs

Committed 1 Nov 2026, BEFORE any fusion-arm run on partition v2. After the S6 runs, the matching
paragraph ships into the report verbatim; the other is deleted; nothing is retuned after reading the
result. The calibration result (personal AUROC 0.762, both by-person gaps halved) is evidence about
fix 1 and survives either verdict; it must not blur this gate.

## Decision rule (fixed now)

Primary endpoint (as pre-registered in the proposal, D7): the photo arm's AUROC minus the
profile + trend arm's AUROC, on partition v2 development folds, person-clustered bootstrap 95%
interval, photo-model seeds as in the committed photo CV.

- Lower bound of the interval > 0  ->  paragraph A ships.
- Lower bound <= 0 (including a positive point estimate)  ->  paragraph B ships verbatim.

Secondary endpoint (added at the final stage and labelled as such, D7): paired per-meal log-loss
difference, photo vs no-meal arm, clustered by person, one-sided at 5%. Reported under either verdict;
it cannot overturn the primary. No third branch exists, and no rescue ablation follows a negative gate.

## Paragraph A (gain established)

On the development folds the photo arm improved AUROC over profile + trend by {GAIN} (95% CI {LO} to
{HI}, person-clustered); the paired log-loss difference against the no-meal arm was {LL} ({P},
one-sided). The photo's contribution is established on this cohort at development level; the locked
test, scored once, reports the same endpoint on unseen people. The per-person calibration gain
(fix 1) is a separate result and holds in either arm.

## Paragraph B (gain not established)

On the development folds the photo arm's AUROC gain over profile + trend was {GAIN} (95% CI {LO} to
{HI}, person-clustered): the interval includes zero, so the photo's incremental contribution is not
established on this cohort. The product story is therefore sensor + confirm-carbs: the pre-meal answer
rests on the glucose trend and profile, and the photo head is retained as the carb-entry assist inside
the confirm-carbs tier, where an unfamiliar or unreadable photo already routes today. The per-person
calibration gain (fix 1) is independent of this verdict and stands. We report the secondary log-loss
endpoint for completeness ({LL}, {P}); it does not change the verdict.
