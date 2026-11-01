# Week-4 gate (S8): pre-registered decision rule and both verdict paragraphs

Committed 1 Nov 2026, BEFORE any fusion-arm run on partition v2. After the S6 runs, the matching
paragraph ships into the report verbatim; the other is deleted; nothing is retuned after reading the
result.

**VERDICT (run 1 Nov 2026, 23:11, commit f5e9081): primary gain +0.025, 95% CI (-0.010, +0.061) --
lower bound <= 0 -> PARAGRAPH B SHIPS.** Secondary: paired log-loss +0.016 (95% CI -0.007 to +0.038),
one-sided p = 0.094 -- agrees, not significant. Supporting observations recorded for the report, not
as verdict-changers: the shuffled-photo control erases the gain (-0.010), and RQ1 finds the photo
estimate non-inferior to the user's true logged carbs (+0.002, 95% CI -0.027 to +0.027) -- the
binding constraint is the value of meal information on this cohort, not the photo model's quality. The calibration result (personal AUROC 0.762, both by-person gaps halved) is evidence about
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

## Paragraph A (gain established) -- DELETED per the rule; see git history

## Paragraph B (gain not established)

On the development folds the photo arm's AUROC gain over profile + trend was +0.025 (95% CI -0.010 to +0.061, person-clustered): the interval includes zero, so the photo's incremental contribution is not
established on this cohort. The product story is therefore sensor + confirm-carbs: the pre-meal answer
rests on the glucose trend and profile, and the photo head is retained as the carb-entry assist inside
the confirm-carbs tier, where an unfamiliar or unreadable photo already routes today. The per-person
calibration gain (fix 1) is independent of this verdict and stands. We report the secondary log-loss
endpoint for completeness (+0.016, one-sided p = 0.094); it does not change the verdict.
