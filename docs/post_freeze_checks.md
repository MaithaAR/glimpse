# Post-freeze exploratory checks, 27 Nov 2026

Written on the day of the deadline, before the runs below; plan, code and results share one commit, so the
history cannot show the order. These are not pre-registered studies: the plan and
the runs are the same day, so they are reported as exploratory. The served model (v1), its cut-offs, the
frozen files and every locked number are unchanged. `just post-freeze` reruns all of it (about 20 minutes with
the data and weights); results go to outputs/post_freeze/post_freeze.json.

Why: the course's lecture tests include several the submission had not run, and the report did not show what
the decision layer does when it refers more meals.

| check | what is measured | what would count as a problem |
|---|---|---|
| JPEG compression | the served photo model on 200 development photos (seed 20260927), re-encoded at quality 90, 50, 20, 10: carb error, shift from the clean estimate, share keeping the same carb range | a large shift at quality 50 or above, which phones produce |
| crop augmentation | heads on frozen features (the v3 recipe as fixed in R5), trained with crops keeping 80% of the photo, with no crop, and with no augmentation; out-of-person carb MAE, 3 seeds | the crop making the estimate worse: a crop changes how much food is visible, and the label is grams |
| embedding map | t-SNE of the frozen photo summaries; share of each photo's 10 nearest neighbours with the same carb range and with the same person, against chance | neighbours sharing the person far more than the carb range |
| photo estimate ladder | fine-tuned, ridge on frozen features, and their average on the log scale, from the committed out-of-fold files | none; it states which rung is best |
| decision layer | v1's development answers at the evaluated cut-offs: a refer band of 0.02, 0.05, 0.08 around the cut, and referring the 10, 20, 30% of meals closest to it; person-bootstrap intervals | referring more meals not lowering the missed share among cleared meals |

Not done, and why: a zero-shot baseline (needs CLIP weights this offline build cannot fetch), a human baseline
(no rater), a full fine-tune of all 27.8M weights (hours on two CPUs), energy per answer (not measured, so no
figure is given).

## Results (27 Nov; outputs/post_freeze/post_freeze.json)

| check | result | reading |
|---|---|---|
| JPEG compression, 200 photos (in-sample for the served checkpoint) | carb MAE 14.9 g clean; quality 90: median shift 0.4 g, 98% keep their carb range; quality 50: 1.8 g, 96%; quality 20: 4.2 g, 92.5%; quality 10: 7.4 g, 84% | stable at the qualities phones produce; degrades only at quality 20 and below |
| crop augmentation, heads on frozen features, 3 seeds, 460 scored photos | carb MAE 26.9 g with crops keeping 80% (26.8 to 27.3), 26.9 g without the crop (26.4 to 27.5), 30.2 g with no augmentation (29.5 to 30.7) | the crop neither helps nor hurts the grams estimate; flips and light jitter account for the gain. The portion concern is not borne out, and the model does not read portion size anyway (probe E) |
| embedding map, 1,221 photos | 10 nearest neighbours: 49% share the carb range (chance 28%), 14% share the person (chance 3%); among neighbours from other people, 49% share the carb range | the summaries group by meal content, and that grouping holds across people |
| photo estimate ladder, 460 scored photos | fine-tuned 22.6 g (Spearman 0.53), ridge on frozen features 20.4 g (0.56), their average on the log scale 20.1 g (0.57) | at 1,221 photos the frozen features beat the fine-tune on grams; averaging the two is the best estimate. Not served: v1's files are frozen |
| decision layer, 460 answered development meals at the evaluated cut-offs | no band: 23.7% cleared, 7.3% of clears missed (2.6 to 14.4%), cost reduction 19.7%; refer within 0.02 of the cut: 17.2%, 6.3%, 18.3%; within 0.05: 13.3%, 4.9% (0.0 to 11.6%), 18.3%; within 0.08: 7.0%, 6.2%, 12.6%; referring the closest 10 / 20 / 30%: missed 8.4 / 6.2 / 6.7% | a wider refer band lowers clears faster than it lowers misses, and the intervals overlap throughout. The narrow band is not what limits the product; discrimination is |

The 19.7% cost reduction here is on the 460 answered meals with no refer band; the report's 14.4% is on all 481
meals with each fold's own band and counts the 21 photo-less meals as referred.
