# Photo-model probes and the int8 study: pre-registration (26 Nov 2026, before either run)

Written by the assistant after the graders noted that the ConvNeXt, which holds almost all the parameters,
was never probed at the image level and that int8 was timed without a quality side. Both use the frozen
photo model (checkpoints/photo_convnext_final_v2_e14.pt) on the 1,221 development photos it was trained on
(in-sample, so these describe what the model does, not how well it generalises) and on the 14 real phone
photos it never saw (samples/local_photos). Output: outputs/probes/photo_probes.json, outputs/efficiency/int8.json.

## Probe D: where the photo model looks
For each photo, an occlusion of the central region (a grey square covering the middle 50% of the image
area) and an occlusion of equal area at the border (a grey frame). Measured: the change in predicted
log(1+grams) and whether the carb range changes.
Hypothesis: the model reads the food, which sits mostly in the middle of these photos, so the central
occlusion moves the grams more than the border one on at least 60% of photos.
Hypothesis rejected if the share is below 60%; reported as measured either way.

## Probe E: portion size
The same photo zoomed in (the central 60% of each side, resized back to full size), so the food covers
more of the frame. Measured: the share of photos whose predicted grams rise, and the median change.
Expectation: grams rise on more than half the photos if apparent size drives the estimate. Reported as
measured.

## int8
The frozen model with dynamic int8 quantisation of its linear layers, against fp32, on every development
photo: carb-range agreement, the median absolute difference in grams, the carb MAE against logged carbs
for both (in-sample), and the median time per photo. For v1 (photo fusion), the decision agreement of the
frozen v1 fusion (Platt only, frozen cut-offs) with fp32 against int8 photo inputs, on the meals with a
photo. v2's risk does not use the photo, so int8 changes only its carb assist. int8 is not shipped.
