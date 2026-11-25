# README section kept for the record (moved here 26 Nov)

## Promised in the proposal vs delivered (as of the intermediate; kept for the record; the final reconciliation of every commitment is in docs/evaluation_protocol.md)
| Proposal | Now | Why |
|---|---|---|
| EfficientNet-B0 photo model | ConvNeXt-Tiny + carb-bin head | instructor feedback |
| Fusion MLP with quantiles + monotone carbs | logistic regression | feedback ("overkill"), small data |
| DINOv2 unfamiliar-photo check | nearest-neighbour check in the photo model's features | no extra model; DINOv2 comparison planned for the final |
| Calibration + cost-based cut-offs | cut-offs from the costs (frozen before the test); recalibration planned | test calibration shows over-prediction |
| Conformal rise intervals, W&B, Emirati photo set, user test | final submission | not required at this stage |
| Gradio app in Docker | `just app` (Gradio) + Dockerfile (CLI demo and self-test) | done |

Changes made after seeing results (C grid, ridge arm, carb-bin check, cut-off amendment) are listed and
labelled in [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md).

