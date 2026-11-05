# GLIMPSE task runner: short names for the commands used most often. Run `just` to list them.

# list the recipes
default:
    @just --list

# install Python 3.14 + locked libraries, download the data and the pinned photo weights, prepare the dataset
install:
    uv sync
    uv run glimpse download-data
    uv run glimpse download-weights
    uv run glimpse prepare

# rebuild the meal table and the model-ready dataset from the downloaded data
prepare:
    uv run glimpse prepare

# partition v2 (final stage): splits/dev_splits_v2.json + the asserted meal-level reconciliation
splits-v2:
    uv run glimpse splits-v2

# run the test suite
test:
    uv run pytest

# load and validate configs/default.yaml, start logging, set the seed
check-config:
    uv run glimpse check-config

# check code style and errors (no print(), sorted imports, line length)
lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

# lint, then test
check: lint test

# count every model's parameters against the limits (writes outputs/param_budget.csv)
params:
    uv run glimpse params

# train both models
train: train-scratch train-photo train-final

# train the from-scratch SensorCNN: epochs chosen on inner folds, then the 28-person model + checkpoint
train-scratch:
    uv run glimpse train --model scratch

# train the photo model: frozen ConvNeXt-Tiny + heads; last stage kept only if the inner folds show a gain
train-photo:
    uv run glimpse train --model photo

# cross-validated out-of-fold predictions of the CNN and every comparison model (main estimate) -> outputs/cv/
cv:
    uv run glimpse cv

# logged-carbs check: profile + trend with vs without the logged carbs, same folds; decides whether the photo CV runs
carbs-check:
    uv run glimpse carbs-check

# photo cross-validation: recipe re-chosen inside each outer fold; OOF + nested carb estimates (~1.5 h, resumable)
photo-cv *ARGS:
    uv run glimpse photo-cv {{ARGS}}

# fusion: profile + trend with no meal / photo / logged carbs / shuffled photo (+ post-hoc ridge arm); ~2 min
fusion:
    uv run glimpse fusion

# S6 ablations: D8 profile-excluded training + input-channel ablations (same folds and C as `fusion`)
ablate:
    uv run glimpse ablate

# offline W&B runs regenerated from the committed CSVs/JSONs (wandb sync outputs/wandb/... uploads them)
wandb-export:
    uv run glimpse wandb-export

# S10: unfamiliar-photo check against the 14 real phone photos; --apply writes the re-chosen cut (R7)
ood *ARGS:
    uv run glimpse ood {{ARGS}}

# S9: decision-flip rates under the four pre-registered perturbations (CGM dropout/gap, photo blur/dark)
robustness:
    uv run glimpse robustness

probes:
    uv run glimpse probes

efficiency:
    uv run glimpse efficiency

monitoring:
    uv run glimpse monitoring

stress:
    uv run glimpse predict-folder samples/stress outputs/stress_run

# final 28-person pipeline: fusion + thresholds chosen on the validation people and frozen (~8 min; test untouched)
train-final:
    uv run glimpse final

# cut-off amendment: being rebuilt on partition v2 (S5, D4); the command says so and exits
amend-thresholds:
    uv run glimpse amend-thresholds

# development evaluation on partition v2 (D2): every risk comes from Pipeline.answer, one Pipeline per fold
evaluate:
    uv run glimpse evaluate

# seed spread of the final photo model: 2 more seeds scored on the validation people (~10 min; nothing deployed changes)
photo-seeds:
    uv run glimpse photo-seeds

# recalibration: being rebuilt on partition v2 (S5, D3/D4); the command says so and exits
recalibrate:
    uv run glimpse recalibrate

# demo with the frozen pipeline: worked cases from the test people, two can't-judge cases, response time
demo:
    uv run glimpse demo

# answer one meal from your own files, e.g. just predict --cgm my_cgm.csv --photo lunch.jpg --age 45 --bmi 28 --a1c 5.9 --fasting 100
predict *ARGS:
    uv run glimpse predict {{ARGS}}

# answer every meal folder under IN (the instructor's entry point), e.g. just predict-folder samples/instructor outputs/folder_run
predict-folder IN OUT="outputs/folder_run":
    uv run glimpse predict-folder {{IN}} {{OUT}}

# offline check that every frozen part loads and the full path runs (no dataset needed)
selftest:
    uv run glimpse selftest

# Linux container (needs Docker): build, offline self-test, and the demo with your own CGMacros copy mounted read-only
docker-build:
    docker build -t glimpse .

docker-selftest:
    docker run --rm --network none glimpse selftest

docker-demo:
    docker run --rm --network none -v "$PWD/data/cgmacros:/app/data/cgmacros:ro" -v "$PWD/data/dataset.json:/app/data/dataset.json:ro" -v "$PWD/data/dataset.npz:/app/data/dataset.npz:ro" glimpse demo

# the GLIMPSE screen in the browser (http://127.0.0.1:7860); needs the optional `app` extra (Gradio)
app:
    uv run --extra app python -m glimpse.app

# remove generated files (keeps downloaded data, the locked environment, checkpoints and committed results)
clean:
    rm -rf logs .pytest_cache .ruff_cache outputs/feedback outputs/demo/examples outputs/demo/test_meals.csv outputs/demo/blank_photo.jpg
    find . -name "__pycache__" -type d -prune -exec rm -rf {} +

wordcount:
    uv run python -c "import re; t=open('docs/final_report.md').read(); print(len(re.sub(r'<!--.*?-->','',t,flags=re.S).split()), 'words (budget 3,000)')"
