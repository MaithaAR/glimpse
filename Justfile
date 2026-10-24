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
photo-cv:
    uv run glimpse photo-cv

# fusion: profile + trend with no meal / photo / logged carbs / shuffled photo (+ post-hoc ridge arm); ~2 min
fusion:
    uv run glimpse fusion

# final 28-person pipeline: fusion + thresholds chosen on the validation people and frozen (~8 min; test untouched)
train-final:
    uv run glimpse final

# freeze the amended decision cut-offs (main = from the costs alone) before the test is opened; then commit them
amend-thresholds:
    uv run glimpse amend-thresholds

# full evaluation -> outputs/evaluation/ (run `just cv` first if the C grid changed; scores the fixed test once)
evaluate:
    uv run glimpse evaluate

# demo with the frozen pipeline: worked cases from the test people, two can't-judge cases, response time
demo:
    uv run glimpse demo

# answer one meal from your own files, e.g. just predict --cgm my_cgm.csv --photo lunch.jpg --age 45 --bmi 28 --a1c 5.9 --fasting 100
predict *ARGS:
    uv run glimpse predict {{ARGS}}

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
    rm -rf logs .pytest_cache .ruff_cache outputs/feedback outputs/demo/examples outputs/demo/test_meals.csv outputs/demo/blank_photo.jpg outputs/data outputs/quality
    find . -name "__pycache__" -type d -prune -exec rm -rf {} +
