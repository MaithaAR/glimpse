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
train: train-scratch train-photo

# train the from-scratch SensorCNN: epochs chosen on inner folds, then the 28-person model + checkpoint
train-scratch:
    uv run glimpse train --model scratch

# train the photo model: frozen ConvNeXt-Tiny + heads; last stage kept only if the inner folds show a gain
train-photo:
    uv run glimpse train --model photo

# cross-validated out-of-fold predictions of the CNN and every comparison model (main estimate) -> outputs/cv/
cv:
    uv run glimpse cv

# evaluate models and baselines, writing results to outputs/ (built in Step 5)
evaluate:
    @echo "Not built yet: evaluation is added in Step 5." && exit 1

# end-to-end demo on the sample inputs (built in Step 6)
demo:
    @echo "Not built yet: the demo is added in Step 6." && exit 1

# remove generated files (keeps downloaded data and the locked environment)
clean:
    rm -rf outputs logs .pytest_cache .ruff_cache
    find . -name "__pycache__" -type d -prune -exec rm -rf {} +
