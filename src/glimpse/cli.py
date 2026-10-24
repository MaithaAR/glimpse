"""The `glimpse` command. Every run loads and checks the settings, starts logging and fixes the seed first."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from glimpse import __version__
from glimpse.config import GlimpseConfig, load_config
from glimpse.utils.logging import setup_logging
from glimpse.utils.seed import set_seed


def _check_config(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Report the key settings so a user can see what the run will use."""
    log = logger.bind(stage="config")
    log.info("spike = rise >= {} mg/dL within {} min", cfg.label.spike_threshold_mgdl, cfg.label.horizon_min)
    log.info("target groups: {}; costs low_risk={} caution={} abstain={}", cfg.cohort.target_groups,
             cfg.costs.low_risk, cfg.costs.caution, cfg.costs.abstain)  # fmt: skip
    log.info("settings are valid")
    return 0


def _download_data(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Download CGMacros CSVs (and photos unless --no-photos) into the folder above cgmacros_root."""
    from glimpse.data.download import download_cgmacros

    download_cgmacros(Path(cfg.data.cgmacros_root).parent, with_photos=not args.no_photos)
    return 0


def _build_meals(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Build and validate the meal table; report the proposal's counts."""
    from glimpse.data.meals import build_meal_table

    build_meal_table(cfg)
    return 0


def _prepare(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Meal table -> model-ready dataset (windows, profile inputs, labels) saved to data/dataset.*."""
    from glimpse.data.dataset import build_dataset
    from glimpse.data.meals import build_meal_table
    from glimpse.data.normalise import Normaliser
    from glimpse.data.splits import make_splits, meal_rows, save_split_outputs

    build_meal_table(cfg)
    ds = build_dataset(cfg)
    ds.save(Path("data/dataset"))
    logger.bind(stage="data").info("dataset saved to data/dataset.npz and data/dataset.json")
    sp = make_splits(ds, cfg)
    save_split_outputs(ds, sp, cfg)
    tr = meal_rows(ds, set(sp.train), scored_only=False)  # scaling statistics from TRAINING people only
    Normaliser.fit(ds.glucose[tr], ds.heart_rate[tr], ds.profile[tr], ds.profile_names).save(
        Path("checkpoints/normaliser.json")
    )
    logger.bind(stage="data").info("splits in splits/dev_splits.json; normaliser in checkpoints/normaliser.json")
    from glimpse.data.quality import write_quality_outputs

    write_quality_outputs(ds, sp, cfg)
    return 0


def _params(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Parameter budget table -> outputs/param_budget.csv."""
    from glimpse.models.budget import write_budget

    write_budget(Path(cfg.paths.outputs) / "param_budget.csv")
    return 0


def _train(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Train a model on the prepared dataset and the committed splits."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.training.scratch import train_scratch

    ds, sp = MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json"))
    if args.model == "scratch":
        train_scratch(ds, sp, cfg)
    else:
        from glimpse.training.photo import train_photo

        train_photo(ds, sp, cfg)
    return 0


def _cv(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Cross-validated out-of-fold predictions of every model (the main estimate) -> outputs/cv/."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.training.cv import run_cv

    run_cv(MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json")), cfg)
    return 0


def _carbs_check(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Logged-carbs check: does adding the TRUE carbs to profile + trend raise AUROC enough to try photo carbs?"""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.training.fusion import run_carbs_check

    run_carbs_check(MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json")), cfg)
    return 0


def _photo_cv(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Photo carb estimates for every development meal from models that never saw that person (Step 4c-2)."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.training.photo_cv import run_photo_cv

    run_photo_cv(MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json")), cfg)
    return 0


def _fusion(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Profile + trend with each kind of meal information: no meal, photo, logged carbs, controls (Step 4c-3)."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.training.fusion import run_fusion

    run_fusion(MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json")), cfg)
    return 0


def _final(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """28-person fusion + thresholds frozen on the validation people (Step 5a); test people untouched."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.training.final import train_final

    train_final(MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json")), cfg)
    return 0


def _amend(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Freeze the amended decision cut-offs before the test is opened (Step 5b)."""
    from glimpse.data.splits import Splits
    from glimpse.run_evaluation import amend_thresholds

    amend_thresholds(Splits.load(Path("splits/dev_splits.json")), cfg)
    return 0


def _evaluate(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Main estimate, research questions, calibration, usefulness, fixed test (once), targets, figures."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.run_evaluation import run_evaluation

    run_evaluation(MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json")), cfg)
    return 0


def _demo(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Worked cases, can't-judge cases and response time with the frozen pipeline (Step 6a)."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.demo_run import run_demo

    run_demo(MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json")), cfg)
    return 0


def _predict(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """One meal from your own files -> the answer as JSON on screen."""
    import json
    import sys
    from datetime import datetime

    from glimpse.demo import Glimpse, MealInput, read_cgm_csv

    gt, gv, ht, hv = read_cgm_csv(args.cgm)
    nan = float("nan")
    profile = {"age": args.age, "bmi": args.bmi, "a1c_pct": args.a1c, "fasting_glucose_mgdl": args.fasting,
               "prior_spike_rate": nan if args.prior_rate is None else args.prior_rate,
               "prior_rise": nan if args.prior_rise is None else args.prior_rise}  # fmt: skip
    when = datetime.fromisoformat(args.meal_time) if args.meal_time else max(gt)
    a = Glimpse(cfg).predict(MealInput(meal_time=when, glucose_times=gt, glucose_mgdl=gv, hr_times=ht, hr_bpm=hv,
                                       profile=profile, photo=Path(args.photo) if args.photo else None))  # fmt: skip
    sys.stdout.write(json.dumps(a.to_dict(), indent=2) + "\n")
    return 0


def _selftest(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Check that every frozen part loads offline and the full path runs (used inside the container)."""
    from glimpse.demo_run import selftest

    selftest(cfg)
    return 0


def _download_weights(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Fetch and verify the pinned photo-model weights (needed once; everything else then runs offline)."""
    from glimpse.models.weights import download_weights

    download_weights(cfg.photo)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line, load + validate the config, set up logging and seeds, then run the sub-command."""
    p = argparse.ArgumentParser(prog="glimpse", description="GLIMPSE: pre-meal glucose-spike risk")
    p.add_argument("--version", action="version", version=f"glimpse {__version__}")
    p.add_argument("--config", default="configs/default.yaml", help="settings file (checked at startup)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("check-config", help="load and validate the settings, then report them")
    dl = sub.add_parser("download-data", help="download CGMacros CSVs and before-meal photos (resumable)")
    dl.add_argument("--no-photos", action="store_true", help="CSVs only")
    sub.add_parser("download-weights", help="download and verify the pinned photo-model weights")
    sub.add_parser("build-meals", help="build and validate the labelled meal table")
    sub.add_parser("params", help="count every model's parameters against the project limits")
    tr = sub.add_parser("train", help="train a model")
    tr.add_argument(
        "--model",
        choices=["scratch", "photo"],
        required=True,
        help="scratch = SensorCNN (from scratch); photo = PhotoCarbModel (pretrained, fine-tuned)",
    )
    sub.add_parser("cv", help="5-fold person cross-validation of the CNN and the comparison models")
    sub.add_parser("carbs-check", help="does adding the logged carbs to profile + trend raise AUROC? (Step 4c-1)")
    sub.add_parser("photo-cv", help="out-of-fold and nested photo carb estimates + baselines (Step 4c-2)")
    sub.add_parser("fusion", help="add photo / logged carbs to profile + trend, with controls (Step 4c-3)")
    sub.add_parser("final", help="28-person fusion and frozen decision thresholds (Step 5a)")
    sub.add_parser("amend-thresholds", help="freeze the amended decision cut-offs before the test is opened")
    sub.add_parser("evaluate", help="full evaluation into outputs/evaluation/ (scores the fixed test once)")
    sub.add_parser("demo", help="worked cases, can't-judge cases and response time (Step 6a)")
    pr = sub.add_parser("predict", help="answer one meal from your own photo, CGM CSV and profile")
    pr.add_argument("--cgm", required=True, help="CSV with timestamp, glucose_mgdl (and optionally hr_bpm)")
    pr.add_argument("--photo", help="before-meal photo (omit if none: the answer is then 'can't judge')")
    pr.add_argument("--meal-time", help="YYYY-MM-DDTHH:MM (default: last reading in the CSV)")
    for name in ("age", "bmi", "a1c", "fasting"):
        pr.add_argument(f"--{name}", type=float, required=True)
    pr.add_argument("--prior-rate", type=float, help="spike rate in your onboarding week (0-1), optional")
    pr.add_argument("--prior-rise", type=float, help="mean rise in your onboarding week (mg/dL), optional")
    sub.add_parser("selftest", help="check the frozen pipeline loads offline and runs (no dataset needed)")
    sub.add_parser("prepare", help="meal table, dataset, person splits and the training-only normaliser")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ValidationError as e:  # show every bad setting in plain words, then stop
        setup_logging()
        logger.bind(stage="config").error("invalid settings in {}:\n{}", args.config, e)
        return 2
    setup_logging(cfg.paths.logs)
    set_seed(cfg.seed)
    logger.bind(stage="cli").info("command={} config={}", args.command, args.config)
    commands = {
        "check-config": _check_config,
        "download-data": _download_data,
        "download-weights": _download_weights,
        "build-meals": _build_meals,
        "prepare": _prepare,
        "params": _params,
        "train": _train,
        "cv": _cv,
        "carbs-check": _carbs_check,
        "photo-cv": _photo_cv,
        "fusion": _fusion,
        "final": _final,
        "amend-thresholds": _amend,
        "evaluate": _evaluate,
        "demo": _demo,
        "predict": _predict,
        "selftest": _selftest,
    }
    return commands[args.command](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
