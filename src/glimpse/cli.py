"""Command-line entry point: `uv run glimpse <command>` (all recipes in the Justfile call this)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from glimpse.config import load_config
from glimpse.utils.logging import setup_logging
from glimpse.utils.seed import set_seed


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run the end-to-end pipeline on one sample folder and print the validated RiskCard as JSON."""
    from glimpse.pipeline.predict import predict

    card = predict(args.input, args.cfg)
    sys.stdout.write(card.model_dump_json(indent=2) + "\n")
    out = Path(args.cfg.paths.outputs) / "demo"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(args.input).name}.json"
    path.write_text(card.model_dump_json(indent=2))
    logger.bind(stage="demo").info("risk card written to {}", path)
    return 0


def _cmd_download_data(args: argparse.Namespace) -> int:
    """Download CGMacros CSVs and photos (resumable)."""
    from glimpse.data.download import download_cgmacros

    download_cgmacros(Path(args.cfg.data.cgmacros_root).parent, with_photos=not args.no_photos)
    return 0


def _cmd_download_weights(args: argparse.Namespace) -> int:
    """Download pretrained weights into the local cache."""
    from glimpse.data.download import download_weights

    download_weights(Path(args.cfg.paths.weights_cache))
    return 0


def _cmd_build_meals(args: argparse.Namespace) -> int:
    """Build the labelled meal table and save it as parquet."""
    from glimpse.data.meals import add_personal_prior, build_meals

    meals = add_personal_prior(build_meals(args.cfg), args.cfg)
    path = Path(args.cfg.paths.meals_table)
    path.parent.mkdir(parents=True, exist_ok=True)
    meals.to_parquet(path)
    logger.bind(stage="data").info("meal table written to {} ({} rows)", path, len(meals))
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    """Build the validated dataset, the person-grouped splits, the training-only normaliser and data reports."""
    from glimpse.data.dataset import build_dataset
    from glimpse.data.normalise import Normaliser
    from glimpse.data.quality import write_reports
    from glimpse.data.splits import make_splits

    cfg = args.cfg
    ds = build_dataset(cfg)
    ds.save(Path(cfg.paths.dataset))
    sp = make_splits(ds, cfg)
    sp.save(Path(cfg.paths.splits))
    train = [i for i, s in enumerate(ds.samples) if s.person_id in set(sp.train)]
    norm = Normaliser.fit(ds.glucose[train], ds.heart_rate[train], ds.tabular[train], ds.tabular_names)
    norm.save(Path(cfg.paths.normaliser))
    write_reports(ds, sp, Path(cfg.data.cgmacros_root), Path(cfg.paths.outputs) / "data")
    logger.bind(stage="prepare").info("dataset, splits, normaliser and data reports written")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train the requested model on the prepared dataset and splits."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits

    cfg = args.cfg
    ds = MealDataset.load(Path(cfg.paths.dataset))
    sp = Splits.load(Path(cfg.paths.splits))
    if args.model == "scratch":
        from glimpse.training.scratch import train_scratch

        train_scratch(ds, sp, cfg)
        return 0
    return _cmd_not_ready(args)


def _cmd_params(args: argparse.Namespace) -> int:
    """Write the programmatic parameter-budget table to outputs/."""
    from glimpse.models.budget import write_budget

    write_budget(Path(args.cfg.paths.outputs) / "param_budget.csv")
    return 0


def _cmd_not_ready(args: argparse.Namespace) -> int:
    """Placeholder for milestones not built yet."""
    logger.bind(stage=args.command).error("'{}' is not implemented yet in this milestone", args.command)
    return 2


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load and validate the config, set up logging and seeds, then dispatch."""
    p = argparse.ArgumentParser(prog="glimpse", description="GLIMPSE pre-meal glucose-spike risk")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config (validated at startup)")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="run the end-to-end pipeline on a sample folder")
    d.add_argument("--input", required=True, help="folder with meal.json, cgm.csv and an optional photo")
    d.set_defaults(func=_cmd_demo)
    dd = sub.add_parser("download-data", help="download CGMacros CSVs and photos")
    dd.add_argument("--no-photos", action="store_true")
    dd.set_defaults(func=_cmd_download_data)
    sub.add_parser("download-weights", help="download pretrained weights").set_defaults(func=_cmd_download_weights)
    sub.add_parser("build-meals", help="build the labelled meal table").set_defaults(func=_cmd_build_meals)
    sub.add_parser("prepare", help="dataset, splits, normaliser, data reports").set_defaults(func=_cmd_prepare)
    sub.add_parser("params", help="parameter budget table").set_defaults(func=_cmd_params)
    t = sub.add_parser("train", help="train a model")
    t.add_argument("--model", choices=["scratch", "finetune"], required=True)
    t.set_defaults(func=_cmd_train)
    sub.add_parser("evaluate", help="evaluate models and baselines").set_defaults(func=_cmd_not_ready)
    for sp in sub.choices.values():  # allow --config after the sub-command too (as the Justfile does)
        sp.add_argument("--config", default=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.cfg = load_config(args.config)
    setup_logging(args.cfg.paths.logs)
    set_seed(args.cfg.seed)
    logger.bind(stage="cli").info("command={} config={}", args.command, args.config)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
