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

    build_meal_table(cfg)
    ds = build_dataset(cfg)
    ds.save(Path("data/dataset"))
    logger.bind(stage="data").info("dataset saved to data/dataset.npz and data/dataset.json")
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
    sub.add_parser("build-meals", help="build and validate the labelled meal table")
    sub.add_parser("prepare", help="meal table + model-ready dataset (windows, profile inputs, labels)")
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
        "build-meals": _build_meals,
        "prepare": _prepare,
    }
    return commands[args.command](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
