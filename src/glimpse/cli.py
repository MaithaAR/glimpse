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
        from glimpse.training.train_photo import train_photo

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
    """The logged-carbs gate was answered at the intermediate stage; its result stands as submitted."""
    logger.bind(stage="fusion").error(
        "the logged-carbs check (Step 4c-1) was answered at the intermediate stage and its result stands "
        "(commit 983dbbb); it is not rerun on partition v2."
    )
    return 2


def _photo_cv(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Photo carb estimates for every development meal from models that never saw that person (partition v2)."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import SplitsV2
    from glimpse.training.photo_cv import run_photo_cv

    sp = SplitsV2.load(Path("splits/dev_splits_v2.json"))
    run_photo_cv(MealDataset.load(Path("data/dataset")), sp, cfg, only_fold=args.fold)
    return 0


def _fusion(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Profile + trend with each kind of meal information, on partition v2 (the gate's arm comparison)."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import SplitsV2
    from glimpse.training.fusion import run_fusion

    run_fusion(MealDataset.load(Path("data/dataset")), SplitsV2.load(Path("splits/dev_splits_v2.json")), cfg)
    return 0


def _final(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """28-person fusion + thresholds frozen on the validation people (Step 5a); test people untouched."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.training.final import train_final

    train_final(MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json")), cfg)
    return 0


def _amend(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Decision cut-offs are being rebuilt on partition v2 (S5, design D4)."""
    logger.bind(stage="thresholds").error(
        "the cut-off amendment is being rebuilt on partition v2 (S5, design D4: minimum expected cost on "
        "recalibrated inner-fold risk). The intermediate-stage amendment is commit 983dbbb."
    )
    return 2


def _photo_seeds(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Seed spread of the final photo model on the validation people (the deployed checkpoint is untouched)."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.training.train_photo import photo_seed_check

    photo_seed_check(MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json")), cfg)
    return 0


def _recalibrate(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Recalibration is being rebuilt on partition v2 (S5, designs D3/D4)."""
    logger.bind(stage="recalibrate").error(
        "recalibration is being rebuilt on partition v2 (S5: Platt on inner-fold out-of-fold risk, then the "
        "per-person expanding-window intercept). The intermediate-stage version is commit 983dbbb."
    )
    return 2


def _evaluate(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Development evaluation on partition v2: every risk from Pipeline.answer, one Pipeline per fold (D2)."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import SplitsV2
    from glimpse.evaluation.run import evaluate_dev

    evaluate_dev(MealDataset.load(Path("data/dataset")), SplitsV2.load(Path("splits/dev_splits_v2.json")), cfg)
    return 0


def _demo(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Worked cases, can't-judge cases and response time with the frozen pipeline (Step 6a)."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits
    from glimpse.demo_cases import run_demo

    run_demo(MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json")), cfg)
    return 0


def _predict(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """One meal from your own files -> the validated answer as JSON on screen (bad input -> 'can't judge')."""
    import sys

    from glimpse.serve import Pipeline, answer_from_files

    profile = {"age": args.age, "bmi": args.bmi, "a1c_pct": args.a1c, "fasting_glucose_mgdl": args.fasting,
               "prior_spike_rate": args.prior_rate, "prior_rise": args.prior_rise}  # fmt: skip
    a = answer_from_files(Pipeline(cfg), args.cgm, args.photo, args.meal_time, profile)
    sys.stdout.write(a.model_dump_json(indent=2) + "\n")
    return 0


def _predict_folder(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Every meal folder under IN -> OUT/results.jsonl + OUT/summary.md (bad meals become per-item entries)."""
    from glimpse.serve import Pipeline
    from glimpse.serve.folder import predict_folder

    predict_folder(Pipeline(cfg), args.in_dir, args.out_dir)
    return 0


def _robustness(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """S9: decision-flip rates under the four pre-registered input perturbations."""
    from pathlib import Path

    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import SplitsV2
    from glimpse.studies.robustness import run_robustness

    run_robustness(MealDataset.load(Path("data/dataset")), SplitsV2.load(Path("splits/dev_splits_v2.json")), cfg)
    return 0


def _timing(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Pre-registered meal-time shift robustness (docs/final_design.md, 7 Nov)."""
    from pathlib import Path

    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import SplitsV2
    from glimpse.studies.robustness import run_timing

    run_timing(MealDataset.load(Path("data/dataset")), SplitsV2.load(Path("splits/dev_splits_v2.json")), cfg)
    return 0


def _figures(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Render the report figures from committed evidence files (pre-commitment: nothing retuned)."""
    from glimpse.figures import run_figures

    run_figures(cfg)
    return 0


def _finalize(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Regenerate the serving artifacts per the pre-registered 5 Nov amendment (docs/final_design.md)."""
    from pathlib import Path

    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import SplitsV2
    from glimpse.training.finalize import finalize

    finalize(MealDataset.load(Path("data/dataset")), SplitsV2.load(Path("splits/dev_splits_v2.json")), cfg)
    return 0


def _monitoring(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """S11: draw the drift baseline the deployed monitor would redraw weekly (docs/monitoring.md)."""
    from glimpse.studies.monitoring import run_monitoring

    run_monitoring(cfg)
    return 0


def _efficiency(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """S12: parameters vs the caps, serving latency and artifact sizes of the deployed pipeline."""
    from glimpse.studies.efficiency import run_efficiency

    run_efficiency(cfg)
    return 0


def _probes(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """S13: photo-swap and flat-vs-rising CGM counterfactuals (pre-registered in docs/probes_s13.md)."""
    from pathlib import Path

    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import SplitsV2
    from glimpse.studies.probes import run_probes

    run_probes(MealDataset.load(Path("data/dataset")), SplitsV2.load(Path("splits/dev_splits_v2.json")), cfg)
    return 0


def _ood(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """S10: score the real phone photos under the unfamiliar-photo check and re-choose the cut (rule R7)."""
    from glimpse.studies.ood import run_ood

    run_ood(cfg, apply=args.apply)
    return 0


def _wandb_export(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Replay every history CSV and results summary into offline W&B runs under outputs/wandb/."""
    from glimpse.wandb_export import export_wandb

    export_wandb(cfg.paths.outputs)
    return 0


def _ablate(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """S6 ablations: D8 profile-excluded split and input-channel ablations (needs `just fusion` outputs)."""
    from pathlib import Path

    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import SplitsV2
    from glimpse.studies.ablations import run_channel_ablation, run_profile_excluded

    ds = MealDataset.load(Path("data/dataset"))
    sp = SplitsV2.load(Path("splits/dev_splits_v2.json"))
    run_profile_excluded(ds, sp, cfg)
    run_channel_ablation(ds, sp, cfg)
    return 0


def _splits_v2(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Partition v2 (design D1): dev folds + training-only T2D + locked, with the asserted reconciliation."""
    from pathlib import Path

    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits, make_splits_v2, reconciliation_v2

    ds = MealDataset.load(Path("data/dataset"))
    sp = make_splits_v2(ds, cfg)
    sp.save(Path("splits/dev_splits_v2.json"))
    Path("splits/reconciliation.md").write_text(reconciliation_v2(ds, Splits.load(Path("splits/dev_splits.json")), sp))
    logger.bind(stage="splits").info("splits/dev_splits_v2.json and splits/reconciliation.md written")
    return 0


def _selftest(cfg: GlimpseConfig, args: argparse.Namespace) -> int:
    """Check that every frozen part loads offline and the full path runs (used inside the container)."""
    from glimpse.demo_cases import selftest

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
    pcv = sub.add_parser("photo-cv", help="out-of-fold and nested photo carb estimates + baselines (partition v2)")
    pcv.add_argument("--fold", type=int, help="run this one fold only (dry run); default: every fold")
    sub.add_parser("fusion", help="add photo / logged carbs to profile + trend, with controls (Step 4c-3)")
    sub.add_parser("final", help="28-person fusion and frozen decision thresholds (Step 5a)")
    sub.add_parser("amend-thresholds", help="freeze the amended decision cut-offs before the test is opened")
    sub.add_parser("evaluate", help="dev evaluation on partition v2: every risk from Pipeline.answer per fold")
    sub.add_parser("photo-seeds", help="retrain the final photo model with 2 more seeds; validation spread only")
    sub.add_parser(
        "recalibrate", help="validation-chosen cut-offs on recalibrated risk, checked on CV (no test labels)"
    )
    sub.add_parser("demo", help="worked cases, can't-judge cases and response time (Step 6a)")
    pr = sub.add_parser("predict", help="answer one meal from your own photo, CGM CSV and profile")
    pr.add_argument("--cgm", required=True, help="CSV with timestamp, glucose_mgdl (and optionally hr_bpm)")
    pr.add_argument("--photo", help="before-meal photo (omit if none: the answer is then 'can't judge')")
    pr.add_argument("--meal-time", help="YYYY-MM-DDTHH:MM (default: last reading in the CSV)")
    for name in ("age", "bmi", "a1c", "fasting"):
        pr.add_argument(f"--{name}", type=float, required=True)
    pr.add_argument("--prior-rate", type=float, help="spike rate in your onboarding week (0-1), optional")
    pr.add_argument("--prior-rise", type=float, help="mean rise in your onboarding week (mg/dL), optional")
    pf = sub.add_parser("predict-folder", help="answer every meal folder under IN (photo.*, cgm.csv, profile.json)")
    pf.add_argument("in_dir", help="folder with one sub-folder per meal")
    pf.add_argument("out_dir", help="where results.jsonl and summary.md are written")
    sub.add_parser("ablate", help="D8 profile-excluded split + input-channel ablations (run `fusion` first)")
    sub.add_parser("wandb-export", help="offline W&B runs from every history CSV and summary in outputs/")
    sub.add_parser("robustness", help="decision-flip rates under CGM dropout/gap and photo blur/dark (S9)")
    sub.add_parser("probes", help="photo-swap and flat-vs-rising CGM counterfactual probes (S13)")
    sub.add_parser("efficiency", help="parameters vs caps, serving latency, artifact sizes (S12)")
    sub.add_parser("monitoring", help="drift-baseline figure for the monitoring design (S11)")
    sub.add_parser("finalize", help="regenerate the frozen serving artifacts (pre-registered amendment)")
    sub.add_parser("figures", help="render the report figures from committed evidence (outputs/figures)")
    sub.add_parser("timing", help="meal-time shift robustness, 15 min earlier and later (pre-registered)")
    od = sub.add_parser("ood", help="unfamiliar-photo check vs real phone photos; re-choose the cut (R7)")
    od.add_argument("--apply", action="store_true", help="write the chosen cut into the reference file")
    sub.add_parser("splits-v2", help="write partition v2 (dev folds, T2D train-only) + asserted reconciliation")
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
        "recalibrate": _recalibrate,
        "photo-seeds": _photo_seeds,
        "demo": _demo,
        "predict": _predict,
        "predict-folder": _predict_folder,
        "ablate": _ablate,
        "wandb-export": _wandb_export,
        "ood": _ood,
        "robustness": _robustness,
        "probes": _probes,
        "efficiency": _efficiency,
        "monitoring": _monitoring,
        "finalize": _finalize,
        "figures": _figures,
        "timing": _timing,
        "splits-v2": _splits_v2,
        "selftest": _selftest,
    }
    return commands[args.command](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
