"""Trains the fine-tuned photo model (`just train-photo`) and measures its seed spread (`just photo-seeds`).

Produces checkpoints/photo_convnext_*.pt, outputs/training/photo_convnext_{history.csv,curves.png,choice.json}
and outputs/training/photo_convnext_seeds.json (report: training table and seed-spread row)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import Splits
from glimpse.models.photo_model import PhotoCarbModel
from glimpse.models.weights import weights_path
from glimpse.training.common import EpochLogger, plot_curves
from glimpse.training.photo import (
    MODEL,
    _evaluate,
    _last_stage_state,
    _subset_views,
    build_cache,
    choose_recipe,
    embed_maps,
    grams,
    measures,
    photo_rows,
    predict,
    train_heads,
    train_last_stage,
)


def train_photo(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> Path:
    """Inner folds choose the head epochs and keep/reject phase 2; then the 28-person model is trained and saved."""
    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))
    hp, pc, log = cfg.training.photo, cfg.photo, logger.bind(stage="train", model=MODEL)
    out, cache = Path(cfg.paths.outputs) / "training", Path("data/cache")
    root, views = Path(cfg.data.cgmacros_root), hp.augmented_views
    model = PhotoCarbModel(pc.arch, weights=weights_path(pc), dropout=pc.dropout)
    fresh_heads = {k: v.clone() for k, v in model.state_dict().items() if "head." in k and "backbone" not in k}
    R, V = photo_rows(ds, set(sp.train), root), photo_rows(ds, set(sp.validation), root)
    gR, gV = grams(ds, R), grams(ds, V)
    log.info("photos: {} training (28 people), {} validation (4 people)", len(R), len(V))
    maps_eval = build_cache(model, ds, np.concatenate([R, V]), cfg, "maps_eval", cache / "photo_maps_eval.pt")
    mR, mV = maps_eval[: len(R)], maps_eval[len(R) :]
    eR, eV = embed_maps(model, mR), embed_maps(model, mV)
    maps_aug = build_cache(model, ds, R, cfg, "maps_aug", cache / "photo_maps_aug.pt") if hp.try_last_stage else None
    person = np.array([ds.samples[i].person_id for i in R])
    elog = EpochLogger(out / f"{MODEL}_history.csv")
    pristine_stage = _last_stage_state(model)

    best_epochs, mae1, mae2 = [], [], []
    for k, held in enumerate(sp.inner_folds):
        fit, hold = np.where(~np.isin(person, held))[0], np.where(np.isin(person, held))[0]
        model.load_state_dict(fresh_heads, strict=False)
        e = train_heads(
            model,
            eR[fit],
            gR[fit],
            hp,
            hp.max_epochs,
            cfg.seed + k,
            f"inner-fold-{k + 1}",
            elog,
            (eR[hold], gR[hold]),
            hp.patience,
        )
        best_epochs.append(e)
        mae1.append(_evaluate(model, eR[hold], gR[hold], hp)[1])
        if hp.try_last_stage:
            curve = train_last_stage(
                model,
                _subset_views(maps_aug, fit, views),
                views,
                gR[fit],
                hp,
                hp.last_stage_max_epochs,
                cfg.seed + k,
                f"inner-fold-{k + 1}-last-stage",
                elog,
                (mR[hold], gR[hold]),
                start_epoch=e,
            )
            mae2.append(curve)
            model.load_state_dict(pristine_stage, strict=False)
        log.info(
            "inner fold {}: heads best epoch {}, carb error {:.1f} g{}",
            k + 1,
            e,
            mae1[-1],
            f"; with last stage per epoch {[round(m, 1) for m in mae2[-1]]}" if mae2 else "",
        )
    e1, keep, e2 = choose_recipe(best_epochs, mae1, mae2, hp)
    choice = {
        "heads_best_epochs": best_epochs,
        "heads_epochs": e1,
        "inner_mae_heads_only_g": mae1,
        "inner_mae_last_stage_per_epoch_g": mae2,
        "last_stage_kept": keep,
        "last_stage_epochs": e2 if keep else 0,
        "rule": f"keep the last stage only if the mean inner-fold error drops by >= {hp.last_stage_min_gain_g} g",
    }
    log.info(
        "heads: {} epochs; last stage {} (inner-fold carb error {:.1f} g heads only{})",
        e1,
        f"KEPT for {e2} epochs" if keep else "rejected",
        float(np.mean(mae1)),
        f", {float(np.min(np.mean(mae2, axis=0))):.1f} g with last stage" if mae2 else "",
    )

    model.load_state_dict(fresh_heads, strict=False)
    train_heads(model, eR, gR, hp, e1, cfg.seed, "final", elog, (eV, gV))
    if keep:
        train_last_stage(model, maps_aug, views, gR, hp, e2, cfg.seed, "final", elog, (mV, gV), start_epoch=e1)
    elog.close()
    pred, probs = predict(model, embed_maps(model, mV) if keep else eV)
    val = measures(pred, probs, gV)
    strategy = "frozen backbone, heads trained" + (" + last stage fine-tuned" if keep else "")
    plot_curves(
        elog.records,
        "final",
        out / f"{MODEL}_curves.png",
        f"Photo model (ConvNeXt-Tiny): {strategy}",
        marker_epoch=e1 if keep else None,
    )
    (out / f"{MODEL}_choice.json").write_text(json.dumps(choice | {"validation": val}, indent=2))
    state = model.trainable_state(pc.weights_sha256)
    if not keep:  # only the heads changed
        state["state"] = {k: v for k, v in state["state"].items() if not k.startswith("backbone.")}
    state["state"] = {k: v.half() for k, v in state["state"].items()}
    ckpt_dir = Path("checkpoints")
    for old in ckpt_dir.glob(f"{MODEL}_e*.pt"):
        old.unlink()
    path = ckpt_dir / f"{MODEL}_e{e1 + (e2 if keep else 0):02d}_valmae{val['mae_g']:.1f}.pt"
    torch.save(
        state
        | {
            "arch": pc.arch,
            "strategy": strategy,
            "heads_epochs": e1,
            "last_stage_epochs": e2 if keep else 0,
            "validation": val,
            "seed": cfg.seed,
        },
        path,
    )
    (ckpt_dir / f"{MODEL}_latest.txt").write_text(path.name)
    log.info(
        "checkpoint {} ({:.2f} MB); validation carb error {:.1f} g, Spearman {:.2f}, bin F1 {:.2f}",
        path,
        path.stat().st_size / 1e6,
        val["mae_g"],
        val["spearman"],
        val["bin_f1"],
    )
    return path


def photo_seed_check(ds: MealDataset, sp: Splits, cfg: GlimpseConfig, extra_seeds: tuple[int, ...] = (1, 2, 3)) -> dict:
    """Seed spread of the shipped recipe (the finalize constants, 12 heads + 2 last-stage epochs): retrain it on
    the 28 training people with 3 fresh seeds and score the 4 validation people. The deployed checkpoint is not a
    row here: it trained on all 37 non-locked people, validation included, so scoring it on them is train-on-test.
    Nothing is saved to `checkpoints/`; the deployed checkpoint is untouched. No test labels are used."""
    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))
    hp, pc, log = cfg.training.photo, cfg.photo, logger.bind(stage="train", model=MODEL)
    out, cache = Path(cfg.paths.outputs) / "training", Path("data/cache")
    root, views = Path(cfg.data.cgmacros_root), hp.augmented_views
    from glimpse.training.finalize import HEADS_EPOCHS, LAST_STAGE_EPOCHS  # the one source of the shipped recipe

    e1, e2 = HEADS_EPOCHS, LAST_STAGE_EPOCHS
    model = PhotoCarbModel(pc.arch, weights=weights_path(pc), dropout=pc.dropout)
    fresh_heads = {k: v.clone() for k, v in model.state_dict().items() if "head." in k and "backbone" not in k}
    pristine_stage = _last_stage_state(model)
    R, V = photo_rows(ds, set(sp.train), root), photo_rows(ds, set(sp.validation), root)
    gR, gV = grams(ds, R), grams(ds, V)
    maps_eval = build_cache(model, ds, np.concatenate([R, V]), cfg, "maps_eval", cache / "photo_maps_eval.pt")
    mR, mV = maps_eval[: len(R)], maps_eval[len(R) :]
    eR, eV = embed_maps(model, mR), embed_maps(model, mV)
    maps_aug = build_cache(model, ds, R, cfg, "maps_aug", cache / "photo_maps_aug.pt") if e2 > 0 else None
    runs: list[dict] = []
    for s in extra_seeds:
        model.load_state_dict(fresh_heads, strict=False)
        model.load_state_dict(pristine_stage, strict=False)
        train_heads(model, eR, gR, hp, e1, cfg.seed + s, f"seed-{s}", None, (eV, gV))
        if e2 > 0:
            train_last_stage(model, maps_aug, views, gR, hp, e2, cfg.seed + s, f"seed-{s}", None, (mV, gV),
                             start_epoch=e1)  # fmt: skip
        pred, probs = predict(model, embed_maps(model, mV) if e2 > 0 else eV)
        val = measures(pred, probs, gV)
        runs.append({"seed": cfg.seed + s, "source": "retrained, not saved", **val})
        log.info("seed {}: validation carb error {:.1f} g, Spearman {:.2f}, bin F1 {:.2f}",
                 cfg.seed + s, val["mae_g"], val["spearman"], val["bin_f1"])  # fmt: skip
    summary = {k: {"mean": float(np.mean([r[k] for r in runs])), "sd": float(np.std([r[k] for r in runs], ddof=1))}
               for k in ("mae_g", "spearman", "bin_f1")}  # fmt: skip
    res = {"recipe": {"heads_epochs": e1, "last_stage_epochs": e2}, "runs": runs, "summary": summary,
           "note": "validation people only; fresh seeds of the shipped 12+2 recipe; the deployed checkpoint is "
                   "not a row (it trained on the validation people) and is not replaced"}  # fmt: skip
    (out / f"{MODEL}_seeds.json").write_text(json.dumps(res, indent=2))
    log.info("photo seed spread over {} seeds: carb error {:.1f} +/- {:.1f} g, bin F1 {:.2f} +/- {:.2f} -> {}",
             len(runs), summary["mae_g"]["mean"], summary["mae_g"]["sd"], summary["bin_f1"]["mean"],
             summary["bin_f1"]["sd"], out / f"{MODEL}_seeds.json")  # fmt: skip
    return res
