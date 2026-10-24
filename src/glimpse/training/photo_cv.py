"""Photo cross-validation (Step 4c-2): the photo model's carb estimate for every development meal, made by a model
that never saw that person.

- Out-of-fold (OOF) estimates: for each of the 5 outer folds, the 4b recipe is chosen again inside that fold's
  training people only (same 4 inner folds as Steps 3c and 4c-1), trained on all of them, and used on the held-out
  people. These feed the TEST side of the fusion in 4c-3.
- Nested estimates: inside each outer fold, the chosen recipe is refitted once per inner fold (fixed epochs, no
  early stopping) and used on that inner fold's held-out people. Every training person gets an estimate from a model
  that never saw them, made the same way as the OOF estimates. These feed the TRAINING side of the fusion.
- Baselines on the same folds: the training median (always the same guess) and frozen ConvNeXt summaries + ridge
  regression (no fine-tuning), to show whether training the heads and last stage earned its keep.
Each finished outer fold is saved, so an interrupted run continues where it stopped."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import Splits, meal_rows
from glimpse.models.photo_model import PhotoCarbModel, carb_bin
from glimpse.models.weights import weights_path
from glimpse.training.common import EpochLogger
from glimpse.training.cv import _inner_folds, outer_training_people
from glimpse.training.photo import (
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

RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]
BINS = ["p_bin0", "p_bin1", "p_bin2", "p_bin3"]


def fold_plan(sp: Splits, groups: dict[int, str], cfg: GlimpseConfig) -> list[dict]:
    """Who trains and who is held out, per outer fold and inner fold (the same folds as Steps 3c and 4c-1)."""
    plan = []
    for k, fold in enumerate(sp.outer_folds):
        people = outer_training_people(sp, k, groups)
        plan.append({"fold": k + 1, "held": sorted(fold), "train": people,
                     "inner": _inner_folds(people, groups, cfg.cv.inner_folds, cfg.seed + 100 + k)})  # fmt: skip
    return plan


def _one_hot_bins(g: np.ndarray) -> np.ndarray:
    """Bin 'probabilities' of a model that only gives grams: 1 for the bin its estimate falls in."""
    b = carb_bin(torch.from_numpy(g.astype(np.float32))).numpy()
    return np.eye(4)[b]


def _ridge(emb: np.ndarray, g: np.ndarray, alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(emb, np.log1p(g))


def _ridge_predict(m, emb: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(m.predict(emb)), 0, None)


class _Recipe:
    """Holds the model, the caches and the untouched starting weights, and trains one recipe from scratch."""

    def __init__(self, model, maps_eval, maps_aug, emb, g, cfg):
        self.model, self.maps_eval, self.maps_aug, self.emb, self.g = model, maps_eval, maps_aug, emb, g
        self.hp, self.views = cfg.training.photo, cfg.training.photo.augmented_views
        self.fresh = {k: v.clone() for k, v in model.state_dict().items() if "head." in k and "backbone" not in k}
        self.pristine = _last_stage_state(model)

    def reset(self) -> None:
        self.model.load_state_dict(self.fresh, strict=False)
        self.model.load_state_dict(self.pristine, strict=False)

    def select(self, pos_fit, pos_hold, seed, run, elog) -> tuple[int, float, list[float]]:
        """One inner fold of the recipe choice: heads with early stopping, then the last-stage trial."""
        self.reset()
        hp, g = self.hp, self.g
        e = train_heads(self.model, self.emb[pos_fit], g[pos_fit], hp, hp.max_epochs, seed, run, elog,
                        (self.emb[pos_hold], g[pos_hold]), hp.patience)  # fmt: skip
        pred, _ = predict(self.model, self.emb[pos_hold])
        mae1 = float(np.mean(np.abs(pred - g[pos_hold])))
        curve = []
        if hp.try_last_stage:
            curve = train_last_stage(self.model, _subset_views(self.maps_aug, pos_fit, self.views), self.views,
                                     g[pos_fit], hp, hp.last_stage_max_epochs, seed, f"{run}-last-stage", elog,
                                     (self.maps_eval[pos_hold], g[pos_hold]), start_epoch=e)  # fmt: skip
        self.reset()
        return e, mae1, curve

    def fit_predict(self, pos_fit, pos_out, e1, keep, e2, seed, run, elog) -> tuple[np.ndarray, np.ndarray]:
        """Train the chosen recipe (fixed epochs) on pos_fit, then estimate grams + bin probabilities for pos_out."""
        self.reset()
        train_heads(self.model, self.emb[pos_fit], self.g[pos_fit], self.hp, e1, seed, run, elog)
        if keep:
            train_last_stage(self.model, _subset_views(self.maps_aug, pos_fit, self.views), self.views,
                             self.g[pos_fit], self.hp, e2, seed, f"{run}-last-stage", elog, start_epoch=e1)  # fmt: skip
            out = predict(self.model, embed_maps(self.model, self.maps_eval[pos_out]))
        else:
            out = predict(self.model, self.emb[pos_out])
        self.reset()
        return out


def _run_key(cfg: GlimpseConfig, rows: np.ndarray, f: dict) -> str:
    """Identity of one outer fold's run: photos, settings and people. A saved fold is reused only if it matches."""
    raw = json.dumps([rows.tolist(), cfg.training.photo.model_dump(), cfg.photo.model_dump(), cfg.seed, f])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_photo_cv(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> dict:
    """OOF + nested photo estimates, baselines and a summary -> outputs/photo_cv/."""
    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))
    log, out = logger.bind(stage="photo_cv"), Path(cfg.paths.outputs) / "photo_cv"
    out.mkdir(parents=True, exist_ok=True)
    groups = {s.person_id: s.group for s in ds.samples}
    dev = set(groups) - set(sp.locked)
    root, pc = Path(cfg.data.cgmacros_root), cfg.photo
    rows = photo_rows(ds, dev, root)
    person = np.array([ds.samples[i].person_id for i in rows])
    assert not set(person) & set(sp.locked), "a locked person's photo is in the photo CV"
    g = grams(ds, rows)
    log.info("photo CV: {} photos of {} development people (locked people excluded)", len(rows), len(set(person)))

    model = PhotoCarbModel(pc.arch, weights=weights_path(pc), dropout=pc.dropout)
    cache = Path("data/cache")
    maps_eval = build_cache(model, ds, rows, cfg, "maps_eval", cache / "photo_cv_maps_eval.pt")
    maps_aug = build_cache(model, ds, rows, cfg, "maps_aug", cache / "photo_cv_maps_aug.pt") \
        if cfg.training.photo.try_last_stage else None  # fmt: skip
    emb = embed_maps(model, maps_eval)  # frozen summaries (pretrained weights untouched)
    emb_np = emb.numpy()
    recipe = _Recipe(model, maps_eval, maps_aug, emb, g, cfg)
    elog = EpochLogger(out / "photo_cv_history.csv")
    target = {p for fold in sp.outer_folds for p in fold}  # the 23 development target people (scored)
    scored = set(meal_rows(ds, target, scored_only=True).tolist())

    oof_parts, nested_parts, choices = [], [], []
    for f in fold_plan(sp, groups, cfg):
        k, t0 = f["fold"], time.time()
        key, saved = _run_key(cfg, rows, f), out / f"fold{k}.json"
        if saved.is_file() and json.loads(saved.read_text()).get("key") == key:
            log.info("fold {}: reusing saved results", k)
            choices.append(json.loads(saved.read_text()))
            oof_parts.append(pd.read_csv(out / f"fold{k}_oof.csv"))
            nested_parts.append(pd.read_csv(out / f"fold{k}_nested.csv"))
            continue
        pos_t, pos_h = np.where(np.isin(person, f["train"]))[0], np.where(np.isin(person, f["held"]))[0]
        assert not set(person[pos_t]) & set(f["held"]), "a held-out person is in training"

        # 1. choose the recipe inside this fold's training people
        best, mae1, mae2, ridge_err = [], [], [], {a: [] for a in RIDGE_ALPHAS}
        for j, held in enumerate(f["inner"]):
            fit, hold = pos_t[~np.isin(person[pos_t], held)], pos_t[np.isin(person[pos_t], held)]
            e, m1, curve = recipe.select(fit, hold, cfg.seed + 10 * k + j, f"fold{k}-inner{j + 1}", elog)
            best.append(e)
            mae1.append(m1)
            if curve:
                mae2.append(curve)
            for a in RIDGE_ALPHAS:
                pr = _ridge_predict(_ridge(emb_np[fit], g[fit], a), emb_np[hold])
                ridge_err[a].append(float(np.mean(np.abs(pr - g[hold]))))
        e1, keep, e2 = choose_recipe(best, mae1, mae2, cfg.training.photo)
        alpha = min(RIDGE_ALPHAS, key=lambda a: float(np.mean(ridge_err[a])))

        # 2. nested estimates: chosen recipe refitted per inner fold, used on that fold's held-out people
        nested = []
        for j, held in enumerate(f["inner"]):
            fit, hold = pos_t[~np.isin(person[pos_t], held)], pos_t[np.isin(person[pos_t], held)]
            pg, pb = recipe.fit_predict(fit, hold, e1, keep, e2, cfg.seed + 10 * k + j, f"fold{k}-nested{j + 1}",
                                        elog)  # fmt: skip
            part = {"fold": k, "inner": j + 1, "row": rows[hold], "person_id": person[hold], "carbs_g": g[hold],
                    "photo_g": pg, **dict(zip(BINS, pb.T, strict=True))}  # fmt: skip
            nested.append(pd.DataFrame(part))
        nested_df = pd.concat(nested)
        assert sorted(nested_df.row) == sorted(rows[pos_t]), "every training photo needs exactly one nested estimate"

        # 3. out-of-fold estimates + baselines for the held-out people
        pg, pb = recipe.fit_predict(pos_t, pos_h, e1, keep, e2, cfg.seed + 10 * k, f"fold{k}-final", elog)
        ridge_g = _ridge_predict(_ridge(emb_np[pos_t], g[pos_t], alpha), emb_np[pos_h])
        oof = pd.DataFrame({"fold": k, "row": rows[pos_h], "person_id": person[pos_h],
                            "group": [groups[p] for p in person[pos_h]],
                            "scored": [int(r) in scored for r in rows[pos_h]], "carbs_g": g[pos_h], "photo_g": pg,
                            **dict(zip(BINS, pb.T, strict=True)), "ridge_g": ridge_g,
                            "median_g": float(np.median(g[pos_t]))})  # fmt: skip
        ch = {"key": key, "fold": k, "training_photos": len(pos_t), "held_out_photos": len(pos_h),
              "heads_best_epochs": best, "heads_epochs": e1, "inner_mae_heads_only_g": mae1,
              "inner_mae_last_stage_per_epoch_g": mae2, "last_stage_kept": keep, "last_stage_epochs": e2,
              "ridge_alpha": alpha, "minutes": round((time.time() - t0) / 60, 1)}  # fmt: skip
        log.info("fold {}: heads {} epochs, last stage {}, ridge alpha {}; {:.1f} min", k, e1,
                 f"kept {e2} epochs" if keep else "rejected", alpha, ch["minutes"])  # fmt: skip
        oof.to_csv(out / f"fold{k}_oof.csv", index=False)
        nested_df.to_csv(out / f"fold{k}_nested.csv", index=False)
        saved.write_text(json.dumps(ch, indent=2))
        choices.append(ch)
        oof_parts.append(oof)
        nested_parts.append(nested_df)
    elog.close()

    oof, nested_df = pd.concat(oof_parts, ignore_index=True), pd.concat(nested_parts, ignore_index=True)
    oof.to_csv(out / "photo_oof.csv", index=False)
    nested_df.to_csv(out / "photo_nested.csv", index=False)
    summary = summarise_photo_cv(oof, nested_df, choices, len(scored))
    (out / "photo_cv_summary.json").write_text(json.dumps(summary, indent=2))
    for m, v in summary["scored_meals_with_photo"]["methods"].items():
        log.info("{:<26} carb error {:5.1f} g | Spearman {:.2f} | bin F1 {:.2f}", m, v["mae_g"], v["spearman"],
                 v["bin_f1"])  # fmt: skip
    log.info("nested estimates (fusion training side): carb error {:.1f} g", summary["nested_mae_g"])
    return summary


def summarise_photo_cv(oof: pd.DataFrame, nested: pd.DataFrame, choices: list[dict], n_scored: int) -> dict:
    """Carb error, rank agreement and bin F1 per method over the scored meals that have a photo, plus per fold."""
    s = oof[oof.scored]
    true = s.carbs_g.to_numpy()
    methods = {
        "photo model (fine-tuned)": (s.photo_g.to_numpy(), s[BINS].to_numpy()),
        "frozen ConvNeXt + ridge": (s.ridge_g.to_numpy(), _one_hot_bins(s.ridge_g.to_numpy())),
        "training median": (s.median_g.to_numpy(), _one_hot_bins(s.median_g.to_numpy())),
    }
    per_fold = {int(k): float(np.mean(np.abs(f.photo_g - f.carbs_g))) for k, f in s.groupby("fold")}
    return {
        "scored_meals_with_photo": {"n": int(len(s)), "of_scored": n_scored, "people": int(s.person_id.nunique()),
                                    "methods": {m: measures(p, b, true) for m, (p, b) in methods.items()}},
        "photo_model_mae_by_fold_g": per_fold,
        "nested_mae_g": float(np.mean(np.abs(nested.photo_g - nested.carbs_g))),
        "folds": choices,
        "note": "one seed; nested estimates come from models trained on fewer people, so they are a little worse "
                "than the OOF ones",
    }  # fmt: skip


def ridge_estimates(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """POST-HOC (added after 4c-2): frozen ConvNeXt + ridge, with the alpha grid widened because 1000 (the old top)
    won in every fold. Same folds; OOF estimates for held-out people and nested ones for training people, exactly
    like the fine-tuned model, so ridge can be a secondary fusion arm. Takes about a minute (cached photos)."""
    log, out = logger.bind(stage="photo_cv"), Path(cfg.paths.outputs) / "photo_cv"
    out.mkdir(parents=True, exist_ok=True)
    groups = {s.person_id: s.group for s in ds.samples}
    dev, grid = set(groups) - set(sp.locked), cfg.fusion.ridge_alpha_grid_posthoc
    rows = photo_rows(ds, dev, Path(cfg.data.cgmacros_root))
    person, g = np.array([ds.samples[i].person_id for i in rows]), grams(ds, rows)
    model = PhotoCarbModel(cfg.photo.arch, weights=weights_path(cfg.photo), dropout=cfg.photo.dropout)
    maps = build_cache(model, ds, rows, cfg, "maps_eval", Path("data/cache") / "photo_cv_maps_eval.pt")
    emb = embed_maps(model, maps).numpy()
    oof, nested, alphas = [], [], {}
    for f in fold_plan(sp, groups, cfg):
        k = f["fold"]
        pos_t, pos_h = np.where(np.isin(person, f["train"]))[0], np.where(np.isin(person, f["held"]))[0]
        splits = [(pos_t[~np.isin(person[pos_t], h)], pos_t[np.isin(person[pos_t], h)]) for h in f["inner"]]
        err = {a: np.mean([np.mean(np.abs(_ridge_predict(_ridge(emb[a_], g[a_], a), emb[b_]) - g[b_]))
                           for a_, b_ in splits]) for a in grid}  # fmt: skip
        alphas[k] = min(grid, key=lambda a: float(err[a]))
        for j, (fit, hold) in enumerate(splits):
            pr = _ridge_predict(_ridge(emb[fit], g[fit], alphas[k]), emb[hold])
            part = {"fold": k, "inner": j + 1, "row": rows[hold], "carbs_g": g[hold], "ridge_g": pr}
            nested.append(pd.DataFrame(part))
        pr = _ridge_predict(_ridge(emb[pos_t], g[pos_t], alphas[k]), emb[pos_h])
        oof.append(pd.DataFrame({"fold": k, "row": rows[pos_h], "carbs_g": g[pos_h], "ridge_g": pr}))
    oof_df, nested_df = pd.concat(oof, ignore_index=True), pd.concat(nested, ignore_index=True)
    oof_df.to_csv(out / "ridge_posthoc_oof.csv", index=False)
    nested_df.to_csv(out / "ridge_posthoc_nested.csv", index=False)
    info = {"alpha_grid": grid, "alpha_by_fold": alphas}
    log.info("post-hoc ridge: alpha per fold {}", alphas)
    return oof_df, nested_df, info
