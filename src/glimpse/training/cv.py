"""Cross-validation: the main estimate. For each of the 5 outer folds, every model is trained on all OTHER
development people (target people of the other 4 folds + all type 2 diabetes people) and predicts the scored
meals of the held-out fold. Every model gets the same folds, a normaliser refitted on the fold's training people,
and one setting tuned on inner folds inside those training people. Result: one out-of-fold prediction per scored
meal of the 23 development target people, per model and per seed."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import log_loss, roc_auc_score

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.normalise import Normaliser
from glimpse.data.splits import Splits, _folds, meal_rows
from glimpse.training.baselines import gbm, logistic, profile_features, trend_features
from glimpse.training.scratch import _inputs, fit, predict_proba

MODELS = ["prevalence", "profile_lr", "trend_lr", "trend_gbm", "sensor_cnn"]


def outer_training_people(sp: Splits, k: int, groups: dict[int, str]) -> list[int]:
    """Training people for outer fold k: every development person except the people of fold k."""
    dev = (set(groups) - set(sp.locked)) - set(sp.outer_folds[k])
    return sorted(dev)


def _inner_folds(people: list[int], groups: dict[int, str], k: int, seed: int) -> list[list[int]]:
    """Inner folds for tuning, each with a mix of target and type 2 diabetes people."""
    rng = np.random.default_rng(seed)
    t = [p for p in people if groups[p] != "T2D"]
    o = [p for p in people if groups[p] == "T2D"]
    return [sorted(a + b) for a, b in zip(_folds(t, k, rng), _folds(o, k, rng), strict=True)]


def _tune_sklearn(make, grid, X_fn, ds, people, inner, y_all) -> tuple:
    """Pick the setting with the lowest average held-out log loss over the inner folds."""
    scores = []
    for v in grid:
        losses = []
        for held in inner:
            fr = meal_rows(ds, set(people) - set(held), scored_only=False)
            vr = meal_rows(ds, set(held), scored_only=True)
            m = make(v).fit(X_fn(ds, fr), y_all[fr])
            losses.append(log_loss(y_all[vr], m.predict_proba(X_fn(ds, vr))[:, 1], labels=[0, 1]))
        scores.append(float(np.mean(losses)))
    return grid[int(np.argmin(scores))], scores


def _cnn_epochs(ds, people, inner, cfg, seed) -> int:
    """Median best epoch over the inner folds (normaliser refitted each time), as for the 28-person model."""
    hp, best = cfg.training.scratch, []
    for j, held in enumerate(inner):
        fr = meal_rows(ds, set(people) - set(held), scored_only=False)
        vr = meal_rows(ds, set(held), scored_only=True)
        norm = Normaliser.fit(ds.glucose[fr], ds.heart_rate[fr], ds.profile[fr], ds.profile_names)
        _, e = fit(ds, fr, norm, hp, cfg, hp.max_epochs, seed + j, f"cv-inner-{j}",
                   val=(_inputs(ds, vr, norm), ds.labels()[vr]), patience=hp.patience)  # fmt: skip
        best.append(e)
    return int(np.median(best))


def run_cv(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> pd.DataFrame:
    """Out-of-fold predictions of every model for every seed -> outputs/cv/."""
    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))
    log, out = logger.bind(stage="cv"), Path(cfg.paths.outputs) / "cv"
    out.mkdir(parents=True, exist_ok=True)
    groups = {s.person_id: s.group for s in ds.samples}
    y = ds.labels()
    rows_out, choices = [], []
    for k, fold in enumerate(sp.outer_folds):
        people = outer_training_people(sp, k, groups)
        assert not set(people) & set(fold), "a held-out person is in training"
        tr, te = meal_rows(ds, set(people), scored_only=False), meal_rows(ds, set(fold), scored_only=True)
        inner = _inner_folds(people, groups, cfg.cv.inner_folds, cfg.seed + 100 + k)
        c_prof, _ = _tune_sklearn(lambda c: logistic(c, cfg.seed), cfg.baselines.logreg_c_grid, profile_features,
                                  ds, people, inner, y)  # fmt: skip
        c_trend, _ = _tune_sklearn(lambda c: logistic(c, cfg.seed), cfg.baselines.logreg_c_grid, trend_features,
                                   ds, people, inner, y)  # fmt: skip
        depth, _ = _tune_sklearn(lambda d: gbm(d, cfg.seed), cfg.baselines.gbm_depth_grid, trend_features, ds,
                                 people, inner, y)  # fmt: skip
        epochs = _cnn_epochs(ds, people, inner, cfg, cfg.seed + 10 * k)
        choices.append({"fold": k + 1, "training_people": len(people), "scored_meals": len(te), "profile_lr_C": c_prof,
                        "trend_lr_C": c_trend, "gbm_depth": depth, "cnn_epochs": epochs})  # fmt: skip
        log.info("fold {}: {} training people, {} scored meals; tuned C={}/{} depth={} epochs={}", k + 1, len(people),
                 len(te), c_prof, c_trend, depth, epochs)  # fmt: skip
        norm = Normaliser.fit(ds.glucose[tr], ds.heart_rate[tr], ds.profile[tr], ds.profile_names)
        for seed in cfg.cv.seeds:
            preds = {
                "prevalence": np.full(len(te), y[tr].mean()),
                "profile_lr": logistic(c_prof, seed).fit(profile_features(ds, tr), y[tr]).predict_proba(
                    profile_features(ds, te))[:, 1],
                "trend_lr": logistic(c_trend, seed).fit(trend_features(ds, tr), y[tr]).predict_proba(
                    trend_features(ds, te))[:, 1],
                "trend_gbm": gbm(depth, seed).fit(trend_features(ds, tr), y[tr]).predict_proba(
                    trend_features(ds, te))[:, 1],
            }  # fmt: skip
            cnn, _ = fit(ds, tr, norm, cfg.training.scratch, cfg, epochs, seed, f"cv-fold-{k + 1}")
            preds["sensor_cnn"] = predict_proba(cnn, _inputs(ds, te, norm))
            for j, i in enumerate(te):
                rows_out.append({"seed": seed, "fold": k + 1, "row": int(i), "person_id": ds.samples[i].person_id,
                                 "group": ds.samples[i].group, "spike": int(y[i]),
                                 **{m: float(preds[m][j]) for m in MODELS}})  # fmt: skip
    oof = pd.DataFrame(rows_out)
    oof.to_csv(out / "oof_predictions.csv", index=False)
    summary = summarise(oof)
    checks = loss_check(ds, sp) if Path("checkpoints/sensor_cnn_latest.txt").is_file() else {}
    (out / "cv_summary.json").write_text(json.dumps({"tuning": choices, **summary, "loss_check": checks}, indent=2))
    for m in MODELS:
        seeds = ", ".join(f"{v:.3f}" for v in summary["auroc_pooled_by_seed"][m])
        msg = "{:<11} AUROC pooled {:.3f} (seeds: {}) | mean within folds {:.3f}"
        log.info(msg, m, summary["auroc_pooled_mean"][m], seeds, summary["auroc_per_fold_mean"][m])
    return oof


def summarise(oof: pd.DataFrame) -> dict:
    """AUROC per model and seed, two ways. Pooled: all out-of-fold meals ranked together. Per fold: AUROC inside each
    fold, then averaged. Pooling also mixes the folds' different baselines (e.g. a constant guess differs per fold),
    so both are reported; 95% ranges by person come in the evaluation step."""

    def auc(g: pd.DataFrame, m: str) -> float:
        """AUROC of model m on these meals (0.5 if only one class or a constant guess)."""
        return float(roc_auc_score(g.spike, g[m])) if g[m].nunique() > 1 and g.spike.nunique() > 1 else 0.5

    pooled = {m: [auc(g, m) for _, g in oof.groupby("seed")] for m in MODELS}
    per_fold = {m: [float(np.mean([auc(f, m) for _, f in g.groupby("fold")])) for _, g in oof.groupby("seed")]
                for m in MODELS}  # fmt: skip
    return {"scored_meals_per_seed": int(len(oof) / oof.seed.nunique()), "people": int(oof.person_id.nunique()),
            "auroc_pooled_by_seed": pooled, "auroc_pooled_mean": {m: float(np.mean(v)) for m, v in pooled.items()},
            "auroc_per_fold_mean": {m: float(np.mean(v)) for m, v in per_fold.items()},
            "auroc_seed_sd": {m: float(np.std(v)) for m, v in pooled.items()}}  # fmt: skip


def loss_check(ds: MealDataset, sp: Splits) -> dict:
    """Why was the 28-person model's validation loss below its training loss? Re-score with dropout and noise
    OFF, split by cohort, next to each group's spike rate (fewer spikes -> lower loss for the same model)."""
    from glimpse.training.scratch import evaluate, load_bundle

    model, norm, _ = load_bundle(Path("checkpoints") / Path("checkpoints/sensor_cnn_latest.txt").read_text().strip())
    y, res = ds.labels(), {}
    groups = {"train, target people": [p for p in sp.train if ds_group(ds, p) != "T2D"],
              "train, type 2 diabetes": [p for p in sp.train if ds_group(ds, p) == "T2D"],
              "validation": sp.validation}  # fmt: skip
    for name, people in groups.items():
        rows = meal_rows(ds, set(people), scored_only=True)
        loss, auc = evaluate(model, _inputs(ds, rows, norm), y[rows])
        res[name] = {
            "loss_eval_mode": round(loss, 4),
            "auroc": round(auc, 4),
            "spike_rate": round(float(y[rows].mean()), 3),
        }
    logger.bind(stage="cv").info("loss check (dropout and noise off): {}", res)
    return res


def ds_group(ds: MealDataset, person: int) -> str:
    """Cohort of a person."""
    return next(s.group for s in ds.samples if s.person_id == person)
