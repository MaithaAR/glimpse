"""The final 28-person pipeline (Step 5a), used by the evaluation's fixed test and by the demo.

1. Photo estimates. Training people: nested, i.e. the 4b recipe refitted on each inner fold of the 28 people and
   used on that fold's held-out people (made exactly as in the cross-validation). Validation people: the saved 4b
   photo model (trained on all 28 people), as at use time.
2. Fusion: logistic regression on profile + trend + photo inputs (and the no-meal version for comparison), C chosen
   on the same inner folds, fitted on all 28 people.
3. Thresholds: low risk / abstain / caution chosen on the 4 validation people by expected cost, then saved with a
   hash BEFORE any test person is scored. The pooled cross-validated predictions give a stability check.
The test people are not touched here."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import Splits, meal_rows
from glimpse.evaluation import choose_thresholds, decide, usefulness_report
from glimpse.models.photo_model import PhotoCarbModel
from glimpse.models.weights import weights_path
from glimpse.training.baselines import logistic, trend_features
from glimpse.training.common import EpochLogger
from glimpse.training.cv import _tune_sklearn
from glimpse.training.fusion import PHOTO_COLS, photo_table, with_table
from glimpse.training.photo import build_cache, embed_maps, grams, load_photo_bundle, photo_rows, predict
from glimpse.training.photo_cv import BINS, _Recipe

FINAL = Path("checkpoints/fusion_final.joblib")
THRESHOLDS = Path("checkpoints/thresholds.json")


def cost_dict(cfg: GlimpseConfig) -> dict:
    """Costs as (spike, no spike) per answer."""
    c = cfg.costs
    return {"low_risk": tuple(c.low_risk), "caution": tuple(c.caution), "abstain": tuple(c.abstain)}


def hash_thresholds(d: dict) -> str:
    """Fingerprint of the frozen thresholds (everything except the hash itself)."""
    body = {k: v for k, v in d.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_thresholds(path: Path = THRESHOLDS) -> dict:
    """Read the frozen thresholds and refuse them if they were edited after freezing."""
    d = json.loads(path.read_text())
    if hash_thresholds(d) != d["sha256"]:
        raise ValueError("thresholds.json was changed after it was frozen")
    return d


def _photo_checkpoint() -> Path:
    """The saved 28-person photo model (Step 4b)."""
    return Path("checkpoints") / Path("checkpoints/photo_convnext_latest.txt").read_text().strip()


def checkpoint_estimates(cfg: GlimpseConfig, ds: MealDataset, rows_all, maps_eval, people: set[int]) -> pd.DataFrame:
    """Photo estimates of the saved 28-person photo model (4b) for these people's photos."""
    model, _ = load_photo_bundle(cfg, _photo_checkpoint())
    person = np.array([ds.samples[i].person_id for i in rows_all])
    pos = np.where(np.isin(person, sorted(people)))[0]
    g, b = predict(model, embed_maps(model, maps_eval[pos]))  # current last stage (fine-tuned if it was kept)
    return pd.DataFrame({"row": rows_all[pos], "photo_g": g, **dict(zip(BINS, b.T, strict=True))})


def train_final(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> dict:
    """Fit the 28-person fusion, choose and freeze thresholds on the validation people."""
    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))
    log, out = logger.bind(stage="final"), Path(cfg.paths.outputs) / "final"
    out.mkdir(parents=True, exist_ok=True)
    groups = {s.person_id: s.group for s in ds.samples}
    assert not set(sp.train) & (set(sp.test) | set(sp.validation) | set(sp.locked)), "split overlap"
    root, pc = Path(cfg.data.cgmacros_root), cfg.photo
    dev = set(groups) - set(sp.locked)
    rows_all = photo_rows(ds, dev, root)  # same photos and order as the photo CV caches (reused, not recomputed)
    person = np.array([ds.samples[i].person_id for i in rows_all])
    fresh = PhotoCarbModel(pc.arch, weights=weights_path(pc), dropout=pc.dropout)
    maps_eval = build_cache(fresh, ds, rows_all, cfg, "maps_eval", Path("data/cache/photo_cv_maps_eval.pt"))
    maps_aug = build_cache(fresh, ds, rows_all, cfg, "maps_aug", Path("data/cache/photo_cv_maps_aug.pt"))
    g = grams(ds, rows_all)

    # 1. nested photo estimates for the 28 training people (4b recipe, refitted per inner fold)
    _, bundle = load_photo_bundle(cfg, _photo_checkpoint())
    e1, e2 = int(bundle["heads_epochs"]), int(bundle["last_stage_epochs"])
    recipe = _Recipe(fresh, maps_eval, maps_aug, embed_maps(fresh, maps_eval), g, cfg)
    elog = EpochLogger(out / "final_photo_history.csv")
    pos_t = np.where(np.isin(person, sp.train))[0]
    nested = []
    for j, held in enumerate(sp.inner_folds):
        fit, hold = pos_t[~np.isin(person[pos_t], held)], pos_t[np.isin(person[pos_t], held)]
        pg, pb = recipe.fit_predict(fit, hold, e1, e2 > 0, e2, cfg.seed + j, f"final-nested{j + 1}", elog)
        nested.append(pd.DataFrame({"row": rows_all[hold], "photo_g": pg, **dict(zip(BINS, pb.T, strict=True))}))
        log.info("nested photo estimates, inner fold {}: {} photos", j + 1, len(hold))
    elog.close()
    nested_df = pd.concat(nested, ignore_index=True)
    assert sorted(nested_df.row) == sorted(rows_all[pos_t]), "every training photo needs one nested estimate"
    val_df = checkpoint_estimates(cfg, ds, rows_all, maps_eval, set(sp.validation))
    table = photo_table(nested_df) | photo_table(val_df)

    # 2. fusion on the 28 people (C from the same inner folds)
    y, grid = ds.labels(), cfg.baselines.logreg_c_grid
    tr, va = meal_rows(ds, set(sp.train), scored_only=False), meal_rows(ds, set(sp.validation), scored_only=True)
    X = {"no_meal": trend_features, "photo": with_table(table, len(PHOTO_COLS))}
    models, chosen = {}, {}
    for arm, fx in X.items():
        c, _ = _tune_sklearn(lambda c: logistic(c, cfg.seed), grid, fx, ds, sp.train, sp.inner_folds, y)
        chosen[arm] = c
        models[arm] = logistic(c, cfg.seed).fit(fx(ds, tr), y[tr])
    p_val = {a: models[a].predict_proba(X[a](ds, va))[:, 1] for a in X}
    auc = {a: float(roc_auc_score(y[va], p)) for a, p in p_val.items()}

    # 3. thresholds on the validation people; stability check on the pooled CV predictions
    costs = cost_dict(cfg)
    lo, hi, cost = choose_thresholds(p_val["photo"], y[va], costs)
    cv_oof = pd.read_csv(Path(cfg.paths.outputs) / "fusion" / "fusion_oof.csv")
    lo_cv, hi_cv, _ = choose_thresholds(cv_oof.photo.to_numpy(), cv_oof.spike.to_numpy(), costs)
    targets = cfg.targets.model_dump()
    use_val = usefulness_report(decide(p_val["photo"], lo, hi), y[va], costs, targets)
    use_cv = usefulness_report(decide(cv_oof.photo.to_numpy(), lo, hi), cv_oof.spike.to_numpy(), costs, targets)
    frozen = {"model": "fusion: profile + trend + photo (fine-tuned ConvNeXt-Tiny)", "low_risk_below": lo,
              "caution_from": hi, "chosen_on": f"{len(sp.validation)} validation people ({len(va)} meals)",
              "costs": {k: list(v) for k, v in costs.items()},
              "stability_check_pooled_cv": {"low_risk_below": lo_cv, "caution_from": hi_cv}}  # fmt: skip
    frozen["sha256"] = hash_thresholds(frozen)
    THRESHOLDS.write_text(json.dumps(frozen, indent=2))
    joblib.dump({"models": models, "C": chosen, "photo_cols": PHOTO_COLS, "photo_heads_epochs": e1,
                 "photo_last_stage_epochs": e2, "train_people": sp.train}, FINAL)  # fmt: skip
    truth = dict(zip(rows_all.tolist(), g.tolist(), strict=True))
    report = {"validation_auroc": auc, "C": chosen, "thresholds": frozen, "validation_usefulness": use_val,
              "pooled_cv_usefulness_same_thresholds": use_cv, "validation_cost": cost,
              "nested_photo_mae_g": float(np.mean(np.abs(nested_df.photo_g - nested_df.row.map(truth))))}  # fmt: skip
    (out / "final_report.json").write_text(json.dumps(report, indent=2))
    log.info("validation AUROC: no meal {:.3f}, photo {:.3f} ({} meals, {} people)", auc["no_meal"], auc["photo"],
             len(va), len(sp.validation))  # fmt: skip
    log.info("thresholds (frozen, hash {}): low risk below {:.2f}, caution from {:.2f}; pooled CV would pick "
             "{:.2f} / {:.2f}", frozen["sha256"][:12], lo, hi, lo_cv, hi_cv)  # fmt: skip
    for name, u in (("validation", use_val), ("pooled CV", use_cv)):
        log.info("{}: low risk {:.0%}, caution {:.0%}, abstain {:.0%}; spikes among low risk {:.0%}; cost {:.3f} vs "
                 "best simple policy {:.3f} ({:+.0%}); useful={}", name, u["share_low_risk"], u["share_caution"],
                 u["share_abstain"], u["missed_among_cleared"], u["cost"], u["best_trivial_cost"],
                 -u["cost_reduction"], u["useful"])  # fmt: skip
    log.info("saved {} and {} (test people untouched)", FINAL, THRESHOLDS)
    return report
