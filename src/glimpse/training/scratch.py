"""Train the from-scratch sensor CNN: person-grouped CV chooses the epoch count, then one final run."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from torch import nn

from glimpse.config import GlimpseConfig, ScratchTrainCfg
from glimpse.data.augment import augment_sequence
from glimpse.data.dataset import MealDataset
from glimpse.data.normalise import ModelInputs, Normaliser
from glimpse.data.splits import Splits
from glimpse.models.sensor_cnn import SensorCNN, count_parameters
from glimpse.training.common import EpochLogger, EpochRecord, plot_curves, warmup_cosine

MODEL = "sensor_cnn"


def _rows(ds: MealDataset, people: set[int], eval_only: bool) -> np.ndarray:
    """Row indices for `people` (meals passing pre-meal checks).

    Onboarding meals are never used: their outcomes form the personal-prior feature, so training on them would put
    each meal's own label inside its inputs. Training uses all post-onboarding meals; evaluation only isolated ones.
    """
    keep = [
        i
        for i, s in enumerate(ds.samples)
        if s.person_id in people and s.premeal_ok and not s.onboarding and (s.isolated or not eval_only)
    ]
    return np.array(keep)


def stratified_person_folds(people: list[int], target: set[int], k: int, seed: int) -> list[list[int]]:
    """Split people into k folds with target-cohort and other people spread evenly (seeded)."""
    rng = np.random.default_rng(seed)
    a = [int(p) for p in rng.permutation(sorted(p for p in people if p in target))]
    b = [int(p) for p in rng.permutation(sorted(p for p in people if p not in target))]
    return [sorted(a[i::k] + b[i::k]) for i in range(k)]


def _inputs(ds: MealDataset, rows: np.ndarray, norm: Normaliser) -> ModelInputs:
    """Scaled model inputs for the given rows."""
    return norm.transform(ds.glucose[rows], ds.heart_rate[rows], ds.tabular[rows])


@torch.no_grad()
def predict_proba(model: SensorCNN, x: ModelInputs) -> np.ndarray:
    """Spike probabilities (sigmoid applied here, not inside the model)."""
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(x.sequence), torch.from_numpy(x.tabular))).numpy()


@torch.no_grad()
def _evaluate(model: SensorCNN, x: ModelInputs, y: np.ndarray) -> tuple[float, float]:
    """Validation BCE and AUROC."""
    model.eval()
    logits = model(torch.from_numpy(x.sequence), torch.from_numpy(x.tabular))
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(y).float()).item()
    auc = roc_auc_score(y, torch.sigmoid(logits).numpy()) if len(set(y)) == 2 else float("nan")
    return loss, float(auc)


def _fit(
    ds: MealDataset, train_rows: np.ndarray, norm: Normaliser, hp: ScratchTrainCfg, epochs: int, seed: int,
    run: str, elog: EpochLogger, val: tuple[ModelInputs, np.ndarray] | None, patience: int | None,
) -> tuple[SensorCNN, int]:  # fmt: skip
    """Train for up to `epochs`; with `patience`, stop early on validation loss and return the best epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, y = _inputs(ds, train_rows, norm), ds.labels()[train_rows].astype(np.float32)
    model = SensorCNN(n_tabular=x.tabular.shape[1], channels=hp.channels, hidden=hp.hidden, dropout=hp.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    sched = warmup_cosine(opt, hp.warmup_epochs, epochs)
    best, best_epoch, best_state, waited = float("inf"), epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order, total = rng.permutation(len(y)), 0.0
        for start in range(0, len(y), hp.batch_size):
            b = order[start : start + hp.batch_size]
            seq = augment_sequence(x.sequence[b], rng) if hp.augment else x.sequence[b]
            logits = model(torch.from_numpy(seq), torch.from_numpy(x.tabular[b]))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        vl, va = _evaluate(model, *val) if val else (None, None)
        elog.log(EpochRecord(model=MODEL, run=run, epoch=epoch, lr=lr, train_loss=total / len(y), val_loss=vl,
                             val_metric=va, val_metric_name="auroc"))  # fmt: skip
        if patience is not None and vl is not None:
            if vl < best - 1e-4:
                best, best_epoch, waited = vl, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
                if waited >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def train_scratch(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> Path:
    """Choose the epoch count by person-grouped CV inside the training people, retrain on all of them, save."""
    hp, out = cfg.training.scratch, Path(cfg.paths.outputs) / "training"
    log = logger.bind(stage="train", model=MODEL)
    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))  # never oversubscribe the CPU
    elog = EpochLogger(out / f"{MODEL}_history.csv", Path(cfg.training.tensorboard_dir) / MODEL)
    train_people = sorted(sp.train)
    target = {s.person_id for s in ds.samples if s.group in cfg.cohort.target_groups}
    folds = stratified_person_folds(train_people, target, hp.cv_folds, cfg.seed)
    best_epochs: list[int] = []
    oof_p, oof_y, oof_target, oof_rows = [], [], [], []
    log.info("cross-validation: {} folds over {} training people", hp.cv_folds, len(train_people))
    for k, held in enumerate(folds):
        fit_people = set(train_people) - set(held)
        fr = _rows(ds, fit_people, eval_only=False)
        norm = Normaliser.fit(ds.glucose[fr], ds.heart_rate[fr], ds.tabular[fr], ds.tabular_names)  # fold-only stats
        vr = _rows(ds, set(held), eval_only=True)
        val = (_inputs(ds, vr, norm), ds.labels()[vr])
        m, be = _fit(ds, fr, norm, hp, hp.max_epochs, cfg.seed + k, f"cv-fold-{k}", elog, val, hp.patience)
        best_epochs.append(be)
        oof_p.append(predict_proba(m, val[0]))
        oof_y.append(val[1])
        oof_target.append(np.array([ds.samples[i].person_id in target for i in vr]))
        oof_rows.append(vr)
        log.info("fold {} best epoch {}", k, be)
    epochs = int(np.median(best_epochs))
    p, y, t = np.concatenate(oof_p), np.concatenate(oof_y), np.concatenate(oof_target)
    cv = {"best_epochs": best_epochs, "epochs_chosen": epochs, "auroc_all_heldout": float(roc_auc_score(y, p)),
          "auroc_target_cohort_heldout": float(roc_auc_score(y[t], p[t])), "n_heldout_meals": int(len(y)),
          "n_heldout_target_meals": int(t.sum())}  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{MODEL}_cv_summary.json").write_text(json.dumps(cv, indent=2))
    rows = np.concatenate(oof_rows)  # held-out predictions: used later to choose the abstention band (never test)
    pd.DataFrame({"row": rows, "person_id": [ds.samples[i].person_id for i in rows], "target_cohort": t,
                  "spike": y, "p_spike": p}).to_csv(out / f"{MODEL}_oof.csv", index=False)  # fmt: skip
    log.info("CV: best epochs {} -> {} epochs; held-out AUROC all {:.3f}, target cohort only {:.3f}",
             best_epochs, epochs, cv["auroc_all_heldout"], cv["auroc_target_cohort_heldout"])  # fmt: skip

    norm = Normaliser.load(Path(cfg.paths.normaliser))  # fitted on all training people by `glimpse prepare`
    tr = _rows(ds, set(sp.train), eval_only=False)
    vr = _rows(ds, set(sp.validation), eval_only=True)
    val = (_inputs(ds, vr, norm), ds.labels()[vr])
    log.info(
        "final training: {} meals from {} people; monitoring on {} validation meals", len(tr), len(sp.train), len(vr)
    )
    model, _ = _fit(ds, tr, norm, hp, epochs, cfg.seed, "final", elog, val, patience=None)
    vl, va = _evaluate(model, *val)
    elog.close()
    plot_curves(elog.records, "final", out / f"{MODEL}_curves.png",
                f"Sensor CNN (from scratch): {epochs} epochs chosen by {hp.cv_folds}-fold person CV")  # fmt: skip
    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{MODEL}_e{epochs:02d}_valauc{va:.3f}.pt"
    torch.save({"state_dict": model.state_dict(), "hparams": hp.model_dump(), "n_tabular": len(ds.tabular_names),
                "tabular_names": ds.tabular_names, "normaliser": norm.model_dump(), "epochs": epochs,
                "val_loss": vl, "val_auroc": va, "cv_best_epochs": best_epochs, "seed": cfg.seed,
                "n_parameters": count_parameters(model), "version": f"{MODEL}-e{epochs}"}, path)  # fmt: skip
    (ckpt_dir / f"{MODEL}_latest.txt").write_text(path.name)
    log.info("checkpoint written to {} ({} parameters, validation AUROC {:.3f})", path, count_parameters(model), va)
    return path
