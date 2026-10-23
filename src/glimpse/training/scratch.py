"""Train the from-scratch SensorCNN.

1. Number of epochs: for each of the 4 inner folds, train on the other training people, watch the held-out fold,
   and note the epoch with the lowest validation loss. Take the median (middle value).
2. Final model: train on all 28 training people for that many epochs. The 4 validation people are only watched
   (curve for the report), never used to stop or choose anything here.
3. Save one bundle: weights + normaliser + settings, named with the epoch and the validation AUROC."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from torch import nn

from glimpse.config import GlimpseConfig, ScratchTrainCfg
from glimpse.data.augment import add_sensor_noise
from glimpse.data.dataset import MealDataset
from glimpse.data.normalise import ModelInputs, Normaliser
from glimpse.data.splits import Splits, meal_rows
from glimpse.models.sensor_cnn import SensorCNN, count_parameters
from glimpse.training.common import EpochLogger, EpochRecord, plot_curves, warmup_cosine

MODEL = "sensor_cnn"


def _inputs(ds: MealDataset, rows: np.ndarray, norm: Normaliser) -> ModelInputs:
    """Scaled model inputs for the given meal rows."""
    return norm.transform(ds.glucose[rows], ds.heart_rate[rows], ds.profile[rows])


@torch.no_grad()
def predict_proba(model: SensorCNN, x: ModelInputs) -> np.ndarray:
    """Spike probabilities (sigmoid of the model's raw scores)."""
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(x.sequence), torch.from_numpy(x.profile))).numpy()


@torch.no_grad()
def evaluate(model: SensorCNN, x: ModelInputs, y: np.ndarray) -> tuple[float, float]:
    """Loss and AUROC on held-out meals."""
    model.eval()
    logits = model(torch.from_numpy(x.sequence), torch.from_numpy(x.profile))
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(y).float()).item()
    auc = roc_auc_score(y, torch.sigmoid(logits).numpy()) if len(set(y)) == 2 else float("nan")
    return float(loss), float(auc)


def fit(ds: MealDataset, rows: np.ndarray, norm: Normaliser, hp: ScratchTrainCfg, cfg: GlimpseConfig, epochs: int,
        seed: int, run: str, elog: EpochLogger | None = None, val: tuple[ModelInputs, np.ndarray] | None = None,
        patience: int | None = None) -> tuple[SensorCNN, int]:  # fmt: skip
    """Train for up to `epochs`. With `patience`, keep the weights of the epoch with the lowest validation loss
    and stop early. Returns the model and its best epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = ds.labels()[rows].astype(np.float32)
    model = SensorCNN(n_profile=ds.profile.shape[1], channels=hp.channels, hidden=hp.hidden, dropout=hp.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    sched = warmup_cosine(opt, hp.warmup_epochs, epochs)
    best, best_epoch, best_state, waited = float("inf"), epochs, None, 0
    for epoch in range(1, epochs + 1):
        g, h = add_sensor_noise(ds.glucose[rows], ds.heart_rate[rows], cfg.augment, rng)  # new noise every epoch
        x = norm.transform(g, h, ds.profile[rows])
        model.train()
        order, total = rng.permutation(len(y)), 0.0
        for start in range(0, len(y), hp.batch_size):
            b = order[start : start + hp.batch_size]
            logits = model(torch.from_numpy(x.sequence[b]), torch.from_numpy(x.profile[b]))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        vl, va = evaluate(model, *val) if val else (None, None)
        if elog:
            elog.log(EpochRecord(model=MODEL, run=run, epoch=epoch, lr=lr, train_loss=total / len(y), val_loss=vl,
                                 val_auroc=va))  # fmt: skip
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


def choose_epochs(ds: MealDataset, sp: Splits, cfg: GlimpseConfig, elog: EpochLogger) -> tuple[int, list[int]]:
    """Median best epoch over the inner folds (normaliser refitted on each fold's training part)."""
    hp, best = cfg.training.scratch, []
    for k, held in enumerate(sp.inner_folds):
        fit_rows = meal_rows(ds, set(sp.train) - set(held), scored_only=False)
        val_rows = meal_rows(ds, set(held), scored_only=True)
        norm = Normaliser.fit(ds.glucose[fit_rows], ds.heart_rate[fit_rows], ds.profile[fit_rows], ds.profile_names)
        val = (_inputs(ds, val_rows, norm), ds.labels()[val_rows])
        _, e = fit(ds, fit_rows, norm, hp, cfg, hp.max_epochs, cfg.seed + k, f"inner-fold-{k + 1}", elog, val,
                   hp.patience)  # fmt: skip
        best.append(e)
        logger.bind(stage="train", model=MODEL).info("inner fold {}: best epoch {}", k + 1, e)
    return int(np.median(best)), best


def save_bundle(model: SensorCNN, norm: Normaliser, hp: ScratchTrainCfg, path: Path, **info) -> None:
    """Weights + normaliser + settings in one file, so the model is always used with its own scaling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "normaliser": norm.model_dump(), "hparams": hp.model_dump(),
                "n_parameters": count_parameters(model), **info}, path)  # fmt: skip


def load_bundle(path: Path) -> tuple[SensorCNN, Normaliser, dict]:
    """Rebuild the model and its normaliser from a saved bundle (only plain data is stored, no code)."""
    b = torch.load(path, weights_only=True)
    norm = Normaliser.model_validate(b["normaliser"])
    hp = b["hparams"]
    model = SensorCNN(n_profile=len(norm.profile_names), channels=hp["channels"], hidden=hp["hidden"],
                      dropout=hp["dropout"])  # fmt: skip
    model.load_state_dict(b["state_dict"])
    return model.eval(), norm, b


def train_scratch(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> Path:
    """Choose epochs on the inner folds, train the 28-person model, save curves and the checkpoint bundle."""
    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))
    hp, out, log = cfg.training.scratch, Path(cfg.paths.outputs) / "training", logger.bind(stage="train", model=MODEL)
    elog = EpochLogger(out / f"{MODEL}_history.csv")
    epochs, inner_best = choose_epochs(ds, sp, cfg, elog)
    log.info("inner folds' best epochs {} -> training for {} epochs", inner_best, epochs)
    norm = Normaliser.load(Path("checkpoints/normaliser.json"))  # fitted on the 28 training people by `prepare`
    tr = meal_rows(ds, set(sp.train), scored_only=False)
    vr = meal_rows(ds, set(sp.validation), scored_only=True)
    val = (_inputs(ds, vr, norm), ds.labels()[vr])
    log.info("final model: {} training meals from {} people; watching {} validation meals", len(tr), len(sp.train),
             len(vr))  # fmt: skip
    model, _ = fit(ds, tr, norm, hp, cfg, epochs, cfg.seed, "final", elog, val, patience=None)
    vl, va = evaluate(model, *val)
    elog.close()
    plot_curves(elog.records, "final", out / f"{MODEL}_curves.png",
                f"SensorCNN (from scratch): {epochs} epochs, chosen on the inner folds")  # fmt: skip
    (out / f"{MODEL}_epoch_choice.json").write_text(json.dumps({"inner_best_epochs": inner_best, "epochs": epochs}))
    ckpt_dir = Path("checkpoints")
    for old in ckpt_dir.glob(f"{MODEL}_e*.pt"):  # keep one current checkpoint
        old.unlink()
    path = ckpt_dir / f"{MODEL}_e{epochs:02d}_valauc{va:.3f}.pt"
    save_bundle(model, norm, hp, path, epochs=epochs, val_loss=vl, val_auroc=va, seed=cfg.seed,
                inner_best_epochs=inner_best)  # fmt: skip
    (ckpt_dir / f"{MODEL}_latest.txt").write_text(path.name)
    log.info("checkpoint {} ({} parameters); validation loss {:.3f}, AUROC {:.3f}", path, count_parameters(model),
             vl, va)  # fmt: skip
    return path
