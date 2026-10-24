"""Shared training helpers: the learning-rate schedule, per-epoch logging (screen/log file + CSV) and curve plots."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict


class EpochRecord(BaseModel):
    """What is recorded after every epoch."""

    model_config = ConfigDict(extra="forbid")
    model: str
    run: str  # e.g. "inner-fold-2" or "final"
    epoch: int
    lr: float
    train_loss: float
    val_loss: float | None = None
    val_metric: float | None = None  # AUROC for the sensor model, carb error in grams for the photo model
    val_metric_name: str = "auroc"


def warmup_cosine(optimizer: torch.optim.Optimizer, warmup: int, total: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Learning rate: small steps rising to full size over `warmup` epochs, then a smooth (cosine) fade to zero."""

    def factor(epoch: int) -> float:
        """Learning-rate multiplier: linear warm-up, then cosine decay."""
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EpochLogger:
    """Writes every epoch to Loguru (screen + logs/glimpse.log) and to a CSV file."""

    def __init__(self, csv_path: Path) -> None:
        """Start a fresh CSV file with a header row."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self._f = open(csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=list(EpochRecord.model_fields))
        self._w.writeheader()
        self.records: list[EpochRecord] = []

    def log(self, rec: EpochRecord) -> None:
        """Record one epoch."""
        self.records.append(rec)
        self._w.writerow(rec.model_dump())
        self._f.flush()
        fmt = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
        logger.bind(model=rec.model, run=rec.run, epoch=rec.epoch).info(
            "train_loss={:.4f} val_loss={} val_{}={} lr={:.2e}",
            rec.train_loss, fmt(rec.val_loss), rec.val_metric_name, fmt(rec.val_metric), rec.lr)  # fmt: skip

    def close(self) -> None:
        """Close the CSV file."""
        self._f.close()


def plot_curves(records: list[EpochRecord], run: str, path: Path, title: str, marker_epoch: int | None = None) -> None:
    """Training and validation loss (left) and the validation measure (right) for one run, saved as a PNG.
    `marker_epoch` draws a dashed line, e.g. where the photo model's last stage was unfrozen."""
    import matplotlib

    matplotlib.use("Agg")  # draw to a file, no window
    import matplotlib.pyplot as plt

    rs = [r for r in records if r.run == run]
    ep = [r.epoch for r in rs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.plot(ep, [r.train_loss for r in rs], label="training")
    if any(r.val_loss is not None for r in rs):
        a.plot(ep, [r.val_loss for r in rs], label="validation")
    a.set(xlabel="epoch", ylabel="loss")
    a.legend()
    if any(r.val_metric is not None for r in rs):
        b.plot(ep, [r.val_metric for r in rs], color="tab:green")
    names = {"auroc": "validation AUROC", "mae_g": "validation carb error (g)"}
    b.set(xlabel="epoch", ylabel=names.get(rs[0].val_metric_name, rs[0].val_metric_name) if rs else "")
    for ax in (a, b):
        ax.grid(alpha=0.3)
        if marker_epoch is not None:
            ax.axvline(marker_epoch + 0.5, color="grey", linestyle="--", linewidth=1)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.bind(stage="train").info("learning curves written to {}", path)
