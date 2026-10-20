"""Shared training utilities: schedules, epoch logging (Loguru + CSV + TensorBoard) and learning-curve plots."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict


class EpochRecord(BaseModel):
    """One logged epoch."""

    model_config = ConfigDict(extra="forbid")
    model: str
    run: str  # e.g. "cv-fold-2" or "final"
    epoch: int
    lr: float
    train_loss: float
    val_loss: float | None = None
    val_metric: float | None = None  # AUROC (sensor model) or carb MAE in grams (photo model)
    val_metric_name: str
    phase: str = "-"


def warmup_cosine(optimizer: torch.optim.Optimizer, warmup: int, total: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up for `warmup` epochs, then cosine decay to zero at `total` epochs (stepped once per epoch)."""

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EpochLogger:
    """Writes every epoch to Loguru, a CSV history file and TensorBoard."""

    def __init__(self, csv_path: Path, tb_dir: Path) -> None:
        """Start a fresh CSV file and a TensorBoard writer."""
        from torch.utils.tensorboard import SummaryWriter

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self._f = open(csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=list(EpochRecord.model_fields))
        self._w.writeheader()
        self.tb = SummaryWriter(log_dir=str(tb_dir))
        self.records: list[EpochRecord] = []

    def log(self, rec: EpochRecord) -> None:
        """Record one epoch everywhere."""
        self.records.append(rec)
        self._w.writerow(rec.model_dump())
        self._f.flush()
        tag = f"{rec.model}/{rec.run}"
        self.tb.add_scalar(f"{tag}/train_loss", rec.train_loss, rec.epoch)
        self.tb.add_scalar(f"{tag}/lr", rec.lr, rec.epoch)
        if rec.val_loss is not None:
            self.tb.add_scalar(f"{tag}/val_loss", rec.val_loss, rec.epoch)
        if rec.val_metric is not None:
            self.tb.add_scalar(f"{tag}/val_{rec.val_metric_name}", rec.val_metric, rec.epoch)
        logger.bind(model=rec.model, run=rec.run, epoch=rec.epoch, phase=rec.phase).info(
            "train_loss={:.4f} val_loss={} val_{}={} lr={:.2e}",
            rec.train_loss,
            "-" if rec.val_loss is None else f"{rec.val_loss:.4f}",
            rec.val_metric_name,
            "-" if rec.val_metric is None else f"{rec.val_metric:.4f}",
            rec.lr,
        )

    def close(self) -> None:
        """Flush and close the CSV and TensorBoard files."""
        self._f.close()
        self.tb.close()


def plot_curves(records: list[EpochRecord], run: str, path: Path, title: str, marker_epoch: int | None = None) -> None:
    """Training/validation loss and the validation metric for one run, saved as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = [r for r in records if r.run == run]
    ep = [r.epoch for r in rs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.plot(ep, [r.train_loss for r in rs], label="train")
    if any(r.val_loss is not None for r in rs):
        a.plot(ep, [r.val_loss for r in rs], label="validation")
    a.set_xlabel("epoch")
    a.set_ylabel("loss")
    a.legend()
    if any(r.val_metric is not None for r in rs):
        b.plot(ep, [r.val_metric for r in rs], color="tab:green")
        b.set_ylabel(f"validation {rs[0].val_metric_name}")
    b.set_xlabel("epoch")
    for ax in (a, b):
        ax.grid(alpha=0.3)
        if marker_epoch is not None:
            ax.axvline(marker_epoch, color="grey", linestyle="--", linewidth=1)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.bind(stage="train").info("learning curves written to {}", path)
