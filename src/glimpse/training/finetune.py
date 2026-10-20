"""Fine-tune the pretrained photo model (ConvNeXt-Tiny -> served carbs).

Strategy (CPU budget, 922 training photos): the frozen lower backbone runs once per photo and is cached, with
`views_per_photo` augmented copies per training photo. Phase 1 trains only the heads (frozen backbone). Phase 2
additionally unfreezes stage 3; it is kept only if person-grouped CV on the training people shows a lower carb error.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageOps
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch import nn

from glimpse.config import FinetuneCfg, GlimpseConfig
from glimpse.data.augment import photo_transforms
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import Splits
from glimpse.models.photo_model import PhotoCarbModel, carb_bin
from glimpse.training.common import EpochLogger, EpochRecord, plot_curves, warmup_cosine
from glimpse.training.scratch import stratified_person_folds

MODEL = "photo_convnext"


class FeatureCache:
    """Cached stage-2 feature maps (fp16) with the dataset row each one came from."""

    def __init__(self, feats: torch.Tensor, rows: np.ndarray) -> None:
        """feats: (N, 384, 14, 14); rows: (N,) dataset row index per cached item."""
        self.feats, self.rows = feats, rows

    def select(self, people_rows: set[int]) -> FeatureCache:
        """Subset to cached items whose dataset row is in `people_rows`."""
        m = np.isin(self.rows, list(people_rows))
        return FeatureCache(self.feats[torch.from_numpy(m)], self.rows[m])


def photo_rows(ds: MealDataset, people: set[int], photo_root: Path) -> np.ndarray:
    """Rows with an existing photo and a served-carbohydrate label."""
    return np.array([i for i, s in enumerate(ds.samples) if s.person_id in people and s.photo_relpath
                     and s.carbs_served_g is not None and (photo_root / s.photo_relpath).is_file()])  # fmt: skip


def _load(path: Path) -> Image.Image:
    """Open a photo upright in RGB."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


@torch.no_grad()
def build_cache(model: PhotoCarbModel, ds: MealDataset, rows: np.ndarray, photo_root: Path, views: int, train: bool,
                path: Path, seed: int, batch: int = 32) -> FeatureCache:  # fmt: skip
    """Run the frozen lower backbone once per (photo, view) and save; reuse the file when rows and views match."""
    log = logger.bind(stage="cache", model=MODEL)
    if path.is_file():
        d = torch.load(path, weights_only=True)  # tensors only: safe to load
        if np.array_equal(d["rows"].numpy(), np.repeat(rows, views)):
            log.info("reusing feature cache {}", path)
            return FeatureCache(d["feats"], d["rows"].numpy())
    torch.manual_seed(seed)
    tf = photo_transforms(model_name_of(model), train=train)
    model.eval()
    out, idx = [], np.repeat(rows, views)
    for start in range(0, len(idx), batch):
        part = idx[start : start + batch]
        x = torch.stack([tf(_load(photo_root / ds.samples[i].photo_relpath)) for i in part])
        out.append(model.lower(x).half())
        if (start // batch) % 20 == 0:
            log.info("cached {}/{} ({})", min(start + batch, len(idx)), len(idx), path.name)
    feats = torch.cat(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feats": feats, "rows": torch.from_numpy(idx)}, path)
    return FeatureCache(feats, idx)


def model_name_of(model: PhotoCarbModel) -> str:
    """timm architecture name of the backbone."""
    return model.backbone.pretrained_cfg.get("architecture", "convnext_tiny")


def _targets(ds: MealDataset, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """log1p(carbs), carb bin, and raw grams for each cached item."""
    g = np.array([ds.samples[i].carbs_served_g for i in rows], dtype=np.float32)
    gt = torch.from_numpy(g)
    return torch.log1p(gt), carb_bin(gt), g


def _loss(reg: torch.Tensor, logits: torch.Tensor, y_log: torch.Tensor, y_bin: torch.Tensor, hp: FinetuneCfg):
    """Huber on log-carbs plus weighted cross-entropy on the carb bin."""
    huber = nn.functional.huber_loss(reg, y_log, delta=hp.huber_delta)
    return huber + hp.bin_loss_weight * nn.functional.cross_entropy(logits, y_bin)


@torch.no_grad()
def predict_cache(model: PhotoCarbModel, cache: FeatureCache, emb: torch.Tensor | None = None):
    """Predicted grams and bin probabilities for cached items (uses precomputed embeddings when given)."""
    model.eval()
    if emb is None:
        emb = torch.cat([model.upper(cache.feats[i : i + 64].float()) for i in range(0, len(cache.rows), 64)])
    reg, logits = model.heads(emb)
    return torch.expm1(reg).clamp(min=0).numpy(), torch.softmax(logits, -1).numpy()


def metrics(grams_pred: np.ndarray, probs: np.ndarray, grams_true: np.ndarray) -> tuple[float, float, float]:
    """Carb MAE (g), Spearman rank correlation, and carb-bin macro-F1."""
    true_bin = carb_bin(torch.from_numpy(grams_true)).numpy()
    return (float(np.mean(np.abs(grams_pred - grams_true))), float(spearmanr(grams_pred, grams_true).statistic),
            float(f1_score(true_bin, probs.argmax(1), average="macro")))  # fmt: skip


@torch.no_grad()
def _embed(model: PhotoCarbModel, cache: FeatureCache) -> torch.Tensor:
    """Pooled embeddings with the current (frozen) stage 3: phase 1 trains heads on these directly."""
    model.eval()
    return torch.cat([model.upper(cache.feats[i : i + 64].float()) for i in range(0, len(cache.rows), 64)])


def _train(model: PhotoCarbModel, ds: MealDataset, train: FeatureCache, val: FeatureCache | None, hp: FinetuneCfg,
           phase: int, epochs: int, seed: int, run: str, elog: EpochLogger | None, start_epoch: int = 0,
           patience: int | None = None, schedule_epochs: int | None = None,
           ) -> tuple[int, float, list[float]]:  # fmt: skip
    """Train heads (phase 1, on cached embeddings) or stage 3 + heads (phase 2, on cached maps). Returns the best
    epoch, the last validation MAE, and the validation MAE after every epoch (early stopping on validation loss when
    `patience` is set). `schedule_epochs` sets the cosine length, so a run cut short follows the CV schedule."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.set_trainable(stage3=phase == 2)
    y_log, y_bin, _ = _targets(ds, train.rows)
    heads = list(model.reg_head.parameters()) + list(model.bin_head.parameters())
    groups = [{"params": heads, "lr": hp.lr_heads}]
    if phase == 2:
        stage = list(model.backbone.stages[3].parameters()) + list(model.backbone.head.parameters())
        groups.append({"params": stage, "lr": hp.lr_stage3})
    opt = torch.optim.AdamW(groups, weight_decay=hp.weight_decay)
    sched = warmup_cosine(opt, 1 if phase == 2 else 2, schedule_epochs or epochs)
    emb_train = _embed(model, train) if phase == 1 else None
    emb_val = _embed(model, val) if (val is not None and phase == 1) else None
    best, best_epoch, best_state, waited, maes = float("inf"), epochs, None, 0, []
    for e in range(1, epochs + 1):
        model.train()
        if phase == 1:
            model.backbone.eval()
        order, total = rng.permutation(len(train.rows)), 0.0
        for s in range(0, len(order), hp.batch_size):
            b = torch.from_numpy(order[s : s + hp.batch_size])
            emb = emb_train[b] if phase == 1 else model.upper(train.feats[b].float())
            reg, logits = model.heads(emb)
            loss = _loss(reg, logits, y_log[b], y_bin[b], hp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        lr = opt.param_groups[-1]["lr"]
        sched.step()
        vloss = vmae = None
        if val is not None:
            gp, pr = predict_cache(model, val, emb_val)
            vy_log, vy_bin, vg = _targets(ds, val.rows)
            with torch.no_grad():
                emb_v = emb_val if emb_val is not None else _embed(model, val)
                r, lg = model.heads(emb_v)
                vloss = float(_loss(r, lg, vy_log, vy_bin, hp))
            vmae = metrics(gp, pr, vg)[0]
            maes.append(vmae)
        if elog is not None:
            rec = EpochRecord(model=MODEL, run=run, epoch=start_epoch + e, lr=lr, train_loss=total / len(order),
                              val_loss=vloss, val_metric=vmae, val_metric_name="mae_g",
                              phase=f"phase{phase}")  # fmt: skip
            elog.log(rec)
        if patience is not None and vloss is not None:
            if vloss < best - 1e-4:
                best, best_epoch, waited = vloss, e, 0
                best_state = {k: v.clone() for k, v in model.trainable_state().items()}
            else:
                waited += 1
                if waited >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return best_epoch, (vmae if vmae is not None else float("nan")), maes


def finetune_photo(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> Path:
    """CV on training people chooses the phase-1 epochs, the phase-2 epochs and whether phase 2 helps at all (the
    validation people are never used to choose anything); then one final run; save a small checkpoint."""
    hp, out = cfg.training.finetune, Path(cfg.paths.outputs) / "training"
    log = logger.bind(stage="train", model=MODEL)
    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))  # never oversubscribe the CPU
    root, cache_dir = Path(cfg.data.cgmacros_root), Path(cfg.paths.feature_cache)
    base = PhotoCarbModel(hp.arch, pretrained=True, cache_dir=str(cfg.paths.weights_cache), dropout=hp.dropout)
    pristine = {k: v.clone() for k, v in base.trainable_state().items()}
    tr_rows = photo_rows(ds, set(sp.train), root)
    ev_rows = photo_rows(ds, set(sp.train) | set(sp.validation) | set(sp.test), root)
    log.info("photos: {} training (x{} augmented views), {} for evaluation", len(tr_rows), hp.views_per_photo,
             len(ev_rows))  # fmt: skip
    aug = build_cache(base, ds, tr_rows, root, hp.views_per_photo, True, cache_dir / "train_aug.pt", cfg.seed)
    evc = build_cache(base, ds, ev_rows, root, 1, False, cache_dir / "eval.pt", cfg.seed)
    elog = EpochLogger(out / f"{MODEL}_history.csv", Path(cfg.training.tensorboard_dir) / MODEL)

    def rows_of(people) -> set[int]:
        return {int(i) for i in ev_rows if ds.samples[i].person_id in set(people)}

    target = {s.person_id for s in ds.samples if s.group in cfg.cohort.target_groups}
    folds = stratified_person_folds(sorted(sp.train), target, hp.cv_folds, cfg.seed)
    p1_epochs, mae_p1, curves_p2 = [], [], []
    for k, held in enumerate(folds):
        fit_people = set(sp.train) - set(held)
        tr_c, va_c = aug.select(rows_of(fit_people)), evc.select(rows_of(held))
        base.load_state_dict(pristine, strict=False)
        be, _, _ = _train(base, ds, tr_c, va_c, hp, 1, hp.phase1_max_epochs, cfg.seed + k, f"cv-fold-{k}", elog,
                          patience=hp.patience)  # fmt: skip
        g, pr = predict_cache(base, va_c)
        mae_p1.append(metrics(g, pr, _targets(ds, va_c.rows)[2])[0])
        if hp.phase2_epochs:
            _, _, c = _train(base, ds, tr_c, va_c, hp, 2, hp.phase2_epochs, cfg.seed + k, f"cv-fold-{k}", elog,
                             start_epoch=be)  # fmt: skip
            curves_p2.append(c)
        p1_epochs.append(be)
        log.info("fold {}: phase-1 best epoch {}, held-out MAE phase1 {:.1f} g, phase2 per epoch {}", k, be,
                 mae_p1[-1], [round(m, 1) for m in curves_p2[-1]] if curves_p2 else "-")  # fmt: skip
    e1 = int(np.median(p1_epochs))
    # phase-2 length: the epoch with the lowest fold-averaged held-out MAE (1..phase2_epochs)
    mean_p2 = np.mean(curves_p2, axis=0) if curves_p2 else np.array([])
    e2 = int(np.argmin(mean_p2)) + 1 if len(mean_p2) else 0
    use_p2 = e2 > 0 and float(mean_p2[e2 - 1]) < float(np.mean(mae_p1))
    cv = {"phase1_best_epochs": p1_epochs, "phase1_epochs_chosen": e1, "cv_mae_phase1_g": mae_p1,
          "cv_mae_phase2_per_epoch_g": curves_p2, "cv_mae_phase2_mean_per_epoch_g": mean_p2.tolist(),
          "phase2_epochs_chosen": e2 if use_p2 else 0, "phase2_kept": use_p2, "n_train_photos": int(len(tr_rows)),
          "views_per_photo": hp.views_per_photo}  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{MODEL}_cv_summary.json").write_text(json.dumps(cv, indent=2))
    log.info("CV MAE phase1 {:.1f} g; phase2 per epoch {} -> phase 2 {}", np.mean(mae_p1),
             [round(float(m), 1) for m in mean_p2], f"kept for {e2} epochs" if use_p2 else "not kept")  # fmt: skip

    base.load_state_dict(pristine, strict=False)
    tr_c, va_c = aug, evc.select(rows_of(sp.validation))
    _, vmae, _ = _train(base, ds, tr_c, va_c, hp, 1, e1, cfg.seed, "final", elog)
    total = e1
    if use_p2:
        _, vmae, _ = _train(base, ds, tr_c, va_c, hp, 2, e2, cfg.seed, "final", elog, start_epoch=e1,
                            schedule_epochs=hp.phase2_epochs)  # fmt: skip
        total += e2
    elog.close()
    strategy = "frozen backbone (heads only)" + (" + stage 3 fine-tuned" if use_p2 else "")
    plot_curves(elog.records, "final", out / f"{MODEL}_curves.png", f"Photo model (ConvNeXt-Tiny): {strategy}",
                marker_epoch=e1 if use_p2 else None)  # fmt: skip
    ckpt = Path(cfg.paths.checkpoints) / f"{MODEL}_e{total:02d}_valmae{vmae:.1f}.pt"
    torch.save({"trainable_state": {k: v.half() for k, v in base.trainable_state().items()}, "arch": hp.arch,
                "phase2_kept": use_p2, "epochs": total, "val_mae_g": vmae, "hparams": hp.model_dump(),
                "seed": cfg.seed, "version": f"{MODEL}-e{total}"}, ckpt)  # fmt: skip
    (Path(cfg.paths.checkpoints) / f"{MODEL}_latest.txt").write_text(ckpt.name)
    log.info("checkpoint {} ({:.1f} MB); strategy: {}", ckpt, ckpt.stat().st_size / 1e6, strategy)
    return ckpt
