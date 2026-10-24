"""Train the photo model (served carbohydrate from the before-meal photo).

The backbone is pretrained and frozen, so each photo is passed through it ONCE and the result is saved (cache).
Phase 1: only the two heads learn, from the saved 768 numbers (seconds per run).
Phase 2 (trial): the backbone's last stage also learns, from saved stage-3 inputs with gentle augmentation.
It is kept only if it lowers the inner folds' carb error by at least `last_stage_min_gain_g` grams.
All choices are made inside the 28 training people; the 4 validation people are only watched for the curves."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch import nn

from glimpse.config import GlimpseConfig, PhotoTrainCfg
from glimpse.data.dataset import MealDataset
from glimpse.data.photos import eval_transform, open_photo, train_transform
from glimpse.data.splits import Splits
from glimpse.models.photo_model import PhotoCarbModel, carb_bin
from glimpse.models.weights import weights_path
from glimpse.training.common import EpochLogger, EpochRecord, plot_curves, warmup_cosine

MODEL = "photo_convnext"
CACHE_VERSION = 1


# ---------------------------------------------------------------- which photos
def photo_rows(ds: MealDataset, people: set[int], root: Path) -> np.ndarray:
    """Meals of these people that have a photo on disk and a carbohydrate label."""
    return np.array(
        [
            i
            for i, s in enumerate(ds.samples)
            if s.person_id in people
            and s.photo_relpath
            and s.carbs_served_g is not None
            and (root / s.photo_relpath).is_file()
        ],
        dtype=int,
    )


def grams(ds: MealDataset, rows: np.ndarray) -> np.ndarray:
    """Served carbohydrate labels (g)."""
    return np.array([ds.samples[i].carbs_served_g for i in rows], dtype=np.float32)


# ---------------------------------------------------------------- cache (one backbone pass per photo)
def cache_key(rows: np.ndarray, cfg: GlimpseConfig, kind: str) -> str:
    """Identity of a cache: which photos, which weights, which preparation. Any change -> recompute."""
    pc = cfg.photo
    raw = json.dumps(
        [
            CACHE_VERSION,
            kind,
            rows.tolist(),
            pc.weights_sha256,
            pc.arch,
            pc.crop_pct,
            cfg.training.photo.augmented_views,
        ]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@torch.no_grad()
def build_cache(
    model: PhotoCarbModel, ds: MealDataset, rows: np.ndarray, cfg: GlimpseConfig, kind: str, path: Path
) -> torch.Tensor:
    """Stage-3 inputs (384x14x14, half precision) of each photo: kind='maps_eval' = the test-time view,
    kind="maps_aug" = `augmented_views` augmented copies (training photos, phase 2 only). Reused if the key matches."""
    key = cache_key(rows, cfg, kind)
    if path.is_file():
        saved = torch.load(path, weights_only=True)
        if saved.get("key") == key:
            logger.bind(stage="cache").info("reusing {} ({} items)", path.name, len(saved["x"]))
            return saved["x"]
    root, pc = Path(cfg.data.cgmacros_root), cfg.photo
    views = cfg.training.photo.augmented_views if kind == "maps_aug" else 1
    tf = train_transform(pc.arch) if kind == "maps_aug" else eval_transform(pc.arch, pc.crop_pct)
    torch.manual_seed(cfg.seed)  # augmented copies are the same on every run
    model.eval()
    out, todo = [], np.repeat(rows, views)
    for start in range(0, len(todo), 32):
        part = todo[start : start + 32]
        x = torch.stack([tf(open_photo(root / ds.samples[i].photo_relpath, pc.min_side_px)) for i in part])
        maps = model.lower(x)
        out.append(maps.half())
        if (start // 32) % 10 == 0:
            logger.bind(stage="cache").info("{}: {}/{} photos", path.name, min(start + 32, len(todo)), len(todo))
    x = torch.cat(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"key": key, "x": x}, path)
    return x


# ---------------------------------------------------------------- loss and measures
def loss_fn(reg: torch.Tensor, logits: torch.Tensor, g: torch.Tensor, hp: PhotoTrainCfg) -> torch.Tensor:
    """Huber on log(1+grams) (robust to wild labels) + weighted cross-entropy on the carb bin."""
    huber = nn.functional.huber_loss(reg, torch.log1p(g), delta=hp.huber_delta)
    return huber + hp.bin_loss_weight * nn.functional.cross_entropy(logits, carb_bin(g))


def measures(pred_g: np.ndarray, bin_probs: np.ndarray, true_g: np.ndarray) -> dict[str, float]:
    """Carb error (MAE, g), rank agreement (Spearman) and carb-bin macro-F1."""
    true_bin = carb_bin(torch.from_numpy(true_g)).numpy()
    return {
        "mae_g": float(np.mean(np.abs(pred_g - true_g))),
        "spearman": float(spearmanr(pred_g, true_g).statistic) if np.ptp(pred_g) > 0 else 0.0,
        "bin_f1": float(f1_score(true_bin, bin_probs.argmax(1), average="macro", labels=[0, 1, 2, 3])),
    }


@torch.no_grad()
def predict(model: PhotoCarbModel, emb: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Grams (>= 0) and bin probabilities from summaries."""
    model.eval()
    reg, logits = model.heads(emb)
    return torch.expm1(reg).clamp(min=0).numpy(), torch.softmax(logits, 1).numpy()


@torch.no_grad()
def embed_maps(model: PhotoCarbModel, maps: torch.Tensor) -> torch.Tensor:
    """Summaries from saved stage-3 inputs with the CURRENT last stage (needed after phase 2)."""
    model.eval()
    return torch.cat([model.upper(maps[i : i + 64].float()) for i in range(0, len(maps), 64)])


def _evaluate(model, emb, g, hp) -> tuple[float, float]:
    """Validation loss and carb error."""
    model.eval()
    with torch.no_grad():
        reg, logits = model.heads(emb)
        loss = float(loss_fn(reg, logits, torch.from_numpy(g), hp))
    pred, probs = predict(model, emb)
    return loss, measures(pred, probs, g)["mae_g"]


# ---------------------------------------------------------------- phase 1 and phase 2
def train_heads(model, emb, g, hp, epochs, seed, run, elog=None, val=None, patience=None) -> int:
    """Phase 1: heads only, from saved summaries. With `patience`, keep the best epoch by validation loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.set_trainable(last_stage=False)
    params = list(model.reg_head.parameters()) + list(model.bin_head.parameters())
    opt = torch.optim.AdamW(params, lr=hp.lr_heads, weight_decay=hp.weight_decay)
    sched = warmup_cosine(opt, hp.warmup_epochs, epochs)
    gt = torch.from_numpy(g)
    best, best_epoch, best_state, waited = float("inf"), epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order, total = rng.permutation(len(g)), 0.0
        for s in range(0, len(g), hp.batch_size):
            b = torch.from_numpy(order[s : s + hp.batch_size])
            reg, logits = model.heads(emb[b])
            loss = loss_fn(reg, logits, gt[b], hp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        vl, vm = _evaluate(model, *val, hp) if val else (None, None)
        if elog:
            elog.log(
                EpochRecord(
                    model=MODEL,
                    run=run,
                    epoch=epoch,
                    lr=lr,
                    train_loss=total / len(g),
                    val_loss=vl,
                    val_metric=vm,
                    val_metric_name="mae_g",
                )
            )
        if patience is not None and vl is not None:
            if vl < best - 1e-4:
                best, best_epoch, waited = vl, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items() if "head" in k}
            else:
                waited += 1
                if waited >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return best_epoch


def train_last_stage(
    model, maps_aug, views, g, hp, epochs, seed, run, elog=None, val=None, start_epoch=0
) -> list[float]:
    """Phase 2: last stage + heads learn from saved stage-3 inputs; each epoch uses one random augmented copy
    per photo. Returns the validation carb error after each epoch (if `val` = (maps, grams) is given)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.set_trainable(last_stage=True)
    heads = list(model.reg_head.parameters()) + list(model.bin_head.parameters())
    stage = list(model.backbone.stages[-1].parameters()) + list(model.backbone.head.parameters())
    opt = torch.optim.AdamW(
        [{"params": heads, "lr": hp.lr_heads}, {"params": stage, "lr": hp.lr_last_stage}], weight_decay=hp.weight_decay
    )
    gt, n, maes = torch.from_numpy(g), len(g), []
    for e in range(1, epochs + 1):
        model.train()
        pick = rng.integers(0, views, n)  # which augmented copy of each photo this epoch
        order, total = rng.permutation(n), 0.0
        for s in range(0, n, 32):
            b = order[s : s + 32]
            x = maps_aug[torch.from_numpy(b * views + pick[b])].float()
            reg, logits = model.heads(model.upper(x))
            loss = loss_fn(reg, logits, gt[torch.from_numpy(b)], hp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        vl = vm = None
        if val is not None:
            vl, vm = _evaluate(model, embed_maps(model, val[0]), val[1], hp)
            maes.append(vm)
        if elog:
            elog.log(
                EpochRecord(
                    model=MODEL,
                    run=run,
                    epoch=start_epoch + e,
                    lr=hp.lr_last_stage,
                    train_loss=total / n,
                    val_loss=vl,
                    val_metric=vm,
                    val_metric_name="mae_g",
                )
            )
    model.set_trainable(last_stage=False)
    return maes


# ---------------------------------------------------------------- the full recipe on the fixed split
def _subset_views(maps_aug: torch.Tensor, pos: np.ndarray, views: int) -> torch.Tensor:
    """Augmented copies belonging to the photos at positions `pos` (copies of one photo are stored together)."""
    idx = (pos[:, None] * views + np.arange(views)[None, :]).ravel()
    return maps_aug[torch.from_numpy(idx)]


def _last_stage_state(model: PhotoCarbModel) -> dict:
    """Copy of the last stage + final norm (to undo a phase-2 trial)."""
    return {
        k: v.clone() for k, v in model.state_dict().items() if k.startswith(("backbone.stages.3.", "backbone.head."))
    }


def load_photo_bundle(cfg: GlimpseConfig, path: Path) -> tuple[PhotoCarbModel, dict]:
    """Pretrained weights (verified, local) + the trained parts from a photo checkpoint."""
    b = torch.load(path, weights_only=True)
    if b["backbone_sha256"] != cfg.photo.weights_sha256:
        raise ValueError("photo checkpoint was trained on different base weights")
    model = PhotoCarbModel(cfg.photo.arch, weights=weights_path(cfg.photo), dropout=cfg.photo.dropout)
    model.load_state_dict({k: v.float() for k, v in b["state"].items()}, strict=False)
    return model.eval(), b


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
    e1 = int(np.median(best_epochs))
    keep, e2 = False, 0
    if mae2:
        per_epoch = np.mean(mae2, axis=0)
        e2 = int(np.argmin(per_epoch)) + 1
        keep = float(np.mean(mae1) - per_epoch[e2 - 1]) >= hp.last_stage_min_gain_g
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
        f", {float(np.mean(mae2, axis=0)[e2 - 1]):.1f} g with last stage" if mae2 else "",
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
