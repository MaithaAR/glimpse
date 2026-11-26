"""R1-R8 (docs/robustness_v3.md, registered before any run): the course's remaining tests on development
people, and a regularised photo branch (v3) built only as a candidate. `glimpse robustness-v3` writes
outputs/robustness_v3/robustness_v3.json and the v3 out-of-fold carb estimates. Nothing frozen is read for
writing; the locked eight's outcomes are never read (R8 reads only their sex and glycaemic status)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image, ImageOps
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from glimpse.studies.final_checks import V1, _lr

OUT = Path("outputs/robustness_v3")
PHOTO_COLS = ["p_bin1", "p_bin2", "p_bin3"]
GRID = [{"wd": wd, "dropout": dr, "views": v} for wd in (1e-4, 1e-2, 1e-1) for dr in (0.2, 0.5) for v in (0, 2)]


class Heads(nn.Module):
    """The photo model's two heads (grams, carb range) on the frozen 768-number summary."""

    def __init__(self, width: int = 768, dropout: float = 0.2) -> None:
        super().__init__()
        self.reg = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 1))
        self.bins = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reg(x).squeeze(-1), self.bins(x)


def _loss(hp, reg, logits, g):
    from glimpse.training.photo import loss_fn

    return loss_fn(reg, logits, g, hp)


@torch.no_grad()
def _eval_loss(m: Heads, hp, x: torch.Tensor, g: torch.Tensor) -> float:
    m.eval()
    return float(_loss(hp, *m(x), g))


@torch.no_grad()
def _predict(m: Heads, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    m.eval()
    reg, logits = m(x)
    return torch.expm1(reg).clamp(min=0).numpy(), torch.softmax(logits, 1).numpy()


def _train(hp, x, g, conf: dict, epochs: int, seed: int, val=None, patience: int | None = None) -> dict:
    """Heads with AdamW, warm-up and cosine (the v1 recipe), weight decay and dropout from `conf`. With `val`,
    stop on validation loss and report the best epoch and the train-validation gap there."""
    from glimpse.training.common import warmup_cosine

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Heads(x.shape[1], conf["dropout"])
    opt = torch.optim.AdamW(m.parameters(), lr=hp.lr_heads, weight_decay=conf["wd"])
    sched = warmup_cosine(opt, hp.warmup_epochs, epochs)
    best, best_epoch, best_state, waited, gap = float("inf"), epochs, None, 0, None
    for epoch in range(1, epochs + 1):
        m.train()
        order, total = rng.permutation(len(g)), 0.0
        for s in range(0, len(g), hp.batch_size):
            b = torch.from_numpy(order[s : s + hp.batch_size])
            loss = _loss(hp, *m(x[b]), g[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        if val is not None:
            vl = _eval_loss(m, hp, *val)
            if vl < best - 1e-4:
                best, best_epoch, waited = vl, epoch, 0
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
                gap = vl - total / len(g)
            else:
                waited += 1
                if patience is not None and waited >= patience:
                    break
    if best_state is not None:
        m.load_state_dict(best_state)
    return {"model": m, "best_epoch": best_epoch, "val_loss": best, "gap": gap}


def _setup(cfg, ds, sp) -> dict:
    from glimpse.models.photo_model import PhotoCarbModel
    from glimpse.models.weights import weights_path
    from glimpse.training.photo import build_cache, embed_maps, grams, photo_rows
    from glimpse.training.photo_cv import fold_plan_v2

    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))
    dev = {s.person_id for s in ds.samples} - set(sp.locked)
    rows = photo_rows(ds, dev, Path(cfg.data.cgmacros_root))
    pc = cfg.photo
    model = PhotoCarbModel(pc.arch, weights=weights_path(pc), dropout=pc.dropout)
    cache = Path("data/cache")
    maps_eval = build_cache(model, ds, rows, cfg, "maps_eval", cache / "photo_cv_maps_eval.pt")
    maps_aug = build_cache(model, ds, rows, cfg, "maps_aug", cache / "photo_cv_maps_aug.pt")
    emb, emb_aug = embed_maps(model, maps_eval), embed_maps(model, maps_aug)
    person = np.array([ds.samples[i].person_id for i in rows])
    return {"rows": rows, "person": person, "g": torch.from_numpy(grams(ds, rows)), "emb": emb,
            "emb_aug": emb_aug.view(len(rows), -1, emb.shape[1]), "plan": fold_plan_v2(sp, cfg)}  # fmt: skip


def _stack(S: dict, pos: np.ndarray, views: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Training items: the test-time summary of each photo plus `views` augmented copies."""
    x, g = [S["emb"][pos]], [S["g"][pos]]
    for v in range(views):
        x.append(S["emb_aug"][pos, v])
        g.append(S["g"][pos])
    return torch.cat(x), torch.cat(g)


def _pos(S: dict, people, drop: set | None = None) -> np.ndarray:
    keep = np.isin(S["person"], list(people))
    if drop:
        keep &= ~np.isin(S["rows"], list(drop))
    return np.where(keep)[0]


def v3_oof(cfg, S: dict, seed: int, drop_by_fold: dict | None = None, fixed: list | None = None) -> tuple:
    """R5/R6/R7: per outer fold, choose the configuration on the 4 inner folds (mean validation carb MAE), train
    on all the fold's training people for the median best epoch, estimate the held-out people."""
    hp, parts, chosen = cfg.training.photo, [], []
    for i, f in enumerate(S["plan"]):
        drop = (drop_by_fold or {}).get(f["fold"])
        if fixed is None:
            scores = []
            for conf in GRID:
                maes, eps, gaps = [], [], []
                for inner in f["inner"]:
                    fit = _pos(S, set(f["train"]) - set(inner["exclude"]), drop)
                    hold = _pos(S, inner["select"])
                    r = _train(hp, *_stack(S, fit, conf["views"]), conf, hp.max_epochs, seed,
                               (S["emb"][hold], S["g"][hold]), hp.patience)  # fmt: skip
                    pred, _ = _predict(r["model"], S["emb"][hold])
                    maes.append(float(np.mean(np.abs(pred - S["g"][hold].numpy()))))
                    eps.append(r["best_epoch"])
                    gaps.append(r["gap"])
                scores.append({**conf, "mae": float(np.mean(maes)), "epochs": int(np.median(eps)),
                               "gap": float(np.mean(gaps))})  # fmt: skip
            best = min(scores, key=lambda s: s["mae"])
        else:
            best = fixed[i]
        chosen.append(best)
        fit, held = _pos(S, f["train"], drop), _pos(S, f["held"])
        r = _train(hp, *_stack(S, fit, best["views"]), best, best["epochs"], seed)
        pred, probs = _predict(r["model"], S["emb"][held])
        parts.append(pd.DataFrame({"fold": f["fold"], "row": S["rows"][held], "photo_g": pred,
                                   **{f"p_bin{j}": probs[:, j] for j in range(4)}}))  # fmt: skip
    return pd.concat(parts, ignore_index=True), chosen


# ---------------------------------------------------------------- the harness (S1/S2's fusion, no intercept)
def _harness_frame(ds) -> pd.DataFrame:
    from glimpse.training.baselines import trend_features

    dev = pd.read_csv(V1).dropna(subset=["risk"])
    oof = pd.read_csv("outputs/photo_cv/photo_oof.csv")
    dev = dev[dev.row.isin(oof.row)].reset_index(drop=True)
    dev.attrs["trend"] = trend_features(ds, dev.row.to_numpy())
    return dev


def harness(dev: pd.DataFrame, photo: pd.DataFrame | None, seed: int, drop_by_fold: dict | None = None) -> np.ndarray:
    """Out-of-fold fusion risk for every development meal; photo=None leaves the photo inputs out."""
    x = dev.attrs["trend"]
    if photo is not None:
        o = photo.set_index("row").loc[dev.row]
        x = np.hstack([x, np.c_[np.log1p(o.photo_g.to_numpy()), o[PHOTO_COLS].to_numpy()]])
    y, fold, p = dev.spike.to_numpy().astype(int), dev.fold.to_numpy(), np.full(len(dev), np.nan)
    for k in np.unique(fold):
        tr = fold != k
        if drop_by_fold:
            tr &= ~dev.row.isin(drop_by_fold.get(int(k), set())).to_numpy()
        p[fold == k] = _lr(seed).fit(x[tr], y[tr]).predict_proba(x[fold == k])[:, 1]
    return p


def paired(dev: pd.DataFrame, a: np.ndarray, b: np.ndarray, draws: int = 2000, seed: int = 0) -> dict:
    """AUROC(a) - AUROC(b), person-clustered bootstrap."""
    y, pid = dev.spike.to_numpy().astype(int), dev.person_id.to_numpy()
    people, rng, d = np.unique(pid), np.random.default_rng(seed), []
    idx = {q: np.where(pid == q)[0] for q in people}
    for _ in range(draws):
        i = np.concatenate([idx[q] for q in rng.choice(people, len(people))])
        if len(set(y[i])) == 2:
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return {"point": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}  # fmt: skip


def _carb(dev_rows: np.ndarray, df: pd.DataFrame, ds) -> dict:
    from glimpse.training.photo import measures

    d = df.set_index("row").loc[dev_rows]
    g = np.array([ds.samples[i].carbs_served_g for i in dev_rows], dtype=np.float32)
    return measures(d.photo_g.to_numpy(), d[[f"p_bin{j}" for j in range(4)]].to_numpy(), g)


# ---------------------------------------------------------------- R1, R2: training sanity
def _sensor_setup(ds, sp):
    from glimpse.data.normalise import Normaliser
    from glimpse.data.splits import meal_rows

    rows = meal_rows(ds, set(sp.training_people(0)), scored_only=False)
    norm = Normaliser.fit(ds.glucose[rows], ds.heart_rate[rows], ds.profile[rows], ds.profile_names)
    x = norm.transform(ds.glucose[rows], ds.heart_rate[rows], ds.profile[rows])
    return (torch.from_numpy(x.sequence), torch.from_numpy(x.profile),
            torch.from_numpy(ds.labels()[rows].astype(np.float32)))  # fmt: skip


def _sensor_model(cfg, ds):
    from glimpse.models.sensor_cnn import SensorCNN

    hp = cfg.training.scratch
    return SensorCNN(n_profile=ds.profile.shape[1], channels=hp.channels, hidden=hp.hidden, dropout=hp.dropout)


def r1_r2(cfg, ds, sp, S: dict) -> dict:
    hp = cfg.training.photo
    fit = _pos(S, S["plan"][0]["train"])
    xp, gp = S["emb"][fit], S["g"][fit]
    seq, prof, ys = _sensor_setup(ds, sp)

    def photo_step(m):
        def step(b):
            return _loss(hp, *m(xp[b]), gp[b])

        return step

    def sensor_step(m):
        def step(b):
            return nn.functional.binary_cross_entropy_with_logits(m(seq[b], prof[b]), ys[b])

        return step

    def range_test(model, step, n: int, used: float) -> dict:
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.0)
        lrs, losses, smooth, low = np.logspace(-6, 0, 200), [], None, float("inf")
        for lr in lrs:
            for gr in opt.param_groups:
                gr["lr"] = float(lr)
            model.train()
            loss = step(torch.from_numpy(rng.choice(n, 32, replace=False)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            smooth = loss.item() if smooth is None else 0.9 * smooth + 0.1 * loss.item()
            losses.append(smooth)
            low = min(low, smooth)
            if smooth > 4 * low:
                break
        at = float(lrs[int(np.argmin(losses))])
        return {"lr_at_min_loss": at, "suggested": at / 10, "used": used,
                "used_within_10x": bool(at / 100 <= used <= at)}  # fmt: skip

    def one_batch(model, step, n: int) -> dict:
        torch.manual_seed(cfg.seed)
        b = torch.from_numpy(np.random.default_rng(cfg.seed).choice(n, 32, replace=False))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        start, norms = None, []
        for t in range(300):
            model.train()
            loss = step(b)
            opt.zero_grad()
            loss.backward()
            norms.append(float(torch.sqrt(sum((q.grad**2).sum() for q in model.parameters() if q.grad is not None))))
            opt.step()
            start = loss.item() if t == 0 else start
        model.eval()
        with torch.no_grad():
            end = step(b).item()
        return {"start_loss": start, "end_loss": end, "fits": bool(end < 0.1 * start),
                "grad_norm_first": norms[0], "grad_norm_last": norms[-1], "grad_norm_max": max(norms)}  # fmt: skip

    return {"_photo": (photo_step, len(gp)), "_sensor": (sensor_step, len(ys)), "_fns": (range_test, one_batch)}


def run_r1_r2(cfg, ds, sp, S: dict) -> dict:
    hp = cfg.training.photo
    parts = r1_r2(cfg, ds, sp, S)
    range_test, one_batch = parts["_fns"]
    photo_step, n_p = parts["_photo"]
    sensor_step, n_s = parts["_sensor"]
    m = Heads(dropout=cfg.photo.dropout)
    r1p = range_test(m, photo_step(m), n_p, hp.lr_heads)
    m = _sensor_model(cfg, ds)
    r1s = range_test(m, sensor_step(m), n_s, cfg.training.scratch.lr)
    m = Heads(dropout=cfg.photo.dropout)
    r2p = one_batch(m, photo_step(m), n_p)
    m = _sensor_model(cfg, ds)
    r2s = one_batch(m, sensor_step(m), n_s)
    return {"r1_lr_range": {"photo_heads": r1p, "sensor_cnn": r1s},
            "r2_one_batch": {"photo_heads": r2p, "sensor_cnn": r2s}}  # fmt: skip


# ---------------------------------------------------------------- R3: label audit
def r3_label_audit(ds) -> dict:
    dev = pd.read_csv(V1).dropna(subset=["risk"]).reset_index(drop=True)
    p = dev.risk.clip(1e-6, 1 - 1e-6)
    dev["loss"] = -(dev.spike * np.log(p) + (1 - dev.spike) * np.log(1 - p))
    meals = pd.read_csv("data/meals.csv", parse_dates=["t"])
    key = {(int(r.sid), r.t): r for r in meals.itertuples()}

    def flags(row: int) -> dict:
        s = ds.samples[row]
        r = key[(s.person_id, pd.Timestamp(s.meal_time))]
        return {"boundary_rise": bool(abs(r.rise - 50) <= 5),
                "poor_coverage": bool(r.coverage < 0.9 or r.max_gap > 15),
                "partial_eating": bool(pd.notna(r.carbs_consumed) and pd.notna(r.carbs_served)
                                       and r.carbs_consumed < r.carbs_served),
                "high_baseline": bool(r.base >= 140)}  # fmt: skip

    f = pd.DataFrame([flags(int(r)) for r in dev.row])
    worst = dev.loss.nlargest(20).index
    res = {c: {"worst_20": float(f.loc[worst, c].mean()), "all_development": float(f[c].mean())} for c in f}
    for c in res:
        res[c]["enrichment"] = res[c]["worst_20"] / res[c]["all_development"] if res[c]["all_development"] else None
    examples = dev.loc[worst, ["row", "person_id", "risk", "spike"]].head(5).to_dict(orient="records")
    return {"meals": int(len(dev)), "flags": res, "worst_examples": examples,
            "worst_20_spiked": int(dev.loc[worst, "spike"].sum())}  # fmt: skip


# ---------------------------------------------------------------- R4: classical photo baseline
def _hsv_hist(path: Path) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((224, 224)).convert("HSV")
    a = np.asarray(im).reshape(-1, 3)
    return np.concatenate([np.histogram(a[:, c], bins=8, range=(0, 256))[0] for c in range(3)]) / len(a)


def r4_classical(cfg, ds, S: dict) -> pd.DataFrame:
    root = Path(cfg.data.cgmacros_root)
    feats = np.stack([_hsv_hist(root / ds.samples[i].photo_relpath) for i in S["rows"]])
    g, parts = S["g"].numpy(), []
    for f in S["plan"]:
        fit, held = _pos(S, f["train"]), _pos(S, f["held"])
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(feats[fit], np.log1p(g[fit]))
        pred = np.clip(np.expm1(m.predict(feats[held])), 0, None)
        parts.append(pd.DataFrame({"row": S["rows"][held], "photo_g": pred}))
    df = pd.concat(parts, ignore_index=True)
    from glimpse.training.photo_cv import _one_hot_bins

    oh = _one_hot_bins(df.photo_g.to_numpy())
    for j in range(4):
        df[f"p_bin{j}"] = oh[:, j]
    return df


# ---------------------------------------------------------------- R6 drop sets, R8 re-weighting
def d8_drops(ds, S: dict, dev: pd.DataFrame) -> tuple[dict, dict]:
    """Per outer fold: photo rows and development meal rows whose exact macro profile appears among the
    held-out people's meals (the D8 rule)."""
    from glimpse.studies.ablations import _macro_map

    mm = _macro_map()

    def key(i: int):
        s = ds.samples[i]
        return mm.get((s.person_id, pd.Timestamp(s.meal_time)))

    photo_drop, fusion_drop = {}, {}
    for f in S["plan"]:
        held_keys = {key(i) for i, s in enumerate(ds.samples) if s.person_id in set(f["held"])} - {None}
        photo_drop[f["fold"]] = {int(r) for r in S["rows"] if key(int(r)) in held_keys}
        fusion_drop[f["fold"]] = {int(r) for r in dev.row if key(int(r)) in held_keys}
    return photo_drop, fusion_drop


def r8_reweighted(sp) -> dict:
    bio = pd.read_csv("data/cgmacros/CGMacros/bio.csv").set_index("subject")
    sex = bio.iloc[:, [c.lower().startswith(("gender", "sex")) for c in bio.columns]].iloc[:, 0]
    dev = pd.read_csv(V1).dropna(subset=["risk"]).reset_index(drop=True)
    dev["sex"] = dev.person_id.map(sex)
    meals = pd.read_csv("data/meals.csv").drop_duplicates("sid").set_index("sid")
    dev["status"] = dev.person_id.map(meals.group)
    locked = pd.DataFrame({"sex": [sex.get(p) for p in sp.locked], "status": [meals.group.get(p) for p in sp.locked]})
    target = locked.value_counts(normalize=True)
    have = dev.groupby(["sex", "status"]).size() / len(dev)
    w = dev.apply(lambda r: target.get((r.sex, r.status), 0.0) / have[(r.sex, r.status)], axis=1).to_numpy()
    y = dev.spike.to_numpy().astype(int)
    cleared = (dev.decision == "low risk").to_numpy()
    return {"locked_mix": {f"{a}/{b}": float(v) for (a, b), v in target.items()},
            "auroc_unweighted": float(roc_auc_score(y, dev.risk)),
            "auroc_reweighted": float(roc_auc_score(y, dev.risk, sample_weight=w)),
            "spikes_among_clears_reweighted": float(np.sum(w * cleared * y) / max(np.sum(w * cleared), 1e-9)),
            "cells_missing": [f"{a}/{b}" for (a, b) in target.index if (a, b) not in have]}  # fmt: skip


# ---------------------------------------------------------------- run
def run_robustness_v3(cfg, ds, sp) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    log = logger.bind(stage="robustness-v3")
    S = _setup(cfg, ds, sp)
    dev = _harness_frame(ds)
    res: dict = {"what": __doc__.splitlines()[0], "photos": int(len(S["rows"])), "harness_meals": int(len(dev))}
    res.update(run_r1_r2(cfg, ds, sp, S))
    log.info("R1/R2: {}", {k: res[k] for k in ("r1_lr_range", "r2_one_batch")})
    res["r3_label_audit"] = r3_label_audit(ds)
    log.info("R3: {}", res["r3_label_audit"]["flags"])

    v1 = pd.read_csv("outputs/photo_cv/photo_oof.csv")
    ridge = v1[["row", "ridge_g"]].rename(columns={"ridge_g": "photo_g"})
    rows_eval = np.intersect1d(v1.row, dev.row)
    classical = r4_classical(cfg, ds, S)
    res["r4_classical"] = {"classical_colour_ridge": _carb(rows_eval, classical, ds),
                           "frozen_summaries_ridge_mae_g": float(np.mean(np.abs(
                               ridge.set_index("row").loc[rows_eval].photo_g.to_numpy()
                               - np.array([ds.samples[i].carbs_served_g for i in rows_eval])))),
                           "v1_fine_tuned": _carb(rows_eval, v1, ds),
                           "zero_shot": "not run: no network to fetch CLIP weights in this environment"}  # fmt: skip
    log.info("R4: {}", res["r4_classical"])

    v3, chosen = v3_oof(cfg, S, cfg.seed)
    v3.to_csv(OUT / "v3_photo_oof.csv", index=False)
    p_v1, p_v3, p_none = (harness(dev, v1, cfg.seed), harness(dev, v3, cfg.seed), harness(dev, None, cfg.seed))
    y = dev.spike.to_numpy().astype(int)
    res["r5_v3"] = {"chosen_per_fold": chosen, "carb_v3": _carb(rows_eval, v3, ds), "carb_v1": _carb(rows_eval, v1, ds),
                    "gap_v3_mean": float(np.mean([c["gap"] for c in chosen])), "gap_v1": 0.35,
                    "harness_auroc": {"v1": float(roc_auc_score(y, p_v1)), "v3": float(roc_auc_score(y, p_v3)),
                                      "no_photo": float(roc_auc_score(y, p_none))},
                    "v3_minus_v1": paired(dev, p_v3, p_v1), "v1_photo_gain": paired(dev, p_v1, p_none),
                    "v3_photo_gain": paired(dev, p_v3, p_none)}  # fmt: skip
    log.info("R5: {}", {k: res["r5_v3"][k] for k in ("gap_v3_mean", "harness_auroc", "v3_minus_v1")})

    photo_drop, fusion_drop = d8_drops(ds, S, dev)
    v3d8, _ = v3_oof(cfg, S, cfg.seed, drop_by_fold=photo_drop, fixed=chosen)
    v3d8.to_csv(OUT / "v3_d8_photo_oof.csv", index=False)
    pd8_v3, pd8_v1 = harness(dev, v3d8, cfg.seed, fusion_drop), harness(dev, v1, cfg.seed, fusion_drop)
    pd8_none = harness(dev, None, cfg.seed, fusion_drop)
    res["r6_d8"] = {"photo_training_dropped_per_fold": {k: len(v) for k, v in photo_drop.items()},
                    "fusion_training_dropped_per_fold": {k: len(v) for k, v in fusion_drop.items()},
                    "carb_v3_d8": _carb(rows_eval, v3d8, ds),
                    "photo_gain_v3_d8": paired(dev, pd8_v3, pd8_none),
                    "photo_gain_v1_photo_d8_fusion": paired(dev, pd8_v1, pd8_none)}  # fmt: skip
    log.info("R6: {}", res["r6_d8"])

    seeds = []
    for s in range(5):
        df, _ = v3_oof(cfg, S, cfg.seed + 1 + s, fixed=chosen)
        seeds.append(float(roc_auc_score(y, harness(dev, df, cfg.seed))))
    gain = res["r5_v3"]["v3_photo_gain"]["point"]
    res["r7_seeds"] = {"harness_auroc": seeds, "range": float(max(seeds) - min(seeds)), "photo_gain": gain,
                       "range_at_least_half_gain": bool(max(seeds) - min(seeds) >= abs(gain) / 2)}  # fmt: skip
    log.info("R7: {}", res["r7_seeds"])
    res["r8_reweighted"] = r8_reweighted(sp)
    log.info("R8: {}", res["r8_reweighted"])
    (OUT / "robustness_v3.json").write_text(json.dumps(res, indent=2, default=float))
    return res
