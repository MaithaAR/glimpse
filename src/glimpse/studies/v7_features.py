"""Inputs for the v7 design (docs/v5_model_fix.md, Part 3): everything is known before the meal. Static table:
logged macronutrients, more of the profile, context and long-window CGM statistics. Photo table: protein, fat,
fibre and calories from ridge heads on the frozen photo features, fitted without the person."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA

OUT = Path("outputs/v5_model_fix")
MACROS = ["protein", "fat", "fiber", "kcal"]
ROOT = Path("data/cgmacros/CGMacros")


def static_table(ds) -> pd.DataFrame:
    """One row per dataset sample, in sample order."""
    M = pd.read_csv("data/meals.csv", parse_dates=["t"]).set_index(["sid", "t"])
    bio = pd.read_csv(ROOT / "bio.csv").rename(columns=lambda c: c.strip()).set_index("subject")
    raw = {}
    rows = []
    for s in ds.samples:
        m = M.loc[(s.person_id, pd.Timestamp(s.meal_time))]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        b = bio.loc[s.person_id]
        if s.person_id not in raw:
            d = pd.read_csv(ROOT / f"CGMacros-{s.person_id:03d}" / f"CGMacros-{s.person_id:03d}.csv",
                            parse_dates=["Timestamp"]).set_index("Timestamp")  # fmt: skip
            raw[s.person_id] = (pd.to_numeric(d["Dexcom GL"], errors="coerce").dropna(), d["Meal Type"].dropna())
        g, mt = raw[s.person_id]
        t = pd.Timestamp(s.meal_time)
        w4, w24 = g[t - pd.Timedelta(hours=4) : t], g[t - pd.Timedelta(hours=24) : t]
        near = mt[t - pd.Timedelta(minutes=5) : t + pd.Timedelta(minutes=5)]
        kind = str(near.iloc[0]).strip().lower() if len(near) else ""
        ins, fg = float(b["Insulin"]), float(b["Fasting GLU - PDL (Lab)"])
        rows.append({
            "protein": np.log1p(m.protein_served), "fat": np.log1p(m.fat_served), "fiber": np.log1p(m.fiber_served),
            "kcal": np.log1p(m.kcal_served),
            "carb_share": 4 * m.carbs_served / m.kcal_served if m.kcal_served > 0 else np.nan,
            "male": float(str(b["Gender"]).strip().upper().startswith("M")), "log_insulin": np.log(ins),
            "log_trig": np.log(float(b["Triglycerides"])), "hdl": float(b["HDL"]), "log_homa": np.log(ins * fg / 405),
            "since_meal": np.log1p(min(float(m.since_meal), 24.0)) if np.isfinite(m.since_meal) else np.nan,
            "mets60": m.mets60, "hr_slope": m.hr_slope,
            "breakfast": float(kind.startswith("break")), "lunch": float(kind.startswith("lunch")),
            "dinner": float(kind.startswith("dinner")),
            "g4_mean": w4.mean() if len(w4) > 20 else np.nan, "g4_sd": w4.std() if len(w4) > 20 else np.nan,
            "g4_max": w4.max() if len(w4) > 20 else np.nan,
            "g24_mean": w24.mean() if len(w24) > 100 else np.nan, "g24_sd": w24.std() if len(w24) > 100 else np.nan,
            "g24_above140": (w24 > 140).mean() if len(w24) > 100 else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows)


def photo_macros(ds, sp, cfg) -> np.ndarray:
    """[fold 1..5][sample][macro]: log(1 + estimate) from ridge on the frozen photo features. For fold k the
    held-out people get the model fitted on k's training people; training people get nested estimates from
    inner fits that excluded them. NaN where there is no photo."""
    from glimpse.models.photo_model import PhotoCarbModel
    from glimpse.models.weights import weights_path
    from glimpse.training.photo import build_cache, embed_maps, photo_rows
    from glimpse.training.photo_cv import _ridge, _ridge_predict, fold_plan_v2

    groups = {s.person_id: s.group for s in ds.samples}
    rows = photo_rows(ds, set(groups) - set(sp.locked), Path(cfg.data.cgmacros_root))
    person = np.array([ds.samples[i].person_id for i in rows])
    model = PhotoCarbModel(cfg.photo.arch, weights=weights_path(cfg.photo), dropout=cfg.photo.dropout)
    maps = build_cache(model, ds, rows, cfg, "maps_eval", Path("data/cache") / "photo_cv_maps_eval.pt")
    emb = embed_maps(model, maps).detach().numpy()
    st = static_table(ds)
    out = np.full((5, len(ds.samples), len(MACROS)), np.nan)
    alpha = 1000.0  # the committed carb ridge picked the grid's upper end in every fold; not re-tuned here
    for f in fold_plan_v2(sp, cfg):
        k = f["fold"] - 1
        pos_t, pos_h = np.where(np.isin(person, f["train"]))[0], np.where(np.isin(person, f["held"]))[0]
        for j, name in enumerate(MACROS):
            y = np.expm1(st[name].to_numpy()[rows])
            ok = np.isfinite(y)
            for i in f["inner"]:
                fit, hold = pos_t[~np.isin(person[pos_t], i["exclude"])], pos_t[np.isin(person[pos_t], i["exclude"])]
                fit = fit[ok[fit]]
                out[k, rows[hold], j] = np.log1p(_ridge_predict(_ridge(emb[fit], y[fit], alpha), emb[hold]))
            fit = pos_t[ok[pos_t]]
            out[k, rows[pos_h], j] = np.log1p(_ridge_predict(_ridge(emb[fit], y[fit], alpha), emb[pos_h]))
    return out


def microbiome_pcs(train_people: set[int], n: int = 3) -> pd.DataFrame:
    """Principal components of the gut-health scores, fitted on the training people only; index = subject."""
    g = pd.read_csv(ROOT / "gut_health_test.csv", encoding="utf-8-sig").set_index("subject")
    g = g.apply(pd.to_numeric, errors="coerce").fillna(2.0)
    tr = g.loc[[p for p in g.index if p in train_people]]
    mu = tr.mean()
    pca = PCA(n_components=n, random_state=0).fit(tr - mu)
    return pd.DataFrame(pca.transform(g - mu), index=g.index)


def build(ds, sp, cfg) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    static_table(ds).to_csv(OUT / "v7_static.csv", index=False)
    np.save(OUT / "v7_photo_macros.npy", photo_macros(ds, sp, cfg).astype(np.float32))
    logger.bind(stage="v7").info("v7 inputs written to {}", OUT)


# ---------------------------------------------------------------- self-supervised CGM forecaster (family 4)
def cgm_series(person: int) -> pd.Series:
    d = pd.read_csv(ROOT / f"CGMacros-{person:03d}" / f"CGMacros-{person:03d}.csv", parse_dates=["Timestamp"])
    g = pd.to_numeric(d.set_index("Timestamp")["Dexcom GL"], errors="coerce").dropna()
    return g.groupby(level=0).mean().resample("5min").mean()


def _windows(g: pd.Series, stride: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Unlabelled training pairs from one person's CGM: 2 h in (24 slots), the rise over the next 2 h out
    (peak minus the mean of the last 30 min), the meal label's own definition applied at every time."""
    v = g.to_numpy(float)
    X, y = [], []
    for i in range(24, len(v) - 24, stride):
        a, b = v[i - 24 : i], v[i : i + 24]
        if np.isfinite(a).sum() >= 20 and np.isfinite(b).sum() >= 18 and np.isfinite(a[-6:]).any():
            X.append(pd.Series(a).ffill().bfill().to_numpy())
            y.append(np.nanmax(b) - np.nanmean(a[-6:]))
    return np.array(X, np.float32), np.array(y, np.float32)


def _net():
    from torch import nn

    return nn.Sequential(nn.Conv1d(2, 16, 5, padding=2), nn.ReLU(), nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 1))  # fmt: skip


def _prep(X: np.ndarray):
    import torch

    x = (X - 120.0) / 40.0
    return torch.from_numpy(np.stack([x, x - x[:, -1:]], axis=1).astype(np.float32))  # level and shape channels


def train_forecaster(people: list[int], series: dict, epochs: int = 3, seed: int = 0):
    """A small 1-D CNN that forecasts the 2-hour rise from the 2 hours before, trained on every usable time of
    the given people's CGM (no meal labels, no meal times)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    parts = [_windows(series[p]) for p in people]
    X, y = _prep(np.concatenate([p[0] for p in parts])), torch.from_numpy(np.concatenate([p[1] for p in parts]) / 50.0)
    net, rng = _net(), np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), 256):
            b = order[s : s + 256]
            loss = nn.functional.huber_loss(net(X[b]).squeeze(1), y[b].clamp(-1, 4))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def forecast_feature(net, ds, rows: np.ndarray) -> np.ndarray:
    """The forecaster's 2-hour rise (in units of 50 mg/dL) for the given meals' pre-meal windows."""
    import torch

    g = pd.DataFrame(ds.glucose[rows]).ffill(axis=1).bfill(axis=1).to_numpy(np.float32)
    with torch.no_grad():
        return net(_prep(g)).squeeze(1).numpy()
