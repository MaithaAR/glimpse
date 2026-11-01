"""S6 ablations. D8: the recurring-meal risk the proposal flagged -- training meals whose exact macro
profile also appears among the held-out people's meals are removed from that fold's training set, and the
photo arm's gain is re-estimated. Channels: the photo arm refitted with one input channel removed at a
time. Both reuse each fold's already-chosen C (stored by `just fusion`), so the ONLY thing that changes
is the manipulated ingredient; nothing is retuned. Report: §3 ablations, next to the main gain."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.meal_table import MACROS
from glimpse.data.splits import SplitsV2, assert_not_locked, meal_rows
from glimpse.evaluation.metrics import paired_person_bootstrap
from glimpse.training.baselines import logistic, trend_features
from glimpse.training.fusion import PHOTO_COLS, photo_table, with_table

OUT = Path("outputs/ablations")
# trend_features columns: 6 trend numbers, heart rate, then the 8 profile inputs; the photo block is last
CHANNELS = {"full": [], "no_heart_rate": [6], "no_profile": list(range(7, 15))}


def macro_key(vals) -> tuple:
    """A meal's macro profile as a comparable key (rounded; missing values compare equal)."""
    return tuple(None if pd.isna(v) else round(float(v), 1) for v in vals)


def _macro_map(path: str | Path = "data/meals.csv") -> dict:
    """Meal (person, time) -> its served macro profile, as the proposal's recurring-meal analysis defined it."""
    m = pd.read_csv(path, parse_dates=["t"])
    cols = [f"{c}_served" for c in MACROS]
    return {(int(r.sid), pd.Timestamp(r.t)): macro_key([getattr(r, c) for c in cols]) for r in m.itertuples()}


def _stored_c(arm: str) -> dict[int, float]:
    tun = json.loads(Path("outputs/fusion/fusion_summary.json").read_text())["tuning"]
    return {int(t["fold"]): float(t[f"{arm}_C"]) for t in tun}


def _tables(cfg: GlimpseConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    pcv = Path(cfg.paths.outputs) / "photo_cv"
    return pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")


def _gain(df: pd.DataFrame, a: str, b: str, B: int, seed: int) -> dict:
    from sklearn.metrics import roc_auc_score

    y, s = df.spike.to_numpy(), df.person_id.to_numpy()
    _, lo, hi = paired_person_bootstrap(y, df[a].to_numpy(), df[b].to_numpy(), s, B=B, seed=seed)
    return {"point": float(roc_auc_score(y, df[a]) - roc_auc_score(y, df[b])), "ci95": [lo, hi]}


def run_profile_excluded(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig) -> dict:
    """D8: refit the photo and no-meal arms per fold after dropping every training meal whose macro profile
    also appears among that fold's held-out people (all their logged meals, not only scored ones)."""
    log, mm, y = logger.bind(stage="ablate"), _macro_map(), ds.labels()
    p_oof, p_nested = _tables(cfg)
    c_photo, c_none = _stored_c("photo"), _stored_c("no_meal")
    rows_out, dropped = [], {}
    for k, fold in enumerate(sp.dev_folds):
        kk = k + 1
        assert_not_locked(list(sp.training_people(k)) + list(fold), sp.locked, f"D8 fold {kk}")
        pt = photo_table(p_nested[p_nested.fold == kk]) | photo_table(p_oof[p_oof.fold == kk])
        held_keys = {mm[key] for key in mm if key[0] in set(fold)}
        tr = meal_rows(ds, set(sp.training_people(k)), scored_only=False)
        keep = np.array([i for i in tr
                         if mm.get((ds.samples[i].person_id, pd.Timestamp(ds.samples[i].meal_time))) not in held_keys])
        dropped[kk] = {"training_meals": int(len(tr)), "dropped": int(len(tr) - len(keep))}
        te = meal_rows(ds, set(fold), scored_only=True)
        fx = {"photo": with_table(pt, len(PHOTO_COLS)), "no_meal": trend_features}
        cs = {"photo": c_photo[kk], "no_meal": c_none[kk]}
        pred = {a: logistic(cs[a], cfg.seed).fit(f(ds, keep), y[keep]).predict_proba(f(ds, te))[:, 1]
                for a, f in fx.items()}  # fmt: skip
        for j, i in enumerate(te):
            rows_out.append({"fold": kk, "row": int(i), "person_id": ds.samples[i].person_id, "spike": int(y[i]),
                             "has_photo": int(i) in pt, "photo": float(pred["photo"][j]),
                             "no_meal": float(pred["no_meal"][j])})  # fmt: skip
    df = pd.DataFrame(rows_out)
    B, seed = cfg.fusion.bootstrap_draws, cfg.seed
    res = {"what": "D8 profile-excluded split: training meals sharing an exact macro profile with a held-out "
                   "person removed; same folds, same C as the main arms, nothing retuned",
           "dropped_per_fold": dropped,
           "share_dropped": float(sum(d["dropped"] for d in dropped.values())
                                  / sum(d["training_meals"] for d in dropped.values())),
           "gain_photo_vs_no_meal": _gain(df, "photo", "no_meal", B, seed),
           "gain_photo_meals_only": _gain(df[df.has_photo.astype(bool)], "photo", "no_meal", B, seed)}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "d8_profile_excluded_oof.csv", index=False)
    (OUT / "d8_profile_excluded.json").write_text(json.dumps(res, indent=2))
    g = res["gain_photo_meals_only"]
    log.info("D8 profile-excluded: {:.0%} of training meals dropped; photo gain {:+.3f} ({:+.3f} to {:+.3f})",
             res["share_dropped"], g["point"], *g["ci95"])  # fmt: skip
    return res


def run_channel_ablation(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig) -> dict:
    """The photo arm refitted with one input channel removed (heart rate; the 8 profile inputs), same folds,
    same C. The no-photo case is already the fusion's no_meal arm; the no-CGM case fails the product's own
    input checks, so it is not a meaningful variant."""
    from sklearn.metrics import roc_auc_score

    log, y = logger.bind(stage="ablate"), ds.labels()
    p_oof, p_nested = _tables(cfg)
    c_photo = _stored_c("photo")
    rows_out = []
    for k, fold in enumerate(sp.dev_folds):
        kk = k + 1
        pt = photo_table(p_nested[p_nested.fold == kk]) | photo_table(p_oof[p_oof.fold == kk])
        base = with_table(pt, len(PHOTO_COLS))
        tr = meal_rows(ds, set(sp.training_people(k)), scored_only=False)
        te = meal_rows(ds, set(fold), scored_only=True)
        pred = {}
        for name, cols in CHANNELS.items():
            fx = base if not cols else (lambda d, r, c=cols, b=base: np.delete(b(d, r), c, axis=1))
            pred[name] = logistic(c_photo[kk], cfg.seed).fit(fx(ds, tr), y[tr]).predict_proba(fx(ds, te))[:, 1]
        for j, i in enumerate(te):
            rows_out.append({"fold": kk, "row": int(i), "person_id": ds.samples[i].person_id, "spike": int(y[i]),
                             **{n: float(pred[n][j]) for n in CHANNELS}})  # fmt: skip
    df = pd.DataFrame(rows_out)
    B, seed = cfg.fusion.bootstrap_draws, cfg.seed
    res = {"what": "photo arm with one input channel removed; same folds, same C, nothing retuned",
           "auroc": {n: float(roc_auc_score(df.spike, df[n])) for n in CHANNELS},
           "delta_vs_full": {n: _gain(df, n, "full", B, seed) for n in CHANNELS if n != "full"}}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "channels_oof.csv", index=False)
    (OUT / "channels.json").write_text(json.dumps(res, indent=2))
    for n, v in res["delta_vs_full"].items():
        log.info("channel ablation {:<14} AUROC {:.3f} | delta vs full {:+.3f} ({:+.3f} to {:+.3f})", n,
                 res["auroc"][n], v["point"], *v["ci95"])  # fmt: skip
    return res
