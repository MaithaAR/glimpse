"""v4 candidate study (docs/v4_within_person.md, registered 26 Nov before any run): a label audit, then two
candidates aimed at ranking meals within a person. Development people only; nothing served or frozen changes
(`glimpse v4-within-person`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn

from glimpse.data.augment import add_sensor_noise
from glimpse.data.normalise import Normaliser
from glimpse.data.splits import assert_not_locked, meal_rows
from glimpse.evaluation.report_metrics import with_intercept
from glimpse.models.sensor_cnn import SensorCNN
from glimpse.training.baselines import logistic, trend_features
from glimpse.training.common import warmup_cosine
from glimpse.training.cv import _tune_sklearn
from glimpse.training.fusion import PHOTO_COLS, photo_table
from glimpse.training.scratch import _inputs, predict_proba

OUT = Path("outputs/v4_within_person")
N_TREND, PAIRS, RANK_WEIGHT, GAIN, GUARD = 7, 512, 1.0, 0.03, 0.02


def _per_person_auc(d: pd.DataFrame, col: str) -> pd.Series:
    return pd.Series({p: roc_auc_score(g.spike, g[col]) for p, g in d.groupby("person_id") if g.spike.nunique() == 2})


def _endpoint(d: pd.DataFrame, cand: str, base: str) -> dict:
    """The registered primary endpoint: mean within-person AUROC, candidate minus baseline, people bootstrap."""
    a, b = _per_person_auc(d, cand), _per_person_auc(d, base)
    diff = (a - b).to_numpy()
    rng = np.random.default_rng(0)
    boot = [float(rng.choice(diff, len(diff)).mean()) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pooled = float(roc_auc_score(d.spike, d[cand]) - roc_auc_score(d.spike, d[base]))
    ok = bool(diff.mean() >= GAIN and lo > 0 and pooled >= -GUARD)
    return {"people": len(diff), "within_person_auroc": {"candidate": float(a.mean()), "baseline": float(b.mean())},
            "gain": float(diff.mean()), "ci95": [float(lo), float(hi)], "people_better": int((diff > 0).sum()),
            "people_worse": int((diff < 0).sum()),
            "pooled_auroc": {"candidate": float(roc_auc_score(d.spike, d[cand])),
                             "baseline": float(roc_auc_score(d.spike, d[base])), "difference": pooled},
            "verdict": "success" if ok else "not shown"}  # fmt: skip


def label_audit(ds, dev: pd.DataFrame) -> dict:
    """Part A, descriptive: single signals within a person, labels near the cut, rise against peak."""
    d = dev[dev.risk_raw.notna()].copy()
    tf = trend_features(ds, d.row.to_numpy())
    s = [ds.samples[int(r)] for r in d.row]
    d["carbs"] = [x.carbs_served_g if x.carbs_served_g is not None else np.nan for x in s]
    d["neg_last"], d["neg_slope30"] = -tf[:, 0], -tf[:, 1]
    d["neg_baseline"] = [-x.baseline_mgdl for x in s]
    d["rise"] = [x.rise_mgdl for x in s]
    d["peak"] = [x.baseline_mgdl + x.rise_mgdl for x in s]
    a1 = {}
    for c in ("carbs", "neg_last", "neg_slope30", "neg_baseline"):
        v = _per_person_auc(d[d[c].notna()], c)
        a1[c] = {"mean_within_person_auroc": float(v.mean()), "people": len(v)}
    near = (d.rise - 50).abs() <= 5
    by_person = near.groupby(d.person_id).mean()
    a2 = {"share_within_5_of_cut": float(near.mean()), "per_person_range": [float(by_person.min()),
                                                                           float(by_person.max())],
          "within_person_auroc_raw_all": float(_per_person_auc(d, "risk_raw").mean()),
          "within_person_auroc_raw_without_them": float(_per_person_auc(d[~near], "risk_raw").mean()),
          "within_person_auroc_shipped_all": float(_per_person_auc(d, "risk").mean())}  # fmt: skip
    rho = {c: float(np.mean([spearmanr(g.risk_raw, g[c]).statistic for _, g in d.groupby("person_id") if len(g) > 4]))
           for c in ("rise", "peak")}  # fmt: skip
    return {"meals": len(d), "A1_single_signals": a1, "A2_labels_near_cut": a2, "A3_mean_within_person_spearman": rho}


def centred_trend(ds, people: set[int]) -> dict[int, np.ndarray]:
    """Row -> the 7 trend numbers minus the mean over the same person's earlier meals (0 for the first)."""
    out = {}
    for p in people:
        rows = sorted((i for i, s in enumerate(ds.samples) if s.person_id == p), key=lambda i: ds.samples[i].meal_time)
        t = trend_features(ds, np.array(rows))[:, :N_TREND]
        for j, r in enumerate(rows):
            out[r] = t[j] - np.nanmean(t[:j], axis=0) if j else np.zeros(N_TREND)
    return out


def run_b1(ds, sp, cfg) -> pd.DataFrame:
    """Person-centred fusion beside the v1 raw fusion, in the fusion study's own loop."""
    pcv = Path(cfg.paths.outputs) / "photo_cv"
    p_oof, p_nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
    y, grid, nan = ds.labels(), cfg.baselines.logreg_c_grid, [float("nan")] * len(PHOTO_COLS)
    ct = centred_trend(ds, {s.person_id for s in ds.samples} - set(sp.locked))
    rows_out = []
    for k, fold in enumerate(sp.dev_folds):
        people = sp.training_people(k)
        assert_not_locked(list(people) + list(fold), sp.locked, f"v4 fold {k + 1}")
        tr, te = meal_rows(ds, set(people), scored_only=False), meal_rows(ds, set(fold), scored_only=True)
        pt = photo_table(p_nested[p_nested.fold == k + 1]) | photo_table(p_oof[p_oof.fold == k + 1])

        def base(d, rows, pt=pt):
            return np.hstack([trend_features(d, rows), np.array([pt.get(int(i), nan) for i in rows])])

        def cand(d, rows, base=base):
            return np.hstack([base(d, rows), np.array([ct[int(i)] for i in rows])])

        preds = {}
        for name, fx in (("v1_raw", base), ("b1", cand)):
            c, _ = _tune_sklearn(lambda c: logistic(c, cfg.seed), grid, fx, ds, people, sp.inner_folds[k], y)
            preds[name] = logistic(c, cfg.seed).fit(fx(ds, tr), y[tr]).predict_proba(fx(ds, te))[:, 1]
            preds[name + "_C"] = c
        for j, i in enumerate(te):
            rows_out.append({"fold": k + 1, "row": int(i), "person_id": ds.samples[i].person_id, "spike": int(y[i]),
                             "has_photo": int(i) in pt, "v1_raw": float(preds["v1_raw"][j]),
                             "b1": float(preds["b1"][j]), "b1_C": preds["b1_C"]})  # fmt: skip
    return pd.DataFrame(rows_out)


def _fit_rank(ds, rows, norm, cfg, epochs: int, seed: int) -> SensorCNN:
    """training/scratch.py's fit with the registered within-person pairwise term added to the loss."""
    hp = cfg.training.scratch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = ds.labels()[rows].astype(np.float32)
    pid = np.array([ds.samples[i].person_id for i in rows])
    pos, neg = [], []
    for p in np.unique(pid):
        a, b = np.where((pid == p) & (y == 1))[0], np.where((pid == p) & (y == 0))[0]
        if len(a) and len(b):
            aa, bb = np.meshgrid(a, b)
            pos.append(aa.ravel())
            neg.append(bb.ravel())
    pos, neg = np.concatenate(pos), np.concatenate(neg)
    model = SensorCNN(n_profile=ds.profile.shape[1], channels=hp.channels, hidden=hp.hidden, dropout=hp.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    sched = warmup_cosine(opt, hp.warmup_epochs, epochs)
    for _ in range(epochs):
        g, h = add_sensor_noise(ds.glucose[rows], ds.heart_rate[rows], cfg.augment, rng)
        x = norm.transform(g, h, ds.profile[rows])
        seq, prof = torch.from_numpy(x.sequence), torch.from_numpy(x.profile)
        model.train()
        order = rng.permutation(len(y))
        for start in range(0, len(y), hp.batch_size):
            b = order[start : start + hp.batch_size]
            loss = nn.functional.binary_cross_entropy_with_logits(model(seq[b], prof[b]), torch.from_numpy(y[b]))
            pick = rng.choice(len(pos), min(PAIRS, len(pos)), replace=False)
            sp_, sn_ = model(seq[pos[pick]], prof[pos[pick]]), model(seq[neg[pick]], prof[neg[pick]])
            loss = loss + RANK_WEIGHT * nn.functional.softplus(sn_ - sp_).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model


def run_b2(ds, sp, cfg) -> pd.DataFrame:
    """SensorCNN with the ranking term on the committed cross-validation's folds, epochs and seed 0."""
    torch.set_num_threads(min(cfg.training.num_threads, 2))
    oof = pd.read_csv("outputs/cv/oof_predictions.csv")
    oof = oof[oof.seed == 0].copy()
    epochs = {t["fold"]: t["cnn_epochs"] for t in json.loads(Path("outputs/cv/cv_summary.json").read_text())["tuning"]}
    everyone = {s.person_id for s in ds.samples} - set(sp.locked)
    out = {}
    for f, g in oof.groupby("fold"):
        people = everyone - set(g.person_id)
        assert_not_locked(list(people) + list(set(g.person_id)), sp.locked, f"v4 B2 fold {f}")
        tr, te = meal_rows(ds, people, scored_only=False), g.row.to_numpy()
        norm = Normaliser.fit(ds.glucose[tr], ds.heart_rate[tr], ds.profile[tr], ds.profile_names)
        model = _fit_rank(ds, tr, norm, cfg, epochs[int(f)], 0)
        out |= dict(zip(te.tolist(), predict_proba(model, _inputs(ds, te, norm)).tolist(), strict=True))
    oof["b2"] = oof.row.map(out)
    return oof[["fold", "row", "person_id", "spike", "sensor_cnn", "b2"]]


def run_v4_within_person(cfg, ds, sp) -> dict:
    log = logger.bind(stage="v4")
    OUT.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv("outputs/evaluation_v2/dev_answers.csv")
    res = {"what": __doc__.splitlines()[0], "registration": "docs/v4_within_person.md",
           "A_label_audit": label_audit(ds, dev)}  # fmt: skip
    log.info("A: {}", res["A_label_audit"])
    b1 = run_b1(ds, sp, cfg)
    ref = pd.read_csv("outputs/fusion/fusion_oof.csv").set_index("row").photo
    drift = float(np.abs(b1.set_index("row").v1_raw - ref.loc[b1.row].to_numpy()).max())
    res["B1_baseline_reproduces_fusion_oof"] = {"max_abs_difference": drift, "ok": drift < 1e-9}
    if drift >= 1e-9:
        (OUT / "v4_within_person.json").write_text(json.dumps(res, indent=2))
        raise SystemExit(f"the refitted v1 raw fusion differs from fusion_oof.csv by {drift:.2e}; stopped")
    a = b1[b1.row.isin(dev[dev.risk_raw.notna()].row)].copy()
    c_fold = b1.groupby("fold").b1_C.first().tolist()
    res["B1_person_centred_fusion"] = _endpoint(a, "b1", "v1_raw") | {"meals": len(a), "C_per_fold": c_fold}
    lg = {c: pd.Series(np.log(a[c] / (1 - a[c])).to_numpy(), index=a.row) for c in ("b1", "v1_raw")}
    wi = {c: with_intercept(dev, lg[c], platt=False) for c in lg}
    yy = a.set_index("row").spike
    res["B1_person_centred_fusion"]["with_intercept_replayed_no_platt"] = {
        c: float(roc_auc_score(yy.loc[wi[c].index], wi[c])) for c in wi
    }
    b1.to_csv(OUT / "b1_oof.csv", index=False)
    log.info("B1: {}", res["B1_person_centred_fusion"])
    b2 = run_b2(ds, sp, cfg)
    res["B2_sensor_cnn_ranking_loss"] = _endpoint(b2, "b2", "sensor_cnn") | {"meals": len(b2)}
    b2.to_csv(OUT / "b2_oof.csv", index=False)
    log.info("B2: {}", res["B2_sensor_cnn_ranking_loss"])
    (OUT / "v4_within_person.json").write_text(json.dumps(res, indent=2))
    return res
