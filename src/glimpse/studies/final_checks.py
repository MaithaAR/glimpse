"""Registered last-day checks (`glimpse final-checks`; docs/final_switch.md): O1 photo OOD detection, O2 out-of-scope
inputs, M1 monitoring power against volume, C1 calibration under an adaptive intercept, S1 shuffled labels, S2
learning curve, S3 slices, S4 identity shortcut, S5 train against validation curves. Descriptive; nothing frozen
changes and the locked eight are not touched. Writes outputs/final_checks/final_checks.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("outputs/final_checks/final_checks.json")
V1 = Path("outputs/evaluation_v2/dev_answers.csv")
V2 = Path("outputs/evaluation_sensor_first/dev_answers.csv")


def _generated_non_meals(seed: int) -> list[Image.Image]:
    """24 non-meal pictures a user might send by mistake: text pages, app screens, patterns, plain colours, noise."""
    rng = np.random.default_rng(seed)
    out = []
    for _i in range(6):  # text pages (receipts, documents)
        im = Image.new("RGB", (480, 640), (250, 250, 245))
        d = ImageDraw.Draw(im)
        for y in range(30, 620, 22):
            d.text((20, y), "".join(rng.choice(list("ABCDEFGHIJ 0123456789.,$")) for _ in range(40)), fill=(20, 20, 20))
        out.append(im)
    for _i in range(6):  # app screens: bars, buttons, a header
        im = Image.new("RGB", (390, 844), tuple(int(v) for v in rng.integers(200, 255, 3)))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 390, 90], fill=tuple(int(v) for v in rng.integers(0, 120, 3)))
        for y in range(120, 800, 90):
            d.rounded_rectangle([20, y, 370, y + 70], 12, fill=tuple(int(v) for v in rng.integers(150, 255, 3)))
            d.text((40, y + 25), f"Item {y}", fill=(0, 0, 0))
        out.append(im)
    for i in range(6):  # stripes and checks at random angles and colours
        a = rng.integers(0, 255, (2, 3))
        f = int(rng.integers(8, 40))
        yy, xx = np.indices((400, 400))
        mask = ((xx + (i % 3) * yy) // f) % 2 == 0
        arr = np.where(mask[..., None], a[0], a[1]).astype(np.uint8)
        out.append(Image.fromarray(arr))
    for _i in range(3):  # plain colours
        out.append(Image.new("RGB", (400, 400), tuple(int(v) for v in rng.integers(0, 255, 3))))
    for _i in range(3):  # noise
        out.append(Image.fromarray(rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)))
    return out


def o1_photo_ood(cfg) -> dict:
    from glimpse.data.photos import PhotoError, open_photo
    from glimpse.demo_cases import synthetic_images
    from glimpse.serve import Pipeline
    from glimpse.serve.rules import leave_one_out_distances, photo_problem

    app = Pipeline(cfg, sensor_first=False)
    rng = np.random.default_rng(cfg.seed)
    loo = leave_one_out_distances(app.ref, app.k).numpy()
    held = rng.choice(loo, 60, replace=False)

    def score(img) -> float:
        if photo_problem(img, cfg.prototype.min_photo_contrast) is not None:
            return float("inf")  # refused by the contrast rule before the model: counted as detected
        return app.unfamiliar_score(app.embed_image(img))

    real = []
    for p in sorted(Path("samples/local_photos").glob("*")):
        try:
            real.append(score(open_photo(p)))
        except PhotoError:
            continue
    ood = [score(im) for im in synthetic_images().values()] + [score(im) for im in _generated_non_meals(cfg.seed)]
    shot = Path("samples/instructor/09_screenshot")
    ood += [score(open_photo(next(shot.glob("photo.*"))))]
    s_in, s_out = np.r_[held, real], np.array(ood)
    y = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]
    s = np.nan_to_num(np.r_[s_in, s_out], posinf=10.0)
    cut = app.ood_cut
    return {"in_distribution": int(len(s_in)), "out_of_distribution": int(len(s_out)),
            "auroc": float(roc_auc_score(y, s)), "cut": cut,
            "flagged_in": float(np.mean(s_in > cut)), "flagged_out": float(np.mean(s_out > cut)),
            "caught_by_contrast_rule": int(np.isinf(s_out).sum()), "real_photos_max": float(max(real))}  # fmt: skip


def o2_out_of_scope(cfg, ds, sp) -> dict:
    from glimpse.data.splits import meal_rows
    from glimpse.training.baselines import trend_features

    dev_people = sorted({p for f in sp.dev_folds for p in f})
    rows_dev = np.array(sorted(meal_rows(ds, set(dev_people), scored_only=True)))
    rows_t2d = np.array(sorted(meal_rows(ds, set(sp.t2d), scored_only=True)))
    xd, xt = trend_features(ds, rows_dev), trend_features(ds, rows_t2d)
    pid = np.array([ds.samples[r].person_id for r in rows_dev])
    med = np.nanmedian(xd, axis=0)
    xd, xt = np.where(np.isnan(xd), med, xd), np.where(np.isnan(xt), med, xt)

    def maha(ref: np.ndarray, q: np.ndarray) -> np.ndarray:
        mu, cov = ref.mean(0), np.cov(ref, rowvar=False) + 1e-3 * np.eye(ref.shape[1])
        inv = np.linalg.inv(cov)
        d = q - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))

    s_dev = np.empty(len(xd))
    for p in dev_people:  # leave one person out: a new in-scope person is scored against the others
        s_dev[pid == p] = maha(xd[pid != p], xd[pid == p])
    s_t2d = maha(xd, xt)
    y = np.r_[np.zeros(len(s_dev)), np.ones(len(s_t2d))]
    names = ["HbA1c", "fasting glucose"]
    only = {n: float(roc_auc_score(y, np.r_[xd[:, i], xt[:, i]])) for n, i in zip(names, (9, 10), strict=True)}
    return {"dev_meals": int(len(s_dev)), "t2d_meals": int(len(s_t2d)), "t2d_people": len(sp.t2d),
            "auroc": float(roc_auc_score(y, np.r_[s_dev, s_t2d])), "auroc_single_input": only,
            "flag_at_dev_q95": float(np.mean(s_t2d > np.quantile(s_dev, 0.95)))}  # fmt: skip


def m1_power(cfg) -> dict:
    from glimpse.studies import monitoring_power as mp

    out = {}
    group = pd.read_csv("outputs/cv/oof_predictions.csv").groupby("person_id").group.first()
    for name, path in (("v1", V1), ("v2", V2)):
        df = pd.read_csv(path).dropna(subset=["risk"])
        df["group"] = df.person_id.map(group)
        people, pre_d = df.person_id.unique(), df[df.group == "prediabetes"]
        rng = np.random.default_rng(cfg.seed)
        res = {}
        for answers in (120, 240, 480, 960):
            within, hits, false = answers // mp.BLOCK, [], []
            for _ in range(300):
                before = mp._blocks(mp._stream(df, people, rng), mp.PRE)
                after = []
                while len(after) < within:
                    after += mp._blocks(mp._stream(pre_d, pre_d.person_id.unique(), rng), len(pre_d) // mp.BLOCK)
                blocks = before + after[:within]
                x = np.array([b.spike.mean() - b.risk.mean() for b in blocks], dtype=float)
                alarms = mp._cusum(x)
                false.append(any(a < mp.PRE for a in alarms))
                hits.append(any(a >= mp.PRE for a in alarms))
            res[str(answers)] = {"detected": float(np.mean(hits)), "false_alarm_before": float(np.mean(false))}
        out[name] = res
    return out


def c1_calibration_null(cfg) -> dict:
    from glimpse.evaluation.metrics import reliability
    from glimpse.evaluation.report_metrics import _logit, with_intercept

    out = {}
    for name, path, summ in (("v1", V1, "outputs/evaluation_v2/dev_summary.json"),
                             ("v2", V2, "outputs/evaluation_sensor_first/dev_summary.json")):  # fmt: skip
        dev = pd.read_csv(path)
        a = dev[dev.risk.notna()]
        obs = float(reliability(a.risk.to_numpy(float), a.spike.to_numpy(float), bins=15)[1])
        chk = with_intercept(dev, pd.Series(_logit(dev.risk_raw.to_numpy(float)), index=dev.row), True, summ)
        assert float((chk.loc[a.row].to_numpy() - a.risk.to_numpy(float)).__abs__().max()) < 1e-6, name
        logits = pd.Series(_logit(dev.risk_raw.to_numpy(float)), index=dev.row)
        rng = np.random.default_rng(cfg.seed)
        sims = []
        for _ in range(200):
            sim = dev.copy()
            p = sim.risk.fillna(0.5).to_numpy(float)
            sim["spike"] = (rng.random(len(sim)) < p).astype(int)
            r = with_intercept(sim, logits, platt=True, summary=summ)
            s = sim.set_index("row").loc[r.index]
            sims.append(float(reliability(r.to_numpy(float), s.spike.to_numpy(float), bins=15)[1]))
        out[name] = {"observed": obs, "null_median": float(np.median(sims)),
                     "share_null_at_or_below_observed": float(np.mean(np.array(sims) <= obs))}  # fmt: skip
    return out


def _v1_inputs(ds) -> tuple[pd.DataFrame, np.ndarray]:
    from glimpse.training.baselines import trend_features

    dev = pd.read_csv(V1).dropna(subset=["risk"])
    oof = pd.read_csv("outputs/photo_cv/photo_oof.csv").set_index("row")
    dev = dev[dev.row.isin(oof.index)].reset_index(drop=True)
    o = oof.loc[dev.row]
    x = np.hstack([trend_features(ds, dev.row.to_numpy()),
                   np.c_[np.log1p(o.photo_g), o.p_bin1, o.p_bin2, o.p_bin3]])  # fmt: skip
    return dev, x


def _lr(seed: int):
    from sklearn.impute import SimpleImputer

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.03, max_iter=2000, random_state=seed))  # fmt: skip


def s1_s2(cfg, ds) -> dict:
    dev, x = _v1_inputs(ds)
    y, fold = dev.spike.to_numpy().astype(int), dev.fold.to_numpy()
    rng = np.random.default_rng(cfg.seed)

    def oof(train_people_by_fold, shuffle: bool) -> float:
        p = np.full(len(dev), np.nan)
        for k in np.unique(fold):
            tr = np.isin(dev.person_id, train_people_by_fold[k]) & (fold != k)
            yt = rng.permutation(y[tr]) if shuffle else y[tr]
            p[fold == k] = _lr(cfg.seed).fit(x[tr], yt).predict_proba(x[fold == k])[:, 1]
        return float(roc_auc_score(y, p))

    everyone = {k: dev.person_id[fold != k].unique() for k in np.unique(fold)}
    full = oof(everyone, False)
    shuffled = [oof(everyone, True) for _ in range(20)]
    curve = {}
    for n in (4, 8, 12):
        vals = []
        for _ in range(10):
            sub = {k: rng.choice(v, min(n, len(v)), replace=False) for k, v in everyone.items()}
            vals.append(oof(sub, False))
        curve[str(n)] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    curve["all (about 18)"] = {"mean": full, "min": full, "max": full}
    return {"meals": int(len(dev)), "note": "development people only, no intercept, C fixed at 0.03",
            "s1_shuffled": {"mean": float(np.mean(shuffled)), "min": float(np.min(shuffled)),
                            "max": float(np.max(shuffled)), "real": full},
            "s2_learning_curve": curve}  # fmt: skip


def s3_slices(cfg) -> dict:
    bio = pd.read_csv("data/cgmacros/CGMacros/bio.csv")
    bio = bio.set_index("subject")
    dev = pd.read_csv(V1).dropna(subset=["risk"])
    dev["sex"] = dev.person_id.map(bio["Gender"])
    dev["age"] = pd.cut(dev.person_id.map(bio["Age"]), [0, 39, 55, 200], labels=["under 40", "40 to 55", "over 55"])
    out = {}
    for col in ("sex", "age"):
        for g, q in dev.groupby(col, observed=True):
            cl = q.decision == "low risk"
            out[f"{col} {g}"] = {"people": int(q.person_id.nunique()), "meals": int(len(q)),
                                 "auroc": float(roc_auc_score(q.spike, q.risk)) if q.spike.nunique() > 1 else None,
                                 "cleared": int(cl.sum()), "missed": int(q.spike[cl].sum())}  # fmt: skip
    per = []
    for p, q in dev.groupby("person_id"):
        if q.spike.nunique() > 1:
            per.append({"person": int(p), "meals": int(len(q)), "auroc": float(roc_auc_score(q.spike, q.risk))})
    per.sort(key=lambda r: r["auroc"])
    return {"slices": out, "worst_three_people": per[:3], "people_with_both_outcomes": len(per)}


def s4_identity(cfg, ds) -> dict:
    dev, x = _v1_inputs(ds)
    xp, pid = x[:, 15:], dev.person_id.to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.seed)
    pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), xp, pid, cv=cv)
    majority = float(pd.Series(pid).value_counts(normalize=True).iloc[0])
    xs = x[:, :15]
    xs = np.where(np.isnan(xs), np.nanmedian(xs, 0), xs)
    pred_s = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), xs, pid, cv=cv)
    return {"meals": int(len(pid)), "people": int(len(np.unique(pid))), "majority_rate": majority,
            "photo_inputs_accuracy": float(np.mean(pred == pid)),
            "sensor_profile_inputs_accuracy": float(np.mean(pred_s == pid))}  # fmt: skip


def s5_curves() -> dict:
    out = {}
    for name, f, metric in (("photo model", "outputs/training/photo_convnext_history.csv", "val_metric"),
                            ("SensorCNN", "outputs/training/sensor_cnn_history.csv", "val_auroc")):  # fmt: skip
        h = pd.read_csv(f)
        last = h.sort_values("epoch").groupby("run").tail(1)
        best = h.loc[h.groupby("run").val_loss.idxmin()]
        out[name] = {"runs": int(h.run.nunique()),
                     "final_train_loss_mean": float(last.train_loss.mean()),
                     "final_val_loss_mean": float(last.val_loss.mean()),
                     "gap_at_best_val_epoch_mean": float((best.val_loss - best.train_loss).mean()),
                     "best_epoch_median": float(best.epoch.median()),
                     metric + "_at_best_mean": float(best[metric].mean())}  # fmt: skip
    return out


def run_final_checks(cfg, ds, sp) -> dict:
    log = logger.bind(stage="final-checks")
    res: dict = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["what"] = __doc__.splitlines()[0]
    only = set(filter(None, __import__("os").environ.get("FINAL_CHECKS_ONLY", "").split(",")))
    for key, fn in (("s5", s5_curves), ("s3", lambda: s3_slices(cfg)), ("s4", lambda: s4_identity(cfg, ds)),
                    ("s1_s2", lambda: s1_s2(cfg, ds)), ("o2", lambda: o2_out_of_scope(cfg, ds, sp)),
                    ("c1", lambda: c1_calibration_null(cfg)), ("m1", lambda: m1_power(cfg)),
                    ("o1", lambda: o1_photo_ood(cfg))):  # fmt: skip
        if only and key not in only:
            continue
        res[key] = fn()
        log.info("{}: {}", key, json.dumps(res[key])[:400])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=2, default=float))
    torch.set_num_threads(2)
    return res
