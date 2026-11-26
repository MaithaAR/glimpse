# ruff: noqa: E501, E741
"""Post-freeze exploratory checks, 27 Nov (docs/post_freeze_checks.md): lecture tests the submission had skipped
and an analysis of the decision layer. The served model, its cut-offs and every locked number are unchanged;
nothing here is pre-registered beyond the plan file written the same day (`glimpse post-freeze`)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import spearmanr

OUT = Path("outputs/post_freeze")
JPEG_Q = (90, 50, 20, 10)
N_JPEG = 200


def _bin(g: np.ndarray) -> np.ndarray:
    from glimpse.training.photo import carb_bin

    return carb_bin(torch.from_numpy(np.asarray(g, np.float32))).numpy()


def jpeg_stress(cfg, ds, sp) -> dict:
    """The served photo model on 200 development photos, re-encoded as JPEG at four qualities."""
    from PIL import Image

    from glimpse.data.photos import eval_transform, open_photo
    from glimpse.training.final import _photo_checkpoint
    from glimpse.training.photo import load_photo_bundle, photo_rows

    dev = {p for f in sp.dev_folds for p in f}
    rows = photo_rows(ds, dev, Path(cfg.data.cgmacros_root))
    rows = np.sort(np.random.default_rng(cfg.seed).choice(rows, N_JPEG, replace=False))
    model, _ = load_photo_bundle(cfg, _photo_checkpoint())
    model.eval()
    tf, root = eval_transform(cfg.photo.arch, cfg.photo.crop_pct), Path(cfg.data.cgmacros_root)
    true = np.array([ds.samples[i].carbs_served_g for i in rows], float)

    def estimate(quality: int | None) -> tuple[np.ndarray, np.ndarray]:
        g, b = [], []
        for s in range(0, len(rows), 25):
            ims = []
            for i in rows[s : s + 25]:
                im = open_photo(root / ds.samples[i].photo_relpath, cfg.photo.min_side_px).convert("RGB")
                if quality is not None:
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=quality)
                    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
                ims.append(tf(im))
            with torch.no_grad():
                reg, logits = model(torch.stack(ims))
            g.append(torch.expm1(reg).clamp(min=0).numpy())
            b.append(logits.argmax(1).numpy())
        return np.concatenate(g), np.concatenate(b)

    g0, b0 = estimate(None)
    res = {"photos": len(rows), "note": "served checkpoint on development photos it was trained on (in-sample)",
           "clean": {"mae_g": float(np.mean(np.abs(g0 - true)))}}  # fmt: skip
    for q in JPEG_Q:
        g, b = estimate(q)
        res[f"q{q}"] = {"mae_g": float(np.mean(np.abs(g - true))), "median_abs_shift_g": float(np.median(np.abs(g - g0))),
                        "p95_abs_shift_g": float(np.percentile(np.abs(g - g0), 95)),
                        "same_range_as_clean": float(np.mean(b == b0))}  # fmt: skip
    return res


def crop_ablation(cfg, ds, sp) -> dict:
    """Heads on frozen features, the v3 recipe as chosen in R5 (weight decay 0.1, dropout 0.5, 2 views, epochs per
    fold), with three kinds of training view: crops keeping 80% (as shipped), no crop, and no augmentation."""
    from timm.data import create_transform, resolve_model_data_config

    from glimpse.data.photos import open_photo
    from glimpse.models.photo_model import PhotoCarbModel
    from glimpse.models.weights import weights_path
    from glimpse.studies.robustness_v3 import _setup, v3_oof
    from glimpse.training.photo import embed_maps

    S = _setup(cfg, ds, sp)
    fixed = json.loads(Path("outputs/robustness_v3/robustness_v3.json").read_text())["r5_v3"]["chosen_per_fold"]
    cache = Path("data/cache/post_freeze_emb_nocrop.pt")
    if cache.is_file():
        emb_nc = torch.load(cache, weights_only=True)
    else:
        import timm

        pc, root = cfg.photo, Path(cfg.data.cgmacros_root)
        model = PhotoCarbModel(pc.arch, weights=weights_path(pc), dropout=pc.dropout).eval()
        dc = resolve_model_data_config(timm.create_model(pc.arch, pretrained=False, num_classes=0))
        tf = create_transform(**dc, is_training=True, scale=(1.0, 1.0), ratio=(1.0, 1.0), hflip=0.5,
                              color_jitter=(0.1, 0.1, 0.0), auto_augment=None, re_prob=0.0)  # fmt: skip
        torch.manual_seed(cfg.seed)
        out, todo = [], np.repeat(S["rows"], 2)
        for s in range(0, len(todo), 32):
            x = torch.stack(
                [tf(open_photo(root / ds.samples[i].photo_relpath, pc.min_side_px)) for i in todo[s : s + 32]]
            )
            with torch.no_grad():
                out.append(embed_maps(model, model.lower(x)))
        emb_nc = torch.cat(out).view(len(S["rows"]), 2, -1)
        torch.save(emb_nc, cache)
    true, scored = S["g"].numpy(), None
    oof = pd.read_csv("outputs/photo_cv/photo_oof.csv")
    scored = set(oof[oof.scored].row)

    def run(name: str, emb_aug, views: int) -> dict:
        S2 = dict(S, emb_aug=emb_aug)
        maes = []
        for seed in (0, 1, 2):
            df, _ = v3_oof(cfg, S2, seed, fixed=[dict(f, views=views) for f in fixed])
            df = df[df.row.isin(scored)]
            pos = {int(r): i for i, r in enumerate(S["rows"])}
            t = np.array([true[pos[int(r)]] for r in df.row])
            maes.append({"mae_g": float(np.mean(np.abs(df.photo_g.to_numpy() - t))),
                         "spearman": float(spearmanr(df.photo_g, t).statistic)})  # fmt: skip
        return {"views": views, "seeds": maes, "mae_g_mean": float(np.mean([m["mae_g"] for m in maes])),
                "mae_g_range": [min(m["mae_g"] for m in maes), max(m["mae_g"] for m in maes)]}  # fmt: skip

    return {"recipe": "v3 heads on frozen features, configuration fixed from R5", "scored_photos": len(scored),
            "crop_80pct_flip_jitter": run("crop", S["emb_aug"], 2), "no_crop_flip_jitter": run("nocrop", emb_nc, 2),
            "no_augmentation": run("none", S["emb_aug"], 0)}  # fmt: skip


def embedding_map(cfg, ds, sp) -> dict:
    """t-SNE of the frozen photo summaries, and what the 10 nearest neighbours share: the carb range or the person."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors

    from glimpse.studies.robustness_v3 import _setup

    S = _setup(cfg, ds, sp)
    emb, g, person = S["emb"].detach().numpy(), S["g"].numpy(), S["person"]
    b = _bin(g)
    nn_ = NearestNeighbors(n_neighbors=11).fit(emb)
    idx = nn_.kneighbors(emb, return_distance=False)[:, 1:]
    same_bin, same_person = float(np.mean(b[idx] == b[:, None])), float(np.mean(person[idx] == person[:, None]))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(b))
    chance_bin, chance_person = float(np.mean(b[perm] == b)), float(np.mean(person[perm] == person))
    other = person[idx] != person[:, None]
    same_bin_other_people = float(np.mean((b[idx] == b[:, None])[other]))
    xy = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(emb)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    sc = ax[0].scatter(xy[:, 0], xy[:, 1], c=b, cmap="viridis", s=6)
    ax[0].set_title("coloured by carb range (4 ranges)")
    fig.colorbar(sc, ax=ax[0], ticks=[0, 1, 2, 3])
    ax[1].scatter(xy[:, 0], xy[:, 1], c=pd.factorize(person)[0], cmap="tab20", s=6)
    ax[1].set_title("coloured by person (37)")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("t-SNE of the frozen photo summaries, 1,221 photos")
    fig.tight_layout()
    Path("outputs/figures").mkdir(exist_ok=True)
    fig.savefig("outputs/figures/figA3_embedding.png", dpi=160)
    plt.close(fig)
    return {"photos": len(b), "neighbours": 10,
            "share_same_carb_range": same_bin, "chance_same_carb_range": chance_bin,
            "share_same_person": same_person, "chance_same_person": chance_person,
            "share_same_carb_range_among_other_people": same_bin_other_people}  # fmt: skip


def photo_average() -> dict:
    """The fine-tuned estimate, the ridge estimate, and their average on the log scale (committed out-of-fold files)."""
    a = pd.read_csv("outputs/photo_cv/photo_oof.csv")
    a = a[a.scored]
    t, ft, rd = a.carbs_g.to_numpy(), a.photo_g.to_numpy(), a.ridge_g.to_numpy()
    avg = np.expm1((np.log1p(ft) + np.log1p(rd)) / 2)
    tb = _bin(t)

    def m(p):
        from sklearn.metrics import f1_score

        return {"mae_g": float(np.mean(np.abs(p - t))), "spearman": float(spearmanr(p, t).statistic),
                "range_f1": float(f1_score(tb, _bin(p), average="macro"))}  # fmt: skip

    return {
        "photos": len(a),
        "fine_tuned": m(ft),
        "ridge_frozen": m(rd),
        "average_log_scale": m(avg),
        "note": "range F1 here bins each grams estimate; the report's 0.61 for the fine-tune uses its range head",
    }


def decision_layer(cfg) -> dict:
    """v1's development answers at the evaluated (inner-fold) cut-offs: what a wider refer band and selective
    answering do, with person-bootstrap 95% intervals. Analysis only; the served cut-offs are unchanged."""
    from glimpse.evaluation.metrics import person_bootstrap_ci
    from glimpse.training.final import cost_dict

    d = pd.read_csv("outputs/evaluation_v2/dev_answers.csv")
    d = d[d.risk.notna()].reset_index(drop=True)
    ev = json.loads(Path("outputs/evaluation_v2/dev_summary.json").read_text())
    cut = d.fold.map({f["fold"]: f["low_risk_below"] for f in ev["folds"]}).to_numpy()
    y, r, sid, costs = d.spike.to_numpy().astype(int), d.risk.to_numpy(float), d.person_id.to_numpy(), cost_dict(cfg)

    def cost(act: np.ndarray, yy: np.ndarray) -> float:  # 0 clear, 1 caution, 2 refer
        tab = np.array([costs["low_risk"], costs["caution"], costs["abstain"]], float)
        return float(tab[act, 1 - yy].sum())

    def stats(act: np.ndarray):
        def fn(i):
            cl = act[i] == 0
            return float(y[i][cl].mean()) if cl.any() else float("nan")

        def red(i):
            return 1 - cost(act[i], y[i]) / cost(np.ones(len(i), int), y[i])

        miss, cr = person_bootstrap_ci(sid, fn), person_bootstrap_ci(sid, red)
        ans = act < 2
        return {"cleared_share": float((act == 0).mean()), "referred_share": float((act == 2).mean()),
                "missed_among_cleared": miss[0], "missed_ci95": [miss[1], miss[2]],
                "cost_reduction": cr[0], "cost_reduction_ci95": [cr[1], cr[2]],
                "error_on_answered": float(np.mean((act[ans] == 0) == (y[ans] == 1)))}  # fmt: skip

    base = np.where(r < cut, 0, 1)
    res = {"meals": len(d), "no_band": stats(base), "wider_band": {}, "selective": {}}
    for w in (0.02, 0.05, 0.08):
        act = base.copy()
        act[np.abs(r - cut) < w] = 2
        res["wider_band"][f"refer_within_{w:.2f}_of_cut"] = stats(act)
    order = np.argsort(np.abs(r - cut))
    for share in (0.1, 0.2, 0.3):
        act = base.copy()
        act[order[: int(round(share * len(d)))]] = 2
        res["selective"][f"refer_closest_{int(share * 100)}pct"] = stats(act)
    return res


def probe_b_premise(ds) -> dict:
    """Probe B expected a rising pre-meal trace to raise risk. Is the premise in the data? Spearman correlation of
    the 30-minute pre-meal slope with the measured rise, the peak and the label, 481 development meals,
    person-bootstrap intervals."""
    from glimpse.evaluation.metrics import person_bootstrap_ci
    from glimpse.training.baselines import trend_features

    d = pd.read_csv("outputs/evaluation_v2/dev_answers.csv")
    rows = d.row.to_numpy()
    slope = trend_features(ds, rows)[:, 1]
    rise = np.array([ds.samples[int(r)].rise_mgdl for r in rows])
    peak = rise + np.array([ds.samples[int(r)].baseline_mgdl for r in rows])
    sid, ok, out = d.person_id.to_numpy(), np.isfinite(slope), {"meals": int(np.isfinite(slope).sum())}
    for name, v in (("rise", rise), ("peak", peak), ("spike", d.spike.to_numpy().astype(float))):
        pt, lo, hi = person_bootstrap_ci(sid[ok], lambda i, v=v: float(spearmanr(slope[ok][i], v[ok][i]).statistic))
        out[f"spearman_slope30_vs_{name}"] = {"point": pt, "ci95": [lo, hi]}
    return out


def run_post_freeze(cfg, ds, sp) -> dict:
    log = logger.bind(stage="post-freeze")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    res = {"what": __doc__.splitlines()[0], "plan": "docs/post_freeze_checks.md"}
    for name, fn in (("photo_average", photo_average), ("probe_b_premise", lambda: probe_b_premise(ds)),
                     ("decision_layer", lambda: decision_layer(cfg)),
                     ("embedding_map", lambda: embedding_map(cfg, ds, sp)), ("jpeg_stress", lambda: jpeg_stress(cfg, ds, sp)),
                     ("crop_ablation", lambda: crop_ablation(cfg, ds, sp))):  # fmt: skip
        res[name] = fn()
        log.info("{}: {}", name, json.dumps(res[name])[:600])
        (OUT / "post_freeze.json").write_text(json.dumps(res, indent=2))
    return res
