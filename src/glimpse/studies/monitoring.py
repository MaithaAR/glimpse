"""S11 monitoring stub: the drift-baseline panel a deployed monitor would redraw weekly, drawn once
from committed artifacts (docs/monitoring.md is the design). No new modelling: training leave-one-out
kNN distances vs the deployed cut vs the real phone photos from S10."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from glimpse.config import GlimpseConfig
from glimpse.serve.rules import leave_one_out_distances

OUT = Path("outputs/monitoring")


def run_monitoring(cfg: GlimpseConfig) -> dict:
    """Training LOO distance distribution + deployed cut + real-photo distances -> one figure + json."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = logger.bind(stage="monitoring")
    ref = torch.load(cfg.prototype.ood_reference_file, weights_only=True)
    loo = leave_one_out_distances(ref["emb"].float(), int(ref["k"])).numpy()
    cut = float(ref["cut"])
    real = pd.read_csv("outputs/ood/real_photo_scores.csv")
    real = real[real.set == "local_photos"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(loo, bins=40, color="#9ecae1", edgecolor="white", label="training photos (leave-one-out)")
    ax.axvline(cut, color="#d62728", lw=1.5, label=f"deployed cut {cut:.3f}")
    for i, (_, r) in enumerate(real.iterrows()):
        ax.axvline(r.score, color="#555555", lw=0.8, alpha=0.6,
                   label="real phone photos (n=14)" if i == 0 else None)  # fmt: skip
    ax.set_xlabel("kNN distance to training reference (drift statistic)")
    ax.set_ylabel("photos")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "drift_baseline.png", dpi=150)
    plt.close(fig)
    res = {"what": "drift baseline for the deployed monitor (design: docs/monitoring.md)",
           "deployed_cut": cut, "training_loo_median": float(pd.Series(loo).median()),
           "training_loo_q90": float(pd.Series(loo).quantile(0.9)),
           "real_photos_median": float(real.score.median()),
           "real_photos_above_cut": int((real.score > cut).sum()), "real_photos_n": int(len(real))}  # fmt: skip
    # Rubric 4e (pre-registered 22 Nov): a simulated four-week stream; weeks 1-3 are seeded draws from
    # the training LOO distances, week 4 is the six synthetic non-meal pictures (camera fault / wrong
    # app); the committed alert rule (weekly median above the training q90) should fire on week 4 only.
    from glimpse.data.photos import eval_transform
    from glimpse.demo_cases import synthetic_images
    from glimpse.models.weights import weights_path
    from glimpse.serve import knn_distance
    from glimpse.training.photo import load_photo_bundle
    from glimpse.training.final import _photo_checkpoint

    model, _ = load_photo_bundle(cfg, _photo_checkpoint())
    model.eval()
    tf, norm = eval_transform(cfg.photo.arch, cfg.photo.crop_pct), torch.nn.functional.normalize
    with torch.no_grad():
        syn = [float(knn_distance(ref["emb"].float(), norm(model.embed(tf(im)[None]), dim=1),
                                  int(ref["k"]))[0]) for im in synthetic_images().values()]  # fmt: skip
    rng = np.random.default_rng(cfg.seed)
    q90 = float(pd.Series(loo).quantile(0.9))
    weeks = [sorted(rng.choice(loo, 20, replace=False).tolist()) for _ in range(3)] + [sorted(syn)]
    stream = [{"week": i + 1, "n": len(w), "median": float(np.median(w)),
               "alert": bool(np.median(w) > q90)} for i, w in enumerate(weeks)]  # fmt: skip
    res["simulated_stream"] = {"rule": "alert when the weekly median distance exceeds the training "
                                       "q90 (%.3f)" % q90, "weeks": stream,
                               "fires_on_week_4_only": [w["alert"] for w in stream] == [False, False, False, True]}
    for w in stream:
        log.info("simulated week {}: median {:.3f} -> {}", w["week"], w["median"],
                 "ALERT" if w["alert"] else "ok")  # fmt: skip
    (OUT / "monitoring.json").write_text(json.dumps(res, indent=2))
    log.info("drift baseline: training LOO median {:.3f} (q90 {:.3f}), cut {:.3f}; real photos median "
             "{:.3f}, {} of {} above the cut", res["training_loo_median"], res["training_loo_q90"], cut,
             res["real_photos_median"], res["real_photos_above_cut"], res["real_photos_n"])  # fmt: skip
    return res
