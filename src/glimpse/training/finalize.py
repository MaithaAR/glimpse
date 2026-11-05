"""Serving-artifact regeneration (the 5 Nov amendment in docs/final_design.md, pre-registered before
this ran): the final photo model, unfamiliar-photo reference, fusion, calibration and frozen cut-offs,
all from partition v2 at the all-development level. Every number here is FIXED by the amendment --
median CV epochs, the unchanged q0.99 reference rule, the committed C recipe, the pooled out-of-fold
replay. Nothing is selected in this file. S16 only scores the locked 8 against these artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.nn.functional import normalize

from glimpse.calibration import choose_m_and_cutoffs
from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.photos import eval_transform, open_photo
from glimpse.data.splits import SplitsV2, assert_not_locked, meal_rows
from glimpse.evaluation.metrics import apply_platt, fit_platt
from glimpse.models.photo_model import PhotoCarbModel
from glimpse.models.weights import weights_path
from glimpse.serve.rules import knn_distance, leave_one_out_distances
from glimpse.training.baselines import logistic
from glimpse.training.cv import _tune_sklearn
from glimpse.training.final import FINAL, THRESHOLDS, cost_dict, hash_thresholds
from glimpse.training.fusion import PHOTO_COLS, photo_table, with_table
from glimpse.training.common import EpochLogger
from glimpse.training.photo import (
    build_cache,
    embed_maps,
    grams,
    photo_rows,
    train_heads,
    train_last_stage,
)

OUT = Path("outputs/finalize")
HEADS_EPOCHS, LAST_STAGE_EPOCHS = 12, 2  # medians of the committed per-fold CV choices (amendment §1)
REAL_PHOTOS, BUDGET = Path("samples/local_photos"), 0.02  # amendment §2: the S10 criteria, unchanged
FLAGGED_CASES = ("09_screenshot", "08_dark_photo")  # instructor photos the check must still flag


def _final_photo_model(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig, elog) -> tuple:
    """Amendment §1: the CV recipe at the all-development level, fixed epochs, no selection."""
    log, pc, hp = logger.bind(stage="finalize"), cfg.photo, cfg.training.photo
    dev = {s.person_id for s in ds.samples} - set(sp.locked)
    rows = photo_rows(ds, dev, Path(cfg.data.cgmacros_root))
    assert_not_locked(np.array([ds.samples[i].person_id for i in rows]), sp.locked, "finalize photo")
    g = grams(ds, rows)
    model = PhotoCarbModel(pc.arch, weights=weights_path(pc), dropout=pc.dropout)
    cache = Path("data/cache")
    maps_eval = build_cache(model, ds, rows, cfg, "maps_eval", cache / "photo_cv_maps_eval.pt")
    maps_aug = build_cache(model, ds, rows, cfg, "maps_aug", cache / "photo_cv_maps_aug.pt")
    emb = embed_maps(model, maps_eval)
    log.info("final photo model: {} photos of {} people; heads {} epochs, last stage {} epochs",
             len(rows), len(dev), HEADS_EPOCHS, LAST_STAGE_EPOCHS)  # fmt: skip
    train_heads(model, emb, g, hp, HEADS_EPOCHS, cfg.seed, "finalize-heads", elog)
    train_last_stage(model, maps_aug, hp.augmented_views, g, hp, LAST_STAGE_EPOCHS, cfg.seed,
                     "finalize-last-stage", elog, start_epoch=HEADS_EPOCHS)  # fmt: skip
    model.eval()
    state = model.trainable_state(pc.weights_sha256)
    state["state"] = {k: v.half() for k, v in state["state"].items()}
    path = Path("checkpoints") / f"photo_convnext_final_v2_e{HEADS_EPOCHS + LAST_STAGE_EPOCHS:02d}.pt"
    torch.save(state | {"arch": pc.arch, "strategy": "frozen backbone, heads trained + last stage fine-tuned",
                        "heads_epochs": HEADS_EPOCHS, "last_stage_epochs": LAST_STAGE_EPOCHS,
                        "recipe": "amendment 5 Nov: medians of the per-fold CV choices",
                        "seed": cfg.seed}, path)  # fmt: skip
    Path("checkpoints/photo_convnext_latest.txt").write_text(path.name)
    return model, maps_eval, path


def _final_reference(model, maps_eval, cfg: GlimpseConfig) -> dict:
    """Amendment §2: reference rebuilt under the final model; the deployed q-rule unchanged; the S10
    criteria must hold or this run STOPS -- no sweep, no new rule."""
    log, pc = logger.bind(stage="finalize"), cfg.prototype
    ref = normalize(embed_maps(model, maps_eval).float(), dim=1)
    k = pc.ood_neighbours
    loo = leave_one_out_distances(ref, k)
    cut = float(torch.quantile(loo, pc.ood_quantile))
    in_dist = float((loo > cut).float().mean())
    tf = eval_transform(cfg.photo.arch, cfg.photo.crop_pct)

    @torch.no_grad()
    def score(p: Path) -> float:
        emb = normalize(model.embed(tf(open_photo(p, cfg.photo.min_side_px))[None]), dim=1)
        return float(knn_distance(ref, emb, k)[0])

    real = {p.name: score(p) for p in sorted(REAL_PHOTOS.glob("*.jpg"))}
    flagged_real = sum(v > cut for v in real.values())
    cases = {c: score(next(Path(f"samples/instructor/{c}").glob("photo.*"))) for c in FLAGGED_CASES}
    checks = {"in_dist_flagged": in_dist, "in_dist_budget_ok": in_dist <= BUDGET,
              "real_photos_flagged": int(flagged_real), "real_photos_ok": flagged_real == 0,
              "instructor_flagged_ok": all(v > cut for v in cases.values()),
              "instructor_scores": cases}  # fmt: skip
    log.info("reference: {} photos, cut {:.3f} (q{:.2f}); in-dist flagged {:.1%}; real photos flagged "
             "{}/{}; screenshot/blank flagged: {}", len(ref), cut, pc.ood_quantile, in_dist,
             flagged_real, len(real), checks["instructor_flagged_ok"])  # fmt: skip
    if not (checks["in_dist_budget_ok"] and checks["real_photos_ok"] and checks["instructor_flagged_ok"]):
        raise RuntimeError(f"the pre-registered reference criteria do not hold: {checks}; "
                           "stopping (no sweep) -- record and decide in the open")  # fmt: skip
    torch.save({"emb": ref.half(), "cut": cut, "k": k}, Path(pc.ood_reference_file))
    return checks | {"cut": cut, "real_photo_scores": real}


def finalize(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig) -> dict:
    """The whole amendment, in order; writes outputs/finalize/finalize.json and the frozen artifacts."""
    log = logger.bind(stage="finalize")
    OUT.mkdir(parents=True, exist_ok=True)
    elog = EpochLogger(OUT / "finalize_history.csv")
    model, maps_eval, ckpt = _final_photo_model(ds, sp, cfg, elog)
    elog.close()
    ref_report = _final_reference(model, maps_eval, cfg)

    # amendment §3: fusion -- C by the committed recipe with the five dev folds as holds
    y = ds.labels()
    oof, nested = pd.read_csv("outputs/photo_cv/photo_oof.csv"), pd.read_csv("outputs/photo_cv/photo_nested.csv")
    pt = photo_table(nested[nested.fold == 1]) | photo_table(oof)  # T2D rows: fold 1's nested (recorded)
    fx = with_table(pt, len(PHOTO_COLS))
    non_locked = sorted({s.person_id for s in ds.samples} - set(sp.locked))
    c, _ = _tune_sklearn(lambda v: logistic(v, cfg.seed), cfg.baselines.logreg_c_grid, fx, ds,
                         non_locked, [list(f) for f in sp.dev_folds], y)  # fmt: skip
    tr = meal_rows(ds, set(non_locked), scored_only=False)
    lr = logistic(c, cfg.seed).fit(fx(ds, tr), y[tr])
    log.info("final fusion: C={} on {} meals of {} people", c, len(tr), len(non_locked))

    # amendment §4: calibration from the pooled out-of-fold replay of the five-fold evaluation
    da = pd.read_csv("outputs/evaluation_v2/dev_answers.csv").dropna(subset=["risk_raw"])
    p_oof, y_oof = da.risk_raw.to_numpy(), da.spike.to_numpy()
    person = da.person_id.to_numpy()
    times = np.array([ds.samples[int(r)].meal_time for r in da.row], dtype="datetime64[s]")
    a, b = fit_platt(p_oof, y_oof)
    if a <= 0:
        log.warning("pooled Platt slope {:.3f} <= 0; identity used instead", a)
        a, b = 1.0, 0.0
    sel = choose_m_and_cutoffs(apply_platt(p_oof, a, b), y_oof, person, times, cost_dict(cfg))
    sel.pop("inner_risks", None)
    log.info("frozen from the all-dev replay: Platt ({:.2f}, {:+.2f}), m={}, cut-offs ({:.2f}, {:.2f})",
             a, b, sel["m"], sel["low_risk_below"], sel["caution_from"])  # fmt: skip

    # amendment §5: freeze (bundle + sha-guarded thresholds + config values to match)
    frozen = {"main": {"low_risk_below": sel["low_risk_below"], "caution_from": sel["caution_from"]},
              "m_personal": sel["m"], "platt": [a, b],
              "provenance": "all-development replay; amendment of 5 Nov 2026 in docs/final_design.md"}  # fmt: skip
    frozen["sha256"] = hash_thresholds(frozen)
    THRESHOLDS.write_text(json.dumps(frozen, indent=2))
    joblib.dump({"models": {"photo": lr}, "calibration": (a, b), "m_personal": sel["m"], "C": c,
                 "photo_checkpoint": ckpt.name, "train_people": non_locked}, FINAL)  # fmt: skip
    _write_config_cutoffs(sel["low_risk_below"], sel["caution_from"])
    report = {"photo_checkpoint": ckpt.name, "reference": ref_report, "fusion_C": c,
              "platt": [a, b], "m": sel["m"], "low_risk_below": sel["low_risk_below"],
              "caution_from": sel["caution_from"], "replay_cost": sel["cost"],
              "cost_vs_m_curve": sel["curve"], "oof_meals": int(len(da))}  # fmt: skip
    (OUT / "finalize.json").write_text(json.dumps(report, indent=2))
    log.info("saved {}, {}, {} and config cut-offs; run selftest + instructor + stress + app next",
             ckpt.name, FINAL, THRESHOLDS)  # fmt: skip
    return report


def _write_config_cutoffs(lo: float, hi: float) -> None:
    """The config's prototype cut-offs must equal the frozen file (the Pipeline refuses a mismatch)."""
    import re

    p = Path("configs/default.yaml")
    s = p.read_text()
    s, n1 = re.subn(r"(low_risk_below:\s*)[0-9.]+", rf"\g<1>{lo}", s, count=1)
    s, n2 = re.subn(r"(caution_from:\s*)[0-9.]+", rf"\g<1>{hi}", s, count=1)
    if n1 != 1 or n2 != 1:
        raise RuntimeError("could not update the prototype cut-offs in configs/default.yaml")
    p.write_text(s)
