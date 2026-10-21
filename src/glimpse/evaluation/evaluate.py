"""`glimpse evaluate`: every reported number comes from the 5 held-out test people; every choice (baseline settings,
ridge strength, abstention band) is made without them. Results go to outputs/eval/ as CSV, JSON and PNG."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from glimpse.config import GlimpseConfig
from glimpse.data.augment import photo_transforms
from glimpse.data.dataset import MealDataset
from glimpse.data.meals import build_meals
from glimpse.data.normalise import Normaliser
from glimpse.data.splits import Splits
from glimpse.evaluation.baselines import engineered_features, fit_gbm, fit_logistic, personal_prior  # fmt: skip
from glimpse.evaluation.metrics import (
    decide,
    ece_equal_mass,
    ece_equal_width,
    expected_cost,
    person_bootstrap,
    select_band,  # fmt: skip
    usefulness_report,
)
from glimpse.models.photo_model import PhotoCarbModel, carb_bin
from glimpse.models.sensor_cnn import SensorCNN
from glimpse.schemas import DecisionEval, LatencyEval, PhotoEval, SpikeEval, TargetStatus
from glimpse.training.finetune import _load, photo_rows
from glimpse.training.scratch import _inputs, _rows, predict_proba

log = logger.bind(stage="evaluate")
CNN = "sensor CNN (scratch)"
MACRO_PROFILE = ["carbs_consumed", "protein_consumed", "fat_consumed", "fiber_consumed"]


# ---------------------------------------------------------------- loading
def load_sensor_cnn(cfg: GlimpseConfig) -> tuple[SensorCNN, Normaliser, str]:
    """The committed sensor-CNN checkpoint named in sensor_cnn_latest.txt, with its own normaliser."""
    ckdir = Path(cfg.paths.checkpoints)
    path = ckdir / (ckdir / "sensor_cnn_latest.txt").read_text().strip()
    ck = torch.load(path, weights_only=False)  # our own checkpoint (contains plain metadata, no code)
    hp = ck["hparams"]
    model = SensorCNN(n_tabular=ck["n_tabular"], channels=hp["channels"], hidden=hp["hidden"], dropout=hp["dropout"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), Normaliser.model_validate(ck["normaliser"]), ck["version"]


def load_photo_model(cfg: GlimpseConfig) -> tuple[PhotoCarbModel, dict, str]:
    """Pretrained ConvNeXt-Tiny from the local weight cache plus the fine-tuned weights (not yet applied)."""
    ckdir = Path(cfg.paths.checkpoints)
    path = ckdir / (ckdir / "photo_convnext_latest.txt").read_text().strip()
    ck = torch.load(path, weights_only=False)
    model = PhotoCarbModel(ck["arch"], pretrained=True, cache_dir=str(cfg.paths.weights_cache),
                           dropout=ck["hparams"]["dropout"])  # fmt: skip
    return model.eval(), {k: v.float() for k, v in ck["trainable_state"].items()}, ck["version"]


# ---------------------------------------------------------------- spike risk
def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC that raises ValueError for one-class samples (skipped by the bootstrap)."""
    return float(roc_auc_score(y, p))


def spike_eval(name: str, kind: str, split: str, p: np.ndarray, y: np.ndarray, sids: np.ndarray, bins: int,
               p_cnn: np.ndarray | None, seed: int) -> SpikeEval:  # fmt: skip
    """Discrimination and calibration of one model; paired person-bootstrap difference against the CNN."""
    delta = delta_ci = None
    if p_cnn is not None:
        delta = _auroc(y, p) - _auroc(y, p_cnn)
        delta_ci = person_bootstrap(lambda yy, a, b: _auroc(yy, a) - _auroc(yy, b), (y, p, p_cnn), sids, seed=seed)
    return SpikeEval(model=name, kind=kind, split=split, n_meals=len(y), n_people=len(np.unique(sids)),
                     n_spikes=int(y.sum()), auroc=_auroc(y, p),
                     auroc_ci=person_bootstrap(_auroc, (y, p), sids, seed=seed),
                     auprc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)),
                     ece_equal_width=ece_equal_width(p, y, bins)[0], ece_equal_mass=ece_equal_mass(p, y, bins),
                     delta_auroc_vs_cnn=delta, delta_ci=delta_ci)  # fmt: skip


def spike_predictions(ds: MealDataset, sp: Splits, cfg: GlimpseConfig, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Predictions of every spike model for `rows`; baselines are fitted on the CNN's own training meals."""
    tr = _rows(ds, set(sp.train), eval_only=False)
    y_tr = ds.labels()[tr]
    x_tr, x = engineered_features(ds, tr), engineered_features(ds, rows)
    target_tr = [i for i in tr if ds.samples[i].group in cfg.cohort.target_groups]
    rate = float(ds.labels()[target_tr].mean())
    model, norm, _ = load_sensor_cnn(cfg)
    return {
        "training spike rate (prevalence)": np.full(len(rows), rate),
        "personal prior (onboarding spike rate)": personal_prior(ds, rows, rate),
        "logistic regression (engineered features)": fit_logistic(x_tr, y_tr, cfg.seed).predict_proba(x)[:, 1],
        "gradient boosting (engineered features)": fit_gbm(x_tr, y_tr, cfg.seed).predict_proba(x)[:, 1],
        CNN: predict_proba(model, _inputs(ds, rows, norm)),
    }


KIND = {
    "training spike rate (prevalence)": "trivial",
    "personal prior (onboarding spike rate)": "heuristic",
    "logistic regression (engineered features)": "classical",
    "gradient boosting (engineered features)": "classical",
    CNN: "scratch",
}


def reliability_plot(p: np.ndarray, y: np.ndarray, bins: int, path: Path, title: str) -> list[dict]:
    """Equal-width reliability diagram with the meal count printed in every bin, plus the prediction histogram."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ece, table = ece_equal_width(p, y, bins)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    a.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    for t in table:
        if t["n"]:
            a.bar(t["lo"], t["observed_rate"], width=t["hi"] - t["lo"], align="edge", alpha=0.6, color="tab:blue",
                  edgecolor="white")  # fmt: skip
            a.plot(t["mean_predicted"], t["observed_rate"], "o", color="tab:red", ms=4)
            a.text((t["lo"] + t["hi"]) / 2, min(t["observed_rate"] + 0.03, 0.97), f"n={t['n']}", ha="center",
                   fontsize=7)  # fmt: skip
    a.set(xlim=(0, 1), ylim=(0, 1), xlabel="predicted spike probability", ylabel="observed spike rate",
          title=f"ECE {ece:.3f} ({bins} equal-width bins)")  # fmt: skip
    a.legend(loc="upper left", fontsize=7)
    b.hist(p[y == 0], bins=np.linspace(0, 1, bins + 1), alpha=0.6, label="no spike")
    b.hist(p[y == 1], bins=np.linspace(0, 1, bins + 1), alpha=0.6, label="spike")
    b.set(xlabel="predicted spike probability", ylabel="meals", title="prediction distribution")
    b.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("reliability diagram written to {}", path)
    return table


# ---------------------------------------------------------------- abstention band and decisions
def choose_band(cfg: GlimpseConfig, p_val: np.ndarray, y_val: np.ndarray, n_people: int) -> dict:
    """Decision band chosen on the validation people with the final model (never test): minimum expected cost at
    the configured costs, with spikes among cleared meals and abstentions capped at the usefulness targets. The
    same rule on the training people's out-of-fold predictions is reported for comparison only (fold models are
    not the final model, so their band does not transfer)."""
    u, costs = cfg.targets.usefulness, cfg.costs.model_dump()
    lo, hi = select_band(p_val, y_val, costs, u.max_abstain, u.max_missed_among_cleared)
    band = {"low_risk_below": lo, "caution_at_or_above": hi, "costs": costs,
            "source": f"validation people, final model ({len(y_val)} meals, {n_people} people)"}  # fmt: skip
    oof_path = Path(cfg.paths.outputs) / "training" / "sensor_cnn_oof.csv"
    if oof_path.is_file():
        oof = pd.read_csv(oof_path)
        oof = oof[oof.target_cohort]
        band["for_comparison_out_of_fold_band"] = select_band(oof.p_spike.to_numpy(), oof.spike.to_numpy(), costs,
                                                              u.max_abstain, u.max_missed_among_cleared)  # fmt: skip
    return band


def decision_eval(p: np.ndarray, y: np.ndarray, sids: np.ndarray, cfg: GlimpseConfig, split: str,
                  source: str) -> DecisionEval:  # fmt: skip
    """Business metrics at the configured band (cost reduction with a person-bootstrap CI); cost reduction also for
    other missed-spike costs with the band fixed."""
    lo, hi = cfg.decision.low_risk_below, cfg.decision.caution_at_or_above
    costs = cfg.costs.model_dump()
    a = decide(p, lo, hi)
    rep = usefulness_report(a, y, costs, cfg.targets.model_dump())
    sweep = {}
    for c in cfg.costs.sweep_missed_spike:
        cc = {**costs, "low_risk": (c, costs["low_risk"][1])}
        trivial = min(expected_cost(np.zeros_like(y), y, cc), expected_cost(np.ones_like(y), y, cc))
        sweep[f"{c:g}:1"] = 1 - expected_cost(a, y, cc) / trivial
    missed = rep["missed_among_cleared"]

    def reduction(yy: np.ndarray, aa: np.ndarray) -> float:
        trivial = min(expected_cost(np.zeros_like(yy), yy, costs), expected_cost(np.ones_like(yy), yy, costs))
        return 1 - expected_cost(aa, yy, costs) / trivial

    return DecisionEval(split=split, thresholds=(lo, hi), threshold_source=source,
                        share_low_risk=rep["share_low_risk"], share_caution=rep["share_caution"],
                        share_abstain=rep["share_abstain"],
                        spike_rate_among_cleared=None if np.isnan(missed) else missed, cost=rep["cost"],
                        best_trivial_cost=rep["best_trivial_cost"], cost_reduction=rep["cost_reduction"],
                        cost_reduction_ci=person_bootstrap(reduction, (y, a), sids, seed=cfg.seed),
                        cost_reduction_by_missed_spike_cost=sweep)  # fmt: skip


# ---------------------------------------------------------------- photo
@torch.no_grad()
def _photo_forward(model: PhotoCarbModel, ds: MealDataset, rows: np.ndarray, root: Path, pooled_only: bool):
    """Pooled embeddings (frozen) or (grams, bin probabilities) (fine-tuned) for the given photo rows."""
    tf = photo_transforms("convnext_tiny", train=False)
    out = []
    for s in range(0, len(rows), 32):
        x = torch.stack([tf(_load(root / ds.samples[i].photo_relpath)) for i in rows[s : s + 32]])
        emb = model.upper(model.lower(x))
        if pooled_only:
            out.append(emb)
        else:
            reg, logits = model.heads(emb)
            out.append(torch.cat([reg[:, None], logits], 1))
    z = torch.cat(out).numpy()
    return z if pooled_only else (np.expm1(z[:, 0]).clip(min=0), torch.softmax(torch.from_numpy(z[:, 1:]), 1).numpy())


def _photo_eval(name: str, pred: np.ndarray, bins_pred: np.ndarray, true: np.ndarray, sids: np.ndarray,
                excluded: np.ndarray, seed: int) -> PhotoEval:  # fmt: skip
    """Carb MAE (person-bootstrap CI), rank correlation, bin macro-F1, and MAE on profile-excluded meals."""
    mae = lambda a, b: float(np.mean(np.abs(a - b)))  # noqa: E731
    keep = ~excluded
    return PhotoEval(model=name, n_photos=len(true), n_people=len(np.unique(sids)), mae_g=mae(pred, true),
                     mae_ci=person_bootstrap(mae, (pred, true), sids, seed=seed),
                     spearman=None if np.ptp(pred) == 0 else float(spearmanr(pred, true).statistic),
                     bin_macro_f1=float(f1_score(carb_bin(torch.from_numpy(true)).numpy(), bins_pred,
                                                 average="macro", labels=[0, 1, 2, 3])),
                     mae_profile_excluded_g=mae(pred[keep], true[keep]) if keep.any() else None,
                     n_profile_excluded=int(keep.sum()))  # fmt: skip


def repeated_profile(ds: MealDataset, cfg: GlimpseConfig, rows: np.ndarray, train_people: set[int]) -> np.ndarray:
    """True where a meal's exact consumed-macro profile (carbs, protein, fat, fibre) also occurs among training
    people's meals: the photo model may have seen the same packaged meal from someone else."""
    m = build_meals(cfg)
    m["key"] = list(zip(*(m[c].round(1) for c in MACRO_PROFILE), strict=True))
    seen = set(m[m.sid.isin(train_people)].key)
    by_meal = dict(zip(zip(m.sid, m.t, strict=True), m.key, strict=True))
    return np.array([by_meal.get((ds.samples[i].person_id, pd.Timestamp(ds.samples[i].meal_time))) in seen
                     for i in rows])  # fmt: skip


def evaluate_photo(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> tuple[list[PhotoEval], dict]:
    """Constant median vs frozen ConvNeXt + ridge (no fine-tuning) vs the fine-tuned model, on test photos."""
    root = Path(cfg.data.cgmacros_root)
    rows = {k: photo_rows(ds, set(getattr(sp, k)), root) for k in ("train", "validation", "test")}
    grams = {k: np.array([ds.samples[i].carbs_served_g for i in r]) for k, r in rows.items()}
    sids = np.array([ds.samples[i].person_id for i in rows["test"]])
    excluded = repeated_profile(ds, cfg, rows["test"], set(sp.train))
    log.info("photos: train {}, validation {}, test {} ({} test meals share a training meal's macro profile)",
             *(len(r) for r in rows.values()), int(excluded.sum()))  # fmt: skip
    model, tuned, version = load_photo_model(cfg)
    cache = Path(cfg.paths.feature_cache) / "frozen_pooled.pt"
    allrows = np.concatenate(list(rows.values()))
    if cache.is_file() and np.array_equal(torch.load(cache, weights_only=True)["rows"].numpy(), allrows):
        emb = torch.load(cache, weights_only=True)["emb"].numpy()
    else:
        log.info("embedding {} photos with the frozen pretrained backbone", len(allrows))
        emb = _photo_forward(model, ds, allrows, root, pooled_only=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rows": torch.from_numpy(allrows), "emb": torch.from_numpy(emb)}, cache)
    n_tr, n_va = len(rows["train"]), len(rows["validation"])
    e_tr, e_va, e_te = emb[:n_tr], emb[n_tr : n_tr + n_va], emb[n_tr + n_va :]
    alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    val_mae = [float(np.mean(np.abs(np.expm1(Ridge(alpha=a).fit(e_tr, np.log1p(grams["train"])).predict(e_va))
                                    - grams["validation"]))) for a in alphas]  # fmt: skip
    alpha = alphas[int(np.argmin(val_mae))]
    ridge = Ridge(alpha=alpha).fit(e_tr, np.log1p(grams["train"]))
    frozen = np.expm1(ridge.predict(e_te)).clip(min=0)
    model.load_state_dict(tuned, strict=False)
    tuned_g, tuned_bins = _photo_forward(model, ds, rows["test"], root, pooled_only=False)
    median = np.full(len(sids), float(np.median(grams["train"])))
    to_bin = lambda g: carb_bin(torch.from_numpy(g)).numpy()  # noqa: E731
    true = grams["test"]
    res = [_photo_eval("constant (training median)", median, to_bin(median), true, sids, excluded, cfg.seed),
           _photo_eval(f"frozen ConvNeXt-Tiny + ridge (alpha {alpha:g}, no fine-tuning)", frozen, to_bin(frozen),
                       true, sids, excluded, cfg.seed),
           _photo_eval(f"fine-tuned ConvNeXt-Tiny ({version})", tuned_g, tuned_bins.argmax(1), true, sids,
                       excluded, cfg.seed)]  # fmt: skip
    return res, {"ridge_alpha_by_validation_mae": dict(zip(map(str, alphas), val_mae, strict=True))}


# ---------------------------------------------------------------- latency
@torch.no_grad()
def measure_latency(ds: MealDataset, rows: np.ndarray, photo: np.ndarray, cfg: GlimpseConfig) -> LatencyEval:
    """Warm single-meal CPU latency: sensor CNN (scaling + forward) and photo model (decode + transform + forward)."""
    model, norm, _ = load_sensor_cnn(cfg)
    pm, tuned, _ = load_photo_model(cfg)
    pm.load_state_dict(tuned, strict=False)
    tf = photo_transforms("convnext_tiny", train=False)
    root = Path(cfg.data.cgmacros_root)
    ts, tp = [], []
    for k, (i, j) in enumerate(zip(rows, photo, strict=False)):
        t0 = time.perf_counter()
        x = _inputs(ds, np.array([i]), norm)
        model(torch.from_numpy(x.sequence), torch.from_numpy(x.tabular))
        t1 = time.perf_counter()
        pm(tf(_load(root / ds.samples[j].photo_relpath))[None])
        t2 = time.perf_counter()
        if k >= 3:  # warm-up excluded
            ts.append((t1 - t0) * 1000)
            tp.append((t2 - t1) * 1000)
    ts, tp = np.array(ts), np.array(tp)
    q = lambda v, x: float(np.percentile(v, x))  # noqa: E731
    return LatencyEval(n_meals=len(ts), threads=torch.get_num_threads(), sensor_p50=q(ts, 50),
                       sensor_p95=q(ts, 95), photo_p50=q(tp, 50), photo_p95=q(tp, 95), total_p50=q(ts + tp, 50),
                       total_p95=q(ts + tp, 95))  # fmt: skip


# ---------------------------------------------------------------- targets
def _status(point_ok: bool, ci_ok: bool) -> str:
    """'met' only when the whole CI satisfies the target; 'on track' when only the point estimate does."""
    return "met" if ci_ok else ("on track" if point_ok else "at risk")


def targets_table(spike: list[SpikeEval], photo: list[PhotoEval], dec: DecisionEval, lat: LatencyEval,
                  ece_ci: tuple[float, float], cfg: GlimpseConfig) -> list[TargetStatus]:  # fmt: skip
    """Proposal success criteria next to the current test results."""
    t, u = cfg.targets, cfg.targets.usefulness
    cnn = next(s for s in spike if s.model == CNN)
    lr = next(s for s in spike if s.model.startswith("logistic"))
    d, (dlo, dhi) = -lr.delta_auroc_vs_cnn, (-lr.delta_ci[1], -lr.delta_ci[0])
    const, tuned = photo[0], photo[-1]
    cleared_ok = dec.share_low_risk >= u.min_cleared
    missed_ok = dec.spike_rate_among_cleared is not None and dec.spike_rate_among_cleared <= u.max_missed_among_cleared
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    cr_lo, cr_hi = dec.cost_reduction_ci
    n = f"test: {cnn.n_meals} meals, {cnn.n_people} people"
    rows = [
        TargetStatus(criterion="RQ2: sensor CNN beats engineered trends (AUROC, vs logistic regression)",
                     target="lower 95% bound of CNN - LR > 0", result=f"{d:+.3f} ({dlo:+.3f}, {dhi:+.3f})",
                     status="met" if dlo > 0 else "at risk",
                     note=f"{n}; a CI target, so the status follows the CI (a point gain alone is not a win)"),
        TargetStatus(criterion="Calibration of the sensor CNN", target=f"ECE <= {t.ece_max}",
                     result=f"{cnn.ece_equal_width:.3f} ({ece_ci[0]:.3f}, {ece_ci[1]:.3f})",
                     status=_status(cnn.ece_equal_width <= t.ece_max, ece_ci[1] <= t.ece_max),
                     note="10 equal-width bins; no recalibration yet"),
        TargetStatus(criterion="Usefulness: cost reduction vs best trivial policy",
                     target=f">= {pct(u.min_cost_reduction)}",
                     result=f"{pct(dec.cost_reduction)} ({pct(cr_lo)}, {pct(cr_hi)})",
                     status=_status(dec.cost_reduction >= u.min_cost_reduction,
                                    dec.cost_reduction_ci[0] >= u.min_cost_reduction),
                     note="costs 3:1; band chosen on validation; other usefulness rows are point estimates"),
        TargetStatus(criterion="Usefulness: meals cleared as low risk", target=f">= {pct(u.min_cleared)}",
                     result=pct(dec.share_low_risk), status="on track" if cleared_ok else "at risk"),
        TargetStatus(criterion="Usefulness: spikes among cleared meals",
                     target=f"<= {pct(u.max_missed_among_cleared)}",
                     result="n/a (none cleared)" if dec.spike_rate_among_cleared is None
                     else pct(dec.spike_rate_among_cleared), status="on track" if missed_ok else "at risk"),
        TargetStatus(criterion="Usefulness: abstention rate", target=f"<= {pct(u.max_abstain)}",
                     result=pct(dec.share_abstain),
                     status="on track" if dec.share_abstain <= u.max_abstain else "at risk"),
        TargetStatus(criterion="Latency (sensor + photo model, warm CPU)", target=f"p95 <= {t.latency_p95_ms:g} ms",
                     result=f"p95 {lat.total_p95:.0f} ms", status="met" if lat.total_p95 <= t.latency_p95_ms
                     else "at risk", note=f"{lat.threads} threads, excludes app/network"),
        TargetStatus(criterion="Photo gate: fine-tuned carb MAE", target="< 23.1 g and below the constant",
                     result=f"{tuned.mae_g:.1f} g ({tuned.mae_ci[0]:.1f}, {tuned.mae_ci[1]:.1f}) vs constant "
                     f"{const.mae_g:.1f} g", status=_status(tuned.mae_g < min(23.1, const.mae_g),
                                                           tuned.mae_ci[1] < min(23.1, const.mae_g))),
        TargetStatus(criterion="RQ1: photo-predicted vs reference macros in the risk model",
                     target="non-inferior (lower bound > -0.03)", result="not run", status="deferred",
                     note="needs the photo + sensor fusion (final stage)"),
    ]  # fmt: skip
    return rows


# ---------------------------------------------------------------- orchestration
def run_evaluation(cfg: GlimpseConfig) -> Path:
    """Run every evaluation and write outputs/eval/."""
    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))
    out = Path(cfg.paths.outputs) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    ds, sp = MealDataset.load(Path(cfg.paths.dataset)), Splits.load(Path(cfg.paths.splits))
    assert not set(sp.test) & (set(sp.train) | set(sp.validation)), "test people leak into development splits"
    bins, seed = cfg.targets.ece_bins, cfg.seed
    res: dict[str, dict] = {}
    for split in ("validation", "test"):
        rows = _rows(ds, set(getattr(sp, split)), eval_only=True)
        y, sids = ds.labels()[rows], np.array([ds.samples[i].person_id for i in rows])
        preds = spike_predictions(ds, sp, cfg, rows)
        res[split] = {"rows": rows, "y": y, "sids": sids, "preds": preds}
    te = res["test"]
    spike = [spike_eval(m, KIND[m], "test", p, te["y"], te["sids"], bins, None if m == CNN else te["preds"][CNN],
                        seed) for m, p in te["preds"].items()]  # fmt: skip
    pd.DataFrame([s.model_dump() for s in spike]).to_csv(out / "spike_test.csv", index=False)
    for s in spike:
        msg = "AUROC {:.3f} ({:.3f}-{:.3f}) AUPRC {:.3f} Brier {:.3f} ECE {:.3f}"
        log.bind(split="test", model=s.model).info(msg, s.auroc, *s.auroc_ci, s.auprc, s.brier, s.ece_equal_width)
    per_person = []
    for pid in np.unique(te["sids"]):
        m = te["sids"] == pid
        row = {"person_id": int(pid), "meals": int(m.sum()), "spikes": int(te["y"][m].sum())}
        for name, p in te["preds"].items():
            ok = 0 < te["y"][m].sum() < m.sum()
            row[name] = float(roc_auc_score(te["y"][m], p[m])) if ok else None
        per_person.append(row)
    pd.DataFrame(per_person).to_csv(out / "spike_test_per_person.csv", index=False)
    p_cnn = te["preds"][CNN]
    table = reliability_plot(p_cnn, te["y"], bins, out / "reliability_sensor_cnn_test.png",
                             f"Sensor CNN, test ({len(p_cnn)} meals, {len(np.unique(te['sids']))} people)")  # fmt: skip
    pd.DataFrame(table).to_csv(out / "calibration_bins_sensor_cnn_test.csv", index=False)
    ece_ci = person_bootstrap(lambda yy, pp: ece_equal_width(pp, yy, bins)[0], (te["y"], p_cnn), te["sids"],
                              seed=seed)  # fmt: skip

    va = res["validation"]
    band = choose_band(cfg, va["preds"][CNN], va["y"], len(np.unique(va["sids"])))
    (out / "abstention_band.json").write_text(json.dumps(band, indent=2))
    configured = (cfg.decision.low_risk_below, cfg.decision.caution_at_or_above)
    if (band["low_risk_below"], band["caution_at_or_above"]) != configured:
        log.warning("config band {} differs from the validation choice {}; update configs/default.yaml", configured,
                    (band["low_risk_below"], band["caution_at_or_above"]))  # fmt: skip
    source = band["source"]
    decisions = [
        decision_eval(res[s]["preds"][CNN], res[s]["y"], res[s]["sids"], cfg, s, source) for s in ("validation", "test")
    ]
    (out / "decisions.json").write_text(json.dumps([d.model_dump() for d in decisions], indent=2))
    for d in decisions:
        log.bind(split=d.split).info("band {}: cleared {:.1%} (spikes among cleared {}), caution {:.1%}, abstain "
                                     "{:.1%}, cost reduction {:+.1%}", d.thresholds, d.share_low_risk,
                                     d.spike_rate_among_cleared, d.share_caution, d.share_abstain,
                                     d.cost_reduction)  # fmt: skip

    photo, photo_meta = evaluate_photo(ds, sp, cfg)
    pd.DataFrame([p.model_dump() for p in photo]).to_csv(out / "photo_test.csv", index=False)
    for p in photo:
        log.bind(split="test", model=p.model).info("carb MAE {:.1f} g ({:.1f}-{:.1f}), Spearman {}, bin F1 {:.2f}",
                                                   p.mae_g, *p.mae_ci, p.spearman, p.bin_macro_f1)  # fmt: skip
    root = Path(cfg.data.cgmacros_root)
    photo_te = photo_rows(ds, set(sp.test), root)
    lat = measure_latency(ds, te["rows"][:43], photo_te[:43], cfg)
    log.info("latency p95: sensor {:.1f} ms, photo {:.1f} ms, total {:.1f} ms", lat.sensor_p95, lat.photo_p95,
             lat.total_p95)  # fmt: skip
    targets = targets_table(spike, photo, decisions[1], lat, ece_ci, cfg)
    pd.DataFrame([t.model_dump() for t in targets]).to_csv(out / "targets_status.csv", index=False)
    summary = {"test_people": sp.test, "spike": [s.model_dump() for s in spike],
               "sensor_cnn_ece_ci": ece_ci, "photo": [p.model_dump() for p in photo], **photo_meta,
               "decisions": [d.model_dump() for d in decisions], "latency_ms": lat.model_dump(),
               "targets": [t.model_dump() for t in targets]}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("evaluation written to {}", out)
    return out
