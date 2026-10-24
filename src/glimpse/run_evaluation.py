"""Step 5b: the evaluation. Two commands.

`glimpse amend-thresholds` (before the test is opened): the cut-offs chosen on the 4 validation people failed the
stability check, so the main cut-offs become the ones that follow from the costs alone (no data). The pre-registered
cut-offs and cut-offs chosen on the cross-validation WITHOUT the test people stay as secondary rows. Everything is
frozen with a hash and must be committed before the test can be scored.

`glimpse evaluate`: one results folder (outputs/evaluation/):
- the main estimate: every model on the 5-fold person cross-validation (23 people), with 95% ranges by person;
- the research questions (RQ1 photo vs logged carbs, RQ2 CNN vs hand-made trends) and the photo gain;
- calibration of the main model (reliability diagram, ECE) and its day-to-day usefulness with the main cut-offs;
- the fixed test (5 people), scored once with the frozen 28-person pipeline (a sanity check: these people are also
  inside the cross-validation);
- the targets status table and two figures."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import Splits, meal_rows
from glimpse.evaluation import (
    choose_thresholds,
    cost_only_thresholds,
    decide,
    expected_cost,
    noninferiority,
    paired_person_bootstrap,
    person_bootstrap_ci,
    reliability,
    usefulness_report,
)
from glimpse.models.photo_model import PhotoCarbModel
from glimpse.models.weights import weights_path
from glimpse.training.baselines import trend_features
from glimpse.training.final import FINAL, THRESHOLDS, checkpoint_estimates, cost_dict, hash_thresholds, load_thresholds
from glimpse.training.fusion import PHOTO_COLS, photo_table, with_table
from glimpse.training.photo import build_cache, photo_rows

AMENDED = Path("checkpoints/thresholds_amended.json")
OUT = Path("outputs/evaluation")
LADDER = [  # (label, column): the order of the results table and the AUROC figure
    ("Base rate (always the same guess)", "prevalence"),
    ("Profile only (logistic regression)", "profile_lr"),
    ("Profile + trend (logistic regression)", "no_meal"),
    ("Profile + trend (gradient boosting)", "trend_gbm"),
    ("Profile + SensorCNN (from scratch)", "sensor_cnn"),
    ("+ photo carbs (MAIN)", "photo"),
    ("+ logged carbs (reference)", "logged_carbs"),
    ("+ shuffled photo (control)", "photo_shuffled"),
    ("+ ridge photo (post-hoc)", "ridge_photo"),
    ("+ photo grams only, no bins (post-hoc)", "photo_grams_only"),
]


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _sweep_costs(cfg: GlimpseConfig) -> list[dict]:
    """The cost settings fixed before training for the sensitivity check: missed-spike cost x abstain multiplier."""
    c = cfg.costs
    return [{"missed_spike": m, "abstain_multiplier": k, "costs": {"low_risk": (m, 0.0), "caution": tuple(c.caution),
             "abstain": (c.abstain[0] * k, c.abstain[1] * k)}}
            for m in c.sweep_missed_spike for k in c.sweep_abstain_multiplier]  # fmt: skip


# ---------------------------------------------------------------- amendment (before the test is opened)
def amend_thresholds(sp: Splits, cfg: GlimpseConfig) -> dict:
    """Freeze the main (cost-only), pre-registered and CV-without-test cut-offs before any test person is scored."""
    if (OUT / "test_scored.json").is_file():
        raise RuntimeError("the test was already scored; the cut-offs can no longer be amended")
    pre = load_thresholds(THRESHOLDS)
    costs = cost_dict(cfg)
    lo, hi = cost_only_thresholds(costs)
    assert sorted(sp.test) == sorted(sp.outer_folds[0]), "the test people must be outer fold 1"
    oof = pd.read_csv(Path(cfg.paths.outputs) / "fusion" / "fusion_oof.csv")
    dev = oof[oof.fold != 1]  # cross-validated predictions WITHOUT the test people
    lo_cv, hi_cv, _ = choose_thresholds(dev.photo.to_numpy(), dev.spike.to_numpy(), costs)
    d = {
        "status": "amendment made after Step 5a and BEFORE the test was opened",
        "reason": "cut-offs chosen on the 4 validation people (spike rate 19% vs 33% in the cross-validation) failed "
                  "the pre-registered stability check (no cost saving on the pooled cross-validation)",
        "main": {"rule": "cost-only (from the pre-registered costs; uses no data)", "low_risk_below": lo,
                 "caution_from": hi},
        "secondary_preregistered": {"low_risk_below": pre["low_risk_below"], "caution_from": pre["caution_from"],
                                    "sha256": pre["sha256"]},
        "secondary_cv_without_test": {"low_risk_below": lo_cv, "caution_from": hi_cv,
                                      "chosen_on": f"{len(dev)} cross-validated meals, test people excluded"},
        "sensitivity": "cost-only cut-offs recomputed for every pre-registered cost setting (costs.sweep_*)",
        "success_targets": cfg.targets.usefulness.model_dump(),
        "test_role": "sanity check; the test people are also inside the cross-validation (main estimate)",
    }  # fmt: skip
    d["sha256"] = hash_thresholds(d)
    AMENDED.write_text(json.dumps(d, indent=2))
    logger.bind(stage="evaluate").info(
        "amended cut-offs frozen (hash {}): MAIN low risk < {:.3f}, caution >= {:.3f}; pre-registered {:.2f}/{:.2f}; "
        "CV without test {:.2f}/{:.2f}. Commit {} before running `just evaluate`.",
        d["sha256"][:12], lo, hi, pre["low_risk_below"], pre["caution_from"], lo_cv, hi_cv, AMENDED)  # fmt: skip
    return d


def _amendment_committed() -> bool:
    """The amendment must be in git, unchanged, before the test is scored (a dated, tamper-evident record)."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(AMENDED)], capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(AMENDED)]).returncode == 0
    return tracked and clean


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- the evaluation
def _usefulness_ci(p, y, sids, lo, hi, costs, targets, B, seed) -> dict:
    """Usefulness numbers at fixed cut-offs, each with a 95% range by person."""
    rep = usefulness_report(decide(p, lo, hi), y, costs, targets)

    def stat(key):
        """A function of meal indices returning one usefulness number (for the person bootstrap)."""

        def f(i):
            """The number on these meals (NaN if undefined)."""
            r = usefulness_report(decide(p[i], lo, hi), y[i], costs, targets)
            return float(r[key]) if r[key] == r[key] else float("nan")

        return f

    for key in ("cost_reduction", "share_low_risk", "missed_among_cleared", "share_abstain"):
        rep[f"{key}_ci95"] = list(person_bootstrap_ci(sids, stat(key), B, seed)[1:])
    return rep


def _status(point: float, lo: float, hi: float, ok) -> str:
    """met = the whole 95% range meets the target; on track = only the estimate does; otherwise at risk."""
    if ok(lo) and ok(hi):
        return "met"
    return "on track" if ok(point) else "at risk"


def _cv_table(cfg: GlimpseConfig) -> pd.DataFrame:
    """One row per scored meal: cross-validated predictions of every model (CNN averaged over its 3 seeds)."""
    out = Path(cfg.paths.outputs)
    need = [out / "cv" / "oof_predictions.csv", out / "fusion" / "fusion_oof.csv"]
    missing = [str(f) for f in need if not f.is_file()]
    if missing:  # committed in the repository; rebuild with: just cv, just photo-cv, just fusion (see README)
        raise FileNotFoundError(f"cross-validated predictions missing: {missing}; run `just cv photo-cv fusion`")
    cv = pd.read_csv(need[0])
    models = ["prevalence", "profile_lr", "trend_lr", "trend_gbm", "sensor_cnn"]
    how = {"fold": "first", "person_id": "first", "spike": "first", **dict.fromkeys(models, "mean")}
    cv = cv.groupby("row").agg(how)
    fu = pd.read_csv(Path(cfg.paths.outputs) / "fusion" / "fusion_oof.csv").set_index("row")
    df = cv.join(fu.drop(columns=["fold", "person_id", "spike"]), how="inner").reset_index()
    if len(df) != len(cv) or len(df) != len(fu):
        raise ValueError("cross-validation and fusion predictions cover different meals: re-run `just cv`")
    gap = float(np.max(np.abs(df.trend_lr - df.no_meal)))
    logger.bind(stage="evaluate").info("check: trend LR (Step 3c) vs fusion base (4c) differ by at most {:.2e}", gap)
    return df


def run_evaluation(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> dict:
    """Write outputs/evaluation/ (results.json, results_table.csv, results.md, two figures)."""
    log, B, seed = logger.bind(stage="evaluate"), cfg.fusion.bootstrap_draws, cfg.seed
    OUT.mkdir(parents=True, exist_ok=True)
    amended = load_thresholds(AMENDED)
    if not _amendment_committed():
        raise RuntimeError(f"commit {AMENDED} before scoring the test (git add + commit)")
    costs, targets = cost_dict(cfg), cfg.targets.model_dump()

    # 1. main estimate: cross-validation over the 23 development target people
    df = _cv_table(cfg)
    y, sids = df.spike.to_numpy(), df.person_id.to_numpy()
    rows = []
    for label, col in LADDER:
        p = df[col].to_numpy()
        auc, lo, hi = person_bootstrap_ci(sids, lambda i, p=p: _auc(y[i], p[i]), B, seed)
        within = float(np.mean([_auc(f.spike.to_numpy(), f[col].to_numpy()) for _, f in df.groupby("fold")]))
        rows.append({"model": label, "column": col, "auroc": auc, "auroc_lo": lo, "auroc_hi": hi,
                     "auroc_within_fold": within, "auprc": float(average_precision_score(y, p)),
                     "brier": float(brier_score_loss(y, p))})  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "results_table.csv", index=False)

    def gain(a, b):
        """AUROC of a minus b with a paired 95% range by person."""
        m, lo, hi = paired_person_bootstrap(y, df[a].to_numpy(), df[b].to_numpy(), sids, B=B, seed=seed)
        return {"point": _auc(y, df[a].to_numpy()) - _auc(y, df[b].to_numpy()), "ci95": [lo, hi]}

    arms = ["photo", "logged_carbs", "photo_shuffled", "ridge_photo", "photo_grams_only"]
    gains = {c: gain(c, "no_meal") for c in arms}
    gains["bins_added"] = gain("photo", "photo_grams_only")  # what the carb-bin head adds (post-hoc)
    rq1 = gain("photo", "logged_carbs")
    rq1["verdict"] = noninferiority(*rq1["ci95"], cfg.targets.rq1_noninferiority_margin)
    rq2 = gain("sensor_cnn", "no_meal")
    rq2["verdict"] = "CNN better" if rq2["ci95"][0] > cfg.targets.rq2_min_lower_bound else "tie or worse"

    # 2. calibration and usefulness of the main model (cross-validated, main cut-offs use no data)
    pm = df.photo.to_numpy()
    rel, ece = reliability(pm, y, cfg.targets.ece_bins)
    ece_ci = person_bootstrap_ci(sids, lambda i: reliability(pm[i], y[i], cfg.targets.ece_bins)[1], B, seed)
    lo_m, hi_m = amended["main"]["low_risk_below"], amended["main"]["caution_from"]
    use_cv = _usefulness_ci(pm, y, sids, lo_m, hi_m, costs, targets, B, seed)

    # 3. the fixed test, scored once with the frozen 28-person pipeline
    test = _score_test(ds, sp, cfg, amended, costs, targets, B, seed)

    # 4. targets status (rules fixed before training)
    u, margin = cfg.targets.usefulness, cfg.targets.rq1_noninferiority_margin
    status = [
        ("Photo adds to spike prediction (main result)", gains["photo"], lambda v: v > 0),
        ("RQ1: photo not worse than logged carbs by > margin", rq1, lambda v: v > -margin),
        ("RQ2: SensorCNN beats hand-made trends", rq2, lambda v: v > cfg.targets.rq2_min_lower_bound),
    ]  # fmt: skip
    st = [{"target": n, "estimate": g["point"], "ci95": g["ci95"], "status": _status(g["point"], *g["ci95"], ok)}
          for n, g, ok in status]  # fmt: skip
    st.append({"target": f"Calibration: ECE <= {cfg.targets.ece_max}", "estimate": ece, "ci95": list(ece_ci[1:]),
               "status": _status(ece, *ece_ci[1:], lambda v: v <= cfg.targets.ece_max)})  # fmt: skip
    for key, name, ok in [("cost_reduction", f"Cost reduction >= {u.min_cost_reduction:.0%}",
                           lambda v: v >= u.min_cost_reduction),
                          ("share_low_risk", f"Meals cleared >= {u.min_cleared:.0%}", lambda v: v >= u.min_cleared),
                          ("missed_among_cleared", f"Spikes among cleared <= {u.max_missed_among_cleared:.0%}",
                           lambda v: v <= u.max_missed_among_cleared),
                          ("share_abstain", f"Can't judge <= {u.max_abstain:.0%}",
                           lambda v: v <= u.max_abstain)]:  # fmt: skip
        st.append({"target": name + " (CV, main cut-offs)", "estimate": use_cv[key], "ci95": use_cv[f"{key}_ci95"],
                   "status": _status(use_cv[key], *use_cv[f"{key}_ci95"], ok)})  # fmt: skip
    st.append({"target": f"Latency p95 <= {cfg.targets.latency_p95_ms:.0f} ms", "estimate": None, "ci95": None,
               "status": "measured with the demo (Step 6)"})  # fmt: skip

    res = {"cv_people": int(len(set(sids))), "cv_meals": int(len(df)), "cv_spike_rate": float(y.mean()),
           "table": rows, "gains_vs_profile_trend": gains, "rq1": rq1, "rq2": rq2,
           "calibration": {"ece": ece, "ece_ci95": list(ece_ci[1:]), "bins": rel},
           "usefulness_cv_main_cutoffs": use_cv, "cutoffs": amended, "test": test, "targets_status": st}  # fmt: skip
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    _figures(table, rel, ece)
    tc = test["test"]["calibration"]
    _reliability_figure(tc["bins"], tc["ece"], "Main model, fixed test (5 people, 96 meals)",
                        OUT / "fig3_reliability_test.png")  # fmt: skip
    _markdown(res)
    _log(log, res)
    return res


def _score_test(ds, sp, cfg, amended, costs, targets, B, seed) -> dict:
    """The 28-person pipeline on the 5 test people, with all three frozen cut-off rows and the cost sweep."""
    bundle = joblib.load(FINAL)
    marker = OUT / "test_scored.json"
    stamp = {"amendment_sha256": amended["sha256"], "model_sha256": _sha(FINAL)}
    if marker.is_file() and json.loads(marker.read_text()) != stamp:
        raise RuntimeError("the test was already scored with a different model or cut-offs; it is scored once")
    groups = {s.person_id: s.group for s in ds.samples}
    rows_all = photo_rows(ds, set(groups) - set(sp.locked), Path(cfg.data.cgmacros_root))
    model = PhotoCarbModel(cfg.photo.arch, weights=weights_path(cfg.photo), dropout=cfg.photo.dropout)
    maps = build_cache(model, ds, rows_all, cfg, "maps_eval", Path("data/cache/photo_cv_maps_eval.pt"))
    table = photo_table(checkpoint_estimates(cfg, ds, rows_all, maps, set(sp.validation) | set(sp.test)))
    fx = {"no_meal": trend_features, "photo": with_table(table, len(PHOTO_COLS))}
    out = {}
    for name, people in (("validation", sp.validation), ("test", sp.test)):
        r = meal_rows(ds, set(people), scored_only=True)
        y, s = ds.labels()[r], np.array([ds.samples[i].person_id for i in r])
        p = {a: bundle["models"][a].predict_proba(fx[a](ds, r))[:, 1] for a in fx}
        out[name] = {"people": len(people), "meals": len(r), "spike_rate": float(y.mean()),
                     "mean_predicted_risk": float(p["photo"].mean())}  # fmt: skip
        if name == "validation":  # calibration check of the 28-person model, before the test numbers
            continue
        for a in fx:
            out[name][f"auroc_{a}"] = list(person_bootstrap_ci(s, lambda i, q=p[a], t=y: _auc(t[i], q[i]), B, seed))
        out[name]["per_person"] = [{"person": int(q), "meals": int((s == q).sum()),
                                    "spike_rate": float(y[s == q].mean()),
                                    "auroc_photo": _auc(y[s == q], p["photo"][s == q])}
                                   for q in np.unique(s)]  # fmt: skip
        cut = {"main (cost-only)": amended["main"], "pre-registered (validation)": amended["secondary_preregistered"],
               "CV without test people": amended["secondary_cv_without_test"]}  # fmt: skip
        out[name]["decisions"] = {k: {"low_risk_below": v["low_risk_below"], "caution_from": v["caution_from"],
                                      **_usefulness_ci(p["photo"], y, s, v["low_risk_below"], v["caution_from"], costs,
                                                       targets, B, seed)} for k, v in cut.items()}  # fmt: skip
        sweep = []
        for sc in _sweep_costs(cfg):
            lo, hi = cost_only_thresholds(sc["costs"])
            act = decide(p["photo"], lo, hi)
            c = sc["costs"]
            trivial = min(expected_cost(np.zeros_like(y), y, c), expected_cost(np.ones_like(y), y, c))
            sweep.append({"missed_spike": sc["missed_spike"], "abstain_multiplier": sc["abstain_multiplier"],
                          "low_risk_below": lo, "caution_from": hi,
                          "cost_reduction": 1 - expected_cost(act, y, sc["costs"]) / trivial})  # fmt: skip
        out[name]["cost_sweep_main_rule"] = sweep
        out[name]["table"] = _test_table(ds, sp, cfg, r, y, s, p, B, seed)
        rel_t, ece_t = reliability(p["photo"], y, cfg.targets.ece_bins)
        ece_t_ci = person_bootstrap_ci(s, lambda i, q=p["photo"], t=y: reliability(q[i], t[i], cfg.targets.ece_bins)[1],
                                       B, seed)  # fmt: skip
        out[name]["calibration"] = {"ece": ece_t, "ece_ci95": list(ece_t_ci[1:]), "bins": rel_t}
    marker.write_text(json.dumps(stamp, indent=2))
    return out


def _test_table(ds, sp, cfg, r, y, s, p, B, seed) -> list[dict]:
    """Every baseline and the main model on the SAME fixed test meals with the same metrics. Baselines are refitted
    on the same 28 training people (one setting tuned on the same 4 inner folds); the SensorCNN is its saved
    28-person checkpoint. Nothing here is tuned on the test people."""
    from glimpse.training.baselines import gbm, logistic, profile_features
    from glimpse.training.cv import _tune_sklearn
    from glimpse.training.scratch import _inputs, load_bundle, predict_proba

    y_all, tr, b = ds.labels(), meal_rows(ds, set(sp.train), scored_only=False), cfg.baselines
    c_prof, _ = _tune_sklearn(lambda c: logistic(c, cfg.seed), b.logreg_c_grid, profile_features, ds, sp.train,
                              sp.inner_folds, y_all)  # fmt: skip
    depth, _ = _tune_sklearn(lambda d: gbm(d, cfg.seed), b.gbm_depth_grid, trend_features, ds, sp.train,
                             sp.inner_folds, y_all)  # fmt: skip
    cnn, norm, _ = load_bundle(Path("checkpoints") / Path("checkpoints/sensor_cnn_latest.txt").read_text().strip())
    preds = {
        "prevalence": np.full(len(r), y_all[tr].mean()),
        "profile_lr": logistic(c_prof, cfg.seed).fit(profile_features(ds, tr), y_all[tr]).predict_proba(
            profile_features(ds, r))[:, 1],
        "no_meal": p["no_meal"],
        "trend_gbm": gbm(depth, cfg.seed).fit(trend_features(ds, tr), y_all[tr]).predict_proba(
            trend_features(ds, r))[:, 1],
        "sensor_cnn": predict_proba(cnn, _inputs(ds, r, norm)),
        "photo": p["photo"],
    }  # fmt: skip
    labels = dict(LADDER)
    rows = []
    for col, q in preds.items():
        auc, lo, hi = person_bootstrap_ci(s, lambda i, q=q: _auc(y[i], q[i]), B, seed)
        rows.append({"model": next(k for k, v in labels.items() if v == col), "column": col, "auroc": auc,
                     "auroc_lo": lo, "auroc_hi": hi, "auprc": float(average_precision_score(y, q)),
                     "brier": float(brier_score_loss(y, q)),
                     "ece": reliability(q, y, cfg.targets.ece_bins)[1]})  # fmt: skip
    pd.DataFrame(rows).to_csv(OUT / "test_table.csv", index=False)
    return rows


def _reliability_figure(rel: list, ece: float, title: str, path: Path) -> None:
    """Reliability diagram: predicted risk vs observed spike rate per equal-width bin, meals per bin printed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    m, o, n = zip(*rel, strict=True)
    ax.plot([0, 1], [0, 1], ls="--", c="#95a5a6", lw=1)
    ax.scatter(m, o, s=[10 + 2 * k for k in n], c="#c0392b", alpha=0.8)
    for x_, y_, k in zip(m, o, n, strict=True):
        ax.annotate(str(k), (x_, y_), textcoords="offset points", xytext=(6, -10), fontsize=7, color="#555")
    ax.set_xlabel("predicted spike risk (mean in bin)")
    ax.set_ylabel("observed spike rate in bin")
    ax.set_title(f"{title}\nECE {ece:.3f}; 15 equal-width bins; numbers = meals", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _figures(table: pd.DataFrame, rel: list, ece: float) -> None:
    """Figure 1: AUROC ladder with 95% ranges. Figure 2: reliability diagram of the main model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    grey = ("photo_shuffled", "ridge_photo", "photo_grams_only")
    colors = ["#c0392b" if c == "photo" else "#7f8c8d" if c in grey else "#2c3e50" for c in t.column]
    for k, (r, col) in enumerate(zip(t.itertuples(), colors, strict=True)):
        ax.errorbar(r.auroc, k, xerr=[[r.auroc - r.auroc_lo], [r.auroc_hi - r.auroc]], fmt="o", color=col, capsize=3)
    base = float(table.loc[table.column == "no_meal", "auroc"].iloc[0])
    ax.axvline(base, ls="--", c="#95a5a6", lw=1)
    ax.axvline(0.5, ls=":", c="#bdc3c7", lw=1)
    ax.set_yticks(range(len(t)), t.model, fontsize=8)
    ax.set_xlabel("AUROC, 5-fold person cross-validation (23 people)\nbars = 95% range by person", fontsize=9)
    ax.set_title("Spike warning: what each input adds", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_auroc_ladder.png", dpi=200)
    plt.close(fig)

    _reliability_figure(rel, ece, "Main model, cross-validation (23 people, 481 meals)", OUT / "fig2_reliability.png")


def _markdown(res: dict) -> None:
    """results.md: the tables in a form that can be pasted into the report."""
    L = ["## Cross-validation (main estimate)", "",
         "| Model | AUROC (95% range) | within folds | AUPRC | Brier |", "|---|---|---|---|---|"]  # fmt: skip
    for r in res["table"]:
        L.append(f"| {r['model']} | {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) | "
                 f"{r['auroc_within_fold']:.3f} | {r['auprc']:.3f} | {r['brier']:.3f} |")  # fmt: skip
    L += ["", f"Base spike rate {res['cv_spike_rate']:.0%} ({res['cv_meals']} meals, {res['cv_people']} people).", "",
          "## Targets status", "", "| Target | Estimate | 95% range | Status |", "|---|---|---|---|"]  # fmt: skip
    for s in res["targets_status"]:
        est = "-" if s["estimate"] is None else f"{s['estimate']:+.3f}"
        ci = "-" if s["ci95"] is None else f"{s['ci95'][0]:+.3f} to {s['ci95'][1]:+.3f}"
        L.append(f"| {s['target']} | {est} | {ci} | {s['status']} |")
    t = res["test"]["test"]
    head = f"## Fixed test (sanity check): {t['people']} people, {t['meals']} meals, spike rate {t['spike_rate']:.0%}"
    L += [
        "",
        head,
        "",
        "| Model (fitted on the 28 training people) | AUROC (95% range) | AUPRC | Brier | ECE |",
        "|---|---|---|---|---|",
    ]
    for r in t["table"]:
        L.append(f"| {r['model']} | {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) | {r['auprc']:.3f} | "
                 f"{r['brier']:.3f} | {r['ece']:.3f} |")  # fmt: skip
    L += ["",
          "| Cut-offs | low risk below | caution from | cleared | spikes among cleared | can't judge | cost change |",
          "|---|---|---|---|---|---|---|"]  # fmt: skip
    for k, d in t["decisions"].items():
        L.append(f"| {k} | {d['low_risk_below']:.3f} | {d['caution_from']:.3f} | {d['share_low_risk']:.0%} | "
                 f"{d['missed_among_cleared']:.0%} | {d['share_abstain']:.0%} | "
                 f"{-d['cost_reduction']:+.0%} |")  # fmt: skip
    (OUT / "results.md").write_text("\n".join(L) + "\n")


def _log(log, res: dict) -> None:
    for r in res["table"]:
        log.info("{:<40} AUROC {:.3f} ({:.3f}-{:.3f}) | within folds {:.3f}", r["model"], r["auroc"], r["auroc_lo"],
                 r["auroc_hi"], r["auroc_within_fold"])  # fmt: skip
    for k, g in res["gains_vs_profile_trend"].items():
        log.info("gain {:<15} {:+.3f} ({:+.3f} to {:+.3f})", k, g["point"], *g["ci95"])
    log.info("RQ1 photo - logged {:+.3f} ({:+.3f} to {:+.3f}) -> {}", res["rq1"]["point"], *res["rq1"]["ci95"],
             res["rq1"]["verdict"])  # fmt: skip
    log.info("RQ2 CNN - trend LR {:+.3f} ({:+.3f} to {:+.3f}) -> {}", res["rq2"]["point"], *res["rq2"]["ci95"],
             res["rq2"]["verdict"])  # fmt: skip
    c = res["calibration"]
    log.info("calibration ECE {:.3f} ({:.3f} to {:.3f})", c["ece"], *c["ece_ci95"])
    v = res["test"]["validation"]
    log.info("28-person model on validation: mean predicted {:.0%} vs observed {:.0%}", v["mean_predicted_risk"],
             v["spike_rate"])  # fmt: skip
    t = res["test"]["test"]
    log.info("TEST ({} people, {} meals, spike rate {:.0%}): AUROC photo {:.3f} ({:.3f}-{:.3f}), no meal {:.3f}",
             t["people"], t["meals"], t["spike_rate"], *t["auroc_photo"], t["auroc_no_meal"][0])  # fmt: skip
    for r in t["table"]:
        log.info("test, {:<38} AUROC {:.3f} ({:.3f}-{:.3f}) | ECE {:.3f}", r["model"], r["auroc"], r["auroc_lo"],
                 r["auroc_hi"], r["ece"])  # fmt: skip
    log.info("test calibration of the main model: ECE {:.3f} ({:.3f} to {:.3f})", t["calibration"]["ece"],
             *t["calibration"]["ece_ci95"])  # fmt: skip
    for k, d in t["decisions"].items():
        log.info("test, {:<28} cleared {:.0%}, spikes among cleared {:.0%}, can't judge {:.0%}, cost {:+.0%} "
                 "vs best simple policy", k, d["share_low_risk"], d["missed_among_cleared"], d["share_abstain"],
                 -d["cost_reduction"])  # fmt: skip
    for s in res["targets_status"]:
        log.info("target: {:<55} {}", s["target"], s["status"])
    log.info("written to {}/ (results, tables and figures)", OUT)
