"""Evaluation v2 on partition v2 (designs D1, D2): every reported development risk is produced by
`Pipeline.answer`, one Pipeline per outer fold with that fold's fusion and precomputed photo estimates,
so evaluation and deployment are the same code path by construction. A held-out meal without a photo
becomes the same 'can't judge' the app would give; nothing is filled in. Decision cut-offs are the
pre-S5 frozen ones, so decisions here are provisional: the reported numbers at this stage are AUROC
(person-clustered bootstrap) and log-loss only (D4 lands in S5). Report: §3."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics import log_loss, roc_auc_score

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import SplitsV2, assert_not_locked, meal_rows
from glimpse.evaluation.metrics import person_bootstrap_ci
from glimpse.serve import PhotoEstimate, Pipeline
from glimpse.training.baselines import logistic
from glimpse.training.cv import _tune_sklearn
from glimpse.training.fusion import PHOTO_COLS, photo_table, with_table

OUT = Path("outputs/evaluation_v2")


def fold_estimates(df: pd.DataFrame) -> dict[int, PhotoEstimate]:
    """Out-of-fold photo CV rows -> the Pipeline's precomputed-estimate table for the held-out people."""
    return {int(r.row): PhotoEstimate(grams=float(r.photo_g),
                                      bin_probs=[float(r.p_bin0), float(r.p_bin1), float(r.p_bin2),
                                                 float(r.p_bin3)])
            for r in df.itertuples()}  # fmt: skip


def fold_fusion(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig, k: int, pt: dict) -> tuple:
    """Fold k's fusion: C chosen on the fold's inner folds (target holds, D1), fitted on its training meals
    with nested photo estimates -- features in exactly the order Pipeline.answer builds them."""
    y = ds.labels()
    fx = with_table(pt, len(PHOTO_COLS))
    c, _ = _tune_sklearn(lambda v: logistic(v, cfg.seed), cfg.baselines.logreg_c_grid, fx, ds,
                         sp.training_people(k), sp.inner_folds[k], y)  # fmt: skip
    tr = meal_rows(ds, set(sp.training_people(k)), scored_only=False)
    return logistic(c, cfg.seed).fit(fx(ds, tr), y[tr]), c


def evaluate_dev(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig) -> dict:
    """Every development meal answered by its fold's Pipeline -> dev_answers.csv + dev_summary.json."""
    from glimpse.demo_cases import meal_input

    log, pcv = logger.bind(stage="evaluate"), Path(cfg.paths.outputs) / "photo_cv"
    if not (pcv / "photo_oof.csv").is_file():
        raise FileNotFoundError("run `just photo-cv` first")
    oof, nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
    y, rows_out, chosen = ds.labels(), [], {}
    for k, held in enumerate(sp.dev_folds):
        kk = k + 1
        of, nf = oof[oof.fold == kk], nested[nested.fold == kk]
        if of.empty or not set(of.person_id) <= set(held):  # provenance: must be the partition-v2 run
            raise RuntimeError(f"fold {kk}: photo CV outputs do not match splits/dev_splits_v2.json; "
                               f"rerun `just photo-cv`")  # fmt: skip
        assert_not_locked(list(sp.training_people(k)) + list(held), sp.locked, f"evaluate fold {kk}")
        lr, chosen[f"fold{kk}_C"] = fold_fusion(ds, sp, cfg, k, photo_table(nf) | photo_table(of))
        app = Pipeline(cfg, photo_table=fold_estimates(of), fusion=lr)
        for row in meal_rows(ds, set(held), scored_only=True):
            a = app.answer(meal_input(cfg, ds, int(row)), key=int(row))
            rows_out.append({"fold": kk, "row": int(row), "person_id": int(ds.samples[row].person_id),
                             "spike": int(y[row]), "risk": a.risk, "decision": a.decision,
                             "carb_source": a.carb_source, "reason": a.reason})  # fmt: skip
        log.info("fold {}: {} meals answered (C={})", kk, int((pd.DataFrame(rows_out).fold == kk).sum()),
                 chosen[f"fold{kk}_C"])  # fmt: skip
    df = pd.DataFrame(rows_out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "dev_answers.csv", index=False)
    sc = df.dropna(subset=["risk"])
    s, yy, p = sc.person_id.to_numpy(), sc.spike.to_numpy(), sc.risk.to_numpy()
    B, seed = cfg.fusion.bootstrap_draws, cfg.seed
    auc = person_bootstrap_ci(s, lambda i: roc_auc_score(yy[i], p[i]), B, seed)
    per_fold = [{"fold": int(kk), "meals": int(len(g)), "answered": int(g.risk.notna().sum()),
                 "auroc": float(roc_auc_score(g.dropna(subset=["risk"]).spike, g.dropna(subset=["risk"]).risk))}
                for kk, g in df.groupby("fold")]  # fmt: skip
    res = {
        "note": "risks come from Pipeline.answer, one Pipeline per fold (D2); decisions use the pre-S5 "
                "frozen cut-offs and are PROVISIONAL -- the reported numbers here are AUROC and log-loss only",
        "meals": int(len(df)), "answered_with_risk": int(len(sc)),
        "cant_judge": int((df.decision == "can't judge").sum()),
        "auroc_photo_dev": {"point": auc[0], "ci95": [auc[1], auc[2]]},
        "log_loss_photo_dev": float(log_loss(yy, p, labels=[0, 1])),
        "per_fold": per_fold, "fusion_C": chosen,
    }  # fmt: skip
    (OUT / "dev_summary.json").write_text(json.dumps(res, indent=2))
    log.info("development (23 people, {} meals, {} answered): AUROC {:.3f} ({:.3f}-{:.3f}), log-loss {:.3f}",
             res["meals"], res["answered_with_risk"], auc[0], auc[1], auc[2], res["log_loss_photo_dev"])  # fmt: skip
    log.info("can't judge: {} meals (no usable photo or borderline; decisions provisional until S5)",
             res["cant_judge"])  # fmt: skip
    return res
