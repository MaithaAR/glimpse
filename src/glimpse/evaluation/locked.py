"""`glimpse evaluate-locked` (S16): the 8 locked people, scored exactly once against the frozen
artifacts through the same Pipeline.answer as everything else, each person's meals replayed in
deployment order under the 2-hour label gate. The guard refuses a second run: the committed first
result is the result, reported as measured (docs/final_design.md D7). Nothing here tunes anything."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import log_loss, roc_auc_score

from glimpse.calibration import LABEL_LAG_MIN
from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import SplitsV2, meal_rows
from glimpse.evaluation.metrics import ece_equal_mass, person_bootstrap_ci, usefulness_report
from glimpse.serve import Pipeline
from glimpse.training.final import cost_dict

OUT = Path("outputs/locked_eval")


def run_locked(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig) -> dict:
    log = logger.bind(stage="locked")
    if (OUT / "locked_answers.csv").exists():
        raise RuntimeError("the locked set has already been scored once (outputs/locked_eval exists); "
                           "the pre-registration forbids a second run")  # fmt: skip
    from glimpse.demo_cases import meal_input

    OUT.mkdir(parents=True, exist_ok=True)
    app, y, lag = Pipeline(cfg), ds.labels(), timedelta(minutes=LABEL_LAG_MIN)
    rows_out = []
    for pid in sorted(sp.locked):
        hist: list[tuple[float, int, object]] = []
        for row in sorted(meal_rows(ds, {pid}, scored_only=True), key=lambda r: ds.samples[r].meal_time):
            t = ds.samples[row].meal_time
            gated = [(z, lab) for z, lab, tm in hist if tm + lag <= t]
            a = app.answer(meal_input(cfg, ds, int(row)), key=int(row), history=gated or None)
            rows_out.append({"row": int(row), "person_id": int(pid), "spike": int(y[row]),
                             "risk": a.risk, "risk_raw": a.risk_raw, "n_history": len(gated),
                             "decision": a.decision, "carb_source": a.carb_source, "reason": a.reason})  # fmt: skip
            if a.fusion_logit is not None:
                hist.append((a.fusion_logit, int(y[row]), t))
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT / "locked_answers.csv", index=False)
    ans = df.dropna(subset=["risk"])
    yv, s_ids = ans.spike.to_numpy(), ans.person_id.to_numpy()

    def _auc(col: str) -> dict:
        pv = ans[col].to_numpy(float)
        point, lo, hi = person_bootstrap_ci(s_ids, lambda i: float(roc_auc_score(yv[i], pv[i]))
                                            if yv[i].min() != yv[i].max() else float("nan"))  # fmt: skip
        return {"auroc": {"point": point, "ci95": [lo, hi]},
                "log_loss": float(log_loss(yv, pv, labels=[0, 1])),
                "ece_equal_mass": {"bins15": float(ece_equal_mass(pv, yv)),
                                   "bins10": float(ece_equal_mass(pv, yv, 10))}}  # fmt: skip

    act = df.decision.map({"low risk": 0, "caution": 1, "can't judge": 2}).to_numpy()
    use = usefulness_report(act, df.spike.to_numpy(), cost_dict(cfg), cfg.targets.model_dump())
    cl = df[df.decision == "low risk"].reset_index(drop=True)
    mp, mlo, mhi = person_bootstrap_ci(cl.person_id.to_numpy(),
                                       lambda i: float(np.mean(cl.spike.to_numpy()[i])))  # fmt: skip
    res = {"what": "the 8 locked people, scored once against the frozen artifacts (S16); every number "
                   "as measured, per the pre-registration",
           "people": int(df.person_id.nunique()), "meals": int(len(df)), "answered": int(len(ans)),
           "spike_rate": float(df.spike.mean()),
           "risk_personal": _auc("risk"), "risk_raw": _auc("risk_raw"),
           "decisions": df.decision.value_counts().to_dict(),
           "missed_among_cleared": {"point": mp, "ci95_person_bootstrap": [mlo, mhi],
                                    "misses": int(cl.spike.sum()), "cleared": int(len(cl))},
           "usefulness": use}  # fmt: skip
    (OUT / "locked_summary.json").write_text(json.dumps(res, indent=2))
    log.info("locked 8: {} meals ({} answered), spike rate {:.1%}; personal AUROC {:.3f} ({:.3f}-{:.3f}), "
             "raw {:.3f}; decisions {}; missed-among-cleared {}/{} ({:.1%}); cost reduction {:.1%} -> {}",
             len(df), len(ans), df.spike.mean(), res["risk_personal"]["auroc"]["point"],
             *res["risk_personal"]["auroc"]["ci95"], res["risk_raw"]["auroc"]["point"], res["decisions"],
             res["missed_among_cleared"]["misses"], res["missed_among_cleared"]["cleared"], mp,
             use["cost_reduction"], OUT / "locked_summary.json")  # fmt: skip
    return res
