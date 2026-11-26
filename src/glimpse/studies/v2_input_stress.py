"""v2 under stress on the inputs it uses (`glimpse v2-input-stress`; registered in docs/v2_explain.md, V2-S).

Every development meal answered by its fold's v2 pipeline (the evaluation's recipe) with gated history from
the clean answers, then again with heart rate removed or halved, the onboarding priors removed, the history
cut to its last 3 or 10 meals, or 10% / 25% of history labels flipped. Descriptive; nothing is chosen."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from glimpse.calibration import LABEL_LAG_MIN
from glimpse.data.splits import meal_rows
from glimpse.studies.stress_quality import _fold_pipelines, _summary
from glimpse.training.final import cost_dict

OUT = Path("outputs/robustness/v2_input_stress.json")
CONDITIONS = ("hr_removed", "hr_half", "priors_removed", "history_last_3", "history_last_10",
              "history_labels_10pct_flipped", "history_labels_25pct_flipped")  # fmt: skip


def _meal(m, cond: str, rng: np.random.Generator):
    if cond == "hr_removed":
        return m.model_copy(update={"hr_times": [], "hr_bpm": []})
    if cond == "hr_half":
        keep = rng.uniform(size=len(m.hr_times)) > 0.5
        return m.model_copy(update={"hr_times": [t for t, k in zip(m.hr_times, keep, strict=True) if k],
                                    "hr_bpm": [v for v, k in zip(m.hr_bpm, keep, strict=True) if k]})  # fmt: skip
    if cond == "priors_removed":
        return m.model_copy(update={"profile": m.profile | {"prior_spike_rate": float("nan"),
                                                            "prior_rise": float("nan")}})  # fmt: skip
    return m


def _history(gated, cond: str, rng: np.random.Generator):
    if not gated or not cond.startswith("history"):
        return gated
    if cond == "history_last_3":
        return gated[-3:]
    if cond == "history_last_10":
        return gated[-10:]
    share = 0.10 if "10pct" in cond else 0.25
    return [(z, 1 - lab if rng.uniform() < share else lab) for z, lab in gated]


def run_v2_input_stress(ds, sp, cfg) -> dict:
    from glimpse.demo_cases import meal_input

    log, pcv, y = logger.bind(stage="v2-input-stress"), Path(cfg.paths.outputs) / "photo_cv", ds.labels()
    oof, nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
    costs, targets, lag = cost_dict(cfg), cfg.targets.model_dump(), timedelta(minutes=LABEL_LAG_MIN)
    rng = np.random.default_rng(cfg.seed)
    rows: dict[str, list] = {c: [] for c in ("clean", *CONDITIONS)}
    for k, held in enumerate(sp.dev_folds):
        tab, _ = _fold_pipelines(ds, sp, cfg, k, True, oof, nested)
        for pid in sorted(held):
            hist: list = []
            for row in sorted(meal_rows(ds, {pid}, scored_only=True), key=lambda r: ds.samples[r].meal_time):
                t = ds.samples[row].meal_time
                gated = [(z, lab) for z, lab, tm in hist if tm + lag <= t] or None
                m = meal_input(cfg, ds, int(row))
                meta = {"row": int(row), "person_id": int(pid), "spike": int(y[row])}
                clean = tab.answer(m, key=int(row), history=gated)
                rows["clean"].append(meta | {"risk": clean.risk, "decision": clean.decision})
                for c in CONDITIONS:
                    a = tab.answer(_meal(m, c, rng), key=int(row), history=_history(gated, c, rng))
                    rows[c].append(meta | {"risk": a.risk, "decision": a.decision,
                                           "shift": None if a.risk is None or clean.risk is None
                                           else a.risk - clean.risk})  # fmt: skip
                if clean.fusion_logit is not None:
                    hist.append((clean.fusion_logit, int(y[row]), t))
        log.info("fold {} done", k + 1)
    base = pd.DataFrame(rows["clean"])
    res: dict = {"what": __doc__.splitlines()[0], "clean": _summary(base, base, costs, targets)}
    for c in CONDITIONS:
        p = pd.DataFrame(rows[c])
        res[c] = _summary(base, p, costs, targets) | {
            "mean_risk_shift": float(p["shift"].dropna().mean()),
            "share_risk_lower": float((p["shift"].dropna() < 0).mean()),
        }
        log.info("{:<30} AUROC {:.3f} -> {:.3f} | caution->low {} low->caution {} | mean shift {:+.3f}", c,
                 res[c]["auroc_clean"], res[c]["auroc_perturbed"], res[c]["caution_to_low_risk"],
                 res[c]["low_risk_to_caution"], res[c]["mean_risk_shift"])  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    return res
