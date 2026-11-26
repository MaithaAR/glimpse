"""Rule R22 and the R14 amendment (docs/rule_ledger.md, registered 26 Nov before any run): their effect counted
on the committed development answers of both products (`glimpse rules-r22`). The rules are applied to each
answer with approximately the serving code's last reading and 30-min slope (from the 5-min slot means); risks
are unchanged. On the locked eight only the number of affected clears is counted; their outcomes are not
read."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from glimpse.evaluation.report_metrics import policies
from glimpse.serve.rules import LOW_GLUCOSE_REFUSE, LOW_GLUCOSE_WARN, premeal_rule, slope_last_30

OUT = Path("outputs/rules_r22")
FILES = {"v1": "outputs/evaluation_v2/dev_answers.csv", "v2": "outputs/evaluation_sensor_first/dev_answers.csv"}


def _last_and_slope(ds, row: int) -> tuple[float, float | None]:
    g = ds.glucose[row]
    t0 = datetime(2000, 1, 1)
    times = [t0 + timedelta(minutes=5 * i) for i in range(len(g)) if np.isfinite(g[i])]
    vals = [float(v) for v in g if np.isfinite(v)]
    return vals[-1], slope_last_30(times, vals)


def apply_rules(d: pd.DataFrame, ds) -> tuple[pd.DataFrame, dict]:
    d = d.copy()
    ls = [_last_and_slope(ds, int(r)) for r in d.row]
    d["last"], d["slope"] = [x[0] for x in ls], [x[1] for x in ls]
    falling_low = ((d["last"] >= LOW_GLUCOSE_REFUSE) & (d["last"] < LOW_GLUCOSE_WARN)
                   & (d["slope"].fillna(0) < 0) & (d.decision != "can't judge"))  # fmt: skip
    r22 = (d.decision == "low risk") & ~falling_low & np.array([premeal_rule(a, b) is not None for a, b in ls])
    new = d.decision.where(~falling_low, "can't judge").where(~r22, "caution")
    counts = {"r14_falling_low_refused": int(falling_low.sum()), "r22_clears_to_caution": int(r22.sum())}
    return d.assign(decision_new=new, r22=r22, r14=falling_low), counts


def _useful(d: pd.DataFrame, col: str, costs: dict) -> dict:
    x = d.assign(decision=d[col])  # all meals, as the report counts them (no risk: refused)
    p = policies(x, costs)
    cl = x.decision == "low risk"
    return {"cost_reduction": p["cost_reduction_vs_best_constant"], "cleared_share": float(cl.mean()),
            "cant_judge_share": float((x.decision == "can't judge").mean()), "cleared": int(cl.sum()),
            "missed_among_cleared": float(x.spike[cl].mean()) if cl.any() else None,
            "missed": int(x.spike[cl].sum())}  # fmt: skip


def run_rules_r22(cfg, ds) -> dict:
    from glimpse.training.final import cost_dict

    costs, res = cost_dict(cfg), {"what": __doc__.splitlines()[0]}
    for name, f in FILES.items():
        d, counts = apply_rules(pd.read_csv(f), ds)
        res[name] = {**counts, "spiked_among_r22": int(d.spike[d.r22].sum()),
                     "before": _useful(d, "decision", costs), "after": _useful(d, "decision_new", costs)}  # fmt: skip
    lk, counts = apply_rules(pd.read_csv("outputs/locked_eval/locked_answers.csv"), ds)
    res["locked_counted_only"] = counts  # outcomes of the affected locked meals are not read
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rules_r22.json").write_text(json.dumps(res, indent=2))
    logger.bind(stage="rules-r22").info("{}", res)
    return res
