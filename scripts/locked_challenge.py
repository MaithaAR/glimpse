"""Development-only challenge of the locked result (docs/locked_challenge.md): how often do 8 of the 23
development people give a result as poor as the locked eight? Reads committed answers; changes nothing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ACTION = {"low risk": 0, "caution": 1, "can't judge": 2}
REF = {"auroc": 0.645, "cleared": 0.065, "cost_reduction": -0.057, "spike_rate": 0.40}


def cost(actions: np.ndarray, y: np.ndarray, c: np.ndarray) -> float:
    return float(np.mean(c[actions, np.where(y == 1, 0, 1)]))


def one(df: pd.DataFrame, c: np.ndarray, u: dict) -> dict:
    y = df.spike.to_numpy().astype(int)
    a = df.decision.map(ACTION).to_numpy()
    trivial = min(cost(np.zeros_like(y), y, c), cost(np.ones_like(y), y, c))
    cleared = a == 0
    missed = float(y[cleared].mean()) if cleared.any() else float("nan")
    cr = 1 - cost(a, y, c) / trivial
    met = [cr >= u["min_cost_reduction"], cleared.mean() >= u["min_cleared"],
           bool(cleared.any()) and missed <= u["max_missed_among_cleared"], (a == 2).mean() <= u["max_abstain"]]  # fmt: skip
    s = df[df.risk.notna()]
    auc = roc_auc_score(s.spike, s.risk) if s.spike.nunique() == 2 else float("nan")
    return dict(auroc=auc, cleared=float(cleared.mean()), cost_reduction=cr, missed=missed, met=int(sum(met)),
                spike_rate=float(y.mean()))  # fmt: skip


def shares(d: pd.DataFrame) -> dict:
    return dict(draws=len(d), at_most_1_of_4_targets=float((d.met <= 1).mean()),
                auroc_at_or_below_locked=float((d.auroc <= REF["auroc"]).mean()),
                cleared_at_or_below_locked=float((d.cleared <= REF["cleared"]).mean()),
                cost_reduction_at_or_below_locked=float((d.cost_reduction <= REF["cost_reduction"]).mean()),
                auroc_median=float(d.auroc.median()), auroc_p2_5=float(d.auroc.quantile(0.025)),
                auroc_p97_5=float(d.auroc.quantile(0.975)), targets_met_mean=float(d.met.mean()))  # fmt: skip


def main() -> None:
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    c = np.array([cfg["costs"]["low_risk"], cfg["costs"]["caution"], cfg["costs"]["abstain"]], float)
    u = cfg["targets"]["usefulness"]
    df = pd.read_csv("outputs/evaluation_v2/dev_answers.csv")
    people = np.sort(df.person_id.unique())
    rng = np.random.default_rng(20260927)
    rows = [one(df[df.person_id.isin(rng.choice(people, 8, replace=False))], c, u) for _ in range(5000)]
    d = pd.DataFrame(rows)
    out = dict(what="8 of 23 development people, 5,000 draws without replacement; served answers before R22",
               people=int(len(people)), meals=int(len(df)), all_development=one(df, c, u), reference_locked=REF,
               all_draws=shares(d), draws_with_spike_rate_at_least_40pct=shares(d[d.spike_rate >= REF["spike_rate"]]))  # fmt: skip
    Path("outputs/locked_challenge").mkdir(parents=True, exist_ok=True)
    Path("outputs/locked_challenge/locked_challenge.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
