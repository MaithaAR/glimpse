"""Replay of the outcome and behaviour monitoring rules on the logged answers (`glimpse monitoring-replay`).

Rules registered in docs/monitoring.md (26 Nov) before this ran. Reads only the committed answer files:
dev answers in `row` order, then the single locked run's answers in `row` order. A week is a block of 20
answers; a 30-day window is four blocks. Descriptive: nothing is refitted and no threshold changes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score

OUT = Path("outputs/monitoring")
BLOCK, WINDOW, MIN_CLEARED = 20, 4, 10
LOCKED_AUROC = 0.645  # docs/monitoring.md signal 2: the locked-set value


def _blocks(df: pd.DataFrame, stream: str) -> list[pd.DataFrame]:
    """Consecutive blocks of 20 answers in `row` order; a final short block is dropped (as registered)."""
    df = df.sort_values("row").reset_index(drop=True)
    return [df.iloc[i : i + BLOCK].assign(stream=stream) for i in range(0, len(df) - BLOCK + 1, BLOCK)]


def run_monitoring_replay(cfg) -> dict:
    """Weekly behaviour shares and rolling outcome signals over dev then locked answers."""
    cap = cfg.targets.usefulness.max_missed_among_cleared
    dev = pd.read_csv("outputs/evaluation_v2/dev_answers.csv")
    locked = pd.read_csv("outputs/locked_eval/locked_answers.csv")
    blocks = _blocks(dev, "dev") + _blocks(locked, "locked")
    weeks = []
    for i, b in enumerate(blocks):
        shares = {k: float((b.decision == k).mean()) for k in ("low risk", "caution", "can't judge")}
        weeks.append({"week": i + 1, "stream": b.stream.iloc[0], **shares})
    dev_max_caution = max(w["caution"] for w in weeks if w["stream"] == "dev")
    for w in weeks:
        w["behaviour_alert"] = w["caution"] > dev_max_caution
    windows = []
    for i in range(len(blocks) - WINDOW + 1):
        w = pd.concat(blocks[i : i + WINDOW])
        streams = set(w.stream)
        cleared = w[w.decision == "low risk"]
        scored = w[w.risk.notna()]
        row = {"weeks": f"{i + 1}-{i + WINDOW}", "stream": streams.pop() if len(streams) == 1 else "mixed",
               "cleared": int(len(cleared)), "missed": int(cleared.spike.sum())}  # fmt: skip
        if len(cleared) >= MIN_CLEARED:
            row["missed_among_cleared"] = float(cleared.spike.mean())
            row["outcome_alert"] = row["missed_among_cleared"] > cap
        else:
            row["missed_among_cleared"], row["outcome_alert"] = None, "too few cleared to judge"
        auc = roc_auc_score(scored.spike, scored.risk) if scored.spike.nunique() == 2 else float("nan")
        row["auroc"] = float(auc)
        row["auroc_below_locked_value"] = bool(auc < LOCKED_AUROC)
        row["calibration_in_the_large"] = float(scored.risk.mean() - scored.spike.mean())
        windows.append(row)
    summary = {}
    for s in ("dev", "mixed", "locked"):
        ws = [w for w in windows if w["stream"] == s]
        wk = [w for w in weeks if w["stream"] == s]
        summary[s] = {"windows": len(ws),
                      "outcome_alerts": sum(w["outcome_alert"] is True for w in ws),
                      "outcome_too_few_cleared": sum(w["outcome_alert"] == "too few cleared to judge" for w in ws),
                      "auroc_below_locked_value": sum(w["auroc_below_locked_value"] for w in ws),
                      "calibration_in_the_large_median": float(np.median([w["calibration_in_the_large"] for w in ws]))
                      if ws else None,
                      "weeks": len(wk), "behaviour_alerts": sum(w["behaviour_alert"] for w in wk),
                      "caution_share_median": float(np.median([w["caution"] for w in wk])) if wk else None}  # fmt: skip
    res = {"what": __doc__.splitlines()[0], "rules": "docs/monitoring.md, replay section (registered 26 Nov)",
           "block": BLOCK, "window_blocks": WINDOW, "min_cleared": MIN_CLEARED, "cap": cap,
           "behaviour_threshold_dev_max_caution": dev_max_caution,
           "summary": summary, "weeks": weeks, "windows": windows}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay.json").write_text(json.dumps(res, indent=2))
    d, lk = summary["dev"], summary["locked"]
    logger.bind(stage="monitoring").info(
        "replay: dev {} windows, {} outcome alerts, {} behaviour alerts in {} weeks; locked {} windows, {} outcome "
        "alerts, {} too few cleared, {} behaviour alerts in {} weeks (threshold {:.0%})", d["windows"],
        d["outcome_alerts"], d["behaviour_alerts"], d["weeks"], lk["windows"], lk["outcome_alerts"],
        lk["outcome_too_few_cleared"], lk["behaviour_alerts"], lk["weeks"], dev_max_caution)  # fmt: skip
    return res
