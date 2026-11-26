"""Monitoring replay 4 (`glimpse monitoring-power`; registered 26 Nov in docs/monitoring.md before its run):
how often the two CUSUM rules catch a realistic cohort shift in simulated v2 answer streams, how often they
raise a false alarm before it, and how fast. Data-free: committed v2 development answers only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ANSWERS = Path("outputs/evaluation_sensor_first/dev_answers.csv")
OUT = Path("outputs/monitoring/power.json")
BLOCK, PRE, BURN, WITHIN, K, H, STREAMS = 20, 12, 8, 6, 0.5, 4.0, 500


def _stream(df: pd.DataFrame, people: np.ndarray, rng) -> pd.DataFrame:
    """People in random order, each person's meals in time (row) order, concatenated."""
    order = rng.permutation(people)
    return pd.concat([df[df.person_id == p].sort_values("row") for p in order], ignore_index=True)


def _blocks(s: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    return [s.iloc[i * BLOCK : (i + 1) * BLOCK] for i in range(n)]


def _cusum(x: np.ndarray) -> list[int]:
    """Alarm block indices of a one-sided upper CUSUM standardised on the first BURN blocks."""
    ref = x[:BURN]
    sd = ref.std(ddof=1) or 1.0
    s, alarms = 0.0, []
    for i, v in enumerate((x - ref.mean()) / sd):
        if i < BURN:
            continue
        s = max(0.0, s + v - K)
        if s > H:
            alarms.append(i)
            s = 0.0
    return alarms


def run_monitoring_power(cfg) -> dict:
    df = pd.read_csv(ANSWERS)
    group = pd.read_csv("outputs/cv/oof_predictions.csv").groupby("person_id").group.first()
    df["group"] = df.person_id.map(group)
    people = df.person_id.unique()
    pre_d = df[df.group == "prediabetes"]
    rng = np.random.default_rng(cfg.seed)
    stats = {"calibration": [], "caution_share": []}
    post_blocks = PRE + WITHIN + 6
    for _ in range(STREAMS):
        before = _blocks(_stream(df, people, rng), PRE)
        after = []
        while len(after) < post_blocks - PRE:  # repeat prediabetic people in new random orders as needed
            after += _blocks(_stream(pre_d, pre_d.person_id.unique(), rng), len(pre_d) // BLOCK)
        blocks = before + after[: post_blocks - PRE]
        for name, fn in (("calibration", lambda b: b.spike[b.risk.notna()].mean() - b.risk.mean()),
                         ("caution_share", lambda b: (b.decision == "caution").mean())):  # fmt: skip
            alarms = _cusum(np.array([fn(b) for b in blocks], dtype=float))
            false = any(a < PRE for a in alarms)
            post = [a for a in alarms if a >= PRE]
            stats[name].append({"false_before": false, "delay": (post[0] - PRE + 1) if post else None})
    res = {"what": __doc__.splitlines()[0], "streams": STREAMS, "block": BLOCK, "switch_after_blocks": PRE,
           "burn_in_blocks": BURN, "k": K, "h": H,
           "spike_rate_all": float(df.spike.mean()), "spike_rate_after": float(pre_d.spike.mean())}  # fmt: skip
    for name, rows in stats.items():
        d = [r["delay"] for r in rows if r["delay"] is not None]
        res[name] = {"false_alarm_before_switch": float(np.mean([r["false_before"] for r in rows])),
                     "detected_within_6_blocks": float(np.mean([r["delay"] is not None and r["delay"] <= WITHIN
                                                                for r in rows])),
                     "detected_by_end": float(np.mean([r["delay"] is not None for r in rows])),
                     "median_delay_blocks": float(np.median(d)) if d else None}  # fmt: skip
    OUT.write_text(json.dumps(res, indent=2))
    for name in stats:
        r = res[name]
        logger.bind(stage="monitoring-power").info("{}: false alarm {:.0%}, detected within 6 blocks {:.0%}, "
                                                    "median delay {}", name, r["false_alarm_before_switch"],
                                                    r["detected_within_6_blocks"], r["median_delay_blocks"])  # fmt: skip
    return res
