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


def at_deployed_cutoffs(dev: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Replay 2: each dev answer with a risk re-decided at the deployed cut-offs (no refit); answers without
    a risk stay can't judge."""
    from glimpse.serve.rules import decision

    out = dev.copy()
    has = out.risk.notna()
    out.loc[has, "decision"] = [decision(float(r), lo, hi) for r in out.loc[has, "risk"]]
    return out


def run_monitoring_replay(cfg) -> dict:
    """Replay 1 (dev at each fold's cut-offs) and replay 2 (dev at the deployed cut-offs), each followed by
    the locked answers; both are written to outputs/monitoring/replay.json."""
    from glimpse.training.final import load_thresholds

    cap = cfg.targets.usefulness.max_missed_among_cleared
    dev = pd.read_csv("outputs/evaluation_v2/dev_answers.csv")
    locked = pd.read_csv("outputs/locked_eval/locked_answers.csv")
    th = load_thresholds()["main"]
    res = replay(dev, locked, cap)
    res["deployed_operating_point"] = replay(at_deployed_cutoffs(dev, th["low_risk_below"], th["caution_from"]),
                                             locked, cap)  # fmt: skip
    res["deployed_operating_point"]["cutoffs"] = [th["low_risk_below"], th["caution_from"]]
    import joblib

    from glimpse.training.final import FINAL

    res["cusum_served"] = cusum_served(dev, locked, tuple(joblib.load(FINAL)["calibration"]),
                                       (th["low_risk_below"], th["caution_from"]))  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay.json").write_text(json.dumps(res, indent=2))
    for name, r in (("replay 1", res), ("replay 2 (deployed cut-offs)", res["deployed_operating_point"])):
        d, lk = r["summary"]["dev"], r["summary"]["locked"]
        logger.bind(stage="monitoring").info(
            "{}: dev {} windows, {} outcome alerts, {} behaviour alerts in {} weeks; locked {} windows, {} outcome "
            "alerts, {} too few cleared, {} behaviour alerts in {} weeks (threshold {:.0%})", name, d["windows"],
            d["outcome_alerts"], d["behaviour_alerts"], d["weeks"], lk["windows"], lk["outcome_alerts"],
            lk["outcome_too_few_cleared"], lk["behaviour_alerts"], lk["weeks"],
            r["behaviour_threshold_dev_max_caution"])  # fmt: skip
    return res


def cusum_served(dev: pd.DataFrame, locked: pd.DataFrame, platt: tuple[float, float],
                 cuts: tuple[float, float] | None = None, k: float = 0.5, h: float = 4.0) -> dict:  # fmt: skip
    """Replay 3 (registered 26 Nov): one-sided CUSUM on the served (Platt-only) risk's per-block
    calibration (spike rate minus mean risk) and on the caution share, standardised on the dev blocks.

    Correction (26 Nov, after review): the caution share registered here is taken from the logged decisions,
    whose dev part used each fold's own cut-offs, so a shift in it at the dev/locked seam can come from the
    cut-off change alone. Two added rows put both streams at the deployed cut-offs: the shipped risk
    (`caution_share_deployed_cutoffs`) and the served Platt-only risk (`caution_share_served`). They are
    descriptive, added after the first result was seen."""
    lk = locked.copy()
    q = np.clip(lk.risk_raw.to_numpy(float), 1e-6, 1 - 1e-6)
    lk["served"] = 1 / (1 + np.exp(-(platt[0] * np.log(q / (1 - q)) + platt[1])))
    lk.loc[lk.risk_raw.isna(), "served"] = np.nan
    dv = dev.assign(served=dev.risk_platt)
    if cuts is not None:
        from glimpse.serve.rules import decision

        dv["dec_deployed"] = at_deployed_cutoffs(dev, *cuts).decision.to_numpy()
        lk["dec_deployed"] = lk.decision
        for d in (dv, lk):
            has = d.served.notna()
            d["dec_served"] = d.decision
            d.loc[has, "dec_served"] = [decision(float(r), *cuts) for r in d.loc[has, "served"]]
    blocks = _blocks(dv, "dev") + _blocks(lk, "locked")
    first_locked = next(i + 1 for i, b in enumerate(blocks) if b.stream.iloc[0] == "locked")
    out = {}
    for name, fn in (("calibration", lambda b: float(b.spike[b.served.notna()].mean() - b.served.mean())),
                     ("caution_share", lambda b: float((b.decision == "caution").mean())),
                     ("caution_share_deployed_cutoffs", lambda b: float((b.dec_deployed == "caution").mean())),
                     ("caution_share_served", lambda b: float((b.dec_served == "caution").mean()))):  # fmt: skip
        if name != "calibration" and name != "caution_share" and cuts is None:
            continue
        x = np.array([fn(b) for b in blocks])
        ref = x[[b.stream.iloc[0] == "dev" for b in blocks]]
        z = (x - ref.mean()) / ref.std(ddof=1)
        s, alarms, path = 0.0, [], []
        for i, zi in enumerate(z):
            s = max(0.0, s + zi - k)
            path.append(s)
            if s > h:
                alarms.append({"week": i + 1, "stream": blocks[i].stream.iloc[0]})
                s = 0.0
        out[name] = {"x": x.tolist(), "cusum": path, "alarms": alarms,
                     "dev_false_alarms": sum(a["stream"] == "dev" for a in alarms),
                     "first_locked_alarm_week": next((a["week"] for a in alarms if a["stream"] == "locked"), None),
                     "first_locked_week": first_locked}  # fmt: skip
    return {"k": k, "h": h, **out}


def replay(dev: pd.DataFrame, locked: pd.DataFrame, cap: float) -> dict:
    """Weekly behaviour shares and rolling outcome signals over dev then locked answers."""
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
    return {"what": __doc__.splitlines()[0], "rules": "docs/monitoring.md, replay sections (registered 26 Nov)",
            "block": BLOCK, "window_blocks": WINDOW, "min_cleared": MIN_CLEARED, "cap": cap,
            "behaviour_threshold_dev_max_caution": dev_max_caution,
            "summary": summary, "weeks": weeks, "windows": windows}  # fmt: skip
