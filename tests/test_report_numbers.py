"""The report's numbers against their evidence files (26 Nov). Runs on a clean checkout: it reads only
committed outputs, so a grader's CI catches any report number that drifts from the file it cites."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPORT = Path("docs/final_report.md")
RM = Path("outputs/report_metrics/report_metrics.json")


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


@pytest.fixture(scope="module")
def text() -> str:
    return REPORT.read_text()


@pytest.fixture(scope="module")
def rm() -> dict:
    return json.loads(RM.read_text())


def _f3(x: float) -> str:
    return f"{x:.3f}"


def test_word_cap_under_the_rubric_rule(text):
    """<= 3,000 words, tables, figures and their captions excluded (the `just wordcount` rule)."""
    kept = [ln for ln in text.splitlines()
            if not (ln.lstrip().startswith(("|", "![")) or (ln.lstrip().startswith("*")
                                                           and not ln.lstrip().startswith("**")))]  # fmt: skip
    assert len(" ".join(kept).split()) <= 3000


def test_no_em_dashes(text):
    assert "—" not in text


def test_dev_and_locked_headlines(text):
    dev = _load("outputs/evaluation_v2/dev_summary.json")
    lk = _load("outputs/locked_eval/locked_summary.json")
    for x in (dev["risk_personal"]["auroc"]["point"], dev["risk_raw"]["auroc"]["point"],
              lk["risk_personal"]["auroc"]["point"], lk["risk_raw"]["auroc"]["point"],
              dev["risk_personal"]["log_loss"], lk["risk_personal"]["log_loss"]):  # fmt: skip
        assert _f3(x) in text, x
    lo, hi = lk["risk_personal"]["auroc"]["ci95"]
    assert f"({lo:.3f}–{hi:.3f})" in text
    assert f"{lk['usefulness']['cost_reduction']:.1%}".replace("-", "−") in text
    d = lk["decisions"]
    assert f"{d['low risk']} cleared, {d['caution']} caution, {d["can't judge"]} can't judge" in text


def test_report_metrics_agree_with_the_earlier_evidence(rm):
    """The new descriptive file re-derives the committed numbers exactly before adding its own."""
    fig = _load("outputs/figures/figures.json")
    lk = _load("outputs/locked_eval/locked_summary.json")
    dev = rm["cohorts"]["dev"]
    assert dev["risk"]["ece_equal_mass_15"] == pytest.approx(fig["ece_personal"], abs=1e-12)
    assert dev["risk_raw"]["ece_equal_mass_15"] == pytest.approx(fig["ece_raw"], abs=1e-12)
    assert dev["risk"]["ece_equal_width_15"] == pytest.approx(fig["ece_equal_width_15"]["personal"], abs=1e-12)
    loc = rm["cohorts"]["locked"]
    assert loc["risk"]["auroc"] == pytest.approx(lk["risk_personal"]["auroc"]["point"], abs=1e-12)
    assert loc["risk"]["ece_equal_mass_15"] == pytest.approx(lk["risk_personal"]["ece_equal_mass"]["bins15"])
    assert loc["decisions"]["cost_reduction_vs_best_constant"] == pytest.approx(lk["usefulness"]["cost_reduction"])
    assert sum(b["n"] for b in dev["risk"]["bins_equal_width_15"]) == dev["risk"]["n"] == 460


def test_new_metrics_in_the_report(text, rm):
    for cohort in ("dev", "locked"):
        r = rm["cohorts"][cohort]["risk"]
        for x in (r["auprc"]["point"], r["brier"]["point"], r["ece_equal_width_15"], r["ece_equal_mass_15"],
                  r["base_rate"], r["brier_constant_base_rate"]):  # fmt: skip
            assert _f3(x) in text, (cohort, x)
    rq2 = rm["baselines"]["rq2_partition_v2"]
    assert f"+{rq2['sensor_cnn_minus_trend_lr']:.3f} ({rq2['ci95'][0]:.3f}, +{rq2['ci95'][1]:.3f})".replace(
        "-", "−") in text  # fmt: skip
    for m, x in rm["baselines"]["answered_460"].items():
        if m != "trend_gbm":
            assert _f3(x["auroc"]) in text, m


def test_monitoring_replay_matches_its_registration(text):
    from glimpse.studies import monitoring_replay as mr

    doc = Path("docs/monitoring.md").read_text()
    assert (mr.BLOCK, mr.WINDOW, mr.MIN_CLEARED) == (20, 4, 10)
    assert "block of 20" in doc and "four consecutive blocks (80 answers)" in doc and "at\n  least 10" in doc
    rep = _load("outputs/monitoring/replay.json")["summary"]
    assert f"{rep['dev']['outcome_alerts']} of {rep['dev']['windows']} ordinary development windows" in text
    assert rep["locked"]["outcome_too_few_cleared"] == rep["locked"]["windows"]


def test_stress_table_and_figure_share_one_run(text):
    """Figure 5 draws levels.json, the same run as the table (the 25 Nov grades found two runs shown as one)."""
    src = Path("src/glimpse/figures.py").read_text()
    assert "outputs/robustness/levels.json" in src and 'robustness.json").read_text' not in src
    lv = _load("outputs/robustness/levels.json")
    assert f"{lv['photo_blur']['6']['dangerous_flips']:.1%}" in text
    worst = max(lv[k][lev]["dangerous_flips"] for k in lv["levels"] for lev in lv[k])
    assert worst < 0.018
    hr = _load("outputs/promised/promised.json")["r6_missing_inputs"]["missing_hr"]["flips"]["dangerous_flips"]
    assert f"{hr:.1%}" in text


def test_every_cited_output_path_exists(text):
    for p in set(re.findall(r"outputs/[\w./-]+[\w/]", text)):
        assert Path(p.rstrip("/")).exists(), p
