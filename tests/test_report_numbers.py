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
    assert f"{rep['dev']['outcome_alerts']} of {rep['dev']['windows']} ordinary dev windows" in text
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


def _tables(text: str) -> dict[str, list[str]]:
    """Every markdown table row in the report, keyed by its first cell (bold stripped)."""
    rows = {}
    for ln in text.splitlines():
        if ln.startswith("|") and not ln.startswith("|---"):
            cells = [c.strip().replace("**", "") for c in ln.strip().strip("|").split("|")]
            rows[cells[0]] = cells[1:]
    return rows


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace("-", "−")


def test_results_table_cells(text, rm):
    t = _tables(text)
    dev, lk = rm["cohorts"]["dev"]["risk"], rm["cohorts"]["locked"]["risk"]
    row = t["dev + personal intercept (shipped)"]
    a, b = dev["auprc"], dev["brier"]
    assert row[1] == f"{a['point']:.3f} ({a['ci95'][0]:.3f}–{a['ci95'][1]:.3f})"
    assert row[2] == f"{b['point']:.3f} ({b['ci95'][0]:.3f}–{b['ci95'][1]:.3f})"
    assert row[4] == f"{dev['ece_equal_width_15']:.3f} / {dev['ece_equal_mass_15']:.3f}"
    row = t["locked shipped (150 of 153)"]
    assert row[0].startswith(f"{lk['auroc']:.3f}")
    assert row[4] == f"{lk['ece_equal_width_15']:.3f} / {lk['ece_equal_mass_15']:.3f}"
    pg = rm["paired_personal_minus_raw"]
    d = pg["dev"]["auroc"]
    assert t["AUROC"][0].startswith(f"+{d['point']:.3f} ({d['ci95'][0]:.3f}, +{d['ci95'][1]:.3f})".replace("-", "−"))
    assert f"{pg['dev']['folds_improved']} of 5 folds" in t["AUROC"][0]
    ll = pg["dev"]["log_loss"]
    assert t["log-loss"][0] == f"{ll['point']:.3f} ({ll['ci95'][0]:.3f}, {ll['ci95'][1]:.3f})".replace("-", "−")


def test_cost_grid_and_usefulness_cells(text, rm):
    t = _tables(text)
    sweep = {
        (r["missed_spike"], r["abstain_multiplier"]): r
        for r in _load("outputs/evaluation_v2/dev_summary.json")["cost_sweep"]
    }
    for label, ms in (("3×", 3.0), ("5× (declared)", 5.0), ("7×", 7.0), ("10×", 10.0)):
        for cell, am in zip(t[label], (0.5, 1.0, 2.0, 4.0), strict=True):
            r = sweep[(ms, am)]
            assert (
                cell == f"{_pct(r['cost_reduction'])} · {_pct(r['share_low_risk'])} · {_pct(r['missed_among_cleared'])}"
            )
    dec = {k: rm["cohorts"][k]["decisions"] for k in ("dev", "locked")}
    row = t["cost reduction vs the best constant policy (always caution, both cohorts)"]
    ci = rm["band_and_served_calibration"]["cost_reduction_ci95"]

    def withci(x, k):
        lo, hi = ci[k]
        return f"{_pct(x)} ({lo * 100:+.1f}, {hi * 100:+.1f})".replace("-", "−")

    assert row[1:3] == [withci(dec[k]["cost_reduction_vs_best_constant"], k) for k in ("dev", "locked")]
    row = t["mean cost per meal: shipped / always caution / always low risk"]
    for cell, k in zip(row[1:3], ("dev", "locked"), strict=True):
        assert cell == " / ".join(f"{dec[k][p]:.3f}" for p in ("shipped", "always caution", "always low risk"))
    er = rm["errors"]
    assert t["people with a cleared meal"][1] == f"{er['dev']['people_with_a_clear']} of {er['dev']['people']}"
    d, lk = er["dev"], er["locked"]
    want = f"{d['false_cautions']} of {d['cautions']}; {lk['false_cautions']} of {lk['cautions']}"
    assert t["false caution (caution, no spike)"][0] == want


def test_stress_table_cells(text):
    t = _tables(text)
    lv = _load("outputs/robustness/levels.json")

    def u(x):
        return f"{x * 100:.1f}%" if 0.99 < x < 1 else f"{x * 100:.0f}%"

    for label, kind, second in (
        ("CGM dropout 10/30/50% (same decision / low risk↔caution)", "cgm_dropout", "dangerous_flips"),
        ("photo blur r=2/6/10 (same decision / low risk↔caution)", "photo_blur", "dangerous_flips"),
    ):
        for cell, level in zip(t[label], lv["levels"][kind], strict=True):
            c = lv[kind][f"{level:g}"]
            assert cell == f"{u(c['unchanged'])} / {c[second] * 100:.1f}%", (label, cell)


def test_parameter_table_cells(text, rm):
    t = _tables(text)
    ft = rm["fine_tuned_parameters"]["fine_tuned_total"]
    assert t["fine-tuned, deployed (last ConvNeXt stage + heads)"][0] == f"{ft / 1e6:.1f}M"


def test_served_point_and_calibration_cells(text, rm):
    t = _tables(text)
    so = rm["served_operating_point"]
    row = t["cost reduction vs the best constant policy (always caution, both cohorts)"]
    ci = rm["band_and_served_calibration"]["cost_reduction_ci95"]
    cells = [f"{_pct(so[k]['cost_reduction'])} ({ci['served_' + k][0] * 100:+.1f}, {ci['served_' + k][1] * 100:+.1f})"
             for k in ("dev", "locked")]  # fmt: skip
    assert row[3] == " / ".join(cells).replace("-", "−")
    row = t["missed among cleared"]
    d, lk = so["dev"], so["locked"]
    assert row[3] == (f"{_pct(d['missed_among_cleared'])} ({d['n_missed']} of {d['n_cleared']}) / "
                      f"{_pct(lk['missed_among_cleared'])} ({lk['n_missed']} of {lk['n_cleared']})")  # fmt: skip
    fl = rm["ece_noise_floor"]
    row = t["ECE a perfectly calibrated model shows at this size: median, 95th percentile"]
    assert row[:2] == [f"{fl[k]['perfectly_calibrated_median']:.3f}, {fl[k]['perfectly_calibrated_q95']:.3f}"
                       for k in ("dev", "locked")]  # fmt: skip
    assert t["ECE, registered 15 equal-width bins (target ≤ 0.05)"][0] == f"{fl['dev']['observed']:.3f}"
    cil = rm["calibration_in_the_large_pooled"]
    assert t["calibration-in-the-large (mean risk − spike rate)"][:2] == [f"{cil[k]:.3f}".replace("-", "−")
                                                                         for k in ("dev", "locked")]  # fmt: skip
    po = rm["person_offsets"]

    def s(x):
        return f"{x:+.2f}".replace("-", "−")

    assert t["per-person offset b, logit: median (10th, 90th percentile)"][:2] == [
        f"{s(po[k]['median'])} ({s(po[k]['q10'])}, {s(po[k]['q90'])})" for k in ("dev", "locked")]  # fmt: skip


def test_monitoring_and_coverage_claims(text, rm):
    rep = _load("outputs/monitoring/replay.json")
    t = _tables(text)
    assert (
        t["2a, deployed cut-offs on the evaluated (with-history) risk"][1]
        == f"{rep['deployed_operating_point']['summary']['dev']['outcome_alerts']} of 21 dev windows"
    )
    assert rm["heart_rate_r2_from_other_inputs"] == pytest.approx(0.19, abs=0.005)
    cr = {r["abstain_share"]: r for r in rm["coverage_risk"]["dev"]}
    miss = [cr[a]["missed_among_cleared"] for a in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)]
    assert f"{min(miss) * 100:.1f}–{max(miss) * 100:.1f}%" in text
    assert f"from {cr[0.0]['cleared_share'] * 100:.0f}% to {cr[0.3]['cleared_share'] * 100:.0f}%" in text


def test_fair_baseline_and_band_cells(text, rm):
    t = _tables(text)
    fb = rm["fair_baselines"]
    assert fb["reproduces_shipped_max_abs_diff"] < 1e-12

    def d(x):
        return f"{x:+.3f}".replace("-", "−")

    for label, key in (("profile only", "profile_lr"), ("trend LR", "trend_lr"),
                       ("gradient boosting on trend + profile", "trend_gbm"),
                       ("SensorCNN, 3 seeds averaged", "sensor_cnn_3seed_mean")):  # fmt: skip
        r = fb[key]
        assert t[label] == [f"{r['auroc_with_intercept']:.3f}",
                            f"{d(r['shipped_minus_this'])} ({d(r['ci95'][0])}, {d(r['ci95'][1])})"], label  # fmt: skip
    b = rm["band_and_served_calibration"]["band"]
    assert f"{b['cost_without_band']:.3f} against {b['cost_shipped']:.3f}" in text
    assert f"spiked at {b['borderline_spike_rate'] * 100:.0f}%" in text
    sc = rm["band_and_served_calibration"]["served_calibration"]
    row = t["ECE, registered 15 equal-width bins (target ≤ 0.05)"]
    assert row[2] == f"{sc['dev']['ece_equal_width_15']:.3f} / {sc['locked']['ece_equal_width_15']:.3f}"
