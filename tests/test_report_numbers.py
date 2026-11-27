"""The report's numbers against their evidence files (26 Nov). Runs on a clean checkout: it reads only
committed outputs, so a grader's CI catches any report number that drifts from the file it cites."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPORT = Path("docs/final_report.md")
APPENDIX = Path("docs/report_appendix.md")  # secondary tables moved out of the report (26 Nov); checked the same
ARCHIVE = Path("docs/archive/report_v1_prose.md")  # the pre-v2 prose, archived; its numbers stay checked
RM = Path("outputs/report_metrics/report_metrics.json")


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


@pytest.fixture(scope="module")
def text() -> str:
    return "\n".join(p.read_text() for p in (REPORT, APPENDIX, ARCHIVE) if p.exists())


@pytest.fixture(scope="module")
def rm() -> dict:
    return json.loads(RM.read_text())


def _f3(x: float) -> str:
    return f"{x:.3f}"


def test_word_cap_under_the_rubric_rule():
    """<= 3,000 words, tables, figures and their captions excluded (the `just wordcount` rule)."""
    kept = [ln for ln in REPORT.read_text().splitlines()
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
    assert f"{d['low risk']} low risk,\n{d['caution']} caution, {d["can't judge"]} can't judge" in text


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

    for label, key in (("profile only", "profile_lr"), ("trend + profile LR", "trend_lr"),
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


def test_status_table_timing_cusum_and_what_if(text, rm):
    t = _tables(text)
    lk = _load("outputs/locked_eval/locked_summary.json")
    assert t["AUROC, shipped (95% person-clustered)"][1].startswith(f"{lk['risk_personal']['auroc']['point']:.3f}")
    gb = rm["fair_baselines"]["trend_gbm"]
    assert t["same intercept on gradient boosting (sensors + profile)"][0].startswith(
        f"{gb['auroc_with_intercept']:.3f}"
    )
    st = _load("outputs/efficiency/stage_timing.json")
    row = t["whole photo path / no-photo path (sensors + profile only)"][0]
    assert row == f"{st['photo_path_median_ms']:.1f} / {st['no_photo_path_median_ms']:.2f}"
    assert t["backbone's share of the photo path"] == [f"{st['backbone_share_of_photo_path']:.0%}"]
    assert f"backbone is {st['backbone_share_of_photo_path']:.0%} of the photo path's time" in text
    cu = _load("outputs/monitoring/replay.json")["cusum_served"]
    assert cu["calibration"]["alarms"] == [] and cu["caution_share"]["dev_false_alarms"] == 0
    week = cu["caution_share"]["first_locked_alarm_week"] - cu["caution_share"]["first_locked_week"] + 1
    assert f"one alarm, {week}th locked week" in text
    assert cu["caution_share_deployed_cutoffs"]["alarms"] == [] and cu["caution_share_served"]["alarms"] == []
    np_ = rm["fair_baselines"]["no-photo fusion (the gate's no-meal arm)"]
    assert t["fusion without the photo (the gate's no-meal arm)"] == [
        f"{np_['auroc_with_intercept']:.3f}",
        f"+{np_['shipped_minus_this']:.3f} ({np_['ci95'][0]:+.3f}, {np_['ci95'][1]:+.3f})".replace("-", "−"),
    ]
    sh = rm["share_cis"]
    for row, key in (("cleared as low risk", "cleared"), ("can't judge", "cant_judge")):
        for j, coh in ((1, "dev"), (2, "locked")):
            c = sh[coh][key]
            want = f"{c['point'] * 100:.1f}% ({c['ci95'][0] * 100:.1f}–{c['ci95'][1] * 100:.1f}%)"
            assert t[row][j] == want, (row, coh)
    cs, fw = rm["cold_start_what_if"], rm["falling_now_what_if"]
    d0, d3, l0, l3 = cs["dev"]["0"], cs["dev"]["3"], cs["locked"]["0"], cs["locked"]["3"]

    def cell(c: dict) -> str:
        return f"{c['cleared']}, {c['missed']}, {_pct(c['cost_reduction'])}"

    assert t["as shipped"] == [cell(d0), cell(l0)]
    assert t["cold-start hold: no clear before the person's third labelled meal"] == [cell(d3), cell(l3)]
    fl = {"cleared": fw["cleared"] - fw["would_flip"], "missed": fw["misses_total"] - fw["misses_avoided"],
          "cost_reduction": fw["cost_reduction"]}  # fmt: skip
    assert t['"falling now": never clear a meal in the steepest-falling tenth of all meals'] == [cell(fl), "–"]
    sp = rm["served_imputed_priors"]
    row = t["no history and no onboarding priors (the samples' path; dev, approximate)"][2]
    assert row == (f"AUROC {sp['auroc']:.3f}; {sp['share_low_risk'] * 100:.1f}% cleared, {sp['n_missed']} of "
                   f"{sp['n_cleared']} spike; cost {_pct(sp['cost_reduction'])} / –; with rule R13: 0 cleared, "
                   f"{_pct(sp['with_rule_r13']['cost_reduction'])}")  # fmt: skip
    pc = _load("outputs/probes/probe_c.json")
    assert all(v["expected_direction_met"] for v in pc["inputs"].values())
    ip = pc["imputed_priors"]
    assert f"in-sample clears from {ip['cleared_own_priors']} to {ip['cleared_imputed']}" in text.replace("\n", " ")
    ef = rm["ece_noise_floor"]
    assert t["share of perfectly calibrated draws scoring at or above the observed ECE"] == [
        f"{ef['dev']['share_of_perfectly_calibrated_draws_at_or_above_observed']:.0%}",
        f"{ef['locked']['share_of_perfectly_calibrated_draws_at_or_above_observed']:.0%}",
        "–",
    ]


def test_subgroup_table(text, rm):
    t = _tables(text)
    for g in ("healthy", "prediabetes"):
        r = rm["subgroups"][g]
        lo, hi = r["auroc_ci95"]
        assert t[g] == [
            f"{r['people']}, {r['meals']}, {r['spike_rate'] * 100:.1f}%",
            f"{r['auroc']:.3f} ({lo:.3f}–{hi:.3f})",
            f"{r['mean_risk_minus_spike_rate']:+.3f}".replace("-", "−"),
            f"{r['cleared_share'] * 100:.1f}% ({r['cleared']}), {r['missed']}, "
            f"{r['people_with_a_clear']} of {r['people']}",
        ], g


def test_round8_rows(text, rm):
    t = _tables(text)
    ex = rm["usefulness_excluding_fold_1"]
    row = t[f"dev excluding fold 1 ({ex['meals']} meals): cleared, missed among them, cost saved"]
    assert (
        row[1] == f"{ex['cleared_share'] * 100:.1f}%, {ex['missed']} of {ex['cleared']}, {_pct(ex['cost_reduction'])}"
    )
    cu = _load("outputs/monitoring/replay.json")["cusum_served"]["caution_share_served"]
    assert cu["alarms"] == []
    assert t["the same on the served (no-history) decisions (added after that result)"][1] == (
        f"no alarm; peak {max(cu['cusum']):.1f} of 4"
    )
    pr = _load("outputs/promised/promised.json")
    r3 = pr["r3_label_sensitivity"]
    want = " / ".join(f"{r3[k]['auroc_personal']:.3f} ({r3[k]['spike_rate'] * 100:.0f}%)"
                      for k in ("spike_30", "spike_50", "spike_70"))  # fmt: skip
    assert t["label threshold 30 / 50 / 70 mg/dL: AUROC (spike rate)"] == [want]
    t5 = rm["t5_photo_and_heart_rate"]
    pb = [v["photo_block"] for v in t5["false_clears"].values() if v["class"].startswith("T5")]
    assert max(pb) < 0 and f"(−{-max(pb):.2f} to −{-min(pb):.2f}" in text
    ok = t5["cleared_without_spike"]
    assert f"equally on {ok['share_photo_block_negative']:.0%} of the cleared meals" in text
    assert f"(median −{-ok['median_photo_block']:.2f})" in text


def test_locked_decisions_follow_the_current_cut_off_rule():
    """The committed locked answers (scored once, 25 Nov) re-decided by today's decision function at the frozen
    cut-offs: identical, so the post-tag serving changes (R12, reason wording) did not alter any locked decision
    rule. Nothing is rescored; no metric is recomputed."""
    import pandas as pd

    from glimpse.serve.rules import decision
    from glimpse.training.final import load_thresholds

    th = load_thresholds()["main"]
    la = pd.read_csv("outputs/locked_eval/locked_answers.csv")
    has = la.risk.notna()
    redecided = [decision(float(r), th["low_risk_below"], th["caution_from"]) for r in la.risk[has]]
    assert redecided == la.decision[has].tolist()


def test_r13_cells(text, rm):
    sp = rm["served_imputed_priors"]["with_rule_r13"]
    assert sp["share_low_risk"] == 0
    assert f"{sp['share_abstain'] * 100:.1f}% can't judge" in text
    assert f"with rule R13: 0 cleared, {_pct(sp['cost_reduction'])}" in text


def test_within_person_rows(text, rm):
    t = _tables(text)
    wp = rm["within_person"]
    mw = wp["mean_within_person_auroc"]
    key = f"mean within-person AUROC ({wp['people_with_both_outcomes']} people with both outcomes), raw → shipped"
    d = wp["shipped_minus_raw_within_person"]
    assert t[key][0] == (
        f"{mw['risk_raw']:.3f} → {mw['risk']:.3f}; −{-d['point']:.3f} (−{-d['ci95'][0]:.3f}, "
        f"−{-d['ci95'][1]:.3f}), {d['people_worse']} of {wp['people_with_both_outcomes']} worse"
    )
    assert f"AUROC falls from {mw['risk_raw']:.3f} to {mw['risk']:.3f}" in text
    h = wp["history_only_running_rate"]
    row = t["no model: each person's running spike rate, same gate, shrunk to the other folds' rate"]
    assert row == [f"{h['auroc']:.3f}", f"+{h['shipped_minus_this']:.3f} (+{h['ci95'][0]:.3f}, +{h['ci95'][1]:.3f})"]
    assert f"does {h['shipped_minus_this']:.3f} better than a" in text


def test_cgm_removal_and_ood_rows(text, rm):
    t = _tables(text)
    c = rm["fair_baselines"]["cgm_removal"]
    assert t["CGM removal: the no-photo arm minus profile only (both with the intercept)"][1] == (
        f"+{c['no_photo_minus_profile_only']:.3f} (+{c['ci95'][0]:.3f}, +{c['ci95'][1]:.3f})"
    )
    mo = _load("outputs/monitoring/monitoring.json")
    n, med, above = mo["real_photos_n"], mo["real_photos_median"], mo["real_photos_above_cut"]
    assert t["real phone photos of local meals"] == [str(n), f"{med:.3f}", f"{above} of {n}"]
    assert t["training photos, leave-one-out (the budget)"][1] == f"{mo['training_loo_median']:.3f}"
    sc = {x["photo"]: x["score"] for x in _load("outputs/ood/ood_summary.json")["scores"] if x["set"] == "instructor"}
    assert t["instructor screenshot / dark photo / local dish"][1] == " / ".join(
        f"{sc[k]:.3f}" for k in ("09_screenshot", "08_dark_photo", "11_local_dish")
    )


def test_signal_2b_row(text):
    sm = _load("outputs/monitoring/replay.json")["summary"]
    row = _tables(text)["2b rolling AUROC below 0.645, 80-answer windows"]
    want = (
        f"{sm['dev']['auroc_below_locked_value']} of {sm['dev']['windows']} dev windows, "
        f"{sm['locked']['auroc_below_locked_value']} of {sm['locked']['windows']} locked"
    )
    assert row[1].startswith(want)


def _pc(x: float) -> str:
    return f"{x * 100:.1f}%".replace("-", "−")


def _iv(ci: list[float], pct: bool = False) -> str:
    if pct:
        return f"{ci[0] * 100:.1f}–{ci[1] * 100:.1f}%"
    return f"({ci[0]:+.3f}, {ci[1]:+.3f})".replace("-", "−")


def test_v2_cells_in_the_report():
    """v2 (docs/v2_sensor_first.md) cells against outputs/report_metrics/v2_metrics.json, report tables only."""
    t = _tables(REPORT.read_text())
    v2 = _load("outputs/report_metrics/v2_metrics.json")
    a = v2["auroc"]
    p460 = v2["paired_v2_minus_v1"]["auroc_v2"]
    assert t["AUROC (95% person-clustered)"][2] == (
        f"{a['point']:.3f} ({a['ci95'][0]:.3f}–{a['ci95'][1]:.3f}); {p460:.3f} on v1's 460"
    )
    assert t["meals answered with a risk"][2] == str(v2["answered_with_risk"])
    c = _load("outputs/v2_explain/v2_explain.json")["calibration"]
    lo, hi = c["observed_ci95_biased_upward"]
    assert abs(c["observed"] - v2["shipped"]["ece_equal_width_15"]) < 1e-12
    assert lo > c["observed"] and hi > 0.05 and c["verdict"] == "on track"  # the bootstrap is biased upward
    assert t["ECE, registered 15 bins (target ≤ 0.05)"][2] == (
        f"{c['observed']:.3f}, below a perfect model's median {c['perfectly_calibrated_median']:.3f}: {c['verdict']}"
    )
    assert c["share_of_perfectly_calibrated_draws_at_or_above_observed"] > 0.98  # "lower than 99% of draws"
    d = v2["decisions"]
    assert t["spikes among cleared meals (cap 10%)"][2] == (
        f"{_pc(d['missed_among_cleared'])} ({_iv(d['missed_ci']['ci95'], pct=True)})"
    )
    u = t["cost reduction vs always caution"]
    assert u[3] == f"{_pc(d['cost_reduction'])} ({d['cost_reduction_ci']['ci95'][0] * 100:+.1f}, " \
                   f"{d['cost_reduction_ci']['ci95'][1] * 100:+.1f})".replace("-", "−")  # fmt: skip
    assert t["cleared as low risk"][3] == f"{_pc(d['share_low_risk'])} ({_iv(d['cleared_ci']['ci95'], pct=True)})"
    assert t["can't judge"][3] == f"{_pc(d['share_abstain'])} ({_iv(d['cant_judge_ci']['ci95'], pct=True)})"
    miss = f"{_pc(d['missed_among_cleared'])} ({_iv(d['missed_ci']['ci95'], pct=True)})"
    assert t["missed among cleared"][3] == f"{miss}, {d['missed_n']} of {d['cleared_n']}"
    s = v2["served_no_history"]
    assert t["missed among cleared"][4] == f"{_pc(s['missed_among_cleared'])} ({s['n_missed']} of {s['n_cleared']})"
    p = v2["paired_v2_minus_v1"]
    row = t["v2 minus v1, AUROC on the 460 meals both answer (margin 0.03)"][0]
    assert (
        row.startswith(f"{p['v2_minus_v1']:+.3f} {_iv(p['ci95'])}".replace("-", "−")) and not p["noninferior_at_0.03"]
    )
    w = v2["within_person"]
    assert t["mean within-person AUROC, intercept minus raw: v1 / v2"][0].endswith(
        f"/ {w['shipped_minus_raw']:+.3f} {_iv(w['ci95'])}".replace("-", "−")
    )
    r13 = v2["samples_path"]["with_r13"]
    assert {**t, **_tables(Path("docs/report_appendix.md").read_text())}[
        "the samples' path (no history, no priors), with rule R13"
    ][2] == f"0 cleared; cost {_pc(r13['cost_reduction'])}"
    sp = v2["shipped"]
    assert t["v2 dev + personal intercept (alternative)"][1:4] == [
        f"{sp['auprc']['point']:.3f} ({sp['auprc']['ci95'][0]:.3f}–{sp['auprc']['ci95'][1]:.3f})",
        f"{sp['brier']['point']:.3f} ({sp['brier']['ci95'][0]:.3f}–{sp['brier']['ci95'][1]:.3f})",
        f"{sp['log_loss']:.3f}",
    ]
    ta = _tables(Path("docs/report_appendix.md").read_text())  # the subgroup table moved there (page limit)
    for g in ("healthy", "prediabetes"):
        r = v2["subgroups"][g]
        assert ta[f"{g}, both products"][3:] == [
            f"{r['auroc']:.3f}",
            f"{r['cleared_share'] * 100:.1f}% ({r['cleared']}), {r['missed']}",
        ], g


def test_photo_probe_and_int8_cells():
    text = REPORT.read_text() + "\n" + APPENDIX.read_text()  # probe E row moved there (page limit)
    pr, i8 = _load("outputs/probes/photo_probes.json"), _load("outputs/efficiency/int8.json")
    assert (f"more on {pr['dev_in_sample']['share_centre_moves_more']:.0%} of development photos (in-sample) and "
            f"{pr['local_unseen']['share_centre_moves_more']:.0%} of 14 unseen") in text  # fmt: skip
    assert pr["dev_in_sample"]["hypothesis_D_holds"]
    assert (f"grams rise on {pr['dev_in_sample']['zoom_share_grams_rise']:.0%} of development photos (in-sample); "
            f"{pr['local_unseen']['zoom_share_grams_rise']:.0%} of the 14 unseen, inconclusive") in text  # fmt: skip
    assert (f"carb MAE {i8['carb_mae_fp32_in_sample']:.1f} → {i8['carb_mae_int8_in_sample']:.1f} g") in text
    assert f"{i8['v1_decision_agreement']:.1%} of v1's decisions kept" in text
    assert f"{i8['median_ms_int8']:.0f} ms against {i8['median_ms_fp32']:.0f} ms per photo" in text


def test_monitoring_power_row():
    pw = _load("outputs/monitoring/power.json")
    c, s = pw["calibration"], pw["caution_share"]
    want = (f"calibration CUSUM: {c['detected_within_6_blocks']:.0%} detect within 120 answers, "
            f"{c['false_alarm_before_switch']:.0%} false alarms before the switch; caution share: "
            f"{s['detected_within_6_blocks']:.0%}, {s['false_alarm_before_switch']:.0%}")  # fmt: skip
    assert want in REPORT.read_text() + APPENDIX.read_text()  # the v2 row moved to the appendix


def test_stress_quality_cells():
    """docs/stress_quality.md study, against outputs/robustness/stress_quality.json."""
    t = _tables(REPORT.read_text() + "\n" + APPENDIX.read_text())  # v2-only stress rows moved there (page limit)
    sq = _load("outputs/robustness/stress_quality.json")
    v2, v1 = sq["v2_sensor_first"], sq["v1_photo_fusion"]
    lv = sq["levels"]
    row = t["CGM dropout 10 / 30 / 50 / 70%"]
    d2, d1 = v2["cgm_dropout"], v1["cgm_dropout"]
    assert row[0].startswith(
        " / ".join(f"{d2[str(x)]['auroc_perturbed']:.3f}" for x in lv["cgm_dropout"][:3])
        + f"; at 70%, {d2['0.7']['auroc_perturbed']:.3f} on the {d2['0.7']['co_answered']} still answered "
        f"(their clean {d2['0.7']['auroc_clean']:.3f})"
    )
    assert row[2] == (
        " / ".join(f"{d1[str(x)]['auroc_perturbed']:.3f}" for x in lv["cgm_dropout"][:3])
        + f"; at 70%, {d1['0.7']['auroc_perturbed']:.3f} on {d1['0.7']['co_answered']} "
        f"(clean {d1['0.7']['auroc_clean']:.3f})"
    )
    assert row[3] == " / ".join(str(v1["cgm_dropout"][str(x)]["caution_to_low_risk"]) for x in lv["cgm_dropout"])
    assert all(v2[k][str(x)]["caution_to_low_risk"] == 0 for k in ("cgm_dropout", "photo_blur", "photo_dark")
               for x in lv[k])  # fmt: skip
    for g in (20, 22, 25, 40):
        assert v2["cgm_gap"][str(g)]["answered_perturbed"] == 0 and v1["cgm_gap"][str(g)]["answered_perturbed"] == 0
    blur = t["photo blur radius 2 / 6 / 10"]
    assert blur[2].startswith(
        " / ".join(f"{v1['photo_blur'][str(x)]['auroc_perturbed']:.3f}" for x in lv["photo_blur"])
    )
    assert blur[3] == " / ".join(str(v1["photo_blur"][str(x)]["caution_to_low_risk"]) for x in lv["photo_blur"])
    vs = _load("outputs/robustness/v2_input_stress.json")
    worst = max(r["caution_to_low_risk"] for p in (v1, v2) for k in lv for r in p[k].values())
    worst_v2_inputs = max(vs[c]["caution_to_low_risk"] for c in vs if c not in ("what", "clean"))
    assert vs["hr_removed"]["caution_to_low_risk"] == worst_v2_inputs > worst
    assert f"{worst_v2_inputs} cautions\nturned low risk" in REPORT.read_text()
    hr = t["heart rate removed / half its readings dropped"]
    assert hr[0] == (
        f"{vs['hr_removed']['auroc_perturbed']:.3f} / {vs['hr_half']['auroc_perturbed']:.3f}; "
        f"{_pc(vs['hr_removed']['missed_among_cleared'])} / {_pc(vs['hr_half']['missed_among_cleared'])}"
    )
    fl = t["10% / 25% of history labels flipped"]
    a10, a25 = vs["history_labels_10pct_flipped"], vs["history_labels_25pct_flipped"]
    assert fl[0] == (
        f"{a10['auroc_perturbed']:.3f} / {a25['auroc_perturbed']:.3f}; "
        f"{_pc(a10['missed_among_cleared'])} / {_pc(a25['missed_among_cleared'])}"
    )
    pr = t["onboarding priors removed (history kept)"]
    q = vs["priors_removed"]
    assert pr[0] == f"{q['auroc_perturbed']:.3f}; {_pc(q['missed_among_cleared'])}"


def test_v2_explain_cells():
    """docs/v2_explain.md study, against outputs/v2_explain/v2_explain.json."""
    text = REPORT.read_text()
    t = _tables(text)
    r = _load("outputs/v2_explain/v2_explain.json")
    c = r["calibration"]
    floor = t["ECE a perfectly calibrated model shows at this size (median; 95th centile)"]
    assert floor[2] == f"{c['perfectly_calibrated_median']:.3f}; {c['perfectly_calibrated_q95']:.3f}"
    tax = r["taxonomy"]
    near = t["near-cut false clear (within 0.031 of the fold's cut)"][0]
    conf = t["confident false clear"][0]
    assert near.endswith(f"/ {tax['by_class']['T4 near-cut']} of 118")
    assert conf.endswith(f"/ {tax['by_class']['T5 confident']} of 118")
    assert tax["false_clears"] == 15 and tax["by_group"]["prediabetes"] == 6 and tax["frozen_band_refers"] == 4
    fb = r["frozen_band_in_sample"]
    assert (
        f"refers {fb['in_band']} meals ({fb['in_band_share']:.0%}, spike rate {fb['in_band_spike_rate']:.0%}), "
        f"clears {fb['share_low_risk']:.0%} with {fb['missed']} of {fb['cleared']} spiking"
    ) in " ".join(Path("docs/report_appendix.md").read_text().split())  # moved to the appendix (page limit)
    wi = r["prediabetes_whatif"]
    assert wi["20"]["prediabetes_missed"] == 2 and wi["20"]["prediabetes_cleared"] == 3
    hr = r["heart_rate"]
    assert (
        f"({hr['within_person_coefficient']:.2f}, {hr['within_person_ci95'][0]:.2f} to "
        f"{hr['within_person_ci95'][1]:.2f}, person fixed effects)"
    ).replace("-", "−") in " ".join(APPENDIX.read_text().split())
    assert f"moves v2's logit by {r['shap']['split']['mean_abs_intercept_shift']:.2f}" in APPENDIX.read_text()
    tp = r["temporal"]["last_third"]
    flat = " ".join(Path("docs/report_appendix.md").read_text().split())  # moved to the appendix (page limit)
    assert f"last third of meals v2 scores {tp['auroc']:.3f} ({tp['ci95'][0]:.3f}–{tp['ci95'][1]:.3f})" in flat
    f2 = r["temporal"]["first_two_thirds"]
    assert (
        f"rise from {_pc(f2['missed_among_cleared'])} (7 of {f2['cleared']}) to {_pc(tp['missed_among_cleared'])} (8 of"
        in flat
    )
    cv = r["carb_value"]
    assert t["the carb estimate's own value, 817 out-of-fold photos: MAE; carb range right"][0] == (
        f"{cv['mae_photo_g']:.1f} g against {cv['mae_median_g']:.1f} g for typing the training median; "
        f"{cv['range_accuracy_photo']:.0%} against {cv['range_accuracy_median']:.0%}"
    )


def test_v2_probes_monitors_and_budget_cells():
    """docs/v2_hardening.md Part B and the budget CSV against the report (26 Nov)."""
    import csv

    text = REPORT.read_text()
    t = _tables(text + "\n" + APPENDIX.read_text())  # the v2 monitoring rows moved there (page limit)
    vp = _load("outputs/probes/v2_probes.json")
    assert f"on only {vp['probe_b']['share_rising_raises_risk']:.1%} of meals in both products" in text
    pc = vp["probe_c"]
    lo, hi = sorted(pc["inputs"][n]["share_decision_differs"] for n in ("age", "prior spike rate", "prior rise"))[::2]
    assert set(pc["top_three"]) == {"age", "prior spike rate", "prior rise"}
    appendix = Path("docs/report_appendix.md").read_text()  # probes A and C moved there (page limit)
    assert f"age and the two priors move the most ({lo * 100:.0f}–{hi:.0%})" in appendix
    mo = vp["monitors"]
    row = t["v2 inputs (registered 26 Nov): share of answers without heart rate; Brier of the last 20 answers"]
    assert row[1].startswith(
        f"heart rate: no clean alarm, the watch-off block fires; Brier: {mo['clean_false_alarms_brier']} clean "
        f"alarms in {mo['clean_blocks']}, label errors caught in {mo['label_error_block_fire_rate']:.0%} of blocks; "
        "expectation not met"
    )
    assert mo["clean_false_alarms_hr"] == 0 and mo["watch_off_block_fires"]
    ch = _load("outputs/v2_explain/v2_explain.json")["carb_heads"]
    assert f"{ch['disagree_share']:.0%} of {ch['photos']} out-of-fold photos" in text
    rows = {r["model"]: int(r["parameters"]) for r in csv.DictReader(open("outputs/param_budget.csv"))}
    assert rows["v2 fusion, alternative (logistic regression: 15 profile/trend inputs)"] == 16
    assert rows["SensorCNN (1-D CNN)"] == 5217
    sc = "SensorCNN, the from-scratch network: it ties the trend features, so it is evaluated, not served"
    assert t[sc][:2] == ["5,217", "5,217"]


def test_round13_cells():
    """Within-person levels, the 481-meal baselines and the like-for-like photo gain (26 Nov, fourteenth pass)."""
    text = REPORT.read_text()
    t = _tables(text)
    rm, v2 = _load("outputs/report_metrics/report_metrics.json"), _load("outputs/report_metrics/v2_metrics.json")
    w1, w2 = rm["within_person"]["mean_within_person_auroc"], v2["within_person"]
    t = {**t, **_tables(Path("docs/report_appendix.md").read_text())}  # status rows moved there (page limit)
    assert t["mean within-person AUROC (a person's own meals)"] == [
        f"{w1['risk']:.2f} (raw {w1['risk_raw']:.2f})",
        "–",
        f"{w2['shipped']:.2f} (raw {w2['raw']:.2f})",
    ]
    b = rm["baselines"]["all_481"]
    keys = ("prevalence", "profile_lr", "trend_lr", "trend_gbm", "sensor_cnn_3seed_mean")
    assert t["the same baselines on v2's 481 meals"][0] == " / ".join(f"{b[k]['auroc']:.3f}" for k in keys)
    p = v2["paired_v2_minus_v1"]
    assert abs(p["v2_minus_v1"] - (p["auroc_v2"] - p["auroc_v1"])) < 1e-12  # the observed point, not the bootstrap mean
    f = rm["fair_baselines"]["no-photo fusion (the gate's no-meal arm)"]
    assert (
        f"the photo adds {f['shipped_minus_this']:.3f} ({f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f})".replace("-", "−")
    ) in " ".join(text.split())


def test_r16_counts_and_sweep():
    """Rule R16's registered predictions against outputs/probes/v2_hr_sweep.json."""
    r = _load("outputs/probes/v2_hr_sweep.json")
    assert r["outside"]["development"]["outside"] == 0
    lo, hi = r["heart_rate_range"]
    for s in r["sweep"]:
        assert (s["rule_on"] == "can't judge") == (s["hr_bpm"] > hi) or s["rule_off"] == "can't judge"
    r1 = _load("outputs/probes/v1_hr_sweep.json")
    assert r1["outside"]["development"]["outside"] == 0 and r1["outside"]["locked"]["outside"] == 0
    assert r["outside"]["locked"]["outside"] == 0
    assert "Their effect on scored decisions is in docs/rule_ledger.md" in " ".join(REPORT.read_text().split())
    first_clear = min(x["hr_bpm"] for x in r1["sweep"] if x["rule_off"] == "low risk")
    assert first_clear == 140 and all(x["rule_on"] != "low risk" for x in r1["sweep"])


def test_final_checks_and_d8_cells():
    """docs/final_switch.md checks (O1, O2, M1, C1, S1-S5) and D8 against their evidence files."""
    text = " ".join(REPORT.read_text().split())
    f = _load("outputs/final_checks/final_checks.json")
    s1 = f["s1_s2"]["s1_shuffled"]
    assert f"{s1['mean']:.3f} (range {s1['min']:.3f}–{s1['max']:.3f}), against {s1['real']:.3f} unshuffled" in text
    c = f["s1_s2"]["s2_learning_curve"]
    assert " / ".join(f"{c[k]['mean']:.3f}" for k in ("4", "8", "12", "all (about 18)")) in text
    s4 = f["s4"]
    assert f"{s4['photo_inputs_accuracy']:.1%} of meals against {s4['majority_rate']:.1%}" in text
    o1, o2 = f["o1"], f["o2"]
    assert f"AUROC {o1['auroc']:.3f}" in text and o1["real_photos_max"] < o1["cut"]
    assert f"AUROC {o2['auroc']:.2f}" in APPENDIX.read_text() and o2["auroc_single_input"]["HbA1c"] == 1.0
    m = f["m1"]["v1"]
    assert " / ".join(f"{m[k]['detected']:.0%}" for k in ("120", "240", "480", "960")) + " detected" in text
    c1 = f["c1"]
    assert f"({c1['v1']['share_null_at_or_below_observed']:.0%} of draws at or below)" in APPENDIX.read_text()
    assert f"all but {c1['v2']['share_null_at_or_below_observed']:.1%}" in APPENDIX.read_text()
    d8 = _load("outputs/ablations/d8_profile_excluded.json")
    p = _load("outputs/report_metrics/report_metrics.json")["d8_paired"]
    assert (
        f"fell to {d8['gain_photo_vs_no_meal']['point']:.3f}, against {d8['control_random_drop_gain']['point']:+.3f}"
    ).replace("-", "−") in text
    assert (
        f"difference {p['matched_removed_minus_random_drop']:.3f}, {p['ci95'][0]:.3f} to {p['ci95'][1]:+.3f}".replace(
            "-", "−"
        )
    ) in text


def test_robustness_v3_cells(text):
    """R1-R8 (docs/robustness_v3.md) against outputs/robustness_v3/robustness_v3.json."""
    r = _load("outputs/robustness_v3/robustness_v3.json")
    flat = " ".join(text.split())
    lr = r["r1_lr_range"]
    assert f"{lr['photo_heads']['suggested']:.1e}".replace("e-03", "e-3") in flat
    assert f"{lr['sensor_cnn']['suggested']:.1e}".replace("e-03", "e-3") in flat
    assert all(v["used_within_10x"] for v in lr.values())
    ob = r["r2_one_batch"]
    assert all(v["fits"] for v in ob.values())
    assert f"loss {ob['photo_heads']['start_loss']:.2f} → {ob['photo_heads']['end_loss']:.3f}" in flat
    fl = r["r3_label_audit"]["flags"]["boundary_rise"]
    assert f"{fl['worst_20']:.0%} have a rise within 5 mg/dL" in flat and f"against {fl['all_development']:.0%}" in flat
    c4 = r["r4_classical"]
    assert (
        f"ridge {c4['classical_colour_ridge']['mae_g']:.1f} g, against {c4['frozen_summaries_ridge_mae_g']:.1f} g"
        in flat
    )
    v3 = r["r5_v3"]
    assert {(c["wd"], c["dropout"], c["views"]) for c in v3["chosen_per_fold"]} == {(0.1, 0.5, 2)}
    assert f"gap is {v3['gap_v3_mean']:.2f} against v1's 0.35" in flat
    assert f"from {v3['carb_v1']['mae_g']:.1f} to {v3['carb_v3']['mae_g']:.1f} g" in flat
    d = v3["v3_minus_v1"]
    assert f"{d['point']:.3f} ({d['ci95'][0]:.3f}, {d['ci95'][1]:+.3f})".replace("-", "−") in flat
    g6 = r["r6_d8"]["photo_gain_v3_d8"]
    assert f"{g6['point']:+.3f} ({g6['ci95'][0]:.3f}, {g6['ci95'][1]:+.3f})".replace("-", "−") in flat
    s7 = r["r7_seeds"]
    assert f"seed spread ({s7['range']:.3f})" in flat and f"ordinary gain ({s7['photo_gain']:.3f};" in flat
    r8 = r["r8_reweighted"]
    assert f"from {r8['auroc_unweighted']:.3f} to {r8['auroc_reweighted']:.3f}" in flat


def test_rules_r22_and_holm_cells(text):
    """Rule R22's counted effect and the Holm check (26 Nov) against their committed outputs."""
    r = _load("outputs/rules_r22/rules_r22.json")
    flat = " ".join(text.split())
    a1, a2 = r["v1"]["after"], r["v2"]["after"]
    assert r["v1"]["before"]["cleared"] == 109 and r["v1"]["r22_clears_to_caution"] == 22
    assert f"{a1['cleared_share']:.1%}; {a1['missed_among_cleared']:.1%} ({a1['missed']} of {a1['cleared']})" in flat
    assert f"{a2['cleared_share']:.1%}; {a2['missed_among_cleared']:.1%}" in flat
    assert f"turns {r['v1']['r22_clears_to_caution']} of v1's" in flat or "turns 22 of v1's 109" in flat
    assert f"only {r['v1']['spiked_among_r22']} of which spiked" in flat
    assert f"{r['locked_counted_only']['r22_clears_to_caution']} of 10 clears become caution" in flat
    h = _load("outputs/report_metrics/holm.json")["contrasts"]
    survivors = {k for k, v in h.items() if v["survives_holm_0.05"]}
    assert survivors == {"v1 minus a running spike rate", "within-person AUROC, intercept minus raw (v1)"}


def test_external_and_post_freeze_cells():
    """27 Nov additions: the external check of the frozen v2 and the post-freeze checks, against their outputs."""
    flat = " ".join(REPORT.read_text().split())
    e = _load("outputs/external_bigideas/external_bigideas.json")
    v = e["v2_frozen"]
    cell = f"{v['auroc']:.3f} ({v['ci95'][0]:.3f}–{v['ci95'][1]:.3f})"
    assert flat.count(cell) >= 3 and e["people"] == 12 and e["meals_scored"] == 201
    assert f"| {cell} | – | {v['brier']:.3f} ({v['brier_constant']:.3f}) | – | {v['ece']:.3f} / – |" in flat
    d = e["decisions_at_frozen_cutoffs"]
    assert (
        f"ECE {v['ece']:.3f}; {d['cleared_share']:.1%} cleared, {d['missed_among_cleared']:.1%} of them spiked" in flat
    )
    assert f"({d['missed']} of {d['cleared']})" in flat and abs(v["mean_risk_minus_spike_rate"]) < 0.005 + 1e-3
    assert (
        f"within-person {v['within_person']['mean']:.3f}, against {e['v2_without_intercept']['within_person']['mean']:.3f}"  # noqa: E501
        in flat
    )
    p2 = _load("outputs/external_bigideas/external_part2.json")
    raw, sim = p2["raw_no_intercept"], p2["photo_sized_error_simulation"]
    assert (
        f"scores {raw['with_carbs']:.3f} on these meals, {raw['gain']:+.3f} ({raw['ci95'][0]:+.3f}, {raw['ci95'][1]:+.3f})"  # noqa: E501
        in flat
    )
    assert 0.05 < sim["mean_gain"] < raw["gain"] and p2["refit_matches_frozen_v2_max_abs_logit_difference"] < 1e-6
    assert (
        f"scores {raw['with_carbs']:.3f} on these meals, {raw['gain']:+.3f} ({raw['ci95'][0]:+.3f}, {raw['ci95'][1]:+.3f})"  # noqa: E501
        in flat
    )
    ck = _load("outputs/external_bigideas/external_checks.json")
    big = ck["carb_gain_meals_of_200_kcal_or_more"]
    assert f"the gain is {big['point']:+.3f} ({big['ci95'][0]:+.3f}, {big['ci95'][1]:+.3f})" in flat
    assert f"{ck['people_ranked_better_with_carbs']} of {ck['people_with_both_outcomes']} people rank better" in flat
    assert (
        f"ranks {ck['people_ranked_below_chance_by_frozen_v2']} of {ck['people_with_both_outcomes']} people's own"
        in flat
    )
    cg = _load("outputs/external_bigideas/external_part3.json")["cgmacros_out_of_fold"]["usefulness"]["cost_reduction"]
    assert f"saves {cg['point']:.1%} ({cg['ci95'][0]:+.1%} to {cg['ci95'][1]:+.1%})".replace("-", "−") in flat
    p3 = _load("outputs/external_bigideas/external_part3.json")["external"]["usefulness"]
    cr, cs = p3["cost_reduction"], p3["cleared_share"]
    assert flat.count(f"{cr['point']:.1%} ({cr['ci95'][0]:.1%}–{cr['ci95'][1]:.1%})".replace("%–", "–")) >= 2
    assert (
        f"{p3['missed']} of its {p3['cleared']} clears spiked" in flat and f"clears {cs['point']:.1%} of meals" in flat
    )
    lc, rr = e["logged_carbs_alone"], e["running_spike_rate"]
    assert (f"| {rr['auroc']:.3f} ({rr['ci95'][0]:.3f}–{rr['ci95'][1]:.3f}) / "
            f"{lc['auroc']:.3f} ({lc['ci95'][0]:.3f}–{lc['ci95'][1]:.3f}) / {raw['with_carbs']:.3f} |") in flat  # fmt: skip  # noqa: E501
    pb = _load("outputs/post_freeze/post_freeze.json")["probe_b_premise"]["spearman_slope30_vs_rise"]
    assert f"(Spearman {pb['point']:+.2f}, {pb['ci95'][0]:+.2f} to {pb['ci95'][1]:+.2f})".replace("-", "−") in flat
    o1 = _load("outputs/final_checks/final_checks.json")["o1"]
    assert f"flags {round(o1['flagged_out'] * o1['out_of_distribution'])} of the {o1['out_of_distribution']}" in flat
    assert f"and {round(o1['flagged_in'] * o1['in_distribution'])} of the {o1['in_distribution']} meal photos" in flat
    fw = _load("outputs/report_metrics/report_metrics.json")["falling_now_what_if"]
    assert (
        f"flips {fw['would_flip']} clears and avoids {fw['misses_avoided']} of the {fw['misses_total']} misses" in flat
    )
    v3 = _load("outputs/robustness_v3/robustness_v3.json")["r5_v3"]["carb_v3"]
    assert (
        f"range heads on frozen features reach {v3['bin_f1']:.2f} (v3, a regularised heads-only candidate, appendix)"
        in flat
    )
    p = _load("outputs/post_freeze/post_freeze.json")
    j, c, m, a, dl = p["jpeg_stress"], p["crop_ablation"], p["embedding_map"], p["photo_average"], p["decision_layer"]
    assert (
        f"quality 50: median shift {j['q50']['median_abs_shift_g']:.1f} g, {j['q50']['same_range_as_clean']:.0%} keep"
        in flat
    )
    assert (
        f"quality 20: {j['q20']['same_range_as_clean']:.1%}; quality 10: {j['q10']['same_range_as_clean']:.0%}" in flat
    )
    crop, nocrop, none = (c[k] for k in ("crop_80pct_flip_jitter", "no_crop_flip_jitter", "no_augmentation"))
    assert (
        f"carb MAE {crop['mae_g_mean']:.1f} g with crops keeping 80%, {nocrop['mae_g_mean']:.1f} g without the crop"
        in flat
    )
    assert (
        f"(seeds {nocrop['mae_g_range'][0]:.1f}–{nocrop['mae_g_range'][1]:.1f}), {none['mae_g_mean']:.1f} g with no augmentation"  # noqa: E501
        in flat
    )
    assert f"{m['share_same_carb_range']:.0%} share its carb range (chance {m['chance_same_carb_range']:.0%})" in flat
    assert f"{m['share_same_person']:.0%} its person (chance {m['chance_same_person']:.0%})" in flat
    assert (f"fine-tuned {a['fine_tuned']['mae_g']:.1f} g, ridge on frozen features {a['ridge_frozen']['mae_g']:.1f} g, "  # noqa: E501
            f"their average {a['average_log_scale']['mae_g']:.1f} g") in flat  # fmt: skip
    b, n = dl["wider_band"]["refer_within_0.05_of_cut"], dl["no_band"]
    assert f"refers {b['referred_share']:.0%} of meals and leaves {b['missed_among_cleared']:.1%}" in flat
    assert f"({b['missed_ci95'][0]:.1%}–{b['missed_ci95'][1]:.1%})".replace("%–", "–") in flat
    assert f"against {n['missed_among_cleared']:.1%} with no band (intervals overlap)" in flat
    assert f"clears fall from {n['cleared_share']:.1%} to {b['cleared_share']:.1%} of the {dl['meals']} meals" in flat
    eff = _load("outputs/efficiency/efficiency.json")["latency_s"]
    assert f"p95 {eff['warm_p95'] * 1000:.0f}, first answer {eff['cold_first_answer']:.2f} s" in flat
