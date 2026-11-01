"""Evaluation: metrics over predictions and decisions. The decision-MAKING functions (decide,
threshold selection) live in glimpse.training.thresholds -- selection is a training-time activity;
this package only measures. Report: §3 evaluation and ablations."""

from glimpse.evaluation.metrics import (
    apply_platt,
    calibration_by_person,
    choose_thresholds,
    cost_only_thresholds,
    coverage_report,
    decide,
    ece_equal_mass,
    expected_cost,
    fit_platt,
    load_locked_set,
    noninferiority,
    paired_person_bootstrap,
    participant_weighted_cqr_quantile,
    person_bootstrap_ci,
    person_folds,
    reliability,
    usefulness_report,
)

__all__ = [
    "apply_platt",
    "calibration_by_person",
    "choose_thresholds",
    "cost_only_thresholds",
    "coverage_report",
    "decide",
    "ece_equal_mass",
    "expected_cost",
    "fit_platt",
    "load_locked_set",
    "noninferiority",
    "paired_person_bootstrap",
    "participant_weighted_cqr_quantile",
    "person_bootstrap_ci",
    "person_folds",
    "reliability",
    "usefulness_report",
]
