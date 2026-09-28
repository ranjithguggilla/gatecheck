from __future__ import annotations

import numpy as np
import pandas as pd

from gatecheck import targeting

from .conftest import make_players


def test_features_are_the_two_post_treatment_columns(players):
    features = targeting.build_features(players)
    assert list(features.columns) == ["retention_1", "log_gamerounds"]
    assert features["log_gamerounds"].min() >= 0.0


def test_leakage_report_shows_the_engagement_feature_adding_apparent_skill():
    frame = make_players(n_control=4_000, n_treatment=4_000, seed=31)
    # Tie day-7 retention to the engagement column so the leak is unmistakable.
    frame["retention_7"] = frame["sum_gamerounds"] > 200
    report = targeting.leakage_report(frame, seed=1, folds=3)
    assert report.auc_with_engagement > report.auc_day_one_only
    assert report.leak_contribution_auc > 0.1
    assert report.auc_majority_baseline == 0.5


def test_decile_spread_is_near_zero_when_treatment_does_nothing():
    generator = np.random.default_rng(41)
    size = 6_000
    features = np.column_stack([generator.random(size), generator.normal(size=size)])
    treatment = generator.integers(0, 2, size=size)
    outcome = (generator.random(size) < 0.2).astype("int64")
    spread = targeting.decile_uplift_spread(features, treatment, outcome, seed=2, folds=3)
    assert abs(spread) < 0.15


def test_placebo_p_value_never_reaches_zero():
    frame = make_players(n_control=1_200, n_treatment=1_200, seed=46)
    report = targeting.placebo_test(
        frame, "gate_30", "gate_40", "retention_7", seed=6, placebo_runs=3
    )
    assert report.permutation_p_value >= 1.0 / (1 + report.placebo_runs)


def test_placebo_test_reports_no_signal_on_random_assignment():
    frame = make_players(n_control=2_500, n_treatment=2_500, rate_treatment=0.20, seed=43)
    report = targeting.placebo_test(
        frame, "gate_30", "gate_40", "retention_7", seed=3, placebo_runs=4
    )
    assert report.permutation_p_value > 0.05
    assert "no usable targeting signal" in report.verdict
    assert report.placebo_runs == 4


def test_placebo_report_is_serialisable():
    frame = make_players(n_control=1_500, n_treatment=1_500, seed=44)
    report = targeting.placebo_test(
        frame, "gate_30", "gate_40", "retention_7", seed=4, placebo_runs=2
    )
    payload = report.as_dict()
    assert isinstance(payload, dict)
    assert set(payload) >= {"observed_spread_pp", "permutation_p_value", "verdict"}
    assert len(payload["spreads_pp"]) == report.placebo_runs


def test_single_class_arm_does_not_crash_the_t_learner():
    frame = make_players(n_control=400, n_treatment=400, seed=45)
    frame.loc[frame["version"] == "gate_40", "retention_7"] = False
    features = targeting.build_features(frame).to_numpy()
    treatment = (frame["version"] == "gate_40").astype("int64").to_numpy()
    outcome = frame["retention_7"].astype("int64").to_numpy()
    spread = targeting.decile_uplift_spread(features, treatment, outcome, seed=5, folds=2)
    assert np.isfinite(spread)


def test_absolute_percentile_is_never_below_the_largest_signed_spread():
    frame = make_players(n_control=1_500, n_treatment=1_500, seed=47)
    report = targeting.placebo_test(
        frame, "gate_30", "gate_40", "retention_7", seed=7, placebo_runs=4
    )
    assert report.placebo_max_abs_pp >= report.placebo_abs_p95_pp >= 0.0


def test_group_uplift_is_zero_without_both_arms():
    treatment = np.array([1, 1, 1])
    outcome = np.array([1, 0, 1])
    assert targeting._group_uplift(treatment, outcome, np.arange(3)) == 0.0


def test_build_features_handles_zero_rounds():
    frame = pd.DataFrame(
        {
            "userid": [1, 2],
            "version": ["gate_30", "gate_40"],
            "sum_gamerounds": [0, 0],
            "retention_1": [False, True],
            "retention_7": [False, False],
        }
    )
    features = targeting.build_features(frame)
    assert features["log_gamerounds"].tolist() == [0.0, 0.0]
