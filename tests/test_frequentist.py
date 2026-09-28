from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.proportion import proportions_ztest

from gatecheck import frequentist


def test_z_statistic_matches_statsmodels():
    result = frequentist.two_proportion_test("retention_7", 8_502, 44_700, 8_279, 45_489)
    reference_z, reference_p = proportions_ztest(
        count=np.array([8_279, 8_502]), nobs=np.array([45_489, 44_700])
    )
    assert result.z_statistic == pytest.approx(reference_z, rel=1e-9)
    assert result.p_value == pytest.approx(reference_p, rel=1e-9)


def test_chi_square_agrees_with_the_z_test_on_direction():
    result = frequentist.two_proportion_test("retention_7", 8_502, 44_700, 8_279, 45_489)
    assert result.absolute_diff_pp < 0
    assert result.chi_square_p_value == pytest.approx(result.p_value, rel=0.1)


def test_no_difference_gives_a_p_value_of_one():
    result = frequentist.two_proportion_test("flat", 2_000, 10_000, 2_000, 10_000)
    assert result.z_statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.diff_ci_low_pp < 0 < result.diff_ci_high_pp


def test_interval_brackets_the_point_estimate():
    result = frequentist.two_proportion_test("retention_7", 8_502, 44_700, 8_279, 45_489)
    assert result.diff_ci_low_pp < result.absolute_diff_pp < result.diff_ci_high_pp
    assert result.relative_ci_low_pct < result.relative_lift_pct < result.relative_ci_high_pct


def test_newcombe_interval_stays_inside_the_unit_range():
    low, high = frequentist.newcombe_interval(1, 20, 19, 20)
    assert -1.0 <= low < high <= 1.0


def test_relative_interval_is_undefined_without_successes():
    low, high = frequentist.log_risk_ratio_interval(0, 100, 5, 100)
    assert np.isnan(low) and np.isnan(high)


def test_successes_above_arm_size_are_rejected():
    with pytest.raises(ValueError, match="must not exceed"):
        frequentist.two_proportion_test("bad", 20, 10, 5, 10)


def test_empty_arm_is_rejected():
    with pytest.raises(ValueError, match="at least one unit"):
        frequentist.two_proportion_test("bad", 0, 0, 5, 10)


def test_continuous_comparison_reports_both_tests():
    generator = np.random.default_rng(3)
    control = generator.gamma(2.0, 30.0, size=5_000)
    treatment = generator.gamma(2.0, 30.0, size=5_000)
    comparison = frequentist.compare_continuous(
        "sum_gamerounds", control, treatment, float(np.quantile(control, 0.999))
    )
    assert comparison.welch_p_value > 0.01
    assert comparison.mann_whitney_p_value > 0.01
    assert -1.0 <= comparison.rank_biserial <= 1.0


def test_continuous_comparison_detects_a_real_shift():
    generator = np.random.default_rng(4)
    control = generator.gamma(2.0, 30.0, size=5_000)
    treatment = generator.gamma(2.0, 45.0, size=5_000)
    comparison = frequentist.compare_continuous(
        "sum_gamerounds", control, treatment, float(np.quantile(treatment, 0.999))
    )
    assert comparison.welch_p_value < 1e-6
    assert comparison.mann_whitney_p_value < 1e-6
    assert comparison.cohens_d > 0


def test_multiplicity_adjustment_never_lowers_a_p_value():
    adjusted = frequentist.adjust_p_values(
        ["a", "b", "c"], [0.0016, 0.0744, 0.0502]
    )
    for entry in adjusted.values():
        assert entry["holm_p_value"] >= entry["raw_p_value"]
        assert entry["benjamini_hochberg_p_value"] >= entry["raw_p_value"]
    assert adjusted["a"]["holm_significant"]
    assert not adjusted["b"]["holm_significant"]


def test_a_metric_nobody_hit_is_handled_rather_than_crashing():
    result = frequentist.two_proportion_test("never", 0, 5_000, 0, 5_000)
    assert result.chi_square == 0.0
    assert result.chi_square_p_value == 1.0
    assert result.p_value == pytest.approx(1.0)


def test_a_metric_everybody_hit_is_handled_too():
    result = frequentist.two_proportion_test("always", 5_000, 5_000, 5_000, 5_000)
    assert result.chi_square == 0.0
    assert result.absolute_diff_pp == pytest.approx(0.0)
