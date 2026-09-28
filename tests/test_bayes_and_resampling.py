from __future__ import annotations

import numpy as np
import pytest

from gatecheck import bayes, resampling


def test_posterior_is_undecided_when_the_arms_match():
    comparison = bayes.posterior_comparison(
        "flat", 2_000, 10_000, 2_000, 10_000, draws=60_000, floor_pp=-0.2, seed=1
    )
    assert comparison.prob_treatment_better == pytest.approx(0.5, abs=0.02)
    assert comparison.hdi_low_pp < 0 < comparison.hdi_high_pp


def test_posterior_is_decisive_on_the_real_counts():
    comparison = bayes.posterior_comparison(
        "retention_7", 8_502, 44_700, 8_279, 45_489, draws=120_000, floor_pp=-0.2, seed=2
    )
    assert comparison.prob_treatment_better < 0.01
    assert comparison.hdi_high_pp < 0
    assert comparison.prob_worse_than_floor > 0.9


def test_expected_losses_are_non_negative_and_opposed():
    comparison = bayes.posterior_comparison(
        "retention_7", 8_502, 44_700, 8_279, 45_489, draws=40_000, floor_pp=-0.2, seed=3
    )
    assert comparison.expected_loss_if_ship_pp >= 0
    assert comparison.expected_loss_if_hold_pp >= 0
    assert comparison.expected_loss_if_ship_pp > comparison.expected_loss_if_hold_pp


def test_posterior_mean_sits_between_the_prior_and_the_data():
    comparison = bayes.posterior_comparison(
        "m", 10, 100, 90, 100, draws=20_000, floor_pp=-0.2, seed=4
    )
    assert 0.10 < comparison.posterior_mean_control < 0.12
    assert 0.88 < comparison.posterior_mean_treatment < 0.90


def test_posterior_rejects_zero_draws():
    with pytest.raises(ValueError, match="draws must be positive"):
        bayes.posterior_comparison("m", 1, 10, 2, 10, draws=0, floor_pp=0.0, seed=1)


def test_posterior_density_integrates_to_about_one():
    xs, density = bayes.posterior_density(8_502, 44_700)
    assert np.trapezoid(density, xs) == pytest.approx(1.0, abs=1e-3)


def test_bootstrap_interval_brackets_the_point_estimate():
    result = resampling.bootstrap_proportion_difference(
        "retention_7", 8_502, 44_700, 8_279, 45_489, draws=20_000, floor_pp=-0.2, seed=5
    )
    assert result.ci_low_pp < result.point_estimate_pp < result.ci_high_pp
    assert result.mean_pp == pytest.approx(result.point_estimate_pp, abs=0.02)
    assert result.share_of_draws_negative > 0.99


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    kwargs = {
        "metric": "m",
        "successes_control": 8_502,
        "n_control": 44_700,
        "successes_treatment": 8_279,
        "n_treatment": 45_489,
        "draws": 4_000,
        "floor_pp": -0.2,
    }
    first = resampling.bootstrap_proportion_difference(**kwargs, seed=6)
    second = resampling.bootstrap_proportion_difference(**kwargs, seed=6)
    third = resampling.bootstrap_proportion_difference(**kwargs, seed=7)
    assert first.ci_low_pp == second.ci_low_pp
    assert first.ci_low_pp != third.ci_low_pp


def test_bootstrap_matches_the_analytic_standard_error():
    n_control = n_treatment = 20_000
    rate = 0.19
    successes = int(rate * n_control)
    result = resampling.bootstrap_proportion_difference(
        "m", successes, n_control, successes, n_treatment, draws=40_000, floor_pp=-0.2, seed=8
    )
    analytic = np.sqrt(2 * rate * (1 - rate) / n_control) * 100.0
    assert result.std_error_pp == pytest.approx(analytic, rel=0.05)


def test_bootstrap_rejects_zero_draws():
    with pytest.raises(ValueError, match="draws must be positive"):
        resampling.bootstrap_proportion_difference("m", 1, 10, 1, 10, draws=0, floor_pp=0.0, seed=1)


def test_trimmed_mean_bootstrap_covers_zero_when_arms_match():
    generator = np.random.default_rng(9)
    control = generator.gamma(2.0, 30.0, size=3_000)
    treatment = generator.gamma(2.0, 30.0, size=3_000)
    point, low, high = resampling.bootstrap_trimmed_mean_difference(
        control, treatment, draws=400, seed=10
    )
    assert low < point < high
    assert low < 0 < high
