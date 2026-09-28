from __future__ import annotations

import numpy as np
import pytest

from gatecheck import peeking


@pytest.fixture(scope="module")
def coin_flips() -> np.ndarray:
    return (np.random.default_rng(21).random(12_000) < 0.19).astype("float64")


def test_trajectories_have_the_requested_shape(coin_flips):
    matrix = peeking.aa_z_trajectories(coin_flips, replications=25, looks=5, seed=1)
    assert matrix.shape == (25, 5)
    assert np.isfinite(matrix).all()


def test_zero_looks_is_rejected(coin_flips):
    with pytest.raises(ValueError, match="at least 1"):
        peeking.aa_z_trajectories(coin_flips, replications=5, looks=0, seed=1)


def test_too_many_looks_for_the_sample_is_rejected():
    with pytest.raises(ValueError, match="not enough observations"):
        peeking.aa_z_trajectories(np.ones(10), replications=2, looks=50, seed=1)


def test_a_single_look_is_calibrated_at_the_nominal_rate(coin_flips):
    study = peeking.peeking_study("m", coin_flips, replications=800, looks=1, alpha=0.05, seed=2)
    assert study.fixed_horizon_false_positive_rate == pytest.approx(0.05, abs=0.03)
    assert study.naive_peeking_false_positive_rate == study.fixed_horizon_false_positive_rate


def test_peeking_inflates_the_false_positive_rate(coin_flips):
    study = peeking.peeking_study("m", coin_flips, replications=800, looks=10, alpha=0.05, seed=3)
    assert study.naive_peeking_false_positive_rate > study.fixed_horizon_false_positive_rate
    assert study.inflation_factor > 1.5


def test_the_calibrated_boundary_pulls_the_rate_back_to_target(coin_flips):
    study = peeking.peeking_study("m", coin_flips, replications=1_000, looks=10, alpha=0.05, seed=4)
    assert study.calibrated_boundary_z > 1.96
    assert study.calibrated_false_positive_rate == pytest.approx(0.05, abs=0.035)
    assert study.boundary_z_ratio > 1.0
    assert study.sample_size_inflation_at_80_power > study.boundary_z_ratio


def test_the_sample_size_cost_is_a_multiplier_above_one():
    assert peeking.boundary_sample_size_cost(1.9599639845400545, 0.05) == pytest.approx(1.0)
    assert peeking.boundary_sample_size_cost(2.53, 0.05) > 1.4
    assert peeking.boundary_sample_size_cost(3.0, 0.05) > peeking.boundary_sample_size_cost(
        2.5, 0.05
    )


def test_boundary_calibration_is_monotone_in_alpha():
    matrix = np.random.default_rng(5).normal(size=(4_000, 8))
    strict = peeking.calibrate_constant_boundary(matrix, alpha=0.01)
    loose = peeking.calibrate_constant_boundary(matrix, alpha=0.10)
    assert strict > loose


def test_sequential_readout_walks_forward_and_finds_a_real_effect():
    generator = np.random.default_rng(6)
    control = (generator.random(20_000) < 0.30).astype("float64")
    treatment = (generator.random(20_000) < 0.20).astype("float64")
    readout = peeking.sequential_readout(
        "m", control, treatment, looks=5, alpha=0.05, boundary_z=2.5, seed=7
    )
    assert len(readout.z_values) == 5
    assert readout.first_look_crossing_naive is not None
    assert readout.first_look_crossing_boundary is not None
    assert readout.players_saved_by_early_stop >= 0
    assert readout.players_per_arm_used == 20_000


def test_sequential_readout_finds_nothing_when_there_is_nothing():
    generator = np.random.default_rng(8)
    control = (generator.random(8_000) < 0.25).astype("float64")
    treatment = (generator.random(8_000) < 0.25).astype("float64")
    readout = peeking.sequential_readout(
        "m", control, treatment, looks=4, alpha=0.05, boundary_z=3.5, seed=9
    )
    assert readout.first_look_crossing_boundary is None
    assert readout.players_saved_by_early_stop == 0
