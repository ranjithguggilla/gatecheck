from __future__ import annotations

import numpy as np
import pytest

from gatecheck import bandits


def test_uniform_splits_the_horizon_exactly_in_half():
    result, _, _ = bandits.replay_policy("uniform", (0.19, 0.18), horizon=2_000, seeds=3, seed=1)
    assert result.mean_pulls_inferior_arm == pytest.approx(1_000.0)
    assert result.mean_share_best_arm == pytest.approx(0.5)
    assert result.std_pulls_inferior_arm == pytest.approx(0.0)


def test_thompson_sends_most_traffic_to_the_better_arm():
    result, _, _ = bandits.replay_policy("thompson", (0.30, 0.15), horizon=3_000, seeds=8, seed=2)
    assert result.mean_share_best_arm > 0.75
    assert result.identified_best_arm_share > 0.8


def test_thompson_beats_uniform_on_regret():
    uniform, _, _ = bandits.replay_policy("uniform", (0.30, 0.15), horizon=3_000, seeds=8, seed=3)
    thompson, _, _ = bandits.replay_policy("thompson", (0.30, 0.15), horizon=3_000, seeds=8, seed=3)
    assert thompson.mean_regret < uniform.mean_regret
    assert thompson.mean_retained_players > uniform.mean_retained_players


def test_identical_arms_leave_regret_at_zero():
    result, _, _ = bandits.replay_policy("thompson", (0.20, 0.20), horizon=1_500, seeds=4, seed=4)
    assert result.mean_regret == pytest.approx(0.0)


def test_ucb1_pulls_each_arm_before_it_starts_exploiting():
    result, _, _ = bandits.replay_policy("ucb1", (0.40, 0.10), horizon=1_200, seeds=4, seed=5)
    assert result.mean_pulls_inferior_arm >= 1.0
    assert result.mean_share_best_arm > 0.5


def test_epsilon_greedy_keeps_exploring():
    result, _, _ = bandits.replay_policy(
        "epsilon_greedy", (0.40, 0.10), horizon=2_000, seeds=6, seed=6
    )
    assert 0.0 < result.mean_pulls_inferior_arm < result.horizon


def test_regret_curve_is_recorded_and_non_decreasing():
    _, checkpoints, curve = bandits.replay_policy(
        "thompson", (0.30, 0.15), horizon=2_000, seeds=5, seed=7, record_every=250
    )
    assert checkpoints[-1] == 2_000
    assert np.all(np.diff(curve) >= -1e-9)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown policy"):
        bandits.replay_policy("bandit_of_dreams", (0.2, 0.1), horizon=10, seeds=1, seed=1)


def test_non_positive_horizon_is_rejected():
    with pytest.raises(ValueError, match="must both be positive"):
        bandits.replay_policy("thompson", (0.2, 0.1), horizon=0, seeds=1, seed=1)


def test_compare_policies_covers_every_policy():
    results, checkpoints, curves = bandits.compare_policies(
        (0.30, 0.15), horizon=1_000, seeds=4, seed=8
    )
    assert set(results) == set(bandits.POLICIES)
    assert set(curves) == set(bandits.POLICIES)
    assert len(checkpoints) == len(curves["thompson"])


def test_every_policy_starts_from_the_same_seed():
    results, _, _ = bandits.compare_policies((0.30, 0.15), horizon=800, seeds=3, seed=11)
    direct, _, _ = bandits.replay_policy("ucb1", (0.30, 0.15), horizon=800, seeds=3, seed=11)
    assert results["ucb1"].mean_regret == pytest.approx(direct.mean_regret)


def test_savings_against_uniform_are_positive_for_thompson():
    uniform, _, _ = bandits.replay_policy("uniform", (0.30, 0.15), horizon=2_000, seeds=6, seed=9)
    thompson, _, _ = bandits.replay_policy("thompson", (0.30, 0.15), horizon=2_000, seeds=6, seed=9)
    savings = bandits.players_spared(uniform, thompson)
    assert savings["players_spared_inferior_arm"] > 0
    assert savings["regret_reduction_pct"] > 0
