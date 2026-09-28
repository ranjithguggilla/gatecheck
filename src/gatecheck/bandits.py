"""Adaptive allocation, replayed against the outcomes the experiment actually produced."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

POLICIES = ("uniform", "epsilon_greedy", "ucb1", "thompson")


@dataclass(frozen=True)
class BanditResult:
    policy: str
    seeds: int
    horizon: int
    arm_rates: tuple[float, float]
    mean_regret: float
    std_regret: float
    mean_pulls_inferior_arm: float
    std_pulls_inferior_arm: float
    mean_share_best_arm: float
    mean_retained_players: float
    identified_best_arm_share: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _uniform(step: int, successes: np.ndarray, pulls: np.ndarray, rng: np.random.Generator):
    seeds = successes.shape[1]
    return np.full(seeds, step % 2, dtype="int64")


def _epsilon_greedy(
    step: int,
    successes: np.ndarray,
    pulls: np.ndarray,
    rng: np.random.Generator,
    epsilon: float = 0.1,
):
    seeds = successes.shape[1]
    means = np.where(pulls > 0, successes / np.maximum(pulls, 1), 1.0)
    greedy = means.argmax(axis=0)
    explore = rng.random(seeds) < epsilon
    random_arm = rng.integers(0, 2, size=seeds)
    return np.where(explore, random_arm, greedy)


def _ucb1(step: int, successes: np.ndarray, pulls: np.ndarray, rng: np.random.Generator):
    unplayed = pulls == 0
    if unplayed.any():
        # Force one pull of each arm before the bound means anything.
        forced = np.where(unplayed[0], 0, np.where(unplayed[1], 1, -1))
        if (forced >= 0).all():
            return forced.astype("int64")
    safe_pulls = np.maximum(pulls, 1)
    means = successes / safe_pulls
    bonus = np.sqrt(2.0 * np.log(max(step, 2)) / safe_pulls)
    scores = np.where(unplayed, np.inf, means + bonus)
    return scores.argmax(axis=0)


def _thompson(step: int, successes: np.ndarray, pulls: np.ndarray, rng: np.random.Generator):
    failures = pulls - successes
    draws = rng.beta(successes + 1.0, failures + 1.0)
    return draws.argmax(axis=0)


_POLICY_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "uniform": _uniform,
    "epsilon_greedy": _epsilon_greedy,
    "ucb1": _ucb1,
    "thompson": _thompson,
}


def replay_policy(
    policy: str,
    arm_rates: tuple[float, float],
    horizon: int,
    seeds: int,
    seed: int,
    record_every: int = 500,
) -> tuple[BanditResult, np.ndarray, np.ndarray]:
    """Run one allocation policy over a horizon of arrivals, across many seeds at once.

    Rewards come from each arm's observed day-7 outcome pool. Drawing with
    replacement from a column of zeros and ones is exactly a Bernoulli draw at
    that arm's measured rate, so the replay is faithful to the data and still
    vectorises across all seeds in one pass.
    """
    if policy not in _POLICY_FUNCTIONS:
        raise ValueError(f"unknown policy {policy!r}; expected one of {sorted(_POLICY_FUNCTIONS)}")
    if horizon <= 0 or seeds <= 0:
        raise ValueError("horizon and seeds must both be positive")
    rng = np.random.default_rng(seed)
    rates = np.asarray(arm_rates, dtype="float64").reshape(2, 1)
    best_arm = int(rates.argmax())
    best_rate = float(rates.max())
    successes = np.zeros((2, seeds), dtype="float64")
    pulls = np.zeros((2, seeds), dtype="float64")
    regret = np.zeros(seeds, dtype="float64")
    rewards_total = np.zeros(seeds, dtype="float64")
    choose = _POLICY_FUNCTIONS[policy]
    checkpoints: list[int] = []
    regret_track: list[np.ndarray] = []
    seed_index = np.arange(seeds)
    for step in range(1, horizon + 1):
        arms = choose(step, successes, pulls, rng)
        chosen_rate = rates[arms, 0]
        reward = (rng.random(seeds) < chosen_rate).astype("float64")
        successes[arms, seed_index] += reward
        pulls[arms, seed_index] += 1.0
        regret += best_rate - chosen_rate
        rewards_total += reward
        if step % record_every == 0 or step == horizon:
            checkpoints.append(step)
            regret_track.append(regret.copy())
    inferior_pulls = pulls[1 - best_arm]
    final_means = np.where(pulls > 0, successes / np.maximum(pulls, 1), 0.0)
    result = BanditResult(
        policy=policy,
        seeds=int(seeds),
        horizon=int(horizon),
        arm_rates=(float(arm_rates[0]), float(arm_rates[1])),
        mean_regret=float(regret.mean()),
        std_regret=float(regret.std(ddof=1)),
        mean_pulls_inferior_arm=float(inferior_pulls.mean()),
        std_pulls_inferior_arm=float(inferior_pulls.std(ddof=1)),
        mean_share_best_arm=float((pulls[best_arm] / horizon).mean()),
        mean_retained_players=float(rewards_total.mean()),
        identified_best_arm_share=float((final_means.argmax(axis=0) == best_arm).mean()),
    )
    return result, np.asarray(checkpoints), np.vstack(regret_track).mean(axis=1)


def compare_policies(
    arm_rates: tuple[float, float],
    horizon: int,
    seeds: int,
    seed: int,
    policies: tuple[str, ...] = POLICIES,
) -> tuple[dict[str, BanditResult], np.ndarray, dict[str, np.ndarray]]:
    """Run every policy on the same arrivals budget and line the results up."""
    results: dict[str, BanditResult] = {}
    curves: dict[str, np.ndarray] = {}
    checkpoints = np.array([], dtype="int64")
    for policy in policies:
        # Every policy starts from the same seed, so the comparison between them is
        # not partly a comparison between random streams.
        result, checkpoints, curve = replay_policy(policy, arm_rates, horizon, seeds, seed)
        results[policy] = result
        curves[policy] = curve
    return results, checkpoints, curves


def players_spared(baseline: BanditResult, candidate: BanditResult) -> dict[str, float]:
    """Translate the regret gap into players and retained users."""
    spared = baseline.mean_pulls_inferior_arm - candidate.mean_pulls_inferior_arm
    return {
        "players_spared_inferior_arm": float(spared),
        "extra_retained_players": float(
            candidate.mean_retained_players - baseline.mean_retained_players
        ),
        "regret_reduction": float(baseline.mean_regret - candidate.mean_regret),
        "regret_reduction_pct": float(
            (baseline.mean_regret - candidate.mean_regret) / baseline.mean_regret * 100.0
        )
        if baseline.mean_regret > 0
        else float("nan"),
    }
