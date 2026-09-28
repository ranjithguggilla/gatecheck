"""What repeated looks at a running experiment actually cost, measured on this data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PeekingStudy:
    metric: str
    replications: int
    calibration_replications: int
    validation_replications: int
    looks: int
    alpha: float
    n_per_pseudo_arm: int
    fixed_horizon_false_positive_rate: float
    naive_peeking_false_positive_rate: float
    bonferroni_false_positive_rate: float
    calibrated_boundary_z: float
    calibrated_false_positive_rate: float
    inflation_factor: float
    boundary_z_ratio: float
    sample_size_inflation_at_80_power: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pooled_z(successes_a: np.ndarray, successes_b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Pooled two proportion z-statistic, broadcast over replications and looks."""
    rate_a = successes_a / n
    rate_b = successes_b / n
    pooled = (successes_a + successes_b) / (2.0 * n)
    variance = pooled * (1.0 - pooled) * (2.0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (rate_b - rate_a) / np.sqrt(variance)
    # Early looks can land on zero conversions in both halves, which is a zero
    # variance rather than an infinite effect. Those looks contribute nothing.
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def aa_z_trajectories(
    outcomes: np.ndarray, replications: int, looks: int, seed: int
) -> np.ndarray:
    """Split one arm against itself over and over, and record the z-statistic at each look.

    Any effect these trajectories show is noise by construction: both halves come
    from the same players under the same build. That makes them the right ruler
    for how often a stopping rule fires when there is nothing to find.
    """
    if looks < 1:
        raise ValueError("looks must be at least 1")
    values = np.asarray(outcomes, dtype="float64")
    half = len(values) // 2
    if half < looks:
        raise ValueError("not enough observations for the requested number of looks")
    rng = np.random.default_rng(seed)
    step = half // looks
    checkpoints = np.arange(1, looks + 1) * step
    z_matrix = np.empty((replications, looks), dtype="float64")
    for index in range(replications):
        shuffled = rng.permutation(values)
        cumulative_a = np.cumsum(shuffled[:half])[checkpoints - 1]
        cumulative_b = np.cumsum(shuffled[half : 2 * half])[checkpoints - 1]
        z_matrix[index] = _pooled_z(cumulative_a, cumulative_b, checkpoints.astype("float64"))
    return z_matrix


def calibrate_constant_boundary(
    z_matrix: np.ndarray, alpha: float, tolerance: float = 1e-4
) -> float:
    """Find the flat z-boundary whose family-wise false positive rate hits alpha.

    This is the Pocock idea, except the constant is read off replications of this
    dataset rather than taken from a table built for a different look schedule.
    """
    peak = np.abs(z_matrix).max(axis=1)
    low, high = 1.0, 8.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if (peak > middle).mean() > alpha:
            low = middle
        else:
            high = middle
        if high - low < tolerance:
            break
    return float((low + high) / 2.0)


def boundary_sample_size_cost(boundary_z: float, alpha: float, power: float = 0.80) -> float:
    """How much larger a sample has to be to keep the same power under the boundary.

    Raising the critical value buys back the error rate that peeking spent, and the
    bill arrives as sample size. The ratio of z thresholds understates it, because
    sample size scales with the square of the combined threshold and the power term
    does not move. Reported as a multiplier: 1.45 means 45% more players.
    """
    z_beta = float(stats.norm.isf(1.0 - power))
    z_alpha = float(stats.norm.isf(alpha / 2.0))
    return float(((boundary_z + z_beta) / (z_alpha + z_beta)) ** 2)


def peeking_study(
    metric: str,
    control_outcomes: np.ndarray,
    replications: int,
    looks: int,
    alpha: float,
    seed: int,
) -> PeekingStudy:
    """Quantify the peeking penalty and price the fix.

    The replications are split in half: the first half calibrates the boundary,
    the second half measures how it behaves on data it has never seen. Calibrating
    and reporting on the same replications would flatter the boundary.
    """
    z_matrix = aa_z_trajectories(control_outcomes, replications, looks, seed)
    split = replications // 2
    calibration, validation = z_matrix[:split], z_matrix[split:]
    critical = float(stats.norm.isf(alpha / 2.0))
    bonferroni_critical = float(stats.norm.isf(alpha / (2.0 * looks)))
    boundary = calibrate_constant_boundary(calibration, alpha)
    peak_validation = np.abs(validation).max(axis=1)
    fixed_rate = float((np.abs(validation[:, -1]) > critical).mean())
    naive_rate = float((peak_validation > critical).mean())
    return PeekingStudy(
        metric=metric,
        replications=int(replications),
        calibration_replications=int(split),
        validation_replications=int(replications - split),
        looks=int(looks),
        alpha=float(alpha),
        n_per_pseudo_arm=int(len(control_outcomes) // 2),
        fixed_horizon_false_positive_rate=fixed_rate,
        naive_peeking_false_positive_rate=naive_rate,
        bonferroni_false_positive_rate=float((peak_validation > bonferroni_critical).mean()),
        calibrated_boundary_z=boundary,
        calibrated_false_positive_rate=float((peak_validation > boundary).mean()),
        inflation_factor=float(naive_rate / fixed_rate) if fixed_rate > 0 else float("nan"),
        boundary_z_ratio=float(boundary / critical),
        sample_size_inflation_at_80_power=boundary_sample_size_cost(boundary, alpha),
    )


@dataclass(frozen=True)
class SequentialReadout:
    metric: str
    looks: int
    boundary_z: float
    naive_boundary_z: float
    players_per_arm_used: int
    sample_sizes: list[int]
    z_values: list[float]
    first_look_crossing_naive: int | None
    first_look_crossing_boundary: int | None
    players_saved_by_early_stop: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sequential_readout(
    metric: str,
    control_outcomes: np.ndarray,
    treatment_outcomes: np.ndarray,
    looks: int,
    alpha: float,
    boundary_z: float,
    seed: int,
) -> SequentialReadout:
    """Walk the real experiment forward and see when each rule would have called it.

    Players arrive in an arbitrary order in the file, so the arrival order is
    shuffled once with a seed of its own, kept separate from the one driving the A/A
    replications so the trajectory and the boundary it is judged against are not the
    same draw. Both arms are truncated to the smaller of the two, since a look is
    defined by equal exposure; on this data that sets aside 789 treatment players.
    The point is the shape of the trajectory, not the exact look at which it crosses.
    """
    rng = np.random.default_rng(seed)
    control = rng.permutation(np.asarray(control_outcomes, dtype="float64"))
    treatment = rng.permutation(np.asarray(treatment_outcomes, dtype="float64"))
    per_arm = min(len(control), len(treatment))
    step = per_arm // looks
    checkpoints = np.arange(1, looks + 1) * step
    z_values = _pooled_z(
        np.cumsum(control[:per_arm])[checkpoints - 1],
        np.cumsum(treatment[:per_arm])[checkpoints - 1],
        checkpoints.astype("float64"),
    )
    naive_critical = float(stats.norm.isf(alpha / 2.0))
    crossings_naive = np.flatnonzero(np.abs(z_values) > naive_critical)
    crossings_boundary = np.flatnonzero(np.abs(z_values) > boundary_z)
    first_boundary = int(crossings_boundary[0] + 1) if crossings_boundary.size else None
    saved = 0 if first_boundary is None else int(2 * (per_arm - checkpoints[first_boundary - 1]))
    return SequentialReadout(
        metric=metric,
        looks=int(looks),
        boundary_z=float(boundary_z),
        naive_boundary_z=naive_critical,
        players_per_arm_used=int(per_arm),
        sample_sizes=[int(value) for value in checkpoints],
        z_values=[float(value) for value in z_values],
        first_look_crossing_naive=int(crossings_naive[0] + 1) if crossings_naive.size else None,
        first_look_crossing_boundary=first_boundary,
        players_saved_by_early_stop=saved,
    )
