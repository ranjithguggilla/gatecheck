"""Bootstrap intervals, without pretending a Bernoulli sample needs brute force."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    draws: int
    point_estimate_pp: float
    mean_pp: float
    std_error_pp: float
    ci_low_pp: float
    ci_high_pp: float
    share_of_draws_negative: float
    share_of_draws_below_floor: float
    floor_pp: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_proportion_difference(
    metric: str,
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    draws: int,
    floor_pp: float,
    seed: int,
) -> BootstrapResult:
    """Bootstrap the difference in two rates, exactly and cheaply.

    Resampling a 0/1 column with replacement and taking the mean is the same
    experiment as drawing Binomial(n, p_hat) / n, so the loop over 90,189 rows is
    replaced by two binomial draws per replicate. Same distribution, a few
    milliseconds instead of a few minutes.
    """
    if draws <= 0:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    rate_c = successes_control / n_control
    rate_t = successes_treatment / n_treatment
    sampled_c = rng.binomial(n_control, rate_c, size=draws) / n_control
    sampled_t = rng.binomial(n_treatment, rate_t, size=draws) / n_treatment
    differences = (sampled_t - sampled_c) * 100.0
    low, high = np.percentile(differences, [2.5, 97.5])
    return BootstrapResult(
        metric=metric,
        draws=int(draws),
        point_estimate_pp=float((rate_t - rate_c) * 100.0),
        mean_pp=float(differences.mean()),
        std_error_pp=float(differences.std(ddof=1)),
        ci_low_pp=float(low),
        ci_high_pp=float(high),
        share_of_draws_negative=float((differences < 0).mean()),
        share_of_draws_below_floor=float((differences < floor_pp).mean()),
        floor_pp=float(floor_pp),
    )


def bootstrap_difference_samples(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    """The raw replicate differences in percentage points, for plotting."""
    rng = np.random.default_rng(seed)
    rate_c = successes_control / n_control
    rate_t = successes_treatment / n_treatment
    # Control first, matching the order in bootstrap_proportion_difference, so the
    # plotted replicates are the same sample the summary describes.
    sampled_c = rng.binomial(n_control, rate_c, size=draws) / n_control
    sampled_t = rng.binomial(n_treatment, rate_t, size=draws) / n_treatment
    return (sampled_t - sampled_c) * 100.0


def bootstrap_trimmed_mean_difference(
    control: np.ndarray,
    treatment: np.ndarray,
    draws: int,
    seed: int,
    trim: float = 0.01,
    chunk: int = 250,
) -> tuple[float, float, float]:
    """Bootstrap the difference in trimmed means for the skewed engagement metric.

    Chunked so the index matrix never gets large enough to matter, and trimmed
    rather than winsorised so a single 49,854 round player cannot set the scale.
    """
    control = np.asarray(control, dtype="float64")
    treatment = np.asarray(treatment, dtype="float64")
    rng = np.random.default_rng(seed)
    point = float(
        stats.trim_mean(treatment, trim) - stats.trim_mean(control, trim)
    )
    replicates: list[np.ndarray] = []
    remaining = draws
    while remaining > 0:
        size = min(chunk, remaining)
        idx_c = rng.integers(0, len(control), size=(size, len(control)))
        idx_t = rng.integers(0, len(treatment), size=(size, len(treatment)))
        replicates.append(
            stats.trim_mean(treatment[idx_t], trim, axis=1)
            - stats.trim_mean(control[idx_c], trim, axis=1)
        )
        remaining -= size
    samples = np.concatenate(replicates)
    low, high = np.percentile(samples, [2.5, 97.5])
    return point, float(low), float(high)
