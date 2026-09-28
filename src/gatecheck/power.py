"""How small an effect this experiment could have seen, and how big a sample the next one needs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PowerReport:
    metric: str
    baseline_rate: float
    n_per_arm: int
    alpha: float
    target_power: float
    mde_absolute_pp: float
    mde_relative_pct: float
    achieved_power_at_observed_effect: float
    observed_effect_pp: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def required_n_per_arm(
    baseline_rate: float, absolute_lift: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Two sided, equal allocation, normal approximation sample size."""
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must sit in (0, 1)")
    if absolute_lift == 0:
        raise ValueError("absolute_lift must be non-zero")
    treated_rate = np.clip(baseline_rate + absolute_lift, 1e-9, 1 - 1e-9)
    z_alpha = stats.norm.isf(alpha / 2.0)
    z_beta = stats.norm.isf(1.0 - power)
    pooled = (baseline_rate + treated_rate) / 2.0
    numerator = (
        z_alpha * np.sqrt(2 * pooled * (1 - pooled))
        + z_beta
        * np.sqrt(baseline_rate * (1 - baseline_rate) + treated_rate * (1 - treated_rate))
    ) ** 2
    return int(np.ceil(numerator / absolute_lift**2))


def achieved_power(
    baseline_rate: float, absolute_lift: float, n_per_arm: int, alpha: float = 0.05
) -> float:
    """Power of a completed test against the effect it happened to observe."""
    treated_rate = np.clip(baseline_rate + absolute_lift, 1e-9, 1 - 1e-9)
    pooled = (baseline_rate + treated_rate) / 2.0
    se_null = np.sqrt(2 * pooled * (1 - pooled) / n_per_arm)
    se_alt = np.sqrt(
        (baseline_rate * (1 - baseline_rate) + treated_rate * (1 - treated_rate)) / n_per_arm
    )
    critical = stats.norm.isf(alpha / 2.0) * se_null
    shift = abs(absolute_lift)
    return float(
        stats.norm.sf((critical - shift) / se_alt) + stats.norm.cdf((-critical - shift) / se_alt)
    )


def minimum_detectable_effect(
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
    tolerance: float = 1e-7,
) -> float:
    """Smallest absolute lift this sample size can find, by bisection on the power curve."""
    low, high = tolerance, 0.5
    for _ in range(200):
        middle = (low + high) / 2.0
        if achieved_power(baseline_rate, middle, n_per_arm, alpha) < power:
            low = middle
        else:
            high = middle
        if high - low < tolerance:
            break
    return float((low + high) / 2.0)


def power_report(
    metric: str,
    baseline_rate: float,
    n_per_arm: int,
    observed_effect: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> PowerReport:
    mde = minimum_detectable_effect(baseline_rate, n_per_arm, alpha, target_power)
    return PowerReport(
        metric=metric,
        baseline_rate=float(baseline_rate),
        n_per_arm=int(n_per_arm),
        alpha=float(alpha),
        target_power=float(target_power),
        mde_absolute_pp=float(mde * 100.0),
        mde_relative_pct=float(mde / baseline_rate * 100.0),
        achieved_power_at_observed_effect=achieved_power(
            baseline_rate, observed_effect, n_per_arm, alpha
        ),
        observed_effect_pp=float(observed_effect * 100.0),
    )


def sample_size_curve(
    baseline_rate: float,
    relative_lifts: np.ndarray,
    alpha: float = 0.05,
    power: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    """Required n per arm across a sweep of relative effect sizes."""
    sizes = np.array(
        [
            required_n_per_arm(baseline_rate, baseline_rate * lift, alpha, power)
            for lift in relative_lifts
        ],
        dtype="int64",
    )
    return relative_lifts, sizes
