"""Beta-Binomial posteriors and the loss a decision would carry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import arviz as az
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PosteriorComparison:
    metric: str
    prior_alpha: float
    prior_beta: float
    draws: int
    posterior_mean_control: float
    posterior_mean_treatment: float
    prob_treatment_better: float
    diff_mean_pp: float
    hdi_low_pp: float
    hdi_high_pp: float
    expected_loss_if_ship_pp: float
    expected_loss_if_hold_pp: float
    prob_worse_than_floor: float
    floor_pp: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def posterior_comparison(
    metric: str,
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    draws: int,
    floor_pp: float,
    seed: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> PosteriorComparison:
    """Conjugate update, then Monte Carlo for everything that is not closed form.

    A uniform Beta(1, 1) prior is deliberate: with 45,000 players per arm the
    prior is worth about one observation, so nobody can claim the answer was
    assumed rather than measured.
    """
    if draws <= 0:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    post_c = stats.beta(prior_alpha + successes_control, prior_beta + n_control - successes_control)
    post_t = stats.beta(
        prior_alpha + successes_treatment, prior_beta + n_treatment - successes_treatment
    )
    sample_c = post_c.rvs(size=draws, random_state=rng)
    sample_t = post_t.rvs(size=draws, random_state=rng)
    difference = (sample_t - sample_c) * 100.0
    hdi = az.hdi(difference, prob=0.95)
    treatment_better = sample_t > sample_c
    # Expected loss is the average size of the mistake, counted only when the
    # decision turns out to be the wrong one.
    loss_ship = float(np.where(treatment_better, 0.0, -difference).mean())
    loss_hold = float(np.where(treatment_better, difference, 0.0).mean())
    return PosteriorComparison(
        metric=metric,
        prior_alpha=float(prior_alpha),
        prior_beta=float(prior_beta),
        draws=int(draws),
        posterior_mean_control=float(post_c.mean()),
        posterior_mean_treatment=float(post_t.mean()),
        prob_treatment_better=float(treatment_better.mean()),
        diff_mean_pp=float(difference.mean()),
        hdi_low_pp=float(hdi[0]),
        hdi_high_pp=float(hdi[1]),
        expected_loss_if_ship_pp=loss_ship,
        expected_loss_if_hold_pp=loss_hold,
        prob_worse_than_floor=float((difference < floor_pp).mean()),
        floor_pp=float(floor_pp),
    )


def difference_samples(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    draws: int,
    seed: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> np.ndarray:
    """Posterior draws of the difference in percentage points, for plotting."""
    rng = np.random.default_rng(seed)
    post_c = stats.beta(prior_alpha + successes_control, prior_beta + n_control - successes_control)
    post_t = stats.beta(
        prior_alpha + successes_treatment, prior_beta + n_treatment - successes_treatment
    )
    # Control first, matching posterior_comparison, so the histogram and the
    # summary describe the same draws.
    sampled_c = post_c.rvs(size=draws, random_state=rng)
    sampled_t = post_t.rvs(size=draws, random_state=rng)
    return (sampled_t - sampled_c) * 100.0


def posterior_density(
    successes: int, n: int, prior_alpha: float = 1.0, prior_beta: float = 1.0, grid: int = 800
) -> tuple[np.ndarray, np.ndarray]:
    """Grid of the posterior density, for the chart."""
    posterior = stats.beta(prior_alpha + successes, prior_beta + n - successes)
    low, high = posterior.ppf([1e-5, 1 - 1e-5])
    xs = np.linspace(low, high, grid)
    return xs, posterior.pdf(xs)
