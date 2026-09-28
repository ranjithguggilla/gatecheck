"""Fixed horizon tests, confidence intervals and multiplicity control."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class ProportionComparison:
    """Everything a reviewer asks about a binary metric, computed once."""

    metric: str
    n_control: int
    n_treatment: int
    successes_control: int
    successes_treatment: int
    rate_control: float
    rate_treatment: float
    absolute_diff_pp: float
    relative_lift_pct: float
    z_statistic: float
    p_value: float
    diff_ci_low_pp: float
    diff_ci_high_pp: float
    relative_ci_low_pct: float
    relative_ci_high_pct: float
    chi_square: float
    chi_square_p_value: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _wilson_bounds(successes: int, n: int, z: float) -> tuple[float, float]:
    rate = successes / n
    denominator = 1.0 + z**2 / n
    centre = (rate + z**2 / (2 * n)) / denominator
    spread = (z / denominator) * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
    return float(centre - spread), float(centre + spread)


def newcombe_interval(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Newcombe's hybrid score interval for a difference of two proportions.

    Preferred over the textbook Wald interval because it stays inside [-1, 1] and
    keeps its nominal coverage when a rate drifts near zero or one.
    """
    z = float(stats.norm.isf(alpha / 2.0))
    low_c, high_c = _wilson_bounds(successes_control, n_control, z)
    low_t, high_t = _wilson_bounds(successes_treatment, n_treatment, z)
    rate_c = successes_control / n_control
    rate_t = successes_treatment / n_treatment
    delta = rate_t - rate_c
    lower = delta - np.hypot(rate_t - low_t, high_c - rate_c)
    upper = delta + np.hypot(high_t - rate_t, rate_c - low_c)
    return float(max(lower, -1.0)), float(min(upper, 1.0))


def log_risk_ratio_interval(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Confidence interval for relative lift, built on the log scale.

    Relative lift is a ratio, so its sampling distribution is skewed. Working in
    logs and exponentiating back keeps the interval asymmetric the way it should be.
    """
    if min(successes_control, successes_treatment) == 0:
        return float("nan"), float("nan")
    rate_c = successes_control / n_control
    rate_t = successes_treatment / n_treatment
    log_rr = np.log(rate_t / rate_c)
    se = np.sqrt(
        (1 - rate_t) / successes_treatment + (1 - rate_c) / successes_control
    )
    z = float(stats.norm.isf(alpha / 2.0))
    return float(np.exp(log_rr - z * se) - 1.0), float(np.exp(log_rr + z * se) - 1.0)


def two_proportion_test(
    metric: str,
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> ProportionComparison:
    """Pooled two sided z-test, with a chi-square cross-check and both intervals."""
    if min(n_control, n_treatment) <= 0:
        raise ValueError("both arms need at least one unit")
    if not 0 <= successes_control <= n_control or not 0 <= successes_treatment <= n_treatment:
        raise ValueError("successes must not exceed the arm size")
    rate_c = successes_control / n_control
    rate_t = successes_treatment / n_treatment
    pooled = (successes_control + successes_treatment) / (n_control + n_treatment)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    # A zero standard error means nobody converted in either arm, or everybody did.
    # There is no signal and no division to do, so the statistic is zero by definition.
    z_stat = float((rate_t - rate_c) / se) if se > 0 else 0.0
    p_value = float(2.0 * stats.norm.sf(abs(z_stat)))
    table = np.array(
        [
            [successes_treatment, n_treatment - successes_treatment],
            [successes_control, n_control - successes_control],
        ]
    )
    if table.sum(axis=0).min() == 0:
        # Nobody converted in either arm, or everybody did. There is no
        # association to test, and scipy refuses a table with an empty column.
        chi2, chi2_p = 0.0, 1.0
    else:
        chi2, chi2_p, _, _ = stats.chi2_contingency(table, correction=True)
    diff_low, diff_high = newcombe_interval(
        successes_control, n_control, successes_treatment, n_treatment, alpha
    )
    rel_low, rel_high = log_risk_ratio_interval(
        successes_control, n_control, successes_treatment, n_treatment, alpha
    )
    return ProportionComparison(
        metric=metric,
        n_control=int(n_control),
        n_treatment=int(n_treatment),
        successes_control=int(successes_control),
        successes_treatment=int(successes_treatment),
        rate_control=float(rate_c),
        rate_treatment=float(rate_t),
        absolute_diff_pp=float((rate_t - rate_c) * 100.0),
        relative_lift_pct=float((rate_t / rate_c - 1.0) * 100.0) if rate_c > 0 else float("nan"),
        z_statistic=z_stat,
        p_value=p_value,
        diff_ci_low_pp=diff_low * 100.0,
        diff_ci_high_pp=diff_high * 100.0,
        relative_ci_low_pct=rel_low * 100.0,
        relative_ci_high_pct=rel_high * 100.0,
        chi_square=float(chi2),
        chi_square_p_value=float(chi2_p),
    )


@dataclass(frozen=True)
class ContinuousComparison:
    metric: str
    mean_control: float
    mean_treatment: float
    median_control: float
    median_treatment: float
    welch_t: float
    welch_p_value: float
    mann_whitney_u: float
    mann_whitney_p_value: float
    rank_biserial: float
    cohens_d: float
    winsor_cut: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_continuous(
    metric: str,
    control: np.ndarray,
    treatment: np.ndarray,
    winsor_cut: float,
) -> ContinuousComparison:
    """Welch's t on winsorised values plus a rank test on the raw ones.

    The raw distribution is heavily right skewed, so neither test alone is
    convincing. Agreement between them is the thing worth reporting.
    """
    control = np.asarray(control, dtype="float64")
    treatment = np.asarray(treatment, dtype="float64")
    clipped_c = np.clip(control, None, winsor_cut)
    clipped_t = np.clip(treatment, None, winsor_cut)
    t_stat, t_p = stats.ttest_ind(clipped_t, clipped_c, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(treatment, control, alternative="two-sided")
    n_product = len(control) * len(treatment)
    pooled_sd = np.sqrt((clipped_c.var(ddof=1) + clipped_t.var(ddof=1)) / 2.0)
    return ContinuousComparison(
        metric=metric,
        mean_control=float(control.mean()),
        mean_treatment=float(treatment.mean()),
        median_control=float(np.median(control)),
        median_treatment=float(np.median(treatment)),
        welch_t=float(t_stat),
        welch_p_value=float(t_p),
        mann_whitney_u=float(u_stat),
        mann_whitney_p_value=float(u_p),
        rank_biserial=float(2.0 * u_stat / n_product - 1.0),
        cohens_d=float((clipped_t.mean() - clipped_c.mean()) / pooled_sd)
        if pooled_sd > 0
        else 0.0,
        winsor_cut=float(winsor_cut),
    )


def adjust_p_values(
    labels: Sequence[str], p_values: Sequence[float], alpha: float = 0.05
) -> dict[str, dict[str, Any]]:
    """Holm and Benjamini-Hochberg over the whole metric family.

    Three metrics get read off this experiment, so the chance of at least one
    false flag at 0.05 each is closer to 14% than 5% unless it is controlled.
    """
    p_array = np.asarray(p_values, dtype="float64")
    holm_reject, holm_p, _, _ = multipletests(p_array, alpha=alpha, method="holm")
    bh_reject, bh_p, _, _ = multipletests(p_array, alpha=alpha, method="fdr_bh")
    return {
        label: {
            "raw_p_value": float(p_array[index]),
            "holm_p_value": float(holm_p[index]),
            "holm_significant": bool(holm_reject[index]),
            "benjamini_hochberg_p_value": float(bh_p[index]),
            "benjamini_hochberg_significant": bool(bh_reject[index]),
        }
        for index, label in enumerate(labels)
    }
