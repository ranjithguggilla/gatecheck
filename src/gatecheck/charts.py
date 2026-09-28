"""Every figure the README embeds, written to assets/."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import bayes  # noqa: E402

PALETTE = {
    "control": "#2f6f8f",
    "treatment": "#c4642a",
    "accent": "#3f7a52",
    "warn": "#a8324a",
    "grid": "#d8d8d8",
}


def _style(axis: plt.Axes) -> None:
    axis.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def retention_bars(comparisons: Sequence[Any], control: str, treatment: str, path: Path) -> Path:
    """Both retention metrics side by side with their intervals on the difference."""
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = [control, treatment]
    for axis, comparison in zip(axes, comparisons, strict=True):
        rates = [comparison.rate_control * 100.0, comparison.rate_treatment * 100.0]
        bars = axis.bar(labels, rates, color=[PALETTE["control"], PALETTE["treatment"]], width=0.55)
        for bar, rate in zip(bars, rates, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.35,
                f"{rate:.2f}%",
                ha="center",
                fontsize=10,
            )
        axis.set_title(
            f"{comparison.metric}\n{comparison.absolute_diff_pp:+.2f}pp "
            f"[{comparison.diff_ci_low_pp:+.2f}, {comparison.diff_ci_high_pp:+.2f}]  "
            f"p={comparison.p_value:.4g}",
            fontsize=10,
        )
        axis.set_ylabel("retained players (%)")
        axis.set_ylim(0, max(rates) * 1.25)
        _style(axis)
    figure.suptitle("Retention by arm, with the interval on the difference", fontsize=12)
    return _save(figure, path)


def bootstrap_histogram(samples: np.ndarray, result: Any, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    axis.hist(samples, bins=90, color=PALETTE["control"], alpha=0.85, edgecolor="white")
    for value, colour, label in (
        (0.0, "black", "no effect"),
        (result.point_estimate_pp, PALETTE["treatment"], "observed"),
        (result.floor_pp, PALETTE["warn"], "tolerance floor"),
    ):
        axis.axvline(value, color=colour, linestyle="--", linewidth=1.4, label=label)
    axis.set_xlabel("day-7 retention difference (percentage points)")
    axis.set_ylabel("bootstrap replicates")
    axis.set_title(
        f"{result.draws:,} bootstrap replicates: "
        f"{result.share_of_draws_negative * 100:.1f}% land below zero",
        fontsize=11,
    )
    axis.legend(frameon=False, fontsize=9)
    _style(axis)
    return _save(figure, path)


def posterior_panel(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    comparison: Any,
    control_label: str,
    treatment_label: str,
    path: Path,
    seed: int,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for successes, total, colour, label in (
        (successes_control, n_control, PALETTE["control"], control_label),
        (successes_treatment, n_treatment, PALETTE["treatment"], treatment_label),
    ):
        xs, density = bayes.posterior_density(successes, total)
        axes[0].plot(xs * 100.0, density, color=colour, linewidth=2.0, label=label)
        axes[0].fill_between(xs * 100.0, density, color=colour, alpha=0.18)
    axes[0].set_xlabel("day-7 retention rate (%)")
    axes[0].set_ylabel("posterior density")
    axes[0].set_title("Posterior for each arm", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    _style(axes[0])

    draws = bayes.difference_samples(
        successes_control, n_control, successes_treatment, n_treatment, comparison.draws, seed
    )
    counts, edges, patches = axes[1].hist(draws, bins=120, color=PALETTE["accent"], alpha=0.35)
    centres = (edges[:-1] + edges[1:]) / 2.0
    for patch, centre in zip(patches, centres, strict=True):
        if comparison.hdi_low_pp <= centre <= comparison.hdi_high_pp:
            patch.set_alpha(0.85)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.3, label="no effect")
    axes[1].set_xlabel("difference in day-7 retention (percentage points)")
    axes[1].set_ylabel("posterior draws")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_title(
        f"95% HDI {comparison.hdi_low_pp:+.2f} to {comparison.hdi_high_pp:+.2f}pp; "
        f"P(better) = {comparison.prob_treatment_better * 100:.2f}%",
        fontsize=11,
    )
    _style(axes[1])
    return _save(figure, path)


def peeking_panel(study: Any, readout: Any, path: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    names = [
        "one look\nat the end",
        f"{study.looks} looks\nno correction",
        f"{study.looks} looks\nBonferroni",
        f"{study.looks} looks\ncalibrated z={study.calibrated_boundary_z:.2f}",
    ]
    values = [
        study.fixed_horizon_false_positive_rate * 100.0,
        study.naive_peeking_false_positive_rate * 100.0,
        study.bonferroni_false_positive_rate * 100.0,
        study.calibrated_false_positive_rate * 100.0,
    ]
    colours = [PALETTE["control"], PALETTE["warn"], PALETTE["accent"], PALETTE["accent"]]
    bars = axes[0].bar(names, values, color=colours, width=0.6)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.4,
            f"{value:.1f}%",
            ha="center",
            fontsize=10,
        )
    axes[0].axhline(
        study.alpha * 100.0,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=f"target {study.alpha * 100:.0f}%",
    )
    axes[0].set_ylabel("false positives on A/A splits (%)")
    axes[0].set_title(
        f"{study.validation_replications:,} held-out A/A splits of the control arm", fontsize=11
    )
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].tick_params(axis="x", labelsize=8)
    _style(axes[0])

    axes[1].plot(
        readout.sample_sizes,
        readout.z_values,
        marker="o",
        color=PALETTE["treatment"],
        linewidth=1.8,
    )
    for level, colour, label in (
        (readout.naive_boundary_z, PALETTE["warn"], "uncorrected 1.96"),
        (readout.boundary_z, PALETTE["accent"], f"calibrated {readout.boundary_z:.2f}"),
    ):
        for sign in (1, -1):
            axes[1].axhline(sign * level, color=colour, linestyle="--", linewidth=1.2)
        axes[1].plot([], [], color=colour, linestyle="--", label=label)
    axes[1].set_xlabel("players observed per arm")
    axes[1].set_ylabel("z statistic on day-7 retention")
    axes[1].set_title("The real experiment, read look by look", fontsize=11)
    axes[1].legend(frameon=False, fontsize=9)
    _style(axes[1])
    return _save(figure, path)


def bandit_curves(
    checkpoints: np.ndarray, curves: dict[str, np.ndarray], results: dict[str, Any], path: Path
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    colours = {
        "uniform": PALETTE["warn"],
        "epsilon_greedy": "#8a6d3b",
        "ucb1": PALETTE["control"],
        "thompson": PALETTE["accent"],
    }
    for policy, curve in curves.items():
        axes[0].plot(
            checkpoints,
            curve,
            label=policy.replace("_", " "),
            color=colours.get(policy, "grey"),
            linewidth=1.9,
        )
    axes[0].set_xlabel("players allocated")
    axes[0].set_ylabel("cumulative regret (retained players forgone)")
    axes[0].set_title("Regret against always serving the better build", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    _style(axes[0])

    policies = list(results)
    exposure = [results[policy].mean_pulls_inferior_arm for policy in policies]
    bars = axes[1].bar(
        [policy.replace("_", "\n") for policy in policies],
        exposure,
        color=[colours.get(policy, "grey") for policy in policies],
        width=0.6,
    )
    for bar, value in zip(bars, exposure, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(exposure) * 0.015,
            f"{value:,.0f}",
            ha="center",
            fontsize=10,
        )
    axes[1].set_ylabel("players served the worse build")
    axes[1].set_title(
        f"Exposure over {results[policies[0]].horizon:,} arrivals, "
        f"mean of {results[policies[0]].seeds} seeds",
        fontsize=11,
    )
    _style(axes[1])
    return _save(figure, path)


def power_curve(
    relative_lifts: np.ndarray, sizes: np.ndarray, n_actual: int, report: Any, path: Path
) -> Path:
    figure, axis = plt.subplots(figsize=(8.4, 4.4))
    axis.plot(relative_lifts * 100.0, sizes, color=PALETTE["control"], linewidth=2.0)
    axis.axhline(n_actual, color=PALETTE["treatment"], linestyle="--", linewidth=1.4,
                 label=f"players per arm in this test ({n_actual:,})")
    axis.axvline(
        report.mde_relative_pct,
        color=PALETTE["accent"],
        linestyle="--",
        linewidth=1.4,
        label=f"detectable at 80% power ({report.mde_relative_pct:.1f}% relative)",
    )
    axis.set_yscale("log")
    axis.set_xlabel("relative change in day-7 retention to detect (%)")
    axis.set_ylabel("players needed per arm (log scale)")
    axis.set_title("What this sample size could and could not have found", fontsize=11)
    axis.legend(frameon=False, fontsize=9)
    _style(axis)
    return _save(figure, path)


def engagement_distribution(
    control: np.ndarray, treatment: np.ndarray, control_label: str, treatment_label: str,
    comparison: Any, path: Path
) -> Path:
    figure, axis = plt.subplots(figsize=(8.6, 4.3))
    bins = np.logspace(0, np.log10(max(control.max(), treatment.max()) + 1), 60)
    axis.hist(control + 1, bins=bins, alpha=0.6, label=control_label, color=PALETTE["control"])
    axis.hist(
        treatment + 1, bins=bins, alpha=0.6, label=treatment_label, color=PALETTE["treatment"]
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("game rounds played in 14 days (plus one, log scale)")
    axis.set_ylabel("players (log scale)")
    axis.set_title(
        f"Engagement is heavy tailed: medians {comparison.median_control:.0f} and "
        f"{comparison.median_treatment:.0f}, rank test p={comparison.mann_whitney_p_value:.3f}",
        fontsize=11,
    )
    axis.legend(frameon=False, fontsize=9)
    _style(axis)
    return _save(figure, path)


def placebo_panel(report: Any, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(8.6, 4.4))
    spreads = np.asarray(report.spreads_pp, dtype="float64")
    limit = max(np.abs(spreads).max(), abs(report.observed_spread_pp)) * 1.2
    axis.hist(
        spreads,
        bins=max(len(spreads) // 3, 8),
        range=(-limit, limit),
        color=PALETTE["control"],
        alpha=0.75,
        edgecolor="white",
        label=f"{report.placebo_runs} placebo splits of the control arm",
    )
    axis.plot(
        spreads, np.full_like(spreads, -0.25), "|", color=PALETTE["control"], markersize=12
    )
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.1)
    axis.axvline(
        report.observed_spread_pp,
        color=PALETTE["warn"],
        linewidth=2.6,
        label=f"real experiment {report.observed_spread_pp:+.2f}pp",
    )
    axis.set_xlabel("top decile minus bottom decile uplift (percentage points)")
    axis.set_ylabel("placebo splits")
    axis.set_title(
        f"Targeting placebo test over {report.placebo_runs} splits, "
        f"permutation p={report.permutation_p_value:.2f}",
        fontsize=11,
    )
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    _style(axis)
    return _save(figure, path)
