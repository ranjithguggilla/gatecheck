"""One entry point that runs the whole read and writes the artefacts."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels

from . import (
    bandits,
    bayes,
    charts,
    data,
    frequentist,
    peeking,
    power,
    resampling,
    targeting,
    validity,
)
from .config import Config
from .decision import MetricEvidence, evaluate_experiment, financial_impact


@dataclass
class RunArtefacts:
    metrics: dict[str, Any]
    figures: dict[str, Path] = field(default_factory=dict)

    def write(self, results_dir: Path) -> Path:
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / "metrics.json"
        path.write_text(json.dumps(self.metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def _successes(frame: pd.DataFrame, metric: str) -> int:
    return int(frame[metric].sum())


def run(config: Config | None = None, make_figures: bool = True) -> RunArtefacts:
    """Run validity gates, inference, simulations and the verdict, in that order."""
    config = config or Config()
    config.paths.ensure()

    players = data.load_players(config.paths.raw_csv, config.paths.interim_parquet)
    split = data.split_arms(players, config.control_arm, config.treatment_arm)

    srm = validity.srm_check(
        split.n_control, split.n_treatment, 0.5, config.rules.srm_alarm_p
    )
    duplicates = validity.duplicate_check(players)
    audit = validity.outlier_audit(
        players[config.engagement_metric], config.engagement_winsor_quantile
    )
    gates = validity.gate_summary(srm, duplicates, audit)
    gates["covariate_balance"] = validity.balance_check(players).as_dict()
    gates["assignment_entropy_bits"] = validity.assignment_entropy(
        np.array([split.n_control, split.n_treatment])
    )

    metric_names = (config.primary_metric, *config.secondary_metrics)
    comparisons = {
        metric: frequentist.two_proportion_test(
            metric,
            _successes(split.control, metric),
            split.n_control,
            _successes(split.treatment, metric),
            split.n_treatment,
            config.rules.alpha,
        )
        for metric in metric_names
    }
    engagement = frequentist.compare_continuous(
        config.engagement_metric,
        split.control[config.engagement_metric].to_numpy(),
        split.treatment[config.engagement_metric].to_numpy(),
        audit.winsor_cut,
    )
    trimmed_point, trimmed_low, trimmed_high = resampling.bootstrap_trimmed_mean_difference(
        split.control[config.engagement_metric].to_numpy(),
        split.treatment[config.engagement_metric].to_numpy(),
        draws=config.trimmed_bootstrap_draws,
        seed=config.random_seed,
    )

    family_labels = [*metric_names, config.engagement_metric]
    family_p_values = [comparisons[metric].p_value for metric in metric_names] + [
        engagement.mann_whitney_p_value
    ]
    multiplicity = frequentist.adjust_p_values(family_labels, family_p_values, config.rules.alpha)

    primary = comparisons[config.primary_metric]
    bootstrap = resampling.bootstrap_proportion_difference(
        config.primary_metric,
        primary.successes_control,
        primary.n_control,
        primary.successes_treatment,
        primary.n_treatment,
        config.bootstrap_draws,
        config.rules.practical_floor_pp,
        config.random_seed,
    )
    posterior = bayes.posterior_comparison(
        config.primary_metric,
        primary.successes_control,
        primary.n_control,
        primary.successes_treatment,
        primary.n_treatment,
        config.posterior_draws,
        config.rules.practical_floor_pp,
        config.random_seed,
    )
    power_summary = power.power_report(
        config.primary_metric,
        primary.rate_control,
        min(primary.n_control, primary.n_treatment),
        primary.rate_treatment - primary.rate_control,
        config.rules.alpha,
    )

    control_primary = split.control[config.primary_metric].to_numpy(dtype="float64")
    treatment_primary = split.treatment[config.primary_metric].to_numpy(dtype="float64")
    peeking_summary = peeking.peeking_study(
        config.primary_metric,
        control_primary,
        config.aa_replications,
        config.interim_looks,
        config.rules.alpha,
        config.random_seed,
    )
    readout = peeking.sequential_readout(
        config.primary_metric,
        control_primary,
        treatment_primary,
        config.interim_looks,
        config.rules.alpha,
        peeking_summary.calibrated_boundary_z,
        config.random_seed + 101,
    )

    bandit_results, checkpoints, curves = bandits.compare_policies(
        (primary.rate_control, primary.rate_treatment),
        config.bandit_horizon,
        config.bandit_seeds,
        config.random_seed,
    )
    savings = {
        policy: bandits.players_spared(bandit_results["uniform"], result)
        for policy, result in bandit_results.items()
        if policy != "uniform"
    }

    leakage = targeting.leakage_report(players, config.random_seed, config.leakage_folds)
    placebo = targeting.placebo_test(
        players,
        config.control_arm,
        config.treatment_arm,
        config.primary_metric,
        config.random_seed,
        config.placebo_runs,
    )

    evidence = MetricEvidence(
        metric=config.primary_metric,
        absolute_diff_pp=primary.absolute_diff_pp,
        ci_low_pp=primary.diff_ci_low_pp,
        ci_high_pp=primary.diff_ci_high_pp,
        p_value=primary.p_value,
        posterior_prob_treatment_better=posterior.prob_treatment_better,
        prob_worse_than_floor=posterior.prob_worse_than_floor,
    )
    verdict = evaluate_experiment(
        evidence, config.rules, gates["passed"], gates["blocking_reason"]
    )
    money = financial_impact(
        primary.absolute_diff_pp, primary.diff_ci_low_pp, primary.diff_ci_high_pp, config.business
    )

    metrics: dict[str, Any] = {
        "dataset": {
            "rows": int(len(players)),
            "control_arm": config.control_arm,
            "treatment_arm": config.treatment_arm,
            "n_control": split.n_control,
            "n_treatment": split.n_treatment,
            "primary_metric": config.primary_metric,
        },
        "validity": gates,
        "frequentist": {metric: comparison.as_dict() for metric, comparison in comparisons.items()},
        "engagement": {
            **engagement.as_dict(),
            "trimmed_mean_difference": trimmed_point,
            "trimmed_mean_ci_low": trimmed_low,
            "trimmed_mean_ci_high": trimmed_high,
        },
        "multiplicity": multiplicity,
        "bootstrap": bootstrap.as_dict(),
        "bayes": posterior.as_dict(),
        "power": power_summary.as_dict(),
        "peeking": {"study": peeking_summary.as_dict(), "sequential_readout": readout.as_dict()},
        "bandits": {
            "policies": {policy: result.as_dict() for policy, result in bandit_results.items()},
            "savings_against_uniform": savings,
        },
        "targeting": {"leakage": leakage.as_dict(), "placebo": placebo.as_dict()},
        "decision": {"verdict": verdict.as_dict(), "financial_impact": money.as_dict()},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "arviz": az.__version__,
            "seed": config.random_seed,
        },
    }

    artefacts = RunArtefacts(metrics=metrics)
    if make_figures:
        assets = config.paths.assets
        ordered = [config.primary_metric, *config.secondary_metrics]
        artefacts.figures["retention"] = charts.retention_bars(
            [comparisons[name] for name in ordered],
            config.control_arm,
            config.treatment_arm,
            assets / "retention_by_arm.png",
        )
        artefacts.figures["bootstrap"] = charts.bootstrap_histogram(
            resampling.bootstrap_difference_samples(
                primary.successes_control,
                primary.n_control,
                primary.successes_treatment,
                primary.n_treatment,
                config.bootstrap_draws,
                config.random_seed,
            ),
            bootstrap,
            assets / "bootstrap_difference.png",
        )
        artefacts.figures["posterior"] = charts.posterior_panel(
            primary.successes_control,
            primary.n_control,
            primary.successes_treatment,
            primary.n_treatment,
            posterior,
            config.control_arm,
            config.treatment_arm,
            assets / "posterior_day7.png",
            config.random_seed,
        )
        artefacts.figures["peeking"] = charts.peeking_panel(
            peeking_summary, readout, assets / "peeking_cost.png"
        )
        artefacts.figures["bandits"] = charts.bandit_curves(
            checkpoints, curves, bandit_results, assets / "bandit_regret.png"
        )
        lifts = np.linspace(0.01, 0.20, 40)
        _, sizes = power.sample_size_curve(primary.rate_control, lifts, config.rules.alpha)
        artefacts.figures["power"] = charts.power_curve(
            lifts, sizes, min(split.n_control, split.n_treatment), power_summary,
            assets / "power_curve.png",
        )
        artefacts.figures["engagement"] = charts.engagement_distribution(
            split.control[config.engagement_metric].to_numpy(dtype="float64"),
            split.treatment[config.engagement_metric].to_numpy(dtype="float64"),
            config.control_arm,
            config.treatment_arm,
            engagement,
            assets / "engagement_distribution.png",
        )
        artefacts.figures["placebo"] = charts.placebo_panel(
            placebo, assets / "targeting_placebo.png"
        )
        metrics["figures"] = {name: path.name for name, path in artefacts.figures.items()}
    return artefacts
