"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import data, pipeline, validity
from .config import Config

app = typer.Typer(add_completion=False, help="Experiment validity, inference and decision tooling.")
console = Console()


def _config(seed: int | None) -> Config:
    config = Config()
    return config if seed is None else config.with_seed(seed)


@app.command("validate")
def validate(seed: int = typer.Option(None, help="Override the random seed.")) -> None:
    """Run the validity gates only, and stop before any inference."""
    config = _config(seed)
    players = data.load_players(config.paths.raw_csv, config.paths.interim_parquet)
    split = data.split_arms(players, config.control_arm, config.treatment_arm)
    gates = validity.gate_summary(
        validity.srm_check(split.n_control, split.n_treatment, 0.5, config.rules.srm_alarm_p),
        validity.duplicate_check(players),
        validity.outlier_audit(
            players[config.engagement_metric], config.engagement_winsor_quantile
        ),
    )
    console.print(
        Panel(
            "\n".join(f"- {reason}" for reason in gates["reasons"]),
            title=f"validity gates: {'pass' if gates['passed'] else 'fail'}",
        )
    )
    raise typer.Exit(code=0 if gates["passed"] else 1)


@app.command("run")
def run(
    seed: int = typer.Option(None, help="Override the random seed."),
    skip_figures: bool = typer.Option(False, help="Skip writing chart files."),
) -> None:
    """Run the full read and write results/metrics.json plus the figures."""
    config = _config(seed)
    artefacts = pipeline.run(config, make_figures=not skip_figures)
    path = artefacts.write(config.paths.results)
    _print_summary(artefacts.metrics)
    console.print(f"[green]metrics written to[/green] {path}")


@app.command("report")
def report(
    metrics_path: Path = typer.Option(
        None, help="Read an existing metrics.json instead of re-running."
    ),
) -> None:
    """Print the headline numbers from a completed run."""
    config = Config()
    path = metrics_path or config.paths.results / "metrics.json"
    if not path.exists():
        console.print(f"[red]no metrics at {path}; run `gatecheck run` first[/red]")
        raise typer.Exit(code=1)
    _print_summary(json.loads(path.read_text(encoding="utf-8")))


def _print_summary(metrics: dict) -> None:
    primary = metrics["dataset"]["primary_metric"]
    table = Table(title="headline numbers", show_lines=False)
    table.add_column("what")
    table.add_column("value", justify="right")
    freq = metrics["frequentist"][primary]
    table.add_row("players", f"{metrics['dataset']['rows']:,}")
    table.add_row(f"{primary} control", f"{freq['rate_control'] * 100:.2f}%")
    table.add_row(f"{primary} treatment", f"{freq['rate_treatment'] * 100:.2f}%")
    table.add_row("absolute difference", f"{freq['absolute_diff_pp']:+.2f}pp")
    table.add_row(
        "95% interval",
        f"{freq['diff_ci_low_pp']:+.2f} to {freq['diff_ci_high_pp']:+.2f}pp",
    )
    table.add_row("p value", f"{freq['p_value']:.4g}")
    table.add_row("P(change helps)", f"{metrics['bayes']['prob_treatment_better'] * 100:.2f}%")
    table.add_row(
        "naive peeking false positives",
        f"{metrics['peeking']['study']['naive_peeking_false_positive_rate'] * 100:.1f}%",
    )
    table.add_row(
        "annual revenue effect",
        f"${metrics['decision']['financial_impact']['annual_revenue_delta_usd']:,.0f}",
    )
    console.print(table)
    verdict = metrics["decision"]["verdict"]
    console.print(
        Panel(
            "\n".join(f"- {reason}" for reason in verdict["reasons"]),
            title=(
                f"recommendation: {verdict['recommendation']} "
                f"({verdict['confidence']} confidence)"
            ),
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
