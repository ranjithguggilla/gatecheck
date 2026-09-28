from __future__ import annotations

import json

import pytest

from gatecheck import app, pipeline
from gatecheck.decision import MetricEvidence


@pytest.fixture(scope="module")
def artefacts_and_config(tmp_path_factory):
    """One end to end run on the fixture data, shared by the tests below."""
    import dataclasses

    import pandas as pd

    from gatecheck.config import Config, Paths

    from .conftest import make_players

    root = tmp_path_factory.mktemp("run")
    raw = root / "data" / "raw" / "cookie_cats.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    frame: pd.DataFrame = make_players(n_control=3_000, n_treatment=3_000, seed=77)
    frame.to_csv(raw, index=False)
    config = dataclasses.replace(
        Config(),
        paths=Paths(
            root=root,
            raw_csv=raw,
            interim_parquet=root / "data" / "interim" / "players.parquet",
            assets=root / "assets",
            results=root / "results",
        ),
        bootstrap_draws=800,
        posterior_draws=8_000,
        aa_replications=40,
        interim_looks=4,
        bandit_seeds=3,
        bandit_horizon=500,
        placebo_runs=2,
        leakage_folds=3,
    )
    return pipeline.run(config, make_figures=True), config


def test_the_run_reports_every_section(artefacts_and_config):
    artefacts, _ = artefacts_and_config
    expected = {
        "dataset",
        "validity",
        "frequentist",
        "engagement",
        "multiplicity",
        "bootstrap",
        "bayes",
        "power",
        "peeking",
        "bandits",
        "targeting",
        "decision",
        "environment",
    }
    assert expected <= set(artefacts.metrics)


def test_the_run_writes_every_figure(artefacts_and_config):
    artefacts, _ = artefacts_and_config
    assert len(artefacts.figures) == 8
    for path in artefacts.figures.values():
        assert path.exists()
        assert path.stat().st_size > 5_000


def test_metrics_json_round_trips(artefacts_and_config):
    artefacts, config = artefacts_and_config
    path = artefacts.write(config.paths.results)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["dataset"]["rows"] == artefacts.metrics["dataset"]["rows"]


def test_the_verdict_is_one_of_the_three_words(artefacts_and_config):
    artefacts, _ = artefacts_and_config
    verdict = artefacts.metrics["decision"]["verdict"]
    assert verdict["recommendation"] in {"ship", "hold", "roll back"}
    assert verdict["reasons"]


def test_arm_sizes_add_up(artefacts_and_config):
    artefacts, _ = artefacts_and_config
    dataset = artefacts.metrics["dataset"]
    assert dataset["n_control"] + dataset["n_treatment"] == dataset["rows"]


def test_the_dashboard_reads_a_completed_run(artefacts_and_config):
    artefacts, config = artefacts_and_config
    artefacts.write(config.paths.results)
    metrics = app.load_metrics(config.paths.results / "metrics.json")
    evidence = app.evidence_from_metrics(metrics)
    assert isinstance(evidence, MetricEvidence)
    assert evidence.metric == "retention_7"
    assert 0.0 <= evidence.posterior_prob_treatment_better <= 1.0


def test_the_dashboard_widens_the_interval_when_alpha_tightens(artefacts_and_config):
    artefacts, config = artefacts_and_config
    artefacts.write(config.paths.results)
    metrics = app.load_metrics(config.paths.results / "metrics.json")
    loose = app.evidence_from_metrics(metrics, alpha=0.10)
    strict = app.evidence_from_metrics(metrics, alpha=0.01)
    assert strict.ci_low_pp < loose.ci_low_pp
    assert strict.ci_high_pp > loose.ci_high_pp


def test_the_dashboard_says_so_when_there_is_no_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="gatecheck run"):
        app.load_metrics(tmp_path / "missing.json")


def test_skipping_figures_leaves_the_numbers_intact(fast_config):
    artefacts = pipeline.run(fast_config, make_figures=False)
    assert artefacts.figures == {}
    assert "figures" not in artefacts.metrics
    assert artefacts.metrics["dataset"]["rows"] == 2_400
