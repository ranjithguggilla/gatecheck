from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gatecheck.config import Config, Paths


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def make_players(
    n_control: int = 1_200,
    n_treatment: int = 1_200,
    rate_control: float = 0.20,
    rate_treatment: float = 0.16,
    seed: int = 11,
) -> pd.DataFrame:
    """A small stand-in with the same schema as the real file."""
    generator = np.random.default_rng(seed)
    frames = []
    for arm, size, rate in (
        ("gate_30", n_control, rate_control),
        ("gate_40", n_treatment, rate_treatment),
    ):
        retention_7 = generator.random(size) < rate
        frames.append(
            pd.DataFrame(
                {
                    "userid": np.arange(len(frames) * 10**6, len(frames) * 10**6 + size),
                    "version": pd.array([arm] * size, dtype="string"),
                    "sum_gamerounds": generator.integers(0, 400, size=size),
                    "retention_1": generator.random(size) < rate + 0.25,
                    "retention_7": retention_7,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def players() -> pd.DataFrame:
    return make_players()


@pytest.fixture
def fast_config(tmp_path: Path, players: pd.DataFrame) -> Config:
    """A config wired to a temporary directory and small simulation budgets."""
    raw = tmp_path / "data" / "raw" / "cookie_cats.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    players.to_csv(raw, index=False)
    paths = Paths(
        root=tmp_path,
        raw_csv=raw,
        interim_parquet=tmp_path / "data" / "interim" / "players.parquet",
        assets=tmp_path / "assets",
        results=tmp_path / "results",
    )
    return dataclasses.replace(
        Config(),
        paths=paths,
        bootstrap_draws=500,
        posterior_draws=5_000,
        aa_replications=40,
        interim_looks=4,
        bandit_seeds=4,
        bandit_horizon=400,
        placebo_runs=2,
        leakage_folds=3,
    )
