"""Every knob the pipeline exposes, in one place."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BusinessAssumptions:
    """Inputs to the money side of the decision.

    None of these come from the dataset -- they are the numbers a product team
    would bring to the review, and they are deliberately easy to override so a
    reader can plug in their own and watch the verdict move.
    """

    monthly_new_players: int = 1_500_000
    arpdau_usd: float = 0.085
    days_monetised_per_retained_player: float = 21.0
    variant_rollout_share: float = 1.0

    def annual_new_players(self) -> float:
        return self.monthly_new_players * 12.0

    def revenue_per_retained_player(self) -> float:
        return self.arpdau_usd * self.days_monetised_per_retained_player


@dataclass(frozen=True)
class DecisionRules:
    """Thresholds the verdict is read off.

    ``practical_floor_pp`` is the amount of day-7 retention the team is willing
    to trade away for a change that is cheap to keep. It does not change a
    recommendation on its own: an inconclusive read stays a hold however bad the
    point estimate looks. What it does is separate a loss worth acting on from one
    that is merely real, which is the difference between high and moderate
    confidence on a roll back.
    """

    alpha: float = 0.05
    min_posterior_prob_to_ship: float = 0.95
    practical_floor_pp: float = -0.20
    srm_alarm_p: float = 0.0005


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    raw_csv: Path = REPO_ROOT / "data" / "raw" / "cookie_cats.csv"
    interim_parquet: Path = REPO_ROOT / "data" / "interim" / "players.parquet"
    assets: Path = REPO_ROOT / "assets"
    results: Path = REPO_ROOT / "results"

    def ensure(self) -> None:
        for directory in (self.interim_parquet.parent, self.assets, self.results):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    rules: DecisionRules = field(default_factory=DecisionRules)
    business: BusinessAssumptions = field(default_factory=BusinessAssumptions)

    control_arm: str = "gate_30"
    treatment_arm: str = "gate_40"
    primary_metric: str = "retention_7"
    secondary_metrics: tuple[str, ...] = ("retention_1",)
    engagement_metric: str = "sum_gamerounds"

    engagement_winsor_quantile: float = 0.999
    bootstrap_draws: int = 20_000
    posterior_draws: int = 400_000
    aa_replications: int = 2_000
    interim_looks: int = 10
    bandit_seeds: int = 200
    bandit_horizon: int = 60_000
    placebo_runs: int = 60
    leakage_folds: int = 5
    trimmed_bootstrap_draws: int = 2_000
    random_seed: int = 20_240_917

    def with_seed(self, seed: int) -> Config:
        return replace(self, random_seed=seed)
