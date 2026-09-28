"""Whether this experiment supports a targeting model, tested rather than assumed."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LEAKY_FEATURES = ("retention_1", "log_gamerounds")
HONEST_FEATURES = ("retention_1",)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the two candidate features, both measured after assignment."""
    return pd.DataFrame(
        {
            "retention_1": frame["retention_1"].astype("float64"),
            "log_gamerounds": np.log1p(frame["sum_gamerounds"].astype("float64")),
        }
    )


def _pipeline(seed: int) -> Pipeline:
    """Scaler and model in one object so the fold boundary is never crossed."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )


@dataclass(frozen=True)
class LeakageReport:
    n_rows: int
    base_rate: float
    auc_with_engagement: float
    auc_with_engagement_std: float
    auc_day_one_only: float
    auc_day_one_only_std: float
    auc_majority_baseline: float
    leak_contribution_auc: float
    note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def leakage_report(frame: pd.DataFrame, seed: int, folds: int = 5) -> LeakageReport:
    """Measure how much apparent skill comes from a feature that spans the outcome window.

    ``sum_gamerounds`` counts every round played in the fourteen days after install,
    so it already contains day 8 to day 14 activity. Predicting day-7 retention from
    it scores well and is worthless: at the moment a decision would be made, the
    number does not exist yet. Putting a figure on that gap is more persuasive than
    a warning in a docstring.
    """
    features = build_features(frame)
    target = frame["retention_7"].astype("int64").to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    leaky = cross_val_score(
        _pipeline(seed),
        features.loc[:, list(LEAKY_FEATURES)],
        target,
        cv=splitter,
        scoring="roc_auc",
    )
    honest = cross_val_score(
        _pipeline(seed),
        features.loc[:, list(HONEST_FEATURES)],
        target,
        cv=splitter,
        scoring="roc_auc",
    )
    return LeakageReport(
        n_rows=int(len(frame)),
        base_rate=float(target.mean()),
        auc_with_engagement=float(leaky.mean()),
        auc_with_engagement_std=float(leaky.std(ddof=1)),
        auc_day_one_only=float(honest.mean()),
        auc_day_one_only_std=float(honest.std(ddof=1)),
        auc_majority_baseline=0.5,
        leak_contribution_auc=float(leaky.mean() - honest.mean()),
        note=(
            "The engagement feature spans days 1 to 14, so it overlaps the day-7 outcome "
            "window. The gap between the two rows is the size of the leak, not a modelling win."
        ),
    )


def _t_learner_scores(
    features: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    train_index: np.ndarray,
    score_index: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Two models, one per arm, differenced on held-out rows."""
    scores = np.zeros(len(score_index), dtype="float64")
    for arm, sign in ((1, 1.0), (0, -1.0)):
        rows = train_index[treatment[train_index] == arm]
        if len(np.unique(outcome[rows])) < 2:
            # One arm produced a single outcome value, so a classifier cannot be
            # fitted. Its predicted probability is that constant, which is the
            # honest contribution rather than a silently dropped term.
            scores += sign * float(outcome[rows].mean())
            continue
        model = _pipeline(seed).fit(features[rows], outcome[rows])
        scores += sign * model.predict_proba(features[score_index])[:, 1]
    return scores


def decile_uplift_spread(
    features: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    seed: int,
    folds: int = 4,
) -> float:
    """Top decile minus bottom decile of realised uplift, out of fold.

    Scores are produced on rows the two arm models never saw, then the actual
    treatment effect inside the top and bottom deciles of that score is measured.
    A model with real signal separates them; one fitting noise does not.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = np.zeros(len(outcome), dtype="float64")
    strata = treatment * 2 + outcome
    for train_index, score_index in splitter.split(features, strata):
        scores[score_index] = _t_learner_scores(
            features, treatment, outcome, train_index, score_index, seed
        )
    order = np.argsort(scores)
    cut = max(len(order) // 10, 1)
    top, bottom = order[-cut:], order[:cut]
    return float(_group_uplift(treatment, outcome, top) - _group_uplift(treatment, outcome, bottom))


def _group_uplift(treatment: np.ndarray, outcome: np.ndarray, rows: np.ndarray) -> float:
    treated = outcome[rows][treatment[rows] == 1]
    control = outcome[rows][treatment[rows] == 0]
    if len(treated) == 0 or len(control) == 0:
        return 0.0
    return float(treated.mean() - control.mean())


@dataclass(frozen=True)
class PlaceboReport:
    observed_spread_pp: float
    placebo_runs: int
    placebo_mean_pp: float
    placebo_std_pp: float
    placebo_abs_p95_pp: float
    placebo_max_abs_pp: float
    permutation_p_value: float
    verdict: str
    spreads_pp: list[float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def placebo_test(
    frame: pd.DataFrame,
    control_arm: str,
    treatment_arm: str,
    outcome_column: str,
    seed: int,
    placebo_runs: int = 30,
) -> PlaceboReport:
    """Compare the real uplift spread against spreads found where none can exist.

    The placebo runs split the control arm into two pseudo-arms. Every player in
    them saw the same build, so any spread the model finds there is the procedure
    talking to itself. If the real number sits inside that distribution, the
    targeting model has found nothing.
    """
    features_all = build_features(frame).to_numpy()
    treatment_all = (frame["version"] == treatment_arm).astype("int64").to_numpy()
    outcome_all = frame[outcome_column].astype("int64").to_numpy()
    observed = decile_uplift_spread(features_all, treatment_all, outcome_all, seed)

    control_frame = frame.loc[frame["version"] == control_arm].reset_index(drop=True)
    control_features = build_features(control_frame).to_numpy()
    control_outcome = control_frame[outcome_column].astype("int64").to_numpy()
    rng = np.random.default_rng(seed)
    spreads = np.empty(placebo_runs, dtype="float64")
    for index in range(placebo_runs):
        pseudo = rng.integers(0, 2, size=len(control_outcome))
        spreads[index] = decile_uplift_spread(
            control_features, pseudo, control_outcome, seed + index + 1
        )
    # The observed statistic belongs in its own null set, so the count and the
    # denominator both carry a plus one. Without it the smallest reachable p value
    # is zero, which no permutation test can honestly claim.
    exceed = int((np.abs(spreads) >= abs(observed)).sum())
    p_value = float((1 + exceed) / (1 + placebo_runs))
    verdict = (
        "no usable targeting signal: the real spread is inside the placebo distribution"
        if p_value > 0.05
        else "the real spread exceeds the placebo distribution"
    )
    return PlaceboReport(
        observed_spread_pp=float(observed * 100.0),
        placebo_runs=int(placebo_runs),
        placebo_mean_pp=float(spreads.mean() * 100.0),
        placebo_std_pp=float(spreads.std(ddof=1) * 100.0),
        placebo_abs_p95_pp=float(np.percentile(np.abs(spreads), 95) * 100.0),
        placebo_max_abs_pp=float(np.abs(spreads).max() * 100.0),
        permutation_p_value=p_value,
        verdict=verdict,
        spreads_pp=[float(value * 100.0) for value in spreads],
    )
