"""Checks that run before anybody is allowed to look at the result."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SrmResult:
    """Sample ratio mismatch: did randomisation deal the hands it promised?"""

    n_control: int
    n_treatment: int
    expected_share_treatment: float
    observed_share_treatment: float
    chi_square: float
    p_value: float
    alarm_threshold: float
    triggered: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def srm_check(
    n_control: int,
    n_treatment: int,
    expected_share_treatment: float = 0.5,
    alarm_threshold: float = 0.0005,
) -> SrmResult:
    """One degree of freedom chi-square goodness of fit on the split itself.

    The alarm threshold is far stricter than a normal 0.05 because with millions
    of assignments a healthy split still wanders, and every team that uses 0.05
    here ends up chasing ghosts.
    """
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("both arms need at least one unit")
    if not 0.0 < expected_share_treatment < 1.0:
        raise ValueError("expected_share_treatment must sit in (0, 1)")
    total = n_control + n_treatment
    expected_treatment = total * expected_share_treatment
    expected_control = total - expected_treatment
    chi_square = float(
        (n_treatment - expected_treatment) ** 2 / expected_treatment
        + (n_control - expected_control) ** 2 / expected_control
    )
    p_value = float(stats.chi2.sf(chi_square, df=1))
    return SrmResult(
        n_control=int(n_control),
        n_treatment=int(n_treatment),
        expected_share_treatment=float(expected_share_treatment),
        observed_share_treatment=float(n_treatment / total),
        chi_square=chi_square,
        p_value=p_value,
        alarm_threshold=float(alarm_threshold),
        triggered=bool(p_value < alarm_threshold),
    )


@dataclass(frozen=True)
class DuplicateResult:
    n_rows: int
    n_unique_ids: int
    n_duplicate_ids: int
    n_ids_in_both_arms: int

    @property
    def clean(self) -> bool:
        return self.n_duplicate_ids == 0 and self.n_ids_in_both_arms == 0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "clean": self.clean}


def duplicate_check(frame: pd.DataFrame) -> DuplicateResult:
    """A player counted twice, or assigned to both arms, poisons everything downstream."""
    n_rows = len(frame)
    n_unique = int(frame["userid"].nunique())
    arms_per_id = frame.groupby("userid")["version"].nunique()
    return DuplicateResult(
        n_rows=n_rows,
        n_unique_ids=n_unique,
        n_duplicate_ids=n_rows - n_unique,
        n_ids_in_both_arms=int((arms_per_id > 1).sum()),
    )


@dataclass(frozen=True)
class OutlierAudit:
    metric: str
    maximum: float
    p99: float
    p999: float
    ratio_max_to_p99: float
    n_above_p999: int
    winsor_cut: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def outlier_audit(values: pd.Series, winsor_quantile: float = 0.999) -> OutlierAudit:
    """Describe the top tail so the engagement test is not one player's story."""
    numeric = values.astype("float64")
    p99 = float(numeric.quantile(0.99))
    p999 = float(numeric.quantile(0.999))
    maximum = float(numeric.max())
    return OutlierAudit(
        metric=str(values.name),
        maximum=maximum,
        p99=p99,
        p999=p999,
        ratio_max_to_p99=float(maximum / p99) if p99 > 0 else float("inf"),
        n_above_p999=int((numeric > p999).sum()),
        winsor_cut=float(numeric.quantile(winsor_quantile)),
    )


@dataclass(frozen=True)
class BalanceCheck:
    """A pre-treatment covariate would go here. This dataset has none, and that matters."""

    pre_treatment_columns: tuple[str, ...]
    post_treatment_columns: tuple[str, ...]
    can_test_covariate_balance: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def balance_check(frame: pd.DataFrame) -> BalanceCheck:
    """Classify every column by whether it was known before assignment.

    ``userid`` is the only column that existed before the gate moved. Everything
    else was measured in the fourteen days after, so there is nothing to test
    balance on and nothing legitimate to condition a targeting model on.
    """
    pre = tuple(column for column in frame.columns if column in {"userid"})
    post = tuple(
        column for column in frame.columns if column not in {"userid", "version"}
    )
    return BalanceCheck(
        pre_treatment_columns=pre,
        post_treatment_columns=post,
        can_test_covariate_balance=False,
        note=(
            "userid is the only pre-assignment column, so covariate balance cannot be "
            "tested and no subgroup split in this dataset is causally interpretable."
        ),
    )


def gate_summary(
    srm: SrmResult, duplicates: DuplicateResult, audit: OutlierAudit
) -> dict[str, Any]:
    """Roll the gates up into a pass/fail plus the reasons, in plain words."""
    reasons: list[str] = []
    blocking: str | None = None
    if srm.triggered:
        reasons.append(
            f"sample ratio mismatch: split is {srm.observed_share_treatment:.4f} "
            f"treatment against {srm.expected_share_treatment:.2f} expected "
            f"(p={srm.p_value:.2e})"
        )
        blocking = reasons[-1]
    elif srm.p_value < 0.05:
        reasons.append(
            f"split is mildly uneven (p={srm.p_value:.4f}) but stays above the "
            f"{srm.alarm_threshold:g} alarm threshold, so the read is allowed to stand"
        )
    else:
        reasons.append(f"split looks like a fair coin (p={srm.p_value:.3f})")
    if not duplicates.clean:
        reasons.append(
            f"{duplicates.n_duplicate_ids} duplicated ids and "
            f"{duplicates.n_ids_in_both_arms} ids in both arms"
        )
        blocking = blocking or reasons[-1]
    else:
        reasons.append(f"all {duplicates.n_unique_ids} player ids appear exactly once, in one arm")
    reasons.append(
        f"{audit.metric} tops out at {audit.maximum:,.0f} against a 99th percentile of "
        f"{audit.p99:,.0f} ({audit.ratio_max_to_p99:.0f}x), so the engagement test is run "
        f"winsorised at {audit.winsor_cut:,.0f} and with a rank test alongside it"
    )
    passed = (not srm.triggered) and duplicates.clean
    return {
        "passed": bool(passed),
        "blocking_reason": blocking,
        "reasons": reasons,
        "srm": srm.as_dict(),
        "duplicates": duplicates.as_dict(),
        "engagement_outliers": audit.as_dict(),
    }


def assignment_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of the arm split in bits, a one-number health read."""
    counts = np.asarray(counts, dtype="float64")
    if counts.sum() <= 0:
        raise ValueError("counts must sum to a positive number")
    shares = counts / counts.sum()
    shares = shares[shares > 0]
    return float(-(shares * np.log2(shares)).sum())
