"""The verdict, the reasons for it, and what it is worth in money."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import BusinessAssumptions, DecisionRules

SHIP = "ship"
HOLD = "hold"
ROLL_BACK = "roll back"


@dataclass(frozen=True)
class MetricEvidence:
    """The handful of numbers the verdict is allowed to depend on."""

    metric: str
    absolute_diff_pp: float
    ci_low_pp: float
    ci_high_pp: float
    p_value: float
    posterior_prob_treatment_better: float
    prob_worse_than_floor: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Verdict:
    recommendation: str
    confidence: str
    reasons: list[str]
    blocking_gate: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_experiment(
    primary: MetricEvidence,
    rules: DecisionRules,
    validity_passed: bool,
    validity_reason: str | None = None,
) -> Verdict:
    """Turn the evidence into one of three words, and say why in plain language.

    Kept free of file paths, plotting and configuration loading on purpose: this
    is the function the dashboard calls and the function the tests pin down, so it
    takes numbers in and gives a decision out.

    The practical floor never flips a recommendation by itself. An inconclusive read
    stays a hold however bad the point estimate looks, because a loss that cannot be
    distinguished from zero is not a loss anyone should act on. What the floor does
    is grade a roll back that is already justified: a whole interval below it earns
    high confidence, an interval straddling it earns moderate.
    """
    reasons: list[str] = []
    if not validity_passed:
        return Verdict(
            recommendation=HOLD,
            confidence="none",
            reasons=[
                "the validity gates did not pass, so the measured effect is not trustworthy yet",
                validity_reason or "see the validity section for the failing gate",
            ],
            blocking_gate="validity",
        )

    significant = primary.p_value < rules.alpha
    posterior_supports_ship = (
        primary.posterior_prob_treatment_better >= rules.min_posterior_prob_to_ship
    )
    worse_than_floor = primary.ci_high_pp < rules.practical_floor_pp
    direction = "down" if primary.absolute_diff_pp < 0 else "up"

    reasons.append(
        f"{primary.metric} moves {direction} by {abs(primary.absolute_diff_pp):.2f}pp "
        f"(95% interval {primary.ci_low_pp:+.2f}pp to {primary.ci_high_pp:+.2f}pp, "
        f"p={primary.p_value:.4g})"
    )
    reasons.append(
        f"the posterior puts the chance the change is an improvement at "
        f"{primary.posterior_prob_treatment_better * 100:.2f}%"
    )

    if significant and primary.absolute_diff_pp < 0:
        recommendation = ROLL_BACK
        confidence = "high" if worse_than_floor else "moderate"
        if worse_than_floor:
            reasons.append(
                f"the whole interval sits below the {rules.practical_floor_pp:+.2f}pp floor the "
                f"team agreed to tolerate, so the loss is both real and large enough to act on"
            )
        else:
            reasons.append(
                f"the loss is statistically clear but part of the interval is inside the "
                f"{rules.practical_floor_pp:+.2f}pp tolerance, so the size of the damage is the "
                f"open question, not its direction"
            )
        reasons.append(
            f"the posterior chance the change is worse than the floor is "
            f"{primary.prob_worse_than_floor * 100:.1f}%"
        )
    elif significant and posterior_supports_ship:
        recommendation = SHIP
        confidence = "high"
        reasons.append("both the frequentist test and the posterior agree the change helps")
    else:
        recommendation = HOLD
        confidence = "low"
        reasons.append(
            "the evidence does not separate the two builds at the agreed error rate, so "
            "shipping would be a coin flip dressed up as a decision"
        )
    return Verdict(
        recommendation=recommendation,
        confidence=confidence,
        reasons=reasons,
        blocking_gate=None,
    )


@dataclass(frozen=True)
class FinancialImpact:
    absolute_diff_pp: float
    ci_low_pp: float
    ci_high_pp: float
    annual_new_players: float
    rollout_share: float
    revenue_per_retained_player_usd: float
    retained_players_delta_per_year: float
    annual_revenue_delta_usd: float
    annual_revenue_delta_low_usd: float
    annual_revenue_delta_high_usd: float
    assumptions: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def financial_impact(
    absolute_diff_pp: float,
    ci_low_pp: float,
    ci_high_pp: float,
    business: BusinessAssumptions,
) -> FinancialImpact:
    """Price the retention move, with every assumption written down next to the number.

    The arithmetic is deliberately simple. A retention change is converted to
    retained players, retained players to monetised days, monetised days to
    revenue. Anyone who disagrees with the result can point at the assumption they
    would change rather than at the model.
    """
    annual_players = business.annual_new_players() * business.variant_rollout_share
    revenue_each = business.revenue_per_retained_player()

    def to_usd(points: float) -> float:
        return annual_players * (points / 100.0) * revenue_each

    retained_delta = annual_players * (absolute_diff_pp / 100.0)
    return FinancialImpact(
        absolute_diff_pp=float(absolute_diff_pp),
        ci_low_pp=float(ci_low_pp),
        ci_high_pp=float(ci_high_pp),
        annual_new_players=float(annual_players),
        rollout_share=float(business.variant_rollout_share),
        revenue_per_retained_player_usd=float(revenue_each),
        retained_players_delta_per_year=float(retained_delta),
        annual_revenue_delta_usd=float(to_usd(absolute_diff_pp)),
        annual_revenue_delta_low_usd=float(to_usd(ci_low_pp)),
        annual_revenue_delta_high_usd=float(to_usd(ci_high_pp)),
        assumptions=[
            f"{business.monthly_new_players:,} new players a month reach the gate, "
            f"{business.variant_rollout_share:.0%} of them under the change",
            f"a player retained at day 7 monetises for "
            f"{business.days_monetised_per_retained_player:.0f} further days",
            f"average revenue per daily active player is ${business.arpdau_usd:.3f}",
            "the day-7 retention gap is assumed to persist rather than close on its own",
            "no change in install volume, store ranking or acquisition cost is modelled",
        ],
    )
