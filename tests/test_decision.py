from __future__ import annotations

import pytest

from gatecheck.config import BusinessAssumptions, DecisionRules
from gatecheck.decision import (
    HOLD,
    ROLL_BACK,
    SHIP,
    MetricEvidence,
    evaluate_experiment,
    financial_impact,
)

RULES = DecisionRules()


def evidence(**overrides) -> MetricEvidence:
    base = {
        "metric": "retention_7",
        "absolute_diff_pp": -0.82,
        "ci_low_pp": -1.33,
        "ci_high_pp": -0.31,
        "p_value": 0.0016,
        "posterior_prob_treatment_better": 0.0008,
        "prob_worse_than_floor": 0.99,
    }
    base.update(overrides)
    return MetricEvidence(**base)


def test_a_clear_loss_below_the_floor_is_a_roll_back():
    verdict = evaluate_experiment(evidence(), RULES, validity_passed=True)
    assert verdict.recommendation == ROLL_BACK
    assert verdict.confidence == "high"
    assert any("floor" in reason for reason in verdict.reasons)


def test_a_significant_loss_inside_the_tolerance_lowers_confidence():
    verdict = evaluate_experiment(
        evidence(absolute_diff_pp=-0.18, ci_low_pp=-0.30, ci_high_pp=-0.05, p_value=0.01),
        RULES,
        validity_passed=True,
    )
    assert verdict.recommendation == ROLL_BACK
    assert verdict.confidence == "moderate"


def test_a_clear_gain_is_a_ship():
    verdict = evaluate_experiment(
        evidence(
            absolute_diff_pp=0.9,
            ci_low_pp=0.4,
            ci_high_pp=1.4,
            p_value=0.0004,
            posterior_prob_treatment_better=0.999,
            prob_worse_than_floor=0.0,
        ),
        RULES,
        validity_passed=True,
    )
    assert verdict.recommendation == SHIP
    assert verdict.confidence == "high"


def test_an_inconclusive_read_is_a_hold():
    verdict = evaluate_experiment(
        evidence(
            absolute_diff_pp=-0.10,
            ci_low_pp=-0.45,
            ci_high_pp=0.25,
            p_value=0.42,
            posterior_prob_treatment_better=0.29,
            prob_worse_than_floor=0.2,
        ),
        RULES,
        validity_passed=True,
    )
    assert verdict.recommendation == HOLD
    assert verdict.confidence == "low"


def test_a_gain_the_posterior_does_not_back_is_a_hold():
    verdict = evaluate_experiment(
        evidence(
            absolute_diff_pp=0.4,
            ci_low_pp=0.02,
            ci_high_pp=0.8,
            p_value=0.04,
            posterior_prob_treatment_better=0.90,
            prob_worse_than_floor=0.0,
        ),
        DecisionRules(min_posterior_prob_to_ship=0.99),
        validity_passed=True,
    )
    assert verdict.recommendation == HOLD


def test_a_failed_validity_gate_blocks_everything():
    verdict = evaluate_experiment(
        evidence(), RULES, validity_passed=False, validity_reason="sample ratio mismatch"
    )
    assert verdict.recommendation == HOLD
    assert verdict.blocking_gate == "validity"
    assert "sample ratio mismatch" in verdict.reasons[1]


def test_a_stricter_floor_can_change_the_confidence():
    strict = evaluate_experiment(evidence(), DecisionRules(practical_floor_pp=-2.0), True)
    assert strict.recommendation == ROLL_BACK
    assert strict.confidence == "moderate"


def test_every_verdict_explains_itself():
    verdict = evaluate_experiment(evidence(), RULES, validity_passed=True)
    assert len(verdict.reasons) >= 3
    assert all(isinstance(reason, str) and reason for reason in verdict.reasons)


def test_a_loss_prices_out_negative():
    impact = financial_impact(-0.82, -1.33, -0.31, BusinessAssumptions())
    assert impact.annual_revenue_delta_usd < 0
    assert impact.annual_revenue_delta_low_usd < impact.annual_revenue_delta_usd
    assert impact.annual_revenue_delta_usd < impact.annual_revenue_delta_high_usd


def test_impact_scales_linearly_with_player_volume():
    small = financial_impact(
        -0.82, -1.33, -0.31, BusinessAssumptions(monthly_new_players=1_000_000)
    )
    large = financial_impact(
        -0.82, -1.33, -0.31, BusinessAssumptions(monthly_new_players=2_000_000)
    )
    assert large.annual_revenue_delta_usd == pytest.approx(2.0 * small.annual_revenue_delta_usd)


def test_a_partial_rollout_softens_the_impact():
    full = financial_impact(-0.82, -1.33, -0.31, BusinessAssumptions())
    half = financial_impact(-0.82, -1.33, -0.31, BusinessAssumptions(variant_rollout_share=0.5))
    assert half.annual_revenue_delta_usd == pytest.approx(0.5 * full.annual_revenue_delta_usd)


def test_zero_effect_is_worth_nothing():
    impact = financial_impact(0.0, -0.4, 0.4, BusinessAssumptions())
    assert impact.annual_revenue_delta_usd == pytest.approx(0.0)


def test_every_assumption_is_written_down():
    impact = financial_impact(-0.82, -1.33, -0.31, BusinessAssumptions())
    assert len(impact.assumptions) >= 4
    assert any(
        "ARPDAU" in line or "revenue per daily active" in line for line in impact.assumptions
    )
