from __future__ import annotations

import numpy as np
import pytest

from gatecheck import validity


def test_even_split_is_not_flagged():
    result = validity.srm_check(50_000, 50_000)
    assert result.chi_square == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert not result.triggered


def test_lopsided_split_is_flagged():
    result = validity.srm_check(40_000, 50_000)
    assert result.p_value < 1e-10
    assert result.triggered
    assert result.observed_share_treatment > 0.5


def test_srm_matches_the_hand_computed_statistic():
    result = validity.srm_check(44_700, 45_489)
    total = 44_700 + 45_489
    expected = total / 2
    manual = (45_489 - expected) ** 2 / expected + (44_700 - expected) ** 2 / expected
    assert result.chi_square == pytest.approx(manual)


def test_srm_rejects_empty_arms():
    with pytest.raises(ValueError, match="at least one unit"):
        validity.srm_check(0, 10)


def test_srm_rejects_an_impossible_share():
    with pytest.raises(ValueError, match="must sit in"):
        validity.srm_check(10, 10, expected_share_treatment=1.5)


def test_duplicate_check_is_clean_on_clean_data(players):
    result = validity.duplicate_check(players)
    assert result.clean
    assert result.n_duplicate_ids == 0


def test_duplicate_check_catches_a_player_in_both_arms(players):
    dirty = players.copy()
    dirty.loc[0, "userid"] = dirty.loc[len(dirty) - 1, "userid"]
    result = validity.duplicate_check(dirty)
    assert not result.clean
    assert result.n_duplicate_ids == 1
    assert result.n_ids_in_both_arms == 1


def test_outlier_audit_reports_the_tail(players):
    audit = validity.outlier_audit(players["sum_gamerounds"], 0.99)
    assert audit.maximum >= audit.p999 >= audit.p99
    assert audit.winsor_cut <= audit.maximum


def test_balance_check_finds_no_pre_treatment_covariate(players):
    check = validity.balance_check(players)
    assert check.pre_treatment_columns == ("userid",)
    assert not check.can_test_covariate_balance
    assert "sum_gamerounds" in check.post_treatment_columns


def test_gate_summary_passes_when_every_gate_passes(players):
    summary = validity.gate_summary(
        validity.srm_check(1_200, 1_200),
        validity.duplicate_check(players),
        validity.outlier_audit(players["sum_gamerounds"]),
    )
    assert summary["passed"]
    assert summary["blocking_reason"] is None
    assert len(summary["reasons"]) == 3


def test_gate_summary_fails_and_explains_itself(players):
    summary = validity.gate_summary(
        validity.srm_check(30_000, 50_000),
        validity.duplicate_check(players),
        validity.outlier_audit(players["sum_gamerounds"]),
    )
    assert not summary["passed"]
    assert any("sample ratio mismatch" in reason for reason in summary["reasons"])
    assert "sample ratio mismatch" in summary["blocking_reason"]


def test_gate_summary_names_the_gate_that_actually_failed(players):
    dirty = players.copy()
    dirty.loc[0, "userid"] = dirty.loc[len(dirty) - 1, "userid"]
    summary = validity.gate_summary(
        validity.srm_check(1_200, 1_200),
        validity.duplicate_check(dirty),
        validity.outlier_audit(dirty["sum_gamerounds"]),
    )
    assert not summary["passed"]
    assert "ids in both arms" in summary["blocking_reason"]


def test_entropy_is_one_bit_for_an_even_split():
    assert validity.assignment_entropy(np.array([500, 500])) == pytest.approx(1.0)
    assert validity.assignment_entropy(np.array([900, 100])) < 1.0


def test_entropy_rejects_empty_counts():
    with pytest.raises(ValueError, match="positive"):
        validity.assignment_entropy(np.array([0, 0]))
