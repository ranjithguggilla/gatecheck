"""Property based checks on the statistics, because hand-picked cases miss the edges."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from gatecheck import frequentist, power, validity

ARM = st.integers(min_value=30, max_value=200_000)
SUCCESS_SHARE = st.floats(min_value=0.001, max_value=0.999, allow_nan=False)


@settings(max_examples=120, deadline=None)
@given(ARM, SUCCESS_SHARE, ARM, SUCCESS_SHARE)
def test_p_value_always_lands_in_the_unit_interval(n_c, share_c, n_t, share_t):
    result = frequentist.two_proportion_test(
        "m", int(n_c * share_c), n_c, int(n_t * share_t), n_t
    )
    assert 0.0 <= result.p_value <= 1.0
    assert math.isfinite(result.z_statistic)


@settings(max_examples=120, deadline=None)
@given(ARM, SUCCESS_SHARE, ARM, SUCCESS_SHARE)
def test_swapping_the_arms_flips_the_sign_and_keeps_the_p_value(n_c, share_c, n_t, share_t):
    successes_c, successes_t = int(n_c * share_c), int(n_t * share_t)
    forward = frequentist.two_proportion_test("m", successes_c, n_c, successes_t, n_t)
    backward = frequentist.two_proportion_test("m", successes_t, n_t, successes_c, n_c)
    assert forward.p_value == backward.p_value
    assert forward.absolute_diff_pp == -backward.absolute_diff_pp


@settings(max_examples=120, deadline=None)
@given(ARM, SUCCESS_SHARE, ARM, SUCCESS_SHARE)
def test_the_interval_is_ordered_and_stays_in_range(n_c, share_c, n_t, share_t):
    low, high = frequentist.newcombe_interval(
        int(n_c * share_c), n_c, int(n_t * share_t), n_t
    )
    assert -1.0 <= low <= high <= 1.0


@settings(max_examples=120, deadline=None)
@given(ARM, SUCCESS_SHARE, ARM, SUCCESS_SHARE)
def test_the_interval_brackets_the_observed_difference(n_c, share_c, n_t, share_t):
    successes_c, successes_t = int(n_c * share_c), int(n_t * share_t)
    difference = successes_t / n_t - successes_c / n_c
    low, high = frequentist.newcombe_interval(successes_c, n_c, successes_t, n_t)
    assert low <= difference <= high


@settings(max_examples=80, deadline=None)
@given(
    st.floats(min_value=0.02, max_value=0.90),
    st.floats(min_value=0.001, max_value=0.05),
)
def test_sample_size_and_power_agree_with_each_other(baseline, lift):
    needed = power.required_n_per_arm(baseline, lift, power=0.80)
    assert needed > 0
    assert power.achieved_power(baseline, lift, needed) >= 0.79


@settings(max_examples=80, deadline=None)
@given(st.integers(min_value=100, max_value=500_000), st.integers(min_value=100, max_value=500_000))
def test_srm_p_value_is_a_probability(n_c, n_t):
    result = validity.srm_check(n_c, n_t)
    assert 0.0 <= result.p_value <= 1.0
    assert result.chi_square >= 0.0


@settings(max_examples=60, deadline=None)
@given(st.integers(min_value=1, max_value=10**6))
def test_a_perfectly_even_split_is_never_flagged(half):
    assert not validity.srm_check(half, half).triggered
