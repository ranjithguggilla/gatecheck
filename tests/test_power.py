from __future__ import annotations

import numpy as np
import pytest

from gatecheck import power


def test_bigger_effects_need_fewer_players():
    small = power.required_n_per_arm(0.19, 0.005)
    large = power.required_n_per_arm(0.19, 0.02)
    assert small > large > 0


def test_required_n_scales_roughly_with_the_inverse_square_of_the_effect():
    base = power.required_n_per_arm(0.19, 0.01)
    halved = power.required_n_per_arm(0.19, 0.005)
    assert halved / base == pytest.approx(4.0, rel=0.1)


def test_power_rises_with_sample_size():
    low = power.achieved_power(0.19, 0.008, 5_000)
    high = power.achieved_power(0.19, 0.008, 45_000)
    assert 0.0 < low < high < 1.0


def test_mde_is_the_effect_that_lands_on_the_target_power():
    mde = power.minimum_detectable_effect(0.19, 44_700, power=0.80)
    assert power.achieved_power(0.19, mde, 44_700) == pytest.approx(0.80, abs=0.01)


def test_report_marks_the_observed_effect_as_detectable_here():
    report = power.power_report("retention_7", 0.19020, 44_700, -0.008201)
    assert report.mde_absolute_pp < abs(report.observed_effect_pp)
    assert report.achieved_power_at_observed_effect > 0.80


def test_report_marks_a_tiny_effect_as_underpowered():
    report = power.power_report("retention_7", 0.19020, 44_700, -0.001)
    assert report.achieved_power_at_observed_effect < 0.30


def test_zero_effect_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        power.required_n_per_arm(0.19, 0.0)


def test_impossible_baseline_is_rejected():
    with pytest.raises(ValueError, match="must sit in"):
        power.required_n_per_arm(1.4, 0.01)


def test_sample_size_curve_is_monotone_decreasing():
    lifts = np.linspace(0.02, 0.20, 12)
    _, sizes = power.sample_size_curve(0.19, lifts)
    assert np.all(np.diff(sizes) < 0)
