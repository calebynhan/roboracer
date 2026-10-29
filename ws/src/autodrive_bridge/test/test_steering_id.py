"""autodrive_bridge.steering_id (steering_monitor's fit and the adapter's steering_gain): numpy only."""
import math

import numpy as np
import pytest

from autodrive_bridge.steering_id import SteadyFilter, compensate, fit_gain, model_curvature

L = 0.33


def synthetic(gain, n=400, seed=0):
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-0.5, 0.5, n)
    speed = rng.uniform(1.0, 7.0, n)
    yaw_rate = gain * model_curvature(delta, L) * speed * (1 + rng.normal(0, 0.02, n))
    return delta, speed, yaw_rate


def test_fit_recovers_the_gain():
    rows, (g, n) = fit_gain(*synthetic(1.8), L)
    assert g == pytest.approx(1.8, rel=0.02)
    assert n > 300
    assert all(gb == pytest.approx(1.8, rel=0.05) for _, _, gb, nb in rows if gb is not None)


def test_fit_skips_straight_slow_and_sparse_samples():
    delta = np.full(50, 0.01)          # straight driving only
    _, (g, n) = fit_gain(delta, np.full(50, 3.0), np.zeros(50), L)
    assert g is None and n == 0
    _, (g, _) = fit_gain(*synthetic(1.0, n=10), L)
    assert g is None                    # fewer than min_samples


def test_compensate_undoes_the_gain():
    for gain in (0.7, 1.8, 2.5):
        for delta in (-0.4, 0.1, 0.3):
            applied = compensate(delta, gain)
            assert gain * math.tan(applied) == pytest.approx(math.tan(delta))
    assert compensate(0.3, 1.0) == 0.3
    assert compensate(0.3, None) == 0.3
    assert compensate(0.3, 0.0) == 0.3


def test_steady_filter_waits_for_held_steering():
    f = SteadyFilter(hold=0.3, tolerance=0.02)
    assert not f.steady(0.0, 0.2)
    assert not f.steady(0.2, 0.21)
    assert f.steady(0.35, 0.205)
    assert not f.steady(0.4, 0.3)      # steering moved: start over
    assert f.steady(0.75, 0.3)
