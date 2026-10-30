"""control.profile: pure pursuit's speed profile (sector scales), swerve speed cap and pose prediction."""
import math

import numpy as np
import pytest

from control.profile import (arc_lengths, limit_accelerations, predict_with_commands, profile_accelerations,
                             read_sectors, sector_scales, speed_profile, swerve_cap, write_sectors)


def loop(n=200, length=20.0):
    """A closed loop of n points 'length' m long (a circle), its segment lengths and arc lengths."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = length / (2 * np.pi)
    xy = np.column_stack([r * np.cos(t), r * np.sin(t)])
    return (xy,) + arc_lengths(xy)


def test_arc_lengths():
    _, ds, s, length = loop()
    assert length == pytest.approx(20.0, rel=1e-3)
    assert s[0] == 0.0 and s[-1] == pytest.approx(length - ds[-1])


def test_limit_accelerations_respects_both_limits():
    _, ds, _, _ = loop()
    v = np.full(200, 5.0)
    v[100] = 1.0   # a slow point
    out = limit_accelerations(v, ds, a_pos=2.0, a_neg=4.0)
    assert out[100] == 1.0 and (out <= v + 1e-12).all()
    acc = profile_accelerations(out, ds)
    assert acc.max() <= 2.0 + 1e-9 and acc.min() >= -4.0 - 1e-9
    # braking into it starts (25 - 1) / (2 x 4) = 3 m before, accelerating out takes 6 m: shorter
    slow = np.flatnonzero(out < 5.0 - 1e-9)
    before, after = 100 - slow.min(), slow.max() - 100
    assert before * 0.1 == pytest.approx(3.0, abs=0.15) and after * 0.1 == pytest.approx(6.0, abs=0.15)


def test_speed_profile_without_sectors_is_the_scaled_raceline():
    _, ds, s, length = loop()
    vx = np.linspace(2.0, 6.0, 200)
    assert speed_profile(vx, ds, s, length, 0.5) == pytest.approx(vx * 0.5)
    assert speed_profile(vx, ds, s, length, 0.5, [1.0, 1.0]) == pytest.approx(vx * 0.5)
    assert speed_profile(-vx, ds, s, length, 0.5).max() == 0.0   # never reversing


def test_sector_scales_and_their_boundaries():
    _, ds, s, length = loop()
    assert list(sector_scales([0.0, 4.9, 5.0, 19.99], 20.0, [1, 2, 3, 4])) == [1, 1, 2, 4]
    vx = np.full(200, 4.0)
    v = speed_profile(vx, ds, s, length, 1.0, [1.0, 1.5], a_pos=2.0, a_neg=4.0)
    assert v[50] == pytest.approx(4.0) and v[150] == pytest.approx(6.0)
    acc = profile_accelerations(v, ds)
    assert acc.max() <= 2.0 + 1e-9 and acc.min() >= -4.0 - 1e-9
    assert v[100] < 6.0 and v[199] < 6.0   # accelerating into the fast sector, braking out of it


def test_sector_limits_default_to_the_racelines_own():
    _, ds, s, length = loop()
    vx = 4.0 + np.sin(np.linspace(0, 2 * np.pi, 200, endpoint=False))   # planned accelerations ~ +-1.3 m/s^2
    planned = profile_accelerations(vx, ds)
    v = speed_profile(vx, ds, s, length, 1.0, [0.8, 1.2])
    acc = profile_accelerations(v, ds)
    assert acc.max() <= max(1.0, planned.max()) + 1e-9 and acc.min() >= min(-1.0, planned.min()) - 1e-9


def test_swerve_cap_brakes_into_the_swerve():
    _, ds, s, length = loop()
    kappa = np.zeros(200)
    shifted = np.zeros(200, dtype=bool)
    kappa[100:110], shifted[100:110] = 2.0, True   # a swerve of curvature 2 / m
    cap = swerve_cap(kappa, shifted, s, length, max_lateral_accel=8.0, max_decel=4.0)
    assert cap[105] == pytest.approx(2.0)                       # sqrt(8 / 2)
    assert cap[90] == pytest.approx(math.sqrt(4.0 + 2 * 4.0 * (s[100] - s[90])))
    assert np.isfinite(cap).all() and cap[99] < cap[50]
    kappa[50] = 5.0   # curvature where the path is not shifted (the raceline's own corner): no cap
    assert swerve_cap(kappa, shifted, s, length, 8.0, 4.0)[50] == pytest.approx(cap[50])
    assert np.isinf(swerve_cap(kappa, np.zeros(200, dtype=bool), s, length, 8.0, 4.0)).all()


L = 0.33


def test_predict_without_commands_uses_the_yaw_rate():
    assert predict_with_commands((0.0, 0.0, 0.0), 2.0, 0.0, 0.0, 0.5, [], 0.1, L) == pytest.approx((1.0, 0.0, 0.0))
    assert predict_with_commands((1.0, 2.0, 0.3), 5.0, 1.0, 3.0, 3.0, [], 0.1, L) == (1.0, 2.0, 0.3)
    # a quarter circle of radius 1 m: from (0, 0) heading +x, turning left, ends at (1, 1) heading +y
    got = predict_with_commands((0.0, 0.0, 0.0), 1.0, 1.0, 0.0, math.pi / 2, [], 0.1, L, dt=0.001)
    assert got == pytest.approx((1.0, 1.0, math.pi / 2), abs=1e-4)


def test_predict_drives_the_commands_in_flight():
    # driving straight; a left turn of radius 1 m was commanded 0.05 s ago and acts after 0.1 s of latency: the
    # next 0.05 s are straight, the 0.1 s after that on the arc
    steer = math.atan(L / 1.0)
    commands = [(9.0, 0.0), (10.0 - 0.05, steer)]
    x, y, yaw = predict_with_commands((0.0, 0.0, 0.0), 2.0, 0.0, 10.0, 10.0 + 0.15, commands, 0.1, L, dt=0.001)
    assert yaw == pytest.approx(2.0 * 0.1 / 1.0, abs=1e-3)   # 0.2 m on the arc: 0.2 rad
    assert x == pytest.approx(0.1 + math.sin(0.2), abs=2e-3) and y == pytest.approx(1 - math.cos(0.2), abs=2e-3)
    # the yaw rate measured now (0) is not what the car will do
    assert predict_with_commands((0.0, 0.0, 0.0), 2.0, 0.0, 10.0, 10.05, commands, 0.1, L)[2] == 0.0


def test_sectors_file_round_trip(tmp_path):
    path = str(tmp_path / 'sectors.yaml')
    write_sectors(path, [1.0, 1.05, 0.95], 30.6625, comment='tuned from 5 laps\nrun B')
    assert read_sectors(path) == ([1.0, 1.05, 0.95], pytest.approx(30.6625, abs=1e-3))
    text = open(path).read()
    assert '# tuned from 5 laps' in text and '# run B' in text


@pytest.mark.parametrize('content', ['scales: []', 'scales: [1.0, -0.5]', 'length_m: 3.0', 'scales: [1.0, .nan]'])
def test_bad_sectors_file(tmp_path, content):
    path = tmp_path / 'sectors.yaml'
    path.write_text(content + '\n')
    with pytest.raises(ValueError):
        read_sectors(str(path))
