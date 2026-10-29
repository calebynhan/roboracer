"""control.telemetry (lap_logger_node's numbers) and control.motion (pure pursuit's latency prediction): numpy only."""
import math

import numpy as np
import pytest

from control.motion import predict_pose
from control.telemetry import LapStats, format_row, lateral_offset, world_to_map

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])   # counter-clockwise


def test_lateral_offset_sign_and_size():
    assert lateral_offset(SQUARE, 5.0, 0.3) == pytest.approx(0.3)     # inside = left of a CCW loop
    assert lateral_offset(SQUARE, 5.0, -0.2) == pytest.approx(-0.2)
    assert lateral_offset(SQUARE, 10.4, 5.0) == pytest.approx(-0.4)
    assert lateral_offset(SQUARE, 0.5, 9.0) == pytest.approx(0.5)     # on the closing segment


def test_world_to_map_inverts_the_origin():
    origin = (3.0, -2.0, math.pi / 2)
    assert world_to_map(origin, 3.0, -2.0, math.pi / 2) == pytest.approx((0.0, 0.0, 0.0))
    x, y, yaw = world_to_map(origin, 3.0, -1.0, math.pi / 2)   # 1 m along the origin's heading
    assert (x, y, yaw) == pytest.approx((1.0, 0.0, 0.0))


def test_lap_stats_row():
    st = LapStats()
    for lat in (0.1, -0.3, 0.2):
        st.add(lat, 5.0, 4.0, 0.05)
    row = st.row(1, 7.0, 6.98, 0)
    assert row['lat_max_m'] == pytest.approx(0.3)
    assert row['lat_rms_m'] == pytest.approx(math.sqrt((0.01 + 0.09 + 0.04) / 3))
    assert row['speed_err_rms_mps'] == pytest.approx(1.0)
    assert row['samples'] == 3
    assert format_row(row).startswith('1,7.0000,6.9800,0,')


def test_lap_stats_without_command_or_localization():
    row = LapStats().row(1, 7.0, float('nan'), -1)
    assert math.isnan(row['lat_rms_m']) and row['samples'] == 0


def test_predict_straight_and_arc():
    assert predict_pose(0.0, 0.0, 0.0, 2.0, 0.0, 0.5) == pytest.approx((1.0, 0.0, 0.0))
    assert predict_pose(1.0, 2.0, 0.3, 5.0, 1.0, 0.0) == (1.0, 2.0, 0.3)
    # a quarter circle of radius 1 m: from (0, 0) heading +x, turning left, ends at (1, 1) heading +y
    x, y, yaw = predict_pose(0.0, 0.0, 0.0, 1.0, 1.0, math.pi / 2)
    assert (x, y, yaw) == pytest.approx((1.0, 1.0, math.pi / 2))
