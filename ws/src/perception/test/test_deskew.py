"""perception.deskew, and the particle filter on scans swept while the car moves (as a real lidar's are)."""
import math

import numpy as np
import pytest

from perception.deskew import deskew, interpolate_poses, relative_poses


def test_a_wall_ahead_is_straight_again():
    # the car drives +x at 7 m/s while one 25 ms scan sweeps a wall across x = 5 m (world)
    v, sweep, n = 7.0, 0.025, 91
    angles = np.radians(np.linspace(-45, 45, n))
    t = np.linspace(0.0, sweep, n)
    car_x = v * t
    ranges = (5.0 - car_x) / np.cos(angles)            # what each beam measured from where the car was
    raw = ranges * np.cos(angles)
    assert raw.max() - raw.min() == pytest.approx(v * sweep, rel=1e-6)   # 0.175 m: the wall slanted
    rel = relative_poses((0.0, 0.0, 0.0), car_x, np.zeros(n), np.zeros(n))
    r, a = deskew(ranges, angles, rel, (0.0, 0.0, 0.0))
    assert r * np.cos(a) == pytest.approx(np.full(n, 5.0))


def test_a_sensor_off_base_link_and_a_turning_car():
    # base_link turns 0.2 rad and moves while the lidar (0.27 m ahead) sweeps: the points come out where a
    # still car at the start would have seen them
    sensor = (0.27, 0.0, 0.0)
    world = np.array([[4.0, 1.0], [3.0, -2.0], [-1.0, 3.0]])
    poses = [(0.0, 0.0, 0.0), (0.1, 0.02, 0.1), (0.2, 0.06, 0.2)]
    ranges, angles = [], []
    for (px, py), (x, y, yaw) in zip(world, poses):   # each point seen from the sensor at its beam's pose
        sx, sy = x + math.cos(yaw) * sensor[0], y + math.sin(yaw) * sensor[0]
        ranges.append(math.hypot(px - sx, py - sy))
        angles.append(math.atan2(py - sy, px - sx) - yaw)
    rel = relative_poses((0.0, 0.0, 0.0), *np.array(poses).T)
    r, a = deskew(ranges, angles, rel, sensor)
    got = np.column_stack((sensor[0] + r * np.cos(a), r * np.sin(a)))
    assert got == pytest.approx(world)


def test_interpolate_poses_across_the_yaw_wrap():
    x, y, yaw = interpolate_poses([0.0, 1.0], [(0.0, 0.0, 3.1), (2.0, 0.0, -3.1)], np.array([0.5, 2.0]))
    assert x == pytest.approx([1.0, 2.0])            # held past the newest pose
    assert math.cos(yaw[0]) == pytest.approx(-1.0, abs=1e-3)   # through pi, not through 0


rclpy = pytest.importorskip('rclpy')
from sensor_msgs.msg import LaserScan  # noqa: E402

from perception.mcl import compose  # noqa: E402
from test_mcl import ANGLE_INC, ANGLE_MIN, N_SCAN, RANGE_MAX, lap, scan_from  # noqa: E402
from test_mcl_node import Car, lmap, tracks  # noqa: E402,F401  (fixtures)


def lerp(a, b, u):
    turn = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
    return (a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), a[2] + u * turn)


def swept_scan(c, start, end, stamp, sweep, groups=27):
    """A scan stamped at its first beam, its beams taken while the car moved from start to end (in groups)."""
    m = LaserScan()
    m.header.frame_id = 'laser'
    m.header.stamp.sec, m.header.stamp.nanosec = int(stamp), int(round((stamp % 1) * 1e9)) % 1_000_000_000
    m.angle_min, m.angle_increment = ANGLE_MIN, ANGLE_INC
    m.range_min, m.range_max = 0.1, RANGE_MAX
    m.time_increment = sweep / (N_SCAN - 1)
    r = np.empty(N_SCAN)
    for g, idx in enumerate(np.array_split(np.arange(N_SCAN), groups)):
        r[idx] = scan_from(c.lmap, lerp(start, end, (idx.mean() / (N_SCAN - 1))), c.rng)[idx]
    m.ranges = r.astype(np.float32).tolist()
    return m


def track_error(c, deskew_on, every=3):
    """Mean / worst position error of the filter over a fast stretch (every 3rd 0.08 m pose per 25 ms: 9.6 m/s),
    each scan swept over the 25 ms the car drove from the last pose to this one."""
    P = lap()
    c.stand(P[92])
    assert c.node.localized
    prev, errors = P[92], []
    for k in range(92 + every, 92 + every * 60, every):
        c.stamp += c.dt
        c.node.odom_cb(c.odom(P[k], speed=0.08 * every / c.dt))
        c.node.scan_cb(swept_scan(c, prev, P[k], c.stamp - c.dt, c.dt))
        if c.node.job is not None:
            c.node.job[0].result()
        est = compose(c.node.map_T_odom, compose(c.odom_frame, P[k]))
        errors.append(math.hypot(est[0] - P[k][0], est[1] - P[k][1]))
        prev = P[k]
    return float(np.mean(errors)), float(np.max(errors))


def run(lmap, deskew_on):
    c = Car(lmap, deskew='true' if deskew_on else 'false')   # one rclpy context at a time
    try:
        return track_error(c, deskew_on)
    finally:
        c.close()


def test_particle_filter_on_swept_scans(tracks, lmap):
    skewed = run(lmap, False)
    straight = run(lmap, True)
    assert straight[0] < 0.7 * skewed[0], (straight, skewed)
    assert straight[1] < 0.2, straight
