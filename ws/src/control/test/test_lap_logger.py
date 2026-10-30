"""Functional test: lap_logger_node on a circular raceline, with ground truth, a localizer and the sim's counters.

Needs a sourced ROS 2 environment with the control package built:
    python3 -m pytest ws/src/control/test/test_lap_logger.py
"""
import math
import os
import random
import signal
import subprocess
import tempfile
import time

import numpy as np
import pytest
from PIL import Image

rclpy = pytest.importorskip('rclpy')
from ackermann_msgs.msg import AckermannDriveStamped  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from std_msgs.msg import Float32, Int32  # noqa: E402

from control.telemetry import LAP_COLUMNS, SECTOR_COLUMNS, read_runs  # noqa: E402

DOMAIN = random.randint(1, 101)
R, SPEED, DT = 4.0, 5.0, 0.02
ORIGIN = (10.0, -5.0, 0.5)   # the map origin in the sim world (map_to_world.yaml)
LOC_OFFSET = 0.1             # m the fake localizer is off in x


def make_track(d):
    """tracks/circle: a raceline on a circle of radius R, walls at R - 1 and R + 1, map_to_world.yaml."""
    os.makedirs(d)
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    xy = np.column_stack([R * np.cos(t), R * np.sin(t)])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    rows = np.column_stack([s, xy, t, np.full(len(t), 1 / R), np.full(len(t), SPEED), np.zeros(len(t))])
    np.savetxt(os.path.join(d, 'raceline.csv'), rows, delimiter=';', fmt='%.7f',
               header='s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2')
    with open(os.path.join(d, 'map_to_world.yaml'), 'w') as f:
        f.write(f'frame_id: world\nx_m: {ORIGIN[0]}\ny_m: {ORIGIN[1]}\nyaw_rad: {ORIGIN[2]}\n')
    res, size = 0.05, 280   # 14 m square centred on the circle
    yy, xx = np.mgrid[0:size, 0:size]
    px, py = (xx + 0.5) * res - 7.0, (size - 1 - yy + 0.5) * res - 7.0
    r = np.hypot(px, py)
    img = np.full((size, size), 254, dtype=np.uint8)
    img[(np.abs(r - (R - 1)) < 0.05) | (np.abs(r - (R + 1)) < 0.05)] = 0
    Image.fromarray(img).save(os.path.join(d, 'map.pgm'))
    with open(os.path.join(d, 'map.yaml'), 'w') as f:
        f.write('image: map.pgm\nmode: trinary\nresolution: 0.05\norigin: [-7.0, -7.0, 0.0]\nnegate: 0\n'
                'occupied_thresh: 0.65\nfree_thresh: 0.25\n')


def world(x, y, yaw):
    ox, oy, oyaw = ORIGIN
    c, s = math.cos(oyaw), math.sin(oyaw)
    return ox + c * x - s * y, oy + s * x + c * y, yaw + oyaw


def stamp(msg, t):
    msg.header.stamp.sec, msg.header.stamp.nanosec = int(t), int(round((t % 1) * 1e9)) % 1_000_000_000


@pytest.fixture
def tracks():
    with tempfile.TemporaryDirectory() as d:
        make_track(os.path.join(d, 'circle'))
        yield d


def run_logger(tracks, *params):
    env = dict(os.environ, TRACKS_DIR=tracks, ROS_DOMAIN_ID=str(DOMAIN))
    args = ['ros2', 'run', 'control', 'lap_logger_node', '--ros-args', '-p', 'track:=circle', '-p', 'sectors:=4']
    for p in params:
        args += ['-p', p]
    return subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    return proc.stdout.read()


def test_laps_from_ground_truth(tracks):
    proc = run_logger(tracks, 'laps:=2', 'tag:=unit')
    rclpy.init(domain_id=DOMAIN)
    node = rclpy.create_node('lap_logger_test')
    try:
        gt = node.create_publisher(Odometry, '/ground_truth/odom', 50)
        loc = node.create_publisher(PoseStamped, '/localized_pose', 50)
        drive = node.create_publisher(AckermannDriveStamped, '/drive', 10)
        collisions = node.create_publisher(Int32, '/autodrive/roboracer_1/collision_count', 10)
        last_lap = node.create_publisher(Float32, '/autodrive/roboracer_1/last_lap_time', 10)
        end = time.time() + 20.0
        while min(p.get_subscription_count() for p in (gt, loc, drive, collisions, last_lap)) == 0:
            assert proc.poll() is None, 'node exited:\n' + proc.stdout.read()
            assert time.time() < end, 'lap_logger_node did not subscribe'
            rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.5)
        lap_s = 2 * math.pi * R / SPEED
        t0, phase0 = 100.0, -2.0 / R   # starts 2 m before the line (s = 0 at angle 0)
        k = 0
        while proc.poll() is None and k * DT < 3.5 * lap_s:
            t = t0 + k * DT
            a = phase0 + SPEED * k * DT / R
            x, y, yaw = R * math.cos(a), R * math.sin(a), a + math.pi / 2
            wx, wy, wyaw = world(x, y, yaw)
            o = Odometry()
            stamp(o, t)
            o.header.frame_id = 'world'
            o.pose.pose.position.x, o.pose.pose.position.y = wx, wy
            o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(wyaw / 2), math.cos(wyaw / 2)
            o.twist.twist.linear.x = SPEED
            gt.publish(o)
            if k % 2:   # the localizer: half the rate, LOC_OFFSET off, stamped like the ground truth it saw
                p = PoseStamped()
                stamp(p, t)
                p.header.frame_id = 'map'
                p.pose.position.x, p.pose.position.y = x + LOC_OFFSET, y
                p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
                loc.publish(p)
            d = AckermannDriveStamped()
            d.drive.speed = SPEED - 1.0
            drive.publish(d)
            crashed = k * DT > 2.5 * lap_s   # a crash half way round the second timed lap (the first crossing,
            #                                  2 m after the start, is not timed: no run-up)
            collisions.publish(Int32(data=4 if crashed else 3))   # as the sim: every frame
            if crashed:
                last_lap.publish(Float32(data=5.02))
            k += 1
            time.sleep(0.002)
            rclpy.spin_once(node, timeout_sec=0)
        proc.wait(timeout=10)
    finally:
        out = stop(proc)
        node.destroy_node()
        rclpy.shutdown()
    assert proc.returncode == 0, out
    assert '2 laps logged' in out, out

    runs = read_runs(os.path.join(tracks, 'circle', 'laps.csv'))
    assert len(runs) == 1 and 'tag=unit' in runs[0][0] and 'source=ground_truth' in runs[0][0]
    first, second = runs[0][1]
    assert first['lap_time_s'] == pytest.approx(lap_s, abs=0.01)
    assert second['lap_time_s'] == pytest.approx(lap_s, abs=0.01)
    assert (first['valid'], first['collisions']) == (1, 0)
    assert (second['valid'], second['collisions']) == (0, 1)
    assert second['sim_lap_time_s'] == pytest.approx(5.02, abs=1e-4)
    assert first['lat_max_m'] < 0.01
    assert first['loc_rms_m'] == pytest.approx(LOC_OFFSET, abs=0.005)
    assert first['speed_err_rms_mps'] == pytest.approx(1.0, abs=1e-3)
    assert first['speed_mean_mps'] == pytest.approx(SPEED, abs=1e-3)
    # the footprint's outer front corner is ~0.84 m inside the outer wall (radius R + 1)
    assert 0.75 < first['clearance_min_m'] < 0.9

    with open(os.path.join(tracks, 'circle', 'laps_sectors.csv')) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines[0] == ','.join(SECTOR_COLUMNS) and lines[1].startswith('# run')
    sectors = [dict(zip(SECTOR_COLUMNS, map(float, ln.split(',')))) for ln in lines[2:]]
    assert len(sectors) == 8 and [r['sector'] for r in sectors[:4]] == [0, 1, 2, 3]
    assert sum(r['time_s'] for r in sectors[:4]) == pytest.approx(lap_s, abs=0.05)

    summary = subprocess.run(['ros2', 'run', 'control', 'lap_summary', 'circle'], text=True, capture_output=True,
                             env=dict(os.environ, TRACKS_DIR=tracks))
    assert summary.returncode == 0, summary.stderr
    row = summary.stdout.splitlines()[1].split()
    assert row[:4] == ['circle', 'unit', '2', '1'] and float(row[4]) == pytest.approx(lap_s, abs=0.01)


def test_header_written_once(tracks):
    path = os.path.join(tracks, 'circle', 'laps.csv')

    def lines():
        if not os.path.isfile(path):
            return []
        with open(path) as f:
            return f.read().splitlines()

    for n in (1, 2):
        proc = run_logger(tracks, 'tag:=again')
        end = time.time() + 20.0
        while sum(ln.startswith('# run') for ln in lines()) < n and time.time() < end:
            time.sleep(0.1)
        out = stop(proc)
        assert sum(ln.startswith('# run') for ln in lines()) == n, out
    lines = lines()
    assert lines[0] == ','.join(LAP_COLUMNS)
    assert sum(ln == ','.join(LAP_COLUMNS) for ln in lines) == 1
    assert sum(ln.startswith('# run') for ln in lines) == 2


def test_missing_ground_truth_offset_is_an_error(tracks):
    os.remove(os.path.join(tracks, 'circle', 'map_to_world.yaml'))
    proc = run_logger(tracks, 'pose_source:=ground_truth')
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    out = stop(proc)
    assert 'map_to_world.yaml' in out
