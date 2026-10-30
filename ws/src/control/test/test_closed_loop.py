"""Closed loop: pure_pursuit_node driving sim_car.SimCar (a kinematic bicycle with actuation delay, steering slew
and a steering gain) round a stadium, measuring the distance from the raceline.

What it checks, on a car pure pursuit's geometry does not model:
  latency:=<the delay>  a car with a large delay (0.2 s) tracks as one without (plain: it weaves)
  the adapter's steering_gain (SimCar compensate) takes the error of a car that turns 1.8x harder away
These are kinematic effects only (no tires, no slip): what they are worth in AutoDRIVE is for lap_logger_node.
Measured this way (stadium, 1.5 m half turns, rms / worst m): 4 m/s no delay 0.033 / 0.08; 0.05 s 0.015 / 0.04;
0.1 s 0.018 / 0.05; 0.15 s 0.12 / 0.33; 0.2 s 0.25 / 0.75; 0.3 s 0.65 / 1.2; any of them compensated 0.033 / 0.08.
7 m/s: none 0.14 / 0.26, 0.1 s 0.07 / 0.14, 0.15 s 0.21 / 0.49, 0.2 s 0.45 / 1.06, compensated 0.145 / 0.27.
Steering gain 1.8: 0.110 / 0.256, calibrated 0.034 / 0.082.

Needs a sourced ROS 2 environment with the control package built; ~1 min.
"""
import os
import random
import signal
import subprocess
import tempfile
import time

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')

from control.telemetry import Track  # noqa: E402
from sim_car import SimCar  # noqa: E402
from test_pure_pursuit import hairpin, write_raceline  # noqa: E402

DOMAIN = random.randint(1, 101)
SPEED = 4.0      # m/s, at speed_scale 1.0: 0.22 rad of steering in the 1.5 m radius half turns
DRIVE_S, SETTLE_S = 14.0, 4.0


@pytest.fixture(scope='module')
def tracks():
    with tempfile.TemporaryDirectory() as d:
        xy, psi = hairpin(10.0, 3.0, 0.1)
        write_raceline(d, 'stadium', xy, psi, 0.0, SPEED)
        yield d


def lap(tracks, car_args, pp_args=(), track='stadium'):
    """RMS and worst distance [m] from the raceline after SETTLE_S s, driving DRIVE_S s from its start."""
    env = dict(os.environ, TRACKS_DIR=tracks, ROS_DOMAIN_ID=str(DOMAIN))
    proc = subprocess.Popen(['ros2', 'launch', 'control', 'pure_pursuit.launch.py', f'track:={track}',
                             'speed_scale:=1.0', *pp_args], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    rclpy.init(domain_id=DOMAIN)
    node = rclpy.create_node('closed_loop')
    errors = []
    try:
        xy, psi = hairpin(10.0, 3.0, 0.1)
        track = Track(xy)
        car = SimCar(node, (xy[0, 0], xy[0, 1], psi[0]), **car_args)
        end = time.time() + 20.0
        while car.pose_pub.get_subscription_count() == 0:
            assert proc.poll() is None and time.time() < end, proc.stdout.read() if proc.poll() is not None else ''
            rclpy.spin_once(node, timeout_sec=0.1)

        def measure(t):
            if t > SETTLE_S:
                errors.append(track.locate(car.x, car.y, car.yaw)[1])
        car.run(DRIVE_S, on_step=measure)
        assert car.drives > 100, f'{car.drives} /drive commands'
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        node.destroy_node()
        rclpy.shutdown()
    e = np.abs(errors)
    return float(np.sqrt(np.mean(e ** 2))), float(e.max())


@pytest.fixture(scope='module')
def ideal(tracks):
    return lap(tracks, dict())


def test_latency_compensation_takes_a_large_delay_out(tracks, ideal):
    # 0.2 s: plain pure pursuit weaves (rms 0.25 m, worst 0.75 m at 4 m/s); compensated it tracks as with no delay.
    # Up to ~0.1 s plain tracks as well or better: the delay offsets pure pursuit's early turn-in (README)
    delayed = lap(tracks, dict(delay=0.2))
    compensated = lap(tracks, dict(delay=0.2), ['latency:=0.2'])
    print(f'ideal {ideal}, 0.2 s delay {delayed}, compensated {compensated}')
    assert delayed[0] > 3 * ideal[0]
    assert compensated[0] == pytest.approx(ideal[0], rel=0.3, abs=0.01)


def test_steering_gain_takes_the_mismatch_out(tracks, ideal):
    mismatched = lap(tracks, dict(steering_gain=1.8))
    calibrated = lap(tracks, dict(steering_gain=1.8, compensate=1.8))
    print(f'ideal {ideal}, gain 1.8 {mismatched}, calibrated {calibrated}')
    assert calibrated[0] == pytest.approx(ideal[0], rel=0.3, abs=0.01)
    assert mismatched[0] > 2 * calibrated[0]
