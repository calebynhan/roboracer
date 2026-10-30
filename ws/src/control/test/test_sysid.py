"""control.sysid and fit_vehicle on synthetic driving with a known steering gain and delay; the recorder node."""
import math
import os
import random
import re
import signal
import subprocess
import time

import numpy as np
import pytest

from control import fit_vehicle
from control.sysid import COLUMNS, analyze, estimate_delay, fit_gain, load_samples, resample, steady

L = 0.33
GAIN, DELAY = 1.8, 0.08
KFF, KA = 0.04, 0.05   # throttle = KFF x speed + KA x acceleration
HERE = os.path.dirname(os.path.abspath(__file__))
VEHICLE = os.path.join(HERE, '..', 'config', 'vehicle.yaml')


def synthetic(gain=GAIN, delay=DELAY, duration=120.0, dt=0.01, seed=0, compensate=1.0):
    """A car whose path curvature is gain x tan(applied steering) / L, the applied steering the /drive command
    delay s late (divided through atan(tan() / compensate), as the adapter's steering_gain does). The command:
    steering held for 0.5-1.5 s at a time, moving between holds at 3 rad/s; speed 1-7 m/s."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, dt)
    cmd = np.zeros_like(t)
    target, until = 0.0, 0.0
    for i in range(1, len(t)):
        if t[i] >= until:
            target, until = rng.uniform(-0.45, 0.45), t[i] + rng.uniform(0.5, 1.5)
        cmd[i] = cmd[i - 1] + np.clip(target - cmd[i - 1], -3.0 * dt, 3.0 * dt)
    lag = int(round(delay / dt))
    applied = np.arctan(np.tan(np.concatenate([np.zeros(lag), cmd[:-lag]])) / compensate)
    speed = 4.0 + 3.0 * np.sin(2 * np.pi * t / 23.0)
    accel = 3.0 * 2 * np.pi / 23.0 * np.cos(2 * np.pi * t / 23.0)
    yaw_rate = gain * speed * np.tan(applied) / L * (1 + rng.normal(0, 0.02, len(t)))
    return {'t': t, 'delta_cmd': cmd, 'v_cmd': speed, 'a_cmd': np.zeros_like(t), 'delta_applied': applied,
            'throttle': KFF * speed + KA * accel, 'vx': speed, 'vy': np.zeros_like(t), 'yaw_rate': yaw_rate}


def test_gain_and_delay_recovered():
    r = analyze(synthetic(), L)
    assert r['delta_applied']['gain'] == pytest.approx(GAIN, rel=0.02)
    assert r['delta_cmd']['gain'] == pytest.approx(GAIN, rel=0.03)   # no compensation: the same
    assert all(g == pytest.approx(GAIN, rel=0.05) for _, _, g, n in r['delta_applied']['bands'] if g is not None)
    assert r['delay_s'] == pytest.approx(DELAY, abs=0.01) and r['delay_corr'] > 0.9
    assert r['lateral_p99'] > 3.0 and r['speed_max'] == pytest.approx(7.0, abs=0.05)
    assert r['throttle'] is None   # no speed_kff given


def test_throttle_feedforward_recovered():
    ka, n_a, kb, n_b = analyze(synthetic(), L, kff=KFF)['throttle']
    assert ka == pytest.approx(KA, rel=0.05) and kb == pytest.approx(KA, rel=0.05)
    assert n_a > 100 and n_b > 100


def test_compensated_steering_reads_calibrated_on_the_command():
    r = analyze(synthetic(compensate=GAIN), L)
    assert r['delta_applied']['gain'] == pytest.approx(GAIN, rel=0.02)   # the car's own gain, unchanged
    assert r['delta_cmd']['gain'] == pytest.approx(1.0, rel=0.03)


def test_fit_gain_leaves_out_straight_and_slow_samples():
    _, (g, n) = fit_gain(np.full(50, 0.01), np.full(50, 3.0), np.zeros(50), L)
    assert g is None and n == 0
    _, (g, n) = fit_gain(np.full(50, 0.2), np.full(50, 0.5), np.ones(50), L)   # below 1 m/s
    assert g is None and n == 0
    bands, (g, n) = fit_gain(np.full(30, 0.2), np.full(30, 2.5), np.full(30, 2.5 * 1.5 * math.tan(0.2) / L), L)
    assert g == pytest.approx(1.5) and n == 30
    assert [b for b in bands if b[2] is not None] == [(2.0, 3.0, pytest.approx(1.5), 30)]


def test_steady_needs_the_steering_held():
    t = np.arange(0.0, 2.0, 0.01)
    delta = np.where(t < 1.0, 0.2, 0.2 + (t - 1.0))   # held, then turning
    ok = steady(t, delta, hold=0.3, tolerance=0.02)
    assert not ok[:29].any() and ok[30:100].all()        # the first 0.3 s cannot be judged
    assert not ok[130:].any()


def test_resample_masks_gaps():
    t = np.concatenate([np.arange(0.0, 1.0, 0.01), np.arange(2.0, 3.0, 0.01)])   # a 1 s gap
    grid, (v,) = resample(t, [np.sin(t)], 0.01, max_gap=0.1)
    inside_gap = (grid > 1.15) & (grid < 1.85)
    assert np.isnan(v[inside_gap]).all() and np.isfinite(v[grid < 0.98]).all()


def test_delay_of_a_constant_command_is_unknown():
    t = np.arange(0.0, 10.0, 0.01)
    assert estimate_delay(t, np.ones_like(t), np.sin(t))[0] is None


def write_csv(path, samples):
    with open(path, 'w') as f:
        f.write('# sysid_recorder: test\n' + ','.join(COLUMNS) + '\n')
        for row in zip(*(samples[c] for c in COLUMNS)):
            f.write(','.join(f'{v:.6f}' for v in row) + '\n')


def test_load_samples_round_trip(tmp_path):
    s = synthetic(duration=5.0)
    write_csv(tmp_path / 'r.csv', s)
    got = load_samples(str(tmp_path / 'r.csv'))
    assert set(got) == set(COLUMNS)
    assert np.allclose(got['yaw_rate'], s['yaw_rate'], atol=1e-5)


def test_fit_vehicle_prints_the_settings(tmp_path, capsys):
    write_csv(tmp_path / 'r.csv', synthetic())
    assert fit_vehicle.main([str(tmp_path / 'r.csv'), '--vehicle', VEHICLE, '--kff', str(KFF)]) == 0
    out = capsys.readouterr().out
    number = lambda pattern: float(re.search(pattern, out).group(1))  # noqa: E731
    assert number(r'steering_gain:=([0-9.]+)') == pytest.approx(GAIN, rel=0.02)
    assert number(r'latency:=([0-9.]+)') == pytest.approx(DELAY, abs=0.01)
    assert number(r'speed_ka:=([0-9.]+)') == pytest.approx(KA, rel=0.05)
    assert 'off by 1.8' in out   # recorded without the adapter's compensation


def test_fit_vehicle_on_the_car(tmp_path, capsys):
    s = synthetic()
    s['delta_applied'][:] = np.nan   # the car reports no applied steering, nor throttle
    s['throttle'][:] = np.nan
    write_csv(tmp_path / 'r.csv', s)
    assert fit_vehicle.main([str(tmp_path / 'r.csv'), '--vehicle', VEHICLE]) == 0
    out = capsys.readouterr().out
    assert 'steering_gain:=' not in out and 'speed_ka' not in out
    assert float(re.search(r'steering_angle_to_servo_gain by ([0-9.]+)', out).group(1)) == pytest.approx(GAIN, rel=0.03)


def test_recorder_node(tmp_path):
    rclpy = pytest.importorskip('rclpy')
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float32
    domain = random.randint(1, 101)
    out = tmp_path / 'rec.csv'
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain))
    proc = subprocess.Popen(['ros2', 'run', 'control', 'sysid_recorder_node', '--ros-args', '--params-file', VEHICLE,
                             '-p', f'out:={out}'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    rclpy.init(domain_id=domain)
    node = rclpy.create_node('sysid_test')
    try:
        drive = node.create_publisher(AckermannDriveStamped, '/drive', 10)
        steer = node.create_publisher(Float32, '/autodrive/roboracer_1/steering', 10)
        throttle = node.create_publisher(Float32, '/autodrive/roboracer_1/throttle', 10)
        gt = node.create_publisher(Odometry, '/ground_truth/odom', 10)
        end = time.time() + 20.0
        while min(p.get_subscription_count() for p in (drive, steer, throttle, gt)) == 0:
            assert proc.poll() is None and time.time() < end, proc.stdout.read() if proc.poll() is not None else ''
            rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.3)
        d = AckermannDriveStamped()
        d.drive.steering_angle, d.drive.speed, d.drive.acceleration = 0.2, 3.0, 1.0
        drive.publish(d)
        steer.publish(Float32(data=0.5))
        throttle.publish(Float32(data=0.25))
        time.sleep(0.3)
        for k in range(5):
            o = Odometry()
            o.header.stamp.sec = 100 + k
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z = 3.0, 0.1, 1.2
            gt.publish(o)
            time.sleep(0.05)
        time.sleep(0.5)
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        log = proc.communicate(timeout=10)[0]
        node.destroy_node()
        rclpy.shutdown()
    assert 'fit them with' in log, log
    s = load_samples(str(out))
    assert len(s['t']) == 5 and s['t'][0] == 100.0
    assert s['delta_cmd'][-1] == pytest.approx(0.2) and s['a_cmd'][-1] == pytest.approx(1.0)
    with open(VEHICLE) as f:
        max_steer = float(f.read().split('max_steering_angle:')[1].split()[0])
    assert s['delta_applied'][-1] == pytest.approx(0.5 * max_steer)
    assert s['throttle'][-1] == pytest.approx(0.25)
    assert s['vy'][-1] == pytest.approx(0.1) and s['yaw_rate'][-1] == pytest.approx(1.2)
