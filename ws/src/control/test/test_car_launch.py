"""car.launch.py starts the real car's additions (IMU transform, gyro odometry, emergency brake); the race launch
files take their new arguments."""
import os
import random
import signal
import subprocess
import time

import pytest

pytest.importorskip('rclpy')


def run(args, env, secs):
    proc = subprocess.Popen(['ros2', 'launch', 'control', *args], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    time.sleep(secs)
    os.killpg(proc.pid, signal.SIGINT)
    try:
        return proc.communicate(timeout=15)[0]
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        return proc.communicate()[0]


def test_car_launch_starts_its_nodes():
    env = dict(os.environ, ROS_DOMAIN_ID=str(random.randint(1, 101)), PYTHONUNBUFFERED='1')
    out = run(['car.launch.py', 'imu_roll:=3.14159'], env, 6.0)
    assert 'Odometry: wheel speed + gyro yaw rate' in out, out
    assert 'Emergency brake on /scan' in out, out
    # Ctrl-C (SIGINT to the process group, as here) and launch's own SIGINT: the nodes stop cleanly all the same
    for node in ('imu_odom_node', 'safety_node'):
        assert f'[{node}-' in out and 'Traceback' not in out, out
        assert any(f'[{node}-' in ln and 'finished cleanly' in ln for ln in out.splitlines()), out
    assert "from 'base_link' to 'imu'" in out, out
    out = run(['car.launch.py', 'safety:=false'], env, 4.0)
    assert 'Odometry: wheel speed' in out and 'Emergency brake' not in out, out


@pytest.mark.parametrize('launch_file, args', [
    ('race.launch.py', ['safety', 'latency', 'speed_preview_time', 'accel_feedforward', 'sector_speeds',
                        'avoid_max_lateral_accel', 'vehicle']),
    ('pure_pursuit.launch.py', ['latency', 'speed_preview_time', 'accel_feedforward', 'sector_speeds',
                                'avoid_max_lateral_accel', 'vehicle'])])
def test_launch_arguments(launch_file, args):
    out = subprocess.run(['ros2', 'launch', 'control', launch_file, '--show-args'], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    for a in args:
        assert f"'{a}':" in out.stdout, (a, out.stdout)
