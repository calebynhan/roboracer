"""control.safety and safety_node: braking for what is in the car's path, not for what is beside it."""
import math
import os
import random
import signal
import subprocess
import time

import numpy as np
import pytest

from control.safety import along_arc, in_the_way, scan_points, stopping_distance

HERE = os.path.dirname(os.path.abspath(__file__))
VEHICLE = os.path.join(HERE, '..', 'config', 'vehicle.yaml')


def test_along_arc_straight_and_curved():
    s, d = along_arc([[2.0, 0.3]], 0.0)
    assert (s[0], d[0]) == (2.0, 0.3)
    # left turn of radius 1: the point a quarter circle on is on the arc, pi/2 m along it
    s, d = along_arc([[1.0, 1.0], [1.2, 1.0]], 1.0)
    assert s[0] == pytest.approx(math.pi / 2) and d[0] == pytest.approx(0.0, abs=1e-9)
    assert d[1] == pytest.approx(-0.2)   # farther from the centre: outside = right of a left turn
    s, d = along_arc([[1.0, -1.0]], -1.0)   # the same, turning right
    assert s[0] == pytest.approx(math.pi / 2) and d[0] == pytest.approx(0.0, abs=1e-9)


def wall(x0, y_from, y_to, n=20):
    return np.column_stack([np.full(n, x0), np.linspace(y_from, y_to, n)])


def test_wall_ahead_within_stopping_distance():
    pts = wall(2.0, -0.5, 0.5)
    hit, dist = in_the_way(pts, 0.0, speed=4.0)            # stop = 0.4 + 2.0 + 0.15 = 2.55 m from the bumper
    assert hit and dist == pytest.approx(2.0 - 0.42)
    assert not in_the_way(wall(4.0, -0.5, 0.5), 0.0, speed=4.0)[0]   # 3.58 m ahead: not yet
    assert in_the_way(wall(4.0, -0.5, 0.5), 0.0, speed=6.0)[0]       # faster: yes


def test_wall_beside_or_turned_away_from_is_not_in_the_way():
    beside = np.column_stack([np.linspace(-0.5, 3.0, 40), np.full(40, 0.3)])   # 0.3 m left, along the car
    assert not in_the_way(beside, 0.0, speed=6.0)[0]
    # a wall 1.5 m straight ahead while the car turns left on a 1 m radius: off the arc
    assert in_the_way(wall(1.5, -0.3, 0.3), 0.0, speed=3.0)[0]
    assert not in_the_way(wall(1.5, -0.3, 0.3), 1.0, speed=3.0)[0]


def test_single_return_ignored():
    assert not in_the_way(np.array([[1.0, 0.0]]), 0.0, speed=3.0)[0]
    assert in_the_way(np.array([[1.0, 0.0], [1.0, 0.05]]), 0.0, speed=3.0)[0]


def test_behind_the_bumper_ignored():
    # returns between the rear axle and the bumper (inside the body's length) are the car's own or noise
    assert not in_the_way(wall(0.3, -0.1, 0.1), 0.0, speed=3.0)[0]


def test_scan_points_and_stopping_distance():
    pts = scan_points([1.0, math.inf, 2.0, 0.01], -math.pi / 2, math.pi / 2, (0.27, 0.0, 0.0), range_min=0.06)
    assert pts == pytest.approx(np.array([[0.27, -1.0], [0.27, 2.0]]))   # right, left (inf, too short: out)
    assert stopping_distance(4.0, 4.0, 0.1, 0.15) == pytest.approx(0.4 + 2.0 + 0.15)
    assert stopping_distance(-1.0, 4.0, 0.1, 0.15) == pytest.approx(0.15)


def test_safety_node():
    rclpy = pytest.importorskip('rclpy')
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool
    from tf2_ros import StaticTransformBroadcaster

    domain = random.randint(1, 101)
    proc = subprocess.Popen(['ros2', 'run', 'control', 'safety_node', '--ros-args', '--params-file', VEHICLE],
                            env=dict(os.environ, ROS_DOMAIN_ID=str(domain)), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    rclpy.init(domain_id=domain)
    node = rclpy.create_node('safety_test')
    brakes, active = [], []
    try:
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'base_link', 'laser'
        tf.transform.translation.x = 0.27
        tf.transform.rotation.w = 1.0
        static = StaticTransformBroadcaster(node)
        static.sendTransform(tf)
        scan_pub = node.create_publisher(LaserScan, '/scan', 10)
        odom_pub = node.create_publisher(Odometry, '/odom', 10)
        drive_pub = node.create_publisher(AckermannDriveStamped, '/drive', 10)
        node.create_subscription(AckermannDriveStamped, '/safety/drive', brakes.append, 10)
        node.create_subscription(Bool, '/safety/active', lambda m: active.append(m.data), 10)
        end = time.time() + 20.0
        while scan_pub.get_subscription_count() == 0 or odom_pub.get_subscription_count() == 0:
            assert proc.poll() is None and time.time() < end, proc.stdout.read() if proc.poll() is not None else ''
            rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.5)

        def scan(obstacle):
            s = LaserScan()
            s.header.frame_id = 'laser'
            s.header.stamp = node.get_clock().now().to_msg()
            s.angle_min, s.angle_max, s.angle_increment = -math.pi / 2, math.pi / 2, math.pi / 180
            s.range_min, s.range_max = 0.06, 10.0
            r = [5.0] * 181
            if obstacle:
                for k in range(85, 96):   # +-5 deg ahead, 1.5 m from the lidar
                    r[k] = 1.5
            s.ranges = r
            return s

        def spin_for(secs, obstacle, speed=3.0):
            end = time.time() + secs
            while time.time() < end:
                o = Odometry()
                o.twist.twist.linear.x = speed
                odom_pub.publish(o)
                d = AckermannDriveStamped()
                d.drive.speed, d.drive.steering_angle = speed, 0.0
                drive_pub.publish(d)
                scan_pub.publish(scan(obstacle))
                for _ in range(5):
                    rclpy.spin_once(node, timeout_sec=0.005)

        spin_for(1.0, obstacle=False)
        assert not brakes and active and not any(active)
        spin_for(0.5, obstacle=True)
        assert brakes and all(b.drive.speed == 0.0 for b in brakes) and active[-1]
        spin_for(0.2, obstacle=False)   # clear, but within hold (0.5 s): still braking
        assert active[-1]
        spin_for(1.0, obstacle=False)
        assert not active[-1]
        n = len(brakes)
        spin_for(0.5, obstacle=False)
        assert len(brakes) == n   # released: nothing more
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        out = proc.communicate(timeout=10)[0]
        node.destroy_node()
        rclpy.shutdown()
    assert 'EMERGENCY BRAKE' in out and 'releasing the brake' in out, out
