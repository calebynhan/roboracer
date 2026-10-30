"""perception.imu_odom and imu_odom_node: wheel speed + gyro odometry, its standstill bias, a turned IMU."""
import math
import os
import random
import signal
import subprocess
import time

import pytest

from perception.imu_odom import GyroOdometry


def drive(o, t0, secs, speed, yaw_rate, bias=0.0, dt=0.01):
    t = t0
    while t < t0 + secs - 1e-9:
        t += dt
        o.set_speed(t, speed)
        o.gyro(t, yaw_rate + bias)
    return t


def test_straight_and_circle():
    o = GyroOdometry()
    drive(o, 0.0, 2.0, 2.0, 0.0)
    assert (o.x, o.y, o.yaw) == pytest.approx((4.0 - 0.02, 0.0, 0.0), abs=1e-6)   # the first sample starts it
    o = GyroOdometry()
    drive(o, 0.0, math.pi + 0.01, 1.0, 1.0)   # half a circle of radius 1 m, turning left
    assert o.x == pytest.approx(0.0, abs=0.03) and o.y == pytest.approx(2.0, abs=0.03)
    assert abs(o.yaw) == pytest.approx(math.pi, abs=0.02)


def test_bias_learned_at_standstill_and_heading_held():
    bias = 0.02   # rad/s: ~70 deg of drift in a minute
    o = GyroOdometry(still_time=1.0)
    t = drive(o, 0.0, 3.0, 0.0, 0.0, bias)
    assert o.yaw == pytest.approx(0.0, abs=0.025)   # at most the first still_time s drift, then held
    assert o.bias == pytest.approx(bias, abs=1e-6)
    yaw0 = o.yaw
    drive(o, t, 10.0, 2.0, 0.0, bias)               # driving straight: the bias no longer turns it
    assert o.yaw == pytest.approx(yaw0, abs=1e-9)


def test_nothing_before_a_wheel_speed():
    o = GyroOdometry()
    assert not o.gyro(0.0, 1.0)
    o.set_speed(0.0, 1.0)
    assert not o.gyro(0.01, 1.0) and o.x == 0.0   # the first gyro sample only starts the clock
    assert o.gyro(0.02, 1.0) and o.x == pytest.approx(0.01, abs=1e-6)


def test_out_of_order_gyro_sample_dropped():
    o = GyroOdometry()
    o.set_speed(0.0, 1.0)
    o.gyro(1.0, 0.0)
    assert o.gyro(1.1, 0.0)
    x = o.x
    assert not o.gyro(1.05, 0.0) and not o.gyro(1.1, 0.0)   # older, the same: nothing published
    assert o.x == x and o.t == 1.1
    assert o.gyro(1.2, 0.0) and o.x == pytest.approx(x + 0.1)   # integrated once from 1.1


def test_imu_odom_node():
    rclpy = pytest.importorskip('rclpy')
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from tf2_ros import StaticTransformBroadcaster

    domain = random.randint(1, 101)
    proc = subprocess.Popen(['ros2', 'run', 'perception', 'imu_odom_node'],
                            env=dict(os.environ, ROS_DOMAIN_ID=str(domain)), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    rclpy.init(domain_id=domain)
    node = rclpy.create_node('imu_odom_test')
    got = []
    try:
        # the IMU upside down: rolled 180 deg, so its z axis points down and it reads the yaw rate negated
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'base_link', 'imu'
        tf.transform.rotation.x, tf.transform.rotation.w = 1.0, 0.0
        static = StaticTransformBroadcaster(node)
        static.sendTransform(tf)
        wheel = node.create_publisher(Odometry, '/odom/wheel', 10)
        imu = node.create_publisher(Imu, '/sensors/imu/raw', 10)
        node.create_subscription(Odometry, '/odom', got.append, 50)
        end = time.time() + 20.0
        while wheel.get_subscription_count() == 0 or imu.get_subscription_count() == 0:
            assert proc.poll() is None and time.time() < end, proc.stdout.read() if proc.poll() is not None else ''
            rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(1.0)   # TF
        t = 100.0
        for k in range(300):   # 3 s (stamps) at 1 m/s turning left at 0.5 rad/s
            t += 0.01
            w = Odometry()
            w.header.stamp.sec, w.header.stamp.nanosec = int(t), int(round((t % 1) * 1e9)) % 1_000_000_000
            w.twist.twist.linear.x = 1.0
            wheel.publish(w)
            m = Imu()
            m.header = w.header
            m.header.frame_id = 'imu'
            m.angular_velocity.z = -0.5   # upside down
            imu.publish(m)
            time.sleep(0.003)
            rclpy.spin_once(node, timeout_sec=0)
        end = time.time() + 1.0
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        out = proc.communicate(timeout=10)[0]
        node.destroy_node()
        rclpy.shutdown()
    # the test publishes faster than an IMU would, so some samples drop (sensor-data QoS); the steps are timed
    # by the stamps, so the yaw still adds up
    assert len(got) > 100, out
    first, last = got[0], got[-1]
    stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9  # noqa: E731
    yaw = 2 * math.atan2(last.pose.pose.orientation.z, last.pose.pose.orientation.w)
    assert last.twist.twist.angular.z == pytest.approx(0.5)
    assert yaw == pytest.approx(0.5 * (stamp(last) - stamp(first) + 0.01), abs=0.05)   # turning left, not right
    assert last.header.frame_id == 'odom' and last.child_frame_id == 'base_link'
