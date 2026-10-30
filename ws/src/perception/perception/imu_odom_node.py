#!/usr/bin/env python3
"""Odometry for the real car: the VESC's wheel speed with the IMU's gyro for the yaw (perception/imu_odom.py).

    ros2 run perception imu_odom_node     (control car.launch.py starts it)

Subscribed: wheel_odom_topic (/odom/wheel: f1tenth_system's vesc_to_odom, remapped there from odom and with its
publish_tf false), imu_topic (/sensors/imu/raw). Published at the IMU's rate, stamped with its messages: /odom
(odom -> base_link, twist: wheel speed and the gyro's yaw rate less its bias) and, with publish_tf, TF
odom -> base_link, as vesc_to_odom did. The gyro is turned into base_link with TF base_link -> the IMU's frame
(car.launch.py publishes it from the mounting angles), so an IMU mounted upside down or turned still gives
base_link's yaw rate. The bias is learned while the car stands still (see imu_odom.py); its value is logged.
gyro_scale: rad/s per unit of the IMU's angular velocity, 1.0 for a driver that follows REP 103 (rad/s); a VESC
driver that reports deg/s needs pi / 180 = 0.0174533. Check it once: turn the car a full turn by hand and the
yaw in /odom must come back to where it started.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from perception.imu_odom import GyroOdometry


def rotation(q):
    """3x3 rotation matrix of the quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class ImuOdom(Node):
    def __init__(self):
        super().__init__('imu_odom')
        p = self.declare_parameter
        self.base_frame = p('base_frame', 'base_link').value
        self.publish_tf = p('publish_tf', True).value
        self.gyro_scale = p('gyro_scale', 1.0).value   # rad/s per IMU unit (pi / 180 for a driver in deg/s)
        self.odom = GyroOdometry(still_speed=p('still_speed', 0.02).value, still_time=p('still_time', 1.0).value,
                                 bias_alpha=p('bias_alpha', 0.02).value)
        self.imu_rot = {}             # IMU frame -> its rotation into base_link
        self.bias_logged = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, p('odom_topic', '/odom').value, 10)
        self.create_subscription(Odometry, p('wheel_odom_topic', '/odom/wheel').value, self.wheel_cb, 10)
        self.create_subscription(Imu, p('imu_topic', '/sensors/imu/raw').value, self.imu_cb, qos_profile_sensor_data)
        self.create_timer(5.0, self.log_bias)
        self.get_logger().info('Odometry: wheel speed + gyro yaw rate, waiting for the wheels and the IMU')

    def stamp(self, msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def wheel_cb(self, msg):
        v = msg.twist.twist.linear.x
        if math.isfinite(v):
            self.odom.set_speed(self.stamp(msg), v)

    def gyro_z(self, msg):
        frame = msg.header.frame_id or self.base_frame
        if frame not in self.imu_rot:
            if frame == self.base_frame:
                self.imu_rot[frame] = np.eye(3)
            else:
                try:
                    t = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
                except TransformException as e:
                    self.get_logger().warning(f'waiting for TF {self.base_frame} -> {frame}: {e}',
                                              throttle_duration_sec=2.0)
                    return None
                q = t.transform.rotation
                self.imu_rot[frame] = rotation((q.x, q.y, q.z, q.w))
        w = msg.angular_velocity
        return self.gyro_scale * float((self.imu_rot[frame] @ np.array([w.x, w.y, w.z]))[2])

    def imu_cb(self, msg):
        wz = self.gyro_z(msg)
        if wz is None or not math.isfinite(wz):
            return
        if not self.odom.gyro(self.stamp(msg), wz):
            return
        o = self.odom
        qz, qw = math.sin(o.yaw / 2.0), math.cos(o.yaw / 2.0)
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'odom'
        out.child_frame_id = self.base_frame
        out.pose.pose.position.x, out.pose.pose.position.y = o.x, o.y
        out.pose.pose.orientation.z, out.pose.pose.orientation.w = qz, qw
        out.twist.twist.linear.x = o.speed
        out.twist.twist.angular.z = o.yaw_rate
        self.pub.publish(out)
        if self.publish_tf:
            tf = TransformStamped()
            tf.header = out.header
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x, tf.transform.translation.y = o.x, o.y
            tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
            self.tf.sendTransform(tf)

    def log_bias(self):
        o = self.odom
        if o.bias_samples and o.bias != self.bias_logged:
            self.bias_logged = o.bias
            self.get_logger().info(f'gyro bias {math.degrees(o.bias):.3f} deg/s ({o.bias_samples} still samples)')


def main():
    rclpy.init()
    node = ImuOdom()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:   # Ctrl-C under ros2 launch sends a second SIGINT during cleanup
            pass


if __name__ == '__main__':
    main()
