#!/usr/bin/env python3
"""Emergency brake on the lidar (control/safety.py): a stop on /safety/drive when the car's path is blocked
within stopping distance, whatever the controller and the localization think.

    ros2 run control safety_node --ros-args --params-file <vehicle.yaml>   (car.launch.py; race.launch.py safety:=true)

Per /scan: the path is the arc of the newest /drive steering (younger than cmd_timeout s; else the /odom yaw rate
over speed), the speed /odom's. In the way (safety.in_the_way): a stop (speed 0, the commanded steering) on
/safety/drive, then one per scan until hold s after the path last was clear; /safety/active (Bool) says which.
The command overrides /drive: in the sim the adapter takes /safety/drive first (safety_timeout), on the car
f1tenth_system's ackermann_mux does, as an input above navigation (control/config/car_mux.yaml).
"""
import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

from control.safety import in_the_way, scan_points, stopping_distance


class SafetyNode(Node):
    def __init__(self):
        super().__init__('safety_node')
        p = self.declare_parameter
        self.wheelbase = p('wheelbase', Parameter.Type.DOUBLE).value
        if self.wheelbase is None:
            raise RuntimeError('wheelbase unset: load control/config/vehicle.yaml')
        self.half_width = p('half_width', 0.135).value      # m, half the car's width
        self.margin = p('margin', 0.05).value               # m added to it either side
        self.front = p('front', 0.42).value                 # m from base_link (rear axle) to the front bumper
        self.decel = p('decel', 4.0).value                  # m/s^2 the car surely brakes at
        self.reaction = p('reaction', 0.1).value            # s from a scan to braking (lidar, this node, actuation)
        self.margin_ahead = p('margin_ahead', 0.15).value   # m kept to the obstacle
        self.min_points = p('min_points', 2).value          # returns in the way needed to brake
        self.hold = p('hold', 0.5).value                    # s braking after the path is clear again
        self.cmd_timeout = p('cmd_timeout', 0.3).value      # s: an older /drive steering is not the path
        self.base_frame = p('base_frame', 'base_link').value
        self.steering = None        # (clock time, steering angle) of the newest /drive
        self.speed, self.yaw_rate = 0.0, 0.0
        self.last_hit = None        # clock time of the last scan with something in the way
        self.active = False
        self.laser = {}             # scan frame -> (x, y, yaw) in base_frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(LaserScan, p('scan_topic', '/scan').value, self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, p('odom_topic', '/odom').value, self.odom_cb, 10)
        self.create_subscription(AckermannDriveStamped, p('drive_topic', '/drive').value, self.drive_cb, 10)
        self.brake_pub = self.create_publisher(AckermannDriveStamped, p('safety_topic', '/safety/drive').value, 10)
        self.active_pub = self.create_publisher(Bool, '/safety/active', 10)
        self.get_logger().info(
            f'Emergency brake on /scan: {self.half_width + self.margin:.2f} m either side of the path, stopping '
            f'distance at {self.decel} m/s^2 + {self.reaction} s + {self.margin_ahead} m '
            f'({stopping_distance(5.0, self.decel, self.reaction, self.margin_ahead):.2f} m at 5 m/s)')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def drive_cb(self, msg):
        if math.isfinite(msg.drive.steering_angle):
            self.steering = (self.now(), float(msg.drive.steering_angle))

    def odom_cb(self, msg):
        v, w = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        if math.isfinite(v) and math.isfinite(w):
            self.speed, self.yaw_rate = v, w

    def laser_pose(self, frame):
        if frame not in self.laser:
            try:
                t = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
            except TransformException as e:
                self.get_logger().warning(f'waiting for TF {self.base_frame} -> {frame}: {e}',
                                          throttle_duration_sec=2.0)
                return None
            q = t.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.laser[frame] = (t.transform.translation.x, t.transform.translation.y, yaw)
        return self.laser[frame]

    def curvature(self):
        if self.steering is not None and self.now() - self.steering[0] <= self.cmd_timeout:
            return math.tan(self.steering[1]) / self.wheelbase
        return self.yaw_rate / self.speed if self.speed > 0.3 else 0.0

    def scan_cb(self, msg):
        laser = self.laser_pose(msg.header.frame_id)
        if laser is None:
            return
        points = scan_points(msg.ranges, msg.angle_min, msg.angle_increment, laser, msg.range_min, msg.range_max)
        hit, distance = in_the_way(points, self.curvature(), self.speed, self.half_width, self.margin, self.front,
                                   self.decel, self.reaction, self.margin_ahead, self.min_points)
        now = self.now()
        if hit:
            if not self.active:
                self.get_logger().warning(f'EMERGENCY BRAKE: something {distance:.2f} m ahead in the path at '
                                          f'{self.speed:.2f} m/s')
            self.last_hit = now
        active = self.last_hit is not None and now - self.last_hit <= self.hold
        if self.active and not active:
            self.get_logger().info('path clear: releasing the brake')
        self.active = active
        if active:
            brake = AckermannDriveStamped()
            brake.header.stamp = self.get_clock().now().to_msg()
            brake.header.frame_id = self.base_frame
            brake.drive.steering_angle = self.steering[1] if self.steering is not None else 0.0
            brake.drive.speed = 0.0
            self.brake_pub.publish(brake)
        self.active_pub.publish(Bool(data=active))


def main():
    rclpy.init()
    node = SafetyNode()
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
