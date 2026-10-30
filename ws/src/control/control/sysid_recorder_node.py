#!/usr/bin/env python3
"""Records what vehicle identification needs, during any driving: fit_vehicle then fits the steering gain, the
actuation delay and the accelerations seen (control/sysid.py).

    ros2 run control sysid_recorder_node --ros-args --params-file <vehicle.yaml> [-p source:=ground_truth|odom]
        [-p drive_topic:=/drive] [-p out:=<file.csv>]

One CSV row per measurement message (columns sysid.COLUMNS):
  t              its stamp [s]
  delta_cmd, v_cmd, a_cmd   the command (drive_topic, /drive; on the car driven by joystick: /teleop) received
                 last before that stamp: steering angle, speed, acceleration (NaN before the first). By the stamp,
                 not when the measurement arrived, so its transport delay is not counted as the car's
  delta_applied  the steering the sim applied (vehicle_ns/steering, -1..1, x max_steering_angle), NaN on the car
  throttle       the throttle the sim applied (vehicle_ns/throttle), NaN on the car
  vx, vy, yaw_rate         the car's velocity in base_link and yaw rate: source ground_truth (the default, the
                 sim only) is /ground_truth/odom; source odom is /odom (sim: wheel_odom, encoders + gyro; the car:
                 the gyro-fused odometry of car.launch.py, so use source:=odom there; vy is then NaN). Nothing on
                 it for 5 s: a warning
out: default ~/ws/log/sysid_<date>_<time>.csv. Stop with Ctrl-C; then: ros2 run control fit_vehicle <out>
"""
import bisect
import datetime
import math
import os
from collections import deque

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32

from control.sysid import COLUMNS


class SysidRecorder(Node):
    def __init__(self):
        super().__init__('sysid_recorder')
        p = self.declare_parameter
        self.source = p('source', 'ground_truth').value
        ns = p('vehicle_ns', '/autodrive/roboracer_1').value
        self.max_steer = p('max_steering_angle', Parameter.Type.DOUBLE).value
        if self.max_steer is None:
            raise RuntimeError('max_steering_angle unset: load control/config/vehicle.yaml')
        if self.source not in ('ground_truth', 'odom'):
            raise ValueError(f"source must be ground_truth or odom, got '{self.source}'")
        out = p('out', '').value or os.path.expanduser(
            f'~/ws/log/sysid_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv')
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        self.out = out
        self.f = open(out, 'w')
        self.f.write(f'# sysid_recorder: source={self.source} max_steering_angle={self.max_steer}\n'
                     + ','.join(COLUMNS) + '\n')
        self.cmds = deque(maxlen=500)   # (clock time received, (steering, speed, acceleration))
        self.cmd_t = []                 # their times, for bisect
        self.applied = math.nan
        self.throttle = math.nan
        self.rows = 0
        sim_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        drive_topic = p('drive_topic', '/drive').value
        self.create_subscription(AckermannDriveStamped, drive_topic, self.drive_cb, 10)
        self.create_subscription(Float32, ns + '/steering', self.steering_cb, sim_qos)
        self.create_subscription(Float32, ns + '/throttle', self.throttle_cb, sim_qos)
        topic = '/ground_truth/odom' if self.source == 'ground_truth' else p('odom_topic', '/odom').value
        self.create_subscription(Odometry, topic, self.odom_cb, 50)
        self.create_timer(10.0, self.report)
        self.topic = topic
        self.check = self.create_timer(5.0, self.check_input)
        self.get_logger().info(f'Recording {topic} and {drive_topic} to {out}')

    def check_input(self):
        self.check.cancel()
        if self.rows == 0:
            self.get_logger().warning(f'nothing on {self.topic} for 5 s' + (
                ': on the car use -p source:=odom (car.launch.py\'s odometry)' if self.source == 'ground_truth'
                else ''))

    def drive_cb(self, msg):
        d = msg.drive
        self.cmds.append((self.get_clock().now().nanoseconds * 1e-9,
                          (float(d.steering_angle), float(d.speed), float(d.acceleration))))
        self.cmd_t = [c[0] for c in self.cmds]

    def cmd_at(self, t):
        """The command received last at or before clock time t (the newest if t is outside their span: a stamp
        on another clock)."""
        if not self.cmds:
            return (math.nan, math.nan, math.nan)
        if not self.cmd_t[0] <= t <= self.cmd_t[-1] + 1.0:
            return self.cmds[-1][1]
        i = bisect.bisect_right(self.cmd_t, t) - 1
        return self.cmds[max(i, 0)][1]

    def steering_cb(self, msg):
        self.applied = float(msg.data) * self.max_steer

    def throttle_cb(self, msg):
        self.throttle = float(msg.data)

    def odom_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        v = msg.twist.twist.linear
        vy = v.y if self.source == 'ground_truth' else math.nan
        row = (t,) + self.cmd_at(t) + (self.applied, self.throttle, v.x, vy, msg.twist.twist.angular.z)
        self.f.write(','.join(f'{x:.6f}' if math.isfinite(x) else 'nan' for x in row) + '\n')
        self.rows += 1

    def report(self):
        self.f.flush()
        self.get_logger().info(f'{self.rows} samples in {self.out}')


def main():
    rclpy.init()
    node = SysidRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.f.close()
        print(f'{node.rows} samples in {node.out}; fit them with: ros2 run control fit_vehicle {node.out}')
        try:
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:   # Ctrl-C under ros2 launch sends a second SIGINT during cleanup
            pass


if __name__ == '__main__':
    main()
