#!/usr/bin/env python3
"""Steering gain from normal driving (sim only): how much harder or softer the car turns than the bicycle model.

    ros2 run autodrive_bridge steering_monitor --ros-args \
        --params-file $(ros2 pkg prefix control)/share/control/config/vehicle.yaml [-p out:=steering_id.csv]

Run it during any driving (wall follower laps, races, teleop circles): it needs no special manoeuvre. It pairs
the sim's applied steering (vehicle_ns/steering, -1..1, x max_steering_angle) with the ground-truth speed and
yaw rate (/ground_truth/odom), keeps the samples where the steering has been held for hold s, and every
report_period s logs the fitted gain g = measured curvature / (tan(delta) / wheelbase) per speed band
(steering_id.py). Put g into adapter.yaml's steering_gain. The steering it reads is what the sim applied, after
the adapter's compensation, so g is the car's own gain whatever steering_gain is set to. out: also write the
samples as CSV.
"""
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32

from autodrive_bridge.steering_id import SteadyFilter, fit_gain


class SteeringMonitor(Node):
    def __init__(self):
        super().__init__('steering_monitor')
        p = self.declare_parameter
        ns = p('vehicle_ns', '/autodrive/roboracer_1').value
        self.wheelbase = p('wheelbase', Parameter.Type.DOUBLE).value
        self.max_steer = p('max_steering_angle', Parameter.Type.DOUBLE).value
        if self.wheelbase is None or self.max_steer is None:
            raise RuntimeError('wheelbase / max_steering_angle unset: load control/config/vehicle.yaml')
        self.filter = SteadyFilter(p('hold', 0.3).value, p('tolerance', 0.02).value)
        out = p('out', '').value
        self.csv = open(out, 'w') if out else None
        if self.csv:
            self.csv.write('# t,delta_rad,speed_mps,yaw_rate_radps\n')
        self.delta = None
        self.samples = ([], [], [])
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Float32, ns + '/steering', self.steering_cb, qos)
        self.create_subscription(Odometry, '/ground_truth/odom', self.gt_cb, 10)
        self.create_timer(p('report_period', 10.0).value, self.report)

    def steering_cb(self, msg):
        self.delta = float(msg.data) * self.max_steer

    def gt_cb(self, msg):
        if self.delta is None:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        if not self.filter.steady(t, self.delta):
            return
        v, w = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        for lst, val in zip(self.samples, (self.delta, v, w)):
            lst.append(val)
        if self.csv:
            self.csv.write(f'{t:.4f},{self.delta:.5f},{v:.4f},{w:.5f}\n')

    def report(self):
        rows, (g_all, n_all) = fit_gain(*self.samples, self.wheelbase)
        bands = ', '.join(f'{lo:g}-{hi:g} m/s: ' + (f'{g:.2f} ({n})' if g is not None else f'- ({n})')
                          for lo, hi, g, n in rows if n)
        if g_all is None:
            self.get_logger().info(f'{n_all} steady turning samples so far, need more (drive some corners)')
            return
        self.get_logger().info(f'steering gain g = {g_all:.2f} over {n_all} samples; by speed: {bands}')
        if self.csv:
            self.csv.flush()


def main():
    rclpy.init()
    node = SteeringMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.report()
        if node.csv:
            node.csv.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
