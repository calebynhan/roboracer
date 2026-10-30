"""A kinematic bicycle car for closed-loop tests of pure_pursuit_node: /drive in, /localized_pose and /odom out.

Not a stand-in for AutoDRIVE (no tires, no slip): it reproduces what pure pursuit's geometry does not model, so
the options that are meant to fix that can be measured:
  delay          /drive commands act this many s after they arrive (lidar, localization, actuation)
  steering_rate  rad/s the wheels turn at most (the sim: 0.524 rad in 0.164 s, racecar.ini t_delta)
  steering_gain  path curvature = steering_gain x tan(applied angle) / wheelbase (the repo measured ~1.8 in the sim)
  compensate     the adapter's steering_gain: the applied angle is atan(tan(command) / compensate)
  speed_tau      s, first-order lag of the speed to the command
Real time: pure_pursuit_node checks pose ages against its clock.
"""
import math
import time
from collections import deque

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class SimCar:
    def __init__(self, node, pose, wheelbase=0.33, max_steer=0.524, delay=0.0, steering_rate=3.2, steering_gain=1.0,
                 compensate=1.0, speed_tau=0.2, pose_rate=50.0):
        self.node = node
        self.x, self.y, self.yaw = pose
        self.v, self.delta = 0.0, 0.0
        self.L, self.max_steer = wheelbase, max_steer
        self.delay, self.rate, self.gain = delay, steering_rate, steering_gain
        self.comp, self.tau = compensate, speed_tau
        self.period = 1.0 / pose_rate
        self.cmd = (0.0, 0.0)              # (steering, speed) acting now
        self.queue = deque()               # (time it acts, steering, speed)
        self.drives = 0
        self.pose_pub = node.create_publisher(PoseStamped, '/localized_pose', 10)
        self.odom_pub = node.create_publisher(Odometry, '/odom', 10)
        node.create_subscription(AckermannDriveStamped, '/drive', self.drive_cb, 10)

    def drive_cb(self, msg):
        steer = max(-self.max_steer, min(self.max_steer, float(msg.drive.steering_angle)))
        if self.comp != 1.0:
            steer = math.atan(math.tan(steer) / self.comp)
        self.queue.append((time.monotonic() + self.delay, steer, float(msg.drive.speed)))
        self.drives += 1

    def publish(self):
        now = self.node.get_clock().now().to_msg()
        p = PoseStamped()
        p.header.stamp, p.header.frame_id = now, 'map'
        p.pose.position.x, p.pose.position.y = self.x, self.y
        p.pose.orientation.z, p.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = now, 'odom', 'base_link'
        o.twist.twist.linear.x = self.v
        o.twist.twist.angular.z = self.v * self.gain * math.tan(self.delta) / self.L
        self.odom_pub.publish(o)   # before the pose: pure pursuit predicts with the newest /odom
        self.pose_pub.publish(p)

    def step(self, dt):
        t = time.monotonic()
        while self.queue and self.queue[0][0] <= t:
            _, steer, speed = self.queue.popleft()
            self.cmd = (steer, speed)
        steer, speed = self.cmd
        self.delta += max(-self.rate * dt, min(self.rate * dt, steer - self.delta))
        self.v += (speed - self.v) * min(1.0, dt / self.tau)
        k = self.gain * math.tan(self.delta) / self.L
        dyaw = self.v * k * dt
        self.x += self.v * dt * math.cos(self.yaw + 0.5 * dyaw)
        self.y += self.v * dt * math.sin(self.yaw + 0.5 * dyaw)
        self.yaw = math.atan2(math.sin(self.yaw + dyaw), math.cos(self.yaw + dyaw))

    def run(self, secs, on_step=None, dt=0.002):
        """Drive for secs s of wall-clock time; on_step(t) after each physics step."""
        start = last_pub = time.monotonic()
        prev = start
        while (t := time.monotonic()) - start < secs:
            self.step(t - prev)
            prev = t
            if t - last_pub >= self.period:
                last_pub = t
                self.publish()
            rclpy.spin_once(self.node, timeout_sec=0)
            if on_step:
                on_step(t - start)
            time.sleep(dt)
