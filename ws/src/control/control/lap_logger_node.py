#!/usr/bin/env python3
"""Lap benchmark (sim only): one CSV row per lap, so every change to the stack is a number.

    ros2 run control lap_logger_node --ros-args -p track:=<name> [-p tag:=<label>] [-p laps:=0]

Run it next to race.sh. Per lap (the sim's lap_count ticking over) it appends to tracks/<name>/laps.csv:
  lap_time_s         between two lap_count changes (ROS clock), sim_lap_time_s the sim's own last_lap_time
  collisions         collision_count change over the lap
  lat_rms_m, lat_max_m    ground-truth distance from the raceline (+ = left), RMS and worst
  speed_err_rms_mps  ground-truth speed - /drive speed (what the speed loop fails to follow)
  speed_mean_mps, speed_max_mps
  loc_rms_m, loc_max_m    /localized_pose - ground truth (localization error), when the localizer publishes
Ground truth (/ground_truth/odom, sim world frame) goes into the map frame with tracks/<name>/map_to_world.yaml
(save_map writes it while mapping in the sim). The first lap_count change only starts the first lap. A
comment line per run records the tag and the time, so runs of different settings can share the file.
laps > 0: exit after that many laps.
"""
import datetime
import math
import os

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32, Int32

from perception.tracks import track_dir
from planning.raceline import read_raceline

from control.telemetry import COLUMNS, LapStats, format_row, lateral_offset, read_map_to_world, world_to_map

MAX_LOC_AGE = 0.1   # s: a localized pose older than this is not compared with ground truth


class LapLogger(Node):
    def __init__(self):
        super().__init__('lap_logger')
        p = self.declare_parameter
        track = p('track', '').value
        ns = p('vehicle_ns', '/autodrive/roboracer_1').value
        d = track_dir(track)
        raceline = p('raceline_path', '').value or os.path.join(d, 'raceline.csv')
        self.out = p('out', '').value or os.path.join(d, 'laps.csv')
        tag = p('tag', '').value
        self.max_laps = p('laps', 0).value
        rl = read_raceline(raceline)
        self.xy = rl[:-1, 1:3] if np.allclose(rl[0, 1:3], rl[-1, 1:3]) else rl[:, 1:3]
        self.origin = read_map_to_world(os.path.join(d, 'map_to_world.yaml'))

        self.stats = LapStats()
        self.lap_count = None
        self.lap_start = None
        self.collisions = None
        self.lap_collisions0 = None
        self.sim_last_lap = float('nan')
        self.speed_cmd = None
        self.loc = None          # (t, x, y) of the newest /localized_pose
        self.logged = 0

        new = not os.path.isfile(self.out)
        with open(self.out, 'a') as f:
            if new:
                f.write(','.join(COLUMNS) + '\n')
            f.write(f'# run {datetime.datetime.now().astimezone().isoformat(timespec="seconds")} '
                    f'tag={tag or "-"} raceline={raceline}\n')

        self.create_subscription(Odometry, '/ground_truth/odom', self.gt_cb, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.drive_cb, 10)
        self.create_subscription(PoseStamped, p('pose_topic', '/localized_pose').value, self.pose_cb, 10)
        self.create_subscription(Int32, ns + '/lap_count', self.lap_cb, 10)
        self.create_subscription(Int32, ns + '/collision_count', self.collision_cb, 10)
        self.create_subscription(Float32, ns + '/last_lap_time', self.last_lap_cb, 10)
        self.get_logger().info(f'Logging laps on {raceline} to {self.out}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def drive_cb(self, msg):
        self.speed_cmd = msg.drive.speed

    def pose_cb(self, msg):
        if msg.header.frame_id == 'map':
            self.loc = (self.now(), msg.pose.position.x, msg.pose.position.y)

    def collision_cb(self, msg):
        self.collisions = msg.data

    def last_lap_cb(self, msg):
        self.sim_last_lap = float(msg.data)

    def gt_cb(self, msg):
        if self.lap_start is None:
            return
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        x, y, _ = world_to_map(self.origin, msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        v = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        loc_err = None
        if self.loc is not None and self.now() - self.loc[0] <= MAX_LOC_AGE:
            loc_err = math.hypot(self.loc[1] - x, self.loc[2] - y)
        self.stats.add(lateral_offset(self.xy, x, y), v, self.speed_cmd, loc_err)

    def lap_cb(self, msg):
        if msg.data == self.lap_count:
            return
        first = self.lap_count is None
        self.lap_count = msg.data
        t = self.now()
        if not first and self.lap_start is not None:
            collisions = (self.collisions - self.lap_collisions0
                          if self.collisions is not None and self.lap_collisions0 is not None else -1)
            self.logged += 1
            row = self.stats.row(self.logged, t - self.lap_start, self.sim_last_lap, collisions)
            with open(self.out, 'a') as f:
                f.write(format_row(row) + '\n')
            self.get_logger().info(
                f"lap {self.logged}: {row['lap_time_s']:.2f} s (sim {row['sim_lap_time_s']:.2f} s), "
                f"{collisions} collisions, lateral rms {row['lat_rms_m']:.3f} / max {row['lat_max_m']:.3f} m, "
                f"speed error rms {row['speed_err_rms_mps']:.2f} m/s, localization rms {row['loc_rms_m']:.3f} m")
        self.lap_start = t
        self.lap_collisions0 = self.collisions
        self.stats = LapStats()
        if self.max_laps > 0 and self.logged >= self.max_laps:
            raise SystemExit(0)


def main():
    rclpy.init()
    node = LapLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
