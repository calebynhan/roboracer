#!/usr/bin/env python3
"""Lap benchmark: one row per flying lap (and one per sector of it), so every change to the stack is a number.

    ros2 run control lap_logger_node --ros-args -p track:=<name> [-p tag:=<label>] [-p laps:=0]
    (ws/scripts/benchmark.sh runs it next to race.sh and prints the result)

Where the car is comes from the sim's ground truth (/ground_truth/odom, through tracks/<name>/map_to_world.yaml,
which save_map writes when mapping in the sim) or, without that file or with pose_source:=localized (the real
car), from pose_topic (/localized_pose) with the speed from /odom. The pose is placed on the raceline
(control/telemetry.py Track): arc length s and the signed distance from the line. A lap ends where the car
crosses s = 0 (the start of raceline.csv, near the pose mapping started from) forward. Timing starts at a
crossing after the car has driven half a lap since the logger started or since the last sim respawn or reset (the
pose jumping, which also marks the lap it happened in as interrupted), so every lap logged is a flying lap: a car
reset to just behind the line launches from rest over it, and that lap is not timed. laps > 0: exit after that
many laps.

Appends to tracks/<name>/laps.csv (one row per lap; a '# run ...' comment line starts each run, with the tag and,
last, run_args: the race's launch args, which benchmark.sh passes and tune_sectors checks):
  lap_time_s         crossing to crossing, interpolated between poses; sim_lap_time_s: the sim's own last lap
                     (its timing line may sit elsewhere), nan without the sim's restricted topics
  valid              1 unless the lap was interrupted or had a collision
  collisions         the sim's collision_count over the lap, -1 if unknown
  lat_rms_m, lat_max_m    distance from the raceline (RMS, worst)
  clearance_min_m    smallest distance from the car's footprint to a mapped wall (Clearance; <= 0: touching)
  speed_mean_mps, speed_max_mps, speed_err_rms_mps (speed - the /drive speed: what the speed loop does not follow)
  loc_rms_m, loc_max_m, loc_yaw_rms_deg   pose_topic against ground truth at the same stamps (sim only)
and to tracks/<name>/laps_sectors.csv one row per lap and sector (sectors: equal lengths of the raceline from
s = 0): time_s, lat_max_m, clearance_min_m, speed_mean_mps (control's tune_sectors reads it).
"""
import datetime
import math
import os
from collections import deque

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32, Int32

from perception.mcl import LikelihoodMap
from perception.tracks import read_map_to_world, track_dir, world_to_map
from planning.raceline import read_raceline

from control.telemetry import (LAP_COLUMNS, SECTOR_COLUMNS, Clearance, LapCounter, LapStats, Track, format_row,
                               wrap)

MAX_DT = 0.2          # s: a longer gap between poses (a stalled stream) is not counted as time in a sector
MAX_CMD_AGE = 0.2     # s: an older /drive speed is not compared with the car's
GT_HISTORY = 1.0      # s of ground truth kept to compare the localized poses against, at their stamps


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LapLogger(Node):
    def __init__(self):
        super().__init__('lap_logger')
        p = self.declare_parameter
        track = p('track', '').value
        d = track_dir(track)
        raceline = p('raceline_path', '').value or os.path.join(d, 'raceline.csv')
        out_dir = p('out_dir', '').value or d
        source = p('pose_source', 'auto').value             # auto | ground_truth | localized
        pose_topic = p('pose_topic', '/localized_pose').value
        ns = p('vehicle_ns', '/autodrive/roboracer_1').value
        self.n_sectors = int(p('sectors', 10).value)
        self.max_laps = int(p('laps', 0).value)
        tag = p('tag', '').value
        run_args = p('run_args', '').value   # the race's launch args (benchmark.sh), for the run line
        if self.n_sectors < 1:
            raise ValueError(f'sectors must be >= 1, got {self.n_sectors}')
        if not os.path.isfile(raceline):
            raise FileNotFoundError(f'{raceline} not found (set track or raceline_path)')

        self.track = Track(read_raceline(raceline)[:, 1:3])
        self.counter = LapCounter(self.track.length)
        self.origin = read_map_to_world(track) if track else None
        if source == 'auto':
            source = 'ground_truth' if self.origin is not None else 'localized'
        if source == 'ground_truth' and self.origin is None:
            raise FileNotFoundError(f'pose_source ground_truth needs {d}/map_to_world.yaml (mapped in the sim)')
        if source not in ('ground_truth', 'localized'):
            raise ValueError(f"pose_source must be auto, ground_truth or localized, got '{source}'")
        self.source = source
        map_yaml = os.path.join(d, 'map.yaml')
        self.clearance = None
        if os.path.isfile(map_yaml):
            self.clearance = Clearance(LikelihoodMap.from_yaml(map_yaml),
                                       front=p('footprint_front', 0.42).value, rear=p('footprint_rear', 0.08).value,
                                       width=p('footprint_width', 0.27).value)

        self.stats = None            # LapStats of the lap in progress (None before the first crossing)
        self.prev_t = None
        self.laps = 0
        self.drive = None            # (clock time, speed) of the newest /drive
        self.odom_speed = None       # localized source: the speed from /odom
        self.gt = deque()            # (stamp, x, y, yaw) in the map frame, for the localization error
        self.collisions = None       # the sim's collision_count, latest
        self.lap_collisions = None   # ... at the start of the lap in progress
        self.sim_last_lap = math.nan

        self.laps_csv = os.path.join(out_dir, 'laps.csv')
        self.sectors_csv = os.path.join(out_dir, 'laps_sectors.csv')
        run = (f'# run {datetime.datetime.now().astimezone().isoformat(timespec="seconds")} tag={tag or "-"} '
               f'track={track or "-"} source={source} sectors={self.n_sectors} raceline={raceline}'
               + (f' args={run_args}' if run_args else '') + '\n')
        for path, columns in ((self.laps_csv, LAP_COLUMNS), (self.sectors_csv, SECTOR_COLUMNS)):
            new = not os.path.isfile(path)
            with open(path, 'a') as f:
                if new:
                    f.write(','.join(columns) + '\n')
                f.write(run)

        if source == 'ground_truth':
            self.create_subscription(Odometry, '/ground_truth/odom', self.gt_cb, 50)
        else:
            self.create_subscription(Odometry, p('odom_topic', '/odom').value, self.odom_cb, 50)
        self.create_subscription(PoseStamped, pose_topic, self.pose_cb, 50)
        self.create_subscription(AckermannDriveStamped, '/drive', self.drive_cb, 10)
        self.create_subscription(Int32, ns + '/collision_count', self.collision_cb, 10)
        self.create_subscription(Float32, ns + '/last_lap_time', self.last_lap_cb, 10)
        if source == 'ground_truth':
            self.gt_check = self.create_timer(5.0, self.check_ground_truth)
        self.get_logger().info(
            f'Logging laps of {raceline} ({self.track.length:.1f} m, {self.n_sectors} sectors) from '
            f'{"ground truth" if source == "ground_truth" else pose_topic} to {self.laps_csv}'
            + (f'; stopping after {self.max_laps} laps' if self.max_laps else '')
            + ('' if self.clearance else '; no map.yaml: no wall clearance'))

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def check_ground_truth(self):
        self.gt_check.cancel()
        if not self.gt:
            self.get_logger().warning('no /ground_truth/odom for 5 s: the bridge runs with publish_restricted_topics '
                                      'false? Then use -p pose_source:=localized')

    def drive_cb(self, msg):
        self.drive = (self.now(), float(msg.drive.speed))

    def odom_cb(self, msg):
        v = msg.twist.twist.linear
        self.odom_speed = math.hypot(v.x, v.y)

    def collision_cb(self, msg):
        self.collisions = int(msg.data)
        if self.lap_collisions is None and self.stats is not None:
            self.lap_collisions = self.collisions

    def last_lap_cb(self, msg):
        self.sim_last_lap = float(msg.data)

    def gt_cb(self, msg):
        pose = msg.pose.pose
        x, y, yaw = world_to_map(self.origin, pose.position.x, pose.position.y, yaw_of(pose.orientation))
        t = stamp_s(msg.header.stamp)
        self.gt.append((t, x, y, yaw))
        while self.gt and self.gt[0][0] < t - GT_HISTORY:
            self.gt.popleft()
        self.sample(t, x, y, yaw, math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y))

    def pose_cb(self, msg):
        if msg.header.frame_id != 'map':
            return
        t = stamp_s(msg.header.stamp)
        x, y, yaw = msg.pose.position.x, msg.pose.position.y, yaw_of(msg.pose.orientation)
        if self.source == 'localized':
            if self.odom_speed is not None:
                self.sample(t, x, y, yaw, self.odom_speed)
            return
        if self.stats is None:
            return
        truth = self.gt_at(t)
        if truth is not None:
            self.stats.add_localization(math.hypot(x - truth[0], y - truth[1]), abs(wrap(yaw - truth[2])))

    def gt_at(self, t):
        """Ground truth (x, y, yaw) at stamp t, interpolated; None outside the history (no extrapolation)."""
        if len(self.gt) < 2 or not self.gt[0][0] <= t <= self.gt[-1][0]:
            return None
        stamps = [g[0] for g in self.gt]
        i = max(1, int(np.searchsorted(stamps, t)))
        (t0, x0, y0, a0), (t1, x1, y1, a1) = self.gt[i - 1], self.gt[i]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return x0 + u * (x1 - x0), y0 + u * (y1 - y0), a0 + u * wrap(a1 - a0)

    def sample(self, t, x, y, yaw, speed):
        prev_t, self.prev_t = self.prev_t, t
        if prev_t is not None and t <= prev_t:
            return   # the same frame again (or time went back: a restarted bag)
        s, offset, _ = self.track.locate(x, y, yaw)
        event = self.counter.update(t, s)
        if event is not None and event[0] == 'lap' and self.stats is not None:
            self.finish(event[2], event[3])
        if event is not None:
            timing = event[0] == 'start' or event[4]
            self.stats = LapStats(self.n_sectors) if timing else None
            self.lap_collisions = self.collisions
            if event[0] == 'start':
                self.get_logger().info('crossed the start at speed: timing laps')
            elif not timing:
                self.get_logger().info('reset or respawn in that lap: the next one is not timed (not at speed)')
        if self.stats is None:
            return
        dt = t - prev_t if prev_t is not None and t - prev_t <= MAX_DT else 0.0
        cmd = self.drive[1] if self.drive is not None and self.now() - self.drive[0] <= MAX_CMD_AGE else None
        sector = min(self.n_sectors - 1, int(s / self.track.length * self.n_sectors))
        clear = self.clearance(x, y, yaw) if self.clearance is not None else None
        self.stats.add(sector, dt, offset, speed, cmd, clear)

    def finish(self, lap_time, interrupted):
        self.laps += 1
        collisions = (self.collisions - self.lap_collisions
                      if self.collisions is not None and self.lap_collisions is not None else -1)
        row, sectors = self.stats.rows(self.laps, lap_time, self.sim_last_lap, collisions, interrupted,
                                       self.track.length)
        with open(self.laps_csv, 'a') as f:
            f.write(format_row(row, LAP_COLUMNS) + '\n')
        with open(self.sectors_csv, 'a') as f:
            for r in sectors:
                f.write(format_row(r, SECTOR_COLUMNS) + '\n')
        flag = '' if row['valid'] else ' (NOT VALID: ' + ('interrupted' if interrupted else 'collision') + ')'
        self.get_logger().info(
            f"lap {self.laps}: {lap_time:.3f} s{flag}, lateral error rms {row['lat_rms_m']:.3f} / max "
            f"{row['lat_max_m']:.3f} m, wall clearance min {row['clearance_min_m']:.3f} m, speed "
            f"{row['speed_mean_mps']:.2f} (max {row['speed_max_mps']:.2f}) m/s, speed error rms "
            f"{row['speed_err_rms_mps']:.2f} m/s, localization rms {row['loc_rms_m']:.3f} m")
        if self.max_laps and self.laps >= self.max_laps:
            self.get_logger().info(f'{self.laps} laps logged to {self.laps_csv}')
            raise SystemExit(0)


def main():
    rclpy.init()
    node = LapLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:   # Ctrl-C under ros2 launch sends a second SIGINT during cleanup
            pass


if __name__ == '__main__':
    main()
