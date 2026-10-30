"""Lidar scan deskew: each beam where it would have been seen from where the car was at the scan's stamp.

A real lidar sweeps its beams over the scan period (the Hokuyo UST-10LX: 1081 beams in 25 ms), so at speed the
car moves while one scan is taken: at 7 m/s, 0.18 m between the first and the last beam, and the walls bend.
sensor_msgs/LaserScan stamps the first beam; beam i comes time_increment x i later. With the odometry's poses at
those times, every beam's end point goes into base_link as it was at the stamp. The sim ray-casts each scan at
one instant (time_increment 0): nothing changes there.
"""
import math

import numpy as np


def interpolate_poses(stamps, poses, t):
    """(x, y, yaw) arrays at times t, linearly interpolated between the poses (n x 3) at stamps (increasing);
    clamped at either end. The yaw is unwrapped first."""
    stamps = np.asarray(stamps, dtype=float)
    poses = np.asarray(poses, dtype=float)
    yaw = np.unwrap(poses[:, 2])
    return (np.interp(t, stamps, poses[:, 0]), np.interp(t, stamps, poses[:, 1]), np.interp(t, stamps, yaw))


def relative_poses(ref, x, y, yaw):
    """Poses (arrays) in the frame of the pose ref = (x, y, yaw): ref^-1 * pose."""
    c, s = math.cos(ref[2]), math.sin(ref[2])
    dx, dy = np.asarray(x) - ref[0], np.asarray(y) - ref[1]
    a = np.asarray(yaw) - ref[2]
    return c * dx + s * dy, -s * dx + c * dy, np.arctan2(np.sin(a), np.cos(a))


def deskew(ranges, angles, rel, sensor_pose):
    """Ranges and angles of beams seen from the sensor at the reference time, given for each beam the pose of
    base_link when it was taken relative to base_link at the reference time (rel: arrays x, y, yaw) and the
    sensor's pose in base_link."""
    r, a = np.asarray(ranges, dtype=float), np.asarray(angles, dtype=float)
    sx, sy, syaw = sensor_pose
    rx, ry, ryaw = rel
    # the end point in base_link at the beam's time
    bx = sx + r * np.cos(a + syaw)
    by = sy + r * np.sin(a + syaw)
    # into base_link at the reference time
    c, s = np.cos(ryaw), np.sin(ryaw)
    px, py = rx + c * bx - s * by, ry + s * bx + c * by
    # and back to the sensor there
    cs, ss = math.cos(syaw), math.sin(syaw)
    qx, qy = cs * (px - sx) + ss * (py - sy), -ss * (px - sx) + cs * (py - sy)
    return np.hypot(qx, qy), np.arctan2(qy, qx)
