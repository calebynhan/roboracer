"""Emergency braking from the lidar alone (safety_node): is anything in the car's path within stopping distance?

The path is an arc of the curvature the car is about to drive (the /drive steering command: tan(steering) /
wheelbase, the same curvature every controller here assumes), swept by the car's width plus a margin. A lidar
return is in the way when it lies within half_width + margin of that arc and ahead of the front bumper by less
than the stopping distance

    stop = speed x reaction + speed^2 / (2 x decel) + margin_ahead

so a wall beside the car, or one a curve turns away from, is not. min_points returns are needed, so a single
spurious one does not stop the car. This is a last line of defence against a wrong pose or a stuck controller,
not obstacle avoidance: at racing speed in narrow corridors a short reaction and a realistic decel keep it from
braking for walls the car is about to turn past.
"""
import math

import numpy as np


def scan_points(ranges, angle_min, angle_increment, laser_pose, range_min=0.0, range_max=math.inf):
    """Valid returns of a LaserScan as base_link points (n x 2). laser_pose: (x, y, yaw) of the lidar in base_link."""
    r = np.asarray(ranges, dtype=float)
    a = angle_min + np.arange(len(r)) * angle_increment + laser_pose[2]
    ok = np.isfinite(r) & (r > range_min) & (r < range_max)
    return np.column_stack((laser_pose[0] + r[ok] * np.cos(a[ok]), laser_pose[1] + r[ok] * np.sin(a[ok])))


def along_arc(points, curvature):
    """(s, d) of base_link points relative to the arc base_link drives from the origin along +x with curvature
    (+ left): s the arc length to the point's projection (negative behind), d its distance from the arc (+ left)."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if abs(curvature) < 1e-6:
        return p[:, 0], p[:, 1]
    sign = 1.0 if curvature > 0 else -1.0
    radius = 1.0 / abs(curvature)
    x, y = p[:, 0], sign * p[:, 1]                    # a right turn mirrored into a left one
    theta = np.arctan2(x, radius - y)                 # angle round the centre (0, radius), from the car
    d = radius - np.hypot(x, radius - y)
    return radius * theta, sign * d


def stopping_distance(speed, decel, reaction, margin):
    v = max(0.0, speed)
    return v * reaction + v * v / (2.0 * decel) + margin


def in_the_way(points, curvature, speed, half_width=0.135, margin=0.05, front=0.42, decel=4.0, reaction=0.1,
               margin_ahead=0.15, min_points=2):
    """(braking needed, distance ahead of the bumper to the nearest return in the swept path or inf)."""
    s, d = along_arc(points, curvature)
    ahead = s - front
    stop = stopping_distance(speed, decel, reaction, margin_ahead)
    hit = (np.abs(d) <= half_width + margin) & (ahead >= 0.0) & (ahead <= stop)
    if np.count_nonzero(hit) < min_points:
        return False, math.inf
    return True, float(np.min(ahead[hit]))
