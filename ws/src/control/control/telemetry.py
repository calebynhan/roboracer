"""Per-lap telemetry for lap_logger_node: numpy only, so it is testable without ROS.

  world_to_map:    a sim ground-truth pose (world frame) into the map frame, with tracks/<name>/map_to_world.yaml
  lateral_offset:  signed distance of a point from a closed polyline (the raceline), + = left of it
  LapStats:        samples over one lap -> one CSV row (RMS / worst lateral error, speed error, localization error)
"""
import math

import numpy as np

COLUMNS = ('lap', 'lap_time_s', 'sim_lap_time_s', 'collisions', 'lat_rms_m', 'lat_max_m', 'speed_err_rms_mps',
           'speed_mean_mps', 'speed_max_mps', 'loc_rms_m', 'loc_max_m', 'samples')


def read_map_to_world(path):
    """map_to_world.yaml (save_map): the map origin's pose in the sim world frame, (x, y, yaw)."""
    vals = {}
    with open(path) as f:
        for line in f:
            if ':' in line and not line.lstrip().startswith('#'):
                k, v = line.split(':', 1)
                vals[k.strip()] = v.strip()
    return float(vals['x_m']), float(vals['y_m']), float(vals['yaw_rad'])


def world_to_map(origin, x, y, yaw):
    """A world-frame pose in the map frame, origin = the map origin's world pose (read_map_to_world)."""
    ox, oy, oyaw = origin
    c, s = math.cos(oyaw), math.sin(oyaw)
    dx, dy = x - ox, y - oy
    return c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(yaw - oyaw), math.cos(yaw - oyaw))


def lateral_offset(xy, x, y):
    """Signed distance [m] of (x, y) from the closed polyline xy (n x 2, the last point joins the first):
    the distance to the nearest segment, + when the point is left of it (driving direction)."""
    a = np.asarray(xy, dtype=float)
    b = np.roll(a, -1, axis=0)
    ab = b - a
    seg2 = np.einsum('ij,ij->i', ab, ab)
    p = np.array([x, y], dtype=float)
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(seg2, 1e-12), 0.0, 1.0)
    foot = a + t[:, None] * ab
    d = np.hypot(*(p - foot).T)
    i = int(np.argmin(d))
    cross = ab[i, 0] * (p[1] - a[i, 1]) - ab[i, 1] * (p[0] - a[i, 0])
    return float(math.copysign(d[i], cross) if cross != 0 else d[i])


def _rms(v):
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else float('nan')


def _max_abs(v):
    return float(np.max(np.abs(v))) if len(v) else float('nan')


class LapStats:
    """Samples of one lap. add() per ground-truth pose; row() at the lap's end."""

    def __init__(self):
        self.lat, self.speed_err, self.speed, self.loc = [], [], [], []

    def add(self, lateral, speed, speed_cmd=None, loc_err=None):
        self.lat.append(lateral)
        self.speed.append(speed)
        if speed_cmd is not None:
            self.speed_err.append(speed - speed_cmd)
        if loc_err is not None:
            self.loc.append(loc_err)

    def row(self, lap, lap_time, sim_lap_time, collisions):
        return {
            'lap': lap, 'lap_time_s': lap_time, 'sim_lap_time_s': sim_lap_time, 'collisions': collisions,
            'lat_rms_m': _rms(self.lat), 'lat_max_m': _max_abs(self.lat),
            'speed_err_rms_mps': _rms(self.speed_err),
            'speed_mean_mps': float(np.mean(self.speed)) if self.speed else float('nan'),
            'speed_max_mps': float(np.max(self.speed)) if self.speed else float('nan'),
            'loc_rms_m': _rms(self.loc), 'loc_max_m': _max_abs(self.loc), 'samples': len(self.lat),
        }


def format_row(row):
    return ','.join(f'{row[k]:.4f}' if isinstance(row[k], float) else str(row[k]) for k in COLUMNS)
