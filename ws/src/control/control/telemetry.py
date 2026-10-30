"""Lap telemetry for lap_logger_node and lap_summary (no ROS, so the tests drive it directly).

  Track        the raceline as a closed polyline: where a pose is along it (arc length s, signed offset, + left)
  LapCounter   laps from that arc length: a lap ends where the car crosses s = 0 forward, after driving at least
               min_fraction of a lap since the last one (jitter at the line counts once). The crossing time is
               interpolated between the samples around it. A teleport (a sim respawn or reset: a step no car can
               drive) never crosses the line and marks the lap as interrupted. A crossing starts timing only
               after the car has driven run_up of a lap since the counter started or the last teleport, so a car
               reset to just behind the line, which launches from rest over it, is not timed until the next lap:
               every lap logged is a flying lap.
  LapStats     what one lap's samples add up to: one laps.csv row, and one laps_sectors.csv row per sector
  Clearance    distance from the car's footprint to the nearest mapped wall
  summarize    laps.csv runs -> one line each (lap_summary)
"""
import math

import numpy as np
from scipy.ndimage import map_coordinates

LAP_COLUMNS = ('lap', 'lap_time_s', 'sim_lap_time_s', 'valid', 'collisions', 'lat_rms_m', 'lat_max_m',
               'clearance_min_m', 'speed_mean_mps', 'speed_max_mps', 'speed_err_rms_mps', 'loc_rms_m', 'loc_max_m',
               'loc_yaw_rms_deg', 'samples')
SECTOR_COLUMNS = ('lap', 'sector', 's_start_m', 's_end_m', 'time_s', 'lat_max_m', 'clearance_min_m',
                  'speed_mean_mps')


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Track:
    """A closed raceline (xy: n x 2 in driving order, no repeated closing point) and where poses are along it.

    locate() searches the segments within window m of arc length around the last answer, so a pose cannot jump
    to another stretch of track that passes close by (iros2026: two legs 1.07 m apart, driven in opposite
    directions); farther than reacquire m from that answer (the first pose, a teleport), it searches the
    segments heading within 90 deg of the car."""

    def __init__(self, xy, window=2.0, reacquire=1.0):
        xy = np.asarray(xy, dtype=float)
        if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        seg = np.roll(xy, -1, axis=0) - xy
        keep = np.hypot(seg[:, 0], seg[:, 1]) > 1e-9
        self.a = xy[keep]
        self.ab = (np.roll(self.a, -1, axis=0) - self.a)
        self.seg_len = np.hypot(self.ab[:, 0], self.ab[:, 1])
        if len(self.a) < 3:
            raise ValueError('a raceline needs at least 3 distinct points')
        self.s0 = np.concatenate(([0.0], np.cumsum(self.seg_len)[:-1]))
        self.length = float(self.seg_len.sum())
        self.heading = np.arctan2(self.ab[:, 1], self.ab[:, 0])
        step = float(np.median(self.seg_len))
        self.window = max(1, int(math.ceil(window / step)))
        self.reacquire = reacquire
        self.last = None

    def _nearest(self, idx, p):
        a, ab = self.a[idx], self.ab[idx]
        t = np.clip(np.einsum('ij,ij->i', p - a, ab) / self.seg_len[idx] ** 2, 0.0, 1.0)
        foot = a + t[:, None] * ab
        d = np.hypot(p[0] - foot[:, 0], p[1] - foot[:, 1])
        k = int(np.argmin(d))
        return idx[k], t[k], d[k]

    def locate(self, x, y, yaw=None):
        """(s [m along the line], offset [m, + left of it], segment index) of the point (x, y)."""
        p = np.array([x, y], dtype=float)
        n = len(self.a)
        found = None
        if self.last is not None:
            near = (self.last + np.arange(-self.window, self.window + 1)) % n
            found = self._nearest(near, p)
            if found[2] > self.reacquire:
                found = None
        if found is None:
            idx = np.arange(n)
            if yaw is not None:
                along = np.flatnonzero(np.cos(self.heading - yaw) > 0)
                idx = along if along.size else idx
            found = self._nearest(idx, p)
        i, t, d = found
        self.last = int(i)
        ab = self.ab[i]
        cross = ab[0] * (p[1] - self.a[i, 1]) - ab[1] * (p[0] - self.a[i, 0])
        return float(self.s0[i] + t * self.seg_len[i]), float(d if cross >= 0 else -d), int(i)


class LapCounter:
    """Laps from the arc length s along a track of the given length (see the module docstring).

    update(t, s) returns None, ('start', t_cross) where timing starts, or ('lap', t_cross, lap_time, interrupted,
    timing) at the end of each lap (timing: the next lap is timed too, False after a teleport: the car is not up to
    speed). max_step: m between two samples beyond which the car teleported."""

    def __init__(self, length, min_fraction=0.5, max_step=1.5, run_up=0.5):
        self.length = float(length)
        self.min_fraction = min_fraction
        self.max_step = max_step
        self.run_up = run_up * self.length
        self.prev = None          # (t, s)
        self.start = None         # time the current lap started, None while not timing
        self.progress = 0.0       # m driven forward since then
        self.driven = 0.0         # m driven forward since the counter started or the last teleport
        self.interrupted = False

    def update(self, t, s):
        prev, self.prev = self.prev, (t, s)
        if prev is None:
            return None
        ds = s - prev[1]
        crossed = ds < -0.5 * self.length       # past the end of the line onto its start: forward over s = 0
        if crossed:
            ds += self.length
        elif ds > 0.5 * self.length:            # backward over s = 0
            ds -= self.length
        if abs(ds) > self.max_step:
            self.interrupted = True
            self.driven = 0.0
            return None
        self.progress += ds
        self.driven += ds
        if not crossed:
            return None
        # interpolated to where the car crossed s = 0
        t_cross = prev[0] + (t - prev[0]) * (self.length - prev[1]) / ds if ds > 0 else t
        flying = self.driven >= self.run_up
        if self.start is None:
            if flying:
                self.start, self.progress, self.interrupted = t_cross, s, False
                return ('start', t_cross)
            return None
        if self.progress < self.min_fraction * self.length:
            return None
        lap = ('lap', t_cross, t_cross - self.start, self.interrupted, flying)
        self.start = t_cross if flying else None
        self.progress, self.interrupted = s, False
        return lap


def _rms(v):
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else math.nan


def _max_abs(v):
    return float(np.max(np.abs(v))) if len(v) else math.nan


class LapStats:
    """The samples of one lap. add() per pose of the car (sector = its sector index), add_localization() per
    localization error; rows() at the lap's end."""

    def __init__(self, n_sectors):
        self.n = n_sectors
        self.lat, self.speed, self.speed_err, self.loc, self.loc_yaw = [], [], [], [], []
        self.clearance = math.inf
        self.sec_time = np.zeros(n_sectors)
        self.sec_lat = np.zeros(n_sectors)
        self.sec_clear = np.full(n_sectors, math.inf)
        self.sec_speed = np.zeros(n_sectors)
        self.sec_count = np.zeros(n_sectors, dtype=int)

    def add(self, sector, dt, lateral, speed, speed_cmd=None, clearance=None):
        self.lat.append(lateral)
        self.speed.append(speed)
        if speed_cmd is not None:
            self.speed_err.append(speed - speed_cmd)
        self.sec_time[sector] += dt
        self.sec_lat[sector] = max(self.sec_lat[sector], abs(lateral))
        self.sec_speed[sector] += speed
        self.sec_count[sector] += 1
        if clearance is not None:
            self.clearance = min(self.clearance, clearance)
            self.sec_clear[sector] = min(self.sec_clear[sector], clearance)

    def add_localization(self, position_error, yaw_error):
        self.loc.append(position_error)
        self.loc_yaw.append(yaw_error)

    def rows(self, lap, lap_time, sim_lap_time, collisions, interrupted, length):
        """(laps.csv row, [laps_sectors.csv rows]) as dicts. valid: no teleport and no collision in the lap."""
        nan_if_inf = lambda v: v if math.isfinite(v) else math.nan  # noqa: E731
        row = {
            'lap': lap, 'lap_time_s': lap_time, 'sim_lap_time_s': sim_lap_time,
            'valid': int(not interrupted and collisions <= 0), 'collisions': collisions,
            'lat_rms_m': _rms(self.lat), 'lat_max_m': _max_abs(self.lat),
            'clearance_min_m': nan_if_inf(self.clearance),
            'speed_mean_mps': float(np.mean(self.speed)) if self.speed else math.nan,
            'speed_max_mps': float(np.max(self.speed)) if self.speed else math.nan,
            'speed_err_rms_mps': _rms(self.speed_err), 'loc_rms_m': _rms(self.loc), 'loc_max_m': _max_abs(self.loc),
            'loc_yaw_rms_deg': math.degrees(_rms(self.loc_yaw)) if self.loc_yaw else math.nan,
            'samples': len(self.lat),
        }
        edges = np.linspace(0.0, length, self.n + 1)
        sectors = [{
            'lap': lap, 'sector': k, 's_start_m': float(edges[k]), 's_end_m': float(edges[k + 1]),
            'time_s': float(self.sec_time[k]), 'lat_max_m': float(self.sec_lat[k]) if self.sec_count[k] else math.nan,
            'clearance_min_m': nan_if_inf(float(self.sec_clear[k])),
            'speed_mean_mps': float(self.sec_speed[k] / self.sec_count[k]) if self.sec_count[k] else math.nan,
        } for k in range(self.n)]
        return row, sectors


def format_row(row, columns):
    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return str(int(v))
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return 'nan' if not math.isfinite(v) else f'{v:.4f}'
    return ','.join(fmt(row[k]) for k in columns)


class Clearance:
    """Distance [m] from the car's footprint to the nearest mapped wall, from the distance field of a
    perception.mcl.LikelihoodMap (cell centre to nearest occupied cell centre; minus half a cell to the cell's
    edge). The footprint: a rectangle front m ahead of and rear m behind base_link (the rear axle), width m wide,
    sampled at its corners and edge midpoints. <= 0: the body is on a wall cell (touching it, at the map's
    resolution)."""

    def __init__(self, lmap, front=0.42, rear=0.08, width=0.27):
        self.lmap = lmap
        xs, ys = (-rear, 0.5 * (front - rear), front), (-0.5 * width, 0.0, 0.5 * width)
        self.points = np.array([(x, y) for x in xs for y in ys if not (x == xs[1] and y == 0.0)])

    def __call__(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        px = x + c * self.points[:, 0] - s * self.points[:, 1]
        py = y + s * self.points[:, 0] + c * self.points[:, 1]
        m = self.lmap
        col = (px - m.ox) / m.res - 0.5
        row = m.h - 1 - ((py - m.oy) / m.res - 0.5)
        d = map_coordinates(m.dist, [row, col], order=1, mode='nearest')
        return float(d.min() - 0.5 * m.res)


def read_runs(path):
    """laps.csv -> [(run comment line, [row dicts])], in file order."""
    runs, header = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('# run'):
                runs.append((line[2:], []))
            elif line.startswith('#'):
                continue
            elif header is None:
                header = line.split(',')
            else:
                if not runs:
                    runs.append(('run (no header line)', []))
                vals = line.split(',')
                runs[-1][1].append({k: float(v) for k, v in zip(header, vals)})
    return runs


def summarize(rows):
    """One run's laps -> dict: laps, valid laps, mean / best valid lap time, collisions, worst lateral error,
    smallest clearance, localization RMS (over the laps that have them)."""
    valid = [r for r in rows if r.get('valid', 1) == 1]
    times = [r['lap_time_s'] for r in valid]

    def finite(key):
        return [r[key] for r in rows if key in r and math.isfinite(r[key])]

    loc = finite('loc_rms_m')
    lat, clear = finite('lat_max_m'), finite('clearance_min_m')
    return {
        'laps': len(rows), 'valid': len(valid),
        'mean_s': float(np.mean(times)) if times else math.nan, 'best_s': float(min(times)) if times else math.nan,
        'collisions': int(sum(max(0.0, r.get('collisions', 0.0)) for r in rows)),
        'lat_max_m': max(lat) if lat else math.nan, 'clearance_min_m': min(clear) if clear else math.nan,
        'loc_rms_m': float(np.sqrt(np.mean(np.square(loc)))) if loc else math.nan,
    }
