"""Pure pursuit's speed profile and pose prediction (no ROS, so the tests drive it directly).

  speed profile   the raceline's vx x speed_scale, x a speed scale per sector (tracks/<name>/sectors.yaml, written
                  by tune_sectors) where enabled. Where the sector scales change, the profile is brought back within
                  the raceline's own acceleration and braking (the largest it plans): a forward pass accelerates out
                  of each slower stretch, a backward pass brakes into it, twice round the loop.
  swerve cap      with obstacle avoidance: the speed /local_path's curvature allows where it leaves the raceline,
                  sqrt(max_lateral_accel / kappa), and braking into it from behind at max_decel
  prediction      where the car will be when a command acts (latency compensation): driven forward with the
                  commands already sent but not yet acting (a Smith predictor), or at its speed and yaw rate
"""
import math
import os

import numpy as np
import yaml

SECTORS_FILE = 'sectors.yaml'


def arc_lengths(xy):
    """Closed polyline xy (n x 2, no repeated closing point): segment lengths i -> i + 1 (the last closes the
    loop), arc length of each point, lap length."""
    seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
    return seg, np.concatenate(([0.0], np.cumsum(seg)[:-1])), float(seg.sum())


def limit_accelerations(v, ds, a_pos, a_neg, passes=2):
    """v (closed loop, per point; ds[i] the distance to point i + 1) lowered where reaching it would take more
    than a_pos accelerating or a_neg braking: v[i+1]^2 <= v[i]^2 + 2 a_pos ds[i] forward, v[i]^2 <= v[i+1]^2 +
    2 a_neg ds[i] backward. Never raised."""
    v = np.array(v, dtype=float)
    n = len(v)
    for _ in range(passes):
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2.0 * a_pos * ds[i]))
    for _ in range(passes):
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * a_neg * ds[i]))
    return v


def profile_accelerations(v, ds):
    """The acceleration [m/s^2] from each point to the next along a closed speed profile."""
    v = np.asarray(v, dtype=float)
    return (np.roll(v, -1) ** 2 - v ** 2) / (2.0 * np.maximum(ds, 1e-9))


def sector_scales(s, length, scales):
    """The scale of each point: scales[k] for the k-th of len(scales) equal sectors of the lap from s = 0."""
    n = len(scales)
    k = np.minimum((np.asarray(s) / length * n).astype(int), n - 1)
    return np.asarray(scales, dtype=float)[k]


def speed_profile(vx, ds, s, length, speed_scale, scales=None, a_pos=None, a_neg=None):
    """The drive speed per raceline point: max(vx, 0) x speed_scale, x the sector scales if given (then limited
    to a_pos / a_neg where they change it; defaults: the largest acceleration and braking of vx itself)."""
    v = np.maximum(np.asarray(vx, dtype=float), 0.0) * speed_scale
    if scales is None or np.allclose(scales, 1.0):
        return v
    a = profile_accelerations(np.maximum(vx, 0.0), ds)
    a_pos = a_pos if a_pos is not None else max(1.0, float(a.max()))
    a_neg = a_neg if a_neg is not None else max(1.0, float(-a.min()))
    return limit_accelerations(v * sector_scales(s, length, scales), ds, a_pos, a_neg)


def swerve_cap(kappa, shifted, s, length, max_lateral_accel, max_decel):
    """Speed cap per /local_path point: sqrt(max_lateral_accel / |kappa|) where the path is shifted off the
    raceline (shifted: bool per point), and braking into those points from behind at max_decel:
    min over them of sqrt(cap_j^2 + 2 max_decel d_ij), d_ij the distance along the loop from i on to j.
    Infinite where nothing ahead limits it."""
    k = np.abs(np.asarray(kappa, dtype=float))
    j = np.flatnonzero(np.asarray(shifted, dtype=bool) & (k > 1e-6))
    cap = np.full(len(k), np.inf)
    if not j.size:
        return cap
    s = np.asarray(s, dtype=float)
    d = (s[j][:, None] - s[None, :]) % length                       # (cap points, all points)
    return np.min(np.sqrt(max_lateral_accel / k[j][:, None] + 2.0 * max_decel * d), axis=0)


def predict_with_commands(pose, speed, yaw_rate, t0, t1, commands, latency, wheelbase, dt=0.01):
    """The pose at t1 from the pose at t0, driving the commands in flight: a command sent at t acts from
    t + latency (commands: (time sent, steering) in time order). Its curvature is tan(steering) / wheelbase (the
    calibrated car every controller here assumes); before the first command acts, the measured yaw rate over the
    speed. The speed is held (measured)."""
    x, y, yaw = pose
    times = [c[0] for c in commands]
    t = t0
    while t < t1 - 1e-9:
        h = min(dt, t1 - t)
        i = int(np.searchsorted(times, t + 0.5 * h - latency, side='right')) - 1   # the command acting then
        k = math.tan(commands[i][1]) / wheelbase if i >= 0 else (yaw_rate / speed if abs(speed) > 0.1 else 0.0)
        dyaw = speed * k * h
        x += speed * h * math.cos(yaw + 0.5 * dyaw)
        y += speed * h * math.sin(yaw + 0.5 * dyaw)
        yaw += dyaw
        t += h
    return x, y, math.atan2(math.sin(yaw), math.cos(yaw))


def read_sectors(path):
    """sectors.yaml -> (scales list, lap length the sectors were made for [m] or None)."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    scales = [float(v) for v in data.get('scales', [])]
    if not scales or not all(math.isfinite(v) and v > 0 for v in scales):
        raise ValueError(f'{path}: scales must be a non-empty list of positive numbers')
    length = data.get('length_m')
    return scales, (float(length) if length is not None else None)


def write_sectors(path, scales, length, comment=''):
    lines = ['# speed scale per sector (equal lengths of the raceline from s = 0); pure pursuit sector_speeds:=true',
             '# multiplies its speed_scale by them. Written by control tune_sectors.']
    lines += [f'# {ln}' for ln in comment.splitlines()]
    lines += [f'length_m: {length:.3f}', 'scales: [' + ', '.join(f'{v:.3f}' for v in scales) + ']']
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(tmp, path)
