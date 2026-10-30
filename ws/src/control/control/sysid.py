"""Vehicle identification from recorded driving (sysid_recorder_node's CSV; fit_vehicle prints the result).

Numbers the stack otherwise guesses, from any driving (wall follower laps, races, teleop): no special
manoeuvre, so no track time and no wall in the way.

  steering gain  g = measured path curvature / bicycle-model curvature tan(delta) / wheelbase, through the origin,
                 per speed band, from steady samples only (steering held for hold s). g > 1: the car turns
                 harder than the model says (wheel_odom.py measured the model reading ~55 % of the sim's yaw
                 rate, g ~ 1.8). Measured on the steering the car applied (the sim reports it), g is the car's own;
                 on the /drive command after a calibration, it is what is left of it (1 = calibrated).
  delay          s from a /drive steering command to the path curvature following it: the lag that maximizes
                 the correlation of commanded and measured curvature (actuation latency + steering slew)
  throttle       sim only: the throttle per m/s^2 of acceleration beyond the adapter's speed feedforward
                 (throttle - speed_kff x speed = speed_ka x acceleration, least squares, accelerating and braking
                 apart): the adapter's speed_ka / speed_ka_brake
  limits seen    the accelerations the car reached (percentiles, not limits: it may never have been pushed)
"""
import math

import numpy as np

COLUMNS = ('t', 'delta_cmd', 'v_cmd', 'a_cmd', 'delta_applied', 'throttle', 'vx', 'vy', 'yaw_rate')
SPEED_BANDS = (1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 12.0)


def load_samples(path):
    """sysid CSV (comment lines '#', then a header line) -> {column: array}, rows in time order."""
    header, rows = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if header is None:
                header = line.split(',')
                continue
            fields = line.split(',')
            try:   # a truncated last line (the recorder killed while writing) is skipped
                if len(fields) == len(header):
                    rows.append([float(v) for v in fields])
            except ValueError:
                pass
    if header is None:
        raise ValueError(f'{path}: no header line')
    data = np.array(rows, dtype=float).reshape(-1, len(header))
    order = np.argsort(data[:, header.index('t')], kind='stable')
    return {k: data[order, i] for i, k in enumerate(header)}


def steady(t, delta, hold=0.3, tolerance=0.02):
    """True where the steering has stayed within tolerance rad over the hold s before (and at) each sample."""
    t, delta = np.asarray(t, float), np.asarray(delta, float)
    ok = np.zeros(len(t), dtype=bool)
    j = 0
    for i in range(len(t)):
        while t[i] - t[j] > hold:
            j += 1
        if t[i] - t[0] < hold:
            continue
        window = delta[j:i + 1]
        ok[i] = np.isfinite(window).all() and window.max() - window.min() <= tolerance
    return ok


def fit_gain(delta, speed, yaw_rate, wheelbase, bands=SPEED_BANDS, min_delta=0.03, min_samples=20):
    """Least-squares g (through the origin) per speed band [lo, hi) and over all samples, from (steady) samples.

    Samples below min_delta rad (straight driving: the curvature is mostly noise) or below the first band edge
    are left out. Returns ([(lo, hi, g or None, n)], (g or None, n)); None for fewer than min_samples."""
    delta, speed, yaw_rate = (np.asarray(a, dtype=float) for a in (delta, speed, yaw_rate))
    ok = (np.abs(delta) >= min_delta) & (speed >= bands[0]) & np.isfinite(delta + speed + yaw_rate)
    k0 = np.tan(delta[ok]) / wheelbase
    k = yaw_rate[ok] / speed[ok]
    v = speed[ok]

    def gain(mask):
        n = int(mask.sum())
        if n < min_samples:
            return None, n
        return float(np.dot(k[mask], k0[mask]) / np.dot(k0[mask], k0[mask])), n

    rows = [(lo, hi) + gain((v >= lo) & (v < hi)) for lo, hi in zip(bands[:-1], bands[1:])]
    return rows, gain(np.ones(len(v), dtype=bool))


def resample(t, values, dt, max_gap=0.1):
    """values (arrays over t) on a uniform grid of step dt, linearly interpolated; NaN on the grid points more
    than max_gap s from a finite sample (no data there, or the value was NaN: not measurable)."""
    grid = np.arange(t[0], t[-1], dt)
    out = []
    for v in values:
        good = np.isfinite(v)
        if good.sum() < 2:
            out.append(np.full(len(grid), np.nan))
            continue
        tg = t[good]
        r = np.interp(grid, tg, v[good])
        i = np.searchsorted(tg, grid)
        last = len(tg) - 1
        near = np.minimum(np.abs(grid - tg[np.clip(i - 1, 0, last)]), np.abs(tg[np.clip(i, 0, last)] - grid))
        r[near > max_gap] = np.nan
        out.append(r)
    return grid, out


def estimate_delay(t, command, response, max_lag=0.4, dt=0.005):
    """The lag [s] in [0, max_lag] at which response best follows command (normalized cross-correlation,
    refined between grid steps), and that correlation. NaNs (no data, too slow to measure) are left out of
    each lag's sums. (None, 0) when the command does not vary."""
    _, (u, y) = resample(np.asarray(t, float), [np.asarray(command, float), np.asarray(response, float)], dt)
    lags = np.arange(0, int(round(max_lag / dt)) + 1)
    corr = np.full(len(lags), -np.inf)
    for i, lag in enumerate(lags):
        a, b = u[:len(u) - lag], y[lag:]
        good = np.isfinite(a) & np.isfinite(b)
        if good.sum() < 50:
            continue
        a, b = a[good] - a[good].mean(), b[good] - b[good].mean()
        den = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
        if den > 1e-12:
            corr[i] = float(np.dot(a, b)) / den
    if not np.isfinite(corr).any():
        return None, 0.0
    i = int(np.argmax(corr))
    lag = float(lags[i])
    if 0 < i < len(lags) - 1 and np.isfinite(corr[i - 1]) and np.isfinite(corr[i + 1]):
        c0, c1, c2 = corr[i - 1], corr[i], corr[i + 1]
        den = c0 - 2 * c1 + c2
        if den < 0:
            lag += 0.5 * (c0 - c2) / den
    return lag * dt, float(corr[i])


def accelerations(t, speed, dt=0.02, window=0.1):
    """Longitudinal acceleration [m/s^2] on a uniform grid: the speed smoothed over window s, differentiated."""
    grid, (v,) = resample(np.asarray(t, float), [np.asarray(speed, float)], dt)
    k = max(1, int(round(window / dt)))
    kernel = np.ones(k) / k
    vs = np.convolve(np.nan_to_num(v), kernel, mode='same')
    valid = np.convolve(np.isfinite(v).astype(float), kernel, mode='same') > 0.999   # a full window of data
    vs[~valid] = np.nan
    a = np.gradient(vs, dt)   # NaN wherever a neighbour is: no difference across an edge of the data
    return grid, vs, a


def fit_throttle(throttle, speed, accel, kff, min_accel=0.5, min_samples=50):
    """(ka accelerating, n, ka braking, n): throttle - kff x speed = ka x accel, least squares through the origin
    over the samples accelerating (accel > min_accel) and braking (accel < -min_accel) apart. None for fewer
    than min_samples."""
    throttle, speed, accel = (np.asarray(a, dtype=float) for a in (throttle, speed, accel))
    ok = np.isfinite(throttle + speed + accel) & (speed > 0.5)
    excess = throttle - kff * speed
    out = []
    for mask in (ok & (accel > min_accel), ok & (accel < -min_accel)):
        n = int(mask.sum())
        out += [float(np.dot(excess[mask], accel[mask]) / np.dot(accel[mask], accel[mask])) if n >= min_samples
                else None, n]
    return tuple(out)


def analyze(samples, wheelbase, hold=0.3, tolerance=0.02, max_lag=0.4, kff=None):
    """Everything fit_vehicle reports, as a dict (see the module docstring)."""
    t, vx, w = samples['t'], samples['vx'], samples['yaw_rate']
    speed = np.hypot(vx, np.nan_to_num(samples.get('vy', np.zeros_like(vx))))
    result = {'samples': len(t), 'duration_s': float(t[-1] - t[0]) if len(t) else 0.0}
    for name in ('delta_applied', 'delta_cmd'):
        delta = samples.get(name)
        if delta is None or not np.isfinite(delta).any():
            result[name] = None
            continue
        hold_ok = steady(t, delta, hold, tolerance)
        bands, overall = fit_gain(delta[hold_ok], speed[hold_ok], w[hold_ok], wheelbase)
        result[name] = {'bands': bands, 'gain': overall[0], 'n': overall[1]}
    moving = speed > 1.0
    k_meas = np.where(moving, w / np.maximum(speed, 1e-6), np.nan)
    k_cmd = np.where(moving, np.tan(samples['delta_cmd']) / wheelbase, np.nan)
    result['delay_s'], result['delay_corr'] = estimate_delay(t, k_cmd, k_meas, max_lag)
    grid, vs, a = accelerations(t, speed)
    result['throttle'] = None
    if kff is not None and 'throttle' in samples and np.isfinite(samples['throttle']).any():
        _, (thr,) = resample(t, [samples['throttle']], grid[1] - grid[0] if len(grid) > 1 else 0.02)
        thr = thr[:len(vs)]
        result['throttle'] = fit_throttle(thr, vs[:len(thr)], a[:len(thr)], kff)
    fast = vs > 1.0
    acc = a[fast & np.isfinite(a)]
    result['accel_p99'] = float(np.percentile(acc, 99)) if acc.size else math.nan
    result['brake_p99'] = float(-np.percentile(acc, 1)) if acc.size else math.nan
    ay = np.abs(speed * w)[moving & np.isfinite(w)]
    result['lateral_p99'] = float(np.percentile(ay, 99)) if ay.size else math.nan
    result['speed_max'] = float(np.nanmax(speed)) if len(speed) else math.nan
    if 'vy' in samples and np.isfinite(samples['vy']).any():
        slip = np.degrees(np.abs(np.arctan2(samples['vy'], np.maximum(np.abs(vx), 1e-6))))[moving]
        result['slip_p99_deg'] = float(np.nanpercentile(slip, 99)) if slip.size else math.nan
    return result
