"""Steering gain identification (numpy only; steering_monitor feeds it from the sim).

The kinematic bicycle model says a steering angle delta drives a path of curvature tan(delta) / wheelbase.
The measured curvature is yaw rate / speed (sim ground truth). The gain

    g = measured curvature / model curvature

is fitted per speed band, through the origin, from steady samples only (steering held, the car not
sliding through a transient). g > 1: the car turns harder than the model (wheel_odom.py measured the
model reading ~55 % of the sim's yaw rate, g ~ 1.8). adapter.yaml's steering_gain takes g: the adapter
then commands atan(tan(delta) / g), so a /drive angle drives the model's curvature.
"""
import math

import numpy as np


def model_curvature(delta, wheelbase):
    return np.tan(delta) / wheelbase


def fit_gain(delta, speed, yaw_rate, wheelbase, bands=(1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 12.0),
             min_delta=0.03, min_samples=20):
    """Least-squares g (through the origin) per speed band [lo, hi) and over all samples.

    delta [rad], speed [m/s], yaw_rate [rad/s]: equal-length arrays of steady samples. Samples below
    min_delta (straight driving, curvature mostly noise) or below the first band edge are left out.
    Returns [(lo, hi, g, n)], g None for a band with fewer than min_samples, and (g_all, n_all)."""
    delta, speed, yaw_rate = (np.asarray(a, dtype=float) for a in (delta, speed, yaw_rate))
    ok = (np.abs(delta) >= min_delta) & (speed >= bands[0]) & np.isfinite(delta + speed + yaw_rate)
    k0 = model_curvature(delta[ok], wheelbase)
    k = yaw_rate[ok] / speed[ok]
    v = speed[ok]

    def gain(mask):
        n = int(mask.sum())
        if n < min_samples:
            return None, n
        return float(np.dot(k[mask], k0[mask]) / np.dot(k0[mask], k0[mask])), n

    rows = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        g, n = gain((v >= lo) & (v < hi))
        rows.append((lo, hi, g, n))
    return rows, gain(np.ones_like(v, dtype=bool))


def compensate(delta, gain):
    """The angle to command so a car with this steering gain drives tan(delta) / wheelbase."""
    if gain is None or gain <= 0 or not math.isfinite(gain) or gain == 1.0:
        return delta
    return math.atan(math.tan(delta) / gain)


class SteadyFilter:
    """Keeps a sample only when the steering has been held within tolerance for hold seconds."""

    def __init__(self, hold=0.3, tolerance=0.02):
        self.hold, self.tolerance = hold, tolerance
        self.ref, self.since = None, None

    def steady(self, t, delta):
        if self.ref is None or abs(delta - self.ref) > self.tolerance:
            self.ref, self.since = delta, t
            return False
        return t - self.since >= self.hold
