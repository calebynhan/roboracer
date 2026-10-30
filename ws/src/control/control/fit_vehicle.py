#!/usr/bin/env python3
"""Fit the car from a sysid_recorder_node recording and print the settings it implies (control/sysid.py).

    ros2 run control fit_vehicle <recording.csv> [--vehicle <vehicle.yaml>] [--hold 0.3] [--max-lag 0.4]

Prints:
  the steering gain g (measured / bicycle-model curvature) per speed band, on the steering the sim applied and on
      the /drive command: the sim's adapter takes g as steering_gain; on the car, divide the VESC's
      steering_angle_to_servo_gain by it. g on the /drive command near 1: already calibrated.
  the delay from a /drive steering command to the path following it: pure pursuit's latency
  sim only: the adapter's acceleration feedforward speed_ka / speed_ka_brake (throttle per m/s^2 beyond its
      speed feedforward speed_kff, read from autodrive_bridge's adapter.yaml or --kff)
  the accelerations the driving reached, next to racecar.ini's planning limits (what was reached is a floor for
      what the car can do, not a ceiling)
vehicle.yaml (default: control's config/vehicle.yaml) gives the wheelbase.
"""
import argparse
import configparser
import json
import math
import os
import sys

import yaml

from control.sysid import SPEED_BANDS, analyze, load_samples


def share(package, *path):
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory(package), *path)
    except Exception:   # not installed: the source tree
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(os.path.dirname(here), package, *path)


def planning_limits(racecar):
    """The planning config's safe_traj limits and mue, for comparison (None if it cannot be read)."""
    ini = configparser.ConfigParser()
    if not ini.read(racecar):
        return None
    opts = json.loads(ini.get('OPTIMIZATION_OPTIONS', 'optim_opts_mintime'))
    return {k: opts.get(k) for k in ('ax_pos_safe', 'ax_neg_safe', 'ay_safe', 'mue')}


def adapter_kff():
    """The sim adapter's speed_kff (autodrive_bridge config/adapter.yaml), None if it cannot be read."""
    try:
        with open(share('autodrive_bridge', 'config', 'adapter.yaml')) as f:
            return float(yaml.safe_load(f)['adapter']['ros__parameters']['speed_kff'])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def gain_lines(name, fit):
    if fit is None:
        return [f'  {name}: not recorded']
    if fit['gain'] is None:
        return [f"  {name}: too few steady turning samples ({fit['n']}): drive more corners"]
    bands = '  '.join(f'{lo:g}-{hi:g} m/s {g:.2f} ({n})' for lo, hi, g, n in fit['bands'] if g is not None)
    return [f"  {name}: g = {fit['gain']:.3f} ({fit['n']} samples)", f'      by speed: {bands}']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording', help="sysid_recorder_node's CSV")
    ap.add_argument('--vehicle', default=share('control', 'config', 'vehicle.yaml'), help='vehicle.yaml')
    ap.add_argument('--hold', type=float, default=0.3, help='s the steering must be held for a steady sample')
    ap.add_argument('--max-lag', type=float, default=0.4, help='s, the longest delay searched')
    ap.add_argument('--kff', type=float, default=None, help="the adapter's speed_kff (default: adapter.yaml's)")
    ap.add_argument('--racecar', default=share('planning', 'config', 'racecar.ini'),
                    help='the planning config to compare with (the real car: racecar_car.ini)')
    args = ap.parse_args(argv)
    with open(args.vehicle) as f:
        car = yaml.safe_load(f)['/**']['ros__parameters']
    wheelbase = car['wheelbase']
    samples = load_samples(args.recording)
    if len(samples['t']) < 100:
        sys.exit(f'fit_vehicle: {args.recording} has {len(samples["t"])} samples; record more driving')
    kff = args.kff if args.kff is not None else adapter_kff()
    r = analyze(samples, wheelbase, hold=args.hold, max_lag=args.max_lag, kff=kff)

    out = [f"fit_vehicle: {r['samples']} samples over {r['duration_s']:.1f} s ({args.recording}), "
           f"wheelbase {wheelbase} m",
           '', 'Steering gain g = measured curvature / (tan(steering) / wheelbase), steady samples '
           f'(steering held {args.hold} s), speed bands {SPEED_BANDS[0]:g}-{SPEED_BANDS[-1]:g} m/s:']
    out += gain_lines('on the steering the sim applied', r['delta_applied'])
    out += gain_lines('on the /drive command          ', r['delta_cmd'])
    applied = r['delta_applied']['gain'] if r['delta_applied'] else None
    cmd = r['delta_cmd']['gain'] if r['delta_cmd'] else None
    calibrated = cmd is not None and abs(cmd - 1.0) <= 0.05
    if applied is not None:   # the sim: its adapter compensates
        out.append(f'  -> sim: adapter steering_gain {applied:.3f} '
                   f'(ros2 launch autodrive_bridge autodrive_launch.py steering_gain:={applied:.3f})')
        if cmd is not None:
            out.append('     the /drive command already drives the model\'s curvature (within 5 %)' if calibrated else
                       f'     the /drive command is off by {cmd:.3f}: the bridge ran without that steering_gain')
    elif cmd is not None:     # the car: the VESC's servo gain maps the /drive angle
        out.append('  -> the /drive steering is calibrated (within 5 %)' if calibrated else
                   f'  -> car: divide the VESC\'s steering_angle_to_servo_gain by {cmd:.3f} '
                   '(f1tenth_system config/vesc.yaml), then record and fit again')
    out.append('')
    if r['delay_s'] is None:
        out.append('Delay: the /drive steering does not vary enough to measure it')
    else:
        weak = '' if r['delay_corr'] >= 0.7 else ' (weak correlation: unreliable; drive more corners)'
        out.append(f"Delay from a /drive steering command to the path curvature: {r['delay_s']:.3f} s "
                   f"(correlation {r['delay_corr']:.2f}){weak}")
        out.append(f"  -> pure pursuit latency:={r['delay_s']:.3f} (race.sh <name> latency:={r['delay_s']:.3f})")
    if r['throttle'] is not None:
        ka, n_a, kb, n_b = r['throttle']
        out.append('')
        if ka is None and kb is None:
            out.append(f'Throttle: too few accelerating ({n_a}) / braking ({n_b}) samples to fit speed_ka')
        else:
            fmt = lambda k, n: f'{k:.4f} ({n} samples)' if k is not None else f'- ({n} samples)'  # noqa: E731
            out.append(f'Throttle per m/s^2 beyond speed_kff {kff} x speed: accelerating {fmt(ka, n_a)}, '
                       f'braking {fmt(kb, n_b)}')
            out.append('  -> sim: adapter speed_ka' + (f':={ka:.4f}' if ka is not None else ' (not enough data)')
                       + ', speed_ka_brake' + (f':={kb:.4f}' if kb is not None else ' (not enough data)')
                       + ' (with pure pursuit accel_feedforward:=true)')
    out.append('')
    out.append(f"Accelerations reached (99th percentile, above 1 m/s): +{r['accel_p99']:.2f} / "
               f"-{r['brake_p99']:.2f} m/s^2 longitudinal, {r['lateral_p99']:.2f} m/s^2 lateral; "
               f"top speed {r['speed_max']:.2f} m/s")
    if not math.isnan(r.get('slip_p99_deg', math.nan)):
        out.append(f"  sideslip 99th percentile {r['slip_p99_deg']:.1f} deg (ground truth)")
    limits = planning_limits(args.racecar)
    if limits:
        out.append(f"  {os.path.basename(args.racecar)} plans with ax_pos_safe {limits['ax_pos_safe']}, ax_neg_safe "
                   f"{limits['ax_neg_safe']}, ay_safe {limits['ay_safe']}, mue {limits['mue']} "
                   f"(x g = {limits['mue'] * 9.81:.2f} m/s^2)")
    out.append('  These are what this driving reached, not the car\'s limits: raise the planning limits in steps '
               'and watch lap_logger_node\'s wall clearance.')
    print('\n'.join(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
