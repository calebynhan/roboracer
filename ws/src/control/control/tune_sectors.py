#!/usr/bin/env python3
"""Tune pure pursuit's speed per sector from lap_logger_node's laps: one step per benchmark run.

    ros2 run control tune_sectors <track> [--step 0.03] [--down 0.05] [--floor 0.05] [--target 0.12]
                                          [--min 0.7] [--max 1.3] [--dry-run]

The loop (each run's laps must be driven with the sectors.yaml the step before wrote):
    benchmark.sh <track> --tag s0 speed_scale:=1.0       # first run: no sectors.yaml, every scale is 1.0
    tune_sectors <track>                                 # writes tracks/<track>/sectors.yaml
    benchmark.sh <track> --tag s1 speed_scale:=1.0 sector_speeds:=true
    tune_sectors <track>                                 # ... until it changes nothing
Per sector (laps_sectors.csv, the newest run), over all its laps:
  hit     the sector of a lap with a collision with that lap's smallest wall clearance, or clearance <= 0
  tight   smallest wall clearance below floor m
          -> scale - down (not below min)
  roomy   smallest wall clearance above target m and no hit: scale + step (not above max)
  otherwise unchanged. Steps up are small and down large, so it settles below the limit, not on it.
Once sectors.yaml exists, it refuses a run that was not driven with it: older than the file, or whose launch args
(the run line's args=, which benchmark.sh records) lack sector_speeds:=true or are unknown (--force overrides
that), and one whose sectors do not match it.
"""
import argparse
import datetime
import math
import os
import sys

import numpy as np

from perception.tracks import track_dir
from planning.raceline import read_raceline

from control.profile import SECTORS_FILE, arc_lengths, read_sectors, write_sectors
from control.telemetry import read_runs


def run_time(line):
    """The start time of a run from its '# run <iso time> ...' line (read_runs drops the '# '), or None."""
    parts = line.split()
    try:
        return datetime.datetime.fromisoformat(parts[1]).timestamp()
    except (IndexError, ValueError):
        return None


def tune(scales, sector_rows, lap_rows, step=0.03, down=0.05, floor=0.05, target=0.12, lo=0.7, hi=1.3):
    """New scales and a reason per sector (see the module docstring). sector_rows: laps_sectors.csv rows of one
    run; lap_rows: its laps.csv rows (for the collisions)."""
    n = len(scales)
    clear = np.full(n, math.inf)
    hit = np.zeros(n, dtype=bool)
    collided = {int(r['lap']) for r in lap_rows if r.get('collisions', 0) > 0}
    by_lap = {}
    for r in sector_rows:
        k = int(r['sector'])
        c = r['clearance_min_m']
        if math.isfinite(c):
            clear[k] = min(clear[k], c)
            hit[k] |= c <= 0.0
            by_lap.setdefault(int(r['lap']), []).append((c, k))
    for lap in collided:
        if by_lap.get(lap):
            hit[min(by_lap[lap])[1]] = True   # the collision: where that lap came closest to a wall
    new, why = [], []
    for k in range(n):
        if not math.isfinite(clear[k]):
            new.append(scales[k])
            why.append('no clearance measured (no map.yaml?)')
        elif hit[k] or clear[k] < floor:
            new.append(max(lo, scales[k] - down))
            why.append('collision' if hit[k] and clear[k] > 0 else f'clearance {clear[k]:.3f} m < {floor} m')
        elif clear[k] > target:
            new.append(min(hi, scales[k] + step))
            why.append(f'clearance {clear[k]:.3f} m > {target} m')
        else:
            new.append(scales[k])
            why.append(f'clearance {clear[k]:.3f} m')
    return new, why


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('track')
    ap.add_argument('--step', type=float, default=0.03, help='scale added to a roomy sector')
    ap.add_argument('--down', type=float, default=0.05, help='scale taken from a tight sector or one with a hit')
    ap.add_argument('--floor', type=float, default=0.05, help='m: a sector closer to a wall than this is tight')
    ap.add_argument('--target', type=float, default=0.12, help='m: a sector farther from the walls is roomy')
    ap.add_argument('--min', type=float, default=0.7, dest='lo', help='smallest scale')
    ap.add_argument('--max', type=float, default=1.3, dest='hi', help='largest scale')
    ap.add_argument('--dry-run', action='store_true', help='print, do not write sectors.yaml')
    ap.add_argument('--force', action='store_true',
                    help='take the newest run even if its launch args do not show sector_speeds:=true')
    args = ap.parse_args(argv)

    d = track_dir(args.track)
    laps_csv, sectors_csv = os.path.join(d, 'laps.csv'), os.path.join(d, 'laps_sectors.csv')
    for f in (laps_csv, sectors_csv):
        if not os.path.isfile(f):
            sys.exit(f'tune_sectors: {f} not found: run benchmark.sh {args.track} first')
    sector_runs = [r for r in read_runs(sectors_csv) if r[1]]
    if not sector_runs:
        sys.exit(f'tune_sectors: {sectors_csv} has no laps yet')
    line, sector_rows = sector_runs[-1]
    lap_rows = next((rows for ln, rows in reversed(read_runs(laps_csv)) if ln == line), [])
    n = int(max(r['sector'] for r in sector_rows)) + 1
    laps = sorted({int(r['lap']) for r in sector_rows})

    xy = read_raceline(os.path.join(d, 'raceline.csv'))[:, 1:3]
    if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    length = arc_lengths(xy)[2]
    path = os.path.join(d, SECTORS_FILE)
    if os.path.isfile(path):
        scales, made_for = read_sectors(path)
        started = run_time(line)
        if started is not None and started < os.path.getmtime(path):
            sys.exit(f'tune_sectors: the newest run ({line}) is older than {path}: race with '
                     'sector_speeds:=true (benchmark.sh ... sector_speeds:=true) and tune again')
        _, has_args, run_args = line.partition(' args=')
        if not args.force and not has_args:
            sys.exit(f'tune_sectors: the newest run\'s launch args are unknown ({line}): run it with benchmark.sh '
                     '(which records them), or --force if it was driven with sector_speeds:=true')
        if not args.force and 'sector_speeds:=true' not in run_args.split():
            sys.exit(f'tune_sectors: the newest run was driven without sector_speeds:=true (args {run_args}), at '
                     f'scale 1.0 rather than {path}\'s scales: race with it and tune again')
        if len(scales) != n:
            sys.exit(f'tune_sectors: {path} has {len(scales)} sectors, the run {n}')
        if made_for is not None and abs(made_for - length) > 0.01 * length:
            sys.exit(f'tune_sectors: {path} was made for a {made_for:.2f} m raceline, this one is {length:.2f} m')
    else:
        scales = [1.0] * n

    new, why = tune(scales, sector_rows, lap_rows, args.step, args.down, args.floor, args.target, args.lo, args.hi)
    print(f'{args.track}: {len(laps)} laps of {line}')
    print(f"{'sector':>6} {'s from':>8} {'s to':>8}  {'scale':>6} -> {'new':>6}  why")
    for k in range(n):
        mark = '' if new[k] == scales[k] else ('  +' if new[k] > scales[k] else '  -')
        print(f'{k:>6} {k * length / n:>8.2f} {(k + 1) * length / n:>8.2f}  {scales[k]:>6.3f} -> {new[k]:>6.3f}  '
              f'{why[k]}{mark}')
    changed = sum(a != b for a, b in zip(new, scales))
    if args.dry_run:
        print(f'dry run: {changed} sectors would change')
    elif changed:
        write_sectors(path, new, length, comment=f'tuned from {len(laps)} laps of {line}')
        print(f'{changed} sectors changed -> {path}; race with sector_speeds:=true')
    else:
        print('nothing to change: settled')
    return 0


if __name__ == '__main__':
    sys.exit(main())
