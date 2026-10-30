#!/usr/bin/env python3
"""Summarize lap_logger_node's runs: one line per run, to compare settings.

    ros2 run control lap_summary <track> [<track> ...] [--last N]

Reads tracks/<track>/laps.csv (TRACKS_DIR or ~/ws/tracks). Per run: laps, valid laps (no collision or
interruption), mean and best valid lap time, collisions, worst distance from the raceline, smallest wall
clearance, localization RMS. --last: only the newest N runs of each track.
"""
import argparse
import math
import os
import re
import sys

from perception.tracks import track_dir

from control.telemetry import read_runs, summarize


def fmt(v, spec):
    if isinstance(v, float) and not math.isfinite(v):
        return '-'.rjust(int(re.match(r'\d+', spec).group()))
    return format(v, spec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tracks', nargs='+', help='track names (directories under the tracks dir)')
    ap.add_argument('--last', type=int, default=0, help='only the newest N runs of each track')
    args = ap.parse_args(argv)
    print(f"{'track':<18} {'tag':<16} {'laps':>4} {'valid':>5} {'mean s':>8} {'best s':>8} {'coll':>4} "
          f"{'lat max':>8} {'clear min':>9} {'loc rms':>8}")
    found = False
    for track in args.tracks:
        path = os.path.join(track_dir(track), 'laps.csv')
        if not os.path.isfile(path):
            print(f'{track:<18} no laps.csv ({path})')
            continue
        runs = read_runs(path)
        for line, rows in runs[-args.last:] if args.last > 0 else runs:
            tag = re.search(r'tag=(\S+)', line)
            s = summarize(rows)
            found = found or bool(rows)
            print(f"{track:<18} {tag.group(1) if tag else '-':<16} {s['laps']:>4} {s['valid']:>5} "
                  f"{fmt(s['mean_s'], '8.3f')} {fmt(s['best_s'], '8.3f')} {s['collisions']:>4} "
                  f"{fmt(s['lat_max_m'], '8.3f')} {fmt(s['clearance_min_m'], '9.3f')} {fmt(s['loc_rms_m'], '8.3f')}")
    return 0 if found else 1


if __name__ == '__main__':
    sys.exit(main())
