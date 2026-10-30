"""control.tune_sectors: per-sector speed scales from lap_logger_node's laps."""
import datetime
import math
import os

import numpy as np
import pytest

from control import tune_sectors
from control.profile import read_sectors
from control.telemetry import LAP_COLUMNS, SECTOR_COLUMNS


def sector_row(lap, k, clearance, n=4, length=20.0):
    return {'lap': lap, 'sector': k, 's_start_m': k * length / n, 's_end_m': (k + 1) * length / n, 'time_s': 1.0,
            'lat_max_m': 0.1, 'clearance_min_m': clearance, 'speed_mean_mps': 5.0}


def test_tune_rules():
    rows = [sector_row(1, 0, 0.20), sector_row(1, 1, 0.08), sector_row(1, 2, 0.03), sector_row(1, 3, 0.15),
            sector_row(2, 0, 0.18), sector_row(2, 1, 0.09), sector_row(2, 2, 0.07), sector_row(2, 3, 0.065)]
    new, why = tune_sectors.tune([1.0, 1.0, 1.0, 1.0], rows, [{'lap': 1, 'collisions': 0}, {'lap': 2, 'collisions': 1}])
    # 0: roomy in both laps; 1: in between; 2: tight (0.03); 3: lap 2's closest -> its collision
    assert new == [pytest.approx(1.03), 1.0, pytest.approx(0.95), pytest.approx(0.95)]
    assert why[2].startswith('clearance 0.030 m < 0.05') and why[3] == 'collision'


def test_tune_bounds_and_touching():
    rows = [sector_row(1, 0, 0.5), sector_row(1, 1, -0.01)]
    new, why = tune_sectors.tune([1.29, 0.72], rows, [{'lap': 1, 'collisions': 0}], lo=0.7, hi=1.3)
    assert new == [1.3, 0.7] and why[1].startswith('clearance -0.010')


def test_tune_without_clearance_keeps_the_scale():
    new, why = tune_sectors.tune([1.1], [sector_row(1, 0, math.nan, n=1)], [])
    assert new == [1.1] and 'no clearance' in why[0]


def write_track(d, runs, args='speed_scale:=1.0 sector_speeds:=true'):
    """A circle raceline of 20 m and laps.csv / laps_sectors.csv with runs [(iso time, [(lap, collisions,
    [clearance per sector])])], their run lines with args (None: none recorded)."""
    os.makedirs(d, exist_ok=True)
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    r = 20.0 / (2 * np.pi)
    xy = np.column_stack([r * np.cos(t), r * np.sin(t)])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    np.savetxt(os.path.join(d, 'raceline.csv'), np.column_stack([s, xy, t, np.zeros((400, 3)) + [1 / r, 4, 0]]),
               delimiter=';', header='s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2')
    with open(os.path.join(d, 'laps.csv'), 'w') as fl, open(os.path.join(d, 'laps_sectors.csv'), 'w') as fs:
        fl.write(','.join(LAP_COLUMNS) + '\n')
        fs.write(','.join(SECTOR_COLUMNS) + '\n')
        for when, laps in runs:
            run = f'# run {when} tag=t track=x source=ground_truth sectors=4 raceline=r' + (
                f' args={args}' if args is not None else '') + '\n'
            fl.write(run)
            fs.write(run)
            for lap, collisions, clearances in laps:
                lap_row = dict.fromkeys(LAP_COLUMNS, 0)
                lap_row.update(lap=lap, collisions=collisions, valid=int(collisions == 0))
                fl.write(','.join(str(lap_row[c]) for c in LAP_COLUMNS) + '\n')
                for k, c in enumerate(clearances):
                    row = sector_row(lap, k, c)
                    fs.write(','.join(str(row[c]) for c in SECTOR_COLUMNS) + '\n')


def iso(offset_s):
    return (datetime.datetime.now().astimezone() + datetime.timedelta(seconds=offset_s)).isoformat(timespec='seconds')


def test_tune_sectors_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('TRACKS_DIR', str(tmp_path))
    d = str(tmp_path / 'loop')
    write_track(d, [(iso(-600), [(1, 0, [0.5, 0.5, 0.5, 0.5])]),
                    (iso(-60), [(1, 0, [0.2, 0.08, 0.03, 0.15]), (2, 1, [0.18, 0.09, 0.07, 0.065])])])
    assert tune_sectors.main(['loop']) == 0
    scales, length = read_sectors(os.path.join(d, 'sectors.yaml'))
    assert scales == [1.03, 1.0, 0.95, 0.95]   # the newest run only
    assert length == pytest.approx(20.0, rel=1e-3)
    assert '3 sectors changed' in capsys.readouterr().out

    # the same run again: it was not driven with these scales
    with pytest.raises(SystemExit, match='older than'):
        tune_sectors.main(['loop'])

    # a new run, with them: tuned from there
    write_track(d, [(iso(60), [(1, 0, [0.2, 0.2, 0.2, 0.2])])])
    assert tune_sectors.main(['loop', '--dry-run']) == 0
    assert 'dry run: 4 sectors would change' in capsys.readouterr().out
    assert read_sectors(os.path.join(d, 'sectors.yaml'))[0] == [1.03, 1.0, 0.95, 0.95]
    assert tune_sectors.main(['loop']) == 0
    assert read_sectors(os.path.join(d, 'sectors.yaml'))[0] == [1.06, 1.03, 0.98, 0.98]

    # a run driven without the scales, or one whose args are unknown: refused unless --force
    for args, why in (('speed_scale:=1.0', 'without sector_speeds:=true'), (None, 'launch args are unknown')):
        write_track(d, [(iso(120), [(1, 0, [0.2, 0.2, 0.2, 0.2])])], args=args)
        with pytest.raises(SystemExit, match=why):
            tune_sectors.main(['loop'])
    assert tune_sectors.main(['loop', '--force']) == 0


def test_tune_sectors_needs_laps(tmp_path, monkeypatch):
    monkeypatch.setenv('TRACKS_DIR', str(tmp_path))
    with pytest.raises(SystemExit, match='not found'):
        tune_sectors.main(['nothing'])
