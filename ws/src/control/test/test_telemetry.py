"""control.telemetry: where a pose is along the raceline, laps, per-lap and per-sector stats, wall clearance."""
import math

import numpy as np
import pytest

from control.telemetry import (LAP_COLUMNS, Clearance, LapCounter, LapStats, Track, format_row, read_runs,
                               summarize)
from perception.mcl import LikelihoodMap


def circle(radius=4.0, n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)   # counter-clockwise, from (radius, 0)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def hairpin(length=10.0, gap=1.0, step=0.1):
    """+x along y = 0, a half turn, -x back along y = gap (the legs `gap` apart, driven in opposite directions)."""
    x = np.arange(0.0, length, step)
    t = np.linspace(-np.pi / 2, np.pi / 2, int(np.pi * gap / 2 / step), endpoint=False)
    r = gap / 2
    return np.concatenate([np.column_stack([x, np.zeros_like(x)]),
                           np.column_stack([length + r * np.cos(t), r + r * np.sin(t)]),
                           np.column_stack([length - x, np.full_like(x, gap)]),
                           np.column_stack([-r * np.cos(t), r - r * np.sin(t)])])


def test_locate_arc_length_and_side():
    track = Track(circle())
    assert track.length == pytest.approx(2 * np.pi * 4.0, rel=1e-3)
    s, offset, _ = track.locate(4.0, 0.0, math.pi / 2)
    assert s == pytest.approx(0.0, abs=1e-6) and offset == pytest.approx(0.0, abs=1e-6)
    s, offset, _ = track.locate(0.0, 3.7, math.pi)            # a quarter lap on, 0.3 m inside = left
    assert s == pytest.approx(track.length / 4, abs=0.01) and offset == pytest.approx(0.3, abs=0.01)
    _, offset, _ = track.locate(-4.5, 0.0, -math.pi / 2)
    assert offset == pytest.approx(-0.5, abs=0.01)


def test_closing_point_is_dropped():
    xy = circle()
    assert Track(np.vstack([xy, xy[:1]])).length == pytest.approx(Track(xy).length)


def test_locate_keeps_its_leg():
    # 0.55 m off the +x leg is 0.45 m from the -x leg: the first pose picks the leg by heading, the next one
    # stays on it although the other leg is closer
    track = Track(hairpin())
    s, offset, _ = track.locate(5.0, 0.55, 0.0)
    assert s == pytest.approx(5.0, abs=0.01) and offset == pytest.approx(0.55, abs=0.01)
    s, offset, _ = track.locate(5.1, 0.6, 0.0)
    assert s == pytest.approx(5.1, abs=0.01) and offset == pytest.approx(0.6, abs=0.01)


def drive_laps(counter, length, speed=5.0, dt=0.02, laps=3.0, s0=0.0):
    events, t, s = [], 0.0, s0
    for _ in range(int(laps * length / speed / dt)):
        e = counter.update(t, s % length)
        if e:
            events.append(e)
        t += dt
        s += speed * dt
    return events


def test_laps_are_flying_laps_timed_at_the_line():
    length = 20.0
    events = drive_laps(LapCounter(length), length, laps=4.0, s0=15.0)   # starts 5 m before the line
    # the first crossing comes after 5 m, less than half a lap of run-up: timing starts at the second (25 m)
    assert events[0][0] == 'start' and events[0][1] == pytest.approx(5.0, abs=1e-6)
    laps = [e for e in events if e[0] == 'lap']
    assert len(laps) == 2
    for e in laps:
        assert e[2] == pytest.approx(4.0, abs=1e-6) and not e[3] and e[4]   # interpolated: exact at any rate


def test_jitter_at_the_line_counts_once():
    c = LapCounter(20.0, run_up=0.0)
    samples = [(0.0, 19.9), (0.1, 0.05), (0.2, 19.95), (0.3, 0.1)]   # over the line, back, over again
    events = [c.update(t, s) for t, s in samples]
    assert [e[0] for e in events if e] == ['start']


def test_teleport_interrupts_the_lap_and_never_counts_as_crossing():
    length = 20.0
    c = LapCounter(length, run_up=0.0)
    drive_laps(c, length, laps=0.25, s0=19.0)              # started, 4 m into lap 1
    assert c.start is not None
    # a respawn back over the line, to just after s = 0: no lap, and the lap is interrupted
    assert c.update(c.prev[0] + 0.02, 0.5) is None
    assert c.interrupted
    t, s = c.prev
    events = []
    while s < length + 1.0:
        t, s = t + 0.02, s + 0.1
        e = c.update(t, s % length)
        events.append(e) if e else None
    assert [e[0] for e in events] == ['lap'] and events[0][3]


def test_reset_behind_the_line_is_not_a_flying_lap():
    # a reset puts the car 0.05 m behind the line: it launches over it from rest. That lap ends interrupted and
    # the next one is not timed; timing starts again a lap later
    length = 20.0
    c = LapCounter(length)
    drive_laps(c, length, laps=1.6, s0=0.0)             # 0 -> 32 m: timing started at 20 m
    assert c.start is not None
    t = c.prev[0]
    assert c.update(t + 0.5, length - 0.05) is None    # the reset
    events, s = [], length - 0.05
    for k in range(1, 1100):                            # 1 m/s from rest, 0.02 s steps: 22 m
        e = c.update(t + 0.5 + 0.02 * k, (s + 0.02 * k) % length)
        if e:
            events.append(e)
    assert [e[0] for e in events] == ['lap', 'start']
    assert events[0][3] and not events[0][4]           # interrupted, and the next lap is not timed
    assert events[1][1] == pytest.approx(t + 0.5 + 20.05, abs=0.03)   # a lap after the line


def test_backward_driving_does_not_end_a_lap():
    c = LapCounter(20.0, run_up=0.0)
    for t, s in enumerate([19.5, 0.5]):   # start
        c.update(float(t), s)
    events = [c.update(float(t), s) for t, s in enumerate([0.2, 19.8, 19.5], start=2)]   # back over the line
    events += [c.update(float(t), s) for t, s in enumerate([19.8, 0.3], start=5)]   # and forward again
    assert not any(events)   # the progress since the start is not half a lap


def test_lap_stats_rows():
    st = LapStats(4)
    for k, lat in enumerate((0.1, -0.3, 0.2, 0.05)):
        st.add(k, 0.5, lat, 5.0, 4.0, clearance=0.2 - 0.01 * k)
    st.add_localization(0.05, math.radians(1.0))
    st.add_localization(0.05, math.radians(1.0))
    row, sectors = st.rows(1, 7.0, 6.98, 0, False, 20.0)
    assert row['valid'] == 1 and row['collisions'] == 0
    assert row['lat_max_m'] == pytest.approx(0.3)
    assert row['clearance_min_m'] == pytest.approx(0.17)
    assert row['speed_err_rms_mps'] == pytest.approx(1.0)
    assert row['loc_rms_m'] == pytest.approx(0.05) and row['loc_yaw_rms_deg'] == pytest.approx(1.0)
    assert [r['s_start_m'] for r in sectors] == [0.0, 5.0, 10.0, 15.0]
    assert sectors[1]['lat_max_m'] == pytest.approx(0.3) and sectors[1]['time_s'] == pytest.approx(0.5)
    assert format_row(row, LAP_COLUMNS).startswith('1,7.0000,6.9800,1,0,')
    row, _ = st.rows(2, 7.0, math.nan, 1, False, 20.0)
    assert row['valid'] == 0 and ',nan,' in format_row(row, LAP_COLUMNS)


def test_lap_stats_without_command_localization_or_map():
    st = LapStats(2)
    st.add(0, 0.1, 0.0, 3.0)
    row, sectors = st.rows(1, 5.0, math.nan, -1, True, 10.0)
    assert row['valid'] == 0   # interrupted
    assert math.isnan(row['speed_err_rms_mps']) and math.isnan(row['loc_rms_m'])
    assert math.isnan(row['clearance_min_m']) and math.isnan(sectors[1]['speed_mean_mps'])


def test_clearance_of_the_footprint():
    # a wall along y = 1.0 m (cells with y in [1.0, 1.05)), 5 cm cells, origin (-5, -5)
    occ = np.zeros((200, 200), dtype=bool)
    occ[200 - 1 - 120, :] = True   # row of y = -5 + 120 * 0.05 = 1.0
    lmap = LikelihoodMap(occ, ~occ, 0.05, (-5.0, -5.0))
    clear = Clearance(lmap, front=0.42, rear=0.08, width=0.27)
    # car along +x at y = 0.5: its left side at y = 0.635, 0.365 m from the wall's lower edge at 1.0
    assert clear(0.0, 0.5, 0.0) == pytest.approx(0.365, abs=0.03)
    # yawed 90 deg (nose toward the wall): the front at y = 0.92
    assert clear(0.0, 0.5, math.pi / 2) == pytest.approx(0.08, abs=0.03)
    assert clear(0.0, 1.025, 0.0) < 0.0   # on the wall (the distance field is unsigned: not beyond it)


def test_read_runs_and_summarize(tmp_path):
    path = tmp_path / 'laps.csv'
    header = ','.join(LAP_COLUMNS)
    rows = ['1,7.1,nan,1,0,0.05,0.2,0.1,5,7,0.3,0.04,0.1,1.0,300',
            '2,6.9,nan,1,0,0.05,0.25,0.08,5,7,0.3,0.06,0.1,1.0,300',
            '3,9.0,nan,0,1,0.05,0.6,-0.01,5,7,0.3,nan,nan,nan,300']
    path.write_text('\n'.join([header, '# run A tag=old', '1,8.0,nan,1,0,0.1,0.3,0.1,4,6,0.3,nan,nan,nan,300',
                               '# run B tag=new'] + rows) + '\n')
    runs = read_runs(str(path))
    assert [r[0] for r in runs] == ['run A tag=old', 'run B tag=new']
    s = summarize(runs[1][1])
    assert (s['laps'], s['valid'], s['collisions']) == (3, 2, 1)
    assert s['mean_s'] == pytest.approx(7.0) and s['best_s'] == pytest.approx(6.9)
    assert s['lat_max_m'] == pytest.approx(0.6) and s['clearance_min_m'] == pytest.approx(-0.01)
    assert s['loc_rms_m'] == pytest.approx(math.sqrt((0.04 ** 2 + 0.06 ** 2) / 2))
