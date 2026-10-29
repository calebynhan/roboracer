"""racecar.ini (read by TUM's optimizer, which cannot load ROS parameters) must match the car
geometry in control/config/vehicle.yaml, the one place it is set."""
import configparser
import json
import math
import os

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
VEHICLE = os.path.join(HERE, '..', '..', 'control', 'config', 'vehicle.yaml')
RACECAR = os.path.join(HERE, '..', 'config', 'racecar.ini')
ADAPTER = os.path.join(HERE, '..', '..', 'autodrive_bridge', 'config', 'adapter.yaml')


def test_racecar_ini_matches_vehicle_yaml():
    with open(VEHICLE) as f:
        car = yaml.safe_load(f)['/**']['ros__parameters']
    ini = configparser.ConfigParser()
    ini.read(RACECAR)
    veh = json.loads(ini.get('GENERAL_OPTIONS', 'veh_params'))
    mintime = json.loads(ini.get('OPTIMIZATION_OPTIONS', 'vehicle_params_mintime'))
    assert mintime['delta_max'] == pytest.approx(car['max_steering_angle'])
    assert mintime['wheelbase_front'] + mintime['wheelbase_rear'] == pytest.approx(car['wheelbase'])
    assert veh['curvlim'] == pytest.approx(math.tan(car['max_steering_angle']) / car['wheelbase'], abs=0.01)


def test_width_opt_fits_the_car():
    ini = configparser.ConfigParser()
    ini.read(RACECAR)
    width = json.loads(ini.get('GENERAL_OPTIONS', 'veh_params'))['width']
    width_opt = json.loads(ini.get('OPTIMIZATION_OPTIONS', 'optim_opts_mintime'))['width_opt']
    assert width_opt >= width   # the car fits inside its own margin


def test_planned_accelerations_are_what_the_adapter_drives():
    """safe_traj's acceleration ellipse must match the sim adapter's speed ramp: a plan accelerating faster than
    max_acceleration cannot be followed, and braking harder than max_deceleration is cut short."""
    with open(ADAPTER) as f:
        adapter = yaml.safe_load(f)['adapter']['ros__parameters']
    ini = configparser.ConfigParser()
    ini.read(RACECAR)
    opts = json.loads(ini.get('OPTIMIZATION_OPTIONS', 'optim_opts_mintime'))
    assert opts['ax_pos_safe'] == pytest.approx(adapter['max_acceleration'])
    assert abs(opts['ax_neg_safe']) <= adapter['max_deceleration']
