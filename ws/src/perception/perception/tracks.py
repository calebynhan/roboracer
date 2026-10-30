"""Location of the per-track run-time files (see docs/pipeline.md), and the sim's map -> world offset."""
import math
import os

import yaml


def tracks_dir():
    # absolute: cartographer_node opens write_state's filename from its own working directory
    return os.path.abspath(os.path.expanduser(os.environ.get('TRACKS_DIR', '~/ws/tracks')))


def track_dir(name):
    return os.path.join(tracks_dir(), name)


def read_map_to_world(track):
    """(x, y, yaw) of the map origin in the sim world (tracks/<track>/map_to_world.yaml, save_map writes it while
    mapping in the sim), or None: mapped on the real car, or before the file was written."""
    path = os.path.join(track_dir(track), 'map_to_world.yaml')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        w = yaml.safe_load(f)
    return w['x_m'], w['y_m'], w['yaw_rad']


def world_to_map(origin, x, y, yaw):
    """Car pose in the sim world -> map frame; origin = (x, y, yaw) of the map origin in the world."""
    ox, oy, oyaw = origin
    c, s = math.cos(oyaw), math.sin(oyaw)
    dx, dy = x - ox, y - oy
    return c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(yaw - oyaw), math.cos(yaw - oyaw))
