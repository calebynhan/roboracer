#!/usr/bin/env bash
# Record a run to a rosbag, so localization and control changes can be replayed against it offline:
#
#   ws/scripts/record.sh [label] [extra topics ...]
#
# Writes ws/bags/<label>_<date>_<time>/ (default label: run) with the sensors, odometry, TF, the commands, the
# localizer's output and, in the sim, ground truth and lap counters (missing topics are skipped). Ctrl-C stops.
set -eo pipefail

label=${1:-run}
shift || true
source "$(dirname "$0")/common.sh"

out="$HOME/ws/bags/${label}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$out")"
topics=(
    /scan /sensors/imu/raw /odom /odom/wheel /tf /tf_static
    /drive /safety/drive /safety/active
    /localized_pose /tracked_pose /local_path /local_path/valid /raceline/provenance
    /ground_truth/odom /autodrive/roboracer_1/lap_count /autodrive/roboracer_1/collision_count
    /autodrive/roboracer_1/last_lap_time /autodrive/roboracer_1/steering /autodrive/roboracer_1/throttle
    /autodrive/roboracer_1/sim_reset
    "$@"
)
echo "Recording ${#topics[@]} topics to $out (Ctrl-C stops)"
exec ros2 bag record -o "$out" "${topics[@]}"
