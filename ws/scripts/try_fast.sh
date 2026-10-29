#!/usr/bin/env bash
# Baseline vs. the fast planner profile, unattended: N flying laps on <name>'s raceline, then the same map
# re-planned with planning/config/racecar_fast.ini (into tracks/<name>_fast), N laps on that, and a comparison.
#
#   ws/scripts/try_fast.sh <name> [--laps N] [race launch args, e.g. speed_scale:=1.0]
#
# Needs the bridge running, the sim connected in Autonomous, and tracks/<name>/ from map_track.sh in the sim
# (map.pbstream, raceline.csv, map_to_world.yaml). control lap_logger_node writes tracks/<name>/laps.csv and
# tracks/<name>_fast/laps.csv; autodrive_bridge steering_monitor runs during the baseline (its fit is printed).
# The fast stage runs only if the fast raceline is a clean mintime line (build_raceline exit 0).
set -eo pipefail

usage="usage: try_fast.sh <track name> [--laps N] [race launch args, e.g. speed_scale:=1.0]"
name=${1:?$usage}
shift
laps=5
race_args=()
while [ $# -gt 0 ]; do
    case $1 in
        --laps) laps=${2:?$usage}; shift ;;
        *:=*) race_args+=("$1") ;;
        *) echo "$usage" >&2; exit 1 ;;
    esac
    shift
done
[[ " ${race_args[*]} " == *" speed_scale:="* ]] || race_args+=(speed_scale:=1.0)
source "$(dirname "$0")/common.sh"
here=$(cd "$(dirname "$0")" && pwd)

for f in map.pbstream raceline.csv map_to_world.yaml; do
    [ -e "$tracks/$name/$f" ] || { echo "$tracks/$name/$f missing: map the track in the sim first (map_track.sh $name)" >&2; exit 1; }
done

# stop a background launch started under job control: SIGINT to its process group, then SIGTERM
stop_group() {
    [ -n "$1" ] || return 0
    kill -INT -- "-$1" 2>/dev/null || return 0
    for _ in $(seq 20); do kill -0 -- "-$1" 2>/dev/null || return 0; sleep 0.5; done
    kill -TERM -- "-$1" 2>/dev/null || true
}

logdir="$HOME/ws/log"
mkdir -p "$logdir"
race=
monitor=
trap 'stop_group "$race"; stop_group "$monitor"' EXIT

# bench <track> <tag>: race it, log $laps flying laps, stop the race
bench() {
    local track=$1 tag=$2
    echo "== $tag: $laps flying laps on $track (${race_args[*]}; log $logdir/try_fast_$tag.log)"
    set -m
    "$here/race.sh" "$track" "${race_args[@]}" >"$logdir/try_fast_$tag.log" 2>&1 &
    race=$!
    set +m
    # the logger starts once the car is under way, so the first lap_count change (the start line) starts a
    # flying lap rather than counting the reset; laps + 2 laps of 90 s at most
    sleep 8
    kill -0 "$race" 2>/dev/null || { echo "race.sh $track stopped, see $logdir/try_fast_$tag.log" >&2; exit 1; }
    # the executable itself, not `ros2 run`: timeout's SIGINT must reach the node
    timeout --signal=INT $((90 * (laps + 2))) \
        "$(ros2 pkg prefix control)/lib/control/lap_logger_node" --ros-args -p track:="$track" -p tag:="$tag" -p laps:="$laps" \
        || echo "lap logger timed out or failed: fewer than $laps laps logged for $tag" >&2
    stop_group "$race"
    race=
}

set -m
ros2 run autodrive_bridge steering_monitor --ros-args \
    --params-file "$(ros2 pkg prefix control)/share/control/config/vehicle.yaml" >"$logdir/try_fast_steering.log" 2>&1 &
monitor=$!
set +m
bench "$name" baseline
stop_group "$monitor"
monitor=

fast="${name}_fast"
mkdir -p "$tracks/$fast"
cp "$tracks/$name"/{map.pbstream,map.pgm,map.yaml,trajectory.csv,map_to_world.yaml} "$tracks/$fast/"
echo "== planning $fast with racecar_fast.ini"
status=0
ros2 run planning build_raceline "$fast" --config "$HOME/ws/src/planning/config/racecar_fast.ini" || status=$?
if [ "$status" -ne 0 ]; then
    echo "build_raceline $fast: exit $status (not a clean mintime line): fast stage skipped, see the output above" >&2
else
    bench "$fast" fast
fi
trap - EXIT

echo
echo "== steering (steering_monitor, put g into autodrive_bridge/config/adapter.yaml steering_gain)"
grep -E 'steering gain|steady turning' "$logdir/try_fast_steering.log" | tail -1 || echo "  no fit (see $logdir/try_fast_steering.log)"
echo "== laps (mean over flying laps)"
for t in "$name" "$fast"; do
    f="$tracks/$t/laps.csv"
    [ -f "$f" ] || continue
    # the newest run's rows: after the last '# run' line
    awk -F, -v t="$t" '
        /^# run/ {n = 0; sum = 0; col = 0; lat = 0; next}
        /^[0-9]/ {n++; sum += $2; col += $4; if ($6 > lat) lat = $6}
        END {if (n) printf "  %-24s %d laps, mean %.2f s, %d collisions, worst lateral error %.2f m\n", t, n, sum / n, col, lat
             else printf "  %-24s no laps logged\n", t}' "$f"
done
