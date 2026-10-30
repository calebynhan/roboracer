#!/usr/bin/env bash
# Race a mapped track for N flying laps and log every lap (control lap_logger_node), unattended:
#
#   ws/scripts/benchmark.sh <name> [--laps N] [--tag LABEL] [--timeout S] [--no-race] [race launch args]
#   e.g. ws/scripts/benchmark.sh porto --laps 5 --tag baseline speed_scale:=1.0
#
# Starts the lap logger, then race.sh <name> with the launch args (which resets the sim car), waits for N laps
# (default 5; at most --timeout s, default 120 + 90 per lap), stops both and prints this run next to the
# track's earlier ones (control lap_summary). --no-race: only log, while something else drives (teleop, a
# race started by hand). Writes tracks/<name>/laps.csv and laps_sectors.csv (the run line records the launch
# args, which tune_sectors checks); logs in ws/log/benchmark_<name>_*. Exit status 1 when fewer than N laps
# were logged (timed out, race.sh stopped). Needs the bridge running and the sim connected (or, on the real car,
# the car's stack).
set -eo pipefail

usage="usage: benchmark.sh <track name> [--laps N] [--tag LABEL] [--timeout S] [--no-race] [race launch args]"
name=${1:?$usage}
shift
laps=5
tag=
timeout_s=
race=true
race_args=()
while [ $# -gt 0 ]; do
    case $1 in
        --laps) laps=${2:?$usage}; shift ;;
        --tag) tag=${2:?$usage}; shift ;;
        --timeout) timeout_s=${2:?$usage}; shift ;;
        --no-race) race=false ;;
        *:=*) race_args+=("$1") ;;
        *) echo "$usage" >&2; exit 1 ;;
    esac
    shift
done
[[ $laps =~ ^[1-9][0-9]*$ ]] || { echo "--laps must be a positive integer" >&2; exit 1; }
timeout_s=${timeout_s:-$((120 + 90 * laps))}
# default tag: the launch args (speed_scale:=1.0 -> speed_scale=1.0), joined with _
[ -n "$tag" ] || tag=$(IFS=_; echo "${race_args[*]:-default}" | sed 's/:=/=/g' | tr -c 'A-Za-z0-9_.=\n-' '_')
source "$(dirname "$0")/common.sh"
here=$(cd "$(dirname "$0")" && pwd)

[ -f "$tracks/$name/raceline.csv" ] || { echo "$tracks/$name/raceline.csv missing; run map_track.sh $name" >&2; exit 1; }

# stop a background launch started under job control: SIGINT to its process group, then SIGTERM
stop_group() {
    [ -n "$1" ] || return 0
    kill -INT -- "-$1" 2>/dev/null || return 0
    for _ in $(seq 20); do kill -0 -- "-$1" 2>/dev/null || return 0; sleep 0.5; done
    kill -TERM -- "-$1" 2>/dev/null || true
}

stamp=$(date +%Y%m%d_%H%M%S)
logger_log="$HOME/ws/log/benchmark_${name}_${stamp}_laps.log"
race_log="$HOME/ws/log/benchmark_${name}_${stamp}_race.log"
mkdir -p "$(dirname "$logger_log")"
logger=
racer=
trap 'stop_group "$racer"; stop_group "$logger"' EXIT

# job control: each gets its own process group and keeps SIGINT, so stopping it stops what it started
set -m
# as YAML strings: a tag or track name like 2 or true would be read as a number or a bool
ros2 run control lap_logger_node --ros-args -p "track:='$name'" -p "tag:='$tag'" -p laps:="$laps" \
    -p "run_args:='${race_args[*]}'" >"$logger_log" 2>&1 &
logger=$!
set +m
for _ in $(seq 40); do grep -q 'Logging laps' "$logger_log" 2>/dev/null && break; sleep 0.25; done
kill -0 "$logger" 2>/dev/null || { echo "lap logger failed to start, see $logger_log:" >&2; tail -5 "$logger_log" >&2; exit 1; }
grep -m1 'Logging laps' "$logger_log" | sed 's/^.*\]: /  /'

if $race; then
    set -m
    "$here/race.sh" "$name" "${race_args[@]}" >"$race_log" 2>&1 &
    racer=$!
    set +m
    echo "Racing $name (${race_args[*]:-defaults}; log $race_log), $laps laps, tag $tag"
else
    echo "Logging $laps laps on $name while something else drives (tag $tag)"
fi

start=$SECONDS
shown=0
while kill -0 "$logger" 2>/dev/null; do
    if [ -n "$racer" ] && ! kill -0 "$racer" 2>/dev/null; then
        echo "race.sh stopped before $laps laps, see $race_log:" >&2
        tail -5 "$race_log" >&2
        break
    fi
    if [ $((SECONDS - start)) -ge "$timeout_s" ]; then
        echo "timed out after $timeout_s s" >&2
        break
    fi
    done_laps=$(grep -c '\]: lap [0-9]*:' "$logger_log" || true)
    if [ "$done_laps" -gt "$shown" ]; then
        grep '\]: lap [0-9]*:' "$logger_log" | sed -n "$((shown + 1)),${done_laps}p" | sed 's/^.*\]: /  /'
        shown=$done_laps
    fi
    sleep 1
done
grep '\]: lap [0-9]*:' "$logger_log" | sed -n "$((shown + 1)),\$p" | sed 's/^.*\]: /  /' || true
done_laps=$(grep -c '\]: lap [0-9]*:' "$logger_log" || true)
stop_group "$racer"
racer=
stop_group "$logger"
logger=
trap - EXIT

echo
ros2 run control lap_summary "$name" || true
if [ "$done_laps" -lt "$laps" ]; then
    echo "INCOMPLETE: $done_laps of $laps laps logged (see $logger_log and $race_log)" >&2
    exit 1
fi
