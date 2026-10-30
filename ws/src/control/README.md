# control

Pure pursuit on a generated raceline, and two followers for the mapping lap(s): the wall follower (the default) and an experimental follow the gap with disparity extension (see below). `pure_pursuit_node` takes the car's pose in the `map` frame,
finds the closest raceline waypoint, picks the first waypoint one lookahead ahead (lookahead =
`lookahead_time` * drive speed, clipped to [`lookahead_min`, `lookahead_distance`]), steers on the arc
through it and drives at the raceline speed scaled by `speed_scale`. The speed is taken at the target
waypoint, not the closest one: the car brakes early into corners, but on corner exits it can exceed the
profile at its own position near `speed_scale` 1.0. With `adaptive_lookahead:=true` the lookahead is
also capped by the curvature of the raceline near the car, so at speed it does not cut corners (on by
default, `adaptive_lookahead:=false` turns it off; see Adaptive lookahead below). Five more options are off by
default (Options, below). See `docs/pipeline.md` for where it sits in the stack.

Tools around it: `lap_logger_node` and `ws/scripts/benchmark.sh` measure every lap (Lap benchmark),
`tune_sectors` sets the speed per sector from those laps, `sysid_recorder_node` and `fit_vehicle` measure the car
(Vehicle identification), `safety_node` brakes for what the lidar sees in the car's path (Emergency brake), and
`car.launch.py` adds what the real car needs to f1tenth_system (Real car).

`follow_the_gap_node` is a reactive alternative that needs no map: it steers toward the farthest point
(the middle of the run of farthest beams, since ranges are capped at `max_range`) of the widest free
gap within `fov` (default ±90°) of straight ahead in `/scan`, at a constant `speed` (`ros2 run control follow_the_gap_node`, or
`ros2 launch control follow_the_gap.launch.py`, which loads `config/vehicle.yaml`). Its steering is
clipped to `max_steering_angle` (default 0.524 rad, the value in `config/vehicle.yaml`).
`tools/lidar_plot.py` plots its scan processing live (WebAgg, not installed).

`disparity_gap_node` is a follow the gap with disparity extension that drives the mapping lap(s) (experimental, see below).

## Run

```bash
ros2 launch control pure_pursuit.launch.py track:=<name>
ros2 launch control pure_pursuit.launch.py track:=<name> speed_scale:=0.3
ros2 launch control race.launch.py track:=<name> avoid:=true   # + obstacle avoidance
ros2 launch control race.launch.py track:=<name> speed_scale:=1.0   # adaptive lookahead is on by default
ros2 launch control race.launch.py track:=<name> sector_speeds:=true safety:=true   # tuned sectors, emergency brake
ros2 launch control race.launch.py track:=<name> vehicle:=$(ros2 pkg prefix control)/share/control/config/vehicle_car.yaml
```

Localization must be running: the default pose topic is `/localized_pose` (perception's
`localization_gate` or `mcl_node`, which `race.launch.py` starts), so the car waits until it has
localized (~2-3 s at the start and after each sim reset). `pose_topic:=/tracked_pose` (Cartographer raw)
bypasses that gate. The sim's `/ground_truth/pose` is in the sim `world` frame, not `map`
(the offset is in `tracks/<name>/map_to_world.yaml`), so pure pursuit refuses it.

With `avoid:=true`, pure pursuit steers on `/local_path` instead of the raceline: planning's
`obstacle_spliner_node` (started by `race.launch.py avoid:=true`) publishes the raceline with its points
shifted sideways around an obstacle the map did not have, one point per raceline point. Speeds stay the
raceline's.

## Safety

Each pose is checked before it is used; a failed check publishes a stop (speed 0, steering 0) and a
warning (at most once a second), and driving resumes on the next good pose:

- the position must be finite and the quaternion finite with a norm within 1 ± 0.1;
- `header.frame_id` must be `map`;
- the pose must be at most `max_pose_age` s old (an unstamped pose counts as stale);
- the car must be within `max_offset` m of the raceline (with `avoid`, of `/local_path`);
- the closest raceline point must head within 90 deg of the car. If it does not (a spin, a 180 deg flipped
  pose, a teleport), the whole line is searched for the closest point heading along the car; with none
  within `flip_reacquire_distance` the car stops (and stays stopped until it heads along a line);
- with `avoid`: a `/local_path` must have arrived, be at most `max_path_age` s old, have one pose per
  raceline point in the `map` frame, and `/local_path/valid` must be true (a collision-free path exists).

A command that is not finite is replaced by a stop, the speed is never negative (a negative `vx_mps`
drives at 0, `speed_scale` must be > 0). Once a pose has arrived, a 20 Hz watchdog stops the car when no
pose has come for `max_pose_age` s (localization withheld after a sim reset or while `mcl_node`
searches, a crashed localizer): it publishes a stop on every tick for `stop_burst` s, then nothing
(`fast_bridge` holds zero throttle without `/drive`, and teleop is not fought). A burst, not one stop:
the adapter restarts its speed ramp near the measured speed after a pause in `/drive`, so a lone stop
means near-cruise throttle; only a steady stream brakes. The localizers publish at 40-100 Hz, so it does
not fire while racing. Ctrl-C publishes 0.5 s of 20 Hz stops before exiting.

These are coarse. Cartographer extrapolates `/tracked_pose` (`use_pose_extrapolator` defaults to true;
`pose_publish_period_sec` 1e-2), so its stamps stay fresh even when SLAM stalls: the age check only
catches DDS backlog or a non-Cartographer publisher. `max_offset` only catches gross errors: a wrong
pose on a nearby leg (iros2026's legs are 1.07 m apart) passes it. The `/scan`-based stop is `safety_node`
(Emergency brake): `race.launch.py safety:=true`, and on the real car `car.launch.py`.

## Topics

| Topic | Type | |
|---|---|---|
| `pose_topic` (`/localized_pose`) | `geometry_msgs/PoseStamped` | subscribed, must be in the `map` frame |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | published, one command per pose (and the watchdog's stops) |
| `/local_path` | `nav_msgs/Path` | subscribed with `avoid`, the path steered on |
| `/local_path/valid` | `std_msgs/Bool` | subscribed with `avoid`, false = stop |
| `/odom` | `nav_msgs/Odometry` | subscribed with `latency` > 0: the speed for the prediction |

## Parameters

| Parameter | Default | |
|---|---|---|
| `track` | `''` | loads `$TRACKS_DIR/<track>/raceline.csv`, `TRACKS_DIR` defaults to `~/ws/tracks` |
| `raceline_path` | `''` | explicit raceline file, overrides `track` |
| `pose_topic` | `/localized_pose` | any `PoseStamped` of `base_link` in the `map` frame |
| `speed_scale` | `0.5` | speed = raceline `vx_mps` * `speed_scale`, must be > 0 |
| `max_speed` | `0.0` | m/s cap on the drive speed, 0 = no cap |
| `lookahead_time` | `0.4` | s; lookahead = `lookahead_time` * drive speed, clipped to [`lookahead_min`, `lookahead_distance`] |
| `lookahead_min` | `0.6` | m, about twice the wheelbase |
| `lookahead_distance` | `3.0` | m, maximum lookahead |
| `adaptive_lookahead` | `true` | cap the lookahead by the curvature of the raceline near the car (below); off = the speed term only |
| `adaptive_lookahead_cut` | `0.1` | m; corner cut allowed: on a curve of radius R the lookahead is at most sqrt(2 * cut * R) |
| `adaptive_lookahead_min_time` | `0.25` | s; the capped lookahead is never below this * drive speed |
| `adaptive_lookahead_slope` | `0.5` | m per m: the cap rises by this much per metre of distance from the curve (before and after it). Node parameter only (not a launch argument) |
| `adaptive_lookahead_smooth` | `0.5` | m; the raceline's curvature is averaged over this length of arc. Node parameter only (not a launch argument) |
| `search_window` | `2.0` | m; the closest point is searched this far along the line either side of the previous one |
| `reacquire_distance` | `1.0` | m; farther than this from that point, the whole line is searched (points heading within 90 deg of the car) |
| `flip_reacquire_distance` | `0.5` | m; when the closest point heads against the car, a point heading along it must be this close, else stop (iros2026's opposite legs are 1.07 m apart) |
| `max_pose_age` | `0.5` | s; an older pose gets a stop, and so does no pose for this long (same as `fast_bridge`'s `/drive` watchdog) |
| `stop_burst` | `1.5` | s of 20 Hz stops after the poses are lost, enough for the adapter to brake |
| `max_offset` | `3.0` | m; farther than this from the raceline, the pose is taken as bad and the car stops |
| `avoid` | `false` | steer on `/local_path` (obstacle avoidance) instead of the raceline |
| `max_path_age` | `0.5` | s; with `avoid`, an older `/local_path` gets a stop |
| `latency` | `0.0` | s from a `/drive` to the car acting on it (`fit_vehicle`); > 0: the pose is predicted to then (Options) |
| `odom_topic` | `/odom` | with `latency`: the speed |
| `speed_preview_time` | `-1.0` | s; >= 0: the speed is the profile's at the car's closest point plus planned speed x this, not at the target |
| `accel_feedforward` | `false` | `/drive` acceleration = the profile's (for the sim adapter's `speed_ka`); 0 while `max_speed` or the swerve cap holds the speed |
| `sector_speeds` | `false` | the speed profile x `tracks/<track>/sectors.yaml` (`tune_sectors`) |
| `sectors_path` | `''` | another sectors file. Node parameter only |
| `avoid_max_lateral_accel` | `0.0` | m/s^2; with `avoid`, the speed where `/local_path` leaves the raceline is capped at sqrt(this / its curvature); 0 = off |
| `avoid_max_decel` | `3.5` | m/s^2 braking into that cap. Node parameter only |
| `wheelbase` | none | m, from `config/vehicle.yaml` (required) |
| `max_steering_angle` | none | rad, steering is clipped to this; from `config/vehicle.yaml` (required) |

The raceline is the TUM output format: `;` separated, `#` header
`s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2`.

### Adaptive lookahead

Pure pursuit steers on the chord to its target, so on a curve of curvature kappa it settles inside the
line, about lookahead^2 * kappa / 2 deep (in the sim about 0.6 of that), and at a corner entry its target is
already in the turn while the car is still on the straight. The racelines pass 0.25-0.4 m from the walls,
so at `speed_scale` 1.0 (lookahead 1-2.4 m) the car cut into the inside walls of Porto's and Berlin's
corners. With `adaptive_lookahead:=true` (`control/lookahead.py`) each raceline point i gets a cap,
precomputed once from the line's geometry:

    cap_i = max(lookahead_min, min over points j of (sqrt(2 * cut / |kappa_j|) + slope * |arc distance i to j|))
    lookahead = min(clip(lookahead_time * speed, lookahead_min, lookahead_distance), max(cap_closest, min_time * speed))

so on a curve the cut is at most about `adaptive_lookahead_cut` while `min_time` * speed is below the cap;
where the floor wins it is larger (4 m/s on a radius of 1 m: lookahead 1.0 m, cut about 0.5 m), which is
the trade for enough preview at speed. The cap shrinks toward a curve (and grows back after it) by at
most `slope` m per metre driven (no steps, no chatter), with curves found by arc length, so uneven point
spacing and repeated points do not matter. On straights the cap is infinite and nothing changes; at `speed_scale` 0.5 the speed
term is mostly below the cap already. `kappa` comes from the raceline's points (heading change per metre,
averaged over `adaptive_lookahead_smooth` m), not the csv column, so it also works for hand-made lines.
With `avoid:=true` the cap is still the raceline's (per raceline point, which `/local_path` keeps one to
one), not the curvature of `/local_path`'s swerve around an obstacle.
Sim (MCL, 2026-10-29): Porto at 1.0, 0 collisions in 23 laps of 7.00 s (off: 25 collisions in 180 s);
Berlin at 1.0, 0 collisions, 12.38 s laps (off: 30 collisions); lap times at 0.5 unchanged.

### Options

Each is off by default; turn one on at a time and compare laps (`benchmark.sh`, below). `test_closed_loop.py`
drives the node closed loop against a kinematic bicycle car (`test/sim_car.py`: no tires or slip) to check what
each does to that model; what they are worth in AutoDRIVE or on the car is for the benchmark.

- `latency` (s): a command acts on the car that much later (`fit_vehicle` measures it). The pose is first driven
  forward to then, from its stamp, with the commands this node sent that act meanwhile (a Smith predictor; their
  curvature is tan(steering) / wheelbase, so calibrate the steering first). Pure pursuit turns in early and a small
  delay offsets some of that: on the kinematic car (stadium, 1.5 m half turns, 4 m/s) the RMS distance from the
  line is 0.033 m without delay, 0.015-0.018 m with 0.05-0.1 s, 0.12 m with 0.15 s, 0.25 m with 0.2 s and 0.65 m
  with 0.3 s (it weaves), and 0.033 m with any of them compensated. So leave it off below ~0.1 s (the sim's delay
  is tens of ms) and turn it on from ~0.15 s (a real car's servo, VESC and lidar together can reach that).
- `speed_preview_time` (s): the speed is taken at the car's own point plus planned speed x this, instead of at
  the target point (which brakes early into corners and runs over the profile out of them).
- `accel_feedforward`: `/drive`'s acceleration carries the profile's, for the sim adapter's throttle feedforward
  (`speed_ka`, `fit_vehicle` fits it). On the car f1tenth_system's `ackermann_to_vesc` drives the speed only.
- `sector_speeds`: the speed profile x a scale per sector (Speed per sector, below).
- `avoid_max_lateral_accel` (m/s^2): with `avoid`, a swerve round an obstacle is no longer driven at the
  raceline's speed: where `/local_path` leaves the raceline the speed is capped at sqrt(this / its curvature),
  braking into it at `avoid_max_decel`. Set it at about what the raceline was planned with (racecar.ini `ay_safe`,
  or `mue` x g).

On the same kinematic car the lookahead floor `adaptive_lookahead_min_time` decides how much pure pursuit cuts
at speed: at 7 m/s, 0.25 s (the default) 0.14 m RMS, 0.18 s 0.05 m, 0.12 s 0.02 m. The floor is there because a
short lookahead makes pure pursuit oscillate on a car with tires, which that model does not have: it is the next
thing to benchmark in AutoDRIVE, not a default to change on this evidence.

## Lap benchmark

`lap_logger_node` writes one row per flying lap to `tracks/<name>/laps.csv` and one per lap and sector to
`laps_sectors.csv`; `ws/scripts/benchmark.sh` races N laps with it unattended and prints the run next to the
track's earlier ones (`lap_summary`). A lap ends where the car crosses the raceline's s = 0 forward (the first
crossing starts the first lap); a sim respawn or reset marks the lap as interrupted.

```bash
scripts/benchmark.sh <name> --laps 5 --tag baseline speed_scale:=1.0    # race.sh + lap_logger_node, then lap_summary
scripts/benchmark.sh <name> --laps 5 --tag preview speed_scale:=1.0 speed_preview_time:=0.1
ros2 run control lap_summary <name> [<name> ...] [--last N]              # the runs side by side
ros2 run control lap_logger_node --ros-args -p track:=<name> -p tag:=<label> [-p laps:=N]   # on its own
```

Where the car is: sim ground truth (`/ground_truth/odom` through `map_to_world.yaml`, which save_map writes when
mapping in the sim), or `pose_topic` with the `/odom` speed without that file or with `pose_source:=localized`
(the real car).

| `laps.csv` column | |
|---|---|
| `lap_time_s`, `sim_lap_time_s` | crossing to crossing (interpolated between poses); the sim's own last lap (its timing line may sit elsewhere) |
| `valid`, `collisions` | 1 unless the lap was interrupted or had a collision; the sim's `collision_count` over the lap (-1 unknown) |
| `lat_rms_m`, `lat_max_m` | distance from the raceline |
| `clearance_min_m` | smallest distance from the car's footprint (`footprint_front` 0.42 / `footprint_rear` 0.08 m of base_link, `footprint_width` 0.27 m) to a mapped wall; <= 0: touching |
| `speed_mean_mps`, `speed_max_mps`, `speed_err_rms_mps` | speed, and speed minus the `/drive` speed (what the speed loop does not follow) |
| `loc_rms_m`, `loc_max_m`, `loc_yaw_rms_deg` | `pose_topic` against ground truth at the same stamps (sim only) |

`laps_sectors.csv`: `time_s`, `lat_max_m`, `clearance_min_m`, `speed_mean_mps` per lap and sector (`sectors`,
default 10 equal lengths of the raceline from s = 0).

## Speed per sector

`tune_sectors` turns the laps into a speed scale per sector (`tracks/<name>/sectors.yaml`), one step per run:
a sector that came within `--floor` (0.05 m) of a wall, or where a lap with a collision came closest, loses
`--down` (0.05); one that stayed beyond `--target` (0.12 m) gains `--step` (0.03), within [`--min`, `--max`]
(0.7, 1.3). Pure pursuit with `sector_speeds:=true` multiplies its profile by them, kept within the raceline's
own acceleration and braking where they change. It refuses a run older than sectors.yaml (driven with other
scales) and a sectors.yaml made for another raceline.

```bash
scripts/benchmark.sh <name> --tag s0 speed_scale:=1.0                       # no sectors.yaml yet: every scale 1.0
ros2 run control tune_sectors <name>                                        # writes tracks/<name>/sectors.yaml
scripts/benchmark.sh <name> --tag s1 speed_scale:=1.0 sector_speeds:=true
ros2 run control tune_sectors <name>                                        # ... until it changes nothing
```

## Vehicle identification

`sysid_recorder_node` records during any driving (wall follower laps, races, teleop: no special manoeuvre) and
`fit_vehicle` turns the recording into settings:

```bash
ros2 run control sysid_recorder_node --ros-args --params-file $(ros2 pkg prefix control)/share/control/config/vehicle.yaml
#   on the car: -p source:=odom (default ground_truth, the sim), and -p drive_topic:=/teleop when driving by
#   joystick; Ctrl-C prints the file
ros2 run control fit_vehicle ~/ws/log/sysid_<date>_<time>.csv    # [--racecar <racecar_car.ini>] on the car
```

- the steering gain g = measured path curvature / (tan(steering) / wheelbase), per speed band, from samples with
  the steering held: on the steering the sim applied it is the sim adapter's `steering_gain`
  (`autodrive_launch.py steering_gain:=g`); on the car, divide the VESC's `steering_angle_to_servo_gain` by the
  gain on the `/drive` command. The repo's notes suggest g ~ 1.8 in the sim (`wheel_odom.py`): on the kinematic
  car of `test_closed_loop.py`, a 1.8x car tracks 3x worse until calibrated.
- the delay from a `/drive` steering command to the path following it: pure pursuit's `latency`
- sim only: the throttle per m/s^2 beyond the speed feedforward: the adapter's `speed_ka` / `speed_ka_brake`
- the accelerations reached, next to racecar.ini's planning limits

## Emergency brake

`safety_node` brakes when lidar returns lie in the car's path within stopping distance, whatever the controller
and the localization think: the path is the arc of the `/drive` steering (the `/odom` yaw rate without a recent
one), `half_width` + `margin` (0.135 + 0.05 m) either side, from the front bumper (`front` 0.42 m ahead of
base_link) to speed x `reaction` (0.1 s) + speed^2 / (2 `decel` (4 m/s^2)) + `margin_ahead` (0.15 m) ahead;
`min_points` (2) returns needed. It publishes a stop (the commanded steering, speed 0) on `/safety/drive` per scan
until `hold` (0.5 s) after the path was last clear, and `/safety/active`. The sim adapter takes `/safety/drive`
before `/drive` (`safety_timeout` 0.2 s); on the car it is an input of f1tenth_system's `ackermann_mux` above
navigation and below the joystick (`config/car_mux.yaml`). A wall the arc turns away from is not in the way; one
straight ahead at the turn-in of a corner can be, at speed: tune `decel` and `reaction` on the car at low speed
first. Off in `race.launch.py` (`safety:=true`), on in `car.launch.py`.

## Real car

The stack has not run on the car yet. f1tenth_system brings up the lidar, VESC and joystick; `car.launch.py` adds:

```bash
ros2 launch f1tenth_stack bringup_launch.py
ros2 launch control car.launch.py [imu_roll:=0.0] [imu_pitch:=0.0] [imu_yaw:=0.0] [safety:=true]
scripts/record.sh <label>     # a rosbag of the run: ws/bags/<label>_<date>_<time>
```

- TF `base_link -> imu`: zero translation (Cartographer needs the IMU at base_link) and the VESC's mounting rotation
- `perception imu_odom_node`: `/odom` and TF `odom -> base_link` from the wheel speed and the IMU's gyro: f1tenth_system's
  `vesc_to_odom` takes the yaw from the steering model, which read ~55 % of the yaw rate in the sim. It learns the
  gyro's bias while the car stands still.
- `safety_node`, the emergency brake

Once, in f1tenth_system: `vesc_to_odom` with `publish_tf: false` and its `odom` remapped to `/odom/wheel`, and the
`safety` input of `config/car_mux.yaml` in its `mux.yaml`. Calibrate the VESC (`speed_to_erpm_gain`,
`steering_angle_to_servo_gain` / `offset`), check the IMU's units (`imu_odom_node` `gyro_scale`: a full turn by hand
must bring the yaw back), and measure the car into `config/vehicle_car.yaml` (wheelbase, steering lock: now
f1tenth_gym's defaults), planning's `config/racecar_car.ini` and the brake's body in `car.launch.py`. Map with
`map_track.sh <name>` driving by joystick (its followers load the sim's `vehicle.yaml`), re-plan with
`build_raceline <name> --config .../racecar_car.ini` (`map_track.sh` plans with `racecar.ini`), and race with
`vehicle:=<that vehicle_car.yaml>`, a low `max_speed` and `speed_scale` first. `lap_logger_node` on the car:
`pose_source:=localized` (no `map_to_world.yaml`) and the car's `footprint_*`. `perception mcl_node`
deskews the swept Hokuyo scans with the odometry (`deskew`, on): at 9.6 m/s with a 25 ms sweep its error fell from
0.125 to 0.038 m in `test_deskew.py`; the obstacle spliner does the same.

## Wall follower

`wall_follow_node` drives the mapping lap(s) of an unseen track: it needs only `/scan`, no map or
localization. It measures the left and right wall distances (two beams per side, projected
`lookahead` m ahead), steers with a PD controller to stay centred, and slows down in corners. With
`laps > 0` it counts returns to its start position on `pose_topic`, then stops the car and exits.

```bash
scripts/map_track.sh <name> --wall-follow [--laps 2] [speed:=1.0]   # mapping with the wall follower
ros2 launch control wall_follow.launch.py speed:=1.0 debug:=true     # on its own, until Ctrl+C
```

| Topic | Type | |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | subscribed |
| `pose_topic` (`/tracked_pose`) | `geometry_msgs/PoseStamped` | subscribed when `laps > 0` (Cartographer, also while mapping) |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | published, one command per scan; zero on exit |

| Parameter | Default | |
|---|---|---|
| `speed` | `1.0` | m/s on straights; keep it low for clean scan matching |
| `kp`, `kd` | `1.05`, `0.10` | rad per m off centre, rad per m/s of drift |
| `lookahead` | `0.8` | m ahead at which the wall distances are projected |
| `beam_angle_deg` | `45` | angle between the side beam and the forward beam |
| `max_range` | `5.0` | m, a side with no return within this has no wall |
| `track_width` | `1.2` | m, holds half of it from the wall when only one wall is visible |
| `corner_slowdown` | `0.5` | fraction of `speed` shed at full steering |
| `max_steering_angle` | `0.524` | rad, steering is clipped to this; `wall_follow.launch.py` loads it from `config/vehicle.yaml` |
| `laps` | `0` | stop after this many laps, 0 = never |
| `lap_leave_radius`, `lap_return_radius`, `lap_min_distance` | `3.0`, `1.5`, `10.0` | m: a lap is leaving the start by the first, coming back within the second, after driving at least the third |
| `debug` | `false` | log wall distances, steering and speed |

## Follow the gap with disparity extension

`disparity_gap_node` drives the mapping lap(s) of an unseen track, like the wall follower: it needs
only `/scan`, no map or localization. At every jump of more than
`disparity_jump` between neighbouring beams it extends the nearer range sideways by half the car's
width plus `safety_margin`, so every beam that is still long is a direction the whole car fits through.
It then takes the runs of beams longer than the first of `free_thresholds` (relaxing it if there are
none) within `fov`, picks the deepest run (the widest among near-equal depths), aims at its deepest
point, shifted to the middle of the narrowest spot along the way, and never turns toward something
beside the car's body. Speed scales with the clear distance straight ahead, between `min_speed` and
`max_speed`. With `laps > 0` it counts returns to its start position on `pose_topic` (the wall
follower's `LapCounter`), then stops the car and exits.

```bash
scripts/map_track.sh <name> --gap-follow [--laps 2] [speed:=1.0]   # mapping with follow the gap
ros2 launch control disparity_gap.launch.py speed:=1.0              # on its own, until Ctrl+C
```

It never runs alongside `race.sh` or the other followers: they all publish `/drive`.
It was tuned in the F1TENTH gym (Levine) and drives Porto in AutoDRIVE (2 laps at ~1 m/s, past an obstacle at the side of the track; one dead ahead is untested).

**Experimental: use `--wall-follow` for mapping.** Known issue: the gap follower takes the inside line through
corners, so its mapping lap is ~13-15 % shorter than the centerline (Porto: 26.8 m vs 30.7 m; the wall follower
drives 30.2 m). `build_raceline` rejects a centerline more than 15 % off the lap, so planning fails about every
other run ("centerline is ... m but the mapping lap is ... m"). The map itself is fine (its centerline is within
~5 cm median of a wall-follower map). Not fixed yet: drive closer to the centre, or replace the length check.

| Topic | Type | |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | subscribed |
| `pose_topic` (`/tracked_pose`) | `geometry_msgs/PoseStamped` | subscribed when `laps > 0` (Cartographer, also while mapping) |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | published, one command per scan; a stop when no beam ahead has a return, and zero on exit |

| Parameter | Default | |
|---|---|---|
| `max_speed` | `1.0` | m/s cap; the launch file's `speed`. Keep it low for clean scan matching |
| `laps` | `0` | stop after this many laps, 0 = never |
| `lap_leave_radius`, `lap_return_radius`, `lap_min_distance` | `3.0`, `1.5`, `10.0` | m, as for the wall follower |
| `min_speed` | `0.5` | m/s, unless `max_speed` is lower |
| `speed_per_metre` | `0.6` | m/s per metre of clear road within ±8° ahead, halved at full steering |
| `max_range` | `10.0` | m, longer returns are clipped to this |
| `fov` | `π/2` | rad, only beams within ± this can be aimed at |
| `car_width`, `safety_margin` | `0.27`, `0.05` | m, the extension radius is half the width plus the margin |
| `disparity_jump` | `0.35` | m between neighbouring beams that counts as an obstacle edge |
| `free_thresholds` | `[3.0, 2.0, 1.2, 0.6]` | m, tried in order until a gap exists |
| `path_lookahead`, `path_cap` | `6.0`, `1.5` | m: how far along the path to look for its narrowest spot; room beyond the cap on a side counts as open |
| `side_clearance`, `push_gain` | `0.08`, `2.0` | m kept beside the body; rad per metre something is inside it |
| `body_behind_lidar`, `body_ahead_of_lidar` | `0.45`, `0.15` | m, the car's body along x relative to the LiDAR |
| `max_steering_angle` | `0.524` | rad, steering is clipped to this; the launch file loads it from `config/vehicle.yaml` |

## Test

```bash
python3 -m pytest ws/src/control/test   # needs a sourced ROS 2 environment and the built package
```

The tests publish `/scan`, `/tracked_pose`, `/localized_pose` and read `/drive`: with the bridge running, give them their
own `ROS_DOMAIN_ID` (e.g. `ROS_DOMAIN_ID=42 python3 -m pytest ...`) so they don't drive the sim car.
`test_closed_loop.py` drives pure pursuit against `test/sim_car.py` in real time (~1.5 min).

## Credit

Adapted from the F1TENTH / UPenn ESE 6150 pure-pursuit lab template (MIT, see `LICENSE`).
