"""ros2 launch autodrive_bridge autodrive_launch.py [publish_restricted_topics:=true] [publish_camera:=false]
        [port:=4567] [steering_gain:=..] [speed_ka:=..] [speed_ka_brake:=..] [max_acceleration:=..]
        [max_deceleration:=..]

fast_bridge: AutoDRIVE app <-> ROS 2 over Socket.IO on :4567 (/autodrive/roboracer_1/*)
adapter:     sim sensors -> /scan, /sensors/imu/raw, /ground_truth/odom;  /drive -> throttle/steering
wheel_odom:  rear-wheel encoders -> /odom + TF odom -> base_link (sim stand-in for vesc_to_odom)
foxglove_bridge on :8765

Teleop runs in its own terminal: ros2 run autodrive_bridge teleop_keyboard

steering_gain, speed_ka, speed_ka_brake, max_acceleration, max_deceleration override the adapter's values in
config/adapter.yaml for this run (control fit_vehicle prints them); left out, adapter.yaml's stand.

Stop it with Ctrl-C in its terminal (or kill -INT -- -<its process group>): a SIGINT to the launch
process alone reaches no node, they get SIGTERM only 5 s later. When fast_bridge exits (it could not
bind the port: the old bridge still runs) the whole launch ends, rather than running a second
adapter and wheel_odom beside the old ones.
"""
import tempfile

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# adapter parameters a launch argument may override (empty: adapter.yaml's value)
ADAPTER_OVERRIDES = ('steering_gain', 'speed_ka', 'speed_ka_brake', 'max_acceleration', 'max_deceleration')


def adapter(context, vehicle, config):
    # in a params file of their own, under adapter: as in adapter.yaml, after it. A parameters dict would go in a
    # /** (any node) section, and adapter.yaml's adapter: section beats that whatever the order
    overrides = {}
    for name in ADAPTER_OVERRIDES:
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            try:
                overrides[name] = float(value)
            except ValueError:
                raise RuntimeError(f'{name}:={value} is not a number')
    params = [vehicle, config]
    if overrides:
        with tempfile.NamedTemporaryFile('w', prefix='adapter_overrides_', suffix='.yaml', delete=False) as f:
            yaml.safe_dump({'adapter': {'ros__parameters': overrides}}, f)
        params.append(f.name)
    return [Node(package='autodrive_bridge', executable='adapter', name='adapter', output='screen',
                 parameters=params)]


def generate_launch_description():
    vehicle = PathJoinSubstitution([FindPackageShare('control'), 'config', 'vehicle.yaml'])
    config = PathJoinSubstitution([FindPackageShare('autodrive_bridge'), 'config', 'adapter.yaml'])

    return LaunchDescription([
        # restricted = sim ground truth and lap stats, off in competition
        DeclareLaunchArgument('publish_restricted_topics', default_value='true'),
        # nothing in the stack uses the camera, and decoding it every frame slows the sim loop
        DeclareLaunchArgument('publish_camera', default_value='false'),
        DeclareLaunchArgument('port', default_value='4567'),   # the sim's Connect port
        *[DeclareLaunchArgument(name, default_value='', description=f"adapter {name}; empty: adapter.yaml's")
          for name in ADAPTER_OVERRIDES],
        Node(package='autodrive_bridge', executable='fast_bridge', name='fast_bridge', output='screen',
             parameters=[{'publish_restricted_topics': LaunchConfiguration('publish_restricted_topics'),
                          'publish_camera': LaunchConfiguration('publish_camera'),
                          'port': LaunchConfiguration('port')}],
             on_exit=Shutdown(reason='fast_bridge exited')),
        OpaqueFunction(function=adapter, args=[vehicle, config]),
        Node(package='autodrive_bridge', executable='wheel_odom', name='wheel_odom', output='screen',
             parameters=[vehicle, config]),
        Node(package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge'),
    ])
