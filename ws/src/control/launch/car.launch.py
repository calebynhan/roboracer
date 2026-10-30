"""The stack's additions to f1tenth_system's bringup on the real car: IMU transform, gyro odometry, emergency brake.

    ros2 launch f1tenth_stack bringup_launch.py     # f1tenth_system: lidar, VESC, joystick, ackermann_mux
    ros2 launch control car.launch.py [vehicle:=<vehicle.yaml>] [safety:=true] [imu_roll:=0.0] [imu_pitch:=0.0]
        [imu_yaw:=0.0]

Then map with map_track.sh <name>, driving the laps by joystick (its --wall-follow / --gap-follow launches load
the sim's vehicle.yaml), re-plan with the car's model:
    ros2 run planning build_raceline <name> --config $(ros2 pkg prefix planning)/share/planning/config/racecar_car.ini
(map_track.sh plans with racecar.ini, the sim car), and race with race.sh <name> vehicle:=<config/vehicle_car.yaml>.

Starts:
  static TF base_link -> imu   zero translation (Cartographer needs the IMU at base_link, its tracking frame) and
                               the VESC's mounting rotation (imu_roll / imu_pitch / imu_yaw, rad)
  perception imu_odom_node     /odom/wheel (vesc_to_odom) + /sensors/imu/raw -> /odom + TF odom -> base_link: the
                               yaw from the gyro, not from vesc_to_odom's steering model
  control safety_node          safety:=true (default): emergency brake on /safety/drive, with the F1TENTH car's
                               body (half_width 0.155 m, front 0.45 m: measure yours)
f1tenth_system, once (see control/README.md, Real car):
  vesc_to_odom: publish_tf false, its odom topic remapped to /odom/wheel (imu_odom_node publishes /odom and the TF)
  ackermann_mux: the safety input of config/car_mux.yaml, above navigation (/drive) and below the joystick
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    vehicle = LaunchConfiguration('vehicle')
    return LaunchDescription([
        DeclareLaunchArgument('vehicle',
                              default_value=PathJoinSubstitution([FindPackageShare('control'), 'config',
                                                                  'vehicle_car.yaml']),
                              description="the car's vehicle.yaml (wheelbase, steering lock)"),
        DeclareLaunchArgument('safety', default_value='true', description='start the emergency brake'),
        DeclareLaunchArgument('imu_roll', default_value='0.0', description='rad, the IMU mounting rotation'),
        DeclareLaunchArgument('imu_pitch', default_value='0.0', description='rad'),
        DeclareLaunchArgument('imu_yaw', default_value='0.0', description='rad'),
        Node(package='tf2_ros', executable='static_transform_publisher', name='base_link_to_imu',
             arguments=['--x', '0', '--y', '0', '--z', '0',
                        '--roll', LaunchConfiguration('imu_roll'), '--pitch', LaunchConfiguration('imu_pitch'),
                        '--yaw', LaunchConfiguration('imu_yaw'), '--frame-id', 'base_link', '--child-frame-id', 'imu']),
        Node(package='perception', executable='imu_odom_node', name='imu_odom', output='screen'),
        # the F1TENTH car's body (f1tenth_gym: 0.31 m wide; rear axle to front bumper ~0.45 m, estimated): measure
        Node(package='control', executable='safety_node', name='safety_node', output='screen',
             parameters=[vehicle, {'half_width': 0.155, 'front': 0.45}],
             condition=IfCondition(LaunchConfiguration('safety'))),
    ])
