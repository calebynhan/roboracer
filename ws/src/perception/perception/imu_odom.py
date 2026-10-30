"""Wheel speed + gyro odometry (imu_odom_node), the car's counterpart of the sim's wheel_odom with use_imu_yaw_rate.

f1tenth_system's vesc_to_odom integrates the yaw from the steering angle, the bicycle model that read only ~55 %
of the sim's real yaw rate (autodrive_bridge wheel_odom.py), so Cartographer and the particle filter got turns
the car did not make. Here the yaw rate is the IMU's gyro instead:

    yaw += (gyro_z - bias) dt,   x, y += speed dt along the mean heading of the step

The gyro's bias is learned while the car stands still (|speed| below still_speed for still_time s): an
exponential average of the gyro then, and the yaw held (zero velocity update), so a parked car does not turn.
"""
import math


class GyroOdometry:
    def __init__(self, still_speed=0.02, still_time=1.0, bias_alpha=0.02, max_dt=0.1):
        self.still_speed, self.still_time = still_speed, still_time
        self.bias_alpha, self.max_dt = bias_alpha, max_dt
        self.x = self.y = self.yaw = 0.0
        self.bias = 0.0
        self.bias_samples = 0
        self.speed = None           # newest wheel speed
        self.moving_t = None        # time the car was last seen moving (None: never seen)
        self.t = None               # time of the last gyro sample integrated
        self.yaw_rate = 0.0

    def set_speed(self, t, speed):
        self.speed = speed
        if abs(speed) >= self.still_speed or self.moving_t is None:
            self.moving_t = t

    def still(self, t):
        """Below still_speed now and since still_time s (not just no wheel message for that long)."""
        return (self.moving_t is not None and abs(self.speed) < self.still_speed
                and t - self.moving_t >= self.still_time)

    def gyro(self, t, wz):
        """Integrate one gyro sample. False when there is nothing new to publish: before a wheel speed, at the
        first gyro sample after it (which only starts the clock), and for a sample not newer than the last (dropped:
        /odom's stamps must keep increasing, Cartographer aborts on unsorted data)."""
        if self.speed is None:
            return False
        if self.t is not None and t <= self.t:
            return False
        prev, self.t = self.t, t
        if prev is None:
            return False
        dt = min(t - prev, self.max_dt)
        if self.still(t):
            # standing still: the gyro reads its bias; the heading is held
            self.bias += (1.0 if self.bias_samples == 0 else self.bias_alpha) * (wz - self.bias)
            self.bias_samples += 1
            self.yaw_rate = 0.0
            return True
        self.yaw_rate = wz - self.bias
        dyaw = self.yaw_rate * dt
        heading = self.yaw + 0.5 * dyaw
        self.x += self.speed * dt * math.cos(heading)
        self.y += self.speed * dt * math.sin(heading)
        self.yaw = math.atan2(math.sin(self.yaw + dyaw), math.cos(self.yaw + dyaw))
        return True
