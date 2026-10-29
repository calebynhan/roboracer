"""Latency compensation for pure pursuit: where the car will be when a command takes effect (numpy only)."""
import math


def predict_pose(x, y, yaw, speed, yaw_rate, dt):
    """The pose after dt s at constant speed and yaw rate (an arc; a straight line when yaw_rate ~ 0)."""
    if dt <= 0.0:
        return x, y, yaw
    dyaw = yaw_rate * dt
    if abs(dyaw) < 1e-6:
        return x + speed * dt * math.cos(yaw), y + speed * dt * math.sin(yaw), yaw
    r = speed / yaw_rate
    return (x + r * (math.sin(yaw + dyaw) - math.sin(yaw)),
            y - r * (math.cos(yaw + dyaw) - math.cos(yaw)),
            math.atan2(math.sin(yaw + dyaw), math.cos(yaw + dyaw)))
