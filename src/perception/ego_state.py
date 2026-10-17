"""
Ego pose from GNSS + IMU + speedometer (no simulator ground truth).

GNSS -> CARLA world x, y with the map's geo-reference (lat_0 / lon_0 in the
OpenDRIVE header, exactly what the leaderboard and ORION's agent use:
leaderboard/utils/route_manipulation._get_latlon_ref, ORION planner.py
gps_to_location).  Heading from the IMU compass (0 = north = -y, clockwise;
yaw = compass - pi/2, checked to 0.1 deg against motion on our recordings).

A small complementary filter: predict with speed and heading every tick,
pull toward the GNSS fix with gain K.  The leaderboard's GNSS has no noise by
default, but the filter keeps the pose smooth if it does, and bridges NaN
compass readings at start-up.
"""

import math

import numpy as np

EARTH_RADIUS_EQUA = 6378137.0


def gps_to_xy(lat, lon, lat_ref, lon_ref):
    """ORION / Bench2DriveZoo RoutePlanner.gps_to_location, returning CARLA
    world (x, y)."""
    scale = math.cos(lat_ref * math.pi / 180.0)
    my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
    mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
    y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0)) - my
    x = mx - scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    return x, y


def xy_to_gps(x, y, lat_ref, lon_ref):
    """Inverse of gps_to_xy (for tests)."""
    scale = math.cos(lat_ref * math.pi / 180.0)
    lon = (x + scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0) * 180.0 / (
        math.pi * EARTH_RADIUS_EQUA * scale)
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0)) - y
    lat = math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) * 360.0 / math.pi - 90.0
    return lat, lon


def compass_to_yaw(compass):
    return compass - math.pi / 2


class EgoStateEstimator:
    K = 0.3          # GNSS correction gain per tick

    def __init__(self, lat_ref, lon_ref, antenna_x=0.0):
        # GNSS reports the ANTENNA position; ours (ORION's rig) sits 1.4 m
        # behind the vehicle origin -- uncorrected, a constant 1.4 m pose error.
        self.lat_ref, self.lon_ref = lat_ref, lon_ref
        self.antenna_x = antenna_x
        self.x = self.y = self.yaw = None
        self.speed = 0.0
        self.yaw_rate = 0.0

    def update(self, gnss, compass, speed, gyro_z=0.0, dt=0.05):
        """gnss (lat, lon[, alt]); compass rad (may be NaN); speed m/s;
        gyro_z rad/s.  Returns (x, y, yaw, speed, yaw_rate)."""
        gx, gy = gps_to_xy(float(gnss[0]), float(gnss[1]), self.lat_ref, self.lon_ref)
        if compass is not None and np.isfinite(compass):
            yaw_c = compass_to_yaw(float(compass))
        else:
            yaw_c = self.yaw if self.yaw is not None else 0.0
        gx -= self.antenna_x * math.cos(yaw_c)        # antenna -> vehicle origin
        gy -= self.antenna_x * math.sin(yaw_c)
        if self.x is None:
            self.x, self.y, self.yaw = gx, gy, yaw_c
        else:
            px = self.x + speed * math.cos(self.yaw) * dt
            py = self.y + speed * math.sin(self.yaw) * dt
            self.x = px + self.K * (gx - px)
            self.y = py + self.K * (gy - py)
            self.yaw = yaw_c
        self.speed = float(speed)
        self.yaw_rate = float(gyro_z) if np.isfinite(gyro_z) else 0.0
        return self.x, self.y, self.yaw, self.speed, self.yaw_rate
