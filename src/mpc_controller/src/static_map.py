"""
Static scene structure from the map, for observation v3.

86% of the nominal MPCC's collisions (A5000 b0, 2026-10-13) are with static
objects -- poles, fences, walls, signs -- which neither the CBF (vehicles
only) nor the v2 observation contains.  The ORION teacher sees them on its
cameras, so a student without them is asked to copy reactions to things it
cannot see.  Here they come from the map, as an HD map would provide them on
a real car: no extra network, one vectorised lookup per step.

One StaticMap per town, built from world.get_level_bbs() with a CARLA server
(StaticMap.from_world) and cached as an npz, so offline dataset augmentation
(no server) and the env at run time use the identical table:

    python tools/student/dump_static_map.py --town Town01      # once per town

Features (StaticMap.features): the N_SLOTS nearest objects, distance measured
to the nearest point of each object's footprint (a building is a big box, not
its centre), restricted to a window around the ego and expressed in the ego
frame: per slot (present, forward, lateral).  Empty slots are
(0, FAR_FWD, 0), like the v2 obstacle slots.
"""

import collections
import os

import numpy as np

N_SLOTS = 8
PER_SLOT = 3
FEATURE_DIM = N_SLOTS * PER_SLOT
FWD_MIN, FWD_MAX = -3.0, 30.0   # m, window in the ego frame
LAT_MAX = 10.0
FAR_FWD = 30.0
# Objects whose bottom is this far above the local road are overhead
# (traffic-light arms, signs on gantries, bridges) -- not collidable.
OVERHEAD_M = 2.5
# Lane-centre spacing used to find boxes that cover the road.
LANE_SAMPLE_M = 1.0

# carla.CityObjectLabel names; road surfaces, sidewalks and terrain are
# excluded (always near, not obstacles), as are bridges (overhead).
LABELS = ('Buildings', 'Walls', 'Fences', 'Poles', 'TrafficLight', 'TrafficSigns',
          'Vegetation', 'GuardRail', 'Static', 'Other')

CACHE_DIR = os.environ.get('STATIC_MAP_DIR') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), 'data', 'static_maps')


def cache_path(town):
    return os.path.join(CACHE_DIR, f'{town}.npz')


class StaticMap:
    def __init__(self, cx, cy, ex, ey, yaw, label):
        self.cx, self.cy = np.asarray(cx, float), np.asarray(cy, float)
        self.ex, self.ey = np.asarray(ex, float), np.asarray(ey, float)
        self.yaw = np.asarray(yaw, float)
        self.label = np.asarray(label)
        self.c, self.s = np.cos(self.yaw), np.sin(self.yaw)
        self.reach = np.hypot(self.ex, self.ey)   # centre -> farthest corner

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def from_world(cls, world, carla_map):
        import carla
        rows = []
        for name in LABELS:
            label = getattr(carla.CityObjectLabel, name, None)
            if label is None:
                continue
            for bb in world.get_level_bbs(label):
                loc = bb.location
                ground = carla_map.get_waypoint(loc, project_to_road=True)
                gz = ground.transform.location.z if ground is not None else loc.z
                if (loc.z - bb.extent.z) - gz > OVERHEAD_M:
                    continue
                rows.append((loc.x, loc.y, bb.extent.x, bb.extent.y,
                             np.deg2rad(bb.rotation.yaw), name))
        if not rows:
            raise RuntimeError('get_level_bbs returned no static objects')
        cx, cy, ex, ey, yaw, label = zip(*rows)
        sm = cls(cx, cy, ex, ey, yaw, label)
        # Some level boxes are merged geometry spanning the map (Town01 has
        # one 3 km across), so every pose is "inside" them.  Nothing real
        # stands on a driving-lane centreline: drop any box covering one.
        lanes = [w for w in carla_map.generate_waypoints(LANE_SAMPLE_M)
                 if w.lane_type == carla.LaneType.Driving]
        pts = np.array([[w.transform.location.x, w.transform.location.y] for w in lanes])
        return sm.without_boxes_covering(pts)

    def contains(self, i, px, py):
        """bool mask: which of the points lie inside box i's footprint."""
        dx, dy = px - self.cx[i], py - self.cy[i]
        lx = self.c[i] * dx + self.s[i] * dy
        ly = -self.s[i] * dx + self.c[i] * dy
        return (np.abs(lx) <= self.ex[i]) & (np.abs(ly) <= self.ey[i])

    def without_boxes_covering(self, pts):
        """Copy without the boxes that contain any of pts (N, 2)."""
        keep = np.ones(len(self.cx), bool)
        for i in range(len(self.cx)):
            near = np.hypot(pts[:, 0] - self.cx[i], pts[:, 1] - self.cy[i]) <= self.reach[i]
            if near.any() and self.contains(i, pts[near, 0], pts[near, 1]).any():
                keep[i] = False
        self.dropped = collections.Counter(self.label[~keep].tolist())
        out = StaticMap(self.cx[keep], self.cy[keep], self.ex[keep], self.ey[keep],
                        self.yaw[keep], self.label[keep])
        out.dropped = self.dropped
        return out

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, cx=self.cx, cy=self.cy, ex=self.ex, ey=self.ey,
                 yaw=self.yaw, label=self.label)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d['cx'], d['cy'], d['ex'], d['ey'], d['yaw'], d['label'])

    @classmethod
    def for_town(cls, town, world=None, carla_map=None):
        """Cached table if present, else build from the server and cache it."""
        path = cache_path(town)
        if os.path.exists(path):
            return cls.load(path)
        if world is None:
            raise FileNotFoundError(
                f'{path} missing -- run tools/student/dump_static_map.py --town {town}')
        sm = cls.from_world(world, carla_map)
        sm.save(path)
        return sm

    # ── query ───────────────────────────────────────────────────────────

    def nearest_points(self, x, y, radius):
        """Nearest footprint point of every object within `radius` of (x, y).
        Returns (px, py, dist, idx)."""
        dx, dy = x - self.cx, y - self.cy
        near = np.hypot(dx, dy) - self.reach < radius
        idx = np.nonzero(near)[0]
        dx, dy = dx[idx], dy[idx]
        c, s = self.c[idx], self.s[idx]
        lx = np.clip(c * dx + s * dy, -self.ex[idx], self.ex[idx])
        ly = np.clip(-s * dx + c * dy, -self.ey[idx], self.ey[idx])
        px = self.cx[idx] + c * lx - s * ly
        py = self.cy[idx] + s * lx + c * ly
        return px, py, np.hypot(px - x, py - y), idx

    @staticmethod
    def empty_features():
        out = np.zeros((N_SLOTS, PER_SLOT))
        out[:, 1] = FAR_FWD
        return out.ravel()

    def features(self, x, y, yaw):
        """(FEATURE_DIM,) static-object slots around an ego at (x, y) with
        heading yaw (radians, CARLA world frame)."""
        out = self.empty_features().reshape(N_SLOTS, PER_SLOT)
        px, py, dist, _ = self.nearest_points(x, y, FWD_MAX + LAT_MAX)
        if len(px):
            rx, ry = px - x, py - y
            ch, sh = np.cos(yaw), np.sin(yaw)
            fwd = ch * rx + sh * ry
            lat = -sh * rx + ch * ry
            ok = (fwd >= FWD_MIN) & (fwd <= FWD_MAX) & (np.abs(lat) <= LAT_MAX)
            order = np.argsort(dist[ok])[:N_SLOTS]
            k = len(order)
            out[:k, 0] = 1.0
            out[:k, 1] = fwd[ok][order]
            out[:k, 2] = lat[ok][order]
        return out.ravel()


def yaw_from_compass(compass):
    """CARLA IMU compass (0 = north = -y, clockwise) -> world yaw, radians."""
    return np.asarray(compass, float) - np.pi / 2
