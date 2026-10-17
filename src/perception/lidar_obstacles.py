"""
LiDAR obstacles without a learned detector, for the sensor-only Bench2Drive
setting (no simulator ground truth, no HD map).

    tracker = LidarObstacleTracker(mount=(1.3, 0.0, 2.5, 0.0))
    tracks = tracker.update(points_sensor, ego_xy_yaw=(x, y, yaw), t=timestamp)

Pipeline per tick:
  1. sensor frame -> vehicle frame (mount offset + yaw); drop the ego's own box
  2. ground removal on a 2 m grid: a point is an obstacle point if it is
     0.25-2.8 m above the lowest point of its cell (local ground; handles
     slopes) -- no plane fit, robust to the Town12/13 hills
  3. vehicle frame -> world frame with the ESTIMATED ego pose
  4. merge with the previous tick's obstacle points: Bench2Drive's LiDAR spins
     at 10 Hz against a 20 Hz tick, so one message is half a sweep
  5. cluster on a 0.4 m occupancy grid, gaps up to ~1.2 m bridged
     (8-connected components)
  6. track clusters by nearest neighbour with constant-velocity prediction;
     velocity = smoothed centroid motion; 'stopped_for' = time since the
     track last moved > 1 m/s
  7. guess a coarse type from the footprint: vehicle / walker / prop

Everything physically on the road shows up -- cars, pedestrians, cones,
warning signs -- and things hidden behind others really are hidden.  Which
clusters matter (the road corridor) is decided by the caller in route
coordinates (carlaEnv._set_obstacles_from_tracks).

Point format: CARLA's LidarMeasurement as the leaderboard parses it, (N, 4)
float32 x, y, z, intensity in the sensor frame (x forward, y right, z up --
Unreal's left-handed frame, the same one carla.Transform uses).
"""

import math

import numpy as np

try:
    from scipy import ndimage
except ImportError:          # pragma: no cover
    ndimage = None


class LidarObstacleTracker:
    GROUND_CELL = 2.0
    MIN_H, MAX_H = 0.25, 2.8      # above local ground
    CLUSTER_CELL = 0.4
    DILATE = 1                    # bridge gaps up to ~(2*DILATE+1)*cell between points
    MIN_POINTS = 3
    RANGE_FWD, RANGE_BACK, RANGE_SIDE = 70.0, 25.0, 25.0
    EGO_HALF = (2.6, 1.2)         # ego footprint to drop (vehicle frame)
    GATE_M = 2.5                  # association radius
    COAST_S = 0.3                 # keep unmatched tracks this long
    V_ALPHA = 0.25                # velocity smoothing
    MOVING_V = 1.0
    V_DEADBAND = 0.6              # centroid jitter on static objects reaches ~0.5 m/s

    def __init__(self, mount=(1.3, 0.0, 2.5, 0.0)):
        self.mx, self.my, self.mz, myaw = mount
        self.myaw = math.radians(myaw)
        self.prev_pts = None
        self.tracks = {}
        self.next_id = 0
        self.t_prev = None

    # ── geometry ──────────────────────────────────────────────────────────

    def _obstacle_points(self, pts, ego):
        """(M, 3) world x, y, height-above-ground of obstacle points."""
        if pts is None or len(pts) == 0:
            return np.zeros((0, 3))
        p = np.asarray(pts, dtype=np.float64)[:, :3]
        c, s = math.cos(self.myaw), math.sin(self.myaw)
        vx = self.mx + c * p[:, 0] - s * p[:, 1]
        vy = self.my + s * p[:, 0] + c * p[:, 1]
        vz = self.mz + p[:, 2]
        keep = ((vx < self.RANGE_FWD) & (vx > -self.RANGE_BACK) & (np.abs(vy) < self.RANGE_SIDE)
                & ~((np.abs(vx) < self.EGO_HALF[0]) & (np.abs(vy) < self.EGO_HALF[1])))
        vx, vy, vz = vx[keep], vy[keep], vz[keep]
        if len(vx) == 0:
            return np.zeros((0, 3))
        gi = np.floor(vx / self.GROUND_CELL).astype(np.int64)
        gj = np.floor(vy / self.GROUND_CELL).astype(np.int64)
        key = (gi - gi.min()) * 100000 + (gj - gj.min())
        uniq, inv = np.unique(key, return_inverse=True)
        cell_min = np.full(len(uniq), np.inf)
        np.minimum.at(cell_min, inv, vz)
        h = vz - cell_min[inv]
        ob = (h > self.MIN_H) & (h < self.MAX_H)
        vx, vy, h = vx[ob], vy[ob], h[ob]
        ex, ey, eyaw = ego
        ce, se = math.cos(eyaw), math.sin(eyaw)
        wx = ex + ce * vx - se * vy
        wy = ey + se * vx + ce * vy
        return np.stack([wx, wy, h], axis=1)

    def _clusters(self, P):
        if len(P) == 0:
            return []
        cell = self.CLUSTER_CELL
        ij = np.floor(P[:, :2] / cell).astype(np.int64)
        lo = ij.min(axis=0)
        ij -= lo
        shape = tuple(ij.max(axis=0) + 1)
        grid = np.zeros(shape, dtype=bool)
        grid[ij[:, 0], ij[:, 1]] = True
        if ndimage is not None:
            # A car's LiDAR return is sparse (near faces only, gaps between
            # rings); without bridging, one car became 3-5 small clusters and
            # its track restarted every tick (velocity never estimated).
            conn = ndimage.binary_dilation(grid, structure=np.ones((3, 3), bool),
                                           iterations=self.DILATE) if self.DILATE else grid
            lab, n = ndimage.label(conn, structure=np.ones((3, 3), dtype=int))
        else:                                               # pragma: no cover
            raise RuntimeError('scipy is required for LiDAR clustering')
        pl = lab[ij[:, 0], ij[:, 1]]
        out = []
        for k in range(1, n + 1):
            m = pl == k
            if m.sum() < self.MIN_POINTS:
                continue
            q = P[m]
            xy = q[:, :2]
            cen = xy.mean(axis=0)
            if len(xy) >= 3:
                ev, evec = np.linalg.eigh(np.cov((xy - cen).T) + 1e-6 * np.eye(2))
                proj = (xy - cen) @ evec
                ext = proj.max(axis=0) - proj.min(axis=0)
                length, width = float(max(ext)), float(min(ext))
            else:
                length = width = 0.3
            out.append(dict(x=float(cen[0]), y=float(cen[1]), length=max(length, 0.3),
                            width=max(width, 0.3), height=float(q[:, 2].max()),
                            n=int(m.sum()), pts=xy))
        return out

    @staticmethod
    def _type(c):
        L, W, H = c['length'], c['width'], c['height']
        if L > 2.5 or (L > 1.6 and W > 1.2):
            return 'vehicle.lidar'
        if L < 1.3 and W < 1.0 and H > 1.2:
            return 'walker.lidar'
        if 1.2 <= L <= 2.4 and W < 1.0:
            return 'vehicle.bike.lidar'
        return 'static.prop.lidar'

    # ── tracking ──────────────────────────────────────────────────────────

    def update(self, points_sensor, ego_xy_yaw, t):
        dt = 0.05 if self.t_prev is None else max(t - self.t_prev, 1e-3)
        self.t_prev = t
        cur = self._obstacle_points(points_sensor, ego_xy_yaw)
        merged = cur if self.prev_pts is None else np.vstack([cur, self.prev_pts])
        self.prev_pts = cur
        clusters = self._clusters(merged)

        # predict, then greedy nearest-neighbour association
        for tr in self.tracks.values():
            tr['px'] = tr['x'] + tr['vx'] * dt
            tr['py'] = tr['y'] + tr['vy'] * dt
        pairs = []
        for ci, c in enumerate(clusters):
            for tid, tr in self.tracks.items():
                d = math.hypot(c['x'] - tr['px'], c['y'] - tr['py'])
                if d < self.GATE_M:
                    pairs.append((d, ci, tid))
        pairs.sort()
        used_c, used_t = set(), set()
        for d, ci, tid in pairs:
            if ci in used_c or tid in used_t:
                continue
            used_c.add(ci)
            used_t.add(tid)
            c, tr = clusters[ci], self.tracks[tid]
            vx_m = (c['x'] - tr['x']) / dt
            vy_m = (c['y'] - tr['y']) / dt
            a = self.V_ALPHA
            tr['vx'] = (1 - a) * tr['vx'] + a * vx_m
            tr['vy'] = (1 - a) * tr['vy'] + a * vy_m
            tr.update(x=c['x'], y=c['y'], length=c['length'], width=c['width'],
                      height=max(tr['height'] * 0.8, c['height']), n=c['n'],
                      pts=c['pts'], last_seen=t, age=tr['age'] + dt)
            if math.hypot(tr['vx'], tr['vy']) > self.MOVING_V:
                tr['last_moving'] = t
            tr['type'] = self._type(tr)
        for ci, c in enumerate(clusters):
            if ci in used_c:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = dict(id=tid, x=c['x'], y=c['y'], vx=0.0, vy=0.0,
                                    length=c['length'], width=c['width'], height=c['height'],
                                    n=c['n'], pts=c['pts'], last_seen=t, age=0.0,
                                    last_moving=None, type=self._type(c))
        for tid in [k for k, tr in self.tracks.items() if t - tr['last_seen'] > self.COAST_S]:
            del self.tracks[tid]

        out = []
        for tr in self.tracks.values():
            if tr['last_seen'] != t:
                continue
            # A young track's velocity is not trustworthy yet, and below the
            # deadband it is centroid jitter (visible faces change), not motion.
            still = tr['age'] < 0.25 or math.hypot(tr['vx'], tr['vy']) < self.V_DEADBAND
            out.append(dict(id=tr['id'], x=tr['x'], y=tr['y'],
                            vx=0.0 if still else tr['vx'], vy=0.0 if still else tr['vy'],
                            length=tr['length'], width=tr['width'], height=tr['height'],
                            type=tr['type'], pts=tr['pts'],
                            stopped_for=(99.0 if tr['last_moving'] is None
                                         else t - tr['last_moving'])))
        return out
