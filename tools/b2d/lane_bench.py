#!/usr/bin/env python
"""
Offline lane-centre benchmark on recorded Bench2Drive episodes: how far is
the lane centre the sensor-only agent steers by from the TRUE one?

    python tools/b2d/lane_bench.py results/b2d/rec_train_clips2/rec/* \\
        [--max-episodes 10] [--device cpu|cuda] [--csv lane_bench.csv]

Needs recordings made after 2026-10-19 (meta 'gt.route_fl' = the true lane
centre ahead in the true ego frame) with CAM_FRONT and route.json.  No CARLA.

For every tick the same YOLOP output (every 2nd tick, as the agent) feeds
three lane estimates, each driving its own lane-tracking reference (ref0):
  spline   the densified sparse route alone (no camera)
  legacy   the agent's tracker until 2026-10-19, ported faithfully from
           mpcc_b2d_agent._update_lane + CarlaMPCEnv.set_lane_offsets /
           lane_offset (half-lane jump rejection, re-anchor after 10 rejects,
           30 % blend of the last window, fade to the spline 20 m past it)
  belief   perception/lane_belief.py
Pose: the recorded true pose plus GNSS-like noise (AR(1), sigma 0.26 m; yaw
0.3 deg), i.e. what the estimator gives.  Errors: |lateral distance| between
the estimated and true lane centre at 5 / 15 / 25 m ahead, in the true ego
frame -- overall and split by condition (YOLOP fit fresh vs blind > 1 s,
junction, ego off its lane = overtaking, route lane change).
Images are the recording's JPEG q20 (ORION's format) -- slightly worse than
the live 1600x900 frame, so absolute errors are a little pessimistic; the
comparison between trackers is what this is for.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from global_planner.src.frenet_world_converter_python import FrenetConverter  # noqa: E402
from perception.lane_belief import LaneBelief, fit_points_on_route  # noqa: E402
from perception.lanes import CameraGeometry, LanePerception, fit_lane_centre  # noqa: E402
from perception.route_geometry import densify_route, route_sag_bound  # noqa: E402

CAM = dict(W=1600, H=900, fov_deg=70.0, x=0.80, y=0.0, z=1.60)   # ORION_CAMS[0], record mode
AHEAD = (5.0, 15.0, 25.0)


class Route:
    def __init__(self, nodes):
        xy = [(n[0], n[1]) for n in nodes]
        ncmd = [int(n[3]) for n in nodes]
        dense, seg = densify_route(xy)
        nb = route_sag_bound(xy)
        self.fc = FrenetConverter(dense)
        self.cmds = np.array([ncmd[k] for k in seg])
        self.bound = np.array([nb[k] for k in seg])
        ws, hint = [], None
        for x, y in dense:
            s, _, _ = self.fc.world_to_frenet(x, y, 0, s_hint=hint)
            ws.append(s)
            hint = s
        self.s = np.array(ws)
        self.junction = np.isin(self.cmds, (1, 2, 3))

    def in_junction(self, s):
        near = np.abs(self.s - s) <= 1.0
        return bool(np.any(self.junction[near]))

    def at(self, arr, s):
        return arr[min(int(np.searchsorted(self.s, s)), len(arr) - 1)]


class LegacyLane:
    """mpcc_b2d_agent._update_lane + CarlaMPCEnv.set_lane_offsets/lane_offset."""
    JUMP, REACQ, BLEND, DECAY = 1.75, 10, 0.3, 20.0

    def __init__(self, route):
        self.r, self.ls, self.ld, self.rejects = route, None, None, 0

    def offset(self, s):
        if self.ls is None:
            return 0.0
        if s <= self.ls[0]:
            return float(self.ld[0])
        if s >= self.ls[-1]:
            return float(self.ld[-1]) * max(0.0, 1.0 - (s - self.ls[-1]) / self.DECAY)
        return float(np.interp(s, self.ls, self.ld))

    def ref0(self, s, d):
        lane_change = self.r.at(self.r.cmds, s + 10.0) in (5, 6)
        if self.ls is None or lane_change or self.rejects >= self.REACQ:
            return -d
        return self.offset(s) - d

    def update(self, fit, ref0, ex, ey, yaw, s):
        if fit is None:
            return
        c0, c1, c2 = fit['coeffs']
        xn = max(fit['x_min'], 6.0)
        if abs(c0 + c1 * xn + c2 * xn * xn - ref0) > self.JUMP:
            self.rejects += 1
            return
        self.rejects = 0
        ss, dd, _ = fit_points_on_route(fit, ex, ey, yaw, self.r.fc, s, step=2.5)
        out = []
        for s_q, d_q in zip(ss, dd):
            b = self.r.at(self.r.bound, s_q)
            out.append(0.0 if self.r.in_junction(s_q) else float(np.clip(d_q, -b, b)))
        if len(ss) >= 2 and np.all(np.diff(ss) > 0):
            d_arr = np.array(out)
            if self.ls is not None:
                prev = np.array([self.offset(x) for x in ss])
                d_arr = (1 - self.BLEND) * prev + self.BLEND * d_arr
            self.ls, self.ld = ss, d_arr


class BeliefLane:
    def __init__(self, route):
        self.r = route
        self.b = LaneBelief(prior_sigma_fn=lambda s: route.at(route.bound, s) / 2.0,
                            junction_fn=route.in_junction)

    def offset(self, s):
        return self.b.offset(s)

    def ref0(self, s, d):
        self.b.advance(s)
        if self.r.at(self.r.cmds, s + 10.0) in (5, 6):
            self.b.reset_range(s, s + 40.0)
            return -d
        return self.b.offset(s + 5.0) - d

    def update(self, fit, ref0, ex, ey, yaw, s):
        if fit is None:
            return
        ss, dd, xx = fit_points_on_route(fit, ex, ey, yaw, self.r.fc, s, step=1.0)
        if len(ss) >= 3:
            self.b.update_fit(ss, dd, xx, fit.get('two_line_frac', 1.0))


class SplineLane:
    def offset(self, s):
        return 0.0

    def ref0(self, s, d):
        return -d

    def update(self, *a):
        pass


def lateral_errors(route, offset_fn, s, gt_route, tx, ty, tyaw):
    """|est - true| lateral (left) at AHEAD metres, in the true ego frame."""
    cy, sy = math.cos(tyaw), math.sin(tyaw)
    f_e, l_e = [], []
    for ds in np.arange(0.0, 44.0, 2.0):
        sk = min(s + ds, route.fc.get_path_length())
        wx, wy, _ = route.fc.frenet_to_world(sk, offset_fn(sk), 0.0)
        dx, dy = wx - tx, wy - ty
        f_e.append(cy * dx + sy * dy)
        l_e.append(sy * dx - cy * dy)
    g = np.asarray(gt_route, float)
    f_e, l_e = np.asarray(f_e), np.asarray(l_e)
    out = []
    for a in AHEAD:
        if g[:, 0].max() < a or f_e.max() < a or not np.all(np.diff(f_e) > 0):
            out.append(np.nan)
            continue
        o = np.argsort(g[:, 0])
        out.append(abs(np.interp(a, f_e, l_e) - np.interp(a, g[o, 0], g[o, 1])))
    return out


def run_episode(ep, lanes, rng):
    metas = [json.loads(l) for l in open(os.path.join(ep, 'meta.jsonl')) if l.strip()]
    if not metas or not (metas[0].get('gt') or {}).get('route_fl'):
        print(f'  {os.path.basename(ep)}: no gt.route_fl (old recording) -- skipped')
        return []
    route = Route(json.load(open(os.path.join(ep, 'route.json')))['nodes'])
    trackers = dict(spline=SplineLane(), legacy=LegacyLane(route), belief=BeliefLane(route))
    nx = ny = 0.0
    rho = 0.98
    s_hint, last_fit_t, rows = None, -1e9, []
    for i, m in enumerate(metas):
        gt = m.get('gt') or {}
        if not gt.get('route_fl'):
            continue
        tx, ty, tyaw = m['x'], m['y'], m['compass'] - math.pi / 2
        nx = rho * nx + math.sqrt(1 - rho * rho) * 0.26 * rng.normal()
        ny = rho * ny + math.sqrt(1 - rho * rho) * 0.26 * rng.normal()
        ex, ey, eyaw = tx + nx, ty + ny, tyaw + math.radians(0.3) * rng.normal()
        s, d, _ = route.fc.world_to_frenet(ex, ey, eyaw, s_hint=s_hint)
        s_hint = s
        refs = {k: t.ref0(s, d) for k, t in trackers.items()}
        fresh = i % 2 == 0                                   # YOLOP every 2nd tick, as live
        X = Y = None
        if fresh:
            p = os.path.join(ep, 'CAM_FRONT', f"{m['step']:06d}.jpg")
            import cv2
            img = cv2.imread(p)
            if img is not None:
                _, ll, r = lanes.masks(img)
                v, u = np.nonzero(ll)
                if len(u) >= 20:
                    X, Y = lanes.cam.pixels_to_ground(u / r, v / r)
        for k, t in trackers.items():
            fit = fit_lane_centre(X, Y, ref0=refs[k]) if X is not None else None
            if k == 'legacy' and fit is not None:
                last_fit_t = m['step'] * 0.05
            t.update(fit, refs[k], ex, ey, eyaw, s)
        row = dict(ep=os.path.basename(ep)[:40], step=m['step'],
                   blind=m['step'] * 0.05 - last_fit_t > 1.0,
                   junction=bool(m['scene'].get('in_junction')),
                   offlane=abs(gt.get('d', 0.0)) > 1.75,
                   lane_change=route.at(route.cmds, s + 10.0) in (5, 6))
        for k, t in trackers.items():
            e = lateral_errors(route, t.offset, s, gt['route_fl'], tx, ty, tyaw)
            for a, v in zip(AHEAD, e):
                row[f'{k}_{int(a)}'] = v
        rows.append(row)
    return rows


def summarise(rows):
    names = ('spline', 'legacy', 'belief')
    conds = [('ALL', lambda r: True), ('YOLOP fresh', lambda r: not r['blind']),
             ('YOLOP blind >1s', lambda r: r['blind']), ('junction', lambda r: r['junction']),
             ('off lane (overtaking)', lambda r: r['offlane']),
             ('route lane change', lambda r: r['lane_change'])]
    print(f"\n{len(rows)} ticks, {len({r['ep'] for r in rows})} episodes.  "
          f"|lane-centre error| (m): median / p90 / share > 1.75 m (half a lane)")
    for a in AHEAD:
        print(f'\n  at {int(a)} m ahead          ' + ''.join(f'{n:>24s}' for n in names))
        for cname, cf in conds:
            sel = [r for r in rows if cf(r)]
            if not sel:
                continue
            cells = []
            for n in names:
                v = np.array([r[f'{n}_{int(a)}'] for r in sel], float)
                v = v[np.isfinite(v)]
                cells.append('-' if not len(v) else
                             f'{np.median(v):5.2f} /{np.percentile(v, 90):5.2f} /{np.mean(v > 1.75) * 100:4.0f}%')
            print(f'  {cname:22s} n={len(sel):6d}' + ''.join(f'{c:>24s}' for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episodes', nargs='+')
    ap.add_argument('--max-episodes', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--yolop', default=os.path.join(ROOT, 'models/yolop/yolop-640-640.onnx'))
    ap.add_argument('--csv', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    eps = [e for e in sorted(set(sum([glob.glob(p) for p in args.episodes], [])))
           if os.path.exists(os.path.join(e, 'meta.jsonl'))]
    if args.max_episodes:
        eps = eps[:args.max_episodes]
    lanes = LanePerception(args.yolop, camera=CameraGeometry(**CAM), device=args.device)
    rng = np.random.default_rng(args.seed)
    rows = []
    for ep in eps:
        r = run_episode(ep, lanes, rng)
        print(f'  {os.path.basename(ep)[:60]}: {len(r)} ticks', flush=True)
        rows += r
    if not rows:
        sys.exit('no usable ticks')
    summarise(rows)
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'\n[csv] {args.csv}')


if __name__ == '__main__':
    main()
