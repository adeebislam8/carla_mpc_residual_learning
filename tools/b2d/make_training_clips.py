#!/usr/bin/env python
"""
Cut short single-scenario training clips out of Bench2Drive's long TRAINING
routes -- the same shape as its 220 evaluation clips, but disjoint from them.

    conda activate b2d
    python tools/b2d/make_training_clips.py --b2d ~/Documents/nett/Bench2Drive \\
        --ability Overtaking Emergency_Brake --per-type 8 \\
        --out tools/b2d/routes/train_clips.xml

leaderboard/data/routes_training.xml holds 90 long CARLA-Leaderboard routes
(all Town12) with hundreds of scenarios on them and none of the evaluation
routes.  For each scenario of a wanted type this traces the route through its
keypoints on the OFFLINE map (CARLA OpenDRIVE + GlobalRoutePlanner, no server,
as Bench2Drive's ability_benchmark.py does), finds the point nearest the
scenario's trigger, and keeps --before m before to --after m after it, written
as 2 m waypoints like the evaluation XML, with only that one scenario.

Clips are spread across source routes (round robin) so one long route does
not dominate a type.  Route ids are 900000 + n, so they can never collide with
evaluation ids.
"""

import argparse
import collections
import copy
import math
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_subset import ABILITY  # noqa: E402

ID_BASE = 900000


def load_offline_map(town, carla_root):
    import carla
    root = os.path.join(carla_root, 'CarlaUE4', 'Content', 'Carla', 'Maps')
    for p in (os.path.join(root, 'OpenDrive', f'{town}.xodr'),
              os.path.join(root, town, 'OpenDrive', f'{town}.xodr')):
        if os.path.exists(p):
            return carla.Map(town, open(p).read())
    raise FileNotFoundError(f'no OpenDRIVE file for {town} under {root}')


def dense_route(grp, keypoints):
    """[(x, y, z)] at ~1 m along the route through the keypoints."""
    import carla
    out = []
    for a, b in zip(keypoints[:-1], keypoints[1:]):
        for wp, _ in grp.trace_route(carla.Location(*a), carla.Location(*b)):
            loc = wp.transform.location
            p = (loc.x, loc.y, loc.z)
            if not out or math.dist(p[:2], out[-1][:2]) > 0.5:
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--b2d', required=True)
    ap.add_argument('--routes', default='leaderboard/data/routes_training.xml')
    ap.add_argument('--carla-root', default=os.environ.get('CARLA_ROOT', ''))
    ap.add_argument('--ability', nargs='*', default=['Overtaking', 'Emergency_Brake'])
    ap.add_argument('--types', nargs='*', default=[])
    ap.add_argument('--per-type', type=int, default=8)
    ap.add_argument('--extra-per-type', nargs='*', default=[],
                    help='TYPE=N overrides, e.g. AccidentTwoWays=20')
    ap.add_argument('--before', type=float, default=50.0)
    ap.add_argument('--after', type=float, default=150.0)
    ap.add_argument('--spacing', type=float, default=2.0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if not args.carla_root:
        sys.exit('set CARLA_ROOT (CARLA 0.9.15) or pass --carla-root')

    from agents.navigation.global_route_planner import GlobalRoutePlanner

    wanted = set(args.types)
    for a in args.ability:
        wanted |= set(ABILITY[a])
    quota = {t: args.per_type for t in wanted}
    for kv in args.extra_per_type:
        t, n = kv.split('=')
        quota[t] = int(n)
        wanted.add(t)                   # a type named here is wanted even outside --ability

    src = ET.parse(os.path.join(os.path.expanduser(args.b2d), args.routes)).getroot()
    # (type) -> list of (route element, scenario element), round-robin over routes
    by_type = collections.defaultdict(lambda: collections.defaultdict(list))
    for route in src.findall('route'):
        sc = route.find('scenarios')
        for s in (sc.findall('scenario') if sc is not None else []):
            if s.get('type') in wanted:
                by_type[s.get('type')][route.get('id')].append((route, s))

    picks = []
    for t, per_route in by_type.items():
        queues = [list(v) for v in per_route.values()]
        chosen = []
        while len(chosen) < quota[t] and any(queues):
            for q in queues:
                if q and len(chosen) < quota[t]:
                    chosen.append(q.pop(0))
        picks += chosen

    maps, grps, dense_cache = {}, {}, {}
    out_root = ET.Element('routes')
    counts, n = collections.Counter(), 0
    for route, scen in picks:
        town = route.get('town')
        if town not in maps:
            print(f'loading offline map {town} ...', flush=True)
            maps[town] = load_offline_map(town, args.carla_root)
            grps[town] = GlobalRoutePlanner(maps[town], 1.0)
        rid = route.get('id')
        if rid not in dense_cache:
            kps = [(float(p.get('x')), float(p.get('y')), float(p.get('z')))
                   for p in route.find('waypoints').findall('position')]
            dense_cache[rid] = dense_route(grps[town], kps)
        pts = dense_cache[rid]
        tp = scen.find('trigger_point')
        trig = (float(tp.get('x')), float(tp.get('y')))
        i0 = min(range(len(pts)), key=lambda i: math.dist(pts[i][:2], trig))
        if math.dist(pts[i0][:2], trig) > 10.0:
            print(f'  skip {scen.get("name")} (route {rid}): trigger {math.dist(pts[i0][:2], trig):.0f} m off route')
            continue
        # cumulative distance along the dense route
        cum = [0.0]
        for a, b in zip(pts[:-1], pts[1:]):
            cum.append(cum[-1] + math.dist(a[:2], b[:2]))
        lo = max(0.0, cum[i0] - args.before)
        hi = min(cum[-1], cum[i0] + args.after)
        clip, next_d = [], lo
        for p, c in zip(pts, cum):
            if lo <= c <= hi and c >= next_d:
                clip.append(p)
                next_d = c + args.spacing
        if len(clip) < 20:
            continue
        r = ET.SubElement(out_root, 'route', id=str(ID_BASE + n), town=town)
        wps = ET.SubElement(r, 'waypoints')
        for x, y, z in clip:
            ET.SubElement(wps, 'position', x=f'{x:.1f}', y=f'{y:.1f}', z=f'{z:.1f}')
        scs = ET.SubElement(r, 'scenarios')
        scs.append(copy.deepcopy(scen))
        w = route.find('weathers')
        if w is not None:
            r.append(copy.deepcopy(w))
        counts[scen.get('type')] += 1
        n += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ET.ElementTree(out_root).write(args.out)
    print(f'{n} clips -> {args.out}')
    for t in sorted(counts):
        print(f'  {t:36s} {counts[t]}')
    missing = wanted - set(counts)
    if missing:
        print(f'  (none available for: {sorted(missing)})')


if __name__ == '__main__':
    main()
