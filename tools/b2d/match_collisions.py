#!/usr/bin/env python
"""
For every collision Bench2Drive recorded, show what the agent was doing then.

    python tools/b2d/match_collisions.py results/b2d/rules_hb [--types HardBreakRoute]

eval.json gives each collision's object type/id and world (x, y); the agent's
traces (tools/b2d/mpcc_b2d_agent.py, runs after 2026-10-17) log the ego's x, y
every step.  The step closest to the collision point is printed with the ~1 s
before it: speed, throttle/brake, lateral offset, the followed lead, the
overtake gate, and the nearest actor.
"""

import argparse
import csv
import glob
import json
import math
import os
import re

COLL = re.compile(r"type=(\S+) and id=(\d+) at \(x=([-\d.]+), y=([-\d.]+)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--types', nargs='*', default=None, help='only these scenario types')
    ap.add_argument('--before', type=float, default=1.0, help='seconds of context')
    args = ap.parse_args()

    traces = {}
    for f in glob.glob(os.path.join(args.run_dir, 'traces', '*.csv')):
        m = re.match(r'RouteScenario_(\d+)_rep0_(Town\w+?)_(\w+)\.csv', os.path.basename(f))
        if m:
            traces[m.group(1)] = (m.group(3), f)
    recs = json.load(open(os.path.join(args.run_dir, 'eval.json')))['_checkpoint']['records']
    for r in recs:
        rid = r['route_id'].split('_')[1]
        typ, path = traces.get(rid, ('?', None))
        if args.types and typ not in args.types:
            continue
        colls = [c for k, v in r['infractions'].items() if k.startswith('collisions')
                 for c in v]
        if not colls:
            continue
        print(f'===== {typ} {rid}  ({r["status"]})')
        rows = list(csv.DictReader(open(path))) if path else []
        if rows and 'x' not in rows[0]:
            print('  (trace has no x/y -- re-run with the current agent)')
            rows = []
        for c in colls:
            m = COLL.search(str(c))
            if not m:
                print('  ', c)
                continue
            otype, oid, cx, cy = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
            print(f'  HIT {otype} id={oid} at ({cx:.1f}, {cy:.1f})')
            if not rows:
                continue
            i = min(range(len(rows)), key=lambda k: math.hypot(
                float(rows[k]['x']) - cx, float(rows[k]['y']) - cy))
            n_back = int(args.before / 0.05)
            for x in rows[max(0, i - n_back):i + 3:4]:
                mark = '>>' if x is rows[i] else '  '
                print(f"   {mark} t={x['t']:>6} v={x['v']:>5} thr={x['thr']:>6} "
                      f"st={x['steer']:>6} d={x['d']:>6} lead={x['lead_ds']:>5}/{x['lead_v']:>5} "
                      f"gate={x.get('gate', '')} near={x.get('near_m', '')}m "
                      f"{x.get('near_type', '')}#{x.get('near_id', '')}")


if __name__ == '__main__':
    main()
