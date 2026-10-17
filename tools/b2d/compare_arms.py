#!/usr/bin/env python
"""
Side-by-side Bench2Drive results for several arms, per scenario type.

    python tools/b2d/compare_arms.py --set full                       # every *_full run found
    python tools/b2d/compare_arms.py --set full --arms base rules tactical
    python tools/b2d/compare_arms.py --set ot_eb --csv results/b2d/compare_ot_eb.csv

Reads results/b2d/<arm>_<set>/eval.json and the routes XML of that set; uses
Bench2Drive's success definition (Completed/Perfect, no infraction except
min-speed; a route with no record = failed, score 0).  Prints:

  1. headline per arm: driving score, success, route completion, collisions/route
  2. per ability (Bench2Drive's grouping)
  3. per scenario type: successes per arm, the best arm marked with '*'
"""

import argparse
import csv
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_subset import ABILITY  # noqa: E402
from summarize_b2d import TRAFFIC_SIGNS, route_num, success  # noqa: E402

ALL_AB = dict(ABILITY, Traffic_Signs=sorted(TRAFFIC_SIGNS))


def routes_file(set_name, b2d):
    if set_name == 'full':
        return os.path.join(b2d, 'leaderboard', 'data', 'bench2drive220.xml')
    return os.path.join(HERE, 'routes', f'{set_name}.xml')


def load_arm(path, types):
    recs = {}
    if os.path.exists(path):
        for r in json.load(open(path))['_checkpoint']['records']:
            recs[route_num(r)] = r
    out = {}
    for rid, t in types.items():
        r = recs.get(rid)
        if r is None:
            out[rid] = dict(type=t, ok=False, ds=0.0, rc=0.0, coll=0, missing=True)
        else:
            coll = sum(len(v) for k, v in r.get('infractions', {}).items()
                       if k.startswith('collisions'))
            out[rid] = dict(type=t, ok=success(r), ds=r['scores']['score_composed'],
                            rc=r['scores']['score_route'], coll=coll, missing=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', default='full')
    ap.add_argument('--arms', nargs='*', default=None, help='default: every <arm>_<set> run found')
    ap.add_argument('--results', default=os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                                       'results', 'b2d'))
    ap.add_argument('--b2d', default=os.path.expanduser(
        os.environ.get('B2D', '~/Documents/nett/Bench2Drive')))
    ap.add_argument('--csv', default=None, help='also write the per-type table here')
    args = ap.parse_args()

    rf = routes_file(args.set, args.b2d)
    types = {}
    for route in ET.parse(rf).getroot().findall('route'):
        sc = route.find('scenarios')
        first = sc.find('scenario') if sc is not None else None
        types[route.get('id')] = first.get('type') if first is not None else 'none'
    arms = args.arms or sorted(os.path.basename(p)[:-len('_' + args.set)]
                               for p in glob.glob(os.path.join(args.results, f'*_{args.set}'))
                               if os.path.exists(os.path.join(p, 'eval.json')))
    if not arms:
        sys.exit(f'no results/b2d/*_{args.set}/eval.json found')
    data = {a: load_arm(os.path.join(args.results, f'{a}_{args.set}', 'eval.json'), types)
            for a in arms}
    n = len(types)
    w = max(10, max(len(a) for a in arms) + 2)

    print(f'\n=== {args.set}: {n} routes ({rf})')
    print(f"{'':28s}" + ''.join(f'{a:>{w}s}' for a in arms))
    rows = [('driving score', lambda d: sum(x['ds'] for x in d.values()) / n, '{:.1f}'),
            ('success %', lambda d: 100 * sum(x['ok'] for x in d.values()) / n, '{:.1f}'),
            ('route completion %', lambda d: sum(x['rc'] for x in d.values()) / n, '{:.1f}'),
            ('collisions / route', lambda d: sum(x['coll'] for x in d.values()) / n, '{:.2f}'),
            ('routes missing', lambda d: sum(x['missing'] for x in d.values()), '{:d}')]
    for name, fn, fmt in rows:
        print(f'{name:28s}' + ''.join(f'{fmt.format(fn(data[a])):>{w}s}' for a in arms))

    print('\nper ability (success %):')
    for ab, ab_types in ALL_AB.items():
        vals = []
        for a in arms:
            sel = [x for x in data[a].values() if x['type'] in ab_types]
            vals.append(100 * sum(x['ok'] for x in sel) / len(sel) if sel else float('nan'))
        if all(v != v for v in vals):
            continue
        cnt = sum(1 for x in data[arms[0]].values() if x['type'] in ab_types)
        print(f'  {ab + f" ({cnt})":26s}' + ''.join(f'{v:>{w}.1f}' for v in vals))

    print("\nper scenario type (successes; '*' = best arm, ties all marked):")
    print(f"{'':36s}" + ''.join(f'{a:>{w}s}' for a in arms))
    table = []
    for t in sorted(set(types.values())):
        ks = [sum(x['ok'] for x in data[a].values() if x['type'] == t) for a in arms]
        m = sum(1 for x in data[arms[0]].values() if x['type'] == t)
        best = max(ks)
        cells = [f"{k}/{m}{'*' if k == best and best > 0 and len(set(ks)) > 1 else ' '}" for k in ks]
        print(f'  {t:34s}' + ''.join(f'{c:>{w}s}' for c in cells))
        table.append([t, m] + ks)
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            cw = csv.writer(f)
            cw.writerow(['scenario_type', 'routes'] + arms)
            cw.writerows(table)
        print(f'\n[csv] {args.csv}')


if __name__ == '__main__':
    main()
