#!/usr/bin/env python
"""
One table over results/<prefix>_<case>_<town>.json from probe_scenarios.sh.

    python tools/summarize_scenarios.py                 # prefix 'probe'
    python tools/summarize_scenarios.py --prefix probey # the YOLO probe

Per (case, town): outcome rates, how often the scenario actually happened
(spawned / triggered), whether the ego got past it, whether the collision was
WITH the scenario car, and mean NPCs spawned.  The verdict column applies the
selection rule: a case is worth targeting where the MPCC succeeds 30-80%.
"""

import argparse
import collections
import glob
import json
import os
import re

CASE_NAMES = {'s0': 'normal', 's1': 'hard_brake', 's2': 'stopped',
              's3': 'stopped_oncoming', 's4': 'cut_in', 's5': 'dense'}


def pct(n, d):
    return f'{100.0 * n / d:5.1f}' if d else '   - '


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prefix', default='probe')
    ap.add_argument('--dir', default='results')
    args = ap.parse_args()

    pat = re.compile(rf'^{re.escape(args.prefix)}_(s\d)_(.+)\.json$')
    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, f'{args.prefix}_s*_*.json'))):
        m = pat.match(os.path.basename(path))
        if not m:
            continue
        eps = json.load(open(path)).get('per_episode', [])
        if eps:
            rows.append((m.group(1), m.group(2), eps))
    if not rows:
        raise SystemExit(f'no {args.prefix}_s*_*.json in {args.dir}')

    hdr = (f"{'case':18s} {'town':9s} {'n':>4s} {'succ%':>6s} {'coll%':>6s} {'stall%':>6s} "
           f"{'tout%':>6s} {'spawn%':>6s} {'trig%':>6s} {'pass%':>6s} {'hitScn%':>7s} "
           f"{'npc':>4s}  crash kinds / verdict")
    print(hdr)
    print('-' * len(hdr))
    for case, town, eps in sorted(rows, key=lambda r: (r[0], r[1])):
        n = len(eps)
        out = collections.Counter(e.get('outcome') for e in eps)
        coll = [e for e in eps if e.get('outcome') == 'collision']
        kinds = collections.Counter(
            'vehicle' if str(e.get('collision_other')).startswith('vehicle')
            else str(e.get('collision_other')).split('.')[0] for e in coll)
        scen = [e for e in eps if e.get('scenario') not in (None, 'none')]
        sp = sum(bool(e.get('scenario_spawned')) for e in scen)
        tr = sum(bool(e.get('scenario_triggered')) for e in scen)
        pa = sum(bool(e.get('scenario_passed')) for e in scen)
        hit = sum(bool(e.get('scenario_hit')) for e in scen)
        npcs = [e['npc_spawned'] for e in eps if e.get('npc_spawned') is not None]
        succ = out['success'] / n
        verdict = ('TARGET' if 0.3 <= succ <= 0.8 else
                   'too easy' if succ > 0.8 else 'too hard/broken?')
        print(f"{CASE_NAMES.get(case, case)[:18]:18s} {town:9s} {n:4d} "
              f"{pct(out['success'], n)} {pct(out['collision'], n)} "
              f"{pct(out['stall'], n)} {pct(out['timeout'], n)} "
              f"{pct(sp, len(scen))} {pct(tr, len(scen))} {pct(pa, len(scen))} "
              f"{pct(hit, len(coll)):>7s} "
              f"{(sum(npcs) / len(npcs)) if npcs else float('nan'):4.1f}  "
              f"{dict(kinds)} {verdict}")
        variants = collections.Counter(e.get('scenario_variant') for e in scen)
        if len(variants) > 1 or (variants and '' not in variants):
            by = collections.defaultdict(collections.Counter)
            for e in scen:
                by[e.get('scenario_variant')][e.get('outcome')] += 1
            for v, c in sorted(by.items(), key=lambda kv: str(kv[0])):
                tot = sum(c.values())
                print(f"{'':18s}   variant {str(v):24s} n={tot:3d} "
                      f"succ {pct(c['success'], tot)}%  coll {pct(c['collision'], tot)}%")
    print('\nhitScn% = share of collisions that were WITH the scenario car(s); '
          'spawn/trig/pass% are over scenario episodes.')
    print('Verdict: TARGET = 30-80% success (room for a learned layer, not broken).')


if __name__ == '__main__':
    main()
