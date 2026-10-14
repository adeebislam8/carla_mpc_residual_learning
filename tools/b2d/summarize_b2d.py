#!/usr/bin/env python
"""
Score a Bench2Drive run (any route subset) with Bench2Drive's own definitions.

    python tools/b2d/summarize_b2d.py --routes tools/b2d/routes/ot_eb.xml \
        results/b2d/mpcc_ot_eb/eval.json [more.json ...]

  driving score  mean score_composed over the routes in the file(s)
  success        status Completed/Perfect AND no infraction except min_speed
                 (tools/merge_route_json.py, tools/ability_benchmark.py)
  per ability    success rate over that ability's scenario types

Unlike merge_route_json.py this does not divide by 220, so a subset is scored
over its own routes -- say which subset in the paper.  The scenario type of a
route comes from the routes XML (first <scenario type=...>), as in
ability_benchmark.py; a route in the XML with no record (CARLA crashed and it
never finished) counts as a failure with score 0, as Bench2Drive does.
Traffic_Signs is approximate: Bench2Drive's own version also re-checks junction
behaviour on failed routes with the HD map.
"""

import argparse
import collections
import json
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from make_subset import ABILITY  # noqa: E402

TRAFFIC_SIGNS = {'BlockedIntersection', 'OppositeVehicleTakingPriority',
                 'OppositeVehicleRunningRedLight', 'PedestrianCrossing', 'VehicleTurningRoute',
                 'VehicleTurningRoutePedestrian', 'EnterActorFlow', 'CrossingBicycleFlow',
                 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn',
                 'NonSignalizedJunctionLeftTurnEnterFlow', 'SignalizedJunctionLeftTurn',
                 'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow',
                 'T_Junction', 'VanillaNonSignalizedTurn',
                 'VanillaSignalizedTurnEncounterGreenLight',
                 'VanillaSignalizedTurnEncounterRedLight',
                 'VanillaNonSignalizedTurnEncounterStopsign'}
ALL = dict(ABILITY, Traffic_Signs=sorted(TRAFFIC_SIGNS))


def route_num(rec):
    """'RouteScenario_1711_rep0' -> '1711'."""
    parts = rec.get('route_id', '').split('_')
    return parts[1] if len(parts) > 1 else rec.get('route_id', '')


def success(rec):
    if rec.get('status') not in ('Completed', 'Perfect'):
        return False
    return not any(v for k, v in rec.get('infractions', {}).items()
                   if k != 'min_speed_infractions')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--routes', required=True, help='the routes XML that was run')
    ap.add_argument('evals', nargs='+')
    args = ap.parse_args()

    types = {}
    for route in ET.parse(args.routes).getroot().findall('route'):
        sc = route.find('scenarios')
        first = sc.find('scenario') if sc is not None else None
        types[route.get('id')] = first.get('type') if first is not None else 'none'
    by_id = {}
    for p in args.evals:
        for r in json.load(open(p))['_checkpoint']['records']:
            by_id[route_num(r)] = r
    missing = [rid for rid in types if rid not in by_id]
    recs = [by_id[rid] for rid in types if rid in by_id]
    for rid in missing:     # never finished: Bench2Drive scores these as 0 / failed
        recs.append({'route_id': f'RouteScenario_{rid}_rep0', 'status': 'Missing',
                     'scores': {'score_composed': 0.0, 'score_route': 0.0},
                     'infractions': {}})
    if not recs:
        raise SystemExit('no records')
    if missing:
        print(f'WARNING: {len(missing)} of {len(types)} routes have no record '
              f'(scored 0) -- re-run the same command to resume them')
    n = len(recs)
    ds = sum(r['scores']['score_composed'] for r in recs) / n
    rc = sum(r['scores']['score_route'] for r in recs) / n
    sr = sum(success(r) for r in recs) / n
    print(f'{n} routes   driving score {ds:6.2f}   route completion {rc:6.2f}   '
          f'success {100 * sr:5.1f}%')

    inf = collections.Counter()
    for r in recs:
        for k, v in r.get('infractions', {}).items():
            inf[k] += len(v)
    print('infractions:', ', '.join(f'{k} {v}' for k, v in inf.most_common() if v))
    status = collections.Counter(r.get('status') for r in recs)
    print('status     :', dict(status))

    by_ab = collections.defaultdict(lambda: [0, 0])
    by_type = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        t, ok = types.get(route_num(r), '?'), success(r)
        by_type[t][0] += ok
        by_type[t][1] += 1
        for ab, ab_types in ALL.items():
            if t in ab_types:
                by_ab[ab][0] += ok
                by_ab[ab][1] += 1
    print('\nper ability (success %):')
    for ab, (k, m) in by_ab.items():
        print(f'  {ab:16s} {100 * k / m:5.1f}%  ({k}/{m})')
    print('\nper scenario type:')
    for t, (k, m) in sorted(by_type.items()):
        print(f'  {t:40s} {k}/{m}')


if __name__ == '__main__':
    main()
