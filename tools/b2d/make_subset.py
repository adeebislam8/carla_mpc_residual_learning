#!/usr/bin/env python
"""
Write a Bench2Drive route subset (XML) selected by ability or scenario type.

    python tools/b2d/make_subset.py --b2d ~/Documents/nett/Bench2Drive \
        --ability Overtaking Emergency_Brake --out tools/b2d/routes_ot_eb.xml
    python tools/b2d/make_subset.py --b2d ... --ability Overtaking --per-type 1 \
        --out tools/b2d/routes_smoke.xml          # 9 routes, one per scenario type

Ability -> scenario-type mapping is Bench2Drive's own (tools/ability_benchmark.py),
copied here so the subset and their per-ability metric agree.  The official
full set is leaderboard/data/bench2drive220.xml (44 types x 5 routes).
"""

import argparse
import collections
import os
import xml.etree.ElementTree as ET

ABILITY = {
    "Overtaking": ['Accident', 'AccidentTwoWays', 'ConstructionObstacle',
                   'ConstructionObstacleTwoWays', 'HazardAtSideLaneTwoWays',
                   'HazardAtSideLane', 'ParkedObstacleTwoWays', 'ParkedObstacle',
                   'VehicleOpensDoorTwoWays'],
    "Merging": ['CrossingBicycleFlow', 'EnterActorFlow', 'HighwayExit', 'InterurbanActorFlow',
                'HighwayCutIn', 'InterurbanAdvancedActorFlow', 'MergerIntoSlowTrafficV2',
                'MergerIntoSlowTraffic', 'NonSignalizedJunctionLeftTurn',
                'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow',
                'ParkingExit', 'SequentialLaneChange', 'SignalizedJunctionLeftTurn',
                'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow'],
    "Emergency_Brake": ['BlockedIntersection', 'DynamicObjectCrossing', 'HardBreakRoute',
                        'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight',
                        'ParkingCutIn', 'PedestrianCrossing', 'ParkingCrossingPedestrian',
                        'StaticCutIn', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian',
                        'ControlLoss'],
    "Give_Way": ['InvadingTurn', 'YieldToEmergencyVehicle'],
}


def route_types(route):
    return [sc.get('type') for sc in route.iter('scenario')]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--b2d', required=True, help='Bench2Drive repo root')
    ap.add_argument('--routes', default='leaderboard/data/bench2drive220.xml')
    ap.add_argument('--ability', nargs='*', default=[], choices=sorted(ABILITY))
    ap.add_argument('--types', nargs='*', default=[], help='explicit scenario types')
    ap.add_argument('--all-types', action='store_true',
                    help='every scenario type in the routes file (abl44: --all-types --per-type 1)')
    ap.add_argument('--per-type', type=int, default=0,
                    help='keep at most N routes per scenario type (0 = all)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    wanted = set(args.types)
    for a in args.ability:
        wanted |= set(ABILITY[a])
    tree = ET.parse(os.path.join(os.path.expanduser(args.b2d), args.routes))
    root = tree.getroot()
    if args.all_types:
        wanted |= {t for r in root for t in route_types(r)}
    if not wanted:
        raise SystemExit('give --ability, --types or --all-types')
    kept, per = [], collections.Counter()
    for route in list(root):
        types = [t for t in route_types(route) if t in wanted]
        if not types:
            root.remove(route)
            continue
        t = types[0]
        if args.per_type and per[t] >= args.per_type:
            root.remove(route)
            continue
        per[t] += 1
        kept.append((route.get('id'), route.get('town'), t))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tree.write(args.out)
    print(f'{len(kept)} routes -> {args.out}')
    for t, n in sorted(per.items()):
        print(f'  {t:36s} {n}')
    missing = wanted - set(per)
    if missing:
        print(f'  (no routes found for: {sorted(missing)})')


if __name__ == '__main__':
    main()
