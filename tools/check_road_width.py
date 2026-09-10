#!/usr/bin/env python
"""
Is the lateral corridor real road, or invented?

get_road_width_at_waypoint() does:

    left_width = lane_width / 2
    if left_lane is Driving:  left_width += left_lane.lane_width
    else:                     left_width += 4        # <-- phantom

That `else` grants 4 m of drivable space when no adjacent driving lane exists.
If it fires often, the MPC's corridor extends into grass, fences and poles --
which is where 90% of the collisions are landing.  n_min = -4.8 is consistent
with BOTH a real 4 m oncoming lane and 4 m of invented space, so it has to be
checked against the map rather than inferred.

    python tools/check_road_width.py --town Town01
"""
import argparse
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit('/tools/', 1)[0] + '/src')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--town', default='Town01')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--spacing', type=float, default=5.0,
                    help='waypoint sampling distance (m)')
    ap.add_argument('--margin', type=float, default=1.2,
                    help='safety_margin used in mpc_controller_python.solve()')
    args = ap.parse_args()

    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    if world.get_map().name.split('/')[-1] != args.town:
        world = client.load_world(args.town)
    carla_map = world.get_map()

    wps = carla_map.generate_waypoints(args.spacing)
    driving = [w for w in wps if w.lane_type == carla.LaneType.Driving]
    print(f"{args.town}: {len(driving)} driving waypoints at {args.spacing} m\n")

    phantom = 0
    real = 0
    left_kinds = Counter()
    widths = []

    for w in driving:
        lw = w.lane_width
        left = w.get_left_lane()
        if left is not None and left.lane_type == carla.LaneType.Driving:
            left_width = lw / 2.0 + left.lane_width
            real += 1
            left_kinds['Driving (real lane)'] += 1
        else:
            left_width = lw / 2.0 + 4.0
            phantom += 1
            left_kinds[str(left.lane_type) if left is not None else 'None'] += 1
        widths.append((lw, left_width, -left_width + args.margin))

    n = len(driving)
    print("LEFT SIDE")
    print(f"  real adjacent driving lane : {real:6d}  ({100*real/n:5.1f}%)")
    print(f"  PHANTOM +4 m applied       : {phantom:6d}  ({100*phantom/n:5.1f}%)  <-- invented road")
    print("\n  what get_left_lane() actually returned where the phantom fired:")
    for k, v in left_kinds.most_common():
        print(f"      {k:<28}{v:6d}")

    import statistics
    lws = [a for a, _, _ in widths]
    nmins = [c for _, _, c in widths]
    print("\nRESULTING CORRIDOR")
    print(f"  lane_width  mean {statistics.mean(lws):.2f}  min {min(lws):.2f}  max {max(lws):.2f}")
    print(f"  n_min       mean {statistics.mean(nmins):.2f}  min {min(nmins):.2f}  max {max(nmins):.2f}")
    print(f"\n  Your benchmark logged n_min = -4.80 constantly.")
    print(f"  If the phantom share above is high, that -4.8 is mostly scenery.")


if __name__ == '__main__':
    main()
