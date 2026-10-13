#!/usr/bin/env python
"""
Dump a town's static-object table (obs v3) to data/static_maps/<Town>.npz.

    python tools/student/dump_static_map.py --town Town01 Town02 Town03

Needs a running CARLA server (residual_mpc env).  Loads each town only if it
is not already loaded.  The env builds the same table on first use, so this
is only needed before augment_static.py, which runs without a server.
"""

import argparse
import collections
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from mpc_controller.src.static_map import StaticMap, cache_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--town', nargs='+', required=True)
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--force', action='store_true', help='rebuild an existing cache')
    args = ap.parse_args()

    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    for town in args.town:
        path = cache_path(town)
        if os.path.exists(path) and not args.force:
            print(f'{town}: {path} exists (use --force to rebuild)')
            continue
        world = client.get_world()
        if world.get_map().name.split('/')[-1] != town:
            world = client.load_world(town)
        sm = StaticMap.from_world(world, world.get_map())
        sm.save(path)
        counts = collections.Counter(sm.label.tolist())
        print(f'{town}: {len(sm.cx)} objects -> {path}')
        dropped = getattr(sm, 'dropped', {})
        print(f'   dropped {sum(dropped.values())} boxes covering a lane centre: '
              + (', '.join(f'{k} {v}' for k, v in dropped.most_common()) or 'none'))
        print('   ' + ', '.join(f'{k} {v}' for k, v in counts.most_common()))
        print(f'   footprint half-size median {np.median(np.maximum(sm.ex, sm.ey)):.2f} m, '
              f'max {np.max(np.maximum(sm.ex, sm.ey)):.1f} m')


if __name__ == '__main__':
    main()
