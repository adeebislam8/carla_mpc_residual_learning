#!/usr/bin/env python
"""
Add obs v3 static-object features to recorded episodes, without CARLA.

    python tools/student/augment_static.py --rec ~/Documents/nett/orion_data/rec_town01

For every episode: reads meta.jsonl (x, y, compass per recorded frame, logged
at the same tick as the observation) and obs.npy (v2), writes obs_v3.npy =
[v2 | StaticMap.features], exactly what the env's obs_version='v3' returns at
run time.  ORION's labels are untouched, so nothing is re-recorded or
re-labelled.  Then build the dataset with --obs-file obs_v3.npy.

Needs data/static_maps/<Town>.npz (tools/student/dump_static_map.py).

The heading comes from the IMU compass.  Its convention is checked against
the direction of motion on every run, and the script refuses to write if the
two disagree -- a wrong sign would mirror every static object.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from mpc_controller.src.static_map import (StaticMap, yaw_from_compass,  # noqa: E402
                                           N_SLOTS, PER_SLOT)

OBS_V2_DIM = 68


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def town_of(ep_dir):
    try:
        flags = json.load(open(ep_dir / 'summary.json')).get('flags', {})
        if flags.get('town'):
            return flags['town']
    except (OSError, ValueError):
        pass
    return ep_dir.name.split('_')[0]


def heading_check(episodes):
    """Median |compass yaw - motion heading|, motion over 5 frames that moved
    > 1.5 m (straight-ish driving; compass taken at the middle frame)."""
    errs, k = [], 5
    for recs in episodes:
        if len(recs) <= k:
            continue
        x = np.array([r['x'] for r in recs])
        y = np.array([r['y'] for r in recs])
        c = np.array([r.get('compass', np.nan) for r in recs], float)
        dx, dy = x[k:] - x[:-k], y[k:] - y[:-k]
        moved = np.hypot(dx, dy) > 1.5
        motion = np.arctan2(dy, dx)
        yaw = yaw_from_compass(c[k // 2:k // 2 + len(dx)])
        ok = moved & np.isfinite(yaw)
        errs.append(wrap(yaw[ok] - motion[ok]))
    e = np.abs(np.concatenate(errs)) if errs else np.array([])
    return (np.degrees(np.median(e)) if len(e) else float('nan')), len(e)


def fill_compass(recs):
    """Compass can be NaN on the first IMU tick; carry the nearest valid one."""
    c = np.array([r.get('compass', np.nan) for r in recs], float)
    if np.isnan(c).all():
        raise ValueError('no valid compass in episode')
    idx = np.where(np.isfinite(c), np.arange(len(c)), 0)
    np.maximum.accumulate(idx, out=idx)
    c = c[idx]
    first = np.argmax(np.isfinite(c))
    c[:first] = c[first]
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rec', nargs='+', required=True, help='recording root dir(s)')
    ap.add_argument('--out-name', default='obs_v3.npy')
    ap.add_argument('--max-heading-err-deg', type=float, default=10.0)
    ap.add_argument('--force', action='store_true', help='overwrite existing files')
    args = ap.parse_args()

    eps = [ep for root in args.rec for ep in sorted(Path(root).expanduser().iterdir())
           if (ep / 'meta.jsonl').exists() and (ep / 'obs.npy').exists()]
    if not eps:
        sys.exit('no recorded episodes found')
    loaded = [[json.loads(l) for l in open(ep / 'meta.jsonl') if l.strip()] for ep in eps]

    err, n = heading_check(loaded)
    print(f'compass vs motion heading: median error {err:.1f} deg over {n} frames')
    if not err <= args.max_heading_err_deg:
        sys.exit('heading convention check FAILED -- not writing anything '
                 '(static objects would be rotated/mirrored)')

    maps, stats = {}, dict(frames=0, filled=0, inside=0)
    nearest_by_outcome = {'collision': [], 'other': []}
    written = skipped = 0
    for ep, recs in zip(eps, loaded):
        out = ep / args.out_name
        if out.exists() and not args.force:
            skipped += 1
            continue
        obs = np.load(ep / 'obs.npy')
        if obs.shape != (len(recs), OBS_V2_DIM):
            print(f'  {ep.name}: obs {obs.shape} vs {len(recs)} records -- skipped')
            continue
        town = town_of(ep)
        if town not in maps:
            maps[town] = StaticMap.for_town(town)
        yaw = yaw_from_compass(fill_compass(recs))
        feats = np.stack([maps[town].features(r['x'], r['y'], yaw[i])
                          for i, r in enumerate(recs)]).astype(np.float32)
        np.save(out, np.concatenate([obs.astype(np.float32), feats], axis=1))
        written += 1

        slots = feats.reshape(len(feats), N_SLOTS, PER_SLOT)
        present = slots[..., 0] > 0.5
        d = np.where(present, np.hypot(slots[..., 1], slots[..., 2]), np.inf).min(axis=1)
        stats['frames'] += len(feats)
        stats['filled'] += int(present.sum())
        stats['inside'] += int((d < 0.3).sum())
        outcome = json.load(open(ep / 'summary.json')).get('outcome')
        tail = d[-40:] if outcome == 'collision' else d
        nearest_by_outcome['collision' if outcome == 'collision' else 'other'].append(
            tail[np.isfinite(tail)])

    print(f'{written} episodes written, {skipped} already had {args.out_name}')
    if stats['frames']:
        print(f"  slots filled per frame: {stats['filled'] / stats['frames']:.2f} of {N_SLOTS}")
        print(f"  frames with an object < 0.3 m (inside a footprint): "
              f"{stats['inside'] / stats['frames'] * 100:.1f}%  (should be ~0)")
        for k, v in nearest_by_outcome.items():
            if v and len(np.concatenate(v)):
                v = np.concatenate(v)
                label = 'last 40 frames of collision eps' if k == 'collision' else 'other episodes'
                print(f'  nearest static object, {label}: median {np.median(v):.2f} m')


if __name__ == '__main__':
    main()
