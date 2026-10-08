#!/usr/bin/env python
"""
Sanity-check episodes written by record_orion_episodes.py before spending
A5000 time labelling them.

    python tools/orion/check_orion_recording.py data/orion_test/Town01_s1_e000
    python tools/orion/check_orion_recording.py data/orion_rec/*  --montage-every 0

Per episode it checks, and prints FAIL lines for:
  * every camera has one 1600x900 JPEG per meta record, no gaps
  * steps strictly increasing, ORION inputs finite (compass, speed, IMU)
  * obs.npy rows == meta records
and reports, for you to eyeball:
  * outcome, frames, disk size
  * navigation-command histogram and every junction segment with its heading
    change -- dpsi < 0 should be a LEFT turn.  Confirm against the montage.
  * montage PNGs of all six cameras (<episode>/montage_<step>.png), laid out
    as a driver would see them: front-left / front / front-right on top,
    back-left / back / back-right below.  Check that the side views look the
    right way and that no debug lines are painted on the road.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
CMD_NAMES = {1: 'LEFT', 2: 'RIGHT', 3: 'STRAIGHT', 4: 'LANEFOLLOW',
             5: 'CHANGELANELEFT', 6: 'CHANGELANERIGHT'}


def _segments(records):
    """Contiguous runs of junction_ahead == True: (first_step, last_step,
    command, dpsi at the first record)."""
    segs, cur = [], None
    for r in records:
        if r.get('junction_ahead'):
            if cur is None:
                cur = [r['step'], r['step'], r['command'], r['dpsi']]
            else:
                cur[1] = r['step']
        elif cur is not None:
            segs.append(cur)
            cur = None
    if cur is not None:
        segs.append(cur)
    return segs


def montage(ep, step, out_path, scale=0.25):
    import cv2
    tiles = []
    for cam in CAMS:
        img = cv2.imread(str(ep / cam / f'{step:06d}.jpg'))
        img = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.putText(img, cam, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        tiles.append(img)
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv2.imwrite(str(out_path), grid)


def check(ep, montage_every, montage_steps):
    import cv2
    fails = []
    meta_path = ep / 'meta.jsonl'
    if not meta_path.exists():
        return [f'no meta.jsonl in {ep}']
    records = [json.loads(l) for l in open(meta_path) if l.strip()]
    if not records:
        return ['meta.jsonl is empty']
    steps = [r['step'] for r in records]
    summary = json.load(open(ep / 'summary.json')) if (ep / 'summary.json').exists() else {}

    if any(b <= a for a, b in zip(steps, steps[1:])):
        fails.append('steps not strictly increasing')
    stride = summary.get('flags', {}).get('stride', 1)
    gaps = [b - a for a, b in zip(steps, steps[1:]) if b - a != stride]
    if gaps:
        fails.append(f'{len(gaps)} step gaps (expected every {stride})')

    for cam in CAMS:
        files = sorted((ep / cam).glob('*.jpg'))
        if len(files) != len(records):
            fails.append(f'{cam}: {len(files)} jpgs for {len(records)} records')
    first = cv2.imread(str(ep / 'CAM_FRONT' / f'{steps[0]:06d}.jpg'))
    if first is None or first.shape[:2] != (900, 1600):
        fails.append(f'CAM_FRONT frame shape {None if first is None else first.shape}, '
                     f'ORION needs (900, 1600, 3)')

    for key in ('compass', 'speed'):
        vals = np.array([r[key] for r in records], dtype=float)
        if not np.isfinite(vals).all():
            fails.append(f'{key}: {np.sum(~np.isfinite(vals))} non-finite values')
    for key in ('accel', 'gyro'):
        vals = np.array([r[key] for r in records], dtype=float)
        if not np.isfinite(vals).all():
            fails.append(f'{key}: non-finite values')

    if (ep / 'obs.npy').exists():
        obs = np.load(ep / 'obs.npy')
        if len(obs) != len(records):
            fails.append(f'obs.npy has {len(obs)} rows for {len(records)} records')
        obs_shape = obs.shape
    else:
        fails.append('no obs.npy')
        obs_shape = None

    size_mb = sum(f.stat().st_size for f in ep.rglob('*') if f.is_file()) / 1e6
    speeds = np.array([r['speed'] for r in records])
    cmds = [r['command'] for r in records]
    hist = {CMD_NAMES.get(c, c): cmds.count(c) for c in sorted(set(cmds))}

    print(f'\n== {ep.name}: outcome={summary.get("outcome")} frames={len(records)} '
          f'steps {steps[0]}..{steps[-1]}  {size_mb:.0f} MB  obs {obs_shape}')
    print(f'   speed median {np.median(speeds):.1f} m/s, max {speeds.max():.1f}')
    print(f'   commands: {hist}')
    segs = _segments(records)
    for a, b, cmd, dpsi in segs:
        print(f'   junction steps {a}-{b}: {CMD_NAMES.get(cmd, cmd):10s} '
              f'dpsi {np.degrees(dpsi):+6.1f} deg   <- montage_{a:06d}.png')

    # Montages: the first frame, the start of every junction segment (where
    # the command sign must be checked), any requested steps, and optionally
    # every N records.
    want = {steps[0]} | {a for a, _, _, _ in segs} | set(montage_steps)
    if montage_every > 0:
        want |= set(steps[::montage_every])
    for s in sorted(want):
        if s in steps:
            montage(ep, s, ep / f'montage_{s:06d}.png')
    print(f'   wrote {len(want & set(steps))} montage(s) into {ep}')
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episodes', nargs='+')
    ap.add_argument('--montage-every', type=int, default=100,
                    help='also write a montage every N records (0 = off)')
    ap.add_argument('--montage-steps', type=int, nargs='*', default=[])
    args = ap.parse_args()

    n_fail = 0
    for ep in args.episodes:
        fails = check(Path(ep), args.montage_every, args.montage_steps)
        for f in fails:
            print(f'   FAIL {f}')
        n_fail += bool(fails)
    print(f'\n{len(args.episodes) - n_fail}/{len(args.episodes)} episodes passed')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
