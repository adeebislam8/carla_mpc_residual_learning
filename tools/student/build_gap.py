#!/usr/bin/env python
"""
Gap-acceptance dataset (2026-10-31): the junction-entry wait / enter decision,
labelled by a direct teacher (the privileged expert, --yield-v2), with inputs
from src/student/tactical.py gap_input (36 frame features + gap features from
the recording's 'actors_wide' when present).  Written in train_tactical.py's
layout (only the yield head is labelled; go / dv masked out):

    python tools/student/build_gap.py --rec results/b2d/recwide_train_clips2/rec \
        --labels results/b2d/recwide_train_clips2/labels_privileged_y2 --out data/gap_v6.npz
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from student.tactical import (GAP_INPUT_NAMES, frame_features, gap_features,  # noqa: E402
                              gap_input, wedged_frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rec', required=True)
    ap.add_argument('--labels', required=True, help='direct-teacher label dir (expert_teacher.py)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    X, Y, EP, names = [], [], [], []
    n_wide = n_frames = 0
    for ep in sorted(glob.glob(os.path.join(args.rec, '*'))):
        name = os.path.basename(ep)
        lf = os.path.join(args.labels, name + '.npz')
        if not os.path.exists(lf) or not os.path.exists(os.path.join(ep, 'meta.jsonl')):
            continue
        lab = np.load(lf)
        idx = {int(s): j for j, s in enumerate(lab['steps'])}
        recs = [json.loads(l) for l in open(os.path.join(ep, 'meta.jsonl')) if l.strip()]
        if not recs or 'scene' not in recs[0]:
            continue
        frames = [frame_features(r['scene']) for r in recs]
        gframes = [gap_features(r['scene']) for r in recs]
        n_wide += sum('actors_wide' in r['scene'] for r in recs)
        n_frames += len(recs)
        wedge = wedged_frames(recs)
        k = len(names)
        for i, r in enumerate(recs):
            j = idx.get(int(r['step']))
            if j is None or wedge[i] or lab['empty'][j]:
                continue
            y = int(lab['yield_'][j])
            if y < 0:
                continue
            X.append(gap_input(frames, gframes, i))
            Y.append(y)
            EP.append(k)
        names.append(name)
    if not X:
        sys.exit('no gap decision frames found')
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    n = len(Y)
    z = np.zeros(n, np.float32)
    f = np.zeros(n, bool)
    np.savez_compressed(args.out, X=X, y_go=z, m_go=f, y_yield=Y, m_yield=np.ones(n, bool),
                        dv=z, m_dv=f, y_slow=z, m_slow=f, ep=np.asarray(EP, np.int32),
                        episodes=np.array(names), feature_names=np.array(GAP_INPUT_NAMES),
                        config=json.dumps(dict(kind='gap', labels=args.labels,
                                               wide_frac=n_wide / max(n_frames, 1))))
    print(f'{len(names)} episodes, {n} gap decision frames (wait {Y.mean():.2f}) -> {args.out}; '
          f'scenes with actors_wide: {100 * n_wide / max(n_frames, 1):.0f} %')


if __name__ == '__main__':
    main()
