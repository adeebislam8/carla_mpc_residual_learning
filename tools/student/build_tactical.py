#!/usr/bin/env python
"""
Bench2Drive recordings + ORION labels -> the tactical student's dataset.

    python tools/student/build_tactical.py \\
        --rec results/b2d/rec_train_clips/rec --labels results/b2d/rec_train_clips/labels \\
        --out data/tactical_r0.npz

Per recorded frame (meta.jsonl 'scene', written by tools/b2d/mpcc_b2d_agent.py)
and ORION's plan for that frame (orion_infer.py npz, 'plan_fl' + 'steps'):

  X       features: src/student/tactical.py frame_features, stacked over
          HIST_LAGS (now, 0.5 s, 1 s ago)
  y_go    1 ORION pulls out to pass / 0 it stays behind; m_go marks decision
          frames (in-lane slow blocker 8-40 m ahead, ego in lane, unambiguous)
  y_slow  1 ORION plans to slow down more than it usually does at this speed;
          m_slow marks confident frames.  ORION's planned speed is anchored to
          the ego's (the BC inertia problem found in round 0), so the raw
          dv = v_plan(1-1.5 s) - v is debiased by its median per 1 m/s speed bin
          (fit on all frames) before thresholding.
  ep      episode index, for episode-level train/val splits

Rule-checked labels (default; --no-filter for the raw-teacher ablation):
ORION is a strong but fallible teacher.  A 'go' (pull out now) is DROPPED
from the training mask when a gap check on the SAME frame's features says the
pass is unsafe -- observable quantities only (what the student itself sees),
so the filter adds no expert/learner asymmetry (LEAD):
  * oncoming car in the passing lane meets us before the pass is done:
    onc_tmeet < T_pass + 1.5 s, T_pass = (blk_ds + blk_extent + 15) / max(v, 5)
  * a faster car close behind in the passing lane (pbehind_ds > -15 m and
    pbehind_vs > v + 1)
  * a slower car ahead in the passing lane inside the pass zone
'wait' labels are kept (conservative); waits with a clear passing lane behind
a long-stopped blocker are only COUNTED (teacher over-caution), or flipped to
go with --flip-cautious.  Hydra-MDP-style rule supervision of a learned
policy, applied to the teacher's labels.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from student.tactical import (FEATURE_NAMES, FRAME_NAMES, HIST_LAGS,  # noqa: E402
                              frame_features, go_label, plan_dv, stack_history)

F = {n: i for i, n in enumerate(FRAME_NAMES)}
PASS_MARGIN_M, PASS_MIN_V, MEET_MARGIN_S = 15.0, 5.0, 1.5


def go_unsafe(f):
    """Reason string if passing now is unsafe by the frame's own features
    (frame_features vector), else None."""
    v = f[F['v']]
    t_pass = (f[F['blk_ds']] + f[F['blk_extent']] + PASS_MARGIN_M) / max(v, PASS_MIN_V)
    if f[F['onc_present']] and f[F['onc_tmeet']] < t_pass + MEET_MARGIN_S:
        return 'oncoming'
    if f[F['pbehind_present']] and f[F['pbehind_ds']] > -15.0 and f[F['pbehind_vs']] > v + 1.0:
        return 'from_behind'
    if (f[F['pahead_present']] and f[F['pahead_vs']] < v
            and f[F['pahead_ds']] < f[F['blk_ds']] + f[F['blk_extent']] + 10.0):
        return 'slow_ahead'
    return None


def wait_overcautious(f):
    """Clear passing lane and a blocker stationary >= 10 s."""
    return (not f[F['onc_present']] or f[F['onc_tmeet']] > 15.0) and \
        not f[F['pbehind_present']] and not f[F['pahead_present']] and \
        f[F['blk_stopped']] >= 10.0 and abs(f[F['blk_vs']]) < 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rec', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--slow-on', type=float, default=-1.5, help='debiased dv below -> slow=1')
    ap.add_argument('--slow-off', type=float, default=-0.5, help='debiased dv above -> slow=0')
    ap.add_argument('--no-filter', action='store_true',
                    help='raw ORION go labels (the ablation); default: rule-checked')
    ap.add_argument('--flip-cautious', action='store_true',
                    help='turn over-cautious waits into go (default: only count them)')
    args = ap.parse_args()

    labels = {}
    for root in args.labels:
        for p in glob.glob(os.path.join(os.path.expanduser(root), '*.npz')):
            labels[os.path.splitext(os.path.basename(p))[0]] = p
    X, YG, MG, DV, V, EP, names = [], [], [], [], [], [], []
    import collections
    fstats = collections.Counter()
    for root in args.rec:
        for ep in sorted(glob.glob(os.path.join(os.path.expanduser(root), '*'))):
            meta_p = os.path.join(ep, 'meta.jsonl')
            name = os.path.basename(ep)
            if not os.path.exists(meta_p) or name not in labels:
                continue
            recs = [json.loads(l) for l in open(meta_p) if l.strip()]
            if not recs or 'scene' not in recs[0]:
                print(f'  {name}: no scene in meta (old recorder) -- skipped')
                continue
            lab = np.load(labels[name])
            by_step = {int(s): j for j, s in enumerate(lab['steps'])}
            frames = [frame_features(r['scene']) for r in recs]
            n_ep = len(names)
            for i, r in enumerate(recs):
                j = by_step.get(int(r['step']))
                if j is None or lab['empty'][j]:
                    continue
                sc, plan = r['scene'], lab['plan_fl'][j]
                X.append(stack_history(frames, i))
                g = go_label(sc, plan)
                if g == 1:
                    why = go_unsafe(frames[i])
                    if why:
                        fstats['go_unsafe_' + why] += 1
                        if not args.no_filter:
                            g = -1                      # dropped: the teacher's mistake
                elif g == 0 and wait_overcautious(frames[i]):
                    fstats['wait_overcautious'] += 1
                    if args.flip_cautious and not args.no_filter:
                        g = 1
                YG.append(max(g, 0))
                MG.append(g >= 0)
                DV.append(plan_dv(sc, plan))
                V.append(sc['v'])
                EP.append(n_ep)
            names.append(name)
    if not X:
        sys.exit('no labelled frames found')
    X = np.asarray(X, np.float32)
    DV, V = np.asarray(DV, np.float32), np.asarray(V, np.float32)
    # Debias ORION's planned-speed change by speed bin (inertia).
    bins = np.clip(V.astype(int), 0, 19)
    med = np.array([np.median(DV[bins == b]) if (bins == b).sum() >= 30 else np.median(DV)
                    for b in range(20)])
    dvd = DV - med[bins]
    y_slow = (dvd < args.slow_on).astype(np.float32)
    m_slow = (dvd < args.slow_on) | (dvd > args.slow_off)
    np.savez_compressed(args.out, X=X, y_go=np.asarray(YG, np.float32), m_go=np.asarray(MG),
                        y_slow=y_slow, m_slow=m_slow, dv=DV, dv_debiased=dvd,
                        ep=np.asarray(EP, np.int32), episodes=np.array(names),
                        feature_names=np.array(FEATURE_NAMES),
                        config=json.dumps(dict(hist_lags=HIST_LAGS, slow_on=args.slow_on,
                                               slow_off=args.slow_off,
                                               filtered=not args.no_filter,
                                               flip_cautious=args.flip_cautious,
                                               filter_stats=dict(fstats),
                                               dv_bias_by_speed=med.round(3).tolist())))
    mg = np.asarray(MG)
    yg = np.asarray(YG)
    print(f'{len(names)} episodes, {len(X)} frames -> {args.out}')
    print(f'  go decision frames: {mg.sum()} ({mg.mean() * 100:.1f}%); '
          f'go=1 {int(yg[mg].sum())}, wait=0 {int((1 - yg[mg]).sum())}')
    print(f'  slow: confident {m_slow.sum()} ({m_slow.mean() * 100:.0f}%), '
          f'slow=1 {int(y_slow[m_slow].sum())}')
    print('  dv bias by speed bin (m/s): ' + ' '.join(f'{x:+.1f}' for x in med))
    print(f"  rule check ({'applied' if not args.no_filter else 'counted only, --no-filter'}): "
          + (', '.join(f'{k} {v}' for k, v in sorted(fstats.items())) or 'nothing flagged'))


if __name__ == '__main__':
    main()
