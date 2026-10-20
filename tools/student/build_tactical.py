#!/usr/bin/env python
"""
Bench2Drive recordings + teacher labels -> the tactical student's dataset.

    python tools/student/build_tactical.py \\
        --rec results/b2d/rec_train_clips/rec --labels results/b2d/rec_train_clips/labels \\
        --out data/tactical_orion.npz
    # two teachers combined (ORION + SimLingo):
    python tools/student/build_tactical.py --rec R --labels L_orion --labels2 L_simlingo \\
        --combine soft --out data/tactical_both.npz
    # DAgger: one --labels (and --labels2) root per --rec root, in the same order
    python tools/student/build_tactical.py --rec R0 R1 --labels L0 L1 --out ...

Teachers (one npz per episode, keyed by the episode directory name):
  plan teachers   ORION (tools/orion/orion_infer.py) / SimLingo (tools/simlingo/
                  simlingo_infer.py): 'plan_fl' + 'steps'; labels derived here
  direct teachers privileged / rules experts (tools/student/expert_teacher.py):
                  'go', 'yield_', 'dv' per step

Per recorded frame (meta.jsonl 'scene', written by tools/b2d/mpcc_b2d_agent.py):

  X        features: src/student/tactical.py frame_features, stacked over
           HIST_LAGS (now, 0.5 s, 1 s ago)
  y_go     1 the teacher pulls out to pass / 0 it stays behind; m_go marks
           decision frames (in-lane slow blocker 8-40 m ahead, ego in lane)
  y_yield  1 it stops short of the next junction entry / 0 it enters; m_yield
           marks decision frames (src/student/tactical.py yield_label).  For
           plan teachers, frames where the TRUTH shows a red/yellow light or a
           stop sign at that entry are masked: the teacher stopped for the
           light, which our rule layer handles, not for traffic
  dv       the planned speed change at ~1-1.5 s.  Plan teachers anchor their
           planned speed to ours (the BC inertia found in round 0), so their
           raw dv is debiased by its median per 1 m/s speed bin (per teacher);
           direct teachers' dv is used as is.  y_slow / m_slow (dv < slow_on /
           > slow_off) are kept for the v1 metrics
  ep       episode index (unique per --rec root), for episode-level splits

Combining two teachers (--labels2, --combine):
  soft   disagreement -> 0.5 soft target; dv = mean            (default)
  agree  keep only frames where both give the same label; dv kept if both are
         on the same side of the slow thresholds (mean)
  safe   disagreement resolved conservatively: wait (go 0), yield (1), dv = min
  The agreement table (both heads, Cohen's kappa) is printed and stored.

Rule-checked go labels (default; --no-filter for the raw-teacher ablation):
a teacher's 'go' is DROPPED when a gap check on the SAME frame's features says
the pass is unsafe -- observable quantities only (what the student itself
sees), so the filter adds no expert/learner asymmetry (LEAD):
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
import collections
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from student.tactical import (FEATURE_NAMES, FRAME_NAMES, HIST_LAGS,  # noqa: E402
                              frame_features, go_label, plan_dv, stack_history, yield_label)

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


def rule_stop_at_entry(rec):
    """Truth: a red/yellow light or a stop sign governs the next junction entry."""
    gt, e = rec.get('gt'), rec['scene'].get('entry_ds')
    if not gt or e is None:
        return False
    if gt.get('light') in ('red', 'yellow') and gt.get('light_ds') is not None \
            and gt['light_ds'] < e + 5.0:
        return True
    return gt.get('sign_ds') is not None and gt['sign_ds'] < e + 5.0


def wedged_frames(recs, v_stop=0.3, thr_push=0.3, after=20, v_free=1.0):
    """Frames where the car pushes (throttle > thr_push) yet has not moved
    (v < v_stop) for `after` ticks -- wedged against what it hit -- until it
    moves again (v > v_free).  rec_train_clips2: ~60 % of all standstill ticks
    were this; labels there are noise (the teacher plans for a car that cannot
    move) and they swamp the decision frames."""
    out, run, wedged = [], 0, False
    for r in recs:
        v = float(r['scene']['v'])
        thr = float((r.get('u_final') or [0.0, 0.0])[0])
        run = run + 1 if (v < v_stop and thr > thr_push) else 0
        if run >= after:
            wedged = True
        elif v > v_free:
            wedged = False
        out.append(wedged)
    return out


def load_labels(root):
    out = {}
    for p in glob.glob(os.path.join(os.path.expanduser(root), '*.npz')):
        out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def teacher_frame(lab, j, rec, sc):
    """(go, yield, dv_raw, is_plan) of one teacher on one frame."""
    if 'go' in lab.files:                                   # direct (expert) teacher
        dv = float(lab['dv'][j])
        return int(lab['go'][j]), int(lab['yield_'][j]), dv, False
    plan = lab['plan_fl'][j]
    y = yield_label(sc, plan)
    if y >= 0 and rule_stop_at_entry(rec):
        y = -1                                              # stopped for the light / sign
    # Pull-out is judged against the TRUE lane centre when the recording has
    # it (and the true in-lane check); the perceived one is off by > half a
    # lane on ~1/3 of sensor-only ticks.  Features stay perceived.
    gt = rec.get('gt') or {}
    sc_go = dict(sc, route_fl=gt['route_fl'], d=gt['d']) if gt.get('route_fl') else sc
    return go_label(sc_go, plan), y, plan_dv(sc, plan), True


def debias(dv, v, is_plan):
    """Plan teachers: subtract the median dv per 1 m/s speed bin (inertia)."""
    dv = np.asarray(dv, np.float32)
    out, med = dv.copy(), None
    if is_plan:
        bins = np.clip(np.asarray(v).astype(int), 0, 19)
        ok = np.isfinite(dv)
        allmed = np.median(dv[ok]) if ok.any() else 0.0
        med = np.array([np.median(dv[ok & (bins == b)]) if (ok & (bins == b)).sum() >= 30 else allmed
                        for b in range(20)])
        out = dv - med[bins]
    return out, med


def kappa(a, b):
    """Cohen's kappa of two binary label vectors."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float('nan')
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float('nan')


def combine(a, b, mode, conservative):
    """Two label vectors (-1 = none) -> (target, mask) for one binary head."""
    a, b = np.asarray(a), np.asarray(b)
    y = np.zeros(len(a), np.float32)
    m = np.zeros(len(a), bool)
    both, only_a, only_b = (a >= 0) & (b >= 0), (a >= 0) & (b < 0), (a < 0) & (b >= 0)
    same = both & (a == b)
    y[same], m[same] = a[same], True
    diff = both & (a != b)
    if mode == 'soft':
        y[diff], m[diff] = 0.5, True
    elif mode == 'safe':
        y[diff], m[diff] = conservative, True
    if mode != 'agree':
        y[only_a], m[only_a] = a[only_a], True
        y[only_b], m[only_b] = b[only_b], True
    return y, m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rec', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True, help='one per --rec root (or one for all)')
    ap.add_argument('--labels2', nargs='+', default=None, help='second teacher, same pairing')
    ap.add_argument('--combine', choices=('soft', 'agree', 'safe'), default='soft')
    ap.add_argument('--out', required=True)
    ap.add_argument('--slow-on', type=float, default=-1.5, help='debiased dv below -> slow=1')
    ap.add_argument('--slow-off', type=float, default=-0.5, help='debiased dv above -> slow=0')
    ap.add_argument('--no-filter', action='store_true',
                    help='raw teacher go labels (the ablation); default: rule-checked')
    ap.add_argument('--flip-cautious', action='store_true',
                    help='turn over-cautious waits into go (default: only count them)')
    ap.add_argument('--frac', type=float, default=1.0,
                    help='keep this fraction of EPISODES (data-size ablation E3b)')
    args = ap.parse_args()

    def pair(lst, name):
        if lst is None:
            return None
        if len(lst) == 1:
            return lst * len(args.rec)
        if len(lst) != len(args.rec):
            sys.exit(f'--{name}: give one root, or one per --rec root')
        return lst
    L1, L2 = pair(args.labels, 'labels'), pair(args.labels2, 'labels2')
    two = L2 is not None

    X, V, EP, names = [], [], [], []
    T = {k: [] for k in ('g1', 'y1', 'dv1', 'g2', 'y2', 'dv2')}
    plan1 = plan2 = False
    fstats = collections.Counter()
    for k, root in enumerate(args.rec):
        lab1, lab2 = load_labels(L1[k]), (load_labels(L2[k]) if two else None)
        eps = sorted(glob.glob(os.path.join(os.path.expanduser(root), '*')))
        if args.frac < 1.0:
            keep = np.random.default_rng(0).permutation(len(eps))[:max(1, int(round(args.frac * len(eps))))]
            eps = [eps[i] for i in sorted(keep)]
        for ep in eps:
            meta_p = os.path.join(ep, 'meta.jsonl')
            name = os.path.basename(ep)
            if not os.path.exists(meta_p) or name not in lab1 or (two and name not in lab2):
                continue
            recs = [json.loads(l) for l in open(meta_p) if l.strip()]
            if not recs or 'scene' not in recs[0]:
                print(f'  {name}: no scene in meta (old recorder) -- skipped')
                continue
            if 'entry_ds' not in recs[0]['scene']:
                print(f'  {name}: pre-v2 scene (no entry_ds) -- skipped')
                continue
            la = np.load(lab1[name])
            lb = np.load(lab2[name]) if two else None
            ia = {int(s): j for j, s in enumerate(la['steps'])}
            ib = {int(s): j for j, s in enumerate(lb['steps'])} if two else None
            frames = [frame_features(r['scene']) for r in recs]
            wedge = wedged_frames(recs)
            fstats['wedged_frames_dropped'] += int(sum(wedge))
            n_ep = len(names)
            for i, r in enumerate(recs):
                if wedge[i]:
                    continue
                ja = ia.get(int(r['step']))
                jb = ib.get(int(r['step'])) if two else None
                if ja is None or la['empty'][ja] or (two and (jb is None or lb['empty'][jb])):
                    continue
                sc = r['scene']
                rows = [teacher_frame(la, ja, r, sc)] + ([teacher_frame(lb, jb, r, sc)] if two else [])
                for t, (g, y, dv, is_plan) in enumerate(rows, 1):
                    if g == 1:
                        why = go_unsafe(frames[i])
                        if why:
                            fstats[f't{t}_go_unsafe_{why}'] += 1
                            if not args.no_filter:
                                g = -1                      # dropped: the teacher's mistake
                    elif g == 0 and wait_overcautious(frames[i]):
                        fstats[f't{t}_wait_overcautious'] += 1
                        if args.flip_cautious and not args.no_filter:
                            g = 1
                    T[f'g{t}'].append(g), T[f'y{t}'].append(y), T[f'dv{t}'].append(dv)
                    if t == 1:
                        plan1 = plan1 or is_plan
                    else:
                        plan2 = plan2 or is_plan
                X.append(stack_history(frames, i))
                V.append(sc['v'])
                EP.append(n_ep)
            names.append(f'{k}:{name}')
    if not X:
        sys.exit('no labelled frames found')
    X = np.asarray(X, np.float32)
    V = np.asarray(V, np.float32)
    dv1, med1 = debias(T['dv1'], V, plan1)
    g1, y1 = np.asarray(T['g1']), np.asarray(T['y1'])
    agreement = {}
    if two:
        dv2, med2 = debias(T['dv2'], V, plan2)
        g2, y2 = np.asarray(T['g2']), np.asarray(T['y2'])
        for head, a, b in (('go', g1, g2), ('yield', y1, y2)):
            both = (a >= 0) & (b >= 0)
            agreement[head] = dict(frames_both=int(both.sum()),
                                   agree=float((a[both] == b[both]).mean()) if both.any() else float('nan'),
                                   kappa=kappa(a[both], b[both]),
                                   t1_pos=float(a[both].mean()) if both.any() else float('nan'),
                                   t2_pos=float(b[both].mean()) if both.any() else float('nan'))
        s1 = np.where(dv1 < args.slow_on, 1, np.where(dv1 > args.slow_off, 0, -1))
        s2 = np.where(dv2 < args.slow_on, 1, np.where(dv2 > args.slow_off, 0, -1))
        ok = (s1 >= 0) & (s2 >= 0)
        agreement['dv'] = dict(corr=float(np.corrcoef(dv1, dv2)[0, 1]),
                               slow_agree=float((s1[ok] == s2[ok]).mean()) if ok.any() else float('nan'),
                               slow_kappa=kappa(s1[ok], s2[ok]))
        y_go, m_go = combine(g1, g2, args.combine, conservative=0.0)
        y_yield, m_yield = combine(y1, y2, args.combine, conservative=1.0)
        if args.combine == 'safe':
            dv = np.minimum(dv1, dv2)
        elif args.combine == 'agree':
            dv = np.where(ok & (s1 == s2), 0.5 * (dv1 + dv2), np.nan)
        else:
            dv = 0.5 * (dv1 + dv2)
    else:
        y_go, m_go = np.maximum(g1, 0).astype(np.float32), g1 >= 0
        y_yield, m_yield = np.maximum(y1, 0).astype(np.float32), y1 >= 0
        dv = dv1
    m_dv = np.isfinite(dv)
    dv = np.where(m_dv, dv, 0.0).astype(np.float32)
    y_slow = (dv < args.slow_on).astype(np.float32)
    m_slow = m_dv & ((dv < args.slow_on) | (dv > args.slow_off))
    cfg = dict(hist_lags=HIST_LAGS, slow_on=args.slow_on, slow_off=args.slow_off,
               filtered=not args.no_filter, flip_cautious=args.flip_cautious,
               filter_stats=dict(fstats), frac=args.frac,
               teachers=[L1[0]] + ([L2[0]] if two else []), combine=args.combine if two else None,
               agreement=agreement,
               dv_bias_by_speed=None if med1 is None else med1.round(3).tolist())
    np.savez_compressed(args.out, X=X, y_go=y_go, m_go=m_go, y_yield=y_yield, m_yield=m_yield,
                        dv=dv, m_dv=m_dv, y_slow=y_slow, m_slow=m_slow,
                        ep=np.asarray(EP, np.int32), episodes=np.array(names),
                        feature_names=np.array(FEATURE_NAMES), config=json.dumps(cfg))
    print(f'{len(names)} episodes, {len(X)} frames -> {args.out}')
    for head, y, m in (('go', y_go, m_go), ('yield', y_yield, m_yield)):
        print(f'  {head:5s} decision frames {int(m.sum()):6d} ({m.mean() * 100:4.1f}%): '
              f'positive {float(y[m].sum()):.0f}, soft {int(((y == 0.5) & m).sum())}')
    print(f'  dv: {int(m_dv.sum())} frames, mean {dv[m_dv].mean():+.2f}, slow-confident '
          f'{int(m_slow.sum())} (slow=1 {int(y_slow[m_slow].sum())})')
    if med1 is not None:
        print('  dv bias by speed bin (teacher 1, m/s): ' + ' '.join(f'{x:+.1f}' for x in med1))
    print(f"  rule check ({'applied' if not args.no_filter else 'counted only, --no-filter'}): "
          + (', '.join(f'{k} {v}' for k, v in sorted(fstats.items())) or 'nothing flagged'))
    for head, a in agreement.items():
        print(f'  TEACHER AGREEMENT {head:5s}: ' + ', '.join(
            f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in a.items()))


if __name__ == '__main__':
    main()
