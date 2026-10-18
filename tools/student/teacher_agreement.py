#!/usr/bin/env python
"""
Teacher agreement check: how often do ORION, SimLingo, the privileged expert
and the rule expert make the SAME tactical decision on the same frames?

    python tools/student/teacher_agreement.py --rec results/b2d/rec_train_clips/rec \\
        --teacher orion=results/b2d/rec_train_clips/labels \\
        --teacher simlingo=results/b2d/rec_train_clips/labels_simlingo \\
        --teacher privileged=results/b2d/rec_train_clips/labels_privileged \\
        --teacher rules=results/b2d/rec_train_clips/labels_rules [--csv out.csv]

Per pair of teachers and per head (go: pass vs wait; yield: wait vs enter at a
junction entry; slow: debiased dv < -1.5 vs > -0.5): frames where both decide,
raw agreement and Cohen's kappa; and per scenario type for go and yield.
Labels are derived exactly as build_tactical.py derives them (rule check off:
the raw teachers).  For the paper: a VLA teacher that agrees with the
privileged expert more than the rule expert does is carrying knowledge our
own stack lacks.
"""

import argparse
import collections
import csv
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_tactical import debias, kappa, load_labels, teacher_frame  # noqa: E402


def scenario_type(name):
    m = re.match(r'RouteScenario_\d+_rep\d+_Town\d+_([A-Za-z]+)', name)
    return m.group(1) if m else 'unknown'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rec', nargs='+', required=True)
    ap.add_argument('--teacher', action='append', required=True, metavar='NAME=DIR')
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()
    teachers = dict(t.split('=', 1) for t in args.teacher)
    labs = {n: load_labels(d) for n, d in teachers.items()}
    names = list(teachers)
    G = {n: [] for n in names}
    Y = {n: [] for n in names}
    DV = {n: [] for n in names}
    plan = {n: False for n in names}
    V, TYPE = [], []
    for root in args.rec:
        for ep in sorted(glob.glob(os.path.join(os.path.expanduser(root), '*'))):
            meta_p = os.path.join(ep, 'meta.jsonl')
            name = os.path.basename(ep)
            if not os.path.exists(meta_p) or any(name not in labs[n] for n in names):
                continue
            recs = [json.loads(l) for l in open(meta_p) if l.strip()]
            if not recs or 'entry_ds' not in recs[0].get('scene', {}):
                continue
            L = {n: np.load(labs[n][name]) for n in names}
            idx = {n: {int(s): j for j, s in enumerate(L[n]['steps'])} for n in names}
            for r in recs:
                js = {n: idx[n].get(int(r['step'])) for n in names}
                if any(js[n] is None or L[n]['empty'][js[n]] for n in names):
                    continue
                for n in names:
                    g, y, dv, is_plan = teacher_frame(L[n], js[n], r, r['scene'])
                    G[n].append(g), Y[n].append(y), DV[n].append(dv)
                    plan[n] = plan[n] or is_plan
                V.append(r['scene']['v'])
                TYPE.append(scenario_type(name))
    if not V:
        sys.exit('no frame labelled by every teacher')
    V, TYPE = np.asarray(V), np.asarray(TYPE)
    S = {}
    for n in names:
        G[n], Y[n] = np.asarray(G[n]), np.asarray(Y[n])
        d, _ = debias(DV[n], V, plan[n])
        S[n] = np.where(d < -1.5, 1, np.where(d > -0.5, 0, -1))
    rows = []
    print(f'{len(V)} frames labelled by all of: {", ".join(names)}\n')
    print(f"{'pair':28s} {'head':6s} {'frames':>7s} {'agree':>6s} {'kappa':>6s}  pos rate a / b")
    for a, b in itertools.combinations(names, 2):
        for head, H in (('go', G), ('yield', Y), ('slow', S)):
            both = (H[a] >= 0) & (H[b] >= 0)
            if not both.any():
                continue
            ag = float((H[a][both] == H[b][both]).mean())
            k = kappa(H[a][both], H[b][both])
            print(f'{a + " vs " + b:28s} {head:6s} {int(both.sum()):7d} {ag:6.2f} {k:6.2f}  '
                  f'{H[a][both].mean():.2f} / {H[b][both].mean():.2f}')
            rows.append(dict(pair=f'{a}-{b}', head=head, type='ALL', frames=int(both.sum()),
                             agree=round(ag, 3), kappa=round(k, 3)))
            if head == 'slow':
                continue
            per = collections.defaultdict(list)
            for t in np.unique(TYPE[both]):
                sel = both & (TYPE == t)
                per[t] = (int(sel.sum()), float((H[a][sel] == H[b][sel]).mean()))
                rows.append(dict(pair=f'{a}-{b}', head=head, type=t, frames=per[t][0],
                                 agree=round(per[t][1], 3), kappa=round(kappa(H[a][sel], H[b][sel]), 3)))
            worst = sorted(per.items(), key=lambda kv: kv[1][1])[:3]
            print(' ' * 36 + 'least agreement: ' +
                  ', '.join(f'{t} {ag:.2f} (n={n})' for t, (n, ag) in worst))
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'\n[csv] {args.csv}')


if __name__ == '__main__':
    main()
