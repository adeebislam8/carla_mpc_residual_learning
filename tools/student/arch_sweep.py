#!/usr/bin/env python
"""
Student architecture ablation, offline: 5-fold cross-validation over TRAINING
episodes (no evaluation routes), one member per fold.  Decides MLP vs GRU for
the headline student before any closed-loop run.

    conda activate orion        # any env with torch
    python tools/student/arch_sweep.py --data data/tactical_simlingo.npz data/tactical_both.npz

Arms: MLP 16x2 / 64x2 (current) / 256x3 over 3 stacked lags, MLP 64x2 with no
history, GRU 64 over the SEQ_LAGS window (1.5 s).  Reports go AUC (hard
decision frames) and dv MAE, mean +- sd over folds, plus GRU - MLP64 per fold.
Datasets need frames_all / seq_idx (build_tactical.py since 2026-10-28).
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'src'))
from student.tactical import N_FRAME  # noqa: E402

DV_SCALE = 3.0
ARMS = {                     # name: (kind, hidden, input)
    'mlp16x2': ('mlp', [16, 16], 'lags'),
    'mlp64x2 (current)': ('mlp', [64, 64], 'lags'),
    'mlp256x3': ('mlp', [256, 256, 256], 'lags'),
    'mlp64x2 no-history': ('mlp', [64, 64], 'now'),
    'gru64 1.5s': ('gru', 64, 'seq'),
}


def auc(y, p):
    o = np.argsort(p)
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=25)
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    for path in args.data:
        d = np.load(path)
        if 'seq_idx' not in d.files:
            sys.exit(f'{path}: no seq_idx -- rebuild with build_tactical.py')
        X, ep = d['X'].astype(np.float32), d['ep']
        Y = np.stack([d['y_go'], d['dv'] / DV_SCALE], 1).astype(np.float32)
        M = np.stack([d['m_go'] * (d['y_go'] != 0.5), d['m_dv']], 1).astype(np.float32)
        Fall, S = d['frames_all'], d['seq_idx']
        eps = np.random.default_rng(0).permutation(np.unique(ep))
        res = {k: [] for k in ARMS}
        print(f'\n== {path}: {len(X)} frames, {len(eps)} episodes, {int((M[:, 0] > 0).sum())} go frames')
        for fi, vf in enumerate(np.array_split(eps, args.folds)):
            va = np.isin(ep, vf)
            tr = ~va
            pos = Y[tr, 0][M[tr, 0] > 0].mean()
            pw = torch.tensor(float(np.clip((1 - pos) / pos, 1, 20)), device=dev)
            for name, (kind, h, inp) in ARMS.items():
                torch.manual_seed(fi)
                if inp == 'seq':
                    mu, sd = X[tr, :N_FRAME].mean(0), np.maximum(X[tr, :N_FRAME].std(0), 1e-3)
                    Xin = np.clip((Fall - mu) / sd, -5, 5).astype(np.float32)[S]
                else:
                    Xi = X if inp == 'lags' else X[:, :N_FRAME]
                    mu, sd = Xi[tr].mean(0), np.maximum(Xi[tr].std(0), 1e-3)
                    Xin = np.clip((Xi - mu) / sd, -5, 5).astype(np.float32)
                if kind == 'gru':
                    class Net(nn.Module):
                        def __init__(s):
                            super().__init__()
                            s.g, s.o = nn.GRU(N_FRAME, h, batch_first=True), nn.Linear(h, 2)

                        def forward(s, x):
                            return s.o(s.g(x)[0][:, -1])
                    model = Net().to(dev)
                else:
                    L, din = [], Xin.shape[1]
                    for k in h:
                        L += [nn.Linear(din, k), nn.ReLU()]
                        din = k
                    model = nn.Sequential(*L, nn.Linear(din, 2)).to(dev)
                Xt, Yt, Mt = (torch.as_tensor(a, device=dev) for a in (Xin, Y, M))

                def loss(i):
                    o = model(Xt[i])
                    l1 = nn.functional.binary_cross_entropy_with_logits(
                        o[:, 0], Yt[i, 0], reduction='none', pos_weight=pw)
                    l2 = nn.functional.huber_loss(o[:, 1], Yt[i, 1], reduction='none')
                    m = Mt[i]
                    return ((l1 * m[:, 0]).sum() / m[:, 0].sum().clamp(min=1)
                            + (l2 * m[:, 1]).sum() / m[:, 1].sum().clamp(min=1))
                opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
                tri = torch.as_tensor(np.where(tr)[0], device=dev)
                vai = torch.as_tensor(np.where(va)[0], device=dev)
                best, state, t0 = 1e9, None, time.time()
                for _ in range(args.epochs):
                    model.train()
                    for b in tri[torch.randperm(len(tri), device=dev)].split(512):
                        opt.zero_grad()
                        loss(b).backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        vl = loss(vai).item()
                    if vl < best:
                        best, state = vl, {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(state)
                with torch.no_grad():
                    O = model(Xt[vai]).cpu().numpy()
                mg, md = M[va, 0] > 0, M[va, 1] > 0
                a = auc(Y[va, 0][mg], O[mg, 0])
                mae = float(np.abs(O[md, 1] - Y[va, 1][md]).mean() * DV_SCALE)
                res[name].append((a, mae))
                print(f'  fold {fi} {name:20s} go AUC {a:.3f}  dv MAE {mae:.3f}  ({time.time() - t0:.0f} s)',
                      flush=True)
        print(f'-- {os.path.basename(path)} ({args.folds}-fold episode CV, mean +- sd)')
        for k, v in res.items():
            v = np.array(v)
            print(f'  {k:20s} go AUC {v[:, 0].mean():.3f} +- {v[:, 0].std():.3f}   '
                  f'dv MAE {v[:, 1].mean():.3f} +- {v[:, 1].std():.3f}')
        diff = np.array(res['gru64 1.5s'])[:, 0] - np.array(res['mlp64x2 (current)'])[:, 0]
        print(f'  GRU - MLP64 go AUC per fold: {np.round(diff, 3).tolist()}  '
              f'mean {diff.mean():+.3f}, GRU better in {int((diff > 0).sum())}/{len(diff)} folds')


if __name__ == '__main__':
    main()
