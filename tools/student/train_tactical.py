#!/usr/bin/env python
"""
Train the tactical student (go / slow heads) on build_tactical.py datasets.

    conda activate orion        # any env with torch
    python tools/student/train_tactical.py --data data/tactical_r0.npz --out models/tactical_r0

An ensemble of small MLPs (default 5 x [64, 64], ~7k parameters each), each on
a bootstrap of training EPISODES, two sigmoid outputs trained with masked,
class-balanced BCE.  Validation is a fixed set of held-out episodes.  Exports
numpy weights for src/student/tactical.py (no torch at run time):

    <out>/tactical.npz, tactical_meta.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def auc(y, p):
    """ROC AUC without sklearn (Mann-Whitney)."""
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--members', type=int, default=5)
    ap.add_argument('--hidden', type=int, nargs='+', default=[64, 64])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cpu', action='store_true', help='train on CPU (the cost study)')
    args = ap.parse_args()

    import time
    import torch
    import torch.nn as nn
    t_start, cpu_start = time.perf_counter(), time.process_time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    parts, off = [], 0
    for p in args.data:
        d = np.load(Path(p).expanduser(), allow_pickle=False)
        part = {k: d[k] for k in ('X', 'y_go', 'm_go', 'y_slow', 'm_slow', 'ep')}
        part['ep'] = part['ep'] + off
        off += len(d['episodes'])
        names = d['feature_names'].tolist()
        parts.append(part)
    D = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    eps = np.unique(D['ep'])
    perm = np.random.default_rng(12345).permutation(eps)
    n_val = max(1, int(round(args.val_frac * len(eps))))
    val_eps, train_eps = set(perm[:n_val].tolist()), perm[n_val:]
    is_val = np.isin(D['ep'], list(val_eps))
    tr = ~is_val
    X = D['X'].astype(np.float32)
    mean, std = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-3)
    Xn = np.clip((X - mean) / std, -5, 5)
    Y = np.stack([D['y_go'], D['y_slow']], 1).astype(np.float32)
    M = np.stack([D['m_go'], D['m_slow']], 1).astype(np.float32)
    # Class balance per head (positives are rare for both).
    pos_w = []
    for h in range(2):
        m = M[tr, h] > 0
        p = Y[tr, h][m].mean() if m.any() else 0.5
        pos_w.append(float(np.clip((1 - p) / max(p, 1e-3), 1.0, 20.0)))
    pos_w = torch.tensor(pos_w)
    dev = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    def net():
        layers, din = [], X.shape[1]
        for h in args.hidden:
            layers += [nn.Linear(din, h), nn.ReLU()]
            din = h
        layers += [nn.Linear(din, 2)]
        return nn.Sequential(*layers).to(dev)

    T = lambda a: torch.as_tensor(a, device=dev)
    Xv, Yv, Mv = T(Xn[is_val]), T(Y[is_val]), T(M[is_val])

    def loss_fn(model, x, y, m):
        logit = model(x)
        l = nn.functional.binary_cross_entropy_with_logits(
            logit, y, reduction='none', pos_weight=pos_w.to(dev))
        return (l * m).sum() / m.sum().clamp(min=1.0)

    members, ep_of = [], D['ep']
    for e in range(args.members):
        boot = rng.choice(train_eps, size=len(train_eps), replace=True)
        idx = np.concatenate([np.where((ep_of == b) & tr)[0] for b in boot])
        Xt, Yt, Mt = T(Xn[idx]), T(Y[idx]), T(M[idx])
        model = net()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, state = float('inf'), None
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(Xt), device=dev)
            for b in range(0, len(Xt), args.batch):
                s = order[b:b + args.batch]
                loss = loss_fn(model, Xt[s], Yt[s], Mt[s])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model, Xv, Yv, Mv).item()
            if vl < best:
                best, state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        members.append(model)
        print(f'member {e}: {len(idx)} frames, best val loss {best:.4f}')

    with torch.no_grad():
        P = torch.stack([torch.sigmoid(m(Xv)) for m in members]).mean(0).cpu().numpy()
    metrics = {}
    for h, name in enumerate(('go', 'slow')):
        m = M[is_val, h] > 0
        y = Y[is_val, h][m]
        p = P[m, h]
        metrics[f'{name}_val_frames'] = int(m.sum())
        metrics[f'{name}_val_pos_rate'] = float(y.mean()) if m.any() else float('nan')
        metrics[f'{name}_val_auc'] = auc(y, p)
        metrics[f'{name}_val_acc'] = float(((p >= 0.5) == (y > 0.5)).mean()) if m.any() else float('nan')

    # Training cost for the paper (vs teacher labelling / end-to-end training).
    train_cost = dict(wall_s=round(time.perf_counter() - t_start, 1),
                      cpu_s=round(time.process_time() - cpu_start, 1), device=dev,
                      device_name=(torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'),
                      train_frames=int(tr.sum()), members=args.members, epochs=args.epochs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights, n_layers = {}, None
    for e, model in enumerate(members):
        lins = [l for l in model if isinstance(l, nn.Linear)]
        n_layers = len(lins)
        for l, lin in enumerate(lins):
            weights[f'W{e}_{l}'] = lin.weight.detach().cpu().numpy().T.astype(np.float64)
            weights[f'b{e}_{l}'] = lin.bias.detach().cpu().numpy().astype(np.float64)
    np.savez(out / 'tactical.npz', **weights)
    n_params = int(sum(v.size for v in weights.values()))
    meta = dict(n_members=args.members, n_layers=n_layers, hidden=args.hidden,
                x_mean=mean.tolist(), x_std=std.tolist(), feature_names=names,
                n_params_total=n_params, n_params_per_member=n_params // args.members,
                train_episodes=int(len(train_eps)), val_episodes=int(n_val),
                pos_weight=pos_w.tolist(), metrics=metrics, args=vars(args),
                train_cost=train_cost,
                datasets=[str(p) for p in args.data])
    json.dump(meta, open(out / 'tactical_meta.json', 'w'), indent=2)
    print(f'\ntactical student -> {out}  ({n_params} parameters, {n_params // args.members} per member)')
    print(f'  training cost: {train_cost}')
    for k, v in metrics.items():
        print(f'  {k:22s} {v:.3f}' if isinstance(v, float) else f'  {k:22s} {v}')


if __name__ == '__main__':
    main()
