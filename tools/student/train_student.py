#!/usr/bin/env python
"""
Train the distilled residual student (supervised, from ORION labels).

    python tools/student/train_student.py \\
        --data ~/Documents/nett/orion_data/ds_round0.npz --out models/student_r0

    # DAgger: aggregate every round's dataset
    python tools/student/train_student.py --data ds_round0.npz ds_round1.npz --out models/student_r1

    # data-efficiency curve: same validation episodes, fewer training episodes
    python tools/student/train_student.py --data ds_round0.npz --fraction 0.25 --out models/student_r0_f25

Runs in the `orion` conda env (torch + GPU); a CPU is fine too -- the model is
tiny.  Output, consumed by student.policy.StudentPolicy (numpy only):

    <out>/student.npz, student_meta.json   weights + normalisation + metrics
    <out>/train_config.json                policy contract for benchmark_mpcc.py
    <out>/train_log.csv                    per-epoch losses

Model: an ensemble of MLPs, obs (68) -> K x 2 normalised residual actions,
tanh output.  Each member trains on a bootstrap resample of the training
EPISODES (frames within an episode are strongly correlated, so resampling
frames would understate the spread).  The ensemble spread is what the
optional support gate uses at run time.

Loss: masked Huber on every chunk step + a small L1 pull toward zero.  The
pull matters: where ORION and the MPCC merely differ in style the target is
noise, and the residual should default to doing nothing.

Validation is split by episode and is identical for every --fraction, so
data-efficiency runs are comparable.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def load_datasets(paths):
    parts, config, ep_offset, names = [], None, 0, []
    for p in paths:
        d = np.load(Path(p).expanduser(), allow_pickle=False)
        cfg = json.loads(str(d['config']))
        if config is None:
            config = cfg
        elif any(cfg[k] != config[k] for k in ('chunk', 'residual_max', 'window', 'obs_version')):
            raise SystemExit(f'{p}: dataset config {cfg} differs from {config}')
        part = {k: d[k] for k in ('X', 'Y', 'M', 'active', 'precoll', 'ep')}
        part['ep'] = part['ep'] + ep_offset
        ep_offset += len(d['episodes'])
        names += [f'{Path(p).stem}/{n}' for n in d['episodes']]
        parts.append(part)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return data, config, names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--members', type=int, default=5)
    ap.add_argument('--hidden', type=int, nargs='+', default=[128, 128])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--huber', type=float, default=0.1)
    ap.add_argument('--l1', type=float, default=1e-3, help='pull toward zero residual')
    ap.add_argument('--precoll-weight', type=float, default=1.0,
                    help='loss weight on frames shortly before a collision')
    ap.add_argument('--all-frames', action='store_true',
                    help='also train on frames with the residual window closed')
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--fraction', type=float, default=1.0,
                    help='use this fraction of the TRAINING episodes')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    # Policy contract: the env flags this student is valid under.  Written to
    # train_config.json; benchmark_mpcc.py refuses to evaluate under others.
    ap.add_argument('--residual-mode', default='adaptive')
    ap.add_argument('--authority-horizon', type=int, default=10)
    ap.add_argument('--qc', type=float, default=0.5)
    ap.add_argument('--gate-depth', type=float, default=0.98)
    ap.add_argument('--route-max', type=float, default=150.0)
    ap.add_argument('--target-speed', type=float, default=15.0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    data, cfg, names = load_datasets(args.data)
    K = int(cfg['chunk'])
    use = np.ones(len(data['X']), bool) if args.all_frames else data['active']

    # Episode split: fixed validation episodes regardless of --fraction.
    episodes = np.unique(data['ep'])
    perm = np.random.default_rng(12345).permutation(episodes)
    n_val = max(1, int(round(args.val_frac * len(episodes))))
    val_eps = set(perm[:n_val].tolist())
    train_pool = perm[n_val:]
    n_train = max(1, int(round(args.fraction * len(train_pool))))
    train_eps = rng.permutation(train_pool)[:n_train]

    is_val = np.isin(data['ep'], list(val_eps)) & use
    is_train_pool = np.isin(data['ep'], train_eps) & use
    if is_train_pool.sum() == 0 or is_val.sum() == 0:
        raise SystemExit(
            f'empty split: train {is_train_pool.sum()} val {is_val.sum()} frames '
            f'from {len(episodes)} episodes.  Too few episodes have an obstacle '
            f'inside the residual window -- add data, or pass --all-frames to '
            f'also train on closed-window frames.')

    X = data['X'].astype(np.float32)
    mean = X[is_train_pool].mean(axis=0)
    std = np.maximum(X[is_train_pool].std(axis=0), 1e-3)
    Xn = (X - mean) / std
    W = np.where(data['precoll'], args.precoll_weight, 1.0).astype(np.float32)

    def to_t(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=dev)

    Xv, Yv, Mv, Wv = (to_t(Xn[is_val]), to_t(data['Y'][is_val]),
                      to_t(data['M'][is_val]), to_t(W[is_val]))

    def make_net():
        layers, d_in = [], X.shape[1]
        for h in args.hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU()]
            d_in = h
        layers += [nn.Linear(d_in, K * 2), nn.Tanh()]
        return nn.Sequential(*layers).to(dev)

    def loss_fn(net, x, y, m, w):
        pred = net(x).view(-1, K, 2)
        hub = nn.functional.huber_loss(pred, y, reduction='none', delta=args.huber).sum(-1)
        hub = (hub * m * w[:, None]).sum() / (m * w[:, None]).sum().clamp(min=1.0)
        return hub + args.l1 * pred.abs().mean(), pred

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = csv.writer(open(out / 'train_log.csv', 'w', newline=''))
    log.writerow(['member', 'epoch', 'train_loss', 'val_loss'])

    members, member_val = [], []
    ep_of = data['ep']
    for e in range(args.members):
        # Bootstrap over training EPISODES.
        boot = rng.choice(train_eps, size=len(train_eps), replace=True)
        counts = {ep: int((boot == ep).sum()) for ep in np.unique(boot)}
        idx = np.concatenate([np.repeat(np.where((ep_of == ep) & is_train_pool)[0], c)
                              for ep, c in counts.items()])
        Xt, Yt, Mt, Wt = (to_t(Xn[idx]), to_t(data['Y'][idx]),
                          to_t(data['M'][idx]), to_t(W[idx]))
        net = make_net()
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, best_state = float('inf'), None
        for epoch in range(args.epochs):
            net.train()
            order = torch.randperm(len(Xt), device=dev)
            tot = 0.0
            for b in range(0, len(Xt), args.batch):
                sel = order[b:b + args.batch]
                loss, _ = loss_fn(net, Xt[sel], Yt[sel], Mt[sel], Wt[sel])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(sel)
            net.eval()
            with torch.no_grad():
                vl, _ = loss_fn(net, Xv, Yv, Mv, Wv)
            log.writerow([e, epoch, tot / len(Xt), vl.item()])
            if vl.item() < best:
                best = vl.item()
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        members.append(net)
        member_val.append(best)
        print(f'member {e}: {len(train_eps)} bootstrap episodes, '
              f'{len(idx)} frames, best val loss {best:.4f}')

    # Ensemble metrics on validation, first chunk step.
    with torch.no_grad():
        preds = torch.stack([m(Xv).view(-1, K, 2) for m in members])
    mean_pred = preds.mean(0)[:, 0].cpu().numpy()
    spread = preds.std(0)[:, 0].mean().item()
    y0 = data['Y'][is_val][:, 0]
    metrics = {}
    for d, name in enumerate(('throttle', 'steering')):
        ss_res = np.sum((y0[:, d] - mean_pred[:, d]) ** 2)
        ss_tot = np.sum((y0[:, d] - y0[:, d].mean()) ** 2)
        metrics[f'val_r2_{name}'] = float(1 - ss_res / max(ss_tot, 1e-12))
        big = np.abs(y0[:, d]) > 0.1
        metrics[f'val_sign_agree_{name}'] = (
            float(np.mean(np.sign(mean_pred[big, d]) == np.sign(y0[big, d])))
            if big.any() else float('nan'))
        metrics[f'val_mae_{name}'] = float(np.mean(np.abs(y0[:, d] - mean_pred[:, d])))
    metrics['val_ensemble_std'] = spread
    metrics['val_loss_members'] = member_val

    # Export for numpy inference: Linear weight (out, in) -> W (in, out).
    weights, n_layers = {}, None
    for e, net in enumerate(members):
        lins = [m for m in net if isinstance(m, nn.Linear)]
        n_layers = len(lins)
        for l, lin in enumerate(lins):
            weights[f'W{e}_{l}'] = lin.weight.detach().cpu().numpy().T.astype(np.float64)
            weights[f'b{e}_{l}'] = lin.bias.detach().cpu().numpy().astype(np.float64)
    np.savez(out / 'student.npz', **weights)
    n_params = int(sum(v.size for v in weights.values()))
    meta = dict(
        n_members=args.members, n_layers=n_layers, hidden=args.hidden, chunk=K,
        residual_max=cfg['residual_max'], window=cfg['window'],
        obs_version=cfg['obs_version'], obs_mean=mean.tolist(), obs_std=std.tolist(),
        n_params_total=n_params, n_params_per_member=n_params // args.members,
        train_episodes=int(len(train_eps)), val_episodes=int(n_val),
        train_frames=int(is_train_pool.sum()), val_frames=int(is_val.sum()),
        fraction=args.fraction, datasets=[str(p) for p in args.data],
        dataset_config=cfg, metrics=metrics, args=vars(args))
    with open(out / 'student_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    contract = dict(residual_mode=args.residual_mode, residual_max=cfg['residual_max'],
                    authority_horizon=args.authority_horizon,
                    residual_window=cfg['window'], obs_version=cfg['obs_version'],
                    qc=args.qc, gate_depth=args.gate_depth, route_max=args.route_max,
                    target_speed=args.target_speed)
    with open(out / 'train_config.json', 'w') as f:
        json.dump(contract, f, indent=2)

    print(f'\nstudent -> {out}  ({n_params} parameters, {args.members} members, '
          f'{n_params // args.members} each)')
    print(f'  train {len(train_eps)} episodes / {int(is_train_pool.sum())} frames, '
          f'val {n_val} episodes / {int(is_val.sum())} frames')
    for k, v in metrics.items():
        if not isinstance(v, list):
            print(f'  {k:28s} {v:+.3f}')


if __name__ == '__main__':
    main()
