#!/usr/bin/env python
"""
Benchmark the nominal MPCC over a fixed set of episodes and write a report.

Phase 1 of the plan: settle the nominal controller, then freeze it.  This is
MPCC only -- the env is stepped with action = [0, 0], so the residual RL path
contributes nothing and what is measured is purely the model-based controller.

Every run uses the same seed and episode count, so two configurations are driven
through the same spawn/goal sequence and can be compared directly.  Results go to
a human-readable .txt and a machine-readable .json.

    # measure a configuration
    python tools/benchmark_mpcc.py --label baseline --episodes 20

    # change one thing in the model, re-measure
    python tools/benchmark_mpcc.py --label n_ref --episodes 20

    # compare
    python tools/benchmark_mpcc.py --compare results/baseline.json results/n_ref.json

Judge `n_ref` on overtakes and collisions, NOT on solver failure rate -- the
attractor deliberately raises |n| during passes, and |n| > 2 m is the dominant
condition among the remaining solver failures.  A rise there alongside more
overtakes and fewer collisions is a trade worth taking.
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))


def run(args):
    from mpc_controller.envs.carlaEnv import CarlaMPCEnv

    env = CarlaMPCEnv(
        host=args.host,
        port=args.port,
        towns=[args.town],
        episodes_per_town=10 ** 9,
        target_speed=args.target_speed,
        max_steps=args.max_steps,
        seed=args.seed,
        residual_mode='fixed',      # alpha == 1, but action is 0 -> pure MPCC
    )

    episodes = []
    zero = np.zeros(2, dtype=float)
    t_start = time.time()

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()

            n_hist, v_hist = [], []
            steps = 0
            done = False
            info = {}

            while not done and steps < args.max_steps:
                obs, reward, done, truncated, info = env.step(zero)
                n_hist.append(float(env.current_d))
                v_hist.append(float(env.current_speed))
                steps += 1

            rec = {
                'episode': ep,
                'outcome': info.get('done_reason', 'timeout'),
                'steps': steps,
                'overtakes': len(env.overtaken_npcs),
                'progress_m': float(env.current_s),
                'path_length_m': float(env.path_length),
                'progress_frac': float(env.current_s / max(env.path_length, 1e-6)),
                'lap_time_s': float(info.get('lap_time', float('nan'))),
                'mean_speed': float(np.mean(v_hist)) if v_hist else 0.0,
                'max_abs_n': float(np.max(np.abs(n_hist))) if n_hist else 0.0,
                'frac_n_gt2': float(np.mean(np.abs(n_hist) > 2.0)) if n_hist else 0.0,
                'solver_failure_rate': float(info.get('solver_failure_rate', 0.0)),
                'solver_failures': int(info.get('solver_failures', 0)),
                'collision_kind': info.get('collision_kind'),
                'collision_other': info.get('collision_other'),
                'collision_speed': info.get('collision_speed'),
                'collision_off_corridor': info.get('collision_off_corridor'),
                'collision_in_fallback': info.get('collision_in_fallback'),
            }
            episodes.append(rec)
            print(f"  ep {ep:3d}  {rec['outcome']:<9} "
                  f"overtakes={rec['overtakes']:2d}  "
                  f"progress={100*rec['progress_frac']:5.1f}%  "
                  f"solver_fail={100*rec['solver_failure_rate']:.2f}%")
    except KeyboardInterrupt:
        print("\ninterrupted -- reporting on the episodes completed so far")
    finally:
        env.close()

    return {
        'label': args.label,
        'episodes_requested': args.episodes,
        'episodes_completed': len(episodes),
        'seed': args.seed,
        'target_speed': args.target_speed,
        'town': args.town,
        'max_steps': args.max_steps,
        'wall_time_s': time.time() - t_start,
        'per_episode': episodes,
    }


def summarize(res):
    eps = res['per_episode']
    n = len(eps)
    if n == 0:
        return {}

    def frac(reason):
        return sum(1 for e in eps if e['outcome'] == reason) / n

    def mean(key):
        return statistics.mean(e[key] for e in eps)

    finished = [e for e in eps if e['outcome'] == 'success']
    return {
        'n_episodes': n,
        'success_rate': frac('success'),
        'collision_rate': frac('collision'),
        'stall_rate': frac('stall'),
        'timeout_rate': frac('timeout'),
        'error_rate': frac('error'),
        'overtakes_total': sum(e['overtakes'] for e in eps),
        'overtakes_per_episode': mean('overtakes'),
        'mean_progress_frac': mean('progress_frac'),
        'mean_speed': mean('mean_speed'),
        'mean_max_abs_n': mean('max_abs_n'),
        'mean_frac_n_gt2': mean('frac_n_gt2'),
        'mean_solver_failure_rate': mean('solver_failure_rate'),
        'mean_lap_time_success': (statistics.mean(e['lap_time_s'] for e in finished)
                                  if finished else float('nan')),
    }


def write_report(res, path):
    s = summarize(res)
    L = []
    L.append("=" * 72)
    L.append(f"NOMINAL MPCC BENCHMARK  --  {res['label']}")
    L.append("=" * 72)
    L.append(f"  episodes      : {res['episodes_completed']}/{res['episodes_requested']}")
    L.append(f"  seed          : {res['seed']}   (same seed => same spawn sequence)")
    L.append(f"  target_speed  : {res['target_speed']} m/s")
    L.append(f"  town          : {res.get('town', res.get('towns', ['?'])[0])}")
    L.append(f"  wall time     : {res['wall_time_s']/60:.1f} min")
    L.append(f"  driven with action = [0, 0]  -> pure MPCC, no RL residual")
    L.append("")

    if not s:
        L.append("  no episodes completed")
    else:
        L.append("OUTCOMES")
        L.append("-" * 72)
        for k in ('success', 'collision', 'stall', 'timeout', 'error'):
            L.append(f"  {k:<12} {100*s[k + '_rate']:6.1f}%")
        L.append("")
        L.append("BEHAVIOUR  (the numbers that decide Phase 1)")
        L.append("-" * 72)
        L.append(f"  overtakes / episode      {s['overtakes_per_episode']:8.2f}"
                 f"   (total {s['overtakes_total']})")
        L.append(f"  mean progress            {100*s['mean_progress_frac']:8.1f}%")
        L.append(f"  mean speed               {s['mean_speed']:8.2f} m/s")
        L.append(f"  mean lap time (success)  {s['mean_lap_time_success']:8.2f} s")
        L.append("")
        L.append("LATERAL USE  (expected to RISE if the overtake attractor works)")
        L.append("-" * 72)
        L.append(f"  mean max |n|             {s['mean_max_abs_n']:8.2f} m")
        L.append(f"  fraction of steps |n|>2  {100*s['mean_frac_n_gt2']:8.2f}%")
        L.append("")
        L.append("SOLVER  (secondary -- a rise here is acceptable if overtakes rise too)")
        L.append("-" * 72)
        L.append(f"  mean failure rate        {100*s['mean_solver_failure_rate']:8.2f}%")
        L.append("")
        colls = [e for e in res['per_episode'] if e['outcome'] == 'collision'
                 and e.get('collision_kind')]
        if colls:
            L.append("COLLISION BREAKDOWN  (what is actually being hit)")
            L.append("-" * 72)
            from collections import Counter
            for label, key in (("impact side", 'collision_kind'),
                               ("other actor", 'collision_other')):
                cnt = Counter(str(e[key]) for e in colls)
                L.append(f"  by {label}:")
                for k, v in cnt.most_common():
                    L.append(f"      {k:<34}{v:4d}  ({100*v/len(colls):5.1f}%)")
            off = sum(1 for e in colls if e.get('collision_off_corridor'))
            fb = sum(1 for e in colls if e.get('collision_in_fallback'))
            spd = [e['collision_speed'] for e in colls
                   if e.get('collision_speed') is not None]
            L.append(f"  outside the lateral corridor:   {off:4d}"
                     f"  ({100*off/len(colls):5.1f}%)  <- tracking failure")
            L.append(f"  MPC had fallen back to fallback:{fb:4d}"
                     f"  ({100*fb/len(colls):5.1f}%)  <- solver failure, not control")
            if spd:
                L.append(f"  mean speed at impact:           {statistics.mean(spd):6.2f} m/s")
            L.append("")

        L.append("PER EPISODE")
        L.append("-" * 72)
        L.append(f"  {'ep':>3} {'outcome':<10}{'steps':>7}{'ovt':>5}"
                 f"{'prog%':>8}{'max|n|':>8}{'fail%':>8}")
        for e in res['per_episode']:
            L.append(f"  {e['episode']:>3} {e['outcome']:<10}{e['steps']:>7}"
                     f"{e['overtakes']:>5}{100*e['progress_frac']:>8.1f}"
                     f"{e['max_abs_n']:>8.2f}{100*e['solver_failure_rate']:>8.2f}")

    text = "\n".join(L) + "\n"
    with open(path, 'w') as f:
        f.write(text)
    print("\n" + text)
    print(f"[report] {path}")


def compare(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        runs.append((r['label'], summarize(r), r))

    rows = [
        ('overtakes / episode', 'overtakes_per_episode', 1.0, 'higher better'),
        ('success rate %', 'success_rate', 100.0, 'higher better'),
        ('collision rate %', 'collision_rate', 100.0, 'LOWER better'),
        ('mean progress %', 'mean_progress_frac', 100.0, 'higher better'),
        ('mean speed m/s', 'mean_speed', 1.0, ''),
        ('mean max |n| m', 'mean_max_abs_n', 1.0, 'rise expected'),
        ('steps |n|>2 %', 'mean_frac_n_gt2', 100.0, 'rise expected'),
        ('solver failure %', 'mean_solver_failure_rate', 100.0, 'secondary'),
    ]

    L = ["=" * 78, "MPCC CONFIGURATION COMPARISON", "=" * 78, ""]
    header = f"  {'metric':<22}" + "".join(f"{lab[:12]:>14}" for lab, _, _ in runs)
    L.append(header)
    L.append("  " + "-" * 74)
    for name, key, scale, note in rows:
        line = f"  {name:<22}"
        for _, s, _ in runs:
            line += f"{scale*s.get(key, float('nan')):>14.2f}" if s else f"{'-':>14}"
        L.append(line + (f"   ({note})" if note else ""))
    L.append("")
    L.append("  Episode counts: " + ", ".join(
        f"{lab}={s.get('n_episodes', 0)}" for lab, s, _ in runs))
    L.append("")
    L.append("  With ~20 episodes these differences are indicative, not significant.")
    L.append("  Decide on overtakes and collisions together; ignore small solver moves.")
    text = "\n".join(L) + "\n"
    print(text)
    out = 'results/comparison.txt'
    os.makedirs('results', exist_ok=True)
    with open(out, 'w') as f:
        f.write(text)
    print(f"[report] {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='run', help='name for this configuration')
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--seed', type=int, default=2547)
    ap.add_argument('--target-speed', type=float, default=15.0,
                    help='drop this for the non-racing reframe (see WORKLOG.md)')
    ap.add_argument('--max-steps', type=int, default=1500)
    # Only towns[0] reaches load_world(), and the town-rotation block in reset()
    # is commented out, so exactly one map is ever used.  Town01 is what every
    # previous run and every saved diagnostics/*.npz used -- keep it unless you
    # deliberately want a different map, or comparisons break.
    ap.add_argument('--town', default='Town01',
                    help='map to benchmark on (default Town01, matching prior runs)')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--compare', nargs='+', metavar='JSON',
                    help='compare previously saved runs instead of driving')
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    res = run(args)
    with open(os.path.join(args.out_dir, f"{args.label}.json"), 'w') as f:
        json.dump(res, f, indent=2)
    write_report(res, os.path.join(args.out_dir, f"{args.label}.txt"))


if __name__ == '__main__':
    main()
