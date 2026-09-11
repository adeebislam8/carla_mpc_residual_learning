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
    """Drive `episodes` episodes for each seed and collect per-episode records."""
    # Diagnostics default ON here.  run_mpc_python.sh sets CARLA_MPC_DIAG but the
    # benchmark invokes python directly, so every benchmark run so far recorded
    # nothing -- which is why diagnostics/ was empty and the Frenet jump rate
    # was never measured.  Set before importing the env: MPCController reads the
    # variable in its constructor.
    if not args.no_diag:
        os.environ.setdefault("CARLA_MPC_DIAG", "1")

    from mpc_controller.envs.carlaEnv import CarlaMPCEnv

    episodes = []
    t_start = time.time()

    # One env per seed.  The seed reaches random.seed() inside CarlaMPCEnv's
    # constructor and drives spawn/goal selection, so it cannot be changed on a
    # live env.  Averaging across seeds is what makes a number reportable: a
    # single seed measures one route sequence, not the controller.
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        env = CarlaMPCEnv(
            host=args.host,
            port=args.port,
            towns=[args.town],
            episodes_per_town=10 ** 9,
            target_speed=args.target_speed,
            max_steps=args.max_steps,
            seed=seed,
            steer_norm_deg=args.steer_norm_deg,
            qc=args.qc,
            a_long_obs=args.a_long,
            b_lat_obs=args.b_lat,
            apex_gain=args.apex_gain,
            gate_depth=args.gate_depth,
            lookahead=args.lookahead,
            r3_cap=args.r3_cap,
            residual_mode='fixed',   # alpha == 1, but action is 0 -> pure MPCC
        )
        interrupted = False
        try:
            _run_seed(env, args, seed, episodes)
        except KeyboardInterrupt:
            print("\ninterrupted -- reporting on the episodes completed so far")
            interrupted = True
        finally:
            try:
                env.close()
            except Exception:
                pass
        if interrupted:
            break

    return {
        'label': args.label,
        'episodes_per_seed': args.episodes,
        'seeds': list(args.seeds),
        'episodes_completed': len(episodes),
        'target_speed': args.target_speed,
        'steer_norm_deg': args.steer_norm_deg,
        'qc': args.qc,
        'a_long': args.a_long,
        'b_lat': args.b_lat,
        'apex_gain': args.apex_gain,
        'gate_depth': args.gate_depth,
        'lookahead': args.lookahead,
        'r3_cap': args.r3_cap,
        'town': args.town,
        'max_steps': args.max_steps,
        'rendered': bool(args.render),
        'camera': args.camera if args.render else None,
        'slowdown': args.slowdown,
        'wall_time_s': time.time() - t_start,
        'per_episode': episodes,
    }


def _run_seed(env, args, seed, episodes):
    zero = np.zeros(2, dtype=float)
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

            # Watching costs nothing measurable: render() only repositions the
            # spectator, and CARLA runs in synchronous mode with a fixed
            # delta, so neither the camera nor --slowdown changes the physics.
            # Metrics stay comparable between rendered and unrendered runs.
            if args.render:
                env.render(camera_mode=args.camera)
                if args.slowdown > 0:
                    time.sleep(args.slowdown)

        rec = {
            'seed': seed,
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
        print(f"  s{seed} ep {ep:3d}  {rec['outcome']:<9} "
              f"overtakes={rec['overtakes']:2d}  "
              f"progress={100*rec['progress_frac']:5.1f}%  "
              f"solver_fail={100*rec['solver_failure_rate']:.2f}%")


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


PAPER_METRICS = [
    ('Success rate (%)',        'success_rate',             100.0),
    ('Collision rate (%)',      'collision_rate',           100.0),
    ('Timeout rate (%)',        'timeout_rate',             100.0),
    ('Route completion (%)',    'mean_progress_frac',       100.0),
    ('Overtakes / episode',     'overtakes_per_episode',      1.0),
    ('Mean speed (m/s)',        'mean_speed',                 1.0),
    ('Lap time, success (s)',   'mean_lap_time_success',      1.0),
    ('Solver failure (%)',      'mean_solver_failure_rate', 100.0),
]


def summarize_per_seed(res):
    """summarize() applied to each seed separately."""
    out = {}
    for seed in res.get('seeds', [res.get('seed')]):
        eps = [e for e in res['per_episode'] if e.get('seed') == seed]
        if eps:
            out[seed] = summarize({'per_episode': eps})
    return out


def across_seeds(per_seed):
    """mean and sample std of each metric across seeds -- what goes in the paper."""
    stats = {}
    for _, key, _ in PAPER_METRICS:
        vals = [s[key] for s in per_seed.values()
                if key in s and not (isinstance(s[key], float) and np.isnan(s[key]))]
        if not vals:
            continue
        stats[key] = (statistics.mean(vals),
                      statistics.stdev(vals) if len(vals) > 1 else 0.0,
                      len(vals))
    return stats


def write_report(res, path):
    s = summarize(res)
    L = []
    L.append("=" * 72)
    L.append(f"NOMINAL MPCC BENCHMARK  --  {res['label']}")
    L.append("=" * 72)
    L.append(f"  episodes      : {res['episodes_completed']} total")
    seeds = res.get('seeds', [res.get('seed')])
    L.append(f"  seeds         : {', '.join(str(x) for x in seeds)}"
             f"   ({res.get('episodes_per_seed', '?')} episodes each)")
    L.append(f"  target_speed  : {res['target_speed']} m/s")
    if 'steer_norm_deg' in res:
        g = 70.0 / res['steer_norm_deg']
        L.append(f"  steer norm    : {res['steer_norm_deg']} deg  -> {g:.3f}x feedforward")
    L.append(f"  qc (lateral)  : {res.get('qc') or 0.05}")
    L.append(f"  ellipse       : a={res.get('a_long') or 4}  b={res.get('b_lat') or 2}"
             f"   apex_gain={res.get('apex_gain') or 0}")
    _qc = res.get('qc') or 0.05
    _gd = res.get('gate_depth') if res.get('gate_depth') is not None else 0.8
    L.append(f"  lane-hold qc  : {_qc:.3f} normally / {_qc*(1-_gd):.3f} while overtaking")
    _la = res.get('lookahead') or 5.0
    _r3 = res.get('r3_cap') or 3e-2
    L.append(f"  corner slowdown: lookahead {_la} m, r3_cap {_r3}"
             f"  -> derTheta {0.4/(2*0.015):.1f} straight / {0.4/(2*_r3):.1f} in a bend")
    L.append(f"  town          : {res.get('town', res.get('towns', ['?'])[0])}")
    L.append(f"  wall time     : {res['wall_time_s']/60:.1f} min")
    L.append(f"  driven with action = [0, 0]  -> pure MPCC, no RL residual")
    if res.get('rendered'):
        L.append(f"  rendered      : yes ({res.get('camera')}), slowdown "
                 f"{res.get('slowdown')} s/step -- physics unaffected")
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
        per_seed = summarize_per_seed(res)
        if len(per_seed) > 1:
            L.append("PER SEED")
            L.append("-" * 72)
            L.append(f"  {'seed':>8}{'eps':>6}{'succ%':>8}{'coll%':>8}"
                     f"{'ovt/ep':>9}{'route%':>9}{'fail%':>8}")
            for seed, st in sorted(per_seed.items()):
                L.append(f"  {seed:>8}{st['n_episodes']:>6}"
                         f"{100*st['success_rate']:>8.1f}{100*st['collision_rate']:>8.1f}"
                         f"{st['overtakes_per_episode']:>9.2f}"
                         f"{100*st['mean_progress_frac']:>9.1f}"
                         f"{100*st['mean_solver_failure_rate']:>8.2f}")
            L.append("")
            L.append("PAPER TABLE  (mean +- std across seeds)")
            L.append("-" * 72)
            st = across_seeds(per_seed)
            for name, key, scale in PAPER_METRICS:
                if key not in st:
                    continue
                m, sd, k = st[key]
                L.append(f"  {name:<26}{scale*m:8.2f}  +-{scale*sd:7.2f}   (n={k} seeds)")
            L.append("")
            L.append("  Report these as mean +- std over seeds, with the episode")
            L.append("  count per seed stated.  std over 3 seeds is a weak estimate --")
            L.append("  quote it, do not run significance tests on it.")
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
            ks = [e['collision_kappa'] for e in colls
                  if e.get('collision_kappa') is not None
                  and e['collision_kappa'] == e['collision_kappa']]
            if ks:
                STRAIGHT = 0.02
                pos = sum(1 for k in ks if k > STRAIGHT)
                neg = sum(1 for k in ks if k < -STRAIGHT)
                flat = len(ks) - pos - neg
                L.append("  by path curvature at impact:")
                L.append(f"      kappa > +{STRAIGHT}  (one turn dir) {pos:4d}"
                         f"  ({100*pos/len(ks):5.1f}%)")
                L.append(f"      kappa < -{STRAIGHT}  (other dir)    {neg:4d}"
                         f"  ({100*neg/len(ks):5.1f}%)")
                L.append(f"      |kappa| <= {STRAIGHT} (straight)     {flat:4d}"
                         f"  ({100*flat/len(ks):5.1f}%)")
                L.append("      a large imbalance = one turn direction is much")
                L.append("      harder; the corridor is ~6x wider left than right")

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
    ap.add_argument('--seeds', type=int, nargs='+', default=[2547],
                    help='one run per seed; metrics are averaged across them')
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
    ap.add_argument('--steer-norm-deg', type=float, default=45.0,
                    help='steering feedforward: CARLA applies over 70 deg, so 45 '
                         'gives 1.556x gain. SMALLER = MORE steering. 70 = none.')
    ap.add_argument('--qc', type=float, default=None,
                    help='lateral tracking weight (default 5e-2 from the model). '
                         'Higher = holds the path harder.')
    ap.add_argument('--a-long', type=float, default=None,
                    help='CBF ellipse longitudinal semi-axis (model default 4). '
                         'LOWER = barrier easier to satisfy = less overtaking.')
    ap.add_argument('--b-lat', type=float, default=None,
                    help='CBF ellipse lateral semi-axis (model default 2). '
                         'Sets the minimum lateral clearance when passing.')
    ap.add_argument('--apex-gain', type=float, default=None,
                    help='signed metres per unit curvature; leans the car toward '
                         'the inside of a bend. 0 = off. Negate if it leans wrong.')
    ap.add_argument('--gate-depth', type=float, default=None,
                    help='lateral relaxation while overtaking (model default 0.8). '
                         'Pair a HIGH qc with a HIGH depth to hold the lane hard '
                         'without making passes expensive.')
    ap.add_argument('--lookahead', type=float, default=None,
                    help='m of curvature preview for the corner slowdown '
                         '(model default 5.0 = ~0.5 s at 10 m/s).')
    ap.add_argument('--r3-cap', type=float, default=None,
                    help='cap on the curvature slowdown (model default 3e-2, '
                         'saturates at R=6 m). Higher = slower in tight turns.')
    ap.add_argument('--no-diag', action='store_true',
                    help='disable solver diagnostics (on by default; writes '
                         'diagnostics/*.npz for analyze_solver_failures.py)')
    ap.add_argument('--render', action='store_true',
                    help='move the CARLA spectator to follow the ego so you can '
                         'watch; does not affect the metrics')
    ap.add_argument('--camera', default='follow',
                    choices=['follow', 'top_down', 'side', 'first_person'],
                    help='spectator view when --render is set (default: follow)')
    ap.add_argument('--slowdown', type=float, default=0.0, metavar='SEC',
                    help='sleep this long per step so the run is watchable '
                         '(0.02 is comfortable); does not affect the metrics')
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
