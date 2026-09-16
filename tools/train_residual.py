#!/usr/bin/env python
"""
Train the residual policy on top of the nominal MPCC.

Mirrors tools/benchmark_mpcc.py: the same controller flags, so the residual is
trained against exactly the controller it will be evaluated with.  A residual
learns "given THIS nominal behaviour, what correction helps" -- train it against
a different MPCC config and it is invalid (project spec section 8).

    python tools/train_residual.py --label sac_v1 --algo sac --timesteps 300000

Then evaluate through the SAME harness as the nominal, so the numbers are
directly comparable:

    python tools/benchmark_mpcc.py --label b0 --seeds 1 2 3 --episodes 20
    python tools/benchmark_mpcc.py --label b2 --model models/sac_v1/final.zip \\
        --residual-mode fixed    --seeds 1 2 3 --episodes 20
    python tools/benchmark_mpcc.py --label b5 --model models/sac_v1/final.zip \\
        --residual-mode adaptive --seeds 1 2 3 --episodes 20
    python tools/benchmark_mpcc.py --compare results/b0.json results/b2.json results/b5.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))


def build_env(args, seed, for_eval=False):
    from mpc_controller.envs.carlaEnv import CarlaMPCEnv
    return CarlaMPCEnv(
        host=args.host, port=args.port, towns=[args.town],
        episodes_per_town=10 ** 9,
        target_speed=args.target_speed, max_steps=args.max_steps,
        seed=seed + (10_000 if for_eval else 0),
        residual_mode=args.residual_mode,
        residual_max=args.residual_max,
        # controller config -- must match evaluation
        qc=args.qc, gate_depth=args.gate_depth,
        a_long_obs=args.a_long, b_lat_obs=args.b_lat,
        apex_gain=args.apex_gain, lookahead=args.lookahead, r3_cap=args.r3_cap,
        route_min_m=args.route_min, route_max_m=args.route_max,
        npc_min=args.npc_min, npc_max=args.npc_max,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='residual')
    ap.add_argument('--algo', default='sac', choices=['sac', 'td3', 'ppo'])
    ap.add_argument('--timesteps', type=int, default=300_000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--buffer-size', type=int, default=300_000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--ent-coef', default='0.005',
                    help="SAC entropy coefficient. 'auto' PUSHES the residual "
                         "away from zero, which is wrong for a correction "
                         "policy -- a small fixed value is usually better.")
    ap.add_argument('--no-normalize', action='store_true',
                    help='skip VecNormalize (not recommended: the observation '
                         'mixes s in metres up to ~900 with angles ~0.1)')
    # controller config, mirroring benchmark_mpcc.py
    ap.add_argument('--residual-mode', default='fixed',
                    choices=['fixed', 'adaptive'])
    ap.add_argument('--residual-max', type=float, default=0.1)
    ap.add_argument('--qc', type=float, default=0.5)
    ap.add_argument('--gate-depth', type=float, default=0.98)
    ap.add_argument('--a-long', type=float, default=None)
    ap.add_argument('--b-lat', type=float, default=None)
    ap.add_argument('--apex-gain', type=float, default=None)
    ap.add_argument('--lookahead', type=float, default=None)
    ap.add_argument('--r3-cap', type=float, default=None)
    ap.add_argument('--route-min', type=float, default=50.0)
    ap.add_argument('--route-max', type=float, default=None)
    ap.add_argument('--npc-min', type=int, default=0)
    ap.add_argument('--npc-max', type=int, default=7)
    ap.add_argument('--target-speed', type=float, default=15.0)
    ap.add_argument('--max-steps', type=int, default=1500)
    ap.add_argument('--town', default='Town01')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    args = ap.parse_args()

    from stable_baselines3 import SAC, TD3, PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.noise import NormalActionNoise
    import numpy as np

    out = os.path.join('models', args.label)
    os.makedirs(out, exist_ok=True)

    # ONE env.  train_rlmpc.train_sac built a second CarlaMPCEnv on the same
    # port for evaluation; both constructors call load_world() and set
    # synchronous_mode on the SAME server, so two sync clients end up fighting
    # over world.tick().  Evaluate afterwards with benchmark_mpcc.py instead --
    # that also keeps evaluation identical to the nominal-controller runs.
    env = DummyVecEnv([lambda: Monitor(build_env(args, args.seed))])
    if not args.no_normalize:
        # The observation mixes arc length (0-900 m) with lateral error (~1 m)
        # and angles (~0.1 rad).  Unnormalised, the large-magnitude features
        # dominate the first layer and the small ones are invisible -- this
        # usually matters more than the choice of algorithm.
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # tensorboard_log raises ImportError at learn() time if tensorboard is not
    # installed -- and because the save happens in a `finally`, that produced an
    # UNTRAINED model on disk that looked like a successful run.
    try:
        import tensorboard  # noqa: F401
        tb = os.path.join(out, 'tb')
    except ImportError:
        tb = None
        print("tensorboard not installed -- logging to CSV only "
              "(pip install tensorboard to enable)")

    common = dict(policy='MlpPolicy', env=env, verbose=1, seed=args.seed,
                  tensorboard_log=tb,
                  policy_kwargs=dict(net_arch=[256, 256]))

    if args.algo == 'sac':
        ent = args.ent_coef if args.ent_coef == 'auto' else float(args.ent_coef)
        model = SAC(learning_rate=args.lr, buffer_size=args.buffer_size,
                    batch_size=args.batch_size, learning_starts=5000,
                    gradient_steps=1, tau=0.005, gamma=0.99,
                    ent_coef=ent, **common)
    elif args.algo == 'td3':
        # Deterministic policy: no entropy bonus pushing the residual away from
        # zero.  For a correction that should be small unless it helps, that is
        # the right inductive bias.
        n_act = env.action_space.shape[-1]
        model = TD3(learning_rate=args.lr, buffer_size=args.buffer_size,
                    batch_size=args.batch_size, learning_starts=5000,
                    gradient_steps=1, tau=0.005, gamma=0.99,
                    action_noise=NormalActionNoise(np.zeros(n_act),
                                                   0.1 * np.ones(n_act)),
                    **common)
    else:
        model = PPO(learning_rate=args.lr, n_steps=2048,
                    batch_size=args.batch_size, gamma=0.99, **common)

    ckpt = CheckpointCallback(save_freq=25_000, save_path=out,
                              name_prefix=args.label)
    print(f"\n=== training {args.algo.upper()} for {args.timesteps} steps -> {out}\n")
    trained = False
    try:
        model.learn(total_timesteps=args.timesteps, callback=[ckpt],
                    progress_bar=True)
        trained = True
    except KeyboardInterrupt:
        print("\ninterrupted -- saving what we have")
        trained = True
    except Exception as e:
        # Do NOT save on an unexpected failure: a randomly-initialised policy
        # written to final.zip is indistinguishable from a trained one and gets
        # silently evaluated later.
        print(f"\nTRAINING FAILED: {e!r}")
        trained = False
        raise
    finally:
        if trained:
            model.save(os.path.join(out, 'final'))
            if not args.no_normalize:
                env.save(os.path.join(out, 'vecnormalize.pkl'))
            print(f"\nsaved -> {os.path.join(out, 'final.zip')}")
        env.close()


if __name__ == '__main__':
    main()
