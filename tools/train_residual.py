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
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Flags that change what the policy sees or how its action is applied.  A
# policy is only valid under the values it was trained with, so they are saved
# next to the model (train_config.json) and benchmark_mpcc.py refuses to
# evaluate under different ones.
POLICY_CONTRACT = ('residual_mode', 'residual_max', 'authority_horizon',
                   'residual_window', 'obs_version', 'qc', 'gate_depth',
                   'route_max', 'target_speed')


def make_skip_wrapper():
    """Built lazily so importing this module does not need gymnasium."""
    import gymnasium as gym
    import numpy as np

    class SkipInactiveSteps(gym.Wrapper):
        """
        Training only: fast-forward through steps where the residual window
        is closed, driving them with a zero residual.

        With --residual-window the env already ignores the residual outside
        the window.  Without this wrapper the agent would still be queried on
        every one of those steps and every transition -- whose action had no
        effect -- would go into the replay buffer, which is mostly lane
        keeping.  Here the agent only sees decision points near obstacles.

        Rewards earned while fast-forwarding are NOT credited to the agent's
        last action, except the terminal step's (collision / success), which
        does end its episode.  Crediting a few hundred steps of MPCC-driven
        progress to one residual action is exactly the attribution noise this
        exists to remove.
        """

        def __init__(self, env, max_reset_tries=20):
            super().__init__(env)
            self.max_reset_tries = max_reset_tries
            self._zero = np.zeros(env.action_space.shape)

        def _window_open(self):
            return self.env.unwrapped._residual_window_active()

        def _fast_forward(self, obs, info):
            """Step with a zero residual until the window opens or the episode
            ends.  Returns (obs, terminal_reward, terminated, truncated, info,
            steps_skipped)."""
            n = 0
            while not self._window_open():
                obs, r, term, trunc, info = self.env.step(self._zero)
                n += 1
                if term or trunc:
                    return obs, r, term, trunc, info, n
            return obs, 0.0, False, False, info, n

        def reset(self, **kwargs):
            for attempt in range(self.max_reset_tries):
                obs, info = self.env.reset(**kwargs)
                kwargs.pop('seed', None)    # seed only the first reset
                obs, _, term, trunc, info, _ = self._fast_forward(obs, info)
                if not (term or trunc):
                    return obs, info
            print(f"  [skip] {self.max_reset_tries} episodes in a row never "
                  f"opened the residual window -- handing back the last one")
            return obs, info

        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action)
            skipped = 0
            if not (term or trunc):
                obs, r_end, term, trunc, info, skipped = self._fast_forward(obs, info)
                r += r_end
            info['skipped_steps'] = skipped
            return obs, r, term, trunc, info

    return SkipInactiveSteps


def make_outcome_logger(path):
    """Learning curve for the paper: episode outcome vs environment steps.

    Written per episode to CSV, so success/collision rate against training
    steps can be plotted without re-running anything.  'env_steps' counts
    agent steps (fast-forwarded steps are not agent steps), 'sim_steps' counts
    simulator ticks, which is the honest sample-efficiency x-axis."""
    from stable_baselines3.common.callbacks import BaseCallback

    class OutcomeLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.recent = []
            self.sim_steps = 0
            self.episode = 0
            self._f = open(path, 'w', newline='')
            self._w = csv.writer(self._f)
            self._w.writerow(['episode', 'env_steps', 'sim_steps', 'outcome',
                              'success_rate_last50', 'collision_rate_last50'])

        def _on_step(self):
            for info, done in zip(self.locals['infos'], self.locals['dones']):
                self.sim_steps += 1 + int(info.get('skipped_steps', 0))
                if not done:
                    continue
                outcome = info.get('done_reason', 'unknown')
                self.episode += 1
                self.recent = (self.recent + [outcome])[-50:]
                succ = self.recent.count('success') / len(self.recent)
                coll = self.recent.count('collision') / len(self.recent)
                self._w.writerow([self.episode, self.num_timesteps,
                                  self.sim_steps, outcome,
                                  f'{succ:.3f}', f'{coll:.3f}'])
                self._f.flush()
                self.logger.record('outcome/success_rate_last50', succ)
                self.logger.record('outcome/collision_rate_last50', coll)
            return True

        def _on_training_end(self):
            self._f.close()

    return OutcomeLogger()


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
        authority_horizon=args.authority_horizon,
        residual_window_m=args.residual_window,
        obs_version=args.obs_version,
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
    ap.add_argument('--residual-max', type=float, default=0.1,
                    help='residual size at alpha = 1.  Raise above 0.1 only '
                         'with --residual-mode adaptive and --authority-horizon '
                         '10: below that the gate cannot see throttle')
    ap.add_argument('--authority-horizon', type=int, default=1,
                    help='steps the authority gate rolls the final action '
                         'forward; 1 = original one-step check (throttle-blind)')
    ap.add_argument('--residual-window', type=float, default=None, metavar='M',
                    help='residual acts only with an obstacle 0 < ds < M ahead; '
                         'training also skips the agent past closed-window steps')
    ap.add_argument('--obs-version', default='v1', choices=['v1', 'v2'],
                    help="'v2' drops route-position features (see carlaEnv)")
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

    if args.residual_max > 0.1 and (args.residual_mode != 'adaptive'
                                    or args.authority_horizon < 2):
        ap.error('--residual-max above 0.1 needs --residual-mode adaptive and '
                 '--authority-horizon >= 2: a one-step gate cannot see throttle, '
                 'and a fixed-mode residual is not gated at all')

    out = os.path.join('models', args.label)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'train_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    def make_env():
        e = build_env(args, args.seed)
        if args.residual_window is not None:
            e = make_skip_wrapper()(e)
        return Monitor(e)

    # ONE env.  train_rlmpc.train_sac built a second CarlaMPCEnv on the same
    # port for evaluation; both constructors call load_world() and set
    # synchronous_mode on the SAME server, so two sync clients end up fighting
    # over world.tick().  Evaluate afterwards with benchmark_mpcc.py instead --
    # that also keeps evaluation identical to the nominal-controller runs.
    env = DummyVecEnv([make_env])
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

    # save_vecnormalize: a checkpoint is only evaluable with the observation
    # statistics it was trained under.  Without them only final.zip could be
    # evaluated, and the learning curve the paper needs could not be drawn.
    ckpt = CheckpointCallback(save_freq=25_000, save_path=out,
                              name_prefix=args.label,
                              save_vecnormalize=not args.no_normalize)
    outcomes = make_outcome_logger(os.path.join(out, 'outcomes.csv'))
    print(f"\n=== training {args.algo.upper()} for {args.timesteps} steps -> {out}\n")
    trained = False
    try:
        model.learn(total_timesteps=args.timesteps, callback=[ckpt, outcomes],
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
