"""
Deployed student: a small MLP ensemble evaluated in plain numpy.

Training (tools/student/train_student.py) uses torch; deployment does not, so
the CARLA-side env needs no torch at all, and the per-step cost is a few
small matrix products -- the "efficient" half of the paper's claim.

The network maps the v2 observation to a K-step chunk of normalised residual
actions.  StudentPolicy turns that into one action per step:

  * window closed (no obstacle within residual_window_m): return zeros and do
    not run the network at all -- the env would ignore the residual anyway.
  * chunk_mode='speculative' (idea 1): draft K steps with one forward pass,
    play them one per step while the CBF authority gate accepts them whole
    (alpha_safe ~ 1 on the previous step), and re-query on the first step the
    gate cut, or when the chunk runs out.  The gate is the verifier.
  * chunk_mode='every_step': query every step, use the first action.

Counters (n_steps, n_active, n_queries) give the inference saving directly.

Files written by the trainer, in one directory:
  student.npz         W{e}_{l}, b{e}_{l} for member e, layer l
  student_meta.json   obs_mean, obs_std, chunk K, residual_max, n_members, ...
  train_config.json   the policy contract benchmark_mpcc.py checks
"""

import json
import os

import numpy as np


class StudentEnsemble:
    def __init__(self, weights, meta):
        self.meta = meta
        self.n_members = int(meta['n_members'])
        self.n_layers = int(meta['n_layers'])
        self.K = int(meta['chunk'])
        self.mean = np.asarray(meta['obs_mean'], dtype=np.float64)
        self.std = np.asarray(meta['obs_std'], dtype=np.float64)
        self.layers = [[(weights[f'W{e}_{l}'], weights[f'b{e}_{l}'])
                        for l in range(self.n_layers)]
                       for e in range(self.n_members)]

    @classmethod
    def load(cls, path):
        d = path if os.path.isdir(path) else os.path.dirname(path)
        with open(os.path.join(d, 'student_meta.json')) as f:
            meta = json.load(f)
        weights = dict(np.load(os.path.join(d, 'student.npz')))
        return cls(weights, meta)

    def forward(self, obs):
        """obs (D,) or (N, D) -> (mean, std) each (..., K, 2), in [-1, 1]."""
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / self.std
        outs = []
        for member in self.layers:
            h = x
            for i, (W, b) in enumerate(member):
                h = h @ W + b
                h = np.maximum(h, 0.0) if i < self.n_layers - 1 else np.tanh(h)
            outs.append(h.reshape(h.shape[:-1] + (self.K, 2)))
        outs = np.stack(outs)
        return outs.mean(axis=0), outs.std(axis=0)

    def n_params(self):
        return int(sum(W.size + b.size for m in self.layers for W, b in m))


class StudentPolicy:
    """SB3-style predict() wrapper around StudentEnsemble, aware of the env's
    residual window and authority gate."""

    def __init__(self, path, chunk_mode='speculative', accept_alpha=0.999,
                 support_sigma=None, axes='both'):
        if chunk_mode not in ('speculative', 'every_step'):
            raise ValueError(chunk_mode)
        # axes='steer' zeroes the throttle residual.  ORION's planned speed is
        # anchored to the ego's current speed (it sees it as input), so its
        # longitudinal labels say "hold whatever speed you have" -- in closed
        # loop that has no restoring force and the first round-0 student that
        # acted everywhere slowed to a crawl (0.68 m/s on Town03).
        if axes not in ('both', 'steer', 'throttle'):
            raise ValueError(axes)
        self._mask = {'both': np.array([1.0, 1.0]), 'steer': np.array([0.0, 1.0]),
                      'throttle': np.array([1.0, 0.0])}[axes]
        self.net = StudentEnsemble.load(path)
        self.chunk_mode = chunk_mode
        self.accept_alpha = accept_alpha
        # g_support = clip(1 - mean ensemble std / support_sigma, 0, 1);
        # None leaves the env's support gate at 1.
        self.support_sigma = support_sigma
        self.env = None
        self.total = dict(n_steps=0, n_active=0, n_queries=0)
        self.reset()

    def attach(self, env):
        self.env = env

    def reset(self):
        """Call at every episode start."""
        self._chunk, self._chunk_std, self._idx = None, None, 0
        self.episode = dict(n_steps=0, n_active=0, n_queries=0)

    def _count(self, key):
        self.episode[key] += 1
        self.total[key] += 1

    def _window_open(self):
        return self.env is None or self.env._residual_window_active()

    def _last_step_accepted(self):
        if self.env is None:
            return True
        info = getattr(self.env, 'last_authority_info', None) or {}
        return info.get('alpha_safe', 1.0) >= self.accept_alpha

    def predict(self, obs, deterministic=True):
        self._count('n_steps')
        if not self._window_open():
            self._chunk = None
            return np.zeros(2), None
        self._count('n_active')

        reuse = (self.chunk_mode == 'speculative'
                 and self._chunk is not None
                 and self._idx < self.net.K
                 and self._last_step_accepted())
        if not reuse:
            mean, std = self.net.forward(obs)
            self._chunk, self._chunk_std, self._idx = mean, std, 0
            self._count('n_queries')

        action = self._chunk[self._idx]
        std = self._chunk_std[self._idx]
        self._idx += 1

        if self.support_sigma and self.env is not None:
            self.env.support_gate = float(np.clip(
                1.0 - std.mean() / self.support_sigma, 0.0, 1.0))
        return np.asarray(action, dtype=float) * self._mask, None

    def query_fraction(self, which='episode'):
        c = self.episode if which == 'episode' else self.total
        return c['n_queries'] / max(c['n_steps'], 1)
