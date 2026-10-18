"""
Probabilistic lane-centre belief along the route (sensor-only setting).

The leaderboard gives every agent only a SPARSE route (a node every <= 50 m);
between nodes the densified spline is a guess that cuts bends by up to ~4 m.
YOLOP (perception/lanes.py) corrects it where lane lines are visible -- but
the old tracker kept only the last YOLOP window, blended 30/70 with no notion
of confidence, and faded back to the raw spline within 20 m of it.  With no
fit on ~1/3 of ticks (overtaking, junctions, missed lines) the lane centre was
off by ~3 m there, and every decision measured against it went wrong too
(Bench2Drive abl44: GT lanes 73.7 DS vs 53.6, 2026-10-19).

Here the lane centre is a BELIEF: 1 m bins along the route, each with a mean
offset mu from the spline (+ right, metres) and a standard deviation sigma.

  prior     a bin entering the window: mu = 0 (the spline), sigma = how far the
            spline can sag there (route_geometry.route_sag_bound / 2, >= 0.3 m)
  predict   the road is static: mu stays, only our pose drifts -> variance
            grows by q_per_m per metre driven.  Bins are never forgotten while
            in the window: a lane seen once stays known while YOLOP is blind
  measure   a YOLOP fit -> lane-centre offsets at ~1 m spacing, 4-35 m ahead,
            variance growing with distance and doubled for one-line fits
  identity  the WHOLE fit is tested against the belief (median normalised
            innovation): consistent -> used; off by one lane width -> it is the
            ADJACENT lane (we are overtaking) -> shifted back and used;
            otherwise rejected.  The same implausible offset in reset_n fits
            in a row means the belief itself is wrong: those bins are reset to
            the prior and the fit is used
  update    scalar Kalman per bin
  junction  no lane markings: no measurements, the prior (sigma_junction)

offset(s), sigma(s) and window() feed the MPCC (lane reference + corridor) and
the decision layer (gate / lane assignment / student feature, phase B).  Cost:
~100 bins x a few flops per tick.
"""

import math

import numpy as np


def fit_points_on_route(fit, ex, ey, yaw, fc, s_ego, step=1.0, x_lo=4.0, x_hi=35.0):
    """A YOLOP lane-centre fit (vehicle frame: x forward, y right, quadratic
    right(x)) -> its points as route Frenet (s, d) plus their distance ahead x.
    Shared by the agent and the offline benchmark."""
    c0, c1, c2 = fit['coeffs']
    cy, sy = math.cos(yaw), math.sin(yaw)
    ss, dd, xx = [], [], []
    for xv in np.arange(max(fit['x_min'], x_lo), min(fit['x_max'], x_hi), step):
        yv = c0 + c1 * xv + c2 * xv * xv
        wx, wy = ex + cy * xv - sy * yv, ey + sy * xv + cy * yv
        s_q, d_q, _ = fc.world_to_frenet(wx, wy, 0, s_hint=s_ego + xv)
        ss.append(s_q), dd.append(d_q), xx.append(xv)
    return np.array(ss), np.array(dd), np.array(xx)


class LaneBelief:
    def __init__(self, prior_sigma_fn=None, junction_fn=None, bin_m=1.0, back_m=15.0,
                 ahead_m=80.0, q_per_m=0.003, sigma_min=0.15, sigma_prior_min=0.3,
                 sigma_junction=0.9, lane_w=3.5, gate_k=3.0, adj_tol=1.0, reset_n=8,
                 reset_tol=0.7, r_scale=3.0):
        self.prior_sigma_fn = prior_sigma_fn or (lambda s: 1.0)
        self.junction_fn = junction_fn or (lambda s: False)
        self.bin, self.back, self.ahead = bin_m, back_m, ahead_m
        self.q, self.var_min = q_per_m, sigma_min ** 2
        self.sigma_prior_min, self.sigma_junction = sigma_prior_min, sigma_junction
        self.lane_w, self.gate_k, self.adj_tol = lane_w, gate_k, adj_tol
        self.reset_n, self.reset_tol, self.r_scale = reset_n, reset_tol, r_scale
        self.k0 = None                       # bin index of mu[0]
        self.mu = np.zeros(0)
        self.var = np.zeros(0)
        self.s_last = None
        self._rejects = []                   # median innovations of consecutive rejected fits
        self.stats = dict(fits=0, accepted=0, adjacent=0, rejected=0, resets=0)

    # ── bins ──────────────────────────────────────────────────────────────
    def _prior(self, k):
        s = (k + 0.5) * self.bin
        if self.junction_fn(s):
            sig = self.sigma_junction
        else:
            sig = max(self.sigma_prior_min, float(self.prior_sigma_fn(s)))
        return 0.0, sig * sig

    def advance(self, s):
        """Slide the window to [s - back, s + ahead]; new bins get the prior,
        existing ones the pose-drift process noise."""
        kmin = int(math.floor((s - self.back) / self.bin))
        kmax = int(math.floor((s + self.ahead) / self.bin))
        if self.k0 is None or kmin > self.k0 + len(self.mu) or kmax < self.k0:
            pri = [self._prior(k) for k in range(kmin, kmax + 1)]
            self.k0 = kmin
            self.mu = np.array([p[0] for p in pri])
            self.var = np.array([p[1] for p in pri])
        else:
            if self.s_last is not None:
                self.var += self.q * abs(s - self.s_last)
            if kmin > self.k0:                                  # drop bins behind
                self.mu, self.var = self.mu[kmin - self.k0:], self.var[kmin - self.k0:]
                self.k0 = kmin
            k_end = self.k0 + len(self.mu) - 1
            if kmax > k_end:                                    # new bins ahead
                pri = [self._prior(k) for k in range(k_end + 1, kmax + 1)]
                self.mu = np.r_[self.mu, [p[0] for p in pri]]
                self.var = np.r_[self.var, [p[1] for p in pri]]
        self.s_last = s

    def _idx(self, s):
        return int(math.floor(s / self.bin)) - self.k0

    # ── measurement ───────────────────────────────────────────────────────
    def update_fit(self, s_q, d_q, x_q, two_line_frac=1.0):
        """One YOLOP fit: lane-centre offsets d_q at route s_q, x_q metres
        ahead of the car.  Returns 'accepted' | 'adjacent' | 'rejected' |
        'reset' | 'empty'."""
        if self.k0 is None:
            return 'empty'
        s_q, d_q, x_q = (np.asarray(a, float) for a in (s_q, d_q, x_q))
        idx = np.array([self._idx(s) for s in s_q])
        ok = (idx >= 0) & (idx < len(self.mu))
        ok &= np.array([not self.junction_fn(s) for s in s_q], bool) if len(s_q) else ok
        if ok.sum() < 3:
            return 'empty'
        idx, z, x = idx[ok], d_q[ok], x_q[ok]
        r = (0.25 + 0.012 * x) ** 2 * self.r_scale * (1.0 + 2.0 * (1.0 - two_line_frac))
        self.stats['fits'] += 1
        med, med_norm = self._identity_test(idx, z, r)
        verdict = 'accepted'
        if med_norm > self.gate_k:
            if abs(abs(med) - self.lane_w) < self.adj_tol:
                z = z - math.copysign(self.lane_w, med)       # the neighbouring lane
                verdict = 'adjacent'
            else:
                self._rejects.append(med)
                recent = self._rejects[-self.reset_n:]
                if len(recent) >= self.reset_n and max(recent) - min(recent) < 2 * self.reset_tol:
                    for i in idx:                                # the belief was wrong:
                        m0, v0 = self._prior(self.k0 + i)        # back to the prior, as
                        self.mu[i], self.var[i] = m0, max(v0, med * med + 1.0)  # unsure as the miss
                    self._rejects = []
                    self.stats['resets'] += 1
                    verdict = 'reset'
                else:
                    self.stats['rejected'] += 1
                    return 'rejected'
        self._rejects = []
        self.stats['accepted' if verdict != 'adjacent' else 'adjacent'] += 1
        for i, zi, ri in zip(idx, z, r):
            K = self.var[i] / (self.var[i] + ri)
            self.mu[i] += K * (zi - self.mu[i])
            self.var[i] = max((1.0 - K) * self.var[i], self.var_min)
        return verdict

    CONF_SIGMA = 0.6        # bins this sure decide a fit's lane identity
    CONT_M = 25.0           # ... else continuity with a sure bin this far behind the fit

    def _identity_test(self, idx, z, r):
        """(median innovation, median normalised innovation) of a fit, judged
        only where the belief is SURE: on bins with sigma < CONF_SIGMA, or --
        if the fit overlaps fewer than 3 of them -- its first points against
        the nearest sure bin behind it (lane offsets are continuous).  Judged
        against loose prior bins, a fit of the neighbouring lane would look
        plausible and capture the belief (seen in the synthetic overtake test)."""
        sure = self.var[idx] < self.CONF_SIGMA ** 2
        if sure.sum() >= 3:
            nu = z[sure] - self.mu[idx[sure]]
            S = self.var[idx[sure]] + r[sure]
        else:
            behind = np.nonzero(self.var[:idx[0]] < self.CONF_SIGMA ** 2)[0]
            if len(behind) == 0 or (idx[0] - behind[-1]) * self.bin > self.CONT_M:
                nu, S = z - self.mu[idx], self.var[idx] + r      # nothing sure: plain test
            else:
                j = behind[-1]
                gap = (idx[:5] - j) * self.bin
                nu = z[:5] - self.mu[j]
                S = self.var[j] + r[:5] + (0.05 * gap) ** 2   # offsets drift slowly along s
        return float(np.median(nu)), float(np.median(np.abs(nu) / np.sqrt(S)))

    def reset_range(self, s_from, s_to):
        """Back to the prior over [s_from, s_to] (a route lane-change segment)."""
        if self.k0 is None:
            return
        for i in range(max(self._idx(s_from), 0), min(self._idx(s_to) + 1, len(self.mu))):
            self.mu[i], self.var[i] = self._prior(self.k0 + i)

    # ── queries ───────────────────────────────────────────────────────────
    def window(self):
        """(s at bin centres, mu, sigma) over the whole window."""
        if self.k0 is None:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        s = (np.arange(len(self.mu)) + self.k0 + 0.5) * self.bin
        return s, self.mu.copy(), np.sqrt(self.var)

    def offset(self, s):
        s_b, mu, _ = self.window()
        return float(np.interp(s, s_b, mu)) if len(s_b) else 0.0

    def sigma(self, s):
        s_b, _, sig = self.window()
        return float(np.interp(s, s_b, sig)) if len(s_b) else float('inf')
