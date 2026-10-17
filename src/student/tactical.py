"""
Tactical student: the decisions the MPCC cannot make, distilled from ORION.

Bench2Drive (ot_eb, 2026-10-17) left the MPCC+CBF -- even with hand-coded
behaviour rules -- at ~49% Overtaking against ORION's 71%, and every remaining
failure was a JUDGMENT: pulling into the oncoming lane without a gap, not
telling an accident from a queue, not slowing for a hazard about to appear.
So the student does not nudge throttle/steer.  It outputs two decisions the
MPCC executes:

  go     P(pass the in-lane blocker now)  -> scales the MPCC's overtake gate;
         below 0.5 the blocker is treated as a lead to stop behind (wait)
  slow   P(slow down now: anticipated hazard) -> a speed cap for the MPCC

The CBF and the safety gate stay in the loop; 'go' only ALLOWS a pass.

One numeric SCENE per tick -- built by the Bench2Drive agent online and saved
in the recorder's meta.jsonl offline -- feeds both training and deployment,
so the two can never see different features:

  scene = {v, d, gate_rule, lead_ds, lead_v, stop_ds, junction_ds,
           actors: [{type, ds, d, vs, vd, stopped_for}, ...],  # Frenet, ego-relative s
           route_fl: [[fwd, left], ...]}                        # route centre ahead, ego frame

Labels come from ORION's planned path (plan_fl: 6 waypoints, 0.5 s apart,
(forward, left) in the ego frame) compared with route_fl: does the plan pull
into the passing lane (go) or stay behind (wait)?  does its planned speed
drop below ours more than usual at this speed (slow)?
"""

import json
import os
from collections import deque

import numpy as np

FAR = 120.0
HIST_LAGS = (0, 10, 20)        # ticks: now, 0.5 s ago, 1.0 s ago (20 Hz)

# Lateral bands (Frenet d, + is right; the passing lane is on the left).
LANE_D = 2.0
PASS_D = (-7.5, -1.8)
RIGHT_D = (1.5, 4.5)
EMERGENCY = ('ambulance', 'firetruck', 'police')

FRAME_NAMES = [
    'v', 'd', 'gate_rule', 'lead_ds', 'lead_v', 'stop_ds', 'junction_ds',
    'blk_present', 'blk_ds', 'blk_vs', 'blk_stopped', 'blk_extent', 'blk_count',
    'onc_present', 'onc_ds', 'onc_vs', 'onc_tmeet',
    'pbehind_present', 'pbehind_ds', 'pbehind_vs',
    'pahead_present', 'pahead_ds', 'pahead_vs',
    'pstatic_present', 'pstatic_ds',
    'right_present', 'right_ds',
    'cross_present', 'cross_ds', 'cross_t',
    'walker_present', 'walker_ds', 'walker_absd',
    'emerg_present', 'emerg_ds',
]
N_FRAME = len(FRAME_NAMES)
FEATURE_NAMES = [f'{n}@{lag}' for lag in HIST_LAGS for n in FRAME_NAMES]


def _nearest(items, key):
    return min(items, key=key) if items else None


def blocker_info(sc):
    """The in-lane obstacle a pass would be about: nearest one ahead in our lane
    that is stationary or slower than 5 m/s, plus how far the blocked stretch
    extends (consecutive in-lane slow objects < 12 m apart)."""
    acts = sc['actors']
    inlane = sorted([a for a in acts if abs(a['d']) < LANE_D and 0.0 < a['ds'] < 60.0
                     and a['vs'] < 5.0 and not a['type'].startswith('walker')],
                    key=lambda a: a['ds'])
    if not inlane:
        return None
    end, count = inlane[0]['ds'], 1
    for a in inlane[1:]:
        if a['ds'] - end < 12.0:
            end, count = a['ds'], count + 1
        else:
            break
    b = inlane[0]
    return dict(ds=b['ds'], vs=b['vs'], stopped=min(b.get('stopped_for', 99.0), 10.0),
                extent=end - b['ds'], count=count)


def frame_features(sc):
    v, d = float(sc['v']), float(sc['d'])
    acts = sc['actors']
    f = dict(v=v, d=d, gate_rule=float(sc.get('gate_rule', 0.0)),
             lead_ds=min(sc.get('lead_ds') if sc.get('lead_ds') is not None else FAR, FAR),
             lead_v=sc.get('lead_v') or 0.0,
             stop_ds=min(sc.get('stop_ds') if sc.get('stop_ds') is not None else FAR, FAR),
             junction_ds=min(sc.get('junction_ds') if sc.get('junction_ds') is not None else FAR, FAR))
    b = blocker_info(sc)
    f.update(blk_present=float(b is not None), blk_ds=b['ds'] if b else FAR,
             blk_vs=b['vs'] if b else 0.0, blk_stopped=b['stopped'] if b else 0.0,
             blk_extent=b['extent'] if b else 0.0, blk_count=b['count'] if b else 0.0)
    veh = [a for a in acts if a['type'].startswith('vehicle')]
    lane = [a for a in veh if PASS_D[0] < a['d'] < PASS_D[1]]
    onc = _nearest([a for a in lane if a['vs'] < -0.5 and 0.0 < a['ds'] < FAR],
                   key=lambda a: a['ds'])
    f.update(onc_present=float(onc is not None), onc_ds=onc['ds'] if onc else FAR,
             onc_vs=onc['vs'] if onc else 0.0,
             onc_tmeet=min(onc['ds'] / max(v - onc['vs'], 0.5), 20.0) if onc else 20.0)
    pb = _nearest([a for a in lane if a['vs'] > 0.5 and -40.0 < a['ds'] <= 0.0],
                  key=lambda a: -a['ds'])
    pa = _nearest([a for a in lane if a['vs'] > 0.5 and 0.0 < a['ds'] < 40.0],
                  key=lambda a: a['ds'])
    ps = _nearest([a for a in acts if PASS_D[0] < a['d'] < PASS_D[1] and abs(a['vs']) < 0.5
                   and 0.0 < a['ds'] < 60.0], key=lambda a: a['ds'])
    f.update(pbehind_present=float(pb is not None), pbehind_ds=pb['ds'] if pb else -40.0,
             pbehind_vs=pb['vs'] if pb else 0.0,
             pahead_present=float(pa is not None), pahead_ds=pa['ds'] if pa else 40.0,
             pahead_vs=pa['vs'] if pa else 0.0,
             pstatic_present=float(ps is not None), pstatic_ds=ps['ds'] if ps else 60.0)
    rt = _nearest([a for a in acts if RIGHT_D[0] < a['d'] < RIGHT_D[1] and 0.0 < a['ds'] < 30.0],
                  key=lambda a: a['ds'])
    f.update(right_present=float(rt is not None), right_ds=rt['ds'] if rt else 30.0)
    cross = []
    for a in acts:
        rel = a['d'] - d
        if abs(a['vd']) > 0.8 and rel * a['vd'] < 0 and 0.0 < a['ds'] < 40.0:
            cross.append((max(abs(rel) - LANE_D, 0.0) / abs(a['vd']), a))
    c = min(cross, key=lambda p: p[0]) if cross else None
    f.update(cross_present=float(c is not None and c[0] < 6.0),
             cross_ds=c[1]['ds'] if c and c[0] < 6.0 else 40.0,
             cross_t=min(c[0], 6.0) if c else 6.0)
    w = _nearest([a for a in acts if a['type'].startswith('walker') and 0.0 < a['ds'] < 40.0],
                 key=lambda a: a['ds'])
    f.update(walker_present=float(w is not None), walker_ds=w['ds'] if w else 40.0,
             walker_absd=abs(w['d']) if w else 10.0)
    em = _nearest([a for a in veh if any(k in a['type'] for k in EMERGENCY)
                   and -20.0 < a['ds'] < 60.0], key=lambda a: abs(a['ds']))
    f.update(emerg_present=float(em is not None), emerg_ds=em['ds'] if em else 60.0)
    return np.array([float(f[n]) for n in FRAME_NAMES], dtype=np.float32)


def stack_history(frames, i):
    """Features at index i from per-frame vectors: now and HIST_LAGS ago
    (clamped at the episode start)."""
    return np.concatenate([frames[max(i - lag, 0)] for lag in HIST_LAGS])


# ── labels from ORION's plan ────────────────────────────────────────────────

def plan_lateral_offsets(plan_fl, route_fl):
    """Signed lateral offset (left +) of each plan waypoint from the route
    centreline, both in the ego frame: plan left - route left at the same
    forward distance."""
    r = np.asarray(route_fl, dtype=float)
    if len(r) < 2:
        return np.asarray(plan_fl)[:, 1]
    order = np.argsort(r[:, 0])
    rf, rl = r[order, 0], r[order, 1]
    p = np.asarray(plan_fl, dtype=float)
    return p[:, 1] - np.interp(p[:, 0], rf, rl)


def plan_speeds(plan_fl):
    """Planned speed between consecutive 0.5 s waypoints (first from the ego)."""
    p = np.vstack([[0.0, 0.0], np.asarray(plan_fl, dtype=float)])
    return np.linalg.norm(np.diff(p, axis=0), axis=1) / 0.5


def go_label(sc, plan_fl, go_m=1.5, wait_m=0.5):
    """1 = ORION pulls out to pass, 0 = it stays behind, -1 = not a decision
    frame or ambiguous.  A decision frame: we are in our lane and a slow /
    stationary in-lane blocker is 8-40 m ahead."""
    b = blocker_info(sc)
    if b is None or not (8.0 < b['ds'] < 40.0) or abs(sc['d']) > 1.0:
        return -1
    # Passing is to the LEFT (d < 0, plan left > 0): offsets of the 1.5-3 s points.
    off = plan_lateral_offsets(plan_fl, sc['route_fl'])[2:]
    m = float(np.max(off))
    if m > go_m:
        return 1
    if m < wait_m:
        return 0
    return -1


def plan_dv(sc, plan_fl):
    """Planned speed at ~1-1.5 s minus our speed (ORION's raw slow-down signal;
    biased by its speed inertia -- debiased per speed bin in build_tactical)."""
    vp = plan_speeds(plan_fl)
    return float(np.mean(vp[1:3]) - sc['v'])


# ── deployment ─────────────────────────────────────────────────────────────

class TacticalEnsemble:
    def __init__(self, weights, meta):
        self.meta = meta
        self.mean = np.asarray(meta['x_mean'])
        self.std = np.asarray(meta['x_std'])
        self.n_members, self.n_layers = int(meta['n_members']), int(meta['n_layers'])
        self.layers = [[(weights[f'W{e}_{l}'], weights[f'b{e}_{l}'])
                        for l in range(self.n_layers)] for e in range(self.n_members)]

    @classmethod
    def load(cls, d):
        meta = json.load(open(os.path.join(d, 'tactical_meta.json')))
        return cls(dict(np.load(os.path.join(d, 'tactical.npz'))), meta)

    def forward(self, x):
        """x (D,) -> (p_go, p_slow, std_go, std_slow)."""
        z = np.clip((np.asarray(x, float) - self.mean) / self.std, -5.0, 5.0)
        outs = []
        for member in self.layers:
            h = z
            for i, (W, b) in enumerate(member):
                h = h @ W + b
                if i < self.n_layers - 1:
                    h = np.maximum(h, 0.0)
            outs.append(1.0 / (1.0 + np.exp(-h)))
        o = np.stack(outs)
        return o[:, 0].mean(), o[:, 1].mean(), o[:, 0].std(), o[:, 1].std()


class TacticalPolicy:
    """Per-tick decisions with a short hold, so 'go' / 'wait' does not flicker
    between ticks (a pass half-started and aborted is the failure we are
    fixing)."""

    HOLD_TICKS = 10          # keep a decision >= 0.5 s
    CONTEXT_M = 60.0         # the network runs only if some actor is this close ahead

    def __init__(self, path, go_threshold=0.5, slow_threshold=0.5):
        self.net = TacticalEnsemble.load(path)
        self.go_threshold, self.slow_threshold = go_threshold, slow_threshold
        self.frames = deque(maxlen=max(HIST_LAGS) + 1)
        self.go, self.slow, self._held = True, False, 0
        self.last = {}
        self.n_queries = 0
        self.n_net = 0           # ticks the network actually ran (efficiency)

    def reset(self):
        self.frames.clear()
        self.go, self.slow, self._held = True, False, 0

    @classmethod
    def has_context(cls, scene):
        """Anything to decide about: an actor ahead within CONTEXT_M (or just
        behind, in the passing lane).  With an empty road both heads' answers
        are fixed (go = no blocker to wait for, slow = no hazard), so the
        network is skipped -- the learned layer runs only when it matters."""
        return any(-20.0 < a['ds'] < cls.CONTEXT_M for a in scene['actors'])

    def step(self, scene):
        self.frames.append(frame_features(scene))       # history stays complete
        self.n_queries += 1
        if not self.has_context(scene):
            self.go, self.slow, self._held = True, False, self.HOLD_TICKS
            self.last = dict(p_go=1.0, p_slow=0.0, std_go=0.0, std_slow=0.0, net=False)
            return self.go, self.slow
        fr = list(self.frames)
        x = np.concatenate([fr[max(len(fr) - 1 - lag, 0)] for lag in HIST_LAGS])
        p_go, p_slow, s_go, s_slow = self.net.forward(x)
        self.n_net += 1
        self._held += 1
        if self._held >= self.HOLD_TICKS:
            go, slow = p_go >= self.go_threshold, p_slow >= self.slow_threshold
            if go != self.go or slow != self.slow:
                self.go, self.slow, self._held = go, slow, 0
        self.last = dict(p_go=float(p_go), p_slow=float(p_slow),
                         std_go=float(s_go), std_slow=float(s_slow), net=True)
        return self.go, self.slow
