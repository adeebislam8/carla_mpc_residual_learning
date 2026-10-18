"""
Route geometry shared by the Bench2Drive agent (tools/b2d/mpcc_b2d_agent.py)
and the offline lane benchmark (tools/b2d/lane_bench.py): the sparse
leaderboard plan densified to a 1 m path, and how far the real lane can lie
from that path per node segment (the lane belief's prior uncertainty).
"""

import numpy as np


def densify_route(nodes, spacing=1.0):
    """Sparse route nodes [(x, y)] -> points every `spacing` m along a
    shape-preserving cubic (PCHIP, chord-length parametrised) through them, plus
    for each output point the index of the node segment it lies in.  (A natural
    CubicSpline overshot ~5 m sideways on a straight before a junction: node
    spacing jumps from 50 m to ~14 m there.)"""
    from scipy.interpolate import PchipInterpolator as CubicSpline
    P = np.asarray(nodes, dtype=float)
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 0.5]
    P = P[keep]
    idx_map = np.nonzero(keep)[0]
    if len(P) < 2:
        return P.tolist(), [0] * len(P)
    t = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if len(P) == 2:
        tt = np.arange(0.0, t[-1], spacing)
        pts = P[0] + np.outer(tt / t[-1], P[1] - P[0])
    else:
        cx, cy = CubicSpline(t, P[:, 0]), CubicSpline(t, P[:, 1])
        tt = np.arange(0.0, t[-1], spacing)
        pts = np.stack([cx(tt), cy(tt)], axis=1)
    seg = np.clip(np.searchsorted(t, tt, side='right') - 1, 0, len(P) - 2)
    return pts.tolist(), idx_map[seg].tolist()


def route_sag_bound(nodes, margin=1.0, cap=5.0):
    """Per node segment: how far the real lane can lie from the densified
    route.  A chord of length L between nodes on a road turning by dtheta per
    chord misses the arc by its sagitta ~ L * dtheta / 8 (R60 bend, 45 m nodes:
    4.2 m, as measured).  The perceived lane correction is clipped to this, so
    on a straight a mis-tracked lane cannot drag the car across lanes."""
    P0 = np.asarray(nodes, dtype=float)
    keep = np.r_[True, np.hypot(*np.diff(P0, axis=0).T) > 0.5]     # as densify_route
    P, idx_map = P0[keep], np.nonzero(keep)[0]
    n = len(P)
    if n < 2:
        return [margin] * max(len(P0), 1)
    v = np.diff(P, axis=0)
    L = np.hypot(v[:, 0], v[:, 1])
    h = np.arctan2(v[:, 1], v[:, 0])
    turn = np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)       # at inner nodes
    out = []
    for k in range(n - 1):
        dth = max(turn[k - 1] if k >= 1 else 0.0, turn[k] if k < len(turn) else 0.0)
        out.append(float(min(cap, margin + L[k] * dth / 8.0)))
    out.append(out[-1])
    # back to ORIGINAL node indices (densify_route's seg ids)
    pos = np.clip(np.searchsorted(idx_map, np.arange(len(P0)), side='right') - 1, 0, n - 1)
    return [out[p] for p in pos]
