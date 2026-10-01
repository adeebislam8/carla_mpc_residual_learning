#!/usr/bin/env python3
"""
Dose-response readout for tools/sweep_overtake.sh.

Success rate is the WRONG primary metric here and is reported last: removing the
lateral attractor removes overtaking, so episodes convert into timeouts and
success falls by construction.  The metrics that can actually discriminate are
the ones attributable to driving left into something:

    collisions / km            exposure-normalised, so route length cannot
                               manufacture the effect (which is how the earlier
                               "shorter routes are safer" result went wrong --
                               per-km collisions were 4.13 vs 1.46 the OTHER way)
    static-object share        the attractor's victims are poles, lights,
                               guardrails and fences -- things the CBF has no
                               term for
    left-impact share          the attractor pulls LEFT; if it is the cause,
                               the left/right asymmetry must shrink with it
    straight-road share        overtakes happen on straights

A monotone trend across three dose levels is much harder to get by chance than a
single contrast, so the ordering matters more than any one p-value.

    python tools/analyze_overtake.py results/ovt_*.json
"""
import json
import sys
from collections import Counter
from math import erf, sqrt

STATIC_PREFIXES = ('static.', 'traffic.')     # traffic lights/signs are furniture


def z2p(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def prop_test(k1, n1, k2, n2):
    if min(n1, n2) == 0:
        return float('nan')
    p1, p2, p = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return z2p((p1 - p2) / se) if se > 0 else 1.0


def load(path):
    res = json.load(open(path))
    eps = res.get('per_episode', [])
    if not eps:
        return None
    colls = [e for e in eps if e.get('outcome') == 'collision']
    km = sum(e['progress_m'] for e in eps) / 1000.0
    others = [e.get('collision_other') or '?' for e in colls]
    sides = [e.get('collision_kind') for e in colls if e.get('collision_kind')]
    ks = [e['collision_kappa'] for e in colls
          if e.get('collision_kappa') is not None
          and e['collision_kappa'] == e['collision_kappa']]
    n = len(eps)
    return {
        'label': res.get('label', path),
        # None means the model default -2.5, not "no attractor"
        'n_ovt': -2.5 if res.get('n_overtake') is None else res['n_overtake'],
        'n': n, 'n_coll': len(colls), 'km': km,
        'coll_per_km': len(colls) / km if km else float('nan'),
        'ovt_per_km': sum(e['overtakes'] for e in eps) / km if km else float('nan'),
        'ovt_per_ep': sum(e['overtakes'] for e in eps) / n,
        'success': sum(1 for e in eps if e.get('outcome') == 'success') / n,
        'timeout': sum(1 for e in eps if e.get('outcome') == 'timeout') / n,
        'static_k': sum(1 for o in others if o.startswith(STATIC_PREFIXES)),
        'vehicle_k': sum(1 for o in others if o.startswith('vehicle')),
        'left_k': sum(1 for s in sides if s == 'left'),
        'right_k': sum(1 for s in sides if s == 'right'),
        'n_sides': len(sides),
        'straight_k': sum(1 for k in ks if abs(k) <= 0.02),
        'n_kappa': len(ks),
        'top': Counter(others).most_common(5),
    }


def main(paths):
    cells = [c for c in (load(p) for p in paths) if c]
    if len(cells) < 2:
        print("need at least 2 cells")
        return 1
    cells.sort(key=lambda c: -c['n_ovt'])        # 0.0 first, -2.5 last

    print("=" * 78)
    print("OVERTAKE ATTRACTOR DOSE-RESPONSE")
    print("=" * 78)
    print("\n  PRIMARY -- attributable to driving left into something")
    print(f"  {'n_overtake':>11}{'n':>5}{'coll/km':>9}{'static%':>9}{'left%':>8}"
          f"{'right%':>8}{'strght%':>9}{'ovt/km':>8}")
    for c in cells:
        sp = 100 * c['static_k'] / c['n_coll'] if c['n_coll'] else float('nan')
        lp = 100 * c['left_k'] / c['n_sides'] if c['n_sides'] else float('nan')
        rp = 100 * c['right_k'] / c['n_sides'] if c['n_sides'] else float('nan')
        st = 100 * c['straight_k'] / c['n_kappa'] if c['n_kappa'] else float('nan')
        print(f"  {c['n_ovt']:>11.2f}{c['n']:>5}{c['coll_per_km']:>9.2f}{sp:>9.1f}"
              f"{lp:>8.1f}{rp:>8.1f}{st:>9.1f}{c['ovt_per_km']:>8.2f}")

    print("\n  SECONDARY -- expected to get WORSE; not evidence against")
    print(f"  {'n_overtake':>11}{'success%':>10}{'timeout%':>10}{'ovt/ep':>9}")
    for c in cells:
        print(f"  {c['n_ovt']:>11.2f}{100*c['success']:>10.2f}"
              f"{100*c['timeout']:>10.2f}{c['ovt_per_ep']:>9.2f}")

    base = min(cells, key=lambda c: c['n_ovt'])       # -2.5
    off = max(cells, key=lambda c: c['n_ovt'])        # 0.0
    print(f"\n  vs baseline (n_overtake = {base['n_ovt']:.2f}):")
    for c in cells:
        if c is base:
            continue
        p_st = prop_test(base['static_k'], base['n_coll'], c['static_k'], c['n_coll'])
        p_lr = prop_test(base['left_k'], base['n_sides'], c['left_k'], c['n_sides'])
        print(f"    n={c['n_ovt']:>5.2f}  static-share p={p_st:.4f} {star(p_st):<4}"
              f"  left-share p={p_lr:.4f} {star(p_lr)}")

    print("\n  monotonicity (3+ cells only -- ordering beats any single p-value):")
    for key, lab, want in (('coll_per_km', 'collisions/km', 'falls as |n| falls'),
                           ('ovt_per_km', 'overtakes/km', 'falls as |n| falls')):
        vals = [c[key] for c in cells]
        mono = all(a <= b for a, b in zip(vals, vals[1:]))
        print(f"    {lab:16} {[f'{v:.2f}' for v in vals]}  "
              f"{'MONOTONE' if mono else 'not monotone'}  ({want})")

    print("\n  what each cell hits most:")
    for c in cells:
        top = "  ".join(f"{o}={k}" for o, k in c['top'])
        print(f"    n={c['n_ovt']:>5.2f}  {top}")

    print("\n" + "=" * 78)
    if off['n_ovt'] == 0.0 and off['n_coll'] and base['n_coll']:
        sb = off['static_k'] / off['n_coll']
        s0 = base['static_k'] / base['n_coll']
        if sb < s0:
            print("  n_overtake=0 cut the static-object share: hypothesis SURVIVES.")
        else:
            print("  n_overtake=0 did NOT cut the static-object share.")
            print("  HYPOTHESIS FALSIFIED -- drop it, as with dynamic infeasibility")
            print("  and corridor geometry before it.")
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
