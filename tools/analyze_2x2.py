#!/usr/bin/env python3
"""
Factorial readout for tools/sweep_2x2.sh.

The point of the 2x2 is the INTERACTION, so four pairwise tests against the
baseline are the wrong summary -- they would miss exactly the effect the design
exists to find.  This reports, for each outcome:

    main effect of alat slack   = mean(alat on)  - mean(alat off)
    main effect of margin       = mean(margin 1.5) - mean(margin 4.0)
    interaction                 = (D - C) - (B - A)

A large interaction is the hypothesis being tested: narrowing the corridor
should only help once the plan inside it is executable.  If the interaction is
flat and both main effects are flat, the infeasibility story is dead.

The decisive column is not the collision rate but its SPLIT -- outside the
corridor (tracking failure) versus inside it (the corridor contains the
furniture).  A change that only moves collisions between those two buckets has
not fixed anything.

    python tools/analyze_2x2.py results/x22_*.json
"""
import json
import sys
from math import erf, sqrt


def z2p(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def load(path):
    with open(path) as f:
        res = json.load(f)
    eps = res.get('per_episode', [])
    colls = [e for e in eps if e.get('outcome') == 'collision']
    n = len(eps)
    if n == 0:
        return None

    def frac(pred):
        return sum(1 for e in eps if pred(e)) / n

    kappas = [e['collision_kappa'] for e in colls
              if e.get('collision_kappa') is not None
              and e['collision_kappa'] == e['collision_kappa']]
    spds = [e['collision_speed'] for e in colls if e.get('collision_speed') is not None]
    return {
        'label': res.get('label', path),
        'alat': res.get('alat_slack'),
        'margin': res.get('junction_margin', 4.0),
        'n': n,
        'succ_k': sum(1 for e in eps if e.get('outcome') == 'success'),
        'coll_k': len(colls),
        'timeout': frac(lambda e: e.get('outcome') == 'timeout'),
        # the split that actually decides the hypothesis
        'out_k': sum(1 for e in colls if e.get('collision_off_corridor')),
        'in_k': sum(1 for e in colls if not e.get('collision_off_corridor')),
        'mean_speed': sum(e['mean_speed'] for e in eps) / n,
        'impact_speed': (sum(spds) / len(spds)) if spds else float('nan'),
        'n_kappa': len(kappas),
        'curved_frac': (sum(1 for k in kappas if abs(k) > 0.01) / len(kappas)) if kappas else float('nan'),
    }


def contrast(cells, key_k, key_n='n'):
    """Return cell rates keyed by (alat_on, margin_narrow)."""
    out = {}
    for c in cells:
        out[(c['alat'] is not None, c['margin'] < 3.0)] = (c[key_k], c[key_n])
    return out


def factorial(cells, key_k, label, lower_better):
    g = contrast(cells, key_k)
    need = [(False, False), (True, False), (False, True), (True, True)]
    if not all(k in g for k in need):
        print(f"  {label}: incomplete 2x2 (have {sorted(g)}) -- skipping")
        return
    (ka, na), (kb, nb) = g[(False, False)], g[(True, False)]
    (kc, nc), (kd, nd) = g[(False, True)], g[(True, True)]
    A, B, C, D = ka / na, kb / nb, kc / nc, kd / nd

    # SEs from the binomial of each cell; contrasts are independent samples.
    def se(p, n):
        return p * (1 - p) / n

    me_alat = ((B + D) - (A + C)) / 2
    se_alat = sqrt((se(A, na) + se(B, nb) + se(C, nc) + se(D, nd))) / 2
    me_marg = ((C + D) - (A + B)) / 2
    se_marg = se_alat
    inter = (D - C) - (B - A)
    se_int = sqrt(se(A, na) + se(B, nb) + se(C, nc) + se(D, nd))

    arrow = "lower better" if lower_better else "higher better"
    print(f"\n  {label}   ({arrow})")
    print(f"        {'':16}{'margin 4.0':>13}{'margin 1.5':>13}")
    print(f"        {'alat 1e-3':16}{100*A:12.2f}%{100*C:12.2f}%")
    print(f"        {'alat 1e3':16}{100*B:12.2f}%{100*D:12.2f}%")
    for nm, eff, s in (("main effect: alat  ", me_alat, se_alat),
                       ("main effect: margin", me_marg, se_marg),
                       ("INTERACTION        ", inter, se_int)):
        z = eff / s if s > 0 else 0.0
        print(f"        {nm}  {100*eff:+7.2f}pp   z={z:+5.2f}  p={z2p(z):7.4f}  {star(z2p(z))}")


def main(paths):
    cells = [c for c in (load(p) for p in paths) if c]
    if len(cells) < 4:
        print(f"need all 4 cells, got {len(cells)}")
        return 1
    cells.sort(key=lambda c: (c['margin'] < 3.0, c['alat'] is not None))

    print("=" * 78)
    print("2x2 FACTORIAL:  lateral-accel slack  x  junction corridor margin")
    print("=" * 78)
    print(f"\n  {'cell':22}{'alat':>8}{'margin':>8}{'n':>6}{'succ%':>8}{'coll%':>8}"
          f"{'out%':>8}{'in%':>7}{'v':>7}{'v_imp':>7}")
    for c in cells:
        a = '1e-3' if c['alat'] is None else f"{c['alat']:g}"
        print(f"  {c['label']:22}{a:>8}{c['margin']:8.1f}{c['n']:6d}"
              f"{100*c['succ_k']/c['n']:8.2f}{100*c['coll_k']/c['n']:8.2f}"
              f"{100*c['out_k']/c['n']:8.2f}{100*c['in_k']/c['n']:7.2f}"
              f"{c['mean_speed']:7.2f}{c['impact_speed']:7.2f}")

    for key, lab, lower in (('succ_k', 'SUCCESS RATE', False),
                            ('coll_k', 'COLLISION RATE', True),
                            ('out_k', 'COLLISIONS OUTSIDE CORRIDOR  (tracking failure)', True),
                            ('in_k', 'COLLISIONS INSIDE CORRIDOR  (corridor holds furniture)', True)):
        factorial(cells, key, lab, lower)

    print("\n" + "=" * 78)
    print("PREDICTION CHECKS  (stated before the run)")
    print("=" * 78)
    by = {(c['alat'] is not None, c['margin'] < 3.0): c for c in cells}
    A, B, C, D = by[(False, False)], by[(True, False)], by[(False, True)], by[(True, True)]

    def rate(c, k):
        return 100 * c[k] / c['n']
    checks = [
        ("alat alone lowers mean speed",
         B['mean_speed'] < A['mean_speed'],
         f"{A['mean_speed']:.2f} -> {B['mean_speed']:.2f} m/s"),
        ("alat alone lowers impact speed",
         B['impact_speed'] < A['impact_speed'],
         f"{A['impact_speed']:.2f} -> {B['impact_speed']:.2f} m/s"),
        ("alat alone cuts OUTSIDE-corridor collisions",
         rate(B, 'out_k') < rate(A, 'out_k'),
         f"{rate(A,'out_k'):.2f}% -> {rate(B,'out_k'):.2f}%"),
        ("alat alone raises timeouts",
         B['timeout'] > A['timeout'],
         f"{100*A['timeout']:.2f}% -> {100*B['timeout']:.2f}%"),
        ("margin alone reproduces the regression",
         rate(C, 'coll_k') > rate(A, 'coll_k'),
         f"{rate(A,'coll_k'):.2f}% -> {rate(C,'coll_k'):.2f}%"),
        ("BOTH cuts INSIDE-corridor collisions",
         rate(D, 'in_k') < rate(A, 'in_k'),
         f"{rate(A,'in_k'):.2f}% -> {rate(D,'in_k'):.2f}%"),
        ("BOTH keeps OUTSIDE-corridor down too",
         rate(D, 'out_k') < rate(A, 'out_k'),
         f"{rate(A,'out_k'):.2f}% -> {rate(D,'out_k'):.2f}%"),
        ("BOTH is the best cell on collisions",
         rate(D, 'coll_k') == min(rate(x, 'coll_k') for x in (A, B, C, D)),
         "  ".join(f"{x['label'].split('_')[1]}={rate(x,'coll_k'):.1f}%" for x in (A, B, C, D))),
    ]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:46} {detail}")

    if all(c['n_kappa'] == 0 for c in cells):
        print("\n  NOTE: no collision_kappa recorded -- these runs predate the "
              "whitelist fix in benchmark_mpcc.py.")
    else:
        print(f"\n  curvature: fraction of collisions at |kappa| > 0.01")
        for c in cells:
            print(f"    {c['label']:22} {100*c['curved_frac']:6.1f}%  (n={c['n_kappa']})")
    print("\n  If both main effects AND the interaction are flat, the "
          "infeasibility hypothesis is dead -- stop pursuing it.")
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
