#!/usr/bin/env python
"""
Attribute every Bench2Drive DS point lost (100 - DS) to a failure cause and an
OWNER -- the part of the stack that could fix it (2026-11-03):

  STUDENT     the tactical student's decisions (pass / wait / speed cap)
  STUDENT*    decisions the student is kept out of today (junctions, beyond its
              perception scope) -- learnable in principle
  CONTROLLER  the MPCC base (geometry, braking, standstills too close to pass)
  PERCEPTION  sensing (lights misread, pedestrians, no emergency-vehicle class)
  RULES       traffic-rule logic with a correct perception

DS = RC * prod(penalties) (checked exact against eval.json): the route-
incomplete part (100 - RC) goes to the cause of the stop (B2D 'TickRuntime' is
the 200 s game-time limit -> the longest standstill is classified), the
penalty part RC * (1 - P) is split over the infractions by log share.
Each infraction is matched to its trace tick by position.

    python tools/b2d/attribute_losses.py base=results/full/sbase_nolane_full \
        vla=results/full/stu_v5_fix_full priv=results/teachers/stu_privileged_v5_full
(a name ending in 'base' is treated as student-free)
"""
import collections, csv, glob, json, math, os, re, sys

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_subset import ABILITY
JUNC = set(ABILITY['Merging'] + ABILITY['Give_Way'] + ABILITY.get('Traffic_Signs', [])) | {
    'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn',
    'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian',
    'BlockedIntersection', 'VanillaNonSignalizedTurn', 'VanillaSignalizedTurnEncounterGreenLight', 'VanillaSignalizedTurnEncounterRedLight',
    'VanillaNonSignalizedTurnEncounterStopsign', 'SignalizedJunctionLeftTurnEnterFlow', 'NonSignalizedJunctionLeftTurnEnterFlow', 'T_Junction'}
F = dict(collisions_pedestrian=.5, collisions_vehicle=.6, collisions_layout=.65, red_light=.7, stop_infraction=.8,
         scenario_timeouts=.7, yield_emergency_vehicle_infractions=.7)
OWNER = {
    'pass decision: collision during/after pulling out': 'STUDENT',
    'pass: hit the obstacle / static actor while passing': 'CONTROLLER',
    'student slow-down -> rear-ended': 'STUDENT',
    'stuck: student waiting behind blocker': 'STUDENT',
    'lane invasion while passing (B2D penalty)': 'STUDENT',
    'junction: crossing/merging traffic hit us or we hit it': 'STUDENT*',
    'side/cut-in collision off junction': 'STUDENT*',
    'in-lane obstacle not perceived': 'PERCEPTION',
    'pedestrian collision': 'PERCEPTION',
    'emergency vehicle not yielded (no siren class)': 'PERCEPTION',
    'red light run, light misread': 'PERCEPTION',
    'red light run, light read correctly': 'RULES',
    'stop sign': 'RULES',
    'stuck: light/stop hold': 'RULES',
    'in-lane: perceived but braked late': 'CONTROLLER',
    'rear-ended after base braking': 'CONTROLLER',
    'hit while stopped': 'CONTROLLER',
    'layout collision': 'CONTROLLER',
    'stuck: other': 'CONTROLLER',
    'unmatched collision': 'CONTROLLER',
    'timeout: too slow overall': 'CONTROLLER',
    'timeout: wedged after collision': 'CONTROLLER',
    'timeout: light/stop hold': 'RULES',
    'timeout: student waiting behind blocker': 'STUDENT',
    'timeout: stopped too close behind obstacle to pull out': 'CONTROLLER',
    'timeout: stopped, nothing ahead (junction yield / creep)': 'STUDENT*',
    'route incomplete, other': 'CONTROLLER'}
fl = lambda x: float(x) if x not in ('', 'None', 'nan') else math.nan

def trace(run, rid):
    f = glob.glob(f'{run}/traces/{rid}_*.csv')
    if not f: return None
    T = list(csv.DictReader(open(f[0])))
    return T, np.array([fl(t['x']) for t in T]), np.array([fl(t['y']) for t in T])

def at(tr, s):
    m = re.search(r'x=(-?[\d.]+), y=(-?[\d.]+)', s)
    if tr is None or not m: return None
    T, X, Y = tr
    return int(np.nanargmin((X - float(m.group(1))) ** 2 + (Y - float(m.group(2))) ** 2))

def collision(tr, i, s, typ, dv_on):
    T = tr[0]; win = T[max(0, i - 60):i + 1]
    ev = max(fl(t['v']) for t in T[max(0, i - 10):i + 1])
    m = re.search(r'id=(\d+)', s); cid = m.group(1) if m else None
    rows = [t for t in T[max(0, i - 20):i + 1] if t['any_id'] == cid]
    passing = any(fl(t['gate']) > 0.5 and abs(fl(t['d'])) > 1.2 for t in win)
    if 'walker' in s: return 'pedestrian collision'
    if ev < 0.5: return 'hit while stopped'
    if not rows: return 'unmatched collision'
    ds, dd = fl(rows[0]['any_ds']), fl(rows[0]['any_dd'])
    if ds < -2:
        cap = dv_on and any(fl(t['st_dv']) < -0.5 and t['student_net'] == '1' for t in win)
        return 'student slow-down -> rear-ended' if cap else 'rear-ended after base braking'
    if passing:
        pv = fl(rows[0]['any_v'])
        return 'pass decision: collision during/after pulling out' if pv > 2 else 'pass: hit the obstacle / static actor while passing'
    if abs(dd) < 2:
        pre = T[max(0, i - 30):max(1, i - 5)]
        seen = sum(not math.isnan(fl(t['per_ds'])) for t in pre if not math.isnan(fl(t['gt_ds'])))
        gtn = sum(not math.isnan(fl(t['gt_ds'])) for t in pre)
        return 'in-lane obstacle not perceived' if gtn and seen < 0.3 * gtn else 'in-lane: perceived but braked late'
    return 'junction: crossing/merging traffic hit us or we hit it' if typ in JUNC else 'side/cut-in collision off junction'

def stuck(tr, i, dv_on):
    if tr is None or i is None: return 'stuck: other'
    win = tr[0][max(0, i - 100):i + 1]
    if sum(t['light_hold'] == '1' or not math.isnan(fl(t['stop_ds'])) for t in win) > 0.5 * len(win): return 'stuck: light/stop hold'
    if sum(t['student_net'] == '1' and fl(t['st_go']) < 0.5 and fl(t['gate']) < 0.5 for t in win) > 0.5 * len(win) and dv_on is not None:
        return 'stuck: student waiting behind blocker'
    return 'stuck: other'

def timeout(tr, student):
    """B2D 'TickRuntime' = the 200 s game-time limit: classify the longest standstill."""
    if tr is None: return 'timeout: too slow overall'
    T = tr[0]; best, cur, st = (0, 0), 0, 0
    for i, t in enumerate(T):
        if fl(t['v']) < 0.3:
            st = i if cur == 0 else st; cur += 1
            if cur > best[0]: best = (cur, st)
        else: cur = 0
    n, i0 = best
    if n * 0.05 < 20: return 'timeout: too slow overall'
    w = T[i0:i0 + n]
    if sum(t['wedge'] == '1' for t in w) > 0.5 * n: return 'timeout: wedged after collision'
    if sum(t['light_hold'] == '1' or not math.isnan(fl(t['stop_ds'])) for t in w) > 0.5 * n: return 'timeout: light/stop hold'
    if student and sum(t['student_net'] == '1' and fl(t['st_go']) < 0.5 and not math.isnan(fl(t['gt_ds'])) and fl(t['gt_ds']) < 15 for t in w) > 0.5 * n:
        return 'timeout: student waiting behind blocker'
    if sum(not math.isnan(fl(t['gt_ds'])) and fl(t['gt_ds']) < 15 for t in w) > 0.5 * n: return 'timeout: stopped too close behind obstacle to pull out'
    return 'timeout: stopped, nothing ahead (junction yield / creep)'

def run_arm(run, student):
    out = collections.defaultdict(float); routes = collections.defaultdict(set)
    for r in json.load(open(run + '/eval.json'))['_checkpoint']['records']:
        rid, typ, inf = r['route_id'], r['scenario_name'].rsplit('_', 1)[0], r['infractions']
        rc, P = r['scores']['score_route'], r['scores']['score_penalty']
        tr = trace(run, rid)
        items = []                                   # (cause, -log factor)
        for k, f in F.items():
            for s in inf[k]:
                i = at(tr, s)
                if k == 'collisions_vehicle': c = collision(tr, i, s, typ, student) if i is not None else 'unmatched collision'
                elif k == 'collisions_pedestrian': c = 'pedestrian collision'
                elif k == 'collisions_layout': c = 'layout collision'
                elif k == 'red_light':
                    seen = tr is not None and i is not None and any(t['light'] in ('red', 'yellow') for t in tr[0][max(0, i - 60):i + 1])
                    c = 'red light run, light read correctly' if seen else 'red light run, light misread'
                elif k == 'stop_infraction': c = 'stop sign'
                elif k == 'yield_emergency_vehicle_infractions': c = 'emergency vehicle not yielded (no siren class)'
                else: c = 'route incomplete, other'
                items.append((c, -math.log(f)))
        for s in inf['outside_route_lanes']:
            m = re.search(r'([\d.]+)% of the completed', s)
            if m and float(m.group(1)) < 100: items.append(('lane invasion while passing (B2D penalty)', -math.log(1 - float(m.group(1)) / 100)))
        tot = sum(w for _, w in items)
        for c, w in items:
            out[c] += rc * (1 - P) * w / tot; routes[c].add(rid)
        if rc < 99.9:
            if 'blocked' in r['status'] or inf['vehicle_blocked']:
                s = (inf['vehicle_blocked'] or [''])[0]; c = stuck(tr, at(tr, s), student)
            elif 'TickRuntime' in r['status']: c = timeout(tr, student)
            else: c = 'route incomplete, other'
            out[c] += 100 - rc; routes[c].add(rid)
    return {c: v / 220 for c, v in out.items()}, routes

def main():
    arms = {}
    for a in sys.argv[1:]:
        name, path = a.split('=', 1)
        arms[name] = (path, None if name.endswith('base') else True)
    if not arms: sys.exit(__doc__)
    first = list(arms)[min(1, len(arms) - 1)]
    R = {a: run_arm(p, s) for a, (p, s) in arms.items()}
    causes = sorted(OWNER, key=lambda c: (OWNER[c], -R[first][0].get(c, 0)))
    print(f'DS points lost per route (mean over routes); DS = 100 - total\n{"owner":11s}{"cause":56s}' + ''.join(f'{a:>14s}' for a in arms) + f'   routes({first})')
    own = collections.defaultdict(lambda: collections.defaultdict(float))
    for c in causes:
        vals = [R[a][0].get(c, 0) for a in arms]
        for a, v in zip(arms, vals): own[OWNER[c]][a] += v
        if max(vals) >= 0.05: print(f'{OWNER[c]:11s}{c:56s}' + ''.join(f'{v:14.2f}' for v in vals) + f'   {len(R[first][1].get(c, ()))}')
    print('\nby owner')
    for o in own: print(f'{o:11s}' + ''.join(f'{own[o][a]:14.2f}' for a in arms))
    print(f'{"TOTAL":11s}' + ''.join(f'{sum(R[a][0].values()):14.2f}' for a in arms) + '   (= 100 - DS)')


if __name__ == '__main__':
    main()
