#!/usr/bin/env python
"""
Per-module compute of the MPCC agent, from the ms_* columns of the traces
(the efficiency table of the paper).

    python tools/b2d/timing_b2d.py results/b2d/sbase_full [results/b2d/srules_full ...]

Columns (wall time per 20 Hz tick, ms; median / p95 / mean over all ticks of
all routes, the first 40 ticks of each route dropped as warm-up):
  ego        GNSS/IMU pose estimate
  lane       YOLOP lane perception incl. fit (network runs every 2nd tick)
  yolop_net  YOLOP ONNX inference alone (ticks it ran)
  lidar      LiDAR ground removal + clustering + tracking + corridor filter
  light      traffic-light / stop-sign step incl. YOLO
  yolo_net   YOLOv8n inference alone
  tactical   distilled student (scene features + ensemble), if enabled
  mpc        control step: obstacle prep + MPCC + safety layer
  ocp        acados SQP-RTI solve alone
  total      the agent's whole run_step (sensor data already decoded)
Duty cycles: share of ticks the light detector ran (light_gate_m), the
student's network ran (no actor near = skipped), the OCP solution was reused
(mpc_reuse), and the deadlock creep was active.
Reference: ORION (7B VLA) measured 451 ms/frame median on the same A5000.
"""

import csv
import glob
import os
import sys

import numpy as np

WARMUP = 40
ORDER = ['ego', 'lane', 'yolop_net', 'lidar', 'light', 'yolo_net', 'tactical', 'mpc', 'ocp', 'total']


def load(run):
    vals, duty = {}, dict(ticks=0, light=0, student_n=0, student=0, reused=0, creep=0)
    files = sorted(glob.glob(os.path.join(run, 'traces', '*.csv')))
    for p in files:
        rows = list(csv.DictReader(open(p)))[WARMUP:]
        for r in rows:
            duty['ticks'] += 1
            duty['light'] += r.get('ms_yolo_net') not in (None, '')
            if r.get('student_net') not in (None, ''):
                duty['student_n'] += 1
                duty['student'] += r['student_net'] == '1'
            duty['reused'] += r.get('mpc_reused') == '1'
            duty['creep'] += r.get('creep') == '1'
            for k, v in r.items():
                if k.startswith('ms_') and v not in ('', None):
                    vals.setdefault(k[3:], []).append(float(v))
    return vals, len(files), duty


def main():
    runs = sys.argv[1:] or ['results/b2d/sbase_full']
    for run in runs:
        vals, n, duty = load(run)
        if not vals:
            print(f'{run}: no ms_* columns (traces from before timing was added?)')
            continue
        print(f'\n{run}  ({n} routes)')
        print(f"{'module':10s} {'median':>8s} {'p95':>8s} {'mean':>8s} {'ticks':>8s}")
        for k in ORDER + sorted(set(vals) - set(ORDER)):
            if k not in vals:
                continue
            a = np.array(vals[k])
            if k.endswith('_net'):
                a = a[a > 0]                  # only the ticks the network ran
                if not len(a):
                    continue
            print(f'{k:10s} {np.median(a):8.2f} {np.percentile(a, 95):8.2f} {a.mean():8.2f} {len(a):8d}')
        t = max(duty['ticks'], 1)
        print(f"duty cycle: light detector {100 * duty['light'] / t:.0f}% of ticks; "
              + (f"student network {100 * duty['student'] / duty['student_n']:.0f}% of ticks; "
                 if duty['student_n'] else '')
              + f"OCP reused {100 * duty['reused'] / t:.0f}%; creeping {100 * duty['creep'] / t:.1f}%")
        if 'total' in vals:
            med = np.median(vals['total'])
            print(f'total median {med:.1f} ms/tick -> {1000 / max(med, 1e-6):.0f} Hz; '
                  f'ORION 451 ms -> {451 / max(med, 1e-6):.0f}x')


if __name__ == '__main__':
    main()
