#!/usr/bin/env python
"""
Offline compute / energy benchmark of the sensor-only stack on real
Bench2Drive inputs, WITHOUT CARLA on the GPU (clean numbers for the paper).

Inputs: frames dumped by the agent during a run (B2D_DUMP_DIR, every 20th
tick: front image PNG + LiDAR + pose).

    DUMP=1 TAG=dump CONFIG=tools/b2d/config_sbase.json ./tools/b2d/run_b2d.sh smoke
    pkill -f CarlaUE4                                   # free the GPU
    python tools/b2d/bench_stack.py --frames results/b2d/dump_smoke/dump --device cuda
    python tools/b2d/bench_stack.py --frames ... --device cpu
    python tools/b2d/bench_stack.py --frames ... --device cpu \\
        --yolop models/yolop/yolop-640-640-int8.onnx       # tools/b2d/quantize_yolop.py
    [--tactical models/tactical_r0]                        # + the student's forward pass

Per module (each measured in its own block so energy is attributable):
  yolop    YOLOP inference + lane fit, per call (the agent calls it every 2nd tick)
  yolo     traffic-light YOLOv8n on the cropped ROI (the agent: near junctions only)
  lidar    ground removal + clustering + tracking
  student  tactical ensemble forward (if --tactical)
Reports ms/frame (median, p95), CPU-s/frame, GPU J/frame above idle and extra
GPU memory (EnergyMeter), and the per-TICK cost after the agent's schedules
(YOLOP every 2nd tick; YOLO and student duty cycles from a run's traces if
--traces is given).  MPCC solve time comes from the run traces (ms_ocp; CPU).
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))


def load_frames(d, limit):
    pngs = sorted(glob.glob(os.path.join(d, '*.png')))[:limit]
    if not pngs:
        sys.exit(f'no dumped frames in {d} (run with DUMP=1)')
    import cv2
    out = []
    for p in pngs:
        z = np.load(p[:-4] + '.npz')
        out.append(dict(img=cv2.imread(p), lidar=z['lidar'], pose=z['pose']))
    return out


def timed(fn, frames, warmup):
    for f in frames[:warmup]:
        fn(f)
    from efficiency.power import EnergyMeter
    ts = []
    with EnergyMeter() as em:
        for f in frames:
            t0 = time.perf_counter()
            fn(f)
            ts.append(time.perf_counter() - t0)
    r = em.report(len(frames))
    ts = np.array(ts) * 1000
    r.update(ms_median=round(float(np.median(ts)), 2), ms_p95=round(float(np.percentile(ts, 95)), 2))
    return r


def duty_from_traces(run):
    """Fraction of ticks the light detector / student network ran in a real run."""
    import csv
    n = light = net = net_n = 0
    for p in glob.glob(os.path.join(run, 'traces', '*.csv')):
        for r in csv.DictReader(open(p)):
            n += 1
            light += r.get('ms_yolo_net') not in (None, '', '0', '0.0')
            if r.get('student_net') not in (None, ''):
                net_n += 1
                net += r['student_net'] == '1'
    return dict(light=light / max(n, 1), student=(net / net_n) if net_n else None, ticks=n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--yolop', default=os.path.join(ROOT, 'models/yolop/yolop-640-640.onnx'))
    ap.add_argument('--tactical', default=None)
    ap.add_argument('--traces', default=None, help='a run dir: duty cycles of YOLO / student')
    ap.add_argument('--limit', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--out', default=None, help='JSON (default: <frames>/bench_<device>.json)')
    args = ap.parse_args()

    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''           # nothing may touch the GPU
    frames = load_frames(args.frames, args.limit)
    res = dict(device=args.device, frames=len(frames), yolop_model=os.path.basename(args.yolop))

    from perception.lanes import LanePerception
    lanes = LanePerception(args.yolop, every=1, device=args.device)
    res['yolop'] = timed(lambda f: lanes.update(f['img']), frames, args.warmup)
    res['yolop']['providers'] = lanes.sess.get_providers()

    from perception.traffic_lights import TrafficLightDetector
    tl = TrafficLightDetector(device=args.device)
    res['yolo'] = timed(lambda f: tl.detect(f['img']), frames, args.warmup)

    from perception.lidar_obstacles import LidarObstacleTracker
    lid = LidarObstacleTracker()
    res['lidar'] = timed(lambda f: lid.update(f['lidar'], tuple(f['pose'][:3]), float(f['pose'][4])),
                         frames, args.warmup)

    if args.tactical:
        from student.tactical import TacticalEnsemble, FEATURE_NAMES
        net = TacticalEnsemble.load(args.tactical)
        x = np.zeros(len(FEATURE_NAMES))
        res['student'] = timed(lambda f: net.forward(x), frames, args.warmup)
        res['student']['params'] = int(net.meta.get('n_params_total', 0))

    # Per-tick cost after the agent's schedules.
    duty = duty_from_traces(args.traces) if args.traces else dict(light=1.0, student=1.0)
    res['duty'] = duty
    per_tick = 0.5 * res['yolop']['ms_median'] + duty['light'] * res['yolo']['ms_median'] \
        + res['lidar']['ms_median']
    if 'student' in res and duty.get('student') is not None:
        per_tick += duty['student'] * res['student']['ms_median']
    res['perception_ms_per_tick'] = round(per_tick, 2)
    if res['yolop'].get('gpu_j_per_frame') is not None:
        j = 0.5 * res['yolop']['gpu_j_per_frame'] + duty['light'] * res['yolo']['gpu_j_per_frame']
        res['perception_gpu_j_per_tick'] = round(j, 4)

    out = args.out or os.path.join(args.frames, f'bench_{args.device}_{os.path.splitext(res["yolop_model"])[0]}.json')
    json.dump(res, open(out, 'w'), indent=2)
    print(f'\n{len(frames)} frames, device {args.device}, YOLOP {res["yolop_model"]}')
    print(f"{'module':8s} {'ms med':>8s} {'p95':>8s} {'CPU-s':>8s} {'GPU J':>8s} {'GPU MB':>8s}")
    for k in ('yolop', 'yolo', 'lidar', 'student'):
        if k in res:
            r = res[k]
            print(f"{k:8s} {r['ms_median']:8.2f} {r['ms_p95']:8.2f} {r['cpu_s_per_frame']:8.4f} "
                  f"{r.get('gpu_j_per_frame') if r.get('gpu_j_per_frame') is not None else '-':>8} "
                  f"{r.get('gpu_extra_mem_mb') if r.get('gpu_extra_mem_mb') is not None else '-':>8}")
    print(f"duty: light {duty['light']:.2f}, student {duty.get('student')}; "
          f"perception per tick {res['perception_ms_per_tick']} ms"
          + (f", {res['perception_gpu_j_per_tick']} J GPU" if 'perception_gpu_j_per_tick' in res else ''))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
