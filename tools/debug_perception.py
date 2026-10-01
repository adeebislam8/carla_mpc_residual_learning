#!/usr/bin/env python
"""
Visual + numeric sanity check for the pretrained-YOLO obstacle pipeline
(carlaEnv.py's use_perception path) -- NOT a benchmark.

Runs one episode with --perception on, drives the nominal MPCC (zero
residual) for a handful of steps, and for each step:
  - saves the raw camera frame with YOLO boxes drawn, so a sign/axis error
    in the mount transform or FOV shows up visually (e.g. boxes on cars but
    the frame looks like it's pointed at the sky).
  - back-projects every detection to world (x, y) and matches it against the
    nearest REAL NPC position (ground truth, from the same actors the old
    ground-truth obstacle path reads), so a bug in
    CarlaMPCEnv._pixel_to_ground_world shows up as a large, consistent
    position error rather than as a benchmark number that's merely "worse".

This is the 30-second check mentioned when the perception path was added:
run it before trusting any --perception benchmark result.

Usage:
    python tools/debug_perception.py --town Town01 --steps 40
    python tools/debug_perception.py --town Town01 --steps 40 --save-every 1
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from mpc_controller.envs.carlaEnv import CarlaMPCEnv  # noqa: E402


def _draw_boxes(frame_rgb, boxes, classes):
    import cv2
    frame_bgr = frame_rgb[:, :, ::-1].copy()  # RGB -> BGR for cv2
    for box in boxes.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        color = (0, 255, 0) if cls_id in classes else (0, 0, 255)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_bgr, str(cls_id), (x1, max(y1 - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame_bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='Town01')
    ap.add_argument('--seed', type=int, default=2547)
    ap.add_argument('--steps', type=int, default=300,
                     help='20 Hz control loop, so 300 steps = 15 s of driving. '
                          'NPCs are spread across the whole route, not '
                          'clustered at the start, so a short run can easily '
                          'see nothing even with a working pipeline.')
    ap.add_argument('--save-every', type=int, default=5,
                     help='save an annotated frame every N steps (default 5)')
    ap.add_argument('--match-radius', type=float, default=5.0, metavar='M',
                     help='max distance (m) to count a detection as matched '
                          'to a real NPC, rather than a false positive/'
                          'badly-projected point')
    ap.add_argument('--out-dir', default='results/perception_debug')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    env = CarlaMPCEnv(
        host=args.host, port=args.port,
        towns=[args.town], episodes_per_town=10 ** 9,
        seed=args.seed, npc_min=5, npc_max=7,
        use_perception=True,
    )

    print(f"resetting on {args.town} ...")
    env.reset()

    all_errors = []
    unmatched_detections = 0
    frames_with_detections = 0

    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))

        frame = env._perception_frame
        if frame is None:
            print(f"step {step:3d}: no camera frame yet")
            if terminated or truncated:
                break
            continue

        model = CarlaMPCEnv._get_yolo_model()
        results = model.predict(frame, verbose=False, conf=0.35)[0]

        # Ground truth NPC world positions this step, for matching -- but only
        # ones actually in the window the real obstacle pipeline cares about
        # (-5 < ds < 30 m, same as _detect_obstacles). Counting every alive
        # NPC regardless of position was misleading: a route can easily have
        # several NPCs total while none are anywhere near the camera.
        gt_positions = []
        nearest_ds = None
        for npc_data in env.racing_npcs:
            npc = npc_data.get("actor")
            if npc is None or not npc.is_alive:
                continue
            loc = npc.get_location()
            try:
                s_obs, _, _ = env.frenet_converter.world_to_frenet(
                    loc.x, loc.y, 0, s_hint=npc_data.get("s"))
            except Exception:
                continue
            ds = s_obs - env.current_s
            if nearest_ds is None or abs(ds) < abs(nearest_ds):
                nearest_ds = ds
            if -5.0 < ds < 30.0:
                gt_positions.append((loc.x, loc.y))

        step_errors = []
        n_detections = 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in CarlaMPCEnv._PERCEPTION_CLASSES:
                continue
            n_detections += 1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            world_xy = env._pixel_to_ground_world((x1 + x2) / 2.0, y2)
            if world_xy is None:
                continue
            wx, wy = world_xy
            if not gt_positions:
                continue
            dists = [math.hypot(wx - gx, wy - gy) for gx, gy in gt_positions]
            best = min(dists)
            if best <= args.match_radius:
                step_errors.append(best)
            else:
                unmatched_detections += 1

        if n_detections:
            frames_with_detections += 1
        all_errors.extend(step_errors)

        err_txt = (f"mean err {np.mean(step_errors):.2f} m "
                   f"(n={len(step_errors)})" if step_errors else "no matches")
        ds_txt = f"{nearest_ds:+.1f}" if nearest_ds is not None else "?"
        print(f"step {step:3d}: {n_detections} detections, "
              f"{len(gt_positions)} NPC(s) in the -5..30m window "
              f"(nearest any NPC: ds={ds_txt} m) -> {err_txt}")

        if args.save_every > 0 and step % args.save_every == 0:
            import cv2
            annotated = _draw_boxes(frame, results, CarlaMPCEnv._PERCEPTION_CLASSES)
            out_path = os.path.join(args.out_dir, f"frame_{step:04d}.png")
            cv2.imwrite(out_path, annotated)

        if terminated or truncated:
            print("episode ended, stopping")
            break

    env.close()

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"town: {args.town}  steps: {args.steps}\n")
        f.write(f"frames with >=1 detection: {frames_with_detections}\n")
        f.write(f"matched detections: {len(all_errors)}\n")
        f.write(f"unmatched detections (false positive or bad projection, "
                f">{args.match_radius} m from any real NPC): "
                f"{unmatched_detections}\n")
        if all_errors:
            f.write(f"back-projection error: mean {np.mean(all_errors):.2f} m, "
                    f"median {np.median(all_errors):.2f} m, "
                    f"max {np.max(all_errors):.2f} m\n")
        else:
            f.write("no matched detections at all -- check the mount "
                    "transform/FOV/intrinsics before trusting anything else.\n")

    print(f"\nsaved annotated frames + summary to {args.out_dir}/")
    print(open(summary_path).read())


if __name__ == '__main__':
    main()
