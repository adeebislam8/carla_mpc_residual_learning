#!/usr/bin/env python
"""
Record MPCC episodes with ORION's camera rig, for offline labelling by
tools/orion/orion_infer.py on the A5000.

Runs on the CARLA machine (16 GB card) in the project env.  The ego is driven
by the nominal MPCC (action = 0), exactly as in benchmark_mpcc.py, with the
same controller flags; ORION's six cameras and an IMU ride along.

    python tools/orion/record_orion_episodes.py --town Town01 --seeds 1 \\
        --episodes 20 --route-max 150 --qc 0.5 --gate-depth 0.98 \\
        --out data/orion_rec

    # then, on the A5000 (frames are already JPEG quality 20, so no re-encode):
    python tools/orion/orion_infer.py --orion-root ~/Orion --jpeg-quality 0 \\
        --episode data/orion_rec/Town01_s1_e* --out data/orion_labels

One directory per episode, in the layout orion_infer.py documents:

    <out>/<town>_s<seed>_e<NNN>/
      CAM_FRONT/000001.jpg ...   six cameras, BGR, 1600x900, JPEG quality 20
      meta.jsonl                 one record per simulator tick (see below)
      obs.npy                    (T, D) env observation after each tick, in
                                 --obs-version layout -- the student's input
      summary.json               outcome, length, flags

meta.jsonl fields.  ORION's inputs: step, x, y, compass, speed, accel, gyro,
command, near_xy.  Everything else is for building the student dataset:

    s, d, alpha              ego Frenet state after this tick
    obstacles, obstacle_vs   [s, d] and path speed of the selected obstacles
    corridor                 (n_min, n_max) at s
    u_nom_prev_step,         commands applied DURING the tick that produced
    u_final_prev_step        this frame.  The nominal for THIS frame's state is
                             computed on the next tick, so it is the NEXT
                             record's u_nom_prev_step.  Shift by one when
                             pairing with ORION's label for this frame.
    dpsi, junction_ahead     raw inputs of the navigation command, so the
                             command can be recomputed without re-recording

Conventions the label builder must respect:
  * Our steering is the MPC model's: the env sends control.steer =
    -steering to CARLA, so u_*[1] > 0 steers LEFT.  ORION's PID outputs
    CARLA steer (> 0 = right).  Model steering = -ORION steer.
  * Our throttle is signed, brake < 0.  ORION gives throttle and brake
    separately: model throttle = throttle - brake.

Navigation command (ORION's 6 RoadOption values) from route geometry:
    LANEFOLLOW (4) unless a junction lies within --junction-lookahead metres
    along the route; then LEFT (1) / RIGHT (2) when the route heading turns
    more than --turn-threshold-deg over the next --turn-window metres, else
    STRAIGHT (3).  CARLA's yaw grows clockwise seen from above, so a left turn
    is a NEGATIVE heading change.  Not verified on a real junction yet: check
    the first recorded turn's dpsi sign against the video before labelling a
    large batch.  Lane changes (5, 6) are never emitted -- the route has none.
"""

import argparse
import json
import math
import os
import queue
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'src'))

# Must match ORION_SENSORS in orion_infer.py, which carries the calibration.
ORION_SENSORS = [
    dict(id='CAM_FRONT',       x=0.80,  y=0.0,   z=1.60, yaw=0.0,    fov=70),
    dict(id='CAM_FRONT_LEFT',  x=0.27,  y=-0.55, z=1.60, yaw=-55.0,  fov=70),
    dict(id='CAM_FRONT_RIGHT', x=0.27,  y=0.55,  z=1.60, yaw=55.0,   fov=70),
    dict(id='CAM_BACK',        x=-2.0,  y=0.0,   z=1.60, yaw=180.0,  fov=110),
    dict(id='CAM_BACK_LEFT',   x=-0.32, y=-0.55, z=1.60, yaw=-110.0, fov=70),
    dict(id='CAM_BACK_RIGHT',  x=-0.32, y=0.55,  z=1.60, yaw=110.0,  fov=70),
]
IMG_W, IMG_H = 1600, 900
JPEG_QUALITY = 20     # what ORION's agent feeds the model; see orion_infer.py

# RoadOption values ORION's command2hot expects.
CMD_LEFT, CMD_RIGHT, CMD_STRAIGHT, CMD_LANEFOLLOW = 1, 2, 3, 4

# The env's debug.draw_* calls put real geometry into the world, which every
# camera sees (the env itself suppresses them under use_perception for the
# same reason).  ORION must not be shown painted corridor lines.
_DEBUG_DRAWERS = ('_draw_road_boundaries_ahead', '_visualize_vehicle_footprint',
                  '_visualize_cbf_ellipses', '_visualize_mpc_prediction',
                  '_visualize_detected_obstacles', '_visualize_lidar_obstacles')


class OrionRig:
    """ORION's six cameras plus an IMU, attached to the ego, synchronised to
    the world tick through one queue per sensor."""

    def __init__(self, world, vehicle):
        import carla
        self.world = world
        self.sensors, self.queues = {}, {}
        lib = world.get_blueprint_library()
        for spec in ORION_SENSORS:
            bp = lib.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(IMG_W))
            bp.set_attribute('image_size_y', str(IMG_H))
            bp.set_attribute('fov', str(spec['fov']))
            tf = carla.Transform(carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
                                 carla.Rotation(yaw=spec['yaw']))
            self._spawn(spec['id'], bp, tf, vehicle)
        imu = lib.find('sensor.other.imu')
        self._spawn('IMU', imu, carla.Transform(carla.Location(x=-1.4)), vehicle)

    def _spawn(self, name, bp, tf, vehicle):
        actor = self.world.spawn_actor(bp, tf, attach_to=vehicle)
        q = queue.Queue()
        actor.listen(q.put)
        self.sensors[name], self.queues[name] = actor, q

    def read(self, frame, timeout=5.0):
        """Data for exactly this world frame from every sensor; older
        measurements are discarded.  Raises TimeoutError if one never comes."""
        out = {}
        for name, q in self.queues.items():
            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f'{name}: no data for frame {frame}')
                data = q.get(timeout=remaining)
                if data.frame == frame:
                    out[name] = data
                    break
                if data.frame > frame:
                    raise TimeoutError(f'{name}: skipped past frame {frame}')
        return out

    def destroy(self):
        for actor in self.sensors.values():
            try:
                actor.stop()
                actor.destroy()
            except Exception:
                pass
        self.sensors, self.queues = {}, {}


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def navigation_command(env, args):
    """(command, dpsi, junction_ahead) from the route ahead of the ego."""
    import carla
    fc, s0 = env.frenet_converter, env.current_s
    s_end = env.path_length
    junction = False
    for ds in np.arange(0.0, args.junction_lookahead + 1e-6, 2.0):
        s = min(s0 + ds, s_end)
        x, y, _ = fc.frenet_to_world(s, 0.0, 0.0)
        wp = env.map.get_waypoint(carla.Location(x=x, y=y, z=0.0),
                                  project_to_road=True)
        if wp is not None and wp.is_junction:
            junction = True
            break
    _, _, yaw0 = fc.frenet_to_world(s0, 0.0, 0.0)
    _, _, yaw1 = fc.frenet_to_world(min(s0 + args.turn_window, s_end), 0.0, 0.0)
    dpsi = _wrap(yaw1 - yaw0)
    if not junction:
        return CMD_LANEFOLLOW, dpsi, False
    thr = math.radians(args.turn_threshold_deg)
    if dpsi < -thr:
        return CMD_LEFT, dpsi, True
    if dpsi > thr:
        return CMD_RIGHT, dpsi, True
    return CMD_STRAIGHT, dpsi, True


def _vec(v):
    return [float(v.x), float(v.y), float(v.z)]


def record_episode(env, ep_dir, args):
    import cv2

    for name in _DEBUG_DRAWERS:
        if hasattr(env, name):
            setattr(env, name, lambda *a, **k: None)

    obs, _ = env.reset()
    rig = OrionRig(env.world, env.vehicle)
    ep_dir.mkdir(parents=True, exist_ok=True)
    for spec in ORION_SENSORS:
        (ep_dir / spec['id']).mkdir(exist_ok=True)
    jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    observations, n_saved, outcome = [], 0, 'running'
    zero = np.zeros(env.action_space.shape)
    try:
        with open(ep_dir / 'meta.jsonl', 'w') as meta_f:
            for _ in range(args.max_steps):
                obs, _, done, trunc, info = env.step(zero)
                frame = env.world.get_snapshot().frame
                data = rig.read(frame)

                if env.current_step % args.stride == 0:
                    for spec in ORION_SENSORS:
                        img = data[spec['id']]
                        bgra = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
                            (img.height, img.width, 4))
                        cv2.imwrite(str(ep_dir / spec['id'] / f'{env.current_step:06d}.jpg'),
                                    bgra[:, :, :3], jpeg)
                    imu = data['IMU']
                    tf = env.vehicle.get_transform()
                    cmd, dpsi, junction = navigation_command(env, args)
                    s_near = min(env.current_s + args.near_node_m, env.path_length)
                    nx, ny, _ = env.frenet_converter.frenet_to_world(s_near, 0.0, 0.0)
                    trace = env._step_trace[-1] if env._step_trace else {}
                    rec = {
                        'step': int(env.current_step),
                        'x': float(tf.location.x), 'y': float(tf.location.y),
                        'compass': float(imu.compass),
                        'speed': float(env.current_speed),
                        'accel': _vec(imu.accelerometer),
                        'gyro': _vec(imu.gyroscope),
                        'command': cmd, 'near_xy': [float(nx), float(ny)],
                        'dpsi': float(dpsi), 'junction_ahead': junction,
                        's': float(env.current_s), 'd': float(env.current_d),
                        'alpha': float(env.current_alpha),
                        'obstacles': env.selected_obstacles.tolist(),
                        'obstacle_vs': env.selected_obstacle_vs.tolist(),
                        'corridor': env._corridor_at(env.current_s),
                        'u_nom_prev_step': [trace.get('mpc_throttle'),
                                            trace.get('mpc_steering')],
                        'u_final_prev_step': [trace.get('final_throttle'),
                                              trace.get('final_steering')],
                    }
                    meta_f.write(json.dumps(rec) + '\n')
                    observations.append(np.asarray(obs, dtype=np.float32))
                    n_saved += 1

                if done or trunc:
                    outcome = info.get('done_reason', 'unknown')
                    break
    finally:
        rig.destroy()

    np.save(ep_dir / 'obs.npy', np.stack(observations) if observations
            else np.zeros((0, env.state_dim), np.float32))
    with open(ep_dir / 'summary.json', 'w') as f:
        json.dump({'outcome': outcome, 'frames': n_saved,
                   'last_step': int(env.current_step), 'flags': vars(args)},
                  f, indent=2)
    return outcome, n_saved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='data/orion_rec')
    ap.add_argument('--town', default='Town01')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1])
    ap.add_argument('--episodes', type=int, default=10, help='per seed')
    ap.add_argument('--stride', type=int, default=1,
                    help='save every Nth tick (1 = 20 Hz, what ORION ran at)')
    ap.add_argument('--obs-version', default='v2', choices=['v1', 'v2'])
    # controller flags, as in benchmark_mpcc.py
    ap.add_argument('--qc', type=float, default=0.5)
    ap.add_argument('--gate-depth', type=float, default=0.98)
    ap.add_argument('--route-min', type=float, default=50.0)
    ap.add_argument('--route-max', type=float, default=150.0)
    ap.add_argument('--target-speed', type=float, default=15.0)
    ap.add_argument('--max-steps', type=int, default=1500)
    ap.add_argument('--npc-min', type=int, default=0)
    ap.add_argument('--npc-max', type=int, default=7)
    # navigation command
    ap.add_argument('--junction-lookahead', type=float, default=20.0, metavar='M')
    ap.add_argument('--turn-window', type=float, default=30.0, metavar='M')
    ap.add_argument('--turn-threshold-deg', type=float, default=30.0)
    ap.add_argument('--near-node-m', type=float, default=10.0,
                    help="distance ahead of the 'next route node' ORION's PID logs")
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    args = ap.parse_args()

    from mpc_controller.envs.carlaEnv import CarlaMPCEnv
    out = Path(args.out)
    for seed in args.seeds:
        env = CarlaMPCEnv(
            host=args.host, port=args.port, towns=[args.town],
            episodes_per_town=10 ** 9, target_speed=args.target_speed,
            max_steps=args.max_steps, seed=seed, qc=args.qc,
            gate_depth=args.gate_depth, route_min_m=args.route_min,
            route_max_m=args.route_max, npc_min=args.npc_min,
            npc_max=args.npc_max, obs_version=args.obs_version)
        try:
            for e in range(args.episodes):
                ep_dir = out / f'{args.town}_s{seed}_e{e:03d}'
                t0 = time.time()
                outcome, n = record_episode(env, ep_dir, args)
                print(f'{ep_dir.name}: {outcome:9s} {n:4d} frames '
                      f'({time.time() - t0:.0f} s wall)')
        finally:
            env.close()


if __name__ == '__main__':
    main()
