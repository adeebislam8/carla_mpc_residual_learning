#!/usr/bin/env python
"""
Offline ORION teacher inference -- runs in ORION's own conda env on the A5000.

ORION (ICCV 2025, github.com/xiaomi-mlab/Orion) is the teacher for residual
distillation.  It never runs inside our CARLA loop: the recorder saves frames
on the 16 GB card, and this script labels them afterwards.

The preprocessing is a line-for-line port of ORION's own closed-loop agent
(team_code/orion_b2d_agent.py, OrionAgent.run_step) minus everything that
needs the Bench2Drive leaderboard: sensors come from disk instead of
input_data, and pose comes from the recorded world pose instead of GNSS.
Anything that differs from the agent is marked DEVIATION.

Two modes:

    # 1. Plumbing + VRAM check, no recorded data needed (synthetic frames).
    #    Outputs are meaningless; it proves the model loads and fits.
    python tools/orion/orion_infer.py --orion-root ~/Orion --selftest 20

    # 2. Label recorded episodes.
    python tools/orion/orion_infer.py --orion-root ~/Orion \\
        --episode data/orion_rec/Town01_s1_e000 data/orion_rec/Town01_s1_e001 \\
        --out data/orion_labels

Recorded episode layout (the recorder must write exactly this):

    <episode>/
      CAM_FRONT/000000.jpg  ...      one dir per camera in CAMS, BGR, 1600x900,
      CAM_FRONT_LEFT/...             rig exactly as in ORION_SENSORS below
      ...
      meta.jsonl                     one JSON object per recorded step:
        {"step": int,                simulator tick index (0.05 s ticks)
         "x": float, "y": float,     CARLA world location of the ego
         "compass": float,           IMU compass, radians (raw sensor value)
         "speed": float,             m/s
         "accel": [ax, ay, az],      IMU accelerometer
         "gyro": [wx, wy, wz],       IMU gyroscope
         "command": int,             RoadOption value 1..6 (4 = LANEFOLLOW)
         "near_xy": [x, y]}          next route node, CARLA world coords

Output per episode: <out>/<episode_name>.npz with
    plan      (T, 6, 2)  raw ORION waypoints, ORION/lidar frame, 0.5 s apart
    plan_fl   (T, 6, 2)  same, as (forward, left) metres in the ego frame
    ctrl      (T, 3)     steer, throttle, brake from ORION's own PID
    desired_speed (T,)   speed target the PID derived from the plan
    empty     (T,)       True where ORION returned the all-zero fallback plan
    steps     (T,)       simulator tick of each row
    latency_s (T,)       wall time per forward pass

Setup on the A5000 (Linux, CUDA 11.8 toolkit for building ORION's ops):

    git clone https://github.com/xiaomi-mlab/Orion.git ~/Orion && cd ~/Orion
    conda create -n orion python=3.8 -y && conda activate orion
    pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1 \\
        --index-url https://download.pytorch.org/whl/cu118
    pip install -v -e . && pip install -r requirements.txt
    # requirements.txt pins flash-attn==0.2.8, but mmcv/models/utils/attention.py
    # imports the flash-attn 2.x API (flash_attn_varlen_kvpacked_func).  Install
    # a 2.x build for torch 2.4 / cu118 instead.  A5000 is Ampere (sm_86), which
    # flash-attn 2 supports.
    mkdir -p ckpts
    huggingface-cli download exiawsh/pretrain_qformer --local-dir ckpts/pretrain_qformer
    huggingface-cli download poleyzdk/Orion Orion.pth --local-dir ckpts

Status: written against ORION's code as of 2026-10-08 but NOT yet run -- the
dev machine has no GPU.  Run --selftest first.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# ORION's camera rig, copied from OrionAgent.sensors().  The recorder spawns
# these in CARLA alongside our own 3 YOLO cameras.  The calibration matrices
# below are only valid for exactly this rig.
ORION_SENSORS = [
    dict(id='CAM_FRONT',       x=0.80,  y=0.0,   z=1.60, yaw=0.0,    fov=70),
    dict(id='CAM_FRONT_LEFT',  x=0.27,  y=-0.55, z=1.60, yaw=-55.0,  fov=70),
    dict(id='CAM_FRONT_RIGHT', x=0.27,  y=0.55,  z=1.60, yaw=55.0,   fov=70),
    dict(id='CAM_BACK',        x=-2.0,  y=0.0,   z=1.60, yaw=180.0,  fov=110),
    dict(id='CAM_BACK_LEFT',   x=-0.32, y=-0.55, z=1.60, yaw=-110.0, fov=70),
    dict(id='CAM_BACK_RIGHT',  x=-0.32, y=0.55,  z=1.60, yaw=110.0,  fov=70),
]
IMG_W, IMG_H = 1600, 900

# Fixed calibration, copied verbatim from OrionAgent.setup().
LIDAR2IMG = {
    'CAM_FRONT': np.array([
        [1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
        [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
        [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_FRONT_LEFT': np.array([
        [6.03961325e-14, 1.39475744e+03, 0.00000000e+00, -9.20539908e+02],
        [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
        [-8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_FRONT_RIGHT': np.array([
        [1.31064327e+03, -4.77035138e+02, 0.00000000e+00, -4.06010608e+02],
        [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
        [8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK': np.array([
        [-5.60166031e+02, -8.00000000e+02, 0.00000000e+00, -1.28800000e+03],
        [5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
        [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, -1.61000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK_LEFT': np.array([
        [-1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -6.84385123e+02],
        [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
        [-9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK_RIGHT': np.array([
        [3.60989788e+02, -1.34723223e+03, 0.00000000e+00, -1.04238127e+02],
        [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
        [9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
}
LIDAR2CAM = {
    'CAM_FRONT': np.array([
        [1., 0., 0., 0.],
        [0., 0., -1., -0.24],
        [0., 1., 0., -1.19],
        [0., 0., 0., 1.]]),
    'CAM_FRONT_LEFT': np.array([
        [0.57357644, 0.81915204, 0., -0.22517331],
        [0., 0., -1., -0.24],
        [-0.81915204, 0.57357644, 0., -0.82909407],
        [0., 0., 0., 1.]]),
    'CAM_FRONT_RIGHT': np.array([
        [0.57357644, -0.81915204, 0., 0.22517331],
        [0., 0., -1., -0.24],
        [0.81915204, 0.57357644, 0., -0.82909407],
        [0., 0., 0., 1.]]),
    'CAM_BACK': np.array([
        [-1., 0., 0., 0.],
        [0., 0., -1., -0.24],
        [0., -1., 0., -1.61],
        [0., 0., 0., 1.]]),
    'CAM_BACK_LEFT': np.array([
        [-0.34202014, 0.93969262, 0., -0.25388956],
        [0., 0., -1., -0.24],
        [-0.93969262, -0.34202014, 0., -0.49288953],
        [0., 0., 0., 1.]]),
    'CAM_BACK_RIGHT': np.array([
        [-0.34202014, -0.93969262, 0., 0.25388956],
        [0., 0., -1., -0.24],
        [0.93969262, -0.34202014, 0., -0.49288953],
        [0., 0., 0., 1.]]),
}
LIDAR2EGO = np.array([[0., 1., 0., -0.39],
                      [-1., 0., 0., 0.],
                      [0., 0., 1., 1.84],
                      [0., 0., 0., 1.]])

# Modules OrionAgent / test.py keep in fp32 when enabling fp16.
_KEEP_FP32 = ('map_head', 'pts_bbox_head')


def _custom_wrap_fp16_model(model):
    """Same as custom_wrap_fp16_model() in ORION's agent and test.py."""
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True
    for name in _KEEP_FP32:
        model._modules[name].fp16_enabled = False


def _invert_pose(pose):
    inv = np.zeros((4, 4), dtype=np.float32)
    r, t = pose[:3, :3], pose[:3, 3]
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ t
    inv[3, 3] = 1.0
    return inv


def _command_index(command):
    """command2nohot / command2hot from the agent: RoadOption 1..6 -> 0..5."""
    if command < 0:
        command = 4
    return command - 1


class OrionTeacher:
    def __init__(self, orion_root, ckpt, dt=0.05, jpeg_quality=20):
        orion_root = str(Path(orion_root).expanduser().resolve())
        # The config refers to 'ckpts/pretrain_qformer/' relative to the repo.
        os.chdir(orion_root)
        sys.path.insert(0, orion_root)

        from mmcv import Config
        from mmcv.models import build_model
        from mmcv.utils import load_checkpoint
        from mmcv.datasets.pipelines import Compose
        from mmcv.parallel.collate import collate
        from mmcv.core.bbox import get_box_type
        from pyquaternion import Quaternion
        from team_code.pid_controller import PIDController

        self._collate = collate
        self._Quaternion = Quaternion
        self._PIDController = PIDController
        self._box_type, _ = get_box_type('LiDAR')

        # DEVIATION: OrionAgent loads orion_stage3_agent.py, which is the FP32
        # config (>32 GB).  We build from the FP16 config (>17 GB) and borrow
        # only the agent's inference_only_pipeline, which the FP16 config lacks.
        cfg = Config.fromfile('adzoo/orion/configs/orion_stage3_fp16.py')
        agent_cfg = Config.fromfile('adzoo/orion/configs/orion_stage3_agent.py')
        cfg.model.train_cfg = None

        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        _custom_wrap_fp16_model(model)
        load_checkpoint(model, str(Path(ckpt).expanduser()), map_location='cpu')
        self.model = model.cuda().eval()

        self.pipeline = Compose([
            p for p in agent_cfg.inference_only_pipeline
            if p['type'] != 'LoadMultiViewImageFromFilesInCeph'
        ])
        self.dt = dt
        self.jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        self.pid = None

    def reset(self):
        """Call at every episode start: fresh PID windows.  ORION's temporal
        memory refreshes itself when scene_token changes (pre_update_memory)."""
        self.pid = self._PIDController()

    def _jpeg_roundtrip(self, img):
        # The agent JPEG-encodes every frame at quality 20 before inference,
        # so ORION ran closed-loop on heavily compressed images.  Match it --
        # unless the recorder already saved at that quality (--jpeg-quality 0),
        # where a second pass would compress twice.
        if self.jpeg_params[1] <= 0:
            return img
        _, buf = cv2.imencode('.jpg', img, self.jpeg_params)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    @torch.no_grad()
    def step(self, imgs, m, scene_token):
        """
        imgs: dict cam -> BGR uint8 (900, 1600, 3)
        m:    one meta.jsonl record
        Returns (plan (6,2), ctrl (3,), desired_speed, latency_s).
        """
        Q = self._Quaternion
        results = {
            'lidar2img': [], 'lidar2cam': [], 'cam_intrinsic': [], 'img': [],
            'folder': ' ',
            'scene_token': scene_token,
            'frame_idx': m['step'],
            # The agent uses step / 20 with one step per 0.05 s tick.
            'timestamp': m['step'] * self.dt,
            'box_type_3d': self._box_type,
        }
        for cam in CAMS:
            results['lidar2img'].append(LIDAR2IMG[cam])
            results['lidar2cam'].append(LIDAR2CAM[cam])
            results['cam_intrinsic'].append(
                LIDAR2IMG[cam] @ np.linalg.inv(LIDAR2CAM[cam]))
            results['img'].append(self._jpeg_roundtrip(imgs[cam]))
        results['lidar2img'] = np.stack(results['lidar2img'], axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'], axis=0)

        raw_theta = m['compass'] if not np.isnan(m['compass']) else 0.0
        ego_theta = -raw_theta + np.pi / 2
        # DEVIATION: the agent derives pos from GNSS through a fitted lat/lon
        # reference, which reproduces CARLA world (x, y).  We use the recorded
        # world location directly.  Verify on the first real episode that plans
        # point along the road; a sign error here shows up as mirrored plans.
        pos = (m['x'], m['y'])

        can_bus = np.zeros(18)
        can_bus[0] = pos[0]
        can_bus[1] = -pos[1]
        can_bus[3:7] = list(Q(axis=[0, 0, 1], radians=ego_theta))
        can_bus[7] = m['speed']
        can_bus[10:13] = m['accel']
        can_bus[11] *= -1
        can_bus[13:16] = -np.asarray(m['gyro'], dtype=float)
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180
        results['can_bus'] = can_bus

        cmd = _command_index(m['command'])
        results['command'] = cmd
        hot = np.zeros(6)
        hot[cmd] = 1
        results['ego_fut_cmd'] = hot

        ego2world = np.eye(4)
        ego2world[0:3, 0:3] = Q(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2, 3] = can_bus[0:2]
        lidar2global = ego2world @ LIDAR2EGO
        results['ego_pose'] = lidar2global
        results['ego_pose_inv'] = _invert_pose(lidar2global)
        results['lidar2ego'] = LIDAR2EGO
        results['l2g_r_mat'] = lidar2global[0:3, 0:3]
        results['l2g_t'] = lidar2global[0:3, 3]

        stacked = np.stack(results['img'], axis=-1)
        results['img_shape'] = stacked.shape
        results['ori_shape'] = stacked.shape
        results['pad_shape'] = stacked.shape

        # Local target for the PID, as in the agent.  The PID ignores it
        # (use_target_to_aim = False) but logs it.
        near = np.array([m['near_xy'][0] - can_bus[0],
                         -m['near_xy'][1] - can_bus[1]])
        rot = np.array([[np.cos(raw_theta), -np.sin(raw_theta)],
                        [np.sin(raw_theta), np.cos(raw_theta)]])
        local_command_xy = rot @ near

        results = self.pipeline(results)
        batch = self._collate([results], samples_per_gpu=1)
        for key, data in batch.items():
            if key != 'img_metas' and torch.is_tensor(data[0]):
                data[0] = data[0].cuda()
            if key == 'input_ids':
                for i in range(len(data[0])):
                    for k in range(len(data[0][i])):
                        data[0][i][k] = data[0][i][k].cuda()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model(batch, return_loss=False)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        plan = out[0]['pts_bbox']['ego_fut_preds'].cpu().numpy()
        steer, throttle, brake, meta = self.pid.control_pid(
            plan, np.float64(m['speed']), local_command_xy)
        # DEVIATION: the agent then zeroes throttle above 5 m/s and caps it at
        # 0.75.  That is a closed-loop safety hack of the agent, not ORION's
        # plan, so it is NOT applied here -- the desired_speed column keeps the
        # speed ORION actually planned.
        ctrl = np.array([np.clip(float(steer), -1, 1),
                         float(throttle),
                         float(brake)])
        return plan, ctrl, float(meta['desired_speed']), latency


def plan_to_forward_left(plan):
    """
    ORION waypoints are in its lidar frame.  LIDAR2EGO maps lidar +y to ego +x
    (forward) and lidar +x to ego -y, and the PID steers with
    angle = 90deg - atan2(y, x), i.e. it treats +y as forward and +x as right.
    So (forward, left) = (y, -x).  Check on a straight road: forward should
    grow by ~speed*0.5 m per waypoint and left should stay near zero.
    """
    return np.stack([plan[..., 1], -plan[..., 0]], axis=-1)


def _load_episode(ep_dir):
    with open(ep_dir / 'meta.jsonl') as f:
        metas = [json.loads(line) for line in f if line.strip()]
    for cam in CAMS:
        if not (ep_dir / cam).is_dir():
            raise FileNotFoundError(f'{ep_dir} has no {cam}/ -- recorder rig mismatch')
    return metas


def _read_frame(ep_dir, step):
    imgs = {}
    for cam in CAMS:
        img = cv2.imread(str(ep_dir / cam / f'{step:06d}.jpg'), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'{ep_dir}/{cam}/{step:06d}.jpg')
        if img.shape[:2] != (IMG_H, IMG_W):
            raise ValueError(f'{cam} frame is {img.shape[:2]}, ORION needs {(IMG_H, IMG_W)}')
        imgs[cam] = img
    return imgs


def _run_sequence(teacher, frames, scene_token):
    """frames: iterable of (imgs, meta).  Returns the column arrays."""
    teacher.reset()
    cols = {k: [] for k in ('plan', 'ctrl', 'desired_speed', 'steps', 'latency_s')}
    for imgs, m in frames:
        plan, ctrl, v_des, lat = teacher.step(imgs, m, scene_token)
        cols['plan'].append(plan)
        cols['ctrl'].append(ctrl)
        cols['desired_speed'].append(v_des)
        cols['steps'].append(m['step'])
        cols['latency_s'].append(lat)
    out = {k: np.asarray(v) for k, v in cols.items()}
    out['plan_fl'] = plan_to_forward_left(out['plan'])
    # ORION substitutes zeros(6, 2) when its text output is unparseable.
    out['empty'] = np.all(out['plan'] == 0, axis=(1, 2))
    return out


def _memory_report():
    gb = 1024 ** 3
    return (f'peak allocated {torch.cuda.max_memory_allocated() / gb:.2f} GB, '
            f'peak reserved {torch.cuda.max_memory_reserved() / gb:.2f} GB, '
            f'device total {torch.cuda.get_device_properties(0).total_memory / gb:.2f} GB')


def _summary(name, out):
    lat = out['latency_s'][1:] if len(out['latency_s']) > 1 else out['latency_s']
    print(f'[{name}] {len(out["steps"])} frames | '
          f'latency median {np.median(lat) * 1000:.0f} ms (first call excluded) | '
          f'empty plans {out["empty"].mean() * 100:.1f}% | '
          f'desired speed median {np.median(out["desired_speed"]):.2f} m/s')


def selftest(teacher, n):
    """Synthetic frames: checks load, preprocessing, VRAM and latency only."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(IMG_H, IMG_W, 3), dtype=np.uint8)

    def frames():
        for i in range(n):
            imgs = {cam: base for cam in CAMS}
            m = dict(step=i, x=0.0, y=-0.5 * i, compass=0.0, speed=5.0,
                     accel=[0.0, 0.0, 9.81], gyro=[0.0, 0.0, 0.0],
                     command=4, near_xy=[0.0, -0.5 * i - 20.0])
            yield imgs, m

    out = _run_sequence(teacher, frames(), scene_token='selftest')
    _summary('selftest', out)
    print('first plan (forward, left) m:\n', np.round(out['plan_fl'][0], 2))
    print(_memory_report())
    print('NOTE: synthetic frames -- the plan values mean nothing.  This only '
          'proves the model loads, runs and fits.')


def label_episodes(teacher, episodes, out_dir, stride):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        ep = Path(ep)
        metas = _load_episode(ep)[::stride]
        frames = ((_read_frame(ep, m['step']), m) for m in metas)
        out = _run_sequence(teacher, frames, scene_token=ep.name)
        np.savez_compressed(out_dir / f'{ep.name}.npz', **out)
        _summary(ep.name, out)
    print(_memory_report())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--orion-root', required=True, help='path to the cloned ORION repo')
    ap.add_argument('--ckpt', default='ckpts/Orion.pth',
                    help='ORION checkpoint, relative to --orion-root or absolute')
    ap.add_argument('--selftest', type=int, metavar='N',
                    help='run N synthetic frames and report VRAM / latency')
    ap.add_argument('--episode', nargs='+', default=[], help='recorded episode dirs')
    ap.add_argument('--out', default='orion_labels', help='output dir for .npz labels')
    ap.add_argument('--stride', type=int, default=1,
                    help='label every Nth recorded step (timestamps stay correct)')
    ap.add_argument('--dt', type=float, default=0.05,
                    help='simulator tick, s; ours and Bench2Drive are both 0.05')
    ap.add_argument('--jpeg-quality', type=int, default=20,
                    help='match the agent (20); 0 = frames were already saved at '
                         'quality 20, do not re-encode')
    args = ap.parse_args()

    if not args.selftest and not args.episode:
        ap.error('give --selftest N or --episode DIR [DIR ...]')

    # Resolve user paths before OrionTeacher chdirs into the ORION repo.
    episodes = [Path(e).expanduser().resolve() for e in args.episode]
    out_dir = Path(args.out).expanduser().resolve()

    teacher = OrionTeacher(args.orion_root, args.ckpt, dt=args.dt,
                           jpeg_quality=args.jpeg_quality)
    print('model loaded |', _memory_report())

    if args.selftest:
        selftest(teacher, args.selftest)
    if episodes:
        label_episodes(teacher, episodes, out_dir, args.stride)


if __name__ == '__main__':
    main()
