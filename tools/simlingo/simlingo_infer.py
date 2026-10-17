#!/usr/bin/env python
"""
SimLingo (CVPR 2025, CARLA Challenge 2024 winner; Bench2Drive DS 86.6) as an
offline TEACHER: label our recorded Bench2Drive episodes, in the same .npz
layout tools/orion/orion_infer.py writes, so build_tactical.py and the whole
student pipeline run unchanged (the teacher ablation: ORION vs SimLingo).

    conda activate simlingo                      # tools/simlingo/setup_simlingo.sh
    python tools/simlingo/simlingo_infer.py --simlingo-root ~/Documents/nett/simlingo \\
        --episode results/b2d/rec_train_clips/rec/* --out results/b2d/rec_train_clips/labels_simlingo
    python tools/simlingo/simlingo_infer.py --simlingo-root ... --selftest 20

Needs recordings made with 'record_simlingo': true (config_record.json): the
SimLingo camera (1024x512, FOV 110, at -1.5/0/2.0) + route.json (the
leaderboard's downsampled plan) in each episode.

The inputs are rebuilt exactly as SimLingo's own agent (team_code/
agent_simlingo.py, tick()) builds them:
  image    bottom 4.8/16 cropped, InternVL2 dynamic_preprocess (448, max 2 tiles)
  speed    rounded to 0.1 m/s
  route    target point + next target point from ITS RoutePlanner (min 7.5 m,
           max 50 m) on the downsampled plan, ego frame via inverse_conversion_2d
  prompt   "Current speed: {v} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>.
           Predict the waypoints."  (eval_route_as = 'target_point', no CoT)
Pose: the recorded true position and IMU compass (SimLingo's agent uses its
UKF estimate -- same quantity, less noise).

Outputs per episode (orion_infer.py-compatible):
  steps, empty, plan_fl (6 x [forward, left], 0.5 s apart, from SimLingo's
  speed waypoints at 0.25 s), path_fl (its spatial route), latency_s.
Its ego frame is x forward, y RIGHT (CARLA); left = -y.  A sign check against
our route (scene route_fl) is printed per episode -- it must be positive.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def find_ckpt(root):
    """A checkpoint file whose ../../../.hydra/config.yaml exists (the layout
    agent_simlingo.py expects)."""
    for p in sorted(Path(root).rglob('*')):
        if p.suffix in ('.pt', '.pth', '.bin', '.ckpt') and p.is_file() \
                and (p.parent.parent.parent / '.hydra' / 'config.yaml').exists():
            return p
    return None


class SimLingoTeacher:
    def __init__(self, root, ckpt):
        import torch
        root = Path(root).expanduser().resolve()
        os.chdir(root)                                       # its relative 'pretrained/' cache
        sys.path[:0] = [str(root), str(root / 'team_code')]
        import hydra
        from omegaconf import OmegaConf
        from transformers import AutoConfig, AutoProcessor
        self.torch = torch
        ckpt = Path(ckpt) if ckpt else find_ckpt(root)
        if ckpt is None or not ckpt.exists():
            raise FileNotFoundError('SimLingo checkpoint not found (download: tools/simlingo/setup_simlingo.sh)')
        cfg = OmegaConf.load(ckpt.parent.parent.parent / '.hydra' / 'config.yaml')
        cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
        self.cfg = cfg
        processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
        self.tok = processor.tokenizer if 'tokenizer' in processor.__dict__ else processor
        self.tok.add_special_tokens({'additional_special_tokens': [
            '<WAYPOINTS>', '<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>', '<ORG_WAYPOINTS>',
            '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>', '<TARGET_POINT>']})
        self.tok.padding_side = 'left'
        cache_dir = f"pretrained/{cfg.model.vision_model.variant.split('/')[1]}"
        dd = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.dev = torch.device('cuda')
        self.model = hydra.utils.instantiate(cfg.model, cfg_data_module=cfg.data_module,
                                             processor=processor, cache_dir=cache_dir,
                                             _recursive_=False).to(self.dev)
        torch.set_default_dtype(dd)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        # conversation template + image-token count, as the agent does
        conv_py = Path(cache_dir) / 'conversation.py'
        if not conv_py.exists():
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=cfg.model.vision_model.variant, local_dir=cache_dir)
        spec = importlib.util.spec_from_file_location('get_conv_template', conv_py)
        self.conv = importlib.util.module_from_spec(spec)
        sys.modules['get_conv_template'] = self.conv
        spec.loader.exec_module(self.conv)
        tc = AutoConfig.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
        image_size = tc.force_image_size or tc.vision_config.image_size
        self.num_image_token = int((image_size // tc.vision_config.patch_size) ** 2
                                   * tc.downsample_ratio ** 2)
        from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
        from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
        from team_code.simlingo_utils import get_camera_extrinsics, get_camera_intrinsics
        import team_code.transfuser_utils as t_u
        from team_code.nav_planner import RoutePlanner
        self.DrivingInput, self.LanguageLabel = DrivingInput, LanguageLabel
        self.build_transform, self.dynamic_preprocess = build_transform, dynamic_preprocess
        self.K, self.E, self.t_u, self.RoutePlanner = (get_camera_intrinsics,
                                                       get_camera_extrinsics, t_u, RoutePlanner)
        self.transform = build_transform(input_size=448)

    def new_route(self, nodes):
        """nodes: [(x, y, z, cmd)] -- the leaderboard's downsampled plan."""
        rp = self.RoutePlanner(7.5, 50.0)          # agent_simlingo.py route_planner_min/max
        # (its run_step never reads the command; the target points are what matter)
        plan = [(SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=z)), int(c))
                for x, y, z, c in nodes]
        rp.set_route(plan, gps=False)
        self.rp = rp

    def _prompt(self, speed):
        prompt = f'Current speed: {speed} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.'
        template = self.conv.get_conv_template('internlm2-chat')
        template.append_message(template.roles[0], '<image>\n' + prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        system = template.system_template.replace('{system_message}', template.system_message) + template.sep
        query = query.replace(system, '')
        image_tokens = '<img>' + '<IMG_CONTEXT>' * self.num_image_token * 2 + '</img>'
        return query.replace('<image>', image_tokens, 1)

    def infer(self, img_bgr, x, y, compass, speed):
        import cv2
        from PIL import Image
        torch, t_u = self.torch, self.t_u
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)            # (recorded JPEG q95 = its re-encode)
        rgb = rgb[:int(rgb.shape[0] - (rgb.shape[0] * 4.8) // 16), :, :]
        H, W = rgb.shape[:2]
        tiles = self.dynamic_preprocess(Image.fromarray(rgb), image_size=448,
                                        use_thumbnail=self.cfg.model.vision_model.use_global_img, max_num=2)
        pv = torch.stack([self.transform(t) for t in tiles])
        pv = pv.view(1, 1, pv.shape[0], 3, pv.shape[2], pv.shape[3])
        yaw = t_u.preprocess_compass(compass)
        gps = np.array([x, y])
        route = self.rp.run_step(np.array([x, y, 0.0]))
        tp = route[1][0] if len(route) > 1 else route[0][0]
        ntp = route[2][0] if len(route) > 2 else tp
        etp = t_u.inverse_conversion_2d(tp[:2], gps, yaw)
        entp = t_u.inverse_conversion_2d(ntp[:2], gps, yaw)
        tps = np.array([etp, entp])
        speed = round(float(speed), 1)
        q = self._prompt(speed)
        tk = self.tok([q], padding=True, return_tensors='pt', return_offsets_mapping=True,
                      add_special_tokens=False)
        valid = tk['input_ids'] != self.tok.pad_token_id
        ll = self.LanguageLabel(phrase_ids=tk['input_ids'].to(self.dev), phrase_valid=valid.to(self.dev),
                                phrase_mask=valid.to(self.dev),
                                placeholder_values=[{self.tok.convert_tokens_to_ids('<TARGET_POINT>'): tps}],
                                language_string=[q], loss_masking=None)
        di = self.DrivingInput(
            camera_images=pv.to(self.dev).bfloat16(), image_sizes=None,
            camera_intrinsics=self.K(W, H, 110).view(1, 3, 3).float().to(self.dev),
            camera_extrinsics=self.E().view(1, 4, 4).float().to(self.dev),
            vehicle_speed=torch.FloatTensor([speed]).unsqueeze(0).to(self.dev),
            target_point=torch.from_numpy(etp[None]).float().to(self.dev),
            prompt=ll, prompt_inference=ll)
        with torch.no_grad():
            wps, path, _ = self.model(di)
        return (None if wps is None else wps.float()[0].cpu().numpy(),
                None if path is None else path.float()[0].cpu().numpy())


def to_plan_fl(speed_wps, dt_in=0.25, n=6, dt_out=0.5):
    """SimLingo speed waypoints (x fwd, y right; point k at (k+1)*dt_in s) ->
    our plan_fl: n points (forward, left) at dt_out, linear in time."""
    w = np.asarray(speed_wps, float)
    t_in = (np.arange(len(w)) + 1) * dt_in
    t_out = (np.arange(n) + 1) * dt_out
    fx = np.interp(t_out, t_in, w[:, 0], left=0.0)
    fy = np.interp(t_out, t_in, w[:, 1], left=0.0)
    if t_out[-1] > t_in[-1]:                        # extrapolate at the last speed
        v = (w[-1] - w[-2]) / dt_in
        late = t_out > t_in[-1]
        fx[late] = w[-1, 0] + v[0] * (t_out[late] - t_in[-1])
        fy[late] = w[-1, 1] + v[1] * (t_out[late] - t_in[-1])
    return np.stack([fx, -fy], 1)


def label_episode(teacher, ep, stride, cam='SIMLINGO_FRONT'):
    import cv2
    metas = [json.loads(l) for l in open(ep / 'meta.jsonl') if l.strip()][::stride]
    route = json.load(open(ep / 'route.json'))
    teacher.new_route(route['nodes'])
    steps, empty, plans, paths, lat, sign = [], [], [], [], [], []
    for m in metas:
        p = ep / cam / f"{m['step']:06d}.jpg"
        img = cv2.imread(str(p)) if p.exists() else None
        steps.append(m['step'])
        if img is None:
            empty.append(True)
            plans.append(np.zeros((6, 2)))
            paths.append(np.zeros((20, 2)))
            continue
        t0 = time.perf_counter()
        wps, path = teacher.infer(img, m['x'], m['y'], m['compass'], m['speed'])
        lat.append(time.perf_counter() - t0)
        empty.append(wps is None)
        plans.append(to_plan_fl(wps) if wps is not None else np.zeros((6, 2)))
        pf = np.stack([path[:, 0], -path[:, 1]], 1) if path is not None else np.zeros((20, 2))
        paths.append(pf[:20] if len(pf) >= 20 else np.vstack([pf, np.repeat(pf[-1:], 20 - len(pf), 0)]))
        rf = np.asarray(m.get('scene', {}).get('route_fl') or [], float)
        if len(rf) >= 2 and path is not None:                  # sign check: path vs our route
            fwd = pf[:, 0]
            ok = (fwd > 3) & (fwd < rf[:, 0].max())
            if ok.sum() >= 2:
                o = np.argsort(rf[:, 0])
                sign.append(np.corrcoef(pf[ok, 1], np.interp(fwd[ok], rf[o, 0], rf[o, 1]))[0, 1])
    out = dict(steps=np.array(steps), empty=np.array(empty), plan_fl=np.array(plans, np.float32),
               path_fl=np.array(paths, np.float32), latency_s=np.array(lat, np.float32),
               teacher=np.array('simlingo'))
    s = np.nanmean(sign) if sign else float('nan')
    print(f'[{ep.name}] {len(steps)} frames, {int(np.sum(empty))} empty, '
          f'{1000 * np.median(lat) if lat else float("nan"):.0f} ms/frame, '
          f'path-vs-route lateral corr {s:+.2f} (must be > 0)', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--simlingo-root', required=True)
    ap.add_argument('--ckpt', default=None, help='default: found under --simlingo-root')
    ap.add_argument('--episode', nargs='+', default=[])
    ap.add_argument('--out', default='simlingo_labels')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--selftest', type=int, metavar='N')
    args = ap.parse_args()
    episodes = [Path(e).expanduser().resolve() for e in args.episode]
    out_dir = Path(args.out).expanduser().resolve()
    sys.path.insert(0, str(REPO / 'src'))
    from efficiency.power import EnergyMeter

    teacher = SimLingoTeacher(args.simlingo_root, args.ckpt)
    print(f'SimLingo loaded: {teacher.n_params / 1e9:.2f} B parameters, '
          f'{teacher.torch.cuda.max_memory_reserved() / 2**30:.1f} GB reserved', flush=True)
    if args.selftest:
        teacher.new_route([(0, 0, 0, 4), (50, 0, 0, 4), (100, 0, 0, 4)])
        img = (np.random.default_rng(0).random((512, 1024, 3)) * 255).astype(np.uint8)
        with EnergyMeter() as em:
            for i in range(args.selftest):
                teacher.infer(img, float(i), 0.0, np.pi / 2, 5.0)
        print('selftest (synthetic image -- outputs meaningless):', em.report(args.selftest),
              f'| peak {teacher.torch.cuda.max_memory_reserved() / 2**30:.1f} GB')
    if episodes:
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        with EnergyMeter() as em:
            for ep in episodes:
                if not (ep / 'route.json').exists() or not (ep / 'SIMLINGO_FRONT').is_dir():
                    print(f'[{ep.name}] no SimLingo camera / route.json (record_simlingo off) -- skipped')
                    continue
                out = label_episode(teacher, ep, args.stride)
                np.savez_compressed(out_dir / f'{ep.name}.npz', **out)
                n += len(out['steps'])
        rep = em.report(n)
        cost_p = out_dir / 'labelling_cost.json'
        runs = json.load(open(cost_p)) if cost_p.exists() else []
        runs.append(dict(rep, teacher='simlingo', params_b=round(teacher.n_params / 1e9, 2),
                         peak_vram_gb=round(teacher.torch.cuda.max_memory_reserved() / 2**30, 2)))
        json.dump(runs, open(cost_p, 'w'), indent=2)
        print('labelling cost:', runs[-1])


if __name__ == '__main__':
    main()
