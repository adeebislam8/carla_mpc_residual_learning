#!/usr/bin/env python
"""
Render a CARLA recorder log (run_b2d.sh with RECORD=1) to video frames from a
chase camera behind the ego car, for the submission video.  Needs a running
CARLA server (same version as the recording) and the b2d conda env.

    ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 &        # in $CARLA_ROOT
    python tools/b2d/render_replay.py --log results/b2d/video_base_video/recorder/<route>.log \
        --out results/b2d/frames/base_24367 --label "MPCC base"
    ffmpeg -framerate 20 -i results/b2d/frames/base_24367/%05d.jpg -pix_fmt yuv420p base_24367.mp4

The ego car is the replayed vehicle with role_name 'hero' (Leaderboard 2.0).
"""

import argparse
import os
import queue
import re

import carla
import cv2
import numpy as np


def log_duration(client, path):
    info = client.show_recorder_file_info(path, False)
    m = re.search(r'Duration:\s*([\d.]+)', info)
    return float(m.group(1)) if m else 60.0


def find_hero(world):
    vehicles = world.get_actors().filter('vehicle.*')
    for v in vehicles:
        if v.attributes.get('role_name') == 'hero':
            return v
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', required=True, help='recorder .log (absolute path as seen by the server)')
    ap.add_argument('--out', required=True, help='frame directory')
    ap.add_argument('--label', default='')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--start', type=float, default=0.0, help='seconds into the log')
    ap.add_argument('--duration', type=float, default=0.0, help='0 = to the end')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--behind', type=float, default=8.0)
    ap.add_argument('--up', type=float, default=4.0)
    ap.add_argument('--pitch', type=float, default=-15.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    client = carla.Client('localhost', a.port)
    client.set_timeout(120.0)
    log = os.path.abspath(a.log)
    total = log_duration(client, log)
    dur = a.duration if a.duration > 0 else total - a.start
    print(f'{log}: {total:.1f} s recorded, rendering {a.start:.1f}..{a.start + dur:.1f} s')

    client.replay_file(log, a.start, dur, 0, False)      # loads the recorded map
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    world.apply_settings(settings)
    for _ in range(5):
        world.tick()
    hero = find_hero(world)
    if hero is None:
        raise SystemExit('no vehicle with role_name=hero in the replay')

    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(a.width))
    bp.set_attribute('image_size_y', str(a.height))
    bp.set_attribute('fov', '90')
    tf = carla.Transform(carla.Location(x=-a.behind, z=a.up), carla.Rotation(pitch=a.pitch))
    cam = world.spawn_actor(bp, tf, attach_to=hero, attachment_type=carla.AttachmentType.SpringArmGhost)
    q = queue.Queue()
    cam.listen(q.put)
    try:
        for n in range(int(dur / 0.05)):
            frame = world.tick()
            img = q.get(timeout=10.0)
            while img.frame < frame:
                img = q.get(timeout=10.0)
            bgr = np.frombuffer(img.raw_data, np.uint8).reshape(a.height, a.width, 4)[:, :, :3].copy()
            if a.label:
                cv2.rectangle(bgr, (0, 0), (len(a.label) * 22 + 30, 56), (0, 0, 0), -1)
                cv2.putText(bgr, a.label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(a.out, f'{n:05d}.jpg'), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    finally:
        cam.stop()
        cam.destroy()
        client.stop_replayer(False)
        settings.synchronous_mode = False
        world.apply_settings(settings)
    print(f'{n + 1} frames -> {a.out}')


if __name__ == '__main__':
    main()
