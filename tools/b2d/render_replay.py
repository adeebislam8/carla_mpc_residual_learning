#!/usr/bin/env python
"""
Render a CARLA recorder log (run_b2d.sh with RECORD=1) to video frames from a
chase camera behind the ego car, for the submission video.  Needs a running
CARLA server (same version as the recording) and the b2d conda env.

    ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 &        # in $CARLA_ROOT
    python tools/b2d/render_replay.py --log results/b2d/vbase_video/recorder/RouteScenario_24367_rep0.log \
        --out frames/base_24367 --label "MPCC base"

Default 960x810 panels at 30 fps: two side by side, padded to 1920x1080 for iMovie.

The ego car is the replayed vehicle with role_name 'hero' (Leaderboard 2.0).
"""

import argparse
import math
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
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=810)
    ap.add_argument('--behind', type=float, default=7.0)
    ap.add_argument('--up', type=float, default=2.8)
    ap.add_argument('--aim', type=float, default=1.0, help='height on the car the camera looks at (car centred)')
    ap.add_argument('--fps', type=int, default=30, help='render rate; the replayer interpolates the 20 Hz log')
    ap.add_argument('--smooth', type=float, default=0.15, help='heading follow rate per frame (1 = rigid)')
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
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 1.0 / a.fps
    world.apply_settings(settings)
    client.set_replayer_time_factor(0.01)      # hold the replay while the map streams in
    for _ in range(5):
        world.tick()
    hero = find_hero(world)
    if hero is None:
        raise SystemExit('no vehicle with role_name=hero in the replay')
    # Large maps (Town12/13) only stream tiles around the hero/spectator; in a
    # replay the camera ended up in unloaded tiles (sky below the horizon).
    spectator = world.get_spectator()
    def park_spectator():
        spectator.set_transform(carla.Transform(hero.get_transform().location + carla.Location(z=30),
                                                carla.Rotation(pitch=-90)))
    for attr in ('tile_stream_distance', 'actor_active_distance'):
        if hasattr(settings, attr):
            setattr(settings, attr, 3000.0)
    world.apply_settings(settings)
    for _ in range(60):
        park_spectator()
        world.tick()
    client.set_replayer_time_factor(1.0)

    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(a.width))
    bp.set_attribute('image_size_y', str(a.height))
    bp.set_attribute('fov', '90')
    # Camera rigidly attached (no one-tick lag, so the car does not jump in the
    # frame), but its heading is a smoothed copy of the car's: the attached
    # SpringArm shook with every small yaw jitter of the replayed car.
    pitch = -math.degrees(math.atan2(a.up - a.aim, a.behind))
    def rel_tf(ys):
        d = math.radians((ys - hero.get_transform().rotation.yaw + 180.0) % 360.0 - 180.0)
        return carla.Transform(carla.Location(-a.behind * math.cos(d), -a.behind * math.sin(d), a.up),
                               carla.Rotation(pitch=pitch, yaw=math.degrees(d)))
    ys = hero.get_transform().rotation.yaw
    cam = world.spawn_actor(bp, rel_tf(ys), attach_to=hero, attachment_type=carla.AttachmentType.Rigid)
    q = queue.Queue()
    cam.listen(q.put)
    try:
        for n in range(int(dur * a.fps)):
            yaw = hero.get_transform().rotation.yaw
            ys += a.smooth * ((yaw - ys + 180.0) % 360.0 - 180.0)
            cam.set_transform(rel_tf(ys))
            park_spectator()
            frame = world.tick()
            img = q.get(timeout=10.0)
            while img.frame < frame:
                img = q.get(timeout=10.0)
            bgr = np.frombuffer(img.raw_data, np.uint8).reshape(a.height, a.width, 4)[:, :, :3].copy()
            if a.label:
                cv2.rectangle(bgr, (0, 0), (len(a.label) * 17 + 24, 46), (0, 0, 0), -1)
                cv2.putText(bgr, a.label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
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
