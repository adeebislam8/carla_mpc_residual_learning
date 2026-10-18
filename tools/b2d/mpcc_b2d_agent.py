"""
Bench2Drive / CARLA Leaderboard 2.0 agent: our MPCC+CBF (+ optional distilled
student) driving the official Bench2Drive routes.

    TEAM_AGENT=<repo>/tools/b2d/mpcc_b2d_agent.py
    TEAM_CONFIG=<repo>/tools/b2d/config_mpcc.json      (see tools/b2d/run_b2d.sh)

The agent does not re-implement the controller.  It builds a CarlaMPCEnv
attached to the leaderboard's world (external_world=...), gives it the
leaderboard route as the Frenet path, and per tick calls exactly the methods
the training/benchmark env uses: _update_vehicle_state, _detect_obstacles,
_get_observation (student) and _compute_final_action (MPCC + gated residual).
So a Bench2Drive number is a number for the same controller as the rest of
the paper.

What the leaderboard needs that our env never had, kept here and minimal:
  * route: the leaderboard's dense (1 m) route, not our own planner
  * obstacles: every vehicle, walker and static prop near the route (ground
    truth from the world -- 'perception': true in the config swaps in the
    env's 3-camera YOLO pipeline)
  * traffic rules: stop for red/yellow lights at their stop line, and come to
    a full stop once at each stop sign.  Stop lines come from the map, light
    state from the simulator.  They enter the MPCC as a CONSTRAINT -- a
    per-stage speed cap from the braking envelope (MPCController.stop_s) -- so
    the OCP plans the stop; a hard brake fires only if it still carries speed
    into the last 2 m (counted, printed at the end).  Our own env has no
    traffic lights in play, so without this every signalised route would fail
    on red-light infractions that say nothing about the method.

Config (JSON, all optional):
  steer_fix (true), qc (0.5), gate_depth (0.98), target_speed (15.0),
  perception (false), traffic_rules (true),
  student (null | dir), student_axes ('steer'), student_chunk_mode ('speculative'),
  support_sigma (null), obs_version ('v2'), residual_max (0.5),
  authority_horizon (10), residual_window (null), obstacle_radius_m (45.0),
  stall_kick ('clear': no anti-stall lunge into stopped obstacles),
  follow (true: speed cap to stop behind the nearest in-lane obstacle),
  predict_obstacles (true: CBF uses constant-velocity obstacle positions),
  gate_mode ('pass_slow': overtake only stationary/slow cars in our lane),
  full_brake (true: D down to -1), breakaway (true: >= 0.45 throttle from rest
  on a clear path), behavior_rules (false: the hand-coded decision-layer arm),
  record (false: write ORION's 6 cameras + meta for offline labelling to
  $B2D_RECORD_DIR/<route>/, the layout tools/orion/orion_infer.py reads),
  tactical (null | dir: the go/slow student, src/student/tactical.py),
  scene_radius_m (110: actors kept for the tactical scene)

Efficiency / robustness (sensor-only):
  light_gate_m (60: run the light/stop-sign detector only within this of the
    next junction entry on the route; null = every tick)
  device ('cuda' | 'cpu': where YOLOP / YOLO run -- the CPU-only study)
  mpc_reuse (false: on a clear road, solve the OCP every 2nd tick and apply
    the previous solution's next stage in between)
  creep_after_s (null | s: stopped this long with nothing to wait for (no stop
    line, no junction ahead) -> creep past the blocker at creep_v m/s, the CBF
    on, for at most creep_max_s)
  obstacle_face (false: LiDAR obstacles enter the CBF as points on the
    cluster's near face / inner edge instead of its centroid)
  obstacle_priority (true: the MPC's 6 obstacle slots go to objects laterally
    within 3 m of the ego first, then by distance; false = by distance along
    the route, behind-first -- the old behaviour.  Its smoke "win" (65.6 vs
    59.1) was a parallel run that shared the acados solver dir; clean n1:
    noprio 60.6 vs 55.5, inside the 9-route noise of ~10 DS -- undecided)
  aeb (true: emergency brake under every arm -- full brake when an obstacle
    the MPC's PLANNED path will overlap is < AEB_TTC_S time-to-contact away;
    a planned pass around it does not trigger)
  record_simlingo (false: also record SimLingo's front camera, 1024x512 FOV 110
    at (-1.5, 0, 2.0), for the SimLingo teacher -- tools/simlingo/)
Env B2D_DUMP_DIR: every 20th tick, front image + LiDAR + pose -> npz, the
  offline input of tools/b2d/bench_stack.py (energy / CPU / INT8 study).

privileged (true | false).  true: ground-truth actors and light states from the
simulator plus the HD map (an ORACLE-perception setting).  false: the
Bench2Drive SENSORS-track setting the end-to-end models (UniAD, VAD, ORION) are
evaluated in -- only on-board sensors and the route:
  * ego pose        GNSS + IMU compass + speedometer (perception/ego_state.py)
  * obstacles       LiDAR, ground removal + clustering + tracking, no learned
                    detector (perception/lidar_obstacles.py)
  * traffic lights  front camera: pretrained COCO YOLO + colour (perception/
                    traffic_lights.py); stop signs likewise
  * route           the leaderboard's DOWNSAMPLED plan (a node every <= 50 m +
                    junction entries/exits, as the base agent class gives every
                    agent) densified by a cubic spline -- not the full 1 m route
  * lane centre     front camera, pretrained YOLOP lane lines projected to the
                    ground (perception/lanes.py) -> offset from that spline
  * stop lines / junctions   from the route's own commands (junction = LEFT /
                    RIGHT / STRAIGHT segments); no HD map
  * corridor        the MPCC's fixed two-lane default (no lane widths)
The simulator is read only for the trace/diagnostics, never for control --
except in the perception ablation (E4), which swaps ONE module for ground truth
while the rest stays sensor-only:
  gt_obstacles (false)  true actors (vehicles, walkers, props) as tracks, fed
                        through the same _set_obstacles_from_tracks as LiDAR
  gt_lights (false)     true light states + stop signs (the privileged
                        _rule_stop_target); camera detector off
  sparse_route false    GT lanes: the dense 1 m lane-centre route; YOLOP off

Recording uses ORION's own Bench2Drive sensor rig (team_code/orion_b2d_agent.py)
and its RoutePlanner command logic, so the teacher sees exactly its training
inputs; images are JPEG quality 20, as ORION's agent re-encodes them.
"""

import json
import math
import os
import random
import sys
import time

import numpy as np
import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))

DEFAULTS = dict(steer_fix=True, qc=0.5, gate_depth=0.98, target_speed=15.0,
                perception=False, traffic_rules=True, student=None,
                student_axes='steer', student_chunk_mode='speculative',
                support_sigma=None, obs_version='v2', residual_max=0.5,
                authority_horizon=10, residual_window=None, obstacle_radius_m=45.0,
                stall_kick='clear', follow=True, predict_obstacles=True,
                gate_mode='pass_slow', full_brake=True, breakaway=True,
                behavior_rules=False, rules_queue=False,
                record=False, tactical=None, scene_radius_m=110.0,
                privileged=True, lidar_mount=[1.3, 0.0, 2.5, 0.0],
                sparse_route=None, yolop='models/yolop/yolop-640-640.onnx',
                light_gate_m=60.0, device='cuda', creep_after_s=None,
                creep_max_s=20.0, creep_v=2.0, mpc_reuse=False, record_simlingo=False,
                obstacle_face=False, obstacle_priority=True,
                gt_obstacles=False, gt_lights=False, aeb=True)

# Props that are flat decals, not obstacles: Bench2Drive's ControlLoss scenario
# puts static.prop.dirtdebris01 on the road, and treating it as a stopped car
# made the MPCC 'overtake' it into the opposite lane at 13 m/s (ot_eb traces).
DRIVABLE_PROPS = ('dirtdebris',)

# ORION's Bench2Drive sensor rig, verbatim (team_code/orion_b2d_agent.py).
ORION_CAMS = [
    ('CAM_FRONT', 0.80, 0.0, 0.0, 70), ('CAM_FRONT_LEFT', 0.27, -0.55, -55.0, 70),
    ('CAM_FRONT_RIGHT', 0.27, 0.55, 55.0, 70), ('CAM_BACK', -2.0, 0.0, 180.0, 110),
    ('CAM_BACK_LEFT', -0.32, -0.55, -110.0, 70), ('CAM_BACK_RIGHT', -0.32, 0.55, 110.0, 70)]
JPEG_QUALITY = 20
# SimLingo's camera (team_code/config_simlingo.py): id, x, y, z, w, h, fov
SIMLINGO_CAM = ('SIMLINGO_FRONT', -1.5, 0.0, 2.0, 1024, 512, 110)
DUMP_EVERY = 20
GNSS_X = -1.4          # IMU/GNSS mount (ORION's rig); the estimator corrects for it


class RoutePlanner:
    """ORION's RoutePlanner (Bench2DriveZoo team_code/planner.py) on world
    coordinates: pops route nodes within min_distance, returns the current and
    next node with their RoadOption -- the 'command' / 'near_xy' ORION uses."""

    def __init__(self, plan_world, min_distance=4.0, max_distance=50.0):
        from collections import deque
        self.route = deque((np.array([tf.location.x, tf.location.y]), cmd)
                           for tf, cmd in plan_world)
        self.min_distance, self.max_distance = min_distance, max_distance

    def run_step(self, pos):
        if len(self.route) == 1:
            return self.route[0], self.route[0]
        to_pop, farthest, cum = 0, -np.inf, 0.0
        for i in range(1, len(self.route)):
            if cum > self.max_distance:
                break
            cum += np.linalg.norm(self.route[i][0] - self.route[i - 1][0])
            dist = np.linalg.norm(self.route[i][0] - pos)
            if dist <= self.min_distance and dist > farthest:
                farthest, to_pop = dist, i
        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
        return self.route[0], self.route[1]


# Comfortable stopping deceleration for the rule layer, m/s^2.
RULE_DECEL = 4.0
STOP_LINE_MARGIN_M = 1.5
LIGHT_GREEN_N = 3            # green detections that release a red hold ...
LIGHT_GREEN_S = 1.0          # ... spread over this long with no red/yellow between
LIGHT_PASSED_M = 3.0         # past the held entry by this much: committed, release
LIGHT_HOLD_MAX_S = 25.0      # no red/yellow seen for this long: release
LIGHT_REHOLD_S = 8.0         # after a green release, ignore 'red' at that entry this long
# Emergency brake (aeb): a safety layer under every arm, like the CBF.
AEB_TTC_S = 1.6              # brake when time-to-contact along the route drops below this
AEB_LEN_M = 4.0              # ego centre -> front (2.4) + slack: obstacle points are centres or faces
AEB_LAT_M = 1.7              # lateral overlap: ego half-width 1.0 + obstacle ~0.7
AEB_MIN_GAP_M = 1.0          # this close and still moving: brake whatever the closing speed
AEB_ON_N = 2                 # consecutive triggering ticks before braking (single-frame phantoms)
AEB_OFF_N = 10               # clear ticks before releasing (0.5 s)


def get_entry_point():
    return 'MPCCAgent'


def _local_to_world(tf, loc):
    """(x, y) of a point given in an actor's local frame (trigger volumes)."""
    yaw = math.radians(tf.rotation.yaw)
    return (tf.location.x + loc.x * math.cos(yaw) - loc.y * math.sin(yaw),
            tf.location.y + loc.x * math.sin(yaw) + loc.y * math.cos(yaw))


def densify_route(nodes, spacing=1.0):
    """Sparse route nodes [(x, y)] -> points every `spacing` m along a
    shape-preserving cubic (PCHIP, chord-length parametrised) through them, plus
    for each output point the index of the node segment it lies in.  (A natural
    CubicSpline overshot ~5 m sideways on a straight before a junction: node
    spacing jumps from 50 m to ~14 m there.)"""
    from scipy.interpolate import PchipInterpolator as CubicSpline
    P = np.asarray(nodes, dtype=float)
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 0.5]
    P = P[keep]
    idx_map = np.nonzero(keep)[0]
    if len(P) < 2:
        return P.tolist(), [0] * len(P)
    t = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if len(P) == 2:
        tt = np.arange(0.0, t[-1], spacing)
        pts = P[0] + np.outer(tt / t[-1], P[1] - P[0])
    else:
        cx, cy = CubicSpline(t, P[:, 0]), CubicSpline(t, P[:, 1])
        tt = np.arange(0.0, t[-1], spacing)
        pts = np.stack([cx(tt), cy(tt)], axis=1)
    seg = np.clip(np.searchsorted(t, tt, side='right') - 1, 0, len(P) - 2)
    return pts.tolist(), idx_map[seg].tolist()


def route_sag_bound(nodes, margin=1.0, cap=5.0):
    """Per node segment: how far the real lane can lie from the densified
    route.  A chord of length L between nodes on a road turning by dtheta per
    chord misses the arc by its sagitta ~ L * dtheta / 8 (R60 bend, 45 m nodes:
    4.2 m, as measured).  The perceived lane correction is clipped to this, so
    on a straight a mis-tracked lane cannot drag the car across lanes."""
    P0 = np.asarray(nodes, dtype=float)
    keep = np.r_[True, np.hypot(*np.diff(P0, axis=0).T) > 0.5]     # as densify_route
    P, idx_map = P0[keep], np.nonzero(keep)[0]
    n = len(P)
    if n < 2:
        return [margin] * max(len(P0), 1)
    v = np.diff(P, axis=0)
    L = np.hypot(v[:, 0], v[:, 1])
    h = np.arctan2(v[:, 1], v[:, 0])
    turn = np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)       # at inner nodes
    out = []
    for k in range(n - 1):
        dth = max(turn[k - 1] if k >= 1 else 0.0, turn[k] if k < len(turn) else 0.0)
        out.append(float(min(cap, margin + L[k] * dth / 8.0)))
    out.append(out[-1])
    # back to ORIGINAL node indices (densify_route's seg ids)
    pos = np.clip(np.searchsorted(idx_map, np.arange(len(P0)), side='right') - 1, 0, n - 1)
    return [out[p] for p in pos]


def _road_width(wp, junction_margin=4.0):
    """Same corridor rule as PathPlanner.get_road_width_at_waypoint, without
    building a GlobalRoutePlanner (slow on Town12/13)."""
    from global_planner.src.global_path_publisher_python import PathPlanner
    shim = type('Shim', (), {'junction_margin': junction_margin})()
    return PathPlanner.get_road_width_at_waypoint(shim, wp)


class MPCCAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self.cfg = dict(DEFAULTS)
        self.core, self.policy, self.trace, self.trace_dir = None, None, [], None
        # Bench2Drive's evaluator passes '<config>+<route save name>' (its own
        # agents split it to name per-route outputs) -- keep the file part.
        self.route_name = None
        # (It APPENDS one per route to the same string -- 'cfg+r1+r2+...' --
        # so the file is the first segment and this route is the last.)
        if path_to_conf_file and '+' in path_to_conf_file:
            parts = path_to_conf_file.split('+')
            path_to_conf_file, self.route_name = parts[0], parts[-1]
        if path_to_conf_file:
            # Fail loudly: a relative path resolved from the Bench2Drive dir
            # once silently fell back to the defaults (the 'nofollow' smoke
            # run was really the default config).
            if not os.path.isfile(path_to_conf_file):
                raise FileNotFoundError(f'agent config {path_to_conf_file!r} not found')
            with open(path_to_conf_file) as f:
                self.cfg.update(json.load(f))
        # Model paths in a config are relative to the repo (the evaluator runs
        # from the Bench2Drive dir).
        for key in ('student', 'tactical', 'yolop'):
            if self.cfg.get(key) and not os.path.isabs(self.cfg[key]):
                self.cfg[key] = os.path.join(_REPO, self.cfg[key])
        print(f'[MPCCAgent] config {path_to_conf_file}: {self.cfg}', flush=True)
        self.core = None
        self.policy = None
        self._actors = {}            # actor id -> npc dict (keeps Frenet hints)
        self._stop_signs_done = set()
        self._stop_hold_until = None
        self._rule_lights = []       # (s_stop, traffic_light) along the route
        self._rule_signs = []        # (s_stop, stop_sign_actor)
        self.n_steps = 0
        self.n_rule_overrides = 0
        self.n_aeb = 0                   # ticks the emergency brake held
        self._aeb_on, self._aeb_off = 0, 0
        self._aeb_active = False
        self.t_control = []
        # Per-step trace -> <B2D_TRACE_DIR>/route_<n>.csv + .png (run_b2d.sh sets it)
        self.trace = []
        self.trace_dir = os.environ.get('B2D_TRACE_DIR')
        # Recording for ORION labelling, and the tactical student.
        self.rec_dir, self.rec_meta, self.rec_obs = None, None, []
        if self.cfg['record']:
            root = os.environ.get('B2D_RECORD_DIR')
            if not root:
                raise RuntimeError("config 'record': true needs B2D_RECORD_DIR")
            name = '_'.join((self.route_name or f'route_{int(time.time())}').split('_')[:5])
            self.rec_dir = os.path.join(root, name)
            for cam, *_ in ORION_CAMS:
                os.makedirs(os.path.join(self.rec_dir, cam), exist_ok=True)
            if self.cfg['record_simlingo']:
                os.makedirs(os.path.join(self.rec_dir, SIMLINGO_CAM[0]), exist_ok=True)
            self.rec_meta = open(os.path.join(self.rec_dir, 'meta.jsonl'), 'w')
        self.tactical = None
        if self.cfg['tactical']:
            from student.tactical import TacticalPolicy
            self.tactical = TacticalPolicy(self.cfg['tactical'])
        self._planner = None
        self.n_go = self.n_wait = self.n_slow = 0
        self.n_yield = 0                 # ticks the student held us at a junction entry
        self._yield_t0, self._yield_off_until = None, -1e9
        self.dump_dir = os.environ.get('B2D_DUMP_DIR')
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
        self.n_light_runs = 0            # ticks the light detector actually ran
        self._light_hold = None          # red hold: {e: entry s, t: last red/yellow, green: run}
        self.n_light_holds = 0
        self._stuck_t, self._creep_t, self._creep_cool = 0.0, 0.0, 0.0
        self.n_creep_events, self.n_creep_ticks = 0, 0

    def sensors(self):
        # Ground-truth obstacles come from the world; the speedometer keeps the
        # leaderboard's sensor interface (and its sync wait) happy.
        speed = [{'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'SPEED'}]
        sensor_only = not self.cfg.get('privileged', True)
        if not self.cfg.get('record') and not sensor_only:
            return speed
        # Recording: ORION's whole rig.  Sensor-only driving alone: just the
        # front camera (traffic lights / stop signs) -- 6 x 1600x900 renders
        # are the slowest part of a tick.
        use = ORION_CAMS if self.cfg.get('record') else ORION_CAMS[:1]
        cams = [{'type': 'sensor.camera.rgb', 'x': x, 'y': y, 'z': 1.60,
                 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
                 'width': 1600, 'height': 900, 'fov': fov, 'id': cid}
                for cid, x, y, yaw, fov in use]
        if self.cfg.get('record') and self.cfg.get('record_simlingo'):
            cid, x, y, z, w, h, fov = SIMLINGO_CAM
            cams.append({'type': 'sensor.camera.rgb', 'x': x, 'y': y, 'z': z,
                         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                         'width': w, 'height': h, 'fov': fov, 'id': cid})
        if sensor_only:
            mx, my, mz, myaw = self.cfg['lidar_mount']
            cams.append({'type': 'sensor.lidar.ray_cast', 'x': mx, 'y': my, 'z': mz,
                         'roll': 0.0, 'pitch': 0.0, 'yaw': myaw, 'id': 'LIDAR'})
        return cams + [
            {'type': 'sensor.other.imu', 'x': GNSS_X, 'y': 0.0, 'z': 0.0, 'roll': 0.0,
             'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.05, 'id': 'IMU'},
            {'type': 'sensor.other.gnss', 'x': GNSS_X, 'y': 0.0, 'z': 0.0, 'roll': 0.0,
             'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.01, 'id': 'GPS'}] + speed

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._dense_route = global_plan_world_coord      # 1 m (Transform, RoadOption)

    # ── setup on the first tick (the hero exists, the world is loaded) ──────

    def _init_core(self):
        from mpc_controller.envs.carlaEnv import CarlaMPCEnv
        from global_planner.src.frenet_world_converter_python import FrenetConverter

        world = CarlaDataProvider.get_world()
        cmap = CarlaDataProvider.get_map()
        if self.hero_actor is None:
            self.get_hero()
        town = cmap.name.split('/')[-1]
        c = self.cfg
        # CarlaMPCEnv.__init__ calls random.seed(); the scenario runner draws
        # from the same global RNG, so keep the benchmark's stream untouched.
        rng_state = random.getstate()
        core = CarlaMPCEnv(
            external_world=world, towns=[town], target_speed=c['target_speed'],
            qc=c['qc'], gate_depth=c['gate_depth'], steer_feedback_fix=c['steer_fix'],
            use_perception=c['perception'], obs_version=c['obs_version'],
            residual_mode='adaptive', residual_max=c['residual_max'],
            authority_horizon=c['authority_horizon'],
            residual_window_m=c['residual_window'], stall_kick=c['stall_kick'],
            follow=c['follow'], predict_obstacles=c['predict_obstacles'],
            gate_mode=c['gate_mode'], full_brake=c['full_brake'],
            breakaway=c['breakaway'], behavior_rules=c['behavior_rules'],
            rules_queue=c['rules_queue'])
        # Stop lines need the speed cap to bind even with follow/predict off.
        core._binding_v_slack = True
        random.setstate(rng_state)
        core.vehicle = self.hero_actor
        core.map = cmap

        # Route -> Frenet path + corridor table, as _generate_path does.
        sensor_only = not c['privileged']
        if c['sparse_route'] is None:
            c['sparse_route'] = sensor_only
        pts, wps, cmds = [], [], []
        if c['sparse_route']:
            # What every end-to-end agent gets: the base class's downsampled
            # plan.  A node's RoadOption labels the segment that starts there.
            nodes = [(tf.location.x, tf.location.y) for tf, _ in self._global_plan_world_coord]
            ncmd = [int(getattr(cm, 'value', cm)) for _, cm in self._global_plan_world_coord]
            dense, seg = densify_route(nodes)
            nb = route_sag_bound(nodes)
            self._dense_route_used = []
            lane_bound = []
            for (x, y), k in zip(dense, seg):
                pts.append((x, y))
                cmds.append(ncmd[k])
                wps.append(None)
                lane_bound.append(nb[k])
            self._lane_bound = np.array(lane_bound)
        for tf, cmd in ([] if c['sparse_route'] else self._dense_route):
            p = (tf.location.x, tf.location.y)
            if pts and math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) < 0.5:
                continue
            pts.append(p)
            cmds.append(int(getattr(cmd, 'value', cmd)))
            wps.append(None if sensor_only else
                       cmap.get_waypoint(tf.location, project_to_road=True,
                                         lane_type=carla.LaneType.Driving))
        core.frenet_converter = FrenetConverter(pts)
        core.path_length = core.frenet_converter.get_path_length()
        widths, ws, hint = [], [], None
        for (x, y), wp in zip(pts, wps):
            widths.append(_road_width(wp) if wp is not None else (2.0, 2.0))
            s, _, _ = core.frenet_converter.world_to_frenet(x, y, 0, s_hint=hint)
            ws.append(s)
            hint = s
        core._road_width_s = np.array(ws)
        if sensor_only:
            # No HD map: the MPCC's fixed two-lane corridor; junctions from the
            # route's own commands (RoadOption LEFT/RIGHT/STRAIGHT = 1/2/3).
            core._road_widths = None
            self._route_s, self._route_cmds = np.array(ws), np.array(cmds)
            core._route_junction = np.array([cm in (1, 2, 3) for cm in cmds])
            j = core._route_junction
            self._junction_entries = [float(ws[i]) for i in range(len(j))
                                      if j[i] and (i == 0 or not j[i - 1])]
        else:
            core._road_widths = np.array(widths)
            core._route_junction = np.array([bool(wp.is_junction) if wp is not None else False
                                             for wp in wps])
            j = core._route_junction                # entries: the student's yield target
            self._junction_entries = [float(ws[i]) for i in range(len(j))
                                      if j[i] and (i == 0 or not j[i - 1])]
        core.racing_npcs = []
        core.current_s = 0.0
        core._ego_s_hint = 0.0
        core._initialize_mpc()
        core.mpc_reuse = bool(c['mpc_reuse'])
        core.obstacle_face = bool(c['obstacle_face'])
        core.obstacle_priority = bool(c['obstacle_priority'])
        if c['perception']:
            core._attach_perception_cameras()
        self.core = core

        if c['student']:
            from student.policy import StudentPolicy
            self.policy = StudentPolicy(c['student'], chunk_mode=c['student_chunk_mode'],
                                        support_sigma=c['support_sigma'],
                                        axes=c['student_axes'])
            self.policy.attach(core)
        self._planner = RoutePlanner(self._global_plan_world_coord)
        if self.rec_dir is not None:
            # The downsampled plan every agent gets: teachers that run their own
            # route planner (SimLingo) replay it offline.
            with open(os.path.join(self.rec_dir, 'route.json'), 'w') as f:
                json.dump({'nodes': [(tf.location.x, tf.location.y, tf.location.z,
                                      int(getattr(cm, 'value', cm)))
                                     for tf, cm in self._global_plan_world_coord]}, f)
        if sensor_only:
            from leaderboard.utils.route_manipulation import _get_latlon_ref
            from perception.ego_state import EgoStateEstimator
            from perception.lidar_obstacles import LidarObstacleTracker
            self._ego_est = EgoStateEstimator(*_get_latlon_ref(world), antenna_x=GNSS_X)
            self._lidar = LidarObstacleTracker(mount=tuple(c['lidar_mount']))
            self._tl = None
            if c['traffic_rules'] and not c['gt_lights']:
                from perception.traffic_lights import TrafficLightDetector
                self._tl = TrafficLightDetector(device=c['device'])
            self._last_applied = (0.0, 0.0, 0.0)          # throttle, brake, steer
            self._lanes = None
            if c['sparse_route']:
                from perception.lanes import LanePerception, CameraGeometry
                fx, fy, fyaw, ffov = ORION_CAMS[0][1:]
                self._lanes = LanePerception(c['yolop'], camera=CameraGeometry(
                    W=1600, H=900, fov_deg=ffov, x=fx, y=fy, z=1.60), device=c['device'])
                core.mpc_controller.n_lane_fn = core.lane_offset
            self._stop_entries_done = set()
            self._stop_sign_seen_t = -1e9
            # Stop-line context for the rules arm: every junction entry.
            core._stop_lines_s = list(self._junction_entries)
            # DIAGNOSTIC ONLY (_perception_check), never read by control: the
            # dense lane-centre route (true lane offset) and the lights whose
            # stop lines are on our route (true light state).
            self._diag_fc = None
            if c['sparse_route']:
                dp = []
                for tf, _ in self._dense_route:
                    p = (tf.location.x, tf.location.y)
                    if not dp or math.hypot(p[0] - dp[-1][0], p[1] - dp[-1][1]) >= 0.5:
                        dp.append(p)
                self._diag_fc = FrenetConverter(dp)
            self._index_traffic_rules(world)
            self._diag_signs = list(self._rule_signs)     # truth for recorded labels only
            if c['gt_lights']:                  # E4: true lights + stop signs drive the stop
                self._diag_lights = list(self._rule_lights)
            else:
                self._diag_lights, self._rule_lights = self._rule_lights, []
                self._rule_signs = []
            self._gt_last_moving = {}           # gt_obstacles: actor id -> last time moving
        elif c['traffic_rules']:
            self._index_traffic_rules(world)
            # Stop-line context for the behavior_rules arm (queued traffic).
            core._stop_lines_s = sorted([s for s, _ in self._rule_lights] +
                                        [s for s, _ in self._rule_signs])

    def _index_traffic_rules(self, world):
        """Stop lines of every traffic light / stop sign that lie on our route."""
        fc = self.core.frenet_converter
        for tl in world.get_actors().filter('traffic.traffic_light'):
            for wp in tl.get_stop_waypoints():
                loc = wp.transform.location
                s, d, _ = fc.world_to_frenet(loc.x, loc.y, 0)
                # (4 m on the sparse spline: it sags up to ~4 m on bends)
                if abs(d) < (4.0 if self.cfg['sparse_route'] else 2.0) \
                        and 0.0 < s < self.core.path_length:
                    self._rule_lights.append((float(s), tl))
        for st in world.get_actors().filter('traffic.stop'):
            x, y = _local_to_world(st.get_transform(), st.trigger_volume.location)
            s, d, _ = fc.world_to_frenet(x, y, 0)
            if abs(d) < 4.0 and 0.0 < s < self.core.path_length:
                self._rule_signs.append((float(s), st))
        self._rule_lights.sort(key=lambda r: r[0])
        self._rule_signs.sort(key=lambda r: r[0])

    # ── per tick ────────────────────────────────────────────────────────────

    def _refresh_obstacles(self):
        """All vehicles, walkers and static props near the ego -> core.racing_npcs
        (the env's ground-truth detector then projects them onto the route)."""
        world = CarlaDataProvider.get_world()
        ego = self.hero_actor.get_location()
        r = self.cfg['obstacle_radius_m']
        if self.cfg['record'] or self.tactical is not None:
            r = max(r, self.cfg['scene_radius_m'])
        alive = {}
        for a in world.get_actors():
            t = a.type_id
            if a.id == self.hero_actor.id or not (
                    t.startswith('vehicle.') or t.startswith('walker.pedestrian')
                    or t.startswith('static.prop')):
                continue
            if any(k in t for k in DRIVABLE_PROPS):
                continue        # decals you drive over (ControlLoss debris)
            loc = a.get_location()
            if loc.distance(ego) > r:
                continue
            npc = self._actors.get(a.id) or {'actor': a, 's': None}
            # Keep a Frenet hint per actor (the env's NPCs get theirs from the
            # NPC controller); unhinted lookups can jump branch on loops.
            hint = npc['s'] if npc['s'] is not None else self.core.current_s
            npc['s'], _, _ = self.core.frenet_converter.world_to_frenet(
                loc.x, loc.y, 0, s_hint=hint)
            alive[a.id] = npc
        self._actors = alive
        self.core.racing_npcs = list(alive.values())

    def _rule_stop_target(self):
        """Distance to the stop line we must stop at now, or None."""
        core = self.core
        s = core.current_s
        reach = 6.0 + core.current_speed ** 2 / (2 * RULE_DECEL) + 5.0
        for s_stop, tl in self._rule_lights:
            ds = s_stop - s
            if -1.0 < ds < reach and tl.state in (carla.TrafficLightState.Red,
                                                 carla.TrafficLightState.Yellow):
                # Already too close to stop comfortably on yellow: go through.
                if tl.state == carla.TrafficLightState.Yellow and \
                        core.current_speed ** 2 / (2 * max(ds, 0.1)) > 2 * RULE_DECEL:
                    continue
                return ds
        for s_stop, st in self._rule_signs:
            if st.id in self._stop_signs_done:
                continue
            ds = s_stop - s
            if -2.0 < ds < reach:
                if core.current_speed < 0.1:
                    self._stop_signs_done.add(st.id)    # full stop made
                    continue
                return ds
        return None

    def run_step(self, input_data, timestamp):
        if self.core is None:
            self._init_core()
        core = self.core
        t0 = time.perf_counter()
        self._ms = {}                    # per-module wall time this tick (trace ms_*)
        self.n_steps += 1
        core.current_step = self.n_steps

        if self.cfg['privileged']:
            core._update_vehicle_state()
            core._ego_s_hint = core.current_s
            if not self.cfg['perception']:
                self._refresh_obstacles()
            core._detect_obstacles()
        else:
            self._sense(input_data, timestamp)

        # Traffic rules enter the MPCC as a constraint: the stop line caps
        # every stage's speed by the braking envelope (MPCController.stop_s),
        # so the OCP plans the stop itself.
        if not self.cfg['traffic_rules']:
            stop_ds = None
        elif self.cfg['privileged'] or self.cfg['gt_lights']:
            stop_ds = self._rule_stop_target()
        else:
            t1 = time.perf_counter()
            stop_ds = self._sensor_stop_target(input_data, timestamp)
            self._ms['light'] = time.perf_counter() - t1
        core.mpc_controller.stop_s = (None if stop_ds is None
                                      else core.current_s + stop_ds)
        core.mpc_controller.stop_margin = STOP_LINE_MARGIN_M

        scene = None
        if self.tactical is None:
            core.mpc_controller.v_cap = None     # (a creep may have set it)
        t1 = time.perf_counter()
        if self.cfg['record'] or self.tactical is not None:
            scene = self._scene(stop_ds)
        if self.tactical is not None:
            dec = self.tactical.step(scene)
            go, dv = dec['go'], dec['dv']
            blocked = scene.get('_blocker') is not None
            core.gate_scale = 1.0 if go else 0.0
            core.tactical_wait = (not go) and blocked
            core.mpc_controller.v_cap = None if dv is None else max(core.current_speed + dv, 0.0)
            self.n_go += int(go and blocked)
            self.n_wait += int((not go) and blocked)
            self.n_slow += int(dv is not None)
            # yield: hold at the next junction entry (the light's stop machinery)
            ys = self._yield_target(dec['yield_'], scene, timestamp)
            if ys is not None and (stop_ds is None or ys < stop_ds):
                stop_ds = ys
                core.mpc_controller.stop_s = core.current_s + stop_ds
            self._ms['tactical'] = time.perf_counter() - t1
        creeping = self._update_creep(stop_ds)

        action = np.zeros(2)
        if self.policy is not None:
            action = self.policy.predict(core._get_observation())[0]
        t1 = time.perf_counter()
        u_nom_thr, u_nom_steer, thr, steer = core._compute_final_action(action)
        self._ms['mpc'] = time.perf_counter() - t1
        self._ms['ocp'] = getattr(core.mpc_controller, 'last_solve_s', 0.0)

        # Last resort only: the soft OCP still carrying speed into the line --
        # and, once (nearly) stopped there, no throttle: the slacked speed cap
        # let the OCP pulse throttle at v ~ 0.4 m/s and creep over the line.
        if stop_ds is not None and stop_ds - STOP_LINE_MARGIN_M < 2.0:
            if core.current_speed > 1.0:
                thr = -1.0
                self.n_rule_overrides += 1
            elif thr > -0.3:
                thr = -0.3                     # hold the brake

        if self.cfg['aeb'] and self._aeb_step(core):
            thr = -1.0
            self.n_aeb += 1

        control = carla.VehicleControl()
        if thr >= 0:
            control.throttle, control.brake = float(thr), 0.0
        else:
            control.throttle, control.brake = 0.0, float(-thr)
        control.steer = float(-steer)            # env convention (carlaEnv.step)
        if not self.cfg['privileged']:
            self._last_applied = (control.throttle, control.brake, control.steer)
        self.t_control.append(time.perf_counter() - t0)
        self._ms['total'] = self.t_control[-1]
        if self.rec_dir is not None:
            self._record(input_data, scene, (u_nom_thr, u_nom_steer), (thr, steer))
        if self.trace_dir:
            lead = core.mpc_controller.lead
            loc = self.hero_actor.get_location()
            # Nearest obstacle in world coordinates: lets a collision in
            # eval.json (which has x, y) be matched to the step and object.
            near = min(((a['actor'].get_location().distance(loc), a['actor'])
                        for a in self._actors.values() if a.get('actor') is not None),
                       default=(None, None), key=lambda p: p[0])
            self.trace.append(dict(
                t=round(timestamp, 2), x=round(loc.x, 2), y=round(loc.y, 2),
                near_m=None if near[0] is None else round(near[0], 1),
                near_type=None if near[1] is None else near[1].type_id,
                near_id=None if near[1] is None else near[1].id,
                gate=round(float(core.mpc_controller.gate_on), 2),
                s=round(core.current_s, 2), d=round(core.current_d, 2),
                v=round(core.current_speed, 2), thr=round(float(thr), 3),
                steer=round(float(steer), 3),
                stop_ds=None if stop_ds is None else round(stop_ds, 1),
                lead_ds=None if lead is None else round(lead[0] - core.current_s, 1),
                lead_v=None if lead is None else round(lead[1], 1),
                n_obs=int((core.selected_obstacles[:, 0] > -50).sum()),
                alpha=round(float(core.last_authority_info.get('alpha', 1.0)), 2),
                creep=int(creeping), stuck_t=round(self._stuck_t, 1),
                student_net=(None if self.tactical is None
                             else int(self.tactical.last.get('net', False))),
                **({} if self.tactical is None else dict(
                    st_go=round(float(self.tactical.last.get('p_go', 1.0)), 2),
                    st_yield=round(float(self.tactical.last.get('p_yield', 0.0)), 2),
                    st_dv=round(float(self.tactical.last.get('dv', 0.0)), 2))),
                mpc_reused=int(getattr(core.mpc_controller, 'last_reused', False)),
                ocp_fail=int(getattr(core.mpc_controller, 'consecutive_failures', 0)),
                light_hold=int(getattr(self, '_light_hold', None) is not None),
                aeb=int(self._aeb_active),
                n_cand=getattr(core, '_n_cand', None), n_near=getattr(core, '_n_near', None),
                **{f'ms_{k}': round(1000.0 * v, 2) for k, v in self._ms.items()},
                **(self._perception_check() if not self.cfg['privileged'] else {})))
        return control

    YIELD_MAX_S = 15.0       # longest the student may hold us at one entry ...
    YIELD_COOL_S = 10.0      # ... then its yield is ignored this long (no deadlock)

    def _yield_target(self, want, scene, timestamp):
        """Stop distance for the student's 'yield' (the next junction entry,
        like a red light), or None.  A yield held YIELD_MAX_S is dropped for
        YIELD_COOL_S: the student can delay us, never strand us."""
        e = scene.get('entry_ds')
        if not want or e is None or scene.get('in_junction') or e > 30.0 \
                or timestamp < self._yield_off_until:
            self._yield_t0 = None
            return None
        if self._yield_t0 is None:
            self._yield_t0 = timestamp
        if timestamp - self._yield_t0 > self.YIELD_MAX_S:
            self._yield_t0, self._yield_off_until = None, timestamp + self.YIELD_COOL_S
            return None
        self.n_yield += 1
        return e - 0.5

    def _aeb_hit(self, core):
        """Is an obstacle about to be hit?  Time-to-contact along the route
        below AEB_TTC_S (or closer than AEB_MIN_GAP_M while moving), AND the
        MPC's planned path at that time overlaps the obstacle's
        constant-velocity position laterally -- so a pass the OCP is already
        steering around does not brake (fallback: hold the current d)."""
        v = core.current_speed
        if v < 0.3:
            return False
        mpc = core.mpc_controller
        plan = None
        if getattr(mpc, 'consecutive_failures', 0) == 0:
            try:
                plan = mpc.get_predicted_trajectory()       # (N, 2): s, d per stage
            except Exception:
                plan = None
        dt = getattr(mpc, 'dt', 0.05)
        for (s_o, d_o), (vs, vd) in zip(core.selected_obstacles, core.selected_obstacle_vel):
            if s_o <= -50.0:
                continue
            gap = s_o - core.current_s - AEB_LEN_M
            if gap > 40.0 or s_o - core.current_s <= 0.0:
                continue
            closing = v - vs
            if gap > AEB_MIN_GAP_M and (closing <= 0.3 or gap / closing > AEB_TTC_S):
                continue
            t = 0.0 if gap <= 0.0 or closing <= 0.3 else gap / closing
            d_ego = core.current_d
            if plan is not None and len(plan):
                d_ego = float(plan[min(int(round(t / dt)), len(plan) - 1)][1])
            if abs(d_o + vd * t - d_ego) < AEB_LAT_M:
                return True
        return False

    def _aeb_step(self, core):
        """Debounced emergency brake: on after AEB_ON_N triggering ticks in a
        row, off after AEB_OFF_N clear ticks (or once stopped)."""
        if self._aeb_hit(core):
            self._aeb_on, self._aeb_off = self._aeb_on + 1, 0
        else:
            self._aeb_on, self._aeb_off = 0, self._aeb_off + 1
        if self._aeb_on >= AEB_ON_N:
            self._aeb_active = True
        elif self._aeb_active and (self._aeb_off >= AEB_OFF_N or core.current_speed < 0.3):
            self._aeb_active = False
        return self._aeb_active

    STUCK_V, MOVING_V = 0.3, 1.0
    CREEP_CONTEXT_M = 40.0      # a junction entry this close ahead = a queue / light: wait
    CREEP_DONE_M = 15.0         # moved this far since the creep began = past the blocker
    CREEP_COOLDOWN_S = 10.0

    def _update_creep(self, stop_ds, dt=0.05):
        """Deadlock escape (creep_after_s): stopped for creep_after_s with
        nothing legitimate to wait for -- no stop line held, no junction entry
        within CREEP_CONTEXT_M ahead (a queue at a light) -- then for at most
        creep_max_s: every stationary obstacle counts as passable (gate on,
        student 'wait' overridden), no follow cap, speed capped at creep_v, and
        enough throttle to break away from rest.  The CBF stays in the loop.
        Bench2Drive scores a route that times out (TickRuntime) or is blocked
        60 s near zero, whatever the reason."""
        core, c = self.core, self.cfg['creep_after_s']
        core.creep = False
        if c is None:
            return False
        v = core.current_speed
        if v < self.STUCK_V:
            self._stuck_t += dt
        elif v > self.MOVING_V:
            self._stuck_t = 0.0
        self._creep_cool = max(self._creep_cool - dt, 0.0)
        if self._creep_t > 0.0:                                     # creeping
            self._creep_t += dt
            if (self._creep_t > self.cfg['creep_max_s']
                    or core.current_s - self._creep_s0 > self.CREEP_DONE_M):
                self._creep_t, self._stuck_t = 0.0, 0.0
                self._creep_cool = self.CREEP_COOLDOWN_S
                return False
        else:
            waiting = stop_ds is not None or any(
                -1.0 < e - core.current_s < self.CREEP_CONTEXT_M
                for e in getattr(core, '_stop_lines_s', []) or [])
            if self._stuck_t < c or waiting or self._creep_cool > 0.0:
                return False
            self._creep_t, self._creep_s0 = dt, core.current_s
            self.n_creep_events += 1
        core.creep = True
        core.gate_scale, core.tactical_wait = 1.0, False
        cap = self.cfg['creep_v']
        vc = core.mpc_controller.v_cap
        core.mpc_controller.v_cap = cap if vc is None else min(vc, cap)
        self.n_creep_ticks += 1
        return True

    def _write_trace(self):
        import csv
        os.makedirs(self.trace_dir, exist_ok=True)
        # e.g. RouteScenario_1773_rep0_Town12_ParkedObstacle_1_25_... -> route, town, type
        name = self.route_name or f'route_{int(time.time())}'
        stem = os.path.join(self.trace_dir, '_'.join(name.split('_')[:5]))
        with open(stem + '.csv', 'w', newline='') as f:
            # union of keys in first-seen order (a column may start mid-route)
            cols = list(dict.fromkeys(k for r in self.trace for k in r))
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.trace)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            tr = self.trace
            t = [r['t'] for r in tr]
            fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            ax[0].plot(t, [r['s'] for r in tr], 'k-', label='route s')
            ax[0].set_ylabel('s [m]')
            ax[1].plot(t, [r['v'] for r in tr], 'k-', label='speed')
            ax[1].plot(t, [r['lead_ds'] if r['lead_ds'] is not None else float('nan') for r in tr],
                       'r.', ms=2, label='lead gap [m]')
            ax[1].plot(t, [r['stop_ds'] if r['stop_ds'] is not None else float('nan') for r in tr],
                       'b.', ms=2, label='stop line [m]')
            ax[1].set_ylim(-1, 30)
            ax[1].legend(fontsize=7)
            ax[1].set_ylabel('speed [m/s] / gaps')
            ax[2].plot(t, [r['d'] for r in tr], 'g-', label='lateral d (+right)')
            ax[2].plot(t, [r['thr'] for r in tr], 'm-', lw=0.7, label='throttle(+)/brake(-)')
            ax[2].legend(fontsize=7)
            ax[2].set_xlabel('game time [s]')
            for a in ax:
                a.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(stem + '.png', dpi=80)
            plt.close(fig)
        except Exception as e:
            print(f'[MPCCAgent] trace plot failed: {e}', flush=True)
        print(f'[MPCCAgent] trace -> {stem}.csv', flush=True)

    # ── sensor-only perception ──────────────────────────────────────────────

    def _sense(self, input_data, timestamp):
        core = self.core
        t1 = time.perf_counter()
        imu = input_data['IMU'][1]
        gnss = input_data['GPS'][1]
        speed = float(input_data['SPEED'][1]['speed'])
        x, y, yaw, v, yaw_rate = self._ego_est.update(
            gnss[:2], float(imu[-1]), speed, gyro_z=float(imu[5]))
        thr, brk, steer = self._last_applied
        core._update_vehicle_state_from(x, y, yaw, v, yaw_rate, thr, brk, steer)
        t2 = time.perf_counter()
        self._ms['ego'] = t2 - t1
        if self._lanes is not None:
            self._update_lane(input_data['CAM_FRONT'][1][:, :, :3], x, y, yaw)
            # YOLOP network alone (0 on ticks it is skipped) vs fit + projection
            self._ms['yolop_net'] = self._lanes.last_net_s
            self._ms['lane'] = time.perf_counter() - t2
        t2 = time.perf_counter()
        if self.cfg['gt_obstacles']:
            tracks = self._gt_tracks(timestamp)
        else:
            tracks = self._lidar.update(input_data['LIDAR'][1], (x, y, yaw), timestamp)
        core._set_obstacles_from_tracks(tracks)
        self._ms['lidar'] = time.perf_counter() - t2
        if self.dump_dir and self.n_steps % DUMP_EVERY == 0:
            self._dump_sensors(input_data, x, y, yaw, v, timestamp)

    def _gt_tracks(self, t):
        """E4 gt_obstacles: the true actors in the LiDAR tracker's output format
        (perception/lidar_obstacles.py), so everything downstream is unchanged:
        box centre, true velocity, box outline as 'pts' (corners + edge
        midpoints: long objects still get their length points), true type,
        stopped_for = time since last moving (99 = never seen moving)."""
        hero = self.hero_actor
        ego = hero.get_location()
        r = self.cfg['obstacle_radius_m']
        out = []
        for a in CarlaDataProvider.get_world().get_actors():
            ty = a.type_id
            if a.id == hero.id or not (ty.startswith('vehicle.') or ty.startswith('walker.pedestrian')
                                       or ty.startswith('static.prop')):
                continue
            if any(k in ty for k in DRIVABLE_PROPS):
                continue
            tf = a.get_transform()
            if tf.location.distance(ego) > r:
                continue
            bb = a.bounding_box
            yaw = math.radians(tf.rotation.yaw)
            cy, sy = math.cos(yaw), math.sin(yaw)
            cx = tf.location.x + cy * bb.location.x - sy * bb.location.y
            cyy = tf.location.y + sy * bb.location.x + cy * bb.location.y
            ex, ey = bb.extent.x, bb.extent.y
            pts = [(cx + cy * u * ex - sy * w * ey, cyy + sy * u * ex + cy * w * ey)
                   for u, w in ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))]
            vel = a.get_velocity()
            if math.hypot(vel.x, vel.y) > 1.0:
                self._gt_last_moving[a.id] = t
            lm = self._gt_last_moving.get(a.id)
            out.append(dict(id=a.id, x=cx, y=cyy, vx=vel.x, vy=vel.y,
                            length=2 * ex, width=2 * ey, height=2 * bb.extent.z,
                            type=ty, pts=pts, stopped_for=99.0 if lm is None else t - lm))
        return out

    def _dump_sensors(self, input_data, x, y, yaw, v, t):
        """Raw on-board inputs for the offline efficiency benchmark
        (tools/b2d/bench_stack.py): front image as lossless PNG, LiDAR points,
        estimated pose."""
        import cv2
        name = '_'.join((self.route_name or 'route').split('_')[:3])
        stem = os.path.join(self.dump_dir, f'{name}_{self.n_steps:06d}')
        cv2.imwrite(stem + '.png', input_data['CAM_FRONT'][1][:, :, :3])
        np.savez_compressed(stem + '.npz', lidar=np.asarray(input_data['LIDAR'][1], np.float32),
                            pose=np.array([x, y, yaw, v, t], np.float64))

    LANE_JUMP_M = 1.75        # half a lane: a fit further from the tracked lane is rejected
    LANE_REACQUIRE = 10       # consecutive rejected YOLOP frames (~1 s) -> re-anchor to route

    def _update_lane(self, img, ex, ey, yaw):
        """YOLOP lane centre (vehicle frame) -> offset from the spline route,
        sampled along it; inside junctions the spline itself is the reference
        (no lane markings there)."""
        core = self.core
        # Track ONE lane: search where the tracked lane's centre should be now
        # (vehicle-frame y = its offset from the route - ours), so it keeps its
        # identity while we pass beside it.  Re-anchor to the route (y = -d)
        # on route lane-change segments (RoadOption 5/6), at start, and after
        # ~1 s of rejected fits.
        i = min(int(np.searchsorted(self._route_s, core.current_s + 10.0)), len(self._route_cmds) - 1)
        lane_change = self._route_cmds[i] in (5, 6)
        lost = getattr(self, '_lane_rejects', 0) >= self.LANE_REACQUIRE
        if getattr(core, '_lane_s', None) is None or lane_change or lost:
            ref0 = -core.current_d
        else:
            ref0 = core.lane_offset(core.current_s) - core.current_d
        est = self._lanes.update(img, ref0=ref0)
        self._lane_valid = bool(est.get('valid'))
        if not self._lane_valid or not est.get('fresh'):
            return
        c0, c1, c2 = est['coeffs']
        x_near = max(est['x_min'], 6.0)
        if abs(c0 + c1 * x_near + c2 * x_near ** 2 - ref0) > self.LANE_JUMP_M:
            self._lane_rejects = getattr(self, '_lane_rejects', 0) + 1
            self._lane_valid = False
            return
        self._lane_rejects = 0
        cy, sy = math.cos(yaw), math.sin(yaw)
        ss, dd = [], []
        for xv in np.arange(max(est['x_min'], 4.0), min(est['x_max'], 35.0), 2.5):
            yv = c0 + c1 * xv + c2 * xv * xv
            wx, wy = ex + cy * xv - sy * yv, ey + sy * xv + cy * yv
            s_q, d_q, _ = core.frenet_converter.world_to_frenet(wx, wy, 0, s_hint=core.current_s + xv)
            ss.append(s_q)
            # Bounded by how wrong the sparse route can be here (route_sag_bound)
            j = min(int(np.searchsorted(self._route_s, s_q)), len(self._lane_bound) - 1)
            b = self._lane_bound[j]
            dd.append(0.0 if core._in_junction(s_q) else float(np.clip(d_q, -b, b)))
        if len(ss) >= 2 and np.all(np.diff(ss) > 0):
            core.set_lane_offsets(ss, dd)

    def _perception_check(self):
        """DIAGNOSTIC ONLY (trace), never used for control: the nearest
        object ahead in our lane by perception vs by ground truth, and the
        pose estimate's error -- to validate the sensor-only stack."""
        core = self.core
        out = {}
        loc = self.hero_actor.get_location()
        out['pose_err'] = round(math.hypot(self._ego_est.x - loc.x, self._ego_est.y - loc.y), 2)
        per = [a for a in getattr(core, '_perceived', []) if abs(a['d']) < 2.0 and 0 < a['ds'] < 40]
        p = min(per, key=lambda a: a['ds']) if per else None
        out['per_ds'] = None if p is None else p['ds']
        out['per_d'] = None if p is None else round(p['d'], 2)
        out['per_type'] = None if p is None else p['type']
        # its track: id (re-segmentations show as id changes), along-route /
        # lateral speed and stopped time as the gate / follow logic saw them
        for k in ('id', 'vs', 'vd', 'stopped_for', 'length'):
            out[f'per_{k}'] = None if p is None else p.get(k)
        # GT relative to the TRUE ego position (not the estimate), measured to
        # the object's NEAR face (LiDAR sees the face, not the centre), lateral
        # in the same lane frame as perception.
        s_ego, _, _ = core.frenet_converter.world_to_frenet(loc.x, loc.y, 0, s_hint=core.current_s)
        gt = []
        anyn = None          # nearest actor in ANY direction: matches eval.json collision ids
        for a in CarlaDataProvider.get_world().get_actors():
            t = a.type_id
            if a.id == self.hero_actor.id or not (t.startswith('vehicle.') or t.startswith('walker.')
                                                  or t.startswith('static.prop')):
                continue
            al = a.get_location()
            dist = al.distance(loc)
            if dist > 45.0:
                continue
            s_o, d_o, _ = core.frenet_converter.world_to_frenet(al.x, al.y, 0, s_hint=s_ego)
            if anyn is None or dist < anyn[0]:
                v = a.get_velocity()
                anyn = (dist, a.id, t, s_o - s_ego, d_o - core.current_d, math.hypot(v.x, v.y))
            d_o -= core.lane_offset(s_o)
            ds = s_o - s_ego - a.bounding_box.extent.x
            if abs(d_o) < 2.0 and 0 < ds < 40:
                gt.append((ds, round(d_o, 2), t))
        # (relative s / d of its CENTRE from the true ego, and its true speed)
        for k, i in (('m', 0), ('id', 1), ('type', 2), ('ds', 3), ('dd', 4), ('v', 5)):
            val = None if anyn is None else anyn[i]
            out[f'any_{k}'] = round(val, 2) if isinstance(val, float) else val
        g = min(gt) if gt else None
        out['gt_ds'] = None if g is None else round(g[0], 2)
        out['gt_d'] = None if g is None else g[1]
        out['gt_type'] = None if g is None else g[2]
        out['light'] = (self._tl.last.get('light') if self._tl is not None else None)
        # True state of the next light on our route whose stop line is <= 60 m
        # ahead (simulator; diagnostic only).  (hero.get_traffic_light() is
        # set only inside the trigger box -- it read 'none' on every tick.)
        nxt = [(s_l - s_ego, tl) for s_l, tl in getattr(self, '_diag_lights', [])
               if -1.0 < s_l - s_ego < 60.0]
        out['gt_light'] = (str(min(nxt, key=lambda r: r[0])[1].state).split('.')[-1].lower()
                           if nxt else None)
        out['n_route_lights'] = len(getattr(self, '_diag_lights', []))   # GT index sanity
        # Our lateral offset from the ROUTE's lane centre, estimated vs true.
        # True = offset from the dense lane-centre route (not the nearest map
        # lane: that switches lanes with us when we pass, hiding lane-tracking
        # failures).
        out['lane_valid'] = bool(getattr(self, '_lane_valid', False))
        est_off = core.current_d - core.lane_offset(core.current_s)
        try:
            if getattr(self, '_diag_fc', None) is not None:
                self._diag_s, true_off, _ = self._diag_fc.world_to_frenet(
                    loc.x, loc.y, 0, s_hint=getattr(self, '_diag_s', None))
            else:
                wp = CarlaDataProvider.get_map().get_waypoint(loc, project_to_road=True,
                                                              lane_type=carla.LaneType.Driving)
                wl = wp.transform.location
                wyaw = math.radians(wp.transform.rotation.yaw)
                true_off = -math.sin(wyaw) * (loc.x - wl.x) + math.cos(wyaw) * (loc.y - wl.y)
            out['lane_off_est'] = round(est_off, 2)
            out['lane_off_true'] = round(true_off, 2)
        except Exception:
            pass
        return out

    def _sensor_stop_target(self, input_data, timestamp):
        """Distance to where we must stop now, from the camera alone: a red
        (or still-stoppable yellow) light ahead stops us at the next junction
        entry on the route; a stop sign seen in the last 3 s means one full
        stop at the next entry."""
        core = self.core
        if self._tl is None:
            return None
        # Lights and stop signs govern junction entries only, and the route
        # says where those are: outside light_gate_m of the next one, skip the
        # detector (efficiency; also no false 'lights' from amber construction
        # lamps on open road).  A stop sign seen in the last 3 s still holds.
        g = self.cfg['light_gate_m']
        if g is not None and self._light_hold is None and timestamp - self._stop_sign_seen_t >= 3.0:
            s = core.current_s
            if not any(-1.0 < e - s < g for e in self._junction_entries):
                self._tl.history.clear()
                self._tl.last = {}
                return None
        det = self._tl.detect(input_data['CAM_FRONT'][1][:, :, :3])
        self.n_light_runs += 1
        self._ms['yolo_net'] = self._tl.last_net_s
        if det['stop_sign']:
            self._stop_sign_seen_t = timestamp
        s, v = core.current_s, core.current_speed
        light = det['light']
        hold = self._light_hold
        if hold is not None:
            # RED HOLD: once stopping for a light, keep the stop until the
            # light is SEEN green -- not merely until it stops being seen red.
            # A single 'unknown' / missed frame used to drop the target, the
            # OCP throttled, the car crept past entry+1 m and the light was
            # forgotten (tsmoke: 3 of 6 red-light infractions).  Green must
            # last LIGHT_GREEN_S with no red/yellow in between: a 0.1 s false
            # 'green' burst on a true red released 3-in-a-row (sbase_tlights
            # 2373, 3869).  'none'/'unknown' neither count nor reset.
            # Released also once past the entry (already committed) or after
            # LIGHT_HOLD_MAX_S without red/yellow (light out of view at the
            # line, Town03/07 style).
            if light in ('red', 'yellow'):
                hold['t'] = timestamp
                hold['green'], hold['g0'] = 0, None
            elif light == 'green':
                hold['green'] += 1
                if hold['g0'] is None:
                    hold['g0'] = timestamp
            green_ok = hold['green'] >= LIGHT_GREEN_N \
                and timestamp - hold['g0'] >= LIGHT_GREEN_S
            if green_ok or s > hold['e'] + LIGHT_PASSED_M \
                    or timestamp - hold['t'] > LIGHT_HOLD_MAX_S:
                self._light_hold = None
                self.n_light_holds += 1
                # a green-released entry is not re-held on a stray 'red'
                # (the cross street's light) for LIGHT_REHOLD_S
                self._light_released = (hold['e'], timestamp)
            else:
                return hold['e'] - s - 0.5
        reach = 6.0 + v ** 2 / (2 * RULE_DECEL) + 10.0
        nxt = [e for e in self._junction_entries if -1.0 < e - s < reach]
        if not nxt:
            return None
        e = nxt[0]
        ds = e - s - 0.5
        rel = getattr(self, '_light_released', None)
        fresh = rel is None or rel[0] != e or timestamp - rel[1] > LIGHT_REHOLD_S
        if fresh and (light == 'red' or (light == 'yellow'
                                         and v ** 2 / (2 * max(ds, 0.1)) <= 2 * RULE_DECEL)):
            self._light_hold = dict(e=e, t=timestamp, green=0, g0=None)
            return ds
        if timestamp - self._stop_sign_seen_t < 3.0 and e not in self._stop_entries_done:
            if v < 0.1 and ds < 3.0:
                self._stop_entries_done.add(e)          # full stop made
                return None
            return ds
        return None

    def _scene(self, stop_ds):
        """The tactical scene (src/student/tactical.py) for this tick: every
        kept actor in ego-relative Frenet terms, and the route ahead in the ego
        frame (for labelling ORION's plan against)."""
        core = self.core
        if self.cfg['privileged']:
            tf = self.hero_actor.get_transform()
            yaw = math.radians(tf.rotation.yaw)
            ex, ey = tf.location.x, tf.location.y
        else:                       # the estimated pose, never the simulator's
            ex, ey, yaw = self._ego_est.x, self._ego_est.y, self._ego_est.yaw
        cy, sy = math.cos(yaw), math.sin(yaw)
        acts = []
        if not self.cfg['privileged']:
            acts = list(getattr(core, '_perceived', []))
        for npc in (self._actors.values() if self.cfg['privileged'] else []):
            a = npc.get('actor')
            if a is None or npc.get('s') is None:
                continue
            loc, vel = a.get_location(), a.get_velocity()
            s_o, d_o, _ = core.frenet_converter.world_to_frenet(loc.x, loc.y, 0, s_hint=npc['s'])
            s2, d2, _ = core.frenet_converter.world_to_frenet(
                loc.x + 0.5 * vel.x, loc.y + 0.5 * vel.y, 0, s_hint=s_o)
            if math.hypot(vel.x, vel.y) > 1.0:
                npc['last_moving_step'] = self.n_steps
            lm = npc.get('last_moving_step')
            acts.append(dict(type=a.type_id, ds=round(s_o - core.current_s, 2), d=round(d_o, 2),
                             vs=round((s2 - s_o) / 0.5, 2), vd=round((d2 - d_o) / 0.5, 2),
                             stopped_for=99.0 if lm is None else round((self.n_steps - lm) * 0.05, 2)))
        route_fl = []
        for k in range(1, 21):
            s_k = min(core.current_s + 2.0 * k, core.path_length)
            x, y, _ = core.frenet_converter.frenet_to_world(s_k, core.lane_offset(s_k), 0.0)
            dx, dy = x - ex, y - ey
            route_fl.append([round(cy * dx + sy * dy, 2), round(sy * dx - cy * dy, 2)])
        lead = core.mpc_controller.lead
        jd = None
        if core._route_junction is not None:
            ahead = (core._road_width_s > core.current_s) & core._route_junction
            if ahead.any():
                jd = float(core._road_width_s[ahead].min() - core.current_s)
        d_lane = core.current_d - core.lane_offset(core.current_s)
        ent = [e - core.current_s for e in getattr(self, '_junction_entries', [])
               if e - core.current_s > 0.0]
        sc = dict(v=round(core.current_speed, 3), d=round(d_lane, 3),
                  lead_ds=None if lead is None else round(lead[0] - core.current_s, 2),
                  lead_v=None if lead is None else round(lead[1], 2),
                  stop_ds=None if stop_ds is None else round(stop_ds, 2),
                  junction_ds=jd, entry_ds=round(min(ent), 2) if ent else None,
                  in_junction=bool(core._in_junction(core.current_s)),
                  actors=acts, route_fl=route_fl)
        from student.tactical import blocker_info
        sc['_blocker'] = blocker_info(sc)
        return sc

    def _record(self, input_data, scene, u_nom, u_final):
        """One frame in tools/orion/orion_infer.py's episode layout, plus the
        tactical scene and the MPCC's own action for the student."""
        import cv2
        step = self.n_steps
        for cam, *_ in ORION_CAMS:
            img = input_data[cam][1][:, :, :3]
            cv2.imwrite(os.path.join(self.rec_dir, cam, f'{step:06d}.jpg'), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if self.cfg['record_simlingo']:
            cam = SIMLINGO_CAM[0]
            # its agent re-encodes with cv2's default JPEG quality (95)
            cv2.imwrite(os.path.join(self.rec_dir, cam, f'{step:06d}.jpg'),
                        input_data[cam][1][:, :, :3], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        imu = input_data['IMU'][1]
        loc = self.hero_actor.get_location()
        (_, cmd), (near, _) = self._planner.run_step(np.array([loc.x, loc.y]))
        sc = {k: v for k, v in scene.items() if not k.startswith('_')}
        rec = dict(step=step, x=float(loc.x), y=float(loc.y), compass=float(imu[-1]),
                   speed=float(input_data['SPEED'][1]['speed']),
                   accel=[float(v) for v in imu[:3]], gyro=[float(v) for v in imu[3:6]],
                   command=int(getattr(cmd, 'value', cmd)),
                   near_xy=[float(near[0]), float(near[1])],
                   s=float(self.core.current_s), d=float(self.core.current_d),
                   alpha=float(self.core.current_alpha),
                   u_nom=[float(u_nom[0]), float(u_nom[1])],
                   u_final=[float(u_final[0]), float(u_final[1])], scene=sc,
                   gt=self._gt_scene(), student=(None if self.tactical is None
                                                 else dict(self.tactical.last)))
        self.rec_meta.write(json.dumps(rec) + '\n')
        self.rec_obs.append(np.asarray(self.core._get_observation(), dtype=np.float32))

    def _gt_scene(self):
        """Simulator truth for this frame, written to the recording for LABELS
        only (the privileged expert teacher, masking teacher stops at red
        lights); never a student feature.  Frenet in the TRUE lane-centre frame
        (the dense route) from the TRUE ego pose: actors as in the scene
        (ds, d, vs, vd, stopped_for), the next light on the route and its
        state, the next stop sign, and the route's curvature ahead."""
        core = self.core
        fc = getattr(self, '_diag_fc', None) or core.frenet_converter
        loc = self.hero_actor.get_location()
        s_e, d_e, _ = fc.world_to_frenet(loc.x, loc.y, 0, s_hint=getattr(self, '_gt_s', None))
        self._gt_s = s_e
        moving = self.__dict__.setdefault('_gt_rec_moving', {})
        acts = []
        for a in CarlaDataProvider.get_world().get_actors():
            ty = a.type_id
            if a.id == self.hero_actor.id or not (ty.startswith('vehicle.') or ty.startswith('walker.')
                                                  or ty.startswith('static.prop')):
                continue
            if any(k in ty for k in DRIVABLE_PROPS):
                continue
            al = a.get_location()
            if al.distance(loc) > self.cfg['scene_radius_m']:
                continue
            vel = a.get_velocity()
            s_o, d_o, _ = fc.world_to_frenet(al.x, al.y, 0, s_hint=s_e)
            s2, d2, _ = fc.world_to_frenet(al.x + 0.5 * vel.x, al.y + 0.5 * vel.y, 0, s_hint=s_o)
            if math.hypot(vel.x, vel.y) > 1.0:
                moving[a.id] = self.n_steps
            lm = moving.get(a.id)
            acts.append(dict(type=ty, ds=round(s_o - s_e, 2), d=round(d_o, 2),
                             vs=round((s2 - s_o) / 0.5, 2), vd=round((d2 - d_o) / 0.5, 2),
                             len=round(2 * a.bounding_box.extent.x, 2),
                             stopped_for=99.0 if lm is None else round((self.n_steps - lm) * 0.05, 2)))
        lights = getattr(self, '_diag_lights', None) or getattr(self, '_rule_lights', [])
        signs = getattr(self, '_diag_signs', None) or getattr(self, '_rule_signs', [])
        s_c = core.current_s                    # light / sign stop lines are on our route's s
        nl = [(s_l - s_c, tl) for s_l, tl in lights if -1.0 < s_l - s_c < 80.0]
        nl = min(nl, key=lambda r: r[0]) if nl else None
        ns = [s_l - s_c for s_l, _ in signs if -2.0 < s_l - s_c < 80.0]
        kap = 0.0                               # max |curvature| 5-40 m ahead (3-point)
        pts = [fc.frenet_to_world(min(s_e + k, fc.get_path_length()), 0.0, 0.0)[:2]
               for k in range(0, 45, 5)]
        for p0, p1, p2 in zip(pts, pts[1:], pts[2:]):
            a_ = math.dist(p0, p1); b_ = math.dist(p1, p2); c_ = math.dist(p0, p2)
            area2 = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
            if a_ * b_ * c_ > 1e-6:
                kap = max(kap, 2.0 * area2 / (a_ * b_ * c_))
        # The TRUE lane centre ahead in the true ego frame: teacher plans are
        # judged against it (pull-out = go), not against the perceived route,
        # which is off by > half a lane on ~1/3 of sbase ticks (n1/n2 analysis).
        tf = self.hero_actor.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        route_fl = []
        for k in range(1, 21):
            x, y, _ = fc.frenet_to_world(min(s_e + 2.0 * k, fc.get_path_length()), 0.0, 0.0)
            dx, dy = x - loc.x, y - loc.y
            route_fl.append([round(cy_ * dx + sy_ * dy, 2), round(sy_ * dx - cy_ * dy, 2)])
        return dict(d=round(d_e, 2), actors=acts, route_fl=route_fl,
                    light_ds=None if nl is None else round(nl[0], 2),
                    light=None if nl is None else str(nl[1].state).split('.')[-1].lower(),
                    sign_ds=round(min(ns), 2) if ns else None, kappa=round(kap, 4))

    def _close_recording(self):
        if self.rec_meta is not None:
            self.rec_meta.close()
            self.rec_meta = None
            if self.rec_obs:
                np.save(os.path.join(self.rec_dir, 'obs.npy'), np.stack(self.rec_obs))
            with open(os.path.join(self.rec_dir, 'summary.json'), 'w') as f:
                json.dump({'route': self.route_name, 'frames': len(self.rec_obs),
                           'outcome': None, 'driver': 'mpcc' if self.tactical is None
                           else 'tactical', 'flags': self.cfg}, f, indent=2, default=str)
            print(f'[MPCCAgent] recorded {len(self.rec_obs)} frames -> {self.rec_dir}', flush=True)

    def destroy(self):
        # Also runs when setup() failed half-way: nothing may be assumed set.
        try:
            self._close_recording()
        except Exception as e:
            print(f'[MPCCAgent] closing recording failed: {e}', flush=True)
        if getattr(self, 'tactical', None) is not None:
            print(f'[MPCCAgent] tactical: go {self.n_go} / wait {self.n_wait} ticks with a '
                  f'blocker, speed cap {self.n_slow} ticks, yield {self.n_yield} ticks', flush=True)
        if getattr(self, 'trace_dir', None) and getattr(self, 'trace', None):
            try:
                self._write_trace()
            except Exception as e:
                print(f'[MPCCAgent] trace write failed: {e}', flush=True)
        if getattr(self, 'core', None) is not None:
            n = len(self.t_control)
            if n:
                print(f'[MPCCAgent] {n} steps, control time median '
                      f'{1000 * np.median(self.t_control):.1f} ms, p95 '
                      f'{1000 * np.percentile(self.t_control, 95):.1f} ms', flush=True)
            print(f'[MPCCAgent] rule-layer brake overrides: {self.n_rule_overrides} '
                  f'(stops the MPCC failed to plan); emergency-brake ticks {self.n_aeb}', flush=True)
            if not self.cfg.get('privileged', True):
                print(f'[MPCCAgent] light detector ran {self.n_light_runs}/{n} ticks; '
                      f'creep {self.n_creep_events} events, {self.n_creep_ticks} ticks; '
                      f'red holds released {self.n_light_holds}', flush=True)
            if self.policy is not None:
                print(f'[MPCCAgent] student counters {self.policy.total}', flush=True)
            try:
                cams = getattr(self.core, '_perception_cameras', None) or {}
                for cam in (cams.values() if isinstance(cams, dict) else cams):
                    cam.destroy()
            except Exception:
                pass
        self.core = None
