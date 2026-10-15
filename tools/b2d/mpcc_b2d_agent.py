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
  gate_mode ('pass_slow': overtake only stationary/slow cars in our lane)
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
                gate_mode='pass_slow')

# Comfortable stopping deceleration for the rule layer, m/s^2.
RULE_DECEL = 4.0
STOP_LINE_MARGIN_M = 1.5


def get_entry_point():
    return 'MPCCAgent'


def _local_to_world(tf, loc):
    """(x, y) of a point given in an actor's local frame (trigger volumes)."""
    yaw = math.radians(tf.rotation.yaw)
    return (tf.location.x + loc.x * math.cos(yaw) - loc.y * math.sin(yaw),
            tf.location.y + loc.x * math.sin(yaw) + loc.y * math.cos(yaw))


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
        if path_to_conf_file and '+' in path_to_conf_file:
            path_to_conf_file, self.route_name = path_to_conf_file.split('+', 1)
        if path_to_conf_file:
            # Fail loudly: a relative path resolved from the Bench2Drive dir
            # once silently fell back to the defaults (the 'nofollow' smoke
            # run was really the default config).
            if not os.path.isfile(path_to_conf_file):
                raise FileNotFoundError(f'agent config {path_to_conf_file!r} not found')
            with open(path_to_conf_file) as f:
                self.cfg.update(json.load(f))
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
        self.t_control = []
        # Per-step trace -> <B2D_TRACE_DIR>/route_<n>.csv + .png (run_b2d.sh sets it)
        self.trace = []
        self.trace_dir = os.environ.get('B2D_TRACE_DIR')

    def sensors(self):
        # Ground-truth obstacles come from the world; the speedometer keeps the
        # leaderboard's sensor interface (and its sync wait) happy.
        return [{'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'SPEED'}]

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
            gate_mode=c['gate_mode'])
        # Stop lines need the speed cap to bind even with follow/predict off.
        core._binding_v_slack = True
        random.setstate(rng_state)
        core.vehicle = self.hero_actor
        core.map = cmap

        # Route -> Frenet path + corridor table, as _generate_path does.
        pts, wps = [], []
        for tf, _ in self._dense_route:
            p = (tf.location.x, tf.location.y)
            if pts and math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) < 0.5:
                continue
            pts.append(p)
            wps.append(cmap.get_waypoint(tf.location, project_to_road=True,
                                         lane_type=carla.LaneType.Driving))
        core.frenet_converter = FrenetConverter(pts)
        core.path_length = core.frenet_converter.get_path_length()
        widths, ws, hint = [], [], None
        for (x, y), wp in zip(pts, wps):
            widths.append(_road_width(wp) if wp is not None else (2.0, 2.0))
            s, _, _ = core.frenet_converter.world_to_frenet(x, y, 0, s_hint=hint)
            ws.append(s)
            hint = s
        core._road_widths = np.array(widths)
        core._road_width_s = np.array(ws)
        core.racing_npcs = []
        core.current_s = 0.0
        core._ego_s_hint = 0.0
        core._initialize_mpc()
        if c['perception']:
            core._attach_perception_cameras()
        self.core = core

        if c['student']:
            from student.policy import StudentPolicy
            self.policy = StudentPolicy(c['student'], chunk_mode=c['student_chunk_mode'],
                                        support_sigma=c['support_sigma'],
                                        axes=c['student_axes'])
            self.policy.attach(core)
        if c['traffic_rules']:
            self._index_traffic_rules(world)

    def _index_traffic_rules(self, world):
        """Stop lines of every traffic light / stop sign that lie on our route."""
        fc = self.core.frenet_converter
        for tl in world.get_actors().filter('traffic.traffic_light'):
            for wp in tl.get_stop_waypoints():
                loc = wp.transform.location
                s, d, _ = fc.world_to_frenet(loc.x, loc.y, 0)
                if abs(d) < 2.0 and 0.0 < s < self.core.path_length:
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
        alive = {}
        for a in world.get_actors():
            t = a.type_id
            if a.id == self.hero_actor.id or not (
                    t.startswith('vehicle.') or t.startswith('walker.pedestrian')
                    or t.startswith('static.prop')):
                continue
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
        self.n_steps += 1
        core.current_step = self.n_steps

        core._update_vehicle_state()
        core._ego_s_hint = core.current_s
        if not self.cfg['perception']:
            self._refresh_obstacles()
        core._detect_obstacles()

        # Traffic rules enter the MPCC as a constraint: the stop line caps
        # every stage's speed by the braking envelope (MPCController.stop_s),
        # so the OCP plans the stop itself.
        stop_ds = self._rule_stop_target() if self.cfg['traffic_rules'] else None
        core.mpc_controller.stop_s = (None if stop_ds is None
                                      else core.current_s + stop_ds)
        core.mpc_controller.stop_margin = STOP_LINE_MARGIN_M

        action = np.zeros(2)
        if self.policy is not None:
            action = self.policy.predict(core._get_observation())[0]
        _, _, thr, steer = core._compute_final_action(action)

        # Last resort only: the soft OCP still carrying speed into the line.
        if stop_ds is not None and stop_ds - STOP_LINE_MARGIN_M < 2.0 \
                and core.current_speed > 1.0:
            thr = -1.0
            self.n_rule_overrides += 1

        control = carla.VehicleControl()
        if thr >= 0:
            control.throttle, control.brake = float(thr), 0.0
        else:
            control.throttle, control.brake = 0.0, float(-thr)
        control.steer = float(-steer)            # env convention (carlaEnv.step)
        self.t_control.append(time.perf_counter() - t0)
        if self.trace_dir:
            lead = core.mpc_controller.lead
            self.trace.append(dict(
                t=round(timestamp, 2), s=round(core.current_s, 2), d=round(core.current_d, 2),
                v=round(core.current_speed, 2), thr=round(float(thr), 3),
                steer=round(float(steer), 3),
                stop_ds=None if stop_ds is None else round(stop_ds, 1),
                lead_ds=None if lead is None else round(lead[0] - core.current_s, 1),
                lead_v=None if lead is None else round(lead[1], 1),
                n_obs=int((core.selected_obstacles[:, 0] > -50).sum()),
                alpha=round(float(core.last_authority_info.get('alpha', 1.0)), 2)))
        return control

    def _write_trace(self):
        import csv
        os.makedirs(self.trace_dir, exist_ok=True)
        # e.g. RouteScenario_1773_rep0_Town12_ParkedObstacle_1_25_... -> route, town, type
        name = self.route_name or f'route_{int(time.time())}'
        stem = os.path.join(self.trace_dir, '_'.join(name.split('_')[:5]))
        with open(stem + '.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.trace[0]))
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

    def destroy(self):
        # Also runs when setup() failed half-way: nothing may be assumed set.
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
                  f'(stops the MPCC failed to plan)', flush=True)
            if self.policy is not None:
                print(f'[MPCCAgent] student counters {self.policy.total}', flush=True)
            try:
                cams = getattr(self.core, '_perception_cameras', None) or {}
                for cam in (cams.values() if isinstance(cams, dict) else cams):
                    cam.destroy()
            except Exception:
                pass
        self.core = None
