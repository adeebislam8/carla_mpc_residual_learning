"""
Scripted hard scenarios (Bench2Drive-style), one event per episode.

The default episode -- slow NPCs in the ego's own lane, nothing oncoming --
is near-saturated for the fixed MPCC+CBF (~99% success at 150 m).  These are
the vehicle interactions where a short-horizon MPCC with an obstacle-blind
overtake target is known to be weak, and where the ORION teacher is strongest
on Bench2Drive (Overtaking 71%, Emergency Brake 78% success):

  hard_brake        lead car cruises, then brakes hard to a stop when the ego
                    follows within ~12 m; holds 3 s, drives on   (HardBreakRoute)
  stopped           a parked car (variant 'parked') or a 3-car pile-up
                    (variant 'accident') blocking the ego lane    (ParkedObstacle /
                                                                  Accident)
  stopped_oncoming  'accident' + a stream of traffic in the passing lane:
                    oncoming if that lane runs the other way, else same-
                    direction traffic to merge behind             (*TwoWays)
  cut_in            a car parked beside the lane pulls into the ego lane
                    when the ego is ~15 m behind it              (ParkingCutIn)

Dense background traffic (S5) needs no code: --npc-min/--npc-max.

Scenario actors live in env.racing_npcs like any NPC (so ground-truth
detection, perception and the CBF see them), with extra keys read by
CarlaMPCEnv._frenet_follow_controller:
  d_target    lateral offset the NPC tracks (default 0, the route centreline)
  direction   +1 along the route, -1 against it (oncoming)
  hold        brake fully when target speed is 0 (parked cars stay put)
  hard_brake  brake 1.0 instead of the usual <= 0.5 when slowing
  scenario    True: never recycled by _respawn_if_finished

In scenario episodes the random in-lane NPCs are NOT spawned: NPCs have no
collision avoidance, so a cruising lead car would rear-end them.  The
scenario's own RNG is separate, so routes match the normal runs seed for seed.
"""

import random

import numpy as np

SCENARIOS = ('none', 'hard_brake', 'stopped', 'stopped_oncoming', 'cut_in')

EVENT_MIN_AHEAD_M = 35.0     # first scenario actor at least this far ahead
EVENT_END_MARGIN_M = 25.0    # and this far before the goal


def make_scenario(name, env, seed):
    if name in (None, 'none'):
        return None
    cls = {'hard_brake': HardBrake, 'stopped': Stopped,
           'stopped_oncoming': StoppedOncoming, 'cut_in': CutIn}[name]
    return cls(env, random.Random(seed))


class Scenario:
    name = 'base'

    def __init__(self, env, rng):
        self.env, self.rng = env, rng
        self.actors = []          # npc dicts this scenario owns
        self.variant = ''
        self.spawned = False
        self.triggered = False
        self.passed = False
        self.event_s = None

    # ── helpers ────────────────────────────────────────────────────────

    def _event_s(self, lo=0.35, hi=0.55):
        env = self.env
        end = env.path_length - env.goal_margin_m - EVENT_END_MARGIN_M
        start = env.current_s + EVENT_MIN_AHEAD_M
        if end <= start:
            return None
        return float(np.clip(env.current_s + self.rng.uniform(lo, hi) * env.path_length,
                             start, end))

    def _spawn(self, s, d, target_speed, yaw_offset_deg=0.0, direction=1, **keys):
        """Spawn one vehicle at Frenet (s, d); returns its npc dict or None."""
        import carla
        env = self.env
        x, y, yaw = env.frenet_converter.frenet_to_world(s, d, 0.0)
        heading = np.rad2deg(yaw) + (180.0 if direction < 0 else 0.0) + yaw_offset_deg
        z = env.vehicle.get_location().z + 0.5 if env.vehicle is not None else 0.5
        bps = env.world.get_blueprint_library().filter('vehicle.bmw.*')
        for dz in (0.0, 0.5, 1.0):
            actor = env.world.try_spawn_actor(
                self.rng.choice(bps),
                carla.Transform(carla.Location(x=x, y=y, z=z + dz),
                                carla.Rotation(yaw=heading)))
            if actor is not None:
                npc = dict(actor=actor, s=s, target_speed=target_speed,
                           d_target=d, direction=direction, scenario=True, **keys)
                self.actors.append(npc)
                return npc
        return None

    def _side_lane_d(self, s, side):
        """Frenet d of the adjacent driving lane centre at s ('left'/'right'),
        and whether it runs the opposite way.  None if there is no such lane."""
        import carla
        env = self.env
        x, y, _ = env.frenet_converter.frenet_to_world(s, 0.0, 0.0)
        z = env.vehicle.get_location().z if env.vehicle is not None else 0.0
        wp = env.map.get_waypoint(carla.Location(x=x, y=y, z=z), project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
        if wp is None:
            return None
        lane = wp.get_left_lane() if side == 'left' else wp.get_right_lane()
        if lane is None or lane.lane_type != carla.LaneType.Driving:
            return None
        loc = lane.transform.location
        _, d, _ = env.frenet_converter.world_to_frenet(loc.x, loc.y, 0.0, s_hint=s)
        return float(d), (lane.lane_id * wp.lane_id) < 0

    def _ego_ds(self, npc):
        return npc['s'] - self.env.current_s

    # ── interface ──────────────────────────────────────────────────────

    def spawn(self):
        raise NotImplementedError

    def update(self):
        """Called every step before the NPC controllers."""
        env = self.env
        blocking = [a for a in self.actors if a.get('blocks', True)]
        if blocking and not self.passed:
            if env.current_s > max(a['s'] for a in blocking) + 5.0:
                self.passed = True

    def metrics(self):
        coll_id = self.env.collision_info.get('other_id')
        ids = {a['actor'].id for a in self.actors if a.get('actor') is not None}
        return {'scenario': self.name, 'scenario_variant': self.variant,
                'scenario_spawned': self.spawned, 'scenario_triggered': self.triggered,
                'scenario_passed': self.passed,
                'scenario_hit': bool(coll_id is not None and coll_id in ids)}


class HardBrake(Scenario):
    name = 'hard_brake'
    CRUISE = 8.0
    TRIGGER_GAP = 12.0
    HOLD_S = 3.0

    def spawn(self):
        env = self.env
        s = env.current_s + 20.0
        if s > env.path_length - env.goal_margin_m - EVENT_END_MARGIN_M:
            return []
        lead = self._spawn(s, 0.0, self.CRUISE, hard_brake=True)
        self.spawned = lead is not None
        self.variant = 'lead'
        self._stop_step = None
        self.event_s = s
        return list(self.actors)

    def update(self):
        super().update()
        if not self.actors:
            return
        lead, env = self.actors[0], self.env
        if lead.get('actor') is None:
            return
        v = lead['actor'].get_velocity()
        speed = float(np.hypot(v.x, v.y))
        if not self.triggered and 0.0 < self._ego_ds(lead) < self.TRIGGER_GAP and speed > 5.0:
            self.triggered = True
            lead['target_speed'] = 0.0
            lead['hold'] = True
        elif self.triggered and lead['target_speed'] == 0.0 and speed < 0.3:
            if self._stop_step is None:
                self._stop_step = env.current_step          # came to rest
            elif (env.current_step - self._stop_step) * env.mpc_dt > self.HOLD_S:
                lead['target_speed'] = self.CRUISE         # drive on
                lead['hold'] = False


class Stopped(Scenario):
    name = 'stopped'

    def _place_blockage(self, variant, lo=0.35, hi=0.55):
        s = self._event_s(lo, hi)
        if s is None:
            return False
        self.event_s = s
        if variant == 'parked':
            self._spawn(s, 0.3, 0.0, hold=True)
        else:   # accident: three cars, the first two slewed across the lane
            self._spawn(s, -0.3, 0.0, yaw_offset_deg=25.0, hold=True)
            self._spawn(s + 6.0, 0.4, 0.0, yaw_offset_deg=-20.0, hold=True)
            self._spawn(s + 12.0, 0.0, 0.0, hold=True)
        self.variant = variant
        self.spawned = len(self.actors) > 0
        self.triggered = self.spawned       # present from the start
        return self.spawned

    def spawn(self):
        self._place_blockage(self.rng.choice(['parked', 'accident']))
        return list(self.actors)


class StoppedOncoming(Stopped):
    name = 'stopped_oncoming'
    N_STREAM = 3
    GAP = (25.0, 45.0)
    SPEED = (6.0, 9.0)

    def spawn(self):
        # Blockage earlier on the route than in 'stopped', so the oncoming
        # stream has room beyond it.
        if not self._place_blockage('accident', 0.25, 0.4):
            return []
        side = self._side_lane_d(self.event_s, 'left')
        if side is None:
            self.variant = 'accident_no_left_lane'
            return list(self.actors)
        self.lane_d, opposite = side
        self.direction = -1 if opposite else 1
        self.variant = 'accident_oncoming' if opposite else 'accident_left_traffic'
        # Oncoming: a stream starting beyond the blockage, spaced further
        # ahead.  Same-direction (multi-lane road): a faster stream starting
        # behind the blockage, spaced further back, that overtakes the ego.
        s = self.event_s + 30.0 if opposite else self.event_s - 40.0
        step = 1.0 if opposite else -1.0
        for _ in range(self.N_STREAM):
            if not self._spawn_stream(s):
                break                       # ran off the route
            s += step * self.rng.uniform(*self.GAP)
        return list(self.actors)

    def _spawn_stream(self, s):
        if not (2.0 < s < self.env.path_length - 5.0):
            return None
        speed = self.rng.uniform(*self.SPEED) + (0.0 if self.direction < 0 else 3.0)
        return self._spawn(s, self.lane_d, speed, direction=self.direction, blocks=False)

    def update(self):
        super().update()
        if not hasattr(self, 'lane_d'):
            return
        env = self.env
        for npc in self.actors:
            if npc.get('blocks', True) or npc.get('actor') is None:
                continue
            # Recycle the stream so the passing lane stays busy while the ego
            # is near the blockage: oncoming cars that are behind the ego, or
            # same-direction cars far ahead of it, go back to the far end.
            if self.direction < 0 and npc['s'] < env.current_s - 15.0:
                new_s = min(max(env.current_s, self.event_s) + 60.0 + self.rng.uniform(0.0, 30.0),
                            env.path_length - 5.0)
            elif self.direction > 0 and npc['s'] > env.current_s + 60.0:
                new_s = min(env.current_s, self.event_s) - 30.0 - self.rng.uniform(0.0, 20.0)
            else:
                continue
            if 2.0 < new_s <= env.path_length - 5.0 and new_s > env.current_s + 20.0 \
                    or self.direction > 0 and 2.0 < new_s < env.current_s - 20.0:
                self._teleport(npc, new_s)

    def _teleport(self, npc, s):
        import carla
        env = self.env
        x, y, yaw = env.frenet_converter.frenet_to_world(s, self.lane_d, 0.0)
        heading = np.rad2deg(yaw) + (180.0 if self.direction < 0 else 0.0)
        z = env.vehicle.get_location().z + 0.5
        try:
            npc['actor'].set_transform(carla.Transform(carla.Location(x=x, y=y, z=z),
                                                       carla.Rotation(yaw=heading)))
            npc['s'] = s
        except RuntimeError:
            npc['actor'] = None


class CutIn(Scenario):
    name = 'cut_in'
    TRIGGER_DS = 15.0
    SPEED = 5.0

    def spawn(self):
        s = self._event_s()
        if s is None:
            return []
        self.event_s = s
        side = self._side_lane_d(s, 'right')
        d0 = side[0] if side is not None else 2.6     # adjacent lane, else the kerb
        self.variant = 'from_lane' if side is not None else 'from_kerb'
        npc = self._spawn(s, d0, 0.0, hold=True)
        if npc is None and side is None:
            npc = self._spawn(s, 2.2, 0.0, hold=True)
        self.spawned = npc is not None
        return list(self.actors)

    def update(self):
        super().update()
        if not self.actors or self.triggered:
            return
        npc = self.actors[0]
        if npc.get('actor') is not None and 0.0 < self._ego_ds(npc) < self.TRIGGER_DS:
            self.triggered = True
            npc['target_speed'] = self.SPEED
            npc['hold'] = False
            npc['d_target'] = 0.0
