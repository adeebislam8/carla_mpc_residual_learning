import gymnasium as gym
import numpy as np
import carla
import random
import time
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
from scipy.interpolate import make_interp_spline
import os
import sys
import weakref
import traceback
import gc

# <repo>/src -- so the global_planner / mpc_controller packages resolve
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from global_planner.src.frenet_world_converter_python import FrenetConverter
from global_planner.src.global_path_publisher_python import PathPlanner
from mpc_controller.src.mpc_controller_python import MPCController
from mpc_controller.src.residual_authority import ResidualAuthority



class CarlaMPCEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    _shared_planner_cache = {}  # Shared across ALL instances
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 2000,
        timeout: float = 20.0,
        towns: List[str] = ['Town01', 'Town02', 'Town03', 'Town04'],
        episodes_per_town: int = 99999,
        target_speed: float = 15,  # m/s
        max_steps: int = 1000,
        lookahead_distance: float = 100.0,
        num_obstacles: int = 6,
        state_dim: int = 116,
        mpc_horizon: int = 30,
        mpc_dt: float = 0.05,
        seed: int = 2547,
        discrete_actions: bool = False,
        render_mode: Optional[str] = None,
        steer_norm_deg: float = 45.0,
        qc: float = None,
        residual_mode: str = 'adaptive',
        residual_max: float = 0.1,
    ):
        """
        residual_mode: 'adaptive' derives the residual authority from the CBF
            feasibility of the *final* action (project spec Section 6);
            'fixed' reproduces the previous law u = u_nom + residual_max * pi(o)
            and is kept so the B2 baseline stays runnable.
        residual_max: residual magnitude that alpha = 1 corresponds to.  Left at
            0.1 so 'adaptive' at full authority matches the old fixed law, which
            makes the B2-vs-B5 comparison a clean single-variable change.
        """
        super().__init__()
        random.seed(seed)
        
        # CARLA connection
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.load_world(towns[0])
        self.map = self.world.get_map()
        
        # Multi-town setup
        self.available_towns = towns
        self.current_town = towns[0]
        self.episodes_per_town = episodes_per_town
        self.episode_count = 0
        
        # Environment parameters
        self.target_speed = target_speed
        self.max_steps = max_steps
        self.current_step = 0
        self.lookahead_distance = lookahead_distance
        self.num_obstacles = num_obstacles
        self.state_dim = state_dim
        
        # MPC parameters
        self.mpc_horizon = mpc_horizon
        self.mpc_dt = mpc_dt
        
        # Vehicle state
        self.vehicle = None
        self.collision_sensor = None
        self.collision_info = {}
        self.lane_invasion_sensor = None
        self.current_speed = 0.0
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.current_steering = 0.0
        # Real actuator range, read from the spawned vehicle in _initialize_mpc().
        # 70 deg is the Model 3 value; overwritten once the actor exists.
        self._carla_max_steer = np.deg2rad(70.0)
        self.lateral_accel = 0.0
        
        self._road_widths = None
        self._road_width_s = None
        
        # Frenet frame
        self.frenet_converter: Optional[FrenetConverter] = None
        self.path_planner: Optional[PathPlanner] = None
        self.current_s = 0.0
        self._ego_s_hint = None   # branch-continuity hint, per episode
        self.current_d = 0.0
        self.current_alpha = 0.0
        self.path_length = 0.0
        
        # MPC controller
        self.mpc_controller: Optional[MPCController] = None
        self.mpc_prediction = np.zeros((mpc_horizon, 2))
        # self._last_mpc_time_ms = 0.0
        
        # Obstacles
        self.selected_obstacles = np.ones((num_obstacles, 2)) * -100
        self.overtaken_npcs = set()
        
        # Episode state
        self.collision = False
        self.lane_invasion = False
        self.sensor_detected_obstacles = []
        self.start_time = 0.0
        
        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float64
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(state_dim),
            high=np.inf * np.ones(state_dim),
            shape=(state_dim,),
            dtype=np.float64
        )

        # ── Residual authority ────────────────────────────────
        # Steering feedforward gain: CARLA applies steer over 70 deg, so
        # normalising by 45 gives 1.556x.  Smaller => more gain.
        self.steer_norm_deg = steer_norm_deg
        self.qc = qc          # lateral tracking weight; None = model default (5e-2)
        self.residual_mode = residual_mode
        self.residual_max = residual_max
        self.residual_authority = None   # built in _initialize_mpc()
        self.last_authority_info = {}
        self.authority_history = []      # per-step alpha, for Section 17 metrics

        # ── Recording ─────────────────────────────────────────
        self.camera_sensor = None
        self.video_frames = []
        self.ego_trajectory = []   # (step, x, y, s, d)
        self.npc_trajectory = []   # (step, x, y, s, d) per NPC
        self.recording = False
        self.record_episode = 0
        
        # Set up CARLA world
        self._setup_world()
        
    def _setup_world(self):
        """Configure CARLA world for synchronous mode"""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.mpc_dt
        settings.tile_stream_distance = 5000  # Load tiles within 5km (was probably 2000 default)
        settings.actor_active_distance = 5000  # Keep actors active within 5km
        self.world.apply_settings(settings)
        
        # Enable traffic manager in sync mode
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

    def _attach_camera(self):
        """Attach top-down RGB camera for video recording"""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')

        # Fixed offset above vehicle — top down
        camera_transform = carla.Transform(
            carla.Location(x=0, y=0, z=20),
            carla.Rotation(pitch=-90, yaw=0, roll=0)
        )

        self.camera_sensor = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )

        weak_self = weakref.ref(self)
        self.camera_sensor.listen(
            lambda image: CarlaMPCEnv._on_camera_image(weak_self, image)
        )
        print("✓ Recording camera attached")

    @staticmethod
    def _on_camera_image(weak_self, image):
        self = weak_self()
        if self is not None and self.recording:
            try:
                import numpy as np
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                # BGRA → RGB
                frame = array[:, :, :3][:, :, ::-1].copy()
                self.video_frames.append(frame)
            except Exception:
                pass
    
    def start_recording(self, episode_num=0):
        """Call this before running the episode you want to record"""
        self.video_frames = []
        self.ego_trajectory = []
        self.npc_trajectory = []
        self.recording = True
        self.record_episode = episode_num
        self._attach_camera()
        print(f"🎥 Recording started for episode {episode_num}")

    def save_recording(self, label='hybrid', output_dir='./recordings'):
        """Call this after episode ends to save video + trajectory"""
        import os
        import cv2
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        self.recording = False
        os.makedirs(output_dir, exist_ok=True)
        tag = f"ep{self.record_episode}_{label}"

        # ── Save Video ────────────────────────────────────
        if self.video_frames:
            video_path = os.path.join(output_dir, f"{tag}.mp4")
            h, w = self.video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
            for frame in self.video_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"✓ Video saved: {video_path}  ({len(self.video_frames)} frames)")
        else:
            print("⚠ No video frames captured")

        # ── Save Trajectory Plot ──────────────────────────
        if not self.ego_trajectory:
            print("⚠ No trajectory data")
            return

        ego = np.array(self.ego_trajectory)    # (N, 5): step,x,y,s,d
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Trajectory Analysis — {label}  (Episode {self.record_episode})',
                    fontsize=13, fontweight='bold')

        # ── LEFT: World XY plot ───────────────────────────
        ax = axes[0]

        # Road boundaries
        if self._road_widths is not None and self.frenet_converter is not None:
            s_pts = np.linspace(0, self.path_length, 200)
            left_x, left_y, right_x, right_y, center_x, center_y = [], [], [], [], [], []
            for s in s_pts:
                idx = np.argmin(np.abs(self._road_width_s - s))
                lw = self._road_widths[idx, 0]
                rw = self._road_widths[idx, 1]
                xl, yl, _ = self.frenet_converter.frenet_to_world(s, -lw, 0)
                xr, yr, _ = self.frenet_converter.frenet_to_world(s,  rw, 0)
                xc, yc, _ = self.frenet_converter.frenet_to_world(s, 0.0, 0)
                left_x.append(xl);  left_y.append(yl)
                right_x.append(xr); right_y.append(yr)
                center_x.append(xc); center_y.append(yc)

            ax.fill_betweenx(left_y,  left_x,  right_x,
                            alpha=0.08, color='gray', label='Road')
            ax.plot(left_x,   left_y,   'b-', linewidth=1.0, alpha=0.5)
            ax.plot(right_x,  right_y,  'b-', linewidth=1.0, alpha=0.5)
            ax.plot(center_x, center_y, 'g--', linewidth=0.8,
                    alpha=0.6, label='Centerline')

        # Ego trajectory
        ax.plot(ego[:, 1], ego[:, 2], 'b-', linewidth=2, label='Ego (MPCC+CBF+PPO)')
        ax.plot(ego[0, 1],  ego[0, 2],  'go', markersize=8, label='Start')
        ax.plot(ego[-1, 1], ego[-1, 2], 'rs', markersize=8, label='End')

        # NPC trajectories
        colors = ['orange', 'red', 'purple', 'brown']
        if self.npc_trajectory:
            npc_data = np.array(self.npc_trajectory)
            npc_ids = np.unique(npc_data[:, 0]).astype(int)
            for i, nid in enumerate(npc_ids):
                mask = npc_data[:, 0] == nid
                nd = npc_data[mask]
                c = colors[i % len(colors)]
                ax.plot(nd[:, 2], nd[:, 3], '-', color=c,
                        linewidth=1.5, alpha=0.8, label=f'NPC {i+1}')
                ax.plot(nd[0, 2], nd[0, 3], 'o', color=c, markersize=6)

        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title('World Coordinates')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # ── RIGHT: Frenet d vs s plot ─────────────────────
        ax2 = axes[1]

        # Road boundaries in Frenet
        if self._road_widths is not None:
            s_pts = np.linspace(0, self.path_length, 200)
            n_left_arr  = [-self._road_widths[np.argmin(np.abs(self._road_width_s - s)), 0]
                        for s in s_pts]
            n_right_arr = [ self._road_widths[np.argmin(np.abs(self._road_width_s - s)), 1]
                            for s in s_pts]
            ax2.fill_between(s_pts, n_left_arr, n_right_arr,
                            alpha=0.08, color='gray')
            ax2.plot(s_pts, n_left_arr,  'b-', linewidth=1.0, alpha=0.5)
            ax2.plot(s_pts, n_right_arr, 'b-', linewidth=1.0, alpha=0.5)
            ax2.axhline(y=0, color='g', linestyle='--',
                        linewidth=0.8, alpha=0.6, label='Centerline')

        # Ego Frenet trajectory
        ax2.plot(ego[:, 3], ego[:, 4], 'b-', linewidth=2,
                label='Ego (MPCC+CBF+PPO)')

        # NPC Frenet trajectories
        if self.npc_trajectory:
            npc_data = np.array(self.npc_trajectory)
            npc_ids = np.unique(npc_data[:, 0]).astype(int)
            for i, nid in enumerate(npc_ids):
                mask = npc_data[:, 0] == nid
                nd = npc_data[mask]
                c = colors[i % len(colors)]
                ax2.plot(nd[:, 1], nd[:, 4], '-', color=c,
                        linewidth=1.5, alpha=0.8, label=f'NPC {i+1}')

        ax2.set_xlabel('Arc Length s (m)')
        ax2.set_ylabel('Lateral Deviation d (m)')
        ax2.set_title('Frenet Frame')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        traj_path = os.path.join(output_dir, f"{tag}_trajectory.pdf")
        plt.savefig(traj_path, dpi=150, bbox_inches='tight')
        plt.savefig(traj_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Trajectory saved: {traj_path}")

        # ── Save raw numpy data too ───────────────────────
        np.save(os.path.join(output_dir, f"{tag}_ego.npy"), ego)
        if self.npc_trajectory:
            np.save(os.path.join(output_dir, f"{tag}_npc.npy"),
                    np.array(self.npc_trajectory))
        print(f"✓ Raw data saved to {output_dir}/")

    def _spawn_ego_vehicle(self, spawn_point: carla.Transform) -> bool:
        """Spawn ego vehicle at given spawn point"""
        try:
            all_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in all_vehicles:
                try:
                    if vehicle.is_alive:
                        vehicle.destroy()
                except:
                    pass
            
            # Tick a few times to ensure cleanup
            for _ in range(3):
                self.world.tick()
                time.sleep(0.02)
            
            print("✓ All vehicles cleared")
        except Exception as e:
            print(f"Warning: Error while clearing vehicles: {e}")
        
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('role_name', 'ego_vehicle')
        
        try:
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.world.tick()  # Let CARLA register the vehicle
            return True
        except RuntimeError as e:
            print(f"Failed to spawn vehicle: {e}")
            return False
    
    def _attach_sensors(self):
        """Attach collision and lane invasion sensors"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        # Lane invasion sensor
        lane_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

        # Obstacle detection sensor
        obstacle_bp = blueprint_library.find('sensor.other.obstacle')
        obstacle_bp.set_attribute('distance', '50')
        obstacle_bp.set_attribute('hit_radius', '2.0')
        obstacle_bp.set_attribute('debug_linetrace', 'true')
        
        self.obstacle_sensor = self.world.spawn_actor(
            obstacle_bp,
            carla.Transform(carla.Location(x=2.5, z=1.0)),
            attach_to=self.vehicle
        )
        
        # Store detected obstacles from sensor
        self.sensor_detected_obstacles = []
        # Use a weak listener to avoid issues during cleanup
        weak_self = weakref.ref(self)
        self.obstacle_sensor.listen(
            lambda event: CarlaMPCEnv._on_obstacle_detected_static(weak_self, event)
        )
        
        self.world.tick()
    
    def _on_collision(self, event):
        """
        Collision callback.

        Records what was hit and the state at impact, not just a boolean.  A 70%
        collision rate with no attribution is the same position the solver was in
        before diagnostics: the fix is unknowable without knowing whether the car
        is rear-ending the lead vehicle (failing to slow), sideswiping during a
        pass (the overtake manoeuvre), or leaving the road (tracking failure).
        Each implies a different fix.

        Runs on CARLA's sensor thread, so everything is wrapped -- a throw here
        would be swallowed silently and lose the episode.
        """
        self.collision = True
        if self.collision_info:
            return  # keep the first impact only

        info = {'other': 'unknown', 'kind': 'unknown'}
        try:
            other = getattr(event, 'other_actor', None)
            info['other'] = getattr(other, 'type_id', 'unknown') or 'unknown'
            info['other_is_vehicle'] = info['other'].startswith('vehicle.')

            # Impulse in world frame -> ego frame, to tell front/rear/side apart.
            imp = event.normal_impulse
            yaw = np.deg2rad(self.vehicle.get_transform().rotation.yaw)
            c, sn = np.cos(yaw), np.sin(yaw)
            fwd = imp.x * c + imp.y * sn
            lat = -imp.x * sn + imp.y * c
            info['impulse'] = float(np.sqrt(imp.x**2 + imp.y**2 + imp.z**2))
            info['impulse_fwd'] = float(fwd)
            info['impulse_lat'] = float(lat)

            # NOTE: sign convention is unverified -- check one real collision
            # against the video before trusting front/rear, and flip if needed.
            if abs(fwd) >= abs(lat):
                info['kind'] = 'front' if fwd < 0 else 'rear'
            else:
                info['kind'] = 'left' if lat > 0 else 'right'

            info['speed'] = float(self.current_speed)
            info['n'] = float(self.current_d)
            info['alpha'] = float(self.current_alpha)
            info['progress_frac'] = float(
                self.current_s / max(self.path_length, 1e-6))
            info['overtakes_so_far'] = len(self.overtaken_npcs)

            # Path curvature at the impact point.  The corridor is ~6x wider to
            # the left than the right, and tracking error pushes a car to the
            # OUTSIDE of a curve, so one turn direction has far less budget than
            # the other.  Recording kappa lets that be measured instead of
            # derived from sign conventions.
            try:
                info['kappa'] = float(
                    self.frenet_converter.get_curvature(self.current_s))
            except Exception:
                info['kappa'] = float('nan')

            # Off-road: outside the lateral corridor when it hit.
            if self._road_widths is not None and self._road_width_s is not None:
                idx = np.argmin(np.abs(self._road_width_s - self.current_s))
                info['off_corridor'] = bool(
                    self.current_d < -self._road_widths[idx, 0]
                    or self.current_d > self._road_widths[idx, 1])

            # Was the MPC solving, or had the fallback taken over?
            mpc = self.mpc_controller
            info['in_fallback'] = bool(
                mpc is not None and getattr(mpc, 'consecutive_failures', 0) > 0)
        except Exception as e:
            info['error'] = repr(e)

        self.collision_info = info
    
    def _on_lane_invasion(self, event):
        """Lane invasion callback - only trigger for solid markings"""
        for marking in event.crossed_lane_markings:
            if marking.type == carla.LaneMarkingType.Solid:
                self.lane_invasion = True
                break
    
    @staticmethod
    def _on_obstacle_detected_static(weak_self, event):
        """Static callback with proper None checks"""
        self = weak_self()
        if self is not None:
            try:
                # Check if sensor data still exists
                if hasattr(self, 'sensor_detected_obstacles'):
                    if event.other_actor is not None and event.other_actor.is_alive:
                        self.sensor_detected_obstacles.append({
                            'actor': event.other_actor,
                            'distance': event.distance,
                            'location': event.other_actor.get_location(),
                            'timestamp': time.time()
                        })
            except (RuntimeError, AttributeError):
                # Gracefully handle destroyed actors/sensors
                pass

    def _on_obstacle_detected(self, event):
        """Callback for obstacle detection sensor"""
        try:
            if event.other_actor is not None and hasattr(self, 'sensor_detected_obstacles'):
                self.sensor_detected_obstacles.append({
                    'actor': event.other_actor,
                    'distance': event.distance,
                    'location': event.other_actor.get_location(),
                    'timestamp': time.time()
                })
        except (RuntimeError, AttributeError):
            # Sensor or actor may be destroyed during callback
            pass
    
    def _generate_path(self, start, goal):
        cache_key = self.current_town  # e.g. 'Town12'
        
        if cache_key not in CarlaMPCEnv._shared_planner_cache:
            CarlaMPCEnv._shared_planner_cache[cache_key] = PathPlanner(
                self.world, self.map
            )
        
        self.path_planner = CarlaMPCEnv._shared_planner_cache[cache_key]
        
        waypoints = self.path_planner.calculate_route(start.location, goal.location)
        
        if not waypoints or len(waypoints) < 4:
            del waypoints
            return False
        
        waypoint_coords = []
        road_widths = []
        for wp in waypoints:
            waypoint_coords.append([wp.transform.location.x, wp.transform.location.y])
            lw, rw = self.path_planner.get_road_width_at_waypoint(wp)
            road_widths.append([lw, rw])
        
        del waypoints  # Free CARLA Waypoint objects
        gc.collect()
        
        self.frenet_converter = FrenetConverter(waypoint_coords)
        self.path_length = self.frenet_converter.get_path_length()
        self._road_widths = np.array(road_widths)

        road_width_s_list = []
        for x, y in waypoint_coords:
            s, _, _ = self.frenet_converter.world_to_frenet(x, y, 0)
            road_width_s_list.append(s)
        
        self._road_width_s = np.array(road_width_s_list)
        
        return True
    
    # def _spawn_frenet_racers(self, num_cars=5, min_gap=12.0, ego_buffer=20.0):
    #     vehicles = []
    #     blueprints = self.world.get_blueprint_library().filter('vehicle.bmw.*')

    #     used_s = []
    #     for _ in range(num_cars):
    #         spawned = False
    #         for _try in range(50):  # More attempts
    #             s = random.uniform(2.0, self.path_length - 5.0)

    #             ds = s - self.current_s

    #             # Only spawn AHEAD of ego, not behind
    #             if ds < ego_buffer:  # Must be at least ego_buffer ahead
    #                 continue
                
    #             # Keep distance from other NPCs
    #             if any(abs(s - s_used) < min_gap for s_used in used_s):
    #                 continue

    #             x, y, yaw = self.frenet_converter.frenet_to_world(s, 0.0, 0.0)
    #             transform = carla.Transform(
    #                 carla.Location(x=x, y=y, z=0.5),
    #                 carla.Rotation(yaw=np.rad2deg(yaw))
    #             )

    #             bp = random.choice(blueprints)
    #             try:
    #                 npc = self.world.try_spawn_actor(bp, transform)
    #                 if npc is not None:
    #                     used_s.append(s)
    #                     vehicles.append({
    #                         "actor": npc,
    #                         "s": s,
    #                         "target_speed": random.uniform(4.0, 6.0)
    #                     })
    #                     spawned = True
    #                     break
    #             except Exception as e:
    #                 continue

    #         if not spawned:
    #             print(f"Warning: Could not spawn NPC after 50 attempts, skipping.")

    #     if vehicles:
    #         self.world.tick()
    #     return vehicles

    def _spawn_frenet_racers(self, num_cars=5, min_gap=12.0, ego_buffer=20.0):
        vehicles = []
        blueprints = self.world.get_blueprint_library().filter('vehicle.bmw.*')
        used_s = []

        # ── FIXED NPC at s = 15 ───────────────────────────────
        fixed_s = self.current_s + 15.0  # 15m ahead of ego
        x, y, yaw = self.frenet_converter.frenet_to_world(fixed_s, 0.0, 0.0)
        transform = carla.Transform(
            carla.Location(x=x, y=y, z=0.5),
            carla.Rotation(yaw=np.rad2deg(yaw))
        )
        bp = random.choice(blueprints)
        try:
            npc = self.world.try_spawn_actor(bp, transform)
            if npc is not None:
                used_s.append(fixed_s)
                vehicles.append({
                    "actor": npc,
                    "s": fixed_s,
                    "target_speed": 3.0  # slow so ego catches up quickly
                })
                print(f"✓ Fixed NPC spawned at s={fixed_s:.1f}")
            else:
                print(f"⚠ Fixed NPC failed to spawn at s={fixed_s:.1f}")
        except Exception as e:
            print(f"⚠ Fixed NPC spawn error: {e}")
        # ─────────────────────────────────────────────────────

        # Random NPCs as before
        for _ in range(num_cars):
            spawned = False
            for _try in range(50):
                s = random.uniform(2.0, self.path_length - 5.0)
                ds = s - self.current_s
                if ds < ego_buffer:
                    continue
                if any(abs(s - s_used) < min_gap for s_used in used_s):
                    continue

                x, y, yaw = self.frenet_converter.frenet_to_world(s, 0.0, 0.0)
                transform = carla.Transform(
                    carla.Location(x=x, y=y, z=0.5),
                    carla.Rotation(yaw=np.rad2deg(yaw))
                )
                bp = random.choice(blueprints)
                try:
                    npc = self.world.try_spawn_actor(bp, transform)
                    if npc is not None:
                        used_s.append(s)
                        vehicles.append({
                            "actor": npc,
                            "s": s,
                            "target_speed": random.uniform(4.0, 6.0)
                        })
                        spawned = True
                        break
                except Exception as e:
                    continue

            if not spawned:
                print(f"Warning: Could not spawn NPC after 50 attempts, skipping.")

        if vehicles:
            self.world.tick()
        return vehicles

    def _frenet_follow_controller(self, npc_data, dt):
        npc = npc_data["actor"]

        transform = npc.get_transform()
        x = transform.location.x
        y = transform.location.y
        yaw = np.deg2rad(transform.rotation.yaw)

        s, d, alpha = self.frenet_converter.world_to_frenet(x, y, yaw)

        # Lookahead along Frenet path
        lookahead = 6.0
        s_ref = s + lookahead
        if s_ref > self.path_length:
            s_ref -= self.path_length

        x_ref, y_ref, yaw_ref = self.frenet_converter.frenet_to_world(s_ref, 0.0, 0.0)

        # Heading error
        heading_error = (yaw_ref - yaw + np.pi) % (2*np.pi) - np.pi

        # Lateral correction from Frenet d
        k_d = 0.6
        k_yaw = 1.2

        steer = k_yaw * heading_error + k_d * (-d)
        steer = np.clip(steer, -1.0, 1.0)

        # Speed control
        v = npc.get_velocity()
        speed = np.linalg.norm([v.x, v.y])
        target_speed = npc_data["target_speed"]

        speed_error = target_speed - speed
        control = carla.VehicleControl()

        if speed_error > 0:
            control.throttle = np.clip(0.5 * speed_error, 0.0, 0.75)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = np.clip(-0.5 * speed_error, 0.0, 0.5)

        control.steer = steer
        npc.apply_control(control)

        # Update s (for looping)
        npc_data["s"] = s

    def _respawn_if_finished(self, npc_data):
        s = npc_data["s"]

        if s > self.path_length - 3.0:
            s_new = 2.0
            respawn_buffer = 5.0
            
            # Check if ego vehicle is too close to the spawn point
            ego_in_respawn_zone = (self.current_s < s_new + respawn_buffer)
            
            if ego_in_respawn_zone:
                npc = npc_data["actor"]
                try:
                    if npc.is_alive:
                        npc.destroy()
                except:
                    pass
                
                # Mark this NPC as destroyed
                npc_data["actor"] = None
                return
            
            # Safe to respawn
            x, y, yaw = self.frenet_converter.frenet_to_world(s_new, 0.0, 0.0)

            npc = npc_data["actor"]
            npc.set_transform(carla.Transform(
                carla.Location(x=x, y=y, z=0.5),
                carla.Rotation(yaw=np.rad2deg(yaw))
            ))

            npc_data["s"] = s_new


    def _initialize_mpc(self):
        """Initialize ACADOS MPC controller"""
        if self.frenet_converter is None:
            raise ValueError("Frenet converter must be initialized before MPC")
        
        self.mpc_controller = MPCController(
            horizon=self.mpc_horizon,
            dt=self.mpc_dt,
            frenet_converter=self.frenet_converter,
            target_speed=self.target_speed
        )
        
        # Sample path to get curvature
        s_samples = np.linspace(0, self.path_length, 100)
        kappa_samples = [self.frenet_converter.get_curvature(s) for s in s_samples]
        kappa_spline = make_interp_spline(s_samples, kappa_samples, k=3)
        
        # Give the controller the ego's real footprint so the corridor margin
        # matches the car actually being driven, not a hardcoded Model 3.
        self.mpc_controller.steer_norm_deg = self.steer_norm_deg
        self.mpc_controller.qc = self.qc
        try:
            wheels = self.vehicle.get_physics_control().wheels
            real = max(w.max_steer_angle for w in wheels)
            self._carla_max_steer = float(np.deg2rad(real))
            self.mpc_controller.carla_max_steer = self._carla_max_steer
            print(f"  steering: CARLA max_steer_angle = {real:.1f} deg")
        except Exception as e:
            print(f"  could not read steering limit ({e}); using 70 deg")

        try:
            ext = self.vehicle.bounding_box.extent
            self.mpc_controller.veh_length = float(2.0 * ext.x)
            self.mpc_controller.veh_width = float(2.0 * ext.y)
            print(f"  ego footprint: {2*ext.x:.2f} x {2*ext.y:.2f} m")
        except Exception as e:
            print(f"  could not read ego bounding box ({e}); using defaults")

        # Initialize ACADOS
        self.mpc_controller.initialize_acados(kappa_spline, path_msg=None)
        self.mpc_controller._global_path_length = self.path_length

        # Residual authority re-verifies the CBF on the residual-modified
        # action, so it must be built from the same model the OCP compiled.
        self.residual_authority = ResidualAuthority.from_mpc_model(
            self.mpc_controller.model, dt=self.mpc_dt
        )
    
    def _update_vehicle_state(self):
        """Update vehicle state from CARLA"""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        control = self.vehicle.get_control()
        
        # Convert to Frenet
        x = transform.location.x
        y = transform.location.y
        yaw = np.deg2rad(transform.rotation.yaw)
        
        # Hint with the ego's previous s so the closest-point search stays on the
        # same branch.  Without it a route that doubles back can flip branches --
        # measured 1 m of drift moving s by ~100 m and inverting d's sign, which
        # corrupts progress, curvature, the corridor and the overtake direction
        # in one step.
        self.current_s, self.current_d, self.current_alpha = \
            self.frenet_converter.world_to_frenet(
                x, y, yaw, s_hint=self._ego_s_hint)
        self._ego_s_hint = self.current_s
        
        # Vehicle dynamics
        self.current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        if control.brake > 0:
            self.current_throttle = -control.brake
        else:
            self.current_throttle = control.throttle
        self.current_brake = control.brake
        # Read back with the MODEL's 45 deg, matching how the command was
        # produced, so the MPC's delta state equals what it planned.  See the
        # long comment in mpc_controller_python.solve(): the 45-vs-70 mismatch is
        # a deliberate 1.556x understeer gain, and removing it was much worse.
        self.current_steering = control.steer * np.deg2rad(
            self.mpc_controller.steer_norm_deg if self.mpc_controller else 45.0)

        angular_velocity = self.vehicle.get_angular_velocity()
        yaw_rate = np.deg2rad(angular_velocity.z)
        self.lateral_accel = self.current_speed * yaw_rate

    def _detect_obstacles(self):
        self.selected_obstacles = np.ones((self.num_obstacles, 2)) * -100
        obstacles = []
        
        MAX_OBS_LOOKAHEAD = 30.0  # Only care about obstacles within 30m
        
        for npc_data in self.racing_npcs:
            npc = npc_data.get("actor")
            if npc is None or not npc.is_alive:
                continue
            try:
                loc = npc.get_location()
                s_obs, d_obs, _ = self.frenet_converter.world_to_frenet(
                    loc.x, loc.y, 0, s_hint=npc_data.get("s"))
                ds = s_obs - self.current_s
                if ds > -5.0 and ds < MAX_OBS_LOOKAHEAD:
                    obstacles.append({
                        's': s_obs,
                        'd': d_obs,
                        'distance': ds,
                        'type': 'npc'
                    })
            except:
                pass
        
        obstacles.sort(key=lambda x: x['distance'])
        for i, obs in enumerate(obstacles[:self.num_obstacles]):
            self.selected_obstacles[i] = [obs['s'], obs['d']]
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector:
        - Remaining distance to goal
        - Current Frenet state (s, d, alpha)
        - MPC control outputs
        - Previous MPC control
        - Obstacle positions (Frenet)
        - Vehicle state (speed, accel, throttle, brake, steer)
        - Path curvature samples
        - Current step
        - MPC predicted path
        """
        # Get MPC prediction and control
        mpc_control = self.mpc_controller.get_last_control() if self.mpc_controller else np.zeros(2)
        mpc_prev = self.mpc_controller.get_previous_control() if self.mpc_controller else np.zeros(2)
        
        # Curvature sampling
        s_samples = np.linspace(
            self.current_s,
            min(self.current_s + self.lookahead_distance, self.path_length),
            30
        )
        kappa_samples = np.array([
            self.frenet_converter.get_curvature(s) for s in s_samples
        ])
        
        # Flatten MPC prediction
        mpc_pred_flat = self.mpc_prediction.flatten()
        
        obs = np.concatenate([
            [self.path_length - self.current_s],  # Remaining distance
            [self.current_s, self.current_d, self.current_alpha],  # Frenet state
            mpc_control,  # MPC throttle, steering
            mpc_prev,  # Previous MPC control
            self.selected_obstacles.flatten(),  # Obstacles
            [self.current_speed, self.lateral_accel, self.current_throttle, 
             self.current_brake, self.current_steering],  # Vehicle state
            kappa_samples,  # Curvature
            [self.current_step],  # Step count
            mpc_pred_flat  # MPC prediction
        ])
        
        obs = obs.astype(np.float64)

        # Catch NaN/Inf before they enter the buffer
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return np.clip(obs, -1e6, 1e6)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
    
        if self.collision:
            return -200.0
        
        # Progress reward
        progress = self.current_s - self.prev_s
        reward = progress / (self.target_speed * self.mpc_dt)
        
        # Lane boundary
        idx = np.argmin(np.abs(self._road_width_s - self.current_s))
        n_min = -self._road_widths[idx, 0]
        n_max =  self._road_widths[idx, 1]
        road_width = n_max - n_min
        d_normalized = (self.current_d - n_min) / road_width
        edge_penalty = -2.0 * (2 * d_normalized - 1) ** 4
        reward += edge_penalty
        
        if not (n_min < self.current_d < n_max):
            reward -= 5.0
        
        # Lateral acceleration penalty
        reward -= 0.1 * abs(self.lateral_accel)
        
        # Heading penalty
        reward -= abs(self.current_alpha) * 2.0
        
        # Speed tracking
        speed_error = abs(self.current_speed - self.target_speed)
        reward -= 0.05 * speed_error
        if self.current_speed > self.target_speed * 1.3:
            reward -= 1.0
        
        # Stall penalty
        if self.current_speed < 1.0:
            reward -= 5.0
        
        # Obstacle avoidance
        for obs in self.selected_obstacles:
            if obs[0] > -50:
                dist = np.sqrt(
                    (self.current_s - obs[0])**2 + 
                    ((self.current_d - obs[1]) * 2)**2
                )
                if dist < 6.0:
                    reward -= 3.0
        
        # --- OVERTAKING REWARD ---
        for npc_data in self.racing_npcs:
            npc = npc_data.get("actor")
            if npc is None or not npc.is_alive:
                continue
            npc_id = npc.id
            npc_s = npc_data.get("s", -999)
            
            # Ignore NPCs that have respawned behind us
            if npc_s < self.current_s - 50.0:  # way behind = just respawned
                continue
            
            # NPC must have been meaningfully ahead at some point
            # Only count overtake if NPC is within a reasonable window behind us
            ds = self.current_s - npc_s
            if 5.0 < ds < 40.0 and npc_id not in self.overtaken_npcs:
                self.overtaken_npcs.add(npc_id)
                reward += 50.0
                print(f"🏎️  Overtook NPC {npc_id}! Total overtakes: {len(self.overtaken_npcs)}")
        
        # --- GOAL REACHED ---
        if self.current_s >= self.path_length - 10:
            # Base completion bonus
            reward += 200.0
            
            # Time bonus — faster finish = more reward
            # At target speed, expected time = path_length / target_speed
            expected_time = self.path_length / self.target_speed
            actual_time = time.time() - self.start_time
            time_ratio = expected_time / max(actual_time, 1.0)  # >1 means faster than expected
            time_bonus = 100.0 * time_ratio  # scales with how fast you finished
            reward += time_bonus
            
            # Overtaking bonus at finish — reward total cars beaten
            overtake_finish_bonus = len(self.overtaken_npcs) * 25.0
            reward += overtake_finish_bonus
            
            print(f"🏁 Finished! Time bonus: {time_bonus:.1f}, Overtakes: {len(self.overtaken_npcs)} (+{overtake_finish_bonus:.1f})")
        
        return reward
    
    def _authority_metrics(self) -> Dict:
        """
        Residual-authority metrics from Section 17.2 of the project spec:
        average safe authority, intervention count and strong-suppression
        frequency.  Attached to every step's info so the evaluation scripts can
        aggregate them per episode without extra plumbing.
        """
        metrics = {}

        # Solver health.  A collision immediately after a run of fallback steps
        # is a solver failure, not a control failure -- worth separating when
        # reporting collision rates.
        mpc = self.mpc_controller
        if mpc is not None and getattr(mpc, 'solve_calls', 0) > 0:
            metrics["solver_failures"] = int(mpc.solve_failures)
            metrics["solver_failure_rate"] = float(
                mpc.solve_failures / mpc.solve_calls)
            metrics["in_fallback"] = bool(mpc.consecutive_failures > 0)

        if not self.authority_history:
            return metrics

        alphas = np.asarray(self.authority_history, dtype=float)
        metrics.update({
            "alpha": float(alphas[-1]),
            "alpha_mean": float(alphas.mean()),
            "authority_interventions": int(np.sum(alphas < 1.0 - 1e-6)),
            "strong_suppression_frac": float(np.mean(alphas < 0.2)),
            "cbf_margin": float(self.last_authority_info.get("margin_at_alpha", np.inf)),
            "nominal_infeasible": bool(self.last_authority_info.get("nominal_infeasible", False)),
        })
        return metrics

    def _check_done(self) -> Tuple[bool, Dict]:
        info = {}
        
        # Collision
        if self.collision:
            elapsed = time.time() - self.start_time
            out = {"done_reason": "collision", "lap_time": elapsed}
            out.update({f"collision_{k}": v for k, v in self.collision_info.items()})
            return True, out
        
        # Goal reached
        if self.current_s >= self.path_length - 10:
            elapsed = time.time() - self.start_time
            return True, {"done_reason": "success", "lap_time": elapsed}
        
        # Stalled
        if self.current_speed < 1.0:
            self.speed_stall_count = getattr(self, 'speed_stall_count', 0) + 1
            if self.speed_stall_count > 500:
                elapsed = time.time() - self.start_time
                return True, {"done_reason": "stall", "lap_time": elapsed}
        else:
            self.speed_stall_count = 0
        
        # Max steps
        if self.current_step >= self.max_steps:
            elapsed = time.time() - self.start_time
            return True, {"done_reason": "timeout", "lap_time": elapsed}
        
        return False, {"done_reason": "running"}
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Town rotation
        try:
            self.episode_count += 1
            self.overtaken_npcs = set()
            # if self.episode_count % self.episodes_per_town == 0 and self.episode_count > 0:
            #     self._load_random_town()
            # Disable because of resource
            
            # Flush the previous episode's solver diagnostics before the
            # controller is rebuilt below.
            if self.mpc_controller is not None:
                diag = getattr(self.mpc_controller, 'diagnostics', None)
                if diag is not None:
                    diag.save(tag=f"ep{self.episode_count:04d}")

            # Destroy old vehicle and sensors
            self._destroy_actors()
            
            # Find valid spawn and goal
            max_attempts = 500
            spawn_points = self.map.get_spawn_points()
            for attempt in range(max_attempts):
                # Build a fresh Transform: get_spawn_points() is cached above, so
                # mutating the chosen one would drift its z on every re-pick.
                base_point = random.choice(spawn_points)
                spawn_point = carla.Transform(
                    carla.Location(
                        x=base_point.location.x,
                        y=base_point.location.y,
                        z=base_point.location.z + 1.0,  # avoid collision
                    ),
                    base_point.rotation,
                )
                goal_point = random.choice(spawn_points)

                # Fixed spawn/goal for parameter exploration -- re-enable to pin
                # the episode to one route. Leave commented for normal runs, or
                # the retry loop below can only ever try this single point.
                # spawn_point = carla.Transform(
                #     carla.Location(x=63.340027, y=191.769989, z=1.500000),
                #     carla.Rotation(pitch=0.000000, yaw=-0.000183, roll=0.000000)
                # )

                # goal_point = carla.Transform(
                #     carla.Location(x=-7.530000, y=270.729980, z=0.500000),
                #     carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000)
                # )

                # Ensure minimum distance
                dist = np.sqrt(
                    (spawn_point.location.x - goal_point.location.x)**2 +
                    (spawn_point.location.y - goal_point.location.y)**2
                )
                
                if dist > 50.0:
                    print(f"spawn: {spawn_point}")
                    print(f"goal: {goal_point}")
                    # Spawn vehicle
                    if not self._spawn_ego_vehicle(spawn_point):
                        continue
                    
                    # Attach sensors
                    self._attach_sensors()
                    
                    # Generate path
                    if not self._generate_path(spawn_point, goal_point):
                        self._destroy_actors()
                        #self.vehicle.destroy()
                        continue
                    
                    # Initialize MPC
                    self._initialize_mpc()
                    # self._visualize_path_and_goal(goal_point)

                    self.racing_npcs = self._spawn_frenet_racers(num_cars = random.randint(0, 7))

                    break
            else:
                raise RuntimeError(f"Failed to find valid spawn after {max_attempts} attempts")
            
            # Reset state
            self.current_step = 0
            self.collision = False
            self.collision_info = {}
            self.lane_invasion = False
            self.start_time = time.time()
            self.prev_s = 0.0
            self.authority_history = []
            self.last_authority_info = {}
            self._ego_s_hint = None
            
            # Initial tick
            self.world.tick()
            self._update_vehicle_state()
            self._detect_obstacles()
            
            obs = self._get_observation()
            return obs, {}
        except Exception as e:
            print(f"Error at reset with {e}")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # if self.current_step % 100 == 0:
        #     print(f"[LEAK DIAG] step={self.current_step} | "
        #         f"sensor_obstacles={len(self.sensor_detected_obstacles)} | "
        #         f"racing_npcs={len(self.racing_npcs)} | "
        #         f"gc_objects={len(gc.get_objects())}")
        try:
            # vehicles = self.world.get_actors().filter('vehicle.*')
            # print("Vehicles in world:", len(vehicles))
            # for v in vehicles:
            #     print(v.id, v.type_id)

            self.current_step += 1
            self.prev_s = self.current_s

            for npc_data in self.racing_npcs:
                if npc_data["actor"] is not None and npc_data["actor"].is_alive:
                    self._frenet_follow_controller(npc_data, self.mpc_dt)
                    self._respawn_if_finished(npc_data)
            
            # Run MPC to get base control
            self._update_vehicle_state()
            self._detect_obstacles()
            # self._debug_road_boundaries(every_n_steps=50)

            # _t_mpc_start = time.perf_counter()
            mpc_throttle, mpc_steering = self.mpc_controller.solve(
                s=self.current_s,
                d=self.current_d,
                alpha=self.current_alpha,
                v=self.current_speed,
                obstacles=self.selected_obstacles,
                D=self.current_throttle,
                delta=self.current_steering,
                road_widths=self._road_widths,  # None is okay, MPC will use defaults
                road_width_s=self._road_width_s,
            )
            # _t_mpc_end = time.perf_counter()
            # self._last_mpc_time_ms = (_t_mpc_end - _t_mpc_start) * 1000

            # print("="*20)
            # print(f"MPC Throttle: {mpc_throttle}\nMPC Steering: {mpc_steering}")
            # print("="*20)

            # Apply residual from RL under the adaptive authority.  The
            # residual proposal is scaled by alpha = g_support * alpha_safe,
            # where alpha_safe is the largest scale for which the *final*
            # action still satisfies the modelled CBF and actuator constraints
            # (project spec Sections 6-7).  'fixed' keeps the old law for B2.
            du = np.array([action[0], action[1]], dtype=float) * self.residual_max

            if self.residual_mode == 'adaptive':
                alpha, self.last_authority_info = self.residual_authority.compute(
                    s=self.current_s,
                    n=self.current_d,
                    heading=self.current_alpha,
                    v=self.current_speed,
                    u_nom=(mpc_throttle, mpc_steering),
                    du=du,
                    obstacles=self.selected_obstacles,
                    support_gate=1.0,  # until the support monitor lands
                )
            else:
                alpha = 1.0
                self.last_authority_info = {
                    'alpha_safe': 1.0, 'alpha': 1.0, 'nominal_infeasible': False,
                    'saturated': True, 'margin_at_alpha': np.inf,
                    'support_gate': 1.0,
                }

            self.authority_history.append(alpha)

            final_throttle = np.clip(mpc_throttle + alpha * du[0], -1.0, 1.0)
            final_steering = np.clip(mpc_steering + alpha * du[1], -1.0, 1.0)

            if final_throttle < 0 and self.current_speed < 0.3:
                final_throttle = 1.0  # Prevent Stalling
            
            # print(f"Throttle: {final_throttle} \n Steering: {final_steering}")
            
            # Apply control to vehicle
            control = carla.VehicleControl()
            if final_throttle >= 0:
                control.throttle = final_throttle
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = abs(final_throttle)
            control.steer = - final_steering
            
            self.vehicle.apply_control(control)
            
            # Tick world
            self.world.tick()
            # ── Trajectory logging ────────────────────────────
            if self.recording:
                # Ego
                t = self.vehicle.get_transform()
                self.ego_trajectory.append([
                    self.current_step,
                    t.location.x, t.location.y,
                    self.current_s, self.current_d
                ])
                # NPCs
                for npc_data in self.racing_npcs:
                    npc = npc_data.get("actor")
                    if npc is None or not npc.is_alive:
                        continue
                    try:
                        loc = npc.get_location()
                        s_n, d_n, _ = self.frenet_converter.world_to_frenet(
                            loc.x, loc.y, 0)
                        self.npc_trajectory.append([
                            npc.id,
                            self.current_step,
                            loc.x, loc.y,
                            s_n, d_n
                        ])
                    except:
                        pass
            # ─────────────────────────────────────────────────
            
            # Get new observation
            obs = self._get_observation()
            reward = self._calculate_reward(action)
            done, info = self._check_done()
            info.update(self._authority_metrics())

            if hasattr(self, 'render_mode') and self.render_mode == 'human':
                self._draw_vehicle_info()
            
            self._draw_road_boundaries_ahead()
            # --- comment either line out to remove the overlay -------------
            self._visualize_vehicle_footprint()   # white box + green/red edges
            # self._visualize_detected_obstacles()
            # self._visualize_lidar_obstacles()
            self._visualize_cbf_ellipses()
                
            if self.current_step % 5 == 0:
                self._visualize_mpc_prediction()

            
            return obs, reward, done, False, info
        except Exception as e:
            print(f"Error at step function with {e}")
            traceback.print_exc()
            last_valid_obs = self._get_safe_observation()
            return last_valid_obs, -100.0, True, False, {"done_reason": "error"}
        
    def _get_safe_observation(self):
        """Return a safe fallback observation - never zeros"""
        try:
            return self._get_observation()
        except:
            # Return a neutral observation - not zeros which can cause NaN
            obs = np.zeros(self.state_dim, dtype=np.float64)
            obs[0] = 100.0   # remaining distance - nonzero
            obs[3] = 5.0     # speed - nonzero  
            return obs
    
    def _destroy_actors(self):
        """Safely destroy all actors"""
        if hasattr(self, 'camera_sensor') and self.camera_sensor is not None:
            try:
                self.camera_sensor.stop()
                self.camera_sensor.destroy()
            except: pass
            self.camera_sensor = None
        if hasattr(self, 'obstacle_sensor') and self.obstacle_sensor is not None:
            try:
                self.obstacle_sensor.stop()
            except: pass
        if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
            except: pass
        if hasattr(self, 'lane_invasion_sensor') and self.lane_invasion_sensor is not None:
            try:
                self.lane_invasion_sensor.stop()
            except: pass

        actors_to_destroy = []
        
        # Collect sensors FIRST (destroy them before the vehicle they're attached to)
        if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
            actors_to_destroy.append(self.collision_sensor)
        
        if hasattr(self, 'lane_invasion_sensor') and self.lane_invasion_sensor is not None:
            actors_to_destroy.append(self.lane_invasion_sensor)
        
        if hasattr(self, 'obstacle_sensor') and self.obstacle_sensor is not None:
            actors_to_destroy.append(self.obstacle_sensor)
        
        # Then collect the vehicle
        if self.vehicle is not None:
            actors_to_destroy.append(self.vehicle)
        
        # NPC racers (stored as dicts)
        if hasattr(self, 'racing_npcs'):
            for npc_data in self.racing_npcs:
                npc = npc_data.get("actor", None)
                if npc is not None:
                    try:
                        if npc.is_alive:
                            actors_to_destroy.append(npc)
                    except (RuntimeError, AttributeError):
                        pass

        # Destroy everything safely BEFORE setting to None
        for actor in actors_to_destroy:
            try:
                if actor.is_alive:
                    actor.destroy()
            except (RuntimeError, AttributeError):
                pass

        # NOW set references to None AFTER destruction
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.obstacle_sensor = None
        self.sensor_detected_obstacles = []
        self.vehicle = None
        self.racing_npcs = []

        # Clear MPC controller to avoid stale references
        if self.mpc_controller is not None:
            if hasattr(self.mpc_controller, 'acados_solver') and self.mpc_controller.acados_solver is not None:
                del self.mpc_controller.acados_solver
                self.mpc_controller.acados_solver = None

        # self.mpc_controller = None
        if self.mpc_controller is not None:
            self.mpc_controller.last_control = np.zeros(2)
            self.mpc_controller.previous_control = np.zeros(2)
        self.frenet_converter = None
        # self.path_planner = None
        
        if hasattr(self, 'world') and self.world is not None:
            try:
                self.world.tick()
            except RuntimeError:
                pass

        gc.collect()
    
    def _load_random_town(self):
        print("Preparing to load new town...")
        
        # 1. STOP all sensor callbacks FIRST
        if hasattr(self, 'obstacle_sensor') and self.obstacle_sensor is not None:
            self.obstacle_sensor.stop()  # Stop listening before destroy
        if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
            self.collision_sensor.stop()
        if hasattr(self, 'lane_invasion_sensor') and self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.stop()
        
        # 2. Clear sensor data to prevent callback access
        self.sensor_detected_obstacles = []
        
        # 3. Switch to async BEFORE destroying
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

            tm = self.client.get_trafficmanager(8000)
            tm.set_synchronous_mode(False)
            
            # Let async mode settle
            time.sleep(0.1)
        except Exception as e:
            print(f"Switch to async failed with {e}")
            pass
        
        # 4. Destroy actors (now safely in async)
        self._destroy_actors()
        
        # 5. Wait for destruction to complete
        time.sleep(0.5)
        
        # 6. Load new world
        available = [t for t in self.available_towns if t != self.current_town]
        new_town = random.choice(available) if available else random.choice(self.available_towns)
        
        print(f"Loading new town: {new_town}")
        
        try:
            # Load world (this creates a completely new world object)
            self.world = self.client.load_world(new_town)
            self.map = self.world.get_map()
            self.current_town = new_town
            
            # 7. Wait for world to stabilize BEFORE re-enabling sync
            time.sleep(3.0)
            
            # 8. NOW re-enable sync mode
            self._setup_world()
            
            # 9. Tick to stabilize
            for _ in range(20):
                self.world.tick()
                time.sleep(0.05)
            
            print(f"✓ Successfully loaded {new_town}")
            
        except Exception as e:
            print(f"❌ Error loading new town: {e}")
            raise

    def render(self, mode='human', camera_mode='top_down'):
        """
        Args:
            mode: Rendering mode
            camera_mode: 'top_down', 'follow', 'side', 'first_person'
        """
        if self.vehicle is None:
            return
        
        vehicle_transform = self.vehicle.get_transform()
        spectator = self.world.get_spectator()
        
        if camera_mode == 'top_down':
            # Bird's eye view
            spectator_transform = carla.Transform(
                vehicle_transform.location + carla.Location(z=20),
                carla.Rotation(pitch=-90)
            )
        
        elif camera_mode == 'follow':
            # Behind and above the vehicle
            forward = vehicle_transform.get_forward_vector()
            spectator_transform = carla.Transform(
                vehicle_transform.location - forward * 10 + carla.Location(z=5),
                carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw)
            )
        
        elif camera_mode == 'side':
            # Side view
            right = vehicle_transform.get_right_vector()
            spectator_transform = carla.Transform(
                vehicle_transform.location + right * 10 + carla.Location(z=3),
                carla.Rotation(pitch=-10, yaw=vehicle_transform.rotation.yaw - 90)
            )
        
        elif camera_mode == 'first_person':
            # Driver's perspective
            spectator_transform = carla.Transform(
                vehicle_transform.location + carla.Location(z=1.5),
                vehicle_transform.rotation
            )
        
        else:
            # Default to top down
            spectator_transform = carla.Transform(
                vehicle_transform.location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            )
        
        spectator.set_transform(spectator_transform)
    
    def _visualize_path_and_goal(self, goal_point: carla.Transform):
        """Draw the path and goal in CARLA world"""
        debug = self.world.debug
        
        # 1. Draw goal point as a big red sphere
        debug.draw_point(
            goal_point.location,
            size=0.5,
            color=carla.Color(255, 0, 0),  # Red
            life_time=30.0  # Use 0.0 for a permanent marking
        )
        
        # 3. Draw the entire path as green dots
        if self.frenet_converter is not None:
            s_samples = np.linspace(0, self.path_length, 100)
            for s in s_samples:
                x, y, _ = self.frenet_converter.frenet_to_world(s, 0, 0)
                location = carla.Location(x=x, y=y, z=0.5)
                debug.draw_point(
                    location,
                    size=0.1,
                    color=carla.Color(0, 255, 0),  # Green
                    life_time=30.0
                )
        
        # 4. Draw start point as blue sphere
        if self.vehicle is not None:
            start_location = self.vehicle.get_transform().location
            debug.draw_point(
                start_location,
                size=0.5,
                color=carla.Color(0, 0, 255),  # Blue
                life_time=20.0
            )

    def _visualize_mpc_prediction(self):
        if self.mpc_controller is None:
            return
        
        debug = self.world.debug
        traj = self.mpc_controller.get_predicted_trajectory()

        # # Draw current car position in frenet coordination
        # loc = self.vehicle.get_transform().location
        # debug.draw_point(
        #     loc,
        #     size=0.15,
        #     color=carla.Color(255, 0, 255), # pink
        #     life_time=15.0
        # )

        # # Draw current car position in world coordination
        # s0, d0, a0 = self.current_s, self.current_d, self.current_alpha
        # x0, y0, _ = self.frenet_converter.frenet_to_world(s0, d0, a0)
        # debug.draw_point(
        #     carla.Location(x=x0, y=y0, z=0.8),
        #     size=0.12,
        #     color=carla.Color(0, 255, 255),  # cyan
        #     life_time=15.0
        # )

        # Draw MPC predicted trajectory (small yellow dots)
        for s, d in traj:
            x, y, _ = self.frenet_converter.frenet_to_world(s, d, 0.0)
            debug.draw_point(
                carla.Location(x=x, y=y, z=0.7),
                size=0.10,
                color=carla.Color(255, 0, 255),
                life_time=0.2
            )

    def _draw_road_boundaries_ahead(self, lookahead_distance=30.0):
        if self._road_widths is None or self._road_width_s is None or self.frenet_converter is None:
            return
        
        debug = self.world.debug
        
        s_samples = np.linspace(
            self.current_s,
            min(self.current_s + lookahead_distance, self.path_length),
            30
        )
        
        for s in s_samples:
            idx = np.argmin(np.abs(self._road_width_s - s))
            n_left  =  - self._road_widths[idx, 0]  # positive d = left (includes overtaking lane)
            n_right =  self._road_widths[idx, 1]  # negative d = right (ego lane edge only)
            
            # Left boundary (red) - should be ~5.25m out (full overtaking lane)
            x_left, y_left, _ = self.frenet_converter.frenet_to_world(s, n_left, 0)
            debug.draw_point(
                carla.Location(x=x_left, y=y_left, z=0.3),
                size=0.15,
                color=carla.Color(0, 0, 255),
                life_time=0.15
            )
            
            # Right boundary (blue) - should be ~1.75m out
            x_right, y_right, _ = self.frenet_converter.frenet_to_world(s, n_right, 0)
            debug.draw_point(
                carla.Location(x=x_right, y=y_right, z=0.3),
                size=0.15,
                color=carla.Color(0, 0, 255),
                life_time=0.15
            )
            
            # Centerline (green) - should be exactly on path
            x_c, y_c, _ = self.frenet_converter.frenet_to_world(s, 0.0, 0)
            debug.draw_point(
                carla.Location(x=x_c, y=y_c, z=0.3),
                size=0.10,
                color=carla.Color(0, 255, 0),
                life_time=0.15
            )
            
            # Lane divider (yellow) - left edge of ego lane
            # lane_divider_d = - self._road_widths[idx, 1]  # right_width = lane_width/2
            # x_div, y_div, _ = self.frenet_converter.frenet_to_world(s, lane_divider_d, 0)
            # debug.draw_point(
            #     carla.Location(x=x_div, y=y_div, z=0.3),
            #     size=0.06,
            #     color=carla.Color(255, 255, 0),
            #     life_time=0.15
            # )

    def _visualize_vehicle_footprint(self, life_time=0.06):
        """
        Draw the ego's real bounding box AND the lateral extent the corridor
        constraint actually reasons about, so "the box must not exceed the lane"
        can be checked by eye.

        The constraint is on the vehicle CENTRE with a margin of
            (L/2)|sin a| + (W/2)|cos a| + clearance
        so what matters is not the box itself but where its left/right edges land
        relative to the corridor.  Those edges are drawn as points in the Frenet
        frame at the ego's current s:

            GREEN  edge is inside the corridor  -> constraint holding
            RED    edge is outside              -> the corner is over the line

        Compare the coloured points against the blue corridor dots from
        _draw_road_boundaries_ahead(): green points should always sit inside them.
        A red point with the solver reporting success means the margin is still
        under-sized for that heading error.
        """
        if self.vehicle is None or self.frenet_converter is None:
            return
        if self._road_widths is None or self._road_width_s is None:
            return

        try:
            debug = self.world.debug
            tf = self.vehicle.get_transform()

            # 1. the actual CARLA bounding box, for reference
            bb = self.vehicle.bounding_box
            # life_time just over one step (dt = 0.05) so successive frames do
            # not stack -- overlapping debug draws add up and look like a glow.
            # A dimmer white for the same reason; 255 saturates.
            debug.draw_box(
                carla.BoundingBox(tf.transform(bb.location), bb.extent),
                tf.rotation,
                thickness=0.03,
                color=carla.Color(160, 160, 160),
                life_time=life_time,
            )

            # 2. the modelled lateral extent, using the SAME formula the
            #    controller applies when it sizes the margin
            mpc = self.mpc_controller
            L = getattr(mpc, 'veh_length', 2.0 * bb.extent.x * 2.0)
            W = getattr(mpc, 'veh_width', 2.0 * bb.extent.y * 2.0)
            clearance = getattr(mpc, 'lateral_clearance', 0.30)
            a = self.current_alpha
            half = 0.5 * L * abs(np.sin(a)) + 0.5 * W * abs(np.cos(a))

            idx = np.argmin(np.abs(self._road_width_s - self.current_s))
            n_min = -self._road_widths[idx, 0]
            n_max = self._road_widths[idx, 1]

            for sign in (-1.0, +1.0):
                n_edge = self.current_d + sign * half
                # the constraint also demands `clearance` beyond the edge
                inside = (n_edge - clearance >= n_min) if sign < 0 else \
                         (n_edge + clearance <= n_max)
                colour = carla.Color(0, 255, 0) if inside else carla.Color(255, 0, 0)
                x, y, _ = self.frenet_converter.frenet_to_world(
                    self.current_s, n_edge, 0.0)
                debug.draw_point(
                    carla.Location(x=x, y=y, z=1.2),
                    size=0.08, color=colour, life_time=life_time)
        except Exception:
            pass

    def _visualize_detected_obstacles(self):
        """Visualize obstacles with different colors based on detection method"""
        if self.frenet_converter is None or self.vehicle is None:
            return
        
        debug = self.world.debug
        ego_loc = self.vehicle.get_transform().location
        
        # Visualize sensor-detected obstacles
        for obs_data in self.sensor_detected_obstacles:
            try:
                if obs_data['actor'].is_alive:
                    loc = obs_data['location']
                    # Green for sensor-detected
                    debug.draw_point(
                        carla.Location(x=loc.x, y=loc.y, z=1.0),
                        size=0.25,
                        color=carla.Color(0, 255, 0),  # Green
                        life_time=0.1
                    )

                        # Draw line from ego to obstacle
                    debug.draw_line(
                        ego_loc,
                        obs_loc,
                        thickness=0.03,
                        color=carla.Color(255, 255, 0),  # Yellow
                        life_time=0.1
                    )
            except:
                pass
        
        # Visualize selected obstacles (in observation)
        for i, obs in enumerate(self.selected_obstacles):
            if obs[0] > -50:  # Valid obstacle
                s_obs, d_obs = obs[0], obs[1]
                
                x, y, _ = self.frenet_converter.frenet_to_world(s_obs, d_obs, 0.0)
                obs_loc = carla.Location(x=x, y=y, z=1.0)
                
                # Red for obstacles in observation
                debug.draw_point(
                    obs_loc,
                    size=0.2,
                    color=carla.Color(255, 0, 0),  # Red
                    life_time=0.1
                )
                
                # Draw line from ego to obstacle
                debug.draw_line(
                    ego_loc,
                    obs_loc,
                    thickness=0.03,
                    color=carla.Color(255, 255, 0),  # Yellow
                    life_time=0.1
                )

    def _debug_road_boundaries(self, every_n_steps=50):
        """Compare road widths used in MPC constraints vs draw visualization"""
        if self.current_step % every_n_steps != 0:
            return
        if self._road_widths is None or self._road_width_s is None:
            return

        print(f"\n{'='*60}")
        print(f"[STEP {self.current_step}] ROAD BOUNDARY DEBUG @ s={self.current_s:.2f}")
        print(f"{'='*60}")

        # ---- GLOBAL PATH STATS (one-time overview) ----
        lw_all = self._road_widths[:, 0]  # left widths
        rw_all = self._road_widths[:, 1]  # right widths
        print(f"[GLOBAL] Left  width: min={lw_all.min():.2f}, max={lw_all.max():.2f}, mean={lw_all.mean():.2f}")
        print(f"[GLOBAL] Right width: min={rw_all.min():.2f}, max={rw_all.max():.2f}, mean={rw_all.mean():.2f}")

        # ---- CURRENT POSITION LOOKUP ----
        idx = np.argmin(np.abs(self._road_width_s - self.current_s))
        lw = self._road_widths[idx, 0]
        rw = self._road_widths[idx, 1]

        safety_margin = 1.0

        # What MPC constraint uses (from solve() loop):
        mpc_n_min = -lw + safety_margin   # negative = left boundary
        mpc_n_max =  rw - safety_margin   # positive = right boundary
        mpc_n_min_clamped = max(mpc_n_min, -10.0)
        mpc_n_max_clamped = min(mpc_n_max,  10.0)

        # What draw uses (from _draw_road_boundaries_ahead):
        draw_n_left  = -lw + safety_margin   # left boundary dot
        draw_n_right =  rw - safety_margin   # right boundary dot

        print(f"\n[CURRENT s={self.current_s:.2f}, idx={idx}]")
        print(f"  Raw road widths:  left={lw:.3f}m, right={rw:.3f}m")
        print(f"  MPC n_min (left boundary) : {mpc_n_min:.3f} → clamped: {mpc_n_min_clamped:.3f}")
        print(f"  MPC n_max (right boundary): {mpc_n_max:.3f} → clamped: {mpc_n_max_clamped:.3f}")
        print(f"  Draw n_left  (red dot d)  : {draw_n_left:.3f}")
        print(f"  Draw n_right (blue dot d) : {draw_n_right:.3f}")
        print(f"  MATCH: {np.isclose(mpc_n_min, draw_n_left) and np.isclose(mpc_n_max, draw_n_right)}")

        # ---- CURRENT EGO POSITION ----
        print(f"\n[EGO] current_d={self.current_d:.3f}")
        print(f"  Inside MPC bounds? {mpc_n_min_clamped < self.current_d < mpc_n_max_clamped}")
        print(f"  Distance to left wall : {self.current_d - mpc_n_min_clamped:.3f}m")
        print(f"  Distance to right wall: {mpc_n_max_clamped - self.current_d:.3f}m")

        # ---- LOOKAHEAD: what MPC will use for future steps ----
        print(f"\n[MPC HORIZON PREVIEW] (next {self.mpc_horizon} steps)")
        print(f"  {'i':>3}  {'s_pred':>8}  {'n_min':>8}  {'n_max':>8}  {'width':>8}")
        for i in range(1, self.mpc_horizon, 5):  # every 5 steps
            s_pred = self.current_s + self.target_speed * (self.mpc_dt * self.mpc_horizon / self.mpc_horizon) * i
            idx_pred = np.argmin(np.abs(self._road_width_s - s_pred))
            lw_pred = self._road_widths[idx_pred, 0]
            rw_pred = self._road_widths[idx_pred, 1]
            n_min_p = max(-lw_pred + safety_margin, -10.0)
            n_max_p = min( rw_pred - safety_margin,  10.0)
            print(f"  {i:>3}  {s_pred:>8.2f}  {n_min_p:>8.3f}  {n_max_p:>8.3f}  {n_min_p+n_max_p:>8.3f}")

        print(f"{'='*60}\n")

    def _visualize_cbf_ellipses(self):
        """Draw CBF ellipses around each obstacle in CARLA world"""
        if self.frenet_converter is None:
            return
        
        debug = self.world.debug
        
        a_long = 4  # Must match bicycle_model_mpcc_cbf.py
        b_lat = 2   # Must match bicycle_model_mpcc_cbf.py
        
        for obs in self.selected_obstacles:
            if obs[0] < -50:  # Invalid obstacle
                continue
            
            s_obs, n_obs = obs[0], obs[1]
            
            # Draw ellipse by sampling points around it
            num_points = 36
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                
                s_ellipse = s_obs + a_long * np.cos(angle)
                n_ellipse = n_obs + b_lat * np.sin(angle)
                
                x1, y1, _ = self.frenet_converter.frenet_to_world(
                    s_ellipse, n_ellipse, 0.0)
                
                debug.draw_point(
                    carla.Location(x=x1, y=y1, z=0.5),
                    size=0.05,                    # adjust size as needed
                    color=carla.Color(255, 0, 0), # solid red
                    life_time=0.15
                )
            
            # # Also draw 2x safety distance (where repulsive cost kicks in)
            # for i in range(num_points):
            #     angle = 2 * np.pi * i / num_points
            #     angle_next = 2 * np.pi * (i + 1) / num_points
                
            #     s_outer = s_obs + (a_long * 2) * np.cos(angle)
            #     n_outer = n_obs + (b_lat * 2) * np.sin(angle)
                
            #     s_outer_next = s_obs + (a_long * 2) * np.cos(angle_next)
            #     n_outer_next = n_obs + (b_lat * 2) * np.sin(angle_next)
                
            #     x1, y1, _ = self.frenet_converter.frenet_to_world(s_outer, n_outer, 0.0)
            #     x2, y2, _ = self.frenet_converter.frenet_to_world(s_outer_next, n_outer_next, 0.0)
                
            #     debug.draw_line(
            #         carla.Location(x=x1, y=y1, z=0.5),
            #         carla.Location(x=x2, y=y2, z=0.5),
            #         thickness=0.03,
            #         color=carla.Color(255, 255, 0),  # Yellow = repulsive cost zone
            #         life_time=0.1
            #     )

    def close(self):
        try:
            # Flush the in-progress episode's diagnostics.  reset() only saves
            # the *previous* episode, so without this the run you were watching
            # when you hit Ctrl+C -- usually the interesting one -- is lost.
            if self.mpc_controller is not None:
                diag = getattr(self.mpc_controller, 'diagnostics', None)
                if diag is not None:
                    diag.save(tag=f"ep{self.episode_count:04d}_final")

            self._destroy_actors()

            # Extra safety: remove any leftover vehicles
            if self.world is not None:
                for v in self.world.get_actors().filter('vehicle.*'):
                    try:
                        if v.is_alive:
                            v.destroy()
                    except:
                        pass
                self.world.tick()

            # Restore async mode
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)

            print("✅ CARLA cleaned up")
        except Exception as e:
            print("⚠️ Error during env.close():", e)
