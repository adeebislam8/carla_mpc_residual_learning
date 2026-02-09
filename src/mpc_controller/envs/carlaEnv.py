import gymnasium as gym
import numpy as np
import carla
import random
import time
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
from scipy.interpolate import make_interp_spline
import sys

sys.path.append('/home/ave/Desktop/carla_mpc_residual_learning/src')

from global_planner.src.frenet_world_converter_python import FrenetConverter
from global_planner.src.global_path_publisher_python import PathPlanner
from mpc_controller.src.mpc_controller_python import MPCController



class CarlaMPCEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 2000,
        timeout: float = 10.0,
        towns: List[str] = ['Town01', 'Town02', 'Town03', 'Town04'],
        episodes_per_town: int = 99999,
        target_speed: float = 8.33,  # m/s
        max_steps: int = 1000,
        lookahead_distance: float = 100.0,
        num_obstacles: int = 6,
        state_dim: int = 66,
        mpc_horizon: int = 30,
        mpc_dt: float = 0.05,
        discrete_actions: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        # CARLA connection
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
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
        self.lane_invasion_sensor = None
        self.current_speed = 0.0
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.current_steering = 0.0
        
        self._road_widths = None
        
        # Frenet frame
        self.frenet_converter: Optional[FrenetConverter] = None
        self.path_planner: Optional[PathPlanner] = None
        self.current_s = 0.0
        self.current_d = 0.0
        self.current_alpha = 0.0
        self.path_length = 0.0
        
        # MPC controller
        self.mpc_controller: Optional[MPCController] = None
        self.mpc_prediction = np.zeros((mpc_horizon, 2))
        
        # Obstacles
        self.selected_obstacles = np.ones((num_obstacles, 2)) * -100
        
        # Episode state
        self.collision = False
        self.lane_invasion = False
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
        
        # Set up CARLA world
        self._setup_world()
        
    def _setup_world(self):
        """Configure CARLA world for synchronous mode"""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.mpc_dt
        self.world.apply_settings(settings)
        
        # Enable traffic manager in sync mode
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        
    def _spawn_ego_vehicle(self, spawn_point: carla.Transform) -> bool:
        """Spawn ego vehicle at given spawn point"""
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
        
        self.world.tick()
    
    def _on_collision(self, event):
        """Collision callback"""
        self.collision = True
    
    def _on_lane_invasion(self, event):
        """Lane invasion callback - only trigger for solid markings"""
        for marking in event.crossed_lane_markings:
            if marking.type == carla.LaneMarkingType.Solid:
                self.lane_invasion = True
                break
    
    def _generate_path(self, start: carla.Transform, goal: carla.Transform):
        """Generate global path from start to goal"""
        self.path_planner = PathPlanner(self.world, self.map)
        waypoints = self.path_planner.calculate_route(start.location, goal.location)
        
        if not waypoints or len(waypoints) < 4:
            print("Failed to generate valid path")
            return False
        
        # Initialize Frenet converter with waypoints
        waypoint_coords = [[wp.transform.location.x, wp.transform.location.y] 
                        for wp in waypoints]
        self.frenet_converter = FrenetConverter(waypoint_coords)
        self.path_length = self.frenet_converter.get_path_length()
        
        self._road_widths = []
        for wp in waypoints:
            left_width, right_width = self.path_planner.get_road_width_at_waypoint(wp)
            self._road_widths.append([left_width, right_width])
        self._road_widths = np.array(self._road_widths)
        
        return True
    
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
        
        # Initialize ACADOS
        self.mpc_controller.initialize_acados(kappa_spline, path_msg=None)
    
    def _update_vehicle_state(self):
        """Update vehicle state from CARLA"""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        control = self.vehicle.get_control()
        
        # Convert to Frenet
        x = transform.location.x
        y = transform.location.y
        yaw = np.deg2rad(transform.rotation.yaw)
        
        self.current_s, self.current_d, self.current_alpha = self.frenet_converter.world_to_frenet(x, y, yaw)
        
        # Vehicle dynamics
        self.current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self.current_throttle = control.throttle
        self.current_brake = control.brake
        self.current_steering = control.steer * (45 * np.pi / 180) # Change to radians
    
    def _detect_obstacles(self):
        """Detect and select closest obstacles in Frenet frame"""
        self.selected_obstacles = np.ones((self.num_obstacles, 2)) * -100
        
        # Get all vehicles in world
        vehicles = self.world.get_actors().filter('vehicle.*')
        ego_location = self.vehicle.get_location()
        
        obstacles = []
        for vehicle in vehicles:
            if vehicle.id == self.vehicle.id:
                continue
            
            loc = vehicle.get_location()
            
            # Euclidean distance check
            distance = np.sqrt(
                (ego_location.x - loc.x)**2 + 
                (ego_location.y - loc.y)**2
            )
            
            if distance < 50.0:  # Within 50m
                # Convert to Frenet
                s_obs, d_obs, _ = self.frenet_converter.world_to_frenet(
                    loc.x, loc.y, 0
                )
                
                # Must be ahead and in relevant lateral range
                if s_obs > self.current_s and abs(d_obs) < 5.0:
                    obstacles.append([s_obs, d_obs, s_obs - self.current_s])
        
        # Sort by distance and select closest N
        obstacles.sort(key=lambda x: x[2])
        for i, obs in enumerate(obstacles[:self.num_obstacles]):
            self.selected_obstacles[i] = [obs[0], obs[1]]
    
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
            20
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
            [self.current_speed, 0.0, self.current_throttle, 
             self.current_brake, self.current_steering],  # Vehicle state
            kappa_samples,  # Curvature
            [self.current_step],  # Step count
            mpc_pred_flat  # MPC prediction
        ])
        
        return obs.astype(np.float64)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        reward = 0.0
        
        # Collision penalty
        if self.collision:
            return -100.0
        
        # Progress reward
        progress = self.current_s - self.prev_s
        reward += progress * 10.0
        
        # Lane keeping
        if not (self.current_d < 3.5 and self.current_d > -0.5):
            reward -= 5.0
        
        # Obstacle avoidance
        for obs in self.selected_obstacles:
            if obs[0] > -50:  # Valid obstacle
                dist = np.sqrt(
                    (self.current_s - obs[0])**2 + 
                    ((self.current_d - obs[1]) * 2)**2
                )
                if dist < 6.0:
                    reward -= 5.0
        
        # Heading alignment penalty
        reward -= abs(self.current_alpha) * 5.0
        
        # Speed penalty (stalling)
        if self.current_speed < 1.0:
            reward -= 10.0
        
        # Speed bonus
        if self.current_speed > 5.0:
            reward += 0.001 * self.current_speed**2
        
        # Goal reached bonus
        if self.current_s >= self.path_length - 10:
            reward += 1000.0
        
        # Time penalty
        reward -= self.current_step * 0.005
        
        return reward
    
    def _check_done(self) -> Tuple[bool, Dict]:
        info = {}
        
        # Collision
        if self.collision:
            elapsed = time.time() - self.start_time
            return True, {"done_reason": "collision", "lap_time": elapsed}
        
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
            if self.episode_count % self.episodes_per_town == 0 and self.episode_count > 0:
                self._load_random_town()
            
            # Destroy old vehicle and sensors
            self._destroy_actors()
            
            # Find valid spawn and goal
            max_attempts = 30
            for attempt in range(max_attempts):
                spawn_points = self.map.get_spawn_points()
                spawn_point = random.choice(spawn_points)
                goal_point = random.choice(spawn_points)
                # spawn_point = carla.Transform(
                #     carla.Location(x=10.912545, y=-57.401386, z=0.600000),
                #     carla.Rotation(pitch=0.0, yaw=-0.023438, roll=0.0)
                # )

                # goal_point = carla.Transform(
                #     carla.Location(x=-66.794197, y=12.998389, z=0.600000),
                #     carla.Rotation(pitch=0.0, yaw=-179.840790, roll=0.0)
                # )
                
                # Ensure minimum distance
                dist = np.sqrt(
                    (spawn_point.location.x - goal_point.location.x)**2 +
                    (spawn_point.location.y - goal_point.location.y)**2
                )
                
                if dist > 100.0:
                    # print(f"spawn: {spawn_point}")
                    # print(f"goal: {goal_point}")
                    # Spawn vehicle
                    if not self._spawn_ego_vehicle(spawn_point):
                        continue
                    
                    # Attach sensors
                    self._attach_sensors()
                    
                    # Generate path
                    if not self._generate_path(spawn_point, goal_point):
                        self.vehicle.destroy()
                        continue
                    
                    # Initialize MPC
                    self._initialize_mpc()
                    self._visualize_path_and_goal(goal_point)

                    break
            else:
                raise RuntimeError("Failed to find valid spawn after 30 attempts")
            
            # Reset state
            self.current_step = 0
            self.collision = False
            self.lane_invasion = False
            self.start_time = time.time()
            self.prev_s = 0.0
            
            # Initial tick
            self.world.tick()
            self._update_vehicle_state()
            self._detect_obstacles()
            
            obs = self._get_observation()
            return obs, {}
        except Exception as e:
            print(f"Error at reset with {e}")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        try:
            # vehicles = self.world.get_actors().filter('vehicle.*')
            # print("Vehicles in world:", len(vehicles))
            # for v in vehicles:
            #     print(v.id, v.type_id)

            self.current_step += 1
            self.prev_s = self.current_s
            
            # Run MPC to get base control
            self._update_vehicle_state()
            self._detect_obstacles()

            mpc_throttle, mpc_steering = self.mpc_controller.solve(
                s=self.current_s,
                d=self.current_d,
                alpha=self.current_alpha,
                v=self.current_speed,
                obstacles=self.selected_obstacles,
                D=self.current_throttle,
                delta=self.current_steering,
                road_widths=self._road_widths  # None is okay, MPC will use defaults
            )

            # print("="*20)
            # print(f"MPC Throttle: {mpc_throttle}\nMPC Steering: {mpc_steering}")
            # print("="*20)

            # Apply residual from RL
            residual_throttle = action[0] * 0.1
            residual_steering = action[1] * 0.1
            
            final_throttle = np.clip(mpc_throttle + residual_throttle, -1.0, 1.0)
            final_steering = np.clip(mpc_steering + residual_steering, -1.0, 1.0)
            
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
            
            # Get new observation
            obs = self._get_observation()
            reward = self._calculate_reward(action)
            done, info = self._check_done()

            if hasattr(self, 'render_mode') and self.render_mode == 'human':
                self._draw_vehicle_info()
                
            if self.current_step % 5 == 0:
                self._visualize_mpc_prediction()

            
            return obs, reward, done, False, info
        except Exception as e:
            print(f"Error at step function with {e}")
    
    def _destroy_actors(self):
        """Safely destroy all actors"""
        actors_to_destroy = []
        
        # Collect actors
        if self.collision_sensor is not None:
            actors_to_destroy.append(self.collision_sensor)
            self.collision_sensor = None
        
        if self.lane_invasion_sensor is not None:
            actors_to_destroy.append(self.lane_invasion_sensor)
            self.lane_invasion_sensor = None
        
        if self.vehicle is not None:
            actors_to_destroy.append(self.vehicle)
            self.vehicle = None
        
        # Destroy in batch
        for actor in actors_to_destroy:
            if actor.is_alive:  # Check if still valid
                actor.destroy()
        
        # Clear MPC controller to avoid stale references
        self.mpc_controller = None
        self.frenet_converter = None
        self.path_planner = None
        
        if hasattr(self, 'world') and self.world is not None:
            self.world.tick()

    
    def _load_random_town(self):
        print("Preparing to load new town...")
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.stop()
        
        self._destroy_actors()
        for _ in range(5):
            self.world.tick()
            time.sleep(0.02)

        available = [t for t in self.available_towns if t != self.current_town]
        new_town = random.choice(available) if available else random.choice(self.available_towns)
        
        print(f"Loading new town: {new_town}")
        
        try:
            self.world = self.client.load_world(new_town)
            self.map = self.world.get_map()
            self.current_town = new_town
            self._setup_world()
            
            print("Waiting for world to stabilize...")
            for _ in range(20):
                self.world.tick()
                time.sleep(0.05)
            
            print(f"✓ Successfully loaded {new_town}")
            
        except Exception as e:
            print(f"❌ Error loading new town: {e}")
            
            # Try to recover by reloading current town
            try:
                self.world = self.client.load_world(self.current_town)
                self.map = self.world.get_map()
                self._setup_world()
                for _ in range(20):
                    self.world.tick()
                    time.sleep(0.05)
            except:
                raise RuntimeError(f"Failed to load town and unable to recover: {e}")

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
                vehicle_transform.location + carla.Location(z=50),
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
            life_time=20.0  # Use 0.0 for a permanent marking
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
                    life_time=20.0
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

        # Draw current car position in frenet coordination
        loc = self.vehicle.get_transform().location
        debug.draw_point(
            loc,
            size=0.15,
            color=carla.Color(255, 255, 0),
            life_time=15.0
        )

        # Draw current car position in world coordination
        s0, d0, a0 = self.current_s, self.current_d, self.current_alpha
        x0, y0, _ = self.frenet_converter.frenet_to_world(s0, d0, a0)
        debug.draw_point(
            carla.Location(x=x0, y=y0, z=0.8),
            size=0.12,
            color=carla.Color(0, 255, 255),  # cyan
            life_time=15.0
        )

        # Draw MPC predicted trajectory (small yellow dots)
        for s, d in traj:
            x, y, _ = self.frenet_converter.frenet_to_world(s, d, 0.0)
            debug.draw_point(
                carla.Location(x=x, y=y, z=0.7),
                size=0.08,
                color=carla.Color(255, 255, 0),  # yellow
                life_time=0.2
            )

    def close(self):
        try:
            # Stop sensors
            if self.collision_sensor is not None:
                self.collision_sensor.stop()
            if self.lane_invasion_sensor is not None:
                self.lane_invasion_sensor.stop()

            # Destroy ego + sensors
            self._destroy_actors()

            # Extra safety: remove any leftover vehicles
            if self.world is not None:
                for v in self.world.get_actors().filter('vehicle.*'):
                    try:
                        v.destroy()
                    except:
                        pass
                self.world.tick()

            # Restore async mode so CARLA isn't stuck in sync
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)

            print("✅ CARLA cleaned up")
        except Exception as e:
            print("⚠️ Error during env.close():", e)
