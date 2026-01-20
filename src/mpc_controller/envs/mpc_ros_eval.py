import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append('/home/adeeb/carla-ros-bridge/catkin_ws/')
import gymnasium as gym
import rospy
from stable_baselines3 import SAC,PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.envs.registration import register
import signal
import atexit
import time


from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
import rospy
import numpy as np
from gymnasium import spaces
import carla
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus, CarlaCollisionEvent, CarlaLaneInvasionEvent  # pylint: disable=import-error
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64, Int16
from visualization_msgs.msg import Marker, MarkerArray
from global_planner.msg import FrenetPose, WorldPose
from global_planner.srv import World2FrenetService, Frenet2WorldService
from convert_traj_track import parseReference
from scipy.interpolate import make_interp_spline
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import carla_common.transforms
import threading
import traceback
import random
secure_random = random.SystemRandom()

DEG2RAD = np.pi/180.0
RAD2DEG = 180.0/np.pi
DIST2OBSTACLE = 6.0
PATH_LENGTH = 100.0

shutdown_requested = False

def sigint_handler(sig, frame):
    global shutdown_requested
    rospy.loginfo("Shutdown requested via SIGINT...")
    shutdown_requested = True
    rospy.signal_shutdown("SIGINT")

signal.signal(signal.SIGINT, sigint_handler)

file_path = "/home/ave/Desktop/carla_mpc_residual_learning/debug.txt"
with open(file_path, "w") as file:
    file.write("")

class mpcGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.nmax = 3.5
        self.nmin = -0.5
        self.steer_max = 30 * DEG2RAD
        self.kappa_spline = None
        self.current_s = 0
        self.prev_s = 0
        self.current_d = 0
        self.lookahead_distance = 100
        self.num_of_obs = 3
        self.collision = False
        self.lane_invasion = False
        self.mpc_control_initialized = False
        self.selected_obstacles_initialized = False
        self.predicted_path_initialized = False
        self.ego_state_initialized = False
        self.frenet_pose_initialized = False
        self.ref_path_initialized = False
        self.current_observation = None
        self.current_ego_state_info = np.zeros(5)
        self.mpc_control = np.zeros(2)
        self.prev_mpc_control = np.zeros(2)
        self.max_steps = 1000
        self.current_step = 0
        self.data_lock = threading.Lock()
        self.acados_init = False
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        self.speed_count = 0
        self.state_dim = 60
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.ones(self.state_dim) * -np.inf,
            high=np.ones(self.state_dim) * np.inf,
            shape=(self.state_dim,),
            dtype=np.float64)
        print("Action space: ", self.action_space)
        self.setup_ros()

    def setup_ros(self):
        self.ego_vehicle_status_subscriber = rospy.Subscriber(
            '/carla/ego_vehicle/vehicle_status', CarlaEgoVehicleStatus, self.ego_state_callback, queue_size=1)
        self.selected_obstacles_subscriber = rospy.Subscriber(
            '/mpc_controller/ego_vehicle/selected_obstacles', MarkerArray, self.selected_obstacles_callback, queue_size=1)
        self.reference_path_subscriber = rospy.Subscriber(
            '/global_planner/ego_vehicle/waypoints', Path, self.reference_path_callback, queue_size=1)
        self.predicted_path_subscriber = rospy.Subscriber(
            '/mpc_controller/ego_vehicle/predicted_path', Path, self.predicted_path_callback, queue_size=1)
        self.mpc_control_cmd_subscriber = rospy.Subscriber(
            '/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, self.mpc_control_cmd_callback, queue_size=1)
        self.frenet_state_subscriber = rospy.Subscriber(
            "/global_planner/ego_vehicle/frenet_pose", FrenetPose, self.frenet_state_callback, queue_size=1)
        self.lane_invasion_subscriber = rospy.Subscriber(
            '/carla/ego_vehicle/lane_invasion', CarlaLaneInvasionEvent, self.lane_invasion_callback, queue_size=1)
        self.collision_subscriber = rospy.Subscriber(
            '/carla/ego_vehicle/collision', CarlaCollisionEvent, self.collision_callback, queue_size=1)
        self.acados_init_subscriber = rospy.Subscriber(
            '/mpc_rl/acados_init', Int16, self.acados_init_callback, queue_size=1)

        self.action_publisher = rospy.Publisher('/mpc_rl/residual', Float32MultiArray, queue_size=1)
        self.action_stop_publisher = rospy.Publisher('/mpc_rl/emergency_stop', Int16, queue_size=1)
        self.goal_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.initial_pose_publisher = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)

        self.world2frenet_service = rospy.ServiceProxy('/world2frenet', World2FrenetService)
        self.frenet2world_service = rospy.ServiceProxy('/frenet2world', Frenet2WorldService)
        print("Waiting for services")
        rospy.wait_for_service('/world2frenet')
        rospy.wait_for_service('/frenet2world')
        print("Services available")
        rospy.loginfo("Initialized the MPC Gym environment.")
        print("Initialized the MPC Gym environment.")
        ob, _ = self.reset()
        print("Observation_reset initialized: ", ob)

    def acados_init_callback(self, msg):
        with self.data_lock:
            self.acados_init = msg.data

    def emergency_stop(self):
        control_msg = Int16()
        control_msg.data = 1
        self.action_stop_publisher.publish(control_msg)

    def clear_spawn_point(self, spawn_location, radius=5.0):
        """Remove vehicles from spawn and optionally relocate them"""
        try:
            all_vehicles = self.world.get_actors().filter('vehicle.*')
            spawn_points = self.map.get_spawn_points()
            
            cleared_count = 0
            for vehicle in all_vehicles:
                # Skip ego vehicle
                if vehicle.attributes.get('role_name') == 'ego_vehicle':
                    continue
                
                # Check distance to spawn point
                vehicle_loc = vehicle.get_location()
                distance = np.sqrt(
                    (vehicle_loc.x - spawn_location.x)**2 + 
                    (vehicle_loc.y - spawn_location.y)**2
                )
                
                if distance < radius:
                    # Teleport to random far location
                    far_spawn = secure_random.choice(spawn_points)
                    while self._distance_2d(far_spawn.location, spawn_location) < 50:
                        far_spawn = secure_random.choice(spawn_points)
                    
                    vehicle.set_transform(far_spawn)
                    rospy.loginfo(f"Relocated vehicle {distance:.2f}m away")
                    cleared_count += 1
            
            if cleared_count > 0:
                rospy.sleep(0.3)
                rospy.loginfo(f"Cleared {cleared_count} vehicles from spawn area")
            
            return True
        except Exception as e:
            rospy.logerr(f"Failed to clear spawn point: {e}")
            return False

    def _distance_2d(self, loc1, loc2):
        """Helper to calculate 2D distance"""
        return np.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)
    
    def reset_vehicle(self):
        try: 
            speed = self.current_ego_state_info[0]
            start_time = time.time()
            timeout = 10
            self.emergency_stop()

            while abs(speed) > 0.5 and not shutdown_requested:
                if time.time() - start_time > timeout:
                    rospy.logwarn("Timeout waiting for vehicle to stop")
                    break
                
                self.emergency_stop()
                print("reset_veh emer_stop")
                speed = self.current_ego_state_info[0]
                rospy.sleep(0.5)
                if shutdown_requested:
                    return

            # Generate random spawn and goal points (CARLA format)
            start_point = self.generate_random_spawn_point_carla()
            goal_point = self.generate_random_goal_point_carla(start_point)
            
            print("Start point: ", start_point)
            print("Goal point: ", goal_point)
            
            self.clear_spawn_point(
                carla.Location(x=start_point.location.x, y=start_point.location.y, z=start_point.location.z), 
                radius=10.0
            )
            
            # Convert to ROS format
            start_point_ros = carla_common.transforms.carla_transform_to_ros_pose(start_point)
            spawn_pose = self.carla_spawn_to_ros_pose(start_point_ros)
            self.initial_pose_publisher.publish(spawn_pose)
            
            rospy.sleep(0.5)  # Wait for spawn to register
            
            # Publish goal
            goal_point_ros = carla_common.transforms.carla_transform_to_ros_pose(goal_point)
            goal_pose = self.carla_goal_to_ros_pose(goal_point_ros)
            self.goal_publisher.publish(goal_pose)
            
            emergency_stop_signal = Int16()
            emergency_stop_signal.data = 0
            self.action_stop_publisher.publish(emergency_stop_signal)

            if rospy.is_shutdown():
                return

            rospy.loginfo("Resetting the vehicle to a random spawn point and goal point.")
            rospy.sleep(1.5)  # Increased for path planning
            
        except Exception as e:
            rospy.logerr(f"Error in reset_vehicle: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if not shutdown_requested and not rospy.is_shutdown():
                emergency_stop_signal = Int16()
                emergency_stop_signal.data = 0
                self.action_stop_publisher.publish(emergency_stop_signal)
                rospy.loginfo("Emergency stop released")

    def distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def generate_random_spawn_point(self):
        spawn_points = self.map.get_spawn_points()
        spawn_point = secure_random.choice(spawn_points) if spawn_points else carla.Transform()
        spawn_point = carla_common.transforms.carla_transform_to_ros_pose(spawn_point)
        return spawn_point

    def generate_random_goal_point(self, random_spawn_point):
        spawn_points = self.map.get_spawn_points()
        goal_point = secure_random.choice(spawn_points) if spawn_points else carla.Transform()
        goal_point = carla_common.transforms.carla_transform_to_ros_pose(goal_point)

        while self.distance(random_spawn_point.position, goal_point.position) < PATH_LENGTH:
            goal_point = secure_random.choice(spawn_points) if spawn_points else carla.Transform()
            goal_point = carla_common.transforms.carla_transform_to_ros_pose(goal_point)
            rospy.loginfo("Random goal point is too close to the spawn point. Randomizing goal point again.")

        return goal_point

    def carla_spawn_to_ros_pose(self, carla_pose):
        ros_pose = PoseWithCovarianceStamped()
        ros_pose.header.frame_id = "map"
        ros_pose.pose.pose = carla_pose
        return ros_pose

    def carla_goal_to_ros_pose(self, carla_pose):
        ros_pose = PoseStamped()
        ros_pose.header.frame_id = "map"
        ros_pose.pose = carla_pose
        return ros_pose

    def _get_world_pose(self, frenet_pose):
        request = FrenetPose()
        request.s = frenet_pose.s
        request.d = frenet_pose.d
        response = self.frenet2world_service(request)
        return response.pose

    def _get_frenet_pose(self, pose):
        request = WorldPose()
        request.x = pose.position.x
        request.y = pose.position.y
        _, _, yaw = euler_from_quaternion([pose.orientation.x, pose.orientation.y,
                                           pose.orientation.z, pose.orientation.w])
        request.yaw = yaw

        response = self.world2frenet_service(request)
        if response is None:
            self.loginfo("Failed to get frenet pose")
            dummy = FrenetPose()
            dummy.s = 0
            dummy.d = 0
            return dummy
        return response.frenet_pose

    def ego_state_callback(self, msg):
        with self.data_lock:
            self.current_speed = msg.velocity
            self.current_accel = np.sqrt(msg.acceleration.linear.x**2 + msg.acceleration.linear.y**2) 
            self.current_throttle = msg.control.throttle
            self.current_brake = msg.control.brake
            self.current_steering = msg.control.steer
            self.current_ego_state_info = np.array([self.current_speed, self.current_accel, self.current_throttle,
                                                    self.current_brake, self.current_steering])
            self.ego_state_initialized = True

    def frenet_state_callback(self, msg):
        with self.data_lock:
            self.prev_s = self.current_s
            self.current_s = msg.s
            self.current_d = msg.d
            self.current_alpha = msg.yaw_s
            self.current_frenet_pose = np.array([self.current_s, self.current_d, self.current_alpha])
            self.frenet_pose_initialized = True

    def mpc_control_cmd_callback(self, msg):
        with self.data_lock:
            self.prev_mpc_control = self.mpc_control
            steer = msg.steer
            steer = steer/self.steer_max
            throttle = msg.throttle
            brake = msg.brake
            if brake > 0:
                throttle = -brake
            self.mpc_control = np.array([steer, throttle])
            self.mpc_control_initialized = True

    def selected_obstacles_callback(self, msg):
        with self.data_lock:
            self.selected_obstacles = np.ones((self.num_of_obs, 2)) * -1000
            for i, marker in enumerate(msg.markers):
                obstacle_frene_pose = self._get_frenet_pose(marker.pose)
                # print("Obstacle frenet pose: ", obstacle_frene_pose)
                self.selected_obstacles[i] = [obstacle_frene_pose.s, obstacle_frene_pose.d]
            self.selected_obstacles_initialized = True

    def predicted_path_callback(self, path_msg):
        with self.data_lock:
            predicted_path_cartesian = []
            predicted_path_frenet = []
            for pose in path_msg.poses:
                predicted_path_cartesian.append([pose.pose.position.x, pose.pose.position.y])
                frenet_pose = self._get_frenet_pose(pose.pose)
                predicted_path_frenet.append([frenet_pose.s, frenet_pose.d])

            # self.predicted_path_cartesian = np.array(predicted_path_cartesian)
            self.predicted_path_frenet = np.array(predicted_path_frenet)
            self.predicted_path_initialized = True

    def reference_path_callback(self, path_msg):
        with self.data_lock:
            rospy.loginfo("Initializing reference path.")
            path_msg.poses = path_msg.poses[::5]
            self.x_ref_spline, self.y_ref_spline, self.path_length, dense_s, _, kappa = parseReference(path_msg)
            if self.x_ref_spline is None or self.y_ref_spline is None or self.path_length is None or dense_s is None or kappa is None:
                rospy.loginfo("Failed to parse the reference path.")
                self.reset()
                return
            print("Path length: ", self.path_length, "Dense_s: ", dense_s[-1])
            self.kappa_spline = make_interp_spline(dense_s, kappa, k=3)
            rospy.loginfo("Reference path initialized.")
            self.ref_path_initialized = True

    def collision_callback(self, msg):
        with self.data_lock:
            self.collision = True

    def lane_invasion_callback(self, msg):
        with self.data_lock:
            if 10 in msg.crossed_lane_markings:
                self.lane_invasion = True

    def _calculate_reward(self, observation, action):
        print("action: ", action)
        if self.collision:
            return -100
        # s = observation[2]
        # d = observation[3]
        s = self.current_s
        d = self.current_d
        reward = 0
        if abs(action[0]) > 1:
            reward -= 10
            print("throttle penalty")
        if abs(action[1]) > 1:
            reward -= 10
            print("steer penalty")
        print("Current s: ", self.current_s)
        print("prev s: ", self.prev_s)
        print("Current d: ", d)
        print("Current speed: ", self.current_speed)
        progress = self.current_s - self.prev_s
        reward += progress * 10

        # reward for staying in the lane
        if not (d < 3.5 and d > -0.5):
            reward -= 5

        for obs in self.selected_obstacles:
            if self.distance_to_obs(obs) < DIST2OBSTACLE:
                reward -= 5

        if self.current_speed < 1:
            reward -= 10

        if s >= self.path_length - 10:
            reward += 1000
        
        if self.current_speed > 5:
            reward += 0.001* self.current_speed**2
        reward -= self.current_step * 0.005

        for obs in self.selected_obstacles:
            if self.distance_to_obs(obs) < 1:
                self.inside_obs = True
                reward = 0
        return reward

    def distance_to_obs(self, obs):
        return np.sqrt((self.current_frenet_pose[0] - obs[0])**2 + ((self.current_frenet_pose[1] - obs[1])*2)**2)

    def print_initialized(self):
        rospy.loginfo(
            "Ref path: {}, Ego state: {}, Frenet pose: {}, MPC control: {}"
            .format(self.ref_path_initialized, self.ego_state_initialized, self.frenet_pose_initialized, self.mpc_control_initialized)
        )
        rospy.loginfo(
            "Selected obstacles: {}, Predicted path: {}, Collision: {}, Lane Invasion: {}"
            .format(self.selected_obstacles.flatten(), self.predicted_path_initialized, self.collision, self.lane_invasion)
        )

    def _get_obs(self):
        self.print_initialized()
        start_wait_time = time.time()
        timeout_duration = 5.0
        while not rospy.is_shutdown() and not shutdown_requested and (not self.ref_path_initialized or not self.ego_state_initialized or not self.frenet_pose_initialized or \
                not self.selected_obstacles_initialized or not self.predicted_path_initialized or not self.acados_init):
            
            if time.time() - start_wait_time > timeout_duration:
                rospy.logwarn("TIMEOUT: Observations not received within 5s. Forcing break to avoid hang.")
                emergency_stop_signal = Int16(data=0)
                self.action_stop_publisher.publish(emergency_stop_signal)
                break

            if not self.ref_path_initialized:
                rospy.loginfo("Reference path not initialized.")
            if not self.ego_state_initialized:
                rospy.loginfo("Ego state not initialized.")
            if not self.frenet_pose_initialized:
                rospy.loginfo("Frenet pose not initialized.")
            if not self.selected_obstacles_initialized:
                rospy.loginfo("Selected obstacles not initialized.")
            if not self.predicted_path_initialized:
                rospy.loginfo("Predicted path not initialized.")
            if not self.acados_init:
                rospy.loginfo("Acados not initialized.")
            rospy.loginfo("Waiting for all the topics to get data.")
            rospy.sleep(0.001)

        if self.current_s + self.lookahead_distance > self.path_length:
            self.lookahead_distance = self.path_length - self.current_s

        s_list = np.linspace(self.current_s, self.current_s + self.lookahead_distance, 20)
        x_ref_points = self.x_ref_spline(s_list)
        y_ref_points = self.y_ref_spline(s_list)
        kappa_points = self.kappa_spline(s_list)

        observation = np.concatenate([
            np.array([self.path_length - self.current_frenet_pose[0]]),  # Convert scalar to 1D array
            self.current_frenet_pose,
            self.mpc_control,
            self.prev_mpc_control,
            self.selected_obstacles.flatten(),
            self.current_ego_state_info,
            # reference_sampled_points.flatten(),
            kappa_points,
            [self.current_step],
            self.predicted_path_frenet.flatten()
        ], axis=0)

        self.current_observation = observation
        if observation.shape[0] != self.state_dim:
            rospy.loginfo("Observation shape is incorrect.")
        return observation

    def _reset_obs(self):
        self.collision = False
        self.lane_invasion = False
        self.mpc_control_initialized = False
        self.selected_obstacles_initialized = False
        self.predicted_path_initialized = False
        self.ego_state_initialized = False
        self.frenet_pose_initialized = False
        self.inside_obs = False
        rospy.loginfo("Resetting the observation.")

    def step(self, residual):
        self.current_step += 1
        rospy.sleep(0.04)
        print("step: ", self.current_step)
        # print("Residual: ", residual)
        residual[0] = 0.1 * residual[0]
        residual[1] = 0.1 * residual[1]
        # residual = [0,0]
        print("Stepping with residual: ", residual)
        residual_msg = Float32MultiArray(data=[residual[0], residual[1]])
        self.action_publisher.publish(residual_msg)

        throttle = self.current_observation[4] + residual[0]
        steer = self.current_observation[5] + residual[1]

        rospy.loginfo("Stepped with throttle: {}, steer: {}".format(throttle, steer))
        observation = self._get_obs()
        reward = self._calculate_reward(observation, [throttle, steer])
        print("Reward: ", reward)
        done, info = self.check_done()
        print("Done: ", done)
        print("Info: ", info)

        return observation, reward, done, False, info

    def check_done(self):
        if self.collision:
            self.current_step = 0
            time_diff = time.time() - self.time
            print("Time taken: ", time_diff)
            info = {"lap_time": time_diff, "done": "collision"}
            # info = {"done": "collision"}
            return True, info
        if self.current_s >= (self.path_length - 10):
            self.current_step = 0
            time_diff = time.time() - self.time
            print("Time taken: ", time_diff)
            info = {"lap_time": time_diff, "done": "path end"}
            # info = {"done": "path end"}
            return True, info
        if self.current_speed < 1:
            self.speed_count += 1
            if self.speed_count > 500:
                self.current_step = 0
                time_diff = time.time() - self.time
                print("Time taken: ", time_diff)
                info = {"lap_time": time_diff, "done": "speed low"}
                # info = {"done": "speed low"}
                return True, info
        if self.inside_obs:
            self.current_step = 0
            info = {"done": "inside obs"}
            return True, info
        
        info = {"done": "not done"}
        return False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.speed_count = 0
        self.time = time.time()
        self.reset_vehicle()
        self._reset_obs()
        observation = self._get_obs()
        return observation, {}

    def render(self, mode='human'):
        pass

    def close(self):
        rospy.loginfo("Shutting down mpc gym environment.")
        with open(file_path, 'a') as file:
                file.write("close\n")
        self.emergency_stop()
        rospy.signal_shutdown("Closing the environment")
        if hasattr(self, 'client'):
            self.client = None
        if hasattr(self, 'world'):
            self.world = None

class LapTimeEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(LapTimeEvalCallback, self).__init__(*args, **kwargs)
        self.lap_times = []

    def _on_step(self) -> bool:
        result = super(LapTimeEvalCallback, self)._on_step()

        if self.locals['dones'][0]:
            if self.locals['infos'][0]['done'] == 'path end':
                lap_time = self.locals['infos'][0]['lap_time']
                self.lap_times.append(lap_time)
                #wandb.log({"eval/lap_time": lap_time})
        return result


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_reward = 0.0
        self.episode_length = 0

    def _on_step(self) -> bool:
        # Update reward and length for the current episode
        self.episode_reward += self.locals['rewards'][0]
        self.episode_length += 1

        # Check if the episode is done
        if self.locals['dones'][0]:
            # Check if 'done' in infos is 'path_end'
            print("LOCALS: ", self.locals['infos'][0])
            self.episode_rewards.append(self.episode_reward)
            if self.locals['infos'][0]['done'] == 'path end':
                self.episode_lengths.append(self.episode_length)
                print("INFOs: ", self.locals['infos'][0])
    
            else:
                self.episode_lengths.append(0)                
                
            # Reset the reward and length for the next episode
            self.episode_reward = 0.0
            self.episode_length = 0
        
        return True

register(
    id='mpc-gym-v0',
    entry_point='__main__:mpcGym',
    max_episode_steps=1000,
)

def print_wrappers(env):
    if hasattr(env, 'env'):
        print(type(env))
        print_wrappers(env.env)
    else:
        print(type(env))

def evaluate_best_model(model_path, num_episodes=5, completion_threshold=97):
    """
    Load and evaluate the best model with comprehensive metrics
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to evaluate
        completion_threshold: Path completion ratio to consider as success (default 0.97 = 97%)
    """
    rospy.init_node('mpc_gym_eval_node')
    env = gym.make('mpc-gym-v0')
    
    print(f"Loading model from: {model_path}")
    print(f"Completion threshold: {completion_threshold*100:.0f}% of path")
    try:
        model = SAC.load(model_path)
        with open(file_path, "a") as file:
            file.write("Load model successfully\n")
        print("Model loaded successfully!")
    except Exception as e:
        with open(file_path, "a") as file:
            file.write(f"Failed to load model: {e}\n")
        print(f"Failed to load model: {e}")
        env.close()
        return
    
    # Initialize metrics tracking
    metrics = {
        'total_episodes': num_episodes,
        'successful_completions': 0,
        'collisions': 0,
        'lane_invasions': 0,
        'speed_timeouts': 0,
        'inside_obstacle': 0,
        'lap_times': [],
        'episode_rewards': [],
        'episode_steps': [],
        'completion_rates': [],
        'average_speeds': [],
        'lane_deviations': [],  # Track average |d| per episode
        'termination_reasons': []  # Track detailed termination info
    }
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*60}\n")
        
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        # Episode-specific tracking
        episode_speeds = []
        episode_lane_devs = []
        
        while not done and not rospy.is_shutdown():
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step += 1
            
            # Track metrics during episode
            episode_speeds.append(env.unwrapped.current_speed)
            episode_lane_devs.append(abs(env.unwrapped.current_d))
            
            # Print progress every 50 steps
            if step % 50 == 0:
                completion_pct = (env.unwrapped.current_s / env.unwrapped.path_length * 100) if env.unwrapped.path_length > 0 else 0
                print(f"Step {step}: Reward={reward:.2f}, "
                    f"Cumulative={episode_reward:.2f}, "
                    f"Speed={env.unwrapped.current_speed:.2f} m/s, "
                    f"Lane Dev={abs(env.unwrapped.current_d):.2f}m, "
                    f"Progress={completion_pct:.1f}%")
                if completion_pct >= completion_threshold:
                    with open(file_path, "a") as file:
                        file.write(f"Finish with Completion pct: {completion_pct}\n")
                    break
        
        # Episode finished - collect metrics
        completion_reason = info.get('done', 'unknown')
        
        # Calculate completion rate (progress along path)
        completion_rate = (env.unwrapped.current_s / env.unwrapped.path_length) if env.unwrapped.path_length > 0 else 0
        completion_pct = completion_rate * 100
        metrics['completion_rates'].append(completion_pct)
        
        # Consider episode successful if completed >= threshold of path
        # This handles both 'path end' termination and near-complete scenarios
        is_success = completion_rate >= completion_threshold/100
        
        # Update counters based on completion reason
        if completion_reason == 'collision':
            metrics['collisions'] += 1
            metrics['termination_reasons'].append(f"Collision at {completion_pct:.1f}%")
        elif completion_reason == 'path end' or is_success:
            metrics['successful_completions'] += 1
            if 'lap_time' in info:
                metrics['lap_times'].append(info['lap_time'])
            metrics['termination_reasons'].append(f"Success at {completion_pct:.1f}%")
        elif completion_reason == 'speed low':
            if is_success:
                metrics['successful_completions'] += 1
                if 'lap_time' in info:
                    metrics['lap_times'].append(info['lap_time'])
                metrics['termination_reasons'].append(f"Success (stalled at {completion_pct:.1f}%)")
            else:
                metrics['speed_timeouts'] += 1
                metrics['termination_reasons'].append(f"Speed timeout at {completion_pct:.1f}%")
        elif completion_reason == 'inside obs':
            metrics['inside_obstacle'] += 1
            metrics['termination_reasons'].append(f"Inside obstacle at {completion_pct:.1f}%")
        else:
            metrics['termination_reasons'].append(f"{completion_reason} at {completion_pct:.1f}%")
        
        # Track lane invasion (check if it happened during episode)
        if env.unwrapped.lane_invasion:
            metrics['lane_invasions'] += 1
        
        # Calculate episode statistics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_steps'].append(step)
        
        if len(episode_speeds) > 0:
            metrics['average_speeds'].append(np.mean(episode_speeds))
        else:
            metrics['average_speeds'].append(0.0)
        
        if len(episode_lane_devs) > 0:
            metrics['lane_deviations'].append(np.mean(episode_lane_devs))
        else:
            metrics['lane_deviations'].append(0.0)
        
        # Episode summary
        print(f"\n--- Episode {episode + 1} Summary ---")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Steps: {step}")
        print(f"Completion Reason: {completion_reason}")
        print(f"Path Completion: {completion_pct:.1f}%")
        print(f"Success: {'YES' if is_success else 'NO'}")
        print(f"Avg Speed: {metrics['average_speeds'][-1]:.2f} m/s")
        print(f"Avg Lane Deviation: {metrics['lane_deviations'][-1]:.2f} m")
        if 'lap_time' in info:
            print(f"Lap Time: {info['lap_time']:.2f}s")
        print()
    
    env.close()
    
    # Calculate and display final statistics
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}\n")
    
    # Success metrics
    success_rate = (metrics['successful_completions'] / num_episodes) * 100
    collision_rate = (metrics['collisions'] / num_episodes) * 100
    lane_invasion_rate = (metrics['lane_invasions'] / num_episodes) * 100
    
    print(f"Total Episodes: {num_episodes}")
    print(f"Completion Threshold: {completion_threshold*100:.0f}%")
    print(f"\n--- Completion Statistics ---")
    print(f"Successful Completions: {metrics['successful_completions']} ({success_rate:.1f}%)")
    print(f"Collisions: {metrics['collisions']} ({collision_rate:.1f}%)")
    print(f"Lane Invasions: {metrics['lane_invasions']} ({lane_invasion_rate:.1f}%)")
    print(f"Speed Timeouts: {metrics['speed_timeouts']}")
    print(f"Inside Obstacle: {metrics['inside_obstacle']}")
    
    print(f"\n--- Termination Details ---")
    for i, reason in enumerate(metrics['termination_reasons'], 1):
        print(f"  Episode {i}: {reason}")
    
    print(f"\n--- Performance Metrics ---")
    if len(metrics['lap_times']) > 0:
        print(f"Average Lap Time: {np.mean(metrics['lap_times']):.2f}s (±{np.std(metrics['lap_times']):.2f}s)")
        print(f"Best Lap Time: {np.min(metrics['lap_times']):.2f}s")
    else:
        print(f"Average Lap Time: N/A (no successful completions)")
    
    print(f"Average Reward: {np.mean(metrics['episode_rewards']):.2f} (±{np.std(metrics['episode_rewards']):.2f})")
    print(f"Average Steps: {np.mean(metrics['episode_steps']):.1f} (±{np.std(metrics['episode_steps']):.1f})")
    print(f"Average Speed: {np.mean(metrics['average_speeds']):.2f} m/s (±{np.std(metrics['average_speeds']):.2f})")
    print(f"Average Lane Deviation: {np.mean(metrics['lane_deviations']):.2f}m (±{np.std(metrics['lane_deviations']):.2f})")
    print(f"Average Completion Rate: {np.mean(metrics['completion_rates']):.1f}%")
    
    # Calculate driving score (0-100)
    # Weighted combination of multiple factors
    driving_score = (
        success_rate * 0.4 +  # 40% weight on success
        (100 - collision_rate) * 0.3 +  # 30% weight on safety
        (100 - lane_invasion_rate) * 0.15 +  # 15% weight on lane keeping
        min(np.mean(metrics['completion_rates']), 100) * 0.15  # 15% weight on progress
    )
    
    print(f"\n--- Overall Driving Score ---")
    print(f"Driving Score: {driving_score:.1f}/100")
    
    # Save metrics to file
    metrics_file = model_path.replace('.zip', '_evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Evaluation Metrics for {model_path}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total Episodes: {num_episodes}\n")
        f.write(f"Completion Threshold: {completion_threshold:.0f}%\n\n")
        f.write(f"Completion Statistics:\n")
        f.write(f"  Successful Completions: {metrics['successful_completions']} ({success_rate:.1f}%)\n")
        f.write(f"  Collisions: {metrics['collisions']} ({collision_rate:.1f}%)\n")
        f.write(f"  Lane Invasions: {metrics['lane_invasions']} ({lane_invasion_rate:.1f}%)\n")
        f.write(f"  Speed Timeouts: {metrics['speed_timeouts']}\n")
        f.write(f"  Inside Obstacle: {metrics['inside_obstacle']}\n\n")
        f.write(f"Termination Details:\n")
        for i, reason in enumerate(metrics['termination_reasons'], 1):
            f.write(f"  Episode {i}: {reason}\n")
        f.write(f"\nPerformance Metrics:\n")
        if len(metrics['lap_times']) > 0:
            f.write(f"  Average Lap Time: {np.mean(metrics['lap_times']):.2f}s (±{np.std(metrics['lap_times']):.2f}s)\n")
            f.write(f"  Best Lap Time: {np.min(metrics['lap_times']):.2f}s\n")
        f.write(f"  Average Reward: {np.mean(metrics['episode_rewards']):.2f} (±{np.std(metrics['episode_rewards']):.2f})\n")
        f.write(f"  Average Steps: {np.mean(metrics['episode_steps']):.1f} (±{np.std(metrics['episode_steps']):.1f})\n")
        f.write(f"  Average Speed: {np.mean(metrics['average_speeds']):.2f} m/s\n")
        f.write(f"  Average Lane Deviation: {np.mean(metrics['lane_deviations']):.2f}m\n")
        f.write(f"  Average Completion Rate: {np.mean(metrics['completion_rates']):.1f}%\n")
        f.write(f"Overall Driving Score: {driving_score:.1f}/100\n")
    
    print(f"\nMetrics saved to: {metrics_file}")
    print("Evaluation complete!")
    with open(file_path, "a") as file:
        file.write("Finish Evaluation\n")
    
    return metrics

if __name__ == "__main__":
    evaluate_best_model("/home/ave/Desktop/carla_mpc_residual_learning/src/mpc_controller/envs/sac_mpc.zip", 5)