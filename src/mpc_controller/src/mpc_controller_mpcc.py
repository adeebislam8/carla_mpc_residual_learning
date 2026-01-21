#!/usr/bin/env python
#
# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
Todo:
    - Extract vehicle state information and print (done)
    - Convert the global path into frenet frame 
    - Convert the pose data into frenet frame
    - Pass vehicle state information to the MPC controller
    - Implement the border_cb function
    - Implement the border publishing node

"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import carla

import carla_common.transforms as trans
import collections
import math
import threading
from casadi import *
from scipy.integrate import solve_ivp

import numpy as np
import rospy
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# from carla_ad_agent.vehicle_mpc_controller import VehicleMPCController
from carla_ad_agent.misc import distance_vehicle

from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus  # pylint: disable=import-error
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float64, Int16, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from global_planner.srv import Frenet2WorldService, World2FrenetService
from global_planner.msg import FrenetPose, WorldPose
from acados_mpc.acados_settings_mpcc import acados_settings
from utils.convert_traj_track import parseReference
from scipy.interpolate import make_interp_spline

debug_filepath = "/home/ave/Desktop/carla_mpc_residual_learning/debug.txt"
with open(debug_filepath, "w") as file:
    file.write('')

class Obstacle:
    def __init__(self):
        self.id = -1 # actor id
        self.frenet_s = 0.0 # frenet s coordinate
        self.frenet_d = 0.0
        self.scale_x = 0.0 # bbox length in x direction
        self.scale_y = 0.0 # bbox length in y direction
        self.scale_z = 0.0 # bbox length in z direction
        self.ros_transform = None # transform of the obstacle in ROS coordinate
        self.carla_transform = None # transform of the obstacle in Carla world coordinate
        self.bbox = None # Bounding box w.r.t ego vehicle's local frame

class LocalPlannerMPC(CompatibleNode):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is
    generated on-the-fly. The low-level motion of the vehicle is computed by using two PID
    controllers, one is used for the lateral control and the other for the longitudinal
    control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self):
        super(LocalPlannerMPC, self).__init__("local_planner_mpc")

        role_name = self.get_param("role_name", "ego_vehicle")
        self.control_time_step = self.get_param("control_time_step", 0.05)

        # Fetch the Q and R matrices from parameters
        self.Q_matrix = self.get_param('~Q_matrix', [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # Default Q matrix if not set
        self.R_matrix = self.get_param('~R_matrix', [[1.0, 0.0], [0.0, 1.0]])  # Default R matrix if not set
        self.Tf = 1.0
        self.N = 10
        self.t_delay = 0.03
        self.obs_range = 300
        self.spline_degree = 3
        self.s_list = np.ones(self.N+1)
        self.time = rospy.get_time()
        # Log the matrices for verification
        # self.loginfo("Q matrix: %s", str(self.Q_matrix))
        # self.loginfo("R matrix: %s", str(self.R_matrix))
        self.data_lock = threading.Lock()
        self.initailize = False
        self.derD = 0
        self.derDelta = 0
        self.derTheta = 0
        self._current_pose = None
        self._current_speed = None
        self._current_velocity = None
        self._target_speed = 150.0       # kph
        self._current_accel = None
        self._current_throttle = None
        self._current_brake = None
        self._current_steering = None

        self._buffer_size = 5
        self._waypoints_queue = collections.deque(maxlen=20000)
        self._waypoint_buffer = collections.deque(maxlen=self._buffer_size)
        self._global_path_length = None
        self.objects_frenet_points = np.ones((6, 2), dtype=np.float32) * -100
        self.s = 0
        self.n = 0

        self.throttle_residual = 0
        self.steering_residual = 0
        self.emergency_stop_alert = False

        self.acados_solver = None
        self.path_initialized = False
        self._path_msg = None
        self.time = rospy.get_time()
        # subscribers
        self._odometry_subscriber = self.new_subscription(
            Odometry,
            "/carla/{}/odometry".format(role_name),
            self.odometry_cb,
            qos_profile=10)
        self._ego_status_subscriber = self.new_subscription(
            CarlaEgoVehicleStatus,
            "/carla/{}/vehicle_status".format(role_name),
            self.ego_status_cb,
            qos_profile=10)
        self._path_subscriber = self.new_subscription(
            Path,
            # "/carla/{}/waypoints".format(role_name),
            "/global_planner/{}/waypoints".format(role_name),
            self.path_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        
        self._obstacle_markers_subscriber = self.new_subscription(
            MarkerArray,
            "/carla/markers".format(role_name),
            self.obstacle_markers_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        
        self._mpc_rl_reinit_subscriber = self.new_subscription(
            Int16,
            '/mpc_rl/acados_reinit',
            self.mpc_rl_reinit_cb,
            qos_profile=10)

        self._world2frenet_service = self.new_client(
            World2FrenetService,
            '/world2frenet')
        self._frenet2world_service = self.new_client(
            Frenet2WorldService,
            '/frenet2world')      

        ## Todo: Implement later ##
        self._border_subscriber = self.new_subscription(
            Path,
            "/carla/{}/border_waypoints".format(role_name),
            self.border_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        ## Todo: Implement later ##


        self._target_speed_subscriber = self.new_subscription(
            Float64,
            "/carla/{}/speed_command".format(role_name),
            self.target_speed_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self._mpc_rl_residual_subscriber = self.new_subscription(
            Float32MultiArray,
            '/mpc_rl/residual',
            self.mpc_rl_residual_cb,
            qos_profile=10)
        
        self._mpc_rl_emergency_stop_subscriber = self.new_subscription(
            Int16,
            '/mpc_rl/emergency_stop',
            self.mpc_rl_emergency_stop_cb,
            qos_profile=10)


        # publishers
        self._selected_obstacle_publisher = self.new_publisher(
            MarkerArray,
            "/mpc_controller/{}/selected_obstacles".format(role_name),
            qos_profile=10)
        
        self._mpc_rl_acados_init_publisher = self.new_publisher(
            Int16,
            '/mpc_rl/acados_init',
            qos_profile=10)
        
        self._target_pose_publisher = self.new_publisher(
            Marker,
            "/mpc_controller/{}/next_target".format(role_name),
            qos_profile=10)
        self._control_cmd_publisher = self.new_publisher(
            CarlaEgoVehicleControl,
            "/carla/{}/vehicle_control_cmd".format(role_name),
            qos_profile=10)
        
        self._reference_path_publisher = self.new_publisher(
            Path,
            "/mpc_controller/{}/reference_path".format(role_name),
            qos_profile=10)

        self._predicted_path_publisher = self.new_publisher(
            Path,
            "/mpc_controller/{}/predicted_path".format(role_name),
            qos_profile=10)
        # initializing controller
        # self._vehicle_controller = VehicleMPCController(
        #     self)

    def mpc_rl_reinit_cb(self, msg):
        """Handle request to reinitialize acados solver"""
        if msg.data == 1:
            with self.data_lock:
                self.loginfo("Received acados reinitialization request")
                self.acados_solver = None
                self.path_initialized = False
                self.loginfo("Acados solver set to None for reinitialization")

    def mpc_rl_emergency_stop_cb(self, msg):
        if msg.data == 1:
            self.emergency_stop_alert = True
        if msg.data == 0:
            self.emergency_stop_alert = False

    def mpc_rl_residual_cb(self, msg):
        self.throttle_residual = msg.data[0]
        self.steering_residual = msg.data[1]

    def obstacle_markers_cb(self, marker_array):
        selected_obstacles = []

        for marker in marker_array.markers:
            if marker.color.r == 255.0:
                frenet_pose = self._get_frenet_pose(marker.pose)
                distance = frenet_pose.s - self.s
                
                if distance < self.obs_range and distance > -5:
                    # STRICTER FILTERING: Only consider obstacles within reasonable lane width
                    # Typical lane width is 3.5m, so obstacles at edges (±1.75m) are likely road furniture
                    if abs(frenet_pose.d) < 3.0:  # Changed from 4.5 to 3.0
                        
                        # ADDITIONAL CHECK: Filter out obstacles that are exactly on lane boundaries
                        # (these distances like 0.6869m, 0.9331m are often curbs/lane markers)
                        relative_d = abs(frenet_pose.d - self.n)
                        
                        # Skip if obstacle is likely a road boundary marker
                        # (very consistent distance that suggests static infrastructure)
                        if relative_d > 0.3:  # Only consider obstacles offset from current position
                            obs_marker = marker
                            obs_marker.color.r = 0.0
                            obs_marker.color.g = 255.0
                            obs_marker.color.b = 0.0
                            obs_marker.color.a = 1.0
                            obs_marker.scale.x = marker.scale.x 
                            obs_marker.scale.y = marker.scale.y
                            obs_marker.scale.z = marker.scale.z
                            obs_marker.lifetime = rospy.Duration(0.1)
                            selected_obstacles.append([obs_marker, frenet_pose.s, frenet_pose.d])

        sorted_obstacles = MarkerArray()
        sorted_list = sorted(selected_obstacles, key=lambda x: x[1])
        for i, obs in enumerate(sorted_list):
            obs_marker = obs[0]
            obs_marker.id = i
            sorted_obstacles.markers.append(obs_marker)

        # Only publish up to 3 obstacles
        self._selected_obstacle_publisher.publish(sorted_obstacles.markers[:3])
        
        # CRITICAL: Reset ALL obstacle points first
        self.objects_frenet_points = np.ones((6, 2), dtype=np.float32) * -100
        
        # Then fill in valid obstacles
        for i, obs in enumerate(sorted_obstacles.markers[:3]):
            frenet_pose = self._get_frenet_pose(obs.pose)
            self.objects_frenet_points[i] = np.array([frenet_pose.s, frenet_pose.d], dtype=np.float32)
    def odometry_cb(self, odometry_msg):
        # self.loginfo("Received odometry message")
        with self.data_lock:
            self._current_pose = odometry_msg.pose.pose
            # self.loginfo("odom callback: {}".format(self._current_pose))
            self._current_speed = math.sqrt(odometry_msg.twist.twist.linear.x ** 2 +
                                            odometry_msg.twist.twist.linear.y ** 2 +
                                            odometry_msg.twist.twist.linear.z ** 2) * 3.6 # m/s to km/h
            self._draw_reference_point(self._current_pose)

    def _draw_reference_point(self, pose):
        ref_path = Path()
        ref_path.header.frame_id = "map"
        ref_path.header.stamp = roscomp.ros_timestamp(self.get_time(), from_sec=True)

        # print("Current pose: ", pose)
        frenet_pose = self._get_frenet_pose(pose)
        # self.loginfo("Frenet pose: {}".format(frenet_pose))
        s, d = frenet_pose.s, frenet_pose.d
        # 10 waypoints 10m ahead of the vehicle: 
            # todo: make sure the waypoints are within the length of the path
        for i in range(10):
            request = FrenetPose()
            request.s = s + i * 0.5
            request.d = 0
            request.yaw_s = 0
            response = self._frenet2world_service(request)
            pose_msg = self._world2pose(response)
            pose_stamped = PoseStamped()
            pose_stamped.pose = pose_msg
            ref_path.poses.append(pose_stamped)

        self._reference_path_publisher.publish(ref_path)
    # converts WorldPose to geometry_msgs/Pose
    def _world2pose(self, world_pose):
        world_pose = world_pose.world_pose
        pose = Pose()
        pose.position.x = world_pose.x
        pose.position.y = world_pose.y
        pose.position.z = 0
        yaw = world_pose.yaw
        # self.loginfo("Test _worl2pose Yaw: {}".format(yaw))
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_euler(0, 0, yaw)
        # pose.orientation.x = 0
        # pose.orientation.y = 0
        # pose.orientation.z = 0
        # pose.orientation.w = 1
        return pose

    def _get_world_pose(self, frenet_pose):
        request = FrenetPose()
        request.s = frenet_pose.s
        request.d = frenet_pose.d
        response = self._frenet2world_service(request)
        return response.pose
    
    def _get_frenet_pose(self, pose):
        request = WorldPose()
        request.x = pose.position.x
        request.y = pose.position.y
        _, _, yaw = euler_from_quaternion([pose.orientation.x, pose.orientation.y, 
                                            pose.orientation.z, pose.orientation.w])
        request.yaw = yaw
        # request.v = self._current_speed
        # request.acc = self._current_accel
        # request.target_v = self._target_speed

        response = self._world2frenet_service(request)
        if response is None:
            self.loginfo("Failed to get frenet pose")
            dummy = FrenetPose()
            dummy.s = 0
            dummy.d = 0
            return dummy
        # self.loginfo("Test _get_frenet_pose: {}".format(response))
        return response.frenet_pose

    def ego_status_cb(self, ego_status_msg):
        with self.data_lock:
            self._current_accel = math.sqrt(ego_status_msg.acceleration.linear.x ** 2 +
                                            ego_status_msg.acceleration.linear.y ** 2 +
                                            ego_status_msg.acceleration.linear.z ** 2) * 3.6
            self._current_throttle = ego_status_msg.control.throttle
            self._current_brake = ego_status_msg.control.brake
            self._current_steering = ego_status_msg.control.steer
            self._current_velocity = ego_status_msg.velocity
        ## Todo: Check if 3.6 is the correct conversion factor ##         



    def target_speed_cb(self, target_speed_msg):
        with self.data_lock:
            self._target_speed = target_speed_msg.data

    def path_cb(self, path_msg):
        with self.data_lock:
            self._waypoint_buffer.clear()
            self._waypoints_queue.clear()
            self._waypoints_queue.extend([pose.pose for pose in path_msg.poses])
            self._path_msg = path_msg
            # self.loginfo("Received path message of length: {}".format(len(path_msg)))
            # self.loginfo("Current waypoints queue length: {}".format(len(self._waypoints_queue)))
            # self.loginfo("Current waypoints buffer length: {}".format(len(self._waypoint_buffer)))
            # self.loginfo("First waypoint in queue: {}".format(self._waypoints_queue[0]))

            # sparsify path_msg
            path_msg.poses = path_msg.poses[::5]
            _, _, _, dense_s, _, kappa = parseReference(path_msg)
            kappa_spline = make_interp_spline(dense_s, kappa, k=3)
            self.spline_coeffs = kappa_spline.c
            self.spline_knots = kappa_spline.t
            #self.loginfo("Spline coefficients: {}".format(self.spline_coeffs))
            #self.loginfo("Spline knots: {}".format(self.spline_knots))
            self.path_initialized = True
            self._global_path_length = dense_s[-1]

            self.acados_solver = None
            self.loginfo("Acados Reset")

    ## Todo: Write border publishing node and implement this function ##
    def border_cb(self, path_msg):
        pass
        with self.data_lock:
            self._waypoint_buffer.clear()
            self._waypoints_queue.clear()
            self._waypoints_queue.extend([pose.pose for pose in path_msg.poses])
    ##  -------------------------------------------------------------- ##

    def pose_to_marker_msg(self, pose):
        marker_msg = Marker()
        marker_msg.type = 0
        marker_msg.header.frame_id = "map"
        marker_msg.pose = pose
        marker_msg.scale.x = 1.0
        marker_msg.scale.y = 0.2
        marker_msg.scale.z = 0.2
        marker_msg.color.r = 255.0
        marker_msg.color.a = 1.0
        return marker_msg

    def _dynamics_of_car(self, t, x0) -> list:
        """
        Used for forward propagation. This function takes the dynamics from the acados model.
        """
        ## Race car parameters
        m = 2065.03
        # C1 =  -0.00021201
        # C1 =  0.00021201
        # C2 =  -0.17345602

        lf = 1.169
        lr = 1.801
        C1 = lr / (lr + lf)
        C2 = 1 / (lr + lf) * 0.5

        Cm1 = 9.36424211e+03 
        Cm2 = 4.08690122e+01  
        Cr2 = 2.04799356e+00
        Cr0 = 5.84856121e+02
        Cr3 = 1.13995833e+01
        # print("dyn x0: ", x0)
        s, n, alpha, v, D, delta, theta, derD, derDelta, derTheta = x0
        

        kapparef_s = self.model.kapparef_s

        Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * tanh(Cr3 * v)
        # Fxd = (Cm1 - Cm2*v)*D - Cr2*v**2 - Cr0

        a_long = Fxd / m
        delta = -delta
        sdot = (v * cos(alpha + C1 * delta)) / (1 - kapparef_s(s) * n)
        ndot = v * sin(alpha + C1 * delta)
        alphadot = v * C2 * delta - kapparef_s(s) * sdot                   # alphadot
        # yaw_rate - kapparef_s(s) * sdot,                     # alphadot
        vdot = a_long * cos(C1 * delta)                                  # vdot
        # a_long * cos(C1 * delta) - vglobaldot_s(s) * sdot,
        Ddot = derD
        deltadot = derDelta
        thetadot = derTheta
        a_lat = C2 * v * v * delta + a_long * sin(C1 * delta)

        xdot = [float(sdot), ndot, float(alphadot), vdot, Ddot, deltadot, thetadot, Ddot, deltadot, thetadot]

        # print("xdot: ", xdot)
        return xdot
    
    def dynamics(self, x0):
        """
        Used for forward propagation. This function takes the dynamics from the acados model.
        """
        ## Race car parameters
        m = 2065.03
        # C1 =  -0.00021201
        # C1 =  0.00021201
        # C2 =  -0.17345602

        lf = 1.169
        lr = 1.801
        C1 = lr / (lr + lf)
        C2 = 1 / (lr + lf) * 0.5

        Cm1 = 9.36424211e+03 
        Cm2 = 4.08690122e+01  
        Cr2 = 2.04799356e+00
        Cr0 = 5.84856121e+02
        Cr3 = 1.13995833e+01
        # print("dyn x0: ", x0)
        s, n, alpha, v, D, delta, theta, derD, derDelta, derTheta = x0
        

        kapparef_s = self.model.kapparef_s

        Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * tanh(Cr3 * v)
        # Fxd = (Cm1 - Cm2*v)*D - Cr2*v**2 - Cr0

        a_long = Fxd / m
        delta = -delta
        sdot = (v * cos(alpha + C1 * delta)) / (1 - kapparef_s(s) * n)
        ndot = v * sin(alpha + C1 * delta)
        alphadot = v * C2 * delta - kapparef_s(s) * sdot                   # alphadot
        # yaw_rate - kapparef_s(s) * sdot,                     # alphadot
        vdot = a_long * cos(C1 * delta)                                  # vdot
        # a_long * cos(C1 * delta) - vglobaldot_s(s) * sdot,
        Ddot = derD
        deltadot = derDelta
        thetadot = derTheta
        a_lat = C2 * v * v * delta + a_long * sin(C1 * delta)

        xdot = [float(sdot), ndot, float(alphadot), vdot, Ddot, deltadot, thetadot, Ddot, deltadot, thetadot]
        self.loginfo("a_long: {}".format(a_long))
        self.loginfo("a_lat: {}".format(a_lat))
        # print("xdot: ", xdot)
        return xdot
    

    def propagate_time_delay(self, states: np.array, inputs: np.array) -> np.array:

        # Initial condition on the ODE
        x0 = np.concatenate((states, inputs), axis=0)
        # print("combined x0: ", x0)
        solution = solve_ivp(
            self._dynamics_of_car,
            t_span=[0, self.t_delay],
            y0=x0,
            method="RK45",
            atol=1e-8,
            rtol=1e-8,
        )
        # print("solution: ", solution)
        solution = [x[-1] for x in solution.y]

        # Constraint on max. steering angle
        s, n, alpha, v, D, delta, theta = solution[:7]
        if abs(delta) > self.model.delta_max:
            delta = (
                np.sign(delta) * self.model.delta_max
            )

        # Constraint on max. thrust
        if abs(D) > self.model.throttle_max:
            D = (
                np.sign(D) * self.model.throttle_max
            )

        # Only get the state as solution of where the car will be in t_delay seconds
        return np.array(solution[:7])
    
    def diagnose_constraints(self, stage, file_handle):
        """Diagnose which constraints are violated at a given stage"""
        try:
            x = self.acados_solver.get(stage, "x")
            
            file_handle.write(f"========== CONSTRAINT DIAGNOSTICS FOR STAGE {stage} ==========\n")
            
            # State values
            if x is not None and len(x) >= 7:
                s, n, alpha, v, D, delta, theta = x
                file_handle.write(f"State x[{stage}]: s={s:.4f}, n={n:.4f}, alpha={alpha:.4f}, v={v:.4f}, D={D:.4f}, delta={delta:.4f}, theta={theta:.4f}\n")
                
                # Check state bounds
                file_handle.write("--- State Bounds ---\n")
                if self.model is not None:
                    if self.model.n_min is not None and self.model.n_max is not None:
                        file_handle.write(f"n: {n:.4f} [min={self.model.n_min:.4f}, max={self.model.n_max:.4f}] {'✗ VIOLATED' if n < self.model.n_min or n > self.model.n_max else '✓'}\n")
                    
                    if self.model.v_min is not None and self.model.v_max is not None:
                        file_handle.write(f"v: {v:.4f} [min={self.model.v_min:.4f}, max={self.model.v_max:.4f}] {'✗ VIOLATED' if v < self.model.v_min or v > self.model.v_max else '✓'}\n")
                    
                    if self.model.throttle_min is not None and self.model.throttle_max is not None:
                        file_handle.write(f"D: {D:.4f} [min={self.model.throttle_min:.4f}, max={self.model.throttle_max:.4f}] {'✗ VIOLATED' if D < self.model.throttle_min or D > self.model.throttle_max else '✓'}\n")
                    
                    if self.model.delta_min is not None and self.model.delta_max is not None:
                        file_handle.write(f"delta: {delta:.4f} [min={self.model.delta_min:.4f}, max={self.model.delta_max:.4f}] {'✗ VIOLATED' if delta < self.model.delta_min or delta > self.model.delta_max else '✓'}\n")
                
                # Check obstacle distances
                file_handle.write("--- Obstacle Information ---\n")
                file_handle.write(f"Current position: s={s:.4f}, n={n:.4f}\n")
                
                valid_obs_count = 0
                if hasattr(self, 'objects_frenet_points') and self.objects_frenet_points is not None:
                    for i, obs in enumerate(self.objects_frenet_points):
                        # CRITICAL: Only check valid obstacles
                        if obs is not None and len(obs) >= 2 and obs[0] is not None and obs[1] is not None:
                            if obs[0] > -50 and obs[1] > -50:
                                dist_s = obs[0] - s
                                dist_n = obs[1] - n
                                dist = np.sqrt(dist_s**2 + dist_n**2)
                                file_handle.write(f"Obstacle {i} [VALID]: s={obs[0]:.4f}, n={obs[1]:.4f}, dist={dist:.4f}\n")
                                valid_obs_count += 1
                            else:
                                file_handle.write(f"Obstacle {i} [INVALID]: s={obs[0]:.4f}, n={obs[1]:.4f} (skipped)\n")
                
                file_handle.write(f"Total valid obstacles: {valid_obs_count}\n")
            else:
                file_handle.write("ERROR: Could not retrieve state x\n")
            
            file_handle.write("=" * 60 + "\n")
            
        except Exception as e:
            file_handle.write(f"ERROR in diagnose_constraints: {e}\n")
            import traceback
            traceback.print_exc(file=file_handle)

    def check_path_association(self, pose):
        """Check if current pose can be properly associated with the path"""
        try:
            frenet_pose = self._get_frenet_pose(pose)
            
            # Check if the Frenet conversion seems reasonable
            if frenet_pose is None:
                return False, "Frenet pose is None"
            
            if abs(frenet_pose.d) > 50.0:  # More than 50m from path
                return False, f"Vehicle too far from path: d={frenet_pose.d:.2f}m"
            
            if frenet_pose.s < 0 or (self._global_path_length is not None and frenet_pose.s > self._global_path_length + 10):
                return False, f"s={frenet_pose.s:.2f} outside path bounds [0, {self._global_path_length}]"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Exception: {e}"

    def run_step(self):
        """
        Sets up the OCP problem in acados and solves it
         - initializes the acados 
         - print:
            - current cartesian state (pose+velocity)
            - current frenet state (s, d, yaw_s + frenet velocity, acceleration)
            - current vehicle actuators (throttle, brake, steering)

        """
        # self.loginfo("Starting time: {}".format(rospy.get_time()))
        # self.loginfo("self.time: {}".format(self.time))
        with self.data_lock:

        # debug info
            while not self.path_initialized:
                self.loginfo("Waiting for path to be initialized")
                return
            while not self._current_pose:
                self.loginfo("Waiting for odometry message")
                return
            
            # self.loginfo("Current speed: {}".format(self._current_speed))
            # self.loginfo("Current pose: {}".format(self._current_pose)) 
            # self.loginfo("Current velocity: {}".format(self._current_velocity))
            # self.loginfo("Target speed: {}".format(self._target_speed))
            # self.loginfo("Current throttle: {}".format(self._current_throttle))
            # self.loginfo("Current brake: {}".format(self._current_brake))
            # self.loginfo("Current steering: {}".format(self._current_steering))
            # self.loginfo("Current acceleration: {}".format(self._current_accel))
            self.loginfo("Frenet pose: {}".format(self._get_frenet_pose(self._current_pose)))
            # return
            # initiailize the acados problem
            if self.acados_solver is None:
                acados_init_signal = Int16()
                acados_init_signal.data = 0
                self._mpc_rl_acados_init_publisher.publish(acados_init_signal)
                self.emergency_stop()

                self.constraint, self.model, self.acados_solver = acados_settings(self.Tf, self.N, self.spline_coeffs, self.spline_knots, self._path_msg, self.spline_degree)
                self.loginfo("Initialized acados solver")

            acados_init_signal = Int16()
            acados_init_signal.data = 1
            self._mpc_rl_acados_init_publisher.publish(acados_init_signal)   
   
            frenet_pose = self._get_frenet_pose(self._current_pose)
            # self.loginfo("Frenet pose in run step: {}".format(frenet_pose))
            if self._current_brake != 0:
                D = -self._current_brake
            else:
                D = self._current_throttle

            s, n, alpha, v, D, delta = frenet_pose.s, frenet_pose.d, frenet_pose.yaw_s, self._current_speed, D, self._current_steering
            x0p = np.array([s, n, alpha, v, D, delta, s])
            u0p = np.array([self.derD, self.derDelta, self.derTheta])
            propagated_x = self.propagate_time_delay(x0p, u0p)
            self.acados_solver.set(0, "lbx", propagated_x)
            self.acados_solver.set(0, "ubx", propagated_x)
            dynamics = self.dynamics(np.concatenate((x0p, u0p), axis=0))
            # self.acados_solver.set(0, "x", propagated_x)

            # self.loginfo("Initial state: {}".format(x0p))
            # self.loginfo("Propagated state: {}".format(propagated_x))
            
            self.s = s
            self.n = n
            # print("s: ", s)
            # print("n: ", n) 
            # print("global_path_length: ", self._global_path_length)
            theta = s                 # theta is the arc length progress along centerline
            # x = [s, n, alpha, v, D, delta]
            self.acados_solver.set(0, "x", np.array([s, n, alpha, v, D, delta, theta]))
            # self.acados_solver.set(0, "lbx", np.array([s, n, alpha, v, D, delta, theta]))
            # self.acados_solver.set(0, "ubx", np.array([s, n, alpha, v, D, delta, theta]))

            # self.objects_frenet_points = np.ones((6, 2), dtype=np.float32) * -100
            # for i, object in enumerate(self._obstacles):
            #     if i >= 6:
            #         break
            #     frenet_points = [object.frenet_s, object.frenet_d]
            #     # print("frenet points: ", frenet_points)
            #     # frenet_points.append([object.frenet_s, object.frenet_d])
            #     self.objects_frenet_points[i] = np.array(frenet_points, dtype=np.float32)
            print("Objects frenet points: ", self.objects_frenet_points)

            distance2stop = 0.5 * v
            for i in range(1, self.N):
                s_target = s + self._target_speed * (self.Tf / self.N) * (i+1)
                # if s_target > self._global_path_length:
                #     s_target = self._global_path_length - distance2stop
                #     lbx = np.array([s - 2])
                #     ubx = np.array([s + 2])
                    # self.acados_solver.set(i, "lbx", lbx)
                    # self.acados_solver.set(i, "ubx", ubx)
                    # self.loginfo("s_target: {}".format(s_target))
                # print("s_target_{}: {}".format(i,s_target))

                # yref = np.array([
                #     s_target,     # s
                #     0,                                                       # n
                #     0,                                                       # alpha
                #     0,                                                       # v
                #     0,                                                       # D
                #     0,                                                       # delta
                #     # self.current_time + (self.Tf/self.N) * (i+1),                                # time
                #     0,                                                       # derD   
                #     0,                                                       # derdelta
                # ])
                # self.acados_solver.set(i, "yref", yref)
                self.acados_solver.constraints_set(i, "lh", np.array([
                    self.constraint.along_min,
                    self.constraint.alat_min,
                    self.model.n_min - 0.2,
                    self.model.v_min,
                    self.model.throttle_min,
                    self.model.delta_min,
                    self.constraint.dist_obs1_min,
                    self.constraint.dist_obs2_min,
                    self.constraint.dist_obs3_min,
                    self.constraint.dist_obs4_min,
                    self.constraint.dist_obs5_min,
                    self.constraint.dist_obs6_min,
                ]))
                self.acados_solver.constraints_set(i, "uh", np.array([
                    self.constraint.along_max,
                    self.constraint.alat_max,
                    self.model.n_max + 0.2,
                    self.model.v_max,
                    self.model.throttle_max,
                    self.model.delta_max + 1e-3,
                    self.constraint.dist_obs1_max,
                    self.constraint.dist_obs2_max,
                    self.constraint.dist_obs3_max,
                    self.constraint.dist_obs4_max,
                    self.constraint.dist_obs5_max,
                    self.constraint.dist_obs6_max,
                ]))
                self.acados_solver.set(i, "p", self.objects_frenet_points.flatten())

            # print("obstacles reset")
            # self.objects_frenet_points = np.ones((6, 2), dtype=np.float32) * -100

            s_target = s + self._target_speed * self.Tf
            if s_target > self._global_path_length:
                s_target = self._global_path_length - distance2stop
            # print("s_target_N: ", s_target)111
            yref_N = np.array([
                # -s,
                # (self._global_path_length - distance2stop),
                s_target,     # s
                0,                                     # n
                0,                                     # alpha
                0,                                     # v
                0,                                     # D
                0,                                      # delta
                # 0,                                      # time
            ])
            # self.acados_solver.set(self.N, "yref", yref_N)
            # self.acados_solver.constraints_set(0, "lbx", np.array([s, n, alpha, v, D, delta, theta]))
            # self.acados_solver.constraints_set(0, "ubx", np.array([s, n, alpha, v, D, delta, theta]))
            # if not self.initailize:
                # self.acados_solver.set(0, "lbx", np.array([s, n, alpha, v, D, delta, theta]))
                # self.acados_solver.set(0, "ubx", np.array([s, n, alpha, v, D, delta, theta]))
            #     self.initailize = True
            
            # solve ocp
            # Check if obstacles are too close
            for i, obs in enumerate(self.objects_frenet_points):
                if obs[0] > -50:
                    dist = np.sqrt((s - obs[0])**2 + (n - obs[1])**2)
                    if dist < 1.0:
                        with open(debug_filepath, "a") as file:
                            file.write(f"⚠️  WARNING: Obstacle {i} very close! dist={dist:.4f}m\n")

            # Check if initial state is valid
            if n < self.model.n_min - 0.2 or n > self.model.n_max + 0.2:
                with open(debug_filepath, "a") as file:
                    file.write(f"⚠️  WARNING: Initial n={n:.4f} outside bounds [{self.model.n_min}, {self.model.n_max}]\n")

            if v < self.model.v_min or v > self.model.v_max:
                with open(debug_filepath, "a") as file:
                    file.write(f"⚠️  WARNING: Initial v={v:.4f} outside bounds [{self.model.v_min}, {self.model.v_max}]\n")

            if abs(delta) > self.model.delta_max + 1e-3:
                with open(debug_filepath, "a") as file:
                    file.write(f"⚠️  WARNING: Initial delta={delta:.4f} exceeds max {self.model.delta_max}\n")

            self.loginfo("=" * 40)
            is_valid, reason = self.check_path_association(self._current_pose)
            if not is_valid:
                rospy.logerr(f"Path association failed: {reason}")
                with open(debug_filepath, "a") as f:
                    f.write(f"[{rospy.get_time()}] Path association failed: {reason}\n")
                self.emergency_stop()
                return
            status = self.acados_solver.solve()
            if status != 0:
                self.loginfo("acados returned status {}".format(status))
                if status == 1:
                    self.loginfo("solver failed")
                    self.emergency_stop()
                    self.loginfo("Emergency stop")
                    with open(debug_filepath, "a") as file:
                        file.write("solver failed\n")
                    return
                elif status == 2:
                    self.loginfo("Max number of iterations reached")
                    with open(debug_filepath, "a") as file:
                        file.write("Max number of iterations\n")
                elif status == 3:
                    self.loginfo("Minimum step size reached")
                    with open(debug_filepath, "a") as file:
                        file.write("Min num of iter reached\n")
                elif status == 4:
                    self.loginfo("QP solver failed")
                    # self.emergency_stop()
                    with open(debug_filepath, "a") as file:
                        file.write("QP solver Error\n")
                    self.emergency_stop()

                    self.loginfo("Emergency stop")
                    return

            cost = self.acados_solver.get_cost()
            # self.acados_solver.print_statistics()
            # self.acados_solver.get_stats('residuals')
            self.loginfo("Cost: {}".format(cost))
            # get solution
            for i in range(self.N + 1):
                x = self.acados_solver.get(i, "x")
                self.s_list[i] = x[0]
                
            solution_list = []
            for i in range(self.N):
                solution_list.append(self.acados_solver.get(i, "x"))

            isNaN = False
            predicted_path = Path()
            predicted_path.header.frame_id = "map"
            predicted_path.header.stamp = roscomp.ros_timestamp(self.get_time(), from_sec=True)

            for i, solution in enumerate(solution_list):
                if np.isnan(solution).any():
                    self.loginfo("Nan in solution at index {}".format(i))
                    isNaN = True
                    break

                # self.loginfo("Solution{}: {}".format(i, solution))
                req = FrenetPose(solution[0], 0, 0, solution[1], 0, 0, 0)
                resp = self._frenet2world_service(req)
                pose_msg = self._world2pose(resp)
                pose_stamped = PoseStamped()
                pose_stamped.pose = pose_msg
                predicted_path.poses.append(pose_stamped)

            # print('          s          n     alpha      v         D     delta')
            for i in range(0, self.N+1, 1):
                x = self.acados_solver.get(i, "x")
                # print(f"x{i}: , {x[0]:8.4f}, {x[1]:8.4f}, {x[2]:8.4f}, {x[3]:8.4f}, {x[4]:8.4f}, {x[5]:8.4f}, {x[6]:8.4f}")


            # draw computed trajectory
            if not isNaN:
                self._predicted_path_publisher.publish(predicted_path)

                x0 = self.acados_solver.get(1, "x")
                u0 = self.acados_solver.get(1, "u")
                # print("x0: ", x0)
                # print("u0: ", u0)
                self.derD = u0[0]
                self.derDelta = u0[1]
                self.derTheta = u0[2]

                self.target_D = x0[4]
                self.target_delta = x0[5]
                # print("target_D: ", self.target_D)
                # print("target_delta: ", self.target_delta)

                ########################################################
                #### RESIDUAL MPC - RL ####
                print("Throttle: ", self.target_D)
                print("Steering: ", self.target_delta)
                print("Throttle residual: ", self.throttle_residual)
                print("Steering residual: ", self.steering_residual)
                self.target_D = np.clip(self.target_D + self.throttle_residual, -1, 1)
                normalized_steer = self.target_delta / self.model.delta_max
                steer = np.clip(normalized_steer + self.steering_residual, -1, 1)
                self.target_delta = steer * self.model.delta_max
                ########################################################

          
                if self.target_D >= 0:
                    self.target_gas = self.target_D
                    self.target_brake = 0

                else:
                    self.target_brake = -self.target_D
                    self.target_gas = 0

                self.target_steer = self.target_delta 

                control_msg = CarlaEgoVehicleControl()
                control_msg.steer = self.target_steer
                control_msg.throttle = self.target_gas
                control_msg.brake = self.target_brake
                control_msg.hand_brake = False
                control_msg.manual_gear_shift = False
                # print("Control message: ", control_msg)
                if self.emergency_stop_alert:
                    self.emergency_stop()
                    self.loginfo("Emergency stop from RL")
                else:
                    self._control_cmd_publisher.publish(control_msg)

                current_time = rospy.get_time()
                processing_time = max(current_time - self.time, 1e-9)  # Ensure processing time is never zero
                frequency = 1.0 / processing_time
                # self.loginfo("Processing time: {:.9f} seconds".format(processing_time))
                self.loginfo("Frequency: {:.2f} Hz".format(frequency))
                self.time = current_time
                print(" ------------------------------------------------- ")
    def emergency_stop(self):
        control_msg = CarlaEgoVehicleControl()
        control_msg.steer = 0.0
        control_msg.throttle = 0.0
        control_msg.brake = 0.9
        control_msg.hand_brake = False
        control_msg.manual_gear_shift = False
        self._control_cmd_publisher.publish(control_msg)


def main(args=None):
    """

    main function

    :return:
    """
    roscomp.init("local_planner_mpc", args=args)

    local_planner_mpc = None
    update_timer = None
    try:
        local_planner_mpc = LocalPlannerMPC()
        roscomp.on_shutdown(local_planner_mpc.emergency_stop)

        update_timer = local_planner_mpc.new_timer(
            local_planner_mpc.control_time_step, lambda timer_event=None: local_planner_mpc.run_step())

        local_planner_mpc.spin()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        with open(debug_filepath, "a") as file:
            file.write(f"Error local planner: {e}\n")

    finally:
        roscomp.loginfo('Local planner shutting down.')
        roscomp.shutdown()

if __name__ == "__main__":
    main()
