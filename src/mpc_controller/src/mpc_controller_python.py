import numpy as np
from typing import Tuple, Optional
import sys
import os
import math
from casadi import *
sys.path.append('/home/ave/Desktop/carla_mpc_residual_learning/src/mpc_controller/src')

# Import existing ACADOS setup
from acados_mpc.acados_settings_mpcc import acados_settings
from scipy.integrate import solve_ivp


class MPCController:
    def __init__(
        self,
        horizon: int = 20,
        dt: float = 0.05,
        frenet_converter = None,
        target_speed: float = 8.33,
        t_delay: float = 0.03  # Time delay for propagation
    ):
        """
        Args:
            horizon: MPC prediction horizon (N)
            dt: Timestep duration
            frenet_converter: FrenetConverter instance
            target_speed: Target velocity (m/s)
            t_delay: Time delay for control propagation
        """
        self.horizon = horizon
        self.N = horizon  # Same as horizon
        self.dt = dt
        self.Tf = dt * horizon  # Total prediction time
        self.t_delay = t_delay
        self.frenet_converter = frenet_converter
        self.target_speed = target_speed
        
        # ACADOS solver (initialized lazily)
        self.acados_solver = None
        self.constraint = None
        self.model = None
        
        # Control history
        self.last_control = np.zeros(2)
        self.previous_control = np.zeros(2)
        
        # Control derivatives (initialized to zero)
        self.derD = 0.0
        self.derDelta = 0.0
        self.derTheta = 0.0
    
    def initialize_acados(self, path_curvature_spline, path_msg=None):
        """
        Initialize ACADOS solver with path information
        """
        # Call YOUR existing ACADOS setup function
        self.constraint, self.model, self.acados_solver = acados_settings(
            Tf=self.Tf,
            N=self.N,
            coeffs=path_curvature_spline.c,
            knots=path_curvature_spline.t,
            path_msg=path_msg,
            degree=3
        )
        
        print("✓ ACADOS solver initialized")
    
    def solve(
        self, 
        s: float,
        d: float, 
        alpha: float, 
        v: float, 
        obstacles: np.ndarray,  # Shape (num_obstacles, 2) [s, d]
        D: float = 0.0, 
        delta: float = 0.0,
        road_widths: Optional[np.ndarray] = None,
        target_lane_d: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Args:
            s: Current arc length
            d: Lateral deviation
            alpha: Heading error
            v: Velocity (m/s)
            obstacles: Array of obstacles in Frenet frame (N, 2) [s, d]
            D: Current throttle/brake
            delta: Current steering angle
            road_widths: Optional road width constraints
            target_lane_d: Target lateral position (None = stay in current lane)
            
        Returns:
            (throttle, steering): Control outputs in [-1, 1]
        """

        if self.acados_solver is None:
            print("⚠️  ACADOS solver not initialized, using fallback controller")
            return self._fallback_controller(d, alpha, v)
        
        # 1. Propagate with time delay
        x0p = np.array([s, d, alpha, v, D, delta, s])
        u0p = np.array([self.derD, self.derDelta, self.derTheta])
        propagated_x = self.propagate_time_delay(x0p, u0p)
        
        # 2. IMPROVED: Relaxed initial constraints to allow lateral movement
        propagated_x_lower = propagated_x.copy()
        propagated_x_upper = propagated_x.copy()
        #propagated_x_upper[0] += 2.0  # Allow up to 2m forward movement in s
        #propagated_x_lower[1] -= 1.5  # IMPROVED: Allow 1.5m lateral movement LEFT
        #propagated_x_upper[1] += 1.5  # IMPROVED: Allow 1.5m lateral movement RIGHT
        
        self.acados_solver.set(0, "lbx", propagated_x_lower)
        self.acados_solver.set(0, "ubx", propagated_x_upper)
        
        # 3. Set initial state
        self.acados_solver.set(0, "x", np.array([s, d, alpha, v, D, delta, s]))

        # Set obstacle parameters for stage 0
        self.acados_solver.set(0, "p", obstacles.flatten())
        
        # 4. Warm-start with trajectory toward target lane
        if target_lane_d is not None:
            # Create a trajectory that smoothly transitions to target lane
            for i in range(self.N):
                progress = i / self.N  # 0 to 1
                
                # Linear interpolation from current d to target_lane_d
                interp_d = d + progress * (target_lane_d - d)
                interp_s = s + self.target_speed * self.dt * i
                interp_alpha = 0.0  # Assume heading aligns with path
                interp_v = self.target_speed  # Maintain target speed
                
                # Warm-start state guess
                x_warmstart = np.array([
                    interp_s,
                    interp_d,
                    interp_alpha,
                    interp_v,
                    D,
                    delta,
                    interp_s
                ])
                
                self.acados_solver.set(i, "x", x_warmstart)
        
        # 5. Set obstacle parameters
        self.acados_solver.set(0, "p", obstacles.flatten())
        
        # 6. Count valid obstacles (those with s > -50)
        valid_obs_count = np.sum(obstacles[:, 0] > -50)
        
        # 7. Set constraints for each horizon step
        for i in range(1, self.N):
            # Predict arc length at timestep i
            s_pred = s + self.target_speed * (self.Tf / self.N) * i
            
            # Get road bounds (use defaults if not provided)
            if road_widths is not None and len(road_widths) > 0:
                # Get road width at predicted s
                idx = int(s_pred / 1.0)  # Assuming 1m spacing
                idx = np.clip(idx, 0, len(road_widths) - 1)
                n_left = road_widths[idx, 0]
                n_right = -road_widths[idx, 1]
            else:
                # Default road bounds
                n_left = 0.5
                n_right = -0.5
            
            # Add safety margin
            safety_margin = 0.2  # IMPROVED: Reduced from 0.4 for more flexibility
            n_min_adaptive = n_right + safety_margin
            n_max_adaptive = n_left - safety_margin
            
            # Clamp to reasonable values
            n_min_adaptive = max(n_min_adaptive, -10.0)
            n_max_adaptive = min(n_max_adaptive, 10.0)
            
            # Build constraint arrays
            lh_constraints = np.array([
                self.constraint.along_min,
                self.constraint.alat_min,
                n_min_adaptive - 0.1,
                self.model.v_min,
                self.model.throttle_min,
                self.model.delta_min,
                self.constraint.dist_obs1_min,
                self.constraint.dist_obs2_min,
                self.constraint.dist_obs3_min,
                self.constraint.dist_obs4_min,
                self.constraint.dist_obs5_min,
                self.constraint.dist_obs6_min,
            ])
            
            uh_constraints = np.array([
                self.constraint.along_max,
                self.constraint.alat_max,
                n_max_adaptive + 0.1,
                self.model.v_max,
                self.model.throttle_max,
                self.model.delta_max + 1e-3,
                self.constraint.dist_obs1_max,
                self.constraint.dist_obs2_max,
                self.constraint.dist_obs3_max,
                self.constraint.dist_obs4_max,
                self.constraint.dist_obs5_max,
                self.constraint.dist_obs6_max,
            ])

            # Disable constraints for invalid obstacles
            for obs_idx in range(valid_obs_count, 6):
                constraint_idx = 6 + obs_idx  # Obstacle constraints start at index 6
                lh_constraints[constraint_idx] = -1e9
                uh_constraints[constraint_idx] = 1e9

            # Set constraints for this timestep
            self.acados_solver.constraints_set(i, "lh", lh_constraints)
            self.acados_solver.constraints_set(i, "uh", uh_constraints)
            
            # Set obstacle parameters for this timestep
            self.acados_solver.set(i, "p", obstacles.flatten())
        
        distance2stop = 0.5 * v
        s_target = s + self.target_speed * self.Tf
        
        if hasattr(self, '_global_path_length') and self._global_path_length is not None:
            if s_target > self._global_path_length:
                s_target = self._global_path_length - distance2stop

        # 8. Solve ACADOS OCP
        status = self.acados_solver.solve()
        
        if status != 0:
            print(f"⚠️  ACADOS solver failed with status {status}")
            if status == 4:
                print("    QP solver failed - constraints may be infeasible")
            return self._fallback_controller(d, alpha, v)
        
        # 9. Extract control from solution
        x0 = self.acados_solver.get(1, "x")
        u0 = self.acados_solver.get(1, "u")
        
        # Update control derivatives for next iteration
        self.derD = u0[0]
        self.derDelta = u0[1]
        self.derTheta = u0[2]
        
        # Extract target states
        target_D = x0[4]
        target_delta = x0[5]
        
        # Convert to normalized throttle/steering
        if target_D >= 0:
            throttle = target_D
        else:
            throttle = target_D  # Negative for braking
        
        steering = target_delta / self.model.delta_max  # Normalize by max angle
        
        # Clip to valid range
        throttle = np.clip(throttle, -1.0, 1.0)
        steering = np.clip(steering, -1.0, 1.0)
        
        # Update control history
        self.previous_control = self.last_control.copy()
        self.last_control = np.array([throttle, steering])
        
        return throttle, steering
    
    def _fallback_controller(self, d: float, alpha: float, v: float) -> Tuple[float, float]:
        """Simple P controller as fallback"""
        # Lateral control
        print("Use control fallback")
        steering = -0.3 * d - 0.5 * alpha
        steering = np.clip(steering, -1.0, 1.0)
        steering = - steering
        
        # Longitudinal control
        speed_error = self.target_speed - v
        throttle = 0.3 * speed_error
        throttle = np.clip(throttle, -1.0, 1.0)
        
        return throttle, steering
    
    def get_last_control(self) -> np.ndarray:
        """Get last control output"""
        return self.last_control
    
    def get_previous_control(self) -> np.ndarray:
        """Get previous control output"""
        return self.previous_control
    
    def get_predicted_trajectory(self) -> np.ndarray:
        """
        Get MPC predicted trajectory
        
        Returns:
            Array of shape (N, 2) with [s, d] predictions
        """
        if self.acados_solver is None:
            return np.zeros((self.horizon, 2))
        
        trajectory = []
        for i in range(self.horizon):
            x = self.acados_solver.get(i, "x")
            trajectory.append([x[0], x[1]])  # s, d
        
        return np.array(trajectory)
    
    def _dynamics_of_car(self, t, x0) -> list:
        """
        Vehicle dynamics for forward propagation
        """
        # Race car parameters
        m = 2065.03
        lf = 1.169
        lr = 1.801
        C1 = lr / (lr + lf)
        C2 = 1 / (lr + lf) * 0.5

        Cm1 = 9.36424211e+03 
        Cm2 = 4.08690122e+01  
        Cr2 = 2.04799356e+00
        Cr0 = 5.84856121e+02
        Cr3 = 1.13995833e+01
        
        s, n, alpha, v, D, delta, theta, derD, derDelta, derTheta = x0
        
        kapparef_s = self.model.kapparef_s

        Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * tanh(Cr3 * v)
        a_long = Fxd / m
        delta = -delta
        
        sdot = (v * cos(alpha + C1 * delta)) / (1 - kapparef_s(s) * n)
        ndot = v * sin(alpha + C1 * delta)
        alphadot = v * C2 * delta - kapparef_s(s) * sdot
        vdot = a_long * cos(C1 * delta)
        Ddot = derD
        deltadot = derDelta
        thetadot = derTheta

        xdot = [float(sdot), ndot, float(alphadot), vdot, Ddot, deltadot, thetadot, Ddot, deltadot, thetadot]
        return xdot

    def propagate_time_delay(self, states: np.array, inputs: np.array) -> np.array:
        """
        Propagate state forward by time delay using vehicle dynamics
        """
        # Initial condition on the ODE
        x0 = np.concatenate((states, inputs), axis=0)
        
        solution = solve_ivp(
            self._dynamics_of_car,
            t_span=[0, self.t_delay],
            y0=x0,
            method="RK45",
            atol=1e-8,
            rtol=1e-8,
        )
        
        solution = [x[-1] for x in solution.y]

        # Extract states and apply constraints
        s, n, alpha, v, D, delta, theta = solution[:7]
        
        if abs(delta) > self.model.delta_max:
            delta = np.sign(delta) * self.model.delta_max

        if abs(D) > self.model.throttle_max:
            D = np.sign(D) * self.model.throttle_max

        return np.array([s, n, alpha, v, D, delta, theta])