import numpy as np
from typing import Tuple, Optional
import sys
import os
import math
from casadi import *
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing ACADOS setup
from acados_mpc.acados_settings_mpcc import acados_settings
from solver_diagnostics import SolverDiagnostics
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

        # Vehicle footprint, for the lateral corridor margin.  Defaults are a
        # Tesla Model 3; CarlaMPCEnv overwrites them from the spawned actor's
        # bounding box in _initialize_mpc().
        self.veh_length = 4.69
        self.veh_width = 1.85

        # CARLA's control.steer is normalised [-1, 1] over the vehicle's REAL
        # max_steer_angle, which is 70 deg for vehicle.tesla.model3 -- not the
        # 45 deg model.delta_max used to model the planned steering.  Dividing by
        # delta_max made the car steer 70/45 = 1.556x more than commanded, and
        # carlaEnv read the angle back with the same wrong constant, so the MPC's
        # own delta state was wrong by that factor and it never compensated.
        # Over-steering past the tyres' peak slip angle produces UNDERSTEER, which
        # is why the prediction turned while the car went straight.
        # CarlaMPCEnv overwrites this from the spawned actor's physics control.
        self.carla_max_steer = np.deg2rad(70.0)

        # Steering feedforward, expressed as the angle the normalisation divides
        # by.  CARLA's real max_steer_angle is 70 deg, so dividing by 45 applies
        # 70/45 = 1.556x the planned steering -- a deliberate understeer
        # compensation (see solve()).  SMALLER value = MORE gain.
        #   45 deg -> 1.556x   (current; best measured, 65% collisions)
        #   70 deg -> 1.000x   (no gain; much worse, 93.3%)
        #   40 deg -> 1.750x   35 deg -> 2.000x  (untested)
        self.steer_norm_deg = 45.0

        # Situational speed limits, applied as per-stage bounds on v (the speed
        # bound is already set per stage, so this needs no model change).
        #
        # 1. CURVATURE.  v^2*|kappa| <= alat_max is the lateral acceleration the
        #    PATH demands, independent of the commanded steering.  The model's own
        #    a_lat = C2*v^2*delta is computed from delta in a slip-free kinematic
        #    bicycle, so it under-estimates what a real tyre must supply and never
        #    forces a slow-down before a corner.  This does.
        # 2. OFF-PATH RECOVERY.  The cost is qc*n^2 + qa*alpha^2 with no
        #    saturation, so once the car is well off the path those terms demand a
        #    correction the tyres cannot deliver at speed -- n and alpha grow, the
        #    command grows, and it diverges instead of recovering.  Capping speed
        #    while off-path makes the correction executable.
        # Set either to None to disable.
        self.curvature_speed_limit = False   # OFF: not in the 65% baseline
        self.recover_speed_cap = None     # OFF: not in the 65% baseline (6.0 to enable)
        self.recover_margin = 0.5         # m OUTSIDE the corridor before it fires
        #
        # NOTE: this used to trigger on abs(n) > 2.5, which collided exactly with
        # n_overtake = -2.5 in the cost -- so reaching the intended overtaking
        # position was treated as an off-road emergency and the car braked
        # alongside the vehicle it was passing.  At cap 0 it floored to the 1 m/s
        # minimum and overtaking stopped completely (overtakes 0.88 -> 0.20,
        # timeouts 0% -> 33%).  Being offset WITHIN the corridor is deliberate;
        # only being outside it is an emergency.

        # Tracking diagnostics: does the PLAN leave the corridor, or does the car
        # fail to follow a plan that stayed inside it?  Four corridor changes have
        # now failed to move "outside the corridor" off 64-69%, and raising Zl[2]
        # tenfold changed nothing -- both consistent with a compliant plan the
        # vehicle does not execute.  This measures it directly.
        self.last_pred_n = None      # stage-1 predicted n from the previous solve
        self.last_track_err = 0.0    # |actual n - previously predicted n|
        self.last_plan_violation = 0.0  # how far the PLAN leaves the corridor
        self.lateral_clearance = 0.30

        # Solver-failure handling.  last_good_control is the most recent control
        # from a successful solve; the fallback blends away from it over
        # fallback_blend_steps rather than snapping to centreline tracking.
        self.last_good_control = np.zeros(2)
        self.consecutive_failures = 0
        self.fallback_blend_steps = max(1, int(round(1.0 / dt)))  # ~1 s
        self.solve_failures = 0
        self.solve_calls = 0

        # Records solver inputs so failures can be attributed instead of
        # guessed at.  Off unless CARLA_MPC_DIAG=1.
        self.diagnostics = SolverDiagnostics()

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
        target_lane_d: Optional[float] = None,
        road_width_s = None
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
        # self.acados_solver.set(0, "x", np.array([s, d, alpha, v, D, delta, s]))
        self.acados_solver.set(0, "x", propagated_x)

        # Set obstacle parameters for stage 0
        # self.acados_solver.set(0, "p", obstacles.flatten())
        
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
            
            # Constant margin.  A heading-aware margin
            #     (L/2)|sin a| + (W/2)|cos a| + clearance
            # was tried and REVERTED: collisions 75% -> 83.3%, success 25% ->
            # 16.7%, route completion 56.3% -> 44.8%.  Two reasons it failed.
            #
            # (1) right_width is only ~2.0 m (half a lane; Town01 has no right
            #     lane), so past a ~0.4 rad heading error the margin exceeded it
            #     and n_max went NEGATIVE -- the constraint then *required* the
            #     car to sit left of its own lane centre.  The inversion guard
            #     below never caught it because n_min < n_max throughout; the
            #     corridor stayed ordered and simply migrated off-lane.
            # (2) the per-stage heading came from the previous solution, so the
            #     bounds swung on every solve during a turn.  A fast time-varying
            #     hard bound destabilises SQP_RTI, which takes one warm-started
            #     iteration.
            #
            # The wider lesson from three attempts (Zl[2], junction width, this):
            # tightening the corridor never helped, and "outside the corridor"
            # stayed at 64-69% of collisions through every configuration.  The car
            # is not violating a corridor it can see -- it is failing to TRACK the
            # trajectory it planned.  That is model mismatch, not a constraint.
            safety_margin = 1.2
            if road_widths is not None and road_width_s is not None and len(road_widths) > 0:
                idx = np.argmin(np.abs(road_width_s - s_pred))
                n_min_adaptive = -road_widths[idx, 0] + safety_margin   # negative = world left
                n_max_adaptive =  road_widths[idx, 1] - safety_margin   # positive = world right
            else:
                n_min_adaptive = -5.25 + safety_margin  # fallback: full road left
                n_max_adaptive =  1.75 - safety_margin  # fallback: ego lane right
            
            # Add safety margin
            
            # Clamp to reasonable values
            n_min_adaptive = max(n_min_adaptive, -10.0)
            n_max_adaptive = min(n_max_adaptive, 10.0)

            # A heading-dependent margin can exceed the corridor width on a
            # narrow road (e.g. left_width 2.30 m against a 2.65 m worst-case
            # margin at a = pi/2), which would invert the bounds and hand the QP
            # an empty constraint set.  Keep a small feasible band centred on the
            # corridor instead -- being slightly outside is recoverable, an
            # infeasible stage is not.
            if n_min_adaptive > n_max_adaptive:
                mid = 0.5 * (n_min_adaptive + n_max_adaptive)
                n_min_adaptive, n_max_adaptive = mid - 0.05, mid + 0.05
            
            # Build constraint arrays
            # Per-stage speed ceiling (see __init__).
            v_stage_max = self.model.v_max
            if self.curvature_speed_limit:
                try:
                    s_q = min(max(s_pred, 0.0),
                              getattr(self, '_global_path_length', s_pred) or s_pred)
                    k_pred = abs(float(self.model.kapparef_s(s_q)))
                    if np.isfinite(k_pred) and k_pred > 1e-4:
                        v_stage_max = min(
                            v_stage_max,
                            float(np.sqrt(self.constraint.alat_max / k_pred)))
                except Exception:
                    pass
            if self.recover_speed_cap is not None:
                off_corridor = (d < n_min_adaptive - self.recover_margin
                                or d > n_max_adaptive + self.recover_margin)
                if off_corridor:
                    v_stage_max = min(v_stage_max, self.recover_speed_cap)
            v_stage_max = max(v_stage_max, 1.0)   # never demand a full stop

            lh_constraints = np.array([
                self.constraint.along_min,
                self.constraint.alat_min,
                n_min_adaptive,
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
                n_max_adaptive,
                v_stage_max,
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

            if i == 1:  # keep one representative stage for the diagnostics
                diag_lh, diag_uh = lh_constraints, uh_constraints
                diag_n_min, diag_n_max = n_min_adaptive, n_max_adaptive

        distance2stop = 0.5 * v
        s_target = s + self.target_speed * self.Tf
        
        if hasattr(self, '_global_path_length') and self._global_path_length is not None:
            if s_target > self._global_path_length:
                s_target = self._global_path_length - distance2stop

        # 8. Solve ACADOS OCP
        self.solve_calls += 1
        status = self.acados_solver.solve()

        # Record what the solver was fed, whether or not it succeeded -- the
        # failures only mean something next to the successes.
        if self.diagnostics is not None and self.diagnostics.enabled:
            path_len = getattr(self, '_global_path_length', None) or 0.0
            self.diagnostics.record(
                status=status, s=s, n=d, alpha=alpha, v=v, D=D, delta=delta,
                propagated_x=propagated_x, obstacles=obstacles,
                kappa_fn=self.model.kapparef_s, path_length=path_len,
                n_min=locals().get('diag_n_min', np.nan),
                n_max=locals().get('diag_n_max', np.nan),
                lh=locals().get('diag_lh'), uh=locals().get('diag_uh'),
                v_min=self.model.v_min, v_max=self.model.v_max,
                horizon_s=[s + self.target_speed * self.dt * i
                           for i in range(self.N + 1)],
                solver=self.acados_solver,
            )

        if status != 0:
            # print(f"\n{'='*50}")
            # print(f"⚠️ ACADOS failed status={status}")
            self.solve_failures += 1
            self.consecutive_failures += 1
            # SQP_RTI warm-starts from the previous iterate, so a NaN solution
            # (HPIPM status 3) would otherwise seed every subsequent solve and
            # the controller never recovers within the episode.
            self._reset_solver_iterate(s, d, alpha, v, D, delta)
            return self._fallback_controller(d, alpha, v)

        self.consecutive_failures = 0

        # 9. Extract control from solution
        x0 = self.acados_solver.get(1, "x")

        # One-step tracking error: compare where we actually are against where
        # the previous solve said we would be after one step.
        if self.last_pred_n is not None:
            self.last_track_err = abs(d - self.last_pred_n)
        self.last_pred_n = float(x0[1])

        # Does the planned trajectory itself leave the corridor?
        try:
            worst = 0.0
            for k in range(1, self.N):
                n_k = float(self.acados_solver.get(k, "x")[1])
                worst = max(worst, diag_n_min - n_k, n_k - diag_n_max)
            self.last_plan_violation = max(0.0, worst)
        except Exception:
            self.last_plan_violation = 0.0

        # Amend the record appended before the solve was unpacked; record() runs
        # earlier so that failed solves are captured too.
        if self.diagnostics is not None and self.diagnostics.records:
            self.diagnostics.records[-1].update(
                track_err=float(self.last_track_err),
                plan_violation=float(self.last_plan_violation))
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
        
        # Normalise by the MODEL's delta_max (45 deg), NOT CARLA's actual
        # max_steer_angle (70 deg, measured -- see tools/check_steering.py).
        #
        # This looks like a units bug and is not.  Dividing by 45 while CARLA
        # applies over 70 means the vehicle receives 70/45 = 1.556x the planned
        # steering, and carlaEnv reads the angle back with the same 45 deg, so the
        # MPC sees exactly what it planned.  The net effect is a constant 1.556x
        # feedforward gain that compensates the understeer a slip-free kinematic
        # bicycle cannot predict.
        #
        # "Fixing" it to divide by 70 was tried and was much worse: collisions
        # 83.3% -> 93.3%, success 16.7% -> 6.7%, route completion 44.8% -> 30.8%,
        # with left-side impacts doubling (16% -> 33.9%) as the car ran wide on
        # every turn.  The vehicle genuinely needs that extra steering.
        #
        # Treat 45.0 as a tunable understeer gain, not a measurement.
        steering = target_delta / np.deg2rad(self.steer_norm_deg)
        
        # Clip to valid range
        throttle = np.clip(throttle, -1.0, 1.0)
        steering = np.clip(steering, -1.0, 1.0)
        
        # Update control history
        self.previous_control = self.last_control.copy()
        self.last_control = np.array([throttle, steering])
        self.last_good_control = self.last_control.copy()

        return throttle, steering

    def _reset_solver_iterate(self, s, d, alpha, v, D, delta):
        """
        Clear a poisoned iterate after a failed solve.

        Prefers the solver's own reset(); older acados_template builds do not
        expose it, so fall back to overwriting every stage with a straight
        constant-speed guess, which is enough to get the next RTI step off a
        finite starting point.
        """
        try:
            self.acados_solver.reset()
        except (AttributeError, NotImplementedError):
            pass

        try:
            for i in range(self.N + 1):
                s_guess = s + self.target_speed * self.dt * i
                self.acados_solver.set(
                    i, "x",
                    np.array([s_guess, d, alpha, v, D, delta, s_guess], dtype=float))
                if i < self.N:
                    self.acados_solver.set(i, "u", np.zeros(3))
        except Exception as e:
            print(f"⚠️  Could not reset ACADOS iterate: {e}")

        # The stored derivatives came from the failed solve; they feed the next
        # delay propagation, so drop them too.
        self.derD = 0.0
        self.derDelta = 0.0
        self.derTheta = 0.0

    def _fallback_controller(self, d: float, alpha: float, v: float) -> Tuple[float, float]:
        """
        Control to apply when the OCP fails.

        The previous version was a pure centreline-tracking P controller.  That
        is actively dangerous during obstacle avoidance: it drives d -> 0, which
        is where the obstacle being avoided usually is, so a solver hiccup
        mid-overtake steered the vehicle back into the vehicle it was passing.

        Instead, hold the last control from a successful solve and blend toward
        the P controller over fallback_blend_steps.  A brief failure therefore
        continues the committed manoeuvre, while a sustained one still converges
        to something that tracks the path.
        """
        # Lateral control
        # print("Use control fallback")
        steering = -0.3 * d - 0.5 * alpha
        steering = np.clip(steering, -1.0, 1.0)
        steering = - steering

        # Longitudinal control
        speed_error = self.target_speed - v
        throttle = 0.3 * speed_error
        throttle = np.clip(throttle, -1.0, 1.0)

        # Blend from the last good control toward the P controller.  w = 0 on
        # the first failed step, reaching 1 after fallback_blend_steps.
        w = min(1.0, self.consecutive_failures / float(self.fallback_blend_steps))
        throttle = (1.0 - w) * self.last_good_control[0] + w * throttle
        steering = (1.0 - w) * self.last_good_control[1] + w * steering

        throttle = float(np.clip(throttle, -1.0, 1.0))
        steering = float(np.clip(steering, -1.0, 1.0))

        self.previous_control = self.last_control.copy()
        self.last_control = np.array([throttle, steering])
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
        C2 = 1 / (lr + lf)

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
        
        # OPTION 1: Fast Forward Euler (recommended for t_delay < 0.05s)
        if self.t_delay < 0.05:
            xdot = self._dynamics_of_car(0, x0)
            solution = x0 + self.t_delay * np.array(xdot)
        else:
            # OPTION 2: RK45 for longer delays
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

    def update_path(self, kappa_spline, path_length: float):
        """
        Update path parameters without recompiling ACADOS.
        Call this each episode instead of initialize_acados.
        """
        self._global_path_length = path_length
        
        # Update the curvature spline in the model
        # ACADOS uses the spline coefficients as parameters
        # We just need to update them in the solver at each stage
        for i in range(self.N + 1):
            self.acados_solver.set(i, "p", np.zeros(12))  # reset params first
        
        # Update the kapparef in the model so dynamics use new path
        self.model.kapparef_s = kappa_spline
        
        # Reset control history for new episode
        self.last_control = np.zeros(2)
        self.previous_control = np.zeros(2)
        self.last_good_control = np.zeros(2)
        self.consecutive_failures = 0
        self.solve_failures = 0
        self.solve_calls = 0
        self.derD = 0.0
        self.derDelta = 0.0
        self.derTheta = 0.0
        
        print("✓ MPC path updated (no recompile)")