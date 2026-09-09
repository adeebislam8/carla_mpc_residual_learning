# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from casadi import *
# from tracks.readDataFcn import getTrack
import math
SAFETY_DISTANCE = 1.0 # At ellipse boundary
DEG2RAD = math.pi/180.0
RAD2DEG = 180.0/math.pi
DIST2STOP = 0


def distance2obs_casadi_elliptical(s, n, s_obs, n_obs, a=3.5, b=1.4):
    """
    Elliptical safety zone around obstacle
    
    Args:
        s, n: Ego vehicle position in Frenet frame
        s_obs, n_obs: Obstacle position in Frenet frame
        a: Longitudinal semi-axis (meters) - controls fore/aft safety
        b: Lateral semi-axis (meters) - controls side-to-side safety
    
    Returns:
        Normalized distance in elliptical metric
        - distance = 1.0 means exactly at safety boundary
        - distance > 1.0 means safe
        - distance < 1.0 means violation
    """
    # Ignore obstacles that are far behind (same as before)
    condition = s > (s_obs + 5)
    
    # Elliptical distance formula:
    # sqrt((Δs/a)² + (Δn/b)²)
    distance = if_else(
        condition, 
        999999,  # Ignore obstacles behind
        sqrt(((s - s_obs)/a)**2 + ((n - n_obs)/b)**2)  # ELLIPTICAL
    )
    return distance


def bicycle_model(dt, coeff, knots, path_msg, degree=3, use_cbf=True):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "Spatialbicycle_model"
    cbf_min = SAFETY_DISTANCE if use_cbf else -999999

    kapparef_s = Function.bspline('kapparef_s', [knots], coeff, [degree], 1)
    path_length = knots[-1]

    print("path_length: ", path_length)


    ## Race car parameters
    m = 2065.03
    lf = 1.169
    lr = 1.801
    C1 = lr / (lr + lf)
    C2 = 1 / (lr + lf)
    # C2 = 1 / (lr + lf) 

    Cm1 = 9.36424211e+03 
    Cm2 = 4.08690122e+01  
    Cr2 = 2.04799356e+00
    Cr0 = 5.84856121e+02
    Cr3 = 1.13995833e+01

    ## CasADi Model
    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    n_diff = MX.sym("n_diff")
    alpha = MX.sym("alpha")
    v = MX.sym("v")
    v_diff = MX.sym("v_diff")
    D = MX.sym("D")
    delta = MX.sym("delta")
    theta = MX.sym("theta")
    dt_ = MX.sym("dt")
    dt_ = dt
    # yaw_rate = MX.sym("yaw_rate")
    # x = vertcat(s, n, alpha, v, D, delta, time)
    x = vertcat(s, n, alpha, v, D, delta, theta)

    # controls
    derD = MX.sym("derD")
    derDelta = MX.sym("derDelta")
    derTheta = MX.sym("derTheta")
    u = vertcat(derD, derDelta, derTheta)

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    n_diffdot = MX.sym("n_diffdot")
    alphadot = MX.sym("alphadot")
    vdot = MX.sym("vdot")
    v_diffdot = MX.sym("v_diffdot")
    Ddot = MX.sym("Ddot")
    deltadot = MX.sym("deltadot")
    thetadot = MX.sym("thetadot")
    xdot = vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot, thetadot)

    # algebraic variables
    z = vertcat([])

    # parameters
    # p = vertcat([])

    """ obstacle avoidance """
    # parameters
    s_obs1 = MX.sym("s_obs1")
    n_obs1 = MX.sym("n_obs1")

    s_obs2 = MX.sym("s_obs2")
    n_obs2 = MX.sym("n_obs2")

    s_obs3 = MX.sym("s_obs3")
    n_obs3 = MX.sym("n_obs3")

    s_obs4 = MX.sym("s_obs4")
    n_obs4 = MX.sym("n_obs4")

    s_obs5 = MX.sym("s_obs5")
    n_obs5 = MX.sym("n_obs5")

    s_obs6 = MX.sym("s_obs6")
    n_obs6 = MX.sym("n_obs6")

    """ spline coeffients and knots """

    p = vertcat(s_obs1, n_obs1, 
                s_obs2, n_obs2,
                s_obs3, n_obs3,
                s_obs4, n_obs4,
                s_obs5, n_obs5,
                s_obs6, n_obs6)

    """---------------------"""

    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * tanh(Cr3 * v)
    # Fxd = (Cm1 - Cm2*v)*D - Cr2*v**2 - Cr0

    a_long = Fxd / m
    delta = -delta
    sdot = (v * cos(alpha + C1 * delta)) / (1 - kapparef_s(s) * n)
    ndot = v * sin(alpha + C1 * delta)
    f_expl = vertcat(
        sdot,                                                      # sdot
        ndot,                               # ndot
        v * C2 * delta - kapparef_s(s) * sdot,                   # alphadot
        # yaw_rate - kapparef_s(s) * sdot,                     # alphadot
        a_long * cos(C1 * delta),                                  # vdot
        derD,
        derDelta,
        derTheta,
    )
    # constraint on forces
    a_lat = C2 * v * v * delta + a_long * sin(C1 * delta)

    obs_gamma = 1.0
    a_long = 4  # Longitudinal semi-axis (meters)
    b_lat = 2   # Lateral semi-axis (meters)


    """ wrong """
    # b1 = sqrt(((s - s_obs1)/1.0)**2 + ((n - n_obs1)/1.0)**2) 
    # b2 = sqrt(((s - s_obs2)/1.0)**2 + ((n - n_obs2)/1.0)**2) 
    # b3 = sqrt(((s - s_obs3)/1.0)**2 + ((n - n_obs3)/1.0)**2) 
    # b4 = sqrt(((s - s_obs4)/1.0)**2 + ((n - n_obs4)/1.0)**2) 
    # b5 = sqrt(((s - s_obs5)/1.0)**2 + ((n - n_obs5)/1.0)**2) 
    # b6 = sqrt(((s - s_obs6)/1.0)**2 + ((n - n_obs6)/1.0)**2) 
    b1 = distance2obs_casadi_elliptical(s, n, s_obs1, n_obs1, a_long, b_lat)
    b2 = distance2obs_casadi_elliptical(s, n, s_obs2, n_obs2, a_long, b_lat)
    b3 = distance2obs_casadi_elliptical(s, n, s_obs3, n_obs3, a_long, b_lat)
    b4 = distance2obs_casadi_elliptical(s, n, s_obs4, n_obs4, a_long, b_lat)
    b5 = distance2obs_casadi_elliptical(s, n, s_obs5, n_obs5, a_long, b_lat)
    b6 = distance2obs_casadi_elliptical(s, n, s_obs6, n_obs6, a_long, b_lat)
    #all_dist = Function('all_dist', [s, n, s_obs1, n_obs1, s_obs2, n_obs2, s_obs3, n_obs3, s_obs4, n_obs4, s_obs5, n_obs5, s_obs6, n_obs6], [b1, b2, b3, b4, b5, b6])
    s_next = s + sdot * dt_
    n_next = n + ndot * dt_

    b1_next = distance2obs_casadi_elliptical(s_next, n_next, s_obs1, n_obs1, a_long, b_lat)
    b2_next = distance2obs_casadi_elliptical(s_next, n_next, s_obs2, n_obs2, a_long, b_lat)
    b3_next = distance2obs_casadi_elliptical(s_next, n_next, s_obs3, n_obs3, a_long, b_lat)
    b4_next = distance2obs_casadi_elliptical(s_next, n_next, s_obs4, n_obs4, a_long, b_lat)
    b5_next = distance2obs_casadi_elliptical(s_next, n_next, s_obs5, n_obs5, a_long, b_lat)
    b6_next = distance2obs_casadi_elliptical(s_next, n_next, s_obs6, n_obs6, a_long, b_lat)
    print("s_next_type: ", type(s_next))
    print("n_next_type: ", type(n_next))
    #all_dist_next = Function('all_dist_next', [s_next, n_next, s_obs1, n_obs1, s_obs2, n_obs2, s_obs3, n_obs3, s_obs4, n_obs4, s_obs5, n_obs5, s_obs6, n_obs6], [b1_next, b2_next, b3_next, b4_next, b5_next, b6_next])


    dist_obs1 = (b1_next - b1)/dt_ + obs_gamma * b1
    dist_obs2 = (b2_next - b2)/dt_ + obs_gamma * b2
    dist_obs3 = (b3_next - b3)/dt_ + obs_gamma * b3
    dist_obs4 = (b4_next - b4)/dt_ + obs_gamma * b4
    dist_obs5 = (b5_next - b5)/dt_ + obs_gamma * b5
    dist_obs6 = (b6_next - b6)/dt_ + obs_gamma * b6
    """ wrong """


    # Model bounds
    # model.n_min = -4.25  # width of the track [m]
    # model.n_max = 1.0  # width of the track [m]
    model.n_min = -4.25  # or dynamic from road_widths
    model.n_max = 1.0

    model.v_min = 0  # width of the track [m]
    model.v_max = path_length  # width of the track [m]
    # model.v_max = 120  # width of the track [m]

    model.throttle_min = -0.5
    model.throttle_max = 1.0

 
    model.delta_min = -45 * DEG2RAD  # minimum steering angle [rad]
    model.delta_max = 45 * DEG2RAD  # maximum steering angle [rad]

    # input bounds
    model.ddelta_min = -10  # minimum change rate of stering angle [rad/s]
    model.ddelta_max = 10  # maximum change rate of steering angle [rad/s]
    model.dthrottle_min = -50  # -10.0  # minimum throttle change rate
    model.dthrottle_max = 50 # 10.0  # maximum throttle change rate
    model.dtheta_min = -50
    model.dtheta_max = 200

    # nonlinear constraint
    constraint.alat_min = -10  # minimum lateral force [m/s^2]
    constraint.alat_max =  10 # maximum lateral force [m/s^1]

    constraint.along_min = -5  # minimum longitudinal force [m/s^2]
    constraint.along_max = 5 # maximum longitudinal force [m/s^2]

    """ obstacle avoidance """
    constraint.dist_obs1_min = cbf_min
    constraint.dist_obs1_max = 999999

    constraint.dist_obs2_min = cbf_min
    constraint.dist_obs2_max = 999999

    constraint.dist_obs3_min = cbf_min
    constraint.dist_obs3_max = 999999

    constraint.dist_obs4_min = cbf_min
    constraint.dist_obs4_max = 999999

    constraint.dist_obs5_min = cbf_min
    constraint.dist_obs5_max = 999999

    constraint.dist_obs6_min = cbf_min
    constraint.dist_obs6_max = 999999


    """ ------------------ """


    constraint.expr = vertcat(a_long, a_lat, n, v, D, delta, dist_obs1, dist_obs2, dist_obs3, dist_obs4, dist_obs5, dist_obs6)   

    # Define initial conditions
    model.x0 = np.array([0, 0, 0, 0, 0, 0, 0])
    ql = 3e-1     ## if this is low, the car starts to lag; theta is further than s
    qc = 5e-2      # lateral penalty
    qa = 9e-1      # Weight for heading error (alpha)
    gamma = 4e-1  ## TODO: Need to check what is the max
    r1 = 1e-1
    r2 = 1e-2
    # r3 = 2.4e-2
    # kappa_val = kapparef_s(s)
    # r3 = 1e-2 + 5e-1 * kappa_val**2  # fast straight, slows at corners
    # Lookahead distance in meters — how far ahead to check
    lookahead = 5.0  # tune this: larger = brakes earlier

    # Sample curvature ahead
    kappa_now     = kapparef_s(s)
    kappa_ahead1  = kapparef_s(fmin(s + lookahead * 0.33, path_length - 1e-3))   # 5m ahead
    kappa_ahead2  = kapparef_s(fmin(s + lookahead * 0.66, path_length - 1e-3))   # 10m ahead  
    kappa_ahead3  = kapparef_s(fmin(s + lookahead,        path_length - 1e-3))           # 15m ahead

    # Take the maximum curvature in the lookahead window
    kappa_max_ahead = fmax(fabs(kappa_now), 
                    fmax(fabs(kappa_ahead1),
                    fmax(fabs(kappa_ahead2), fabs(kappa_ahead3))))

    r3 = fmin(1.5e-2 + 5.5e-1 * kappa_max_ahead**2, 3e-2)
    k1 = 5e-1
    p1 = 1e-1
    ds1 = s_obs1 - s
    scale1 = if_else(ds1 > 1.0, 1.0, if_else(ds1 > -3.0, 0.2, 0.0))

    # closest_distance = fmin(dist_obs1, fmin(dist_obs2, fmin(dist_obs3, fmin(dist_obs4, fmin(dist_obs5, dist_obs6)))))
    model.cost_expr_ext_cost = (
        (ql * (s - theta) ** 2) 
        + qc * (1 - 0.8 * if_else(ds1 > 0, if_else(ds1 < 15.0, 1.0, 0.0), 0.0)) * n**2
        + qa * alpha**2
        - gamma * derTheta * fmax(0, sign(path_length - s - DIST2STOP))
        + r1 * derD**2 * fmax(0, sign(path_length - s - DIST2STOP))
        + r2 * derDelta**2 * fmax(0, sign(path_length - s - DIST2STOP))
        + r3 * derTheta**2 * fmax(0, sign(path_length - s - DIST2STOP))
        # + scale1 * k1 * (1/fmax(1,(dist_obs1 - 2*SAFETY_DISTANCE) + 1e-7))
        # + k1 * (1/fmax(1,(dist_obs2 - 2*SAFETY_DISTANCE) + 1e-7))
        # + k1 * (1/fmax(1,(dist_obs3 - 2*SAFETY_DISTANCE) + 1e-7))
        - p1 * (fabs(path_length - s - DIST2STOP + 1e-7)) * fmin(0, sign(path_length - s - DIST2STOP))
    )
    model.cost_expr_ext_cost_e =    (     0

    )
    # overtake_zone = exp(-0.2 * (s_obs1 - s)**2)
    # model.cost_expr_ext_cost += -3.0 * overtake_zone * fabs(n)

    # Define model struct
    params = types.SimpleNamespace()
    params.C1 = C1
    params.C2 = C2
    params.Cm1 = Cm1
    params.Cm2 = Cm2
    params.Cr0 = Cr0
    params.Cr2 = Cr2
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params
    model.kapparef_s = kapparef_s
    
    return model, constraint