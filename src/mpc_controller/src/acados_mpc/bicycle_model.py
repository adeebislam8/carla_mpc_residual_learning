# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from casadi import *
# from tracks.readDataFcn import getTrack
from utils.convert_traj_track import parseReference, parseGlobal
import math

DEG2RAD = math.pi/180.0
RAD2DEG = 180.0/math.pi

def bicycle_model(dt, coeff, knots, path_msg, degree=3):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "Spatialbicycle_model"

    kapparef_s = Function.bspline('kapparef_s', [knots], coeff, [degree], 1)

    # load track parameters
    # [ _, _, _, _, _, s0, _, kapparef] = parseReference(reference_msg)
    # kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)

    # [ss, _, _, dd_ds, dv_ds] = parseGlobal(global_path_frenet_msg)
    # nglobaldot_s = interpolant("nglobal_s", "bspline", [ss], dd_ds)
    # vglobaldot_s = interpolant("vglobal_s", "bspline", [ss], dv_ds)

    ## need curvature (kapparef) spline for global path along s axis
        
    # _, _, _, dense_s, _, kappa_ref = parseReference(path_msg)
    # kapparef_s = interpolant("kapparef_s", "bspline", [dense_s], kappa_ref)



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
    time = MX.sym("time")
    # yaw_rate = MX.sym("yaw_rate")
    # x = vertcat(s, n, alpha, v, D, delta, time)
    x = vertcat(s, n, alpha, v, D, delta)

    # controls
    derD = MX.sym("derD")
    derDelta = MX.sym("derDelta")
    u = vertcat(derD, derDelta)

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    n_diffdot = MX.sym("n_diffdot")
    alphadot = MX.sym("alphadot")
    vdot = MX.sym("vdot")
    v_diffdot = MX.sym("v_diffdot")
    Ddot = MX.sym("Ddot")
    deltadot = MX.sym("deltadot")
    # yaw_ratedot = MX.sym("yaw_ratedot")
    time_dot = MX.sym("time_dot")
    time_dot = 1
    # xdot = vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot, time_dot)
    xdot = vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot)

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

    # a_long = (aa*np.log(D)/np.log(1+ab) + ac) * v + ba**np.log(D)/np.log(1+bb) + bc
    # Fxd = along * m

    # 
    # dynamics
    # Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 # for mpcc
    # Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * tanh(5 * v)
    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * tanh(Cr3 * v)
    # Fxd = (Cm1 - Cm2*v)*D - Cr2*v**2 - Cr0

    a_long = Fxd / m
    delta = -delta
    sdot = (v * cos(alpha + C1 * delta)) / (1 - kapparef_s(s) * n)
    ndot = v * sin(alpha + C1 * delta)
    # f_expl = vertcat(
    #     sdot,                                                      # sdot
    #     ndot,                               # ndot
    #     v * sin(alpha + C1 * delta) - nglobaldot_s(s) * sdot,      
    #     v * C2 * delta - kapparef_s(s) * sdot,                   # alphadot
    #     # yaw_rate - kapparef_s(s) * sdot,                     # alphadot
    #     a_long * cos(C1 * delta),                                  # vdot
    #     a_long * cos(C1 * delta) - vglobaldot_s(s) * sdot,
    #     derD,
    #     derDelta,
    #     # (C1 * v * cos(C1 * delta) * derDelta + sin(C1*delta) * a_long * cos(C1 * delta))/lr,    # "vdot = a_long * cos(C1 * delta)"
    # )
    f_expl = vertcat(
        sdot,                                                      # sdot
        ndot,                               # ndot
        # v * sin(alpha + C1 * delta) - nglobaldot_s(s) * sdot,      
        v * C2 * delta - kapparef_s(s) * sdot,                   # alphadot
        # yaw_rate - kapparef_s(s) * sdot,                     # alphadot
        a_long * cos(C1 * delta),                                  # vdot
        # a_long * cos(C1 * delta) - vglobaldot_s(s) * sdot,
        derD,
        derDelta,
        # time_dot,
        # (C1 * v * cos(C1 * delta) * derDelta + sin(C1*delta) * a_long * cos(C1 * delta))/lr,    # "vdot = a_long * cos(C1 * delta)"
    )
    # constraint on forces
    a_lat = C2 * v * v * delta + a_long * sin(C1 * delta)

    gamma = 0.7
    SAFETY_DISTANCE = 5

    """ wrong """
    b1 = sqrt(((s - s_obs1)/1.0)**2 + ((n - n_obs1)/1.0)**2) 
    b2 = sqrt(((s - s_obs2)/1.0)**2 + ((n - n_obs2)/1.0)**2) 
    b3 = sqrt(((s - s_obs3)/1.0)**2 + ((n - n_obs3)/1.0)**2) 
    b4 = sqrt(((s - s_obs4)/1.0)**2 + ((n - n_obs4)/1.0)**2) 
    b5 = sqrt(((s - s_obs5)/1.0)**2 + ((n - n_obs5)/1.0)**2) 
    b6 = sqrt(((s - s_obs6)/1.0)**2 + ((n - n_obs6)/1.0)**2) 
    s_next = s + sdot * dt
    n_next = n + ndot * dt
    b1_next = sqrt(((s_next - s_obs1)/1.0)**2 + ((n_next - n_obs1)/1.0)**2) 
    b2_next = sqrt(((s_next - s_obs2)/1.0)**2 + ((n_next - n_obs2)/1.0)**2) 
    b3_next = sqrt(((s_next - s_obs3)/1.0)**2 + ((n_next - n_obs3)/1.0)**2) 
    b4_next = sqrt(((s_next - s_obs4)/1.0)**2 + ((n_next - n_obs4)/1.0)**2) 
    b5_next = sqrt(((s_next - s_obs5)/1.0)**2 + ((n_next - n_obs5)/1.0)**2) 
    b6_next = sqrt(((s_next - s_obs6)/1.0)**2 + ((n_next - n_obs6)/1.0)**2) 

    # dist_obs1 = b1_next - b1 + gamma * b1
    # dist_obs2 = b2_next - b2 + gamma * b2
    # dist_obs3 = b3_next - b3 + gamma * b3
    # dist_obs4 = b4_next - b4 + gamma * b4
    # dist_obs5 = b5_next - b5 + gamma * b5
    # dist_obs6 = b6_next - b6 + gamma * b6
    """ wrong """

    dist_obs1 = sqrt((s - s_obs1)*(s - s_obs1) + (n - n_obs1)*(n - n_obs1))
    dist_obs2 = sqrt((s - s_obs2)*(s - s_obs2) + (n - n_obs2)*(n - n_obs2))
    dist_obs3 = sqrt((s - s_obs3)*(s - s_obs3) + (n - n_obs3)*(n - n_obs3))
    dist_obs4 = sqrt((s - s_obs4)*(s - s_obs4) + (n - n_obs4)*(n - n_obs4))
    dist_obs5 = sqrt((s - s_obs5)*(s - s_obs5) + (n - n_obs5)*(n - n_obs5))
    dist_obs6 = sqrt((s - s_obs6)*(s - s_obs6) + (n - n_obs6)*(n - n_obs6))

    # # dist_obs1 = sqrt((s - s_obs1)*(s - s_obs1) + (n - n_obs1)*(n - n_obs1)*0.5) - SAFETY_DISTANCE 
    # # dist_obs2 = sqrt((s - s_obs2)*(s - s_obs2) + (n - n_obs2)*(n - n_obs2)*0.5) - SAFETY_DISTANCE
    # # dist_obs3 = sqrt((s - s_obs3)*(s - s_obs3) + (n - n_obs3)*(n - n_obs3)*0.5) - SAFETY_DISTANCE
    # # dist_obs4 = sqrt((s - s_obs4)*(s - s_obs4) + (n - n_obs4)*(n - n_obs4)*0.5) - SAFETY_DISTANCE
    # # dist_obs5 = sqrt((s - s_obs5)*(s - s_obs5) + (n - n_obs5)*(n - n_obs5)*0.5) - SAFETY_DISTANCE
    # # dist_obs6 = sqrt((s - s_obs6)*(s - s_obs6) + (n - n_obs6)*(n - n_obs6)*0.5) - SAFETY_DISTANCE

    # h1_dot = ((s - s_obs1)*0.25*sdot + (n - n_obs1)*ndot)/dist_obs1
    # h2_dot = ((s - s_obs2)*0.25*sdot + (n - n_obs2)*ndot)/dist_obs2
    # h3_dot = ((s - s_obs3)*0.25*sdot + (n - n_obs3)*ndot)/dist_obs3
    # h4_dot = ((s - s_obs4)*0.25*sdot + (n - n_obs4)*ndot)/dist_obs4
    # h5_dot = ((s - s_obs5)*0.25*sdot + (n - n_obs5)*ndot)/dist_obs5
    # h6_dot = ((s - s_obs6)*0.25*sdot + (n - n_obs6)*ndot)/dist_obs6



    # # h1 = sqrt(dist_obs1**2 - SAFETY_DISTANCE**2)    
    # # h2 = sqrt(dist_obs2**2 - SAFETY_DISTANCE**2)    
    # # h3 = sqrt(dist_obs3**2 - SAFETY_DISTANCE**2)    
    # # h4 = sqrt(dist_obs4**2 - SAFETY_DISTANCE**2)    
    # # h5 = sqrt(dist_obs5**2 - SAFETY_DISTANCE**2)    
    # # h6 = sqrt(dist_obs6**2 - SAFETY_DISTANCE**2)
    # h1 = dist_obs1 - SAFETY_DISTANCE    
    # h2 = dist_obs2 - SAFETY_DISTANCE    
    # h3 = dist_obs3 - SAFETY_DISTANCE    
    # h4 = dist_obs4 - SAFETY_DISTANCE    
    # h5 = dist_obs5 - SAFETY_DISTANCE    
    # h6 = dist_obs6 - SAFETY_DISTANCE


    # dist_obs1 = h1_dot + gamma * h1
    # dist_obs2 = h2_dot + gamma * h2
    # dist_obs3 = h3_dot + gamma * h3
    # dist_obs4 = h4_dot + gamma * h4
    # dist_obs5 = h5_dot + gamma * h5
    # dist_obs6 = h6_dot + gamma * h6

    # Model bounds
    model.n_min = -1.7  # width of the track [m]
    model.n_max = 2.0  # width of the track [m]
    # model.n_min = -4.0  # width of the track [m]
    # model.n_max = 4.0  # width of the track [m]

    model.v_min = 0  # width of the track [m]
    model.v_max = 150  # width of the track [m]

    # state bounds
    model.throttle_min = -1.0
    model.throttle_max = 1.0

    # model.delta_min = -0.27  # minimum steering angle [rad]
    # model.delta_max = 0.27  # maximum steering angle [rad]

    model.delta_min = -30 * DEG2RAD  # minimum steering angle [rad]
    model.delta_max = 30 * DEG2RAD  # maximum steering angle [rad]

    # input bounds
    model.ddelta_min = -1  # minimum change rate of stering angle [rad/s]
    model.ddelta_max = 1  # maximum change rate of steering angle [rad/s]
    model.dthrottle_min = -1  # -10.0  # minimum throttle change rate
    model.dthrottle_max = 1 # 10.0  # maximum throttle change rate

    # nonlinear constraint
    constraint.alat_min = -5  # minimum lateral force [m/s^2]
    constraint.alat_max =  5  # maximum lateral force [m/s^1]

    constraint.along_min = -10  # minimum longitudinal force [m/s^2]
    constraint.along_max = 10 # maximum longitudinal force [m/s^2]

    """ obstacle avoidance """
    # constraint.dist_obs1_min = SAFETY_DISTANCE
    # constraint.dist_obs1_max = 999999

    # constraint.dist_obs2_min = SAFETY_DISTANCE
    # constraint.dist_obs2_max = 999999

    # constraint.dist_obs3_min = SAFETY_DISTANCE
    # constraint.dist_obs3_max = 999999

    # constraint.dist_obs4_min = SAFETY_DISTANCE
    # constraint.dist_obs4_max = 999999

    # constraint.dist_obs5_min = SAFETY_DISTANCE
    # constraint.dist_obs5_max = 999999

    # constraint.dist_obs6_min = SAFETY_DISTANCE
    # constraint.dist_obs6_max = 999999


    """ ------------------ """

    # define constraints struct
    # constraint.along = Function("a_lat", [x, u], [a_long])
    # constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.expr = vertcat(
        a_long, a_lat, n, v, D, delta, 
        # dist_obs1, dist_obs2, dist_obs3, dist_obs4, dist_obs5, dist_obs6
        )


    # Define initial conditions
    model.x0 = np.array([0, 0, 0, 0, 0, 0])

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
    
    return model, constraint