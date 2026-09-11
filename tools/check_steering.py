#!/usr/bin/env python
"""
Measure CARLA's real steering limit and compare it with the 45 deg the
controller assumes.

The chain is:
    MPC      steering = delta / 45deg                 (mpc_controller_python.py)
    env      control.steer = -steering                (carlaEnv.py:1260)
    CARLA    applied angle = steer * max_steer_angle  (vehicle physics)
    env      current_steering = steer * 45deg         (carlaEnv.py:829)

If max_steer_angle != 45 deg every command is scaled by max/45, AND the delta
fed back to the MPC is wrong by the same factor -- so the controller cannot see
the error and never compensates.  ratio < 1 means the car turns LESS than the
MPC asked for: the prediction turns while the car keeps going straight.

    python tools/check_steering.py
"""
import argparse
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--bp', default='vehicle.tesla.model3')
    ap.add_argument('--assumed-deg', type=float, default=45.0)
    args = ap.parse_args()

    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()

    bp = world.get_blueprint_library().filter(args.bp)[0]
    sp = world.get_map().get_spawn_points()[0]
    veh = None
    try:
        veh = world.try_spawn_actor(bp, sp)
        if veh is None:
            raise RuntimeError(f"could not spawn {args.bp} at the first spawn point")
        world.tick() if world.get_settings().synchronous_mode else world.wait_for_tick()

        pc = veh.get_physics_control()
        print(f"{args.bp}")
        print(f"  mass {pc.mass:.1f} kg")
        angles = []
        for i, w in enumerate(pc.wheels):
            side = ['FL', 'FR', 'RL', 'RR'][i] if i < 4 else str(i)
            print(f"  wheel {side}: max_steer_angle = {w.max_steer_angle:6.2f} deg")
            angles.append(w.max_steer_angle)
        real = max(angles)

        bb = veh.bounding_box.extent
        print(f"  bounding box: {2*bb.x:.2f} x {2*bb.y:.2f} m")

        print("\nCOMPARISON")
        print(f"  controller assumes : {args.assumed_deg:.2f} deg")
        print(f"  CARLA actually has : {real:.2f} deg")
        ratio = real / args.assumed_deg
        print(f"  ratio (real/assumed): {ratio:.3f}")

        if abs(ratio - 1.0) < 0.02:
            print("\n  MATCH -- steering scale is not the problem.")
        else:
            worse = "LESS" if ratio < 1 else "MORE"
            print(f"\n  MISMATCH: the car turns {worse} than the MPC commands,")
            print(f"  by a factor of {ratio:.3f}.")
            for d in (10, 20, 30, 45):
                cmd = d / args.assumed_deg
                got = cmd * real
                print(f"     MPC wants {d:2d} deg -> steer {cmd:+.3f} -> car does {got:5.2f} deg")
            print(f"\n  Fix: set model.delta_max = {math.radians(real):.4f}"
                  f"  ({real:.2f} deg) in bicycle_model_mpcc_cbf.py,")
            print(f"  and the 45-deg constant in carlaEnv.py:829, so all three agree.")
    finally:
        if veh is not None:
            veh.destroy()


if __name__ == '__main__':
    main()
