#!/usr/bin/env bash
# Run the MPC controller only (no ROS, no RL residual).
# Requires a CARLA server already running: ./carla/CarlaUE4.sh

set -e

export PYTHONPATH=$PYTHONPATH:${PWD}/carla/PythonAPI/carla/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"${PWD}/src/mpc_controller/src/acados/lib"
export ACADOS_SOURCE_DIR="${PWD}/src/mpc_controller/src/acados/"

python train_rlmpc.py --mode test_mpc "$@"
