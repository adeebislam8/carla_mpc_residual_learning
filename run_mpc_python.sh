#!/usr/bin/env bash
# Run the MPC controller only (no ROS, no RL residual).
# Requires a CARLA server already running: ./carla/CarlaUE4.sh

set -e

export PYTHONPATH=$PYTHONPATH:${PWD}/carla/PythonAPI/carla/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"${PWD}/src/mpc_controller/src/acados/lib"
export ACADOS_SOURCE_DIR="${PWD}/src/mpc_controller/src/acados/"

# Solver diagnostics on by default -- this is the debugging entry point, and
# recording is pure numpy with no I/O until the per-episode save.  Writes
# diagnostics/*.npz relative to $PWD; read them with
#   python tools/analyze_solver_failures.py diagnostics/
# Disable with: CARLA_MPC_DIAG=0 ./run_mpc_python.sh
export CARLA_MPC_DIAG="${CARLA_MPC_DIAG:-1}"
echo "CARLA_MPC_DIAG=${CARLA_MPC_DIAG}  (diagnostics -> ${PWD}/diagnostics)"

python train_rlmpc.py --mode test_mpc "$@"
