#!/usr/bin/env bash

# Trap Ctrl+C and cleanup
trap cleanup INT TERM

cleanup() {
    echo "Shutting down all processes..."
    pkill -P $$  # Kill all child processes
    kill 0       # Kill all processes in the group
    exit 0
}

export PYTHONPATH=$PYTHONPATH:${PWD}/carla/PythonAPI/carla/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"${PWD}/src/mpc_controller/src/acados/lib"
export ACADOS_SOURCE_DIR="${PWD}/src/mpc_controller/src/acados/"


source devel/setup.bash

roslaunch mpc_controller mpcc_all.launch &
LAUNCH_PID=$!

sleep 2

python ${PWD}/carla/PythonAPI/examples/generate_traffic.py --asynch --filterv vehicle.toyota* -s 1 -w 0 -n 40 &
TRAFFIC_PID=$!

sleep 2

python ${PWD}/src/mpc_controller/envs/mpc_ros_eval.py
PYTHON_PID=$!

wait $PYTHON_PID

cleanup