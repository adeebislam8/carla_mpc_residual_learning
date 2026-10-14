#!/usr/bin/env bash
# Probe the nominal MPCC on every hard-scenario case, then print one table.
#
#   ./tools/probe_scenarios.sh                         # Town01-03, seed 1, 30 eps
#   PERCEPTION=1 ./tools/probe_scenarios.sh            # YOLO 3-camera vehicles
#   TOWNS="Town04 Town05" SEEDS="1 2" ./tools/probe_scenarios.sh
#   CASES="s1 s3" ./tools/probe_scenarios.sh           # a subset
#   TAG=watch CASES=s3 EPISODES=2 EXTRA="--render" ./tools/probe_scenarios.sh   # watch it
#   MPCC2=1 ./tools/probe_scenarios.sh                 # improved MPCC (follow, prediction)
#   TAG=check EPISODES=3 EXTRA="--scenario-trace results/traces" ./tools/probe_scenarios.sh
#       # per-episode CSV + PNG of ego and scenario cars, to check the geometry
#
# Cases (src/mpc_controller/src/scenarios.py):
#   s0  normal       default random NPCs (now actually spawned: --traffic-fix)
#   s1  hard_brake   lead car brakes hard to a stop          (HardBreakRoute)
#   s2  stopped      parked car or 3-car accident in lane    (ParkedObstacle/Accident)
#   s3  stopped_oncoming  accident + traffic in the passing lane  (*TwoWays)
#   s4  cut_in       parked car pulls into the ego lane      (ParkingCutIn)
#   s5  dense        8-15 random NPCs
#
# Needs a CARLA 0.9.16 server already running.  Runs in the foreground; each
# (case, town) is one benchmark_mpcc.py call writing results/probe_<case>_<town>.json,
# skipped if it exists, so re-running resumes.  Routes are 100-150 m so every
# scenario has room to place its event.

set -uo pipefail
cd "$(dirname "$0")/.."

TOWNS=${TOWNS:-"Town01 Town02 Town03"}
SEEDS=${SEEDS:-"1"}
EPISODES=${EPISODES:-30}
CASES=${CASES:-"s0 s1 s2 s3 s4 s5"}
PERCEPTION=${PERCEPTION:-0}
TAG=${TAG:-}
[ "$PERCEPTION" = "1" ] && TAG="${TAG}y"

BASE="--seeds $SEEDS --episodes $EPISODES --route-min 100 --route-max 150 \
--qc 0.5 --gate-depth 0.98 --steer-fix --spawn-check --traffic-fix"
[ "$PERCEPTION" = "1" ] && BASE="$BASE --perception"
EXTRA=${EXTRA:-}
# STALL_KICK=clear: no anti-stall lunge into stopped obstacles (carlaEnv stall_kick).
STALL_KICK=${STALL_KICK:-always}
BASE="$BASE --stall-kick $STALL_KICK"
[ "$STALL_KICK" != "always" ] && TAG="${TAG}k${STALL_KICK}"
# MPCC2=1: the improved nominal controller -- no stall lunge, following
# distance, constant-velocity obstacle prediction (tag 'm2').
MPCC2=${MPCC2:-0}
if [ "$MPCC2" = "1" ]; then
  BASE="$BASE --stall-kick clear --follow --predict-obstacles"
  TAG="${TAG}m2"
fi

case_flags() {
  case "$1" in
    s0) echo "--scenario none" ;;
    s1) echo "--scenario hard_brake" ;;
    s2) echo "--scenario stopped" ;;
    s3) echo "--scenario stopped_oncoming" ;;
    s4) echo "--scenario cut_in" ;;
    s5) echo "--scenario none --npc-min 8 --npc-max 15" ;;
    *)  echo "unknown case $1" >&2; return 1 ;;
  esac
}

mkdir -p results
START=$(date +%s)
for T in $TOWNS; do
  for C in $CASES; do
    LABEL="probe${TAG}_${C}_${T}"
    if [ -f "results/${LABEL}.json" ]; then
      echo "=== ${LABEL}: exists, skipped"; continue
    fi
    FLAGS=$(case_flags "$C") || exit 1
    echo; echo "=== ${LABEL}   (elapsed $((($(date +%s)-START)/60)) min)"
    # shellcheck disable=SC2086
    python tools/benchmark_mpcc.py --label "$LABEL" --town "$T" $BASE $FLAGS $EXTRA
  done
done

echo; echo "finished in $((($(date +%s)-START)/60)) min"
python tools/summarize_scenarios.py --prefix "probe${TAG}"
