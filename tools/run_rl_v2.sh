#!/usr/bin/env bash
# RL v2: train one SAC residual under the 2026-10-08 fixes, then evaluate it
# cross-town against the nominal MPCC baseline.
#
#   ./tools/run_rl_v2.sh              # train + evaluate
#   ./tools/run_rl_v2.sh --eval-only  # evaluate an existing models/sac_v2/final.zip
#
# What changed from RL v1 (see WORKLOG 2026-10-08):
#   1. --residual-window 25    residual acts only with an obstacle < 25 m ahead,
#                              and training skips the agent past other steps
#   2. --authority-horizon 10  gate rolls the action 0.5 s forward, so it sees
#                              throttle; corridor bound added; steering sign fixed
#      counterfactual reward   same rollout, scores progress + safety margins
#   3. --obs-version v2        no absolute s / step count / remaining distance;
#                              relative obstacle positions and speeds
#   4. outcomes.csv + checkpoints with VecNormalize, for the learning curve
#   and --residual-max 0.5, allowed now that the gate covers throttle and road.
#
# Requires a CARLA 0.9.16 server already running.  Run ./tools/smoke_test.sh
# first -- it exercises exactly these flags in ~10 minutes.

set -uo pipefail
cd "$(dirname "$0")/.."

LABEL="sac_v2"
# Agent steps, not simulator ticks: with the window, each agent step is one
# decision near an obstacle, and many more ticks pass in between.  outcomes.csv
# records both.  150k is sized for one night; extend if the curve is still rising.
TIMESTEPS=150000
TRAIN_TOWN="Town01"
EVAL_TOWNS=("Town01" "Town02" "Town03")
EVAL="--seeds 1 2 3 4 5 --episodes 30"
FLAGS="--residual-mode adaptive --residual-max 0.5 --authority-horizon 10 \
--residual-window 25 --obs-version v2 --route-max 150 --qc 0.5 --gate-depth 0.98"
# The nominal baselines this run is compared against (2026-10-06 replication).
BASELINE_TAG="1006"

EVAL_ONLY=0
[ "${1:-}" = "--eval-only" ] && EVAL_ONLY=1
mkdir -p results models
START=$(date +%s)

if [ "$EVAL_ONLY" -eq 0 ]; then
  if [ -f "models/${LABEL}/final.zip" ]; then
    echo "models/${LABEL}/final.zip exists -- move it aside or use --eval-only"; exit 1
  fi
  echo ">>> TRAIN ${LABEL}: ${TIMESTEPS} agent steps on ${TRAIN_TOWN}"
  # shellcheck disable=SC2086
  python tools/train_residual.py --label "$LABEL" --algo sac \
    --timesteps "$TIMESTEPS" --town "$TRAIN_TOWN" $FLAGS \
    2>&1 | tee "results/train_${LABEL}.log"
  [ -f "models/${LABEL}/final.zip" ] || { echo "!!! training produced no model"; exit 1; }
fi

for T in "${EVAL_TOWNS[@]}"; do
  echo; echo ">>> EVAL ${LABEL} on ${T}   (elapsed $((($(date +%s)-START)/60)) min)"
  # shellcheck disable=SC2086
  python tools/benchmark_mpcc.py --label "${LABEL}_${T}" --town "$T" \
    --model "models/${LABEL}/final.zip" --algo sac $EVAL $FLAGS \
    2>&1 | tee "results/${LABEL}_${T}.log"
  B0="results/b0_${T}_${BASELINE_TAG}.json"
  if [ -f "$B0" ] && [ -f "results/${LABEL}_${T}.json" ]; then
    python tools/benchmark_mpcc.py --compare "$B0" "results/${LABEL}_${T}.json" \
      | tee "results/compare_${LABEL}_${T}.txt"
  else
    echo "  (no ${B0} to compare against)"
  fi
done

echo; echo "finished in $((($(date +%s)-START)/60)) min"
echo "learning curve : models/${LABEL}/outcomes.csv"
echo "checkpoints    : models/${LABEL}/${LABEL}_<N>_steps.zip (+ vecnormalize)"
