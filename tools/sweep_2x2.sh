#!/usr/bin/env bash
# 2x2: lateral-acceleration slack  x  junction corridor margin, on Town01.
#
# WHY A FACTORIAL AND NOT A SWEEP.  Town01's 95 baseline collisions split almost
# evenly into two failure modes that trade against each other:
#
#   51.6% OUTSIDE the corridor  -- tracking failure.  alat_max=8 is softened at
#       (1e-3, 1e-3) while the corridor n is at (2e0, 1e3), so violating the
#       lateral-acceleration limit is ~1e6 times cheaper than deviating.  Facing
#       a corner tighter than the tyres allow, the optimiser plans the
#       impossible corner every time.  Mean speed at impact is 13.39 m/s against
#       a mean driving speed of 10.19; a 10 m intersection turn at 13.39 m/s
#       demands 17.9 m/s^2, i.e. 2.24x the limit.
#
#   48.4% INSIDE the corridor   -- the corridor itself contains the furniture.
#       JUNCTION_MARGIN=4.0 grants 4 m of extra left width at junctions, and
#       traffic lights alone are 14.7% of Town01 collisions.
#
# Fixing either alone moves collisions from one mode into the other, which is
# why thirteen single-variable configs all landed in one noise band and why
# JUNCTION_MARGIN=1.5 looked like a failure (65% -> 75%) when tried by itself.
# Only the factorial separates the main effects from the interaction.
#
# PREDICTIONS, recorded before the run:
#   alat alone   -> mean + impact speed fall, OUTSIDE-corridor collisions drop,
#                   timeouts rise
#   margin alone -> reproduces the ~75% regression
#   both         -> INSIDE-corridor collisions drop AND outside stays down
#   If the alat cells do not reduce outside-corridor collisions, the
#   infeasibility hypothesis is wrong and should be dropped.
#
# Usage:  ./tools/sweep_2x2.sh [--dry] [--episodes N] [--seeds "1 2 3"] [--town T]
set -uo pipefail
cd "$(dirname "$0")/.."

TOWN="Town01"
EPISODES=50            # x3 seeds = 150/cell, matching the b0 runs
SEEDS="1 2 3"
CONTROLLER="--qc 0.5 --gate-depth 0.98"
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry)      DRY=1; shift ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --seeds)    SEEDS="$2"; shift 2 ;;
    --town)     TOWN="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

# name | extra flags.  Cell A is the established baseline -- do not change its
# flags without re-measuring, it is the only anchor to the 63.33% number.
CELLS=(
  "A_base|"
  "B_alat|--alat-slack 1e3"
  "C_margin|--junction-margin 1.5"
  "D_both|--alat-slack 1e3 --junction-margin 1.5"
)

mkdir -p results
START=$(date +%s)
FAILED=()
JSONS=()

echo "=================================================================="
echo "2x2  alat slack x junction margin   ${TOWN}"
echo "  seeds ${SEEDS} x ${EPISODES} episodes = $(( $(echo $SEEDS | wc -w) * EPISODES )) per cell"
echo "=================================================================="

for cell in "${CELLS[@]}"; do
  name="${cell%%|*}"; flags="${cell#*|}"
  label="x22_${name}_${TOWN}"
  echo; echo ">>> ${label}   ${flags:-(baseline)}    elapsed $((($(date +%s)-START)/60))m"
  if [ "$DRY" -eq 1 ]; then
    echo "    (dry) python tools/benchmark_mpcc.py --label $label --town $TOWN --seeds $SEEDS --episodes $EPISODES $CONTROLLER $flags"
    continue
  fi
  # shellcheck disable=SC2086
  if python tools/benchmark_mpcc.py --label "$label" --town "$TOWN" \
       --seeds $SEEDS --episodes "$EPISODES" $CONTROLLER $flags \
       2>&1 | tee "results/${label}.log"; then
    [ -f "results/${label}.json" ] && JSONS+=("results/${label}.json")
  else
    echo "!!! FAILED: ${label} -- continuing"; FAILED+=("$label")
  fi
done

if [ "$DRY" -eq 0 ] && [ "${#JSONS[@]}" -ge 2 ]; then
  echo; echo "--- 2x2 COMPARISON ---"
  python tools/benchmark_mpcc.py --compare "${JSONS[@]}" | tee "results/x22_${TOWN}.txt"
  echo; echo "--- FACTORIAL ANALYSIS (main effects + interaction) ---"
  python tools/analyze_2x2.py "${JSONS[@]}" | tee -a "results/x22_${TOWN}.txt"
fi

echo
echo "finished in $((($(date +%s)-START)/60)) min"
[ "${#FAILED[@]}" -gt 0 ] && { echo "FAILED:"; printf '   %s\n' "${FAILED[@]}"; }
echo "  per-cell reports: results/x22_*_${TOWN}.txt"
