#!/usr/bin/env bash
# Does the overtake attractor cause the collisions?
#
# n_overtake = -2.5 pulls the car 2.5 m LEFT (d > 0 is right) whenever an NPC
# sits in the gate window (-8 to +25 m).  It is obstacle-BLIND: it commits to
# that target with no check that the space is clear, and the CBF only guards
# the 6 modelled NPC slots.  Nothing in the formulation knows a pole is there.
#
# EVIDENCE (Town01, 150 episodes, cell A of the 2x2):
#   74% of collisions on straight road (|kappa| <= 0.02)
#   left impacts 3.00x right, p = 1.3e-03   -- and this attractor pulls LEFT
#   >= 40 of 100 collisions are STATIC objects hit from INSIDE the corridor
#   50% is roadside furniture (lights 16, poles 14, guardrail 13, fence 7)
#   impact speed 12.58 m/s vs mean 10.11    -- overtaking means accelerating
#   fallback 0%                             -- not a solver problem
#   collisions/km vs overtakes/km: r = +0.775 over 17 configs (t = +4.75)
#
# READ THIS ON collisions/km AND THE STATIC-OBJECT COUNT, NOT SUCCESS RATE.
# Removing the attractor removes overtaking, so episodes convert into timeouts
# and success will look flat or worse BY CONSTRUCTION.  That is the expected
# cost of the manipulation, not a refutation of it.
#
# PREDICTIONS, recorded before the run:
#   n=0.0   -> collisions/km falls; static-object share falls; LEFT-impact
#              share falls toward the right-impact share; overtakes ~ 0;
#              timeouts rise sharply
#   n=-1.25 -> intermediate on all of the above (a dose-response, which is
#              much harder to get by chance than a single contrast)
#   FALSIFIED if n=0.0 does not cut static-object collisions.  Two hypotheses
#   (dynamic infeasibility, corridor geometry) have already died here; this one
#   gets the same treatment.
#
# Usage:  ./tools/sweep_overtake.sh [--dry] [--episodes N] [--seeds "1 2 3"] [--town T]
set -uo pipefail
cd "$(dirname "$0")/.."

TOWN="Town01"
EPISODES=50            # x3 seeds = 150/cell, matching every other Town01 run
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

# Baseline LAST so a crash mid-sweep still leaves the two manipulations done:
# the baseline is already measured (cell A of the 2x2, 66.67% / 27.33%) and is
# the one cell we can afford to lose.
CELLS=(
  "off|--n-overtake 0.0"
  "half|--n-overtake -1.25"
  "base|"
)

mkdir -p results
START=$(date +%s)
FAILED=(); JSONS=()

echo "=================================================================="
echo "overtake-attractor sweep   ${TOWN}"
echo "  seeds ${SEEDS} x ${EPISODES} episodes = $(( $(echo $SEEDS | wc -w) * EPISODES )) per cell"
echo "  primary metric: collisions/km and static-object share, NOT success"
echo "=================================================================="

for cell in "${CELLS[@]}"; do
  name="${cell%%|*}"; flags="${cell#*|}"
  label="ovt_${name}_${TOWN}"
  echo; echo ">>> ${label}   ${flags:-(baseline, n_overtake=-2.5)}   elapsed $((($(date +%s)-START)/60))m"
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
  echo; echo "--- COMPARISON ---"
  python tools/benchmark_mpcc.py --compare "${JSONS[@]}" | tee "results/ovt_${TOWN}.txt"
  echo; echo "--- DOSE-RESPONSE ON THE ATTRIBUTABLE METRICS ---"
  python tools/analyze_overtake.py "${JSONS[@]}" | tee -a "results/ovt_${TOWN}.txt"
fi

echo
echo "finished in $((($(date +%s)-START)/60)) min"
[ "${#FAILED[@]}" -gt 0 ] && { echo "FAILED:"; printf '   %s\n' "${FAILED[@]}"; }
echo "  per-cell reports: results/ovt_*_${TOWN}.txt"
