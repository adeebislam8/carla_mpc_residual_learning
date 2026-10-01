#!/usr/bin/env bash
# Paired comparison: ground-truth obstacles vs the pretrained-YOLO perception
# pipeline, same controller, same seeds (so both arms drive the identical
# spawn/goal/NPC sequence -- CarlaMPCEnv's per-episode RNG is seeded from
# (seed, episode_count) alone, not from anything perception touches).
#
# WHY THIS RUN: a 5-episode smoke test (2026-10-01, Town01, route-max 150)
# showed success fall from the known ground-truth baseline (81.67% at this
# exact config) to 40%, with a mechanism shift, not just a rate shift:
#   - 100% of the 3 collisions were OUTSIDE the corridor (ground truth has a
#     substantial INSIDE-corridor share -- see mpcc-overtake-attractor)
#   - mean max |n| 4.86 m, two episodes hit 7.67 m / 8.35 m -- far bigger
#     excursions than ground truth produces
#   - 2 of the 3 collisions logged ZERO completed overtakes yet still show
#     the largest |n| swings, suggesting the overtake-gate gets triggered by
#     flickering per-step detections (debug_perception.py showed a single
#     real car's detection count flip 1/1/2/1/0/1/0... frame to frame) and
#     swerves without ever finishing the pass the overtakes counter credits.
# n=5/1 seed is too small to trust the magnitude -- this run is to find out
# whether it replicates at a defensible sample size.
#
# PREDICTIONS, recorded before running:
#   - collision rate: perception > ground truth, and the gap is large enough
#     to be significant even at this sample size (it would need to be; the
#     smoke test gap was ~3x)
#   - outside-corridor share of collisions: perception > ground truth
#   - mean max |n|: perception > ground truth
#   - static-object / left-side collision signature: present in BOTH arms
#     (same underlying attractor mechanism) but larger in perception
#   - solver failure rate: no meaningful difference (the smoke test showed
#     0% in both -- this is a trajectory problem, not a numerical one)
#   FALSIFIED (for the "noisy detection destabilizes the attractor" story) if
#   perception's outside-corridor share and mean |n| are NOT both higher than
#   ground truth's -- that would mean perception is just noisier overall
#   rather than specifically worse at the attractor's lateral swerve.
#
# Usage:  ./tools/sweep_perception.sh [--dry] [--episodes N] [--seeds "1 2 .."] [--town T]
set -uo pipefail
cd "$(dirname "$0")/.."

TOWN="Town01"
EPISODES=30
SEEDS="1 2 3 4 5 6 7 8 9 10"     # x30 episodes = 300 per arm
CONTROLLER="--qc 0.5 --gate-depth 0.98 --route-max 150"
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

# Ground truth first: it's the fast, already-trusted arm, so if the run gets
# cut short overnight we still have a fresh same-seed reference to pair
# against whatever of the perception arm completed.
CELLS=(
  "gt|"
  "perception|--perception"
)

mkdir -p results
START=$(date +%s)
FAILED=(); JSONS=()

N_EPS=$(( $(echo $SEEDS | wc -w) * EPISODES ))
echo "=================================================================="
echo "perception vs ground-truth sweep   ${TOWN}"
echo "  seeds ${SEEDS} x ${EPISODES} episodes = ${N_EPS} per arm"
echo "  ground truth arm is fast; perception arm is CPU-YOLO-bound --"
echo "  smoke test averaged ~20 s/episode there, so budget ~$(( N_EPS * 20 / 60 )) min"
echo "  for the perception arm alone"
echo "=================================================================="

for cell in "${CELLS[@]}"; do
  name="${cell%%|*}"; flags="${cell#*|}"
  label="percep_${name}_${TOWN}"
  echo; echo ">>> ${label}   ${flags:-(ground truth)}   elapsed $((($(date +%s)-START)/60))m"
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
  python tools/benchmark_mpcc.py --compare "${JSONS[@]}" | tee "results/percep_${TOWN}.txt"
  echo
  echo "Headline table above is the aggregate numbers only. The collision"
  echo "BREAKDOWN (inside/outside corridor, max |n|, static-object share, by"
  echo "side) is per-arm in results/percep_gt_${TOWN}.txt and"
  echo "results/percep_perception_${TOWN}.txt -- diff those for the"
  echo "mechanism-shift predictions above, the compare table above only has"
  echo "the aggregate numbers."
fi

echo
echo "finished in $((($(date +%s)-START)/60)) min"
[ "${#FAILED[@]}" -gt 0 ] && { echo "FAILED:"; printf '   %s\n' "${FAILED[@]}"; }
echo "  per-arm reports: results/percep_*_${TOWN}.txt"
