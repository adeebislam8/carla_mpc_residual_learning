#!/usr/bin/env bash
# Parameter sweep for the nominal MPCC.
#
# Edit the CONFIGS block below, run, and it benchmarks every configuration in
# turn and prints one comparison table at the end.  Each config is a label
# followed by whatever flags benchmark_mpcc.py accepts, so anything exposed
# there can be swept without touching Python.
#
#   ./tools/sweep.sh                 # run the sweep
#   ./tools/sweep.sh --dry-run       # print what it would do, run nothing
#   ./tools/sweep.sh --compare-only  # re-print the table from existing results
#
# Requires a CARLA server already running.  Roughly 13 min per config at the
# default 3 seeds x 20 episodes, so a 4-config sweep is about an hour.

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------- CONFIGS ---
# "label|extra flags"   -- the label names results/<label>.{json,txt}
# Keep a no-flag baseline first so every sweep has its own reference point
# measured in the same session, rather than compared against an older run.
CONFIGS=(
  "qc_005|"
  "qc_010|--qc 0.10"
  "qc_020|--qc 0.20"
  "qc_050|--qc 0.50"
)

# Applied to every config.  Same seeds and town for all of them, or the
# comparison is meaningless.
COMMON="--seeds 1 2 3 4 5 --episodes 30"

# Set to 1 to keep each config's solver diagnostics in diagnostics_<label>/
KEEP_DIAGNOSTICS=1
# -----------------------------------------------------------------------------

DRY=0; COMPARE_ONLY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --compare-only) COMPARE_ONLY=1 ;;
    *) echo "unknown option: $a"; exit 1 ;;
  esac
done

LABELS=()
for cfg in "${CONFIGS[@]}"; do LABELS+=("${cfg%%|*}"); done

if [ "$COMPARE_ONLY" -eq 0 ]; then
  echo "=================================================================="
  echo "SWEEP: ${#CONFIGS[@]} configs x ${COMMON}"
  echo "=================================================================="
  START=$(date +%s)

  for cfg in "${CONFIGS[@]}"; do
    label="${cfg%%|*}"
    flags="${cfg#*|}"

    echo
    echo "------------------------------------------------------------------"
    echo ">>> ${label}   ${flags:-(defaults)}"
    echo "------------------------------------------------------------------"
    [ "$DRY" -eq 1 ] && { echo "    (dry run)"; continue; }

    # Fresh diagnostics dir per config so runs are not mixed together.
    rm -rf diagnostics
    # shellcheck disable=SC2086
    python tools/benchmark_mpcc.py --label "$label" $COMMON $flags \
      2>&1 | tee "results/${label}.log"

    if [ "$KEEP_DIAGNOSTICS" -eq 1 ] && [ -d diagnostics ]; then
      rm -rf "diagnostics_${label}"
      mv diagnostics "diagnostics_${label}"
      echo
      echo "--- solver diagnostics: ${label} ---"
      python tools/analyze_solver_failures.py "diagnostics_${label}/" --top 0 \
        2>&1 | sed -n '/FRENET CONVERTER/,/^$/p;/^FAILURES/p' || true
    fi
  done

  [ "$DRY" -eq 0 ] && \
    echo && echo "sweep finished in $((($(date +%s)-START)/60)) min"
fi

[ "$DRY" -eq 1 ] && exit 0

# ------------------------------------------------------------- COMPARISON ---
JSONS=()
for l in "${LABELS[@]}"; do
  [ -f "results/${l}.json" ] && JSONS+=("results/${l}.json")
done

if [ "${#JSONS[@]}" -lt 2 ]; then
  echo "need at least 2 completed configs to compare (found ${#JSONS[@]})"
  exit 0
fi

echo
python tools/benchmark_mpcc.py --compare "${JSONS[@]}"
echo "per-config reports: results/<label>.txt   logs: results/<label>.log"
