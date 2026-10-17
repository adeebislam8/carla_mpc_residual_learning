#!/usr/bin/env bash
# Overnight queue: run several (arm, set) jobs ONE AFTER ANOTHER (no parallel
# CARLA), then score everything into one report.  Foreground.
#
#   ./tools/b2d/overnight.sh                     # default queue below, NAME=n1
#   NAME=n2 JOBS="sbase:smoke srules:smoke" ./tools/b2d/overnight.sh
#   NAME=n1 REPORT_ONLY=1 ./tools/b2d/overnight.sh   # re-make the report (e.g. after Ctrl-C)
#
# Everything lands in results/b2d/<NAME>/<arm>_<set>/ plus
# results/b2d/<NAME>/report.txt -- copy that one folder.  Re-running the same
# NAME resumes: finished jobs are skipped (eval.json resume), so after a crash
# or Ctrl-C just run the same command again.
#
# Default queue, most important first (each line: why):
#   sbase:tlights        red hold + 1 s green release, clean (no parallel solver race)
#   srules:tlights       clean srules vs sbase on lights / junction lanes
#   sbase:smoke          clean smoke baseline (priority on = default)
#   sbase_noprio:smoke   clean obstacle_priority ablation
#   sbase_face:smoke     clean face-points test (priority + face)
#   srules:smoke         srules on overtaking
#   sbase:abl44          first look at every scenario type (E3 subset baseline)
#   srules:abl44         same for srules
# Rough times on the A5000: tlights ~1-1.5 h, smoke ~0.5-1 h, abl44 ~2.5-3.5 h.

set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
NAME=${NAME:-n1}
JOBS=${JOBS:-"sbase:tlights srules:tlights sbase:smoke sbase_noprio:smoke sbase_face:smoke srules:smoke sbase:abl44 srules:abl44"}
TL_TYPES="VanillaSignalizedTurnEncounterRedLight VanillaSignalizedTurnEncounterGreenLight VanillaNonSignalizedTurnEncounterStopsign OppositeVehicleRunningRedLight"
B2D=${B2D:-$HOME/Documents/nett/Bench2Drive}
ROOT=results/b2d/$NAME
REPORT=$ROOT/report.txt
mkdir -p "$ROOT"

config_of() {
  case "$1" in
    base)   echo tools/b2d/config_mpcc.json ;;
    rules)  echo tools/b2d/config_rules.json ;;
    *)      [ -f "tools/b2d/config_$1.json" ] && echo "tools/b2d/config_$1.json" \
              || { echo "unknown arm $1" >&2; return 1; } ;;
  esac
}

# --- preflight: refuse to run stale code (the per-process solver dir fix)
if [ "${REPORT_ONLY:-0}" != "1" ]; then
  grep -q getpid src/mpc_controller/src/acados_mpc/acados_settings_mpcc.py \
    || { echo "acados_settings_mpcc.py has no per-process export dir -- copy the new code first"; exit 1; }
  grep -q LIGHT_GREEN_S tools/b2d/mpcc_b2d_agent.py \
    || { echo "mpcc_b2d_agent.py is old (no LIGHT_GREEN_S) -- copy the new code first"; exit 1; }
  for J in $JOBS; do config_of "${J%%:*}" > /dev/null || exit 1; done
  pkill -9 -f run_b2d.sh 2>/dev/null; pkill -9 -f leaderboard_evaluator 2>/dev/null
  pkill -9 -f CarlaUE4 2>/dev/null; sleep 5
  rm -rf "$B2D/c_generated_code" "$B2D/acados_ocp.json"     # the old shared solver
  START=$(date +%s)
  echo "== overnight $NAME started $(date)  jobs: $JOBS" | tee "$ROOT/progress.txt"
  for J in $JOBS; do
    ARM=${J%%:*}; SET=${J##*:}
    CFG=$(config_of "$ARM")
    T0=$(date +%s)
    echo "== [$(date +%H:%M)] $ARM on $SET ($CFG)" | tee -a "$ROOT/progress.txt"
    # TAG with a slash: run_b2d.sh writes results/b2d/$NAME/${ARM}_${SET}
    TYPES="$TL_TYPES" TAG="$NAME/$ARM" CONFIG="$CFG" ./tools/b2d/run_b2d.sh "$SET"
    pkill -9 -f CarlaUE4 2>/dev/null; sleep 10
    echo "   done in $(( ($(date +%s) - T0) / 60 )) min" | tee -a "$ROOT/progress.txt"
  done
  echo "== all jobs done in $(( ($(date +%s) - START) / 60 )) min, $(date)" | tee -a "$ROOT/progress.txt"
fi

# --- report
{
  echo "######## overnight $NAME  ($(date))"
  cat "$ROOT/progress.txt" 2>/dev/null
  SETS=$(for J in $JOBS; do echo "${J##*:}"; done | awk '!seen[$0]++')
  for SET in $SETS; do
    echo; echo "================================================ $SET"
    python3 tools/b2d/compare_arms.py --set "$SET" --results "$ROOT" --csv "$ROOT/compare_${SET}.csv"
    for J in $JOBS; do
      ARM=${J%%:*}; [ "${J##*:}" = "$SET" ] || continue
      RUN=$ROOT/${ARM}_${SET}
      [ -f "$RUN/eval.json" ] || { echo; echo "---- $ARM $SET: no results"; continue; }
      echo; echo "---- $ARM $SET: score"
      TAG="$NAME/$ARM" ./tools/b2d/run_b2d.sh score "$SET"
      echo; echo "---- $ARM $SET: perception"
      python3 tools/b2d/check_perception.py "$RUN"
      echo; echo "---- $ARM $SET: timing"
      python3 tools/b2d/timing_b2d.py "$RUN"
      echo; echo "---- $ARM $SET: crashes in run.log"
      grep -c "agent has crashed" "$RUN/run.log" 2>/dev/null || true
    done
  done
} 2>&1 | tee "$REPORT"
echo; echo "== report: $REPORT   copy the whole folder: $ROOT"
