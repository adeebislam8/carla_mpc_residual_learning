#!/usr/bin/env bash
# Run our MPCC agent on Bench2Drive routes (foreground).  Needs setup_b2d.sh.
#
#   ./tools/b2d/run_b2d.sh smoke            # 9 Overtaking routes, one per type
#   ./tools/b2d/run_b2d.sh ot_eb            # Overtaking + Emergency_Brake, 105 routes
#   ./tools/b2d/run_b2d.sh full             # all 220 official routes
#   CONFIG=tools/b2d/config_student.json TAG=student ./tools/b2d/run_b2d.sh ot_eb
#
# The leaderboard starts (and kills) its own CARLA 0.9.15 server -- stop any
# other CARLA first.  Results: results/b2d/<TAG>_<SET>/eval.json; it RESUMES
# from that file if re-run, so after a CARLA crash just run the same command.
# Then:  ./tools/b2d/run_b2d.sh score <SET>   (driving score, success, per-ability)

set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
NETT=${NETT:-$HOME/Documents/nett}
export CARLA_ROOT=${CARLA15:-$NETT/CARLA_0.9.15}
B2D=${B2D:-$NETT/Bench2Drive}
SET=${1:-smoke}
TAG=${TAG:-mpcc}
CONFIG=$(realpath "${CONFIG:-$REPO/tools/b2d/config_mpcc.json}")   # absolute: the run cd's away
PORT=${PORT:-30000}
TM_PORT=${TM_PORT:-50000}
GPU_RANK=${GPU_RANK:-0}
OUT=$REPO/results/b2d/${TAG}_${SET}
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}
PY="$CONDA run --no-capture-output -n b2d python"

ROUTES=$REPO/tools/b2d/routes/${SET}.xml

if [ "$SET" = "score" ]; then
  S2=${2:?score needs the set name}
  RUN=$REPO/results/b2d/${TAG}_${S2}
  R=$REPO/tools/b2d/routes/${S2}.xml
  [ "$S2" = "full" ] && R=$B2D/leaderboard/data/bench2drive220.xml
  # merge_route_json.py divides by 220; for subsets score over the subset.
  python3 "$REPO/tools/b2d/summarize_b2d.py" --routes "$R" "$RUN/eval.json"
  exit 0
fi

mkdir -p "$OUT" "$REPO/tools/b2d/routes"     # not for 'score' (made empty *_score dirs)
case "$SET" in
  smoke) [ -f "$ROUTES" ] || python3 "$REPO/tools/b2d/make_subset.py" --b2d "$B2D" \
           --ability Overtaking --per-type 1 --out "$ROUTES" ;;
  ot_eb) [ -f "$ROUTES" ] || python3 "$REPO/tools/b2d/make_subset.py" --b2d "$B2D" \
           --ability Overtaking Emergency_Brake --out "$ROUTES" ;;
  full)  ROUTES=$B2D/leaderboard/data/bench2drive220.xml ;;
  *)     # any other name: the routes of the scenario types in TYPES, e.g.
         #   TYPES="HardBreakRoute" TAG=dbg ./tools/b2d/run_b2d.sh hb
         [ -n "${TYPES:-}" ] || { echo "unknown set $SET (set TYPES=... for a custom set)"; exit 1; }
         [ -f "$ROUTES" ] || python3 "$REPO/tools/b2d/make_subset.py" --b2d "$B2D" \
           --types $TYPES --out "$ROUTES" ;;
esac

cd "$B2D"
export PYTHONPATH=$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$B2D/leaderboard:$B2D/scenario_runner:$REPO/src:${PYTHONPATH:-}
export SCENARIO_RUNNER_ROOT=$B2D/scenario_runner
export LEADERBOARD_ROOT=$B2D/leaderboard
export IS_BENCH2DRIVE=True
export PYTHONNOUSERSITE=1
echo "== $SET: $(grep -c '<route ' "$ROUTES") routes, agent config $CONFIG -> $OUT"
# The command goes in a script that 'conda run' executes: passed directly,
# conda parses the evaluator's own flags (--debug=0) as its options.
cat > "$OUT/eval_cmd.sh" <<EOS
cd "$B2D"
export PYTHONPATH="$PYTHONPATH" SCENARIO_RUNNER_ROOT="$SCENARIO_RUNNER_ROOT" \\
       LEADERBOARD_ROOT="$LEADERBOARD_ROOT" IS_BENCH2DRIVE=True PYTHONNOUSERSITE=1 \\
       CARLA_ROOT="$CARLA_ROOT" B2D_TRACE_DIR="$OUT/traces"
exec python "$B2D/leaderboard/leaderboard/leaderboard_evaluator.py" \\
  --routes="$ROUTES" --repetitions=1 --track=SENSORS \\
  --checkpoint="$OUT/eval.json" \\
  --agent="$REPO/tools/b2d/mpcc_b2d_agent.py" --agent-config="$CONFIG" \\
  --debug=0 --resume=True --port=$PORT --traffic-manager-port=$TM_PORT \\
  --gpu-rank=$GPU_RANK
EOS
# CARLA crashes on the large maps (segfault while loading Town12/13 is
# common); Bench2Drive's own scripts restart until every route has a result.
# The evaluator resumes from eval.json, so each retry continues where the last
# stopped; a route that keeps crashing is eventually recorded as failed.
MAX_TRY=${MAX_TRY:-60}      # restarts; also stops after 2 restarts without progress
LAST=-1; STALL=0
progress() {
  python3 -c "
import json, sys
try:
    p = json.load(open('$OUT/eval.json'))['_checkpoint'].get('progress') or [0, 1]
except Exception:
    p = [0, 1]
print(p[0], p[1])"
}
for TRY in $(seq 1 "$MAX_TRY"); do
  "$CONDA" run --no-capture-output -n b2d bash "$OUT/eval_cmd.sh" 2>&1 | tee -a "$OUT/run.log"
  read DONE TOTAL < <(progress)
  echo "== attempt $TRY: $DONE / $TOTAL routes recorded"
  [ "$DONE" -ge "$TOTAL" ] && [ "$TOTAL" -gt 1 ] && break
  if [ "$DONE" -le "$LAST" ]; then STALL=$((STALL + 1)); else STALL=0; fi
  LAST=$DONE
  [ "$STALL" -ge 2 ] && { echo "== no progress in 2 restarts -- stopping; check run.log"; break; }
  echo "== restarting (CARLA crash?) in 15 s ..."
  # Only OUR server (by its RPC port): parallel arms run their own CARLA.
  pkill -9 -f "carla-rpc-port=$PORT" 2>/dev/null; sleep 15
done
echo "== done.  Score: ./tools/b2d/run_b2d.sh score $SET"
