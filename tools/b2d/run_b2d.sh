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
CONFIG=${CONFIG:-$REPO/tools/b2d/config_mpcc.json}
PORT=${PORT:-30000}
TM_PORT=${TM_PORT:-50000}
GPU_RANK=${GPU_RANK:-0}
OUT=$REPO/results/b2d/${TAG}_${SET}
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}
PY="$CONDA run --no-capture-output -n b2d python"

mkdir -p "$OUT" "$REPO/tools/b2d/routes"
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

case "$SET" in
  smoke) [ -f "$ROUTES" ] || python3 "$REPO/tools/b2d/make_subset.py" --b2d "$B2D" \
           --ability Overtaking --per-type 1 --out "$ROUTES" ;;
  ot_eb) [ -f "$ROUTES" ] || python3 "$REPO/tools/b2d/make_subset.py" --b2d "$B2D" \
           --ability Overtaking Emergency_Brake --out "$ROUTES" ;;
  full)  ROUTES=$B2D/leaderboard/data/bench2drive220.xml ;;
  *)     echo "unknown set $SET"; exit 1 ;;
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
       CARLA_ROOT="$CARLA_ROOT"
exec python "$B2D/leaderboard/leaderboard/leaderboard_evaluator.py" \\
  --routes="$ROUTES" --repetitions=1 --track=SENSORS \\
  --checkpoint="$OUT/eval.json" \\
  --agent="$REPO/tools/b2d/mpcc_b2d_agent.py" --agent-config="$CONFIG" \\
  --debug=0 --resume=True --port=$PORT --traffic-manager-port=$TM_PORT \\
  --gpu-rank=$GPU_RANK
EOS
"$CONDA" run --no-capture-output -n b2d bash "$OUT/eval_cmd.sh" 2>&1 | tee -a "$OUT/run.log"
echo "== done.  Score: ./tools/b2d/run_b2d.sh score $SET"
