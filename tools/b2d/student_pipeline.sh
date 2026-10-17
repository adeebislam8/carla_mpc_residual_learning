#!/usr/bin/env bash
# Tactical-student pipeline on Bench2Drive, one command, foreground, resumable.
# Runs in the SENSOR-ONLY setting (config_record / config_tactical have
# privileged: false): the student's scene comes from LiDAR tracks, the agent
# from GNSS/IMU, camera lights -- the same inputs class as UniAD / VAD / ORION.
#
#   ./tools/b2d/student_pipeline.sh            # all stages
#   STAGES="label build train" ./tools/b2d/student_pipeline.sh
#
# Stages:
#   clips  cut single-scenario TRAINING clips out of Bench2Drive's long training
#          routes (disjoint from the 220 evaluation routes)   [b2d env, no CARLA]
#   rec    our rules-arm agent drives them; ORION's 6 cameras + scene recorded
#          -> results/b2d/rec_train_clips/rec/<route>/          [b2d env + CARLA]
#   label  ORION labels every recorded route offline           [orion env, GPU]
#          (TEACHERS="orion simlingo": SimLingo too            [simlingo env, GPU])
#   build  features + go/slow labels -> data/tactical_r0.npz (rule-checked labels)
#          and data/tactical_r0_raw.npz (raw teacher, the ablation); per teacher
#   train  ensemble -> models/tactical_r0 (+ _raw)              [orion env (torch)]
#   eval   rules arm + tactical student on EVAL_SET (default: all 220 official
#          routes), then every arm run on that set side by side, per scenario
#          type (tools/b2d/compare_arms.py)                      [b2d env + CARLA]
#
# CARLA (rec, eval) and ORION (label) both need the GPU, so they never overlap.
# Size (defaults): ~100 clips, ~90k frames: ~4-6 h recording, ~12 h ORION
# labelling, ~30-35 GB of JPEGs until labelled.  PRUNE=1 deletes a route's
# JPEGs once its labels are written (meta/obs are all the student needs).

set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
NETT=${NETT:-$HOME/Documents/nett}
export CARLA_ROOT=${CARLA15:-$NETT/CARLA_0.9.15}
B2D=${B2D:-$NETT/Bench2Drive}
ORION_ROOT=${ORION_ROOT:-$NETT/Orion}
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}
STAGES=${STAGES:-"clips rec label build train eval"}
PER_TYPE=${PER_TYPE:-3}          # Emergency_Brake types (+ VehicleOpensDoorTwoWays)
TWOWAYS=${TWOWAYS:-10}           # the student's target: TwoWays types
ONEWAY=${ONEWAY:-6}              # one-way Overtaking types (pass cases without oncoming)
PRUNE=${PRUNE:-0}
TEACHERS=${TEACHERS:-orion}      # orion | "orion simlingo" (teacher ablation)
SIMLINGO_ROOT=${SIMLINGO_ROOT:-$NETT/simlingo}
EVAL_SET=${EVAL_SET:-full}       # full = all 220 routes; ot_eb = the 105-route subset
CLIPS=tools/b2d/routes/train_clips.xml
RUN=results/b2d/rec_train_clips
DATA=data/tactical_r0.npz
MODEL=models/tactical_r0
has() { [[ " $STAGES " == *" $1 "* ]]; }
say() { echo; echo "[$(date +%H:%M:%S)] == $*"; }

if has clips && [ ! -f "$CLIPS" ]; then
  say "cutting training clips"
  PYTHONNOUSERSITE=1 PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$B2D/leaderboard" \
    "$CONDA" run --no-capture-output -n b2d python tools/b2d/make_training_clips.py \
    --b2d "$B2D" --carla-root "$CARLA_ROOT" --per-type "$PER_TYPE" \
    --extra-per-type AccidentTwoWays=$TWOWAYS ConstructionObstacleTwoWays=$TWOWAYS \
                     ParkedObstacleTwoWays=$TWOWAYS HazardAtSideLaneTwoWays=$TWOWAYS \
                     Accident=$ONEWAY ConstructionObstacle=$ONEWAY ParkedObstacle=$ONEWAY HazardAtSideLane=$ONEWAY \
    --out "$CLIPS" || exit 1
fi

if has rec; then
  say "recording (rules arm drives, ORION rig on)"
  TAG=rec CONFIG=tools/b2d/config_record.json ./tools/b2d/run_b2d.sh train_clips
fi

if has label; then
  say "ORION labelling"
  mkdir -p "$RUN/labels"
  todo=()
  for ep in "$RUN"/rec/*/; do
    ep=${ep%/}; n=$(basename "$ep")
    [ -f "$ep/meta.jsonl" ] || continue
    [ -f "$RUN/labels/$n.npz" ] || todo+=("$ep")
  done
  echo "   ${#todo[@]} routes to label"
  if [ ${#todo[@]} -gt 0 ]; then
    pkill -9 -f CarlaUE4 2>/dev/null; sleep 5     # free the GPU for ORION
    PYTHONNOUSERSITE=1 "$CONDA" run --no-capture-output -n orion python tools/orion/orion_infer.py \
      --orion-root "$ORION_ROOT" --jpeg-quality 0 --episode "${todo[@]}" --out "$RUN/labels"
  fi
  if [[ " $TEACHERS " == *" simlingo "* ]]; then
    say "SimLingo labelling"
    stodo=()
    for ep in "$RUN"/rec/*/; do
      ep=${ep%/}; n=$(basename "$ep")
      [ -f "$ep/meta.jsonl" ] || continue
      [ -f "$RUN/labels_simlingo/$n.npz" ] || stodo+=("$ep")
    done
    echo "   ${#stodo[@]} routes to label"
    if [ ${#stodo[@]} -gt 0 ]; then
      pkill -9 -f CarlaUE4 2>/dev/null; sleep 5
      PYTHONNOUSERSITE=1 "$CONDA" run --no-capture-output -n simlingo python tools/simlingo/simlingo_infer.py \
        --simlingo-root "$SIMLINGO_ROOT" --episode "${stodo[@]}" --out "$RUN/labels_simlingo"
    fi
  fi
  if [ "$PRUNE" = "1" ]; then
    for ep in "$RUN"/rec/*/; do
      n=$(basename "${ep%/}")
      [ -f "$RUN/labels/$n.npz" ] || continue
      [[ " $TEACHERS " == *" simlingo "* ]] && [ ! -f "$RUN/labels_simlingo/$n.npz" ] && continue
      rm -rf "$ep"/CAM_* "$ep"/SIMLINGO_FRONT && echo "   pruned $n"
    done
  fi
fi

if has build; then
  say "building the tactical dataset"
  mkdir -p data
  python3 tools/student/build_tactical.py --rec "$RUN/rec" --labels "$RUN/labels" --out "$DATA" || exit 1
  python3 tools/student/build_tactical.py --rec "$RUN/rec" --labels "$RUN/labels" --no-filter \
    --out "${DATA%.npz}_raw.npz" || exit 1
  if [[ " $TEACHERS " == *" simlingo "* ]]; then
    python3 tools/student/build_tactical.py --rec "$RUN/rec" --labels "$RUN/labels_simlingo" \
      --out data/tactical_simlingo.npz || exit 1
  fi
fi

if has train; then
  say "training the tactical student"
  PYTHONNOUSERSITE=1 "$CONDA" run --no-capture-output -n orion python tools/student/train_tactical.py \
    --data "$DATA" --out "$MODEL" || exit 1
  PYTHONNOUSERSITE=1 "$CONDA" run --no-capture-output -n orion python tools/student/train_tactical.py \
    --data "${DATA%.npz}_raw.npz" --out "${MODEL}_raw" || exit 1
  if [[ " $TEACHERS " == *" simlingo "* ]]; then
    PYTHONNOUSERSITE=1 "$CONDA" run --no-capture-output -n orion python tools/student/train_tactical.py \
      --data data/tactical_simlingo.npz --out models/tactical_simlingo || exit 1
  fi
fi

if has eval; then
  say "evaluating sensor-only rules arm + tactical student on $EVAL_SET"
  TAG=tactical CONFIG=tools/b2d/config_tactical.json ./tools/b2d/run_b2d.sh "$EVAL_SET"
  say "all arms on $EVAL_SET, side by side"
  python3 tools/b2d/compare_arms.py --set "$EVAL_SET" --b2d "$B2D" \
    --csv "results/b2d/compare_${EVAL_SET}.csv"
fi
say "done"
