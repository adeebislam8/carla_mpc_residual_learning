#!/usr/bin/env bash
# Tactical-student pipeline on Bench2Drive (v2, 2026-10-18), foreground, resumable.
# SENSOR-ONLY setting: the student's scene comes from LiDAR tracks, the agent
# from GNSS/IMU, camera lights -- the same input class as UniAD / VAD / ORION.
# Student v2 heads: go (pass/wait), yield (wait at a junction entry), dv (speed).
#
#   ./tools/b2d/student_pipeline.sh                         # round 0, all stages
#   STAGES="label experts build train agree" ./tools/b2d/student_pipeline.sh
#   ROUND=1 ./tools/b2d/student_pipeline.sh                 # DAgger round
#
# Round 0 (the sbase arm drives, so states match the sbase + student arm):
#   clips    cut single-scenario TRAINING clips out of Bench2Drive's long training
#            routes (disjoint from the 220 evaluation routes)    [b2d env, no CARLA]
#   rec      sbase drives them; ORION's 6 cameras + SimLingo camera + scene +
#            simulator truth ('gt', labels only) -> results/b2d/rec_train_clips2/rec/
#   label    ORION and SimLingo label every frame offline        [GPU, no CARLA]
#   experts  privileged expert (truth) + rule expert (perceived) labels   [CPU]
#   build    one dataset per teacher: orion, simlingo, both (ORION+SimLingo,
#            COMBINE=soft|agree|safe), privileged, rules -> data/tactical_<t>.npz
#   train    one student per teacher -> models/tactical_<t>        [torch, minutes]
#   agree    teacher agreement table -> results/b2d/teacher_agreement.csv
# Round 1 (ROUND=1, DAgger): the round-0 student (DAGGER_DRIVER, default
#   'simlingo') on the FINAL base drives the same clips with the light camera rig
#   (front + SimLingo only; config_record_dagger.json record_orion false)
#   -> results/b2d/rec_r1_train_clips2/rec/; SimLingo + the privileged / rule
#   experts label ITS states; build = round-0 + round-1 -> models/tactical_<t>_r1
#   (orion / both stay round-0 only).
#
# Evaluate afterwards with overnight.sh, e.g.
#   NAME=stu JOBS="stu_both:abl44 stu_orion:abl44 stu_simlingo:abl44 stu_privileged:abl44" \
#     ./tools/b2d/overnight.sh
#
# CARLA (rec) and the teachers (label) both need the GPU, so they never overlap.
# Size (defaults): ~100 clips, ~90k frames: ~4-6 h recording, ~12 h ORION +
# a few h SimLingo labelling, ~35 GB of JPEGs until labelled.  PRUNE=1 deletes
# a route's JPEGs once every image teacher has labelled it.

set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
NETT=${NETT:-$HOME/Documents/nett}
export CARLA_ROOT=${CARLA15:-$NETT/CARLA_0.9.15}
B2D=${B2D:-$NETT/Bench2Drive}
ORION_ROOT=${ORION_ROOT:-$NETT/Orion}
SIMLINGO_ROOT=${SIMLINGO_ROOT:-$NETT/simlingo}
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}
STAGES=${STAGES:-"clips rec label experts build train agree"}
ROUND=${ROUND:-0}
PER_TYPE=${PER_TYPE:-3}          # Emergency_Brake types (+ VehicleOpensDoorTwoWays)
TWOWAYS=${TWOWAYS:-10}           # the go head's target: TwoWays types
ONEWAY=${ONEWAY:-6}              # one-way Overtaking types (pass cases without oncoming)
JUNCTION=${JUNCTION:-6}          # the yield head's target: junction gap-acceptance types
PRUNE=${PRUNE:-0}
COMBINE=${COMBINE:-soft}
STRIDE=${STRIDE:-1}             # label every STRIDE-th frame (2 = 10 Hz: half the teacher GPU time)
DAGGER_DRIVER=${DAGGER_DRIVER:-simlingo}
# Round 1 (DAgger) records the LIGHT rig (no ORION cameras): ORION and the
# ORION+SimLingo combination stay round-0 only (teacher comparison at round 0,
# DAgger for the selected teacher + the camera-free experts).
if [ "${ROUND:-0}" = "1" ]; then TEACHERS="simlingo privileged rules"
else TEACHERS="orion simlingo both privileged rules"; fi
# v2 clip set (adds junction gap-acceptance types for the yield head); a new
# name so an old train_clips.xml is never reused
CLIPS=tools/b2d/routes/train_clips2.xml
R0=results/b2d/rec_train_clips2
R1=results/b2d/rec_r1_train_clips2
if [ "$ROUND" = "1" ]; then RUN=$R1; SUF=_r1; else RUN=$R0; SUF=; fi
PY="$CONDA run --no-capture-output -n"
has() { [[ " $STAGES " == *" $1 "* ]]; }
say() { echo; echo "[$(date +%H:%M:%S)] == $*"; }

if has clips && [ ! -f "$CLIPS" ]; then
  say "cutting training clips"
  PYTHONNOUSERSITE=1 PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$B2D/leaderboard" \
    $PY b2d python tools/b2d/make_training_clips.py \
    --b2d "$B2D" --carla-root "$CARLA_ROOT" --per-type "$PER_TYPE" \
    --extra-per-type AccidentTwoWays=$TWOWAYS ConstructionObstacleTwoWays=$TWOWAYS \
                     ParkedObstacleTwoWays=$TWOWAYS HazardAtSideLaneTwoWays=$TWOWAYS \
                     Accident=$ONEWAY ConstructionObstacle=$ONEWAY ParkedObstacle=$ONEWAY HazardAtSideLane=$ONEWAY \
                     SignalizedJunctionLeftTurn=$JUNCTION NonSignalizedJunctionLeftTurn=$JUNCTION \
                     NonSignalizedJunctionRightTurn=$JUNCTION SignalizedJunctionRightTurn=$JUNCTION \
                     MergerIntoSlowTraffic=$JUNCTION InterurbanActorFlow=$JUNCTION \
                     EnterActorFlow=$JUNCTION CrossingBicycleFlow=$JUNCTION \
    --out "$CLIPS" || exit 1
fi

if has rec; then
  if [ "$ROUND" = "1" ]; then
    [ -d "models/tactical_${DAGGER_DRIVER}" ] || { echo "no models/tactical_${DAGGER_DRIVER}: train round 0 first"; exit 1; }
    say "DAgger recording: student '${DAGGER_DRIVER}' drives, ORION + SimLingo rigs on"
    python3 - "$DAGGER_DRIVER" <<'EOF'
import json, sys
c = json.load(open('tools/b2d/config_record_dagger.json'))
c['tactical'] = f'models/tactical_{sys.argv[1]}'
json.dump(c, open('tools/b2d/config_record_dagger.json', 'w'), indent=2)
EOF
    TAG=rec_r1 CONFIG=tools/b2d/config_record_dagger.json ./tools/b2d/run_b2d.sh train_clips2
  else
    say "recording (sbase drives, ORION + SimLingo rigs on)"
    TAG=rec CONFIG=tools/b2d/config_record.json ./tools/b2d/run_b2d.sh train_clips2
  fi
fi

todo_for() {    # episodes of $RUN without a label file in $1
  local out=$1; local -a t=()
  for ep in "$RUN"/rec/*/; do
    ep=${ep%/}; n=$(basename "$ep")
    [ -f "$ep/meta.jsonl" ] || continue
    [ -f "$out/$n.npz" ] || t+=("$ep")
  done
  printf '%s\n' "${t[@]}"
}

if has label; then
  mkdir -p "$RUN/labels" "$RUN/labels_simlingo"
  if [ "$ROUND" = "1" ]; then todo=(); say "ORION labelling: skipped (round 1: light rig)"
  else mapfile -t todo < <(todo_for "$RUN/labels"); say "ORION labelling: ${#todo[@]} routes"; fi
  if [ ${#todo[@]} -gt 0 ]; then
    pkill -9 -f CarlaUE4 2>/dev/null; sleep 5     # free the GPU
    PYTHONNOUSERSITE=1 $PY orion python tools/orion/orion_infer.py \
      --orion-root "$ORION_ROOT" --jpeg-quality 0 --stride "$STRIDE" --skip-wedged --episode "${todo[@]}" --out "$RUN/labels"
  fi
  mapfile -t todo < <(todo_for "$RUN/labels_simlingo")
  say "SimLingo labelling: ${#todo[@]} routes"
  if [ ${#todo[@]} -gt 0 ]; then
    pkill -9 -f CarlaUE4 2>/dev/null; sleep 5
    PYTHONNOUSERSITE=1 $PY simlingo python tools/simlingo/simlingo_infer.py \
      --simlingo-root "$SIMLINGO_ROOT" --stride "$STRIDE" --episode "${todo[@]}" --out "$RUN/labels_simlingo"
  fi
  if [ "$PRUNE" = "1" ]; then
    for ep in "$RUN"/rec/*/; do
      n=$(basename "${ep%/}")
      [ -f "$RUN/labels/$n.npz" ] && [ -f "$RUN/labels_simlingo/$n.npz" ] || continue
      rm -rf "$ep"/CAM_* "$ep"/SIMLINGO_FRONT && echo "   pruned $n"
    done
  fi
fi

if has experts; then
  say "expert teachers (privileged truth, perceived rules)"
  python3 tools/student/expert_teacher.py --mode privileged --stride "$STRIDE" --rec "$RUN/rec" --out "$RUN/labels_privileged" || exit 1
  python3 tools/student/expert_teacher.py --mode rules --stride "$STRIDE" --rec "$RUN/rec" --out "$RUN/labels_rules" || exit 1
fi

labels_dir() {  # teacher -> label dir name
  case "$1" in orion) echo labels ;; simlingo) echo labels_simlingo ;;
               privileged) echo labels_privileged ;; rules) echo labels_rules ;; esac
}

if has build; then
  mkdir -p data
  if [ "$ROUND" = "1" ]; then RECS="$R0/rec $R1/rec"; else RECS="$RUN/rec"; fi
  for t in $TEACHERS; do
    say "building dataset: $t (round $ROUND)"
    if [ "$t" = "both" ]; then
      L1=""; L2=""
      for r in $RECS; do L1="$L1 ${r%/rec}/labels"; L2="$L2 ${r%/rec}/labels_simlingo"; done
      python3 tools/student/build_tactical.py --rec $RECS --labels $L1 --labels2 $L2 \
        --combine "$COMBINE" --out "data/tactical_both${SUF}.npz" || exit 1
    else
      L=""; for r in $RECS; do L="$L ${r%/rec}/$(labels_dir "$t")"; done
      python3 tools/student/build_tactical.py --rec $RECS --labels $L \
        --out "data/tactical_${t}${SUF}.npz" || exit 1
    fi
  done
fi

if has train; then
  for t in $TEACHERS; do
    say "training student: $t (round $ROUND)"
    PYTHONNOUSERSITE=1 $PY orion python tools/student/train_tactical.py \
      --data "data/tactical_${t}${SUF}.npz" --out "models/tactical_${t}${SUF}" || exit 1
  done
fi

if has agree; then
  say "teacher agreement (round $ROUND)"
  ORION_T=(); [ "$ROUND" = "1" ] || ORION_T=(--teacher orion="$RUN/labels")
  python3 tools/student/teacher_agreement.py --rec "$RUN/rec" \
    "${ORION_T[@]}" --teacher simlingo="$RUN/labels_simlingo" \
    --teacher privileged="$RUN/labels_privileged" --teacher rules="$RUN/labels_rules" \
    --csv "results/b2d/teacher_agreement${SUF}.csv"
fi
say "done"
