#!/usr/bin/env bash
# Teacher comparison under the v5 recipe (2026-10-30): the go / dv heads from
# each teacher, junction frames masked (--mask-junction 15), no yield head.
# 'both' = models/tactical_v5 (make_student_v5.sh; its yield head is unused by
# config_stu_v5_noyield.json), so only the other four are built here.
# -> models/tactical_<t>_v5, evaluated with config_stu_<t>_v5.json.
# CPU build + ~40 s GPU training each: safe to run while CARLA evaluates.
set -euo pipefail
cd "$(dirname "$0")/../.."
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}
R=${R:-results/b2d/rec_train_clips2}
mkdir -p data
for t in orion simlingo privileged rules; do
  case $t in orion) L=labels ;; simlingo) L=labels_simlingo ;; privileged) L=labels_privileged ;; rules) L=labels_rules ;; esac
  echo "== $t"
  python3 tools/student/build_tactical.py --rec "$R/rec" --labels "$R/$L" --mask-junction 15 \
    --out "data/tactical_${t}_v5.npz"
  PYTHONNOUSERSITE=1 $CONDA run --no-capture-output -n orion python tools/student/train_tactical.py \
    --data "data/tactical_${t}_v5.npz" --out "models/tactical_${t}_v5"
done
