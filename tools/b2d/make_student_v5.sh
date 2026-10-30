#!/usr/bin/env bash
# Student v5 (2026-10-30, from the stu3 abl44 gain/loss analysis) -- no new
# recording or VLA labelling, minutes on the A5000:
#   go + dv  from ORION + SimLingo ('both', soft), junction frames masked
#            (--mask-junction 15: the VLAs' junction speeds are timed to THEIR
#            trajectory; copying them on ours caused the junction-flow collisions)
#   yield    from the privileged expert, gap-acceptance rule v2 (--yield-v2)
# -> models/tactical_v5.  Run-time options live in the configs:
#   config_stu_v5.json          go/dv/yield + junction gate 15 m + wait only behind
#                               stopped blockers + ensemble deferral
#   config_stu_v5_noyield.json  the same without the yield head
set -euo pipefail
cd "$(dirname "$0")/../.."
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}
R=${R:-results/b2d/rec_train_clips2}
STRIDE=${STRIDE:-2}
python3 tools/student/expert_teacher.py --mode privileged --yield-v2 --stride "$STRIDE" \
  --rec "$R/rec" --out "$R/labels_privileged_y2"
mkdir -p data
python3 tools/student/build_tactical.py --rec "$R/rec" --labels "$R/labels" --labels2 "$R/labels_simlingo" \
  --combine soft --mask-junction 15 --yield-from "$R/labels_privileged_y2" --out data/tactical_v5.npz
PYTHONNOUSERSITE=1 $CONDA run --no-capture-output -n orion python tools/student/train_tactical.py \
  --data data/tactical_v5.npz --out models/tactical_v5
