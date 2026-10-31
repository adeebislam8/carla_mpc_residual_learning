#!/usr/bin/env bash
# Unattended gap-acceptance pipeline (2026-10-31), foreground:
#   1. wait until the running overnight.sh queue has finished
#   2. record the 151 training clips with the WIDE student scene (scene only, no
#      images; config_record_wide.json)                         -> $R/rec
#   3. privileged expert labels, gap rule v2                     -> $R/labels_privileged_y2
#   4. gap dataset + train the gap head                          -> data/gap_v6.npz, models/gap_v6
#   5. gate: the wide view must make the decision learnable (val AUC >= AUC_MIN,
#      narrow-view baseline 0.71) and the recording must carry actors_wide;
#      pass -> full 220 of stu_v6 (v5-fix + gap head), else the fallback batch.
#
#   ./tools/b2d/gap_pipeline.sh 2>&1 | tee -a results/b2d/gap_pipeline.log
set -uo pipefail
cd "$(dirname "$0")/../.."
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}
R=results/b2d/recwide_train_clips2
AUC_MIN=${AUC_MIN:-0.80}
PASS_JOBS=${PASS_JOBS:-"stu_v6:full"}
FALLBACK_JOBS=${FALLBACK_JOBS:-"stu_simlingo_v5:abl44 stu_privileged_v5:abl44 stumap_v5:abl44"}
say() { echo; echo "[$(date '+%m-%d %H:%M')] == $*"; }

say "waiting for the running overnight queue"
while pgrep -f overnight.sh >/dev/null; do sleep 300; done

say "recording wide-scene clips -> $R/rec"
TAG=recwide CONFIG=tools/b2d/config_record_wide.json ./tools/b2d/run_b2d.sh train_clips2 \
  > "$R.log" 2>&1
n=$(ls -d "$R"/rec/*/ 2>/dev/null | wc -l)
say "recorded $n episodes"
[ "$n" -ge 100 ] || { say "too few episodes -> fallback"; GATE=fail; }

if [ "${GATE:-}" != "fail" ]; then
  say "privileged gap labels (rule v2)"
  python3 tools/student/expert_teacher.py --mode privileged --yield-v2 --stride 2 \
    --rec "$R/rec" --out "$R/labels_privileged_y2" || GATE=fail
fi
if [ "${GATE:-}" != "fail" ]; then
  say "gap dataset"
  mkdir -p data
  python3 tools/student/build_gap.py --rec "$R/rec" --labels "$R/labels_privileged_y2" \
    --out data/gap_v6.npz || GATE=fail
fi
if [ "${GATE:-}" != "fail" ]; then
  say "training the gap head"
  PYTHONNOUSERSITE=1 $CONDA run --no-capture-output -n orion python tools/student/train_tactical.py \
    --data data/gap_v6.npz --out models/gap_v6 || GATE=fail
fi
if [ "${GATE:-}" != "fail" ]; then
  GATE=$(python3 - "$AUC_MIN" <<'EOF'
import json, sys, numpy as np
auc = json.load(open('models/gap_v6/tactical_meta.json'))['metrics']['yield_val_auc']
wide = json.loads(str(np.load('data/gap_v6.npz')['config']))['wide_frac']
ok = auc >= float(sys.argv[1]) and wide > 0.5
print('pass' if ok else 'fail', file=sys.stdout)
print(f'gap head val AUC {auc:.3f} (need >= {sys.argv[1]}), scenes with actors_wide {100 * wide:.0f} %',
      file=sys.stderr)
EOF
)
fi
say "gate: ${GATE:-fail}"

if [ "${GATE:-fail}" = "pass" ]; then
  say "full 220 with the gap head: $PASS_JOBS"
  NAME=gap JOBS="$PASS_JOBS" ./tools/b2d/overnight.sh
else
  say "fallback batch: $FALLBACK_JOBS"
  NAME=teach JOBS="$FALLBACK_JOBS" ./tools/b2d/overnight.sh
fi
say "done"
