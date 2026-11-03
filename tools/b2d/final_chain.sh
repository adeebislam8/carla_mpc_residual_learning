#!/usr/bin/env bash
# Final unattended chain before the Nov 10 results freeze (2026-11-03), foreground:
#   1. pass-head-only VLA student, full 220 (config_stu_v5_goonly: v5-fix model,
#      go head only -> no VLA speed cap)                       -> results/b2d/heads/
#   2. the rest of the perceived-rules teacher arm (resumes)     -> results/b2d/teachers/
#   3. DAgger round 1 of the VLA student: v5-fix drives with the heads that won
#      step 1 vs v5-fix (higher full-220 DS), ORION + SimLingo label its states,
#      round 0 + 1 -> models/tactical_both_v5_r1                -> results/b2d/pipeline_r1.log
#   4. the DAgger student, full 220 (config_stu_both_v5_r1, written here) -> results/b2d/dagger/
#   5. efficiency: offline per-module benchmark (GPU / CPU / CPU+INT8, GPU memory)
#      on dumped smoke frames + a closed-loop CPU-only abl44 run   -> results/b2d/eff/
#   6. repeat full 220 of the final student (highest DS of v5-fix / go-only /
#      DAgger), then of the base                                 -> results/b2d/repeat/
# Every step continues after a failure of the one before (logged).  Re-running
# the script resumes: finished full-220 jobs are skipped by overnight.sh.
#
#   ./tools/b2d/final_chain.sh 2>&1 | tee -a results/b2d/final_chain.log
set -uo pipefail
cd "$(dirname "$0")/../.."
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}
V5FIX=${V5FIX:-results/b2d/full/stu_v5_fix_full}
GOONLY=results/b2d/heads/stu_v5_goonly_full
DAGGER=results/b2d/dagger/stu_both_v5_r1_full
SKIP=${SKIP:-}                        # e.g. SKIP="1 2" to skip steps already done
say() { echo; echo "[$(date '+%m-%d %H:%M')] == $*"; }
skip() { [[ " $SKIP " == *" $1 "* ]]; }
ds() {  # mean DS of a run's eval.json over its records (empty if missing)
  python3 - "$1" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1] + '/eval.json'))['_checkpoint']['records']
    if len(r) >= 200: print(f"{sum(x['scores']['score_composed'] for x in r) / len(r):.2f}")
except Exception:
    pass
PY
}
df -h . | tail -1

skip 1 || { say "1. pass-head-only student, full 220"
            NAME=heads JOBS="stu_v5_goonly:full" ./tools/b2d/overnight.sh || say "step 1 failed"; }
skip 2 || { say "2. perceived-rules teacher arm (rest)"
            NAME=teachers JOBS="stu_rules_v5:full" ./tools/b2d/overnight.sh || say "step 2 failed"; }

D_FIX=$(ds "$V5FIX"); D_GO=$(ds "$GOONLY")
HEADS=go,dv
[ -n "$D_FIX" ] && [ -n "$D_GO" ] && python3 -c "import sys; sys.exit(0 if $D_GO > $D_FIX else 1)" && HEADS=go
say "heads: v5-fix DS ${D_FIX:-?}, go-only DS ${D_GO:-?} -> DAgger driver and student use heads '$HEADS'"

if ! skip 3; then
  say "3. DAgger round 1 (VLA student drives, ORION + SimLingo label)"
  if [ -d results/b2d/rec_r1_train_clips2 ] && [ ! -f results/b2d/rec_r1_train_clips2/.final_chain ]; then
    mv results/b2d/rec_r1_train_clips2 "results/b2d/rec_r1_old_$(date +%m%d%H%M)"
  fi
  mkdir -p results/b2d/rec_r1_train_clips2 && touch results/b2d/rec_r1_train_clips2/.final_chain
  df -h . | tail -1
  ROUND=1 MASKJ=15 STRIDE=2 DAGGER_DRIVER=v5 DAGGER_ORION=1 DAGGER_HEADS=$HEADS TEACHERS="both privileged" \
    STAGES="rec label experts build train" ./tools/b2d/student_pipeline.sh 2>&1 | tee -a results/b2d/pipeline_r1.log
  [ "${PIPESTATUS[0]}" = "0" ] || say "step 3 failed (see pipeline_r1.log)"
fi

python3 - "$HEADS" <<'PY'
import json, sys
c = json.load(open('tools/b2d/config_stu_v5_fix.json'))
c['tactical'] = 'models/tactical_both_v5_r1'
c['tactical_heads'] = sys.argv[1].split(',')
json.dump(c, open('tools/b2d/config_stu_both_v5_r1.json', 'w'), indent=2)
PY
if ! skip 4; then
  if [ -d models/tactical_both_v5_r1 ]; then
    say "4. DAgger student, full 220"
    NAME=dagger JOBS="stu_both_v5_r1:full" ./tools/b2d/overnight.sh || say "step 4 failed"
  else
    say "4. skipped: no models/tactical_both_v5_r1"
  fi
fi

if ! skip 5; then
  say "5a. efficiency: dump smoke frames with the v5-fix student"
  DUMP=1 TAG=dump CONFIG=tools/b2d/config_stu_v5_fix.json ./tools/b2d/run_b2d.sh smoke > results/b2d/dump_smoke.log 2>&1 \
    || say "dump run failed"
  pkill -9 -f CarlaUE4 2>/dev/null; sleep 5
  mkdir -p results/b2d/eff
  say "5b. offline per-module benchmark (GPU, CPU, CPU + INT8 YOLOP)"
  B="$CONDA run --no-capture-output -n b2d python tools/b2d/bench_stack.py --frames results/b2d/dump_smoke/dump \
     --tactical models/tactical_v5 --traces results/b2d/dump_smoke"
  $B --device cuda --out results/b2d/eff/bench_cuda.json || say "bench cuda failed"
  $B --device cpu --out results/b2d/eff/bench_cpu.json || say "bench cpu failed"
  [ -f models/yolop/yolop-640-640-int8.onnx ] && { $B --device cpu --yolop models/yolop/yolop-640-640-int8.onnx \
    --out results/b2d/eff/bench_cpu_int8.json || say "bench int8 failed"; }
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv > results/b2d/eff/gpu_idle.csv 2>&1
  say "5c. closed-loop CPU-only student run (abl44)"
  NAME=eff JOBS="stu_v5_fix_cpu:abl44" ./tools/b2d/overnight.sh || say "CPU-only run failed"
fi

if ! skip 6; then
  D_DAG=$(ds "$DAGGER")
  read -r FINAL <<<"$(python3 - "$D_FIX" "$D_GO" "$D_DAG" <<'PY'
import sys
c = [(float(d), a) for d, a in zip(sys.argv[1:], ['stu_v5_fix', 'stu_v5_goonly', 'stu_both_v5_r1']) if d]
print(max(c)[1] if c else 'stu_v5_fix')
PY
)"
  say "6. repeats: final student = $FINAL (DS v5-fix ${D_FIX:-?} / go-only ${D_GO:-?} / DAgger ${D_DAG:-?}), then the base"
  NAME=repeat JOBS="$FINAL:full sbase_nolane:full" ./tools/b2d/overnight.sh || say "step 6 failed"
fi
say "final chain done"
