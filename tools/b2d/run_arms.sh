#!/usr/bin/env bash
# Run several agent configurations ("arms") on a Bench2Drive route set, then
# score them all -- one command, foreground.
#
#   ./tools/b2d/run_arms.sh                    # full 220 routes, 3 arms, one after another
#   PARALLEL=1 ./tools/b2d/run_arms.sh         # all arms at once on the same GPU
#   SET=ot_eb ARMS="base rules" ./tools/b2d/run_arms.sh
#
# Arms (name -> config):
#   base     tools/b2d/config_mpcc.json         MPCC + physics fixes, minimal pass rule
#   rules    tools/b2d/config_rules.json        + hand-coded decision rules
#   rulesq   tools/b2d/config_rules_queue.json  + queue rule (brittleness ablation)
# Results: results/b2d/<arm>_<SET>/ ; each arm resumes if re-run.
#
# PARALLEL=1: each arm gets its own CARLA server and ports (30000+100*k,
# 50000+100*k) on the same GPU.  ~3x faster, but needs ~3x GPU memory (Town12/13
# are heavy; 24 GB is enough for 3 in practice) and crashes are more frequent --
# the per-arm restart loop handles them.  Requires setup_b2d.sh's crash-kill
# patch (re-run setup_b2d.sh once), else one crash kills every arm's server.

set -uo pipefail
cd "$(dirname "$0")/../.."
SET=${SET:-full}
ARMS=${ARMS:-"base rules rulesq"}
PARALLEL=${PARALLEL:-0}
mkdir -p results/b2d

config_of() {
  case "$1" in
    base)   echo tools/b2d/config_mpcc.json ;;
    rules)  echo tools/b2d/config_rules.json ;;
    rulesq) echo tools/b2d/config_rules_queue.json ;;
    *)      [ -f "tools/b2d/config_$1.json" ] && echo "tools/b2d/config_$1.json" \
              || { echo "unknown arm $1" >&2; return 1; } ;;
  esac
}

START=$(date +%s)
k=0
PIDS=()
for ARM in $ARMS; do
  CFG=$(config_of "$ARM") || exit 1
  PORT=$((30000 + 100 * k)); TM=$((50000 + 100 * k))
  echo "== arm $ARM: $CFG, ports $PORT/$TM -> results/b2d/${ARM}_${SET}"
  if [ "$PARALLEL" = "1" ]; then
    mkdir -p "results/b2d/${ARM}_${SET}"
    TAG=$ARM CONFIG=$CFG PORT=$PORT TM_PORT=$TM ./tools/b2d/run_b2d.sh "$SET" \
      > "results/b2d/${ARM}_${SET}/arm.out" 2>&1 &
    PIDS+=($!)
    sleep 60          # stagger the CARLA launches
  else
    TAG=$ARM CONFIG=$CFG PORT=$PORT TM_PORT=$TM ./tools/b2d/run_b2d.sh "$SET"
  fi
  k=$((k + 1))
done

if [ "$PARALLEL" = "1" ]; then
  echo "== running in parallel; progress:  tail -f results/b2d/<arm>_${SET}/arm.out"
  while true; do
    alive=0
    for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
    line="[$(( ($(date +%s) - START) / 60 )) min]"
    for ARM in $ARMS; do
      n=$(python3 -c "
import json
try: print(json.load(open('results/b2d/${ARM}_${SET}/eval.json'))['_checkpoint']['progress'][0])
except Exception: print(0)")
      line="$line  $ARM $n"
    done
    echo "$line"
    [ "$alive" -eq 0 ] && break
    sleep 300
  done
fi

echo; echo "== all arms done in $(( ($(date +%s) - START) / 60 )) min"
for ARM in $ARMS; do
  echo; echo "################ $ARM ($SET)"
  TAG=$ARM ./tools/b2d/run_b2d.sh score "$SET"
done
