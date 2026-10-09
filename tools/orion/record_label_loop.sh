#!/usr/bin/env bash
# Record -> label (-> optionally prune), one seed at a time.  With PRUNE=1 disk use stays at about one
# chunk of JPEGs instead of the whole dataset (~25 GB for 300 episodes).
#
#   ./tools/orion/record_label_loop.sh                  # seeds 101..112, 25 episodes each
#   SEEDS="101 102" EPISODES=10 PRUNE=1 ./tools/orion/record_label_loop.sh
#
# SEEDS: a seed fixes the whole route/traffic sequence (CarlaMPCEnv's per-
# episode RNG), and benchmark_mpcc.py evaluates on seeds 1-5.  Recording on
# those seeds would train the student on the exact evaluation routes, so
# training data uses 101+ (round 0), 201+ (DAgger), 301+ (Town02 adaptation).
#
# STUDENT=<dir>: DAgger round -- the student drives (under the authority gate)
# instead of the nominal MPCC; use a fresh seed range and REC/LAB names, e.g.
#   STUDENT=models/student_r0 SEEDS="201 202 203 204" TOWN=Town01 \
#     D=~/Documents/nett/orion_data/dagger1 ./tools/orion/record_label_loop.sh
#
# PRUNE=1 deletes each episode's JPEGs once its labels verify.  Default 0:
# keep the frames, so they can be re-labelled later (e.g. a compressed or
# token-pruned teacher, or a different --stride) without re-recording.
#
# Per seed:
#   1. start CARLA, record EPISODES episodes        (residual_mpc env)
#   2. check the recording, stop CARLA              (frees the GPU for ORION)
#   3. label every episode of this seed             (orion env)
#   4. delete the camera JPEGs of every episode whose label file loads and has
#      as many rows as the recording -- meta.jsonl, obs.npy and summary.json
#      stay, which is all the student needs
#
# One seed = one chunk because the recorder names episodes e000.. per call:
# reusing a seed would overwrite earlier episodes.  Seeds already labelled are
# skipped, so the script can be re-run after an interruption.
#
# CARLA and ORION share the A5000 (ORION needs ~16 GB), so they never run at
# the same time.  The script kills CarlaUE4 -- do not run another CARLA job on
# this machine meanwhile.

set -uo pipefail
cd "$(dirname "$0")/../.."

SEEDS=${SEEDS:-"101 102 103 104 105 106 107 108 109 110 111 112"}
PRUNE=${PRUNE:-0}
STUDENT=${STUDENT:-}
for _s in $SEEDS; do
  if [ "$_s" -le 5 ]; then
    echo "seed $_s is an evaluation seed (benchmark uses 1-5) -- refusing"; exit 1
  fi
done
EPISODES=${EPISODES:-25}
TOWN=${TOWN:-Town01}
D=${D:-$HOME/Documents/nett/orion_data}
REC="$D/rec_${TOWN,,}"
LAB="$D/labels_${TOWN,,}"
ORION_ROOT=${ORION_ROOT:-$HOME/Documents/nett/Orion}
CONTROLLER="--route-max 150 --qc 0.5 --gate-depth 0.98"
MIN_FREE_GB=${MIN_FREE_GB:-8}
: "${CARLA_ROOT:?set CARLA_ROOT}"
# The conda binary, not the shell function, so this also works under nohup.
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}

mkdir -p "$REC" "$LAB"
log() { echo "[$(date +%H:%M:%S)] $*"; }

free_gb() { df -BG --output=avail "$D" | tail -1 | tr -dc '0-9'; }

start_carla() {
  pkill -f CarlaUE4 2>/dev/null; sleep 5
  "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen > "$D/carla.log" 2>&1 &
  for _ in $(seq 1 30); do
    sleep 5
    if "$CONDA" run -n residual_mpc python -c \
        "import carla; c=carla.Client('localhost',2000); c.set_timeout(5); c.get_server_version()" \
        >/dev/null 2>&1; then
      log "CARLA up"; return 0
    fi
  done
  log "CARLA did not come up -- see $D/carla.log"; return 1
}

stop_carla() { pkill -f CarlaUE4 2>/dev/null; sleep 10; log "CARLA stopped"; }

labelled_ok() {   # labelled_ok <episode dir>  -> 0 if its npz is complete
  local ep=$1 npz="$LAB/$(basename "$1").npz"
  [ -f "$npz" ] || return 1
  "$CONDA" run -n orion python - "$ep" "$npz" <<'EOF' >/dev/null 2>&1
import sys, numpy as np
ep, npz = sys.argv[1], sys.argv[2]
n_rec = sum(1 for l in open(f'{ep}/meta.jsonl') if l.strip())
lab = np.load(npz)
sys.exit(0 if len(lab['steps']) == n_rec and n_rec > 0 else 1)
EOF
}

for SEED in $SEEDS; do
  # Skip a seed only when all of its episodes exist and every one has a
  # complete label file -- pruned or not.
  n_done=0
  for ep in "$REC/${TOWN}_s${SEED}_e"*; do
    [ -d "$ep" ] && labelled_ok "$ep" && n_done=$((n_done + 1))
  done
  if [ "$n_done" -ge "$EPISODES" ]; then
    log "seed $SEED already recorded and labelled ($n_done episodes) -- skipping"; continue
  fi
  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
    log "only $(free_gb) GB free (< ${MIN_FREE_GB}) -- stopping before seed $SEED"; exit 1
  fi

  log "=== seed $SEED: record $EPISODES episodes ($(free_gb) GB free)"
  start_carla || exit 1
  # shellcheck disable=SC2086
  "$CONDA" run --no-capture-output -n residual_mpc python tools/orion/record_orion_episodes.py \
    --town "$TOWN" --seeds "$SEED" --episodes "$EPISODES" $CONTROLLER --out "$REC" \
    ${STUDENT:+--student "$STUDENT"} \
    2>&1 | grep -E "^${TOWN}_s|Error|error" | tee -a "$D/record.log"
  "$CONDA" run --no-capture-output -n residual_mpc python tools/orion/check_orion_recording.py \
    "$REC/${TOWN}_s${SEED}_e"* --montage-every 0 2>&1 | tail -1 | tee -a "$D/record.log"
  stop_carla

  log "=== seed $SEED: label"
  "$CONDA" run --no-capture-output -n orion python tools/orion/orion_infer.py \
    --orion-root "$ORION_ROOT" --jpeg-quality 0 \
    --episode "$REC/${TOWN}_s${SEED}_e"* --out "$LAB" 2>&1 \
    | grep -E "^\[|peak|Error|error" | tee -a "$D/label.log"

  for ep in "$REC/${TOWN}_s${SEED}_e"*; do
    if ! labelled_ok "$ep"; then
      log "  $(basename "$ep"): label missing or incomplete -- frames kept"
    elif [ "$PRUNE" = "1" ]; then
      rm -rf "$ep"/CAM_* "$ep"/montage_*.png
    fi
  done
  log "seed $SEED done ($(free_gb) GB free)"
done
log "all seeds done.  labels: $LAB   recordings: $REC"
