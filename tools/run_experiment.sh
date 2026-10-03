#!/usr/bin/env bash
# Train the residual policy, then evaluate the full ablation across towns.
#
# Designed to run unattended overnight: one config failing does not abort the
# rest, everything is logged to results/, and the comparison tables are written
# to disk as well as printed.
#
#   ./tools/run_experiment.sh                  # train + evaluate
#   ./tools/run_experiment.sh --eval-only      # skip training, use existing models
#   ./tools/run_experiment.sh --dry-run        # print the plan, run nothing
#
# Requires a CARLA server already running.  Rough cost at the defaults:
# 2 algos x 300k steps is the bulk (several hours), evaluation adds
# ~12 min per (arm x town).

set -uo pipefail            # NOT -e: a failed arm must not kill the night
cd "$(dirname "$0")/.."

# ------------------------------------------------------------------ CONFIG ---
# Algorithms to train.  TD3 first: no entropy bonus pushing the residual away
# from zero, which is the right bias for a correction that should be small
# unless it helps.  SAC second with a small fixed ent-coef for comparison.
ALGOS=("td3" "sac")
TIMESTEPS=300000

# Nominal controller.  MUST match between training and evaluation -- a residual
# learns "given THIS nominal behaviour, what correction helps", so training it
# against a different MPCC config makes it invalid.
CONTROLLER="--qc 0.5 --gate-depth 0.98 --route-max 150"

TRAIN_TOWN="Town01"                 # train on ONE town; cross-town IS the shift
EVAL_TOWNS=("Town01" "Town02" "Town03")
# 5 seeds x 30 episodes = 150 per arm per town.  60 episodes could not resolve
# anything in the MPCC sweeps -- four configs landed between 61.7% and 70.7%
# with p = 0.32..0.85, and the seed-to-seed spread on a single config was 23
# points.  The ablation has to resolve B0 vs B5, so the sample size has to be
# able to see an effect of that size.
EVAL="--seeds 1 2 3 4 5 --episodes 30"

# Ablation arms per town:  label | residual mode ('' = no residual)
#   b0  nominal MPCC, no residual          (the baseline everything is measured against)
#   b2  fixed residual scale               (the previous architecture)
#   b5  CBF-derived adaptive authority     (Contribution 1)
#
# TRAIN/EVAL MODE MATCH.  Until 2026-09-22 this script never passed
# --residual-mode to train_residual.py, so BOTH arms were served by a single
# policy trained under the 'fixed' default (u = u_nom + 0.1*pi(o)) while b5 was
# evaluated under 'adaptive' (u = u_nom + alpha_safe*0.1*pi(o)).  The policy had
# never seen its own actions scaled by alpha_safe, so b5 measured a train/test
# mismatch rather than the authority mechanism -- which is the most likely
# reason b5 came out WORSE than b2 in 5 of 6 comparisons.  Each mode now trains
# its own policy and evaluates it under the same law it was trained on.
ARMS=(
  "b0|"
  "b2|fixed"
  "b5|adaptive"
)

# Modes needing a trained policy, derived from ARMS so the two cannot drift.
TRAIN_MODES=()
for _a in "${ARMS[@]}"; do
  _m="${_a#*|}"
  [ -n "$_m" ] && case " ${TRAIN_MODES[*]-} " in *" $_m "*) ;; *) TRAIN_MODES+=("$_m") ;; esac
done

# models/<algo>_<mode>_v1/.  A pre-2026-09-22 models/<algo>_v1 was trained
# 'fixed' and is still valid for b2 -- reuse it with
#     mv models/sac_v1 models/sac_fixed_v1
# rather than retraining.  Training skips any mode whose model already exists.
model_dir() { echo "models/$1_$2_v1"; }
# ------------------------------------------------------------------------------

DRY=0; EVAL_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)   DRY=1; shift ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    # e.g. --algos "sac"  -- TD3 was significantly harmful (success -9.56pp,
    # p = 2.3e-05), so training an adaptive TD3 costs ~5 h to re-confirm a
    # known regression.
    --algos)     read -r -a ALGOS <<< "$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p results models
START=$(date +%s)
FAILED=()
run() {  # run <description> <command...>
  echo; echo "=================================================================="
  echo ">>> $1"
  echo "    elapsed $((($(date +%s)-START)/60))m"
  echo "=================================================================="
  shift
  if [ "$DRY" -eq 1 ]; then echo "    (dry run) $*"; return 0; fi
  if "$@"; then return 0; else
    echo "!!! FAILED: $* -- continuing"; FAILED+=("$*"); return 1
  fi
}

# ---------------------------------------------------------------- TRAINING ---
if [ "$EVAL_ONLY" -eq 0 ]; then
  for algo in "${ALGOS[@]}"; do
    for mode in "${TRAIN_MODES[@]}"; do
      label="${algo}_${mode}_v1"
      if [ -f "$(model_dir "$algo" "$mode")/final.zip" ]; then
        echo ">>> SKIP TRAIN ${label} -- $(model_dir "$algo" "$mode")/final.zip exists"
        continue
      fi
      # shellcheck disable=SC2086
      run "TRAIN ${label} (${TIMESTEPS} steps, ${TRAIN_TOWN}, mode=${mode})" \
        python tools/train_residual.py --label "$label" --algo "$algo" \
          --residual-mode "$mode" \
          --timesteps "$TIMESTEPS" --town "$TRAIN_TOWN" $CONTROLLER \
          2>&1 | tee "results/train_${label}.log"
    done
  done
fi

# -------------------------------------------------------------- EVALUATION ---
for algo in "${ALGOS[@]}"; do
  for town in "${EVAL_TOWNS[@]}"; do
    JSONS=()
    for arm in "${ARMS[@]}"; do
      name="${arm%%|*}"; mode="${arm#*|}"
      label="${algo}_${name}_${town}"

      # b0 is the nominal controller: no --model, action is always [0,0].
      # It is identical for every algo, so compute it once and reuse.
      if [ "$name" = "b0" ]; then
        label="b0_${town}"
        [ -f "results/${label}.json" ] && { JSONS+=("results/${label}.json"); continue; }
        MODEL_FLAGS=""
      else
        # The policy trained UNDER THIS MODE, not a single shared one.
        MODEL="$(model_dir "$algo" "$mode")/final.zip"
        if [ "$DRY" -eq 0 ] && [ ! -f "$MODEL" ]; then
          echo "!!! no model at ${MODEL} -- skipping ${label}"
          FAILED+=("eval ${label}: model missing"); continue
        fi
        MODEL_FLAGS="--model ${MODEL} --algo ${algo} --residual-mode ${mode}"
      fi

      # shellcheck disable=SC2086
      run "EVAL ${label}" \
        python tools/benchmark_mpcc.py --label "$label" --town "$town" \
          $EVAL $CONTROLLER $MODEL_FLAGS \
          2>&1 | tee "results/${label}.log"
      [ -f "results/${label}.json" ] && JSONS+=("results/${label}.json")
    done

    if [ "$DRY" -eq 0 ] && [ "${#JSONS[@]}" -ge 2 ]; then
      echo; echo "--- ABLATION: ${algo} on ${town} ---"
      python tools/benchmark_mpcc.py --compare "${JSONS[@]}" \
        | tee "results/ablation_${algo}_${town}.txt"
    fi
  done
done

echo
echo "=================================================================="
echo "finished in $((($(date +%s)-START)/60)) min"
[ "${#FAILED[@]}" -gt 0 ] && { echo "FAILED steps:"; printf '   %s\n' "${FAILED[@]}"; }
echo
echo "per-town ablations : results/ablation_<algo>_<town>.txt"
echo "per-run reports    : results/<label>.txt"
echo "models             : models/<algo>_v1/final.zip"
echo "=================================================================="
