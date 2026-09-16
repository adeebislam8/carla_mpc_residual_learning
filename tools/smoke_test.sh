#!/usr/bin/env bash
# End-to-end smoke test: ~10 minutes, exercises every code path the overnight
# run depends on, with tiny parameters.
#
#   ./tools/smoke_test.sh
#
# Run this before ANY long job.  Two defects killed a 10-hour run that this
# would have caught in minutes:
#   * tensorboard_log raised ImportError inside learn(), and because the save
#     was in a `finally`, an UNTRAINED model was written that looked fine.
#   * VecNormalize.load(path, venv=None) raised AttributeError at eval time.
# Both only appear when training and evaluation are actually executed, so a
# --dry-run cannot find them.

set -uo pipefail
cd "$(dirname "$0")/.."

TMP_LABEL="smoke"
PASS=0; FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
step() { echo; echo "=================================================="; \
         echo ">>> $1"; echo "=================================================="; }

rm -rf "models/${TMP_LABEL}" "results/${TMP_LABEL}"*.json \
       "results/${TMP_LABEL}"*.txt diagnostics_smoke 2>/dev/null

# ---------------------------------------------------------------- 1. NOMINAL
step "1/5  nominal MPCC (no model) -- env, solver, reward, report"
if python tools/benchmark_mpcc.py --label "${TMP_LABEL}_b0" \
     --seeds 1 --episodes 2 --route-max 150 --qc 0.5 --gate-depth 0.98 \
     > "results/${TMP_LABEL}_b0.log" 2>&1; then
  ok "nominal run completed"
else
  bad "nominal run crashed -- see results/${TMP_LABEL}_b0.log"
  tail -20 "results/${TMP_LABEL}_b0.log"; echo; echo "ABORTING"; exit 1
fi
[ -f "results/${TMP_LABEL}_b0.json" ] && ok "wrote json" || bad "no json written"

# ---------------------------------------------------------------- 2. TRAINING
step "2/5  training (2000 steps) -- learn(), save, vecnormalize"
if python tools/train_residual.py --label "$TMP_LABEL" --algo sac \
     --timesteps 2000 --buffer-size 5000 \
     --route-max 150 --qc 0.5 --gate-depth 0.98 \
     > "results/${TMP_LABEL}_train.log" 2>&1; then
  ok "training completed"
else
  bad "training crashed"; tail -20 "results/${TMP_LABEL}_train.log"
fi

# The bug that mattered: a model saved by `finally` after learn() crashed.
if grep -qE "ep_rew_mean|time/total_timesteps|rollout/" "results/${TMP_LABEL}_train.log"; then
  ok "learn() actually ran (training output present)"
else
  bad "NO training output -- learn() likely never executed; any saved model is UNTRAINED"
  grep -iE "error|not installed" "results/${TMP_LABEL}_train.log" | head -3
fi
[ -f "models/${TMP_LABEL}/final.zip" ] && ok "model saved" || bad "no model saved"
[ -f "models/${TMP_LABEL}/vecnormalize.pkl" ] && ok "vecnormalize saved" \
  || bad "no vecnormalize.pkl (evaluation would feed the policy raw observations)"

# ------------------------------------------------------- 3/4. EVAL WITH MODEL
for MODE in fixed adaptive; do
  step "3/5  evaluation with model, residual-mode=${MODE}"
  if [ ! -f "models/${TMP_LABEL}/final.zip" ]; then
    bad "skipped (${MODE}) -- no model"; continue
  fi
  if python tools/benchmark_mpcc.py --label "${TMP_LABEL}_${MODE}" \
       --model "models/${TMP_LABEL}/final.zip" --algo sac \
       --residual-mode "$MODE" --seeds 1 --episodes 2 \
       --route-max 150 --qc 0.5 --gate-depth 0.98 \
       > "results/${TMP_LABEL}_${MODE}.log" 2>&1; then
    ok "eval (${MODE}) completed"
    grep -q "applying observation normalisation" "results/${TMP_LABEL}_${MODE}.log" \
      && ok "  VecNormalize applied" || bad "  VecNormalize NOT applied"
  else
    bad "eval (${MODE}) crashed"; tail -15 "results/${TMP_LABEL}_${MODE}.log"
  fi
done

# ------------------------------------------------------------- 5. COMPARISON
step "5/5  comparison table"
JS=(); for f in "${TMP_LABEL}_b0" "${TMP_LABEL}_fixed" "${TMP_LABEL}_adaptive"; do
  [ -f "results/${f}.json" ] && JS+=("results/${f}.json"); done
if [ "${#JS[@]}" -ge 2 ] && python tools/benchmark_mpcc.py --compare "${JS[@]}" >/dev/null 2>&1; then
  ok "comparison produced"
else
  bad "comparison failed (${#JS[@]} json files)"
fi

echo; echo "=================================================="
echo "  PASS ${PASS}   FAIL ${FAIL}"
if [ "$FAIL" -eq 0 ]; then
  echo "  pipeline is sound -- safe to start the long run"
else
  echo "  DO NOT start the overnight run until these are fixed"
fi
echo "=================================================="
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
