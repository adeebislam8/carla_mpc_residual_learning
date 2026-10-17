#!/usr/bin/env bash
# SimLingo teacher (RenzKa/simlingo, Apache-2.0) for offline labelling on the
# A5000.  Foreground; safe to re-run (skips finished steps).
#
#   ./tools/simlingo/setup_simlingo.sh
#   conda run -n simlingo python tools/simlingo/simlingo_infer.py \
#       --simlingo-root ~/Documents/nett/simlingo --selftest 20
#
# Follows the repo README: conda env from environment.yaml, torch 2.2.0,
# flash-attn 2.7.0.post2; checkpoint from huggingface RenzKa/simlingo.
# nvidia-ml-py added for the energy numbers (src/efficiency/power.py).
set -euo pipefail
NETT=${NETT:-$HOME/Documents/nett}
ROOT=${SIMLINGO_ROOT:-$NETT/simlingo}
CARLA_ROOT=${CARLA15:-$NETT/CARLA_0.9.15}
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}

if [ ! -d "$ROOT/.git" ]; then
  git clone https://github.com/RenzKa/simlingo.git "$ROOT"
fi
if ! "$CONDA" env list | grep -q '^simlingo '; then
  "$CONDA" env create -f "$ROOT/environment.yaml"
fi
PIP="$CONDA run --no-capture-output -n simlingo env PYTHONNOUSERSITE=1 python -m pip"
$PIP install torch==2.2.0
$PIP install flash-attn==2.7.0.post2 --no-build-isolation || \
  echo "!! flash-attn build failed -- check CUDA_HOME / nvcc (same trap as ORION: conda cuda-toolkit)"
$PIP install nvidia-ml-py huggingface_hub

if [ ! -d "$ROOT/ckpts" ] || [ -z "$(ls -A "$ROOT/ckpts" 2>/dev/null)" ]; then
  "$CONDA" run --no-capture-output -n simlingo python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='RenzKa/simlingo', local_dir='$ROOT/ckpts')"
fi
echo "== checkpoint files:"
find "$ROOT/ckpts" -maxdepth 6 \( -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name 'config.yaml' \) | head -20
echo "== simlingo_infer.py finds the checkpoint whose ../../../.hydra/config.yaml exists;"
echo "   if none is listed above in that layout, pass --ckpt explicitly."

# team_code imports carla + agents.navigation (CARLA PythonAPI)
"$CONDA" env config vars set -n simlingo \
  PYTHONPATH="$ROOT:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla" >/dev/null
echo "== done. Selftest:"
echo "   conda run -n simlingo python $(cd "$(dirname "$0")/../.." && pwd)/tools/simlingo/simlingo_infer.py --simlingo-root $ROOT --selftest 20"
