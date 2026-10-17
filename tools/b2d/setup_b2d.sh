#!/usr/bin/env bash
# One-time Bench2Drive setup for the MPCC agent (A5000 or the 8 GB machine).
# Foreground, re-runnable: each step is skipped if already done.
#
#   ./tools/b2d/setup_b2d.sh
#
# Puts, under $NETT (default ~/Documents/nett):
#   CARLA_0.9.15/     CARLA 0.9.15 + AdditionalMaps (Town11-15; most B2D routes
#                     are in Town12/13).  ~25 GB download, ~40 GB unpacked.
#   Bench2Drive/      the benchmark (leaderboard + scenario_runner + routes)
# and creates conda env 'b2d' = a clone of residual_mpc (acados, our code)
# with the CARLA 0.9.15 client and the leaderboard's Python deps.
#
# Why a separate CARLA: Bench2Drive's scenarios and route files target 0.9.15;
# the rest of the project stays on 0.9.16.  The 0.9.16 server must NOT be
# running while Bench2Drive runs (the leaderboard starts its own 0.9.15 server).

set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)    # before any cd below
# NETT: where CARLA 0.9.15 and Bench2Drive go.  SRC_ENV: the conda env with
# acados + our deps to clone (A5000: residual_mpc; 8 GB machine: mpc_rl).
NETT=${NETT:-$HOME/Documents/nett}
SRC_ENV=${SRC_ENV:-residual_mpc}
CARLA15=$NETT/CARLA_0.9.15
B2D=$NETT/Bench2Drive
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}
# downloads.carlasim.com: the backblaze mirror Bench2Drive's README uses failed
# the TLS handshake on the A5000 (2026-10-15).  Override with URL=... if needed.
URL=${URL:-https://downloads.carlasim.com/Linux}

mkdir -p "$NETT"
echo "== free disk: $(df -h "$NETT" | awk 'NR==2{print $4}') (need ~70 GB for download + unpack)"

if [ ! -x "$CARLA15/CarlaUE4.sh" ]; then
  mkdir -p "$CARLA15" && cd "$CARLA15"
  wget -c "$URL/CARLA_0.9.15.tar.gz"     # resumes; no-op when complete
  tar -xzf CARLA_0.9.15.tar.gz
  mkdir -p Import && cd Import
  wget -c "$URL/AdditionalMaps_0.9.15.tar.gz"
  cd .. && bash ImportAssets.sh
  echo "== CARLA 0.9.15 ready (you may delete the two .tar.gz files)"
else
  echo "== CARLA 0.9.15 already at $CARLA15"
fi

if [ ! -d "$B2D/leaderboard" ]; then
  git clone --depth 1 https://github.com/Thinklab-SJTU/Bench2Drive.git "$B2D"
else
  echo "== Bench2Drive already at $B2D"
fi

# Python 3.10 compatibility: Bench2Drive targets py3.7.  Element.getchildren()
# was removed in 3.9; list(elem) is the identical drop-in.  Idempotent.
# (grep exits 1 when nothing matches -- e.g. already patched -- which set -e
# -o pipefail would treat as fatal, hence the '|| true'.)
{ grep -rl 'getchildren()' --include='*.py' "$B2D/leaderboard" "$B2D/scenario_runner" || true; } \
  | xargs -r sed -i -E 's/([A-Za-z_][A-Za-z0-9_.]*)\.getchildren\(\)/list(\1)/g'
N_LEFT=$( { grep -rl 'getchildren()' --include='*.py' "$B2D/leaderboard" "$B2D/scenario_runner" || true; } | wc -l)
echo "== py3.10 patch: $N_LEFT files still use getchildren (want 0)"
# Parallel arms on ONE GPU: after a crash the evaluator kills every CARLA with
# '-graphicsadapter=<gpu>', i.e. the other arms' servers too.  Kill by the
# crashed run's own RPC port instead.  Idempotent.
sed -i "s/grep '-graphicsadapter=\"+ str(args.gpu_rank) + \"'/grep '-carla-rpc-port=\"+ str(args.port) + \"'/" \
  "$B2D/leaderboard/leaderboard/leaderboard_evaluator.py"
echo "== crash-kill patch: $(grep -c "carla-rpc-port=\"+ str(args.port)" "$B2D/leaderboard/leaderboard/leaderboard_evaluator.py") (want 1)"

if ! "$CONDA" env list | grep -q '^b2d '; then
  "$CONDA" create -y --name b2d --clone "$SRC_ENV"
fi
# carla 0.9.15 has a cp310 wheel.  Leaderboard/scenario_runner pins are for
# py3.7 (numpy 1.18, opencv 4.2); unpinned modern versions are used instead,
# except py_trees, whose 0.8 API the scenario runner needs.
# The source env may get some packages from ~/.local, which Bench2Drive runs
# hide (PYTHONNOUSERSITE=1) -- so everything our agent imports is listed here.
#   ours:        gymnasium scipy matplotlib casadi opencv (+ acados_template below)
#   Bench2Drive: py-trees networkx shapely psutil xmlschema ephem tabulate six
#                simple-watchdog-timer dictor requests pexpect transforms3d,
#                setuptools<81 (pkg_resources), simplejson (optional)
#   B2D_YOLO=1:  ultralytics (pulls PyTorch, ~2.5 GB) for 'perception': true and
#                the sensor-only lights; onnxruntime-gpu + YOLOP weights for lanes
PYTHONNOUSERSITE=1 "$CONDA" run -n b2d python -m pip install \
  "carla==0.9.15" "py-trees==0.8.3" networkx shapely psutil xmlschema ephem \
  tabulate six simple-watchdog-timer dictor requests pexpect transforms3d \
  gymnasium scipy matplotlib "casadi==3.7.2" opencv-python-headless \
  "setuptools<81" simplejson
if [ "${B2D_YOLO:-0}" = "1" ]; then
  PYTHONNOUSERSITE=1 "$CONDA" run -n b2d python -m pip install ultralytics onnxruntime-gpu \
    nvidia-ml-py      # + GPU energy for tools/b2d/bench_stack.py
  # YOLOP lane lines for the sparse-route (sensor-only) agent: pretrained
  # hustvl/YOLOP, BDD100K; perception/lanes.py
  mkdir -p "$REPO/models/yolop"
  if [ ! -s "$REPO/models/yolop/yolop-640-640.onnx" ]; then
    wget -q -O "$REPO/models/yolop/yolop-640-640.onnx" \
      https://github.com/hustvl/YOLOP/raw/main/weights/yolop-640-640.onnx
  fi
fi
# acados_template is an editable/local install in the source env (the acados
# submodule), which a conda clone does not always carry over: find where the
# source env imports it from and install that directory into b2d too, and copy
# the source env's activate.d (ACADOS_SOURCE_DIR / LD_LIBRARY_PATH).
if ! PYTHONNOUSERSITE=1 "$CONDA" run -n b2d python -c "import acados_template" 2>/dev/null; then
  SRC_PKG=$REPO/src/mpc_controller/src/acados/interfaces/acados_template   # the submodule
  if [ ! -f "$SRC_PKG/setup.py" ] && [ ! -f "$SRC_PKG/pyproject.toml" ]; then
    SRC_PKG=$("$CONDA" run -n "$SRC_ENV" python -c \
      "import acados_template, os; print(os.path.dirname(os.path.dirname(acados_template.__file__)))" | tail -1)
  fi
  echo "== installing acados_template from $SRC_PKG into b2d"
  PYTHONNOUSERSITE=1 "$CONDA" run -n b2d python -m pip install -e "$SRC_PKG"
  SRC_PREFIX=$("$CONDA" run -n "$SRC_ENV" python -c "import sys; print(sys.prefix)" | tail -1)
  B2D_PREFIX=$("$CONDA" run -n b2d python -c "import sys; print(sys.prefix)" | tail -1)
  if [ -d "$SRC_PREFIX/etc/conda/activate.d" ]; then
    mkdir -p "$B2D_PREFIX/etc/conda/activate.d"
    cp -n "$SRC_PREFIX"/etc/conda/activate.d/* "$B2D_PREFIX/etc/conda/activate.d/" 2>/dev/null || true
  fi
fi
# Import the whole agent stack exactly as the leaderboard will see it, so a
# missing module shows up here rather than one per failed run.
PYTHONNOUSERSITE=1 \
PYTHONPATH="$CARLA15/PythonAPI:$CARLA15/PythonAPI/carla:$B2D/leaderboard:$B2D/scenario_runner:$REPO/src" \
  "$CONDA" run -n b2d python -c "
import carla, py_trees, networkx, shapely, acados_template, casadi, gymnasium, scipy
from agents.navigation.global_route_planner import GlobalRoutePlanner
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from mpc_controller.envs.carlaEnv import CarlaMPCEnv
from student.policy import StudentPolicy
print('b2d env OK: carla', carla.__file__, '| casadi', casadi.__version__)
"
echo "== done.  Next: ./tools/b2d/run_b2d.sh smoke"
