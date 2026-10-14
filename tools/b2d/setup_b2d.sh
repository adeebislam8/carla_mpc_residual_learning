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
NETT=${NETT:-$HOME/Documents/nett}
CARLA15=$NETT/CARLA_0.9.15
B2D=$NETT/Bench2Drive
CONDA=${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}
# downloads.carlasim.com: the backblaze mirror Bench2Drive's README uses failed
# the TLS handshake on the A5000 (2026-10-15).  Override with URL=... if needed.
URL=${URL:-https://downloads.carlasim.com/Linux}

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

if ! "$CONDA" env list | grep -q '^b2d '; then
  "$CONDA" create -y --name b2d --clone residual_mpc
fi
# carla 0.9.15 has a cp310 wheel.  Leaderboard/scenario_runner pins are for
# py3.7 (numpy 1.18, opencv 4.2); unpinned modern versions are used instead,
# except py_trees, whose 0.8 API the scenario runner needs.
PYTHONNOUSERSITE=1 "$CONDA" run -n b2d python -m pip install \
  "carla==0.9.15" "py-trees==0.8.3" networkx shapely psutil xmlschema ephem \
  tabulate six simple-watchdog-timer dictor requests pexpect transforms3d
PYTHONNOUSERSITE=1 "$CONDA" run -n b2d python -c \
  "import carla, py_trees, acados_template; print('b2d env OK, carla', carla.__file__)"
echo "== done.  Next: ./tools/b2d/run_b2d.sh smoke"
