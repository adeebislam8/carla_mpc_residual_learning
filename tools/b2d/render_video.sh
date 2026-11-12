#!/usr/bin/env bash
# Render base vs EFF-Student side-by-side clips from RECORD=1 recorder logs.
# Restarts CARLA before every render (a crash costs one route, not the rest) and
# skips routes whose clip already exists.  Run in the b2d env:
#   ./tools/b2d/render_video.sh 24757:video2 24816:video2 2539:video ...
# Each argument is <route id>:<run set>, reading results/b2d/v{base,eff}_<set>/recorder.
# Output: clip_<id>.mp4 (1920x1080, 30 fps) in the repo root; FORCE=1 re-renders.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CARLA_ROOT=${CARLA_ROOT:-$HOME/Documents/nett/CARLA_0.9.15}
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}
cd "$REPO"

carla_up() {
  pkill -f CarlaUE4; pkill -f leaderboard_evaluator; sleep 5
  (cd "$CARLA_ROOT" && ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 >/dev/null 2>&1 &)
  for _ in $(seq 1 30); do
    python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(5); c.get_server_version()" 2>/dev/null \
      && { echo "carla up"; return 0; }
    sleep 5
  done
  echo "carla did not start"; return 1
}

render() {   # render <id> <set> <base|eff> <label>
  local log=results/b2d/v$3_$2/recorder/RouteScenario_$1_rep0.log
  [ -f "$log" ] || { echo "missing $log"; return 1; }
  carla_up || return 1
  rm -rf "frames/$3_$1"
  python tools/b2d/render_replay.py --log "$log" --out "frames/$3_$1" --label "$4"
}

for arg in "$@"; do
  id=${arg%%:*}; set_=${arg#*:}
  if [ "${FORCE:-0}" != "1" ] && [ -s "clip_$id.mp4" ] && [ "clip_$id.mp4" -nt "results/b2d/veff_$set_/recorder" ]; then
    echo "== $id: clip exists, skip"; continue
  fi
  echo "== $id ($set_)"
  render "$id" "$set_" base "MPCC base"          || { echo "FAILED: $id base"; continue; }
  render "$id" "$set_" eff  "EFF-Student (ours)" || { echo "FAILED: $id eff"; continue; }
  nb=$(ls "frames/base_$id" | wc -l); ne=$(ls "frames/eff_$id" | wc -l)
  t=$(awk -v a="$nb" -v b="$ne" 'BEGIN{m=(a>b)?a:b; printf "%.2f", m/30+1}')
  ffmpeg -nostdin -y -v error -stats -framerate 30 -i "frames/base_$id/%05d.jpg" -framerate 30 -i "frames/eff_$id/%05d.jpg" \
    -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=120[a];[1:v]tpad=stop_mode=clone:stop_duration=120[b];[a][b]hstack=inputs=2,pad=1920:1080:0:135" \
    -t "$t" -pix_fmt yuv420p "clip_$id.mp4" && echo "== $id -> clip_$id.mp4 ($t s)"
done
pkill -f CarlaUE4
ls -la clip_*.mp4
