#!/usr/bin/env bash
# Creates the two environments and checks that MuJoCo can render:
#   - simulation side (this repo, Python 3.10): uv sync
#   - policy side (upstream openpi submodule, Python 3.11): uv sync in third_party/openpi
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO"

command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
if [[ $(uname) == Linux ]]; then
  # A C compiler builds openpi's evdev dependency; OpenCV needs libGL and GLib; MuJoCo renders through libEGL.
  missing=()
  command -v cc >/dev/null || missing+=(build-essential)
  for pair in libGL.so.1:libgl1 libEGL.so.1:libegl1 libglib-2.0.so.0:libglib2.0-0; do
    ldconfig -p | grep -q "${pair%%:*}" || missing+=("${pair##*:}")
  done
  if ((${#missing[@]})); then
    echo "Missing system packages. On Ubuntu/Debian: sudo apt-get install -y ${missing[*]}" >&2
    exit 1
  fi
fi
git submodule update --init third_party/openpi
log "simulation environment"
uv sync
log "openpi environment"
(cd "$OPENPI_DIR" && uv sync)
log "checking offscreen rendering (MUJOCO_GL=${MUJOCO_GL:-egl})"
sim_python - <<'EOF'
import numpy as np
from resteer.sim import libero as sim
env = sim.make_env(seed=0)
sim.reset(env, num_warmup_steps=1)
image = sim.camera_images(sim.get_observation(env))[0]
assert image.shape == (512, 512, 3) and image.max() > 0, "rendering returned an empty image"
env.close()
print("LIBERO steerability scene renders correctly.")
EOF
