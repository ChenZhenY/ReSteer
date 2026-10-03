#!/usr/bin/env bash
# Creates an environment that runs the paper-era research code, for the parity checks in this directory.
#   tests/parity/setup_old_env.sh OLD_REPO WORK_DIR
# OLD_REPO: checkout of the research repository (pi0-anytime-steerability @ 2db0966).
# Afterwards: source WORK_DIR/old_env.sh (sets PATH, PYTHONPATH and LIBERO_CONFIG_PATH for the old code).
set -euo pipefail
OLD_REPO=$(cd "$1" && pwd) WORK=$2
mkdir -p "$WORK"
uv venv --python 3.10 "$WORK/venv_old"
VIRTUAL_ENV="$WORK/venv_old" uv pip install \
  numpy==1.22.4 mujoco==3.2.3 robosuite==1.4.1 bddl==1.0.1 future easydict pyyaml termcolor h5py==3.10.0 \
  scipy==1.10.1 opencv-python==4.6.0.66 pillow==10.4.0 "imageio[ffmpeg]==2.35.1" transforms3d==0.4.2 tyro tqdm \
  matplotlib websockets msgpack typing_extensions cloudpickle "gym==0.25.2" "torch==2.4.1" --extra-index-url https://download.pytorch.org/whl/cpu \
  --override <(echo "pynput ; sys_platform == 'never'")
VIRTUAL_ENV="$WORK/venv_old" uv pip install --no-deps -e "$OLD_REPO/packages/openpi-client"
# The old LIBERO copy prompts for a config on first import; write one pointing at the old package.
mkdir -p "$WORK/libero_cfg"
LIB="$OLD_REPO/third_party/modified_libero/libero/libero"
cat >"$WORK/libero_cfg/config.yaml" <<EOF
benchmark_root: $LIB
bddl_files: $LIB/bddl_files
init_states: $LIB/init_files
datasets: $LIB/../datasets
assets: $LIB/assets
EOF
cat >"$WORK/old_env.sh" <<EOF
export PATH="$WORK/venv_old/bin:\$PATH"
export PYTHONPATH="$OLD_REPO:$OLD_REPO/third_party/modified_libero\${PYTHONPATH:+:\$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$WORK/libero_cfg" MUJOCO_GL=\${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-egl}
EOF
echo "source $WORK/old_env.sh to use the old code"
