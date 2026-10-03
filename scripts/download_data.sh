#!/usr/bin/env bash
# Downloads the LIBERO-Goal demonstrations (Hugging Face, ~6.4 GB) and builds the demonstration
# state bank used by SteerGen (and optionally by CMI).
#
#   scripts/download_data.sh [DATA_DIR]      # default: data/
set -euo pipefail
source "$(dirname "$0")/common.sh"
DATA=${1:-$REPO/data}

uvx --from huggingface_hub hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
  --include "libero_goal/*" --local-dir "$DATA/libero_raw"
sim_python -m resteer.states build-from-raw --raw-dir "$DATA/libero_raw/libero_goal" \
  --out "$DATA/states/libero_goal_demos.hdf5"
