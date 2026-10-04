#!/usr/bin/env bash
# Serves a policy with upstream openpi (foreground), for use with the python entry points directly.
#
#   scripts/serve_policy.sh                                            # released pi0.5 LIBERO policy
#   scripts/serve_policy.sh --checkpoint exported/resteer_srbc --port 8001 --gpu 1
#
# ReSteer checkpoints load with the stock pi05_libero config (see policy/resteer_policy/export_checkpoint.py).
set -euo pipefail
source "$(dirname "$0")/common.sh"

CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero CONFIG=pi05_libero PORT=8000 GPU=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
CUDA_VISIBLE_DEVICES=$GPU exec uv run --project "$OPENPI_DIR" python "$OPENPI_DIR/scripts/serve_policy.py" \
  --port "$PORT" policy:checkpoint --policy.config="$CONFIG" --policy.dir="$CHECKPOINT"
