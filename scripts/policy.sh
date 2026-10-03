#!/usr/bin/env bash
# Runs a resteer_policy module inside openpi's environment (training, dataset conversion, export).
#
#   scripts/policy.sh train pi05_libero_resteer_steergen --exp-name steergen --fsdp-devices 2 --batch-size 64
#   scripts/policy.sh convert_to_lerobot --help
#   scripts/policy.sh export_checkpoint checkpoints/pi05_libero_resteer_srbc/srbc/9999 exported/resteer_srbc
set -euo pipefail
source "$(dirname "$0")/common.sh"
module=$1; shift
cd "$REPO"
PYTHONPATH="$REPO/policy${PYTHONPATH:+:$PYTHONPATH}" exec uv run --project "$OPENPI_DIR" \
  python -m "resteer_policy.$module" "$@"
