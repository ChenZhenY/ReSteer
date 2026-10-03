#!/usr/bin/env bash
# Resuming an interrupted evaluation must reproduce the uninterrupted run exactly (same object layouts,
# same observations). Uses the deterministic fake policy server.
#   tests/parity/resume_parity.sh WORK_DIR
set -euo pipefail
WORK=$1
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$WORK/resume_parity" && cd "$WORK/resume_parity" && rm -rf full resumed ./*.log
"$REPO/.venv/bin/python" "$REPO/tests/fake_policy_server.py" --port 8785 --log full.log >server_full.out 2>&1 &
S1=$!
"$REPO/.venv/bin/python" "$REPO/tests/fake_policy_server.py" --port 8786 --log resumed.log >server_resumed.out 2>&1 &
S2=$!
trap 'kill $S1 $S2 2>/dev/null' EXIT
sleep 3
eval_run() { uv run --project "$REPO" python -m resteer.eval.steerability --tasks 3 --targets 0 3 --num-repeats 2 --max-steps 60 "$@" >>eval.log 2>&1; }
eval_run --out full --port 8785 --switch-steps 0 5 10
eval_run --out resumed --port 8786 --switch-steps 0      # "interrupted" after the first switch step
eval_run --out resumed --port 8786 --switch-steps 0 5 10 # resumes at step 5 from the saved RNG state
if cmp -s full.log resumed.log; then echo "resume parity: identical ($(wc -l <full.log) requests)"; else
  echo "resume parity: DIFFERENT"; exit 1; fi
