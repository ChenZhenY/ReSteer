#!/usr/bin/env bash
# Runs the end-to-end launchers against the fake policy server (no GPU): server start/stop, per-task
# workers, resume, scoring and CMI computation.
#   tests/launcher_check.sh WORK_DIR
set -euo pipefail
WORK=$1
REPO=$(cd "$(dirname "$0")/.." && pwd)
export OPENPI_DIR="$REPO/tests/fake_openpi"
# The fake openpi environment has a stand-in scripts/serve_policy.py, not JAX: use the stock-server launch path.
export SERVER_IMPL=openpi
rm -rf "$WORK/launcher_check" && mkdir -p "$WORK/launcher_check"
cd "$REPO"
scripts/eval_steerability.sh --quick --port 8900 --servers 2 --out "$WORK/launcher_check/eval" -- --max-steps 30
test -f "$WORK/launcher_check/eval/summary.json"
scripts/eval_steerability.sh --quick --port 8900 --servers 2 --out "$WORK/launcher_check/eval" -- --max-steps 30  # resumes: nothing left
scripts/eval_steerability.sh --quick --workers-per-task 2 --port 8920 --servers 2 --out "$WORK/launcher_check/eval_sharded" -- --max-steps 30
test -f "$WORK/launcher_check/eval_sharded/task_3_put_the_bowl_on_the_plate/step_050.json"
scripts/run_cmi.sh --tasks "3" --num-rollouts 1 --port 8910 --servers 1 --out "$WORK/launcher_check/cmi" -- --low-cmi-percent 20
test -f "$WORK/launcher_check/cmi/low_cmi_tuples.json"
# A policy server that dies during startup must fail the launcher at once, with the end of its log, instead of
# waiting out the readiness timeout. (The batched server cannot start in the fake environment: no JAX.)
start=$SECONDS
if SERVER_IMPL=batched scripts/eval_steerability.sh --quick --port 8930 --servers 1 --out "$WORK/launcher_check/crash" \
  >"$WORK/launcher_check/crash.log" 2>&1; then
  echo "a crashing policy server was not detected"; exit 1
fi
if ((SECONDS - start > 120)); then echo "a crashing policy server was detected only after $((SECONDS - start))s"; exit 1; fi
grep -q "exited during startup" "$WORK/launcher_check/crash.log"
grep -q "No module named" "$WORK/launcher_check/crash.log"
if pgrep -f fake_policy_server.py >/dev/null; then echo "leftover fake servers"; exit 1; fi
echo "launcher check passed"
