#!/usr/bin/env bash
# CMI proxy of a policy, end to end:
#   1. collect the states the policy visits while following each task (skipped with --states),
#   2. sample 32 action chunks per state and instruction and estimate entropies,
#   3. compute CMI per (state, instruction) and select low-CMI switching configurations.
#
#   scripts/run_cmi.sh --out results/cmi/pi05_libero
#   scripts/run_cmi.sh --checkpoint exported/resteer_srbc --states results/cmi/pi05_libero/states.hdf5 \
#       --out results/cmi/resteer_srbc          # compare policies on the same state set
#
# Options: --checkpoint, --config, --gpus, --servers, --port, --tasks, --server, --render-gpus, --max-restarts
#          (as in eval_steerability.sh),
#          --states BANK (reuse a state bank), --num-rollouts N (default 10), --prompt-style title
#          (reproduces the paper's CMI runs), -- ARGS (extra args for resteer.cmi.compute_cmi).
set -euo pipefail
source "$(dirname "$0")/common.sh"

CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero CONFIG=pi05_libero GPUS=0 SERVERS="" PORT=8000
TASKS="0 1 2 3 4 5 6 7 8 9" STATES="" ROLLOUTS=10 PROMPT_STYLE=lower OUT="" EXTRA=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --server) SERVER_IMPL=$2; shift 2 ;;
    --render-gpus) RENDER_GPUS=$2; shift 2 ;;
    --servers) SERVERS=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --tasks) TASKS=$2; shift 2 ;;
    --states) STATES=$2; shift 2 ;;
    --num-rollouts) ROLLOUTS=$2; shift 2 ;;
    --prompt-style) PROMPT_STYLE=$2; shift 2 ;;
    --max-restarts) MAX_RESTARTS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n $OUT ]] || { echo "--out is required" >&2; exit 2; }
IFS=',' read -r -a gpu_list <<<"$GPUS"
SERVERS=${SERVERS:-$((2 * ${#gpu_list[@]}))}
mkdir -p "$OUT/logs"
trap stop_servers EXIT

# run_per_task <label> <command...>: one background worker per task, round-robin over servers.
run_per_task() {
  local label=$1; shift
  local i=0 pids=() failed=0
  for task in $TASKS; do
    export MUJOCO_EGL_DEVICE_ID=$(render_gpu "$i")
    run_worker "$OUT/logs/${label}_task_$task.log" sim_python "$@" --port "${PORTS[$((i % ${#PORTS[@]}))]}" --tasks "$task" &
    pids+=("$!"); i=$((i + 1))
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
  [[ $failed -eq 0 ]] || { log "$failed $label worker(s) failed, see $OUT/logs"; return 1; }
}

start_servers "$SERVERS" "$GPUS" "$PORT" "$CHECKPOINT" "$CONFIG" "$OUT/logs"
if [[ -z $STATES ]]; then
  log "collecting policy rollouts ($ROLLOUTS per task)"
  run_per_task collect -m resteer.cmi.collect_states --out-dir "$OUT/states" --num-rollouts "$ROLLOUTS"
  sim_python -m resteer.states merge --inputs "$OUT"/states/rollouts_*.hdf5 --out "$OUT/states.hdf5"
  STATES="$OUT/states.hdf5"
fi
log "sampling actions at the states of $STATES"
run_per_task sample -m resteer.cmi.sample_actions --states "$STATES" --out "$OUT" --prompt-style "$PROMPT_STYLE"
stop_servers
sim_python -m resteer.cmi.compute_cmi "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
