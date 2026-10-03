#!/usr/bin/env bash
# SRBC data collection, end to end:
#   1. starts policy servers for the collecting policy (in the paper: the SteerGen-finetuned pi0.5),
#   2. rolls out every switch configuration and keeps the successful switches (one worker per source task),
#   3. indexes the episodes (<out>/database.json).
#
#   # switch configurations from a steerability evaluation of the collecting policy (success rate 0.2-0.9):
#   scripts/run_srbc.sh --checkpoint CKPT --source success_rate --results results/steergen --out data/srbc/success_rate
#   # or from its low-CMI switch points (scripts/run_cmi.sh output):
#   scripts/run_srbc.sh --checkpoint CKPT --source cmi --results results/cmi/steergen --out data/srbc/cmi
#
# Then mix the pools and convert them for training (docs/srbc.md):
#   uv run python -m resteer.srbc.build_dataset sample --db1 data/srbc/success_rate/database.json \
#       --db2 data/srbc/cmi/database.json --total-demos 2000 --ratio 0.5 --out data/srbc/mix.json
#   uv run python -m resteer.srbc.build_dataset combine --database data/srbc/mix.json --out data/srbc/srbc.hdf5
#   scripts/policy.sh convert_to_lerobot data/srbc/srbc.hdf5 --repo-id resteer/libero_goal_srbc
#
# Options:
#   --checkpoint, --config, --gpus, --servers, --port, --tasks   as in eval_steerability.sh
#   --source success_rate|cmi   where the switch configurations come from (required)
#   --results PATH              success_rate: steerability results directory;
#                               cmi: resteer.cmi.compute_cmi output directory or a tuples JSON (required)
#   --workers-per-task N        split each task's switch configurations over N workers (default 1)
#   --render-gpus 2,3           GPUs the simulation workers render on (default: their server's GPU)
#   --max-restarts N            restart a failed worker up to N times; it resumes where it stopped (default 3)
#   --paper                     the paper's collector and launch settings: --paper-compat, 400 steps,
#                               25 attempts per configuration, and success rate <= 0.7 (success_rate source)
#   -- ARGS                     extra arguments for `python -m resteer.srbc.collect`
# Interrupted runs resume where they stopped when re-run with the same --out.
set -euo pipefail
source "$(dirname "$0")/common.sh"

CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero CONFIG=pi05_libero GPUS=0 SERVERS="" PORT=8000
TASKS="0 1 2 3 4 5 6 7 8 9" SOURCE="" RESULTS="" PAPER=0 WPT=1 OUT="" EXTRA=()
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
    --source) SOURCE=$2; shift 2 ;;
    --results) RESULTS=$2; shift 2 ;;
    --paper) PAPER=1; shift ;;
    --workers-per-task) WPT=$2; shift 2 ;;
    --max-restarts) MAX_RESTARTS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n $OUT && -n $SOURCE && -n $RESULTS ]] || { echo "--source, --results and --out are required" >&2; exit 2; }
if [[ $PAPER -eq 1 ]]; then
  EXTRA=(--paper-compat --max-steps 400 --max-attempts-per-config 25 ${EXTRA[@]+"${EXTRA[@]}"})
  [[ $SOURCE == success_rate ]] && EXTRA+=(--max-success-rate 0.7)
fi
IFS=',' read -r -a gpu_list <<<"$GPUS"
SERVERS=${SERVERS:-$((2 * ${#gpu_list[@]}))}
mkdir -p "$OUT/logs"
# stop_servers can return nonzero for servers that already exited; never let that abort the run.
trap 'stop_servers || true' EXIT

start_servers "$SERVERS" "$GPUS" "$PORT" "$CHECKPOINT" "$CONFIG" "$OUT/logs"
i=0 worker_pids=()
for task in $TASKS; do
  for ((shard = 0; shard < WPT; shard++)); do
    port=${PORTS[$((i % ${#PORTS[@]}))]}
    export MUJOCO_EGL_DEVICE_ID=$(render_gpu "$i")
    logfile="$OUT/logs/task_${task}_shard${shard}.log"
    log "task $task shard $shard/$WPT -> server $port (log $logfile)"
    run_worker "$logfile" sim_python -m resteer.srbc.collect --source "$SOURCE" --results "$RESULTS" --out "$OUT" \
      --port "$port" --tasks "$task" --num-shards "$WPT" --shard "$shard" ${EXTRA[@]+"${EXTRA[@]}"} &
    worker_pids+=("$!")
    i=$((i + 1))
  done
done
failed=0
for pid in "${worker_pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
stop_servers || true
[[ $failed -eq 0 ]] || log "$failed worker(s) failed; see $OUT/logs (re-run the same command to resume)"
sim_python -m resteer.srbc.build_dataset index --data-dir "$OUT"
exit $failed
