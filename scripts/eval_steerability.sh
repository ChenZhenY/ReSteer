#!/usr/bin/env bash
# Steerability evaluation of a policy on LIBERO-Goal (the paper's protocol), end to end:
# starts openpi policy servers, runs one evaluation worker per source task, prints the score.
#
#   scripts/eval_steerability.sh --out results/pi05_libero                        # released pi0.5 LIBERO policy
#   scripts/eval_steerability.sh --checkpoint exported/resteer_srbc --out results/resteer_srbc
#   scripts/eval_steerability.sh --quick --out results/smoke                      # ~40 rollouts sanity check
#
# Options:
#   --checkpoint DIR|gs://...  policy checkpoint (default: gs://openpi-assets/checkpoints/pi05_libero)
#   --config NAME              openpi config used to load it (default: pi05_libero)
#   --gpus 0,1                 GPUs for policy servers (default: 0)
#   --servers N                number of policy servers (default: 2 per GPU)
#   --render-gpus 2,3          GPUs the simulation workers render on (default: the GPU of their policy server;
#                              rendering competes with inference, so dedicated GPUs help when available)
#   --server batched|openpi    policy server (default batched: resteer_policy.serve_batched; openpi: stock
#                              scripts/serve_policy.py). Same policy; batching gives several times the throughput.
#   --port P                   first server port (default: 8000)
#   --tasks "0 1 ..."          source task ids (default: all 10)
#   --num-repeats N            rollouts per (task, switch step, target) (default: 10)
#   --workers-per-task N       split each task's switch steps over N workers (default 1). With N > 1 every
#                              (task, switch step) is seeded independently (--seed-scope step): statistically
#                              equivalent to the paper's protocol, but not the same object layouts.
#   --quick                    tasks 3 4, switch steps 0 50, 1 repeat
#   --max-restarts N           restart a failed worker up to N times; it resumes where it stopped (default 3)
#   -- ARGS                    extra arguments for `python -m resteer.eval.steerability`
# Interrupted runs resume where they stopped when re-run with the same --out.
set -euo pipefail
source "$(dirname "$0")/common.sh"

CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero CONFIG=pi05_libero GPUS=0 SERVERS="" PORT=8000
TASKS="0 1 2 3 4 5 6 7 8 9" STEPS="$(seq -s ' ' 0 5 95)" REPEATS=10 WPT=1 OUT="" EXTRA=()
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
    --num-repeats) REPEATS=$2; shift 2 ;;
    --workers-per-task) WPT=$2; shift 2 ;;
    --max-restarts) MAX_RESTARTS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --quick) TASKS="3 4"; REPEATS=1; STEPS="0 50"; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n $OUT ]] || { echo "--out is required" >&2; exit 2; }
IFS=',' read -r -a gpu_list <<<"$GPUS"
SERVERS=${SERVERS:-$((2 * ${#gpu_list[@]}))}
mkdir -p "$OUT/logs"
trap stop_servers EXIT

start_servers "$SERVERS" "$GPUS" "$PORT" "$CHECKPOINT" "$CONFIG" "$OUT/logs"
i=0 worker_pids=()
read -r -a step_list <<<"$STEPS"
scope=task; [[ $WPT -gt 1 ]] && scope=step
for task in $TASKS; do
  for ((w = 0; w < WPT; w++)); do
    steps=()  # round-robin share of the switch steps
    for ((j = w; j < ${#step_list[@]}; j += WPT)); do steps+=("${step_list[$j]}"); done
    [[ ${#steps[@]} -gt 0 ]] || continue
    port=${PORTS[$((i % ${#PORTS[@]}))]}
    export MUJOCO_EGL_DEVICE_ID=$(render_gpu "$i")
    logfile="$OUT/logs/task_${task}_worker${w}.log"
    log "task $task steps ${steps[*]} -> server $port (log $logfile)"
    run_worker "$logfile" sim_python -m resteer.eval.steerability --out "$OUT" --port "$port" --tasks "$task" \
      --switch-steps "${steps[@]}" --seed-scope "$scope" --num-repeats "$REPEATS" ${EXTRA[@]+"${EXTRA[@]}"} &
    worker_pids+=("$!")
    i=$((i + 1))
  done
done
failed=0
for pid in "${worker_pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
stop_servers
[[ $failed -eq 0 ]] || log "$failed worker(s) failed; see $OUT/logs (re-run the same command to resume)"
sim_python -m resteer.eval.score "$OUT"
exit $failed
