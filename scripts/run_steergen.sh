#!/usr/bin/env bash
# SteerGen steering-data generation, end to end on one machine (no policy server needed):
#   1. state bank: demonstration states (step_matched) or replayed + stage-labelled states (stage_matched),
#   2. bridges between tasks, one worker per source task,
#   3. optionally keep only the bridges near low-CMI switch states (output of scripts/run_cmi.sh).
#
#   scripts/run_steergen.sh --out data/steergen                             # the paper's setting
#   scripts/run_steergen.sh --out data/steergen_low_cmi --cmi results/cmi/pi05_libero --max-tuples 300
#   scripts/run_steergen.sh --mode stage_matched --out data/steergen_stage  # experimental
#
# Convert for training (see docs/training.md):
#   scripts/policy.sh convert_to_lerobot <out>/bridges --repo-id resteer/libero_goal_steergen      # or <out>/selected
#
# Options:
#   --raw-dir DIR       raw LIBERO-Goal demos (default: data/libero_raw/libero_goal, see download_data.sh)
#   --mode MODE         step_matched (default) | stage_matched
#   --states BANK       reuse a state bank (default: data/states/libero_goal_demos.hdf5 for step_matched,
#                       <out>/states/libero_goal_regen.hdf5, built and labelled here, for stage_matched)
#   --tasks "0 1 ..."   source task ids (default: all 10); every other task is a target
#   --seed N            default 0; every worker uses it, so results depend on --tasks but not on timing
#   --cmi DIR           compute_cmi output directory for CMI-guided selection
#   --max-tuples N      number of lowest-CMI tuples to use (default: all tuples in --cmi)
#   --prompt-style S    lower (default) | title (the paper-era dataset labels)
#   --render-gpus 0,1   GPUs the workers render on, round-robin (default: MuJoCo's default EGL device)
#   --max-restarts N    restart a failed worker up to N times (default 3); it resumes, with the same output
#   -- ARGS             extra arguments for `python -m resteer.steergen.generate`
set -euo pipefail
source "$(dirname "$0")/common.sh"

RAW_DIR=$REPO/data/libero_raw/libero_goal MODE=step_matched TASKS="0 1 2 3 4 5 6 7 8 9" SEED=0
CMI="" MAX_TUPLES="" PROMPT_STYLE=lower OUT="" STATES="" EXTRA=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --raw-dir) RAW_DIR=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --states) STATES=$2; shift 2 ;;
    --tasks) TASKS=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --cmi) CMI=$2; shift 2 ;;
    --max-tuples) MAX_TUPLES=$2; shift 2 ;;
    --prompt-style) PROMPT_STYLE=$2; shift 2 ;;
    --render-gpus) RENDER_GPUS=$2; shift 2 ;;
    --max-restarts) MAX_RESTARTS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n $OUT ]] || { echo "--out is required" >&2; exit 2; }
[[ -e $OUT/bridges ]] && { echo "$OUT/bridges already exists; choose another --out" >&2; exit 2; }
mkdir -p "$OUT/logs" "$OUT/bridges"

# per_task <label> <tasks> <command...>: one background worker per task; "{task}" in the command is replaced.
per_task() {
  local label=$1 tasks=$2; shift 2
  local pids=() failed=0 i=0
  for task in $tasks; do
    if [[ -n $RENDER_GPUS ]]; then export MUJOCO_EGL_DEVICE_ID=$(render_gpu "$i"); fi
    run_worker "$OUT/logs/${label}_$task.log" sim_python "${@//\{task\}/$task}" &
    pids+=("$!")
    i=$((i + 1))
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
  [[ $failed -eq 0 ]] || { log "$failed $label worker(s) failed, see $OUT/logs"; return 1; }
}

GEN_ARGS=()
if [[ $MODE == step_matched ]]; then
  STATES=${STATES:-$REPO/data/states/libero_goal_demos.hdf5}
  if [[ ! -f $STATES ]]; then
    log "building the demonstration state bank from $RAW_DIR"
    sim_python -m resteer.states build-from-raw --raw-dir "$RAW_DIR" --out "$STATES"
  fi
else
  STATES=${STATES:-$OUT/states/libero_goal_regen.hdf5}
  if [[ ! -f $STATES ]]; then
    log "replaying the LIBERO-Goal demos into a state bank"
    per_task regen "0 1 2 3 4 5 6 7 8 9" -m resteer.steergen.regen_states --raw-dir "$RAW_DIR" --tasks "{task}" \
      --seed "$SEED" --out "$OUT/states/regen_task_{task}.hdf5"
    sim_python -m resteer.states merge --inputs "$OUT"/states/regen_*.hdf5 --out "$STATES"
    sim_python -m resteer.steergen.label_stages "$STATES"
  fi
  GEN_ARGS+=(--raw-dir "$RAW_DIR")
fi

log "generating $MODE bridges (logs in $OUT/logs)"
per_task generate "$TASKS" -m resteer.steergen.generate --mode "$MODE" --states "$STATES" --source-tasks "{task}" \
  --seed "$SEED" --prompt-style "$PROMPT_STYLE" --out "$OUT/bridges/bridges_task_{task}.hdf5" \
  ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}

if [[ -n $CMI ]]; then
  log "keeping bridges near low-CMI switch states from $CMI"
  sim_python -m resteer.steergen.select_by_cmi --episodes "$OUT"/bridges/*.hdf5 --cmi "$CMI" \
    ${MAX_TUPLES:+--max-tuples "$MAX_TUPLES"} --out "$OUT/selected/bridges_low_cmi.hdf5"
fi
log "done: $OUT"
