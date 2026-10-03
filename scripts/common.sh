# Shared helpers for the ReSteer launch scripts. Source it; do not execute it.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$REPO/third_party/openpi}"
export GIT_LFS_SKIP_SMUDGE=1
# One BLAS/numba thread per simulation worker; the workers themselves run in parallel.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"

SERVER_PIDS=()

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

# resteer (simulation) environment
sim_python() { uv run --project "$REPO" python "$@"; }

# Policy server implementation: "batched" (resteer_policy.serve_batched: openpi's policy with requests from
# concurrent workers batched through the model, several times the throughput) or "openpi" (openpi's own
# scripts/serve_policy.py, unmodified). Both serve the same policy; see policy/resteer_policy/serve_batched.py.
SERVER_IMPL="${SERVER_IMPL:-batched}"

# start_server <gpu> <port> <checkpoint> <config> <mem_fraction> <log_file>
# Starts a policy server in the background and records its PID in SERVER_PIDS.
start_server() {
  local gpu=$1 port=$2 checkpoint=$3 config=$4 mem=$5 logfile=$6
  mkdir -p "$(dirname "$logfile")"
  if [[ $SERVER_IMPL == batched ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION="$mem" PYTHONPATH="$REPO/policy" \
      uv run --project "$OPENPI_DIR" python -m resteer_policy.serve_batched --port "$port" \
      --config "$config" --checkpoint "$checkpoint" >"$logfile" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION="$mem" \
      uv run --project "$OPENPI_DIR" python "$OPENPI_DIR/scripts/serve_policy.py" --port "$port" \
      policy:checkpoint --policy.config="$config" --policy.dir="$checkpoint" >"$logfile" 2>&1 &
  fi
  SERVER_PIDS+=("$!")
  log "$SERVER_IMPL policy server on GPU $gpu port $port (pid $!, log $logfile)"
}

# wait_for_server <port> [timeout_s] - waits until the server answers /healthz.
wait_for_server() {
  local port=$1 timeout=${2:-1800} start=$SECONDS
  until curl -sf "http://127.0.0.1:$port/healthz" >/dev/null; do
    if (( SECONDS - start > timeout )); then log "server on port $port not ready after ${timeout}s"; return 1; fi
    sleep 5
  done
}

# render_gpu <worker_index>: the GPU a simulation worker renders on (RENDER_GPUS if set, else its server's GPU).
RENDER_GPUS="${RENDER_GPUS:-}"
render_gpu() {
  if [[ -n $RENDER_GPUS ]]; then
    local list; IFS=',' read -r -a list <<<"$RENDER_GPUS"
    echo "${list[$(($1 % ${#list[@]}))]}"
  else
    echo "${PORT_GPUS[$(($1 % ${#PORT_GPUS[@]}))]}"
  fi
}

# Simulation workers that fail are restarted up to MAX_RESTARTS times. They resume from their saved results, so a
# restart does not change the outcome. Under heavy rendering load the NVIDIA EGL driver can abort a process (seen
# on H100s, which have little graphics hardware); a restart recovers from that.
MAX_RESTARTS="${MAX_RESTARTS:-3}"

# run_worker <log_file> <command...>: runs a worker, restarting it when it fails.
run_worker() {
  local logfile=$1 attempt=0
  shift
  : >"$logfile"
  until "$@" >>"$logfile" 2>&1; do
    if ((attempt >= MAX_RESTARTS)); then return 1; fi
    attempt=$((attempt + 1))
    log "worker failed, restart $attempt/$MAX_RESTARTS (log $logfile)"
    echo "=== restart $attempt/$MAX_RESTARTS ===" >>"$logfile"
    sleep 10
  done
}

# Never fails (safe under `set -e` and in EXIT traps, also when servers already exited).
stop_servers() {
  local pid
  for pid in ${SERVER_PIDS[@]+"${SERVER_PIDS[@]}"}; do
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  SERVER_PIDS=()
  return 0
}

# Fraction of a GPU's memory that the policy servers on it take together (JAX preallocates it). Lower it (e.g. 0.6)
# when simulation workers also render on the servers' GPUs.
SERVER_MEM_FRACTION="${SERVER_MEM_FRACTION:-0.9}"

# start_servers <num_servers> <gpus_csv> <first_port> <checkpoint> <config> <log_dir>
# Places servers round-robin on the GPUs and waits until all are ready. Sets PORTS and PORT_GPUS
# (the GPU of each server; workers render on it via MUJOCO_EGL_DEVICE_ID).
start_servers() {
  local n=$1 gpus=$2 first_port=$3 checkpoint=$4 config=$5 log_dir=$6
  IFS=',' read -r -a gpu_list <<<"$gpus"
  local per_gpu=$(( (n + ${#gpu_list[@]} - 1) / ${#gpu_list[@]} ))
  local mem
  mem=$(awk -v k="$per_gpu" -v f="$SERVER_MEM_FRACTION" 'BEGIN { printf "%.2f", f / k }')
  PORTS=() PORT_GPUS=()
  for ((i = 0; i < n; i++)); do
    local port=$((first_port + i)) gpu=${gpu_list[$((i % ${#gpu_list[@]}))]}
    start_server "$gpu" "$port" "$checkpoint" "$config" "$mem" "$log_dir/server_$port.log"
    PORTS+=("$port") PORT_GPUS+=("$gpu")
  done
  for port in "${PORTS[@]}"; do wait_for_server "$port" || { stop_servers; return 1; }; done
  log "${#PORTS[@]} policy server(s) ready"
}
