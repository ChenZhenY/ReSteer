#!/usr/bin/env bash
# Exact parity of the steerability evaluation with the paper-era script.
#
# Runs examples/libero_steerability/main_steerability.py --test_full_rollout (old) and
# resteer.eval.steerability (new) against the deterministic fake policy server, and requires
# identical request streams (same observations, in the same order) and identical rollout outcomes.
#   tests/parity/eval_parity.sh OLD_REPO WORK_DIR STATES_HDF5 [TASK_ID]
# STATES_HDF5: a bank with one `<task>_demo` group per LIBERO-Goal task (only the task names are used).
set -euo pipefail
OLD_REPO=$(cd "$1" && pwd) WORK=$2 STATES=$3 TASK=${4:-3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$WORK/eval_parity" && cd "$WORK/eval_parity" && rm -rf old new ./*.log

serve() { "$REPO/.venv/bin/python" "$REPO/tests/fake_policy_server.py" --port "$1" --log "$2" >"$2.out" 2>&1 & echo $!; }
PID_OLD=$(serve 8765 requests_old.log) PID_NEW=$(serve 8766 requests_new.log)
trap 'kill $PID_OLD $PID_NEW 2>/dev/null' EXIT
sleep 5

(source "$WORK/old_env.sh" && python "$OLD_REPO/examples/libero_steerability/main_steerability.py" \
  --hdf5_file_path "$STATES" --bddl_scene libero_goal --sampling_strategy uniform --num_samples_per_task 10 \
  --num_experiments_per_prompt 2 --output_dir old --task_id "$TASK" --port 8765 --max_steps 300 \
  --test_full_rollout > old.log 2>&1) &
OLD_RUN=$!
uv run --project "$REPO" python -m resteer.eval.steerability --out new --port 8766 --tasks "$TASK" \
  --switch-steps 0 5 --num-repeats 2 > new.log 2>&1 &
NEW_RUN=$!
wait $OLD_RUN $NEW_RUN

uv run --project "$REPO" python - "$TASK" <<'EOF'
import json, pathlib, sys
from resteer import tasks
task = tasks.get_task(int(sys.argv[1]))
old_req, new_req = pathlib.Path("requests_old.log").read_text(), pathlib.Path("requests_new.log").read_text()
print(f"requests: old {len(old_req.splitlines())}, new {len(new_req.splitlines())}, identical: {old_req == new_req}")
old = {}
for p in pathlib.Path("old").glob("single_state_libero_goal/*.json"):
    d = json.loads(p.read_text())["state_experiment_result"]
    for block in d["prompt_results"].values():
        target = tasks.get_task(block["prompt"]).name
        old[(d["step_idx"], target)] = [(e["success"], e["steps_taken"]) for e in block["experiments"]]
new = {}
for p in pathlib.Path("new").glob(f"task_{task.id}_*/step_*.json"):
    d = json.loads(p.read_text())
    for r in d["rollouts"]:
        new.setdefault((d["switch_step"], r["target"]), []).append((r["success"], r["steps_taken"]))
print(f"cells: old {len(old)}, new {len(new)}, outcomes identical: {old == new}")
print(f"successes: old {sum(s for v in old.values() for s, _ in v)}, new {sum(s for v in new.values() for s, _ in v)}")
sys.exit(0 if old_req == new_req and old == new and old else 1)
EOF
