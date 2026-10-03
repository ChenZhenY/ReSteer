#!/usr/bin/env bash
# Parity of CMI sampling with the paper-era script (main_steerability_entropy.py).
#
# Both run against the deterministic fake server in batched mode with Title Case prompts (the
# paper's setting). Requires identical request streams (same scene states, observations and order)
# and entropies equal up to float rounding (torch vs NumPy).
#   tests/parity/cmi_parity.sh OLD_REPO WORK_DIR STATES_HDF5 [TASK_ID]
# STATES_HDF5: a bank whose states are indexed with 10 warm-up states (e.g. from resteer.cmi.collect_states).
set -euo pipefail
OLD_REPO=$(cd "$1" && pwd) WORK=$2 STATES=$3 TASK=${4:-3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$WORK/cmi_parity" && cd "$WORK/cmi_parity" && rm -rf old new ./*.log

serve() { "$REPO/.venv/bin/python" "$REPO/tests/fake_policy_server.py" --batch --port "$1" --log "$2" >"$2.out" 2>&1 & echo $!; }
PID_OLD=$(serve 8775 requests_old.log) PID_NEW=$(serve 8776 requests_new.log)
trap 'kill $PID_OLD $PID_NEW 2>/dev/null' EXIT
sleep 5

(source "$WORK/old_env.sh" && cd "$OLD_REPO/examples/libero_steerability" && python main_steerability_entropy.py \
  --hdf5_file_path "$STATES" --bddl_scene libero_goal --sampling_strategy uniform --num_samples_per_task 10 \
  --batch_size 8 --num_infer_samples 32 --task_id "$TASK" --port 8775 --output_dir "$WORK/cmi_parity/old" \
  > "$WORK/cmi_parity/old.log" 2>&1) &
OLD_RUN=$!
uv run --project "$REPO" python -m resteer.cmi.sample_actions --states "$STATES" --out new --port 8776 \
  --tasks "$TASK" --num-steps 10 --offset 10 --prompt-style title > new.log 2>&1 &
NEW_RUN=$!
wait $OLD_RUN $NEW_RUN

uv run --project "$REPO" python - <<'EOF'
import json, pathlib, sys
import numpy as np
from resteer import tasks
old_req, new_req = pathlib.Path("requests_old.log").read_text(), pathlib.Path("requests_new.log").read_text()
print(f"requests: old {len(old_req.splitlines())}, new {len(new_req.splitlines())}, identical: {old_req == new_req}")
old = {}
for p in pathlib.Path("old").glob("single_state_libero_goal/*.json"):
    d = json.loads(p.read_text())["state_experiment_result"]
    old[(d["step_idx"], "__state__")] = float(np.mean(d["action_chunks_state_entropy"]))
    for block in d["prompt_results"].values():
        old[(d["step_idx"], tasks.get_task(block["prompt"]).name)] = float(block["experiments"][0]["action_entropy"])
new = {}
for p in pathlib.Path("new").glob("task_*/state_*.json"):
    d = json.loads(p.read_text())
    new[(d["state_index"], "__state__")] = d["state_entropy"]
    new.update({(d["state_index"], t): h for t, h in d["prompt_entropy"].items()})
diff = max(abs(old[k] - new[k]) for k in old) if old.keys() == new.keys() else float("inf")
print(f"entropy entries: old {len(old)}, new {len(new)}, max abs difference {diff:.2e}")
sys.exit(0 if old_req == new_req and diff < 1e-4 and old else 1)
EOF
