import json

import pytest

from resteer import tasks
from resteer.srbc import select


def _write_step(results_dir, task, k, rates):
    task_dir = results_dir / f"task_{task.id}_{task.name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"step_{k:03d}.json").write_text(
        json.dumps({"source_task": task.name, "switch_step": k, "success_rate": rates, "rollouts": []})
    )


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_from_eval_results_filters_and_orders(tmp_path):
    src = tasks.get_task(3)
    rates = {t.name: 0.5 for t in tasks.TASKS}
    rates[tasks.TASKS[0].name] = 0.1  # too low
    rates[tasks.TASKS[1].name] = 1.0  # too high
    rates[tasks.TASKS[2].name] = 0.9  # upper bound is inclusive
    _write_step(tmp_path, src, 5, rates)
    _write_step(tmp_path, src, 0, {tasks.TASKS[9].name: 0.2, tasks.TASKS[4].name: 0.3})
    _write_step(tmp_path, tasks.get_task(1), 0, {tasks.TASKS[2].name: 0.5})

    configs = select.from_eval_results(tmp_path)
    keys = [(c.source, c.target, c.switch_step) for c in configs]
    # Source tasks by id, then switch step, then target id; never target == source.
    assert keys[0] == (tasks.TASKS[1].name, tasks.TASKS[2].name, 0)
    assert keys[1:3] == [(src.name, tasks.TASKS[4].name, 0), (src.name, tasks.TASKS[9].name, 0)]
    step5_targets = [t for s, t, k in keys if s == src.name and k == 5]
    assert step5_targets == [t.name for t in tasks.TASKS if t.id not in (0, 1, 3)]
    assert all(c.success_rate is not None for c in configs)

    assert [c.source for c in select.from_eval_results(tmp_path, source_tasks=(1,))] == [tasks.TASKS[1].name]


def test_compute_cmi_output_directory(tmp_path):
    tuple_ = {"source_task": "put_the_bowl_on_the_plate", "target_task": "turn_on_the_stove",
              "old_prompt": "Put The Bowl On The Plate", "new_prompt": "Turn On The Stove",
              "new_prompt_step": 40, "policy_step": 30, "cmi": 0.0, "mutual_info": 0.0, "demo_name": "demo_2_states"}  # fmt: skip
    combined = {"task_name": "all", "parameters": {}, "low_mutual_info_tuples": [tuple_]}
    _write_json(tmp_path / "low_cmi_tuples.json", combined)
    _write_json(tmp_path / "task_3_put_the_bowl_on_the_plate" / "low_cmi_tuples.json", combined)  # not re-read

    configs = select.from_low_cmi_tuples(tmp_path)
    assert [(c.source, c.target, c.switch_step, c.state_index) for c in configs] == [
        ("put_the_bowl_on_the_plate", "turn_on_the_stove", 30, 40)
    ]
    assert select.from_low_cmi_tuples(tmp_path / "low_cmi_tuples.json") == configs
    assert select.from_low_cmi_tuples(tmp_path, source_tasks=(0,)) == []


def test_missing_tuples_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        select.from_low_cmi_tuples(tmp_path)
