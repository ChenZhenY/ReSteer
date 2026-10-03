import json

import h5py
import numpy as np

from resteer import utils
from resteer.data import episodes
from resteer.steergen import select_by_cmi

PLATE, STOVE, CABINET = "put_the_bowl_on_the_plate", "put_the_bowl_on_the_stove", "put_the_bowl_on_top_of_the_cabinet"


def _bridges():
    return {
        f"{s}__{t}__t{k:03d}": {"source_task": s, "target_task": t, "step": k}
        for s, t in [(PLATE, STOVE), (PLATE, CABINET)]
        for k in range(0, 50, 5)
    }


def _tuple(old, new, step, cmi, policy_step=None):
    t = {"old_prompt": old.replace("_", " ").title(), "new_prompt": new.replace("_", " "), "new_prompt_step": step,
         "mutual_info": cmi}  # fmt: skip
    if policy_step is not None:
        t["policy_step"] = policy_step
    return t


def test_window_and_pair_matching():
    tuples = [_tuple(PLATE, STOVE, 20, 0.01)]
    keep, used = select_by_cmi.select(_bridges(), tuples, None, window=2, step_field="new_prompt_step")
    assert keep == {f"{PLATE}__{STOVE}__t020"}
    keep, _ = select_by_cmi.select(_bridges(), tuples, None, window=5, step_field="new_prompt_step")
    assert keep == {f"{PLATE}__{STOVE}__t{k:03d}" for k in (15, 20, 25)}
    assert used == tuples


def test_lowest_cmi_tuples_first_and_step_field():
    tuples = [_tuple(PLATE, STOVE, 30, 0.5, policy_step=20), _tuple(PLATE, CABINET, 40, 0.1, policy_step=30)]
    keep, used = select_by_cmi.select(_bridges(), tuples, max_tuples=1, window=0, step_field="policy_step")
    assert [t["mutual_info"] for t in used] == [0.1]
    assert keep == {f"{PLATE}__{CABINET}__t030"}
    # Paper-era tuples have no policy_step: fall back to new_prompt_step.
    keep, _ = select_by_cmi.select(_bridges(), [_tuple(PLATE, STOVE, 10, 0.2)], None, 0, "policy_step")
    assert keep == {f"{PLATE}__{STOVE}__t010"}


def _write_bridges(path, names_and_steps):
    with episodes.EpisodeWriter(path) as writer:
        for name, (source, target, step) in names_and_steps.items():
            builder = episodes.EpisodeBuilder()
            for _ in range(2):
                image = np.zeros((256, 256, 3), np.uint8)
                builder.add(action=np.zeros(7), agentview=image, wrist=image, state=np.zeros(8))
            segment = episodes.Segment(0, 2, target, target.replace("_", " "))
            writer.write(
                name, builder, [segment], "steergen", {"source_task": source, "target_task": target, "step": step}
            )


def test_cli_copies_selection_and_writes_manifest(tmp_path):
    names = {f"b{k}": (PLATE, STOVE, k) for k in (0, 10, 20)}
    _write_bridges(tmp_path / "bridges.hdf5", names)
    utils.write_json(
        tmp_path / "cmi" / "task_3" / "low.json", {"low_mutual_info_tuples": [_tuple(PLATE, STOVE, 11, 0.0)]}
    )
    out = tmp_path / "selected.hdf5"
    select_by_cmi.main(select_by_cmi.Args(episodes=(tmp_path / "bridges.hdf5",), out=out, cmi=(tmp_path / "cmi",)))
    with h5py.File(out, "r") as f:
        assert list(f["episodes"]) == ["b10"]
        assert f.attrs["format"] == episodes.FORMAT
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["kept"] == ["b10"] and manifest["num_candidates"] == 3

    select_by_cmi.main(select_by_cmi.Args(episodes=(tmp_path / "bridges.hdf5",), out=out, random_keep=2, seed=1))
    first = json.loads(out.with_suffix(".manifest.json").read_text())["kept"]
    select_by_cmi.main(select_by_cmi.Args(episodes=(tmp_path / "bridges.hdf5",), out=out, random_keep=2, seed=1))
    assert json.loads(out.with_suffix(".manifest.json").read_text())["kept"] == first and len(first) == 2


def test_duplicate_tuples_from_per_task_and_aggregate_files_count_once(tmp_path):
    tuples = [_tuple(PLATE, STOVE, 10, 0.1), _tuple(PLATE, CABINET, 20, 0.2)]
    utils.write_json(tmp_path / "task_3_x" / "low_cmi_tuples.json", {"low_mutual_info_tuples": tuples})
    utils.write_json(tmp_path / "low_cmi_tuples.json", {"low_mutual_info_tuples": tuples})
    utils.write_json(tmp_path / "summary.json", {"cmi": 0.3})
    assert len(select_by_cmi.load_tuples((tmp_path,))) == 2
