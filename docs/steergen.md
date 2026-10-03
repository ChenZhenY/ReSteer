# SteerGen: steering-data generation

SteerGen creates *bridges*: short trajectories that start in a state of one LIBERO-Goal task (A)
and move the end effector to the pose of a state of another task (B). Each bridge is labelled
with B's instruction.

The bridges are co-trained with the original LIBERO data, which shows how to finish B from such
poses. Together they teach the policy to switch tasks when the instruction changes mid-execution.

The bridges only move the end effector:
- the motion is straight-line in position with SLERP in orientation, tracked in closed loop by the
  OSC controller;
- objects stay where the source state left them;
- no task-success check is run on a bridge; it is kept if the robot reaches the target pose.

## Pipeline

```bash
scripts/download_data.sh                        # raw LIBERO-Goal demos + data/states/libero_goal_demos.hdf5
scripts/run_steergen.sh --out data/steergen     # bridges, one worker per source task
scripts/policy.sh convert_to_lerobot data/steergen/bridges --repo-id resteer/libero_goal_steergen
```

CMI-guided variant: keep only the bridges near the switch states where the policy is least
steerable. This needs a CMI run first (see docs/cmi.md).

```bash
scripts/run_cmi.sh --out results/cmi/pi05_libero
scripts/run_steergen.sh --out data/steergen_cmi --cmi results/cmi/pi05_libero --max-tuples 300
scripts/policy.sh convert_to_lerobot data/steergen_cmi/selected --repo-id resteer/libero_goal_steergen_cmi
```

The individual steps are `python -m resteer.steergen.<module>` CLIs; `--help` lists every option:

| Module | What it does |
|---|---|
| `resteer.states build-from-raw` | Demonstration state bank: the recorded MuJoCo states of every raw demo. |
| `regen_states` | Replays the raw demos (no-op actions skipped) into a state bank; used by `stage_matched`. |
| `label_stages` | Adds stage labels (0 pre-grasp, 1 transport, 2 place, from the gripper opening/closing) and the remaining end-effector path length (`dist_to_go`) to a bank. |
| `generate` | Writes bridges to an episode file (format in `resteer/data/episodes.py`). Re-running it with the same arguments resumes: bridges already in the file are kept, and the result is identical to an uninterrupted run. |
| `select_by_cmi` | Copies the bridges near low-CMI tuples into a new file, plus a manifest. It can also sample randomly (`--random-keep`), as the data-efficiency baseline. |

## Generation modes

**`step_matched`** (default; this produced the paper's SteerGen datasets):
- For every source task A, target task B ≠ A, and step t (default 0–49, the pre-grasp phase of
  LIBERO-Goal demos), take the state at step t of a random demo of A and of B.
- Interpolate from A's end-effector pose to B's, with waypoints ≤ 5 mm and ≤ 0.05 quaternion distance
  apart. The gripper stays open.
- Keep the bridge if the final 7-D pose error (position plus quaternion) is ≤ 0.05.
- The motion is executed in metres/radians and stored in LIBERO's action space: ×20 position, ×2
  rotation, as the paper's post-processing did.

**`stage_matched`** (experimental; ported from the last paper-era generator):
- Only start states in the **transport** stage are used, and only for target tasks that carry the
  same object: the four bowl tasks with each other, and the two wine-bottle tasks with each other.
- The target is the transport-stage state of B that minimizes `dist_to_go + |eef_B − eef_A|` among
  up to 50 random candidates.
- With `--raw-dir`, the source demo is replayed online up to step t, so the object is really
  grasped. The gripper command keeps the start state's closed/open state.
- Keep the bridge if the final position error is ≤ 0.1 m.
- It needs a bank from `regen_states` + `label_stages`; `run_steergen.sh --mode stage_matched`
  builds one.

## Conventions

- **Steps.** A bridge's `step` is an index into the bank it came from:
  - demonstration banks: steps since the demo started;
  - `regen_states` banks: non-no-op actions replayed after 10 settle steps.

  CMI tuples (docs/cmi.md) carry two step fields:
  - `new_prompt_step`: an index into the CMI state bank, whose policy rollouts start with 10 warm-up
    states;
  - `policy_step`: steps since the warm-up.

  `select_by_cmi` matches `|bridge step − tuple step| ≤ --window` (default 2). It uses `policy_step`
  unless `--step-field new_prompt_step`.
- **Randomness.** `random.Random(--seed)` picks demos and candidates. NumPy's global RNG is seeded
  once with `--seed` and fixes each environment's fixture placement: one environment per task pair,
  built in source-major order. The same command gives the same bridges.
- **Episode contents.**
  - One segment over the whole bridge; its prompt is B's instruction (`--prompt-style`).
  - Upright 256×256 agent-view and wrist images; 8-D state; LIBERO-space actions.
  - The MuJoCo state of every frame.
  - `meta`: tasks, step, demos, final error, seed, controller, action units.

## Differences from the paper-era code

The paper-era scripts live in mimiclabs-priv (`libero_steerability` branch @ `096a32a`):
`collect_interpolation_data.py` (v1), `collect_interpolation_data_v3.py`, `label_libero_goal_states.py`,
`playback_libero_collect_sim_state.py` and `delete_high_mi_demos.py`.

| Paper-era behavior | Release | Flag to reproduce |
|---|---|---|
| Dataset labels were Title Case (`Put The Bowl On The Stove`); the policy is queried in lower case | lower case | `--prompt-style title` |
| `dist_to_go` stored the distance to the *next* state (its accumulation loop ran forward) | remaining path length | `label_stages --legacy-dist-to-go` |
| Demo choice unseeded; fixture placement from the unseeded global RNG | seeded (`--seed`) | – |
| Actions scaled by an interactive yes/no post-processing prompt | scaled in code; recorded in `meta` | – |
| Low-CMI selection deleted files and matched bridges by file index (equal to the step only when no bridge had failed) | copies the selection; matches by recorded step | – |
| v3: no candidate list when ≤ 50 feasible states; candidate poses read with a one-element offset into the flattened state; non-state datasets counted as demos; debug PNGs written to the working directory | fixed | – |
| v1 stopped at a `breakpoint()`; regen asked before overwriting and appended the MuJoCo version to the file name | no prompts; the version is stored in the file attributes | – |

We checked before the release that the paper-era scripts and this port are **bitwise identical** on the same
demos and seeds (3 demos per task, two bowl tasks), for:
- the regenerated state banks (states, `act`, `qacc`);
- the stage labels and legacy `dist_to_go`;
- the step-matched bridges (actions, both camera streams, MuJoCo states, 8-D states).

## Notes on the paper datasets

- `interpolation_*_1122` was generated by v1 with t in 0–49 and later downsampled by CMI
  (`*_MI_downsample_0109`). The success threshold used then (0.05 in the script, 1.0 in one
  launcher) is not recorded in the data.
- `…transporting_1210` came from v3.
- The MuJoCo version used for the paper-era state banks was 3.3.7. This release pins 3.2.3, the
  version of the paper's evaluation environment. Regenerated banks can therefore differ slightly
  from the paper's.
