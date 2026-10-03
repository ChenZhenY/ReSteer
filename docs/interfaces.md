# Interfaces between the simulation side and the policy side

The two sides share no code. These formats are the whole contract.

## Policy server (websocket, msgpack)

This is openpi's protocol; see `resteer/client/policy_client.py`.

1. **Handshake.** The client connects, and the server sends a metadata dict. A server that supports batched
   inference lists `"infer_batch"` in `metadata["methods"]`.
2. **Request.** One observation dict:

   | Key | Type | Content |
   |---|---|---|
   | `observation/image` | uint8 [224, 224, 3] | upright agent-view image |
   | `observation/wrist_image` | uint8 [224, 224, 3] | upright wrist image |
   | `observation/state` | float64 [8] | eef_pos, eef_axis_angle, gripper_qpos |
   | `prompt` | str | the instruction |

   A batched request is a list of such dicts.
3. **Reply.** A dict with `actions`: float [horizon, 7], delta end-effector pose plus gripper in LIBERO units.
   A text frame reports a server error.

## State banks (HDF5)

Written by `resteer.states` and `resteer.cmi.collect_states`:
- `<task_name>_demo/demo_<i>_states`: [T, 79] float64 flattened MuJoCo states `[time, qpos, qvel]`.
- `attrs["num_warmup_states"]`: the index of the state at policy step 0.
- Optional SteerGen labels: `<task_name>_stages/demo_<i>` and `<task_name>_dist_to_go/demo_<i>`.

## Episode files (HDF5): data generation to training

Written by `resteer.data.episodes` (SteerGen, SRBC); read by `policy/resteer_policy/convert_to_lerobot.py`.

- **File attributes:** `format = "resteer-episodes"`, `version = 1`.
- **Per-episode datasets** under `episodes/<name>/`:

  | Dataset | Type | Content |
  |---|---|---|
  | `actions` | float32 [T, 7] | the action executed after observing frame t |
  | `images/agentview`, `images/wrist` | uint8 [T, 256, 256, 3] | upright camera images |
  | `state` | float32 [T, 8] | eef_pos, eef_axis_angle, gripper_qpos |
  | `sim_states` (optional) | float64 [T, 79] | MuJoCo states |

- **Per-episode attributes:**
  - `segments`: JSON list of `{start, end, task, prompt}`. Frames `[start, end)` were generated under instruction
    `prompt`, and each segment becomes one LeRobot episode with task string `prompt`.
  - `source`: `"steergen"` or `"srbc"`.
  - `meta`: JSON provenance.

## LeRobot datasets

This is the openpi LIBERO format:
- features `image`, `wrist_image` (256×256×3), `state` (8), `actions` (7);
- `fps = 10`, `robot_type = "panda"`;
- the per-episode `task` string is used as the prompt (`prompt_from_task=True`).

## Checkpoints

This is openpi's checkpoint format:
- `params/` holds the parameters;
- `assets/physical-intelligence/libero/norm_stats.json` holds the normalization statistics, so the stock
  `pi05_libero` config loads the checkpoint.
