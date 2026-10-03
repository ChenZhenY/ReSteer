# SRBC: self-refining behavioral cloning

SRBC turns the policy's own successful task switches into training data. The collecting policy (in the paper,
π0.5 after SteerGen co-training) is rolled out on switch configurations where it is steerable but unreliable.
Every rollout that reaches the new task's goal after the switch is kept. Fine-tuning on these rollouts together
with LIBERO and SteerGen data gives the SRBC policy (`pi05_libero_resteer_srbc`, see docs/training.md).

## Pipeline

```bash
# 1. switch configurations: a steerability evaluation of the collecting policy (docs/protocol.md)
#    and/or its CMI analysis (docs/cmi.md)
scripts/eval_steerability.sh --checkpoint CKPT --out results/steergen
scripts/run_cmi.sh --checkpoint CKPT --out results/cmi/steergen
# 2. collect successful switches, one worker per source task
scripts/run_srbc.sh --checkpoint CKPT --source success_rate --results results/steergen --out data/srbc/success_rate
scripts/run_srbc.sh --checkpoint CKPT --source cmi --results results/cmi/steergen --out data/srbc/cmi
# 3. mix the two pools into one training set
uv run python -m resteer.srbc.build_dataset sample --db1 data/srbc/success_rate/database.json \
    --db2 data/srbc/cmi/database.json --total-demos 2000 --ratio 0.5 --out data/srbc/mix.json
uv run python -m resteer.srbc.build_dataset combine --database data/srbc/mix.json --out data/srbc/srbc.hdf5
# 4. convert for openpi
scripts/policy.sh convert_to_lerobot data/srbc/srbc.hdf5 --repo-id resteer/libero_goal_srbc
```

Each step is a CLI; `--help` lists every option.

| Module | What it does |
|---|---|
| `resteer.srbc.select` | Lists the switch configurations a results directory yields (a dry run of step 2). |
| `resteer.srbc.collect` | Rolls out the configurations and writes successful switches to episode files. |
| `resteer.srbc.build_dataset` | `index`, `sample` and `combine` episode files into one training set. |
| `resteer_policy.convert_to_lerobot` | Episode files to a LeRobot dataset (runs in openpi's environment, via `scripts/policy.sh`). |

### 1. Switch configurations (`select`)
A configuration is (source task, target task, switch step). There are two sources.

- **`success_rate`**: cells of a steerability evaluation (the `task_*/step_*.json` of `resteer.eval.steerability`).
  - A cell is kept when its success rate lies in `[--min-success-rate, --max-success-rate]` (default
    `[0.2, 0.9]`) and its target differs from its source.
  - These are the switches the policy can make, but not reliably.
  - Order: source task, then switch step, then target task.
- **`cmi`**: the low-CMI tuples of `resteer.cmi.compute_cmi`, i.e. the switch points where the instruction
  barely changes what the policy does.
  - Pass the output directory (its combined `low_cmi_tuples.json` is read) or a tuples file.
  - Paper-era result directories (`task_<N>/step_threshold_100/*_low_mi_tuples.json`) are also accepted.
  - A tuple indexes a state of a rollout state bank. The switch step is its `policy_step`: the bank index minus
    the 10 warm-up states, i.e. the step at which the CMI was measured.
  - Tuples whose target equals the source are kept unless `--no-include-same-task` is passed. compute_cmi
    already drops them by default; paper-era files contain them.

### 2. Collection (`collect`)
- **Rollouts.** Each rollout is an evaluation rollout (`resteer.eval.rollout`): a fresh random scene, 10
  zero-action warm-up steps, and 512×512 rendering. The policy follows the source instruction and is switched
  to the target instruction at the configuration's step, for at most `--max-steps` (default 300) steps.
- **Success.** The target task's goal must hold at a step after the switch.
- **Limits.** A configuration stops after `--successes-per-config` (5) successes or `--max-attempts-per-config`
  (50) attempts. At most `--max-total-rollouts` (10000) rollouts run per source task.
- **Seeding.** Each source task uses one environment, seeded once with `--seed` (7). Object layouts differ
  between attempts, and a task's data does not depend on which other tasks share the process.
- **Output.** Episodes go to `<out>/<source_task>.hdf5` (format in docs/interfaces.md). Each episode holds every
  step's observation and the action taken, in two segments:
  - frames `[0, k)`, labelled with the source instruction;
  - frames `[k, T)`, labelled with the target instruction;
  - a switch at step 0 gives a single, target-labelled segment.
  - Episode `meta` records the configuration, its success rate or CMI, the attempt index and the step of success.
- **Prompts.** The policy is always queried with lower-case instructions. `--prompt-style title` stores Title
  Case labels instead, as the paper-era datasets did.
- **Resuming.** Re-running the same command resumes. Successes are counted from the episode files. Attempts and
  the object-layout RNG state come from `<out>/<source_task>.progress.json`, so an interrupted collection
  continues with the same layouts as an uninterrupted one.

### 3. Dataset building (`build_dataset`)
- `index --data-dir D` lists the episodes of every episode file below `D` in `D/database.json`, grouped by
  source task. Paths are relative to the JSON file.
- `sample --db1 A --db2 B --total-demos N --ratio r` draws `N` episodes, a fraction `r` from `A`, with
  `random.Random(--seed 42)`. Strategies:
  - `random`: uniform within each database;
  - `stratified`: per source task, proportional to that task's share of the pooled episodes;
  - `balanced`: an equal share per source task.

  The arithmetic, including rounding and clamping to the available episodes, is that of the paper's sampler.
- `combine --database J --out F` copies the listed episodes into one file as `000000`, `000001`, …, each with
  its origin in `meta`.

### 4. LeRobot conversion
Every segment becomes one LeRobot episode whose task string is the segment's label. The features are those of
openpi's LIBERO datasets:
- `image` and `wrist_image` (256×256×3);
- `state` (8) and `actions` (7);
- 10 fps.

Episodes are streamed one at a time. `--drop-zero-switch` skips SRBC episodes switched at step 0, as the paper's
converter did.

## Differences from the paper-era scripts
`scripts/run_srbc.sh --paper` (i.e. `collect --paper-compat` plus the paper's launch settings: 400 steps,
25 attempts per configuration, success rate ≤ 0.7) reproduces the paper-era collectors exactly.
`tests/parity/srbc_parity.py` checks that the paper-era scripts and this collector send the same policy requests
and record the same frames.

| Paper-era behavior | Default now | Reproduce with |
|---|---|---|
| A new environment seeded with 7 for every rollout: every attempt started from the same object layout | one environment per source task, seeded once | `--paper-compat` |
| 224×224 rendering, no warm-up steps (unlike evaluation) | evaluation settings: 512×512, 10 warm-up steps | `--paper-compat` |
| Frames stored the observation of the most recent policy query: in 4 of 5 steps, the observation of up to 4 steps earlier paired with the current action | the observation of every step | `--paper-compat` |
| CMI tuples switched at the raw state-bank index, 10 steps later than the state at which the CMI was measured | the policy step of that state | `--paper-compat` |
| Labels in Title Case ("Put The Bowl On The Plate") | lower case, as the policy is queried | `--prompt-style title` (implied by `--paper-compat`) |
| 250 steps by default; the launch scripts passed 400 | 300, as in evaluation | `--max-steps` |
| Switches at step 0 were dropped when converting to LeRobot | kept | `convert_to_lerobot --drop-zero-switch` |
| A policy-server error ended the rollout as a failure | the collection stops (re-run to resume) | none |

The paper-era pipeline wrote one HDF5 file per episode, indexed them with absolute paths, and loaded a whole
dataset into memory for conversion. The release writes one file per source task, uses relative paths, and
streams episodes.
