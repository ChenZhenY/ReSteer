# CMI: conditional mutual information between instruction and action

At a state s, `I(A;L|s) = H(A|s) − H(A|s,L)`. It measures how much the instruction changes what the policy does.
Low CMI marks states where switching the instruction barely changes the behavior, a necessary condition
for poor steerability.

## Pipeline (`scripts/run_cmi.sh`)

### 1. States: `python -m resteer.cmi.collect_states`
- For each task: 10 policy rollouts under the task's own instruction, from random layouts.
- Each rollout has a 10-step zero-action warm-up, then 400 policy steps with no early stop.
- The flattened MuJoCo state is recorded after reset, after each warm-up step, and after each policy step.
- State `10 + k` of a rollout is the state at which evaluation would switch the instruction at step k.
  The bank records this offset (`num_warmup_states = 10`).
- Any state bank works. `python -m resteer.states build-from-raw` makes one from LIBERO demonstrations
  (offset 0).

### 2. Samples: `python -m resteer.cmi.sample_actions`
- **State selection.** For k ∈ {0, 5, …, 95}, one bank state `offset + k` is taken from a random rollout long
  enough to contain it (Python `random`, seeded per task).
- **Scene setup.** For each (state, instruction): reset the scene, `set_state`, forward, one zero-action step.
  Then draw 32 action chunks. Resets re-sample fixture placements that `set_state` does not restore, so this
  exact sequence is kept.
- **Sampling.** Servers that advertise batched inference get batches of 8. Stock openpi servers get 32
  sequential requests; each draws fresh noise, so the samples are independent.
- **Action representation.** Each chunk is reduced to its end-point displacement: the sum of the chunk's delta
  xyz actions.

### 3. Entropies (`resteer/cmi/kde.py`)
- `H(A|s,l)`: over the 32 end points for instruction l.
- `H(A|s)`: over the pooled 320 end points of all instructions.
- The estimator is `−mean_i log(mean_j exp(−‖x_i−x_j‖²/2h²) + 1e−8)` with Scott bandwidth `h`, re-estimated per
  sample set. It is a relative entropy in [0, log N] and does not include the Gaussian normalizing constant.
  This is a NumPy port of the paper's torch code, with the same float32 arithmetic.

### 4. CMI: `python -m resteer.cmi.compute_cmi`
- `CMI(s,l) = max(0, H(A|s) − H(A|s,l))`.
- Normalized: `CMI(s,l) / H(A|s)`, the default and what the paper reports. Pass `--no-normalize` for raw values.
- A task's CMI is the mean over its (state, instruction) cells, including its own instruction. The overall CMI
  is the unweighted mean over tasks.
- **Low-CMI tuples** (`low_cmi_tuples.json`, per task and combined): the `--low-cmi-percent` (default 50)
  lowest-CMI cells.
  - Each tuple gives a switching configuration (source task, target instruction, switch step).
  - CMI-guided SRBC and SteerGen selection consume these tuples.
  - `policy_step` is the switch step; `new_prompt_step` is the bank index, kept for paper-era compatibility.

## Differences from the paper-era scripts
- **Prompt case.** The paper-era CMI runs queried Title Case instructions ("Put The Bowl On The Plate"), while
  evaluation and training use lower case. The release queries lower case by default;
  `--prompt-style title` (`scripts/run_cmi.sh --prompt-style title`) reproduces the paper's numbers.
- **Low-CMI selection.** The paper-era selection:
  - thresholded the bank index at 100, which silently dropped the last switch step (policy step 95);
  - kept same-task cells, which produce no switching data.

  The release selects over all steps and excludes same-task cells by default. `--paper-compat` restores the old
  selection.
- **Comparing policies.** The paper computed every model's CMI on one fixed state bank (rollouts of the SteerGen
  policy). To compare policies, pass the same `--states` bank.
