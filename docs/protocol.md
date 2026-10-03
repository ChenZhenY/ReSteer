# Steerability evaluation protocol

Implemented in `resteer/eval/rollout.py` (one rollout) and `resteer/eval/steerability.py` (loop, output,
resume). Constants are those of the paper's evaluation.

## Scene and tasks
- **Scene.** The 10 LIBERO-Goal tasks share one scene, `libero_goal_steerability.bddl`. Its success dictionary
  (`env.step(action, return_success_dict=True)`) reports every task's goal predicate. Tasks, ids (alphabetical)
  and goal keys are listed in `resteer/tasks.py`.
- **Cameras.** Rendered at 512×512.
- **Policy input.** The agent-view and wrist images rotated by 180°, padded and resized to 224×224 (PIL
  bilinear), plus the 8-D state `eef_pos, eef_axis_angle, gripper_qpos`. Instructions are the lower-case task
  names ("put the bowl on the plate").

## One rollout (source task i, target task j, switch step k)
1. `env.reset()` samples a random object layout. Then 10 steps of the all-zero action.
2. For `step` in 0…299:
   - the observation is read;
   - if `step == k`, the instruction becomes task j's and queued actions are dropped;
   - when the queue is empty, the policy is queried and the first 5 actions of the returned chunk are queued;
   - one action is executed.
3. Success: task j's goal predicate holds after some step with `step > k`. With `k = 0` the source instruction
   is never used, and the rollout is a plain task-j rollout.

## Evaluation loop and score
- **Schedule.** For each source task i: k ∈ {0, 5, …, 95} → target j ∈ all 10 tasks → 10 repeats.
- **Score.** A cell (i, k, j) has success rate s(i,k,j). The **steerability score** is the mean of s over the
  18,000 rollouts with j ≠ i (10·9 task pairs × 20 switch steps × 10 repeats).
  - `resteer.eval.score` also reports the mean including j = i. The paper-era aggregation scripts printed that
    mean.
  - It also reports per-task scores, per-switch-step scores and the full success matrices.

## Reproducibility
- **Object layouts** come from NumPy's global RNG, which `env.seed(7)` resets once per source task after the
  environment is created. Only `env.reset()` consumes it, in the fixed loop order above.
  - So the layouts do not depend on the policy.
  - Each `step_KKK.json` stores the RNG state at its end, and a resumed run continues exactly where an
    uninterrupted run would have been.
  - A switch step in progress is saved after every rollout (`step_KKK.json.partial`, with the RNG state), so an
    interrupted worker loses at most one rollout. The launcher restarts failed workers (`--max-restarts`).
  - Parallelize over source tasks (one worker per task), not within a task. `--workers-per-task N` seeds every
    (task, switch step) on its own instead (`--seed-scope step`): statistically the same protocol, other layouts.
- **Rollout errors.**
  - An exception raised by the environment is recorded in the rollout's `error` field and counts as a failure,
    as in the paper.
  - A lost connection to the policy server aborts the run instead. The rollout is not recorded, and it is
    redone on resume. The paper-era code counted such rollouts as failures.
- **Parity with the paper-era script.** The paper-era script (`main_steerability.py --test_full_rollout`) makes
  the same sequence of environment calls. With a deterministic policy it produces identical rollouts; this is
  checked in `tests/parity`.

## Differences from openpi's LIBERO example
The openpi `examples/libero/main.py` evaluation differs in these ways:
- LIBERO-Goal's per-task scenes and fixed init states, instead of random layouts in the shared steerability scene;
- 256×256 rendering instead of 512×512;
- a warm-up action with gripper −1 instead of all zeros;
- upstream LIBERO's 0.03 m `check_ontop` threshold instead of 0.1 m (see `third_party/libero/RESTEER_CHANGES.md`).

Numbers from the two evaluations are therefore not directly comparable.
