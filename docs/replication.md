# Reproducing the paper's LIBERO results with this release

This page records one end-to-end run of the release pipeline on a single 8×H100 node, from data generation to
fine-tuned policies, next to the paper's numbers. It uses the launchers and modules described in the README.

**Pipeline:**
1. SteerGen data.
2. π0.5 + SteerGen fine-tuning.
3. Steerability and CMI.
4. SRBC data, collected with the SteerGen policy.
5. SRBC fine-tuning.
6. Steerability and CMI again.

## Setup
- **Hardware and software:** one 8×H100 80 GB node, Ubuntu 22.04, NVIDIA driver 580.95. Simulation in the `resteer`
  env (Python 3.10, mujoco 3.2.3, robosuite 1.4.1); policies in upstream openpi @ `215abfb` (unmodified).
- **Evaluation:**
  - the full protocol: 10 source tasks × 20 switch steps (0–95) × 10 target tasks × 10 repeats (20,000 rollouts per
    policy), batched policy servers;
  - `--workers-per-task`, so each (task, switch step) is seeded independently (statistically the same protocol as
    the paper, but not the same object layouts).
- **SteerGen data:** `step_matched` bridges for all 90 ordered task pairs × steps 0–49 = 4,500 bridges.
  - As in the paper's dataset, no final-pose filter was applied (`--success-threshold 1000`); that check was added
    to the generator after the paper's data was made.
  - Labels are lower case.
- **SteerGen fine-tuning:** `pi05_libero_resteer_steergen`, LIBERO + SteerGen (weights 1:3), 2,000 steps, batch 64
  (FSDP over 4 GPUs; the paper used 2×H100 with the same batch).
- **SRBC data:** successful switches of the SteerGen policy from two pools of switch configurations
  ([docs/srbc.md](srbc.md)):
  - success-rate pool: (source, target ≠ source, switch step) cells with SteerGen-policy success rate in [0.2, 0.7];
  - CMI pool: the SteerGen policy's low-CMI configurations (bottom 50%, lower-case instructions);
  - collection: up to 25 attempts and 5 successes per configuration, 400 steps;
  - 2,000 episodes sampled, half from each pool.
- **SRBC fine-tuning:** `pi05_libero_resteer_srbc`, LIBERO + SteerGen + SRBC (1:3:15), initialized from SteerGen,
  3,000 steps, batch 128, FSDP 8 (the paper's 8×A40 recipe).
- **CMI:**
  - state bank: 10 rollouts per task of the SteerGen policy (as in the paper);
  - 20 states per task, 32 samples per instruction, normalized CMI;
  - reported with lower-case instructions (release default) and Title Case (the paper's CMI runs).

## Commands
Paths are relative to the repository. `<steergen>` is `checkpoints/pi05_libero_resteer_steergen/steergen/1999`, and
`<srbc>` is `checkpoints/pi05_libero_resteer_srbc/srbc/2999`.

```bash
# SteerGen data and policy
scripts/download_data.sh
scripts/run_steergen.sh --out data/steergen -- --success-threshold 1000
scripts/policy.sh convert_to_lerobot data/steergen/bridges --repo-id resteer/libero_goal_steergen
scripts/policy.sh train pi05_libero_resteer_steergen --exp-name steergen --fsdp-devices 4 --batch-size 64

# steerability (repeat with --checkpoint <steergen> and <srbc>) and CMI on one state bank
scripts/eval_steerability.sh --gpus 4,5,6 --render-gpus 5,6 --workers-per-task 4 --out results/eval/pi05_libero
scripts/run_cmi.sh --checkpoint <steergen> --out results/cmi/steergen
scripts/run_cmi.sh --states results/cmi/steergen/states.hdf5 --out results/cmi/pi05_libero
# add --prompt-style title to run_cmi.sh for the paper's Title Case CMI

# SRBC data and policy
scripts/run_srbc.sh --checkpoint <steergen> --source success_rate --results results/eval/steergen \
    --out data/srbc/success_rate -- --max-steps 400 --max-attempts-per-config 25 --max-success-rate 0.7
scripts/run_srbc.sh --checkpoint <steergen> --source cmi --results results/cmi/steergen \
    --out data/srbc/cmi -- --max-steps 400 --max-attempts-per-config 25
uv run python -m resteer.srbc.build_dataset sample --db1 data/srbc/success_rate/database.json \
    --db2 data/srbc/cmi/database.json --ratio 0.5 --total-demos 2000 --out data/srbc/sampled_2000.json
uv run python -m resteer.srbc.build_dataset combine --database data/srbc/sampled_2000.json --out data/srbc/srbc.hdf5
scripts/policy.sh convert_to_lerobot data/srbc/srbc.hdf5 --repo-id resteer/libero_goal_srbc
scripts/policy.sh train pi05_libero_resteer_srbc --exp-name srbc --weight-loader.params-path <steergen>/params \
    --fsdp-devices 8 --batch-size 128 --num-train-steps 3000
```

Our run differed from these commands in ways that do not change the mixture:
- **Sharding.** Each LeRobot dataset was converted in five shards in parallel, and each shard got its dataset's
  weight (`--data.repo-ids` / `--data.dataset-weights`). The weights are per frame, so the mixture is unchanged.
- **Parallelism.** SteerGen generation used two workers per source task, and evaluations more workers per task.
- **Layout.** Servers and rendering were spread over the GPUs as described below.

## Results

### Steerability
Full protocol, 20,000 rollouts per policy. The paper-era aggregation averaged over all ten target instructions,
including the source task's own, so the paper's numbers compare with the middle column.

| Policy | Score (target ≠ source) | Incl. target = source | Paper |
|---|---|---|---|
| π0.5 (released `pi05_libero`) | **0.340** | 0.404 | 0.403 |
| + SteerGen | **0.573** | 0.604 | 0.49 |
| + SteerGen + SRBC (ReSteer) | **0.596** | 0.629 | 0.51 |

By switch step (target ≠ source):

| Switch step | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 | 65 | 70 | 75 | 80 | 85 | 90 | 95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| π0.5 | 0.97 | 0.98 | 0.82 | 0.59 | 0.46 | 0.34 | 0.27 | 0.24 | 0.23 | 0.20 | 0.20 | 0.20 | 0.18 | 0.18 | 0.16 | 0.16 | 0.13 | 0.17 | 0.15 | 0.17 |
| + SteerGen | 0.89 | 0.92 | 0.91 | 0.89 | 0.89 | 0.84 | 0.81 | 0.73 | 0.66 | 0.59 | 0.52 | 0.46 | 0.37 | 0.34 | 0.32 | 0.25 | 0.26 | 0.27 | 0.27 | 0.24 |
| + SteerGen, Title Case labels (3 repeats) | 0.98 | 0.97 | 0.91 | 0.83 | 0.69 | 0.57 | 0.46 | 0.34 | 0.30 | 0.24 | 0.22 | 0.21 | 0.21 | 0.20 | 0.19 | 0.23 | 0.21 | 0.23 | 0.23 | 0.23 |
| + SteerGen + SRBC | 0.92 | 0.94 | 0.92 | 0.91 | 0.92 | 0.88 | 0.83 | 0.75 | 0.65 | 0.62 | 0.57 | 0.49 | 0.44 | 0.36 | 0.31 | 0.29 | 0.28 | 0.29 | 0.27 | 0.28 |

By source task (target ≠ source):

| Source task | π0.5 | + SteerGen | + SteerGen + SRBC |
|---|---|---|---|
| open the middle drawer of the cabinet | 0.28 | 0.54 | 0.58 |
| open the top drawer and put the bowl inside | 0.29 | 0.53 | 0.51 |
| push the plate to the front of the stove | 0.27 | 0.56 | 0.55 |
| put the bowl on the plate | 0.41 | 0.67 | 0.69 |
| put the bowl on the stove | 0.42 | 0.67 | 0.71 |
| put the bowl on top of the cabinet | 0.42 | 0.59 | 0.63 |
| put the cream cheese in the bowl | 0.42 | 0.67 | 0.73 |
| put the wine bottle on the rack | 0.30 | 0.49 | 0.53 |
| put the wine bottle on top of the cabinet | 0.30 | 0.44 | 0.46 |
| turn on the stove | 0.29 | 0.56 | 0.58 |

### CMI
Normalized CMI, mean over tasks. All policies are measured on the same states: 20 per task, from rollouts of the
SteerGen policy.

| Policy | Lower-case instructions (release default) | Title Case (the paper's CMI runs) | Paper |
|---|---|---|---|
| π0.5 | 0.100 | 0.204 | 0.217 |
| + SteerGen | 0.262 | 0.309 | 0.31 |
| + SteerGen + SRBC | 0.252 | 0.310 | 0.30 |

### Data and training
| | |
|---|---|
| SteerGen bridges | 4,500 (90 task pairs × steps 0–49), 156,496 frames; none dropped |
| SteerGen fine-tuning | 2,000 steps in 38 min on 4 H100s; loss 0.106 (step 0) → 0.012 (step 1,900) |
| SRBC success-rate pool | 651 configurations, 6,519 rollouts, 3,189 successful switches |
| SRBC CMI pool | 900 low-CMI configurations, 9,662 rollouts, 4,016 successful switches |
| SRBC training set | 2,000 episodes, half from each pool: 3,945 LeRobot episodes (the segments before and after each switch), 444,692 frames |
| SRBC fine-tuning | 3,000 steps in 52 min on 8 H100s, from the SteerGen checkpoint; loss 0.024 (step 0) → 0.019 (step 2,900) |

### Discussion
- **π0.5 and CMI reproduce the paper.**
  - Steerability: 0.404 vs 0.403.
  - Title-Case CMI: 0.204 vs 0.217 for π0.5, and 0.309 vs 0.31 for SteerGen.
- **SteerGen scores higher than in the paper** (0.604 vs 0.49). The cause is the casing of the training labels.
  - Our bridges carry the lower-case instructions the policy is evaluated with. The paper-era SteerGen dataset stored
    Title Case task strings, which the PaliGemma tokenizer does not normalize.
  - Ablation: we converted the same 4,500 bridges with Title Case task strings, trained with the identical recipe
    (loss 0.0114 at step 1,900 vs 0.0117), and evaluated with 3 repeats per cell (6,000 rollouts).
  - That model scores 0.477 (0.423 with target ≠ source), matching the paper's 0.49.

    | SteerGen labels | Score (target ≠ source) | Incl. target = source | Rollouts |
    |---|---|---|---|
    | lower case (release default) | 0.573 | 0.604 | 20,000 |
    | Title Case (paper-era data) | 0.423 | 0.477 | 6,000 |
  - By switch step, the gain is largest at steps 15–50 (+0.30 to +0.54), the phase the bridges cover (steps 0–49).
    It is smaller but positive later (+0.07 to +0.19 from step 60 on). At steps 0–5, SteerGen is slightly below π0.5
    (0.89–0.92 vs 0.97–0.98).
- **SRBC adds a small gain on top of SteerGen, as in the paper:** +0.023 (0.573 → 0.596; paper: 0.49 → 0.51).
  - The standard error of the difference is about 0.005.
  - The gain is spread over the switch steps and over 8 of the 10 source tasks. The other two are within noise.
- **CMI.**
  - Lower-case CMI separates the policies more clearly than Title Case: 0.100 → 0.262 vs 0.204 → 0.309. It is also
    the setting that matches how the policies are queried.
  - CMI does not grow further with SRBC (0.252 / 0.310 vs 0.262 / 0.309), matching the paper (0.30 vs 0.31).

## Notes on running at this scale
- **Rendering on H100s.** H100s have very little graphics hardware. On this node the NVIDIA EGL driver aborted
  simulation workers (SIGABRT in `mjr_readPixels`, inside `libnvidia-eglcore`).
  - On five of the eight GPUs this happened every few minutes, to all workers of the GPU at once. The other three
    were stable throughout.
  - We rendered on the three stable GPUs only (up to 25 workers each); the others served policies and trained.
  - The launchers restart failed workers (`--max-restarts`).
  - Work is checkpointed finely, and results are identical to an uninterrupted run:
    - evaluation after every rollout (a crash costs at most one rollout);
    - SteerGen after every bridge;
    - SRBC after every attempt;
    - CMI after every sampled state.
  - GPUs with full graphics hardware (L40, A40, RTX) do not need any of this.
- **Wall-clock time** on this node (rendering on three GPUs):

  | Phase | Time |
  |---|---|
  | SteerGen generation (20 workers) and LeRobot conversion | ~1 h + 15 min |
  | SteerGen fine-tuning, 2,000 steps (4 H100s) | 38 min |
  | Steerability evaluation, 20,000 rollouts (40–60 workers on 2–3 rendering GPUs) | 2.2–3.7 h |
  | CMI, 4 runs sharing one state bank | 77 min |
  | SRBC collection, both pools (16,181 rollouts of up to 400 steps) | 5.7 h |
  | SRBC dataset (sampling, conversion) | 29 min |
  | SRBC fine-tuning, 3,000 steps (8 H100s) | 52 min |

- **Disk.** The run wrote about 400 GB, and a training checkpoint with optimizer state is 42 GB. Break-down:
  - raw SRBC episodes: 145 GB;
  - LeRobot datasets: 119 GB, including 33 GB for `physical-intelligence/libero`;
  - SteerGen bridges: 16 GB.

  The first SRBC training run lost its final checkpoint to a full disk quota. `scripts/policy.sh export_checkpoint`
  keeps only `params` and the normalization statistics (12 GB).
