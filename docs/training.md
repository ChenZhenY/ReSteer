# Training with upstream openpi

Training uses the unmodified openpi submodule (`third_party/openpi`, pinned) and its own locked environment.
`policy/resteer_policy` adds three things:

- **Weighted co-training on several LeRobot datasets** (`multidata.py`).
  - openpi builds its training dataset in `openpi.training.data_loader.create_torch_dataset`. ReSteer wraps
    that single function.
  - A config's `repo_ids` become a concatenation in which dataset i is repeated `w_i` times, and each dataset
    keeps its own task table.
  - With shuffling, a frame of dataset i is drawn with probability `w_i / Σ_j w_j N_j`. This is the same
    per-frame weighting as the paper's weighted sampler (which sampled with replacement).
  - The mixture is therefore proportional to `w_i · N_i`, not to `w_i`.
- **The ReSteer configs** (`configs.py`), derived from upstream `pi05_libero`: model, optimizer, schedule, EMA.

  | Config | Datasets | Weights | Initialization | Steps |
  |---|---|---|---|---|
  | `pi05_libero_resteer_steergen` | LIBERO, SteerGen | 1:3 | released `pi05_libero` | 2k |
  | `pi05_libero_resteer_srbc` | LIBERO, SteerGen, SRBC | 1:3:15 | the SteerGen checkpoint | 10k |

  - Both use the released `pi05_libero` normalization statistics.
  - With 10k warmup steps, the learning rate is still ramping up (to 1e−5) at the end of the 2k-step SteerGen
    run. This matches the paper's runs.
- **Checkpoint export** (`export_checkpoint.py`): see below.

## Fine-tuning on your machine

Training runs locally with openpi; nothing is uploaded.

1. **Datasets.** Training reads LeRobot datasets from `$HF_LEROBOT_HOME` (default `~/.cache/huggingface/lerobot`).
   - `physical-intelligence/libero` (the LIBERO demonstrations, about 33 GB) is downloaded from the Hugging Face
     Hub on first use. No account is needed.
   - Convert your generated episodes under the configs' default repo ids, so training finds them without extra
     flags:

     ```bash
     scripts/policy.sh convert_to_lerobot data/steergen/bridges --repo-id resteer/libero_goal_steergen
     scripts/policy.sh convert_to_lerobot data/srbc/srbc.hdf5 --repo-id resteer/libero_goal_srbc
     ```

2. **Base weights.** `pi05_libero`'s parameters and normalization statistics download once from openpi's public
   bucket into `$OPENPI_DATA_HOME` (default `~/.cache/openpi`). To start from a local copy, pass
   `--weight-loader.params-path /path/to/params`.
3. **Train.** Both commands below use the default datasets:

   ```bash
   scripts/policy.sh train pi05_libero_resteer_steergen --exp-name steergen --fsdp-devices 2 --batch-size 64
   scripts/policy.sh train pi05_libero_resteer_srbc --exp-name srbc \
       --weight-loader.params-path checkpoints/pi05_libero_resteer_steergen/steergen/1999/params \
       --fsdp-devices 2 --batch-size 64
   ```

   - Use `--data.repo-ids` (with matching `--data.dataset-weights`) to train on other datasets.
   - All openpi training flags apply. Useful ones: `--num-train-steps`, `--save-interval`, `--overwrite`,
     `--resume`.
   - Weights & Biases logging is off; `--wandb-enabled` turns it on.
4. **Outputs.** Checkpoints go to `checkpoints/<config>/<exp-name>/<step>`. The last step is `num_train_steps − 1`.
   - A checkpoint with optimizer state takes about 42 GB; `export_checkpoint` keeps the 12 GB needed to serve it.

**Hardware.** π0.5 is fully fine-tuned with FSDP, sharded over `--fsdp-devices` GPUs. Parameters, optimizer state
and EMA weights take about 80 GB, so each GPU holds roughly 80 GB / N plus activations.
- **Known to work:**
  - 2×H100 80 GB (`--fsdp-devices 2 --batch-size 64`) and 8×A40 48 GB (`--fsdp-devices 8 --batch-size 128`): the
    paper's runs.
  - 4×H100 (batch 64) and 8×H100 (batch 128): the replication in [replication.md](replication.md).
- **Does not fit:** 2×48 GB GPUs (we tried 2×L40), even with `--ema-decay None` and batch 2. Use more GPUs, or
  larger ones.

## Serving and exporting checkpoints

- **Serving.** Checkpoints trained with these configs store their normalization statistics where openpi's
  `pi05_libero` config looks for them, so the stock config serves them:

  ```bash
  scripts/serve_policy.sh --checkpoint checkpoints/pi05_libero_resteer_srbc/srbc/9999
  ```

- **Exporting.** `scripts/policy.sh export_checkpoint <step dir> <out>` copies the parameters and statistics
  for release, dropping the optimizer state. Checkpoints from the paper's runs stored their statistics under
  `assets/libero/`; export moves them to `assets/physical-intelligence/libero/`.
