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

## Running

```bash
# Local LeRobot datasets live under $HF_LEROBOT_HOME/<repo_id> (default ~/.cache/huggingface/lerobot).
scripts/policy.sh convert_to_lerobot --help

scripts/policy.sh train pi05_libero_resteer_steergen --exp-name steergen \
    --data.repo-ids physical-intelligence/libero <you>/libero_goal_steergen --fsdp-devices 2 --batch-size 64
scripts/policy.sh train pi05_libero_resteer_srbc --exp-name srbc \
    --data.repo-ids physical-intelligence/libero <you>/libero_goal_steergen <you>/libero_goal_srbc \
    --weight-loader.params-path checkpoints/pi05_libero_resteer_steergen/steergen/1999/params \
    --fsdp-devices 2 --batch-size 64
```

- All openpi training flags apply. Useful ones: `--num-train-steps`, `--save-interval`, `--no-wandb-enabled`,
  `--overwrite`, `--resume`.
- Checkpoints go to `checkpoints/<config>/<exp-name>/<step>`. The last step is `num_train_steps − 1`.
- The paper's runs used `--fsdp-devices 2 --batch-size 64` on 2×H100, or `--fsdp-devices 8 --batch-size 128` on
  8×A40.

## Serving and exporting checkpoints

- **Serving.** Checkpoints trained with these configs store their normalization statistics where openpi's
  `pi05_libero` config looks for them, so the stock config serves them:

  ```bash
  scripts/serve_policy.sh --checkpoint checkpoints/pi05_libero_resteer_srbc/srbc/9999
  ```

- **Exporting.** `scripts/policy.sh export_checkpoint <step dir> <out>` copies the parameters and statistics
  for release, dropping the optimizer state. Checkpoints from the paper's runs stored their statistics under
  `assets/libero/`; export moves them to `assets/physical-intelligence/libero/`.
