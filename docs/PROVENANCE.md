# Provenance

This release was assembled from the research code used for the paper.

| Source | Commit | Became |
|---|---|---|
| `ChenZhenY/pi0-anytime-steerability` (`main`) | `2db0966` | `resteer/eval`, `resteer/cmi`, `resteer/srbc`, `resteer/states.py`, `third_party/libero` |
| `GaTech-RL2/mimiclabs-priv` (`libero_steerability`) | `096a32a` | `resteer/steergen` |
| `ChenZhenY/openpi` (`main`, a fork of openpi `5bff19b`) | `4b20c17` | `policy/resteer_policy` (co-training and configs); the fork itself is not needed |
| `Physical-Intelligence/openpi` | `215abfb` | `third_party/openpi` (submodule, unmodified) |

Mapping from the research scripts:

| Research script | Release |
|---|---|
| `examples/libero_steerability/main_steerability.py --test_full_rollout` | `resteer/eval/steerability.py`, `resteer/eval/rollout.py` |
| `utils/{combine_*,simplify_experiment_results,aggregate_all_tasks_summary}.py` | `resteer/eval/score.py` (`--legacy` reads old result directories) |
| `examples/libero_steerability/main_steerability_entropy.py`, `entropy_utils.py` | `resteer/cmi/sample_actions.py`, `resteer/cmi/kde.py` |
| `visualization/viz_mutual_information.py` | `resteer/cmi/compute_cmi.py` |
| `examples/libero_steerability/collect_policy_rollouts.py`, `merge_rollout_hdf5.py` | `resteer/cmi/collect_states.py`, `resteer/states.py merge` |
| `utils/hdf5_states_combiner.py` | `resteer/states.py build-from-raw` |
| `examples/libero_steerability/filtered_bc_libero_full_rollout{,_mi}.py` | `resteer/srbc/{select,collect}.py` |
| `utils/hdf5_{database_generator,database_sampler,demo_combiner}.py` | `resteer/srbc/build_dataset.py` |
| `utils/convert_combined_hdf5_to_lerobot.py` | `policy/resteer_policy/convert_to_lerobot.py` |
| mimiclabs `collect_interpolation_data{,_v3}.py`, `label_libero_goal_states.py`, `playback_libero_collect_sim_state.py`, `delete_high_mi_demos.py` | `resteer/steergen/*` |
| openpi fork `MultiDatasetLiberoDataConfig`, `pi05_libero_cotraining_{interpolation,SRBC}` | `policy/resteer_policy/{multidata,configs}.py` |

## Behavior changes relative to the research code

Each change is documented in the module that makes it. Where a change affects numbers, a flag reproduces the
paper-era behavior.

| Change | Reproduce the old behavior with |
|---|---|
| The headline score excludes target = source cells | none; the summary also reports the including-same-task mean |
| CMI queries lower-case instructions | `--prompt-style title` |
| SRBC records a fresh observation every step, uses one environment per worker, renders at 512 and warms up | `--paper-compat` |
| Generated datasets are labelled with lower-case instructions | `--prompt-style title` |
| Low-CMI selection covers all switch steps and excludes same-task cells | `compute_cmi --paper-compat` |
| Resuming a partial evaluation works (it was silently disabled by a merge) | none |
| Server failures abort the run instead of being counted as task failures | none |
| SteerGen demo choice is seeded | none |
| SteerGen v3 bugs are fixed (undefined candidates, joint-state offset) | none |
| Action scaling is recorded instead of asked interactively | none |
