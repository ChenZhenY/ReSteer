# Changes relative to upstream LIBERO

This is a modified copy of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
(MIT license, see `LICENSE`). Differences below are relative to upstream `master` @ `8f1084e`.

**Do not replace this copy with upstream LIBERO.** Items 1–3 change task-success semantics,
and every ReSteer number depends on them.

## Behavioral changes (affect results)
1. **Per-goal success dictionary.**
   - `BDDLBaseDomain.step(action, return_success_dict=False)` and `_check_success(return_success_dict=False)`
     (`envs/bddl_base_domain.py`) forward the flag down to the `_check_success` of every problem class
     (`envs/problems/*.py`).
   - With the flag set, `done` becomes a dict with one entry per goal predicate, e.g.
     `{"on_akita_black_bowl_1_plate_1": True, ...}`.
   - The key is `"_".join(predicate)`.
   - `ControlEnv.step` and `CustomControlEnv.step` forward the flag (`envs/env_wrapper.py`).
2. **More lenient "on top" check.** `check_ontop` (`envs/object_states/base_object_states.py`) accepts an xy
   offset below **0.1** m instead of 0.03 m.
3. **Steerability scene.** `bddl_files/libero_steerability/libero_goal_steerability.bddl` is the LIBERO-Goal
   scene whose `:goal` is the conjunction of all 10 LIBERO-Goal task goals. One environment therefore reports
   the success of every task. `assets/articulated_objects/flat_stove.xml` gains a `base_region` site used by
   that scene.

## Added
- `CustomControlEnv` (`envs/env_wrapper.py`): a `ControlEnv` variant that takes explicit
  `controller_configs`. SteerGen uses it for closed-loop end-effector interpolation.

## Robustness fixes made for the ReSteer release
- `ControlEnv.reset` / `CustomControlEnv.reset`: upstream retried inside `finally: continue`, which swallowed
  *all* exceptions and looped forever (for example when EGL was misconfigured). They now retry only
  `RandomizationError`, at most 100 times. Successful resets behave identically.
- `libero/__init__.py`: when `$LIBERO_CONFIG_PATH/config.yaml` (default `~/.libero/config.yaml`) is missing,
  a default config pointing at this package is written. Upstream prompted with `input()` instead.
  ReSteer sets `LIBERO_CONFIG_PATH` to its own directory and passes BDDL files by absolute path, so an
  existing upstream `~/.libero/config.yaml` does not interfere.
- `libero/__init__.py` was added so an (editable) install exposes the package; upstream relied on `PYTHONPATH`.
- The gym-based vector env (`envs/venv.py`) was removed.

## Removed (not used by ReSteer)
Only what ReSteer loads is kept: the environment code (`envs/`), the steerability BDDL and the assets of its scene.
The kept files were found by tracing which modules and files the evaluation and SteerGen environments load. The
release's tests and a full LIBERO run use exactly this set.
- `notebooks/`, `images/`, `scripts/`, `templates/`, `benchmark_scripts/`.
- The lifelong-learning code (`libero/lifelong/`, `libero/configs/`).
- The benchmark suites and their data: `benchmark/`, `init_files/`, and every BDDL file except
  `libero_steerability/libero_goal_steerability.bddl`. This includes the LIBERO-Spatial analogue
  `libero_spatial_steerability.bddl` and an unrelated project's out-of-distribution variants.
- The helpers in `utils/` and `envs/textures.py`, which nothing imports.
- The visualization environments `SegmentationRenderEnv` and `DemoRenderEnv` (`envs/env_wrapper.py`).
- Assets of other scenes: 545 of 585 files. The kept 40 cover the kitchen table scene, its textures and the seven
  objects of LIBERO-Goal (bowl, plate, cream cheese, wine bottle, wine rack, stove, cabinet).
  - Object classes for other objects remain registered, but their assets are gone. Building another LIBERO scene
    with this copy fails with a missing-file error; use upstream LIBERO for that.
