"""SteerGen: stage-aware steering data generation for LIBERO-Goal.

Pipeline (each step is a ``python -m resteer.steergen.<module>`` CLI):

1. State bank: ``resteer.states build-from-raw`` (demo states, used by ``step_matched``) or
   ``regen_states`` (replayed states, used by ``stage_matched``), then ``label_stages``.
2. ``generate``: bridges from states of one task to the end-effector pose of another task's state.
3. ``select_by_cmi`` (optional): keep bridges near low-CMI switch states.
4. ``policy/resteer_policy/convert_to_lerobot.py``: episode file -> LeRobot dataset for training.

See docs/steergen.md.
"""
