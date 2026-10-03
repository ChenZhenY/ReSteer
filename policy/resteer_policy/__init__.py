"""ReSteer extensions for upstream openpi: co-training configs, dataset conversion, checkpoint export.

This package runs inside openpi's own environment (``uv run --project third_party/openpi``) and does
not depend on the ``resteer`` simulation package; the two sides only exchange files (episode HDF5,
LeRobot datasets, checkpoints) and websocket messages.
"""
