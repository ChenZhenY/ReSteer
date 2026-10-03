"""Small I/O helpers."""

import base64
import json
import logging
import os
import pathlib
import subprocess
import tempfile

import numpy as np


def to_jsonable(obj):
    """Recursively converts NumPy scalars/arrays (and tuples) into JSON-serializable Python objects."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return obj


def write_json(path: "str | os.PathLike", obj, indent: int = 2) -> None:
    """Writes JSON atomically, so an interrupted run never leaves a truncated result file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(to_jsonable(obj), f, indent=indent)
        os.replace(tmp, path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: "str | os.PathLike"):
    with open(path) as f:
        return json.load(f)


def git_revision() -> str | None:
    try:
        root = pathlib.Path(__file__).resolve().parent
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def encode_rng_state(state) -> dict:
    """JSON-serializable form of ``np.random.get_state()``."""
    name, keys, pos, has_gauss, cached = state
    return {
        "name": name,
        "keys": base64.b64encode(np.asarray(keys, dtype=np.uint32).tobytes()).decode(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def decode_rng_state(d: dict):
    """Inverse of :func:`encode_rng_state`, for ``np.random.set_state``."""
    keys = np.frombuffer(base64.b64decode(d["keys"]), dtype=np.uint32)
    return (d["name"], keys, d["pos"], d["has_gauss"], d["cached_gaussian"])
