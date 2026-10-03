"""Compares resteer.cmi.kde with the paper's torch implementation (entropy_utils.KDE).

Run in an environment with torch and the old repository on PYTHONPATH (see setup_old_env.sh):
    python tests/parity/kde_vs_torch.py OLD_REPO
"""

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "examples" / "libero_steerability"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from entropy_utils import KDE  # noqa: E402

from resteer.cmi import kde  # noqa: E402

rng = np.random.default_rng(0)
worst = 0.0
for trial in range(200):
    n = 320 if trial % 2 else 32
    scale = 10.0 ** rng.uniform(-3, 1)
    x = (rng.normal(size=(n, 3)) * scale * rng.uniform(0.1, 3, size=3)).astype(np.float32)
    if trial % 5 == 0:  # clustered, like end points of a confident policy
        x[: n // 2] = x[0] + 1e-4 * rng.normal(size=(n // 2, 3)).astype(np.float32)
    reference = float(KDE().kde_entropy(torch.from_numpy(x).unsqueeze(0)).squeeze())
    ours = kde.kde_entropy(x)
    worst = max(worst, abs(reference - ours) / max(1.0, abs(reference)))
print(f"max relative difference over 200 sample sets: {worst:.2e}")
sys.exit(0 if worst < 1e-5 else 1)
