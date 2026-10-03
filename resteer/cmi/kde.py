"""Kernel density entropy estimate used for the CMI proxy (NumPy port of the paper's torch code).

For samples x_1..x_N in R^d::

    h    = mean_d(std_d(x)) * N^(-1/(d+4))                      (Scott's rule; std with ddof=1)
    p(i) = (1/N) * sum_j exp(-||x_i - x_j||^2 / (2 h^2))        (unnormalized Gaussian kernel, self-term included)
    H    = -(1/N) * sum_i log(p(i) + 1e-8)

The kernel omits the Gaussian normalizing constant and the bandwidth is re-estimated for every
sample set, so H is a relative (not differential) entropy in [0, log N]. The arithmetic follows the
original float32 torch implementation (std accumulated in float64, as torch does on CPU).

A degenerate set (all samples identical, e.g. from a deterministic policy) has h = 0 and gives NaN,
as in the original code; CMI is not defined for deterministic policies.
"""

import numpy as np


def scott_bandwidth(x: np.ndarray) -> float:
    n, d = x.shape
    std = np.std(x.astype(np.float64), axis=0, ddof=1).astype(np.float32)
    return float(std.mean(dtype=np.float32)) * n ** (-1.0 / (d + 4))


def kde_entropy(samples: np.ndarray) -> float:
    """Entropy estimate of an [N, d] sample set (see module docstring)."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError(f"Expected an [N>=2, d] array, got shape {x.shape}")
    h = scott_bandwidth(x)
    sq_dist = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    kernel = np.exp(-sq_dist / np.float32(2 * h**2))
    density = kernel.sum(axis=1) / np.float32(len(x))
    return float(-np.log(density + np.float32(1e-8)).mean())
