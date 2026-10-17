"""AUC, bootstrap confidence intervals and permutation p-values."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata


def auc(scores: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count one half).

    ``labels`` are 1 for the positive class (evaluation prompts). Higher scores should mean
    "more evaluation-like".
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both classes must be present")
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(
    scores: npt.ArrayLike,
    labels: npt.ArrayLike,
    n_boot: int,
    level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the AUC (resampling each class separately)."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("both classes must be present")
    values = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.choice(pos, size=len(pos), replace=True)
        n = rng.choice(neg, size=len(neg), replace=True)
        values[i] = auc(np.concatenate([p, n]), np.concatenate([np.ones(len(p)), np.zeros(len(n))]))
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def permutation_p(
    observed: float,
    labels: npt.ArrayLike,
    statistic: Callable[[np.ndarray], float],
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """One-sided permutation p-value ``(1 + #{perm >= observed}) / (1 + n_permutations)``.

    ``statistic`` maps a (permuted) label vector to the statistic, re-running the whole
    pipeline, so the null distribution accounts for training.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    y = np.asarray(labels)
    exceed = 0
    for _ in range(n_permutations):
        if statistic(rng.permutation(y)) >= observed - 1e-12:
            exceed += 1
    return (1 + exceed) / (1 + n_permutations)
