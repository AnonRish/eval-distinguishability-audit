#!/usr/bin/env python3
"""How wide is the OOF-bootstrap interval compared with the real sampling spread of the AUC?

For many independent *null* corpora (no planted signal) this prints the spread (standard
deviation) of the cross-validated AUC across corpora, next to the spread an AUC computed from
independent scores would have. The ratio is the variance inflation caused by training on
overlapping folds, which the OOF-bootstrap interval ignores.
"""

from __future__ import annotations

import math

import numpy as np

from eval_distinguishability import make_synthetic
from eval_distinguishability.audit import _folds, _oof_scores
from eval_distinguishability.features import build
from eval_distinguishability.stats import auc


def main(replicates: int = 60) -> None:
    for n in (100, 300):
        values = []
        for seed in range(replicates):
            eval_corpus, real_corpus = make_synthetic(n, 0.0, seed=1000 + seed)
            texts = [*eval_corpus.texts, *real_corpus.texts]
            y = np.array([1] * n + [0] * n)
            features = build("word", texts)
            scores = _oof_scores(features.matrix, y, _folds(y, None, 5, 0), 1.0, 0)
            values.append(auc(scores, y))
        spread = float(np.std(values, ddof=1))
        ideal = math.sqrt((2 * n + 1) / (12 * n * n))
        print(
            f"n={n} per class: mean AUC {np.mean(values):.3f}, spread {spread:.3f}, "
            f"independent-scores spread {ideal:.3f}, inflation {spread / ideal:.2f}x"
        )


if __name__ == "__main__":
    main()
