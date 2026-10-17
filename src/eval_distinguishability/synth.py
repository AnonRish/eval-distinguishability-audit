"""Synthetic corpora with a *known* amount of separability, for verifying the audit.

Documents are random words drawn uniformly from a fixed vocabulary, so the two classes are
identical in distribution except for the planted differences:

* ``marker_rate``: a fraction of evaluation documents contain a marker token that never
  occurs in real documents (the best possible classifier has AUC ``0.5 + marker_rate / 2``);
* ``length_shift``: evaluation documents have that many more words on average (the AUC of
  document length alone is computed exactly by :func:`expected_length_auc`).
"""

from __future__ import annotations

import random
from statistics import NormalDist

from .corpus import Corpus

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def vocabulary(size: int = 400, seed: int = 12345) -> list[str]:
    """``size`` distinct five-letter pseudo-words (deterministic)."""
    rng = random.Random(seed)
    words: set[str] = set()
    while len(words) < size:
        words.add("".join(rng.choice(_LETTERS) for _ in range(5)))
    return sorted(words)


def _length(rng: random.Random, mean: float, sd: float) -> int:
    return max(1, round(rng.gauss(mean, sd)))


def make_synthetic(
    n_per_class: int,
    marker_rate: float = 0.0,
    seed: int = 0,
    *,
    length_shift: float = 0.0,
    filler_mean: float = 40.0,
    filler_sd: float = 8.0,
    n_markers: int = 1,
    vocab_size: int = 400,
) -> tuple[Corpus, Corpus]:
    """Return ``(evaluation_corpus, real_corpus)`` with the planted differences above."""
    if n_per_class < 1:
        raise ValueError("n_per_class must be >= 1")
    if not 0.0 <= marker_rate <= 1.0:
        raise ValueError("marker_rate must be in [0, 1]")
    if n_markers < 1:
        raise ValueError("n_markers must be >= 1")
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")
    rng = random.Random(seed)
    vocab = vocabulary(vocab_size)
    marker_tokens = [f"zq{i}" for i in range(n_markers)]

    def document(mean: float, marked: bool) -> str:
        words = [rng.choice(vocab) for _ in range(_length(rng, mean, filler_sd))]
        if marked:
            words[rng.randrange(len(words))] = rng.choice(marker_tokens)
        return " ".join(words)

    eval_docs = tuple(
        document(filler_mean + length_shift, rng.random() < marker_rate) for _ in range(n_per_class)
    )
    real_docs = tuple(document(filler_mean, False) for _ in range(n_per_class))
    return Corpus(eval_docs), Corpus(real_docs)


def expected_marker_auc(marker_rate: float) -> float:
    """Best achievable AUC when a fraction of evaluation documents carry a unique marker."""
    return 0.5 + marker_rate / 2.0


def _length_pmf(mean: float, sd: float, upper: int) -> list[float]:
    """P(length = j) for j = 0..upper, where length = max(1, round(N(mean, sd)))."""
    cdf = NormalDist(mean, sd).cdf
    pmf = [0.0] * (upper + 1)
    pmf[1] = cdf(1.5)
    for j in range(2, upper + 1):
        pmf[j] = cdf(j + 0.5) - cdf(j - 0.5)
    return pmf


def expected_length_auc(
    length_shift: float, filler_mean: float = 40.0, filler_sd: float = 8.0
) -> float:
    """Exact AUC of word count as a classifier for :func:`make_synthetic` corpora."""
    upper = int(filler_mean + length_shift + 12 * filler_sd) + 1
    eval_pmf = _length_pmf(filler_mean + length_shift, filler_sd, upper)
    real_pmf = _length_pmf(filler_mean, filler_sd, upper)
    below = 0.0
    total = 0.0
    for j in range(1, upper + 1):
        total += eval_pmf[j] * (below + 0.5 * real_pmf[j])
        below += real_pmf[j]
    return total
