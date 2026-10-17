from __future__ import annotations

import numpy as np
import pytest

from eval_distinguishability import auc, expected_length_auc, expected_marker_auc, make_synthetic
from eval_distinguishability.synth import vocabulary


def test_vocabulary() -> None:
    words = vocabulary(50)
    assert len(words) == len(set(words)) == 50 and all(len(w) == 5 for w in words)
    assert vocabulary(50) == words  # deterministic
    assert vocabulary(50, seed=1) != words


def test_markers_appear_only_in_evaluation_documents_at_the_requested_rate() -> None:
    eval_corpus, real_corpus = make_synthetic(2000, marker_rate=0.3, seed=1, n_markers=3)
    markers = {"zq0", "zq1", "zq2"}
    assert len(eval_corpus) == len(real_corpus) == 2000
    marked = [t for t in eval_corpus.texts if markers & set(t.split())]
    assert abs(len(marked) / 2000 - 0.3) < 0.03
    assert not any(markers & set(t.split()) for t in real_corpus.texts)
    used = {w for t in marked for w in t.split() if w in markers}
    assert used == markers
    assert all(sum(w in markers for w in t.split()) <= 1 for t in eval_corpus.texts)


def test_lengths_follow_the_requested_distribution() -> None:
    plain, shifted = make_synthetic(
        3000, 0.0, seed=2, length_shift=6.0, filler_mean=40, filler_sd=8
    )
    lengths_plain = np.array([len(t.split()) for t in shifted.texts])
    lengths_shifted = np.array([len(t.split()) for t in plain.texts])
    assert abs(lengths_plain.mean() - 40) < 0.6
    assert abs(lengths_shifted.mean() - 46) < 0.6
    assert abs(lengths_plain.std() - 8) < 0.6


def test_generation_is_deterministic_and_seeded() -> None:
    first = make_synthetic(20, 0.5, seed=3)
    assert make_synthetic(20, 0.5, seed=3) == first
    assert make_synthetic(20, 0.5, seed=4) != first


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_per_class": 0},
        {"n_per_class": 5, "marker_rate": 1.5},
        {"n_per_class": 5, "marker_rate": -0.1},
        {"n_per_class": 5, "n_markers": 0},
        {"n_per_class": 5, "vocab_size": 1},
    ],
)
def test_invalid_arguments(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        make_synthetic(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("shift", [0.0, 4.0, 8.0, 16.0])
def test_exact_length_auc_matches_the_empirical_length_auc(shift: float) -> None:
    eval_corpus, real_corpus = make_synthetic(4000, 0.0, seed=5, length_shift=shift)
    lengths = [len(t.split()) for t in (*eval_corpus.texts, *real_corpus.texts)]
    labels = [1] * 4000 + [0] * 4000
    assert auc(lengths, labels) == pytest.approx(expected_length_auc(shift), abs=0.015)


def test_expected_aucs() -> None:
    assert expected_length_auc(0.0) == pytest.approx(0.5, abs=1e-9)
    values = [expected_length_auc(s) for s in (0, 2, 4, 8, 16)]
    assert values == sorted(values) and values[-1] < 1.0
    assert expected_length_auc(8.0, 40.0, 8.0) == pytest.approx(0.7602, abs=1e-3)  # Phi(1 / sqrt 2)
    assert (expected_marker_auc(0.0), expected_marker_auc(0.5), expected_marker_auc(1.0)) == (
        0.5,
        0.75,
        1.0,
    )
