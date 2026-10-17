from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sklearn.metrics import roc_auc_score

from eval_distinguishability import auc, bootstrap_ci, permutation_p


@st.composite
def scored_labels(draw: st.DrawFn) -> tuple[list[int], list[int]]:
    n = draw(st.integers(4, 40))
    scores = draw(st.lists(st.integers(0, 6), min_size=n, max_size=n))  # many ties on purpose
    labels = draw(st.lists(st.integers(0, 1), min_size=n, max_size=n))
    if len(set(labels)) < 2:
        labels[0], labels[-1] = 0, 1
    return scores, labels


@given(scored_labels())
def test_auc_matches_scikit_learn_including_ties(case: tuple[list[int], list[int]]) -> None:
    scores, labels = case
    assert auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


@given(scored_labels())
def test_auc_symmetries(case: tuple[list[int], list[int]]) -> None:
    scores, labels = case
    value = auc(scores, labels)
    assert 0.0 <= value <= 1.0
    assert auc([-s for s in scores], labels) == pytest.approx(1.0 - value, abs=1e-12)
    assert auc(scores, [1 - y for y in labels]) == pytest.approx(1.0 - value, abs=1e-12)
    assert auc([2 * s + 7 for s in scores], labels) == pytest.approx(value, abs=1e-12)


def test_auc_extremes_and_errors() -> None:
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    with pytest.raises(ValueError, match="both classes"):
        auc([1, 2, 3], [1, 1, 1])
    with pytest.raises(ValueError, match="both classes"):
        auc([1, 2, 3], [0, 0, 0])


def _gaussian_case(
    rng: np.random.Generator, n: int, d_prime: float
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.concatenate([rng.normal(d_prime, 1.0, n), rng.normal(0.0, 1.0, n)])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    return scores, labels


@pytest.mark.slow
def test_bootstrap_interval_has_close_to_nominal_coverage_for_a_fixed_scoring_rule() -> None:
    """For scores that do not depend on the resampled data the percentile bootstrap should cover
    the true AUC about 95% of the time (the AUC of two unit Gaussians d' apart is
    Phi(d' / sqrt 2))."""
    d_prime, n = 1.0, 120
    truth = NormalDist().cdf(d_prime / math.sqrt(2))
    rng = np.random.default_rng(2026)
    hits = 0
    replicates = 150
    for _ in range(replicates):
        scores, labels = _gaussian_case(rng, n, d_prime)
        low, high = bootstrap_ci(scores, labels, 200, 0.95, rng)
        hits += low <= truth <= high
    coverage = hits / replicates
    assert 0.89 <= coverage <= 0.995, coverage


def test_bootstrap_interval_properties() -> None:
    rng = np.random.default_rng(1)
    scores, labels = _gaussian_case(rng, 200, 1.0)
    point = auc(scores, labels)
    low, high = bootstrap_ci(scores, labels, 400, 0.95, np.random.default_rng(5))
    assert low < point < high
    again = bootstrap_ci(scores, labels, 400, 0.95, np.random.default_rng(5))
    assert (low, high) == again  # deterministic for a fixed generator seed
    narrow = bootstrap_ci(scores, labels, 400, 0.50, np.random.default_rng(5))
    assert narrow[1] - narrow[0] < high - low  # a lower level gives a narrower interval
    big_scores, big_labels = _gaussian_case(rng, 2000, 1.0)
    big_low, big_high = bootstrap_ci(big_scores, big_labels, 200, 0.95, np.random.default_rng(5))
    assert big_high - big_low < high - low  # more data, narrower interval


def test_bootstrap_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="level"):
        bootstrap_ci([1, 2], [0, 1], 10, 1.0, rng)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci([1, 2], [0, 1], 0, 0.95, rng)
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_ci([1, 2], [1, 1], 10, 0.95, rng)


def test_permutation_p_formula_and_extremes() -> None:
    labels = np.array([0, 0, 1, 1, 0, 1])
    rng = np.random.default_rng(0)
    assert permutation_p(2.0, labels, lambda y: 1.0, 19, rng) == pytest.approx(
        1 / 20
    )  # never reached
    assert permutation_p(0.0, labels, lambda y: 1.0, 19, rng) == 1.0  # always reached
    seen: list[tuple[int, ...]] = []

    def record(permuted: np.ndarray) -> float:
        seen.append(tuple(sorted(permuted.tolist())))
        return 0.0

    permutation_p(1.0, labels, record, 10, np.random.default_rng(1))
    assert set(seen) == {tuple(sorted(labels.tolist()))}  # permutations keep the class counts
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_p(1.0, labels, lambda y: 0.0, 0, rng)


@pytest.mark.slow
def test_permutation_p_values_are_valid_under_the_null() -> None:
    rng = np.random.default_rng(7)
    n = 40
    labels = np.array([1] * n + [0] * n)
    p_values = []
    for _ in range(300):
        x = rng.normal(size=2 * n)  # no relationship with the labels

        def statistic(y: np.ndarray, x: np.ndarray = x) -> float:
            return float(x[y == 1].mean() - x[y == 0].mean())

        p_values.append(permutation_p(statistic(labels), labels, statistic, 99, rng))
    rate = np.mean(np.array(p_values) <= 0.05)
    assert 0.01 <= rate <= 0.10, rate
    assert 0.40 <= np.mean(p_values) <= 0.60
