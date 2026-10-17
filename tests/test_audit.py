from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from eval_distinguishability import (
    AuditConfig,
    AuditError,
    Corpus,
    effect_label,
    expected_length_auc,
    expected_marker_auc,
    make_synthetic,
    run_audit,
)
from eval_distinguishability import audit as audit_module

QUICK = AuditConfig(n_permutations=30, n_bootstrap=100, families=("word",), report_features=False)
LOW_NOISE: dict[str, Any] = {"filler_mean": 10, "filler_sd": 2, "vocab_size": 30}


def planted(n: int, rate: float, seed: int, **extra: Any) -> tuple[Corpus, Corpus]:
    return make_synthetic(n, rate, seed, **{**LOW_NOISE, **extra})


@pytest.mark.parametrize("rate", [0.3, 0.6, 0.9])
def test_planted_markers_are_recovered_near_the_best_possible_auc(rate: float) -> None:
    observed = []
    for seed in range(3):
        report = run_audit(*planted(200, rate, seed), QUICK)
        observed.append(report.families[0].auc)
        # No classifier can beat the Bayes-optimal AUC by more than sampling noise.
        assert observed[-1] <= expected_marker_auc(rate) + 0.07
    assert abs(np.mean(observed) - expected_marker_auc(rate)) <= 0.06


def test_a_strong_signal_is_declared_distinguishable_with_exact_permutation_resolution() -> None:
    report = run_audit(*planted(200, 0.9, 0), QUICK)
    family = report.families[0]
    assert report.verdict == "distinguishable" and report.distinguishable
    assert family.p_value == pytest.approx(1 / 31)  # no permutation reached the observed AUC
    assert family.p_adjusted == pytest.approx(1 / 31)
    assert family.ci_low > 0.5 and family.ci_low < family.auc < family.ci_high
    assert report.effect in ("moderate", "strong") and report.best_family == "word"


@pytest.mark.slow
def test_null_corpora_rarely_look_distinguishable() -> None:
    config = AuditConfig(
        n_permutations=25, n_bootstrap=60, families=("word",), report_features=False, alpha=0.1
    )
    verdicts, aucs = [], []
    for seed in range(30):
        report = run_audit(*planted(80, 0.0, 100 + seed), config)
        verdicts.append(report.distinguishable)
        aucs.append(report.families[0].auc)
    assert np.mean(verdicts) <= 0.30, np.mean(verdicts)  # nominal rate 0.1
    assert abs(np.mean(aucs) - 0.5) <= 0.035


def test_length_differences_are_caught_by_the_surface_family() -> None:
    eval_corpus, real_corpus = make_synthetic(250, 0.0, seed=4, length_shift=8.0)
    config = AuditConfig(
        n_permutations=20, n_bootstrap=100, families=("word", "surface"), report_features=False
    )
    report = run_audit(eval_corpus, real_corpus, config)
    by_name = {f.name: f for f in report.families}
    assert report.best_family == "surface"
    assert by_name["surface"].auc == pytest.approx(expected_length_auc(8.0), abs=0.06)
    assert by_name["surface"].auc > by_name["word"].auc
    for family in report.families:  # Bonferroni across the two families tried
        assert family.p_adjusted == pytest.approx(min(1.0, 2 * family.p_value))


def test_grouped_cross_validation_prevents_duplicate_leakage() -> None:
    eval_corpus, real_corpus = make_synthetic(60, 0.0, seed=8)  # no real signal at all
    config = AuditConfig(
        n_permutations=10, n_bootstrap=50, families=("word",), report_features=False, dedupe=False
    )

    def triplicate(corpus: Corpus, tag: str) -> Corpus:
        texts = tuple(t for t in corpus.texts for _ in range(3))
        groups = tuple(f"{tag}{i}" for i in range(len(corpus)) for _ in range(3))
        return Corpus(texts, groups)

    leaky = run_audit(
        Corpus(triplicate(eval_corpus, "e").texts),
        Corpus(triplicate(real_corpus, "r").texts),
        config,
    )
    careful = run_audit(triplicate(eval_corpus, "e"), triplicate(real_corpus, "r"), config)
    assert not leaky.grouped and careful.grouped
    assert leaky.families[0].auc > 0.8  # copies of the same prompt on both sides of the split
    assert careful.families[0].auc < 0.65  # with groups the leak disappears


def test_one_sided_groups_are_supported() -> None:
    eval_corpus, real_corpus = planted(40, 0.5, 1)
    grouped = Corpus(eval_corpus.texts, tuple(f"g{i // 2}" for i in range(len(eval_corpus))))
    report = run_audit(grouped, real_corpus, QUICK)
    assert report.grouped


def test_duplicates_are_removed_counted_and_reported() -> None:
    eval_corpus, real_corpus = planted(60, 0.5, 2)
    doubled = Corpus((*eval_corpus.texts, *eval_corpus.texts[:10]))
    report = run_audit(doubled, real_corpus, QUICK)
    assert report.duplicates_removed == (10, 0) and report.n_eval == 60
    kept = run_audit(doubled, real_corpus, AuditConfig(**{**QUICK.__dict__, "dedupe": False}))
    assert kept.duplicates_removed == (0, 0) and kept.n_eval == 70
    shared = Corpus((*real_corpus.texts[:5], *eval_corpus.texts))
    warned = run_audit(shared, real_corpus, QUICK)
    assert warned.cross_class_duplicates == 5
    assert any("appear in both corpora" in w for w in warned.warnings)


def test_warnings_for_small_and_imbalanced_samples() -> None:
    small_eval, small_real = planted(20, 0.5, 3)
    report = run_audit(small_eval, small_real, QUICK)
    assert any("small sample" in w for w in report.warnings)
    big_eval, _ = planted(120, 0.5, 4)
    _, tiny_real = planted(20, 0.0, 5)
    imbalanced = run_audit(big_eval, tiny_real, QUICK)
    assert any("imbalanced" in w for w in imbalanced.warnings)
    clean = run_audit(*planted(60, 0.5, 6), QUICK)
    assert clean.warnings == ()


def test_the_planted_marker_is_the_top_feature_and_points_at_evaluation_prompts() -> None:
    config = AuditConfig(n_permutations=10, n_bootstrap=50, families=("word", "surface"), top_k=5)
    report = run_audit(*planted(200, 0.6, 0), config)
    top = report.top_features["word"]
    assert top[0].feature == "zq0" and top[0].direction == "eval"
    assert top[0].df_eval is not None and top[0].df_eval > 90 and top[0].df_real == 0
    assert [abs(w.weight) for w in top] == sorted((abs(w.weight) for w in top), reverse=True)
    surface = report.top_features["surface"]
    assert len(surface) == 5  # dense features carry no privacy filter ...
    assert all(w.df_eval is None and w.df_real is None for w in surface)  # ... or document counts
    assert "dense feature" in report.render()


def test_rare_tokens_are_withheld_from_the_feature_report_by_min_df() -> None:
    eval_corpus, real_corpus = planted(100, 0.0, 9)
    texts = list(eval_corpus.texts)
    for i in range(3):
        texts[i] = texts[i] + " secretname"  # appears in only 3 evaluation prompts
    corpus = Corpus(tuple(texts))
    wide = AuditConfig(n_permutations=5, n_bootstrap=20, families=("word",), top_k=500, min_df=1)
    strict = AuditConfig(n_permutations=5, n_bootstrap=20, families=("word",), top_k=500, min_df=5)
    assert "secretname" in {
        w.feature for w in run_audit(corpus, real_corpus, wide).top_features["word"]
    }
    assert "secretname" not in {
        w.feature for w in run_audit(corpus, real_corpus, strict).top_features["word"]
    }


def test_feature_reporting_can_be_switched_off() -> None:
    report = run_audit(*planted(60, 0.6, 0), QUICK)
    assert report.top_features == {}


def test_ablation_distinguishes_concentrated_from_diffuse_signals() -> None:
    config = AuditConfig(
        n_permutations=5,
        n_bootstrap=20,
        families=("word",),
        report_features=False,
        ablation_ks=(5, 200),
    )
    concentrated = run_audit(*planted(200, 0.9, 0, n_markers=1), config)
    diffuse = run_audit(*planted(200, 0.9, 0, n_markers=40), config)
    assert concentrated.ablation_family == "word"
    c0, c5, c200 = concentrated.ablation
    d0, d5, _ = diffuse.ablation
    assert (c0.removed, c5.removed, c200.removed) == (0, 5, 200)
    assert c0.auc == concentrated.families[0].auc
    # One telltale token: dropping just 5 features already costs a lot, and dropping the
    # n-grams around it leaves chance rather than "anti-signal" (the ranking is learned per fold).
    assert c0.auc - c5.auc > 0.1
    assert abs(c200.auc - 0.5) < 0.1
    # Forty rare markers: no handful of features carries the signal.
    assert d0.auc - d5.auc < 0.05


def test_ablation_skips_counts_larger_than_the_feature_set() -> None:
    config = AuditConfig(
        n_permutations=5,
        n_bootstrap=20,
        families=("word",),
        report_features=False,
        ablation_ks=(5, 10_000),
    )
    report = run_audit(*planted(60, 0.5, 0), config)
    assert [p.removed for p in report.ablation] == [0, 5]


def test_the_verdict_requires_the_interval_to_exclude_chance_as_well() -> None:
    config = AuditConfig(
        n_permutations=10, n_bootstrap=100, families=("word",), report_features=False, alpha=0.99
    )
    report = run_audit(*planted(100, 0.0, 11), config)
    assert report.families[0].p_adjusted < 0.99  # the permutation test alone would say "yes"
    assert report.families[0].ci_low <= 0.5 and report.verdict == "no_evidence"


@pytest.mark.parametrize(
    "value,label",
    [
        (0.40, "negligible"),
        (0.549, "negligible"),
        (0.55, "weak"),
        (0.649, "weak"),
        (0.65, "moderate"),
        (0.799, "moderate"),
        (0.80, "strong"),
        (1.0, "strong"),
    ],
)
def test_effect_labels(value: float, label: str) -> None:
    assert effect_label(value) == label


def test_runs_are_deterministic_for_a_seed() -> None:
    corpora = planted(80, 0.5, 12)
    first = run_audit(*corpora, QUICK).to_json()
    again = run_audit(*corpora, QUICK).to_json()
    other = run_audit(*corpora, AuditConfig(**{**QUICK.__dict__, "seed": 1})).to_json()
    assert first == again and first != other


def test_report_serialisation_and_rendering() -> None:
    config = AuditConfig(
        n_permutations=5, n_bootstrap=20, families=("word", "char", "surface"), top_k=3
    )
    report = run_audit(*planted(80, 0.6, 0), config)
    payload = json.loads(json.dumps(report.to_json()))
    assert payload["verdict"] == report.verdict and len(payload["families"]) == 3
    assert payload["config"]["families"] == ["word", "char", "surface"]
    text = report.render()
    assert "Evaluation-vs-real distinguishability audit" in text
    assert "Verdict:" in text and "Most discriminative word features" in text
    assert "Signal concentration (word features" in text and "Reading this:" in text
    quiet = run_audit(*planted(80, 0.0, 1), QUICK)
    assert "Most discriminative" not in quiet.render()
    small = run_audit(*planted(20, 0.5, 3), QUICK)
    assert "WARNING: small sample" in small.render()


def test_invalid_inputs_are_rejected() -> None:
    eval_corpus, real_corpus = planted(60, 0.5, 0)
    with pytest.raises(AuditError, match="at least"):
        run_audit(Corpus(eval_corpus.texts[:5]), real_corpus, QUICK)
    for kwargs in (
        {"n_splits": 1},
        {"n_permutations": 0},
        {"n_bootstrap": 0},
        {"ci_level": 1.0},
        {"alpha": 0.0},
        {"min_df": 0},
        {"top_k": 0},
        {"c": 0.0},
        {"families": ()},
    ):
        with pytest.raises(AuditError):
            AuditConfig(**kwargs)
    # Distinct prompts made only of one-character tokens leave the word vocabulary empty.
    unusable = Corpus(tuple(f"{chr(97 + i % 26)} {chr(97 + i // 26)} x" for i in range(40)))
    other = Corpus(tuple(f"{chr(97 + i % 26)} {chr(97 + i // 26)} y" for i in range(40)))
    with pytest.raises(AuditError, match="no feature family"):
        run_audit(unusable, other, QUICK)


def test_a_training_fold_with_a_single_class_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_folds(
        y: np.ndarray, groups: Any, n_splits: int, seed: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(np.flatnonzero(y == 1), np.flatnonzero(y == 0))]

    monkeypatch.setattr(audit_module, "_folds", broken_folds)
    with pytest.raises(AuditError, match="single class"):
        run_audit(*planted(40, 0.5, 0), QUICK)


def test_all_default_families_run_together() -> None:
    config = AuditConfig(n_permutations=5, n_bootstrap=20)
    report = run_audit(*planted(60, 0.6, 0), config)
    assert [f.name for f in report.families] == ["word", "char", "surface"]
    assert all(f.n_features > 0 for f in report.families)


def test_reports_without_discriminative_features_or_ablation_render_cleanly() -> None:
    withheld = AuditConfig(n_permutations=5, n_bootstrap=20, families=("word",), min_df=10**6)
    report = run_audit(*planted(60, 0.6, 0), withheld)
    assert report.top_features == {"word": ()}
    assert "Most discriminative" not in report.render()
    dense_only = AuditConfig(n_permutations=5, n_bootstrap=20, families=("surface",))
    surface = run_audit(*planted(60, 0.6, 0), dense_only)
    assert surface.ablation == () and surface.ablation_family is None
    assert "Signal concentration" not in surface.render()


def test_a_family_that_builds_no_columns_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval_distinguishability import features as features_module

    empty = features_module.FeatureSet("word", np.zeros((120, 0)), [], False)
    monkeypatch.setitem(features_module.BUILDERS, "word", lambda texts: empty)
    config = AuditConfig(
        n_permutations=5, n_bootstrap=20, families=("word", "surface"), report_features=False
    )
    report = run_audit(*planted(60, 0.6, 0), config)
    assert [f.name for f in report.families] == ["surface"]
    assert any("no features survived pruning" in w for w in report.warnings)


def test_a_permutation_count_that_cannot_reach_alpha_is_flagged() -> None:
    few = AuditConfig(
        n_permutations=10, n_bootstrap=20, families=("word", "surface"), report_features=False
    )
    report = run_audit(*planted(100, 0.9, 0), few)
    assert any("smallest attainable adjusted p-value is 0.182" in w for w in report.warnings)
    assert report.verdict == "no_evidence"  # even a perfect signal cannot be declared with so few
    enough = AuditConfig(
        n_permutations=60, n_bootstrap=20, families=("word",), report_features=False
    )
    assert not any("attainable" in w for w in run_audit(*planted(100, 0.9, 0), enough).warnings)
    single = AuditConfig(
        n_permutations=30, n_bootstrap=20, families=("word",), alpha=0.05, report_features=False
    )
    assert not any("attainable" in w for w in run_audit(*planted(100, 0.9, 0), single).warnings)
    one = AuditConfig(
        n_permutations=30, n_bootstrap=20, families=("word",), alpha=0.01, report_features=False
    )
    assert any("1 feature family:" in w for w in run_audit(*planted(100, 0.9, 0), one).warnings)
