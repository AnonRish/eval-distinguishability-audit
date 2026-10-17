"""The audit: can a classifier tell evaluation prompts from real usage?

Method. For each feature family we train a regularised logistic regression with stratified
(optionally grouped) cross-validation, pool the out-of-fold scores and compute the AUC
(0.5 = indistinguishable, 1.0 = perfectly separable). The AUC gets a bootstrap interval and a
permutation p-value obtained by re-running the *whole* pipeline on shuffled labels. The best
family is tested with a Bonferroni correction across the families tried.

Caveats the report repeats: the vectorisers' vocabularies and idf weights are fitted on all
prompts (unsupervised, no labels), and the bootstrap interval treats out-of-fold scores as
independent, so it is slightly optimistic. A low AUC shows that *these* classifiers failed, not
that no signal exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import normalize

from .corpus import Corpus, cross_duplicates, dedupe
from .features import FeatureSet, build
from .stats import auc, bootstrap_ci, permutation_p

EFFECT_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.55, "negligible"),
    (0.65, "weak"),
    (0.80, "moderate"),
)
SMALL_SAMPLE = 50


class AuditError(ValueError):
    """The audit cannot be run on the given input."""


@dataclass(frozen=True)
class AuditConfig:
    """Settings for :func:`run_audit` (all deterministic given ``seed``)."""

    n_splits: int = 5
    seed: int = 0
    n_permutations: int = 100
    n_bootstrap: int = 500
    ci_level: float = 0.95
    alpha: float = 0.05
    min_df: int = 5
    top_k: int = 15
    families: tuple[str, ...] = ("word", "char", "surface")
    dedupe: bool = True
    ablation_ks: tuple[int, ...] = (5, 20, 50, 200)
    report_features: bool = True
    c: float = 1.0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise AuditError("n_splits must be >= 2")
        if self.n_permutations < 1 or self.n_bootstrap < 1:
            raise AuditError("n_permutations and n_bootstrap must be >= 1")
        if not 0.0 < self.ci_level < 1.0 or not 0.0 < self.alpha < 1.0:
            raise AuditError("ci_level and alpha must be in (0, 1)")
        if self.min_df < 1 or self.top_k < 1 or self.c <= 0:
            raise AuditError("min_df and top_k must be >= 1 and c must be positive")
        if not self.families:
            raise AuditError("at least one feature family is required")


@dataclass(frozen=True)
class FamilyResult:
    name: str
    auc: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float
    n_features: int


@dataclass(frozen=True)
class FeatureWeight:
    feature: str
    weight: float
    direction: str  # "eval" or "real"
    df_eval: int | None  # prompts containing the feature (None for dense features)
    df_real: int | None


@dataclass(frozen=True)
class AblationPoint:
    removed: int
    auc: float


@dataclass(frozen=True)
class AuditReport:
    n_eval: int
    n_real: int
    duplicates_removed: tuple[int, int]
    cross_class_duplicates: int
    grouped: bool
    families: tuple[FamilyResult, ...]
    best_family: str
    best_auc: float
    effect: str
    verdict: str  # "distinguishable" or "no_evidence"
    top_features: dict[str, tuple[FeatureWeight, ...]]
    ablation_family: str | None
    ablation: tuple[AblationPoint, ...]
    warnings: tuple[str, ...]
    config: dict[str, Any]

    @property
    def distinguishable(self) -> bool:
        return self.verdict == "distinguishable"

    def to_json(self) -> dict[str, Any]:
        return {
            "n_eval": self.n_eval,
            "n_real": self.n_real,
            "duplicates_removed": {
                "eval": self.duplicates_removed[0],
                "real": self.duplicates_removed[1],
            },
            "cross_class_duplicates": self.cross_class_duplicates,
            "grouped": self.grouped,
            "families": [f.__dict__ for f in self.families],
            "best_family": self.best_family,
            "best_auc": self.best_auc,
            "effect": self.effect,
            "verdict": self.verdict,
            "top_features": {
                name: [w.__dict__ for w in weights] for name, weights in self.top_features.items()
            },
            "ablation_family": self.ablation_family,
            "ablation": [a.__dict__ for a in self.ablation],
            "warnings": list(self.warnings),
            "config": self.config,
        }

    def render(self) -> str:
        cfg = self.config
        lines = [
            "Evaluation-vs-real distinguishability audit",
            f"  evaluation prompts: {self.n_eval} "
            f"(exact duplicates removed: {self.duplicates_removed[0]})",
            f"  real prompts:       {self.n_real} "
            f"(exact duplicates removed: {self.duplicates_removed[1]})",
            f"  {cfg['n_splits']}-fold CV (grouped: {'yes' if self.grouped else 'no'}), "
            f"{cfg['n_permutations']} permutations, {cfg['n_bootstrap']} bootstrap resamples, "
            f"seed {cfg['seed']}",
            "",
            f"{'family':<9}{'AUC':>7}  {'CI':<17}{'p':>8}{'p(adj)':>9}{'features':>10}",
        ]
        level = f"{cfg['ci_level']:.0%}"
        for f in self.families:
            ci = f"[{f.ci_low:.3f}, {f.ci_high:.3f}]"
            lines.append(
                f"{f.name:<9}{f.auc:>7.3f}  {ci:<17}{f.p_value:>8.4f}"
                f"{f.p_adjusted:>9.4f}{f.n_features:>10}"
            )
        lines.append(
            f"(intervals are {level} bootstrap intervals; p-values are one-sided permutation tests)"
        )
        lines.append("")
        label = "DISTINGUISHABLE" if self.distinguishable else "NO EVIDENCE OF DISTINGUISHABILITY"
        lines.append(
            f"Verdict: {label} (best family: {self.best_family}, AUC {self.best_auc:.3f}, "
            f"effect: {self.effect})"
        )
        for family, weights in self.top_features.items():
            if not weights:
                continue
            lines.append("")
            lines.append(
                f"Most discriminative {family} features (weight > 0 means evaluation-like):"
            )
            for w in weights:
                where = (
                    "dense feature"
                    if w.df_eval is None
                    else f"in {w.df_eval} eval / {w.df_real} real prompts"
                )
                lines.append(f"  {w.weight:+8.3f}  {w.feature!r:<28} {where}")
        if self.ablation:
            lines.append("")
            lines.append(
                f"Signal concentration ({self.ablation_family} features, "
                "AUC after removing the top-k):"
            )
            for point in self.ablation:
                lines.append(f"  k = {point.removed:<4} AUC {point.auc:.3f}")
        if self.warnings:
            lines.append("")
            lines.extend(f"WARNING: {w}" for w in self.warnings)
        lines.append("")
        lines.append(
            "Reading this: a high AUC means these classifiers can tell your evaluation prompts "
            "from real usage, so a model might too; a low AUC means these classifiers could not, "
            "which is "
            "weaker evidence than it sounds (a model may use features these do not)."
        )
        return "\n".join(lines)


def effect_label(value: float) -> str:
    """Conventional wording for an AUC (thresholds are conventions, not statistics)."""
    for limit, label in EFFECT_THRESHOLDS:
        if value < limit:
            return label
    return "strong"


def _folds(
    y: np.ndarray, groups: np.ndarray | None, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    placeholder = np.zeros(len(y))
    if groups is None:
        return list(
            StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(placeholder, y)
        )
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(placeholder, y, groups))


def _require_two_classes(train_labels: np.ndarray) -> None:
    if len(set(train_labels.tolist())) < 2:
        raise AuditError("a training fold contained a single class; use fewer folds")


def _oof_scores(
    matrix: Any, y: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]], c: float, seed: int
) -> np.ndarray:
    scores = np.zeros(len(y))
    for train, test in folds:
        _require_two_classes(y[train])
        model = LogisticRegression(C=c, solver="liblinear", random_state=seed)
        model.fit(matrix[train], y[train])
        scores[test] = model.decision_function(matrix[test])
    return scores


def _cv_auc(
    matrix: Any, y: np.ndarray, groups: np.ndarray | None, cfg: AuditConfig
) -> tuple[float, np.ndarray]:
    folds = _folds(y, groups, cfg.n_splits, cfg.seed)
    scores = _oof_scores(matrix, y, folds, cfg.c, cfg.seed)
    return auc(scores, y), scores


def _evaluate_family(
    features: FeatureSet, y: np.ndarray, groups: np.ndarray | None, cfg: AuditConfig, index: int
) -> tuple[float, float, float, float]:
    observed, scores = _cv_auc(features.matrix, y, groups, cfg)
    rng = np.random.default_rng([cfg.seed, index, 1])
    low, high = bootstrap_ci(scores, y, cfg.n_bootstrap, cfg.ci_level, rng)

    def statistic(permuted: np.ndarray) -> float:
        return _cv_auc(features.matrix, permuted, groups, cfg)[0]

    rng_p = np.random.default_rng([cfg.seed, index, 2])
    p_value = permutation_p(observed, y, statistic, cfg.n_permutations, rng_p)
    return observed, low, high, p_value


def _document_frequencies(matrix: Any, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    csr = matrix.tocsr()
    return csr[y == 1].getnnz(axis=0), csr[y == 0].getnnz(axis=0)


def _top_features(
    features: FeatureSet, y: np.ndarray, cfg: AuditConfig
) -> tuple[FeatureWeight, ...]:
    model = LogisticRegression(C=cfg.c, solver="liblinear", random_state=cfg.seed)
    model.fit(features.matrix, y)
    weights = model.coef_[0]
    df_eval: np.ndarray | None = None
    df_real: np.ndarray | None = None
    if features.sparse:
        df_eval, df_real = _document_frequencies(features.matrix, y)
        eligible = np.flatnonzero(df_eval + df_real >= cfg.min_df)
    else:
        eligible = np.arange(len(weights))
    order = eligible[np.argsort(-np.abs(weights[eligible]), kind="stable")][: cfg.top_k]
    return tuple(
        FeatureWeight(
            feature=features.names[i],
            weight=float(weights[i]),
            direction="eval" if weights[i] > 0 else "real",
            df_eval=None if df_eval is None else int(df_eval[i]),
            df_real=None if df_real is None else int(df_real[i]),
        )
        for i in order
    )


def _ablation(
    features: FeatureSet,
    y: np.ndarray,
    groups: np.ndarray | None,
    cfg: AuditConfig,
    baseline: float,
) -> tuple[AblationPoint, ...]:
    """AUC after removing the ``k`` most discriminative features, for each ``k``.

    The ranking is learned on each *training* fold only and the reduced model is scored on the
    held-out fold, so choosing which features to drop never sees the test labels (ranking on
    all the data would push the AUC below 0.5 for the leftover chance correlations).

    Features overlap (the bigrams around one telltale token all carry its signal), so a signal
    that comes from a single token can need many removals before the AUC reaches chance. The
    diagnostic is informative about *how fast* the AUC falls: a steep drop at small ``k`` means
    the signal is concentrated in a few features and easy to fix; a flat curve means it is
    diffuse.
    """
    n_features = features.matrix.shape[1]
    ks = [k for k in sorted(set(cfg.ablation_ks)) if k < n_features]
    scores = {k: np.zeros(len(y)) for k in ks}
    for train, test in _folds(y, groups, cfg.n_splits, cfg.seed):
        _require_two_classes(y[train])
        x_train, x_test = features.matrix[train], features.matrix[test]
        ranking_model = LogisticRegression(C=cfg.c, solver="liblinear", random_state=cfg.seed)
        ranking = np.argsort(-np.abs(ranking_model.fit(x_train, y[train]).coef_[0]), kind="stable")
        for k in ks:
            keep = np.setdiff1d(np.arange(n_features), ranking[:k])
            # Re-normalise the rows: tf-idf vectors are unit length, so the *norm* left over
            # after dropping columns would otherwise leak how much weight the dropped ones had.
            reduced_train = normalize(x_train[:, keep])
            reduced_test = normalize(x_test[:, keep])
            model = LogisticRegression(C=cfg.c, solver="liblinear", random_state=cfg.seed)
            model.fit(reduced_train, y[train])
            scores[k][test] = model.decision_function(reduced_test)
    return (AblationPoint(0, baseline), *(AblationPoint(k, auc(scores[k], y)) for k in ks))


def _combine(first: Corpus, second: Corpus) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    texts = [*first.texts, *second.texts]
    y = np.array([1] * len(first) + [0] * len(second))
    if first.groups is None and second.groups is None:
        return texts, y, None
    a = first.groups or tuple(f"row{i}" for i in range(len(first)))
    b = second.groups or tuple(f"row{i}" for i in range(len(second)))
    groups = np.array([f"e:{g}" for g in a] + [f"r:{g}" for g in b])
    return texts, y, groups


def run_audit(
    eval_corpus: Corpus, real_corpus: Corpus, config: AuditConfig | None = None
) -> AuditReport:
    """Run the audit and return a report. Deterministic for a fixed ``config.seed``."""
    cfg = config or AuditConfig()
    removed = (0, 0)
    if cfg.dedupe:
        eval_corpus, removed_eval = dedupe(eval_corpus)
        real_corpus, removed_real = dedupe(real_corpus)
        removed = (removed_eval, removed_real)
    smallest = min(len(eval_corpus), len(real_corpus))
    if smallest < max(10, cfg.n_splits):
        raise AuditError(
            f"need at least {max(10, cfg.n_splits)} distinct prompts per class "
            f"(smallest class has {smallest})"
        )
    texts, y, groups = _combine(eval_corpus, real_corpus)
    warnings: list[str] = []
    if smallest < SMALL_SAMPLE:
        warnings.append(
            f"small sample ({smallest} prompts in the smaller class): intervals are wide"
        )
    ratio = max(len(eval_corpus), len(real_corpus)) / smallest
    if ratio > 5:
        warnings.append(
            f"classes are imbalanced ({ratio:.1f}:1); AUC is robust to this but power is not"
        )
    shared = cross_duplicates(eval_corpus, real_corpus)
    if shared:
        warnings.append(
            f"{shared} prompt(s) appear in both corpora (indistinguishable by construction)"
        )

    results: list[tuple[FamilyResult, FeatureSet]] = []
    n_families = len(cfg.families)
    smallest_p = n_families / (1 + cfg.n_permutations)
    if smallest_p >= cfg.alpha:
        warnings.append(
            f"only {cfg.n_permutations} permutations across {n_families} feature famil"
            f"{'y' if n_families == 1 else 'ies'}: the smallest attainable adjusted p-value is "
            f"{smallest_p:.3f}, which is not below alpha={cfg.alpha}, so the verdict cannot be "
            "'distinguishable'; increase the number of permutations"
        )
    for index, name in enumerate(cfg.families):
        try:
            features = build(name, texts)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        observed, low, high, p_value = _evaluate_family(features, y, groups, cfg, index)
        result = FamilyResult(
            name=name,
            auc=observed,
            ci_low=low,
            ci_high=high,
            p_value=p_value,
            p_adjusted=min(1.0, p_value * n_families),
            n_features=features.matrix.shape[1],
        )
        results.append((result, features))
    if not results:
        raise AuditError("no feature family could be built: " + "; ".join(warnings))

    best, _ = max(results, key=lambda pair: pair[0].auc)
    significant = best.p_adjusted < cfg.alpha and best.ci_low > 0.5
    top: dict[str, tuple[FeatureWeight, ...]] = {}
    if cfg.report_features:
        for result, features in results:
            top[result.name] = _top_features(features, y, cfg)
    ablation: tuple[AblationPoint, ...] = ()
    ablation_family: str | None = None
    for result, features in results:
        if features.sparse and (ablation_family is None or result.name == "word"):
            ablation_family = result.name
            ablation = _ablation(features, y, groups, cfg, result.auc)
    return AuditReport(
        n_eval=len(eval_corpus),
        n_real=len(real_corpus),
        duplicates_removed=removed,
        cross_class_duplicates=shared,
        grouped=groups is not None,
        families=tuple(r for r, _ in results),
        best_family=best.name,
        best_auc=best.auc,
        effect=effect_label(best.auc),
        verdict="distinguishable" if significant else "no_evidence",
        top_features=top,
        ablation_family=ablation_family,
        ablation=ablation,
        warnings=tuple(warnings),
        config={
            "n_splits": cfg.n_splits,
            "seed": cfg.seed,
            "n_permutations": cfg.n_permutations,
            "n_bootstrap": cfg.n_bootstrap,
            "ci_level": cfg.ci_level,
            "alpha": cfg.alpha,
            "min_df": cfg.min_df,
            "families": list(cfg.families),
        },
    )
