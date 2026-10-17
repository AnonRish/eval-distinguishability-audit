"""eval-distinguishability-audit: can evaluation prompts be told apart from real usage?"""

from __future__ import annotations

__version__ = "0.1.0"

from .audit import (
    AblationPoint,
    AuditConfig,
    AuditError,
    AuditReport,
    FamilyResult,
    FeatureWeight,
    effect_label,
    run_audit,
)
from .corpus import Corpus, CorpusError, corpus_from_items, dedupe, load_corpus
from .stats import auc, bootstrap_ci, permutation_p
from .synth import expected_length_auc, expected_marker_auc, make_synthetic

__all__ = [
    "AblationPoint",
    "AuditConfig",
    "AuditError",
    "AuditReport",
    "Corpus",
    "CorpusError",
    "FamilyResult",
    "FeatureWeight",
    "__version__",
    "auc",
    "bootstrap_ci",
    "corpus_from_items",
    "dedupe",
    "effect_label",
    "expected_length_auc",
    "expected_marker_auc",
    "load_corpus",
    "make_synthetic",
    "permutation_p",
    "run_audit",
]
