"""Feature families: word n-grams, character n-grams, and surface statistics."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

SURFACE_NAMES: tuple[str, ...] = (
    "log_chars",
    "log_words",
    "mean_word_length",
    "log_lines",
    "frac_digits",
    "frac_upper",
    "frac_punctuation",
    "frac_whitespace",
    "frac_non_ascii",
    "question_marks",
    "code_fences",
    "has_url",
    "starts_upper",
    "ends_terminal_punctuation",
    "log_sentences",
)

_URL = re.compile(r"https?://", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
_PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def surface_features(text: str) -> list[float]:
    """Fifteen cheap formatting statistics (see :data:`SURFACE_NAMES`)."""
    chars = len(text)
    words = text.split()
    n_words = len(words)
    denom = max(chars, 1)
    stripped = text.rstrip()
    return [
        math.log1p(chars),
        math.log1p(n_words),
        (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        math.log1p(text.count("\n") + 1 if text else 0),
        sum(c.isdigit() for c in text) / denom,
        sum(c.isupper() for c in text) / denom,
        sum(c in _PUNCTUATION for c in text) / denom,
        sum(c.isspace() for c in text) / denom,
        sum(ord(c) > 127 for c in text) / denom,
        float(text.count("?")),
        float(text.count("```")),
        1.0 if _URL.search(text) else 0.0,
        1.0 if text[:1].isupper() else 0.0,
        1.0 if stripped[-1:] in {".", "!", "?"} else 0.0,
        math.log1p(len(_SENTENCE_END.findall(text))),
    ]


@dataclass
class FeatureSet:
    """A design matrix for one feature family, with feature names (for reporting)."""

    name: str
    matrix: Any  # scipy sparse or numpy array
    names: list[str]
    sparse: bool


def word_features(texts: list[str]) -> FeatureSet:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, lowercase=True, sublinear_tf=True, max_features=20_000
    )
    matrix = vectorizer.fit_transform(texts)
    return FeatureSet("word", matrix, list(vectorizer.get_feature_names_out()), True)


def char_features(texts: list[str]) -> FeatureSet:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        lowercase=False,
        sublinear_tf=True,
        max_features=30_000,
    )
    matrix = vectorizer.fit_transform(texts)
    return FeatureSet("char", matrix, list(vectorizer.get_feature_names_out()), True)


def surface_matrix(texts: list[str]) -> FeatureSet:
    raw = np.array([surface_features(t) for t in texts], dtype=float)
    scaled = StandardScaler().fit_transform(raw)
    return FeatureSet("surface", scaled, list(SURFACE_NAMES), False)


BUILDERS = {"word": word_features, "char": char_features, "surface": surface_matrix}


def build(family: str, texts: list[str]) -> FeatureSet:
    """Build one feature family; raises ``ValueError`` if no features survive."""
    if family not in BUILDERS:
        raise ValueError(f"unknown feature family {family!r}; choose from {sorted(BUILDERS)}")
    try:
        features = BUILDERS[family](texts)
    except ValueError as exc:  # e.g. an empty vocabulary after min_df pruning
        raise ValueError(f"{family} features unavailable: {exc}") from exc
    if features.matrix.shape[1] == 0:
        raise ValueError(f"{family} features unavailable: no features survived pruning")
    return features
