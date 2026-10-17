"""Corpora of prompts: loading, validation and de-duplication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 20_000


class CorpusError(ValueError):
    """The input is not a usable corpus."""


@dataclass(frozen=True)
class Corpus:
    """Texts with optional group labels (documents sharing a group stay in one CV fold)."""

    texts: tuple[str, ...]
    groups: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.groups is not None and len(self.groups) != len(self.texts):
            raise CorpusError("groups must have one entry per text")

    def __len__(self) -> int:
        return len(self.texts)


def _clean(text: object, index: int) -> str:
    if not isinstance(text, str):
        raise CorpusError(f"item {index} is not a string")
    return text[:MAX_CHARS]


def corpus_from_items(items: list[object], group_field: str = "group") -> Corpus:
    """Build a corpus from strings or ``{"text": ..., "group": ...}`` objects."""
    texts: list[str] = []
    groups: list[str] = []
    have_groups = False
    for index, item in enumerate(items):
        if isinstance(item, dict):
            if "text" not in item:
                raise CorpusError(f"item {index} has no 'text' field")
            texts.append(_clean(item["text"], index))
            group = item.get(group_field)
            if group is not None:
                have_groups = True
            groups.append(str(group) if group is not None else f"__row{index}")
        else:
            texts.append(_clean(item, index))
            groups.append(f"__row{index}")
    return Corpus(tuple(texts), tuple(groups) if have_groups else None)


def load_corpus(path: Path | str, group_field: str = "group") -> Corpus:
    """Load ``.jsonl`` (strings or objects with ``text``), ``.json`` (a list) or plain text
    (one prompt per non-empty line)."""
    file = Path(path)
    try:
        raw = file.read_text("utf-8")
    except OSError as exc:
        raise CorpusError(f"cannot read {file}: {exc.strerror}") from exc
    except UnicodeDecodeError as exc:
        raise CorpusError(f"{file} is not valid UTF-8") from exc
    suffix = file.suffix.lower()
    items: list[object]
    try:
        if suffix == ".jsonl":
            items = [json.loads(line) for line in raw.splitlines() if line.strip()]
        elif suffix == ".json":
            loaded = json.loads(raw)
            if not isinstance(loaded, list):
                raise CorpusError(f"{file} must contain a JSON list")
            items = loaded
        else:
            items = [line for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{file}: invalid JSON: {exc}") from exc
    if not items:
        raise CorpusError(f"{file} contains no prompts")
    return corpus_from_items(items, group_field)


def dedupe(corpus: Corpus) -> tuple[Corpus, int]:
    """Drop exact duplicate texts (keeping the first); returns the corpus and how many went."""
    seen: set[str] = set()
    keep: list[int] = []
    for index, text in enumerate(corpus.texts):
        if text not in seen:
            seen.add(text)
            keep.append(index)
    removed = len(corpus) - len(keep)
    if removed == 0:
        return corpus, 0
    texts = tuple(corpus.texts[i] for i in keep)
    groups = None if corpus.groups is None else tuple(corpus.groups[i] for i in keep)
    return Corpus(texts, groups), removed


def cross_duplicates(first: Corpus, second: Corpus) -> int:
    """Number of distinct texts that occur in both corpora."""
    return len(set(first.texts) & set(second.texts))
