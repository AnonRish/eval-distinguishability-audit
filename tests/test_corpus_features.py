from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_distinguishability import Corpus, CorpusError, corpus_from_items, dedupe, load_corpus
from eval_distinguishability.corpus import MAX_CHARS, cross_duplicates
from eval_distinguishability.features import SURFACE_NAMES, build, surface_features


def test_loading_jsonl_json_and_text(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text('{"text": "one", "group": "g1"}\n\n"two"\n')
    corpus = load_corpus(tmp_path / "a.jsonl")
    assert corpus.texts == ("one", "two") and corpus.groups is not None
    assert corpus.groups[0] == "g1" and corpus.groups[1].startswith("__row")
    (tmp_path / "b.json").write_text(json.dumps(["x", {"text": "y"}]))
    assert load_corpus(tmp_path / "b.json").texts == ("x", "y")
    assert load_corpus(tmp_path / "b.json").groups is None
    (tmp_path / "c.txt").write_text("first prompt\n\n  \nsecond prompt\n")
    assert load_corpus(tmp_path / "c.txt").texts == ("first prompt", "second prompt")
    (tmp_path / "d.jsonl").write_text('{"text": "t", "cluster": 5}\n')
    assert load_corpus(tmp_path / "d.jsonl", group_field="cluster").groups == ("5",)


@pytest.mark.parametrize(
    "name,content,message",
    [
        ("e.jsonl", "{not json}\n", "invalid JSON"),
        ("e.json", "{}", "JSON list"),
        ("e.json", "[1]", "not a string"),
        ("e.jsonl", '{"nope": 1}\n', "no 'text'"),
        ("e.txt", "\n\n", "no prompts"),
        ("e.json", "[]", "no prompts"),
    ],
)
def test_loading_errors(tmp_path: Path, name: str, content: str, message: str) -> None:
    (tmp_path / name).write_text(content)
    with pytest.raises(CorpusError, match=message):
        load_corpus(tmp_path / name)


def test_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="cannot read"):
        load_corpus(tmp_path / "missing.txt")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(CorpusError, match="UTF-8"):
        load_corpus(tmp_path / "bad.txt")


def test_long_texts_are_truncated_and_groups_validated() -> None:
    corpus = corpus_from_items(["x" * (MAX_CHARS + 50)])
    assert len(corpus.texts[0]) == MAX_CHARS
    with pytest.raises(CorpusError, match="one entry per text"):
        Corpus(("a", "b"), ("g",))
    assert len(Corpus(("a", "b"))) == 2


def test_dedupe_and_cross_duplicates() -> None:
    corpus = Corpus(("a", "b", "a", "c", "b"), ("1", "2", "3", "4", "5"))
    cleaned, removed = dedupe(corpus)
    assert cleaned.texts == ("a", "b", "c") and cleaned.groups == ("1", "2", "4") and removed == 2
    same, none_removed = dedupe(cleaned)
    assert same is cleaned and none_removed == 0
    plain, _ = dedupe(Corpus(("z", "z")))
    assert plain.texts == ("z",) and plain.groups is None
    assert cross_duplicates(Corpus(("a", "b", "c")), Corpus(("c", "d", "a"))) == 2


def test_surface_features_on_crafted_strings() -> None:
    names = dict(
        zip(
            SURFACE_NAMES,
            surface_features("Hello, World 42!\nSecond line? https://x.io ```code```"),
            strict=True,
        )
    )
    assert (
        names["question_marks"] == 1.0 and names["code_fences"] == 2.0 and names["has_url"] == 1.0
    )
    assert names["starts_upper"] == 1.0 and names["ends_terminal_punctuation"] == 0.0
    assert names["frac_digits"] == pytest.approx(
        2 / len("Hello, World 42!\nSecond line? https://x.io ```code```")
    )
    assert names["log_lines"] == pytest.approx(__import__("math").log1p(2))
    empty = dict(zip(SURFACE_NAMES, surface_features(""), strict=True))
    assert all(v == 0.0 for v in empty.values())
    ends = dict(zip(SURFACE_NAMES, surface_features("A question?"), strict=True))
    assert ends["ends_terminal_punctuation"] == 1.0 and ends["log_sentences"] > 0
    unicode_ = dict(zip(SURFACE_NAMES, surface_features("caf\u00e9"), strict=True))
    assert unicode_["frac_non_ascii"] == pytest.approx(0.25)
    assert len(SURFACE_NAMES) == len(surface_features("x"))


def test_feature_families_have_names_and_shapes() -> None:
    texts = [f"alpha beta gamma {i % 3} delta" for i in range(30)]
    for family, sparse in (("word", True), ("char", True), ("surface", False)):
        features = build(family, texts)
        assert features.sparse is sparse and features.matrix.shape == (30, len(features.names))
    assert build("surface", texts).names == list(SURFACE_NAMES)


def test_unavailable_feature_families_raise() -> None:
    with pytest.raises(ValueError, match="unknown feature family"):
        build("semantic", ["a"])
    with pytest.raises(ValueError, match="word features unavailable"):
        build("word", ["a b", "c d", "e f"])  # only one-character tokens: empty vocabulary
