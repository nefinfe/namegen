"""Tests for the data / corpora loaders."""

from __future__ import annotations

import pytest

from namegen.data.loaders import available_languages, load_corpus, load_rule_pack


def test_available_languages_covers_bundled_set() -> None:
    langs = available_languages()
    assert "en" in langs
    assert "sv" in langs


def test_load_corpus_returns_provenance_pairs() -> None:
    shards = load_corpus("en", slot="given")
    assert shards, "expected at least one English given-name shard"
    for entry, names in shards:
        assert entry.license
        assert entry.language == "en"
        assert names
        # Comments should be skipped, names should be stripped.
        for n in names:
            assert n == n.strip()
            assert not n.startswith("#")


def test_load_corpus_filters_by_gender() -> None:
    only_f = load_corpus("en", slot="given", gender="f")
    only_m = load_corpus("en", slot="given", gender="m")
    assert only_f and only_m
    assert {e.gender for e, _ in only_f} == {"f"}
    assert {e.gender for e, _ in only_m} == {"m"}


def test_load_rule_pack_unknown_raises() -> None:
    with pytest.raises(KeyError):
        load_rule_pack("zz")


def test_load_rule_pack_sv_has_diacritics() -> None:
    pack = load_rule_pack("sv")
    chars = pack.get("phonotactics", {}).get("allowed_characters", "")
    assert "å" in chars
    assert "ä" in chars
    assert "ö" in chars
