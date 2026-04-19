"""End-to-end tests for :class:`namegen.Generator`."""

from __future__ import annotations

import pytest

from namegen import GeneratedName, Generator, Profile, ProfileError


@pytest.fixture(scope="module")
def gen() -> Generator:
    return Generator.load("default")


def test_generate_returns_requested_n(gen: Generator) -> None:
    names = gen.generate(Profile(language="en", slot="given"), n=5, seed=7)
    assert len(names) == 5
    assert all(isinstance(r, GeneratedName) for r in names)
    assert all(r.text for r in names)


def test_generate_deterministic(gen: Generator) -> None:
    p = Profile(language="sv", slot="given", gender="f", era="1850-1900")
    a = gen.generate(p, n=5, seed=123)
    b = gen.generate(p, n=5, seed=123)
    assert [r.text for r in a] == [r.text for r in b]


def test_generate_distinct_within_batch(gen: Generator) -> None:
    # With a reasonable corpus and distinct retries the orchestrator should
    # not return the exact same string twice in a row.
    p = Profile(language="en", slot="given")
    names = gen.generate(p, n=8, seed=5)
    texts = [r.text for r in names]
    assert len(set(texts)) >= len(texts) - 1  # allow at most one collision


def test_validate_known_name_scores_high(gen: Generator) -> None:
    p = Profile(language="en", slot="given")
    res = gen.validate("Mary", p)
    # Mary is in the training corpus so novelty should pin low, but the
    # phonotactic / orthographic axes must remain healthy.
    assert res.components["phonotactic"] >= 0.9
    assert res.components["orthographic"] >= 0.9
    assert res.novelty == 0.0


def test_generate_rejects_bad_n(gen: Generator) -> None:
    with pytest.raises(ValueError):
        gen.generate(Profile(language="en"), n=0)


def test_generate_unknown_language(gen: Generator) -> None:
    with pytest.raises(ProfileError):
        gen.generate(Profile(language="zz"), n=1)


def test_generator_bundle_and_languages(gen: Generator) -> None:
    assert gen.bundle == "default"
    assert "en" in gen.supported_languages
    assert "sv" in gen.supported_languages


def test_explanation_only_when_requested(gen: Generator) -> None:
    p = Profile(language="en", slot="given")
    names = gen.generate(p, n=1, seed=9, explain=False)
    assert names[0].explanation is None
    names2 = gen.generate(p, n=1, seed=9, explain=True)
    assert names2[0].explanation is not None


def test_provenance_carries_license(gen: Generator) -> None:
    p = Profile(language="sv", slot="surname")
    names = gen.generate(p, n=1, seed=1)
    prov = names[0].provenance
    assert any("CC0" in tag for tag in prov)
