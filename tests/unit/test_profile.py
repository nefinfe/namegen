"""Unit tests for :class:`namegen.Profile`."""

from __future__ import annotations

import dataclasses

import pytest

from namegen import Profile, ProfileError


def test_valid_profile_roundtrip() -> None:
    p = Profile(language="sv", era="1850-1900", gender="f", slot="given")
    assert p.primary_language == "sv"
    assert p.era_bounds == (1850, 1900)


def test_profile_is_frozen() -> None:
    p = Profile(language="en")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.language = "sv"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language": "ENGLISH"},
        {"language": "e"},
        {"language": "en-us"},  # region subtag must be upper-case per our regex
    ],
)
def test_invalid_language_rejected(kwargs: dict[str, str]) -> None:
    with pytest.raises(ProfileError):
        Profile(**kwargs)


@pytest.mark.parametrize("era", ["18-1900", "1900", "1900/2000", "2000-1900"])
def test_invalid_era_rejected(era: str) -> None:
    with pytest.raises(ProfileError):
        Profile(language="en", era=era)


def test_invalid_gender_and_slot() -> None:
    with pytest.raises(ProfileError) as exc:
        Profile(language="en", gender="female")  # type: ignore[arg-type]
    assert exc.value.suggestions
    with pytest.raises(ProfileError):
        Profile(language="en", slot="nickname")  # type: ignore[arg-type]


def test_cache_key_is_hashable_and_stable() -> None:
    p1 = Profile(language="en", era="1900-1950", gender="f")
    p2 = Profile(language="en", era="1900-1950", gender="f")
    assert p1.cache_key() == p2.cache_key()
    assert hash(p1.cache_key()) == hash(p2.cache_key())
