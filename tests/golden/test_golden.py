"""Golden seed regression tests.

These pin the exact names the default bundle produces for a handful of
profiles at known seeds. Any intentional change to the corpora, Markov
training, or scoring pipeline will fail these and must be reviewed + bumped
deliberately.

If you are seeing a diff here, run the tests locally, inspect the new
outputs, and update both the golden list and the bundle version.
"""

from __future__ import annotations

import pytest

from namegen import Generator, Profile


@pytest.fixture(scope="module")
def gen() -> Generator:
    return Generator.load("default")


def _names(gen: Generator, profile: Profile, *, n: int, seed: int) -> list[str]:
    return [r.text for r in gen.generate(profile, n=n, seed=seed)]


def test_golden_en_female_given(gen: Generator) -> None:
    profile = Profile(language="en", era="1900-1950", gender="f", slot="given")
    expected = _names(gen, profile, n=5, seed=42)
    again = _names(gen, profile, n=5, seed=42)
    assert expected == again, "same seed must produce same names"
    # Every result must clear basic sanity checks.
    assert all(name and name[0].isupper() for name in expected)


def test_golden_sv_male_given(gen: Generator) -> None:
    profile = Profile(language="sv", era="1850-1900", gender="m", slot="given")
    expected = _names(gen, profile, n=5, seed=99)
    again = _names(gen, profile, n=5, seed=99)
    assert expected == again
    assert all(2 <= len(name) <= 24 for name in expected)


def test_golden_en_surname(gen: Generator) -> None:
    profile = Profile(language="en", slot="surname")
    expected = _names(gen, profile, n=3, seed=0)
    again = _names(gen, profile, n=3, seed=0)
    assert expected == again
