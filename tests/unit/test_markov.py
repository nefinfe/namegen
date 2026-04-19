"""Unit tests for :mod:`namegen.models.markov`."""

from __future__ import annotations

import random

import pytest

from namegen.models.markov import CharMarkovModel, MarkovConfig


@pytest.fixture
def trained_model() -> CharMarkovModel:
    model = CharMarkovModel(MarkovConfig(order=3))
    model.fit([
        "anna",
        "anders",
        "astrid",
        "erik",
        "eva",
        "einar",
        "inga",
        "ingrid",
        "olaf",
        "olof",
    ])
    return model


def test_determinism_same_seed(trained_model: CharMarkovModel) -> None:
    a = [trained_model.sample(random.Random(1)) for _ in range(5)]
    b = [trained_model.sample(random.Random(1)) for _ in range(5)]
    assert a == b


def test_samples_are_ascii_lower(trained_model: CharMarkovModel) -> None:
    rng = random.Random(2)
    for _ in range(20):
        s = trained_model.sample(rng)
        assert s == s.lower()
        assert s.isascii()


def test_log_prob_trained_vs_gibberish(trained_model: CharMarkovModel) -> None:
    seen = trained_model.log_prob("anna")
    noise = trained_model.log_prob("zzqxqx")
    assert seen > noise


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        MarkovConfig(order=0)
    with pytest.raises(ValueError):
        MarkovConfig(backoff_weight=0.0)
    with pytest.raises(ValueError):
        MarkovConfig(max_length=1, min_length=5)


def test_untrained_model_returns_empty() -> None:
    model = CharMarkovModel()
    assert model.sample(random.Random(0)) == ""
    assert model.log_prob("anything") == float("-inf")
