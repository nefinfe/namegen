"""Unit tests for validators."""

from __future__ import annotations

from namegen.data.loaders import load_rule_pack
from namegen.validators.novelty import NoveltyValidator, _levenshtein
from namegen.validators.orthographic import OrthographicValidator
from namegen.validators.phonotactic import PhonotacticValidator
from namegen.validators.scorer import RealismScorer, ValidatorResult


def test_phonotactic_allows_standard_english() -> None:
    rules = load_rule_pack("en")
    v = PhonotacticValidator(rules.data)
    assert v.validate("Mary").score == 1.0


def test_phonotactic_rejects_forbidden_digraph() -> None:
    rules = load_rule_pack("en")
    v = PhonotacticValidator(rules.data)
    res = v.validate("Qxander")
    assert res.score < 0.5
    assert "qx" in res.reason


def test_phonotactic_too_short_and_too_long() -> None:
    rules = load_rule_pack("en")
    v = PhonotacticValidator(rules.data)
    assert v.validate("a").score == 0.0
    assert v.validate("x" * 40).score == 0.0


def test_phonotactic_disallowed_char_en_but_allowed_sv() -> None:
    en = PhonotacticValidator(load_rule_pack("en").data)
    sv = PhonotacticValidator(load_rule_pack("sv").data)
    assert en.validate("Åke").score == 0.0
    assert sv.validate("Åke").score == 1.0


def test_orthographic_requires_capital() -> None:
    v = OrthographicValidator(load_rule_pack("en").data)
    assert v.validate("Anna").score == 1.0
    assert v.validate("anna").score < 1.0


def test_orthographic_runs_of_letters() -> None:
    v = OrthographicValidator(load_rule_pack("en").data)
    assert v.validate("Aaaaaarg").score < 0.5


def test_novelty_verbatim_match() -> None:
    v = NoveltyValidator({"Anna", "Erik"})
    assert v.validate("anna").score == 0.0
    assert v.validate("Anders").score == 1.0


def test_novelty_edit_threshold() -> None:
    v = NoveltyValidator({"Erik"}, edit_threshold=1)
    res = v.validate("Erika")  # edit distance 1
    assert res.score < 1.0
    assert v.validate("Beatrix").score == 1.0


def test_levenshtein_basic() -> None:
    assert _levenshtein("", "") == 0
    assert _levenshtein("abc", "abc") == 0
    assert _levenshtein("abc", "abd") == 1
    assert _levenshtein("kitten", "sitting") == 3


def test_realism_scorer_geometric_mean() -> None:
    s = RealismScorer({"phonotactic": 1.0, "orthographic": 1.0, "model_prior": 1.0})
    score = s.score(
        {
            "phonotactic": ValidatorResult(1.0, "ok"),
            "orthographic": ValidatorResult(1.0, "ok"),
            "model_prior": 1.0,
        }
    )
    assert score == 1.0

    dragged = s.score(
        {
            "phonotactic": ValidatorResult(0.01, "violation"),
            "orthographic": ValidatorResult(1.0, "ok"),
            "model_prior": 1.0,
        }
    )
    # geometric mean is much lower than the arithmetic mean (~0.67)
    assert dragged < 0.5
