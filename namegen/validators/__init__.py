"""Validators score candidate names on independent realism axes.

Each validator exposes a pure function ``validate(text, context) -> ValidatorResult``
so they can be unit-tested in isolation and composed by the scorer.
"""

from namegen.validators.novelty import NoveltyValidator
from namegen.validators.orthographic import OrthographicValidator
from namegen.validators.phonotactic import PhonotacticValidator
from namegen.validators.scorer import RealismScorer, ValidatorResult

__all__ = [
    "NoveltyValidator",
    "OrthographicValidator",
    "PhonotacticValidator",
    "RealismScorer",
    "ValidatorResult",
]
