"""Composite realism scorer.

The scorer is deliberately small: it takes per-axis scores in ``[0, 1]`` and
combines them with a weighted geometric mean. The geometric mean means a
single catastrophic failure (e.g. phonotactics 0.01) drags the composite
down, even if every other axis is 1.0 - which is what we want.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidatorResult:
    """Return value of any validator.

    Attributes
    ----------
    score:
        Quality score in ``[0, 1]``. Higher is better.
    reason:
        Short human-readable rationale. Surfaced to users via
        :meth:`Generator.validate` and the ``--explain`` CLI flag.
    """

    score: float
    reason: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


#: Default per-axis weights for :meth:`RealismScorer.score`. These are a
#: placeholder until M7's per-language tuning lands. Phonotactics is the
#: strongest signal we have in M1; novelty is checked separately and only
#: clamps the composite.
DEFAULT_WEIGHTS: dict[str, float] = {
    "phonotactic": 0.45,
    "orthographic": 0.20,
    "model_prior": 0.35,
}


class RealismScorer:
    """Combine per-axis :class:`ValidatorResult` scores into a composite."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        w = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        total = sum(w.values())
        if total <= 0:
            raise ValueError("at least one positive weight is required")
        # Normalise so callers can pass unnormalised weights.
        self._weights = {k: v / total for k, v in w.items() if v > 0}

    @property
    def weights(self) -> dict[str, float]:
        """Return a copy of the normalised weight mapping."""
        return dict(self._weights)

    def score(self, components: dict[str, ValidatorResult | float]) -> float:
        """Return the weighted geometric mean of ``components``.

        Unknown component keys are ignored so validators can evolve
        independently of the scorer.
        """
        log_sum = 0.0
        weight_sum = 0.0
        for key, weight in self._weights.items():
            if key not in components:
                continue
            raw = components[key]
            v = raw.score if isinstance(raw, ValidatorResult) else float(raw)
            # Floor tiny values so a single zero doesn't collapse log() to -inf.
            log_sum += weight * math.log(max(v, 1e-6))
            weight_sum += weight
        if weight_sum == 0:
            return 0.0
        return math.exp(log_sum / weight_sum)
