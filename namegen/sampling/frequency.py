"""Frequency-calibrated resampling.

Draws from a candidate pool while keeping the empirical frequency profile
Zipfian-correct against an era frequency table.

**Status**: placeholder for M1. The class validates its own interface so
downstream code can be written against it today, but the ``calibrate`` method
is currently a pass-through that preserves insertion order.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence


class FrequencyCalibrator:
    """Reweight a candidate pool toward a target frequency distribution.

    Parameters
    ----------
    target_distribution:
        Mapping from ``rank -> probability mass`` or from ``name -> mass``.
        When ``None`` the calibrator is a pass-through.
    """

    def __init__(
        self, target_distribution: dict[str, float] | dict[int, float] | None = None
    ) -> None:
        self._target = target_distribution or {}

    def calibrate(
        self,
        candidates: Sequence[str],
        rng: random.Random,
        *,
        n: int,
    ) -> list[str]:
        """Return ``n`` names drawn from ``candidates``.

        With no target distribution this is just a uniform draw without
        replacement (falling back to draws with replacement if the pool is
        smaller than ``n``).
        """
        if n <= 0:
            return []
        if not candidates:
            return []
        pool = list(candidates)
        if not self._target:
            if len(pool) >= n:
                return rng.sample(pool, n)
            return [rng.choice(pool) for _ in range(n)]
        # M7 will replace this with proper importance reweighting.
        sample_key = next(iter(self._target))
        if isinstance(sample_key, str):
            name_targets: dict[str, float] = self._target  # type: ignore[assignment]
            weights = [float(name_targets.get(c, 1.0)) for c in pool]
        else:
            weights = [1.0 for _ in pool]
        return rng.choices(pool, weights=weights, k=n)

    def known_targets(self) -> Iterable[str | int]:
        return self._target.keys()
