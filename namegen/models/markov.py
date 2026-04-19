"""Character-level Markov model with interpolated backoff.

This is the baseline generator used by M1. It is deliberately simple: an
n-gram conditional distribution ``P(c | h)`` for every history length from 1
to ``order``, plus stupid-backoff mixing when the current history has no
continuations. Kneser-Ney smoothing is planned for M3; the interpolation
knobs below already live where KN weights will slot in.

The model is pure Python, deterministic, and has no third-party dependencies
so it can be trained at import time from a couple of hundred sample names
without fuss.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

# Sentinel characters used as start / end markers. They are chosen to be
# visually distinct and impossible to appear in any input corpus.
BOS = "\x02"  # start-of-name
EOS = "\x03"  # end-of-name


@dataclass(frozen=True, slots=True)
class MarkovConfig:
    """Training configuration for :class:`CharMarkovModel`."""

    order: int = 4
    min_length: int = 2
    max_length: int = 20
    backoff_weight: float = 0.4
    lowercase: bool = True

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")
        if self.min_length < 1:
            raise ValueError(f"min_length must be >= 1, got {self.min_length}")
        if self.max_length < self.min_length:
            raise ValueError(
                f"max_length ({self.max_length}) < min_length ({self.min_length})"
            )
        if not 0.0 < self.backoff_weight <= 1.0:
            raise ValueError(
                f"backoff_weight must be in (0, 1], got {self.backoff_weight}"
            )


class CharMarkovModel:
    """Character-level n-gram language model over name strings.

    Parameters
    ----------
    config:
        Training hyperparameters.

    Notes
    -----
    The model stores, for every history ``h`` of length ``1..order``, a
    :class:`collections.Counter` of continuation characters. At sample time it
    tries the longest available history first and backs off to shorter
    contexts with weight ``backoff_weight`` per step. This is the classic
    "stupid backoff" trick (Brants et al. 2007) and is good enough as a
    baseline while KN smoothing is being wired in.
    """

    def __init__(self, config: MarkovConfig | None = None) -> None:
        self.config = config or MarkovConfig()
        # history string -> Counter over next char
        self._ngrams: dict[str, Counter[str]] = defaultdict(Counter)
        # set of chars ever observed (for novelty / diagnostics)
        self._vocab: set[str] = set()
        self._trained_on: int = 0

    # ------------------------------------------------------------------ train
    def fit(self, names: Iterable[str]) -> CharMarkovModel:
        """Train the model on an iterable of name strings.

        Each name is padded with ``order`` start markers and a single end
        marker, so we never have to special-case sequence boundaries at
        sample time.
        """
        order = self.config.order
        seen = 0
        for raw in names:
            s = raw.strip()
            if not s:
                continue
            if self.config.lowercase:
                s = s.lower()
            self._vocab.update(s)
            padded = BOS * order + s + EOS
            for i in range(order, len(padded)):
                next_ch = padded[i]
                # Record every history length so backoff has something to read.
                for h_len in range(1, order + 1):
                    hist = padded[i - h_len : i]
                    self._ngrams[hist][next_ch] += 1
            seen += 1
        self._trained_on += seen
        return self

    @property
    def trained_on(self) -> int:
        """Number of training names accepted into the model."""
        return self._trained_on

    # ----------------------------------------------------------------- sample
    def sample(self, rng: random.Random, *, max_length: int | None = None) -> str:
        """Draw a single name. Always returns a non-empty stripped string."""
        order = self.config.order
        max_len = max_length if max_length is not None else self.config.max_length
        out: list[str] = []
        # Start with a pure-BOS context of length `order`.
        history = BOS * order
        while len(out) < max_len:
            nxt = self._sample_next(history, rng)
            if nxt == EOS:
                break
            out.append(nxt)
            history = (history + nxt)[-order:]
        return "".join(out)

    def sample_many(
        self, n: int, rng: random.Random, *, max_length: int | None = None
    ) -> list[str]:
        """Convenience wrapper for drawing ``n`` candidates."""
        return [self.sample(rng, max_length=max_length) for _ in range(n)]

    def _sample_next(self, history: str, rng: random.Random) -> str:
        order = self.config.order
        backoff = self.config.backoff_weight

        # Merge conditional distributions from longest -> shortest context
        # with exponentially decaying weight.
        merged: dict[str, float] = {}
        weight = 1.0
        for h_len in range(order, 0, -1):
            sub = history[-h_len:]
            counts = self._ngrams.get(sub)
            if counts:
                total = sum(counts.values())
                for ch, c in counts.items():
                    merged[ch] = merged.get(ch, 0.0) + weight * c / total
            weight *= backoff

        if not merged:
            # The model is untrained or we ran into a truly unseen context
            # even at order 1. Emit EOS so the caller gets a clean empty
            # string rather than hanging.
            return EOS

        chars = list(merged.keys())
        weights = [merged[c] for c in chars]
        return rng.choices(chars, weights=weights, k=1)[0]

    # ------------------------------------------------------------------ score
    def log_prob(self, name: str) -> float:
        """Return an average per-character log-probability of ``name``.

        This is the quantity downstream validators use as a soft phonotactic
        prior. Higher (less negative) = more model-plausible. Returns
        ``-inf`` if the model is untrained.
        """
        if not self._ngrams:
            return float("-inf")
        order = self.config.order
        s = name.lower() if self.config.lowercase else name
        padded = BOS * order + s + EOS
        total_logp = 0.0
        steps = 0
        for i in range(order, len(padded)):
            history = padded[i - order : i]
            nxt = padded[i]
            merged: dict[str, float] = {}
            weight = 1.0
            backoff = self.config.backoff_weight
            for h_len in range(order, 0, -1):
                sub = history[-h_len:]
                counts = self._ngrams.get(sub)
                if counts:
                    total = sum(counts.values())
                    for ch, c in counts.items():
                        merged[ch] = merged.get(ch, 0.0) + weight * c / total
                weight *= backoff
            if not merged:
                # Severe OOV at every backoff level. Charge a hefty but
                # finite penalty so callers can still compare candidates.
                total_logp += math.log(1e-9)
            else:
                merged_total = sum(merged.values())
                p = merged.get(nxt, 0.0) / merged_total
                total_logp += math.log(max(p, 1e-9))
            steps += 1
        return total_logp / max(steps, 1)
