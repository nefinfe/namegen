"""Novelty filter.

Guards against three failure modes:

1. verbatim memorization of a training name (``exact_match`` -> novelty 0.0)
2. trivial perturbations (edit distance 1) of a training name
3. impersonation of a notable real person (``do_not_impersonate`` list)

The v0.1 implementation uses Python's :mod:`difflib` style Levenshtein
distance computed directly so we don't need an extra dependency. It is
plenty fast for the ~10k name corpora shipped in the default bundle.
"""

from __future__ import annotations

from namegen.validators.scorer import ValidatorResult


def _levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein distance. O(len(a)*len(b)) time, O(len(b)) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        curr[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[len(b)]


class NoveltyValidator:
    """Score how far a candidate is from any training / do-not-impersonate name."""

    def __init__(
        self,
        corpus: set[str] | None = None,
        *,
        do_not_impersonate: set[str] | None = None,
        edit_threshold: int = 1,
    ) -> None:
        self._corpus: set[str] = {c.lower() for c in (corpus or set())}
        self._dni: set[str] = {c.lower() for c in (do_not_impersonate or set())}
        if edit_threshold < 0:
            raise ValueError("edit_threshold must be >= 0")
        self._edit_threshold = edit_threshold

    def validate(self, text: str) -> ValidatorResult:
        s = text.strip().lower()
        if not s:
            return ValidatorResult(0.0, "empty")
        if s in self._dni:
            return ValidatorResult(0.0, "matches do-not-impersonate list")
        if s in self._corpus:
            return ValidatorResult(0.0, "verbatim match in corpus")
        if self._edit_threshold > 0:
            # Scan is O(N); corpora we ship are small. A trigram index is on
            # the roadmap for M7 when corpora grow past ~100k entries.
            for ref in self._corpus:
                # Cheap length filter so we don't compute Levenshtein on
                # obviously-unrelated strings.
                if abs(len(ref) - len(s)) > self._edit_threshold:
                    continue
                if _levenshtein(s, ref) <= self._edit_threshold:
                    return ValidatorResult(
                        0.2,
                        f"within edit distance {self._edit_threshold} of a known name",
                    )
        return ValidatorResult(1.0, "novel")

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)
