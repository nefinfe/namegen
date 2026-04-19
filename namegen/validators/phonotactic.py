"""Phonotactic validator.

In M1 we do not yet run G2P. Instead, the validator checks cheap orthographic
proxies that correlate strongly with phonotactic legality in Latin-script
European languages:

* every codepoint is in the language's ``allowed_characters`` set
* no forbidden digraph (``qx``, ``bz``, ...) occurs
* vowel-harmony constraints are respected for the languages that declare them
* length is within ``min_length``/``max_length``

The full G2P-based checker arrives in M4. The interface below is stable: the
validator returns a :class:`ValidatorResult` with a ``reason`` string that
names the first constraint a candidate violated.
"""

from __future__ import annotations

from typing import Any

from namegen.validators.scorer import ValidatorResult


class PhonotacticValidator:
    """Lightweight orthographic-proxy phonotactic checker."""

    def __init__(self, rules: dict[str, Any]) -> None:
        phono: dict[str, Any] = dict(rules.get("phonotactics") or {})
        chars = phono.get("allowed_characters", "") or ""
        self._allowed: set[str] = set(str(chars).lower()) if chars else set()
        forbidden = phono.get("forbidden_digraphs") or []
        self._forbidden_digraphs: tuple[str, ...] = tuple(
            str(d).lower() for d in forbidden
        )
        self._min_length = int(phono.get("min_length", 2))
        self._max_length = int(phono.get("max_length", 24))
        vh = phono.get("vowel_harmony")
        self._vowel_harmony: tuple[frozenset[str], ...] | None = None
        if isinstance(vh, list) and all(isinstance(g, str) for g in vh):
            self._vowel_harmony = tuple(frozenset(g.lower()) for g in vh)

    # Public so other validators / tests can reuse the same sets.
    @property
    def allowed_characters(self) -> frozenset[str]:
        return frozenset(self._allowed)

    def validate(self, text: str) -> ValidatorResult:
        s = text.strip().lower()
        if len(s) < self._min_length:
            return ValidatorResult(0.0, f"too short (<{self._min_length})")
        if len(s) > self._max_length:
            return ValidatorResult(0.0, f"too long (>{self._max_length})")
        if self._allowed:
            # Treat the space between name components and a hyphen as always
            # allowed so we can validate full-name strings cheaply.
            extras = {" ", "-", "'"}
            for ch in s:
                if ch.isalpha() and ch not in self._allowed:
                    return ValidatorResult(0.0, f"disallowed character {ch!r}")
                if not ch.isalpha() and ch not in extras:
                    return ValidatorResult(0.0, f"disallowed character {ch!r}")
        for dg in self._forbidden_digraphs:
            if dg in s:
                return ValidatorResult(0.1, f"forbidden digraph {dg!r}")
        if self._vowel_harmony is not None:
            vowels = [c for c in s if any(c in g for g in self._vowel_harmony)]
            if vowels:
                groups = [g for g in self._vowel_harmony if vowels[0] in g]
                if groups:
                    home = groups[0]
                    if any(v not in home for v in vowels):
                        return ValidatorResult(
                            0.2, "vowel-harmony violation"
                        )
        return ValidatorResult(1.0, "ok")
