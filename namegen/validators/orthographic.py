"""Orthographic validator.

Checks spelling-level conventions independent of phonotactics:

* capitalization (each slot component starts with an uppercase letter)
* diacritic legality (every diacritic appears in the language's allowed set)
* no pathological repeated characters (``aaaa``)
"""

from __future__ import annotations

from typing import Any

from namegen.validators.scorer import ValidatorResult


class OrthographicValidator:
    """Spelling-convention checker with per-language diacritic allowlist."""

    def __init__(self, rules: dict[str, Any]) -> None:
        ortho: dict[str, Any] = dict(rules.get("orthography") or {})
        self._max_repeat: int = int(ortho.get("max_repeat", 3))
        self._require_capital: bool = bool(ortho.get("require_capital", True))
        diacritics = ortho.get("allowed_diacritics", "") or ""
        self._allowed_diacritics: frozenset[str] = frozenset(
            str(diacritics).lower()
        )
        # Everything else in the ascii range is always allowed.
        self._ascii_letters = frozenset("abcdefghijklmnopqrstuvwxyz")

    def validate(self, text: str) -> ValidatorResult:
        s = text.strip()
        if not s:
            return ValidatorResult(0.0, "empty")
        # Capitalization
        if self._require_capital:
            for part in s.replace("-", " ").split():
                if not part:
                    continue
                # Allow McX / O'Brien style compounds by only checking the first
                # letter - a fuller check lands with morphology in M5.
                if not part[0].isupper():
                    return ValidatorResult(0.3, "missing capital")
        # Repeated chars
        if self._max_repeat > 0:
            run = 1
            for a, b in zip(s.lower(), s.lower()[1:], strict=False):
                run = run + 1 if a == b else 1
                if run > self._max_repeat:
                    return ValidatorResult(0.2, f"run of {run} {a!r}")
        # Diacritics / unknown letters
        lower = s.lower()
        for ch in lower:
            if ch in self._ascii_letters or ch in {" ", "-", "'"}:
                continue
            if ch.isalpha() and ch not in self._allowed_diacritics:
                return ValidatorResult(0.3, f"unexpected letter {ch!r}")
        return ValidatorResult(1.0, "ok")
