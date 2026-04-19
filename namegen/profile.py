"""Profile: the conditioning vector for generation.

A :class:`Profile` bundles the cultural / linguistic / social axes that the
generator conditions on. Profiles are frozen dataclasses so they can be used as
cache keys and safely shared across threads.

Invalid combinations (unknown language, malformed era string, etc.) are caught
eagerly at construction time and raise :class:`ProfileError` with a suggestion
payload the CLI can surface to users.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Gender = Literal["f", "m", "x", "any"]
NameSlot = Literal["given", "surname", "patronymic", "matronymic", "full"]

_ERA_PATTERN = re.compile(r"^(\d{3,4})-(\d{3,4})$")
_LANG_PATTERN = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")

_VALID_GENDERS: frozenset[str] = frozenset({"f", "m", "x", "any"})
_VALID_SLOTS: frozenset[str] = frozenset(
    {"given", "surname", "patronymic", "matronymic", "full"}
)


class ProfileError(ValueError):
    """Raised when a :class:`Profile` is built with an invalid combination.

    Carries an optional ``suggestions`` list of strings with near-matches or
    hints that the CLI surfaces back to users.
    """

    def __init__(self, message: str, *, suggestions: list[str] | None = None) -> None:
        super().__init__(message)
        self.suggestions = suggestions or []


@dataclass(frozen=True, slots=True)
class Profile:
    """Conditioning vector for name generation.

    Parameters
    ----------
    language:
        BCP-47-ish language tag: ``"en"``, ``"sv"``, ``"en-US"``, ...
        Only the primary subtag is used for model routing in v0.1.
    era:
        Half-open year range ``"YYYY-YYYY"``. ``None`` means "any era".
    gender:
        One of ``f``, ``m``, ``x``, ``any``.
    slot:
        Which name slot to produce.
    region:
        Optional free-form region tag carried through to provenance.
    social_class:
        Optional free-form class tag (``"peasant"``, ``"noble"``, ...).
        Unused by the baseline generator; preserved for later milestones.
    """

    language: str
    era: str | None = None
    gender: Gender = "any"
    slot: NameSlot = "given"
    region: str | None = None
    social_class: str | None = None
    extras: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not _LANG_PATTERN.match(self.language):
            raise ProfileError(
                f"invalid language tag: {self.language!r}",
                suggestions=["en", "sv", "en-US"],
            )
        if self.era is not None:
            m = _ERA_PATTERN.match(self.era)
            if not m:
                raise ProfileError(
                    f"invalid era {self.era!r}; expected 'YYYY-YYYY'",
                    suggestions=["1800-1850", "1900-1950", "2000-2025"],
                )
            start, end = int(m.group(1)), int(m.group(2))
            if start >= end:
                raise ProfileError(
                    f"era start {start} must be strictly less than end {end}"
                )
        if self.gender not in _VALID_GENDERS:
            raise ProfileError(
                f"invalid gender {self.gender!r}",
                suggestions=sorted(_VALID_GENDERS),
            )
        if self.slot not in _VALID_SLOTS:
            raise ProfileError(
                f"invalid slot {self.slot!r}",
                suggestions=sorted(_VALID_SLOTS),
            )

    @property
    def primary_language(self) -> str:
        """Return the primary language subtag (``"en"`` for ``"en-US"``)."""
        return self.language.split("-", 1)[0]

    @property
    def era_bounds(self) -> tuple[int, int] | None:
        """Return the ``(start, end)`` tuple for ``era`` or ``None``."""
        if self.era is None:
            return None
        m = _ERA_PATTERN.match(self.era)
        assert m is not None  # validated in __post_init__
        return int(m.group(1)), int(m.group(2))

    def cache_key(self) -> tuple[object, ...]:
        """Return a hashable, stable key for this profile."""
        return (
            self.primary_language,
            self.era,
            self.gender,
            self.slot,
            self.region,
            self.social_class,
            self.extras,
        )
