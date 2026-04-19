"""Public API for namegen.

Keep this module import-light. The heavy lifting is delegated to
:mod:`namegen.orchestrator`, which is loaded lazily so that importing the
package has negligible cost.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from namegen.profile import Profile, ProfileError

if TYPE_CHECKING:
    from namegen.orchestrator import Orchestrator

__all__ = ["GeneratedName", "Generator", "Profile", "ProfileError"]


@dataclass(frozen=True, slots=True)
class GeneratedName:
    """A single generated name plus everything needed to audit it.

    Attributes
    ----------
    text:
        The rendered name as it would appear to a human reader.
    realism:
        Composite realism score in ``[0, 1]``. Higher is more realistic.
    novelty:
        ``[0, 1]``. ``1.0`` means "not present in any training corpus";
        ``0.0`` means "verbatim copy of a known name".
    components:
        Per-axis score breakdown, e.g. ``{"phonotactic": 0.9, "orthographic": 0.8}``.
    models_used:
        Identifiers of the models that contributed to the candidate.
    provenance:
        Corpora + license tags that informed the underlying models for this
        language / era / slot.
    seed:
        RNG seed used for reproducibility of this single draw, when known.
    explanation:
        Optional human-readable rationale string built by the explanation
        layer when ``explain=True`` is passed to :meth:`Generator.generate`.
    """

    text: str
    realism: float
    novelty: float
    components: dict[str, float] = field(default_factory=dict)
    models_used: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    seed: int | None = None
    explanation: str | None = None


class Generator:
    """The public entry-point. Immutable once loaded.

    Examples
    --------
    >>> from namegen import Generator, Profile
    >>> g = Generator.load("default")
    >>> names = g.generate(Profile(language="en"), n=3, seed=42)
    >>> len(names)
    3
    """

    def __init__(self, orchestrator: Orchestrator, bundle_name: str) -> None:
        self._orchestrator = orchestrator
        self._bundle_name = bundle_name

    @classmethod
    def load(cls, bundle: str = "default") -> Generator:
        """Load the named model bundle.

        ``bundle="default"`` uses the small sample corpora shipped with the
        library. Future bundles (downloaded via ``namegen models download``)
        will register themselves by name.
        """
        # Imported lazily so that `import namegen` stays cheap and free of
        # heavy dependencies.
        from namegen.orchestrator import Orchestrator

        orchestrator = Orchestrator.from_bundle(bundle)
        return cls(orchestrator, bundle)

    @property
    def bundle(self) -> str:
        return self._bundle_name

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return self._orchestrator.supported_languages()

    def generate(
        self,
        profile: Profile,
        *,
        n: int = 1,
        seed: int | None = None,
        explain: bool = False,
        max_attempts_per_name: int = 50,
    ) -> list[GeneratedName]:
        """Draw ``n`` names matching ``profile``.

        Parameters
        ----------
        profile:
            The conditioning vector. Invalid profiles raise
            :class:`ProfileError`.
        n:
            Number of names to return. Must be ``>= 1``.
        seed:
            Optional integer seed. Same seed + bundle + profile => same names.
        explain:
            When true, each result carries a human-readable ``explanation``.
        max_attempts_per_name:
            Validators may reject candidates; the orchestrator will retry up
            to this many times per slot before giving up and returning the
            best-scoring candidate it has seen.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if max_attempts_per_name < 1:
            raise ValueError(
                f"max_attempts_per_name must be >= 1, got {max_attempts_per_name}"
            )
        rng = random.Random(seed)
        return self._orchestrator.generate(
            profile,
            n=n,
            rng=rng,
            explain=explain,
            max_attempts=max_attempts_per_name,
            seed=seed,
        )

    def validate(self, text: str, profile: Profile) -> GeneratedName:
        """Score an existing name against ``profile``.

        Returns a :class:`GeneratedName` with scored components. Useful for
        the CLI ``namegen validate`` command and for regression tests.
        """
        return self._orchestrator.validate(text, profile)
