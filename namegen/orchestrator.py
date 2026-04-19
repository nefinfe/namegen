"""Generation orchestrator.

Resolves a :class:`~namegen.profile.Profile` into a model + validator stack,
drives the candidate loop, and builds :class:`~namegen.api.GeneratedName`
results. It is deliberately state-light: all heavy structures (rule packs,
trained Markov models) are cached per profile-key so repeated calls with the
same profile are fast.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from namegen.data.loaders import (
    CorpusEntry,
    RulePack,
    available_languages,
    load_corpus,
    load_rule_pack,
)
from namegen.models.markov import CharMarkovModel, MarkovConfig
from namegen.profile import Profile, ProfileError
from namegen.validators.novelty import NoveltyValidator
from namegen.validators.orthographic import OrthographicValidator
from namegen.validators.phonotactic import PhonotacticValidator
from namegen.validators.scorer import RealismScorer, ValidatorResult

if TYPE_CHECKING:
    from namegen.api import GeneratedName


_SUPPORTED_BUNDLES: frozenset[str] = frozenset({"default"})


@dataclass(slots=True)
class _TrainedProfile:
    """Everything we cache per profile cache-key."""

    language: str
    model: CharMarkovModel
    rule_pack: RulePack
    phonotactic: PhonotacticValidator
    orthographic: OrthographicValidator
    novelty: NoveltyValidator
    scorer: RealismScorer
    corpus_entries: tuple[CorpusEntry, ...]
    corpus_names: frozenset[str] = field(default_factory=frozenset)
    provenance: tuple[str, ...] = ()


class Orchestrator:
    """Stateful, lazily-trained coordinator behind :class:`~namegen.api.Generator`."""

    def __init__(self, bundle_name: str) -> None:
        if bundle_name not in _SUPPORTED_BUNDLES:
            raise ValueError(
                f"unknown bundle {bundle_name!r}. Known: {sorted(_SUPPORTED_BUNDLES)}"
            )
        self._bundle_name = bundle_name
        self._cache: dict[tuple[object, ...], _TrainedProfile] = {}

    # -- construction ----------------------------------------------------
    @classmethod
    def from_bundle(cls, bundle_name: str) -> Orchestrator:
        return cls(bundle_name)

    def supported_languages(self) -> tuple[str, ...]:
        return tuple(available_languages())

    # -- public API ------------------------------------------------------
    def generate(
        self,
        profile: Profile,
        *,
        n: int,
        rng: random.Random,
        explain: bool,
        max_attempts: int,
        seed: int | None,
    ) -> list[GeneratedName]:
        trained = self._resolve(profile)
        results: list[GeneratedName] = []
        seen: set[str] = set()
        # Each returned name gets a private retry budget.
        for _ in range(n):
            best = self._draw_one(trained, rng, max_attempts, seen)
            results.append(self._to_result(best, trained, profile, explain, seed))
            seen.add(best.text.lower())
        return results

    def validate(self, text: str, profile: Profile) -> GeneratedName:
        trained = self._resolve(profile)
        components, model_prior = self._score_all(trained, text)
        composite = trained.scorer.score(
            {**components, "model_prior": model_prior}
        )
        novelty = trained.novelty.validate(text)
        return self._build_generated_name(
            text=text,
            components=components,
            realism=composite,
            novelty=novelty.score,
            trained=trained,
            profile=profile,
            explain=True,
            seed=None,
            model_prior=model_prior,
        )

    # -- internals -------------------------------------------------------
    def _resolve(self, profile: Profile) -> _TrainedProfile:
        key = profile.cache_key()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        lang = profile.primary_language
        try:
            rules = load_rule_pack(lang)
        except KeyError as e:
            raise ProfileError(
                f"no rule pack for language {lang!r}",
                suggestions=list(self.supported_languages()),
            ) from e
        entries = load_corpus(
            lang,
            slot=None if profile.slot == "full" else profile.slot,
            era=profile.era,
            gender=profile.gender if profile.gender != "any" else None,
        )
        if not entries:
            # Fall back to unfiltered corpus for the language rather than
            # failing outright; the validators will still honour the profile.
            entries = load_corpus(lang)
        if not entries:
            raise ProfileError(
                f"no corpora available for language {lang!r}",
                suggestions=list(self.supported_languages()),
            )

        corpus_names: list[str] = []
        provenance: list[str] = []
        for entry, names in entries:
            corpus_names.extend(names)
            provenance.append(f"{entry.id}[{entry.license}]")

        model = CharMarkovModel(MarkovConfig(order=4))
        model.fit(corpus_names)

        phono = PhonotacticValidator(rules.data)
        ortho = OrthographicValidator(rules.data)
        novelty = NoveltyValidator(set(corpus_names))
        scorer = RealismScorer()

        trained = _TrainedProfile(
            language=lang,
            model=model,
            rule_pack=rules,
            phonotactic=phono,
            orthographic=ortho,
            novelty=novelty,
            scorer=scorer,
            corpus_entries=tuple(e for e, _ in entries),
            corpus_names=frozenset(n.lower() for n in corpus_names),
            provenance=tuple(provenance),
        )
        self._cache[key] = trained
        return trained

    @dataclass(slots=True)
    class _Candidate:
        text: str
        components: dict[str, ValidatorResult]
        model_prior: float
        composite: float
        novelty: ValidatorResult

    def _draw_one(
        self,
        trained: _TrainedProfile,
        rng: random.Random,
        max_attempts: int,
        already_seen: set[str],
    ) -> Orchestrator._Candidate:
        best: Orchestrator._Candidate | None = None
        for _ in range(max_attempts):
            raw = trained.model.sample(rng)
            if not raw:
                continue
            # Title-case like a real name. Handles hyphens and spaces.
            text = _titlecase(raw)
            if text.lower() in already_seen:
                continue
            components, model_prior = self._score_all(trained, text)
            novelty = trained.novelty.validate(text)
            composite = trained.scorer.score(
                {**components, "model_prior": model_prior}
            )
            # Novelty clamp: verbatim copies are unusable even if everything
            # else is perfect.
            if novelty.score == 0.0:
                composite *= 0.0
            cand = Orchestrator._Candidate(
                text=text,
                components=components,
                model_prior=model_prior,
                composite=composite,
                novelty=novelty,
            )
            if best is None or cand.composite > best.composite:
                best = cand
            # 0.8 is high enough to accept immediately and stop burning attempts.
            if cand.composite >= 0.8 and novelty.score >= 0.8:
                return cand
        if best is not None:
            return best
        # Degenerate fallback; shouldn't happen with any non-empty corpus.
        return Orchestrator._Candidate(
            text="",
            components={},
            model_prior=0.0,
            composite=0.0,
            novelty=ValidatorResult(0.0, "no candidate"),
        )

    def _score_all(
        self, trained: _TrainedProfile, text: str
    ) -> tuple[dict[str, ValidatorResult], float]:
        phono = trained.phonotactic.validate(text)
        ortho = trained.orthographic.validate(text)
        # Map average log-prob in (-inf, 0] to a score in (0, 1]. The -2.5
        # denominator is roughly "average per-char log-prob at which a model
        # trained on a ~50-word corpus starts producing obviously weird
        # strings". Tuned empirically on the bundled sample data.
        logp = trained.model.log_prob(text)
        model_prior = math.exp(max(logp, -10.0) / 2.5)
        return {"phonotactic": phono, "orthographic": ortho}, min(
            max(model_prior, 0.0), 1.0
        )

    def _to_result(
        self,
        cand: Orchestrator._Candidate,
        trained: _TrainedProfile,
        profile: Profile,
        explain: bool,
        seed: int | None,
    ) -> GeneratedName:
        return self._build_generated_name(
            text=cand.text,
            components=cand.components,
            realism=cand.composite,
            novelty=cand.novelty.score,
            trained=trained,
            profile=profile,
            explain=explain,
            seed=seed,
            model_prior=cand.model_prior,
            novelty_reason=cand.novelty.reason,
        )

    def _build_generated_name(
        self,
        *,
        text: str,
        components: dict[str, ValidatorResult],
        realism: float,
        novelty: float,
        trained: _TrainedProfile,
        profile: Profile,
        explain: bool,
        seed: int | None,
        model_prior: float,
        novelty_reason: str = "",
    ) -> GeneratedName:
        from namegen.api import GeneratedName  # local import to break cycle

        breakdown = {k: v.score for k, v in components.items()}
        breakdown["model_prior"] = model_prior
        breakdown["novelty"] = novelty

        explanation: str | None = None
        if explain:
            bits: list[str] = []
            for key, res in components.items():
                bits.append(f"{key}={res.score:.2f} ({res.reason})")
            bits.append(f"model_prior={model_prior:.2f}")
            bits.append(f"novelty={novelty:.2f}" + (f" ({novelty_reason})" if novelty_reason else ""))
            explanation = (
                f"Generated for profile(lang={profile.primary_language}, era={profile.era}, "
                f"gender={profile.gender}, slot={profile.slot}). " + "; ".join(bits)
            )

        return GeneratedName(
            text=text,
            realism=realism,
            novelty=novelty,
            components=breakdown,
            models_used=("char-markov-o4",),
            provenance=trained.provenance,
            seed=seed,
            explanation=explanation,
        )


def _titlecase(raw: str) -> str:
    """Title-case a raw Markov sample, respecting hyphens and apostrophes."""
    if not raw:
        return raw
    parts: list[str] = []
    for chunk in raw.split(" "):
        sub_parts: list[str] = []
        for piece in chunk.split("-"):
            if piece:
                sub_parts.append(piece[:1].upper() + piece[1:])
            else:
                sub_parts.append(piece)
        parts.append("-".join(sub_parts))
    return " ".join(parts)
