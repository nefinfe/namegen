"""Loaders for corpora and rule packs.

The data layer is content-addressed via a top-level ``corpora.yaml`` manifest.
For M1 we only load name wordlists plus per-language rule YAMLs; full
license-tagged parquet corpora land in M2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import yaml

_CORPORA_MANIFEST = "corpora.yaml"


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """One corpus shard in the manifest.

    Attributes
    ----------
    id:
        Manifest-unique identifier, e.g. ``"en-given-sample"``.
    language:
        Primary language subtag.
    slot:
        Which :class:`~namegen.profile.NameSlot` the entries populate.
    path:
        Path relative to ``namegen/data/``.
    license:
        SPDX tag (``"CC0-1.0"``, ``"PDDL-1.0"``, ``"Synthetic"``, ...).
    era:
        Optional era string, matching :class:`~namegen.profile.Profile.era`.
    gender:
        Optional gender filter the corpus applies.
    source:
        Free-form source description carried through to name provenance.
    """

    id: str
    language: str
    slot: str
    path: str
    license: str
    era: str | None = None
    gender: str | None = None
    source: str = ""


@dataclass(frozen=True, slots=True)
class RulePack:
    """Per-language rule pack loaded from YAML."""

    language: str
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def _data_root() -> Any:
    return files("namegen.data")


def _read_manifest() -> dict[str, Any]:
    root = _data_root()
    path = root.joinpath(_CORPORA_MANIFEST)
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"{_CORPORA_MANIFEST} must be a mapping at the top level")
    return loaded


def _corpus_entries() -> list[CorpusEntry]:
    manifest = _read_manifest()
    raw = manifest.get("corpora", [])
    if not isinstance(raw, list):
        raise RuntimeError("'corpora' must be a list")
    out: list[CorpusEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError(f"corpus entry must be a mapping, got {type(item)}")
        try:
            out.append(
                CorpusEntry(
                    id=str(item["id"]),
                    language=str(item["language"]),
                    slot=str(item["slot"]),
                    path=str(item["path"]),
                    license=str(item["license"]),
                    era=item.get("era"),
                    gender=item.get("gender"),
                    source=str(item.get("source", "")),
                )
            )
        except KeyError as e:
            raise RuntimeError(f"corpus entry missing required key {e}") from e
    return out


def available_languages() -> list[str]:
    """Return the sorted list of primary language tags in the manifest."""
    return sorted({e.language for e in _corpus_entries()})


def load_corpus(
    language: str,
    slot: str | None = None,
    *,
    era: str | None = None,
    gender: str | None = None,
) -> list[tuple[CorpusEntry, list[str]]]:
    """Load every corpus matching the given filters.

    The return value is a list of ``(entry, names)`` pairs so callers can
    inspect the provenance of each shard.
    """
    root = _data_root()
    matched: list[tuple[CorpusEntry, list[str]]] = []
    for entry in _corpus_entries():
        if entry.language != language:
            continue
        if slot is not None and entry.slot != slot and entry.slot != "any":
            continue
        if era is not None and entry.era is not None and entry.era != era:
            continue
        if (
            gender is not None
            and entry.gender is not None
            and entry.gender not in (gender, "any")
        ):
            continue
        path = root.joinpath(entry.path)
        with path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        matched.append((entry, lines))
    return matched


def load_rule_pack(language: str) -> RulePack:
    """Load ``data/rules/<language>.yaml``. Unknown languages raise ``KeyError``."""
    root = _data_root()
    path = root.joinpath("rules", f"{language}.yaml")
    if not path.is_file():
        raise KeyError(f"no rule pack for language {language!r}")
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"rule pack for {language!r} must be a mapping")
    return RulePack(language=language, data=loaded)
