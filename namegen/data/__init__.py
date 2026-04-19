"""Data layer: corpora manifest, rule packs, bundled sample wordlists."""

from namegen.data.loaders import (
    CorpusEntry,
    RulePack,
    available_languages,
    load_corpus,
    load_rule_pack,
)

__all__ = [
    "CorpusEntry",
    "RulePack",
    "available_languages",
    "load_corpus",
    "load_rule_pack",
]
