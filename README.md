# namegen

Research-grade realistic name generator. A Python-first library and CLI focused
on **linguistic realism**: names that obey the phonotactics, morphology,
orthography, and socio-historical distributions of real naming traditions.

> Status: **early alpha**. This repository currently ships milestone **M1 (skeleton
> and contracts)** with a functional baseline character-Markov generator, small
> bundled sample corpora for English and Swedish, and a CLI. See
> [plans/namegen-research-grade-realistic-name-generator.md](plans/namegen-research-grade-realistic-name-generator.md)
> for the full roadmap.

## Install

From a source checkout:

```bash
pip install -e '.[dev]'
```

Python 3.11+ required.

## Quick start

```python
from namegen import Generator, Profile

g = Generator.load("default")
profile = Profile(language="en", era="1900-1950", gender="f", slot="given")
for name in g.generate(profile, n=5, seed=42, explain=True):
    print(f"{name.text:<16} realism={name.realism:.3f}  novelty={name.novelty:.2f}")
```

Or via the CLI:

```bash
namegen generate --lang en --era 1900-1950 --gender f --n 10 --seed 42 --explain
namegen profiles list
namegen validate "Astrid Bergsdotter" --lang sv --era 1850-1900
```

## What is "realistic"?

A generated name is scored on six axes:

1. **Phonotactic validity** - its phoneme sequence is legal in the target language.
2. **Orthographic plausibility** - spelling follows language conventions.
3. **Morphological well-formedness** - patronymic suffixes, gendered endings, etc.
4. **Distributional fit** - rare names are rare, common ones common.
5. **Socio-historical consistency** - plausible for the requested era and region.
6. **Novelty** - not a verbatim copy of a corpus entry.

Each generated name carries a composite **Realism Score** plus per-axis
breakdown and provenance (which model, which corpus, which license).

## Non-goals (for v1)

- No web UI, no hosted API, no auth, no multi-tenant infra.
- No real-person impersonation; explicit novelty filter against training corpora.
- No "fantasy race" generators beyond what falls out of linguistic models.

## Repository layout

```
namegen/
  api.py              # Generator, Profile, GeneratedName
  orchestrator.py     # candidate pool + scoring + sampling
  models/             # markov, (future) phoneme markov, transformer, morphology
  validators/         # phonotactic, orthographic, novelty, scorer
  data/               # corpora manifest, rule packs, sample wordlists
  sampling/           # frequency-calibrated sampler
  cli/                # typer-based CLI
tests/
  unit/               # fast deterministic tests
  golden/             # seeded output regression tests
```

## Roadmap (milestones)

- [x] M1 - Skeleton, core types, CLI shell, sample corpora, char-Markov baseline
- [ ] M2 - Corpora manifest with license tags, full loader
- [ ] M3 - Kneser-Ney smoothing, golden tests per language
- [ ] M4 - G2P engine + phoneme-level Markov + real phonotactic validator
- [ ] M5 - Morphology rule packs and composer (Icelandic, Russian, Arabic)
- [ ] M6 - Small Transformer per language family (opt-in `[train]` extra)
- [ ] M7 - Realism scorer tuning, Zipf-correct sampler, do-not-impersonate list
- [ ] M8 - Explanations and full provenance records
- [ ] M9 - Evaluation harness (automatic + human)
- [ ] M10 - Language expansion: Japanese, Arabic, Spanish, Finnish, Yoruba
- [ ] M11 - 1.0 hardening, docs, SemVer freeze

## License

MIT. See [LICENSE](LICENSE). Bundled sample corpora are either public-domain or
synthetic and are license-tagged in [namegen/data/corpora.yaml](namegen/data/corpora.yaml).
