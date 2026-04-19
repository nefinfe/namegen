"""``namegen`` command-line interface."""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer

from namegen import GeneratedName, Generator, Profile, ProfileError
from namegen.data.loaders import available_languages, load_rule_pack

app = typer.Typer(
    name="namegen",
    help="Research-grade realistic name generator.",
    no_args_is_help=True,
    add_completion=False,
)

profiles_app = typer.Typer(
    name="profiles", help="Inspect available profile metadata.", no_args_is_help=True
)
models_app = typer.Typer(
    name="models", help="List and (future) download model bundles.", no_args_is_help=True
)
app.add_typer(profiles_app, name="profiles")
app.add_typer(models_app, name="models")


# --------------------------------------------------------------------- helpers
def _safe_build_profile(
    lang: str,
    era: str | None,
    gender: str,
    slot: str,
    region: str | None,
) -> Profile:
    try:
        return Profile(
            language=lang, era=era, gender=gender, slot=slot, region=region  # type: ignore[arg-type]
        )
    except ProfileError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        if e.suggestions:
            typer.secho(f"  try: {', '.join(e.suggestions)}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=2) from e


def _load_generator(bundle: str) -> Generator:
    try:
        return Generator.load(bundle)
    except Exception as e:
        typer.secho(f"error: cannot load bundle {bundle!r}: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e


# ------------------------------------------------------------------- generate
@app.command("generate", help="Generate names matching a profile.")
def generate(
    lang: Annotated[str, typer.Option("--lang", help="Primary language tag, e.g. en, sv.")] = "en",
    era: Annotated[str | None, typer.Option("--era", help="Era 'YYYY-YYYY'.")] = None,
    gender: Annotated[str, typer.Option("--gender", help="f, m, x, or any.")] = "any",
    slot: Annotated[str, typer.Option("--slot", help="given|surname|patronymic|matronymic|full.")] = "given",
    region: Annotated[str | None, typer.Option("--region", help="Optional region tag.")] = None,
    n: Annotated[int, typer.Option("--n", min=1, help="Number of names.")] = 5,
    seed: Annotated[int | None, typer.Option("--seed", help="Deterministic RNG seed.")] = None,
    explain: Annotated[bool, typer.Option("--explain", help="Print realism breakdown.")] = False,
    output: Annotated[
        str, typer.Option("--output", help="Output format: table|json|jsonl|csv.")
    ] = "table",
    bundle: Annotated[str, typer.Option("--bundle", help="Model bundle name.")] = "default",
) -> None:
    profile = _safe_build_profile(lang, era, gender, slot, region)
    gen = _load_generator(bundle)
    results = gen.generate(profile, n=n, seed=seed, explain=explain)

    fmt = output.lower()
    if fmt == "json":
        typer.echo(json.dumps([_name_to_dict(r) for r in results], indent=2, ensure_ascii=False))
        return
    if fmt == "jsonl":
        for r in results:
            typer.echo(json.dumps(_name_to_dict(r), ensure_ascii=False))
        return
    if fmt == "csv":
        typer.echo("name,realism,novelty")
        for r in results:
            typer.echo(f"{r.text},{r.realism:.4f},{r.novelty:.4f}")
        return

    # pretty table (default)
    width = max((len(r.text) for r in results), default=4)
    typer.echo(f"{'name'.ljust(width)}  realism  novelty")
    typer.echo(f"{'-' * width}  -------  -------")
    for r in results:
        typer.echo(f"{r.text.ljust(width)}  {r.realism:7.3f}  {r.novelty:7.3f}")
        if explain and r.explanation:
            typer.echo(f"    {r.explanation}")


def _name_to_dict(result: GeneratedName) -> dict[str, object]:
    # ``GeneratedName`` is a frozen dataclass; asdict keeps provenance tuples.
    from dataclasses import asdict

    return asdict(result)


# -------------------------------------------------------------------- validate
@app.command("validate", help="Score an existing name against a profile.")
def validate(
    text: Annotated[str, typer.Argument(help="The name to score.")],
    lang: Annotated[str, typer.Option("--lang")] = "en",
    era: Annotated[str | None, typer.Option("--era")] = None,
    gender: Annotated[str, typer.Option("--gender")] = "any",
    slot: Annotated[str, typer.Option("--slot")] = "given",
    bundle: Annotated[str, typer.Option("--bundle")] = "default",
) -> None:
    profile = _safe_build_profile(lang, era, gender, slot, None)
    gen = _load_generator(bundle)
    r = gen.validate(text, profile)
    typer.echo(f"{r.text}: realism={r.realism:.3f} novelty={r.novelty:.3f}")
    for k, v in r.components.items():
        typer.echo(f"  {k:<14} {v:.3f}")
    if r.explanation:
        typer.echo(r.explanation)


# -------------------------------------------------------------------- profiles
@profiles_app.command("list", help="List all languages with bundled corpora.")
def profiles_list() -> None:
    langs = available_languages()
    if not langs:
        typer.echo("(no languages registered)")
        return
    for lang in langs:
        typer.echo(lang)


@profiles_app.command("show", help="Show the rule pack for a language.")
def profiles_show(lang: Annotated[str, typer.Argument(help="Language tag, e.g. sv.")]) -> None:
    try:
        pack = load_rule_pack(lang)
    except KeyError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    typer.echo(f"language: {pack.language}")
    typer.echo(json.dumps(pack.data, indent=2, ensure_ascii=False, sort_keys=True))


# ---------------------------------------------------------------------- models
@models_app.command("list", help="List known model bundles.")
def models_list() -> None:
    # v0.1: only the bundled "default" pack exists. External bundles will be
    # registered here once M6 adds the neural bundle format.
    typer.echo("default")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app() or 0)
