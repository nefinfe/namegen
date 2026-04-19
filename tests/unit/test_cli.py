"""Smoke tests for the typer CLI."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from namegen.cli.main import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "namegen" in result.stdout.lower()


def test_cli_generate_table() -> None:
    result = runner.invoke(app, ["generate", "--lang", "en", "--n", "3", "--seed", "1"])
    assert result.exit_code == 0, result.stdout
    assert "realism" in result.stdout


def test_cli_generate_json() -> None:
    result = runner.invoke(
        app,
        ["generate", "--lang", "sv", "--n", "2", "--seed", "2", "--output", "json"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert len(payload) == 2
    for item in payload:
        assert item["text"]
        assert 0.0 <= item["realism"] <= 1.0


def test_cli_profiles_list() -> None:
    result = runner.invoke(app, ["profiles", "list"])
    assert result.exit_code == 0
    assert "en" in result.stdout
    assert "sv" in result.stdout


def test_cli_profiles_show_unknown_language() -> None:
    result = runner.invoke(app, ["profiles", "show", "zz"])
    assert result.exit_code != 0


def test_cli_validate_known_name() -> None:
    result = runner.invoke(app, ["validate", "Mary", "--lang", "en", "--slot", "given"])
    assert result.exit_code == 0
    assert "Mary" in result.stdout


def test_cli_bad_era_is_reported() -> None:
    result = runner.invoke(app, ["generate", "--lang", "en", "--era", "bogus"])
    assert result.exit_code != 0
