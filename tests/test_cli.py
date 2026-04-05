"""Tests for CLI commands."""

from click.testing import CliRunner

from tribe.cli import main


def test_version_command():
    """Version command outputs version string."""
    runner = CliRunner()
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    assert "Tribe v" in result.output
    assert "Neural Content Analysis" in result.output


def test_backends_command():
    """Backends command runs without error."""
    runner = CliRunner()
    result = runner.invoke(main, ["backends"])
    assert result.exit_code == 0
    assert "Backend Status" in result.output
    assert "Hardware:" in result.output
    assert "TRIBE v2 Rust:" in result.output
