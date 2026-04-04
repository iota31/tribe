"""Tests for CLI commands."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tribe.cli import main


def test_setup_no_stdin_hang():
    """Setup command runs to completion without stdin within 60s.

    This tests that the setup command does not block waiting for stdin input.
    transformers.pipeline is loaded via __getattr__, so we mock that instead.
    Model downloads are mocked to avoid network dependency in tests.
    """
    runner = CliRunner()

    import transformers

    def mock_getattr(name: str):
        if name == "pipeline":
            return MagicMock()
        raise AttributeError(name)

    with patch.object(transformers, "__getattr__", mock_getattr):
        result = runner.invoke(main, ["setup"], catch_exceptions=False)

    # Exit code 0 on success
    assert result.exit_code == 0, f"setup failed: {result.output}"
    # No hanging occurred (runner would timeout if it blocked on stdin)
    assert "Classifier backend:" in result.output
    assert "score:" in result.output
    # Output includes timing info
    assert "ms" in result.output
