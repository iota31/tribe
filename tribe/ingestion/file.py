"""Read content from local files."""

from __future__ import annotations

import sys
from pathlib import Path


def read_file(path: str) -> str:
    """Read text content from a local file.

    Args:
        path: Path to the file. Use "-" for stdin.

    Returns:
        The file contents as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
    """
    if path == "-":
        text = sys.stdin.read()
        if not text.strip():
            raise ValueError("No input received from stdin")
        return text.strip()

    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = filepath.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise ValueError(f"File is empty: {path}")

    return text.strip()
