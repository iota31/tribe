"""Main analysis orchestrator — ties ingestion, backends, and output together."""

from __future__ import annotations

import re

from tribe.backends.router import get_backend
from tribe.ingestion.file import read_file
from tribe.ingestion.media import detect_media_type, is_media_file
from tribe.ingestion.url import fetch_url
from tribe.schema import ContentAnalysis

URL_PATTERN = re.compile(r"^https?://")


def analyze(input_source: str) -> ContentAnalysis:
    """Analyze content for manipulation using TRIBE v2 brain encoding.

    Args:
        input_source: URL, file path, or "-" for stdin.

    Returns:
        ContentAnalysis with detected manipulation signals.
    """
    backend = get_backend()

    # Determine input type and ingest content
    if URL_PATTERN.match(input_source):
        text = fetch_url(input_source)
        result = backend.analyze_text(text)
        result.source_url = input_source
        return result

    if input_source == "-":
        text = read_file("-")
        return backend.analyze_text(text)

    # Local file
    if is_media_file(input_source):
        media_type = detect_media_type(input_source)
        return backend.analyze_media(input_source, media_type)

    text = read_file(input_source)
    return backend.analyze_text(text)
