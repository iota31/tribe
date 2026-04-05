"""Fetch and extract article text from URLs."""

from __future__ import annotations

import httpx
from trafilatura import extract


def fetch_url(url: str, timeout: float = 30.0) -> str:
    """Fetch a URL and extract the main article text.

    Uses trafilatura to strip ads, navigation, boilerplate.

    Args:
        url: The URL to fetch and analyze.
        timeout: HTTP request timeout in seconds.

    Returns:
        Extracted article text.

    Raises:
        ValueError: If the URL cannot be fetched or no text is extracted.
    """
    response = httpx.get(
        url,
        timeout=timeout,
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; Tribe/0.1; " "+https://github.com/tribe-analyze/tribe)"
            )
        },
    )
    response.raise_for_status()

    text = extract(response.text, include_comments=False, include_tables=True)
    if not text or not text.strip():
        raise ValueError(f"Could not extract text content from {url}")

    return text.strip()
