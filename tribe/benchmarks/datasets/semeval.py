"""SemEval-2020 Task 11 dataset downloader and parser.

Downloads propaganda detection benchmark data from Zenodo and parses
article texts with their technique-level span annotations.
"""

from __future__ import annotations

import io
import logging
import re
import tarfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DOWNLOAD_URL = "https://zenodo.org/api/records/3952415/files/datasets-v2.tgz/content"

_ARTICLE_PATTERN = re.compile(r"article(\d+)\.txt$")


def download(data_dir: Path) -> None:
    """Download and extract the SemEval-2020 Task 11 dataset.

    Skips download if the extracted data directory already exists.

    Args:
        data_dir: Directory to extract the dataset into.

    Raises:
        httpx.HTTPStatusError: If the download fails.
    """
    articles_dir = data_dir / "datasets" / "train-articles"
    if articles_dir.exists() and any(articles_dir.glob("*.txt")):
        logger.info("SemEval dataset already present at %s, skipping download.", data_dir)
        return

    logger.info("Downloading SemEval-2020 Task 11 dataset from Zenodo...")
    response = httpx.get(DOWNLOAD_URL, follow_redirects=True, timeout=120.0)
    response.raise_for_status()

    logger.info("Extracting archive to %s...", data_dir)
    buf = io.BytesIO(response.content)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")

    logger.info("SemEval dataset ready at %s.", data_dir)


def _parse_annotations(labels_path: Path) -> dict[str, list[tuple[str, int, int]]]:
    """Parse a TSV annotation file into a mapping of article_id -> spans.

    Args:
        labels_path: Path to the TSV labels file.

    Returns:
        Dict mapping article ID strings to lists of (technique, start, end) tuples.
    """
    annotations: dict[str, list[tuple[str, int, int]]] = {}
    text = labels_path.read_text(encoding="utf-8").strip()
    if not text:
        return annotations

    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 4:
            logger.warning("Skipping malformed annotation line: %s", line)
            continue
        article_id, technique, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])
        annotations.setdefault(article_id, []).append((technique, start, end))

    return annotations


def load(data_dir: Path) -> list[dict]:
    """Load the SemEval-2020 Task 11 dataset from an extracted directory.

    Args:
        data_dir: Root directory containing the extracted dataset.

    Returns:
        List of dicts with keys: id, text, propaganda_spans, propaganda_density.

    Raises:
        FileNotFoundError: If the expected directories are missing.
    """
    articles_dir = data_dir / "datasets" / "train-articles"
    if not articles_dir.exists():
        raise FileNotFoundError(f"Articles directory not found: {articles_dir}")

    # Collect all annotations from all label files
    labels_dir = data_dir / "datasets" / "train-labels-task2-technique-classification"
    all_annotations: dict[str, list[tuple[str, int, int]]] = {}
    if labels_dir.exists():
        for labels_file in labels_dir.glob("*.labels"):
            parsed = _parse_annotations(labels_file)
            for aid, spans in parsed.items():
                all_annotations.setdefault(aid, []).extend(spans)

    results: list[dict] = []
    for article_path in sorted(articles_dir.glob("article*.txt")):
        match = _ARTICLE_PATTERN.search(article_path.name)
        if not match:
            continue

        article_id = match.group(1)
        text = article_path.read_text(encoding="utf-8", errors="replace")
        spans = all_annotations.get(article_id, [])

        # Calculate propaganda density: total propaganda chars / total chars
        total_chars = len(text)
        if total_chars == 0 or not spans:
            density = 0.0
        else:
            # Use a set to avoid double-counting overlapping spans
            propaganda_chars: set[int] = set()
            for _, start, end in spans:
                propaganda_chars.update(range(start, end))
            density = len(propaganda_chars) / total_chars

        results.append(
            {
                "id": article_id,
                "text": text,
                "propaganda_spans": spans,
                "propaganda_density": density,
            }
        )

    logger.info("Loaded %d articles from SemEval dataset.", len(results))
    return results
