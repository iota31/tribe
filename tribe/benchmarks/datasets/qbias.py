"""Qbias/AllSides news dataset loader.

21,754 news articles labeled for political bias (left/center/right).
Source: https://github.com/irgroup/Qbias

For TRIBE benchmarking: binary task = extreme bias (left or right) vs center.
Centrist outlets tend toward factual reporting; left/right toward rhetoric.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

QBIAS_URL = (
    "https://raw.githubusercontent.com/irgroup/Qbias/main/"
    "allsides_balanced_news_headlines-texts.csv"
)


def download(data_dir: Path) -> Path:
    """Download Qbias CSV to data_dir if not already present."""
    csv_path = data_dir / "qbias_allsides.csv"
    if csv_path.exists():
        logger.info("Qbias dataset already present at %s", csv_path)
        return csv_path

    logger.info("Downloading Qbias from %s", QBIAS_URL)
    import httpx

    with httpx.stream("GET", QBIAS_URL, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with open(csv_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    logger.info("Qbias downloaded to %s", csv_path)
    return csv_path


def load(
    data_dir: Path | None = None,
    sample_size: int | None = None,
    balanced: bool = True,
) -> list[dict]:
    """Load Qbias articles with binary bias labels.

    Args:
        data_dir: Directory containing qbias_allsides.csv
        sample_size: If set, return this many samples total (half manipulative, half neutral)
        balanced: If True, balance the two classes (equal manipulative/neutral)

    Returns:
        List of dicts with: id, text, manipulative (bool), original_bias, source
        where manipulative=True for left/right bias, False for center.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    csv_path = data_dir / "qbias_allsides.csv"
    if not csv_path.exists():
        download(data_dir)

    items = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = row.get("text", "").strip()
            bias = row.get("bias_rating", "").strip().lower()
            title = row.get("title", "").strip()

            if not text or not bias:
                continue

            # Prepend title to text for more context
            full_text = f"{title}. {text}" if title else text

            items.append(
                {
                    "id": f"qbias_{i:06d}",
                    "text": full_text,
                    "manipulative": bias in ("left", "right"),
                    "original_bias": bias,
                    "source": row.get("source", ""),
                    "topic": row.get("tags", ""),
                }
            )

    if balanced:
        manip = [x for x in items if x["manipulative"]]
        neutral = [x for x in items if not x["manipulative"]]
        n = min(len(manip), len(neutral))
        # Deterministic shuffling for reproducibility
        import random

        rng = random.Random(42)
        rng.shuffle(manip)
        rng.shuffle(neutral)
        items = manip[:n] + neutral[:n]
        rng.shuffle(items)

    if sample_size is not None and sample_size < len(items):
        import random

        rng = random.Random(42)
        items = rng.sample(items, sample_size)

    logger.info(
        "Loaded %d Qbias items (%d manipulative, %d neutral)",
        len(items),
        sum(1 for x in items if x["manipulative"]),
        sum(1 for x in items if not x["manipulative"]),
    )

    return items
