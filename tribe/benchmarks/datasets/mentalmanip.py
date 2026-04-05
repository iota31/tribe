"""MentalManip dataset downloader and parser.

Downloads conversational manipulation detection data from the MentalManip
GitHub repository and parses it using Python's csv module (NOT pandas,
which misparses the columns per the dataset authors).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/audreycs/MentalManip/main/"
    "mentalmanip_dataset/mentalmanip_con.csv"
)

_FILENAME = "mentalmanip_con.csv"


def download(data_dir: Path) -> None:
    """Download the MentalManip CSV dataset.

    Skips download if the file already exists.

    Args:
        data_dir: Directory to save the CSV into.

    Raises:
        httpx.HTTPStatusError: If the download fails.
    """
    csv_path = data_dir / _FILENAME
    if csv_path.exists():
        logger.info("MentalManip dataset already present at %s, skipping download.", csv_path)
        return

    logger.info("Downloading MentalManip dataset...")
    data_dir.mkdir(parents=True, exist_ok=True)
    response = httpx.get(DOWNLOAD_URL, follow_redirects=True, timeout=60.0)
    response.raise_for_status()

    csv_path.write_text(response.text, encoding="utf-8")
    logger.info("MentalManip dataset saved to %s.", csv_path)


def load(data_dir: Path) -> list[dict]:
    """Load the MentalManip dataset from a downloaded CSV.

    Args:
        data_dir: Directory containing the mentalmanip_con.csv file.

    Returns:
        List of dicts with keys: id, text, manipulative, techniques.

    Raises:
        FileNotFoundError: If the CSV file is missing.
    """
    csv_path = data_dir / _FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"MentalManip CSV not found at {csv_path}. Run download() first.")

    results: list[dict] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialogue_id = row.get("ID", "").strip()
            text = row.get("Dialogue", "").strip()
            manipulative_raw = row.get("Manipulative", "").strip()
            technique_raw = row.get("Technique", "").strip()

            manipulative = manipulative_raw.lower() in ("yes", "true", "1")

            # Parse techniques: may be empty string, single value, or semicolon-separated
            if technique_raw:
                techniques = [t.strip() for t in technique_raw.split(";") if t.strip()]
            else:
                techniques = []

            results.append(
                {
                    "id": dialogue_id,
                    "text": text,
                    "manipulative": manipulative,
                    "techniques": techniques,
                }
            )

    logger.info("Loaded %d dialogues from MentalManip dataset.", len(results))
    return results
