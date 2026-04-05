"""Content ingestion — fetch and normalize content from URLs, files, and stdin."""

from tribe.ingestion.file import read_file
from tribe.ingestion.media import detect_media_type, is_media_file
from tribe.ingestion.url import fetch_url

__all__ = ["fetch_url", "read_file", "detect_media_type", "is_media_file"]
