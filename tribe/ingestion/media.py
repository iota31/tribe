"""Media file detection and preprocessing."""

from __future__ import annotations

from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}
TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".xml", ".json", ".csv"}


def detect_media_type(path: str) -> str:
    """Detect the media type of a file by extension.

    Args:
        path: Path to the file.

    Returns:
        One of: "video", "audio", "text"
    """
    ext = Path(path).suffix.lower()

    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "text"


def is_media_file(path: str) -> bool:
    """Check if a file is a video or audio file."""
    return detect_media_type(path) in ("video", "audio")
