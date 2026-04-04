"""Tests for content ingestion."""

import pytest

from tribe.ingestion.file import read_file
from tribe.ingestion.media import detect_media_type, is_media_file


def test_detect_video():
    assert detect_media_type("video.mp4") == "video"
    assert detect_media_type("clip.avi") == "video"
    assert detect_media_type("movie.mkv") == "video"


def test_detect_audio():
    assert detect_media_type("song.mp3") == "audio"
    assert detect_media_type("podcast.wav") == "audio"
    assert detect_media_type("track.flac") == "audio"


def test_detect_text():
    assert detect_media_type("article.txt") == "text"
    assert detect_media_type("notes.md") == "text"
    assert detect_media_type("data.csv") == "text"
    assert detect_media_type("no-extension") == "text"


def test_is_media_file():
    assert is_media_file("video.mp4") is True
    assert is_media_file("song.mp3") is True
    assert is_media_file("document.txt") is False


def test_read_local_file():
    text = read_file("tests/fixtures/neutral_article.txt")
    assert len(text) > 100
    assert "Study Finds Moderate Exercise" in text


def test_read_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        read_file("nonexistent.txt")


def test_read_empty_file_fails():
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='tests/fixtures') as f:
        f.write("")
        f.flush()
        path = f.name
    try:
        with pytest.raises(ValueError, match="empty"):
            read_file(path)
    finally:
        os.unlink(path)
