"""Base class for analysis backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tribe.schema import ContentAnalysis


class AnalysisBackend(ABC):
    """Abstract base for content analysis backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier for output metadata."""

    @abstractmethod
    def analyze_text(self, text: str) -> ContentAnalysis:
        """Analyze text content for manipulation.

        Args:
            text: The text to analyze.

        Returns:
            A ContentAnalysis with detected techniques, emotions, and scores.
        """

    @abstractmethod
    def analyze_media(self, path: str, media_type: str) -> ContentAnalysis:
        """Analyze a media file for manipulation.

        Args:
            path: Path to the media file.
            media_type: "video" or "audio".

        Returns:
            A ContentAnalysis with detected techniques, emotions, and scores.
        """

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the backend models are loaded and ready."""
