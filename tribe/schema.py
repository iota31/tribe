"""Unified output schema for all analysis backends."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TextSpan:
    """A span of text within the analyzed content."""

    text: str
    start: int
    end: int


@dataclass
class Technique:
    """A detected propaganda/manipulation technique."""

    name: str
    confidence: float
    description: str
    emotion_target: str
    spans: list[TextSpan] | None = None


@dataclass
class Emotion:
    """A detected emotion in the content."""

    name: str
    confidence: float


@dataclass
class NeuralAnalysis:
    """Brain activation analysis from TRIBE v2."""

    network_scores: dict[str, float]
    manipulation_ratio: float
    dominant_network: str
    dominant_regions: list[str]
    interpretation: str


@dataclass
class ContentAnalysis:
    """Unified output from any analysis backend."""

    # What the content is trying to trigger
    primary_trigger: str
    trigger_confidence: float

    # Manipulation assessment
    manipulation_score: float  # 0.0 - 10.0

    # Detected techniques and emotions
    techniques: list[Technique]
    emotions: list[Emotion]

    # Neural prediction (TRIBE v2 only)
    neural: NeuralAnalysis | None

    # Content metadata
    content_type: str
    content_length: int
    source_url: str | None = None

    # Analysis metadata
    backend: str = "tribe_v2_rust"
    processing_time_ms: int = 0
    model_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        result = asdict(self)
        # Remove None neural field if not present
        if result["neural"] is None:
            del result["neural"]
        # Remove None spans from techniques
        for tech in result["techniques"]:
            if tech["spans"] is None:
                del tech["spans"]
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
