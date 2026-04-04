"""Tests for the unified schema."""

import json

from tribe.schema import ContentAnalysis, Emotion, NeuralAnalysis, Technique, TextSpan


def test_content_analysis_to_dict():
    """ContentAnalysis serializes to dict correctly."""
    analysis = ContentAnalysis(
        primary_trigger="Fear",
        trigger_confidence=0.82,
        manipulation_score=7.2,
        techniques=[
            Technique(
                name="Appeal to Fear/Prejudice",
                confidence=0.87,
                description="Exploits fear without evidence",
                emotion_target="fear",
            )
        ],
        emotions=[Emotion(name="anger", confidence=0.65)],
        neural=None,
        content_type="text",
        content_length=500,
        source_url="https://example.com",
        backend="classifier",
        processing_time_ms=120,
    )

    result = analysis.to_dict()

    assert result["primary_trigger"] == "Fear"
    assert result["manipulation_score"] == 7.2
    assert len(result["techniques"]) == 1
    assert result["techniques"][0]["name"] == "Appeal to Fear/Prejudice"
    assert result["emotions"][0]["name"] == "anger"
    # neural is removed from dict when None (compact serialization)
    assert "neural" not in result
    assert result["source_url"] == "https://example.com"


def test_content_analysis_to_json():
    """ContentAnalysis serializes to JSON correctly."""
    analysis = ContentAnalysis(
        primary_trigger="Anger",
        trigger_confidence=0.5,
        manipulation_score=3.0,
        techniques=[],
        emotions=[],
        neural=NeuralAnalysis(
            network_scores={"Salience": 0.73, "Executive_Control": 0.21},
            manipulation_ratio=3.4,
            dominant_network="Salience",
            dominant_regions=["insula"],
            interpretation="Strong emotional salience",
        ),
        content_type="text",
        content_length=200,
        backend="tribe_v2",
        processing_time_ms=4200,
    )

    result = json.loads(analysis.to_json())

    assert result["primary_trigger"] == "Anger"
    assert result["manipulation_score"] == 3.0
    assert result["neural"]["manipulation_ratio"] == 3.4
    assert result["backend"] == "tribe_v2"


def test_text_span():
    """TextSpan is properly structured."""
    span = TextSpan(text="government FAILED", start=0, end=17)
    assert span.text == "government FAILED"
    assert span.start == 0
    assert span.end == 17
