"""Tests for output renderers."""

import json

from tribe.output.narrative import render_narrative, render_quiet
from tribe.output.json_output import render_json
from tribe.schema import ContentAnalysis, Emotion, NeuralAnalysis, Technique


def _make_analysis(manipulation_score=7.2, trigger="Fear", neural=None):
    return ContentAnalysis(
        primary_trigger=trigger,
        trigger_confidence=0.82,
        manipulation_score=manipulation_score,
        techniques=[
            Technique(
                name="Appeal to Fear/Prejudice",
                confidence=0.87,
                description="Exploits fear to push a conclusion",
                emotion_target="fear",
            ),
            Technique(
                name="Loaded Language",
                confidence=0.74,
                description="Uses emotionally charged words",
                emotion_target="anger",
            ),
        ],
        emotions=[
            Emotion(name="anger", confidence=0.65),
            Emotion(name="fear", confidence=0.45),
        ],
        neural=neural,
        content_type="text",
        content_length=500,
        backend="classifier",
        processing_time_ms=120,
    )


def test_narrative_rendering_high_manipulation():
    """High manipulation score renders with warning icon."""
    analysis = _make_analysis(manipulation_score=7.2)
    output = render_narrative(analysis)
    assert "ENGINEERED" in output or "engineered" in output
    assert "7.2" in output
    assert "Fear" in output


def test_narrative_rendering_low_manipulation():
    """Low manipulation score renders with info icon."""
    analysis = _make_analysis(manipulation_score=2.0)
    output = render_narrative(analysis)
    assert "Tends" in output or "Low" in output
    assert "2.0" in output


def test_narrative_verbose_shows_techniques():
    """Verbose mode includes per-technique breakdown."""
    analysis = _make_analysis()
    output = render_narrative(analysis, verbose=True)
    assert "Appeal to Fear" in output
    assert "confidence" in output.lower()


def test_narrative_verbose_shows_neural():
    """Verbose mode includes neural network breakdown when available."""
    neural = NeuralAnalysis(
        network_scores={
            "Salience": 0.73,
            "Default_Mode": 0.61,
            "Executive_Control": 0.21,
            "Limbic": 0.52,
            "Dorsal_Attention": 0.29,
            "Visual": 0.1,
            "Somatomotor": 0.1,
        },
        manipulation_ratio=3.4,
        dominant_network="Salience",
        dominant_regions=["insula", "rostralanteriorcingulate"],
        interpretation="Strong emotional salience with suppressed rational processing.",
    )
    analysis = _make_analysis(neural=neural)
    output = render_narrative(analysis, verbose=True)
    assert "Salience" in output
    assert "3.4" in output
    # Network labels use spaces, not underscores
    assert "Executive Control" in output


def test_quiet_rendering():
    """Quiet mode outputs single line."""
    analysis = _make_analysis()
    output = render_quiet(analysis)
    lines = output.strip().split("\n")
    assert len(lines) == 1
    assert "7.2" in output
    assert "Fear" in output


def test_json_rendering():
    """JSON output is valid and contains all fields."""
    analysis = _make_analysis()
    output = render_json(analysis)
    data = json.loads(output)
    assert data["primary_trigger"] == "Fear"
    assert data["manipulation_score"] == 7.2
    assert len(data["techniques"]) == 2
    assert data["backend"] == "classifier"
