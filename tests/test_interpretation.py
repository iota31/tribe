"""Tests for interpretation layer."""

import numpy as np

from tribe.interpretation.neural import (
    compute_manipulation_ratio,
    compute_network_scores,
    YEO7_NETWORKS,
)
from tribe.interpretation.technique import (
    compute_manipulation_score,
    identify_primary_trigger,
    TECHNIQUE_EMOTION_MAP,
    TRIGGER_DISPLAY_NAMES,
)
from tribe.schema import Emotion, Technique


def test_technique_emotion_map_covers_semeval():
    """All 14 SemEval techniques are in the emotion map."""
    expected = [
        "Appeal to Authority",
        "Appeal to Fear/Prejudice",
        "Bandwagon",
        "Black-and-White Fallacy",
        "Causal Oversimplification",
        "Doubt",
        "Exaggeration/Minimisation",
        "Flag-Waving",
        "Loaded Language",
        "Name Calling/Labeling",
        "Repetition",
        "Slogans",
        "Thought-Terminating Cliche",
        "Whataboutism/Red Herring",
    ]
    for technique in expected:
        assert technique in TECHNIQUE_EMOTION_MAP, f"Missing: {technique}"


def test_compute_manipulation_score_no_techniques():
    """No techniques detected gives low manipulation score."""
    emotions = [Emotion(name="anger", confidence=0.5)]
    score = compute_manipulation_score([], emotions)
    assert 0 < score <= 3.0


def test_compute_manipulation_score_with_techniques():
    """High-confidence fear appeal gives high manipulation score."""
    techniques = [
        Technique(
            name="Appeal to Fear/Prejudice",
            confidence=0.9,
            description="Fear",
            emotion_target="fear",
        ),
        Technique(
            name="Loaded Language",
            confidence=0.8,
            description="Emotion",
            emotion_target="anger",
        ),
    ]
    emotions = [Emotion(name="anger", confidence=0.7)]
    score = compute_manipulation_score(techniques, emotions)
    assert score >= 6.0


def test_identify_primary_trigger_fear():
    """Fear appeal identifies Fear as primary trigger."""
    techniques = [
        Technique(
            name="Appeal to Fear/Prejudice",
            confidence=0.9,
            description="Fear",
            emotion_target="fear",
        ),
    ]
    trigger, conf = identify_primary_trigger(techniques, [])
    assert trigger == "Fear"
    assert conf > 0.5


def test_identify_primary_trigger_anger():
    """Loaded language identifies Anger as primary trigger."""
    techniques = [
        Technique(
            name="Loaded Language",
            confidence=0.85,
            description="Anger",
            emotion_target="anger",
        ),
    ]
    trigger, conf = identify_primary_trigger(techniques, [])
    assert trigger == "Anger"


def test_identify_primary_trigger_no_input():
    """Empty input returns neutral."""
    trigger, conf = identify_primary_trigger([], [])
    assert trigger == "neutral"


def test_manipulation_ratio_calculation():
    """High emotional / low rational = high ratio."""
    network_scores = {
        "Salience": 0.8,
        "Default_Mode": 0.7,
        "Limbic": 0.6,
        "Executive_Control": 0.2,
        "Dorsal_Attention": 0.3,
        "Visual": 0.5,
        "Somatomotor": 0.4,
    }
    ratio = compute_manipulation_ratio(network_scores)
    # (0.8 + 0.7 + 0.6) / (0.2 + 0.3) = 2.1 / 0.5 = 4.2
    assert 4.0 <= ratio <= 4.5


def test_manipulation_ratio_balanced():
    """Balanced networks give ratio of 1.5.

    emotional_sum = Salience(0.5) + Limbic(0.5) + Default_Mode(0.5) = 1.5
    rational_sum = Executive_Control(0.5) + Dorsal_Attention(0.5) = 1.0
    ratio = 1.5 / 1.0 = 1.5
    """
    network_scores = {net: 0.5 for net in YEO7_NETWORKS.values()}
    ratio = compute_manipulation_ratio(network_scores)
    assert 1.3 <= ratio <= 1.7


def test_network_scores():
    """Network scores computed from activation vector."""
    # Mock: 10242 LH + 10242 RH = 20484 total
    # Each hemisphere chunk: 1024 vertices per network
    # LH: 4=Salience(1024), 6=ExecControl(1024), 7=DefaultMode(1024), 1=Visual(1024), 0=rest(6146)
    # RH: same pattern
    lh = [4]*1024 + [6]*1024 + [7]*1024 + [1]*1024 + [0]*6146
    rh = [4]*1024 + [6]*1024 + [7]*1024 + [1]*1024 + [0]*6146
    network_ids = np.array(lh + rh, dtype=np.int32)

    activation = np.zeros(20484)
    activation[:1024] = 0.8   # Salience (LH)
    activation[1024:2048] = 0.2  # ExecControl (LH)
    activation[2048:3072] = 0.6  # DefaultMode (LH)
    # RH half
    activation[10242:11266] = 0.8   # Salience (RH)
    activation[11266:12290] = 0.2  # ExecControl (RH)
    activation[12290:13314] = 0.6  # DefaultMode (RH)

    scores = compute_network_scores(activation, network_ids)
    assert 0.78 <= scores["Salience"] <= 0.82
    assert 0.18 <= scores["Executive_Control"] <= 0.22
    assert 0.58 <= scores["Default_Mode"] <= 0.62


def test_trigger_display_names():
    """All emotion targets have human-readable display names."""
    for key in TECHNIQUE_EMOTION_MAP.values():
        assert key in TRIGGER_DISPLAY_NAMES or key in (
            "manipulation", "deference"
        )
