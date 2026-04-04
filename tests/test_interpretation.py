"""Tests for interpretation layer."""

import numpy as np

from tribe.interpretation.neural import (
    compute_manipulation_ratio,
    compute_network_scores,
    load_yeo7_network_ids,
    YEO7_NETWORKS,
)
from pathlib import Path
from tribe.interpretation.technique import (
    compute_manipulation_score,
    identify_primary_trigger,
    TECHNIQUE_EMOTION_MAP,
    TECHNIQUE_POTENCY,
    TRIGGER_DISPLAY_NAMES,
)
from tribe.schema import Emotion, Technique
from tribe.backends.classifier import _technique_description


# The 18 QCRI technique labels
QCRI_TECHNIQUE_LABELS = [
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
    "Glittering Generalities",
    "Intentional Vagueness",
    "Misrepresentation of Someone's Position (Or Quoting)",
    "Transfer",
]


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


def test_load_yeo7_network_ids():
    """load_yeo7_network_ids returns correct shape and valid network IDs."""
    atlas_dir = Path(__file__).parent.parent / "tribe" / "interpretation" / "atlas"
    ids = load_yeo7_network_ids(atlas_dir)

    # ISC-4: shape (20484,)
    assert ids.shape == (20484,), f"Expected (20484,), got {ids.shape}"

    # ISC-5: values 0-7 only
    unique = sorted(set(ids))
    assert unique == [0, 1, 2, 3, 4, 5, 6, 7], f"Unexpected values: {unique}"

    # Each hemisphere has 10242 vertices
    lh = ids[:10242]
    rh = ids[10242:]
    assert lh.shape == (10242,)
    assert rh.shape == (10242,)

    # Both hemispheres have all 7 networks + medial wall (0)
    assert len(sorted(set(lh))) == 8
    assert len(sorted(set(rh))) == 8


def test_qcri_technique_labels_covered():
    """All 18 QCRI technique labels are in maps and have descriptions."""
    for label in QCRI_TECHNIQUE_LABELS:
        assert label in TECHNIQUE_EMOTION_MAP, f"Missing in EMOTION_MAP: {label}"
        assert label in TECHNIQUE_POTENCY, f"Missing in POTENCY: {label}"
        desc = _technique_description(label)
        assert desc is not None
        assert len(desc) > 0
        assert desc != f"Uses {label.lower()} to influence perception", (
            f"No description provided for: {label}"
        )
