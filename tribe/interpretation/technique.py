"""Technique-to-emotion mapping and manipulation scoring for classifier backend."""

from __future__ import annotations

from tribe.schema import Emotion, Technique

# Map each SemEval propaganda technique to its primary emotional target
TECHNIQUE_EMOTION_MAP: dict[str, str] = {
    "Appeal to Authority": "deference",
    "Appeal to Fear/Prejudice": "fear",
    "Appeal to fear/prejudice": "fear",
    "Bandwagon": "social_pressure",
    "Black-and-White Fallacy": "urgency",
    "Black-and-white Fallacy": "urgency",
    "Causal Oversimplification": "false_clarity",
    "Doubt": "distrust",
    "Exaggeration/Minimisation": "anxiety",
    "Flag-Waving": "tribalism",
    "Flag-waving": "tribalism",
    "Loaded Language": "anger",
    "Name Calling/Labeling": "contempt",
    "Name calling/Labeling": "contempt",
    "Repetition": "familiarity",
    "Slogans": "resonance",
    "Thought-Terminating Cliche": "shutdown",
    "Thought-terminating Cliche": "shutdown",
    "Whataboutism/Red Herring": "confusion",
    # Binary classification model labels
    "propaganda": "manipulation",
    "PROPAGANDA": "manipulation",
}

# Potency weights: how strongly each technique tends to manipulate
# Higher weight = more manipulative per unit of confidence
TECHNIQUE_POTENCY: dict[str, float] = {
    "Appeal to Fear/Prejudice": 1.0,
    "Appeal to fear/prejudice": 1.0,
    "Loaded Language": 0.9,
    "Name Calling/Labeling": 0.85,
    "Name calling/Labeling": 0.85,
    "Flag-Waving": 0.8,
    "Flag-waving": 0.8,
    "Black-and-White Fallacy": 0.75,
    "Black-and-white Fallacy": 0.75,
    "Exaggeration/Minimisation": 0.7,
    "Bandwagon": 0.7,
    "Thought-Terminating Cliche": 0.65,
    "Thought-terminating Cliche": 0.65,
    "Doubt": 0.6,
    "Causal Oversimplification": 0.6,
    "Appeal to Authority": 0.55,
    "Slogans": 0.5,
    "Repetition": 0.45,
    "Whataboutism/Red Herring": 0.65,
    "propaganda": 0.7,
    "PROPAGANDA": 0.7,
}

# Map broad emotion targets to human-readable trigger names
TRIGGER_DISPLAY_NAMES: dict[str, str] = {
    "fear": "Fear",
    "anger": "Anger",
    "contempt": "Contempt",
    "tribalism": "Tribalism",
    "urgency": "Urgency",
    "anxiety": "Anxiety",
    "social_pressure": "Social Pressure",
    "distrust": "Distrust",
    "false_clarity": "False Clarity",
    "confusion": "Confusion",
    "shutdown": "Cognitive Shutdown",
    "familiarity": "False Familiarity",
    "resonance": "Emotional Resonance",
    "deference": "Deference",
    "manipulation": "Manipulation",
    # Emotion classifier labels
    "disgust": "Disgust",
    "joy": "Joy",
    "sadness": "Sadness",
    "surprise": "Surprise",
    "neutral": "Neutral",
}


def compute_manipulation_score(
    techniques: list[Technique],
    emotions: list[Emotion],
) -> float:
    """Compute aggregate manipulation score from detected techniques and emotions.

    Returns a score from 0.0 to 10.0.
    """
    if not techniques:
        # No techniques detected — check if emotions alone signal manipulation
        negative_emotions = sum(
            e.confidence
            for e in emotions
            if e.name.lower() in ("anger", "fear", "disgust", "sadness")
        )
        # Pure emotion without technique = low manipulation score
        return round(min(negative_emotions * 3.0, 3.0), 1)

    # Weighted sum of technique confidence * potency
    raw_score = sum(
        t.confidence * TECHNIQUE_POTENCY.get(t.name, 0.5) for t in techniques
    )

    # Normalize: single high-confidence technique ≈ 5.0,
    # multiple high-confidence techniques push toward 10.0
    normalized = min(raw_score * 4.0, 10.0)

    # Boost if emotion classifier confirms matching emotional valence
    emotion_boost = 0.0
    if emotions:
        top_emotion = emotions[0]
        if top_emotion.name.lower() in ("anger", "fear", "disgust"):
            # Strong negative emotion confirmed — boost score
            emotion_boost = top_emotion.confidence * 1.5

    final_score = min(normalized + emotion_boost, 10.0)
    return round(final_score, 1)


def identify_primary_trigger(
    techniques: list[Technique],
    emotions: list[Emotion],
) -> tuple[str, float]:
    """Identify the primary emotional trigger from techniques and emotions.

    Returns:
        Tuple of (trigger_name, confidence).
    """
    if not techniques and not emotions:
        return ("neutral", 0.0)

    # Count weighted votes for each emotion target
    trigger_votes: dict[str, float] = {}

    for t in techniques:
        target = t.emotion_target
        weight = t.confidence * TECHNIQUE_POTENCY.get(t.name, 0.5)
        trigger_votes[target] = trigger_votes.get(target, 0.0) + weight

    # Factor in direct emotion detection
    for e in emotions:
        emotion_name = e.name.lower()
        if emotion_name != "neutral":
            trigger_votes[emotion_name] = (
                trigger_votes.get(emotion_name, 0.0) + e.confidence * 0.8
            )

    if not trigger_votes:
        return ("neutral", 0.0)

    # Find the dominant trigger
    primary = max(trigger_votes, key=trigger_votes.get)  # type: ignore[arg-type]
    display_name = TRIGGER_DISPLAY_NAMES.get(primary, primary.title())
    confidence = min(trigger_votes[primary] / 1.5, 1.0)

    return (display_name, round(confidence, 3))
