"""Classifier backend — lightweight propaganda and emotion detection."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from tribe.backends.base import AnalysisBackend
from tribe.interpretation.technique import (
    TECHNIQUE_EMOTION_MAP,
    compute_manipulation_score,
    identify_primary_trigger,
)
from tribe.schema import ContentAnalysis, Emotion, Technique


# Propaganda model: IDA-SERICS/PropagandaDetection (DistilBERT, 67M params)
PROPAGANDA_MODEL = "IDA-SERICS/PropagandaDetection"

# Emotion model: 7-class emotion detection (DistilRoBERTa, 82M params)
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Max tokens per chunk for transformer models
MAX_CHUNK_LENGTH = 512


class ClassifierBackend(AnalysisBackend):
    """Lightweight classifier backend using DistilBERT + DistilRoBERTa.

    Runs on CPU, ~150MB total model size, <200ms inference.
    """

    def __init__(self) -> None:
        self._propaganda_pipe = None
        self._emotion_pipe = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "classifier"

    def _ensure_loaded(self) -> None:
        """Lazy-load models on first use."""
        if self._loaded:
            return

        from transformers import pipeline

        self._propaganda_pipe = pipeline(
            "text-classification",
            model=PROPAGANDA_MODEL,
            truncation=True,
            max_length=MAX_CHUNK_LENGTH,
        )
        self._emotion_pipe = pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            truncation=True,
            max_length=MAX_CHUNK_LENGTH,
            top_k=None,  # return all emotion scores
        )
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def _chunk_text(self, text: str, max_words: int = 400) -> list[str]:
        """Split text into chunks for processing.

        Splits by paragraphs first, then by sentence count if still too long.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text]

        chunks = []
        current_chunk = []
        current_words = 0

        for para in paragraphs:
            para_words = len(para.split())
            if current_words + para_words > max_words and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_words = para_words
            else:
                current_chunk.append(para)
                current_words += para_words

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [text]

    def _run_propaganda_detection(self, text: str) -> list[dict]:
        """Run propaganda technique detection on text."""
        chunks = self._chunk_text(text)
        all_results = []

        for chunk in chunks:
            results = self._propaganda_pipe(chunk)
            if isinstance(results, list):
                if results and isinstance(results[0], dict):
                    all_results.extend(results)
                elif results and isinstance(results[0], list):
                    for r in results:
                        all_results.extend(r)

        return all_results

    def _run_emotion_detection(self, text: str) -> list[dict]:
        """Run emotion classification on text."""
        chunks = self._chunk_text(text)
        all_scores: dict[str, list[float]] = {}

        for chunk in chunks:
            results = self._emotion_pipe(chunk)
            # results is a list of lists of dicts for top_k=None
            if results and isinstance(results[0], list):
                for item in results[0]:
                    label = item["label"]
                    score = item["score"]
                    all_scores.setdefault(label, []).append(score)
            elif results and isinstance(results[0], dict):
                for item in results:
                    label = item["label"]
                    score = item["score"]
                    all_scores.setdefault(label, []).append(score)

        # Average scores across chunks
        averaged = []
        for label, scores in all_scores.items():
            averaged.append({"label": label, "score": sum(scores) / len(scores)})

        return sorted(averaged, key=lambda x: x["score"], reverse=True)

    def analyze_text(self, text: str) -> ContentAnalysis:
        """Analyze text for manipulation techniques and emotional triggers."""
        self._ensure_loaded()
        start_time = time.monotonic()

        # Run both classifiers in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            propaganda_future = executor.submit(self._run_propaganda_detection, text)
            emotion_future = executor.submit(self._run_emotion_detection, text)

            propaganda_results = propaganda_future.result()
            emotion_results = emotion_future.result()

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Build technique list from propaganda results
        techniques = self._build_techniques(propaganda_results)

        # Build emotion list
        emotions = [
            Emotion(name=e["label"], confidence=round(e["score"], 3))
            for e in emotion_results
            if e["score"] > 0.05  # filter noise
        ]

        # Compute aggregate scores
        manipulation_score = compute_manipulation_score(techniques, emotions)
        primary_trigger, trigger_confidence = identify_primary_trigger(
            techniques, emotions
        )

        word_count = len(text.split())

        return ContentAnalysis(
            primary_trigger=primary_trigger,
            trigger_confidence=trigger_confidence,
            manipulation_score=manipulation_score,
            techniques=techniques,
            emotions=emotions,
            neural=None,
            content_type="text",
            content_length=word_count,
            backend=self.name,
            processing_time_ms=elapsed_ms,
            model_versions={
                "propaganda": PROPAGANDA_MODEL,
                "emotion": EMOTION_MODEL,
            },
        )

    def _build_techniques(self, raw_results: list[dict]) -> list[Technique]:
        """Convert raw classifier output to Technique objects."""
        # The propaganda model may output labels like "propaganda" / "not propaganda"
        # or specific technique names depending on the model variant.
        # We handle both cases.
        techniques = []
        seen_labels = {}

        for result in raw_results:
            label = result.get("label", "")
            score = result.get("score", 0.0)

            # Skip low-confidence and non-propaganda labels
            if score < 0.3:
                continue
            if label.lower() in ("not propaganda", "not_propaganda", "non-propaganda"):
                continue

            # Aggregate scores for same label across chunks
            if label in seen_labels:
                seen_labels[label].append(score)
            else:
                seen_labels[label] = [score]

        for label, scores in seen_labels.items():
            avg_score = sum(scores) / len(scores)
            emotion_target = TECHNIQUE_EMOTION_MAP.get(label, "manipulation")
            description = _technique_description(label)

            techniques.append(
                Technique(
                    name=label,
                    confidence=round(avg_score, 3),
                    description=description,
                    emotion_target=emotion_target,
                )
            )

        # Sort by confidence descending
        techniques.sort(key=lambda t: t.confidence, reverse=True)
        return techniques

    def analyze_media(self, path: str, media_type: str) -> ContentAnalysis:
        """Classifier backend does not support media analysis directly.

        Returns a placeholder indicating TRIBE v2 is needed for media.
        """
        return ContentAnalysis(
            primary_trigger="unknown",
            trigger_confidence=0.0,
            manipulation_score=0.0,
            techniques=[],
            emotions=[],
            neural=None,
            content_type=media_type,
            content_length=0,
            backend=self.name,
            processing_time_ms=0,
            model_versions={},
        )


def _technique_description(label: str) -> str:
    """Get a human-readable description for a propaganda technique."""
    descriptions = {
        "Appeal to Authority": "Uses authority figures to bypass critical thinking",
        "Appeal to Fear/Prejudice": "Exploits fear to push a conclusion without evidence",
        "Appeal to fear/prejudice": "Exploits fear to push a conclusion without evidence",
        "Bandwagon": "Implies everyone agrees, pressuring conformity",
        "Black-and-White Fallacy": "Presents only two options when more exist",
        "Black-and-white Fallacy": "Presents only two options when more exist",
        "Causal Oversimplification": "Reduces complex causes to a single simple one",
        "Doubt": "Questions credibility to undermine without evidence",
        "Exaggeration/Minimisation": "Inflates or deflates facts to distort reality",
        "Flag-Waving": "Appeals to patriotism or group identity over reason",
        "Flag-waving": "Appeals to patriotism or group identity over reason",
        "Loaded Language": "Uses emotionally charged words to influence perception",
        "Name Calling/Labeling": "Attaches negative labels to dismiss without argument",
        "Name calling/Labeling": "Attaches negative labels to dismiss without argument",
        "Repetition": "Repeats claims to make them feel true through familiarity",
        "Slogans": "Uses catchy phrases to replace critical thinking",
        "Thought-Terminating Cliche": "Uses stock phrases to shut down analysis",
        "Thought-terminating Cliche": "Uses stock phrases to shut down analysis",
        "Whataboutism/Red Herring": "Deflects from the issue by raising unrelated topics",
        "propaganda": "Content uses propaganda techniques to manipulate perception",
    }
    return descriptions.get(label, f"Uses {label.lower()} to influence perception")
