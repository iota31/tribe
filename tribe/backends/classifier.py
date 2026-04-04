"""Classifier backend — lightweight propaganda and emotion detection."""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, PretrainedConfig

from tribe.backends.base import AnalysisBackend
from tribe.backends.qcri_architecture import (
    BertForTokenAndSequenceJointClassification,
    QCRI_LABEL_MAP,
    QCRI_MODEL,
    QCRI_TOKEN_TAGS,
)
from tribe.interpretation.technique import (
    TECHNIQUE_EMOTION_MAP,
    compute_manipulation_score,
    identify_primary_trigger,
)
from tribe.schema import ContentAnalysis, Emotion, Technique


# Emotion model: 7-class emotion detection (DistilRoBERTa, 82M params)
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Max tokens per chunk for transformer models
MAX_CHUNK_LENGTH = 512


class _QCRITechniqueWrapper:
    """Wrapper around the QCRI BertForTokenAndSequenceJointClassification model.

    The QCRI model is a token-level classifier that requires 'bert-base-cased'
    tokenizer (not included in the model repo). This wrapper handles loading,
    inference, and result aggregation in a pipeline-compatible interface.
    """

    def __init__(self, model_name: str) -> None:
        self._model = None
        self._tokenizer = None
        self._model_name = model_name

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the QCRI model and tokenizer."""
        if self._model is not None:
            return

        from huggingface_hub import snapshot_download

        # Resolve the cached model directory
        model_path = snapshot_download(self._model_name)

        # Load tokenizer from the QCRI model's cached files
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model using custom architecture
        config = PretrainedConfig.from_pretrained(model_path)
        self._model = BertForTokenAndSequenceJointClassification(config)
        import torch

        state_dict = torch.load(
            model_path + "/pytorch_model.bin",
            map_location="cpu",
            weights_only=False,
        )
        self._model.load_state_dict(state_dict, strict=False)
        self._model.eval()

    def __call__(self, text: str) -> list[dict]:
        """Run technique detection on a single text.

        Returns a list of dicts with 'label' and 'score' for each technique
        (compatible with the text-classification pipeline output format).
        """
        self._ensure_model_loaded()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CHUNK_LENGTH,
            padding=True,
        )
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = self._model(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask,
            )

        # Aggregate raw logits per technique: take the MAX per technique
        # across all non-padding tokens. This captures the strongest evidence
        # for each technique anywhere in the text.
        seq_len = attention_mask[0].sum().item()
        token_logits = outputs.token_logits[0, :seq_len]  # (seq_len, 20)

        technique_logits: dict[int, list[float]] = defaultdict(list)
        for pos in range(seq_len):
            logits_for_pos = token_logits[pos].tolist()
            for idx, logit in enumerate(logits_for_pos):
                technique_logits[idx].append(logit)

        # Aggregate: max logit per technique across all positions
        aggregated = {}
        for idx, logit_list in technique_logits.items():
            raw_tag = QCRI_TOKEN_TAGS[idx]
            if raw_tag in ("<PAD>", "O"):
                continue
            # Map to standard technique name
            mapped_tag = QCRI_LABEL_MAP.get(raw_tag, raw_tag)
            # Keep highest logit for this mapped technique (max-pool)
            if mapped_tag not in aggregated or max(logit_list) > aggregated[mapped_tag]:
                aggregated[mapped_tag] = max(logit_list)

        # Apply softmax over the aggregated logits to get normalized probabilities
        if not aggregated:
            return []

        logit_values = list(aggregated.values())
        logit_tensor = torch.tensor(logit_values)
        probs = F.softmax(logit_tensor, dim=0)

        result = []
        for tag, logit in aggregated.items():
            idx = list(aggregated.keys()).index(tag)
            result.append({"label": tag, "score": round(probs[idx].item(), 4)})

        return result


class ClassifierBackend(AnalysisBackend):
    """Lightweight classifier backend using QCRI BERT + DistilRoBERTa.

    Runs on CPU, ~150MB total model size, <200ms inference.
    """

    def __init__(self) -> None:
        self._technique_wrapper = _QCRITechniqueWrapper(QCRI_MODEL)
        self._emotion_pipe = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "classifier"

    def _ensure_loaded(self) -> None:
        """Lazy-load models on first use."""
        if self._loaded:
            return

        # Pre-load the QCRI technique model (eagerly load, not lazy)
        # to get any startup errors early
        self._technique_wrapper._ensure_model_loaded()

        from transformers import pipeline

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

    def _run_technique_detection(self, text: str) -> list[dict]:
        """Run QCRI technique detection on text (all 18 techniques, multi-label)."""
        chunks = self._chunk_text(text)

        # Run all chunks through the technique wrapper
        all_scores: dict[str, list[float]] = defaultdict(list)
        for chunk in chunks:
            results = self._technique_wrapper(chunk)
            for item in results:
                all_scores[item["label"]].append(item["score"])

        # Average scores across chunks
        averaged = []
        for label, scores in all_scores.items():
            averaged.append({"label": label, "score": sum(scores) / len(scores)})

        return averaged

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
            technique_future = executor.submit(self._run_technique_detection, text)
            emotion_future = executor.submit(self._run_emotion_detection, text)

            technique_results = technique_future.result()
            emotion_results = emotion_future.result()

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Build technique list from QCRI technique results
        techniques = self._build_techniques(technique_results)

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
                "technique": QCRI_MODEL,
                "emotion": EMOTION_MODEL,
            },
        )

    def _build_techniques(self, raw_results: list[dict]) -> list[Technique]:
        """Convert raw classifier output to Technique objects."""
        techniques = []

        for result in raw_results:
            label = result.get("label", "")
            score = result.get("score", 0.0)

            # Skip low-confidence labels
            if score < 0.01:
                continue

            emotion_target = TECHNIQUE_EMOTION_MAP.get(label, "manipulation")
            description = _technique_description(label)

            techniques.append(
                Technique(
                    name=label,
                    confidence=round(score, 3),
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
        "Glittering Generalities": "Uses vague, positive language to win approval without substance",
        "Intentional Vagueness": "Uses ambiguous language to obscure meaning and avoid accountability",
        "Loaded Language": "Uses emotionally charged words to influence perception",
        "Misrepresentation of Someone's Position (Or Quoting)": "Distorts or takes statements out of context",
        "Name Calling/Labeling": "Attaches negative labels to dismiss without argument",
        "Name calling/Labeling": "Attaches negative labels to dismiss without argument",
        "Repetition": "Repeats claims to make them feel true through familiarity",
        "Slogans": "Uses catchy phrases to replace critical thinking",
        "Thought-Terminating Cliche": "Uses stock phrases to shut down analysis",
        "Thought-terminating Cliche": "Uses stock phrases to shut down analysis",
        "Transfer": "Associates a subject with a positive or negative symbol to transfer sentiment",
        "Whataboutism/Red Herring": "Deflects from the issue by raising unrelated topics",
        "propaganda": "Content uses propaganda techniques to manipulate perception",
    }
    return descriptions.get(label, f"{label.lower()} to influence perception")
