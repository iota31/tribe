"""TRIBE v2 backend — neural brain activation prediction.

Requires GPU with ≥10GB VRAM and the tribev2 package.
Uses Meta's TRIBE v2 model to predict fMRI brain activation patterns,
then maps those activations to functional brain networks using the
Yeo 2011 7-network parcellation.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from tribe.backends.base import AnalysisBackend
from tribe.backends.router import HardwareInfo
from tribe.interpretation.neural import interpret_activation
from tribe.schema import ContentAnalysis


# Neural manipulation ratio to manipulation score (0-10) mapping
def _ratio_to_score(ratio: float) -> float:
    """Convert neural manipulation ratio to 0-10 score."""
    if ratio <= 1.0:
        return round(ratio * 2.0, 1)  # 0-2
    if ratio <= 2.0:
        return round(2.0 + (ratio - 1.0) * 3.0, 1)  # 2-5
    if ratio <= 3.0:
        return round(5.0 + (ratio - 2.0) * 2.5, 1)  # 5-7.5
    return round(min(7.5 + (ratio - 3.0) * 1.25, 10.0), 1)  # 7.5-10


# Map dominant network to primary emotion trigger
NETWORK_TRIGGER_MAP: dict[str, str] = {
    "Salience": "Fear",
    "Default_Mode": "Self-Referential Anxiety",
    "Limbic": "Outrage",
    "Executive_Control": "Analytical Engagement",
    "Dorsal_Attention": "Focused Attention",
}


class TribeV2Backend(AnalysisBackend):
    """TRIBE v2 backend — predicts brain activation patterns from content.

    Uses LLaMA 3.2 (text), V-JEPA2 (video), Wav2Vec-BERT (audio)
    feeding a fusion transformer that predicts fMRI at 20,484 cortical vertices.
    """

    def __init__(self, hardware: HardwareInfo) -> None:
        self._hardware = hardware
        self._model = None
        self._network_ids = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "tribe_v2"

    def _ensure_loaded(self) -> None:
        """Lazy-load TRIBE v2 model and Yeo atlas."""
        if self._loaded:
            return

        try:
            from tribev2 import TribeModel
        except ImportError:
            raise ImportError(
                "TRIBE v2 package not installed. "
                "Install with: pip install tribev2\n"
                "Build the Rust binary instead: see README for instructions."
            )

        from tribe.interpretation.neural import load_yeo7_network_ids

        self._model = TribeModel.from_pretrained("facebook/tribev2")
        self._network_ids = load_yeo7_network_ids()
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def analyze_text(self, text: str) -> ContentAnalysis:
        """Analyze text by predicting brain activation patterns."""
        self._ensure_loaded()
        start_time = time.monotonic()

        # Write text to temp file for TRIBE v2 API
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            text_path = f.name

        try:
            # Run TRIBE v2 prediction
            df = self._model.get_events_dataframe(text_path=text_path)
            preds, segments = self._model.predict(events=df)

            # preds shape: (n_timesteps, ~20484)
            import numpy as np

            activation = np.array(preds)

            # Interpret neural activation
            neural = interpret_activation(activation, self._network_ids)

            # Derive manipulation score and trigger from neural analysis
            manipulation_score = _ratio_to_score(neural.manipulation_ratio)
            primary_trigger = NETWORK_TRIGGER_MAP.get(neural.dominant_network, "Manipulation")
            emotional_sum = sum(
                neural.network_scores.get(net, 0.0)
                for net in ("Salience", "Default_Mode", "Limbic")
                if neural.network_scores.get(net, 0.0) > 0
            )
            total_positive = sum(s for s in neural.network_scores.values() if s > 0)
            trigger_confidence = min(
                (emotional_sum / total_positive) if total_positive > 0 else 0.0, 1.0
            )

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            return ContentAnalysis(
                primary_trigger=primary_trigger,
                trigger_confidence=round(trigger_confidence, 3),
                manipulation_score=manipulation_score,
                techniques=[],  # TRIBE v2 doesn't detect specific techniques
                emotions=[],  # Neural networks, not emotion labels
                neural=neural,
                content_type="text",
                content_length=len(text.split()),
                backend=self.name,
                processing_time_ms=elapsed_ms,
                model_versions={
                    "tribe_v2": "facebook/tribev2",
                    "atlas": "Yeo2011_7Networks",
                },
            )
        finally:
            Path(text_path).unlink(missing_ok=True)

    def analyze_media(self, path: str, media_type: str) -> ContentAnalysis:
        """Analyze video or audio by predicting brain activation patterns."""
        self._ensure_loaded()
        start_time = time.monotonic()

        # Build events dataframe based on media type
        if media_type == "video":
            df = self._model.get_events_dataframe(video_path=path)
        elif media_type == "audio":
            df = self._model.get_events_dataframe(audio_path=path)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")

        preds, segments = self._model.predict(events=df)

        import numpy as np

        activation = np.array(preds)
        neural = interpret_activation(activation, self._network_ids)

        manipulation_score = _ratio_to_score(neural.manipulation_ratio)
        primary_trigger = NETWORK_TRIGGER_MAP.get(neural.dominant_network, "Manipulation")
        emotional_sum = sum(
            neural.network_scores.get(net, 0.0)
            for net in ("Salience", "Default_Mode", "Limbic")
            if neural.network_scores.get(net, 0.0) > 0
        )
        total_positive = sum(s for s in neural.network_scores.values() if s > 0)
        trigger_confidence = min(
            (emotional_sum / total_positive) if total_positive > 0 else 0.0, 1.0
        )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return ContentAnalysis(
            primary_trigger=primary_trigger,
            trigger_confidence=round(trigger_confidence, 3),
            manipulation_score=manipulation_score,
            techniques=[],
            emotions=[],
            neural=neural,
            content_type=media_type,
            content_length=0,
            backend=self.name,
            processing_time_ms=elapsed_ms,
            model_versions={
                "tribe_v2": "facebook/tribev2",
                "atlas": "Yeo2011_7Networks",
            },
        )
