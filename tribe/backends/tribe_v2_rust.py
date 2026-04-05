"""TRIBE v2 backend via tribev2-rs Rust binary.

Uses the publicly-accessible eugenehp/tribev2 fork on HuggingFace Hub
(identical weights as facebook/tribev2, no license approval needed) via the
tribev2-rs Rust inference engine, Metal GPU-accelerated on Apple Silicon.

Architecture:
  Text Input
      │
      ▼
  tribev2-infer (Rust binary)
    ├─ LLaMA GGUF → text features (Metal GPU)
    ├─ Fusion transformer → fMRI predictions (Metal GPU)
    └─ Output: float32 binary (n_timesteps × 20484 vertices)
      │
      ▼
  Python: read binary → Yeo 7-network → ContentAnalysis
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from tribe.backends.base import AnalysisBackend
from tribe.backends.router import HardwareInfo
from tribe.interpretation.neural import interpret_activation, load_yeo7_network_ids
from tribe.schema import ContentAnalysis

logger = logging.getLogger(__name__)

# Binary name for the Rust CLI
RUST_BINARY_NAME = "tribev2-infer"

# Layer positions for LLaMA 3.2 3B text feature extraction
# Must be "0.5,1.0" (2 groups × 3072 = 6144 dims) to match
# the fusion model's expected input dimensionality.
LAYER_POSITIONS = "0.5,1.0"

# Default n_timesteps for prediction
DEFAULT_N_TIMESTEPS = 100


def _find_rust_binary() -> Path | None:
    """Find tribev2-infer binary in common locations."""
    candidates = [
        Path("/tmp/tribev2-rs/target/release/tribev2-infer"),
        Path.home() / ".local/bin/tribev2-infer",
        Path.home() / ".cargo/bin/tribev2-infer",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    # Also check PATH
    result = shutil.which(RUST_BINARY_NAME)
    if result:
        return Path(result)
    return None


def _find_eugenehp_model_files() -> dict[str, Path] | None:
    """Locate eugenehp/tribev2 model files in HF cache."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return None

    try:
        cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        # Find the snapshot dir
        eugene_dir = None
        for d in cache_base.iterdir():
            if d.name == "models--eugenehp--tribev2" and d.is_dir():
                eugene_dir = d
                break
        if eugene_dir is None:
            return None

        # Find latest snapshot
        snapshots_dir = eugene_dir / "snapshots"
        if not snapshots_dir.is_dir():
            return None
        snapshots = sorted(snapshots_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
        if not snapshots:
            return None
        snap = snapshots[0]

        config = snap / "config.yaml"
        weights = snap / "model.safetensors"
        build_args = snap / "build_args.json"

        if config.exists() and weights.exists() and build_args.exists():
            return {
                "config": config,
                "weights": weights,
                "build_args": build_args,
            }
    except Exception:
        pass
    return None


def _find_llama_gguf() -> Path | None:
    """Find a LLaMA 3.2 3B GGUF file in common locations."""
    candidates = [
        # Ollama blob (already downloaded)
        Path.home()
        / ".ollama/models/blobs"
        / "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _trigger_from_persuasion(persuasion_scores: dict[str, float]) -> str:
    """Determine the primary trigger from persuasion region scores."""
    if not persuasion_scores:
        return "Unknown"

    vmPFC = abs(persuasion_scores.get("vmPFC", 0.0))
    dlPFC = abs(persuasion_scores.get("dlPFC", 0.0))
    insula = abs(persuasion_scores.get("insula", 0.0))
    temporal = abs(persuasion_scores.get("temporal_pole", 0.0))
    precuneus = abs(persuasion_scores.get("precuneus", 0.0))

    max_val = max(vmPFC, dlPFC, insula, temporal, precuneus, 0.001)

    # Determine the dominant persuasion mechanism
    if vmPFC / max_val > 0.6 and dlPFC / max_val < 0.4:
        return "Value Manipulation"
    if insula / max_val > 0.6:
        return "Emotional Arousal"
    if temporal / max_val > 0.6:
        return "Social Pressure"
    if precuneus / max_val > 0.6:
        return "Self-Relevance"
    if dlPFC / max_val > 0.6:
        return "Analytical Engagement"
    return "Persuasion"


class TribeV2RustBackend(AnalysisBackend):
    """TRIBE v2 via tribev2-rs Rust binary (Metal GPU-accelerated).

    Uses the publicly-accessible ``eugenehp/tribev2`` fork on HuggingFace Hub
    (identical weights, no license approval needed) and the ``tribev2-infer``
    CLI from the tribev2-rs Rust crate.  The Rust binary handles:
    - LLaMA 3.2 3B text feature extraction via llama-cpp-4 (Metal GPU)
    - Fusion transformer inference
    - Output as float32 binary

    Python side handles:
    - Subprocess orchestration
    - Binary result parsing
    - Yeo 7-network interpretation
    """

    def __init__(self, hardware: HardwareInfo) -> None:
        self._hardware = hardware
        self._binary_path: Path | None = None
        self._model_files: dict[str, Path] | None = None
        self._gguf_path: Path | None = None
        self._network_ids: np.ndarray | None = None
        self._checked = False
        self._available = False
        self._check_availability()

    def _check_availability(self) -> None:
        """Check all dependencies and cache their paths."""
        if self._checked:
            return
        self._checked = True

        self._binary_path = _find_rust_binary()
        self._model_files = _find_eugenehp_model_files()
        self._gguf_path = _find_llama_gguf()

        if self._binary_path and self._model_files and self._gguf_path:
            self._available = True
            logger.info(
                "TribeV2RustBackend available: binary=%s, model=%s, gguf=%s",
                self._binary_path.name,
                self._model_files["weights"].name,
                self._gguf_path.name,
            )
        else:
            missing = []
            if not self._binary_path:
                missing.append(
                    "tribev2-infer binary (run: cargo build --release --bin tribev2-infer)"
                )
            if not self._model_files:
                missing.append("eugenehp/tribev2 model (downloaded on first run)")
            if not self._gguf_path:
                missing.append("LLaMA 3.2 3B GGUF (available via Ollama)")
            logger.warning(
                "TribeV2RustBackend unavailable: %s.",
                ", ".join(missing),
            )

    @property
    def name(self) -> str:
        return "tribe_v2_rust"

    def is_loaded(self) -> bool:
        return self._available

    def _ensure_network_ids(self) -> None:
        if self._network_ids is None:
            self._network_ids = load_yeo7_network_ids()

    def _run_inference(self, input_args: list[str], timeout: int = 120) -> np.ndarray:
        """Run tribev2-infer and return the raw activation array.

        Args:
            input_args: Additional CLI args for the input modality
                (e.g. ["--prompt", text] or ["--video-path", path]).
            timeout: Subprocess timeout in seconds.

        Returns:
            Activation array of shape (n_timesteps, 20484).
        """
        config_path = self._model_files["config"]
        weights_path = self._model_files["weights"]
        build_args_path = self._model_files["build_args"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.bin"

            cmd = [
                str(self._binary_path),
                "--config",
                str(config_path),
                "--weights",
                str(weights_path),
                "--build-args",
                str(build_args_path),
                "--llama-model",
                str(self._gguf_path),
                *input_args,
                "--output",
                str(output_path),
                "--n-timesteps",
                str(DEFAULT_N_TIMESTEPS),
                "--layer-positions",
                LAYER_POSITIONS,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace")
                logger.error("tribev2-infer failed: %s", stderr[-500:])
                raise RuntimeError(
                    f"tribev2-infer subprocess failed (exit {result.returncode}): "
                    f"{stderr[-300:]}"
                )

            if not output_path.exists():
                raise RuntimeError(f"tribev2-infer did not produce output file: {output_path}")

            data = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
            return data.reshape(DEFAULT_N_TIMESTEPS, 20484)

    def _interpret_and_build_result(
        self,
        activation: np.ndarray,
        content_type: str,
        content_length: int,
        start_time: float,
    ) -> ContentAnalysis:
        """Interpret neural activation and build ContentAnalysis."""
        neural = interpret_activation(activation, self._network_ids)

        # Use the new persuasion signal for scoring (region-level, science-backed)
        from tribe.interpretation.neural import persuasion_signal_to_score

        manipulation_score = persuasion_signal_to_score(neural.persuasion_signal)

        # Primary trigger from persuasion analysis
        primary_trigger = _trigger_from_persuasion(neural.persuasion_scores)
        trigger_confidence = neural.persuasion_signal

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return ContentAnalysis(
            primary_trigger=primary_trigger,
            trigger_confidence=round(trigger_confidence, 3),
            manipulation_score=manipulation_score,
            techniques=[],
            emotions=[],
            neural=neural,
            content_type=content_type,
            content_length=content_length,
            backend=self.name,
            processing_time_ms=elapsed_ms,
            model_versions={
                "tribe_v2": "eugenehp/tribev2 (safetensors)",
                "llm": "LLaMA 3.2 3B (GGUF via llama-cpp-4 Metal)",
                "atlas": "Yeo2011_7Networks + Destrieux",
            },
        )

    def _ensure_available(self) -> None:
        if not self._available:
            raise RuntimeError(
                "TribeV2RustBackend not available. "
                "Ensure tribev2-infer binary, eugenehp/tribev2 model, and "
                "LLaMA GGUF are available."
            )

    def analyze_text(self, text: str) -> ContentAnalysis:
        """Analyze text via LLaMA embeddings (--prompt)."""
        self._ensure_available()
        self._ensure_network_ids()
        start_time = time.monotonic()

        activation = self._run_inference(["--prompt", text])
        return self._interpret_and_build_result(activation, "text", len(text.split()), start_time)

    def analyze_text_via_audio(self, text: str) -> ContentAnalysis:
        """Analyze text via TTS → audio → Wav2Vec-BERT pipeline (--text-path).

        This routes text through the native audio encoder rather than LLaMA,
        producing stronger brain encoding signal for manipulation detection.
        """
        self._ensure_available()
        self._ensure_network_ids()
        start_time = time.monotonic()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            text_path = f.name

        try:
            activation = self._run_inference(["--text-path", text_path], timeout=300)
            return self._interpret_and_build_result(
                activation, "text_via_audio", len(text.split()), start_time
            )
        finally:
            Path(text_path).unlink(missing_ok=True)

    def analyze_media(self, path: str, media_type: str) -> ContentAnalysis:
        """Analyze video or audio via TRIBE v2 brain encoding.

        Args:
            path: Path to the media file (.mp4, .wav, .mp3, etc.).
            media_type: "video" or "audio".
        """
        self._ensure_available()
        self._ensure_network_ids()
        start_time = time.monotonic()

        if media_type == "video":
            input_args = ["--video-path", path]
            timeout = 600  # Video processing is slower
        elif media_type == "audio":
            input_args = ["--audio-path", path]
            timeout = 300
        else:
            raise ValueError(f"Unsupported media type: {media_type}")

        activation = self._run_inference(input_args, timeout=timeout)
        return self._interpret_and_build_result(activation, media_type, 0, start_time)
