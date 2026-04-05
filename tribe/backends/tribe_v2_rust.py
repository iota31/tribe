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
        from huggingface_hub import hf_hub_download
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
        Path.home() / ".ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


# Ratio → score mapping (same as tribe_v2.py)
def _ratio_to_score(ratio: float) -> float:
    if ratio <= 1.0:
        return round(ratio * 2.0, 1)
    if ratio <= 2.0:
        return round(2.0 + (ratio - 1.0) * 3.0, 1)
    if ratio <= 3.0:
        return round(5.0 + (ratio - 2.0) * 2.5, 1)
    return round(min(7.5 + (ratio - 3.0) * 1.25, 10.0), 1)


NETWORK_TRIGGER_MAP = {
    "Salience": "Fear",
    "Default_Mode": "Self-Referential Anxiety",
    "Limbic": "Outrage",
    "Executive_Control": "Analytical Engagement",
    "Dorsal_Attention": "Focused Attention",
}


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
                missing.append("tribev2-infer binary (run: cargo build --release --bin tribev2-infer)")
            if not self._model_files:
                missing.append("eugenehp/tribev2 model (downloaded on first run)")
            if not self._gguf_path:
                missing.append("LLaMA 3.2 3B GGUF (available via Ollama)")
            logger.warning(
                "TribeV2RustBackend unavailable: %s. Falling back to classifier.",
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

    def analyze_text(self, text: str) -> ContentAnalysis:
        """Analyze text via TRIBE v2 neural brain encoding (Rust binary)."""
        if not self._available:
            raise RuntimeError(
                "TribeV2RustBackend not available. "
                "Ensure tribev2-infer binary, eugenehp/tribev2 model, and "
                "LLaMA GGUF are available."
            )

        self._ensure_network_ids()
        start_time = time.monotonic()

        config_path = self._model_files["config"]
        weights_path = self._model_files["weights"]
        build_args_path = self._model_files["build_args"]
        gguf_path = self._gguf_path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.bin"

            # Build the tribev2-infer command
            cmd = [
                str(self._binary_path),
                "--config", str(config_path),
                "--weights", str(weights_path),
                "--build-args", str(build_args_path),
                "--llama-model", str(gguf_path),
                "--prompt", text,
                "--output", str(output_path),
                "--n-timesteps", str(DEFAULT_N_TIMESTEPS),
                "--layer-positions", LAYER_POSITIONS,
            ]

            # Run with stderr suppressed (verbose info from Rust binary)
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace")
                logger.error("tribev2-infer failed: %s", stderr[-500:])
                raise RuntimeError(
                    f"tribev2-infer subprocess failed (exit {result.returncode}): "
                    f"{stderr[-300:]}"
                )

            # Parse binary output
            if not output_path.exists():
                raise RuntimeError(
                    f"tribev2-infer did not produce output file: {output_path}"
                )

            data = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
            n_timesteps = DEFAULT_N_TIMESTEPS
            n_vertices = 20484
            activation = data.reshape(n_timesteps, n_vertices)

        # Interpret via Yeo 7-network
        neural = interpret_activation(activation, self._network_ids)

        manipulation_score = _ratio_to_score(neural.manipulation_ratio)
        primary_trigger = NETWORK_TRIGGER_MAP.get(
            neural.dominant_network, "Manipulation"
        )
        # Confidence: ratio of emotional network activation to total activation,
        # giving a 0-1 measure of how "emotional" the predicted response is.
        emotional_sum = sum(
            neural.network_scores.get(net, 0.0)
            for net in ("Salience", "Default_Mode", "Limbic")
            if neural.network_scores.get(net, 0.0) > 0
        )
        total_positive = sum(
            s for s in neural.network_scores.values() if s > 0
        )
        if total_positive > 0:
            trigger_confidence = min(emotional_sum / total_positive, 1.0)
        else:
            trigger_confidence = 0.0

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return ContentAnalysis(
            primary_trigger=primary_trigger,
            trigger_confidence=round(trigger_confidence, 3),
            manipulation_score=manipulation_score,
            techniques=[],  # Neural network — no technique labels
            emotions=[],  # Neural network — no emotion labels
            neural=neural,
            content_type="text",
            content_length=len(text.split()),
            backend=self.name,
            processing_time_ms=elapsed_ms,
            model_versions={
                "tribe_v2": "eugenehp/tribev2 (safetensors)",
                "llm": "LLaMA 3.2 3B (GGUF via llama-cpp-4 Metal)",
                "atlas": "Yeo2011_7Networks",
            },
        )

    def analyze_media(self, path: str, media_type: str) -> ContentAnalysis:
        """Media analysis not yet supported via Rust backend."""
        return ContentAnalysis(
            primary_trigger="unsupported",
            trigger_confidence=0.0,
            manipulation_score=0.0,
            techniques=[],
            emotions=[],
            neural=None,
            content_type=media_type,
            content_length=0,
            backend=self.name,
            processing_time_ms=0,
            model_versions=self.model_versions if hasattr(self, "model_versions") else {},
        )
