"""Backend router — auto-detect hardware and select analysis backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tribe.backends.base import AnalysisBackend


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""

    has_cuda: bool = False
    has_mps: bool = False
    cuda_device_name: str = ""
    vram_gb: float = 0.0

    @property
    def has_gpu(self) -> bool:
        return self.has_cuda or self.has_mps


def detect_hardware() -> HardwareInfo:
    """Detect available GPU hardware."""
    info = HardwareInfo()

    try:
        import torch

        if torch.cuda.is_available():
            info.has_cuda = True
            info.cuda_device_name = torch.cuda.get_device_name(0)
            info.vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info.has_mps = True
    except ImportError:
        pass

    return info


def get_backend(
    force_backend: str | None = None,
    hardware: HardwareInfo | None = None,
) -> AnalysisBackend:
    """Get the TRIBE v2 Rust analysis backend.

    The Rust binary (tribev2-infer) handles GPU/CPU selection internally
    via llama.cpp. Metal GPU is used when available, CPU otherwise.

    Args:
        force_backend: Ignored (kept for API compatibility). Only TRIBE v2 Rust is available.
        hardware: Pre-detected hardware info. Auto-detected if None.

    Returns:
        An initialized TribeV2RustBackend instance.

    Raises:
        RuntimeError: If the tribev2-infer binary or required models are not found.
    """
    if hardware is None:
        hardware = detect_hardware()

    from tribe.backends.tribe_v2_rust import TribeV2RustBackend

    backend = TribeV2RustBackend(hardware)
    if backend.is_loaded():
        return backend

    raise RuntimeError(
        "TRIBE v2 Rust backend not available. Required components:\n\n"
        "  1. Build the tribev2-infer binary:\n"
        "     # GPU (MacBook M-series, ~25s inference)\n"
        "     cd /tmp/tribev2-rs && cargo build --release --bin tribev2-infer "
        '--features "default,llama-metal"\n\n'
        "     # CPU (any machine, ~2-5 min inference)\n"
        "     cd /tmp/tribev2-rs && cargo build --release --bin tribev2-infer "
        "--features default\n\n"
        "  2. Download LLaMA 3.2 3B:\n"
        "     ollama pull llama3.2\n\n"
        "  3. The eugenehp/tribev2 model weights (downloaded automatically on first run)\n\n"
        "Run 'tribe backends' to check what's missing."
    )
