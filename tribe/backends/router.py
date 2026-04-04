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

    @property
    def can_run_tribe_v2(self) -> bool:
        if self.has_cuda:
            return self.vram_gb >= 10.0
        if self.has_mps:
            # MPS uses unified memory; assume 16GB+ is enough
            return True
        return False


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
    """Get the appropriate analysis backend.

    Args:
        force_backend: Override auto-detection. "tribe" or "cls".
        hardware: Pre-detected hardware info. Auto-detected if None.

    Returns:
        An initialized AnalysisBackend instance.
    """
    if hardware is None:
        hardware = detect_hardware()

    if force_backend == "tribe":
        from tribe.backends.tribe_v2 import TribeV2Backend

        return TribeV2Backend(hardware)

    if force_backend == "cls":
        from tribe.backends.classifier import ClassifierBackend

        return ClassifierBackend()

    # Auto-detect
    if hardware.can_run_tribe_v2:
        try:
            import tribev2  # noqa: F401 — eager check before backend instantiation
            from tribe.backends.tribe_v2 import TribeV2Backend

            return TribeV2Backend(hardware)
        except (ImportError, Exception):
            pass

    from tribe.backends.classifier import ClassifierBackend

    return ClassifierBackend()
