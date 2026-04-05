"""Analysis backends — TRIBE v2 Rust (GPU/CPU)."""

from tribe.backends.router import detect_hardware, get_backend

__all__ = ["get_backend", "detect_hardware"]
