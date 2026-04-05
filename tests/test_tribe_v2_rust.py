"""Tests for TribeV2RustBackend."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestFindHelpers:
    """Test the _find_* helper functions."""

    def test_find_rust_binary(self):
        from tribe.backends.tribe_v2_rust import _find_rust_binary

        result = _find_rust_binary()
        # May be None if binary not built, but should be a Path if found
        if result is not None:
            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_file()

    def test_find_llama_gguf(self):
        from tribe.backends.tribe_v2_rust import _find_llama_gguf

        result = _find_llama_gguf()
        # May be None if GGUF not present
        if result is not None:
            assert isinstance(result, Path)
            assert result.exists()

    def test_find_eugenehp_model_files_returns_dict(self):
        from tribe.backends.tribe_v2_rust import _find_eugenehp_model_files

        result = _find_eugenehp_model_files()
        # May be None if model not downloaded
        if result is not None:
            assert isinstance(result, dict)
            assert "config" in result
            assert "weights" in result
            assert "build_args" in result
            assert all(isinstance(p, Path) for p in result.values())


class TestBackendInterface:
    """Test that TribeV2RustBackend satisfies the AnalysisBackend interface."""

    def test_name_property(self):
        from tribe.backends.router import HardwareInfo
        from tribe.backends.tribe_v2_rust import TribeV2RustBackend

        backend = TribeV2RustBackend(HardwareInfo())
        assert backend.name == "tribe_v2_rust"

    def test_is_loaded_returns_bool(self):
        from tribe.backends.router import HardwareInfo
        from tribe.backends.tribe_v2_rust import TribeV2RustBackend

        backend = TribeV2RustBackend(HardwareInfo())
        assert isinstance(backend.is_loaded(), bool)


class TestRouterAutoDetection:
    """Test that router auto-selects Rust backend when available."""

    def test_router_returns_rust_backend_or_errors(self):
        from tribe.backends.router import get_backend

        try:
            backend = get_backend()
            assert backend.name == "tribe_v2_rust"
        except RuntimeError:
            # Expected when binary is not installed
            pass


class TestAnalyzeTextIntegration:
    """Integration test for analyze_text — requires full setup."""

    def test_analyze_text_produces_valid_content_analysis(self):
        from tribe.backends.router import HardwareInfo
        from tribe.backends.tribe_v2_rust import TribeV2RustBackend

        backend = TribeV2RustBackend(HardwareInfo())
        if not backend.is_loaded():
            pytest.skip("TribeV2RustBackend not available (missing binary/model/GGUF)")

        fixture = Path(__file__).parent / "fixtures" / "manipulative_article.txt"
        text = fixture.read_text()

        result = backend.analyze_text(text)

        assert result.backend == "tribe_v2_rust"
        assert 0.0 <= result.manipulation_score <= 10.0
        assert 0.0 <= result.trigger_confidence <= 1.0
        assert result.content_type == "text"
        assert result.content_length > 0
        assert result.processing_time_ms > 0
        assert result.model_versions["tribe_v2"] == "eugenehp/tribev2 (safetensors)"
        assert result.neural is not None

    def test_analyze_text_model_versions_includes_llm(self):
        from tribe.backends.router import HardwareInfo
        from tribe.backends.tribe_v2_rust import TribeV2RustBackend

        backend = TribeV2RustBackend(HardwareInfo())
        if not backend.is_loaded():
            pytest.skip("TribeV2RustBackend not available")

        result = backend.analyze_text("The government is hiding the truth from you.")
        assert "llm" in result.model_versions
        assert "llama" in result.model_versions["llm"].lower()

    def test_analyze_text_neural_has_network_scores(self):
        from tribe.backends.router import HardwareInfo
        from tribe.backends.tribe_v2_rust import TribeV2RustBackend

        backend = TribeV2RustBackend(HardwareInfo())
        if not backend.is_loaded():
            pytest.skip("TribeV2RustBackend not available")

        result = backend.analyze_text("Breaking: scientists discover miracle cure.")
        assert result.neural is not None
        assert isinstance(result.neural.network_scores, dict)
        assert len(result.neural.network_scores) > 0
        assert isinstance(result.neural.dominant_network, str)
        assert isinstance(result.neural.manipulation_ratio, float)
