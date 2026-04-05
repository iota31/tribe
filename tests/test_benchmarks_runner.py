"""Tests for the benchmark runner module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _mock_analysis_result(score: float = 5.0) -> MagicMock:
    """Create a mock AnalysisResult with the given score."""
    result = MagicMock()
    result.manipulation_score = score
    result.neural = MagicMock()
    result.neural.manipulation_ratio = 1.5
    result.neural.dominant_network = "salience"
    result.primary_trigger = "emotional_appeal"
    result.processing_time_ms = 42.0
    return result


class TestRunBenchmarkPaired:
    """Tests for run_benchmark with the paired dataset."""

    @patch("tribe.benchmarks.runner.get_backend")
    def test_returns_results_dict(self, mock_get_backend: MagicMock, tmp_path: Path) -> None:
        """Should return a dict with expected top-level keys."""
        backend = MagicMock()
        backend.analyze_text.return_value = _mock_analysis_result(6.0)
        mock_get_backend.return_value = backend

        from tribe.benchmarks.runner import run_benchmark

        result = run_benchmark("paired", results_dir=tmp_path)

        assert isinstance(result, dict)
        assert result["dataset"] == "paired"
        assert "n_total" in result
        assert "n_successful" in result
        assert "n_failed" in result
        assert "scores" in result
        assert "metrics" in result
        assert "timestamp" in result

    @patch("tribe.benchmarks.runner.get_backend")
    def test_paired_metrics_keys(self, mock_get_backend: MagicMock, tmp_path: Path) -> None:
        """Paired benchmark should produce win_rate and paired t-test metrics."""
        backend = MagicMock()
        # Manipulative items score higher
        call_count = [0]

        def side_effect(text: str) -> MagicMock:
            call_count[0] += 1
            # Odd calls = manipulative (pair format: manip first, neutral second)
            score = 7.0 if call_count[0] % 2 == 1 else 3.0
            return _mock_analysis_result(score)

        backend.analyze_text.side_effect = side_effect
        mock_get_backend.return_value = backend

        from tribe.benchmarks.runner import run_benchmark

        result = run_benchmark("paired", results_dir=tmp_path)
        metrics = result["metrics"]

        assert "win_rate" in metrics
        assert "mean_diff" in metrics
        assert "t_statistic" in metrics
        assert "p_value" in metrics
        assert "n_pairs" in metrics

    @patch("tribe.benchmarks.runner.get_backend")
    def test_saves_json_file(self, mock_get_backend: MagicMock, tmp_path: Path) -> None:
        """Should save results as JSON in the results directory."""
        backend = MagicMock()
        backend.analyze_text.return_value = _mock_analysis_result()
        mock_get_backend.return_value = backend

        from tribe.benchmarks.runner import run_benchmark

        run_benchmark("paired", results_dir=tmp_path)

        output_path = tmp_path / "paired_results.json"
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["dataset"] == "paired"

    @patch("tribe.benchmarks.runner.get_backend")
    def test_handles_analysis_errors(self, mock_get_backend: MagicMock, tmp_path: Path) -> None:
        """Should record errors without crashing."""
        backend = MagicMock()
        backend.analyze_text.side_effect = RuntimeError("model not loaded")
        mock_get_backend.return_value = backend

        from tribe.benchmarks.runner import run_benchmark

        result = run_benchmark("paired", results_dir=tmp_path)

        assert result["n_failed"] == result["n_total"]
        assert result["n_successful"] == 0
        assert all("error" in s for s in result["scores"])


class TestRunBenchmarkInvalidDataset:
    """Tests for unknown dataset names."""

    def test_raises_value_error(self, tmp_path: Path) -> None:
        """Should raise ValueError for unknown dataset."""
        from tribe.benchmarks.runner import run_benchmark

        with pytest.raises(ValueError, match="Unknown dataset"):
            run_benchmark("nonexistent", results_dir=tmp_path)


class TestRunAll:
    """Tests for run_all."""

    @patch("tribe.benchmarks.runner.run_benchmark")
    def test_runs_all_three_datasets(self, mock_run: MagicMock) -> None:
        """Should call run_benchmark for paired, mentalmanip, and semeval."""
        mock_run.return_value = {"dataset": "test", "metrics": {}}

        from tribe.benchmarks.runner import run_all

        result = run_all()

        assert "paired" in result
        assert "mentalmanip" in result
        assert "semeval" in result
        assert mock_run.call_count == 3
