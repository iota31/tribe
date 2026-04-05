"""Tests for the benchmarks SVG visualization module."""

from __future__ import annotations

import json


def _make_results_json(dataset: str, label_1_score: float, label_0_score: float) -> dict:
    """Create a minimal results dict matching expected schema."""
    return {
        "dataset": dataset,
        "scores": [
            {
                "id": "1",
                "text_preview": "manip text",
                "manipulation_score": label_1_score,
                "label": 1,
            },
            {
                "id": "2",
                "text_preview": "neutral text",
                "manipulation_score": label_0_score,
                "label": 0,
            },
            {
                "id": "3",
                "text_preview": "manip text 2",
                "manipulation_score": label_1_score + 0.5,
                "label": 1,
            },
            {
                "id": "4",
                "text_preview": "neutral text 2",
                "manipulation_score": label_0_score - 0.2,
                "label": 0,
            },
        ],
        "metrics": {
            "auc_roc": 0.82,
            "cohens_d": 0.65,
            "mean_positive": label_1_score,
            "mean_negative": label_0_score,
        },
    }


class TestGenerateSummaryBarChart:
    """Tests for summary bar chart SVG generation."""

    def test_creates_svg_file(self, tmp_path):
        """generate_summary_bar_chart creates an SVG file."""
        from tribe.benchmarks.visualize import generate_summary_bar_chart

        results = {
            "semeval": _make_results_json("semeval", 6.5, 3.2),
        }
        out = tmp_path / "bar.svg"
        generate_summary_bar_chart(results, out)

        assert out.exists()
        content = out.read_text()
        assert "<svg" in content
        assert "semeval" in content.lower()

    def test_svg_dimensions(self, tmp_path):
        """SVG has correct width and height."""
        from tribe.benchmarks.visualize import generate_summary_bar_chart

        results = {"semeval": _make_results_json("semeval", 6.0, 3.0)}
        out = tmp_path / "bar.svg"
        generate_summary_bar_chart(results, out)

        content = out.read_text()
        assert 'width="800"' in content
        assert 'height="400"' in content

    def test_dark_background(self, tmp_path):
        """SVG uses dark background style."""
        from tribe.benchmarks.visualize import generate_summary_bar_chart

        results = {"semeval": _make_results_json("semeval", 6.0, 3.0)}
        out = tmp_path / "bar.svg"
        generate_summary_bar_chart(results, out)

        content = out.read_text()
        assert "#0d1117" in content

    def test_multiple_datasets(self, tmp_path):
        """Bar chart handles multiple datasets."""
        from tribe.benchmarks.visualize import generate_summary_bar_chart

        results = {
            "semeval": _make_results_json("semeval", 6.5, 3.2),
            "propaganda": _make_results_json("propaganda", 7.0, 2.8),
            "fakenews": _make_results_json("fakenews", 5.5, 3.5),
        }
        out = tmp_path / "bar.svg"
        generate_summary_bar_chart(results, out)

        content = out.read_text()
        assert out.exists()
        # All dataset names should appear
        assert "semeval" in content.lower()
        assert "propaganda" in content.lower()
        assert "fakenews" in content.lower()


class TestGenerateSeparationPlot:
    """Tests for separation plot SVG generation."""

    def test_creates_svg_file(self, tmp_path):
        """generate_separation_plot creates an SVG file."""
        from tribe.benchmarks.visualize import generate_separation_plot

        results = {"semeval": _make_results_json("semeval", 6.5, 3.2)}
        out = tmp_path / "sep.svg"
        generate_separation_plot(results, out)

        assert out.exists()
        content = out.read_text()
        assert "<svg" in content

    def test_svg_dimensions(self, tmp_path):
        """SVG has correct width and height."""
        from tribe.benchmarks.visualize import generate_separation_plot

        results = {"semeval": _make_results_json("semeval", 6.0, 3.0)}
        out = tmp_path / "sep.svg"
        generate_separation_plot(results, out)

        content = out.read_text()
        assert 'width="800"' in content
        assert 'height="500"' in content

    def test_contains_rect_elements(self, tmp_path):
        """Separation plot contains rect elements for box plots."""
        from tribe.benchmarks.visualize import generate_separation_plot

        results = {"semeval": _make_results_json("semeval", 6.0, 3.0)}
        out = tmp_path / "sep.svg"
        generate_separation_plot(results, out)

        content = out.read_text()
        assert "<rect" in content


class TestGenerateAll:
    """Tests for the generate_all orchestrator."""

    def test_reads_json_and_creates_svgs(self, tmp_path):
        """generate_all reads JSON files and creates both SVG outputs."""
        from tribe.benchmarks.visualize import generate_all

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Write sample results files
        for name in ["semeval", "propaganda"]:
            data = _make_results_json(name, 6.0, 3.0)
            (results_dir / f"{name}_results.json").write_text(json.dumps(data))

        generate_all(results_dir, output_dir)

        assert (output_dir / "summary_bar_chart.svg").exists()
        assert (output_dir / "separation_plot.svg").exists()

    def test_empty_results_dir(self, tmp_path):
        """generate_all handles empty results directory gracefully."""
        from tribe.benchmarks.visualize import generate_all

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Should not raise, just skip (no results to plot)
        generate_all(results_dir, output_dir)
