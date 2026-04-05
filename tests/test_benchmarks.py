"""Tests for benchmark dataset parsers."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paired dataset tests (no I/O, embedded data)
# ---------------------------------------------------------------------------


class TestPairedDataset:
    """Tests for the internal paired benchmark dataset."""

    def test_load_returns_list(self) -> None:
        """Should return a list of dicts."""
        from tribe.benchmarks.datasets.paired import load

        result = load()
        assert isinstance(result, list)

    def test_load_returns_50_items(self) -> None:
        """Should return 25 pairs = 50 total items."""
        from tribe.benchmarks.datasets.paired import load

        result = load()
        assert len(result) == 50

    def test_item_has_required_keys(self) -> None:
        """Each item must have id, text, manipulative, topic, pair_id."""
        from tribe.benchmarks.datasets.paired import load

        required_keys = {"id", "text", "manipulative", "topic", "pair_id"}
        for item in load():
            assert required_keys.issubset(item.keys()), f"Missing keys in {item['id']}"

    def test_pair_ids_range_1_to_25(self) -> None:
        """pair_id values should be integers 1 through 25."""
        from tribe.benchmarks.datasets.paired import load

        pair_ids = {item["pair_id"] for item in load()}
        assert pair_ids == set(range(1, 26))

    def test_each_pair_has_manipulative_and_neutral(self) -> None:
        """Each pair_id should have exactly one manipulative and one neutral."""
        from tribe.benchmarks.datasets.paired import load

        pairs: dict[int, list[bool]] = {}
        for item in load():
            pairs.setdefault(item["pair_id"], []).append(item["manipulative"])
        for pid, flags in pairs.items():
            assert sorted(flags) == [False, True], f"Pair {pid} missing a variant"

    def test_text_length_minimum(self) -> None:
        """Each text should be at least 100 characters (substantive)."""
        from tribe.benchmarks.datasets.paired import load

        for item in load():
            assert len(item["text"]) >= 100, f"{item['id']} text too short"

    def test_id_format(self) -> None:
        """IDs should follow the pattern pair_NN_manipulative or pair_NN_neutral."""
        import re

        from tribe.benchmarks.datasets.paired import load

        pattern = re.compile(r"^pair_\d{2}_(manipulative|neutral)$")
        for item in load():
            assert pattern.match(item["id"]), f"Bad ID format: {item['id']}"

    def test_manipulative_flag_matches_id(self) -> None:
        """The manipulative bool should match the id suffix."""
        from tribe.benchmarks.datasets.paired import load

        for item in load():
            if item["id"].endswith("_manipulative"):
                assert item["manipulative"] is True
            else:
                assert item["manipulative"] is False

    def test_25_unique_topics(self) -> None:
        """There should be 25 distinct topics."""
        from tribe.benchmarks.datasets.paired import load

        topics = {item["topic"] for item in load()}
        assert len(topics) == 25


# ---------------------------------------------------------------------------
# SemEval dataset tests (mocked I/O)
# ---------------------------------------------------------------------------


class TestSemEvalDataset:
    """Tests for SemEval-2020 Task 11 parser."""

    def _make_fake_tgz(self, tmp_path: Path) -> bytes:
        """Create a minimal fake .tgz matching SemEval structure."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            # Article file
            article_content = b"This is a test article about propaganda techniques."
            info = tarfile.TarInfo(name="datasets/train-articles/article111111111.txt")
            info.size = len(article_content)
            tar.addfile(info, io.BytesIO(article_content))

            # TSV annotation file
            tsv_content = b"111111111\tLoaded_Language\t10\t20\n111111111\tName_Calling\t25\t35\n"
            info = tarfile.TarInfo(
                name="datasets/train-labels-task2-technique-classification/train-task2-TC.labels"
            )
            info.size = len(tsv_content)
            tar.addfile(info, io.BytesIO(tsv_content))
        return buf.getvalue()

    def test_download_creates_file(self, tmp_path: Path) -> None:
        """download() should fetch and extract the archive."""
        from tribe.benchmarks.datasets.semeval import download

        fake_tgz = self._make_fake_tgz(tmp_path)
        mock_response = MagicMock()
        mock_response.content = fake_tgz
        mock_response.raise_for_status = MagicMock()

        with patch("tribe.benchmarks.datasets.semeval.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_response
            download(tmp_path)

        # Should have extracted files
        assert any(tmp_path.rglob("*.txt"))

    def test_download_skips_if_present(self, tmp_path: Path) -> None:
        """download() should not re-download if data already exists."""
        from tribe.benchmarks.datasets.semeval import download

        # Create marker directory
        (tmp_path / "datasets").mkdir()
        (tmp_path / "datasets" / "train-articles").mkdir(parents=True)
        (tmp_path / "datasets" / "train-articles" / "article1.txt").write_text("exists")

        with patch("tribe.benchmarks.datasets.semeval.httpx") as mock_httpx:
            download(tmp_path)
            mock_httpx.get.assert_not_called()

    def test_load_parses_articles_and_annotations(self, tmp_path: Path) -> None:
        """load() should combine articles with their annotation spans."""
        from tribe.benchmarks.datasets.semeval import load

        # Set up extracted directory structure
        articles_dir = tmp_path / "datasets" / "train-articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "article111111111.txt").write_text(
            "This is a test article about propaganda techniques."
        )

        labels_dir = tmp_path / "datasets" / "train-labels-task2-technique-classification"
        labels_dir.mkdir(parents=True)
        (labels_dir / "train-task2-TC.labels").write_text(
            "111111111\tLoaded_Language\t10\t20\n111111111\tName_Calling\t25\t35\n"
        )

        result = load(tmp_path)
        assert len(result) == 1
        assert result[0]["id"] == "111111111"
        assert "test article" in result[0]["text"]
        assert len(result[0]["propaganda_spans"]) == 2
        assert result[0]["propaganda_spans"][0] == ("Loaded_Language", 10, 20)
        assert isinstance(result[0]["propaganda_density"], float)
        assert 0.0 < result[0]["propaganda_density"] < 1.0

    def test_load_density_calculation(self, tmp_path: Path) -> None:
        """propaganda_density should be propaganda chars / total chars."""
        from tribe.benchmarks.datasets.semeval import load

        articles_dir = tmp_path / "datasets" / "train-articles"
        articles_dir.mkdir(parents=True)
        text = "a" * 100
        (articles_dir / "article222222222.txt").write_text(text)

        labels_dir = tmp_path / "datasets" / "train-labels-task2-technique-classification"
        labels_dir.mkdir(parents=True)
        # Span covers chars 0-50 = 50 chars out of 100
        (labels_dir / "train-task2-TC.labels").write_text("222222222\tFear\t0\t50\n")

        result = load(tmp_path)
        assert result[0]["propaganda_density"] == pytest.approx(0.5)

    def test_load_no_annotations(self, tmp_path: Path) -> None:
        """Articles with no annotations should have empty spans and 0.0 density."""
        from tribe.benchmarks.datasets.semeval import load

        articles_dir = tmp_path / "datasets" / "train-articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "article333333333.txt").write_text("Clean article with no propaganda.")

        labels_dir = tmp_path / "datasets" / "train-labels-task2-technique-classification"
        labels_dir.mkdir(parents=True)
        (labels_dir / "train-task2-TC.labels").write_text("")

        result = load(tmp_path)
        assert len(result) == 1
        assert result[0]["propaganda_spans"] == []
        assert result[0]["propaganda_density"] == 0.0


# ---------------------------------------------------------------------------
# MentalManip dataset tests (mocked I/O)
# ---------------------------------------------------------------------------


class TestMentalManipDataset:
    """Tests for MentalManip CSV parser."""

    SAMPLE_CSV = (
        "ID,Dialogue,Manipulative,Technique,Vulnerability\n"
        '1,"A: Hello. B: You never listen to me.",Yes,"Guilt-tripping","Low self-esteem"\n'
        '2,"A: How are you? B: I am fine.",No,"",""\n'
    )

    def test_download_creates_file(self, tmp_path: Path) -> None:
        """download() should fetch and save the CSV."""
        from tribe.benchmarks.datasets.mentalmanip import download

        mock_response = MagicMock()
        mock_response.text = self.SAMPLE_CSV
        mock_response.raise_for_status = MagicMock()

        with patch("tribe.benchmarks.datasets.mentalmanip.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_response
            download(tmp_path)

        assert (tmp_path / "mentalmanip_con.csv").exists()

    def test_download_skips_if_present(self, tmp_path: Path) -> None:
        """download() should not re-download if CSV exists."""
        from tribe.benchmarks.datasets.mentalmanip import download

        (tmp_path / "mentalmanip_con.csv").write_text(self.SAMPLE_CSV)

        with patch("tribe.benchmarks.datasets.mentalmanip.httpx") as mock_httpx:
            download(tmp_path)
            mock_httpx.get.assert_not_called()

    def test_load_parses_csv(self, tmp_path: Path) -> None:
        """load() should parse the CSV into structured dicts."""
        from tribe.benchmarks.datasets.mentalmanip import load

        (tmp_path / "mentalmanip_con.csv").write_text(self.SAMPLE_CSV)

        result = load(tmp_path)
        assert len(result) == 2

    def test_load_item_keys(self, tmp_path: Path) -> None:
        """Each item must have id, text, manipulative, techniques."""
        from tribe.benchmarks.datasets.mentalmanip import load

        (tmp_path / "mentalmanip_con.csv").write_text(self.SAMPLE_CSV)

        for item in load(tmp_path):
            assert "id" in item
            assert "text" in item
            assert "manipulative" in item
            assert "techniques" in item

    def test_load_manipulative_flag(self, tmp_path: Path) -> None:
        """manipulative field should be a bool."""
        from tribe.benchmarks.datasets.mentalmanip import load

        (tmp_path / "mentalmanip_con.csv").write_text(self.SAMPLE_CSV)

        result = load(tmp_path)
        assert result[0]["manipulative"] is True
        assert result[1]["manipulative"] is False

    def test_load_techniques_list(self, tmp_path: Path) -> None:
        """techniques should be a list (possibly empty)."""
        from tribe.benchmarks.datasets.mentalmanip import load

        (tmp_path / "mentalmanip_con.csv").write_text(self.SAMPLE_CSV)

        result = load(tmp_path)
        assert result[0]["techniques"] == ["Guilt-tripping"]
        assert result[1]["techniques"] == []

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """load() should raise FileNotFoundError if CSV missing."""
        from tribe.benchmarks.datasets.mentalmanip import load

        with pytest.raises(FileNotFoundError):
            load(tmp_path)
