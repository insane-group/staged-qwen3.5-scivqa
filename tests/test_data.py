"""Tests for the data loading module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from staged_qwen3_5_scivqa.data import (
    load_test_dataset,
    load_vqa_dataset,
)


@pytest.mark.unit
def test_load_vqa_dataset(tmp_data_dir: Path) -> None:
    """Test VQA dataset loading with mock data."""
    data_root = tmp_data_dir / "test_paper"
    with (
        patch("staged_qwen3_5_scivqa.data.COMPETITION_DATA_DIR", data_root),
        patch("staged_qwen3_5_scivqa.data.Image.open") as mock_open,
    ):
        mock_img = MagicMock()
        mock_img.crop.return_value = MagicMock()
        mock_open.return_value = mock_img

        samples, valid, invalid = load_vqa_dataset("", answer_types=["Factoid"])

        # Should find the sample annotation with one Factoid question
        assert valid >= 1


@pytest.mark.unit
def test_load_vqa_dataset_all_types(tmp_data_dir: Path) -> None:
    """Test VQA dataset loading without type filter."""
    data_root = tmp_data_dir / "test_paper"
    with (
        patch("staged_qwen3_5_scivqa.data.COMPETITION_DATA_DIR", data_root),
        patch("staged_qwen3_5_scivqa.data.Image.open") as mock_open,
    ):
        mock_img = MagicMock()
        mock_img.crop.return_value = MagicMock()
        mock_open.return_value = mock_img

        samples, valid, invalid = load_vqa_dataset("")

        # Should find both Factoid and Yes/No questions
        assert valid >= 2


@pytest.mark.unit
def test_load_test_dataset(tmp_data_dir: Path) -> None:
    """Test test dataset loading with caches."""
    summary_cache = {"test_paper/fig1": {"a": "Test summary"}}
    extraction_cache = {"test_paper/fig1": {"a": "a,b;c,d"}}

    data_root = tmp_data_dir / "test_paper"
    with (
        patch("staged_qwen3_5_scivqa.data.COMPETITION_DATA_DIR", data_root),
        patch("staged_qwen3_5_scivqa.data.Image.open") as mock_open,
    ):
        mock_img = MagicMock()
        mock_img.crop.return_value = MagicMock()
        mock_open.return_value = mock_img

        samples = load_test_dataset(
            "",
            summary_cache=summary_cache,
            extraction_cache=extraction_cache,
        )

        assert len(samples) >= 2
        assert "messages" in samples[0]
        assert "meta" in samples[0]
