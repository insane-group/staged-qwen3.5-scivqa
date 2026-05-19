"""Tests for the analysis module."""

from pathlib import Path

import pytest

from staged_qwen3_5_scivqa.analysis import (
    analyze_dataset,
    report_missing_data_extraction,
    report_yes_no_balance,
)


@pytest.mark.unit
def test_analyze_dataset(tmp_data_dir: Path, mock_tokenizer) -> None:
    """Test dataset analysis with mock tokenizer."""
    df = analyze_dataset(tmp_data_dir, mock_tokenizer)

    assert len(df) > 0
    assert "answer_tokens" in df.columns
    assert "image_width" in df.columns
    assert "image_height" in df.columns
    assert "image_pixels" in df.columns


@pytest.mark.unit
def test_report_yes_no_balance(tmp_data_dir: Path) -> None:
    """Test Yes/No balance reporting."""
    report = report_yes_no_balance(tmp_data_dir)

    assert "summary" in report
    assert report["summary"]["total_processed"] >= 1
    assert report["summary"]["yes"] >= 1


@pytest.mark.unit
def test_report_missing_data_extraction(tmp_data_dir: Path) -> None:
    """Test missing data extraction reporting."""
    total, missing, pct = report_missing_data_extraction(tmp_data_dir)

    assert total >= 2
    # The sample has data_extraction for subfigure "a", so missing should be 0
    assert missing == 0
    assert pct == 0.0
