"""Tests for the context extraction module."""

from pathlib import Path

import pytest

from staged_qwen3_5_scivqa.context import get_paper_context


@pytest.mark.unit
def test_get_paper_context(tmp_data_dir: Path) -> None:
    """Test context extraction with a mock content.json structure."""
    json_file = tmp_data_dir / "test_paper" / "images" / "fig1.json"
    context = get_paper_context(json_file, window_size=2)

    assert "Figure 1: Growth rate vs temperature." in context
    assert "Introduction paragraph before the figure." in context
    assert "Second paragraph before the figure." in context
    assert "First paragraph after the figure." in context
    assert "Second paragraph after the figure." in context


@pytest.mark.unit
def test_get_paper_context_missing_content(tmp_data_dir: Path) -> None:
    """Test context extraction when content.json doesn't exist."""
    json_file = tmp_data_dir / "other_paper" / "images" / "fig1.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text("{}")

    context = get_paper_context(json_file)
    assert "not found" in context


@pytest.mark.unit
def test_get_paper_context_image_not_in_content(tmp_data_dir: Path) -> None:
    """Test context extraction when the image is not listed in content.json."""
    json_file = tmp_data_dir / "test_paper" / "images" / "fig_missing.json"
    json_file.write_text("{}")

    context = get_paper_context(json_file)
    assert "not found" in context
