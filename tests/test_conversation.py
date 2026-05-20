"""Tests for the conversation module."""

from unittest.mock import MagicMock

import pytest

from staged_qwen3_5_scivqa.conversation import (
    convert_to_conversation,
    convert_to_inference_conversation,
)


@pytest.mark.unit
def test_convert_to_conversation() -> None:
    """Test training conversation format."""
    mock_image = MagicMock()
    result = convert_to_conversation("Hello <image>", mock_image, "World")

    assert "messages" in result
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"][0]["text"] == "World"


@pytest.mark.unit
def test_convert_to_inference_conversation() -> None:
    """Test inference conversation format with metadata."""
    mock_image = MagicMock()
    result = convert_to_inference_conversation(
        "Hello <image>", mock_image, sample_id="test", sub_fig="a"
    )

    assert "messages" in result
    assert "meta" in result
    assert result["meta"]["sample_id"] == "test"
    assert result["meta"]["sub_fig"] == "a"
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
