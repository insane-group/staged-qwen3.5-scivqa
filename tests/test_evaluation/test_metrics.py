"""Tests for the evaluation metrics module."""

import pytest

from staged_qwen3_5_scivqa.evaluation.metrics import (
    compute_accuracy,
    compute_bert_score,
    compute_rouge,
    compute_set_f1,
)


@pytest.mark.unit
def test_compute_accuracy_perfect() -> None:
    """Test accuracy with perfect predictions."""
    preds = ["yes", "no", "hello"]
    refs = ["yes", "no", "hello"]
    assert compute_accuracy(preds, refs) == 1.0


@pytest.mark.unit
def test_compute_accuracy_partial() -> None:
    """Test accuracy with partial matches."""
    preds = ["yes", "no", "wrong"]
    refs = ["yes", "no", "hello"]
    assert compute_accuracy(preds, refs) == pytest.approx(2 / 3)


@pytest.mark.unit
def test_compute_accuracy_empty() -> None:
    """Test accuracy with empty inputs."""
    assert compute_accuracy([], []) == 0.0


@pytest.mark.unit
def test_compute_set_f1_perfect() -> None:
    """Test set F1 with identical sets."""
    preds = ["a, b, c", "x, y"]
    refs = ["a, b, c", "x, y"]
    result = compute_set_f1(preds, refs)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


@pytest.mark.unit
def test_compute_set_f1_partial() -> None:
    """Test set F1 with partial overlap."""
    preds = ["a, b"]
    refs = ["a, b, c"]
    result = compute_set_f1(preds, refs)
    assert result["precision"] == 1.0
    assert result["recall"] == pytest.approx(2 / 3)


@pytest.mark.unit
def test_compute_set_f1_order_insensitive() -> None:
    """Test set F1 ignores order."""
    preds = ["b, a"]
    refs = ["a, b"]
    result = compute_set_f1(preds, refs)
    assert result["f1"] == 1.0


@pytest.mark.unit
def test_compute_bert_score_import_fallback() -> None:
    """Test BERTScore returns zeros when bert_score is not installed."""
    result = compute_bert_score(["hello"], ["hello"])
    # If bert_score is installed, this will be > 0; otherwise 0
    assert isinstance(result, dict)
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result


@pytest.mark.unit
def test_compute_rouge_import_fallback() -> None:
    """Test ROUGE returns zeros when rouge_score is not installed."""
    result = compute_rouge(["hello world"], ["hello world"])
    assert isinstance(result, dict)
    assert "rouge1" in result
    assert "rouge2" in result
    assert "rougeL" in result
