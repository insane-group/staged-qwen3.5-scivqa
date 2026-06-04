"""Tests for the data loading module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from staged_qwen3_5_scivqa.data import (
    balance_yes_no,
    build_summary_dataset,
    build_table_dataset,
    build_vqa_dataset,
    load_summary_from_hub,
    load_table_from_hub,
    load_test_dataset,
    load_vqa_dataset,
    load_vqa_from_hub,
)

# ── Existing tests (unchanged) ────────────────────────────────────────


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


# ── Tests for HF dataset builders ─────────────────────────────────────


def _make_category_dir(base: Path, category: str, sample_annotation: dict):
    """Create a competition-data-style directory tree under *base*.

    Produces::

        base/<category>/main/sub/paper1/images/fig1.json
        base/<category>/main/sub/paper1/images/fig1.jpg
        base/<category>/main/sub/paper1/content.json
    """
    from PIL import Image as PilImage

    paper_dir = base / category / "main" / "sub" / "paper1"
    images_dir = paper_dir / "images"
    images_dir.mkdir(parents=True)

    with open(images_dir / "fig1.json", "w") as f:
        json.dump(sample_annotation, f)

    content = [
        {"type": "text", "text": "Context before."},
        {
            "type": "image",
            "img_path": "images/fig1.jpg",
            "img_caption": ["Figure 1: test figure."],
        },
        {"type": "text", "text": "Context after."},
    ]
    with open(paper_dir / "content.json", "w") as f:
        json.dump(content, f)

    # Create a valid 10×10 RGB JPEG using PIL
    img = PilImage.new("RGB", (10, 10), color="white")
    img.save(images_dir / "fig1.jpg", "JPEG")


@pytest.mark.unit
def test_build_vqa_dataset(tmp_data_dir: Path, sample_annotation: dict) -> None:
    """Test building a VQA DatasetDict from mock competition data."""
    _make_category_dir(tmp_data_dir, "train", sample_annotation)
    _make_category_dir(tmp_data_dir, "dev", sample_annotation)

    with patch("staged_qwen3_5_scivqa.data.COMPETITION_DATA_DIR", tmp_data_dir):
        dd = build_vqa_dataset(("train", "dev"), push_to_hub=False, token=None)

    assert "train" in dd
    assert "validation" in dd
    assert set(dd["train"].column_names) == {
        "sample_id",
        "sub_figure",
        "question",
        "question_type",
        "answer_type",
        "answer",
        "context",
        "summary",
        "table",
        "image",
        "bbox",
    }
    assert len(dd["train"]) == 2  # Factoid + Yes/No
    assert dd["train"][0]["answer_type"] in ("Factoid", "Yes/No")
    assert dd["train"][0]["answer"] in ("Yes", "200 C")


@pytest.mark.unit
def test_build_summary_dataset(tmp_data_dir: dict, sample_annotation: dict) -> None:
    """Test building a Summary DatasetDict from mock competition data."""
    _make_category_dir(tmp_data_dir, "train", sample_annotation)

    with patch("staged_qwen3_5_scivqa.data.COMPETITION_DATA_DIR", tmp_data_dir):
        dd = build_summary_dataset(("train",), push_to_hub=False, token=None)

    assert "train" in dd
    assert set(dd["train"].column_names) == {
        "sample_id",
        "sub_figure",
        "summary",
        "context",
        "image",
        "bbox",
    }
    assert len(dd["train"]) == 1
    assert "growth rate" in dd["train"][0]["summary"]


@pytest.mark.unit
def test_build_table_dataset(tmp_data_dir: dict, sample_annotation: dict) -> None:
    """Test building a Table DatasetDict from mock competition data."""
    _make_category_dir(tmp_data_dir, "train", sample_annotation)

    with patch("staged_qwen3_5_scivqa.data.COMPETITION_DATA_DIR", tmp_data_dir):
        dd = build_table_dataset(("train",), push_to_hub=False, token=None)

    assert "train" in dd
    assert set(dd["train"].column_names) == {
        "sample_id",
        "sub_figure",
        "table",
        "context",
        "image",
        "bbox",
    }
    assert len(dd["train"]) == 1
    assert ";" in dd["train"][0]["table"]


# ── Tests for HF dataset loaders ──────────────────────────────────────


@pytest.mark.unit
def test_load_vqa_from_hub() -> None:
    """Test loading VQA dataset from HF (mocked)."""
    from datasets import Dataset, Features, Sequence, Value
    from datasets import Image as HfImage
    from PIL import Image as PilImage

    features = Features(
        {
            "sample_id": Value("string"),
            "sub_figure": Value("string"),
            "question": Value("string"),
            "question_type": Value("string"),
            "answer_type": Value("string"),
            "answer": Value("string"),
            "context": Value("string"),
            "summary": Value("string"),
            "table": Value("string"),
            "image": HfImage(),
            "bbox": Sequence(Value("int32"), length=4),
        }
    )

    mock_ds = Dataset.from_list(
        [
            {
                "sample_id": "p1/fig1",
                "sub_figure": "a",
                "question": "What is the max?",
                "question_type": "Factoid",
                "answer_type": "Factoid",
                "answer": "200 C",
                "context": "Some context",
                "summary": "",
                "table": "",
                "image": PilImage.new("RGB", (10, 10)),
                "bbox": [0, 0, 100, 100],
            },
            {
                "sample_id": "p1/fig1",
                "sub_figure": "a",
                "question": "Does it saturate?",
                "question_type": "Yes/No",
                "answer_type": "Yes/No",
                "answer": "Yes",
                "context": "Some context",
                "summary": "",
                "table": "",
                "image": PilImage.new("RGB", (10, 10)),
                "bbox": [0, 0, 100, 100],
            },
        ],
        features=features,
    )

    with patch("datasets.load_dataset", return_value=mock_ds):
        ds = load_vqa_from_hub(repo_id="test/repo", split="train")

    assert len(ds) == 2
    assert ds[0]["answer_type"] == "Factoid"


@pytest.mark.unit
def test_load_vqa_from_hub_with_filter() -> None:
    """Test loading VQA with answer type filter (mocked)."""
    from datasets import Dataset, Features, Sequence, Value
    from datasets import Image as HfImage
    from PIL import Image as PilImage

    features = Features(
        {
            "sample_id": Value("string"),
            "sub_figure": Value("string"),
            "question": Value("string"),
            "question_type": Value("string"),
            "answer_type": Value("string"),
            "answer": Value("string"),
            "context": Value("string"),
            "summary": Value("string"),
            "table": Value("string"),
            "image": HfImage(),
            "bbox": Sequence(Value("int32"), length=4),
        }
    )

    mock_ds = Dataset.from_list(
        [
            {
                "sample_id": "p1/fig1",
                "sub_figure": "a",
                "question": "What is the max?",
                "question_type": "Factoid",
                "answer_type": "Factoid",
                "answer": "200 C",
                "context": "Ctx",
                "summary": "",
                "table": "",
                "image": PilImage.new("RGB", (10, 10)),
                "bbox": [0, 0, 100, 100],
            },
            {
                "sample_id": "p1/fig1",
                "sub_figure": "a",
                "question": "Does it?",
                "question_type": "Yes/No",
                "answer_type": "Yes/No",
                "answer": "Yes",
                "context": "Ctx",
                "summary": "",
                "table": "",
                "image": PilImage.new("RGB", (10, 10)),
                "bbox": [0, 0, 100, 100],
            },
        ],
        features=features,
    )

    with patch("datasets.load_dataset", return_value=mock_ds):
        filtered = load_vqa_from_hub(
            repo_id="test/repo", split="train", answer_types=["Factoid"]
        )

    assert len(filtered) == 1
    assert filtered[0]["answer_type"] == "Factoid"


@pytest.mark.unit
def test_load_summary_from_hub() -> None:
    """Test loading Summary dataset from HF (mocked)."""
    from datasets import Dataset, Features, Sequence, Value
    from datasets import Image as HfImage
    from PIL import Image as PilImage

    features = Features(
        {
            "sample_id": Value("string"),
            "sub_figure": Value("string"),
            "summary": Value("string"),
            "context": Value("string"),
            "image": HfImage(),
            "bbox": Sequence(Value("int32"), length=4),
        }
    )

    mock_ds = Dataset.from_list(
        [
            {
                "sample_id": "p1/fig1",
                "sub_figure": "a",
                "summary": "Growth increases with temperature.",
                "context": "Paper context",
                "image": PilImage.new("RGB", (10, 10)),
                "bbox": [0, 0, 100, 100],
            }
        ],
        features=features,
    )

    with patch("datasets.load_dataset", return_value=mock_ds):
        ds = load_summary_from_hub(repo_id="test/summary", split="train")

    assert len(ds) == 1
    assert "Growth" in ds[0]["summary"]


@pytest.mark.unit
def test_load_table_from_hub() -> None:
    """Test loading Table dataset from HF (mocked)."""
    from datasets import Dataset, Features, Sequence, Value
    from datasets import Image as HfImage
    from PIL import Image as PilImage

    features = Features(
        {
            "sample_id": Value("string"),
            "sub_figure": Value("string"),
            "table": Value("string"),
            "context": Value("string"),
            "image": HfImage(),
            "bbox": Sequence(Value("int32"), length=4),
        }
    )

    mock_ds = Dataset.from_list(
        [
            {
                "sample_id": "p1/fig1",
                "sub_figure": "a",
                "table": "x,y;1,2;3,4",
                "context": "Paper context",
                "image": PilImage.new("RGB", (10, 10)),
                "bbox": [0, 0, 100, 100],
            }
        ],
        features=features,
    )

    with patch("datasets.load_dataset", return_value=mock_ds):
        ds = load_table_from_hub(repo_id="test/table", split="train")

    assert len(ds) == 1
    assert ";" in ds[0]["table"]


# ── Tests for enhanced loaders with source="hf" ───────────────────────


@pytest.mark.unit
def test_load_vqa_dataset_source_hf() -> None:
    """Test VQA dataset loading with source='hf' (mocked Hub dataset)."""
    from datasets import Dataset, Features, Sequence, Value
    from datasets import Image as HfImage
    from PIL import Image as PilImage

    features = Features(
        {
            "sample_id": Value("string"),
            "sub_figure": Value("string"),
            "question": Value("string"),
            "question_type": Value("string"),
            "answer_type": Value("string"),
            "answer": Value("string"),
            "context": Value("string"),
            "summary": Value("string"),
            "table": Value("string"),
            "image": HfImage(),
            "bbox": Sequence(Value("int32"), length=4),
        }
    )

    mock_ds = Dataset.from_list(
        [
            {
                "sample_id": "p1/fig1",
                "sub_figure": "a",
                "question": "Does it saturate?",
                "question_type": "Yes/No",
                "answer_type": "Yes/No",
                "answer": "Yes",
                "context": "Ctx",
                "summary": "Summary text",
                "table": "x,y;1,2",
                "image": PilImage.new("RGB", (10, 10)),
                "bbox": [0, 0, 100, 100],
            }
        ],
        features=features,
    )

    with patch("datasets.load_dataset", return_value=mock_ds):
        samples, valid, invalid = load_vqa_dataset(
            "train", source="hf", hf_repo_id="test/vqa"
        )

    assert valid == 1
    assert invalid == 0
    assert "messages" in samples[0]


# ── Tests for utility functions ───────────────────────────────────────


@pytest.mark.unit
def test_balance_yes_no() -> None:
    """Test Yes/No class balancing."""
    sample_yes = {
        "messages": [
            {"role": "user", "content": []},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Yes"}],
            },
        ]
    }
    sample_no = {
        "messages": [
            {"role": "user", "content": []},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "No"}],
            },
        ]
    }

    unbalanced = [sample_yes] * 5 + [sample_no] * 2
    balanced = balance_yes_no(unbalanced, seed=42)

    yes_count = sum(
        1 for s in balanced if s["messages"][1]["content"][0]["text"] == "Yes"
    )
    no_count = sum(
        1 for s in balanced if s["messages"][1]["content"][0]["text"] == "No"
    )

    assert yes_count == no_count
    assert len(balanced) > len(unbalanced)


@pytest.mark.unit
def test_balance_yes_no_single_class() -> None:
    """Test balancing when only one class is present (no-op)."""
    sample_yes = {
        "messages": [
            {"role": "user", "content": []},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Yes"}],
            },
        ]
    }
    samples = [sample_yes] * 3
    balanced = balance_yes_no(samples)
    assert len(balanced) == 3


@pytest.mark.unit
def test_balance_yes_no_other_types() -> None:
    """Test balancing with non-YN samples passed through."""
    yn = {
        "messages": [
            {"role": "user", "content": []},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Yes"}],
            },
        ]
    }
    other = {
        "messages": [
            {"role": "user", "content": []},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Some factoid answer"}],
            },
        ]
    }
    balanced = balance_yes_no([yn] * 3 + [other], seed=42)
    # Other type should still be present
    texts = [s["messages"][1]["content"][0]["text"] for s in balanced]
    assert "Some factoid answer" in texts
