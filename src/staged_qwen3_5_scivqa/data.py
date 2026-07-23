"""Data loading and dataset iteration for Sci-ImageMiner competition data."""

from __future__ import annotations

import json
import random
import warnings
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Literal

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from datasets import Image as HfImage
from PIL import Image
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

from staged_qwen3_5_scivqa.config import (
    COMPETITION_DATA_DIR,
    CONTEXT_WINDOW_SIZE,
    HF_SUMMARY_REPO,
    HF_TABLE_REPO,
    HF_VQA_REPO,
    PROMPTS,
)
from staged_qwen3_5_scivqa.context import get_paper_context
from staged_qwen3_5_scivqa.conversation import convert_to_conversation
from staged_qwen3_5_scivqa.preprocessing import clean_answer, clean_summary, clean_table

_SPLIT_MAP: dict[str, str] = {"train": "train", "dev": "validation", "test": "test"}


def _competition_images(category: str):
    """Iterate annotation JSON files in a competition data split.

    Yields (json_path, annotation_dict) for each valid annotation file.
    Skips content.json, .vscode, and files outside images/ directories.

    """
    case_dir = COMPETITION_DATA_DIR / category
    json_files = sorted(case_dir.rglob("*.json"))
    pbar = tqdm(json_files, desc=f"Scanning {category}")
    for json_file in pbar:
        fullpath = str(json_file)
        if (
            "content.json" in json_file.name
            or "images" not in fullpath
            or ".vscode" in fullpath
        ):
            continue
        pbar.set_description(json_file.name)
        img_path = json_file.with_suffix(".jpg")
        if not img_path.exists():
            warnings.warn(f"Image not found: {img_path}")
            continue
        with open(json_file) as f:
            yield json_file, json.load(f)


def _bbox_to_tuple(box: dict) -> tuple[int, int, int, int]:
    """Convert a bbox dict to a (left, top, width, height) tuple."""
    return (box["x"], box["y"], box["width"], box["height"])


# ── HF Dataset builders ────────────────────────────────────────────────


def _generate_vqa_rows(cat: str) -> Generator[dict, None, None]:
    """Yield cleaned VQA dataset rows one by one for Dataset.from_generator."""
    items = list(_competition_images(cat))
    hf_split = _SPLIT_MAP.get(cat, cat)

    # Wrap the file loop with tqdm so you see real-time progress during generator execution
    for json_file, data in tqdm(items, desc=f"Building '{hf_split}' split"):
        full_img = Image.open(json_file.with_suffix(".jpg").absolute())
        context = get_paper_context(json_file, window_size=CONTEXT_WINDOW_SIZE)
        bboxes = data.get("bbox", {})

        for sub_key, q_list in data.get("vqa", {}).items():
            box = bboxes.get(sub_key)
            if box is None:
                continue

            left, top, w, h = _bbox_to_tuple(box)
            sub_image = full_img.crop((left, top, left + w, top + h))
            summary = data.get("summarization", {}).get(sub_key, "") or ""
            table_text = data.get("data_extraction", {}).get(sub_key, "") or ""

            for q_obj in q_list:
                question = q_obj.get("question") or q_obj.get("questions", "")
                question_type = q_obj.get("question_type", "")
                answer_type = q_obj.get("answer_type", "")
                raw_answer = q_obj.get("answer", "")

                cleaned, is_valid = clean_answer(raw_answer, answer_type)
                if not is_valid:
                    continue

                yield {
                    "sample_id": data.get("sample_id", json_file.stem),
                    "sub_figure": sub_key,
                    "question": question,
                    "question_type": question_type,
                    "answer_type": answer_type,
                    "answer": cleaned,
                    "context": context,
                    "summary": summary,
                    "table": table_text,
                    "image": sub_image,
                    "bbox": [left, top, w, h],
                }


def build_vqa_dataset(
    categories: tuple[str, ...] = ("train", "dev", "test"),
    *,
    repo_id: str | None = None,
    token: str | None = None,
    push_to_hub: bool = True,
) -> DatasetDict:
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

    dataset_splits = {}
    for cat in categories:
        hf_split = _SPLIT_MAP.get(cat, cat)
        # Dataset.from_generator streams data directly into PyArrow shards on disk
        dataset_splits[hf_split] = Dataset.from_generator(
            _generate_vqa_rows,
            gen_kwargs={"cat": cat},
            features=features,
        )

    dd = DatasetDict(dataset_splits)

    if push_to_hub:
        rid = repo_id or HF_VQA_REPO
        print(f"Pushing dataset to Hugging Face Hub ({rid})...")
        dd.push_to_hub(rid, token=token, max_shard_size="500MB")

    return dd


def _generate_summary_rows(cat: str) -> Generator[dict, None, None]:
    """Yield cleaned summary dataset rows one by one for Dataset.from_generator."""
    items = list(_competition_images(cat))
    hf_split = _SPLIT_MAP.get(cat, cat)

    for json_file, data in tqdm(items, desc=f"Building '{hf_split}' summary split"):
        full_img = Image.open(json_file.with_suffix(".jpg").absolute())
        context = get_paper_context(json_file, window_size=CONTEXT_WINDOW_SIZE)
        bboxes = data.get("bbox", {})

        for sub_key, summary_text in data.get("summarization", {}).items():
            box = bboxes.get(sub_key)
            if box is None:
                continue

            left, top, w, h = _bbox_to_tuple(box)
            sub_image = full_img.crop((left, top, left + w, top + h))
            cleaned, is_valid = clean_summary(summary_text)
            if not is_valid:
                continue

            yield {
                "sample_id": data.get("sample_id", json_file.stem),
                "sub_figure": sub_key,
                "summary": cleaned,
                "context": context,
                "image": sub_image,
                "bbox": [left, top, w, h],
            }


def build_summary_dataset(
    categories: tuple[str, ...] = ("train", "dev", "test"),
    *,
    repo_id: str | None = None,
    token: str | None = None,
    push_to_hub: bool = True,
) -> DatasetDict:
    """Build a cleaned Summary DatasetDict from competition data and push to HF.

    Schema
    ------
    .. code-block:: python

        {
            "sample_id": str,
            "sub_figure": str,
            "summary": str,            # cleaned summary
            "context": str,            # paper context
            "image": PIL.Image,        # cropped sub-figure
            "bbox": [x, y, w, h],
        }
    """
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

    dataset_splits = {}
    for cat in categories:
        hf_split = _SPLIT_MAP.get(cat, cat)
        dataset_splits[hf_split] = Dataset.from_generator(
            _generate_summary_rows,
            gen_kwargs={"cat": cat},
            features=features,
        )

    dd = DatasetDict(dataset_splits)

    if push_to_hub:
        rid = repo_id or HF_SUMMARY_REPO
        tqdm.write(f"Pushing summary dataset to Hugging Face Hub ({rid})...")
        dd.push_to_hub(rid, token=token, max_shard_size="500MB")

    return dd


def _generate_table_rows(cat: str) -> Generator[dict, None, None]:
    """Yield cleaned table dataset rows one by one for Dataset.from_generator."""
    items = list(_competition_images(cat))
    hf_split = _SPLIT_MAP.get(cat, cat)

    for json_file, data in tqdm(items, desc=f"Building '{hf_split}' table split"):
        full_img = Image.open(json_file.with_suffix(".jpg").absolute())
        context = get_paper_context(json_file, window_size=CONTEXT_WINDOW_SIZE)
        bboxes = data.get("bbox", {})

        for sub_key, table_text in data.get("data_extraction", {}).items():
            box = bboxes.get(sub_key)
            if box is None:
                continue

            left, top, w, h = _bbox_to_tuple(box)
            sub_image = full_img.crop((left, top, left + w, top + h))
            cleaned, is_valid = clean_table(table_text)
            if not is_valid:
                continue

            yield {
                "sample_id": data.get("sample_id", json_file.stem),
                "sub_figure": sub_key,
                "table": cleaned,
                "context": context,
                "image": sub_image,
                "bbox": [left, top, w, h],
            }


def build_table_dataset(
    categories: tuple[str, ...] = ("train", "dev", "test"),
    *,
    repo_id: str | None = None,
    token: str | None = None,
    push_to_hub: bool = True,
) -> DatasetDict:
    """Build a cleaned Table DatasetDict from competition data and push to HF.

    Schema
    ------
    .. code-block:: python

        {
            "sample_id": str,
            "sub_figure": str,
            "table": str,              # cleaned, dense format (; rows, , cols)
            "context": str,            # paper context
            "image": PIL.Image,        # cropped sub-figure
            "bbox": [x, y, w, h],
        }
    """
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

    dataset_splits = {}
    for cat in categories:
        hf_split = _SPLIT_MAP.get(cat, cat)
        dataset_splits[hf_split] = Dataset.from_generator(
            _generate_table_rows,
            gen_kwargs={"cat": cat},
            features=features,
        )

    dd = DatasetDict(dataset_splits)

    if push_to_hub:
        rid = repo_id or HF_TABLE_REPO
        tqdm.write(f"Pushing table dataset to Hugging Face Hub ({rid})...")
        dd.push_to_hub(rid, token=token, max_shard_size="500MB")

    return dd


# ── HF Dataset loaders ────────────────────────────────────────────────


def load_vqa_from_hub(
    repo_id: str | None = None,
    split: str = "train",
    answer_types: list[str] | None = None,
) -> Dataset:
    """Load the cleaned VQA dataset from HuggingFace Hub.

    Args:
        repo_id: HF dataset repo. Defaults to :data:`HF_VQA_REPO`.
        split: Dataset split (``train``, ``validation``, ``test``).
        answer_types: Optional filter on ``answer_type`` column.

    Returns:
        A :class:`datasets.Dataset` with the same schema as
        :func:`build_vqa_dataset`.

    """
    from datasets import load_dataset as hf_load_dataset

    rid = repo_id or HF_VQA_REPO
    ds = hf_load_dataset(rid, split=split, token=None)
    if answer_types:
        ds = ds.filter(lambda x: x["answer_type"] in set(answer_types))
    return ds


def load_summary_from_hub(
    repo_id: str | None = None,
    split: str = "train",
) -> Dataset:
    """Load the cleaned Summary dataset from HuggingFace Hub."""
    from datasets import load_dataset as hf_load_dataset

    rid = repo_id or HF_SUMMARY_REPO
    return hf_load_dataset(rid, split=split, token=None)


def load_table_from_hub(
    repo_id: str | None = None,
    split: str = "train",
) -> Dataset:
    """Load the cleaned Table dataset from HuggingFace Hub."""
    from datasets import load_dataset as hf_load_dataset

    rid = repo_id or HF_TABLE_REPO
    return hf_load_dataset(rid, split=split, token=None)


# ── Conversation-formatted loaders (enhanced) ─────────────────────────


def load_vqa_dataset(
    category: str,
    answer_types: list[str] | None = None,
    *,
    source: Literal["competition", "hf"] = "competition",
    hf_repo_id: str | None = None,
) -> tuple[list[dict], int, int]:
    """Load VQA dataset for a given category (train/dev/test).

    Args:
        category: The data split name (e.g. ``"train"``, ``"dev"``,
            ``"test"``).
        answer_types: Filter to specific answer types. ``None`` means all.
        source: Where to read the data from.

            - ``"competition"``: read from the local competition data
              directory.
            - ``"hf"``: read from the HuggingFace Hub dataset.

        hf_repo_id: HF dataset repo (only used when *source* is ``"hf"``).
            Defaults to :data:`HF_VQA_REPO`.

    Returns:
        Tuple of (samples list, valid count, invalid count). Each sample is
        a Unsloth conversation dict with ``"messages"`` key.

    """
    if source == "hf":
        hf_split = _SPLIT_MAP.get(category, category)
        ds = load_vqa_from_hub(hf_repo_id, split=hf_split, answer_types=answer_types)
        samples: list[dict] = []
        for row in ds:
            answer_type = row["answer_type"]
            prompt = PROMPTS[answer_type].format(
                question=row["question"],
                question_type=row["question_type"],
                context=row["context"],
                summary=row["summary"] or "N/A",
                table=row["table"] or "N/A",
            )
            sample = convert_to_conversation(prompt, row["image"], row["answer"])
            samples.append(sample)
        return samples, len(samples), 0

    # ── competition source (original behaviour) ──
    case_dir = COMPETITION_DATA_DIR / category

    samples = []
    json_files = list(case_dir.rglob("*.json"))
    pbar = tqdm(json_files, desc="Processing Subfigures")

    valid_count = 0
    invalid_count = 0

    for json_file in pbar:
        fullpath = str(json_file)
        if (
            "content.json" in json_file.name
            or "images" not in fullpath
            or ".vscode" in fullpath
        ):
            continue

        pbar.set_description(f"Processing {json_file.name}")

        with open(json_file) as f:
            data = json.load(f)

        img_path = json_file.with_suffix(".jpg")
        if not img_path.exists():
            warnings.warn(f"Image not found: {img_path}")
            continue

        full_img = Image.open(img_path.absolute())
        context = get_paper_context(json_file, window_size=CONTEXT_WINDOW_SIZE)

        bboxes = data.get("bbox", {})

        for sub_key, q_list in data.get("vqa", {}).items():
            if sub_key not in bboxes:
                warnings.warn(f"Subfigure {sub_key} missing bbox in {json_file.name}")
                continue

            box = bboxes[sub_key]
            left = box["x"]
            top = box["y"]
            right = left + box["width"]
            bottom = top + box["height"]

            sub_image = full_img.crop((left, top, right, bottom))

            summary = data.get("summarization", {}).get(sub_key, None)
            table = data.get("data_extraction", {}).get(sub_key, None)

            for q_obj in q_list:
                question_text = q_obj.get("question") or q_obj.get("questions")
                question_type = q_obj.get("question_type", "")
                answer_type = q_obj.get("answer_type", "")

                if answer_types and answer_type not in answer_types:
                    continue

                if answer_type not in PROMPTS:
                    continue

                human_prompt = PROMPTS[answer_type].format(
                    question=question_text,
                    question_type=question_type,
                    context=context,
                    summary=summary if summary is not None else "N/A",
                    table=table if table is not None else "N/A",
                )

                raw_response = q_obj.get("answer", "")

                cleaned_response, is_valid = clean_answer(
                    raw_answer=raw_response,
                    expected_type=answer_type,
                )

                if not is_valid:
                    invalid_count += 1
                    continue

                valid_count += 1

                sample = convert_to_conversation(
                    human_prompt,
                    sub_image,
                    cleaned_response,
                )
                samples.append(sample)

    return samples, valid_count, invalid_count


def load_summary_dataset(
    category: str,
    *,
    source: Literal["competition", "hf"] = "competition",
    hf_repo_id: str | None = None,
) -> tuple[list[dict], int, int]:
    """Load summarization dataset for a given category.

    Args:
        category: The data split name.
        source: ``"competition"`` or ``"hf"``.
        hf_repo_id: HF dataset repo (used when *source* is ``"hf"``).

    Returns:
        Tuple of (samples, valid_count, invalid_count).

    """
    if source == "hf":
        hf_split = _SPLIT_MAP.get(category, category)
        ds = load_summary_from_hub(hf_repo_id, split=hf_split)
        samples = []
        for row in ds:
            prompt = PROMPTS.get("Summary", "").format(context=row["context"])
            sample = convert_to_conversation(prompt, row["image"], row["summary"])
            samples.append(sample)
        return samples, len(samples), 0

    # ── competition source (original behaviour) ──
    case_dir = COMPETITION_DATA_DIR / category

    samples = []
    json_files = list(case_dir.rglob("*.json"))
    pbar = tqdm(json_files, desc="Processing Summaries")

    valid_count = 0
    invalid_count = 0

    for json_file in pbar:
        fullpath = str(json_file)
        if (
            "content.json" in json_file.name
            or "images" not in fullpath
            or ".vscode" in fullpath
        ):
            continue

        pbar.set_description(f"Processing {json_file.name}")

        with open(json_file) as f:
            data = json.load(f)

        img_path = json_file.with_suffix(".jpg")
        if not img_path.exists():
            continue

        full_img = Image.open(img_path.absolute())
        context = get_paper_context(json_file, window_size=CONTEXT_WINDOW_SIZE)

        bboxes = data.get("bbox", {})

        for sub_key, summary_text in data.get("summarization", {}).items():
            if sub_key not in bboxes:
                continue

            box = bboxes[sub_key]
            left = box["x"]
            top = box["y"]
            right = left + box["width"]
            bottom = top + box["height"]

            sub_image = full_img.crop((left, top, right, bottom))

            cleaned_summary, is_valid = clean_summary(summary_text)
            if not is_valid:
                invalid_count += 1
                continue

            valid_count += 1

            prompt = PROMPTS.get("Summary", "").format(context=context)
            sample = convert_to_conversation(prompt, sub_image, cleaned_summary)
            samples.append(sample)

    return samples, valid_count, invalid_count


def load_table_dataset(
    category: str,
    *,
    source: Literal["competition", "hf"] = "competition",
    hf_repo_id: str | None = None,
) -> tuple[list[dict], int, int]:
    """Load table extraction dataset for a given category.

    Args:
        category: The data split name.
        source: ``"competition"`` or ``"hf"``.
        hf_repo_id: HF dataset repo (used when *source* is ``"hf"``).

    Returns:
        Tuple of (samples, valid_count, invalid_count).

    """
    if source == "hf":
        hf_split = _SPLIT_MAP.get(category, category)
        ds = load_table_from_hub(hf_repo_id, split=hf_split)
        samples = []
        for row in ds:
            prompt = PROMPTS.get("Table", "").format(context=row["context"])
            sample = convert_to_conversation(prompt, row["image"], row["table"])
            samples.append(sample)
        return samples, len(samples), 0

    # ── competition source (original behaviour) ──
    case_dir = COMPETITION_DATA_DIR / category

    samples = []
    json_files = list(case_dir.rglob("*.json"))
    pbar = tqdm(json_files, desc="Processing Tables")

    valid_count = 0
    invalid_count = 0

    for json_file in pbar:
        fullpath = str(json_file)
        if (
            "content.json" in json_file.name
            or "images" not in fullpath
            or ".vscode" in fullpath
        ):
            continue

        pbar.set_description(f"Processing {json_file.name}")

        with open(json_file) as f:
            data = json.load(f)

        img_path = json_file.with_suffix(".jpg")
        if not img_path.exists():
            continue

        full_img = Image.open(img_path.absolute())
        context = get_paper_context(json_file, window_size=CONTEXT_WINDOW_SIZE)

        bboxes = data.get("bbox", {})

        for sub_key, table_text in data.get("data_extraction", {}).items():
            if sub_key not in bboxes:
                continue

            box = bboxes[sub_key]
            left = box["x"]
            top = box["y"]
            right = left + box["width"]
            bottom = top + box["height"]

            sub_image = full_img.crop((left, top, right, bottom))

            cleaned_table, is_valid = clean_table(table_text)
            if not is_valid:
                invalid_count += 1
                continue

            valid_count += 1

            prompt = PROMPTS.get("Table", "").format(context=context)
            sample = convert_to_conversation(prompt, sub_image, cleaned_table)
            samples.append(sample)

    return samples, valid_count, invalid_count


def load_test_dataset(
    category: str,
    summary_cache: dict | None = None,
    extraction_cache: dict | None = None,
    answer_types: list[str] | None = None,
    *,
    source: Literal["competition", "hf"] = "competition",
    hf_repo_id: str | None = None,
) -> list[dict]:
    """Load test dataset for inference with cached summary/table evidence.

    When *source* is ``"hf"``, the VQA questions are loaded from the HF
    dataset while *summary_cache* and *extraction_cache* (predicted
    summaries / tables from upstream pipeline stages) are still merged in
    from the caller-supplied dicts.

    Args:
        category: The data split name (typically ``"test"``).
        summary_cache: Pre-generated summary cache dict
            (``{sample_id: {sub_fig: summary}}``).
        extraction_cache: Pre-generated table cache dict.
        answer_types: Filter to specific answer types. ``None`` means all.
        source: ``"competition"`` or ``"hf"``.
        hf_repo_id: HF dataset repo (used when *source* is ``"hf"``).

    Returns:
        List of inference conversation dicts with ``"messages"`` and
        ``"meta"`` keys.

    """
    if summary_cache is None:
        summary_cache = {}
    if extraction_cache is None:
        extraction_cache = {}

    if source == "hf":
        hf_split = _SPLIT_MAP.get(category, category)
        ds = load_vqa_from_hub(hf_repo_id, split=hf_split, answer_types=answer_types)
        samples = []
        for row in ds:
            sid = row["sample_id"]
            sub_key = row["sub_figure"]
            answer_type = row["answer_type"]
            prompt = PROMPTS[answer_type].format(
                question=row["question"],
                question_type=row["question_type"],
                context=row["context"],
                summary=summary_cache.get(sid, {}).get(sub_key, "N/A"),
                table=extraction_cache.get(sid, {}).get(sub_key, "N/A"),
            )
            from staged_qwen3_5_scivqa.conversation import (
                convert_to_inference_conversation,
            )

            sample = convert_to_inference_conversation(
                prompt,
                row["image"],
                sample_id=sid,
                sub_fig=sub_key,
                question_type=row["question_type"],
                answer_type=answer_type,
                question=row["question"],
            )
            samples.append(sample)
        return samples

    # ── competition source (original behaviour) ──
    case_dir = COMPETITION_DATA_DIR / category
    samples = []
    json_files = list(case_dir.rglob("*.json"))

    for json_file in json_files:
        fullpath = str(json_file)
        if (
            "content.json" in json_file.name
            or "images" not in fullpath
            or ".vscode" in fullpath
        ):
            continue

        with open(json_file) as f:
            data = json.load(f)

        sample_id = data.get("sample_id", "")
        img_path = json_file.with_suffix(".jpg")
        if not img_path.exists():
            continue

        full_img = Image.open(img_path.absolute())
        context = get_paper_context(json_file, window_size=CONTEXT_WINDOW_SIZE)

        bboxes = data.get("bbox", {})

        for sub_key, q_list in data.get("vqa", {}).items():
            if sub_key not in bboxes:
                continue

            box = bboxes[sub_key]
            left = box["x"]
            top = box["y"]
            right = left + box["width"]
            bottom = top + box["height"]

            sub_image = full_img.crop((left, top, right, bottom))

            summary = summary_cache.get(sample_id, {}).get(sub_key, "N/A")
            table = extraction_cache.get(sample_id, {}).get(sub_key, "N/A")

            for q_obj in q_list:
                question_text = q_obj.get("question") or q_obj.get("questions")
                question_type = q_obj.get("question_type", "")
                answer_type = q_obj.get("answer_type", "")

                if answer_types and answer_type not in answer_types:
                    continue

                if answer_type not in PROMPTS:
                    continue

                human_prompt = PROMPTS[answer_type].format(
                    question=question_text,
                    question_type=question_type,
                    context=context,
                    summary=summary,
                    table=table,
                )

                from staged_qwen3_5_scivqa.conversation import (
                    convert_to_inference_conversation,
                )

                sample = convert_to_inference_conversation(
                    human_prompt,
                    sub_image,
                    sample_id=sample_id,
                    sub_fig=sub_key,
                    question_type=question_type,
                    answer_type=answer_type,
                    question=question_text,
                )
                samples.append(sample)

    return samples


# ── Dataset post-processing utilities ──────────────────────────────────


def filter_by_token_budget(
    samples: list[dict],
    processor: Any,
    max_seq_length: int,
    max_new_tokens: int,
) -> list[dict]:
    """Filter conversation samples exceeding token budget limits.

    Uses :func:`~staged_qwen3_5_scivqa.analysis.calculate_token_stats` to
    compute per-sample token counts, then removes samples whose
    ``total_tokens > max_seq_length`` or
    ``assistant_tokens > max_new_tokens``.

    Args:
        samples: List of Unsloth conversation dicts.
        processor: A HuggingFace processor (tokenizer + image processor).
        max_seq_length: Maximum total sequence length (prompt + image +
            answer).
        max_new_tokens: Maximum allowed answer token count.

    Returns:
        Filtered list of conversation dicts.

    """
    from staged_qwen3_5_scivqa.analysis import calculate_token_stats

    df = calculate_token_stats(samples, processor)
    keep = df[
        (df["total_tokens"] <= max_seq_length)
        & (df["assistant_tokens"] <= max_new_tokens)
    ].index.tolist()
    return [samples[i] for i in keep]


def balance_yes_no(samples: list[dict], seed: int = 3407) -> list[dict]:
    """Upsample the minority Yes/No class to balance the dataset.

    Operates on conversation-formatted samples (output of
    :func:`load_vqa_dataset`).  Only samples where the assistant answer is
    ``"Yes"`` or ``"No"`` are considered for balancing; non-Y/N samples
    are passed through unchanged.

    Args:
        samples: List of Unsloth conversation dicts.
        seed: Random seed for reproducible upsampling.

    Returns:
        List with balanced Yes/No classes (both classes equal in size).

    """
    yes_samples = []
    no_samples = []
    other_samples = []

    for s in samples:
        answer = _extract_answer(s)
        if answer == "Yes":
            yes_samples.append(s)
        elif answer == "No":
            no_samples.append(s)
        else:
            other_samples.append(s)

    if not yes_samples or not no_samples:
        return samples

    if len(yes_samples) < len(no_samples):
        minority, majority = yes_samples, no_samples
    else:
        minority, majority = no_samples, yes_samples

    rng = random.Random(seed)
    upsampled = rng.choices(minority, k=len(majority))
    return upsampled + majority + other_samples


def _extract_answer(sample: dict) -> str:
    """Extract the assistant answer text from a conversation sample."""
    try:
        val = sample["messages"][1]["content"][0]["text"]
        return str(val)
    except (KeyError, IndexError, TypeError):
        return ""
