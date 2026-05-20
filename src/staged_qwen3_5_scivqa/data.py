"""Data loading and dataset iteration for Sci-ImageMiner competition data."""

import json
import warnings

from PIL import Image
from tqdm.auto import tqdm

from staged_qwen3_5_scivqa.config import (
    COMPETITION_DATA_DIR,
    CONTEXT_WINDOW_SIZE,
    PROMPTS,
)
from staged_qwen3_5_scivqa.context import get_paper_context
from staged_qwen3_5_scivqa.conversation import convert_to_conversation
from staged_qwen3_5_scivqa.preprocessing import clean_answer


def load_vqa_dataset(
    category: str,
    answer_types: list[str] | None = None,
) -> tuple[list[dict], int, int]:
    """Load VQA dataset for a given category (train/dev/test).

    Args:
        category: The data split name (e.g. "train", "dev", "test").
        answer_types: Filter to specific answer types. None means all VQA types.

    Returns:
        Tuple of (samples list, valid count, invalid count).

    """
    case_dir = COMPETITION_DATA_DIR / category

    samples: list[dict] = []
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
                    raw_answer=raw_response, expected_type=answer_type
                )

                if not is_valid:
                    invalid_count += 1
                    continue

                valid_count += 1

                sample = convert_to_conversation(
                    human_prompt, sub_image, cleaned_response
                )
                samples.append(sample)

    return samples, valid_count, invalid_count


def load_summary_dataset(category: str) -> tuple[list[dict], int, int]:
    """Load summarization dataset for a given category.

    Args:
        category: The data split name.

    Returns:
        Tuple of (samples list, valid count, invalid count).

    """
    from staged_qwen3_5_scivqa.preprocessing import clean_summary

    case_dir = COMPETITION_DATA_DIR / category

    samples: list[dict] = []
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


def load_table_dataset(category: str) -> tuple[list[dict], int, int]:
    """Load table extraction dataset for a given category.

    Args:
        category: The data split name.

    Returns:
        Tuple of (samples list, valid count, invalid count).

    """
    from staged_qwen3_5_scivqa.preprocessing import clean_table

    case_dir = COMPETITION_DATA_DIR / category

    samples: list[dict] = []
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
) -> list[dict]:
    """Load test dataset for inference with cached summary/table evidence.

    Args:
        category: The data split name (typically "test").
        summary_cache: Pre-generated summary cache dict.
        extraction_cache: Pre-generated table extraction cache dict.
        answer_types: Filter to specific answer types. None means all.

    Returns:
        List of inference conversation dicts with "messages" and "meta" keys.

    """
    if summary_cache is None:
        summary_cache = {}
    if extraction_cache is None:
        extraction_cache = {}

    case_dir = COMPETITION_DATA_DIR / category
    samples: list[dict] = []
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
