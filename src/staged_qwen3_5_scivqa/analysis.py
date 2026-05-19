"""Dataset analysis: token statistics, image dimensions, and quality reports."""

import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from staged_qwen3_5_scivqa.preprocessing import clean_answer


def analyze_dataset(
    case_dir: Path, tokenizer: PreTrainedTokenizer, max_samples: int | None = None
) -> pd.DataFrame:
    """Analyze dataset statistics: answer token counts and image dimensions.

    Args:
        case_dir: Path to the competition data split directory.
        tokenizer: A HuggingFace tokenizer for counting tokens.
        max_samples: Limit analysis to first N samples (None for all).

    Returns:
        DataFrame with columns: file, subfigure, answer_tokens,
        image_width, image_height, image_pixels.

    """
    stats = []
    json_files = list(case_dir.rglob("*.json"))

    if max_samples:
        json_files = json_files[:max_samples]

    pbar = tqdm(json_files, desc="Analyzing Data")

    for json_file in pbar:
        fullpath = str(json_file)
        if "images" not in fullpath or ".vscode" in fullpath:
            continue

        pbar.set_description(f"Processing {json_file.name}")

        with open(json_file) as f:
            data = json.load(f)

        bboxes = data.get("bbox", {})

        for sub_key, q_list in data.get("vqa", {}).items():
            if sub_key not in bboxes:
                continue

            box = bboxes[sub_key]
            width = box["width"]
            height = box["height"]

            for q_obj in q_list:
                gt_response = q_obj.get("answer", "")

                answer_tokens = len(
                    tokenizer.encode(gt_response, add_special_tokens=False)
                )

                stats.append(
                    {
                        "file": json_file.name,
                        "subfigure": sub_key,
                        "answer_tokens": answer_tokens,
                        "image_width": width,
                        "image_height": height,
                        "image_pixels": width * height,
                    }
                )

    return pd.DataFrame(stats)


def calculate_token_stats(
    samples: list[dict], processor, max_samples: int | None = None
) -> pd.DataFrame:
    """Calculate token length statistics for a dataset of conversation samples.

    Computes both assistant-only tokens (for MAX_NEW_TOKENS) and total
    sequence tokens (for max_length) including image encoding.

    Args:
        samples: List of conversation dicts with "messages" key.
        processor: A HuggingFace processor (tokenizer + image processor).
        max_samples: Limit analysis to first N samples (None for all).

    Returns:
        DataFrame with columns: assistant_tokens, total_tokens,
        image_width, image_height.

    """
    stats = []

    samples_to_process = samples[:max_samples] if max_samples else samples

    for sample in tqdm(samples_to_process, desc="Calculating token lengths"):
        messages = sample["messages"]

        image = messages[0]["content"][1]["image"]
        assistant_text = messages[1]["content"][0]["text"]

        assistant_tokens = len(
            processor.tokenizer.encode(assistant_text, add_special_tokens=False)
        )

        full_text = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )

        inputs = processor(
            text=full_text, images=image, add_special_tokens=False, return_tensors="pt"
        )
        total_tokens = inputs["input_ids"].shape[1]

        stats.append(
            {
                "assistant_tokens": assistant_tokens,
                "total_tokens": total_tokens,
                "image_width": image.width,
                "image_height": image.height,
            }
        )

    return pd.DataFrame(stats)


def report_and_fix_questions(
    case_dir: Path, target_answer_type: str = "Yes/No"
) -> dict:
    """Iterate through dataset, clean target answers, and report quality.

    Args:
        case_dir: Path to the competition data split directory.
        target_answer_type: The answer type to analyze (e.g. "Yes/No", "List").

    Returns:
        Dict with "summary" and "details" keys containing counts and records.

    """
    json_files = list(case_dir.rglob("*.json"))

    report_data: dict = {
        "summary": {"total_processed": 0, "perfect": 0, "fixed": 0, "unfixable": 0},
        "details": {
            "perfect": [],
            "fixed": [],
            "unfixable": [],
        },
    }

    for json_file in json_files:
        if "images" not in str(json_file) or ".vscode" in str(json_file):
            continue

        with open(json_file) as f:
            data = json.load(f)

        for sub_key, q_list in data.get("vqa", {}).items():
            for q_obj in q_list:
                answer_type = q_obj.get("answer_type", "")
                if answer_type != target_answer_type:
                    continue

                report_data["summary"]["total_processed"] += 1

                actual_answer = q_obj.get("answer", "")
                question_text = q_obj.get("question") or q_obj.get("questions", "")

                cleaned_answer, is_valid = clean_answer(
                    actual_answer, target_answer_type
                )

                record = {
                    "file": json_file.name,
                    "subfigure": sub_key,
                    "question": question_text,
                    "actual_answer": actual_answer,
                    "cleaned_answer": cleaned_answer,
                }

                if not is_valid:
                    report_data["summary"]["unfixable"] += 1
                    report_data["details"]["unfixable"].append(record)

                elif actual_answer == cleaned_answer:
                    report_data["summary"]["perfect"] += 1
                    report_data["details"]["perfect"].append(record)

                else:
                    report_data["summary"]["fixed"] += 1
                    report_data["details"]["fixed"].append(record)

    return report_data


def report_yes_no_balance(case_dir: Path) -> dict:
    """Report the Yes/No class balance in the dataset.

    Args:
        case_dir: Path to the competition data split directory.

    Returns:
        Dict with "summary" containing total, yes, and no counts.

    """
    json_files = list(case_dir.rglob("*.json"))

    report_data: dict = {
        "summary": {"total_processed": 0, "yes": 0, "no": 0},
        "details": {"yes": [], "no": []},
    }

    for json_file in json_files:
        if "images" not in str(json_file) or ".vscode" in str(json_file):
            continue

        with open(json_file) as f:
            data = json.load(f)

        for _sub_key, q_list in data.get("vqa", {}).items():
            for q_obj in q_list:
                answer_type = q_obj.get("answer_type", "")
                if answer_type != "Yes/No":
                    continue

                report_data["summary"]["total_processed"] += 1

                actual_answer = q_obj.get("answer", "")
                cleaned_answer, is_valid = clean_answer(actual_answer, "Yes/No")

                if cleaned_answer == "Yes":
                    report_data["summary"]["yes"] += 1
                elif cleaned_answer == "No":
                    report_data["summary"]["no"] += 1

    return report_data


def report_missing_data_extraction(case_dir: Path) -> tuple[int, int, float]:
    """Calculate the count and percentage of VQA questions missing table data.

    Args:
        case_dir: Path to the competition data split directory.

    Returns:
        Tuple of (total_questions, missing_count, percentage).

    """
    json_files = list(case_dir.rglob("*.json"))

    total_questions = 0
    missing_extraction_questions = 0

    for json_file in json_files:
        if "images" not in str(json_file) or ".vscode" in str(json_file):
            continue

        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        vqa_data = data.get("vqa", {})
        data_ext = data.get("data_extraction", {})

        for sub_key, q_list in vqa_data.items():
            num_questions = len(q_list)
            total_questions += num_questions

            if sub_key not in data_ext:
                missing_extraction_questions += num_questions

    if total_questions == 0:
        return 0, 0, 0.0

    percentage = (missing_extraction_questions / total_questions) * 100

    return total_questions, missing_extraction_questions, percentage
