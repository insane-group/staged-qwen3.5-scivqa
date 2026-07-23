"""SMT pipeline runner: model loading and full SMT execution loop."""

import json
from pathlib import Path

from staged_qwen3_5_scivqa.config import (
    COMPETITION_DATA_DIR,
)


def load_smt_model(
    model_id: str = "unsloth/Qwen3.5-9B",
    max_new_tokens: int = 2048,
) -> object:
    """Load a text-only LLM wrapped by outlines for grammar-constrained decoding.

    Args:
        model_id: HuggingFace model identifier.
        max_new_tokens: Maximum tokens to generate (used for model config).

    Returns:
        An outlines-wrapped model ready for CFG-constrained generation.

    """
    import outlines
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = (
        "{% set enable_thinking = false %}\n" + tokenizer.chat_template
    )

    lm = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
    )

    return outlines.from_transformers(lm, tokenizer)  # type: ignore[arg-type]


def generate_smt_gen_kwargs(
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 1.5,
    repetition_penalty: float = 1.0,
) -> dict[str, float | int]:
    """Build generation kwargs dict for SMT model inference.

    Returns a dict suitable for passing as **gen_kwargs to
    ``smt.pipeline.reflect`` or ``smt.pipeline.generate_declarations``.

    """
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "min_p": min_p,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }


def run_smt_pipeline(
    category: str,
    output_path: Path,
    summary_cache_path: Path | None = None,
    extraction_cache_path: Path | None = None,
    model_id: str = "unsloth/Qwen3.5-9B",
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 1.5,
    repetition_penalty: float = 1.0,
    max_retries: int = 3,
    verbose: bool = False,
) -> dict:
    """Run the full SMT-LIB grammar-constrained decoding pipeline.

    Iterates over all VQA questions in the competition data, runs each through
    the SMT reflection pipeline, and saves results as a state JSON.

    Args:
        category: Data split name (train/dev/test).
        output_path: Path to save the SMT state JSON.
        summary_cache_path: Path to summary state JSON (from summarization stage).
        extraction_cache_path: Path to table extraction state JSON.
        model_id: HuggingFace model identifier.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling top_p.
        top_k: Top-k sampling.
        min_p: Minimum probability threshold.
        presence_penalty: Presence penalty.
        repetition_penalty: Repetition penalty.
        max_retries: Maximum retry attempts per question.
        verbose: Whether to print debug output.

    Returns:
        The SMT state dict mapping sample_id -> sub_fig -> question -> {code, output}.

    """
    from PIL import Image
    from tqdm.auto import tqdm

    from staged_qwen3_5_scivqa.smt.pipeline import reflect

    split_dir = COMPETITION_DATA_DIR / category
    if not split_dir.exists():
        raise FileNotFoundError(f"Competition data directory not found: {split_dir}")

    summary_cache = {}
    if summary_cache_path and summary_cache_path.exists():
        with open(summary_cache_path) as f:
            raw = json.load(f)
            if isinstance(raw, list):
                for item in raw:
                    sid = item.get("sample_id", "")
                    if sid:
                        summary_cache[sid] = item.get("summarization", {})
            elif isinstance(raw, dict):
                summary_cache = raw

    extraction_cache = {}
    if extraction_cache_path and extraction_cache_path.exists():
        with open(extraction_cache_path) as f:
            raw = json.load(f)
            if isinstance(raw, list):
                for item in raw:
                    sid = item.get("sample_id", "")
                    if sid:
                        extraction_cache[sid] = item.get("extraction", {})
            elif isinstance(raw, dict):
                extraction_cache = raw

    state = {}
    if output_path.exists():
        with open(output_path) as f:
            state = json.load(f)

    json_files = list(split_dir.rglob("*.json"))

    tasks = []
    for json_file in json_files:
        fullpath = str(json_file)
        if (
            "content.json" in json_file.name
            or "images" not in fullpath
            or ".vscode" in fullpath
        ):
            continue

        img_path = json_file.with_suffix(".jpg")
        if not img_path.exists():
            continue

        with open(json_file) as f:
            data = json.load(f)

        sample_id = data.get("sample_id")
        if not sample_id:
            continue

        if sample_id not in state:
            state[sample_id] = {}

        bboxes = data.get("bbox", {})
        vqa_data = data.get("vqa", {})

        for sub_key, q_list in vqa_data.items():
            if sub_key not in state[sample_id]:
                state[sample_id][sub_key] = {}

            table = extraction_cache.get(sample_id, {}).get(sub_key, None)
            if not table:
                continue

            if sub_key not in bboxes:
                continue

            for q_obj in q_list:
                question_text = q_obj.get("question") or q_obj.get("questions")
                if not question_text:
                    continue

                if question_text in state[sample_id][sub_key]:
                    continue

                tasks.append(
                    {
                        "json_file": json_file,
                        "img_path": img_path,
                        "sample_id": sample_id,
                        "sub_key": sub_key,
                        "q_obj": q_obj,
                        "question_text": question_text,
                        "box": bboxes[sub_key],
                        "summary": summary_cache.get(sample_id, {}).get(sub_key, ""),
                        "table": table,
                    }
                )

    model = load_smt_model(model_id, max_new_tokens)

    gen_kwargs = generate_smt_gen_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        min_p=min_p,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )

    current_img_path: Path | None = None
    full_img: Image.Image | None = None
    processed_cnt = 0

    pbar = tqdm(tasks, desc=f"Processing {category} split")

    for task in pbar:
        sample_id = task["sample_id"]
        sub_key = task["sub_key"]

        pbar.set_description(f"SMT: {sample_id} | Sub: {sub_key}")

        if current_img_path != task["img_path"]:
            current_img_path = task["img_path"]
            full_img = Image.open(current_img_path.absolute())

        box = task["box"]
        left, top = box["x"], box["y"]
        right, bottom = left + box["width"], top + box["height"]
        if full_img is None:
            continue
        crop = full_img.crop((left, top, right, bottom))

        smt_code, solver_output = reflect(
            model=model,
            q_obj=task["q_obj"],
            image=crop,
            summary=task["summary"],
            table=task["table"],
            max_retries=max_retries,
            verbose=verbose,
            do_sample=True,
            **gen_kwargs,
        )

        state[sample_id][sub_key][task["question_text"]] = {
            "code": smt_code,
            "output": solver_output,
        }

        processed_cnt += 1

        if processed_cnt % 5 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(state, f, indent=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(state, f, indent=4)

    return state
