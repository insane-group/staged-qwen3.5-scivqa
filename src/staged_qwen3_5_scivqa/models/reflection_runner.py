"""Reflection runner: model loading and answer reflection execution."""

from pathlib import Path

from staged_qwen3_5_scivqa.config import (
    REFLECTION_MAX_SEQUENCE_LENGTH,
)
from staged_qwen3_5_scivqa.smt.reflection import reflect_answers


def load_reflection_model(
    model_id: str = "unsloth/Qwen3.5-9B",
    max_seq_length: int = REFLECTION_MAX_SEQUENCE_LENGTH,
    load_in_4bit: bool = True,
) -> tuple[object, object]:
    """Load a FastLanguageModel for answer reflection.

    Args:
        model_id: HuggingFace model identifier.
        max_seq_length: Maximum sequence length for the model.
        load_in_4bit: Whether to load in 4-bit quantization.

    Returns:
        Tuple of (model, tokenizer).

    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def run_reflection(
    model_id: str = "unsloth/Qwen3.5-9B",
    initial_state_path: Path | None = None,
    smt_state_path: Path | None = None,
    reflection_state_path: Path | None = None,
    final_submission_path: Path | None = None,
    max_seq_length: int = REFLECTION_MAX_SEQUENCE_LENGTH,
    load_in_4bit: bool = True,
) -> dict:
    """Run answer reflection using SMT solver output.

    Loads the reflection model, reads initial VQA answers and SMT state,
    then rewrites answers based on SMT verification results.

    Args:
        model_id: HuggingFace model identifier.
        initial_state_path: Path to initial VQA state JSON.
        smt_state_path: Path to SMT state JSON.
        reflection_state_path: Path to save reflection state JSON.
        final_submission_path: Path to save final submission JSON.
        max_seq_length: Maximum sequence length for the model.
        load_in_4bit: Whether to load in 4-bit quantization.

    Returns:
        The reflected state dict.

    """
    if initial_state_path is None:
        raise ValueError("initial_state_path is required")
    if smt_state_path is None:
        raise ValueError("smt_state_path is required")
    if reflection_state_path is None:
        raise ValueError("reflection_state_path is required")
    if final_submission_path is None:
        raise ValueError("final_submission_path is required")

    model, tokenizer = load_reflection_model(
        model_id=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    return reflect_answers(
        model=model,
        tokenizer=tokenizer,
        initial_state=_load_json(initial_state_path),
        smt_data=_load_json(smt_state_path),
        reflection_state_path=reflection_state_path,
        final_submission_path=final_submission_path,
    )


def _load_json(path: Path) -> dict:
    """Load a JSON file and return as dict."""
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    with open(path) as f:
        data = f.read()
    import json

    result: dict = json.loads(data)
    if isinstance(result, list):
        converted: dict = {}
        for item in result:
            sid = item.get("sample_id", "")
            if sid:
                converted[sid] = item.get("vqa", {})
        return converted
    return result
