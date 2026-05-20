"""Inference utilities: generation, state tracking, and resumable runs."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


def run_inference(
    model,
    tokenizer,
    samples: list[dict],
    state_path: Path,
    generation_kwargs: dict[str, Any],
    checkpoint_every: int = 5,
) -> dict:
    """Run inference over samples with resumable state tracking.

    Args:
        model: The loaded model (FastVisionModel or FastLanguageModel).
        tokenizer: The tokenizer.
        samples: List of conversation dicts with "messages" and "meta".
        state_path: Path to the state JSON file for resumable runs.
        generation_kwargs: Dict of generation parameters.
        checkpoint_every: Save state every N samples.

    Returns:
        The final state dict mapping sample_id -> sub_fig -> list of answers.

    """
    state: dict = defaultdict(lambda: defaultdict(list))

    if state_path.exists():
        with open(state_path) as f:
            loaded = json.load(f)
        for k, v in loaded.items():
            for sub_k, sub_v in v.items():
                state[k][sub_k].extend(sub_v)

    processed = 0

    for sample in tqdm(samples, desc="Running inference"):
        meta = sample.get("meta", {})
        sample_id = meta.get("sample_id", "")
        sub_fig = meta.get("sub_fig", "")
        question_text = meta.get("question", "")

        if any(
            ans.get("question") == question_text for ans in state[sample_id][sub_fig]
        ):
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": sample["messages"][0]["content"][0]["text"],
                    },
                ],
            }
        ]

        image = sample["messages"][0]["content"][1]["image"]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

        generated_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )

        state[sample_id][sub_fig].append(
            {
                "question_type": meta.get("question_type", ""),
                "answer_type": meta.get("answer_type", ""),
                "question": question_text,
                "answer": generated_text.strip(),
            }
        )

        processed += 1

        if processed % checkpoint_every == 0:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=4)

    return dict(state)


def save_submission(
    state: dict,
    output_path: Path,
    key: str = "vqa",
) -> None:
    """Save state dict as a competition submission JSON file.

    Args:
        state: The state dict (sample_id -> sub_fig -> answers).
        output_path: Path to write the submission JSON.
        key: The top-level key in each submission record (e.g. "vqa").

    """
    submission = [{"sample_id": k, key: v} for k, v in state.items()]

    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)
