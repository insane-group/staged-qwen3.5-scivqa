"""Answer reflection/rewriting using SMT solver output."""

import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from staged_qwen3_5_scivqa.config import (
    ENABLE_THINKING,
    PROMPT_REWRITE,
    REFLECTION_MAX_NEW_TOKENS,
    REFLECTION_MIN_P,
    REFLECTION_REPETITION_PENALTY,
    REFLECTION_TEMPERATURE,
    REFLECTION_TOP_K,
    REFLECTION_TOP_P,
)


def reflect_answers(
    model,
    tokenizer,
    initial_state: dict,
    smt_data: dict,
    reflection_state_path: Path,
    final_submission_path: Path,
) -> dict:
    """Rewrite initial VQA answers using SMT solver output.

    Args:
        model: The loaded FastLanguageModel.
        tokenizer: The tokenizer.
        initial_state: The initial finetuning state dict.
        smt_data: The SMT state dict (sample_id -> sub_fig -> question -> {code, output}).
        reflection_state_path: Path to save the reflection state.
        final_submission_path: Path to save the final submission JSON.

    Returns:
        The reflected state dict.

    """
    reflected_state: dict = defaultdict(lambda: defaultdict(list))

    if reflection_state_path.exists():
        with open(reflection_state_path) as f:
            loaded_state = json.load(f)
        for k, v in loaded_state.items():
            for sub_k, sub_v in v.items():
                reflected_state[k][sub_k].extend(sub_v)

    for sample_id, sub_figs in tqdm(
        initial_state.items(), desc="Running Code Reflection"
    ):
        if sample_id not in reflected_state:
            reflected_state[sample_id] = {}

        for sub_fig, q_list in sub_figs.items():
            if sub_fig not in reflected_state[sample_id]:
                reflected_state[sample_id][sub_fig] = []

            for q_obj in q_list:
                question_text = q_obj.get("question", "")
                question_type = q_obj.get("question_type", "")
                answer_type = q_obj.get("answer_type", "")
                initial_answer = q_obj.get("answer", "")

                if not initial_answer:
                    warnings.warn(
                        f"No initial answer for {sample_id} | Sub: {sub_fig} | "
                        f"Question: '{question_text}'. Skipping reflection."
                    )
                    reflected_state[sample_id][sub_fig].append(
                        {
                            "question_type": question_type,
                            "answer_type": answer_type,
                            "question": question_text,
                            "answer": initial_answer,
                        }
                    )
                    continue

                if any(
                    ans.get("question") == question_text
                    for ans in reflected_state[sample_id][sub_fig]
                ):
                    continue

                sub_fig_data = smt_data.get(sample_id, {}).get(sub_fig, {})
                smt_entry = sub_fig_data.get(question_text, {})

                code = smt_entry.get("code")
                output = smt_entry.get("output")

                if code is None:
                    final_answer = initial_answer
                else:
                    rewrite_prompt = PROMPT_REWRITE.format(
                        question_type=question_type,
                        question=question_text,
                        answer_type=answer_type,
                        answer_cache=initial_answer,
                        code=code,
                        output=output if output else "N/A",
                    )

                    messages = [{"role": "user", "content": rewrite_prompt}]

                    input_text = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        enable_thinking=ENABLE_THINKING,
                    )

                    inputs = tokenizer(text=input_text, return_tensors="pt").to("cuda")

                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=REFLECTION_MAX_NEW_TOKENS,
                        use_cache=True,
                        temperature=REFLECTION_TEMPERATURE,
                        min_p=REFLECTION_MIN_P,
                        top_p=REFLECTION_TOP_P,
                        top_k=REFLECTION_TOP_K,
                        repetition_penalty=REFLECTION_REPETITION_PENALTY,
                    )

                    raw_generated_text = tokenizer.decode(
                        output_ids[0][inputs["input_ids"].shape[-1] :],
                        skip_special_tokens=True,
                    )

                    match = re.search(
                        r"<ANSWER>(.*?)</ANSWER>",
                        raw_generated_text,
                        re.DOTALL | re.IGNORECASE,
                    )

                    parsed_answer = ""
                    hit_max_tokens = True
                    if match:
                        parsed_answer = match.group(1).strip()
                        hit_max_tokens = False

                    is_empty = len(parsed_answer) == 0
                    is_rambling = len(parsed_answer) > (2 * len(initial_answer))
                    is_too_short = answer_type not in {"Factoid", "Yes/No"} and len(
                        parsed_answer
                    ) < (0.4 * len(initial_answer))

                    if is_empty or hit_max_tokens or is_rambling or is_too_short:
                        reason = (
                            "Empty"
                            if is_empty
                            else (
                                "Hit Max/No Marker"
                                if hit_max_tokens
                                else ("Rambling" if is_rambling else "Too Short")
                            )
                        )
                        warnings.warn(
                            f"Fallback triggered for {sample_id} | Sub: {sub_fig} | "
                            f"Reason: {reason}. Reverting to initial answer."
                        )
                        final_answer = initial_answer
                    else:
                        final_answer = parsed_answer

                reflected_state[sample_id][sub_fig].append(
                    {
                        "question_type": question_type,
                        "answer_type": answer_type,
                        "question": question_text,
                        "answer": final_answer,
                    }
                )

                with open(reflection_state_path, "w") as f:
                    json.dump(reflected_state, f)

    final_submission = [{"sample_id": k, "vqa": v} for k, v in reflected_state.items()]

    with open(final_submission_path, "w") as f:
        json.dump(final_submission, f, indent=2)

    return dict(reflected_state)
