"""SMT pipeline: table parsing, declaration generation, and full reflection loop."""

import re

from PIL import Image

from staged_qwen3_5_scivqa.config import (
    EXAMPLES_PASS1A,
    EXAMPLES_PASS1B,
    EXAMPLES_PASS2,
    PREAMBLE,
    PROMPT_TEMPLATE_PASS1A,
    PROMPT_TEMPLATE_PASS1B,
    PROMPT_TEMPLATE_PASS2,
    PROMPT_TEMPLATE_PLANNING,
    PROMPT_TEMPLATE_REFLECTION,
)
from staged_qwen3_5_scivqa.smt.grammars import (
    SMT_CFG_PASS1A,
    build_dynamic_phase1b_cfg,
    build_dynamic_phase2_cfg,
)
from staged_qwen3_5_scivqa.smt.solver import validate_smt


def clean_duplicate_declarations(declarations_str: str) -> str:
    """Remove duplicate (declare-const ...) lines from a declarations string.

    Args:
        declarations_str: Raw declarations string.

    Returns:
        Deduplicated declarations string.

    """
    seen_declarations: set[str] = set()
    clean_lines: list[str] = []

    for line in declarations_str.split("\n"):
        match = re.search(
            r"\(declare-const\s+([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+)\)", line
        )
        if match:
            var_name = match.group(1)
            var_type = match.group(2)
            signature = f"{var_name}_{var_type}"

            if signature in seen_declarations:
                continue
            seen_declarations.add(signature)

        clean_lines.append(line)

    return "\n".join(clean_lines)


def deduplicate_anchors(anchors_str: str) -> str:
    """Deduplicate (f series x y) assertions, keeping the last value per (series, x).

    Args:
        anchors_str: Raw anchors string.

    Returns:
        Deduplicated anchors string.

    """
    anchors: dict[tuple[str, str], str] = {}
    lines = anchors_str.strip().split("\n")
    for line in lines:
        match = re.search(
            r"\(assert\s+\(=\s+\(f\s+([a-zA-Z0-9_]+)\s+([0-9.]+)\)\s+([0-9.]+)\)\)",
            line,
        )
        if match:
            series, x, y = match.groups()
            anchors[(series, x)] = y

    clean_lines = [f"(assert (= (f {s} {x}) {y}))" for (s, x), y in anchors.items()]
    return "\n".join(clean_lines)


def parse_table_deterministically(
    table_str: str,
) -> tuple[str, str, list[str]]:
    """Parse a Markdown/CSV table and generate SMT Pass 1A declarations + 1B anchors.

    Args:
        table_str: The table string (Markdown or dense format).

    Returns:
        Tuple of (declarations_str, anchors_str, valid_numbers_list).

    """
    lines = [line.strip() for line in table_str.strip().split("\n") if line.strip()]

    rows: list[list[str]] = []
    for line in lines:
        if "|" in line:
            cols = [c.strip() for c in line.split("|")]
            if line.startswith("|"):
                cols = cols[1:]
            if line.endswith("|"):
                cols = cols[:-1]
            if all(c == "" or "-" in c for c in cols):
                continue
            rows.append(cols)

    if not rows:
        for line in lines:
            rows.append([c.strip() for c in re.split(r"\t|,", line)])

    if not rows:
        return "", "", []

    headers = rows[0]
    data = rows[1:]
    y_cols = headers[1:]

    declarations: list[str] = []
    anchors: list[str] = []
    valid_numbers: list[str] = []

    seen_names: set[str] = set()
    col_to_clean_name: dict[int, str] = {}

    for i, y_col in enumerate(y_cols):
        base_name = re.sub(r"[^a-zA-Z0-9]", "", y_col)[:15]
        if not base_name:
            base_name = f"col{i}"
        elif base_name[0].isdigit():
            base_name = f"v_{base_name}"

        clean_name = base_name
        counter = 2
        while clean_name in seen_names:
            clean_name = f"{base_name}_{counter}"
            counter += 1

        seen_names.add(clean_name)
        col_to_clean_name[i] = clean_name

        ent = f"{clean_name}_entity"
        ser = f"{clean_name}_series"
        inc_bool = f"{clean_name}_is_inc_bool"
        dec_bool = f"{clean_name}_is_dec_bool"

        safe_y_col = re.sub(r"[^\x20-\x7E]", "", y_col).replace('"', "'")

        declarations.append(f"(declare-const {ent} Entity)")
        declarations.append(f"(declare-const {ser} Series)")
        declarations.append(f"(declare-const {inc_bool} Bool)")
        declarations.append(f"(declare-const {dec_bool} Bool)")

        anchors.append(f'(assert (= (name_of {ent}) "{safe_y_col}"))')
        anchors.append(f"(assert (attr {ser} {ent}))")

    declarations.extend(
        [
            "(declare-const max_val Real)",
            "(declare-const min_val Real)",
            "(declare-const target_val Real)",
            "(declare-const cond_bool Bool)",
            "(declare-const temp_real_1 Real)",
            "(declare-const temp_real_2 Real)",
            "(declare-const temp_real_3 Real)",
            "(declare-const temp_bool_1 Bool)",
            "(declare-const temp_bool_2 Bool)",
        ]
    )

    for i in range(1, len(y_cols) + 1):
        declarations.append(f"(declare-const rank{i}_entity Entity)")

    def parse_messy_number(val_str: str) -> float:
        clean_str = re.sub(r"[\s~<>,\xa0]", "", val_str)
        multiplier = 1.0

        if clean_str.lower().endswith("k"):
            multiplier = 1000.0
            clean_str = clean_str[:-1]
        elif clean_str.lower().endswith("m"):
            multiplier = 1000000.0
            clean_str = clean_str[:-1]
        elif clean_str.endswith("%"):
            multiplier = 0.01
            clean_str = clean_str[:-1]

        return float(clean_str) * multiplier

    for row_idx, row in enumerate(data):
        if not row or len(row) <= 1:
            continue
        try:
            x_val = parse_messy_number(row[0])
        except ValueError:
            x_val = float(row_idx)

        x_val_str = f"{x_val:.1f}" if x_val.is_integer() else str(x_val)
        valid_numbers.append(x_val_str)

        for i, y_str in enumerate(row[1:]):
            if i in col_to_clean_name:
                try:
                    y_val = parse_messy_number(y_str)
                    y_val_str = f"{y_val:.1f}" if y_val.is_integer() else str(y_val)
                    valid_numbers.append(y_val_str)

                    ser = f"{col_to_clean_name[i]}_series"
                    anchors.append(f"(assert (= (f {ser} {x_val_str}) {y_val_str}))")
                except ValueError:
                    continue

    return "\n".join(declarations), "\n".join(anchors), list(set(valid_numbers))


def generate_declarations(
    model,
    q_obj: dict,
    image: Image.Image,
    summary: str,
    max_retries: int = 3,
    verbose: bool = False,
    **gen_kwargs,
) -> tuple[str | None, str]:
    """Generate and validate SMT Pass 1A declarations and Pass 1B anchors.

    Args:
        model: The outlines model.
        q_obj: The question object dict.
        image: The cropped subfigure image.
        summary: The summary text.
        max_retries: Maximum retry attempts.
        verbose: Whether to print debug output.
        **gen_kwargs: Generation parameters.

    Returns:
        Tuple of (declarations_string | None, message).

    """
    from outlines.inputs import Chat

    question_text = q_obj.get("question") or q_obj.get("questions")
    question_type = q_obj.get("question_type", "")
    answer_type = q_obj.get("answer_type", "")

    first_pass_text = PROMPT_TEMPLATE_PASS1A.format(
        question=question_text,
        question_type=question_type,
        answer_type=answer_type,
        summary=summary,
        preamble=PREAMBLE,
        example=EXAMPLES_PASS1A.get(answer_type, ""),
    )

    prompt_pass1a = Chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": first_pass_text},
                ],
            }
        ]
    )

    declarations = ""
    pass1a_success = False

    for attempt in range(max_retries):
        declarations = model(prompt_pass1a, SMT_CFG_PASS1A, **gen_kwargs)
        declarations = declarations.strip()
        declarations = clean_duplicate_declarations(declarations)

        test_smt_pass1a = f""";; --- [PREAMBLE] ---
{PREAMBLE}
;; --- [PASS 1A: Declarations] Attempt {attempt + 1} ---
{declarations}
(check-sat)
"""

        pass1a_success, output = validate_smt(test_smt_pass1a)

        if verbose:
            print(
                f"[PASS 1A - Attempt {attempt + 1}]\n[Code]\n{test_smt_pass1a}\n[Output]\n{output}\n"
            )

        if pass1a_success:
            break

        output_lower = output.lower()
        if "already been defined" in output_lower:
            reflection_text = (
                f"The SMT solver rejected your declarations with this error:\n{output}\n\n"
                f"ERROR: You declared the same variable twice. Remove the duplicate declaration."
            )
        else:
            reflection_text = (
                f"The SMT solver rejected your declarations with this error:\n{output}\n\n"
                f"Please correct the syntax."
            )

        prompt_pass1a.add_assistant_message([{"type": "text", "text": declarations}])
        prompt_pass1a.add_user_message([{"type": "text", "text": reflection_text}])

    if not pass1a_success:
        return None, "Failed to generate valid Pass 1A declarations after retries."

    try:
        dynamic_cfg_pass1b = build_dynamic_phase1b_cfg(declarations)
    except Exception as e:
        return None, f"Failed to compile dynamic Phase 1B grammar: {e}"

    pass1b_text = PROMPT_TEMPLATE_PASS1B.format(
        summary=summary,
        declarations=declarations,
        example=EXAMPLES_PASS1B.get(answer_type, ""),
    )

    prompt_pass1b = Chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": pass1b_text},
                ],
            }
        ]
    )

    anchors = ""
    pass1b_success = False

    for attempt in range(max_retries):
        anchors = model(prompt_pass1b, dynamic_cfg_pass1b, **gen_kwargs)
        anchors = anchors.strip()
        anchors = deduplicate_anchors(anchors)

        test_smt_pass1b = f""";; --- [PREAMBLE] ---
{PREAMBLE}
;; --- [PASS 1A: Declarations] ---
{declarations}
;; --- [PASS 1B: Anchors] Attempt {attempt + 1} ---
{anchors}
(check-sat)
"""

        pass1b_success, output = validate_smt(test_smt_pass1b)

        if verbose:
            print(
                f"[PASS 1B - Attempt {attempt + 1}]\n[Code]\n{test_smt_pass1b}\n[Output]\n{output}\n"
            )

        if pass1b_success:
            break

        output_lower = output.lower()
        if "unsat" in output_lower:
            reflection_text = (
                "The solver returned 'unsat' (unsatisfiable). Your data mathematically contradicts itself. "
                "Did you assign two different y-values to the same Series at the same x-coordinate? "
            )
        else:
            reflection_text = (
                f"The SMT solver rejected your anchors with this error:\n{output}\n\n"
                f"Please correct the extraction syntax."
            )

        prompt_pass1b.add_assistant_message([{"type": "text", "text": anchors}])
        prompt_pass1b.add_user_message([{"type": "text", "text": reflection_text}])

    if not pass1b_success:
        return None, "Failed to generate valid Pass 1B anchors after retries."

    return declarations, anchors


def reflect(
    model,
    q_obj: dict,
    image: Image.Image,
    summary: str,
    table: str | None,
    max_retries: int = 3,
    verbose: bool = False,
    **gen_kwargs,
) -> tuple[str | None, str]:
    """Full SMT reflection pipeline: table parsing → planning → SMT generation.

    Args:
        model: The outlines model.
        q_obj: The question object dict.
        image: The cropped subfigure image.
        summary: The summary text.
        table: The pre-extracted table string (or None for free-form).
        max_retries: Maximum retry attempts for Pass 2.
        verbose: Whether to print debug output.
        **gen_kwargs: Generation parameters.

    Returns:
        Tuple of (smt_code_string | None, solver_output).

    """
    from outlines.inputs import Chat

    question_text = q_obj.get("question") or q_obj.get("questions")
    question_type = q_obj.get("question_type", "")
    answer_type = q_obj.get("answer_type", "")

    if table:
        declarations, anchors, valid_numbers = parse_table_deterministically(table)
    else:
        declarations_result, msg = generate_declarations(
            model,
            q_obj,
            image,
            summary,
            max_retries=max_retries,
            verbose=False,
            **gen_kwargs,
        )
        if declarations_result is None:
            return None, msg
        declarations = declarations_result
        anchors = ""
        valid_numbers = re.findall(r"-?\d+\.\d+", anchors)

    full_kb = f"{declarations}\n{anchors}"

    if verbose:
        print("[KNOWLEDGE BASE EXTRACTED]")
        print(full_kb)
        print("-" * 40)

    plan_text = PROMPT_TEMPLATE_PLANNING.format(
        question=question_text,
        question_type=question_type,
        answer_type=answer_type,
        summary=summary,
        declarations=declarations,
        anchors=anchors,
    )

    plan_prompt = Chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": plan_text},
                ],
            }
        ]
    )

    scratchpad_plan = model(plan_prompt, **gen_kwargs)

    if verbose:
        print("[INITIAL PLANNING SCRATCHPAD GENERATED]")
        print(scratchpad_plan)
        print("-" * 40)

    try:
        dynamic_cfg_pass2 = build_dynamic_phase2_cfg(
            full_kb, valid_numbers, answer_type=answer_type
        )
    except Exception as e:
        return None, f"Failed to compile dynamic Phase 2 grammar: {e}"

    for attempt in range(max_retries):
        second_pass_text = PROMPT_TEMPLATE_PASS2.format(
            question=question_text,
            question_type=question_type,
            answer_type=answer_type,
            summary=summary,
            preamble=PREAMBLE,
            declarations=declarations,
            anchors=anchors,
            scratchpad_plan=scratchpad_plan,
            example=EXAMPLES_PASS2.get(answer_type, ""),
        )

        prompt_pass2 = Chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": second_pass_text},
                    ],
                }
            ]
        )

        logic = model(prompt_pass2, dynamic_cfg_pass2, **gen_kwargs)

        final_smt = f""";; --- [PREAMBLE] ---
{PREAMBLE}
;; --- [KNOWLEDGE BASE] ---
{full_kb}
;; --- [PASS 2: Logic & Execution] Attempt {attempt + 1} ---
{logic}
"""

        success, output = validate_smt(final_smt)

        if success:
            has_bool = "AnsBool" in logic
            has_string = "AnsString" in logic
            has_real = "AnsReal" in logic

            if answer_type == "Yes/No" and not has_bool:
                success = False
                output = "LOGICAL ERROR: You forgot to assign your final conclusion to AnsBool. You must include an assertion like (= AnsBool ...)"
            elif answer_type in ["Factoid", "Paragraph"] and not has_string:
                success = False
                output = "LOGICAL ERROR: You forgot to assign your final conclusion to AnsString. Use an (ite ...) statement to generate the text based on your logic."
            elif not (has_bool or has_string or has_real):
                success = False
                output = "LOGICAL ERROR: You forgot to assign your final conclusion to AnsString or AnsReal or AnsBool."

        if verbose:
            print(
                f"[PASS 2 - Attempt {attempt + 1}]\n[Code]\n{final_smt}\n[Output]\n{output}\n"
            )

        if success:
            return final_smt, output

        if attempt == max_retries - 1:
            break

        reflection_text = PROMPT_TEMPLATE_REFLECTION.format(
            question=question_text,
            question_type=question_type,
            answer_type=answer_type,
            summary=summary,
            declarations=declarations,
            anchors=anchors,
            previous_plan=scratchpad_plan,
            generated_code=logic,
            feedback=output,
        )

        reflection_prompt = Chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": reflection_text},
                    ],
                }
            ]
        )

        scratchpad_plan = model(reflection_prompt, **gen_kwargs)

        if verbose:
            print(
                f"[REFORMULATED PLANNING SCRATCHPAD - Preparing for Attempt {attempt + 2}]"
            )
            print(scratchpad_plan)
            print("-" * 40)

    return None, "Failed to reach a valid logical formulation in Pass 2 after retries."
