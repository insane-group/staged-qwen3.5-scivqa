"""SMT-LIB grammar definitions and dynamic CFG builders."""

import re

from outlines.types import CFG

from staged_qwen3_5_scivqa.config import (
    SMT_LIB_GRAMMAR_PASS1A,
)

# Pre-compiled CFG for Pass 1A
SMT_CFG_PASS1A: CFG = CFG(SMT_LIB_GRAMMAR_PASS1A)


def build_dynamic_phase1b_cfg(declarations: str) -> CFG:
    """Build a dynamic CFG for Pass 1B based on Pass 1A declarations.

    Forces the model to extract anchors and assert names ONLY for the
    specific variables declared in Pass 1A.

    Args:
        declarations: The Pass 1A declarations string.

    Returns:
        A dynamically constructed CFG object.

    """
    entities = re.findall(r"\(declare-const\s+([a-zA-Z0-9_]+)\s+Entity\)", declarations)
    series = re.findall(r"\(declare-const\s+([a-zA-Z0-9_]+)\s+Series\)", declarations)
    reals = re.findall(r"\(declare-const\s+([a-zA-Z0-9_]+)\s+Real\)", declarations)

    entity_rule = (
        " | ".join([f'"{e}"' for e in entities]) if entities else '"DUMMY_ENTITY"'
    )
    series_rule = " | ".join([f'"{s}"' for s in series]) if series else '"DUMMY_SERIES"'
    real_rule = " | ".join([f'"{r}"' for r in reals]) if reals else '"DUMMY_REAL"'

    dynamic_grammar = rf"""
    ?start: script
    script: metadata_asserts data_anchors

    metadata_asserts: meta_assert*
    meta_assert: name_assert | attr_assert

    name_assert: "(assert (= (name_of " ENTITY_SYM ") " STRING_LIT "))\n"
    attr_assert: "(assert (attr " SERIES_SYM " " ENTITY_SYM "))\n"

    data_anchors: anchor_assert anchor_assert anchor_assert anchor_assert anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert? anchor_assert?

    anchor_assert: "(assert (= (f " SERIES_SYM " " coordinate_val ") " coordinate_val "))\n"
    coordinate_val: DECIMAL | LOGIC_VAR_REAL

    ENTITY_SYM: {entity_rule}
    SERIES_SYM: {series_rule}
    LOGIC_VAR_REAL: {real_rule}
    DECIMAL: /-?[0-9]+\.[0-9]+/

    STRING_LIT: /"[\x20-\x7E]*"/
    """
    return CFG(dynamic_grammar)


def build_dynamic_phase2_cfg(
    declarations: str,
    valid_numbers: list[str] | None = None,
    answer_type: str | None = None,
) -> CFG:
    """Build a dynamic CFG for Pass 2 based on declarations and answer type.

    Args:
        declarations: The combined Pass 1A + 1B declarations string.
        valid_numbers: List of valid numeric values from the table.
        answer_type: The VQA answer type ("Yes/No", "Factoid", "List", "Paragraph").

    Returns:
        A dynamically constructed CFG object.

    """
    if valid_numbers is None:
        valid_numbers = []

    entities = re.findall(r"\(declare-const\s+([a-zA-Z0-9_]+)\s+Entity\)", declarations)
    series = re.findall(r"\(declare-const\s+([a-zA-Z0-9_]+)\s+Series\)", declarations)
    bools = re.findall(r"\(declare-const\s+([a-zA-Z0-9_]+)\s+Bool\)", declarations)
    reals = re.findall(r"\(declare-const\s+([a-zA-Z0-9_]+)\s+Real\)", declarations)

    entity_rule = (
        " | ".join([f'"{e}"' for e in entities]) if entities else '"DUMMY_ENTITY"'
    )
    series_rule = " | ".join([f'"{s}"' for s in series]) if series else '"DUMMY_SERIES"'
    bool_rule = " | ".join([f'"{b}"' for b in bools]) if bools else '"DUMMY_BOOL"'
    real_rule = " | ".join([f'"{r}"' for r in reals]) if reals else '"DUMMY_REAL"'

    allowed_nums = set(
        valid_numbers
        + ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "10.0", "100.0", "-1.0"]
    )
    num_rule = " | ".join([f'"{n}"' for n in allowed_nums])

    logic_seq_rule = "logic_assert " + " ".join(["logic_assert?"] * 32)

    if answer_type == "Yes/No":
        script_rule = "script: logic_sequence final_bool_assert check_sat_cmd get_value_cmd exit_cmd"
        final_assert_rule = (
            'final_bool_assert: "(assert (= AnsBool " calculated_bool_term "))\\n"'
        )
    elif answer_type in ["Factoid", "Paragraph"]:
        script_rule = "script: logic_sequence final_string_assert check_sat_cmd get_value_cmd exit_cmd"
        final_assert_rule = 'final_string_assert: "(assert (= AnsString " calculated_string_term "))\\n"'
    else:
        script_rule = "script: logic_sequence final_answer_assert check_sat_cmd get_value_cmd exit_cmd"
        final_assert_rule = """
    final_answer_assert: "(assert (= AnsBool " calculated_bool_term "))\\n"
                       | "(assert (= AnsReal " calculated_real_term "))\\n"
                       | "(assert (= AnsString " calculated_string_term "))\\n"
        """

    dynamic_grammar = rf"""
    ?start: script

    calculated_string_term: string_preamble_call | string_expr
    calculated_bool_term: bool_preamble_call | bool_expr | LOGIC_VAR_BOOL
    calculated_real_term: real_preamble_call | real_expr | LOGIC_VAR_REAL

    {script_rule}

    logic_sequence: {logic_seq_rule}

    logic_assert: "(assert (= " LOGIC_VAR_BOOL " " bool_term "))\n"
                | "(assert (= " LOGIC_VAR_REAL " " real_term "))\n"
                | "(assert (= " ENTITY_SYM " " ENTITY_SYM "))\n"
                | "(assert " bool_term ")\n"

    {final_assert_rule}

    check_sat_cmd: "(check-sat)\n"

    get_value_cmd: "(get-value (" gv_list "))\n"
    gv_list: gv_item (" " gv_item)*
    gv_item: LOGIC_VAR_BOOL | LOGIC_VAR_REAL | "AnsBool" | "AnsReal" | "AnsString" | ENTITY_SYM | SERIES_SYM | string_preamble_call | bool_preamble_call | real_preamble_call | STRING_LIT

    exit_cmd: "(exit)\n"

    ?real_term: DECIMAL | LOGIC_VAR_REAL | real_preamble_call | real_expr | "epsilon" | "AnsReal"
    ?bool_term: "true" | "false" | LOGIC_VAR_BOOL | bool_preamble_call | bool_expr | "AnsBool"
    ?string_term: STRING_LIT | string_preamble_call | string_expr | "AnsString"

    real_expr: "(" REAL_OP " " real_term " " real_term ")"
             | "(- " real_term ")"
             | "(ite " bool_term " " real_term " " real_term ")"

    bool_expr: "(" BOOL_BIN_OP " " bool_term " " bool_term ")"
             | "(not " bool_term ")"
             | "(" COMP_OP " " real_term " " real_term ")"
             | "(= " real_term " " real_term ")"
             | "(= " bool_term " " bool_term ")"
             | "(= " string_term " " string_term ")"
             | "(= " ENTITY_SYM " " ENTITY_SYM ")"
             | "(distinct " ENTITY_SYM " " ENTITY_SYM ")"
             | "(ite " bool_term " " bool_term " " bool_term ")"

    string_expr: "(ite " bool_term " " string_term " " string_term ")"

    REAL_OP: "+" | "-" | "*" | "/"
    BOOL_BIN_OP: "and" | "or" | "=" | "=>"
    COMP_OP: ">" | "<" | ">=" | "<=" | "="

    real_preamble_call: "(f " SERIES_SYM " " real_term ")"
                      | "(diff " real_term " " real_term ")"

    bool_preamble_call: "(attr " SERIES_SYM " " ENTITY_SYM ")"
                      | "(is_gt " SERIES_SYM " " SERIES_SYM " " real_term ")"
                      | "(is_eq " SERIES_SYM " " SERIES_SYM " " real_term ")"
                      | "(is_inc " SERIES_SYM " " real_term " " real_term ")"
                      | "(is_dec " SERIES_SYM " " real_term " " real_term ")"
                      | "(is_const " SERIES_SYM " " real_term " " real_term ")"
                      | "(is_at_val " SERIES_SYM " " real_term " " real_term ")"
                      | "(is_peak " SERIES_SYM " " real_term " " real_term " " real_term ")"

    string_preamble_call: "(name_of " ENTITY_SYM ")"
                        | "(unit_of " SERIES_SYM ")"

    ENTITY_SYM: {entity_rule}
    SERIES_SYM: {series_rule}
    LOGIC_VAR_BOOL: {bool_rule}
    LOGIC_VAR_REAL: {real_rule}

    DECIMAL: {num_rule}
    STRING_LIT: /"[\x20-\x7E]*"/
    """
    return CFG(dynamic_grammar)
