# %% [markdown]
# # Qwen3.5 Image-to-SMT via Grammar-Constrained Decoding
#
# > Originally, adapted from [Qwen3_5_(0_8B)_Vision.ipynb](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(0_8B)_Vision.ipynb#scrollTo=gGFzmplrEy9I)
#
# ![Qwen3.5](https://qianwen-res.oss-accelerate.aliyuncs.com/logo_qwen3.5.png)
#
# Qwen3.5 features the following enhancement:
#
# - **Unified Vision-Language Foundation**: Early fusion training on multimodal tokens achieves cross-generational parity with Qwen3 and outperforms Qwen3-VL models across reasoning, coding, agents, and visual understanding benchmarks.
# - **Efficient Hybrid Architecture**: Gated Delta Networks combined with sparse Mixture-of-Experts deliver high-throughput inference with minimal latency and cost overhead.
# - **Scalable RL Generalization**: Reinforcement learning scaled across million-agent environments with progressively complex task distributions for robust real-world adaptability.
# - **Global Linguistic Coverage**: Expanded support to 201 languages and dialects, enabling inclusive, worldwide deployment with nuanced cultural and regional understanding.
# - **Next-Generation Training Infrastructure**: Near-100% multimodal training efficiency compared to text-only training and asynchronous RL frameworks supporting massive-scale agent scaffolds and environment orchestration.

# %%
import os
import subprocess
import tempfile
import json
from pathlib import Path
from outlines.types import CFG
from outlines.inputs import Chat
from PIL import Image
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import outlines
from tqdm.auto import tqdm

# %%
# MODEL_ID = "unsloth/Qwen3.5-35B-A3B"

MODEL_ID = "unsloth/Qwen3.5-9B"
MAX_NEW_TOKENS = 2048

# # https://unsloth.ai/docs/models/qwen3.5#recommended-settings
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 1.5  # Changed from 0.0
REPETITION_PENALTY = 1.0  # Changed from 1.1 to disable

BASE_DIR = Path.cwd().parent
CATEGORIES = ["train", "dev"]

COMPETITION_DATA_DIR = BASE_DIR / "ALD-E-ImageMiner" / "icdar2026-competition-data"

STATE_FILE = BASE_DIR / f"smt_{'_'.join(CATEGORIES)}_state.json"
SMT_FILE = BASE_DIR / f"smt_{'_'.join(CATEGORIES)}.json"

CVC5_PATH = Path.home() / "cvc5-Linux-x86_64-shared" / "bin" / "cvc5"

# %% [markdown]
# <a name="Data"></a>
# ### 🧪 Data Preparation
#
# To convert our Sci-ImageMiner VQA data into the format required by Qwen3.5 (specifically for use with Unsloth), we need to restructure the data into a "messages" format.
#
# The Qwen/Unsloth format expects a list of conversations where the image and the text prompt are bundled together in the user role, and the ground truth is in the assistant role, as follows:
#
# ```python
# [
#     { "role": "user",
#     "content": [{"type": "text",  "text": Q}, {"type": "image", "image": image} ]
#     },
#     { "role": "assistant",
#     "content": [{"type": "text",  "text": A} ]
#     },
# ]
# ```

# %%
# A stripped down EBNF conforming to strict guided decoding rules.
# 1. Comments are impossible.
# 2. Infinite loops are prevented by a forced exit lifecycle.
# 3. Type-safety is enforced lexically via suffixes/prefixes.
# 4. Hardcoding the preamble signatures
# 5. Prohibit the LLM from re-declaring preamble functions or using generic sorts for Entities/Series
# 6. Use a Phase-Locked Grammar. This forces the LLM to follow a one-way path: first declarations, then data anchors, then logic, then exit.
# 7. We limit underscores and character length to prevent 'series_series_series' loops
# 8. Prevent an infinite token loop by removing the * quantifier and physically cap the number of allowed repetitions.

# ==========================================
# PASS 1A: PURE DECLARATIONS ONLY
# ==========================================
SMT_LIB_GRAMMAR_PASS1A = r"""
?start: script

# 1. Limit the number of declarations to a safe maximum (e.g., 20)
# This prevents the model from rambling until it hits the token limit.
script: decl_line decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line? decl_line?

# 2. Ensure each line is an atomic, unbreakable unit
decl_line: entity_decl | series_decl | real_decl | bool_decl

entity_decl: "(declare-const " ENTITY_SYM " Entity)\n"
series_decl: "(declare-const " SERIES_SYM " Series)\n"
real_decl:   "(declare-const " LOGIC_VAR_REAL " Real)\n"
bool_decl:   "(declare-const " LOGIC_VAR_BOOL " Bool)\n"

ENTITY_SYM: /[a-zA-Z][a-zA-Z0-9]{0,24}_entity/
SERIES_SYM: /[a-zA-Z][a-zA-Z0-9]{0,24}_series/
LOGIC_VAR_BOOL: /[a-zA-Z][a-zA-Z0-9]{0,24}_(drop|inc|dec|const|stable|bool)/
LOGIC_VAR_REAL: /[a-zA-Z][a-zA-Z0-9]{0,24}_(val|max|min|coord|cycle)/
"""

SMT_CFG_PASS1A = CFG(SMT_LIB_GRAMMAR_PASS1A)

# %%
PROMPT_TEMPLATE_PLANNING = """
<image>

[SUMMARY]
{summary}

[METADATA]
Question Type: {question_type}
Answer Type: {answer_type}
Question: {question}

[KNOWLEDGE BASE (EXTRACTED DATA)]
{declarations}
{anchors}

[TASK]
Create a logical plan to answer the question using ONLY the provided Knowledge Base and Summary.

Follow this exact format:
1. ANALYSIS: Identify exactly what the question seeks.
2. DATA EVALUATION:
   - If 'List': Identify all relevant items and their order.
   - If 'Yes/No': Determine the truth value by comparing specific data points.
   - If 'Factoid/Paragraph': Locate the specific string or entity that directly answers the prompt.
3. SMT STRATEGY: List the variables and functions needed.
   - CRITICAL: Root variables (AnsBool, AnsString, AnsReal) CANNOT be assigned raw literals directly (e.g., no `(= AnsString "Answer")` or `(= AnsBool true)`). You MUST use a calculated term.
   - If Answer Type is 'Yes/No', you MUST assign the result to 'AnsBool' via a logical expression (e.g., `(> val1 val2)`) or a previously computed boolean variable.
   - If Answer Type is 'Factoid' or 'Paragraph', you MUST assign the result to 'AnsString' via a string expression (e.g., an `(ite ...)` statement) or a preamble call (e.g., `(name_of ...)`).
   - If Answer Type is 'List', you MUST assign the result to 'AnsBool', 'AnsString' or 'AnsReal' using calculated expressions.
   - FALLBACK RULE: If the required information is missing or undecidable from the KB/Summary, you MUST still generate a valid calculated SMT assignment:
     * For 'Yes/No': Assign `AnsBool` to a false expression (e.g., `(= AnsBool (= 1.0 0.0))`).
     * For 'List', 'Factoid', or 'Paragraph': Assign `AnsString` using a dummy conditional (e.g., `(= AnsString (ite true "Information not found in context" ""))`).
4. FINAL ANSWER: Provide the answer exactly as it should appear in the final output.

[PLANNING SCRATCHPAD]
"""

PROMPT_TEMPLATE_REFLECTION = """
<image>

[SUMMARY]
{summary}

[METADATA]
Question Type: {question_type}
Answer Type: {answer_type}
Question: {question}

[KNOWLEDGE BASE (EXTRACTED DATA)]
{declarations}
{anchors}

[PREVIOUS PLAN]
{previous_plan}

[GENERATED SMT-LIB CODE]
{generated_code}

[FEEDBACK / ERROR]
{feedback}

[TASK]
Your previous plan was translated into the SMT-LIB code above, but it failed validation.

Follow this exact format to recover:
1. ERROR ANALYSIS: Identify exactly why the code failed based on the feedback.
   - Was it a structural error? (e.g., missing the required `AnsBool` or `AnsString` assignment).
   - Did the solver return 'unsat'? (Look for contradictory data points, incorrect sorting, or faulty logic bindings).
   - Was there a syntax error? (Look for incorrect function arity, missing variables, or unmapped series).
2. REVISED SMT STRATEGY: State exactly which variables (e.g., AnsBool, AnsString, rankN_entity) must be assigned to fix the structural or logical error.
4. REVISED FINAL ANSWER: Provide the expected answer format.

[REVISED PLANNING SCRATCHPAD]
"""

# %%
PROMPT_TEMPLATE_PASS1A = """
<image>

[SUMMARY]
{summary}

[METADATA]
Question Type: {question_type}
Answer Type: {answer_type}
Question: {question}

[TASK]
Your goal is to build a "Knowledge Base" using the provided image and summary.
1. Declare all Entities and Series.
2. Pre-declare the logical variables (Bool/Real) you will need to perform reasoning in Pass 2.

[STRICT RULES]
- PURE DECLARATIONS ONLY: Do not perform any logic, and do NOT use `(assert ...)`.
- NO ANCHORS YET: Do NOT extract data points (f series x) or assert names/attributes here. You will do that in the next step.
- SCHEMA LOCK: You must declare any variable (e.g., max_val) here, or you won't be able to use it in Pass 2.
- SERIES: You MUST declare a separate Series variable (e.g., carbon_series, oxygen_series) for every individual data series present in the table. Notice the mandatory '_series' suffix!

[AVAILABLE SMT-LIB ENVIRONMENT]
{preamble}

[EXAMPLE]
{example}
"""

PROMPT_TEMPLATE_PASS1B = """
<image>

[SUMMARY]
{summary}

[TASK]
Using the variables declared in the Knowledge Base, your task is to:
1. Assert the names and attributes (`name_of`, `attr`) mapping Series to Entities.
2. Extract the exact numeric data points (`f`) for each Series.

[STRICT RULES]
- NO NEW DECLARATIONS: You must only use the exact variables provided below.
- DECIMALS ONLY: SMT-LIB requires explicit decimals for Real types (e.g., 0.0 instead of 0).
- ANCHORS: Extract representative anchors (at least 4) for the declared series.
- UNIQUE VALUES: Each Series can have only ONE value per x-coordinate. Do not assert (f s x y1) and (f s x y2) if y1 != y2.
- CONSISTENCY: Ensure your extracted points form a single, logical curve.

[AVAILABLE SMT-LIB ENVIRONMENT]
{preamble}

[KNOWLEDGE BASE (FROM PASS 1A)]
{declarations}

[EXAMPLE]
{example}
"""

# %%
EXAMPLES_PASS1A = {
    "Yes/No": "(declare-const s2p_entity Entity)\n(declare-const trace_series Series)\n(declare-const max_val Real)\n(declare-const threshold_met_bool Bool)",
    "Factoid": "(declare-const structural_region_entity Entity)\n(declare-const sidewall_entity Entity)\n(declare-const identified_issue_entity Entity)\n(declare-const sidewall_is_part_of_region_bool Bool)\n(declare-const issue_forms_on_sidewall_bool Bool)",
    "List": "(declare-const e_CF2_entity Entity)\n(declare-const e_CF_entity Entity)\n(declare-const e_C_entity Entity)\n(declare-const s_CF2_series Series)\n(declare-const s_CF_series Series)\n(declare-const s_C_series Series)\n(declare-const rank1_entity Entity)\n(declare-const rank2_entity Entity)\n(declare-const rank3_entity Entity)",
    "Paragraph": "(declare-const o1s_entity Entity)\n(declare-const o1s_series Series)\n(declare-const o_initial_drop_bool Bool)\n(declare-const o_steady_decrease_bool Bool)",
}

EXAMPLES_PASS1B = {
    "Yes/No": '(assert (= (name_of s2p_entity) "S2p"))\n(assert (attr trace_series s2p_entity))\n(assert (= (f trace_series 0.0) 0.0))\n(assert (= (f trace_series 10.0) 1.8))\n(assert (= (f trace_series 20.0) 1.6))\n(assert (= (f trace_series 30.0) 1.6))\n(assert (= (f trace_series 50.0) 1.9))\n(assert (= (f trace_series 75.0) 1.2))',
    "Factoid": '(assert (= (name_of structural_region_entity) "Multi-quantum well"))\n(assert (= (name_of sidewall_entity) "sidewall"))\n(assert (= (name_of identified_issue_entity) "Etch damage layer"))',
    "List": '(assert (= (name_of e_CF2_entity) "CF2"))\n(assert (= (name_of e_CF_entity) "CF"))\n(assert (= (name_of e_C_entity) "C"))\n(assert (attr s_CF2_series e_CF2_entity))\n(assert (attr s_CF_series e_CF_entity))\n(assert (attr s_C_series e_C_entity))\n(assert (= (f s_CF2_series 6.25) 5.75))\n(assert (= (f s_CF_series 6.25) 4.15))\n(assert (= (f s_C_series 6.25) 2.50))',
    "Paragraph": '(assert (= (name_of o1s_entity) "O1s"))\n(assert (attr o1s_series o1s_entity))\n(assert (= (f o1s_series 0.0) 16.0))\n(assert (= (f o1s_series 10.0) 10.5))\n(assert (= (f o1s_series 75.0) 7.5))',
}

# %%
PROMPT_TEMPLATE_PASS2 = """
<image>

[SUMMARY]
{summary}

[METADATA]
Question Type: {question_type}
Answer Type: {answer_type}
Question: {question}

[APPROVED LOGICAL PLAN]
{scratchpad_plan}

[TASK]
Translate your APPROVED LOGICAL PLAN into strict SMT-LIB code using ONLY the variables provided in the Knowledge Base.

[STRICT RULES]
- NO DECLARATIONS: Do not use 'declare-const'. Use only the variables provided in the Knowledge Base.
- DIRECT ASSERTIONS: Use the pre-declared Boolean/Real variables to represent your logic.
- FINITE SEARCH: Use (or ...) for maxima/minima iterating ONLY over the exact x-values asserted in the Knowledge Base. Do not hallucinate or guess data points.
- USE PREAMBLE: Use functions like 'is_dec', 'is_inc', and 'is_eq' to assign values to the pre-declared Booleans.
- LIST/RANKING: If 'List', prove the requested observations (trends, comparisons, or order). Use (ite ...) to construct a descriptive 'AnsString' containing the list of findings or use 'rankN_entity' only if ranking is explicitly requested.
- OUTPUT MATCHES ANSWER TYPE: Ensure your final `(get-value ...)` matches the expected Answer Type format.

[AVAILABLE SMT-LIB ENVIRONMENT]
{preamble}

[KNOWLEDGE BASE (FROM PASS 1)]
{declarations}
{anchors}

[EXAMPLE]
{example}
"""

# %%
EXAMPLES_PASS2 = {
    "Yes/No": "(assert (or (= max_val (f trace_series 0.0)) (or (= max_val (f trace_series 10.0)) (or (= max_val (f trace_series 20.0)) (or (= max_val (f trace_series 30.0)) (or (= max_val (f trace_series 50.0)) (= max_val (f trace_series 75.0))))))))\n(assert (and (>= max_val (f trace_series 0.0)) (and (>= max_val (f trace_series 10.0)) (and (>= max_val (f trace_series 20.0)) (and (>= max_val (f trace_series 30.0)) (and (>= max_val (f trace_series 50.0)) (>= max_val (f trace_series 75.0))))))))\n(assert (= AnsBool (> max_val 2.0)))\n(check-sat)\n(get-value (AnsBool))\n(exit)",
    "Factoid": '(assert (= sidewall_is_part_of_region_bool true))\n(assert (= issue_forms_on_sidewall_bool true))\n(assert (ite (and sidewall_is_part_of_region_bool issue_forms_on_sidewall_bool) (= AnsString (name_of identified_issue_entity)) (= AnsString "Unknown")))\n(check-sat)\n(get-value (AnsString))\n(exit)',
    "List": "(assert (is_gt s_CF2_series s_CF_series 6.25))\n(assert (is_gt s_CF_series s_C_series 6.25))\n(assert (= rank1_entity e_CF2_entity))\n(assert (= rank2_entity e_CF_entity))\n(assert (= rank3_entity e_C_entity))\n(check-sat)\n(get-value ((name_of rank1_entity) (name_of rank2_entity) (name_of rank3_entity)))\n(exit)",
    "Paragraph": '(assert (= o_initial_drop_bool (is_dec o1s_series 0.0 10.0)))\n(assert (= o_steady_decrease_bool (is_dec o1s_series 10.0 75.0)))\n(assert (= AnsString (ite (and o_initial_drop_bool o_steady_decrease_bool) "Oxygen steadily decreases" "Oxygen fluctuates")))\n(check-sat)\n(get-value (AnsString))\n(exit)',
}

# %%
PREAMBLE = """
;; --- UNIVERSAL PREAMBLE ---
;; Logic: Combined Linear Real Arithmetic and Strings (supported by Z3/CVC5)
(set-logic ALL)

;; 1. TYPE DEFINITIONS
(declare-sort Series 0)
(declare-sort Entity 0)
(declare-const epsilon Real)
(assert (= epsilon 0.001))

;; 2. CORE MAPPING FUNCTIONS

;; Maps a Series and a real input (e.g., time) to a real value
(declare-fun f (Series Real) Real)

;; Indicates whether a Series has a given Entity as an attribute
(declare-fun attr (Series Entity) Bool)

;; Returns the string name of an Entity
(declare-fun name_of (Entity) String)

;; Returns the unit (as string) associated with a Series
(declare-fun unit_of (Series) String)

;; 3. GEOMETRIC & TREND PRIMITIVES

;; Computes absolute difference between two real numbers
(define-fun diff ((a Real) (b Real)) Real
  (ite (>= (- a b) 0.0) (- a b) (- b a)))

;; Checks if Series s1 is significantly greater than s2 at point x (with tolerance epsilon)
(define-fun is_gt ((s1 Series) (s2 Series) (x Real)) Bool
  (> (- (f s1 x) (f s2 x)) epsilon))

;; Checks if Series s1 and s2 are approximately equal at point x (within epsilon)
(define-fun is_eq ((s1 Series) (s2 Series) (x Real)) Bool
  (<= (diff (f s1 x) (f s2 x)) epsilon))

;; Checks if Series s is increasing between x1 and x2 (beyond epsilon threshold)
(define-fun is_inc ((s Series) (x1 Real) (x2 Real)) Bool
  (> (- (f s x2) (f s x1)) epsilon))

;; Checks if Series s is decreasing between x1 and x2 (beyond epsilon threshold)
(define-fun is_dec ((s Series) (x1 Real) (x2 Real)) Bool
  (> (- (f s x1) (f s x2)) epsilon))

;; 4. EXTREMA & INTERSECTION

;; Checks if Series s has value approximately equal to y at point x (within epsilon)
(define-fun is_at_val ((s Series) (x Real) (y Real)) Bool
  (<= (diff (f s x) y) epsilon))

;; Checks if x_m is a local peak (maximum) compared to neighbors x_l and x_r
(define-fun is_peak ((s Series) (x_l Real) (x_m Real) (x_r Real)) Bool
  (and (>= (f s x_m) (f s x_l)) (>= (f s x_m) (f s x_r))))

;; Checks if Series s is approximately constant between x1 and x2
(define-fun is_const ((s Series) (x1 Real) (x2 Real)) Bool
  (<= (diff (f s x1) (f s x2)) epsilon))

;; 5. STANDARDIZED OUTPUT VARIABLES
(declare-const AnsBool Bool)
(declare-const AnsReal Real)
(declare-const AnsString String)
"""


# %%
def validate_smt(code: str) -> tuple[bool, str]:
    """
    Validates SMT-LIB code by executing cvc5.
    Returns: (bool: is_satisfiable, str: output_message)
    """
    # 1. Write to a secure temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".smt2", delete=False) as tf:
        tf.write(code)
        temp_path = tf.name

    try:
        # 2. Execute solver with strict timeout and specific flags
        # --produce-models is necessary if you plan to call (get-model) later
        result = subprocess.run(
            [
                CVC5_PATH,
                "--lang",
                "smt2",
                "--produce-models",
                "--incremental",
                temp_path,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # 3. Precise Status Parsing
        # The first non-empty line of SMT-LIB output is typically the status
        lines = [line for line in stdout.split("\n") if line.strip()]
        status = lines[0].lower() if lines else ""

        if stderr or "error" in stdout.lower():
            return False, stderr if stderr else stdout
        if "sat" == status:
            return True, stdout
        elif "unsat" == status:
            return False, stdout
        elif "unknown" == status:
            return False, stdout
        else:
            return False, f"Unexpected Solver Output: {stdout}"
    except subprocess.TimeoutExpired:
        return (
            False,
            "Timeout: The solver took too long (potential infinite search space).",
        )
    except Exception as e:
        return False, f"Internal Execution Error: {str(e)}"
    finally:
        # Ensure cleanup even if execution fails
        if os.path.exists(temp_path):
            os.remove(temp_path)


# %%
def clean_duplicate_declarations(declarations_str: str) -> str:
    seen_declarations = set()
    clean_lines = []

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


def deduplicate_anchors(anchors_str):
    anchors = {}
    lines = anchors_str.strip().split("\n")
    for line in lines:
        # Regex to find (f series x)
        match = re.search(
            r"\(assert\s+\(=\s+\(f\s+([a-zA-Z0-9_]+)\s+([0-9.]+)\)\s+([0-9.]+)\)\)",
            line,
        )
        if match:
            series, x, y = match.groups()
            anchors[(series, x)] = y

    clean_lines = [f"(assert (= (f {s} {x}) {y}))" for (s, x), y in anchors.items()]
    return "\n".join(clean_lines)


def build_dynamic_phase1b_cfg(declarations: str) -> CFG:
    """
    Pass 1B dynamically forces the model to extract anchors and assert names
    ONLY for the specific variables declared in 1A.
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

    # Most SMT-LIB parsers (especially cvc5 and Z3 in certain configurations) expect 7-bit ASCII for string literals, the solver crashes.
    STRING_LIT: /"[\x20-\x7E]*"/
    """
    return CFG(dynamic_grammar)


def generate_declarations(
    model, q_obj, image, summary, max_retries=3, verbose=False, **gen_kwargs
):
    question_text = q_obj.get("question") or q_obj.get("questions")
    question_type = q_obj.get("question_type", "")
    answer_type = q_obj.get("answer_type", "")

    # ==========================================
    # PASS 1A: Generate and Validate Declarations
    # ==========================================
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

    # ==========================================
    # PASS 1B: Generate and Validate Anchors
    # ==========================================
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


# %%
def build_dynamic_phase2_cfg(
    declarations: str, valid_numbers: list[str] = None, answer_type: str = None
) -> CFG:
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

    # --- THE ALLOWED NUMBERS INJECTION ---
    # Merge the table's numbers with common base numbers the LLM might need for math
    allowed_nums = set(
        valid_numbers
        + ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "10.0", "100.0", "-1.0"]
    )
    num_rule = " | ".join([f'"{n}"' for n in allowed_nums])

    logic_seq_rule = "logic_assert " + " ".join(["logic_assert?"] * 32)

    # Define the specific final assertion based on answer type
    # Use calculated terms for all final assertions to ensure the model performs reasoning rather than hardcoding.
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

    # Capped to physically prevent re-assignment loops.
    logic_sequence: {logic_seq_rule}

    logic_assert: "(assert (= " LOGIC_VAR_BOOL " " bool_term "))\n"
                | "(assert (= " LOGIC_VAR_REAL " " real_term "))\n"
                | "(assert (= " ENTITY_SYM " " ENTITY_SYM "))\n"
                | "(assert " bool_term ")\n"

    # Injected type-specific final assertion
    {final_assert_rule}

    check_sat_cmd: "(check-sat)\n"

    get_value_cmd: "(get-value (" gv_list "))\n"
    gv_list: gv_item (" " gv_item)*
    gv_item: LOGIC_VAR_BOOL | LOGIC_VAR_REAL | "AnsBool" | "AnsReal" | "AnsString" | ENTITY_SYM | SERIES_SYM | string_preamble_call | bool_preamble_call | real_preamble_call | STRING_LIT

    exit_cmd: "(exit)\n"

    # --- STRICT TYPED TERMS ---
    ?real_term: DECIMAL | LOGIC_VAR_REAL | real_preamble_call | real_expr | "epsilon" | "AnsReal"
    ?bool_term: "true" | "false" | LOGIC_VAR_BOOL | bool_preamble_call | bool_expr | "AnsBool"
    ?string_term: STRING_LIT | string_preamble_call | string_expr | "AnsString"

    # --- STRICT TYPED EXPRESSIONS (Arity Enforced) ---
    real_expr: "(" REAL_OP " " real_term " " real_term ")"
             | "(- " real_term ")"
             | "(ite " bool_term " " real_term " " real_term ")"

    bool_expr: "(" BOOL_BIN_OP " " bool_term " " bool_term ")"
             | "(not " bool_term ")"
             | "(" COMP_OP " " real_term " " real_term ")"
             | "(= " real_term " " real_term ")"       # Explicit Real Equality
             | "(= " bool_term " " bool_term ")"       # Explicit Boolean Equality
             | "(= " string_term " " string_term ")"   # Explicit String Equality
             | "(= " ENTITY_SYM " " ENTITY_SYM ")"     # Explicit Entity Equality
             | "(distinct " ENTITY_SYM " " ENTITY_SYM ")"
             | "(ite " bool_term " " bool_term " " bool_term ")"

    string_expr: "(ite " bool_term " " string_term " " string_term ")"

    REAL_OP: "+" | "-" | "*" | "/"
    BOOL_BIN_OP: "and" | "or" | "=" | "=>"
    COMP_OP: ">" | "<" | ">=" | "<=" | "="

    # --- PREAMBLE CALLS SEPARATED BY RETURN TYPE ---
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

    # Replaced the generic regex with the dynamic exact-match rule
    DECIMAL: {num_rule}
    # Most SMT-LIB parsers (especially cvc5 and Z3 in certain configurations) expect 7-bit ASCII for string literals, the solver crashes.
    STRING_LIT: /"[\x20-\x7E]*"/
    """
    return CFG(dynamic_grammar)


# %%
def parse_table_deterministically(table_str: str) -> tuple[str, str, list]:
    """Parses a Markdown/CSV table and deterministically generates SMT Pass 1A & 1B."""
    lines = [line.strip() for line in table_str.strip().split("\n") if line.strip()]

    # 1. Parse rows (Markdown/CSV/TSV)
    rows = []
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

    if not rows:  # Fallback
        for line in lines:
            rows.append([c.strip() for c in re.split(r"\t|,", line)])

    if not rows:
        return "", "", []

    headers = rows[0]
    data = rows[1:]
    y_cols = headers[1:]

    declarations = []
    anchors = []
    valid_numbers = []

    # TRACKER: Ensure unique names for SMT constants
    seen_names = set()
    col_to_clean_name = {}  # Maps column index to its unique ID

    # 2. Process Headers first to ensure uniqueness
    for i, y_col in enumerate(y_cols):
        # Initial cleaning
        base_name = re.sub(r"[^a-zA-Z0-9]", "", y_col)[:15]
        if not base_name:
            base_name = f"col{i}"
        elif base_name[0].isdigit():
            # Prepend 'v_' so the SMT symbol starts with a letter
            base_name = f"v_{base_name}"

        # Deduplication logic
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

    # Standard logic variables
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

    # Helper function to parse messy numeric strings
    def parse_messy_number(val_str: str) -> float:
        # Strip spaces, commas, and common approximation symbols
        clean_str = re.sub(r"[\s~<>,\xa0]", "", val_str)
        multiplier = 1.0

        # Handle k (thousands), M (millions), or % (percent)
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

    # 4. Extract Data Points using the mapped unique names
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
                    # Still gracefully skip "N/A" or completely empty cells
                    continue

    return "\n".join(declarations), "\n".join(anchors), list(set(valid_numbers))


# %%
def reflect(
    model, q_obj, image, summary, table, max_retries=3, verbose=False, **gen_kwargs
):
    question_text = q_obj.get("question") or q_obj.get("questions")
    question_type = q_obj.get("question_type", "")
    answer_type = q_obj.get("answer_type", "")

    # ==========================================
    # PASS 1: Declarations & Anchors
    # ==========================================
    if table:
        declarations, anchors, valid_numbers = parse_table_deterministically(table)
    else:
        # TODO!: This isn't meant to be called in the current SODA pipeline since we always have tables, but we can keep it as a fallback for more free-form data in the future.
        declarations, anchors = generate_declarations(
            model,
            q_obj,
            image,
            summary,
            max_retries=3,
            verbose=False,
            **gen_kwargs,
        )
        valid_numbers = re.findall(r"-?\d+\.\d+", anchors)

    full_kb = f"{declarations}\n{anchors}"

    if verbose:
        print("[KNOWLEDGE BASE EXTRACTED]")
        print(full_kb)
        print("-" * 40)

    # ==========================================
    # PASS 1.5: The Planning Scratchpad (Initial)
    # ==========================================
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

    # Generate the plan WITHOUT the strict CFG so the model can think freely in English
    scratchpad_plan = model(plan_prompt, **gen_kwargs)

    if verbose:
        print("[INITIAL PLANNING SCRATCHPAD GENERATED]")
        print(scratchpad_plan)
        print("-" * 40)

    # ==========================================
    # PHASE 2: SMT Logic Translation & Reflection
    # ==========================================
    try:
        dynamic_cfg_pass2 = build_dynamic_phase2_cfg(
            full_kb, valid_numbers, answer_type=answer_type
        )
    except Exception as e:
        return None, f"Failed to compile dynamic Phase 2 grammar: {e}"

    for attempt in range(max_retries):
        # 1. Build prompt_pass2 dynamically per attempt so it includes the CURRENT plan
        second_pass_text = PROMPT_TEMPLATE_PASS2.format(
            question=question_text,
            question_type=question_type,
            answer_type=answer_type,
            summary=summary,
            preamble=PREAMBLE,
            declarations=declarations,
            anchors=anchors,
            scratchpad_plan=scratchpad_plan,  # Injects the new/updated plan
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

        # 2. Generate with the strict CFG to guarantee syntax
        logic = model(prompt_pass2, dynamic_cfg_pass2, **gen_kwargs)

        final_smt = f""";; --- [PREAMBLE] ---
{PREAMBLE}
;; --- [KNOWLEDGE BASE] ---
{full_kb}
;; --- [PASS 2: Logic & Execution] Attempt {attempt + 1} ---
{logic}
"""

        success, output = validate_smt(final_smt)

        # Validate semantic bindings and mathematical proofs
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

        # 3. Dynamic Reflection: Update the unconstrained PLANNER on failure
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

        # Create a new prompt specifically for the reflection step
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

        # 4. Generate the REFORMULATED plan
        scratchpad_plan = model(reflection_prompt, **gen_kwargs)

        if verbose:
            print(
                f"[REFORMULATED PLANNING SCRATCHPAD - Preparing for Attempt {attempt + 2}]"
            )
            print(scratchpad_plan)
            print("-" * 40)

    return None, "Failed to reach a valid logical formulation in Pass 2 after retries."


# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use torch.float16 if your GPU is older and doesn't support bfloat16
    bnb_4bit_use_double_quant=True,  # Optional: Saves a bit more memory
    bnb_4bit_quant_type="nf4",  # Optional: Recommended for better performance
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

lm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", quantization_config=bnb_config
)

# %%
tokenizer.chat_template = (
    "{% set enable_thinking = false %}\n" + tokenizer.chat_template
)

# %%
model = outlines.from_transformers(lm, tokenizer)

# %%
if STATE_FILE.exists():
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
    print(f"Loaded existing state with {len(state)} samples.")
else:
    state = {}
    print("Initialized empty state.")

# %%
for split in tqdm(CATEGORIES, desc="Categories", position=0):
    if split not in state:
        state[split] = {}

    split_dir = COMPETITION_DATA_DIR / split
    assert split_dir.exists(), f"Directory for split '{split}' not found!"

    json_files = list(split_dir.rglob("*.json"))

    pbar = tqdm(json_files, desc=f"Processing {split} split", position=1, leave=False)
    for json_file in pbar:
        fullpath = str(json_file)

        if (
            "content.json" in json_file.name
            or "images" not in fullpath
            or ".vscode" in fullpath
        ):
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        sample_id = data.get("sample_id", None)
        assert sample_id, f"sample_id missing in {json_file}"

        # Initialize sample_id under the current split in state
        if sample_id not in state[split]:
            state[split][sample_id] = {}

        img_path = json_file.with_suffix(".jpg")
        if not img_path.exists():
            continue

        full_img = None
        bboxes = data.get("bbox", {})
        vqa_data = data.get("vqa", {})
        summarization = data.get("summarization", {})
        data_extraction = data.get("data_extraction", {})

        # 2. Iterate through subfigures
        for sub_key, q_list in vqa_data.items():
            if sub_key not in state[split][sample_id]:
                state[split][sample_id][sub_key] = {}

            # 3. CRITICAL: Check if a data table extract is available
            table = data_extraction.get(sub_key, None)
            if not table:
                continue

            if sub_key not in bboxes:
                continue

            # Open and crop the image only when we know we need it
            if full_img is None:
                full_img = Image.open(img_path.absolute())

            box = bboxes[sub_key]
            left, top = box["x"], box["y"]
            right, bottom = left + box["width"], top + box["height"]
            crop = full_img.crop((left, top, right, bottom))

            summary = summarization.get(sub_key, "N/A")

            # 4. Process each question
            for q_obj in q_list:
                question_text = q_obj.get("question") or q_obj.get("questions")
                if not question_text:
                    continue

                # Skip if we already successfully populated this question
                if question_text in state[split][sample_id][sub_key]:
                    continue

                pbar.set_description(f"Reflecting: {sample_id} | Sub: {sub_key}")

                # 5. Run the iterative reflection
                smt_code, solver_output = reflect(
                    model=model,
                    q_obj=q_obj,
                    image=crop,
                    summary=summary,
                    table=table,
                    max_retries=3,
                    verbose=False,  # Set to True if you want to debug in the cell output
                    do_sample=True,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    min_p=MIN_P,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    repetition_penalty=REPETITION_PENALTY,
                )

                # 6. Populate the dictionary under the specific split
                state[split][sample_id][sub_key][question_text] = {
                    "code": smt_code,
                    "output": solver_output,
                }

                # 7. Checkpoint to disk instantly
                with open(STATE_FILE, "w") as f:
                    json.dump(state, f, indent=4)

print("Pipeline execution complete! State fully synced.")

# %%
print(f"Saving final consolidated data to {SMT_FILE}...")
with open(SMT_FILE, "w") as f:
    json.dump(state, f, indent=4)

# Delete the temporary state file now that the final write is complete
if STATE_FILE.exists():
    STATE_FILE.unlink()
    print(f"Deleted temporary state file: {STATE_FILE}")

print("All done! Data successfully written to SMT_FILE.")
