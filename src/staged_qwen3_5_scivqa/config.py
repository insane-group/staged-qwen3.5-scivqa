"""Application configuration and constants for staged Qwen3.5 SciVQA."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# ── Model ─────────────────────────────────────────────────────────────
MODEL_ID: str = "unsloth/Qwen3.5-9B"

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR: Path = Path.cwd().parent
DATA_DIR: Path = BASE_DIR / "data"
COMPETITION_DATA_DIR: Path = (
    BASE_DIR / "ALD-E-ImageMiner" / "icdar2026-competition-data"
)
CVC5_PATH: Path = Path.home() / "cvc5-Linux-x86_64-shared" / "bin" / "cvc5"

# ── LoRA Checkpoints ──────────────────────────────────────────────────
LORA_CHECKPOINT_BASE: str = f"Sci-ImageMiner-{MODEL_ID.split('/')[1]}-LORA"
LORA_CHECKPOINT_FACTOID: str = f"{LORA_CHECKPOINT_BASE}-FACTOID"
LORA_CHECKPOINT_LIST: str = f"{LORA_CHECKPOINT_BASE}-LIST"
LORA_CHECKPOINT_PARAGRAPH: str = f"{LORA_CHECKPOINT_BASE}-PARAGRAPH"
LORA_CHECKPOINT_YESNO: str = f"{LORA_CHECKPOINT_BASE}-YESNO"
LORA_CHECKPOINT_SUMMARY: str = f"{LORA_CHECKPOINT_BASE}-SUMMARY"
LORA_CHECKPOINT_EXTRACTION: str = f"{LORA_CHECKPOINT_BASE}-EXTRACTION"

# ── Token Budgets per Answer Type ─────────────────────────────────────
TOKEN_BUDGETS: dict[str, dict[str, int]] = {
    "Yes/No": {"max_new_tokens": 1, "max_sequence_length": 3072},
    "Paragraph": {"max_new_tokens": 176, "max_sequence_length": 3072},
    "Factoid": {"max_new_tokens": 72, "max_sequence_length": 2560},
    "List": {"max_new_tokens": 144, "max_sequence_length": 2560},
    "Summary": {"max_new_tokens": 256, "max_sequence_length": 4096},
    "Table": {"max_new_tokens": 768, "max_sequence_length": 3072},
}

# ── Inference Settings ────────────────────────────────────────────────
ENABLE_THINKING: bool = False
TEMPERATURE: float = 0.7
MIN_P: float = 0.01
TOP_P: float = 0.8
TOP_K: int = 20

# ── Training Settings ─────────────────────────────────────────────────
NUM_TRAIN_EPOCHS: int = 5
LORA_R: int = 16
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.0
LORA_RANDOM_STATE: int = 3407

SFT_PER_DEVICE_BATCH_SIZE: int = 2
SFT_GRADIENT_ACCUMULATION_STEPS: int = 4
SFT_WARMUP_RATIO: float = 0.05
SFT_LEARNING_RATE: float = 2e-4
SFT_WEIGHT_DECAY: float = 0.001

# ── SMT Settings ──────────────────────────────────────────────────────
SMT_MAX_NEW_TOKENS: int = 2048
SMT_TEMPERATURE: float = 1.0
SMT_TOP_P: float = 0.95
SMT_TOP_K: int = 20
SMT_MIN_P: float = 0.0
SMT_PRESENCE_PENALTY: float = 1.5
SMT_REPETITION_PENALTY: float = 1.0

# ── Reflection Settings ───────────────────────────────────────────────
REFLECTION_TEMPERATURE: float = 0.2
REFLECTION_TOP_P: float = 0.1
REFLECTION_TOP_K: int = 20
REFLECTION_MIN_P: float = 0.0
REFLECTION_MAX_NEW_TOKENS: int = 256
REFLECTION_MAX_SEQUENCE_LENGTH: int = 4096
REFLECTION_REPETITION_PENALTY: float = 1.0

# ── HuggingFace Datasets ──────────────────────────────────────────────
HF_BASE_REPO: str = "VaMaSi/staged-qwen3.5-scivqa"
HF_VQA_REPO: str = f"{HF_BASE_REPO}-vqa"
HF_SUMMARY_REPO: str = f"{HF_BASE_REPO}-summary"
HF_TABLE_REPO: str = f"{HF_BASE_REPO}-table"
HF_TOKEN: str | None = None

# ── Context Window ────────────────────────────────────────────────────
CONTEXT_WINDOW_SIZE: int = 2

# ── VQA Prompts ───────────────────────────────────────────────────────
PROMPT_YES_NO: str = """
<image>

[SUMMARY]
{summary}

[TABLE]
{table}

Additional context from the original paper:
{context}

Answer the following scientific figure question by reasoning strictly over the
information visible in the figure and the provided context.

Question type: {question_type}
Question: {question}

Strict requirements:
1. Identify the main variables shown (axes, units, and any legend information).
2. Ignore decorative graphics, schematics, arrows, and background elements.
3. Use the provided caption/context only to support interpretation when necessary.
4. Do not speculate or infer beyond what is visually supported.
5. Output plain text only, with no JSON, no code fences, and no surrounding
   explanatory text.
6. Output your answer STRICTLY as "Yes" or "No" (title case).

Example:
Yes
"""

PROMPT_FACTOID: str = """
<image>

[SUMMARY]
{summary}

[TABLE]
{table}

Additional context from the original paper:
{context}

Answer the following scientific figure question by reasoning strictly over the
information visible in the figure and the provided context.

Question type: {question_type}
Question: {question}

Strict requirements:
1. Identify the main variables shown (axes, units, and any legend information).
2. Ignore decorative graphics, schematics, arrows, and background elements.
3. Use the provided caption/context only to support interpretation when necessary.
4. Do not speculate or infer beyond what is visually supported.
5. Output plain text only, with no JSON, no code fences, and no surrounding
   explanatory text.
6. Output your answer STRICTLY as a concise term or short phrase.

Example:
The feature corresponds to an interband electronic transition or optical
absorption edge.
"""

PROMPT_LIST: str = """
<image>

[SUMMARY]
{summary}

[TABLE]
{table}

Additional context from the original paper:
{context}

Answer the following scientific figure question by reasoning strictly over the
information visible in the figure and the provided context.

Question type: {question_type}
Question: {question}

Strict requirements:
1. Identify the main variables shown (axes, units, and any legend information).
2. Ignore decorative graphics, schematics, arrows, and background elements.
3. Use the provided caption/context only to support interpretation when necessary.
4. Do not speculate or infer beyond what is visually supported.
5. Output plain text only, with no JSON, no bullet points, no numbered lists,
   no code fences, and no surrounding explanatory text.
6. Output your answer STRICTLY as comma-separated values (order-insensitive).

Example:
Absence of pits or voids, Smooth and continuous surface, Lack of corrosive
attack patterns, Reduced by-product interaction with copper
"""

PROMPT_PARAGRAPH: str = """
<image>

[SUMMARY]
{summary}

[TABLE]
{table}

Additional context from the original paper:
{context}

Answer the following scientific figure question by reasoning strictly over the
information visible in the figure and the provided context.

Question type: {question_type}
Question: {question}

Strict requirements:
1. Identify the main variables shown (axes, units, and any legend information).
2. Ignore decorative graphics, schematics, arrows, and background elements.
3. Use the provided caption/context only to support interpretation when necessary.
4. Do not speculate or infer beyond what is visually supported.
5. Output plain text only, with no JSON, no bullet points, no numbered lists,
   no code fences, and no surrounding explanatory text.
6. Output your answer STRICTLY as a paragraph containing at least 3 sentences
   providing an explanatory answer.

Example:
Multiple well-defined interfaces, Alternating high-contrast layers, Disruption
of continuous grain boundaries, Uniform nanometer-scale layer thickness
"""

PROMPTS: dict[str, str] = {
    "Yes/No": PROMPT_YES_NO,
    "Factoid": PROMPT_FACTOID,
    "List": PROMPT_LIST,
    "Paragraph": PROMPT_PARAGRAPH,
}

# ── Summary Prompt ────────────────────────────────────────────────────
PROMPT_SUMMARY: str = """
<image>

Additional context from the original paper:
{context}

Generate a concise summary of the scientific chart or plot.

Strict requirements:

1. Identify the main variables shown (axes, units, and any legend information).
2. Describe the key trends, patterns, or relationships visible in the data (e.g., increases, decreases, peaks, correlations, distributions).
3. Highlight any notable quantitative information if clearly visible (e.g., approximate ranges, maxima/minima, relative differences).
4. Focus on scientifically meaningful insights rather than low-level visual details.
5. Use the provided caption/context only to support interpretation when necessary.
6. Do not speculate or infer beyond what is visually supported.
7. Keep the summary factual, precise, and grounded in the figure.

Output format:

- 1–3 sentences only.
- No bullet points, no extra text.

Example:

Polar heatmap of Al₂O₃ etch rate distribution (nm/cycle) across a wafer, with radial axis (0–3 cm from center) and angular labels (0°–360°). Rates peak centrally at 0.139 nm/cycle and decrease toward the edge (~0.131–0.133 nm/cycle), indicating a ~6% radial gradient with minor angular variation. This suggests high isotropy consistent with optimized ALE processes.
"""

# ── Table Extraction Prompt ───────────────────────────────────────────
PROMPT_TABLE: str = """
<image>

Additional context from the original paper:
{context}

Extract the structured data represented in the scientific chart.

Strict requirements:

1. Identify all relevant chart regions (axes, ticks, labels, legends, plotted data).
2. Determine the field/column names from axis labels, legends, or annotations.
3. Reconstruct the underlying tabular structure represented in the chart.
4. Extract all visible textual and numerical values as accurately as possible.
5. Strip all standard table formatting. Use commas (,) to separate columns and semicolons (;) to separate rows. Do not use spaces around the delimiters.
6. Preserve units exactly as shown in the figure.
7. Ignore decorative elements, schematics, arrows, and non-data graphics.
8. Do not infer or extrapolate missing values—only include explicitly visible data.
9. Ensure the output is complete, consistent, and machine-readable.

Output format:

- A single dense text string representing the table.
- Columns separated by commas `,`.
- Rows separated by semicolons `;`.
- No explanations or extra text.

Example:

Time (s),Mass Change (ng/cm²);0,0;2000,-500;4000,-1000;6000,-1500
"""

# ── SMT Prompts ───────────────────────────────────────────────────────
PROMPT_TEMPLATE_PASS1A: str = """
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
2. Pre-declare the logical variables (Bool/Real) you will need to perform
   reasoning in Pass 2.

[STRICT RULES]
- PURE DECLARATIONS ONLY: Do not perform any logic, and do NOT use `(assert ...)`.
- NO ANCHORS YET: Do NOT extract data points (f series x) or assert
  names/attributes here. You will do that in the next step.
- SCHEMA LOCK: You must declare any variable (e.g., max_val) here, or you
  won't be able to use it in Pass 2.
- SERIES: You MUST declare a separate Series variable (e.g., carbon_series,
  oxygen_series) for every individual data series present in the table.
  Notice the mandatory '_series' suffix!

[AVAILABLE SMT-LIB ENVIRONMENT]
{preamble}

[EXAMPLE]
{example}
"""

PROMPT_TEMPLATE_PASS1B: str = """
<image>

[SUMMARY]
{summary}

[TASK]
Using the variables declared in the Knowledge Base, your task is to:
1. Assert the names and attributes (`name_of`, `attr`) mapping Series to Entities.
2. Extract the exact numeric data points (`f`) for each Series.

[STRICT RULES]
- NO NEW DECLARATIONS: You must only use the exact variables provided below.
- DECIMALS ONLY: SMT-LIB requires explicit decimals for Real types
  (e.g., 0.0 instead of 0).
- ANCHORS: Extract representative anchors (at least 4) for the declared series.
- UNIQUE VALUES: Each Series can have only ONE value per x-coordinate.
  Do not assert (f s x y1) and (f s x y2) if y1 != y2.
- CONSISTENCY: Ensure your extracted points form a single, logical curve.

[AVAILABLE SMT-LIB ENVIRONMENT]
{preamble}

[KNOWLEDGE BASE (FROM PASS 1A)]
{declarations}

[EXAMPLE]
{example}
"""

PROMPT_TEMPLATE_PASS2: str = """
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
Translate your APPROVED LOGICAL PLAN into strict SMT-LIB code using ONLY the
variables provided in the Knowledge Base.

[STRICT RULES]
- NO DECLARATIONS: Do not use 'declare-const'. Use only the variables provided
  in the Knowledge Base.
- DIRECT ASSERTIONS: Use the pre-declared Boolean/Real variables to represent
  your logic.
- FINITE SEARCH: Use (or ...) for maxima/minima iterating ONLY over the exact
  x-values asserted in the Knowledge Base. Do not hallucinate or guess data
  points.
- USE PREAMBLE: Use functions like 'is_dec', 'is_inc', and 'is_eq' to assign
  values to the pre-declared Booleans.
- LIST/RANKING: If 'List', prove the requested observations (trends,
  comparisons, or order). Use (ite ...) to construct a descriptive 'AnsString'
  containing the list of findings or use 'rankN_entity' only if ranking is
  explicitly requested.
- OUTPUT MATCHES ANSWER TYPE: Ensure your final `(get-value ...)` matches the
  expected Answer Type format.

[AVAILABLE SMT-LIB ENVIRONMENT]
{preamble}

[KNOWLEDGE BASE (FROM PASS 1)]
{declarations}
{anchors}

[EXAMPLE]
{example}
"""

PROMPT_TEMPLATE_PLANNING: str = """
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
Create a logical plan to answer the question using ONLY the provided
Knowledge Base and Summary.

Follow this exact format:
1. ANALYSIS: Identify exactly what the question seeks.
2. DATA EVALUATION:
   - If 'List': Identify all relevant items and their order.
   - If 'Yes/No': Determine the truth value by comparing specific data points.
   - If 'Factoid/Paragraph': Locate the specific string or entity that
     directly answers the prompt.
3. SMT STRATEGY: List the variables and functions needed.
   - CRITICAL: Root variables (AnsBool, AnsString, AnsReal) CANNOT be
     assigned raw literals directly (e.g., no `(= AnsString "Answer")`
     or `(= AnsBool true)`). You MUST use a calculated term.
   - If Answer Type is 'Yes/No', you MUST assign the result to 'AnsBool'
     via a logical expression (e.g., `(> val1 val2)`) or a previously
     computed boolean variable.
   - If Answer Type is 'Factoid' or 'Paragraph', you MUST assign the
     result to 'AnsString' via a string expression (e.g., an `(ite ...)`
     statement) or a preamble call (e.g., `(name_of ...)`).
   - If Answer Type is 'List', you MUST assign the result to 'AnsBool',
     'AnsString' or 'AnsReal' using calculated expressions.
   - FALLBACK RULE: If the required information is missing or undecidable
     from the KB/Summary, you MUST still generate a valid calculated
     SMT assignment:
     * For 'Yes/No': Assign `AnsBool` to a false expression
       (e.g., `(= AnsBool (= 1.0 0.0))`).
     * For 'List', 'Factoid', or 'Paragraph': Assign `AnsString` using
       a dummy conditional (e.g.,
       `(= AnsString (ite true "Information not found in context" ""))`).
4. FINAL ANSWER: Provide the answer exactly as it should appear in the
   final output.

[PLANNING SCRATCHPAD]
"""

PROMPT_TEMPLATE_REFLECTION: str = """
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
Your previous plan was translated into the SMT-LIB code above, but it
failed validation.

Follow this exact format to recover:
1. ERROR ANALYSIS: Identify exactly why the code failed based on the
   feedback.
   - Was it a structural error? (e.g., missing the required `AnsBool`
     or `AnsString` assignment).
   - Did the solver return 'unsat'? (Look for contradictory data points,
     incorrect sorting, or faulty logic bindings).
   - Was there a syntax error? (Look for incorrect function arity,
     missing variables, or unmapped series).
2. REVISED SMT STRATEGY: State exactly which variables (e.g., AnsBool,
   AnsString, rankN_entity) must be assigned to fix the structural or
   logical error.
4. REVISED FINAL ANSWER: Provide the expected answer format.

[REVISED PLANNING SCRATCHPAD]
"""

PROMPT_REWRITE: str = """
You are given an initial answer generated by a vision model, along with
the SMT-LIB mathematical logic code executed to verify the figure's
underlying data, and its resulting CVC5 solver output.

Question type: {question_type}
Question: {question}
Answer Type: {answer_type}

[INITIAL ANSWER]
{answer_cache}

[SMT-LIB CODE EXECUTED]
{code}

[SOLVER OUTPUT]
{output}

Evaluate the initial answer against the solver output.
- If the solver output contradicts the initial answer, rewrite the answer
  using the solver's mathematically/logically precise result.
- If the solver output confirms the initial answer (or is irrelevant/
  failed), rewrite the initial answer ONLY to ensure it perfectly aligns
  with the Strict Requirements below.

Strict Requirements:
1. Output plain text only, with no JSON, no code fences, and no
   surrounding explanatory text for the final answer.
2. Do NOT include any of the following in your final answer: the initial
   answer, the SMT-LIB code, the solver output, or any commentary on
   them. Your final answer should be a standalone response to the
   question.
3. You MUST adhere strictly to the format expected for a '{answer_type}'
   question. Apply the exact formatting rule corresponding to the
   question type below:
   - Yes/No: Output STRICTLY as "Yes" or "No" (title case). No
     punctuation.
   - Factoid: Output ONLY the exact entity, number, or phrase. Do not
     write full sentences, and do not use introductory filler (e.g.,
     write "5" instead of "The answer is 5").
   - List: Output STRICTLY as an exhaustive, comma-separated list. Do
     not include bullet points, numbered lists, or introductory text.
   - Paragraph: Output STRICTLY as a single paragraph containing at
     least 3 sentences. Ensure the explanation is semantically cohesive,
     grammatically correct, and logically explains the solver's findings.
4. You MUST enclose your final answer within <ANSWER>...</ANSWER> tags
   to clearly indicate the answer portion of your response.
"""

# ── SMT Preamble ──────────────────────────────────────────────────────
PREAMBLE: str = """
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

;; Checks if Series s1 is significantly greater than s2 at point x
;; (with tolerance epsilon)
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

# ── SMT Grammar Pass 1A ───────────────────────────────────────────────
SMT_LIB_GRAMMAR_PASS1A: str = r"""
?start: script

# 1. Limit the number of declarations to a safe maximum (e.g., 20)
# This prevents the model from rambling until it hits the token limit.
script: decl_line decl_line? decl_line? decl_line? decl_line? decl_line?
    decl_line? decl_line? decl_line? decl_line? decl_line? decl_line?
    decl_line? decl_line? decl_line? decl_line? decl_line? decl_line?
    decl_line? decl_line?

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

# ── SMT Examples Pass 1A ──────────────────────────────────────────────
# fmt: off
EXAMPLES_PASS1A: dict[str, str] = {
    "Yes/No": "(declare-const s2p_entity Entity)\n(declare-const trace_series Series)\n(declare-const max_val Real)\n(declare-const threshold_met_bool Bool)",
    "Factoid": "(declare-const structural_region_entity Entity)\n(declare-const sidewall_entity Entity)\n(declare-const identified_issue_entity Entity)\n(declare-const sidewall_is_part_of_region_bool Bool)\n(declare-const issue_forms_on_sidewall_bool Bool)",
    "List": "(declare-const e_CF2_entity Entity)\n(declare-const e_CF_entity Entity)\n(declare-const e_C_entity Entity)\n(declare-const s_CF2_series Series)\n(declare-const s_CF_series Series)\n(declare-const s_C_series Series)\n(declare-const rank1_entity Entity)\n(declare-const rank2_entity Entity)\n(declare-const rank3_entity Entity)",
    "Paragraph": "(declare-const o1s_entity Entity)\n(declare-const o1s_series Series)\n(declare-const o_initial_drop_bool Bool)\n(declare-const o_steady_decrease_bool Bool)",
}

EXAMPLES_PASS1B: dict[str, str] = {
    "Yes/No": '(assert (= (name_of s2p_entity) "S2p"))\n(assert (attr trace_series s2p_entity))\n(assert (= (f trace_series 0.0) 0.0))\n(assert (= (f trace_series 10.0) 1.8))\n(assert (= (f trace_series 20.0) 1.6))\n(assert (= (f trace_series 30.0) 1.6))\n(assert (= (f trace_series 50.0) 1.9))\n(assert (= (f trace_series 75.0) 1.2))',
    "Factoid": '(assert (= (name_of structural_region_entity) "Multi-quantum well"))\n(assert (= (name_of sidewall_entity) "sidewall"))\n(assert (= (name_of identified_issue_entity) "Etch damage layer"))',
    "List": '(assert (= (name_of e_CF2_entity) "CF2"))\n(assert (= (name_of e_CF_entity) "CF"))\n(assert (= (name_of e_C_entity) "C"))\n(assert (attr s_CF2_series e_CF2_entity))\n(assert (attr s_CF_series e_CF_entity))\n(assert (attr s_C_series e_C_entity))\n(assert (= (f s_CF2_series 6.25) 5.75))\n(assert (= (f s_CF_series 6.25) 4.15))\n(assert (= (f s_C_series 6.25) 2.50))',
    "Paragraph": '(assert (= (name_of o1s_entity) "O1s"))\n(assert (attr o1s_series o1s_entity))\n(assert (= (f o1s_series 0.0) 16.0))\n(assert (= (f o1s_series 10.0) 10.5))\n(assert (= (f o1s_series 75.0) 7.5))',
}

EXAMPLES_PASS2: dict[str, str] = {
    "Yes/No": "(assert (or (= max_val (f trace_series 0.0)) (or (= max_val (f trace_series 10.0)) (or (= max_val (f trace_series 20.0)) (or (= max_val (f trace_series 30.0)) (or (= max_val (f trace_series 50.0)) (= max_val (f trace_series 75.0))))))))\n(assert (and (>= max_val (f trace_series 0.0)) (and (>= max_val (f trace_series 10.0)) (and (>= max_val (f trace_series 20.0)) (and (>= max_val (f trace_series 30.0)) (and (>= max_val (f trace_series 50.0)) (>= max_val (f trace_series 75.0))))))))\n(assert (= AnsBool (> max_val 2.0)))\n(check-sat)\n(get-value (AnsBool))\n(exit)",
    "Factoid": '(assert (= sidewall_is_part_of_region_bool true))\n(assert (= issue_forms_on_sidewall_bool true))\n(assert (ite (and sidewall_is_part_of_region_bool issue_forms_on_sidewall_bool) (= AnsString (name_of identified_issue_entity)) (= AnsString "Unknown")))\n(check-sat)\n(get-value (AnsString))\n(exit)',
    "List": "(assert (is_gt s_CF2_series s_CF_series 6.25))\n(assert (is_gt s_CF_series s_C_series 6.25))\n(assert (= rank1_entity e_CF2_entity))\n(assert (= rank2_entity e_CF_entity))\n(assert (= rank3_entity e_C_entity))\n(check-sat)\n(get-value ((name_of rank1_entity) (name_of rank2_entity) (name_of rank3_entity)))\n(exit)",
    "Paragraph": '(assert (= o_initial_drop_bool (is_dec o1s_series 0.0 10.0)))\n(assert (= o_steady_decrease_bool (is_dec o1s_series 10.0 75.0)))\n(assert (= AnsString (ite (and o_initial_drop_bool o_steady_decrease_bool) "Oxygen steadily decreases" "Oxygen fluctuates")))\n(check-sat)\n(get-value (AnsString))\n(exit)',
}
# fmt: on

# ── Pydantic Settings (replaces settings.py) ────────────────────────────


class ModelConfig(BaseModel):
    model_id: str = Field(default="unsloth/Qwen3.5-0.8B")
    load_in_4bit: bool = False
    max_seq_length: int = 4096


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    random_state: int = 3407
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


class TrainingConfig(BaseModel):
    epochs: int = 5
    batch_size: int = 2
    grad_accum: int = 4
    lr: float = 2e-4
    optim: str = "adamw_8bit"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.001
    seed: int = 3407
    logging_steps: int = 1
    scheduler: str = "linear"


class InferenceConfig(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.01
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    enable_thinking: bool = False


class StageBudget(BaseModel):
    max_new_tokens: int
    max_sequence_length: int


class StageConfig(BaseModel):
    yes_no: StageBudget = Field(
        default=StageBudget(max_new_tokens=1, max_sequence_length=3072)
    )
    paragraph: StageBudget = Field(
        default=StageBudget(max_new_tokens=176, max_sequence_length=3072)
    )
    factoid: StageBudget = Field(
        default=StageBudget(max_new_tokens=72, max_sequence_length=2560)
    )
    list: StageBudget = Field(
        default=StageBudget(max_new_tokens=144, max_sequence_length=2560)
    )
    summary: StageBudget = Field(
        default=StageBudget(max_new_tokens=256, max_sequence_length=4096)
    )
    table: StageBudget = Field(
        default=StageBudget(max_new_tokens=768, max_sequence_length=3072)
    )

    def get(self, name: str) -> StageBudget:
        return getattr(self, name)  # type: ignore[no-any-return]


class PathsConfig(BaseModel):
    data_dir: Path = Field(default_factory=lambda: Path.cwd().parent / "data")
    competition_data_dir: Path = Field(
        default_factory=lambda: (
            Path.cwd().parent / "ALD-E-ImageMiner" / "icdar2026-competition-data"
        )
    )
    output_dir: Path = Field(default_factory=lambda: Path.cwd().parent / "models")
    cvc5_path: Path = Field(
        default_factory=lambda: (
            Path.home() / "cvc5-Linux-x86_64-shared" / "bin" / "cvc5"
        )
    )


class SMTConfig(BaseModel):
    model_id: str = Field(default="unsloth/Qwen3.5-9B")
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    max_retries: int = 3


class ReflectionConfig(BaseModel):
    model_id: str = Field(default="unsloth/Qwen3.5-9B")
    max_new_tokens: int = 256
    max_sequence_length: int = 4096
    temperature: float = 0.2
    top_p: float = 0.1
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    load_in_4bit: bool = True


class HFConfig(BaseModel):
    token: str | None = None
    hub_repo_id: str | None = None
    push_checkpoints: bool = True
    push_datasets: bool = False
    load_from_hub: bool = False
    dataset_repo_id: str | None = None


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "staged-qwen3.5-scivqa"
    entity: str | None = None
    run_name: str | None = None


class SciVQAConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCIVQA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    stages: StageConfig = Field(default_factory=StageConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    smt: SMTConfig = Field(default_factory=SMTConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    hf: HFConfig = Field(default_factory=HFConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    category: str = "train,dev"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: tuple[PydanticBaseSettingsSource, ...] = (
            init_settings,
            env_settings,
            dotenv_settings,
        )
        if _yaml_config_path and Path(_yaml_config_path).exists():
            sources = (
                init_settings,
                YamlConfigSettingsSource(settings_cls, Path(_yaml_config_path)),
                env_settings,
                dotenv_settings,
            )
        return sources

    def get_stage_budget(self, stage_name: str) -> StageBudget:
        return self.stages.get(stage_name)

    def get_lora_checkpoint_name(self, stage: str) -> str:
        base = f"Sci-ImageMiner-{self.model.model_id.split('/')[-1]}-LORA"
        mapping = {
            "summary": f"{base}-SUMMARY",
            "table": f"{base}-EXTRACTION",
            "factoid": f"{base}-FACTOID",
            "list": f"{base}-LIST",
            "paragraph": f"{base}-PARAGRAPH",
            "yes_no": f"{base}-YESNO",
        }
        return mapping.get(stage, f"{base}-{stage.upper()}")

    def get_state_path(self, stage: str) -> Path:
        mapping = {
            "summary": self.paths.data_dir
            / f"submission_finetuning_summary_{self.category}_state.json",
            "table": self.paths.data_dir
            / f"submission_finetuning_extraction_{self.category}_state.json",
            "vqa": self.paths.data_dir
            / f"submission_finetuning_{self.category}_state.json",
            "smt": self.paths.data_dir / f"smt_{self.category}_state.json",
            "reflection": self.paths.data_dir
            / f"submission_reflection_{self.category}_state.json",
            "submission": self.paths.data_dir
            / f"submission_final_{self.category}.json",
        }
        return mapping.get(stage, self.paths.data_dir / f"{stage}_state.json")


# Module-level storage for YAML config path (set by load_config)
_yaml_config_path: str | None = None


def load_config(
    config_path: str | Path | None = None,
    **overrides,
) -> SciVQAConfig:
    """Load SciVQAConfig with optional YAML file and CLI overrides."""
    global _yaml_config_path
    _yaml_config_path = str(config_path) if config_path else None
    cfg = SciVQAConfig()
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
    return cfg
