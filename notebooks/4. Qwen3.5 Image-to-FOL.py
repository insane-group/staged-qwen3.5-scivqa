# %% [markdown]
# # Qwen3.5 Image-to-FOL
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
from unsloth import FastVisionModel
# Force unsloth to be on top

import json
from pathlib import Path

from PIL import Image
import warnings
from tqdm.auto import tqdm
import sys
import io
import traceback
from z3 import (
    Solver,
    Int,
    Real,
    Bool,
    Function,
    DeclareSort,
    RealSort,
    BoolSort,
    Const,
    StringVal,
    sat,
    unsat,
)
from collections import defaultdict

# %%
MODEL_ID = "unsloth/Qwen3.5-0.8B"

BASE_DIR = Path.cwd().parent
CATEGORY = "train"

COMPETITION_DATA_DIR = BASE_DIR / "ALD-E-ImageMiner" / "icdar2026-competition-data"
CASE_DIR = COMPETITION_DATA_DIR / CATEGORY

DATA_DIR = BASE_DIR / "data"

STATE_FILE = BASE_DIR / "fol_state.json"
FOL_FILE = BASE_DIR / "fol.json"

# %%
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_ID,
    load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)
FastVisionModel.for_inference(model)  # Enable for inference!

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
PROMPT_TEMPLATE = """
<image>

Your task is to translate the provided scientific figure into a First-Order Logic (FOL) knowledge base and convert the accompanying question into a Z3-Solver query.

Question: {question}

[STRICT OUTPUT RULES]
1. NO MARKDOWN: Do not use ```python or any formatting fences. Output raw text only.
2. NO EXPLANATIONS: Start directly with 'from z3 import *'. No preamble or post-amble.
3. DATA FIDELITY: Every data point, trend, and axis limit visible in the image must be encoded as a s.add() constraint.

[Z3 SCHEMA SETUP]
You MUST use this foundational schema:
from z3 import *
s = Solver()

# Universe definitions
Panel = DeclareSort('Panel')
Series = DeclareSort('Series')

# Predicates & Functions
# axis(panel, type, label, unit)
axis = Function('axis', Panel, StringSort(), StringSort(), StringSort(), BoolSort())
# data_point(series, x_val, y_val)
data_point = Function('data_point', Series, RealSort(), RealSort(), BoolSort())
# metadata(panel, caption_info)
metadata = Function('metadata', Panel, StringSort(), BoolSort())

[TRANSLATION LOGIC]
1. DECLARE ENTITIES: Define panels (p_a, p_b) and series (s1, s2) using Const().
2. ENCODE TRUTH:
   - Extract x/y labels and units: s.add(axis(p_a, StringVal('x'), StringVal('temperature'), StringVal('C')))
   - Extract every visible data point: s.add(data_point(s1, 150.0, 1.2))
   - Encode trends: If a line is strictly increasing, add constraints for all x1, x2 where x1 < x2, y1 < y2.
3. FORMULATE QUERY:
   - Translate "{question}" into a Z3 variable (e.g., 'Ans').
   - If the question asks for a "maximum," use an Optimizer() or a loop to find the highest 'y' value among the data_point predicates.
   - For "Yes/No" questions, define a Bool 'ans_bool' and constrain it to the logic of the question.

[FINAL BLOCK]
The code must conclude with:
if s.check() == sat:
    m = s.model()
    # Logic to print the final answer in the requested format (Yes/No, Factoid, List, or Paragraph)
    print(f"Answer: {{m}}")
else:
    print("Unsatisfiable: Logic does not match image data.")
"""


# %%
def validate_fol(model_output):
    """
    Now actually captures stderr and stdout separately.
    """
    # 1. Clean markdown formatting
    code = model_output.replace("```python", "").replace("```", "").strip()

    output_buffer = io.StringIO()
    error_buffer = io.StringIO()

    # Define the Z3 environment
    exec_globals = {
        "z3": sys.modules["z3"],
        "Solver": Solver,
        "Int": Int,
        "Real": Real,
        "Bool": Bool,
        "Function": Function,
        "DeclareSort": DeclareSort,
        "RealSort": RealSort,
        "BoolSort": BoolSort,
        "Const": Const,
        "StringVal": StringVal,
        "sat": sat,
        "unsat": unsat,
    }

    # Redirect both stdout AND stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = output_buffer
        sys.stderr = error_buffer

        exec(code, exec_globals)

        return True, output_buffer.getvalue()

    except Exception:
        # If it fails, we combine the captured stderr with the traceback
        trace = traceback.format_exc()
        full_error = (
            f"Captured Stderr:\n{error_buffer.getvalue()}\n\nTraceback:\n{trace}"
        )
        return False, full_error

    finally:
        # ALWAYS restore the original streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def query_qwen(image, instruction, tokenizer, model):
    """
    Refactored to take specific image and instruction strings.
    """
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,  # Increased for Z3 code blocks
        use_cache=True,
        temperature=0.7,  # Lowered for more consistent code generation
    )

    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()


def reflect(image_data, initial_instruction, tokenizer, model, max_retries=3):
    """Uses a multi-turn chat history for error correction."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": initial_instruction},
            ],
        }
    ]

    for _ in range(max_retries):
        raw_output = query_qwen(image_data, messages, tokenizer, model)
        success, result = validate_fol(raw_output)

        if success:
            return raw_output  # Return the generated code on success

        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": raw_output}]}
        )

        error_msg = (
            f"Your previous Z3 code failed with the following traceback:\n\n{result}\n\n"
            f"Please fix the code. Ensure you still answer the original question and strictly follow the schema."
        )
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": error_msg}]}
        )

    return None


# %%
if STATE_FILE.exists():
    with open(STATE_FILE, "r") as f:
        saved_state = json.load(f)
    state = defaultdict(lambda: defaultdict(dict), saved_state)
    print(f"Loaded existing state from {STATE_FILE}. Resuming inference...")
else:
    state = defaultdict(lambda: defaultdict(dict))

# %%
json_files = list(CASE_DIR.rglob("*.json"))
pbar = tqdm(json_files, desc="Generating FOL")

for json_file in pbar:
    fullpath = str(json_file)
    if "images" not in fullpath or ".vscode" in fullpath:
        continue

    pbar.set_description(f"Processing {json_file}")

    with open(json_file, "r") as f:
        data = json.load(f)

    existing_answers = state.get(data["sample_id"], {})
    if any(existing_answers):
        continue

    sample_id = data.get("sample_id", json_file.stem)
    img_path = json_file.with_suffix(".jpg")
    assert img_path.exists(), f"{json_file.name} does not exist"

    # Open the full source image once
    full_img = Image.open(img_path.absolute())

    # Extract bounding box info
    bboxes = data.get("bbox", {})

    # Iterate through subfigures (a, b, etc.) present in the VQA data
    pbar_subs = tqdm(data.get("vqa", {}).items(), desc="Subfigures", leave=False)
    for sub_key, q_list in pbar_subs:
        # Skip if there's no bounding box for this subfigure
        if sub_key not in bboxes:
            warnings.warn(f"Subfigure {sub_key} missing bbox in {json_file.name}")
            continue

        # Get coordinates and crop
        box = bboxes[sub_key]
        left = box["x"]
        top = box["y"]
        right = left + box["width"]
        bottom = top + box["height"]

        # Create the sub-image crop
        sub_image = full_img.crop((left, top, right, bottom))

        # Process every question associated with this specific sub-figure
        pbar_qs = tqdm(q_list, desc="Questions", leave=False)
        for q_obj in pbar_qs:
            question_text = q_obj.get("question") or q_obj.get("questions")

            # Check filesystem state to see if we already processed this exact question
            if question_text in state[sample_id].get(sub_key, {}):
                continue

            pbar.set_description(f"Processing {sample_id} - {sub_key}")

            instruction = PROMPT_TEMPLATE.format(
                question=question_text,
                question_type=q_obj.get("question_type", ""),
                answer_type=q_obj.get("answer_type", ""),
            )

            # Run self-correction
            fol_logic = reflect(sub_image, instruction, tokenizer, model)

            # Update state and persist to filesystem immediately
            if not fol_logic:
                warnings.warn(
                    f"FOL generation failed for {sample_id}, {sub_key}, {question_text}"
                )

            state[sample_id][sub_key][question_text] = fol_logic
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)

# %%
with open(FOL_FILE, "w") as f:
    json.dump(state, f, indent=2)

print(f"Saved {len(state)} samples to {FOL_FILE}")

if STATE_FILE.exists():
    STATE_FILE.unlink()
