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

# https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune#official-recommended-settings
ENABLE_THINKING = False
TEMPERATURE = 0.7
MIN_P = 0.01
TOP_P = 0.8
TOP_K = 20

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
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
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

[TASK]
You are a formal logic extractor for scientific figures. Convert the visual evidence in the image and the question into a discrete Z3 reasoning program.
The solver must reason ONLY from explicitly extracted visual anchor points.
[STRICT RULES]

NO QUANTIFIERS
Do NOT use ForAll or Exists.
DISCRETE VISUAL ANCHORS
All numeric reasoning must rely on explicitly extracted coordinates from the figure.
DOMAIN BOUNDS
All numeric reasoning must remain within the axis range visible in the plot.
RELATIONAL GROUNDING
Any relational predicate must also be backed by numeric anchor inequalities.
LEGEND GROUNDING
Every visual series must be mapped to a material or category label from the legend.
PANEL STRUCTURE
If the figure contains multiple panels, explicitly link each series to a panel.
FINITE SEARCH SPACE
When solving for maxima/minima, restrict candidate values using Or() over the extracted anchor x-values.
[Z3 SCHEMA]
from z3 import *
s = Solver()

---------- CORE SORTS ----------
Series = DeclareSort('Series')
Material = DeclareSort('Material')
Panel = DeclareSort('Panel')
Category = DeclareSort('Category')
---------- IDENTIFICATION ----------
material_name = Function('material_name', Material, StringSort())
series_material = Function('series_material', Series, Material)
series_panel = Function('series_panel', Series, Panel)
category_name = Function('category_name', Category, StringSort())

---------- NUMERIC DATA ----------
anchor points extracted from figure
val = Function('val', Series, RealSort(), RealSort())

categorical values (bar charts etc.)
cat_val = Function('cat_val', Series, Category, RealSort())

---------- RELATIONAL PREDICATES ----------
higher_at = Function('higher_at', Series, Series, RealSort(), BoolSort())
increasing_between = Function('increasing_between', Series, RealSort(), RealSort(), BoolSort())
peak_at = Function('peak_at', Series, RealSort(), BoolSort())

---------- AXIS DOMAIN ----------
xmin = Real('xmin')
xmax = Real('xmax')
[VISUAL EXTRACTION]

Identify panels → p1, p2 …
Identify series → s1, s2 …
Map each series to its legend material/category name.
Example:
m1 = Const('m1', Material)
s1 = Const('s1', Series)
p1 = Const('p1', Panel)
s.add(material_name(m1) == StringVal("Al2O3"))
s.add(series_material(s1, m1))
s.add(series_panel(s1, p1))
[ANCHOR DATA EXTRACTION]
Extract at least 5 anchor points per series:
• minimum x
• maximum x
• peaks
• intersections
• distinctive points
Example:
s.add(val(s1,200.0) == 1.1)
s.add(val(s1,250.0) == 1.4)
s.add(val(s1,300.0) == 1.2)
[DOMAIN CONSTRAINT]
Example:
s.add(xmin == 150)
s.add(xmax == 350)
[RELATIONAL DEFINITIONS]
Relational predicates must be supported by numeric anchors.
Example:
s.add(higher_at(s1,s2,300) == (val(s1,300) > val(s2,300)))
Example increasing trend:
s.add(increasing_between(s1,200,250))
s.add(val(s1,250) > val(s1,200))
Example peak:
s.add(peak_at(s1,250))
s.add(val(s1,250) > val(s1,200))
s.add(val(s1,250) > val(s1,300))
[CATEGORICAL DATA (BAR CHARTS)]
Example:
c1 = Const('c1', Category)
s.add(category_name(c1) == StringVal("Precursor A"))
s.add(cat_val(s1,c1) == 1.3)
[QUESTION MODELING]
Question: {question}
Answer type: {answer_type}
Define exactly one answer variable.
Yes/No:
AnsBool = Bool('AnsBool')
Factoid:
AnsString = String('AnsString')
Numeric:
AnsReal = Real('AnsReal')
List:
Use Boolean flags per material.
Example:
in_list_m1 = Bool('in_list_m1')
s.add(in_list_m1 == (val(s1,250) > 1.0))
Paragraph:
Solve for the underlying fact(s) first (peak location, value difference, etc.).
[FINITE MAXIMUM SEARCH EXAMPLE]
AnsReal = Real('AnsReal')
s.add(Or(
AnsReal == 200,
AnsReal == 250,
AnsReal == 300
))
s.add(val(s1,AnsReal) >= val(s1,200))
s.add(val(s1,AnsReal) >= val(s1,250))
s.add(val(s1,AnsReal) >= val(s1,300))
[FINAL EXECUTION]
if s.check() == sat:

m = s.model()

if "{answer_type}" == "Yes/No":
    print("Final Answer:", bool(m[AnsBool]))

elif "{answer_type}" == "Factoid":
    print("Final Answer:", m[AnsString])

elif "{answer_type}" == "Numeric":
    print("Final Answer:", m[AnsReal])

elif "{answer_type}" == "List":
    results = [m[material_name(material)].as_string() for material in [m1, m2, m3...] if is_true(m[in_list_material])]
    print("Final Answer:", ", ".join(results))

else:
    print(f"Logic Result: Max value is {{m[val(s1, AnsReal)]}} at {{m[AnsReal]}}")
else:
    print("Unsatisfiable: extracted visual facts contradict constraints.")
"""


# %%
def validate_fol(code):
    """
    Now actually captures stderr and stdout separately.
    """
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

    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=ENABLE_THINKING
    )
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,  # Increased for Z3 code blocks
        use_cache=True,
        temperature=TEMPERATURE,
        min_p=MIN_P,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    model_output = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()

    return model_output.replace("```python", "").replace("```", "").strip()


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
