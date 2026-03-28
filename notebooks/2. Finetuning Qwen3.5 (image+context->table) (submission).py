# %% [markdown]
# # Finetuning Qwen3.5 (image+context->table)
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
from collections import defaultdict

# %%
MODEL_ID = "unsloth/Qwen3.5-9B"

MAX_NEW_TOKENS = 256
NUM_TRAIN_EPOCHS = 5

# https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune#official-recommended-settings
ENABLE_THINKING = False
TEMPERATURE = 0.7
MIN_P = 0.01
TOP_P = 0.8
TOP_K = 20

LORA_CHECKPOINT = f"Sci-ImageMiner-{MODEL_ID.split('/')[1]}-LORA-EXTRACTION"

BASE_DIR = Path.cwd().parent
CATEGORY = "dev"

COMPETITION_DATA_DIR = BASE_DIR / "ALD-E-ImageMiner" / "icdar2026-competition-data"
CASE_DIR = COMPETITION_DATA_DIR / CATEGORY

DATA_DIR = BASE_DIR / "data"
STATE_FILE = BASE_DIR / f"submission_finetuning_extraction_{CATEGORY}_state.json"
SUBMISSION_PATH = BASE_DIR / f"submission_finetuning_extraction_{CATEGORY}.json"

# %%
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_ID,
    load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)

# %% [markdown]
# <a name="Data"></a>
# ### 🧪 Data Preparation
#
# To convert our Sci-ImageMiner VQA data into the format required by Qwen2-VL (specifically for use with Unsloth), we need to restructure the data into a "messages" format.
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

Additional context from the original paper:
{context}

Extract the structured data represented in the scientific chart by reconstructing it as a Markdown table.

Strict requirements:

1. Identify all relevant chart regions (axes, ticks, labels, legends, plotted data).
2. Determine the field/column names from axis labels, legends, or annotations.
3. Reconstruct the underlying tabular structure represented in the chart.
4. Extract all visible textual and numerical values as accurately as possible.
5. If multiple series exist, include them as separate columns or clearly distinguish them.
6. Preserve units exactly as shown in the figure.
7. Ignore decorative elements, schematics, arrows, and non-data graphics.
8. Do not infer or extrapolate missing values—only include explicitly visible data.
9. Ensure the table is complete, consistent, and machine-readable.

Output format:

- A valid Markdown table only.
- First row: column names.
- Second row: separator (|---|---|...|).
- Subsequent rows: extracted data values.
- No explanations or extra text.

Example:

| Time (s) | Mass Change (ng/cm²) |
|---|---|
| 0 | 0 |
| 2000 | -500 |
| 4000 | -1000 |
| 6000 | -1500 |
"""


# %%
def convert_to_conversation(prompt: str, image: Image, **metadata: dict) -> dict:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]
    return {"messages": conversation, "meta": metadata}


def get_paper_context(json_file_path, window_size=2):
    """
    Finds the parent content.json, extracts the image caption, and
    grabs a sliding window of text blocks (e.g., 2 before, 2 after)
    surrounding the image for highly targeted context.
    """
    # Navigate up from .../16/images/fig_2.json to .../16/content.json
    content_json_path = json_file_path.parent.parent / "content.json"

    assert content_json_path.exists(), f"{content_json_path}"

    # The image path as it appears in content.json (e.g., "images/fig_2.jpg")
    target_img_path = f"images/{json_file_path.stem}.jpg"

    with open(content_json_path, "r", encoding="utf-8") as f:
        content_data = json.load(f)

    img_index = -1
    caption_text = ""

    # Locate the image block in the flat JSON array
    for idx, block in enumerate(content_data):
        if block.get("type") == "image" and block.get("img_path") == target_img_path:
            img_index = idx
            if "img_caption" in block and block["img_caption"]:
                caption_text = " ".join(block["img_caption"])
            break

    if img_index == -1:
        return "Specific context not found for this image."

    # Gather text blocks BEFORE the image
    text_before = []
    for i in range(img_index - 1, -1, -1):
        block = content_data[i]
        if block.get("type") == "text" and "text" in block:
            text_before.insert(0, block["text"])  # Keep chronological order
            if len(text_before) == window_size:
                break

    # Gather text blocks AFTER the image
    text_after = []
    for i in range(img_index + 1, len(content_data)):
        block = content_data[i]
        if block.get("type") == "text" and "text" in block:
            text_after.append(block["text"])
            if len(text_after) == window_size:
                break

    # Assemble the final context string
    context_blocks = []
    if caption_text:
        context_blocks.append(f"Image Caption: {caption_text}")

    context_blocks.extend(text_before)
    context_blocks.extend(text_after)

    return "\n\n".join(context_blocks)


def load_test_dataset(case_dir: Path) -> list[dict]:
    samples = []
    json_files = list(case_dir.rglob("*.json"))
    pbar = tqdm(json_files, desc="Processing Subfigures")

    for json_file in pbar:
        fullpath = str(json_file)
        if (
            "content.json" in json_file.name
            or "images" not in fullpath
            or ".vscode" in fullpath
        ):
            continue

        pbar.set_description(f"Processing {json_file}")

        with open(json_file, "r") as f:
            data = json.load(f)

        img_path = json_file.with_suffix(".jpg")
        assert img_path.exists(), f"{json_file.name} does not exist"

        # Open the full source image once
        full_img = Image.open(img_path.absolute())
        context = get_paper_context(json_file)

        # Extract bounding box info
        bboxes = data.get("bbox", {})

        # Iterate through subfigures (a, b, etc.) present in the VQA data
        for sub_key, table in data.get("data_extraction", {}).items():
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

            # Process Data Extraction associated with this sub-figure
            instruction = PROMPT_TEMPLATE.format(context=context)
            sample = convert_to_conversation(
                instruction,
                sub_image,
                sample_id=data["sample_id"],
                sub_fig=sub_key,
            )
            samples.append(sample)

    return samples


# %% [markdown]
# Let's convert the dataset into the "correct" format for finetuning:

# %%
dataset = load_test_dataset(CASE_DIR)

# %% [markdown]
# We look at how the conversations are structured for the first example:

# %%
dataset[0]["messages"][0]["content"][1]["image"]

# %%
dataset[0]["messages"]

# %% [markdown]
# <a name="Submission"></a>
# ### 📜 Creating our submission
#
# Let's now create our submission! First, we must load the LoRA adapters we saved for inference!

# %%
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=LORA_CHECKPOINT,
    load_in_4bit=True,  # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model)  # Enable for inference!

# %%
state = defaultdict(dict)

if STATE_FILE.exists():
    with open(STATE_FILE, "r") as f:
        saved_state = json.load(f)

    state = defaultdict(
        dict,
        {k: defaultdict(list, v) for k, v in saved_state.items()},
    )
    print(f"Loaded existing state from {STATE_FILE}. Resuming inference...")
else:
    state = defaultdict(dict)

# %%
for sample in tqdm(dataset, desc="Running Inference"):
    meta = sample["meta"]

    existing_answers = state.get(meta["sample_id"], {}).get(meta["sub_fig"], [])
    if any(existing_answers):
        continue

    messages_content = sample["messages"][0]["content"]

    image = messages_content[1]["image"]
    instruction = messages_content[0]["text"]

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
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        temperature=TEMPERATURE,
        min_p=MIN_P,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    # Decode only the newly generated tokens
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()

    state[meta["sample_id"]][meta["sub_fig"]] = generated

    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# %%
submission = []
for sample_id, sub_figs in state.items():
    submission.append(
        {
            "sample_id": sample_id,
            "data_extraction": {
                sub_fig: answer for sub_fig, answer in sub_figs.items()
            },
        }
    )

with SUBMISSION_PATH.open("w") as f:
    json.dump(submission, f, indent=2)
print(f"Saved {len(submission)} samples to {SUBMISSION_PATH}")
