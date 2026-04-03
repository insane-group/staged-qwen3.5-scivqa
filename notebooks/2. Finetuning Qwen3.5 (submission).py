# %% [markdown]
# # Finetuning Qwen3.5
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
import warnings
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict

# %%
MODEL_ID = "unsloth/Qwen3.5-9B"
MAX_NEW_TOKENS = 256
MAX_SEQUENCE_LENGTH = 4096

# https://unsloth.ai/docs/models/qwen3.5#recommended-settings
ENABLE_THINKING = False
TEMPERATURE = 0.7
MIN_P = 0.01
TOP_P = 0.8
TOP_K = 20

LORA_CHECKPOINT = f"Sci-ImageMiner-{MODEL_ID.split('/')[1]}-LORA"

BASE_DIR = Path.cwd().parent
CATEGORY = "test"

COMPETITION_DATA_DIR = BASE_DIR / "ALD-E-ImageMiner" / "icdar2026-competition-data"
CASE_DIR = COMPETITION_DATA_DIR / CATEGORY

STATE_FILE = BASE_DIR / f"submission_finetuning_{CATEGORY}_state.json"
SUBMISSION_PATH = BASE_DIR / f"submission_finetuning_{CATEGORY}.json"

SUMMARY_CACHE_PATH = BASE_DIR / f"submission_finetuning_summary_{CATEGORY}_state.json"
EXTRACTION_CACHE_PATH = (
    BASE_DIR / f"submission_finetuning_extraction_{CATEGORY}_state.json"
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
PROMPT_YES_NO = """
<image>

[SUMMARY]
{summary}

[TABLE]
{table}

Additional context from the original paper:
{context}

Answer the following scientific figure question by reasoning strictly over the information visible in the figure and the provided context.

Question type: {question_type}
Question: {question}

Strict requirements:
1. Identify the main variables shown (axes, units, and any legend information).
2. Ignore decorative graphics, schematics, arrows, and background elements.
3. Use the provided caption/context only to support interpretation when necessary.
4. Do not speculate or infer beyond what is visually supported.
5. Output plain text only, with no JSON, no code fences, and no surrounding explanatory text.
6. Output your answer STRICTLY as "Yes" or "No" (title case).

Example:
Yes
"""

PROMPT_FACTOID = """
<image>

[SUMMARY]
{summary}

[TABLE]
{table}

Additional context from the original paper:
{context}

Answer the following scientific figure question by reasoning strictly over the information visible in the figure and the provided context.

Question type: {question_type}
Question: {question}

Strict requirements:
1. Identify the main variables shown (axes, units, and any legend information).
2. Ignore decorative graphics, schematics, arrows, and background elements.
3. Use the provided caption/context only to support interpretation when necessary.
4. Do not speculate or infer beyond what is visually supported.
5. Output plain text only, with no JSON, no code fences, and no surrounding explanatory text.
6. Output your answer STRICTLY as a concise term or short phrase.

Example:
The feature corresponds to an interband electronic transition or optical absorption edge.
"""

PROMPT_LIST = """
<image>

[SUMMARY]
{summary}

[TABLE]
{table}

Additional context from the original paper:
{context}

Answer the following scientific figure question by reasoning strictly over the information visible in the figure and the provided context.

Question type: {question_type}
Question: {question}

Strict requirements:
1. Identify the main variables shown (axes, units, and any legend information).
2. Ignore decorative graphics, schematics, arrows, and background elements.
3. Use the provided caption/context only to support interpretation when necessary.
4. Do not speculate or infer beyond what is visually supported.
5. Output plain text only, with no JSON, no bullet points, no numbered lists, no code fences, and no surrounding explanatory text.
6. Output your answer STRICTLY as comma-separated values (order-insensitive).

Example:
Absence of pits or voids, Smooth and continuous surface, Lack of corrosive attack patterns, Reduced by-product interaction with copper
"""

PROMPT_PARAGRAPH = """
<image>

[SUMMARY]
{summary}

[TABLE]
{table}

Additional context from the original paper:
{context}

Answer the following scientific figure question by reasoning strictly over the information visible in the figure and the provided context.

Question type: {question_type}
Question: {question}

Strict requirements:
1. Identify the main variables shown (axes, units, and any legend information).
2. Ignore decorative graphics, schematics, arrows, and background elements.
3. Use the provided caption/context only to support interpretation when necessary.
4. Do not speculate or infer beyond what is visually supported.
5. Output plain text only, with no JSON, no bullet points, no numbered lists, no code fences, and no surrounding explanatory text.
6. Output your answer STRICTLY as a paragraph containing at least 3 sentences providing an explanatory answer.

Example:
Multiple well-defined interfaces, Alternating high-contrast layers, Disruption of continuous grain boundaries, Uniform nanometer-scale layer thickness
"""

# %%
PROMPTS = {
    "Yes/No": PROMPT_YES_NO,
    "Factoid": PROMPT_FACTOID,
    "List": PROMPT_LIST,
    "Paragraph": PROMPT_PARAGRAPH,
}


# %%
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


def load_test_dataset(
    case_dir: Path, summary_cache: dict, extraction_cache: dict
) -> list[dict]:
    samples = []
    json_files = list(case_dir.rglob("*.json"))
    pbar = tqdm(json_files, desc="Converting Test to Qwen Format")

    for json_file in pbar:
        fullpath = str(json_file)
        if (
            "content.json" in json_file.name
            or "images" not in fullpath
            or ".vscode" in fullpath
        ):
            continue

        pbar.set_description(f"Processing {json_file.name}")

        with open(json_file, "r") as f:
            data = json.load(f)

        sample_id = data["sample_id"]
        img_path = json_file.with_suffix(".jpg")
        if not img_path.exists():
            continue

        # 2. Open source image once per JSON file
        full_img = Image.open(img_path.absolute())
        context = get_paper_context(json_file)
        bboxes = data.get("bbox", {})

        # 3. Iterate through subfigures (a, b, etc.)
        for sub_fig, q_list in data.get("vqa", {}).items():
            # 4. Check for bounding box and crop
            if sub_fig not in bboxes:
                warnings.warn(f"Subfigure {sub_fig} missing bbox in {json_file.name}")
                continue

            box = bboxes[sub_fig]
            left = box["x"]
            top = box["y"]
            right = left + box["width"]
            bottom = top + box["height"]

            sub_image = full_img.crop((left, top, right, bottom))

            summary = summary_cache.get(sample_id, {}).get(sub_fig)
            table = extraction_cache.get(sample_id, {}).get(sub_fig)

            for q_obj in q_list:
                question_text = q_obj.get("question") or q_obj.get("questions")
                question_type = q_obj.get("question_type", "")
                answer_type = q_obj.get("answer_type", "")

                human_prompt = PROMPTS[answer_type].format(
                    question=question_text,
                    question_type=question_type,
                    context=context,
                    summary=summary
                    if summary is not None
                    else "N/A",  # Aligned with training code fallback
                    table=table
                    if table is not None
                    else "N/A",  # Aligned with training code fallback
                )

                # 5. Format for inference (User role only)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": human_prompt},
                            {"type": "image", "image": sub_image},
                        ],
                    }
                ]

                samples.append(
                    {
                        "messages": conversation,
                        "meta": {
                            "sample_id": sample_id,
                            "sub_fig": sub_fig,
                            "question_type": question_type,
                            "answer_type": answer_type,
                            "question": question_text,
                        },
                    }
                )

    return samples


# %% [markdown]
# Let's convert the dataset into the "correct" format for finetuning:

# %%
summary_cache = None
with open(SUMMARY_CACHE_PATH, "r") as f:
    summary_cache = json.load(f)

extraction_cache = None
with open(EXTRACTION_CACHE_PATH, "r") as f:
    extraction_cache = json.load(f)

dataset = load_test_dataset(CASE_DIR, summary_cache, extraction_cache)

# %% [markdown]
# We look at how the conversations are structured for the first example:

# %%
dataset[0]["messages"][0]["content"][1]["image"]

# %%
dataset[0]

# %% [markdown]
# <a name="Submission"></a>
# ### 📜 Creating our submission
#
# Let's now create our submission! First, we must load the LoRA adapters we saved for inference!

# %%
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=LORA_CHECKPOINT,
    load_in_4bit=True,  # Set to False for 16bit LoRA
    max_seq_length=MAX_SEQUENCE_LENGTH,  # Must match the max_length used during training
)
FastVisionModel.for_inference(model)  # Enable for inference!

# %%
state = defaultdict(lambda: defaultdict(list))

if STATE_FILE.exists():
    with open(STATE_FILE, "r") as f:
        saved_state = json.load(f)

    state = defaultdict(
        lambda: defaultdict(list),
        {k: defaultdict(list, v) for k, v in saved_state.items()},
    )
    print(f"Loaded existing state from {STATE_FILE}. Resuming inference...")
else:
    state = defaultdict(lambda: defaultdict(list))

# %%
for sample in tqdm(dataset, desc="Running Inference"):
    meta = sample["meta"]

    existing_answers = state.get(meta["sample_id"], {}).get(meta["sub_fig"], [])
    if any(ans.get("question") == meta["question"] for ans in existing_answers):
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

    state[meta["sample_id"]][meta["sub_fig"]].append(
        {
            "question_type": meta["question_type"],
            "question": meta["question"],
            "answer_type": meta["answer_type"],
            "answer": generated,
        }
    )

    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# %%
submission = []
for sample_id, sub_figs in state.items():
    submission.append(
        {
            "sample_id": sample_id,
            "vqa": {sub_fig: q_list for sub_fig, q_list in sub_figs.items()},
        }
    )

with SUBMISSION_PATH.open("w") as f:
    json.dump(submission, f, indent=2)
print(f"Saved {len(submission)} samples to {SUBMISSION_PATH}")
