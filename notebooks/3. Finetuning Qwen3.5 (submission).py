# %% [markdown]
# # Creating our submission
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
LORA_CHECKPOINT = "Sci-ImageMiner-Qwen3.5-0.8B-LORA"
ENABLE_THINKING = True
MAX_NEW_TOKENS = 256

# https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune#official-recommended-settings
if ENABLE_THINKING:
    TEMPERATURE = 0.6
    MIN_P = 0.0
    TOP_P = 0.95
    TOP_K = 20
else:
    TEMPERATURE = 0.7
    MIN_P = 0.01
    TOP_P = 0.8
    TOP_K = 20

BASE_DIR = Path.cwd().parent
CATEGORY = "dev"

COMPETITION_DATA_DIR = BASE_DIR / "ALD-E-ImageMiner" / "icdar2026-competition-data"
CASE_DIR = COMPETITION_DATA_DIR / CATEGORY

DATA_DIR = BASE_DIR / "data"
STATE_FILE = BASE_DIR / f"submission_finetuning_{CATEGORY}_state.json"
SUBMISSION_PATH = BASE_DIR / f"submission_finetuning_{CATEGORY}.json"

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

Answer the following scientific figure question by reasoning strictly over the information visible in the figure.

Question type: {question_type}
Answer type: {answer_type}
Question: {question}

Strict requirements:

1. Detect each distinct plot (subfigure a, b, c, etc.).
2. Process only valid chart regions (axes, ticks, labels, legends, plotted data).
3. Ignore decorative graphics, schematics, arrows, background elements, and repeated fragments.
4. Answer based only on visible chart data and annotations.
5. Do not speculate beyond the visual evidence.
6. Strictly follow the answer type format:
   - "Yes/No": respond with only "yes" or "no".
   - "Factoid": concise term or short phrase only.
   - "List": comma-separated values only (order-insensitive).
   - "Paragraph": at least three sentences of explanation.
7. For comparative or trend questions, reference only observable relationships.
8. No stylistic commentary or extraneous explanations.

Output your answer as plain text only, with no JSON, no labels, no code fences, and no surrounding text.
"""


# %%
def load_test_dataset(case_dir: Path) -> list[dict]:
    samples = []
    json_files = list(case_dir.rglob("*.json"))
    pbar = tqdm(json_files, desc="Converting Test to Qwen Format")

    for json_file in pbar:
        fullpath = str(json_file)
        if "images" not in fullpath or ".vscode" in fullpath:
            continue

        pbar.set_description(f"Processing {json_file.name}")

        with open(json_file, "r") as f:
            data = json.load(f)

        img_path = json_file.with_suffix(".jpg")
        if not img_path.exists():
            continue

        # 2. Open source image once per JSON file
        full_img = Image.open(img_path.absolute())
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

            for q_obj in q_list:
                question_text = q_obj.get("question") or q_obj.get("questions")
                question_type = q_obj.get("question_type", "")
                answer_type = q_obj.get("answer_type", "")

                human_prompt = PROMPT_TEMPLATE.format(
                    question=question_text,
                    question_type=question_type,
                    answer_type=answer_type,
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
                            "sample_id": data["sample_id"],
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
dataset = load_test_dataset(CASE_DIR)

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

if STATE_FILE.exists():
    STATE_FILE.unlink()
