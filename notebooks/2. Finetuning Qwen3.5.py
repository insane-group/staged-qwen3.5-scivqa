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
from unsloth.trainer import UnslothVisionDataCollator

# Force unsloth to be on top
import re
import json
from pathlib import Path

import torch
from PIL import Image
import warnings
from tqdm.auto import tqdm
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer
import pandas as pd
import matplotlib.pyplot as plt

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

LORA_CHECKPOINT = f"Sci-ImageMiner-{MODEL_ID.split('/')[1]}-LORA"

BASE_DIR = Path.cwd().parent
CATEGORY = "train"

COMPETITION_DATA_DIR = BASE_DIR / "ALD-E-ImageMiner" / "icdar2026-competition-data"
CASE_DIR = COMPETITION_DATA_DIR / CATEGORY

DATA_DIR = BASE_DIR / "data"

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
PROMPT_YES_NO = """
<image>

Answer the following scientific figure question by reasoning strictly over the information visible in the figure.

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

Answer the following scientific figure question by reasoning strictly over the information visible in the figure.

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

Answer the following scientific figure question by reasoning strictly over the information visible in the figure.

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

Answer the following scientific figure question by reasoning strictly over the information visible in the figure.

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
def clean_answer(raw_answer: str, expected_type: str) -> tuple[str, bool]:
    """
    Cleans the answer and checks if it conforms to the expected type.
    Returns: (cleaned_answer, is_valid_format)
    """
    if not raw_answer or not isinstance(raw_answer, str):
        return "", False

    # Basic cleaning
    cleaned = raw_answer.strip()

    if expected_type == "Yes/No":
        ans_lower = cleaned.lower()
        # Use regex word boundaries (\b) to avoid matching "eyes", "note", "nothing", etc.
        if re.search(r"\byes\b", ans_lower):
            return "Yes", True
        elif re.search(r"\bno\b", ans_lower):
            return "No", True
        return cleaned, False

    elif expected_type == "List":
        # Split by comma, strip whitespace, and drop any empty strings
        elements = [item.strip() for item in cleaned.split(",") if item.strip()]
        return ", ".join(elements), len(elements) > 0

    elif expected_type == "Factoid":
        # Remove trailing punctuation often mistakenly added by LLMs to short terms
        cleaned_factoid = cleaned.rstrip(".!? \n")
        return cleaned_factoid, len(cleaned_factoid) > 0

    elif expected_type == "Paragraph":
        # Replace multiple spaces/newlines with single spaces for a clean paragraph
        cleaned_para = re.sub(r"\s+", " ", cleaned).strip()
        sentences = [s for s in re.split(r"[.!?]+", cleaned_para) if s.strip()]
        return cleaned_para, len(sentences) >= 1

    return cleaned, True


def convert_to_conversation(prompt: str, image: Image, response: str) -> dict:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
    ]
    return {"messages": conversation}


def load_dataset(case_dir: Path) -> list[dict]:
    samples = []
    json_files = list(case_dir.rglob("*.json"))
    pbar = tqdm(json_files, desc="Processing Subfigures")

    # Simple trackers for your sanity
    valid_count = 0
    invalid_count = 0

    for json_file in pbar:
        fullpath = str(json_file)
        if "images" not in fullpath or ".vscode" in fullpath:
            continue

        pbar.set_description(f"Processing {json_file.name}")

        with open(json_file, "r") as f:
            data = json.load(f)

        img_path = json_file.with_suffix(".jpg")
        assert img_path.exists(), f"{json_file.name} does not exist"

        # Open the full source image once
        full_img = Image.open(img_path.absolute())

        # Extract bounding box info
        bboxes = data.get("bbox", {})

        # Iterate through subfigures (a, b, etc.) present in the VQA data
        for sub_key, q_list in data.get("vqa", {}).items():
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
            for q_obj in q_list:
                question_text = q_obj.get("question") or q_obj.get("questions")
                question_type = q_obj.get("question_type", "")
                answer_type = q_obj.get("answer_type", "")

                human_prompt = PROMPTS[answer_type].format(
                    question=question_text,
                    question_type=question_type,
                )

                raw_response = q_obj.get("answer", "")

                # --- APPLY CLEANING HERE ---
                cleaned_response, is_valid = clean_answer(
                    raw_answer=raw_response, expected_type=answer_type
                )

                # Skip invalid formats to keep the fine-tuning dataset pristine
                if not is_valid:
                    invalid_count += 1
                    continue

                valid_count += 1

                # Pass the cropped sub_image and the CLEANED response
                sample = convert_to_conversation(
                    human_prompt, sub_image, cleaned_response
                )
                samples.append(sample)

    return samples, valid_count, invalid_count


# %% [markdown]
# Let's convert the dataset into the "correct" format for finetuning:

# %%
dataset, valid_count, invalid_count = load_dataset(CASE_DIR)

print("-" * 40)
print("📋 DATASET CREATION SUMMARY")
print(f"Added to dataset (Valid): {valid_count}")
print(f"Skipped (Invalid format): {invalid_count}")
print("-" * 40)

# %% [markdown]
# We look at how the conversations are structured for the first example:

# %%
dataset[0]["messages"][0]["content"][1]["image"]

# %%
dataset[0]["messages"]


# %%
def calculate_token_stats(dataset_samples, processor, max_samples=500):
    stats = []

    # Limit samples for speed during exploration
    samples_to_process = (
        dataset_samples[:max_samples] if max_samples else dataset_samples
    )

    for sample in tqdm(samples_to_process, desc="Calculating token lengths"):
        messages = sample["messages"]

        # 1. Extract the image and assistant text
        image = messages[0]["content"][1]["image"]
        assistant_text = messages[1]["content"][0]["text"]

        # 2. Calculate MAX_NEW_TOKENS (Assistant output only)
        # Access the underlying text tokenizer inside the processor
        assistant_tokens = len(
            processor.tokenizer.encode(assistant_text, add_special_tokens=False)
        )

        # 3. Calculate max_length (Entire sequence: Image + Prompt + Formatting + Answer)
        # Apply the chat template to the full multi-turn conversation
        full_text = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )

        # Pass the image and the formatted text using keyword arguments
        inputs = processor(
            text=full_text, images=image, add_special_tokens=False, return_tensors="pt"
        )
        total_tokens = inputs["input_ids"].shape[1]

        stats.append(
            {
                "assistant_tokens": assistant_tokens,
                "total_tokens": total_tokens,
                "image_width": image.width,
                "image_height": image.height,
            }
        )

    return pd.DataFrame(stats)


# %%
# Run the analysis
df_tokens = calculate_token_stats(dataset, tokenizer, max_samples=200)

# Display statistics
print("\n--- Token Length Statistics ---")
display(
    df_tokens[["assistant_tokens", "total_tokens"]].describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
    )
)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
df_tokens["assistant_tokens"].hist(
    bins=30, ax=axes[0], color="salmon", edgecolor="black"
)
axes[0].set_title("Assistant Tokens (for MAX_NEW_TOKENS)")
axes[0].set_xlabel("Token Count")

df_tokens["total_tokens"].hist(bins=30, ax=axes[1], color="skyblue", edgecolor="black")
axes[1].set_title("Total Sequence Tokens (for max_length)")
axes[1].set_xlabel("Token Count")
plt.show()

# %% [markdown]
# Let's first see before we do any finetuning what the model outputs for the first example!

# %%
FastVisionModel.for_inference(model)  # Enable for inference!

image = dataset[0]["messages"][0]["content"][1]["image"]
instruction = dataset[0]["messages"][0]["content"][0]["text"]

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

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=MAX_NEW_TOKENS,
    use_cache=True,
    temperature=TEMPERATURE,
    min_p=MIN_P,
    top_p=TOP_P,
    top_k=TOP_K,
)

# %% [markdown]
# <a name="Training"></a>
# ### 🚀 Training the model
#
# Now let's train our model. We now add LoRA adapters for parameter efficient finetuning - this allows us to only efficiently train 1% of all parameters.
#
# We do 60 steps to speed things up, but we can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
#
# > We could also use `DPOTrainer` and `GRPOTrainer` for reinforcement learning!!

# %%
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=16,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# %%
FastVisionModel.for_training(model)  # Enable for training!

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(
        model, tokenizer, max_seq_length=4096, resize="max"
    ),  # https://github.com/unslothai/unsloth/issues/2764
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        # max_steps=30,
        num_train_epochs=NUM_TRAIN_EPOCHS,  # Set this instead of max_steps for full training runs
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # For Weights and Biases
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=4096,
    ),
)

# %%
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# %%
trainer_stats = trainer.train()

# %%
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# %% [markdown]
# <a name="Inference"></a>
# ### 🍴 Inference
#
# Let's run the model! You can change the instruction and input - leave the output blank!
#
# We use `min_p = 0.1` and `temperature = 1.5`. Read this [Tweet](https://x.com/menhguin/status/1826132708508213629) for more information on why.

# %%
FastVisionModel.for_inference(model)  # Enable for inference!

image = dataset[0]["messages"][0]["content"][1]["image"]
instruction = dataset[0]["messages"][0]["content"][0]["text"]

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

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=MAX_NEW_TOKENS,
    use_cache=True,
    temperature=TEMPERATURE,
    min_p=MIN_P,
    top_p=TOP_P,
    top_k=TOP_K,
)

# %% [markdown]
# <a name="Saving"></a>
# ### 💾 Saving, loading finetuned models
#
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
#
# > This **ONLY** saves the LoRA adapters, and not the full model.

# %%
model.save_pretrained(LORA_CHECKPOINT)  # Local saving
tokenizer.save_pretrained(LORA_CHECKPOINT)
# model.push_to_hub(f"billsioros/{LORA_CHECKPOINT}", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub(f"billsioros/{LORA_CHECKPOINT}", token = "YOUR_HF_TOKEN") # Online saving

# %% [markdown]
# Let's now load the LoRA adapters we just saved for inference!

# %%
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=LORA_CHECKPOINT,
    load_in_4bit=True,  # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model)  # Enable for inference!

image = dataset[0]["messages"][0]["content"][1]["image"]
instruction = dataset[0]["messages"][0]["content"][0]["text"]

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

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=MAX_NEW_TOKENS,
    use_cache=True,
    temperature=TEMPERATURE,
    min_p=MIN_P,
    top_p=TOP_P,
    top_k=TOP_K,
)
