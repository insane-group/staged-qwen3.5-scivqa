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

import json
from pathlib import Path

import torch
from PIL import Image
import warnings
from tqdm.auto import tqdm
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer

# %%
MODEL_ID = "unsloth/Qwen3.5-0.8B"
MAX_NEW_TOKENS = 256
NUM_TRAIN_EPOCHS = 2

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

    for json_file in pbar:
        fullpath = str(json_file)
        if "images" not in fullpath or ".vscode" in fullpath:
            continue

        pbar.set_description(f"Processing {json_file}")

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

                human_prompt = PROMPT_TEMPLATE.format(
                    question=question_text,
                    question_type=question_type,
                    answer_type=answer_type,
                )

                gt_response = q_obj.get("answer", "")

                # Pass the cropped sub_image instead of the full_img
                sample = convert_to_conversation(human_prompt, sub_image, gt_response)
                samples.append(sample)

    return samples


# %% [markdown]
# Let's convert the dataset into the "correct" format for finetuning:

# %%
dataset = load_dataset(CASE_DIR)

# %% [markdown]
# We look at how the conversations are structured for the first example:

# %%
dataset[0]["messages"][0]["content"][1]["image"]

# %%
dataset[0]["messages"]

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
    data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
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
        max_length=2048,
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
