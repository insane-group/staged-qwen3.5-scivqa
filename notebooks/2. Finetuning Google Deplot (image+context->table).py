# %% [markdown]
# # Finetuning Google DePlot for Chart-to-Table Extraction (image+context->table)
#
# > Originally, adapted from [image_captioning_pix2struct.ipynb](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb)
#
# This notebook adapts the data loading pipeline from the Qwen3.5 example and integrates it with the Pix2Struct finetuning loop. It trains the `google/deplot` model to extract markdown tables from chart images, utilizing the surrounding paper context as a text prompt.

# %%
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
import copy
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

# %%
MODEL_ID = "google/deplot"

# Pix2Struct/DePlot specific hyperparameters
MAX_PATCHES = 1024  # Resolution equivalent (e.g., 1024, 2048)
MAX_NEW_TOKENS = 2048  # Covers the 99th percentile (1806) and most of the max (2603).
EPOCHS = 5000
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-05
BATCH_SIZE = 2
PATIENCE = EPOCHS  # Number of epochs to wait for improvement before stopping
NUM_WARMUP_PERCENTAGE = 0.025

BASE_DIR = Path.cwd().parent
CHECKPOINT_PATH = BASE_DIR / "deplot-finetuned"

CATEGORY = "train"
CATEGORY_DEV = "dev"

COMPETITION_DATA_DIR = BASE_DIR / "ALD-E-ImageMiner" / "icdar2026-competition-data"
CASE_DIR = COMPETITION_DATA_DIR / CATEGORY
DEV_DIR = COMPETITION_DATA_DIR / CATEGORY_DEV

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


def load_dataset(case_dir: Path) -> list[dict]:
    samples = []
    json_files = list(case_dir.rglob("*.json"))
    pbar = tqdm(json_files, desc="Processing Subfigures")

    for json_file in pbar:
        fullpath = str(json_file)
        if "images" not in fullpath or ".vscode" in fullpath:
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        img_path = json_file.with_suffix(".jpg")
        if not img_path.exists():
            continue

        full_img = Image.open(img_path.absolute())
        context = get_paper_context(json_file)
        bboxes = data.get("bbox", {})

        for sub_key, gt_table in data.get("data_extraction", {}).items():
            if not gt_table.strip() or sub_key not in bboxes:
                continue

            box = bboxes[sub_key]
            sub_image = full_img.crop(
                (box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"])
            )

            # Format prompt and store sample
            instruction = PROMPT_TEMPLATE.format(context=context)
            samples.append(
                {"image": sub_image.copy(), "prompt": instruction, "target": gt_table}
            )

    return samples


# %% [markdown]
# Let's convert the dataset into the "correct" format for finetuning:

# %%
raw_train_data = load_dataset(CASE_DIR)

# %% [markdown]
# We look at how the conversations are structured for the first example:

# %%
raw_train_data[0]

# %%
raw_dev_data = load_dataset(DEV_DIR)

# %%
processor = Pix2StructProcessor.from_pretrained(MODEL_ID)
model = Pix2StructForConditionalGeneration.from_pretrained(MODEL_ID)


# %%
class DeplotDataset(Dataset):
    def __init__(self, dataset, processor, max_patches):
        self.dataset = dataset
        self.processor = processor
        self.max_patches = max_patches

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Pass both image and text (prompt) to the processor
        encoding = self.processor(
            images=item["image"],
            text=item["prompt"],
            return_tensors="pt",
            add_special_tokens=True,
            max_patches=self.max_patches,
        )

        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["target"] = item[
            "target"
        ]  # The actual markdown table we want to generate
        return encoding


def collator(batch):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    targets = [item["target"] for item in batch]

    # FIX: Use processor.tokenizer directly to avoid triggering the image processor
    text_inputs = processor.tokenizer(
        text=targets,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True,
        max_length=MAX_NEW_TOKENS,
        truncation=True,
    )

    new_batch["labels"] = text_inputs.input_ids

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch


# %%
train_dataset = DeplotDataset(raw_train_data, processor, MAX_PATCHES)
train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collator
)

# %%
dev_dataset = DeplotDataset(raw_dev_data, processor, MAX_PATCHES)
dev_dataloader = DataLoader(
    dev_dataset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collator
)

# %% [markdown]
# <a name="Training"></a>
# ### 🚀 Training the model

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# %%
optimizer = Adafactor(
    model.parameters(),
    scale_parameter=False,
    relative_step=False,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# %%
# 2. Dynamically calculate steps
total_steps = len(train_dataloader) * EPOCHS
num_warmup_steps = int(NUM_WARMUP_PERCENTAGE * total_steps)

# 3. Swap linear scheduler for cosine scheduler
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
)

# %%
best_val_loss = float("inf")
epochs_no_improve = 0
best_model_wts = copy.deepcopy(model.state_dict())

# %%
epoch_pbar = tqdm(range(EPOCHS), desc="Total Epochs")

for epoch in epoch_pbar:
    # --- Training Phase ---
    model.train()
    train_loss = 0

    # Batch progress bar for training
    train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Train]", leave=False)

    for idx, batch in enumerate(train_pbar):
        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        outputs = model(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        # Update progress bar with the current batch loss
        train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    avg_train_loss = train_loss / len(train_dataloader)

    # --- Validation Phase ---
    model.eval()
    val_loss = 0

    # Batch progress bar for validation
    val_pbar = tqdm(dev_dataloader, desc=f"Epoch {epoch + 1} [Val]", leave=False)

    with torch.no_grad():
        for batch in val_pbar:
            labels = batch.pop("labels").to(device)
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)

            outputs = model(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                labels=labels,
            )

            v_loss = outputs.loss.item()
            val_loss += v_loss

            # Update progress bar with current validation loss
            val_pbar.set_postfix({"batch_loss": f"{v_loss:.4f}"})

    avg_val_loss = val_loss / len(dev_dataloader)

    # Update the main epoch progress bar with the final averages
    epoch_pbar.set_postfix(
        {"train_loss": f"{avg_train_loss:.4f}", "val_loss": f"{avg_val_loss:.4f}"}
    )

    # --- Early Stopping Check ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            tqdm.write(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

# %%
# Load best model weights before continuing
print("\nTraining complete. Loading best model weights based on validation loss.")
model.load_state_dict(best_model_wts)

model.save_pretrained(CHECKPOINT_PATH)
processor.save_pretrained(CHECKPOINT_PATH)
print(f"Model saved to {CHECKPOINT_PATH}")

# %% [markdown]
# # 🧪 Inference Test

# %%
print(f"Loading finetuned model and processor from {CHECKPOINT_PATH}...")

# Load the model and processor from the saved checkpoint
checkpoint_model = Pix2StructForConditionalGeneration.from_pretrained(
    CHECKPOINT_PATH
).to(device)
checkpoint_processor = Pix2StructProcessor.from_pretrained(CHECKPOINT_PATH)
checkpoint_model.eval()

# %%
# Grab a single test example from the validation (dev) set
test_sample = raw_dev_data[0]

# Prepare inputs using the loaded processor
inputs = checkpoint_processor(
    images=test_sample["image"],
    text=test_sample["prompt"],
    return_tensors="pt",
    max_patches=MAX_PATCHES,
).to(device)

print("Generating table...")
with torch.no_grad():
    generated_ids = checkpoint_model.generate(
        flattened_patches=inputs.flattened_patches,
        attention_mask=inputs.attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
    )

# Decode the prediction
generated_table = checkpoint_processor.decode(
    generated_ids[0], skip_special_tokens=True
)

print("\n--- Ground Truth ---")
print(test_sample["target"])
print("\n--- Generated Table ---")
print(generated_table)
