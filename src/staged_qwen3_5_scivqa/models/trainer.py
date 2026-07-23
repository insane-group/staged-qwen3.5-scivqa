"""Training configuration and trainer wrappers for Unsloth SFT."""

from trl import SFTConfig

from staged_qwen3_5_scivqa.config import (
    SFT_GRADIENT_ACCUMULATION_STEPS,
    SFT_LEARNING_RATE,
    SFT_PER_DEVICE_BATCH_SIZE,
    SFT_WARMUP_RATIO,
    SFT_WEIGHT_DECAY,
)


def get_sft_config(
    max_length: int = 4096,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = SFT_PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps: int = SFT_GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio: float = SFT_WARMUP_RATIO,
    learning_rate: float = SFT_LEARNING_RATE,
    weight_decay: float = SFT_WEIGHT_DECAY,
    output_dir: str = "outputs",
    **kwargs,
) -> SFTConfig:
    """Build an SFTConfig for Unsloth vision fine-tuning.

    Args:
        max_length: Maximum sequence length.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        warmup_ratio: Warmup ratio for learning rate scheduler.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        output_dir: Output directory for checkpoints.
        **kwargs: Additional SFTConfig parameters.

    Returns:
        An SFTConfig object configured for vision fine-tuning.

    """
    return SFTConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=max_length,
        **kwargs,
    )
