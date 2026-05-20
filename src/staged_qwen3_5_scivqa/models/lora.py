"""LoRA configuration for parameter-efficient fine-tuning."""

from typing import Any

from staged_qwen3_5_scivqa.config import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_RANDOM_STATE,
)


def get_lora_config(
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    bias: str = "none",
    random_state: int = LORA_RANDOM_STATE,
    use_rslora: bool = False,
    loftq_config: dict | None = None,
) -> dict[str, Any]:
    """Build LoRA configuration dict for FastVisionModel.get_peft_model().

    Args:
        finetune_vision_layers: Whether to finetune vision layers.
        finetune_language_layers: Whether to finetune language layers.
        finetune_attention_modules: Whether to finetune attention modules.
        finetune_mlp_modules: Whether to finetune MLP modules.
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        bias: Bias configuration ("none", "all", "lora_only").
        random_state: Random seed for reproducibility.
        use_rslora: Whether to use rank-stabilized LoRA.
        loftq_config: LoftQ configuration (None to disable).

    Returns:
        Dict of LoRA configuration parameters.

    """
    return {
        "finetune_vision_layers": finetune_vision_layers,
        "finetune_language_layers": finetune_language_layers,
        "finetune_attention_modules": finetune_attention_modules,
        "finetune_mlp_modules": finetune_mlp_modules,
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": bias,
        "random_state": random_state,
        "use_rslora": use_rslora,
        "loftq_config": loftq_config,
    }
