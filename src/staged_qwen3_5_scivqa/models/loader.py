"""Model loading utilities for Qwen3.5 vision and language models."""

from typing import Any

import torch
from transformers import BitsAndBytesConfig


def get_bnb_config(
    compute_dtype: torch.dtype = torch.bfloat16,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
) -> BitsAndBytesConfig:
    """Create a BitsAndBytes 4-bit quantization config.

    Args:
        compute_dtype: The compute dtype (bfloat16 recommended).
        use_double_quant: Whether to use double quantization.
        quant_type: Quantization type ("nf4" recommended).

    Returns:
        A BitsAndBytesConfig object.

    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_quant_type=quant_type,
    )


def get_generation_kwargs(
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    min_p: float = 0.01,
    top_p: float = 0.8,
    top_k: int = 20,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    """Build generation kwargs dict for model.generate().

    Args:
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        min_p: Minimum probability threshold.
        top_p: Nucleus sampling top-p.
        top_k: Top-k sampling.
        presence_penalty: Presence penalty.
        repetition_penalty: Repetition penalty.
        enable_thinking: Whether to enable thinking mode.

    Returns:
        Dict of generation kwargs.

    """
    return {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "temperature": temperature,
        "min_p": min_p,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "enable_thinking": enable_thinking,
    }
