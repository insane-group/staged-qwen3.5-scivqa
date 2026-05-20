"""Tests for the models.loader module."""

import pytest
import torch

from staged_qwen3_5_scivqa.models.loader import get_bnb_config, get_generation_kwargs


@pytest.mark.unit
def test_get_bnb_config() -> None:
    """Test BitsAndBytes config construction."""
    config = get_bnb_config()

    assert config.load_in_4bit is True
    assert config.bnb_4bit_compute_dtype == torch.bfloat16
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_quant_type == "nf4"


@pytest.mark.unit
def test_get_bnb_config_custom() -> None:
    """Test BitsAndBytes config with custom parameters."""
    config = get_bnb_config(
        compute_dtype=torch.float16,
        use_double_quant=False,
        quant_type="fp4",
    )

    assert config.bnb_4bit_compute_dtype == torch.float16
    assert config.bnb_4bit_use_double_quant is False
    assert config.bnb_4bit_quant_type == "fp4"


@pytest.mark.unit
def test_get_generation_kwargs() -> None:
    """Test generation kwargs construction."""
    kwargs = get_generation_kwargs(
        max_new_tokens=100,
        temperature=0.5,
    )

    assert kwargs["max_new_tokens"] == 100
    assert kwargs["temperature"] == 0.5
    assert kwargs["use_cache"] is True
    assert "top_p" in kwargs
    assert "top_k" in kwargs
