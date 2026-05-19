"""Tests for the models.lora module."""

import pytest

from staged_qwen3_5_scivqa.models.lora import get_lora_config


@pytest.mark.unit
def test_get_lora_config_defaults() -> None:
    """Test LoRA config with default parameters."""
    config = get_lora_config()

    assert config["r"] == 16
    assert config["lora_alpha"] == 16
    assert config["lora_dropout"] == 0.0
    assert config["bias"] == "none"
    assert config["random_state"] == 3407
    assert config["finetune_vision_layers"] is True
    assert config["finetune_language_layers"] is True


@pytest.mark.unit
def test_get_lora_config_custom() -> None:
    """Test LoRA config with custom parameters."""
    config = get_lora_config(
        r=32,
        lora_alpha=32,
        finetune_vision_layers=False,
        use_rslora=True,
    )

    assert config["r"] == 32
    assert config["lora_alpha"] == 32
    assert config["finetune_vision_layers"] is False
    assert config["use_rslora"] is True
