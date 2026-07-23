"""Unit tests for Pydantic Settings configuration hierarchy."""

import os

import pytest

from staged_qwen3_5_scivqa.config import (
    HFConfig,
    InferenceConfig,
    LoRAConfig,
    ModelConfig,
    PathsConfig,
    ReflectionConfig,
    SciVQAConfig,
    SMTConfig,
    StageBudget,
    StageConfig,
    TrainingConfig,
    WandbConfig,
    load_config,
)


@pytest.mark.unit
class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.model_id == "unsloth/Qwen3.5-0.8B"
        assert cfg.load_in_4bit is False
        assert cfg.max_seq_length == 4096


@pytest.mark.unit
class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.0
        assert cfg.random_state == 3407

    def test_custom_values(self):
        cfg = LoRAConfig(r=32, lora_alpha=64, lora_dropout=0.1)
        assert cfg.r == 32
        assert cfg.lora_alpha == 64
        assert cfg.lora_dropout == 0.1


@pytest.mark.unit
class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 5
        assert cfg.batch_size == 2
        assert cfg.grad_accum == 4
        assert cfg.lr == 2e-4
        assert cfg.optim == "adamw_8bit"
        assert cfg.warmup_ratio == 0.05
        assert cfg.weight_decay == 0.001
        assert cfg.seed == 3407


@pytest.mark.unit
class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.8
        assert cfg.top_k == 20
        assert cfg.min_p == 0.01
        assert cfg.enable_thinking is False


@pytest.mark.unit
class TestStageBudget:
    def test_creation(self):
        budget = StageBudget(max_new_tokens=128, max_sequence_length=2048)
        assert budget.max_new_tokens == 128
        assert budget.max_sequence_length == 2048


@pytest.mark.unit
class TestStageConfig:
    def test_defaults(self):
        cfg = StageConfig()
        assert cfg.yes_no.max_new_tokens == 1
        assert cfg.paragraph.max_new_tokens == 176
        assert cfg.factoid.max_new_tokens == 72
        assert cfg.list.max_new_tokens == 144
        assert cfg.summary.max_new_tokens == 256
        assert cfg.table.max_new_tokens == 768

    def test_get_method(self):
        cfg = StageConfig()
        assert cfg.get("yes_no") == cfg.yes_no
        assert cfg.get("summary") == cfg.summary
        assert cfg.get("table") == cfg.table


@pytest.mark.unit
class TestPathsConfig:
    def test_defaults(self):
        cfg = PathsConfig()
        assert cfg.data_dir.name == "data"
        assert cfg.output_dir.name == "models"

    def test_custom_paths(self, tmp_path):
        cfg = PathsConfig(data_dir=tmp_path / "my_data")
        assert cfg.data_dir == tmp_path / "my_data"


@pytest.mark.unit
class TestSMTConfig:
    def test_defaults(self):
        cfg = SMTConfig()
        assert cfg.max_new_tokens == 2048
        assert cfg.temperature == 1.0
        assert cfg.presence_penalty == 1.5


@pytest.mark.unit
class TestReflectionConfig:
    def test_defaults(self):
        cfg = ReflectionConfig()
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 0.2
        assert cfg.top_p == 0.1


@pytest.mark.unit
class TestHFConfig:
    def test_defaults(self):
        cfg = HFConfig()
        assert cfg.token is None
        assert cfg.hub_repo_id is None
        assert cfg.push_checkpoints is True
        assert cfg.push_datasets is False
        assert cfg.load_from_hub is False


@pytest.mark.unit
class TestWandbConfig:
    def test_defaults(self):
        cfg = WandbConfig()
        assert cfg.enabled is False
        assert cfg.project == "staged-qwen3.5-scivqa"
        assert cfg.entity is None


@pytest.mark.unit
class TestSciVQAConfig:
    def test_defaults(self):
        cfg = SciVQAConfig()
        assert cfg.model.model_id == "unsloth/Qwen3.5-0.8B"
        assert cfg.lora.r == 16
        assert cfg.training.epochs == 5
        assert cfg.category == "train,dev"

    def test_get_stage_budget(self):
        cfg = SciVQAConfig()
        budget = cfg.get_stage_budget("yes_no")
        assert budget.max_new_tokens == 1
        assert budget.max_sequence_length == 3072

    def test_get_lora_checkpoint_name(self):
        cfg = SciVQAConfig()
        name = cfg.get_lora_checkpoint_name
        assert name("summary") == "Sci-ImageMiner-Qwen3.5-0.8B-LORA-SUMMARY"
        assert name("table") == "Sci-ImageMiner-Qwen3.5-0.8B-LORA-EXTRACTION"
        assert name("factoid") == "Sci-ImageMiner-Qwen3.5-0.8B-LORA-FACTOID"
        assert name("yes_no") == "Sci-ImageMiner-Qwen3.5-0.8B-LORA-YESNO"

    def test_get_state_path(self, tmp_path):
        cfg = SciVQAConfig(paths=PathsConfig(data_dir=tmp_path))
        vqa_path = cfg.get_state_path("vqa")
        assert "submission_finetuning" in vqa_path.name
        assert vqa_path.parent == tmp_path

        smt_path = cfg.get_state_path("smt")
        assert "smt_" in smt_path.name

        submission_path = cfg.get_state_path("submission")
        assert "submission_final" in submission_path.name


@pytest.mark.unit
class TestLoadConfig:
    def test_load_defaults(self):
        cfg = load_config()
        assert cfg.model.model_id == "unsloth/Qwen3.5-0.8B"
        assert cfg.category == "train,dev"

    def test_load_with_category_override(self):
        cfg = load_config(category="dev")
        assert cfg.category == "dev"

    def test_load_with_nested_override(self):
        cfg = load_config(**{"model.model_id": "custom/model"})
        assert cfg.model.model_id == "custom/model"

    def test_load_with_yaml_path(self, tmp_path):
        import yaml

        yaml_file = tmp_path / "config.yaml"
        config = {
            "model": {"model_id": "yaml/model"},
        }
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        cfg = load_config(config_path=str(yaml_file))
        assert cfg.model.model_id == "yaml/model"


@pytest.mark.unit
class TestEnvVarLoading:
    def test_env_prefix(self):
        from staged_qwen3_5_scivqa.config import _yaml_config_path

        original_yaml = _yaml_config_path
        original_env = os.environ.get("SCIVQA_MODEL__MODEL_ID")
        try:
            from staged_qwen3_5_scivqa import config

            config._yaml_config_path = None
            os.environ["SCIVQA_MODEL__MODEL_ID"] = "test/model"
            cfg = SciVQAConfig()
            assert cfg.model.model_id == "test/model"
        finally:
            from staged_qwen3_5_scivqa import config

            config._yaml_config_path = original_yaml
            if original_env is None:
                os.environ.pop("SCIVQA_MODEL__MODEL_ID", None)
            else:
                os.environ["SCIVQA_MODEL__MODEL_ID"] = original_env

    def test_env_nested(self):
        original = os.environ.get("SCIVQA_TRAINING__EPOCHS")
        try:
            os.environ["SCIVQA_TRAINING__EPOCHS"] = "10"
            cfg = SciVQAConfig()
            assert cfg.training.epochs == 10
        finally:
            if original is None:
                os.environ.pop("SCIVQA_TRAINING__EPOCHS", None)
            else:
                os.environ["SCIVQA_TRAINING__EPOCHS"] = original


@pytest.mark.integration
class TestYamlConfigLoading:
    def test_full_yaml_config(self, tmp_path):
        import yaml

        yaml_file = tmp_path / "pipeline.yaml"
        config = {
            "model": {"model_id": "custom/model", "load_in_4bit": False},
            "training": {"epochs": 3, "lr": 1e-5},
            "stages": {"yes_no": {"max_new_tokens": 2, "max_sequence_length": 1024}},
            "wandb": {"enabled": True, "project": "my-project"},
            "hf": {"push_checkpoints": True},
        }
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        cfg = load_config(config_path=str(yaml_file))
        assert cfg.model.model_id == "custom/model"
        assert cfg.model.load_in_4bit is False
        assert cfg.training.epochs == 3
        assert cfg.training.lr == 1e-5
        assert cfg.wandb.enabled is True
        assert cfg.hf.push_checkpoints is True
