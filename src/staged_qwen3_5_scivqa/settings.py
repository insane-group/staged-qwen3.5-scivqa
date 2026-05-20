"""Pydantic Settings configuration hierarchy for the Sci-ImageMiner CLI.

Config priority (highest to lowest):
1. CLI overrides (passed programmatically)
2. Environment variables (SCIVQA_* prefix)
3. .env file
4. YAML config file (--config flag)
5. Field defaults (match config.py constants)
"""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# ── Nested config models ──────────────────────────────────────────────


class ModelConfig(BaseModel):
    model_id: str = Field(default="unsloth/Qwen3.5-9B")
    load_in_4bit: bool = True
    max_seq_length: int = 4096


class LoRAConfig(BaseModel):
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    random_state: int = 3407
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


class TrainingConfig(BaseModel):
    epochs: int = 5
    batch_size: int = 2
    grad_accum: int = 4
    lr: float = 2e-4
    optim: str = "adamw_8bit"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.001
    seed: int = 3407
    logging_steps: int = 1
    scheduler: str = "linear"


class InferenceConfig(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.01
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    enable_thinking: bool = False


class StageBudget(BaseModel):
    max_new_tokens: int
    max_sequence_length: int


class StageConfig(BaseModel):
    yes_no: StageBudget = Field(
        default=StageBudget(max_new_tokens=1, max_sequence_length=3072)
    )
    paragraph: StageBudget = Field(
        default=StageBudget(max_new_tokens=176, max_sequence_length=3072)
    )
    factoid: StageBudget = Field(
        default=StageBudget(max_new_tokens=72, max_sequence_length=2560)
    )
    list: StageBudget = Field(
        default=StageBudget(max_new_tokens=144, max_sequence_length=2560)
    )
    summary: StageBudget = Field(
        default=StageBudget(max_new_tokens=256, max_sequence_length=4096)
    )
    table: StageBudget = Field(
        default=StageBudget(max_new_tokens=768, max_sequence_length=3072)
    )

    def get(self, name: str) -> StageBudget:
        return getattr(self, name)  # type: ignore[no-any-return]


class PathsConfig(BaseModel):
    data_dir: Path = Field(default_factory=lambda: Path.cwd().parent / "data")
    competition_data_dir: Path = Field(
        default_factory=lambda: Path.cwd().parent
        / "ALD-E-ImageMiner"
        / "icdar2026-competition-data"
    )
    output_dir: Path = Field(default_factory=lambda: Path.cwd().parent / "models")
    cvc5_path: Path = Field(
        default_factory=lambda: Path.home()
        / "cvc5-Linux-x86_64-shared"
        / "bin"
        / "cvc5"
    )


class SMTConfig(BaseModel):
    model_id: str = Field(default="unsloth/Qwen3.5-9B")
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    max_retries: int = 3


class ReflectionConfig(BaseModel):
    model_id: str = Field(default="unsloth/Qwen3.5-9B")
    max_new_tokens: int = 256
    max_sequence_length: int = 4096
    temperature: float = 0.2
    top_p: float = 0.1
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    load_in_4bit: bool = True


class HFConfig(BaseModel):
    token: str | None = None
    hub_repo_id: str | None = None
    push_checkpoints: bool = False
    push_datasets: bool = False
    load_from_hub: bool = False
    dataset_repo_id: str | None = None


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "staged-qwen3.5-scivqa"
    entity: str | None = None
    run_name: str | None = None


# ── Root settings ─────────────────────────────────────────────────────


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCIVQA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    stages: StageConfig = Field(default_factory=StageConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    smt: SMTConfig = Field(default_factory=SMTConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    hf: HFConfig = Field(default_factory=HFConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    category: str = "test"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: tuple[PydanticBaseSettingsSource, ...] = (
            init_settings,
            env_settings,
            dotenv_settings,
        )
        from staged_qwen3_5_scivqa.settings import _yaml_config_path

        if _yaml_config_path and Path(_yaml_config_path).exists():
            sources = (
                init_settings,
                YamlConfigSettingsSource(settings_cls, Path(_yaml_config_path)),
                env_settings,
                dotenv_settings,
            )
        return sources

    def get_stage_budget(self, stage_name: str) -> StageBudget:
        return self.stages.get(stage_name)

    def get_lora_checkpoint_name(self, stage: str) -> str:
        base = f"Sci-ImageMiner-{self.model.model_id.split('/')[-1]}-LORA"
        mapping = {
            "summary": f"{base}-SUMMARY",
            "table": f"{base}-EXTRACTION",
            "factoid": f"{base}-FACTOID",
            "list": f"{base}-LIST",
            "paragraph": f"{base}-PARAGRAPH",
            "yes_no": f"{base}-YESNO",
        }
        return mapping.get(stage, f"{base}-{stage.upper()}")

    def get_state_path(self, stage: str) -> Path:
        mapping = {
            "summary": self.paths.data_dir
            / f"submission_finetuning_summary_{self.category}_state.json",
            "table": self.paths.data_dir
            / f"submission_finetuning_extraction_{self.category}_state.json",
            "vqa": self.paths.data_dir
            / f"submission_finetuning_{self.category}_state.json",
            "smt": self.paths.data_dir / f"smt_{self.category}_state.json",
            "reflection": self.paths.data_dir
            / f"submission_reflection_{self.category}_state.json",
            "submission": self.paths.data_dir
            / f"submission_final_{self.category}.json",
        }
        return mapping.get(stage, self.paths.data_dir / f"{stage}_state.json")


# Module-level storage for YAML config path (set by load_config)
_yaml_config_path: str | None = None


def load_config(
    config_path: str | Path | None = None,
    **overrides,
) -> PipelineConfig:
    """Load PipelineConfig with optional YAML file and CLI overrides."""
    global _yaml_config_path
    _yaml_config_path = str(config_path) if config_path else None
    cfg = PipelineConfig()
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
    return cfg
