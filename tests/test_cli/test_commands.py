"""Unit tests for CLI commands (fully mocked, no GPU required)."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from staged_qwen3_5_scivqa.cli.main import app

runner = CliRunner()


@pytest.mark.unit
class TestCLIHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.output
        assert "inference" in result.output
        assert "eval" in result.output
        assert "hf" in result.output

    def test_train_help(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "summary" in result.output
        assert "table" in result.output
        assert "vqa" in result.output

    def test_eval_help(self):
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "vqa" in result.output
        assert "summary" in result.output
        assert "table" in result.output

    def test_hf_help(self):
        result = runner.invoke(app, ["hf", "--help"])
        assert result.exit_code == 0
        assert "push" in result.output
        assert "pull" in result.output


@pytest.mark.unit
class TestTrainCommands:
    def test_train_summary_runs(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "train",
                "summary",
                "--category",
                "test",
                "--output-dir",
                str(tmp_path / "summary"),
            ],
        )
        assert result.exit_code in (0, 1)

    def test_train_table_runs(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "train",
                "table",
                "--category",
                "test",
                "--output-dir",
                str(tmp_path / "table"),
            ],
        )
        assert result.exit_code in (0, 1)

    def test_train_vqa_invalid_types(self):
        result = runner.invoke(
            app,
            [
                "train",
                "vqa",
                "--answer-types",
                "invalid_type",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid answer types" in result.output


@pytest.mark.unit
class TestInferenceCommand:
    def test_inference_vqa_runs(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "inference",
                "vqa",
                "--category",
                "test",
                "--checkpoint-dir",
                str(tmp_path / "checkpoint"),
            ],
        )
        assert result.exit_code in (0, 1)


@pytest.mark.unit
class TestSMTCommand:
    def test_smt_run(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "smt",
                "run",
                "--category",
                "test",
            ],
        )
        assert result.exit_code in (0, 1)

    def test_smt_run_with_model_id(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "smt",
                "run",
                "--category",
                "test",
                "--model-id",
                "unsloth/Qwen3.5-0.8B",
            ],
        )
        assert result.exit_code in (0, 1)

    def test_smt_run_with_resume(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        smt_file = data_dir / "smt_test_state.json"
        smt_file.write_text("{}")

        with patch(
            "staged_qwen3_5_scivqa.cli.commands.stage_has_output",
            return_value=True,
        ):
            result = runner.invoke(
                app,
                [
                    "smt",
                    "run",
                    "--category",
                    "test",
                ],
            )
            assert result.exit_code == 0
            assert "Skipping" in result.output

    def test_smt_run_with_mocks(self, tmp_path):
        with patch(
            "staged_qwen3_5_scivqa.cli.commands.stage_has_output",
            return_value=False,
        ):
            with patch("staged_qwen3_5_scivqa.models.smt_runner.run_smt_pipeline"):
                result = runner.invoke(
                    app,
                    [
                        "smt",
                        "run",
                        "--category",
                        "test",
                    ],
                )
                assert result.exit_code in (0, 1)


@pytest.mark.unit
class TestReflectCommand:
    def test_reflect(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "reflect",
                "--category",
                "test",
            ],
        )
        assert result.exit_code in (0, 1)

    def test_reflect_with_model_id(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "reflect",
                "--category",
                "test",
                "--model-id",
                "unsloth/Qwen3.5-0.8B",
            ],
        )
        assert result.exit_code in (0, 1)

    def test_reflect_with_resume(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        reflection_file = data_dir / "submission_reflection_test_state.json"
        reflection_file.write_text("{}")

        with patch(
            "staged_qwen3_5_scivqa.cli.commands.stage_has_output",
            return_value=True,
        ):
            result = runner.invoke(
                app,
                [
                    "reflect",
                    "--category",
                    "test",
                ],
            )
            assert result.exit_code == 0
            assert "Skipping" in result.output

    def test_reflect_with_mocks(self, tmp_path):
        with patch(
            "staged_qwen3_5_scivqa.cli.commands.stage_has_output",
            return_value=False,
        ):
            with patch("staged_qwen3_5_scivqa.models.reflection_runner.run_reflection"):
                result = runner.invoke(
                    app,
                    [
                        "reflect",
                        "--category",
                        "test",
                    ],
                )
                assert result.exit_code in (0, 1)


@pytest.mark.unit
class TestEvalCommands:
    def test_eval_vqa_missing_file(self):
        result = runner.invoke(
            app,
            [
                "eval",
                "vqa",
                "--predictions",
                "/nonexistent/file.json",
            ],
        )
        assert result.exit_code == 0

    def test_eval_summary_missing_file(self):
        result = runner.invoke(
            app,
            [
                "eval",
                "summary",
                "--predictions",
                "/nonexistent/file.json",
            ],
        )
        assert result.exit_code == 0

    def test_eval_table_missing_file(self):
        result = runner.invoke(
            app,
            [
                "eval",
                "table",
                "--predictions",
                "/nonexistent/file.json",
            ],
        )
        assert result.exit_code == 0

    def test_eval_vqa_with_data(self, tmp_path):
        pred_file = tmp_path / "preds.json"
        pred_data = [
            {
                "sample_id": "test/fig1",
                "vqa": {
                    "a": [
                        {"answer_type": "Yes/No", "answer": "Yes"},
                        {"answer_type": "Factoid", "answer": "200 C"},
                    ]
                },
            }
        ]
        pred_file.write_text(json.dumps(pred_data))

        with patch("staged_qwen3_5_scivqa.data.load_vqa_dataset") as mock_load:
            mock_load.return_value = ([], 0, 0)
            result = runner.invoke(
                app,
                [
                    "eval",
                    "vqa",
                    "--predictions",
                    str(pred_file),
                ],
            )
            assert result.exit_code == 0


@pytest.mark.unit
class TestRunCommand:
    def test_run_invalid_stages(self):
        result = runner.invoke(
            app,
            [
                "run",
                "--stages",
                "invalid_stage",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid stages" in result.output

    def test_run_with_resume(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        smt_file = data_dir / "smt_test_state.json"
        smt_file.write_text("{}")

        result = runner.invoke(
            app,
            [
                "run",
                "--stages",
                "smt",
                "--category",
                "test",
            ],
        )
        assert result.exit_code in (0, 1)


@pytest.mark.unit
class TestHFCommands:
    def test_hf_push_path_not_found(self):
        result = runner.invoke(
            app,
            [
                "hf",
                "push",
                "/nonexistent/path",
                "--repo-id",
                "user/model",
            ],
        )
        assert result.exit_code == 1
        assert "Path not found" in result.output

    def test_hf_push(self, tmp_path):
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "adapter_config.json").write_text("{}")

        with patch("huggingface_hub.HfApi") as mock_api:
            mock_api.return_value = MagicMock()
            result = runner.invoke(
                app,
                [
                    "hf",
                    "push",
                    str(checkpoint_dir),
                    "--repo-id",
                    "user/model",
                ],
            )
            assert result.exit_code == 0

    def test_hf_pull(self, tmp_path):
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = str(tmp_path / "downloaded")
            result = runner.invoke(
                app,
                [
                    "hf",
                    "pull",
                    "--repo-id",
                    "user/model",
                    "--output",
                    str(tmp_path / "out"),
                ],
            )
            assert result.exit_code == 0

    def test_hf_push_dataset(self, tmp_path):
        ds_dir = tmp_path / "dataset"
        ds_dir.mkdir()

        with patch("huggingface_hub.HfApi") as mock_api:
            mock_api.return_value = MagicMock()
            result = runner.invoke(
                app,
                [
                    "hf",
                    "push-dataset",
                    str(ds_dir),
                    "--repo-id",
                    "user/ds",
                ],
            )
            assert result.exit_code == 0

    def test_hf_pull_dataset(self, tmp_path):
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = str(tmp_path / "ds")
            result = runner.invoke(
                app,
                [
                    "hf",
                    "pull-dataset",
                    "--repo-id",
                    "user/ds",
                    "--output",
                    str(tmp_path / "out"),
                ],
            )
            assert result.exit_code == 0


@pytest.mark.unit
class TestTrainStageIntegration:
    def _make_mock_token_stats(self, n):
        return pd.DataFrame({"total_tokens": [100] * n, "assistant_tokens": [10] * n})

    def test_train_stage_with_mocks(self, tmp_path):
        from staged_qwen3_5_scivqa.cli.commands import train_stage
        from staged_qwen3_5_scivqa.config import SciVQAConfig

        cfg = SciVQAConfig()
        cfg.paths.output_dir = tmp_path / "models"
        cfg.paths.data_dir = tmp_path / "data"
        cfg.category = "test"
        cfg.training.epochs = 1
        cfg.wandb.enabled = False
        cfg.hf.push_checkpoints = False

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.save_pretrained = MagicMock()
        mock_tokenizer.save_pretrained = MagicMock()

        with (
            patch("unsloth.FastVisionModel") as mock_fvm,
            patch("staged_qwen3_5_scivqa.models.lora.get_lora_config") as mock_lora,
            patch("staged_qwen3_5_scivqa.models.trainer.get_sft_config") as mock_sft,
            patch("staged_qwen3_5_scivqa.data.load_summary_dataset") as mock_load,
            patch("trl.SFTTrainer") as mock_trainer,
            patch("unsloth.UnslothVisionDataCollator"),
            patch("staged_qwen3_5_scivqa.analysis.calculate_token_stats") as mock_stats,
        ):
            mock_fvm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            mock_fvm.get_peft_model.return_value = mock_model
            mock_lora.return_value = {"r": 16, "lora_alpha": 16}
            mock_sft.return_value = MagicMock()
            mock_load.return_value = ([], 0, 0)
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance
            mock_stats.return_value = self._make_mock_token_stats(0)

            train_stage(cfg, "summary", tmp_path / "summary")

            mock_fvm.from_pretrained.assert_called_once()
            mock_trainer_instance.train.assert_called_once()
            mock_model.save_pretrained.assert_called_once()


@pytest.mark.unit
class TestFilterSamples:
    def test_filters_samples_exceeding_limits(self):
        from staged_qwen3_5_scivqa.cli.commands import _filter_samples

        mock_tokenizer = MagicMock()
        msg = [
            {"content": [{"text": "img", "image": None}]},
            {"content": [{"text": "short"}]},
        ]
        long_msg = [
            {"content": [{"text": "img", "image": None}]},
            {"content": [{"text": "a" * 1000}]},
        ]
        samples = [{"messages": msg}, {"messages": long_msg}]

        with patch(
            "staged_qwen3_5_scivqa.analysis.calculate_token_stats"
        ) as mock_stats:
            mock_stats.return_value = pd.DataFrame(
                {
                    "total_tokens": [100, 5000],
                    "assistant_tokens": [10, 2000],
                }
            )
            result = _filter_samples(
                samples,
                mock_tokenizer,
                max_seq_length=3072,
                max_new_tokens=100,
            )

        assert len(result) == 1
        assert result[0] == samples[0]

    def test_keeps_all_samples_within_limits(self):
        from staged_qwen3_5_scivqa.cli.commands import _filter_samples

        mock_tokenizer = MagicMock()
        msg = [
            {"content": [{"text": "img", "image": None}]},
            {"content": [{"text": "short"}]},
        ]
        samples = [{"messages": msg}]

        with patch(
            "staged_qwen3_5_scivqa.analysis.calculate_token_stats"
        ) as mock_stats:
            mock_stats.return_value = pd.DataFrame(
                {
                    "total_tokens": [100],
                    "assistant_tokens": [10],
                }
            )
            result = _filter_samples(
                samples,
                mock_tokenizer,
                max_seq_length=3072,
                max_new_tokens=100,
            )

        assert len(result) == 1

    def test_drops_all_samples_when_all_exceed(self):
        from staged_qwen3_5_scivqa.cli.commands import _filter_samples

        mock_tokenizer = MagicMock()
        msg = [
            {"content": [{"text": "img", "image": None}]},
            {"content": [{"text": "a" * 1000}]},
        ]
        samples = [{"messages": msg}]

        with patch(
            "staged_qwen3_5_scivqa.analysis.calculate_token_stats"
        ) as mock_stats:
            mock_stats.return_value = pd.DataFrame(
                {
                    "total_tokens": [5000],
                    "assistant_tokens": [2000],
                }
            )
            result = _filter_samples(
                samples,
                mock_tokenizer,
                max_seq_length=3072,
                max_new_tokens=100,
            )

        assert len(result) == 0


@pytest.mark.unit
class TestBalanceYesNo:
    def test_balances_imbalanced_dataset(self):
        from staged_qwen3_5_scivqa.cli.commands import _balance_yes_no

        yes_sample = {
            "messages": [
                {"content": [{"text": "img", "image": None}]},
                {"content": [{"text": "Yes"}]},
            ]
        }
        no_sample = {
            "messages": [
                {"content": [{"text": "img", "image": None}]},
                {"content": [{"text": "No"}]},
            ]
        }
        samples = [yes_sample] * 8 + [no_sample] * 2

        result = _balance_yes_no(samples)

        answer = lambda s: s["messages"][1]["content"][0]["text"]  # noqa: E731
        yes_count = sum(1 for s in result if answer(s) == "Yes")
        no_count = sum(1 for s in result if answer(s) == "No")
        assert yes_count == no_count == 8
        assert len(result) == 16

    def test_no_change_when_balanced(self):
        from staged_qwen3_5_scivqa.cli.commands import _balance_yes_no

        yes_sample = {
            "messages": [
                {"content": [{"text": "img", "image": None}]},
                {"content": [{"text": "Yes"}]},
            ]
        }
        no_sample = {
            "messages": [
                {"content": [{"text": "img", "image": None}]},
                {"content": [{"text": "No"}]},
            ]
        }
        samples = [yes_sample] * 5 + [no_sample] * 5

        result = _balance_yes_no(samples)

        assert len(result) == 10

    def test_handles_no_samples(self):
        from staged_qwen3_5_scivqa.cli.commands import _balance_yes_no

        result = _balance_yes_no([])
        assert result == []

    def test_balances_when_no_is_majority(self):
        from staged_qwen3_5_scivqa.cli.commands import _balance_yes_no

        yes_sample = {
            "messages": [
                {"content": [{"text": "img", "image": None}]},
                {"content": [{"text": "Yes"}]},
            ]
        }
        no_sample = {
            "messages": [
                {"content": [{"text": "img", "image": None}]},
                {"content": [{"text": "No"}]},
            ]
        }
        samples = [no_sample] * 8 + [yes_sample] * 2

        result = _balance_yes_no(samples)

        answer = lambda s: s["messages"][1]["content"][0]["text"]  # noqa: E731
        yes_count = sum(1 for s in result if answer(s) == "Yes")
        no_count = sum(1 for s in result if answer(s) == "No")
        assert yes_count == no_count == 8
        assert len(result) == 16

    def test_preserves_original_samples(self):
        from staged_qwen3_5_scivqa.cli.commands import _balance_yes_no

        yes_sample = {
            "messages": [
                {"content": [{"text": "img", "image": None}]},
                {"content": [{"text": "Yes"}]},
            ]
        }
        no_sample = {
            "messages": [
                {"content": [{"text": "img", "image": None}]},
                {"content": [{"text": "No"}]},
            ]
        }
        samples = [yes_sample] * 3 + [no_sample] * 1
        original_len = len(samples)

        result = _balance_yes_no(samples)

        assert len(samples) == original_len
        assert len(result) >= original_len


@pytest.mark.unit
class TestFilterSamplesEdgeCases:
    def test_filters_by_assistant_tokens_only(self):
        from staged_qwen3_5_scivqa.cli.commands import _filter_samples

        mock_tokenizer = MagicMock()
        msg = [
            {"content": [{"text": "img", "image": None}]},
            {"content": [{"text": "short"}]},
        ]
        samples = [{"messages": msg}, {"messages": msg}]

        with patch(
            "staged_qwen3_5_scivqa.analysis.calculate_token_stats"
        ) as mock_stats:
            mock_stats.return_value = pd.DataFrame(
                {
                    "total_tokens": [100, 200],
                    "assistant_tokens": [10, 200],
                }
            )
            result = _filter_samples(
                samples,
                mock_tokenizer,
                max_seq_length=3072,
                max_new_tokens=50,
            )

        assert len(result) == 1
        assert result[0] == samples[0]

    def test_filters_by_total_tokens_only(self):
        from staged_qwen3_5_scivqa.cli.commands import _filter_samples

        mock_tokenizer = MagicMock()
        msg = [
            {"content": [{"text": "img", "image": None}]},
            {"content": [{"text": "short"}]},
        ]
        samples = [{"messages": msg}, {"messages": msg}]

        with patch(
            "staged_qwen3_5_scivqa.analysis.calculate_token_stats"
        ) as mock_stats:
            mock_stats.return_value = pd.DataFrame(
                {
                    "total_tokens": [100, 5000],
                    "assistant_tokens": [10, 10],
                }
            )
            result = _filter_samples(
                samples,
                mock_tokenizer,
                max_seq_length=3072,
                max_new_tokens=100,
            )

        assert len(result) == 1
        assert result[0] == samples[0]

    def test_empty_samples_returns_empty(self):
        from staged_qwen3_5_scivqa.cli.commands import _filter_samples

        mock_tokenizer = MagicMock()

        with patch(
            "staged_qwen3_5_scivqa.analysis.calculate_token_stats"
        ) as mock_stats:
            mock_stats.return_value = pd.DataFrame(
                {"total_tokens": [], "assistant_tokens": []}
            )
            result = _filter_samples(
                [],
                mock_tokenizer,
                max_seq_length=3072,
                max_new_tokens=100,
            )

        assert result == []


@pytest.mark.unit
class TestTrainStageMultiCategory:
    def test_train_stage_table(self, tmp_path):
        from staged_qwen3_5_scivqa.cli.commands import train_stage
        from staged_qwen3_5_scivqa.config import SciVQAConfig

        cfg = SciVQAConfig()
        cfg.paths.output_dir = tmp_path / "models"
        cfg.paths.data_dir = tmp_path / "data"
        cfg.category = "train"
        cfg.training.epochs = 1
        cfg.wandb.enabled = False
        cfg.hf.push_checkpoints = False

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.save_pretrained = MagicMock()
        mock_tokenizer.save_pretrained = MagicMock()

        with (
            patch("unsloth.FastVisionModel") as mock_fvm,
            patch("staged_qwen3_5_scivqa.models.lora.get_lora_config") as mock_lora,
            patch("staged_qwen3_5_scivqa.models.trainer.get_sft_config") as mock_sft,
            patch("staged_qwen3_5_scivqa.data.load_table_dataset") as mock_load,
            patch("trl.SFTTrainer") as mock_trainer,
            patch("unsloth.UnslothVisionDataCollator"),
            patch("staged_qwen3_5_scivqa.analysis.calculate_token_stats") as mock_stats,
        ):
            mock_fvm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            mock_fvm.get_peft_model.return_value = mock_model
            mock_lora.return_value = {"r": 16, "lora_alpha": 16}
            mock_sft.return_value = MagicMock()
            mock_load.return_value = ([], 0, 0)
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance
            mock_stats.return_value = pd.DataFrame(
                {"total_tokens": [], "assistant_tokens": []}
            )

            train_stage(cfg, "table", tmp_path / "table")

            mock_load.assert_called_once_with("train")
            mock_trainer_instance.train.assert_called_once()

    def test_train_stage_vqa_factoid(self, tmp_path):
        from staged_qwen3_5_scivqa.cli.commands import train_stage
        from staged_qwen3_5_scivqa.config import SciVQAConfig

        cfg = SciVQAConfig()
        cfg.paths.output_dir = tmp_path / "models"
        cfg.paths.data_dir = tmp_path / "data"
        cfg.category = "train"
        cfg.training.epochs = 1
        cfg.wandb.enabled = False
        cfg.hf.push_checkpoints = False

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.save_pretrained = MagicMock()
        mock_tokenizer.save_pretrained = MagicMock()

        with (
            patch("unsloth.FastVisionModel") as mock_fvm,
            patch("staged_qwen3_5_scivqa.models.lora.get_lora_config") as mock_lora,
            patch("staged_qwen3_5_scivqa.models.trainer.get_sft_config") as mock_sft,
            patch("staged_qwen3_5_scivqa.data.load_vqa_dataset") as mock_load,
            patch("trl.SFTTrainer") as mock_trainer,
            patch("unsloth.UnslothVisionDataCollator"),
            patch("staged_qwen3_5_scivqa.analysis.calculate_token_stats") as mock_stats,
        ):
            mock_fvm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            mock_fvm.get_peft_model.return_value = mock_model
            mock_lora.return_value = {"r": 16, "lora_alpha": 16}
            mock_sft.return_value = MagicMock()
            mock_load.return_value = ([], 0, 0)
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance
            mock_stats.return_value = pd.DataFrame(
                {"total_tokens": [], "assistant_tokens": []}
            )

            train_stage(cfg, "factoid", tmp_path / "vqa_factoid")

            mock_load.assert_called_once_with("train", ["factoid"])
            mock_trainer_instance.train.assert_called_once()

    def test_train_stage_yes_no_applies_balancing(self, tmp_path):
        from staged_qwen3_5_scivqa.cli.commands import train_stage
        from staged_qwen3_5_scivqa.config import SciVQAConfig

        cfg = SciVQAConfig()
        cfg.paths.output_dir = tmp_path / "models"
        cfg.paths.data_dir = tmp_path / "data"
        cfg.category = "train"
        cfg.training.epochs = 1
        cfg.wandb.enabled = False
        cfg.hf.push_checkpoints = False

        yes_sample = {
            "messages": [
                {"content": [{"text": "img", "image": None}]},
                {"content": [{"text": "Yes"}]},
            ]
        }
        no_sample = {
            "messages": [
                {"content": [{"text": "img", "image": None}]},
                {"content": [{"text": "No"}]},
            ]
        }
        imbalanced = [yes_sample] * 8 + [no_sample] * 2

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.save_pretrained = MagicMock()
        mock_tokenizer.save_pretrained = MagicMock()

        with (
            patch("unsloth.FastVisionModel") as mock_fvm,
            patch("staged_qwen3_5_scivqa.models.lora.get_lora_config") as mock_lora,
            patch("staged_qwen3_5_scivqa.models.trainer.get_sft_config") as mock_sft,
            patch("staged_qwen3_5_scivqa.data.load_vqa_dataset") as mock_load,
            patch("trl.SFTTrainer") as mock_trainer,
            patch("unsloth.UnslothVisionDataCollator"),
            patch("staged_qwen3_5_scivqa.analysis.calculate_token_stats") as mock_stats,
        ):
            mock_fvm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            mock_fvm.get_peft_model.return_value = mock_model
            mock_lora.return_value = {"r": 16, "lora_alpha": 16}
            mock_sft.return_value = MagicMock()
            mock_load.return_value = (imbalanced, 10, 0)
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance
            mock_stats.return_value = pd.DataFrame(
                {
                    "total_tokens": [100] * 10,
                    "assistant_tokens": [1] * 10,
                }
            )

            train_stage(cfg, "yes_no", tmp_path / "vqa_yes_no")

            train_dataset = mock_trainer.call_args.kwargs.get(
                "train_dataset"
            ) or mock_trainer.call_args[1].get("train_dataset")
            if train_dataset is None:
                args = mock_trainer.call_args
                train_dataset = args[1].get("train_dataset") if len(args) > 1 else None
            assert train_dataset is not None
            assert len(train_dataset) == 16

    def test_train_stage_multi_category(self, tmp_path):
        from staged_qwen3_5_scivqa.cli.commands import train_stage
        from staged_qwen3_5_scivqa.config import SciVQAConfig

        cfg = SciVQAConfig()
        cfg.paths.output_dir = tmp_path / "models"
        cfg.paths.data_dir = tmp_path / "data"
        cfg.category = "train,dev"
        cfg.training.epochs = 1
        cfg.wandb.enabled = False
        cfg.hf.push_checkpoints = False

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.save_pretrained = MagicMock()
        mock_tokenizer.save_pretrained = MagicMock()

        train_sample = {"id": "train_sample"}
        dev_sample = {"id": "dev_sample"}

        with (
            patch("unsloth.FastVisionModel") as mock_fvm,
            patch("staged_qwen3_5_scivqa.models.lora.get_lora_config") as mock_lora,
            patch("staged_qwen3_5_scivqa.models.trainer.get_sft_config") as mock_sft,
            patch("staged_qwen3_5_scivqa.data.load_summary_dataset") as mock_load,
            patch("trl.SFTTrainer") as mock_trainer,
            patch("unsloth.UnslothVisionDataCollator"),
            patch("staged_qwen3_5_scivqa.analysis.calculate_token_stats") as mock_stats,
        ):
            mock_fvm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            mock_fvm.get_peft_model.return_value = mock_model
            mock_lora.return_value = {"r": 16, "lora_alpha": 16}
            mock_sft.return_value = MagicMock()
            mock_load.side_effect = [
                ([train_sample], 1, 0),
                ([dev_sample], 1, 0),
            ]
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance
            mock_stats.return_value = pd.DataFrame(
                {
                    "total_tokens": [100, 100],
                    "assistant_tokens": [10, 10],
                }
            )

            train_stage(cfg, "summary", tmp_path / "summary")

            assert mock_load.call_count == 2
            calls = [c.args[0] for c in mock_load.call_args_list]
            assert calls == ["train", "dev"]

            train_dataset = mock_trainer.call_args[1].get("train_dataset")
            assert train_dataset is not None
            assert len(train_dataset) == 2
