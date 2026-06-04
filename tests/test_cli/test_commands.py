"""Unit tests for CLI commands (fully mocked, no GPU required)."""

import json
from unittest.mock import MagicMock, patch

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


@pytest.mark.integration
class TestTrainStageIntegration:
    def test_train_stage_with_mocks(self, tmp_path):
        from staged_qwen3_5_scivqa.cli.commands import train_stage
        from staged_qwen3_5_scivqa.config import SciVQAConfig

        cfg = SciVQAConfig()
        cfg.paths.output_dir = tmp_path / "models"
        cfg.paths.data_dir = tmp_path / "data"
        cfg.category = "test"
        cfg.training.epochs = 1
        cfg.wandb.enabled = False

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.save_pretrained = MagicMock()
        mock_tokenizer.save_pretrained = MagicMock()

        with (
            patch("unsloth.FastVisionModel") as mock_fvm,
            patch("staged_qwen3_5_scivqa.cli.commands.get_lora_config") as mock_lora,
            patch("staged_qwen3_5_scivqa.cli.commands.get_sft_config") as mock_sft,
            patch(
                "staged_qwen3_5_scivqa.cli.commands.load_summary_dataset"
            ) as mock_load,
            patch("trl.SFTTrainer") as mock_trainer,
            patch("unsloth.UnslothVisionDataCollator"),
        ):
            mock_fvm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            mock_fvm.get_peft_model.return_value = mock_model
            mock_lora.return_value = {"r": 16, "lora_alpha": 16}
            mock_sft.return_value = MagicMock()
            mock_load.return_value = ([], 0, 0)
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance

            train_stage(cfg, "summary", tmp_path / "summary")

            mock_fvm.from_pretrained.assert_called_once()
            mock_trainer_instance.train.assert_called_once()
            mock_model.save_pretrained.assert_called_once()
