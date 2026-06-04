"""Tests for CLI utility helpers."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from staged_qwen3_5_scivqa.cli.utils import (
    finish_wandb,
    load_json_state,
    print_metrics_table,
    print_skip_header,
    print_stage_header,
    print_submission_summary,
    pull_dataset,
    pull_from_hub,
    push_dataset,
    push_to_hub,
    save_json_state,
    setup_wandb,
    stage_has_output,
)
from staged_qwen3_5_scivqa.config import PathsConfig, SciVQAConfig


@pytest.mark.unit
class TestPrintHelpers:
    def test_print_stage_header(self, capsys):
        print_stage_header("summary", 1, 3)
        captured = capsys.readouterr()
        assert "Stage 1/3" in captured.out
        assert "SUMMARY" in captured.out

    def test_print_skip_header(self, capsys):
        print_skip_header("SMT")
        captured = capsys.readouterr()
        assert "Skipping SMT" in captured.out

    def test_print_metrics_table(self, capsys):
        metrics = {"accuracy": 0.85, "f1": 0.82}
        print_metrics_table(metrics, "Test Metrics")
        captured = capsys.readouterr()
        assert "accuracy" in captured.out
        assert "0.85" in captured.out

    def test_print_submission_summary(self, capsys):
        state = {
            "sample1": {"fig_a": [{"answer": "Yes"}]},
            "sample2": {"fig_b": [{"answer": "No"}, {"answer": "Maybe"}]},
        }
        print_submission_summary(state, "Test")
        captured = capsys.readouterr()
        assert "Samples: 2" in captured.out
        assert "Questions: 3" in captured.out


@pytest.mark.unit
class TestWandbHelpers:
    def test_setup_wandb_disabled(self):
        from staged_qwen3_5_scivqa.config import WandbConfig

        cfg = WandbConfig(enabled=False)
        result = setup_wandb(cfg)
        assert result is False

    def test_setup_wandb_no_import(self, capsys):
        from staged_qwen3_5_scivqa.config import WandbConfig

        cfg = WandbConfig(enabled=True)
        with patch.dict("sys.modules", {"wandb": None}):
            result = setup_wandb(cfg)
            assert result is False

    def test_finish_wandb_no_run(self):
        with patch.dict("sys.modules", {"wandb": MagicMock(run=None)}):
            finish_wandb()


@pytest.mark.unit
class TestStateHelpers:
    def test_load_json_state_existing(self, tmp_path):
        data = {"key": "value"}
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(data))
        result = load_json_state(state_file)
        assert result == data

    def test_load_json_state_missing(self, capsys):
        result = load_json_state(Path("/nonexistent/path.json"))
        assert result == {}
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_save_json_state(self, tmp_path):
        data = {"sample_id": "test", "vqa": {}}
        state_file = tmp_path / "output.json"
        save_json_state(state_file, data)
        assert state_file.exists()
        loaded = json.loads(state_file.read_text())
        assert loaded == data

    def test_stage_has_output_true(self, tmp_path):
        state_file = tmp_path / "smt_test_state.json"
        state_file.touch()
        cfg = SciVQAConfig(paths=PathsConfig(data_dir=tmp_path))
        assert stage_has_output(cfg, "smt") is True

    def test_stage_has_output_false(self, tmp_path):
        cfg = SciVQAConfig(paths=PathsConfig(data_dir=tmp_path))
        assert stage_has_output(cfg, "smt") is False


@pytest.mark.unit
class TestHFHelpers:
    def test_push_to_hub(self, capsys):
        with patch("huggingface_hub.HfApi") as mock_api:
            mock_instance = MagicMock()
            mock_api.return_value = mock_instance
            result = push_to_hub(Path("/tmp/test"), "user/model", token="tok")
            mock_instance.upload_folder.assert_called_once()
            assert "huggingface.co" in result

    def test_pull_from_hub(self, capsys):
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = "/tmp/downloaded"
            result = pull_from_hub("user/model", Path("/tmp/out"), token="tok")
            mock_dl.assert_called_once()
            assert result == Path("/tmp/downloaded")

    def test_push_dataset(self, capsys):
        with patch("huggingface_hub.HfApi") as mock_api:
            mock_instance = MagicMock()
            mock_api.return_value = mock_instance
            result = push_dataset(Path("/tmp/data"), "user/ds", token="tok")
            mock_instance.upload_folder.assert_called_once()
            assert "huggingface.co/datasets" in result

    def test_pull_dataset(self, capsys):
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = "/tmp/ds"
            result = pull_dataset("user/ds", Path("/tmp/out"), token="tok")
            mock_dl.assert_called_once()
            assert result == Path("/tmp/ds")
