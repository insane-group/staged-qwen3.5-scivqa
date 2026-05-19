"""Tests for the reflection runner module."""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestLoadReflectionModel:
    @pytest.mark.skip(reason="Requires GPU for unsloth import")
    def test_load_reflection_model(self):
        """Test reflection model loading with mocked dependencies."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("unsloth.FastLanguageModel") as mock_flm:
            mock_flm.from_pretrained.return_value = (
                mock_model,
                mock_tokenizer,
            )

            from staged_qwen3_5_scivqa.models.reflection_runner import (
                load_reflection_model,
            )

            model, tokenizer = load_reflection_model(
                model_id="unsloth/Qwen3.5-9B",
                max_seq_length=4096,
                load_in_4bit=True,
            )

            mock_flm.from_pretrained.assert_called_once_with(
                model_name="unsloth/Qwen3.5-9B",
                load_in_4bit=True,
                max_seq_length=4096,
                dtype=None,
            )
            mock_flm.for_inference.assert_called_once_with(mock_model)
            assert model == mock_model
            assert tokenizer == mock_tokenizer


@pytest.mark.unit
class TestLoadJson:
    def test_load_json_dict(self, tmp_path):
        """Test loading JSON dict file."""
        data = {"sample1": {"a": [{"question": "Q1", "answer": "A1"}]}}
        path = tmp_path / "state.json"
        path.write_text(json.dumps(data))

        from staged_qwen3_5_scivqa.models.reflection_runner import _load_json

        result = _load_json(path)
        assert result == data

    def test_load_json_list(self, tmp_path):
        """Test loading JSON list file converts to dict."""
        data = [
            {
                "sample_id": "sample1",
                "vqa": {"a": [{"question": "Q1", "answer": "A1"}]},
            }
        ]
        path = tmp_path / "state.json"
        path.write_text(json.dumps(data))

        from staged_qwen3_5_scivqa.models.reflection_runner import _load_json

        result = _load_json(path)
        assert "sample1" in result
        assert result["sample1"] == {"a": [{"question": "Q1", "answer": "A1"}]}

    def test_load_json_missing(self, tmp_path):
        """Test loading missing JSON file raises error."""
        path = tmp_path / "missing.json"

        from staged_qwen3_5_scivqa.models.reflection_runner import _load_json

        with pytest.raises(FileNotFoundError):
            _load_json(path)


@pytest.mark.unit
class TestRunReflection:
    def test_run_reflection_missing_paths(self):
        """Test run_reflection raises error when paths are missing."""
        from staged_qwen3_5_scivqa.models.reflection_runner import (
            run_reflection,
        )

        with pytest.raises(ValueError, match="initial_state_path"):
            run_reflection()

    def test_run_reflection_full(self, tmp_path):
        """Test full reflection pipeline with mocked dependencies."""
        initial_state = {
            "sample1": {
                "a": [
                    {
                        "question_type": "trend",
                        "answer_type": "Yes/No",
                        "question": "Is the trend increasing?",
                        "answer": "Yes",
                    }
                ]
            }
        }
        smt_data = {
            "sample1": {
                "a": {
                    "Is the trend increasing?": {
                        "code": "(assert (= AnsBool true))",
                        "output": "sat",
                    }
                }
            }
        }

        initial_path = tmp_path / "initial.json"
        smt_path = tmp_path / "smt.json"
        reflection_path = tmp_path / "reflection.json"
        submission_path = tmp_path / "submission.json"

        initial_path.write_text(json.dumps(initial_state))
        smt_path.write_text(json.dumps(smt_data))

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch(
                "staged_qwen3_5_scivqa.models.reflection_runner.load_reflection_model",
                return_value=(mock_model, mock_tokenizer),
            ),
            patch(
                "staged_qwen3_5_scivqa.models.reflection_runner.reflect_answers"
            ) as mock_reflect,
        ):
            mock_reflect.return_value = {
                "sample1": {
                    "a": [
                        {
                            "answer": "Yes",
                            "question": "Is the trend increasing?",
                        }
                    ]
                }
            }

            from staged_qwen3_5_scivqa.models.reflection_runner import (
                run_reflection,
            )

            result = run_reflection(
                model_id="unsloth/Qwen3.5-9B",
                initial_state_path=initial_path,
                smt_state_path=smt_path,
                reflection_state_path=reflection_path,
                final_submission_path=submission_path,
            )

            mock_reflect.assert_called_once()
            assert isinstance(result, dict)
