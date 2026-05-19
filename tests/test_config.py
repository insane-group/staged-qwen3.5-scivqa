"""Tests for the config module."""

from staged_qwen3_5_scivqa.config import (
    COMPETITION_DATA_DIR,
    MODEL_ID,
    PROMPTS,
    TOKEN_BUDGETS,
)


class TestConfig:
    def test_model_id(self) -> None:
        assert MODEL_ID == "unsloth/Qwen3.5-9B"

    def test_token_budgets_structure(self) -> None:
        expected_types = {"Yes/No", "Paragraph", "Factoid", "List", "Summary", "Table"}
        assert set(TOKEN_BUDGETS.keys()) == expected_types

        for answer_type, budget in TOKEN_BUDGETS.items():
            assert "max_new_tokens" in budget
            assert "max_sequence_length" in budget
            assert isinstance(budget["max_new_tokens"], int)
            assert isinstance(budget["max_sequence_length"], int)

    def test_prompts_structure(self) -> None:
        expected_types = {"Yes/No", "Factoid", "List", "Paragraph"}
        assert set(PROMPTS.keys()) == expected_types

        for answer_type, prompt in PROMPTS.items():
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert "{question}" in prompt
            assert "{question_type}" in prompt
            assert "{context}" in prompt

    def test_competition_data_dir(self) -> None:
        assert "ALD-E-ImageMiner" in str(COMPETITION_DATA_DIR)
        assert "icdar2026-competition-data" in str(COMPETITION_DATA_DIR)
