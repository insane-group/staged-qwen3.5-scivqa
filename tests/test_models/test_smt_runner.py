"""Tests for the SMT runner module (minimal, avoiding heavy imports)."""

import pytest


@pytest.mark.unit
class TestSmtRunnerModule:
    def test_module_exists(self):
        """Test that the smt_runner module can be imported."""
        import staged_qwen3_5_scivqa.models.smt_runner as mod

        assert hasattr(mod, "load_smt_model")
        assert hasattr(mod, "run_smt_pipeline")
