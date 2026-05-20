"""Tests for the SMT solver module."""

from unittest.mock import MagicMock, patch

import pytest

from staged_qwen3_5_scivqa.smt.solver import validate_smt


@pytest.mark.unit
def test_validate_smt_sat() -> None:
    """Test SMT validation with a satisfiable result."""
    mock_result = MagicMock()
    mock_result.stdout = "sat\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        success, output = validate_smt("(check-sat)")
        assert success is True


@pytest.mark.unit
def test_validate_smt_unsat() -> None:
    """Test SMT validation with an unsatisfiable result."""
    mock_result = MagicMock()
    mock_result.stdout = "unsat\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        success, output = validate_smt("(check-sat)")
        assert success is False


@pytest.mark.unit
def test_validate_smt_error() -> None:
    """Test SMT validation with a solver error."""
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = "parse error"

    with patch("subprocess.run", return_value=mock_result):
        success, output = validate_smt("(invalid)")
        assert success is False
        assert "parse error" in output


@pytest.mark.unit
def test_validate_smt_timeout() -> None:
    """Test SMT validation with a timeout."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = TimeoutError("timeout")
        success, output = validate_smt("(check-sat)")
        assert success is False
        assert "Timeout" in output or "timeout" in output.lower() or "Error" in output
