"""Tests for the SMT grammars module."""

import pytest

from staged_qwen3_5_scivqa.smt.grammars import (
    SMT_CFG_PASS1A,
    build_dynamic_phase1b_cfg,
    build_dynamic_phase2_cfg,
)


@pytest.mark.unit
def test_smt_cfg_pass1a_exists() -> None:
    """Test that the Pass 1A CFG is compiled successfully."""
    assert SMT_CFG_PASS1A is not None


@pytest.mark.unit
def test_build_dynamic_phase1b_cfg() -> None:
    """Test dynamic Pass 1B CFG construction."""
    declarations = (
        "(declare-const test_entity Entity)\n"
        "(declare-const test_series Series)\n"
        "(declare-const max_val Real)"
    )

    cfg = build_dynamic_phase1b_cfg(declarations)
    assert cfg is not None


@pytest.mark.unit
def test_build_dynamic_phase2_cfg_yesno() -> None:
    """Test dynamic Pass 2 CFG for Yes/No answer type."""
    declarations = (
        "(declare-const test_entity Entity)\n"
        "(declare-const test_series Series)\n"
        "(declare-const cond_bool Bool)\n"
        "(declare-const max_val Real)"
    )

    cfg = build_dynamic_phase2_cfg(declarations, answer_type="Yes/No")
    assert cfg is not None


@pytest.mark.unit
def test_build_dynamic_phase2_cfg_factoid() -> None:
    """Test dynamic Pass 2 CFG for Factoid answer type."""
    declarations = (
        "(declare-const test_entity Entity)\n"
        "(declare-const test_series Series)\n"
        "(declare-const cond_bool Bool)\n"
        "(declare-const max_val Real)"
    )

    cfg = build_dynamic_phase2_cfg(declarations, answer_type="Factoid")
    assert cfg is not None


@pytest.mark.unit
def test_build_dynamic_phase2_cfg_with_numbers() -> None:
    """Test dynamic Pass 2 CFG with valid numbers injection."""
    declarations = (
        "(declare-const test_entity Entity)\n(declare-const test_series Series)"
    )
    valid_numbers = ["0.0", "1.0", "2.5", "10.0"]

    cfg = build_dynamic_phase2_cfg(
        declarations, valid_numbers=valid_numbers, answer_type="List"
    )
    assert cfg is not None
