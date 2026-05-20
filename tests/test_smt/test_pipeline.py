"""Tests for the SMT pipeline module."""

import pytest

from staged_qwen3_5_scivqa.smt.pipeline import (
    clean_duplicate_declarations,
    deduplicate_anchors,
    parse_table_deterministically,
)


@pytest.mark.unit
def test_clean_duplicate_declarations() -> None:
    """Test removal of duplicate declarations."""
    declarations = (
        "(declare-const x_entity Entity)\n"
        "(declare-const x_entity Entity)\n"
        "(declare-const y_series Series)"
    )

    result = clean_duplicate_declarations(declarations)
    # Should have only one x_entity declaration
    assert result.count("x_entity") == 1


@pytest.mark.unit
def test_deduplicate_anchors() -> None:
    """Test deduplication of (f series x y) assertions."""
    anchors = (
        "(assert (= (f s1 0.0) 1.0))\n"
        "(assert (= (f s1 0.0) 2.0))\n"
        "(assert (= (f s1 1.0) 3.0))"
    )

    result = deduplicate_anchors(anchors)
    # Should keep only the last value for (s1, 0.0)
    assert result.count("(f s1 0.0)") == 1
    assert "2.0" in result  # Last value wins


@pytest.mark.unit
def test_parse_table_deterministically_simple() -> None:
    """Test table parsing with a simple 2-column table."""
    table = "Time,Value\n0,1.0\n1,2.0\n2,3.0"
    declarations, anchors, valid_numbers = parse_table_deterministically(table)

    assert "Entity" in declarations
    assert "Series" in declarations
    assert len(valid_numbers) > 0


@pytest.mark.unit
def test_parse_table_deterministically_markdown() -> None:
    """Test table parsing with Markdown format."""
    table = "| Time | Value |\n|---|---|\n| 0 | 1.0 |\n| 1 | 2.0 |"
    declarations, anchors, valid_numbers = parse_table_deterministically(table)

    assert "Entity" in declarations
    assert "Series" in declarations


@pytest.mark.unit
def test_parse_table_deterministically_empty() -> None:
    """Test table parsing with empty input."""
    declarations, anchors, valid_numbers = parse_table_deterministically("")
    assert declarations == ""
    assert anchors == ""
    assert valid_numbers == []


@pytest.mark.unit
def test_parse_table_deterministically_special_chars() -> None:
    """Test table parsing with special characters in headers."""
    table = "Temp (°C),Rate (nm/cycle);100,0.5;200,0.8"
    declarations, anchors, valid_numbers = parse_table_deterministically(table)

    assert "Entity" in declarations
    assert "Series" in declarations
