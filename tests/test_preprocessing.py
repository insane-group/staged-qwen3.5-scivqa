"""Tests for the preprocessing module."""

import pytest

from staged_qwen3_5_scivqa.preprocessing import (
    clean_answer,
    clean_summary,
    clean_table,
    dense_to_markdown,
    parse_markdown_to_grid,
)


class TestCleanAnswerYesNo:
    @pytest.mark.unit
    def test_clean_yes(self) -> None:
        result, valid = clean_answer("Yes", "Yes/No")
        assert result == "Yes"
        assert valid is True

    @pytest.mark.unit
    def test_clean_yes_lowercase(self) -> None:
        result, valid = clean_answer("yes", "Yes/No")
        assert result == "Yes"
        assert valid is True

    @pytest.mark.unit
    def test_clean_yes_mixed(self) -> None:
        result, valid = clean_answer("YEs", "Yes/No")
        assert result == "Yes"
        assert valid is True

    @pytest.mark.unit
    def test_clean_no(self) -> None:
        result, valid = clean_answer("No", "Yes/No")
        assert result == "No"
        assert valid is True

    @pytest.mark.unit
    def test_clean_no_lowercase(self) -> None:
        result, valid = clean_answer("no", "Yes/No")
        assert result == "No"
        assert valid is True

    @pytest.mark.unit
    def test_invalid_answer(self) -> None:
        result, valid = clean_answer("maybe", "Yes/No")
        assert valid is False

    @pytest.mark.unit
    def test_avoid_false_positive_eyes(self) -> None:
        """Word boundary check: 'eyes' should not match 'yes'."""
        result, valid = clean_answer("eyes", "Yes/No")
        assert valid is False

    @pytest.mark.unit
    def test_avoid_false_positive_note(self) -> None:
        """Word boundary check: 'note' should not match 'no'."""
        result, valid = clean_answer("note", "Yes/No")
        assert valid is False


class TestCleanAnswerFactoid:
    @pytest.mark.unit
    def test_clean_factoid(self) -> None:
        result, valid = clean_answer("smooth, pit-free copper surface", "Factoid")
        assert result == "smooth, pit-free copper surface"
        assert valid is True

    @pytest.mark.unit
    def test_empty_factoid(self) -> None:
        result, valid = clean_answer("", "Factoid")
        assert valid is False


class TestCleanAnswerList:
    @pytest.mark.unit
    def test_clean_list(self) -> None:
        result, valid = clean_answer("item1, item2, item3", "List")
        assert result == "item1, item2, item3"
        assert valid is True

    @pytest.mark.unit
    def test_empty_list(self) -> None:
        result, valid = clean_answer("", "List")
        assert valid is False


class TestCleanAnswerParagraph:
    @pytest.mark.unit
    def test_clean_paragraph(self) -> None:
        result, valid = clean_answer(
            "Indicates bulk reduction, not only surface effects. "
            "Suggests lithiation of the bulk material.",
            "Paragraph",
        )
        assert valid is True

    @pytest.mark.unit
    def test_paragraph_whitespace_normalization(self) -> None:
        result, valid = clean_answer(
            "First   sentence.  Second   sentence.", "Paragraph"
        )
        assert "  " not in result
        assert valid is True


class TestCleanSummary:
    @pytest.mark.unit
    def test_clean_summary_valid(self) -> None:
        result, valid = clean_summary("The growth rate peaks at 200 C.")
        assert valid is True

    @pytest.mark.unit
    def test_clean_summary_empty(self) -> None:
        result, valid = clean_summary("")
        assert valid is False

    @pytest.mark.unit
    def test_clean_summary_removes_bullets(self) -> None:
        result, valid = clean_summary("- First point\n- Second point.")
        assert result.startswith("First point")
        assert valid is True


class TestCleanTable:
    @pytest.mark.unit
    def test_clean_table_dense(self) -> None:
        result, valid = clean_table("a,b;c,d;e,f")
        assert valid is True

    @pytest.mark.unit
    def test_clean_table_markdown(self) -> None:
        md = "| a | b |\n|---|---|\n| c | d |"
        result, valid = clean_table(md)
        assert valid is True

    @pytest.mark.unit
    def test_clean_table_empty(self) -> None:
        result, valid = clean_table("")
        assert valid is False


class TestParseMarkdownToGrid:
    @pytest.mark.unit
    def test_parse_simple_table(self) -> None:
        md = "| a | b |\n|---|---|\n| c | d |"
        grid = parse_markdown_to_grid(md)
        assert len(grid) == 2
        assert grid[0] == ["a", "b"]
        assert grid[1] == ["c", "d"]

    @pytest.mark.unit
    def test_parse_empty(self) -> None:
        grid = parse_markdown_to_grid("")
        assert grid == []


class TestDenseToMarkdown:
    @pytest.mark.unit
    def test_dense_to_markdown(self) -> None:
        dense = "a,b;c,d"
        result = dense_to_markdown(dense)
        assert "|" in result
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "d" in result

    @pytest.mark.unit
    def test_dense_to_markdown_empty(self) -> None:
        result = dense_to_markdown("")
        assert result == ""
