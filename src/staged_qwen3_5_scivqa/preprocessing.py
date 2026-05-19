"""Answer preprocessing and validation for all VQA answer types."""

import re


def clean_answer(raw_answer: str, expected_type: str) -> tuple[str, bool]:
    """Clean and validate an answer string for a given answer type.

    Args:
        raw_answer: The raw answer string from the dataset.
        expected_type: One of "Yes/No", "List", "Factoid", "Paragraph".

    Returns:
        Tuple of (cleaned_answer, is_valid_format).

    """
    if not raw_answer or not isinstance(raw_answer, str):
        return "", False

    cleaned = raw_answer.strip()

    if expected_type == "Yes/No":
        ans_lower = cleaned.lower()
        if re.search(r"\byes\b", ans_lower):
            return "Yes", True
        elif re.search(r"\bno\b", ans_lower):
            return "No", True
        return cleaned, False

    elif expected_type == "List":
        elements = [item.strip() for item in cleaned.split(",") if item.strip()]
        return ", ".join(elements), len(elements) > 0

    elif expected_type == "Factoid":
        cleaned_factoid = cleaned.strip()
        return cleaned_factoid, len(cleaned_factoid) > 0

    elif expected_type == "Paragraph":
        cleaned_para = re.sub(r"\s+", " ", cleaned).strip()
        sentences = [s for s in re.split(r"[.!?]+", cleaned_para) if s.strip()]
        return cleaned_para, len(sentences) >= 1

    return cleaned, True


def clean_summary(raw_summary: str) -> tuple[str, bool]:
    """Clean and validate a summary string.

    Removes bullet points, normalizes whitespace, and validates
    that the summary contains at least one sentence.

    Args:
        raw_summary: The raw summary string.

    Returns:
        Tuple of (cleaned_summary, is_valid).

    """
    if not raw_summary or not isinstance(raw_summary, str):
        return "", False

    cleaned = raw_summary.strip()

    # Remove common bullet points at the start of lines
    cleaned = re.sub(r"^\s*[•\-\*]\s+", "", cleaned, flags=re.MULTILINE)

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Validate at least one sentence
    sentences = [s for s in re.split(r"[.!?]+", cleaned) if s.strip()]
    return cleaned, len(sentences) >= 1


def clean_table(raw_table: str) -> tuple[str, bool]:
    """Clean and validate a table extraction string.

    Validates that the table has a header and at least one data row,
    then converts to dense format (commas for columns, semicolons for rows).

    Args:
        raw_table: The raw table string (Markdown or dense format).

    Returns:
        Tuple of (cleaned_table, is_valid).

    """
    if not raw_table or not isinstance(raw_table, str):
        return "", False

    cleaned = raw_table.strip()

    # If already in dense format (contains semicolons), validate structure
    if ";" in cleaned:
        rows = cleaned.split(";")
        return cleaned, len(rows) >= 2

    # If in Markdown format, convert to dense
    grid = parse_markdown_to_grid(cleaned)
    if grid and len(grid) >= 2:
        dense = ";".join(",".join(cell.strip() for cell in row) for row in grid)
        return dense, True

    return cleaned, len(cleaned) > 0


def parse_markdown_to_grid(md_string: str) -> list[list[str]]:
    """Parse a Markdown table string into a 2D grid of cell values.

    Args:
        md_string: A Markdown-formatted table string.

    Returns:
        A list of lists, where each inner list is a row of cell values.

    """
    if not md_string or not isinstance(md_string, str):
        return []

    lines = [line.strip() for line in md_string.strip().split("\n") if line.strip()]
    if len(lines) < 3:
        return []

    grid = []
    for i, line in enumerate(lines):
        if i == 1:  # Skip the markdown separator line (e.g., |---|---|)
            continue

        row_str = re.sub(r"^\||\|$", "", line.strip())
        cells = re.split(r"(?<!\\)\|", row_str)

        cleaned_cells = []
        for cell in cells:
            clean_cell = cell.replace("\\|", "|").strip()
            # Sanitize delimiters so they don't break the dense format downstream
            clean_cell = clean_cell.replace(",", " ").replace(";", " ")
            cleaned_cells.append(clean_cell)

        # Only add the row if it actually contains data
        if any(cleaned_cells):
            grid.append(cleaned_cells)

    return grid


def dense_to_markdown(dense_str: str) -> str:
    """Convert a dense table string back to aligned Markdown format.

    Args:
        dense_str: Dense format string (commas=columns, semicolons=rows).

    Returns:
        An aligned Markdown table string.

    """
    dense_str = dense_str.strip()
    if not dense_str:
        return ""
    rows = [row.split(",") for row in dense_str.split(";")]
    if not rows:
        return ""

    # Calculate column widths
    col_widths = [
        max(len(cell.strip()) for cell in col) for col in zip(*rows, strict=False)
    ]

    lines = []
    for row_idx, row in enumerate(rows):
        cells = [cell.strip().ljust(col_widths[i]) for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(cells) + " |")
        if row_idx == 0:
            separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
            lines.append(separator)

    return "\n".join(lines)
