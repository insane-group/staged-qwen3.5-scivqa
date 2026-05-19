"""Paper context extraction from structured content.json files."""

import json
from pathlib import Path


def get_paper_context(json_file_path: Path, window_size: int = 2) -> str:
    """Extract sliding-window text context around a figure from content.json.

    Finds the parent content.json, extracts the image caption, and grabs
    a sliding window of text blocks (e.g. 2 before, 2 after) surrounding
    the image for highly targeted context.

    Args:
        json_file_path: Path to the figure annotation JSON (e.g. images/fig_2.json).
        window_size: Number of text blocks to include before and after the image.

    Returns:
        A string containing the caption and surrounding text blocks.

    """
    content_json_path = json_file_path.parent.parent / "content.json"

    if not content_json_path.exists():
        return "Specific context not found for this image."

    target_img_path = f"images/{json_file_path.stem}.jpg"

    with open(content_json_path, encoding="utf-8") as f:
        content_data = json.load(f)

    img_index = -1
    caption_text = ""

    for idx, block in enumerate(content_data):
        if block.get("type") == "image" and block.get("img_path") == target_img_path:
            img_index = idx
            if "img_caption" in block and block["img_caption"]:
                caption_text = " ".join(block["img_caption"])
            break

    if img_index == -1:
        return "Specific context not found for this image."

    text_before: list[str] = []
    for i in range(img_index - 1, -1, -1):
        block = content_data[i]
        if block.get("type") == "text" and "text" in block:
            text_before.insert(0, block["text"])
            if len(text_before) == window_size:
                break

    text_after = []
    for i in range(img_index + 1, len(content_data)):
        block = content_data[i]
        if block.get("type") == "text" and "text" in block:
            text_after.append(block["text"])
            if len(text_after) == window_size:
                break

    context_blocks = []
    if caption_text:
        context_blocks.append(f"Image Caption: {caption_text}")

    context_blocks.extend(text_before)
    context_blocks.extend(text_after)

    return "\n\n".join(context_blocks)
