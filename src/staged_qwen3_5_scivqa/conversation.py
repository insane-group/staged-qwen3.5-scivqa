"""Conversation formatting for Qwen/Unsloth multimodal models."""

from typing import Any

from PIL import Image


def convert_to_conversation(prompt: str, image: Image.Image, response: str) -> dict:
    """Build a Qwen/Unsloth conversation dict from prompt, image, and response.

    Args:
        prompt: The user-side text prompt (may include <image> token).
        image: A PIL Image object.
        response: The assistant-side ground-truth text.

    Returns:
        A dict with a "messages" key containing the conversation list.

    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
    ]
    return {"messages": conversation}


def convert_to_inference_conversation(
    prompt: str, image: Image.Image, **metadata: Any
) -> dict:
    """Build an inference-only conversation dict (no assistant role).

    Args:
        prompt: The user-side text prompt.
        image: A PIL Image object.
        **metadata: Additional key-value pairs to store in the "meta" field.

    Returns:
        A dict with "messages" and "meta" keys.

    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]
    return {"messages": conversation, "meta": metadata}
