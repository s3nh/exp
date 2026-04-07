from __future__ import annotations
"""Shared utility functions for the pipeline."""
import json
import re
from typing import Any

_SAFE_COMPONENT_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from text that may wrap a JSON block."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        cleaned = "\n".join(lines[start:end])
    return cleaned


def try_parse_json(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse JSON from text, stripping markdown fences. Returns (parsed, error)."""
    cleaned = strip_json_fences(text)
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result, ""
        return None, f"Expected JSON object, got {type(result).__name__}"
    except json.JSONDecodeError as exc:
        return None, str(exc)


def sanitize_path_component(value: str) -> str:
    """
    Ensure a path component is safe (alphanumeric + underscore only).
    Raises ValueError for any component that could enable path traversal.
    """
    if not _SAFE_COMPONENT_RE.match(value):
        raise ValueError(
            f"Unsafe path component {value!r}: only alphanumeric characters and "
            "underscores are allowed"
        )
    return value
