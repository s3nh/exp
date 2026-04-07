from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailVerdict:
    passed: bool
    flags: list[str] = field(default_factory=list)
    blocked_reason: str | None = None
    sanitized_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
