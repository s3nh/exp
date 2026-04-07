from __future__ import annotations
from typing import Any
from core.guardrails.verdict import GuardrailVerdict


class InputGuardChain:
    """Pre-LLM input validation chain."""

    def __init__(self, config: dict[str, Any]):
        self._config = config

    def check_length(self, prompt: str) -> GuardrailVerdict:
        max_prompt = self._config.get("max_prompt_length", 32000)
        if len(prompt) > max_prompt:
            return GuardrailVerdict(
                passed=False,
                blocked_reason=f"Prompt exceeds max length ({len(prompt)} > {max_prompt})",
                flags=["INPUT_TOO_LONG"],
            )
        return GuardrailVerdict(passed=True)