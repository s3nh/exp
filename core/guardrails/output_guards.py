from __future__ import annotations
import json
import re
from typing import Any
from core.guardrails.verdict import GuardrailVerdict


class OutputGuardChain:
    """Post-LLM output validation chain. All config-driven."""

    def __init__(self, config: dict[str, Any]):
        self._config = config

    def check_length(self, response: str) -> GuardrailVerdict:
        max_len = self._config.get("max_response_length", 10000)
        if len(response) > max_len:
            return GuardrailVerdict(
                passed=False,
                blocked_reason=f"Response exceeds max length ({len(response)} > {max_len})",
                flags=["OUTPUT_TOO_LONG"],
            )
        return GuardrailVerdict(passed=True)

    def check_json(self, response: str) -> GuardrailVerdict:
        if not self._config.get("must_be_valid_json", False):
            return GuardrailVerdict(passed=True)
        try:
            json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
            text_to_parse = json_match.group(1) if json_match else response
            json.loads(text_to_parse)
            return GuardrailVerdict(passed=True, sanitized_text=text_to_parse)
        except (json.JSONDecodeError, AttributeError):
            return GuardrailVerdict(
                passed=False,
                blocked_reason="LLM response is not valid JSON",
                flags=["INVALID_JSON"],
            )

    def check_blocked_content(self, response: str) -> GuardrailVerdict:
        flags: list[str] = []
        for pattern in self._config.get("block_if_contains", []):
            if re.search(pattern, response):
                return GuardrailVerdict(
                    passed=False,
                    blocked_reason=f"Response contains blocked content: {pattern}",
                    flags=["BLOCKED_CONTENT_DETECTED"],
                )
        return GuardrailVerdict(passed=True, flags=flags)

    def check_field_hallucination(
        self, response: str, expected_fields: list[str]
    ) -> GuardrailVerdict:
        """Detect if LLM invented fields not in the schema."""
        flags: list[str] = []
        try:
            json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
            text_to_parse = json_match.group(1) if json_match else response
            parsed = json.loads(text_to_parse)
            if isinstance(parsed, dict):
                actual_keys = {
                    k for k in parsed.keys()
                    if not k.startswith("_") and not k.endswith("_reasoning")
                }
                expected_set = set(expected_fields)
                hallucinated = actual_keys - expected_set
                if hallucinated:
                    flags.append(
                        f"HALLUCINATED_FIELDS:{','.join(hallucinated)}"
                    )
        except (json.JSONDecodeError, AttributeError):
            pass
        return GuardrailVerdict(passed=True, flags=flags)
