from __future__ import annotations
import re
import logging
from typing import Any
from core.guardrails.engine import GuardrailVerdict

logger = logging.getLogger(__name__)


class PIIScanner:
    """
    Config-driven PII scanner.
    Loads global + country-specific patterns from YAML.
    Supports redaction, flagging, or blocking.
    """

    def __init__(self, pii_config: dict[str, Any]):
        self._global_patterns = self._compile_patterns(
            pii_config.get("global", [])
        )
        self._country_patterns: dict[str, list] = {}
        for country, patterns in pii_config.get("country_specific", {}).items():
            self._country_patterns[country] = self._compile_patterns(patterns)

    def _compile_patterns(
        self, pattern_defs: list[dict]
    ) -> list[dict[str, Any]]:
        compiled = []
        for p in pattern_defs:
            compiled.append(
                {
                    "name": p["name"],
                    "regex": re.compile(p["pattern"]),
                    "replacement": p["replacement"],
                    "sensitivity": p.get("sensitivity", "medium"),
                }
            )
        return compiled

    def scan_and_redact(
        self, text: str, country: str | None = None
    ) -> GuardrailVerdict:
        flags: list[str] = []
        sanitized = text
        match_count = 0

        all_patterns = list(self._global_patterns)
        if country and country in self._country_patterns:
            all_patterns.extend(self._country_patterns[country])

        for pattern in all_patterns:
            found = pattern["regex"].findall(sanitized)
            if found:
                match_count += len(found)
                flags.append(f"PII_{pattern['name'].upper()}_DETECTED")
            sanitized = pattern["regex"].sub(pattern["replacement"], sanitized)

        if match_count:
            logger.warning(f"PII detected: {match_count} instance(s)")

        return GuardrailVerdict(
            passed=True,
            flags=flags,
            sanitized_text=sanitized,
            metadata={"pii_matches_count": match_count},
        )
