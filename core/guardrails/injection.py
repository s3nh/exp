from __future__ import annotations
import re
import math
import logging
from collections import Counter
from core.guardrails.verdict import GuardrailVerdict

logger = logging.getLogger(__name__)


class InjectionDetector:
    """
    Multi-layer prompt injection detection.
    Layer 1: Regex pattern matching (fast, deterministic)
    Layer 2: Structural analysis (detects role-hijacking attempts)
    Layer 3: Entropy analysis (detects encoded/obfuscated payloads)
    """

    def __init__(self, blocked_patterns: list[str]):
        self._patterns = [re.compile(p) for p in blocked_patterns]

    def scan(self, text: str) -> GuardrailVerdict:
        flags: list[str] = []

        # Layer 1: Pattern matching
        for pattern in self._patterns:
            if pattern.search(text):
                return GuardrailVerdict(
                    passed=False,
                    blocked_reason=f"Injection pattern detected: {pattern.pattern}",
                    flags=["INJECTION_PATTERN_MATCH"],
                )

        # Layer 2: Structural analysis
        role_injection_markers = [
            r"```\s*(system|assistant|user)\s*\n",
            r"<\|?(system|assistant|user)\|?>",
            r"\n(system|assistant|user)\s*:",
        ]
        for marker in role_injection_markers:
            if re.search(marker, text, re.IGNORECASE):
                flags.append("STRUCTURAL_INJECTION_SUSPECT")
                return GuardrailVerdict(
                    passed=False,
                    blocked_reason="Role injection structure detected in input",
                    flags=flags,
                )

        # Layer 3: Entropy check
        if self._high_entropy_segment(text):
            flags.append("HIGH_ENTROPY_SEGMENT")

        return GuardrailVerdict(passed=True, flags=flags)

    @staticmethod
    def _high_entropy_segment(
        text: str, window: int = 100, threshold: float = 4.5
    ) -> bool:
        """Sliding window entropy check for obfuscated injection payloads."""
        for i in range(0, max(1, len(text) - window), window // 2):
            segment = text[i : i + window]
            if not segment:
                continue
            freq = Counter(segment)
            length = len(segment)
            entropy = -sum(
                (c / length) * math.log2(c / length) for c in freq.values()
            )
            if entropy > threshold:
                return True
        return False
