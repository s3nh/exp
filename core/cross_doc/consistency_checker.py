from __future__ import annotations
import json
import logging
from typing import Any

from core.inference import InferenceRouter
from core.guardrails.engine import GuardrailEngine

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """
    Checks consistency of values across multiple documents.
    Uses deterministic checks for numeric comparisons,
    LLM for semantic/fuzzy comparisons.
    """

    def __init__(self, router: InferenceRouter, guardrails: GuardrailEngine):
        self._router = router
        self._guardrails = guardrails

    def check_numeric_consistency(
        self,
        values: dict[str, float | None],
        tolerance_percent: float = 10.0,
    ) -> dict[str, Any]:
        """Deterministic numeric consistency check."""
        valid_values = {k: v for k, v in values.items() if v is not None}

        if len(valid_values) < 2:
            return {
                "is_consistent": True,
                "reason": "Insufficient values to compare",
                "values": valid_values,
            }

        nums = list(valid_values.values())
        avg = sum(nums) / len(nums)
        tolerance = tolerance_percent / 100.0

        outliers = {}
        for source, val in valid_values.items():
            deviation = abs(val - avg) / max(avg, 1)
            if deviation > tolerance:
                outliers[source] = {
                    "value": val,
                    "deviation_percent": round(deviation * 100, 1),
                }

        return {
            "is_consistent": len(outliers) == 0,
            "average": round(avg, 2),
            "tolerance_percent": tolerance_percent,
            "outliers": outliers,
            "values": valid_values,
        }

    async def check_semantic_consistency(
        self,
        field_name: str,
        values: dict[str, str],
    ) -> dict[str, Any]:
        """LLM-assisted semantic consistency check for text fields."""
        prompt = f"""Compare these values for the field "{field_name}" from different documents.
Determine if they refer to the same thing despite possible formatting differences.

Values from different sources:
{json.dumps(values, indent=2)}

Respond with JSON:
{{
  "is_consistent": true/false,
  "canonical_value": "the most accurate/complete version",
  "variations_explained": "explanation of differences",
  "confidence": 0.0-1.0
}}"""

        try:
            raw = await self._router.run_raw(prompt, temperature=0.0)
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Semantic consistency check failed: {e}")
            return {
                "is_consistent": None,
                "confidence": 0.0,
                "error": str(e),
            }