from __future__ import annotations
import json
import logging
from typing import Any

from core.inference import InferenceRouter
from core.guardrails.engine import GuardrailEngine

logger = logging.getLogger(__name__)


class EntityResolver:
    """
    Resolves whether entities across documents refer to the same
    real-world person or company. Uses fuzzy LLM matching for names
    with spelling variations, transliterations, abbreviations.
    """

    def __init__(self, router: InferenceRouter, guardrails: GuardrailEngine):
        self._router = router
        self._guardrails = guardrails

    async def resolve(
        self,
        entity_a: dict[str, Any],
        entity_b: dict[str, Any],
        entity_type: str = "person",
    ) -> dict[str, Any]:
        prompt = f"""You are an entity resolution system. Determine if these two {entity_type} records
refer to the same real-world entity.

## Entity A
{json.dumps(entity_a, indent=2)}

## Entity B
{json.dumps(entity_b, indent=2)}

Consider:
- Minor spelling variations, transliterations, abbreviations
- Date format differences
- Address formatting differences

Respond with JSON:
{{
  "is_same_entity": true/false,
  "confidence": 0.0-1.0,
  "matching_fields": ["field1", "field2"],
  "mismatching_fields": ["field3"],
  "reasoning": "explanation"
}}"""

        try:
            raw = await self._router.run_raw(prompt, temperature=0.0)
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Entity resolution failed: {e}")
            return {
                "is_same_entity": None,
                "confidence": 0.0,
                "reasoning": f"Resolution failed: {e}",
            }