from __future__ import annotations
import json
import logging
from typing import Any

from core.inference import InferenceRouter
from core.guardrails.engine import GuardrailEngine

logger = logging.getLogger(__name__)


class ContradictionDetector:
    """
    Detects contradictions between documents in a credit case.
    Critical for identifying fraud signals or data quality issues.
    """

    def __init__(self, router: InferenceRouter, guardrails: GuardrailEngine):
        self._router = router
        self._guardrails = guardrails

    async def detect(
        self,
        documents_data: dict[str, dict[str, Any]],
        focus_areas: list[str] | None = None,
    ) -> dict[str, Any]:
        focus = ""
        if focus_areas:
            focus = f"\n\nFocus especially on: {', '.join(focus_areas)}"

        prompt = f"""You are a fraud detection and data quality system for credit applications.
Analyze the following extracted data from multiple documents and identify any contradictions.

## Document Data
{json.dumps(documents_data, indent=2)}
{focus}

Look for:
1. Income amounts that don't match across documents
2. Employment details that contradict each other
3. Dates that don't make logical sense
4. Address discrepancies
5. Debt amounts not matching between application and bureau report
6. Any other logical contradictions

Respond with JSON:
{{
  "contradictions_found": true/false,
  "contradictions": [
    {{
      "type": "income_mismatch|employment_conflict|date_inconsistency|address_mismatch|debt_discrepancy|other",
      "severity": "low|medium|high|critical",
      "description": "detailed explanation",
      "documents_involved": ["doc1", "doc2"],
      "fields_involved": ["field1", "field2"],
      "fraud_signal": true/false
    }}
  ],
  "overall_risk_assessment": "low|medium|high",
  "recommendation": "proceed|flag_for_review|escalate|reject"
}}"""

        try:
            raw = await self._router.run_raw(prompt, temperature=0.0)
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
            return {
                "contradictions_found": None,
                "error": str(e),
                "recommendation": "flag_for_review",
            }