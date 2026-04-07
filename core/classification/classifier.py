from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
import yaml

from core.case.case_context import ClassificationResult, NEEDS_MANUAL_CLASSIFICATION
from core.context import PipelineContext
from core.guardrails.engine import GuardrailEngine
from core.inference import InferenceRouter
from core.prompt_compiler import PromptCompiler

logger = logging.getLogger(__name__)

NEEDS_MANUAL = NEEDS_MANUAL_CLASSIFICATION


class DocumentClassifier:
    """
    LLM-based document classifier.

    Modes:
    - Zero-shot: classify against known types for a country
    - Confidence-gated: if confidence < threshold, flag for analyst
    """

    def __init__(
        self,
        router: InferenceRouter,
        guardrails: GuardrailEngine,
        confidence_threshold: float = 0.75,
        template_dir: str = "core/templates",
    ) -> None:
        self._router = router
        self._guardrails = guardrails
        self._confidence_threshold = confidence_threshold
        self._compiler = PromptCompiler(template_dir)
        self._country_types_cache: dict[str, list[dict[str, Any]]] = {}

    def load_country_doc_types(self, country: str) -> list[dict[str, Any]]:
        """Load registered document types for a country from YAML."""
        if country in self._country_types_cache:
            return self._country_types_cache[country]

        path = Path(f"configs/countries/{country}/classification.yaml")
        if not path.exists():
            logger.warning("No classification config for country %s at %s", country, path)
            return []

        with open(path) as f:
            raw = yaml.safe_load(f)

        doc_types: list[dict[str, Any]] = raw.get("document_types", [])
        self._country_types_cache[country] = doc_types
        return doc_types

    async def classify(
        self,
        doc_id: str,
        document_text: str,
        country: str,
        pipeline_ctx: PipelineContext,
    ) -> ClassificationResult:
        """
        Classify a document against registered country doc types.

        If confidence < threshold, sets detected_type = NEEDS_MANUAL_CLASSIFICATION.
        """
        registered_types = self.load_country_doc_types(country)

        if not registered_types:
            logger.warning("[%s] No registered types for country %s", doc_id, country)
            return ClassificationResult(
                doc_id=doc_id,
                detected_type=NEEDS_MANUAL,
                confidence=0.0,
                alternative_types=[],
                reasoning="No registered document types found for country.",
            )

        prompt = self._compiler.compile_classification_prompt(
            document_text=document_text,
            registered_types=registered_types,
            country=country,
        )

        guard_result = self._guardrails.check_input(prompt, pipeline_ctx)
        if not guard_result.passed:
            logger.warning("[%s] Classification prompt blocked by guardrails", doc_id)
            return ClassificationResult(
                doc_id=doc_id,
                detected_type=NEEDS_MANUAL,
                confidence=0.0,
                alternative_types=[],
                reasoning=f"Blocked by guardrails: {guard_result.blocked_reason}",
            )

        try:
            raw_response = await self._router.generate(
                prompt=prompt,
                temperature=0.0,
            )
        except Exception as exc:
            logger.error("[%s] LLM call failed during classification: %s", doc_id, exc)
            return ClassificationResult(
                doc_id=doc_id,
                detected_type=NEEDS_MANUAL,
                confidence=0.0,
                alternative_types=[],
                reasoning=f"LLM error: {exc}",
            )

        out_guard = self._guardrails.check_output(raw_response, pipeline_ctx)
        response_text = out_guard.sanitized_text or raw_response

        return self._parse_classification_response(doc_id, response_text)

    def _parse_classification_response(
        self, doc_id: str, response_text: str
    ) -> ClassificationResult:
        """Parse LLM JSON response into ClassificationResult."""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                start = 1
                end = len(lines)
                for i in range(len(lines) - 1, 0, -1):
                    if lines[i].strip() == "```":
                        end = i
                        break
                text = "\n".join(lines[start:end])
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("[%s] Failed to parse classification JSON: %s", doc_id, exc)
            return ClassificationResult(
                doc_id=doc_id,
                detected_type=NEEDS_MANUAL,
                confidence=0.0,
                alternative_types=[],
                reasoning=f"JSON parse error: {exc}. Raw: {response_text[:200]}",
            )

        detected_type: str = data.get("detected_type", NEEDS_MANUAL)
        confidence: float = float(data.get("confidence", 0.0))
        alternatives: list[dict[str, Any]] = data.get("alternative_types", [])
        reasoning: str = data.get("reasoning", "")

        if confidence < self._confidence_threshold:
            detected_type = NEEDS_MANUAL

        return ClassificationResult(
            doc_id=doc_id,
            detected_type=detected_type,
            confidence=confidence,
            alternative_types=alternatives,
            reasoning=reasoning,
        )
