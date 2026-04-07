from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any

from core.case.case_context import (
    CaseContext,
    ConditionCheckResult,
    ReviewReport,
    DocumentStatus,
)
from core.context import PipelineContext
from core.guardrails.engine import GuardrailEngine
from core.inference import InferenceRouter
from core.prompt_compiler import PromptCompiler

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


class ReviewReportBuilder:
    """Assembles the full analyst review report from all pipeline phases."""

    def __init__(
        self,
        router: InferenceRouter,
        guardrails: GuardrailEngine,
        template_dir: str = "core/templates",
    ) -> None:
        self._router = router
        self._guardrails = guardrails
        self._compiler = PromptCompiler(template_dir)

    async def build(
        self,
        case: CaseContext,
        pipeline_ctx: PipelineContext,
    ) -> ReviewReport:
        """
        Build the full review report:
        1. Document inventory
        2. Extraction results per document
        3. Condition check results
        4. Flagged issues (sorted by severity)
        5. LLM-generated narrative summary
        6. Analyst recommendations
        """
        document_inventory = self._build_document_inventory(case)
        extraction_results = case.get_all_extracted_data()
        condition_results = case.condition_results

        flagged_issues = self._collect_flagged_issues(condition_results)
        flagged_issues.sort(
            key=lambda x: _SEVERITY_ORDER.get(x.get("severity", "info"), 2)
        )

        narrative_summary = await self._generate_narrative(
            case, pipeline_ctx, document_inventory, extraction_results, condition_results
        )

        recommendations = self._derive_recommendations(condition_results, case)
        requires_attention = case.has_blocking_flags or case.needs_human_review

        return ReviewReport(
            case_id=case.case_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            document_inventory=document_inventory,
            extraction_results=extraction_results,
            condition_results=condition_results,
            flagged_issues=flagged_issues,
            narrative_summary=narrative_summary,
            recommendations=recommendations,
            requires_analyst_attention=requires_attention,
        )

    def _build_document_inventory(self, case: CaseContext) -> list[dict[str, Any]]:
        """Build a list of document inventory entries."""
        inventory: list[dict[str, Any]] = []
        for doc_id, doc in case.documents.items():
            classification = case.classifications.get(doc_id)
            inventory.append(
                {
                    "doc_id": doc_id,
                    "file_name": doc.file_name,
                    "doc_type": doc.doc_type,
                    "category": doc.category,
                    "status": doc.status.value,
                    "upload_timestamp": doc.upload_timestamp,
                    "classified_as": classification.detected_type if classification else doc.doc_type,
                    "classification_confidence": classification.confidence if classification else None,
                    "manually_overridden": classification.manually_overridden if classification else False,
                }
            )
        return inventory

    def _collect_flagged_issues(
        self, condition_results: list[ConditionCheckResult]
    ) -> list[dict[str, Any]]:
        """Collect all non-passing condition results as flagged issues."""
        issues: list[dict[str, Any]] = []
        for result in condition_results:
            if result.status != "pass":
                issues.append(
                    {
                        "condition_name": result.condition_name,
                        "severity": result.severity,
                        "status": result.status,
                        "analyst_message": result.analyst_message,
                        "sources_used": result.sources_used,
                        "details": result.details,
                    }
                )
        return issues

    def _derive_recommendations(
        self,
        condition_results: list[ConditionCheckResult],
        case: CaseContext,
    ) -> list[str]:
        """Derive analyst recommendations from condition results."""
        recommendations: list[str] = []

        error_conditions = [
            r for r in condition_results if r.status == "fail" and r.severity == "error"
        ]
        if error_conditions:
            names = ", ".join(r.condition_name for r in error_conditions)
            recommendations.append(
                f"Blocking issues found in: {names}. Analyst must resolve before proceeding."
            )

        review_conditions = [r for r in condition_results if r.status == "needs_review"]
        if review_conditions:
            names = ", ".join(r.condition_name for r in review_conditions)
            recommendations.append(f"Items requiring analyst review: {names}.")

        failed_docs = [
            d for d in case.documents.values() if d.status == DocumentStatus.FAILED
        ]
        if failed_docs:
            recommendations.append(
                f"{len(failed_docs)} document(s) failed extraction and need manual review."
            )

        unclassified = [
            doc_id
            for doc_id, clf in case.classifications.items()
            if clf.detected_type == "NEEDS_MANUAL_CLASSIFICATION"
        ]
        if unclassified:
            recommendations.append(
                f"{len(unclassified)} document(s) could not be auto-classified: {', '.join(unclassified)}"
            )

        if not recommendations:
            recommendations.append(
                "All automated checks passed. Case is ready for analyst review."
            )

        return recommendations

    async def _generate_narrative(
        self,
        case: CaseContext,
        pipeline_ctx: PipelineContext,
        document_inventory: list[dict[str, Any]],
        extraction_results: dict[str, Any],
        condition_results: list[ConditionCheckResult],
    ) -> str:
        """Generate LLM narrative summary for analyst."""
        try:
            prompt = self._compiler.compile_review_summary_prompt(
                case=case,
                document_inventory=document_inventory,
                extraction_results=extraction_results,
                condition_results=condition_results,
            )

            guard_result = self._guardrails.check_input(prompt, pipeline_ctx)
            if not guard_result.passed:
                logger.warning(
                    "[%s] Narrative prompt blocked: %s", case.case_id, guard_result.blocked_reason
                )
                return self._fallback_narrative(condition_results)

            raw_response = await self._router.generate(
                prompt=prompt,
                temperature=0.2,
            )

            out_guard = self._guardrails.check_output(raw_response, pipeline_ctx)
            return (out_guard.sanitized_text or raw_response).strip()

        except Exception as exc:
            logger.error("[%s] Narrative generation failed: %s", case.case_id, exc)
            return self._fallback_narrative(condition_results)

    def _fallback_narrative(
        self, condition_results: list[ConditionCheckResult]
    ) -> str:
        """Simple rule-based narrative when LLM narrative fails."""
        total = len(condition_results)
        passed = sum(1 for r in condition_results if r.status == "pass")
        failed = sum(1 for r in condition_results if r.status == "fail")
        review = sum(1 for r in condition_results if r.status == "needs_review")
        return (
            f"Automated review complete. {total} conditions checked: "
            f"{passed} passed, {failed} failed, {review} require analyst review. "
            "Please review flagged issues before making a decision."
        )
