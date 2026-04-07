from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Any
import yaml

from core.case.case_context import (
    CaseContext,
    CaseStatus,
    DocumentStatus,
    ClassificationResult,
    NEEDS_MANUAL_CLASSIFICATION,
    UNKNOWN_DOCUMENT_TYPE,
)
from core.classification.classifier import DocumentClassifier
from core.conditions.checker import ConditionChecker
from core.review.report_builder import ReviewReportBuilder
from core.pipeline_engine import PipelineEngine, PipelineDefinition
from core.prompt_compiler import PromptCompiler
from core.inference import InferenceRouter
from core.guardrails.engine import GuardrailEngine
from core.context import PipelineContext
from core.utils import sanitize_path_component

logger = logging.getLogger(__name__)


class CaseOrchestrator:
    """
    Analyst document review workflow orchestrator.

    Phases:
    1. CLASSIFYING              — Classify each uploaded document
    2. CHECKING_COMPLETENESS    — Check required documents are present
    3. EXTRACTING               — Per-document data extraction
    4. CHECKING_CONDITIONS      — Run analyst condition checks
    5. GENERATING_REPORT        — Build analyst review report
    """

    def __init__(
        self,
        pipeline_engine: PipelineEngine,
        compiler: PromptCompiler,
        router: InferenceRouter,
        guardrails: GuardrailEngine,
    ) -> None:
        self._pipeline_engine = pipeline_engine
        self._compiler = compiler
        self._router = router
        self._guardrails = guardrails
        self._classifier = DocumentClassifier(router, guardrails)
        self._condition_checker = ConditionChecker(router, guardrails)
        self._report_builder = ReviewReportBuilder(router, guardrails)

    async def process_case(
        self,
        case: CaseContext,
        process_config_path: str,
    ) -> CaseContext:
        """Execute the full analyst review pipeline."""

        process_cfg = _load_process_config(process_config_path)
        country = case.country or process_cfg.get("country", "")
        process_type = case.process_type or process_cfg.get("process", "")

        # ═══════════════════════════════════════════════════════
        # PHASE 1: CLASSIFYING
        # ═══════════════════════════════════════════════════════
        case.status = CaseStatus.CLASSIFYING
        logger.info("[%s] Phase: CLASSIFYING", case.case_id)

        await self._classify_documents(case, country)

        # ═══════════════════════════════════════════════════════
        # PHASE 2: CHECKING_COMPLETENESS
        # ═══════════════════════════════════════════════════════
        case.status = CaseStatus.CHECKING_COMPLETENESS
        logger.info("[%s] Phase: CHECKING_COMPLETENESS", case.case_id)

        completeness_issues = self._check_completeness(case, process_cfg)
        if completeness_issues:
            logger.warning(
                "[%s] Completeness issues: %s", case.case_id, completeness_issues
            )
            case.phase_traces.setdefault("completeness", []).append(
                {"issues": completeness_issues}
            )

        # ═══════════════════════════════════════════════════════
        # PHASE 3: EXTRACTING
        # ═══════════════════════════════════════════════════════
        case.status = CaseStatus.EXTRACTING
        logger.info("[%s] Phase: EXTRACTING", case.case_id)

        pipeline_cfg_path = process_cfg.get(
            "pipeline_config", "configs/pipelines/single_doc_extraction.yaml"
        )
        await self._extract_all_documents(case, country, pipeline_cfg_path)

        # ═══════════════════════════════════════════════════════
        # PHASE 4: CHECKING_CONDITIONS
        # ═══════════════════════════════════════════════════════
        case.status = CaseStatus.CHECKING_CONDITIONS
        logger.info("[%s] Phase: CHECKING_CONDITIONS", case.case_id)

        conditions = self._condition_checker.load_conditions(country, process_type)

        conditions_pipeline_ctx = PipelineContext(
            country=country,
            business_unit=process_cfg.get("business_unit", ""),
            document_type="cross_document",
        )
        case.condition_results = await self._condition_checker.check_all(
            case, conditions, conditions_pipeline_ctx
        )

        # ═══════════════════════════════════════════════════════
        # PHASE 5: GENERATING_REPORT
        # ═══════════════════════════════════════════════════════
        case.status = CaseStatus.GENERATING_REPORT
        logger.info("[%s] Phase: GENERATING_REPORT", case.case_id)

        report_pipeline_ctx = PipelineContext(
            country=country,
            business_unit=process_cfg.get("business_unit", ""),
            document_type="review_report",
        )
        case.review_report = await self._report_builder.build(
            case, report_pipeline_ctx
        )

        if case.has_blocking_flags or case.needs_human_review:
            case.status = CaseStatus.NEEDS_ANALYST_ACTION
        else:
            case.status = CaseStatus.READY_FOR_REVIEW

        logger.info(
            "[%s] Pipeline complete. Status: %s", case.case_id, case.status.value
        )
        return case

    async def _classify_documents(
        self, case: CaseContext, country: str
    ) -> None:
        """Classify all pending documents in parallel."""
        tasks = []
        for doc_id, doc in case.documents.items():
            if not doc.text_content:
                logger.warning(
                    "[%s] doc %s has no text, skipping classification",
                    case.case_id,
                    doc_id,
                )
                continue
            pipeline_ctx = PipelineContext(
                country=country,
                business_unit="",
                document_type="unknown",
                document_text=doc.text_content,
            )
            tasks.append(
                self._classifier.classify(
                    doc_id=doc_id,
                    document_text=doc.text_content,
                    country=country,
                    pipeline_ctx=pipeline_ctx,
                )
            )

        if not tasks:
            return

        results: list[ClassificationResult] = await asyncio.gather(
            *tasks, return_exceptions=True
        )
        for result in results:
            if isinstance(result, Exception):
                logger.error("[%s] Classification error: %s", case.case_id, result)
                continue
            case.classifications[result.doc_id] = result
            if result.doc_id in case.documents:
                doc = case.documents[result.doc_id]
                if result.detected_type != NEEDS_MANUAL_CLASSIFICATION:
                    doc.doc_type = result.detected_type

    def _check_completeness(
        self, case: CaseContext, process_cfg: dict[str, Any]
    ) -> list[str]:
        """Check that required document categories are satisfied."""
        issues: list[str] = []
        required_categories: list[dict[str, Any]] = process_cfg.get(
            "required_categories", []
        )

        for cat in required_categories:
            category = cat.get("category", "")
            doc_types = cat.get("doc_types", [])
            min_required = cat.get("min_required", 0)
            max_accepted = cat.get("max_accepted", 999)
            description = cat.get("description", category)

            matching_docs = [
                d
                for d in case.documents.values()
                if d.doc_type in doc_types or d.category == category
            ]

            count = len(matching_docs)
            if count < min_required:
                issues.append(
                    f"Category '{description}': requires at least {min_required} "
                    f"document(s), found {count}"
                )
            elif count > max_accepted:
                issues.append(
                    f"Category '{description}': accepts at most {max_accepted} "
                    f"document(s), found {count}"
                )

        return issues

    async def _extract_all_documents(
        self,
        case: CaseContext,
        country: str,
        pipeline_cfg_path: str,
    ) -> None:
        """Extract data from all classified documents."""
        pipeline_def: PipelineDefinition | None = None
        try:
            pipeline_def = self._pipeline_engine.load_pipeline(pipeline_cfg_path)
        except FileNotFoundError:
            logger.warning(
                "[%s] Pipeline config '%s' not found — skipping extraction",
                case.case_id,
                pipeline_cfg_path,
            )
            return

        for doc_id, doc in case.documents.items():
            if doc.doc_type in ("", NEEDS_MANUAL_CLASSIFICATION, UNKNOWN_DOCUMENT_TYPE):
                logger.warning(
                    "[%s] Skipping extraction for unclassified doc %s",
                    case.case_id,
                    doc_id,
                )
                doc.status = DocumentStatus.SKIPPED
                continue

            try:
                safe_country = sanitize_path_component(country)
                safe_doc_type = sanitize_path_component(doc.doc_type)
                schema = self._compiler.load_schema(safe_country, safe_doc_type)
            except ValueError as exc:
                logger.warning(
                    "[%s] Unsafe path component for doc %s: %s", case.case_id, doc_id, exc
                )
                doc.status = DocumentStatus.SKIPPED
                continue
            except FileNotFoundError:
                logger.warning(
                    "[%s] No schema for doc_type '%s' — skipping extraction of %s",
                    case.case_id,
                    doc.doc_type,
                    doc_id,
                )
                doc.status = DocumentStatus.SKIPPED
                continue
            except Exception as exc:
                logger.error(
                    "[%s] Failed to load schema for %s: %s", case.case_id, doc_id, exc
                )
                doc.status = DocumentStatus.FAILED
                continue

            pipeline_ctx = PipelineContext(
                country=country,
                business_unit=schema.business_unit,
                document_type=doc.doc_type,
                document_text=doc.text_content,
            )
            doc.pipeline_context = pipeline_ctx

            try:
                await self._pipeline_engine.execute(
                    pipeline=pipeline_def,
                    context=pipeline_ctx,
                    schema=schema,
                )
                doc.extracted_data = _flatten_pipeline_results(pipeline_ctx.results)
                doc.status = DocumentStatus.EXTRACTED
                logger.info(
                    "[%s] Extracted doc %s (%s)", case.case_id, doc_id, doc.doc_type
                )
            except Exception as exc:
                logger.error(
                    "[%s] Extraction failed for doc %s: %s", case.case_id, doc_id, exc
                )
                doc.status = DocumentStatus.FAILED


def _load_process_config(path: str) -> dict[str, Any]:
    """Load process config YAML, returning empty dict if not found.

    Validates the resolved path stays within the configs/ directory before
    opening, preventing path traversal attacks.
    """
    configs_base = Path("configs").resolve()
    resolved = Path(path).resolve()
    # Guard against path traversal — is_relative_to is robust against symlinks
    # and OS-level tricks that str.startswith() would miss.
    if not resolved.is_relative_to(configs_base):
        logger.warning("Process config path outside configs dir: %s", path)
        return {}
    # Reuse the already-validated resolved Path object; never open the raw input
    safe_path = resolved
    if not safe_path.exists():
        logger.warning("Process config not found: %s", path)
        return {}
    with safe_path.open() as f:
        return yaml.safe_load(f) or {}


def _flatten_pipeline_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Flatten all pipeline step results, with later steps overriding earlier."""
    flat: dict[str, Any] = {}
    for step_data in results.values():
        flat.update(step_data)
    return flat
