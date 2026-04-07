from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from enum import Enum

from core.context import PipelineContext


class CaseStatus(Enum):
    INTAKE = "intake"
    CLASSIFYING = "classifying"
    CHECKING_COMPLETENESS = "checking_completeness"
    EXTRACTING = "extracting"
    CHECKING_CONDITIONS = "checking_conditions"
    GENERATING_REPORT = "generating_report"
    READY_FOR_REVIEW = "ready_for_review"
    NEEDS_ANALYST_ACTION = "needs_analyst_action"


class DocumentStatus(Enum):
    PENDING = "pending"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DocumentEntry:
    """A single document within a case."""
    doc_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    doc_type: str = ""
    category: str = ""
    file_name: str = ""
    text_content: str = ""
    upload_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    document_date: str | None = None
    document_period: str | None = None
    status: DocumentStatus = DocumentStatus.PENDING
    pipeline_context: PipelineContext | None = None
    extracted_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossValidationResult:
    """Result of a cross-document validation rule."""
    rule_name: str
    passed: bool
    severity: str = "info"
    details: str = ""
    sources_used: list[str] = field(default_factory=list)
    field_comparisons: list[dict[str, Any]] = field(default_factory=list)
    action: str = "none"


@dataclass
class ClassificationResult:
    doc_id: str
    detected_type: str
    confidence: float
    alternative_types: list[dict[str, Any]]
    reasoning: str
    manually_overridden: bool = False
    override_by: str | None = None


@dataclass
class ConditionCheckResult:
    condition_name: str
    status: str  # "pass" | "fail" | "needs_review"
    severity: str  # "error" | "warning" | "info"
    analyst_message: str
    details: dict[str, Any]
    sources_used: list[str]
    check_type: str  # "deterministic" | "llm_assisted"


@dataclass
class ReviewReport:
    case_id: str
    generated_at: str
    document_inventory: list[dict[str, Any]]
    extraction_results: dict[str, Any]
    condition_results: list[ConditionCheckResult]
    flagged_issues: list[dict[str, Any]]
    narrative_summary: str
    recommendations: list[str]
    requires_analyst_attention: bool


@dataclass
class CaseContext:
    """
    Top-level context for a multi-document analyst review case.
    """
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    country: str = ""
    process_type: str = ""
    status: CaseStatus = CaseStatus.INTAKE
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    documents: dict[str, DocumentEntry] = field(default_factory=dict)
    classifications: dict[str, ClassificationResult] = field(default_factory=dict)
    condition_results: list[ConditionCheckResult] = field(default_factory=list)
    review_report: ReviewReport | None = None
    phase_traces: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # Kept for backward compatibility with cross_doc modules
    cross_validations: list[CrossValidationResult] = field(default_factory=list)

    def add_document(self, entry: DocumentEntry) -> None:
        self.documents[entry.doc_id] = entry

    def get_documents_by_type(self, doc_type: str) -> list[DocumentEntry]:
        return [d for d in self.documents.values() if d.doc_type == doc_type]

    def get_documents_by_category(self, category: str) -> list[DocumentEntry]:
        return [d for d in self.documents.values() if d.category == category]

    def get_all_extracted_data(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for doc_id, doc in self.documents.items():
            if doc.status == DocumentStatus.EXTRACTED and doc.extracted_data:
                key = f"{doc.doc_type}.{doc_id}"
                result[key] = doc.extracted_data
        return result

    def get_merged_extraction(self, doc_type: str) -> dict[str, Any]:
        docs = self.get_documents_by_type(doc_type)
        if len(docs) == 1:
            return docs[0].extracted_data
        return {
            "documents": [
                {"doc_id": d.doc_id, "period": d.document_period, **d.extracted_data}
                for d in docs if d.extracted_data
            ]
        }

    @property
    def has_blocking_flags(self) -> bool:
        return any(
            r.severity == "error" and r.status == "fail"
            for r in self.condition_results
        )

    @property
    def needs_human_review(self) -> bool:
        return any(
            r.status == "needs_review"
            for r in self.condition_results
        )
