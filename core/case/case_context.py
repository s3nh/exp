from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from enum import Enum

from core.context import PipelineContext


class CaseStatus(Enum):
    INTAKE = "intake"
    EXTRACTION = "extraction"
    CROSS_VALIDATION = "cross_validation"
    ANALYSIS = "analysis"
    DECISION = "decision"
    HUMAN_REVIEW = "human_review"
    COMPLETED = "completed"
    REJECTED = "rejected"


class DocumentStatus(Enum):
    PENDING = "pending"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DocumentEntry:
    """A single document within a credit case."""
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
class FinancialAnalysis:
    """Aggregated financial analysis across all documents."""
    declared_monthly_income: float | None = None
    verified_monthly_income: float | None = None
    income_sources: list[dict[str, Any]] = field(default_factory=list)
    income_confidence: float = 0.0
    total_monthly_debt_service: float | None = None
    declared_debts: list[dict[str, Any]] = field(default_factory=list)
    bureau_debts: list[dict[str, Any]] = field(default_factory=list)
    undeclared_debts: list[dict[str, Any]] = field(default_factory=list)
    dti_ratio: float | None = None
    dscr: float | None = None
    ltv_ratio: float | None = None
    financial_spreading: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    reasoning: dict[str, str] = field(default_factory=dict)


@dataclass
class CreditDecision:
    """Final credit decision output."""
    decision: str = ""
    confidence: float = 0.0
    rationale: str = ""
    conditions: list[str] = field(default_factory=list)
    risk_grade: str = ""
    flags: list[str] = field(default_factory=list)
    human_review_reasons: list[str] = field(default_factory=list)


@dataclass
class CaseContext:
    """
    Top-level context for a multi-document credit case.
    """
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    country: str = ""
    process_type: str = ""
    status: CaseStatus = CaseStatus.INTAKE
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    documents: dict[str, DocumentEntry] = field(default_factory=dict)
    cross_validations: list[CrossValidationResult] = field(default_factory=list)
    financial_analysis: FinancialAnalysis = field(default_factory=FinancialAnalysis)
    credit_decision: CreditDecision | None = None
    phase_traces: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

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
            cv.action == "block" and not cv.passed
            for cv in self.cross_validations
        )

    @property
    def needs_human_review(self) -> bool:
        return any(
            cv.action in ("flag_for_review", "escalate") and not cv.passed
            for cv in self.cross_validations
        )
