from __future__ import annotations
"""FastAPI endpoints for the analyst document review tool."""

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from pydantic import BaseModel

app = FastAPI(
    title="Analyst Document Review API",
    description="LLM-assisted document review tool for credit analysts",
    version="1.0.0",
)

# In-memory case store (replace with DB in production)
_cases: dict[str, dict[str, Any]] = {}

# Allowed process config paths — prevents arbitrary file reads via the API
_ALLOWED_PROCESS_CONFIGS: set[str] = {
    "configs/countries/germany/process/consumer_credit.yaml",
}


class CreateCaseRequest(BaseModel):
    country: str
    process_type: str


class ReclassifyRequest(BaseModel):
    doc_type: str
    analyst_id: str


@app.post("/cases", status_code=status.HTTP_201_CREATED)
async def create_case(request: CreateCaseRequest) -> dict[str, Any]:
    """Create a new analyst review case."""
    from core.case.case_context import CaseContext

    case_id = uuid.uuid4().hex[:10]
    case = CaseContext(
        case_id=case_id,
        country=request.country,
        process_type=request.process_type,
    )
    _cases[case_id] = {"case": case, "documents": {}}
    return {
        "case_id": case_id,
        "country": case.country,
        "process_type": case.process_type,
        "status": case.status.value,
        "created_at": case.created_at,
    }


@app.post("/cases/{case_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    case_id: str, file: UploadFile = File(...)
) -> dict[str, Any]:
    """Upload a document to a case."""
    from core.case.case_context import DocumentEntry

    if case_id not in _cases:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    content = await file.read()
    text_content = content.decode("utf-8", errors="replace")

    doc = DocumentEntry(
        file_name=file.filename or "unknown",
        text_content=text_content,
    )
    case = _cases[case_id]["case"]
    case.add_document(doc)
    _cases[case_id]["documents"][doc.doc_id] = doc

    return {
        "case_id": case_id,
        "doc_id": doc.doc_id,
        "file_name": doc.file_name,
        "status": doc.status.value,
        "upload_timestamp": doc.upload_timestamp,
    }


@app.post("/cases/{case_id}/process")
async def process_case(
    case_id: str,
    process_config: str = "configs/countries/germany/process/consumer_credit.yaml",
) -> dict[str, Any]:
    """Run the full analyst review pipeline for a case."""
    from core.case.case_context import CaseContext

    if case_id not in _cases:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    if process_config not in _ALLOWED_PROCESS_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown process config '{process_config}'. "
            f"Allowed values: {sorted(_ALLOWED_PROCESS_CONFIGS)}",
        )

    case: CaseContext = _cases[case_id]["case"]

    try:
        from core.case.case_orchestrator import CaseOrchestrator
        from core.pipeline_engine import PipelineEngine
        from core.prompt_compiler import PromptCompiler
        from core.inference import InferenceRouter
        from core.guardrails.engine import GuardrailEngine

        router = InferenceRouter()
        guardrails = GuardrailEngine()
        compiler = PromptCompiler()
        engine = PipelineEngine(router, guardrails)
        orchestrator = CaseOrchestrator(engine, compiler, router, guardrails)

        updated_case = await orchestrator.process_case(case, process_config)
        _cases[case_id]["case"] = updated_case

        return {
            "case_id": case_id,
            "status": updated_case.status.value,
            "requires_analyst_attention": (
                updated_case.review_report.requires_analyst_attention
                if updated_case.review_report
                else None
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/cases/{case_id}/report")
async def get_report(case_id: str) -> dict[str, Any]:
    """Get the analyst review report for a case."""
    if case_id not in _cases:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    case = _cases[case_id]["case"]

    if case.review_report is None:
        raise HTTPException(
            status_code=404,
            detail="Report not yet generated. Run POST /cases/{case_id}/process first.",
        )

    report = case.review_report
    return {
        "case_id": report.case_id,
        "generated_at": report.generated_at,
        "document_inventory": report.document_inventory,
        "condition_results": [
            {
                "condition_name": r.condition_name,
                "status": r.status,
                "severity": r.severity,
                "analyst_message": r.analyst_message,
                "check_type": r.check_type,
            }
            for r in report.condition_results
        ],
        "flagged_issues": report.flagged_issues,
        "narrative_summary": report.narrative_summary,
        "recommendations": report.recommendations,
        "requires_analyst_attention": report.requires_analyst_attention,
    }


@app.post("/cases/{case_id}/documents/{doc_id}/reclassify")
async def reclassify_document(
    case_id: str, doc_id: str, request: ReclassifyRequest
) -> dict[str, Any]:
    """Override the classification of a document (analyst manual override)."""
    from core.case.case_context import ClassificationResult

    if case_id not in _cases:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    case = _cases[case_id]["case"]

    if doc_id not in case.documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found in case '{case_id}'",
        )

    doc = case.documents[doc_id]
    old_type = doc.doc_type

    doc.doc_type = request.doc_type

    existing = case.classifications.get(doc_id)
    case.classifications[doc_id] = ClassificationResult(
        doc_id=doc_id,
        detected_type=request.doc_type,
        confidence=1.0,
        alternative_types=existing.alternative_types if existing else [],
        reasoning=existing.reasoning if existing else "",
        manually_overridden=True,
        override_by=request.analyst_id,
    )

    return {
        "case_id": case_id,
        "doc_id": doc_id,
        "old_type": old_type,
        "new_type": request.doc_type,
        "overridden_by": request.analyst_id,
    }
