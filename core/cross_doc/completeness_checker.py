from __future__ import annotations
import logging
from typing import Any

from core.case.case_context import CaseContext, DocumentStatus

logger = logging.getLogger(__name__)


class DocumentCompletenessChecker:
    """
    Verifies that all extracted documents have the expected fields populated.
    Different from dependency_graph.CompletenessChecker which checks document presence.
    This checks extraction quality within each document.
    """

    def check_extraction_quality(
        self, case: CaseContext
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "total_documents": len(case.documents),
            "extracted": 0,
            "failed": 0,
            "quality_issues": [],
        }

        for doc_id, doc in case.documents.items():
            if doc.status == DocumentStatus.EXTRACTED:
                results["extracted"] += 1

                none_fields = [
                    k for k, v in doc.extracted_data.items()
                    if v is None and not k.startswith("_")
                ]
                if none_fields:
                    results["quality_issues"].append({
                        "doc_id": doc_id,
                        "doc_type": doc.doc_type,
                        "null_fields": none_fields,
                        "extraction_completeness": round(
                            1 - len(none_fields) / max(len(doc.extracted_data), 1), 2
                        ),
                    })

            elif doc.status == DocumentStatus.FAILED:
                results["failed"] += 1
                results["quality_issues"].append({
                    "doc_id": doc_id,
                    "doc_type": doc.doc_type,
                    "issue": "extraction_failed",
                    "extraction_completeness": 0.0,
                })

        results["overall_quality"] = round(
            results["extracted"] / max(results["total_documents"], 1), 2
        )

        return results