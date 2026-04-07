from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any
import yaml

from core.case.case_context import CaseContext

logger = logging.getLogger(__name__)


@dataclass
class DocumentRequirement:
    category: str
    doc_types: list[str]
    min_required: int
    max_accepted: int
    role: str
    validates_fields_in: dict[str, list[str]] = field(default_factory=dict)
    condition: str | None = None
    temporal_constraints: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class CrossValidationRule:
    name: str
    description: str
    sources: list[str]
    on_fail: str = "flag_for_review"
    field_mapping: dict[str, str] = field(default_factory=dict)
    fields: list[str] = field(default_factory=list)
    match_type: str = "exact"
    tolerance_percent: float | None = None
    logic: str | None = None


@dataclass
class ProcessingPhase:
    name: str
    groups: list[list[str]] | None = None
    sequential: list[str] | None = None


@dataclass
class DocumentGraph:
    """Represents the full document dependency structure for a credit process."""
    process: str
    requirements: dict[str, DocumentRequirement]
    cross_validation_rules: list[CrossValidationRule]
    processing_phases: list[ProcessingPhase]

    @classmethod
    def from_yaml(cls, path: str) -> DocumentGraph:
        with open(path) as f:
            raw = yaml.safe_load(f)

        requirements = {}
        for cat, req in raw["document_requirements"].items():
            requirements[cat] = DocumentRequirement(category=cat, **req)

        cv_rules = [
            CrossValidationRule(**r)
            for r in raw.get("cross_validation_rules", [])
        ]

        phases = []
        for phase_def in raw.get("processing_order", []):
            phase = ProcessingPhase(name=phase_def["phase"])
            if "parallel_groups" in phase_def:
                phase.groups = phase_def["parallel_groups"]
            if "sequential" in phase_def:
                phase.sequential = phase_def["sequential"]
            phases.append(phase)

        return cls(
            process=raw["process"],
            requirements=requirements,
            cross_validation_rules=cv_rules,
            processing_phases=phases,
        )


class CompletenessChecker:
    """Validates that a case has all required documents before processing."""

    def __init__(self, graph: DocumentGraph):
        self._graph = graph

    def check(
        self, case: CaseContext, anchor_data: dict[str, Any] | None = None
    ) -> tuple[bool, list[str]]:
        issues: list[str] = []

        for cat, req in self._graph.requirements.items():
            if req.condition and anchor_data:
                if not self._evaluate_condition(req.condition, anchor_data):
                    logger.info(
                        f"Skipping requirement '{cat}': "
                        f"condition '{req.condition}' not met"
                    )
                    continue

            docs = case.get_documents_by_category(cat)
            matching = [d for d in docs if d.doc_type in req.doc_types]

            if len(matching) < req.min_required:
                issues.append(
                    f"Category '{cat}': need at least {req.min_required} "
                    f"of {req.doc_types}, found {len(matching)}"
                )

            if len(matching) > req.max_accepted:
                issues.append(
                    f"Category '{cat}': max {req.max_accepted} accepted, "
                    f"found {len(matching)}"
                )

            for doc_type, constraints in req.temporal_constraints.items():
                typed_docs = [d for d in matching if d.doc_type == doc_type]
                min_count = constraints.get("min_count", 1)
                if len(typed_docs) < min_count:
                    issues.append(
                        f"Category '{cat}', type '{doc_type}': "
                        f"need {min_count}, found {len(typed_docs)}"
                    )

        return len(issues) == 0, issues

    @staticmethod
    def _evaluate_condition(condition: str, data: dict[str, Any]) -> bool:
        try:
            flat = {}
            for key, val in data.items():
                if isinstance(val, dict):
                    for k2, v2 in val.items():
                        flat[f"{key}.{k2}"] = v2
                else:
                    flat[key] = val

            expr = condition
            for key, val in sorted(flat.items(), key=lambda x: -len(x[0])):
                if key in expr:
                    expr = expr.replace(key, repr(val))

            allowed_names = {"True": True, "False": False, "None": None}
            return bool(eval(expr, {"__builtins__": {}}, allowed_names))
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {condition} -- {e}")
            return True
