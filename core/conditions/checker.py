from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any
import yaml

from core.case.case_context import CaseContext, ConditionCheckResult, DocumentEntry
from core.context import PipelineContext
from core.guardrails.engine import GuardrailEngine
from core.inference import InferenceRouter
from core.prompt_compiler import PromptCompiler
from core.utils import try_parse_json

logger = logging.getLogger(__name__)


@dataclass
class ConditionDefinition:
    name: str
    description: str
    applies_to: list[str] = field(default_factory=list)  # doc types; empty = all
    check_type: str = "deterministic"  # "deterministic" | "llm_assisted"
    logic: dict[str, Any] | None = None  # for deterministic
    llm_logic: str | None = None  # for llm_assisted
    sources: list[str] | None = None  # for cross-doc
    on_fail: str = "needs_review"
    severity: str = "warning"
    analyst_message: str = ""


class ConditionChecker:
    """
    Runs analyst condition checks dynamically from YAML definitions.
    Supports deterministic and LLM-assisted checks.
    """

    def __init__(
        self,
        router: InferenceRouter,
        guardrails: GuardrailEngine,
        template_dir: str = "core/templates",
    ) -> None:
        self._router = router
        self._guardrails = guardrails
        self._compiler = PromptCompiler(template_dir)

    def load_conditions(
        self, country: str, process_type: str
    ) -> list[ConditionDefinition]:
        """Load condition definitions from configs/countries/{country}/conditions/*.yaml"""
        process_cfg_path = Path(
            f"configs/countries/{country}/process/{process_type}.yaml"
        )
        condition_files: list[str] = []
        if process_cfg_path.exists():
            with open(process_cfg_path) as f:
                proc_cfg = yaml.safe_load(f)
            condition_files = proc_cfg.get("condition_files", [])
        else:
            logger.warning(
                "No process config at %s — loading all condition YAMLs", process_cfg_path
            )
            conditions_dir = Path(f"configs/countries/{country}/conditions")
            if conditions_dir.exists():
                condition_files = [p.stem for p in conditions_dir.glob("*.yaml")]

        conditions: list[ConditionDefinition] = []
        for cf in condition_files:
            yaml_path = Path(f"configs/countries/{country}/conditions/{cf}.yaml")
            if not yaml_path.exists():
                logger.warning("Condition file not found: %s", yaml_path)
                continue
            with open(yaml_path) as f:
                raw = yaml.safe_load(f)
            for item in raw.get("conditions", []):
                conditions.append(ConditionDefinition(**item))

        return conditions

    async def check_all(
        self,
        case: CaseContext,
        conditions: list[ConditionDefinition],
        pipeline_ctx: PipelineContext,
    ) -> list[ConditionCheckResult]:
        """Run all condition checks, returning results with PASS/FAIL/NEEDS_REVIEW."""
        results: list[ConditionCheckResult] = []

        for condition in conditions:
            try:
                if condition.check_type == "deterministic":
                    doc_results = self._run_deterministic(condition, case)
                    results.extend(doc_results)
                elif condition.check_type == "llm_assisted":
                    relevant_data = self._gather_relevant_data(condition, case)
                    result = await self._check_llm_assisted(
                        condition, relevant_data, pipeline_ctx
                    )
                    results.append(result)
                else:
                    logger.warning(
                        "Unknown check_type '%s' for condition '%s'",
                        condition.check_type,
                        condition.name,
                    )
            except Exception as exc:
                logger.error("Error running condition '%s': %s", condition.name, exc)
                results.append(
                    ConditionCheckResult(
                        condition_name=condition.name,
                        status="needs_review",
                        severity=condition.severity,
                        analyst_message=f"Condition check error: {exc}",
                        details={"error": str(exc)},
                        sources_used=[],
                        check_type=condition.check_type,
                    )
                )

        return results

    def _run_deterministic(
        self,
        condition: ConditionDefinition,
        case: CaseContext,
    ) -> list[ConditionCheckResult]:
        """Run deterministic checks against documents matching applies_to."""
        results: list[ConditionCheckResult] = []

        target_docs: list[DocumentEntry] = []
        if condition.applies_to:
            for doc_type in condition.applies_to:
                target_docs.extend(case.get_documents_by_type(doc_type))
        else:
            target_docs = list(case.documents.values())

        if not target_docs:
            return results

        for doc in target_docs:
            result = self._check_deterministic(condition, doc)
            results.append(result)

        return results

    def _check_deterministic(
        self,
        condition: ConditionDefinition,
        doc: DocumentEntry,
    ) -> ConditionCheckResult:
        """
        Deterministic checks:
        - within_months: check if a date field is within N months of today
        - after_today: check date hasn't expired
        - equals, not_equals, greater_than, less_than
        """
        logic = condition.logic or {}
        field_name: str = logic.get("field", "")
        operator: str = logic.get("operator", "")
        expected_value = logic.get("value")

        field_value = doc.extracted_data.get(field_name) or getattr(doc, field_name, None)

        sources_used = [doc.doc_id]

        if field_value is None:
            try:
                message = condition.analyst_message.format(
                    **{field_name: "MISSING"}, doc_id=doc.doc_id
                )
            except (KeyError, ValueError):
                message = f"Field '{field_name}' not found in {doc.doc_id}"
            return ConditionCheckResult(
                condition_name=condition.name,
                status="needs_review",
                severity=condition.severity,
                analyst_message=message,
                details={"field": field_name, "value": None, "reason": "field_missing"},
                sources_used=sources_used,
                check_type="deterministic",
            )

        passed, reason = self._evaluate_operator(operator, field_value, expected_value)

        if passed:
            status = "pass"
        elif condition.on_fail == "fail":
            status = "fail"
        else:
            status = "needs_review"

        try:
            message = condition.analyst_message.format(**{field_name: field_value})
        except (KeyError, ValueError):
            message = condition.analyst_message or reason

        return ConditionCheckResult(
            condition_name=condition.name,
            status=status,
            severity=condition.severity if not passed else "info",
            analyst_message=message if not passed else "Check passed",
            details={
                "field": field_name,
                "value": str(field_value),
                "operator": operator,
                "expected": str(expected_value),
                "reason": reason,
                "doc_id": doc.doc_id,
            },
            sources_used=sources_used,
            check_type="deterministic",
        )

    def _evaluate_operator(
        self, operator: str, value: Any, expected: Any
    ) -> tuple[bool, str]:
        """Evaluate an operator against a value. Returns (passed, reason)."""
        today = date.today()

        if operator == "within_months":
            months = int(expected)
            try:
                doc_date = _parse_date(value)
                cutoff = _months_ago(today, months)
                passed = doc_date >= cutoff
                days_old = (today - doc_date).days
                return passed, f"Date {doc_date} is {days_old} days old (cutoff: {cutoff})"
            except ValueError as exc:
                return False, f"Cannot parse date: {exc}"

        if operator == "after_today":
            try:
                expiry = _parse_date(value)
                passed = expiry >= today
                return passed, f"Expiry {expiry} {'is valid' if passed else 'has passed'}"
            except ValueError as exc:
                return False, f"Cannot parse date: {exc}"

        if operator == "equals":
            return str(value) == str(expected), f"{value} == {expected}"
        if operator == "not_equals":
            return str(value) != str(expected), f"{value} != {expected}"
        if operator == "greater_than":
            try:
                return float(value) > float(expected), f"{value} > {expected}"
            except (TypeError, ValueError):
                return False, f"Cannot compare {value} > {expected}"
        if operator == "less_than":
            try:
                return float(value) < float(expected), f"{value} < {expected}"
            except (TypeError, ValueError):
                return False, f"Cannot compare {value} < {expected}"

        return False, f"Unknown operator: {operator}"

    def _gather_relevant_data(
        self,
        condition: ConditionDefinition,
        case: CaseContext,
    ) -> dict[str, Any]:
        """Gather extracted data from source documents for LLM check."""
        relevant: dict[str, Any] = {}
        sources = condition.sources or []
        for doc_type in sources:
            docs = case.get_documents_by_type(doc_type)
            if docs:
                relevant[doc_type] = [
                    {"doc_id": d.doc_id, **d.extracted_data} for d in docs
                ]
        return relevant

    async def _check_llm_assisted(
        self,
        condition: ConditionDefinition,
        relevant_data: dict[str, Any],
        pipeline_ctx: PipelineContext,
    ) -> ConditionCheckResult:
        """LLM-assisted condition check."""
        sources_used = list(relevant_data.keys())

        if not relevant_data:
            return ConditionCheckResult(
                condition_name=condition.name,
                status="needs_review",
                severity=condition.severity,
                analyst_message="No source documents available for cross-document check",
                details={"reason": "missing_sources"},
                sources_used=sources_used,
                check_type="llm_assisted",
            )

        previous_results = pipeline_ctx.all_results_flat if pipeline_ctx else {}
        prompt = self._compiler.compile_condition_check_prompt(
            condition=condition,
            relevant_data=relevant_data,
            previous_results=previous_results,
            country=pipeline_ctx.country if pipeline_ctx else "",
        )

        guard_result = self._guardrails.check_input(prompt, pipeline_ctx)
        if not guard_result.passed:
            return ConditionCheckResult(
                condition_name=condition.name,
                status="needs_review",
                severity=condition.severity,
                analyst_message=f"Prompt blocked by guardrails: {guard_result.blocked_reason}",
                details={"blocked": True},
                sources_used=sources_used,
                check_type="llm_assisted",
            )

        try:
            raw_response = await self._router.generate(
                prompt=prompt,
                temperature=0.0,
            )
        except Exception as exc:
            logger.error("LLM error in condition '%s': %s", condition.name, exc)
            return ConditionCheckResult(
                condition_name=condition.name,
                status="needs_review",
                severity=condition.severity,
                analyst_message=f"LLM error: {exc}",
                details={"error": str(exc)},
                sources_used=sources_used,
                check_type="llm_assisted",
            )

        out_guard = self._guardrails.check_output(raw_response, pipeline_ctx)
        response_text = out_guard.sanitized_text or raw_response

        return self._parse_llm_condition_response(condition, response_text, sources_used)

    def _parse_llm_condition_response(
        self,
        condition: ConditionDefinition,
        response_text: str,
        sources_used: list[str],
    ) -> ConditionCheckResult:
        """Parse LLM JSON response for a condition check."""
        data, parse_error = try_parse_json(response_text)
        if data is None:
            return ConditionCheckResult(
                condition_name=condition.name,
                status="needs_review",
                severity=condition.severity,
                analyst_message=f"Could not parse LLM response: {parse_error}",
                details={"raw": response_text[:300]},
                sources_used=sources_used,
                check_type="llm_assisted",
            )

        status: str = data.get("status", "needs_review")
        if status not in ("pass", "fail", "needs_review"):
            status = "needs_review"

        return ConditionCheckResult(
            condition_name=condition.name,
            status=status,
            severity=condition.severity if status != "pass" else "info",
            analyst_message=data.get("analyst_message", condition.analyst_message),
            details=data.get("details", {}),
            sources_used=sources_used,
            check_type="llm_assisted",
        )


def _parse_date(value: Any) -> date:
    """Parse a date from various string formats."""
    if isinstance(value, date):
        return value
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%Y", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")


def _months_ago(ref: date, months: int) -> date:
    """Return the calendar date that is exactly `months` calendar months before `ref`.

    Uses calendar-accurate month arithmetic (handles month-end clamping),
    e.g. 3 months before 2024-03-31 → 2024-12-31, not 2024-12-01.
    """
    import calendar
    year = ref.year
    month = ref.month - months
    while month <= 0:
        month += 12
        year -= 1
    # Clamp day to the last valid day of the target month
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(ref.day, max_day))
