from __future__ import annotations
import asyncio
import logging
from typing import Any

from core.case.case_context import (
    CaseContext, CaseStatus, DocumentEntry, DocumentStatus,
    CrossValidationResult, FinancialAnalysis, CreditDecision,
)
from core.case.dependency_graph import (
    DocumentGraph, CompletenessChecker, CrossValidationRule,
)
from core.prompt_compiler import PromptCompiler
from core.inference import InferenceRouter
from core.guardrails.engine import GuardrailEngine
from core.pipeline_engine import PipelineEngine
from core.context import PipelineContext
from core.cross_doc.entity_resolver import EntityResolver
from core.cross_doc.consistency_checker import ConsistencyChecker
from core.cross_doc.contradiction_detector import ContradictionDetector
from core.financial.ratios import RatioCalculator
from core.financial.income_reconciliation import IncomeReconciler
from core.financial.spreading import FinancialSpreader
from core.financial.decision_engine import DecisionEngine

logger = logging.getLogger(__name__)


class CaseOrchestrator:
    """
    Top-level orchestrator for multi-document credit cases.

    Phases:
    1. INTAKE         — Validate document completeness
    2. EXTRACTION     — Extract data from each document (parallel where possible)
    3. CROSS_VALIDATION — Cross-document consistency checks
    4. ANALYSIS       — Financial spreading, ratio calculations
    5. DECISION       — Credit decision (approve / reject / human review)
    """

    def __init__(
        self,
        pipeline_engine: PipelineEngine,
        compiler: PromptCompiler,
        router: InferenceRouter,
        guardrails: GuardrailEngine,
    ):
        self._pipeline_engine = pipeline_engine
        self._compiler = compiler
        self._router = router
        self._guardrails = guardrails

        # Sub-engines
        self._entity_resolver = EntityResolver(router, guardrails)
        self._consistency_checker = ConsistencyChecker(router, guardrails)
        self._contradiction_detector = ContradictionDetector(router, guardrails)
        self._ratio_calculator = RatioCalculator()
        self._income_reconciler = IncomeReconciler(router, guardrails)
        self._spreader = FinancialSpreader(router, guardrails)
        self._decision_engine = DecisionEngine(router, guardrails)

    async def process_case(
        self,
        case: CaseContext,
        graph: DocumentGraph,
    ) -> CaseContext:
        """Execute the full credit case pipeline."""

        # ═══════════════════════════════════════════════════════
        # PHASE 1: INTAKE — Completeness check
        # ═══════════════════════════════════════════════════════
        case.status = CaseStatus.INTAKE
        logger.info(f"[{case.case_id}] Phase: INTAKE")

        # First extract the anchor document (application) to evaluate conditions
        anchor_data = await self._extract_anchor(case, graph)

        checker = CompletenessChecker(graph)
        is_complete, issues = checker.check(case, anchor_data)

        if not is_complete:
            logger.warning(
                f"[{case.case_id}] Incomplete case: {issues}"
            )
            case.phase_traces["intake"] = [
                {"completeness_check": "failed", "issues": issues}
            ]
            # Don't hard-fail — proceed with what we have but flag it
            case.financial_analysis.flags.append(
                f"INCOMPLETE_DOCUMENTATION: {'; '.join(issues)}"
            )

        # ═══════════════════════════════════════════════════════
        # PHASE 2: EXTRACTION — Per-document extraction
        # ═══════════════════════════════════════════════════════
        case.status = CaseStatus.EXTRACTION
        logger.info(f"[{case.case_id}] Phase: EXTRACTION")

        extraction_phase = next(
            p for p in graph.processing_phases if p.name == "extraction"
        )

        for group in (extraction_phase.groups or []):
            # Each group can be processed in parallel
            tasks = []
            for category in group:
                docs = case.get_documents_by_category(category)
                for doc in docs:
                    if doc.status == DocumentStatus.EXTRACTED:
                        continue  # Already extracted (e.g. anchor)
                    tasks.append(self._extract_document(case, doc))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(
                            f"[{case.case_id}] Document extraction failed: {result}"
                        )

        # ═══════════════════════════════════════════════════════
        # PHASE 3: CROSS-DOCUMENT VALIDATION
        # ═══════════════════════════════════════════════════════
        case.status = CaseStatus.CROSS_VALIDATION
        logger.info(f"[{case.case_id}] Phase: CROSS_VALIDATION")

        cv_phase = next(
            p for p in graph.processing_phases if p.name == "cross_document_validation"
        )

        for rule_name in (cv_phase.sequential or []):
            rule = next(
                (r for r in graph.cross_validation_rules if r.name == rule_name),
                None,
            )
            if rule is None:
                continue

            result = await self._run_cross_validation(case, rule)
            case.cross_validations.append(result)

            # If a blocking rule fails, stop processing
            if not result.passed and result.action == "block":
                logger.error(
                    f"[{case.case_id}] Blocking cross-validation failed: "
                    f"{rule_name} — {result.details}"
                )
                case.status = CaseStatus.REJECTED
                case.credit_decision = CreditDecision(
                    decision="reject",
                    confidence=1.0,
                    rationale=f"Blocked by cross-validation rule: {rule_name}. "
                              f"{result.details}",
                    risk_grade="E",
                    flags=[f"BLOCKED_{rule_name.upper()}"],
                )
                return case

        # ═══════════════════════════════════════════════════════
        # PHASE 4: FINANCIAL ANALYSIS
        # ═══════════════════════════════════════════════════════
        case.status = CaseStatus.ANALYSIS
        logger.info(f"[{case.case_id}] Phase: ANALYSIS")

        analysis_phase = next(
            p for p in graph.processing_phases if p.name == "financial_analysis"
        )

        for step_name in (analysis_phase.sequential or []):
            await self._run_financial_step(case, step_name, graph)

        # ═══════════════════════════════════════════════════════
        # PHASE 5: CREDIT DECISION
        # ═══════════════════════════════════════════════════════
        if case.needs_human_review:
            case.status = CaseStatus.HUMAN_REVIEW
            case.credit_decision = CreditDecision(
                decision="refer_to_human",
                rationale="Cross-validation flags require human review",
                human_review_reasons=[
                    f"{cv.rule_name}: {cv.details}"
                    for cv in case.cross_validations
                    if not cv.passed
                ],
            )
        else:
            case.status = CaseStatus.DECISION
            case.credit_decision = await self._make_credit_decision(case, graph)

        case.status = CaseStatus.COMPLETED
        return case

    # ─────────────────────────────────────────────────────────
    # PHASE IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────

    async def _extract_anchor(
        self, case: CaseContext, graph: DocumentGraph
    ) -> dict[str, Any] | None:
        """Extract the anchor document (application form) first."""
        anchor_req = next(
            (r for r in graph.requirements.values() if r.role == "anchor"), None
        )
        if anchor_req is None:
            return None

        anchor_docs = case.get_documents_by_category(anchor_req.category)
        if not anchor_docs:
            return None

        anchor = anchor_docs[0]
        await self._extract_document(case, anchor)
        return anchor.extracted_data

    async def _extract_document(
        self, case: CaseContext, doc: DocumentEntry
    ) -> None:
        """Run the single-document extraction pipeline for one document."""
        logger.info(
            f"[{case.case_id}] Extracting {doc.doc_type} ({doc.doc_id})"
        )

        schema = self._compiler.load_schema(case.country, doc.doc_type)
        pipeline_def = self._pipeline_engine.load_pipeline(
            "configs/pipelines/single_doc_extraction.yaml"
        )

        doc.pipeline_context = PipelineContext(
            country=case.country,
            business_unit=schema.business_unit,
            document_type=doc.doc_type,
            document_text=doc.text_content,
        )

        try:
            result_ctx = await self._pipeline_engine.execute(
                pipeline_def, doc.pipeline_context, schema
            )
            # Merge all step results into final extraction
            merged: dict[str, Any] = {}
            for step_result in result_ctx.results.values():
                merged.update(step_result)
            doc.extracted_data = merged
            doc.status = DocumentStatus.EXTRACTED
        except Exception as e:
            logger.error(
                f"[{case.case_id}] Extraction failed for {doc.doc_type} "
                f"({doc.doc_id}): {e}"
            )
            doc.status = DocumentStatus.FAILED

    async def _run_cross_validation(
        self, case: CaseContext, rule: CrossValidationRule
    ) -> CrossValidationResult:
        """Execute a cross-document validation rule using LLM + deterministic logic."""
        logger.info(
            f"[{case.case_id}] Cross-validation: {rule.name}"
        )

        # Gather relevant data from all source categories
        source_data: dict[str, Any] = {}
        for source_cat in rule.sources:
            docs = case.get_documents_by_category(source_cat)
            for doc in docs:
                if doc.extracted_data:
                    key = f"{source_cat}.{doc.doc_type}"
                    if key in source_data:
                        # Multiple docs of same type — aggregate
                        if not isinstance(source_data[key], list):
                            source_data[key] = [source_data[key]]
                        source_data[key].append(doc.extracted_data)
                    else:
                        source_data[key] = doc.extracted_data

        # Deterministic check first (if field_mapping with tolerance)
        if rule.field_mapping and rule.tolerance_percent is not None:
            return self._deterministic_cross_check(rule, source_data)

        # Fuzzy / logic-based check — use LLM
        if rule.logic or rule.match_type == "fuzzy":
            return await self._llm_cross_check(case, rule, source_data)

        # Simple exact field match
        return self._exact_field_check(rule, source_data)

    def _deterministic_cross_check(
        self,
        rule: CrossValidationRule,
        source_data: dict[str, Any],
    ) -> CrossValidationResult:
        """
        Numeric cross-check with tolerance.
        E.g., declared income vs salary slip vs tax return vs bank deposits.
        """
        values: dict[str, float | None] = {}
        comparisons: list[dict[str, Any]] = []

        for field_ref, description in rule.field_mapping.items():
            # Parse "category.doc_type.field_name" reference
            parts = field_ref.split(".")
            value = self._resolve_field(parts, source_data)
            values[field_ref] = value
            comparisons.append({
                "field": field_ref,
                "description": description,
                "value": value,
            })

        # Find numeric values and check tolerance
        numeric_vals = [v for v in values.values() if isinstance(v, (int, float))]

        if len(numeric_vals) < 2:
            return CrossValidationResult(
                rule_name=rule.name,
                passed=True,
                severity="info",
                details="Insufficient numeric values to compare",
                field_comparisons=comparisons,
            )

        avg = sum(numeric_vals) / len(numeric_vals)
        tolerance = rule.tolerance_percent / 100.0
        outliers = [
            v for v in numeric_vals if abs(v - avg) / max(avg, 1) > tolerance
        ]

        passed = len(outliers) == 0
        return CrossValidationResult(
            rule_name=rule.name,
            passed=passed,
            severity="warning" if not passed else "info",
            details=(
                f"Values within {rule.tolerance_percent}% tolerance"
                if passed
                else f"Variance exceeds {rule.tolerance_percent}%: {comparisons}"
            ),
            sources_used=rule.sources,
            field_comparisons=comparisons,
            action=rule.on_fail if not passed else "none",
        )

    async def _llm_cross_check(
        self,
        case: CaseContext,
        rule: CrossValidationRule,
        source_data: dict[str, Any],
    ) -> CrossValidationResult:
        """Use LLM for fuzzy matching or complex logic-based cross-validation."""
        import json
        from jinja2 import Environment, FileSystemLoader

        jinja = Environment(
            loader=FileSystemLoader("core/templates"),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = jinja.get_template("cross_document_validation.j2")
        prompt = template.render(
            rule=rule,
            source_data=source_data,
            country=case.country,
        )

        # Run through guardrails + inference
        input_verdict = self._guardrails.check_input(
            prompt,
            PipelineContext(country=case.country),
        )
        if not input_verdict.passed:
            return CrossValidationResult(
                rule_name=rule.name,
                passed=False,
                severity="error",
                details=f"Guardrail blocked: {input_verdict.blocked_reason}",
                action="escalate",
            )

        safe_prompt = input_verdict.sanitized_text or prompt
        raw = await self._router.run_raw(safe_prompt, temperature=0.0)

        try:
            result = json.loads(raw)
            return CrossValidationResult(
                rule_name=rule.name,
                passed=result.get("is_consistent", False),
                severity=result.get("severity", "warning"),
                details=result.get("explanation", ""),
                sources_used=rule.sources,
                field_comparisons=result.get("comparisons", []),
                action=rule.on_fail if not result.get("is_consistent", False) else "none",
            )
        except json.JSONDecodeError:
            return CrossValidationResult(
                rule_name=rule.name,
                passed=False,
                severity="error",
                details=f"LLM returned invalid JSON for cross-validation",
                action="escalate",
            )

    def _exact_field_check(
        self, rule: CrossValidationRule, source_data: dict[str, Any]
    ) -> CrossValidationResult:
        """Exact string match across documents."""
        comparisons: list[dict[str, Any]] = []
        all_match = True

        for field_name in rule.fields:
            values_seen: dict[str, Any] = {}
            for source_key, data in source_data.items():
                if isinstance(data, dict) and field_name in data:
                    values_seen[source_key] = data[field_name]

            unique_values = set(str(v).strip().lower() for v in values_seen.values())
            field_matches = len(unique_values) <= 1
            all_match = all_match and field_matches

            comparisons.append({
                "field": field_name,
                "values": values_seen,
                "matches": field_matches,
            })

        return CrossValidationResult(
            rule_name=rule.name,
            passed=all_match,
            severity="error" if not all_match else "info",
            details="All fields match" if all_match else f"Mismatch detected",
            sources_used=rule.sources,
            field_comparisons=comparisons,
            action=rule.on_fail if not all_match else "none",
        )

    @staticmethod
    def _resolve_field(
        parts: list[str], source_data: dict[str, Any]
    ) -> Any:
        """Resolve a dotted field reference against source data."""
        # Try progressively longer key prefixes
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            remainder = parts[i:]
            if prefix in source_data:
                obj = source_data[prefix]
                for key in remainder:
                    if isinstance(obj, dict):
                        obj = obj.get(key)
                    elif isinstance(obj, list):
                        # Average across multiple docs of same type
                        vals = [
                            item.get(key)
                            for item in obj
                            if isinstance(item, dict) and key in item
                        ]
                        numeric = [v for v in vals if isinstance(v, (int, float))]
                        return sum(numeric) / len(numeric) if numeric else None
                    else:
                        return None
                return obj
        return None

    async def _run_financial_step(
        self,
        case: CaseContext,
        step_name: str,
        graph: DocumentGraph,
    ) -> None:
        """Execute a financial analysis step."""
        logger.info(f"[{case.case_id}] Financial step: {step_name}")

        if step_name == "income_reconciliation":
            reconciled = await self._income_reconciler.reconcile(case)
            case.financial_analysis.verified_monthly_income = reconciled.verified_amount
            case.financial_analysis.income_sources = reconciled.sources
            case.financial_analysis.income_confidence = reconciled.confidence
            case.financial_analysis.flags.extend(reconciled.flags)

        elif step_name == "financial_spreading":
            # Only if business financial docs are present
            bs_docs = case.get_documents_by_type("balance_sheet")
            pnl_docs = case.get_documents_by_type("pnl_statement")
            if bs_docs or pnl_docs:
                spreading = await self._spreader.spread(case, bs_docs, pnl_docs)
                case.financial_analysis.financial_spreading = spreading

        elif step_name == "ratio_calculation":
            self._calculate_ratios(case, graph)

        elif step_name == "credit_decision":
            pass  # Handled in phase 5

    def _calculate_ratios(
        self, case: CaseContext, graph: DocumentGraph
    ) -> None:
        """
        Deterministic ratio calculations.
        LLM extracts → Python calculates. Never let LLM do math.
        """
        fa = case.financial_analysis

        # DTI = Total Monthly Debt / Verified Monthly Income
        if fa.verified_monthly_income and fa.total_monthly_debt_service:
            fa.dti_ratio = round(
                fa.total_monthly_debt_service / fa.verified_monthly_income, 4
            )

        # DSCR = Net Operating Income / Total Debt Service
        spreading = fa.financial_spreading
        if spreading:
            noi = spreading.get("normalized_noi")
            annual_debt = (fa.total_monthly_debt_service or 0) * 12
            if noi and annual_debt > 0:
                fa.dscr = round(noi / annual_debt, 4)

        # LTV = Loan Amount / Property Value
        app_data = {}
        for doc in case.get_documents_by_type("loan_application"):
            app_data = doc.extracted_data
            break
        collateral_docs = case.get_documents_by_type("property_appraisal")
        if app_data.get("loan_amount") and collateral_docs:
            property_value = collateral_docs[0].extracted_data.get("appraised_value")
            if property_value:
                fa.ltv_ratio = round(
                    app_data["loan_amount"] / property_value, 4
                )

        # Load country-specific thresholds from rules
        self._evaluate_ratio_rules(case, graph)

    def _evaluate_ratio_rules(
        self, case: CaseContext, graph: DocumentGraph
    ) -> None:
        """Apply country-specific ratio thresholds."""
        import yaml

        rules_path = f"configs/countries/{case.country}/rules/dti_rules.yaml"
        try:
            with open(rules_path) as f:
                rules = yaml.safe_load(f)
        except FileNotFoundError:
            return

        fa = case.financial_analysis
        for rule in rules.get("thresholds", []):
            ratio_name = rule["ratio"]
            max_value = rule["max_value"]
            actual = getattr(fa, ratio_name, None)

            if actual is not None and actual > max_value:
                fa.flags.append(
                    f"{ratio_name.upper()}_EXCEEDED: "
                    f"{actual:.2%} > {max_value:.2%} (threshold)"
                )

    async def _make_credit_decision(
        self, case: CaseContext, graph: DocumentGraph
    ) -> CreditDecision:
        """
        Final credit decision.
        Deterministic rules first, LLM for nuanced rationale.
        """
        return await self._decision_engine.decide(case, graph)
