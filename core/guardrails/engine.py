from __future__ import annotations
import hashlib
import logging
import yaml

from core.guardrails.verdict import GuardrailVerdict
from core.guardrails.input_guards import InputGuardChain
from core.guardrails.output_guards import OutputGuardChain
from core.guardrails.pii import PIIScanner
from core.guardrails.injection import InjectionDetector
from core.context import PipelineContext

logger = logging.getLogger(__name__)


class GuardrailEngine:
    """
    Central guardrail orchestrator. Runs before and after every LLM call.
    Config-driven: all rules loaded from YAML, zero hardcoded policies.
    """

    def __init__(
        self,
        global_config_path: str = "configs/guardrails/global.yaml",
        pii_config_path: str = "configs/guardrails/pii_patterns.yaml",
    ):
        with open(global_config_path) as f:
            self._config = yaml.safe_load(f)
        with open(pii_config_path) as f:
            self._pii_config = yaml.safe_load(f)

        self._input_chain = InputGuardChain(self._config["input_guards"])
        self._output_chain = OutputGuardChain(self._config["output_guards"])
        self._pii_scanner = PIIScanner(self._pii_config)
        self._injection_detector = InjectionDetector(
            self._config["input_guards"]["blocked_content_patterns"]
        )

    def check_input(
        self, prompt: str, context: PipelineContext
    ) -> GuardrailVerdict:
        """Run all input guards before sending prompt to LLM."""
        flags: list[str] = []
        sanitized = prompt

        # 1. Document hash integrity check
        if self._config["input_guards"].get("require_document_hash_match"):
            current_hash = hashlib.sha256(
                context.document_text.encode()
            ).hexdigest()
            if (
                context.original_input_hash
                and current_hash != context.original_input_hash
            ):
                return GuardrailVerdict(
                    passed=False,
                    blocked_reason="Document text was tampered with after ingestion",
                    flags=["TAMPERING_DETECTED"],
                )

        # 2. Length limits
        length_verdict = self._input_chain.check_length(prompt)
        if not length_verdict.passed:
            return length_verdict

        # 3. Prompt injection detection
        injection_verdict = self._injection_detector.scan(prompt)
        if not injection_verdict.passed:
            logger.warning(
                f"[{context.run_id}] Injection attempt blocked: "
                f"{injection_verdict.blocked_reason}"
            )
            return injection_verdict
        flags.extend(injection_verdict.flags)

        # 4. PII scanning on input (if enabled)
        if self._config.get("pii", {}).get("scan_input"):
            pii_verdict = self._pii_scanner.scan_and_redact(
                sanitized, country=context.country
            )
            sanitized = pii_verdict.sanitized_text or sanitized
            flags.extend(pii_verdict.flags)

        return GuardrailVerdict(
            passed=True, flags=flags, sanitized_text=sanitized
        )

    def check_output(
        self,
        raw_response: str,
        context: PipelineContext,
        expected_fields: list[str] | None = None,
    ) -> GuardrailVerdict:
        """Run all output guards after receiving LLM response."""
        flags: list[str] = []
        sanitized = raw_response

        # 1. Response length
        length_verdict = self._output_chain.check_length(raw_response)
        if not length_verdict.passed:
            return length_verdict

        # 2. Must be valid JSON
        json_verdict = self._output_chain.check_json(raw_response)
        if not json_verdict.passed:
            return json_verdict
        if json_verdict.sanitized_text:
            sanitized = json_verdict.sanitized_text

        # 3. Blocked content patterns
        content_verdict = self._output_chain.check_blocked_content(raw_response)
        flags.extend(content_verdict.flags)
        if not content_verdict.passed:
            return content_verdict

        # 4. Hallucination signal: references fields not in schema
        if expected_fields:
            hallucination_verdict = self._output_chain.check_field_hallucination(
                raw_response, expected_fields
            )
            flags.extend(hallucination_verdict.flags)

        # 5. PII scanning on output
        if self._config.get("pii", {}).get("scan_output"):
            pii_verdict = self._pii_scanner.scan_and_redact(
                sanitized, country=context.country
            )
            sanitized = pii_verdict.sanitized_text or sanitized
            flags.extend(pii_verdict.flags)
            if pii_verdict.flags and self._config["pii"].get("action") == "block":
                return GuardrailVerdict(
                    passed=False,
                    blocked_reason="PII detected in LLM output",
                    flags=pii_verdict.flags,
                )

        return GuardrailVerdict(
            passed=True, flags=flags, sanitized_text=sanitized
        )
