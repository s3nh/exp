from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

from core.context import PipelineContext, StepTrace, StepStatus
from core.guardrails.engine import GuardrailEngine
from core.inference import InferenceRouter
from core.prompt_compiler import PromptCompiler
from core.utils import try_parse_json

logger = logging.getLogger(__name__)


@dataclass
class StepConfig:
    name: str
    prompt_template: str
    retry_on_failure: bool = True
    max_retries: int = 3
    use_reflection: bool = True
    output_schema: list[str] | None = None
    temperature: float = 0.0
    inject_previous_results: bool = True


@dataclass
class PipelineDefinition:
    name: str
    steps: list[StepConfig]
    backend_key: str = "default"


class RetryController:
    """
    Retry with reflection and self-healing.
    On failure, generates a reflection prompt that includes the error
    and asks the LLM to self-correct.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self._max_retries = max_retries

    async def execute_with_retry(
        self,
        step: StepConfig,
        prompt: str,
        router: InferenceRouter,
        guardrails: GuardrailEngine,
        context: PipelineContext,
        compiler: PromptCompiler,
    ) -> dict[str, Any]:
        """Execute a step with retry and self-healing reflection."""
        last_error: str = ""
        last_raw: str = ""

        for attempt in range(1, step.max_retries + 1):
            start = time.monotonic()

            current_prompt = prompt
            if attempt > 1 and step.use_reflection and last_error:
                current_prompt = compiler.compile_reflection_prompt(
                    failed_step_name=step.name,
                    original_prompt=prompt,
                    raw_response=last_raw,
                    errors=[last_error],
                    context=context.all_results_flat,
                )

            guard_in = guardrails.check_input(current_prompt, context)
            if not guard_in.passed:
                duration = (time.monotonic() - start) * 1000
                trace = StepTrace(
                    step_name=step.name,
                    attempt=attempt,
                    prompt_sent=current_prompt,
                    raw_response="",
                    parsed_output=None,
                    validation_errors=[f"Input blocked: {guard_in.blocked_reason}"],
                    guardrail_flags=guard_in.flags,
                    reflection=None,
                    duration_ms=duration,
                    status=StepStatus.FAILED_TERMINAL,
                )
                context.add_trace(trace)
                raise RuntimeError(
                    f"Step '{step.name}' blocked by input guardrails: {guard_in.blocked_reason}"
                )

            try:
                raw_response = await router.generate(
                    prompt=current_prompt,
                    temperature=step.temperature,
                )
            except Exception as exc:
                last_error = str(exc)
                last_raw = ""
                duration = (time.monotonic() - start) * 1000
                trace = StepTrace(
                    step_name=step.name,
                    attempt=attempt,
                    prompt_sent=current_prompt,
                    raw_response="",
                    parsed_output=None,
                    validation_errors=[str(exc)],
                    guardrail_flags=[],
                    reflection=None,
                    duration_ms=duration,
                    status=(
                        StepStatus.FAILED_RETRYING
                        if attempt < step.max_retries
                        else StepStatus.FAILED_TERMINAL
                    ),
                )
                context.add_trace(trace)
                if attempt >= step.max_retries or not step.retry_on_failure:
                    raise
                logger.warning(
                    "Step '%s' attempt %d/%d failed: %s — retrying",
                    step.name, attempt, step.max_retries, exc,
                )
                continue

            guard_out = guardrails.check_output(raw_response, context)
            response_text = guard_out.sanitized_text or raw_response
            last_raw = response_text

            parsed, parse_error = try_parse_json(response_text)
            duration = (time.monotonic() - start) * 1000

            if parsed is not None:
                schema_errors: list[str] = []
                if step.output_schema:
                    schema_errors = [
                        f"Missing key: {k}"
                        for k in step.output_schema
                        if k not in parsed
                    ]

                if not schema_errors:
                    trace = StepTrace(
                        step_name=step.name,
                        attempt=attempt,
                        prompt_sent=current_prompt,
                        raw_response=response_text,
                        parsed_output=parsed,
                        validation_errors=[],
                        guardrail_flags=guard_out.flags,
                        reflection=None,
                        duration_ms=duration,
                        status=StepStatus.SUCCESS,
                    )
                    context.add_trace(trace)
                    context.set_result(step.name, parsed)
                    return parsed
                else:
                    last_error = f"Schema validation errors: {schema_errors}"
            else:
                last_error = f"JSON parse error: {parse_error}"

            trace = StepTrace(
                step_name=step.name,
                attempt=attempt,
                prompt_sent=current_prompt,
                raw_response=response_text,
                parsed_output=parsed,
                validation_errors=[last_error],
                guardrail_flags=guard_out.flags,
                reflection=None,
                duration_ms=duration,
                status=(
                    StepStatus.FAILED_RETRYING
                    if attempt < step.max_retries
                    else StepStatus.FAILED_TERMINAL
                ),
            )
            context.add_trace(trace)

            if attempt >= step.max_retries or not step.retry_on_failure:
                if parsed is not None:
                    context.set_result(step.name, parsed)
                    return parsed
                raise RuntimeError(
                    f"Step '{step.name}' failed after {attempt} attempts: {last_error}"
                )

            logger.warning(
                "Step '%s' attempt %d/%d failed: %s — retrying",
                step.name, attempt, step.max_retries, last_error,
            )

        raise RuntimeError(f"Step '{step.name}' exhausted all retries")


class PipelineEngine:
    """
    Multi-step pipeline engine.
    Each step's results are available to downstream steps.
    Previous results can mutate downstream prompts.
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

    def load_pipeline(self, path: str) -> PipelineDefinition:
        """Load pipeline definition from YAML."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Pipeline config not found: {path}")

        with open(p) as f:
            raw = yaml.safe_load(f)

        steps = [
            StepConfig(
                name=s["name"],
                prompt_template=s.get("prompt_template", "extraction_base.j2"),
                retry_on_failure=s.get("retry_on_failure", True),
                max_retries=s.get("max_retries", 3),
                use_reflection=s.get("use_reflection", True),
                output_schema=s.get("output_schema"),
                temperature=s.get("temperature", 0.0),
                inject_previous_results=s.get("inject_previous_results", True),
            )
            for s in raw.get("steps", [])
        ]

        return PipelineDefinition(
            name=raw["name"],
            steps=steps,
            backend_key=raw.get("backend_key", "default"),
        )

    async def execute(
        self,
        pipeline: PipelineDefinition,
        context: PipelineContext,
        schema: Any,  # ExtractionSchema
    ) -> PipelineContext:
        """Execute all steps in the pipeline sequentially."""
        retry_ctrl = RetryController(max_retries=3)

        for step in pipeline.steps:
            logger.info(
                "Pipeline '%s' — executing step '%s'", pipeline.name, step.name
            )

            previous_context: dict[str, Any] = {}
            if step.inject_previous_results:
                previous_context = context.all_results_flat

            document_text = context.document_text
            prompt = self._compiler.compile(
                schema=schema,
                document_text=document_text,
                context=previous_context,
            )

            try:
                await retry_ctrl.execute_with_retry(
                    step=step,
                    prompt=prompt,
                    router=self._router,
                    guardrails=self._guardrails,
                    context=context,
                    compiler=self._compiler,
                )
            except RuntimeError as exc:
                logger.error(
                    "Pipeline '%s' step '%s' failed: %s", pipeline.name, step.name, exc
                )
                if step.retry_on_failure:
                    raise
                logger.warning(
                    "Continuing pipeline despite step '%s' failure", step.name
                )

        return context


# Re-export for backward compatibility; new code should import from core.utils
_try_parse_json = try_parse_json
