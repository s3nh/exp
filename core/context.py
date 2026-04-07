from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from enum import Enum


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED_RETRYING = "failed_retrying"
    FAILED_TERMINAL = "failed_terminal"
    SKIPPED = "skipped"


@dataclass
class StepTrace:
    """Full audit trail for a single step execution attempt."""
    step_name: str
    attempt: int
    prompt_sent: str
    raw_response: str
    parsed_output: dict[str, Any] | None
    validation_errors: list[str]
    guardrail_flags: list[str]
    reflection: str | None
    duration_ms: float
    status: StepStatus
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class PipelineContext:
    """
    Immutable-append shared state that flows through the entire pipeline.
    Every step can read all previous results. No step can mutate another's output.
    """
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    country: str = ""
    business_unit: str = ""
    document_type: str = ""
    document_text: str = ""
    original_input_hash: str = ""

    # Accumulated results: step_name -> final validated output
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Full trace log for every attempt of every step
    traces: list[StepTrace] = field(default_factory=list)

    # Metadata injected by guardrails
    guardrail_metadata: dict[str, Any] = field(default_factory=dict)

    # Dynamic overrides that steps can set for downstream steps
    step_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_result(self, step_name: str) -> dict[str, Any] | None:
        return self.results.get(step_name)

    def set_result(self, step_name: str, data: dict[str, Any]) -> None:
        self.results[step_name] = data

    def add_trace(self, trace: StepTrace) -> None:
        self.traces.append(trace)

    def set_override(self, from_step: str, target_step: str, overrides: dict) -> None:
        """Allow a step to inject dynamic config into a downstream step."""
        key = f"{from_step}->{target_step}"
        self.step_overrides[key] = overrides

    def get_overrides_for(self, target_step: str) -> dict[str, Any]:
        """Collect all overrides targeting this step from any upstream step."""
        merged: dict[str, Any] = {}
        for key, overrides in self.step_overrides.items():
            if key.endswith(f"->{target_step}"):
                merged.update(overrides)
        return merged

    @property
    def last_failed_trace(self) -> StepTrace | None:
        for trace in reversed(self.traces):
            if trace.status in (StepStatus.FAILED_RETRYING, StepStatus.FAILED_TERMINAL):
                return trace
        return None

    @property
    def all_results_flat(self) -> dict[str, Any]:
        """Flatten all step results into a single dict for downstream prompts."""
        flat: dict[str, Any] = {}
        for step_name, data in self.results.items():
            for k, v in data.items():
                flat[f"{step_name}.{k}"] = v
        return flat
