from __future__ import annotations
import re
from typing import Any
from core.prompt_compiler import ExtractionSchema


def validate_output(output: dict[str, Any], schema: ExtractionSchema) -> list[str]:
    """Validate extracted output against schema field definitions."""
    errors: list[str] = []
    for fld in schema.fields:
        value = output.get(fld.name)

        if fld.required and value is None:
            errors.append(f"Missing required field: {fld.name}")
            continue

        if value is None:
            continue

        if fld.type == "float" and not isinstance(value, (int, float)):
            errors.append(f"{fld.name}: expected number, got {type(value).__name__}")

        if fld.type == "boolean" and not isinstance(value, bool):
            errors.append(f"{fld.name}: expected boolean, got {type(value).__name__}")

        if fld.type == "string" and not isinstance(value, str):
            errors.append(f"{fld.name}: expected string, got {type(value).__name__}")

        if fld.pattern and isinstance(value, str):
            if not re.match(fld.pattern, value):
                errors.append(
                    f"{fld.name}: '{value}' doesn't match pattern {fld.pattern}"
                )
    return errors


def evaluate_rules(
    output: dict[str, Any], rules: list[dict[str, Any]]
) -> list[str]:
    """Evaluate business rules against extracted output. Returns triggered rule actions."""
    triggered: list[str] = []
    for rule in rules:
        condition = rule.get("condition", "")
        action = rule.get("action", "unknown")
        try:
            # Build safe eval context from output
            safe_ctx = {k: v for k, v in output.items() if not k.startswith("_")}
            if eval(condition, {"__builtins__": {}}, safe_ctx):
                triggered.append(f"{action} (severity: {rule.get('severity', 'info')})")
        except Exception:
            pass  # Rule couldn't be evaluated -- skip
    return triggered
