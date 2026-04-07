from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import yaml
from jinja2 import Environment, FileSystemLoader

from core.utils import sanitize_path_component as _sanitize_path_component


@dataclass
class FieldSpec:
    name: str
    type: str
    required: bool = False
    pattern: str | None = None
    alias: list[str] = field(default_factory=list)
    reasoning: str | None = None
    depends_on: list[str] = field(default_factory=list)
    validation: str | None = None
    currency: str | None = None


@dataclass
class ExtractionSchema:
    country: str
    business_unit: str
    document_type: str
    fields: list[FieldSpec]
    rules: list[dict[str, Any]]
    schema_version: str = "1.0"


class PromptCompiler:
    """
    Compiles extraction schemas into LLM prompts dynamically.
    Zero static prompts -- everything is generated from config.
    """

    def __init__(self, template_dir: str = "core/templates"):
        self._jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._schema_cache: dict[str, ExtractionSchema] = {}

    def load_schema(self, country: str, doc_type: str) -> ExtractionSchema:
        cache_key = f"{country}/{doc_type}"
        if cache_key not in self._schema_cache:
            # Sanitize inputs to prevent path traversal
            safe_country = _sanitize_path_component(country)
            safe_doc_type = _sanitize_path_component(doc_type)
            path = Path(f"configs/countries/{safe_country}/doc_types/{safe_doc_type}.yaml")
            if not path.exists():
                # Fallback to defaults
                path = Path(f"configs/countries/_defaults/doc_types/{safe_doc_type}.yaml")
            with open(path) as f:
                raw = yaml.safe_load(f)
            self._schema_cache[cache_key] = ExtractionSchema(
                country=raw.get("country", country),
                business_unit=raw.get("business_unit", "general"),
                document_type=raw.get("document_type", doc_type),
                fields=[FieldSpec(**fld) for fld in raw.get("fields", [])],
                rules=raw.get("rules", []),
                schema_version=raw.get("schema_version", "1.0"),
            )
        return self._schema_cache[cache_key]

    def compile(
        self,
        schema: ExtractionSchema,
        document_text: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Compile schema + document into a final LLM prompt."""
        extraction_fields = [f for f in schema.fields if not f.reasoning]
        reasoning_fields = [f for f in schema.fields if f.reasoning]

        template = self._jinja_env.get_template("extraction_base.j2")
        return template.render(
            schema=schema,
            extraction_fields=extraction_fields,
            reasoning_fields=reasoning_fields,
            document_text=document_text,
            context=context or {},
            rules=schema.rules,
        )

    def compile_classification_prompt(
        self,
        document_text: str,
        registered_types: list[dict],
        country: str,
    ) -> str:
        """Generate classification prompt from country's registered doc types."""
        template = self._jinja_env.get_template("classification.j2")
        return template.render(
            document_text=document_text,
            registered_types=registered_types,
            country=country,
        )

    def compile_condition_check_prompt(
        self,
        condition: Any,
        relevant_data: dict,
        previous_results: dict,
        country: str,
    ) -> str:
        """Generate condition check prompt from condition definition."""
        template_name = (
            "cross_doc_condition.j2"
            if getattr(condition, "sources", None)
            else "condition_check.j2"
        )
        template = self._jinja_env.get_template(template_name)
        return template.render(
            condition=condition,
            relevant_data=relevant_data,
            previous_results=previous_results,
            country=country,
        )

    def compile_review_summary_prompt(
        self,
        case: Any,
        document_inventory: list[dict],
        extraction_results: dict,
        condition_results: list,
    ) -> str:
        """Generate review summary prompt for analyst narrative."""
        template = self._jinja_env.get_template("review_summary.j2")
        return template.render(
            case=case,
            document_inventory=document_inventory,
            extraction_results=extraction_results,
            condition_results=condition_results,
        )

    def compile_reflection_prompt(
        self,
        failed_step_name: str,
        original_prompt: str,
        raw_response: str,
        errors: list[str],
        context: dict,
    ) -> str:
        """Generate self-healing reflection prompt when a step fails."""
        template = self._jinja_env.get_template("reflection_repair.j2")
        return template.render(
            failed_step_name=failed_step_name,
            original_prompt=original_prompt,
            raw_response=raw_response,
            errors=errors,
            context=context,
        )
