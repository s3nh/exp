from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import yaml
from jinja2 import Environment, FileSystemLoader


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
            path = Path(f"configs/countries/{country}/doc_types/{doc_type}.yaml")
            if not path.exists():
                # Fallback to defaults
                path = Path(f"configs/countries/_defaults/doc_types/{doc_type}.yaml")
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
