# Dynamic Extraction Engine

Config-driven, multi-step LLM extraction engine with guardrails, cross-document validation,
and credit decisioning. Built for multi-country scalability with zero prompt copy-paste.

## Architecture

```
configs/     → YAML schemas, pipelines, guardrails, rules (business analysts maintain these)
core/        → Engine code (prompt compiler, inference, guardrails, pipeline, case orchestrator)
extensions/  → Optional agentic steps
api/         → FastAPI endpoints
```

## Principles

- **Prompts are compiled artifacts**, not source code
- **LLM extracts text and writes narratives**; Python makes decisions and does math
- **YAML defines what varies by country**; the engine is universal
- **Agentic is an extension**, not the base

## Quick Start

```bash
pip install -e ".[dev]"
uvicorn api.main:app --reload
```

## Adding a New Country

1. Create `configs/countries/{country}/doc_types/*.yaml`
2. Create `configs/countries/{country}/rules/*.yaml`
3. Create `configs/countries/{country}/processes/*.yaml`
4. Add PII patterns to `configs/guardrails/pii_patterns.yaml`
5. No code changes required.
```
