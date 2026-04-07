# Analyst Document Review Tool

An LLM-assisted document review tool for credit analysts. The system helps analysts review financial documents submitted for credit applications — it does **not** make credit decisions. Every output is a recommendation for analyst review.

## Overview

The tool processes multi-document cases through a configurable pipeline:

1. **Classification** — LLM classifies each uploaded document against country-registered types
2. **Completeness Check** — Validates that all required document categories are present
3. **Extraction** — Per-document data extraction using YAML-defined schemas
4. **Condition Checks** — Deterministic and LLM-assisted analyst condition checks
5. **Report Generation** — Assembles a full analyst review report with narrative summary

## Architecture

```
api/
  main.py               FastAPI endpoints (create case, upload docs, process, report)

core/
  case/
    case_context.py     CaseContext, DocumentEntry, CaseStatus dataclasses
    case_orchestrator.py  5-phase pipeline orchestrator
    dependency_graph.py   Document dependency graph & completeness checker

  classification/
    classifier.py       LLM-based document classifier (confidence-gated)

  conditions/
    checker.py          Deterministic + LLM-assisted condition checks

  review/
    report_builder.py   Analyst review report assembler

  pipeline_engine.py    Multi-step extraction engine with retry & reflection
  prompt_compiler.py    Jinja2-based prompt compiler (zero static prompts)
  inference.py          InferenceRouter (vLLM, Google GenAI backends)
  guardrails/           Input/output guardrail chain (PII, injection, length)
  context.py            PipelineContext & StepTrace audit trail

configs/
  countries/
    germany/
      classification.yaml         Registered document types for Germany
      doc_types/                  Extraction schemas per document type
      conditions/                 Condition check YAML definitions
      process/consumer_credit.yaml  Process configuration

  guardrails/
    global.yaml         Input/output guardrail rules
    pii_patterns.yaml   PII detection patterns (global + country-specific)

  pipelines/
    single_doc_extraction.yaml  Default extraction pipeline

core/templates/         Jinja2 prompt templates
```

## Key Design Principles

- **Analyst tool, not a decision engine** — all outputs are recommendations
- **Zero static prompts** — every prompt is generated from YAML schemas and Jinja2 templates
- **Config-driven** — countries, document types, conditions, and pipelines are all YAML-configured
- **Guardrails on every LLM call** — input validation, output validation, PII scanning, injection detection
- **Retry with reflection** — failed extraction steps self-heal via LLM reflection prompts
- **Full audit trail** — every step attempt is logged in `PipelineContext.traces`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/cases` | Create a new review case |
| POST | `/cases/{id}/documents` | Upload a document to a case |
| POST | `/cases/{id}/process` | Run the full review pipeline |
| GET  | `/cases/{id}/report` | Get the analyst review report |
| POST | `/cases/{id}/documents/{doc_id}/reclassify` | Analyst manual classification override |

## Configuration

### Adding a New Country

1. Create `configs/countries/{country}/classification.yaml` with document types
2. Add extraction schemas in `configs/countries/{country}/doc_types/{type}.yaml`
3. Add condition definitions in `configs/countries/{country}/conditions/*.yaml`
4. Create a process config in `configs/countries/{country}/process/{process}.yaml`

### Adding a New Condition Check

Add a condition definition to a country conditions YAML:

```yaml
conditions:
  - name: my_condition
    description: "What this checks"
    applies_to: ["salary_slip"]       # doc types; empty = all
    check_type: deterministic          # or llm_assisted
    logic:
      field: net_salary
      operator: greater_than
      value: 0
    severity: error
    on_fail: fail
    analyst_message: "Net salary must be positive"
```

## Requirements

- Python 3.11+
- See `pyproject.toml` for dependencies

```bash
pip install -e ".[dev]"
uvicorn api.main:app --reload
```
