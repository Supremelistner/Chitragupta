# Chitragupta — Shared Data Contracts & Service Architecture

## Overview

Chitragupta consists of four microservices plus a shared contracts package:

| Service | Port | Responsibility |
|---------|------|----------------|
| **Document Management** | 8080 | Document lifecycle, storage, metadata, ingestion, search |
| **Model Service** | 8081 | Model inference (OCR, extraction, classification) |
| **Validator Service** | 8083 | Document validation, template matching, authenticity checks |
| **Web Search Service** | 8082 | Web search, scraping, downloading |
| **Shared Contracts** | — | Canonical IDs, status enums, relationship models |

## Data Ownership

```
┌─────────────────────────────────────────────────────────────┐
│                    POSTGRESQL (Source of Truth)              │
│                                                             │
│  documents          ← Document identity + lifecycle          │
│  document_versions  ← Per-version metadata + state          │
│  document_chunks    ← Semantic chunk records                 │
│  audit_events       ← Access control audit trail             │
│  document_relationships ← Provenance links between docs     │
│  validation_runs    ← Validator service execution records    │
│  template_versions  ← Versioned document templates           │
│  model_executions   ← Model service execution records        │
│  document_alerts    ← Alerts from validator service          │
│  schema_migrations  ← Migration tracking                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QDRANT (Semantic Retrieval)               │
│                                                             │
│  document_chunks collection                                  │
│    - Point ID = chunk_id (deterministic, bidirectional)     │
│    - Payload references PostgreSQL IDs only                  │
│    - Vectors for semantic similarity search                  │
│    - Payload fields for filtering (privacy, type, etc.)     │
│                                                             │
│  Qdrant is NOT the canonical metadata store.                 │
│  All authoritative state lives in PostgreSQL.                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    FILESYSTEM (Document Binaries)            │
│                                                             │
│  data/files/documents/{document_id}/v{version}/{filename}   │
│                                                             │
│  PostgreSQL stores storage_key (relative path), never the   │
│  binary itself.  Storage adapter behind FileStorage         │
│  protocol — swap for S3/MinIO without changing domain logic.│
└─────────────────────────────────────────────────────────────┘
```

## ID Relationships

```
Document (document_id: str)
  └─ Version (version: int)  ←── PK: (document_id, version)
       ├─ File (storage_key, sha256, file_size)
       ├─ OCR/Extraction (extracted_text, model_extraction)
       ├─ Chunks (chunk_id, chunk_index)  ←── Qdrant point_id = chunk_id
       ├─ Validation Runs (run_id)
       │    ├─ Template (template_id, template_version)
       │    └─ Model Execution (run_id → model_executions)
       ├─ Alerts (id, severity, alert_type)
       └─ Access Audit (intent, decision, redacted)
```

## Cross-Service Communication

### Document → Model Service
- **Purpose**: Text extraction, metadata extraction, classification
- **Contract**: `InferenceRequest` → `InferenceResult`
- **Transport**: HTTP (localhost:8081) or MCP
- **Provenance**: Model execution record stored in PostgreSQL

### Document → Validator Service
- **Purpose**: Structural + temporal validation during ingestion
- **Contract**: `ValidationRequest` → `ValidationResult`
- **Transport**: In-process (same process) or MCP
- **Provenance**: Validation run record stored in PostgreSQL

### Validator → Document Service
- **Purpose**: Alert when document is fake/suspicious/expired
- **Contract**: `POST /alerts` with alert payload
- **Transport**: HTTP (localhost:8080)
- **Provenance**: Alert record stored in PostgreSQL

### Document → Web Search Service
- **Purpose**: Download forms, templates, reference documents
- **Contract**: `DownloadRequest` → `DownloadResult`
- **Transport**: HTTP (localhost:8082) or MCP
- **Provenance**: Downloaded files fed into document ingestion

### Agent/Orchestrator → All Services
- **Purpose**: Orchestrate document processing workflows
- **Contract**: MCP tools on each service
- **Transport**: MCP (stdio or HTTP)

## Schema Ownership Rules

1. **PostgreSQL is the source of truth** for all structured metadata
2. **Qdrant is a search index** — never the authoritative store
3. **Filesystem stores binaries only** — PostgreSQL stores stable file addresses
4. **Stable IDs across boundaries** — no integer auto-increment IDs exposed
5. **Every result has provenance** — which model, template, timestamp produced it
6. **Services don't share databases** — each reads/writes its own tables
7. **Cross-service queries use stable IDs** — document_id, version, chunk_id

## Migration Strategy

- Migrations 001–004: Document Management Service (existing)
- Migration 005: Shared contracts (validation_runs, template_versions, model_executions, document_alerts)
- Future migrations: Add tables as needed, never modify existing schemas destructively

## Template Versioning

Templates are versioned independently of documents:

| Template ID | Version | Document Type | Sub-Type |
|-------------|---------|---------------|----------|
| aadhaar_card_v1 | 1.0.0 | identity_document | aadhaar |
| aadhaar_letter_v1 | 1.0.0 | identity_document | aadhaar |
| eaadhaar_v1 | 1.0.0 | identity_document | aadhaar |
| pan_card_v1 | 1.0.0 | identity_document | pan |
| marksheet_v1 | 1.0.0 | academic_record | marksheet |
| passport_v1 | 1.0.0 | identity_document | passport |

Template versions allow:
- Adding new fields without breaking existing documents
- Deprecating old templates while keeping them for historical validation
- A/B testing template rules

## Provenance Chain

Every piece of data traces back to its source:

```
Document Upload
  → document_versions (created_at, storage_key)
  → model_executions (task=TEXT_EXTRACTION, model_id, latency_ms)
  → validation_runs (template_id, status, risk_score)
  → document_chunks → Qdrant points (chunk_id, document_id, version)
```

## Future MCP Tools (Agent/Orchestrator)

The shared contracts enable these MCP tools without redesigning:

| Tool | Returns | Uses |
|------|---------|------|
| `get_document_context` | Full document metadata + validation status | PostgreSQL |
| `get_document_validation` | Validation history + temporal status | PostgreSQL |
| `search_documents` | Semantic search results | Qdrant → PostgreSQL |
| `get_document_lineage` | Provenance chain (model runs, validation runs) | PostgreSQL |
| `get_template_info` | Template schema + version history | PostgreSQL |
| `check_document_expiry` | Temporal validity + days until expiry | PostgreSQL |
