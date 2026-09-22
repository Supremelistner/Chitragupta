# Document Management Service — Architecture

## Overview

The Document Management Service is an independent microservice responsible for
the complete document lifecycle: upload, storage, vision/OCR extraction,
classification, validation, metadata, chunking, indexing, semantic retrieval,
evidence retrieval, and controlled content access.

Other services must not directly access the underlying document database, vector
database, OCR/vision provider, or file storage. The future agent consumes only
document-level MCP tools; Qdrant, PostgreSQL, OCR/vision, and storage remain
replaceable infrastructure adapters owned by this service.

---

## Data Ownership

Three storage backends serve distinct purposes. No backend should be used outside
its designated responsibility.

### PostgreSQL — Canonical Metadata & State

**Owns:** All structured document information that is the source of truth.

- Document identity and lifecycle timestamps
- Version records (file metadata, processing state, OCR results, indexing status)
- Privacy classification
- Extracted text and excerpts
- Audit events for sensitive access
- Document relationship graph (contextual links between documents)

**Does NOT own:** Vectors, embeddings, original file binaries.

**Schema:** Defined in `src/document_mgmt_service/schemas/__init__.py` as versioned
migrations (currently 3 migrations).

### Qdrant — Vectors + Retrieval Payload

**Owns:** Semantic search vectors and the payload fields needed for filtered
retrieval and provenance.

- Embedding vectors (one per chunk)
- Payload: chunk_id, document_id, version, page_number, text, privacy, etc.

**Does NOT own:** Canonical document state, authorization decisions, file binaries.

**Payload schema:** The canonical payload contract is
`QdrantSemanticChunkStoreAdapter._build_payload` in
`src/document_mgmt_service/infrastructure/qdrant.py`; collection config +
plaintext-filterable fields are documented in `QDRANT_COLLECTION_CONFIG`
in the same module.

**Point ID strategy:** Uses `chunk_id` (deterministic string derived from
`document_id:version:chunk_index`). This enables bidirectional tracing:
- PostgreSQL chunk → Qdrant point: join on `chunk_id`
- Qdrant point → PostgreSQL record: use `document_id` + `version` from payload

### Filesystem (or Object Storage) — Document Binaries Only

**Owns:** The actual file bytes (PDF, images, scanned documents).

**Does NOT own:** Any metadata, vectors, or authorization state.

**Interface:** `FileStorage` protocol in `domain/ports.py`.
**Implementation:** `LocalFileStorageAdapter` in `infrastructure/storage.py`.

**V1:** Stores files under a configurable local `data/` directory.
**Future:** Replace with S3/MinIO adapter — no domain logic changes needed.

**Configuration:**
- `DOCUMENT_SERVICE_FILE_STORAGE_ROOT` env var (default: `./data/files`)

---

## Bidirectional Traceability

Records across all three backends can be traced using shared stable IDs:

```
PostgreSQL (document_versions)
    ├── document_id + version  →  canonical state
    ├── storage_key            →  filesystem binary
    └── chunk records          →  identified by chunk_id

Qdrant (points)
    ├── point_id = chunk_id    →  directly maps to PostgreSQL chunk
    ├── document_id + version  →  FK to PostgreSQL document_versions
    └── page_number, char offsets  →  precise source location

Filesystem
    └── storage_key            →  stored in PostgreSQL document_versions.storage_key
```

---

## Schema Versioning

PostgreSQL migrations are defined as ordered `Migration` dataclasses in
`schemas/__init__.py`. A `schema_migrations` tracking table records which
migrations have been applied, enabling safe incremental upgrades.

Current migrations:
1. **001** — Core document tables (`documents`, `document_versions`)
2. **002** — Audit events (`audit_events`)
3. **003** — Document relationships (`document_relationships`)

---

## Access Control

The service enforces a three-level progressive disclosure model:

- **Level 1 — Description:** Safe, non-identifying description always available.
- **Level 2 — Specific information:** Policy-evaluated per request (ALLOW / DENY / REQUIRE_APPROVAL / REDACT).
- **Level 3 — Whole document:** Retrieved from original storage only when policy permits.

The agent may request information but cannot grant itself permission.
Sensitive operations are authorized, policy-checked, optionally user-approved,
and auditable.

---

## Ingestion Pipeline

```
Upload → Store binary → Vision/OCR extraction → Template/reference discovery
    → Validate document structure/content → Generate safe metadata
    → Classify privacy → Chunk text → Embed chunks → Upsert to Qdrant
    → Update PostgreSQL state
```

Each stage updates the `processing_status` field in PostgreSQL, providing
full observability into ingestion progress.

### Template & Reference Knowledge

The document manager owns validation orchestration. A vision/model adapter can
classify the uploaded document and extract text/fields. A template discovery
adapter can use curated templates and web-search-backed references to identify
which document templates the upload may match. Validation results are stored as
metadata/provenance; the upload response itself remains a terse success/failure
status.

### MCP Tool Boundary

The MCP server exposes document-level operations only:

- `upload_document`
- `get_document_metadata`
- `get_document_description`
- `list_documents`
- `search_documents`
- `search_document_content`
- `get_evidence`
- `get_page`
- `get_document`
- `request_sensitive_access`
- `delete_document`

It does not expose raw Qdrant, PostgreSQL, OCR, or storage operations.

`delete_document` cascades across all three stores in one call — Postgres
rows (versions plus the parent row, with `latest_version` recomputed when
only some versions are removed), the file-storage blobs for exactly the
deleted versions, and the Qdrant vectors. It is scoped by `user_id`, so an
account can only delete its own documents, and it is *not* offered to the
chat LLM — deletion is a UI-initiated action only.

---

## Key Design Principles

1. **PostgreSQL is the source of truth** for all non-vector metadata.
2. **Qdrant is a search index** — not a document store or auth authority.
3. **Filesystem is for binaries only** — behind an interface that can swap to S3.
4. **Stable IDs everywhere** — UUIDs for documents, deterministic strings for chunks.
5. **No backend accessed directly** — all access goes through typed port interfaces.
6. **Security is architectural** — the agent cannot bypass the service boundary.
