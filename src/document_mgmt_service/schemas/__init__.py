"""Versioned PostgreSQL schema definitions for the Document Management service.

This module defines the canonical database schema as incremental migrations.
Each migration is a (version, description, sql) tuple that can be applied
sequentially.  The schema catalog is the single source of truth for all
PostgreSQL structures used by this service.

Ownership:
    PostgreSQL = canonical metadata, state, access policies, audit records.
    Qdrant     = vectors + retrieval payload (see qdrant_payload.py).
    Filesystem = document binaries only (see infrastructure/storage.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    description: str
    sql: str


# ---------------------------------------------------------------------------
# Migration 001 — Core document tables
# ---------------------------------------------------------------------------
MIGRATION_001 = Migration(
    version=1,
    description="Core document tables: documents, document_versions",
    sql="""
-- documents: canonical document identity and lifecycle timestamps.
-- One row per logical document (across all versions).
CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    latest_version  INTEGER NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- document_versions: one row per uploaded file version.
-- Stores all structured metadata, ingestion state, OCR results,
-- and semantic indexing status for that version.
CREATE TABLE IF NOT EXISTS document_versions (
    document_id             TEXT        NOT NULL,
    version                 INTEGER     NOT NULL,
    original_filename       TEXT        NOT NULL,
    content_type            TEXT,
    file_kind               TEXT        NOT NULL,
    storage_key             TEXT        NOT NULL,
    file_size_bytes         BIGINT      NOT NULL,
    sha256                  TEXT        NOT NULL,
    privacy                 TEXT        NOT NULL,
    processing_status       TEXT        NOT NULL,
    metadata                JSONB       NOT NULL DEFAULT '{}'::jsonb,
    description             TEXT,
    extracted_text          TEXT,
    extracted_text_excerpt  TEXT,
    semantic_index_status   TEXT        NOT NULL DEFAULT 'PENDING',
    semantic_indexed_at     TIMESTAMPTZ,
    chunk_count             INTEGER     NOT NULL DEFAULT 0,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at            TIMESTAMPTZ,
    error_message           TEXT,
    PRIMARY KEY (document_id, version),
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_document_versions_status
    ON document_versions (processing_status);

CREATE INDEX IF NOT EXISTS idx_document_versions_privacy
    ON document_versions (privacy);

CREATE INDEX IF NOT EXISTS idx_document_versions_semantic_status
    ON document_versions (semantic_index_status);
""",
)

# ---------------------------------------------------------------------------
# Migration 002 — Audit events
# ---------------------------------------------------------------------------
MIGRATION_002 = Migration(
    version=2,
    description="Audit events for sensitive access tracking",
    sql="""
-- audit_events: records every access decision that is not routine.
-- Used for accountability and compliance.  Does NOT store the sensitive
-- value itself — only that an access was attempted and the outcome.
CREATE TABLE IF NOT EXISTS audit_events (
    id              BIGSERIAL   PRIMARY KEY,
    document_id     TEXT        NOT NULL,
    version         INTEGER     NOT NULL,
    intent          TEXT        NOT NULL,
    query           TEXT,
    requestor       TEXT,
    access_action   TEXT        NOT NULL,
    access_reason   TEXT        NOT NULL,
    privacy         TEXT        NOT NULL,
    redacted        BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_audit_events_document
    ON audit_events (document_id, version);

CREATE INDEX IF NOT EXISTS idx_audit_events_created
    ON audit_events (created_at DESC);
""",
)

# ---------------------------------------------------------------------------
# Migration 003 — Document relationships (contextual/provenance links)
# ---------------------------------------------------------------------------
MIGRATION_003 = Migration(
    version=3,
    description="Document relationship graph for contextual retrieval",
    sql="""
-- document_relationships: task/contextual relationships between documents.
-- Supports graph-style retrieval alongside vector search.
-- Relationship types follow the Chitragupta architecture spec:
--   REQUIRED_FOR, SUPPORTS, PROVES, REFERENCES, CONTRADICTS,
--   SUPERSEDES, DEPENDS_ON, COMPLEMENTS
CREATE TABLE IF NOT EXISTS document_relationships (
    id                  BIGSERIAL   PRIMARY KEY,
    source_document_id  TEXT        NOT NULL,
    target_document_id  TEXT        NOT NULL,
    relationship_type   TEXT        NOT NULL,
    metadata            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (source_document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
    FOREIGN KEY (target_document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
    UNIQUE (source_document_id, target_document_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_doc_rel_source
    ON document_relationships (source_document_id);

CREATE INDEX IF NOT EXISTS idx_doc_rel_target
    ON document_relationships (target_document_id);

CREATE INDEX IF NOT EXISTS idx_doc_rel_type
    ON document_relationships (relationship_type);
""",
)

# ---------------------------------------------------------------------------
# Migration 005 — Owner/relation/summary/expiry + extracted fields
# ---------------------------------------------------------------------------
MIGRATION_005 = Migration(
    version=5,
    description="Add owner/relation/summary/expiry and split extracted fields for redaction",
    sql="""
-- Adds columns that let the orchestrator know whose document this is
-- (self / mother / etc.) and how to render it in chat.
-- Also splits the document body into:
--   * summary  — short, redacted, safe to show
--   * extracted_fields — structured values (license_number, dob, ...),
--                          encrypted at rest by the application
--   * expiry_date — used by the validator to flag expired docs
-- The existing ``extracted_text`` column remains for backward compatibility.

ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS summary             TEXT,
    ADD COLUMN IF NOT EXISTS extracted_fields    JSONB,
    ADD COLUMN IF NOT EXISTS owner_type          TEXT,
    ADD COLUMN IF NOT EXISTS relation            TEXT,
    ADD COLUMN IF NOT EXISTS relation_name       TEXT,
    ADD COLUMN IF NOT EXISTS expiry_date         TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_doc_versions_owner_type
    ON document_versions (owner_type);

CREATE INDEX IF NOT EXISTS idx_doc_versions_relation
    ON document_versions (relation);

CREATE INDEX IF NOT EXISTS idx_doc_versions_expiry
    ON document_versions (expiry_date);

-- GIN index for the structured fields JSONB
CREATE INDEX IF NOT EXISTS idx_doc_versions_extracted_fields
    ON document_versions USING GIN (extracted_fields);
""",
)

MIGRATION_004 = Migration(
    version=4,
    description="Model extraction columns for structured document metadata",
    sql="""
-- Adds columns for model-service extraction output.
-- model_extraction stores the full JSON schema output from the model service.
-- Derived columns are extracted for indexed queries.
-- Existing privacy/description/metadata columns remain for backward compatibility.
-- IF NOT EXISTS (like every other migration here): a crash between the
-- ALTER and the schema_migrations bookkeeping must be safely retryable.

ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS model_extraction     JSONB,
    ADD COLUMN IF NOT EXISTS description_safe     TEXT,
    ADD COLUMN IF NOT EXISTS description_detailed TEXT,
    ADD COLUMN IF NOT EXISTS extraction_confidence REAL,
    ADD COLUMN IF NOT EXISTS document_type        TEXT,
    ADD COLUMN IF NOT EXISTS document_sub_type    TEXT,
    ADD COLUMN IF NOT EXISTS language_primary     TEXT,
    ADD COLUMN IF NOT EXISTS pii_types            JSONB;

CREATE INDEX IF NOT EXISTS idx_doc_versions_doc_type
    ON document_versions (document_type);

CREATE INDEX IF NOT EXISTS idx_doc_versions_sub_type
    ON document_versions (document_sub_type);

CREATE INDEX IF NOT EXISTS idx_doc_versions_language
    ON document_versions (language_primary);

CREATE INDEX IF NOT EXISTS idx_doc_versions_confidence
    ON document_versions (extraction_confidence);

-- GIN index for PII type queries
CREATE INDEX IF NOT EXISTS idx_doc_versions_pii_types
    ON document_versions USING GIN (pii_types);

-- GIN index for model_extraction JSONB queries
CREATE INDEX IF NOT EXISTS idx_doc_versions_model_extraction
    ON document_versions USING GIN (model_extraction);
""",
)

# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------
MIGRATIONS: list[Migration] = sorted(
    [MIGRATION_001, MIGRATION_002, MIGRATION_003, MIGRATION_004, MIGRATION_005],
    key=lambda m: m.version,
)


def apply_migrations(connection_factory) -> None:  # type: ignore[no-untyped-def]
    """Apply all pending migrations in order.

    Uses a lightweight ``schema_migrations`` tracking table so already-applied
    migrations are skipped on subsequent calls.
    """
    with connection_factory() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version     INTEGER PRIMARY KEY,
                    description TEXT    NOT NULL,
                    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        conn.commit()

    for migration in MIGRATIONS:
        _apply_one(connection_factory, migration)


def _apply_one(connection_factory, migration: Migration) -> None:  # type: ignore[no-untyped-def]
    with connection_factory() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s",
                (migration.version,),
            )
            if cur.fetchone() is not None:
                return
            cur.execute(migration.sql)
            cur.execute(
                "INSERT INTO schema_migrations (version, description) VALUES (%s, %s)",
                (migration.version, migration.description),
            )
        conn.commit()
