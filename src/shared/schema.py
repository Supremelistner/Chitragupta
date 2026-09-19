"""Shared PostgreSQL schema — migrations for cross-service data contracts.

These migrations extend the Document Management Service schema with tables
that are shared across services:

    - validation_runs: Record of every validation execution
    - template_versions: Versioned document templates
    - model_executions: Record of every model service call
    - document_alerts: Validation alerts sent to document service

Ownership:
    PostgreSQL = canonical metadata, state, and provenance (this module)
    Qdrant = vectors + retrieval payload (see document_mgmt_service/infrastructure/qdrant.py)
    Filesystem = document binaries (see storage.py)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    description: str
    sql: str


# ---------------------------------------------------------------------------
# Migration 005 — Validation runs and template versions
# ---------------------------------------------------------------------------
MIGRATION_005 = Migration(
    version=5,
    description="Validation runs, template versions, and model executions",
    sql="""
-- validation_runs: record of every validation execution.
-- Links to document_versions via (document_id, version).
-- Links to model_executions via model_execution_id.
CREATE TABLE IF NOT EXISTS validation_runs (
    run_id              TEXT        PRIMARY KEY,
    document_id         TEXT        NOT NULL,
    version             INTEGER     NOT NULL,
    template_id         TEXT,
    template_version    TEXT,
    status              TEXT        NOT NULL DEFAULT 'UNKNOWN_DOCUMENT',
    risk_score          REAL        NOT NULL DEFAULT 0.0,
    is_authentic        BOOLEAN     NOT NULL DEFAULT TRUE,
    violations          JSONB       NOT NULL DEFAULT '[]'::jsonb,
    temporal_status     TEXT        NOT NULL DEFAULT 'UNKNOWN',
    temporal_tags       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    structural_confidence REAL      NOT NULL DEFAULT 0.0,
    model_execution_id  TEXT,
    notes               JSONB       NOT NULL DEFAULT '[]'::jsonb,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    FOREIGN KEY (document_id, version)
        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_validation_runs_document
    ON validation_runs (document_id, version);

CREATE INDEX IF NOT EXISTS idx_validation_runs_template
    ON validation_runs (template_id);

CREATE INDEX IF NOT EXISTS idx_validation_runs_status
    ON validation_runs (status);

CREATE INDEX IF NOT EXISTS idx_validation_runs_started
    ON validation_runs (started_at DESC);


-- template_versions: versioned document templates.
-- One row per template variant (e.g., aadhaar_card_v1, marksheet_v1).
-- Supports multiple versions of the same template for backward compatibility.
CREATE TABLE IF NOT EXISTS template_versions (
    template_id         TEXT        NOT NULL,
    version             TEXT        NOT NULL DEFAULT '1.0.0',
    document_type       TEXT        NOT NULL,
    document_sub_type   TEXT        NOT NULL,
    variant_name        TEXT        NOT NULL,
    description         TEXT,
    field_rules         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    structural_rules    JSONB       NOT NULL DEFAULT '[]'::jsonb,
    temporal_rules      JSONB       NOT NULL DEFAULT '[]'::jsonb,
    field_count         INTEGER     NOT NULL DEFAULT 0,
    temporal_rule_count INTEGER     NOT NULL DEFAULT 0,
    is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (template_id, version)
);

CREATE INDEX IF NOT EXISTS idx_template_versions_type
    ON template_versions (document_type, document_sub_type);

CREATE INDEX IF NOT EXISTS idx_template_versions_active
    ON template_versions (is_active) WHERE is_active = TRUE;


-- model_executions: record of every model service inference call.
-- Provides provenance: which model, which provider, what input/output.
CREATE TABLE IF NOT EXISTS model_executions (
    run_id              TEXT        PRIMARY KEY,
    document_id         TEXT        NOT NULL,
    version             INTEGER     NOT NULL,
    task                TEXT        NOT NULL,
    provider            TEXT        NOT NULL,
    model_id            TEXT        NOT NULL,
    input_hash          TEXT,
    output_summary      TEXT,
    confidence          REAL,
    latency_ms          REAL,
    token_usage         JSONB,
    error               TEXT,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    FOREIGN KEY (document_id, version)
        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_model_executions_document
    ON model_executions (document_id, version);

CREATE INDEX IF NOT EXISTS idx_model_executions_task
    ON model_executions (task);

CREATE INDEX IF NOT EXISTS idx_model_executions_started
    ON model_executions (started_at DESC);


-- document_alerts: validation alerts received by the document service.
-- Records every alert from the validator service for audit trail.
CREATE TABLE IF NOT EXISTS document_alerts (
    id                  BIGSERIAL   PRIMARY KEY,
    document_id         TEXT        NOT NULL,
    version             INTEGER,
    severity            TEXT        NOT NULL,
    alert_type          TEXT        NOT NULL,
    message             TEXT,
    risk_score          REAL        NOT NULL DEFAULT 0.0,
    source              TEXT        NOT NULL DEFAULT 'validator_service',
    details             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    received_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_document_alerts_document
    ON document_alerts (document_id, version);

CREATE INDEX IF NOT EXISTS idx_document_alerts_severity
    ON document_alerts (severity);

CREATE INDEX IF NOT EXISTS idx_document_alerts_received
    ON document_alerts (received_at DESC);
""",
)


# Migration registry — import and merge with Document Service migrations
SHARED_MIGRATIONS: list[Migration] = sorted(
    [MIGRATION_005],
    key=lambda m: m.version,
)


def apply_shared_migrations(connection_factory) -> None:  # type: ignore[no-untyped-def]
    """Apply shared migrations (validation_runs, template_versions, model_executions)."""
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

    for migration in SHARED_MIGRATIONS:
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
