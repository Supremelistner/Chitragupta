"""Test helpers for the document-management service.

With the docker-only refactor, all document service tests now require a live
Postgres + Qdrant. This module exposes small helpers that wire up
per-test-isolated instances of both:

* ``build_postgres_repo()`` — opens a Postgres connection against the project's
  DSN, ensures the schema, then issues ``SET search_path`` so all queries in
  this test go through a private, randomly-named namespace. The namespace is
  torn down on ``addCleanup`` (a CASCADE DROP SCHEMA), so tests can run in
  any order without leaving residue.

* ``build_qdrant_store()`` — creates a randomly-named Qdrant collection,
  hands back a configured :class:`QdrantSemanticChunkStoreAdapter`, and
  cleans up the collection on teardown.

If either dependency is unreachable the test is skipped with a clear message
— the test suite must remain runnable locally even when the docker stack is
down. CI should fail when the deps are missing, but we get that for free via
``unittest.SkipTest``.
"""
from __future__ import annotations

import os
import secrets
import sys
import unittest
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_mgmt_service.infrastructure.postgres import (
    PostgreSQLRepositoryAdapter,
    create_psycopg_connection_factory,
)
from document_mgmt_service.infrastructure.qdrant import QdrantSemanticChunkStoreAdapter


DEFAULT_DSN = os.environ.get(
    "DOCUMENT_SERVICE_POSTGRES_DSN",
    "postgresql://chitragupta:chitragupta_dev@localhost:5432/chitragupta",
)
DEFAULT_QDRANT_URL = os.environ.get(
    "DOCUMENT_SERVICE_QDRANT_URL", "http://localhost:6333"
)


def _can_connect_postgres(dsn: str) -> bool:
    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


def _can_connect_qdrant(url: str) -> bool:
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def build_postgres_repo(
    testcase: unittest.TestCase,
    *,
    dsn: str = DEFAULT_DSN,
) -> PostgreSQLRepositoryAdapter:
    """Create a per-test Postgres repository.

    Creates a private schema, runs migrations into it, and returns a repo
    bound to that schema. The schema is dropped on test teardown.
    """
    if not _can_connect_postgres(dsn):
        raise unittest.SkipTest(
            f"Postgres is not reachable at {dsn}. Start the docker compose "
            "stack (`python activate.py --bg`) to run this test."
        )
    import psycopg  # type: ignore[import-not-found]

    base_factory = create_psycopg_connection_factory(dsn, connect_timeout=2)

    # Create private schema
    schema_name = f"t_{secrets.token_hex(6)}"
    with base_factory() as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{schema_name}"')
        conn.commit()

    def factory() -> Any:
        conn = base_factory()
        # All DDL/DML in this test must land in the private schema.
        with conn.cursor() as cur:
            cur.execute(f'SET search_path TO "{schema_name}"')
        return conn

    repo = PostgreSQLRepositoryAdapter(factory)
    # The migrations file is keyed off search_path; running ensure_schema
    # against our private schema gives us an isolated copy of all tables.
    repo.ensure_schema()

    def _drop() -> None:
        try:
            with base_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
                conn.commit()
        except Exception:
            pass  # Best-effort cleanup; do not mask real test failures.

    testcase.addCleanup(_drop)
    return repo


def build_qdrant_store(
    testcase: unittest.TestCase,
    *,
    base_url: str = DEFAULT_QDRANT_URL,
    dimension: int = 256,
    encryption_key: str = "",
) -> QdrantSemanticChunkStoreAdapter:
    """Create a per-test Qdrant collection + configured adapter."""
    if not _can_connect_qdrant(base_url):
        raise unittest.SkipTest(
            f"Qdrant is not reachable at {base_url}. Start the docker compose "
            "stack (`python activate.py --bg`) to run this test."
        )
    import json
    import urllib.request

    collection_name = f"t_{secrets.token_hex(6)}"
    adapter = QdrantSemanticChunkStoreAdapter(
        base_url=base_url,
        collection_name=collection_name,
        encryption_key=encryption_key,
        dimension=dimension,
        strict=True,
    )

    def _drop() -> None:
        try:
            url = f"{base_url.rstrip('/')}/collections/{collection_name}"
            req = urllib.request.Request(url, method="DELETE")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception:
            pass

    testcase.addCleanup(_drop)
    return adapter


def skip_if_stack_down(testcase: unittest.TestCase) -> None:
    """Convenience: raise SkipTest if either dependency is missing."""
    if not _can_connect_postgres(DEFAULT_DSN):
        raise unittest.SkipTest(
            "Postgres is not reachable — start `python activate.py --bg`."
        )
    if not _can_connect_qdrant(DEFAULT_QDRANT_URL):
        raise unittest.SkipTest(
            "Qdrant is not reachable — start `python activate.py --bg`."
        )
