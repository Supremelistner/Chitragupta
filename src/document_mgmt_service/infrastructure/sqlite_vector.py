"""SQLite-backed vector store for semantic chunks.

Persistent replacement for InMemoryQdrantVectorStore.
Stores chunk metadata + vectors in SQLite with cosine similarity search.
Uses only Python stdlib — zero external dependencies.

For V1: cosine similarity computed in Python (dimension=256, fast enough).
Can be swapped for Qdrant/PostgreSQL pgvector in production.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from document_mgmt_service.application.search import SemanticChunkMatch, SemanticChunkStore
from document_mgmt_service.domain.models import (
    DocumentPrivacyClassification,
    SemanticChunkRecord,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    text TEXT,
    start_char INTEGER,
    end_char INTEGER,
    privacy TEXT DEFAULT 'OPEN_NOT_PUBLIC',
    metadata TEXT DEFAULT '{}',
    description TEXT,
    original_filename TEXT,
    content_type TEXT,
    created_at TEXT,
    vector BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_ver
    ON semantic_chunks (document_id, version);

CREATE INDEX IF NOT EXISTS idx_chunks_privacy
    ON semantic_chunks (privacy);
"""


def _vector_to_blob(vector: Sequence[float]) -> bytes:
    """Pack float vector into compact binary blob using struct."""
    import struct
    return struct.pack(f"{len(vector)}f", *vector)


def _blob_to_vector(blob: bytes, expected_dim: int) -> list[float]:
    """Unpack binary blob back to float vector."""
    import struct
    return list(struct.unpack(f"{expected_dim}f", blob))


class SQLiteVectorStore(SemanticChunkStore):
    """SQLite-backed vector store.

    Stores chunks with their embedding vectors.
    Cosine similarity search computed in Python — fast enough for V1
    with dimension=256 and <10k chunks.
    """

    def __init__(self, db_path: str = "./data/chitragupta.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._dimension: int = 256
        self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def ping(self) -> None:
        if self._conn is None:
            raise RuntimeError("SQLite vector store connection is closed")
        self._conn.execute("SELECT 1")

    def set_dimension(self, dimension: int) -> None:
        """Set the expected vector dimension (for blob unpacking)."""
        self._dimension = dimension

    def upsert(self, *, chunks: Sequence[SemanticChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        assert self._conn is not None

        for chunk, vector in zip(chunks, vectors, strict=True):
            blob = _vector_to_blob(vector)
            self._conn.execute(
                """
                INSERT INTO semantic_chunks (
                    chunk_id, document_id, version, chunk_index, page_number,
                    text, start_char, end_char, privacy, metadata,
                    description, original_filename, content_type, created_at, vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = excluded.text,
                    vector = excluded.vector,
                    metadata = excluded.metadata,
                    description = excluded.description
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.version,
                    chunk.chunk_index,
                    chunk.page_number,
                    chunk.text,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.privacy.value,
                    json.dumps(chunk.metadata, default=str),
                    chunk.description,
                    chunk.original_filename,
                    chunk.content_type,
                    chunk.created_at.isoformat() if chunk.created_at else None,
                    blob,
                ),
            )
        self._conn.commit()

    def search(
        self,
        *,
        query_vector: Sequence[float],
        limit: int,
        document_id: str | None = None,
        version: int | None = None,
        privacy: DocumentPrivacyClassification | None = None,
    ) -> list[SemanticChunkMatch]:
        assert self._conn is not None

        # Build WHERE clause for pre-filtering
        conditions: list[str] = []
        params: list[Any] = []
        if document_id is not None:
            conditions.append("document_id = ?")
            params.append(document_id)
        if version is not None:
            conditions.append("version = ?")
            params.append(version)
        if privacy is not None:
            conditions.append("privacy = ?")
            params.append(privacy.value)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        rows = self._conn.execute(
            f"SELECT * FROM semantic_chunks {where}",
            params,
        ).fetchall()

        # Compute cosine similarity in Python
        q = list(query_vector)
        scored: list[SemanticChunkMatch] = []
        for row in rows:
            blob = row["vector"]
            if blob is None:
                continue
            vec = _blob_to_vector(blob, self._dimension)
            score = _cosine_similarity(q, vec)
            chunk = SemanticChunkRecord(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                version=int(row["version"]),
                chunk_index=int(row["chunk_index"]),
                page_number=int(row["page_number"]) if row["page_number"] is not None else None,
                text=row["text"] or "",
                start_char=int(row["start_char"] or 0),
                end_char=int(row["end_char"] or 0),
                privacy=DocumentPrivacyClassification(row["privacy"]),
                metadata=json.loads(row["metadata"] or "{}"),
                description=row["description"],
                original_filename=row["original_filename"],
                content_type=row["content_type"],
                created_at=_parse_iso(row["created_at"]),
            )
            scored.append(SemanticChunkMatch(chunk=chunk, score=score))

        scored.sort(key=lambda item: (-item.score, item.chunk.document_id, item.chunk.version, item.chunk.chunk_index))
        return scored[:limit]

    def delete_document(self, document_id: str, version: int | None = None) -> None:
        assert self._conn is not None
        if version is not None:
            self._conn.execute(
                "DELETE FROM semantic_chunks WHERE document_id = ? AND version = ?",
                (document_id, version),
            )
        else:
            self._conn.execute(
                "DELETE FROM semantic_chunks WHERE document_id = ?",
                (document_id,),
            )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    if length == 0:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(length))
    left_norm = sum(v * v for v in left[:length]) ** 0.5
    right_norm = sum(v * v for v in right[:length]) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
