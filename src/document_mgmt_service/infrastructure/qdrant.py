"""Qdrant vector store adapter.

Communicates with Qdrant over its REST API (port 6333).
Creates the collection automatically on first use.
Uses only Python stdlib (urllib) — no qdrant-client dependency needed.

Can be swapped for pgvector, Pinecone, or any other vector DB.
"""
from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
import uuid
from typing import Any, Sequence

from document_mgmt_service.application.search import SemanticChunkMatch, SemanticChunkStore
from document_mgmt_service.domain.models import (
    DocumentPrivacyClassification,
    SemanticChunkRecord,
)

logger = logging.getLogger("document_mgmt_service.qdrant")


class QdrantSemanticChunkStoreAdapter(SemanticChunkStore):
    """Qdrant-backed vector store.

    Stores chunk metadata as payload and vectors as Qdrant points.
    Collection is auto-created with the right dimension on first use.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:6333",
        collection_name: str = "document_chunks",
        encryption_key: str = "",
        dimension: int = 256,
        strict: bool = True,
    ) -> None:
        """Qdrant-backed vector store.

        This adapter always runs in strict mode. Connection failures,
        collection-creation errors, and upsert errors all raise
        ``RuntimeError`` — there is no silent in-memory fallback. The
        document service is expected to run with a live Qdrant container;
        if the container is missing or misconfigured, the service must
        fail fast so the operator notices.
        """
        self._base_url = base_url.rstrip("/")
        self._collection_name = collection_name
        self._dimension: int = dimension
        self._encryption_key = encryption_key
        self._strict = strict
        self._ensure_collection()

    def set_dimension(self, dimension: int) -> None:
        self._dimension = dimension

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Always strict."""
        try:
            url = f"{self._base_url}/collections/{self._collection_name}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return  # Already exists
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise RuntimeError(
                    f"Qdrant collection check failed: HTTP {e.code} {e.reason}"
                ) from e
        except Exception as e:
            raise RuntimeError(
                f"Qdrant unreachable at {self._base_url}: {e.__class__.__name__}: {e}"
            ) from e

        # Create collection
        url = f"{self._base_url}/collections/{self._collection_name}"
        payload = json.dumps({
            "vectors": {
                "size": self._dimension,
                "distance": "Cosine",
            },
        }).encode()
        req = urllib.request.Request(
            url, data=payload, method="PUT",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info(
                    "Created Qdrant collection '%s' (dim=%d)",
                    self._collection_name, self._dimension,
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create Qdrant collection "
                f"'{self._collection_name}' at {self._base_url}: "
                f"{e.__class__.__name__}: {e}"
            ) from e

    def ping(self) -> None:
        url = f"{self._base_url}/healthz"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"Qdrant health check failed: HTTP {resp.status}"
                    )
        except Exception as exc:
            raise RuntimeError(
                f"Qdrant ping failed at {self._base_url}: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc

    def upsert(self, *, chunks: Sequence[SemanticChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            raw_payload = self._build_payload(chunk)
            encrypted = self._encrypt_if_configured(raw_payload, chunk)
            points.append({
                "id": _to_qdrant_point_id(chunk.chunk_id),
                "vector": list(vector),
                "payload": encrypted,
            })

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            url = f"{self._base_url}/collections/{self._collection_name}/points?wait=true"
            body = json.dumps({"points": batch}).encode()
            req = urllib.request.Request(
                url, data=body, method="PUT",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status not in (200, 201):
                        raise RuntimeError(
                            f"Qdrant upsert returned HTTP {resp.status}"
                        )
            except Exception as e:
                raise RuntimeError(
                    f"Qdrant upsert failed: {e.__class__.__name__}: {e}"
                ) from e

    def search(
        self,
        *,
        query_vector: Sequence[float],
        limit: int,
        document_id: str | None = None,
        version: int | None = None,
        privacy: DocumentPrivacyClassification | None = None,
    ) -> list[SemanticChunkMatch]:
        # Build filter conditions
        must_filters = []
        if document_id is not None:
            must_filters.append({
                "key": "document_id",
                "match": {"value": document_id},
            })
        if version is not None:
            must_filters.append({
                "key": "version",
                "match": {"value": version},
            })
        if privacy is not None:
            must_filters.append({
                "key": "privacy",
                "match": {"value": privacy.value},
            })

        search_body: dict[str, Any] = {
            "vector": list(query_vector),
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        if must_filters:
            search_body["filter"] = {"must": must_filters}

        url = f"{self._base_url}/collections/{self._collection_name}/points/search"
        body = json.dumps(search_body).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            raise RuntimeError(
                f"Qdrant search failed: {e.__class__.__name__}: {e}"
            ) from e

        results = []
        for hit in data.get("result", []):
            payload = self._decrypt_if_configured(hit.get("payload", {}))
            chunk = SemanticChunkRecord(
                chunk_id=payload.get("chunk_id", hit.get("id", "")),
                document_id=payload.get("document_id", ""),
                version=int(payload.get("version", 0)),
                chunk_index=int(payload.get("chunk_index", 0)),
                page_number=payload.get("page_number"),
                text=payload.get("text", ""),
                start_char=int(payload.get("start_char", 0)),
                end_char=int(payload.get("end_char", 0)),
                privacy=DocumentPrivacyClassification(payload.get("privacy", "OPEN_NOT_PUBLIC")),
                metadata=payload.get("metadata", {}),
                description=payload.get("description"),
                original_filename=payload.get("original_filename"),
                content_type=payload.get("content_type"),
                created_at=_parse_iso(payload.get("created_at")),
            )
            score = float(hit.get("score", 0.0))
            results.append(SemanticChunkMatch(chunk=chunk, score=score))

        return results

    def delete_document(self, document_id: str, version: int | None = None) -> None:
        must_filters = [{"key": "document_id", "match": {"value": document_id}}]
        if version is not None:
            must_filters.append({"key": "version", "match": {"value": version}})

        url = f"{self._base_url}/collections/{self._collection_name}/points/delete?wait=true"
        body = json.dumps({"filter": {"must": must_filters}}).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                logger.debug("Deleted Qdrant points for document %s v%s", document_id, version)
        except Exception as e:
            raise RuntimeError(
                f"Qdrant delete failed: {e.__class__.__name__}: {e}"
            ) from e

    def close(self) -> None:
        pass  # No persistent connections to close


    def _build_payload(self, chunk: SemanticChunkRecord) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "version": chunk.version,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "text": chunk.text,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "privacy": chunk.privacy.value,
            "metadata": chunk.metadata,
            "description": chunk.description,
            "original_filename": chunk.original_filename,
            "content_type": chunk.content_type,
            "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
        }
        # Surface field_pointers as a top-level plaintext field so Qdrant
        # can filter on it. The metadata copy is removed (still kept under
        # the encrypted "metadata" key for back-compat).
        if chunk.metadata and "field_pointers" in chunk.metadata:
            payload["field_pointers"] = chunk.metadata.get("field_pointers")
        if chunk.metadata and "owner_type" in chunk.metadata:
            payload["owner_type"] = chunk.metadata.get("owner_type")
        if chunk.metadata and "relation" in chunk.metadata:
            payload["relation"] = chunk.metadata.get("relation")
        if chunk.metadata and "relation_name" in chunk.metadata:
            payload["relation_name"] = chunk.metadata.get("relation_name")
        return payload

    def _encrypt_if_configured(
        self, payload: dict[str, Any], chunk: SemanticChunkRecord,
    ) -> dict[str, Any]:
        if not self._encryption_key:
            return payload
        from document_mgmt_service.infrastructure.encryption import encrypt_payload
        upload_date = chunk.created_at.isoformat() if chunk.created_at else ""
        return encrypt_payload(
            payload,
            description=chunk.description or "",
            upload_date=upload_date,
            master_key=self._encryption_key,
        )

    def _decrypt_if_configured(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._encryption_key:
            return payload
        from document_mgmt_service.infrastructure.encryption import decrypt_payload_with_date
        return decrypt_payload_with_date(payload, self._encryption_key)

def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
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


def _parse_iso(value: str | None):
    if not value:
        return None
    try:
        from datetime import datetime, timezone
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None

# Qdrant point IDs must be either unsigned integers or UUIDs.
# Our domain uses deterministic string chunk_ids (e.g. "doc:v1:0"),
# so we map them to UUID5 in a fixed namespace. The mapping is
# stable across processes and restarts, which is what we need for
# idempotent upserts.
_QDRANT_POINT_NAMESPACE = uuid.UUID("5b2c4e0a-7a99-4f1a-9f3b-1d0c0a3f7c11")


def _to_qdrant_point_id(chunk_id: str) -> str:
    """Map a deterministic string chunk_id to a stable UUID5."""
    return str(uuid.uuid5(_QDRANT_POINT_NAMESPACE, str(chunk_id)))

