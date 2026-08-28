"""Qdrant vector store adapter.

Communicates with Qdrant over its REST API (port 6333).
Creates the collection automatically on first use.
Uses only Python stdlib (urllib) — no qdrant-client dependency needed.

Can be swapped for pgvector, Pinecone, or any other vector DB.
"""
from __future__ import annotations

import json
import logging
import math
import urllib.request
import urllib.error
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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._collection_name = collection_name
        self._dimension: int = 256
        self._encryption_key = encryption_key
        self._use_memory = False
        self._memory_chunks: dict[str, tuple[SemanticChunkRecord, tuple[float, ...], dict[str, Any]]] = {}
        self._ensure_collection()

    def set_dimension(self, dimension: int) -> None:
        self._dimension = dimension

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Falls back to in-memory if unreachable."""
        try:
            url = f"{self._base_url}/collections/{self._collection_name}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return  # Already exists
        except urllib.error.HTTPError as e:
            if e.code != 404:
                logger.warning("Qdrant collection check failed: %s", e)
                return
        except Exception as e:
            logger.warning("Qdrant unreachable, falling back to in-memory: %s", e)
            self._use_memory = True
            return

        # Create collection
        try:
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
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info(
                    "Created Qdrant collection '%s' (dim=%d)",
                    self._collection_name, self._dimension,
                )
        except Exception as e:
            logger.warning("Failed to create Qdrant collection, falling back to in-memory: %s", e)
            self._use_memory = True

    def ping(self) -> None:
        if self._use_memory:
            return
        url = f"{self._base_url}/healthz"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Qdrant health check failed: {resp.status}")

    def upsert(self, *, chunks: Sequence[SemanticChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        if self._use_memory:
            for chunk, vector in zip(chunks, vectors, strict=True):
                raw_payload = self._build_payload(chunk)
                encrypted = self._encrypt_if_configured(raw_payload, chunk)
                self._memory_chunks[chunk.chunk_id] = (chunk, tuple(vector), encrypted)
            return

        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            raw_payload = self._build_payload(chunk)
            encrypted = self._encrypt_if_configured(raw_payload, chunk)
            points.append({
                "id": chunk.chunk_id,
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
                        logger.warning("Qdrant upsert returned %d", resp.status)
            except Exception as e:
                logger.warning("Qdrant upsert failed, falling back to in-memory: %s", e)
                self._use_memory = True
                # Re-do this batch in memory
                for chunk, vector in zip(chunks[i:i + batch_size], vectors[i:i + batch_size], strict=True):
                    raw_payload = self._build_payload(chunk)
                    encrypted = self._encrypt_if_configured(raw_payload, chunk)
                    self._memory_chunks[chunk.chunk_id] = (chunk, tuple(vector), encrypted)

    def search(
        self,
        *,
        query_vector: Sequence[float],
        limit: int,
        document_id: str | None = None,
        version: int | None = None,
        privacy: DocumentPrivacyClassification | None = None,
    ) -> list[SemanticChunkMatch]:
        if self._use_memory:
            return self._memory_search(query_vector, limit, document_id, version, privacy)

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
            logger.error("Qdrant search failed: %s", e)
            return []

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
        if self._use_memory:
            to_delete = [
                cid for cid, (chunk, _, _) in self._memory_chunks.items()
                if chunk.document_id == document_id and (version is None or chunk.version == version)
            ]
            for cid in to_delete:
                del self._memory_chunks[cid]
            return

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
            logger.error("Qdrant delete failed: %s", e)

    def close(self) -> None:
        pass  # No persistent connections to close


    def _build_payload(self, chunk: SemanticChunkRecord) -> dict[str, Any]:
        return {
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

    def _memory_search(
        self,
        query_vector: Sequence[float],
        limit: int,
        document_id: str | None,
        version: int | None,
        privacy: DocumentPrivacyClassification | None,
    ) -> list[SemanticChunkMatch]:
        q = tuple(query_vector)
        scored: list[SemanticChunkMatch] = []
        for chunk, stored_vec, _payload in self._memory_chunks.values():
            if document_id is not None and chunk.document_id != document_id:
                continue
            if version is not None and chunk.version != version:
                continue
            if privacy is not None and chunk.privacy != privacy:
                continue
            score = _cosine_similarity(q, stored_vec)
            scored.append(SemanticChunkMatch(chunk=chunk, score=score))
        scored.sort(key=lambda item: (-item.score, item.chunk.document_id, item.chunk.version, item.chunk.chunk_index))
        return scored[:limit]


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
