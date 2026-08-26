from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import RLock
from typing import Sequence

from document_mgmt_service.application.search import SemanticChunkMatch, SemanticChunkStore
from document_mgmt_service.domain.models import DocumentPrivacyClassification, SemanticChunkRecord


@dataclass(frozen=True, slots=True)
class _StoredChunk:
    chunk: SemanticChunkRecord
    vector: tuple[float, ...]


class QdrantSemanticChunkStoreAdapter(SemanticChunkStore):
    def __init__(self, *, collection_name: str = "document_chunks") -> None:
        self._collection_name = collection_name
        self._lock = RLock()
        self._chunks: dict[str, _StoredChunk] = {}

    def ping(self) -> None:
        return None

    def upsert(self, *, chunks: Sequence[SemanticChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._chunks[chunk.chunk_id] = _StoredChunk(chunk=chunk, vector=tuple(vector))

    def search(
        self,
        *,
        query_vector: Sequence[float],
        limit: int,
        document_id: str | None = None,
        version: int | None = None,
        privacy: DocumentPrivacyClassification | None = None,
    ) -> list[SemanticChunkMatch]:
        query = tuple(query_vector)
        with self._lock:
            scored = [
                SemanticChunkMatch(chunk=stored.chunk, score=self._cosine_similarity(query, stored.vector))
                for stored in self._chunks.values()
                if self._matches_filters(stored.chunk, document_id, version, privacy)
            ]
        scored.sort(key=lambda item: (-item.score, item.chunk.document_id, item.chunk.version, item.chunk.chunk_index))
        return scored[:limit]

    def delete_document(self, document_id: str, version: int | None = None) -> None:
        with self._lock:
            to_delete = [
                chunk_id
                for chunk_id, stored in self._chunks.items()
                if stored.chunk.document_id == document_id
                and (version is None or stored.chunk.version == version)
            ]
            for chunk_id in to_delete:
                self._chunks.pop(chunk_id, None)

    def _matches_filters(
        self,
        chunk: SemanticChunkRecord,
        document_id: str | None,
        version: int | None,
        privacy: DocumentPrivacyClassification | None,
    ) -> bool:
        if document_id is not None and chunk.document_id != document_id:
            return False
        if version is not None and chunk.version != version:
            return False
        if privacy is not None and chunk.privacy != privacy:
            return False
        return True

    def _cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        length = min(len(left), len(right))
        if length == 0:
            return 0.0
        dot = sum(left[i] * right[i] for i in range(length))
        left_norm = sum(value * value for value in left[:length]) ** 0.5
        right_norm = sum(value * value for value in right[:length]) ** 0.5
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)
