from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import blake2b
import math
import re
from typing import Any, Protocol, Sequence

from document_mgmt_service.domain.models import (
    DocumentFileKind,
    DocumentPrivacyClassification,
    SemanticContentSearchResult,
    SemanticDocumentSearchResult,
    SemanticEvidenceResult,
    DocumentVersionRecord,
    SemanticChunkRecord,
)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class SemanticChunkMatch:
    chunk: SemanticChunkRecord
    score: float


class SemanticChunkStore(Protocol):
    def upsert(self, *, chunks: Sequence[SemanticChunkRecord], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(
        self,
        *,
        query_vector: Sequence[float],
        limit: int,
        document_id: str | None = None,
        version: int | None = None,
        privacy: DocumentPrivacyClassification | None = None,
    ) -> list[SemanticChunkMatch]: ...

    def delete_document(self, document_id: str, version: int | None = None) -> None: ...


class TextEmbeddingService:
    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            index = self._token_index(token)
            vector[index] += 1.0
            if len(token) > 3:
                for ngram in self._character_ngrams(token, 3):
                    vector[self._token_index(ngram)] += 0.25
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def _token_index(self, token: str) -> int:
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimension

    def _character_ngrams(self, token: str, size: int) -> list[str]:
        if len(token) <= size:
            return [token]
        return [token[i : i + size] for i in range(len(token) - size + 1)]


class SemanticChunker:
    def __init__(self, *, max_chunk_chars: int = 900, chunk_overlap: int = 120) -> None:
        self._max_chunk_chars = max_chunk_chars
        self._chunk_overlap = min(chunk_overlap, max_chunk_chars // 2)

    def chunk(self, record: DocumentVersionRecord) -> list[SemanticChunkRecord]:
        page_chunks = self._split_pages(record)
        chunks: list[SemanticChunkRecord] = []
        chunk_index = 0
        for page_number, page_text, page_offset in page_chunks:
            for start, end, text in self._chunk_page(page_text):
                if not text.strip():
                    continue
                chunks.append(
                    SemanticChunkRecord(
                        chunk_id=self._chunk_id(record.document_id, record.version, chunk_index),
                        document_id=record.document_id,
                        version=record.version,
                        chunk_index=chunk_index,
                        page_number=page_number,
                        text=text,
                        start_char=page_offset + start,
                        end_char=page_offset + end,
                        privacy=record.privacy,
                        metadata={
                            **dict(record.metadata),
                            "file_kind": record.file_kind.value,
                            "semantic_index_status": record.semantic_index_status.value,
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                            # V2: owner + relation are indexed so the LLM can
                            # answer questions like "show me my mother's docs".
                            "owner_type": getattr(record, "owner_type", None),
                            "relation": getattr(record, "relation", None),
                            "relation_name": getattr(record, "relation_name", None),
                            # V2: field_pointers is the set of fields the model
                            # extracted. The actual values never enter Qdrant.
                            "field_pointers": (
                                list((record.extracted_fields or {}).keys())
                                if record.extracted_fields else None
                            ),
                        },
                        description=record.description,
                        original_filename=record.original_filename,
                        content_type=record.content_type,
                        created_at=record.completed_at or record.updated_at or record.created_at,
                    )
                )
                chunk_index += 1

        if not chunks:
            fallback_text = self._fallback_text(record)
            chunks.append(
                SemanticChunkRecord(
                    chunk_id=self._chunk_id(record.document_id, record.version, 0),
                    document_id=record.document_id,
                    version=record.version,
                    chunk_index=0,
                    page_number=1,
                    text=fallback_text,
                    start_char=0,
                    end_char=len(fallback_text),
                    privacy=record.privacy,
                    metadata={
                        **dict(record.metadata),
                        "file_kind": record.file_kind.value,
                        "semantic_index_status": record.semantic_index_status.value,
                        "page_number": 1,
                        "chunk_index": 0,
                        "owner_type": getattr(record, "owner_type", None),
                        "relation": getattr(record, "relation", None),
                        "relation_name": getattr(record, "relation_name", None),
                        "field_pointers": (
                            list((record.extracted_fields or {}).keys())
                            if record.extracted_fields else None
                        ),
                        "source": "document_summary",
                    },
                    description=record.description,
                    original_filename=record.original_filename,
                    content_type=record.content_type,
                    created_at=record.completed_at or record.updated_at or record.created_at,
                )
            )
        return chunks

    def _split_pages(self, record: DocumentVersionRecord) -> list[tuple[int, str, int]]:
        text = record.extracted_text or ""
        if not text.strip():
            return [(1, self._fallback_text(record), 0)]

        pages = text.split("\f")
        offsets: list[tuple[int, str, int]] = []
        cursor = 0
        for index, page in enumerate(pages, start=1):
            offsets.append((index, page, cursor))
            cursor += len(page) + 1
        return offsets

    def _chunk_page(self, page_text: str) -> list[tuple[int, int, str]]:
        page_text = page_text.strip()
        if not page_text:
            return []
        if len(page_text) <= self._max_chunk_chars:
            return [(0, len(page_text), page_text)]

        chunks: list[tuple[int, int, str]] = []
        start = 0
        text_len = len(page_text)
        while start < text_len:
            end = min(start + self._max_chunk_chars, text_len)
            if end < text_len:
                boundary = page_text.rfind(" ", start, end)
                if boundary > start + 120:
                    end = boundary
            chunk_text = page_text[start:end].strip()
            if chunk_text:
                chunks.append((start, end, chunk_text))
            if end >= text_len:
                break
            start = max(end - self._chunk_overlap, start + 1)
        return chunks

    def _fallback_text(self, record: DocumentVersionRecord) -> str:
        pieces = [
            record.original_filename,
            record.description or "",
            " ".join(f"{key}:{value}" for key, value in sorted(record.metadata.items())),
        ]
        return " ".join(piece for piece in pieces if piece).strip() or record.original_filename

    def _chunk_id(self, document_id: str, version: int, chunk_index: int) -> str:
        return f"{document_id}:v{version}:c{chunk_index}"


class SemanticSearchService:
    def __init__(
        self,
        *,
        store: SemanticChunkStore,
        embedder: TextEmbeddingService,
        chunker: SemanticChunker,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._chunker = chunker

    def index_document(self, record: DocumentVersionRecord) -> int:
        chunks = self._chunker.chunk(record)
        vectors = [self._embedder.embed(chunk.text) for chunk in chunks]
        self._store.upsert(chunks=chunks, vectors=vectors)
        return len(chunks)

    def delete_document(self, document_id: str, version: int | None = None) -> None:
        self._store.delete_document(document_id, version)

    def search_documents(
        self,
        query: str,
        *,
        limit: int = 10,
        privacy: DocumentPrivacyClassification | None = None,
    ) -> list[SemanticDocumentSearchResult]:
        matches = self._store.search(
            query_vector=self._embedder.embed(query),
            limit=max(limit * 4, limit),
            privacy=privacy,
        )
        grouped: dict[tuple[str, int], dict[str, Any]] = {}
        for match in matches:
            key = (match.chunk.document_id, match.chunk.version)
            bucket = grouped.setdefault(
                key,
                {
                    "score": match.score,
                    "chunk": match.chunk,
                    "evidence": [],
                },
            )
            if match.score > bucket["score"]:
                bucket["score"] = match.score
                bucket["chunk"] = match.chunk
            if len(bucket["evidence"]) < 3:
                bucket["evidence"].append(self._evidence_payload(match))

        ranked = sorted(
            grouped.items(),
            key=lambda item: (-item[1]["score"], item[0][0], item[0][1]),
        )
        return [
            SemanticDocumentSearchResult(
                document_id=document_id,
                version=version,
                score=bucket["score"],
                privacy=bucket["chunk"].privacy,
                description=bucket["chunk"].description,
                metadata=dict(bucket["chunk"].metadata),
                provenance=self._provenance_payload(bucket["chunk"]),
                evidence=tuple(bucket["evidence"]),
            )
            for (document_id, version), bucket in ranked[:limit]
        ]

    def search_content(
        self,
        query: str,
        *,
        limit: int = 10,
        document_id: str | None = None,
        version: int | None = None,
        privacy: DocumentPrivacyClassification | None = None,
    ) -> list[SemanticContentSearchResult]:
        matches = self._store.search(
            query_vector=self._embedder.embed(query),
            limit=limit,
            document_id=document_id,
            version=version,
            privacy=privacy,
        )
        return [
            SemanticContentSearchResult(
                chunk_id=match.chunk.chunk_id,
                document_id=match.chunk.document_id,
                version=match.chunk.version,
                score=match.score,
                privacy=match.chunk.privacy,
                text=match.chunk.text,
                metadata=dict(match.chunk.metadata),
                provenance=self._provenance_payload(match.chunk),
            )
            for match in matches
        ]

    def retrieve_evidence(
        self,
        *,
        document_id: str,
        version: int,
        query: str,
        limit: int = 5,
    ) -> list[SemanticEvidenceResult]:
        matches = self._store.search(
            query_vector=self._embedder.embed(query),
            limit=limit,
            document_id=document_id,
            version=version,
        )
        return [
            SemanticEvidenceResult(
                document_id=match.chunk.document_id,
                version=match.chunk.version,
                query=query,
                score=match.score,
                privacy=match.chunk.privacy,
                text=match.chunk.text,
                metadata=dict(match.chunk.metadata),
                provenance=self._provenance_payload(match.chunk),
            )
            for match in matches
        ]

    def _evidence_payload(self, match: SemanticChunkMatch) -> dict[str, Any]:
        return {
            "chunk_id": match.chunk.chunk_id,
            "text": match.chunk.text,
            "score": match.score,
            "provenance": self._provenance_payload(match.chunk),
        }

    def _provenance_payload(self, chunk: SemanticChunkRecord) -> dict[str, Any]:
        return {
            "document_id": chunk.document_id,
            "version": chunk.version,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "char_start": chunk.start_char,
            "char_end": chunk.end_char,
            "original_filename": chunk.original_filename,
            "content_type": chunk.content_type,
        }
