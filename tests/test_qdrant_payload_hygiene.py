"""Live-stack guard: Qdrant payloads must never leak plaintext PII hints.

Regression test for the P0 ``_enc_desc`` hole (full model description,
names included, stored in plaintext next to the ciphertext). Ingests a
document through the real pipeline with encryption ON, then scrolls the
temp collection asserting every point's shape:

  * no ``_enc_desc`` / ``_enc_date``-as-PII hints (only ``_enc_date`` +
    ``_enc_v`` markers, both non-sensitive),
  * ``text`` / ``description`` / ``metadata`` / ``original_filename``
    are ``enc:``-prefixed (or null),
  * a search round-trip still decrypts (read path intact).

Skips cleanly when Postgres/Qdrant are unreachable.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import urllib.request
from pathlib import Path

from document_mgmt_service.application.ingestion import (
    IngestionDependencies,
    IngestionService,
)
from document_mgmt_service.application.search import (
    SemanticChunker,
    SemanticSearchService,
    TextEmbeddingService,
)
from document_mgmt_service.domain.models import DocumentIngestionRequest
from document_mgmt_service.infrastructure.storage import LocalFileStorageAdapter

from tests._live_stack import (
    DEFAULT_QDRANT_URL,
    build_postgres_repo,
    build_qdrant_store,
)


class _FakeOCR:
    def ping(self): pass
    def extract_text(self, path): return "extracted text mentioning Ravi"
    def close(self): pass


def _scroll_all(base_url: str, collection: str) -> list[dict]:
    points: list[dict] = []
    offset = None
    while True:
        body: dict = {"limit": 100, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/collections/{collection}/points/scroll",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())["result"]
        points.extend(result.get("points", []))
        offset = result.get("next_page_offset")
        if not offset:
            return points


class PayloadHygieneTests(unittest.TestCase):
    def test_no_plaintext_hints_and_values_encrypted(self):
        repo = build_postgres_repo(self)
        storage = LocalFileStorageAdapter(Path(tempfile.mkdtemp()))
        store = build_qdrant_store(self, dimension=32, encryption_key="test-key")
        search = SemanticSearchService(
            store=store,
            embedder=TextEmbeddingService(dimension=32),
            chunker=SemanticChunker(max_chunk_chars=60, chunk_overlap=10),
        )
        service = IngestionService(dependencies=IngestionDependencies(
            repository=repo, storage=storage, ocr=_FakeOCR(),
            semantic_search=search,
        ))
        result = service.ingest(DocumentIngestionRequest(
            original_filename="aadhaar_card.png",
            content=b"%PDF-1.4 hygiene probe",
            content_type="application/pdf",
        ))
        self.assertGreater(result.chunk_count, 0)

        points = _scroll_all(DEFAULT_QDRANT_URL, store._collection_name)  # noqa: SLF001
        self.assertTrue(points, "expected indexed points")
        for point in points:
            payload = point["payload"]
            self.assertNotIn("_enc_desc", payload)
            self.assertEqual(payload.get("_enc_v"), 2)
            for field in ("text", "description", "metadata", "original_filename"):
                value = payload.get(field)
                self.assertTrue(
                    value is None or (isinstance(value, str) and value.startswith("enc:")),
                    f"{field} must be encrypted, got: {str(value)[:40]}",
                )

    def test_search_round_trip_still_decrypts(self):
        repo = build_postgres_repo(self)
        storage = LocalFileStorageAdapter(Path(tempfile.mkdtemp()))
        store = build_qdrant_store(self, dimension=32, encryption_key="test-key")
        search = SemanticSearchService(
            store=store,
            embedder=TextEmbeddingService(dimension=32),
            chunker=SemanticChunker(max_chunk_chars=60, chunk_overlap=10),
        )
        service = IngestionService(dependencies=IngestionDependencies(
            repository=repo, storage=storage, ocr=_FakeOCR(),
            semantic_search=search,
        ))
        result = service.ingest(DocumentIngestionRequest(
            original_filename="note.pdf",
            content=b"%PDF-1.4 roundtrip probe",
            content_type="application/pdf",
        ))
        matches = search.search_documents("extracted text", limit=5)
        self.assertTrue(matches, "expected decrypted search matches")


if __name__ == "__main__":
    unittest.main()
