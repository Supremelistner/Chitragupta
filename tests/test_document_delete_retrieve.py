"""Tests for document delete (cascade) and gated whole-document retrieve.

These cover the My-Documents-panel backend added in the UX refresh:

* ``DocumentAccessService.delete_document`` — whole-document vs single-version
  deletes across Postgres + file storage (+ Qdrant, best-effort), with
  ``latest_version`` recompute and per-user ownership scoping.
* ``DocumentAccessService.get_document`` approval gate — a sensitive document
  requires approval, and ``approval_granted=True`` lets the owner retrieve it
  (the loop the panel's "Open" button drives deterministically).

They use the live Postgres repo (per-test private schema) and a temp-dir
file store, so they exercise the real cascade, not mocks. Skips cleanly if
the docker stack is down.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_mgmt_service.application.access import (
    ApprovalRequiredError,
    DocumentAccessService,
)
from document_mgmt_service.application.search import (
    SemanticChunker,
    SemanticSearchService,
    TextEmbeddingService,
)
from document_mgmt_service.domain.models import (
    DocumentFileKind,
    DocumentPrivacyClassification,
    DocumentProcessingStatus,
    DocumentVersionRecord,
    SemanticIndexStatus,
)
from document_mgmt_service.infrastructure.storage import LocalFileStorageAdapter

from tests._live_stack import build_postgres_repo, build_qdrant_store


def _make_record(
    document_id: str,
    version: int,
    *,
    user_id: str = "__local__",
    privacy: DocumentPrivacyClassification = DocumentPrivacyClassification.SENSITIVE,
    storage_key: str | None = None,
) -> DocumentVersionRecord:
    return DocumentVersionRecord(
        document_id=document_id,
        version=version,
        original_filename="aadhaar.png",
        content_type="image/png",
        file_kind=DocumentFileKind.IMAGE,
        storage_key=storage_key or f"documents/{document_id}/v{version}/aadhaar.png",
        file_size_bytes=8,
        sha256="abc",
        privacy=privacy,
        processing_status=DocumentProcessingStatus.INDEXED,
        semantic_index_status=SemanticIndexStatus.INDEXED,
        owner_type="SELF",
        summary="Aadhaar card",
        description_safe="Aadhaar card",
        extracted_fields={"aadhaar_number": "9999 8888 7777"},
        user_id=user_id,
    )


class DocumentDeleteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = build_postgres_repo(self)
        self.qdrant = build_qdrant_store(self)
        self._tmp = tempfile.mkdtemp()
        self.storage = LocalFileStorageAdapter(Path(self._tmp))
        search = SemanticSearchService(
            store=self.qdrant,
            embedder=TextEmbeddingService(dimension=256),
            chunker=SemanticChunker(),
        )
        self.access = DocumentAccessService(
            repository=self.repo, storage=self.storage, search=search,
        )

    def _write_blob(self, storage_key: str) -> Path:
        p = Path(self._tmp) / storage_key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"filebytes")
        return p

    def test_delete_whole_document_removes_all_versions_and_files(self) -> None:
        self.repo.upsert_version(_make_record("doc-A", 1))
        self.repo.upsert_version(_make_record("doc-A", 2))
        p1 = self._write_blob("documents/doc-A/v1/aadhaar.png")
        p2 = self._write_blob("documents/doc-A/v2/aadhaar.png")

        result = self.access.delete_document("doc-A", user_id="__local__")

        self.assertTrue(result["deleted"])
        self.assertEqual(result["scope"], "document")
        self.assertEqual(result["versions_removed"], 2)
        self.assertEqual(self.repo.list_versions("doc-A"), [])
        self.assertFalse(p1.exists())
        self.assertFalse(p2.exists())

    def test_delete_single_version_keeps_others_and_recomputes_latest(self) -> None:
        self.repo.upsert_version(_make_record("doc-B", 1))
        self.repo.upsert_version(_make_record("doc-B", 2))
        self._write_blob("documents/doc-B/v1/aadhaar.png")
        self._write_blob("documents/doc-B/v2/aadhaar.png")

        result = self.access.delete_document("doc-B", version=2, user_id="__local__")

        self.assertTrue(result["deleted"])
        self.assertEqual(result["scope"], "version")
        remaining = self.repo.list_versions("doc-B")
        self.assertEqual([r.version for r in remaining], [1])
        # Parent latest_version recomputed to the surviving version.
        summaries = {s.document_id: s for s in self.repo.list_documents()}
        self.assertEqual(summaries["doc-B"].latest_version, 1)

    def test_delete_last_remaining_version_removes_parent(self) -> None:
        self.repo.upsert_version(_make_record("doc-C", 1))
        self._write_blob("documents/doc-C/v1/aadhaar.png")

        self.access.delete_document("doc-C", version=1, user_id="__local__")

        self.assertNotIn("doc-C", {s.document_id for s in self.repo.list_documents()})

    def test_delete_scoped_to_owner_cannot_touch_another_users_doc(self) -> None:
        self.repo.upsert_version(_make_record("doc-D", 1, user_id="alice"))
        self._write_blob("documents/doc-D/v1/aadhaar.png")

        # Bob tries to delete Alice's document -> KeyError (404 at transport).
        with self.assertRaises(KeyError):
            self.access.delete_document("doc-D", user_id="bob")
        # Alice's document is untouched.
        self.assertEqual([r.version for r in self.repo.list_versions("doc-D")], [1])

    def test_delete_missing_document_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            self.access.delete_document("nope", user_id="__local__")


class DocumentRetrieveGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = build_postgres_repo(self)
        self.qdrant = build_qdrant_store(self)
        self._tmp = tempfile.mkdtemp()
        self.storage = LocalFileStorageAdapter(Path(self._tmp))
        search = SemanticSearchService(
            store=self.qdrant,
            embedder=TextEmbeddingService(dimension=256),
            chunker=SemanticChunker(),
        )
        self.access = DocumentAccessService(
            repository=self.repo, storage=self.storage, search=search,
        )

    def _write_blob(self, storage_key: str) -> None:
        p = Path(self._tmp) / storage_key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"originalfilebytes")

    def test_sensitive_document_requires_approval(self) -> None:
        self.repo.upsert_version(_make_record("doc-S", 1))
        self._write_blob("documents/doc-S/v1/aadhaar.png")
        with self.assertRaises(ApprovalRequiredError):
            self.access.get_document("doc-S", 1, user_id="__local__")

    def test_sensitive_document_retrievable_after_approval(self) -> None:
        self.repo.upsert_version(_make_record("doc-S2", 1))
        self._write_blob("documents/doc-S2/v1/aadhaar.png")
        resp = self.access.get_document(
            "doc-S2", 1, user_id="__local__", approval_granted=True,
        )
        self.assertEqual(resp.payload["access_action"], "ALLOW")
        self.assertEqual(resp.payload["filename"], "aadhaar.png")
        self.assertTrue(resp.payload["content_base64"])

    def test_open_document_retrievable_without_approval(self) -> None:
        self.repo.upsert_version(
            _make_record("doc-O", 1, privacy=DocumentPrivacyClassification.OPEN)
        )
        self._write_blob("documents/doc-O/v1/aadhaar.png")
        resp = self.access.get_document("doc-O", 1, user_id="__local__")
        self.assertEqual(resp.payload["access_action"], "ALLOW")
        self.assertTrue(resp.payload["content_base64"])


if __name__ == "__main__":
    unittest.main()
